//! Pure investigation snapshot identities, records, and offline verification.
//!
//! Filesystem storage and process execution deliberately live in the CLI host.

pub mod identity;
pub mod manifest;

use std::collections::HashSet;

use crate::error::{Error, ErrorKind};
use crate::ir::decode_model;
use crate::json::{self, Value};
use crate::model::data_from_json;

use identity::{artifact_digest, snapshot_digest, Digest};
use manifest::{ArtifactKind, ArtifactRef, Manifest, MAX_ANCESTRY, MAX_OBJECTS};

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    pub snapshot_id: Digest,
    pub schema_valid: bool,
    pub reference_closure_valid: bool,
    pub object_integrity_valid: bool,
    pub current_results_valid: bool,
    pub engine_artifacts_available: bool,
    pub replay_recorded: bool,
    pub object_count: usize,
    pub ancestry_depth: usize,
}

impl Verification {
    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            (
                "verification_format".into(),
                Value::Str("v0-provisional".into()),
            ),
            (
                "snapshot_id".into(),
                Value::Str(self.snapshot_id.prefixed()),
            ),
            ("schema_valid".into(), Value::Bool(self.schema_valid)),
            (
                "reference_closure_valid".into(),
                Value::Bool(self.reference_closure_valid),
            ),
            (
                "object_integrity_valid".into(),
                Value::Bool(self.object_integrity_valid),
            ),
            (
                "current_results_valid".into(),
                Value::Bool(self.current_results_valid),
            ),
            (
                "engine_artifacts_available".into(),
                Value::Bool(self.engine_artifacts_available),
            ),
            ("replay_recorded".into(), Value::Bool(self.replay_recorded)),
            ("object_count".into(), Value::Int(self.object_count as i64)),
            (
                "ancestry_depth".into(),
                Value::Int(self.ancestry_depth as i64),
            ),
            ("verification_executes_recipes".into(), Value::Bool(false)),
            ("authenticity_verified".into(), Value::Bool(false)),
        ])
    }
}

fn validate_artifact(reference: &ArtifactRef, bytes: &[u8]) -> Result<(), Error> {
    match reference.kind {
        ArtifactKind::ModelIr => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| malformed("model_ir artifact must be UTF-8 JSON"))?;
            decode_model(&json::parse(text)?)?;
        }
        ArtifactKind::Data => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| malformed("data artifact must be UTF-8 JSON"))?;
            data_from_json(&json::parse(text)?)?;
        }
        ArtifactKind::Inspection => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| malformed("inspection artifact must be UTF-8 JSON"))?;
            let value = json::parse(text)?;
            if value.get("inspection_format").and_then(Value::as_str) != Some("v0-provisional") {
                return Err(malformed(
                    "inspection artifact needs inspection_format \"v0-provisional\"",
                ));
            }
        }
        ArtifactKind::PosteriorDraws => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| malformed("posterior_draws artifact must be UTF-8 NDJSON"))?;
            crate::protocol::diagnose_ndjson(text)?;
        }
        ArtifactKind::Diagnostics => {
            marker(bytes, "diagnostics_format")?;
        }
        ArtifactKind::PosteriorCheck => {
            marker(bytes, "posterior_check_format")?;
        }
        ArtifactKind::EngineCapabilities => {
            marker(bytes, "capabilities_format")?;
        }
        ArtifactKind::ReplayReport => {
            marker(bytes, "replay_format")?;
        }
        ArtifactKind::EngineBinary | ArtifactKind::InvestigationManifest => {}
    }
    Ok(())
}

fn marker(bytes: &[u8], field: &str) -> Result<(), Error> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| malformed(format!("artifact carrying {field} must be UTF-8 JSON")))?;
    let value = json::parse(text)?;
    if value.get(field).and_then(Value::as_str) != Some("v0-provisional") {
        return Err(malformed(format!(
            "artifact needs {field} \"v0-provisional\""
        )));
    }
    Ok(())
}

struct VerifyState {
    objects: HashSet<String>,
    snapshots: HashSet<String>,
    replay_recorded: bool,
    engine_available: bool,
    max_depth: usize,
}

fn verify_recursive(
    manifest_bytes: &[u8],
    depth: usize,
    loader: &mut impl FnMut(&ArtifactRef) -> Result<Vec<u8>, Error>,
    state: &mut VerifyState,
) -> Result<Manifest, Error> {
    if depth > MAX_ANCESTRY {
        return Err(malformed(format!(
            "snapshot ancestry exceeds the {MAX_ANCESTRY}-manifest limit"
        )));
    }
    state.max_depth = state.max_depth.max(depth);
    let snapshot = snapshot_digest(manifest_bytes);
    if !state.snapshots.insert(snapshot.0.clone()) {
        return Err(malformed(format!(
            "snapshot ancestry cycle detected at {}",
            snapshot.prefixed()
        )));
    }
    let manifest = Manifest::parse_bytes(manifest_bytes)?;
    let mut loaded_parent: Option<Vec<u8>> = None;
    for reference in manifest.direct_references() {
        let key = reference.sha256.as_str().to_string();
        let bytes = loader(reference)?;
        if bytes.len() != reference.bytes {
            return Err(malformed(format!(
                "object {} has {} bytes but its reference requires {}",
                reference.sha256.prefixed(),
                bytes.len(),
                reference.bytes
            )));
        }
        let actual = artifact_digest(&bytes);
        if actual != reference.sha256 {
            return Err(malformed(format!(
                "object integrity failure for {}; exact bytes hash to {}",
                reference.sha256.prefixed(),
                actual.prefixed()
            )));
        }
        if state.objects.insert(key) && state.objects.len() > MAX_OBJECTS {
            return Err(malformed(format!(
                "bundle exceeds the {MAX_OBJECTS}-object limit"
            )));
        }
        validate_artifact(reference, &bytes)?;
        state.replay_recorded |= reference.kind == ArtifactKind::ReplayReport;
        state.engine_available |= reference.kind == ArtifactKind::EngineBinary;
        if manifest
            .source
            .as_ref()
            .is_some_and(|source| source.manifest == *reference)
        {
            loaded_parent = Some(bytes);
        }
    }
    if let Some(source) = &manifest.source {
        let parent_bytes = loaded_parent.ok_or_else(|| {
            malformed("source manifest was not available in the verified object closure")
        })?;
        let parent_snapshot = snapshot_digest(&parent_bytes);
        if parent_snapshot != source.snapshot_id {
            return Err(malformed(format!(
                "source snapshot identity mismatch: manifest computes to {}, expected {}",
                parent_snapshot.prefixed(),
                source.snapshot_id.prefixed()
            )));
        }
        let parent = verify_recursive(&parent_bytes, depth + 1, loader, state)?;
        if parent.decision(&source.decision).is_none() {
            return Err(malformed(format!(
                "source decision {:?} does not exist in parent snapshot {}",
                source.decision,
                source.snapshot_id.prefixed()
            )));
        }
    }
    Ok(manifest)
}

/// Verify exact manifest bytes and all local/ancestral objects without running
/// any recipe or following a network link.
pub fn verify_bundle(
    manifest_bytes: &[u8],
    mut loader: impl FnMut(&ArtifactRef) -> Result<Vec<u8>, Error>,
) -> Result<(Manifest, Verification), Error> {
    let mut state = VerifyState {
        objects: HashSet::new(),
        snapshots: HashSet::new(),
        replay_recorded: false,
        engine_available: false,
        max_depth: 0,
    };
    let manifest = verify_recursive(manifest_bytes, 0, &mut loader, &mut state)?;
    let verification = Verification {
        snapshot_id: snapshot_digest(manifest_bytes),
        schema_valid: true,
        reference_closure_valid: true,
        object_integrity_valid: true,
        current_results_valid: true,
        engine_artifacts_available: state.engine_available,
        replay_recorded: state.replay_recorded,
        object_count: state.objects.len(),
        ancestry_depth: state.max_depth,
    };
    Ok((manifest, verification))
}
