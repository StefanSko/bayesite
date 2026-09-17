//! Pure investigation snapshot identities, records, and offline verification.
//!
//! Filesystem storage and process execution deliberately live in the CLI host.

pub mod identity;
pub mod manifest;

use std::collections::{HashMap, HashSet};

use crate::error::{Error, ErrorKind};
use crate::fingerprint::model_data_fingerprint;
use crate::ir::decode_model;
use crate::json::{self, Value};
use crate::model::{data_from_json, Posterior};

use identity::{artifact_digest, snapshot_digest, Digest};
use manifest::{
    ArtifactKind, ArtifactRef, Manifest, Operation, Outcome, MAX_ANCESTRY, MAX_OBJECTS,
};

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
    /// Exact verified closure, excluding the separately supplied root
    /// manifest. Host orchestration uses this list; reports do not serialize it.
    pub object_digests: Vec<Digest>,
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
            ("verification_complete".into(), Value::Bool(true)),
            ("findings".into(), Value::Array(vec![])),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationDimension {
    FormatSchema,
    ReferenceClosure,
    ObjectIntegrity,
    CurrentResults,
}

impl VerificationDimension {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FormatSchema => "format_schema",
            Self::ReferenceClosure => "reference_closure",
            Self::ObjectIntegrity => "object_integrity",
            Self::CurrentResults => "current_results",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerificationFailure {
    pub dimension: VerificationDimension,
    pub error: Error,
}

impl VerificationFailure {
    pub fn new(dimension: VerificationDimension, error: Error) -> Self {
        Self { dimension, error }
    }

    pub fn to_value(&self, manifest_bytes: Option<&[u8]>) -> Value {
        // Recursive schema, closure, integrity, and result checks are
        // interleaved to stay bounded. A failure proves its own dimension
        // false, but cannot prove any other recursive dimension complete. The
        // root manifest schema is the sole completed fact after a later-stage
        // failure.
        let status = |dimension| {
            if self.dimension == dimension {
                Value::Bool(false)
            } else if dimension == VerificationDimension::FormatSchema && manifest_bytes.is_some() {
                Value::Bool(true)
            } else {
                Value::Null
            }
        };
        Value::Object(vec![
            (
                "verification_format".into(),
                Value::Str("v0-provisional".into()),
            ),
            (
                "snapshot_id".into(),
                manifest_bytes.map_or(Value::Null, |bytes| {
                    Value::Str(snapshot_digest(bytes).prefixed())
                }),
            ),
            ("verification_complete".into(), Value::Bool(false)),
            (
                "schema_valid".into(),
                status(VerificationDimension::FormatSchema),
            ),
            (
                "reference_closure_valid".into(),
                status(VerificationDimension::ReferenceClosure),
            ),
            (
                "object_integrity_valid".into(),
                status(VerificationDimension::ObjectIntegrity),
            ),
            (
                "current_results_valid".into(),
                status(VerificationDimension::CurrentResults),
            ),
            ("engine_artifacts_available".into(), Value::Null),
            ("replay_recorded".into(), Value::Null),
            ("object_count".into(), Value::Null),
            ("ancestry_depth".into(), Value::Null),
            ("verification_executes_recipes".into(), Value::Bool(false)),
            ("authenticity_verified".into(), Value::Bool(false)),
            (
                "findings".into(),
                Value::Array(vec![Value::Object(vec![
                    (
                        "dimension".into(),
                        Value::Str(self.dimension.as_str().into()),
                    ),
                    ("error".into(), Value::Str(self.error.kind.name().into())),
                    ("message".into(), Value::Str(self.error.message.clone())),
                ])]),
            ),
        ])
    }
}

fn checked<T>(
    dimension: VerificationDimension,
    result: Result<T, Error>,
) -> Result<T, VerificationFailure> {
    result.map_err(|error| VerificationFailure::new(dimension, error))
}

fn verification_error(
    dimension: VerificationDimension,
    message: impl Into<String>,
) -> VerificationFailure {
    VerificationFailure::new(dimension, malformed(message))
}

fn reject_duplicate_json_fields(value: &Value, context: &str) -> Result<(), Error> {
    match value {
        Value::Object(entries) => {
            let mut names = HashSet::new();
            for (name, child) in entries {
                if !names.insert(name.as_str()) {
                    return Err(malformed(format!(
                        "{context} has duplicate field {name:?}; remove one"
                    )));
                }
                reject_duplicate_json_fields(child, context)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                reject_duplicate_json_fields(child, context)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn parse_json_artifact(bytes: &[u8], context: &str) -> Result<Value, Error> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| malformed(format!("{context} must be UTF-8 JSON")))?;
    let value = json::parse(text)?;
    reject_duplicate_json_fields(&value, context)?;
    if !matches!(value, Value::Object(_)) {
        return Err(malformed(format!("{context} must be a JSON object")));
    }
    Ok(value)
}

fn required_array(value: &Value, name: &str, context: &str) -> Result<(), Error> {
    if value.get(name).and_then(Value::as_array).is_none() {
        Err(malformed(format!(
            "{context} needs an array field {name:?}"
        )))
    } else {
        Ok(())
    }
}

fn validate_inspection(value: &Value) -> Result<(), Error> {
    let context = "inspection artifact";
    for name in [
        "free_slots",
        "density_factors",
        "data",
        "structural_discrepancies",
    ] {
        required_array(value, name, context)?;
    }
    if value
        .get("unconstrained_parameter_count")
        .and_then(Value::as_i64)
        .is_none()
        || !matches!(value.get("execution_metadata"), Some(Value::Object(_)))
        || !matches!(value.get("declarations"), Some(Value::Object(_)))
        || !matches!(value.get("density_accounting"), Some(Value::Object(_)))
    {
        return Err(malformed(
            "inspection artifact needs execution_metadata, unconstrained_parameter_count, declarations, and density_accounting",
        ));
    }
    for (index, slot) in value
        .get("free_slots")
        .and_then(Value::as_array)
        .expect("checked")
        .iter()
        .enumerate()
    {
        if slot.get("name").and_then(Value::as_str).is_none()
            || slot.get("shape").and_then(Value::as_array).is_none()
            || slot.get("offset").and_then(Value::as_i64).is_none()
            || slot.get("length").and_then(Value::as_i64).is_none()
            || !matches!(slot.get("resolved_constraint"), Some(Value::Object(_)))
        {
            return Err(malformed(format!(
                "inspection artifact free_slots[{index}] is missing typed layout fields"
            )));
        }
    }
    for (index, factor) in value
        .get("density_factors")
        .and_then(Value::as_array)
        .expect("checked")
        .iter()
        .enumerate()
    {
        if factor.get("name").and_then(Value::as_str).is_none()
            || !matches!(factor.get("distribution"), Some(Value::Object(_)))
            || !matches!(factor.get("value_expression"), Some(Value::Object(_)))
        {
            return Err(malformed(format!(
                "inspection artifact density_factors[{index}] is missing name/distribution/value_expression"
            )));
        }
    }
    Ok(())
}

fn validate_artifact(reference: &ArtifactRef, bytes: &[u8]) -> Result<(), Error> {
    match reference.kind {
        ArtifactKind::ModelIr => {
            let value = parse_json_artifact(bytes, "model_ir artifact")?;
            decode_model(&value)?;
        }
        ArtifactKind::Data => {
            let value = parse_json_artifact(bytes, "data artifact")?;
            data_from_json(&value)?;
        }
        ArtifactKind::Inspection => {
            let value = parse_json_artifact(bytes, "inspection artifact")?;
            if value.get("inspection_format").and_then(Value::as_str) != Some("v0-provisional") {
                return Err(malformed(
                    "inspection artifact needs inspection_format \"v0-provisional\"",
                ));
            }
            validate_inspection(&value)?;
        }
        ArtifactKind::PosteriorDraws => {
            let text = std::str::from_utf8(bytes)
                .map_err(|_| malformed("posterior_draws artifact must be UTF-8 NDJSON"))?;
            for (index, line) in text.lines().enumerate() {
                let value = json::parse(line)?;
                reject_duplicate_json_fields(&value, &format!("posterior_draws line {index}"))?;
            }
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
    let value = parse_json_artifact(bytes, &format!("artifact carrying {field}"))?;
    if value.get(field).and_then(Value::as_str) != Some("v0-provisional") {
        return Err(malformed(format!(
            "artifact needs {field} \"v0-provisional\""
        )));
    }
    Ok(())
}

fn validate_sample_execution_inputs(manifest: &Manifest, state: &VerifyState) -> Result<(), Error> {
    for execution in &manifest.executions {
        if execution.outcome != Outcome::Completed {
            continue;
        }
        let recipe = manifest
            .recipes
            .iter()
            .find(|recipe| recipe.id == execution.recipe)
            .expect("manifest relationships validated recipe references");
        if recipe.operation != Operation::Sample {
            continue;
        }
        let output = execution
            .output
            .as_ref()
            .expect("completed execution has output");
        let model = state
            .object_bytes
            .get(recipe.model.sha256.as_str())
            .expect("direct model reference was loaded");
        let data = state
            .object_bytes
            .get(recipe.data.sha256.as_str())
            .expect("direct data reference was loaded");
        let fit = state
            .object_bytes
            .get(output.sha256.as_str())
            .expect("direct output reference was loaded");
        let model_text = std::str::from_utf8(model)
            .map_err(|_| malformed("sample recipe model must be UTF-8"))?;
        let data_text =
            std::str::from_utf8(data).map_err(|_| malformed("sample recipe data must be UTF-8"))?;
        let expected = model_data_fingerprint(model_text, data_text);
        let header_text = std::str::from_utf8(fit)
            .map_err(|_| malformed("sample output must be UTF-8 NDJSON"))?
            .lines()
            .next()
            .ok_or_else(|| malformed("sample output is empty"))?;
        let header = json::parse(header_text)?;
        let recorded = header
            .get("model_data_fingerprint")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                malformed(
                    "investigation sample output must carry a model/data fingerprint from exact recipe inputs",
                )
            })?;
        if recorded != expected {
            return Err(malformed(format!(
                "sample execution {:?} model/data fingerprint does not match its exact recipe model/data; mark the fit historical and rerun",
                execution.id
            )));
        }

        let recorded_settings = header
            .get("settings")
            .ok_or_else(|| malformed("sample output header needs settings"))?;
        let integer_pairs = [
            ("seed", recipe.settings.get("seed"), header.get("seed")),
            (
                "chains",
                recipe.settings.get("chains"),
                header.get("chain_count"),
            ),
            (
                "warmup",
                recipe.settings.get("warmup"),
                recorded_settings.get("num_warmup"),
            ),
            (
                "draws",
                recipe.settings.get("draws"),
                recorded_settings.get("num_draws"),
            ),
            (
                "max_treedepth",
                recipe.settings.get("max_treedepth"),
                recorded_settings.get("max_treedepth"),
            ),
        ];
        let integer_mismatch = integer_pairs.iter().find(|(_, expected, recorded)| {
            expected.and_then(Value::as_i64) != recorded.and_then(Value::as_i64)
        });
        let float_pairs = [
            (
                "target_accept",
                recipe.settings.get("target_accept"),
                recorded_settings.get("target_accept"),
            ),
            (
                "initial_step_size",
                recipe.settings.get("initial_step_size"),
                recorded_settings.get("initial_step_size"),
            ),
        ];
        let float_mismatch = float_pairs.iter().find(|(_, expected, recorded)| {
            expected.and_then(Value::as_f64) != recorded.and_then(Value::as_f64)
        });
        if let Some((name, _, _)) = integer_mismatch.or(float_mismatch) {
            return Err(malformed(format!(
                "sample execution {:?} output contradicts recipe setting {name:?}; mark the fit historical and rerun the exact recipe",
                execution.id
            )));
        }

        // The legacy combined fingerprint is only a compatibility check, not
        // snapshot identity. Independently compare the fit's parameter layout
        // to the posterior built from the separately hashed model and data.
        let meta = decode_model(&json::parse(model_text)?)?;
        let bound_data = data_from_json(&json::parse(data_text)?)?;
        let expected_packing = Posterior::new(meta, bound_data)?.packing();
        let params = header
            .get("params")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("sample output header needs params for compatibility"))?;
        if params.len() != expected_packing.len() {
            return Err(malformed(format!(
                "sample execution {:?} parameter layout does not match its recipe model/data",
                execution.id
            )));
        }
        for (param, (expected_name, expected_shape)) in params.iter().zip(&expected_packing) {
            let name = param.get("name").and_then(Value::as_str);
            let shape = param.get("shape").and_then(Value::as_array);
            let shape_matches = shape.is_some_and(|shape| {
                shape.len() == expected_shape.len()
                    && shape.iter().zip(expected_shape).all(|(got, expected)| {
                        got.as_i64()
                            .is_some_and(|got| got >= 0 && got as usize == *expected)
                    })
            });
            if name != Some(expected_name.as_str()) || !shape_matches {
                return Err(malformed(format!(
                    "sample execution {:?} parameter layout does not match its recipe model/data",
                    execution.id
                )));
            }
        }
    }
    Ok(())
}

struct VerifyState {
    objects: HashSet<String>,
    object_bytes: HashMap<String, Vec<u8>>,
    snapshots: HashSet<String>,
    replay_recorded: bool,
    engine_available: bool,
    max_depth: usize,
}

fn verify_recursive(
    manifest_bytes: &[u8],
    depth: usize,
    loader: &mut impl FnMut(&Digest) -> Result<Vec<u8>, Error>,
    state: &mut VerifyState,
) -> Result<Manifest, VerificationFailure> {
    if depth >= MAX_ANCESTRY {
        return Err(verification_error(
            VerificationDimension::ReferenceClosure,
            format!("snapshot ancestry exceeds the {MAX_ANCESTRY}-manifest limit"),
        ));
    }
    state.max_depth = state.max_depth.max(depth);
    let snapshot = snapshot_digest(manifest_bytes);
    if !state.snapshots.insert(snapshot.0.clone()) {
        return Err(verification_error(
            VerificationDimension::ReferenceClosure,
            format!(
                "snapshot ancestry cycle detected at {}",
                snapshot.prefixed()
            ),
        ));
    }
    let manifest = checked(
        VerificationDimension::FormatSchema,
        Manifest::parse_bytes(manifest_bytes),
    )?;
    let mut loaded_parent: Option<Vec<u8>> = None;
    for reference in manifest.direct_references() {
        let key = reference.sha256.as_str().to_string();
        let bytes = checked(
            VerificationDimension::ReferenceClosure,
            loader(&reference.sha256),
        )?;
        if bytes.len() != reference.bytes {
            return Err(verification_error(
                VerificationDimension::ObjectIntegrity,
                format!(
                    "object {} has {} bytes but its reference requires {}",
                    reference.sha256.prefixed(),
                    bytes.len(),
                    reference.bytes
                ),
            ));
        }
        let actual = artifact_digest(&bytes);
        if actual != reference.sha256 {
            return Err(verification_error(
                VerificationDimension::ObjectIntegrity,
                format!(
                    "object integrity failure for {}; exact bytes hash to {}",
                    reference.sha256.prefixed(),
                    actual.prefixed()
                ),
            ));
        }
        if state.objects.insert(key.clone()) && state.objects.len() > MAX_OBJECTS {
            return Err(verification_error(
                VerificationDimension::ReferenceClosure,
                format!("bundle exceeds the {MAX_OBJECTS}-object limit"),
            ));
        }
        state
            .object_bytes
            .entry(key)
            .or_insert_with(|| bytes.clone());
        checked(
            VerificationDimension::CurrentResults,
            validate_artifact(reference, &bytes),
        )?;
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
    checked(
        VerificationDimension::CurrentResults,
        validate_sample_execution_inputs(&manifest, state),
    )?;
    if let Some(source) = &manifest.source {
        let parent_bytes = loaded_parent.ok_or_else(|| {
            verification_error(
                VerificationDimension::ReferenceClosure,
                "source manifest was not available in the verified object closure",
            )
        })?;
        let parent_snapshot = snapshot_digest(&parent_bytes);
        if parent_snapshot != source.snapshot_id {
            return Err(verification_error(
                VerificationDimension::ReferenceClosure,
                format!(
                    "source snapshot identity mismatch: manifest computes to {}, expected {}",
                    parent_snapshot.prefixed(),
                    source.snapshot_id.prefixed()
                ),
            ));
        }
        let parent = verify_recursive(&parent_bytes, depth + 1, loader, state)?;
        if parent.decision(&source.decision).is_none() {
            return Err(verification_error(
                VerificationDimension::ReferenceClosure,
                format!(
                    "source decision {:?} does not exist in parent snapshot {}",
                    source.decision,
                    source.snapshot_id.prefixed()
                ),
            ));
        }
    }
    for decision in &manifest.decisions {
        for citation in &decision.cites {
            if state.objects.contains(citation.as_str()) {
                continue;
            }
            let bytes = loader(citation).map_err(|_| {
                verification_error(
                    VerificationDimension::ReferenceClosure,
                    format!(
                        "decision {:?} cites missing artifact {}; restore its exact bytes",
                        decision.id,
                        citation.prefixed()
                    ),
                )
            })?;
            if bytes.len() > manifest::MAX_OBJECT_BYTES {
                return Err(verification_error(
                    VerificationDimension::ObjectIntegrity,
                    format!(
                        "cited object {} exceeds the {}-byte limit",
                        citation.prefixed(),
                        manifest::MAX_OBJECT_BYTES
                    ),
                ));
            }
            let actual = artifact_digest(&bytes);
            if actual != *citation {
                return Err(verification_error(
                    VerificationDimension::ObjectIntegrity,
                    format!(
                        "decision {:?} citation integrity failure: expected {}, exact bytes hash to {}",
                        decision.id,
                        citation.prefixed(),
                        actual.prefixed()
                    ),
                ));
            }
            if state.objects.insert(citation.as_str().to_string())
                && state.objects.len() > MAX_OBJECTS
            {
                return Err(verification_error(
                    VerificationDimension::ReferenceClosure,
                    format!("bundle exceeds the {MAX_OBJECTS}-object limit"),
                ));
            }
            state
                .object_bytes
                .entry(citation.as_str().to_string())
                .or_insert(bytes);
        }
    }
    Ok(manifest)
}

/// Verify exact manifest bytes and all local/ancestral objects without running
/// any recipe or following a network link, retaining the failed dimension for
/// machine-readable CLI reports.
pub fn verify_bundle_detailed(
    manifest_bytes: &[u8],
    mut loader: impl FnMut(&Digest) -> Result<Vec<u8>, Error>,
) -> Result<(Manifest, Verification), VerificationFailure> {
    let mut state = VerifyState {
        objects: HashSet::new(),
        object_bytes: HashMap::new(),
        snapshots: HashSet::new(),
        replay_recorded: false,
        engine_available: false,
        max_depth: 0,
    };
    let manifest = verify_recursive(manifest_bytes, 0, &mut loader, &mut state)?;
    let mut object_digests = state
        .objects
        .iter()
        .map(|digest| Digest::parse(digest, "verified object digest"))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| VerificationFailure::new(VerificationDimension::ObjectIntegrity, error))?;
    object_digests.sort_by(|left, right| left.as_str().cmp(right.as_str()));
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
        object_digests,
    };
    Ok((manifest, verification))
}

pub fn verify_bundle(
    manifest_bytes: &[u8],
    loader: impl FnMut(&Digest) -> Result<Vec<u8>, Error>,
) -> Result<(Manifest, Verification), Error> {
    verify_bundle_detailed(manifest_bytes, loader).map_err(|failure| failure.error)
}
