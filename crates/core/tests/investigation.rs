use std::collections::HashMap;

use bayesite_core::investigation::identity::{artifact_digest, snapshot_digest};
use bayesite_core::investigation::manifest::{
    ArtifactKind, ArtifactRef, Decision, DecisionKind, EngineIdentity, EvidenceSelection,
    EvidenceStatus, Execution, Manifest, Operation, Outcome, Recipe,
};
use bayesite_core::investigation::verify_bundle;
use bayesite_core::json::{self, Value};

fn reference(bytes: &[u8], kind: ArtifactKind, format: &str) -> ArtifactRef {
    ArtifactRef {
        sha256: artifact_digest(bytes),
        bytes: bytes.len(),
        kind,
        format: format.into(),
    }
}

fn fixture_bundle() -> (Vec<u8>, HashMap<String, Vec<u8>>, Manifest) {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let model =
        std::fs::read(format!("{root}/examples/investigation-counts/poisson.json")).unwrap();
    let data = std::fs::read(format!("{root}/examples/investigation-counts/data.json")).unwrap();
    let inspection = br#"{"inspection_format":"v0-provisional"}"#.to_vec();
    let engine = b"test engine bytes".to_vec();
    let capabilities = br#"{"capabilities_format":"v0-provisional"}"#.to_vec();
    let model_ref = reference(&model, ArtifactKind::ModelIr, "bayeswire-ir-v1");
    let data_ref = reference(&data, ArtifactKind::Data, "bayesite-data-json-v1");
    let inspection_ref = reference(
        &inspection,
        ArtifactKind::Inspection,
        "inspection-v0-provisional",
    );
    let engine_ref = reference(&engine, ArtifactKind::EngineBinary, "native-executable");
    let capabilities_ref = reference(
        &capabilities,
        ArtifactKind::EngineCapabilities,
        "capabilities-v0-provisional",
    );
    let recipe = Recipe::new(
        "inspect-initial".into(),
        Operation::Inspect,
        model_ref.clone(),
        data_ref.clone(),
        None,
        EngineIdentity {
            binary: engine_ref.clone(),
            capabilities: capabilities_ref.clone(),
            target: "test-target".into(),
            profile: "release".into(),
        },
        Value::Object(vec![]),
    )
    .unwrap();
    let execution = Execution {
        id: "execution-1".into(),
        recipe: recipe.id.clone(),
        recipe_sha256: recipe.sha256.clone(),
        outcome: Outcome::Completed,
        output: Some(inspection_ref.clone()),
        error: None,
    };
    let manifest = Manifest {
        question: "What is the expected daily count?".into(),
        estimand_description: "Expected count for one exchangeable day".into(),
        estimand_parameter: "mean_daily_count".into(),
        source: None,
        model: model_ref.clone(),
        data: data_ref.clone(),
        decisions: vec![Decision {
            id: "initial-likelihood".into(),
            parent: None,
            reason: "Start with a Poisson baseline.".into(),
            cites: vec![model_ref.sha256.clone()],
            kind: DecisionKind::Note,
        }],
        recipes: vec![recipe],
        executions: vec![execution],
        evidence: vec![EvidenceSelection {
            name: "effective-model".into(),
            execution: "execution-1".into(),
            status: EvidenceStatus::Current,
        }],
        interpretation: "The baseline is intentionally limited.".into(),
        unresolved_questions: vec!["Does a dispersion model reproduce the tail?".into()],
    };
    let manifest_bytes = manifest.to_bytes().unwrap();
    let mut objects = HashMap::new();
    for bytes in [model, data, inspection, engine, capabilities] {
        objects.insert(artifact_digest(&bytes).0, bytes);
    }
    (manifest_bytes, objects, manifest)
}

#[test]
fn bundle_verification_hashes_received_bytes_and_checks_closure() {
    let (manifest_bytes, objects, _) = fixture_bundle();
    let (_, report) = verify_bundle(&manifest_bytes, |reference| {
        objects
            .get(reference.sha256.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap();
    assert!(report.schema_valid);
    assert!(report.reference_closure_valid);
    assert!(report.object_integrity_valid);
    assert!(report.current_results_valid);
    assert!(report.engine_artifacts_available);
    assert!(!report.replay_recorded);
    assert_eq!(report.snapshot_id, snapshot_digest(&manifest_bytes));

    let mut differently_encoded = manifest_bytes.clone();
    differently_encoded.push(b'\n');
    assert_ne!(
        snapshot_digest(&manifest_bytes),
        snapshot_digest(&differently_encoded)
    );
}

#[test]
fn tampered_and_missing_objects_fail_without_execution() {
    let (manifest_bytes, objects, _) = fixture_bundle();
    let mut tampered = objects.clone();
    let first = tampered.values_mut().next().unwrap();
    first.push(b'!');
    let error = verify_bundle(&manifest_bytes, |reference| {
        tampered
            .get(reference.sha256.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap_err();
    assert!(error.message.contains("bytes") || error.message.contains("integrity"));

    let error = verify_bundle(&manifest_bytes, |_reference| {
        Err(bayesite_core::Error::new(
            bayesite_core::error::ErrorKind::MalformedDocument,
            "missing object",
        ))
    })
    .unwrap_err();
    assert_eq!(error.message, "missing object");
}

#[test]
fn recipe_identity_frames_model_and_data_separately() {
    let engine_bytes = b"engine";
    let capabilities = br#"{"capabilities_format":"v0-provisional"}"#;
    let engine = EngineIdentity {
        binary: reference(
            engine_bytes,
            ArtifactKind::EngineBinary,
            "native-executable",
        ),
        capabilities: reference(
            capabilities,
            ArtifactKind::EngineCapabilities,
            "capabilities-v0-provisional",
        ),
        target: "test-target".into(),
        profile: "release".into(),
    };
    let recipe = |id: &str, model: &[u8], data: &[u8]| {
        Recipe::new(
            id.into(),
            Operation::Inspect,
            reference(model, ArtifactKind::ModelIr, "bayeswire-ir-v1"),
            reference(data, ArtifactKind::Data, "bayesite-data-json-v1"),
            None,
            engine.clone(),
            Value::Object(vec![]),
        )
        .unwrap()
    };
    let left = recipe("left", b"a", b"bc");
    let right = recipe("right", b"ab", b"c");
    assert_ne!(left.sha256, right.sha256);
}

#[test]
fn stale_or_conflicting_current_evidence_is_rejected() {
    let (_, _, manifest) = fixture_bundle();
    let mut value = manifest.to_value();
    let evidence = match value.get("evidence").unwrap() {
        Value::Array(entries) => entries.clone(),
        _ => unreachable!(),
    };
    let mut doubled = evidence.clone();
    let mut duplicate = evidence[0].clone();
    if let Value::Object(entries) = &mut duplicate {
        entries
            .iter_mut()
            .find(|(name, _)| name == "name")
            .unwrap()
            .1 = Value::Str("also-current".into());
    }
    doubled.push(duplicate);
    if let Value::Object(entries) = &mut value {
        entries
            .iter_mut()
            .find(|(name, _)| name == "evidence")
            .unwrap()
            .1 = Value::Array(doubled);
    }
    let error = Manifest::parse(&value).unwrap_err();
    assert!(error.message.contains("conflicting current evidence"));
}

#[test]
fn fixed_identity_vector_is_stable() {
    let (bytes, _, manifest) = fixture_bundle();
    let recipe = &manifest.recipes[0];
    assert_eq!(
        recipe.sha256.as_str(),
        "09f59f4bf0eea62d7bf0d2156964731bedf13f7d711b4114941ff087e1adc05c"
    );
    assert_eq!(
        snapshot_digest(&bytes).as_str(),
        "11e4d7ee097d44f18c46333024275276878ec58b464665016fb7aaa0d750b39a"
    );
    assert_eq!(
        json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap(),
        manifest.to_value()
    );
}
