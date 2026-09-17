use std::collections::HashMap;
use std::process::Command;

use bayesite_core::fingerprint::model_data_fingerprint;
use bayesite_core::inspect::inspect_json;
use bayesite_core::investigation::identity::{artifact_digest, snapshot_digest};
use bayesite_core::investigation::manifest::{
    ArtifactKind, ArtifactRef, Decision, DecisionInputs, DecisionKind, EngineIdentity,
    EvidenceSelection, EvidenceStatus, Execution, Manifest, Operation, Outcome, Recipe,
};
use bayesite_core::investigation::verify_bundle;
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::data_from_json;
use bayesite_core::predictive::{
    prior_predictive_ndjson_lines_from_full_data_with_model_data_fingerprint,
    PriorPredictiveSettings,
};

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
    let inspection = format!(
        "{}\n",
        inspect_json(
            decode_model(&json::parse(std::str::from_utf8(&model).unwrap()).unwrap()).unwrap(),
            data_from_json(&json::parse(std::str::from_utf8(&data).unwrap()).unwrap()).unwrap(),
        )
        .unwrap()
    )
    .into_bytes();
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
            inputs: DecisionInputs {
                model: model_ref.clone(),
                data: data_ref.clone(),
            },
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

fn prior_predictive_bundle() -> (Vec<u8>, HashMap<String, Vec<u8>>, Manifest) {
    let (_, mut objects, mut manifest) = fixture_bundle();
    let model = objects
        .get(manifest.model.sha256.as_str())
        .expect("model object")
        .clone();
    let data = objects
        .get(manifest.data.sha256.as_str())
        .expect("data object")
        .clone();
    let model_text = std::str::from_utf8(&model).unwrap();
    let data_text = std::str::from_utf8(&data).unwrap();
    let meta = decode_model(&json::parse(model_text).unwrap()).unwrap();
    let bound_data = data_from_json(&json::parse(data_text).unwrap()).unwrap();
    let fingerprint = model_data_fingerprint(model_text, data_text);
    let output = format!(
        "{}\n",
        prior_predictive_ndjson_lines_from_full_data_with_model_data_fingerprint(
            meta,
            bound_data,
            &PriorPredictiveSettings { num_draws: 2 },
            17,
            &fingerprint,
        )
        .unwrap()
        .join("\n")
    )
    .into_bytes();
    let output_ref = reference(
        &output,
        ArtifactKind::PriorPredictiveDraws,
        "prior-predictive-v0-provisional-ndjson",
    );
    let engine = manifest.recipes[0].engine.clone();
    let recipe = Recipe::new(
        "prior-predictive-initial".into(),
        Operation::PriorPredictive,
        manifest.model.clone(),
        manifest.data.clone(),
        None,
        engine,
        json::parse(r#"{"draws":2,"seed":17}"#).unwrap(),
    )
    .unwrap();
    manifest.recipes = vec![recipe.clone()];
    manifest.executions = vec![Execution {
        id: "execution-prior-predictive".into(),
        recipe: recipe.id.clone(),
        recipe_sha256: recipe.sha256.clone(),
        outcome: Outcome::Completed,
        output: Some(output_ref.clone()),
        error: None,
    }];
    manifest.evidence = vec![EvidenceSelection {
        name: "prior-predictive".into(),
        execution: "execution-prior-predictive".into(),
        status: EvidenceStatus::Current,
    }];
    objects.insert(output_ref.sha256.0.clone(), output);
    (manifest.to_bytes().unwrap(), objects, manifest)
}

#[test]
fn bundle_verification_hashes_received_bytes_and_checks_closure() {
    let (manifest_bytes, objects, _) = fixture_bundle();
    let (_, report) = verify_bundle(&manifest_bytes, |reference| {
        objects
            .get(reference.as_str())
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
            .get(reference.as_str())
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
fn rejects_self_parent_decision_lineage() {
    let (_, _, manifest) = fixture_bundle();
    let mut value = manifest.to_value();
    if let Value::Object(entries) = &mut value {
        let decisions = entries
            .iter_mut()
            .find(|(name, _)| name == "decisions")
            .map(|(_, value)| value)
            .unwrap();
        let first = match decisions {
            Value::Array(entries) => &mut entries[0],
            _ => unreachable!(),
        };
        if let Value::Object(entries) = first {
            entries
                .iter_mut()
                .find(|(name, _)| name == "parent")
                .unwrap()
                .1 = Value::Str("initial-likelihood".into());
        }
    }
    let error = Manifest::parse(&value).unwrap_err();
    assert!(error.message.contains("earlier local decision"));
}

#[test]
fn rejects_marker_only_and_duplicate_marker_inspection_artifacts() {
    for inspection in [
        br#"{"inspection_format":"v0-provisional"}"#.to_vec(),
        br#"{"inspection_format":"v0-provisional","inspection_format":"unsupported"}"#.to_vec(),
    ] {
        let (_, mut objects, mut manifest) = fixture_bundle();
        let inspection_ref = reference(
            &inspection,
            ArtifactKind::Inspection,
            "inspection-v0-provisional",
        );
        manifest.executions[0].output = Some(inspection_ref.clone());
        objects.insert(inspection_ref.sha256.0.clone(), inspection);
        let bytes = manifest.to_bytes().unwrap();
        let error = verify_bundle(&bytes, |reference| {
            objects
                .get(reference.as_str())
                .cloned()
                .ok_or_else(|| panic!("missing test object"))
        })
        .unwrap_err();
        assert!(
            error.message.contains("inspection") || error.message.contains("duplicate"),
            "{}",
            error.message
        );
    }
}

#[test]
fn verifies_prior_predictive_settings_and_exact_inputs() {
    let (bytes, objects, _) = prior_predictive_bundle();
    verify_bundle(&bytes, |reference| {
        objects
            .get(reference.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap();
}

#[test]
fn rejects_malformed_or_contradictory_prior_predictive_outputs() {
    let (_, objects, manifest) = prior_predictive_bundle();
    let valid_output = objects
        .get(
            manifest.executions[0]
                .output
                .as_ref()
                .unwrap()
                .sha256
                .as_str(),
        )
        .unwrap();
    let valid_text = String::from_utf8(valid_output.clone()).unwrap();
    let wrong_seed = valid_text
        .replacen("\"seed\":17", "\"seed\":18", 1)
        .into_bytes();
    let wrong_fingerprint = valid_text
        .replacen(
            "\"model_data_fingerprint\":\"sha256:",
            "\"model_data_fingerprint\":\"sha256:0",
            1,
        )
        .into_bytes();
    for output in [
        br#"{"prior_predictive_format":"v0-provisional"}"#.to_vec(),
        br#"{"prior_predictive_format":"v0-provisional","prior_predictive_format":"unsupported"}"#
            .to_vec(),
        wrong_seed,
        wrong_fingerprint,
    ] {
        let mut objects = objects.clone();
        let mut manifest = manifest.clone();
        let output_ref = reference(
            &output,
            ArtifactKind::PriorPredictiveDraws,
            "prior-predictive-v0-provisional-ndjson",
        );
        manifest.executions[0].output = Some(output_ref.clone());
        objects.insert(output_ref.sha256.0.clone(), output);
        let bytes = manifest.to_bytes().unwrap();
        let error = verify_bundle(&bytes, |reference| {
            objects
                .get(reference.as_str())
                .cloned()
                .ok_or_else(|| panic!("missing test object"))
        })
        .unwrap_err();
        assert!(
            error.message.contains("prior_predictive")
                || error.message.contains("duplicate")
                || error.message.contains("recipe setting \"seed\"")
                || error.message.contains("model/data fingerprint"),
            "{}",
            error.message
        );
    }
}

#[test]
fn rejects_current_fit_from_different_exact_model_bytes() {
    let root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
    let poisson_path = format!("{root}/examples/investigation-counts/poisson.json");
    let negative_path = format!("{root}/examples/investigation-counts/negative-binomial.json");
    let data_path = format!("{root}/examples/investigation-counts/data.json");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            &poisson_path,
            "--data",
            &data_path,
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--max-treedepth",
            "4",
            "--target-accept",
            "0.8",
            "--seed",
            "1",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let fit = output.stdout;
    let model = std::fs::read(negative_path).unwrap();
    let data = std::fs::read(data_path).unwrap();
    let engine = b"test engine".to_vec();
    let capabilities = br#"{"capabilities_format":"v0-provisional"}"#.to_vec();
    let model_ref = reference(&model, ArtifactKind::ModelIr, "bayeswire-ir-v1");
    let data_ref = reference(&data, ArtifactKind::Data, "bayesite-data-json-v1");
    let fit_ref = reference(
        &fit,
        ArtifactKind::PosteriorDraws,
        "draws-v0-provisional-ndjson",
    );
    let recipe = Recipe::new(
        "sample-initial".into(),
        Operation::Sample,
        model_ref.clone(),
        data_ref.clone(),
        None,
        EngineIdentity {
            binary: reference(&engine, ArtifactKind::EngineBinary, "native-executable"),
            capabilities: reference(
                &capabilities,
                ArtifactKind::EngineCapabilities,
                "capabilities-v0-provisional",
            ),
            target: "test-target".into(),
            profile: "release".into(),
        },
        json::parse(
            r#"{"chains":1,"warmup":10,"draws":4,"max_treedepth":4,"target_accept":0.8,"initial_step_size":1.0,"seed":1}"#,
        )
        .unwrap(),
    )
    .unwrap();
    let manifest = Manifest {
        question: "Question".into(),
        estimand_description: "Expected daily count".into(),
        estimand_parameter: "mean_daily_count".into(),
        source: None,
        model: model_ref.clone(),
        data: data_ref.clone(),
        decisions: vec![Decision {
            id: "initial-likelihood".into(),
            parent: None,
            reason: "Test stale evidence.".into(),
            cites: vec![model_ref.sha256.clone()],
            kind: DecisionKind::Note,
            inputs: DecisionInputs {
                model: model_ref.clone(),
                data: data_ref.clone(),
            },
        }],
        recipes: vec![recipe.clone()],
        executions: vec![Execution {
            id: "exec-1".into(),
            recipe: recipe.id.clone(),
            recipe_sha256: recipe.sha256.clone(),
            outcome: Outcome::Completed,
            output: Some(fit_ref.clone()),
            error: None,
        }],
        evidence: vec![EvidenceSelection {
            name: "sample-initial".into(),
            execution: "exec-1".into(),
            status: EvidenceStatus::Current,
        }],
        interpretation: "Test".into(),
        unresolved_questions: vec![],
    };
    let bytes = manifest.to_bytes().unwrap();
    let mut objects = HashMap::new();
    for object in [model, data, fit, engine, capabilities] {
        objects.insert(artifact_digest(&object).0, object);
    }
    let error = verify_bundle(&bytes, |reference| {
        objects
            .get(reference.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap_err();
    assert!(error.message.contains("model/data fingerprint"));
}

#[test]
fn ancestry_limit_counts_manifests_not_zero_based_edges() {
    let (mut bytes, mut objects, base) = fixture_bundle();
    let mut current = base;
    for _ in 1..16 {
        let parent_ref = reference(
            &bytes,
            ArtifactKind::InvestigationManifest,
            "investigation-snapshot-v0-provisional",
        );
        objects.insert(parent_ref.sha256.0.clone(), bytes.clone());
        current.source = Some(bayesite_core::investigation::manifest::Source {
            snapshot_id: snapshot_digest(&bytes),
            manifest: parent_ref,
            decision: "initial-likelihood".into(),
        });
        bytes = current.to_bytes().unwrap();
    }
    verify_bundle(&bytes, |reference| {
        objects
            .get(reference.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap();

    let parent_ref = reference(
        &bytes,
        ArtifactKind::InvestigationManifest,
        "investigation-snapshot-v0-provisional",
    );
    objects.insert(parent_ref.sha256.0.clone(), bytes.clone());
    current.source = Some(bayesite_core::investigation::manifest::Source {
        snapshot_id: snapshot_digest(&bytes),
        manifest: parent_ref,
        decision: "initial-likelihood".into(),
    });
    let seventeenth = current.to_bytes().unwrap();
    let error = verify_bundle(&seventeenth, |reference| {
        objects
            .get(reference.as_str())
            .cloned()
            .ok_or_else(|| panic!("missing test object"))
    })
    .unwrap_err();
    assert!(error.message.contains("16-manifest"));
}

#[test]
fn prior_predictive_fixed_identity_vector_is_stable_and_has_no_fit() {
    let (_, _, manifest) = fixture_bundle();
    let recipe = Recipe::new(
        "prior-predictive-initial".into(),
        Operation::PriorPredictive,
        manifest.model.clone(),
        manifest.data.clone(),
        None,
        manifest.recipes[0].engine.clone(),
        json::parse(r#"{"seed":20260915,"draws":200}"#).unwrap(),
    )
    .unwrap();
    assert!(recipe.fit.is_none());
    assert_eq!(
        recipe.sha256.as_str(),
        "07f21d4ead335d6031a076fbd28a17b06014cffe748f6e128354baafb025544e"
    );
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
        "f620b347aa2c73a68dc984cdd010f7178029555056b03051e2a4852b721c76ba"
    );
    assert_eq!(
        json::parse(std::str::from_utf8(&bytes).unwrap()).unwrap(),
        manifest.to_value()
    );
}
