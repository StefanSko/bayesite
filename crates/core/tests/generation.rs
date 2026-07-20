use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::error::ErrorKind;
use bayesite_core::generation::{
    generated_datasets_ndjson_lines, sha256_bytes, GenerationRequest, GenerationSource,
};
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::protocol::{handle_request, ndjson_lines};
use bayesite_core::sampler::{sample, Settings};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn fixture_text(name: &str) -> String {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

fn fixture(name: &str) -> Value {
    json::parse(&fixture_text(name)).unwrap()
}

fn object_entry_mut<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    let Value::Object(entries) = value else {
        panic!("expected object")
    };
    entries
        .iter_mut()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("missing {key}"))
}

fn linear_parts() -> (Value, Value, Value, Value) {
    let fixture = fixture("linear_regression");
    let model = fixture.get("ir").unwrap().clone();
    let full_data = fixture.get("data").unwrap().clone();
    let x = full_data.get("x").unwrap().clone();
    let design = Value::Object(vec![
        (
            "format".to_string(),
            Value::Str("bayescycle.data.json.v1".to_string()),
        ),
        (
            "variables".to_string(),
            Value::Object(vec![("x".to_string(), x)]),
        ),
    ]);
    let parameters = json::parse(
        r#"{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]},"beta":{"dtype":"float64","shape":[],"values":[1.2]},"sigma":{"dtype":"float64","shape":[],"values":[0.4]}}}"#,
    )
    .unwrap();
    (model, design, parameters, full_data)
}

fn non_ancestral_outcome_parts() -> (Value, Value, Value, Value) {
    let (mut model, design, parameters, mut full_data) = linear_parts();
    let child_distribution = json::parse(
        r#"{"node":"Normal","loc":{"node":"DataRef","name":"y"},"scale":{"node":"ConstNode","value":1e-9}}"#,
    )
    .unwrap();
    let observed = Value::Object(vec![
        (
            "node".to_string(),
            Value::Str("ResolvedObserved".to_string()),
        ),
        ("name".to_string(), Value::Str("z".to_string())),
        ("distribution".to_string(), child_distribution.clone()),
    ]);
    let site = Value::Object(vec![
        (
            "node".to_string(),
            Value::Str("ResolvedStochasticSite".to_string()),
        ),
        ("name".to_string(), Value::Str("z".to_string())),
        ("distribution".to_string(), child_distribution),
        (
            "value".to_string(),
            Value::Object(vec![
                ("node".to_string(), Value::Str("DataRef".to_string())),
                ("name".to_string(), Value::Str("z".to_string())),
            ]),
        ),
    ]);
    let hierarchical_beta = json::parse(
        r#"{"node":"Normal","loc":{"node":"ParamRef","name":"alpha"},"scale":{"node":"ConstNode","value":1.0}}"#,
    )
    .unwrap();
    let model_body = object_entry_mut(&mut model, "model");
    let Value::Array(params) = object_entry_mut(model_body, "params") else {
        panic!("expected params")
    };
    let beta = params
        .iter_mut()
        .find(|entry| entry.get("name").and_then(Value::as_str) == Some("beta"))
        .expect("beta parameter");
    let beta_value = object_entry_mut(beta, "value");
    *object_entry_mut(beta_value, "distribution") = hierarchical_beta.clone();
    let Value::Array(observed_nodes) = object_entry_mut(model_body, "observed_nodes") else {
        panic!("expected observed_nodes")
    };
    observed_nodes.push(observed);
    let Value::Array(sites) = object_entry_mut(model_body, "stochastic_sites") else {
        panic!("expected stochastic_sites")
    };
    let beta_site = sites
        .iter_mut()
        .find(|entry| entry.get("name").and_then(Value::as_str) == Some("beta"))
        .expect("beta site");
    *object_entry_mut(beta_site, "distribution") = hierarchical_beta;
    sites.swap(0, 1);
    sites.insert(sites.len() - 1, site);
    let y = full_data.get("y").expect("linear y data").clone();
    let Value::Object(entries) = &mut full_data else {
        panic!("fixture data must be an object")
    };
    entries.push(("z".to_string(), y));
    (model, design, parameters, full_data)
}

fn document_hash(value: &Value) -> String {
    sha256_bytes(json::write(value).unwrap().as_bytes())
}

fn request_for(
    model: Value,
    design: Value,
    source: GenerationSource,
    count: usize,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: document_hash(&model),
        design_hash: document_hash(&design),
        source,
        count,
        seed,
    }
}

fn request(source: GenerationSource, count: usize, seed: u64) -> GenerationRequest {
    let (model, design, _, _) = linear_parts();
    request_for(model, design, source, count, seed)
}

fn with_invalid_matvec(mut model: Value, matrix_name: &str) -> Value {
    let expressions = object_entry_mut(object_entry_mut(&mut model, "model"), "expressions");
    let Value::Array(expressions) = expressions else {
        panic!("expected expressions")
    };
    expressions.push(Value::Object(vec![
        ("name".to_string(), Value::Str("invalid".to_string())),
        (
            "value".to_string(),
            Value::Object(vec![
                ("node".to_string(), Value::Str("MatVecOp".to_string())),
                (
                    "matrix".to_string(),
                    Value::Object(vec![
                        ("node".to_string(), Value::Str("DataRef".to_string())),
                        ("name".to_string(), Value::Str(matrix_name.to_string())),
                    ]),
                ),
                (
                    "vector".to_string(),
                    Value::Object(vec![
                        ("node".to_string(), Value::Str("DataRef".to_string())),
                        ("name".to_string(), Value::Str("x".to_string())),
                    ]),
                ),
            ]),
        ),
    ]));
    model
}

fn docs(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|line| json::parse(line).unwrap())
        .collect()
}

fn variables(document: &Value) -> &Value {
    document.get("variables").expect("canonical variables")
}

fn draw_parameters(draw: &Value) -> &Value {
    variables(draw.get("parameters").expect("draw parameters"))
}

fn draw_dataset(draw: &Value) -> &Value {
    variables(draw.get("dataset").expect("draw dataset"))
}

fn numeric_values(value: &Value) -> Vec<f64> {
    value
        .get("values")
        .and_then(Value::as_array)
        .expect("typed variable values")
        .iter()
        .map(|value| value.as_f64().expect("numeric value"))
        .collect()
}

fn assert_non_ancestral_outcome_draw(draw: &Value) {
    let dataset = draw_dataset(draw);
    let Value::Object(entries) = dataset else {
        panic!("dataset variables must be ordered object entries")
    };
    assert_eq!(
        entries
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["x", "z", "y"]
    );
    let child = numeric_values(dataset.get("z").expect("z generated"));
    let parent = numeric_values(dataset.get("y").expect("y generated"));
    assert_eq!(child.len(), parent.len());
    for (child, parent) in child.iter().zip(parent) {
        assert!((child - parent).abs() < 1e-7);
    }
}

#[test]
fn fixed_and_model_prior_generation_reject_invalid_bound_or_generated_matvec() {
    let (base_model, design, parameters, _) = linear_parts();
    for matrix_name in ["x", "y", "missing_matrix"] {
        let model = with_invalid_matvec(base_model.clone(), matrix_name);
        let parameters_document = json::write(&parameters).unwrap();
        let fixed = GenerationSource::Fixed {
            parameters_hash: sha256_bytes(parameters_document.as_bytes()),
            parameters_document,
        };
        let fixed_error = generated_datasets_ndjson_lines(request_for(
            model.clone(),
            design.clone(),
            fixed,
            1,
            11,
        ))
        .unwrap_err();
        if matrix_name == "missing_matrix" {
            assert_eq!(fixed_error.kind, ErrorKind::MalformedDocument);
            assert!(fixed_error.message.contains("missing_matrix"));
        } else {
            assert_eq!(fixed_error.kind, ErrorKind::DataShapeMismatch);
            assert!(fixed_error.message.contains("matrix must be rank 2"));
        }

        let model_prior = GenerationSource::ModelPrior {
            model_hash: document_hash(&model),
            authored_provenance: None,
        };
        let prior_error =
            generated_datasets_ndjson_lines(request_for(model, design.clone(), model_prior, 1, 13))
                .unwrap_err();
        if matrix_name == "missing_matrix" {
            assert_eq!(prior_error.kind, ErrorKind::MalformedDocument);
            assert!(prior_error.message.contains("missing_matrix"));
        } else {
            assert_eq!(prior_error.kind, ErrorKind::DataShapeMismatch);
            assert!(prior_error.message.contains("matrix must be rank 2"));
        }
    }
}

#[test]
fn fixed_generation_normalizes_continuous_parameter_dtype() {
    let (model, design, _, _) = linear_parts();
    let parameters = json::parse(
        r#"{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"int64","shape":[],"values":[1]},"beta":{"dtype":"int64","shape":[],"values":[2]},"sigma":{"dtype":"int64","shape":[],"values":[1]}}}"#,
    )
    .unwrap();
    let parameters_document = json::write(&parameters).unwrap();
    let source = GenerationSource::Fixed {
        parameters_hash: sha256_bytes(parameters_document.as_bytes()),
        parameters_document,
    };
    let documents =
        docs(&generated_datasets_ndjson_lines(request_for(model, design, source, 1, 17)).unwrap());
    let generated = draw_parameters(&documents[1]);
    for name in ["alpha", "beta", "sigma"] {
        assert_eq!(
            generated
                .get(name)
                .and_then(|value| value.get("dtype"))
                .and_then(Value::as_str),
            Some("float64")
        );
    }
}

#[test]
fn fixed_generation_redraws_outcomes_and_repeats_natural_parameters() {
    let (_, _, parameters, _) = linear_parts();
    let source = GenerationSource::Fixed {
        parameters_hash: document_hash(&parameters),
        parameters_document: json::write(&parameters).unwrap(),
    };
    let lines = generated_datasets_ndjson_lines(request(source.clone(), 3, 17)).unwrap();
    let again = generated_datasets_ndjson_lines(request(source, 3, 17)).unwrap();
    assert_eq!(lines, again, "fixed generation must be deterministic");

    let documents = docs(&lines);
    assert_eq!(documents.len(), 5);
    assert_eq!(documents[0].get("count").and_then(Value::as_i64), Some(3));
    let draws = &documents[1..4];
    assert!(draws
        .iter()
        .all(|draw| draw_parameters(draw) == draw_parameters(&draws[0])));
    assert!(draws
        .iter()
        .all(|draw| draw_dataset(draw).get("x").is_some()));
    assert!(draws
        .iter()
        .all(|draw| draw_dataset(draw).get("y").is_some()));
    assert_ne!(
        draw_dataset(&draws[0]).get("y"),
        draw_dataset(&draws[1]).get("y")
    );
}

#[test]
fn model_prior_redraws_parameters_per_dataset() {
    let (model, _, _, _) = linear_parts();
    let source = GenerationSource::ModelPrior {
        model_hash: document_hash(&model),
        authored_provenance: None,
    };
    let documents = docs(&generated_datasets_ndjson_lines(request(source, 3, 23)).unwrap());
    assert_ne!(
        draw_parameters(&documents[1]),
        draw_parameters(&documents[2]),
        "model-prior parameters must redraw"
    );
    assert_eq!(
        documents[1]
            .get("source_lineage")
            .and_then(|lineage| lineage.get("source_draw_index"))
            .and_then(Value::as_i64),
        Some(0)
    );
}

#[test]
fn fixed_generation_schedules_outcomes_but_emits_metadata_order() {
    let (model, design, parameters, _) = non_ancestral_outcome_parts();
    let source = GenerationSource::Fixed {
        parameters_hash: document_hash(&parameters),
        parameters_document: json::write(&parameters).unwrap(),
    };
    let request = GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: document_hash(&model),
        design_hash: document_hash(&design),
        source,
        count: 2,
        seed: 29,
    };
    let documents = docs(&generated_datasets_ndjson_lines(request).unwrap());

    for draw in &documents[1..3] {
        assert_non_ancestral_outcome_draw(draw);
    }
}

#[test]
fn model_prior_generation_schedules_outcomes_but_emits_metadata_order() {
    let (model, design, _, _) = non_ancestral_outcome_parts();
    let source = GenerationSource::ModelPrior {
        model_hash: document_hash(&model),
        authored_provenance: None,
    };
    let request = GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: document_hash(&model),
        design_hash: document_hash(&design),
        source,
        count: 2,
        seed: 31,
    };
    let documents = docs(&generated_datasets_ndjson_lines(request).unwrap());

    for draw in &documents[1..3] {
        assert_non_ancestral_outcome_draw(draw);
    }
}

#[test]
fn posterior_generation_schedules_outcomes_but_preserves_lineage_and_metadata_order() {
    let (model, design, _, full_data) = non_ancestral_outcome_parts();
    let meta = decode_model(&model).unwrap();
    let fit_data = data_from_json(&full_data).unwrap();
    let posterior = Posterior::new(meta.clone(), fit_data).unwrap();
    let settings = Settings {
        num_warmup: 10,
        num_draws: 4,
        max_treedepth: 4,
        target_accept: 0.8,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 35, 0).unwrap();
    let fit = ndjson_lines(&posterior, &settings, 35, &[(0, chain)])
        .unwrap()
        .join("\n");
    let source = GenerationSource::Posterior {
        fit_hash: sha256_bytes(fit.as_bytes()),
        fit_model_hash: document_hash(&model),
        fit_data_hash: document_hash(&full_data),
        fit_ndjson: fit,
        fit_data_document: json::write(&full_data).unwrap(),
        expected_model_data_fingerprint: None,
    };
    let request = GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: document_hash(&model),
        design_hash: document_hash(&design),
        source,
        count: 2,
        seed: 37,
    };
    let documents = docs(&generated_datasets_ndjson_lines(request).unwrap());

    for draw in &documents[1..3] {
        assert_non_ancestral_outcome_draw(draw);
        assert!(draw
            .get("source_lineage")
            .and_then(|lineage| lineage.get("source_draw_index"))
            .and_then(Value::as_i64)
            .is_some());
    }
}

#[test]
fn posterior_generation_samples_fit_draws_with_replacement_and_fresh_outcomes() {
    let (model, design, _, full_data) = linear_parts();
    let meta = decode_model(&model).unwrap();
    let fit_data = data_from_json(&full_data).unwrap();
    let posterior = Posterior::new(meta.clone(), fit_data).unwrap();
    let settings = Settings {
        num_warmup: 20,
        num_draws: 4,
        max_treedepth: 5,
        target_accept: 0.8,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 31, 0).unwrap();
    let fit = ndjson_lines(&posterior, &settings, 31, &[(0, chain)])
        .unwrap()
        .join("\n");
    let source = GenerationSource::Posterior {
        fit_hash: sha256_bytes(fit.as_bytes()),
        fit_model_hash: document_hash(&model),
        fit_data_hash: document_hash(&full_data),
        fit_ndjson: fit,
        fit_data_document: json::write(&full_data).unwrap(),
        expected_model_data_fingerprint: None,
    };
    let documents = docs(
        &generated_datasets_ndjson_lines(GenerationRequest {
            model_document: json::write(&model).unwrap(),
            design_document: json::write(&design).unwrap(),
            design_hash: document_hash(&design),
            generation_model_hash: document_hash(&model),
            source,
            count: 10,
            seed: 37,
        })
        .unwrap(),
    );
    let source_indices = documents[1..11]
        .iter()
        .map(|draw| {
            draw.get("source_lineage")
                .and_then(|lineage| lineage.get("source_draw_index"))
                .and_then(Value::as_i64)
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(source_indices.iter().all(|index| (0..4).contains(index)));
    assert!(
        (0..source_indices.len()).any(|left| {
            (left + 1..source_indices.len())
                .any(|right| source_indices[left] == source_indices[right])
        }),
        "ten selections from four draws must contain a repeated source"
    );
}

#[test]
fn generation_rejects_bounds_non_param_free_values_and_identity_mismatch() {
    let (_, _, parameters, _) = linear_parts();
    let fixed = GenerationSource::Fixed {
        parameters_hash: document_hash(&parameters),
        parameters_document: json::write(&parameters).unwrap(),
    };
    assert!(
        generated_datasets_ndjson_lines(request(fixed.clone(), 0, 1))
            .unwrap_err()
            .message
            .contains("count")
    );
    assert!(generated_datasets_ndjson_lines(request(fixed, 1001, 1))
        .unwrap_err()
        .message
        .contains("1000"));

    let partial = fixture("partially_observed_mvn");
    let design = partial.get("data").unwrap().clone();
    let model = partial.get("ir").unwrap().clone();
    let error = generated_datasets_ndjson_lines(GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        design_hash: document_hash(&design),
        generation_model_hash: document_hash(&model),
        source: GenerationSource::ModelPrior {
            model_hash: document_hash(&model),
            authored_provenance: None,
        },
        count: 1,
        seed: 1,
    })
    .unwrap_err();
    assert!(error.message.contains("non-Param free value"));

    let error = generated_datasets_ndjson_lines(request(
        GenerationSource::ModelPrior {
            model_hash: format!("sha256:{}", "0".repeat(64)),
            authored_provenance: None,
        },
        1,
        1,
    ))
    .unwrap_err();
    assert!(error.message.contains("model hash"));
}

#[test]
fn protocol_and_cli_match_one_native_generate_operation() {
    let (model, design, parameters, _) = linear_parts();
    let identities = Value::Object(vec![
        (
            "generation_model_hash".to_string(),
            Value::Str(document_hash(&model)),
        ),
        (
            "design_hash".to_string(),
            Value::Str(document_hash(&design)),
        ),
        (
            "parameters_hash".to_string(),
            Value::Str(document_hash(&parameters)),
        ),
    ]);
    let protocol_request = Value::Object(vec![
        ("command".to_string(), Value::Str("generate".to_string())),
        ("model".to_string(), model.clone()),
        ("design".to_string(), design.clone()),
        (
            "parameter_source".to_string(),
            Value::Object(vec![
                ("kind".to_string(), Value::Str("fixed".to_string())),
                ("parameters".to_string(), parameters.clone()),
            ]),
        ),
        ("count".to_string(), Value::Int(2)),
        ("seed".to_string(), Value::Int(41)),
        ("identities".to_string(), identities),
    ]);
    let protocol_output = handle_request(&json::write(&protocol_request).unwrap());
    let protocol_header = json::parse(protocol_output.lines().next().unwrap()).unwrap();
    assert_eq!(
        protocol_header
            .get("generated_datasets_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );

    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("bayesite-generate-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let design_path = dir.join("design.json");
    let parameters_path = dir.join("parameters.json");
    std::fs::write(&model_path, json::write(&model).unwrap()).unwrap();
    std::fs::write(&design_path, json::write(&design).unwrap()).unwrap();
    std::fs::write(&parameters_path, json::write(&parameters).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "generate",
            "--model",
            model_path.to_str().unwrap(),
            "--design",
            design_path.to_str().unwrap(),
            "--source",
            "fixed",
            "--parameters",
            parameters_path.to_str().unwrap(),
            "--count",
            "2",
            "--seed",
            "41",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim_end(),
        protocol_output
    );
    std::fs::remove_dir_all(dir).unwrap();
}
