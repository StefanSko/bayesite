use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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

fn document_hash(value: &Value) -> String {
    sha256_bytes(json::write(value).unwrap().as_bytes())
}

fn request(source: GenerationSource, count: usize, seed: u64) -> GenerationRequest {
    let (model, design, _, _) = linear_parts();
    GenerationRequest {
        meta: decode_model(&model).unwrap(),
        generation_model_hash: document_hash(&model),
        design_hash: document_hash(&design),
        design,
        source,
        count,
        seed,
    }
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

#[test]
fn fixed_generation_redraws_outcomes_and_repeats_natural_parameters() {
    let (_, _, parameters, _) = linear_parts();
    let source = GenerationSource::Fixed {
        parameters_hash: document_hash(&parameters),
        parameters,
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
        fit_data: full_data,
        expected_model_data_fingerprint: None,
    };
    let documents = docs(
        &generated_datasets_ndjson_lines(GenerationRequest {
            meta,
            design_hash: document_hash(&design),
            generation_model_hash: document_hash(&model),
            design,
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
        parameters,
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
        meta: decode_model(&model).unwrap(),
        design_hash: document_hash(&design),
        generation_model_hash: document_hash(&model),
        design,
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
