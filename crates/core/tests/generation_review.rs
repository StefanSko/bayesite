use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::fingerprint::model_data_fingerprint;
use bayesite_core::generation::{
    generated_datasets_ndjson_lines, sha256_bytes, GenerationRequest, GenerationSource,
};
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::protocol::{
    diagnose_ndjson, handle_request, ndjson_lines, ndjson_lines_with_model_data_fingerprint,
};
use bayesite_core::sampler::{sample, Settings};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    json::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn hash(value: &Value) -> String {
    sha256_bytes(json::write(value).unwrap().as_bytes())
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

fn canonical(variables: Vec<(String, Value)>) -> Value {
    Value::Object(vec![
        (
            "format".to_string(),
            Value::Str("bayescycle.data.json.v1".to_string()),
        ),
        ("variables".to_string(), Value::Object(variables)),
    ])
}

fn linear_parts() -> (Value, Value, Value, Value) {
    let fixture = fixture("linear_regression");
    let model = fixture.get("ir").unwrap().clone();
    let full_data = fixture.get("data").unwrap().clone();
    let design = canonical(vec![("x".to_string(), full_data.get("x").unwrap().clone())]);
    let parameters = json::parse(
        r#"{"format":"bayescycle.data.json.v1","variables":{"alpha":{"dtype":"float64","shape":[],"values":[0.5]},"beta":{"dtype":"float64","shape":[],"values":[1.2]},"sigma":{"dtype":"float64","shape":[],"values":[0.4]}}}"#,
    )
    .unwrap();
    (model, design, parameters, full_data)
}

fn sampled_fit(model: &Value, data: &Value) -> String {
    let posterior =
        Posterior::new(decode_model(model).unwrap(), data_from_json(data).unwrap()).unwrap();
    let settings = Settings {
        num_warmup: 20,
        num_draws: 4,
        max_treedepth: 5,
        target_accept: 0.8,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 81, 0).unwrap();
    ndjson_lines(&posterior, &settings, 81, &[(0, chain)])
        .unwrap()
        .join("\n")
}

fn posterior_request(fit: String, seed: u64) -> GenerationRequest {
    let (model, design, _, fit_data) = linear_parts();
    GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: hash(&model),
        design_hash: hash(&design),
        source: GenerationSource::Posterior {
            fit_hash: sha256_bytes(fit.as_bytes()),
            fit_model_hash: hash(&model),
            fit_data_hash: hash(&fit_data),
            fit_ndjson: fit,
            fit_data_document: json::write(&fit_data).unwrap(),
            expected_model_data_fingerprint: None,
        },
        count: 1,
        seed,
    }
}

#[test]
fn fixed_constraint_errors_follow_parameter_order_deterministically() {
    let fixture = fixture("bounded_rates");
    let mut model = fixture.get("ir").unwrap().clone();
    let model_object = object_entry_mut(&mut model, "model");
    for key in ["params", "free_values"] {
        let Value::Array(values) = object_entry_mut(model_object, key) else {
            panic!("expected {key}")
        };
        values.reverse();
    }
    let Value::Array(sites) = object_entry_mut(model_object, "stochastic_sites") else {
        panic!("expected stochastic sites")
    };
    sites[..2].reverse();
    let design = canonical(Vec::new());
    let parameters = json::parse(
        r#"{"format":"bayescycle.data.json.v1","variables":{"level":{"dtype":"float64","shape":[],"values":[4.0]},"p":{"dtype":"float64","shape":[],"values":[2.0]}}}"#,
    )
    .unwrap();
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "bayesite-generate-determinism-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, value) in [
        ("model", model),
        ("design", design),
        ("parameters", parameters),
    ] {
        std::fs::write(
            dir.join(format!("{name}.json")),
            json::write(&value).unwrap(),
        )
        .unwrap();
    }
    for _ in 0..32 {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "generate",
                "--model",
                dir.join("model.json").to_str().unwrap(),
                "--design",
                dir.join("design.json").to_str().unwrap(),
                "--source",
                "fixed",
                "--parameters",
                dir.join("parameters.json").to_str().unwrap(),
                "--count",
                "1",
                "--seed",
                "7",
            ])
            .output()
            .unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("level"), "{stderr}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn protocol_rejects_forged_exact_hashes() {
    let (model, design, parameters, _) = linear_parts();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("generate".to_string())),
        ("model".to_string(), model),
        ("design".to_string(), design),
        (
            "parameter_source".to_string(),
            Value::Object(vec![
                ("kind".to_string(), Value::Str("fixed".to_string())),
                ("parameters".to_string(), parameters),
            ]),
        ),
        ("count".to_string(), Value::Int(1)),
        ("seed".to_string(), Value::Int(7)),
        (
            "identities".to_string(),
            Value::Object(vec![
                (
                    "generation_model_hash".to_string(),
                    Value::Str(format!("sha256:{}", "0".repeat(64))),
                ),
                (
                    "design_hash".to_string(),
                    Value::Str(format!("sha256:{}", "0".repeat(64))),
                ),
                (
                    "parameters_hash".to_string(),
                    Value::Str(format!("sha256:{}", "0".repeat(64))),
                ),
            ]),
        ),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert!(response
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("hash"));
}

#[test]
fn posterior_validates_every_draw_before_seeded_selection() {
    let (model, _, _, data) = linear_parts();
    let fit = sampled_fit(&model, &data);
    let malformed = fit.replacen("\"sigma\":", "\"sigma\":-1.0,\"ignored_sigma\":", 1);
    for seed in 0..16 {
        let error = generated_datasets_ndjson_lines(posterior_request(malformed.clone(), seed))
            .unwrap_err();
        assert!(error.message.contains("sigma") || error.message.contains("fit"));
    }
}

#[test]
fn posterior_rejects_metadata_that_diagnose_rejects() {
    let (model, _, _, data) = linear_parts();
    let fit = sampled_fit(&model, &data);
    let mut lines: Vec<String> = fit.lines().map(str::to_string).collect();
    lines[1] = lines[1]
        .replacen(
            "\"artifact_scope\":\"observed_data_conditioned_parameter_draws\"",
            "\"artifact_scope\":\"wrong_scope\"",
            1,
        )
        .replacen("\"draw_count\":4", "\"draw_count\":999", 1);
    let malformed = lines.join("\n");
    let error = generated_datasets_ndjson_lines(posterior_request(malformed, 1)).unwrap_err();
    assert!(error.message.contains("artifact_scope") || error.message.contains("draw_count"));
}

#[test]
fn output_bound_preflights_before_parameter_or_sampling_work() {
    let (model, _, mut parameters, _) = linear_parts();
    let values = (0..20_000).map(|_| Value::Float(0.0)).collect();
    let design = canonical(vec![(
        "x".to_string(),
        Value::Object(vec![
            ("dtype".to_string(), Value::Str("float64".to_string())),
            ("shape".to_string(), Value::Array(vec![Value::Int(20_000)])),
            ("values".to_string(), Value::Array(values)),
        ]),
    )]);
    let variables = object_entry_mut(&mut parameters, "variables");
    let sigma = object_entry_mut(variables, "sigma");
    *object_entry_mut(sigma, "values") = Value::Array(vec![Value::Float(-1.0)]);
    let error = generated_datasets_ndjson_lines(GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: hash(&model),
        design_hash: hash(&design),
        source: GenerationSource::Fixed {
            parameters_hash: hash(&parameters),
            parameters_document: json::write(&parameters).unwrap(),
        },
        count: 1000,
        seed: 7,
    })
    .unwrap_err();
    assert!(error.message.contains("artifact") && error.message.contains("bytes"));
}

#[test]
fn canonical_design_rejects_bad_specs_and_accepts_top_level_reordering() {
    let (model, design, parameters, _) = linear_parts();
    let mut malformed = design.clone();
    let variables = object_entry_mut(&mut malformed, "variables");
    let x = object_entry_mut(variables, "x");
    *object_entry_mut(x, "dtype") = Value::Str("garbage".to_string());
    let Value::Object(spec) = x else {
        unreachable!()
    };
    spec.push(("extra".to_string(), Value::Int(123)));
    let bad = GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&malformed).unwrap(),
        generation_model_hash: hash(&model),
        design_hash: hash(&malformed),
        source: GenerationSource::Fixed {
            parameters_hash: hash(&parameters),
            parameters_document: json::write(&parameters).unwrap(),
        },
        count: 1,
        seed: 7,
    };
    assert!(generated_datasets_ndjson_lines(bad).is_err());

    let variables = design.get("variables").unwrap().clone();
    let reordered = Value::Object(vec![
        ("variables".to_string(), variables),
        (
            "format".to_string(),
            Value::Str("bayescycle.data.json.v1".to_string()),
        ),
    ]);
    let good = GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&reordered).unwrap(),
        generation_model_hash: hash(&model),
        design_hash: hash(&reordered),
        source: GenerationSource::Fixed {
            parameters_hash: hash(&parameters),
            parameters_document: json::write(&parameters).unwrap(),
        },
        count: 1,
        seed: 7,
    };
    assert!(generated_datasets_ndjson_lines(good).is_ok());
}

#[test]
fn protocol_rejects_byte_distinct_posterior_fingerprint_inputs() {
    let (model, design, _, fit_data) = linear_parts();
    let model_text = json::write(&model).unwrap();
    let fit_data_text = json::write(&fit_data).unwrap();
    let posterior = Posterior::new(
        decode_model(&model).unwrap(),
        data_from_json(&fit_data).unwrap(),
    )
    .unwrap();
    let settings = Settings {
        num_warmup: 20,
        num_draws: 4,
        max_treedepth: 5,
        target_accept: 0.8,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 82, 0).unwrap();
    let fingerprint = model_data_fingerprint(&model_text, &fit_data_text);
    let fit = ndjson_lines_with_model_data_fingerprint(
        &posterior,
        &settings,
        82,
        &[(0, chain)],
        Some(&fingerprint),
    )
    .unwrap()
    .join("\n");
    let changed_model = format!("{model_text}\n");
    let changed_fit_data = format!("{fit_data_text}\n");
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("generate".to_string())),
        ("model".to_string(), Value::Str(changed_model.clone())),
        (
            "design".to_string(),
            Value::Str(json::write(&design).unwrap()),
        ),
        (
            "parameter_source".to_string(),
            Value::Object(vec![
                ("kind".to_string(), Value::Str("posterior".to_string())),
                ("fit".to_string(), Value::Str(fit.clone())),
                ("fit_data".to_string(), Value::Str(changed_fit_data.clone())),
            ]),
        ),
        ("count".to_string(), Value::Int(1)),
        ("seed".to_string(), Value::Int(7)),
        (
            "identities".to_string(),
            Value::Object(vec![
                (
                    "generation_model_hash".to_string(),
                    Value::Str(sha256_bytes(changed_model.as_bytes())),
                ),
                ("design_hash".to_string(), Value::Str(hash(&design))),
                (
                    "fit_hash".to_string(),
                    Value::Str(sha256_bytes(fit.as_bytes())),
                ),
                (
                    "fit_model_hash".to_string(),
                    Value::Str(sha256_bytes(changed_model.as_bytes())),
                ),
                (
                    "fit_data_hash".to_string(),
                    Value::Str(sha256_bytes(changed_fit_data.as_bytes())),
                ),
            ]),
        ),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    assert!(response.contains("model_data_fingerprint"), "{response}");
}

#[test]
fn posterior_accepts_every_fit_metadata_variant_accepted_by_diagnose() {
    let (model, _, _, data) = linear_parts();
    let fit = sampled_fit(&model, &data);
    let mut lines = fit.lines().map(str::to_string).collect::<Vec<_>>();
    let trailer_index = lines.len() - 1;
    for line in &mut lines[1..trailer_index] {
        let mut draw = json::parse(line).unwrap();
        let Value::Object(fields) = &mut draw else {
            unreachable!()
        };
        fields.retain(|(name, _)| {
            !matches!(
                name.as_str(),
                "draw_index"
                    | "draws_format"
                    | "artifact_kind"
                    | "artifact_scope"
                    | "chain_index_base"
                    | "draw_index_base"
                    | "draw_count"
                    | "seed"
            )
        });
        *line = json::write(&draw).unwrap();
    }
    let legacy = lines.join("\n");
    diagnose_ndjson(&legacy).unwrap();
    generated_datasets_ndjson_lines(posterior_request(legacy, 7)).unwrap();
}

#[test]
fn canonical_integer_design_rejects_fractional_payloads() {
    let (model, mut design, parameters, _) = linear_parts();
    let variables = object_entry_mut(&mut design, "variables");
    let x = object_entry_mut(variables, "x");
    *object_entry_mut(x, "dtype") = Value::Str("int32".to_string());
    *object_entry_mut(x, "values") = Value::Array((0..5).map(|_| Value::Float(0.5)).collect());
    let error = generated_datasets_ndjson_lines(GenerationRequest {
        model_document: json::write(&model).unwrap(),
        design_document: json::write(&design).unwrap(),
        generation_model_hash: hash(&model),
        design_hash: hash(&design),
        source: GenerationSource::Fixed {
            parameters_hash: hash(&parameters),
            parameters_document: json::write(&parameters).unwrap(),
        },
        count: 1,
        seed: 7,
    })
    .unwrap_err();
    assert!(error.message.contains("int32") || error.message.contains("integer"));
}

#[test]
fn cli_requires_explicit_generation_count() {
    let (model, design, parameters, _) = linear_parts();
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "bayesite-generate-count-review-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, value) in [
        ("model", model),
        ("design", design),
        ("parameters", parameters),
    ] {
        std::fs::write(
            dir.join(format!("{name}.json")),
            json::write(&value).unwrap(),
        )
        .unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "generate",
            "--model",
            dir.join("model.json").to_str().unwrap(),
            "--design",
            dir.join("design.json").to_str().unwrap(),
            "--source",
            "fixed",
            "--parameters",
            dir.join("parameters.json").to_str().unwrap(),
            "--seed",
            "7",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("--count is required"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn cli_requires_explicit_generation_seed() {
    let (model, design, parameters, _) = linear_parts();
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "bayesite-generate-review-{}-{id}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, value) in [
        ("model", model),
        ("design", design),
        ("parameters", parameters),
    ] {
        std::fs::write(
            dir.join(format!("{name}.json")),
            json::write(&value).unwrap(),
        )
        .unwrap();
    }
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "generate",
            "--model",
            dir.join("model.json").to_str().unwrap(),
            "--design",
            dir.join("design.json").to_str().unwrap(),
            "--source",
            "fixed",
            "--parameters",
            dir.join("parameters.json").to_str().unwrap(),
            "--count",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8(output.stderr)
        .unwrap()
        .contains("--seed is required"));
    std::fs::remove_dir_all(dir).unwrap();
}
