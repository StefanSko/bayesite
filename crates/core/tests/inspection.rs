use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::inspect::{inspect_json, inspect_model};
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::data_from_json;
use bayesite_core::protocol::handle_request;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    json::parse(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn object_entry_mut<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    match value {
        Value::Object(entries) => entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
            .unwrap(),
        _ => panic!("expected object"),
    }
}

fn array_names(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| entry.get("name").and_then(Value::as_str).unwrap())
        .collect()
}

#[test]
fn reports_exact_bound_layout_factor_order_and_jacobians() {
    let fixture = fixture("varying_intercepts_poisson");
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let report = inspect_model(meta, data).unwrap();

    assert_eq!(
        report.get("inspection_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        array_names(report.get("free_slots").unwrap()),
        ["alpha_pop", "sigma_alpha", "z_alpha"]
    );
    let slots = report.get("free_slots").and_then(Value::as_array).unwrap();
    assert_eq!(slots[0].get("offset").and_then(Value::as_i64), Some(0));
    assert_eq!(slots[0].get("length").and_then(Value::as_i64), Some(1));
    assert_eq!(slots[1].get("offset").and_then(Value::as_i64), Some(1));
    assert_eq!(slots[2].get("offset").and_then(Value::as_i64), Some(2));
    assert_eq!(slots[2].get("length").and_then(Value::as_i64), Some(3));
    assert_eq!(
        report
            .get("unconstrained_parameter_count")
            .and_then(Value::as_i64),
        Some(5)
    );
    assert_eq!(
        slots[1]
            .get("resolved_constraint")
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str),
        Some("positive")
    );
    assert_eq!(
        array_names(report.get("density_factors").unwrap()),
        ["alpha_pop", "sigma_alpha", "z_alpha", "y"]
    );
    assert_eq!(
        report
            .get("density_accounting")
            .and_then(|value| value.get("transform_jacobians"))
            .and_then(Value::as_str),
        Some("included_in_evaluated_log_density")
    );
}

#[test]
fn distinguishes_legacy_derived_and_explicit_execution_metadata() {
    let fixture = fixture("linear_regression");
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let explicit = inspect_model(
        decode_model(fixture.get("ir").unwrap()).unwrap(),
        data.clone(),
    )
    .unwrap();

    let mut legacy_ir = fixture.get("ir").unwrap().clone();
    let model = object_entry_mut(&mut legacy_ir, "model");
    *object_entry_mut(model, "free_values") = Value::Array(vec![]);
    *object_entry_mut(model, "stochastic_sites") = Value::Array(vec![]);
    let legacy = inspect_model(decode_model(&legacy_ir).unwrap(), data).unwrap();

    assert_eq!(
        explicit
            .get("execution_metadata")
            .and_then(|value| value.get("free_values"))
            .and_then(Value::as_str),
        Some("explicit")
    );
    assert_eq!(
        legacy
            .get("execution_metadata")
            .and_then(|value| value.get("free_values"))
            .and_then(Value::as_str),
        Some("legacy_derived")
    );
    assert_eq!(
        array_names(explicit.get("free_slots").unwrap()),
        array_names(legacy.get("free_slots").unwrap())
    );
}

#[test]
fn exposes_declared_execution_distribution_mismatch_without_equivalence_claim() {
    let fixture = fixture("linear_regression");
    let mut ir = fixture.get("ir").unwrap().clone();
    let model = object_entry_mut(&mut ir, "model");
    let params = object_entry_mut(model, "params")
        .as_array()
        .unwrap()
        .to_vec();
    let mut changed_params = params;
    let alpha = object_entry_mut(&mut changed_params[0], "value");
    *object_entry_mut(alpha, "distribution") =
        json::parse(r#"{"node":"StudentT","df":4.0,"loc":0.0,"scale":5.0}"#).unwrap();
    *object_entry_mut(model, "params") = Value::Array(changed_params);

    let meta = decode_model(&ir).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let report = inspect_model(meta, data).unwrap();
    let discrepancy = report
        .get("structural_discrepancies")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(
        discrepancy.get("name").and_then(Value::as_str),
        Some("alpha")
    );
    assert_eq!(
        discrepancy
            .get("declared_distribution")
            .and_then(|value| value.get("node"))
            .and_then(Value::as_str),
        Some("StudentT")
    );
    assert_eq!(
        discrepancy
            .get("execution_distribution")
            .and_then(|value| value.get("node"))
            .and_then(Value::as_str),
        Some("Normal")
    );
    assert!(discrepancy
        .get("note")
        .and_then(Value::as_str)
        .unwrap()
        .contains("no mathematical inequivalence"));
}

#[test]
fn exposes_factor_value_expression_mismatch() {
    let fixture = fixture("linear_regression");
    let mut ir = fixture.get("ir").unwrap().clone();
    let model = object_entry_mut(&mut ir, "model");
    let mut sites = object_entry_mut(model, "stochastic_sites")
        .as_array()
        .unwrap()
        .to_vec();
    *object_entry_mut(&mut sites[0], "value") =
        json::parse(r#"{"node":"ParamRef","name":"beta"}"#).unwrap();
    *object_entry_mut(model, "stochastic_sites") = Value::Array(sites);

    let report = inspect_model(
        decode_model(&ir).unwrap(),
        data_from_json(fixture.get("data").unwrap()).unwrap(),
    )
    .unwrap();
    let discrepancies = report
        .get("structural_discrepancies")
        .and_then(Value::as_array)
        .unwrap();
    assert!(discrepancies.iter().any(|discrepancy| {
        discrepancy.get("name").and_then(Value::as_str) == Some("alpha")
            && discrepancy.get("kind").and_then(Value::as_str)
                == Some("declared_value_target_differs_from_execution_factor")
    }));
}

#[test]
fn binding_errors_prevent_partial_inspection() {
    let fixture = fixture("linear_regression");
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let error = inspect_model(meta, vec![]).unwrap_err();
    assert_eq!(error.kind.name(), "DataShapeMismatch");
    assert!(error.message.contains("missing model data"));
}

#[test]
fn cli_and_protocol_emit_equivalent_report_content() {
    let fixture = fixture("linear_regression");
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("bayesite-inspect-{}-{id}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    std::fs::write(
        &model_path,
        json::write(fixture.get("ir").unwrap()).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &data_path,
        json::write(fixture.get("data").unwrap()).unwrap(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "inspect",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let cli = json::parse(std::str::from_utf8(&output.stdout).unwrap()).unwrap();

    let request = Value::Object(vec![
        ("command".into(), Value::Str("inspect".into())),
        ("model".into(), fixture.get("ir").unwrap().clone()),
        ("data".into(), fixture.get("data").unwrap().clone()),
    ]);
    let protocol = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(cli, protocol);

    let direct = inspect_json(
        decode_model(fixture.get("ir").unwrap()).unwrap(),
        data_from_json(fixture.get("data").unwrap()).unwrap(),
    )
    .unwrap();
    assert_eq!(cli, json::parse(&direct).unwrap());
    let _ = std::fs::remove_dir_all(dir);
}
