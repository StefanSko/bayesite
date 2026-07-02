use std::path::PathBuf;
use std::process::Command;

use bayesite_core::json::{self, Value};

fn fixture_text(name: &str) -> String {
    // Conformance fixtures come from the vendored bayeswire corpus. Names
    // prefixed `cli_` are engine-behavior inputs kept under tests/data/
    // because their models use `Truncated`, a core-profile tag this backend
    // does not evaluate yet (see fixtures_eval.rs).
    let corpus = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let local = format!(
        "{}/tests/data/cli_models/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let path = if name.starts_with("cli_") {
        local
    } else {
        corpus
    };
    std::fs::read_to_string(path).expect("fixture readable")
}

fn short_diagnostics_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "bayesite-short-diagnostics-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir created");
    dir
}

fn write_linear_regression_inputs() -> (PathBuf, PathBuf, PathBuf) {
    let fixture = json::parse(&fixture_text("cli_linear_regression")).expect("fixture parses");
    let dir = short_diagnostics_dir("sample");
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    let fit_path = dir.join("fit.jsonl");
    std::fs::write(
        &model_path,
        json::write(fixture.get("ir").expect("fixture ir")).expect("ir writes"),
    )
    .expect("model written");
    std::fs::write(
        &data_path,
        json::write(fixture.get("data").expect("fixture data")).expect("data writes"),
    )
    .expect("data written");
    (model_path, data_path, fit_path)
}

fn write_linear_regression_workflow_inputs() -> (PathBuf, PathBuf, PathBuf) {
    let fixture = json::parse(&fixture_text("cli_linear_regression")).expect("fixture parses");
    let dir = short_diagnostics_dir("workflow");
    let model_path = dir.join("model.json");
    let recover_path = dir.join("recover.json");
    let sbc_path = dir.join("sbc.json");
    let declared_data = Value::Object(vec![(
        "x".to_string(),
        fixture
            .get("data")
            .and_then(|data| data.get("x"))
            .expect("declared x data")
            .clone(),
    )]);
    let sample_settings = Value::Object(vec![
        ("chains".to_string(), Value::Int(1)),
        ("warmup".to_string(), Value::Int(10)),
        ("draws".to_string(), Value::Int(10)),
        ("max_treedepth".to_string(), Value::Int(4)),
        ("target_accept".to_string(), Value::Float(0.8)),
    ]);
    let recover = Value::Object(vec![
        (
            "recover_scenario".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        ("data".to_string(), declared_data.clone()),
        ("seed".to_string(), Value::Int(1)),
        ("interval".to_string(), Value::Float(0.8)),
        ("sample".to_string(), sample_settings.clone()),
    ]);
    let sbc = Value::Object(vec![
        (
            "sbc_scenario".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        ("data".to_string(), declared_data),
        ("seed".to_string(), Value::Int(1)),
        ("replicates".to_string(), Value::Int(1)),
        ("sample".to_string(), sample_settings),
    ]);
    std::fs::write(
        &model_path,
        json::write(fixture.get("ir").expect("fixture ir")).expect("ir writes"),
    )
    .expect("model written");
    std::fs::write(
        &recover_path,
        json::write(&recover).expect("recover writes"),
    )
    .expect("recover scenario written");
    std::fs::write(&sbc_path, json::write(&sbc).expect("sbc writes"))
        .expect("sbc scenario written");
    (model_path, recover_path, sbc_path)
}

fn assert_diagnostic_value_is_json_safe(value: &Value) {
    match value {
        Value::Null => {}
        Value::Float(number) => assert!(number.is_finite(), "diagnostic must be finite or null"),
        Value::Int(_) => {}
        other => panic!("diagnostic must be a number or null, got {other:?}"),
    }
}

fn assert_diagnostic_map_is_json_safe(payload: &Value, field: &str) {
    let entries = match payload.get(field).expect("diagnostic map") {
        Value::Object(entries) => entries,
        other => panic!("diagnostic map must be an object, got {other:?}"),
    };
    for (_, value) in entries {
        assert_diagnostic_value_is_json_safe(value);
    }
}

fn assert_nested_diagnostics_are_json_safe(value: &Value) {
    match value {
        Value::Object(entries) => {
            for (key, item) in entries {
                if key == "rhat" || key == "ess" {
                    match item {
                        Value::Object(values) => {
                            for (_, diagnostic) in values {
                                assert_diagnostic_value_is_json_safe(diagnostic);
                            }
                        }
                        diagnostic => assert_diagnostic_value_is_json_safe(diagnostic),
                    }
                }
                assert_nested_diagnostics_are_json_safe(item);
            }
        }
        Value::Array(items) => {
            for item in items {
                assert_nested_diagnostics_are_json_safe(item);
            }
        }
        Value::Null | Value::Bool(_) | Value::Int(_) | Value::Float(_) | Value::Str(_) => {}
    }
}

#[test]
fn short_sample_and_diagnose_artifacts_do_not_fail_on_unavailable_diagnostics() {
    let (model_path, data_path, fit_path) = write_linear_regression_inputs();
    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().expect("model path utf-8"),
            "--data",
            data_path.to_str().expect("data path utf-8"),
            "--seed",
            "1",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "10",
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "sample stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );
    assert!(sample.stderr.is_empty());

    let fit = String::from_utf8(sample.stdout).expect("sample stdout utf-8");
    let lines: Vec<&str> = fit.lines().collect();
    assert_eq!(lines.len(), 12);
    let trailer = json::parse(lines.last().expect("sample trailer"))
        .expect("trailer parses")
        .get("trailer")
        .expect("trailer object")
        .clone();
    assert_diagnostic_map_is_json_safe(&trailer, "rhat");
    assert_diagnostic_map_is_json_safe(&trailer, "ess");

    std::fs::write(&fit_path, fit).expect("fit written");
    let diagnose = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "diagnose",
            "--fit",
            fit_path.to_str().expect("fit path utf-8"),
        ])
        .output()
        .expect("bayesite diagnose runs");
    assert!(
        diagnose.status.success(),
        "diagnose stderr: {}",
        String::from_utf8_lossy(&diagnose.stderr)
    );
    assert!(diagnose.stderr.is_empty());
    let report = json::parse(&String::from_utf8(diagnose.stdout).expect("diagnose stdout utf-8"))
        .expect("diagnose report parses");
    assert_diagnostic_map_is_json_safe(&report, "rhat");
    assert_diagnostic_map_is_json_safe(&report, "ess");
}

#[test]
fn short_recover_and_sbc_reports_do_not_fail_on_unavailable_diagnostics() {
    let (model_path, recover_path, sbc_path) = write_linear_regression_workflow_inputs();

    let recover = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().expect("model path utf-8"),
            "--scenario",
            recover_path.to_str().expect("recover path utf-8"),
        ])
        .output()
        .expect("bayesite recover runs");
    assert!(
        recover.status.success(),
        "recover stderr: {}",
        String::from_utf8_lossy(&recover.stderr)
    );
    assert!(recover.stderr.is_empty());
    let recover_report =
        json::parse(&String::from_utf8(recover.stdout).expect("recover stdout utf-8"))
            .expect("recover report parses");
    assert_nested_diagnostics_are_json_safe(&recover_report);

    let sbc = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().expect("model path utf-8"),
            "--scenario",
            sbc_path.to_str().expect("sbc path utf-8"),
        ])
        .output()
        .expect("bayesite sbc runs");
    assert!(
        sbc.status.success(),
        "sbc stderr: {}",
        String::from_utf8_lossy(&sbc.stderr)
    );
    assert!(sbc.stderr.is_empty());
    let sbc_report = json::parse(&String::from_utf8(sbc.stdout).expect("sbc stdout utf-8"))
        .expect("sbc report parses");
    assert_nested_diagnostics_are_json_safe(&sbc_report);
}
