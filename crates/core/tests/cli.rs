//! End-to-end checks of the `bayesite` subprocess protocol.

use std::process::Command;

use bayesite_core::json::{self, Value};

fn fixture_text(name: &str) -> String {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

/// Split a fixture into model/data documents on disk; returns their paths.
fn write_fixture_inputs(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let doc = json::parse(&fixture_text(name)).unwrap();
    let dir = std::env::temp_dir().join(format!("bayesite-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    std::fs::write(&data_path, json::write(doc.get("data").unwrap()).unwrap()).unwrap();
    (model_path, data_path)
}

#[test]
fn samples_linear_regression_over_the_subprocess_protocol() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "7",
            "--chains",
            "2",
            "--warmup",
            "300",
            "--draws",
            "200",
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    // header + 2 chains x 200 draws + trailer
    assert_eq!(lines.len(), 1 + 2 * 200 + 1);

    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    let packing: Vec<&str> = header
        .get("packing")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(packing, ["alpha", "beta", "sigma"]);

    let first_draw = json::parse(lines[1]).unwrap();
    assert_eq!(first_draw.get("chain").and_then(Value::as_i64), Some(0));
    let sigma = first_draw
        .get("values")
        .and_then(|v| v.get("sigma"))
        .and_then(Value::as_f64)
        .unwrap();
    assert!(sigma > 0.0, "constrained sigma must be positive");

    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    let trailer = trailer.get("trailer").expect("trailer object");
    let chain_stats = trailer.get("chains").and_then(Value::as_array).unwrap();
    assert_eq!(chain_stats.len(), 2);
    for stats in chain_stats {
        assert!(stats.get("step_size").and_then(Value::as_f64).unwrap() > 0.0);
    }
    // The posterior is well-behaved: both chains should agree.
    for (_, value) in match trailer.get("rhat").unwrap() {
        Value::Object(entries) => entries.iter(),
        _ => panic!("rhat must be an object"),
    } {
        let rhat = value.as_f64().unwrap();
        assert!(rhat < 1.05, "rhat {rhat}");
    }
}

#[test]
fn reports_errors_as_json_on_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            "/nonexistent.json",
            "--data",
            "/nonexistent.json",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let payload = json::parse(stderr.trim()).expect("stderr is a JSON object");
    assert!(payload.get("error").and_then(Value::as_str).is_some());
    assert!(payload.get("message").and_then(Value::as_str).is_some());
}

#[test]
fn unknown_tag_failure_names_the_tag() {
    let dir = std::env::temp_dir().join(format!("bayesite-test-unknown-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    let model = fixture_text("linear_regression").replace("\"Normal\"", "\"FancyDist\"");
    let doc = json::parse(&model).unwrap();
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    std::fs::write(&data_path, json::write(doc.get("data").unwrap()).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("UnknownNodeTag")
    );
    assert!(payload
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("FancyDist"));
}
