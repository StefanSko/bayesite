//! Native tests of the wasm-boundary request handler (the pure seam the
//! unsafe ABI shims delegate to).

use bayesite_core::json::{self, Value};
use bayesite_core::protocol::handle_request;

fn fixture_text(name: &str) -> String {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

#[test]
fn sample_command_returns_single_chain_ndjson() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 200, "num_draws": 100}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(5)),
        ("chain_id".to_string(), Value::Int(3)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 1 + 100 + 1);
    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    let first = json::parse(lines[1]).unwrap();
    assert_eq!(first.get("chain").and_then(Value::as_i64), Some(3));
    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    assert!(trailer.get("trailer").is_some());
}

#[test]
fn diagnostics_command_returns_rhat_and_ess() {
    let request = r#"{"command": "diagnostics",
        "series": [[0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0, -0.2],
                   [0.2, 0.1, -0.4, 0.3, 0.0, -0.1, 0.2, 0.1]]}"#;
    let response = json::parse(&handle_request(request)).unwrap();
    assert!(response.get("rhat").and_then(Value::as_f64).unwrap() > 0.5);
    assert!(response.get("ess").and_then(Value::as_f64).unwrap() > 1.0);
}

#[test]
fn malformed_requests_return_json_errors() {
    for request in ["not json", "{}", r#"{"command": "sample"}"#] {
        let response = json::parse(&handle_request(request)).unwrap();
        assert!(
            response.get("error").and_then(Value::as_str).is_some(),
            "request {request:?} should fail with a typed error"
        );
    }
}
