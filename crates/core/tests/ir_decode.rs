//! IR v1 decoder conformance against the golden corpus and error taxonomy.

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{decode_model, Constraint, Size};
use bayesite_core::json;

fn fixture(name: &str) -> json::Value {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {path}: {e}"));
    json::parse(&text).expect("fixture JSON parses")
}

const ALL_FIXTURES: [&str; 6] = [
    "bounded_rates",
    "eight_schools_non_centered",
    "linear_regression",
    "ordinal_regression",
    "partially_observed_mvn",
    "varying_intercepts_poisson",
];

#[test]
fn decodes_every_golden_fixture_ir() {
    for name in ALL_FIXTURES {
        let doc = fixture(name);
        let ir = doc.get("ir").expect("fixture has ir");
        decode_model(ir).unwrap_or_else(|e| panic!("decoding {name} failed: {e}"));
    }
}

#[test]
fn packing_order_follows_free_values() {
    let doc = fixture("eight_schools_non_centered");
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let free = meta.resolved_free_values();
    let names: Vec<&str> = free.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["mu", "tau", "z"]);
    assert_eq!(free[0].1.constraint, None);
    assert_eq!(free[1].1.constraint, Some(Constraint::Positive));
    assert_eq!(free[2].1.size, Size::Data("n_schools".to_string()));
}

#[test]
fn decodes_constraints() {
    let doc = fixture("bounded_rates");
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let free = meta.resolved_free_values();
    assert_eq!(free[0].1.constraint, Some(Constraint::UnitInterval));
    assert_eq!(
        free[1].1.constraint,
        Some(Constraint::Interval {
            lower: -1.0,
            upper: 3.0
        })
    );
}

#[test]
fn stochastic_sites_are_in_document_order() {
    let doc = fixture("linear_regression");
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let sites = meta.resolved_stochastic_sites();
    let names: Vec<&str> = sites.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["alpha", "beta", "sigma", "y"]);
}

#[test]
fn missing_version_is_unsupported() {
    let doc = json::parse(r#"{"model": {"node": "ModelMeta"}}"#).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedIRVersion);
}

#[test]
fn unknown_version_is_unsupported() {
    let doc = json::parse(r#"{"jaxstanv5_ir": 2, "model": {"node": "ModelMeta"}}"#).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnsupportedIRVersion);
}

#[test]
fn unexpected_envelope_field_is_malformed() {
    let doc =
        json::parse(r#"{"jaxstanv5_ir": 1, "producer": "test", "model": {"node": "ModelMeta"}}"#)
            .unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(err.message.contains("producer"), "message: {}", err.message);
}

#[test]
fn duplicate_envelope_version_field_is_malformed() {
    let text = minimal_model(NORMAL_PARAM).replacen(
        r#""jaxstanv5_ir": 1"#,
        r#""jaxstanv5_ir": 1, "jaxstanv5_ir": 1"#,
        1,
    );
    let doc = json::parse(&text).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("duplicate \"jaxstanv5_ir\""),
        "message: {}",
        err.message
    );
}

fn minimal_model(params_entry: &str) -> String {
    format!(
        r#"{{"jaxstanv5_ir": 1, "model": {{"node": "ModelMeta",
            "params": [{params_entry}],
            "data": [], "observed_nodes": [], "expressions": [],
            "free_values": [], "stochastic_sites": []}}}}"#
    )
}

const NORMAL_PARAM: &str = r#"{"name": "x", "value": {"node": "ResolvedParam",
    "distribution": {"node": "Normal",
        "loc": {"node": "ConstNode", "value": 0.0},
        "scale": {"node": "ConstNode", "value": 1.0}},
    "constraint": null, "size": null}}"#;

#[test]
fn decodes_minimal_model_with_legacy_fallback() {
    let doc = json::parse(&minimal_model(NORMAL_PARAM)).unwrap();
    let meta = decode_model(&doc).unwrap();
    // Empty free_values falls back to params, as in the Python decoder.
    let free = meta.resolved_free_values();
    assert_eq!(free.len(), 1);
    assert_eq!(free[0].0, "x");
    let sites = meta.resolved_stochastic_sites();
    assert_eq!(sites.len(), 1);
    assert_eq!(sites[0].name, "x");
}

#[test]
fn unknown_node_tag_is_typed() {
    let entry = NORMAL_PARAM.replace("\"Normal\"", "\"MyCustomDist\"");
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::UnknownNodeTag);
    assert!(
        err.message.contains("MyCustomDist"),
        "message: {}",
        err.message
    );
}

#[test]
fn missing_field_is_malformed() {
    let entry = NORMAL_PARAM.replace(r#""constraint": null, "#, "");
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

#[test]
fn unexpected_field_is_malformed() {
    let entry = NORMAL_PARAM.replace(r#""constraint": null"#, r#""constraint": null, "bogus": 1"#);
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

#[test]
fn duplicate_node_tag_field_is_malformed() {
    let entry = NORMAL_PARAM.replace(
        r#""distribution": {"node": "Normal","#,
        r#""distribution": {"node": "Normal", "node": "HalfNormal","#,
    );
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("duplicate \"node\""),
        "message: {}",
        err.message
    );
}

#[test]
fn duplicate_map_entry_names_are_malformed() {
    let two = format!("{NORMAL_PARAM}, {NORMAL_PARAM}");
    let doc = json::parse(&minimal_model(&two)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

#[test]
fn map_entry_without_name_value_is_malformed() {
    let doc = json::parse(&minimal_model(r#"{"name": "x"}"#)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

/// A left-deep BinOp chain of `nodes` BinOp nodes as the Normal `loc`.
fn deep_binop_param(nodes: usize) -> String {
    let mut expr = String::from("0.0");
    for _ in 0..nodes {
        expr = format!(r#"{{"node": "BinOp", "op": "+", "left": {expr}, "right": 1.0}}"#);
    }
    NORMAL_PARAM.replace(r#"{"node": "ConstNode", "value": 0.0}"#, &expr)
}

#[test]
fn expression_nesting_at_the_depth_limit_decodes() {
    let entry = deep_binop_param(bayesite_core::ir::MAX_EXPR_DEPTH - 1);
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    decode_model(&doc).unwrap();
}

#[test]
fn expression_nesting_beyond_the_depth_limit_is_a_typed_error_not_a_crash() {
    let entry = deep_binop_param(bayesite_core::ir::MAX_EXPR_DEPTH);
    let doc = json::parse(&minimal_model(&entry)).unwrap();
    let err = decode_model(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(err.message.contains("nesting"), "message: {}", err.message);
}
