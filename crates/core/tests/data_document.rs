//! Canonical data-document parsing: the `bayescycle.data.json.v1` wrapper
//! and the typed dtype vocabulary shared across the toolchain.

use bayesite_core::error::ErrorKind;
use bayesite_core::json;
use bayesite_core::model::data_from_json;

const BARE_DOC: &str = r#"{
  "x": {"dtype": "float64", "shape": [3], "values": [-1.0, 0.0, 1.0]},
  "n": {"dtype": "int64", "shape": [], "values": [8]}
}"#;

const WRAPPED_DOC: &str = r#"{
  "format": "bayescycle.data.json.v1",
  "variables": {
    "x": {"dtype": "float64", "shape": [3], "values": [-1.0, 0.0, 1.0]},
    "n": {"dtype": "int64", "shape": [], "values": [8]}
  }
}"#;

#[test]
fn wrapped_document_parses_identically_to_the_bare_map() {
    let bare = data_from_json(&json::parse(BARE_DOC).unwrap()).unwrap();
    let wrapped = data_from_json(&json::parse(WRAPPED_DOC).unwrap()).unwrap();

    assert_eq!(bare.len(), wrapped.len());
    for ((bare_name, bare_value), (wrapped_name, wrapped_value)) in bare.iter().zip(&wrapped) {
        assert_eq!(bare_name, wrapped_name);
        assert_eq!(bare_value.shape, wrapped_value.shape);
        assert_eq!(bare_value.values, wrapped_value.values);
        assert_eq!(bare_value.integer, wrapped_value.integer);
    }
}

#[test]
fn unknown_format_fails_explicitly_naming_the_format() {
    let doc =
        json::parse(r#"{"format": "bayescycle.data.json.v2", "variables": {"x": [1.0]}}"#).unwrap();
    let err = data_from_json(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("bayescycle.data.json.v2"),
        "message must name the unsupported format: {}",
        err.message
    );
}

#[test]
fn non_string_format_field_fails_explicitly() {
    // `format` is reserved at the top level; a data variable may not use it.
    let doc = json::parse(r#"{"format": {"dtype": "int64", "shape": [], "values": [1]}}"#).unwrap();
    let err = data_from_json(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(err.message.contains("format"), "message: {}", err.message);
}

#[test]
fn wrapped_document_without_variables_fails() {
    let doc = json::parse(r#"{"format": "bayescycle.data.json.v1"}"#).unwrap();
    let err = data_from_json(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("variables"),
        "message: {}",
        err.message
    );
}

#[test]
fn wrapped_document_with_extra_top_level_fields_fails() {
    let doc = json::parse(r#"{"format": "bayescycle.data.json.v1", "variables": {}, "extra": 1}"#)
        .unwrap();
    let err = data_from_json(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(err.message.contains("extra"), "message: {}", err.message);
}

#[test]
fn bare_map_without_format_key_still_parses_variables_named_variables() {
    // Only the reserved `format` key discriminates; `variables` alone is data.
    let doc = json::parse(r#"{"variables": [1.0, 2.0]}"#).unwrap();
    let parsed = data_from_json(&doc).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].0, "variables");
    assert_eq!(parsed[0].1.values, vec![1.0, 2.0]);
}

#[test]
fn bool_dtype_parses_as_integer_with_json_booleans() {
    let doc =
        json::parse(r#"{"flag": {"dtype": "bool", "shape": [3], "values": [true, false, true]}}"#)
            .unwrap();
    let parsed = data_from_json(&doc).unwrap();
    assert_eq!(parsed.len(), 1);
    let (name, value) = &parsed[0];
    assert_eq!(name, "flag");
    assert!(value.integer, "bool data must bind as integer-valued");
    assert_eq!(value.shape, vec![3]);
    assert_eq!(value.values, vec![1.0, 0.0, 1.0]);
}

#[test]
fn bool_dtype_rejects_non_boolean_values() {
    let doc =
        json::parse(r#"{"flag": {"dtype": "bool", "shape": [2], "values": [1, 0]}}"#).unwrap();
    let err = data_from_json(&doc).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("bool"), "message: {}", err.message);
}

#[test]
fn wrapped_bool_variables_parse_through_the_wrapper() {
    let doc = json::parse(
        r#"{
  "format": "bayescycle.data.json.v1",
  "variables": {"flag": {"dtype": "bool", "shape": [2], "values": [false, true]}}
}"#,
    )
    .unwrap();
    let parsed = data_from_json(&doc).unwrap();
    assert_eq!(parsed[0].1.values, vec![0.0, 1.0]);
    assert!(parsed[0].1.integer);
}
