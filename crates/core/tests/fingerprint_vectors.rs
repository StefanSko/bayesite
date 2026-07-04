//! Cross-language test vectors for `model_data_fingerprint`.
//!
//! `tests/golden_ir/fingerprints.json` is vendored from bayeswire
//! (`scripts/vendor_bayeswire.py`) and records `sha256:`-prefixed fingerprint
//! strings computed by the Python reference implementation for every model
//! in the golden corpus, over its matching `tests/golden_ir/data/<name>.json`
//! document. Recomputing them with the Rust SHA-256 must reproduce the exact
//! bytes: any divergence in the hash or the framing fails here.

use bayesite_core::fingerprint::model_data_fingerprint;
use bayesite_core::json::{self, Value};

fn fingerprints_document() -> Value {
    let path = format!(
        "{}/../../tests/golden_ir/fingerprints.json",
        env!("CARGO_MANIFEST_DIR")
    );
    json::parse(&std::fs::read_to_string(path).expect("fingerprints.json readable"))
        .expect("fingerprints.json parses")
}

fn data_dir_names() -> Vec<String> {
    let dir = format!("{}/../../tests/golden_ir/data", env!("CARGO_MANIFEST_DIR"));
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("golden_ir/data directory readable")
        .filter_map(|entry| {
            let path = entry
                .expect("golden_ir/data directory entry readable")
                .path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                return None;
            }
            Some(
                path.file_stem()
                    .expect("data fixture has a file stem")
                    .to_str()
                    .expect("data fixture path is UTF-8")
                    .to_string(),
            )
        })
        .collect();
    names.sort();
    names
}

#[test]
fn fingerprints_json_has_at_least_six_entries() {
    let Value::Object(entries) = fingerprints_document() else {
        panic!("fingerprints.json must be a JSON object");
    };
    assert!(
        entries.len() >= 6,
        "expected at least 6 fingerprint entries, got {}",
        entries.len()
    );
}

#[test]
fn every_data_fixture_has_a_fingerprint_entry() {
    let Value::Object(entries) = fingerprints_document() else {
        panic!("fingerprints.json must be a JSON object");
    };
    let recorded: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();
    for name in data_dir_names() {
        assert!(
            recorded.contains(&name),
            "tests/golden_ir/data/{name}.json has no matching entry in fingerprints.json"
        );
    }
}

#[test]
fn recomputed_fingerprints_match_the_python_reference_vectors() {
    let Value::Object(entries) = fingerprints_document() else {
        panic!("fingerprints.json must be a JSON object");
    };
    assert!(!entries.is_empty(), "fingerprints.json must not be empty");

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    for (name, expected) in &entries {
        let expected = expected
            .as_str()
            .unwrap_or_else(|| panic!("{name}: expected fingerprint must be a string"));

        let model_path = format!("{manifest_dir}/../../tests/golden_ir/{name}.json");
        let data_path = format!("{manifest_dir}/../../tests/golden_ir/data/{name}.json");

        let model_text = std::fs::read_to_string(&model_path)
            .unwrap_or_else(|e| panic!("{name}: cannot read {model_path}: {e}"));
        let data_text = std::fs::read_to_string(&data_path)
            .unwrap_or_else(|e| panic!("{name}: cannot read {data_path}: {e}"));

        let got = model_data_fingerprint(&model_text, &data_text);
        assert_eq!(got, expected, "{name}: fingerprint mismatch");
    }
}
