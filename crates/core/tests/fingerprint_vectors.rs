//! Independent SHA-256 and Bayeswire model/data fingerprint vectors.
//!
//! The corpus vectors are vendored from Bayeswire. The published SHA-256
//! vectors and fixed boundary inputs below characterize this API separately
//! from its implementation.

use std::collections::BTreeSet;
use std::path::Path;

use bayesite_core::fingerprint::{model_data_fingerprint, sha256_bytes};
use bayesite_core::json::{self, Value};

const PREFIX: &str = "sha256:";

fn fingerprints_document() -> Value {
    let path = format!(
        "{}/../../tests/golden_ir/fingerprints.json",
        env!("CARGO_MANIFEST_DIR")
    );
    json::parse(&std::fs::read_to_string(path).expect("fingerprints.json readable"))
        .expect("fingerprints.json parses")
}

fn json_stem_names(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("fixture directory readable")
        .map(|entry| entry.expect("fixture directory entry readable").path())
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
        .filter_map(|path| {
            path.file_stem()
                .and_then(|stem| stem.to_str())
                .map(ToOwned::to_owned)
        })
        .collect()
}

fn assert_digest(input: &[u8], expected_hex: &str) {
    let got = sha256_bytes(input);
    assert_eq!(got, format!("{PREFIX}{expected_hex}"));
    assert_eq!(got.len(), PREFIX.len() + 64);
    assert!(
        got[PREFIX.len()..]
            .bytes()
            .all(|byte| byte.is_ascii_digit()
                || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())),
        "digest must use lowercase hexadecimal: {got}"
    );
}

#[test]
fn published_sha256_vectors() {
    assert_digest(
        b"",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    assert_digest(
        b"abc",
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    assert_digest(
        b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
    assert_digest(
        &vec![b'a'; 1_000_000],
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
    );
}

#[test]
fn sha256_covers_binary_padding_and_large_inputs() {
    assert_digest(
        &[0, 255, 128, 1, 0, 10],
        "ea5b1675daacb1aad22f1515f2ede2a8953681841af651870459c858d0649ce3",
    );
    for (len, expected) in [
        (
            55,
            "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
        ),
        (
            56,
            "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
        ),
        (
            63,
            "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488",
        ),
        (
            64,
            "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
        ),
        (
            65,
            "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781",
        ),
    ] {
        assert_digest(
            &(0..len).map(|value| value as u8).collect::<Vec<_>>(),
            expected,
        );
    }
    assert_digest(
        &(0..4096)
            .map(|i| ((i * 37 + 11) % 256) as u8)
            .collect::<Vec<_>>(),
        "4e441a3533bb2c10cd5649981d395744213e09a336746b5a3458fee4057205ec",
    );
    assert_digest(
        &(0..65_536)
            .map(|i| ((i * 17 + 29) % 256) as u8)
            .collect::<Vec<_>>(),
        "35171a3b38d84c658143e78203683076d07beca32140221055c899b2dcf1cbc5",
    );
}

#[test]
fn framing_is_exact_and_whitespace_sensitive() {
    let model = "{ \"x\": 1 }";
    let data = "{\"data\": [1,2]}";
    assert_eq!(
        model_data_fingerprint(model, data),
        "sha256:5b1670ffaf6a6e3502c6cc5039c06b146cd884c3721867f2a355f05a6587df96"
    );
    assert_ne!(
        model_data_fingerprint("{\"x\":1}", data),
        model_data_fingerprint(model, data)
    );
    assert_ne!(
        model_data_fingerprint(model, "{\"data\":[1,2]}"),
        model_data_fingerprint(model, data)
    );
}

#[test]
fn corpus_names_and_fingerprints_match_exactly() {
    let Value::Object(entries) = fingerprints_document() else {
        panic!("fingerprints.json must be a JSON object");
    };
    let recorded: BTreeSet<String> = entries.iter().map(|(name, _)| name.clone()).collect();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden_ir");
    let mut models = json_stem_names(&root);
    models.remove("hashes");
    models.remove("fingerprints");
    let data = json_stem_names(&root.join("data"));
    assert_eq!(models, data, "model and data fixture names must match");
    assert_eq!(
        models, recorded,
        "fingerprint keys must match corpus fixtures"
    );

    for (name, expected) in entries {
        let expected = expected
            .as_str()
            .unwrap_or_else(|| panic!("{name}: expected fingerprint must be a string"));
        let model_path = root.join(format!("{name}.json"));
        let data_path = root.join("data").join(format!("{name}.json"));
        let model = std::fs::read_to_string(&model_path).unwrap_or_else(|error| {
            panic!("{name}: cannot read {}: {error}", model_path.display())
        });
        let data = std::fs::read_to_string(&data_path)
            .unwrap_or_else(|error| panic!("{name}: cannot read {}: {error}", data_path.display()));
        assert_eq!(
            model_data_fingerprint(&model, &data),
            expected,
            "{name}: fingerprint mismatch"
        );
    }
}
