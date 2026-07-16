//! Model/data fingerprints per the bayeswire spec
//! `model-data-fingerprint-v1.md` (vendored at
//! `docs/model-data-fingerprint-v1.md`).
//!
//! The fingerprint is sha256 over
//! `b"bayescycle-model-data-v1\n" + model + b"\n" + data`, rendered as
//! `sha256:` followed by 64 lowercase hex digits. Producers hash the exact
//! bytes they write; verifiers hash the exact bytes they receive and never
//! reserialize before hashing.

use sha2::{Digest, Sha256};
use std::fmt::Write as FmtWrite;

fn render_digest(digest: impl AsRef<[u8]>) -> String {
    let mut out = String::from("sha256:");
    for byte in digest.as_ref() {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

/// Return a lowercase prefixed SHA-256 digest for exact received bytes.
pub fn sha256_bytes(input: &[u8]) -> String {
    render_digest(Sha256::digest(input))
}

/// Fingerprint a model/data pair per the bayeswire spec
/// `model-data-fingerprint-v1.md`: sha256 over
/// `b"bayescycle-model-data-v1\n" + model + b"\n" + data`, rendered as
/// `sha256:` followed by 64 lowercase hex digits.
///
/// Producers hash the exact bytes they write; verifiers hash the exact
/// bytes they receive and never reserialize before hashing.
pub fn model_data_fingerprint(model_text: &str, data_text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"bayescycle-model-data-v1\n");
    hasher.update(model_text.as_bytes());
    hasher.update(b"\n");
    hasher.update(data_text.as_bytes());
    render_digest(hasher.finalize())
}
