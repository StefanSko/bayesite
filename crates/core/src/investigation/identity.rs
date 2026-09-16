//! Domain-separated identities for investigation artifacts, recipes, and snapshots.

use sha2::{Digest as _, Sha256};

use crate::error::{Error, ErrorKind};

pub const RECIPE_DOMAIN: &[u8] = b"bayesite-investigation-recipe-v0\0";
pub const SNAPSHOT_DOMAIN: &[u8] = b"bayesite-investigation-snapshot-v0\0";

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

/// A validated, lowercase SHA-256 digest without a scheme prefix.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Digest(pub String);

impl Digest {
    pub fn parse(value: &str, context: &str) -> Result<Self, Error> {
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(malformed(format!(
                "{context} must be exactly 64 lowercase hexadecimal characters"
            )));
        }
        Ok(Self(value.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn prefixed(&self) -> String {
        format!("sha256:{}", self.0)
    }
}

/// SHA-256 of exact artifact bytes. No parsing or reserialization occurs.
pub fn artifact_digest(bytes: &[u8]) -> Digest {
    Digest(hex(&Sha256::digest(bytes)))
}

/// Snapshot identity over the exact received manifest bytes and a fixed domain.
pub fn snapshot_digest(manifest_bytes: &[u8]) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(SNAPSHOT_DOMAIN);
    hasher.update(manifest_bytes);
    Digest(hex(&hasher.finalize()))
}

/// Length-prefix one identity field with an unsigned 64-bit big-endian length.
pub fn frame(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_be_bytes());
    hasher.update(bytes);
}

/// Hash an already-framed recipe identity body.
pub fn recipe_digest(update: impl FnOnce(&mut Sha256)) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(RECIPE_DOMAIN);
    update(&mut hasher);
    Digest(hex(&hasher.finalize()))
}
