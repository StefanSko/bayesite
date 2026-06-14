//! Typed error taxonomy, mirroring the Python IR errors where applicable.
//!
//! Messages state what to change, not only what is wrong; they double as
//! repair instructions for agents producing IR documents.

use std::fmt;

/// Error kind, stable across releases; the CLI emits it as the `error` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorKind {
    /// The bytes are not a syntactically valid strict-JSON document.
    MalformedJson,
    /// The document structure violates the IR v1 format.
    MalformedDocument,
    /// The `jaxstanv5_ir` version is missing or unknown.
    UnsupportedIRVersion,
    /// A `"node"` tag is not in the core-profile registry.
    UnknownNodeTag,
    /// Bound data does not match the shapes the model requires.
    DataShapeMismatch,
    /// The log density evaluated to a non-finite value where finiteness is required.
    NonFiniteDensity,
    /// Sampler settings are invalid (e.g. zero draws, bad target accept).
    InvalidSettings,
}

impl ErrorKind {
    /// Stable machine-readable name.
    pub fn name(self) -> &'static str {
        match self {
            ErrorKind::MalformedJson => "MalformedJson",
            ErrorKind::MalformedDocument => "MalformedDocument",
            ErrorKind::UnsupportedIRVersion => "UnsupportedIRVersion",
            ErrorKind::UnknownNodeTag => "UnknownNodeTag",
            ErrorKind::DataShapeMismatch => "DataShapeMismatch",
            ErrorKind::NonFiniteDensity => "NonFiniteDensity",
            ErrorKind::InvalidSettings => "InvalidSettings",
        }
    }
}

/// A typed error with a repair-instruction message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub message: String,
}

impl Error {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.name(), self.message)
    }
}

impl std::error::Error for Error {}
