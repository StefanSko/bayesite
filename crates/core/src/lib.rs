//! Audited sampling core for Bayesite.
//!
//! Parses the `bayeswire_ir` v1 wire format (`docs/ir-format-v1.md`), evaluates
//! the model log density and its gradient with built-in reverse-mode AD, and
//! samples with multinomial NUTS using Stan-style warmup adaptation.
//!
//! The library is a pure function of its arguments: no threads, no
//! filesystem, no clock, no OS entropy. Seeds are explicit arguments and
//! parallelism belongs to callers (CLI threads, web workers).

#![deny(unsafe_code)]

pub mod adapt;
mod ancestral;
pub mod artifact;
pub mod density;
pub mod diagnostics;
pub mod error;
pub mod fingerprint;
pub mod generation;
pub mod ir;
pub mod json;
pub mod linalg;
pub mod model;
pub mod nuts;
pub mod predictive;
pub mod protocol;
pub mod rng;
pub mod sampler;
pub mod special;
pub mod tape;
pub mod tensor;
#[cfg(target_arch = "wasm32")]
#[allow(unsafe_code)]
pub mod wasm_abi;
pub mod workflow;

pub use error::Error;
