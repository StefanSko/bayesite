//! Bounded functional dataset generation and paired artifacts.

use crate::error::{Error, ErrorKind};
use crate::ir::ModelMeta;
use crate::json::Value;

#[derive(Clone, Debug)]
pub struct AuthoredProvenance {
    pub claimed_source_model_hash: String,
    pub claimed_outcome_model_hash: String,
}

#[derive(Clone, Debug)]
pub enum GenerationSource {
    Fixed {
        parameters: Value,
        parameters_hash: String,
    },
    ModelPrior {
        model_hash: String,
        authored_provenance: Option<AuthoredProvenance>,
    },
    Posterior {
        fit_ndjson: String,
        fit_data: Value,
        fit_hash: String,
        fit_model_hash: String,
        fit_data_hash: String,
        expected_model_data_fingerprint: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    pub meta: ModelMeta,
    pub design: Value,
    pub source: GenerationSource,
    pub count: usize,
    pub seed: u64,
    pub generation_model_hash: String,
    pub design_hash: String,
}

pub fn generated_datasets_ndjson_lines(_request: GenerationRequest) -> Result<Vec<String>, Error> {
    Err(Error::new(
        ErrorKind::InvalidSettings,
        "functional generation is not implemented",
    ))
}

pub fn sha256_bytes(_bytes: &[u8]) -> String {
    panic!("functional generation hashing is not implemented")
}
