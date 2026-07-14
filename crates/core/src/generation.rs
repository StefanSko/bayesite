//! Bounded functional dataset generation and paired artifacts.

use crate::artifact::V0_PROVISIONAL;
use crate::error::{Error, ErrorKind};
use crate::fingerprint;
use crate::ir::ModelMeta;
use crate::json::{self, Value};
use crate::model::{data_from_json, data_to_json, DataValue};
use crate::predictive::{
    fixed_generation_pairs, model_prior_generation_pairs, posterior_generation_pairs,
    GenerationLineage, GenerationPair,
};

const MAX_COUNT: usize = 1000;
const MAX_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const FORMAT: &str = V0_PROVISIONAL;
const ARTIFACT_KIND: &str = "generated_dataset_pairs";
const ARTIFACT_SCOPE: &str = "parameter_and_complete_dataset_joint_draws";
const DRAW_INDEX_BASE: &str = "zero_based_generation_order";
const WORKFLOW_PHASES: [&str; 6] = [
    "parse_json",
    "decode_ir",
    "bind_design",
    "draw_parameters",
    "simulate_outcomes",
    "emit_artifact",
];

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

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn validate_hash(value: &str, label: &str) -> Result<(), Error> {
    let hex = value.strip_prefix("sha256:").ok_or_else(|| {
        invalid(format!(
            "{label} must be sha256: followed by 64 lowercase hex digits"
        ))
    })?;
    if hex.len() != 64
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(invalid(format!(
            "{label} must be sha256: followed by 64 lowercase hex digits"
        )));
    }
    Ok(())
}

fn common_fields() -> Vec<(String, Value)> {
    vec![
        (
            "generated_datasets_format".to_string(),
            Value::Str(FORMAT.to_string()),
        ),
        (
            "artifact_kind".to_string(),
            Value::Str(ARTIFACT_KIND.to_string()),
        ),
        (
            "artifact_scope".to_string(),
            Value::Str(ARTIFACT_SCOPE.to_string()),
        ),
    ]
}

fn workflow_phases_value() -> Value {
    Value::Array(
        WORKFLOW_PHASES
            .iter()
            .map(|phase| Value::Str((*phase).to_string()))
            .collect(),
    )
}

fn canonical_variables(document: &Value, label: &str) -> Result<Vec<(String, Value)>, Error> {
    let Value::Object(fields) = document else {
        return Err(malformed(format!(
            "{label} must be a canonical data object"
        )));
    };
    if fields.len() != 2 || fields[0].0 != "format" || fields[1].0 != "variables" {
        return Err(malformed(format!(
            "{label} must contain exactly format then variables"
        )));
    }
    if fields[0].1.as_str() != Some("bayescycle.data.json.v1") {
        return Err(malformed(format!(
            "{label} format must be \"bayescycle.data.json.v1\""
        )));
    }
    let Value::Object(variables) = &fields[1].1 else {
        return Err(malformed(format!("{label} variables must be an object")));
    };
    Ok(variables.clone())
}

fn normalized_variables(document: &Value, label: &str) -> Result<Vec<(String, Value)>, Error> {
    if document.get("format").is_some() {
        canonical_variables(document, label)
    } else {
        let data = data_from_json(document)?;
        rendered_variables(&data, label)
    }
}

fn canonical_document(variables: Vec<(String, Value)>) -> Value {
    Value::Object(vec![
        (
            "format".to_string(),
            Value::Str("bayescycle.data.json.v1".to_string()),
        ),
        ("variables".to_string(), Value::Object(variables)),
    ])
}

fn rendered_variables(
    values: &[(String, DataValue)],
    context: &str,
) -> Result<Vec<(String, Value)>, Error> {
    let Value::Object(entries) = data_to_json(values, context)? else {
        unreachable!("data_to_json always returns an object")
    };
    Ok(entries)
}

fn pair_documents(
    pair: &GenerationPair,
    design_variables: &[(String, Value)],
) -> Result<(Value, Value), Error> {
    let parameters = canonical_document(rendered_variables(
        &pair.parameters,
        "generated parameters",
    )?);
    let mut dataset_variables = design_variables.to_vec();
    dataset_variables.extend(rendered_variables(&pair.outcomes, "generated dataset")?);
    Ok((parameters, canonical_document(dataset_variables)))
}

fn schema(document: &Value, label: &str) -> Result<Value, Error> {
    let variables = canonical_variables(document, label)?;
    variables
        .iter()
        .map(|(name, spec)| {
            let dtype = spec
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| malformed(format!("{label} variable {name} needs dtype")))?;
            let shape = spec
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed(format!("{label} variable {name} needs shape")))?;
            Ok(Value::Object(vec![
                ("name".to_string(), Value::Str(name.clone())),
                ("dtype".to_string(), Value::Str(dtype.to_string())),
                ("shape".to_string(), Value::Array(shape.to_vec())),
            ]))
        })
        .collect::<Result<Vec<_>, Error>>()
        .map(Value::Array)
}

fn source_descriptor(source: &GenerationSource) -> Result<Value, Error> {
    match source {
        GenerationSource::Fixed {
            parameters_hash, ..
        } => {
            validate_hash(parameters_hash, "parameters_hash")?;
            Ok(Value::Object(vec![
                ("kind".to_string(), Value::Str("fixed".to_string())),
                (
                    "parameters_hash".to_string(),
                    Value::Str(parameters_hash.clone()),
                ),
            ]))
        }
        GenerationSource::ModelPrior {
            model_hash,
            authored_provenance,
        } => {
            validate_hash(model_hash, "model_hash")?;
            let provenance = match authored_provenance {
                Some(provenance) => {
                    validate_hash(
                        &provenance.claimed_source_model_hash,
                        "claimed_source_model_hash",
                    )?;
                    validate_hash(
                        &provenance.claimed_outcome_model_hash,
                        "claimed_outcome_model_hash",
                    )?;
                    Value::Object(vec![
                        (
                            "claimed_source_model_hash".to_string(),
                            Value::Str(provenance.claimed_source_model_hash.clone()),
                        ),
                        (
                            "claimed_outcome_model_hash".to_string(),
                            Value::Str(provenance.claimed_outcome_model_hash.clone()),
                        ),
                    ])
                }
                None => Value::Null,
            };
            Ok(Value::Object(vec![
                ("kind".to_string(), Value::Str("model-prior".to_string())),
                ("model_hash".to_string(), Value::Str(model_hash.clone())),
                ("authored_provenance".to_string(), provenance),
            ]))
        }
        GenerationSource::Posterior {
            fit_hash,
            fit_model_hash,
            fit_data_hash,
            ..
        } => {
            validate_hash(fit_hash, "fit_hash")?;
            validate_hash(fit_model_hash, "fit_model_hash")?;
            validate_hash(fit_data_hash, "fit_data_hash")?;
            Ok(Value::Object(vec![
                ("kind".to_string(), Value::Str("posterior".to_string())),
                ("fit_hash".to_string(), Value::Str(fit_hash.clone())),
                (
                    "fit_model_hash".to_string(),
                    Value::Str(fit_model_hash.clone()),
                ),
                (
                    "fit_data_hash".to_string(),
                    Value::Str(fit_data_hash.clone()),
                ),
            ]))
        }
    }
}

fn lineage_value(lineage: &GenerationLineage) -> Value {
    match lineage {
        GenerationLineage::Fixed => {
            Value::Object(vec![("kind".to_string(), Value::Str("fixed".to_string()))])
        }
        GenerationLineage::ModelPrior { source_draw_index } => Value::Object(vec![
            ("kind".to_string(), Value::Str("model-prior".to_string())),
            (
                "source_draw_index".to_string(),
                Value::Int(*source_draw_index as i64),
            ),
        ]),
        GenerationLineage::Posterior {
            source_draw_index,
            chain,
            draw,
        } => Value::Object(vec![
            ("kind".to_string(), Value::Str("posterior".to_string())),
            (
                "source_draw_index".to_string(),
                Value::Int(*source_draw_index as i64),
            ),
            ("chain".to_string(), Value::Int(*chain)),
            ("draw".to_string(), Value::Int(*draw)),
        ]),
    }
}

fn checked_lines(values: Vec<Value>) -> Result<Vec<String>, Error> {
    let mut total = 0usize;
    let mut lines = Vec::with_capacity(values.len());
    for value in values {
        let line = json::write(&value)?;
        let line_bytes = line
            .len()
            .checked_add(1)
            .ok_or_else(|| invalid("generated-dataset artifact line size overflowed this build"))?;
        if line_bytes > MAX_LINE_BYTES {
            return Err(invalid(format!(
                "generated-dataset artifact line exceeds {MAX_LINE_BYTES} bytes"
            )));
        }
        total = total
            .checked_add(line_bytes)
            .ok_or_else(|| invalid("generated-dataset artifact size overflowed this build"))?;
        if total > MAX_ARTIFACT_BYTES {
            return Err(invalid(format!(
                "generated-dataset artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
            )));
        }
        lines.push(line);
    }
    Ok(lines)
}

pub fn generated_datasets_ndjson_lines(request: GenerationRequest) -> Result<Vec<String>, Error> {
    if !(1..=MAX_COUNT).contains(&request.count) {
        return Err(invalid(format!(
            "generate count must be in 1..={MAX_COUNT}"
        )));
    }
    if request.seed > MAX_SAFE_INTEGER {
        return Err(invalid(format!(
            "generate seed must be in 0..={MAX_SAFE_INTEGER}"
        )));
    }
    validate_hash(&request.generation_model_hash, "generation_model_hash")?;
    validate_hash(&request.design_hash, "design_hash")?;
    let design_variables = normalized_variables(&request.design, "generate design")?;
    let design_data = data_from_json(&request.design)?;
    let source = source_descriptor(&request.source)?;
    let pairs = match &request.source {
        GenerationSource::Fixed {
            parameters,
            parameters_hash: _,
        } => {
            canonical_variables(parameters, "generate fixed parameters")?;
            let parameter_data = data_from_json(parameters)?;
            fixed_generation_pairs(
                request.meta.clone(),
                design_data.clone(),
                parameter_data,
                request.count,
                request.seed,
            )?
        }
        GenerationSource::ModelPrior { model_hash, .. } => {
            if model_hash != &request.generation_model_hash {
                return Err(invalid(
                    "model-prior model hash must equal generation model hash",
                ));
            }
            model_prior_generation_pairs(
                request.meta.clone(),
                design_data.clone(),
                request.count,
                request.seed,
            )?
        }
        GenerationSource::Posterior {
            fit_ndjson,
            fit_data,
            fit_hash,
            fit_model_hash,
            expected_model_data_fingerprint,
            ..
        } => {
            if fit_model_hash != &request.generation_model_hash {
                return Err(invalid(
                    "posterior fit model hash must equal generation model hash",
                ));
            }
            if *fit_hash != sha256_bytes(fit_ndjson.as_bytes()) {
                return Err(invalid(
                    "posterior fit hash must match the exact posterior bytes",
                ));
            }
            posterior_generation_pairs(
                request.meta.clone(),
                design_data.clone(),
                data_from_json(fit_data)?,
                fit_ndjson,
                expected_model_data_fingerprint.as_deref(),
                request.count,
                request.seed,
            )?
        }
    };
    if pairs.len() != request.count {
        return Err(invalid("generate core returned the wrong pair count"));
    }
    let mut pair_documents_values = Vec::with_capacity(pairs.len());
    for pair in &pairs {
        pair_documents_values.push(pair_documents(pair, &design_variables)?);
    }
    let Some((first_parameters, first_dataset)) = pair_documents_values.first() else {
        unreachable!("count was validated positive")
    };
    let parameter_schema = schema(first_parameters, "generated parameters")?;
    let dataset_schema = schema(first_dataset, "generated dataset")?;
    for (parameters, dataset) in &pair_documents_values[1..] {
        if schema(parameters, "generated parameters")? != parameter_schema
            || schema(dataset, "generated dataset")? != dataset_schema
        {
            return Err(invalid(
                "generated parameter or dataset schema changed across draws",
            ));
        }
    }

    let mut header = common_fields();
    header.extend([
        ("workflow_phases".to_string(), workflow_phases_value()),
        (
            "generation_model_hash".to_string(),
            Value::Str(request.generation_model_hash.clone()),
        ),
        (
            "design_hash".to_string(),
            Value::Str(request.design_hash.clone()),
        ),
        ("parameter_source".to_string(), source.clone()),
        ("count".to_string(), Value::Int(request.count as i64)),
        ("seed".to_string(), Value::Int(request.seed as i64)),
        (
            "draw_index_base".to_string(),
            Value::Str(DRAW_INDEX_BASE.to_string()),
        ),
        ("parameter_schema".to_string(), parameter_schema),
        ("dataset_schema".to_string(), dataset_schema),
    ]);
    let mut values = Vec::with_capacity(request.count + 2);
    values.push(Value::Object(header));
    for (draw_index, ((parameters, dataset), pair)) in
        pair_documents_values.into_iter().zip(&pairs).enumerate()
    {
        let mut draw = common_fields();
        draw.extend([
            ("draw_index".to_string(), Value::Int(draw_index as i64)),
            ("draw_count".to_string(), Value::Int(request.count as i64)),
            ("parameters".to_string(), parameters),
            ("dataset".to_string(), dataset),
            ("source_lineage".to_string(), lineage_value(&pair.lineage)),
        ]);
        values.push(Value::Object(draw));
    }
    let mut trailer = common_fields();
    trailer.extend([
        ("workflow_phases".to_string(), workflow_phases_value()),
        (
            "generation_model_hash".to_string(),
            Value::Str(request.generation_model_hash),
        ),
        ("design_hash".to_string(), Value::Str(request.design_hash)),
        ("parameter_source".to_string(), source),
        ("count".to_string(), Value::Int(request.count as i64)),
        ("seed".to_string(), Value::Int(request.seed as i64)),
        ("draw_count".to_string(), Value::Int(request.count as i64)),
        ("complete".to_string(), Value::Bool(true)),
    ]);
    values.push(Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(trailer),
    )]));
    checked_lines(values)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    fingerprint::sha256_bytes(bytes)
}
