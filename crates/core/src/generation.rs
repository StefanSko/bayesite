//! Bounded functional dataset generation and paired artifacts.

use crate::artifact::V0_PROVISIONAL;
use crate::error::{Error, ErrorKind};
use crate::fingerprint;
use crate::ir::decode_model;
use crate::json::{self, Value};
use crate::model::{data_from_json, data_to_json, DataValue};
use crate::predictive::{
    fixed_generation_pairs, model_prior_generation_pairs, posterior_generation_pairs,
    GenerationLineage, GenerationPair, PosteriorGenerationSource,
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
        parameters_document: String,
        parameters_hash: String,
    },
    ModelPrior {
        model_hash: String,
        authored_provenance: Option<AuthoredProvenance>,
    },
    Posterior {
        fit_ndjson: String,
        fit_data_document: String,
        fit_hash: String,
        fit_model_hash: String,
        fit_data_hash: String,
        expected_model_data_fingerprint: Option<String>,
    },
}

#[derive(Clone, Debug)]
pub struct GenerationRequest {
    /// Exact JSON bytes (as UTF-8 text) whose hash is recorded as model lineage.
    pub model_document: String,
    /// Exact JSON bytes (as UTF-8 text) whose hash is recorded as design lineage.
    pub design_document: String,
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

fn unique_field<'a>(
    fields: &'a [(String, Value)],
    name: &str,
    label: &str,
) -> Result<&'a Value, Error> {
    let mut matches = fields.iter().filter(|(key, _)| key == name);
    let value = matches
        .next()
        .map(|(_, value)| value)
        .ok_or_else(|| malformed(format!("{label} needs {name}")))?;
    if matches.next().is_some() {
        return Err(malformed(format!("{label} has duplicate {name}")));
    }
    Ok(value)
}

fn canonical_variables(document: &Value, label: &str) -> Result<Vec<(String, Value)>, Error> {
    let Value::Object(fields) = document else {
        return Err(malformed(format!(
            "{label} must be a canonical data object"
        )));
    };
    if fields.len() != 2
        || fields
            .iter()
            .any(|(name, _)| name != "format" && name != "variables")
    {
        return Err(malformed(format!(
            "{label} must contain exactly format and variables"
        )));
    }
    if unique_field(fields, "format", label)?.as_str() != Some("bayescycle.data.json.v1") {
        return Err(malformed(format!(
            "{label} format must be \"bayescycle.data.json.v1\""
        )));
    }
    let Value::Object(variables) = unique_field(fields, "variables", label)? else {
        return Err(malformed(format!("{label} variables must be an object")));
    };
    let mut names = std::collections::HashSet::new();
    for (name, spec) in variables {
        if name.is_empty() || !names.insert(name) {
            return Err(malformed(format!(
                "{label} has an empty or duplicate variable name {name:?}"
            )));
        }
        let Value::Object(spec_fields) = spec else {
            return Err(malformed(format!(
                "{label} variable {name:?} must be a typed value object"
            )));
        };
        if spec_fields.len() != 3
            || spec_fields
                .iter()
                .any(|(key, _)| key != "dtype" && key != "shape" && key != "values")
        {
            return Err(malformed(format!(
                "{label} variable {name:?} must contain exactly dtype, shape, and values"
            )));
        }
        let dtype = unique_field(spec_fields, "dtype", &format!("{label} variable {name:?}"))?
            .as_str()
            .ok_or_else(|| {
                malformed(format!("{label} variable {name:?} dtype must be a string"))
            })?;
        if !matches!(dtype, "bool" | "int32" | "int64" | "float32" | "float64") {
            return Err(malformed(format!(
                "{label} variable {name:?} has unsupported dtype {dtype:?}"
            )));
        }
        let variable_label = format!("{label} variable {name:?}");
        if unique_field(spec_fields, "shape", &variable_label)?
            .as_array()
            .is_none()
        {
            return Err(malformed(format!(
                "{label} variable {name:?} shape must be an array"
            )));
        }
        let values = unique_field(spec_fields, "values", &variable_label)?
            .as_array()
            .ok_or_else(|| {
                malformed(format!("{label} variable {name:?} values must be an array"))
            })?;
        for value in values {
            let valid = match dtype {
                "bool" => matches!(value, Value::Bool(_)),
                "int32" => value
                    .as_i64()
                    .is_some_and(|value| i32::try_from(value).is_ok()),
                "int64" => matches!(value, Value::Int(_)),
                "float32" | "float64" => value.as_f64().is_some(),
                _ => unreachable!("dtype vocabulary was validated"),
            };
            if !valid {
                return Err(malformed(format!(
                    "{label} variable {name:?} contains a value outside dtype {dtype}"
                )));
            }
        }
    }
    // Reuse the binder's numeric, shape-product, boolean, and finite checks.
    data_from_json(document)?;
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

struct CheckedLines {
    total: usize,
    lines: Vec<String>,
}

impl CheckedLines {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            total: 0,
            lines: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, value: Value) -> Result<(), Error> {
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
        self.total = self
            .total
            .checked_add(line_bytes)
            .ok_or_else(|| invalid("generated-dataset artifact size overflowed this build"))?;
        if self.total > MAX_ARTIFACT_BYTES {
            return Err(invalid(format!(
                "generated-dataset artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
            )));
        }
        self.lines.push(line);
        Ok(())
    }

    fn finish(self) -> Vec<String> {
        self.lines
    }
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
    if request.generation_model_hash != sha256_bytes(request.model_document.as_bytes()) {
        return Err(invalid(
            "generation model hash must match the exact received model bytes",
        ));
    }
    if request.design_hash != sha256_bytes(request.design_document.as_bytes()) {
        return Err(invalid(
            "design hash must match the exact received design bytes",
        ));
    }
    let model = json::parse(&request.model_document)?;
    let meta = decode_model(&model)?;
    let design = json::parse(&request.design_document)?;
    let design_variables = normalized_variables(&design, "generate design")?;
    let design_data = data_from_json(&design)?;

    // Every draw line contains this compact canonical design. Reject a request
    // that cannot possibly fit before parameter validation or RNG work clones it.
    let compact_design = json::write(&canonical_document(design_variables.clone()))?;
    let minimum_design_bytes = compact_design
        .len()
        .checked_mul(request.count)
        .ok_or_else(|| invalid("generated-dataset artifact size overflowed this build"))?;
    if minimum_design_bytes > MAX_ARTIFACT_BYTES {
        return Err(invalid(format!(
            "generated-dataset artifact exceeds {MAX_ARTIFACT_BYTES} bytes"
        )));
    }

    let source = source_descriptor(&request.source)?;
    let mut lines = CheckedLines::with_capacity(request.count + 2);
    let mut schemas: Option<(Value, Value)> = None;
    let mut emitted = 0usize;
    {
        let mut emit_pair = |pair: GenerationPair| -> Result<(), Error> {
            let (parameters, dataset) = pair_documents(&pair, &design_variables)?;
            let parameter_schema = schema(&parameters, "generated parameters")?;
            let dataset_schema = schema(&dataset, "generated dataset")?;
            match &schemas {
                Some((expected_parameters, expected_dataset))
                    if *expected_parameters != parameter_schema
                        || *expected_dataset != dataset_schema =>
                {
                    return Err(invalid(
                        "generated parameter or dataset schema changed across draws",
                    ));
                }
                None => {
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
                        ("parameter_schema".to_string(), parameter_schema.clone()),
                        ("dataset_schema".to_string(), dataset_schema.clone()),
                    ]);
                    lines.push(Value::Object(header))?;
                    schemas = Some((parameter_schema, dataset_schema));
                }
                Some(_) => {}
            }
            let mut draw = common_fields();
            draw.extend([
                ("draw_index".to_string(), Value::Int(emitted as i64)),
                ("draw_count".to_string(), Value::Int(request.count as i64)),
                ("parameters".to_string(), parameters),
                ("dataset".to_string(), dataset),
                ("source_lineage".to_string(), lineage_value(&pair.lineage)),
            ]);
            lines.push(Value::Object(draw))?;
            emitted += 1;
            Ok(())
        };

        match &request.source {
            GenerationSource::Fixed {
                parameters_document,
                parameters_hash,
            } => {
                if *parameters_hash != sha256_bytes(parameters_document.as_bytes()) {
                    return Err(invalid(
                        "fixed parameters hash must match the exact received parameter bytes",
                    ));
                }
                let parameters = json::parse(parameters_document)?;
                canonical_variables(&parameters, "generate fixed parameters")?;
                fixed_generation_pairs(
                    meta.clone(),
                    design_data.clone(),
                    data_from_json(&parameters)?,
                    request.count,
                    request.seed,
                    &mut emit_pair,
                )?;
            }
            GenerationSource::ModelPrior { model_hash, .. } => {
                if model_hash != &request.generation_model_hash {
                    return Err(invalid(
                        "model-prior model hash must equal generation model hash",
                    ));
                }
                model_prior_generation_pairs(
                    meta.clone(),
                    design_data.clone(),
                    request.count,
                    request.seed,
                    &mut emit_pair,
                )?;
            }
            GenerationSource::Posterior {
                fit_ndjson,
                fit_data_document,
                fit_hash,
                fit_model_hash,
                fit_data_hash,
                expected_model_data_fingerprint,
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
                if *fit_data_hash != sha256_bytes(fit_data_document.as_bytes()) {
                    return Err(invalid(
                        "posterior fit data hash must match the exact received fit-data bytes",
                    ));
                }
                let fit_data = json::parse(fit_data_document)?;
                posterior_generation_pairs(
                    meta.clone(),
                    design_data.clone(),
                    PosteriorGenerationSource {
                        fit_data: data_from_json(&fit_data)?,
                        fit_ndjson,
                        expected_model_data_fingerprint: expected_model_data_fingerprint.as_deref(),
                    },
                    request.count,
                    request.seed,
                    &mut emit_pair,
                )?;
            }
        }
    }
    if emitted != request.count || schemas.is_none() {
        return Err(invalid("generate core returned the wrong pair count"));
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
    lines.push(Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(trailer),
    )]))?;
    Ok(lines.finish())
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    fingerprint::sha256_bytes(bytes)
}
