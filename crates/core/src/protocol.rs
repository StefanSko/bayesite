//! The v0-provisional NDJSON draws protocol, shared by the CLI and the
//! wasm ABI.
//!
//! Lines: a header object (`draws_format: "v0-provisional"`, artifact
//! kind/scope, parameter names/shapes/coordinate order, packing order,
//! workflow phases, settings, seed, chain count), one object per draw with
//! constrained values keyed by parameter, and a trailer with per-chain
//! diagnostics plus cross-chain R-hat/ESS. The marker is mandatory: the real
//! fit-artifact format is defined
//! elsewhere, and nothing may grow load-bearing dependencies on this one
//! unnoticed.

use crate::diagnostics;
use crate::error::{Error, ErrorKind};
use crate::ir::{decode_model, ModelMeta};
use crate::json::{self, Value};
use crate::model::{data_from_json, DataValue, Posterior};
use crate::predictive::{prior_predictive_ndjson_lines, PriorPredictiveSettings};
use crate::sampler::{sample, ChainDraws, Settings};
use crate::workflow::{recover_report, sbc_report, RecoverSettings, SbcSettings};

/// One constrained draw: (name, shape, values) per parameter.
type ConstrainedDraw = Vec<(String, Vec<usize>, Vec<f64>)>;
type DrawChainMetadata = (bool, Option<i64>, Option<Vec<i64>>);

const SAMPLE_WORKFLOW_PHASES: [&str; 7] = [
    "parse_json",
    "decode_ir",
    "bind_data",
    "build_posterior_state",
    "evaluate_logp_grad",
    "run_nuts",
    "emit_artifact",
];

const SAMPLE_ARTIFACT_KIND: &str = "posterior_draws";
const SAMPLE_ARTIFACT_SCOPE: &str = "observed_data_conditioned_parameter_draws";
const CHAIN_INDEX_BASE: &str = "zero_based_chain_id";

const DIAGNOSE_WORKFLOW_PHASES: [&str; 4] = [
    "parse_fit_ndjson",
    "validate_fit_artifact",
    "recompute_diagnostics",
    "emit_report",
];
const RHAT_STATISTIC: &str = "split_rhat";
const ESS_STATISTIC: &str = "effective_sample_size_geyer_initial_monotone_sequence";

fn tensor_to_value(shape: &[usize], data: &[f64]) -> Value {
    if shape.is_empty() {
        Value::Float(data[0])
    } else {
        Value::Array(data.iter().map(|&v| Value::Float(v)).collect())
    }
}

fn coordinate_order_value(shape: &[usize]) -> Value {
    if shape.contains(&0) {
        return Value::Array(Vec::new());
    }
    let size = shape.iter().product::<usize>().max(1);
    Value::Array(
        (0..size)
            .map(|flat| {
                let mut remainder = flat;
                let mut coordinate = vec![0usize; shape.len()];
                for axis in (0..shape.len()).rev() {
                    let dim = shape[axis];
                    coordinate[axis] = remainder % dim;
                    remainder /= dim;
                }
                Value::Array(
                    coordinate
                        .iter()
                        .map(|&index| Value::Int(index as i64))
                        .collect(),
                )
            })
            .collect(),
    )
}

fn parameter_order_value(packing: &[(String, Vec<usize>)]) -> Value {
    Value::Array(
        packing
            .iter()
            .map(|(name, _)| Value::Str(name.clone()))
            .collect(),
    )
}

fn chain_order_value(chain_ids: &[u64]) -> Value {
    Value::Array(
        chain_ids
            .iter()
            .map(|&chain_id| Value::Int(chain_id as i64))
            .collect(),
    )
}

fn source_chain_order_value(chain_ids: &[i64]) -> Value {
    Value::Array(
        chain_ids
            .iter()
            .map(|&chain_id| Value::Int(chain_id))
            .collect(),
    )
}

fn workflow_phases_value() -> Value {
    Value::Array(
        SAMPLE_WORKFLOW_PHASES
            .iter()
            .map(|phase| Value::Str((*phase).to_string()))
            .collect(),
    )
}

fn diagnose_workflow_phases_value() -> Value {
    Value::Array(
        DIAGNOSE_WORKFLOW_PHASES
            .iter()
            .map(|phase| Value::Str((*phase).to_string()))
            .collect(),
    )
}

fn header_value(
    packing: &[(String, Vec<usize>)],
    settings: &Settings,
    seed: u64,
    chain_ids: &[u64],
    draw_count: usize,
) -> Value {
    Value::Object(vec![
        (
            "draws_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "artifact_kind".to_string(),
            Value::Str(SAMPLE_ARTIFACT_KIND.to_string()),
        ),
        (
            "artifact_scope".to_string(),
            Value::Str(SAMPLE_ARTIFACT_SCOPE.to_string()),
        ),
        ("workflow_phases".to_string(), workflow_phases_value()),
        (
            "params".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, shape)| {
                        Value::Object(vec![
                            ("name".to_string(), Value::Str(name.clone())),
                            (
                                "shape".to_string(),
                                Value::Array(shape.iter().map(|&d| Value::Int(d as i64)).collect()),
                            ),
                            (
                                "coordinate_order".to_string(),
                                coordinate_order_value(shape),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "parameter_count".to_string(),
            Value::Int(packing.len() as i64),
        ),
        (
            "parameter_order".to_string(),
            parameter_order_value(packing),
        ),
        (
            "packing".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, _)| Value::Str(name.clone()))
                    .collect(),
            ),
        ),
        (
            "settings".to_string(),
            Value::Object(vec![
                (
                    "num_warmup".to_string(),
                    Value::Int(settings.num_warmup as i64),
                ),
                (
                    "num_draws".to_string(),
                    Value::Int(settings.num_draws as i64),
                ),
                (
                    "max_treedepth".to_string(),
                    Value::Int(settings.max_treedepth as i64),
                ),
                (
                    "target_accept".to_string(),
                    Value::Float(settings.target_accept),
                ),
            ]),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        (
            "chain_count".to_string(),
            Value::Int(chain_ids.len() as i64),
        ),
        ("chain_order".to_string(), chain_order_value(chain_ids)),
        ("draw_count".to_string(), Value::Int(draw_count as i64)),
        ("chains".to_string(), Value::Int(chain_ids.len() as i64)),
    ])
}

/// Render a complete run as NDJSON lines. `chains` pairs a chain id with
/// its draws; ids appear verbatim in the output (the CLI uses 0..C, a web
/// worker passes its own).
pub fn ndjson_lines(
    posterior: &Posterior,
    settings: &Settings,
    seed: u64,
    chains: &[(u64, ChainDraws)],
) -> Result<Vec<String>, Error> {
    validate_reportable_seed(seed, "sample artifact")?;
    validate_reportable_settings(settings)?;
    validate_fit_artifact_draws(chains)?;
    let draw_count = fit_artifact_draw_count(chains)?;
    let packing = posterior.packing();
    let chain_ids: Vec<u64> = chains.iter().map(|(chain_id, _)| *chain_id).collect();
    let mut lines =
        Vec::with_capacity(2 + chains.iter().map(|(_, c)| c.draws.len()).sum::<usize>());
    lines.push(json::write(&header_value(
        &packing, settings, seed, &chain_ids, draw_count,
    ))?);

    let mut constrained_chains: Vec<Vec<ConstrainedDraw>> = Vec::with_capacity(chains.len());
    for (_, chain) in chains {
        let mut constrained_draws = Vec::with_capacity(chain.draws.len());
        for q in &chain.draws {
            constrained_draws.push(
                posterior
                    .constrain(q)?
                    .into_iter()
                    .map(|(name, tensor)| (name, tensor.shape().to_vec(), tensor.data().to_vec()))
                    .collect::<ConstrainedDraw>(),
            );
        }
        constrained_chains.push(constrained_draws);
    }

    let mut draw_index = 0usize;
    for ((chain_id, _), draws) in chains.iter().zip(&constrained_chains) {
        for (draw_id, constrained) in draws.iter().enumerate() {
            let values = Value::Object(
                constrained
                    .iter()
                    .map(|(name, shape, data)| (name.clone(), tensor_to_value(shape, data)))
                    .collect(),
            );
            let line = Value::Object(vec![
                (
                    "draws_format".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
                (
                    "artifact_kind".to_string(),
                    Value::Str(SAMPLE_ARTIFACT_KIND.to_string()),
                ),
                (
                    "artifact_scope".to_string(),
                    Value::Str(SAMPLE_ARTIFACT_SCOPE.to_string()),
                ),
                ("draw_index".to_string(), Value::Int(draw_index as i64)),
                (
                    "draw_index_base".to_string(),
                    Value::Str("zero_based_retained_draw_order".to_string()),
                ),
                ("seed".to_string(), Value::Int(seed as i64)),
                ("draw_count".to_string(), Value::Int(draw_count as i64)),
                (
                    "chain_count".to_string(),
                    Value::Int(chain_ids.len() as i64),
                ),
                ("chain_order".to_string(), chain_order_value(&chain_ids)),
                ("chain".to_string(), Value::Int(*chain_id as i64)),
                (
                    "chain_index_base".to_string(),
                    Value::Str(CHAIN_INDEX_BASE.to_string()),
                ),
                ("draw".to_string(), Value::Int(draw_id as i64)),
                (
                    "parameter_count".to_string(),
                    Value::Int(packing.len() as i64),
                ),
                (
                    "parameter_order".to_string(),
                    parameter_order_value(&packing),
                ),
                ("values".to_string(), values),
            ]);
            lines.push(json::write(&line)?);
            draw_index += 1;
        }
    }

    // Cross-chain R-hat / ESS per parameter: max over coordinates for
    // R-hat, min for ESS, matching jaxstanv5.diagnostics conventions.
    let mut rhat_entries = Vec::new();
    let mut ess_entries = Vec::new();
    for (param_idx, (name, shape)) in packing.iter().enumerate() {
        let size: usize = shape.iter().product::<usize>().max(1);
        let mut worst_rhat = f64::NEG_INFINITY;
        let mut worst_ess = f64::INFINITY;
        for coord in 0..size {
            let series: Vec<Vec<f64>> = constrained_chains
                .iter()
                .map(|draws| {
                    draws
                        .iter()
                        .map(|constrained| constrained[param_idx].2[coord])
                        .collect()
                })
                .collect();
            worst_rhat = worst_rhat.max(diagnostics::split_rhat(&series));
            worst_ess = worst_ess.min(diagnostics::effective_sample_size(&series));
        }
        rhat_entries.push((name.clone(), Value::Float(worst_rhat)));
        ess_entries.push((name.clone(), Value::Float(worst_ess)));
    }

    let chain_stats = Value::Array(
        chains
            .iter()
            .map(|(chain_id, chain)| {
                Value::Object(vec![
                    ("chain".to_string(), Value::Int(*chain_id as i64)),
                    (
                        "chain_index_base".to_string(),
                        Value::Str(CHAIN_INDEX_BASE.to_string()),
                    ),
                    (
                        "draw_count".to_string(),
                        Value::Int(chain.draws.len() as i64),
                    ),
                    (
                        "divergences".to_string(),
                        Value::Int(chain.divergences as i64),
                    ),
                    (
                        "treedepth_histogram".to_string(),
                        Value::Array(
                            chain
                                .treedepth_histogram
                                .iter()
                                .map(|&c| Value::Int(c as i64))
                                .collect(),
                        ),
                    ),
                    ("step_size".to_string(), Value::Float(chain.step_size)),
                    ("mean_accept".to_string(), Value::Float(chain.mean_accept)),
                ])
            })
            .collect(),
    );
    let trailer = Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(vec![
            (
                "draws_format".to_string(),
                Value::Str("v0-provisional".to_string()),
            ),
            (
                "artifact_kind".to_string(),
                Value::Str(SAMPLE_ARTIFACT_KIND.to_string()),
            ),
            (
                "artifact_scope".to_string(),
                Value::Str(SAMPLE_ARTIFACT_SCOPE.to_string()),
            ),
            ("workflow_phases".to_string(), workflow_phases_value()),
            ("seed".to_string(), Value::Int(seed as i64)),
            (
                "draws_per_chain".to_string(),
                Value::Int(settings.num_draws as i64),
            ),
            ("chain_count".to_string(), Value::Int(chains.len() as i64)),
            ("chain_order".to_string(), chain_order_value(&chain_ids)),
            ("draw_count".to_string(), Value::Int(draw_count as i64)),
            (
                "parameter_count".to_string(),
                Value::Int(packing.len() as i64),
            ),
            (
                "parameter_order".to_string(),
                parameter_order_value(&packing),
            ),
            ("params".to_string(), Value::Int(packing.len() as i64)),
            ("chains".to_string(), chain_stats),
            ("rhat".to_string(), Value::Object(rhat_entries)),
            ("ess".to_string(), Value::Object(ess_entries)),
        ]),
    )]);
    lines.push(json::write(&trailer)?);
    Ok(lines)
}

#[derive(Debug)]
struct ParamSpec {
    name: String,
    shape: Vec<usize>,
    size: usize,
}

#[derive(Debug)]
struct ParsedDraw {
    draw_index: Option<usize>,
    parameter_metadata: bool,
    artifact_metadata: bool,
    draw_count_metadata: Option<i64>,
    chain_metadata: bool,
    chain_count_metadata: Option<i64>,
    chain_order_metadata: Option<Vec<i64>>,
    chain: i64,
    draw: usize,
    values: Vec<Vec<f64>>,
}

fn invalid_fit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn invalid_artifact(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn validate_reportable_seed(seed: u64, context: &str) -> Result<(), Error> {
    if seed <= i64::MAX as u64 {
        Ok(())
    } else {
        Err(invalid_artifact(format!(
            "{context} seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers"
        )))
    }
}

fn validate_reportable_chain_id(chain_id: u64) -> Result<(), Error> {
    if chain_id <= i64::MAX as u64 {
        Ok(())
    } else {
        Err(invalid_artifact(
            "sample artifact chain ids must be in 0..=9223372036854775807 because artifacts report chain ids as JSON integers",
        ))
    }
}

fn validate_reportable_settings(settings: &Settings) -> Result<(), Error> {
    if settings.num_draws > i64::MAX as usize {
        return Err(invalid_artifact(
            "sample artifact settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers",
        ));
    }
    if settings.num_warmup > i64::MAX as usize {
        return Err(invalid_artifact(
            "sample artifact settings.num_warmup must be in 0..=9223372036854775807 because artifacts report warmup counts as JSON integers",
        ));
    }
    Ok(())
}

fn validate_reportable_chain_diagnostics(chain: &ChainDraws) -> Result<(), Error> {
    if chain.divergences > i64::MAX as usize {
        return Err(invalid_artifact(
            "sample artifact chain divergences must be in 0..=9223372036854775807 because artifacts report divergences as JSON integers",
        ));
    }
    if chain
        .treedepth_histogram
        .iter()
        .any(|&count| count > i64::MAX as usize)
    {
        return Err(invalid_artifact(
            "sample artifact treedepth histogram counts must be in 0..=9223372036854775807 because artifacts report treedepth counts as JSON integers",
        ));
    }
    Ok(())
}

fn validate_fit_artifact_draws(chains: &[(u64, ChainDraws)]) -> Result<(), Error> {
    let Some((_, first)) = chains.first() else {
        return Err(invalid_artifact(
            "sample artifacts need at least one chain because they include diagnostics",
        ));
    };
    for (chain_id, chain) in chains {
        validate_reportable_chain_id(*chain_id)?;
        validate_reportable_chain_diagnostics(chain)?;
    }
    let draws_per_chain = first.draws.len();
    if draws_per_chain < 4 {
        return Err(invalid_artifact(
            "sample artifact chains must have at least 4 draws per chain because artifacts include diagnostics",
        ));
    }
    if chains
        .iter()
        .any(|(_, chain)| chain.draws.len() != draws_per_chain)
    {
        return Err(invalid_artifact(
            "sample artifact chains must all have the same number of draws for diagnostics",
        ));
    }
    Ok(())
}

fn fit_artifact_draw_count(chains: &[(u64, ChainDraws)]) -> Result<usize, Error> {
    let mut draw_count = 0usize;
    for (_, chain) in chains {
        draw_count = draw_count
            .checked_add(chain.draws.len())
            .ok_or_else(|| {
                invalid_artifact(
                    "sample artifact draw_count must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers",
                )
            })?;
    }
    if draw_count == 0 || draw_count > i64::MAX as usize {
        return Err(invalid_artifact(
            "sample artifact draw_count must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers",
        ));
    }
    Ok(draw_count)
}

fn parse_param_specs(header: &Value) -> Result<Vec<ParamSpec>, Error> {
    if header.get("draws_format").and_then(Value::as_str) != Some("v0-provisional") {
        return Err(invalid_fit(
            "fit header needs draws_format \"v0-provisional\"; rerun `bayesite sample`",
        ));
    }
    let params = header
        .get("params")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_fit("fit header needs a params array from `bayesite sample`"))?;
    let mut specs = Vec::with_capacity(params.len());
    for param in params {
        let name = param
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid_fit("each params entry needs a string name"))?
            .to_string();
        let shape_values = param
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_fit(format!("parameter {name} needs a shape array")))?;
        let mut shape = Vec::with_capacity(shape_values.len());
        for value in shape_values {
            let dim = value.as_i64().ok_or_else(|| {
                invalid_fit(format!(
                    "parameter {name} shape dimensions must be integers"
                ))
            })?;
            if dim < 0 {
                return Err(invalid_fit(format!(
                    "parameter {name} shape dimensions must be non-negative"
                )));
            }
            shape.push(dim as usize);
        }
        let mut size = 1usize;
        for dim in &shape {
            size = size.checked_mul(*dim).ok_or_else(|| {
                invalid_fit(format!(
                    "parameter {name} shape size is too large for this build; rerun `bayesite sample` to completion"
                ))
            })?;
            if size > i64::MAX as usize {
                return Err(invalid_fit(format!(
                    "parameter {name} shape size is too large for this build; rerun `bayesite sample` to completion"
                )));
            }
        }
        size = size.max(1);
        specs.push(ParamSpec { name, shape, size });
    }
    if specs.is_empty() {
        return Err(invalid_fit(
            "fit header has no parameters; rerun sampling with a model that has free values",
        ));
    }
    for index in 0..specs.len() {
        if specs[..index]
            .iter()
            .any(|existing| existing.name == specs[index].name)
        {
            return Err(invalid_fit(format!(
                "fit header params has duplicate parameter name \"{}\"; rerun `bayesite sample` to completion",
                specs[index].name
            )));
        }
    }
    let packing = header
        .get("packing")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_fit("fit header needs a packing array from `bayesite sample`"))?;
    if packing.len() != specs.len() {
        return Err(invalid_fit(
            "fit header packing must match params order; rerun `bayesite sample` to completion",
        ));
    }
    for (packing_name, spec) in packing.iter().zip(&specs) {
        if packing_name.as_str() != Some(&spec.name) {
            return Err(invalid_fit(
                "fit header packing must match params order; rerun `bayesite sample` to completion",
            ));
        }
    }
    validate_optional_parameter_order(
        header,
        "fit header",
        &specs,
        "fit header parameter_order must match params order; rerun `bayesite sample` to completion",
    )?;
    Ok(specs)
}

fn validate_optional_parameter_order(
    doc: &Value,
    context: &str,
    specs: &[ParamSpec],
    mismatch_message: &str,
) -> Result<(), Error> {
    let Some(value) = doc.get("parameter_order") else {
        return Ok(());
    };
    let order = value.as_array().ok_or_else(|| {
        invalid_fit(format!(
            "{context} parameter_order must be an array of strings"
        ))
    })?;
    if order.len() != specs.len() {
        return Err(invalid_fit(mismatch_message));
    }
    for (value, spec) in order.iter().zip(specs) {
        let Some(name) = value.as_str() else {
            return Err(invalid_fit(format!(
                "{context} parameter_order must be an array of strings"
            )));
        };
        if name != spec.name {
            return Err(invalid_fit(mismatch_message));
        }
    }
    Ok(())
}

fn validate_optional_draw_parameter_metadata(
    line: &Value,
    specs: &[ParamSpec],
) -> Result<bool, Error> {
    let has_count = line.get("parameter_count").is_some();
    let has_order = line.get("parameter_order").is_some();
    match (has_count, has_order) {
        (false, false) => Ok(false),
        (true, false) | (false, true) => Err(invalid_fit(
            "draw line parameter metadata must include both parameter_count and parameter_order when present",
        )),
        (true, true) => {
            let count = line
                .get("parameter_count")
                .and_then(Value::as_i64)
                .ok_or_else(|| {
                    invalid_fit("draw line parameter_count must be an integer when present")
                })?;
            if count < 1 {
                return Err(invalid_fit(
                    "draw line parameter_count must be at least 1 when present",
                ));
            }
            let count = usize::try_from(count).map_err(|_| {
                invalid_fit("draw line parameter_count must fit this build's usize")
            })?;
            if count != specs.len() {
                return Err(invalid_fit(
                    "draw line parameter_count must match fit header params length; rerun `bayesite sample` to completion",
                ));
            }
            validate_optional_parameter_order(
                line,
                "draw line",
                specs,
                "draw line parameter_order must match fit header params order; rerun `bayesite sample` to completion",
            )?;
            Ok(true)
        }
    }
}

fn validate_optional_draw_artifact_metadata(
    line: &Value,
    source_seed: i64,
    has_draw_index: bool,
) -> Result<(bool, Option<i64>), Error> {
    let has_format = line.get("draws_format").is_some();
    let has_kind = line.get("artifact_kind").is_some();
    let has_scope = line.get("artifact_scope").is_some();
    let has_index_base = line.get("draw_index_base").is_some();
    let has_seed = line.get("seed").is_some();
    let has_draw_count = line.get("draw_count").is_some();
    let has_any =
        has_format || has_kind || has_scope || has_index_base || has_seed || has_draw_count;
    if !has_any {
        return Ok((false, None));
    }
    if !has_draw_index {
        return Err(invalid_fit(
            "draw line artifact metadata must include draw_index when present",
        ));
    }
    if !(has_format && has_kind && has_scope && has_index_base && has_seed && has_draw_count) {
        return Err(invalid_fit(
            "draw line artifact metadata must include draw_index, draws_format, artifact_kind, artifact_scope, draw_index_base, seed, and draw_count when present",
        ));
    }
    let format = line
        .get("draws_format")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_fit("draw line draws_format must be a string when present"))?;
    if format != "v0-provisional" {
        return Err(invalid_fit(
            "draw line draws_format must be \"v0-provisional\" when present; rerun `bayesite sample` to completion",
        ));
    }
    parse_sample_artifact_field(line, "draw line", "artifact_kind", SAMPLE_ARTIFACT_KIND)?;
    parse_sample_artifact_field(line, "draw line", "artifact_scope", SAMPLE_ARTIFACT_SCOPE)?;
    let draw_index_base = line
        .get("draw_index_base")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_fit("draw line draw_index_base must be a string when present"))?;
    if draw_index_base != "zero_based_retained_draw_order" {
        return Err(invalid_fit(
            "draw line draw_index_base must be \"zero_based_retained_draw_order\" when present; rerun `bayesite sample` to completion",
        ));
    }
    let seed = line
        .get("seed")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_fit("draw line seed must be an integer when present"))?;
    if seed != source_seed {
        return Err(invalid_fit(
            "draw line seed must match fit header seed; rerun `bayesite sample` to completion",
        ));
    }
    let draw_count = line
        .get("draw_count")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_fit("draw line draw_count must be an integer when present"))?;
    if draw_count < 1 {
        return Err(invalid_fit(
            "draw line draw_count must be at least 1 when present",
        ));
    }
    Ok((true, Some(draw_count)))
}

fn validate_optional_draw_chain_metadata(line: &Value) -> Result<DrawChainMetadata, Error> {
    let has_count = line.get("chain_count").is_some();
    let has_order = line.get("chain_order").is_some();
    match (has_count, has_order) {
        (false, false) => Ok((false, None, None)),
        (true, false) | (false, true) => Err(invalid_fit(
            "draw line chain metadata must include both chain_count and chain_order when present",
        )),
        (true, true) => {
            let count = line
                .get("chain_count")
                .and_then(Value::as_i64)
                .ok_or_else(|| {
                    invalid_fit("draw line chain_count must be an integer when present")
                })?;
            if count < 1 {
                return Err(invalid_fit(
                    "draw line chain_count must be at least 1 when present",
                ));
            }
            let order = line
                .get("chain_order")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid_fit(
                        "draw line chain_order must be an array of non-negative integers when present",
                    )
                })?;
            if order.len() != count as usize {
                return Err(invalid_fit(
                    "draw line chain_order length must match chain_count when present",
                ));
            }
            let mut parsed = Vec::with_capacity(order.len());
            for value in order {
                let chain_id = value.as_i64().ok_or_else(|| {
                    invalid_fit(
                        "draw line chain_order must be an array of non-negative integers when present",
                    )
                })?;
                if chain_id < 0 {
                    return Err(invalid_fit(
                        "draw line chain_order must be an array of non-negative integers when present",
                    ));
                }
                parsed.push(chain_id);
            }
            Ok((true, Some(count), Some(parsed)))
        }
    }
}

fn validate_optional_chain_order(
    doc: &Value,
    context: &str,
    expected: &[i64],
    mismatch_message: &str,
) -> Result<(), Error> {
    let Some(value) = doc.get("chain_order") else {
        return Ok(());
    };
    let order = value.as_array().ok_or_else(|| {
        invalid_fit(format!(
            "{context} chain_order must be an array of non-negative integers"
        ))
    })?;
    if order.len() != expected.len() {
        return Err(invalid_fit(mismatch_message));
    }
    for (value, expected_id) in order.iter().zip(expected) {
        let parsed = value.as_i64().ok_or_else(|| {
            invalid_fit(format!(
                "{context} chain_order must be an array of non-negative integers"
            ))
        })?;
        if parsed < 0 {
            return Err(invalid_fit(format!(
                "{context} chain_order must be an array of non-negative integers"
            )));
        }
        if parsed != *expected_id {
            return Err(invalid_fit(mismatch_message));
        }
    }
    Ok(())
}

fn parse_positive_usize_field(value: &Value, field: &str) -> Result<usize, Error> {
    let parsed = value
        .as_i64()
        .ok_or_else(|| invalid_fit(format!("fit header {field} must be an integer")))?;
    if parsed < 1 {
        return Err(invalid_fit(format!(
            "fit header {field} must be at least 1"
        )));
    }
    usize::try_from(parsed)
        .map_err(|_| invalid_fit(format!("fit header {field} must fit this build's usize")))
}

fn parse_header_chain_count(header: &Value) -> Result<usize, Error> {
    let value = header
        .get("chains")
        .ok_or_else(|| invalid_fit("fit header needs integer chains from `bayesite sample`"))?;
    parse_positive_usize_field(value, "chains")
}

fn validate_optional_header_chain_count(header: &Value, expected: usize) -> Result<(), Error> {
    let Some(value) = header.get("chain_count") else {
        return Ok(());
    };
    let parsed = parse_positive_usize_field(value, "chain_count")?;
    if parsed == expected {
        Ok(())
    } else {
        Err(invalid_fit(
            "fit header chain_count must match fit header chains; rerun `bayesite sample` to completion",
        ))
    }
}

fn validate_optional_header_parameter_count(header: &Value, expected: usize) -> Result<(), Error> {
    let Some(value) = header.get("parameter_count") else {
        return Ok(());
    };
    let parsed = parse_positive_usize_field(value, "parameter_count")?;
    if parsed == expected {
        Ok(())
    } else {
        Err(invalid_fit(
            "fit header parameter_count must match fit header params length; rerun `bayesite sample` to completion",
        ))
    }
}

fn validate_optional_header_draw_count(header: &Value, expected: i64) -> Result<(), Error> {
    let Some(value) = header.get("draw_count") else {
        return Ok(());
    };
    let parsed = value
        .as_i64()
        .ok_or_else(|| invalid_fit("fit header draw_count must be an integer"))?;
    if parsed == expected {
        Ok(())
    } else {
        Err(invalid_fit(
            "fit header draw_count must match retained draw line count; rerun `bayesite sample` to completion",
        ))
    }
}

fn parse_header_draw_count(header: &Value) -> Result<usize, Error> {
    let settings = header
        .get("settings")
        .ok_or_else(|| invalid_fit("fit header needs settings from `bayesite sample`"))?;
    let value = settings.get("num_draws").ok_or_else(|| {
        invalid_fit("fit header settings needs integer num_draws from `bayesite sample`")
    })?;
    parse_positive_usize_field(value, "settings.num_draws")
}

fn parse_header_seed(header: &Value) -> Result<i64, Error> {
    let seed = header
        .get("seed")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_fit("fit header needs integer seed from `bayesite sample`"))?;
    if seed < 0 {
        return Err(invalid_fit(
            "fit header seed must be non-negative; rerun `bayesite sample`",
        ));
    }
    Ok(seed)
}

fn parse_nonnegative_i64_field(value: &Value, field: &str) -> Result<i64, Error> {
    let parsed = value
        .as_i64()
        .ok_or_else(|| invalid_fit(format!("fit header {field} must be an integer")))?;
    if parsed < 0 {
        return Err(invalid_fit(format!(
            "fit header {field} must be non-negative"
        )));
    }
    Ok(parsed)
}

fn parse_header_settings(header: &Value) -> Result<Value, Error> {
    let settings = header
        .get("settings")
        .ok_or_else(|| invalid_fit("fit header needs settings from `bayesite sample`"))?;
    let num_warmup = parse_nonnegative_i64_field(
        settings.get("num_warmup").ok_or_else(|| {
            invalid_fit("fit header settings needs integer num_warmup from `bayesite sample`")
        })?,
        "settings.num_warmup",
    )?;
    let num_draws = parse_positive_usize_field(
        settings.get("num_draws").ok_or_else(|| {
            invalid_fit("fit header settings needs integer num_draws from `bayesite sample`")
        })?,
        "settings.num_draws",
    )?;
    let max_treedepth = parse_positive_usize_field(
        settings.get("max_treedepth").ok_or_else(|| {
            invalid_fit("fit header settings needs integer max_treedepth from `bayesite sample`")
        })?,
        "settings.max_treedepth",
    )?;
    if max_treedepth > 20 {
        return Err(invalid_fit(
            "fit header settings.max_treedepth must be in 1..=20; rerun `bayesite sample`",
        ));
    }
    let target_accept = settings
        .get("target_accept")
        .and_then(Value::as_f64)
        .ok_or_else(|| {
            invalid_fit("fit header settings needs numeric target_accept from `bayesite sample`")
        })?;
    if !(0.0..1.0).contains(&target_accept) {
        return Err(invalid_fit(
            "fit header settings.target_accept must be in (0, 1); rerun `bayesite sample`",
        ));
    }
    Ok(Value::Object(vec![
        ("num_warmup".to_string(), Value::Int(num_warmup)),
        ("num_draws".to_string(), Value::Int(num_draws as i64)),
        (
            "max_treedepth".to_string(),
            Value::Int(max_treedepth as i64),
        ),
        ("target_accept".to_string(), Value::Float(target_accept)),
    ]))
}

fn parse_workflow_phases(doc: &Value, context: &str) -> Result<Option<Vec<String>>, Error> {
    let Some(value) = doc.get("workflow_phases") else {
        return Ok(None);
    };
    let phases = value.as_array().ok_or_else(|| {
        invalid_fit(format!(
            "{context} workflow_phases must be an array of strings"
        ))
    })?;
    let mut parsed = Vec::with_capacity(phases.len());
    for phase in phases {
        parsed.push(
            phase
                .as_str()
                .ok_or_else(|| {
                    invalid_fit(format!(
                        "{context} workflow_phases must be an array of strings"
                    ))
                })?
                .to_string(),
        );
    }
    if parsed.len() != SAMPLE_WORKFLOW_PHASES.len()
        || parsed
            .iter()
            .zip(SAMPLE_WORKFLOW_PHASES)
            .any(|(got, expected)| got.as_str() != expected)
    {
        return Err(invalid_fit(format!(
            "{context} workflow_phases must match the v0-provisional sample workflow; rerun `bayesite sample` to completion"
        )));
    }
    Ok(Some(parsed))
}

fn parse_sample_artifact_field(
    doc: &Value,
    context: &str,
    field: &str,
    expected: &str,
) -> Result<Option<String>, Error> {
    let Some(value) = doc.get(field) else {
        return Ok(None);
    };
    let parsed = value
        .as_str()
        .ok_or_else(|| invalid_fit(format!("{context} {field} must be a string when present")))?;
    if parsed == expected {
        Ok(Some(parsed.to_string()))
    } else {
        Err(invalid_fit(format!(
            "{context} {field} must be \"{expected}\" when present; rerun `bayesite sample` to completion"
        )))
    }
}

fn validate_trailer_draws_format(trailer: &Value) -> Result<(), Error> {
    let Some(value) = trailer.get("draws_format") else {
        return Ok(());
    };
    if value.as_str() == Some("v0-provisional") {
        Ok(())
    } else {
        Err(invalid_fit(
            "fit trailer draws_format must be \"v0-provisional\" when present; rerun `bayesite sample` to completion",
        ))
    }
}

fn validate_optional_trailer_i64(
    trailer: &Value,
    field: &str,
    expected: i64,
    message: &str,
) -> Result<(), Error> {
    let Some(value) = trailer.get(field) else {
        return Ok(());
    };
    let parsed = value
        .as_i64()
        .ok_or_else(|| invalid_fit(format!("fit trailer {field} must be an integer")))?;
    if parsed == expected {
        Ok(())
    } else {
        Err(invalid_fit(message))
    }
}

fn validate_trailer_completion_metadata(
    trailer: &Value,
    source_seed: i64,
    chain_count: usize,
    draw_count: i64,
    draws_per_chain: usize,
    param_count: usize,
) -> Result<(), Error> {
    validate_optional_trailer_i64(
        trailer,
        "seed",
        source_seed,
        "fit trailer seed must match fit header seed; rerun `bayesite sample` to completion",
    )?;
    validate_optional_trailer_i64(
        trailer,
        "chain_count",
        chain_count as i64,
        "fit trailer chain_count must match fit header chains; rerun `bayesite sample` to completion",
    )?;
    validate_optional_trailer_i64(
        trailer,
        "draw_count",
        draw_count,
        "fit trailer draw_count must match retained draw line count; rerun `bayesite sample` to completion",
    )?;
    validate_optional_trailer_i64(
        trailer,
        "draws_per_chain",
        draws_per_chain as i64,
        "fit trailer draws_per_chain must match fit header settings.num_draws; rerun `bayesite sample` to completion",
    )?;
    validate_optional_trailer_i64(
        trailer,
        "parameter_count",
        param_count as i64,
        "fit trailer parameter_count must match fit header params length; rerun `bayesite sample` to completion",
    )?;
    validate_optional_trailer_i64(
        trailer,
        "params",
        param_count as i64,
        "fit trailer params must match fit header params length; rerun `bayesite sample` to completion",
    )
}

fn trailer_completion_metadata_value(trailer: &Value) -> Value {
    Value::Object(vec![
        (
            "draws_format".to_string(),
            Value::Bool(trailer.get("draws_format").is_some()),
        ),
        (
            "artifact_kind".to_string(),
            Value::Bool(trailer.get("artifact_kind").is_some()),
        ),
        (
            "artifact_scope".to_string(),
            Value::Bool(trailer.get("artifact_scope").is_some()),
        ),
        (
            "workflow_phases".to_string(),
            Value::Bool(trailer.get("workflow_phases").is_some()),
        ),
        (
            "seed".to_string(),
            Value::Bool(trailer.get("seed").is_some()),
        ),
        (
            "chain_count".to_string(),
            Value::Bool(trailer.get("chain_count").is_some()),
        ),
        (
            "chain_order".to_string(),
            Value::Bool(trailer.get("chain_order").is_some()),
        ),
        (
            "draw_count".to_string(),
            Value::Bool(trailer.get("draw_count").is_some()),
        ),
        (
            "draws_per_chain".to_string(),
            Value::Bool(trailer.get("draws_per_chain").is_some()),
        ),
        (
            "parameter_count".to_string(),
            Value::Bool(trailer.get("parameter_count").is_some()),
        ),
        (
            "parameter_order".to_string(),
            Value::Bool(trailer.get("parameter_order").is_some()),
        ),
        (
            "params".to_string(),
            Value::Bool(trailer.get("params").is_some()),
        ),
    ])
}

fn workflow_phases_array(phases: &[String]) -> Value {
    Value::Array(
        phases
            .iter()
            .map(|phase| Value::Str(phase.clone()))
            .collect(),
    )
}

fn param_specs_value(specs: &[ParamSpec]) -> Value {
    Value::Array(
        specs
            .iter()
            .map(|spec| {
                Value::Object(vec![
                    ("name".to_string(), Value::Str(spec.name.clone())),
                    (
                        "shape".to_string(),
                        Value::Array(
                            spec.shape
                                .iter()
                                .map(|&dim| Value::Int(dim as i64))
                                .collect(),
                        ),
                    ),
                    (
                        "coordinate_order".to_string(),
                        coordinate_order_value(&spec.shape),
                    ),
                ])
            })
            .collect(),
    )
}

fn packing_value(specs: &[ParamSpec]) -> Value {
    Value::Array(
        specs
            .iter()
            .map(|spec| Value::Str(spec.name.clone()))
            .collect(),
    )
}

fn parse_param_value(value: &Value, spec: &ParamSpec) -> Result<Vec<f64>, Error> {
    if spec.shape.is_empty() {
        return value
            .as_f64()
            .map(|v| vec![v])
            .ok_or_else(|| invalid_fit(format!("draw value for {} must be a number", spec.name)));
    }
    let items = value.as_array().ok_or_else(|| {
        invalid_fit(format!(
            "draw value for {} must be an array matching shape {:?}",
            spec.name, spec.shape
        ))
    })?;
    if items.len() != spec.size {
        return Err(invalid_fit(format!(
            "draw value for {} has {} entries but shape {:?} needs {}",
            spec.name,
            items.len(),
            spec.shape,
            spec.size
        )));
    }
    items
        .iter()
        .map(|item| {
            item.as_f64().ok_or_else(|| {
                invalid_fit(format!(
                    "draw value for {} contains a non-number",
                    spec.name
                ))
            })
        })
        .collect()
}

fn parse_draw(line: &Value, specs: &[ParamSpec], source_seed: i64) -> Result<ParsedDraw, Error> {
    let draw_index = match line.get("draw_index") {
        Some(value) => {
            let parsed = value.as_i64().ok_or_else(|| {
                invalid_fit("draw line draw_index must be a non-negative integer when present")
            })?;
            if parsed < 0 {
                return Err(invalid_fit(
                    "draw line draw_index must be a non-negative integer when present",
                ));
            }
            Some(
                usize::try_from(parsed)
                    .map_err(|_| invalid_fit("draw line draw_index must fit this build's usize"))?,
            )
        }
        None => None,
    };
    let parameter_metadata = validate_optional_draw_parameter_metadata(line, specs)?;
    let (artifact_metadata, draw_count_metadata) =
        validate_optional_draw_artifact_metadata(line, source_seed, draw_index.is_some())?;
    let (chain_metadata, chain_count_metadata, chain_order_metadata) =
        validate_optional_draw_chain_metadata(line)?;
    let chain = line
        .get("chain")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_fit("each draw line needs an integer chain field"))?;
    if chain < 0 {
        return Err(invalid_fit("draw line chain field must be non-negative"));
    }
    let draw = line
        .get("draw")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_fit("each draw line needs an integer draw field"))?;
    if draw < 0 {
        return Err(invalid_fit("draw line draw field must be non-negative"));
    }
    let draw = usize::try_from(draw)
        .map_err(|_| invalid_fit("draw line draw field must fit this build's usize"))?;
    let values = line
        .get("values")
        .ok_or_else(|| invalid_fit("each draw line needs a values object"))?;
    let mut parsed = Vec::with_capacity(specs.len());
    for spec in specs {
        let value = values.get(&spec.name).ok_or_else(|| {
            invalid_fit(format!(
                "draw line is missing value for parameter {}",
                spec.name
            ))
        })?;
        parsed.push(parse_param_value(value, spec)?);
    }
    Ok(ParsedDraw {
        draw_index,
        parameter_metadata,
        artifact_metadata,
        draw_count_metadata,
        chain_metadata,
        chain_count_metadata,
        chain_order_metadata,
        chain,
        draw,
        values: parsed,
    })
}

fn chain_index(chain_ids: &mut Vec<i64>, chain: i64) -> usize {
    if let Some(index) = chain_ids.iter().position(|&id| id == chain) {
        index
    } else {
        chain_ids.push(chain);
        chain_ids.len() - 1
    }
}

fn trailer_chain_ids(chain_stats: &[Value]) -> Result<Vec<i64>, Error> {
    let mut ids = Vec::with_capacity(chain_stats.len());
    for stats in chain_stats {
        let id = stats
            .get("chain")
            .and_then(Value::as_i64)
            .ok_or_else(|| invalid_fit("each fit trailer chain entry needs an integer chain"))?;
        if id < 0 {
            return Err(invalid_fit("fit trailer chain field must be non-negative"));
        }
        if ids.contains(&id) {
            return Err(invalid_fit(
                "fit trailer chains must be unique; rerun `bayesite sample` to completion",
            ));
        }
        ids.push(id);
    }
    Ok(ids)
}

fn validate_trailer_chain_stats(
    chain_stats: &[Value],
    draws_per_chain: usize,
) -> Result<(), Error> {
    for stats in chain_stats {
        if let Some(value) = stats.get("draw_count") {
            let draw_count = value
                .as_i64()
                .ok_or_else(|| invalid_fit("fit trailer chain draw_count must be an integer"))?;
            if draw_count != draws_per_chain as i64 {
                return Err(invalid_fit(
                    "fit trailer chain draw_count must match fit header settings.num_draws; rerun `bayesite sample` to completion",
                ));
            }
        }
        let divergences = stats
            .get("divergences")
            .and_then(Value::as_i64)
            .ok_or_else(|| {
                invalid_fit("each fit trailer chain entry needs an integer divergences")
            })?;
        if divergences < 0 {
            return Err(invalid_fit(
                "fit trailer chain divergences must be non-negative",
            ));
        }
        let histogram = stats
            .get("treedepth_histogram")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                invalid_fit("each fit trailer chain entry needs a treedepth_histogram array")
            })?;
        for count in histogram {
            let count = count.as_i64().ok_or_else(|| {
                invalid_fit("fit trailer treedepth_histogram counts must be integers")
            })?;
            if count < 0 {
                return Err(invalid_fit(
                    "fit trailer treedepth_histogram counts must be non-negative",
                ));
            }
        }
        let step_size = stats
            .get("step_size")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid_fit("each fit trailer chain entry needs a numeric step_size"))?;
        if step_size <= 0.0 {
            return Err(invalid_fit("fit trailer chain step_size must be positive"));
        }
        let mean_accept = stats
            .get("mean_accept")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                invalid_fit("each fit trailer chain entry needs a numeric mean_accept")
            })?;
        if !(0.0..=1.0).contains(&mean_accept) {
            return Err(invalid_fit(
                "fit trailer chain mean_accept must be in [0, 1]",
            ));
        }
    }
    Ok(())
}

fn validate_trailer_chains(
    chain_stats: &[Value],
    draw_chain_ids: &[i64],
    draws_per_chain: usize,
) -> Result<(), Error> {
    validate_trailer_chain_stats(chain_stats, draws_per_chain)?;
    let mut trailer_ids = trailer_chain_ids(chain_stats)?;
    let mut draw_ids = draw_chain_ids.to_vec();
    trailer_ids.sort_unstable();
    draw_ids.sort_unstable();
    if trailer_ids != draw_ids {
        return Err(invalid_fit(
            "fit trailer chains must match draw chain ids; rerun `bayesite sample` to completion",
        ));
    }
    Ok(())
}

/// Diagnose a complete v0-provisional fit NDJSON stream produced by
/// `bayesite sample`.
///
/// Input format is provisional and explicitly marked by the header's
/// `draws_format: "v0-provisional"`. Output is one JSON object with
/// `diagnostics_format: "v0-provisional"`, the source format, source artifact
/// identity when present, source workflow phases when present, draws per chain,
/// per-chain sampler stats from the fit trailer, and recomputed per-parameter
/// R-hat/ESS.
pub fn diagnose_ndjson(text: &str) -> Result<String, Error> {
    let mut lines = text.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| invalid_fit("fit is empty; pass NDJSON from `bayesite sample`"))?;
    let header = json::parse(header_line)?;
    let specs = parse_param_specs(&header)?;
    validate_optional_header_parameter_count(&header, specs.len())?;
    let header_chain_count = parse_header_chain_count(&header)?;
    validate_optional_header_chain_count(&header, header_chain_count)?;
    let header_draw_count = parse_header_draw_count(&header)?;
    let source_seed = parse_header_seed(&header)?;
    let source_settings = parse_header_settings(&header)?;
    let header_workflow_phases = parse_workflow_phases(&header, "fit header")?;
    let header_artifact_kind =
        parse_sample_artifact_field(&header, "fit header", "artifact_kind", SAMPLE_ARTIFACT_KIND)?;
    let header_artifact_scope = parse_sample_artifact_field(
        &header,
        "fit header",
        "artifact_scope",
        SAMPLE_ARTIFACT_SCOPE,
    )?;

    let mut draws = Vec::new();
    let mut trailer: Option<Value> = None;
    let mut chain_ids = Vec::new();
    let mut next_draw_by_chain: Vec<usize> = Vec::new();
    let mut draw_index_metadata: Option<bool> = None;
    let mut draw_parameter_metadata: Option<bool> = None;
    let mut draw_artifact_metadata: Option<bool> = None;
    let mut draw_chain_metadata: Option<bool> = None;
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(invalid_fit(format!(
                "line {} is blank; v0-provisional fit NDJSON has no blank lines",
                line_index + 2
            )));
        }
        let doc = json::parse(line)?;
        if let Some(value) = doc.get("trailer") {
            if trailer.is_some() {
                return Err(invalid_fit(
                    "fit has more than one trailer; keep one complete sample output",
                ));
            }
            trailer = Some(value.clone());
            continue;
        }
        if trailer.is_some() {
            return Err(invalid_fit(
                "fit trailer must be the final line; remove trailing lines after the trailer",
            ));
        }
        let draw = parse_draw(&doc, &specs, source_seed)?;
        let has_draw_index = draw.draw_index.is_some();
        match draw_index_metadata {
            Some(expected) if expected != has_draw_index => {
                return Err(invalid_fit(
                    "fit draw_index metadata must be present on every draw line or omitted from every draw line; rerun `bayesite sample` to completion",
                ));
            }
            Some(_) => {}
            None => draw_index_metadata = Some(has_draw_index),
        }
        match draw_parameter_metadata {
            Some(expected) if expected != draw.parameter_metadata => {
                return Err(invalid_fit(
                    "fit draw parameter metadata must be present on every draw line or omitted from every draw line; rerun `bayesite sample` to completion",
                ));
            }
            Some(_) => {}
            None => draw_parameter_metadata = Some(draw.parameter_metadata),
        }
        match draw_artifact_metadata {
            Some(expected) if expected != draw.artifact_metadata => {
                return Err(invalid_fit(
                    "fit draw artifact metadata must be present on every draw line or omitted from every draw line; rerun `bayesite sample` to completion",
                ));
            }
            Some(_) => {}
            None => draw_artifact_metadata = Some(draw.artifact_metadata),
        }
        match draw_chain_metadata {
            Some(expected) if expected != draw.chain_metadata => {
                return Err(invalid_fit(
                    "fit draw chain metadata must be present on every draw line or omitted from every draw line; rerun `bayesite sample` to completion",
                ));
            }
            Some(_) => {}
            None => draw_chain_metadata = Some(draw.chain_metadata),
        }
        let expected_draw_index = draws.len();
        if let Some(draw_index) = draw.draw_index {
            if draw_index != expected_draw_index {
                return Err(invalid_fit(format!(
                    "draw line draw_index must be {expected_draw_index}, got {draw_index}; fit draw_index values must be contiguous from 0 in retained draw order"
                )));
            }
        }
        let chain = chain_index(&mut chain_ids, draw.chain);
        if next_draw_by_chain.len() < chain_ids.len() {
            next_draw_by_chain.resize(chain_ids.len(), 0);
        }
        let expected_draw = next_draw_by_chain[chain];
        if draw.draw != expected_draw {
            return Err(invalid_fit(format!(
                "draw index for chain {} must be {}, got {}; fit draw indexes must be contiguous from 0",
                draw.chain, expected_draw, draw.draw
            )));
        }
        next_draw_by_chain[chain] += 1;
        draws.push(draw);
    }

    let trailer = trailer.ok_or_else(|| {
        invalid_fit("fit is missing a trailer; rerun `bayesite sample` to completion")
    })?;
    validate_trailer_draws_format(&trailer)?;
    let source_draw_count = i64::try_from(draws.len()).map_err(|_| {
        invalid_fit(
            "fit draw count must be in 0..=9223372036854775807 because diagnostics report draw counts as JSON integers",
        )
    })?;
    for draw in &draws {
        if let Some(draw_count) = draw.draw_count_metadata {
            if draw_count != source_draw_count {
                return Err(invalid_fit(
                    "draw line draw_count must match retained draw line count; rerun `bayesite sample` to completion",
                ));
            }
        }
    }
    validate_optional_header_draw_count(&header, source_draw_count)?;
    validate_trailer_completion_metadata(
        &trailer,
        source_seed,
        header_chain_count,
        source_draw_count,
        header_draw_count,
        specs.len(),
    )?;
    validate_optional_parameter_order(
        &trailer,
        "fit trailer",
        &specs,
        "fit trailer parameter_order must match fit header params order; rerun `bayesite sample` to completion",
    )?;
    let trailer_workflow_phases = parse_workflow_phases(&trailer, "fit trailer")?;
    let trailer_artifact_kind = parse_sample_artifact_field(
        &trailer,
        "fit trailer",
        "artifact_kind",
        SAMPLE_ARTIFACT_KIND,
    )?;
    let trailer_artifact_scope = parse_sample_artifact_field(
        &trailer,
        "fit trailer",
        "artifact_scope",
        SAMPLE_ARTIFACT_SCOPE,
    )?;
    if let (Some(header_phases), Some(trailer_phases)) =
        (&header_workflow_phases, &trailer_workflow_phases)
    {
        if header_phases != trailer_phases {
            return Err(invalid_fit(
                "fit header and trailer workflow_phases must match; rerun `bayesite sample` to completion",
            ));
        }
    }
    let source_workflow_phases = header_workflow_phases.or(trailer_workflow_phases);
    let source_artifact_kind = header_artifact_kind.or(trailer_artifact_kind);
    let source_artifact_scope = header_artifact_scope.or(trailer_artifact_scope);
    let chain_stats = trailer
        .get("chains")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_fit("fit trailer needs a chains array from `bayesite sample`"))?;
    if draws.is_empty() {
        return Err(invalid_fit(
            "fit has no draw lines; rerun `bayesite sample` with at least 4 draws",
        ));
    }
    if chain_ids.len() != header_chain_count {
        return Err(invalid_fit(
            "fit header chains must match draw chain count; rerun `bayesite sample` to completion",
        ));
    }
    validate_optional_chain_order(
        &header,
        "fit header",
        &chain_ids,
        "fit header chain_order must match draw chain ids; rerun `bayesite sample` to completion",
    )?;
    validate_optional_chain_order(
        &trailer,
        "fit trailer",
        &chain_ids,
        "fit trailer chain_order must match draw chain ids; rerun `bayesite sample` to completion",
    )?;
    for draw in &draws {
        if let Some(chain_count) = draw.chain_count_metadata {
            if chain_count != chain_ids.len() as i64 {
                return Err(invalid_fit(
                    "draw line chain_count must match draw chain count; rerun `bayesite sample` to completion",
                ));
            }
        }
        if let Some(chain_order) = &draw.chain_order_metadata {
            if chain_order != &chain_ids {
                return Err(invalid_fit(
                    "draw line chain_order must match draw chain ids; rerun `bayesite sample` to completion",
                ));
            }
        }
    }
    validate_trailer_chains(chain_stats, &chain_ids, header_draw_count)?;

    let mut series_by_param: Vec<Vec<Vec<Vec<f64>>>> = specs
        .iter()
        .map(|spec| vec![vec![Vec::new(); chain_ids.len()]; spec.size])
        .collect();
    for draw in &draws {
        let chain = chain_ids
            .iter()
            .position(|&id| id == draw.chain)
            .expect("chain id was registered");
        for (param_idx, values) in draw.values.iter().enumerate() {
            for (coord, &value) in values.iter().enumerate() {
                series_by_param[param_idx][coord][chain].push(value);
            }
        }
    }

    let draws_per_chain = series_by_param[0][0][0].len();
    if draws_per_chain < 4 {
        return Err(invalid_fit(
            "diagnostics need at least 4 draws per chain; rerun `bayesite sample --draws 4` or more",
        ));
    }
    if draws_per_chain != header_draw_count {
        return Err(invalid_fit(
            "fit header settings.num_draws must match draw count per chain; rerun `bayesite sample` to completion",
        ));
    }
    for chain_series in &series_by_param[0][0] {
        let len = chain_series.len();
        if len != draws_per_chain {
            return Err(invalid_fit(
                "all chains must have the same number of draws for diagnostics",
            ));
        }
    }

    let mut rhat_entries = Vec::with_capacity(specs.len());
    let mut ess_entries = Vec::with_capacity(specs.len());
    for (param_idx, spec) in specs.iter().enumerate() {
        let mut worst_rhat = f64::NEG_INFINITY;
        let mut worst_ess = f64::INFINITY;
        for coord_series in &series_by_param[param_idx] {
            worst_rhat = worst_rhat.max(diagnostics::split_rhat(coord_series));
            worst_ess = worst_ess.min(diagnostics::effective_sample_size(coord_series));
        }
        rhat_entries.push((spec.name.clone(), Value::Float(worst_rhat)));
        ess_entries.push((spec.name.clone(), Value::Float(worst_ess)));
    }

    let mut response_entries = vec![
        (
            "diagnostics_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "workflow_phases".to_string(),
            diagnose_workflow_phases_value(),
        ),
        (
            "source_draws_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "rhat_statistic".to_string(),
            Value::Str(RHAT_STATISTIC.to_string()),
        ),
        (
            "rhat_scope".to_string(),
            Value::Str("max_over_parameter_coordinate_marginals".to_string()),
        ),
        (
            "ess_statistic".to_string(),
            Value::Str(ESS_STATISTIC.to_string()),
        ),
        (
            "ess_scope".to_string(),
            Value::Str("min_over_parameter_coordinate_marginals".to_string()),
        ),
    ];
    if let Some(kind) = source_artifact_kind {
        response_entries.push(("source_artifact_kind".to_string(), Value::Str(kind)));
    }
    if let Some(scope) = source_artifact_scope {
        response_entries.push(("source_artifact_scope".to_string(), Value::Str(scope)));
    }
    response_entries.extend([
        ("source_seed".to_string(), Value::Int(source_seed)),
        (
            "source_chains".to_string(),
            Value::Int(header_chain_count as i64),
        ),
        (
            "source_chain_count".to_string(),
            Value::Int(header_chain_count as i64),
        ),
        (
            "source_chain_order".to_string(),
            source_chain_order_value(&chain_ids),
        ),
        (
            "source_draw_count".to_string(),
            Value::Int(source_draw_count),
        ),
        (
            "source_draw_index_metadata".to_string(),
            Value::Bool(draw_index_metadata.unwrap_or(false)),
        ),
        (
            "source_draw_parameter_metadata".to_string(),
            Value::Bool(draw_parameter_metadata.unwrap_or(false)),
        ),
        (
            "source_draw_artifact_metadata".to_string(),
            Value::Bool(draw_artifact_metadata.unwrap_or(false)),
        ),
        (
            "source_draw_chain_metadata".to_string(),
            Value::Bool(draw_chain_metadata.unwrap_or(false)),
        ),
        (
            "source_parameter_count".to_string(),
            Value::Int(specs.len() as i64),
        ),
        ("source_settings".to_string(), source_settings),
        ("source_params".to_string(), param_specs_value(&specs)),
        ("source_packing".to_string(), packing_value(&specs)),
        ("source_parameter_order".to_string(), packing_value(&specs)),
        (
            "source_trailer_completion_metadata".to_string(),
            trailer_completion_metadata_value(&trailer),
        ),
        (
            "source_workflow_phases".to_string(),
            source_workflow_phases
                .as_deref()
                .map(workflow_phases_array)
                .unwrap_or_else(|| Value::Array(vec![])),
        ),
        (
            "draws_per_chain".to_string(),
            Value::Int(draws_per_chain as i64),
        ),
        ("chains".to_string(), Value::Array(chain_stats.to_vec())),
        ("rhat".to_string(), Value::Object(rhat_entries)),
        ("ess".to_string(), Value::Object(ess_entries)),
    ]);
    let response = Value::Object(response_entries);
    json::write(&response)
}

/// Handle one wasm-boundary request (a JSON document) and render the
/// response text. Pure string-to-string so it is natively testable; the
/// unsafe pointer shims in `wasm_abi.rs` only move bytes.
///
/// Commands:
/// - `{"command":"sample","model":<ir>,"data":<data>,"settings":{...},
///    "seed":N,"chain_id":N}` -> v0-provisional NDJSON (one chain).
/// - `{"command":"diagnose","fit":"<v0-provisional NDJSON>"}`
///   -> v0-provisional JSON diagnostics.
/// - `{"command":"prior-predictive","model":<ir>,"data":<data>,
///    "settings":{"num_draws":N},"seed":N}` -> v0-provisional NDJSON.
/// - `{"command":"recover","model":<ir>,"data":<data>,"settings":{...},
///    "seed":N}` -> v0-provisional JSON report.
/// - `{"command":"sbc","model":<ir>,"data":<data>,"settings":{...},
///    "seed":N}` -> v0-provisional JSON report.
/// - `{"command":"diagnostics","series":[[...],...]}` -> cross-chain
///   `{"rhat":x,"ess":y}` for one scalar coordinate.
///
/// Request and settings objects reject unknown or duplicate fields so misspelled
/// or ambiguous keys do not silently fall back to defaults. Model and data
/// documents keep their own IR/data validation.
///
/// Errors come back as a single v0-provisional JSON repair object.
pub fn handle_request(text: &str) -> String {
    match handle_request_inner(text) {
        Ok(response) => response,
        Err(error) => {
            let payload = Value::Object(vec![
                (
                    "error_format".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
                (
                    "error".to_string(),
                    Value::Str(error.kind.name().to_string()),
                ),
                ("message".to_string(), Value::Str(error.message)),
            ]);
            json::write(&payload).unwrap_or_else(|_| "{\"error\":\"MalformedJson\"}".to_string())
        }
    }
}

fn invalid_request(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn request_model_data(
    request: &Value,
    context: &str,
) -> Result<(ModelMeta, Vec<(String, DataValue)>), Error> {
    let model = request.get("model").ok_or_else(|| {
        invalid_request(format!("{context} request needs a \"model\" IR document"))
    })?;
    let meta = decode_model(model)?;
    let data_doc = request
        .get("data")
        .ok_or_else(|| invalid_request(format!("{context} request needs a \"data\" object")))?;
    if !matches!(data_doc, Value::Object(_)) {
        return Err(invalid_request(format!(
            "{context} request data must be an object"
        )));
    }
    let data = data_from_json(data_doc)?;
    Ok((meta, data))
}

fn request_settings<'a>(request: &'a Value, context: &str) -> Result<Option<&'a Value>, Error> {
    let Some(settings) = request.get("settings") else {
        return Ok(None);
    };
    if !matches!(settings, Value::Object(_)) {
        return Err(invalid_request(format!(
            "{context} request settings must be an object"
        )));
    }
    Ok(Some(settings))
}

fn reject_unknown_fields(doc: &Value, context: &str, allowed: &[&str]) -> Result<(), Error> {
    let Value::Object(entries) = doc else {
        return Err(invalid_request(format!("{context} must be an object")));
    };
    for (index, (name, _)) in entries.iter().enumerate() {
        if entries[..index]
            .iter()
            .any(|(existing, _)| existing == name)
        {
            return Err(invalid_request(format!(
                "{context} has duplicate field \"{name}\"; remove one"
            )));
        }
        if !allowed.contains(&name.as_str()) {
            return Err(invalid_request(format!(
                "{context} has unknown field \"{name}\""
            )));
        }
    }
    Ok(())
}

fn reject_unknown_settings_fields(
    settings: Option<&Value>,
    context: &str,
    allowed: &[&str],
) -> Result<(), Error> {
    if let Some(settings) = settings {
        reject_unknown_fields(settings, &format!("{context} request settings"), allowed)?;
    }
    Ok(())
}

fn validate_diagnostics_series(series: &[Vec<f64>]) -> Result<(), Error> {
    let Some(first) = series.first() else {
        return Err(invalid_request(
            "series needs at least one chain of at least 4 draws",
        ));
    };
    if first.len() < 4 {
        return Err(invalid_request(
            "series needs at least one chain of at least 4 draws",
        ));
    }
    for chain in series {
        if chain.len() < 4 {
            return Err(invalid_request("each series chain needs at least 4 draws"));
        }
        if chain.len() != first.len() {
            return Err(invalid_request(
                "series chains must all have the same number of draws",
            ));
        }
    }
    Ok(())
}

fn request_seed(request: &Value, context: &str) -> Result<u64, Error> {
    let Some(value) = request.get("seed") else {
        return Err(invalid_request(format!(
            "{context} request needs an integer \"seed\""
        )));
    };
    match value {
        Value::Int(seed) if *seed >= 0 => Ok(*seed as u64),
        Value::Int(_) => Err(invalid_request(format!(
            "{context} request seed must be non-negative"
        ))),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(format!(
            "{context} request seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers"
        ))),
        _ => Err(invalid_request(format!(
            "{context} request needs an integer \"seed\""
        ))),
    }
}

fn setting_reportable_draws(
    settings: Option<&Value>,
    default: i64,
    context: &str,
) -> Result<i64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get("num_draws") else {
        return Ok(default);
    };
    match value {
        Value::Int(draws) => Ok(*draws),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(format!(
            "{context} request settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers"
        ))),
        _ => Err(invalid_request(format!(
            "{context} request settings.num_draws must be an integer"
        ))),
    }
}

fn setting_reportable_warmup(
    settings: Option<&Value>,
    default: i64,
    context: &str,
) -> Result<i64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get("num_warmup") else {
        return Ok(default);
    };
    match value {
        Value::Int(warmup) => Ok(*warmup),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(format!(
            "{context} request settings.num_warmup must be in 0..=9223372036854775807 because artifacts report warmup counts as JSON integers"
        ))),
        _ => Err(invalid_request(format!(
            "{context} request settings.num_warmup must be an integer"
        ))),
    }
}

fn setting_reportable_chains(
    settings: Option<&Value>,
    default: i64,
    context: &str,
) -> Result<i64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get("chains") else {
        return Ok(default);
    };
    match value {
        Value::Int(chains) => Ok(*chains),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(format!(
            "{context} request settings.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
        ))),
        _ => Err(invalid_request(format!(
            "{context} request settings.chains must be an integer"
        ))),
    }
}

fn setting_reportable_replicates(settings: Option<&Value>, default: i64) -> Result<i64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get("replicates") else {
        return Ok(default);
    };
    match value {
        Value::Int(replicates) => Ok(*replicates),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(
            "sbc request settings.replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers",
        )),
        _ => Err(invalid_request(
            "sbc request settings.replicates must be an integer",
        )),
    }
}

fn setting_bounded_treedepth(
    settings: Option<&Value>,
    default: i64,
    context: &str,
) -> Result<i64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get("max_treedepth") else {
        return Ok(default);
    };
    match value {
        Value::Int(max_treedepth) => Ok(*max_treedepth),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(invalid_request(format!(
            "{context} request settings.max_treedepth must be in 1..=20"
        ))),
        _ => Err(invalid_request(format!(
            "{context} request settings.max_treedepth must be an integer"
        ))),
    }
}

fn setting_f64(
    settings: Option<&Value>,
    name: &str,
    default: f64,
    context: &str,
) -> Result<f64, Error> {
    let Some(settings) = settings else {
        return Ok(default);
    };
    let Some(value) = settings.get(name) else {
        return Ok(default);
    };
    value.as_f64().ok_or_else(|| {
        invalid_request(format!(
            "{context} request settings.{name} must be a number"
        ))
    })
}

fn positive_usize(value: i64, name: &str) -> Result<usize, Error> {
    if value < 1 {
        Err(invalid_request(format!("{name} must be at least 1")))
    } else {
        usize::try_from(value)
            .map_err(|_| invalid_request(format!("{name} must fit this build's usize")))
    }
}

fn diagnostic_artifact_draws(value: usize, name: &str) -> Result<(), Error> {
    if value >= 4 {
        Ok(())
    } else {
        Err(invalid_request(format!(
            "{name} must be at least 4 because artifacts include diagnostics"
        )))
    }
}

fn validate_max_treedepth(value: usize, name: &str) -> Result<(), Error> {
    if (1..=20).contains(&value) {
        Ok(())
    } else {
        Err(invalid_request(format!("{name} must be in 1..=20")))
    }
}

fn apply_sampler_settings(
    settings_doc: Option<&Value>,
    sampler: &mut Settings,
    context: &str,
) -> Result<(), Error> {
    let warmup = setting_reportable_warmup(settings_doc, sampler.num_warmup as i64, context)?;
    if warmup < 0 {
        return Err(invalid_request(format!(
            "{context} request settings.num_warmup must be non-negative"
        )));
    }
    sampler.num_warmup = warmup as usize;
    sampler.num_draws = positive_usize(
        setting_reportable_draws(settings_doc, sampler.num_draws as i64, context)?,
        &format!("{context} request settings.num_draws"),
    )?;
    diagnostic_artifact_draws(
        sampler.num_draws,
        &format!("{context} request settings.num_draws"),
    )?;
    sampler.max_treedepth = positive_usize(
        setting_bounded_treedepth(settings_doc, sampler.max_treedepth as i64, context)?,
        &format!("{context} request settings.max_treedepth"),
    )?;
    validate_max_treedepth(
        sampler.max_treedepth,
        &format!("{context} request settings.max_treedepth"),
    )?;
    sampler.target_accept = setting_f64(
        settings_doc,
        "target_accept",
        sampler.target_accept,
        context,
    )?;
    if !(0.0..1.0).contains(&sampler.target_accept) {
        return Err(invalid_request(format!(
            "{context} request settings.target_accept must be in (0, 1)"
        )));
    }
    Ok(())
}

fn request_chain_id(request: &Value) -> Result<u64, Error> {
    let Some(value) = request.get("chain_id") else {
        return Ok(0);
    };
    let chain_id = value
        .as_i64()
        .ok_or_else(|| invalid_request("sample request chain_id must be an integer"))?;
    if chain_id < 0 {
        return Err(invalid_request(
            "sample request chain_id must be non-negative",
        ));
    }
    Ok(chain_id as u64)
}

fn handle_request_inner(text: &str) -> Result<String, Error> {
    let request = json::parse(text)?;
    if !matches!(request, Value::Object(_)) {
        return Err(invalid_request("request must be an object"));
    }
    let command = match request.get("command") {
        Some(Value::Str(command)) => Some(command.as_str()),
        Some(_) => return Err(invalid_request("request command must be a string")),
        None => None,
    };
    match command {
        Some("sample") => {
            reject_unknown_fields(
                &request,
                "sample request",
                &["command", "model", "data", "settings", "seed", "chain_id"],
            )?;
            let (meta, data) = request_model_data(&request, "sample")?;
            let posterior = Posterior::new(meta, data)?;
            let mut settings = Settings::default();
            let settings_doc = request_settings(&request, "sample")?;
            reject_unknown_settings_fields(
                settings_doc,
                "sample",
                &["num_warmup", "num_draws", "max_treedepth", "target_accept"],
            )?;
            apply_sampler_settings(settings_doc, &mut settings, "sample")?;
            let seed = request_seed(&request, "sample")?;
            let chain_id = request_chain_id(&request)?;
            let draws = sample(&posterior, &settings, seed, chain_id)?;
            let lines = ndjson_lines(&posterior, &settings, seed, &[(chain_id, draws)])?;
            Ok(lines.join("\n"))
        }
        Some("diagnose") => {
            reject_unknown_fields(&request, "diagnose request", &["command", "fit"])?;
            let fit = request
                .get("fit")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    invalid_request("diagnose request needs \"fit\": a v0-provisional NDJSON string")
                })?;
            diagnose_ndjson(fit)
        }
        Some("prior-predictive") => {
            reject_unknown_fields(
                &request,
                "prior-predictive request",
                &["command", "model", "data", "settings", "seed"],
            )?;
            let (meta, data) = request_model_data(&request, "prior-predictive")?;
            let settings_doc = request_settings(&request, "prior-predictive")?;
            reject_unknown_settings_fields(settings_doc, "prior-predictive", &["num_draws"])?;
            let num_draws = positive_usize(
                setting_reportable_draws(
                    settings_doc,
                    PriorPredictiveSettings::default().num_draws as i64,
                    "prior-predictive",
                )?,
                "prior-predictive request settings.num_draws",
            )?;
            let settings = PriorPredictiveSettings { num_draws };
            let seed = request_seed(&request, "prior-predictive")?;
            let lines = prior_predictive_ndjson_lines(meta, data, &settings, seed)?;
            Ok(lines.join("\n"))
        }
        Some("recover") => {
            reject_unknown_fields(
                &request,
                "recover request",
                &["command", "model", "data", "settings", "seed"],
            )?;
            let (meta, data) = request_model_data(&request, "recover")?;
            let settings_doc = request_settings(&request, "recover")?;
            reject_unknown_settings_fields(
                settings_doc,
                "recover",
                &[
                    "chains",
                    "interval",
                    "num_warmup",
                    "num_draws",
                    "max_treedepth",
                    "target_accept",
                ],
            )?;
            let mut settings = RecoverSettings::default();
            settings.chains = positive_usize(
                setting_reportable_chains(settings_doc, settings.chains as i64, "recover")?,
                "recover request settings.chains",
            )? as u64;
            settings.interval =
                setting_f64(settings_doc, "interval", settings.interval, "recover")?;
            if !(0.0..1.0).contains(&settings.interval) {
                return Err(invalid_request(
                    "recover request settings.interval must be in (0, 1)",
                ));
            }
            apply_sampler_settings(settings_doc, &mut settings.sampler, "recover")?;
            let seed = request_seed(&request, "recover")?;
            recover_report(meta, data, &settings, seed)
        }
        Some("sbc") => {
            reject_unknown_fields(
                &request,
                "sbc request",
                &["command", "model", "data", "settings", "seed"],
            )?;
            let (meta, data) = request_model_data(&request, "sbc")?;
            let settings_doc = request_settings(&request, "sbc")?;
            reject_unknown_settings_fields(
                settings_doc,
                "sbc",
                &[
                    "replicates",
                    "chains",
                    "num_warmup",
                    "num_draws",
                    "max_treedepth",
                    "target_accept",
                ],
            )?;
            let mut settings = SbcSettings::default();
            settings.replicates = positive_usize(
                setting_reportable_replicates(settings_doc, settings.replicates as i64)?,
                "sbc request settings.replicates",
            )?;
            settings.chains = positive_usize(
                setting_reportable_chains(settings_doc, settings.chains as i64, "sbc")?,
                "sbc request settings.chains",
            )? as u64;
            apply_sampler_settings(settings_doc, &mut settings.sampler, "sbc")?;
            let seed = request_seed(&request, "sbc")?;
            sbc_report(meta, data, &settings, seed)
        }
        Some("diagnostics") => {
            reject_unknown_fields(&request, "diagnostics request", &["command", "series"])?;
            let series: Vec<Vec<f64>> = request
                .get("series")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid_request("diagnostics request needs \"series\": an array of chains")
                })?
                .iter()
                .map(|chain| {
                    chain
                        .as_array()
                        .ok_or_else(|| {
                            invalid_request("each series entry must be an array of numbers")
                        })?
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| invalid_request("series values must be numbers"))
                        })
                        .collect()
                })
                .collect::<Result<_, Error>>()?;
            validate_diagnostics_series(&series)?;
            let response = Value::Object(vec![
                (
                    "rhat".to_string(),
                    Value::Float(diagnostics::split_rhat(&series)),
                ),
                (
                    "ess".to_string(),
                    Value::Float(diagnostics::effective_sample_size(&series)),
                ),
            ]);
            json::write(&response)
        }
        Some(command) => Err(invalid_request(format!(
            "unknown command \"{command}\"; supported commands are \"sample\", \"diagnose\", \"diagnostics\", \"prior-predictive\", \"recover\", and \"sbc\""
        ))),
        None => Err(invalid_request(
            "request needs \"command\": \"sample\", \"diagnose\", \"diagnostics\", \"prior-predictive\", \"recover\", or \"sbc\"",
        )),
    }
}
