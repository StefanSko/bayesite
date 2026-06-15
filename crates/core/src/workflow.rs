//! Higher-level workflow reports over the pure runtime phases.
//!
//! These helpers compose existing primitives without adding hidden I/O or
//! entropy: prior predictive simulation, posterior binding, NUTS, and JSON
//! artifact rendering remain explicit inputs and outputs.

use std::collections::HashMap;

use crate::diagnostics;
use crate::error::{Error, ErrorKind};
use crate::ir::ModelMeta;
use crate::json::{self, Value};
use crate::model::{DataValue, Posterior};
use crate::predictive::{simulate_prior_predictive, PriorPredictiveRole, PriorPredictiveSettings};
use crate::sampler::{sample, ChainDraws, Settings};
use crate::tensor::Tensor;

const PARAMETER_SUMMARY_SCALE: &str = "constrained_parameter_value";
const WORKFLOW_FORMAT: &str = "v0-provisional";
const PRIOR_PREDICTIVE_DRAWS: i64 = 1;
const PRIOR_PREDICTIVE_DRAW_INDEX: i64 = 0;
const PRIOR_PREDICTIVE_DRAW_INDEX_BASE: &str = "zero_based_prior_predictive_draw_order";
const PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND: &str = "prior_predictive_draws";
const PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE: &str = "declared_data_conditioned_site_draws";
const POSTERIOR_DRAWS_ARTIFACT_KIND: &str = "posterior_draws";
const POSTERIOR_DRAWS_ARTIFACT_SCOPE: &str = "observed_data_conditioned_parameter_draws";
const REPLICATE_INDEX_BASE: &str = "zero_based_replicate_order";
const RHAT_STATISTIC: &str = "split_rhat";
const SIMULATION_INDEX_BASE: &str = "zero_based_simulation_order";
const ESS_STATISTIC: &str = "effective_sample_size_geyer_initial_monotone_sequence";
const CHAIN_INDEX_BASE: &str = "zero_based_chain_id";

#[derive(Clone, Copy)]
struct RankSupport {
    draws: usize,
    bin_count: usize,
}

#[derive(Clone, Copy)]
struct RecoverParamContext {
    interval: f64,
    rank_support: RankSupport,
    simulation_index: i64,
    prior_seed: u64,
    sample_seed: u64,
}

#[derive(Debug, Clone)]
pub struct RecoverSettings {
    pub chains: u64,
    pub sampler: Settings,
    pub interval: f64,
}

impl Default for RecoverSettings {
    fn default() -> Self {
        Self {
            chains: 4,
            sampler: Settings::default(),
            interval: 0.8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SbcSettings {
    pub replicates: usize,
    pub chains: u64,
    pub sampler: Settings,
}

impl Default for SbcSettings {
    fn default() -> Self {
        Self {
            replicates: 100,
            chains: 4,
            sampler: Settings::default(),
        }
    }
}

struct GeneratedFit {
    prior_seed: u64,
    sample_seed: u64,
    truth: HashMap<String, Tensor>,
    truth_stochastic_sites: HashMap<String, String>,
    generated_observed: Vec<(String, Value)>,
    generated_observed_stochastic_sites: Vec<(String, Value)>,
    generated_observed_shapes: Vec<(String, Value)>,
    generated_observed_coordinate_order: Vec<(String, Value)>,
    generated_observed_integer: Vec<(String, Value)>,
    generated_observed_integer_by_coordinate: Vec<(String, Value)>,
    chains: Vec<(u64, ChainDraws)>,
    packing: Vec<(String, Vec<usize>)>,
    constrained_chains: Vec<Vec<Vec<(String, Tensor)>>>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn validate_reportable_seed(seed: u64, context: &str) -> Result<(), Error> {
    if seed <= i64::MAX as u64 {
        Ok(())
    } else {
        Err(invalid(format!(
            "{context} seed must be in 0..=9223372036854775807 because workflow reports seeds as JSON integers"
        )))
    }
}

fn validate_workflow_chains(chains: u64, context: &str) -> Result<(), Error> {
    if chains == 0 {
        return Err(invalid(format!(
            "{context} sample.chains must be at least 1"
        )));
    }
    if chains > i64::MAX as u64 {
        return Err(invalid(format!(
            "{context} sample.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
        )));
    }
    Ok(())
}

fn validate_sbc_replicates(replicates: usize) -> Result<(), Error> {
    if replicates == 0 {
        return Err(invalid("sbc replicates must be at least 1"));
    }
    if replicates > i64::MAX as usize {
        return Err(invalid(
            "sbc replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers",
        ));
    }
    Ok(())
}

fn validate_sbc_seed_span(seed: u64, replicates: usize) -> Result<(), Error> {
    let last_replicate = (replicates - 1) as u64;
    let Some(last_prior_offset) = last_replicate.checked_mul(2) else {
        return Err(invalid(
            "sbc seed and replicate count must produce reportable derived seeds",
        ));
    };
    let Some(last_sample_offset) = last_prior_offset.checked_add(1) else {
        return Err(invalid(
            "sbc seed and replicate count must produce reportable derived seeds",
        ));
    };
    let Some(last_sample_seed) = seed.checked_add(last_sample_offset) else {
        return Err(invalid(
            "sbc seed and replicate count must produce reportable derived seeds",
        ));
    };
    if last_sample_seed > i64::MAX as u64 {
        return Err(invalid(
            "sbc seed and replicate count must produce reportable derived seeds",
        ));
    }
    Ok(())
}

fn sbc_rank_draws(chains: u64, num_draws: usize) -> Result<usize, Error> {
    let rank_draws = (chains as u128) * (num_draws as u128);
    if rank_draws == 0 || rank_draws > i64::MAX as u128 {
        return Err(invalid(
            "sbc rank_draws must be in 1..=9223372036854775807 because workflow reports rank_draws as a JSON integer; reduce sample.chains or sample.draws",
        ));
    }
    usize::try_from(rank_draws).map_err(|_| {
        invalid(
            "sbc rank_draws must be in 1..=9223372036854775807 because workflow reports rank_draws as a JSON integer; reduce sample.chains or sample.draws",
        )
    })
}

fn workflow_rank_bin_count(rank_draws: usize, context: &str) -> Result<usize, Error> {
    let Some(rank_bin_count) = rank_draws.checked_add(1) else {
        return Err(invalid(format!(
            "{context} rank_bin_count must be in 1..=9223372036854775807 because workflow reports rank_bin_count as a JSON integer; reduce sample.chains or sample.draws"
        )));
    };
    if rank_bin_count > i64::MAX as usize {
        return Err(invalid(format!(
            "{context} rank_bin_count must be in 1..=9223372036854775807 because workflow reports rank_bin_count as a JSON integer; reduce sample.chains or sample.draws"
        )));
    }
    Ok(rank_bin_count)
}

fn recover_posterior_draws(chains: u64, num_draws: usize) -> Result<usize, Error> {
    let posterior_draws = (chains as u128) * (num_draws as u128);
    if posterior_draws == 0 || posterior_draws > i64::MAX as u128 {
        return Err(invalid(
            "recover posterior_draws must be in 1..=9223372036854775807 because workflow reports posterior_draws as a JSON integer; reduce sample.chains or sample.draws",
        ));
    }
    usize::try_from(posterior_draws).map_err(|_| {
        invalid(
            "recover posterior_draws must be in 1..=9223372036854775807 because workflow reports posterior_draws as a JSON integer; reduce sample.chains or sample.draws",
        )
    })
}

fn derived_seed(seed: u64, offset: u64, context: &str) -> Result<u64, Error> {
    let Some(value) = seed.checked_add(offset) else {
        return Err(invalid(format!(
            "{context} seed and replicate count must produce reportable derived seeds"
        )));
    };
    validate_reportable_seed(value, context)?;
    Ok(value)
}

fn validate_workflow_draws(num_draws: usize, context: &str) -> Result<(), Error> {
    if num_draws >= 4 {
        Ok(())
    } else {
        Err(invalid(format!(
            "{context} sample.draws must be at least 4 because workflow reports include diagnostics"
        )))
    }
}

fn validate_workflow_sampler_counts(sampler: &Settings, context: &str) -> Result<(), Error> {
    if sampler.num_draws > i64::MAX as usize {
        return Err(invalid(format!(
            "{context} sample.draws must be in 1..=9223372036854775807 because workflow reports sample.draws as a JSON integer"
        )));
    }
    if sampler.num_warmup > i64::MAX as usize {
        return Err(invalid(format!(
            "{context} sample.warmup must be in 0..=9223372036854775807 because workflow reports sample.warmup as a JSON integer"
        )));
    }
    Ok(())
}

fn validate_workflow_treedepth(max_treedepth: usize, context: &str) -> Result<(), Error> {
    if (1..=20).contains(&max_treedepth) {
        Ok(())
    } else {
        Err(invalid(format!(
            "{context} sample.max_treedepth must be in 1..=20"
        )))
    }
}

fn tensor_to_value(tensor: &Tensor) -> Value {
    if tensor.shape().is_empty() {
        Value::Float(tensor.data()[0])
    } else {
        Value::Array(tensor.data().iter().map(|&v| Value::Float(v)).collect())
    }
}

fn scalar_to_value(value: f64, integer: bool, context: &str) -> Result<Value, Error> {
    if integer {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(invalid(format!(
                "{context} value must be integer and finite, got {value}"
            )));
        }
        if value < i64::MIN as f64 || value > i64::MAX as f64 {
            return Err(invalid(format!(
                "{context} integer value must fit JSON integer range, got {value}"
            )));
        }
        Ok(Value::Int(value as i64))
    } else {
        Ok(Value::Float(value))
    }
}

fn generated_observed_value(
    tensor: &Tensor,
    integer_flags: &[bool],
    context: &str,
) -> Result<Value, Error> {
    if integer_flags.len() != tensor.data().len() {
        return Err(invalid(format!(
            "{context} integer metadata length must match value length"
        )));
    }
    if tensor.shape().is_empty() {
        scalar_to_value(tensor.data()[0], integer_flags[0], context)
    } else {
        Ok(Value::Array(
            tensor
                .data()
                .iter()
                .zip(integer_flags)
                .map(|(&value, &integer)| scalar_to_value(value, integer, context))
                .collect::<Result<Vec<_>, Error>>()?,
        ))
    }
}

fn slice_to_value(shape: &[usize], data: &[f64]) -> Value {
    if shape.is_empty() {
        Value::Float(data[0])
    } else {
        Value::Array(data.iter().map(|&v| Value::Float(v)).collect())
    }
}

fn shape_value(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|&dim| Value::Int(dim as i64)).collect())
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

fn name_order_value(names: &[String]) -> Value {
    Value::Array(names.iter().map(|name| Value::Str(name.clone())).collect())
}

fn entry_order_value(entries: &[(String, Value)]) -> Value {
    Value::Array(
        entries
            .iter()
            .map(|(name, _)| Value::Str(name.clone()))
            .collect(),
    )
}

fn declared_data_order_value(declared_data: &[(String, DataValue)]) -> Value {
    Value::Array(
        declared_data
            .iter()
            .map(|(name, _)| Value::Str(name.clone()))
            .collect(),
    )
}

fn declared_data_value(value: &DataValue, context: &str) -> Result<Value, Error> {
    if value.shape.is_empty() {
        scalar_to_value(value.values[0], value.integer, context)
    } else {
        Ok(Value::Array(
            value
                .values
                .iter()
                .map(|&entry| scalar_to_value(entry, value.integer, context))
                .collect::<Result<Vec<_>, Error>>()?,
        ))
    }
}

fn declared_data_values_value(
    declared_data: &[(String, DataValue)],
    context: &str,
) -> Result<Value, Error> {
    let mut entries = Vec::with_capacity(declared_data.len());
    for (name, value) in declared_data {
        let context = format!("{context} declared data {name:?}");
        entries.push((name.clone(), declared_data_value(value, &context)?));
    }
    Ok(Value::Object(entries))
}

fn declared_data_shapes_value(declared_data: &[(String, DataValue)]) -> Value {
    Value::Object(
        declared_data
            .iter()
            .map(|(name, value)| (name.clone(), shape_value(&value.shape)))
            .collect(),
    )
}

fn declared_data_coordinate_order_value(declared_data: &[(String, DataValue)]) -> Value {
    Value::Object(
        declared_data
            .iter()
            .map(|(name, value)| (name.clone(), coordinate_order_value(&value.shape)))
            .collect(),
    )
}

fn declared_data_integer_value(declared_data: &[(String, DataValue)]) -> Value {
    Value::Object(
        declared_data
            .iter()
            .map(|(name, value)| (name.clone(), Value::Bool(value.integer)))
            .collect(),
    )
}

fn declared_data_integer_by_coordinate_entry(value: &DataValue) -> Value {
    if value.shape.is_empty() {
        Value::Bool(value.integer)
    } else {
        Value::Array(
            value
                .values
                .iter()
                .map(|_| Value::Bool(value.integer))
                .collect(),
        )
    }
}

fn declared_data_integer_by_coordinate_value(declared_data: &[(String, DataValue)]) -> Value {
    Value::Object(
        declared_data
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    declared_data_integer_by_coordinate_entry(value),
                )
            })
            .collect(),
    )
}

fn value_is_integer(tensor: &Tensor) -> bool {
    tensor.data().iter().all(|&value| value.fract() == 0.0)
}

fn tensor_integer_value(tensor: &Tensor) -> Value {
    if tensor.shape().is_empty() {
        Value::Bool(tensor.data()[0].fract() == 0.0)
    } else {
        Value::Array(
            tensor
                .data()
                .iter()
                .map(|&value| Value::Bool(value.fract() == 0.0))
                .collect(),
        )
    }
}

fn integer_flags_value(shape: &[usize], flags: &[bool], context: &str) -> Result<Value, Error> {
    let expected = shape.iter().product::<usize>().max(1);
    if flags.len() != expected {
        return Err(invalid(format!(
            "{context} integer_by_coordinate length must match generated value shape"
        )));
    }
    if shape.is_empty() {
        Ok(Value::Bool(flags[0]))
    } else {
        Ok(Value::Array(
            flags.iter().copied().map(Value::Bool).collect(),
        ))
    }
}

fn quantile(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let weight = pos - lo as f64;
        sorted[lo] * (1.0 - weight) + sorted[hi] * weight
    }
}

fn quantile_index_value(p: f64, draw_count: usize) -> Value {
    debug_assert!(draw_count > 0);
    let position = if draw_count == 1 {
        0.0
    } else {
        p.clamp(0.0, 1.0) * (draw_count - 1) as f64
    };
    Value::Object(vec![
        ("position".to_string(), Value::Float(position)),
        ("floor".to_string(), Value::Int(position.floor() as i64)),
        ("ceil".to_string(), Value::Int(position.ceil() as i64)),
    ])
}

fn summarize_param(
    name: &str,
    stochastic_site: &str,
    shape: &[usize],
    truth: &Tensor,
    chain_values: &[Vec<Vec<f64>>],
    context: RecoverParamContext,
) -> Value {
    let size = shape.iter().product::<usize>().max(1);
    let lower_p = (1.0 - context.interval) / 2.0;
    let upper_p = 1.0 - lower_p;
    let mut means = vec![0.0; size];
    let mut lowers = vec![0.0; size];
    let mut uppers = vec![0.0; size];
    let mut ranks = Vec::with_capacity(size);
    let mut tie_counts = Vec::with_capacity(size);
    let mut contains_by_coord = Vec::with_capacity(size);
    let mut worst_rhat = f64::NEG_INFINITY;
    let mut worst_ess = f64::INFINITY;

    for coord in 0..size {
        let mut pooled = Vec::new();
        for chain in chain_values {
            pooled.extend(chain.iter().map(|draw| draw[coord]));
        }
        let mean = pooled.iter().sum::<f64>() / pooled.len() as f64;
        let truth_value = truth.data()[coord];
        let mut rank = 0i64;
        let mut ties = 0i64;
        for &draw in &pooled {
            if draw < truth_value {
                rank += 1;
            } else if draw == truth_value {
                ties += 1;
            }
        }
        pooled.sort_by(|a, b| a.total_cmp(b));
        let lower = quantile(&pooled, lower_p);
        let upper = quantile(&pooled, upper_p);
        means[coord] = mean;
        lowers[coord] = lower;
        uppers[coord] = upper;
        ranks.push(rank);
        tie_counts.push(ties);
        contains_by_coord.push(Value::Bool(truth_value >= lower && truth_value <= upper));
        let series: Vec<Vec<f64>> = chain_values
            .iter()
            .map(|chain| chain.iter().map(|draw| draw[coord]).collect())
            .collect();
        worst_rhat = worst_rhat.max(diagnostics::split_rhat(&series));
        worst_ess = worst_ess.min(diagnostics::effective_sample_size(&series));
    }

    let interval_contains_truth = contains_by_coord
        .iter()
        .all(|value| matches!(value, Value::Bool(true)));
    Value::Object(vec![
        ("shape".to_string(), shape_value(shape)),
        (
            "coordinate_order".to_string(),
            coordinate_order_value(shape),
        ),
        (
            "stochastic_site".to_string(),
            Value::Str(stochastic_site.to_string()),
        ),
        ("truth".to_string(), tensor_to_value(truth)),
        ("truth_integer".to_string(), tensor_integer_value(truth)),
        (
            "truth_artifact_kind".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "truth_artifact_scope".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "truth_draw_index".to_string(),
            Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX),
        ),
        (
            "truth_draw_index_base".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
        ),
        (
            "simulation".to_string(),
            Value::Int(context.simulation_index),
        ),
        (
            "simulation_index_base".to_string(),
            Value::Str(SIMULATION_INDEX_BASE.to_string()),
        ),
        (
            "prior_seed".to_string(),
            Value::Int(context.prior_seed as i64),
        ),
        (
            "sample_seed".to_string(),
            Value::Int(context.sample_seed as i64),
        ),
        ("seed_schedule".to_string(), recover_seed_schedule_value()),
        (
            "rank_draws".to_string(),
            Value::Int(context.rank_support.draws as i64),
        ),
        (
            "posterior_draws".to_string(),
            Value::Int(context.rank_support.draws as i64),
        ),
        (
            "posterior_draws_artifact_kind".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "posterior_draws_artifact_scope".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "rank_bounds".to_string(),
            rank_bounds_value(context.rank_support.draws),
        ),
        (
            "rank_bin_order".to_string(),
            rank_bin_order_value(context.rank_support.draws),
        ),
        (
            "rank_bin_count".to_string(),
            Value::Int(context.rank_support.bin_count as i64),
        ),
        ("rank".to_string(), int_coord_value(shape, &ranks)),
        ("tie_count".to_string(), int_coord_value(shape, &tie_counts)),
        ("mean".to_string(), slice_to_value(shape, &means)),
        (
            "interval_method".to_string(),
            Value::Str("equal_tailed_linear_quantile".to_string()),
        ),
        (
            "interval_scope".to_string(),
            Value::Str("per_parameter_coordinate_marginal".to_string()),
        ),
        (
            "interval_contains_truth_statistic".to_string(),
            Value::Str("truth_within_closed_interval_all_coordinates".to_string()),
        ),
        (
            "summary_scale".to_string(),
            Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
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
        (
            "interval_bounds".to_string(),
            interval_bounds_value(context.interval, context.rank_support.draws),
        ),
        ("lower".to_string(), slice_to_value(shape, &lowers)),
        ("upper".to_string(), slice_to_value(shape, &uppers)),
        (
            "rank_statistic".to_string(),
            Value::Str("count_posterior_draws_less_than_truth".to_string()),
        ),
        (
            "rank_scope".to_string(),
            Value::Str("per_parameter_coordinate_marginal".to_string()),
        ),
        (
            "tie_statistic".to_string(),
            Value::Str("count_posterior_draws_equal_to_truth".to_string()),
        ),
        (
            "tie_count_bounds".to_string(),
            rank_bounds_value(context.rank_support.draws),
        ),
        (
            "tie_count_bin_order".to_string(),
            rank_bin_order_value(context.rank_support.draws),
        ),
        (
            "tie_count_bin_count".to_string(),
            Value::Int(context.rank_support.bin_count as i64),
        ),
        (
            "interval_contains_truth".to_string(),
            Value::Bool(interval_contains_truth),
        ),
        (
            "interval_contains_truth_by_coordinate".to_string(),
            if shape.is_empty() {
                contains_by_coord[0].clone()
            } else {
                Value::Array(contains_by_coord)
            },
        ),
        ("rhat".to_string(), Value::Float(worst_rhat)),
        ("ess".to_string(), Value::Float(worst_ess)),
        ("name".to_string(), Value::Str(name.to_string())),
    ])
}

fn interval_bounds_value(interval: f64, draw_count: usize) -> Value {
    let lower = (1.0 - interval) / 2.0;
    Value::Object(vec![
        ("interval_probability".to_string(), Value::Float(interval)),
        ("lower_tail_probability".to_string(), Value::Float(lower)),
        ("upper_tail_probability".to_string(), Value::Float(lower)),
        ("lower_quantile".to_string(), Value::Float(lower)),
        ("upper_quantile".to_string(), Value::Float(1.0 - lower)),
        (
            "quantile_index_base".to_string(),
            Value::Str("zero_based_sorted_ascending_posterior_draws".to_string()),
        ),
        (
            "sorted_draw_count".to_string(),
            Value::Int(draw_count as i64),
        ),
        (
            "lower_quantile_index".to_string(),
            quantile_index_value(lower, draw_count),
        ),
        (
            "upper_quantile_index".to_string(),
            quantile_index_value(1.0 - lower, draw_count),
        ),
    ])
}

fn rank_bounds_value(rank_draws: usize) -> Value {
    Value::Object(vec![
        ("min".to_string(), Value::Int(0)),
        ("max".to_string(), Value::Int(rank_draws as i64)),
    ])
}

fn rank_bin_order_value(rank_draws: usize) -> Value {
    Value::Array(
        (0..=rank_draws)
            .map(|rank| Value::Int(rank as i64))
            .collect(),
    )
}

fn treedepth_bin_order_value(bin_count: usize) -> Value {
    Value::Array(
        (0..bin_count)
            .map(|depth| Value::Int(depth as i64))
            .collect(),
    )
}

fn chain_stats(chains: &[(u64, ChainDraws)]) -> Value {
    Value::Array(
        chains
            .iter()
            .map(|(chain_id, chain)| {
                let treedepth_bin_count = chain.treedepth_histogram.len();
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
                                .map(|&count| Value::Int(count as i64))
                                .collect(),
                        ),
                    ),
                    (
                        "treedepth_bin_order".to_string(),
                        treedepth_bin_order_value(treedepth_bin_count),
                    ),
                    (
                        "treedepth_bin_count".to_string(),
                        Value::Int(treedepth_bin_count as i64),
                    ),
                    ("step_size".to_string(), Value::Float(chain.step_size)),
                    ("mean_accept".to_string(), Value::Float(chain.mean_accept)),
                ])
            })
            .collect(),
    )
}

fn chain_order_value(chains: &[(u64, ChainDraws)]) -> Value {
    Value::Array(
        chains
            .iter()
            .map(|(chain_id, _)| Value::Int(*chain_id as i64))
            .collect(),
    )
}

fn replicate_order_value(replicates: usize) -> Value {
    Value::Array(
        (0..replicates)
            .map(|replicate| Value::Int(replicate as i64))
            .collect(),
    )
}

fn sbc_seed_array_value(seed: u64, replicates: usize, seed_offset: u64) -> Result<Value, Error> {
    let mut values = Vec::with_capacity(replicates);
    for replicate in 0..replicates {
        let replicate_offset = (replicate as u64).checked_mul(2).ok_or_else(|| {
            invalid("sbc seed and replicate count must produce reportable derived seeds")
        })?;
        let seed = derived_seed(seed, replicate_offset + seed_offset, "sbc")?;
        values.push(Value::Int(seed as i64));
    }
    Ok(Value::Array(values))
}

struct SamplerSummary {
    chain_count: usize,
    draw_count: usize,
    total_divergences: usize,
    treedepth_histogram: Vec<usize>,
}

impl SamplerSummary {
    fn new(max_treedepth: usize) -> Self {
        Self {
            chain_count: 0,
            draw_count: 0,
            total_divergences: 0,
            treedepth_histogram: vec![0; max_treedepth + 1],
        }
    }

    fn from_chains(
        chains: &[(u64, ChainDraws)],
        max_treedepth: usize,
        context: &str,
    ) -> Result<Self, Error> {
        let mut summary = Self::new(max_treedepth);
        summary.add_chains(chains, context)?;
        Ok(summary)
    }

    fn add_chains(&mut self, chains: &[(u64, ChainDraws)], context: &str) -> Result<(), Error> {
        for (_, chain) in chains {
            self.chain_count = checked_report_count_add(
                self.chain_count,
                1,
                &format!("{context} sampler_summary chain_count"),
            )?;
            self.draw_count = checked_report_count_add(
                self.draw_count,
                chain.draws.len(),
                &format!("{context} sampler_summary draw_count"),
            )?;
            self.total_divergences = checked_report_count_add(
                self.total_divergences,
                chain.divergences,
                &format!("{context} sampler_summary total_divergences"),
            )?;
            for (depth, &count) in chain.treedepth_histogram.iter().enumerate() {
                if depth >= self.treedepth_histogram.len() {
                    return Err(invalid(format!(
                        "{context} sampler_summary treedepth histogram exceeded configured max_treedepth"
                    )));
                }
                self.treedepth_histogram[depth] = checked_report_count_add(
                    self.treedepth_histogram[depth],
                    count,
                    &format!("{context} sampler_summary treedepth_histogram"),
                )?;
            }
        }
        Ok(())
    }

    fn to_value(&self, context: &str) -> Result<Value, Error> {
        Ok(Value::Object(vec![
            (
                "chain_count".to_string(),
                report_count_value(
                    self.chain_count,
                    &format!("{context} sampler_summary chain_count"),
                )?,
            ),
            (
                "draw_count".to_string(),
                report_count_value(
                    self.draw_count,
                    &format!("{context} sampler_summary draw_count"),
                )?,
            ),
            (
                "total_divergences".to_string(),
                report_count_value(
                    self.total_divergences,
                    &format!("{context} sampler_summary total_divergences"),
                )?,
            ),
            (
                "treedepth_histogram".to_string(),
                Value::Array(
                    self.treedepth_histogram
                        .iter()
                        .map(|&count| {
                            report_count_value(
                                count,
                                &format!("{context} sampler_summary treedepth_histogram"),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            ),
            (
                "treedepth_bin_order".to_string(),
                treedepth_bin_order_value(self.treedepth_histogram.len()),
            ),
            (
                "treedepth_bin_count".to_string(),
                report_count_value(
                    self.treedepth_histogram.len(),
                    &format!("{context} sampler_summary treedepth_bin_count"),
                )?,
            ),
        ]))
    }
}

fn checked_report_count_add(left: usize, right: usize, context: &str) -> Result<usize, Error> {
    left.checked_add(right).ok_or_else(|| {
        invalid(format!(
            "{context} must be in 0..=9223372036854775807 because workflow reports counts as JSON integers"
        ))
    })
}

fn report_count_value(count: usize, context: &str) -> Result<Value, Error> {
    if count > i64::MAX as usize {
        Err(invalid(format!(
            "{context} must be in 0..=9223372036854775807 because workflow reports counts as JSON integers"
        )))
    } else {
        Ok(Value::Int(count as i64))
    }
}

fn workflow_phases_value() -> Value {
    Value::Array(
        [
            "parse_json",
            "decode_ir",
            "bind_declared_data",
            "simulate_prior_predictive",
            "bind_declared_and_generated_data",
            "build_posterior_state",
            "evaluate_logp_grad",
            "run_nuts",
            "emit_report",
        ]
        .iter()
        .map(|phase| Value::Str((*phase).to_string()))
        .collect(),
    )
}

fn recover_seed_schedule_value() -> Value {
    Value::Object(vec![
        (
            "prior_seed".to_string(),
            Value::Object(vec![
                ("base_seed".to_string(), Value::Str("seed".to_string())),
                ("offset".to_string(), Value::Int(0)),
            ]),
        ),
        (
            "sample_seed".to_string(),
            Value::Object(vec![
                ("base_seed".to_string(), Value::Str("seed".to_string())),
                ("offset".to_string(), Value::Int(1)),
            ]),
        ),
    ])
}

fn sbc_seed_schedule_value() -> Value {
    Value::Object(vec![
        (
            "prior_seed".to_string(),
            Value::Object(vec![
                ("base_seed".to_string(), Value::Str("seed".to_string())),
                ("replicate_multiplier".to_string(), Value::Int(2)),
                ("offset".to_string(), Value::Int(0)),
            ]),
        ),
        (
            "sample_seed".to_string(),
            Value::Object(vec![
                ("base_seed".to_string(), Value::Str("seed".to_string())),
                ("replicate_multiplier".to_string(), Value::Int(2)),
                ("offset".to_string(), Value::Int(1)),
            ]),
        ),
    ])
}

fn generated_fit(
    meta: ModelMeta,
    declared_data: Vec<(String, DataValue)>,
    sampler: &Settings,
    chains_count: u64,
    prior_seed: u64,
    sample_seed: u64,
    context: &str,
) -> Result<GeneratedFit, Error> {
    let pp_settings = PriorPredictiveSettings { num_draws: 1 };
    let simulated = simulate_prior_predictive(
        meta.clone(),
        declared_data.clone(),
        &pp_settings,
        prior_seed,
    )?;
    let draw = simulated
        .draws
        .first()
        .ok_or_else(|| invalid("recover prior-predictive simulation produced no draw"))?;

    let mut truth: HashMap<String, Tensor> = HashMap::new();
    let mut truth_stochastic_sites: HashMap<String, String> = HashMap::new();
    let mut generated_observed = Vec::new();
    let mut generated_observed_stochastic_sites = Vec::new();
    let mut generated_observed_shapes = Vec::new();
    let mut generated_observed_coordinate_order = Vec::new();
    let mut generated_observed_integer = Vec::new();
    let mut generated_observed_integer_by_coordinate = Vec::new();
    let mut posterior_data = declared_data;
    for (site, (name, tensor)) in simulated.sites.iter().zip(&draw.values) {
        if site.name != *name {
            return Err(invalid(
                "recover prior-predictive site metadata does not match generated values",
            ));
        }
        match site.role {
            PriorPredictiveRole::Parameter => {
                truth.insert(name.clone(), tensor.clone());
                truth_stochastic_sites.insert(name.clone(), site.stochastic_site.clone());
            }
            PriorPredictiveRole::Observed => {
                let integer = value_is_integer(tensor);
                let observed_context = format!("{context} generated observed {name:?}");
                generated_observed.push((
                    name.clone(),
                    generated_observed_value(
                        tensor,
                        &site.integer_by_coordinate,
                        &observed_context,
                    )?,
                ));
                generated_observed_stochastic_sites
                    .push((name.clone(), Value::Str(site.stochastic_site.clone())));
                generated_observed_shapes.push((name.clone(), shape_value(tensor.shape())));
                generated_observed_coordinate_order
                    .push((name.clone(), coordinate_order_value(tensor.shape())));
                generated_observed_integer.push((name.clone(), tensor_integer_value(tensor)));
                generated_observed_integer_by_coordinate.push((
                    name.clone(),
                    integer_flags_value(
                        tensor.shape(),
                        &site.integer_by_coordinate,
                        &observed_context,
                    )?,
                ));
                posterior_data.push((
                    name.clone(),
                    DataValue {
                        shape: tensor.shape().to_vec(),
                        values: tensor.data().to_vec(),
                        integer,
                    },
                ));
            }
        }
    }

    for (name, _) in meta.resolved_free_values() {
        if !truth.contains_key(&name) {
            return Err(invalid(format!(
                "{context} cannot report truth for free value \"{name}\"; the v0 workflow requires a directly simulated stochastic site for every free value"
            )));
        }
    }

    let posterior = Posterior::new(meta, posterior_data)?;
    let mut chains = Vec::with_capacity(chains_count as usize);
    for chain_id in 0..chains_count {
        chains.push((
            chain_id,
            sample(&posterior, sampler, sample_seed, chain_id)?,
        ));
    }

    let packing = posterior.packing();
    let mut constrained_chains: Vec<Vec<Vec<(String, Tensor)>>> = Vec::with_capacity(chains.len());
    for (_, chain) in &chains {
        let mut constrained = Vec::with_capacity(chain.draws.len());
        for q in &chain.draws {
            constrained.push(posterior.constrain(q)?);
        }
        constrained_chains.push(constrained);
    }
    Ok(GeneratedFit {
        prior_seed,
        sample_seed,
        truth,
        truth_stochastic_sites,
        generated_observed,
        generated_observed_stochastic_sites,
        generated_observed_shapes,
        generated_observed_coordinate_order,
        generated_observed_integer,
        generated_observed_integer_by_coordinate,
        chains,
        packing,
        constrained_chains,
    })
}

fn parameter_chain_values(
    constrained_chains: &[Vec<Vec<(String, Tensor)>>],
    param_idx: usize,
) -> Vec<Vec<Vec<f64>>> {
    constrained_chains
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|draw| draw[param_idx].1.data().to_vec())
                .collect()
        })
        .collect()
}

fn rank_and_diagnostics(
    truth: &Tensor,
    chain_values: &[Vec<Vec<f64>>],
) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
    let size = truth.shape().iter().product::<usize>().max(1);
    let mut ranks = Vec::with_capacity(size);
    let mut tie_counts = Vec::with_capacity(size);
    let mut rhats = Vec::with_capacity(size);
    let mut esses = Vec::with_capacity(size);
    for coord in 0..size {
        let truth_value = truth.data()[coord];
        let mut rank = 0i64;
        let mut ties = 0i64;
        for chain in chain_values {
            for draw in chain {
                if draw[coord] < truth_value {
                    rank += 1;
                } else if draw[coord] == truth_value {
                    ties += 1;
                }
            }
        }
        let series: Vec<Vec<f64>> = chain_values
            .iter()
            .map(|chain| chain.iter().map(|draw| draw[coord]).collect())
            .collect();
        ranks.push(rank);
        tie_counts.push(ties);
        rhats.push(diagnostics::split_rhat(&series));
        esses.push(diagnostics::effective_sample_size(&series));
    }
    (ranks, tie_counts, rhats, esses)
}

fn int_coord_value(shape: &[usize], values: &[i64]) -> Value {
    if shape.is_empty() {
        Value::Int(values[0])
    } else {
        Value::Array(values.iter().map(|&value| Value::Int(value)).collect())
    }
}

fn float_coord_value(shape: &[usize], values: &[f64]) -> Value {
    if shape.is_empty() {
        Value::Float(values[0])
    } else {
        Value::Array(values.iter().map(|&value| Value::Float(value)).collect())
    }
}

fn rank_histogram_value(shape: &[usize], histograms: &[Vec<i64>]) -> Value {
    if shape.is_empty() {
        Value::Array(
            histograms[0]
                .iter()
                .map(|&count| Value::Int(count))
                .collect(),
        )
    } else {
        Value::Array(
            histograms
                .iter()
                .map(|histogram| {
                    Value::Array(histogram.iter().map(|&count| Value::Int(count)).collect())
                })
                .collect(),
        )
    }
}

/// Run one v0-provisional recovery scenario and render a JSON report.
pub fn recover_report(
    meta: ModelMeta,
    declared_data: Vec<(String, DataValue)>,
    settings: &RecoverSettings,
    seed: u64,
) -> Result<String, Error> {
    validate_workflow_chains(settings.chains, "recover")?;
    validate_workflow_draws(settings.sampler.num_draws, "recover")?;
    validate_workflow_sampler_counts(&settings.sampler, "recover")?;
    validate_workflow_treedepth(settings.sampler.max_treedepth, "recover")?;
    if !(0.0..1.0).contains(&settings.interval) || settings.interval <= 0.0 {
        return Err(invalid("recover interval must be in (0, 1)"));
    }

    validate_reportable_seed(seed, "recover")?;
    let posterior_draws = recover_posterior_draws(settings.chains, settings.sampler.num_draws)?;
    let rank_bin_count = workflow_rank_bin_count(posterior_draws, "recover")?;
    let rank_support = RankSupport {
        draws: posterior_draws,
        bin_count: rank_bin_count,
    };
    let declared_data_count = declared_data.len();
    let declared_data_order = declared_data_order_value(&declared_data);
    let declared_data_values = declared_data_values_value(&declared_data, "recover")?;
    let declared_data_shapes = declared_data_shapes_value(&declared_data);
    let declared_data_coordinate_order = declared_data_coordinate_order_value(&declared_data);
    let declared_data_integer = declared_data_integer_value(&declared_data);
    let declared_data_integer_by_coordinate =
        declared_data_integer_by_coordinate_value(&declared_data);
    let prior_seed = seed;
    let sample_seed = derived_seed(seed, 1, "recover")?;
    let fit = generated_fit(
        meta,
        declared_data,
        &settings.sampler,
        settings.chains,
        prior_seed,
        sample_seed,
        "recover",
    )?;

    let mut parameter_entries = Vec::new();
    let mut interval_contains_truth_by_parameter = Vec::new();
    for (param_idx, (name, shape)) in fit.packing.iter().enumerate() {
        let Some(truth_value) = fit.truth.get(name) else {
            continue;
        };
        let Some(stochastic_site) = fit.truth_stochastic_sites.get(name) else {
            return Err(invalid(
                "recover prior-predictive truth metadata is missing stochastic_site",
            ));
        };
        let chain_values = parameter_chain_values(&fit.constrained_chains, param_idx);
        let context = RecoverParamContext {
            interval: settings.interval,
            rank_support,
            simulation_index: 0,
            prior_seed,
            sample_seed,
        };
        let summary = summarize_param(
            name,
            stochastic_site,
            shape,
            truth_value,
            &chain_values,
            context,
        );
        let Some(interval_contains_truth) = summary.get("interval_contains_truth") else {
            return Err(invalid(
                "recover parameter summary is missing interval containment fact",
            ));
        };
        interval_contains_truth_by_parameter.push((name.clone(), interval_contains_truth.clone()));
        parameter_entries.push((name.clone(), summary));
    }

    let generated_observed_order = entry_order_value(&fit.generated_observed);
    let generated_observed_count = fit.generated_observed.len();
    let parameter_count = fit.packing.len();
    let report = Value::Object(vec![
        (
            "recover_format".to_string(),
            Value::Str(WORKFLOW_FORMAT.to_string()),
        ),
        (
            "workflow_format".to_string(),
            Value::Str(WORKFLOW_FORMAT.to_string()),
        ),
        (
            "report_kind".to_string(),
            Value::Str("parameter_recovery_facts".to_string()),
        ),
        (
            "report_scope".to_string(),
            Value::Str("single_simulated_dataset".to_string()),
        ),
        ("simulation_count".to_string(), Value::Int(1)),
        (
            "simulation_index_base".to_string(),
            Value::Str(SIMULATION_INDEX_BASE.to_string()),
        ),
        (
            "simulation_order".to_string(),
            Value::Array(vec![Value::Int(0)]),
        ),
        (
            "prior_predictive_draws".to_string(),
            Value::Int(PRIOR_PREDICTIVE_DRAWS),
        ),
        (
            "prior_predictive_draws_artifact_kind".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "prior_predictive_draws_artifact_scope".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        ("workflow_phases".to_string(), workflow_phases_value()),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("prior_seed".to_string(), Value::Int(prior_seed as i64)),
        ("sample_seed".to_string(), Value::Int(sample_seed as i64)),
        ("seed_schedule".to_string(), recover_seed_schedule_value()),
        ("interval".to_string(), Value::Float(settings.interval)),
        (
            "posterior_draws".to_string(),
            Value::Int(posterior_draws as i64),
        ),
        (
            "posterior_draws_artifact_kind".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "posterior_draws_artifact_scope".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "parameter_count".to_string(),
            report_count_value(parameter_count, "recover parameter_count")?,
        ),
        (
            "parameter_report_count".to_string(),
            report_count_value(parameter_entries.len(), "recover parameter_report_count")?,
        ),
        (
            "generated_observed_count".to_string(),
            report_count_value(generated_observed_count, "recover generated_observed_count")?,
        ),
        (
            "generated_observed_artifact_kind".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "generated_observed_artifact_scope".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "generated_observed_draw_index".to_string(),
            Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX),
        ),
        (
            "generated_observed_draw_index_base".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
        ),
        (
            "declared_data_count".to_string(),
            report_count_value(declared_data_count, "recover declared_data_count")?,
        ),
        ("rank_draws".to_string(), Value::Int(posterior_draws as i64)),
        (
            "interval_method".to_string(),
            Value::Str("equal_tailed_linear_quantile".to_string()),
        ),
        (
            "interval_scope".to_string(),
            Value::Str("per_parameter_coordinate_marginal".to_string()),
        ),
        (
            "interval_contains_truth_statistic".to_string(),
            Value::Str("truth_within_closed_interval_all_coordinates".to_string()),
        ),
        (
            "interval_contains_truth_by_parameter".to_string(),
            Value::Object(interval_contains_truth_by_parameter),
        ),
        (
            "rank_statistic".to_string(),
            Value::Str("count_posterior_draws_less_than_truth".to_string()),
        ),
        (
            "rank_scope".to_string(),
            Value::Str("per_parameter_coordinate_marginal".to_string()),
        ),
        (
            "tie_statistic".to_string(),
            Value::Str("count_posterior_draws_equal_to_truth".to_string()),
        ),
        (
            "tie_count_bounds".to_string(),
            rank_bounds_value(posterior_draws),
        ),
        (
            "tie_count_bin_order".to_string(),
            rank_bin_order_value(posterior_draws),
        ),
        (
            "tie_count_bin_count".to_string(),
            Value::Int(rank_bin_count as i64),
        ),
        (
            "parameter_summary_scale".to_string(),
            Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
        ),
        (
            "interval_bounds".to_string(),
            interval_bounds_value(settings.interval, posterior_draws),
        ),
        (
            "rank_bounds".to_string(),
            rank_bounds_value(posterior_draws),
        ),
        (
            "rank_bin_order".to_string(),
            rank_bin_order_value(posterior_draws),
        ),
        (
            "rank_bin_count".to_string(),
            Value::Int(rank_bin_count as i64),
        ),
        (
            "settings".to_string(),
            Value::Object(vec![
                ("chains".to_string(), Value::Int(settings.chains as i64)),
                (
                    "num_warmup".to_string(),
                    Value::Int(settings.sampler.num_warmup as i64),
                ),
                (
                    "num_draws".to_string(),
                    Value::Int(settings.sampler.num_draws as i64),
                ),
                (
                    "max_treedepth".to_string(),
                    Value::Int(settings.sampler.max_treedepth as i64),
                ),
                (
                    "target_accept".to_string(),
                    Value::Float(settings.sampler.target_accept),
                ),
                ("interval".to_string(), Value::Float(settings.interval)),
            ]),
        ),
        (
            "chain_count".to_string(),
            report_count_value(fit.chains.len(), "recover chain_count")?,
        ),
        (
            "generated_observed_order".to_string(),
            generated_observed_order,
        ),
        (
            "generated_observed".to_string(),
            Value::Object(fit.generated_observed),
        ),
        (
            "generated_observed_stochastic_sites".to_string(),
            Value::Object(fit.generated_observed_stochastic_sites),
        ),
        (
            "generated_observed_shapes".to_string(),
            Value::Object(fit.generated_observed_shapes),
        ),
        (
            "generated_observed_coordinate_order".to_string(),
            Value::Object(fit.generated_observed_coordinate_order),
        ),
        (
            "generated_observed_integer".to_string(),
            Value::Object(fit.generated_observed_integer),
        ),
        (
            "generated_observed_integer_by_coordinate".to_string(),
            Value::Object(fit.generated_observed_integer_by_coordinate),
        ),
        ("declared_data_order".to_string(), declared_data_order),
        ("declared_data".to_string(), declared_data_values),
        ("declared_data_shapes".to_string(), declared_data_shapes),
        (
            "declared_data_coordinate_order".to_string(),
            declared_data_coordinate_order,
        ),
        ("declared_data_integer".to_string(), declared_data_integer),
        (
            "declared_data_integer_by_coordinate".to_string(),
            declared_data_integer_by_coordinate,
        ),
        (
            "parameter_order".to_string(),
            parameter_order_value(&fit.packing),
        ),
        (
            "sampler_summary".to_string(),
            SamplerSummary::from_chains(&fit.chains, settings.sampler.max_treedepth, "recover")?
                .to_value("recover")?,
        ),
        ("parameters".to_string(), Value::Object(parameter_entries)),
        ("chain_order".to_string(), chain_order_value(&fit.chains)),
        ("chains".to_string(), chain_stats(&fit.chains)),
    ]);
    json::write(&report)
}

/// Run a v0-provisional simulation-based calibration scenario and render facts.
pub fn sbc_report(
    meta: ModelMeta,
    declared_data: Vec<(String, DataValue)>,
    settings: &SbcSettings,
    seed: u64,
) -> Result<String, Error> {
    validate_sbc_replicates(settings.replicates)?;
    validate_workflow_chains(settings.chains, "sbc")?;
    validate_workflow_draws(settings.sampler.num_draws, "sbc")?;
    validate_workflow_sampler_counts(&settings.sampler, "sbc")?;
    validate_workflow_treedepth(settings.sampler.max_treedepth, "sbc")?;
    validate_reportable_seed(seed, "sbc")?;
    validate_sbc_seed_span(seed, settings.replicates)?;
    let rank_draws = sbc_rank_draws(settings.chains, settings.sampler.num_draws)?;
    let rank_bin_count = workflow_rank_bin_count(rank_draws, "sbc")?;
    let declared_data_count = declared_data.len();
    let declared_data_order = declared_data_order_value(&declared_data);
    let declared_data_values = declared_data_values_value(&declared_data, "sbc")?;
    let declared_data_shapes = declared_data_shapes_value(&declared_data);
    let declared_data_coordinate_order = declared_data_coordinate_order_value(&declared_data);
    let declared_data_integer = declared_data_integer_value(&declared_data);
    let declared_data_integer_by_coordinate =
        declared_data_integer_by_coordinate_value(&declared_data);
    let aggregate_prior_seed_order = sbc_seed_array_value(seed, settings.replicates, 0)?;
    let aggregate_sample_seed_order = sbc_seed_array_value(seed, settings.replicates, 1)?;

    let mut replicate_values = Vec::with_capacity(settings.replicates);
    let mut param_histograms: HashMap<String, (Vec<usize>, Vec<Vec<i64>>)> = HashMap::new();
    let mut param_rank_values: HashMap<String, Vec<Value>> = HashMap::new();
    let mut param_tie_count_values: HashMap<String, Vec<Value>> = HashMap::new();
    let mut param_truth_values: HashMap<String, Vec<Value>> = HashMap::new();
    let mut param_truth_integer_values: HashMap<String, Vec<Value>> = HashMap::new();
    let mut param_truth_draw_index_values: HashMap<String, Vec<Value>> = HashMap::new();
    let mut param_stochastic_sites: HashMap<String, String> = HashMap::new();
    let mut param_order = Vec::new();
    let mut sampler_summary = SamplerSummary::new(settings.sampler.max_treedepth);
    let mut generated_observed_count_per_replicate: Option<usize> = None;
    let mut generated_observed_order_per_replicate: Option<Value> = None;

    for replicate in 0..settings.replicates {
        let replicate_offset = (replicate as u64).checked_mul(2).ok_or_else(|| {
            invalid("sbc seed and replicate count must produce reportable derived seeds")
        })?;
        let prior_seed = derived_seed(seed, replicate_offset, "sbc")?;
        let sample_seed = derived_seed(seed, replicate_offset + 1, "sbc")?;
        let fit = generated_fit(
            meta.clone(),
            declared_data.clone(),
            &settings.sampler,
            settings.chains,
            prior_seed,
            sample_seed,
            "sbc",
        )?;
        let replicate_sampler_summary =
            SamplerSummary::from_chains(&fit.chains, settings.sampler.max_treedepth, "sbc")?;
        sampler_summary.add_chains(&fit.chains, "sbc")?;

        let mut replicate_params = Vec::new();
        for (param_idx, (name, shape)) in fit.packing.iter().enumerate() {
            let Some(truth_value) = fit.truth.get(name) else {
                continue;
            };
            let Some(stochastic_site) = fit.truth_stochastic_sites.get(name) else {
                return Err(invalid(
                    "sbc prior-predictive truth metadata is missing stochastic_site",
                ));
            };
            param_stochastic_sites
                .entry(name.clone())
                .or_insert_with(|| stochastic_site.clone());
            let chain_values = parameter_chain_values(&fit.constrained_chains, param_idx);
            let (ranks, tie_counts, rhats, esses) =
                rank_and_diagnostics(truth_value, &chain_values);
            if !param_histograms.contains_key(name) {
                param_order.push(name.clone());
            }
            let entry = param_histograms.entry(name.clone()).or_insert_with(|| {
                let size = shape.iter().product::<usize>().max(1);
                (shape.clone(), vec![vec![0; rank_draws + 1]; size])
            });
            for (coord, &rank) in ranks.iter().enumerate() {
                entry.1[coord][rank as usize] += 1;
            }
            let rank_value = int_coord_value(shape, &ranks);
            let tie_count_value = int_coord_value(shape, &tie_counts);
            let truth_json = tensor_to_value(truth_value);
            let truth_integer_value = tensor_integer_value(truth_value);
            param_rank_values
                .entry(name.clone())
                .or_default()
                .push(rank_value.clone());
            param_tie_count_values
                .entry(name.clone())
                .or_default()
                .push(tie_count_value.clone());
            param_truth_values
                .entry(name.clone())
                .or_default()
                .push(truth_json.clone());
            param_truth_integer_values
                .entry(name.clone())
                .or_default()
                .push(truth_integer_value.clone());
            param_truth_draw_index_values
                .entry(name.clone())
                .or_default()
                .push(Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX));
            replicate_params.push((
                name.clone(),
                Value::Object(vec![
                    ("shape".to_string(), shape_value(shape)),
                    (
                        "coordinate_order".to_string(),
                        coordinate_order_value(shape),
                    ),
                    (
                        "stochastic_site".to_string(),
                        Value::Str(stochastic_site.clone()),
                    ),
                    ("truth".to_string(), truth_json),
                    ("truth_integer".to_string(), truth_integer_value),
                    (
                        "truth_artifact_kind".to_string(),
                        Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
                    ),
                    (
                        "truth_artifact_scope".to_string(),
                        Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
                    ),
                    (
                        "truth_draw_index".to_string(),
                        Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX),
                    ),
                    (
                        "truth_draw_index_base".to_string(),
                        Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
                    ),
                    ("prior_seed".to_string(), Value::Int(fit.prior_seed as i64)),
                    (
                        "sample_seed".to_string(),
                        Value::Int(fit.sample_seed as i64),
                    ),
                    ("replicate".to_string(), Value::Int(replicate as i64)),
                    (
                        "replicate_index_base".to_string(),
                        Value::Str(REPLICATE_INDEX_BASE.to_string()),
                    ),
                    ("seed_schedule".to_string(), sbc_seed_schedule_value()),
                    ("rank_draws".to_string(), Value::Int(rank_draws as i64)),
                    ("posterior_draws".to_string(), Value::Int(rank_draws as i64)),
                    (
                        "posterior_draws_artifact_kind".to_string(),
                        Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
                    ),
                    (
                        "posterior_draws_artifact_scope".to_string(),
                        Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
                    ),
                    ("rank_bounds".to_string(), rank_bounds_value(rank_draws)),
                    (
                        "rank_bin_order".to_string(),
                        rank_bin_order_value(rank_draws),
                    ),
                    (
                        "rank_bin_count".to_string(),
                        Value::Int(rank_bin_count as i64),
                    ),
                    (
                        "rank_statistic".to_string(),
                        Value::Str("count_posterior_draws_less_than_truth".to_string()),
                    ),
                    (
                        "rank_scope".to_string(),
                        Value::Str("per_parameter_coordinate_marginal".to_string()),
                    ),
                    ("rank".to_string(), rank_value),
                    (
                        "tie_statistic".to_string(),
                        Value::Str("count_posterior_draws_equal_to_truth".to_string()),
                    ),
                    (
                        "tie_count_bounds".to_string(),
                        rank_bounds_value(rank_draws),
                    ),
                    (
                        "tie_count_bin_order".to_string(),
                        rank_bin_order_value(rank_draws),
                    ),
                    (
                        "tie_count_bin_count".to_string(),
                        Value::Int(rank_bin_count as i64),
                    ),
                    (
                        "summary_scale".to_string(),
                        Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
                    ),
                    (
                        "rhat_statistic".to_string(),
                        Value::Str(RHAT_STATISTIC.to_string()),
                    ),
                    (
                        "rhat_scope".to_string(),
                        Value::Str("per_parameter_coordinate_marginal".to_string()),
                    ),
                    (
                        "ess_statistic".to_string(),
                        Value::Str(ESS_STATISTIC.to_string()),
                    ),
                    (
                        "ess_scope".to_string(),
                        Value::Str("per_parameter_coordinate_marginal".to_string()),
                    ),
                    ("tie_count".to_string(), tie_count_value),
                    ("rhat".to_string(), float_coord_value(shape, &rhats)),
                    ("ess".to_string(), float_coord_value(shape, &esses)),
                ]),
            ));
        }
        let generated_observed_order = entry_order_value(&fit.generated_observed);
        let generated_observed_count = fit.generated_observed.len();
        match generated_observed_count_per_replicate {
            Some(expected) if expected != generated_observed_count => {
                return Err(invalid(
                    "sbc generated observed count changed across replicates",
                ));
            }
            None => generated_observed_count_per_replicate = Some(generated_observed_count),
            _ => {}
        }
        match &generated_observed_order_per_replicate {
            Some(expected) if expected != &generated_observed_order => {
                return Err(invalid(
                    "sbc generated observed order changed across replicates",
                ));
            }
            None => {
                generated_observed_order_per_replicate = Some(generated_observed_order.clone());
            }
            _ => {}
        }
        let parameter_count = fit.packing.len();
        replicate_values.push(Value::Object(vec![
            (
                "sbc_format".to_string(),
                Value::Str(WORKFLOW_FORMAT.to_string()),
            ),
            (
                "workflow_format".to_string(),
                Value::Str(WORKFLOW_FORMAT.to_string()),
            ),
            (
                "report_kind".to_string(),
                Value::Str("simulation_based_calibration_replicate_rank_facts".to_string()),
            ),
            (
                "report_scope".to_string(),
                Value::Str("single_simulated_dataset_replicate".to_string()),
            ),
            ("replicate".to_string(), Value::Int(replicate as i64)),
            (
                "replicate_count".to_string(),
                Value::Int(settings.replicates as i64),
            ),
            (
                "replicate_index_base".to_string(),
                Value::Str(REPLICATE_INDEX_BASE.to_string()),
            ),
            (
                "replicate_order".to_string(),
                replicate_order_value(settings.replicates),
            ),
            ("workflow_phases".to_string(), workflow_phases_value()),
            ("prior_seed".to_string(), Value::Int(fit.prior_seed as i64)),
            (
                "sample_seed".to_string(),
                Value::Int(fit.sample_seed as i64),
            ),
            ("seed_schedule".to_string(), sbc_seed_schedule_value()),
            ("rank_draws".to_string(), Value::Int(rank_draws as i64)),
            (
                "prior_predictive_draws".to_string(),
                Value::Int(PRIOR_PREDICTIVE_DRAWS),
            ),
            (
                "prior_predictive_draws_artifact_kind".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
            ),
            (
                "prior_predictive_draws_artifact_scope".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
            ),
            ("posterior_draws".to_string(), Value::Int(rank_draws as i64)),
            (
                "posterior_draws_artifact_kind".to_string(),
                Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
            ),
            (
                "posterior_draws_artifact_scope".to_string(),
                Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
            ),
            ("rank_bounds".to_string(), rank_bounds_value(rank_draws)),
            (
                "rank_bin_order".to_string(),
                rank_bin_order_value(rank_draws),
            ),
            (
                "rank_bin_count".to_string(),
                Value::Int(rank_bin_count as i64),
            ),
            (
                "parameter_summary_scale".to_string(),
                Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
            ),
            (
                "rank_statistic".to_string(),
                Value::Str("count_posterior_draws_less_than_truth".to_string()),
            ),
            (
                "rank_scope".to_string(),
                Value::Str("per_parameter_coordinate_marginal".to_string()),
            ),
            (
                "tie_statistic".to_string(),
                Value::Str("count_posterior_draws_equal_to_truth".to_string()),
            ),
            (
                "tie_count_bounds".to_string(),
                rank_bounds_value(rank_draws),
            ),
            (
                "tie_count_bin_order".to_string(),
                rank_bin_order_value(rank_draws),
            ),
            (
                "tie_count_bin_count".to_string(),
                Value::Int(rank_bin_count as i64),
            ),
            (
                "settings".to_string(),
                Value::Object(vec![
                    ("chains".to_string(), Value::Int(settings.chains as i64)),
                    (
                        "num_warmup".to_string(),
                        Value::Int(settings.sampler.num_warmup as i64),
                    ),
                    (
                        "num_draws".to_string(),
                        Value::Int(settings.sampler.num_draws as i64),
                    ),
                    (
                        "max_treedepth".to_string(),
                        Value::Int(settings.sampler.max_treedepth as i64),
                    ),
                    (
                        "target_accept".to_string(),
                        Value::Float(settings.sampler.target_accept),
                    ),
                ]),
            ),
            (
                "parameter_count".to_string(),
                report_count_value(parameter_count, "sbc replicate parameter_count")?,
            ),
            (
                "parameter_report_count".to_string(),
                report_count_value(
                    replicate_params.len(),
                    "sbc replicate parameter_report_count",
                )?,
            ),
            (
                "chain_count".to_string(),
                report_count_value(fit.chains.len(), "sbc replicate chain_count")?,
            ),
            (
                "declared_data_count".to_string(),
                report_count_value(declared_data_count, "sbc replicate declared_data_count")?,
            ),
            (
                "declared_data_order".to_string(),
                declared_data_order.clone(),
            ),
            (
                "generated_observed_count".to_string(),
                report_count_value(
                    generated_observed_count,
                    "sbc replicate generated_observed_count",
                )?,
            ),
            (
                "generated_observed_artifact_kind".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
            ),
            (
                "generated_observed_artifact_scope".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
            ),
            (
                "generated_observed_draw_index".to_string(),
                Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX),
            ),
            (
                "generated_observed_draw_index_base".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
            ),
            (
                "generated_observed_order".to_string(),
                generated_observed_order,
            ),
            (
                "generated_observed".to_string(),
                Value::Object(fit.generated_observed),
            ),
            (
                "generated_observed_stochastic_sites".to_string(),
                Value::Object(fit.generated_observed_stochastic_sites),
            ),
            (
                "generated_observed_shapes".to_string(),
                Value::Object(fit.generated_observed_shapes),
            ),
            (
                "generated_observed_coordinate_order".to_string(),
                Value::Object(fit.generated_observed_coordinate_order),
            ),
            (
                "generated_observed_integer".to_string(),
                Value::Object(fit.generated_observed_integer),
            ),
            (
                "generated_observed_integer_by_coordinate".to_string(),
                Value::Object(fit.generated_observed_integer_by_coordinate),
            ),
            (
                "parameter_order".to_string(),
                parameter_order_value(&fit.packing),
            ),
            ("parameters".to_string(), Value::Object(replicate_params)),
            (
                "sampler_summary".to_string(),
                replicate_sampler_summary.to_value("sbc")?,
            ),
            ("chain_order".to_string(), chain_order_value(&fit.chains)),
            ("chains".to_string(), chain_stats(&fit.chains)),
        ]));
    }

    let generated_observed_count_per_replicate = generated_observed_count_per_replicate
        .ok_or_else(|| {
            invalid("sbc generated observed metadata is missing; expected at least one replicate")
        })?;
    let generated_observed_order_per_replicate = generated_observed_order_per_replicate
        .ok_or_else(|| {
            invalid("sbc generated observed metadata is missing; expected at least one replicate")
        })?;
    let aggregate_parameter_order = name_order_value(&param_order);
    let mut param_entries = Vec::new();
    for name in param_order {
        let Some((shape, histograms)) = param_histograms.remove(&name) else {
            continue;
        };
        let ranks = param_rank_values.remove(&name).unwrap_or_default();
        let tie_counts = param_tie_count_values.remove(&name).unwrap_or_default();
        let truths = param_truth_values.remove(&name).unwrap_or_default();
        let truth_integer = param_truth_integer_values.remove(&name).unwrap_or_default();
        let truth_draw_index = param_truth_draw_index_values
            .remove(&name)
            .unwrap_or_default();
        let Some(stochastic_site) = param_stochastic_sites.remove(&name) else {
            return Err(invalid(
                "sbc aggregate parameter metadata is missing stochastic_site",
            ));
        };
        param_entries.push((
            name,
            Value::Object(vec![
                ("shape".to_string(), shape_value(&shape)),
                (
                    "coordinate_order".to_string(),
                    coordinate_order_value(&shape),
                ),
                ("stochastic_site".to_string(), Value::Str(stochastic_site)),
                ("rank_draws".to_string(), Value::Int(rank_draws as i64)),
                (
                    "posterior_draws_per_replicate".to_string(),
                    Value::Int(rank_draws as i64),
                ),
                (
                    "posterior_draws_artifact_kind".to_string(),
                    Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
                ),
                (
                    "posterior_draws_artifact_scope".to_string(),
                    Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
                ),
                ("rank_bounds".to_string(), rank_bounds_value(rank_draws)),
                (
                    "rank_bin_order".to_string(),
                    rank_bin_order_value(rank_draws),
                ),
                (
                    "rank_bin_count".to_string(),
                    Value::Int(rank_bin_count as i64),
                ),
                (
                    "rank_statistic".to_string(),
                    Value::Str("count_posterior_draws_less_than_truth".to_string()),
                ),
                (
                    "rank_scope".to_string(),
                    Value::Str("per_parameter_coordinate_marginal".to_string()),
                ),
                (
                    "replicate_count".to_string(),
                    Value::Int(settings.replicates as i64),
                ),
                (
                    "replicate_order".to_string(),
                    replicate_order_value(settings.replicates),
                ),
                (
                    "replicate_index_base".to_string(),
                    Value::Str(REPLICATE_INDEX_BASE.to_string()),
                ),
                ("prior_seed".to_string(), aggregate_prior_seed_order.clone()),
                (
                    "sample_seed".to_string(),
                    aggregate_sample_seed_order.clone(),
                ),
                ("seed_schedule".to_string(), sbc_seed_schedule_value()),
                ("ranks".to_string(), Value::Array(ranks)),
                (
                    "tie_statistic".to_string(),
                    Value::Str("count_posterior_draws_equal_to_truth".to_string()),
                ),
                (
                    "tie_count_bounds".to_string(),
                    rank_bounds_value(rank_draws),
                ),
                (
                    "tie_count_bin_order".to_string(),
                    rank_bin_order_value(rank_draws),
                ),
                (
                    "tie_count_bin_count".to_string(),
                    Value::Int(rank_bin_count as i64),
                ),
                (
                    "rank_histogram_statistic".to_string(),
                    Value::Str("count_simulated_replicates_by_rank".to_string()),
                ),
                (
                    "rank_histogram_scope".to_string(),
                    Value::Str("per_parameter_coordinate_marginal".to_string()),
                ),
                (
                    "summary_scale".to_string(),
                    Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
                ),
                ("tie_counts".to_string(), Value::Array(tie_counts)),
                ("truth".to_string(), Value::Array(truths)),
                ("truth_integer".to_string(), Value::Array(truth_integer)),
                (
                    "truth_artifact_kind".to_string(),
                    Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
                ),
                (
                    "truth_artifact_scope".to_string(),
                    Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
                ),
                (
                    "truth_draw_index".to_string(),
                    Value::Array(truth_draw_index),
                ),
                (
                    "truth_draw_index_base".to_string(),
                    Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
                ),
                (
                    "rank_histogram".to_string(),
                    rank_histogram_value(&shape, &histograms),
                ),
            ]),
        ));
    }
    let parameter_count = param_entries.len();

    let report = Value::Object(vec![
        (
            "sbc_format".to_string(),
            Value::Str(WORKFLOW_FORMAT.to_string()),
        ),
        (
            "workflow_format".to_string(),
            Value::Str(WORKFLOW_FORMAT.to_string()),
        ),
        (
            "report_kind".to_string(),
            Value::Str("simulation_based_calibration_rank_facts".to_string()),
        ),
        (
            "report_scope".to_string(),
            Value::Str("replicated_simulated_datasets".to_string()),
        ),
        (
            "replicate_workflow_phases".to_string(),
            workflow_phases_value(),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        (
            "replicates".to_string(),
            Value::Int(settings.replicates as i64),
        ),
        (
            "replicate_count".to_string(),
            Value::Int(settings.replicates as i64),
        ),
        (
            "replicate_report_count".to_string(),
            report_count_value(replicate_values.len(), "sbc replicate_report_count")?,
        ),
        (
            "replicate_index_base".to_string(),
            Value::Str(REPLICATE_INDEX_BASE.to_string()),
        ),
        (
            "prior_predictive_draws_per_replicate".to_string(),
            Value::Int(PRIOR_PREDICTIVE_DRAWS),
        ),
        (
            "generated_observed_count_per_replicate".to_string(),
            report_count_value(
                generated_observed_count_per_replicate,
                "sbc generated_observed_count_per_replicate",
            )?,
        ),
        (
            "generated_observed_order_per_replicate".to_string(),
            generated_observed_order_per_replicate,
        ),
        (
            "generated_observed_artifact_kind_per_replicate".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "generated_observed_artifact_scope_per_replicate".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "generated_observed_draw_index_per_replicate".to_string(),
            Value::Int(PRIOR_PREDICTIVE_DRAW_INDEX),
        ),
        (
            "generated_observed_draw_index_base_per_replicate".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
        ),
        (
            "prior_predictive_draws_artifact_kind".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "prior_predictive_draws_artifact_scope".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "parameter_count".to_string(),
            report_count_value(parameter_count, "sbc parameter_count")?,
        ),
        (
            "parameter_report_count".to_string(),
            report_count_value(param_entries.len(), "sbc parameter_report_count")?,
        ),
        (
            "declared_data_count".to_string(),
            report_count_value(declared_data_count, "sbc declared_data_count")?,
        ),
        (
            "replicate_order".to_string(),
            replicate_order_value(settings.replicates),
        ),
        (
            "chain_count_per_replicate".to_string(),
            Value::Int(settings.chains as i64),
        ),
        ("rank_draws".to_string(), Value::Int(rank_draws as i64)),
        (
            "posterior_draws_per_replicate".to_string(),
            Value::Int(rank_draws as i64),
        ),
        (
            "posterior_draws_artifact_kind".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_KIND.to_string()),
        ),
        (
            "posterior_draws_artifact_scope".to_string(),
            Value::Str(POSTERIOR_DRAWS_ARTIFACT_SCOPE.to_string()),
        ),
        (
            "rank_statistic".to_string(),
            Value::Str("count_posterior_draws_less_than_truth".to_string()),
        ),
        (
            "rank_scope".to_string(),
            Value::Str("per_parameter_coordinate_marginal".to_string()),
        ),
        (
            "tie_statistic".to_string(),
            Value::Str("count_posterior_draws_equal_to_truth".to_string()),
        ),
        (
            "tie_count_bounds".to_string(),
            rank_bounds_value(rank_draws),
        ),
        (
            "tie_count_bin_order".to_string(),
            rank_bin_order_value(rank_draws),
        ),
        (
            "tie_count_bin_count".to_string(),
            Value::Int(rank_bin_count as i64),
        ),
        (
            "parameter_summary_scale".to_string(),
            Value::Str(PARAMETER_SUMMARY_SCALE.to_string()),
        ),
        ("seed_schedule".to_string(), sbc_seed_schedule_value()),
        ("rank_bounds".to_string(), rank_bounds_value(rank_draws)),
        (
            "rank_bin_order".to_string(),
            rank_bin_order_value(rank_draws),
        ),
        (
            "rank_bin_count".to_string(),
            Value::Int(rank_bin_count as i64),
        ),
        ("declared_data_order".to_string(), declared_data_order),
        ("declared_data".to_string(), declared_data_values),
        ("declared_data_shapes".to_string(), declared_data_shapes),
        (
            "declared_data_coordinate_order".to_string(),
            declared_data_coordinate_order,
        ),
        ("declared_data_integer".to_string(), declared_data_integer),
        (
            "declared_data_integer_by_coordinate".to_string(),
            declared_data_integer_by_coordinate,
        ),
        ("parameter_order".to_string(), aggregate_parameter_order),
        (
            "sampler_summary".to_string(),
            sampler_summary.to_value("sbc")?,
        ),
        (
            "settings".to_string(),
            Value::Object(vec![
                (
                    "replicates".to_string(),
                    Value::Int(settings.replicates as i64),
                ),
                ("chains".to_string(), Value::Int(settings.chains as i64)),
                (
                    "num_warmup".to_string(),
                    Value::Int(settings.sampler.num_warmup as i64),
                ),
                (
                    "num_draws".to_string(),
                    Value::Int(settings.sampler.num_draws as i64),
                ),
                (
                    "max_treedepth".to_string(),
                    Value::Int(settings.sampler.max_treedepth as i64),
                ),
                (
                    "target_accept".to_string(),
                    Value::Float(settings.sampler.target_accept),
                ),
            ]),
        ),
        ("parameters".to_string(), Value::Object(param_entries)),
        (
            "replicate_reports".to_string(),
            Value::Array(replicate_values),
        ),
    ]);
    json::write(&report)
}
