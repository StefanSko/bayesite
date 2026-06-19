//! Prior-predictive simulation over decoded IR.
//!
//! This is a pure runtime layer: callers provide decoded `ModelMeta`, bound
//! data, draw count, and seed; the module returns v0-provisional NDJSON lines.
//! No filesystem, clocks, global entropy, Python, or producer code are used.

use std::collections::HashMap;

use crate::artifact::{
    artifact_identity_entries, coordinate_order_value, entry_order_value, format_marker_field,
    shape_value, POSTERIOR_DRAWS, POSTERIOR_DRAW_INDEX_BASE, POSTERIOR_PREDICTIVE_DRAWS,
    PRIOR_PREDICTIVE_DRAWS, PRIOR_PREDICTIVE_DRAW_INDEX_BASE, V0_PROVISIONAL, WORKFLOW_FORMAT,
};
use crate::error::{Error, ErrorKind};
use crate::ir::{
    BinOpKind, Constraint, DataSchema, Dim, Distribution, Expr, IndexSpec, ModelMeta, Size, UnaryFn,
};
use crate::json::{self, Value};
use crate::model::{DataValue, Posterior};
use crate::rng::Xoshiro256PlusPlus;
use crate::tensor::{gather_map, IndexAtom, Tensor};

#[derive(Debug, Clone)]
pub struct PriorPredictiveSettings {
    pub num_draws: usize,
}

impl Default for PriorPredictiveSettings {
    fn default() -> Self {
        Self { num_draws: 1000 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorPredictiveRole {
    Parameter,
    Observed,
}

impl PriorPredictiveRole {
    pub fn as_str(self) -> &'static str {
        match self {
            PriorPredictiveRole::Parameter => "parameter",
            PriorPredictiveRole::Observed => "observed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PriorPredictiveSite {
    pub name: String,
    pub stochastic_site: String,
    pub role: PriorPredictiveRole,
    pub shape: Vec<usize>,
    pub integer: bool,
    pub integer_by_coordinate: Vec<bool>,
}

#[derive(Debug, Clone)]
pub struct PriorPredictiveDraw {
    pub values: Vec<(String, Tensor)>,
}

#[derive(Debug, Clone)]
pub struct PriorPredictiveRun {
    pub sites: Vec<PriorPredictiveSite>,
    pub draws: Vec<PriorPredictiveDraw>,
}

#[derive(Debug, Clone)]
struct FreeSpec {
    shape: Vec<usize>,
    constraint: Option<Constraint>,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn validate_reportable_seed(seed: u64, context: &str) -> Result<(), Error> {
    if seed <= i64::MAX as u64 {
        Ok(())
    } else {
        Err(invalid(format!(
            "{context} seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers"
        )))
    }
}

fn validate_reportable_draw_count(draws: usize, context: &str) -> Result<(), Error> {
    if draws == 0 {
        Err(invalid(format!(
            "{context} settings.num_draws must be at least 1"
        )))
    } else if draws > i64::MAX as usize {
        Err(invalid(format!(
            "{context} settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers"
        )))
    } else {
        Ok(())
    }
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn mismatch(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataShapeMismatch, message)
}

fn nonfinite(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::NonFiniteDensity, message)
}

fn scalar_int_data(data: &HashMap<String, DataValue>, name: &str) -> Result<i64, Error> {
    let value = data
        .get(name)
        .ok_or_else(|| mismatch(format!("data value \"{name}\" is referenced but not bound")))?;
    if !value.shape.is_empty() {
        return Err(mismatch(format!(
            "data value \"{name}\" must be scalar to be used as a size or dimension"
        )));
    }
    if !value.integer || value.values[0].fract() != 0.0 {
        return Err(mismatch(format!(
            "data value \"{name}\" must be an integer to be used as a size or dimension"
        )));
    }
    Ok(value.values[0] as i64)
}

fn bind_declared_data(
    meta: &ModelMeta,
    data: Vec<(String, DataValue)>,
) -> Result<HashMap<String, DataValue>, Error> {
    let mut data_map = HashMap::new();
    for (name, value) in data {
        if data_map.insert(name.clone(), value).is_some() {
            return Err(mismatch(format!("duplicate data value \"{name}\"")));
        }
    }

    let expected: Vec<&str> = meta.data.iter().map(|(name, _)| name.as_str()).collect();
    let mut missing: Vec<&str> = expected
        .iter()
        .filter(|name| !data_map.contains_key(**name))
        .copied()
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(mismatch(format!(
            "missing model data: {missing:?}; prior-predictive needs every declared data input"
        )));
    }
    let mut extra: Vec<&String> = data_map
        .keys()
        .filter(|name| !expected.contains(&name.as_str()))
        .collect();
    extra.sort_unstable();
    if !extra.is_empty() {
        return Err(mismatch(format!(
            "unexpected prior-predictive data: {extra:?}; pass declared inputs only, not observed values to simulate"
        )));
    }

    for (name, decl) in &meta.data {
        let value = &data_map[name];
        match &decl.schema {
            DataSchema::Rank(rank) => {
                if value.shape.len() as i64 != *rank {
                    return Err(mismatch(format!(
                        "data \"{name}\" must have rank {rank}, got shape {:?}",
                        value.shape
                    )));
                }
            }
            DataSchema::Shape(dims) => {
                if value.shape.len() != dims.len() {
                    return Err(mismatch(format!(
                        "data \"{name}\" must have rank {}, got shape {:?}",
                        dims.len(),
                        value.shape
                    )));
                }
                for (axis, dim) in dims.iter().enumerate() {
                    let expected = match dim {
                        Dim::Fixed(d) => *d,
                        Dim::DataDim(ref_name) => scalar_int_data(&data_map, ref_name)?,
                    };
                    if value.shape[axis] as i64 != expected {
                        return Err(mismatch(format!(
                            "data \"{name}\" axis {axis} must have length {expected}, got {}",
                            value.shape[axis]
                        )));
                    }
                }
            }
        }
    }
    Ok(data_map)
}

fn free_specs(
    meta: &ModelMeta,
    data: &HashMap<String, DataValue>,
) -> Result<HashMap<String, FreeSpec>, Error> {
    let mut specs = HashMap::new();
    for (name, free_value) in meta.resolved_free_values() {
        let shape = match &free_value.size {
            Size::Scalar => vec![],
            Size::Fixed(k) => {
                if *k < 1 {
                    return Err(mismatch(format!(
                        "parameter size for \"{name}\" must be a positive integer, got {k}"
                    )));
                }
                vec![*k as usize]
            }
            Size::Data(ref_name) => {
                let k = scalar_int_data(data, ref_name)?;
                if k < 1 {
                    return Err(mismatch(format!(
                        "data-dependent parameter size \"{ref_name}\" must be a positive integer, got {k}"
                    )));
                }
                vec![k as usize]
            }
        };
        specs.insert(
            name,
            FreeSpec {
                shape,
                constraint: free_value.constraint,
            },
        );
    }
    Ok(specs)
}

struct ForwardEnv<'a> {
    values: HashMap<String, Tensor>,
    data: &'a HashMap<String, DataValue>,
}

impl<'a> ForwardEnv<'a> {
    fn data_tensor(&self, name: &str) -> Result<Tensor, Error> {
        let value = self
            .data
            .get(name)
            .ok_or_else(|| malformed(format!("reference to unknown data value \"{name}\"")))?;
        Ok(Tensor::from_vec(value.shape.clone(), value.values.clone()))
    }

    fn name_tensor(&self, name: &str) -> Result<Tensor, Error> {
        if self.data.contains_key(name) {
            return self.data_tensor(name);
        }
        self.values
            .get(name)
            .cloned()
            .ok_or_else(|| malformed(format!("reference to unknown value \"{name}\"")))
    }

    fn evaluate(&self, expr: &Expr) -> Result<Tensor, Error> {
        match expr {
            Expr::Param(name) | Expr::Data(name) => self.name_tensor(name),
            Expr::Const(v) => Ok(Tensor::scalar(*v)),
            Expr::Bin { op, left, right } => {
                let left = self.evaluate(left)?;
                let right = self.evaluate(right)?;
                left.binary(&right, |a, b| match op {
                    BinOpKind::Add => a + b,
                    BinOpKind::Sub => a - b,
                    BinOpKind::Mul => a * b,
                    BinOpKind::Div => a / b,
                })
            }
            Expr::Unary { function, operand } => {
                let operand = self.evaluate(operand)?;
                Ok(operand.map(|v| match function {
                    UnaryFn::Exp => v.exp(),
                    UnaryFn::Neg => -v,
                    UnaryFn::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                }))
            }
            Expr::Index { base, index } => {
                let base = self.evaluate(base)?;
                let atoms = self.evaluate_index_spec(index)?;
                let map = gather_map(base.shape(), &atoms)?;
                Ok(Tensor::from_vec(
                    map.out_shape,
                    map.map.into_iter().map(|idx| base.data()[idx]).collect(),
                ))
            }
            Expr::VectorScatter {
                length: _,
                observed_idx,
                observed_values,
                missing_idx,
                missing_values,
            } => {
                let obs_pos = self.index_vector(observed_idx)?;
                let mis_pos = self.index_vector(missing_idx)?;
                let obs_values = self.evaluate(observed_values)?;
                let mis_values = self.evaluate(missing_values)?;
                let len = obs_pos.len() + mis_pos.len();
                if obs_values.len() != obs_pos.len() || mis_values.len() != mis_pos.len() {
                    return Err(mismatch(
                        "scatter values must match their index vectors in length",
                    ));
                }
                let mut out = vec![0.0; len];
                for (idx, &value) in obs_pos.iter().zip(obs_values.data()) {
                    out[wrap_scatter_index(*idx, len)?] = value;
                }
                for (idx, &value) in mis_pos.iter().zip(mis_values.data()) {
                    out[wrap_scatter_index(*idx, len)?] = value;
                }
                Ok(Tensor::from_vec(vec![len], out))
            }
        }
    }

    fn index_values(&self, expr: &Expr) -> Result<(Vec<usize>, Vec<i64>), Error> {
        let tensor = self.evaluate(expr)?;
        let mut ints = Vec::with_capacity(tensor.len());
        for &value in tensor.data() {
            if value.fract() != 0.0 {
                return Err(mismatch(format!(
                    "index values must be integers, got {value}"
                )));
            }
            ints.push(value as i64);
        }
        Ok((tensor.shape().to_vec(), ints))
    }

    fn index_vector(&self, expr: &Expr) -> Result<Vec<i64>, Error> {
        let (shape, ints) = self.index_values(expr)?;
        if shape.len() != 1 {
            return Err(mismatch(format!(
                "scatter index vectors must be rank-1, got shape {shape:?}"
            )));
        }
        Ok(ints)
    }

    fn evaluate_index_spec(&self, spec: &IndexSpec) -> Result<Vec<IndexAtom>, Error> {
        match spec {
            IndexSpec::Full => Ok(vec![IndexAtom::Full]),
            IndexSpec::Scalar(expr) => {
                let (shape, ints) = self.index_values(expr)?;
                Ok(vec![if shape.is_empty() {
                    IndexAtom::Scalar(ints[0])
                } else {
                    IndexAtom::Array {
                        shape,
                        values: ints,
                    }
                }])
            }
            IndexSpec::Tuple(items) => {
                let mut atoms = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        IndexSpec::Tuple(_) => {
                            return Err(malformed("nested index tuples are not supported"))
                        }
                        other => atoms.extend(self.evaluate_index_spec(other)?),
                    }
                }
                Ok(atoms)
            }
        }
    }
}

fn wrap_scatter_index(index: i64, len: usize) -> Result<usize, Error> {
    let wrapped = if index < 0 { index + len as i64 } else { index };
    if wrapped < 0 || wrapped >= len as i64 {
        Err(mismatch(format!(
            "scatter index {index} is out of bounds for length {len}"
        )))
    } else {
        Ok(wrapped as usize)
    }
}

fn output_shape(params: &[&Tensor], target: Option<&[usize]>) -> Result<Vec<usize>, Error> {
    if let Some(shape) = target {
        return Ok(shape.to_vec());
    }
    let mut shape = Vec::new();
    for param in params {
        shape = Tensor::broadcast_shapes(&shape, param.shape())?;
    }
    Ok(shape)
}

fn broadcast_param(param: &Tensor, shape: &[usize], name: &str) -> Result<Tensor, Error> {
    param.broadcast_to(shape).map_err(|_| {
        mismatch(format!(
            "cannot broadcast {name} to simulated shape {shape:?}"
        ))
    })
}

fn ensure_finite(value: f64, context: &str) -> Result<f64, Error> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(nonfinite(format!("{context} must be finite, got {value}")))
    }
}

fn sample_gamma(rng: &mut Xoshiro256PlusPlus, shape: f64) -> Result<f64, Error> {
    if !shape.is_finite() || shape <= 0.0 {
        return Err(nonfinite(format!(
            "Gamma shape must be positive and finite, got {shape}"
        )));
    }
    if shape < 1.0 {
        let boosted = sample_gamma(rng, shape + 1.0)?;
        return Ok(boosted * rng.uniform().powf(1.0 / shape));
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.standard_normal();
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v3 = v * v * v;
        let u = rng.uniform();
        if u < 1.0 - 0.0331 * x.powi(4) {
            return Ok(d * v3);
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return Ok(d * v3);
        }
    }
}

fn sample_poisson(rng: &mut Xoshiro256PlusPlus, rate: f64) -> Result<f64, Error> {
    if !rate.is_finite() || rate <= 0.0 {
        return Err(nonfinite(format!(
            "Poisson rate must be positive and finite, got {rate}"
        )));
    }
    if rate > 1000.0 {
        return Err(invalid(
            "prior-predictive Poisson simulation currently supports rates <= 1000; rescale the scenario or use a smaller prior draw"
        ));
    }
    let threshold = (-rate).exp();
    let mut k = 0u64;
    let mut product = 1.0;
    loop {
        k += 1;
        product *= rng.uniform();
        if product <= threshold {
            return Ok((k - 1) as f64);
        }
    }
}

fn sample_binomial(
    rng: &mut Xoshiro256PlusPlus,
    total_count: f64,
    probs: f64,
) -> Result<f64, Error> {
    let total_count = ensure_finite(total_count, "Binomial total_count")?;
    let probs = ensure_finite(probs, "Binomial probs")?;
    if total_count.fract() != 0.0 || total_count < 0.0 {
        return Err(nonfinite(format!(
            "Binomial total_count must be a non-negative integer, got {total_count}"
        )));
    }
    if !(0.0..=1.0).contains(&probs) {
        return Err(nonfinite(format!(
            "Binomial probs must be in [0, 1], got {probs}"
        )));
    }
    if total_count > 1_000_000.0 {
        return Err(invalid(
            "prior-predictive Binomial simulation currently supports total_count <= 1000000",
        ));
    }
    let mut count = 0u64;
    for _ in 0..total_count as u64 {
        if rng.uniform() < probs {
            count += 1;
        }
    }
    Ok(count as f64)
}

fn sample_categorical(rng: &mut Xoshiro256PlusPlus, probs: &[f64]) -> Result<f64, Error> {
    let mut total = 0.0;
    for &prob in probs {
        let prob = ensure_finite(prob, "categorical probability")?;
        if prob < 0.0 {
            return Err(nonfinite(format!(
                "categorical probabilities must be non-negative, got {prob}"
            )));
        }
        total += prob;
    }
    if total <= 0.0 {
        return Err(nonfinite(
            "categorical probabilities must have positive total mass",
        ));
    }
    let mut threshold = rng.uniform() * total;
    for (idx, &prob) in probs.iter().enumerate() {
        threshold -= prob;
        if threshold <= 0.0 {
            return Ok(idx as f64);
        }
    }
    Ok((probs.len() - 1) as f64)
}

fn sample_distribution(
    rng: &mut Xoshiro256PlusPlus,
    env: &ForwardEnv<'_>,
    dist: &Distribution,
    target_shape: Option<&[usize]>,
) -> Result<Tensor, Error> {
    match dist {
        Distribution::Normal { loc, scale } => {
            let loc = env.evaluate(loc)?;
            let scale = env.evaluate(scale)?;
            let shape = output_shape(&[&loc, &scale], target_shape)?;
            let loc = broadcast_param(&loc, &shape, "Normal loc")?;
            let scale = broadcast_param(&scale, &shape, "Normal scale")?;
            let data = loc
                .data()
                .iter()
                .zip(scale.data())
                .map(|(&loc, &scale)| {
                    let loc = ensure_finite(loc, "Normal loc")?;
                    let scale = ensure_finite(scale, "Normal scale")?;
                    if scale <= 0.0 {
                        return Err(nonfinite(format!(
                            "Normal scale must be positive, got {scale}"
                        )));
                    }
                    Ok(loc + scale * rng.standard_normal())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::HalfNormal { scale } => {
            let scale = env.evaluate(scale)?;
            let shape = output_shape(&[&scale], target_shape)?;
            let scale = broadcast_param(&scale, &shape, "HalfNormal scale")?;
            let data = scale
                .data()
                .iter()
                .map(|&scale| {
                    let scale = ensure_finite(scale, "HalfNormal scale")?;
                    if scale <= 0.0 {
                        return Err(nonfinite(format!(
                            "HalfNormal scale must be positive, got {scale}"
                        )));
                    }
                    Ok(scale * rng.standard_normal().abs())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::StudentT { df, loc, scale } => {
            let df = env.evaluate(df)?;
            let loc = env.evaluate(loc)?;
            let scale = env.evaluate(scale)?;
            let shape = output_shape(&[&df, &loc, &scale], target_shape)?;
            let df = broadcast_param(&df, &shape, "StudentT df")?;
            let loc = broadcast_param(&loc, &shape, "StudentT loc")?;
            let scale = broadcast_param(&scale, &shape, "StudentT scale")?;
            let data = df
                .data()
                .iter()
                .zip(loc.data())
                .zip(scale.data())
                .map(|((&df, &loc), &scale)| {
                    let df = ensure_finite(df, "StudentT df")?;
                    let loc = ensure_finite(loc, "StudentT loc")?;
                    let scale = ensure_finite(scale, "StudentT scale")?;
                    if df <= 0.0 {
                        return Err(nonfinite(format!("StudentT df must be positive, got {df}")));
                    }
                    if scale <= 0.0 {
                        return Err(nonfinite(format!(
                            "StudentT scale must be positive, got {scale}"
                        )));
                    }
                    let z = rng.standard_normal();
                    let gamma = sample_gamma(rng, 0.5 * df)?;
                    Ok(loc + scale * z / (gamma / (0.5 * df)).sqrt())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Exponential { rate } => {
            let rate = env.evaluate(rate)?;
            let shape = output_shape(&[&rate], target_shape)?;
            let rate = broadcast_param(&rate, &shape, "Exponential rate")?;
            let data = rate
                .data()
                .iter()
                .map(|&rate| {
                    let rate = ensure_finite(rate, "Exponential rate")?;
                    if rate <= 0.0 {
                        return Err(nonfinite(format!(
                            "Exponential rate must be positive, got {rate}"
                        )));
                    }
                    Ok(-(1.0 - rng.uniform()).ln() / rate)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Uniform { low, high } => {
            let low = env.evaluate(low)?;
            let high = env.evaluate(high)?;
            let shape = output_shape(&[&low, &high], target_shape)?;
            let low = broadcast_param(&low, &shape, "Uniform low")?;
            let high = broadcast_param(&high, &shape, "Uniform high")?;
            let data = low
                .data()
                .iter()
                .zip(high.data())
                .map(|(&low, &high)| {
                    let low = ensure_finite(low, "Uniform low")?;
                    let high = ensure_finite(high, "Uniform high")?;
                    if high <= low {
                        return Err(nonfinite(format!(
                            "Uniform high must be greater than low, got low={low}, high={high}"
                        )));
                    }
                    Ok(low + (high - low) * rng.uniform())
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Beta { alpha, beta } => {
            let alpha = env.evaluate(alpha)?;
            let beta = env.evaluate(beta)?;
            let shape = output_shape(&[&alpha, &beta], target_shape)?;
            let alpha = broadcast_param(&alpha, &shape, "Beta alpha")?;
            let beta = broadcast_param(&beta, &shape, "Beta beta")?;
            let data = alpha
                .data()
                .iter()
                .zip(beta.data())
                .map(|(&alpha, &beta)| {
                    let x = sample_gamma(rng, ensure_finite(alpha, "Beta alpha")?)?;
                    let y = sample_gamma(rng, ensure_finite(beta, "Beta beta")?)?;
                    Ok(x / (x + y))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Bernoulli { probs } => {
            let probs = env.evaluate(probs)?;
            let shape = output_shape(&[&probs], target_shape)?;
            let probs = broadcast_param(&probs, &shape, "Bernoulli probs")?;
            let data = probs
                .data()
                .iter()
                .map(|&p| {
                    let p = ensure_finite(p, "Bernoulli probs")?;
                    if !(0.0..=1.0).contains(&p) {
                        return Err(nonfinite(format!(
                            "Bernoulli probs must be in [0, 1], got {p}"
                        )));
                    }
                    Ok(if rng.uniform() < p { 1.0 } else { 0.0 })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Poisson { rate } => {
            let rate = env.evaluate(rate)?;
            let shape = output_shape(&[&rate], target_shape)?;
            let rate = broadcast_param(&rate, &shape, "Poisson rate")?;
            let data = rate
                .data()
                .iter()
                .map(|&rate| sample_poisson(rng, rate))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::Binomial { total_count, probs } => {
            let total_count = env.evaluate(total_count)?;
            let probs = env.evaluate(probs)?;
            let shape = output_shape(&[&total_count, &probs], target_shape)?;
            let total_count = broadcast_param(&total_count, &shape, "Binomial total_count")?;
            let probs = broadcast_param(&probs, &shape, "Binomial probs")?;
            let data = total_count
                .data()
                .iter()
                .zip(probs.data())
                .map(|(&total_count, &probs)| sample_binomial(rng, total_count, probs))
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => {
            let total_count = env.evaluate(total_count)?;
            let alpha = env.evaluate(alpha)?;
            let beta = env.evaluate(beta)?;
            let shape = output_shape(&[&total_count, &alpha, &beta], target_shape)?;
            let total_count = broadcast_param(&total_count, &shape, "BetaBinomial total_count")?;
            let alpha = broadcast_param(&alpha, &shape, "BetaBinomial alpha")?;
            let beta = broadcast_param(&beta, &shape, "BetaBinomial beta")?;
            let data = total_count
                .data()
                .iter()
                .zip(alpha.data())
                .zip(beta.data())
                .map(|((&total_count, &alpha), &beta)| {
                    let x = sample_gamma(rng, ensure_finite(alpha, "BetaBinomial alpha")?)?;
                    let y = sample_gamma(rng, ensure_finite(beta, "BetaBinomial beta")?)?;
                    sample_binomial(rng, total_count, x / (x + y))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::NegativeBinomial {
            mean,
            overdispersion,
        } => {
            let mean = env.evaluate(mean)?;
            let overdispersion = env.evaluate(overdispersion)?;
            let shape = output_shape(&[&mean, &overdispersion], target_shape)?;
            let mean = broadcast_param(&mean, &shape, "NegativeBinomial mean")?;
            let overdispersion =
                broadcast_param(&overdispersion, &shape, "NegativeBinomial overdispersion")?;
            let data = mean
                .data()
                .iter()
                .zip(overdispersion.data())
                .map(|(&mean, &overdispersion)| {
                    let mean = ensure_finite(mean, "NegativeBinomial mean")?;
                    let overdispersion =
                        ensure_finite(overdispersion, "NegativeBinomial overdispersion")?;
                    if mean <= 0.0 {
                        return Err(nonfinite(format!(
                            "NegativeBinomial mean must be positive, got {mean}"
                        )));
                    }
                    if overdispersion <= 0.0 {
                        return Err(nonfinite(format!(
                            "NegativeBinomial overdispersion must be positive, got {overdispersion}"
                        )));
                    }
                    let rate = sample_gamma(rng, overdispersion)? * mean / overdispersion;
                    sample_poisson(rng, rate)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::MultivariateNormal { mean, scale_tril } => {
            let mean = env.evaluate(mean)?;
            let scale_tril = env.evaluate(scale_tril)?;
            let tril_shape = scale_tril.shape().to_vec();
            if tril_shape.len() != 2 || tril_shape[0] != tril_shape[1] {
                return Err(mismatch(format!(
                    "MultivariateNormal scale_tril must be a square rank-2 matrix; got shape {tril_shape:?}"
                )));
            }
            let event_size = tril_shape[0];
            let shape = match target_shape {
                Some(shape) => shape.to_vec(),
                None if mean.shape().is_empty() && event_size == 1 => vec![1],
                None => mean.shape().to_vec(),
            };
            let mean = mean.broadcast_to(&shape)?;
            if shape.len() != 1 || shape[0] != event_size {
                return Err(mismatch(format!(
                    "MultivariateNormal simulation needs event shape [{event_size}], got {shape:?}"
                )));
            }
            let mut z = Vec::with_capacity(event_size);
            for _ in 0..event_size {
                z.push(rng.standard_normal());
            }
            let mut data = Vec::with_capacity(event_size);
            for row in 0..event_size {
                let mut value = mean.data()[row];
                for (col, &z_col) in z.iter().enumerate().take(row + 1) {
                    value += scale_tril.data()[row * event_size + col] * z_col;
                }
                data.push(value);
            }
            Ok(Tensor::from_vec(shape, data))
        }
        Distribution::OrderedLogistic { eta, cutpoints } => {
            let eta = env.evaluate(eta)?;
            let cutpoints = env.evaluate(cutpoints)?;
            if cutpoints.rank() != 1 {
                return Err(mismatch("OrderedLogistic cutpoints must be a vector"));
            }
            if cutpoints.is_empty() {
                return Err(mismatch("OrderedLogistic requires at least one cutpoint"));
            }
            if !cutpoints.data().windows(2).all(|pair| pair[1] > pair[0]) {
                return Err(nonfinite(
                    "OrderedLogistic cutpoints must be strictly increasing",
                ));
            }
            let shape = output_shape(&[&eta], target_shape)?;
            let eta = broadcast_param(&eta, &shape, "OrderedLogistic eta")?;
            let cutpoints = cutpoints.data().to_vec();
            let mut data = Vec::with_capacity(eta.len());
            for &eta in eta.data() {
                let eta = ensure_finite(eta, "OrderedLogistic eta")?;
                let mut probs = Vec::with_capacity(cutpoints.len() + 1);
                let mut previous = 0.0;
                for &cutpoint in &cutpoints {
                    let cumulative = 1.0 / (1.0 + (-(cutpoint - eta)).exp());
                    probs.push(cumulative - previous);
                    previous = cumulative;
                }
                probs.push(1.0 - previous);
                data.push(sample_categorical(rng, &probs)?);
            }
            Ok(Tensor::from_vec(shape, data))
        }
    }
}

fn satisfies_constraint(value: &Tensor, constraint: &Option<Constraint>) -> bool {
    match constraint {
        None => true,
        Some(Constraint::Positive) => value.data().iter().all(|&v| v > 0.0),
        Some(Constraint::UnitInterval) => value.data().iter().all(|&v| v > 0.0 && v < 1.0),
        Some(Constraint::Interval { lower, upper }) => {
            value.data().iter().all(|&v| v > *lower && v < *upper)
        }
        Some(Constraint::Ordered) => {
            value.rank() == 1 && value.data().windows(2).all(|pair| pair[1] > pair[0])
        }
    }
}

fn sample_constrained_distribution(
    rng: &mut Xoshiro256PlusPlus,
    env: &ForwardEnv<'_>,
    dist: &Distribution,
    target_shape: Option<&[usize]>,
    constraint: &Option<Constraint>,
    name: &str,
) -> Result<Tensor, Error> {
    for _ in 0..10_000 {
        let value = sample_distribution(rng, env, dist, target_shape)?;
        if satisfies_constraint(&value, constraint) {
            return Ok(value);
        }
    }
    Err(invalid(format!(
        "could not draw prior-predictive value for constrained site \"{name}\" after 10000 tries; check the prior mass inside the constraint"
    )))
}

fn scalar_to_value(value: f64, integer: bool, context: &str) -> Result<Value, Error> {
    if integer {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(nonfinite(format!(
                "{context} expected an integer finite generated value, got {value}"
            )));
        }
        if value < i64::MIN as f64 || value > i64::MAX as f64 {
            return Err(invalid(format!(
                "{context} integer generated value must fit JSON integer range, got {value}"
            )));
        }
        Ok(Value::Int(value as i64))
    } else {
        Ok(Value::Float(value))
    }
}

fn tensor_to_value(tensor: &Tensor, integer_flags: &[bool], context: &str) -> Result<Value, Error> {
    if integer_flags.len() != tensor.data().len() {
        return Err(invalid(format!(
            "{context} integer metadata length must match generated value length"
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

fn value_is_integer(tensor: &Tensor) -> bool {
    tensor.data().iter().all(|&value| value.fract() == 0.0)
}

fn integer_flags(tensor: &Tensor) -> Vec<bool> {
    tensor
        .data()
        .iter()
        .map(|&value| value.fract() == 0.0)
        .collect()
}

fn integer_flags_to_value(shape: &[usize], flags: &[bool]) -> Value {
    if shape.is_empty() {
        Value::Bool(flags.first().copied().unwrap_or(false))
    } else {
        Value::Array(flags.iter().copied().map(Value::Bool).collect())
    }
}

fn site_order_to_value(sites: &[PriorPredictiveSite]) -> Value {
    Value::Array(
        sites
            .iter()
            .map(|site| Value::Str(site.name.clone()))
            .collect(),
    )
}

fn data_value_to_value(value: &DataValue, context: &str) -> Result<Value, Error> {
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

fn data_values_to_value(data: &[(String, DataValue)]) -> Result<Value, Error> {
    Ok(Value::Object(
        data.iter()
            .map(|(name, value)| {
                Ok((
                    name.clone(),
                    data_value_to_value(value, "prior-predictive artifact declared data")?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?,
    ))
}

fn data_shapes_to_value(data: &[(String, DataValue)]) -> Value {
    Value::Object(
        data.iter()
            .map(|(name, value)| (name.clone(), shape_value(&value.shape)))
            .collect(),
    )
}

fn data_coordinate_order_to_value(data: &[(String, DataValue)]) -> Value {
    Value::Object(
        data.iter()
            .map(|(name, value)| (name.clone(), coordinate_order_value(&value.shape)))
            .collect(),
    )
}

fn data_integer_to_value(data: &[(String, DataValue)]) -> Value {
    Value::Object(
        data.iter()
            .map(|(name, value)| (name.clone(), Value::Bool(value.integer)))
            .collect(),
    )
}

fn data_integer_by_coordinate_value(value: &DataValue) -> Value {
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

fn data_integer_by_coordinate_to_value(data: &[(String, DataValue)]) -> Value {
    Value::Object(
        data.iter()
            .map(|(name, value)| (name.clone(), data_integer_by_coordinate_value(value)))
            .collect(),
    )
}

fn workflow_phases_value() -> Value {
    Value::Array(
        [
            "parse_json",
            "decode_ir",
            "bind_declared_data",
            "simulate_prior_predictive",
            "emit_artifact",
        ]
        .iter()
        .map(|phase| Value::Str((*phase).to_string()))
        .collect(),
    )
}

fn settings_value(settings: &PriorPredictiveSettings) -> Value {
    Value::Object(vec![(
        "num_draws".to_string(),
        Value::Int(settings.num_draws as i64),
    )])
}

fn prior_predictive_artifact_fields() -> Vec<(String, Value)> {
    let mut entries = vec![format_marker_field("prior_predictive_format")];
    entries.extend(artifact_identity_entries(PRIOR_PREDICTIVE_DRAWS));
    entries
}

fn header_value(
    sites: &[PriorPredictiveSite],
    settings: &PriorPredictiveSettings,
    seed: u64,
    declared_data: &[(String, DataValue)],
) -> Result<Value, Error> {
    let mut entries = prior_predictive_artifact_fields();
    entries.extend([
        ("workflow_phases".to_string(), workflow_phases_value()),
        ("draws".to_string(), Value::Int(settings.num_draws as i64)),
        (
            "draw_count".to_string(),
            Value::Int(settings.num_draws as i64),
        ),
        (
            "draw_index_base".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("settings".to_string(), settings_value(settings)),
        ("site_count".to_string(), Value::Int(sites.len() as i64)),
        ("site_order".to_string(), site_order_to_value(sites)),
        (
            "declared_data_count".to_string(),
            Value::Int(declared_data.len() as i64),
        ),
        (
            "declared_data_order".to_string(),
            entry_order_value(declared_data),
        ),
        (
            "declared_data".to_string(),
            data_values_to_value(declared_data)?,
        ),
        (
            "declared_data_shapes".to_string(),
            data_shapes_to_value(declared_data),
        ),
        (
            "declared_data_coordinate_order".to_string(),
            data_coordinate_order_to_value(declared_data),
        ),
        (
            "declared_data_integer".to_string(),
            data_integer_to_value(declared_data),
        ),
        (
            "declared_data_integer_by_coordinate".to_string(),
            data_integer_by_coordinate_to_value(declared_data),
        ),
        (
            "sites".to_string(),
            Value::Array(
                sites
                    .iter()
                    .map(|site| {
                        Value::Object(vec![
                            ("name".to_string(), Value::Str(site.name.clone())),
                            (
                                "stochastic_site".to_string(),
                                Value::Str(site.stochastic_site.clone()),
                            ),
                            (
                                "role".to_string(),
                                Value::Str(site.role.as_str().to_string()),
                            ),
                            ("shape".to_string(), shape_value(&site.shape)),
                            ("integer".to_string(), Value::Bool(site.integer)),
                            (
                                "integer_by_coordinate".to_string(),
                                integer_flags_to_value(&site.shape, &site.integer_by_coordinate),
                            ),
                            (
                                "coordinate_order".to_string(),
                                coordinate_order_value(&site.shape),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    Ok(Value::Object(entries))
}

/// Simulate a complete prior-predictive run from decoded IR and declared data.
pub fn simulate_prior_predictive(
    meta: ModelMeta,
    data: Vec<(String, DataValue)>,
    settings: &PriorPredictiveSettings,
    seed: u64,
) -> Result<PriorPredictiveRun, Error> {
    if settings.num_draws == 0 {
        return Err(invalid(
            "prior-predictive settings.num_draws must be at least 1",
        ));
    }
    let data = bind_declared_data(&meta, data)?;
    let free_specs = free_specs(&meta, &data)?;
    let sites = meta.resolved_stochastic_sites();
    let mut rng = Xoshiro256PlusPlus::for_chain(seed, 0);
    let mut draws = Vec::with_capacity(settings.num_draws);
    let mut site_specs: Option<Vec<PriorPredictiveSite>> = None;

    for _ in 0..settings.num_draws {
        let mut env = ForwardEnv {
            values: HashMap::new(),
            data: &data,
        };
        let mut values = Vec::with_capacity(sites.len());
        let mut current_sites = Vec::with_capacity(sites.len());
        for site in &sites {
            match &site.value {
                Expr::Param(name) => {
                    let free = free_specs.get(name).ok_or_else(|| {
                        malformed(format!(
                            "stochastic site \"{}\" targets unknown free value \"{name}\"",
                            site.name
                        ))
                    })?;
                    let value = sample_constrained_distribution(
                        &mut rng,
                        &env,
                        &site.distribution,
                        Some(&free.shape),
                        &free.constraint,
                        name,
                    )?;
                    env.values.insert(name.clone(), value.clone());
                    current_sites.push(PriorPredictiveSite {
                        name: name.clone(),
                        stochastic_site: site.name.clone(),
                        role: PriorPredictiveRole::Parameter,
                        shape: value.shape().to_vec(),
                        integer: value_is_integer(&value),
                        integer_by_coordinate: integer_flags(&value),
                    });
                    values.push((name.clone(), value));
                }
                Expr::Data(name) => {
                    if data.contains_key(name) {
                        return Err(mismatch(format!(
                            "prior-predictive site \"{}\" writes observed value \"{name}\", but data already binds it; remove observed values from the data document",
                            site.name
                        )));
                    }
                    let value =
                        sample_distribution(&mut rng, &env, &site.distribution, None)?;
                    env.values.insert(name.clone(), value.clone());
                    current_sites.push(PriorPredictiveSite {
                        name: name.clone(),
                        stochastic_site: site.name.clone(),
                        role: PriorPredictiveRole::Observed,
                        shape: value.shape().to_vec(),
                        integer: value_is_integer(&value),
                        integer_by_coordinate: integer_flags(&value),
                    });
                    values.push((name.clone(), value));
                }
                _ => {
                    return Err(invalid(format!(
                        "prior-predictive site \"{}\" has a non-assignable value expression; only ParamRef and DataRef sites are supported in v0-provisional output",
                        site.name
                    )))
                }
            }
        }
        match &site_specs {
            None => site_specs = Some(current_sites),
            Some(expected) if expected.len() == current_sites.len() => {
                for (expected, got) in expected.iter().zip(&current_sites) {
                    if expected.name != got.name
                        || expected.stochastic_site != got.stochastic_site
                        || expected.role != got.role
                        || expected.shape != got.shape
                        || expected.integer != got.integer
                        || expected.integer_by_coordinate != got.integer_by_coordinate
                    {
                        return Err(mismatch(
                            "prior-predictive site metadata changed across draws; dynamic-shape streams are not supported",
                        ));
                    }
                }
            }
            Some(_) => {
                return Err(mismatch(
                    "prior-predictive site count changed across draws; dynamic stochastic structure is not supported",
                ))
            }
        }
        draws.push(PriorPredictiveDraw { values });
    }

    let sites = site_specs.unwrap_or_default();
    Ok(PriorPredictiveRun { sites, draws })
}

fn collect_index_spec_data_refs(index: &IndexSpec, refs: &mut Vec<String>) {
    match index {
        IndexSpec::Scalar(expr) => collect_expr_data_refs(expr, refs),
        IndexSpec::Full => {}
        IndexSpec::Tuple(items) => {
            for item in items {
                collect_index_spec_data_refs(item, refs);
            }
        }
    }
}

fn collect_expr_data_refs(expr: &Expr, refs: &mut Vec<String>) {
    match expr {
        Expr::Data(name) => refs.push(name.clone()),
        Expr::Param(_) | Expr::Const(_) => {}
        Expr::Bin { left, right, .. } => {
            collect_expr_data_refs(left, refs);
            collect_expr_data_refs(right, refs);
        }
        Expr::Unary { operand, .. } => collect_expr_data_refs(operand, refs),
        Expr::Index { base, index } => {
            collect_expr_data_refs(base, refs);
            collect_index_spec_data_refs(index, refs);
        }
        Expr::VectorScatter {
            length,
            observed_idx,
            observed_values,
            missing_idx,
            missing_values,
        } => {
            collect_expr_data_refs(length, refs);
            collect_expr_data_refs(observed_idx, refs);
            collect_expr_data_refs(observed_values, refs);
            collect_expr_data_refs(missing_idx, refs);
            collect_expr_data_refs(missing_values, refs);
        }
    }
}

fn validate_fixed_truth(
    free_specs: &HashMap<String, FreeSpec>,
    truth: Vec<(String, DataValue)>,
    context: &str,
) -> Result<HashMap<String, Tensor>, Error> {
    let mut truth_map = HashMap::new();
    for (name, value) in truth {
        if truth_map
            .insert(name.clone(), (value.shape, value.values))
            .is_some()
        {
            return Err(mismatch(format!(
                "{context} truth has duplicate free value \"{name}\""
            )));
        }
    }
    let mut missing = free_specs
        .keys()
        .filter(|name| !truth_map.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    missing.sort();
    if missing.len() == 1 {
        return Err(mismatch(format!(
            "{context} truth is missing free value \"{}\"",
            missing[0]
        )));
    }
    if !missing.is_empty() {
        return Err(mismatch(format!(
            "{context} truth is missing free values {missing:?}"
        )));
    }
    let mut unknown = truth_map
        .keys()
        .filter(|name| !free_specs.contains_key(*name))
        .cloned()
        .collect::<Vec<_>>();
    unknown.sort();
    if unknown.len() == 1 {
        return Err(mismatch(format!(
            "{context} truth has unknown free value \"{}\"",
            unknown[0]
        )));
    }
    if !unknown.is_empty() {
        return Err(mismatch(format!(
            "{context} truth has unknown free values {unknown:?}"
        )));
    }

    let mut tensors = HashMap::new();
    for (name, spec) in free_specs {
        let (shape, values) = truth_map.remove(name).expect("truth was validated");
        if shape != spec.shape {
            return Err(mismatch(format!(
                "{context} truth for free value \"{name}\" has shape {shape:?}, expected {:?}",
                spec.shape
            )));
        }
        let tensor = Tensor::from_vec(shape, values);
        if !satisfies_constraint(&tensor, &spec.constraint) {
            return Err(mismatch(format!(
                "{context} truth for free value \"{name}\" violates constraint {:?}",
                spec.constraint
            )));
        }
        tensors.insert(name.clone(), tensor);
    }
    Ok(tensors)
}

/// Simulate observed data from a decoded model with user-supplied constrained
/// free-value truth. The returned value is a normal data document payload:
/// declared inputs first, then generated observed DataRef sites in stochastic
/// site order. It carries no simulation marker so `sample` remains provenance
/// agnostic.
pub fn simulate_data_from_truth(
    meta: ModelMeta,
    declared_data: Vec<(String, DataValue)>,
    truth: Vec<(String, DataValue)>,
    seed: u64,
) -> Result<Vec<(String, DataValue)>, Error> {
    validate_reportable_seed(seed, "simulate")?;
    let output_declared_data = declared_data.clone();
    let data = bind_declared_data(&meta, declared_data)?;
    let free_specs = free_specs(&meta, &data)?;
    let truth_values = validate_fixed_truth(&free_specs, truth, "simulate")?;
    let mut env = ForwardEnv {
        values: truth_values,
        data: &data,
    };
    let mut rng = Xoshiro256PlusPlus::for_chain(seed, 0);
    let mut output = output_declared_data;
    let mut generated_names = Vec::<String>::new();
    for site in meta.resolved_stochastic_sites() {
        match &site.value {
            Expr::Param(_) => {}
            Expr::Data(name) => {
                if data.contains_key(name) {
                    return Err(mismatch(format!(
                        "simulate stochastic site \"{}\" writes data value \"{name}\", but it is already bound as declared data",
                        site.name
                    )));
                }
                if generated_names.iter().any(|existing| existing == name) {
                    return Err(mismatch(format!(
                        "simulate stochastic site \"{}\" writes duplicate generated data value \"{name}\"",
                        site.name
                    )));
                }
                let value = sample_distribution(&mut rng, &env, &site.distribution, None)?;
                let integer = distribution_has_integer_support(&site.distribution);
                env.values.insert(name.clone(), value.clone());
                generated_names.push(name.clone());
                output.push((
                    name.clone(),
                    DataValue {
                        shape: value.shape().to_vec(),
                        values: value.data().to_vec(),
                        integer,
                    },
                ));
            }
            other => {
                let mut refs = Vec::new();
                collect_expr_data_refs(other, &mut refs);
                refs.sort();
                refs.dedup();
                let unbound = refs
                    .into_iter()
                    .filter(|name| !data.contains_key(name) && !env.values.contains_key(name))
                    .collect::<Vec<_>>();
                if !unbound.is_empty() {
                    return Err(invalid(format!(
                        "simulate stochastic site \"{}\" has a non-assignable observed value expression referencing ungenerated data {unbound:?}; only direct DataRef observed sites are supported in v0-provisional simulation",
                        site.name
                    )));
                }
            }
        }
    }
    Ok(output)
}

/// Render a complete prior-predictive run as v0-provisional NDJSON lines.
pub fn prior_predictive_ndjson_lines(
    meta: ModelMeta,
    data: Vec<(String, DataValue)>,
    settings: &PriorPredictiveSettings,
    seed: u64,
) -> Result<Vec<String>, Error> {
    validate_reportable_seed(seed, "prior-predictive artifact")?;
    validate_reportable_draw_count(settings.num_draws, "prior-predictive artifact")?;
    let declared_data = data.clone();
    let run = simulate_prior_predictive(meta, data, settings, seed)?;
    let mut lines = Vec::with_capacity(settings.num_draws + 2);
    lines.push(json::write(&header_value(
        &run.sites,
        settings,
        seed,
        &declared_data,
    )?)?);
    for (draw_id, draw) in run.draws.iter().enumerate() {
        let values = Value::Object(
            run.sites
                .iter()
                .zip(&draw.values)
                .map(|(site, (name, tensor))| {
                    if site.name != *name {
                        return Err(invalid(
                            "prior-predictive site metadata does not match generated values",
                        ));
                    }
                    Ok((
                        name.clone(),
                        tensor_to_value(
                            tensor,
                            &site.integer_by_coordinate,
                            "prior-predictive artifact",
                        )?,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?,
        );
        let mut line_entries = prior_predictive_artifact_fields();
        line_entries.extend([
            ("draw_index".to_string(), Value::Int(draw_id as i64)),
            (
                "draw_index_base".to_string(),
                Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
            ),
            ("seed".to_string(), Value::Int(seed as i64)),
            ("draw".to_string(), Value::Int(draw_id as i64)),
            (
                "draw_count".to_string(),
                Value::Int(settings.num_draws as i64),
            ),
            (
                "declared_data_count".to_string(),
                Value::Int(declared_data.len() as i64),
            ),
            (
                "declared_data_order".to_string(),
                entry_order_value(&declared_data),
            ),
            ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
            ("site_order".to_string(), site_order_to_value(&run.sites)),
            ("values".to_string(), values),
        ]);
        let line = Value::Object(line_entries);
        lines.push(json::write(&line)?);
    }
    let mut trailer_entries = prior_predictive_artifact_fields();
    trailer_entries.extend([
        ("workflow_phases".to_string(), workflow_phases_value()),
        ("draws".to_string(), Value::Int(settings.num_draws as i64)),
        (
            "draw_count".to_string(),
            Value::Int(settings.num_draws as i64),
        ),
        (
            "draw_index_base".to_string(),
            Value::Str(PRIOR_PREDICTIVE_DRAW_INDEX_BASE.to_string()),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("settings".to_string(), settings_value(settings)),
        ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
        ("site_order".to_string(), site_order_to_value(&run.sites)),
        (
            "declared_data_count".to_string(),
            Value::Int(declared_data.len() as i64),
        ),
        (
            "declared_data_order".to_string(),
            entry_order_value(&declared_data),
        ),
        ("sites".to_string(), Value::Int(run.sites.len() as i64)),
    ]);
    lines.push(json::write(&Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(trailer_entries),
    )]))?);
    Ok(lines)
}

#[derive(Debug, Clone)]
struct FitParamSpec {
    name: String,
    shape: Vec<usize>,
    size: usize,
}

#[derive(Debug, Clone)]
struct FitDraw {
    draw_index: usize,
    chain: i64,
    draw: i64,
    values: Vec<Tensor>,
}

#[derive(Debug, Clone)]
struct FitDrawStream {
    source_seed: i64,
    params: Vec<FitParamSpec>,
    draws: Vec<FitDraw>,
}

#[derive(Debug, Clone)]
struct FitSourceDraw {
    draw_index: usize,
    chain: i64,
    draw: i64,
}

#[derive(Debug, Clone)]
pub struct PosteriorPredictiveRun {
    pub sites: Vec<PriorPredictiveSite>,
    pub draws: Vec<PriorPredictiveDraw>,
    source_seed: i64,
    source_draws: Vec<FitSourceDraw>,
}

fn malformed_fit(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn parse_fit_shape(value: &Value, name: &str) -> Result<Vec<usize>, Error> {
    let dims = value
        .as_array()
        .ok_or_else(|| malformed_fit(format!("fit parameter {name} shape must be an array")))?;
    dims.iter()
        .map(|dim| {
            let dim = dim.as_i64().ok_or_else(|| {
                malformed_fit(format!(
                    "fit parameter {name} shape entries must be integers"
                ))
            })?;
            if dim < 0 {
                return Err(malformed_fit(format!(
                    "fit parameter {name} shape entries must be non-negative"
                )));
            }
            Ok(dim as usize)
        })
        .collect()
}

fn fit_shape_size(shape: &[usize], name: &str) -> Result<usize, Error> {
    let mut size = 1usize;
    for dim in shape {
        size = size.checked_mul(*dim).ok_or_else(|| {
            malformed_fit(format!(
                "fit parameter {name} shape is too large for this build"
            ))
        })?;
    }
    Ok(size.max(1))
}

fn parse_fit_params(header: &Value) -> Result<Vec<FitParamSpec>, Error> {
    if header.get("draws_format").and_then(Value::as_str) != Some(V0_PROVISIONAL) {
        return Err(malformed_fit(
            "fit header needs draws_format \"v0-provisional\"; rerun `bayesite sample`",
        ));
    }
    if header.get("artifact_kind").and_then(Value::as_str) != Some(POSTERIOR_DRAWS.kind) {
        return Err(malformed_fit(
            "fit header artifact_kind must be \"posterior_draws\"; pass output from `bayesite sample`",
        ));
    }
    let params = header
        .get("params")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed_fit("fit header needs a params array from `bayesite sample`"))?;
    let mut out = Vec::with_capacity(params.len());
    for param in params {
        let name = param
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed_fit("each fit params entry needs a string name"))?
            .to_string();
        let shape = parse_fit_shape(
            param
                .get("shape")
                .ok_or_else(|| malformed_fit(format!("fit parameter {name} needs a shape")))?,
            &name,
        )?;
        let size = fit_shape_size(&shape, &name)?;
        out.push(FitParamSpec { name, shape, size });
    }
    if out.is_empty() {
        return Err(malformed_fit(
            "fit header has no parameters; posterior-predictive needs a posterior draw stream",
        ));
    }
    Ok(out)
}

fn parse_fit_param_value(value: &Value, spec: &FitParamSpec) -> Result<Tensor, Error> {
    if spec.shape.is_empty() {
        let value = value.as_f64().ok_or_else(|| {
            malformed_fit(format!("fit draw value for {} must be a number", spec.name))
        })?;
        return Ok(Tensor::scalar(value));
    }
    let entries = value.as_array().ok_or_else(|| {
        malformed_fit(format!(
            "fit draw value for {} must be an array matching shape {:?}",
            spec.name, spec.shape
        ))
    })?;
    if entries.len() != spec.size {
        return Err(malformed_fit(format!(
            "fit draw value for {} has {} entries but shape {:?} needs {}",
            spec.name,
            entries.len(),
            spec.shape,
            spec.size
        )));
    }
    let data = entries
        .iter()
        .map(|entry| {
            entry.as_f64().ok_or_else(|| {
                malformed_fit(format!(
                    "fit draw value for {} contains a non-number",
                    spec.name
                ))
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(Tensor::from_vec(spec.shape.clone(), data))
}

fn parse_fit_draw_line(
    line: &Value,
    specs: &[FitParamSpec],
    expected_draw_index: usize,
    source_seed: i64,
) -> Result<FitDraw, Error> {
    let draw_index = line
        .get("draw_index")
        .and_then(Value::as_i64)
        .ok_or_else(|| malformed_fit("fit draw line needs integer draw_index"))?;
    if draw_index < 0 || draw_index as usize != expected_draw_index {
        return Err(malformed_fit(format!(
            "fit draw_index values must be contiguous from 0; expected {expected_draw_index}, got {draw_index}"
        )));
    }
    if line.get("draws_format").and_then(Value::as_str) != Some(V0_PROVISIONAL) {
        return Err(malformed_fit(
            "fit draw line draws_format must be \"v0-provisional\"; rerun `bayesite sample`",
        ));
    }
    if line.get("artifact_kind").and_then(Value::as_str) != Some(POSTERIOR_DRAWS.kind) {
        return Err(malformed_fit(
            "fit draw line artifact_kind must be \"posterior_draws\"; pass output from `bayesite sample`",
        ));
    }
    if line.get("draw_index_base").and_then(Value::as_str) != Some(POSTERIOR_DRAW_INDEX_BASE) {
        return Err(malformed_fit(
            "fit draw line draw_index_base must be \"zero_based_retained_draw_order\"",
        ));
    }
    if line.get("seed").and_then(Value::as_i64) != Some(source_seed) {
        return Err(malformed_fit(
            "fit draw line seed must match fit header seed; rerun `bayesite sample`",
        ));
    }
    let chain = line
        .get("chain")
        .and_then(Value::as_i64)
        .ok_or_else(|| malformed_fit("fit draw line needs integer chain"))?;
    if chain < 0 {
        return Err(malformed_fit("fit draw line chain must be non-negative"));
    }
    let draw = line
        .get("draw")
        .and_then(Value::as_i64)
        .ok_or_else(|| malformed_fit("fit draw line needs integer draw"))?;
    if draw < 0 {
        return Err(malformed_fit("fit draw line draw must be non-negative"));
    }
    let values = line
        .get("values")
        .ok_or_else(|| malformed_fit("fit draw line needs a values object"))?;
    let mut parsed = Vec::with_capacity(specs.len());
    for spec in specs {
        parsed.push(parse_fit_param_value(
            values.get(&spec.name).ok_or_else(|| {
                malformed_fit(format!("fit draw line is missing value for {}", spec.name))
            })?,
            spec,
        )?);
    }
    Ok(FitDraw {
        draw_index: draw_index as usize,
        chain,
        draw,
        values: parsed,
    })
}

fn parse_fit_stream(
    text: &str,
    expected_packing: &[(String, Vec<usize>)],
    expected_posterior_identity_hash: &str,
) -> Result<FitDrawStream, Error> {
    let mut lines = text.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| malformed_fit("fit is empty; pass NDJSON from `bayesite sample`"))?;
    let header = json::parse(header_line)?;
    let params = parse_fit_params(&header)?;
    if params.len() != expected_packing.len()
        || params
            .iter()
            .zip(expected_packing)
            .any(|(got, (name, shape))| got.name != *name || got.shape != *shape)
    {
        return Err(malformed_fit(
            "fit parameter order/shapes must match the model and data; rerun `bayesite sample` for this model/data pair",
        ));
    }
    if header
        .get("posterior_identity_hash")
        .and_then(Value::as_str)
        != Some(expected_posterior_identity_hash)
    {
        return Err(malformed_fit(
            "fit posterior_identity_hash must match the supplied model and data; rerun `bayesite sample` for these inputs",
        ));
    }
    let source_seed = header
        .get("seed")
        .and_then(Value::as_i64)
        .ok_or_else(|| malformed_fit("fit header needs integer seed from `bayesite sample`"))?;
    if source_seed < 0 {
        return Err(malformed_fit("fit header seed must be non-negative"));
    }
    let header_draw_count = header
        .get("draw_count")
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            malformed_fit("fit header needs integer draw_count from `bayesite sample`")
        })?;
    if header_draw_count < 1 {
        return Err(malformed_fit("fit header draw_count must be at least 1"));
    }
    let mut draws = Vec::new();
    let mut trailer: Option<Value> = None;
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(malformed_fit(format!(
                "line {} is blank; v0-provisional fit NDJSON has no blank lines",
                line_index + 2
            )));
        }
        let doc = json::parse(line)?;
        if let Some(value) = doc.get("trailer") {
            if trailer.is_some() {
                return Err(malformed_fit("fit has more than one trailer"));
            }
            trailer = Some(value.clone());
            continue;
        }
        if trailer.is_some() {
            return Err(malformed_fit("fit trailer must be the final line"));
        }
        draws.push(parse_fit_draw_line(
            &doc,
            &params,
            draws.len(),
            source_seed,
        )?);
    }
    let trailer = trailer.ok_or_else(|| {
        malformed_fit("fit is missing a trailer; rerun `bayesite sample` to completion")
    })?;
    if trailer.get("draws_format").and_then(Value::as_str) != Some(V0_PROVISIONAL) {
        return Err(malformed_fit(
            "fit trailer draws_format must be \"v0-provisional\"; rerun `bayesite sample`",
        ));
    }
    if trailer.get("artifact_kind").and_then(Value::as_str) != Some(POSTERIOR_DRAWS.kind) {
        return Err(malformed_fit(
            "fit trailer artifact_kind must be \"posterior_draws\"; pass output from `bayesite sample`",
        ));
    }
    if trailer.get("artifact_scope").and_then(Value::as_str) != Some(POSTERIOR_DRAWS.scope) {
        return Err(malformed_fit(
            "fit trailer artifact_scope must match posterior_draws sample output",
        ));
    }
    if trailer
        .get("posterior_identity_hash")
        .and_then(Value::as_str)
        != Some(expected_posterior_identity_hash)
    {
        return Err(malformed_fit(
            "fit trailer posterior_identity_hash must match the supplied model and data; rerun `bayesite sample` for these inputs",
        ));
    }
    if trailer.get("seed").and_then(Value::as_i64) != Some(source_seed) {
        return Err(malformed_fit(
            "fit trailer seed must match fit header seed; rerun `bayesite sample`",
        ));
    }
    let trailer_draw_count = trailer
        .get("draw_count")
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            malformed_fit("fit trailer needs integer draw_count from `bayesite sample`")
        })?;
    let parsed_draw_count = i64::try_from(draws.len())
        .map_err(|_| malformed_fit("fit draw_count must be reportable as a JSON integer"))?;
    if header_draw_count != parsed_draw_count || trailer_draw_count != parsed_draw_count {
        return Err(malformed_fit(
            "fit header/trailer draw_count must match parsed draw lines; rerun `bayesite sample` to completion",
        ));
    }
    if draws.is_empty() {
        return Err(malformed_fit(
            "fit has no draw lines; rerun `bayesite sample` with retained draws",
        ));
    }
    Ok(FitDrawStream {
        source_seed,
        params,
        draws,
    })
}

fn observed_data_names(meta: &ModelMeta) -> Vec<String> {
    meta.observed_nodes
        .iter()
        .map(|observed| observed.name.clone())
        .collect()
}

fn directly_assignable_observed_site_indices(
    observed_names: &[String],
    sites: &[crate::ir::ResolvedStochasticSite],
) -> Result<Vec<usize>, Error> {
    let mut indices = Vec::with_capacity(observed_names.len());
    let mut covered = vec![false; observed_names.len()];
    for (site_index, site) in sites.iter().enumerate() {
        let Expr::Data(name) = &site.value else {
            continue;
        };
        let Some(observed_index) = observed_names.iter().position(|observed| observed == name)
        else {
            continue;
        };
        if covered[observed_index] {
            return Err(invalid(format!(
                "posterior-predictive observed node \"{name}\" has more than one directly assignable stochastic site"
            )));
        }
        covered[observed_index] = true;
        indices.push(site_index);
    }
    for (observed_name, covered) in observed_names.iter().zip(covered) {
        if !covered {
            return Err(invalid(format!(
                "posterior-predictive observed node \"{observed_name}\" is not directly assignable; only DataRef observed stochastic sites are supported"
            )));
        }
    }
    Ok(indices)
}

fn full_data_map(data: &[(String, DataValue)]) -> Result<HashMap<String, DataValue>, Error> {
    let mut map = HashMap::new();
    for (name, value) in data {
        if map.insert(name.clone(), value.clone()).is_some() {
            return Err(mismatch(format!("duplicate data value \"{name}\"")));
        }
    }
    Ok(map)
}

fn declared_data_from_full(
    meta: &ModelMeta,
    data: &HashMap<String, DataValue>,
) -> Result<Vec<(String, DataValue)>, Error> {
    meta.data
        .iter()
        .map(|(name, _)| {
            Ok((
                name.clone(),
                data.get(name)
                    .cloned()
                    .ok_or_else(|| mismatch(format!("missing declared data value \"{name}\"")))?,
            ))
        })
        .collect()
}

fn posterior_predictive_workflow_phases_value() -> Value {
    Value::Array(
        [
            "parse_json",
            "decode_ir",
            "parse_fit_ndjson",
            "bind_observed_data",
            "simulate_posterior_predictive",
            "emit_artifact",
        ]
        .iter()
        .map(|phase| Value::Str((*phase).to_string()))
        .collect(),
    )
}

fn posterior_predictive_artifact_fields() -> Vec<(String, Value)> {
    let mut entries = vec![format_marker_field("posterior_predictive_format")];
    entries.extend(artifact_identity_entries(POSTERIOR_PREDICTIVE_DRAWS));
    entries
}

fn posterior_site_order_to_value(sites: &[PriorPredictiveSite]) -> Value {
    Value::Array(
        sites
            .iter()
            .map(|site| Value::Str(site.name.clone()))
            .collect(),
    )
}

fn posterior_predictive_header_value(
    run: &PosteriorPredictiveRun,
    seed: u64,
    declared_data: &[(String, DataValue)],
) -> Result<Value, Error> {
    let mut entries = posterior_predictive_artifact_fields();
    entries.extend([
        (
            "workflow_phases".to_string(),
            posterior_predictive_workflow_phases_value(),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("source_fit_seed".to_string(), Value::Int(run.source_seed)),
        ("draw_count".to_string(), Value::Int(run.draws.len() as i64)),
        (
            "draw_index_base".to_string(),
            Value::Str(POSTERIOR_DRAW_INDEX_BASE.to_string()),
        ),
        ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
        (
            "site_order".to_string(),
            posterior_site_order_to_value(&run.sites),
        ),
        (
            "declared_data_count".to_string(),
            Value::Int(declared_data.len() as i64),
        ),
        (
            "declared_data_order".to_string(),
            entry_order_value(declared_data),
        ),
        (
            "declared_data".to_string(),
            data_values_to_value(declared_data)?,
        ),
        (
            "sites".to_string(),
            Value::Array(
                run.sites
                    .iter()
                    .map(|site| {
                        Value::Object(vec![
                            ("name".to_string(), Value::Str(site.name.clone())),
                            (
                                "stochastic_site".to_string(),
                                Value::Str(site.stochastic_site.clone()),
                            ),
                            (
                                "role".to_string(),
                                Value::Str(site.role.as_str().to_string()),
                            ),
                            ("shape".to_string(), shape_value(&site.shape)),
                            ("integer".to_string(), Value::Bool(site.integer)),
                            (
                                "integer_by_coordinate".to_string(),
                                integer_flags_to_value(&site.shape, &site.integer_by_coordinate),
                            ),
                            (
                                "coordinate_order".to_string(),
                                coordinate_order_value(&site.shape),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
    ]);
    Ok(Value::Object(entries))
}

fn posterior_predictive_draw_value(
    draw_index: usize,
    draw: &PriorPredictiveDraw,
    sites: &[PriorPredictiveSite],
    seed: u64,
    source: &FitSourceDraw,
    draw_count: usize,
) -> Result<Value, Error> {
    let values = Value::Object(
        sites
            .iter()
            .zip(&draw.values)
            .map(|(site, (name, tensor))| {
                if site.name != *name {
                    return Err(invalid(
                        "posterior-predictive site metadata does not match generated values",
                    ));
                }
                Ok((
                    name.clone(),
                    tensor_to_value(
                        tensor,
                        &site.integer_by_coordinate,
                        "posterior-predictive artifact",
                    )?,
                ))
            })
            .collect::<Result<Vec<_>, Error>>()?,
    );
    let mut entries = posterior_predictive_artifact_fields();
    entries.extend([
        ("draw_index".to_string(), Value::Int(draw_index as i64)),
        (
            "draw_index_base".to_string(),
            Value::Str(POSTERIOR_DRAW_INDEX_BASE.to_string()),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("draw_count".to_string(), Value::Int(draw_count as i64)),
        (
            "source_fit_draw_index".to_string(),
            Value::Int(source.draw_index as i64),
        ),
        ("source_chain".to_string(), Value::Int(source.chain)),
        ("source_draw".to_string(), Value::Int(source.draw)),
        ("site_count".to_string(), Value::Int(sites.len() as i64)),
        (
            "site_order".to_string(),
            posterior_site_order_to_value(sites),
        ),
        ("values".to_string(), values),
    ]);
    Ok(Value::Object(entries))
}

fn posterior_predictive_trailer_value(
    run: &PosteriorPredictiveRun,
    seed: u64,
    declared_data: &[(String, DataValue)],
) -> Value {
    let mut entries = posterior_predictive_artifact_fields();
    entries.extend([
        (
            "workflow_phases".to_string(),
            posterior_predictive_workflow_phases_value(),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("source_fit_seed".to_string(), Value::Int(run.source_seed)),
        ("draw_count".to_string(), Value::Int(run.draws.len() as i64)),
        (
            "draw_index_base".to_string(),
            Value::Str(POSTERIOR_DRAW_INDEX_BASE.to_string()),
        ),
        ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
        (
            "site_order".to_string(),
            posterior_site_order_to_value(&run.sites),
        ),
        (
            "declared_data_count".to_string(),
            Value::Int(declared_data.len() as i64),
        ),
        (
            "declared_data_order".to_string(),
            entry_order_value(declared_data),
        ),
    ]);
    Value::Object(entries)
}

fn distribution_has_integer_support(distribution: &Distribution) -> bool {
    matches!(
        distribution,
        Distribution::Bernoulli { .. }
            | Distribution::Poisson { .. }
            | Distribution::Binomial { .. }
            | Distribution::BetaBinomial { .. }
            | Distribution::NegativeBinomial { .. }
            | Distribution::OrderedLogistic { .. }
    )
}

fn distribution_integer_flags(distribution: &Distribution, value: &Tensor) -> Vec<bool> {
    let integer = distribution_has_integer_support(distribution);
    value.data().iter().map(|_| integer).collect()
}

fn include_expr_shape(
    env: &ForwardEnv<'_>,
    shape: &mut Vec<usize>,
    expr: &Expr,
) -> Result<(), Error> {
    let value = env.evaluate(expr)?;
    *shape = Tensor::broadcast_shapes(shape, value.shape())?;
    Ok(())
}

fn posterior_predictive_target_shape(
    env: &ForwardEnv<'_>,
    distribution: &Distribution,
    observed_shape: &[usize],
) -> Result<Vec<usize>, Error> {
    let mut shape = observed_shape.to_vec();
    match distribution {
        Distribution::Normal { loc, scale } => {
            include_expr_shape(env, &mut shape, loc)?;
            include_expr_shape(env, &mut shape, scale)?;
        }
        Distribution::HalfNormal { scale } => include_expr_shape(env, &mut shape, scale)?,
        Distribution::StudentT { df, loc, scale } => {
            include_expr_shape(env, &mut shape, df)?;
            include_expr_shape(env, &mut shape, loc)?;
            include_expr_shape(env, &mut shape, scale)?;
        }
        Distribution::Exponential { rate } => include_expr_shape(env, &mut shape, rate)?,
        Distribution::Uniform { low, high } => {
            include_expr_shape(env, &mut shape, low)?;
            include_expr_shape(env, &mut shape, high)?;
        }
        Distribution::Beta { alpha, beta } => {
            include_expr_shape(env, &mut shape, alpha)?;
            include_expr_shape(env, &mut shape, beta)?;
        }
        Distribution::Bernoulli { probs } => include_expr_shape(env, &mut shape, probs)?,
        Distribution::Poisson { rate } => include_expr_shape(env, &mut shape, rate)?,
        Distribution::Binomial { total_count, probs } => {
            include_expr_shape(env, &mut shape, total_count)?;
            include_expr_shape(env, &mut shape, probs)?;
        }
        Distribution::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => {
            include_expr_shape(env, &mut shape, total_count)?;
            include_expr_shape(env, &mut shape, alpha)?;
            include_expr_shape(env, &mut shape, beta)?;
        }
        Distribution::NegativeBinomial {
            mean,
            overdispersion,
        } => {
            include_expr_shape(env, &mut shape, mean)?;
            include_expr_shape(env, &mut shape, overdispersion)?;
        }
        Distribution::MultivariateNormal { mean, scale_tril } => {
            let mean = env.evaluate(mean)?;
            let scale_tril = env.evaluate(scale_tril)?;
            if scale_tril.shape().len() == 2 {
                let event_shape = vec![scale_tril.shape()[0]];
                shape = Tensor::broadcast_shapes(&shape, &event_shape)?;
            }
            shape = Tensor::broadcast_shapes(&shape, mean.shape())?;
        }
        Distribution::OrderedLogistic { eta, cutpoints: _ } => {
            include_expr_shape(env, &mut shape, eta)?;
        }
    }
    Ok(shape)
}

/// Simulate replicated observed values from retained posterior draws.
pub fn simulate_posterior_predictive(
    meta: ModelMeta,
    data: Vec<(String, DataValue)>,
    fit_ndjson: &str,
    seed: u64,
) -> Result<PosteriorPredictiveRun, Error> {
    validate_reportable_seed(seed, "posterior-predictive artifact")?;
    let posterior = Posterior::new(meta.clone(), data.clone())?;
    let packing = posterior.packing();
    let fit = parse_fit_stream(fit_ndjson, &packing, posterior.identity_hash())?;
    let data_map = full_data_map(&data)?;
    let declared_data = declared_data_from_full(&meta, &data_map)?;
    let declared_map = bind_declared_data(&meta, declared_data)?;
    let observed_names = observed_data_names(&meta);
    if observed_names.is_empty() {
        return Err(invalid(
            "posterior-predictive needs at least one observed node to simulate",
        ));
    }
    let sites = meta.resolved_stochastic_sites();
    let observed_site_indices = directly_assignable_observed_site_indices(&observed_names, &sites)?;
    let mut rng = Xoshiro256PlusPlus::for_chain(seed, 0);
    let mut draws = Vec::with_capacity(fit.draws.len());
    let mut site_specs: Option<Vec<PriorPredictiveSite>> = None;

    for fit_draw in &fit.draws {
        let mut env = ForwardEnv {
            values: HashMap::new(),
            data: &declared_map,
        };
        for (spec, value) in fit.params.iter().zip(&fit_draw.values) {
            env.values.insert(spec.name.clone(), value.clone());
        }
        let mut current_sites = Vec::new();
        let mut values = Vec::new();
        for &site_index in &observed_site_indices {
            let site = &sites[site_index];
            let Expr::Data(name) = &site.value else {
                unreachable!("observed_site_indices only contains DataRef sites");
            };
            let observed = data_map.get(name).ok_or_else(|| {
                mismatch(format!(
                    "posterior-predictive missing observed data value \"{name}\""
                ))
            })?;
            let target_shape =
                posterior_predictive_target_shape(&env, &site.distribution, &observed.shape)?;
            let value =
                sample_distribution(&mut rng, &env, &site.distribution, Some(&target_shape))?;
            env.values.insert(name.clone(), value.clone());
            current_sites.push(PriorPredictiveSite {
                name: name.clone(),
                stochastic_site: site.name.clone(),
                role: PriorPredictiveRole::Observed,
                shape: value.shape().to_vec(),
                integer: distribution_has_integer_support(&site.distribution),
                integer_by_coordinate: distribution_integer_flags(&site.distribution, &value),
            });
            values.push((name.clone(), value));
        }
        if current_sites.is_empty() {
            return Err(invalid(
                "posterior-predictive currently supports directly assignable observed stochastic sites only",
            ));
        }
        match &site_specs {
            None => site_specs = Some(current_sites),
            Some(expected) if expected.len() == current_sites.len() => {
                for (expected, got) in expected.iter().zip(&current_sites) {
                    if expected.name != got.name
                        || expected.stochastic_site != got.stochastic_site
                        || expected.shape != got.shape
                        || expected.integer != got.integer
                        || expected.integer_by_coordinate != got.integer_by_coordinate
                    {
                        return Err(mismatch(
                            "posterior-predictive site metadata changed across draws; dynamic-shape streams are not supported",
                        ));
                    }
                }
            }
            Some(_) => {
                return Err(mismatch(
                    "posterior-predictive site count changed across draws; dynamic stochastic structure is not supported",
                ))
            }
        }
        draws.push(PriorPredictiveDraw { values });
    }

    let source_draws = fit
        .draws
        .iter()
        .map(|draw| FitSourceDraw {
            draw_index: draw.draw_index,
            chain: draw.chain,
            draw: draw.draw,
        })
        .collect();
    Ok(PosteriorPredictiveRun {
        sites: site_specs.unwrap_or_default(),
        draws,
        source_seed: fit.source_seed,
        source_draws,
    })
}

pub fn posterior_predictive_ndjson_lines(
    meta: ModelMeta,
    data: Vec<(String, DataValue)>,
    fit_ndjson: &str,
    seed: u64,
) -> Result<Vec<String>, Error> {
    let data_map = full_data_map(&data)?;
    let declared_data = declared_data_from_full(&meta, &data_map)?;
    let run = simulate_posterior_predictive(meta, data, fit_ndjson, seed)?;
    let mut lines = Vec::with_capacity(run.draws.len() + 2);
    lines.push(json::write(&posterior_predictive_header_value(
        &run,
        seed,
        &declared_data,
    )?)?);
    for (draw_index, draw) in run.draws.iter().enumerate() {
        let source = run.source_draws.get(draw_index).ok_or_else(|| {
            malformed_fit("posterior-predictive source fit draw metadata is missing")
        })?;
        lines.push(json::write(&posterior_predictive_draw_value(
            draw_index,
            draw,
            &run.sites,
            seed,
            source,
            run.draws.len(),
        )?)?);
    }
    lines.push(json::write(&Value::Object(vec![(
        "trailer".to_string(),
        posterior_predictive_trailer_value(&run, seed, &declared_data),
    )]))?);
    Ok(lines)
}

fn statistic_value(name: &str, values: &[f64]) -> Option<f64> {
    match name {
        "mean" if values.is_empty() => None,
        "mean" => Some(values.iter().sum::<f64>() / values.len() as f64),
        "sd" if values.is_empty() => None,
        "sd" => {
            let mean = statistic_value("mean", values).expect("non-empty values have a mean");
            Some(
                (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64)
                    .sqrt(),
            )
        }
        "min" if values.is_empty() => None,
        "min" => Some(values.iter().fold(f64::INFINITY, |a, &b| a.min(b))),
        "max" if values.is_empty() => None,
        "max" => Some(values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
        "zero_count" => Some(values.iter().filter(|&&v| v == 0.0).count() as f64),
        _ => None,
    }
}

fn optional_float_value(value: Option<f64>) -> Value {
    match value {
        Some(value) if value.is_finite() => Value::Float(value),
        _ => Value::Null,
    }
}

fn optional_int_count_value(value: Option<usize>) -> Value {
    match value {
        Some(value) => Value::Int(value as i64),
        None => Value::Null,
    }
}

fn stat_summary_value(observed: Option<f64>, replicated: &[Option<f64>]) -> Value {
    let replicated_values: Vec<f64> = replicated.iter().filter_map(|value| *value).collect();
    let less_equal = observed.map(|observed| {
        replicated_values
            .iter()
            .filter(|&&value| value <= observed)
            .count()
    });
    let greater_equal = observed.map(|observed| {
        replicated_values
            .iter()
            .filter(|&&value| value >= observed)
            .count()
    });
    let mut sorted = replicated_values.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    Value::Object(vec![
        ("observed".to_string(), optional_float_value(observed)),
        (
            "replicated_mean".to_string(),
            optional_float_value(statistic_value("mean", &replicated_values)),
        ),
        (
            "replicated_min".to_string(),
            optional_float_value(sorted.first().copied()),
        ),
        (
            "replicated_max".to_string(),
            optional_float_value(sorted.last().copied()),
        ),
        (
            "count_replicated_less_equal_observed".to_string(),
            optional_int_count_value(less_equal),
        ),
        (
            "count_replicated_greater_equal_observed".to_string(),
            optional_int_count_value(greater_equal),
        ),
        (
            "replicated_draw_count".to_string(),
            Value::Int(replicated.len() as i64),
        ),
        (
            "replicated_finite_stat_count".to_string(),
            Value::Int(replicated_values.len() as i64),
        ),
    ])
}

pub fn posterior_check_report(
    meta: ModelMeta,
    data: Vec<(String, DataValue)>,
    fit_ndjson: &str,
    seed: u64,
) -> Result<String, Error> {
    let data_map = full_data_map(&data)?;
    let run = simulate_posterior_predictive(meta, data, fit_ndjson, seed)?;
    let mut checks = Vec::new();
    for (site_idx, site) in run.sites.iter().enumerate() {
        let observed = data_map.get(&site.name).ok_or_else(|| {
            mismatch(format!(
                "posterior-check missing observed data value \"{}\"",
                site.name
            ))
        })?;
        let observed_tensor = Tensor::from_vec(observed.shape.clone(), observed.values.clone())
            .broadcast_to(&site.shape)
            .map_err(|_| {
                mismatch(format!(
                    "posterior-check observed data value \"{}\" cannot broadcast from shape {:?} to posterior-predictive site shape {:?}",
                    site.name, observed.shape, site.shape
                ))
            })?;
        let mut statistic_names = vec!["mean", "sd", "min", "max"];
        if site.integer {
            statistic_names.push("zero_count");
        }
        for statistic in statistic_names {
            let observed_stat = statistic_value(statistic, observed_tensor.data());
            let replicated_stats = run
                .draws
                .iter()
                .map(|draw| statistic_value(statistic, draw.values[site_idx].1.data()))
                .collect::<Vec<_>>();
            checks.push(Value::Object(vec![
                ("site".to_string(), Value::Str(site.name.clone())),
                (
                    "stochastic_site".to_string(),
                    Value::Str(site.stochastic_site.clone()),
                ),
                ("statistic".to_string(), Value::Str(statistic.to_string())),
                ("shape".to_string(), shape_value(&site.shape)),
                (
                    "coordinate_order".to_string(),
                    coordinate_order_value(&site.shape),
                ),
                (
                    "summary".to_string(),
                    stat_summary_value(observed_stat, &replicated_stats),
                ),
            ]));
        }
    }
    let report = Value::Object(vec![
        (
            "posterior_check_format".to_string(),
            Value::Str(V0_PROVISIONAL.to_string()),
        ),
        (
            "workflow_format".to_string(),
            Value::Str(WORKFLOW_FORMAT.to_string()),
        ),
        (
            "report_kind".to_string(),
            Value::Str("posterior_predictive_check_facts".to_string()),
        ),
        (
            "report_scope".to_string(),
            Value::Str("observed_data_vs_posterior_predictive_replicates".to_string()),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        (
            "posterior_predictive_draws".to_string(),
            Value::Int(run.draws.len() as i64),
        ),
        (
            "posterior_predictive_draws_artifact_kind".to_string(),
            Value::Str(POSTERIOR_PREDICTIVE_DRAWS.kind.to_string()),
        ),
        (
            "posterior_predictive_draws_artifact_scope".to_string(),
            Value::Str(POSTERIOR_PREDICTIVE_DRAWS.scope.to_string()),
        ),
        ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
        (
            "site_order".to_string(),
            posterior_site_order_to_value(&run.sites),
        ),
        ("checks".to_string(), Value::Array(checks)),
    ]);
    json::write(&report)
}
