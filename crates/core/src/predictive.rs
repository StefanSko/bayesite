//! Prior-predictive simulation over decoded IR.
//!
//! This is a pure runtime layer: callers provide decoded `ModelMeta`, bound
//! data, draw count, and seed; the module returns v0-provisional NDJSON lines.
//! No filesystem, clocks, global entropy, Python, or producer code are used.

use std::collections::HashMap;

use crate::error::{Error, ErrorKind};
use crate::ir::{
    BinOpKind, Constraint, DataSchema, Dim, Distribution, Expr, IndexSpec, ModelMeta, Size, UnaryFn,
};
use crate::json::{self, Value};
use crate::model::DataValue;
use crate::rng::Xoshiro256PlusPlus;
use crate::tensor::{gather_map, IndexAtom, Tensor};

const PRIOR_PREDICTIVE_DRAW_INDEX_BASE: &str = "zero_based_prior_predictive_draw_order";

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
    let mut shape = target.map_or_else(Vec::new, |shape| shape.to_vec());
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

fn integer_flags_to_value(shape: &[usize], flags: &[bool]) -> Value {
    if shape.is_empty() {
        Value::Bool(flags.first().copied().unwrap_or(false))
    } else {
        Value::Array(flags.iter().copied().map(Value::Bool).collect())
    }
}

fn data_order_to_value(data: &[(String, DataValue)]) -> Value {
    Value::Array(
        data.iter()
            .map(|(name, _)| Value::Str(name.clone()))
            .collect(),
    )
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
            .map(|(name, value)| {
                (
                    name.clone(),
                    Value::Array(
                        value
                            .shape
                            .iter()
                            .map(|&dim| Value::Int(dim as i64))
                            .collect(),
                    ),
                )
            })
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

fn header_value(
    sites: &[PriorPredictiveSite],
    settings: &PriorPredictiveSettings,
    seed: u64,
    declared_data: &[(String, DataValue)],
) -> Result<Value, Error> {
    Ok(Value::Object(vec![
        (
            "prior_predictive_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "artifact_kind".to_string(),
            Value::Str("prior_predictive_draws".to_string()),
        ),
        (
            "artifact_scope".to_string(),
            Value::Str("declared_data_conditioned_site_draws".to_string()),
        ),
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
            data_order_to_value(declared_data),
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
                            (
                                "shape".to_string(),
                                Value::Array(
                                    site.shape.iter().map(|&d| Value::Int(d as i64)).collect(),
                                ),
                            ),
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
    ]))
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
        let line = Value::Object(vec![
            (
                "prior_predictive_format".to_string(),
                Value::Str("v0-provisional".to_string()),
            ),
            (
                "artifact_kind".to_string(),
                Value::Str("prior_predictive_draws".to_string()),
            ),
            (
                "artifact_scope".to_string(),
                Value::Str("declared_data_conditioned_site_draws".to_string()),
            ),
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
                data_order_to_value(&declared_data),
            ),
            ("site_count".to_string(), Value::Int(run.sites.len() as i64)),
            ("site_order".to_string(), site_order_to_value(&run.sites)),
            ("values".to_string(), values),
        ]);
        lines.push(json::write(&line)?);
    }
    lines.push(json::write(&Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(vec![
            (
                "prior_predictive_format".to_string(),
                Value::Str("v0-provisional".to_string()),
            ),
            (
                "artifact_kind".to_string(),
                Value::Str("prior_predictive_draws".to_string()),
            ),
            (
                "artifact_scope".to_string(),
                Value::Str("declared_data_conditioned_site_draws".to_string()),
            ),
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
                data_order_to_value(&declared_data),
            ),
            ("sites".to_string(), Value::Int(run.sites.len() as i64)),
        ]),
    )]))?);
    Ok(lines)
}
