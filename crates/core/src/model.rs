//! Bound model evaluation: data binding, constraint transforms, and the
//! log density with its gradient.
//!
//! Mirrors `src/jaxstanv5/compiler/core.py`: the unconstrained vector `q`
//! is split per the packing-order guarantee, constraints contribute their
//! log-Jacobians, and stochastic sites accumulate in document order.

use std::collections::HashMap;

use crate::density::{self, DistVars};
use crate::error::{Error, ErrorKind};
use crate::ir::{
    BinOpKind, Constraint, DataSchema, Dim, Distribution, Expr, IndexSpec, ModelMeta,
    ResolvedStochasticSite, Size, UnaryFn,
};
use crate::json::Value;
use crate::tape::{Tape, Var};
use crate::tensor::{gather_map, slice_last_map, IndexAtom, Tensor};

/// A bound data value: an f64 tensor plus its declared integerness.
#[derive(Debug, Clone)]
pub struct DataValue {
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub integer: bool,
}

fn mismatch(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataShapeMismatch, message)
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

/// Parse the data document convention used by the fixture corpus and the
/// CLI: `{"<name>": {"dtype": "...", "shape": [...], "values": [...]}}`.
/// A bare number or array is accepted as float64 shorthand.
pub fn data_from_json(document: &Value) -> Result<Vec<(String, DataValue)>, Error> {
    let Value::Object(entries) = document else {
        return Err(mismatch(
            "the data document must be a JSON object keyed by data name",
        ));
    };
    let mut out = Vec::with_capacity(entries.len());
    for (name, spec) in entries {
        out.push((name.clone(), data_value_from_json(name, spec)?));
    }
    Ok(out)
}

fn collect_numbers(name: &str, value: &Value, into: &mut Vec<f64>) -> Result<(), Error> {
    match value {
        Value::Int(i) => {
            into.push(*i as f64);
            Ok(())
        }
        Value::Float(f) => {
            into.push(*f);
            Ok(())
        }
        _ => Err(mismatch(format!(
            "data value \"{name}\" must contain numbers only"
        ))),
    }
}

fn data_value_from_json(name: &str, spec: &Value) -> Result<DataValue, Error> {
    match spec {
        Value::Int(i) => Ok(DataValue {
            shape: vec![],
            values: vec![*i as f64],
            integer: true,
        }),
        Value::Float(f) => Ok(DataValue {
            shape: vec![],
            values: vec![*f],
            integer: false,
        }),
        Value::Array(items) => {
            // Bare (possibly nested) array shorthand; integer iff all ints.
            let mut shape = Vec::new();
            let mut probe = spec;
            while let Value::Array(inner) = probe {
                shape.push(inner.len());
                match inner.first() {
                    Some(first) => probe = first,
                    None => break,
                }
            }
            let mut values = Vec::new();
            let mut integer = true;
            fn walk(
                name: &str,
                value: &Value,
                depth: usize,
                shape: &[usize],
                values: &mut Vec<f64>,
                integer: &mut bool,
            ) -> Result<(), Error> {
                if depth < shape.len() {
                    let Value::Array(items) = value else {
                        return Err(mismatch(format!(
                            "data value \"{name}\" must be a rectangular array"
                        )));
                    };
                    if items.len() != shape[depth] {
                        return Err(mismatch(format!(
                            "data value \"{name}\" must be a rectangular array"
                        )));
                    }
                    for item in items {
                        walk(name, item, depth + 1, shape, values, integer)?;
                    }
                    Ok(())
                } else {
                    if matches!(value, Value::Float(_)) {
                        *integer = false;
                    }
                    collect_numbers(name, value, values)
                }
            }
            let _ = items;
            walk(name, spec, 0, &shape, &mut values, &mut integer)?;
            Ok(DataValue {
                shape,
                values,
                integer,
            })
        }
        Value::Object(_) => {
            let dtype = spec
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| mismatch(format!("data value \"{name}\" needs a dtype string")))?;
            let integer = dtype.starts_with("int") || dtype.starts_with("uint");
            let shape: Vec<usize> = spec
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| mismatch(format!("data value \"{name}\" needs a shape array")))?
                .iter()
                .map(|d| {
                    d.as_i64()
                        .filter(|&d| d >= 0)
                        .map(|d| d as usize)
                        .ok_or_else(|| {
                            mismatch(format!(
                                "data value \"{name}\" shape entries must be non-negative integers"
                            ))
                        })
                })
                .collect::<Result<_, _>>()?;
            let values_field = spec
                .get("values")
                .ok_or_else(|| mismatch(format!("data value \"{name}\" needs a values field")))?;
            let mut values = Vec::new();
            match values_field {
                Value::Array(items) => {
                    for item in items {
                        collect_numbers(name, item, &mut values)?;
                    }
                }
                other => collect_numbers(name, other, &mut values)?,
            }
            let expected: usize = shape.iter().product();
            if values.len() != expected {
                return Err(mismatch(format!(
                    "data value \"{name}\" has {} values but shape {shape:?} needs {expected}",
                    values.len()
                )));
            }
            Ok(DataValue {
                shape,
                values,
                integer,
            })
        }
        _ => Err(mismatch(format!(
            "data value \"{name}\" must be a number, an array, or a dtype/shape/values object"
        ))),
    }
}

#[derive(Debug)]
struct FreeSlot {
    name: String,
    constraint: Option<Constraint>,
    shape: Vec<usize>,
    offset: usize,
    size: usize,
}

/// A model bound to concrete data; evaluates `logp` and its gradient at
/// unconstrained points. Pure: no interior mutability, no I/O.
#[derive(Debug)]
pub struct Posterior {
    free: Vec<FreeSlot>,
    sites: Vec<ResolvedStochasticSite>,
    data: HashMap<String, DataValue>,
    n_params: usize,
}

impl Posterior {
    pub fn new(meta: ModelMeta, data: Vec<(String, DataValue)>) -> Result<Posterior, Error> {
        let mut data_map: HashMap<String, DataValue> = HashMap::new();
        for (name, value) in data {
            if data_map.insert(name.clone(), value).is_some() {
                return Err(mismatch(format!("duplicate data value \"{name}\"")));
            }
        }

        // Expected names: declared data plus observed values (mirrors bind()).
        let mut expected: Vec<&str> = meta.data.iter().map(|(n, _)| n.as_str()).collect();
        expected.extend(meta.observed_nodes.iter().map(|o| o.name.as_str()));
        let mut missing: Vec<&str> = expected
            .iter()
            .filter(|n| !data_map.contains_key(**n))
            .copied()
            .collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(mismatch(format!(
                "missing model data: {missing:?}; bind every declared data and observed value"
            )));
        }
        let mut extra: Vec<&String> = data_map
            .keys()
            .filter(|n| !expected.contains(&n.as_str()))
            .collect();
        extra.sort_unstable();
        if !extra.is_empty() {
            return Err(mismatch(format!(
                "unexpected model data: {extra:?}; the model does not declare these names"
            )));
        }

        // Declared data schema validation.
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
                                "data \"{name}\" axis {axis} must have length {expected}, \
                                 got {}",
                                value.shape[axis]
                            )));
                        }
                    }
                }
            }
        }

        // Free-value shapes per the packing-order guarantee.
        let mut free = Vec::new();
        let mut offset = 0usize;
        for (name, free_value) in meta.resolved_free_values() {
            let shape: Vec<usize> = match &free_value.size {
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
                    let k = scalar_int_data(&data_map, ref_name)?;
                    if k < 1 {
                        return Err(mismatch(format!(
                            "data-dependent parameter size \"{ref_name}\" must be a positive \
                             integer, got {k}"
                        )));
                    }
                    vec![k as usize]
                }
            };
            let size: usize = shape.iter().product::<usize>().max(1);
            free.push(FreeSlot {
                name,
                constraint: free_value.constraint,
                shape,
                offset,
                size,
            });
            offset += size;
        }

        Ok(Posterior {
            free,
            sites: meta.resolved_stochastic_sites(),
            data: data_map,
            n_params: offset,
        })
    }

    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Packing order: name and shape per free value.
    pub fn packing(&self) -> Vec<(String, Vec<usize>)> {
        self.free
            .iter()
            .map(|slot| (slot.name.clone(), slot.shape.clone()))
            .collect()
    }

    /// Log density and gradient at the unconstrained point `q`.
    pub fn logp_grad(&self, q: &[f64]) -> Result<(f64, Vec<f64>), Error> {
        let (tape, root, leaves) = self.build_logp(q)?;
        let logp = tape.value(root).data()[0];
        let grads = tape.backward(root, &leaves);
        let mut grad = Vec::with_capacity(self.n_params);
        for tensor in grads {
            grad.extend_from_slice(tensor.data());
        }
        Ok((logp, grad))
    }

    /// Log density only (forward pass).
    pub fn logp(&self, q: &[f64]) -> Result<f64, Error> {
        let (tape, root, _) = self.build_logp(q)?;
        Ok(tape.value(root).data()[0])
    }

    /// Constrained values per free value, in packing order.
    pub fn constrain(&self, q: &[f64]) -> Result<Vec<(String, Tensor)>, Error> {
        self.validate_q(q)?;
        let mut tape = Tape::new();
        let mut out = Vec::with_capacity(self.free.len());
        for slot in &self.free {
            let leaf = tape.constant(Tensor::from_vec(
                slot.shape.clone(),
                q[slot.offset..slot.offset + slot.size].to_vec(),
            ));
            let constrained = apply_constraint(&mut tape, slot, leaf)?.0;
            out.push((slot.name.clone(), tape.value(constrained).clone()));
        }
        Ok(out)
    }

    fn validate_q(&self, q: &[f64]) -> Result<(), Error> {
        if q.len() != self.n_params {
            return Err(mismatch(format!(
                "unconstrained parameter vector q has wrong length: expected {}, got {}",
                self.n_params,
                q.len()
            )));
        }
        Ok(())
    }

    fn build_logp(&self, q: &[f64]) -> Result<(Tape, Var, Vec<Var>), Error> {
        self.validate_q(q)?;
        let mut tape = Tape::new();
        let mut leaves = Vec::with_capacity(self.free.len());
        let mut values: HashMap<String, Var> = HashMap::new();

        // Constrain and accumulate log-Jacobians in packing order.
        let mut log_jac = tape.constant(Tensor::scalar(0.0));
        for slot in &self.free {
            let leaf = tape.input(Tensor::from_vec(
                slot.shape.clone(),
                q[slot.offset..slot.offset + slot.size].to_vec(),
            ));
            leaves.push(leaf);
            let (constrained, jacobian) = apply_constraint(&mut tape, slot, leaf)?;
            if let Some(jacobian) = jacobian {
                let total = tape.sum(jacobian);
                log_jac = tape.add(log_jac, total);
            }
            values.insert(slot.name.clone(), constrained);
        }

        let mut env = Env {
            tape,
            values,
            data: &self.data,
            data_vars: HashMap::new(),
        };

        let mut lp = log_jac;
        for site in &self.sites {
            let dist = env.evaluate_distribution(&site.distribution)?;
            let value = env.evaluate(&site.value)?;
            let site_lp = density::log_prob(&mut env.tape, &dist, value)?;
            let total = env.tape.sum(site_lp);
            lp = env.tape.add(lp, total);
        }
        Ok((env.tape, lp, leaves))
    }
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

/// Constrained variable and optional elementwise log-Jacobian.
fn apply_constraint(
    tape: &mut Tape,
    slot: &FreeSlot,
    leaf: Var,
) -> Result<(Var, Option<Var>), Error> {
    match &slot.constraint {
        None => Ok((leaf, None)),
        Some(Constraint::Positive) => {
            let constrained = tape.exp(leaf);
            Ok((constrained, Some(leaf)))
        }
        Some(Constraint::UnitInterval) => Ok(interval_constraint(tape, leaf, 0.0, 1.0)),
        Some(Constraint::Interval { lower, upper }) => {
            Ok(interval_constraint(tape, leaf, *lower, *upper))
        }
        Some(Constraint::Ordered) => {
            if tape.value(leaf).rank() != 1 {
                return Err(mismatch(format!(
                    "Ordered constraint on \"{}\" requires vector values",
                    slot.name
                )));
            }
            let constrained = tape.ordered_inverse(leaf);
            let n = tape.value(leaf).len();
            let tail = tape.gather(leaf, slice_last_map(&[n], 1, n));
            Ok((constrained, Some(tail)))
        }
    }
}

fn interval_constraint(tape: &mut Tape, leaf: Var, lower: f64, upper: f64) -> (Var, Option<Var>) {
    let width = upper - lower;
    // inverse: lower + width * sigmoid(y)
    let sig = tape.sigmoid(leaf);
    let width_c = tape.constant(Tensor::scalar(width));
    let scaled = tape.mul(width_c, sig);
    let lower_c = tape.constant(Tensor::scalar(lower));
    let constrained = tape.add(lower_c, scaled);
    // log|J|: log(width) - softplus(-y) - softplus(y)
    let log_width = tape.constant(Tensor::scalar(width.ln()));
    let neg_leaf = tape.neg(leaf);
    let sp_neg = tape.softplus(neg_leaf);
    let term = tape.sub(log_width, sp_neg);
    let sp_pos = tape.softplus(leaf);
    let jacobian = tape.sub(term, sp_pos);
    (constrained, Some(jacobian))
}

/// Expression evaluation environment over one tape.
struct Env<'a> {
    tape: Tape,
    values: HashMap<String, Var>,
    data: &'a HashMap<String, DataValue>,
    data_vars: HashMap<String, Var>,
}

impl<'a> Env<'a> {
    fn data_var(&mut self, name: &str) -> Result<Var, Error> {
        if let Some(var) = self.data_vars.get(name) {
            return Ok(*var);
        }
        let value = self
            .data
            .get(name)
            .ok_or_else(|| malformed(format!("reference to unknown data value \"{name}\"")))?;
        let tensor = Tensor::from_vec(value.shape.clone(), value.values.clone());
        let var = self.tape.constant(tensor);
        self.data_vars.insert(name.to_string(), var);
        Ok(var)
    }

    /// Name lookup: bound data shadows constrained params, mirroring
    /// `values = {**constrained, **bound.data}` in the Python compiler.
    fn name_var(&mut self, name: &str) -> Result<Var, Error> {
        if self.data.contains_key(name) {
            return self.data_var(name);
        }
        self.values
            .get(name)
            .copied()
            .ok_or_else(|| malformed(format!("reference to unknown value \"{name}\"")))
    }

    fn evaluate(&mut self, expr: &Expr) -> Result<Var, Error> {
        match expr {
            Expr::Param(name) | Expr::Data(name) => self.name_var(name),
            Expr::Const(v) => Ok(self.tape.constant(Tensor::scalar(*v))),
            Expr::Bin { op, left, right } => {
                let l = self.evaluate(left)?;
                let r = self.evaluate(right)?;
                Ok(match op {
                    BinOpKind::Add => self.tape.add(l, r),
                    BinOpKind::Sub => self.tape.sub(l, r),
                    BinOpKind::Mul => self.tape.mul(l, r),
                    BinOpKind::Div => self.tape.div(l, r),
                })
            }
            Expr::Unary { function, operand } => {
                let v = self.evaluate(operand)?;
                Ok(match function {
                    UnaryFn::Exp => self.tape.exp(v),
                    UnaryFn::Neg => self.tape.neg(v),
                    UnaryFn::Sigmoid => self.tape.sigmoid(v),
                })
            }
            Expr::Index { base, index } => {
                let base_var = self.evaluate(base)?;
                let atoms = self.evaluate_index_spec(index)?;
                let map = gather_map(self.tape.value(base_var).shape(), &atoms)?;
                Ok(self.tape.gather(base_var, map))
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
                let wrap = |positions: Vec<i64>| -> Result<Vec<usize>, Error> {
                    positions
                        .into_iter()
                        .map(|p| {
                            let wrapped = if p < 0 { p + len as i64 } else { p };
                            if wrapped < 0 || wrapped >= len as i64 {
                                Err(mismatch(format!(
                                    "scatter index {p} is out of bounds for length {len}"
                                )))
                            } else {
                                Ok(wrapped as usize)
                            }
                        })
                        .collect()
                };
                let obs_pos = wrap(obs_pos)?;
                let mis_pos = wrap(mis_pos)?;
                if self.tape.value(obs_values).len() != obs_pos.len()
                    || self.tape.value(mis_values).len() != mis_pos.len()
                {
                    return Err(mismatch(
                        "scatter values must match their index vectors in length",
                    ));
                }
                Ok(self
                    .tape
                    .scatter(len, vec![(obs_values, obs_pos), (mis_values, mis_pos)]))
            }
        }
    }

    /// Evaluate an index expression: must be parameter-free and integral.
    fn index_values(&mut self, expr: &Expr) -> Result<(Vec<usize>, Vec<i64>), Error> {
        let var = self.evaluate(expr)?;
        if self.tape.requires_grad(var) {
            return Err(malformed("index expressions must not depend on parameters"));
        }
        let tensor = self.tape.value(var);
        let mut ints = Vec::with_capacity(tensor.len());
        for &v in tensor.data() {
            if v.fract() != 0.0 {
                return Err(mismatch(format!("index values must be integers, got {v}")));
            }
            ints.push(v as i64);
        }
        Ok((tensor.shape().to_vec(), ints))
    }

    fn index_vector(&mut self, expr: &Expr) -> Result<Vec<i64>, Error> {
        let (shape, ints) = self.index_values(expr)?;
        if shape.len() != 1 {
            return Err(mismatch(format!(
                "scatter index vectors must be rank-1, got shape {shape:?}"
            )));
        }
        Ok(ints)
    }

    fn evaluate_index_spec(&mut self, spec: &IndexSpec) -> Result<Vec<IndexAtom>, Error> {
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

    fn evaluate_distribution(&mut self, dist: &Distribution) -> Result<DistVars, Error> {
        Ok(match dist {
            Distribution::Normal { loc, scale } => DistVars::Normal {
                loc: self.evaluate(loc)?,
                scale: self.evaluate(scale)?,
            },
            Distribution::HalfNormal { scale } => DistVars::HalfNormal {
                scale: self.evaluate(scale)?,
            },
            Distribution::StudentT { df, loc, scale } => DistVars::StudentT {
                df: self.evaluate(df)?,
                loc: self.evaluate(loc)?,
                scale: self.evaluate(scale)?,
            },
            Distribution::Exponential { rate } => DistVars::Exponential {
                rate: self.evaluate(rate)?,
            },
            Distribution::Uniform { low, high } => DistVars::Uniform {
                low: self.evaluate(low)?,
                high: self.evaluate(high)?,
            },
            Distribution::Beta { alpha, beta } => DistVars::Beta {
                alpha: self.evaluate(alpha)?,
                beta: self.evaluate(beta)?,
            },
            Distribution::Bernoulli { probs } => DistVars::Bernoulli {
                probs: self.evaluate(probs)?,
            },
            Distribution::Poisson { rate } => DistVars::Poisson {
                rate: self.evaluate(rate)?,
            },
            Distribution::Binomial { total_count, probs } => DistVars::Binomial {
                total_count: self.evaluate(total_count)?,
                probs: self.evaluate(probs)?,
            },
            Distribution::BetaBinomial {
                total_count,
                alpha,
                beta,
            } => DistVars::BetaBinomial {
                total_count: self.evaluate(total_count)?,
                alpha: self.evaluate(alpha)?,
                beta: self.evaluate(beta)?,
            },
            Distribution::NegativeBinomial {
                mean,
                overdispersion,
            } => DistVars::NegativeBinomial {
                mean: self.evaluate(mean)?,
                overdispersion: self.evaluate(overdispersion)?,
            },
            Distribution::MultivariateNormal { mean, scale_tril } => DistVars::MultivariateNormal {
                mean: self.evaluate(mean)?,
                scale_tril: self.evaluate(scale_tril)?,
            },
            Distribution::OrderedLogistic { eta, cutpoints } => DistVars::OrderedLogistic {
                eta: self.evaluate(eta)?,
                cutpoints: self.evaluate(cutpoints)?,
            },
        })
    }
}
