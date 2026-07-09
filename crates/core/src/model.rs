//! Bound model evaluation: data binding, constraint transforms, and the
//! log density with its gradient.
//!
//! Mirrors `src/jaxstanv5/compiler/core.py`: the unconstrained vector `q`
//! is split per the packing-order guarantee, constraints contribute their
//! log-Jacobians, and stochastic sites accumulate in document order.

use std::collections::HashMap;
use std::fmt::Write as _;

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

/// The canonical wrapped data-document marker shared across the toolchain.
const DATA_DOCUMENT_FORMAT: &str = "bayescycle.data.json.v1";

/// Parse the data document convention used by the fixture corpus and the
/// CLI: either the canonical wrapped form
/// `{"format": "bayescycle.data.json.v1", "variables": {...}}` or a bare
/// map `{"<name>": {"dtype": "...", "shape": [...], "values": [...]}}`.
/// A bare number or array is accepted as float64 shorthand. The `format`
/// key is reserved at the top level: its presence selects the wrapped form,
/// and any value other than the supported marker fails explicitly.
pub fn data_from_json(document: &Value) -> Result<Vec<(String, DataValue)>, Error> {
    let Value::Object(entries) = document else {
        return Err(mismatch(
            "the data document must be a JSON object keyed by data name",
        ));
    };
    if document.get("format").is_some() {
        return wrapped_data_from_json(entries);
    }
    let mut out = Vec::with_capacity(entries.len());
    for (name, spec) in entries {
        out.push((name.clone(), data_value_from_json(name, spec)?));
    }
    Ok(out)
}

fn wrapped_data_from_json(entries: &[(String, Value)]) -> Result<Vec<(String, DataValue)>, Error> {
    let format = entries
        .iter()
        .find(|(name, _)| name == "format")
        .map(|(_, value)| value)
        .expect("caller checked the format key exists");
    let Value::Str(format) = format else {
        return Err(malformed(
            "data document \"format\" is reserved and must be a format marker string",
        ));
    };
    if format != DATA_DOCUMENT_FORMAT {
        return Err(malformed(format!(
            "data document format {format:?} is unsupported; expected {DATA_DOCUMENT_FORMAT:?}"
        )));
    }
    for (name, _) in entries {
        if name != "format" && name != "variables" {
            return Err(malformed(format!(
                "data document has unexpected field {name:?}; \
                 a {DATA_DOCUMENT_FORMAT:?} document carries exactly \"format\" and \"variables\""
            )));
        }
    }
    let Some(Value::Object(variables)) = entries
        .iter()
        .find(|(name, _)| name == "variables")
        .map(|(_, value)| value)
    else {
        return Err(malformed(format!(
            "data document with format {DATA_DOCUMENT_FORMAT:?} needs a \"variables\" object"
        )));
    };
    let mut out = Vec::with_capacity(variables.len());
    for (name, spec) in variables {
        out.push((name.clone(), data_value_from_json(name, spec)?));
    }
    Ok(out)
}

fn data_scalar_to_json(value: f64, integer: bool, context: &str) -> Result<Value, Error> {
    if integer {
        if !value.is_finite() || value.fract() != 0.0 {
            return Err(mismatch(format!(
                "{context} integer value must be finite and integral, got {value}"
            )));
        }
        if value < i64::MIN as f64 || value > i64::MAX as f64 {
            return Err(mismatch(format!(
                "{context} integer value must fit JSON integer range, got {value}"
            )));
        }
        Ok(Value::Int(value as i64))
    } else {
        if !value.is_finite() {
            return Err(mismatch(format!(
                "{context} float value must be finite, got {value}"
            )));
        }
        Ok(Value::Float(value))
    }
}

/// Render a normal Bayesite data document using the typed dtype/shape/values
/// convention accepted by [`data_from_json`]. This preserves integer support
/// for generated discrete observations while keeping `sample` unaware of data
/// provenance.
pub fn data_to_json(data: &[(String, DataValue)], context: &str) -> Result<Value, Error> {
    let mut entries = Vec::with_capacity(data.len());
    for (name, value) in data {
        let value_context = format!("{context} data value \"{name}\"");
        let values = value
            .values
            .iter()
            .map(|&entry| data_scalar_to_json(entry, value.integer, &value_context))
            .collect::<Result<Vec<_>, _>>()?;
        let shape = value
            .shape
            .iter()
            .map(|&dim| {
                if dim > i64::MAX as usize {
                    Err(mismatch(format!(
                        "{value_context} shape dimension must fit JSON integer range, got {dim}"
                    )))
                } else {
                    Ok(Value::Int(dim as i64))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        entries.push((
            name.clone(),
            Value::Object(vec![
                (
                    "dtype".to_string(),
                    Value::Str(if value.integer { "int64" } else { "float64" }.to_string()),
                ),
                ("shape".to_string(), Value::Array(shape)),
                ("values".to_string(), Value::Array(values)),
            ]),
        ));
    }
    Ok(Value::Object(entries))
}

fn collect_booleans(name: &str, value: &Value, into: &mut Vec<f64>) -> Result<(), Error> {
    match value {
        Value::Bool(flag) => {
            into.push(if *flag { 1.0 } else { 0.0 });
            Ok(())
        }
        _ => Err(mismatch(format!(
            "data value \"{name}\" with dtype \"bool\" must contain JSON booleans only"
        ))),
    }
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
            let boolean = dtype == "bool";
            let integer = boolean || dtype.starts_with("int") || dtype.starts_with("uint");
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
            let collect: fn(&str, &Value, &mut Vec<f64>) -> Result<(), Error> = if boolean {
                collect_booleans
            } else {
                collect_numbers
            };
            let mut values = Vec::new();
            match values_field {
                Value::Array(items) => {
                    for item in items {
                        collect(name, item, &mut values)?;
                    }
                }
                other => collect(name, other, &mut values)?,
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

/// Parameter expressions of a distribution, for structural validation.
fn distribution_exprs(dist: &Distribution) -> Vec<&Expr> {
    match dist {
        Distribution::Normal { loc, scale } => vec![loc, scale],
        Distribution::HalfNormal { scale } => vec![scale],
        Distribution::StudentT { df, loc, scale } => vec![df, loc, scale],
        Distribution::Exponential { rate } => vec![rate],
        Distribution::Uniform { low, high } => vec![low, high],
        Distribution::Beta { alpha, beta } => vec![alpha, beta],
        Distribution::Bernoulli { probs } => vec![probs],
        Distribution::Poisson { rate } => vec![rate],
        Distribution::Binomial { total_count, probs } => vec![total_count, probs],
        Distribution::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => vec![total_count, alpha, beta],
        Distribution::NegativeBinomial {
            mean,
            overdispersion,
        } => vec![mean, overdispersion],
        Distribution::MultivariateNormal { mean, scale_tril } => vec![mean, scale_tril],
        Distribution::OrderedLogistic { eta, cutpoints } => vec![eta, cutpoints],
        Distribution::Truncated { base, lower, upper } => {
            let mut exprs = distribution_exprs(base);
            exprs.extend(lower.iter());
            exprs.extend(upper.iter());
            exprs
        }
    }
}

/// Depth-check one expression with an explicit work stack (no recursion, so
/// this is safe to run on arbitrarily deep programmatic input). Mirrors the
/// decoder bound so decoded documents can never fail here.
fn check_expr_depth(root: &Expr) -> Result<(), Error> {
    enum Frame<'a> {
        Expr(&'a Expr, usize),
        Index(&'a IndexSpec, usize),
    }
    let mut stack = vec![Frame::Expr(root, 0)];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Expr(expr, depth) => {
                if depth >= crate::ir::MAX_EXPR_DEPTH {
                    return Err(crate::ir::expr_too_deep());
                }
                match expr {
                    Expr::Param(_) | Expr::Data(_) | Expr::Const(_) => {}
                    Expr::Bin { left, right, .. } => {
                        stack.push(Frame::Expr(left, depth + 1));
                        stack.push(Frame::Expr(right, depth + 1));
                    }
                    Expr::Unary { operand, .. } => stack.push(Frame::Expr(operand, depth + 1)),
                    Expr::Index { base, index } => {
                        stack.push(Frame::Expr(base, depth + 1));
                        stack.push(Frame::Index(index, depth + 1));
                    }
                    Expr::VectorScatter {
                        length,
                        observed_idx,
                        observed_values,
                        missing_idx,
                        missing_values,
                    } => {
                        for child in [
                            length,
                            observed_idx,
                            observed_values,
                            missing_idx,
                            missing_values,
                        ] {
                            stack.push(Frame::Expr(child, depth + 1));
                        }
                    }
                }
            }
            Frame::Index(spec, depth) => {
                if depth >= crate::ir::MAX_EXPR_DEPTH {
                    return Err(crate::ir::expr_too_deep());
                }
                match spec {
                    IndexSpec::Full => {}
                    IndexSpec::Scalar(expr) => stack.push(Frame::Expr(expr, depth + 1)),
                    IndexSpec::Tuple(items) => {
                        for item in items {
                            stack.push(Frame::Index(item, depth + 1));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Depth-check every expression a `ModelMeta` carries. Runs before anything
/// recurses over the meta (evaluation, `Debug` formatting for the identity
/// hash), so a hostile depth yields a typed error instead of a stack overflow.
fn check_meta_expr_depth(meta: &ModelMeta) -> Result<(), Error> {
    let mut exprs: Vec<&Expr> = Vec::new();
    for (_, param) in &meta.params {
        exprs.extend(distribution_exprs(&param.distribution));
    }
    for observed in &meta.observed_nodes {
        exprs.extend(distribution_exprs(&observed.distribution));
    }
    for (_, expr) in &meta.expressions {
        exprs.push(expr);
    }
    for site in &meta.stochastic_sites {
        exprs.extend(distribution_exprs(&site.distribution));
        exprs.push(&site.value);
    }
    for expr in exprs {
        check_expr_depth(expr)?;
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn posterior_identity_hash(meta: &ModelMeta, data: &[(String, DataValue)]) -> String {
    let mut text = String::new();
    let _ = write!(&mut text, "model={meta:?};data=[");
    let mut data_entries = data.iter().collect::<Vec<_>>();
    data_entries.sort_by(|(left_name, _), (right_name, _)| left_name.cmp(right_name));
    for (name, value) in data_entries {
        let _ = write!(
            &mut text,
            "{name}:shape={:?}:integer={}:values=",
            value.shape, value.integer
        );
        for entry in &value.values {
            let _ = write!(&mut text, "{:016x},", entry.to_bits());
        }
        text.push(';');
    }
    text.push(']');
    format!("fnv1a64:{:016x}", fnv1a64(text.as_bytes()))
}

/// A model bound to concrete data; evaluates `logp` and its gradient at
/// unconstrained points. Pure: no interior mutability, no I/O.
#[derive(Debug)]
pub struct Posterior {
    free: Vec<FreeSlot>,
    sites: Vec<ResolvedStochasticSite>,
    data: HashMap<String, DataValue>,
    n_params: usize,
    identity_hash: String,
}

impl Posterior {
    pub fn new(meta: ModelMeta, data: Vec<(String, DataValue)>) -> Result<Posterior, Error> {
        check_meta_expr_depth(&meta)?;
        let identity_hash = posterior_identity_hash(&meta, &data);
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
            identity_hash,
        })
    }

    pub fn n_params(&self) -> usize {
        self.n_params
    }

    pub fn identity_hash(&self) -> &str {
        &self.identity_hash
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

    /// Build the logp graph once for repeated evaluation. The graph
    /// structure of a bound model is point-independent (index expressions
    /// are parameter-free by IR contract), so the compiled tape is replayed
    /// in place per point instead of being rebuilt — this is what keeps the
    /// per-leapfrog-step cost to a forward/backward sweep.
    pub fn compile(&self) -> Result<CompiledLogp, Error> {
        let q0 = vec![0.0; self.n_params];
        let (tape, root, leaves) = self.build_logp(&q0)?;
        Ok(CompiledLogp {
            tape,
            root,
            leaves,
            slots: self
                .free
                .iter()
                .map(|slot| (slot.offset, slot.size))
                .collect(),
            n_params: self.n_params,
            adjoints: Vec::new(),
        })
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

/// A [`Posterior`]'s logp/gradient evaluator with a prebuilt tape: leaves
/// are updated in place, the forward pass is replayed, and the backward pass
/// reuses its adjoint slot buffer. Produces bit-identical results to
/// [`Posterior::logp_grad`], without rebuilding the graph per point.
pub struct CompiledLogp {
    tape: Tape,
    root: Var,
    leaves: Vec<Var>,
    /// (offset, size) into `q` per leaf, in packing order.
    slots: Vec<(usize, usize)>,
    n_params: usize,
    adjoints: Vec<Option<Tensor>>,
}

impl CompiledLogp {
    pub fn n_params(&self) -> usize {
        self.n_params
    }

    /// Log density and gradient at the unconstrained point `q`.
    pub fn logp_grad(&mut self, q: &[f64]) -> Result<(f64, Vec<f64>), Error> {
        if q.len() != self.n_params {
            return Err(mismatch(format!(
                "unconstrained parameter vector q has wrong length: expected {}, got {}",
                self.n_params,
                q.len()
            )));
        }
        for (leaf, (offset, size)) in self.leaves.iter().zip(&self.slots) {
            self.tape.set_leaf(*leaf, &q[*offset..offset + size]);
        }
        self.tape.replay();
        let logp = self.tape.value(self.root).data()[0];
        let grads = self
            .tape
            .backward_into(self.root, &self.leaves, &mut self.adjoints);
        let mut grad = Vec::with_capacity(self.n_params);
        for tensor in grads {
            grad.extend_from_slice(tensor.data());
        }
        Ok((logp, grad))
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
        Some(Constraint::VectorBounds { .. }) => Err(mismatch(
            "VectorBounds constraint is not yet supported at bind time",
        )),
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
            Distribution::Truncated { base, lower, upper } => DistVars::Truncated {
                base: Box::new(self.evaluate_distribution(base)?),
                lower: match lower {
                    Some(expr) => Some(self.evaluate(expr)?),
                    None => None,
                },
                upper: match upper {
                    Some(expr) => Some(self.evaluate(expr)?),
                    None => None,
                },
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOpKind, Distribution, Expr, ResolvedParam, Size};

    #[test]
    fn programmatic_deep_expressions_are_a_typed_error_not_a_crash() {
        // Built iteratively, checked iteratively: depth 4096 must produce a
        // typed error before anything (evaluation, Debug hashing) recurses.
        let mut loc = Expr::Const(0.0);
        for _ in 0..4096 {
            loc = Expr::Bin {
                op: BinOpKind::Add,
                left: Box::new(loc),
                right: Box::new(Expr::Const(1.0)),
            };
        }
        let meta = ModelMeta {
            params: vec![(
                "x".to_string(),
                ResolvedParam {
                    distribution: Distribution::Normal {
                        loc,
                        scale: Expr::Const(1.0),
                    },
                    constraint: None,
                    size: Size::Scalar,
                },
            )],
            data: vec![],
            observed_nodes: vec![],
            expressions: vec![],
            free_values: vec![],
            stochastic_sites: vec![],
        };
        let err = Posterior::new(meta, vec![]).unwrap_err();
        assert_eq!(err.kind, ErrorKind::MalformedDocument);
        assert!(err.message.contains("nesting"), "message: {}", err.message);
    }
}
