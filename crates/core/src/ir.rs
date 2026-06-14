//! IR v1 data model and decoder (docs/ir-format-v1.md, docs/ir-v1-tags.md).
//!
//! The decoder implements the core profile exactly: the closed tag set from
//! `ir-v1-tags.md`, strict field checking, ordered-map semantics, and the
//! typed error taxonomy. Anything outside the core profile fails with
//! `UnknownNodeTag` by design.

use crate::error::{Error, ErrorKind};
use crate::json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryFn {
    Exp,
    Neg,
    Sigmoid,
}

/// A final expression node (closed set; see `ir-v1-tags.md`).
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Param(String),
    Data(String),
    Const(f64),
    Bin {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        function: UnaryFn,
        operand: Box<Expr>,
    },
    Index {
        base: Box<Expr>,
        index: IndexSpec,
    },
    VectorScatter {
        length: Box<Expr>,
        observed_idx: Box<Expr>,
        observed_values: Box<Expr>,
        missing_idx: Box<Expr>,
        missing_values: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum IndexSpec {
    Scalar(Box<Expr>),
    Full,
    Tuple(Vec<IndexSpec>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    Positive,
    Interval { lower: f64, upper: f64 },
    UnitInterval,
    Ordered,
}

/// `size: DataRef | int | None` on parameters and free values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Size {
    Scalar,
    Fixed(i64),
    Data(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    Normal {
        loc: Expr,
        scale: Expr,
    },
    HalfNormal {
        scale: Expr,
    },
    StudentT {
        df: Expr,
        loc: Expr,
        scale: Expr,
    },
    Exponential {
        rate: Expr,
    },
    Uniform {
        low: Expr,
        high: Expr,
    },
    Beta {
        alpha: Expr,
        beta: Expr,
    },
    Bernoulli {
        probs: Expr,
    },
    Poisson {
        rate: Expr,
    },
    Binomial {
        total_count: Expr,
        probs: Expr,
    },
    BetaBinomial {
        total_count: Expr,
        alpha: Expr,
        beta: Expr,
    },
    NegativeBinomial {
        mean: Expr,
        overdispersion: Expr,
    },
    MultivariateNormal {
        mean: Expr,
        scale_tril: Expr,
    },
    OrderedLogistic {
        eta: Expr,
        cutpoints: Expr,
    },
}

/// A data-shape dimension: a literal or a reference to scalar integer data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Fixed(i64),
    DataDim(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataSchema {
    Rank(i64),
    Shape(Vec<Dim>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedParam {
    pub distribution: Distribution,
    pub constraint: Option<Constraint>,
    pub size: Size,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedData {
    pub schema: DataSchema,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedObserved {
    pub name: String,
    pub distribution: Distribution,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedFreeValue {
    pub constraint: Option<Constraint>,
    pub size: Size,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedStochasticSite {
    pub name: String,
    pub distribution: Distribution,
    pub value: Expr,
}

/// Decoded `ModelMeta`. Entry vectors keep document order; order is semantic.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMeta {
    pub params: Vec<(String, ResolvedParam)>,
    pub data: Vec<(String, ResolvedData)>,
    pub observed_nodes: Vec<ResolvedObserved>,
    pub expressions: Vec<(String, Expr)>,
    pub free_values: Vec<(String, ResolvedFreeValue)>,
    pub stochastic_sites: Vec<ResolvedStochasticSite>,
}

impl ModelMeta {
    /// Free NUTS values in packing order, deriving legacy metadata when
    /// `free_values` is empty (mirrors `_resolved_free_values` in Python).
    pub fn resolved_free_values(&self) -> Vec<(String, ResolvedFreeValue)> {
        if !self.free_values.is_empty() {
            return self.free_values.clone();
        }
        self.params
            .iter()
            .map(|(name, param)| {
                (
                    name.clone(),
                    ResolvedFreeValue {
                        constraint: param.constraint.clone(),
                        size: param.size.clone(),
                    },
                )
            })
            .collect()
    }

    /// Log-density sites in evaluation order, deriving legacy metadata when
    /// `stochastic_sites` is empty (mirrors `_resolved_stochastic_sites`).
    pub fn resolved_stochastic_sites(&self) -> Vec<ResolvedStochasticSite> {
        if !self.stochastic_sites.is_empty() {
            return self.stochastic_sites.clone();
        }
        let param_sites = self
            .params
            .iter()
            .map(|(name, param)| ResolvedStochasticSite {
                name: name.clone(),
                distribution: param.distribution.clone(),
                value: Expr::Param(name.clone()),
            });
        let observed_sites = self
            .observed_nodes
            .iter()
            .map(|observed| ResolvedStochasticSite {
                name: observed.name.clone(),
                distribution: observed.distribution.clone(),
                value: Expr::Data(observed.name.clone()),
            });
        param_sites.chain(observed_sites).collect()
    }
}

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

/// A node object: its tag plus strict field access that tracks consumption.
struct NodeFields<'a> {
    tag: &'a str,
    entries: &'a [(String, Value)],
    consumed: Vec<bool>,
}

impl<'a> NodeFields<'a> {
    fn open(value: &'a Value) -> Result<NodeFields<'a>, Error> {
        let Value::Object(entries) = value else {
            return Err(malformed(
                "expected a node object {\"node\": <tag>, ...}; got a non-object value",
            ));
        };
        let tag = value
            .get("node")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("node objects need a string \"node\" tag field"))?;
        Ok(NodeFields {
            tag,
            entries,
            consumed: vec![false; entries.len()],
        })
    }

    fn field(&mut self, name: &str) -> Result<&'a Value, Error> {
        for (i, (key, value)) in self.entries.iter().enumerate() {
            if key == name {
                self.consumed[i] = true;
                return Ok(value);
            }
        }
        Err(malformed(format!(
            "node \"{}\" is missing required field \"{}\"; add it",
            self.tag, name
        )))
    }

    /// All declared fields must have been consumed (besides the tag itself).
    fn finish(self) -> Result<(), Error> {
        for (i, (key, _)) in self.entries.iter().enumerate() {
            if key != "node" && !self.consumed[i] {
                return Err(malformed(format!(
                    "node \"{}\" has unexpected field \"{}\"; remove it",
                    self.tag, key
                )));
            }
        }
        Ok(())
    }
}

fn decode_scalar_f64(value: &Value, context: &str) -> Result<f64, Error> {
    value
        .as_f64()
        .ok_or_else(|| malformed(format!("{context} must be a JSON number")))
}

fn node_tag(value: &Value) -> Option<&str> {
    value.get("node").and_then(Value::as_str)
}

fn decode_expr(value: &Value) -> Result<Expr, Error> {
    // Union fields: a bare scalar is a constant; an object is a node.
    match value {
        Value::Int(i) => return Ok(Expr::Const(*i as f64)),
        Value::Float(f) => return Ok(Expr::Const(*f)),
        _ => {}
    }
    let mut node = NodeFields::open(value)?;
    let expr = match node.tag {
        "ParamRef" => {
            let name = node
                .field("name")?
                .as_str()
                .ok_or_else(|| malformed("ParamRef name must be a string"))?;
            Expr::Param(name.to_string())
        }
        "DataRef" => {
            let name = node
                .field("name")?
                .as_str()
                .ok_or_else(|| malformed("DataRef name must be a string"))?;
            Expr::Data(name.to_string())
        }
        "ConstNode" => Expr::Const(decode_scalar_f64(node.field("value")?, "ConstNode value")?),
        "BinOp" => {
            let op = match node.field("op")?.as_str() {
                Some("+") => BinOpKind::Add,
                Some("-") => BinOpKind::Sub,
                Some("*") => BinOpKind::Mul,
                Some("/") => BinOpKind::Div,
                other => {
                    return Err(malformed(format!(
                        "BinOp op must be one of \"+\", \"-\", \"*\", \"/\"; got {other:?}"
                    )))
                }
            };
            let left = decode_expr(node.field("left")?)?;
            let right = decode_expr(node.field("right")?)?;
            Expr::Bin {
                op,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        "UnaryOp" => {
            let function = match node.field("function")?.as_str() {
                Some("exp") => UnaryFn::Exp,
                Some("neg") => UnaryFn::Neg,
                Some("sigmoid") => UnaryFn::Sigmoid,
                other => {
                    return Err(malformed(format!(
                        "UnaryOp function must be \"exp\", \"neg\", or \"sigmoid\"; got {other:?}"
                    )))
                }
            };
            let operand = decode_expr(node.field("operand")?)?;
            Expr::Unary {
                function,
                operand: Box::new(operand),
            }
        }
        "IndexOp" => {
            let base = decode_expr(node.field("base")?)?;
            let index = decode_index_spec(node.field("index")?)?;
            Expr::Index {
                base: Box::new(base),
                index,
            }
        }
        "VectorScatterOp" => Expr::VectorScatter {
            length: Box::new(decode_expr(node.field("length")?)?),
            observed_idx: Box::new(decode_expr(node.field("observed_idx")?)?),
            observed_values: Box::new(decode_expr(node.field("observed_values")?)?),
            missing_idx: Box::new(decode_expr(node.field("missing_idx")?)?),
            missing_values: Box::new(decode_expr(node.field("missing_values")?)?),
        },
        other => return Err(unknown_tag(other, "an expression")),
    };
    node.finish()?;
    Ok(expr)
}

fn unknown_tag(tag: &str, context: &str) -> Error {
    Error::new(
        ErrorKind::UnknownNodeTag,
        format!(
            "node tag \"{tag}\" is not in the IR v1 core profile (expected {context}); \
             this backend consumes core-profile documents only (docs/ir-v1-tags.md)"
        ),
    )
}

fn decode_index_spec(value: &Value) -> Result<IndexSpec, Error> {
    let mut node = NodeFields::open(value)?;
    let spec = match node.tag {
        "ScalarIndex" => IndexSpec::Scalar(Box::new(decode_expr(node.field("expr")?)?)),
        "FullSlice" => IndexSpec::Full,
        "IndexTuple" => {
            let items = node
                .field("items")?
                .as_array()
                .ok_or_else(|| malformed("IndexTuple items must be an array"))?;
            let mut specs = Vec::with_capacity(items.len());
            for item in items {
                let spec = decode_index_spec(item)?;
                if matches!(spec, IndexSpec::Tuple(_)) {
                    return Err(malformed("nested index tuples are not supported"));
                }
                specs.push(spec);
            }
            IndexSpec::Tuple(specs)
        }
        other => return Err(unknown_tag(other, "an index spec")),
    };
    node.finish()?;
    Ok(spec)
}

fn decode_constraint(value: &Value) -> Result<Option<Constraint>, Error> {
    if matches!(value, Value::Null) {
        return Ok(None);
    }
    let mut node = NodeFields::open(value)?;
    let constraint = match node.tag {
        "Positive" => Constraint::Positive,
        "UnitInterval" => Constraint::UnitInterval,
        "Ordered" => Constraint::Ordered,
        "Interval" => {
            let lower = decode_scalar_f64(node.field("lower")?, "Interval lower")?;
            let upper = decode_scalar_f64(node.field("upper")?, "Interval upper")?;
            if !lower.is_finite() || !upper.is_finite() || lower >= upper {
                return Err(malformed(
                    "Interval bounds must be finite with lower < upper",
                ));
            }
            Constraint::Interval { lower, upper }
        }
        other => return Err(unknown_tag(other, "a constraint")),
    };
    node.finish()?;
    Ok(Some(constraint))
}

fn decode_size(value: &Value) -> Result<Size, Error> {
    match value {
        Value::Null => Ok(Size::Scalar),
        Value::Int(i) => Ok(Size::Fixed(*i)),
        other => {
            if node_tag(other) == Some("DataRef") {
                let mut node = NodeFields::open(other)?;
                let name = node
                    .field("name")?
                    .as_str()
                    .ok_or_else(|| malformed("DataRef name must be a string"))?
                    .to_string();
                node.finish()?;
                Ok(Size::Data(name))
            } else {
                Err(malformed(
                    "size must be null, an integer, or a DataRef node",
                ))
            }
        }
    }
}

fn decode_distribution(value: &Value) -> Result<Distribution, Error> {
    let mut node = NodeFields::open(value)?;
    let expr = |fields: &mut NodeFields<'_>, name: &str| -> Result<Expr, Error> {
        decode_expr(fields.field(name)?)
    };
    let dist = match node.tag {
        "Normal" => Distribution::Normal {
            loc: expr(&mut node, "loc")?,
            scale: expr(&mut node, "scale")?,
        },
        "HalfNormal" => Distribution::HalfNormal {
            scale: expr(&mut node, "scale")?,
        },
        "StudentT" => Distribution::StudentT {
            df: expr(&mut node, "df")?,
            loc: expr(&mut node, "loc")?,
            scale: expr(&mut node, "scale")?,
        },
        "Exponential" => Distribution::Exponential {
            rate: expr(&mut node, "rate")?,
        },
        "Uniform" => Distribution::Uniform {
            low: expr(&mut node, "low")?,
            high: expr(&mut node, "high")?,
        },
        "Beta" => Distribution::Beta {
            alpha: expr(&mut node, "alpha")?,
            beta: expr(&mut node, "beta")?,
        },
        "Bernoulli" => Distribution::Bernoulli {
            probs: expr(&mut node, "probs")?,
        },
        "Poisson" => Distribution::Poisson {
            rate: expr(&mut node, "rate")?,
        },
        "Binomial" => Distribution::Binomial {
            total_count: expr(&mut node, "total_count")?,
            probs: expr(&mut node, "probs")?,
        },
        "BetaBinomial" => Distribution::BetaBinomial {
            total_count: expr(&mut node, "total_count")?,
            alpha: expr(&mut node, "alpha")?,
            beta: expr(&mut node, "beta")?,
        },
        "NegativeBinomial" => Distribution::NegativeBinomial {
            mean: expr(&mut node, "mean")?,
            overdispersion: expr(&mut node, "overdispersion")?,
        },
        "MultivariateNormal" => Distribution::MultivariateNormal {
            mean: expr(&mut node, "mean")?,
            scale_tril: expr(&mut node, "scale_tril")?,
        },
        "OrderedLogistic" => Distribution::OrderedLogistic {
            eta: expr(&mut node, "eta")?,
            cutpoints: expr(&mut node, "cutpoints")?,
        },
        other => return Err(unknown_tag(other, "a distribution")),
    };
    node.finish()?;
    Ok(dist)
}

fn decode_data_schema(value: &Value) -> Result<DataSchema, Error> {
    let mut node = NodeFields::open(value)?;
    let schema = match node.tag {
        "ResolvedDataRankSchema" => {
            let rank = node
                .field("rank")?
                .as_i64()
                .ok_or_else(|| malformed("ResolvedDataRankSchema rank must be an integer"))?;
            DataSchema::Rank(rank)
        }
        "ResolvedDataShapeSchema" => {
            let dims = node
                .field("dims")?
                .as_array()
                .ok_or_else(|| malformed("ResolvedDataShapeSchema dims must be an array"))?;
            let mut decoded = Vec::with_capacity(dims.len());
            for dim in dims {
                decoded.push(match dim {
                    Value::Int(i) => Dim::Fixed(*i),
                    other if node_tag(other) == Some("DataDimRef") => {
                        let mut dim_node = NodeFields::open(other)?;
                        let name = dim_node
                            .field("name")?
                            .as_str()
                            .ok_or_else(|| malformed("DataDimRef name must be a string"))?
                            .to_string();
                        dim_node.finish()?;
                        Dim::DataDim(name)
                    }
                    _ => return Err(malformed("shape dims must be integers or DataDimRef nodes")),
                });
            }
            DataSchema::Shape(decoded)
        }
        other => return Err(unknown_tag(other, "a data schema")),
    };
    node.finish()?;
    Ok(schema)
}

/// Decode an ordered-map field: an array of `{"name", "value"}` entries.
fn decode_map<T>(
    value: &Value,
    context: &str,
    decode_entry: impl Fn(&Value) -> Result<T, Error>,
) -> Result<Vec<(String, T)>, Error> {
    let items = value
        .as_array()
        .ok_or_else(|| malformed(format!("{context} must be an entry array")))?;
    let mut out: Vec<(String, T)> = Vec::with_capacity(items.len());
    for item in items {
        let Value::Object(entries) = item else {
            return Err(malformed(format!(
                "{context} entries must be {{\"name\", \"value\"}} objects"
            )));
        };
        let valid_keys = entries.len() == 2
            && entries.iter().any(|(k, _)| k == "name")
            && entries.iter().any(|(k, _)| k == "value");
        if !valid_keys {
            return Err(malformed(format!(
                "{context} entries must have exactly the keys \"name\" and \"value\""
            )));
        }
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed(format!("{context} entry names must be strings")))?;
        if out.iter().any(|(existing, _)| existing == name) {
            return Err(malformed(format!(
                "{context} has duplicate entry name \"{name}\"; entry names must be unique"
            )));
        }
        let decoded = decode_entry(item.get("value").expect("checked above"))?;
        out.push((name.to_string(), decoded));
    }
    Ok(out)
}

fn decode_resolved_param(value: &Value) -> Result<ResolvedParam, Error> {
    let mut node = NodeFields::open(value)?;
    if node.tag != "ResolvedParam" {
        return Err(unknown_tag(node.tag, "a ResolvedParam"));
    }
    let param = ResolvedParam {
        distribution: decode_distribution(node.field("distribution")?)?,
        constraint: decode_constraint(node.field("constraint")?)?,
        size: decode_size(node.field("size")?)?,
    };
    node.finish()?;
    Ok(param)
}

fn decode_resolved_data(value: &Value) -> Result<ResolvedData, Error> {
    let mut node = NodeFields::open(value)?;
    if node.tag != "ResolvedData" {
        return Err(unknown_tag(node.tag, "a ResolvedData"));
    }
    let data = ResolvedData {
        schema: decode_data_schema(node.field("schema")?)?,
    };
    node.finish()?;
    Ok(data)
}

fn decode_observed(value: &Value) -> Result<ResolvedObserved, Error> {
    let mut node = NodeFields::open(value)?;
    if node.tag != "ResolvedObserved" {
        return Err(unknown_tag(node.tag, "a ResolvedObserved"));
    }
    let observed = ResolvedObserved {
        name: node
            .field("name")?
            .as_str()
            .ok_or_else(|| malformed("ResolvedObserved name must be a string"))?
            .to_string(),
        distribution: decode_distribution(node.field("distribution")?)?,
    };
    node.finish()?;
    Ok(observed)
}

fn decode_free_value(value: &Value) -> Result<ResolvedFreeValue, Error> {
    let mut node = NodeFields::open(value)?;
    if node.tag != "ResolvedFreeValue" {
        return Err(unknown_tag(node.tag, "a ResolvedFreeValue"));
    }
    let free = ResolvedFreeValue {
        constraint: decode_constraint(node.field("constraint")?)?,
        size: decode_size(node.field("size")?)?,
    };
    node.finish()?;
    Ok(free)
}

fn decode_stochastic_site(value: &Value) -> Result<ResolvedStochasticSite, Error> {
    let mut node = NodeFields::open(value)?;
    if node.tag != "ResolvedStochasticSite" {
        return Err(unknown_tag(node.tag, "a ResolvedStochasticSite"));
    }
    let site = ResolvedStochasticSite {
        name: node
            .field("name")?
            .as_str()
            .ok_or_else(|| malformed("ResolvedStochasticSite name must be a string"))?
            .to_string(),
        distribution: decode_distribution(node.field("distribution")?)?,
        value: decode_expr(node.field("value")?)?,
    };
    node.finish()?;
    Ok(site)
}

fn decode_tuple<T>(
    value: &Value,
    context: &str,
    decode_item: impl Fn(&Value) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let items = value
        .as_array()
        .ok_or_else(|| malformed(format!("{context} must be an array")))?;
    items.iter().map(decode_item).collect()
}

/// Decode a versioned IR envelope `{"jaxstanv5_ir": 1, "model": {...}}`.
pub fn decode_model(document: &Value) -> Result<ModelMeta, Error> {
    if !matches!(document, Value::Object(_)) {
        return Err(malformed(
            "the IR document must be a JSON object {\"jaxstanv5_ir\": 1, \"model\": {...}}",
        ));
    }
    match document.get("jaxstanv5_ir") {
        Some(Value::Int(1)) => {}
        Some(other) => {
            return Err(Error::new(
                ErrorKind::UnsupportedIRVersion,
                format!("unsupported jaxstanv5_ir version {other:?}; this backend reads version 1"),
            ))
        }
        None => {
            return Err(Error::new(
                ErrorKind::UnsupportedIRVersion,
                "missing \"jaxstanv5_ir\" version field; set it to 1",
            ))
        }
    }
    let model = document
        .get("model")
        .ok_or_else(|| malformed("the IR envelope is missing the \"model\" field"))?;

    let mut node = NodeFields::open(model)?;
    if node.tag != "ModelMeta" {
        return Err(unknown_tag(node.tag, "the ModelMeta root"));
    }
    let meta = ModelMeta {
        params: decode_map(node.field("params")?, "params", decode_resolved_param)?,
        data: decode_map(node.field("data")?, "data", decode_resolved_data)?,
        observed_nodes: decode_tuple(node.field("observed_nodes")?, "observed_nodes", |v| {
            decode_observed(v)
        })?,
        expressions: decode_map(node.field("expressions")?, "expressions", decode_expr)?,
        free_values: decode_map(node.field("free_values")?, "free_values", decode_free_value)?,
        stochastic_sites: decode_tuple(node.field("stochastic_sites")?, "stochastic_sites", |v| {
            decode_stochastic_site(v)
        })?,
    };
    node.finish()?;
    Ok(meta)
}
