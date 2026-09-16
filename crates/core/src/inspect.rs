//! Pure effective-model inspection.
//!
//! Inspection binds the same [`Posterior`](crate::model::Posterior) used by
//! sampling, then reports its resolved unconstrained layout and actual density
//! factors. It never executes producer code or samples.

use crate::error::Error;
use crate::ir::{
    BinOpKind, Constraint, DataSchema, Dim, Distribution, Expr, IndexSpec, ModelMeta,
    ResolvedFreeValue, Size, UnaryFn,
};
use crate::json::{self, Value};
use crate::model::{DataValue, Posterior, ResolvedConstraint};

pub const INSPECTION_FORMAT: &str = "v0-provisional";

fn string(value: impl Into<String>) -> Value {
    Value::Str(value.into())
}

fn shape_value(shape: &[usize]) -> Value {
    Value::Array(shape.iter().map(|&dim| Value::Int(dim as i64)).collect())
}

fn expr_value(expr: &Expr) -> Value {
    match expr {
        Expr::Param(name) => Value::Object(vec![
            ("node".into(), string("ParamRef")),
            ("name".into(), string(name)),
        ]),
        Expr::Data(name) => Value::Object(vec![
            ("node".into(), string("DataRef")),
            ("name".into(), string(name)),
        ]),
        Expr::Const(value) => Value::Object(vec![
            ("node".into(), string("ConstNode")),
            ("value".into(), Value::Float(*value)),
        ]),
        Expr::Bin { op, left, right } => Value::Object(vec![
            ("node".into(), string("BinOp")),
            (
                "op".into(),
                string(match op {
                    BinOpKind::Add => "+",
                    BinOpKind::Sub => "-",
                    BinOpKind::Mul => "*",
                    BinOpKind::Div => "/",
                }),
            ),
            ("left".into(), expr_value(left)),
            ("right".into(), expr_value(right)),
        ]),
        Expr::Unary { function, operand } => Value::Object(vec![
            ("node".into(), string("UnaryOp")),
            (
                "function".into(),
                string(match function {
                    UnaryFn::Exp => "exp",
                    UnaryFn::Neg => "neg",
                    UnaryFn::Sigmoid => "sigmoid",
                }),
            ),
            ("operand".into(), expr_value(operand)),
        ]),
        Expr::MatVec { matrix, vector } => Value::Object(vec![
            ("node".into(), string("MatVecOp")),
            ("matrix".into(), expr_value(matrix)),
            ("vector".into(), expr_value(vector)),
        ]),
        Expr::Index { base, index } => Value::Object(vec![
            ("node".into(), string("IndexOp")),
            ("base".into(), expr_value(base)),
            ("index".into(), index_value(index)),
        ]),
        Expr::VectorScatter {
            length,
            observed_idx,
            observed_values,
            missing_idx,
            missing_values,
        } => Value::Object(vec![
            ("node".into(), string("VectorScatterOp")),
            ("length".into(), expr_value(length)),
            ("observed_idx".into(), expr_value(observed_idx)),
            ("observed_values".into(), expr_value(observed_values)),
            ("missing_idx".into(), expr_value(missing_idx)),
            ("missing_values".into(), expr_value(missing_values)),
        ]),
    }
}

fn index_value(index: &IndexSpec) -> Value {
    match index {
        IndexSpec::Scalar(expr) => Value::Object(vec![
            ("node".into(), string("ScalarIndex")),
            ("expr".into(), expr_value(expr)),
        ]),
        IndexSpec::Full => Value::Object(vec![("node".into(), string("FullSlice"))]),
        IndexSpec::Tuple(items) => Value::Object(vec![
            ("node".into(), string("IndexTuple")),
            (
                "items".into(),
                Value::Array(items.iter().map(index_value).collect()),
            ),
        ]),
    }
}

/// Re-emit a decoded expression in the documented Bayeswire node encoding.
pub fn expression_wire_value(expr: &Expr) -> Value {
    expr_value(expr)
}

/// Re-emit a decoded distribution in the documented Bayeswire node encoding.
pub fn distribution_wire_value(distribution: &Distribution) -> Value {
    let node = |name: &str, mut fields: Vec<(String, Value)>| {
        let mut entries = vec![("node".into(), string(name))];
        entries.append(&mut fields);
        Value::Object(entries)
    };
    match distribution {
        Distribution::Normal { loc, scale } => node(
            "Normal",
            vec![
                ("loc".into(), expr_value(loc)),
                ("scale".into(), expr_value(scale)),
            ],
        ),
        Distribution::HalfNormal { scale } => {
            node("HalfNormal", vec![("scale".into(), expr_value(scale))])
        }
        Distribution::StudentT { df, loc, scale } => node(
            "StudentT",
            vec![
                ("df".into(), expr_value(df)),
                ("loc".into(), expr_value(loc)),
                ("scale".into(), expr_value(scale)),
            ],
        ),
        Distribution::Exponential { rate } => {
            node("Exponential", vec![("rate".into(), expr_value(rate))])
        }
        Distribution::Uniform { low, high } => node(
            "Uniform",
            vec![
                ("low".into(), expr_value(low)),
                ("high".into(), expr_value(high)),
            ],
        ),
        Distribution::Beta { alpha, beta } => node(
            "Beta",
            vec![
                ("alpha".into(), expr_value(alpha)),
                ("beta".into(), expr_value(beta)),
            ],
        ),
        Distribution::Bernoulli { probs } => {
            node("Bernoulli", vec![("probs".into(), expr_value(probs))])
        }
        Distribution::Poisson { rate } => node("Poisson", vec![("rate".into(), expr_value(rate))]),
        Distribution::Binomial { total_count, probs } => node(
            "Binomial",
            vec![
                ("total_count".into(), expr_value(total_count)),
                ("probs".into(), expr_value(probs)),
            ],
        ),
        Distribution::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => node(
            "BetaBinomial",
            vec![
                ("total_count".into(), expr_value(total_count)),
                ("alpha".into(), expr_value(alpha)),
                ("beta".into(), expr_value(beta)),
            ],
        ),
        Distribution::NegativeBinomial {
            mean,
            overdispersion,
        } => node(
            "NegativeBinomial",
            vec![
                ("mean".into(), expr_value(mean)),
                ("overdispersion".into(), expr_value(overdispersion)),
            ],
        ),
        Distribution::MultivariateNormal { mean, scale_tril } => node(
            "MultivariateNormal",
            vec![
                ("mean".into(), expr_value(mean)),
                ("scale_tril".into(), expr_value(scale_tril)),
            ],
        ),
        Distribution::OrderedLogistic { eta, cutpoints } => node(
            "OrderedLogistic",
            vec![
                ("eta".into(), expr_value(eta)),
                ("cutpoints".into(), expr_value(cutpoints)),
            ],
        ),
        Distribution::Truncated { base, lower, upper } => node(
            "Truncated",
            vec![
                ("base".into(), distribution_wire_value(base)),
                (
                    "lower".into(),
                    lower.as_ref().map(expr_value).unwrap_or(Value::Null),
                ),
                (
                    "upper".into(),
                    upper.as_ref().map(expr_value).unwrap_or(Value::Null),
                ),
            ],
        ),
    }
}

fn declared_constraint_value(constraint: Option<&Constraint>) -> Value {
    match constraint {
        None => Value::Null,
        Some(Constraint::Positive) => Value::Object(vec![("node".into(), string("Positive"))]),
        Some(Constraint::UnitInterval) => {
            Value::Object(vec![("node".into(), string("UnitInterval"))])
        }
        Some(Constraint::Ordered) => Value::Object(vec![("node".into(), string("Ordered"))]),
        Some(Constraint::Interval { lower, upper }) => Value::Object(vec![
            ("node".into(), string("Interval")),
            ("lower".into(), Value::Float(*lower)),
            ("upper".into(), Value::Float(*upper)),
        ]),
        Some(Constraint::VectorBounds { lower, upper }) => Value::Object(vec![
            ("node".into(), string("VectorBounds")),
            (
                "lower".into(),
                lower
                    .as_ref()
                    .map(|name| {
                        Value::Object(vec![
                            ("node".into(), string("DataRef")),
                            ("name".into(), string(name)),
                        ])
                    })
                    .unwrap_or(Value::Null),
            ),
            (
                "upper".into(),
                upper
                    .as_ref()
                    .map(|name| {
                        Value::Object(vec![
                            ("node".into(), string("DataRef")),
                            ("name".into(), string(name)),
                        ])
                    })
                    .unwrap_or(Value::Null),
            ),
        ]),
    }
}

fn resolved_constraint_value(constraint: Option<&ResolvedConstraint>) -> Value {
    match constraint {
        None => Value::Object(vec![("kind".into(), string("unconstrained"))]),
        Some(ResolvedConstraint::Positive) => {
            Value::Object(vec![("kind".into(), string("positive"))])
        }
        Some(ResolvedConstraint::UnitInterval) => Value::Object(vec![
            ("kind".into(), string("unit_interval")),
            ("lower".into(), Value::Float(0.0)),
            ("upper".into(), Value::Float(1.0)),
        ]),
        Some(ResolvedConstraint::Interval { lower, upper }) => Value::Object(vec![
            ("kind".into(), string("interval")),
            ("lower".into(), Value::Float(*lower)),
            ("upper".into(), Value::Float(*upper)),
        ]),
        Some(ResolvedConstraint::Ordered) => {
            Value::Object(vec![("kind".into(), string("ordered"))])
        }
        Some(ResolvedConstraint::VectorBounds { lower, upper }) => Value::Object(vec![
            ("kind".into(), string("vector_bounds")),
            (
                "lower".into(),
                lower
                    .as_ref()
                    .map(|values| Value::Array(values.iter().map(|&v| Value::Float(v)).collect()))
                    .unwrap_or(Value::Null),
            ),
            (
                "upper".into(),
                upper
                    .as_ref()
                    .map(|values| Value::Array(values.iter().map(|&v| Value::Float(v)).collect()))
                    .unwrap_or(Value::Null),
            ),
        ]),
    }
}

fn size_value(size: &Size) -> Value {
    match size {
        Size::Scalar => Value::Null,
        Size::Fixed(value) => Value::Int(*value),
        Size::Data(name) => Value::Object(vec![
            ("node".into(), string("DataRef")),
            ("name".into(), string(name)),
        ]),
    }
}

fn free_declaration(name: &str, free: &ResolvedFreeValue) -> Value {
    Value::Object(vec![
        ("name".into(), string(name)),
        (
            "constraint".into(),
            declared_constraint_value(free.constraint.as_ref()),
        ),
        ("size".into(), size_value(&free.size)),
    ])
}

fn schema_value(schema: &DataSchema) -> Value {
    match schema {
        DataSchema::Rank(rank) => Value::Object(vec![
            ("node".into(), string("ResolvedDataRankSchema")),
            ("rank".into(), Value::Int(*rank)),
        ]),
        DataSchema::Shape(dims) => Value::Object(vec![
            ("node".into(), string("ResolvedDataShapeSchema")),
            (
                "dims".into(),
                Value::Array(
                    dims.iter()
                        .map(|dim| match dim {
                            Dim::Fixed(value) => Value::Int(*value),
                            Dim::DataDim(name) => Value::Object(vec![
                                ("node".into(), string("DataDimRef")),
                                ("name".into(), string(name)),
                            ]),
                        })
                        .collect(),
                ),
            ),
        ]),
    }
}

fn data_report(meta: &ModelMeta, data: &[(String, DataValue)]) -> Value {
    let lookup = |name: &str| {
        data.iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, value)| value)
            .expect("Posterior binding verified all required data")
    };
    let mut entries = Vec::new();
    for (name, declaration) in &meta.data {
        let bound = lookup(name);
        entries.push(Value::Object(vec![
            ("name".into(), string(name)),
            ("role".into(), string("declared_data")),
            ("declared_schema".into(), schema_value(&declaration.schema)),
            ("bound_shape".into(), shape_value(&bound.shape)),
            ("bound_integer".into(), Value::Bool(bound.integer)),
        ]));
    }
    for observed in &meta.observed_nodes {
        let bound = lookup(&observed.name);
        entries.push(Value::Object(vec![
            ("name".into(), string(&observed.name)),
            ("role".into(), string("observed")),
            ("declared_schema".into(), Value::Null),
            ("bound_shape".into(), shape_value(&bound.shape)),
            ("bound_integer".into(), Value::Bool(bound.integer)),
        ]));
    }
    Value::Array(entries)
}

fn discrepancy_report(meta: &ModelMeta) -> Value {
    let sites = meta.resolved_stochastic_sites();
    let mut out = Vec::new();
    for (name, parameter) in &meta.params {
        match sites.iter().find(|site| site.name == *name) {
            Some(site) if site.distribution != parameter.distribution => {
                out.push(Value::Object(vec![
                    ("name".into(), string(name)),
                    (
                        "kind".into(),
                        string("declared_distribution_differs_from_execution_factor"),
                    ),
                    (
                        "declared_distribution".into(),
                        distribution_wire_value(&parameter.distribution),
                    ),
                    (
                        "execution_distribution".into(),
                        distribution_wire_value(&site.distribution),
                    ),
                    (
                        "note".into(),
                        string(
                            "structural difference only; no mathematical inequivalence is asserted",
                        ),
                    ),
                ]))
            }
            None => out.push(Value::Object(vec![
                ("name".into(), string(name)),
                (
                    "kind".into(),
                    string("declaration_without_same_name_execution_factor"),
                ),
            ])),
            Some(_) => {}
        }
    }
    for observed in &meta.observed_nodes {
        match sites.iter().find(|site| site.name == observed.name) {
            Some(site) if site.distribution != observed.distribution => {
                out.push(Value::Object(vec![
                    ("name".into(), string(&observed.name)),
                    (
                        "kind".into(),
                        string("declared_distribution_differs_from_execution_factor"),
                    ),
                    (
                        "declared_distribution".into(),
                        distribution_wire_value(&observed.distribution),
                    ),
                    (
                        "execution_distribution".into(),
                        distribution_wire_value(&site.distribution),
                    ),
                    (
                        "note".into(),
                        string(
                            "structural difference only; no mathematical inequivalence is asserted",
                        ),
                    ),
                ]))
            }
            None => out.push(Value::Object(vec![
                ("name".into(), string(&observed.name)),
                (
                    "kind".into(),
                    string("declaration_without_same_name_execution_factor"),
                ),
            ])),
            Some(_) => {}
        }
    }
    for site in &sites {
        let declared = meta.params.iter().any(|(name, _)| *name == site.name)
            || meta
                .observed_nodes
                .iter()
                .any(|observed| observed.name == site.name);
        if !declared {
            out.push(Value::Object(vec![
                ("name".into(), string(&site.name)),
                (
                    "kind".into(),
                    string("execution_factor_without_same_name_declaration"),
                ),
            ]));
        }
    }
    Value::Array(out)
}

/// Bind and describe the effective model used by log-density evaluation.
pub fn inspect_model(meta: ModelMeta, data: Vec<(String, DataValue)>) -> Result<Value, Error> {
    let explicit_free = !meta.free_values.is_empty();
    let explicit_sites = !meta.stochastic_sites.is_empty();
    let posterior = Posterior::new(meta.clone(), data.clone())?;

    let free_slots = posterior
        .free_slot_details()
        .into_iter()
        .map(|slot| {
            Value::Object(vec![
                ("name".into(), string(slot.name)),
                ("shape".into(), shape_value(&slot.shape)),
                ("offset".into(), Value::Int(slot.offset as i64)),
                ("length".into(), Value::Int(slot.length as i64)),
                (
                    "resolved_constraint".into(),
                    resolved_constraint_value(slot.constraint.as_ref()),
                ),
            ])
        })
        .collect();
    let factors = meta
        .resolved_stochastic_sites()
        .iter()
        .enumerate()
        .map(|(index, site)| {
            Value::Object(vec![
                ("index".into(), Value::Int(index as i64)),
                ("name".into(), string(&site.name)),
                (
                    "distribution".into(),
                    distribution_wire_value(&site.distribution),
                ),
                ("value_expression".into(), expr_value(&site.value)),
            ])
        })
        .collect();
    let declared_params = meta
        .params
        .iter()
        .map(|(name, parameter)| {
            Value::Object(vec![
                ("name".into(), string(name)),
                (
                    "distribution".into(),
                    distribution_wire_value(&parameter.distribution),
                ),
                (
                    "constraint".into(),
                    declared_constraint_value(parameter.constraint.as_ref()),
                ),
                ("size".into(), size_value(&parameter.size)),
            ])
        })
        .collect();
    let declared_observed = meta
        .observed_nodes
        .iter()
        .map(|observed| {
            Value::Object(vec![
                ("name".into(), string(&observed.name)),
                (
                    "distribution".into(),
                    distribution_wire_value(&observed.distribution),
                ),
            ])
        })
        .collect();
    let declared_free = meta
        .free_values
        .iter()
        .map(|(name, free)| free_declaration(name, free))
        .collect();

    Ok(Value::Object(vec![
        ("inspection_format".into(), string(INSPECTION_FORMAT)),
        (
            "execution_metadata".into(),
            Value::Object(vec![
                (
                    "free_values".into(),
                    string(if explicit_free {
                        "explicit"
                    } else {
                        "legacy_derived"
                    }),
                ),
                (
                    "stochastic_sites".into(),
                    string(if explicit_sites {
                        "explicit"
                    } else {
                        "legacy_derived"
                    }),
                ),
            ]),
        ),
        (
            "unconstrained_parameter_count".into(),
            Value::Int(posterior.n_params() as i64),
        ),
        ("free_slots".into(), Value::Array(free_slots)),
        ("density_factors".into(), Value::Array(factors)),
        (
            "declarations".into(),
            Value::Object(vec![
                ("parameters".into(), Value::Array(declared_params)),
                ("observed".into(), Value::Array(declared_observed)),
                ("explicit_free_values".into(), Value::Array(declared_free)),
            ]),
        ),
        ("data".into(), data_report(&meta, &data)),
        ("structural_discrepancies".into(), discrepancy_report(&meta)),
        (
            "density_accounting".into(),
            Value::Object(vec![
                (
                    "transform_jacobians".into(),
                    string("included_in_evaluated_log_density"),
                ),
                ("factor_order".into(), string("density_factors_array_order")),
            ]),
        ),
    ]))
}

/// Render an inspection report to compact JSON.
pub fn inspect_json(meta: ModelMeta, data: Vec<(String, DataValue)>) -> Result<String, Error> {
    json::write(&inspect_model(meta, data)?)
}
