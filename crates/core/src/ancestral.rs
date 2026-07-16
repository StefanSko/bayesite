//! Stable ancestral execution planning for forward simulation.
//!
//! Stochastic-site order is the factor and artifact contract. Forward draws use
//! this separate plan so dependency order never mutates decoded metadata.

use std::collections::HashSet;

use crate::error::{Error, ErrorKind};
use crate::ir::{Distribution, Expr, IndexSpec, ResolvedStochasticSite};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ValueRef {
    Param(String),
    Data(String),
}

impl ValueRef {
    pub(crate) fn param(name: impl Into<String>) -> Self {
        Self::Param(name.into())
    }

    pub(crate) fn data(name: impl Into<String>) -> Self {
        Self::Data(name.into())
    }

    fn label(&self) -> String {
        match self {
            Self::Param(name) => format!("ParamRef({name:?})"),
            Self::Data(name) => format!("DataRef({name:?})"),
        }
    }
}

fn collect_index_refs(index: &IndexSpec, refs: &mut Vec<ValueRef>) {
    match index {
        IndexSpec::Scalar(expr) => collect_expr_refs(expr, refs),
        IndexSpec::Full => {}
        IndexSpec::Tuple(items) => {
            for item in items {
                collect_index_refs(item, refs);
            }
        }
    }
}

fn collect_expr_refs(expr: &Expr, refs: &mut Vec<ValueRef>) {
    match expr {
        Expr::Param(name) => refs.push(ValueRef::Param(name.clone())),
        Expr::Data(name) => refs.push(ValueRef::Data(name.clone())),
        Expr::Const(_) => {}
        Expr::Bin { left, right, .. } => {
            collect_expr_refs(left, refs);
            collect_expr_refs(right, refs);
        }
        Expr::Unary { operand, .. } => collect_expr_refs(operand, refs),
        Expr::Index { base, index } => {
            collect_expr_refs(base, refs);
            collect_index_refs(index, refs);
        }
        Expr::VectorScatter {
            length,
            observed_idx,
            observed_values,
            missing_idx,
            missing_values,
        } => {
            collect_expr_refs(length, refs);
            collect_expr_refs(observed_idx, refs);
            collect_expr_refs(observed_values, refs);
            collect_expr_refs(missing_idx, refs);
            collect_expr_refs(missing_values, refs);
        }
    }
}

fn collect_distribution_refs(distribution: &Distribution, refs: &mut Vec<ValueRef>) {
    match distribution {
        Distribution::Normal { loc, scale } => {
            collect_expr_refs(loc, refs);
            collect_expr_refs(scale, refs);
        }
        Distribution::HalfNormal { scale } => collect_expr_refs(scale, refs),
        Distribution::StudentT { df, loc, scale } => {
            collect_expr_refs(df, refs);
            collect_expr_refs(loc, refs);
            collect_expr_refs(scale, refs);
        }
        Distribution::Exponential { rate } => collect_expr_refs(rate, refs),
        Distribution::Uniform { low, high } => {
            collect_expr_refs(low, refs);
            collect_expr_refs(high, refs);
        }
        Distribution::Beta { alpha, beta } => {
            collect_expr_refs(alpha, refs);
            collect_expr_refs(beta, refs);
        }
        Distribution::Bernoulli { probs } => collect_expr_refs(probs, refs),
        Distribution::Poisson { rate } => collect_expr_refs(rate, refs),
        Distribution::Binomial { total_count, probs } => {
            collect_expr_refs(total_count, refs);
            collect_expr_refs(probs, refs);
        }
        Distribution::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => {
            collect_expr_refs(total_count, refs);
            collect_expr_refs(alpha, refs);
            collect_expr_refs(beta, refs);
        }
        Distribution::NegativeBinomial {
            mean,
            overdispersion,
        } => {
            collect_expr_refs(mean, refs);
            collect_expr_refs(overdispersion, refs);
        }
        Distribution::MultivariateNormal { mean, scale_tril } => {
            collect_expr_refs(mean, refs);
            collect_expr_refs(scale_tril, refs);
        }
        Distribution::OrderedLogistic { eta, cutpoints } => {
            collect_expr_refs(eta, refs);
            collect_expr_refs(cutpoints, refs);
        }
        Distribution::Truncated { base, lower, upper } => {
            collect_distribution_refs(base, refs);
            if let Some(lower) = lower {
                collect_expr_refs(lower, refs);
            }
            if let Some(upper) = upper {
                collect_expr_refs(upper, refs);
            }
        }
    }
}

fn collect_generation_value_refs(value: &Expr, refs: &mut Vec<ValueRef>) {
    if let Expr::VectorScatter {
        length,
        missing_idx,
        ..
    } = value
    {
        // Prior-predictive generation evaluates these fields to determine the
        // draw shape and extract the generated missing coordinates. The
        // direct missing ParamRef is the site's output, not a dependency;
        // observed indexes/values are intentionally not conditioned into the
        // complete prior draw.
        collect_expr_refs(length, refs);
        collect_expr_refs(missing_idx, refs);
    }
}

fn generated_value(site: &ResolvedStochasticSite) -> Option<ValueRef> {
    match &site.value {
        Expr::Param(name) => Some(ValueRef::Param(name.clone())),
        Expr::Data(name) => Some(ValueRef::Data(name.clone())),
        Expr::VectorScatter { missing_values, .. } => match missing_values.as_ref() {
            Expr::Param(name) => Some(ValueRef::Param(name.clone())),
            _ => None,
        },
        Expr::Const(_) | Expr::Bin { .. } | Expr::Unary { .. } | Expr::Index { .. } => None,
    }
}

/// Return selected site indices in stable ancestral execution order.
///
/// `site_indices` must be in metadata order. The first ready site is selected
/// on each pass, preserving existing draw/RNG order whenever metadata is
/// already ancestral.
pub(crate) fn stable_site_order(
    sites: &[ResolvedStochasticSite],
    site_indices: &[usize],
    initially_available: impl IntoIterator<Item = ValueRef>,
    context: &str,
) -> Result<Vec<usize>, Error> {
    let mut available = initially_available.into_iter().collect::<HashSet<_>>();
    let mut pending = site_indices
        .iter()
        .map(|&site_index| {
            let site = &sites[site_index];
            let output = generated_value(site).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidSettings,
                    format!(
                        "{context} stochastic site {:?} is not assignable and cannot be scheduled for generation",
                        site.name
                    ),
                )
            })?;
            let mut dependencies = Vec::new();
            collect_distribution_refs(&site.distribution, &mut dependencies);
            collect_generation_value_refs(&site.value, &mut dependencies);
            dependencies.dedup();
            Ok((site_index, output, dependencies))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut ordered = Vec::with_capacity(pending.len());

    while !pending.is_empty() {
        if let Some(ready) = pending.iter().position(|(_, _, dependencies)| {
            dependencies
                .iter()
                .all(|dependency| available.contains(dependency))
        }) {
            let (site_index, output, _) = pending.remove(ready);
            available.insert(output);
            ordered.push(site_index);
            continue;
        }

        let unresolved = pending
            .iter()
            .map(|(site_index, _, dependencies)| {
                let missing = dependencies
                    .iter()
                    .filter(|dependency| !available.contains(*dependency))
                    .map(ValueRef::label)
                    .collect::<Vec<_>>();
                format!("{:?} -> [{}]", sites[*site_index].name, missing.join(", "))
            })
            .collect::<Vec<_>>();
        return Err(Error::new(
            ErrorKind::InvalidSettings,
            format!(
                "{context} generative dependencies are cyclic or unavailable; unresolved references: [{}]; provide every external value as declared data or supplied truth, and keep one generative owner for every generated value",
                unresolved.join(", ")
            ),
        ));
    }

    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOpKind, UnaryFn};

    #[test]
    fn recursive_reference_walk_covers_every_expression_container() {
        let expression = Expr::VectorScatter {
            length: Box::new(Expr::Data("length".to_string())),
            observed_idx: Box::new(Expr::Index {
                base: Box::new(Expr::Data("observed_idx".to_string())),
                index: IndexSpec::Tuple(vec![
                    IndexSpec::Scalar(Box::new(Expr::Data("index".to_string()))),
                    IndexSpec::Full,
                ]),
            }),
            observed_values: Box::new(Expr::Unary {
                function: UnaryFn::Neg,
                operand: Box::new(Expr::Param("observed_value".to_string())),
            }),
            missing_idx: Box::new(Expr::Data("missing_idx".to_string())),
            missing_values: Box::new(Expr::Bin {
                op: BinOpKind::Add,
                left: Box::new(Expr::Param("missing_value".to_string())),
                right: Box::new(Expr::Const(1.0)),
            }),
        };
        let distribution = Distribution::Truncated {
            base: Box::new(Distribution::Normal {
                loc: expression,
                scale: Expr::Data("scale".to_string()),
            }),
            lower: Some(Expr::Param("lower".to_string())),
            upper: Some(Expr::Data("upper".to_string())),
        };
        let mut refs = Vec::new();
        collect_distribution_refs(&distribution, &mut refs);

        for expected in [
            ValueRef::data("length"),
            ValueRef::data("observed_idx"),
            ValueRef::data("index"),
            ValueRef::param("observed_value"),
            ValueRef::data("missing_idx"),
            ValueRef::param("missing_value"),
            ValueRef::data("scale"),
            ValueRef::param("lower"),
            ValueRef::data("upper"),
        ] {
            assert!(
                refs.contains(&expected),
                "missing {expected:?} from {refs:?}"
            );
        }
    }
}
