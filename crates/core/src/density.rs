//! Distribution log densities as tape graphs.
//!
//! Each builder mirrors the corresponding `log_prob` in
//! `src/jaxstanv5/distributions/` operation for operation (same formula
//! structure, same support masking, same clipping constants), so that f64
//! results agree with the JAX reference to rounding error.

use crate::error::{Error, ErrorKind};
use crate::tape::{Tape, Var};
use crate::tensor::{slice_last_map, Tensor};

/// A distribution with its parameter expressions evaluated to tape vars.
pub enum DistVars {
    Normal {
        loc: Var,
        scale: Var,
    },
    HalfNormal {
        scale: Var,
    },
    StudentT {
        df: Var,
        loc: Var,
        scale: Var,
    },
    Exponential {
        rate: Var,
    },
    Uniform {
        low: Var,
        high: Var,
    },
    Beta {
        alpha: Var,
        beta: Var,
    },
    Bernoulli {
        probs: Var,
    },
    Poisson {
        rate: Var,
    },
    Binomial {
        total_count: Var,
        probs: Var,
    },
    BetaBinomial {
        total_count: Var,
        alpha: Var,
        beta: Var,
    },
    NegativeBinomial {
        mean: Var,
        overdispersion: Var,
    },
    MultivariateNormal {
        mean: Var,
        scale_tril: Var,
    },
    OrderedLogistic {
        eta: Var,
        cutpoints: Var,
    },
    Truncated {
        base: Box<DistVars>,
        lower: Option<Var>,
        upper: Option<Var>,
    },
}

fn mismatch(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataShapeMismatch, message)
}

fn scalar(tape: &mut Tape, v: f64) -> Var {
    tape.constant(Tensor::scalar(v))
}

/// `jnp.clip(x, lo, hi)` as two selects; gradient flows where unclipped.
/// Masks are grad-free tape ops, so clipping recomputes on replay.
fn clip(tape: &mut Tape, x: Var, lo: f64, hi: f64) -> Result<Var, Error> {
    let lo_var = scalar(tape, lo);
    let above = tape.ge(x, lo_var)?;
    let clipped_lo = tape.where_select(above, x, lo_var);
    let hi_var = scalar(tape, hi);
    let below = tape.le(clipped_lo, hi_var)?;
    Ok(tape.where_select(below, clipped_lo, hi_var))
}

/// Where with a -inf fallback, the standard support mask.
fn mask_support(tape: &mut Tape, support: Var, log_density: Var) -> Var {
    let neg_inf = tape.constant(Tensor::scalar(f64::NEG_INFINITY));
    tape.where_select(support, log_density, neg_inf)
}

/// Elementwise (or event-wise) log probability of `value` under `dist`.
pub fn log_prob(tape: &mut Tape, dist: &DistVars, value: Var) -> Result<Var, Error> {
    match dist {
        DistVars::Normal { loc, scale } => Ok(tape.normal_log_prob(value, *loc, *scale)),
        DistVars::HalfNormal { scale } => {
            let standardized = tape.div(value, *scale);
            let lead = scalar(tape, 0.5 * (2.0 / std::f64::consts::PI).ln());
            let log_scale = tape.ln(*scale);
            let term = tape.sub(lead, log_scale);
            let sq = tape.mul(standardized, standardized);
            let half = scalar(tape, 0.5);
            let half_sq = tape.mul(half, sq);
            let log_density = tape.sub(term, half_sq);
            let zero = scalar(tape, 0.0);
            let support = tape.ge(value, zero)?;
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::StudentT { df, loc, scale } => {
            let delta = tape.sub(value, *loc);
            let standardized = tape.div(delta, *scale);
            let half = scalar(tape, 0.5);
            let one = scalar(tape, 1.0);
            let df_plus_1 = tape.add(*df, one);
            let half_df_plus_1 = tape.mul(half, df_plus_1);
            let a = tape.gammaln(half_df_plus_1);
            let half_df = tape.mul(half, *df);
            let b = tape.gammaln(half_df);
            let term = tape.sub(a, b);
            let pi = scalar(tape, std::f64::consts::PI);
            let df_pi = tape.mul(*df, pi);
            let log_df_pi = tape.ln(df_pi);
            let half_log_df_pi = tape.mul(half, log_df_pi);
            let term = tape.sub(term, half_log_df_pi);
            let log_scale = tape.ln(*scale);
            let term = tape.sub(term, log_scale);
            let sq = tape.mul(standardized, standardized);
            let sq_over_df = tape.div(sq, *df);
            let log1p_term = tape.ln_1p(sq_over_df);
            let scaled = tape.mul(half_df_plus_1, log1p_term);
            Ok(tape.sub(term, scaled))
        }
        DistVars::Exponential { rate } => {
            let log_rate = tape.ln(*rate);
            let rate_v = tape.mul(*rate, value);
            let log_density = tape.sub(log_rate, rate_v);
            let zero = scalar(tape, 0.0);
            let support = tape.ge(value, zero)?;
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::Uniform { low, high } => {
            let width = tape.sub(*high, *low);
            let log_width = tape.ln(width);
            let log_density = tape.neg(log_width);
            let ge_low = tape.ge(value, *low)?;
            let le_high = tape.le(value, *high)?;
            let support = tape.and(ge_low, le_high)?;
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::Beta { alpha, beta } => {
            let zero = scalar(tape, 0.0);
            let one_t = scalar(tape, 1.0);
            let v_pos = tape.gt(value, zero)?;
            let v_lt_one = tape.lt(value, one_t)?;
            let a_pos = tape.gt(*alpha, zero)?;
            let b_pos = tape.gt(*beta, zero)?;
            let v_ok = tape.and(v_pos, v_lt_one)?;
            let ab_ok = tape.and(a_pos, b_pos)?;
            let support = tape.and(v_ok, ab_ok)?;

            let safe = clip(tape, value, f64::MIN_POSITIVE, 1.0 - f64::EPSILON)?;
            let ga = tape.gammaln(*alpha);
            let gb = tape.gammaln(*beta);
            let a_plus_b = tape.add(*alpha, *beta);
            let gab = tape.gammaln(a_plus_b);
            let sum_g = tape.add(ga, gb);
            let log_normalizer = tape.sub(sum_g, gab);

            let one = scalar(tape, 1.0);
            let a_m1 = tape.sub(*alpha, one);
            let log_safe = tape.ln(safe);
            let log_density = tape.mul(a_m1, log_safe);
            let b_m1 = tape.sub(*beta, one);
            let neg_safe = tape.neg(safe);
            let log1p_neg = tape.ln_1p(neg_safe);
            let term_b = tape.mul(b_m1, log1p_neg);
            let log_density = tape.add(log_density, term_b);
            let log_density = tape.sub(log_density, log_normalizer);
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::Bernoulli { probs } => {
            let zero = scalar(tape, 0.0);
            let one_t = scalar(tape, 1.0);
            let integer = tape.is_integer(value);
            let v_ge0 = tape.ge(value, zero)?;
            let v_le1 = tape.le(value, one_t)?;
            let p_ge0 = tape.ge(*probs, zero)?;
            let p_le1 = tape.le(*probs, one_t)?;
            let v_int = tape.and(integer, v_ge0)?;
            let p_ok = tape.and(p_ge0, p_le1)?;
            let v_ok = tape.and(v_le1, p_ok)?;
            let support = tape.and(v_int, v_ok)?;

            let first = tape.xlogy(value, *probs);
            let one = scalar(tape, 1.0);
            let one_minus_v = tape.sub(one, value);
            let one2 = scalar(tape, 1.0);
            let one_minus_p = tape.sub(one2, *probs);
            let second = tape.xlogy(one_minus_v, one_minus_p);
            let log_mass = tape.add(first, second);
            Ok(mask_support(tape, support, log_mass))
        }
        DistVars::Poisson { rate } => {
            let zero = scalar(tape, 0.0);
            let integer = tape.is_integer(value);
            let v_ge0 = tape.ge(value, zero)?;
            let rate_pos = tape.gt(*rate, zero)?;
            let v_ok = tape.and(v_ge0, integer)?;
            let support = tape.and(v_ok, rate_pos)?;

            let log_rate = tape.ln(*rate);
            let v_log_rate = tape.mul(value, log_rate);
            let term = tape.sub(v_log_rate, *rate);
            let one = scalar(tape, 1.0);
            let v_plus_1 = tape.add(value, one);
            let g = tape.gammaln(v_plus_1);
            let log_mass = tape.sub(term, g);
            Ok(mask_support(tape, support, log_mass))
        }
        DistVars::Binomial { total_count, probs } => {
            let zero = scalar(tape, 0.0);
            let one_t = scalar(tape, 1.0);
            let int_v = tape.is_integer(value);
            let int_n = tape.is_integer(*total_count);
            let v_ge0 = tape.ge(value, zero)?;
            let n_ge0 = tape.ge(*total_count, zero)?;
            let v_le_n = tape.le(value, *total_count)?;
            let p_ge0 = tape.ge(*probs, zero)?;
            let p_le1 = tape.le(*probs, one_t)?;
            let ints = tape.and(int_v, int_n)?;
            let nonneg = tape.and(v_ge0, n_ge0)?;
            let left = tape.and(ints, nonneg)?;
            let p_ok = tape.and(p_ge0, p_le1)?;
            let right = tape.and(v_le_n, p_ok)?;
            let support = tape.and(left, right)?;

            let failures = tape.sub(*total_count, value);
            let one = scalar(tape, 1.0);
            let n_p1 = tape.add(*total_count, one);
            let g_n = tape.gammaln(n_p1);
            let one2 = scalar(tape, 1.0);
            let v_p1 = tape.add(value, one2);
            let g_v = tape.gammaln(v_p1);
            let one3 = scalar(tape, 1.0);
            let f_p1 = tape.add(failures, one3);
            let g_f = tape.gammaln(f_p1);
            let log_mass = tape.sub(g_n, g_v);
            let log_mass = tape.sub(log_mass, g_f);
            let x1 = tape.xlogy(value, *probs);
            let log_mass = tape.add(log_mass, x1);
            let one4 = scalar(tape, 1.0);
            let q = tape.sub(one4, *probs);
            let x2 = tape.xlogy(failures, q);
            let log_mass = tape.add(log_mass, x2);
            Ok(mask_support(tape, support, log_mass))
        }
        DistVars::BetaBinomial {
            total_count,
            alpha,
            beta,
        } => {
            let zero = scalar(tape, 0.0);
            let int_v = tape.is_integer(value);
            let int_n = tape.is_integer(*total_count);
            let v_ge0 = tape.ge(value, zero)?;
            let n_ge0 = tape.ge(*total_count, zero)?;
            let v_le_n = tape.le(value, *total_count)?;
            let a_pos = tape.gt(*alpha, zero)?;
            let b_pos = tape.gt(*beta, zero)?;
            let ints = tape.and(int_v, int_n)?;
            let nonneg = tape.and(v_ge0, n_ge0)?;
            let left = tape.and(ints, nonneg)?;
            let ab_ok = tape.and(a_pos, b_pos)?;
            let right = tape.and(v_le_n, ab_ok)?;
            let support = tape.and(left, right)?;

            let failures = tape.sub(*total_count, value);
            let one = scalar(tape, 1.0);
            let n_p1 = tape.add(*total_count, one);
            let g_n = tape.gammaln(n_p1);
            let one2 = scalar(tape, 1.0);
            let v_p1 = tape.add(value, one2);
            let g_v = tape.gammaln(v_p1);
            let one3 = scalar(tape, 1.0);
            let f_p1 = tape.add(failures, one3);
            let g_f = tape.gammaln(f_p1);
            let log_choose = tape.sub(g_n, g_v);
            let log_choose = tape.sub(log_choose, g_f);

            let v_plus_a = tape.add(value, *alpha);
            let g_va = tape.gammaln(v_plus_a);
            let f_plus_b = tape.add(failures, *beta);
            let g_fb = tape.gammaln(f_plus_b);
            let log_beta_observed = tape.add(g_va, g_fb);
            let n_plus_a = tape.add(*total_count, *alpha);
            let n_plus_ab = tape.add(n_plus_a, *beta);
            let g_nab = tape.gammaln(n_plus_ab);
            let log_beta_observed = tape.sub(log_beta_observed, g_nab);

            let g_a = tape.gammaln(*alpha);
            let g_b = tape.gammaln(*beta);
            let a_plus_b = tape.add(*alpha, *beta);
            let g_ab = tape.gammaln(a_plus_b);
            let log_beta_prior = tape.add(g_a, g_b);
            let log_beta_prior = tape.sub(log_beta_prior, g_ab);

            let log_mass = tape.add(log_choose, log_beta_observed);
            let log_mass = tape.sub(log_mass, log_beta_prior);
            Ok(mask_support(tape, support, log_mass))
        }
        DistVars::NegativeBinomial {
            mean,
            overdispersion,
        } => {
            let zero = scalar(tape, 0.0);
            let int_v = tape.is_integer(value);
            let v_ge0 = tape.ge(value, zero)?;
            let m_pos = tape.gt(*mean, zero)?;
            let od_pos = tape.gt(*overdispersion, zero)?;
            let v_ok = tape.and(int_v, v_ge0)?;
            let params_ok = tape.and(m_pos, od_pos)?;
            let support = tape.and(v_ok, params_ok)?;

            let total = tape.add(*mean, *overdispersion);
            let v_plus_od = tape.add(value, *overdispersion);
            let g_vod = tape.gammaln(v_plus_od);
            let g_od = tape.gammaln(*overdispersion);
            let one = scalar(tape, 1.0);
            let v_p1 = tape.add(value, one);
            let g_v = tape.gammaln(v_p1);
            let log_mass = tape.sub(g_vod, g_od);
            let log_mass = tape.sub(log_mass, g_v);
            let od_frac = tape.div(*overdispersion, total);
            let x1 = tape.xlogy(*overdispersion, od_frac);
            let log_mass = tape.add(log_mass, x1);
            let mean_frac = tape.div(*mean, total);
            let x2 = tape.xlogy(value, mean_frac);
            let log_mass = tape.add(log_mass, x2);
            Ok(mask_support(tape, support, log_mass))
        }
        DistVars::MultivariateNormal { mean, scale_tril } => {
            let tril_shape = tape.value(*scale_tril).shape().to_vec();
            if tril_shape.len() != 2 || tril_shape[0] != tril_shape[1] {
                return Err(mismatch(format!(
                    "MultivariateNormal scale_tril must be a square rank-2 matrix; \
                     got shape {tril_shape:?} (batched scale factors are not supported \
                     by this backend)"
                )));
            }
            let event_size = tril_shape[0];
            let value = match tape.value(value).rank() {
                0 if event_size == 1 => tape.reshape(value, vec![1]),
                0 => {
                    return Err(mismatch(
                        "MultivariateNormal values must have a trailing event dimension",
                    ))
                }
                1 => value,
                _ => {
                    return Err(mismatch(
                        "batched MultivariateNormal values are not supported by this backend",
                    ))
                }
            };
            if tape.value(value).shape()[0] != event_size {
                return Err(mismatch(format!(
                    "MultivariateNormal values must have trailing dimension {event_size}, \
                     got {}",
                    tape.value(value).shape()[0]
                )));
            }
            Ok(tape.multivariate_normal_log_prob(value, *mean, *scale_tril))
        }
        DistVars::OrderedLogistic { eta, cutpoints } => {
            ordered_logistic_log_prob(tape, *eta, *cutpoints, value)
        }
        DistVars::Truncated { base, lower, upper } => {
            // logp(x) = base_logp(x) - ln(base_cdf(upper) - base_cdf(lower)),
            // masked to the truncation interval; the base builder already
            // masks the base support.
            let base_lp = log_prob(tape, base, value)?;
            let log_z = truncation_log_normalizer(tape, base, *lower, *upper)?;
            let lp = tape.sub(base_lp, log_z);
            let support = match (lower, upper) {
                (Some(l), Some(u)) => {
                    let ge = tape.ge(value, *l)?;
                    let le = tape.le(value, *u)?;
                    Some(tape.and(ge, le)?)
                }
                (Some(l), None) => Some(tape.ge(value, *l)?),
                (None, Some(u)) => Some(tape.le(value, *u)?),
                (None, None) => None,
            };
            Ok(match support {
                Some(support) => mask_support(tape, support, lp),
                None => lp,
            })
        }
    }
}

/// ln(base_cdf(upper) - base_cdf(lower)) as a tape graph, with a missing
/// bound contributing CDF 0 / 1. Computed in log space (Normal) or clamped
/// closed form (Uniform, Exponential) so the normalizer keeps relative
/// precision when the retained mass sits far in a tail, and composed from
/// differentiable ops so gradients flow through symbolic base parameters
/// and bounds.
fn truncation_log_normalizer(
    tape: &mut Tape,
    base: &DistVars,
    lower: Option<Var>,
    upper: Option<Var>,
) -> Result<Var, Error> {
    match base {
        DistVars::Normal { loc, scale } => {
            let standardize = |tape: &mut Tape, bound: Var| {
                let delta = tape.sub(bound, *loc);
                tape.div(delta, *scale)
            };
            match (lower, upper) {
                (Some(l), None) => {
                    // Z = 1 - Phi(a) = Phi(-a).
                    let a = standardize(tape, l);
                    let neg_a = tape.neg(a);
                    Ok(tape.log_ndtr(neg_a))
                }
                (None, Some(u)) => {
                    let b = standardize(tape, u);
                    Ok(tape.log_ndtr(b))
                }
                (Some(l), Some(u)) => {
                    // ln(Phi(b) - Phi(a)) = hi + log1p(-exp(lo - hi)). By
                    // symmetry Phi(b) - Phi(a) = Phi(-a) - Phi(-b), and
                    // log_ndtr keeps full relative precision only in the
                    // left tail, so mirror when the interval sits right of
                    // the mean (naively, Phi(9) - Phi(8) cancels to zero).
                    let a = standardize(tape, l);
                    let b = standardize(tape, u);
                    let neg_a = tape.neg(a);
                    let neg_b = tape.neg(b);
                    let direct_hi = tape.log_ndtr(b);
                    let direct_lo = tape.log_ndtr(a);
                    let mirror_hi = tape.log_ndtr(neg_a);
                    let mirror_lo = tape.log_ndtr(neg_b);
                    let midpoint = tape.add(a, b);
                    let zero = scalar(tape, 0.0);
                    let mirror = tape.gt(midpoint, zero)?;
                    let hi = tape.where_select(mirror, mirror_hi, direct_hi);
                    let lo = tape.where_select(mirror, mirror_lo, direct_lo);
                    let gap = tape.sub(lo, hi);
                    let ratio = tape.exp(gap);
                    let neg_ratio = tape.neg(ratio);
                    let tail = tape.ln_1p(neg_ratio);
                    Ok(tape.add(hi, tail))
                }
                (None, None) => Ok(scalar(tape, 0.0)),
            }
        }
        DistVars::Uniform { low, high } => {
            // Z = (min(u, high) - max(l, low)) / (high - low).
            let eff_lower = match lower {
                Some(l) => {
                    let above = tape.ge(l, *low)?;
                    tape.where_select(above, l, *low)
                }
                None => *low,
            };
            let eff_upper = match upper {
                Some(u) => {
                    let below = tape.le(u, *high)?;
                    tape.where_select(below, u, *high)
                }
                None => *high,
            };
            let width = tape.sub(eff_upper, eff_lower);
            let log_width = tape.ln(width);
            let full = tape.sub(*high, *low);
            let log_full = tape.ln(full);
            Ok(tape.sub(log_width, log_full))
        }
        DistVars::Exponential { rate } => {
            // CDF(t) = 1 - exp(-rate t) on t >= 0, so with l' = max(l, 0):
            // ln Z = -rate l' + ln(1 - exp(-rate (u - l'))).
            let zero = scalar(tape, 0.0);
            let eff_lower = match lower {
                Some(l) => {
                    let positive = tape.ge(l, zero)?;
                    tape.where_select(positive, l, zero)
                }
                None => zero,
            };
            let rate_lower = tape.mul(*rate, eff_lower);
            let head = tape.neg(rate_lower);
            match upper {
                None => Ok(head),
                Some(u) => {
                    let span = tape.sub(u, eff_lower);
                    let rate_span = tape.mul(*rate, span);
                    let neg_rate_span = tape.neg(rate_span);
                    let survival = tape.exp(neg_rate_span);
                    let neg_survival = tape.neg(survival);
                    let tail = tape.ln_1p(neg_survival);
                    Ok(tape.add(head, tail))
                }
            }
        }
        _ => Err(mismatch(
            "Truncated base must be a distribution with a scalar CDF and inverse CDF \
             (Normal, Uniform, or Exponential)",
        )),
    }
}

fn ordered_logistic_log_prob(
    tape: &mut Tape,
    eta: Var,
    cutpoints: Var,
    value: Var,
) -> Result<Var, Error> {
    let cut_shape = tape.value(cutpoints).shape().to_vec();
    if cut_shape.is_empty() {
        return Err(mismatch("OrderedLogistic cutpoints must be a vector"));
    }
    let n_cut = cut_shape[cut_shape.len() - 1];
    if n_cut < 1 {
        return Err(mismatch("OrderedLogistic requires at least one cutpoint"));
    }
    if cut_shape.len() > 1 {
        return Err(mismatch(
            "batched OrderedLogistic cutpoints are not supported by this backend",
        ));
    }
    let category_count = n_cut + 1;

    // batch_shape = broadcast(eta.shape, cutpoints.shape[:-1]) = eta.shape here.
    let batch_shape = tape.value(eta).shape().to_vec();
    // cumulative = sigmoid(cutpoints - eta[..., None])
    let mut eta_col_shape = batch_shape.clone();
    eta_col_shape.push(1);
    let eta_col = tape.reshape(eta, eta_col_shape);
    let shifted = tape.sub(cutpoints, eta_col);
    let cumulative = tape.sigmoid(shifted);

    let cum_shape = tape.value(cumulative).shape().to_vec();
    let first = tape.gather(cumulative, slice_last_map(&cum_shape, 0, 1));
    let hi = tape.gather(cumulative, slice_last_map(&cum_shape, 1, n_cut));
    let lo = tape.gather(cumulative, slice_last_map(&cum_shape, 0, n_cut - 1));
    let middle = tape.sub(hi, lo);
    let last_cum = tape.gather(cumulative, slice_last_map(&cum_shape, n_cut - 1, n_cut));
    let one = scalar(tape, 1.0);
    let last = tape.sub(one, last_cum);
    let probabilities = tape.concat_last(vec![first, middle, last]);

    // batch over observations: broadcast probabilities rows against value.
    let probs_shape = tape.value(probabilities).shape().to_vec();
    let probs_batch = &probs_shape[..probs_shape.len() - 1];
    let out_batch = Tensor::broadcast_shapes(probs_batch, tape.value(value).shape())?;
    let mut probs_full = out_batch.clone();
    probs_full.push(category_count);
    let probs_b = tape.broadcast(probabilities, &probs_full);
    let value_b = tape.broadcast(value, &out_batch);

    // take_along_axis with the clipped integer category per observation;
    // the category indices are re-read per evaluation.
    let selected = tape.take_along_last(probs_b, value_b);

    // Support: integer label in range, ordered cutpoints, positive mass.
    let integer = tape.is_integer(value_b);
    let zero = scalar(tape, 0.0);
    let ge0 = tape.ge(value_b, zero)?;
    let max_label = scalar(tape, (category_count - 1) as f64);
    let le_max = tape.le(value_b, max_label)?;
    let ordered_t = tape.is_strictly_increasing(cutpoints);
    let positive = tape.gt(selected, zero)?;
    let label_int = tape.and(integer, ge0)?;
    let mass_ok = tape.and(ordered_t, positive)?;
    let label_ok = tape.and(le_max, mass_ok)?;
    let support = tape.and(label_int, label_ok)?;

    let safe = clip(tape, selected, f64::MIN_POSITIVE, 1.0)?;
    let log_safe = tape.ln(safe);
    Ok(mask_support(tape, support, log_safe))
}
