//! Distribution log densities as tape graphs.
//!
//! Each builder mirrors the corresponding `log_prob` in
//! `src/jaxstanv5/distributions/` operation for operation (same formula
//! structure, same support masking, same clipping constants), so that f64
//! results agree with the JAX reference to rounding error.

use crate::error::{Error, ErrorKind};
use crate::tape::{Tape, Var};
use crate::tensor::{slice_last_map, GatherMap, Tensor};

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
}

fn mismatch(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::DataShapeMismatch, message)
}

fn scalar(tape: &mut Tape, v: f64) -> Var {
    tape.constant(Tensor::scalar(v))
}

/// Boolean (0.0/1.0) comparison tensor from forward values, broadcasting.
fn cmp(a: &Tensor, b: &Tensor, f: impl Fn(f64, f64) -> bool) -> Result<Tensor, Error> {
    a.binary(b, |x, y| if f(x, y) { 1.0 } else { 0.0 })
}

fn and(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    a.binary(b, |x, y| if x != 0.0 && y != 0.0 { 1.0 } else { 0.0 })
}

fn is_integer_mask(v: &Tensor) -> Tensor {
    v.map(|x| if x == x.floor() { 1.0 } else { 0.0 })
}

/// `jnp.clip(x, lo, hi)` as two selects; gradient flows where unclipped.
fn clip(tape: &mut Tape, x: Var, lo: f64, hi: f64) -> Result<Var, Error> {
    let lo_var = scalar(tape, lo);
    let above = cmp(tape.value(x), &Tensor::scalar(lo), |a, b| a >= b)?;
    let clipped_lo = tape.where_select(above, x, lo_var);
    let hi_var = scalar(tape, hi);
    let below = cmp(tape.value(clipped_lo), &Tensor::scalar(hi), |a, b| a <= b)?;
    Ok(tape.where_select(below, clipped_lo, hi_var))
}

/// Where with a -inf fallback, the standard support mask.
fn mask_support(tape: &mut Tape, support: Tensor, log_density: Var) -> Var {
    let neg_inf = tape.constant(Tensor::scalar(f64::NEG_INFINITY));
    tape.where_select(support, log_density, neg_inf)
}

/// Elementwise (or event-wise) log probability of `value` under `dist`.
pub fn log_prob(tape: &mut Tape, dist: &DistVars, value: Var) -> Result<Var, Error> {
    match dist {
        DistVars::Normal { loc, scale } => {
            let delta = tape.sub(value, *loc);
            let standardized = tape.div(delta, *scale);
            let sq = tape.mul(standardized, standardized);
            let half_neg = scalar(tape, -0.5);
            let term = tape.mul(half_neg, sq);
            let log_scale = tape.ln(*scale);
            let term = tape.sub(term, log_scale);
            let half_log_2pi = scalar(tape, 0.5 * (2.0 * std::f64::consts::PI).ln());
            Ok(tape.sub(term, half_log_2pi))
        }
        DistVars::HalfNormal { scale } => {
            let standardized = tape.div(value, *scale);
            let lead = scalar(tape, 0.5 * (2.0 / std::f64::consts::PI).ln());
            let log_scale = tape.ln(*scale);
            let term = tape.sub(lead, log_scale);
            let sq = tape.mul(standardized, standardized);
            let half = scalar(tape, 0.5);
            let half_sq = tape.mul(half, sq);
            let log_density = tape.sub(term, half_sq);
            let support = cmp(tape.value(value), &Tensor::scalar(0.0), |v, z| v >= z)?;
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
            let support = cmp(tape.value(value), &Tensor::scalar(0.0), |v, z| v >= z)?;
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::Uniform { low, high } => {
            let width = tape.sub(*high, *low);
            let log_width = tape.ln(width);
            let log_density = tape.neg(log_width);
            let ge_low = cmp(tape.value(value), tape.value(*low), |v, l| v >= l)?;
            let le_high = cmp(tape.value(value), tape.value(*high), |v, h| v <= h)?;
            let support = and(&ge_low, &le_high)?;
            Ok(mask_support(tape, support, log_density))
        }
        DistVars::Beta { alpha, beta } => {
            let zero = Tensor::scalar(0.0);
            let one_t = Tensor::scalar(1.0);
            let v_pos = cmp(tape.value(value), &zero, |v, z| v > z)?;
            let v_lt_one = cmp(tape.value(value), &one_t, |v, o| v < o)?;
            let a_pos = cmp(tape.value(*alpha), &zero, |a, z| a > z)?;
            let b_pos = cmp(tape.value(*beta), &zero, |b, z| b > z)?;
            let support = and(&and(&v_pos, &v_lt_one)?, &and(&a_pos, &b_pos)?)?;

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
            let zero = Tensor::scalar(0.0);
            let one_t = Tensor::scalar(1.0);
            let integer = is_integer_mask(tape.value(value));
            let v_ge0 = cmp(tape.value(value), &zero, |v, z| v >= z)?;
            let v_le1 = cmp(tape.value(value), &one_t, |v, o| v <= o)?;
            let p_ge0 = cmp(tape.value(*probs), &zero, |p, z| p >= z)?;
            let p_le1 = cmp(tape.value(*probs), &one_t, |p, o| p <= o)?;
            let support = and(
                &and(&integer, &v_ge0)?,
                &and(&v_le1, &and(&p_ge0, &p_le1)?)?,
            )?;

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
            let zero = Tensor::scalar(0.0);
            let integer = is_integer_mask(tape.value(value));
            let v_ge0 = cmp(tape.value(value), &zero, |v, z| v >= z)?;
            let rate_pos = cmp(tape.value(*rate), &zero, |r, z| r > z)?;
            let support = and(&and(&v_ge0, &integer)?, &rate_pos)?;

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
            let zero = Tensor::scalar(0.0);
            let one_t = Tensor::scalar(1.0);
            let int_v = is_integer_mask(tape.value(value));
            let int_n = is_integer_mask(tape.value(*total_count));
            let v_ge0 = cmp(tape.value(value), &zero, |v, z| v >= z)?;
            let n_ge0 = cmp(tape.value(*total_count), &zero, |n, z| n >= z)?;
            let v_le_n = cmp(tape.value(value), tape.value(*total_count), |v, n| v <= n)?;
            let p_ge0 = cmp(tape.value(*probs), &zero, |p, z| p >= z)?;
            let p_le1 = cmp(tape.value(*probs), &one_t, |p, o| p <= o)?;
            let support = and(
                &and(&and(&int_v, &int_n)?, &and(&v_ge0, &n_ge0)?)?,
                &and(&v_le_n, &and(&p_ge0, &p_le1)?)?,
            )?;

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
            let zero = Tensor::scalar(0.0);
            let int_v = is_integer_mask(tape.value(value));
            let int_n = is_integer_mask(tape.value(*total_count));
            let v_ge0 = cmp(tape.value(value), &zero, |v, z| v >= z)?;
            let n_ge0 = cmp(tape.value(*total_count), &zero, |n, z| n >= z)?;
            let v_le_n = cmp(tape.value(value), tape.value(*total_count), |v, n| v <= n)?;
            let a_pos = cmp(tape.value(*alpha), &zero, |a, z| a > z)?;
            let b_pos = cmp(tape.value(*beta), &zero, |b, z| b > z)?;
            let support = and(
                &and(&and(&int_v, &int_n)?, &and(&v_ge0, &n_ge0)?)?,
                &and(&v_le_n, &and(&a_pos, &b_pos)?)?,
            )?;

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
            let zero = Tensor::scalar(0.0);
            let int_v = is_integer_mask(tape.value(value));
            let v_ge0 = cmp(tape.value(value), &zero, |v, z| v >= z)?;
            let m_pos = cmp(tape.value(*mean), &zero, |m, z| m > z)?;
            let od_pos = cmp(tape.value(*overdispersion), &zero, |o, z| o > z)?;
            let support = and(&and(&int_v, &v_ge0)?, &and(&m_pos, &od_pos)?)?;

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
            let delta = tape.sub(value, *mean);
            let delta = if tape.value(delta).rank() == 0 {
                tape.reshape(delta, vec![1])
            } else {
                delta
            };
            let solved = tape.solve_lower(*scale_tril, delta);
            let solved_sq = tape.mul(solved, solved);
            let quadratic = tape.sum(solved_sq);
            // diag(L) gather: indices i*(n+1)
            let diag_map = GatherMap {
                out_shape: vec![event_size],
                map: (0..event_size).map(|i| i * (event_size + 1)).collect(),
            };
            let diag = tape.gather(*scale_tril, diag_map);
            let log_diag = tape.ln(diag);
            let log_det = tape.sum(log_diag);
            let neg_half = scalar(tape, -0.5);
            let term = tape.mul(neg_half, quadratic);
            let term = tape.sub(term, log_det);
            let const_term = scalar(
                tape,
                0.5 * (event_size as f64) * (2.0 * std::f64::consts::PI).ln(),
            );
            Ok(tape.sub(term, const_term))
        }
        DistVars::OrderedLogistic { eta, cutpoints } => {
            ordered_logistic_log_prob(tape, *eta, *cutpoints, value)
        }
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
    let value_t = tape.value(value).clone();
    let out_batch = Tensor::broadcast_shapes(probs_batch, value_t.shape())?;
    let mut probs_full = out_batch.clone();
    probs_full.push(category_count);
    let probs_b = tape.broadcast(probabilities, &probs_full);
    let value_b = value_t.broadcast_to(&out_batch)?;

    // take_along_axis with the clipped integer category per observation.
    let rows: usize = out_batch.iter().product();
    let mut map = Vec::with_capacity(rows);
    for (row, &raw) in value_b.data().iter().enumerate() {
        let clipped = raw.clamp(0.0, (category_count - 1) as f64);
        map.push(row * category_count + clipped as usize);
    }
    let selected = tape.gather(
        probs_b,
        GatherMap {
            out_shape: out_batch.clone(),
            map,
        },
    );

    // Support: integer label in range, ordered cutpoints, positive mass.
    let integer = is_integer_mask(&value_b);
    let zero = Tensor::scalar(0.0);
    let ge0 = cmp(&value_b, &zero, |v, z| v >= z)?;
    let max_label = Tensor::scalar((category_count - 1) as f64);
    let le_max = cmp(&value_b, &max_label, |v, m| v <= m)?;
    let cut_values = tape.value(cutpoints).data();
    let ordered = cut_values.windows(2).all(|w| w[1] > w[0]);
    let ordered_t = Tensor::scalar(if ordered { 1.0 } else { 0.0 });
    let positive = cmp(tape.value(selected), &zero, |p, z| p > z)?;
    let support = and(
        &and(&integer, &ge0)?,
        &and(&le_max, &and(&ordered_t, &positive)?)?,
    )?;

    let safe = clip(tape, selected, f64::MIN_POSITIVE, 1.0)?;
    let log_safe = tape.ln(safe);
    Ok(mask_support(tape, support, log_safe))
}
