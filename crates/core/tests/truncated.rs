//! Truncated distribution: decode, log density, gradients, and sampling.
//!
//! Adversarial coverage for the `Truncated` core-profile tag: analytic
//! values (mpmath references), extreme bounds, one-sided truncation, mass
//! far in a tail, symbolic base parameters/bounds, compiled-tape replay
//! across support flips, and inverse-CDF prior simulation moments.

// Reference constants keep the mpmath digits verbatim; rustc rounds them to
// the nearest f64.
#![allow(clippy::excessive_precision)]

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::decode_model;
use bayesite_core::json;
use bayesite_core::model::Posterior;
use bayesite_core::predictive::{simulate_prior_predictive, PriorPredictiveSettings};
use bayesite_core::special;

const HALF_LOG_2PI: f64 = 0.918938533204672741780329736406;

/// Standard normal log density.
fn norm_logpdf(x: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    -0.5 * z * z - scale.ln() - HALF_LOG_2PI
}

/// A JSON expression for an optional constant truncation bound.
fn bound_json(bound: Option<f64>) -> String {
    match bound {
        None => "null".to_string(),
        Some(v) => format!(r#"{{"node": "ConstNode", "value": {v:?}}}"#),
    }
}

/// A one-free-value document: `x ~ Truncated(base, lower, upper)`, no
/// constraint, so `q[0]` is the site value itself and `logp(q)` is exactly
/// the truncated log density.
fn truncated_site_doc(base: &str, lower: Option<f64>, upper: Option<f64>) -> String {
    let dist = format!(
        r#"{{"node": "Truncated", "base": {base}, "lower": {}, "upper": {}}}"#,
        bound_json(lower),
        bound_json(upper)
    );
    format!(
        r#"{{"bayeswire_ir": 1, "model": {{"node": "ModelMeta",
            "params": [{{"name": "x", "value": {{"node": "ResolvedParam",
                "distribution": {dist}, "constraint": null, "size": null}}}}],
            "data": [], "observed_nodes": [], "expressions": [],
            "free_values": [{{"name": "x", "value": {{"node": "ResolvedFreeValue",
                "constraint": null, "size": null}}}}],
            "stochastic_sites": [{{"node": "ResolvedStochasticSite", "name": "x",
                "distribution": {dist},
                "value": {{"node": "ParamRef", "name": "x"}}}}]}}}}"#
    )
}

fn normal_base(loc: f64, scale: f64) -> String {
    format!(
        r#"{{"node": "Normal", "loc": {{"node": "ConstNode", "value": {loc:?}}},
            "scale": {{"node": "ConstNode", "value": {scale:?}}}}}"#
    )
}

fn exponential_base(rate: f64) -> String {
    format!(r#"{{"node": "Exponential", "rate": {{"node": "ConstNode", "value": {rate:?}}}}}"#)
}

fn uniform_base(low: f64, high: f64) -> String {
    format!(
        r#"{{"node": "Uniform", "low": {{"node": "ConstNode", "value": {low:?}}},
            "high": {{"node": "ConstNode", "value": {high:?}}}}}"#
    )
}

fn posterior_for(doc: &str) -> Posterior {
    let document = json::parse(doc).expect("inline document parses");
    let meta = decode_model(&document).expect("inline document decodes");
    Posterior::new(meta, Vec::new()).expect("posterior binds")
}

fn assert_rel_close(got: f64, want: f64, rtol: f64, context: &str) {
    assert!(
        (got - want).abs() <= rtol * want.abs().max(1e-300),
        "{context}: got {got:.17e}, want {want:.17e} (rel err {:.3e})",
        ((got - want) / want).abs()
    );
}

/// Simpson quadrature of `exp(logp)` over `[lower, upper]`; a truncated
/// density must integrate to one over its truncation interval no matter how
/// far in a tail the mass sits — this kills normalizers that lose the tail.
fn integrates_to_one(posterior: &Posterior, lower: f64, upper: f64, tol: f64, context: &str) {
    let n = 4000usize; // even
    let h = (upper - lower) / n as f64;
    let mut total = 0.0;
    for i in 0..=n {
        let x = lower + h * i as f64;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        let density = posterior.logp(&[x]).expect("logp evaluates").exp();
        assert!(
            density.is_finite(),
            "{context}: density at {x} must be finite inside the bounds"
        );
        total += weight * density;
    }
    total *= h / 3.0;
    assert!(
        (total - 1.0).abs() < tol,
        "{context}: mass integrates to {total}, want 1 (tol {tol})"
    );
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[test]
fn truncated_normal_documents_decode() {
    for (lower, upper) in [
        (Some(0.0), None),
        (None, Some(2.0)),
        (Some(-1.0), Some(3.0)),
    ] {
        let doc = truncated_site_doc(&normal_base(0.0, 1.0), lower, upper);
        let document = json::parse(&doc).expect("document parses");
        decode_model(&document).expect("Truncated Normal decodes");
    }
}

#[test]
fn truncated_base_without_scalar_cdf_is_a_typed_error() {
    let beta_base = r#"{"node": "Beta", "alpha": {"node": "ConstNode", "value": 2.0},
        "beta": {"node": "ConstNode", "value": 2.0}}"#;
    let doc = truncated_site_doc(beta_base, Some(0.1), Some(0.9));
    let document = json::parse(&doc).expect("document parses");
    let err = decode_model(&document).expect_err("Beta base must be rejected");
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("Truncated") && err.message.contains("Normal"),
        "message names the tag and the supported bases: {}",
        err.message
    );
}

#[test]
fn nested_truncated_base_is_a_typed_error() {
    let nested_base = format!(
        r#"{{"node": "Truncated", "base": {}, "lower": {}, "upper": null}}"#,
        normal_base(0.0, 1.0),
        bound_json(Some(0.0))
    );
    let doc = truncated_site_doc(&nested_base, Some(1.0), None);
    let document = json::parse(&doc).expect("document parses");
    let err = decode_model(&document).expect_err("nested Truncated base must be rejected");
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

#[test]
fn truncated_without_any_bound_is_a_typed_error() {
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), None, None);
    let document = json::parse(&doc).expect("document parses");
    let err = decode_model(&document).expect_err("bound-less Truncated must be rejected");
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert!(
        err.message.contains("lower") || err.message.contains("bound"),
        "message names the missing bounds: {}",
        err.message
    );
}

#[test]
fn truncated_with_unexpected_field_is_malformed() {
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(0.0), None)
        .replace(r#""upper": null}"#, r#""upper": null, "bogus": 1}"#);
    let document = json::parse(&doc).expect("document parses");
    let err = decode_model(&document).expect_err("unexpected field must be rejected");
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
}

// ---------------------------------------------------------------------------
// Log density and gradient: Normal base
// ---------------------------------------------------------------------------

#[test]
fn truncated_normal_lower_only_matches_analytic_half_normal() {
    // Truncated(Normal(0, 1), lower = 0) is the half-normal:
    // logp(x) = norm_logpdf(x) + ln 2.
    let posterior = posterior_for(&truncated_site_doc(&normal_base(0.0, 1.0), Some(0.0), None));
    for x in [0.0, 0.3, 1.7, 2.5] {
        let (logp, grad) = posterior.logp_grad(&[x]).unwrap();
        assert_rel_close(
            logp,
            norm_logpdf(x, 0.0, 1.0) + 2.0f64.ln(),
            1e-13,
            "half-normal logp",
        );
        // The normalizer is value-independent: dlogp/dx = -x exactly.
        assert_rel_close(grad[0], -x, 1e-12, "half-normal gradient");
    }
}

#[test]
fn truncated_normal_below_lower_bound_is_neg_infinity_with_finite_gradient() {
    let posterior = posterior_for(&truncated_site_doc(&normal_base(0.0, 1.0), Some(0.0), None));
    let (logp, grad) = posterior.logp_grad(&[-0.5]).unwrap();
    assert_eq!(logp, f64::NEG_INFINITY);
    assert!(
        grad[0].is_finite(),
        "out-of-support gradient must stay finite, got {}",
        grad[0]
    );
}

#[test]
fn truncated_normal_two_sided_matches_analytic() {
    // The bounded_rates corpus shape: Truncated(Normal(1, 1), -1, 3).
    let posterior = posterior_for(&truncated_site_doc(
        &normal_base(1.0, 1.0),
        Some(-1.0),
        Some(3.0),
    ));
    let log_z = (special::ndtr(2.0) - special::ndtr(-2.0)).ln();
    for x in [-1.0, -0.25, 1.0, 2.6, 3.0] {
        let (logp, grad) = posterior.logp_grad(&[x]).unwrap();
        assert_rel_close(
            logp,
            norm_logpdf(x, 1.0, 1.0) - log_z,
            1e-13,
            "two-sided logp",
        );
        assert!(
            (grad[0] - (1.0 - x)).abs() <= 1e-12,
            "two-sided gradient at {x}: {} vs {}",
            grad[0],
            1.0 - x
        );
    }
    for x in [-1.0 - 1e-9, 3.0 + 1e-9, -25.0, 40.0] {
        let logp = posterior.logp(&[x]).unwrap();
        assert_eq!(logp, f64::NEG_INFINITY, "outside bounds at {x}");
    }
}

#[test]
fn truncated_normal_mass_deep_in_tail_normalizes_to_one() {
    // Both bounds far in the right tail: Z = Phi(9) - Phi(8) suffers
    // catastrophic cancellation computed naively; the normalizer must keep
    // full relative precision there.
    let posterior = posterior_for(&truncated_site_doc(
        &normal_base(0.0, 1.0),
        Some(8.0),
        Some(9.0),
    ));
    // ln(Phi(-8) - Phi(-9)), mpmath at 60 digits.
    let want_log_z = -35.01361859343714811723477;
    let logp = posterior.logp(&[8.25]).unwrap();
    assert_rel_close(
        logp,
        norm_logpdf(8.25, 0.0, 1.0) - want_log_z,
        1e-12,
        "deep-tail two-sided logp",
    );
    integrates_to_one(&posterior, 8.0, 9.0, 1e-8, "deep-tail two-sided");
}

#[test]
fn truncated_normal_far_tail_one_sided_is_finite_and_correct() {
    // lower = 40: 1 - Phi(40) underflows to zero in a naive normalizer,
    // which would turn every in-support logp into +inf garbage.
    let posterior = posterior_for(&truncated_site_doc(
        &normal_base(0.0, 1.0),
        Some(40.0),
        None,
    ));
    // norm_logpdf(40.5) - ln(Phi(-40)), mpmath at 60 digits.
    let (logp, grad) = posterior.logp_grad(&[40.5]).unwrap();
    assert_rel_close(logp, -16.4354965194508845751735, 1e-12, "far-tail logp");
    // Normalizer is value-independent: dlogp/dx = -x.
    assert_rel_close(grad[0], -40.5, 1e-12, "far-tail gradient");
    // Essentially all conditional mass is in [40, 43].
    integrates_to_one(&posterior, 40.0, 43.0, 1e-8, "far-tail one-sided");
}

#[test]
fn truncated_normal_left_and_right_tail_truncations_are_symmetric() {
    // Truncated(N(0,1), [a, b]) at x and Truncated(N(0,1), [-b, -a]) at -x
    // are the same density; both tails must be computed with the same
    // relative accuracy.
    let right = posterior_for(&truncated_site_doc(
        &normal_base(0.0, 1.0),
        Some(0.5),
        Some(2.3),
    ));
    let left = posterior_for(&truncated_site_doc(
        &normal_base(0.0, 1.0),
        Some(-2.3),
        Some(-0.5),
    ));
    for x in [0.5, 0.9, 1.6, 2.3] {
        let lp_right = right.logp(&[x]).unwrap();
        let lp_left = left.logp(&[-x]).unwrap();
        assert_rel_close(lp_left, lp_right, 1e-14, "tail symmetry");
    }
}

#[test]
fn truncated_normal_symbolic_lower_bound_differentiates_the_normalizer() {
    // b ~ Normal(0, 1); x ~ Truncated(Normal(0, 1), lower = b).
    // logp = npdf(b) + npdf(x) - ln Phi(-b), so
    // dlogp/db = -b + phi(b) / Phi(-b).
    let doc = format!(
        r#"{{"bayeswire_ir": 1, "model": {{"node": "ModelMeta",
            "params": [
                {{"name": "b", "value": {{"node": "ResolvedParam",
                    "distribution": {base}, "constraint": null, "size": null}}}},
                {{"name": "x", "value": {{"node": "ResolvedParam",
                    "distribution": {trunc}, "constraint": null, "size": null}}}}],
            "data": [], "observed_nodes": [], "expressions": [],
            "free_values": [
                {{"name": "b", "value": {{"node": "ResolvedFreeValue",
                    "constraint": null, "size": null}}}},
                {{"name": "x", "value": {{"node": "ResolvedFreeValue",
                    "constraint": null, "size": null}}}}],
            "stochastic_sites": [
                {{"node": "ResolvedStochasticSite", "name": "b",
                    "distribution": {base},
                    "value": {{"node": "ParamRef", "name": "b"}}}},
                {{"node": "ResolvedStochasticSite", "name": "x",
                    "distribution": {trunc},
                    "value": {{"node": "ParamRef", "name": "x"}}}}]}}}}"#,
        base = normal_base(0.0, 1.0),
        trunc = format!(
            r#"{{"node": "Truncated", "base": {}, "lower": {{"node": "ParamRef", "name": "b"}},
                "upper": null}}"#,
            normal_base(0.0, 1.0)
        ),
    );
    let posterior = posterior_for(&doc);
    let q = [0.4, 1.2];
    let (logp, grad) = posterior.logp_grad(&q).unwrap();

    let want_logp =
        norm_logpdf(q[0], 0.0, 1.0) + norm_logpdf(q[1], 0.0, 1.0) - special::ndtr(-q[0]).ln();
    assert_rel_close(logp, want_logp, 1e-13, "symbolic-bound logp");
    // mpmath: phi(0.4)/Phi(-0.4) = 1.068756171745620878472933.
    let hazard = 1.068756171745620878472933;
    assert_rel_close(grad[0], -q[0] + hazard, 1e-11, "dlogp/db analytic");
    assert_rel_close(grad[1], -q[1], 1e-12, "dlogp/dx");

    // Central finite differences over the public logp confirm both entries.
    let h = 1e-6;
    for i in 0..2 {
        let mut plus = q;
        plus[i] += h;
        let mut minus = q;
        minus[i] -= h;
        let numeric =
            (posterior.logp(&plus).unwrap() - posterior.logp(&minus).unwrap()) / (2.0 * h);
        assert!(
            (numeric - grad[i]).abs() <= 1e-6 * (1.0 + numeric.abs()),
            "finite difference grad[{i}]: analytic {} vs numeric {numeric}",
            grad[i]
        );
    }
}

#[test]
fn truncated_normal_symbolic_loc_differentiates_the_standardized_bounds() {
    // mu ~ Normal(0, 1); x ~ Truncated(Normal(mu, 1), lower = 0):
    // logp = npdf(mu) + npdf(x - mu) - ln Phi(mu), so
    // dlogp/dmu = -mu + (x - mu) - phi(mu) / Phi(mu).
    let trunc = r#"{"node": "Truncated", "base": {"node": "Normal",
            "loc": {"node": "ParamRef", "name": "mu"},
            "scale": {"node": "ConstNode", "value": 1.0}},
            "lower": {"node": "ConstNode", "value": 0.0}, "upper": null}"#;
    let doc = format!(
        r#"{{"bayeswire_ir": 1, "model": {{"node": "ModelMeta",
            "params": [
                {{"name": "mu", "value": {{"node": "ResolvedParam",
                    "distribution": {base}, "constraint": null, "size": null}}}},
                {{"name": "x", "value": {{"node": "ResolvedParam",
                    "distribution": {trunc}, "constraint": null, "size": null}}}}],
            "data": [], "observed_nodes": [], "expressions": [],
            "free_values": [
                {{"name": "mu", "value": {{"node": "ResolvedFreeValue",
                    "constraint": null, "size": null}}}},
                {{"name": "x", "value": {{"node": "ResolvedFreeValue",
                    "constraint": null, "size": null}}}}],
            "stochastic_sites": [
                {{"node": "ResolvedStochasticSite", "name": "mu",
                    "distribution": {base},
                    "value": {{"node": "ParamRef", "name": "mu"}}}},
                {{"node": "ResolvedStochasticSite", "name": "x",
                    "distribution": {trunc},
                    "value": {{"node": "ParamRef", "name": "x"}}}}]}}}}"#,
        base = normal_base(0.0, 1.0),
    );
    let posterior = posterior_for(&doc);
    let q = [1.3, 2.1]; // mu, x
    let (logp, grad) = posterior.logp_grad(&q).unwrap();
    let (mu, x) = (q[0], q[1]);
    let want_logp = norm_logpdf(mu, 0.0, 1.0) + norm_logpdf(x, mu, 1.0) - special::ndtr(mu).ln();
    assert_rel_close(logp, want_logp, 1e-13, "symbolic-loc logp");
    let phi_mu = (-HALF_LOG_2PI - 0.5 * mu * mu).exp();
    let want_dmu = -mu + (x - mu) - phi_mu / special::ndtr(mu);
    assert!(
        (grad[0] - want_dmu).abs() <= 1e-12 * (1.0 + want_dmu.abs()),
        "dlogp/dmu: {} vs {want_dmu}",
        grad[0]
    );

    let h = 1e-6;
    for i in 0..2 {
        let mut plus = q;
        plus[i] += h;
        let mut minus = q;
        minus[i] -= h;
        let numeric =
            (posterior.logp(&plus).unwrap() - posterior.logp(&minus).unwrap()) / (2.0 * h);
        assert!(
            (numeric - grad[i]).abs() <= 1e-6 * (1.0 + numeric.abs()),
            "finite difference grad[{i}]: analytic {} vs numeric {numeric}",
            grad[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Log density: Exponential and Uniform bases
// ---------------------------------------------------------------------------

#[test]
fn truncated_exponential_matches_analytic() {
    // Exp(rate) on [l, u]: logp(x) = ln r - r x - ln(exp(-r l) - exp(-r u)).
    let rate = 1.5;
    let posterior = posterior_for(&truncated_site_doc(
        &exponential_base(rate),
        Some(1.0),
        Some(3.0),
    ));
    let log_z = ((-rate * 1.0f64).exp() - (-rate * 3.0f64).exp()).ln();
    for x in [1.0, 1.5, 2.9, 3.0] {
        let (logp, grad) = posterior.logp_grad(&[x]).unwrap();
        assert_rel_close(
            logp,
            rate.ln() - rate * x - log_z,
            1e-13,
            "truncated exponential logp",
        );
        assert_rel_close(grad[0], -rate, 1e-12, "truncated exponential gradient");
    }
    for x in [0.5, 3.5, -1.0] {
        assert_eq!(posterior.logp(&[x]).unwrap(), f64::NEG_INFINITY);
    }
    integrates_to_one(&posterior, 1.0, 3.0, 1e-8, "truncated exponential");
}

#[test]
fn truncated_exponential_lower_bound_below_base_support_clamps_to_zero() {
    // The truncation interval [-1, 2] intersects the Exponential support at
    // [0, 2]: Z = 1 - exp(-r * 2), and x in [-1, 0) stays outside support.
    let rate = 0.75;
    let posterior = posterior_for(&truncated_site_doc(
        &exponential_base(rate),
        Some(-1.0),
        Some(2.0),
    ));
    let log_z = (-(-rate * 2.0f64).exp()).ln_1p();
    for x in [0.0, 0.5, 2.0] {
        let logp = posterior.logp(&[x]).unwrap();
        assert_rel_close(
            logp,
            rate.ln() - rate * x - log_z,
            1e-13,
            "clamped-lower exponential logp",
        );
    }
    assert_eq!(posterior.logp(&[-0.5]).unwrap(), f64::NEG_INFINITY);
    integrates_to_one(&posterior, 0.0, 2.0, 1e-8, "clamped-lower exponential");
}

#[test]
fn truncated_exponential_deep_tail_is_finite_and_normalized() {
    // Mass far in the tail: [100, 102] at rate 1. exp(-100) is ~1e-44, so a
    // naive Z in probability space still works in f64 here, but the log-space
    // path must agree and integrate to one.
    let posterior = posterior_for(&truncated_site_doc(
        &exponential_base(1.0),
        Some(100.0),
        Some(102.0),
    ));
    let logp = posterior.logp(&[100.5]).unwrap();
    // ln r - r x - (-r l + ln(1 - exp(-r (u - l))))
    let want = -100.5f64 + 100.0 - (-(-2.0f64).exp()).ln_1p();
    assert_rel_close(logp, want, 1e-12, "deep-tail exponential logp");
    integrates_to_one(&posterior, 100.0, 102.0, 1e-8, "deep-tail exponential");
}

#[test]
fn truncated_uniform_matches_analytic() {
    // U(0, 10) truncated to [2, 3] is U(2, 3).
    let posterior = posterior_for(&truncated_site_doc(
        &uniform_base(0.0, 10.0),
        Some(2.0),
        Some(3.0),
    ));
    for x in [2.0, 2.5, 3.0] {
        let (logp, grad) = posterior.logp_grad(&[x]).unwrap();
        assert!(
            logp.abs() <= 1e-13,
            "truncated uniform logp at {x}: {logp}, want 0"
        );
        assert!(
            grad[0].abs() <= 1e-13,
            "truncated uniform gradient at {x}: {}",
            grad[0]
        );
    }
    for x in [1.9, 3.1] {
        assert_eq!(posterior.logp(&[x]).unwrap(), f64::NEG_INFINITY);
    }
}

#[test]
fn truncated_uniform_bounds_outside_base_support_clamp() {
    // U(0, 1) truncated to [-5, 0.5]: Z = 0.5, so logp = ln 2 on [0, 0.5].
    let posterior = posterior_for(&truncated_site_doc(
        &uniform_base(0.0, 1.0),
        Some(-5.0),
        Some(0.5),
    ));
    assert_rel_close(
        posterior.logp(&[0.25]).unwrap(),
        2.0f64.ln(),
        1e-13,
        "clamped truncated uniform logp",
    );
    assert_eq!(posterior.logp(&[-1.0]).unwrap(), f64::NEG_INFINITY);
    assert_eq!(posterior.logp(&[0.75]).unwrap(), f64::NEG_INFINITY);
}

// ---------------------------------------------------------------------------
// Compiled-tape replay across support flips
// ---------------------------------------------------------------------------

#[test]
fn compiled_truncated_tape_replays_support_flips_bitwise() {
    let posterior = posterior_for(&truncated_site_doc(
        &normal_base(1.0, 2.0),
        Some(-1.0),
        Some(3.0),
    ));
    let mut compiled = posterior.compile().unwrap();
    // In support, out both sides, back in: masks and normalizer must replay
    // to exactly the rebuilt values.
    for x in [0.5, -4.0, 2.9, 7.5, 0.5] {
        let (want_logp, want_grad) = posterior.logp_grad(&[x]).unwrap();
        let (logp, grad) = compiled.logp_grad(&[x]).unwrap();
        assert_eq!(
            logp.to_bits(),
            want_logp.to_bits(),
            "compiled logp at {x}: {logp:e} vs {want_logp:e}"
        );
        assert_eq!(
            grad[0].to_bits(),
            want_grad[0].to_bits(),
            "compiled grad at {x}"
        );
    }
}

// ---------------------------------------------------------------------------
// Prior simulation (inverse-CDF sampling)
// ---------------------------------------------------------------------------

fn simulate_scalar_draws(doc: &str, draws: usize, seed: u64) -> Vec<f64> {
    let document = json::parse(doc).expect("inline document parses");
    let meta = decode_model(&document).expect("inline document decodes");
    let run = simulate_prior_predictive(
        meta,
        vec![],
        &PriorPredictiveSettings { num_draws: draws },
        seed,
    )
    .expect("prior simulation succeeds");
    run.draws
        .iter()
        .map(|draw| draw.values[0].1.data()[0])
        .collect()
}

fn mean_and_sd(values: &[f64]) -> (f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() - 1) as f64;
    (mean, var.sqrt())
}

#[test]
fn truncated_normal_prior_simulation_matches_half_normal_moments() {
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(0.0), None);
    let values = simulate_scalar_draws(&doc, 8192, 41);
    assert!(values.iter().all(|&v| v >= 0.0 && v.is_finite()));
    let (mean, sd) = mean_and_sd(&values);
    // mpmath: mean sqrt(2/pi) = 0.79788, sd sqrt(1 - 2/pi) = 0.60281.
    assert!((mean - 0.7978845608028654).abs() < 0.03, "mean {mean}");
    assert!((sd - 0.6028102749890870).abs() < 0.03, "sd {sd}");
}

#[test]
fn truncated_normal_two_sided_prior_simulation_stays_in_bounds() {
    let doc = truncated_site_doc(&normal_base(1.0, 1.0), Some(-1.0), Some(3.0));
    let values = simulate_scalar_draws(&doc, 8192, 43);
    assert!(values.iter().all(|&v| (-1.0..=3.0).contains(&v)));
    let (mean, sd) = mean_and_sd(&values);
    // Symmetric truncation around loc: mean 1 exactly; mpmath sd 0.87963.
    assert!((mean - 1.0).abs() < 0.04, "mean {mean}");
    assert!((sd - 0.8796256610342398).abs() < 0.05, "sd {sd}");
}

#[test]
fn truncated_normal_far_tail_prior_simulation_is_finite_and_calibrated() {
    // lower = 8: u ~ U(Phi(8), 1) collapses to a handful of distinct doubles
    // computed naively; inverse-CDF sampling must keep tail resolution.
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(8.0), None);
    let values = simulate_scalar_draws(&doc, 4096, 47);
    assert!(
        values
            .iter()
            .all(|&v| v.is_finite() && (8.0..10.0).contains(&v)),
        "far-tail draws must be finite and near the bound"
    );
    let (mean, sd) = mean_and_sd(&values);
    // mpmath: E[X | X > 8] = 8.121368, sd = 0.119687.
    assert!((mean - 8.121368112236113).abs() < 0.02, "mean {mean}");
    assert!((sd - 0.11968660511243900).abs() < 0.02, "sd {sd}");
}

#[test]
fn truncated_normal_extreme_tail_prior_simulation_survives_cdf_underflow() {
    // lower = 40: Phi(-40) underflows f64 entirely, so any sampler that
    // subtracts raw CDF values reports zero probability mass; the log
    // density evaluates the same model without trouble, and simulation
    // must too.
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(40.0), None);
    let values = simulate_scalar_draws(&doc, 4096, 61);
    assert!(
        values
            .iter()
            .all(|&v| v.is_finite() && (40.0..41.0).contains(&v)),
        "extreme-tail draws must be finite and near the bound"
    );
    let (mean, sd) = mean_and_sd(&values);
    // mpmath: E[X | X > 40] = 40.024969, sd = 0.024953.
    assert!((mean - 40.02496884720726).abs() < 0.004, "mean {mean}");
    assert!((sd - 0.024953323998846).abs() < 0.004, "sd {sd}");
}

#[test]
fn truncated_normal_extreme_tail_two_sided_prior_simulation_is_calibrated() {
    // Both CDF endpoints underflow: [40, 40.01] keeps a sliver of tail
    // mass that only log-space sampling can resolve.
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(40.0), Some(40.01));
    let values = simulate_scalar_draws(&doc, 4096, 67);
    assert!(values.iter().all(|&v| (40.0..=40.01).contains(&v)));
    let (mean, sd) = mean_and_sd(&values);
    // mpmath: E = 40.004668, sd = 0.0028754.
    assert!((mean - 40.00466751192174).abs() < 0.0005, "mean {mean}");
    assert!((sd - 0.002875445105096).abs() < 0.0005, "sd {sd}");
}

#[test]
fn truncated_normal_extreme_left_tail_prior_simulation_mirrors() {
    // upper = -40 is the mirror image of lower = 40.
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), None, Some(-40.0));
    let values = simulate_scalar_draws(&doc, 4096, 71);
    assert!(values
        .iter()
        .all(|&v| v.is_finite() && (-41.0..=-40.0).contains(&v)));
    let (mean, _) = mean_and_sd(&values);
    assert!((mean + 40.02496884720726).abs() < 0.004, "mean {mean}");
}

#[test]
fn truncated_exponential_prior_simulation_matches_analytic_mean() {
    let doc = truncated_site_doc(&exponential_base(1.5), Some(1.0), Some(3.0));
    let values = simulate_scalar_draws(&doc, 8192, 53);
    assert!(values.iter().all(|&v| (1.0..=3.0).contains(&v)));
    let (mean, _) = mean_and_sd(&values);
    // mpmath: E = 1.561875.
    assert!((mean - 1.5618752736841548).abs() < 0.04, "mean {mean}");
}

#[test]
fn truncated_uniform_prior_simulation_matches_analytic_mean() {
    let doc = truncated_site_doc(&uniform_base(0.0, 10.0), Some(2.0), Some(3.0));
    let values = simulate_scalar_draws(&doc, 8192, 59);
    assert!(values.iter().all(|&v| (2.0..=3.0).contains(&v)));
    let (mean, _) = mean_and_sd(&values);
    assert!((mean - 2.5).abs() < 0.03, "mean {mean}");
}

#[test]
fn truncated_prior_simulation_is_deterministic_per_seed() {
    let doc = truncated_site_doc(&normal_base(0.0, 1.0), Some(0.0), None);
    let a = simulate_scalar_draws(&doc, 64, 7);
    let b = simulate_scalar_draws(&doc, 64, 7);
    assert_eq!(a, b);
    let c = simulate_scalar_draws(&doc, 64, 8);
    assert_ne!(a, c);
}
