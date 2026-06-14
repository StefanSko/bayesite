//! Statistical checks of the NUTS sampler against analytically known
//! targets, with fixed seeds.
//!
//! Margin reasoning: with ~2000 kept draws and ESS in the hundreds for
//! these unimodal targets, the Monte Carlo standard error of a posterior
//! mean is roughly sd/sqrt(ESS) ~ 0.05*sd. Assertions use ~5x that, so a
//! correct sampler fails with negligible probability while real bias of a
//! tenth of a standard deviation is still caught.

use bayesite_core::ir::{
    decode_model, Constraint, Distribution, Expr, ModelMeta, ResolvedParam, Size,
};
use bayesite_core::json;
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::sampler::{sample, Settings};

fn scalar_normal_model(loc: f64, scale: f64, constraint: Option<Constraint>) -> ModelMeta {
    ModelMeta {
        params: vec![(
            "x".to_string(),
            ResolvedParam {
                distribution: Distribution::Normal {
                    loc: Expr::Const(loc),
                    scale: Expr::Const(scale),
                },
                constraint,
                size: Size::Scalar,
            },
        )],
        data: vec![],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    }
}

fn moments(draws: &[Vec<f64>], dim: usize) -> (f64, f64) {
    let n = draws.len() as f64;
    let mean = draws.iter().map(|q| q[dim]).sum::<f64>() / n;
    let var = draws.iter().map(|q| (q[dim] - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

#[test]
fn standard_normal_moments() {
    let posterior = Posterior::new(scalar_normal_model(0.0, 1.0, None), vec![]).unwrap();
    let settings = Settings {
        num_warmup: 500,
        num_draws: 2000,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 20240601, 0).unwrap();
    assert_eq!(chain.draws.len(), 2000);
    let (mean, var) = moments(&chain.draws, 0);
    assert!(mean.abs() < 0.15, "mean {mean}");
    assert!((var - 1.0).abs() < 0.2, "var {var}");
    assert_eq!(chain.divergences, 0);
    assert!(chain.step_size > 0.1, "step size {}", chain.step_size);
}

#[test]
fn shifted_scaled_normal_moments() {
    let posterior = Posterior::new(scalar_normal_model(2.0, 3.0, None), vec![]).unwrap();
    let settings = Settings {
        num_warmup: 500,
        num_draws: 2000,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 7, 1).unwrap();
    let (mean, var) = moments(&chain.draws, 0);
    assert!((mean - 2.0).abs() < 0.45, "mean {mean}");
    assert!((var.sqrt() - 3.0).abs() < 0.5, "sd {}", var.sqrt());
    // The adapted metric should approach the target variance (9.0).
    assert!(
        chain.inv_mass[0] > 4.0 && chain.inv_mass[0] < 16.0,
        "inv_mass {:?}",
        chain.inv_mass
    );
}

#[test]
fn positive_constrained_normal_is_half_normal() {
    // Normal(0,1) prior under a Positive constraint: the constrained value
    // is half-normal with mean sqrt(2/pi) and sd sqrt(1 - 2/pi).
    let posterior = Posterior::new(
        scalar_normal_model(0.0, 1.0, Some(Constraint::Positive)),
        vec![],
    )
    .unwrap();
    let settings = Settings {
        num_warmup: 500,
        num_draws: 2000,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 99, 0).unwrap();
    let constrained: Vec<f64> = chain
        .draws
        .iter()
        .map(|q| posterior.constrain(q).unwrap()[0].1.data()[0])
        .collect();
    assert!(constrained.iter().all(|&c| c > 0.0));
    let n = constrained.len() as f64;
    let mean = constrained.iter().sum::<f64>() / n;
    let want_mean = (2.0 / std::f64::consts::PI).sqrt();
    assert!(
        (mean - want_mean).abs() < 0.1,
        "mean {mean} want {want_mean}"
    );
}

#[test]
fn chains_are_deterministic_per_seed_and_distinct_across_chain_ids() {
    let posterior = Posterior::new(scalar_normal_model(0.0, 1.0, None), vec![]).unwrap();
    let settings = Settings {
        num_warmup: 100,
        num_draws: 50,
        ..Settings::default()
    };
    let a = sample(&posterior, &settings, 1, 0).unwrap();
    let b = sample(&posterior, &settings, 1, 0).unwrap();
    let c = sample(&posterior, &settings, 1, 1).unwrap();
    assert_eq!(a.draws, b.draws);
    assert_ne!(a.draws, c.draws);
}

#[test]
fn eight_schools_samples_cleanly() {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/eight_schools_non_centered.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let doc = json::parse(&std::fs::read_to_string(path).unwrap()).unwrap();
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let data = data_from_json(doc.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();

    let settings = Settings {
        num_warmup: 400,
        num_draws: 400,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 42, 0).unwrap();
    assert_eq!(chain.draws.len(), 400);
    // The non-centered parameterization should sample without mass
    // divergences; allow a small number rather than zero.
    assert!(chain.divergences < 20, "divergences {}", chain.divergences);
    // tau is the second packed parameter (Positive-constrained).
    for q in &chain.draws {
        let constrained = posterior.constrain(q).unwrap();
        assert_eq!(constrained[1].0, "tau");
        assert!(constrained[1].1.data()[0] > 0.0);
    }
    // Population mean mu should be near the classic ~8 (very loose check).
    let (mu_mean, _) = {
        let n = chain.draws.len() as f64;
        let mean = chain.draws.iter().map(|q| q[0]).sum::<f64>() / n;
        (mean, 0.0)
    };
    assert!((0.0..16.0).contains(&mu_mean), "mu mean {mu_mean}");
}
