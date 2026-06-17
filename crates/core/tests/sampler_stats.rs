//! Statistical checks of the NUTS sampler against analytically known
//! targets, with fixed seeds.
//!
//! Margin reasoning: with ~2000 kept draws and ESS in the hundreds for
//! these unimodal targets, the Monte Carlo standard error of a posterior
//! mean is roughly sd/sqrt(ESS) ~ 0.05*sd. Assertions use ~5x that, so a
//! correct sampler fails with negligible probability while real bias of a
//! tenth of a standard deviation is still caught.

use bayesite_core::ir::{
    decode_model, Constraint, DataSchema, Dim, Distribution, Expr, ModelMeta, ResolvedData,
    ResolvedParam, Size,
};
use bayesite_core::json;
use bayesite_core::model::{data_from_json, DataValue, Posterior};
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

fn vector_normal_model(loc: &[f64], scale: &[f64]) -> (ModelMeta, Vec<(String, DataValue)>) {
    assert_eq!(loc.len(), scale.len());
    let dim = loc.len() as i64;
    let model = ModelMeta {
        params: vec![(
            "x".to_string(),
            ResolvedParam {
                distribution: Distribution::Normal {
                    loc: Expr::Data("loc".to_string()),
                    scale: Expr::Data("scale".to_string()),
                },
                constraint: None,
                size: Size::Fixed(dim),
            },
        )],
        data: vec![
            (
                "loc".to_string(),
                ResolvedData {
                    schema: DataSchema::Shape(vec![Dim::Fixed(dim)]),
                },
            ),
            (
                "scale".to_string(),
                ResolvedData {
                    schema: DataSchema::Shape(vec![Dim::Fixed(dim)]),
                },
            ),
        ],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    };
    let data = vec![
        (
            "loc".to_string(),
            DataValue {
                shape: vec![loc.len()],
                values: loc.to_vec(),
                integer: false,
            },
        ),
        (
            "scale".to_string(),
            DataValue {
                shape: vec![scale.len()],
                values: scale.to_vec(),
                integer: false,
            },
        ),
    ];
    (model, data)
}

fn correlated_mvn_model() -> (ModelMeta, Vec<(String, DataValue)>) {
    let model = ModelMeta {
        params: vec![(
            "x".to_string(),
            ResolvedParam {
                distribution: Distribution::MultivariateNormal {
                    mean: Expr::Data("mean".to_string()),
                    scale_tril: Expr::Data("scale_tril".to_string()),
                },
                constraint: None,
                size: Size::Fixed(2),
            },
        )],
        data: vec![
            (
                "mean".to_string(),
                ResolvedData {
                    schema: DataSchema::Shape(vec![Dim::Fixed(2)]),
                },
            ),
            (
                "scale_tril".to_string(),
                ResolvedData {
                    schema: DataSchema::Shape(vec![Dim::Fixed(2), Dim::Fixed(2)]),
                },
            ),
        ],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    };
    let data = vec![
        (
            "mean".to_string(),
            DataValue {
                shape: vec![2],
                values: vec![1.0, -0.5],
                integer: false,
            },
        ),
        (
            "scale_tril".to_string(),
            DataValue {
                shape: vec![2, 2],
                // Covariance is [[1.0, 0.6], [0.6, 1.0]].
                values: vec![1.0, 0.0, 0.6, 0.8],
                integer: false,
            },
        ),
    ];
    (model, data)
}

fn moments(draws: &[Vec<f64>], dim: usize) -> (f64, f64) {
    let n = draws.len() as f64;
    let mean = draws.iter().map(|q| q[dim]).sum::<f64>() / n;
    let var = draws.iter().map(|q| (q[dim] - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

fn covariance(draws: &[Vec<f64>], i: usize, j: usize) -> f64 {
    let n = draws.len() as f64;
    let mean_i = draws.iter().map(|q| q[i]).sum::<f64>() / n;
    let mean_j = draws.iter().map(|q| q[j]).sum::<f64>() / n;
    draws
        .iter()
        .map(|q| (q[i] - mean_i) * (q[j] - mean_j))
        .sum::<f64>()
        / (n - 1.0)
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
fn vector_diagonal_normal_moments() {
    let loc = [0.0, 2.0, -1.0];
    let scale = [1.0, 0.5, 2.0];
    let (model, data) = vector_normal_model(&loc, &scale);
    let posterior = Posterior::new(model, data).unwrap();
    let settings = Settings {
        num_warmup: 700,
        num_draws: 3000,
        ..Settings::default()
    };

    let chain = sample(&posterior, &settings, 20240618, 2).unwrap();

    assert_eq!(chain.draws.len(), settings.num_draws);
    assert_eq!(chain.divergences, 0);
    for dim in 0..loc.len() {
        let (mean, var) = moments(&chain.draws, dim);
        assert!(
            (mean - loc[dim]).abs() < 0.25 * scale[dim],
            "dim {dim} mean {mean} want {}",
            loc[dim]
        );
        assert!(
            (var.sqrt() - scale[dim]).abs() < 0.25 * scale[dim].max(1.0),
            "dim {dim} sd {} want {}",
            var.sqrt(),
            scale[dim]
        );
    }
    assert!(
        (0.4..2.0).contains(&chain.inv_mass[0])
            && (0.05..1.0).contains(&chain.inv_mass[1])
            && (1.5..8.0).contains(&chain.inv_mass[2]),
        "adapted diagonal metric {:?}",
        chain.inv_mass
    );
}

#[test]
fn correlated_mvn_moments() {
    let (model, data) = correlated_mvn_model();
    let posterior = Posterior::new(model, data).unwrap();
    let settings = Settings {
        num_warmup: 800,
        num_draws: 4000,
        ..Settings::default()
    };

    let chain = sample(&posterior, &settings, 20240619, 0).unwrap();

    assert_eq!(chain.draws.len(), settings.num_draws);
    assert_eq!(chain.divergences, 0);
    let (mean0, var0) = moments(&chain.draws, 0);
    let (mean1, var1) = moments(&chain.draws, 1);
    let cov01 = covariance(&chain.draws, 0, 1);
    assert!((mean0 - 1.0).abs() < 0.2, "mean0 {mean0}");
    assert!((mean1 + 0.5).abs() < 0.2, "mean1 {mean1}");
    assert!((var0 - 1.0).abs() < 0.25, "var0 {var0}");
    assert!((var1 - 1.0).abs() < 0.25, "var1 {var1}");
    assert!((cov01 - 0.6).abs() < 0.25, "cov01 {cov01}");
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
