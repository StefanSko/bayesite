use std::hint::black_box;
use std::time::{Duration, Instant};

use bayesite_core::ir::{
    DataSchema, Dim, Distribution, Expr, ModelMeta, ResolvedData, ResolvedParam, Size,
};
use bayesite_core::model::{DataValue, Posterior};

struct BenchTarget {
    name: String,
    dim: usize,
    posterior: Posterior,
    q: Vec<f64>,
    evaluations: usize,
}

fn data_value_vector(values: &[f64]) -> DataValue {
    DataValue {
        shape: vec![values.len()],
        values: values.to_vec(),
        integer: false,
    }
}

fn data_value_matrix(values: &[Vec<f64>]) -> DataValue {
    DataValue {
        shape: vec![values.len(), values.len()],
        values: values.iter().flat_map(|row| row.iter().copied()).collect(),
        integer: false,
    }
}

fn scalar_normal_target(name: &str, loc: f64, scale: f64, evaluations: usize) -> BenchTarget {
    let posterior = Posterior::new(
        ModelMeta {
            params: vec![(
                "x".to_string(),
                ResolvedParam {
                    distribution: Distribution::Normal {
                        loc: Expr::Const(loc),
                        scale: Expr::Const(scale),
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
        },
        vec![],
    )
    .unwrap();
    BenchTarget {
        name: name.to_string(),
        dim: 1,
        posterior,
        q: vec![loc + 0.25 * scale],
        evaluations,
    }
}

fn diagonal_normal_target(dim: usize, evaluations: usize) -> BenchTarget {
    let loc = (0..dim)
        .map(|i| ((i % 7) as f64 - 3.0) * 0.25)
        .collect::<Vec<_>>();
    let scale = (0..dim)
        .map(|i| 0.5 + (i % 11) as f64 * 0.15)
        .collect::<Vec<_>>();
    let q = loc
        .iter()
        .zip(&scale)
        .enumerate()
        .map(|(i, (&m, &s))| m + (((i % 5) as f64) - 2.0) * 0.1 * s)
        .collect::<Vec<_>>();
    let dim_i64 = dim as i64;
    let posterior = Posterior::new(
        ModelMeta {
            params: vec![(
                "x".to_string(),
                ResolvedParam {
                    distribution: Distribution::Normal {
                        loc: Expr::Data("loc".to_string()),
                        scale: Expr::Data("scale".to_string()),
                    },
                    constraint: None,
                    size: Size::Fixed(dim_i64),
                },
            )],
            data: vec![
                (
                    "loc".to_string(),
                    ResolvedData {
                        schema: DataSchema::Shape(vec![Dim::Fixed(dim_i64)]),
                    },
                ),
                (
                    "scale".to_string(),
                    ResolvedData {
                        schema: DataSchema::Shape(vec![Dim::Fixed(dim_i64)]),
                    },
                ),
            ],
            observed_nodes: vec![],
            expressions: vec![],
            free_values: vec![],
            stochastic_sites: vec![],
        },
        vec![
            ("loc".to_string(), data_value_vector(&loc)),
            ("scale".to_string(), data_value_vector(&scale)),
        ],
    )
    .unwrap();
    BenchTarget {
        name: format!("diagonal_normal_d{dim}"),
        dim,
        posterior,
        q,
        evaluations,
    }
}

fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    l
}

fn correlated_mvn_target(dim: usize, rho: f64, evaluations: usize) -> BenchTarget {
    let mean = (0..dim)
        .map(|i| ((i % 5) as f64 - 2.0) * 0.2)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![rho; dim]; dim];
    for i in 0..dim {
        covariance[i][i] = 1.0;
    }
    let scale_tril = cholesky(&covariance);
    let q = mean
        .iter()
        .enumerate()
        .map(|(i, &m)| m + (((i % 7) as f64) - 3.0) * 0.03)
        .collect::<Vec<_>>();
    let dim_i64 = dim as i64;
    let posterior = Posterior::new(
        ModelMeta {
            params: vec![(
                "x".to_string(),
                ResolvedParam {
                    distribution: Distribution::MultivariateNormal {
                        mean: Expr::Data("mean".to_string()),
                        scale_tril: Expr::Data("scale_tril".to_string()),
                    },
                    constraint: None,
                    size: Size::Fixed(dim_i64),
                },
            )],
            data: vec![
                (
                    "mean".to_string(),
                    ResolvedData {
                        schema: DataSchema::Shape(vec![Dim::Fixed(dim_i64)]),
                    },
                ),
                (
                    "scale_tril".to_string(),
                    ResolvedData {
                        schema: DataSchema::Shape(vec![Dim::Fixed(dim_i64), Dim::Fixed(dim_i64)]),
                    },
                ),
            ],
            observed_nodes: vec![],
            expressions: vec![],
            free_values: vec![],
            stochastic_sites: vec![],
        },
        vec![
            ("mean".to_string(), data_value_vector(&mean)),
            ("scale_tril".to_string(), data_value_matrix(&scale_tril)),
        ],
    )
    .unwrap();
    BenchTarget {
        name: format!("correlated_mvn_rho_{rho:.2}_d{dim}"),
        dim,
        posterior,
        q,
        evaluations,
    }
}

fn measure(target: &BenchTarget) -> (Duration, f64) {
    for _ in 0..32 {
        let _ = black_box(target.posterior.logp_grad(black_box(&target.q)).unwrap());
    }
    let start = Instant::now();
    let mut checksum = 0.0;
    for _ in 0..target.evaluations {
        let (logp, grad) = target.posterior.logp_grad(black_box(&target.q)).unwrap();
        checksum += logp * 1e-12;
        checksum += grad.iter().sum::<f64>() * 1e-12;
    }
    (start.elapsed(), black_box(checksum))
}

fn main() {
    let targets = vec![
        scalar_normal_target("scalar_standard_normal", 0.0, 1.0, 50_000),
        scalar_normal_target("shifted_scaled_normal", 2.0, 3.0, 50_000),
        diagonal_normal_target(32, 20_000),
        diagonal_normal_target(128, 8_000),
        diagonal_normal_target(512, 2_000),
        correlated_mvn_target(16, 0.3, 10_000),
        correlated_mvn_target(32, 0.3, 4_000),
        correlated_mvn_target(64, 0.3, 1_000),
    ];
    println!("target,dim,evaluations,seconds,evals_per_second,ns_per_eval,checksum");
    for target in &targets {
        let (elapsed, checksum) = measure(target);
        let seconds = elapsed.as_secs_f64();
        let evals_per_second = target.evaluations as f64 / seconds;
        let ns_per_eval = seconds * 1e9 / target.evaluations as f64;
        println!(
            "{},{},{},{seconds:.9},{evals_per_second:.2},{ns_per_eval:.2},{checksum:.17e}",
            target.name, target.dim, target.evaluations
        );
    }
}
