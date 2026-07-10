//! Posterior-output and statistical checks for lower-censored Exponential
//! observations represented by a `VectorBounds` free value.
//!
//! Margin reasoning: each fit uses four chains with 500 warmup and 1000 kept
//! draws per chain. For the conjugate Gamma target, the mean margin is five
//! estimated Monte Carlo standard errors, `5 * sd / sqrt(ESS)`. The variance
//! margin is six estimated Monte Carlo standard errors, using the asymptotic
//! variance `(m4 - variance^2) / ESS`. These match the reference-backend
//! thresholds: with ESS in the hundreds or thousands they make seed-specific
//! Monte Carlo failures unlikely while remaining much tighter than the
//! complete-case bias (analytic posterior mean 4.799 versus true rate 2.0).

use bayesite_core::diagnostics::effective_sample_size;
use bayesite_core::ir::{
    Constraint, DataSchema, Dim, Distribution, Expr, ModelMeta, ResolvedData, ResolvedFreeValue,
    ResolvedParam, ResolvedStochasticSite, Size,
};
use bayesite_core::json;
use bayesite_core::model::{DataValue, Posterior};
use bayesite_core::protocol::ndjson_lines;
use bayesite_core::sampler::{sample, ChainDraws, Settings};

const N: i64 = 50;
const N_OBS: i64 = 26;
const N_MIS: i64 = 24;
const TRUE_RATE: f64 = 2.0;
const POSTERIOR_SHAPE: f64 = 27.0;
const POSTERIOR_RATE: f64 = 14.248_222_681_999_998;

const OBSERVED_IDX: [i64; 26] = [
    0, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 18, 26, 28, 30, 31, 32, 34, 35, 42, 45, 46, 47, 48,
    49,
];
const MISSING_IDX: [i64; 24] = [
    1, 2, 3, 8, 14, 15, 19, 20, 21, 22, 23, 24, 25, 27, 29, 33, 36, 37, 38, 39, 40, 41, 43, 44,
];
const OBSERVED_VALUES: [f64; 26] = [
    0.149_883_056_8,
    0.102_728_058_8,
    0.131_045_246_8,
    0.079_591_613_2,
    0.127_551_255_6,
    0.069_753_475_5,
    0.378_911_924_3,
    0.120_340_423_9,
    0.174_535_360_3,
    0.282_447_637_3,
    0.007_268_776_5,
    0.161_369_286_1,
    0.079_328_095,
    0.148_333_139_2,
    0.333_530_689_5,
    0.415_215_486_4,
    0.280_689_110_5,
    0.226_437_679_8,
    0.188_264_924_3,
    0.062_418_959_9,
    0.072_428_362_1,
    0.097_177_078_1,
    0.113_216_121_6,
    0.369_798_010_4,
    0.146_053_762_5,
    0.307_592_416_4,
];
const MISSING_LOWER: [f64; 24] = [
    0.414_421_768_7,
    0.448_544_973,
    0.436_320_936_7,
    0.286_873_336_2,
    0.313_352_087_1,
    0.262_030_424,
    0.416_956_976_2,
    0.449_060_735_6,
    0.434_574_683_1,
    0.380_311_835_7,
    0.311_792_858_3,
    0.261_243_296_6,
    0.252_437_399_5,
    0.355_042_268_8,
    0.449_276_640_6,
    0.260_481_263_2,
    0.356_720_807_3,
    0.419_416_466_8,
    0.449_464_477_4,
    0.432_732_790_1,
    0.377_090_578_8,
    0.308_707_245_1,
    0.253_230_386_8,
    0.292_228_495_6,
];

fn declared_data(name: &str, dims: Vec<Dim>) -> (String, ResolvedData) {
    (
        name.to_string(),
        ResolvedData {
            schema: DataSchema::Shape(dims),
        },
    )
}

fn scalar_data(name: &str, value: i64) -> (String, DataValue) {
    (
        name.to_string(),
        DataValue {
            shape: vec![],
            values: vec![value as f64],
            integer: true,
        },
    )
}

fn vector_data(name: &str, values: &[f64], integer: bool) -> (String, DataValue) {
    (
        name.to_string(),
        DataValue {
            shape: vec![values.len()],
            values: values.to_vec(),
            integer,
        },
    )
}

fn index_data(name: &str, values: &[i64]) -> (String, DataValue) {
    let values = values.iter().map(|&value| value as f64).collect::<Vec<_>>();
    vector_data(name, &values, true)
}

fn rate_param() -> (String, ResolvedParam) {
    (
        "rate".to_string(),
        ResolvedParam {
            distribution: Distribution::Exponential {
                rate: Expr::Const(1.0),
            },
            constraint: Some(Constraint::Positive),
            size: Size::Scalar,
        },
    )
}

fn rate_free_value() -> (String, ResolvedFreeValue) {
    (
        "rate".to_string(),
        ResolvedFreeValue {
            constraint: Some(Constraint::Positive),
            size: Size::Scalar,
        },
    )
}

fn rate_prior_site() -> ResolvedStochasticSite {
    ResolvedStochasticSite {
        name: "rate".to_string(),
        distribution: Distribution::Exponential {
            rate: Expr::Const(1.0),
        },
        value: Expr::Param("rate".to_string()),
    }
}

fn censored_model() -> ModelMeta {
    ModelMeta {
        params: vec![rate_param()],
        data: vec![
            declared_data("n", vec![]),
            declared_data("n_obs", vec![]),
            declared_data("n_mis", vec![]),
            declared_data("observed_idx", vec![Dim::DataDim("n_obs".to_string())]),
            declared_data("missing_idx", vec![Dim::DataDim("n_mis".to_string())]),
            declared_data("observed_values", vec![Dim::DataDim("n_obs".to_string())]),
            declared_data("missing_lower", vec![Dim::DataDim("n_mis".to_string())]),
        ],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![
            rate_free_value(),
            (
                "y".to_string(),
                ResolvedFreeValue {
                    constraint: Some(Constraint::VectorBounds {
                        lower: Some("missing_lower".to_string()),
                        upper: None,
                    }),
                    size: Size::Data("n_mis".to_string()),
                },
            ),
        ],
        stochastic_sites: vec![
            rate_prior_site(),
            ResolvedStochasticSite {
                name: "y".to_string(),
                distribution: Distribution::Exponential {
                    rate: Expr::Param("rate".to_string()),
                },
                value: Expr::VectorScatter {
                    length: Box::new(Expr::Data("n".to_string())),
                    observed_idx: Box::new(Expr::Data("observed_idx".to_string())),
                    observed_values: Box::new(Expr::Data("observed_values".to_string())),
                    missing_idx: Box::new(Expr::Data("missing_idx".to_string())),
                    missing_values: Box::new(Expr::Param("y".to_string())),
                },
            },
        ],
    }
}

fn censored_data() -> Vec<(String, DataValue)> {
    vec![
        scalar_data("n", N),
        scalar_data("n_obs", N_OBS),
        scalar_data("n_mis", N_MIS),
        index_data("observed_idx", &OBSERVED_IDX),
        index_data("missing_idx", &MISSING_IDX),
        vector_data("observed_values", &OBSERVED_VALUES, false),
        vector_data("missing_lower", &MISSING_LOWER, false),
    ]
}

fn complete_case_model() -> ModelMeta {
    ModelMeta {
        params: vec![rate_param()],
        data: vec![declared_data("observed_values", vec![Dim::Fixed(N_OBS)])],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![rate_free_value()],
        stochastic_sites: vec![
            rate_prior_site(),
            ResolvedStochasticSite {
                name: "y".to_string(),
                distribution: Distribution::Exponential {
                    rate: Expr::Param("rate".to_string()),
                },
                value: Expr::Data("observed_values".to_string()),
            },
        ],
    }
}

fn complete_case_data() -> Vec<(String, DataValue)> {
    vec![vector_data("observed_values", &OBSERVED_VALUES, false)]
}

fn settings() -> Settings {
    Settings {
        num_warmup: 500,
        num_draws: 1_000,
        target_accept: 0.9,
        ..Settings::default()
    }
}

fn run_chains(posterior: &Posterior, seed: u64) -> Vec<ChainDraws> {
    let settings = settings();
    (0..4)
        .map(|chain_id| sample(posterior, &settings, seed, chain_id).unwrap())
        .collect()
}

fn constrained_rate_chains(posterior: &Posterior, chains: &[ChainDraws]) -> Vec<Vec<f64>> {
    chains
        .iter()
        .map(|chain| {
            chain
                .draws
                .iter()
                .map(|q| {
                    let constrained = posterior.constrain(q).unwrap();
                    assert_eq!(constrained[0].0, "rate");
                    assert_eq!(constrained[0].1.shape(), &[]);
                    constrained[0].1.data()[0]
                })
                .collect()
        })
        .collect()
}

fn mean(values: &[Vec<f64>]) -> f64 {
    let count = values.iter().map(Vec::len).sum::<usize>() as f64;
    values.iter().flatten().sum::<f64>() / count
}

fn sample_variance(values: &[Vec<f64>], center: f64) -> f64 {
    let count = values.iter().map(Vec::len).sum::<usize>() as f64;
    values
        .iter()
        .flatten()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (count - 1.0)
}

fn divergence_count(chains: &[ChainDraws]) -> usize {
    chains.iter().map(|chain| chain.divergences).sum()
}

#[test]
fn censored_rate_matches_conjugate_gamma_and_reports_bounded_draws() {
    let posterior = Posterior::new(censored_model(), censored_data()).unwrap();
    let chains = run_chains(&posterior, 1_701);
    let rate_chains = constrained_rate_chains(&posterior, &chains);
    let ess = effective_sample_size(&rate_chains);
    let posterior_mean = mean(&rate_chains);
    let variance = sample_variance(&rate_chains, posterior_mean);
    let count = rate_chains.iter().map(Vec::len).sum::<usize>() as f64;
    let fourth_central_moment = rate_chains
        .iter()
        .flatten()
        .map(|value| (value - posterior_mean).powi(4))
        .sum::<f64>()
        / count;
    let mean_mcse = variance.sqrt() / ess.sqrt();
    let variance_mcse = ((fourth_central_moment - variance.powi(2)).max(0.0) / ess).sqrt();
    let expected_mean = POSTERIOR_SHAPE / POSTERIOR_RATE;
    let expected_variance = POSTERIOR_SHAPE / POSTERIOR_RATE.powi(2);

    println!(
        "censored conjugate: mean={posterior_mean:.9} variance={variance:.9} ESS={ess:.1} \
         mean_margin={:.9} variance_margin={:.9}",
        5.0 * mean_mcse,
        6.0 * variance_mcse
    );
    assert!(
        (posterior_mean - expected_mean).abs() <= 5.0 * mean_mcse,
        "mean {posterior_mean} expected {expected_mean}, MCSE {mean_mcse}, ESS {ess}"
    );
    assert!(
        (variance - expected_variance).abs() <= 6.0 * variance_mcse,
        "variance {variance} expected {expected_variance}, MCSE {variance_mcse}, ESS {ess}"
    );
    assert_eq!(divergence_count(&chains), 0);

    for chain in &chains {
        for q in &chain.draws {
            let reported = posterior.constrain(q).unwrap();
            assert_eq!(reported[1].0, "y");
            assert_eq!(reported[1].1.shape(), &[N_MIS as usize]);
            assert!(
                reported[1]
                    .1
                    .data()
                    .iter()
                    .zip(MISSING_LOWER)
                    .all(|(&value, lower)| value > lower),
                "reported y draw did not satisfy its coordinate-wise lower bound"
            );
        }
    }
}

#[test]
fn censored_model_recovers_rate_better_than_complete_case_fit() {
    let censored_posterior = Posterior::new(censored_model(), censored_data()).unwrap();
    let complete_case_posterior =
        Posterior::new(complete_case_model(), complete_case_data()).unwrap();
    let censored_chains = run_chains(&censored_posterior, 1_811);
    let complete_case_chains = run_chains(&complete_case_posterior, 1_811);
    let censored_mean = mean(&constrained_rate_chains(
        &censored_posterior,
        &censored_chains,
    ));
    let complete_case_mean = mean(&constrained_rate_chains(
        &complete_case_posterior,
        &complete_case_chains,
    ));

    println!(
        "bias control: censored_mean={censored_mean:.9} complete_case_mean={complete_case_mean:.9}"
    );
    assert!((censored_mean - TRUE_RATE).abs() < 0.45);
    assert!(complete_case_mean > 4.0);
    assert!(
        (censored_mean - TRUE_RATE).abs() < (complete_case_mean - TRUE_RATE).abs(),
        "censored mean {censored_mean}, complete-case mean {complete_case_mean}"
    );
    assert_eq!(divergence_count(&censored_chains), 0);
    assert_eq!(divergence_count(&complete_case_chains), 0);
}

#[test]
fn censored_output_packing_reports_scalar_rate_and_missing_y_vector() {
    let posterior = Posterior::new(censored_model(), censored_data()).unwrap();
    assert_eq!(
        posterior.packing(),
        vec![
            ("rate".to_string(), vec![]),
            ("y".to_string(), vec![N_MIS as usize]),
        ]
    );

    let output_settings = Settings {
        num_warmup: 50,
        num_draws: 4,
        target_accept: 0.9,
        ..Settings::default()
    };
    let chain = sample(&posterior, &output_settings, 1_901, 0).unwrap();
    let lines = ndjson_lines(&posterior, &output_settings, 1_901, &[(0, chain)]).unwrap();
    let header = json::parse(&lines[0]).unwrap();
    let params = header.get("params").unwrap().as_array().unwrap();
    assert_eq!(params[0].get("name").unwrap().as_str(), Some("rate"));
    assert_eq!(
        params[0].get("shape").unwrap().as_array(),
        Some([].as_slice())
    );
    assert_eq!(params[1].get("name").unwrap().as_str(), Some("y"));
    assert_eq!(
        params[1].get("shape").unwrap().as_array().unwrap()[0].as_i64(),
        Some(N_MIS)
    );

    let draw = json::parse(&lines[1]).unwrap();
    assert!(draw
        .get("values")
        .unwrap()
        .get("rate")
        .unwrap()
        .as_f64()
        .is_some());
    assert_eq!(
        draw.get("values")
            .unwrap()
            .get("y")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        N_MIS as usize
    );
}
