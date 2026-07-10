//! Self-contained prior-predictive analytic checks.

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{
    decode_model, DataSchema, Dim, Distribution, Expr, ModelMeta, ResolvedData, ResolvedFreeValue,
    ResolvedObserved, ResolvedParam, ResolvedStochasticSite, Size,
};
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, DataValue};
use bayesite_core::predictive::{
    prior_predictive_ndjson_lines, simulate_prior_predictive, PriorPredictiveSettings,
};

fn golden_model_and_data(name: &str) -> (ModelMeta, Vec<(String, DataValue)>) {
    let root = format!("{}/../../tests/golden_ir", env!("CARGO_MANIFEST_DIR"));
    let model_document = json::parse(
        &std::fs::read_to_string(format!("{root}/{name}.json")).expect("golden model readable"),
    )
    .expect("golden model parses");
    let data_document = json::parse(
        &std::fs::read_to_string(format!("{root}/data/{name}.json")).expect("golden data readable"),
    )
    .expect("golden data parses");
    (
        decode_model(&model_document).expect("golden model decodes"),
        data_from_json(&data_document).expect("golden data binds"),
    )
}

fn scalar_normal_model(loc: f64, scale: f64) -> ModelMeta {
    let distribution = Distribution::Normal {
        loc: Expr::Const(loc),
        scale: Expr::Const(scale),
    };
    ModelMeta {
        params: vec![(
            "theta".to_string(),
            ResolvedParam {
                distribution: distribution.clone(),
                constraint: None,
                size: Size::Scalar,
            },
        )],
        data: vec![],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![(
            "theta".to_string(),
            ResolvedFreeValue {
                constraint: None,
                size: Size::Scalar,
            },
        )],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "theta".to_string(),
            distribution,
            value: Expr::Param("theta".to_string()),
        }],
    }
}

fn scalar_bernoulli_observed_model(probs: f64) -> ModelMeta {
    let distribution = Distribution::Bernoulli {
        probs: Expr::Const(probs),
    };
    ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![ResolvedObserved {
            name: "y".to_string(),
            distribution: distribution.clone(),
        }],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "y".to_string(),
            distribution,
            value: Expr::Data("y".to_string()),
        }],
    }
}

fn scalar_normal_with_vector_loc_model() -> ModelMeta {
    let distribution = Distribution::Normal {
        loc: Expr::Data("loc".to_string()),
        scale: Expr::Const(1.0),
    };
    ModelMeta {
        params: vec![(
            "theta".to_string(),
            ResolvedParam {
                distribution: distribution.clone(),
                constraint: None,
                size: Size::Scalar,
            },
        )],
        data: vec![(
            "loc".to_string(),
            ResolvedData {
                schema: DataSchema::Shape(vec![Dim::Fixed(3)]),
            },
        )],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![(
            "theta".to_string(),
            ResolvedFreeValue {
                constraint: None,
                size: Size::Scalar,
            },
        )],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "theta_site".to_string(),
            distribution,
            value: Expr::Param("theta".to_string()),
        }],
    }
}

#[test]
fn prior_predictive_rejects_additional_unbounded_factor() {
    let mut model = scalar_normal_model(0.0, 1.0);
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "penalty".to_string(),
        distribution: Distribution::Exponential {
            rate: Expr::Const(1.0),
        },
        value: Expr::Param("theta".to_string()),
    });

    let err =
        simulate_prior_predictive(model, vec![], &PriorPredictiveSettings { num_draws: 1 }, 37)
            .unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(
        err.message.contains("additional stochastic factor"),
        "{}",
        err.message
    );
}

#[test]
fn prior_predictive_rejects_factor_colliding_with_declaration_name() {
    let mut model = scalar_normal_model(0.0, 1.0);
    model.stochastic_sites.insert(
        0,
        ResolvedStochasticSite {
            name: "theta".to_string(),
            distribution: Distribution::Exponential {
                rate: Expr::Const(1.0),
            },
            value: Expr::Param("theta".to_string()),
        },
    );

    let err =
        simulate_prior_predictive(model, vec![], &PriorPredictiveSettings { num_draws: 1 }, 39)
            .unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(
        err.message.contains("additional stochastic factor"),
        "{}",
        err.message
    );
}

#[test]
fn scalar_normal_prior_predictive_matches_analytic_moments() {
    let loc = 2.0;
    let scale = 3.0;
    let settings = PriorPredictiveSettings { num_draws: 8192 };
    let run = simulate_prior_predictive(scalar_normal_model(loc, scale), vec![], &settings, 41)
        .expect("prior predictive simulation succeeds");

    assert_eq!(run.sites.len(), 1);
    assert_eq!(run.sites[0].name, "theta");
    assert_eq!(run.draws.len(), settings.num_draws);

    let values: Vec<f64> = run
        .draws
        .iter()
        .map(|draw| draw.values[0].1.data()[0])
        .collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;

    assert!(
        (mean - loc).abs() < 0.15,
        "mean {mean} should be close to {loc}"
    );
    assert!(
        (variance - scale * scale).abs() < 0.6,
        "variance {variance} should be close to {}",
        scale * scale
    );
}

#[test]
fn censored_exponential_prior_predictive_samples_shifted_missing_values() {
    let (meta, data) = golden_model_and_data("censored_exponential");
    let settings = PriorPredictiveSettings { num_draws: 8192 };
    let run = simulate_prior_predictive(meta, data, &settings, 113)
        .expect("censored Exponential prior predictive succeeds");

    assert_eq!(run.sites[0].name, "rate");
    assert_eq!(run.sites[1].name, "y");
    assert_eq!(run.sites[1].role.as_str(), "observed");
    let mut standardized_residual_sum = 0.0;
    for draw in &run.draws {
        let rate = draw.values[0].1.data()[0];
        let y = draw.values[1].1.data();
        assert!(y[2] >= 1.5, "{}", y[2]);
        assert!(y[4] >= 2.0, "{}", y[4]);
        standardized_residual_sum += rate * (y[2] - 1.5);
        standardized_residual_sum += rate * (y[4] - 2.0);
    }
    let mean = standardized_residual_sum / (2 * settings.num_draws) as f64;
    assert!(
        (mean - 1.0).abs() < 0.04,
        "standardized shifted-Exponential mean {mean} should be close to 1"
    );
}

#[test]
fn interval_censored_normal_prior_predictive_stays_strictly_inside_bounds() {
    let (meta, data) = golden_model_and_data("interval_censored_normal");
    let settings = PriorPredictiveSettings { num_draws: 4096 };
    let run = simulate_prior_predictive(meta, data, &settings, 127)
        .expect("interval-censored Normal prior predictive succeeds");

    assert_eq!(run.sites[1].name, "y");
    for draw in &run.draws {
        let y = draw.values[1].1.data();
        assert!(y[1] > -1.0 && y[1] < 0.5, "{}", y[1]);
        assert!(y[3] > 0.25 && y[3] < 1.75, "{}", y[3]);
    }
}

#[test]
fn param_prior_predictive_does_not_expand_beyond_free_value_shape() {
    let settings = PriorPredictiveSettings { num_draws: 1 };
    let data = vec![(
        "loc".to_string(),
        DataValue {
            shape: vec![3],
            values: vec![0.0, 1.0, 2.0],
            integer: false,
        },
    )];

    let err = simulate_prior_predictive(scalar_normal_with_vector_loc_model(), data, &settings, 19)
        .unwrap_err();

    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert_eq!(
        err.message,
        "cannot broadcast Normal loc to simulated shape []"
    );
}

#[test]
fn scalar_bernoulli_prior_predictive_emits_integer_json_and_matches_mean() {
    let probs = 0.25;
    let settings = PriorPredictiveSettings { num_draws: 4096 };
    let lines =
        prior_predictive_ndjson_lines(scalar_bernoulli_observed_model(probs), vec![], &settings, 7)
            .expect("prior predictive artifact succeeds");

    assert_eq!(lines.len(), settings.num_draws + 2);

    let header = json::parse(&lines[0]).expect("valid header JSON");
    let site = header
        .get("sites")
        .and_then(Value::as_array)
        .and_then(|sites| sites.first())
        .expect("first site");
    assert_eq!(site.get("name").and_then(Value::as_str), Some("y"));
    assert_eq!(site.get("role").and_then(Value::as_str), Some("observed"));
    assert_eq!(site.get("integer"), Some(&Value::Bool(true)));
    assert_eq!(site.get("integer_by_coordinate"), Some(&Value::Bool(true)));

    let mut successes = 0_i64;
    for line in &lines[1..=settings.num_draws] {
        let draw = json::parse(line).expect("valid draw JSON");
        let y = draw
            .get("values")
            .and_then(|values| values.get("y"))
            .expect("draw contains y");
        match y {
            Value::Int(0) => {}
            Value::Int(1) => successes += 1,
            got => panic!("Bernoulli draw should be integer 0 or 1, got {got:?}"),
        }
    }

    let mean = successes as f64 / settings.num_draws as f64;
    assert!(
        (mean - probs).abs() < 0.04,
        "mean {mean} should be close to {probs}"
    );
}
