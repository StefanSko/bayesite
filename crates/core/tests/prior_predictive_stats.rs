//! Self-contained prior-predictive analytic checks.

use bayesite_core::ir::{
    Distribution, Expr, ModelMeta, ResolvedFreeValue, ResolvedParam, ResolvedStochasticSite, Size,
};
use bayesite_core::json::{self, Value};
use bayesite_core::predictive::{
    prior_predictive_ndjson_lines, simulate_prior_predictive, PriorPredictiveSettings,
};

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
    ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "y".to_string(),
            distribution: Distribution::Bernoulli {
                probs: Expr::Const(probs),
            },
            value: Expr::Data("y".to_string()),
        }],
    }
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
