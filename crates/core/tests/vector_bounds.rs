use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{
    Constraint, DataSchema, Distribution, Expr, ModelMeta, ResolvedData, ResolvedFreeValue,
    ResolvedStochasticSite, Size,
};
use bayesite_core::model::{DataValue, Posterior};
use bayesite_core::predictive::{
    simulate_data_from_truth, simulate_prior_predictive, PriorPredictiveSettings,
};

fn data(name: &str, values: Vec<f64>) -> (String, DataValue) {
    (
        name.to_string(),
        DataValue {
            shape: vec![values.len()],
            values,
            integer: false,
        },
    )
}

fn vector_model(
    constraint: Constraint,
    size: Size,
    distribution: Distribution,
    data_names: &[&str],
) -> ModelMeta {
    ModelMeta {
        params: vec![],
        data: data_names
            .iter()
            .map(|name| {
                (
                    (*name).to_string(),
                    ResolvedData {
                        schema: DataSchema::Rank(1),
                    },
                )
            })
            .collect(),
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![(
            "y".to_string(),
            ResolvedFreeValue {
                constraint: Some(constraint),
                size,
            },
        )],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "y".to_string(),
            distribution,
            value: Expr::Param("y".to_string()),
        }],
    }
}

fn normal() -> Distribution {
    Distribution::Normal {
        loc: Expr::Const(0.0),
        scale: Expr::Const(1.0),
    }
}

fn exponential() -> Distribution {
    Distribution::Exponential {
        rate: Expr::Const(1.0),
    }
}

fn partially_observed_model(distribution: Distribution, length: i64) -> ModelMeta {
    ModelMeta {
        params: vec![],
        data: ["lower", "missing_idx", "observed_idx", "observed_values"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    ResolvedData {
                        schema: DataSchema::Rank(1),
                    },
                )
            })
            .collect(),
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![(
            "y".to_string(),
            ResolvedFreeValue {
                constraint: Some(Constraint::VectorBounds {
                    lower: Some("lower".to_string()),
                    upper: None,
                }),
                size: Size::Fixed(1),
            },
        )],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "y".to_string(),
            distribution,
            value: Expr::VectorScatter {
                length: Box::new(Expr::Const(length as f64)),
                observed_idx: Box::new(Expr::Data("observed_idx".to_string())),
                observed_values: Box::new(Expr::Data("observed_values".to_string())),
                missing_idx: Box::new(Expr::Data("missing_idx".to_string())),
                missing_values: Box::new(Expr::Param("y".to_string())),
            },
        }],
    }
}

fn partially_observed_data(lower: f64) -> Vec<(String, DataValue)> {
    vec![
        data("lower", vec![lower]),
        data("missing_idx", vec![0.0]),
        data("observed_idx", vec![]),
        data("observed_values", vec![]),
    ]
}

#[test]
fn vector_bounds_reject_wrong_bound_length() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(2),
        normal(),
        &["lower"],
    );
    let err = Posterior::new(model, vec![data("lower", vec![0.0])]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("wrong length"), "{}", err.message);
}

#[test]
fn vector_bounds_reject_nonfinite_bound() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(2),
        normal(),
        &["lower"],
    );
    let err = Posterior::new(model, vec![data("lower", vec![0.0, f64::INFINITY])]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("finite"), "{}", err.message);
}

#[test]
fn vector_bounds_reject_empty_intervals() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: Some("upper".to_string()),
        },
        Size::Fixed(2),
        normal(),
        &["lower", "upper"],
    );
    let err = Posterior::new(
        model,
        vec![data("lower", vec![0.0, 2.0]), data("upper", vec![1.0, 2.0])],
    )
    .unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("lower < upper"), "{}", err.message);
}

#[test]
fn vector_bounds_reject_bounds_outside_exponential_support() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(1),
        exponential(),
        &["lower"],
    );
    let err = Posterior::new(model, vec![data("lower", vec![-0.5])]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(
        err.message.contains("base distribution support"),
        "{}",
        err.message
    );
}

#[test]
fn vector_bounds_require_a_rank_one_free_value() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Scalar,
        normal(),
        &["lower"],
    );
    let err = Posterior::new(model, vec![data("lower", vec![0.0])]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("rank-1 free value"), "{}", err.message);
}

#[test]
fn vector_bounds_reject_a_missing_referenced_data_name() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("not_bound".to_string()),
            upper: None,
        },
        Size::Fixed(1),
        normal(),
        &[],
    );
    let err = Posterior::new(model, vec![]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("not_bound"), "{}", err.message);
    assert!(
        err.message.contains("missing lower data"),
        "{}",
        err.message
    );
}

#[test]
fn exponential_support_folds_an_upper_only_bound_to_two_sided() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: None,
            upper: Some("upper".to_string()),
        },
        Size::Fixed(1),
        exponential(),
        &["upper"],
    );
    let posterior = Posterior::new(model, vec![data("upper", vec![2.0])]).unwrap();

    // At u=0, y=2*sigmoid(0)=1 and log|dy/du|=log(2)-2*softplus(0)=-log(2).
    let (logp, gradient) = posterior.logp_grad(&[0.0]).unwrap();
    let expected = -1.0 - 2.0_f64.ln();
    assert!((logp - expected).abs() < 1e-15, "{logp} != {expected}");
    assert!((gradient[0] + 0.5).abs() < 1e-15, "{:?}", gradient);
}

#[test]
fn normal_lower_only_bound_remains_one_sided() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(1),
        normal(),
        &["lower"],
    );
    let posterior = Posterior::new(model, vec![data("lower", vec![-1.5])]).unwrap();
    let constrained = posterior.constrain(&[0.0]).unwrap();
    assert_eq!(constrained[0].1.data(), &[-0.5]);
}

#[test]
fn unbounded_vector_scatter_draws_the_full_vector_as_observed() {
    let mut model = partially_observed_model(normal(), 2);
    model.free_values[0].1.constraint = None;
    let declared_data = vec![
        data("lower", vec![0.0]),
        data("missing_idx", vec![1.0]),
        data("observed_idx", vec![0.0]),
        data("observed_values", vec![999.0]),
    ];
    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 1 },
        199,
    )
    .expect("unbounded PartiallyObserved simulation succeeds");

    assert_eq!(run.sites.len(), 1);
    assert_eq!(run.sites[0].name, "y");
    assert_eq!(run.sites[0].role.as_str(), "observed");
    let value = &run.draws[0].values[0].1;
    assert_eq!(value.shape(), &[2]);
    assert_ne!(value.data()[0], 999.0, "observed data must not be inserted");
}

#[test]
fn lower_only_exponential_vector_scatter_is_exact_in_the_extreme_tail() {
    let model = partially_observed_model(exponential(), 1);
    let settings = PriorPredictiveSettings { num_draws: 256 };
    let run = simulate_prior_predictive(model, partially_observed_data(500.0), &settings, 211)
        .expect("memorylessness keeps extreme-tail draws finite");

    for draw in run.draws {
        let value = draw.values[0].1.data()[0];
        assert!(value.is_finite(), "{value}");
        assert!(value >= 500.0, "{value}");
    }
}

#[test]
fn two_sided_normal_vector_bounds_are_finite_in_the_extreme_tail() {
    let mut model = partially_observed_model(normal(), 1);
    model.data.push((
        "upper".to_string(),
        ResolvedData {
            schema: DataSchema::Rank(1),
        },
    ));
    model.free_values[0].1.constraint = Some(Constraint::VectorBounds {
        lower: Some("lower".to_string()),
        upper: Some("upper".to_string()),
    });
    let mut declared_data = partially_observed_data(9.0);
    declared_data.push(data("upper", vec![10.0]));
    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 512 },
        219,
    )
    .expect("stable truncated-Normal simulation succeeds in a saturated tail");

    for draw in run.draws {
        let value = draw.values[0].1.data()[0];
        assert!(value.is_finite(), "{value}");
        assert!(value > 9.0 && value < 10.0, "{value}");
    }
}

#[test]
fn vector_scatter_propagates_the_missing_free_value_to_later_sites() {
    let mut model = partially_observed_model(normal(), 3);
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "z_site".to_string(),
        distribution: Distribution::Normal {
            loc: Expr::Param("y".to_string()),
            scale: Expr::Const(1e-9),
        },
        value: Expr::Data("z".to_string()),
    });
    let declared_data = vec![
        data("lower", vec![0.0]),
        data("missing_idx", vec![1.0]),
        data("observed_idx", vec![0.0, 2.0]),
        data("observed_values", vec![-10.0, 10.0]),
    ];
    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 128 },
        221,
    )
    .expect("later ParamRef resolves the missing-coordinate free value");

    assert_eq!(run.sites.len(), 2);
    assert_eq!(run.sites[0].name, "y");
    assert_eq!(run.sites[0].shape, vec![3]);
    assert_eq!(run.sites[1].name, "z");
    assert_eq!(run.sites[1].shape, vec![1]);
    for draw in run.draws {
        let assembled = &draw.values[0].1;
        let consumer = &draw.values[1].1;
        assert_eq!(assembled.shape(), &[3]);
        assert_eq!(consumer.shape(), &[1]);
        assert!(
            (consumer.data()[0] - assembled.data()[1]).abs() < 1e-7,
            "consumer {:?} should be centered on missing value {:?}",
            consumer.data(),
            assembled.data()[1]
        );
    }
}

#[test]
fn bounded_mvn_vector_scatter_is_explicitly_unsupported_by_forward_simulation() {
    let mut model = partially_observed_model(
        Distribution::MultivariateNormal {
            mean: Expr::Data("mean".to_string()),
            scale_tril: Expr::Data("scale_tril".to_string()),
        },
        2,
    );
    model.free_values[0].1.size = Size::Fixed(2);
    model.data.extend([
        (
            "mean".to_string(),
            ResolvedData {
                schema: DataSchema::Rank(1),
            },
        ),
        (
            "scale_tril".to_string(),
            ResolvedData {
                schema: DataSchema::Rank(2),
            },
        ),
    ]);
    let declared_data = vec![
        data("lower", vec![0.0, 0.0]),
        data("missing_idx", vec![0.0, 1.0]),
        data("observed_idx", vec![]),
        data("observed_values", vec![]),
        data("mean", vec![0.0, 0.0]),
        (
            "scale_tril".to_string(),
            DataValue {
                shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
                integer: false,
            },
        ),
    ];
    let message =
        "bounded MVN PartiallyObserved sites are not supported by prior-predictive simulation";

    let prior_err = simulate_prior_predictive(
        model.clone(),
        declared_data.clone(),
        &PriorPredictiveSettings { num_draws: 1 },
        223,
    )
    .unwrap_err();
    assert_eq!(prior_err.message, message);

    let simulate_err =
        simulate_data_from_truth(model, declared_data, vec![data("y", vec![1.0, 1.0])], 227)
            .unwrap_err();
    assert_eq!(simulate_err.message, message);
}

#[test]
fn simulate_truth_checks_resolved_vector_bounds_by_free_value_name() {
    let model = partially_observed_model(normal(), 1);
    let declared_data = partially_observed_data(1.5);

    simulate_data_from_truth(
        model.clone(),
        declared_data.clone(),
        vec![data("y", vec![2.0])],
        229,
    )
    .expect("truth strictly inside bounds succeeds");

    let err = simulate_data_from_truth(model, declared_data, vec![data("y", vec![1.5])], 233)
        .unwrap_err();
    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(err.message.contains("free value \"y\""), "{}", err.message);
    assert!(
        err.message.contains("violates constraint"),
        "{}",
        err.message
    );
}

#[test]
fn direct_vector_bounds_prior_simulation_fails_without_rejection_loop() {
    let model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(1),
        normal(),
        &["lower"],
    );
    let err = simulate_prior_predictive(
        model,
        vec![data("lower", vec![0.0])],
        &PriorPredictiveSettings { num_draws: 1 },
        239,
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("VectorBounds prior simulation is not implemented"),
        "{}",
        err.message
    );
}

#[test]
fn uniform_support_alignment_wraps_negative_missing_indices() {
    let model = ModelMeta {
        params: vec![],
        data: [
            "lower",
            "missing_idx",
            "observed_idx",
            "observed_values",
            "low",
            "high",
        ]
        .into_iter()
        .map(|name| {
            (
                name.to_string(),
                ResolvedData {
                    schema: DataSchema::Rank(1),
                },
            )
        })
        .collect(),
        observed_nodes: vec![],
        expressions: vec![],
        free_values: vec![(
            "y".to_string(),
            ResolvedFreeValue {
                constraint: Some(Constraint::VectorBounds {
                    lower: Some("lower".to_string()),
                    upper: None,
                }),
                size: Size::Fixed(1),
            },
        )],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "y".to_string(),
            distribution: Distribution::Uniform {
                low: Expr::Data("low".to_string()),
                high: Expr::Data("high".to_string()),
            },
            value: Expr::VectorScatter {
                length: Box::new(Expr::Const(2.0)),
                observed_idx: Box::new(Expr::Data("observed_idx".to_string())),
                observed_values: Box::new(Expr::Data("observed_values".to_string())),
                missing_idx: Box::new(Expr::Data("missing_idx".to_string())),
                missing_values: Box::new(Expr::Param("y".to_string())),
            },
        }],
    };
    let declared_data = vec![
        data("lower", vec![0.5]),
        // -1 wraps to coordinate 1, matching the scatter evaluators.
        data("missing_idx", vec![-1.0]),
        data("observed_idx", vec![0.0]),
        data("observed_values", vec![0.6]),
        data("low", vec![0.0, 0.25]),
        data("high", vec![1.0, 2.25]),
    ];
    let posterior = Posterior::new(model, declared_data).unwrap();

    // The missing upper side folds to high[1] = 2.25, so u = 0 lands at the
    // interval midpoint of [0.5, 2.25].
    let constrained = posterior.constrain(&[0.0]).unwrap();
    assert_eq!(constrained[0].1.data(), &[1.375]);
}

#[test]
fn constrained_scatter_free_values_are_explicitly_unsupported_in_prior_draws() {
    let mut model = partially_observed_model(normal(), 2);
    model.free_values[0].1.constraint = Some(Constraint::Positive);
    let declared_data = vec![
        data("lower", vec![0.0]),
        data("missing_idx", vec![1.0]),
        data("observed_idx", vec![0.0]),
        data("observed_values", vec![0.5]),
    ];
    let err = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 1 },
        241,
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("constrained missing_values free value"),
        "{}",
        err.message
    );
}
