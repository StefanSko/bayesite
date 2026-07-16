use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{
    BinOpKind, Constraint, DataSchema, Distribution, Expr, ModelMeta, ResolvedData,
    ResolvedFreeValue, ResolvedObserved, ResolvedStochasticSite, Size,
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

fn partially_observed_ancestor_model(reversed_sites: bool) -> ModelMeta {
    let mut model = partially_observed_model(normal(), 3);
    let distribution = Distribution::Normal {
        loc: model.stochastic_sites[0].value.clone(),
        scale: Expr::Const(1e-9),
    };
    model.observed_nodes.push(ResolvedObserved {
        name: "z".to_string(),
        distribution: distribution.clone(),
    });
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "z_site".to_string(),
        distribution,
        value: Expr::Data("z".to_string()),
    });
    if reversed_sites {
        model.stochastic_sites.reverse();
    }
    model
}

fn partially_observed_ancestor_data() -> Vec<(String, DataValue)> {
    vec![
        data("lower", vec![0.0]),
        data("missing_idx", vec![1.0]),
        data("observed_idx", vec![0.0, 2.0]),
        data("observed_values", vec![-1_000.0, 1_000.0]),
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
fn vector_bounds_support_comes_from_same_name_owner_not_earlier_factor() {
    let mut model = partially_observed_model(exponential(), 1);
    model.free_values[0].1.constraint = Some(Constraint::VectorBounds {
        lower: None,
        upper: Some("lower".to_string()),
    });
    let owner = model.stochastic_sites[0].clone();
    model.stochastic_sites.insert(
        0,
        ResolvedStochasticSite {
            name: "penalty".to_string(),
            distribution: normal(),
            value: owner.value.clone(),
        },
    );

    let posterior = Posterior::new(model, partially_observed_data(1.0)).unwrap();
    let (logp, gradient) = posterior.logp_grad(&[0.1]).unwrap();

    assert!(logp.is_finite(), "{logp}");
    assert!(gradient[0].is_finite(), "{:?}", gradient);
}

#[test]
fn prior_predictive_rejects_non_owner_vector_scatter_factor() {
    let mut model = partially_observed_model(exponential(), 1);
    let owner = model.stochastic_sites[0].clone();
    model.stochastic_sites.insert(
        0,
        ResolvedStochasticSite {
            name: "penalty".to_string(),
            distribution: normal(),
            value: owner.value.clone(),
        },
    );

    let err = simulate_prior_predictive(
        model,
        partially_observed_data(1.0),
        &PriorPredictiveSettings { num_draws: 1 },
        29,
    )
    .unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(
        err.message.contains("additional stochastic factor"),
        "{}",
        err.message
    );
}

#[test]
fn prior_predictive_rejects_non_owner_direct_parameter_factor() {
    let mut model = vector_model(
        Constraint::VectorBounds {
            lower: Some("lower".to_string()),
            upper: None,
        },
        Size::Fixed(1),
        exponential(),
        &["lower"],
    );
    model.stochastic_sites.insert(
        0,
        ResolvedStochasticSite {
            name: "penalty".to_string(),
            distribution: normal(),
            value: Expr::Param("y".to_string()),
        },
    );

    let err = simulate_prior_predictive(
        model,
        vec![data("lower", vec![1.0])],
        &PriorPredictiveSettings { num_draws: 1 },
        31,
    )
    .unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(
        err.message.contains("additional stochastic factor"),
        "{}",
        err.message
    );
}

#[test]
fn vector_bounds_reject_missing_same_name_owner() {
    let mut model = partially_observed_model(exponential(), 1);
    model.stochastic_sites[0].name = "renamed_y".to_string();

    let err = Posterior::new(model, partially_observed_data(1.0)).unwrap_err();

    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(
        err.message.contains("exactly one same-name owner"),
        "{}",
        err.message
    );
}

#[test]
fn vector_bounds_reject_duplicate_same_name_owners() {
    let mut model = partially_observed_model(exponential(), 1);
    model
        .stochastic_sites
        .push(model.stochastic_sites[0].clone());

    let err = Posterior::new(model, partially_observed_data(1.0)).unwrap_err();

    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(
        err.message.contains("exactly one same-name owner"),
        "{}",
        err.message
    );
}

#[test]
fn vector_bounds_reject_malformed_same_name_owner() {
    let mut model = partially_observed_model(exponential(), 1);
    model.stochastic_sites[0].value = Expr::Bin {
        op: BinOpKind::Add,
        left: Box::new(Expr::Param("y".to_string())),
        right: Box::new(Expr::Const(0.0)),
    };

    let err = Posterior::new(model, partially_observed_data(1.0)).unwrap_err();

    assert_eq!(err.kind, ErrorKind::DataShapeMismatch);
    assert!(
        err.message.contains("must evaluate directly"),
        "{}",
        err.message
    );
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
    let distribution = Distribution::Normal {
        loc: Expr::Param("y".to_string()),
        scale: Expr::Const(1e-9),
    };
    model.observed_nodes.push(ResolvedObserved {
        name: "z".to_string(),
        distribution: distribution.clone(),
    });
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "z_site".to_string(),
        distribution,
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
fn partial_owner_value_dependencies_are_scheduled_ancestrally() {
    let mut model = partially_observed_model(normal(), 1);
    let Expr::VectorScatter { length, .. } = &mut model.stochastic_sites[0].value else {
        panic!("partial owner must be a scatter")
    };
    *length = Box::new(Expr::Data("n".to_string()));
    let n_distribution = Distribution::Bernoulli {
        probs: Expr::Const(1.0),
    };
    model.observed_nodes.push(ResolvedObserved {
        name: "n".to_string(),
        distribution: n_distribution.clone(),
    });
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "n_site".to_string(),
        distribution: n_distribution,
        value: Expr::Data("n".to_string()),
    });
    let run = simulate_prior_predictive(
        model,
        partially_observed_data(0.0),
        &PriorPredictiveSettings { num_draws: 1 },
        217,
    )
    .expect("generated partial length is scheduled before its owner");

    assert_eq!(
        run.sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["y", "n"]
    );
    assert_eq!(run.draws[0].values[0].1.shape(), &[1]);
}

#[test]
fn partial_owner_value_dependency_cycles_fail_before_drawing() {
    let mut model = partially_observed_model(normal(), 1);
    let Expr::VectorScatter { length, .. } = &mut model.stochastic_sites[0].value else {
        panic!("partial owner must be a scatter")
    };
    *length = Box::new(Expr::Data("n".to_string()));
    let n_distribution = Distribution::Bernoulli {
        probs: Expr::Param("y".to_string()),
    };
    model.observed_nodes.push(ResolvedObserved {
        name: "n".to_string(),
        distribution: n_distribution.clone(),
    });
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "n_site".to_string(),
        distribution: n_distribution,
        value: Expr::Data("n".to_string()),
    });

    let err = simulate_prior_predictive(
        model,
        partially_observed_data(0.0),
        &PriorPredictiveSettings { num_draws: 1 },
        218,
    )
    .unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(err.message.contains("cyclic or unavailable"));
    assert!(err.message.contains("n"));
    assert!(err.message.contains("y"));
}

#[test]
fn generated_partial_ancestor_propagates_its_full_vector() {
    let declared_data = partially_observed_ancestor_data();
    let run = simulate_prior_predictive(
        partially_observed_ancestor_model(false),
        declared_data.clone(),
        &PriorPredictiveSettings { num_draws: 4 },
        223,
    )
    .expect("PartiallyObserved ancestor simulation succeeds");

    for draw in run.draws {
        let ancestor = &draw.values[0].1;
        let child = &draw.values[1].1;
        assert_eq!(ancestor.shape(), &[3]);
        assert_eq!(child.shape(), &[3]);
        assert_ne!(ancestor.data()[0], -1_000.0);
        assert_ne!(ancestor.data()[2], 1_000.0);
        for (ancestor, child) in ancestor.data().iter().zip(child.data()) {
            assert!(
                (ancestor - child).abs() < 1e-7,
                "child {child} should consume full generated ancestor {ancestor}"
            );
        }
    }
    assert_eq!(
        declared_data
            .iter()
            .find(|(name, _)| name == "observed_values")
            .expect("conditioning data")
            .1
            .values,
        [-1_000.0, 1_000.0]
    );
}

#[test]
fn distinct_scatter_with_same_missing_slot_keeps_its_own_observed_values() {
    let mut model = partially_observed_ancestor_model(false);
    model.data.push((
        "alternate_observed_values".to_string(),
        ResolvedData {
            schema: DataSchema::Rank(1),
        },
    ));
    let alternate_scatter = Expr::VectorScatter {
        length: Box::new(Expr::Const(3.0)),
        observed_idx: Box::new(Expr::Data("observed_idx".to_string())),
        observed_values: Box::new(Expr::Data("alternate_observed_values".to_string())),
        missing_idx: Box::new(Expr::Data("missing_idx".to_string())),
        missing_values: Box::new(Expr::Param("y".to_string())),
    };
    let child_distribution = Distribution::Normal {
        loc: alternate_scatter,
        scale: Expr::Const(1e-9),
    };
    model.observed_nodes[0].distribution = child_distribution.clone();
    model.stochastic_sites[1].distribution = child_distribution;
    let mut declared_data = partially_observed_ancestor_data();
    declared_data.push(data("alternate_observed_values", vec![-100.0, 100.0]));
    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 1 },
        224,
    )
    .expect("distinct descendant scatter simulates");

    let ancestor = &run.draws[0].values[0].1;
    let child = &run.draws[0].values[1].1;
    assert!((child.data()[0] + 100.0).abs() < 1e-7, "{:?}", child.data());
    assert!((child.data()[2] - 100.0).abs() < 1e-7, "{:?}", child.data());
    assert!((child.data()[1] - ancestor.data()[1]).abs() < 1e-7);
}

#[test]
fn generated_partial_ancestors_sharing_conditioning_data_remain_isolated() {
    let scatter = |name: &str| Expr::VectorScatter {
        length: Box::new(Expr::Const(2.0)),
        observed_idx: Box::new(Expr::Data("shared_observed_idx".to_string())),
        observed_values: Box::new(Expr::Data("shared_observed_values".to_string())),
        missing_idx: Box::new(Expr::Data("shared_missing_idx".to_string())),
        missing_values: Box::new(Expr::Param(name.to_string())),
    };
    let first_distribution = Distribution::Normal {
        loc: Expr::Const(-20.0),
        scale: Expr::Const(0.01),
    };
    let second_distribution = Distribution::Normal {
        loc: Expr::Const(20.0),
        scale: Expr::Const(0.01),
    };
    let first_child_distribution = Distribution::Normal {
        loc: scatter("first"),
        scale: Expr::Const(1e-9),
    };
    let second_child_distribution = Distribution::Normal {
        loc: scatter("second"),
        scale: Expr::Const(1e-9),
    };
    let model = ModelMeta {
        params: vec![],
        data: [
            "shared_observed_idx",
            "shared_observed_values",
            "shared_missing_idx",
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
        observed_nodes: vec![
            ResolvedObserved {
                name: "first_child".to_string(),
                distribution: first_child_distribution.clone(),
            },
            ResolvedObserved {
                name: "second_child".to_string(),
                distribution: second_child_distribution.clone(),
            },
        ],
        expressions: vec![],
        free_values: ["first", "second"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    ResolvedFreeValue {
                        constraint: None,
                        size: Size::Fixed(1),
                    },
                )
            })
            .collect(),
        stochastic_sites: vec![
            ResolvedStochasticSite {
                name: "first".to_string(),
                distribution: first_distribution,
                value: scatter("first"),
            },
            ResolvedStochasticSite {
                name: "second".to_string(),
                distribution: second_distribution,
                value: scatter("second"),
            },
            ResolvedStochasticSite {
                name: "first_child_site".to_string(),
                distribution: first_child_distribution,
                value: Expr::Data("first_child".to_string()),
            },
            ResolvedStochasticSite {
                name: "second_child_site".to_string(),
                distribution: second_child_distribution,
                value: Expr::Data("second_child".to_string()),
            },
        ],
    };
    let declared_data = vec![
        data("shared_observed_idx", vec![0.0]),
        data("shared_observed_values", vec![100.0]),
        data("shared_missing_idx", vec![1.0]),
    ];
    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 2 },
        225,
    )
    .expect("shared partial ancestors simulate independently");

    for draw in run.draws {
        let first = &draw.values[0].1;
        let second = &draw.values[1].1;
        let first_child = &draw.values[2].1;
        let second_child = &draw.values[3].1;
        assert!(first.data()[0] < -10.0, "{:?}", first.data());
        assert!(second.data()[0] > 10.0, "{:?}", second.data());
        for (ancestor, child) in first.data().iter().zip(first_child.data()) {
            assert!((ancestor - child).abs() < 1e-7);
        }
        for (ancestor, child) in second.data().iter().zip(second_child.data()) {
            assert!((ancestor - child).abs() < 1e-7);
        }
    }
}

#[test]
fn reversed_partial_ancestor_is_scheduled_but_emitted_in_metadata_order() {
    let run = simulate_prior_predictive(
        partially_observed_ancestor_model(true),
        partially_observed_ancestor_data(),
        &PriorPredictiveSettings { num_draws: 1 },
        227,
    )
    .expect("PartiallyObserved dependency order is derived structurally");

    assert_eq!(
        run.sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["z", "y"]
    );
    let child = &run.draws[0].values[0].1;
    let ancestor = &run.draws[0].values[1].1;
    for (ancestor, child) in ancestor.data().iter().zip(child.data()) {
        assert!((ancestor - child).abs() < 1e-7);
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

#[test]
fn recover_rejects_partially_observed_free_values_before_drawing() {
    use bayesite_core::sampler::Settings;
    use bayesite_core::workflow::{recover_report, RecoverSettings};

    let model = partially_observed_model(exponential(), 1);
    let settings = RecoverSettings {
        chains: 1,
        sampler: Settings {
            num_warmup: 10,
            num_draws: 10,
            ..Settings::default()
        },
        interval: 0.8,
    };
    let err = recover_report(model, partially_observed_data(1.0), &settings, 23).unwrap_err();
    assert!(
        err.message
            .contains("cannot report truth for free value \"y\""),
        "{}",
        err.message
    );
    assert!(
        err.message
            .contains("directly simulated stochastic site for every free value"),
        "{}",
        err.message
    );
}

#[test]
fn scatter_free_value_shape_must_match_missing_idx_length_in_prior_draws() {
    let mut model = partially_observed_model(normal(), 3);
    model.free_values[0].1.constraint = None;
    model.free_values[0].1.size = Size::Fixed(2);
    let declared_data = vec![
        data("lower", vec![0.0]),
        data("missing_idx", vec![1.0]),
        data("observed_idx", vec![0.0, 2.0]),
        data("observed_values", vec![0.1, 0.2]),
    ];
    let err = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 1 },
        251,
    )
    .unwrap_err();
    assert!(
        err.message
            .contains("scatter values must match their index vectors in length"),
        "{}",
        err.message
    );
    assert!(err.message.contains("shape [2]"), "{}", err.message);
}

#[test]
fn uniform_support_folding_defers_for_data_generated_by_earlier_sites() {
    let mut model = partially_observed_model(
        Distribution::Uniform {
            low: Expr::Data("a".to_string()),
            high: Expr::Const(4.0),
        },
        1,
    );
    let distribution = Distribution::Uniform {
        low: Expr::Const(1.0),
        high: Expr::Const(2.0),
    };
    model.observed_nodes.push(ResolvedObserved {
        name: "a".to_string(),
        distribution: distribution.clone(),
    });
    model.stochastic_sites.insert(
        0,
        ResolvedStochasticSite {
            name: "a".to_string(),
            distribution,
            value: Expr::Data("a".to_string()),
        },
    );
    let run = simulate_prior_predictive(
        model,
        partially_observed_data(2.5),
        &PriorPredictiveSettings { num_draws: 64 },
        263,
    )
    .expect("support folding defers for the generated Uniform low bound");

    for draw in run.draws {
        let y = draw
            .values
            .iter()
            .find(|(name, _)| name == "y")
            .expect("scatter site draw present")
            .1
            .data()[0];
        assert!((2.5..4.0).contains(&y), "{y}");
    }
}
