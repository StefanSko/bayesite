use std::collections::HashMap;

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{
    decode_model, BinOpKind, DataSchema, Distribution, Expr, IndexSpec, ModelMeta, ResolvedData,
    ResolvedFreeValue, ResolvedObserved, ResolvedParam, ResolvedStochasticSite, Size, UnaryFn,
};
use bayesite_core::json;
use bayesite_core::model::{data_from_json, DataValue, Posterior};
use bayesite_core::predictive::{
    simulate_data_from_truth, simulate_posterior_predictive, simulate_prior_predictive,
    PriorPredictiveRun, PriorPredictiveSettings,
};
use bayesite_core::protocol::ndjson_lines;
use bayesite_core::sampler::{sample, Settings};

fn normal(loc: Expr, scale: f64) -> Distribution {
    Distribution::Normal {
        loc,
        scale: Expr::Const(scale),
    }
}

fn hierarchical_model(reversed_sites: bool) -> ModelMeta {
    let location_distribution = normal(Expr::Const(0.0), 1.0);
    let coefficient_distribution = normal(Expr::Param("location".to_string()), 0.2);
    let outcome_distribution = normal(Expr::Param("coefficient".to_string()), 0.1);
    let mut sites = vec![
        ResolvedStochasticSite {
            name: "location_site".to_string(),
            distribution: location_distribution.clone(),
            value: Expr::Param("location".to_string()),
        },
        ResolvedStochasticSite {
            name: "coefficient_site".to_string(),
            distribution: coefficient_distribution.clone(),
            value: Expr::Param("coefficient".to_string()),
        },
        ResolvedStochasticSite {
            name: "outcome_site".to_string(),
            distribution: outcome_distribution.clone(),
            value: Expr::Data("outcome".to_string()),
        },
    ];
    if reversed_sites {
        sites.swap(0, 1);
    }
    ModelMeta {
        params: vec![
            (
                "location".to_string(),
                ResolvedParam {
                    distribution: location_distribution,
                    constraint: None,
                    size: Size::Scalar,
                },
            ),
            (
                "coefficient".to_string(),
                ResolvedParam {
                    distribution: coefficient_distribution,
                    constraint: None,
                    size: Size::Scalar,
                },
            ),
        ],
        data: vec![],
        observed_nodes: vec![ResolvedObserved {
            name: "outcome".to_string(),
            distribution: outcome_distribution,
        }],
        expressions: vec![],
        free_values: vec![
            (
                "location".to_string(),
                ResolvedFreeValue {
                    constraint: None,
                    size: Size::Scalar,
                },
            ),
            (
                "coefficient".to_string(),
                ResolvedFreeValue {
                    constraint: None,
                    size: Size::Scalar,
                },
            ),
        ],
        stochastic_sites: sites,
    }
}

fn outcome_chain_model(reversed_sites: bool) -> ModelMeta {
    let parent_distribution = normal(Expr::Const(0.0), 1.0);
    let child_distribution = normal(Expr::Data("parent".to_string()), 1e-9);
    let mut sites = vec![
        ResolvedStochasticSite {
            name: "parent_site".to_string(),
            distribution: parent_distribution.clone(),
            value: Expr::Data("parent".to_string()),
        },
        ResolvedStochasticSite {
            name: "child_site".to_string(),
            distribution: child_distribution.clone(),
            value: Expr::Data("child".to_string()),
        },
    ];
    if reversed_sites {
        sites.reverse();
    }
    ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![
            ResolvedObserved {
                name: "parent".to_string(),
                distribution: parent_distribution,
            },
            ResolvedObserved {
                name: "child".to_string(),
                distribution: child_distribution,
            },
        ],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: sites,
    }
}

fn forked_outcome_model(order: &[&str]) -> ModelMeta {
    let parent_distribution = normal(Expr::Const(0.0), 1.0);
    let independent_distribution = normal(Expr::Const(10.0), 1.0);
    let child_distribution = normal(Expr::Data("parent".to_string()), 0.1);
    let sibling_distribution = normal(Expr::Data("parent".to_string()), 0.2);
    let definitions = [
        ("parent", parent_distribution.clone()),
        ("independent", independent_distribution.clone()),
        ("child", child_distribution.clone()),
        ("sibling", sibling_distribution.clone()),
    ];
    let sites = order
        .iter()
        .map(|name| {
            let distribution = definitions
                .iter()
                .find(|(candidate, _)| candidate == name)
                .expect("known outcome")
                .1
                .clone();
            ResolvedStochasticSite {
                name: format!("{name}_site"),
                distribution,
                value: Expr::Data((*name).to_string()),
            }
        })
        .collect();
    ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: definitions
            .into_iter()
            .map(|(name, distribution)| ResolvedObserved {
                name: name.to_string(),
                distribution,
            })
            .collect(),
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: sites,
    }
}

fn posterior_outcome_chain_model(reversed_outcomes: bool) -> ModelMeta {
    let parameter_distribution = normal(Expr::Const(0.0), 1.0);
    let parent_distribution = normal(Expr::Param("theta".to_string()), 0.5);
    let child_distribution = normal(Expr::Data("parent".to_string()), 1e-9);
    let parameter_site = ResolvedStochasticSite {
        name: "theta_site".to_string(),
        distribution: parameter_distribution.clone(),
        value: Expr::Param("theta".to_string()),
    };
    let parent_site = ResolvedStochasticSite {
        name: "parent_site".to_string(),
        distribution: parent_distribution.clone(),
        value: Expr::Data("parent".to_string()),
    };
    let child_site = ResolvedStochasticSite {
        name: "child_site".to_string(),
        distribution: child_distribution.clone(),
        value: Expr::Data("child".to_string()),
    };
    let outcome_sites = if reversed_outcomes {
        vec![child_site, parent_site]
    } else {
        vec![parent_site, child_site]
    };
    ModelMeta {
        params: vec![(
            "theta".to_string(),
            ResolvedParam {
                distribution: parameter_distribution,
                constraint: None,
                size: Size::Scalar,
            },
        )],
        data: vec![],
        observed_nodes: vec![
            ResolvedObserved {
                name: "parent".to_string(),
                distribution: parent_distribution,
            },
            ResolvedObserved {
                name: "child".to_string(),
                distribution: child_distribution,
            },
        ],
        expressions: vec![],
        free_values: vec![(
            "theta".to_string(),
            ResolvedFreeValue {
                constraint: None,
                size: Size::Scalar,
            },
        )],
        stochastic_sites: std::iter::once(parameter_site)
            .chain(outcome_sites)
            .collect(),
    }
}

fn scalar_data(name: &str, value: f64) -> (String, DataValue) {
    (
        name.to_string(),
        DataValue {
            shape: vec![],
            values: vec![value],
            integer: false,
        },
    )
}

fn vector_data(name: &str, values: Vec<f64>) -> (String, DataValue) {
    (
        name.to_string(),
        DataValue {
            shape: vec![values.len()],
            values,
            integer: false,
        },
    )
}

fn short_fit(meta: &ModelMeta, data: &[(String, DataValue)]) -> String {
    let posterior = Posterior::new(meta.clone(), data.to_vec()).expect("posterior binds");
    let settings = Settings {
        num_warmup: 2,
        num_draws: 4,
        max_treedepth: 3,
        target_accept: 0.8,
        initial_step_size: 0.25,
    };
    let chain = sample(&posterior, &settings, 431, 0).expect("short fit samples");
    ndjson_lines(&posterior, &settings, 431, &[(0, chain)])
        .expect("fit renders")
        .join("\n")
}

fn named_draw(run: &PriorPredictiveRun) -> HashMap<String, Vec<f64>> {
    run.draws[0]
        .values
        .iter()
        .map(|(name, value)| (name.clone(), value.data().to_vec()))
        .collect()
}

#[test]
fn vendored_composed_fixture_executes_with_non_ancestral_factor_order() {
    let root = format!("{}/../../tests/golden_ir", env!("CARGO_MANIFEST_DIR"));
    let model_document = json::parse(
        &std::fs::read_to_string(format!("{root}/alternative_prior_regression.json"))
            .expect("composed fixture readable"),
    )
    .expect("composed fixture parses");
    let mut model = decode_model(&model_document).expect("composed fixture decodes");
    let outcome_index = model
        .stochastic_sites
        .iter()
        .position(|site| site.name == "y")
        .expect("composed fixture outcome");
    let outcome = model.stochastic_sites.remove(outcome_index);
    model.stochastic_sites.insert(0, outcome);
    let data_document = json::parse(
        &std::fs::read_to_string(format!("{root}/data/alternative_prior_regression.json"))
            .expect("composed fixture data readable"),
    )
    .expect("composed fixture data parses");
    let declared_data = data_from_json(&data_document)
        .expect("composed fixture data decodes")
        .into_iter()
        .filter(|(name, _)| name == "x")
        .collect();

    let run = simulate_prior_predictive(
        model,
        declared_data,
        &PriorPredictiveSettings { num_draws: 2 },
        397,
    )
    .expect("vendored composed fixture executes independently of factor order");

    assert_eq!(
        run.sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["y", "alpha", "beta", "sigma"]
    );
    assert_eq!(run.draws.len(), 2);
}

#[test]
fn prior_predictive_schedules_reversed_parameter_sites_without_reordering_artifacts() {
    let settings = PriorPredictiveSettings { num_draws: 1 };
    let ordered = simulate_prior_predictive(hierarchical_model(false), vec![], &settings, 401)
        .expect("ancestrally ordered model simulates");
    let baseline = named_draw(&ordered);
    assert_eq!(baseline["location"], [-0.41024234441523333]);
    assert_eq!(baseline["coefficient"], [-0.4599393026705723]);
    assert_eq!(baseline["outcome"], [-0.484164834106182]);
    let reversed = simulate_prior_predictive(hierarchical_model(true), vec![], &settings, 401)
        .expect("factor metadata order does not determine draw order");

    assert_eq!(named_draw(&reversed), baseline);
    assert_eq!(
        reversed
            .sites
            .iter()
            .map(|site| site.stochastic_site.as_str())
            .collect::<Vec<_>>(),
        ["coefficient_site", "location_site", "outcome_site"]
    );
    assert_eq!(
        reversed.draws[0]
            .values
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["coefficient", "location", "outcome"]
    );
}

#[test]
fn prior_predictive_schedules_generated_data_dependencies() {
    let run = simulate_prior_predictive(
        outcome_chain_model(true),
        vec![],
        &PriorPredictiveSettings { num_draws: 1 },
        403,
    )
    .expect("generated DataRef dependencies are scheduled ancestrally");

    assert_eq!(
        run.sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["child", "parent"]
    );
    let values = named_draw(&run);
    assert!((values["child"][0] - values["parent"][0]).abs() < 1e-7);
}

#[test]
fn ready_site_ties_keep_original_metadata_order() {
    let settings = PriorPredictiveSettings { num_draws: 1 };
    let expected = simulate_prior_predictive(
        forked_outcome_model(&["independent", "parent", "child", "sibling"]),
        vec![],
        &settings,
        407,
    )
    .expect("explicit stable order simulates");
    let forked = simulate_prior_predictive(
        forked_outcome_model(&["child", "independent", "parent", "sibling"]),
        vec![],
        &settings,
        407,
    )
    .expect("ready ties are resolved by original site index");

    assert_eq!(named_draw(&forked), named_draw(&expected));
    assert_eq!(
        forked
            .sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["child", "independent", "parent", "sibling"]
    );
}

#[test]
fn simulate_schedules_generated_data_but_emits_stochastic_site_order() {
    let output = simulate_data_from_truth(outcome_chain_model(true), vec![], vec![], 409)
        .expect("simulate schedules generated outcomes ancestrally");

    assert_eq!(
        output
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["child", "parent"]
    );
    let values = output
        .iter()
        .map(|(name, value)| (name.as_str(), value.values[0]))
        .collect::<HashMap<_, _>>();
    assert!((values["child"] - values["parent"]).abs() < 1e-7);
}

#[test]
fn simulate_does_not_generate_unassociated_dataref_factors() {
    let mut model = outcome_chain_model(false);
    model.stochastic_sites.push(ResolvedStochasticSite {
        name: "penalty".to_string(),
        distribution: normal(Expr::Const(0.0), 1.0),
        value: Expr::Data("invented".to_string()),
    });

    let output = simulate_data_from_truth(model, vec![], vec![], 411)
        .expect("fixed-truth simulation ignores non-observed density factors");

    assert_eq!(
        output
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["parent", "child"]
    );
}

#[test]
fn posterior_predictive_schedules_observed_outcomes_in_metadata_order() {
    let model = posterior_outcome_chain_model(true);
    let data = vec![scalar_data("parent", 0.25), scalar_data("child", 0.25)];
    let fit = short_fit(&model, &data);
    let run = simulate_posterior_predictive(model, data, &fit, 433)
        .expect("posterior predictive schedules outcomes ancestrally");

    assert_eq!(
        run.sites
            .iter()
            .map(|site| site.name.as_str())
            .collect::<Vec<_>>(),
        ["child", "parent"]
    );
    for draw in run.draws {
        let values = draw
            .values
            .iter()
            .map(|(name, value)| (name.as_str(), value.data()[0]))
            .collect::<HashMap<_, _>>();
        assert!((values["child"] - values["parent"]).abs() < 1e-7);
    }
}

#[test]
fn posterior_predictive_requires_the_observed_declaration_distribution() {
    let mut model = posterior_outcome_chain_model(false);
    let data = vec![scalar_data("parent", 0.25), scalar_data("child", 0.25)];
    model.stochastic_sites[1].distribution = normal(Expr::Const(0.0), 2.0);
    let fit = short_fit(&model, &data);

    let err = simulate_posterior_predictive(model, data, &fit, 439).unwrap_err();

    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert!(err.message.contains("matching directly assignable"));
}

#[test]
fn nested_distribution_references_participate_in_scheduling() {
    let parent_distribution = normal(Expr::Data("design".to_string()), 1.0);
    let indexed_parent = Expr::Index {
        base: Box::new(Expr::Data("parent".to_string())),
        index: IndexSpec::Scalar(Box::new(Expr::Const(0.0))),
    };
    let child_distribution = Distribution::Truncated {
        base: Box::new(Distribution::Normal {
            loc: Expr::Unary {
                function: UnaryFn::Neg,
                operand: Box::new(Expr::Bin {
                    op: BinOpKind::Add,
                    left: Box::new(indexed_parent.clone()),
                    right: Box::new(Expr::Const(0.0)),
                }),
            },
            scale: Expr::Const(1.0),
        }),
        lower: Some(Expr::Bin {
            op: BinOpKind::Sub,
            left: Box::new(indexed_parent),
            right: Box::new(Expr::Const(100.0)),
        }),
        upper: None,
    };
    let model = ModelMeta {
        params: vec![],
        data: vec![(
            "design".to_string(),
            ResolvedData {
                schema: DataSchema::Rank(1),
            },
        )],
        observed_nodes: vec![
            ResolvedObserved {
                name: "parent".to_string(),
                distribution: parent_distribution.clone(),
            },
            ResolvedObserved {
                name: "child".to_string(),
                distribution: child_distribution.clone(),
            },
        ],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![
            ResolvedStochasticSite {
                name: "child_site".to_string(),
                distribution: child_distribution,
                value: Expr::Data("child".to_string()),
            },
            ResolvedStochasticSite {
                name: "parent_site".to_string(),
                distribution: parent_distribution,
                value: Expr::Data("parent".to_string()),
            },
        ],
    };

    let run = simulate_prior_predictive(
        model,
        vec![vector_data("design", vec![0.0, 1.0])],
        &PriorPredictiveSettings { num_draws: 1 },
        443,
    )
    .expect("nested generated references are scheduled before evaluation");

    assert_eq!(run.sites[0].name, "child");
    assert_eq!(run.sites[1].name, "parent");
}

#[test]
fn prior_predictive_reports_unavailable_param_and_data_references() {
    let data_distribution = normal(Expr::Data("missing_data".to_string()), 1.0);
    let data_model = ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![ResolvedObserved {
            name: "outcome".to_string(),
            distribution: data_distribution.clone(),
        }],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![ResolvedStochasticSite {
            name: "outcome_site".to_string(),
            distribution: data_distribution,
            value: Expr::Data("outcome".to_string()),
        }],
    };
    let param_distribution = normal(Expr::Param("missing_param".to_string()), 1.0);
    let param_model = ModelMeta {
        params: vec![(
            "theta".to_string(),
            ResolvedParam {
                distribution: param_distribution.clone(),
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
            name: "theta_site".to_string(),
            distribution: param_distribution,
            value: Expr::Param("theta".to_string()),
        }],
    };

    for (model, missing) in [(data_model, "missing_data"), (param_model, "missing_param")] {
        let err = simulate_prior_predictive(
            model,
            vec![],
            &PriorPredictiveSettings { num_draws: 1 },
            449,
        )
        .unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidSettings);
        assert!(err.message.contains("cyclic or unavailable"));
        assert!(err.message.contains(missing), "{}", err.message);
    }
}

#[test]
fn prior_predictive_reports_cycles_before_drawing() {
    let first_distribution = normal(Expr::Data("second".to_string()), 1.0);
    let second_distribution = normal(Expr::Data("first".to_string()), 1.0);
    let model = ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![
            ResolvedObserved {
                name: "first".to_string(),
                distribution: first_distribution.clone(),
            },
            ResolvedObserved {
                name: "second".to_string(),
                distribution: second_distribution.clone(),
            },
        ],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![
            ResolvedStochasticSite {
                name: "first_site".to_string(),
                distribution: first_distribution,
                value: Expr::Data("first".to_string()),
            },
            ResolvedStochasticSite {
                name: "second_site".to_string(),
                distribution: second_distribution,
                value: Expr::Data("second".to_string()),
            },
        ],
    };
    let settings = PriorPredictiveSettings { num_draws: 1 };
    let first = simulate_prior_predictive(model.clone(), vec![], &settings, 419).unwrap_err();
    let second = simulate_prior_predictive(model, vec![], &settings, 421).unwrap_err();

    assert_eq!(first, second);
    assert_eq!(first.kind, ErrorKind::InvalidSettings);
    assert!(first.message.contains("cyclic or unavailable"));
    for expected in ["first", "second"] {
        assert!(first.message.contains(expected), "{}", first.message);
    }
}
