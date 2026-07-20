use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::ir::{
    decode_model, Constraint, DataSchema, Distribution, Expr, ModelMeta, ResolvedData,
    ResolvedObserved, ResolvedParam, Size,
};
use bayesite_core::json;
use bayesite_core::model::{data_from_json, DataValue, Posterior};
use bayesite_core::predictive::{
    simulate_data_from_truth, simulate_posterior_predictive, simulate_prior_predictive,
    PriorPredictiveSettings,
};
use bayesite_core::protocol::{diagnose_ndjson, ndjson_lines, recover_check_report};
use bayesite_core::sampler::{sample, Settings};

fn normal(loc: Expr) -> Distribution {
    Distribution::Normal {
        loc,
        scale: Expr::Const(1.0),
    }
}

fn matvec_model(matrix_rank: i64, vector_rank: i64) -> ModelMeta {
    let mean = Expr::MatVec {
        matrix: Box::new(Expr::Data("matrix".to_string())),
        vector: Box::new(Expr::Data("vector".to_string())),
    };
    ModelMeta {
        params: vec![(
            "anchor".to_string(),
            ResolvedParam {
                distribution: normal(Expr::Const(0.0)),
                constraint: None,
                size: Size::Scalar,
            },
        )],
        data: vec![
            (
                "matrix".to_string(),
                ResolvedData {
                    schema: DataSchema::Rank(matrix_rank),
                },
            ),
            (
                "vector".to_string(),
                ResolvedData {
                    schema: DataSchema::Rank(vector_rank),
                },
            ),
        ],
        observed_nodes: vec![ResolvedObserved {
            name: "y".to_string(),
            distribution: normal(mean),
        }],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    }
}

fn zero_free_vector_matvec_model(size: Size, data_dependent: bool) -> ModelMeta {
    let mean = Expr::MatVec {
        matrix: Box::new(Expr::Data("matrix".to_string())),
        vector: Box::new(Expr::Param("z".to_string())),
    };
    let mut data = vec![(
        "matrix".to_string(),
        ResolvedData {
            schema: DataSchema::Rank(2),
        },
    )];
    if data_dependent {
        data.insert(
            0,
            (
                "n".to_string(),
                ResolvedData {
                    schema: DataSchema::Rank(0),
                },
            ),
        );
    }
    ModelMeta {
        params: vec![(
            "z".to_string(),
            ResolvedParam {
                distribution: normal(Expr::Const(0.0)),
                constraint: None,
                size,
            },
        )],
        data,
        observed_nodes: vec![ResolvedObserved {
            name: "y".to_string(),
            distribution: normal(mean),
        }],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    }
}

fn invalid_generated_matvec_model() -> ModelMeta {
    let mut meta = invalid_unused_matvec_model();
    meta.expressions = vec![(
        "invalid_generated".to_string(),
        Expr::MatVec {
            matrix: Box::new(Expr::Data("y".to_string())),
            vector: Box::new(Expr::Data("vector".to_string())),
        },
    )];
    meta
}

fn invalid_unused_matvec_model() -> ModelMeta {
    let mut meta = matvec_model(2, 1);
    let invalid = match &meta.observed_nodes[0].distribution {
        Distribution::Normal { loc, .. } => loc.clone(),
        _ => unreachable!("test model uses Normal"),
    };
    meta.expressions.push(("invalid".to_string(), invalid));
    meta.observed_nodes[0].distribution = normal(Expr::Param("anchor".to_string()));
    meta
}

fn value(shape: &[usize], values: &[f64]) -> DataValue {
    DataValue {
        shape: shape.to_vec(),
        values: values.to_vec(),
        integer: false,
    }
}

fn posterior(
    matrix_shape: &[usize],
    matrix_values: &[f64],
    vector_shape: &[usize],
    vector_values: &[f64],
    result_size: usize,
) -> Result<Posterior, Error> {
    Posterior::new(
        matvec_model(matrix_shape.len() as i64, vector_shape.len() as i64),
        vec![
            ("matrix".to_string(), value(matrix_shape, matrix_values)),
            ("vector".to_string(), value(vector_shape, vector_values)),
            (
                "y".to_string(),
                value(&[result_size], &vec![0.0; result_size]),
            ),
        ],
    )
}

#[test]
fn zero_length_free_vector_supports_empty_matvec_contraction() {
    for (size, data_dependent) in [(Size::Fixed(0), false), (Size::Data("n".to_string()), true)] {
        let meta = zero_free_vector_matvec_model(size, data_dependent);
        let mut full_data = vec![
            ("matrix".to_string(), value(&[1, 0], &[])),
            ("y".to_string(), value(&[1], &[0.0])),
        ];
        let mut declared_data = vec![("matrix".to_string(), value(&[1, 0], &[]))];
        if data_dependent {
            let n = DataValue {
                shape: Vec::new(),
                values: vec![0.0],
                integer: true,
            };
            full_data.insert(0, ("n".to_string(), n.clone()));
            declared_data.insert(0, ("n".to_string(), n));
        }

        let posterior = Posterior::new(meta.clone(), full_data.clone()).unwrap();
        let (logp, gradient) = posterior.logp_grad(&[]).unwrap();
        assert!(logp.is_finite());
        assert!(gradient.is_empty());

        let settings = Settings {
            num_warmup: 4,
            num_draws: 4,
            max_treedepth: 4,
            ..Settings::default()
        };
        let chain = sample(&posterior, &settings, 103, 0).unwrap();
        assert!(chain.draws.iter().all(Vec::is_empty));
        let lines = ndjson_lines(&posterior, &settings, 103, &[(0, chain)]).unwrap();
        for line in &lines[1..=settings.num_draws] {
            let document = json::parse(line).unwrap();
            let z = document
                .get("values")
                .and_then(|values| values.get("z"))
                .and_then(|z| z.as_array())
                .expect("draw contains vector parameter z");
            assert!(z.is_empty());
        }
        let fit = lines.join("\n") + "\n";
        let report = diagnose_ndjson(&fit).unwrap();
        let report = json::parse(&report).unwrap();
        assert!(matches!(
            report.get("rhat").and_then(|rhat| rhat.get("z")),
            Some(json::Value::Null)
        ));
        let truth = json::parse(
            r#"{"format":"bayescycle.data.json.v1","variables":{"z":{"dtype":"float64","shape":[0],"values":[]}}}"#,
        )
        .unwrap();
        let recovery = recover_check_report(&fit, &truth, None, 0.8).unwrap();
        let recovery = json::parse(&recovery).unwrap();
        let z = recovery
            .get("targets")
            .and_then(|targets| targets.get("z"))
            .expect("recovery report contains z");
        assert_eq!(z.get("mean").and_then(json::Value::as_array), Some(&[][..]));

        for forged_index in [0, 1, lines.len() - 1] {
            let mut forged_lines = lines.clone();
            forged_lines[forged_index] = forged_lines[forged_index].replacen(
                "\"parameter_count\":1",
                "\"parameter_count\":0",
                1,
            );
            let forged_fit = forged_lines.join("\n") + "\n";
            let forged_error =
                simulate_posterior_predictive(meta.clone(), full_data.clone(), &forged_fit, 99)
                    .unwrap_err();
            assert_eq!(forged_error.kind, ErrorKind::MalformedDocument);
            assert!(forged_error.message.contains("parameter_count"));

            let mut duplicate_lines = lines.clone();
            duplicate_lines[forged_index] = duplicate_lines[forged_index].replacen(
                "\"parameter_count\":1",
                "\"parameter_count\":1,\"parameter_count\":0",
                1,
            );
            let duplicate_fit = duplicate_lines.join("\n") + "\n";
            let duplicate_error =
                simulate_posterior_predictive(meta.clone(), full_data.clone(), &duplicate_fit, 99)
                    .unwrap_err();
            assert_eq!(duplicate_error.kind, ErrorKind::MalformedDocument);
            assert!(duplicate_error
                .message
                .contains("duplicate parameter_count"));
        }
        let mut duplicate_trailer_params = lines.clone();
        let trailer_index = duplicate_trailer_params.len() - 1;
        duplicate_trailer_params[trailer_index] = duplicate_trailer_params[trailer_index].replacen(
            "\"params\":1",
            "\"params\":1,\"params\":0",
            1,
        );
        let duplicate_trailer_fit = duplicate_trailer_params.join("\n") + "\n";
        let duplicate_trailer_error =
            simulate_posterior_predictive(meta.clone(), full_data, &duplicate_trailer_fit, 99)
                .unwrap_err();
        assert_eq!(duplicate_trailer_error.kind, ErrorKind::MalformedDocument);
        assert!(duplicate_trailer_error.message.contains("duplicate params"));

        let prior = simulate_prior_predictive(
            meta,
            declared_data,
            &PriorPredictiveSettings { num_draws: 1 },
            101,
        )
        .unwrap();
        assert_eq!(prior.draws[0].values[0].1.shape(), &[0]);
        assert!(prior.draws[0].values[0].1.data().is_empty());
        assert!(!prior.sites[0].integer);
        assert!(prior.sites[0].integer_by_coordinate.is_empty());
        assert_eq!(prior.draws[0].values[1].1.shape(), &[1]);
    }

    let mut unknown_meta = zero_free_vector_matvec_model(Size::Fixed(0), false);
    unknown_meta.expressions.push((
        "unknown".to_string(),
        Expr::MatVec {
            matrix: Box::new(Expr::Data("missing_matrix".to_string())),
            vector: Box::new(Expr::Param("z".to_string())),
        },
    ));
    let unknown_full_data = vec![
        ("matrix".to_string(), value(&[1, 0], &[])),
        ("y".to_string(), value(&[1], &[0.0])),
    ];
    let unknown_error = Posterior::new(unknown_meta.clone(), unknown_full_data).unwrap_err();
    assert_eq!(unknown_error.kind, ErrorKind::MalformedDocument);
    assert!(unknown_error.message.contains("missing_matrix"));
    let unknown_prior_error = simulate_prior_predictive(
        unknown_meta,
        vec![("matrix".to_string(), value(&[1, 0], &[]))],
        &PriorPredictiveSettings { num_draws: 1 },
        105,
    )
    .unwrap_err();
    assert_eq!(unknown_prior_error.kind, ErrorKind::MalformedDocument);
    assert!(unknown_prior_error.message.contains("missing_matrix"));

    let mut ordered_meta = zero_free_vector_matvec_model(Size::Fixed(0), false);
    ordered_meta.params[0].1.constraint = Some(Constraint::Ordered);
    let ordered = Posterior::new(
        ordered_meta,
        vec![
            ("matrix".to_string(), value(&[1, 0], &[])),
            ("y".to_string(), value(&[1], &[0.0])),
        ],
    )
    .unwrap();
    let ordered_chain = sample(
        &ordered,
        &Settings {
            num_warmup: 4,
            num_draws: 4,
            max_treedepth: 4,
            ..Settings::default()
        },
        107,
        0,
    )
    .unwrap();
    assert!(ordered_chain.draws.iter().all(Vec::is_empty));

    let parameterless_meta = ModelMeta {
        params: vec![],
        data: vec![],
        observed_nodes: vec![ResolvedObserved {
            name: "y".to_string(),
            distribution: normal(Expr::Const(0.0)),
        }],
        expressions: vec![],
        free_values: vec![],
        stochastic_sites: vec![],
    };
    let parameterless_data = vec![("y".to_string(), value(&[], &[0.0]))];
    let parameterless =
        Posterior::new(parameterless_meta.clone(), parameterless_data.clone()).unwrap();
    let parameterless_settings = Settings {
        num_warmup: 4,
        num_draws: 4,
        max_treedepth: 4,
        ..Settings::default()
    };
    let parameterless_chain = sample(&parameterless, &parameterless_settings, 109, 0).unwrap();
    let parameterless_lines = ndjson_lines(
        &parameterless,
        &parameterless_settings,
        109,
        &[(0, parameterless_chain)],
    )
    .unwrap();
    let parameterless_fit = parameterless_lines.join("\n") + "\n";
    let parameterless_report = diagnose_ndjson(&parameterless_fit).unwrap();
    let parameterless_report = json::parse(&parameterless_report).unwrap();
    assert_eq!(
        parameterless_report
            .get("source_parameter_order")
            .and_then(json::Value::as_array),
        Some(&[][..])
    );
    let replicated = simulate_posterior_predictive(
        parameterless_meta.clone(),
        parameterless_data.clone(),
        &parameterless_fit,
        113,
    )
    .unwrap();
    assert_eq!(replicated.draws.len(), 4);

    let malformed_fit = parameterless_fit.replacen("\"values\":{}", "\"values\":null", 1);
    let malformed_error =
        simulate_posterior_predictive(parameterless_meta, parameterless_data, &malformed_fit, 127)
            .unwrap_err();
    assert_eq!(malformed_error.kind, ErrorKind::MalformedDocument);
    assert!(malformed_error.message.contains("values object"));
}

#[test]
fn posterior_evaluates_matrix_vector_expression() {
    let posterior = posterior(
        &[2, 3],
        &[1.0, 2.0, 3.0, -1.0, 0.5, 4.0],
        &[3],
        &[0.25, -0.5, 2.0],
        2,
    )
    .unwrap();

    let (logp, gradient) = posterior.logp_grad(&[0.2]).unwrap();

    assert!(logp.is_finite());
    assert_eq!(gradient.len(), 1);
    assert!(gradient[0].is_finite());
}

#[test]
fn posterior_rejects_matrix_rank_other_than_two() {
    let error = posterior(&[3], &[1.0, 2.0, 3.0], &[3], &[1.0, 2.0, 3.0], 3).unwrap_err();

    assert_eq!(error.kind, ErrorKind::DataShapeMismatch);
    assert!(error.message.contains("matrix must be rank 2"));
}

#[test]
fn posterior_rejects_vector_rank_other_than_one() {
    let error = posterior(&[2, 2], &[1.0, 0.0, 0.0, 1.0], &[1, 2], &[1.0, 2.0], 2).unwrap_err();

    assert_eq!(error.kind, ErrorKind::DataShapeMismatch);
    assert!(error.message.contains("vector must be rank 1"));
}

#[test]
fn posterior_rejects_contraction_dimension_mismatch() {
    let error = posterior(&[2, 3], &[1.0; 6], &[4], &[1.0; 4], 2).unwrap_err();

    assert_eq!(error.kind, ErrorKind::DataShapeMismatch);
    assert!(error.message.contains("matrix shape [2, 3]"));
    assert!(error.message.contains("vector shape [4]"));
}

#[test]
fn prior_predictive_evaluates_matrix_vector_distribution_parameter() {
    let fixture = json::parse(include_str!(
        "../../../tests/golden_ir/fixtures/mvn_non_centered.json"
    ))
    .unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let mut data = data_from_json(fixture.get("data").unwrap()).unwrap();
    data.retain(|(name, _)| name != "y");

    let run = simulate_prior_predictive(meta, data, &PriorPredictiveSettings { num_draws: 3 }, 71)
        .unwrap();

    assert_eq!(run.draws.len(), 3);
    assert_eq!(run.sites.len(), 2);
    for draw in run.draws {
        assert_eq!(draw.values[0].0, "z");
        assert_eq!(draw.values[0].1.shape(), &[3]);
        assert_eq!(draw.values[1].0, "y");
        assert_eq!(draw.values[1].1.shape(), &[3]);
        assert!(draw.values[1]
            .1
            .data()
            .iter()
            .all(|value| value.is_finite()));
    }
}

#[test]
fn prior_predictive_rejects_invalid_matrix_vector_shapes_before_drawing() {
    let error = simulate_prior_predictive(
        matvec_model(2, 1),
        vec![
            ("matrix".to_string(), value(&[2, 3], &[1.0; 6])),
            ("vector".to_string(), value(&[4], &[1.0; 4])),
        ],
        &PriorPredictiveSettings { num_draws: 1 },
        73,
    )
    .unwrap_err();

    assert_eq!(error.kind, ErrorKind::DataShapeMismatch);
    assert!(error.message.contains("matrix shape [2, 3]"));
    assert!(error.message.contains("vector shape [4]"));
}

#[test]
fn forward_bind_paths_reject_invalid_unused_matrix_vector_expression() {
    let data = vec![
        ("matrix".to_string(), value(&[2, 3], &[1.0; 6])),
        ("vector".to_string(), value(&[4], &[1.0; 4])),
    ];

    let prior_error = simulate_prior_predictive(
        invalid_unused_matvec_model(),
        data.clone(),
        &PriorPredictiveSettings { num_draws: 1 },
        79,
    )
    .unwrap_err();
    assert_eq!(prior_error.kind, ErrorKind::DataShapeMismatch);
    assert!(prior_error.message.contains("matrix shape [2, 3]"));

    let simulate_error = simulate_data_from_truth(
        invalid_unused_matvec_model(),
        data,
        vec![("anchor".to_string(), value(&[], &[0.0]))],
        83,
    )
    .unwrap_err();
    assert_eq!(simulate_error.kind, ErrorKind::DataShapeMismatch);
    assert!(simulate_error.message.contains("vector shape [4]"));
}

#[test]
fn forward_paths_complete_deferred_generated_value_validation() {
    let data = vec![
        ("matrix".to_string(), value(&[2, 3], &[1.0; 6])),
        ("vector".to_string(), value(&[3], &[1.0; 3])),
    ];

    let prior_error = simulate_prior_predictive(
        invalid_generated_matvec_model(),
        data.clone(),
        &PriorPredictiveSettings { num_draws: 1 },
        89,
    )
    .unwrap_err();
    assert_eq!(prior_error.kind, ErrorKind::DataShapeMismatch);
    assert!(prior_error.message.contains("matrix must be rank 2"));

    let simulate_error = simulate_data_from_truth(
        invalid_generated_matvec_model(),
        data,
        vec![("anchor".to_string(), value(&[], &[0.0]))],
        97,
    )
    .unwrap_err();
    assert_eq!(simulate_error.kind, ErrorKind::DataShapeMismatch);
    assert!(simulate_error.message.contains("matrix must be rank 2"));
}

#[test]
fn posterior_accepts_zero_sized_matrix_vector_shapes() {
    for (matrix_shape, vector_shape, result_size) in [
        (vec![3, 0], vec![0], 3),
        (vec![0, 3], vec![3], 0),
        (vec![0, 0], vec![0], 0),
    ] {
        let matrix_values = vec![0.0; matrix_shape.iter().product()];
        let vector_values = vec![0.0; vector_shape.iter().product()];
        let posterior = posterior(
            &matrix_shape,
            &matrix_values,
            &vector_shape,
            &vector_values,
            result_size,
        )
        .unwrap();

        let (logp, gradient) = posterior.logp_grad(&[0.0]).unwrap();
        assert!(logp.is_finite());
        assert_eq!(gradient, vec![0.0]);
    }
}
