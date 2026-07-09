use bayesite_core::error::ErrorKind;
use bayesite_core::ir::{
    Constraint, DataSchema, Distribution, Expr, ModelMeta, ResolvedData, ResolvedFreeValue,
    ResolvedStochasticSite, Size,
};
use bayesite_core::model::{DataValue, Posterior};

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
