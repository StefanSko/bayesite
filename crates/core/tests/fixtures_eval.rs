//! Gate G1: log density and gradient must reproduce the committed JAX
//! values for every golden fixture (logp rtol 1e-12, gradient rtol 1e-10).

use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};

const ALL_FIXTURES: [&str; 6] = [
    "bounded_rates",
    "eight_schools_non_centered",
    "linear_regression",
    "ordinal_regression",
    "partially_observed_mvn",
    "varying_intercepts_poisson",
];

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    json::parse(&std::fs::read_to_string(path).expect("fixture readable")).expect("fixture parses")
}

fn rel_close(got: f64, want: f64, rtol: f64) -> bool {
    (got - want).abs() <= rtol * want.abs().max(1e-8)
}

#[test]
fn every_golden_fixture_is_in_the_logp_gradient_gate() {
    let dir = format!(
        "{}/../../tests/golden_ir/fixtures",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut fixture_names: Vec<String> = std::fs::read_dir(dir)
        .expect("fixture directory readable")
        .filter_map(|entry| {
            let path = entry.expect("fixture directory entry readable").path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                return None;
            }
            Some(
                path.file_stem()
                    .expect("fixture has a file stem")
                    .to_str()
                    .expect("fixture path is UTF-8")
                    .to_string(),
            )
        })
        .collect();
    fixture_names.sort();
    let expected: Vec<String> = ALL_FIXTURES
        .iter()
        .map(|name| (*name).to_string())
        .collect();
    assert_eq!(fixture_names, expected);
}

#[test]
fn logp_and_gradient_match_jax_at_committed_points() {
    for name in ALL_FIXTURES {
        let doc = fixture(name);
        let meta = decode_model(doc.get("ir").expect("ir")).expect("ir decodes");
        let data = data_from_json(doc.get("data").expect("data")).expect("data parses");
        let posterior = Posterior::new(meta, data).unwrap_or_else(|e| panic!("{name}: {e}"));

        let evaluations = doc
            .get("evaluations")
            .and_then(Value::as_array)
            .expect("evals");
        assert!(!evaluations.is_empty());
        for (i, eval) in evaluations.iter().enumerate() {
            let q: Vec<f64> = eval
                .get("q")
                .and_then(Value::as_array)
                .expect("q")
                .iter()
                .map(|v| v.as_f64().expect("q numeric"))
                .collect();
            let want_logp = eval
                .get("log_density")
                .and_then(Value::as_f64)
                .expect("logp");
            let want_grad: Vec<f64> = eval
                .get("gradient")
                .and_then(Value::as_array)
                .expect("grad")
                .iter()
                .map(|v| v.as_f64().expect("grad numeric"))
                .collect();

            let (logp, grad) = posterior
                .logp_grad(&q)
                .unwrap_or_else(|e| panic!("{name} eval {i}: {e}"));

            assert!(
                rel_close(logp, want_logp, 1e-12),
                "{name} eval {i}: logp {logp:.17e} != {want_logp:.17e} \
                 (rel err {:.3e})",
                ((logp - want_logp) / want_logp).abs()
            );
            assert_eq!(
                grad.len(),
                want_grad.len(),
                "{name} eval {i}: gradient length"
            );
            for (j, (&g, &w)) in grad.iter().zip(want_grad.iter()).enumerate() {
                assert!(
                    rel_close(g, w, 1e-10),
                    "{name} eval {i} grad[{j}]: {g:.17e} != {w:.17e}"
                );
            }
        }
    }
}

#[test]
fn wrong_q_length_is_a_shape_error() {
    let doc = fixture("linear_regression");
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let data = data_from_json(doc.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    assert_eq!(posterior.n_params(), 3);
    let err = posterior.logp_grad(&[0.0; 4]).unwrap_err();
    assert_eq!(err.kind, bayesite_core::error::ErrorKind::DataShapeMismatch);
}

#[test]
fn missing_data_is_a_shape_error() {
    let doc = fixture("linear_regression");
    let mut data = data_from_json(doc.get("data").unwrap()).unwrap();
    data.retain(|(name, _)| name != "y");
    let err = Posterior::new(decode_model(doc.get("ir").unwrap()).unwrap(), data).unwrap_err();
    assert_eq!(err.kind, bayesite_core::error::ErrorKind::DataShapeMismatch);
    assert!(
        err.message.contains("y"),
        "message names the missing value: {}",
        err.message
    );
}

#[test]
fn constrained_draws_recover_constrained_space() {
    // bounded_rates: p in (0,1), level in (-1,3).
    let doc = fixture("bounded_rates");
    let meta = decode_model(doc.get("ir").unwrap()).unwrap();
    let data = data_from_json(doc.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    let constrained = posterior.constrain(&[-3.0, 4.0]).unwrap();
    assert_eq!(constrained[0].0, "p");
    let p = constrained[0].1.data()[0];
    assert!(p > 0.0 && p < 1.0);
    let level = constrained[1].1.data()[0];
    assert!(level > -1.0 && level < 3.0);
}
