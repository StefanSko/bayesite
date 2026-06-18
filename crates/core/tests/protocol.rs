//! Native tests of the wasm-boundary request handler (the pure seam the
//! unsafe ABI shims delegate to).

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::predictive::{prior_predictive_ndjson_lines, PriorPredictiveSettings};
use bayesite_core::protocol::{diagnose_ndjson, handle_request, ndjson_lines};
use bayesite_core::sampler::{sample, Settings};

fn fixture_text(name: &str) -> String {
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

fn fixture_declared_data(name: &str, declared_data: &[&str]) -> Value {
    let fixture = json::parse(&fixture_text(name)).unwrap();
    match fixture.get("data").unwrap() {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .filter(|(name, _)| declared_data.contains(&name.as_str()))
                .cloned()
                .collect(),
        ),
        _ => panic!("fixture data must be an object"),
    }
}

fn object_entry_mut<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    match value {
        Value::Object(entries) => entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
            .unwrap_or_else(|| panic!("missing object entry {key}")),
        _ => panic!("expected object for key {key}"),
    }
}

fn string_array(value: &Value) -> Vec<&str> {
    match value {
        Value::Array(entries) => entries
            .iter()
            .map(|value| value.as_str().expect("expected string"))
            .collect(),
        _ => panic!("expected array"),
    }
}

fn object_keys(value: &Value) -> Vec<&str> {
    match value {
        Value::Object(entries) => entries.iter().map(|(name, _)| name.as_str()).collect(),
        _ => panic!("expected object"),
    }
}

fn int_array(value: &Value) -> Vec<i64> {
    match value {
        Value::Array(entries) => entries
            .iter()
            .map(|value| value.as_i64().expect("expected integer"))
            .collect(),
        _ => panic!("expected array"),
    }
}

fn assert_treedepth_support(value: &Value, max_treedepth: i64) {
    assert_eq!(
        int_array(
            value
                .get("treedepth_bin_order")
                .expect("treedepth bin order")
        ),
        (0..=max_treedepth).collect::<Vec<_>>()
    );
    assert_eq!(
        value.get("treedepth_bin_count").and_then(Value::as_i64),
        Some(max_treedepth + 1)
    );
}

fn assert_count_support(value: &Value, prefix: &str, max_count: i64) {
    let bounds_name = format!("{prefix}_bounds");
    let order_name = format!("{prefix}_bin_order");
    let count_name = format!("{prefix}_bin_count");
    let bounds = value.get(&bounds_name).expect("count bounds");
    assert_eq!(bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(bounds.get("max").and_then(Value::as_i64), Some(max_count));
    assert_eq!(
        int_array(value.get(&order_name).expect("count bin order")),
        (0..=max_count).collect::<Vec<_>>()
    );
    assert_eq!(
        value.get(&count_name).and_then(Value::as_i64),
        Some(max_count + 1)
    );
}

fn coordinate_order(value: &Value) -> Vec<Vec<i64>> {
    match value {
        Value::Array(entries) => entries.iter().map(int_array).collect(),
        _ => panic!("expected coordinate order array"),
    }
}

fn assert_quantile_index(index: &Value, expected_position: f64) {
    assert!(
        (index.get("position").and_then(Value::as_f64).unwrap() - expected_position).abs() < 1e-12
    );
    assert_eq!(
        index.get("floor").and_then(Value::as_i64),
        Some(expected_position.floor() as i64)
    );
    assert_eq!(
        index.get("ceil").and_then(Value::as_i64),
        Some(expected_position.ceil() as i64)
    );
}

fn assert_interval_quantile_index_metadata(
    interval_bounds: &Value,
    draw_count: i64,
    interval: f64,
) {
    assert_eq!(
        interval_bounds
            .get("quantile_index_base")
            .and_then(Value::as_str),
        Some("zero_based_sorted_ascending_posterior_draws")
    );
    assert_eq!(
        interval_bounds
            .get("sorted_draw_count")
            .and_then(Value::as_i64),
        Some(draw_count)
    );
    let lower_quantile = (1.0 - interval) / 2.0;
    let upper_quantile = 1.0 - lower_quantile;
    let draw_span = (draw_count - 1) as f64;
    assert_quantile_index(
        interval_bounds
            .get("lower_quantile_index")
            .expect("lower quantile index"),
        lower_quantile * draw_span,
    );
    assert_quantile_index(
        interval_bounds
            .get("upper_quantile_index")
            .expect("upper quantile index"),
        upper_quantile * draw_span,
    );
}

fn assert_sample_artifact_identity(value: &Value) {
    assert_eq!(
        value.get("artifact_kind").and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        value.get("artifact_scope").and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
}

fn assert_varying_intercepts_declared_data_values(payload: &Value) {
    assert_eq!(
        string_array(payload.get("declared_data_order").unwrap()),
        ["n_groups", "group_idx", "x"]
    );
    let declared_data = payload.get("declared_data").expect("declared data");
    assert!(matches!(declared_data.get("n_groups"), Some(Value::Int(3))));
    assert_eq!(
        int_array(declared_data.get("group_idx").unwrap()),
        [0, 0, 1, 1, 2, 2]
    );
    assert!(declared_data
        .get("x")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .all(|value| matches!(value, Value::Float(_))));
    assert!(matches!(
        payload
            .get("declared_data_integer_by_coordinate")
            .and_then(|values| values.get("n_groups")),
        Some(Value::Bool(true))
    ));
    assert!(payload
        .get("declared_data_integer_by_coordinate")
        .and_then(|values| values.get("group_idx"))
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .all(|value| matches!(value, Value::Bool(true))));
    assert_eq!(
        coordinate_order(
            payload
                .get("declared_data_coordinate_order")
                .and_then(|values| values.get("n_groups"))
                .expect("n_groups coordinate order")
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        coordinate_order(
            payload
                .get("declared_data_coordinate_order")
                .and_then(|values| values.get("group_idx"))
                .expect("group_idx coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]
    );
}

fn assert_z_alpha_recover_interval_contains_truth_by_coordinate(payload: &Value) {
    let z_alpha = payload
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("z_alpha recovery summary");
    let contains_by_coordinate = z_alpha
        .get("interval_contains_truth_by_coordinate")
        .and_then(Value::as_array)
        .expect("z_alpha interval containment facts");
    assert_eq!(contains_by_coordinate.len(), 3);
    assert!(contains_by_coordinate
        .iter()
        .all(|value| matches!(value, Value::Bool(_))));
    assert!(matches!(
        z_alpha.get("interval_contains_truth"),
        Some(Value::Bool(_))
    ));
}

fn assert_z_alpha_sbc_rank_histograms(payload: &Value) {
    let rank_draws = payload
        .get("rank_draws")
        .and_then(Value::as_i64)
        .expect("sbc rank draws");
    let replicates = payload
        .get("replicates")
        .and_then(Value::as_i64)
        .expect("sbc replicates");
    let z_alpha = payload
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("sbc z_alpha summary");
    assert_eq!(
        payload
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        z_alpha.get("summary_scale").and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        payload.get("rank_bin_count").and_then(Value::as_i64),
        Some(rank_draws + 1)
    );
    assert_eq!(
        z_alpha.get("rank_draws").and_then(Value::as_i64),
        Some(rank_draws)
    );
    assert_eq!(
        z_alpha.get("rank_bin_count").and_then(Value::as_i64),
        Some(rank_draws + 1)
    );
    assert_eq!(
        z_alpha.get("replicate_count").and_then(Value::as_i64),
        Some(replicates)
    );
    assert_eq!(
        z_alpha.get("replicate_index_base").and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    let replicate_order = int_array(
        z_alpha
            .get("replicate_order")
            .expect("sbc z_alpha replicate order"),
    );
    assert_eq!(replicate_order, (0..replicates).collect::<Vec<_>>());
    let seed = payload
        .get("seed")
        .and_then(Value::as_i64)
        .expect("sbc seed");
    let expected_prior_seeds = replicate_order
        .iter()
        .map(|replicate| seed + 2 * replicate)
        .collect::<Vec<_>>();
    let expected_sample_seeds = replicate_order
        .iter()
        .map(|replicate| seed + 2 * replicate + 1)
        .collect::<Vec<_>>();
    assert_eq!(
        int_array(z_alpha.get("prior_seed").expect("sbc z_alpha prior seeds")),
        expected_prior_seeds
    );
    assert_eq!(
        int_array(
            z_alpha
                .get("sample_seed")
                .expect("sbc z_alpha sample seeds")
        ),
        expected_sample_seeds
    );
    let seed_schedule = z_alpha
        .get("seed_schedule")
        .expect("sbc z_alpha seed schedule");
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    let rank_bounds = z_alpha.get("rank_bounds").expect("sbc z_alpha rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(
        rank_bounds.get("max").and_then(Value::as_i64),
        Some(rank_draws)
    );
    assert_eq!(
        z_alpha
            .get("rank_histogram_statistic")
            .and_then(Value::as_str),
        Some("count_simulated_replicates_by_rank")
    );
    assert_eq!(
        z_alpha.get("rank_histogram_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    let rank_histograms = z_alpha
        .get("rank_histogram")
        .and_then(Value::as_array)
        .expect("sbc z_alpha rank histograms");
    assert_eq!(rank_histograms.len(), 3);
    for histogram in rank_histograms {
        let histogram = int_array(histogram);
        assert_eq!(histogram.len() as i64, rank_draws + 1);
        assert_eq!(histogram.iter().sum::<i64>(), replicates);
    }

    let ranks = z_alpha
        .get("ranks")
        .and_then(Value::as_array)
        .expect("sbc z_alpha ranks");
    assert_eq!(ranks.len() as i64, replicates);
    for rank in ranks {
        assert_eq!(int_array(rank).len(), 3);
    }
    let truths = z_alpha
        .get("truth")
        .and_then(Value::as_array)
        .expect("sbc z_alpha truths");
    assert_eq!(truths.len() as i64, replicates);
    for truth in truths {
        assert_eq!(truth.as_array().expect("truth vector").len(), 3);
    }
}

fn assert_workflow_phases(value: &Value, field: &str) {
    let phases: Vec<&str> = value
        .get(field)
        .and_then(Value::as_array)
        .expect("workflow phases")
        .iter()
        .map(|phase| phase.as_str().expect("phase names are strings"))
        .collect();
    assert_eq!(
        phases,
        [
            "parse_json",
            "decode_ir",
            "bind_declared_data",
            "simulate_prior_predictive",
            "bind_declared_and_generated_data",
            "build_posterior_state",
            "evaluate_logp_grad",
            "run_nuts",
            "emit_report",
        ]
    );
}

fn assert_prior_predictive_workflow_phases(value: &Value, field: &str) {
    let phases: Vec<&str> = value
        .get(field)
        .and_then(Value::as_array)
        .expect("workflow phases")
        .iter()
        .map(|phase| phase.as_str().expect("phase names are strings"))
        .collect();
    assert_eq!(
        phases,
        [
            "parse_json",
            "decode_ir",
            "bind_declared_data",
            "simulate_prior_predictive",
            "emit_artifact",
        ]
    );
}

fn assert_diagnose_workflow_phases(value: &Value, field: &str) {
    let phases: Vec<&str> = value
        .get(field)
        .and_then(Value::as_array)
        .expect("workflow phases")
        .iter()
        .map(|phase| phase.as_str().expect("phase names are strings"))
        .collect();
    assert_eq!(
        phases,
        [
            "parse_fit_ndjson",
            "validate_fit_artifact",
            "recompute_diagnostics",
            "emit_report",
        ]
    );
}

#[test]
fn sample_command_returns_single_chain_ndjson() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 200, "num_draws": 100}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(5)),
        ("chain_id".to_string(), Value::Int(3)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 1 + 100 + 1);
    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_sample_artifact_identity(&header);
    assert_eq!(header.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(int_array(header.get("chain_order").unwrap()), [3]);
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(100));
    assert_eq!(
        header.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(header.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    assert_eq!(
        coordinate_order(
            header
                .get("params")
                .and_then(Value::as_array)
                .and_then(|params| params.first())
                .and_then(|param| param.get("coordinate_order"))
                .unwrap()
        ),
        vec![Vec::<i64>::new()]
    );
    let first = json::parse(lines[1]).unwrap();
    assert_eq!(
        first.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_sample_artifact_identity(&first);
    assert_eq!(first.get("draw_index").and_then(Value::as_i64), Some(0));
    assert_eq!(
        first.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_retained_draw_order")
    );
    assert_eq!(first.get("seed").and_then(Value::as_i64), Some(5));
    assert_eq!(first.get("draw_count").and_then(Value::as_i64), Some(100));
    assert_eq!(first.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(int_array(first.get("chain_order").unwrap()), [3]);
    assert_eq!(first.get("chain").and_then(Value::as_i64), Some(3));
    assert_eq!(
        first.get("chain_index_base").and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        first.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(first.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    let trailer = trailer.get("trailer").expect("trailer object");
    assert_sample_artifact_identity(trailer);
    assert_eq!(trailer.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(int_array(trailer.get("chain_order").unwrap()), [3]);
    assert_eq!(
        trailer.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(trailer.get("seed").and_then(Value::as_i64), Some(5));
    assert_eq!(
        trailer.get("draws_per_chain").and_then(Value::as_i64),
        Some(100)
    );
    assert_eq!(trailer.get("params").and_then(Value::as_i64), Some(3));
    assert_eq!(trailer.get("draw_count").and_then(Value::as_i64), Some(100));
    assert_eq!(
        trailer.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(trailer.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let chain_stats = trailer.get("chains").and_then(Value::as_array).unwrap();
    assert_eq!(chain_stats.len(), 1);
    assert_eq!(
        chain_stats[0]
            .get("chain_index_base")
            .and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        chain_stats[0].get("draw_count").and_then(Value::as_i64),
        Some(100)
    );
}

#[test]
fn posterior_predictive_request_returns_ndjson() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(31)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(37)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 1 + 4 + 1);
    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header
            .get("posterior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        header.get("artifact_kind").and_then(Value::as_str),
        Some("posterior_predictive_draws")
    );
}

#[test]
fn posterior_predictive_rejects_partial_observed_site_coverage() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(47)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());

    let mut model = fixture.get("ir").unwrap().clone();
    let model_meta = object_entry_mut(&mut model, "model");
    match object_entry_mut(model_meta, "observed_nodes") {
        Value::Array(observed) => observed.push(json::parse(
            r#"{"node":"ResolvedObserved","name":"z","distribution":{"node":"Normal","loc":0.0,"scale":1.0}}"#,
        ).unwrap()),
        _ => panic!("observed_nodes must be an array"),
    }
    match object_entry_mut(model_meta, "stochastic_sites") {
        Value::Array(sites) => sites.push(json::parse(
            r#"{"node":"ResolvedStochasticSite","name":"z","distribution":{"node":"Normal","loc":0.0,"scale":1.0},"value":0.0}"#,
        ).unwrap()),
        _ => panic!("stochastic_sites must be an array"),
    }
    let mut data = fixture.get("data").unwrap().clone();
    match &mut data {
        Value::Object(entries) => entries.push(("z".to_string(), Value::Float(0.0))),
        _ => panic!("fixture data must be an object"),
    }
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), model),
        ("data".to_string(), data),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(53)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert!(response
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("observed node \"z\" is not directly assignable"));
}

#[test]
fn posterior_predictive_uses_broadcast_likelihood_shape() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let mut data = fixture.get("data").unwrap().clone();
    if let Value::Object(entries) = &mut data {
        for (name, value) in entries {
            if name == "y" {
                *value = Value::Float(0.0);
            }
        }
    }
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data.clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(50)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(51)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let header = json::parse(response.lines().next().unwrap()).unwrap();
    let site = header
        .get("sites")
        .and_then(Value::as_array)
        .and_then(|sites| sites.first())
        .unwrap();
    assert_eq!(int_array(site.get("shape").unwrap()), [5]);
    let first_draw = json::parse(response.lines().nth(1).unwrap()).unwrap();
    assert_eq!(
        first_draw
            .get("values")
            .and_then(|values| values.get("y"))
            .and_then(Value::as_array)
            .unwrap()
            .len(),
        5
    );
}

#[test]
fn posterior_predictive_uses_distribution_integerness_not_observed_json_lexemes() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let mut data = fixture.get("data").unwrap().clone();
    if let Value::Object(entries) = &mut data {
        for (name, value) in entries {
            if name == "y" {
                *value =
                    json::parse(r#"{"dtype":"int64","shape":[5],"values":[0,0,0,0,0]}"#).unwrap();
            }
        }
    }
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data.clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(54)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(55)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let header = json::parse(response.lines().next().unwrap()).unwrap();
    assert_eq!(
        header
            .get("posterior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    let site = header
        .get("sites")
        .and_then(Value::as_array)
        .and_then(|sites| sites.first())
        .unwrap();
    assert!(matches!(site.get("integer"), Some(Value::Bool(false))));
}

#[test]
fn posterior_predictive_rejects_partial_fit_streams() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(57)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let mut lines: Vec<&str> = fit.lines().collect();
    let last_draw_line = lines.len() - 2;
    lines.remove(last_draw_line);
    let partial_fit = lines.join("\n");
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        ("fit".to_string(), Value::Str(partial_fit)),
        ("seed".to_string(), Value::Int(58)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("MalformedDocument")
    );
    assert!(response
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("draw_count"));
}

#[test]
fn posterior_check_handles_empty_observed_tensors() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let mut data = fixture.get("data").unwrap().clone();
    if let Value::Object(entries) = &mut data {
        for (name, value) in entries {
            if name == "x" || name == "y" {
                *value = json::parse(r#"{"dtype":"float64","shape":[0],"values":[]}"#).unwrap();
            }
        }
    }
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data.clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(59)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-check".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data.clone()),
        ("fit".to_string(), Value::Str(fit.clone())),
        ("seed".to_string(), Value::Int(61)),
    ]);
    let yrep_request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(60)),
    ]);
    let yrep = handle_request(&json::write(&yrep_request).unwrap());
    let first_draw = json::parse(yrep.lines().nth(1).unwrap()).unwrap();
    assert!(matches!(
        first_draw
            .get("values")
            .and_then(|values| values.get("y")),
        Some(Value::Array(values)) if values.is_empty()
    ));

    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response
            .get("posterior_check_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    let checks = response.get("checks").and_then(Value::as_array).unwrap();
    let mean = checks
        .iter()
        .find(|check| check.get("statistic").and_then(Value::as_str) == Some("mean"))
        .unwrap();
    assert!(matches!(
        mean.get("summary")
            .and_then(|summary| summary.get("observed")),
        Some(Value::Null)
    ));
}

#[test]
fn posterior_check_request_returns_factual_report() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 10, "num_draws": 4}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(41)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("posterior-check".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        ("fit".to_string(), Value::Str(fit)),
        ("seed".to_string(), Value::Int(43)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response
            .get("posterior_check_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("report_kind").and_then(Value::as_str),
        Some("posterior_predictive_check_facts")
    );
    assert!(response.get("verdict").is_none());
}

#[test]
fn diagnose_command_consumes_fit_ndjson() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let sample_request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_warmup": 200, "num_draws": 100}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(11)),
        ("chain_id".to_string(), Value::Int(0)),
    ]);
    let fit = handle_request(&json::write(&sample_request).unwrap());
    let first_fit_line = json::parse(fit.lines().next().unwrap()).unwrap();
    assert!(first_fit_line.get("error").is_none(), "{first_fit_line:?}");
    let diagnose_request = Value::Object(vec![
        ("command".to_string(), Value::Str("diagnose".to_string())),
        ("fit".to_string(), Value::Str(fit)),
    ]);
    let response = json::parse(&handle_request(&json::write(&diagnose_request).unwrap())).unwrap();
    assert!(response.get("error").is_none(), "{response:?}");
    assert_eq!(
        response.get("diagnostics_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("source_draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("rhat_statistic").and_then(Value::as_str),
        Some("split_rhat")
    );
    assert_eq!(
        response.get("rhat_scope").and_then(Value::as_str),
        Some("max_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        response.get("ess_statistic").and_then(Value::as_str),
        Some("effective_sample_size_geyer_initial_monotone_sequence")
    );
    assert_eq!(
        response.get("ess_scope").and_then(Value::as_str),
        Some("min_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        response.get("source_artifact_kind").and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        response
            .get("source_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_diagnose_workflow_phases(&response, "workflow_phases");
    assert_eq!(
        response.get("source_seed").and_then(Value::as_i64),
        Some(11)
    );
    assert_eq!(
        response.get("source_chains").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response.get("source_chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(int_array(response.get("source_chain_order").unwrap()), [0]);
    assert_eq!(
        response.get("source_draw_count").and_then(Value::as_i64),
        Some(100)
    );
    assert!(matches!(
        response.get("source_draw_index_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        response.get("source_draw_parameter_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        response.get("source_draw_artifact_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        response.get("source_draw_chain_metadata"),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        response
            .get("source_parameter_count")
            .and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        response
            .get("source_settings")
            .and_then(|v| v.get("num_draws"))
            .and_then(Value::as_i64),
        Some(100)
    );
    assert_eq!(
        response
            .get("source_params")
            .and_then(Value::as_array)
            .map(|params| params.len()),
        Some(3)
    );
    assert_eq!(
        response
            .get("source_packing")
            .and_then(Value::as_array)
            .and_then(|packing| packing.first())
            .and_then(Value::as_str),
        Some("alpha")
    );
    assert_eq!(
        string_array(response.get("source_parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    assert_eq!(
        response.get("draws_per_chain").and_then(Value::as_i64),
        Some(100)
    );
    let chains = response
        .get("chains")
        .and_then(Value::as_array)
        .expect("diagnose chain stats");
    assert_eq!(chains.len(), 1);
    assert_eq!(
        chains[0].get("draw_count").and_then(Value::as_i64),
        Some(100)
    );
    let trailer_completion = response
        .get("source_trailer_completion_metadata")
        .expect("source trailer completion metadata");
    for field in [
        "draws_format",
        "artifact_kind",
        "artifact_scope",
        "workflow_phases",
        "seed",
        "chain_count",
        "chain_order",
        "draw_count",
        "draws_per_chain",
        "parameter_count",
        "parameter_order",
        "params",
    ] {
        assert!(matches!(
            trailer_completion.get(field),
            Some(Value::Bool(true))
        ));
    }
}

#[test]
fn diagnose_rejects_lines_after_trailer() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
        r#"{"chain":0,"draw":4,"values":{"alpha":4.0}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer must be the final line; remove trailing lines after the trailer"
    );
}

#[test]
fn diagnose_reports_absent_legacy_trailer_completion_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let response = json::parse(&diagnose_ndjson(&fit).unwrap()).unwrap();
    assert!(response.get("source_artifact_kind").is_none());
    assert!(response.get("source_artifact_scope").is_none());
    assert_eq!(
        response.get("source_chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response.get("source_draw_count").and_then(Value::as_i64),
        Some(4)
    );
    assert!(matches!(
        response.get("source_draw_index_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        response.get("source_draw_parameter_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        response.get("source_draw_artifact_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        response.get("source_draw_chain_metadata"),
        Some(Value::Bool(false))
    ));
    assert_eq!(
        response
            .get("source_parameter_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    let trailer_completion = response
        .get("source_trailer_completion_metadata")
        .expect("source trailer completion metadata");
    for field in [
        "draws_format",
        "artifact_kind",
        "artifact_scope",
        "workflow_phases",
        "seed",
        "chain_count",
        "draw_count",
        "draws_per_chain",
        "parameter_count",
        "params",
    ] {
        assert!(matches!(
            trailer_completion.get(field),
            Some(Value::Bool(false))
        ));
    }
}

#[test]
fn diagnose_rejects_invalid_workflow_phases() {
    let fit = [
        r#"{"draws_format":"v0-provisional","workflow_phases":["decode_ir","run_nuts"],"params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"workflow_phases":["decode_ir","run_nuts"],"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header workflow_phases must match the v0-provisional sample workflow; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_invalid_trailer_draws_format() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draws_format":"vNEXT","chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer draws_format must be \"v0-provisional\" when present; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_completion_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draws_format":"v0-provisional","seed":12,"draws_per_chain":4,"params":1,"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer seed must match fit header seed; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_chain_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draws_format":"v0-provisional","chain_count":2,"draws_per_chain":4,"params":1,"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer chain_count must match fit header chains; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_parameter_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draws_format":"v0-provisional","draws_per_chain":4,"parameter_count":2,"params":1,"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer parameter_count must match fit header params length; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_chain_draw_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draws_format":"v0-provisional","draws_per_chain":4,"params":1,"chains":[{"chain":0,"draw_count":5,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer chain draw_count must match fit header settings.num_draws; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_draw_lines_without_draw_index() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(err.message, "each draw line needs an integer draw field");
}

#[test]
fn diagnose_rejects_noncontiguous_draw_indexes() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw index for chain 0 must be 1, got 0; fit draw indexes must be contiguous from 0"
    );
}

#[test]
fn diagnose_rejects_mismatched_global_draw_index_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"draw_index":0,"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"draw_index":2,"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"draw_index":2,"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"draw_index":3,"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line draw_index must be 1, got 2; fit draw_index values must be contiguous from 0 in retained draw order"
    );
}

#[test]
fn diagnose_rejects_mismatched_draw_artifact_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line artifact metadata must include draw_index when present"
    );

    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index":0,"draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"draws_format":"not-v0","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index":1,"draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index":2,"draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"draws_format":"v0-provisional","artifact_kind":"posterior_draws","artifact_scope":"observed_data_conditioned_parameter_draws","draw_index":3,"draw_index_base":"zero_based_retained_draw_order","seed":11,"draw_count":4,"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line draws_format must be \"v0-provisional\" when present; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_draw_chain_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain_count":1,"chain_order":[0],"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain_count":1,"chain_order":[1],"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain_count":1,"chain_order":[0],"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain_count":1,"chain_order":[0],"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line chain_order must match draw chain ids; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_incomplete_draw_parameter_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"parameter_count":1,"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line parameter metadata must include both parameter_count and parameter_order when present"
    );
}

#[test]
fn diagnose_rejects_mismatched_draw_parameter_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]},{"name":"beta","shape":[]}],"packing":["alpha","beta"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"parameter_count":2,"parameter_order":["alpha","beta"],"chain":0,"draw":0,"values":{"alpha":0.0,"beta":1.0}}"#,
        r#"{"parameter_count":2,"parameter_order":["beta","alpha"],"chain":0,"draw":1,"values":{"alpha":1.0,"beta":2.0}}"#,
        r#"{"parameter_count":2,"parameter_order":["alpha","beta"],"chain":0,"draw":2,"values":{"alpha":2.0,"beta":3.0}}"#,
        r#"{"parameter_count":2,"parameter_order":["alpha","beta"],"chain":0,"draw":3,"values":{"alpha":3.0,"beta":4.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line parameter_order must match fit header params order; rerun `bayesite sample` to completion"
    );

    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"parameter_count":2,"parameter_order":["alpha"],"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"parameter_count":1,"parameter_order":["alpha"],"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "draw line parameter_count must match fit header params length; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_negative_chain_ids() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":-1,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":-1,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":-1,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":-1,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":-1,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(err.message, "draw line chain field must be non-negative");

    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":-1,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(err.message, "fit trailer chain field must be non-negative");
}

#[test]
fn diagnose_rejects_trailer_chain_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":1,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer chains must match draw chain ids; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_header_chain_count_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":2}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header chains must match draw chain count; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_header_chain_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1,"chain_count":2}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header chain_count must match fit header chains; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_header_chain_order_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1,"chain_order":[1]}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header chain_order must match draw chain ids; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_chain_order_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chain_order":[1],"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer chain_order must match draw chain ids; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_header_parameter_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"parameter_count":2,"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header parameter_count must match fit header params length; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_header_total_draw_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1,"draw_count":5}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header draw_count must match retained draw line count; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_header_draw_count_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":5,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header settings.num_draws must match draw count per chain; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_total_draw_count_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"draw_count":5,"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer draw_count must match retained draw line count; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_incomplete_trailer_chain_stats() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "each fit trailer chain entry needs an integer divergences"
    );
}

#[test]
fn diagnose_rejects_header_packing_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["beta"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header packing must match params order; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_header_parameter_order_mismatch() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"parameter_order":["beta"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header parameter_order must match params order; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_mismatched_trailer_parameter_order_metadata() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"parameter_order":["beta"],"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit trailer parameter_order must match fit header params order; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_duplicate_header_params() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]},{"name":"alpha","shape":[]}],"packing":["alpha","alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":0.0}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":1.0}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":2.0}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":3.0}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "fit header params has duplicate parameter name \"alpha\"; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnose_rejects_oversized_header_param_shape() {
    let fit = [
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[9223372036854775807,2]}],"packing":["alpha"],"settings":{"num_warmup":0,"num_draws":4,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":1}"#,
        r#"{"chain":0,"draw":0,"values":{"alpha":[]}}"#,
        r#"{"chain":0,"draw":1,"values":{"alpha":[]}}"#,
        r#"{"chain":0,"draw":2,"values":{"alpha":[]}}"#,
        r#"{"chain":0,"draw":3,"values":{"alpha":[]}}"#,
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[4],"step_size":1.0,"mean_accept":0.9}],"rhat":{},"ess":{}}}"#,
    ]
    .join("\n");
    let err = diagnose_ndjson(&fit).unwrap_err();
    assert_eq!(err.kind, ErrorKind::MalformedDocument);
    assert_eq!(
        err.message,
        "parameter alpha shape size is too large for this build; rerun `bayesite sample` to completion"
    );
}

#[test]
fn diagnostics_command_returns_rhat_and_ess() {
    let request = r#"{"command": "diagnostics",
        "series": [[0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0, -0.2],
                   [0.2, 0.1, -0.4, 0.3, 0.0, -0.1, 0.2, 0.1]]}"#;
    let response = json::parse(&handle_request(request)).unwrap();
    assert!(response.get("rhat").and_then(Value::as_f64).unwrap() > 0.5);
    assert!(response.get("ess").and_then(Value::as_f64).unwrap() > 1.0);
}

#[test]
fn diagnostics_command_rejects_malformed_series_shape() {
    let request = r#"{"command": "diagnostics",
        "series": [[0.1, -0.3, 0.5, 0.2], [0.2, 0.1, -0.4]]}"#;
    let response = json::parse(&handle_request(request)).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("each series chain needs at least 4 draws")
    );

    let request = r#"{"command": "diagnostics",
        "series": [[0.1, -0.3, 0.5, 0.2], [0.2, 0.1, -0.4, 0.3, 0.0]]}"#;
    let response = json::parse(&handle_request(request)).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("series chains must all have the same number of draws")
    );
}

#[test]
fn prior_predictive_command_returns_ndjson() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("prior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        (
            "data".to_string(),
            fixture_declared_data("linear_regression", &["x"]),
        ),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 2}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(13)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 1 + 2 + 1);
    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header
            .get("prior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        header.get("artifact_kind").and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        header.get("artifact_scope").and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        header
            .get("settings")
            .and_then(|value| value.get("num_draws"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(2));
    assert_eq!(
        header.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(header.get("site_count").and_then(Value::as_i64), Some(4));
    assert_eq!(
        header.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_prior_predictive_workflow_phases(&header, "workflow_phases");
    assert_eq!(
        string_array(header.get("declared_data_order").unwrap()),
        ["x"]
    );
    assert_eq!(
        header
            .get("declared_data")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    assert_eq!(
        header
            .get("declared_data_shapes")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .and_then(|shape| shape.first())
            .and_then(Value::as_i64),
        Some(5)
    );
    assert!(matches!(
        header.get("declared_data_integer").and_then(|v| v.get("x")),
        Some(Value::Bool(false))
    ));
    let declared_integer_by_coordinate = header
        .get("declared_data_integer_by_coordinate")
        .and_then(|v| v.get("x"))
        .and_then(Value::as_array)
        .expect("declared x integer flags");
    assert_eq!(declared_integer_by_coordinate.len(), 5);
    assert!(declared_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
    assert_eq!(
        coordinate_order(
            header
                .get("declared_data_coordinate_order")
                .and_then(|v| v.get("x"))
                .expect("declared x coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    let sites = header.get("sites").and_then(Value::as_array).unwrap();
    assert_eq!(
        string_array(header.get("site_order").expect("site order")),
        ["alpha", "beta", "sigma", "y"]
    );
    assert!(matches!(
        sites
            .iter()
            .find(|site| site.get("name").and_then(Value::as_str) == Some("y"))
            .and_then(|site| site.get("integer")),
        Some(Value::Bool(false))
    ));
    let y_site = sites
        .iter()
        .find(|site| site.get("name").and_then(Value::as_str) == Some("y"))
        .expect("y site");
    let y_integer_by_coordinate: Vec<bool> = y_site
        .get("integer_by_coordinate")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| match v {
            Value::Bool(flag) => *flag,
            _ => panic!("integer_by_coordinate entries must be booleans"),
        })
        .collect();
    assert_eq!(y_integer_by_coordinate, [false; 5]);
    assert_eq!(
        coordinate_order(y_site.get("coordinate_order").unwrap()),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    let first = json::parse(lines[1]).unwrap();
    assert_eq!(
        first.get("prior_predictive_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        first.get("artifact_kind").and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first.get("artifact_scope").and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(first.get("draw_index").and_then(Value::as_i64), Some(0));
    assert_eq!(first.get("seed").and_then(Value::as_i64), Some(13));
    assert_eq!(
        first.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(first.get("draw_count").and_then(Value::as_i64), Some(2));
    assert_eq!(first.get("site_count").and_then(Value::as_i64), Some(4));
    assert_eq!(
        first.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            first
                .get("declared_data_order")
                .expect("draw declared data order")
        ),
        ["x"]
    );
    assert_eq!(
        string_array(first.get("site_order").expect("draw site order")),
        ["alpha", "beta", "sigma", "y"]
    );
    assert!(first.get("values").is_some());
    let second = json::parse(lines[2]).unwrap();
    assert_eq!(
        second
            .get("prior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(second.get("draw_index").and_then(Value::as_i64), Some(1));
    assert_eq!(second.get("seed").and_then(Value::as_i64), Some(13));
    assert_eq!(
        second.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        second.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            second
                .get("declared_data_order")
                .expect("second draw declared data order")
        ),
        ["x"]
    );
    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("artifact_kind"))
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("artifact_scope"))
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|value| value.get("settings"))
            .and_then(|value| value.get("num_draws"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("draw_count"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("draw_index_base"))
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("seed"))
            .and_then(Value::as_i64),
        Some(13)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("sites"))
            .and_then(Value::as_i64),
        Some(4)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("site_count"))
            .and_then(Value::as_i64),
        Some(4)
    );
    assert_eq!(
        string_array(
            trailer
                .get("trailer")
                .and_then(|v| v.get("site_order"))
                .expect("trailer site order")
        ),
        ["alpha", "beta", "sigma", "y"]
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("declared_data_count"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            trailer
                .get("trailer")
                .and_then(|v| v.get("declared_data_order"))
                .expect("trailer declared data order")
        ),
        ["x"]
    );
    assert_prior_predictive_workflow_phases(
        trailer.get("trailer").expect("trailer"),
        "workflow_phases",
    );
}

#[test]
fn prior_predictive_integer_sites_emit_integer_values() {
    let fixture = json::parse(&fixture_text("bounded_rates")).unwrap();
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("prior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), Value::Object(vec![])),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 1}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(17)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 3);
    let header = json::parse(lines[0]).unwrap();
    let sites = header.get("sites").and_then(Value::as_array).unwrap();
    let y_site = sites
        .iter()
        .find(|site| site.get("name").and_then(Value::as_str) == Some("y"))
        .expect("y site");
    assert!(matches!(y_site.get("integer"), Some(Value::Bool(true))));
    assert!(matches!(
        y_site.get("integer_by_coordinate"),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        coordinate_order(y_site.get("coordinate_order").unwrap()),
        vec![Vec::<i64>::new()]
    );

    let draw = json::parse(lines[1]).unwrap();
    assert!(matches!(
        draw.get("values").and_then(|values| values.get("y")),
        Some(Value::Int(0 | 1))
    ));
}

#[test]
fn prior_predictive_declared_integer_data_emit_integer_values() {
    let fixture = json::parse(&fixture_text("varying_intercepts_poisson")).unwrap();
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("prior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        (
            "data".to_string(),
            fixture_declared_data(
                "varying_intercepts_poisson",
                &["n_groups", "group_idx", "x"],
            ),
        ),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 1}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(19)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let header = json::parse(response.lines().next().unwrap()).unwrap();
    assert_eq!(
        string_array(header.get("declared_data_order").unwrap()),
        ["n_groups", "group_idx", "x"]
    );
    assert!(matches!(
        header
            .get("declared_data")
            .and_then(|data| data.get("n_groups")),
        Some(Value::Int(3))
    ));
    let group_idx = header
        .get("declared_data")
        .and_then(|data| data.get("group_idx"))
        .and_then(Value::as_array)
        .expect("group_idx values");
    assert_eq!(group_idx.len(), 6);
    assert!(group_idx.iter().all(|value| matches!(value, Value::Int(_))));
    assert!(matches!(
        header
            .get("declared_data")
            .and_then(|data| data.get("x"))
            .and_then(Value::as_array)
            .and_then(|values| values.first()),
        Some(Value::Float(_))
    ));
    assert!(matches!(
        header
            .get("declared_data_integer")
            .and_then(|data| data.get("n_groups")),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        header
            .get("declared_data_integer")
            .and_then(|data| data.get("group_idx")),
        Some(Value::Bool(true))
    ));
}

#[test]
fn prior_predictive_sites_report_source_stochastic_site() {
    let mut fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let ir = object_entry_mut(&mut fixture, "ir");
    let model = object_entry_mut(ir, "model");
    let sites = object_entry_mut(model, "stochastic_sites");
    let first_site = match sites {
        Value::Array(sites) => sites.first_mut().expect("fixture has stochastic sites"),
        _ => panic!("stochastic_sites must be an array"),
    };
    *object_entry_mut(first_site, "name") = Value::Str("alpha_prior_factor".to_string());

    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("prior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        (
            "data".to_string(),
            fixture_declared_data("linear_regression", &["x"]),
        ),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 1}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(13)),
    ]);
    let response = handle_request(&json::write(&request).unwrap());
    let header = json::parse(response.lines().next().unwrap()).unwrap();
    let alpha = header
        .get("sites")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .find(|site| site.get("name").and_then(Value::as_str) == Some("alpha"))
        .expect("alpha site");
    assert_eq!(
        alpha.get("stochastic_site").and_then(Value::as_str),
        Some("alpha_prior_factor")
    );
}

#[test]
fn recover_command_returns_factual_report() {
    let fixture = json::parse(&fixture_text("bounded_rates")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("recover".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), Value::Object(vec![])),
        (
            "settings".to_string(),
            json::parse(
                r#"{"chains": 1, "interval": 0.8, "num_warmup": 20,
                    "num_draws": 20, "max_treedepth": 4}"#,
            )
            .unwrap(),
        ),
        ("seed".to_string(), Value::Int(23)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("recover_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("report_kind").and_then(Value::as_str),
        Some("parameter_recovery_facts")
    );
    assert_eq!(
        response.get("report_scope").and_then(Value::as_str),
        Some("single_simulated_dataset")
    );
    assert_eq!(
        response.get("simulation_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("simulation_index_base")
            .and_then(Value::as_str),
        Some("zero_based_simulation_order")
    );
    assert_eq!(
        int_array(response.get("simulation_order").expect("simulation order")),
        [0]
    );
    assert_eq!(
        response
            .get("prior_predictive_draws")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("prior_predictive_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        response
            .get("prior_predictive_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_workflow_phases(&response, "workflow_phases");
    let recover_seed_schedule = response.get("seed_schedule").expect("seed schedule");
    assert_eq!(
        recover_seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        recover_seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        recover_seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        recover_seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert!(string_array(response.get("declared_data_order").unwrap()).is_empty());
    assert_eq!(
        response.get("declared_data_count").and_then(Value::as_i64),
        Some(0)
    );
    assert!(response.get("generated_observed").is_some());
    assert!(matches!(
        response.get("generated_observed").and_then(|v| v.get("y")),
        Some(Value::Int(0 | 1))
    ));
    assert!(response
        .get("generated_observed_shapes")
        .and_then(|v| v.get("y"))
        .and_then(Value::as_array)
        .is_some());
    assert!(matches!(
        response
            .get("generated_observed_integer")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        response
            .get("generated_observed_integer_by_coordinate")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        coordinate_order(
            response
                .get("generated_observed_coordinate_order")
                .and_then(|v| v.get("y"))
                .expect("generated y coordinate order")
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        string_array(response.get("generated_observed_order").unwrap()),
        ["y"]
    );
    assert_eq!(
        response
            .get("generated_observed_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        response
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        response
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        response
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        response.get("posterior_draws").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        response
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        response
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        response.get("interval_method").and_then(Value::as_str),
        Some("equal_tailed_linear_quantile")
    );
    assert_eq!(
        response.get("interval_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        response
            .get("interval_contains_truth_statistic")
            .and_then(Value::as_str),
        Some("truth_within_closed_interval_all_coordinates")
    );
    assert_eq!(
        response.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        response.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(response.get("rank_draws").and_then(Value::as_i64), Some(20));
    assert_eq!(
        response.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(&response, "tie_count", 20);
    assert_eq!(
        response
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    let rank_bounds = response.get("rank_bounds").expect("rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
    assert_eq!(
        int_array(response.get("rank_bin_order").expect("rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        response.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    let sampler_summary = response.get("sampler_summary").expect("sampler summary");
    assert_eq!(
        sampler_summary.get("chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        sampler_summary.get("draw_count").and_then(Value::as_i64),
        Some(20)
    );
    assert!(sampler_summary
        .get("total_divergences")
        .and_then(Value::as_i64)
        .is_some());
    let treedepth_histogram = int_array(
        sampler_summary
            .get("treedepth_histogram")
            .expect("treedepth histogram"),
    );
    assert_treedepth_support(sampler_summary, 4);
    assert_eq!(treedepth_histogram.len(), 5);
    assert_eq!(treedepth_histogram.iter().sum::<i64>(), 20);
    let chains = response
        .get("chains")
        .and_then(Value::as_array)
        .expect("chain stats");
    assert_eq!(response.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(
        chains.len(),
        response.get("chain_count").and_then(Value::as_i64).unwrap() as usize
    );
    assert_treedepth_support(&chains[0], 4);
    assert_eq!(
        chains[0].get("chain_index_base").and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        chains[0].get("draw_count").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(int_array(response.get("chain_order").unwrap()), [0]);
    let interval_bounds = response.get("interval_bounds").expect("interval bounds");
    let interval_probability = interval_bounds
        .get("interval_probability")
        .and_then(Value::as_f64)
        .unwrap();
    let lower_tail_probability = interval_bounds
        .get("lower_tail_probability")
        .and_then(Value::as_f64)
        .unwrap();
    let upper_tail_probability = interval_bounds
        .get("upper_tail_probability")
        .and_then(Value::as_f64)
        .unwrap();
    assert!((interval_probability - 0.8).abs() < 1e-12);
    assert!((lower_tail_probability - 0.1).abs() < 1e-12);
    assert!((upper_tail_probability - 0.1).abs() < 1e-12);
    assert!(interval_bounds.get("lower_quantile").is_some());
    assert!(interval_bounds.get("upper_quantile").is_some());
    assert_interval_quantile_index_metadata(interval_bounds, 20, 0.8);
    let parameters = response.get("parameters").expect("parameters");
    assert_eq!(
        response
            .get("parameter_report_count")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        object_keys(parameters).len(),
        response
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    let interval_contains_truth_by_parameter = response
        .get("interval_contains_truth_by_parameter")
        .expect("interval containment by parameter");
    assert_eq!(
        object_keys(interval_contains_truth_by_parameter),
        ["p", "level"]
    );
    assert!(
        response.get("interval_contains_truth").is_none(),
        "recover should not collapse parameter facts into a report verdict"
    );
    for name in ["p", "level"] {
        let contains = parameters
            .get(name)
            .and_then(|value| value.get("interval_contains_truth"))
            .expect("parameter interval containment");
        assert!(matches!(contains, Value::Bool(_)));
        assert_eq!(
            interval_contains_truth_by_parameter.get(name),
            Some(contains)
        );
    }
    assert_eq!(
        string_array(response.get("parameter_order").unwrap()),
        ["p", "level"]
    );
    assert_eq!(
        response.get("parameter_count").and_then(Value::as_i64),
        Some(2)
    );
    assert!(parameters
        .get("p")
        .and_then(|v| v.get("shape"))
        .and_then(Value::as_array)
        .is_some());
    assert_eq!(
        coordinate_order(
            parameters
                .get("p")
                .and_then(|value| value.get("coordinate_order"))
                .unwrap()
        ),
        vec![Vec::<i64>::new()]
    );
    assert!(parameters
        .get("p")
        .and_then(|v| v.get("rank"))
        .and_then(Value::as_i64)
        .is_some());
    assert_eq!(
        interval_contains_truth_by_parameter.get("p"),
        parameters
            .get("p")
            .and_then(|v| v.get("interval_contains_truth"))
    );
    assert!(matches!(
        interval_contains_truth_by_parameter.get("p"),
        Some(Value::Bool(_))
    ));
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rank_draws"))
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("posterior_draws"))
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("posterior_draws_artifact_kind"))
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("posterior_draws_artifact_scope"))
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("truth_artifact_kind"))
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("truth_artifact_scope"))
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("truth_draw_index"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("truth_draw_index_base"))
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("simulation"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("simulation_index_base"))
            .and_then(Value::as_str),
        Some("zero_based_simulation_order")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("prior_seed"))
            .and_then(Value::as_i64),
        Some(23)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("sample_seed"))
            .and_then(Value::as_i64),
        Some(24)
    );
    let seed_schedule = parameters
        .get("p")
        .and_then(|v| v.get("seed_schedule"))
        .expect("recover parameter seed schedule");
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    let parameter_rank_bounds = parameters
        .get("p")
        .and_then(|v| v.get("rank_bounds"))
        .expect("parameter rank bounds");
    assert_eq!(
        parameter_rank_bounds.get("min").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        parameter_rank_bounds.get("max").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        int_array(
            parameters
                .get("p")
                .and_then(|v| v.get("rank_bin_order"))
                .expect("parameter rank bin order")
        ),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rank_bin_count"))
            .and_then(Value::as_i64),
        Some(21)
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("interval_method"))
            .and_then(Value::as_str),
        Some("equal_tailed_linear_quantile")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("interval_scope"))
            .and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("interval_contains_truth_statistic"))
            .and_then(Value::as_str),
        Some("truth_within_closed_interval_all_coordinates")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rank_statistic"))
            .and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rank_scope"))
            .and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("tie_statistic"))
            .and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(parameters.get("p").expect("p summary"), "tie_count", 20);
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rhat_statistic"))
            .and_then(Value::as_str),
        Some("split_rhat")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("rhat_scope"))
            .and_then(Value::as_str),
        Some("max_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("ess_statistic"))
            .and_then(Value::as_str),
        Some("effective_sample_size_geyer_initial_monotone_sequence")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("ess_scope"))
            .and_then(Value::as_str),
        Some("min_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        parameters
            .get("p")
            .and_then(|v| v.get("summary_scale"))
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    let parameter_interval_bounds = parameters
        .get("p")
        .and_then(|v| v.get("interval_bounds"))
        .expect("parameter interval bounds");
    let parameter_interval_probability = parameter_interval_bounds
        .get("interval_probability")
        .and_then(Value::as_f64)
        .unwrap();
    let parameter_lower_tail_probability = parameter_interval_bounds
        .get("lower_tail_probability")
        .and_then(Value::as_f64)
        .unwrap();
    let parameter_upper_tail_probability = parameter_interval_bounds
        .get("upper_tail_probability")
        .and_then(Value::as_f64)
        .unwrap();
    assert!((parameter_interval_probability - 0.8).abs() < 1e-12);
    assert!((parameter_lower_tail_probability - 0.1).abs() < 1e-12);
    assert!((parameter_upper_tail_probability - 0.1).abs() < 1e-12);
    assert_interval_quantile_index_metadata(parameter_interval_bounds, 20, 0.8);
    assert!(parameters
        .get("p")
        .and_then(|v| v.get("tie_count"))
        .and_then(Value::as_i64)
        .is_some());
    assert!(response.get("success").is_none());
    assert!(response.get("verdict").is_none());
    assert!(response.get("coverage").is_none());
    assert!(response.get("interpretation").is_none());
    assert!(response.get("recommendation").is_none());
}

#[test]
fn workflow_commands_declared_integer_data_are_json_integers() {
    let fixture = json::parse(&fixture_text("varying_intercepts_poisson")).unwrap();
    let data = fixture_declared_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("recover".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data.clone()),
        (
            "settings".to_string(),
            json::parse(
                r#"{"chains": 2, "interval": 0.8, "num_warmup": 20,
                    "num_draws": 20, "max_treedepth": 4}"#,
            )
            .unwrap(),
        ),
        ("seed".to_string(), Value::Int(23)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("recover_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_varying_intercepts_declared_data_values(&response);
    assert_z_alpha_recover_interval_contains_truth_by_coordinate(&response);

    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sbc".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), data),
        (
            "settings".to_string(),
            json::parse(
                r#"{"replicates": 1, "chains": 2, "num_warmup": 20,
                    "num_draws": 20, "max_treedepth": 4}"#,
            )
            .unwrap(),
        ),
        ("seed".to_string(), Value::Int(29)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_varying_intercepts_declared_data_values(&response);
    assert_z_alpha_sbc_rank_histograms(&response);
}

#[test]
fn sbc_command_returns_rank_report_without_verdict() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sbc".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        (
            "data".to_string(),
            fixture_declared_data("linear_regression", &["x"]),
        ),
        (
            "settings".to_string(),
            json::parse(
                r#"{"replicates": 1, "chains": 1, "num_warmup": 20,
                    "num_draws": 20, "max_treedepth": 4}"#,
            )
            .unwrap(),
        ),
        ("seed".to_string(), Value::Int(29)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("report_kind").and_then(Value::as_str),
        Some("simulation_based_calibration_rank_facts")
    );
    assert_eq!(
        response.get("report_scope").and_then(Value::as_str),
        Some("replicated_simulated_datasets")
    );
    assert_eq!(response.get("replicates").and_then(Value::as_i64), Some(1));
    assert_eq!(
        response.get("replicate_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("replicate_report_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response.get("replicate_index_base").and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    assert_eq!(
        response
            .get("prior_predictive_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("prior_predictive_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        response
            .get("prior_predictive_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        response
            .get("generated_observed_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            response
                .get("generated_observed_order_per_replicate")
                .expect("generated observed order per replicate")
        ),
        ["y"]
    );
    assert_eq!(
        response
            .get("generated_observed_artifact_kind_per_replicate")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        response
            .get("generated_observed_artifact_scope_per_replicate")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        response
            .get("generated_observed_draw_index_per_replicate")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        response
            .get("generated_observed_draw_index_base_per_replicate")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        response
            .get("settings")
            .and_then(|settings| settings.get("replicates"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(int_array(response.get("replicate_order").unwrap()), [0]);
    assert_eq!(response.get("rank_draws").and_then(Value::as_i64), Some(20));
    assert_eq!(
        response
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        response
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        response
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        int_array(response.get("rank_bin_order").expect("rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        response.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    assert_count_support(&response, "tie_count", 20);
    assert_workflow_phases(&response, "replicate_workflow_phases");
    assert_eq!(
        string_array(response.get("declared_data_order").unwrap()),
        ["x"]
    );
    assert_eq!(
        response.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(response.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    assert_eq!(
        response.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    let aggregate_parameters = response
        .get("parameters")
        .expect("aggregate parameter summaries");
    assert_eq!(
        response
            .get("parameter_report_count")
            .and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        object_keys(aggregate_parameters).len(),
        response
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    assert_eq!(
        response
            .get("declared_data")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    assert_eq!(
        response
            .get("declared_data_shapes")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .and_then(|shape| shape.first())
            .and_then(Value::as_i64),
        Some(5)
    );
    assert!(matches!(
        response
            .get("declared_data_integer")
            .and_then(|v| v.get("x")),
        Some(Value::Bool(false))
    ));
    let declared_integer_by_coordinate = response
        .get("declared_data_integer_by_coordinate")
        .and_then(|v| v.get("x"))
        .and_then(Value::as_array)
        .expect("declared x integer flags");
    assert_eq!(declared_integer_by_coordinate.len(), 5);
    assert!(declared_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
    let reports = response
        .get("replicate_reports")
        .and_then(Value::as_array)
        .expect("replicate reports");
    assert_eq!(
        reports.len(),
        response
            .get("replicate_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    let first_report = reports.first().expect("first replicate report");
    assert_eq!(
        first_report.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        first_report.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        first_report.get("report_kind").and_then(Value::as_str),
        Some("simulation_based_calibration_replicate_rank_facts")
    );
    assert_eq!(
        first_report.get("report_scope").and_then(Value::as_str),
        Some("single_simulated_dataset_replicate")
    );
    assert_workflow_phases(first_report, "workflow_phases");
    assert_eq!(
        first_report
            .get("declared_data_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(first_report.get("declared_data_order").unwrap()),
        ["x"]
    );
    let seed_schedule = first_report
        .get("seed_schedule")
        .expect("replicate seed schedule");
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_report
            .get("replicate_index_base")
            .and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    assert_eq!(
        first_report.get("replicate_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(int_array(first_report.get("replicate_order").unwrap()), [0]);
    assert_eq!(
        first_report
            .get("settings")
            .and_then(|settings| settings.get("chains"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_report
            .get("settings")
            .and_then(|settings| settings.get("num_warmup"))
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        first_report
            .get("settings")
            .and_then(|settings| settings.get("num_draws"))
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        first_report
            .get("settings")
            .and_then(|settings| settings.get("max_treedepth"))
            .and_then(Value::as_i64),
        Some(4)
    );
    assert!(
        (first_report
            .get("settings")
            .and_then(|settings| settings.get("target_accept"))
            .and_then(Value::as_f64)
            .unwrap()
            - 0.8)
            .abs()
            < 1e-12
    );
    assert_eq!(
        first_report.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    let first_parameters = first_report
        .get("parameters")
        .expect("first replicate parameter summaries");
    assert_eq!(
        first_report
            .get("parameter_report_count")
            .and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        object_keys(first_parameters).len(),
        first_report
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    assert_eq!(
        first_report
            .get("generated_observed_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_report.get("rank_draws").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        first_report
            .get("prior_predictive_draws")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_report
            .get("prior_predictive_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first_report
            .get("prior_predictive_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        first_report.get("posterior_draws").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        first_report
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        first_report
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    let replicate_rank_bounds = first_report
        .get("rank_bounds")
        .expect("replicate rank bounds");
    assert_eq!(
        replicate_rank_bounds.get("min").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        replicate_rank_bounds.get("max").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        int_array(
            first_report
                .get("rank_bin_order")
                .expect("replicate rank bin order")
        ),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        first_report.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    assert_count_support(first_report, "tie_count", 20);
    assert_eq!(
        first_report
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        first_report.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        first_report.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        first_report.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_eq!(
        string_array(first_report.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let alpha = first_parameters
        .get("alpha")
        .expect("first replicate alpha");
    assert_eq!(
        alpha
            .get("shape")
            .and_then(Value::as_array)
            .map(|shape| shape.len()),
        Some(0)
    );
    assert_eq!(alpha.get("rank_draws").and_then(Value::as_i64), Some(20));
    assert_eq!(
        alpha.get("posterior_draws").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        alpha
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        alpha
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        alpha.get("truth_artifact_kind").and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        alpha.get("truth_artifact_scope").and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        alpha.get("truth_draw_index").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        alpha.get("truth_draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        alpha.get("prior_seed").and_then(Value::as_i64),
        first_report.get("prior_seed").and_then(Value::as_i64)
    );
    assert_eq!(
        alpha.get("sample_seed").and_then(Value::as_i64),
        first_report.get("sample_seed").and_then(Value::as_i64)
    );
    assert_eq!(
        alpha.get("replicate").and_then(Value::as_i64),
        first_report.get("replicate").and_then(Value::as_i64)
    );
    assert_eq!(
        alpha.get("replicate_index_base").and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    let seed_schedule = alpha
        .get("seed_schedule")
        .expect("sbc replicate parameter seed schedule");
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    let alpha_rank_bounds = alpha.get("rank_bounds").expect("alpha rank bounds");
    assert_eq!(
        alpha_rank_bounds.get("min").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        alpha_rank_bounds.get("max").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        int_array(alpha.get("rank_bin_order").expect("alpha rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        alpha.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    assert_eq!(
        alpha.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        alpha.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        alpha.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(alpha, "tie_count", 20);
    assert_eq!(
        alpha.get("summary_scale").and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        alpha.get("rhat_statistic").and_then(Value::as_str),
        Some("split_rhat")
    );
    assert_eq!(
        alpha.get("rhat_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        alpha.get("ess_statistic").and_then(Value::as_str),
        Some("effective_sample_size_geyer_initial_monotone_sequence")
    );
    assert_eq!(
        alpha.get("ess_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        first_report
            .get("generated_observed_integer")
            .and_then(|v| v.get("y"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    let generated_integer_by_coordinate = first_report
        .get("generated_observed_integer_by_coordinate")
        .and_then(|v| v.get("y"))
        .and_then(Value::as_array)
        .expect("generated y integer-by-coordinate flags");
    assert_eq!(generated_integer_by_coordinate.len(), 5);
    assert!(generated_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
    assert_eq!(
        coordinate_order(
            first_report
                .get("generated_observed_coordinate_order")
                .and_then(|v| v.get("y"))
                .expect("generated y coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(
        string_array(first_report.get("generated_observed_order").unwrap()),
        ["y"]
    );
    assert_eq!(
        first_report
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first_report
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        first_report
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        first_report
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        response.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        response.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        response.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    let aggregate_alpha = aggregate_parameters.get("alpha").expect("aggregate alpha");
    assert_eq!(
        aggregate_alpha
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        aggregate_alpha
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        aggregate_alpha
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    let seed_schedule = response.get("seed_schedule").expect("seed schedule");
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("prior_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("base_seed"))
            .and_then(Value::as_str),
        Some("seed")
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("replicate_multiplier"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        seed_schedule
            .get("sample_seed")
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        response
            .get("rank_bounds")
            .and_then(|bounds| bounds.get("max"))
            .and_then(Value::as_i64),
        Some(20)
    );
    let sampler_summary = response.get("sampler_summary").expect("sampler summary");
    assert_eq!(
        response
            .get("chain_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        sampler_summary.get("chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        sampler_summary.get("draw_count").and_then(Value::as_i64),
        Some(20)
    );
    assert!(sampler_summary
        .get("total_divergences")
        .and_then(Value::as_i64)
        .is_some());
    let treedepth_histogram = int_array(
        sampler_summary
            .get("treedepth_histogram")
            .expect("treedepth histogram"),
    );
    assert_treedepth_support(sampler_summary, 4);
    assert_eq!(treedepth_histogram.len(), 5);
    assert_eq!(treedepth_histogram.iter().sum::<i64>(), 20);
    assert!(response.get("success").is_none());
    assert!(response.get("uniformity").is_none());
    assert!(response.get("verdict").is_none());
    let replicate_sampler_summary = first_report
        .get("sampler_summary")
        .expect("replicate sampler summary");
    assert_eq!(
        replicate_sampler_summary
            .get("chain_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        replicate_sampler_summary
            .get("draw_count")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert!(replicate_sampler_summary
        .get("total_divergences")
        .and_then(Value::as_i64)
        .is_some());
    let replicate_treedepth_histogram = int_array(
        replicate_sampler_summary
            .get("treedepth_histogram")
            .expect("replicate treedepth histogram"),
    );
    assert_treedepth_support(replicate_sampler_summary, 4);
    assert_eq!(replicate_treedepth_histogram.len(), 5);
    assert_eq!(replicate_treedepth_histogram.iter().sum::<i64>(), 20);
    let chains = first_report
        .get("chains")
        .and_then(Value::as_array)
        .expect("replicate chain stats");
    assert_eq!(
        first_report.get("chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        chains.len(),
        first_report
            .get("chain_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    assert_treedepth_support(&chains[0], 4);
    assert_eq!(
        chains[0].get("chain_index_base").and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        chains[0].get("draw_count").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(int_array(first_report.get("chain_order").unwrap()), [0]);
    assert!(first_report
        .get("parameters")
        .and_then(|parameters| parameters.get("alpha"))
        .and_then(|parameter| parameter.get("shape"))
        .and_then(Value::as_array)
        .is_some_and(|shape| shape.is_empty()));
    assert!(first_report
        .get("parameters")
        .and_then(|parameters| parameters.get("alpha"))
        .and_then(|parameter| parameter.get("tie_count"))
        .and_then(Value::as_i64)
        .is_some());
    assert_eq!(
        coordinate_order(
            response
                .get("parameters")
                .and_then(|parameters| parameters.get("alpha"))
                .and_then(|parameter| parameter.get("coordinate_order"))
                .unwrap()
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        int_array(
            response
                .get("parameters")
                .and_then(|parameters| parameters.get("alpha"))
                .and_then(|parameter| parameter.get("rank_bin_order"))
                .expect("alpha rank bin order")
        ),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("rank_bin_count"))
            .and_then(Value::as_i64),
        Some(21)
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("truth"))
            .and_then(Value::as_array)
            .map(|truth| truth.len()),
        Some(1)
    );
    assert!(response
        .get("parameters")
        .and_then(|parameters| parameters.get("alpha"))
        .and_then(|parameter| parameter.get("truth"))
        .and_then(Value::as_array)
        .and_then(|truth| truth.first())
        .and_then(Value::as_f64)
        .is_some());
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("truth_artifact_kind"))
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("truth_artifact_scope"))
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("truth_draw_index_base"))
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        int_array(
            response
                .get("parameters")
                .and_then(|parameters| parameters.get("alpha"))
                .and_then(|parameter| parameter.get("truth_draw_index"))
                .expect("alpha truth draw index")
        ),
        [0]
    );
    assert_eq!(
        int_array(
            response
                .get("parameters")
                .and_then(|parameters| parameters.get("alpha"))
                .and_then(|parameter| parameter.get("replicate_order"))
                .expect("alpha replicate order")
        ),
        [0]
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("rank_statistic"))
            .and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("rank_scope"))
            .and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("tie_statistic"))
            .and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_eq!(
        response
            .get("parameters")
            .and_then(|parameters| parameters.get("alpha"))
            .and_then(|parameter| parameter.get("summary_scale"))
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        coordinate_order(
            first_report
                .get("parameters")
                .and_then(|parameters| parameters.get("alpha"))
                .and_then(|parameter| parameter.get("coordinate_order"))
                .unwrap()
        ),
        vec![Vec::<i64>::new()]
    );
    assert!(response.get("success").is_none());
    assert!(response.get("uniformity").is_none());
}

#[test]
fn sbc_command_integer_generated_observed_values_are_json_integers() {
    let fixture = json::parse(&fixture_text("bounded_rates")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sbc".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), Value::Object(vec![])),
        (
            "settings".to_string(),
            json::parse(
                r#"{"replicates": 1, "chains": 1, "num_warmup": 20,
                    "num_draws": 20, "max_treedepth": 4}"#,
            )
            .unwrap(),
        ),
        ("seed".to_string(), Value::Int(29)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    let first_report = response
        .get("replicate_reports")
        .and_then(Value::as_array)
        .and_then(|reports| reports.first())
        .expect("first replicate report");
    assert!(matches!(
        first_report
            .get("generated_observed")
            .and_then(|values| values.get("y")),
        Some(Value::Int(0 | 1))
    ));
    assert!(matches!(
        first_report
            .get("generated_observed_integer")
            .and_then(|values| values.get("y")),
        Some(Value::Bool(true))
    ));
}

#[test]
fn workflow_requests_reject_unknown_fields() {
    let fixture = json::parse(&fixture_text("bounded_rates")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("recover".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), Value::Object(vec![])),
        ("seed".to_string(), Value::Int(23)),
        (
            "setting".to_string(),
            json::parse(r#"{"num_draws": 20}"#).unwrap(),
        ),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("recover request has unknown field \"setting\"")
    );

    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sbc".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), Value::Object(vec![])),
        ("seed".to_string(), Value::Int(29)),
        (
            "settings".to_string(),
            json::parse(r#"{"replicates": 1, "draw_count": 20}"#).unwrap(),
        ),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("sbc request settings has unknown field \"draw_count\"")
    );
}

#[test]
fn workflow_requests_report_data_field_shape() {
    let fixture = json::parse(&fixture_text("bounded_rates")).unwrap();
    for (command, expected) in [
        ("recover", "recover request data must be an object"),
        ("sbc", "sbc request data must be an object"),
    ] {
        let request = Value::Object(vec![
            ("command".to_string(), Value::Str(command.to_string())),
            ("model".to_string(), fixture.get("ir").unwrap().clone()),
            ("data".to_string(), Value::Array(vec![])),
            ("seed".to_string(), Value::Int(23)),
        ]);
        let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected),
            "{command}"
        );
    }
}

#[test]
fn protocol_requests_reject_duplicate_control_fields() {
    let request = r#"{"command": "diagnostics",
        "series": [[0.1, -0.3, 0.5, 0.2], [0.2, 0.1, -0.4, 0.3]],
        "series": [[0.0, 0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.0]]}"#;
    let response = json::parse(&handle_request(request)).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("diagnostics request has duplicate field \"series\"; remove one")
    );
}

#[test]
fn artifact_requests_reject_unreportable_seed_values() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_data = json::write(linear.get("data").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();

    for (context, request) in [
        (
            "sample",
            format!(
                r#"{{"command":"sample","model":{linear_model},"data":{linear_data},"settings":{{"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":9223372036854775808}}"#
            ),
        ),
        (
            "prior-predictive",
            format!(
                r#"{{"command":"prior-predictive","model":{linear_model},"data":{linear_declared_data},"settings":{{"num_draws":1}},"seed":9223372036854775808}}"#
            ),
        ),
        (
            "recover",
            format!(
                r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":9223372036854775808}}"#
            ),
        ),
        (
            "sbc",
            format!(
                r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":1,"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":9223372036854775808}}"#
            ),
        ),
    ] {
        let response = json::parse(&handle_request(&request)).unwrap();
        let expected = format!(
            "{context} request seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers"
        );
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{context}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected.as_str()),
            "{context}"
        );
    }
}

#[test]
fn artifact_requests_reject_unreportable_draw_counts() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_data = json::write(linear.get("data").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();

    for (context, request) in [
        (
            "sample",
            format!(
                r#"{{"command":"sample","model":{linear_model},"data":{linear_data},"settings":{{"num_warmup":0,"num_draws":9223372036854775808,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
        (
            "prior-predictive",
            format!(
                r#"{{"command":"prior-predictive","model":{linear_model},"data":{linear_declared_data},"settings":{{"num_draws":9223372036854775808}},"seed":5}}"#
            ),
        ),
        (
            "recover",
            format!(
                r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"chains":1,"num_warmup":0,"num_draws":9223372036854775808,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
        (
            "sbc",
            format!(
                r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":1,"chains":1,"num_warmup":0,"num_draws":9223372036854775808,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
    ] {
        let response = json::parse(&handle_request(&request)).unwrap();
        let expected = format!(
            "{context} request settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers"
        );
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{context}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected.as_str()),
            "{context}"
        );
    }
}

#[test]
fn artifact_requests_reject_unreportable_warmup_counts() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_data = json::write(linear.get("data").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();

    for (context, request) in [
        (
            "sample",
            format!(
                r#"{{"command":"sample","model":{linear_model},"data":{linear_data},"settings":{{"num_warmup":9223372036854775808,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
        (
            "recover",
            format!(
                r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"chains":1,"num_warmup":9223372036854775808,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
        (
            "sbc",
            format!(
                r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":1,"chains":1,"num_warmup":9223372036854775808,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
    ] {
        let response = json::parse(&handle_request(&request)).unwrap();
        let expected = format!(
            "{context} request settings.num_warmup must be in 0..=9223372036854775807 because artifacts report warmup counts as JSON integers"
        );
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{context}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected.as_str()),
            "{context}"
        );
    }
}

#[test]
fn artifact_requests_reject_unreportable_treedepth_bounds() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_data = json::write(linear.get("data").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();

    for (context, request) in [
        (
            "sample",
            format!(
                r#"{{"command":"sample","model":{linear_model},"data":{linear_data},"settings":{{"num_warmup":0,"num_draws":4,"max_treedepth":9223372036854775808}},"seed":5}}"#
            ),
        ),
        (
            "recover",
            format!(
                r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":9223372036854775808}},"seed":5}}"#
            ),
        ),
        (
            "sbc",
            format!(
                r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":1,"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":9223372036854775808}},"seed":5}}"#
            ),
        ),
    ] {
        let response = json::parse(&handle_request(&request)).unwrap();
        let expected = format!("{context} request settings.max_treedepth must be in 1..=20");
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{context}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected.as_str()),
            "{context}"
        );
    }
}

#[test]
fn workflow_requests_reject_unreportable_chain_counts() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();

    for (context, request) in [
        (
            "recover",
            format!(
                r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"chains":9223372036854775808,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
        (
            "sbc",
            format!(
                r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":1,"chains":9223372036854775808,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
            ),
        ),
    ] {
        let response = json::parse(&handle_request(&request)).unwrap();
        let expected = format!(
            "{context} request settings.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
        );
        assert_eq!(
            response.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{context}"
        );
        assert_eq!(
            response.get("message").and_then(Value::as_str),
            Some(expected.as_str()),
            "{context}"
        );
    }
}

#[test]
fn recover_request_rejects_interval_with_request_path() {
    let bounded = json::parse(&fixture_text("bounded_rates")).unwrap();
    let bounded_model = json::write(bounded.get("ir").unwrap()).unwrap();
    let request = format!(
        r#"{{"command":"recover","model":{bounded_model},"data":{{}},"settings":{{"interval":1.5,"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
    );
    let response = json::parse(&handle_request(&request)).unwrap();
    assert_eq!(
        response.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("recover request settings.interval must be in (0, 1)")
    );
}

#[test]
fn sbc_request_rejects_unreportable_replicates() {
    let linear = json::parse(&fixture_text("linear_regression")).unwrap();
    let linear_model = json::write(linear.get("ir").unwrap()).unwrap();
    let linear_declared_data = json::write(&fixture_declared_data("linear_regression", &["x"]))
        .expect("declared data writes");

    let request = format!(
        r#"{{"command":"sbc","model":{linear_model},"data":{linear_declared_data},"settings":{{"replicates":9223372036854775808,"chains":1,"num_warmup":0,"num_draws":4,"max_treedepth":4}},"seed":5}}"#
    );
    let response = json::parse(&handle_request(&request)).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some(
            "sbc request settings.replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers"
        )
    );
}

#[test]
fn artifact_commands_reject_too_few_draws() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 3}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(5)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("sample request settings.num_draws must be at least 4 because artifacts include diagnostics")
    );
}

#[test]
fn prior_predictive_request_rejects_zero_draws() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        (
            "command".to_string(),
            Value::Str("prior-predictive".to_string()),
        ),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        (
            "data".to_string(),
            fixture_declared_data("linear_regression", &["x"]),
        ),
        (
            "settings".to_string(),
            json::parse(r#"{"num_draws": 0}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(5)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("prior-predictive request settings.num_draws must be at least 1")
    );
}

#[test]
fn artifact_commands_reject_too_large_treedepth() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let request = Value::Object(vec![
        ("command".to_string(), Value::Str("sample".to_string())),
        ("model".to_string(), fixture.get("ir").unwrap().clone()),
        ("data".to_string(), fixture.get("data").unwrap().clone()),
        (
            "settings".to_string(),
            json::parse(r#"{"max_treedepth": 21}"#).unwrap(),
        ),
        ("seed".to_string(), Value::Int(5)),
    ]);
    let response = json::parse(&handle_request(&json::write(&request).unwrap())).unwrap();
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("sample request settings.max_treedepth must be in 1..=20")
    );
}

#[test]
fn ndjson_lines_rejects_too_few_draws() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    let settings = Settings {
        num_warmup: 0,
        num_draws: 3,
        max_treedepth: 4,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 5, 0).unwrap();
    let err = ndjson_lines(&posterior, &settings, 5, &[(0, chain)]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact chains must have at least 4 draws per chain because artifacts include diagnostics"
    );
}

#[test]
fn ndjson_lines_rejects_unreportable_chain_ids() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    let settings = Settings {
        num_warmup: 0,
        num_draws: 4,
        max_treedepth: 4,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 5, 0).unwrap();
    let err = ndjson_lines(&posterior, &settings, 5, &[(i64::MAX as u64 + 1, chain)]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact chain ids must be in 0..=9223372036854775807 because artifacts report chain ids as JSON integers"
    );
}

#[test]
fn ndjson_lines_rejects_unreportable_settings_counts() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    let valid_settings = Settings {
        num_warmup: 0,
        num_draws: 4,
        max_treedepth: 4,
        ..Settings::default()
    };
    let chain = sample(&posterior, &valid_settings, 5, 0).unwrap();

    let bad_draws = Settings {
        num_draws: i64::MAX as usize + 1,
        ..valid_settings
    };
    let err = ndjson_lines(&posterior, &bad_draws, 5, &[(0, chain.clone())]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers"
    );

    let bad_warmup = Settings {
        num_warmup: i64::MAX as usize + 1,
        ..valid_settings
    };
    let err = ndjson_lines(&posterior, &bad_warmup, 5, &[(0, chain)]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact settings.num_warmup must be in 0..=9223372036854775807 because artifacts report warmup counts as JSON integers"
    );
}

#[test]
fn ndjson_lines_rejects_unreportable_chain_diagnostics() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(fixture.get("data").unwrap()).unwrap();
    let posterior = Posterior::new(meta, data).unwrap();
    let settings = Settings {
        num_warmup: 0,
        num_draws: 4,
        max_treedepth: 4,
        ..Settings::default()
    };
    let chain = sample(&posterior, &settings, 5, 0).unwrap();

    let mut bad_divergences = chain.clone();
    bad_divergences.divergences = i64::MAX as usize + 1;
    let err = ndjson_lines(&posterior, &settings, 5, &[(0, bad_divergences)]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact chain divergences must be in 0..=9223372036854775807 because artifacts report divergences as JSON integers"
    );

    let mut bad_histogram = chain;
    bad_histogram.treedepth_histogram[0] = i64::MAX as usize + 1;
    let err = ndjson_lines(&posterior, &settings, 5, &[(0, bad_histogram)]).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sample artifact treedepth histogram counts must be in 0..=9223372036854775807 because artifacts report treedepth counts as JSON integers"
    );
}

#[test]
fn prior_predictive_ndjson_rejects_unreportable_draw_count() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(&fixture_declared_data("linear_regression", &["x"])).unwrap();
    let settings = PriorPredictiveSettings {
        num_draws: i64::MAX as usize + 1,
    };
    let err = prior_predictive_ndjson_lines(meta, data, &settings, 13).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "prior-predictive artifact settings.num_draws must be in 1..=9223372036854775807 because artifacts report draw counts as JSON integers"
    );
}

#[test]
fn prior_predictive_ndjson_rejects_zero_draw_count() {
    let fixture = json::parse(&fixture_text("linear_regression")).unwrap();
    let meta = decode_model(fixture.get("ir").unwrap()).unwrap();
    let data = data_from_json(&fixture_declared_data("linear_regression", &["x"])).unwrap();
    let settings = PriorPredictiveSettings { num_draws: 0 };
    let err = prior_predictive_ndjson_lines(meta, data, &settings, 13).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "prior-predictive artifact settings.num_draws must be at least 1"
    );
}

#[test]
fn malformed_requests_return_json_errors() {
    for request in ["not json", "{}", r#"{"command": "sample"}"#] {
        let response = json::parse(&handle_request(request)).unwrap();
        assert_eq!(
            response.get("error_format").and_then(Value::as_str),
            Some("v0-provisional"),
            "request {request:?} should identify the error format"
        );
        assert!(
            response.get("error").and_then(Value::as_str).is_some(),
            "request {request:?} should fail with a typed error"
        );
    }
}

#[test]
fn non_object_protocol_request_error_names_request_shape() {
    let response = json::parse(&handle_request("[]")).unwrap();
    assert_eq!(
        response.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("request must be an object")
    );
}

#[test]
fn non_string_protocol_command_error_names_field_type() {
    let response = json::parse(&handle_request(r#"{"command":3}"#)).unwrap();
    assert_eq!(
        response.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        response.get("message").and_then(Value::as_str),
        Some("request command must be a string")
    );
}

#[test]
fn unknown_protocol_command_error_names_command_and_supported_commands() {
    let response = json::parse(&handle_request(r#"{"command":"recovr"}"#)).unwrap();
    assert_eq!(
        response.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        response.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    let message = response.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("unknown command \"recovr\""));
    for command in [
        "sample",
        "diagnose",
        "diagnostics",
        "prior-predictive",
        "recover",
        "sbc",
    ] {
        assert!(
            message.contains(command),
            "{command} missing from {message}"
        );
    }
}
