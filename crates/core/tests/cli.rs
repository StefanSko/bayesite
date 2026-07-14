//! End-to-end checks of the `bayesite` subprocess protocol.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::json::{self, Value};

static TEMP_DIR_COUNTER: AtomicU64 = AtomicU64::new(0);

fn fixture_text(name: &str) -> String {
    // Conformance fixtures come from the vendored bayeswire corpus.
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

fn run_bayesite_with_stdin(args: &[&str], stdin: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("bayesite starts");
    let mut child_stdin = child.stdin.take().expect("stdin is piped");
    child_stdin
        .write_all(stdin.as_bytes())
        .expect("stdin writes");
    drop(child_stdin);
    child.wait_with_output().expect("bayesite exits")
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("{prefix}-{}-{id}", std::process::id()))
}

/// Split a fixture into model/data documents on disk; returns their paths.
fn write_fixture_inputs(name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let doc = json::parse(&fixture_text(name)).unwrap();
    let dir = unique_temp_dir(&format!("bayesite-test-{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    std::fs::write(&data_path, json::write(doc.get("data").unwrap()).unwrap()).unwrap();
    (model_path, data_path)
}

fn write_prior_predictive_inputs(
    name: &str,
    declared_data: &[&str],
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let doc = json::parse(&fixture_text(name)).unwrap();
    let dir = unique_temp_dir(&format!("bayesite-test-prior-predictive-{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    let out_path = dir.join("pp.jsonl");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let data = match doc.get("data").unwrap() {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .filter(|(name, _)| declared_data.contains(&name.as_str()))
                .cloned()
                .collect(),
        ),
        _ => panic!("fixture data must be an object"),
    };
    std::fs::write(&data_path, json::write(&data).unwrap()).unwrap();
    (model_path, data_path, out_path)
}

fn declared_data_names(doc: &Value) -> Vec<&str> {
    doc.get("ir")
        .and_then(|ir| ir.get("model"))
        .and_then(|model| model.get("data"))
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|entry| entry.get("name").and_then(Value::as_str).unwrap())
        .collect()
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
        z_alpha
            .get("rank_histogram_statistic")
            .and_then(Value::as_str),
        Some("count_simulated_replicates_by_rank")
    );
    assert_eq!(
        z_alpha.get("rank_histogram_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
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

fn assert_sample_workflow_phases(value: &Value, field: &str) {
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
            "bind_data",
            "build_posterior_state",
            "evaluate_logp_grad",
            "run_nuts",
            "emit_artifact",
        ]
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

fn write_recover_inputs(
    name: &str,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    write_recover_inputs_with_data(name, &[])
}

fn write_recover_inputs_with_data(
    name: &str,
    declared_data: &[&str],
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let doc = json::parse(&fixture_text(name)).unwrap();
    let dir = unique_temp_dir(&format!("bayesite-test-recover-{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let scenario_path = dir.join("scenario.json");
    let out_path = dir.join("recover.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let data = match doc.get("data").unwrap() {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .filter(|(name, _)| declared_data.contains(&name.as_str()))
                .cloned()
                .collect(),
        ),
        _ => panic!("fixture data must be an object"),
    };
    let scenario = Value::Object(vec![
        (
            "recover_scenario".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        ("data".to_string(), data),
        ("seed".to_string(), Value::Int(23)),
        ("interval".to_string(), Value::Float(0.8)),
        (
            "sample".to_string(),
            json::parse(
                r#"{"chains": 1, "warmup": 30, "draws": 30,
                    "max_treedepth": 4, "target_accept": 0.8}"#,
            )
            .unwrap(),
        ),
    ]);
    std::fs::write(&scenario_path, json::write(&scenario).unwrap()).unwrap();
    (model_path, scenario_path, out_path)
}

fn write_sbc_inputs(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    write_sbc_inputs_with_data(name, &[], 2, 1)
}

fn write_sbc_inputs_with_data(
    name: &str,
    declared_data: &[&str],
    replicates: i64,
    chains: i64,
) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let doc = json::parse(&fixture_text(name)).unwrap();
    let dir = unique_temp_dir(&format!("bayesite-test-sbc-{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let scenario_path = dir.join("scenario.json");
    let out_path = dir.join("sbc.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let data = match doc.get("data").unwrap() {
        Value::Object(entries) => Value::Object(
            entries
                .iter()
                .filter(|(name, _)| declared_data.contains(&name.as_str()))
                .cloned()
                .collect(),
        ),
        _ => panic!("fixture data must be an object"),
    };
    let scenario = Value::Object(vec![
        (
            "sbc_scenario".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        ("data".to_string(), data),
        ("seed".to_string(), Value::Int(29)),
        ("replicates".to_string(), Value::Int(replicates)),
        (
            "sample".to_string(),
            Value::Object(vec![
                ("chains".to_string(), Value::Int(chains)),
                ("warmup".to_string(), Value::Int(20)),
                ("draws".to_string(), Value::Int(20)),
                ("max_treedepth".to_string(), Value::Int(4)),
                ("target_accept".to_string(), Value::Float(0.8)),
            ]),
        ),
    ]);
    std::fs::write(&scenario_path, json::write(&scenario).unwrap()).unwrap();
    (model_path, scenario_path, out_path)
}

fn write_fit_input() -> std::path::PathBuf {
    let dir = unique_temp_dir("bayesite-test-diagnose");
    std::fs::create_dir_all(&dir).unwrap();
    let fit_path = dir.join("fit.jsonl");
    let mut lines = Vec::new();
    lines.push(
        r#"{"draws_format":"v0-provisional","params":[{"name":"alpha","shape":[]},{"name":"theta","shape":[2]}],"packing":["alpha","theta"],"settings":{"num_warmup":0,"num_draws":8,"max_treedepth":4,"target_accept":0.8},"seed":11,"chains":2}"#
            .to_string(),
    );
    for chain in 0..2 {
        for draw in 0..8 {
            let x = draw as f64 + chain as f64 * 0.25;
            let line = Value::Object(vec![
                ("chain".to_string(), Value::Int(chain)),
                ("draw".to_string(), Value::Int(draw)),
                (
                    "values".to_string(),
                    Value::Object(vec![
                        ("alpha".to_string(), Value::Float(x)),
                        (
                            "theta".to_string(),
                            Value::Array(vec![Value::Float(x + 1.0), Value::Float(10.0 - x)]),
                        ),
                    ]),
                ),
            ]);
            lines.push(json::write(&line).unwrap());
        }
    }
    lines.push(
        r#"{"trailer":{"chains":[{"chain":0,"divergences":0,"treedepth_histogram":[8],"step_size":1.0,"mean_accept":0.9},{"chain":1,"divergences":0,"treedepth_histogram":[8],"step_size":1.1,"mean_accept":0.88}],"rhat":{},"ess":{}}}"#
            .to_string(),
    );
    std::fs::write(&fit_path, lines.join("\n")).unwrap();
    fit_path
}

#[test]
fn sbc_reports_ranks_without_verdict() {
    let (model_path, scenario_path, out_path) = write_sbc_inputs("bounded_rates");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--replicates",
            "2",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "output should go to --out");

    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("report_kind").and_then(Value::as_str),
        Some("simulation_based_calibration_rank_facts")
    );
    assert_eq!(
        payload.get("report_scope").and_then(Value::as_str),
        Some("replicated_simulated_datasets")
    );
    assert_eq!(payload.get("replicates").and_then(Value::as_i64), Some(2));
    assert_eq!(payload.get("thin").and_then(Value::as_i64), Some(1));
    assert_eq!(
        payload.get("replicate_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload
            .get("replicate_report_count")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload
            .get("chain_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        payload.get("replicate_index_base").and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        payload
            .get("generated_observed_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            payload
                .get("generated_observed_order_per_replicate")
                .expect("generated observed order per replicate")
        ),
        ["y"]
    );
    assert_eq!(
        payload
            .get("generated_observed_artifact_kind_per_replicate")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        payload
            .get("generated_observed_artifact_scope_per_replicate")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        payload
            .get("generated_observed_draw_index_per_replicate")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        payload
            .get("generated_observed_draw_index_base_per_replicate")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        payload
            .get("settings")
            .and_then(|settings| settings.get("replicates"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload.get("parameter_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload.get("declared_data_count").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(payload.get("rank_draws").and_then(Value::as_i64), Some(20));
    assert_eq!(
        payload
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        payload
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        payload
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        payload.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        payload.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        payload.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(&payload, "tie_count", 20);
    assert_eq!(
        payload
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    let seed_schedule = payload.get("seed_schedule").expect("seed schedule");
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
    let rank_bounds = payload.get("rank_bounds").expect("rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
    assert_eq!(
        int_array(payload.get("rank_bin_order").expect("rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        payload.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    let sampler_summary = payload.get("sampler_summary").expect("sampler summary");
    assert_eq!(
        sampler_summary.get("chain_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        sampler_summary.get("draw_count").and_then(Value::as_i64),
        Some(40)
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
    assert_eq!(treedepth_histogram.iter().sum::<i64>(), 40);
    assert_workflow_phases(&payload, "replicate_workflow_phases");
    assert_eq!(int_array(payload.get("replicate_order").unwrap()), [0, 1]);
    let reports = payload
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(reports.len(), 2);
    assert_eq!(
        reports.len(),
        payload
            .get("replicate_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    for report in reports {
        assert_eq!(
            report.get("sbc_format").and_then(Value::as_str),
            Some("v0-provisional")
        );
        assert_eq!(
            report.get("workflow_format").and_then(Value::as_str),
            Some("v0-provisional")
        );
        assert_eq!(
            report.get("report_kind").and_then(Value::as_str),
            Some("simulation_based_calibration_replicate_rank_facts")
        );
        assert_eq!(
            report.get("report_scope").and_then(Value::as_str),
            Some("single_simulated_dataset_replicate")
        );
        assert_workflow_phases(report, "workflow_phases");
        let seed_schedule = report
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
            report.get("replicate_index_base").and_then(Value::as_str),
            Some("zero_based_replicate_order")
        );
        assert_eq!(
            report.get("replicate_count").and_then(Value::as_i64),
            Some(2)
        );
        assert_eq!(int_array(report.get("replicate_order").unwrap()), [0, 1]);
        assert_eq!(
            report.get("declared_data_count").and_then(Value::as_i64),
            Some(0)
        );
        assert!(string_array(report.get("declared_data_order").unwrap()).is_empty());
        assert_eq!(
            report
                .get("settings")
                .and_then(|settings| settings.get("chains"))
                .and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            report
                .get("settings")
                .and_then(|settings| settings.get("num_warmup"))
                .and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            report
                .get("settings")
                .and_then(|settings| settings.get("num_draws"))
                .and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            report
                .get("settings")
                .and_then(|settings| settings.get("max_treedepth"))
                .and_then(Value::as_i64),
            Some(4)
        );
        assert!(
            (report
                .get("settings")
                .and_then(|settings| settings.get("target_accept"))
                .and_then(Value::as_f64)
                .unwrap()
                - 0.8)
                .abs()
                < 1e-12
        );
        assert_eq!(
            report.get("parameter_count").and_then(Value::as_i64),
            Some(2)
        );
        let parameters = report.get("parameters").expect("replicate parameters");
        assert_eq!(
            report.get("parameter_report_count").and_then(Value::as_i64),
            Some(2)
        );
        assert_eq!(
            object_keys(parameters).len(),
            report
                .get("parameter_report_count")
                .and_then(Value::as_i64)
                .unwrap() as usize
        );
        assert_eq!(report.get("chain_count").and_then(Value::as_i64), Some(1));
        assert_eq!(
            report
                .get("generated_observed_count")
                .and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(report.get("rank_draws").and_then(Value::as_i64), Some(20));
        assert_eq!(
            report.get("prior_predictive_draws").and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            report
                .get("prior_predictive_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("prior_predictive_draws")
        );
        assert_eq!(
            report
                .get("prior_predictive_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("declared_data_conditioned_site_draws")
        );
        assert_eq!(
            report.get("posterior_draws").and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            report
                .get("posterior_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("posterior_draws")
        );
        assert_eq!(
            report
                .get("posterior_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("observed_data_conditioned_parameter_draws")
        );
        let rank_bounds = report.get("rank_bounds").expect("replicate rank bounds");
        assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
        assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
        assert_eq!(
            int_array(
                report
                    .get("rank_bin_order")
                    .expect("replicate rank bin order")
            ),
            (0..=20).collect::<Vec<_>>()
        );
        assert_eq!(
            report.get("rank_bin_count").and_then(Value::as_i64),
            Some(21)
        );
        assert_count_support(report, "tie_count", 20);
        assert_eq!(
            report
                .get("parameter_summary_scale")
                .and_then(Value::as_str),
            Some("constrained_parameter_value")
        );
        let sampler_summary = report
            .get("sampler_summary")
            .expect("replicate sampler summary");
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
                .expect("replicate treedepth histogram"),
        );
        assert_treedepth_support(sampler_summary, 4);
        assert_eq!(treedepth_histogram.len(), 5);
        assert_eq!(treedepth_histogram.iter().sum::<i64>(), 20);
        let p = parameters.get("p").expect("replicate p summary");
        assert_eq!(
            p.get("shape")
                .and_then(Value::as_array)
                .map(|shape| shape.len()),
            Some(0)
        );
        assert_eq!(
            p.get("summary_scale").and_then(Value::as_str),
            Some("constrained_parameter_value")
        );
        assert_eq!(
            p.get("rhat_statistic").and_then(Value::as_str),
            Some("split_rhat")
        );
        assert_eq!(
            p.get("rhat_scope").and_then(Value::as_str),
            Some("per_parameter_coordinate_marginal")
        );
        assert_eq!(
            p.get("ess_statistic").and_then(Value::as_str),
            Some("effective_sample_size_geyer_initial_monotone_sequence")
        );
        assert_eq!(
            p.get("ess_scope").and_then(Value::as_str),
            Some("per_parameter_coordinate_marginal")
        );
        assert_eq!(p.get("rank_draws").and_then(Value::as_i64), Some(20));
        assert_eq!(p.get("posterior_draws").and_then(Value::as_i64), Some(20));
        assert_eq!(
            p.get("posterior_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("posterior_draws")
        );
        assert_eq!(
            p.get("posterior_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("observed_data_conditioned_parameter_draws")
        );
        assert_eq!(
            p.get("truth_artifact_kind").and_then(Value::as_str),
            Some("prior_predictive_draws")
        );
        assert_eq!(
            p.get("truth_artifact_scope").and_then(Value::as_str),
            Some("declared_data_conditioned_site_draws")
        );
        assert_eq!(p.get("truth_draw_index").and_then(Value::as_i64), Some(0));
        assert_eq!(
            p.get("truth_draw_index_base").and_then(Value::as_str),
            Some("zero_based_prior_predictive_draw_order")
        );
        assert_eq!(
            p.get("prior_seed").and_then(Value::as_i64),
            report.get("prior_seed").and_then(Value::as_i64)
        );
        assert_eq!(
            p.get("sample_seed").and_then(Value::as_i64),
            report.get("sample_seed").and_then(Value::as_i64)
        );
        assert_eq!(
            p.get("replicate").and_then(Value::as_i64),
            report.get("replicate").and_then(Value::as_i64)
        );
        assert_eq!(
            p.get("replicate_index_base").and_then(Value::as_str),
            Some("zero_based_replicate_order")
        );
        let seed_schedule = p
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
        let p_rank_bounds = p.get("rank_bounds").expect("replicate p rank bounds");
        assert_eq!(p_rank_bounds.get("min").and_then(Value::as_i64), Some(0));
        assert_eq!(p_rank_bounds.get("max").and_then(Value::as_i64), Some(20));
        assert_eq!(
            int_array(p.get("rank_bin_order").expect("replicate p rank bin order")),
            (0..=20).collect::<Vec<_>>()
        );
        assert_eq!(p.get("rank_bin_count").and_then(Value::as_i64), Some(21));
        assert_count_support(p, "tie_count", 20);
        let chains = report
            .get("chains")
            .and_then(Value::as_array)
            .expect("replicate chain stats");
        assert_eq!(
            chains.len(),
            report.get("chain_count").and_then(Value::as_i64).unwrap() as usize
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
        assert_eq!(int_array(report.get("chain_order").unwrap()), [0]);
    }
    assert!(
        payload.get("success").is_none(),
        "sbc should not add a verdict"
    );
    assert!(
        payload.get("uniformity").is_none(),
        "sbc should report ranks, not interpret uniformity"
    );
    assert!(
        payload.get("verdict").is_none(),
        "sbc should report facts, not interpret ranks"
    );

    let parameters = payload.get("parameters").expect("parameters object");
    assert_eq!(
        payload
            .get("parameter_report_count")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        object_keys(parameters).len(),
        payload
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    for name in ["p", "level"] {
        let summary = parameters.get(name).expect("parameter summary");
        assert_eq!(summary.get("rank_draws").and_then(Value::as_i64), Some(20));
        assert_eq!(
            summary
                .get("posterior_draws_per_replicate")
                .and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            summary
                .get("posterior_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("posterior_draws")
        );
        assert_eq!(
            summary
                .get("posterior_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("observed_data_conditioned_parameter_draws")
        );
        assert_eq!(
            summary.get("replicate_count").and_then(Value::as_i64),
            Some(2)
        );
        assert_eq!(
            summary.get("replicate_index_base").and_then(Value::as_str),
            Some("zero_based_replicate_order")
        );
        assert_eq!(
            int_array(
                summary
                    .get("replicate_order")
                    .expect("parameter replicate order")
            ),
            [0, 1]
        );
        assert_eq!(
            summary.get("summary_scale").and_then(Value::as_str),
            Some("constrained_parameter_value")
        );
        assert_eq!(
            summary.get("truth_artifact_kind").and_then(Value::as_str),
            Some("prior_predictive_draws")
        );
        assert_eq!(
            summary.get("truth_artifact_scope").and_then(Value::as_str),
            Some("declared_data_conditioned_site_draws")
        );
        assert_eq!(
            summary.get("truth_draw_index_base").and_then(Value::as_str),
            Some("zero_based_prior_predictive_draw_order")
        );
        let rank_bounds = summary.get("rank_bounds").expect("parameter rank bounds");
        assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
        assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
        assert_eq!(
            int_array(
                summary
                    .get("rank_bin_order")
                    .expect("parameter rank bin order")
            ),
            (0..=20).collect::<Vec<_>>()
        );
        assert_eq!(
            summary.get("rank_bin_count").and_then(Value::as_i64),
            Some(21)
        );
        assert_count_support(summary, "tie_count", 20);
        let truths = summary
            .get("truth")
            .and_then(Value::as_array)
            .expect("parameter truths");
        assert_eq!(truths.len(), 2);
        assert!(truths.iter().all(|truth| truth.as_f64().is_some()));
        assert_eq!(
            int_array(
                summary
                    .get("truth_draw_index")
                    .expect("parameter truth draw index")
            ),
            [0, 0]
        );
        assert_eq!(
            summary
                .get("ranks")
                .and_then(Value::as_array)
                .map(|ranks| ranks.len()),
            Some(2)
        );
        assert_eq!(
            summary
                .get("rank_histogram")
                .and_then(Value::as_array)
                .map(|histogram| histogram.len()),
            Some(21)
        );
        assert_eq!(
            summary
                .get("tie_counts")
                .and_then(Value::as_array)
                .map(|ties| ties.len()),
            Some(2)
        );
        assert_eq!(
            summary
                .get("truth_integer")
                .and_then(Value::as_array)
                .map(|values| values.len()),
            Some(2)
        );
        assert_eq!(
            coordinate_order(summary.get("coordinate_order").unwrap()),
            vec![Vec::<i64>::new()]
        );
    }

    let first = &reports[0];
    assert!(first
        .get("generated_observed")
        .and_then(|v| v.get("y"))
        .is_some());
    assert!(matches!(
        first.get("generated_observed").and_then(|v| v.get("y")),
        Some(Value::Int(0 | 1))
    ));
    assert!(first
        .get("generated_observed_shapes")
        .and_then(|v| v.get("y"))
        .and_then(Value::as_array)
        .is_some());
    assert!(matches!(
        first
            .get("generated_observed_integer")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        first
            .get("generated_observed_integer_by_coordinate")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        coordinate_order(
            first
                .get("generated_observed_coordinate_order")
                .and_then(|v| v.get("y"))
                .expect("generated y coordinate order")
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        first
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        first
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        first
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert!(first
        .get("parameters")
        .and_then(|v| v.get("p"))
        .and_then(|v| v.get("rank"))
        .and_then(Value::as_i64)
        .is_some());
    assert!(first
        .get("parameters")
        .and_then(|v| v.get("p"))
        .and_then(|v| v.get("tie_count"))
        .and_then(Value::as_i64)
        .is_some());
    assert_eq!(
        coordinate_order(
            first
                .get("parameters")
                .and_then(|v| v.get("p"))
                .and_then(|v| v.get("coordinate_order"))
                .unwrap()
        ),
        vec![Vec::<i64>::new()]
    );
    assert!(matches!(
        first
            .get("parameters")
            .and_then(|v| v.get("p"))
            .and_then(|v| v.get("truth_integer")),
        Some(Value::Bool(false))
    ));
}

#[test]
fn recover_reports_truth_data_and_interval_facts() {
    let (model_path, scenario_path, out_path) = write_recover_inputs("bounded_rates");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "output should go to --out");

    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload.get("recover_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("report_kind").and_then(Value::as_str),
        Some("parameter_recovery_facts")
    );
    assert_eq!(
        payload.get("report_scope").and_then(Value::as_str),
        Some("single_simulated_dataset")
    );
    assert_eq!(
        payload.get("simulation_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        payload.get("simulation_index_base").and_then(Value::as_str),
        Some("zero_based_simulation_order")
    );
    assert_eq!(
        int_array(payload.get("simulation_order").expect("simulation order")),
        [0]
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        payload
            .get("prior_predictive_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(payload.get("seed").and_then(Value::as_i64), Some(23));
    assert_workflow_phases(&payload, "workflow_phases");
    assert_eq!(
        payload.get("posterior_draws").and_then(Value::as_i64),
        Some(30)
    );
    assert_eq!(
        payload
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        payload
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        payload.get("parameter_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload
            .get("generated_observed_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        payload.get("declared_data_count").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        payload.get("interval_method").and_then(Value::as_str),
        Some("equal_tailed_linear_quantile")
    );
    assert_eq!(
        payload.get("interval_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        payload
            .get("interval_contains_truth_statistic")
            .and_then(Value::as_str),
        Some("truth_within_closed_interval_all_coordinates")
    );
    assert_eq!(
        payload.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        payload.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(payload.get("rank_draws").and_then(Value::as_i64), Some(30));
    assert_eq!(
        payload
            .get("settings")
            .and_then(|settings| settings.get("interval"))
            .and_then(Value::as_f64),
        Some(0.8)
    );
    assert_eq!(
        payload.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(&payload, "tie_count", 30);
    assert_eq!(
        payload
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    let rank_bounds = payload.get("rank_bounds").expect("rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(30));
    assert_eq!(
        int_array(payload.get("rank_bin_order").expect("rank bin order")),
        (0..=30).collect::<Vec<_>>()
    );
    assert_eq!(
        payload.get("rank_bin_count").and_then(Value::as_i64),
        Some(31)
    );
    let sampler_summary = payload.get("sampler_summary").expect("sampler summary");
    assert_eq!(
        sampler_summary.get("chain_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        sampler_summary.get("draw_count").and_then(Value::as_i64),
        Some(30)
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
    assert_eq!(treedepth_histogram.iter().sum::<i64>(), 30);
    let chains = payload
        .get("chains")
        .and_then(Value::as_array)
        .expect("chain stats");
    assert_eq!(payload.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(
        chains.len(),
        payload.get("chain_count").and_then(Value::as_i64).unwrap() as usize
    );
    assert_treedepth_support(&chains[0], 4);
    assert_eq!(
        chains[0].get("chain_index_base").and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        chains[0].get("draw_count").and_then(Value::as_i64),
        Some(30)
    );
    assert_eq!(int_array(payload.get("chain_order").unwrap()), [0]);
    assert!(
        payload.get("success").is_none(),
        "recover should not add a verdict"
    );
    assert!(
        payload.get("verdict").is_none(),
        "recover should report facts, not interpret them"
    );
    assert!(
        payload.get("coverage").is_none(),
        "one recovery scenario should not claim coverage"
    );
    assert!(
        payload.get("interpretation").is_none(),
        "recover should leave interpretation to the caller"
    );
    assert!(
        payload.get("recommendation").is_none(),
        "recover should not recommend an action"
    );
    assert!(payload
        .get("generated_observed")
        .and_then(|v| v.get("y"))
        .is_some());
    assert!(matches!(
        payload.get("generated_observed").and_then(|v| v.get("y")),
        Some(Value::Int(0 | 1))
    ));
    assert!(payload
        .get("generated_observed_shapes")
        .and_then(|v| v.get("y"))
        .and_then(Value::as_array)
        .is_some());
    assert!(matches!(
        payload
            .get("generated_observed_integer")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        payload
            .get("generated_observed_integer_by_coordinate")
            .and_then(|v| v.get("y")),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        coordinate_order(
            payload
                .get("generated_observed_coordinate_order")
                .and_then(|v| v.get("y"))
                .expect("generated y coordinate order")
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        payload
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        payload
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        payload
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        payload
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    let interval_bounds = payload.get("interval_bounds").expect("interval bounds");
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
    let lower_quantile = interval_bounds
        .get("lower_quantile")
        .and_then(Value::as_f64)
        .unwrap();
    let upper_quantile = interval_bounds
        .get("upper_quantile")
        .and_then(Value::as_f64)
        .unwrap();
    assert!((lower_quantile - 0.1).abs() < 1e-12);
    assert!((upper_quantile - 0.9).abs() < 1e-12);
    assert_interval_quantile_index_metadata(interval_bounds, 30, 0.8);
    let parameters = payload.get("parameters").expect("parameters object");
    assert_eq!(
        payload
            .get("parameter_report_count")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        object_keys(parameters).len(),
        payload
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    let interval_contains_truth_by_parameter = payload
        .get("interval_contains_truth_by_parameter")
        .expect("interval containment by parameter");
    assert_eq!(
        object_keys(interval_contains_truth_by_parameter),
        ["p", "level"]
    );
    assert!(
        payload.get("interval_contains_truth").is_none(),
        "recover should not collapse parameter facts into a report verdict"
    );
    for name in ["p", "level"] {
        let summary = parameters.get(name).expect("parameter summary");
        let contains = summary
            .get("interval_contains_truth")
            .expect("parameter interval containment");
        assert!(matches!(contains, Value::Bool(_)));
        assert_eq!(
            interval_contains_truth_by_parameter.get(name),
            Some(contains)
        );
        assert_eq!(
            summary
                .get("shape")
                .and_then(Value::as_array)
                .map(|shape| shape.len()),
            Some(0)
        );
        assert!(summary.get("truth").and_then(Value::as_f64).is_some());
        assert!(summary.get("mean").and_then(Value::as_f64).is_some());
        assert!(summary.get("lower").and_then(Value::as_f64).is_some());
        assert!(summary.get("upper").and_then(Value::as_f64).is_some());
        assert!(matches!(
            summary.get("truth_integer"),
            Some(Value::Bool(false))
        ));
        assert!(matches!(
            summary.get("interval_contains_truth"),
            Some(Value::Bool(_))
        ));
        assert_eq!(summary.get("rank_draws").and_then(Value::as_i64), Some(30));
        assert_eq!(
            summary.get("posterior_draws").and_then(Value::as_i64),
            Some(30)
        );
        assert_eq!(
            summary
                .get("posterior_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("posterior_draws")
        );
        assert_eq!(
            summary
                .get("posterior_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("observed_data_conditioned_parameter_draws")
        );
        assert_eq!(
            summary.get("truth_artifact_kind").and_then(Value::as_str),
            Some("prior_predictive_draws")
        );
        assert_eq!(
            summary.get("truth_artifact_scope").and_then(Value::as_str),
            Some("declared_data_conditioned_site_draws")
        );
        assert_eq!(
            summary.get("truth_draw_index").and_then(Value::as_i64),
            Some(0)
        );
        assert_eq!(
            summary.get("truth_draw_index_base").and_then(Value::as_str),
            Some("zero_based_prior_predictive_draw_order")
        );
        assert_eq!(summary.get("simulation").and_then(Value::as_i64), Some(0));
        assert_eq!(
            summary.get("simulation_index_base").and_then(Value::as_str),
            Some("zero_based_simulation_order")
        );
        assert_eq!(summary.get("prior_seed").and_then(Value::as_i64), Some(23));
        assert_eq!(summary.get("sample_seed").and_then(Value::as_i64), Some(24));
        let seed_schedule = summary
            .get("seed_schedule")
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
        let rank_bounds = summary.get("rank_bounds").expect("parameter rank bounds");
        assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
        assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(30));
        assert_eq!(
            int_array(
                summary
                    .get("rank_bin_order")
                    .expect("parameter rank bin order")
            ),
            (0..=30).collect::<Vec<_>>()
        );
        assert_eq!(
            summary.get("rank_bin_count").and_then(Value::as_i64),
            Some(31)
        );
        assert_count_support(summary, "tie_count", 30);
        assert_eq!(
            summary.get("interval_method").and_then(Value::as_str),
            Some("equal_tailed_linear_quantile")
        );
        assert_eq!(
            summary.get("interval_scope").and_then(Value::as_str),
            Some("per_parameter_coordinate_marginal")
        );
        assert_eq!(
            summary.get("summary_scale").and_then(Value::as_str),
            Some("constrained_parameter_value")
        );
        assert_eq!(
            summary.get("rhat_statistic").and_then(Value::as_str),
            Some("split_rhat")
        );
        assert_eq!(
            summary.get("rhat_scope").and_then(Value::as_str),
            Some("max_over_parameter_coordinate_marginals")
        );
        assert_eq!(
            summary.get("ess_statistic").and_then(Value::as_str),
            Some("effective_sample_size_geyer_initial_monotone_sequence")
        );
        assert_eq!(
            summary.get("ess_scope").and_then(Value::as_str),
            Some("min_over_parameter_coordinate_marginals")
        );
        let parameter_interval_bounds = summary
            .get("interval_bounds")
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
        assert_interval_quantile_index_metadata(parameter_interval_bounds, 30, 0.8);
        assert!(summary.get("rank").and_then(Value::as_i64).is_some());
        assert!(summary.get("tie_count").and_then(Value::as_i64).is_some());
        assert!(summary.get("rhat").and_then(Value::as_f64).is_some());
        assert!(summary.get("ess").and_then(Value::as_f64).is_some());
        assert_eq!(
            coordinate_order(summary.get("coordinate_order").unwrap()),
            vec![Vec::<i64>::new()]
        );
    }
    assert_eq!(
        payload
            .get("chains")
            .and_then(Value::as_array)
            .map(|chains| chains.len()),
        Some(1)
    );
}

#[test]
fn recover_report_echoes_declared_scenario_data() {
    let (model_path, scenario_path, out_path) =
        write_recover_inputs_with_data("linear_regression", &["x"]);
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload
            .get("declared_data")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    assert_eq!(
        payload
            .get("declared_data_shapes")
            .and_then(|v| v.get("x"))
            .and_then(Value::as_array)
            .and_then(|shape| shape.first())
            .and_then(Value::as_i64),
        Some(5)
    );
    assert!(matches!(
        payload
            .get("declared_data_integer")
            .and_then(|v| v.get("x")),
        Some(Value::Bool(false))
    ));
    let declared_integer_by_coordinate = payload
        .get("declared_data_integer_by_coordinate")
        .and_then(|v| v.get("x"))
        .and_then(Value::as_array)
        .expect("declared x integer flags");
    assert_eq!(declared_integer_by_coordinate.len(), 5);
    assert!(declared_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
}

#[test]
fn workflow_commands_emit_declared_integer_scenario_data_as_json_integers() {
    let (model_path, scenario_path, out_path) = write_recover_inputs_with_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_varying_intercepts_declared_data_values(&payload);
    assert_z_alpha_recover_interval_contains_truth_by_coordinate(&payload);

    let (model_path, scenario_path, out_path) = write_sbc_inputs_with_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
        1,
        2,
    );
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_varying_intercepts_declared_data_values(&payload);
    assert_z_alpha_sbc_rank_histograms(&payload);
}

#[test]
fn prior_predictive_simulates_linear_regression_artifact() {
    let (model_path, data_path, out_path) =
        write_prior_predictive_inputs("linear_regression", &["x"]);
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "prior-predictive",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "13",
            "--draws",
            "3",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "output should go to --out");

    let text = std::fs::read_to_string(out_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + 3 + 1);

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
        Some(3)
    );
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(3));
    assert_eq!(
        header.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(header.get("site_count").and_then(Value::as_i64), Some(4));
    assert_eq!(
        header.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
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
    let phases: Vec<&str> = header
        .get("workflow_phases")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|phase| phase.as_str().unwrap())
        .collect();
    assert_eq!(
        phases,
        [
            "parse_json",
            "decode_ir",
            "bind_declared_data",
            "simulate_prior_predictive",
            "emit_artifact"
        ]
    );
    let sites = header.get("sites").and_then(Value::as_array).unwrap();
    let names: Vec<&str> = sites
        .iter()
        .map(|site| site.get("name").and_then(Value::as_str).unwrap())
        .collect();
    assert_eq!(names, ["alpha", "beta", "sigma", "y"]);
    assert_eq!(
        string_array(header.get("site_order").expect("site order")),
        ["alpha", "beta", "sigma", "y"]
    );
    let y_shape: Vec<i64> = sites[3]
        .get("shape")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    assert_eq!(y_shape, [5]);
    assert!(matches!(sites[3].get("integer"), Some(Value::Bool(false))));
    let y_integer_by_coordinate: Vec<bool> = sites[3]
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
        coordinate_order(sites[3].get("coordinate_order").unwrap()),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );

    let first_draw = json::parse(lines[1]).unwrap();
    assert_eq!(
        first_draw
            .get("prior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        first_draw.get("artifact_kind").and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first_draw.get("artifact_scope").and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        first_draw.get("draw_index").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(first_draw.get("seed").and_then(Value::as_i64), Some(13));
    assert_eq!(
        first_draw.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        first_draw.get("draw_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        first_draw.get("site_count").and_then(Value::as_i64),
        Some(4)
    );
    assert_eq!(
        first_draw
            .get("declared_data_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            first_draw
                .get("declared_data_order")
                .expect("draw declared data order")
        ),
        ["x"]
    );
    assert_eq!(
        string_array(first_draw.get("site_order").expect("draw site order")),
        ["alpha", "beta", "sigma", "y"]
    );
    let values = first_draw.get("values").unwrap();
    assert!(
        values
            .get("sigma")
            .and_then(Value::as_f64)
            .expect("sigma is scalar")
            > 0.0
    );
    assert_eq!(
        values
            .get("y")
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    let last_draw = json::parse(lines[3]).unwrap();
    assert_eq!(
        last_draw
            .get("prior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(last_draw.get("draw_index").and_then(Value::as_i64), Some(2));
    assert_eq!(last_draw.get("seed").and_then(Value::as_i64), Some(13));
    assert_eq!(
        last_draw.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        last_draw.get("declared_data_count").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            last_draw
                .get("declared_data_order")
                .expect("last draw declared data order")
        ),
        ["x"]
    );

    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|v| v.get("prior_predictive_format"))
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
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
        Some(3)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|value| value.get("draw_count"))
            .and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|value| value.get("draw_index_base"))
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|value| value.get("site_count"))
            .and_then(Value::as_i64),
        Some(4)
    );
    assert_eq!(
        trailer
            .get("trailer")
            .and_then(|value| value.get("declared_data_count"))
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            trailer
                .get("trailer")
                .and_then(|value| value.get("declared_data_order"))
                .expect("trailer declared data order")
        ),
        ["x"]
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
        string_array(
            trailer
                .get("trailer")
                .and_then(|v| v.get("site_order"))
                .expect("trailer site order")
        ),
        ["alpha", "beta", "sigma", "y"]
    );
    let trailer_phases: Vec<&str> = trailer
        .get("trailer")
        .and_then(|v| v.get("workflow_phases"))
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|phase| phase.as_str().unwrap())
        .collect();
    assert_eq!(trailer_phases, phases);
}

#[test]
fn prior_predictive_runs_on_assignable_golden_fixtures() {
    for fixture in [
        "linear_regression",
        "bounded_rates",
        "varying_intercepts_poisson",
        "ordinal_regression",
        "eight_schools_non_centered",
    ] {
        let doc = json::parse(&fixture_text(fixture)).unwrap();
        let declared = declared_data_names(&doc);
        let (model_path, data_path, out_path) = write_prior_predictive_inputs(fixture, &declared);
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "prior-predictive",
                "--model",
                model_path.to_str().unwrap(),
                "--data",
                data_path.to_str().unwrap(),
                "--seed",
                "101",
                "--draws",
                "2",
                "--out",
                out_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(
            output.status.success(),
            "{fixture} stderr: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let text = std::fs::read_to_string(out_path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1 + 2 + 1, "{fixture}");
        let header = json::parse(lines[0]).unwrap();
        assert_eq!(
            header
                .get("prior_predictive_format")
                .and_then(Value::as_str),
            Some("v0-provisional"),
            "{fixture}"
        );
        let first = json::parse(lines[1]).unwrap();
        assert!(first.get("values").is_some(), "{fixture}");
    }
}

#[test]
fn samples_linear_regression_over_the_subprocess_protocol() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    // At these deliberately short settings the split R-hat estimate is
    // noisy (~5-10% of seeds land above 1.05 for this fixture); the seed
    // picks a run that converges comfortably.
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "10",
            "--chains",
            "2",
            "--warmup",
            "300",
            "--draws",
            "200",
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8(output.stdout).unwrap();
    let lines: Vec<&str> = stdout.lines().collect();
    // header + 2 chains x 200 draws + trailer
    assert_eq!(lines.len(), 1 + 2 * 200 + 1);

    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_sample_artifact_identity(&header);
    assert_sample_workflow_phases(&header, "workflow_phases");
    assert!(header
        .get("model_data_fingerprint")
        .and_then(Value::as_str)
        .is_some_and(|value| value.starts_with("sha256:")));
    assert_eq!(header.get("chain_count").and_then(Value::as_i64), Some(2));
    assert_eq!(int_array(header.get("chain_order").unwrap()), [0, 1]);
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(400));
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
    let packing: Vec<&str> = header
        .get("packing")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(packing, ["alpha", "beta", "sigma"]);

    let first_draw = json::parse(lines[1]).unwrap();
    assert_eq!(
        first_draw.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_sample_artifact_identity(&first_draw);
    assert_eq!(
        first_draw.get("draw_index").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        first_draw.get("draw_index_base").and_then(Value::as_str),
        Some("zero_based_retained_draw_order")
    );
    assert_eq!(first_draw.get("seed").and_then(Value::as_i64), Some(10));
    assert_eq!(
        first_draw.get("draw_count").and_then(Value::as_i64),
        Some(400)
    );
    assert_eq!(
        first_draw.get("chain_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(int_array(first_draw.get("chain_order").unwrap()), [0, 1]);
    assert_eq!(first_draw.get("chain").and_then(Value::as_i64), Some(0));
    assert_eq!(
        first_draw.get("chain_index_base").and_then(Value::as_str),
        Some("zero_based_chain_id")
    );
    assert_eq!(
        first_draw.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(first_draw.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let sigma = first_draw
        .get("values")
        .and_then(|v| v.get("sigma"))
        .and_then(Value::as_f64)
        .unwrap();
    assert!(sigma > 0.0, "constrained sigma must be positive");
    let first_second_chain_draw = json::parse(lines[1 + 200]).unwrap();
    assert_eq!(
        first_second_chain_draw
            .get("draw_index")
            .and_then(Value::as_i64),
        Some(200)
    );
    assert_eq!(
        first_second_chain_draw.get("chain").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_second_chain_draw.get("draw").and_then(Value::as_i64),
        Some(0)
    );

    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    let trailer = trailer.get("trailer").expect("trailer object");
    assert_sample_artifact_identity(trailer);
    assert_sample_workflow_phases(trailer, "workflow_phases");
    assert_eq!(
        trailer
            .get("model_data_fingerprint")
            .and_then(Value::as_str),
        header.get("model_data_fingerprint").and_then(Value::as_str)
    );
    assert_eq!(trailer.get("chain_count").and_then(Value::as_i64), Some(2));
    assert_eq!(int_array(trailer.get("chain_order").unwrap()), [0, 1]);
    assert_eq!(trailer.get("draw_count").and_then(Value::as_i64), Some(400));
    assert_eq!(
        trailer.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(trailer.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let chain_stats = trailer.get("chains").and_then(Value::as_array).unwrap();
    assert_eq!(chain_stats.len(), 2);
    for stats in chain_stats {
        assert_eq!(
            stats.get("chain_index_base").and_then(Value::as_str),
            Some("zero_based_chain_id")
        );
        assert_eq!(stats.get("draw_count").and_then(Value::as_i64), Some(200));
        assert!(stats.get("step_size").and_then(Value::as_f64).unwrap() > 0.0);
    }
    // The posterior is well-behaved: both chains should agree.
    for (_, value) in match trailer.get("rhat").unwrap() {
        Value::Object(entries) => entries.iter(),
        _ => panic!("rhat must be an object"),
    } {
        let rhat = value.as_f64().unwrap();
        assert!(rhat < 1.05, "rhat {rhat}");
    }
}

#[test]
fn simulate_sample_recover_check_pipeline_uses_plain_data() {
    let (model_path, data_path, generated_path) =
        write_prior_predictive_inputs("linear_regression", &["x"]);
    let truth_path = model_path.with_file_name("truth.json");
    let fit_path = model_path.with_file_name("fit.jsonl");
    let report_path = model_path.with_file_name("recover-check.json");
    std::fs::write(&truth_path, r#"{"alpha": 1.5, "beta": 0.2, "sigma": 0.8}"#).unwrap();

    let simulate = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "simulate",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--truth",
            truth_path.to_str().unwrap(),
            "--seed",
            "11",
            "--out",
            generated_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite simulate runs");
    assert!(
        simulate.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&simulate.stderr)
    );
    let generated = json::parse(&std::fs::read_to_string(&generated_path).unwrap()).unwrap();
    assert!(
        generated.get("simulate_format").is_none(),
        "simulate should write a plain data document"
    );
    assert!(generated.get("x").is_some());
    assert_eq!(
        generated
            .get("y")
            .and_then(|y| y.get("dtype"))
            .and_then(Value::as_str),
        Some("float64")
    );

    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            generated_path.to_str().unwrap(),
            "--seed",
            "12",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--out",
            fit_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );

    let recover_check = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover-check",
            "--fit",
            fit_path.to_str().unwrap(),
            "--truth",
            truth_path.to_str().unwrap(),
            "--interval",
            "0.8",
            "--out",
            report_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite recover-check runs");
    assert!(
        recover_check.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&recover_check.stderr)
    );
    let report = json::parse(&std::fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(
        report.get("recover_check_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert!(report.get("verdict").is_none());
    assert_eq!(
        string_array(report.get("target_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let alpha = report
        .get("targets")
        .and_then(|targets| targets.get("alpha"))
        .expect("alpha target");
    assert_eq!(alpha.get("truth").and_then(Value::as_f64), Some(1.5));
    assert_count_support(alpha, "rank", 4);
}

#[test]
fn posterior_predictive_simulates_observed_sites_from_sample_fit() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let fit_path = model_path.with_file_name("fit.jsonl");
    let yrep_path = model_path.with_file_name("yrep.jsonl");
    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "17",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--out",
            fit_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "posterior-predictive",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--fit",
            fit_path.to_str().unwrap(),
            "--seed",
            "23",
            "--out",
            yrep_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite posterior-predictive runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty());

    let text = std::fs::read_to_string(&yrep_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
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
    assert_eq!(string_array(header.get("site_order").unwrap()), ["y"]);
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(4));

    let draw = json::parse(lines[1]).unwrap();
    assert_eq!(
        draw.get("posterior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert!(draw.get("values").unwrap().get("y").is_some());
    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    let trailer = trailer.get("trailer").expect("trailer object");
    assert_eq!(
        trailer
            .get("posterior_predictive_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
}

#[test]
fn posterior_predictive_rejects_fit_when_data_bytes_change() {
    // The fingerprint binds a fit to the exact model/data file bytes. A
    // whitespace-only edit leaves the parsed data (and the structural
    // identity hash) unchanged, so this pins the sha256 fingerprint path
    // specifically.
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let fit_path = model_path.with_file_name("fit.jsonl");
    let yrep_path = model_path.with_file_name("yrep.jsonl");
    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "17",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--out",
            fit_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );

    let mut data_text = std::fs::read_to_string(&data_path).unwrap();
    data_text.push('\n');
    std::fs::write(&data_path, data_text).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "posterior-predictive",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--fit",
            fit_path.to_str().unwrap(),
            "--seed",
            "23",
            "--out",
            yrep_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite posterior-predictive runs");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("model_data_fingerprint must match the supplied model and data"),
        "stderr: {stderr}"
    );
    assert!(!yrep_path.exists());
}

#[test]
fn wrapped_data_document_round_trips_sample_and_posterior_predictive() {
    // The canonical bayescycle.data.json.v1 wrapper is what workflow runs
    // pass for every command; the fingerprint must line up end to end.
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let bare = std::fs::read_to_string(&data_path).unwrap();
    let wrapped = format!("{{\"format\":\"bayescycle.data.json.v1\",\"variables\":{bare}}}");
    std::fs::write(&data_path, wrapped).unwrap();
    let fit_path = model_path.with_file_name("fit.jsonl");
    let yrep_path = model_path.with_file_name("yrep.jsonl");
    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "17",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--out",
            fit_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "posterior-predictive",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--fit",
            fit_path.to_str().unwrap(),
            "--seed",
            "23",
            "--out",
            yrep_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite posterior-predictive runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = std::fs::read_to_string(&yrep_path).unwrap();
    let header = json::parse(text.lines().next().unwrap()).unwrap();
    assert_eq!(
        header.get("artifact_kind").and_then(Value::as_str),
        Some("posterior_predictive_draws")
    );
}

#[test]
fn posterior_check_reports_factual_summaries_without_verdict() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let fit_path = model_path.with_file_name("fit.jsonl");
    let check_path = model_path.with_file_name("ppc.json");
    let sample = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "19",
            "--chains",
            "1",
            "--warmup",
            "10",
            "--draws",
            "4",
            "--out",
            fit_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite sample runs");
    assert!(
        sample.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&sample.stderr)
    );

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "posterior-check",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--fit",
            fit_path.to_str().unwrap(),
            "--seed",
            "29",
            "--out",
            check_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite posterior-check runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report = json::parse(&std::fs::read_to_string(&check_path).unwrap()).unwrap();
    assert_eq!(
        report.get("posterior_check_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        report.get("report_kind").and_then(Value::as_str),
        Some("posterior_predictive_check_facts")
    );
    assert!(report.get("verdict").is_none());
    assert!(report.get("pass").is_none());
    assert!(report.get("fail").is_none());
    assert!(
        report
            .get("checks")
            .and_then(Value::as_array)
            .unwrap()
            .len()
            >= 4
    );
}

#[test]
fn sample_writes_fit_artifact_to_out_path() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let out_path = model_path.with_file_name("fit.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "17",
            "--chains",
            "2",
            "--warmup",
            "50",
            "--draws",
            "8",
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "sample --out should not write draws to stdout"
    );

    let text = std::fs::read_to_string(&out_path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 1 + 2 * 8 + 1);
    let header = json::parse(lines[0]).unwrap();
    assert_eq!(
        header.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_sample_artifact_identity(&header);
    assert_eq!(header.get("chain_count").and_then(Value::as_i64), Some(2));
    assert_eq!(int_array(header.get("chain_order").unwrap()), [0, 1]);
    assert_eq!(header.get("draw_count").and_then(Value::as_i64), Some(16));
    assert_eq!(
        header.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    let first_draw = json::parse(lines[1]).unwrap();
    assert_eq!(
        first_draw.get("draw_index").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        first_draw.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    assert_eq!(
        string_array(first_draw.get("parameter_order").unwrap()),
        ["alpha", "beta", "sigma"]
    );
    let first_second_chain_draw = json::parse(lines[1 + 8]).unwrap();
    assert_eq!(
        first_second_chain_draw
            .get("draw_index")
            .and_then(Value::as_i64),
        Some(8)
    );
    assert_eq!(
        first_second_chain_draw.get("chain").and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        first_second_chain_draw.get("draw").and_then(Value::as_i64),
        Some(0)
    );
    let trailer = json::parse(lines[lines.len() - 1]).unwrap();
    let trailer = trailer.get("trailer").expect("trailer object");
    assert_sample_artifact_identity(trailer);
    assert_eq!(
        trailer.get("draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(trailer.get("seed").and_then(Value::as_i64), Some(17));
    assert_eq!(
        trailer.get("draws_per_chain").and_then(Value::as_i64),
        Some(8)
    );
    assert_eq!(trailer.get("chain_count").and_then(Value::as_i64), Some(2));
    assert_eq!(int_array(trailer.get("chain_order").unwrap()), [0, 1]);
    assert_eq!(trailer.get("draw_count").and_then(Value::as_i64), Some(16));
    assert_eq!(trailer.get("params").and_then(Value::as_i64), Some(3));
    assert_eq!(
        trailer.get("parameter_count").and_then(Value::as_i64),
        Some(3)
    );
    let chain_stats = trailer.get("chains").and_then(Value::as_array).unwrap();
    assert_eq!(chain_stats.len(), 2);
    for stats in chain_stats {
        assert_eq!(stats.get("draw_count").and_then(Value::as_i64), Some(8));
    }

    let diagnose = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["diagnose", "--fit", out_path.to_str().unwrap()])
        .output()
        .expect("bayesite runs");
    assert!(
        diagnose.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&diagnose.stderr)
    );
    let diagnostics = json::parse(String::from_utf8(diagnose.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        diagnostics
            .get("diagnostics_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        diagnostics.get("rhat_statistic").and_then(Value::as_str),
        Some("split_rhat")
    );
    assert_eq!(
        diagnostics.get("rhat_scope").and_then(Value::as_str),
        Some("max_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        diagnostics.get("ess_statistic").and_then(Value::as_str),
        Some("effective_sample_size_geyer_initial_monotone_sequence")
    );
    assert_eq!(
        diagnostics.get("ess_scope").and_then(Value::as_str),
        Some("min_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        diagnostics
            .get("source_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        diagnostics
            .get("source_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        diagnostics
            .get("source_chain_count")
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        diagnostics.get("source_draw_count").and_then(Value::as_i64),
        Some(16)
    );
    assert!(matches!(
        diagnostics.get("source_draw_index_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        diagnostics.get("source_draw_parameter_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        diagnostics.get("source_draw_artifact_metadata"),
        Some(Value::Bool(true))
    ));
    assert!(matches!(
        diagnostics.get("source_draw_chain_metadata"),
        Some(Value::Bool(true))
    ));
    assert_eq!(
        diagnostics
            .get("source_parameter_count")
            .and_then(Value::as_i64),
        Some(3)
    );
    let diagnostic_chains = diagnostics
        .get("chains")
        .and_then(Value::as_array)
        .expect("diagnostic chains");
    assert_eq!(diagnostic_chains.len(), 2);
    for stats in diagnostic_chains {
        assert_eq!(stats.get("draw_count").and_then(Value::as_i64), Some(8));
    }
    assert_diagnose_workflow_phases(&diagnostics, "workflow_phases");
    assert_sample_workflow_phases(&diagnostics, "source_workflow_phases");
}

#[test]
fn diagnoses_v0_fit_artifact() {
    let fit_path = write_fit_input();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["diagnose", "--fit", fit_path.to_str().unwrap()])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let payload = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("diagnostics_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("source_draws_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("rhat_statistic").and_then(Value::as_str),
        Some("split_rhat")
    );
    assert_eq!(
        payload.get("rhat_scope").and_then(Value::as_str),
        Some("max_over_parameter_coordinate_marginals")
    );
    assert_eq!(
        payload.get("ess_statistic").and_then(Value::as_str),
        Some("effective_sample_size_geyer_initial_monotone_sequence")
    );
    assert_eq!(
        payload.get("ess_scope").and_then(Value::as_str),
        Some("min_over_parameter_coordinate_marginals")
    );
    assert_diagnose_workflow_phases(&payload, "workflow_phases");
    assert_eq!(payload.get("source_seed").and_then(Value::as_i64), Some(11));
    assert_eq!(
        payload.get("source_chains").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        payload
            .get("source_settings")
            .and_then(|v| v.get("num_draws"))
            .and_then(Value::as_i64),
        Some(8)
    );
    assert_eq!(
        payload
            .get("source_params")
            .and_then(Value::as_array)
            .map(|params| params.len()),
        Some(2)
    );
    assert_eq!(
        coordinate_order(
            payload
                .get("source_params")
                .and_then(Value::as_array)
                .and_then(|params| params.get(1))
                .and_then(|param| param.get("coordinate_order"))
                .unwrap()
        ),
        vec![vec![0], vec![1]]
    );
    assert_eq!(
        payload
            .get("source_packing")
            .and_then(Value::as_array)
            .and_then(|packing| packing.get(1))
            .and_then(Value::as_str),
        Some("theta")
    );
    assert_eq!(
        payload.get("draws_per_chain").and_then(Value::as_i64),
        Some(8)
    );
    assert!(matches!(
        payload.get("source_draw_index_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        payload.get("source_draw_parameter_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        payload.get("source_draw_artifact_metadata"),
        Some(Value::Bool(false))
    ));
    assert!(matches!(
        payload.get("source_draw_chain_metadata"),
        Some(Value::Bool(false))
    ));
    assert_eq!(
        payload
            .get("chains")
            .and_then(Value::as_array)
            .map(|chains| chains.len()),
        Some(2)
    );
    assert!(
        payload
            .get("rhat")
            .and_then(|v| v.get("alpha"))
            .and_then(Value::as_f64)
            .unwrap()
            > 0.5
    );
    assert!(
        payload
            .get("ess")
            .and_then(|v| v.get("theta"))
            .and_then(Value::as_f64)
            .unwrap()
            > 1.0
    );
}

#[test]
fn diagnose_writes_report_to_out_path() {
    let fit_path = write_fit_input();
    let dir = unique_temp_dir("bayesite-test-diagnose-out");
    std::fs::create_dir_all(&dir).unwrap();
    let out_path = dir.join("diagnostics.json");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "diagnose",
            "--fit",
            fit_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "diagnose --out should use the path"
    );

    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload.get("diagnostics_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_diagnose_workflow_phases(&payload, "workflow_phases");
}

#[test]
fn diagnose_accepts_fit_stdin_as_json() {
    let fit_path = write_fit_input();
    let fit_stream = std::fs::read_to_string(fit_path).unwrap();
    let output = run_bayesite_with_stdin(&["diagnose", "--fit", "-"], &fit_stream);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "diagnose should keep stderr empty"
    );
    let payload = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("diagnostics_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("source_draw_count").and_then(Value::as_i64),
        Some(16)
    );
    assert_diagnose_workflow_phases(&payload, "workflow_phases");
}

#[test]
fn workflow_commands_accept_one_stdin_input_as_json() {
    let (model_path, scenario_path, out_path) = write_recover_inputs("bounded_rates");
    let scenario = std::fs::read_to_string(scenario_path).unwrap();
    let output = run_bayesite_with_stdin(
        &[
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            "-",
            "--out",
            out_path.to_str().unwrap(),
        ],
        &scenario,
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stdout.is_empty(),
        "recover --out should use the path"
    );
    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload.get("recover_format").and_then(Value::as_str),
        Some("v0-provisional")
    );

    let (model_path, scenario_path, out_path) = write_sbc_inputs("bounded_rates");
    let scenario = std::fs::read_to_string(scenario_path).unwrap();
    let output = run_bayesite_with_stdin(
        &[
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            "-",
            "--replicates",
            "1",
            "--out",
            out_path.to_str().unwrap(),
        ],
        &scenario,
    );
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.is_empty(), "sbc --out should use the path");
    let payload = json::parse(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(
        payload.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
}

#[test]
fn recover_and_sbc_default_to_stdout_json() {
    let (model_path, scenario_path, _out_path) = write_recover_inputs("bounded_rates");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "recover should keep stderr empty");
    let payload = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("recover_format").and_then(Value::as_str),
        Some("v0-provisional")
    );

    let (model_path, scenario_path, _out_path) = write_sbc_inputs("bounded_rates");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--replicates",
            "1",
        ])
        .output()
        .expect("bayesite runs");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty(), "sbc should keep stderr empty");
    let payload = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("sbc_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
}

#[test]
fn reports_errors_as_json_on_stderr() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            "/nonexistent.json",
            "--data",
            "/nonexistent.json",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    let payload = json::parse(stderr.trim()).expect("stderr is a JSON object");
    assert_eq!(
        payload.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert!(payload.get("error").and_then(Value::as_str).is_some());
    assert!(payload.get("message").and_then(Value::as_str).is_some());
}

#[test]
fn data_input_errors_advertise_stdin_support() {
    for command in ["sample", "prior-predictive"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([command, "--model", "/nonexistent.json"])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{command}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some("--data is required (a path or - for stdin)"),
            "{command}"
        );
    }

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["unknown-command"])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    let message = payload.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("sample --model <ir.json|-> --data <data.json|->"));
    assert!(message.contains("prior-predictive --model <ir.json|-> --data <data.json|->"));
}

#[test]
fn data_artifact_commands_report_data_field_shape_as_json() {
    let doc = json::parse(&fixture_text("linear_regression")).unwrap();
    let dir = unique_temp_dir("bayesite-test-data-command-shape");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    std::fs::write(&data_path, "[]").unwrap();

    for (command, message) in [
        ("sample", "sample data must be an object"),
        (
            "prior-predictive",
            "prior-predictive data must be an object",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                command,
                "--model",
                model_path.to_str().unwrap(),
                "--data",
                data_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{command}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{command}"
        );
    }
}

#[test]
fn capabilities_emits_versioned_document_matching_dispatch_table() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["capabilities"])
        .output()
        .expect("bayesite runs");
    assert!(output.status.success());
    assert!(
        output.stderr.is_empty(),
        "capabilities should keep stderr empty"
    );
    let doc = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        doc.get("capabilities_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        doc.get("version").and_then(Value::as_str),
        Some(env!("CARGO_PKG_VERSION"))
    );
    let commands: Vec<String> = match doc.get("commands").expect("commands field") {
        Value::Array(entries) => entries
            .iter()
            .map(|entry| {
                Value::as_str(entry)
                    .expect("command is a string")
                    .to_string()
            })
            .collect(),
        _ => panic!("commands must be an array"),
    };
    assert_eq!(
        commands,
        [
            "sample",
            "diagnose",
            "prior-predictive",
            "generate",
            "posterior-predictive",
            "posterior-check",
            "simulate",
            "recover-check",
            "recover",
            "sbc",
            "capabilities",
        ]
    );
    // Cross-check against the dispatch table through behavior: every
    // advertised command must dispatch (no "unknown command"), and every
    // advertised command must have a usage line.
    let usage_probe = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["definitely-not-a-command"])
        .output()
        .expect("bayesite runs");
    let usage_payload = json::parse(String::from_utf8(usage_probe.stderr).unwrap().trim()).unwrap();
    let usage_text = usage_payload
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    for command in &commands {
        assert!(
            usage_text.contains(&format!("bayesite {command}")),
            "usage text misses {command}"
        );
        let probe = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([command.as_str()])
            .output()
            .expect("bayesite runs");
        if command == "capabilities" {
            assert!(probe.status.success(), "{command}");
            continue;
        }
        let payload = json::parse(String::from_utf8(probe.stderr).unwrap().trim()).unwrap();
        let message = payload.get("message").and_then(Value::as_str).unwrap();
        assert!(
            !message.contains("unknown command"),
            "{command} is advertised but not dispatched: {message}"
        );
    }
    assert_eq!(
        doc.get("ir")
            .and_then(|ir| ir.get("bayeswire_ir"))
            .and_then(Value::as_i64),
        Some(1)
    );
    let schemas = doc.get("schemas").expect("schemas field");
    for schema in [
        "recover_scenario",
        "sbc_scenario",
        "recover_check_targets",
        "error_format",
    ] {
        assert_eq!(
            schemas.get(schema).and_then(Value::as_str),
            Some("v0-provisional"),
            "{schema}"
        );
    }
}

#[test]
fn capabilities_rejects_arguments_with_json_error() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["capabilities", "--verbose"])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "error path should keep stdout empty"
    );
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    let message = payload.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("capabilities takes no arguments"));
}

#[test]
fn unknown_command_error_names_command_and_supported_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["recovr"])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "error path should keep stdout empty"
    );
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    let message = payload.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("unknown command \"recovr\""));
    assert!(message.contains("sample --model <ir.json|-> --data <data.json|->"));
    assert!(message.contains("recover --model <ir.json|-> --scenario <scenario.json|->"));
    assert!(message.contains("sbc --model <ir.json|-> --scenario <scenario.json|->"));
}

#[test]
fn missing_command_error_names_missing_command_and_supported_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    assert!(
        output.stdout.is_empty(),
        "error path should keep stdout empty"
    );
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    let message = payload.get("message").and_then(Value::as_str).unwrap();
    assert!(message.contains("missing command"));
    assert!(message.contains("sample --model <ir.json|-> --data <data.json|->"));
    assert!(message.contains("diagnose --fit <fit.jsonl|->"));
    assert!(message.contains("prior-predictive --model <ir.json|-> --data <data.json|->"));
    assert!(message.contains("recover --model <ir.json|-> --scenario <scenario.json|->"));
    assert!(message.contains("sbc --model <ir.json|-> --scenario <scenario.json|->"));
}

#[test]
fn duplicate_cli_flags_are_json_errors() {
    let (model_path, data_path) = write_fixture_inputs("linear_regression");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
            "--seed",
            "not-a-number",
            "--seed",
            "5",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("sample has duplicate flag --seed; pass it once")
    );

    let (model_path, scenario_path, _out_path) = write_sbc_inputs("bounded_rates");
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
            "--replicates",
            "0",
            "--replicates",
            "2",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("sbc has duplicate flag --replicates; pass it once")
    );
}

#[test]
fn cli_flags_reject_missing_values_before_next_flag_as_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(["sample", "--model", "--data", "/nonexistent.json"])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("flag --model needs a value before --data")
    );

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            "/nonexistent.json",
            "--scenario",
            "--out",
            "/tmp/recover.json",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("flag --scenario needs a value before --out")
    );
}

#[test]
fn sample_rejects_invalid_settings_as_json() {
    for (flag, value, message) in [
        ("--draws", "0", "--draws must be at least 1"),
        (
            "--draws",
            "3",
            "--draws must be at least 4 because sample artifacts include diagnostics",
        ),
        (
            "--draws",
            "9223372036854775808",
            "--draws must be in 1..=9223372036854775807 because sample artifacts report draw counts as JSON integers",
        ),
        (
            "--draws",
            "18446744073709551616",
            "--draws must be in 1..=9223372036854775807 because sample artifacts report draw counts as JSON integers",
        ),
        (
            "--warmup",
            "9223372036854775808",
            "--warmup must be in 0..=9223372036854775807 because sample artifacts report warmup counts as JSON integers",
        ),
        (
            "--warmup",
            "18446744073709551616",
            "--warmup must be in 0..=9223372036854775807 because sample artifacts report warmup counts as JSON integers",
        ),
        ("--max-treedepth", "0", "--max-treedepth must be at least 1"),
        ("--max-treedepth", "21", "--max-treedepth must be in 1..=20"),
        (
            "--max-treedepth",
            "18446744073709551616",
            "--max-treedepth must be in 1..=20",
        ),
        (
            "--target-accept",
            "1.5",
            "--target-accept must be in (0, 1)",
        ),
        (
            "--chains",
            "9223372036854775808",
            "--chains must be in 1..=9223372036854775807 because sample artifacts report chain counts as JSON integers",
        ),
        (
            "--chains",
            "18446744073709551616",
            "--chains must be in 1..=9223372036854775807 because sample artifacts report chain counts as JSON integers",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "sample",
                "--model",
                "/nonexistent.json",
                "--data",
                "/nonexistent.json",
                flag,
                value,
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{flag} {value}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{flag} {value}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{flag} {value}"
        );
    }
}

#[test]
fn prior_predictive_rejects_unreportable_draw_count_as_json() {
    for value in ["9223372036854775808", "18446744073709551616"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "prior-predictive",
                "--model",
                "/nonexistent.json",
                "--data",
                "/nonexistent.json",
                "--draws",
                value,
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{value}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{value}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some("--draws must be in 1..=9223372036854775807 because prior-predictive artifacts report draw counts as JSON integers"),
            "{value}"
        );
    }
}

#[test]
fn prior_predictive_rejects_zero_draw_count_as_json() {
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "prior-predictive",
            "--model",
            "/nonexistent.json",
            "--data",
            "/nonexistent.json",
            "--draws",
            "0",
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("--draws must be at least 1")
    );
}

#[test]
fn recover_rejects_invalid_scenario_sampler_settings_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-recover-invalid-settings");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let scenario_path = dir.join("scenario.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let scenario = json::parse(
        r#"{
            "recover_scenario": "v0-provisional",
            "data": {},
            "seed": 23,
            "sample": {
                "draws": 3
            }
        }"#,
    )
    .unwrap();
    std::fs::write(&scenario_path, json::write(&scenario).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("recover scenario sample.draws must be at least 4 because workflow reports include diagnostics")
    );
}

#[test]
fn workflow_scenarios_report_nested_float_field_paths_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-nested-float");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let scenario_path = dir.join("scenario.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let scenario = json::parse(
        r#"{
            "recover_scenario": "v0-provisional",
            "data": {},
            "seed": 23,
            "sample": {
                "draws": 4,
                "target_accept": "wide"
            }
        }"#,
    )
    .unwrap();
    std::fs::write(&scenario_path, json::write(&scenario).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("recover scenario sample.target_accept must be a number")
    );
}

#[test]
fn recover_scenario_reports_interval_path_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-recover-interval");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let scenario_path = dir.join("scenario.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    let scenario = json::parse(
        r#"{
            "recover_scenario": "v0-provisional",
            "data": {},
            "seed": 23,
            "interval": 1.5,
            "sample": {
                "draws": 4
            }
        }"#,
    )
    .unwrap();
    std::fs::write(&scenario_path, json::write(&scenario).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("recover scenario interval must be in (0, 1)")
    );
}

#[test]
fn workflow_scenarios_report_data_field_shape_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-data-shape");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    for (command, marker, message) in [
        (
            "recover",
            "recover_scenario",
            "recover scenario data must be an object",
        ),
        ("sbc", "sbc_scenario", "sbc scenario data must be an object"),
    ] {
        let scenario_path = dir.join(format!("{command}-scenario.json"));
        let scenario = format!(
            r#"{{
                "{marker}": "v0-provisional",
                "data": [],
                "seed": 23,
                "sample": {{
                    "draws": 4
                }}
            }}"#
        );
        std::fs::write(&scenario_path, scenario).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                command,
                "--model",
                model_path.to_str().unwrap(),
                "--scenario",
                scenario_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{command}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{command}"
        );
    }
}

#[test]
fn workflow_scenarios_reject_unreportable_seed_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-unreportable-seed");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    for (command, marker, message) in [
        (
            "recover",
            "recover_scenario",
            "recover scenario seed must be in 0..=9223372036854775807 because workflow reports seeds as JSON integers",
        ),
        (
            "sbc",
            "sbc_scenario",
            "sbc scenario seed must be in 0..=9223372036854775807 because workflow reports seeds as JSON integers",
        ),
    ] {
        let scenario_path = dir.join(format!("{command}-scenario.json"));
        let scenario = format!(
            r#"{{
                "{marker}": "v0-provisional",
                "data": {{}},
                "seed": 9223372036854775808,
                "sample": {{
                    "draws": 4
                }}
            }}"#
        );
        std::fs::write(&scenario_path, scenario).unwrap();

        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                command,
                "--model",
                model_path.to_str().unwrap(),
                "--scenario",
                scenario_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{command}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{command}"
        );
    }
}

#[test]
fn workflow_scenarios_reject_unreportable_count_fields_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-unreportable-counts");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    for (name, command, marker, extra, message) in [
        (
            "recover-chains",
            "recover",
            "recover_scenario",
            r#""sample": {"chains": 9223372036854775808, "draws": 4}"#,
            "recover scenario sample.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers",
        ),
        (
            "recover-warmup",
            "recover",
            "recover_scenario",
            r#""sample": {"warmup": 9223372036854775808, "draws": 4}"#,
            "recover scenario sample.warmup must be in 0..=9223372036854775807 because workflow reports sample.warmup as a JSON integer",
        ),
        (
            "recover-draws",
            "recover",
            "recover_scenario",
            r#""sample": {"draws": 9223372036854775808}"#,
            "recover scenario sample.draws must be in 1..=9223372036854775807 because workflow reports sample.draws as a JSON integer",
        ),
        (
            "sbc-replicates",
            "sbc",
            "sbc_scenario",
            r#""replicates": 9223372036854775808, "sample": {"draws": 4}"#,
            "sbc scenario replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers",
        ),
        (
            "sbc-thin",
            "sbc",
            "sbc_scenario",
            r#""replicates": 1, "sample": {"draws": 4, "thin": 9223372036854775808}"#,
            "sbc scenario sample.thin must be in 1..=9223372036854775807 because workflow reports sample.thin as a JSON integer",
        ),
        (
            "sbc-max-treedepth",
            "sbc",
            "sbc_scenario",
            r#""replicates": 1, "sample": {"max_treedepth": 9223372036854775808, "draws": 4}"#,
            "sbc scenario sample.max_treedepth must be in 1..=20",
        ),
    ] {
        let scenario_path = dir.join(format!("{name}.json"));
        let scenario = format!(
            r#"{{
                "{marker}": "v0-provisional",
                "data": {{}},
                "seed": 23,
                {extra}
            }}"#
        );
        std::fs::write(&scenario_path, scenario).unwrap();

        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                command,
                "--model",
                model_path.to_str().unwrap(),
                "--scenario",
                scenario_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{name}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{name}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{name}"
        );
    }
}

#[test]
fn sbc_scenario_rejects_invalid_thinning_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-sbc-invalid-thin");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    for (thin, message) in [
        ("0", "sbc scenario sample.thin must be at least 1"),
        (
            "3",
            "sbc scenario sample.thin must divide sample.draws exactly; pick a thin that divides sample.draws",
        ),
    ] {
        let scenario_path = dir.join(format!("thin-{thin}.json"));
        std::fs::write(
            &scenario_path,
            format!(
                r#"{{
                    "sbc_scenario": "v0-provisional",
                    "data": {{}},
                    "seed": 29,
                    "replicates": 1,
                    "sample": {{"draws": 20, "thin": {thin}}}
                }}"#
            ),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "sbc",
                "--model",
                model_path.to_str().unwrap(),
                "--scenario",
                scenario_path.to_str().unwrap(),
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "thin={thin}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "thin={thin}"
        );
    }
}

#[test]
fn workflow_scenarios_reject_unknown_fields_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-unknown-fields");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    let recover_scenario_path = dir.join("recover-scenario.json");
    let recover_scenario = json::parse(
        r#"{
            "recover_scenario": "v0-provisional",
            "data": {},
            "seed": 23,
            "confidence": 0.8
        }"#,
    )
    .unwrap();
    std::fs::write(
        &recover_scenario_path,
        json::write(&recover_scenario).unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            recover_scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("recover scenario has unknown field \"confidence\"")
    );

    let sbc_scenario_path = dir.join("sbc-scenario.json");
    let sbc_scenario = json::parse(
        r#"{
            "sbc_scenario": "v0-provisional",
            "data": {},
            "seed": 29,
            "sample": {
                "draw_count": 20
            }
        }"#,
    )
    .unwrap();
    std::fs::write(&sbc_scenario_path, json::write(&sbc_scenario).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            sbc_scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("sbc scenario sample has unknown field \"draw_count\"")
    );
}

#[test]
fn workflow_scenarios_reject_duplicate_fields_as_json() {
    let doc = json::parse(&fixture_text("bounded_rates")).unwrap();
    let dir = unique_temp_dir("bayesite-test-workflow-duplicate-fields");
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();

    let recover_scenario_path = dir.join("recover-scenario.json");
    std::fs::write(
        &recover_scenario_path,
        r#"{
            "recover_scenario": "v0-provisional",
            "data": {},
            "seed": -1,
            "seed": 23
        }"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "recover",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            recover_scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("recover scenario has duplicate field \"seed\"; remove one")
    );

    let sbc_scenario_path = dir.join("sbc-scenario.json");
    std::fs::write(
        &sbc_scenario_path,
        r#"{
            "sbc_scenario": "v0-provisional",
            "data": {},
            "seed": 29,
            "sample": {
                "draws": 0,
                "draws": 20
            }
        }"#,
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sbc",
            "--model",
            model_path.to_str().unwrap(),
            "--scenario",
            sbc_scenario_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("InvalidSettings")
    );
    assert_eq!(
        payload.get("message").and_then(Value::as_str),
        Some("sbc scenario sample has duplicate field \"draws\"; remove one")
    );
}

#[test]
fn multi_input_commands_reject_multiple_stdin_inputs_as_json() {
    for (command, args, message) in [
        (
            "sample",
            vec!["sample", "--model", "-", "--data", "-"],
            "sample accepts at most one stdin input; use a path for --data when --model is -",
        ),
        (
            "recover",
            vec!["recover", "--model", "-", "--scenario", "-"],
            "recover accepts at most one stdin input; use a path for --scenario when --model is -",
        ),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args(args)
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{command}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{command}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some(message),
            "{command}"
        );
    }
}

#[test]
fn artifact_commands_reject_unreportable_seed_as_json() {
    for command in ["sample", "prior-predictive"] {
        for value in ["9223372036854775808", "18446744073709551616"] {
            let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
                .args([
                    command,
                    "--model",
                    "/nonexistent.json",
                    "--data",
                    "/nonexistent.json",
                    "--seed",
                    value,
                ])
                .output()
                .expect("bayesite runs");
            assert!(!output.status.success(), "{command} {value}");
            let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
            assert_eq!(
                payload.get("error").and_then(Value::as_str),
                Some("InvalidSettings"),
                "{command} {value}"
            );
            assert_eq!(
                payload.get("message").and_then(Value::as_str),
                Some("--seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers"),
                "{command} {value}"
            );
        }
    }
}

#[test]
fn sbc_rejects_unreportable_replicates_as_json() {
    for value in ["9223372036854775808", "18446744073709551616"] {
        let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
            .args([
                "sbc",
                "--model",
                "/nonexistent.json",
                "--scenario",
                "/nonexistent.json",
                "--replicates",
                value,
            ])
            .output()
            .expect("bayesite runs");
        assert!(!output.status.success(), "{value}");
        let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
        assert_eq!(
            payload.get("error").and_then(Value::as_str),
            Some("InvalidSettings"),
            "{value}"
        );
        assert_eq!(
            payload.get("message").and_then(Value::as_str),
            Some("--replicates must be in 1..=9223372036854775807 because sbc reports replicates as a JSON integer"),
            "{value}"
        );
    }
}

#[test]
fn unknown_tag_failure_names_the_tag() {
    let dir = std::env::temp_dir().join(format!("bayesite-test-unknown-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let model_path = dir.join("model.json");
    let data_path = dir.join("data.json");
    let model = fixture_text("linear_regression").replace("\"Normal\"", "\"FancyDist\"");
    let doc = json::parse(&model).unwrap();
    std::fs::write(&model_path, json::write(doc.get("ir").unwrap()).unwrap()).unwrap();
    std::fs::write(&data_path, json::write(doc.get("data").unwrap()).unwrap()).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args([
            "sample",
            "--model",
            model_path.to_str().unwrap(),
            "--data",
            data_path.to_str().unwrap(),
        ])
        .output()
        .expect("bayesite runs");
    assert!(!output.status.success());
    let payload = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        payload.get("error").and_then(Value::as_str),
        Some("UnknownNodeTag")
    );
    assert!(payload
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("FancyDist"));
}
