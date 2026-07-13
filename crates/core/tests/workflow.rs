//! Direct checks for v0-provisional workflow report APIs.

use bayesite_core::error::ErrorKind;
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, DataValue};
use bayesite_core::sampler::Settings;
use bayesite_core::workflow::{recover_report, sbc_report, RecoverSettings, SbcSettings};

fn fixture_text(name: &str) -> String {
    // Conformance fixtures come from the vendored bayeswire corpus.
    let path = format!(
        "{}/../../tests/golden_ir/fixtures/{}.json",
        env!("CARGO_MANIFEST_DIR"),
        name
    );
    std::fs::read_to_string(path).expect("fixture readable")
}

fn fixture_ir(name: &str) -> Value {
    json::parse(&fixture_text(name))
        .unwrap()
        .get("ir")
        .unwrap()
        .clone()
}

fn fixture_declared_data(name: &str, declared_data: &[&str]) -> Vec<(String, DataValue)> {
    let doc = json::parse(&fixture_text(name)).unwrap();
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
    data_from_json(&data).unwrap()
}

fn short_sampler() -> Settings {
    Settings {
        num_warmup: 0,
        num_draws: 3,
        max_treedepth: 4,
        ..Settings::default()
    }
}

fn too_deep_sampler() -> Settings {
    Settings {
        num_warmup: 0,
        num_draws: 4,
        max_treedepth: 21,
        ..Settings::default()
    }
}

fn unreportable_warmup_sampler() -> Settings {
    Settings {
        num_warmup: i64::MAX as usize + 1,
        num_draws: 4,
        max_treedepth: 4,
        ..Settings::default()
    }
}

fn unreportable_draws_sampler() -> Settings {
    Settings {
        num_warmup: 0,
        num_draws: i64::MAX as usize + 1,
        max_treedepth: 4,
        ..Settings::default()
    }
}

fn report_sampler() -> Settings {
    Settings {
        num_warmup: 20,
        num_draws: 20,
        max_treedepth: 4,
        ..Settings::default()
    }
}

fn object_keys(value: &Value) -> Vec<&str> {
    match value {
        Value::Object(entries) => entries.iter().map(|(name, _)| name.as_str()).collect(),
        _ => panic!("expected object"),
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

fn assert_workflow_phases(value: &Value, field: &str) {
    assert_eq!(
        string_array(value.get(field).expect("workflow phases")),
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

fn assert_varying_intercepts_declared_data_values(report: &Value) {
    assert_eq!(
        string_array(report.get("declared_data_order").unwrap()),
        ["n_groups", "group_idx", "x"]
    );
    let declared_data = report.get("declared_data").expect("declared data");
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
    assert_eq!(
        coordinate_order(
            report
                .get("declared_data_coordinate_order")
                .and_then(|values| values.get("n_groups"))
                .expect("n_groups coordinate order")
        ),
        vec![Vec::<i64>::new()]
    );
    assert_eq!(
        coordinate_order(
            report
                .get("declared_data_coordinate_order")
                .and_then(|values| values.get("group_idx"))
                .expect("group_idx coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]
    );
}

fn assert_z_alpha_sbc_rank_histograms(report: &Value) {
    let rank_draws = report
        .get("rank_draws")
        .and_then(Value::as_i64)
        .expect("sbc rank draws");
    let replicates = report
        .get("replicates")
        .and_then(Value::as_i64)
        .expect("sbc replicates");
    let z_alpha = report
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("sbc z_alpha summary");
    assert_eq!(
        report
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        z_alpha.get("summary_scale").and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        int_array(report.get("rank_bin_order").expect("sbc rank bin order")),
        (0..=rank_draws).collect::<Vec<_>>()
    );
    assert_eq!(
        report.get("rank_bin_count").and_then(Value::as_i64),
        Some(rank_draws + 1)
    );
    assert_eq!(
        z_alpha.get("rank_draws").and_then(Value::as_i64),
        Some(rank_draws)
    );
    assert_eq!(
        z_alpha
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(rank_draws)
    );
    assert_eq!(
        z_alpha
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        z_alpha
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        z_alpha.get("truth_artifact_kind").and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        z_alpha.get("truth_artifact_scope").and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        z_alpha.get("truth_draw_index_base").and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
    let truth_draw_index = int_array(
        z_alpha
            .get("truth_draw_index")
            .expect("sbc z_alpha truth draw index"),
    );
    assert_eq!(truth_draw_index, vec![0; replicates as usize]);
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
    let seed = report
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
        int_array(
            z_alpha
                .get("rank_bin_order")
                .expect("sbc z_alpha rank bin order")
        ),
        (0..=rank_draws).collect::<Vec<_>>()
    );
    assert_eq!(
        z_alpha.get("rank_bin_count").and_then(Value::as_i64),
        Some(rank_draws + 1)
    );
    assert_eq!(
        z_alpha.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        z_alpha.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        z_alpha.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
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

fn bounded_rates_with_named_stochastic_sites() -> Value {
    let mut ir = fixture_ir("bounded_rates");
    let model = object_entry_mut(&mut ir, "model");
    let sites = object_entry_mut(model, "stochastic_sites");
    let sites = match sites {
        Value::Array(sites) => sites,
        _ => panic!("stochastic_sites must be an array"),
    };
    let names = ["p_prior_factor", "level_prior_factor", "y_likelihood"];
    for (site, name) in sites.iter_mut().zip(names) {
        *object_entry_mut(site, "name") = Value::Str(name.to_string());
    }
    ir
}

fn bounded_rates_without_level_truth_site() -> Value {
    let mut ir = fixture_ir("bounded_rates");
    let model = object_entry_mut(&mut ir, "model");
    let sites = object_entry_mut(model, "stochastic_sites");
    let sites = match sites {
        Value::Array(sites) => sites,
        _ => panic!("stochastic_sites must be an array"),
    };
    sites.retain(|site| site.get("name").and_then(Value::as_str) != Some("level"));
    ir
}

#[test]
fn workflow_reports_preserve_free_value_parameter_order() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let recover_settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let report =
        json::parse(&recover_report(meta, vec![], &recover_settings, 23).unwrap()).unwrap();
    assert_eq!(
        report.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
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
    assert_eq!(
        string_array(report.get("parameter_order").unwrap()),
        ["p", "level"]
    );
    assert_eq!(
        object_keys(report.get("parameters").unwrap()),
        ["p", "level"]
    );

    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 2,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let report = json::parse(&sbc_report(meta, vec![], &sbc_settings, 29).unwrap()).unwrap();
    assert_eq!(
        report.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(report.get("thin").and_then(Value::as_i64), Some(1));
    assert_eq!(
        report
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        report
            .get("generated_observed_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(
            report
                .get("generated_observed_order_per_replicate")
                .expect("generated observed order per replicate")
        ),
        ["y"]
    );
    assert_eq!(
        report
            .get("generated_observed_artifact_kind_per_replicate")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        report
            .get("generated_observed_artifact_scope_per_replicate")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        report
            .get("generated_observed_draw_index_per_replicate")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        report
            .get("generated_observed_draw_index_base_per_replicate")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
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
    assert_eq!(
        string_array(report.get("parameter_order").unwrap()),
        ["p", "level"]
    );
    assert_eq!(
        object_keys(report.get("parameters").unwrap()),
        ["p", "level"]
    );
    let first_replicate = report
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(
        first_replicate
            .get("workflow_format")
            .and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        first_replicate
            .get("posterior_draws")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        first_replicate
            .get("posterior_draws_artifact_kind")
            .and_then(Value::as_str),
        Some("posterior_draws")
    );
    assert_eq!(
        first_replicate
            .get("posterior_draws_artifact_scope")
            .and_then(Value::as_str),
        Some("observed_data_conditioned_parameter_draws")
    );
    assert_eq!(
        string_array(first_replicate.get("parameter_order").unwrap()),
        ["p", "level"]
    );
    assert_eq!(
        object_keys(first_replicate.get("parameters").unwrap()),
        ["p", "level"]
    );
}

#[test]
fn workflow_reports_parameter_coordinate_order() {
    let data = fixture_declared_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let recover_meta = decode_model(&fixture_ir("varying_intercepts_poisson")).unwrap();
    let recover_settings = RecoverSettings {
        chains: 2,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let recover =
        json::parse(&recover_report(recover_meta, data, &recover_settings, 23).unwrap()).unwrap();
    let z_alpha = recover
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("recover z_alpha summary");
    assert_eq!(
        coordinate_order(z_alpha.get("coordinate_order").unwrap()),
        [vec![0], vec![1], vec![2]]
    );
    let contains_by_coordinate = z_alpha
        .get("interval_contains_truth_by_coordinate")
        .and_then(Value::as_array)
        .expect("recover z_alpha interval containment facts");
    assert_eq!(contains_by_coordinate.len(), 3);
    assert!(contains_by_coordinate
        .iter()
        .all(|value| matches!(value, Value::Bool(_))));
    assert!(matches!(
        z_alpha.get("interval_contains_truth"),
        Some(Value::Bool(_))
    ));

    let data = fixture_declared_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let sbc_meta = decode_model(&fixture_ir("varying_intercepts_poisson")).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 1,
        chains: 2,
        thin: 1,
        sampler: report_sampler(),
    };
    let sbc = json::parse(&sbc_report(sbc_meta, data, &sbc_settings, 29).unwrap()).unwrap();
    let z_alpha = sbc
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("sbc aggregate z_alpha summary");
    assert_eq!(
        coordinate_order(z_alpha.get("coordinate_order").unwrap()),
        [vec![0], vec![1], vec![2]]
    );
    assert_z_alpha_sbc_rank_histograms(&sbc);
    let first_replicate = sbc
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap();
    let z_alpha = first_replicate
        .get("parameters")
        .and_then(|parameters| parameters.get("z_alpha"))
        .expect("sbc replicate z_alpha summary");
    assert_eq!(
        int_array(z_alpha.get("shape").expect("sbc replicate z_alpha shape")),
        [3]
    );
    assert_eq!(z_alpha.get("rank_draws").and_then(Value::as_i64), Some(40));
    let rank_bounds = z_alpha
        .get("rank_bounds")
        .expect("sbc replicate z_alpha rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(40));
    assert_eq!(
        int_array(
            z_alpha
                .get("rank_bin_order")
                .expect("sbc replicate z_alpha rank bin order")
        ),
        (0..=40).collect::<Vec<_>>()
    );
    assert_eq!(
        z_alpha.get("rank_bin_count").and_then(Value::as_i64),
        Some(41)
    );
    assert_eq!(
        coordinate_order(z_alpha.get("coordinate_order").unwrap()),
        [vec![0], vec![1], vec![2]]
    );
    assert_eq!(
        z_alpha.get("summary_scale").and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    assert_eq!(
        first_replicate
            .get("rank_statistic")
            .and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        first_replicate.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        first_replicate.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_eq!(
        z_alpha.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        z_alpha.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        z_alpha.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
}

#[test]
fn workflow_reports_declared_integer_data_as_json_integers() {
    let data = fixture_declared_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let recover_meta = decode_model(&fixture_ir("varying_intercepts_poisson")).unwrap();
    let recover_settings = RecoverSettings {
        chains: 2,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let recover =
        json::parse(&recover_report(recover_meta, data, &recover_settings, 23).unwrap()).unwrap();
    assert_varying_intercepts_declared_data_values(&recover);

    let data = fixture_declared_data(
        "varying_intercepts_poisson",
        &["n_groups", "group_idx", "x"],
    );
    let sbc_meta = decode_model(&fixture_ir("varying_intercepts_poisson")).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 1,
        chains: 2,
        thin: 1,
        sampler: report_sampler(),
    };
    let sbc = json::parse(&sbc_report(sbc_meta, data, &sbc_settings, 29).unwrap()).unwrap();
    assert_varying_intercepts_declared_data_values(&sbc);
}

#[test]
fn recover_report_includes_truth_rank_facts() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let report = json::parse(&recover_report(meta, vec![], &settings, 23).unwrap()).unwrap();
    assert_eq!(
        report.get("workflow_format").and_then(Value::as_str),
        Some("v0-provisional")
    );
    assert_eq!(
        report.get("rank_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_less_than_truth")
    );
    assert_eq!(
        report.get("rank_scope").and_then(Value::as_str),
        Some("per_parameter_coordinate_marginal")
    );
    assert_eq!(
        report.get("simulation_index_base").and_then(Value::as_str),
        Some("zero_based_simulation_order")
    );
    assert_eq!(
        int_array(report.get("simulation_order").expect("simulation order")),
        [0]
    );
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
    assert_eq!(report.get("rank_draws").and_then(Value::as_i64), Some(20));
    assert_eq!(
        report.get("parameter_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        report
            .get("generated_observed_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        report.get("declared_data_count").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        report.get("tie_statistic").and_then(Value::as_str),
        Some("count_posterior_draws_equal_to_truth")
    );
    assert_count_support(&report, "tie_count", 20);
    assert_eq!(
        report
            .get("parameter_summary_scale")
            .and_then(Value::as_str),
        Some("constrained_parameter_value")
    );
    let rank_bounds = report.get("rank_bounds").expect("rank bounds");
    assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
    assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
    assert_eq!(
        int_array(report.get("rank_bin_order").expect("rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        report.get("rank_bin_count").and_then(Value::as_i64),
        Some(21)
    );
    let interval_bounds = report.get("interval_bounds").expect("interval bounds");
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
    assert_interval_quantile_index_metadata(interval_bounds, 20, 0.8);
    let sampler_summary = report.get("sampler_summary").expect("sampler summary");
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
    let chains = report
        .get("chains")
        .and_then(Value::as_array)
        .expect("chain stats");
    assert_eq!(report.get("chain_count").and_then(Value::as_i64), Some(1));
    assert_eq!(
        chains.len(),
        report.get("chain_count").and_then(Value::as_i64).unwrap() as usize
    );
    assert_treedepth_support(&chains[0], 4);
    assert_eq!(
        chains[0].get("draw_count").and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(int_array(report.get("chain_order").unwrap()), [0]);

    let parameters = report.get("parameters").expect("parameters");
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
    let interval_contains_truth_by_parameter = report
        .get("interval_contains_truth_by_parameter")
        .expect("interval containment by parameter");
    assert_eq!(
        object_keys(interval_contains_truth_by_parameter),
        ["p", "level"]
    );
    assert!(
        report.get("interval_contains_truth").is_none(),
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
        assert_eq!(summary.get("rank_draws").and_then(Value::as_i64), Some(20));
        assert_eq!(
            summary.get("posterior_draws").and_then(Value::as_i64),
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
                .and_then(|value| value.get("offset"))
                .and_then(Value::as_i64),
            Some(1)
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
        assert_eq!(
            summary.get("interval_method").and_then(Value::as_str),
            Some("equal_tailed_linear_quantile")
        );
        assert_eq!(
            summary.get("interval_scope").and_then(Value::as_str),
            Some("per_parameter_coordinate_marginal")
        );
        assert_eq!(
            summary
                .get("interval_contains_truth_statistic")
                .and_then(Value::as_str),
            Some("truth_within_closed_interval_all_coordinates")
        );
        assert_eq!(
            summary.get("rank_statistic").and_then(Value::as_str),
            Some("count_posterior_draws_less_than_truth")
        );
        assert_eq!(
            summary.get("rank_scope").and_then(Value::as_str),
            Some("per_parameter_coordinate_marginal")
        );
        assert_eq!(
            summary.get("tie_statistic").and_then(Value::as_str),
            Some("count_posterior_draws_equal_to_truth")
        );
        assert_count_support(summary, "tie_count", 20);
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
        let interval_bounds = summary
            .get("interval_bounds")
            .expect("parameter interval bounds");
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
        assert_interval_quantile_index_metadata(interval_bounds, 20, 0.8);
        assert!(summary.get("rank").and_then(Value::as_i64).is_some());
        assert!(summary.get("tie_count").and_then(Value::as_i64).is_some());
    }
}

#[test]
fn workflow_reports_source_stochastic_sites() {
    let recover_meta = decode_model(&bounded_rates_with_named_stochastic_sites()).unwrap();
    let recover_settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let recover =
        json::parse(&recover_report(recover_meta, vec![], &recover_settings, 23).unwrap()).unwrap();
    assert_eq!(
        recover
            .get("generated_observed_stochastic_sites")
            .and_then(|value| value.get("y"))
            .and_then(Value::as_str),
        Some("y_likelihood")
    );
    assert_eq!(
        recover
            .get("parameters")
            .and_then(|value| value.get("p"))
            .and_then(|value| value.get("stochastic_site"))
            .and_then(Value::as_str),
        Some("p_prior_factor")
    );
    assert_eq!(
        recover
            .get("parameters")
            .and_then(|value| value.get("level"))
            .and_then(|value| value.get("stochastic_site"))
            .and_then(Value::as_str),
        Some("level_prior_factor")
    );

    let sbc_meta = decode_model(&bounded_rates_with_named_stochastic_sites()).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let sbc = json::parse(&sbc_report(sbc_meta, vec![], &sbc_settings, 29).unwrap()).unwrap();
    assert_eq!(
        sbc.get("parameters")
            .and_then(|value| value.get("p"))
            .and_then(|value| value.get("stochastic_site"))
            .and_then(Value::as_str),
        Some("p_prior_factor")
    );
    let first_replicate = sbc
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(
        first_replicate
            .get("generated_observed_stochastic_sites")
            .and_then(|value| value.get("y"))
            .and_then(Value::as_str),
        Some("y_likelihood")
    );
    assert_eq!(
        first_replicate
            .get("parameters")
            .and_then(|value| value.get("level"))
            .and_then(|value| value.get("stochastic_site"))
            .and_then(Value::as_str),
        Some("level_prior_factor")
    );
}

#[test]
fn workflow_reports_shape_preserving_generated_observed_integer_flags() {
    let data = fixture_declared_data("linear_regression", &["x"]);
    let recover_meta = decode_model(&fixture_ir("linear_regression")).unwrap();
    let recover_settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let recover =
        json::parse(&recover_report(recover_meta, data, &recover_settings, 23).unwrap()).unwrap();
    assert_eq!(
        recover
            .get("declared_data_integer_by_coordinate")
            .and_then(|value| value.get("x"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    assert_eq!(
        coordinate_order(
            recover
                .get("declared_data_coordinate_order")
                .and_then(|value| value.get("x"))
                .expect("recover declared x coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(
        string_array(recover.get("declared_data_order").unwrap()),
        ["x"]
    );
    assert_eq!(
        recover
            .get("generated_observed_integer")
            .and_then(|value| value.get("y"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    let recover_generated_integer_by_coordinate = recover
        .get("generated_observed_integer_by_coordinate")
        .and_then(|value| value.get("y"))
        .and_then(Value::as_array)
        .expect("recover generated y integer-by-coordinate flags");
    assert_eq!(recover_generated_integer_by_coordinate.len(), 5);
    assert!(recover_generated_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
    assert_eq!(
        coordinate_order(
            recover
                .get("generated_observed_coordinate_order")
                .and_then(|value| value.get("y"))
                .expect("recover generated y coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(
        string_array(recover.get("generated_observed_order").unwrap()),
        ["y"]
    );
    assert_eq!(
        recover
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        recover
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        recover
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        recover
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );

    let data = fixture_declared_data("linear_regression", &["x"]);
    let sbc_meta = decode_model(&fixture_ir("linear_regression")).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let sbc = json::parse(&sbc_report(sbc_meta, data, &sbc_settings, 29).unwrap()).unwrap();
    assert_eq!(
        sbc.get("declared_data_integer_by_coordinate")
            .and_then(|value| value.get("x"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    assert_eq!(
        coordinate_order(
            sbc.get("declared_data_coordinate_order")
                .and_then(|value| value.get("x"))
                .expect("sbc declared x coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(string_array(sbc.get("declared_data_order").unwrap()), ["x"]);
    let first_replicate = sbc
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap();
    assert_eq!(
        first_replicate
            .get("declared_data_count")
            .and_then(Value::as_i64),
        Some(1)
    );
    assert_eq!(
        string_array(first_replicate.get("declared_data_order").unwrap()),
        ["x"]
    );
    assert_eq!(
        first_replicate
            .get("generated_observed_integer")
            .and_then(|value| value.get("y"))
            .and_then(Value::as_array)
            .map(|values| values.len()),
        Some(5)
    );
    let sbc_generated_integer_by_coordinate = first_replicate
        .get("generated_observed_integer_by_coordinate")
        .and_then(|value| value.get("y"))
        .and_then(Value::as_array)
        .expect("sbc generated y integer-by-coordinate flags");
    assert_eq!(sbc_generated_integer_by_coordinate.len(), 5);
    assert!(sbc_generated_integer_by_coordinate
        .iter()
        .all(|flag| matches!(flag, Value::Bool(false))));
    assert_eq!(
        coordinate_order(
            first_replicate
                .get("generated_observed_coordinate_order")
                .and_then(|value| value.get("y"))
                .expect("sbc generated y coordinate order")
        ),
        [vec![0], vec![1], vec![2], vec![3], vec![4]]
    );
    assert_eq!(
        string_array(first_replicate.get("generated_observed_order").unwrap()),
        ["y"]
    );
    assert_eq!(
        first_replicate
            .get("generated_observed_artifact_kind")
            .and_then(Value::as_str),
        Some("prior_predictive_draws")
    );
    assert_eq!(
        first_replicate
            .get("generated_observed_artifact_scope")
            .and_then(Value::as_str),
        Some("declared_data_conditioned_site_draws")
    );
    assert_eq!(
        first_replicate
            .get("generated_observed_draw_index")
            .and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        first_replicate
            .get("generated_observed_draw_index_base")
            .and_then(Value::as_str),
        Some("zero_based_prior_predictive_draw_order")
    );
}

#[test]
fn sbc_report_uses_thinned_rank_support_but_full_posterior_diagnostics() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 4,
        sampler: report_sampler(),
    };
    let report = json::parse(&sbc_report(meta, vec![], &settings, 29).unwrap()).unwrap();
    assert_eq!(report.get("thin").and_then(Value::as_i64), Some(4));
    assert_eq!(report.get("rank_draws").and_then(Value::as_i64), Some(5));
    assert_eq!(
        report
            .get("posterior_draws_per_replicate")
            .and_then(Value::as_i64),
        Some(20)
    );
    assert_eq!(
        report
            .get("rank_bounds")
            .and_then(|bounds| bounds.get("max"))
            .and_then(Value::as_i64),
        Some(5)
    );
    let Value::Object(parameters) = report.get("parameters").unwrap() else {
        panic!("parameters must be an object");
    };
    for (_, parameter) in parameters {
        assert_eq!(parameter.get("rank_draws").and_then(Value::as_i64), Some(5));
    }
}

#[test]
fn workflow_reports_reject_missing_simulated_truth_for_free_value() {
    let meta = decode_model(&bounded_rates_without_level_truth_site()).unwrap();
    let recover_settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &recover_settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover cannot report truth for free value \"level\"; the v0 workflow requires a directly simulated stochastic site for every free value"
    );

    let meta = decode_model(&bounded_rates_without_level_truth_site()).unwrap();
    let sbc_settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let err = sbc_report(meta, vec![], &sbc_settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc cannot report truth for free value \"level\"; the v0 workflow requires a directly simulated stochastic site for every free value"
    );
}

#[test]
fn recover_report_rejects_too_few_diagnostic_draws() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: short_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover sample.draws must be at least 4 because workflow reports include diagnostics"
    );
}

#[test]
fn sbc_report_rejects_too_few_diagnostic_draws() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: short_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc sample.draws must be at least 4 because workflow reports include diagnostics"
    );
}

#[test]
fn sbc_report_rejects_invalid_thinning() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let mut settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 0,
        sampler: report_sampler(),
    };
    let err = sbc_report(meta.clone(), vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(err.message, "sbc sample.thin must be at least 1");

    settings.thin = 3;
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc sample.thin must divide sample.draws exactly; pick a thin that divides sample.draws"
    );
}

#[test]
fn recover_report_rejects_too_large_treedepth() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: too_deep_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover sample.max_treedepth must be in 1..=20"
    );
}

#[test]
fn sbc_report_rejects_too_large_treedepth() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: too_deep_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(err.message, "sbc sample.max_treedepth must be in 1..=20");
}

#[test]
fn recover_report_rejects_unreportable_sampler_counts() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: unreportable_draws_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover sample.draws must be in 1..=9223372036854775807 because workflow reports sample.draws as a JSON integer"
    );

    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: unreportable_warmup_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover sample.warmup must be in 0..=9223372036854775807 because workflow reports sample.warmup as a JSON integer"
    );
}

#[test]
fn sbc_report_rejects_unreportable_sampler_counts() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: unreportable_draws_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc sample.draws must be in 1..=9223372036854775807 because workflow reports sample.draws as a JSON integer"
    );

    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: unreportable_warmup_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc sample.warmup must be in 0..=9223372036854775807 because workflow reports sample.warmup as a JSON integer"
    );
}

#[test]
fn recover_report_rejects_unreportable_chain_count() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: i64::MAX as u64 + 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover sample.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
    );
}

#[test]
fn sbc_report_rejects_unreportable_chain_count() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: i64::MAX as u64 + 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc sample.chains must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
    );
}

#[test]
fn sbc_report_rejects_unreportable_replicate_count() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: i64::MAX as usize + 1,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers"
    );
}

#[test]
fn sbc_report_rejects_unreportable_rank_draws() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: i64::MAX as u64,
        thin: 1,
        sampler: report_sampler(),
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc rank_draws must be in 1..=9223372036854775807 because workflow reports rank_draws as a JSON integer; reduce sample.chains or sample.draws"
    );
}

#[test]
fn sbc_report_rejects_unreportable_rank_bin_count() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 1,
        chains: 1,
        thin: 1,
        sampler: Settings {
            num_warmup: 0,
            num_draws: i64::MAX as usize,
            max_treedepth: 4,
            ..Settings::default()
        },
    };
    let err = sbc_report(meta, vec![], &settings, 29).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "sbc rank_bin_count must be in 1..=9223372036854775807 because workflow reports rank_bin_count as a JSON integer; reduce sample.chains or sample.draws"
    );
}

#[test]
fn recover_report_rejects_unreportable_rank_bin_count() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: Settings {
            num_warmup: 0,
            num_draws: i64::MAX as usize,
            max_treedepth: 4,
            ..Settings::default()
        },
        interval: 0.8,
    };
    let err = recover_report(meta, vec![], &settings, 23).unwrap_err();
    assert_eq!(err.kind, ErrorKind::InvalidSettings);
    assert_eq!(
        err.message,
        "recover rank_bin_count must be in 1..=9223372036854775807 because workflow reports rank_bin_count as a JSON integer; reduce sample.chains or sample.draws"
    );
}

#[test]
fn recover_report_uses_reportable_derived_seed() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = RecoverSettings {
        chains: 1,
        sampler: report_sampler(),
        interval: 0.8,
    };
    let report = json::parse(&recover_report(meta, vec![], &settings, 23).unwrap()).unwrap();
    assert_eq!(report.get("seed").and_then(Value::as_i64), Some(23));
    assert_eq!(report.get("prior_seed").and_then(Value::as_i64), Some(23));
    assert_eq!(report.get("sample_seed").and_then(Value::as_i64), Some(24));
    let seed_schedule = report.get("seed_schedule").expect("seed schedule");
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
            .and_then(|value| value.get("offset"))
            .and_then(Value::as_i64),
        Some(1)
    );
}

#[test]
fn sbc_report_uses_reportable_replicate_seeds() {
    let meta = decode_model(&fixture_ir("bounded_rates")).unwrap();
    let settings = SbcSettings {
        replicates: 2,
        chains: 1,
        thin: 1,
        sampler: report_sampler(),
    };
    let report = json::parse(&sbc_report(meta, vec![], &settings, 29).unwrap()).unwrap();
    let replicates = report
        .get("replicate_reports")
        .and_then(Value::as_array)
        .unwrap();
    assert_eq!(report.get("replicates").and_then(Value::as_i64), Some(2));
    assert_eq!(
        report.get("replicate_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        report.get("replicate_report_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        report.get("replicate_index_base").and_then(Value::as_str),
        Some("zero_based_replicate_order")
    );
    assert_eq!(
        replicates.len(),
        report
            .get("replicate_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    assert_eq!(
        report
            .get("prior_predictive_draws_per_replicate")
            .and_then(Value::as_i64),
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
        report
            .get("settings")
            .and_then(|settings| settings.get("replicates"))
            .and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        report.get("parameter_count").and_then(Value::as_i64),
        Some(2)
    );
    let aggregate_parameters = report
        .get("parameters")
        .expect("aggregate parameter summaries");
    assert_eq!(
        report.get("parameter_report_count").and_then(Value::as_i64),
        Some(2)
    );
    assert_eq!(
        object_keys(aggregate_parameters).len(),
        report
            .get("parameter_report_count")
            .and_then(Value::as_i64)
            .unwrap() as usize
    );
    assert_eq!(
        report.get("declared_data_count").and_then(Value::as_i64),
        Some(0)
    );
    assert_eq!(
        int_array(report.get("rank_bin_order").expect("rank bin order")),
        (0..=20).collect::<Vec<_>>()
    );
    assert_count_support(&report, "tie_count", 20);
    assert_eq!(int_array(report.get("replicate_order").unwrap()), [0, 1]);
    assert_eq!(
        replicates[0].get("prior_seed").and_then(Value::as_i64),
        Some(29)
    );
    assert_eq!(
        replicates[0].get("sample_seed").and_then(Value::as_i64),
        Some(30)
    );
    assert_eq!(
        replicates[1].get("prior_seed").and_then(Value::as_i64),
        Some(31)
    );
    assert_eq!(
        replicates[1].get("sample_seed").and_then(Value::as_i64),
        Some(32)
    );
    for replicate in replicates {
        assert_workflow_phases(replicate, "workflow_phases");
        let seed_schedule = replicate
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
            replicate
                .get("settings")
                .and_then(|settings| settings.get("chains"))
                .and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            replicate
                .get("settings")
                .and_then(|settings| settings.get("num_warmup"))
                .and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            replicate
                .get("settings")
                .and_then(|settings| settings.get("num_draws"))
                .and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            replicate
                .get("settings")
                .and_then(|settings| settings.get("max_treedepth"))
                .and_then(Value::as_i64),
            Some(4)
        );
        assert!(
            (replicate
                .get("settings")
                .and_then(|settings| settings.get("target_accept"))
                .and_then(Value::as_f64)
                .unwrap()
                - 0.8)
                .abs()
                < 1e-12
        );
        assert_eq!(
            replicate.get("parameter_count").and_then(Value::as_i64),
            Some(2)
        );
        let parameters = replicate.get("parameters").expect("replicate parameters");
        assert_eq!(
            replicate
                .get("parameter_report_count")
                .and_then(Value::as_i64),
            Some(2)
        );
        assert_eq!(
            object_keys(parameters).len(),
            replicate
                .get("parameter_report_count")
                .and_then(Value::as_i64)
                .unwrap() as usize
        );
        assert_eq!(
            replicate
                .get("replicate_index_base")
                .and_then(Value::as_str),
            Some("zero_based_replicate_order")
        );
        assert_eq!(
            replicate
                .get("generated_observed_count")
                .and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            replicate.get("rank_draws").and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(
            int_array(
                replicate
                    .get("rank_bin_order")
                    .expect("replicate rank bin order")
            ),
            (0..=20).collect::<Vec<_>>()
        );
        assert_eq!(
            replicate.get("rank_bin_count").and_then(Value::as_i64),
            Some(21)
        );
        assert_count_support(replicate, "tie_count", 20);
        assert_eq!(
            replicate
                .get("parameter_summary_scale")
                .and_then(Value::as_str),
            Some("constrained_parameter_value")
        );
        assert_eq!(
            replicate
                .get("prior_predictive_draws")
                .and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            replicate
                .get("prior_predictive_draws_artifact_kind")
                .and_then(Value::as_str),
            Some("prior_predictive_draws")
        );
        assert_eq!(
            replicate
                .get("prior_predictive_draws_artifact_scope")
                .and_then(Value::as_str),
            Some("declared_data_conditioned_site_draws")
        );
        for name in ["p", "level"] {
            let summary = parameters.get(name).expect("replicate parameter summary");
            assert_eq!(
                summary.get("posterior_draws").and_then(Value::as_i64),
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
            assert_eq!(
                summary.get("prior_seed").and_then(Value::as_i64),
                replicate.get("prior_seed").and_then(Value::as_i64)
            );
            assert_eq!(
                summary.get("sample_seed").and_then(Value::as_i64),
                replicate.get("sample_seed").and_then(Value::as_i64)
            );
            assert_eq!(
                summary.get("replicate").and_then(Value::as_i64),
                replicate.get("replicate").and_then(Value::as_i64)
            );
            assert_eq!(
                summary.get("replicate_index_base").and_then(Value::as_str),
                Some("zero_based_replicate_order")
            );
            let seed_schedule = summary
                .get("seed_schedule")
                .expect("sbc replicate parameter seed schedule");
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
                summary
                    .get("shape")
                    .and_then(Value::as_array)
                    .map(|shape| shape.len()),
                Some(0)
            );
            assert_eq!(
                int_array(
                    summary
                        .get("rank_bin_order")
                        .expect("replicate parameter rank bin order")
                ),
                (0..=20).collect::<Vec<_>>()
            );
            assert_eq!(
                summary.get("rank_bin_count").and_then(Value::as_i64),
                Some(21)
            );
            assert_count_support(summary, "tie_count", 20);
            assert_eq!(
                summary.get("rhat_statistic").and_then(Value::as_str),
                Some("split_rhat")
            );
            assert_eq!(
                summary.get("rhat_scope").and_then(Value::as_str),
                Some("per_parameter_coordinate_marginal")
            );
            assert_eq!(
                summary.get("ess_statistic").and_then(Value::as_str),
                Some("effective_sample_size_geyer_initial_monotone_sequence")
            );
            assert_eq!(
                summary.get("ess_scope").and_then(Value::as_str),
                Some("per_parameter_coordinate_marginal")
            );
        }
        let rank_bounds = replicate.get("rank_bounds").expect("replicate rank bounds");
        assert_eq!(rank_bounds.get("min").and_then(Value::as_i64), Some(0));
        assert_eq!(rank_bounds.get("max").and_then(Value::as_i64), Some(20));
        let sampler_summary = replicate
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
        let chains = replicate
            .get("chains")
            .and_then(Value::as_array)
            .expect("replicate chain stats");
        assert_eq!(
            replicate.get("chain_count").and_then(Value::as_i64),
            Some(1)
        );
        assert_eq!(
            chains.len(),
            replicate
                .get("chain_count")
                .and_then(Value::as_i64)
                .unwrap() as usize
        );
        assert_treedepth_support(&chains[0], 4);
        assert_eq!(
            chains[0].get("draw_count").and_then(Value::as_i64),
            Some(20)
        );
        assert_eq!(int_array(replicate.get("chain_order").unwrap()), [0]);
    }
    let sampler_summary = report.get("sampler_summary").expect("sampler summary");
    assert_eq!(
        report
            .get("chain_count_per_replicate")
            .and_then(Value::as_i64),
        Some(1)
    );
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
}
