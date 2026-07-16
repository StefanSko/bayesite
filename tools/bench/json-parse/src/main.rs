//! Dependency-free, deterministic JSON parser benchmark.
//!
//! Run with `cargo run --release --locked --manifest-path tools/bench/json-parse/Cargo.toml`.
//! Inputs are preloaded before timing. Each summary retains all nine raw sample
//! durations plus median, MAD, and range for baseline/candidate comparison.

use std::hint::black_box;
use std::time::{Duration, Instant};

use bayesite_core::{ir::decode_model, json};

const SAMPLES: usize = 9;
const WARMUP: usize = 16;
const TARGET_BYTES: usize = 16 * 1024 * 1024;

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure(name: &str, inputs: &[&str], decode: bool) {
    let input_bytes: usize = inputs.iter().map(|input| input.len()).sum();
    for _ in 0..WARMUP {
        for input in inputs {
            let value = json::parse(black_box(*input)).unwrap();
            if decode {
                black_box(decode_model(black_box(&value)).unwrap());
            } else {
                black_box(value);
            }
        }
    }
    let iterations = (TARGET_BYTES / input_bytes.max(1)).clamp(1, 100_000);
    let samples: Vec<_> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                for input in inputs {
                    let value = json::parse(black_box(*input)).unwrap();
                    if decode {
                        black_box(decode_model(black_box(&value)).unwrap());
                    } else {
                        black_box(value);
                    }
                }
            }
            start.elapsed()
        })
        .collect();
    let elapsed = median(samples.clone());
    let median_seconds = elapsed.as_secs_f64();
    let deviations = samples
        .iter()
        .map(|sample| sample.abs_diff(elapsed))
        .collect();
    let mad_seconds = median(deviations).as_secs_f64();
    let min_seconds = samples.iter().min().unwrap().as_secs_f64();
    let max_seconds = samples.iter().max().unwrap().as_secs_f64();
    let bytes = input_bytes as f64 * iterations as f64;
    let documents = inputs.len() as f64 * iterations as f64;
    let raw_samples = samples
        .iter()
        .map(|sample| format!("{:.9}", sample.as_secs_f64()))
        .collect::<Vec<_>>()
        .join(";");
    println!(
        "{name},{input_bytes},{},{iterations},{median_seconds:.9},{mad_seconds:.9},{min_seconds:.9},{max_seconds:.9},{:.2},{:.2},{raw_samples}",
        inputs.len(),
        bytes / median_seconds,
        documents / median_seconds,
    );
}

fn repeated_object(target_bytes: usize) -> String {
    let mut text = String::from("[");
    while text.len() + 32 < target_bytes {
        if text.len() > 1 {
            text.push(',');
        }
        text.push_str(r#"{"key":"é\n\uD834\uDD1E","n":1.0}"#);
    }
    text.push(']');
    text
}

const GOLDEN_CORPUS: &[&str] = &[
    include_str!("../../../../tests/golden_ir/alternative_prior_regression.json"),
    include_str!("../../../../tests/golden_ir/bounded_rates.json"),
    include_str!("../../../../tests/golden_ir/censored_exponential.json"),
    include_str!("../../../../tests/golden_ir/composed_measurements.json"),
    include_str!("../../../../tests/golden_ir/eight_schools_non_centered.json"),
    include_str!("../../../../tests/golden_ir/interval_censored_normal.json"),
    include_str!("../../../../tests/golden_ir/linear_regression.json"),
    include_str!("../../../../tests/golden_ir/ordinal_regression.json"),
    include_str!("../../../../tests/golden_ir/partially_observed_mvn.json"),
    include_str!("../../../../tests/golden_ir/varying_intercepts_poisson.json"),
    include_str!("../../../../tests/golden_ir/vector_bounds_named_owner.json"),
    include_str!("../../../../tests/golden_ir/data/alternative_prior_regression.json"),
    include_str!("../../../../tests/golden_ir/data/bounded_rates.json"),
    include_str!("../../../../tests/golden_ir/data/censored_exponential.json"),
    include_str!("../../../../tests/golden_ir/data/composed_measurements.json"),
    include_str!("../../../../tests/golden_ir/data/eight_schools_non_centered.json"),
    include_str!("../../../../tests/golden_ir/data/interval_censored_normal.json"),
    include_str!("../../../../tests/golden_ir/data/linear_regression.json"),
    include_str!("../../../../tests/golden_ir/data/ordinal_regression.json"),
    include_str!("../../../../tests/golden_ir/data/partially_observed_mvn.json"),
    include_str!("../../../../tests/golden_ir/data/varying_intercepts_poisson.json"),
    include_str!("../../../../tests/golden_ir/data/vector_bounds_named_owner.json"),
    include_str!("../../../../tests/golden_ir/fixtures/alternative_prior_regression.json"),
    include_str!("../../../../tests/golden_ir/fixtures/bounded_rates.json"),
    include_str!("../../../../tests/golden_ir/fixtures/censored_exponential.json"),
    include_str!("../../../../tests/golden_ir/fixtures/composed_measurements.json"),
    include_str!("../../../../tests/golden_ir/fixtures/eight_schools_non_centered.json"),
    include_str!("../../../../tests/golden_ir/fixtures/interval_censored_normal.json"),
    include_str!("../../../../tests/golden_ir/fixtures/linear_regression.json"),
    include_str!("../../../../tests/golden_ir/fixtures/ordinal_regression.json"),
    include_str!("../../../../tests/golden_ir/fixtures/partially_observed_mvn.json"),
    include_str!("../../../../tests/golden_ir/fixtures/varying_intercepts_poisson.json"),
    include_str!("../../../../tests/golden_ir/fixtures/vector_bounds_named_owner.json"),
];

fn main() {
    let depth = format!("{}0{}", "[".repeat(256), "]".repeat(256));
    let object_1k = repeated_object(1024);
    let object_64k = repeated_object(64 * 1024);
    let model = include_str!("../../../../tests/golden_ir/linear_regression.json");
    let protocol_request = format!(r#"{{"command":"capabilities","padding":{object_1k}}}"#);
    let fit_line = r#"{"chain":0,"draw":0,"values":{"alpha":0.0,"beta":[1.0,2.0]},"sample_stats_mode":"per_draw_v2","diverging":false,"tree_depth":1,"tree_accept":0.9,"energy":10.0}"#;

    println!("target,input_bytes,documents_per_iteration,iterations,median_seconds,mad_seconds,min_seconds,max_seconds,bytes_per_second,documents_per_second,raw_samples_seconds");
    for (name, input) in [
        ("scalar", "0"),
        ("tiny_request", r#"{"command":"capabilities"}"#),
        ("escaped_utf8", r#"{"text":"é\n\uD834\uDD1E"}"#),
        ("object_1k", object_1k.as_str()),
        ("object_64k", object_64k.as_str()),
        ("depth_256", depth.as_str()),
        ("protocol_request", protocol_request.as_str()),
        ("fit_ndjson_line", fit_line),
        ("golden_linear_regression", model),
    ] {
        measure(name, &[input], false);
    }
    measure("golden_corpus_parse", GOLDEN_CORPUS, false);
    measure("golden_linear_regression_parse_decode", &[model], true);
}
