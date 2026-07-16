//! Dependency-free, deterministic JSON parser benchmark.
//!
//! Run with `cargo run --release --locked --manifest-path tools/bench/json-parse/Cargo.toml`.
//! Inputs are preloaded before timing. Each row is the median of nine samples.

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

fn iterations(input: &str) -> usize {
    (TARGET_BYTES / input.len().max(1)).clamp(1, 100_000)
}

fn measure(name: &str, input: &str, decode: bool) {
    for _ in 0..WARMUP {
        let value = json::parse(black_box(input)).unwrap();
        if decode {
            black_box(decode_model(black_box(&value)).unwrap());
        } else {
            black_box(value);
        }
    }
    let iterations = iterations(input);
    let elapsed = median(
        (0..SAMPLES)
            .map(|_| {
                let start = Instant::now();
                for _ in 0..iterations {
                    let value = json::parse(black_box(input)).unwrap();
                    if decode {
                        black_box(decode_model(black_box(&value)).unwrap());
                    } else {
                        black_box(value);
                    }
                }
                start.elapsed()
            })
            .collect(),
    );
    let seconds = elapsed.as_secs_f64();
    let bytes = input.len() as f64 * iterations as f64;
    println!(
        "{name},{},{iterations},{seconds:.9},{:.2},{:.2}",
        input.len(),
        bytes / seconds,
        iterations as f64 / seconds
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

fn main() {
    let depth = format!("{}0{}", "[".repeat(256), "]".repeat(256));
    let object_1k = repeated_object(1024);
    let object_64k = repeated_object(64 * 1024);
    let model = include_str!("../../../../tests/golden_ir/linear_regression.json");
    let protocol_request = format!(r#"{{"command":"capabilities","padding":{object_1k}}}"#);
    let fit_line = r#"{"chain":0,"draw":0,"values":{"alpha":0.0,"beta":[1.0,2.0]},"sample_stats_mode":"per_draw_v2","diverging":false,"tree_depth":1,"tree_accept":0.9,"energy":10.0}"#;

    println!("target,input_bytes,iterations,median_seconds,bytes_per_second,documents_per_second");
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
        measure(name, input, false);
    }
    measure("golden_linear_regression_parse_decode", model, true);
}
