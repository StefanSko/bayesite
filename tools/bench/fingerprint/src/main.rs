use std::hint::black_box;
use std::time::{Duration, Instant};

use bayesite_core::fingerprint::{model_data_fingerprint, sha256_bytes};

const SAMPLES: usize = 9;
const WARMUP: usize = 16;

fn deterministic_bytes(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 37 + 11) % 256) as u8).collect()
}

fn median(mut values: Vec<Duration>) -> Duration {
    values.sort_unstable();
    values[values.len() / 2]
}

fn measure_bytes(name: &str, input: &[u8]) {
    for _ in 0..WARMUP {
        black_box(sha256_bytes(black_box(input)));
    }
    let iterations = (16 * 1024 * 1024 / input.len().max(1)).clamp(1, 100_000);
    let samples = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(sha256_bytes(black_box(input)));
            }
            start.elapsed()
        })
        .collect();
    let elapsed = median(samples);
    let seconds = elapsed.as_secs_f64();
    let bytes = input.len() as f64 * iterations as f64;
    println!(
        "{name},{},{iterations},{seconds:.9},{:.2}",
        input.len(),
        bytes / seconds
    );
}

fn measure_framed() {
    let model = include_str!("../../../../tests/golden_ir/linear_regression.json");
    let data = include_str!("../../../../tests/golden_ir/data/linear_regression.json");
    for _ in 0..WARMUP {
        black_box(model_data_fingerprint(black_box(model), black_box(data)));
    }
    let iterations = 100_000;
    let samples = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(model_data_fingerprint(black_box(model), black_box(data)));
            }
            start.elapsed()
        })
        .collect();
    let elapsed = median(samples);
    let bytes = (b"bayescycle-model-data-v1\n".len() + model.len() + 1 + data.len()) as f64
        * iterations as f64;
    let seconds = elapsed.as_secs_f64();
    println!(
        "model_data_linear_regression,{}, {iterations},{seconds:.9},{:.2}",
        bytes as usize / iterations,
        bytes / seconds
    );
}

fn main() {
    println!("target,input_bytes,iterations,median_seconds,bytes_per_second");
    for len in [0, 3, 55, 56, 63, 64, 65, 1_024, 65_536, 1_048_576] {
        measure_bytes(&format!("sha256_{len}"), &deterministic_bytes(len));
    }
    measure_framed();
}
