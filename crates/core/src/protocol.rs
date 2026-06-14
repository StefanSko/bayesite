//! The v0-provisional NDJSON draws protocol, shared by the CLI and the
//! wasm ABI.
//!
//! Lines: a header object (`draws_format: "v0-provisional"`, parameter
//! names/shapes, packing order, settings, seed, chain count), one object
//! per draw with constrained values keyed by parameter, and a trailer
//! with per-chain diagnostics plus cross-chain R-hat/ESS. The marker is
//! mandatory: the real fit-artifact format is defined elsewhere, and
//! nothing may grow load-bearing dependencies on this one unnoticed.

use crate::diagnostics;
use crate::error::Error;
use crate::json::{self, Value};
use crate::model::Posterior;
use crate::sampler::{ChainDraws, Settings};

/// One constrained draw: (name, shape, values) per parameter.
type ConstrainedDraw = Vec<(String, Vec<usize>, Vec<f64>)>;

fn tensor_to_value(shape: &[usize], data: &[f64]) -> Value {
    if shape.is_empty() {
        Value::Float(data[0])
    } else {
        Value::Array(data.iter().map(|&v| Value::Float(v)).collect())
    }
}

fn header_value(
    packing: &[(String, Vec<usize>)],
    settings: &Settings,
    seed: u64,
    chains: usize,
) -> Value {
    Value::Object(vec![
        (
            "draws_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "params".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, shape)| {
                        Value::Object(vec![
                            ("name".to_string(), Value::Str(name.clone())),
                            (
                                "shape".to_string(),
                                Value::Array(shape.iter().map(|&d| Value::Int(d as i64)).collect()),
                            ),
                        ])
                    })
                    .collect(),
            ),
        ),
        (
            "packing".to_string(),
            Value::Array(
                packing
                    .iter()
                    .map(|(name, _)| Value::Str(name.clone()))
                    .collect(),
            ),
        ),
        (
            "settings".to_string(),
            Value::Object(vec![
                (
                    "num_warmup".to_string(),
                    Value::Int(settings.num_warmup as i64),
                ),
                (
                    "num_draws".to_string(),
                    Value::Int(settings.num_draws as i64),
                ),
                (
                    "max_treedepth".to_string(),
                    Value::Int(settings.max_treedepth as i64),
                ),
                (
                    "target_accept".to_string(),
                    Value::Float(settings.target_accept),
                ),
            ]),
        ),
        ("seed".to_string(), Value::Int(seed as i64)),
        ("chains".to_string(), Value::Int(chains as i64)),
    ])
}

/// Render a complete run as NDJSON lines. `chains` pairs a chain id with
/// its draws; ids appear verbatim in the output (the CLI uses 0..C, a web
/// worker passes its own).
pub fn ndjson_lines(
    posterior: &Posterior,
    settings: &Settings,
    seed: u64,
    chains: &[(u64, ChainDraws)],
) -> Result<Vec<String>, Error> {
    let packing = posterior.packing();
    let mut lines =
        Vec::with_capacity(2 + chains.iter().map(|(_, c)| c.draws.len()).sum::<usize>());
    lines.push(json::write(&header_value(
        &packing,
        settings,
        seed,
        chains.len(),
    ))?);

    let mut constrained_chains: Vec<Vec<ConstrainedDraw>> = Vec::with_capacity(chains.len());
    for (_, chain) in chains {
        let mut constrained_draws = Vec::with_capacity(chain.draws.len());
        for q in &chain.draws {
            constrained_draws.push(
                posterior
                    .constrain(q)?
                    .into_iter()
                    .map(|(name, tensor)| (name, tensor.shape().to_vec(), tensor.data().to_vec()))
                    .collect::<ConstrainedDraw>(),
            );
        }
        constrained_chains.push(constrained_draws);
    }

    for ((chain_id, _), draws) in chains.iter().zip(&constrained_chains) {
        for (draw_id, constrained) in draws.iter().enumerate() {
            let values = Value::Object(
                constrained
                    .iter()
                    .map(|(name, shape, data)| (name.clone(), tensor_to_value(shape, data)))
                    .collect(),
            );
            let line = Value::Object(vec![
                ("chain".to_string(), Value::Int(*chain_id as i64)),
                ("draw".to_string(), Value::Int(draw_id as i64)),
                ("values".to_string(), values),
            ]);
            lines.push(json::write(&line)?);
        }
    }

    // Cross-chain R-hat / ESS per parameter: max over coordinates for
    // R-hat, min for ESS, matching jaxstanv5.diagnostics conventions.
    let mut rhat_entries = Vec::new();
    let mut ess_entries = Vec::new();
    for (param_idx, (name, shape)) in packing.iter().enumerate() {
        let size: usize = shape.iter().product::<usize>().max(1);
        let mut worst_rhat = f64::NEG_INFINITY;
        let mut worst_ess = f64::INFINITY;
        for coord in 0..size {
            let series: Vec<Vec<f64>> = constrained_chains
                .iter()
                .map(|draws| {
                    draws
                        .iter()
                        .map(|constrained| constrained[param_idx].2[coord])
                        .collect()
                })
                .collect();
            worst_rhat = worst_rhat.max(diagnostics::split_rhat(&series));
            worst_ess = worst_ess.min(diagnostics::effective_sample_size(&series));
        }
        rhat_entries.push((name.clone(), Value::Float(worst_rhat)));
        ess_entries.push((name.clone(), Value::Float(worst_ess)));
    }

    let chain_stats = Value::Array(
        chains
            .iter()
            .map(|(chain_id, chain)| {
                Value::Object(vec![
                    ("chain".to_string(), Value::Int(*chain_id as i64)),
                    (
                        "divergences".to_string(),
                        Value::Int(chain.divergences as i64),
                    ),
                    (
                        "treedepth_histogram".to_string(),
                        Value::Array(
                            chain
                                .treedepth_histogram
                                .iter()
                                .map(|&c| Value::Int(c as i64))
                                .collect(),
                        ),
                    ),
                    ("step_size".to_string(), Value::Float(chain.step_size)),
                    ("mean_accept".to_string(), Value::Float(chain.mean_accept)),
                ])
            })
            .collect(),
    );
    let trailer = Value::Object(vec![(
        "trailer".to_string(),
        Value::Object(vec![
            ("chains".to_string(), chain_stats),
            ("rhat".to_string(), Value::Object(rhat_entries)),
            ("ess".to_string(), Value::Object(ess_entries)),
        ]),
    )]);
    lines.push(json::write(&trailer)?);
    Ok(lines)
}

/// Handle one wasm-boundary request (a JSON document) and render the
/// response text. Pure string-to-string so it is natively testable; the
/// unsafe pointer shims in `wasm_abi.rs` only move bytes.
///
/// Commands:
/// - `{"command":"sample","model":<ir>,"data":<data>,"settings":{...},
///    "seed":N,"chain_id":N}` -> v0-provisional NDJSON (one chain).
/// - `{"command":"diagnostics","series":[[...],...]}` -> cross-chain
///   `{"rhat":x,"ess":y}` for one scalar coordinate.
///
/// Errors come back as a single JSON object `{"error","message"}`.
pub fn handle_request(text: &str) -> String {
    match handle_request_inner(text) {
        Ok(response) => response,
        Err(error) => {
            let payload = Value::Object(vec![
                (
                    "error".to_string(),
                    Value::Str(error.kind.name().to_string()),
                ),
                ("message".to_string(), Value::Str(error.message)),
            ]);
            json::write(&payload).unwrap_or_else(|_| "{\"error\":\"MalformedJson\"}".to_string())
        }
    }
}

fn handle_request_inner(text: &str) -> Result<String, Error> {
    use crate::error::ErrorKind;
    use crate::ir::decode_model;
    use crate::model::data_from_json;
    use crate::sampler::sample;

    let request = json::parse(text)?;
    let invalid = |message: &str| Error::new(ErrorKind::InvalidSettings, message.to_string());
    match request.get("command").and_then(Value::as_str) {
        Some("sample") => {
            let model = request
                .get("model")
                .ok_or_else(|| invalid("request needs a \"model\" IR document"))?;
            let meta = decode_model(model)?;
            let data = data_from_json(
                request
                    .get("data")
                    .ok_or_else(|| invalid("request needs a \"data\" object"))?,
            )?;
            let posterior = Posterior::new(meta, data)?;
            let mut settings = Settings::default();
            if let Some(spec) = request.get("settings") {
                if let Some(v) = spec.get("num_warmup").and_then(Value::as_i64) {
                    settings.num_warmup = v.max(0) as usize;
                }
                if let Some(v) = spec.get("num_draws").and_then(Value::as_i64) {
                    settings.num_draws = v.max(0) as usize;
                }
                if let Some(v) = spec.get("max_treedepth").and_then(Value::as_i64) {
                    settings.max_treedepth = v.max(0) as usize;
                }
                if let Some(v) = spec.get("target_accept").and_then(Value::as_f64) {
                    settings.target_accept = v;
                }
            }
            let seed = request
                .get("seed")
                .and_then(Value::as_i64)
                .ok_or_else(|| invalid("request needs an integer \"seed\""))?
                as u64;
            let chain_id = request.get("chain_id").and_then(Value::as_i64).unwrap_or(0) as u64;
            let draws = sample(&posterior, &settings, seed, chain_id)?;
            let lines = ndjson_lines(&posterior, &settings, seed, &[(chain_id, draws)])?;
            Ok(lines.join("\n"))
        }
        Some("diagnostics") => {
            let series: Vec<Vec<f64>> = request
                .get("series")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("request needs \"series\": an array of chains"))?
                .iter()
                .map(|chain| {
                    chain
                        .as_array()
                        .ok_or_else(|| invalid("each series entry must be an array of numbers"))?
                        .iter()
                        .map(|v| {
                            v.as_f64()
                                .ok_or_else(|| invalid("series values must be numbers"))
                        })
                        .collect()
                })
                .collect::<Result<_, Error>>()?;
            if series.is_empty() || series[0].len() < 4 {
                return Err(invalid(
                    "series needs at least one chain of at least 4 draws",
                ));
            }
            let response = Value::Object(vec![
                (
                    "rhat".to_string(),
                    Value::Float(diagnostics::split_rhat(&series)),
                ),
                (
                    "ess".to_string(),
                    Value::Float(diagnostics::effective_sample_size(&series)),
                ),
            ]);
            json::write(&response)
        }
        _ => Err(invalid(
            "request needs \"command\": \"sample\" or \"diagnostics\"",
        )),
    }
}
