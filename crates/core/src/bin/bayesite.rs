//! `bayesite` — subprocess sampling protocol over serialized IR.
//!
//! Usage:
//!   bayesite sample --model <ir.json|-> --data <data.json> [--seed N]
//!       [--chains C] [--warmup W] [--draws D] [--max-treedepth T]
//!       [--target-accept A]
//!
//! Output is the v0-provisional NDJSON protocol (see `protocol.rs`) on
//! stdout. Errors are a single JSON object on stderr with a nonzero exit
//! code; messages state what to change.
//!
//! Parallelism lives here, not in the library: one thread per chain.

use std::io::Read;
use std::io::Write;

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::protocol;
use bayesite_core::sampler::{sample, ChainDraws, Settings};

fn usage_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

struct Args {
    model_path: String,
    data_path: String,
    seed: u64,
    chains: u64,
    settings: Settings,
}

fn parse_args(argv: &[String]) -> Result<Args, Error> {
    if argv.is_empty() || argv[0] != "sample" {
        return Err(usage_error(
            "usage: bayesite sample --model <ir.json|-> --data <data.json> [--seed N] \
             [--chains C] [--warmup W] [--draws D] [--max-treedepth T] [--target-accept A]",
        ));
    }
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut seed = 0u64;
    let mut chains = 4u64;
    let mut settings = Settings::default();

    let mut iter = argv[1..].iter();
    while let Some(flag) = iter.next() {
        let mut value_for = |name: &str| -> Result<&String, Error> {
            iter.next()
                .ok_or_else(|| usage_error(format!("flag {name} needs a value")))
        };
        match flag.as_str() {
            "--model" => model_path = Some(value_for("--model")?.clone()),
            "--data" => data_path = Some(value_for("--data")?.clone()),
            "--seed" => {
                seed = value_for("--seed")?
                    .parse()
                    .map_err(|_| usage_error("--seed must be an unsigned integer"))?
            }
            "--chains" => {
                chains = value_for("--chains")?
                    .parse()
                    .map_err(|_| usage_error("--chains must be a positive integer"))?
            }
            "--warmup" => {
                settings.num_warmup = value_for("--warmup")?
                    .parse()
                    .map_err(|_| usage_error("--warmup must be a non-negative integer"))?
            }
            "--draws" => {
                settings.num_draws = value_for("--draws")?
                    .parse()
                    .map_err(|_| usage_error("--draws must be a positive integer"))?
            }
            "--max-treedepth" => {
                settings.max_treedepth = value_for("--max-treedepth")?
                    .parse()
                    .map_err(|_| usage_error("--max-treedepth must be a positive integer"))?
            }
            "--target-accept" => {
                settings.target_accept = value_for("--target-accept")?
                    .parse()
                    .map_err(|_| usage_error("--target-accept must be a number in (0, 1)"))?
            }
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite sample` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let data_path = data_path.ok_or_else(|| usage_error("--data is required (a path)"))?;
    if chains == 0 {
        return Err(usage_error("--chains must be at least 1"));
    }
    Ok(Args {
        model_path,
        data_path,
        seed,
        chains,
        settings,
    })
}

fn read_input(path: &str) -> Result<String, Error> {
    if path == "-" {
        let mut buffer = String::new();
        std::io::stdin()
            .read_to_string(&mut buffer)
            .map_err(|e| usage_error(format!("cannot read stdin: {e}")))?;
        Ok(buffer)
    } else {
        std::fs::read_to_string(path)
            .map_err(|e| usage_error(format!("cannot read \"{path}\": {e}")))
    }
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = parse_args(&argv)?;

    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&read_input(&args.data_path)?)?;
    let data = data_from_json(&data_doc)?;
    let posterior = Posterior::new(meta, data)?;

    // One thread per chain; the library itself stays single-threaded.
    let results: Vec<Result<ChainDraws, Error>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..args.chains)
            .map(|chain_id| {
                let posterior = &posterior;
                let settings = &args.settings;
                let seed = args.seed;
                scope.spawn(move || sample(posterior, settings, seed, chain_id))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("chain thread panicked"))
            .collect()
    });
    let mut chains: Vec<(u64, ChainDraws)> = Vec::with_capacity(results.len());
    for (chain_id, result) in results.into_iter().enumerate() {
        chains.push((chain_id as u64, result?));
    }

    let lines = protocol::ndjson_lines(&posterior, &args.settings, args.seed, &chains)?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in lines {
        writeln!(out, "{line}").map_err(|e| {
            Error::new(
                ErrorKind::InvalidSettings,
                format!("cannot write output: {e}"),
            )
        })?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        let payload = Value::Object(vec![
            (
                "error".to_string(),
                Value::Str(error.kind.name().to_string()),
            ),
            ("message".to_string(), Value::Str(error.message.clone())),
        ]);
        let text =
            json::write(&payload).unwrap_or_else(|_| "{\"error\":\"InvalidSettings\"}".to_string());
        eprintln!("{text}");
        std::process::exit(1);
    }
}
