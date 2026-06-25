//! `bayesite` — subprocess sampling protocol over serialized IR.
//!
//! Usage:
//!   bayesite sample --model <ir.json|-> --data <data.json|-> [--seed N]
//!       [--chains C] [--warmup W] [--draws D] [--max-treedepth T]
//!       [--target-accept A] [--out <fit.jsonl|->]
//!   bayesite diagnose --fit <fit.jsonl|-> [--out <diagnostics.json|->]
//!   bayesite prior-predictive --model <ir.json|-> --data <data.json|->
//!       [--seed N] [--draws D] [--out <pp.jsonl|->]
//!   bayesite posterior-predictive --model <ir.json|-> --data <data.json|->
//!       --fit <fit.jsonl|-> [--seed N] [--out <yrep.jsonl|->]
//!   bayesite posterior-check --model <ir.json|-> --data <data.json|->
//!       --fit <fit.jsonl|-> [--seed N] [--out <ppc.json|->]
//!   bayesite simulate --model <ir.json|-> --data <data.json|->
//!       --truth <truth.json|-> [--seed N] [--out <data.json|->]
//!   bayesite recover-check --fit <fit.jsonl|-> --truth <truth.json|->
//!       [--targets <targets.json|->] [--interval P] [--out <report.json|->]
//!   bayesite recover --model <ir.json|-> --scenario <scenario.json|->
//!       [--out <report.json|->]
//!   bayesite sbc --model <ir.json|-> --scenario <scenario.json|->
//!       [--replicates N] [--out <report.json|->]
//!
//! `sample` writes the v0-provisional NDJSON protocol (see `protocol.rs`).
//! `diagnose`, `recover-check`, `recover`, and `sbc` write one v0-provisional
//! JSON object; `simulate` writes a plain data document; `prior-predictive`
//! writes v0-provisional NDJSON. `-` means stdout/stdin.
//! Errors are a single JSON object on stderr with a nonzero exit code; messages
//! state what to change.
//!
//! Parallelism lives here, not in the library: one thread per chain.

use std::fmt::Write as FmtWrite;
use std::io::Read;
use std::io::Write;

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, data_to_json, DataValue, Posterior};
use bayesite_core::predictive::{
    posterior_check_report_with_model_data_fingerprint,
    posterior_predictive_ndjson_lines_with_model_data_fingerprint, prior_predictive_ndjson_lines,
    simulate_data_from_truth, PriorPredictiveSettings,
};
use bayesite_core::protocol;
use bayesite_core::sampler::{sample, ChainDraws, Settings};
use bayesite_core::workflow::{recover_report, sbc_report, RecoverSettings, SbcSettings};

fn usage_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

const SHA256_INITIAL_STATE: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

const SHA256_ROUND_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_digest(input: &[u8]) -> [u8; 32] {
    let mut state = SHA256_INITIAL_STATE;
    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut message = input.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());

    let mut words = [0u32; 64];
    for chunk in message.chunks_exact(64) {
        for (index, word) in words[..16].iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let mut a = state[0];
        let mut b = state[1];
        let mut c = state[2];
        let mut d = state[3];
        let mut e = state[4];
        let mut f = state[5];
        let mut g = state[6];
        let mut h = state[7];

        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_ROUND_CONSTANTS[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut digest = [0u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn model_data_fingerprint(model_text: &str, data_text: &str) -> String {
    let mut input = Vec::new();
    input.extend_from_slice(b"bayescycle-model-data-v1\n");
    input.extend_from_slice(model_text.as_bytes());
    input.push(b'\n');
    input.extend_from_slice(data_text.as_bytes());
    let digest = sha256_digest(&input);
    let mut out = String::from("sha256:");
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

fn parse_artifact_seed(text: &str) -> Result<u64, Error> {
    let seed: u128 = text
        .parse()
        .map_err(|_| usage_error("--seed must be an unsigned integer"))?;
    if seed > i64::MAX as u128 {
        Err(usage_error(
            "--seed must be in 0..=9223372036854775807 because artifacts report seeds as JSON integers",
        ))
    } else {
        Ok(seed as u64)
    }
}

fn parse_reportable_draw_count(text: &str, name: &str, artifact: &str) -> Result<usize, Error> {
    let value: u128 = text
        .parse()
        .map_err(|_| usage_error(format!("{name} must be a positive integer")))?;
    if value > i64::MAX as u128 {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because {artifact} report draw counts as JSON integers"
        )))
    } else {
        Ok(value as usize)
    }
}

fn parse_reportable_warmup_count(text: &str, name: &str, artifact: &str) -> Result<usize, Error> {
    let value: u128 = text
        .parse()
        .map_err(|_| usage_error(format!("{name} must be a non-negative integer")))?;
    if value > i64::MAX as u128 {
        Err(usage_error(format!(
            "{name} must be in 0..=9223372036854775807 because {artifact} report warmup counts as JSON integers"
        )))
    } else {
        Ok(value as usize)
    }
}

fn parse_sample_chains(text: &str, name: &str, artifact: &str) -> Result<u64, Error> {
    let value: u128 = text
        .parse()
        .map_err(|_| usage_error(format!("{name} must be a positive integer")))?;
    if value > i64::MAX as u128 {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because {artifact} report chain counts as JSON integers"
        )))
    } else {
        Ok(value as u64)
    }
}

fn parse_sbc_replicates(text: &str, name: &str) -> Result<usize, Error> {
    let value: u128 = text
        .parse()
        .map_err(|_| usage_error(format!("{name} must be a positive integer")))?;
    if value > i64::MAX as u128 {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because sbc reports replicates as a JSON integer"
        )))
    } else {
        Ok(value as usize)
    }
}

fn validate_target_accept(value: f64, name: &str) -> Result<(), Error> {
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(usage_error(format!("{name} must be in (0, 1)")))
    }
}

fn validate_diagnostic_draws(value: usize, name: &str, artifact: &str) -> Result<(), Error> {
    if value >= 4 {
        Ok(())
    } else {
        Err(usage_error(format!(
            "{name} must be at least 4 because {artifact} include diagnostics"
        )))
    }
}

fn validate_reportable_draw_count(value: usize, name: &str, artifact: &str) -> Result<(), Error> {
    if value > i64::MAX as usize {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because {artifact} report draw counts as JSON integers"
        )))
    } else {
        Ok(())
    }
}

fn validate_reportable_warmup_count(value: usize, name: &str, artifact: &str) -> Result<(), Error> {
    if value > i64::MAX as usize {
        Err(usage_error(format!(
            "{name} must be in 0..=9223372036854775807 because {artifact} report warmup counts as JSON integers"
        )))
    } else {
        Ok(())
    }
}

fn validate_max_treedepth(value: usize, name: &str) -> Result<(), Error> {
    if (1..=20).contains(&value) {
        Ok(())
    } else {
        Err(usage_error(format!("{name} must be in 1..=20")))
    }
}

fn parse_max_treedepth(text: &str, name: &str) -> Result<usize, Error> {
    let parsed: u128 = text
        .parse()
        .map_err(|_| usage_error(format!("{name} must be a positive integer")))?;
    usize::try_from(parsed).map_err(|_| usage_error(format!("{name} must be in 1..=20")))
}

fn validate_sample_chains(value: u64, name: &str, artifact: &str) -> Result<(), Error> {
    if value == 0 {
        Err(usage_error(format!("{name} must be at least 1")))
    } else if value > i64::MAX as u64 {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because {artifact} report chain counts as JSON integers"
        )))
    } else {
        Ok(())
    }
}

fn validate_sbc_replicates(value: usize, name: &str) -> Result<(), Error> {
    if value == 0 {
        Err(usage_error(format!("{name} must be at least 1")))
    } else if value > i64::MAX as usize {
        Err(usage_error(format!(
            "{name} must be in 1..=9223372036854775807 because sbc reports replicates as a JSON integer"
        )))
    } else {
        Ok(())
    }
}

enum Command {
    Sample(SampleArgs),
    Diagnose(DiagnoseArgs),
    PriorPredictive(PriorPredictiveArgs),
    PosteriorPredictive(PosteriorPredictiveArgs),
    PosteriorCheck(PosteriorCheckArgs),
    Simulate(SimulateArgs),
    RecoverCheck(RecoverCheckArgs),
    Recover(RecoverArgs),
    Sbc(SbcArgs),
}

struct SampleArgs {
    model_path: String,
    data_path: String,
    out_path: String,
    seed: u64,
    chains: u64,
    settings: Settings,
}

struct DiagnoseArgs {
    fit_path: String,
    out_path: String,
}

struct PriorPredictiveArgs {
    model_path: String,
    data_path: String,
    out_path: String,
    seed: u64,
    settings: PriorPredictiveSettings,
}

struct PosteriorPredictiveArgs {
    model_path: String,
    data_path: String,
    fit_path: String,
    out_path: String,
    seed: u64,
}

struct PosteriorCheckArgs {
    model_path: String,
    data_path: String,
    fit_path: String,
    out_path: String,
    seed: u64,
}

struct SimulateArgs {
    model_path: String,
    data_path: String,
    truth_path: String,
    out_path: String,
    seed: u64,
}

struct RecoverCheckArgs {
    fit_path: String,
    truth_path: String,
    targets_path: Option<String>,
    out_path: String,
    interval: f64,
}

struct RecoverArgs {
    model_path: String,
    scenario_path: String,
    out_path: String,
}

struct SbcArgs {
    model_path: String,
    scenario_path: String,
    out_path: String,
    replicates_override: Option<usize>,
}

struct RecoverScenario {
    data: Vec<(String, DataValue)>,
    settings: RecoverSettings,
    seed: u64,
}

struct SbcScenario {
    data: Vec<(String, DataValue)>,
    settings: SbcSettings,
    seed: u64,
}

fn usage() -> &'static str {
    "usage: bayesite sample --model <ir.json|-> --data <data.json|-> [--seed N] \
     [--chains C] [--warmup W] [--draws D] [--max-treedepth T] [--target-accept A] \
     [--out <fit.jsonl|->]\n\
     usage: bayesite diagnose --fit <fit.jsonl|-> [--out <diagnostics.json|->]\n\
     usage: bayesite prior-predictive --model <ir.json|-> --data <data.json|-> \
     [--seed N] [--draws D] [--out <pp.jsonl|->]\n\
     usage: bayesite posterior-predictive --model <ir.json|-> --data <data.json|-> \
     --fit <fit.jsonl|-> [--seed N] [--out <yrep.jsonl|->]\n\
     usage: bayesite posterior-check --model <ir.json|-> --data <data.json|-> \
     --fit <fit.jsonl|-> [--seed N] [--out <ppc.json|->]\n\
     usage: bayesite simulate --model <ir.json|-> --data <data.json|-> \
     --truth <truth.json|-> [--seed N] [--out <data.json|->]\n\
     usage: bayesite recover-check --fit <fit.jsonl|-> --truth <truth.json|-> \
     [--targets <targets.json|->] [--interval P] [--out <report.json|->]\n\
     usage: bayesite recover --model <ir.json|-> --scenario <scenario.json|-> \
     [--out <report.json|->]\n\
     usage: bayesite sbc --model <ir.json|-> --scenario <scenario.json|-> \
     [--replicates N] [--out <report.json|->]"
}

fn parse_args(argv: &[String]) -> Result<Command, Error> {
    let Some(command) = argv.first() else {
        return Err(usage_error(format!("missing command; {}", usage())));
    };
    match command.as_str() {
        "sample" => parse_sample_args(&argv[1..]).map(Command::Sample),
        "diagnose" => parse_diagnose_args(&argv[1..]).map(Command::Diagnose),
        "prior-predictive" => parse_prior_predictive_args(&argv[1..]).map(Command::PriorPredictive),
        "posterior-predictive" => {
            parse_posterior_predictive_args(&argv[1..]).map(Command::PosteriorPredictive)
        }
        "posterior-check" => parse_posterior_check_args(&argv[1..]).map(Command::PosteriorCheck),
        "simulate" => parse_simulate_args(&argv[1..]).map(Command::Simulate),
        "recover-check" => parse_recover_check_args(&argv[1..]).map(Command::RecoverCheck),
        "recover" => parse_recover_args(&argv[1..]).map(Command::Recover),
        "sbc" => parse_sbc_args(&argv[1..]).map(Command::Sbc),
        other => Err(usage_error(format!(
            "unknown command \"{other}\"; {}",
            usage()
        ))),
    }
}

fn validate_single_stdin_input(command: &str, inputs: &[(&str, &str)]) -> Result<(), Error> {
    let mut stdin_input: Option<&str> = None;
    for (name, path) in inputs {
        if *path == "-" {
            if let Some(first) = stdin_input {
                return Err(usage_error(format!(
                    "{command} accepts at most one stdin input; use a path for {name} when {first} is -"
                )));
            }
            stdin_input = Some(name);
        }
    }
    Ok(())
}

fn reject_duplicate_flags(command: &str, argv: &[String], flags: &[&str]) -> Result<(), Error> {
    let mut seen: Vec<&str> = Vec::new();
    let mut index = 0usize;
    while index < argv.len() {
        let flag = argv[index].as_str();
        if flags.contains(&flag) {
            if seen.contains(&flag) {
                return Err(usage_error(format!(
                    "{command} has duplicate flag {flag}; pass it once"
                )));
            }
            seen.push(flag);
            index += 2;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn value_for_flag<'a>(
    iter: &mut std::slice::Iter<'a, String>,
    name: &str,
) -> Result<&'a String, Error> {
    let value = iter
        .next()
        .ok_or_else(|| usage_error(format!("flag {name} needs a value")))?;
    if value.starts_with("--") {
        Err(usage_error(format!(
            "flag {name} needs a value before {value}"
        )))
    } else {
        Ok(value)
    }
}

fn parse_sample_args(argv: &[String]) -> Result<SampleArgs, Error> {
    reject_duplicate_flags(
        "sample",
        argv,
        &[
            "--model",
            "--data",
            "--out",
            "--seed",
            "--chains",
            "--warmup",
            "--draws",
            "--max-treedepth",
            "--target-accept",
        ],
    )?;
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut seed = 0u64;
    let mut chains = 4u64;
    let mut settings = Settings::default();

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--data" => data_path = Some(value_for_flag(&mut iter, "--data")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--seed" => seed = parse_artifact_seed(value_for_flag(&mut iter, "--seed")?)?,
            "--chains" => {
                chains = parse_sample_chains(
                    value_for_flag(&mut iter, "--chains")?,
                    "--chains",
                    "sample artifacts",
                )?
            }
            "--warmup" => {
                settings.num_warmup = parse_reportable_warmup_count(
                    value_for_flag(&mut iter, "--warmup")?,
                    "--warmup",
                    "sample artifacts",
                )?
            }
            "--draws" => {
                settings.num_draws = parse_reportable_draw_count(
                    value_for_flag(&mut iter, "--draws")?,
                    "--draws",
                    "sample artifacts",
                )?
            }
            "--max-treedepth" => {
                settings.max_treedepth = parse_max_treedepth(
                    value_for_flag(&mut iter, "--max-treedepth")?,
                    "--max-treedepth",
                )?
            }
            "--target-accept" => {
                settings.target_accept = value_for_flag(&mut iter, "--target-accept")?
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
    let data_path =
        data_path.ok_or_else(|| usage_error("--data is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "sample",
        &[("--model", &model_path), ("--data", &data_path)],
    )?;
    validate_sample_chains(chains, "--chains", "sample artifacts")?;
    if settings.num_draws == 0 {
        return Err(usage_error("--draws must be at least 1"));
    }
    validate_reportable_draw_count(settings.num_draws, "--draws", "sample artifacts")?;
    validate_reportable_warmup_count(settings.num_warmup, "--warmup", "sample artifacts")?;
    validate_diagnostic_draws(settings.num_draws, "--draws", "sample artifacts")?;
    if settings.max_treedepth == 0 {
        return Err(usage_error("--max-treedepth must be at least 1"));
    }
    validate_max_treedepth(settings.max_treedepth, "--max-treedepth")?;
    validate_target_accept(settings.target_accept, "--target-accept")?;
    Ok(SampleArgs {
        model_path,
        data_path,
        out_path,
        seed,
        chains,
        settings,
    })
}

fn parse_diagnose_args(argv: &[String]) -> Result<DiagnoseArgs, Error> {
    reject_duplicate_flags("diagnose", argv, &["--fit", "--out"])?;
    let mut fit_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--fit" => fit_path = Some(value_for_flag(&mut iter, "--fit")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite diagnose` usage"
                )))
            }
        }
    }
    let fit_path =
        fit_path.ok_or_else(|| usage_error("--fit is required (a path or - for stdin)"))?;
    Ok(DiagnoseArgs { fit_path, out_path })
}

fn parse_prior_predictive_args(argv: &[String]) -> Result<PriorPredictiveArgs, Error> {
    reject_duplicate_flags(
        "prior-predictive",
        argv,
        &["--model", "--data", "--out", "--seed", "--draws"],
    )?;
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut seed = 0u64;
    let mut settings = PriorPredictiveSettings::default();

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--data" => data_path = Some(value_for_flag(&mut iter, "--data")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--seed" => seed = parse_artifact_seed(value_for_flag(&mut iter, "--seed")?)?,
            "--draws" => {
                settings.num_draws = parse_reportable_draw_count(
                    value_for_flag(&mut iter, "--draws")?,
                    "--draws",
                    "prior-predictive artifacts",
                )?
            }
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite prior-predictive` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let data_path =
        data_path.ok_or_else(|| usage_error("--data is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "prior-predictive",
        &[("--model", &model_path), ("--data", &data_path)],
    )?;
    if settings.num_draws == 0 {
        return Err(usage_error("--draws must be at least 1"));
    }
    validate_reportable_draw_count(settings.num_draws, "--draws", "prior-predictive artifacts")?;
    Ok(PriorPredictiveArgs {
        model_path,
        data_path,
        out_path,
        seed,
        settings,
    })
}

fn parse_posterior_predictive_args(argv: &[String]) -> Result<PosteriorPredictiveArgs, Error> {
    reject_duplicate_flags(
        "posterior-predictive",
        argv,
        &["--model", "--data", "--fit", "--out", "--seed"],
    )?;
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut fit_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut seed = 0u64;

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--data" => data_path = Some(value_for_flag(&mut iter, "--data")?.clone()),
            "--fit" => fit_path = Some(value_for_flag(&mut iter, "--fit")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--seed" => seed = parse_artifact_seed(value_for_flag(&mut iter, "--seed")?)?,
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite posterior-predictive` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let data_path =
        data_path.ok_or_else(|| usage_error("--data is required (a path or - for stdin)"))?;
    let fit_path =
        fit_path.ok_or_else(|| usage_error("--fit is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "posterior-predictive",
        &[
            ("--model", &model_path),
            ("--data", &data_path),
            ("--fit", &fit_path),
        ],
    )?;
    Ok(PosteriorPredictiveArgs {
        model_path,
        data_path,
        fit_path,
        out_path,
        seed,
    })
}

fn parse_posterior_check_args(argv: &[String]) -> Result<PosteriorCheckArgs, Error> {
    let args = parse_posterior_predictive_args(argv)?;
    Ok(PosteriorCheckArgs {
        model_path: args.model_path,
        data_path: args.data_path,
        fit_path: args.fit_path,
        out_path: args.out_path,
        seed: args.seed,
    })
}

fn parse_simulate_args(argv: &[String]) -> Result<SimulateArgs, Error> {
    reject_duplicate_flags(
        "simulate",
        argv,
        &["--model", "--data", "--truth", "--out", "--seed"],
    )?;
    let mut model_path: Option<String> = None;
    let mut data_path: Option<String> = None;
    let mut truth_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut seed = 0u64;

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--data" => data_path = Some(value_for_flag(&mut iter, "--data")?.clone()),
            "--truth" => truth_path = Some(value_for_flag(&mut iter, "--truth")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--seed" => seed = parse_artifact_seed(value_for_flag(&mut iter, "--seed")?)?,
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite simulate` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let data_path =
        data_path.ok_or_else(|| usage_error("--data is required (a path or - for stdin)"))?;
    let truth_path =
        truth_path.ok_or_else(|| usage_error("--truth is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "simulate",
        &[
            ("--model", &model_path),
            ("--data", &data_path),
            ("--truth", &truth_path),
        ],
    )?;
    Ok(SimulateArgs {
        model_path,
        data_path,
        truth_path,
        out_path,
        seed,
    })
}

fn parse_recover_check_args(argv: &[String]) -> Result<RecoverCheckArgs, Error> {
    reject_duplicate_flags(
        "recover-check",
        argv,
        &["--fit", "--truth", "--targets", "--out", "--interval"],
    )?;
    let mut fit_path: Option<String> = None;
    let mut truth_path: Option<String> = None;
    let mut targets_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut interval = 0.8f64;

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--fit" => fit_path = Some(value_for_flag(&mut iter, "--fit")?.clone()),
            "--truth" => truth_path = Some(value_for_flag(&mut iter, "--truth")?.clone()),
            "--targets" => targets_path = Some(value_for_flag(&mut iter, "--targets")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--interval" => {
                interval = value_for_flag(&mut iter, "--interval")?
                    .parse()
                    .map_err(|_| usage_error("--interval must be a number in (0, 1)"))?
            }
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite recover-check` usage"
                )))
            }
        }
    }
    let fit_path =
        fit_path.ok_or_else(|| usage_error("--fit is required (a path or - for stdin)"))?;
    let truth_path =
        truth_path.ok_or_else(|| usage_error("--truth is required (a path or - for stdin)"))?;
    let mut inputs = vec![
        ("--fit", fit_path.as_str()),
        ("--truth", truth_path.as_str()),
    ];
    if let Some(targets_path) = &targets_path {
        inputs.push(("--targets", targets_path.as_str()));
    }
    validate_single_stdin_input("recover-check", &inputs)?;
    validate_target_accept(interval, "--interval")?;
    Ok(RecoverCheckArgs {
        fit_path,
        truth_path,
        targets_path,
        out_path,
        interval,
    })
}

fn parse_recover_args(argv: &[String]) -> Result<RecoverArgs, Error> {
    reject_duplicate_flags("recover", argv, &["--model", "--scenario", "--out"])?;
    let mut model_path: Option<String> = None;
    let mut scenario_path: Option<String> = None;
    let mut out_path = "-".to_string();

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--scenario" => scenario_path = Some(value_for_flag(&mut iter, "--scenario")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite recover` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let scenario_path = scenario_path
        .ok_or_else(|| usage_error("--scenario is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "recover",
        &[("--model", &model_path), ("--scenario", &scenario_path)],
    )?;
    Ok(RecoverArgs {
        model_path,
        scenario_path,
        out_path,
    })
}

fn parse_sbc_args(argv: &[String]) -> Result<SbcArgs, Error> {
    reject_duplicate_flags(
        "sbc",
        argv,
        &["--model", "--scenario", "--out", "--replicates"],
    )?;
    let mut model_path: Option<String> = None;
    let mut scenario_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut replicates_override = None;

    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--scenario" => scenario_path = Some(value_for_flag(&mut iter, "--scenario")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--replicates" => {
                let value = parse_sbc_replicates(
                    value_for_flag(&mut iter, "--replicates")?,
                    "--replicates",
                )?;
                validate_sbc_replicates(value, "--replicates")?;
                replicates_override = Some(value);
            }
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite sbc` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let scenario_path = scenario_path
        .ok_or_else(|| usage_error("--scenario is required (a path or - for stdin)"))?;
    validate_single_stdin_input(
        "sbc",
        &[("--model", &model_path), ("--scenario", &scenario_path)],
    )?;
    Ok(SbcArgs {
        model_path,
        scenario_path,
        out_path,
        replicates_override,
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

fn write_text(path: &str, text: &str) -> Result<(), Error> {
    if path == "-" {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "{text}").map_err(|e| {
            Error::new(
                ErrorKind::InvalidSettings,
                format!("cannot write output: {e}"),
            )
        })
    } else {
        std::fs::write(path, format!("{text}\n"))
            .map_err(|e| usage_error(format!("cannot write \"{path}\": {e}")))
    }
}

fn write_lines(path: &str, lines: Vec<String>) -> Result<(), Error> {
    if path == "-" {
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
    } else {
        let mut out = std::fs::File::create(path)
            .map_err(|e| usage_error(format!("cannot create \"{path}\": {e}")))?;
        for line in lines {
            writeln!(out, "{line}").map_err(|e| {
                Error::new(
                    ErrorKind::InvalidSettings,
                    format!("cannot write \"{path}\": {e}"),
                )
            })?;
        }
        Ok(())
    }
}

fn cli_data_from_json(data_doc: &Value, context: &str) -> Result<Vec<(String, DataValue)>, Error> {
    if !matches!(data_doc, Value::Object(_)) {
        return Err(usage_error(format!("{context} data must be an object")));
    }
    data_from_json(data_doc)
}

fn run_sample(args: SampleArgs) -> Result<(), Error> {
    let model_text = read_input(&args.model_path)?;
    let data_text = read_input(&args.data_path)?;
    let fingerprint = model_data_fingerprint(&model_text, &data_text);
    let model_doc = json::parse(&model_text)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&data_text)?;
    let data = cli_data_from_json(&data_doc, "sample")?;
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

    let lines = protocol::ndjson_lines_with_model_data_fingerprint(
        &posterior,
        &args.settings,
        args.seed,
        &chains,
        Some(&fingerprint),
    )?;
    write_lines(&args.out_path, lines)
}

fn run_diagnose(args: DiagnoseArgs) -> Result<(), Error> {
    let text = protocol::diagnose_ndjson(&read_input(&args.fit_path)?)?;
    write_text(&args.out_path, &text)
}

fn run_prior_predictive(args: PriorPredictiveArgs) -> Result<(), Error> {
    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&read_input(&args.data_path)?)?;
    let data = cli_data_from_json(&data_doc, "prior-predictive")?;
    let lines = prior_predictive_ndjson_lines(meta, data, &args.settings, args.seed)?;
    write_lines(&args.out_path, lines)
}

fn run_posterior_predictive(args: PosteriorPredictiveArgs) -> Result<(), Error> {
    let model_text = read_input(&args.model_path)?;
    let data_text = read_input(&args.data_path)?;
    let fingerprint = model_data_fingerprint(&model_text, &data_text);
    let model_doc = json::parse(&model_text)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&data_text)?;
    let data = cli_data_from_json(&data_doc, "posterior-predictive")?;
    let fit = read_input(&args.fit_path)?;
    let lines = posterior_predictive_ndjson_lines_with_model_data_fingerprint(
        meta,
        data,
        &fit,
        args.seed,
        Some(&fingerprint),
    )?;
    write_lines(&args.out_path, lines)
}

fn run_posterior_check(args: PosteriorCheckArgs) -> Result<(), Error> {
    let model_text = read_input(&args.model_path)?;
    let data_text = read_input(&args.data_path)?;
    let fingerprint = model_data_fingerprint(&model_text, &data_text);
    let model_doc = json::parse(&model_text)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&data_text)?;
    let data = cli_data_from_json(&data_doc, "posterior-check")?;
    let fit = read_input(&args.fit_path)?;
    let report = posterior_check_report_with_model_data_fingerprint(
        meta,
        data,
        &fit,
        args.seed,
        Some(&fingerprint),
    )?;
    write_text(&args.out_path, &report)
}

fn run_simulate(args: SimulateArgs) -> Result<(), Error> {
    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let data_doc = json::parse(&read_input(&args.data_path)?)?;
    let data = cli_data_from_json(&data_doc, "simulate")?;
    let truth_doc = json::parse(&read_input(&args.truth_path)?)?;
    if !matches!(truth_doc, Value::Object(_)) {
        return Err(usage_error("simulate truth must be an object"));
    }
    let truth = data_from_json(&truth_doc)?;
    let generated = simulate_data_from_truth(meta, data, truth, args.seed)?;
    let data_doc = data_to_json(&generated, "simulate")?;
    let text = json::write(&data_doc)?;
    write_text(&args.out_path, &text)
}

fn run_recover_check(args: RecoverCheckArgs) -> Result<(), Error> {
    let fit = read_input(&args.fit_path)?;
    let truth_doc = json::parse(&read_input(&args.truth_path)?)?;
    if !matches!(truth_doc, Value::Object(_)) {
        return Err(usage_error("recover-check truth must be an object"));
    }
    let targets_doc = match &args.targets_path {
        Some(path) => Some(json::parse(&read_input(path)?)?),
        None => None,
    };
    let report =
        protocol::recover_check_report(&fit, &truth_doc, targets_doc.as_ref(), args.interval)?;
    write_text(&args.out_path, &report)
}

fn scenario_reportable_int(
    doc: &Value,
    name: &str,
    default: i64,
    label: &str,
    range_message: &str,
) -> Result<i64, Error> {
    let Some(value) = doc.get(name) else {
        return Ok(default);
    };
    match value {
        Value::Int(value) => Ok(*value),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(usage_error(range_message)),
        _ => Err(usage_error(format!("{label} must be an integer"))),
    }
}

fn scenario_float(doc: &Value, name: &str, default: f64, label: &str) -> Result<f64, Error> {
    match doc.get(name).and_then(Value::as_f64) {
        Some(value) => Ok(value),
        None if doc.get(name).is_none() => Ok(default),
        None => Err(usage_error(format!("{label} must be a number"))),
    }
}

fn scenario_seed(doc: &Value, context: &str) -> Result<u64, Error> {
    let Some(value) = doc.get("seed") else {
        return Err(usage_error(format!(
            "{context} scenario needs an integer seed"
        )));
    };
    match value {
        Value::Int(seed) if *seed >= 0 => Ok(*seed as u64),
        Value::Int(_) => Err(usage_error(format!(
            "{context} scenario seed must be non-negative"
        ))),
        Value::Float(number) if *number >= i64::MAX as f64 => Err(usage_error(format!(
            "{context} scenario seed must be in 0..=9223372036854775807 because workflow reports seeds as JSON integers"
        ))),
        _ => Err(usage_error(format!("{context} scenario needs an integer seed"))),
    }
}

fn reject_unknown_fields(doc: &Value, context: &str, allowed: &[&str]) -> Result<(), Error> {
    let Value::Object(entries) = doc else {
        return Err(usage_error(format!("{context} must be an object")));
    };
    for (index, (name, _)) in entries.iter().enumerate() {
        if entries[..index]
            .iter()
            .any(|(existing, _)| existing == name)
        {
            return Err(usage_error(format!(
                "{context} has duplicate field \"{name}\"; remove one"
            )));
        }
        if !allowed.contains(&name.as_str()) {
            return Err(usage_error(format!(
                "{context} has unknown field \"{name}\""
            )));
        }
    }
    Ok(())
}

fn positive_usize(value: i64, name: &str) -> Result<usize, Error> {
    if value < 1 {
        Err(usage_error(format!("{name} must be at least 1")))
    } else {
        Ok(value as usize)
    }
}

fn apply_sample_settings(
    sample_doc: &Value,
    sampler: &mut Settings,
    chains: &mut u64,
    context: &str,
) -> Result<(), Error> {
    if !matches!(sample_doc, Value::Object(_)) {
        return Err(usage_error(format!(
            "{context} scenario sample must be an object"
        )));
    }
    reject_unknown_fields(
        sample_doc,
        &format!("{context} scenario sample"),
        &[
            "chains",
            "warmup",
            "draws",
            "max_treedepth",
            "target_accept",
        ],
    )?;
    let chains_label = format!("{context} scenario sample.chains");
    let parsed_chains = scenario_reportable_int(
        sample_doc,
        "chains",
        *chains as i64,
        &chains_label,
        &format!(
            "{chains_label} must be in 1..=9223372036854775807 because workflow reports chains as JSON integers"
        ),
    )?;
    if parsed_chains < 1 {
        return Err(usage_error(format!(
            "{context} scenario sample.chains must be at least 1"
        )));
    }
    *chains = parsed_chains as u64;
    let warmup_label = format!("{context} scenario sample.warmup");
    let warmup = scenario_reportable_int(
        sample_doc,
        "warmup",
        sampler.num_warmup as i64,
        &warmup_label,
        &format!(
            "{warmup_label} must be in 0..=9223372036854775807 because workflow reports sample.warmup as a JSON integer"
        ),
    )?;
    if warmup < 0 {
        return Err(usage_error(format!(
            "{context} scenario sample.warmup must be non-negative"
        )));
    }
    sampler.num_warmup = warmup as usize;
    let draws_label = format!("{context} scenario sample.draws");
    sampler.num_draws = positive_usize(
        scenario_reportable_int(
            sample_doc,
            "draws",
            sampler.num_draws as i64,
            &draws_label,
            &format!(
                "{draws_label} must be in 1..=9223372036854775807 because workflow reports sample.draws as a JSON integer"
            ),
        )?,
        &draws_label,
    )?;
    validate_diagnostic_draws(sampler.num_draws, &draws_label, "workflow reports")?;
    let max_treedepth_label = format!("{context} scenario sample.max_treedepth");
    sampler.max_treedepth = positive_usize(
        scenario_reportable_int(
            sample_doc,
            "max_treedepth",
            sampler.max_treedepth as i64,
            &max_treedepth_label,
            &format!("{max_treedepth_label} must be in 1..=20"),
        )?,
        &max_treedepth_label,
    )?;
    validate_max_treedepth(sampler.max_treedepth, &max_treedepth_label)?;
    sampler.target_accept = scenario_float(
        sample_doc,
        "target_accept",
        sampler.target_accept,
        &format!("{context} scenario sample.target_accept"),
    )?;
    validate_target_accept(
        sampler.target_accept,
        &format!("{context} scenario sample.target_accept"),
    )?;
    Ok(())
}

fn parse_recover_scenario(document: &Value) -> Result<RecoverScenario, Error> {
    reject_unknown_fields(
        document,
        "recover scenario",
        &["recover_scenario", "data", "seed", "interval", "sample"],
    )?;
    if document.get("recover_scenario").and_then(Value::as_str) != Some("v0-provisional") {
        return Err(usage_error(
            "recover scenario needs recover_scenario \"v0-provisional\"",
        ));
    }
    let data_doc = document
        .get("data")
        .ok_or_else(|| usage_error("recover scenario needs a data object"))?;
    if !matches!(data_doc, Value::Object(_)) {
        return Err(usage_error("recover scenario data must be an object"));
    }
    let data = data_from_json(data_doc)?;
    let seed = scenario_seed(document, "recover")?;
    let interval = scenario_float(document, "interval", 0.8, "recover scenario interval")?;
    validate_target_accept(interval, "recover scenario interval")?;
    let mut settings = RecoverSettings {
        interval,
        ..RecoverSettings::default()
    };
    if let Some(sample_doc) = document.get("sample") {
        apply_sample_settings(
            sample_doc,
            &mut settings.sampler,
            &mut settings.chains,
            "recover",
        )?;
    }
    Ok(RecoverScenario {
        data,
        settings,
        seed,
    })
}

fn parse_sbc_scenario(document: &Value) -> Result<SbcScenario, Error> {
    reject_unknown_fields(
        document,
        "sbc scenario",
        &["sbc_scenario", "data", "seed", "replicates", "sample"],
    )?;
    if document.get("sbc_scenario").and_then(Value::as_str) != Some("v0-provisional") {
        return Err(usage_error(
            "sbc scenario needs sbc_scenario \"v0-provisional\"",
        ));
    }
    let data_doc = document
        .get("data")
        .ok_or_else(|| usage_error("sbc scenario needs a data object"))?;
    if !matches!(data_doc, Value::Object(_)) {
        return Err(usage_error("sbc scenario data must be an object"));
    }
    let data = data_from_json(data_doc)?;
    let seed = scenario_seed(document, "sbc")?;
    let mut settings = SbcSettings::default();
    let replicates = positive_usize(
        scenario_reportable_int(
            document,
            "replicates",
            settings.replicates as i64,
            "sbc scenario replicates",
            "sbc scenario replicates must be in 1..=9223372036854775807 because workflow reports replicates as JSON integers",
        )?,
        "sbc scenario replicates",
    )?;
    validate_sbc_replicates(replicates, "sbc scenario replicates")?;
    settings.replicates = replicates;
    if let Some(sample_doc) = document.get("sample") {
        apply_sample_settings(
            sample_doc,
            &mut settings.sampler,
            &mut settings.chains,
            "sbc",
        )?;
    }
    Ok(SbcScenario {
        data,
        settings,
        seed,
    })
}

fn run_recover(args: RecoverArgs) -> Result<(), Error> {
    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let scenario_doc = json::parse(&read_input(&args.scenario_path)?)?;
    let scenario = parse_recover_scenario(&scenario_doc)?;
    let report = recover_report(meta, scenario.data, &scenario.settings, scenario.seed)?;
    write_text(&args.out_path, &report)
}

fn run_sbc(args: SbcArgs) -> Result<(), Error> {
    let model_doc = json::parse(&read_input(&args.model_path)?)?;
    let meta = decode_model(&model_doc)?;
    let scenario_doc = json::parse(&read_input(&args.scenario_path)?)?;
    let mut scenario = parse_sbc_scenario(&scenario_doc)?;
    if let Some(replicates) = args.replicates_override {
        scenario.settings.replicates = replicates;
    }
    let report = sbc_report(meta, scenario.data, &scenario.settings, scenario.seed)?;
    write_text(&args.out_path, &report)
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&argv)? {
        Command::Sample(args) => run_sample(args),
        Command::Diagnose(args) => run_diagnose(args),
        Command::PriorPredictive(args) => run_prior_predictive(args),
        Command::PosteriorPredictive(args) => run_posterior_predictive(args),
        Command::PosteriorCheck(args) => run_posterior_check(args),
        Command::Simulate(args) => run_simulate(args),
        Command::RecoverCheck(args) => run_recover_check(args),
        Command::Recover(args) => run_recover(args),
        Command::Sbc(args) => run_sbc(args),
    }
}

fn main() {
    if let Err(error) = run() {
        let payload = Value::Object(vec![
            (
                "error_format".to_string(),
                Value::Str("v0-provisional".to_string()),
            ),
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
