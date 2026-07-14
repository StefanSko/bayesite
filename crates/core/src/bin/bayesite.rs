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
//!   bayesite capabilities
//!
//! `sample` writes the v0-provisional NDJSON protocol (see `protocol.rs`).
//! `diagnose`, `recover-check`, `recover`, and `sbc` write one v0-provisional
//! JSON object; `simulate` writes a plain data document; `prior-predictive`
//! writes v0-provisional NDJSON. `capabilities` writes one v0-provisional
//! JSON document advertising the command set, IR version, and input schema
//! versions (see `docs/capabilities-v0.md`). `-` means stdout/stdin.
//! Errors are a single JSON object on stderr with a nonzero exit code; messages
//! state what to change.
//!
//! Parallelism lives here, not in the library: one thread per chain.

use std::io::Read;
use std::io::Write;

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::fingerprint::model_data_fingerprint;
use bayesite_core::generation::{
    generated_datasets_ndjson_lines, sha256_bytes, GenerationRequest, GenerationSource,
};
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
    Generate(GenerateArgs),
    PosteriorPredictive(PosteriorPredictiveArgs),
    PosteriorCheck(PosteriorCheckArgs),
    Simulate(SimulateArgs),
    RecoverCheck(RecoverCheckArgs),
    Recover(RecoverArgs),
    Sbc(SbcArgs),
    Capabilities,
}

type ParseCommandFn = fn(&[String]) -> Result<Command, Error>;

/// The CLI dispatch table. `capabilities` derives its `commands` list from
/// this table, so adding a command here advertises it automatically.
const COMMANDS: &[(&str, ParseCommandFn)] = &[
    ("sample", |argv| {
        parse_sample_args(argv).map(Command::Sample)
    }),
    ("diagnose", |argv| {
        parse_diagnose_args(argv).map(Command::Diagnose)
    }),
    ("prior-predictive", |argv| {
        parse_prior_predictive_args(argv).map(Command::PriorPredictive)
    }),
    ("generate", |argv| {
        parse_generate_args(argv).map(Command::Generate)
    }),
    ("posterior-predictive", |argv| {
        parse_posterior_predictive_args(argv).map(Command::PosteriorPredictive)
    }),
    ("posterior-check", |argv| {
        parse_posterior_check_args(argv).map(Command::PosteriorCheck)
    }),
    ("simulate", |argv| {
        parse_simulate_args(argv).map(Command::Simulate)
    }),
    ("recover-check", |argv| {
        parse_recover_check_args(argv).map(Command::RecoverCheck)
    }),
    ("recover", |argv| {
        parse_recover_args(argv).map(Command::Recover)
    }),
    ("sbc", |argv| parse_sbc_args(argv).map(Command::Sbc)),
    ("capabilities", parse_capabilities_args),
];

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

enum GenerateSourceArgs {
    Fixed {
        parameters_path: String,
    },
    ModelPrior,
    Posterior {
        fit_path: String,
        fit_data_path: String,
    },
}

struct GenerateArgs {
    model_path: String,
    design_path: String,
    source: GenerateSourceArgs,
    out_path: String,
    count: usize,
    seed: u64,
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
     usage: bayesite generate --model <ir.json|-> --design <data.json|-> \
     --source <fixed|model-prior|posterior> [--parameters <params.json|->] \
     [--fit <fit.jsonl|-> --fit-data <data.json|->] [--count N] [--seed N] \
     [--out <generated.jsonl|->]\n\
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
     [--replicates N] [--out <report.json|->]\n\
     usage: bayesite capabilities"
}

fn parse_args(argv: &[String]) -> Result<Command, Error> {
    let Some(command) = argv.first() else {
        return Err(usage_error(format!("missing command; {}", usage())));
    };
    match COMMANDS.iter().find(|(name, _)| *name == command.as_str()) {
        Some((_, parse)) => parse(&argv[1..]),
        None => Err(usage_error(format!(
            "unknown command \"{command}\"; {}",
            usage()
        ))),
    }
}

fn parse_capabilities_args(argv: &[String]) -> Result<Command, Error> {
    if let Some(first) = argv.first() {
        return Err(usage_error(format!(
            "capabilities takes no arguments; remove {first}"
        )));
    }
    Ok(Command::Capabilities)
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

fn parse_generate_args(argv: &[String]) -> Result<GenerateArgs, Error> {
    reject_duplicate_flags(
        "generate",
        argv,
        &[
            "--model",
            "--design",
            "--source",
            "--parameters",
            "--fit",
            "--fit-data",
            "--out",
            "--count",
            "--seed",
        ],
    )?;
    let mut model_path: Option<String> = None;
    let mut design_path: Option<String> = None;
    let mut source_kind: Option<String> = None;
    let mut parameters_path: Option<String> = None;
    let mut fit_path: Option<String> = None;
    let mut fit_data_path: Option<String> = None;
    let mut out_path = "-".to_string();
    let mut count = 100usize;
    let mut seed = 0u64;
    let mut iter = argv.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--model" => model_path = Some(value_for_flag(&mut iter, "--model")?.clone()),
            "--design" => design_path = Some(value_for_flag(&mut iter, "--design")?.clone()),
            "--source" => source_kind = Some(value_for_flag(&mut iter, "--source")?.clone()),
            "--parameters" => {
                parameters_path = Some(value_for_flag(&mut iter, "--parameters")?.clone())
            }
            "--fit" => fit_path = Some(value_for_flag(&mut iter, "--fit")?.clone()),
            "--fit-data" => fit_data_path = Some(value_for_flag(&mut iter, "--fit-data")?.clone()),
            "--out" => out_path = value_for_flag(&mut iter, "--out")?.clone(),
            "--count" => {
                count = parse_reportable_draw_count(
                    value_for_flag(&mut iter, "--count")?,
                    "--count",
                    "generated-dataset artifacts",
                )?
            }
            "--seed" => seed = parse_artifact_seed(value_for_flag(&mut iter, "--seed")?)?,
            other => {
                return Err(usage_error(format!(
                    "unknown flag {other}; see `bayesite generate` usage"
                )))
            }
        }
    }
    let model_path =
        model_path.ok_or_else(|| usage_error("--model is required (a path or - for stdin)"))?;
    let design_path =
        design_path.ok_or_else(|| usage_error("--design is required (a path or - for stdin)"))?;
    let source_kind = source_kind
        .ok_or_else(|| usage_error("--source is required (fixed, model-prior, or posterior)"))?;
    if count == 0 || count > 1000 {
        return Err(usage_error("--count must be in 1..=1000"));
    }
    if seed > 9_007_199_254_740_991 {
        return Err(usage_error(
            "--seed must be in 0..=9007199254740991 for generation interoperability",
        ));
    }
    let source = match source_kind.as_str() {
        "fixed" => {
            let parameters_path = parameters_path
                .ok_or_else(|| usage_error("--parameters is required when --source fixed"))?;
            if fit_path.is_some() || fit_data_path.is_some() {
                return Err(usage_error(
                    "--fit and --fit-data are only valid when --source posterior",
                ));
            }
            GenerateSourceArgs::Fixed { parameters_path }
        }
        "model-prior" => {
            if parameters_path.is_some() || fit_path.is_some() || fit_data_path.is_some() {
                return Err(usage_error(
                    "model-prior generation does not accept --parameters, --fit, or --fit-data",
                ));
            }
            GenerateSourceArgs::ModelPrior
        }
        "posterior" => {
            if parameters_path.is_some() {
                return Err(usage_error(
                    "--parameters is only valid when --source fixed",
                ));
            }
            let fit_path =
                fit_path.ok_or_else(|| usage_error("--fit is required when --source posterior"))?;
            let fit_data_path = fit_data_path
                .ok_or_else(|| usage_error("--fit-data is required when --source posterior"))?;
            GenerateSourceArgs::Posterior {
                fit_path,
                fit_data_path,
            }
        }
        other => {
            return Err(usage_error(format!(
                "--source must be fixed, model-prior, or posterior; got {other:?}"
            )))
        }
    };
    let mut inputs = vec![
        ("--model", model_path.as_str()),
        ("--design", design_path.as_str()),
    ];
    match &source {
        GenerateSourceArgs::Fixed { parameters_path } => {
            inputs.push(("--parameters", parameters_path.as_str()));
        }
        GenerateSourceArgs::ModelPrior => {}
        GenerateSourceArgs::Posterior {
            fit_path,
            fit_data_path,
        } => {
            inputs.push(("--fit", fit_path.as_str()));
            inputs.push(("--fit-data", fit_data_path.as_str()));
        }
    }
    validate_single_stdin_input("generate", &inputs)?;
    Ok(GenerateArgs {
        model_path,
        design_path,
        source,
        out_path,
        count,
        seed,
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

fn run_generate(args: GenerateArgs) -> Result<(), Error> {
    let model_text = read_input(&args.model_path)?;
    let design_text = read_input(&args.design_path)?;
    let model_doc = json::parse(&model_text)?;
    let meta = decode_model(&model_doc)?;
    let design = json::parse(&design_text)?;
    let generation_model_hash = sha256_bytes(model_text.as_bytes());
    let design_hash = sha256_bytes(design_text.as_bytes());
    let source = match args.source {
        GenerateSourceArgs::Fixed { parameters_path } => {
            let parameters_text = read_input(&parameters_path)?;
            let parameters = json::parse(&parameters_text)?;
            GenerationSource::Fixed {
                parameters,
                parameters_hash: sha256_bytes(parameters_text.as_bytes()),
            }
        }
        GenerateSourceArgs::ModelPrior => GenerationSource::ModelPrior {
            model_hash: generation_model_hash.clone(),
            authored_provenance: None,
        },
        GenerateSourceArgs::Posterior {
            fit_path,
            fit_data_path,
        } => {
            let fit_ndjson = read_input(&fit_path)?;
            let fit_data_text = read_input(&fit_data_path)?;
            let fit_data = json::parse(&fit_data_text)?;
            GenerationSource::Posterior {
                fit_hash: sha256_bytes(fit_ndjson.as_bytes()),
                fit_model_hash: generation_model_hash.clone(),
                fit_data_hash: sha256_bytes(fit_data_text.as_bytes()),
                expected_model_data_fingerprint: Some(model_data_fingerprint(
                    &model_text,
                    &fit_data_text,
                )),
                fit_ndjson,
                fit_data,
            }
        }
    };
    let lines = generated_datasets_ndjson_lines(GenerationRequest {
        meta,
        design,
        source,
        count: args.count,
        seed: args.seed,
        generation_model_hash,
        design_hash,
    })?;
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
    thin: Option<&mut usize>,
    context: &str,
) -> Result<(), Error> {
    if !matches!(sample_doc, Value::Object(_)) {
        return Err(usage_error(format!(
            "{context} scenario sample must be an object"
        )));
    }
    let allowed = if thin.is_some() {
        &[
            "chains",
            "warmup",
            "draws",
            "thin",
            "max_treedepth",
            "target_accept",
        ][..]
    } else {
        &[
            "chains",
            "warmup",
            "draws",
            "max_treedepth",
            "target_accept",
        ][..]
    };
    reject_unknown_fields(sample_doc, &format!("{context} scenario sample"), allowed)?;
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
    if let Some(thin) = thin {
        let thin_label = format!("{context} scenario sample.thin");
        *thin = positive_usize(
            scenario_reportable_int(
                sample_doc,
                "thin",
                *thin as i64,
                &thin_label,
                &format!(
                    "{thin_label} must be in 1..=9223372036854775807 because workflow reports sample.thin as a JSON integer"
                ),
            )?,
            &thin_label,
        )?;
        if !sampler.num_draws.is_multiple_of(*thin) {
            return Err(usage_error(format!(
                "{thin_label} must divide sample.draws exactly; pick a thin that divides sample.draws"
            )));
        }
    }
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
            None,
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
            Some(&mut settings.thin),
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

/// The `capabilities` document is a wire contract (`capabilities_format`):
/// field additions are compatible; renames or removals bump the version.
fn capabilities_document() -> Value {
    let commands = COMMANDS
        .iter()
        .map(|(name, _)| Value::Str((*name).to_string()))
        .collect();
    Value::Object(vec![
        (
            "capabilities_format".to_string(),
            Value::Str("v0-provisional".to_string()),
        ),
        (
            "version".to_string(),
            Value::Str(env!("CARGO_PKG_VERSION").to_string()),
        ),
        ("commands".to_string(), Value::Array(commands)),
        (
            "ir".to_string(),
            Value::Object(vec![("bayeswire_ir".to_string(), Value::Int(1))]),
        ),
        (
            "schemas".to_string(),
            Value::Object(vec![
                (
                    "recover_scenario".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
                (
                    "sbc_scenario".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
                (
                    "recover_check_targets".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
                (
                    "error_format".to_string(),
                    Value::Str("v0-provisional".to_string()),
                ),
            ]),
        ),
    ])
}

fn run_capabilities() -> Result<(), Error> {
    let text = json::write(&capabilities_document())?;
    write_text("-", &text)
}

fn run() -> Result<(), Error> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match parse_args(&argv)? {
        Command::Sample(args) => run_sample(args),
        Command::Diagnose(args) => run_diagnose(args),
        Command::PriorPredictive(args) => run_prior_predictive(args),
        Command::Generate(args) => run_generate(args),
        Command::PosteriorPredictive(args) => run_posterior_predictive(args),
        Command::PosteriorCheck(args) => run_posterior_check(args),
        Command::Simulate(args) => run_simulate(args),
        Command::RecoverCheck(args) => run_recover_check(args),
        Command::Recover(args) => run_recover(args),
        Command::Sbc(args) => run_sbc(args),
        Command::Capabilities => run_capabilities(),
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
