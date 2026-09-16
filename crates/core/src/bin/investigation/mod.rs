mod store;
mod workspace;

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::fingerprint::model_data_fingerprint;
use bayesite_core::inspect::inspect_json;
use bayesite_core::investigation::identity::{artifact_digest, snapshot_digest};
use bayesite_core::investigation::manifest::{
    ArtifactKind, ArtifactRef, EngineIdentity, EvidenceSelection, EvidenceStatus, Execution,
    Manifest, Operation, Outcome, Recipe, Source,
};
use bayesite_core::investigation::verify_bundle;
use bayesite_core::ir::decode_model;
use bayesite_core::json::{self, Value};
use bayesite_core::model::{data_from_json, Posterior};
use bayesite_core::predictive::posterior_check_report_with_model_data_fingerprint;
use bayesite_core::protocol;
use bayesite_core::sampler::{sample, ChainDraws, Settings};

use workspace::{Attempt, ConfiguredRecipe, Metadata, Selection, Workspace};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn string(value: impl Into<String>) -> Value {
    Value::Str(value.into())
}

fn usage() -> &'static str {
    "usage: bayesite investigation init --metadata <metadata.json> --model <model.json> --data <data.json> --out <workspace>\n\
     usage: bayesite investigation inspect <workspace>\n\
     usage: bayesite investigation run <workspace> --recipe <id>\n\
     usage: bayesite investigation snapshot <workspace> --out <bundle>\n\
     usage: bayesite investigation verify <bundle>\n\
     usage: bayesite investigation fork <bundle> --at <decision-id> --out <workspace>\n\
     usage: bayesite investigation replay <bundle> --recipe <id> --out <replay-dir>\n\
     usage: bayesite investigation export <bundle> --viewer --public-data-confirmed --out <directory>"
}

fn emit(value: &Value) -> Result<(), Error> {
    println!("{}", json::write(value)?);
    Ok(())
}

fn read_bytes(path: &Path, context: &str) -> Result<Vec<u8>, Error> {
    fs::read(path).map_err(|error| invalid(format!("cannot read {context} {:?}: {error}", path)))
}

fn read_json(path: &Path, context: &str) -> Result<Value, Error> {
    let bytes = read_bytes(path, context)?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| invalid(format!("{context} {:?} must be UTF-8 JSON", path)))?;
    json::parse(text)
}

fn one_positional<'a>(argv: &'a [String], command: &str) -> Result<&'a str, Error> {
    if argv.len() != 1 || argv[0].starts_with("--") {
        return Err(invalid(format!(
            "investigation {command} needs exactly one path; {}",
            usage()
        )));
    }
    Ok(&argv[0])
}

fn parse_flags(
    argv: &[String],
    command: &str,
    positional_count: usize,
    flags: &[&str],
) -> Result<(Vec<String>, HashMap<String, String>), Error> {
    let mut positional = Vec::new();
    let mut parsed = HashMap::new();
    let mut index = 0usize;
    while index < argv.len() {
        let value = &argv[index];
        if value.starts_with("--") {
            if !flags.contains(&value.as_str()) {
                return Err(invalid(format!(
                    "unknown investigation {command} flag {value}; {}",
                    usage()
                )));
            }
            if parsed.contains_key(value) {
                return Err(invalid(format!(
                    "investigation {command} has duplicate flag {value}; pass it once"
                )));
            }
            let next = argv.get(index + 1).ok_or_else(|| {
                invalid(format!(
                    "investigation {command} flag {value} needs a value"
                ))
            })?;
            if next.starts_with("--") {
                return Err(invalid(format!(
                    "investigation {command} flag {value} needs a value before {next}"
                )));
            }
            parsed.insert(value.clone(), next.clone());
            index += 2;
        } else {
            positional.push(value.clone());
            index += 1;
        }
    }
    if positional.len() != positional_count {
        return Err(invalid(format!(
            "investigation {command} needs {positional_count} positional path(s); {}",
            usage()
        )));
    }
    for flag in flags {
        if !parsed.contains_key(*flag) {
            return Err(invalid(format!(
                "investigation {command} requires {flag}; {}",
                usage()
            )));
        }
    }
    Ok((positional, parsed))
}

fn target_name() -> String {
    option_env!("TARGET")
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS))
}

fn profile_name() -> String {
    if cfg!(debug_assertions) {
        "debug".into()
    } else {
        "release".into()
    }
}

fn current_engine_bytes(capabilities: &str) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let executable = std::env::current_exe()
        .map_err(|error| invalid(format!("cannot identify running executable: {error}")))?;
    let binary = read_bytes(&executable, "running executable")?;
    Ok((binary, capabilities.as_bytes().to_vec()))
}

fn create_engine(root: &Path, capabilities: &str) -> Result<EngineIdentity, Error> {
    let (binary, capabilities) = current_engine_bytes(capabilities)?;
    Ok(EngineIdentity {
        binary: store::insert(
            root,
            &binary,
            ArtifactKind::EngineBinary,
            "native-executable",
        )?,
        capabilities: store::insert(
            root,
            &capabilities,
            ArtifactKind::EngineCapabilities,
            "capabilities-v0-provisional",
        )?,
        target: target_name(),
        profile: profile_name(),
    })
}

fn check_engine(
    engine: &EngineIdentity,
    object_root: &Path,
    capabilities: &str,
) -> Result<(), Error> {
    let (binary, current_capabilities) = current_engine_bytes(capabilities)?;
    let binary_digest = artifact_digest(&binary);
    if binary_digest != engine.binary.sha256
        || binary.len() != engine.binary.bytes
        || target_name() != engine.target
        || profile_name() != engine.profile
    {
        return Err(invalid(format!(
            "running engine does not match recipe pin (need binary {}, target {}, profile {}); obtain and explicitly run the pinned engine",
            engine.binary.sha256.prefixed(),
            engine.target,
            engine.profile
        )));
    }
    let capabilities_digest = artifact_digest(&current_capabilities);
    if capabilities_digest != engine.capabilities.sha256 {
        return Err(invalid(format!(
            "running engine capabilities do not match pinned {}; run the exact pinned binary",
            engine.capabilities.sha256.prefixed()
        )));
    }
    store::read(object_root, &engine.binary)?;
    store::read(object_root, &engine.capabilities)?;
    Ok(())
}

fn input_paths(root: &Path) -> (PathBuf, PathBuf) {
    (
        root.join("inputs/model.json"),
        root.join("inputs/data.json"),
    )
}

fn current_inputs(root: &Path) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let (model, data) = input_paths(root);
    Ok((
        read_bytes(&model, "workspace model")?,
        read_bytes(&data, "workspace data")?,
    ))
}

fn insert_inputs(
    root: &Path,
    model: &[u8],
    data: &[u8],
) -> Result<(ArtifactRef, ArtifactRef), Error> {
    Ok((
        store::insert(root, model, ArtifactKind::ModelIr, "bayeswire-ir-v1")?,
        store::insert(root, data, ArtifactKind::Data, "bayesite-data-json-v1")?,
    ))
}

fn validate_model_data(model: &[u8], data: &[u8]) -> Result<(), Error> {
    let model_text = std::str::from_utf8(model).map_err(|_| invalid("model must be UTF-8 JSON"))?;
    let data_text = std::str::from_utf8(data).map_err(|_| invalid("data must be UTF-8 JSON"))?;
    let meta = decode_model(&json::parse(model_text)?)?;
    let data = data_from_json(&json::parse(data_text)?)?;
    Posterior::new(meta, data)?;
    Ok(())
}

fn create_fresh_directory(path: &Path, context: &str) -> Result<(), Error> {
    fs::create_dir(path).map_err(|error| {
        invalid(format!(
            "cannot create fresh {context} {:?}: {error}; choose a path that does not exist",
            path
        ))
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Error> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            invalid(format!(
                "cannot create {:?} without replacement: {error}",
                path
            ))
        })?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| invalid(format!("cannot write {:?}: {error}", path)))
}

fn init(argv: &[String], capabilities: &str) -> Result<(), Error> {
    let (_, flags) = parse_flags(
        argv,
        "init",
        0,
        &["--metadata", "--model", "--data", "--out"],
    )?;
    let metadata_path = Path::new(&flags["--metadata"]);
    let model_path = Path::new(&flags["--model"]);
    let data_path = Path::new(&flags["--data"]);
    let out = Path::new(&flags["--out"]);
    let model = read_bytes(model_path, "model")?;
    let data = read_bytes(data_path, "data")?;
    validate_model_data(&model, &data)?;
    let metadata_value = read_json(metadata_path, "investigation metadata")?;

    create_fresh_directory(out, "workspace")?;
    let result = (|| {
        fs::create_dir_all(out.join("inputs"))
            .map_err(|error| invalid(format!("cannot create workspace inputs: {error}")))?;
        let (model_ref, data_ref) = insert_inputs(out, &model, &data)?;
        let metadata = Metadata::parse(&metadata_value, &model_ref, &data_ref)?;
        let engine = create_engine(out, capabilities)?;
        let workspace = Workspace {
            question: metadata.question,
            estimand_description: metadata.estimand_description,
            estimand_parameter: metadata.estimand_parameter,
            source: None,
            engine,
            decisions: metadata.decisions,
            recipes: metadata.recipes,
            attempts: vec![],
            selections: vec![],
            interpretation: metadata.interpretation,
            unresolved_questions: metadata.unresolved_questions,
        };
        Workspace::parse(&workspace.to_value())?;
        let (model_out, data_out) = input_paths(out);
        fs::write(&model_out, &model)
            .map_err(|error| invalid(format!("cannot write model working copy: {error}")))?;
        fs::write(&data_out, &data)
            .map_err(|error| invalid(format!("cannot write data working copy: {error}")))?;
        workspace::save(out, &workspace)?;
        emit(&Value::Object(vec![
            ("investigation_command".into(), string("init")),
            ("workspace".into(), string(out.display().to_string())),
            ("model_sha256".into(), string(model_ref.sha256.prefixed())),
            ("data_sha256".into(), string(data_ref.sha256.prefixed())),
            (
                "engine_sha256".into(),
                string(workspace.engine.binary.sha256.prefixed()),
            ),
        ]))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(out);
    }
    result
}

fn find_attempt<'a>(workspace: &'a Workspace, execution: &str) -> Option<&'a Attempt> {
    workspace
        .attempts
        .iter()
        .find(|attempt| attempt.execution.id == execution)
}

fn configured<'a>(workspace: &'a Workspace, id: &str) -> Option<&'a ConfiguredRecipe> {
    workspace.recipes.iter().find(|recipe| recipe.id == id)
}

fn current_sample<'a>(
    workspace: &'a Workspace,
    model: &ArtifactRef,
    data: &ArtifactRef,
) -> Result<Option<&'a ArtifactRef>, Error> {
    let mut selected = Vec::new();
    for selection in &workspace.selections {
        let Some(attempt) = find_attempt(workspace, &selection.execution) else {
            continue;
        };
        if attempt.execution.outcome != Outcome::Completed
            || attempt.recipe.operation != Operation::Sample
            || attempt.recipe.model != *model
            || attempt.recipe.data != *data
            || attempt.recipe.engine != workspace.engine
        {
            continue;
        }
        let Some(config) = configured(workspace, &attempt.recipe.id) else {
            continue;
        };
        let Ok(current) = Recipe::new(
            config.id.clone(),
            Operation::Sample,
            model.clone(),
            data.clone(),
            None,
            workspace.engine.clone(),
            config.settings.clone(),
        ) else {
            continue;
        };
        if current.sha256 == attempt.recipe.sha256 {
            if let Some(output) = attempt.execution.output.as_ref() {
                selected.push((selection, output));
            }
        }
    }
    if selected.len() > 1 {
        return Err(invalid(format!(
            "{} current sample selections conflict; edit investigation.json selections to retain exactly one before inspection or downstream execution",
            selected.len()
        )));
    }
    Ok(selected.first().map(|(_, output)| *output))
}

fn resolve_recipe(
    workspace: &Workspace,
    config: &ConfiguredRecipe,
    model: &ArtifactRef,
    data: &ArtifactRef,
) -> Result<Recipe, Error> {
    let fit = match config.operation {
        Operation::Diagnose | Operation::PosteriorCheck => Some(
            current_sample(workspace, model, data)?
                .ok_or_else(|| {
                    invalid(format!(
                        "recipe {:?} needs current sample evidence for the exact current model/data; run a sample recipe first",
                        config.id
                    ))
                })?
                .clone(),
        ),
        Operation::Inspect | Operation::Sample => None,
    };
    Recipe::new(
        config.id.clone(),
        config.operation,
        model.clone(),
        data.clone(),
        fit,
        workspace.engine.clone(),
        config.settings.clone(),
    )
}

fn is_selection_current(
    workspace: &Workspace,
    selection: &Selection,
    model: &ArtifactRef,
    data: &ArtifactRef,
) -> bool {
    let Some(attempt) = find_attempt(workspace, &selection.execution) else {
        return false;
    };
    if attempt.execution.outcome != Outcome::Completed {
        return false;
    }
    let Some(config) = configured(workspace, &attempt.recipe.id) else {
        return false;
    };
    resolve_recipe(workspace, config, model, data)
        .is_ok_and(|current| current.sha256 == attempt.recipe.sha256)
}

fn inherited_evidence(root: &Path, workspace: &Workspace) -> Result<Vec<Value>, Error> {
    let Some(source) = &workspace.source else {
        return Ok(vec![]);
    };
    let bytes = store::read(root, &source.manifest)?;
    let parent = Manifest::parse_bytes(&bytes)?;
    Ok(parent
        .evidence
        .iter()
        .map(|evidence| {
            Value::Object(vec![
                ("name".into(), string(&evidence.name)),
                ("execution".into(), string(&evidence.execution)),
                ("status".into(), string("historical")),
                ("origin".into(), string("source_snapshot")),
            ])
        })
        .collect())
}

fn inspect_workspace(path: &Path) -> Result<(), Error> {
    let workspace = workspace::load(path)?;
    let (model_bytes, data_bytes) = current_inputs(path)?;
    validate_model_data(&model_bytes, &data_bytes)?;
    let (model, data) = insert_inputs(path, &model_bytes, &data_bytes)?;
    let fit = current_sample(&workspace, &model, &data)?.cloned();
    let recipes = workspace
        .recipes
        .iter()
        .map(
            |config| match resolve_recipe(&workspace, config, &model, &data) {
                Ok(recipe) => Value::Object(vec![
                    ("id".into(), string(&config.id)),
                    ("operation".into(), string(config.operation.as_str())),
                    ("state".into(), string("ready")),
                    ("recipe_sha256".into(), string(recipe.sha256.prefixed())),
                ]),
                Err(error) => Value::Object(vec![
                    ("id".into(), string(&config.id)),
                    ("operation".into(), string(config.operation.as_str())),
                    ("state".into(), string("blocked")),
                    ("message".into(), string(error.message)),
                ]),
            },
        )
        .collect::<Vec<_>>();
    let mut evidence = inherited_evidence(path, &workspace)?;
    evidence.extend(workspace.selections.iter().map(|selection| {
        Value::Object(vec![
            ("name".into(), string(&selection.name)),
            ("execution".into(), string(&selection.execution)),
            (
                "status".into(),
                string(
                    if is_selection_current(&workspace, selection, &model, &data) {
                        "current"
                    } else {
                        "historical"
                    },
                ),
            ),
            ("origin".into(), string("workspace")),
        ])
    }));
    emit(&Value::Object(vec![
        (
            "workspace_inspection_format".into(),
            string("v0-provisional"),
        ),
        ("question".into(), string(&workspace.question)),
        (
            "estimand".into(),
            Value::Object(vec![
                (
                    "description".into(),
                    string(&workspace.estimand_description),
                ),
                ("parameter".into(), string(&workspace.estimand_parameter)),
            ]),
        ),
        ("model_sha256".into(), string(model.sha256.prefixed())),
        ("data_sha256".into(), string(data.sha256.prefixed())),
        (
            "current_fit_sha256".into(),
            fit.as_ref()
                .map(|fit| string(fit.sha256.prefixed()))
                .unwrap_or(Value::Null),
        ),
        ("recipes".into(), Value::Array(recipes)),
        ("evidence".into(), Value::Array(evidence)),
        (
            "source".into(),
            workspace
                .source
                .as_ref()
                .map(Source::to_value)
                .unwrap_or(Value::Null),
        ),
    ]))
}

fn setting_i64(settings: &Value, name: &str) -> Result<i64, Error> {
    settings
        .get(name)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid(format!("resolved recipe setting {name} must be an integer")))
}

fn execute_recipe(
    recipe: &Recipe,
    model_bytes: &[u8],
    data_bytes: &[u8],
    fit_bytes: Option<&[u8]>,
) -> Result<Vec<u8>, Error> {
    let model_text =
        std::str::from_utf8(model_bytes).map_err(|_| invalid("recipe model must be UTF-8 JSON"))?;
    let data_text =
        std::str::from_utf8(data_bytes).map_err(|_| invalid("recipe data must be UTF-8 JSON"))?;
    let meta = decode_model(&json::parse(model_text)?)?;
    let data = data_from_json(&json::parse(data_text)?)?;
    let text = match recipe.operation {
        Operation::Inspect => inspect_json(meta, data)?,
        Operation::Sample => {
            let chains = setting_i64(&recipe.settings, "chains")? as u64;
            let settings = Settings {
                num_warmup: setting_i64(&recipe.settings, "warmup")? as usize,
                num_draws: setting_i64(&recipe.settings, "draws")? as usize,
                max_treedepth: setting_i64(&recipe.settings, "max_treedepth")? as usize,
                target_accept: recipe
                    .settings
                    .get("target_accept")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| invalid("target_accept must be a number"))?,
                initial_step_size: recipe
                    .settings
                    .get("initial_step_size")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| invalid("initial_step_size must be a number"))?,
            };
            if chains > 8 || settings.num_warmup > 10_000 || settings.num_draws > 10_000 {
                return Err(invalid(
                    "investigation sampling is bounded to 8 chains and 10000 warmup/draws per chain",
                ));
            }
            let seed = setting_i64(&recipe.settings, "seed")? as u64;
            let posterior = Posterior::new(meta, data)?;
            let results: Vec<Result<ChainDraws, Error>> = std::thread::scope(|scope| {
                let handles = (0..chains)
                    .map(|chain_id| {
                        let posterior = &posterior;
                        let settings = &settings;
                        scope.spawn(move || sample(posterior, settings, seed, chain_id))
                    })
                    .collect::<Vec<_>>();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("chain thread panicked"))
                    .collect()
            });
            let mut chain_draws = Vec::with_capacity(results.len());
            for (chain_id, result) in results.into_iter().enumerate() {
                chain_draws.push((chain_id as u64, result?));
            }
            let fingerprint = model_data_fingerprint(model_text, data_text);
            protocol::ndjson_lines_with_model_data_fingerprint(
                &posterior,
                &settings,
                seed,
                &chain_draws,
                Some(&fingerprint),
            )?
            .join("\n")
        }
        Operation::Diagnose => {
            let fit = std::str::from_utf8(
                fit_bytes.ok_or_else(|| invalid("diagnose recipe needs fit bytes"))?,
            )
            .map_err(|_| invalid("fit must be UTF-8 NDJSON"))?;
            protocol::diagnose_ndjson(fit)?
        }
        Operation::PosteriorCheck => {
            let fit = std::str::from_utf8(
                fit_bytes.ok_or_else(|| invalid("posterior-check recipe needs fit bytes"))?,
            )
            .map_err(|_| invalid("fit must be UTF-8 NDJSON"))?;
            let seed = setting_i64(&recipe.settings, "seed")? as u64;
            let fingerprint = model_data_fingerprint(model_text, data_text);
            posterior_check_report_with_model_data_fingerprint(
                meta,
                data,
                fit,
                seed,
                Some(&fingerprint),
            )?
        }
    };
    let mut bytes = text.into_bytes();
    bytes.push(b'\n');
    Ok(bytes)
}

fn output_description(operation: Operation) -> (ArtifactKind, &'static str) {
    match operation {
        Operation::Inspect => (ArtifactKind::Inspection, "inspection-v0-provisional"),
        Operation::Sample => (ArtifactKind::PosteriorDraws, "draws-v0-provisional-ndjson"),
        Operation::Diagnose => (ArtifactKind::Diagnostics, "diagnostics-v0-provisional"),
        Operation::PosteriorCheck => (
            ArtifactKind::PosteriorCheck,
            "posterior-check-v0-provisional",
        ),
    }
}

fn run_recipe(argv: &[String], capabilities: &str) -> Result<(), Error> {
    let (positional, flags) = parse_flags(argv, "run", 1, &["--recipe"])?;
    let root = Path::new(&positional[0]);
    let mut workspace = workspace::load(root)?;
    check_engine(&workspace.engine, root, capabilities)?;
    let (model_bytes, data_bytes) = current_inputs(root)?;
    validate_model_data(&model_bytes, &data_bytes)?;
    let (model, data) = insert_inputs(root, &model_bytes, &data_bytes)?;
    let recipe_id = &flags["--recipe"];
    let config = configured(&workspace, recipe_id)
        .ok_or_else(|| {
            invalid(format!(
                "workspace has no recipe {recipe_id:?}; add it to investigation.json"
            ))
        })?
        .clone();
    let recipe = resolve_recipe(&workspace, &config, &model, &data)?;
    if workspace
        .attempts
        .iter()
        .any(|attempt| attempt.recipe.id == recipe.id && attempt.recipe.sha256 != recipe.sha256)
    {
        return Err(invalid(format!(
            "recipe id {:?} was already used for different execution inputs; rename the edited recipe (for example, sample-alternative) to preserve history",
            recipe.id
        )));
    }
    let execution_id = format!("exec-{}-{}", workspace.attempts.len() + 1, recipe.id);
    let attempt = Attempt {
        recipe: recipe.clone(),
        execution: Execution {
            id: execution_id.clone(),
            recipe: recipe.id.clone(),
            recipe_sha256: recipe.sha256.clone(),
            outcome: Outcome::Incomplete,
            output: None,
            error: None,
        },
    };
    workspace.attempts.push(attempt);
    workspace::save(root, &workspace)?;

    let fit_bytes = recipe
        .fit
        .as_ref()
        .map(|fit| store::read(root, fit))
        .transpose()?;
    let executed = execute_recipe(&recipe, &model_bytes, &data_bytes, fit_bytes.as_deref());
    let attempt = workspace
        .attempts
        .last_mut()
        .expect("incomplete attempt was appended");
    let output = match executed {
        Ok(bytes) => {
            let (kind, format) = output_description(recipe.operation);
            let reference = store::insert(root, &bytes, kind, format)?;
            attempt.execution.outcome = Outcome::Completed;
            attempt.execution.output = Some(reference.clone());
            reference
        }
        Err(error) => {
            attempt.execution.outcome = Outcome::Failed;
            attempt.execution.error = Some(error.message.clone());
            workspace::save(root, &workspace)?;
            return Err(error);
        }
    };
    let selected = !workspace.selections.iter().any(|selection| {
        find_attempt(&workspace, &selection.execution).is_some_and(|attempt| {
            attempt.recipe.operation == recipe.operation
                && is_selection_current(&workspace, selection, &model, &data)
        })
    });
    if selected {
        workspace.selections.push(Selection {
            name: recipe.id.clone(),
            execution: execution_id.clone(),
        });
    }
    workspace::save(root, &workspace)?;
    emit(&Value::Object(vec![
        ("investigation_command".into(), string("run")),
        ("recipe".into(), string(&recipe.id)),
        ("recipe_sha256".into(), string(recipe.sha256.prefixed())),
        ("execution".into(), string(execution_id)),
        ("outcome".into(), string("completed")),
        ("output_sha256".into(), string(output.sha256.prefixed())),
        ("selected".into(), Value::Bool(selected)),
        (
            "selection_note".into(),
            string(if selected {
                "first current successful result for this operation was selected"
            } else {
                "result retained but not selected; edit investigation.json selections explicitly"
            }),
        ),
    ]))
}

fn collect_manifest(
    root: &Path,
    workspace: &Workspace,
    model: ArtifactRef,
    data: ArtifactRef,
) -> Result<Manifest, Error> {
    let mut recipes: Vec<Recipe> = Vec::new();
    let mut executions = Vec::new();
    for attempt in &workspace.attempts {
        if let Some(existing) = recipes.iter().find(|recipe| recipe.id == attempt.recipe.id) {
            if existing.sha256 != attempt.recipe.sha256 {
                return Err(invalid(format!(
                    "recipe id {:?} refers to more than one identity; rename the edited recipe to preserve both",
                    attempt.recipe.id
                )));
            }
        } else {
            recipes.push(attempt.recipe.clone());
        }
        executions.push(attempt.execution.clone());
    }
    for config in &workspace.recipes {
        let Ok(recipe) = resolve_recipe(workspace, config, &model, &data) else {
            continue;
        };
        if let Some(existing) = recipes.iter().find(|existing| existing.id == recipe.id) {
            if existing.sha256 != recipe.sha256 {
                return Err(invalid(format!(
                    "configured recipe {:?} changed inputs; give the new recipe a new id before snapshotting",
                    recipe.id
                )));
            }
            continue;
        }
        let execution_id = format!("planned-{}", recipe.id);
        recipes.push(recipe.clone());
        executions.push(Execution {
            id: execution_id,
            recipe: recipe.id.clone(),
            recipe_sha256: recipe.sha256.clone(),
            outcome: Outcome::Incomplete,
            output: None,
            error: None,
        });
    }
    let evidence = workspace
        .selections
        .iter()
        .map(|selection| EvidenceSelection {
            name: selection.name.clone(),
            execution: selection.execution.clone(),
            status: if is_selection_current(workspace, selection, &model, &data) {
                EvidenceStatus::Current
            } else {
                EvidenceStatus::Historical
            },
        })
        .collect();
    let manifest = Manifest {
        question: workspace.question.clone(),
        estimand_description: workspace.estimand_description.clone(),
        estimand_parameter: workspace.estimand_parameter.clone(),
        source: workspace.source.clone(),
        model,
        data,
        decisions: workspace.decisions.clone(),
        recipes,
        executions,
        evidence,
        interpretation: workspace.interpretation.clone(),
        unresolved_questions: workspace.unresolved_questions.clone(),
    };
    let bytes = manifest.to_bytes()?;
    let parsed = Manifest::parse_bytes(&bytes)?;
    // Ensure every referenced local object already exists before creating an
    // export directory, including reason-only citations outside computation.
    for reference in parsed.direct_references() {
        store::read(root, reference)?;
    }
    for decision in &parsed.decisions {
        for citation in &decision.cites {
            store::read_digest(root, citation)?;
        }
    }
    Ok(parsed)
}

fn copy_manifest_closure(
    source_root: &Path,
    destination_root: &Path,
    manifest: &Manifest,
    visited: &mut HashSet<String>,
) -> Result<(), Error> {
    for reference in manifest.direct_references() {
        if visited.insert(reference.sha256.as_str().to_string()) {
            store::import_reference(source_root, destination_root, reference)?;
        }
    }
    for decision in &manifest.decisions {
        for citation in &decision.cites {
            if visited.insert(citation.as_str().to_string()) {
                store::import_digest(source_root, destination_root, citation)?;
            }
        }
    }
    if let Some(source) = &manifest.source {
        let bytes = store::read(source_root, &source.manifest)?;
        let parent = Manifest::parse_bytes(&bytes)?;
        copy_manifest_closure(source_root, destination_root, &parent, visited)?;
    }
    Ok(())
}

fn snapshot(argv: &[String]) -> Result<(), Error> {
    let (positional, flags) = parse_flags(argv, "snapshot", 1, &["--out"])?;
    let root = Path::new(&positional[0]);
    let out = Path::new(&flags["--out"]);
    let workspace = workspace::load(root)?;
    let (model_bytes, data_bytes) = current_inputs(root)?;
    validate_model_data(&model_bytes, &data_bytes)?;
    let (model, data) = insert_inputs(root, &model_bytes, &data_bytes)?;
    let manifest = collect_manifest(root, &workspace, model, data)?;
    let manifest_bytes = manifest.to_bytes()?;
    // A supported publication must be verifiable before any destination is
    // created. This also applies the total ancestry bound to the prospective
    // manifest rather than publishing an unusable seventeenth generation.
    verify_bundle(&manifest_bytes, |digest| store::read_digest(root, digest))?;

    create_fresh_directory(out, "snapshot bundle")?;
    let result = (|| {
        let mut visited = HashSet::new();
        copy_manifest_closure(root, out, &manifest, &mut visited)?;
        let path = out.join("manifest.json");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| invalid(format!("cannot publish manifest {:?}: {error}", path)))?;
        file.write_all(&manifest_bytes)
            .and_then(|_| file.sync_all())
            .map_err(|error| invalid(format!("cannot write manifest {:?}: {error}", path)))?;
        let id = snapshot_digest(&manifest_bytes);
        emit(&Value::Object(vec![
            ("investigation_command".into(), string("snapshot")),
            ("snapshot_id".into(), string(id.prefixed())),
            (
                "manifest_bytes".into(),
                Value::Int(manifest_bytes.len() as i64),
            ),
            ("object_count".into(), Value::Int(visited.len() as i64)),
            ("bundle".into(), string(out.display().to_string())),
        ]))
    })();
    if result.is_err() {
        let _ = fs::remove_file(out.join("manifest.json"));
    }
    result
}

fn verify(path: &Path) -> Result<(Manifest, bayesite_core::investigation::Verification), Error> {
    let manifest_bytes = read_bytes(&path.join("manifest.json"), "bundle manifest")?;
    verify_bundle(&manifest_bytes, |digest| store::read_digest(path, digest))
}

fn verify_command(path: &Path) -> Result<(), Error> {
    let (_, report) = verify(path)?;
    emit(&report.to_value())
}

fn fork(argv: &[String]) -> Result<(), Error> {
    let (positional, flags) = parse_flags(argv, "fork", 1, &["--at", "--out"])?;
    let source_root = Path::new(&positional[0]);
    let out = Path::new(&flags["--out"]);
    let at = &flags["--at"];
    let (manifest, verification) = verify(source_root)?;
    if manifest.decision(at).is_none() {
        return Err(invalid(format!(
            "source snapshot has no decision {at:?}; choose an exact recorded decision id"
        )));
    }
    let engine = manifest
        .recipes
        .first()
        .map(|recipe| recipe.engine.clone())
        .ok_or_else(|| invalid("source snapshot has no pinned recipe engine to continue with"))?;
    let manifest_bytes = read_bytes(&source_root.join("manifest.json"), "source manifest")?;
    let source_manifest = store::reference(
        &manifest_bytes,
        ArtifactKind::InvestigationManifest,
        "investigation-snapshot-v0-provisional",
    )?;

    create_fresh_directory(out, "fork workspace")?;
    let result = (|| {
        store::import_all(source_root, out)?;
        let inserted = store::insert(
            out,
            &manifest_bytes,
            ArtifactKind::InvestigationManifest,
            "investigation-snapshot-v0-provisional",
        )?;
        debug_assert_eq!(inserted, source_manifest);
        fs::create_dir_all(out.join("inputs"))
            .map_err(|error| invalid(format!("cannot create fork inputs: {error}")))?;
        let model = store::read(out, &manifest.model)?;
        let data = store::read(out, &manifest.data)?;
        let (model_path, data_path) = input_paths(out);
        fs::write(&model_path, model)
            .map_err(|error| invalid(format!("cannot write fork model copy: {error}")))?;
        fs::write(&data_path, data)
            .map_err(|error| invalid(format!("cannot write fork data copy: {error}")))?;
        let workspace = Workspace {
            question: manifest.question.clone(),
            estimand_description: manifest.estimand_description.clone(),
            estimand_parameter: manifest.estimand_parameter.clone(),
            source: Some(Source {
                snapshot_id: verification.snapshot_id.clone(),
                manifest: source_manifest,
                decision: at.clone(),
            }),
            engine,
            decisions: manifest.decisions.clone(),
            recipes: vec![],
            attempts: vec![],
            selections: vec![],
            interpretation:
                "Continuation is incomplete; no new scientific result has been computed.".into(),
            unresolved_questions: manifest.unresolved_questions.clone(),
        };
        workspace::save(out, &workspace)?;
        emit(&Value::Object(vec![
            ("investigation_command".into(), string("fork")),
            ("workspace".into(), string(out.display().to_string())),
            (
                "source_snapshot_id".into(),
                string(verification.snapshot_id.prefixed()),
            ),
            ("branch_decision".into(), string(at)),
            ("inherited_evidence_status".into(), string("historical")),
        ]))
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(out);
    }
    result
}

fn replay(argv: &[String], capabilities: &str) -> Result<(), Error> {
    let (positional, flags) = parse_flags(argv, "replay", 1, &["--recipe", "--out"])?;
    let source_root = Path::new(&positional[0]);
    let out = Path::new(&flags["--out"]);
    let (manifest, verification) = verify(source_root)?;
    let recipe_id = &flags["--recipe"];
    let recipe = manifest
        .recipes
        .iter()
        .find(|recipe| recipe.id == *recipe_id)
        .ok_or_else(|| invalid(format!("snapshot has no recipe {recipe_id:?}")))?;
    check_engine(&recipe.engine, source_root, capabilities)?;
    let model = store::read(source_root, &recipe.model)?;
    let data = store::read(source_root, &recipe.data)?;
    let fit = recipe
        .fit
        .as_ref()
        .map(|fit| store::read(source_root, fit))
        .transpose()?;

    create_fresh_directory(out, "replay directory")?;
    let result = (|| {
        let output_bytes = execute_recipe(recipe, &model, &data, fit.as_deref())?;
        let (kind, format) = output_description(recipe.operation);
        let output = store::insert(out, &output_bytes, kind, format)?;
        let expected = manifest
            .executions
            .iter()
            .find(|execution| {
                execution.recipe_sha256 == recipe.sha256 && execution.outcome == Outcome::Completed
            })
            .and_then(|execution| execution.output.as_ref());
        let exact = expected.map(|expected| expected.sha256 == output.sha256);
        let report = Value::Object(vec![
            ("replay_format".into(), string("v0-provisional")),
            (
                "source_snapshot_id".into(),
                string(verification.snapshot_id.prefixed()),
            ),
            ("input_integrity".into(), string("verified")),
            ("engine_match".into(), Value::Bool(true)),
            ("execution_outcome".into(), string("completed")),
            ("recipe".into(), string(&recipe.id)),
            ("recipe_sha256".into(), string(recipe.sha256.prefixed())),
            ("output_sha256".into(), string(output.sha256.prefixed())),
            (
                "expected_output_sha256".into(),
                expected
                    .map(|expected| string(expected.sha256.prefixed()))
                    .unwrap_or(Value::Null),
            ),
            (
                "exact_output_bytes_agree".into(),
                exact.map(Value::Bool).unwrap_or(Value::Null),
            ),
            (
                "cross_target_numerical_comparison".into(),
                string("unsupported"),
            ),
        ]);
        let report_bytes = format!("{}\n", json::write(&report)?).into_bytes();
        fs::write(out.join("replay.json"), &report_bytes)
            .map_err(|error| invalid(format!("cannot write replay report: {error}")))?;
        store::insert(
            out,
            &report_bytes,
            ArtifactKind::ReplayReport,
            "replay-v0-provisional",
        )?;
        emit(&report)
    })();
    if result.is_err() {
        let _ = fs::remove_file(out.join("replay.json"));
    }
    result
}

fn export(argv: &[String]) -> Result<(), Error> {
    let source_value = argv
        .first()
        .ok_or_else(|| invalid(format!("export needs a snapshot path; {}", usage())))?;
    let source = Path::new(source_value);
    let mut viewer = false;
    let mut public_data_confirmed = false;
    let mut out_value: Option<&String> = None;
    let mut index = 1usize;
    while index < argv.len() {
        match argv[index].as_str() {
            "--viewer" if !viewer => viewer = true,
            "--public-data-confirmed" if !public_data_confirmed => public_data_confirmed = true,
            "--out" if out_value.is_none() => {
                index += 1;
                out_value =
                    Some(argv.get(index).ok_or_else(|| {
                        invalid("investigation export --out requires a directory")
                    })?);
            }
            other => {
                return Err(invalid(format!(
                    "unknown or duplicate investigation export option {other:?}"
                )))
            }
        }
        index += 1;
    }
    if !viewer || !public_data_confirmed {
        return Err(invalid(
            "investigation export requires explicit --viewer and --public-data-confirmed",
        ));
    }
    let out = Path::new(
        out_value.ok_or_else(|| invalid("investigation export needs --out <directory>"))?,
    );
    let (manifest, verification) = verify(source)?;
    let manifest_bytes = read_bytes(&source.join("manifest.json"), "investigation manifest")?;
    let engine = manifest
        .recipes
        .first()
        .map(|recipe| &recipe.engine)
        .ok_or_else(|| invalid("investigation export needs at least one recipe"))?;
    if manifest
        .recipes
        .iter()
        .any(|recipe| &recipe.engine != engine)
    {
        return Err(invalid(
            "viewer export currently requires every recipe to use one pinned engine",
        ));
    }
    let engine_bytes = store::read(source, &engine.binary)?;

    create_fresh_directory(out, "viewer export")?;
    let result = (|| {
        create_fresh_directory(&out.join("bundle"), "exported bundle directory")?;
        store::import_all(source, &out.join("bundle"))?;
        write_new(&out.join("bundle/manifest.json"), &manifest_bytes)?;
        create_fresh_directory(&out.join("downloads"), "export downloads directory")?;
        let downloaded_engine = out.join("downloads/bayesite-engine");
        write_new(&downloaded_engine, &engine_bytes)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&downloaded_engine, fs::Permissions::from_mode(0o755)).map_err(
                |error| invalid(format!("cannot mark exported engine executable: {error}")),
            )?;
        }

        let assets: [(&str, &[u8]); 3] = [
            (
                "index.html",
                include_bytes!("../../../../../demo/investigation/index.html"),
            ),
            (
                "viewer.js",
                include_bytes!("../../../../../demo/investigation/viewer.js"),
            ),
            (
                "style.css",
                include_bytes!("../../../../../demo/investigation/style.css"),
            ),
        ];
        let mut asset_entries = Vec::new();
        for (name, bytes) in assets {
            write_new(&out.join(name), bytes)?;
            asset_entries.push((name.to_string(), artifact_digest(bytes), bytes.len()));
        }
        write_new(
            &out.join("LICENSE"),
            include_bytes!("../../../../../LICENSE"),
        )?;
        write_new(&out.join("NOTICE"), include_bytes!("../../../../../NOTICE"))?;
        write_new(
            &out.join("PROTOCOL.md"),
            include_bytes!("../../../../../examples/investigation-counts/PROTOCOL.md"),
        )?;
        write_new(
            &out.join("CONTINUING.md"),
            include_bytes!("../../../../../docs/investigation-workspace-v0.md"),
        )?;
        write_new(
            &out.join("IR-FORMAT.md"),
            include_bytes!("../../../../../docs/ir-format-v1.md"),
        )?;
        write_new(
            &out.join("IR-TAGS.md"),
            include_bytes!("../../../../../docs/ir-v1-tags.md"),
        )?;
        write_new(
            &out.join("INSPECTION.md"),
            include_bytes!("../../../../../docs/inspection-v0.md"),
        )?;
        write_new(
            &out.join("PUBLIC-DATA-CONFIRMATION.txt"),
            b"public_data_confirmed=true\nThis explicit publication choice is not a privacy scan, author signature, or scientific approval.\n",
        )?;

        let entry = Value::Object(vec![
            ("publication_format".into(), string("v0-provisional")),
            (
                "snapshot_id".into(),
                string(verification.snapshot_id.prefixed()),
            ),
            ("manifest".into(), string("bundle/manifest.json")),
            ("public_data_confirmed".into(), Value::Bool(true)),
            (
                "engine".into(),
                Value::Object(vec![
                    ("target".into(), string(&engine.target)),
                    ("profile".into(), string(&engine.profile)),
                    ("sha256".into(), string(engine.binary.sha256.as_str())),
                    ("bytes".into(), Value::Int(engine.binary.bytes as i64)),
                    ("download".into(), string("downloads/bayesite-engine")),
                ]),
            ),
            (
                "viewer_assets".into(),
                Value::Object(
                    asset_entries
                        .into_iter()
                        .map(|(name, digest, bytes)| {
                            (
                                name,
                                Value::Object(vec![
                                    ("sha256".into(), string(digest.as_str())),
                                    ("bytes".into(), Value::Int(bytes as i64)),
                                ]),
                            )
                        })
                        .collect(),
                ),
            ),
            (
                "verification_scope".into(),
                string(
                    "The viewer verifies manifest identity and displayed object bytes when WebCrypto is available; use the bundled engine for full recursive CLI verification.",
                ),
            ),
        ]);
        write_new(
            &out.join("entry.json"),
            format!("{}\n", json::write(&entry)?).as_bytes(),
        )?;
        emit(&Value::Object(vec![
            ("investigation_command".into(), string("export")),
            (
                "snapshot_id".into(),
                string(verification.snapshot_id.prefixed()),
            ),
            ("out".into(), string(out.display().to_string())),
            ("public_data_confirmed".into(), Value::Bool(true)),
            (
                "engine_sha256".into(),
                string(engine.binary.sha256.prefixed()),
            ),
        ]))
    })();
    if result.is_err() {
        // A failed export has no usable entry point and is never accepted as a
        // replacement target; leave bytes for diagnosis without clobbering.
        let _ = fs::remove_file(out.join("entry.json"));
    }
    result
}

pub fn run(argv: &[String], capabilities: &str) -> Result<(), Error> {
    let Some(command) = argv.first() else {
        return Err(invalid(format!(
            "investigation needs a subcommand; {}",
            usage()
        )));
    };
    match command.as_str() {
        "init" => init(&argv[1..], capabilities),
        "inspect" => inspect_workspace(Path::new(one_positional(&argv[1..], "inspect")?)),
        "run" => run_recipe(&argv[1..], capabilities),
        "snapshot" => snapshot(&argv[1..]),
        "verify" => verify_command(Path::new(one_positional(&argv[1..], "verify")?)),
        "fork" => fork(&argv[1..]),
        "replay" => replay(&argv[1..], capabilities),
        "export" => export(&argv[1..]),
        other => Err(invalid(format!(
            "unknown investigation subcommand {other:?}; {}",
            usage()
        ))),
    }
}
