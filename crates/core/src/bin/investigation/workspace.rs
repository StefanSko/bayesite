use std::fs;
use std::path::{Path, PathBuf};

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::investigation::identity::artifact_digest;
use bayesite_core::investigation::manifest::{
    validate_identifier, ArtifactRef, Decision, EngineIdentity, Execution, Operation, Recipe,
    Source,
};
use bayesite_core::json::{self, Value};

pub const WORKSPACE_FORMAT: &str = "v0-provisional";
pub const METADATA_FORMAT: &str = "v0-provisional";

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

fn checked_object<'a>(
    value: &'a Value,
    context: &str,
    allowed: &[&str],
    required: &[&str],
) -> Result<&'a [(String, Value)], Error> {
    let Value::Object(entries) = value else {
        return Err(invalid(format!("{context} must be an object")));
    };
    for (index, (name, _)) in entries.iter().enumerate() {
        if entries[..index]
            .iter()
            .any(|(existing, _)| existing == name)
        {
            return Err(invalid(format!(
                "{context} has duplicate field {name:?}; remove one"
            )));
        }
        if !allowed.contains(&name.as_str()) {
            return Err(invalid(format!(
                "{context} has unknown field {name:?}; remove it"
            )));
        }
    }
    for name in required {
        if !entries.iter().any(|(candidate, _)| candidate == name) {
            return Err(invalid(format!("{context} needs field {name:?}")));
        }
    }
    Ok(entries)
}

fn string(value: impl Into<String>) -> Value {
    Value::Str(value.into())
}

fn required_str<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a str, Error> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("{context}.{name} must be a string")))
}

fn required_array<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a [Value], Error> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("{context}.{name} must be an array")))
}

fn text(value: &str, context: &str) -> Result<String, Error> {
    if value.len() > 16_384 {
        Err(invalid(format!(
            "{context} must be at most 16384 UTF-8 bytes"
        )))
    } else {
        Ok(value.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct ConfiguredRecipe {
    pub id: String,
    pub operation: Operation,
    pub settings: Value,
}

impl ConfiguredRecipe {
    fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["id", "operation", "settings"],
            &["id", "operation", "settings"],
        )?;
        Ok(Self {
            id: validate_identifier(required_str(value, "id", context)?, context)?,
            operation: Operation::parse(required_str(value, "operation", context)?, context)?,
            settings: value.get("settings").expect("required").clone(),
        })
    }

    fn to_value(&self) -> Value {
        Value::Object(vec![
            ("id".into(), string(&self.id)),
            ("operation".into(), string(self.operation.as_str())),
            ("settings".into(), self.settings.clone()),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct Attempt {
    pub recipe: Recipe,
    pub execution: Execution,
}

impl Attempt {
    fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["recipe", "execution"],
            &["recipe", "execution"],
        )?;
        let recipe = Recipe::parse(
            value.get("recipe").expect("required"),
            &format!("{context}.recipe"),
        )?;
        let execution = Execution::parse(
            value.get("execution").expect("required"),
            &format!("{context}.execution"),
        )?;
        if execution.recipe != recipe.id || execution.recipe_sha256 != recipe.sha256 {
            return Err(invalid(format!(
                "{context} execution does not match its embedded recipe"
            )));
        }
        Ok(Self { recipe, execution })
    }

    fn to_value(&self) -> Value {
        Value::Object(vec![
            ("recipe".into(), self.recipe.to_value()),
            ("execution".into(), self.execution.to_value()),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct Selection {
    pub name: String,
    pub execution: String,
}

impl Selection {
    fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["name", "execution"],
            &["name", "execution"],
        )?;
        Ok(Self {
            name: validate_identifier(required_str(value, "name", context)?, context)?,
            execution: validate_identifier(required_str(value, "execution", context)?, context)?,
        })
    }

    fn to_value(&self) -> Value {
        Value::Object(vec![
            ("name".into(), string(&self.name)),
            ("execution".into(), string(&self.execution)),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct Workspace {
    pub question: String,
    pub estimand_description: String,
    pub estimand_parameter: String,
    pub source: Option<Source>,
    pub engine: EngineIdentity,
    pub decisions: Vec<Decision>,
    pub recipes: Vec<ConfiguredRecipe>,
    pub attempts: Vec<Attempt>,
    pub selections: Vec<Selection>,
    pub interpretation: String,
    pub unresolved_questions: Vec<String>,
}

impl Workspace {
    pub fn parse(value: &Value) -> Result<Self, Error> {
        let context = "investigation workspace";
        checked_object(
            value,
            context,
            &[
                "investigation_workspace",
                "question",
                "estimand",
                "source",
                "engine",
                "decisions",
                "recipes",
                "attempts",
                "selections",
                "interpretation",
                "unresolved_questions",
            ],
            &[
                "investigation_workspace",
                "question",
                "estimand",
                "source",
                "engine",
                "decisions",
                "recipes",
                "attempts",
                "selections",
                "interpretation",
                "unresolved_questions",
            ],
        )?;
        if required_str(value, "investigation_workspace", context)? != WORKSPACE_FORMAT {
            return Err(invalid(format!(
                "unsupported investigation_workspace; expected {WORKSPACE_FORMAT:?}"
            )));
        }
        let estimand = value.get("estimand").expect("required");
        checked_object(
            estimand,
            "workspace estimand",
            &["description", "parameter"],
            &["description", "parameter"],
        )?;
        let source = match value.get("source").expect("required") {
            Value::Null => None,
            source => Some(Source::parse(source, "workspace source")?),
        };
        let parse_array = |name: &str| required_array(value, name, context);
        let decisions = parse_array("decisions")?
            .iter()
            .enumerate()
            .map(|(index, value)| Decision::parse(value, &format!("workspace decisions[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let recipes = parse_array("recipes")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                ConfiguredRecipe::parse(value, &format!("workspace recipes[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let attempts = parse_array("attempts")?
            .iter()
            .enumerate()
            .map(|(index, value)| Attempt::parse(value, &format!("workspace attempts[{index}]")))
            .collect::<Result<Vec<_>, _>>()?;
        let selections = parse_array("selections")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                Selection::parse(value, &format!("workspace selections[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unresolved_questions = parse_array("unresolved_questions")?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .ok_or_else(|| {
                        invalid(format!(
                            "workspace unresolved_questions[{index}] must be a string"
                        ))
                    })
                    .and_then(|value| text(value, "unresolved question"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let workspace = Self {
            question: text(required_str(value, "question", context)?, "question")?,
            estimand_description: text(
                required_str(estimand, "description", "workspace estimand")?,
                "estimand description",
            )?,
            estimand_parameter: validate_identifier(
                required_str(estimand, "parameter", "workspace estimand")?,
                "estimand parameter",
            )?,
            source,
            engine: EngineIdentity::parse(
                value.get("engine").expect("required"),
                "workspace engine",
            )?,
            decisions,
            recipes,
            attempts,
            selections,
            interpretation: text(
                required_str(value, "interpretation", context)?,
                "interpretation",
            )?,
            unresolved_questions,
        };
        workspace.validate()?;
        Ok(workspace)
    }

    fn validate(&self) -> Result<(), Error> {
        for index in 0..self.decisions.len() {
            let decision = &self.decisions[index];
            if self.decisions[..index]
                .iter()
                .any(|earlier| earlier.id == decision.id)
            {
                return Err(invalid(format!(
                    "workspace has duplicate decision id {:?}",
                    decision.id
                )));
            }
            if let Some(parent) = &decision.parent {
                if !self.decisions[..index]
                    .iter()
                    .any(|earlier| earlier.id == *parent)
                {
                    return Err(invalid(format!(
                        "decision {:?} parent {:?} must name an earlier decision",
                        decision.id, parent
                    )));
                }
            }
        }
        for index in 0..self.recipes.len() {
            if self.recipes[..index]
                .iter()
                .any(|earlier| earlier.id == self.recipes[index].id)
            {
                return Err(invalid(format!(
                    "workspace has duplicate recipe id {:?}",
                    self.recipes[index].id
                )));
            }
        }
        for index in 0..self.attempts.len() {
            if self.attempts[..index]
                .iter()
                .any(|earlier| earlier.execution.id == self.attempts[index].execution.id)
            {
                return Err(invalid(format!(
                    "workspace has duplicate execution id {:?}",
                    self.attempts[index].execution.id
                )));
            }
        }
        for selection in &self.selections {
            if !self
                .attempts
                .iter()
                .any(|attempt| attempt.execution.id == selection.execution)
            {
                return Err(invalid(format!(
                    "selection {:?} references missing execution {:?}",
                    selection.name, selection.execution
                )));
            }
        }
        Ok(())
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("investigation_workspace".into(), string(WORKSPACE_FORMAT)),
            ("question".into(), string(&self.question)),
            (
                "estimand".into(),
                Value::Object(vec![
                    ("description".into(), string(&self.estimand_description)),
                    ("parameter".into(), string(&self.estimand_parameter)),
                ]),
            ),
            (
                "source".into(),
                self.source
                    .as_ref()
                    .map(Source::to_value)
                    .unwrap_or(Value::Null),
            ),
            ("engine".into(), self.engine.to_value()),
            (
                "decisions".into(),
                Value::Array(self.decisions.iter().map(Decision::to_value).collect()),
            ),
            (
                "recipes".into(),
                Value::Array(
                    self.recipes
                        .iter()
                        .map(ConfiguredRecipe::to_value)
                        .collect(),
                ),
            ),
            (
                "attempts".into(),
                Value::Array(self.attempts.iter().map(Attempt::to_value).collect()),
            ),
            (
                "selections".into(),
                Value::Array(self.selections.iter().map(Selection::to_value).collect()),
            ),
            ("interpretation".into(), string(&self.interpretation)),
            (
                "unresolved_questions".into(),
                Value::Array(self.unresolved_questions.iter().map(string).collect()),
            ),
        ])
    }
}

pub struct Metadata {
    pub question: String,
    pub estimand_description: String,
    pub estimand_parameter: String,
    pub decisions: Vec<Decision>,
    pub recipes: Vec<ConfiguredRecipe>,
    pub interpretation: String,
    pub unresolved_questions: Vec<String>,
}

impl Metadata {
    pub fn parse(value: &Value, model: &ArtifactRef, data: &ArtifactRef) -> Result<Self, Error> {
        let context = "investigation metadata";
        checked_object(
            value,
            context,
            &[
                "investigation_metadata",
                "question",
                "estimand",
                "decisions",
                "recipes",
                "interpretation",
                "unresolved_questions",
            ],
            &[
                "investigation_metadata",
                "question",
                "estimand",
                "decisions",
                "recipes",
                "interpretation",
                "unresolved_questions",
            ],
        )?;
        if required_str(value, "investigation_metadata", context)? != METADATA_FORMAT {
            return Err(invalid(format!(
                "unsupported investigation_metadata; expected {METADATA_FORMAT:?}"
            )));
        }
        let estimand = value.get("estimand").expect("required");
        checked_object(
            estimand,
            "metadata estimand",
            &["description", "parameter"],
            &["description", "parameter"],
        )?;
        let decisions = required_array(value, "decisions", context)?
            .iter()
            .enumerate()
            .map(|(index, decision)| {
                let mut decision = decision.clone();
                let Value::Object(entries) = &mut decision else {
                    return Err(invalid(format!("metadata decisions[{index}] must be an object")));
                };
                let cites = entries
                    .iter_mut()
                    .find(|(name, _)| name == "cites")
                    .map(|(_, value)| value)
                    .ok_or_else(|| invalid(format!("metadata decisions[{index}] needs cites")))?;
                let Value::Array(citations) = cites else {
                    return Err(invalid(format!(
                        "metadata decisions[{index}].cites must be an array"
                    )));
                };
                for citation in citations {
                    match citation.as_str() {
                        Some("model") => *citation = string(model.sha256.as_str()),
                        Some("data") => *citation = string(data.sha256.as_str()),
                        Some(other) => {
                            bayesite_core::investigation::identity::Digest::parse(
                                other,
                                "metadata decision citation",
                            )?;
                        }
                        None => {
                            return Err(invalid(format!(
                                "metadata decisions[{index}].cites entries must be model, data, or a digest string"
                            )))
                        }
                    }
                }
                Decision::parse(&decision, &format!("metadata decisions[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let recipes = required_array(value, "recipes", context)?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                ConfiguredRecipe::parse(value, &format!("metadata recipes[{index}]"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let unresolved_questions = required_array(value, "unresolved_questions", context)?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_str()
                    .ok_or_else(|| {
                        invalid(format!(
                            "metadata unresolved_questions[{index}] must be a string"
                        ))
                    })
                    .and_then(|value| text(value, "unresolved question"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            question: text(required_str(value, "question", context)?, "question")?,
            estimand_description: text(
                required_str(estimand, "description", "metadata estimand")?,
                "estimand description",
            )?,
            estimand_parameter: validate_identifier(
                required_str(estimand, "parameter", "metadata estimand")?,
                "estimand parameter",
            )?,
            decisions,
            recipes,
            interpretation: text(
                required_str(value, "interpretation", context)?,
                "interpretation",
            )?,
            unresolved_questions,
        })
    }
}

pub fn workspace_file(root: &Path) -> PathBuf {
    root.join("investigation.json")
}

pub fn load(root: &Path) -> Result<Workspace, Error> {
    let path = workspace_file(root);
    let text = fs::read_to_string(&path)
        .map_err(|error| invalid(format!("cannot read workspace {:?}: {error}", path)))?;
    let mut value = json::parse(&text)?;
    // Mutable workspaces accept the symbolic citations "model" and "data" so
    // an editor never has to calculate content hashes. Every command resolves
    // them from the actual working bytes before validation or snapshotting.
    let model = fs::read(root.join("inputs/model.json")).map_err(|error| {
        invalid(format!(
            "cannot read workspace model for citations: {error}"
        ))
    })?;
    let data = fs::read(root.join("inputs/data.json"))
        .map_err(|error| invalid(format!("cannot read workspace data for citations: {error}")))?;
    let model_digest = artifact_digest(&model);
    let data_digest = artifact_digest(&data);
    if let Some(Value::Array(decisions)) = match &mut value {
        Value::Object(entries) => entries
            .iter_mut()
            .find(|(name, _)| name == "decisions")
            .map(|(_, value)| value),
        _ => None,
    } {
        for decision in decisions {
            let Some(Value::Array(citations)) = (match decision {
                Value::Object(entries) => entries
                    .iter_mut()
                    .find(|(name, _)| name == "cites")
                    .map(|(_, value)| value),
                _ => None,
            }) else {
                continue;
            };
            for citation in citations {
                match citation.as_str() {
                    Some("model") => *citation = string(model_digest.as_str()),
                    Some("data") => *citation = string(data_digest.as_str()),
                    _ => {}
                }
            }
        }
    }
    Workspace::parse(&value)
}

pub fn save(root: &Path, workspace: &Workspace) -> Result<(), Error> {
    fs::create_dir_all(root)
        .map_err(|error| invalid(format!("cannot create workspace {:?}: {error}", root)))?;
    let path = workspace_file(root);
    let temporary = root.join(format!(".investigation.json.tmp-{}", std::process::id()));
    let text = format!("{}\n", json::write(&workspace.to_value())?);
    fs::write(&temporary, text)
        .map_err(|error| invalid(format!("cannot write workspace temporary file: {error}")))?;
    fs::rename(&temporary, &path).map_err(|error| {
        invalid(format!(
            "cannot publish mutable workspace {:?}: {error}",
            path
        ))
    })
}
