//! Typed v0-provisional investigation snapshot records and pure validation.

use std::collections::{HashMap, HashSet};

use sha2::Sha256;

use crate::error::{Error, ErrorKind};
use crate::json::{self, Value};

use super::identity::{self, Digest};

pub const SNAPSHOT_FORMAT: &str = "v0-provisional";
pub const MAX_MANIFEST_BYTES: usize = 1_048_576;
pub const MAX_OBJECT_BYTES: usize = 67_108_864;
pub const MAX_OBJECTS: usize = 256;
pub const MAX_ANCESTRY: usize = 16;
pub const MAX_RECORDS: usize = 256;

fn malformed(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::MalformedDocument, message)
}

fn string(value: impl Into<String>) -> Value {
    Value::Str(value.into())
}

fn checked_object<'a>(
    value: &'a Value,
    context: &str,
    allowed: &[&str],
    required: &[&str],
) -> Result<&'a [(String, Value)], Error> {
    let Value::Object(entries) = value else {
        return Err(malformed(format!("{context} must be an object")));
    };
    for (index, (name, _)) in entries.iter().enumerate() {
        if entries[..index]
            .iter()
            .any(|(existing, _)| existing == name)
        {
            return Err(malformed(format!(
                "{context} has duplicate field {name:?}; remove one"
            )));
        }
        if !allowed.contains(&name.as_str()) {
            return Err(malformed(format!(
                "{context} has unknown field {name:?}; remove it"
            )));
        }
    }
    for name in required {
        if !entries.iter().any(|(candidate, _)| candidate == name) {
            return Err(malformed(format!("{context} needs field {name:?}")));
        }
    }
    Ok(entries)
}

fn required_str<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a str, Error> {
    value
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(|| malformed(format!("{context} field {name:?} must be a string")))
}

fn required_array<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a [Value], Error> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| malformed(format!("{context} field {name:?} must be an array")))
}

fn parse_record_array<T>(
    document: &Value,
    name: &str,
    context: &str,
    parser: impl Fn(&Value, &str) -> Result<T, Error>,
) -> Result<Vec<T>, Error> {
    let entries = required_array(document, name, context)?;
    if entries.len() > MAX_RECORDS {
        return Err(malformed(format!(
            "investigation manifest {name} exceeds the {MAX_RECORDS}-record limit"
        )));
    }
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| parser(entry, &format!("manifest {name}[{index}]")))
        .collect()
}

fn bounded_text(value: &str, context: &str) -> Result<String, Error> {
    if value.len() > 16_384 {
        return Err(malformed(format!(
            "{context} is too long; keep it at or below 16384 UTF-8 bytes"
        )));
    }
    Ok(value.to_string())
}

pub fn validate_identifier(value: &str, context: &str) -> Result<String, Error> {
    if value.is_empty()
        || value.len() > 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(malformed(format!(
            "{context} must be 1..=128 ASCII letters, digits, '.', '_' or '-'"
        )));
    }
    Ok(value.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    ModelIr,
    Data,
    Inspection,
    PosteriorDraws,
    Diagnostics,
    PosteriorCheck,
    EngineBinary,
    EngineCapabilities,
    InvestigationManifest,
    ReplayReport,
}

impl ArtifactKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ModelIr => "model_ir",
            Self::Data => "data",
            Self::Inspection => "inspection",
            Self::PosteriorDraws => "posterior_draws",
            Self::Diagnostics => "diagnostics",
            Self::PosteriorCheck => "posterior_check",
            Self::EngineBinary => "engine_binary",
            Self::EngineCapabilities => "engine_capabilities",
            Self::InvestigationManifest => "investigation_manifest",
            Self::ReplayReport => "replay_report",
        }
    }

    fn parse(value: &str, context: &str) -> Result<Self, Error> {
        match value {
            "model_ir" => Ok(Self::ModelIr),
            "data" => Ok(Self::Data),
            "inspection" => Ok(Self::Inspection),
            "posterior_draws" => Ok(Self::PosteriorDraws),
            "diagnostics" => Ok(Self::Diagnostics),
            "posterior_check" => Ok(Self::PosteriorCheck),
            "engine_binary" => Ok(Self::EngineBinary),
            "engine_capabilities" => Ok(Self::EngineCapabilities),
            "investigation_manifest" => Ok(Self::InvestigationManifest),
            "replay_report" => Ok(Self::ReplayReport),
            other => Err(malformed(format!(
                "{context} artifact kind {other:?} is unsupported"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ArtifactRef {
    pub sha256: Digest,
    pub bytes: usize,
    pub kind: ArtifactKind,
    pub format: String,
}

impl ArtifactRef {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["sha256", "bytes", "kind", "format"],
            &["sha256", "bytes", "kind", "format"],
        )?;
        let sha256 = Digest::parse(required_str(value, "sha256", context)?, context)?;
        let bytes = value
            .get("bytes")
            .and_then(Value::as_i64)
            .ok_or_else(|| malformed(format!("{context} field \"bytes\" must be an integer")))?;
        if bytes < 0 || bytes as u64 > MAX_OBJECT_BYTES as u64 {
            return Err(malformed(format!(
                "{context} bytes must be in 0..={MAX_OBJECT_BYTES}"
            )));
        }
        let kind = ArtifactKind::parse(required_str(value, "kind", context)?, context)?;
        let format = validate_identifier(required_str(value, "format", context)?, context)?;
        Ok(Self {
            sha256,
            bytes: bytes as usize,
            kind,
            format,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("sha256".into(), string(self.sha256.as_str())),
            ("bytes".into(), Value::Int(self.bytes as i64)),
            ("kind".into(), string(self.kind.as_str())),
            ("format".into(), string(&self.format)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineIdentity {
    pub binary: ArtifactRef,
    pub capabilities: ArtifactRef,
    pub target: String,
    pub profile: String,
}

impl EngineIdentity {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["binary", "capabilities", "target", "profile"],
            &["binary", "capabilities", "target", "profile"],
        )?;
        let binary = ArtifactRef::parse(
            value.get("binary").expect("required"),
            &format!("{context}.binary"),
        )?;
        if binary.kind != ArtifactKind::EngineBinary {
            return Err(malformed(format!(
                "{context}.binary kind must be \"engine_binary\""
            )));
        }
        let capabilities = ArtifactRef::parse(
            value.get("capabilities").expect("required"),
            &format!("{context}.capabilities"),
        )?;
        if capabilities.kind != ArtifactKind::EngineCapabilities {
            return Err(malformed(format!(
                "{context}.capabilities kind must be \"engine_capabilities\""
            )));
        }
        Ok(Self {
            binary,
            capabilities,
            target: validate_identifier(required_str(value, "target", context)?, context)?,
            profile: validate_identifier(required_str(value, "profile", context)?, context)?,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("binary".into(), self.binary.to_value()),
            ("capabilities".into(), self.capabilities.to_value()),
            ("target".into(), string(&self.target)),
            ("profile".into(), string(&self.profile)),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Inspect,
    Sample,
    Diagnose,
    PosteriorCheck,
}

impl Operation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::Sample => "sample",
            Self::Diagnose => "diagnose",
            Self::PosteriorCheck => "posterior-check",
        }
    }

    pub fn parse(value: &str, context: &str) -> Result<Self, Error> {
        match value {
            "inspect" => Ok(Self::Inspect),
            "sample" => Ok(Self::Sample),
            "diagnose" => Ok(Self::Diagnose),
            "posterior-check" => Ok(Self::PosteriorCheck),
            other => Err(malformed(format!(
                "{context} operation {other:?} is unsupported; use inspect, sample, diagnose, or posterior-check"
            ))),
        }
    }

    fn output_kind(self) -> ArtifactKind {
        match self {
            Self::Inspect => ArtifactKind::Inspection,
            Self::Sample => ArtifactKind::PosteriorDraws,
            Self::Diagnose => ArtifactKind::Diagnostics,
            Self::PosteriorCheck => ArtifactKind::PosteriorCheck,
        }
    }
}

fn normalized_settings(operation: Operation, value: &Value, context: &str) -> Result<Value, Error> {
    match operation {
        Operation::Inspect | Operation::Diagnose => {
            checked_object(value, context, &[], &[])?;
            Ok(Value::Object(vec![]))
        }
        Operation::Sample => {
            let names = [
                "chains",
                "warmup",
                "draws",
                "max_treedepth",
                "target_accept",
                "initial_step_size",
                "seed",
            ];
            checked_object(value, context, &names, &names)?;
            let positive = |name: &str| -> Result<i64, Error> {
                let parsed = value
                    .get(name)
                    .and_then(Value::as_i64)
                    .ok_or_else(|| malformed(format!("{context}.{name} must be an integer")))?;
                if parsed < 1 {
                    Err(malformed(format!("{context}.{name} must be at least 1")))
                } else {
                    Ok(parsed)
                }
            };
            let chains = positive("chains")?;
            let warmup = value
                .get("warmup")
                .and_then(Value::as_i64)
                .ok_or_else(|| malformed(format!("{context}.warmup must be an integer")))?;
            if warmup < 0 {
                return Err(malformed(format!("{context}.warmup must be non-negative")));
            }
            let draws = positive("draws")?;
            if draws < 4 {
                return Err(malformed(format!(
                    "{context}.draws must be at least 4 because fit artifacts include diagnostics"
                )));
            }
            let max_treedepth = positive("max_treedepth")?;
            if max_treedepth > 20 {
                return Err(malformed(format!(
                    "{context}.max_treedepth must be in 1..=20"
                )));
            }
            let target_accept = value
                .get("target_accept")
                .and_then(Value::as_f64)
                .ok_or_else(|| malformed(format!("{context}.target_accept must be a number")))?;
            if !(0.0..1.0).contains(&target_accept) {
                return Err(malformed(format!(
                    "{context}.target_accept must be in (0, 1)"
                )));
            }
            let initial_step_size = value
                .get("initial_step_size")
                .and_then(Value::as_f64)
                .ok_or_else(|| {
                    malformed(format!("{context}.initial_step_size must be a number"))
                })?;
            if !initial_step_size.is_finite() || initial_step_size <= 0.0 {
                return Err(malformed(format!(
                    "{context}.initial_step_size must be positive and finite"
                )));
            }
            let seed = value
                .get("seed")
                .and_then(Value::as_i64)
                .ok_or_else(|| malformed(format!("{context}.seed must be an integer")))?;
            if seed < 0 {
                return Err(malformed(format!("{context}.seed must be non-negative")));
            }
            Ok(Value::Object(vec![
                ("chains".into(), Value::Int(chains)),
                ("warmup".into(), Value::Int(warmup)),
                ("draws".into(), Value::Int(draws)),
                ("max_treedepth".into(), Value::Int(max_treedepth)),
                ("target_accept".into(), Value::Float(target_accept)),
                ("initial_step_size".into(), Value::Float(initial_step_size)),
                ("seed".into(), Value::Int(seed)),
            ]))
        }
        Operation::PosteriorCheck => {
            checked_object(value, context, &["seed"], &["seed"])?;
            let seed = value
                .get("seed")
                .and_then(Value::as_i64)
                .ok_or_else(|| malformed(format!("{context}.seed must be an integer")))?;
            if seed < 0 {
                return Err(malformed(format!("{context}.seed must be non-negative")));
            }
            Ok(Value::Object(vec![("seed".into(), Value::Int(seed))]))
        }
    }
}

fn frame_ref(hasher: &mut Sha256, reference: &ArtifactRef) {
    identity::frame(hasher, reference.sha256.as_str().as_bytes());
    identity::frame(hasher, &(reference.bytes as u64).to_be_bytes());
    identity::frame(hasher, reference.kind.as_str().as_bytes());
    identity::frame(hasher, reference.format.as_bytes());
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    pub id: String,
    pub sha256: Digest,
    pub operation: Operation,
    pub model: ArtifactRef,
    pub data: ArtifactRef,
    pub fit: Option<ArtifactRef>,
    pub engine: EngineIdentity,
    pub settings: Value,
}

impl Recipe {
    pub fn new(
        id: String,
        operation: Operation,
        model: ArtifactRef,
        data: ArtifactRef,
        fit: Option<ArtifactRef>,
        engine: EngineIdentity,
        settings: Value,
    ) -> Result<Self, Error> {
        let id = validate_identifier(&id, "recipe id")?;
        if model.kind != ArtifactKind::ModelIr || data.kind != ArtifactKind::Data {
            return Err(malformed(
                "recipe model/data references must have model_ir/data kinds",
            ));
        }
        let needs_fit = matches!(operation, Operation::Diagnose | Operation::PosteriorCheck);
        if needs_fit != fit.is_some() {
            return Err(malformed(format!(
                "recipe operation {:?} {} a fit input",
                operation.as_str(),
                if needs_fit {
                    "requires"
                } else {
                    "must not carry"
                }
            )));
        }
        if fit
            .as_ref()
            .is_some_and(|reference| reference.kind != ArtifactKind::PosteriorDraws)
        {
            return Err(malformed(
                "recipe fit input kind must be \"posterior_draws\"",
            ));
        }
        let settings = normalized_settings(operation, &settings, "recipe settings")?;
        let mut recipe = Self {
            id,
            sha256: Digest(String::new()),
            operation,
            model,
            data,
            fit,
            engine,
            settings,
        };
        recipe.sha256 = recipe.compute_digest()?;
        Ok(recipe)
    }

    pub fn compute_digest(&self) -> Result<Digest, Error> {
        let settings = json::write(&self.settings)?;
        Ok(identity::recipe_digest(|hasher| {
            identity::frame(hasher, self.operation.as_str().as_bytes());
            frame_ref(hasher, &self.model);
            frame_ref(hasher, &self.data);
            match &self.fit {
                Some(fit) => {
                    identity::frame(hasher, b"fit");
                    frame_ref(hasher, fit);
                }
                None => identity::frame(hasher, b"no-fit"),
            }
            frame_ref(hasher, &self.engine.binary);
            frame_ref(hasher, &self.engine.capabilities);
            identity::frame(hasher, self.engine.target.as_bytes());
            identity::frame(hasher, self.engine.profile.as_bytes());
            identity::frame(hasher, settings.as_bytes());
        }))
    }

    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["id", "sha256", "operation", "inputs", "engine", "settings"],
            &["id", "sha256", "operation", "inputs", "engine", "settings"],
        )?;
        let id = validate_identifier(required_str(value, "id", context)?, context)?;
        let expected = Digest::parse(required_str(value, "sha256", context)?, context)?;
        let operation = Operation::parse(required_str(value, "operation", context)?, context)?;
        let inputs = value.get("inputs").expect("required");
        checked_object(
            inputs,
            &format!("{context}.inputs"),
            &["model", "data", "fit"],
            &["model", "data", "fit"],
        )?;
        let model = ArtifactRef::parse(
            inputs.get("model").expect("required"),
            &format!("{context}.inputs.model"),
        )?;
        let data = ArtifactRef::parse(
            inputs.get("data").expect("required"),
            &format!("{context}.inputs.data"),
        )?;
        let fit = match inputs.get("fit").expect("required") {
            Value::Null => None,
            reference => Some(ArtifactRef::parse(
                reference,
                &format!("{context}.inputs.fit"),
            )?),
        };
        let engine = EngineIdentity::parse(
            value.get("engine").expect("required"),
            &format!("{context}.engine"),
        )?;
        let mut recipe = Recipe::new(
            id,
            operation,
            model,
            data,
            fit,
            engine,
            value.get("settings").expect("required").clone(),
        )?;
        if recipe.sha256 != expected {
            return Err(malformed(format!(
                "{context} sha256 does not match its execution-relevant fields; expected {}",
                recipe.sha256.as_str()
            )));
        }
        recipe.sha256 = expected;
        Ok(recipe)
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("id".into(), string(&self.id)),
            ("sha256".into(), string(self.sha256.as_str())),
            ("operation".into(), string(self.operation.as_str())),
            (
                "inputs".into(),
                Value::Object(vec![
                    ("model".into(), self.model.to_value()),
                    ("data".into(), self.data.to_value()),
                    (
                        "fit".into(),
                        self.fit
                            .as_ref()
                            .map(ArtifactRef::to_value)
                            .unwrap_or(Value::Null),
                    ),
                ]),
            ),
            ("engine".into(), self.engine.to_value()),
            ("settings".into(), self.settings.clone()),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionKind {
    Note,
    AgentRecommendation,
    HumanApproval,
}

impl DecisionKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::AgentRecommendation => "agent_recommendation",
            Self::HumanApproval => "human_approval",
        }
    }

    fn parse(value: &str, context: &str) -> Result<Self, Error> {
        match value {
            "note" => Ok(Self::Note),
            "agent_recommendation" => Ok(Self::AgentRecommendation),
            "human_approval" => Ok(Self::HumanApproval),
            other => Err(malformed(format!(
                "{context} kind {other:?} must be note, agent_recommendation, or human_approval"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionInputs {
    pub model: ArtifactRef,
    pub data: ArtifactRef,
}

impl DecisionInputs {
    fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(value, context, &["model", "data"], &["model", "data"])?;
        let model = ArtifactRef::parse(
            value.get("model").expect("required"),
            &format!("{context}.model"),
        )?;
        let data = ArtifactRef::parse(
            value.get("data").expect("required"),
            &format!("{context}.data"),
        )?;
        if model.kind != ArtifactKind::ModelIr || data.kind != ArtifactKind::Data {
            return Err(malformed(format!(
                "{context} must carry model_ir and data artifact references"
            )));
        }
        Ok(Self { model, data })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("model".into(), self.model.to_value()),
            ("data".into(), self.data.to_value()),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub id: String,
    pub parent: Option<String>,
    pub reason: String,
    pub cites: Vec<Digest>,
    pub kind: DecisionKind,
    pub inputs: DecisionInputs,
}

impl Decision {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["id", "parent", "reason", "cites", "kind", "inputs"],
            &["id", "parent", "reason", "cites", "kind", "inputs"],
        )?;
        let parent = match value.get("parent").expect("required") {
            Value::Null => None,
            Value::Str(parent) => Some(validate_identifier(parent, context)?),
            _ => {
                return Err(malformed(format!(
                    "{context}.parent must be null or a string"
                )))
            }
        };
        let cites = required_array(value, "cites", context)?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| malformed(format!("{context}.cites entries must be strings")))
                    .and_then(|digest| Digest::parse(digest, &format!("{context}.cites")))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            id: validate_identifier(required_str(value, "id", context)?, context)?,
            parent,
            reason: bounded_text(required_str(value, "reason", context)?, context)?,
            cites,
            kind: DecisionKind::parse(required_str(value, "kind", context)?, context)?,
            inputs: DecisionInputs::parse(
                value.get("inputs").expect("required"),
                &format!("{context}.inputs"),
            )?,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("id".into(), string(&self.id)),
            (
                "parent".into(),
                self.parent.as_ref().map(string).unwrap_or(Value::Null),
            ),
            ("reason".into(), string(&self.reason)),
            (
                "cites".into(),
                Value::Array(
                    self.cites
                        .iter()
                        .map(|digest| string(digest.as_str()))
                        .collect(),
                ),
            ),
            ("kind".into(), string(self.kind.as_str())),
            ("inputs".into(), self.inputs.to_value()),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    Failed,
    Cancelled,
    Unsupported,
    Incomplete,
}

impl Outcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Unsupported => "unsupported",
            Self::Incomplete => "incomplete",
        }
    }

    fn parse(value: &str, context: &str) -> Result<Self, Error> {
        match value {
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "cancelled" => Ok(Self::Cancelled),
            "unsupported" => Ok(Self::Unsupported),
            "incomplete" => Ok(Self::Incomplete),
            other => Err(malformed(format!(
                "{context} outcome {other:?} is unsupported"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Execution {
    pub id: String,
    pub recipe: String,
    pub recipe_sha256: Digest,
    pub outcome: Outcome,
    pub output: Option<ArtifactRef>,
    pub error: Option<String>,
}

impl Execution {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &[
                "id",
                "recipe",
                "recipe_sha256",
                "outcome",
                "output",
                "error",
            ],
            &[
                "id",
                "recipe",
                "recipe_sha256",
                "outcome",
                "output",
                "error",
            ],
        )?;
        let outcome = Outcome::parse(required_str(value, "outcome", context)?, context)?;
        let output = match value.get("output").expect("required") {
            Value::Null => None,
            reference => Some(ArtifactRef::parse(reference, &format!("{context}.output"))?),
        };
        let error = match value.get("error").expect("required") {
            Value::Null => None,
            Value::Str(message) => Some(bounded_text(message, &format!("{context}.error"))?),
            _ => {
                return Err(malformed(format!(
                    "{context}.error must be null or a string"
                )))
            }
        };
        match outcome {
            Outcome::Completed if output.is_none() || error.is_some() => {
                return Err(malformed(format!(
                    "{context} completed outcome requires output and forbids error"
                )))
            }
            Outcome::Completed => {}
            Outcome::Incomplete if output.is_some() || error.is_some() => {
                return Err(malformed(format!(
                    "{context} incomplete outcome has neither output nor invented error/cancellation"
                )))
            }
            Outcome::Incomplete => {}
            _ if output.is_some() || error.is_none() => {
                return Err(malformed(format!(
                    "{context} non-completed outcome requires an error and forbids output"
                )))
            }
            _ => {}
        }
        Ok(Self {
            id: validate_identifier(required_str(value, "id", context)?, context)?,
            recipe: validate_identifier(required_str(value, "recipe", context)?, context)?,
            recipe_sha256: Digest::parse(required_str(value, "recipe_sha256", context)?, context)?,
            outcome,
            output,
            error,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("id".into(), string(&self.id)),
            ("recipe".into(), string(&self.recipe)),
            ("recipe_sha256".into(), string(self.recipe_sha256.as_str())),
            ("outcome".into(), string(self.outcome.as_str())),
            (
                "output".into(),
                self.output
                    .as_ref()
                    .map(ArtifactRef::to_value)
                    .unwrap_or(Value::Null),
            ),
            (
                "error".into(),
                self.error.as_ref().map(string).unwrap_or(Value::Null),
            ),
        ])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    Current,
    Historical,
}

impl EvidenceStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Current => "current",
            Self::Historical => "historical",
        }
    }

    fn parse(value: &str, context: &str) -> Result<Self, Error> {
        match value {
            "current" => Ok(Self::Current),
            "historical" => Ok(Self::Historical),
            other => Err(malformed(format!(
                "{context} status {other:?} must be current or historical"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSelection {
    pub name: String,
    pub execution: String,
    pub status: EvidenceStatus,
}

impl EvidenceSelection {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["name", "execution", "status"],
            &["name", "execution", "status"],
        )?;
        Ok(Self {
            name: validate_identifier(required_str(value, "name", context)?, context)?,
            execution: validate_identifier(required_str(value, "execution", context)?, context)?,
            status: EvidenceStatus::parse(required_str(value, "status", context)?, context)?,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("name".into(), string(&self.name)),
            ("execution".into(), string(&self.execution)),
            ("status".into(), string(self.status.as_str())),
        ])
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source {
    pub snapshot_id: Digest,
    pub manifest: ArtifactRef,
    pub decision: String,
}

impl Source {
    pub fn parse(value: &Value, context: &str) -> Result<Self, Error> {
        checked_object(
            value,
            context,
            &["snapshot_id", "manifest", "decision"],
            &["snapshot_id", "manifest", "decision"],
        )?;
        let manifest = ArtifactRef::parse(
            value.get("manifest").expect("required"),
            &format!("{context}.manifest"),
        )?;
        if manifest.kind != ArtifactKind::InvestigationManifest {
            return Err(malformed(format!(
                "{context}.manifest kind must be investigation_manifest"
            )));
        }
        Ok(Self {
            snapshot_id: Digest::parse(required_str(value, "snapshot_id", context)?, context)?,
            manifest,
            decision: validate_identifier(required_str(value, "decision", context)?, context)?,
        })
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("snapshot_id".into(), string(self.snapshot_id.as_str())),
            ("manifest".into(), self.manifest.to_value()),
            ("decision".into(), string(&self.decision)),
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub question: String,
    pub estimand_description: String,
    pub estimand_parameter: String,
    pub source: Option<Source>,
    pub model: ArtifactRef,
    pub data: ArtifactRef,
    pub decisions: Vec<Decision>,
    pub recipes: Vec<Recipe>,
    pub executions: Vec<Execution>,
    pub evidence: Vec<EvidenceSelection>,
    pub interpretation: String,
    pub unresolved_questions: Vec<String>,
}

impl Manifest {
    pub fn parse_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() > MAX_MANIFEST_BYTES {
            return Err(malformed(format!(
                "manifest exceeds the {MAX_MANIFEST_BYTES}-byte limit"
            )));
        }
        let text =
            std::str::from_utf8(bytes).map_err(|_| malformed("manifest must be UTF-8 JSON"))?;
        let value = json::parse(text)?;
        Self::parse(&value)
    }

    pub fn parse(value: &Value) -> Result<Self, Error> {
        let context = "investigation manifest";
        checked_object(
            value,
            context,
            &[
                "investigation_snapshot",
                "question",
                "estimand",
                "source",
                "inputs",
                "decisions",
                "recipes",
                "executions",
                "evidence",
                "interpretation",
                "unresolved_questions",
            ],
            &[
                "investigation_snapshot",
                "question",
                "estimand",
                "source",
                "inputs",
                "decisions",
                "recipes",
                "executions",
                "evidence",
                "interpretation",
                "unresolved_questions",
            ],
        )?;
        if required_str(value, "investigation_snapshot", context)? != SNAPSHOT_FORMAT {
            return Err(malformed(format!(
                "unsupported investigation_snapshot version; expected {SNAPSHOT_FORMAT:?}"
            )));
        }
        let estimand = value.get("estimand").expect("required");
        checked_object(
            estimand,
            "investigation manifest estimand",
            &["description", "parameter"],
            &["description", "parameter"],
        )?;
        let source = match value.get("source").expect("required") {
            Value::Null => None,
            source => Some(Source::parse(source, "investigation manifest source")?),
        };
        let inputs = value.get("inputs").expect("required");
        checked_object(
            inputs,
            "investigation manifest inputs",
            &["model", "data"],
            &["model", "data"],
        )?;
        let model = ArtifactRef::parse(
            inputs.get("model").expect("required"),
            "investigation manifest inputs.model",
        )?;
        let data = ArtifactRef::parse(
            inputs.get("data").expect("required"),
            "investigation manifest inputs.data",
        )?;
        if model.kind != ArtifactKind::ModelIr || data.kind != ArtifactKind::Data {
            return Err(malformed(
                "investigation manifest input kinds must be model_ir and data",
            ));
        }
        let decisions = parse_record_array(value, "decisions", context, Decision::parse)?;
        let recipes = parse_record_array(value, "recipes", context, Recipe::parse)?;
        let executions = parse_record_array(value, "executions", context, Execution::parse)?;
        let evidence = parse_record_array(value, "evidence", context, EvidenceSelection::parse)?;
        let unresolved_questions = required_array(value, "unresolved_questions", context)?
            .iter()
            .enumerate()
            .map(|(index, question)| {
                question
                    .as_str()
                    .ok_or_else(|| {
                        malformed(format!(
                            "manifest unresolved_questions[{index}] must be a string"
                        ))
                    })
                    .and_then(|question| bounded_text(question, "unresolved question"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let manifest = Self {
            question: bounded_text(required_str(value, "question", context)?, "question")?,
            estimand_description: bounded_text(
                required_str(estimand, "description", "estimand")?,
                "estimand description",
            )?,
            estimand_parameter: validate_identifier(
                required_str(estimand, "parameter", "estimand")?,
                "estimand parameter",
            )?,
            source,
            model,
            data,
            decisions,
            recipes,
            executions,
            evidence,
            interpretation: bounded_text(
                required_str(value, "interpretation", context)?,
                "interpretation",
            )?,
            unresolved_questions,
        };
        manifest.validate_relationships()?;
        Ok(manifest)
    }

    fn validate_relationships(&self) -> Result<(), Error> {
        let mut decision_ids = HashSet::new();
        for decision in &self.decisions {
            if decision_ids.contains(decision.id.as_str()) {
                return Err(malformed(format!(
                    "duplicate decision id {:?}",
                    decision.id
                )));
            }
            if let Some(parent) = &decision.parent {
                if !decision_ids.contains(parent.as_str()) {
                    return Err(malformed(format!(
                        "decision {:?} parent {:?} must name an earlier local decision",
                        decision.id, parent
                    )));
                }
            }
            decision_ids.insert(decision.id.as_str());
        }
        let mut recipes = HashMap::new();
        let mut recipe_digests = HashSet::new();
        for recipe in &self.recipes {
            if recipes.insert(recipe.id.as_str(), recipe).is_some() {
                return Err(malformed(format!("duplicate recipe id {:?}", recipe.id)));
            }
            if !recipe_digests.insert(recipe.sha256.as_str()) {
                return Err(malformed(format!(
                    "duplicate recipe identity {}",
                    recipe.sha256.as_str()
                )));
            }
        }
        let mut executions = HashMap::new();
        for execution in &self.executions {
            if executions
                .insert(execution.id.as_str(), execution)
                .is_some()
            {
                return Err(malformed(format!(
                    "duplicate execution id {:?}",
                    execution.id
                )));
            }
            let recipe = recipes.get(execution.recipe.as_str()).ok_or_else(|| {
                malformed(format!(
                    "execution {:?} references missing recipe {:?}",
                    execution.id, execution.recipe
                ))
            })?;
            if execution.recipe_sha256 != recipe.sha256 {
                return Err(malformed(format!(
                    "execution {:?} recipe_sha256 does not match recipe {:?}",
                    execution.id, execution.recipe
                )));
            }
            if let Some(output) = &execution.output {
                if output.kind != recipe.operation.output_kind() {
                    return Err(malformed(format!(
                        "execution {:?} output kind must be {:?}",
                        execution.id,
                        recipe.operation.output_kind().as_str()
                    )));
                }
            }
        }
        let mut evidence_names = HashSet::new();
        let mut current_operations = HashSet::new();
        let mut current_fit_hashes = HashSet::new();
        for selection in &self.evidence {
            if !evidence_names.insert(selection.name.as_str()) {
                return Err(malformed(format!(
                    "duplicate evidence selection name {:?}",
                    selection.name
                )));
            }
            let execution = executions
                .get(selection.execution.as_str())
                .ok_or_else(|| {
                    malformed(format!(
                        "evidence {:?} references missing execution {:?}",
                        selection.name, selection.execution
                    ))
                })?;
            if selection.status == EvidenceStatus::Current {
                if execution.outcome != Outcome::Completed {
                    return Err(malformed(format!(
                        "current evidence {:?} must reference a completed execution",
                        selection.name
                    )));
                }
                let recipe = recipes
                    .get(execution.recipe.as_str())
                    .expect("execution recipe validated");
                if recipe.model != self.model || recipe.data != self.data {
                    return Err(malformed(format!(
                        "current evidence {:?} recipe inputs do not match snapshot model/data",
                        selection.name
                    )));
                }
                if !current_operations.insert(recipe.operation) {
                    return Err(malformed(format!(
                        "conflicting current evidence for operation {:?}",
                        recipe.operation.as_str()
                    )));
                }
                if recipe.operation == Operation::Sample {
                    current_fit_hashes.insert(
                        execution
                            .output
                            .as_ref()
                            .expect("completed output")
                            .sha256
                            .as_str(),
                    );
                }
            }
        }
        for selection in self
            .evidence
            .iter()
            .filter(|selection| selection.status == EvidenceStatus::Current)
        {
            let execution = executions
                .get(selection.execution.as_str())
                .expect("selection execution validated");
            let recipe = recipes
                .get(execution.recipe.as_str())
                .expect("execution recipe validated");
            if let Some(fit) = &recipe.fit {
                if !current_fit_hashes.contains(fit.sha256.as_str()) {
                    return Err(malformed(format!(
                        "current evidence {:?} uses a fit that is not selected as current sample evidence",
                        selection.name
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn to_value(&self) -> Value {
        Value::Object(vec![
            ("investigation_snapshot".into(), string(SNAPSHOT_FORMAT)),
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
            (
                "inputs".into(),
                Value::Object(vec![
                    ("model".into(), self.model.to_value()),
                    ("data".into(), self.data.to_value()),
                ]),
            ),
            (
                "decisions".into(),
                Value::Array(self.decisions.iter().map(Decision::to_value).collect()),
            ),
            (
                "recipes".into(),
                Value::Array(self.recipes.iter().map(Recipe::to_value).collect()),
            ),
            (
                "executions".into(),
                Value::Array(self.executions.iter().map(Execution::to_value).collect()),
            ),
            (
                "evidence".into(),
                Value::Array(
                    self.evidence
                        .iter()
                        .map(EvidenceSelection::to_value)
                        .collect(),
                ),
            ),
            ("interpretation".into(), string(&self.interpretation)),
            (
                "unresolved_questions".into(),
                Value::Array(self.unresolved_questions.iter().map(string).collect()),
            ),
        ])
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        Ok(json::write(&self.to_value())?.into_bytes())
    }

    pub fn direct_references(&self) -> Vec<&ArtifactRef> {
        let mut out = vec![&self.model, &self.data];
        if let Some(source) = &self.source {
            out.push(&source.manifest);
        }
        for decision in &self.decisions {
            out.extend([&decision.inputs.model, &decision.inputs.data]);
        }
        for recipe in &self.recipes {
            out.extend([&recipe.model, &recipe.data]);
            if let Some(fit) = &recipe.fit {
                out.push(fit);
            }
            out.extend([&recipe.engine.binary, &recipe.engine.capabilities]);
        }
        for execution in &self.executions {
            if let Some(output) = &execution.output {
                out.push(output);
            }
        }
        out
    }

    pub fn decision(&self, id: &str) -> Option<&Decision> {
        self.decisions.iter().find(|decision| decision.id == id)
    }
}
