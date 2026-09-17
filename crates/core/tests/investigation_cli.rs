use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::investigation::identity::{artifact_digest, snapshot_digest};
use bayesite_core::json::{self, Value};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> std::path::PathBuf {
    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "bayesite-investigation-{label}-{}-{id}",
        std::process::id()
    ))
}

fn example(name: &str) -> String {
    format!(
        "{}/../../examples/investigation-counts/{name}",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_bayesite"))
        .args(args)
        .output()
        .expect("bayesite starts")
}

fn success(args: &[&str]) -> Value {
    let output = run(args);
    assert!(
        output.status.success(),
        "args={args:?}\nstderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap()
}

fn object_entry_mut<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
    match value {
        Value::Object(entries) => entries
            .iter_mut()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value)
            .unwrap_or_else(|| panic!("missing {key}")),
        _ => panic!("expected object"),
    }
}

fn add_continuation_records(workspace: &std::path::Path) {
    let path = workspace.join("investigation.json");
    let mut value = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    match object_entry_mut(&mut value, "decisions") {
        Value::Array(decisions) => decisions.push(
            json::parse(
                r#"{
                    "id":"alternative-likelihood",
                    "parent":"initial-likelihood",
                    "reason":"The retained check shows dispersion and tail behavior that the Poisson likelihood does not reproduce, so test a likelihood with a separate positive overdispersion parameter while keeping the estimand fixed.",
                    "cites":["model"],
                    "kind":"note"
                }"#,
            )
            .unwrap(),
        ),
        _ => panic!("decisions array"),
    }
    *object_entry_mut(&mut value, "recipes") = json::parse(
        r#"[
            {"id":"inspect-alternative","operation":"inspect","settings":{}},
            {"id":"sample-alternative","operation":"sample","settings":{"chains":1,"warmup":20,"draws":8,"max_treedepth":6,"target_accept":0.85,"initial_step_size":1.0,"seed":20260918}},
            {"id":"diagnose-alternative","operation":"diagnose","settings":{}},
            {"id":"check-alternative","operation":"posterior-check","settings":{"seed":20260919}}
        ]"#,
    )
    .unwrap();
    *object_entry_mut(&mut value, "interpretation") = Value::Str(
        "The alternative is evaluated as a continuation; inherited Poisson evidence remains historical and no automatic scientific ranking is asserted."
            .into(),
    );
    std::fs::write(path, format!("{}\n", json::write(&value).unwrap())).unwrap();
}

fn expected_rust_target() -> &'static str {
    if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        "aarch64-apple-darwin"
    } else if cfg!(all(target_arch = "x86_64", target_os = "macos")) {
        "x86_64-apple-darwin"
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "musl"
    )) {
        "x86_64-unknown-linux-musl"
    } else if cfg!(all(
        target_arch = "x86_64",
        target_os = "linux",
        target_env = "gnu"
    )) {
        "x86_64-unknown-linux-gnu"
    } else {
        panic!("add the current test target to expected_rust_target")
    }
}

fn initialize(workspace: &std::path::Path) {
    success(&[
        "investigation",
        "init",
        "--metadata",
        &example("metadata.json"),
        "--model",
        &example("poisson.json"),
        "--data",
        &example("data.json"),
        "--out",
        workspace.to_str().unwrap(),
    ]);
}

fn init_and_run_original(workspace: &std::path::Path) {
    initialize(workspace);
    let workspace_path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&workspace_path).unwrap()).unwrap();
    if let Value::Array(recipes) = object_entry_mut(&mut document, "recipes") {
        let sample = recipes
            .iter_mut()
            .find(|recipe| recipe.get("id").and_then(Value::as_str) == Some("sample-initial"))
            .unwrap();
        *object_entry_mut(sample, "settings") = json::parse(
            r#"{"chains":1,"warmup":20,"draws":8,"max_treedepth":6,"target_accept":0.85,"initial_step_size":1.0,"seed":20260916}"#,
        )
        .unwrap();
    }
    std::fs::write(
        &workspace_path,
        format!("{}\n", json::write(&document).unwrap()),
    )
    .unwrap();
    for recipe in [
        "inspect-initial",
        "sample-initial",
        "diagnose-initial",
        "check-initial",
    ] {
        success(&[
            "investigation",
            "run",
            workspace.to_str().unwrap(),
            "--recipe",
            recipe,
        ]);
    }
}

#[test]
fn workspace_engine_uses_canonical_rust_target_vocabulary() {
    let workspace = temp_dir("canonical-target");
    initialize(&workspace);
    let document =
        json::parse(&std::fs::read_to_string(workspace.join("investigation.json")).unwrap())
            .unwrap();
    assert_eq!(
        document
            .get("engine")
            .and_then(|engine| engine.get("target"))
            .and_then(Value::as_str),
        Some(expected_rust_target())
    );
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn fork_restores_inputs_bound_to_the_named_decision_not_snapshot_tip() {
    let author = temp_dir("branch-author");
    let original = temp_dir("branch-original");
    let continuation_workspace = temp_dir("branch-continuation-workspace");
    let continuation = temp_dir("branch-continuation");
    let initial_branch = temp_dir("branch-initial");
    let alternative_branch = temp_dir("branch-alternative");
    initialize(&author);
    success(&[
        "investigation",
        "snapshot",
        author.to_str().unwrap(),
        "--out",
        original.to_str().unwrap(),
    ]);
    success(&[
        "investigation",
        "fork",
        original.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        continuation_workspace.to_str().unwrap(),
    ]);
    std::fs::copy(
        example("negative-binomial.json"),
        continuation_workspace.join("inputs/model.json"),
    )
    .unwrap();
    add_continuation_records(&continuation_workspace);
    success(&[
        "investigation",
        "snapshot",
        continuation_workspace.to_str().unwrap(),
        "--out",
        continuation.to_str().unwrap(),
    ]);

    success(&[
        "investigation",
        "fork",
        continuation.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        initial_branch.to_str().unwrap(),
    ]);
    success(&[
        "investigation",
        "fork",
        continuation.to_str().unwrap(),
        "--at",
        "alternative-likelihood",
        "--out",
        alternative_branch.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(initial_branch.join("inputs/model.json")).unwrap(),
        std::fs::read(example("poisson.json")).unwrap()
    );
    assert_eq!(
        std::fs::read(alternative_branch.join("inputs/model.json")).unwrap(),
        std::fs::read(example("negative-binomial.json")).unwrap()
    );

    for path in [
        author,
        original,
        continuation_workspace,
        continuation,
        initial_branch,
        alternative_branch,
    ] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn inspecting_a_new_decision_freezes_its_branch_inputs_once() {
    let author = temp_dir("freeze-author");
    let original = temp_dir("freeze-original");
    let continuation_workspace = temp_dir("freeze-workspace");
    let continuation = temp_dir("freeze-continuation");
    let branch = temp_dir("freeze-branch");
    initialize(&author);
    success(&[
        "investigation",
        "snapshot",
        author.to_str().unwrap(),
        "--out",
        original.to_str().unwrap(),
    ]);
    success(&[
        "investigation",
        "fork",
        original.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        continuation_workspace.to_str().unwrap(),
    ]);
    let negative = std::fs::read(example("negative-binomial.json")).unwrap();
    std::fs::write(continuation_workspace.join("inputs/model.json"), &negative).unwrap();
    add_continuation_records(&continuation_workspace);
    success(&[
        "investigation",
        "inspect",
        continuation_workspace.to_str().unwrap(),
    ]);
    let mut later_tip = negative.clone();
    later_tip.extend_from_slice(b" \n");
    std::fs::write(continuation_workspace.join("inputs/model.json"), later_tip).unwrap();
    success(&[
        "investigation",
        "snapshot",
        continuation_workspace.to_str().unwrap(),
        "--out",
        continuation.to_str().unwrap(),
    ]);
    success(&[
        "investigation",
        "fork",
        continuation.to_str().unwrap(),
        "--at",
        "alternative-likelihood",
        "--out",
        branch.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(branch.join("inputs/model.json")).unwrap(),
        negative
    );
    for path in [
        author,
        original,
        continuation_workspace,
        continuation,
        branch,
    ] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn export_requires_and_copies_an_explicit_recipient_protocol() {
    let workspace = temp_dir("protocol-author");
    let bundle = temp_dir("protocol-bundle");
    let publication = temp_dir("protocol-publication");
    let protocol = temp_dir("protocol-input.md");
    initialize(&workspace);
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        bundle.to_str().unwrap(),
    ]);
    let protocol_bytes = b"# Recipient task\n\nInvestigate this specific saved question.\n";
    std::fs::write(&protocol, protocol_bytes).unwrap();

    let missing_protocol = run(&[
        "investigation",
        "export",
        bundle.to_str().unwrap(),
        "--viewer",
        "--public-data-confirmed",
        "--out",
        publication.to_str().unwrap(),
    ]);
    assert!(!missing_protocol.status.success());
    assert!(!publication.exists());
    success(&[
        "investigation",
        "export",
        bundle.to_str().unwrap(),
        "--viewer",
        "--public-data-confirmed",
        "--protocol",
        protocol.to_str().unwrap(),
        "--out",
        publication.to_str().unwrap(),
    ]);
    assert_eq!(
        std::fs::read(publication.join("PROTOCOL.md")).unwrap(),
        protocol_bytes
    );

    for path in [workspace, bundle, publication] {
        let _ = std::fs::remove_dir_all(path);
    }
    let _ = std::fs::remove_file(protocol);
}

#[test]
fn fork_and_export_copy_only_the_verified_object_closure() {
    let workspace = temp_dir("closure-author");
    let bundle = temp_dir("closure-bundle");
    let fork = temp_dir("closure-fork");
    let publication = temp_dir("closure-publication");
    let protocol = temp_dir("closure-protocol.md");
    initialize(&workspace);
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        bundle.to_str().unwrap(),
    ]);
    let objects = bundle.join("objects/sha256");
    let valid_orphan = b"unreferenced retained note";
    let valid_digest = artifact_digest(valid_orphan);
    std::fs::write(objects.join(valid_digest.as_str()), valid_orphan).unwrap();
    let corrupt_digest = artifact_digest(b"expected orphan bytes");
    std::fs::write(
        objects.join(corrupt_digest.as_str()),
        b"corrupt orphan bytes",
    )
    .unwrap();

    success(&[
        "investigation",
        "fork",
        bundle.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        fork.to_str().unwrap(),
    ]);
    std::fs::write(&protocol, b"# Recipient task\n").unwrap();
    success(&[
        "investigation",
        "export",
        bundle.to_str().unwrap(),
        "--viewer",
        "--public-data-confirmed",
        "--protocol",
        protocol.to_str().unwrap(),
        "--out",
        publication.to_str().unwrap(),
    ]);
    for root in [
        fork.join("objects/sha256"),
        publication.join("bundle/objects/sha256"),
    ] {
        assert!(!root.join(valid_digest.as_str()).exists());
        assert!(!root.join(corrupt_digest.as_str()).exists());
    }

    for path in [workspace, bundle, fork, publication] {
        let _ = std::fs::remove_dir_all(path);
    }
    let _ = std::fs::remove_file(protocol);
}

#[test]
fn author_replay_fork_continue_snapshot_keeps_source_immutable_and_stale_history_visible() {
    let workspace = temp_dir("author");
    let original = temp_dir("original");
    let replay = temp_dir("replay");
    let fork = temp_dir("fork");
    let continuation = temp_dir("continuation");
    let publication = temp_dir("publication");
    init_and_run_original(&workspace);

    let original_snapshot = success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        original.to_str().unwrap(),
    ]);
    let original_id = original_snapshot
        .get("snapshot_id")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    let original_manifest_before = std::fs::read(original.join("manifest.json")).unwrap();
    let original_objects_before = std::fs::read_dir(original.join("objects/sha256"))
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (entry.file_name(), std::fs::read(entry.path()).unwrap())
        })
        .collect::<Vec<_>>();

    let verification = success(&["investigation", "verify", original.to_str().unwrap()]);
    assert!(matches!(
        verification.get("object_integrity_valid"),
        Some(Value::Bool(true))
    ));
    let replay_report = success(&[
        "investigation",
        "replay",
        original.to_str().unwrap(),
        "--recipe",
        "sample-initial",
        "--out",
        replay.to_str().unwrap(),
    ]);
    assert!(matches!(
        replay_report.get("exact_output_bytes_agree"),
        Some(Value::Bool(true))
    ));

    success(&[
        "investigation",
        "fork",
        original.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        fork.to_str().unwrap(),
    ]);
    std::fs::copy(
        example("negative-binomial.json"),
        fork.join("inputs/model.json"),
    )
    .unwrap();
    add_continuation_records(&fork);

    let before_resampling = success(&["investigation", "inspect", fork.to_str().unwrap()]);
    let evidence = before_resampling
        .get("evidence")
        .and_then(Value::as_array)
        .unwrap();
    assert!(!evidence.is_empty());
    assert!(evidence.iter().all(|entry| {
        entry.get("status").and_then(Value::as_str) == Some("historical")
            && entry.get("origin").and_then(Value::as_str) == Some("source_snapshot")
    }));
    assert!(before_resampling.get("current_fit_sha256") == Some(&Value::Null));

    for recipe in [
        "inspect-alternative",
        "sample-alternative",
        "diagnose-alternative",
        "check-alternative",
    ] {
        success(&[
            "investigation",
            "run",
            fork.to_str().unwrap(),
            "--recipe",
            recipe,
        ]);
    }
    let continuation_snapshot = success(&[
        "investigation",
        "snapshot",
        fork.to_str().unwrap(),
        "--out",
        continuation.to_str().unwrap(),
    ]);
    assert_ne!(
        continuation_snapshot
            .get("snapshot_id")
            .and_then(Value::as_str),
        Some(original_id.as_str())
    );
    let continuation_verification =
        success(&["investigation", "verify", continuation.to_str().unwrap()]);
    assert_eq!(
        continuation_verification
            .get("ancestry_depth")
            .and_then(Value::as_i64),
        Some(1)
    );

    let denied_export = run(&[
        "investigation",
        "export",
        continuation.to_str().unwrap(),
        "--viewer",
        "--protocol",
        &example("PROTOCOL.md"),
        "--out",
        publication.to_str().unwrap(),
    ]);
    assert!(!denied_export.status.success());
    assert!(!publication.exists());
    let continuation_manifest_before = std::fs::read(continuation.join("manifest.json")).unwrap();
    success(&[
        "investigation",
        "export",
        continuation.to_str().unwrap(),
        "--viewer",
        "--public-data-confirmed",
        "--protocol",
        &example("PROTOCOL.md"),
        "--out",
        publication.to_str().unwrap(),
    ]);
    for name in [
        "index.html",
        "viewer.js",
        "style.css",
        "entry.json",
        "LICENSE",
        "NOTICE",
        "PROTOCOL.md",
        "CONTINUING.md",
        "IR-FORMAT.md",
        "IR-TAGS.md",
        "INSPECTION.md",
        "PUBLIC-DATA-CONFIRMATION.txt",
        "downloads/bayesite-engine",
        "bundle/manifest.json",
    ] {
        assert!(publication.join(name).is_file(), "missing exported {name}");
    }
    assert_eq!(
        std::fs::read(publication.join("bundle/manifest.json")).unwrap(),
        continuation_manifest_before
    );
    assert_eq!(
        std::fs::read(continuation.join("manifest.json")).unwrap(),
        continuation_manifest_before
    );
    let viewer_source = std::fs::read_to_string(publication.join("viewer.js")).unwrap();
    assert!(!viewer_source.contains("innerHTML"));
    assert!(!viewer_source.contains("eval("));
    assert!(!viewer_source.contains("new Function"));
    assert!(!viewer_source.contains("http://"));
    assert!(!viewer_source.contains("https://"));
    let existing_export = run(&[
        "investigation",
        "export",
        continuation.to_str().unwrap(),
        "--viewer",
        "--public-data-confirmed",
        "--protocol",
        &example("PROTOCOL.md"),
        "--out",
        publication.to_str().unwrap(),
    ]);
    assert!(!existing_export.status.success());

    assert_eq!(
        std::fs::read(original.join("manifest.json")).unwrap(),
        original_manifest_before
    );
    for (name, bytes) in original_objects_before {
        assert_eq!(
            std::fs::read(original.join("objects/sha256").join(name)).unwrap(),
            bytes
        );
    }

    for path in [workspace, original, replay, fork, continuation, publication] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn snapshot_of_unrun_continuation_records_incomplete_without_current_result() {
    let workspace = temp_dir("incomplete-author");
    let original = temp_dir("incomplete-original");
    let fork = temp_dir("incomplete-fork");
    let incomplete = temp_dir("incomplete-snapshot");
    init_and_run_original(&workspace);
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        original.to_str().unwrap(),
    ]);
    success(&[
        "investigation",
        "fork",
        original.to_str().unwrap(),
        "--at",
        "initial-likelihood",
        "--out",
        fork.to_str().unwrap(),
    ]);
    std::fs::copy(
        example("negative-binomial.json"),
        fork.join("inputs/model.json"),
    )
    .unwrap();
    add_continuation_records(&fork);
    success(&[
        "investigation",
        "snapshot",
        fork.to_str().unwrap(),
        "--out",
        incomplete.to_str().unwrap(),
    ]);
    success(&["investigation", "verify", incomplete.to_str().unwrap()]);
    let manifest =
        json::parse(&std::fs::read_to_string(incomplete.join("manifest.json")).unwrap()).unwrap();
    let executions = manifest
        .get("executions")
        .and_then(Value::as_array)
        .unwrap();
    assert!(executions.iter().any(|execution| {
        execution.get("outcome").and_then(Value::as_str) == Some("incomplete")
            && execution.get("recipe").and_then(Value::as_str) == Some("sample-alternative")
    }));
    assert!(manifest
        .get("evidence")
        .and_then(Value::as_array)
        .unwrap()
        .is_empty());

    for path in [workspace, original, fork, incomplete] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn snapshot_preserves_reason_only_citation_after_unrun_model_edit() {
    let workspace = temp_dir("citation-author");
    let bundle = temp_dir("citation-bundle");
    initialize(&workspace);
    let original_model = std::fs::read(example("poisson.json")).unwrap();
    let original_digest = bayesite_core::fingerprint::sha256_bytes(&original_model)
        .trim_start_matches("sha256:")
        .to_string();
    std::fs::copy(
        example("negative-binomial.json"),
        workspace.join("inputs/model.json"),
    )
    .unwrap();
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        bundle.to_str().unwrap(),
    ]);
    success(&["investigation", "verify", bundle.to_str().unwrap()]);
    assert_eq!(
        std::fs::read(bundle.join("objects/sha256").join(original_digest)).unwrap(),
        original_model
    );
    for path in [workspace, bundle] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn additional_sample_needs_explicit_selection_and_seed_substitution_is_rejected() {
    let workspace = temp_dir("selection-author");
    let bundle = temp_dir("seed-bundle");
    initialize(&workspace);
    let path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    *object_entry_mut(&mut document, "recipes") = json::parse(
        r#"[
            {"id":"sample-a","operation":"sample","settings":{"chains":1,"warmup":8,"draws":4,"max_treedepth":4,"target_accept":0.8,"initial_step_size":1.0,"seed":100}},
            {"id":"sample-b","operation":"sample","settings":{"chains":1,"warmup":8,"draws":4,"max_treedepth":4,"target_accept":0.8,"initial_step_size":1.0,"seed":101}}
        ]"#,
    )
    .unwrap();
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();

    let first = success(&[
        "investigation",
        "run",
        workspace.to_str().unwrap(),
        "--recipe",
        "sample-a",
    ]);
    let second = success(&[
        "investigation",
        "run",
        workspace.to_str().unwrap(),
        "--recipe",
        "sample-b",
    ]);
    assert_eq!(first.get("selected"), Some(&Value::Bool(true)));
    assert_eq!(second.get("selected"), Some(&Value::Bool(false)));
    let inspection = success(&["investigation", "inspect", workspace.to_str().unwrap()]);
    assert_eq!(
        inspection
            .get("evidence")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter(|item| item.get("status").and_then(Value::as_str) == Some("current"))
            .count(),
        1
    );

    let second_execution = second
        .get("execution")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    if let Value::Array(selections) = object_entry_mut(&mut document, "selections") {
        selections.push(
            json::parse(&format!(
                "{{\"name\":\"second-fit\",\"execution\":{}}}",
                json::write(&Value::Str(second_execution)).unwrap()
            ))
            .unwrap(),
        );
    }
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();
    let conflict = run(&["investigation", "inspect", workspace.to_str().unwrap()]);
    assert!(!conflict.status.success());
    assert!(
        String::from_utf8_lossy(&conflict.stderr).contains("current sample selections conflict")
    );

    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    if let Value::Array(selections) = object_entry_mut(&mut document, "selections") {
        selections.pop();
    }
    if let Value::Array(attempts) = object_entry_mut(&mut document, "attempts") {
        let replacement = attempts
            .iter()
            .find(|attempt| {
                attempt
                    .get("recipe")
                    .and_then(|recipe| recipe.get("id"))
                    .and_then(Value::as_str)
                    == Some("sample-b")
            })
            .and_then(|attempt| attempt.get("execution"))
            .and_then(|execution| execution.get("output"))
            .unwrap()
            .clone();
        let first_attempt = attempts
            .iter_mut()
            .find(|attempt| {
                attempt
                    .get("recipe")
                    .and_then(|recipe| recipe.get("id"))
                    .and_then(Value::as_str)
                    == Some("sample-a")
            })
            .unwrap();
        let execution = object_entry_mut(first_attempt, "execution");
        *object_entry_mut(execution, "output") = replacement;
    }
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();
    let publication = run(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        bundle.to_str().unwrap(),
    ]);
    assert!(!publication.status.success());
    assert!(!bundle.exists());
    let publication_error =
        json::parse(String::from_utf8(publication.stderr).unwrap().trim()).unwrap();
    assert!(publication_error
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("recipe setting \"seed\""));

    for path in [workspace, bundle] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn snapshot_rejects_seventeenth_manifest_before_creating_destination() {
    let workspace = temp_dir("ancestry-author");
    let base = temp_dir("ancestry-base");
    let rejected = temp_dir("ancestry-rejected");
    initialize(&workspace);
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        base.to_str().unwrap(),
    ]);
    let mut parent_bytes = std::fs::read(base.join("manifest.json")).unwrap();
    let object_root = workspace.join("objects/sha256");
    for _ in 1..16 {
        let parent_artifact = artifact_digest(&parent_bytes);
        std::fs::write(object_root.join(parent_artifact.as_str()), &parent_bytes).unwrap();
        let mut child = json::parse(std::str::from_utf8(&parent_bytes).unwrap()).unwrap();
        *object_entry_mut(&mut child, "source") = json::parse(&format!(
            "{{\"snapshot_id\":\"{}\",\"manifest\":{{\"sha256\":\"{}\",\"bytes\":{},\"kind\":\"investigation_manifest\",\"format\":\"investigation-snapshot-v0-provisional\"}},\"decision\":\"initial-likelihood\"}}",
            snapshot_digest(&parent_bytes).as_str(),
            parent_artifact.as_str(),
            parent_bytes.len()
        ))
        .unwrap();
        parent_bytes = json::write(&child).unwrap().into_bytes();
    }
    let parent_artifact = artifact_digest(&parent_bytes);
    std::fs::write(object_root.join(parent_artifact.as_str()), &parent_bytes).unwrap();
    let path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    *object_entry_mut(&mut document, "source") = json::parse(&format!(
        "{{\"snapshot_id\":\"{}\",\"manifest\":{{\"sha256\":\"{}\",\"bytes\":{},\"kind\":\"investigation_manifest\",\"format\":\"investigation-snapshot-v0-provisional\"}},\"decision\":\"initial-likelihood\"}}",
        snapshot_digest(&parent_bytes).as_str(),
        parent_artifact.as_str(),
        parent_bytes.len()
    ))
    .unwrap();
    std::fs::write(path, format!("{}\n", json::write(&document).unwrap())).unwrap();

    let output = run(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        rejected.to_str().unwrap(),
    ]);
    assert!(!output.status.success());
    assert!(!rejected.exists());
    assert!(String::from_utf8_lossy(&output.stderr).contains("16-manifest limit"));

    for path in [workspace, base, rejected] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn mismatched_engine_target_is_refused_before_an_attempt_is_created() {
    let workspace = temp_dir("engine-mismatch");
    initialize(&workspace);
    let path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let engine = object_entry_mut(&mut document, "engine");
    *object_entry_mut(engine, "target") = Value::Str("wrong-unknown-target".into());
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();

    let output = run(&[
        "investigation",
        "run",
        workspace.to_str().unwrap(),
        "--recipe",
        "inspect-initial",
    ]);
    assert!(!output.status.success());
    let error = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert!(error
        .get("message")
        .and_then(Value::as_str)
        .unwrap()
        .contains("running engine does not match recipe pin"));
    let document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert!(document
        .get("attempts")
        .and_then(Value::as_array)
        .unwrap()
        .is_empty());
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn reason_only_edit_keeps_unchanged_numerical_evidence_current() {
    let workspace = temp_dir("reason-edit");
    init_and_run_original(&workspace);
    let before = success(&["investigation", "inspect", workspace.to_str().unwrap()]);
    let fit_before = before
        .get("current_fit_sha256")
        .and_then(Value::as_str)
        .unwrap()
        .to_string();
    let path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let decisions = object_entry_mut(&mut document, "decisions");
    let Value::Array(decisions) = decisions else {
        panic!("decisions array")
    };
    *object_entry_mut(&mut decisions[0], "reason") =
        Value::Str("Clarified wording only; no computational input changed.".into());
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();

    let after = success(&["investigation", "inspect", workspace.to_str().unwrap()]);
    assert_eq!(
        after.get("current_fit_sha256").and_then(Value::as_str),
        Some(fit_before.as_str())
    );
    assert!(after
        .get("evidence")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .all(|evidence| evidence.get("status").and_then(Value::as_str) == Some("current")));
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn failed_sampling_retains_prior_evidence_as_historical() {
    let workspace = temp_dir("failed-sample");
    init_and_run_original(&workspace);
    let path = workspace.join("investigation.json");
    let before = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let prior_selection = before
        .get("selections")
        .and_then(Value::as_array)
        .unwrap()
        .first()
        .unwrap()
        .clone();

    let model_path = workspace.join("inputs/model.json");
    let invalid_model = std::fs::read_to_string(&model_path)
        .unwrap()
        .replace("\"value\": 0.1", "\"value\": -0.1");
    assert_ne!(
        invalid_model,
        std::fs::read_to_string(example("poisson.json")).unwrap()
    );
    std::fs::write(&model_path, invalid_model).unwrap();
    let mut document = before;
    *object_entry_mut(&mut document, "recipes") = json::parse(
        r#"[{"id":"sample-invalid-rate","operation":"sample","settings":{"chains":1,"warmup":8,"draws":4,"max_treedepth":4,"target_accept":0.8,"initial_step_size":1.0,"seed":9}}]"#,
    )
    .unwrap();
    std::fs::write(&path, format!("{}\n", json::write(&document).unwrap())).unwrap();

    let output = run(&[
        "investigation",
        "run",
        workspace.to_str().unwrap(),
        "--recipe",
        "sample-invalid-rate",
    ]);
    assert!(!output.status.success());
    let document = json::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
    assert_eq!(
        document
            .get("attempts")
            .and_then(Value::as_array)
            .unwrap()
            .last()
            .and_then(|attempt| attempt.get("execution"))
            .and_then(|execution| execution.get("outcome"))
            .and_then(Value::as_str),
        Some("failed")
    );
    assert_eq!(
        document
            .get("selections")
            .and_then(Value::as_array)
            .unwrap()
            .first(),
        Some(&prior_selection)
    );
    let inspection = success(&["investigation", "inspect", workspace.to_str().unwrap()]);
    assert!(inspection
        .get("evidence")
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .all(|evidence| evidence.get("status").and_then(Value::as_str) == Some("historical")));
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn standalone_and_wrapped_operations_emit_identical_bytes() {
    let workspace = temp_dir("wrapped-parity");
    initialize(&workspace);
    let workspace_path = workspace.join("investigation.json");
    let mut document = json::parse(&std::fs::read_to_string(&workspace_path).unwrap()).unwrap();
    let Value::Array(recipes) = object_entry_mut(&mut document, "recipes") else {
        panic!("recipes array")
    };
    let sample = recipes
        .iter_mut()
        .find(|recipe| recipe.get("id").and_then(Value::as_str) == Some("sample-initial"))
        .unwrap();
    *object_entry_mut(sample, "settings") = json::parse(
        r#"{"chains":1,"warmup":8,"draws":4,"max_treedepth":4,"target_accept":0.85,"initial_step_size":1.0,"seed":20260916}"#,
    )
    .unwrap();
    std::fs::write(
        &workspace_path,
        format!("{}\n", json::write(&document).unwrap()),
    )
    .unwrap();

    let model = workspace.join("inputs/model.json");
    let data = workspace.join("inputs/data.json");
    let fit_path = workspace.join("standalone-fit.jsonl");
    for (recipe, command) in [
        (
            "inspect-initial",
            vec![
                "inspect",
                "--model",
                model.to_str().unwrap(),
                "--data",
                data.to_str().unwrap(),
            ],
        ),
        (
            "sample-initial",
            vec![
                "sample",
                "--model",
                model.to_str().unwrap(),
                "--data",
                data.to_str().unwrap(),
                "--chains",
                "1",
                "--warmup",
                "8",
                "--draws",
                "4",
                "--max-treedepth",
                "4",
                "--target-accept",
                "0.85",
                "--seed",
                "20260916",
            ],
        ),
    ] {
        let wrapped = success(&[
            "investigation",
            "run",
            workspace.to_str().unwrap(),
            "--recipe",
            recipe,
        ]);
        let digest = wrapped
            .get("output_sha256")
            .and_then(Value::as_str)
            .unwrap()
            .trim_start_matches("sha256:");
        let wrapped_bytes = std::fs::read(workspace.join("objects/sha256").join(digest)).unwrap();
        let standalone = run(&command);
        assert!(
            standalone.status.success(),
            "{}",
            String::from_utf8_lossy(&standalone.stderr)
        );
        assert_eq!(
            standalone.stdout, wrapped_bytes,
            "byte mismatch for {recipe}"
        );
        if recipe == "sample-initial" {
            std::fs::write(&fit_path, &standalone.stdout).unwrap();
        }
    }

    for (recipe, command) in [
        (
            "diagnose-initial",
            vec!["diagnose", "--fit", fit_path.to_str().unwrap()],
        ),
        (
            "check-initial",
            vec![
                "posterior-check",
                "--model",
                model.to_str().unwrap(),
                "--data",
                data.to_str().unwrap(),
                "--fit",
                fit_path.to_str().unwrap(),
                "--seed",
                "20260917",
            ],
        ),
    ] {
        let wrapped = success(&[
            "investigation",
            "run",
            workspace.to_str().unwrap(),
            "--recipe",
            recipe,
        ]);
        let digest = wrapped
            .get("output_sha256")
            .and_then(Value::as_str)
            .unwrap()
            .trim_start_matches("sha256:");
        let wrapped_bytes = std::fs::read(workspace.join("objects/sha256").join(digest)).unwrap();
        let standalone = run(&command);
        assert!(
            standalone.status.success(),
            "{}",
            String::from_utf8_lossy(&standalone.stderr)
        );
        assert_eq!(
            standalone.stdout, wrapped_bytes,
            "byte mismatch for {recipe}"
        );
    }
    let _ = std::fs::remove_dir_all(workspace);
}

#[test]
fn verify_detects_tampered_object_and_does_not_run_engine() {
    let workspace = temp_dir("tamper-author");
    let bundle = temp_dir("tamper-bundle");
    init_and_run_original(&workspace);
    success(&[
        "investigation",
        "snapshot",
        workspace.to_str().unwrap(),
        "--out",
        bundle.to_str().unwrap(),
    ]);
    let manifest =
        json::parse(&std::fs::read_to_string(bundle.join("manifest.json")).unwrap()).unwrap();
    let model_digest = manifest
        .get("inputs")
        .and_then(|inputs| inputs.get("model"))
        .and_then(|model| model.get("sha256"))
        .and_then(Value::as_str)
        .unwrap();
    let data_digest = manifest
        .get("inputs")
        .and_then(|inputs| inputs.get("data"))
        .and_then(|data| data.get("sha256"))
        .and_then(Value::as_str)
        .unwrap();
    let object = bundle.join("objects/sha256").join(model_digest);
    let mut bytes = std::fs::read(&object).unwrap();
    bytes.push(b'!');
    std::fs::write(object, bytes).unwrap();
    std::fs::remove_file(bundle.join("objects/sha256").join(data_digest)).unwrap();
    let output = run(&["investigation", "verify", bundle.to_str().unwrap()]);
    assert!(!output.status.success());
    let report = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(
        report.get("verification_complete"),
        Some(&Value::Bool(false))
    );
    assert_eq!(report.get("schema_valid"), Some(&Value::Bool(true)));
    assert_eq!(report.get("reference_closure_valid"), Some(&Value::Null));
    assert_eq!(
        report.get("object_integrity_valid"),
        Some(&Value::Bool(false))
    );
    assert_eq!(report.get("current_results_valid"), Some(&Value::Null));
    let finding = report
        .get("findings")
        .and_then(Value::as_array)
        .and_then(|findings| findings.first())
        .unwrap();
    assert_eq!(
        finding.get("dimension").and_then(Value::as_str),
        Some("object_integrity")
    );
    let error = json::parse(String::from_utf8(output.stderr).unwrap().trim()).unwrap();
    assert_eq!(
        error.get("error_format").and_then(Value::as_str),
        Some("v0-provisional")
    );

    for path in [workspace, bundle] {
        let _ = std::fs::remove_dir_all(path);
    }
}

#[test]
fn verify_missing_manifest_still_emits_a_dimensional_report() {
    let bundle = temp_dir("missing-manifest");
    let output = run(&["investigation", "verify", bundle.to_str().unwrap()]);
    assert!(!output.status.success());
    let report = json::parse(String::from_utf8(output.stdout).unwrap().trim()).unwrap();
    assert_eq!(report.get("snapshot_id"), Some(&Value::Null));
    assert_eq!(report.get("schema_valid"), Some(&Value::Null));
    assert_eq!(
        report.get("reference_closure_valid"),
        Some(&Value::Bool(false))
    );
    assert_eq!(report.get("object_integrity_valid"), Some(&Value::Null));
    assert_eq!(report.get("current_results_valid"), Some(&Value::Null));
    let finding = report
        .get("findings")
        .and_then(Value::as_array)
        .and_then(|findings| findings.first())
        .unwrap();
    assert_eq!(
        finding.get("dimension").and_then(Value::as_str),
        Some("reference_closure")
    );
    assert!(!output.stderr.is_empty());
}
