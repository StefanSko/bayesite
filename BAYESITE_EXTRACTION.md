# Bayesite extraction provenance

This repository was seeded from the experimental Rust branch of `jaxstanv5` and
normalized into a standalone project named **Bayesite**.

Bayesite's endgame is a single-static-binary, agent-operable Bayesian workflow
engine for serialized model IR:

```sh
bayesite sample
bayesite diagnose
bayesite prior-predictive
bayesite recover
bayesite sbc
```

The default agent path should remain one downloaded binary: no Python, no `uvx`,
no NumPy question, and no runtime dependency graph.

## Historical source refs

- `jaxstanv5/main`: `1ecff85` (`Merge PR #41 IR serialization`)
- Rust seed branch: `claude/confident-heisenberg-js7ltl`
- Branch context commit before extraction: `60d0684`

Current IR v1 envelope remains:

```json
{"jaxstanv5_ir": 1, "model": {}}
```

Keep that for v1. Renaming the envelope to Bayesite is a versioned format
decision, not a repository cleanup.

## Context preserved here

- `AGENTS.md` — Bayesite project instructions and invariants.
- `README.md` — runtime/workflow overview.
- `NOTICE` — provenance for ported numerical routines.
- `docs/invariants.md` — Bayesite runtime/workflow invariants.
- `docs/ir-format-v1.md` — current IR v1 wire format.
- `docs/ir-v1-tags.md` — built-in core-profile tag inventory.
- `docs/sampler.md` — NUTS/adaptation behavior notes.
- `tests/golden_ir/` — compatibility fixtures and hashes.
- `crates/core/tests/data/` — Rust-owned numeric/diagnostic fixtures.
- `demo/` — first-class wasm/browser proof path.
- `justfile` — validation entry points.
- `scripts/generate_special_fn_table.py` — optional fixture regeneration tool.
- `scripts/check_rust_backend_posterior.py` — optional cross-backend conformance
  tool against a pinned `jaxstanv5` checkout.

## Spec ownership policy

Do not create a separate spec repository immediately unless it has an owner and
release process. Until then:

- `jaxstanv5` is the canonical IR v1 producer and source of IR decisions.
- Bayesite vendors explicit snapshots of the IR docs and fixtures.
- Syncs should be visible commits: `Sync IR v1 fixtures from jaxstanv5 <commit>`.
- A future neutral spec project makes sense when IR v2 is designed, another
  producer appears, or compatibility matrices become release-critical.

## Runtime/workflow boundary

Keep a hard boundary between the runtime and workflow layers:

```text
core runtime:
  decode IR -> bind data -> logp/grad -> NUTS -> diagnostics

workflow CLI:
  sample -> diagnose -> prior-predictive -> recover -> sbc
```

The workflow CLI may own artifacts and command ergonomics. It must not pollute
core evaluation/sampling semantics.

## Validation

```sh
just check
```

Equivalent cargo gates:

```sh
cargo fmt --check --manifest-path crates/core/Cargo.toml
cargo clippy --all-targets --manifest-path crates/core/Cargo.toml -- -D warnings
cargo test --manifest-path crates/core/Cargo.toml
cargo build --target wasm32-unknown-unknown --manifest-path crates/core/Cargo.toml
```
