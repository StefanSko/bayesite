# AGENTS.md

## Project identity

This directory is the seed for **Bayesite**: a single-static-binary,
agent-operable, SQLite-like Bayesian workflow engine for serialized model IR.

Bayesite consumes a stable, code-free IR and provides the command-line workflow
an agent can use without Python, `uvx`, NumPy, or a dependency graph on the
execution path. It is not a model declaration frontend, plotting toolkit, or
multi-algorithm playground.

Current seed compatibility:

- Source branch: `claude/confident-heisenberg-js7ltl`
- Rust seed commit before extraction context: `561d4a9`
- Compatible IR source: `bayeswire` at the commit recorded in `BAYESWIRE_TAG`
  (spec and corpus vendored byte-identically; see `bayeswire-vendor.json`)
- Current wire envelope: `{"bayeswire_ir": 1, "model": ...}`

The neutral-envelope rename this section once reserved as a deliberate format
decision *was made*, deliberately, when the wire format moved to its own
repository: the envelope became `{"bayeswire_ir": 1, "model": ...}` with no
other encoding change (see the changelog in the bayeswire spec,
`spec/ir-format-v1.md`, vendored here as `docs/ir-format-v1.md`). The retired
`jaxstanv5_ir` key is not accepted; documents carrying it fail with the
standard unsupported-version error. Any future envelope change remains a
versioned format decision owned by bayeswire.

## Scope invariants

- Consume IR; do not add a Rust model declaration language unless explicitly
  redesigned.
- Provide an agent-operable CLI workflow: `sample`, `diagnose`,
  `prior-predictive`, `recover`, and `sbc`.
- Keep the default agent path to one downloaded binary: no Python, no package
  manager, no NumPy question, and no runtime dependency graph.
- Run NUTS only unless there is an explicit design decision to expand scope.
- Keep the Rust core embeddable and SQLite-like: small, auditable, deterministic,
  offline-capable, and suitable for sandboxed use.
- Keep the core zero-dependency unless a dependency is justified by a written
  design decision. `cargo tree` showing only the crate itself is intentional.
- Treat WebAssembly as first-class. A wasm build failure is a project failure.
- Keep browser/demo concerns out of core semantics. CLI and wasm are thin shells
  around the same pure runtime.

## IR compatibility invariants

Read these before changing the decoder, evaluator, or fixtures:

- `docs/ir-format-v1.md` (vendored from bayeswire — normative copy lives there)
- `docs/ir-v1-tags.md` (vendored from bayeswire)
- `docs/invariants.md`
- `tests/golden_ir/` (vendored bayeswire corpus, hash-checked)

Important IR rules:

- The serialized `ModelMeta` is the backend boundary.
- Decoding must execute no producer/user code.
- Node tags and field lists are the wire contract, not producer class names.
- Entry-array order is semantic and must never be reordered.
- `free_values` defines the flat unconstrained NUTS state layout.
- `stochastic_sites` defines log-density factors and value expressions.
- `data` plus `observed_nodes` define required bind inputs.
- Consumers hash received canonical bytes; do not reserialize just to hash.
- Unknown non-core tags fail explicitly with `UnknownNodeTag`.

## Workflow CLI target

The endgame CLI is a single binary with machine-readable commands:

```sh
bayesite sample           --model model.json --data data.json --out fit.jsonl
bayesite diagnose         --fit fit.jsonl
bayesite prior-predictive --model model.json --data data.json --out pp.jsonl
bayesite recover          --model model.json --scenario scenario.json
bayesite sbc              --model model.json --scenario scenario.json --replicates 100
```

The workflow layer may orchestrate artifacts, but it must not blur core runtime
phases or hide state. Standard output/error should remain stable,
machine-readable, and repair-oriented for agents.

## Runtime architecture

Prefer explicit phase boundaries:

1. parse JSON
2. decode IR
3. bind data
4. build posterior/evaluation state
5. evaluate log density and gradient
6. run NUTS
7. emit diagnostics/draws

The library core should remain a pure runtime:

- no hidden filesystem access
- no hidden clock/entropy access
- no global mutable sampler state
- explicit seeds
- typed errors with repair-oriented messages
- deterministic behavior for fixed inputs and seeds

`#![deny(unsafe_code)]` should remain crate-wide. The wasm ABI is the only
allowed exception and should only move bytes across the boundary.

## Development style

- Be precise and brief in notes and errors.
- Prefer small modules with explicit responsibilities.
- Prefer typed enums/structs over loose maps or stringly state.
- Make invalid states hard to represent.
- Avoid speculative abstractions and plugin surfaces.
- Follow the validation ladder in `docs/validation-ladder.md`: start from
  committed IR fixtures, prove decode/protocol behavior, then logp/gradient
  parity, transform/layout checks, sampler mechanics, analytic targets, and
  only then optional oracle comparisons (`bayesjax`, CmdStan, SBC reports).

## Changelog and releases

- `CHANGELOG.md` is the curated user-facing history. Every user-visible PR
  updates its `Unreleased` section, or explains in the PR why no entry is
  needed. Internal implementation notes are not a substitute.
- Release preparation moves `Unreleased` entries into one dated version
  section, updates both `crates/core/Cargo.toml` and `Cargo.lock`, and tags the
  reviewed `main` commit only after the full validation ladder passes.
- The complete procedure is in [`docs/releasing.md`](docs/releasing.md).
  `CLAUDE.md` is a symlink to this file so both agent entry points carry the
  same rules.

## Validation

From this repository:

```sh
python3 scripts/check_validation_ladder.py
```

Equivalent cargo gates remain in `just check-cargo`.

Optional cross-backend checks use the exact `bayesjax` release pinned in
`scripts/check_rust_backend_posterior.py` and must not become part of the
default agent path:

```sh
python3 scripts/check_validation_ladder.py --posterior
```

Pass `--bayescycle-path ../bayescycle` only to test an unpublished checkout.
