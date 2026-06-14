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
- Compatible jaxstanv5 IR source: `main` at `1ecff85`
- Current wire envelope: `{"jaxstanv5_ir": 1, "model": ...}`

Do not rename the v1 wire envelope casually. A neutral or Bayesite-branded
envelope is a deliberate v2 format decision.

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

- `docs/ir-format-v1.md`
- `docs/ir-v1-tags.md`
- `docs/invariants.md`
- `tests/golden_ir/`

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
- Add vertical tests against fixtures before broad refactors.

## Validation

From this repository:

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

Optional cross-backend checks against the Python producer should use a pinned
`jaxstanv5` checkout and must not become part of the default agent path.
