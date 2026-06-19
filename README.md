# Bayesite

Bayesite is a zero-dependency Rust runtime for serialized Bayesian model IR.
The goal is a SQLite-like Bayesian workflow binary: one executable an agent can
run without Python, `uvx`, NumPy, or a runtime dependency graph.

Bayesite is **not** a model declaration frontend. It consumes `jaxstanv5_ir` v1;
`jaxstanv5` is currently the reference Python producer.

## Status

Implemented command surface:

```sh
bayesite sample
bayesite diagnose
bayesite prior-predictive
bayesite posterior-predictive
bayesite posterior-check
bayesite simulate
bayesite recover-check
bayesite recover
bayesite sbc
```

Current runtime capabilities:

- decodes the core `jaxstanv5_ir` v1 profile;
- evaluates log density and gradients with in-tree reverse-mode AD;
- samples posterior draws with NUTS only;
- recomputes R-hat/ESS diagnostics from fit streams;
- emits prior-predictive draws for directly assignable stochastic sites;
- emits posterior-predictive replicated observed draws;
- emits factual posterior predictive check reports;
- simulates plain data documents from fixed supplied truth;
- compares posterior draws to supplied truth with factual recovery checks;
- emits factual single-scenario recovery reports;
- emits factual SBC rank/histogram reports.

Workflow artifacts are **v0-provisional**. Field-level details are in
[`docs/artifacts-v0.md`](docs/artifacts-v0.md). The model IR remains the
separate `jaxstanv5_ir` v1 format documented in
[`docs/ir-format-v1.md`](docs/ir-format-v1.md).

## Current limitations

- Prior predictive supports directly assignable stochastic sites only.
- Posterior predictive supports directly assignable observed stochastic sites only.
- `simulate` generates directly assignable observed `DataRef` sites only.
- `posterior-check` reports built-in generic discrepancies only; no custom discrepancy language yet.
- `recover-check` compares same-shape truth/posterior targets only; use explicit
  target mappings for renamed parameters, not transformed estimands.
- `recover` is a single-scenario factual report, not repeated-scenario coverage
  validation.
- `sbc` reports ranks and histograms but no uniformity verdict or p-value.
- Workflow artifacts are not stable v1 formats yet; consumers must check the
  `v0-provisional` markers.

## Build and validate

```sh
python3 scripts/check_validation_ladder.py
```

This self-contained default gate checks:

- zero-dependency core via `cargo tree`;
- `cargo fmt --check`;
- `cargo clippy -D warnings`;
- release CLI build and JSON-error smoke test;
- Rust tests;
- wasm build.

Equivalent `just` entry point, if installed:

```sh
just check
```

Optional posterior conformance against a pinned `jaxstanv5` checkout:

```sh
python3 scripts/check_validation_ladder.py \
  --posterior \
  --jaxstanv5-path ../jaxstanv5
```

The optional oracle path may use Python/JAX/BlackJAX. It is not part of the
agent execution path.

## CLI examples

```sh
cargo build --release --bin bayesite

./target/release/bayesite sample \
  --model model_ir.json \
  --data data.json \
  --seed 1 \
  --chains 4 \
  --warmup 1000 \
  --draws 1000 \
  --out fit.jsonl

./target/release/bayesite diagnose \
  --fit fit.jsonl \
  --out diagnostics.json

./target/release/bayesite prior-predictive \
  --model model_ir.json \
  --data predictors.json \
  --seed 1 \
  --draws 1000 \
  --out pp.jsonl

./target/release/bayesite posterior-predictive \
  --model model_ir.json \
  --data observed_data.json \
  --fit fit.jsonl \
  --seed 2 \
  --out yrep.jsonl

./target/release/bayesite posterior-check \
  --model model_ir.json \
  --data observed_data.json \
  --fit fit.jsonl \
  --seed 2 \
  --out ppc.json

./target/release/bayesite simulate \
  --model generator_ir.json \
  --data fixed_inputs.json \
  --truth truth.json \
  --seed 1 \
  --out generated_data.json

./target/release/bayesite recover-check \
  --fit fit.jsonl \
  --truth truth.json \
  --targets targets.json \
  --interval 0.8 \
  --out recovery_check.json

./target/release/bayesite recover \
  --model model_ir.json \
  --scenario recover_scenario.json \
  --out recover.json

./target/release/bayesite sbc \
  --model model_ir.json \
  --scenario sbc_scenario.json \
  --replicates 100 \
  --out sbc.json
```

Successful artifact commands write machine-readable JSON/NDJSON to stdout or
`--out` and keep stderr empty. Errors are one machine-readable JSON object with
`error_format: "v0-provisional"`.

For commands with multiple input documents, at most one input path may be `-`.
Bayesite rejects ambiguous stdin use before reading.

## Correctness contract

- **IR/evaluation:** every golden fixture in `tests/golden_ir/fixtures/` must
  match committed JAX log-density and gradient values.
- **Sampler:** fixed-seed statistical tests cover analytic targets; optional
  conformance compares posterior summaries against `jaxstanv5`/BlackJAX.
- **Diagnostics:** split R-hat and ESS match a committed BlackJAX fixture.
- **Special functions:** checked against committed high-precision tables.
- **PRNG:** splitmix64 and xoshiro256++ are pinned to reference vectors.
- **Wasm:** `wasm32-unknown-unknown` build failure is a project failure.

Draws are not expected to be bit-identical to BlackJAX. Agreement is defined by
logp/gradient parity plus statistical checks.

## Project invariants

- Agent path: one binary, no Python, no package manager, no NumPy.
- Core crate: zero dependencies unless a written design decision says otherwise.
- Posterior sampler: NUTS only.
- Runtime phases stay explicit: parse JSON -> decode IR -> bind data -> build
  state -> evaluate logp/grad -> run NUTS -> emit artifact.
- CLI and wasm are thin shells around the same runtime semantics.
- The wasm ABI is the only allowed unsafe boundary.

See [`docs/invariants.md`](docs/invariants.md) and
[`docs/validation-ladder.md`](docs/validation-ladder.md).

## Repository map

| Path | Purpose |
|---|---|
| `crates/core/src/ir.rs` | IR v1 core-profile decoder |
| `crates/core/src/model.rs` | data binding and posterior state |
| `crates/core/src/tape.rs` | reverse-mode AD |
| `crates/core/src/density.rs` | distribution log densities |
| `crates/core/src/nuts.rs`, `adapt.rs`, `sampler.rs` | NUTS and warmup adaptation |
| `crates/core/src/predictive.rs` | prior/posterior-predictive simulation, fixed-truth simulation, and posterior checks |
| `crates/core/src/workflow.rs` | recover/SBC factual reports |
| `crates/core/src/protocol.rs` | v0 artifacts and wasm/native request handler |
| `crates/core/src/bin/bayesite.rs` | CLI dispatcher |
| `tests/golden_ir/` | vendored IR compatibility fixtures |
| `docs/artifacts-v0.md` | provisional workflow artifact details |

## Browser demo

```sh
just wasm-release demo-assets
just demo
```

Then open <http://127.0.0.1:8000/demo/>.

## Provenance

Ported numerical routines and reference algorithms are documented in
[`NOTICE`](NOTICE) and at their implementation sites.
