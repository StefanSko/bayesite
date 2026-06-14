# Bayesite — zero-dependency Bayesian workflow binary

Bayesite is the seed of a single-static-binary, agent-operable Bayesian
workflow engine for serialized model IR: the SQLite-like artifact an agent can
run from the CLI without Python, `uvx`, NumPy, or a runtime dependency graph.

Current implementation status: the Rust core consumes `jaxstanv5_ir` v1
fixtures and implements posterior sampling through `bayesite sample`. The
endgame command surface is:

```sh
bayesite sample
bayesite diagnose
bayesite prior-predictive
bayesite recover
bayesite sbc
```

`jaxstanv5` is currently the reference Python frontend/producer for the IR.
Bayesite is the standalone runtime/workflow binary that consumes that IR.

`cargo tree` for the core should show this crate and nothing else. The JSON
parser, PRNG, special functions, autodiff, sampler, diagnostics, CLI protocol,
and wasm glue are intentionally in-tree and reviewable.

## What it does today

1. Parses the versioned IR v1 wire format
   ([docs/ir-format-v1.md](docs/ir-format-v1.md)) — the **core profile** only.
   Tags outside [docs/ir-v1-tags.md](docs/ir-v1-tags.md) fail with
   `UnknownNodeTag`, by design.
2. Binds concrete data and evaluates the log density and its gradient with its
   own reverse-mode AD over the closed IR op set.
3. Samples with multinomial NUTS (generalized U-turn criterion with
   across-subtree checks) and Stan-style warmup adaptation
   ([docs/sampler.md](docs/sampler.md)).
4. Emits draws and diagnostics in the **v0-provisional** NDJSON protocol via the
   `bayesite` CLI or the wasm ABI.

## Correctness contract

- **Evaluation:** every fixture in `tests/golden_ir/fixtures/` must reproduce
  the committed JAX log density within rtol 1e-12 and the gradient within rtol
  1e-10 (`crates/core/tests/fixtures_eval.rs`).
- **Special functions:** pinned to a committed 400-digit mpmath table
  (`crates/core/tests/data/special_fn_table.json`, generator
  `scripts/generate_special_fn_table.py`).
- **PRNG:** splitmix64 / xoshiro256++ pinned to Vigna's reference outputs.
- **Diagnostics:** split R-hat and ESS match `blackjax.diagnostics` value for
  value against a committed fixture
  (`crates/core/tests/data/diagnostics_fixture.json`).
- **Sampler:** fixed-seed statistical tests against analytic targets
  (`crates/core/tests/sampler_stats.rs`) plus optional cross-backend posterior
  comparison over the golden corpus (`scripts/check_rust_backend_posterior.py`).

Draws are never bit-identical to BlackJAX (different RNGs); equivalence is
logp/grad parity at fixed points plus statistical agreement.

## Layout

| Path | Contents |
|---|---|
| `crates/core/src/json.rs` | strict order-preserving JSON parser/writer |
| `crates/core/src/ir.rs` | IR v1 core-profile decoder, typed errors |
| `crates/core/src/tensor.rs` | f64 tensors, broadcasting, gather maps |
| `crates/core/src/tape.rs` | reverse-mode AD over the closed op set |
| `crates/core/src/density.rs` | distribution log densities, mirrored against the Python reference |
| `crates/core/src/model.rs` | data binding, constraints, `Posterior::logp_grad` |
| `crates/core/src/special.rs` | gammaln/digamma (Lanczos), erf/erfc/ndtr/ndtri (Cephes ports) |
| `crates/core/src/linalg.rs` | Cholesky, triangular solves |
| `crates/core/src/rng.rs` | splitmix64, xoshiro256++, polar normals |
| `crates/core/src/nuts.rs`, `adapt.rs`, `sampler.rs` | NUTS, warmup adaptation, chain orchestration |
| `crates/core/src/diagnostics.rs` | split R-hat, ESS |
| `crates/core/src/protocol.rs` | v0-provisional NDJSON, wasm request handler |
| `crates/core/src/wasm_abi.rs` | the only `unsafe` module: pointer/length shims |
| `crates/core/src/bin/bayesite.rs` | CLI, currently `sample` |
| `demo/` | static browser demo, no bundler or third-party code |

The library is a **pure runtime**: no threads, no filesystem, no clock, no OS
entropy. Seeds are explicit arguments; parallelism belongs to callers (CLI
threads, web workers). This is what makes the wasm target free.

`#![deny(unsafe_code)]` is crate-wide; `wasm_abi.rs` is the single allowed
exception and only moves bytes.

## Building and testing

```sh
python3 scripts/check_validation_ladder.py  # self-contained default gates
just check                                  # same, if just is installed
just wasm-release                           # optimized wasm artifact
```

The validation ladder is documented in
[`docs/validation-ladder.md`](docs/validation-ladder.md). The wasm target
(`rustup target add wasm32-unknown-unknown`) is a build gate from day one: the
build breaking on wasm is a bug.

## CLI

```sh
cargo build --release --bin bayesite
./target/release/bayesite sample \
    --model model_ir.json --data data.json \
    --seed 1 --chains 4 --warmup 1000 --draws 1000
```

stdout is NDJSON: a header (`"draws_format": "v0-provisional"`, parameter
shapes, packing order, settings), one object per draw with constrained values
keyed by parameter, and a trailer with per-chain divergences, tree-depth
histograms, step sizes, and cross-chain R-hat/ESS. Errors are a single JSON
object on stderr (`{"error": "<Kind>", "message": ...}`) with a nonzero exit
code; messages are written as repair instructions.

The `v0-provisional` marker is mandatory: the real fit-artifact format is
defined elsewhere, and nothing may grow load-bearing dependencies on this one
without noticing.

## Browser demo

```sh
just wasm-release demo-assets
just demo   # serves the repo root on http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000/demo/>. The page loads a golden-corpus model, runs
one chain per web worker, and shows posterior summaries, R-hat/ESS (computed by
the same Rust core through the wasm ABI), divergences, tree depths, and
histograms. The only server involved is a static file server.

## Provenance

Ported numerical routines (Cephes, Lanczos coefficients, xoshiro) are documented
in [NOTICE](NOTICE) and at each site.
