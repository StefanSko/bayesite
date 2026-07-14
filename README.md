# Bayesite

Bayesite is a zero-dependency Rust runtime for serialized Bayesian model IR.
The goal is a SQLite-like Bayesian workflow binary: one executable an agent can
run without Python, `uvx`, NumPy, or a runtime dependency graph.

Bayesite is **not** a model declaration frontend. It consumes `bayeswire_ir` v1;
[bayeswire](https://github.com/StefanSko/bayescycle/tree/main/packages/bayeswire)
owns the language and is the reference Python producer. bayeswire ships inside
the [bayescycle](https://github.com/StefanSko/bayescycle) monorepo, with the
normative wire-format spec at its repository root (`spec/`).

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

- decodes the core `bayeswire_ir` v1 profile;
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
[`docs/artifacts-v0.md`](docs/artifacts-v0.md). The model IR is the
separate `bayeswire_ir` v1 format; [`docs/ir-format-v1.md`](docs/ir-format-v1.md)
and [`docs/ir-v1-tags.md`](docs/ir-v1-tags.md) are byte-identical vendored
copies of the normative bayeswire spec, pinned by `BAYESWIRE_TAG` and
hash-checked by the validation ladder.

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

## Install released CLI binary

Released Bayesite CLI artifacts are versioned GitHub Release assets. On macOS
or x86_64 Linux, this installs a pinned release to `~/.local/bin/bayesite`
without a source checkout or Cargo on the execution path:

```sh
VERSION=v0.2.2
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) TARGET=aarch64-apple-darwin ;;
  Darwin-x86_64) TARGET=x86_64-apple-darwin ;;
  Linux-x86_64) TARGET=x86_64-unknown-linux-musl ;;
  *) echo "unsupported platform" >&2; exit 1 ;;
esac

NAME="bayesite-${VERSION}-${TARGET}"
BASE="https://github.com/StefanSko/bayesite/releases/download/${VERSION}"
TMPDIR="$(mktemp -d)"
mkdir -p "$HOME/.local/bin"
cd "$TMPDIR"
curl -fsSLO "${BASE}/${NAME}.tar.gz"
curl -fsSLO "${BASE}/${NAME}.tar.gz.sha256"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 -c "${NAME}.tar.gz.sha256"
else
  sha256sum -c "${NAME}.tar.gz.sha256"
fi
tar -xzf "${NAME}.tar.gz"
install -m 0755 "${NAME}/bayesite" "$HOME/.local/bin/bayesite"
```

For Windows, download `bayesite-${VERSION}-x86_64-pc-windows-msvc.zip` and the
adjacent `.sha256` file from the same release, verify the checksum, unzip, and
put `bayesite.exe` on `PATH`.

Release automation builds these assets, packages `README.md`/`LICENSE`/`NOTICE`
with the executable, writes checksums, and runs the same JSON-error smoke used
by local validation before publishing artifacts.

Developer fallback when Rust/Cargo is already present:

```sh
cargo install \
  --git https://github.com/StefanSko/bayesite \
  --tag v0.2.2 \
  --package bayesite-core \
  --bin bayesite \
  --locked
```

## Build and validate

```sh
python3 scripts/check_validation_ladder.py
```

This default development gate checks:

- zero-dependency core via `cargo tree`;
- vendored bayeswire spec/corpus byte hashes;
- `cargo fmt --check`;
- `cargo clippy -D warnings`;
- release packaging helper tests;
- release CLI build and JSON-error smoke test;
- Rust tests;
- wasm build;
- G6 statistical agreement against a pinned `nuts-rs` checkout.

By default, G6 expects `/tmp/nuts-rs` at the pinned revision documented in
[`docs/validation-ladder.md`](docs/validation-ladder.md), or pass
`--nuts-rs-path` explicitly.

Equivalent `just` entry point, if installed:

```sh
just check
```

Optional posterior conformance against the exactly pinned `bayesjax` release:

```sh
python3 scripts/check_validation_ladder.py --posterior
```

Pass `--bayescycle-path ../bayescycle` to test an unpublished monorepo checkout
instead.

The optional oracle path may use Python/JAX/BlackJAX. It is not part of the
agent execution path. GitHub Actions also runs G6/G7 conformance on a schedule,
manual dispatch, and release tags.

## CLI examples

```sh
bayesite sample \
  --model model_ir.json \
  --data data.json \
  --seed 1 \
  --chains 4 \
  --warmup 1000 \
  --draws 1000 \
  --out fit.jsonl

bayesite diagnose \
  --fit fit.jsonl \
  --out diagnostics.json

bayesite prior-predictive \
  --model model_ir.json \
  --data predictors.json \
  --seed 1 \
  --draws 1000 \
  --out pp.jsonl

bayesite posterior-predictive \
  --model model_ir.json \
  --data observed_data.json \
  --fit fit.jsonl \
  --seed 2 \
  --out yrep.jsonl

bayesite posterior-check \
  --model model_ir.json \
  --data observed_data.json \
  --fit fit.jsonl \
  --seed 2 \
  --out ppc.json

bayesite simulate \
  --model generator_ir.json \
  --data fixed_inputs.json \
  --truth truth.json \
  --seed 1 \
  --out generated_data.json

bayesite recover-check \
  --fit fit.jsonl \
  --truth truth.json \
  --targets targets.json \
  --interval 0.8 \
  --out recovery_check.json

bayesite recover \
  --model model_ir.json \
  --scenario recover_scenario.json \
  --out recover.json

bayesite sbc \
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
  conformance compares posterior summaries against `bayesjax`/BlackJAX.
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
