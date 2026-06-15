# Validation ladder

Bayesite separates the **agent execution path** from the **development
conformance path**.

- Agent path: one Bayesite binary, no Python, no package manager, no NumPy.
- Development path: may use `jaxstanv5`, JAX/BlackJAX, CmdStan, and report
  generators as explicit oracles.

The default Rust gates must stay self-contained. Oracle-backed checks are
explicit slow/conformance gates.

Workflow artifact field details live in [`artifacts-v0.md`](artifacts-v0.md),
not in this ladder. The ladder states what must be tested and in what order.

## Gates

### G0 — Spec snapshot and zero-dependency core

- IR docs and `tests/golden_ir/` are committed snapshots.
- `cargo tree --manifest-path crates/core/Cargo.toml` must show only
  `bayesite-core`.
- The release `bayesite` CLI binary must build as the default agent artifact.
- The wasm target must build.

### G1 — Decode, protocol, and artifact contract

Bayesite must decode every golden IR fixture and reject malformed documents with
typed, repair-oriented JSON errors. This includes malformed v1 envelopes,
duplicate envelope fields, duplicate node tag fields, unknown tags, malformed
maps, and unsupported versions.

The CLI and wasm/native protocol must expose the intended command set:

```text
sample | diagnose | prior-predictive | recover | sbc
```

G1 pins the common artifact contract:

- every workflow artifact/error carries its `v0-provisional` marker;
- stdout/stderr behavior is machine-readable and repair-oriented;
- `--out` and stdin-capable input paths behave consistently;
- duplicate flags/fields, missing flag values, bad command names, bad count
  ranges, bad seeds, and invalid tree-depth bounds fail before hidden work;
- self-describing counts, orders, coordinate orders, artifact kind/scope, and
  index-base metadata remain present where required by
  [`artifacts-v0.md`](artifacts-v0.md);
- unavailable R-hat/ESS values are JSON `null`, not `NaN`, infinity, or late
  serialization failures.

Covered primarily by `tests/ir_decode.rs`, `tests/protocol.rs`, `tests/cli.rs`,
and `tests/short_diagnostics.rs`.

### G2 — Log-density and gradient parity

For every golden fixture:

```text
IR + data + q -> log_density + gradient
```

Bayesite must match committed `jaxstanv5`/JAX oracle values. Current tolerances:

- log density: rtol `1e-12`
- gradient: rtol `1e-10`

Covered by `crates/core/tests/fixtures_eval.rs`.

### G3 — Binding, transforms, and state layout

Focused checks cover:

- `free_values` packing order;
- constraint inverse transforms and Jacobians;
- missing, extra, and wrong-shaped data;
- observed and partially observed coordinates;
- constrained output values.

These should remain fixture-backed whenever possible.

### G4 — Sampler mechanics

Pure Rust checks cover sampler machinery independent of Python:

- PRNG reference vectors;
- deterministic and distinct chain streams;
- NUTS transition behavior;
- divergence handling;
- warmup/adaptation schedule;
- diagnostics formulas.

### G5 — Analytic statistical targets

Fixed-seed runs against analytic targets check posterior means/variances,
divergences, adaptation sanity, and prior-predictive scalar distributions. These
are self-contained Rust tests in files such as
`crates/core/tests/sampler_stats.rs` and
`crates/core/tests/prior_predictive_stats.rs`.

### G6 — Cross-backend posterior comparison

Optional conformance gate using `jaxstanv5` + BlackJAX as an oracle. Compare
posterior summaries over the golden corpus, not bit-identical draws:

```sh
uv run scripts/check_rust_backend_posterior.py --jaxstanv5-path ../jaxstanv5
```

This gate must not become part of the default agent path.

### G7 — CmdStan comparison

Future independent conformance gate for models expressible in Stan. This should
catch shared assumptions between Bayesite and the JAX oracle.

### G8 — Prior predictive

`bayesite prior-predictive` emits v0-provisional NDJSON over decoded IR and
provided declared data. G8 pins:

- CLI and wasm/native protocol request paths;
- declared-data echo metadata and integer JSON serialization;
- generated-site order, shape, coordinate order, source stochastic-site names,
  roles, and integer flags;
- per-draw and trailer provenance/count metadata;
- typed repair errors for zero draws and unsupported/non-assignable sites;
- analytic scalar Normal and Bernoulli simulation checks.

Current limitation: prior predictive supports directly assignable stochastic
sites only. Broader analytic summary checks and `jaxstanv5` reference
comparisons remain future G8 conformance work.

### G9 — Recover

`bayesite recover` currently runs one v0-provisional scenario: simulate
truth/data through the prior-predictive path, fit the generated observed data
with NUTS, and report factual recovery metadata.

G9 pins:

- scenario parsing and repair errors;
- declared-data and generated-observed metadata;
- seed derivation and artifact provenance;
- parameter order, constrained-scale truth, ranks, exact tie counts,
  equal-tailed interval facts, R-hat/ESS labels, and coordinate order;
- aggregate sampler counts and per-chain raw sampler facts;
- absence of aggregate recovery pass/fail or coverage verdicts.

Current limitation: recover is a single-scenario factual report. Repeated-scenario
coverage summaries remain future G9 conformance work.

### G10 — SBC

`bayesite sbc` currently runs v0-provisional simulation-based calibration
scenarios through the pure runtime path:

```text
prior -> simulate data -> sample posterior -> rank true parameter among draws
```

G10 pins:

- scenario parsing, replicate settings, and repair errors;
- declared-data and generated-observed metadata per replicate;
- seed schedule and prior/posterior artifact provenance;
- aggregate and per-replicate parameter order, truth values, ranks, exact tie
  counts, rank support, coordinate order, and rank histograms;
- aggregate and per-replicate sampler count metadata;
- absence of aggregate uniformity, pass/fail, or sampler-quality verdicts.

Current limitation: SBC reports ranks and histograms but no uniformity verdict or
p-value. Broader SBC conformance over larger replicate counts remains future G10
work.

## Default command

The self-contained ladder subset is runnable with:

```sh
python3 scripts/check_validation_ladder.py
```

Optional oracle-backed posterior comparison:

```sh
python3 scripts/check_validation_ladder.py --posterior --jaxstanv5-path ../jaxstanv5
```
