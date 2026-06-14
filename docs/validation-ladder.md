# Validation ladder

Bayesite separates the **agent execution path** from the **development
conformance path**.

- Agent path: one Bayesite binary, no Python, no package manager, no NumPy.
- Development path: may use `jaxstanv5`, JAX/BlackJAX, CmdStan, and report
  generators as oracles.

The default Rust gates must stay self-contained. Oracle-backed checks are
explicit slow/conformance gates.

## Gates

### G0 — Spec snapshot and zero-dependency core

- IR docs and `tests/golden_ir/` are committed snapshots.
- `cargo tree --manifest-path crates/core/Cargo.toml` must show only
  `bayesite-core`.
- The wasm target must build.

### G1 — Decode and protocol contract

Bayesite must decode every golden IR fixture and reject malformed documents with
typed, repair-oriented JSON errors.

Covered by Rust tests such as `tests/ir_decode.rs`, `tests/protocol.rs`, and
`tests/cli.rs`.

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

- `free_values` packing order
- constraint inverse transforms and Jacobians
- missing/extra/wrong-shaped data
- observed and partially observed coordinates
- constrained output values

These should remain fixture-backed whenever possible.

### G4 — Sampler mechanics

Pure Rust checks cover the sampler machinery independent of Python:

- PRNG reference vectors
- deterministic and distinct chain streams
- NUTS transition behavior
- divergence handling
- warmup/adaptation schedule
- diagnostics formulas

### G5 — Analytic statistical targets

Fixed-seed runs against analytic targets check posterior means/variances,
divergences, and adaptation sanity. These are self-contained Rust tests in
`crates/core/tests/sampler_stats.rs`.

### G6 — Cross-backend posterior comparison

Optional conformance gate using `jaxstanv5` + BlackJAX as an oracle. Compare
posterior summaries over the golden corpus, not bit-identical draws:

```sh
uv run scripts/check_rust_backend_posterior.py --jaxstanv5-path ../jaxstanv5
```

### G7 — CmdStan comparison

Future independent conformance gate for models expressible in Stan. This should
catch shared assumptions between Bayesite and the JAX oracle.

### G8 — Prior predictive

When `bayesite prior-predictive` exists, compare simulation shapes and summaries
against analytic cases and `jaxstanv5` reference outputs.

### G9 — Recover

When `bayesite recover` exists, generate parameters/data, fit, and report
coverage of true parameters across repeated scenarios.

### G10 — SBC

Final workflow validation:

```text
prior -> simulate data -> sample posterior -> rank true parameter among draws
```

Report rank histograms, uniformity diagnostics, divergences, R-hat, ESS, and
per-parameter failures.

## Default command

The self-contained ladder subset is runnable with:

```sh
python3 scripts/check_validation_ladder.py
```

Optional oracle-backed posterior comparison:

```sh
python3 scripts/check_validation_ladder.py --posterior --jaxstanv5-path ../jaxstanv5
```
