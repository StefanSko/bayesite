# Validation ladder

Bayesite separates the **agent execution path** from the **development
conformance path**.

- Agent path: one Bayesite binary, no Python, no package manager, no NumPy.
- Development path: uses a pinned `nuts-rs` checkout as an independent NUTS
  oracle and may also use `bayesjax`, JAX/BlackJAX, CmdStan, and report
  generators as explicit additional oracles.

The core crate must stay zero-dependency. The mandatory `nuts-rs` oracle is a
validation-time dependency only and is never linked into Bayesite.

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
sample | diagnose | prior-predictive | posterior-predictive | posterior-check | simulate | recover-check | recover | sbc
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

### G6 — Independent NUTS statistical oracle

Mandatory development gate using a pinned `nuts-rs` checkout as an independent
NUTS implementation. The gate samples analytic Gaussian targets with Bayesite
and `nuts-rs`, then compares summary estimates using batch Monte Carlo standard
errors (MCSE):

```text
z = abs(bayesite_stat - nuts_rs_stat) / sqrt(mcse_bayesite^2 + mcse_nuts_rs^2)
```

Each backend is also compared to analytic truth. The gate covers:

- scalar standard Normal;
- shifted/scaled scalar Normal;
- vector diagonal Normal;
- correlated bivariate MVN.

The pinned upstream revision is:

```text
5332136767cade60bdeec84cd5b2e0f273961d4c
```

Prepare the oracle checkout with:

```sh
git clone https://github.com/pymc-devs/nuts-rs /tmp/nuts-rs
git -C /tmp/nuts-rs checkout 5332136767cade60bdeec84cd5b2e0f273961d4c
```

Run directly with:

```sh
python3 scripts/check_nuts_rs_oracle.py --nuts-rs-path /tmp/nuts-rs
```

This gate must not add dependencies to `bayesite-core` or to the Bayesite agent
execution path. The conformance CI workflow runs this gate on a schedule,
manual dispatch, and release tags.

### G7 — Cross-backend posterior comparison

Optional local conformance gate using `bayesjax` + BlackJAX as an oracle.
Compare posterior summaries over the golden corpus, not bit-identical draws:

```sh
uv run scripts/check_rust_backend_posterior.py
```

The script pins `bayesjax==0.5.0` exactly. Pass
`--bayescycle-path ../bayescycle` only to test an unpublished monorepo checkout.
This gate must not become part of the default agent path. The conformance CI
workflow also runs it on a schedule, manual dispatch, and release tags, so
cross-backend drift is visible without adding Python/JAX to the shipped binary.
After each Bayescycle release, the pin advances in a reviewed Bayesite change
only after G7 passes.

### G8 — CmdStan comparison

Future independent conformance gate for models expressible in Stan. This should
catch shared assumptions between Bayesite and the JAX oracle.

### G9 — Prior predictive

`bayesite prior-predictive` emits v0-provisional NDJSON over decoded IR and
provided declared data. G9 pins:

- CLI and wasm/native protocol request paths;
- declared-data echo metadata and integer JSON serialization;
- generated-site order, shape, coordinate order, source stochastic-site names,
  roles, and integer flags;
- per-draw and trailer provenance/count metadata;
- typed repair errors for zero draws and unsupported/non-assignable sites;
- analytic scalar Normal and Bernoulli simulation checks.

Current limitation: prior predictive supports directly assignable stochastic
sites only. Broader analytic summary checks and `bayesjax` reference
comparisons remain future G9 conformance work.

### G10 — Simulation and recovery checks

`bayesite simulate` generates a plain data document from declared inputs,
supplied constrained free-value truth, a simulation model, and an explicit seed.
`bayesite recover-check` compares a complete posterior fit stream to supplied
truth values without requiring model/data/simulation provenance. `bayesite
recover` remains a single-command convenience workflow that simulates truth/data
through the prior-predictive path, fits generated observed data with NUTS, and
reports factual recovery metadata.

G10 pins:

- fixed-truth simulation parsing and repair errors;
- simulation output as normal sample-consumable data, with declared inputs and
  generated observed values in explicit order;
- fixed-truth validation for missing/unknown free values, shape mismatches, and
  constraint violations;
- recover-check target parsing, default same-name mappings, explicit renamed
  mappings, and same-shape validation;
- recover-check parameter order, constrained-scale truth, ranks, exact tie
  counts, equal-tailed interval facts, R-hat/ESS labels, and coordinate order;
- recover scenario parsing, declared-data and generated-observed metadata, seed
  derivation, artifact provenance, sampler counts, and per-chain raw sampler
  facts;
- absence of aggregate recovery pass/fail or coverage verdicts.

Current limitation: recover is a single-scenario factual report, and
recover-check supports same-shape targets rather than transformed estimands.
Repeated-scenario coverage summaries remain future G10 conformance work.

### G11 — SBC

`bayesite sbc` currently runs v0-provisional simulation-based calibration
scenarios through the pure runtime path:

```text
prior -> simulate data -> sample posterior -> rank true parameter among draws
```

G11 pins:

- scenario parsing, replicate settings, and repair errors;
- declared-data and generated-observed metadata per replicate;
- seed schedule and prior/posterior artifact provenance;
- aggregate and per-replicate parameter order, truth values, ranks, exact tie
  counts, rank support, coordinate order, and rank histograms;
- aggregate and per-replicate sampler count metadata;
- absence of aggregate uniformity, pass/fail, or sampler-quality verdicts.

The binary deliberately remains a verdict-free factual reporter: `bayesite sbc`
still emits ranks and histograms, not a uniformity verdict or p-value. The
development ladder now applies a mandatory, default-on conformance verdict in
`scripts/check_sbc_uniformity.py`. It resolves exact ties by seeded uniform
randomization and tests each parameter-coordinate rank ECDF against a
Monte Carlo-calibrated simultaneous binomial confidence band, with Bonferroni
control across scenarios and parameters, following Säilynoja, Bürkner & Vehtari
(2021). Broader data-averaged SBC variants remain future G11 work.

### G12 — Posterior predictive checks

`bayesite posterior-predictive` runs the pure runtime path:

```text
posterior draw theta_s -> simulate replicated observed data y_rep_s
```

`bayesite posterior-check` builds factual discrepancy summaries on top of that
stream.

G12 pins:

- CLI and wasm/native protocol request paths;
- fit-stream parsing and source posterior draw provenance;
- generated observed-site order, shape, coordinate order, source stochastic-site
  names, integer flags, and per-draw values;
- posterior-check report markers, report identity, posterior-predictive artifact
  provenance, built-in discrepancy names, observed values, replicated summaries,
  and tail-count facts;
- absence of posterior-check pass/fail/model-quality verdicts.

Current limitations: posterior predictive supports directly assignable observed
stochastic sites only, and posterior-check supports built-in generic discrepancy
statistics only.

## Default command

The development ladder is runnable with a pinned `nuts-rs` checkout:

```sh
python3 scripts/check_validation_ladder.py --nuts-rs-path /tmp/nuts-rs
```

Optional oracle-backed posterior comparison:

```sh
python3 scripts/check_validation_ladder.py --posterior
```
