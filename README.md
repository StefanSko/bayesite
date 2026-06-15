# Bayesite — zero-dependency Bayesian workflow binary

Bayesite is the seed of a single-static-binary, agent-operable Bayesian
workflow engine for serialized model IR: the SQLite-like artifact an agent can
run from the CLI without Python, `uvx`, NumPy, or a runtime dependency graph.

Current implementation status: the Rust core consumes `jaxstanv5_ir` v1
fixtures and implements posterior sampling through `bayesite sample`,
diagnostic replay through `bayesite diagnose`, and a first prior-predictive
workflow through `bayesite prior-predictive`, plus a transparent single-scenario
recovery report through `bayesite recover` and rank reporting through
`bayesite sbc`. The endgame command surface is:

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
5. Recomputes fit diagnostics from that provisional stream with
   `bayesite diagnose --fit`.
6. Emits prior-predictive simulations in a separate **v0-provisional** NDJSON
   stream for directly assignable stochastic sites.
7. Runs a single recovery scenario and reports generated truth/data, posterior
   intervals, and sampler diagnostics without adding an aggregate verdict.
8. Runs simulation-based calibration replicates and reports ranks, rank
   histograms, generated data, and sampler diagnostics without adding a
   uniformity verdict.

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
- **Prior predictive:** fixed-seed scalar Normal and Bernoulli simulation checks
  against analytic moments, including integer JSON artifact output
  (`crates/core/tests/prior_predictive_stats.rs`).

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
| `crates/core/src/bin/bayesite.rs` | CLI command dispatcher |
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
build breaking on wasm is a bug. The default ladder also builds the release
`bayesite` CLI binary and smoke-tests its JSON error path, so the agent-facing
artifact is part of routine validation.

## CLI

```sh
cargo build --release --bin bayesite
./target/release/bayesite sample \
    --model model_ir.json --data data.json \
    --seed 1 --chains 4 --warmup 1000 --draws 1000 \
    --out fit.jsonl
./target/release/bayesite diagnose --fit fit.jsonl --out diagnostics.json
./target/release/bayesite prior-predictive \
    --model model_ir.json --data predictors.json \
    --seed 1 --draws 1000 --out pp.jsonl
./target/release/bayesite recover \
    --model model_ir.json --scenario recover_scenario.json \
    --out recover.json
./target/release/bayesite sbc \
    --model model_ir.json --scenario sbc_scenario.json \
    --replicates 100 --out sbc.json
```

`bayesite sample` writes NDJSON to `--out`, or stdout when `--out -` is used or
omitted: a header (`"draws_format": "v0-provisional"`,
`artifact_kind: "posterior_draws"`,
`artifact_scope: "observed_data_conditioned_parameter_draws"`,
parameter shapes, zero-based row-major parameter `coordinate_order`, packing
order, explicit `parameter_order`, explicit `parameter_count`,
`workflow_phases`, settings, and
`chain_count`, `chain_order`, and total retained `draw_count`), one object per
draw with the same `"draws_format": "v0-provisional"` marker and artifact
kind/scope, a zero-based retained-order `draw_index`, `draw_index_base`, seed,
total retained `draw_count`, `chain_count`, `chain_order`, chain id,
`chain_index_base`, per-chain draw index, `parameter_count`,
`parameter_order`, and constrained values keyed by parameter, and a trailer with
`chain_count`, `chain_order`, total retained `draw_count`, `parameter_count`,
`parameter_order`, per-chain draw counts, per-chain `chain_index_base`,
divergences, tree-depth histograms, step sizes, cross-chain R-hat/ESS, the same
`"draws_format": "v0-provisional"` marker, the same artifact kind/scope, and
completion metadata (`seed`, `draws_per_chain`, `params`). The top-level
`draw_count` is the number of retained draw records across all chains. Errors
are a single v0-provisional JSON object on stderr
(`{"error_format": "v0-provisional", "error": "<Kind>", "message": ...}`) with
a nonzero exit code; messages are written as repair instructions.

Artifact commands use `--out` uniformly. `diagnose`, `recover`, and `sbc` write
one JSON object to stdout when `--out` is omitted or `--out -` is used;
`sample` and `prior-predictive` write NDJSON the same way. Successful artifact
commands keep stderr empty. R-hat/ESS values that are mathematically unavailable
for short or degenerate chains are encoded as JSON `null`, never as non-JSON
`NaN` or infinity values.

For commands with multiple input documents, at most one input path may be `-`;
Bayesite rejects ambiguous stdin use before reading. `sample` and
`prior-predictive` accept `--data -` when `--model` is a path, while `recover`
and `sbc` accept `--scenario -` when `--model` is a path. Direct CLI
commands reject duplicate singleton flags, such as repeated `--seed` or
`--replicates`, before value parsing or input reads. A flag that needs a value
also rejects the next `--flag` token as a missing value instead of treating it
as a path.

The `v0-provisional` marker is mandatory: the real fit-artifact format is
defined elsewhere, and nothing may grow load-bearing dependencies on this one
without noticing.

`bayesite diagnose` reads the complete v0-provisional NDJSON fit stream from a
path or `-` and emits one JSON object to stdout, or to `--out` when provided.
The object includes `"diagnostics_format": "v0-provisional"`, diagnose
`workflow_phases`, the source draw format, optional `source_artifact_kind` and
`source_artifact_scope`, `source_seed`, `source_chains`, `source_chain_count`,
`source_chain_order`, `source_draw_count`, `source_draw_index_metadata`,
`source_draw_parameter_metadata`, `source_draw_artifact_metadata`,
`source_draw_chain_metadata`, `source_parameter_count`, `source_settings`,
`source_params` with shapes and `coordinate_order`, `source_packing`, `source_parameter_order`,
`source_trailer_completion_metadata`,
`source_workflow_phases`, draws per chain, per-chain sampler stats from the
trailer, and recomputed per-parameter R-hat/ESS with statistic and
coordinate-reduction labels. If a fit trailer has a
`draws_format` field, `diagnose` validates it against the v0 marker; older v0
artifacts without a trailer marker are still accepted. If artifact kind/scope
fields are present in the header or trailer, `diagnose` validates them against
the v0 sample artifact identity and reports them as source facts. If trailer
completion metadata is present, `diagnose` validates it against the header and
reports which completion fields were present, including artifact kind/scope and
chain count presence. If the optional header or trailer `chain_count` field is
present, `diagnose` validates it against the legacy header `chains` count. If
optional header or trailer `chain_order` fields are present, `diagnose`
validates them against the draw-chain order and reports normalized
`source_chain_order`. If
optional header or trailer top-level `draw_count` fields are present, `diagnose`
validates them against the retained draw-line count and reports the normalized
`source_draw_count`. If draw-line `draw_index` metadata is present, `diagnose`
validates that it appears on every draw line and is contiguous from 0 in
retained draw-line order; legacy streams without `draw_index` remain accepted.
`source_draw_index_metadata` reports which case was observed.
If draw-line `parameter_count`/`parameter_order` metadata is present,
`diagnose` validates that both fields appear on every draw line and match the
header parameter layout; legacy streams without per-draw parameter metadata
remain accepted. `source_draw_parameter_metadata` reports which case was
observed.
If draw-line artifact metadata is present, `diagnose` validates
`draw_index`, `draws_format`, artifact kind/scope, `draw_index_base`, `seed`, and
`draw_count` against the v0 sample artifact and reports
`source_draw_artifact_metadata`; legacy streams without that group remain
accepted.
If draw-line chain metadata is present, `diagnose` validates `chain_count` and
`chain_order` against the observed draw chain ids and reports
`source_draw_chain_metadata`; legacy streams without those fields remain
accepted.
If optional header or trailer `parameter_count` fields are
present, `diagnose` validates them against the header `params` length. If
optional header or trailer `parameter_order` fields are present, `diagnose`
validates them against the header `params` order and reports normalized
`source_parameter_order`. If per-chain trailer `draw_count` fields are present,
`diagnose` validates them
against the header draw count before recomputing diagnostics. Its own
`workflow_phases` are parse fit NDJSON, validate the fit artifact, recompute
diagnostics, and emit the report. This artifact contract is still provisional.

`bayesite prior-predictive` reads the IR plus declared data inputs only; observed
values that should be simulated must be omitted from the data document. It emits
NDJSON with at least one draw and a header
(`"prior_predictive_format": "v0-provisional"`, draw count, seed,
`artifact_kind`, `artifact_scope`, `workflow_phases`, `draw_count`, `settings`,
`draw_index_base`, `site_count`, `site_order`, `declared_data_count`,
`declared_data_order`, declared data values/shapes/aggregate integer flags,
zero-based row-major declared data coordinate order, shape-preserving
`declared_data_integer_by_coordinate` flags, generated site names, source
stochastic-site names, roles, shapes, zero-based row-major `coordinate_order`,
aggregate integer flags, and shape-preserving `integer_by_coordinate` flags),
one object per draw with generated parameters and observations, and a trailer
repeating the format marker, artifact kind/scope, workflow phases, seed,
settings, `draw_count`, `draw_index_base`, `site_count`, `site_order`, and
`declared_data_count`/`declared_data_order`. The header and trailer also keep
the legacy provisional `draws` count field, and the trailer keeps the legacy
provisional `sites` count field, for compatibility.
`artifact_kind` is `prior_predictive_draws`, and `artifact_scope` is
`declared_data_conditioned_site_draws`.
`declared_data_count` reports the number of declared data inputs echoed in the
artifact.
`declared_data_order` records the declared input order explicitly.
`declared_data_coordinate_order` maps declared data values and flat
per-coordinate metadata back to declared input indexes; scalar inputs report one
empty coordinate path, `[[]]`.
Declared data values marked by `declared_data_integer_by_coordinate` are
serialized as JSON integers.
Generated-site `coordinate_order` maps flat per-coordinate metadata back to
site indexes; scalar sites report one empty coordinate path, `[[]]`.
`site_order` records the generated site order explicitly, matching the `sites`
metadata array and draw value object order.
Each generated draw record includes `prior_predictive_format`, `artifact_kind`,
`artifact_scope`, `draw_index`, `draw_index_base`, `seed`, `draw_count`,
`declared_data_count`, `declared_data_order`, `site_count`, and `site_order`, so
a draw line remains auditable when read apart from the header without repeating
full declared data values.
Generated draw coordinates marked by `integer_by_coordinate` are serialized as
JSON integers.
`workflow_phases` is the end-to-end command path: parse JSON, decode IR, bind
declared data, simulate prior predictive values, and emit the artifact.
The current simulator covers directly assignable sites over the core
distribution set
Normal/HalfNormal/StudentT/Exponential/Uniform/Beta/Bernoulli/Poisson/Binomial/
BetaBinomial/NegativeBinomial/MultivariateNormal/OrderedLogistic and fails with
JSON repair errors for non-assignable stochastic-site value expressions. This
artifact contract is still provisional.

`bayesite recover` reads a v0-provisional scenario document:

```json
{
  "recover_scenario": "v0-provisional",
  "data": {},
  "seed": 1,
  "interval": 0.8,
  "sample": {
    "chains": 1,
    "warmup": 100,
    "draws": 100,
    "max_treedepth": 6,
    "target_accept": 0.8
  }
}
```

The `data` object contains declared inputs only; Bayesite simulates observed
values from the prior-predictive path, fits the generated data with NUTS, and
emits one JSON report with `"recover_format": "v0-provisional"`,
`workflow_format`, `report_kind`, `report_scope`, `simulation_count`,
`simulation_index_base`, `simulation_order`, `workflow_phases`, seeds,
`seed_schedule`, settings,
`prior_predictive_draws`,
`prior_predictive_draws_artifact_kind`,
`prior_predictive_draws_artifact_scope`, `posterior_draws`,
`posterior_draws_artifact_kind`, `posterior_draws_artifact_scope`,
`parameter_count`, `parameter_report_count`,
`generated_observed_count`, `generated_observed_artifact_kind`,
`generated_observed_artifact_scope`, `generated_observed_draw_index`,
`generated_observed_draw_index_base`, `declared_data_count`, `chain_count`,
`rank_draws`,
`interval_method`, `interval_scope`,
`interval_contains_truth_statistic`, `interval_contains_truth_by_parameter`,
`rank_statistic`, `rank_scope`, `tie_statistic`, `parameter_summary_scale`,
`interval_bounds`,
`rank_bounds`, `rank_bin_order`, `rank_bin_count`, `tie_count_bounds`,
`tie_count_bin_order`, `tie_count_bin_count`, `generated_observed_order`,
`generated_observed`,
`generated_observed_stochastic_sites`,
`generated_observed_shapes`, `generated_observed_coordinate_order`,
`generated_observed_integer`,
`generated_observed_integer_by_coordinate`, `declared_data`,
`declared_data_order`, `declared_data_shapes`,
`declared_data_coordinate_order`, `declared_data_integer`,
`declared_data_integer_by_coordinate`, `parameter_order`, per-parameter source
stochastic-site name, shape, zero-based row-major `coordinate_order`, truth,
`truth_integer`, `truth_artifact_kind`, `truth_artifact_scope`,
`truth_draw_index`, `truth_draw_index_base`, `simulation`,
`simulation_index_base`, `prior_seed`, `sample_seed`, `seed_schedule`,
`rank_draws`, `posterior_draws`,
`posterior_draws_artifact_kind`, `posterior_draws_artifact_scope`,
`rank_bounds`, rank, tie count, mean,
rank/tie/interval-containment statistic labels, rank and tie-count support
metadata, interval method/scope/bounds,
`summary_scale`, lower/upper interval values,
`interval_contains_truth`, `interval_contains_truth_by_coordinate`, R-hat/ESS
with statistic and coordinate-reduction labels, and explicit `chain_order`,
per-chain kept draw counts and sampler stats plus
`sampler_summary`. The declared data fields echo the parsed scenario inputs
Bayesite conditioned on.
Declared data values marked by `declared_data_integer_by_coordinate` are
serialized as JSON integers.
`generated_observed_integer` uses booleans for scalar observations and boolean
arrays for vector observations, matching the generated value shape. Generated
observed coordinates marked integer are serialized as JSON integers. It does
not emit an aggregate pass/fail field; callers decide how to interpret the
facts.
`interval_contains_truth_by_parameter` is an object keyed by `parameter_order`
whose values mirror each parameter summary's `interval_contains_truth` fact.
It is a factual index, not an aggregate recovery verdict.
`generated_observed_integer_by_coordinate` records the model-side generated
observed integer flags that drive JSON integer serialization, separately from
the realized generated value.
`report_kind` is `parameter_recovery_facts`, `report_scope` is
`single_simulated_dataset`, and `simulation_count` is 1 so the report is not
mistaken for repeated-scenario coverage.
`simulation_index_base` is `zero_based_simulation_order`, and
`simulation_order` is `[0]` for the single v0 recovery simulation.
`generated_observed_order` records the generated observed site order explicitly.
`generated_observed_coordinate_order` records zero-based row-major coordinate
paths for generated observed values, matching the shape and flat per-coordinate
metadata.
`generated_observed_artifact_kind`, `generated_observed_artifact_scope`,
`generated_observed_draw_index`, and `generated_observed_draw_index_base` report
which prior-predictive draw produced the generated observed data Bayesite fit.
`declared_data_order` records the declared input order explicitly.
`declared_data_coordinate_order` maps declared data values and flat
per-coordinate metadata back to declared input indexes; scalar inputs report one
empty coordinate path, `[[]]`.
`parameter_count`, `parameter_report_count`, `generated_observed_count`, and
`declared_data_count` report the sizes of those emitted sections explicitly.
`parameter_report_count` reports the size of the emitted `parameters` section.
The `settings` object repeats the effective sample settings and interval
probability used for the recovery fit, so consumers do not have to infer the
interval from the top-level compatibility field.
`seed_schedule` reports that `prior_seed` is the top-level `seed` plus offset 0
and `sample_seed` is the top-level `seed` plus offset 1.
`prior_predictive_draws` reports the number of prior-predictive site draws used
to simulate the single truth/data set, with artifact kind/scope naming that
simulation source.
Each parameter summary repeats the truth artifact kind/scope and zero-based
truth draw index beside the simulated truth value, and repeats the zero-based
simulation index for the single recovery simulation.
`rank_draws` is the number of posterior draws used for the rank statistic.
`rank` is the count of posterior draws strictly less than the simulated truth,
with exact ties reported separately as `tie_count`. Each parameter summary
repeats `rank_draws`, `posterior_draws`,
`posterior_draws_artifact_kind`, `posterior_draws_artifact_scope`,
`rank_bounds`, `rank_bin_order`, `rank_bin_count`,
`rank_statistic`, `rank_scope`, `tie_statistic`, `tie_count_bounds`,
`tie_count_bin_order`, and `tie_count_bin_count` for the rank and tie-count
values inside that parameter object.
`rank_bin_order` records the rank support explicitly as integers from 0 through
`rank_draws`. `rank_bin_count` records the number of possible rank values, which
is `rank_draws + 1`, so consumers do not have to infer support size from
`rank_bounds`.
`tie_count_bounds`, `tie_count_bin_order`, and `tie_count_bin_count` record the
same count support for exact tie counts.
`parameter_summary_scale` and per-parameter `summary_scale` report
`constrained_parameter_value`, the scale used for truth, means, intervals,
rank comparisons, and containment facts.
`interval_bounds` reports the requested `interval_probability`, lower and upper
excluded tail probabilities, and the lower/upper quantile probabilities used
for each marginal interval. It also reports `quantile_index_base`,
`sorted_draw_count`, and lower/upper quantile index objects with `position`,
`floor`, and `ceil`, making the equal-tailed linear interpolation over sorted
posterior draws auditable without implying a verdict. Each parameter summary
repeats the interval method, scope, containment statistic, and bounds beside
its lower/upper interval values.
`coordinate_order` maps flat per-coordinate arrays back to parameter indexes;
scalar parameters report one empty coordinate path, `[[]]`.
Each chain report includes `draw_count`, `chain_index_base`, divergences,
tree-depth histogram, explicit `treedepth_bin_order` and
`treedepth_bin_count`, step size, and mean acceptance.
`chain_count` reports the size of the emitted `chains` section, and
`chain_order` records the emitted chain-id order explicitly.
`sampler_summary` reports aggregate chain count, kept draw count, total
divergences, a tree-depth histogram, and explicit tree-depth histogram support
metadata for the recovery fit.
`parameter_order` records the IR `free_values` packing order explicitly, and
parameter objects are also emitted in that order. If a packed free value has no
directly simulated truth value, `recover` returns a JSON repair error instead of
emitting a partial parameter report. `workflow_phases` is the end-to-end command
path: parse JSON, decode IR, bind declared data, simulate
prior predictive values, bind declared plus generated data, build posterior
state, evaluate logp/gradient, run NUTS, and emit the report.

Scenario control objects reject unknown or duplicate fields so misspelled or
ambiguous keys fail with a JSON repair error instead of silently falling back to
defaults. The `data` object remains model-defined and may contain declared input
names from the IR.

`bayesite sbc` reads a v0-provisional scenario document:

```json
{
  "sbc_scenario": "v0-provisional",
  "data": {},
  "seed": 1,
  "replicates": 100,
  "sample": {
    "chains": 1,
    "warmup": 100,
    "draws": 100,
    "max_treedepth": 6,
    "target_accept": 0.8
  }
}
```

The optional `--replicates` flag overrides the scenario count. The output is
one JSON report with `"sbc_format": "v0-provisional"`, `workflow_format`,
seeds, effective settings,
`report_kind`, `report_scope`, `replicate_workflow_phases`,
`replicates`, `replicate_count`, `replicate_report_count`,
`replicate_index_base`, `replicate_order`, `chain_count_per_replicate`,
`prior_predictive_draws_per_replicate`,
`generated_observed_count_per_replicate`,
`generated_observed_order_per_replicate`,
`generated_observed_artifact_kind_per_replicate`,
`generated_observed_artifact_scope_per_replicate`,
`generated_observed_draw_index_per_replicate`,
`generated_observed_draw_index_base_per_replicate`,
`prior_predictive_draws_artifact_kind`,
`prior_predictive_draws_artifact_scope`, `parameter_count`,
`parameter_report_count`, `declared_data_count`, `rank_draws`,
`posterior_draws_per_replicate`,
`posterior_draws_artifact_kind`, `posterior_draws_artifact_scope`,
`rank_statistic`, `rank_scope`,
`tie_statistic`, `tie_count_bounds`, `tie_count_bin_order`,
`tie_count_bin_count`, `parameter_summary_scale`, `seed_schedule`,
`rank_bounds`, `rank_bin_order`, `rank_bin_count`,
`sampler_summary`,
`declared_data_order`, `declared_data`,
`declared_data_shapes`, `declared_data_coordinate_order`, `declared_data_integer`,
`declared_data_integer_by_coordinate`, `parameter_order`, per-parameter ranks,
tie counts, rank support, replicate count, source stochastic-site name,
zero-based row-major `coordinate_order`, `summary_scale`, `truth_integer`,
posterior draw count and artifact identity, and rank histograms,
per-replicate
generated observed values, source stochastic-site names, shapes,
zero-based row-major coordinate order, integer flags,
generated observed integer-by-coordinate flags,
explicit generated observed order, `parameter_order`,
per-replicate parameter shape, source stochastic-site name,
truth/`truth_integer`/rank support/rank/tie_count/R-hat/ESS, per-replicate
rank/tie/R-hat/ESS statistic labels, `summary_scale`, per-replicate
`seed_schedule`, prior-predictive draw count and artifact identity, generated-observed artifact
identity and draw index, and per-chain kept draw counts and sampler stats plus
per-replicate `sampler_summary`. The declared data
fields echo the parsed scenario inputs Bayesite conditioned on.
Declared data values marked by `declared_data_integer_by_coordinate` are
serialized as JSON integers.
Per-replicate `generated_observed_integer` uses booleans for scalar observations
and boolean arrays for vector observations, matching the generated value shape.
Generated observed coordinates marked integer are serialized as JSON integers.
Per-replicate `generated_observed_integer_by_coordinate` records the model-side
generated observed integer flags that drive that serialization, separately from
the realized generated value.
`declared_data_order` records the declared input order explicitly.
`declared_data_coordinate_order` maps declared data values and flat
per-coordinate metadata back to declared input indexes; scalar inputs report one
empty coordinate path, `[[]]`.
`replicate_count` repeats the explicit number of simulated SBC replicates; the
legacy provisional `replicates` field remains for compatibility. The
`settings` object also repeats the effective replicate count, after any
`--replicates` override, beside the sampler settings.
`replicate_report_count` reports the size of the emitted `replicate_reports`
section.
Top-level `parameter_count`, `parameter_report_count`, and
`declared_data_count` report the sizes of the aggregate parameter and declared
data sections explicitly. `parameter_report_count` reports the size of the
emitted aggregate `parameters` section.
Top-level `generated_observed_count_per_replicate` and
`generated_observed_order_per_replicate` report the generated-observed section
shape and site order inside each replicate. Top-level
`generated_observed_artifact_kind_per_replicate`,
`generated_observed_artifact_scope_per_replicate`,
`generated_observed_draw_index_per_replicate`, and
`generated_observed_draw_index_base_per_replicate` report the shared
prior-predictive artifact provenance for those generated-observed sections.
Per-replicate `generated_observed_order` records the generated observed site
order explicitly.
Per-replicate `generated_observed_coordinate_order` records zero-based row-major
coordinate paths for generated observed values, matching the shape and flat
per-coordinate metadata.
Per-replicate `generated_observed_artifact_kind`,
`generated_observed_artifact_scope`, `generated_observed_draw_index`, and
`generated_observed_draw_index_base` report which prior-predictive draw
produced that replicate's generated observed data.
Each per-replicate report repeats `sbc_format`, `workflow_format`,
`report_kind`, `report_scope`, `workflow_phases`, `replicate_count`,
`replicate_index_base`, `replicate_order`, sample `settings`,
`rank_draws`, `posterior_draws`, `posterior_draws_artifact_kind`,
`posterior_draws_artifact_scope`, `rank_bounds`, `rank_bin_order`,
`rank_bin_count`, `tie_count_bounds`, `tie_count_bin_order`,
`tie_count_bin_count`, `parameter_summary_scale`, `rank_statistic`, `rank_scope`,
`tie_statistic`, `declared_data_count`, and
`declared_data_order`, and `chain_count` for the ranks inside that replicate, so a single
replicate object remains auditable when read apart from the aggregate report.
Per-replicate declared-data metadata names inherited scenario inputs; full
declared data values are emitted once in the aggregate report.
Per-replicate `settings` records the effective chain count, warmup count, draw
count, maximum tree depth, and target acceptance used for that replicate.
`replicate_index_base` is `zero_based_replicate_order` at aggregate,
per-replicate, aggregate-parameter, and per-replicate-parameter scopes.
Each per-replicate report also includes `parameter_count`,
`parameter_report_count`, and `generated_observed_count` for its emitted
parameter and generated-observed sections. Per-replicate
`parameter_report_count` reports the size of that replicate's emitted
`parameters` section.
Aggregate and per-replicate parameter summaries report `shape`; their
`coordinate_order` fields map flat per-coordinate arrays and rank histograms
back to parameter indexes. Scalar parameters report an empty shape, `[]`, and
one empty coordinate path, `[[]]`.
Aggregate parameter summaries repeat `rank_draws`,
`posterior_draws_per_replicate`, `posterior_draws_artifact_kind`,
`posterior_draws_artifact_scope`, `rank_bounds`,
`rank_bin_order`, and `rank_bin_count`, plus `rank_statistic`, `rank_scope`,
`tie_statistic`, `tie_count_bounds`, `tie_count_bin_order`,
`tie_count_bin_count`, `rank_histogram_statistic`, `rank_histogram_scope`, and
`replicate_count`, `replicate_index_base`, `replicate_order`, `prior_seed`,
`sample_seed`, and `seed_schedule` beside their rank arrays and histograms.
`prior_seed` and `sample_seed` are arrays aligned with `replicate_order`.
They also repeat simulated `truth` values beside
`truth_integer`, `truth_artifact_kind`, `truth_artifact_scope`,
`truth_draw_index`, and `truth_draw_index_base`, with truth values and draw
indexes aligned with `replicate_order`. Per-replicate
parameter summaries repeat `truth_artifact_kind`, `truth_artifact_scope`,
`truth_draw_index`, `truth_draw_index_base`, `prior_seed`, `sample_seed`,
`replicate`, `replicate_index_base`, `seed_schedule`, `rank_draws`, `posterior_draws`,
`posterior_draws_artifact_kind`, `posterior_draws_artifact_scope`,
`rank_bounds`, `rank_bin_order`, `rank_bin_count`, `rank_statistic`,
`rank_scope`, `tie_statistic`, `tie_count_bounds`, `tie_count_bin_order`, and
`tie_count_bin_count` beside their rank and tie-count values.
`parameter_summary_scale` and aggregate/per-replicate parameter `summary_scale`
report `constrained_parameter_value`, the scale used for truth values, rank
comparisons, tie counts, and rank histograms.
`rank_bin_order` records the histogram bin labels explicitly as ranks from 0
through `rank_draws`. `rank_bin_count` records the number of possible rank bins,
which is `rank_draws + 1`, so consumers do not have to infer histogram support
size from the rank bounds.
`tie_count_bounds`, `tie_count_bin_order`, and `tie_count_bin_count` record the
same support for exact tie counts.
`rank_histogram_statistic` is `count_simulated_replicates_by_rank`, and
`rank_histogram_scope` is `per_parameter_coordinate_marginal`; these label the
histogram facts without interpreting their uniformity.
Aggregate vector-parameter `rank_histogram` values are shape-preserving: one
histogram per coordinate in `coordinate_order`, with bins for every rank from 0
through `rank_draws`; each coordinate histogram sums to the replicate count.
Each per-replicate chain report includes `draw_count`, `chain_index_base`,
divergences, tree-depth histogram, explicit `treedepth_bin_order` and
`treedepth_bin_count`, step size, and mean acceptance.
`replicate_order` records the emitted replicate order explicitly.
`chain_count_per_replicate` reports the intended chain section size for each
replicate. Each per-replicate `chain_count` reports the size of that
replicate's emitted `chains` section, `chain_order` records that
replicate's emitted chain-id order, and `chain_index_base` labels the chain id
convention.
`sampler_summary` reports aggregate chain count, kept draw count, total
divergences, a tree-depth histogram, and explicit tree-depth histogram support
metadata across all SBC replicate chains.
Each per-replicate `sampler_summary` reports those same count fields for only
that replicate, including the local tree-depth histogram support metadata.
The rank statistic is the count of posterior draws strictly less than the
simulated truth, with exact ties reported separately.
`posterior_draws_per_replicate` reports the posterior draw count used for each
replicate's rank calculation, and per-replicate `posterior_draws` repeats that
count locally. `prior_predictive_draws_per_replicate` reports the number of
prior-predictive site draws used to simulate each truth/data set, and
per-replicate `prior_predictive_draws` repeats that count locally.
Top-level and per-replicate `seed_schedule` report the deterministic
coefficients used to derive each replicate's prior and sample seeds from the
top-level seed and replicate index. It reports the rank data directly and does
not emit an aggregate uniformity/pass/fail field.
`report_kind` is `simulation_based_calibration_rank_facts`, and `report_scope`
is `replicated_simulated_datasets` so the report is not mistaken for an
aggregate uniformity decision.
Parameter objects preserve the IR `free_values` packing order, and
`parameter_order` records that order explicitly at the aggregate and
per-replicate levels. If a packed free value has no directly simulated truth
value, `sbc` returns a JSON
repair error instead of emitting a partial rank report.
Top-level `replicate_workflow_phases` and per-replicate `workflow_phases` use
the same end-to-end command path as `recover` for each replicate.

The wasm request handler accepts the same runtime command set (`sample`,
`diagnose`, `prior-predictive`, `recover`, and `sbc`, plus scalar
`diagnostics`) and returns the same v0-provisional text artifacts. Request
errors return the same v0-provisional repair object shape as CLI stderr.
Request schemas live with the handler in `crates/core/src/protocol.rs`; request
and settings objects reject unknown or duplicate control fields, while
model/data documents keep their own IR/data validation. Wasm only moves JSON
bytes across the boundary.

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
