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
- The release `bayesite` CLI binary must build as the default agent artifact.
- The wasm target must build.

### G1 — Decode and protocol contract

Bayesite must decode every golden IR fixture and reject malformed documents with
typed, repair-oriented JSON errors, including unexpected fields in the v1 IR
envelope, duplicate envelope fields, and duplicate node tag fields.

Covered by Rust tests such as `tests/ir_decode.rs`, `tests/protocol.rs`, and
`tests/cli.rs`. The CLI gate includes the endgame `sample --out` artifact path
and verifies that the resulting fit stream reports workflow phases and can be
consumed by `diagnose`. The `diagnose --out` path is also pinned, and the report
echoes source fit metadata before recomputing diagnostics and reports its own
replay workflow phases. Diagnose reports R-hat/ESS statistic and
coordinate-reduction labels beside the recomputed diagnostic maps. The fit stream trailer reports the same provisional draw
format marker as the header plus completion metadata, and `diagnose` rejects
mismatched trailer metadata when it is present while reporting which optional
trailer completion fields were present. Diagnose reports source artifact
kind/scope as factual metadata when present and keeps legacy streams without
that metadata accepted. Diagnose reports normalized source chain count,
validates optional header/trailer `chain_count` metadata when present, and keeps
legacy streams without that metadata accepted. Sample CLI and protocol tests pin
explicit fit-stream `chain_order`, and `diagnose` validates optional
header/trailer `chain_order` metadata when present while reporting normalized
`source_chain_order`. Native protocol tests cover the
same fit-artifact diagnostic path through the wasm-boundary request handler.
Sample and diagnose tests pin zero-based row-major parameter `coordinate_order`
in fit headers and diagnostic `source_params`. Sample CLI and protocol tests pin explicit
fit-stream `parameter_order`, and `diagnose` validates optional header/trailer
`parameter_order` metadata when present while reporting normalized
`source_parameter_order`. Sample CLI and protocol tests pin the fit stream
artifact kind/scope as `posterior_draws` over
`observed_data_conditioned_parameter_draws` in both header and trailer. CLI and
protocol tests pin explicit sample `chain_count` in both header and trailer
while preserving the legacy provisional `chains` fields. Sample trailer chain
stats report per-chain `draw_count` as raw retained-draw metadata, and
`diagnose` validates that optional metadata when present. Sample artifacts also
report explicit total retained `draw_count` and `parameter_count` in both header
and trailer while preserving the legacy provisional `draws_per_chain`, `chains`,
`packing`, and `params` fields; `diagnose` reports normalized source draw and parameter
counts and validates optional `draw_count` and `parameter_count` metadata when
present. Sample draw lines report the same provisional format marker and
artifact kind/scope as the header, plus a zero-based retained-order
`draw_index`, `draw_index_base`, seed, total retained `draw_count`,
`chain_count`, `chain_order`, and `chain_index_base`;
`diagnose` validates optional draw-line `draw_index` metadata when present while
keeping legacy draw lines without it accepted. Diagnose reports
`source_draw_index_metadata` so consumers can distinguish indexed sample
streams from legacy streams.
Sample draw lines also report `parameter_count` and `parameter_order`, and
`diagnose` validates optional draw-line parameter metadata when present while
keeping legacy draw lines without it accepted. Diagnose validates optional
draw-line artifact metadata when present, including `draw_index`,
`draws_format`, artifact kind/scope, `draw_index_base`, seed, and `draw_count`,
and validates optional draw-line chain metadata when present, including
`chain_count` and `chain_order`. Diagnose reports
`source_draw_parameter_metadata`, `source_draw_artifact_metadata`, and
`source_draw_chain_metadata` so
consumers can distinguish self-describing sample draw lines from legacy streams.
Sample trailer chain stats also report per-chain `chain_index_base` beside raw
chain ids so extracted sampler diagnostics carry the chain-id convention. CLI
and protocol tests also pin repair text for stdin-capable inputs, positive
diagnose fit replay from stdin, positive single-stdin paths for workflow commands,
positive default stdout JSON for recover and SBC with empty stderr,
v0-provisional repair format markers for CLI stderr and wasm-boundary request
errors,
duplicate direct CLI flags, missing direct CLI flag values, duplicate protocol
control fields, non-object protocol requests, non-string protocol command
fields, data artifact command object paths, reportable seed ranges,
reportable draw/warmup-count ranges,
workflow chain and replicate count ranges, workflow scenario duplicate/count
fields, workflow scenario data object paths, workflow scenario numeric field
paths, protocol workflow data object paths, protocol recover interval field
paths, and tree-depth bounds. The
default validation ladder also smoke-tests the release `bayesite` binary's
missing-arguments failure path so stdout stays empty and stderr remains one
v0-provisional JSON repair object naming the missing command while listing the
supported commands.
CLI tests also pin unknown-command repair text that names the invalid command
while listing the supported command forms. Protocol tests pin the same
unknown-command fact at the wasm-boundary request handler, listing supported
JSON command names without changing workflow semantics.

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

`bayesite prior-predictive` emits a v0-provisional NDJSON stream over decoded
IR. The stream reports workflow phases, declared data inputs,
explicit declared data order, declared data integer flags, shape-preserving
declared data coordinate order and `declared_data_integer_by_coordinate` flags,
explicit declared data count, generated site names, source
stochastic-site names, roles, shapes, zero-based row-major generated-site
coordinate order, integer flags, shape-preserving `integer_by_coordinate`
flags, explicit generated site order, generated draws, artifact kind/scope, and a trailer with format,
artifact kind/scope, phase, seed, settings, explicit `draw_count`, legacy
provisional `draws` count, explicit `site_count`, explicit `site_order`, explicit
`declared_data_count`, explicit `declared_data_order`, explicit
`draw_index_base`, and legacy provisional `sites` count metadata. Zero-draw prior-predictive requests fail
with typed JSON repair messages at CLI, protocol, and artifact-helper
boundaries. The
committed self-contained gate currently pins the
CLI/artifact contract and shape behavior on a golden linear-regression fixture,
then checks that every currently
compatible directly assignable golden fixture can emit v0 draws over the core
distribution set. Self-contained analytic tests check scalar Normal
prior-predictive draws against known mean/variance and scalar Bernoulli draws
against integer JSON artifact output and the expected mean under a fixed seed.
Native protocol tests also cover the wasm-boundary request
handler for the same command, pin declared data order and coordinate order as
explicit artifact metadata, pin integer declared-data values as JSON integers when
`declared_data_integer_by_coordinate` marks them integer, and pin integer
generated-site values as JSON integers when `integer_by_coordinate` marks the
coordinate integer. CLI and protocol tests pin generated-site
`coordinate_order`, generated `site_order`, and the phase list from `parse_json` through
`emit_artifact`, and pin `settings.num_draws` in both header and trailer as
artifact provenance. CLI and protocol tests pin `draw_count`, `site_count`,
`declared_data_count`, `declared_data_order`, and `draw_index_base` in both header and trailer as self-describing
artifact-count metadata. CLI and protocol tests also pin per-draw
`prior_predictive_format`, `artifact_kind`, `artifact_scope`, `draw_index`,
`draw_index_base`, `seed`, `draw_count`, `declared_data_count`,
`declared_data_order`, `site_count`, and `site_order` so generated draw records
remain self-describing when streamed independently without repeating full
declared data values. Broader
analytic summary checks and `jaxstanv5` reference comparisons remain future G8
conformance work.

### G9 — Recover

`bayesite recover` currently runs one v0-provisional scenario: simulate
truth/data through the prior-predictive path, fit the generated data with NUTS,
and report end-to-end workflow phases, posterior draw count, interval
construction metadata, rank draw count, interval bounds, prior-predictive
simulation draw count and artifact identity, factual report kind,
single-simulated-dataset scope, workflow format marker, simulation count, seed
derivation metadata,
declared scenario inputs, generated values and source stochastic-site
names, shapes, generated data integer flags, explicit generated observed order,
explicit declared data order,
shape-preserving declared data integer flags,
simulated truth source stochastic-site names, simulated truth integer flags,
explicit parameter order, explicit per-parameter coordinate order, posterior
intervals, aggregate and by-coordinate interval containment facts, and truth
rank/tie counts among posterior draws, sampler diagnostics, and aggregate
sampler counts. Recover reports include effective sample settings and the
effective interval probability in the `settings` object, while preserving the
legacy top-level `interval` field. Per-chain reports include kept draw counts as
facts rather than
requiring callers to infer them from settings, `chain_count` reports the emitted
chain section size, and `chain_order` records the emitted chain-id order
explicitly. It intentionally avoids an aggregate
pass/fail verdict, coverage claim, interpretation, or recommendation; callers
decide how to use the reported facts. Native workflow and protocol tests cover
the same factual report. CLI tests pin scenario seed range repair text.
Workflow tests also pin that reports fail explicitly when a packed free value
lacks simulated truth, rather than silently omitting the parameter, and that
generated observed integer flags keep the generated value shape and generated
observed order is explicit. Workflow, protocol, and CLI tests pin generated
observed coordinate order and integer-by-coordinate flags as separate
model-side serialization metadata. Protocol and CLI tests pin integer generated
observed values as JSON integers when `generated_observed_integer` marks them
integer. Workflow, protocol, and CLI tests pin declared-data coordinate order
and integer declared-data values as JSON integers when
`declared_data_integer_by_coordinate` marks them integer.
Workflow, protocol, and CLI tests pin recover generated-observed artifact
kind/scope and zero-based draw index metadata so the generated data Bayesite
fits remains tied to its prior-predictive draw.
Workflow, protocol, and CLI tests pin the recover parameter summary scale as
`constrained_parameter_value` at report and per-parameter scope so truth,
intervals, ranks, and containment facts are not interpreted as unconstrained
NUTS-state summaries.
Workflow and protocol tests pin declared data order and recover seed derivation
as explicit report metadata. Workflow, protocol, and CLI tests pin recover truth
rank/tie counts, per-parameter rank and tie-count support, the rank draw count,
explicit `rank_bin_order`, explicit `rank_bin_count`, explicit
`tie_count_bin_order`, and explicit `tie_count_bin_count` as factual fields.
Workflow,
protocol, and CLI tests pin `interval_bounds` as
self-describing interval metadata, including requested interval probability,
excluded lower and upper tail probabilities, lower/upper quantile
probabilities, sorted posterior draw count, zero-based sorted-draw index base,
and lower/upper quantile interpolation index position/floor/ceil facts.
Workflow, protocol, and CLI tests pin parameter-local interval method, scope,
and bounds beside per-parameter lower/upper interval values.
Workflow, protocol, and CLI tests pin parameter-local interval containment,
rank, tie, R-hat, and ESS statistic labels so extracted parameter summaries
remain self-describing factual records rather than verdicts.
Workflow, protocol, and CLI tests pin recover parameter-local posterior draw
count and artifact kind/scope metadata beside rank and interval facts.
Workflow, protocol, and CLI tests pin top-level posterior draw artifact
kind/scope metadata so recover reports name the posterior draw artifact being
summarized.
Workflow, protocol, and CLI tests pin recover prior-predictive simulation draw
count and artifact kind/scope metadata so the generated truth/data source is
named separately from the posterior draw artifact.
Workflow, protocol, and CLI tests pin recover parameter-local truth artifact
kind/scope and zero-based truth draw index metadata beside the simulated truth
values. Workflow, protocol, and CLI tests pin recover parameter-local
simulation index metadata so extracted parameter summaries keep their
single-simulation provenance without becoming verdicts.
Workflow, protocol, and CLI tests pin recover parameter-local prior/sample
seeds and seed schedule metadata so extracted parameter summaries keep their
simulation and posterior-fit provenance.
Workflow, protocol, and CLI tests pin `sampler_summary` as aggregate counts,
not as a recovery-quality verdict. Workflow, protocol, and CLI tests pin
per-chain `draw_count` and `chain_index_base` as raw report fields and pin explicit tree-depth
histogram bin order/count metadata for sampler summaries and chain reports.
CLI and protocol tests pin `chain_order` as explicit report metadata and the
phase list from `parse_json` through `emit_report`. Workflow, protocol, and
CLI tests pin recover `workflow_format` as a v0-provisional report marker.
CLI and protocol tests pin recover `chain_count` as explicit chain-section
count metadata. CLI and
protocol tests pin
`report_kind`,
`report_scope`, `simulation_count`, `simulation_index_base`, and
`simulation_order` so the one-scenario report is not mistaken for a recovery
verdict or repeated-scenario coverage. Workflow tests
pin zero-based row-major
`coordinate_order` for vector parameters, and protocol/CLI tests pin that the
field is present at command boundaries.
Workflow, protocol, and CLI tests pin vector-parameter
`interval_contains_truth_by_coordinate` as a shape-preserving factual field.
Workflow, protocol, and CLI tests pin `interval_contains_truth_by_parameter` as
a by-parameter factual index over the parameter-local containment facts, while
also pinning that no top-level aggregate `interval_contains_truth` verdict is
emitted.
Workflow, protocol, and CLI tests pin explicit recover section counts for
parameters, emitted parameter reports, generated observed values, and declared
data.
Repeated-scenario coverage summaries remain future G9 conformance work.

### G10 — SBC

`bayesite sbc` currently runs v0-provisional simulation-based calibration
scenarios through the pure runtime path:

```text
prior -> simulate data -> sample posterior -> rank true parameter among draws
```

It reports end-to-end replicate workflow phases, factual report kind,
replicated-simulated-dataset scope, rank bounds, rank/tie statistics, rank
scope, seed derivation metadata, declared scenario inputs, explicit declared
data order, shape-preserving declared data integer flags, explicit aggregate
and per-replicate parameter order, rank histograms, generated observed values,
source stochastic-site names, shapes, integer flags, and
explicit generated observed order, truth source stochastic-site names, truth
integer flags, truth/rank/tie-count/R-hat/ESS per parameter and replicate, and
per-chain kept draw counts, sampler diagnostics, per-parameter coordinate
order, per-replicate sampler counts, explicit per-replicate chain order,
explicit chain counts,
explicit aggregate replicate count/order, and aggregate sampler counts. It
intentionally does not emit an aggregate uniformity/pass/fail verdict; callers
decide how to interpret ranks.
Native workflow and protocol tests cover the same rank report. CLI tests pin
scenario seed range repair text. Workflow tests also pin that reports fail
explicitly when a packed free value lacks simulated truth, rather than silently
omitting the parameter, and that generated observed integer flags keep the
generated value shape and generated observed order is explicit. Workflow,
protocol, and CLI tests pin generated observed coordinate order and
integer-by-coordinate flags as separate model-side serialization metadata.
Protocol and CLI tests pin integer generated observed values as JSON integers
when `generated_observed_integer` marks them integer. Workflow, protocol, and
CLI tests pin declared-data coordinate order and integer declared-data values as
JSON integers when `declared_data_integer_by_coordinate` marks them integer.
Workflow, protocol, and CLI tests pin per-replicate generated-observed artifact
kind/scope and zero-based draw index metadata so each simulated dataset remains
tied to its prior-predictive draw.
Workflow, protocol, and CLI tests pin top-level
`generated_observed_count_per_replicate` and
`generated_observed_order_per_replicate`, plus top-level generated-observed
artifact kind/scope and draw-index metadata per replicate, so the
generated-observed section shape, site order, and prior-predictive provenance
inside each replicate are available without reading a replicate object first.
CLI and protocol
tests pin declared data order as explicit report metadata and the phase list
from `parse_json` through `emit_report`. Workflow, protocol, and CLI tests pin
`sampler_summary` as aggregate counts, not as a sampler-quality verdict.
Workflow, protocol, and CLI tests pin per-chain `draw_count` as a raw report
field, `chain_index_base`, explicit tree-depth histogram bin order/count metadata for sampler
summaries and chain reports, and per-replicate `sampler_summary` as local count
metadata.
They also pin explicit top-level `workflow_format`, explicit
`replicate_count`, explicit `replicate_report_count`,
legacy provisional `replicates`, `replicate_index_base`, `replicate_order`,
per-replicate `sbc_format`, per-replicate `workflow_format`, `report_kind`,
`report_scope`, `workflow_phases`,
per-replicate `replicate_count`,
per-replicate `replicate_index_base`, per-replicate `replicate_order`,
per-replicate `rank_draws`,
top-level `posterior_draws_per_replicate`, per-replicate `posterior_draws`,
top-level and per-replicate posterior draw artifact kind/scope metadata,
top-level and per-replicate prior-predictive simulation draw count and artifact
kind/scope metadata, top-level generated-observed count/order and artifact
provenance per replicate, `rank_bounds`, per-replicate `rank_bin_order`,
per-replicate `rank_bin_count`, per-replicate `tie_count_bin_order`,
per-replicate `tie_count_bin_count`, per-replicate `parameter_summary_scale`,
rank statistic labels, per-replicate sample `settings`, per-replicate
`seed_schedule`, per-replicate `chain_order`, and per-replicate declared data
count/order metadata. Workflow, protocol, and CLI tests pin top-level
`chain_count_per_replicate` and per-replicate `chain_count` as explicit count
metadata for emitted chain sections.
CLI and protocol
tests pin `report_kind` and `report_scope` so the SBC report is not mistaken
for a uniformity verdict. Workflow tests pin
zero-based row-major `coordinate_order` for vector parameters, and protocol/CLI
tests pin that the field is present at command boundaries. Workflow, protocol,
and CLI tests pin per-replicate parameter `shape` so extracted replicate
parameter summaries remain self-describing without inferring shape from values.
Workflow, protocol, and CLI tests pin effective `settings.replicates` so
scenario counts and CLI replicate overrides are preserved as settings
provenance, not inferred from aggregate counts.
Workflow, protocol, and CLI tests pin vector-parameter `rank_histogram` as one
rank histogram per coordinate, with each coordinate histogram spanning
`0..=rank_draws` and summing to the replicate count.
Workflow, protocol, and CLI tests pin aggregate and per-replicate parameter
rank and tie-count support metadata, including `rank_bin_order`,
`rank_bin_count`, `tie_count_bin_order`, and `tie_count_bin_count`, so rank,
tie-count arrays, and histograms can be interpreted from the parameter object
itself.
Workflow, protocol, and CLI tests pin aggregate
parameter-local `replicate_order` so aggregate rank arrays can be aligned to
replicate ids without consulting top-level report context, and pin aggregate
parameter-local `replicate_index_base`, `prior_seed`, `sample_seed`,
`seed_schedule`, and `truth` arrays so simulated truth values and the seeds
that produced each replicate remain available beside aggregate ranks with
explicit index-base metadata.
Workflow and protocol tests pin aggregate and per-replicate parameter-local
rank and tie statistic labels, and workflow, protocol, and CLI tests pin
per-replicate R-hat and ESS statistic labels, so extracted parameter summaries
remain self-describing factual records rather than verdicts.
Workflow, protocol, and CLI tests pin aggregate and per-replicate
parameter-local posterior draw count and artifact kind/scope metadata so
extracted SBC parameter summaries remain self-describing.
Workflow, protocol, and CLI tests pin aggregate and per-replicate
parameter-local truth artifact kind/scope and zero-based truth draw index
metadata so simulated truth values remain tied to the prior-predictive draw
that produced them.
Workflow, protocol, and CLI tests pin per-replicate parameter-local
prior/sample seeds and seed schedule metadata so extracted replicate parameter
summaries keep their simulation and posterior-fit provenance.
Workflow, protocol, and CLI tests pin per-replicate parameter-local `replicate`
and `replicate_index_base` metadata so extracted replicate parameter summaries
keep their zero-based simulation index provenance.
Workflow, protocol, and CLI tests pin aggregate parameter-local rank histogram
statistic and scope labels so histogram facts remain self-describing rather
than uniformity verdicts.
Workflow, protocol, and CLI tests pin the SBC parameter summary scale as
`constrained_parameter_value` at report, aggregate-parameter, and
per-replicate-parameter scope so truth, ranks, ties, and histograms are not
interpreted as unconstrained NUTS-state summaries.
Workflow, protocol, and CLI tests pin explicit SBC `rank_bin_order` and
`rank_bin_count` metadata so aggregate rank histogram bins and support size are
labeled rather than inferred. Workflow, protocol, and CLI tests pin explicit
SBC `tie_count_bin_order` and `tie_count_bin_count` metadata so exact tie-count
support is labeled rather than inferred.
Workflow, protocol, and CLI tests pin explicit SBC section counts for aggregate
parameters, emitted aggregate parameter reports, declared data,
generated-observed sections per replicate, and per-replicate parameter
report/generated-observed sections.
Broader SBC conformance, including stable
uniformity summaries over larger replicate counts, remains future G10 work.

## Default command

The self-contained ladder subset is runnable with:

```sh
python3 scripts/check_validation_ladder.py
```

Optional oracle-backed posterior comparison:

```sh
python3 scripts/check_validation_ladder.py --posterior --jaxstanv5-path ../jaxstanv5
```
