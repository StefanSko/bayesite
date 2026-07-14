# Bayescycle/Bayesite v0-provisional artifacts

These workflow artifacts are intentionally **v0-provisional**. They are
machine-readable and tested, but not a stable public artifact format yet.
Consumers must check the relevant format marker before parsing.

The posterior run-directory contract is owned by bayescycle; Bayesite is one
producer/consumer of the current wire shape. This document records the Bayesite
CLI's implementation of that shared contract.

This document covers workflow artifacts only. It does not change the model IR
wire format, which is `{"bayeswire_ir": 1, "model": ...}`.

## Format markers

Every agent-facing workflow artifact or error uses an explicit marker. Plain
Bayesite data documents, including `simulate` output, intentionally do not:

| Surface | Marker |
|---|---|
| `sample` fit stream | `draws_format: "v0-provisional"` |
| `diagnose` report | `diagnostics_format: "v0-provisional"` |
| `prior-predictive` stream | `prior_predictive_format: "v0-provisional"` |
| `generate` paired stream | `generated_datasets_format: "v0-provisional"` |
| `posterior-predictive` stream | `posterior_predictive_format: "v0-provisional"` |
| `posterior-check` report | `posterior_check_format: "v0-provisional"`, `workflow_format: "v0-provisional"` |
| `simulate` data document | no marker; plain data document accepted by `sample` |
| `recover-check` report | `recover_check_format: "v0-provisional"` |
| `recover` report | `recover_format: "v0-provisional"`, `workflow_format: "v0-provisional"` |
| `sbc` report | `sbc_format: "v0-provisional"`, `workflow_format: "v0-provisional"` |
| `capabilities` document | `capabilities_format: "v0-provisional"` |
| CLI/protocol errors | `error_format: "v0-provisional"` |

The marker means the artifact is intentionally provisional. Do not build a
load-bearing consumer that assumes field stability without an explicit format
decision.

## Shared conventions

- Successful artifact commands write JSON/NDJSON to stdout or `--out`; stderr is
  empty.
- Errors are one JSON object on stderr or across the wasm/native protocol:
  `{"error_format":"v0-provisional","error":"<Kind>","message":"..."}`.
- Messages are repair-oriented: they name the field/path or command shape to
  change.
- Counts, orders, coordinate orders, artifact kind/scope, seed metadata,
  optional model/data fingerprints, and index-base labels are emitted explicitly
  where they disambiguate streamed records.
- Parameter summaries use constrained parameter values, not unconstrained NUTS
  state values.
- R-hat/ESS values that are unavailable for short or degenerate chains are JSON
  `null`, never `NaN`, `Infinity`, or a late serialization failure.
- When present, `model_data_fingerprint` is `sha256:` plus the SHA-256 digest of
  `b"bayescycle-model-data-v1\n" + model_file_bytes + b"\n" + data_file_bytes` —
  the exact bytes of the model and data files as received, never a
  re-serialization. Verifiers (`posterior-predictive`, `posterior-check`) hash
  the files they are handed and compare; a byte-level change to either file
  invalidates the fit. The normative definition lives in the bayeswire spec
  (`model-data-fingerprint-v1.md`).
- Data documents are accepted in the canonical wrapped form
  `{"format": "bayescycle.data.json.v1", "variables": {...}}` (what bayescycle
  writes at `run/data.json`) as well as the bare `{name: {dtype, shape,
  values}}` map. The `format` key is reserved: any other value fails
  explicitly. `dtype: "bool"` binds as integer-valued 0/1 and requires JSON
  booleans in `values`.
- Report objects are factual records. They do not add recovery, sampler-quality,
  or SBC uniformity verdicts.

## `bayesite sample`

`sample` emits a v0-provisional NDJSON posterior-draw stream.

Shape:

1. header object
2. one draw object per retained draw
3. trailer object as `{"trailer": {...}}`

Header facts include:

- `draws_format`
- artifact identity: `artifact_kind: "posterior_draws"` and
  `artifact_scope: "observed_data_conditioned_parameter_draws"`
- workflow phases from JSON parse through artifact emission
- parameter shapes, packing order, `parameter_order`, `parameter_count`, and
  zero-based row-major `coordinate_order`
- sampler settings, seed, chain count/order, and retained draw count
- `sample_stats_mode: "per_draw_v2"`, announcing that every draw line carries
  per-draw sampler statistics (`diverging`, `tree_depth`, `tree_accept`,
  `energy`). The Bayesite CLI also emits `model_data_fingerprint` when it can
  fingerprint the exact model/data inputs. Older v0 streams with `per_draw_v1` carry the first three fields
  but no `energy`; older streams without this header field are still accepted
  by `diagnose`; when absent, per-draw stats are unavailable.

Each draw line includes:

- the same draw format and artifact identity
- zero-based retained `draw_index` and `draw_index_base`
- seed, total retained `draw_count`, chain count/order, chain id,
  `chain_index_base`, per-chain draw index, parameter count/order
- constrained parameter values keyed by parameter name
- per-draw sampler statistics: `sample_stats_mode: "per_draw_v2"`,
  `diverging` (bool), `tree_depth` (integer in `0..=max_treedepth`),
  `tree_accept` (float in `[0, 1]`, the trajectory mean Metropolis
  acceptance), and `energy` (finite float, the retained draw's Hamiltonian
  energy including potential plus sampled kinetic energy). The
  `sample_stats_mode` marker and these four fields are present together on
  every new draw line. Older `per_draw_v1` draw lines contain only
  `diverging`, `tree_depth`, and `tree_accept`.

The trailer includes:

- the same draw format and artifact identity
- optional model/data fingerprint metadata when emitted by the producer
- completion metadata: seed, draws per chain, chain count/order, total retained
  draw count, parameter count/order
- per-chain raw sampler facts: retained draw count, divergences, tree-depth
  histogram, step size, mean acceptance
- cross-chain R-hat/ESS maps, with unavailable values encoded as `null`

## `bayesite diagnose`

`diagnose` reads a complete v0-provisional fit NDJSON stream and emits one JSON
report.

The report includes:

- `diagnostics_format: "v0-provisional"`
- diagnose workflow phases: parse fit NDJSON, validate fit artifact, recompute
  diagnostics, emit report
- source artifact facts copied or normalized from the fit: source format,
  artifact kind/scope, seed, settings, chain count/order, draw count, parameter
  count/order, parameter shapes, packing, source workflow phases, and trailer
  completion metadata presence
- per-chain sampler facts from the trailer
- recomputed per-parameter R-hat/ESS maps plus statistic and coordinate-reduction
  labels
- `source_sample_stats_mode`, `source_draw_sample_stats_metadata`, and
  `source_draw_sample_stats_energy_metadata` recording whether the source stream
  carried per-draw sample stats and whether those stats include `energy`
- a `sample_stats` group: when the source stream carried per-draw stats, one
  entry per chain with `draw_count` and a `draws` array in draw order. New
  `per_draw_v2` streams produce `{diverging, tree_depth, tree_accept, energy}`
  objects; older `per_draw_v1` streams produce
  `{diverging, tree_depth, tree_accept}` objects. When the source stream did not
  carry per-draw stats, `sample_stats` is JSON `null`

`diagnose` accepts older v0 streams without some optional metadata, but validates
metadata when it is present. It rejects mismatched counts, non-contiguous draw
indexes, malformed chain/order metadata, lines after the trailer, and
inconsistent per-draw sample stats (present on some draw lines but not others,
or present on draw lines without the header `sample_stats_mode`). It accepts
both `per_draw_v1` and `per_draw_v2` source streams; `per_draw_v2` requires a
finite `energy` number on every draw line, while `per_draw_v1` must not include
`energy`. When per-draw sample stats are present, `diagnose` cross-checks them
against the trailer chain aggregates: the per-draw diverging count must match
`divergences`, the recomputed tree-depth histogram must match
`treedepth_histogram`, and the mean of `tree_accept` must match `mean_accept`
(within 1e-9).

## `bayesite generate`

`generate` consumes one closed model, a canonical design document, one explicit
parameter-source variant (`fixed`, `model-prior`, or `posterior`), a count in
`1..=1000`, and an explicit safe-integer seed. The pure core redraws the source
once per dataset and emits UTF-8 NDJSON:

1. one `generated_dataset_pairs` header with exact model/design/source hashes,
   count/seed, and ordered parameter/dataset schemas;
2. exactly one draw record per requested dataset containing natural-scale
   parameters, a complete canonical dataset, and fixed/prior/posterior source
   lineage;
3. one completion trailer repeating bounded identity, source, count, and seed
   facts.

Fixed values repeat by point-mass semantics while outcomes redraw. Model-prior
parameters redraw per dataset. Posterior draws are selected uniformly with
replacement and each selection receives a fresh outcome draw. Models with
non-Param free values, score-only factors, or non-assignable stochastic sites
fail explicitly; factors are never dropped. The operation is deterministic for
fixed bytes, count, and seed and performs no filesystem, entropy, or clock
access in the core.

## `bayesite prior-predictive`

`prior-predictive` emits a v0-provisional NDJSON stream over decoded IR and
provided declared data inputs. Observed values to be simulated must be omitted
from the input data object.

Shape:

1. header object
2. one generated draw object per requested draw
3. trailer object as `{"trailer": {...}}`

Header/trailer facts include:

- `prior_predictive_format: "v0-provisional"`
- artifact identity: `prior_predictive_draws` over
  `declared_data_conditioned_site_draws`
- workflow phases from JSON parse through artifact emission
- seed, settings, draw count, draw-index base
- declared data count/order, shapes, coordinate order, integer flags, and
  values
- generated site count/order, source stochastic-site names, roles, shapes,
  coordinate order, and integer-by-coordinate flags

Each draw line includes the same format/artifact identity, draw index metadata,
seed, counts/orders, and generated values keyed by site name.

Current scope: directly assignable stochastic sites only. In practice, v0 prior
predictive supports stochastic sites whose value expression is a parameter or a
data reference. Non-assignable stochastic-site expressions fail with a typed
repair error.

## `bayesite posterior-predictive`

`posterior-predictive` reads a complete v0-provisional fit NDJSON stream,
conditions on each retained constrained posterior parameter draw, and emits a
v0-provisional NDJSON stream of replicated observed values.

Shape:

1. header object
2. one replicated-observation draw object per retained posterior draw
3. trailer object as `{"trailer": {...}}`

Header/trailer facts include:

- `posterior_predictive_format: "v0-provisional"`
- artifact identity: `posterior_predictive_draws` over
  `observed_data_conditioned_replicated_observed_data_draws`
- workflow phases from JSON parse through artifact emission
- posterior-predictive seed, source fit seed, source draw count/order, declared
  data metadata, and generated observed-site metadata

Each draw line includes the same format/artifact identity, draw index metadata,
source fit draw provenance, generated site count/order, and generated replicated
observed values keyed by observed data name.

Current scope: directly assignable observed stochastic sites only. In practice,
v0 posterior predictive supports observed stochastic sites whose value
expression is a data reference.

## `bayesite posterior-check`

`posterior-check` generates posterior-predictive replicates and emits one
factual JSON report comparing observed data statistics to replicated statistics.

The report includes:

- `posterior_check_format` and `workflow_format`, both `v0-provisional`
- factual identity: `report_kind: "posterior_predictive_check_facts"` and
  `report_scope: "observed_data_vs_posterior_predictive_replicates"`
- posterior-predictive artifact provenance and generated draw count
- observed site order/count metadata
- built-in discrepancy summaries for each observed site: mean, standard
  deviation, minimum, maximum, and zero count for integer-valued observed data
- tail-count facts for each discrepancy, not pass/fail verdicts

`posterior-check` does not emit an aggregate model-fit verdict, pass/fail result,
or recommendation.

## `bayesite simulate`

`simulate` uses a decoded simulation model, declared input data, supplied
constrained free-value truth, and an explicit seed to generate observed data.
It writes a normal typed Bayesite data document, not a special simulation
artifact, so `sample` does not need to know whether the data was simulated.

Example:

```sh
bayesite simulate \
  --model generator.json \
  --data fixed_inputs.json \
  --truth truth.json \
  --seed 1 \
  --out generated_data.json
```

The `truth` document is keyed by free-value name and may use the same scalar,
array, or typed dtype/shape/values conventions as data documents. `simulate`
requires truth for every free value, rejects unknown truth keys, validates
free-value shape and constraints, and then simulates directly assignable
observed `DataRef` stochastic sites. Generated output contains declared inputs
first and generated observed values after them in stochastic-site order.

Current scope: directly assignable observed data sites only. Non-assignable
observed value expressions fail with a typed repair error. Non-observed prior
factors need not be directly simulatable because fixed truth is already
supplied.

## `bayesite recover-check`

`recover-check` compares a complete v0-provisional posterior fit stream to
supplied reference truth values. It does not need model, data, or simulation
provenance; it only needs posterior draws and truth.

Example:

```sh
bayesite recover-check \
  --fit fit.jsonl \
  --truth truth.json \
  --targets targets.json \
  --interval 0.8 \
  --out recovery_check.json
```

Without `--targets`, every truth key must have the same name as a posterior
parameter, and extra posterior parameters are ignored. With explicit targets,
renamed same-shape comparisons are supported:

```json
{
  "targets": [
    {"name": "alpha_recovery", "truth": "alpha_true", "posterior": "alpha"}
  ]
}
```

The report includes:

- `recover_check_format: "v0-provisional"`
- source posterior artifact kind/scope, chain count/order, and draw count
- target count/order and same-shape target metadata
- supplied truth values and `truth_source: "supplied_truth"`
- posterior means, equal-tailed interval bounds, ranks, exact tie counts,
  interval containment facts, R-hat/ESS, coordinate order, and statistic labels

`recover-check` does not emit pass/fail, verdict, coverage, interpretation, or
recommendation fields.

## `bayesite recover`

`recover` reads a v0-provisional scenario, simulates one truth/data set through
the prior-predictive path, fits the generated observed data with NUTS, and emits
one factual JSON report.

The scenario shape is:

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

The report includes:

- `recover_format` and `workflow_format`, both `v0-provisional`
- factual identity: `report_kind: "parameter_recovery_facts"` and
  `report_scope: "single_simulated_dataset"`
- simulation count/order metadata for the single v0 simulation
- seed schedule, prior-predictive source artifact identity, posterior draw
  artifact identity, declared-data facts, generated-observed facts, and sampler
  settings
- parameter order, per-parameter truth, ranks, exact tie counts, posterior
  means, equal-tailed interval bounds, interval containment facts, R-hat/ESS,
  coordinate order, and artifact provenance
- aggregate sampler counts and per-chain raw sampler facts

`recover` does not emit an aggregate pass/fail verdict, coverage claim,
interpretation, or recommendation.

## `bayesite sbc`

`sbc` reads a v0-provisional scenario, repeats the pure runtime path

```text
prior -> simulate data -> sample posterior -> rank true parameter among draws
```

and emits one factual JSON rank report.

The scenario shape is:

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

The optional CLI `--replicates` flag overrides the scenario count.

The report includes:

- `sbc_format` and `workflow_format`, both `v0-provisional`
- factual identity: `report_kind: "simulation_based_calibration_rank_facts"` and
  `report_scope: "replicated_simulated_datasets"`
- aggregate replicate count/order, seed schedule, rank support metadata,
  declared-data facts, generated-observed provenance per replicate, posterior
  draw artifact identity, and aggregate sampler counts
- aggregate per-parameter rank arrays, tie-count arrays, simulated truths,
  rank histograms, coordinate order, and provenance
- per-replicate reports with generated observed data, parameter rank facts,
  R-hat/ESS, chain order, per-chain raw sampler facts, and local sampler summary

`sbc` does not emit an aggregate uniformity, pass/fail, or sampler-quality
verdict. Consumers decide how to interpret the reported facts.

## Current limitations

- Prior predictive supports directly assignable stochastic sites only.
- Posterior predictive supports directly assignable observed stochastic sites only.
- `posterior-check` has only built-in generic discrepancy summaries; no custom
  discrepancy language yet.
- `recover` is a single-scenario factual report, not repeated-scenario coverage
  validation.
- `sbc` reports ranks and histograms but no uniformity verdict or p-value.
- All workflow artifacts remain v0-provisional until an explicit artifact-format
  stabilization decision.
