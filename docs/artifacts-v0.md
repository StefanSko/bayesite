# Bayesite v0-provisional artifacts

Bayesite workflow artifacts are intentionally **v0-provisional**. They are
machine-readable and tested, but not a stable public artifact format yet.
Consumers must check the relevant format marker before parsing.

This document covers workflow artifacts only. It does not change the model IR
wire format, which remains `{"jaxstanv5_ir": 1, "model": ...}`.

## Format markers

Every agent-facing workflow artifact or error uses an explicit marker:

| Surface | Marker |
|---|---|
| `sample` fit stream | `draws_format: "v0-provisional"` |
| `diagnose` report | `diagnostics_format: "v0-provisional"` |
| `prior-predictive` stream | `prior_predictive_format: "v0-provisional"` |
| `recover` report | `recover_format: "v0-provisional"`, `workflow_format: "v0-provisional"` |
| `sbc` report | `sbc_format: "v0-provisional"`, `workflow_format: "v0-provisional"` |
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
- Counts, orders, coordinate orders, artifact kind/scope, seed metadata, and
  index-base labels are emitted explicitly where they disambiguate streamed
  records.
- Parameter summaries use constrained parameter values, not unconstrained NUTS
  state values.
- R-hat/ESS values that are unavailable for short or degenerate chains are JSON
  `null`, never `NaN`, `Infinity`, or a late serialization failure.
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

Each draw line includes:

- the same draw format and artifact identity
- zero-based retained `draw_index` and `draw_index_base`
- seed, total retained `draw_count`, chain count/order, chain id,
  `chain_index_base`, per-chain draw index, parameter count/order
- constrained parameter values keyed by parameter name

The trailer includes:

- the same draw format and artifact identity
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

`diagnose` accepts older v0 streams without some optional metadata, but validates
metadata when it is present. It rejects mismatched counts, non-contiguous draw
indexes, malformed chain/order metadata, and lines after the trailer.

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
- `recover` is a single-scenario factual report, not repeated-scenario coverage
  validation.
- `sbc` reports ranks and histograms but no uniformity verdict or p-value.
- All workflow artifacts remain v0-provisional until an explicit artifact-format
  stabilization decision.
