# Invariants

Core invariants that should remain true as the codebase changes.

## Scope

- Public workflow: define model -> bind data -> sample with NUTS.
- NUTS is the only inference algorithm.
- BlackJAX is internal.

## Declaration language

- `Param`, `Data`, and `Observed` are declarations.
- `Param(...)` is latent and contributes a prior term.
- `Data.scalar()`, `Data.vector(...)`, `Data.matrix(...)`, and
  `Data.array(...)` are known inputs with shape/rank schemas and contribute no
  log-density term.
- `Observed(...)` is known input and contributes a likelihood term.
- `PartiallyObserved.vector(...)` contributes one continuous log-density factor
  over an assembled vector whose observed coordinates are fixed data and whose
  missing coordinates are free NUTS values.
- A model has one or more stochastic declarations: `Param`, `Observed`, or
  `PartiallyObserved`.
- `Observed` nodes are optional; prior-only models are valid.
- Declaration aliases are invalid: one declaration object maps to one class
  attribute name.
- Declaration classes have no base class other than `object`. A model's meaning
  is local to one decorated class body so that model text stays statically
  parseable by a standalone validator; inheritance is rejected at `@model` time
  even when the base class carries no declarations.

## Phase boundaries

- Class-body syntax capture and resolved model metadata are separate phases.
- Class-body arithmetic, indexing, and supported `jaxstanv5.math` helpers create
  private deferred syntax, never final expression IR.
- Declaration expressions support Python scalar literals as constants; fixed
  non-scalar inputs must be represented explicitly as shaped `Data` declarations.
- Non-scalar fixed distribution parameters in model declarations are invalid;
  they must enter through named `Data` declarations.
- Distributions with symbolic declaration parameters must expose those fields as
  dataclass fields. Opaque non-dataclass distributions may contain only concrete
  parameters.
- Raw JAX/NumPy functions are not declaration-language operations; supported
  symbolic math functions cross the declaration boundary through explicit helper
  nodes.
- `_resolve_model_declaration(...)` is the only transition from declaration
  symbols to named references.
- `bind(...)` is the transition from resolved model class to `BoundModel`.

## IR and module boundaries

- `_deferred.py` is private class-body syntax capture.
- `core.py` does not construct final expression IR.
- `expr.py` is resolved/final IR only.
- Final expression trees contain no declaration symbols, raw declarations,
  deferred syntax tokens, or raw Python tuple/slice indexes.
- Final expression trees may contain explicit unary operation nodes only for
  supported declaration-language unary operations such as `neg`, `exp`, and `sigmoid`.
- `ModelMeta` contains resolved metadata only, including resolved data schemas,
  free NUTS values, and stochastic log-density sites.
- `ModelMeta` is the serialization boundary: `jaxstanv5.ir` round-trips resolved
  metadata only, executes no user code on decode, and uses only the standard
  library.
- Serialized IR node tags, not Python class names, are the wire contract; tag,
  field, or encoding changes require regenerated golden files and a format
  version decision (see `docs/ir-format-v1.md`).
- A model reconstructed with `bindable_from_meta(...)` is indistinguishable
  downstream from one produced by `@model`.
- In serialized `ModelMeta`, `free_values` defines flat NUTS state layout,
  `stochastic_sites` defines log-density factors, and `data` plus
  `observed_nodes` define required bind inputs.
- `BoundModel` contains no inference logic.

## Log density

- The compiler evaluates symbolic distribution arguments before calling
  `Distribution.log_prob(...)`.
- Log density = constraint Jacobians + all stochastic site log-density terms.
- The flat unconstrained parameter vector packs free values in the insertion
  order of `free_values`, or of `params` when `free_values` is empty.
- `Param`, `Observed`, and `PartiallyObserved` lower to stochastic sites plus a
  free/fixed coordinate partition; they are not separate log-density machinery.
- Discrete distributions are observed-likelihood distributions only; NUTS
  parameters are continuous and discrete latent `Param(...)` or
  `PartiallyObserved(...)` declarations are rejected.
- Sampling returns constrained parameter values.
- Public sampler count arguments (`num_chains`, `num_warmup`, and `num_samples`)
  must be at least 1 and are validated before backend execution.
- Sampling result arrays have shape `(num_chains, num_samples, *param_shape)`,
  including declared zero-sized free values.
- `rhat(...)` and `ess(...)` omit zero-sized sample arrays because they have no
  scalar coordinates to reduce.
- NUTS diagnostics are recorded separately for warmup and post-warmup sampling.
- Diagnostic arrays have shape `(num_chains, num_steps)`, where `num_steps` is
  `num_warmup` for warmup diagnostics and `num_samples` for sampling diagnostics.

## Simulation

- Prior and prior-predictive simulation distinguish iid sample dimensions,
  distribution batch dimensions, and event dimensions.
- Distribution samples have shape `iid_sample_shape + batch_shape + event_shape`.
- Discrete observed distributions may be sampled for prior-predictive
  simulation, but discrete latent parameters remain unsupported.
- A model parameter's resolved `param_shape` is the full constrained value shape,
  not necessarily the iid sample shape passed to a distribution.
- Prior simulation derives iid sample shape by removing the distribution's
  `batch_shape + event_shape` suffix from the resolved model value shape.
- Event-shaped priors, such as `MultivariateNormal`, are valid when their event
  shape is part of the resolved model value shape.
- `Interval(lower, upper)` and `UnitInterval()` represent finite open intervals
  only. Their transforms are explicit logit/scaled-logit bijections and their
  inverse-transform Jacobians are part of compiled latent log densities.
- Interval-constrained prior simulation is limited to scalar-event inverse-CDF
  distributions unless explicit multivariate constrained simulation is added.
- Ordered-vector prior simulation is limited to iid scalar-event priors. It samples
  constrained values by sorting iid prior draws along the last axis; it does not
  use the ordered unconstraining transform or a Jacobian because simulation draws
  directly from the normalized constrained prior.
