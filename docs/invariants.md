# Bayesite invariants

Core invariants that should remain true as Bayesite evolves.

## Project scope

- Bayesite is a single-static-binary, agent-operable Bayesian workflow engine.
- The default agent path is one downloaded binary: no Python, no `uvx`, no
  NumPy, and no runtime dependency graph.
- Bayesite consumes serialized model IR. It is not currently a model declaration
  frontend.
- Posterior sampling uses NUTS only unless a deliberate design decision expands
  scope.
- The core remains SQLite-like: small, auditable, deterministic, embeddable,
  offline-capable, and suitable for sandboxed use.
- The Rust core remains zero-dependency unless a written design decision waives
  that invariant. `cargo tree` showing only `bayesite-core` is intentional.
- WebAssembly is first-class. A wasm build failure is a project failure.

## Workflow surface

- The intended command set is `sample`, `diagnose`, `generate`,
  `prior-predictive`, `posterior-predictive`, `posterior-check`, `simulate`, `recover-check`,
  `recover`, and `sbc`.
- CLI stdout/stderr are machine-readable. Errors are JSON objects with typed
  error names and repair-oriented messages.
- Workflow commands may orchestrate artifacts, files, and multiple runtime calls;
  they must not hide probabilistic state transitions inside convenience behavior.
- Artifact and stream formats are versioned or explicitly marked provisional.
  Consumers must not treat provisional formats as stable without an intentional
  format decision.
- Seeds are explicit user inputs. Re-running the same model, data, settings,
  seed, and chain id reproduces the same runtime behavior across supported
  targets, including wasm.
- Functional generation accepts exactly fixed parameters, one closed model's
  prior, or a model/data-compatible posterior source. It redraws the selected
  source once per dataset and emits one natural-scale parameter/complete-data
  pair per draw in one bounded core invocation.
- Prior-predictive simulation claims exactly one declaration-backed stochastic
  site for every Param, Observed, and non-Param free value before drawing.
  Additional density factors are rejected because Bayesite has no factor-aware
  ancestral sampler; they are never independently generated or silently
  discarded.
- Forward simulation derives a stable ancestral execution plan from expression
  dependencies. Original stochastic-site index breaks ready-site ties and
  remains the factor/artifact order; planning never rewrites decoded metadata.
- A generated `PartiallyObserved` value retains separate missing-coordinate and
  complete-vector representations. Descendants evaluating that owner's
  structurally identical scatter consume the complete generated vector without
  mutating declared conditioning data; distinct scatter expressions retain
  their own fields.

## IR boundary

- The serialized IR document is the model boundary. Decoding executes no
  producer or user code.
- Current v1 compatibility uses the historical envelope
  `{"bayeswire_ir": 1, "model": ...}`. Renaming the envelope is a versioned
  format decision, not a cleanup. The two envelope fields each appear exactly
  once; extra or duplicate envelope fields are malformed.
- Node tags and field lists are the wire contract. Producer class names are not
  the contract.
- Entry-array order is semantic everywhere and must never be normalized,
  sorted, or deduplicated silently.
- `free_values` defines the flat unconstrained NUTS state layout. If it is
  empty, `params` is the legacy layout source.
- `stochastic_sites` defines log-density factors and their value expressions.
  If it is empty, legacy parameter and observed sites may be derived from
  `params` and `observed_nodes`.
- `data` plus `observed_nodes` define required bind inputs.
- Consumers hash received bytes for provenance; they do not reserialize just to
  hash.
- Documents outside the core tag profile fail explicitly, normally with
  `UnknownNodeTag`.
- Tag, field, or encoding changes require a deliberate golden-fixture diff and
  an IR version decision.

## Runtime phase boundaries

Bayesite keeps these phases explicit:

1. parse JSON
2. decode IR
3. bind data
4. construct posterior evaluation state
5. evaluate log density and gradient
6. run NUTS
7. emit draws, diagnostics, or workflow artifacts

- The core runtime is pure: no hidden filesystem access, no hidden clock access,
  no OS entropy, no network, and no global mutable sampler state.
- Parallelism belongs to callers and shells, such as CLI chain threads or web
  workers. Core evaluation remains single-runtime-state and deterministic.
- CLI and wasm are thin boundaries around the same runtime semantics.

## Log density and state layout

- The flat unconstrained vector is split strictly in `free_values` order.
- Constraint inverse transforms and their log-Jacobians are part of the latent
  log density.
- Log density equals constraint Jacobians plus all stochastic-site log-density
  terms in IR order.
- Symbolic distribution arguments are evaluated before distribution log-density
  evaluation.
- NUTS state is continuous. Discrete distributions are supported as fixed-data
  likelihood factors, not as latent NUTS coordinates.
- Sampling outputs constrained parameter values.
- Diagnostics are recorded separately for warmup and post-warmup sampling.

## Numerical and portability contract

- JSON parsing/writing, PRNG, special functions, autodiff, tensor operations,
  NUTS, and diagnostics are in-tree and reviewable.
- Ported or reimplemented numerical routines keep provenance in `NOTICE` and at
  the implementation site.
- Special functions are pinned against committed high-precision fixtures.
- PRNG streams are pinned against reference vectors and chain streams are
  deterministic and distinct.
- Split R-hat and ESS behavior is tested against committed fixtures.
- The wasm ABI is the only allowed unsafe boundary and only moves bytes across
  linear memory. `#![deny(unsafe_code)]` remains crate-wide otherwise.

## Fixture and spec snapshots

- `tests/golden_ir/` is the vendored bayeswire conformance corpus for the
  pinned snapshot.
- Bayesite vendors the spec docs and corpus as explicit snapshots from the
  bayeswire repository at the commit recorded in `BAYESWIRE_TAG`
  (`scripts/vendor_bayeswire.py`); `bayeswire-vendor.json` records a sha256
  per vendored file and the validation ladder verifies the bytes on every
  run. IR syncs are visible commits that update the pin, the manifest, and
  the files together.
- The neutral spec home exists: bayeswire owns the normative wire format.
  Only the copies in bayeswire are normative; the copies here are generated
  and must never be edited by hand.
