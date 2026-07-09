# VectorBounds implementation notes (issue #26)

Running log of interesting / surprising developments during implementation.
Work is split pi-first: Claude freezes specs and reviews diffs; Pi (GPT-5.6 Sol)
executes scoped work orders.

## Phase 0 — vendor bayeswire v0.4.0 (2026-07-09)

- Vendored bayeswire v0.4.0 (`8507ba9c`) via `scripts/vendor_bayeswire.py` from a
  detached worktree of the bayescycle monorepo at the tag.
- **Byte-review confirms the additive promise from #26**: all pre-existing golden
  documents, data files, and fixtures are byte-identical; only `fingerprints.json`
  / `hashes.json` gained entries, and the spec docs gained the `VectorBounds`
  changelog section plus one tag-table row (`lower` (value), `upper` (value)).
- **The spec does not prescribe the unconstrained transform** — only the
  semantics ("restrict sampler support, no truncation normalizer"). The transform
  is pinned operationally by the fixtures' oracle evaluations. Verified by hand
  (Python, machine precision, diff 0.0e0 on all 6 evaluation points):
  - lower-only (`censored_exponential`): `y_i = c_i + exp(u_i)`, log-Jacobian `u_i`.
  - both bounds (`interval_censored_normal`): `y_i = lb_i + (ub_i - lb_i) * sigmoid(u_i)`,
    log-Jacobian `log(w_i) - softplus(-u_i) - softplus(u_i)`.
  - This is *exactly* the numerically-stable form the engine's existing
    `interval_constraint` (model.rs) already uses for scalar bounds — the
    vectorized generalization can share the formula verbatim.
- **Baseline red is a single, crisp failure**: `every_golden_fixture_is_in_the_logp_gradient_gate`
  in `fixtures_eval.rs` pins the exact fixture list and noticed the two new
  fixtures. Every pre-existing fixture still passes — behavioral confirmation
  that the vendor bump is additive. The gate design (auto-discover + assert
  list) meant the new capability could not be silently ignored. Nice property.
- Design freeze for the Pi work orders:
  - `Constraint::VectorBounds { lower: Option<String>, upper: Option<String> }`
    (DataRef names), at least one present; legal on free values.
  - Generalize interval machinery to tape-valued bounds (constants for scalar
    literals, data tensors for DataRefs) — one code path, no special case.
  - Truncated's normalized logp is off-limits from the VectorBounds density
    path; predictive reuses scalar CDF/inverse-CDF primitives only.
  - Bounded-MVN unsupported error scoped to forward sampling only.
