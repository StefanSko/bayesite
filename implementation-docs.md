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
## Surprise: reference backend folds support edges into missing bound sides

Reading `bayesjax/_backends/jax/binding.py` at the v0.4.0 tag (to mirror
bind-time validation) surfaced semantics **not stated in #26 or the spec
changelog**, pinned only by the reference implementation:

- **Support-edge folding**: a missing `VectorBounds` side is filled with the
  base distribution's finite support edge before the transform is chosen.
  Exponential/HalfNormal base → lower edge 0; Beta → (0, 1); Uniform →
  (low, high) evaluated from data and aligned via `missing_idx` (full-vector
  parameters are indexed down to missing order); everything else → no edges.
  Consequence: an upper-only bound on an Exponential base becomes a
  *two-sided* interval `[0, ub]` — a different transform (sigmoid, not
  `ub - exp(u)`), hence a different density parameterization. The golden
  fixtures do not cover this combination, so only reading the reference
  caught it; a naive implementation would have silently diverged cross-engine.
- **Support validation** (after folding): all present bound values must lie
  within the base support edges; for bounded support additionally
  `lower < support_upper` and `upper > support_lower` (leave mass inside);
  `lower < upper` elementwise after folding.
- **nextafter clip**: the two-sided inverse transform clips the constrained
  value to `(nextafter(lb→ub), nextafter(ub→lb))` so saturated sigmoids can
  never land exactly on a bound (protects bounded-support base logpdfs, e.g.
  Beta with alpha < 1, from -inf/NaN at the boundary). Lower/upper-only
  transforms are unclipped.
- Bound-side resolution rules: referenced data must exist, be rank-1, have
  length equal to the (rank-1) free value's length, and be entirely finite;
  the free value itself must be rank-1.

## Forward-sampling semantics pinned by the reference (for WP3)

- The engine's prior-predictive currently rejects VectorScatter sites outright
  ("only ParamRef and DataRef sites are supported in v0-provisional output") —
  the "directly assignable sites only" restriction #23 flagged. #26 mandates
  supporting them, so WP3 adds a VectorScatterOp arm to the site dispatch, which
  incidentally enables prior-predictive for *unbounded* partially-observed
  models (e.g. partially_observed_mvn) too.
- Reference (`simulation/core.py`) draws the FULL vector fresh from the base
  distribution (observed data values are not inserted — it is prior-predictive),
  then overwrites missing_idx coordinates with restricted draws, and records
  the assembled vector under the site name among observed values; the free
  value never appears as a parameter draw.
- Restricted draws: Exponential + lower-only uses the memorylessness shift
  `lower + Exp(rate)` (exact even at extreme bounds); every other scalar
  inverse-CDF base uses a plain CDF-space uniform on [cdf(lb), cdf(ub)] then
  inverse CDF. Notably the reference does NOT use complementary/CCDF forms, so
  extreme-tail Normal restrictions share its float64 saturation limitation —
  we mirror it anyway: cross-engine parity beats a unilateral "improvement".
- Bounded MVN forward sampling raises TypeError in the reference; the engine
  mirrors with an explicit unsupported error. Direct ParamRef sites with
  VectorBounds are likewise "not implemented" (mirrors domains.py).

## WP2 review notes (transforms landed)

- Pi's generalization came out cleaner than expected: the old scalar
  `interval_constraint` now routes through a shared `bounded_constraint`
  taking tape-valued bounds, so Interval/UnitInterval/VectorBounds-two-sided
  are literally one code path; the nextafter clip is a flag only the
  VectorBounds arm sets (the reference doesn't clip scalar Interval either —
  faithfulness preserved down to that asymmetry).
- A new `ResolvedConstraint` enum (bind-time twin of `Constraint`) keeps
  DataRef resolution out of the hot path: `apply_constraint` never touches the
  data map. This was the design freeze's intent and it fell out naturally.
- The tape already had `ge`/`le`/`where_select`, so the clip needed no new
  autodiff ops; clipped coordinates get gradient 0 through `where_select`,
  matching JAX `clip` semantics.
- Hand-checked the folding regression test: upper-only bound 2.0 on an
  Exponential(1) base folds to [0, 2]; at u = 0, logp = -1 - ln 2 and
  d(logp)/du = -0.5, both verified analytically. This is the case no golden
  fixture pins — it exists purely because reading the reference surfaced
  folding.
- Full suite green on first Pi attempt: 8-fixture golden gate (logp rtol
  1e-12, gradient rtol 1e-10) including both censored fixtures; clippy and
  fmt clean. Zero review iterations needed.

## Design freeze for the Pi work orders
  - `Constraint::VectorBounds { lower: Option<String>, upper: Option<String> }`
    (DataRef names), at least one present; legal on free values.
  - Generalize interval machinery to tape-valued bounds (constants for scalar
    literals, data tensors for DataRefs) — one code path, no special case.
  - Truncated's normalized logp is off-limits from the VectorBounds density
    path; predictive reuses scalar CDF/inverse-CDF primitives only.
  - Bounded-MVN unsupported error scoped to forward sampling only.
