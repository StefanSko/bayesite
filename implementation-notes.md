# Implementation notes — deterministic VectorBounds owner resolution

Working log for the Bayesite half of bayescycle#56. Interesting and surprising
findings are appended as the red→green implementation proceeds.

## Design handoff (2026-07-10)

- BayesJAX is normative for the current behavior; Bayesite mirrors its
  first-reference heuristic exactly, so the Python/spec change lands first.
- The replacement owner rule is structural and order-independent: exactly one
  stochastic site shares the VectorBounds free value's name, and its value is
  either a direct same-name `Param` or a `VectorScatter` with a direct
  same-name `missing_values` parameter.
- Differently named factors may reference the free slot or duplicate its full
  scatter expression without becoming owners. This keeps a future Factor
  declaration compatible while preventing factors from silently changing the
  coordinate transform.
- Direct parameter owners matter in Rust today: the existing VectorBounds unit
  tests construct same-name `Expr::Param("y")` sites even though the current
  bayeswire declaration surface initially emits VectorBounds for
  PartiallyObserved values.

## Red phase

- Vendoring the bayescycle branch added the adversarial fixture and changed no
  pre-existing golden bytes. The fixture-discovery guard failed first, as
  designed; after admitting the new name, the old engine disagreed with the
  JAX oracle immediately at evaluation 0 (logp -2.462877... vs -4.474171...).
- Focused Rust tests reproduced all four intended failures: the earlier
  differently named scatter factor yielded `-inf` at q=0.1, while missing,
  duplicate, and expression-valued same-name owners all bound successfully.

## Green phase

- `vector_bounds_support_edges` now selects exactly one same-name site before
  inspecting its value. Direct same-name `Expr::Param` owners and scatter
  owners with direct same-name `missing_values` are accepted; every other
  same-name shape is a repair-oriented bind error.
- The broad recursive free-reference and index-reference walkers became dead
  code and were removed. Support alignment now evaluates `missing_idx` only
  for the canonical scatter owner.
- All focused owner tests and the nine-fixture JAX-oracle gate pass. The new
  fixture therefore pins both implementations to the same structural rule.
- The full validation ladder passed: zero-dependency/vendor guards, fmt,
  Clippy with warnings denied, release CLI, 332 Rust tests, wasm build, and
  the pinned nuts-rs statistical oracle (4 targets, 15 summary checks).

## Codex review round 1 — predictive owner alignment

- Codex found a real P1 outside density binding: prior-predictive dispatch
  treated every assignable VectorScatter as generative. The adversarial model
  therefore emitted independent `penalty` and `y` vectors even though posterior
  density evaluation shares one latent `y`.
- Red reproduction pinned the silent wrong artifact: one draw emitted penalty
  `[1.967989...]` and owner y `[1.268627...]` from the same free slot. General
  Factor/product-of-experts simulation has no ancestral interpretation, so the
  safe scoped behavior is an explicit unsupported error for differently named
  assignable factors over VectorBounds free values.
- First fix overreached by applying the new rule to every free value, which
  changed an existing unbounded shape-error fixture (`theta_site` targeting
  `theta`) from DataShapeMismatch to InvalidSettings. The normative owner rule
  is VectorBounds-specific, so validation was narrowed accordingly; both direct
  and scatter non-owner VectorBounds factors reject before drawing, while the
  pre-existing unbounded surface retains its established behavior.
- Post-review full validation ladder passed again, including the restored
  unbounded prior-predictive shape contract, 334 Rust tests, wasm, Clippy, and
  the 15-check nuts-rs oracle.

## Holistic prior-predictive site inventory

- Subsequent BayesJAX review examples (wrapped values, distribution references,
  and a factor reusing a declaration name) showed that reference scanning is
  not a stable way to infer whether a site is generative. The IR sequence mixes
  declaration-backed sites and arbitrary density factors; role must be resolved
  from declaration structure once, before drawing.
- New red Rust tests demonstrate the general failure without VectorBounds: an
  extra unbounded `Param("theta")` factor is independently sampled, and a
  colliding factor also named `theta` produces two unrelated draws with the
  same output name. This is the same silent wrong-answer class as the original
  adversarial scatter.
- The replacement policy claims exactly one matching stochastic site for every
  Param, Observed, and non-Param free declaration, then rejects every unclaimed
  site. Param/Observed matching includes target plus declaration distribution;
  non-Param free values use the canonical same-name direct/scatter owner. This
  preserves differently named legacy parameter-site labels while making
  duplicate, colliding, wrapped, and distribution-only Factors uniformly
  unsupported for prior predictive.
- Two hand-built generative DataRef fixtures lacked matching `observed_nodes`;
  adding the declaration metadata made their intended roles explicit instead
  of retaining an unclassifiable convention. The updated bayeswire spec/corpus
  is vendored at `866fa70`; only its semantic prose and pin changed.
- The complete validation ladder is green: vendor/zero-dependency guards,
  format, Clippy with warnings denied, release and wasm builds, 356 Rust tests,
  release-tooling tests, and the nuts-rs oracle (4 targets, 15 checks).
