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
