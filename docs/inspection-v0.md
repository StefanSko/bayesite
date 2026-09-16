# Effective-model inspection (`v0-provisional`)

`bayesite inspect --model MODEL --data DATA --out REPORT` binds a Bayeswire IR
model to data through the same `Posterior` construction used by sampling. It
does not sample, execute producer code, or evaluate a scientific verdict.
The native/Wasm protocol equivalent is
`{"command":"inspect","model":...,"data":...}`.

The report marker is `"inspection_format":"v0-provisional"`. It contains:

- whether free-value and stochastic-site execution metadata were explicit or
  legacy-derived;
- ordered bound free slots with shape, unconstrained offset/length, and the
  resolved transform/bounds;
- ordered actual density factors with distributions and value expressions in
  Bayeswire node encoding;
- parameter, observed-value, and explicit-free-value declarations alongside
  execution metadata;
- required data roles and bound shapes, without copying values;
- conservative structural discrepancies between same-name declarations and
  factors; and
- an explicit statement that transform Jacobians contribute to evaluated log
  density.

Array order remains semantic. A reported structural difference does not claim
that two expressions are mathematically inequivalent. Binding errors produce no
partial report.
