# Investigation: expected daily support-request count

## Question and estimand

For a deliberately synthetic 30-day public dataset, what expected number of
support requests should planning use for a future day under an exchangeable-day
model?

The estimand is `mean_daily_count`: the likelihood's expected count for one
future exchangeable day. It is not the next observed count, a maximum, or a
queue-service target.

The data are constructed, public teaching data. Three deliberately large days
create a visible dispersion problem; this is not an empirical discovery.

## Initial modeling decision

Decision ID `initial-likelihood`: use one Poisson likelihood with a shared
positive mean. This is a deliberately simple baseline: conditional variance is
forced to equal the mean. The proper prior is
`mean_daily_count ~ Exponential(rate=0.1)`. The unconstrained NUTS coordinate is
mapped through the positive exponential transform, whose Jacobian is included
in the evaluated density.

The observed counts have mean 2.93 and sample standard deviation about 4.55. The
retained posterior check is factual rather than a verdict: compare those values
with the posterior-predictive distributions of mean, standard deviation,
minimum, maximum, and zero count. The standard-deviation and maximum summaries
are expected to expose the Poisson limitation.

`negative-binomial.json` is a regression fixture, not part of the original
published snapshot. It keeps the same expected-count estimand and prior, and
adds a positive overdispersion parameter. The frozen recipient task does not
name this alternative.

## Reproduce the author evidence

From the repository root, regenerate every committed evidence file with one
script (or follow its commands). It builds one locked release binary and uses no
Python on the execution path:

```sh
scripts/generate_investigation_counts_evidence.sh
B=target/release/bayesite
E=examples/investigation-counts/evidence
$B inspect --model examples/investigation-counts/poisson.json \
  --data examples/investigation-counts/data.json --out "$E/inspection.json"
$B sample --model examples/investigation-counts/poisson.json \
  --data examples/investigation-counts/data.json --chains 4 --warmup 250 \
  --draws 250 --max-treedepth 8 --target-accept 0.85 --seed 20260916 \
  --out "$E/fit.jsonl"
$B diagnose --fit "$E/fit.jsonl" --out "$E/diagnostics.json"
$B posterior-check --model examples/investigation-counts/poisson.json \
  --data examples/investigation-counts/data.json --fit "$E/fit.jsonl" \
  --seed 20260917 --out "$E/check.json"
```

The standalone sample command uses its recorded default initial step size of
`1.0`; the investigation recipe spells that value out. The investigation's
`prior-predictive-initial` recipe records 200 draws with seed `20260915`; its
result is stored in the workspace object store and is not one of the committed
`evidence/` files.

`evidence/engine.json` records the exact executable digest, verbatim
`capabilities` document, target, and release profile used for the committed
evidence. The investigation snapshot records model and data independently; the
older fit's combined model/data fingerprint remains only its cooperative
compatibility check.

Create the durable object with the same binary:

```sh
$B investigation init --metadata examples/investigation-counts/metadata.json \
  --model examples/investigation-counts/poisson.json \
  --data examples/investigation-counts/data.json --out study/
$B investigation run study/ --recipe inspect-initial
$B investigation run study/ --recipe prior-predictive-initial
$B investigation run study/ --recipe sample-initial
$B investigation run study/ --recipe diagnose-initial
$B investigation run study/ --recipe check-initial
$B investigation snapshot study/ --out original/
$B investigation verify original/
```

The recipient workflow and editable workspace contract are documented in
[`docs/investigation-workspace-v0.md`](../../docs/investigation-workspace-v0.md).
Automated outcomes and the still-unperformed independent human/browser checks
are separated in [`HANDOFF-REPORT.md`](HANDOFF-REPORT.md).

## Interpretation boundary

The initial evidence can show that this Poisson data-generating story does not
reproduce the synthetic dataset's dispersion. It does not establish why counts
vary, whether the days are exchangeable, whether a negative-binomial likelihood
is scientifically adequate, or whether staffing should optimize this estimand.
Those are unresolved questions, not hidden defaults.
