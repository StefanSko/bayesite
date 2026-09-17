# Investigation: roaches pest-control treatment effect

A real workflow example from Gelman and Hill and *Regression and Other
Stories* (chapter 15), used in the Bayesian workflow literature because its
first model fails a posterior predictive check in an instructive way.

## Data

`data.json` is derived from the public `roaches` dataset in Vehtari's
ROS-Examples repository (262 apartments). Variables, in the bayeswire data
document form:

| name | shape | meaning |
|---|---|---|
| `count` | [262] | roaches caught after treatment (`y`) |
| `X` | [262, 3] | columns `roach1/100` (pre-treatment level, scaled), `treatment`, `senior` |
| `log_exposure` | [262] | log of trap-days (`exposure2`), used as an offset |
| `n`, `k` | [] | 262 and 3, the design dimensions |

Observed counts have mean 25.6, standard deviation 50.9, maximum 357, and 94
zeros out of 262.

## Question and estimand

How much lower is the expected number of roaches per trap-day in treated
apartments than in control apartments, adjusting for the pre-treatment level
and senior-building status? The estimand is the treatment coefficient on the
log-rate scale, `beta[1]` under the column order above. The investigation
format names the parameter vector `beta`; indexing a single coordinate is a
known limitation of the v0 estimand field.

## Initial model (decision `initial-likelihood`)

`poisson.json` is the textbook Poisson regression, hand-rolled in bayeswire IR:

```
count ~ Poisson(exp(log_exposure + alpha + X @ beta))
alpha ~ Normal(0, 2.5)
beta  ~ Normal(0, 2.5)   (vector of 3)
```

`bayesite inspect` reports two free slots, `alpha` at offset 0 and `beta` at
offset 1 with length 3, three density factors, and no structural
discrepancies.

## Retained limitation

With 4 chains of 500 warmup and 500 draws (seed 20260918), diagnostics are
clean: R-hat 1.002 and 1.004, no divergences. The posterior check with seed
20260919 then reports, for the count site:

| statistic | observed | replicated min | replicated max |
|---|---|---|---|
| mean | 25.6 | 24.0 | 27.2 |
| max | 357 | 334 | 485 |
| zero count | 94 | 0 | 2 |

The Poisson reproduces the mean and the maximum and cannot reproduce the
zeros: every one of 2000 replicated datasets has at most 2 zero counts
against 94 observed. This is the limitation the investigation retains. Which
revision to try is not prescribed; `negative-binomial.json` is a regression
fixture showing one likelihood change that keeps the estimand and priors and
adds a positive overdispersion parameter with an `Exponential(1)` prior.

## Run it

```sh
B=target/release/bayesite
E=examples/investigation-roaches
$B inspect --model $E/poisson.json --data $E/data.json
$B investigation init --metadata $E/metadata.json --model $E/poisson.json --data $E/data.json --out study/
$B investigation run study/ --recipe inspect-initial
$B investigation run study/ --recipe sample-initial
$B investigation run study/ --recipe diagnose-initial
$B investigation run study/ --recipe check-initial
$B investigation snapshot study/ --out original/
$B investigation verify original/
```

The whole sequence takes under a second on a laptop. The agent-assisted
continuation is described in
[`experiments/investigation-agent/README.md`](../../experiments/investigation-agent/README.md).
