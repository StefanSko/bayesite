# Sampler provenance and behavior

This documents exactly which algorithm the Rust core implements, which
sources it was implemented from, and where it knowingly deviates from
Stan. The implementation lives in `crates/core/src/nuts.rs`, `adapt.rs`, and
`sampler.rs`.

## Algorithm

**Multinomial NUTS** with the **generalized U-turn criterion including
across-subtree checks**, i.e. the algorithm Stan has shipped since 2.21,
not the original 2014 slice-sampling variant and not the naive
endpoints-only U-turn check (a known correctness trap).

Implemented from:

- Hoffman & Gelman, *The No-U-Turn Sampler* (JMLR 2014) — trajectory
  doubling, the termination idea.
- Betancourt, *A Conceptual Introduction to Hamiltonian Monte Carlo*
  (arXiv:1701.02434) — multinomial sampling over trajectories, biased
  progressive sampling.
- The Stan Reference Manual (MCMC chapters) and the structure of Stan's
  `base_nuts.hpp` — the generalized criterion and the across-subtree
  checks; the `nuts-rs` project (MIT) served as a behavioral cross-check
  only. No code was copied from either.

### Per transition

1. Momentum is resampled from `N(0, M)` with diagonal `M`; kinetic energy
   is `0.5 * p^T M^{-1} p`.
2. The trajectory doubles in a uniformly random direction up to
   `max_treedepth` (default 10).
3. Within a subtree merge, the proposal is selected multinomially by the
   log-sum-exp of point weights `exp(H0 - H)`; at the top level the new
   subtree's proposal is accepted with the *biased progressive* rule
   (always when the new subtree outweighs the old trajectory).
4. The U-turn criterion `p_sharp_minus . rho > 0 && p_sharp_plus . rho > 0`
   (with `p_sharp = M^{-1} p`, `rho` the momentum sum) is checked on the
   merged span **and across both subtree joins**: `rho_left + p_right_begin`
   and `rho_right + p_left_end` inside `build_tree`, and the analogous two
   checks when the top-level trajectory absorbs a new subtree.
5. A trajectory is **divergent** when the energy error exceeds 1000
   (Stan's `max_deltaH`); the subtree is discarded and doubling stops.
6. The dual-averaging statistic is the mean Metropolis acceptance
   `min(1, exp(H0 - H))` over all leapfrog steps of the transition.

## Warmup adaptation

Stan's three-phase windowed adaptation:

- **Step size:** Nesterov dual averaging on `log(eps)` with `delta = 0.8`
  (configurable), `gamma = 0.05`, `t0 = 10`, `kappa = 0.75`,
  `mu = log(10 * eps0)`. A coarse doubling/halving search picks the
  initial step size (one-step acceptance crossing 0.5, threshold
  `log(0.8)` as in Stan's `init_stepsize`).
- **Metric:** diagonal inverse mass matrix from Welford accumulators with
  Stan's regularization `(n/(n+5)) * var + 1e-3 * (5/(n+5))`.
- **Schedule:** init buffer 75 (step size only), doubling windows from a
  base of 25, term buffer 50; the last window extends to the boundary when
  the next doubling would not fit (for 1000 warmup iterations the windows
  close at 100, 150, 250, 450, 950). Short warmups scale the buffers
  to 15% / 75% / 10%. After each window the metric is updated, a fresh
  step-size search runs, and dual averaging restarts around it. The
  post-warmup step size is the dual-averaged value.

## RNG and reproducibility

`xoshiro256++` seeded via splitmix64 (reference vectors committed in
`src/rng.rs` tests). The per-chain stream derivation is part of the output
contract:

```
base  = SplitMix64(seed).next()
state = four SplitMix64(base XOR chain_id) outputs
```

Standard normals use the Marsaglia polar method (no ziggurat, no trig).
Identical (model, data, settings, seed, chain_id) always reproduce
identical draws, on every platform including wasm.

## Known deviations from Stan

- **RNG**: Stan uses a different generator; draws are not comparable
  draw-for-draw with any other system. Equivalence with the JAX backend is
  established at the level of log densities/gradients (fixture rtol 1e-12 /
  1e-10) and posterior statistics (`scripts/check_rust_backend_posterior.py`).
- **Initialization**: uniform(-2, 2) on the unconstrained scale, retried up
  to 100 times for a finite density — same policy as Stan, but without
  user-suppliable inits yet.
- **Energy diagnostic**: retained draws record Hamiltonian energy as
  `sample_stats.energy` in the v0-provisional fit stream. The value is the
  selected phase-space point's `-logp + 0.5 * p^T M^{-1} p` for that
  transition; Bayesite emits the factual energy but no BFMI verdict.
- **Dense metric**: not implemented; the metric is always diagonal.
