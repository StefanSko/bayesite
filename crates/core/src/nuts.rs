//! Multinomial NUTS transition with the generalized U-turn criterion,
//! including the across-subtree checks (Stan >= 2.21 behavior).
//!
//! Implemented from the algorithm descriptions in Hoffman & Gelman (2014),
//! Betancourt (2017, arXiv:1701.02434), and the Stan Reference Manual /
//! Stan's `base_nuts.hpp` structure; see NOTICE. The naive
//! endpoints-only U-turn check is a known correctness trap — the merge
//! checks here test the criterion across subtree joins as well.
//!
//! Diagonal mass matrix: `inv_mass` is the inverse metric (the regularized
//! sample variance), kinetic energy is `0.5 * p^T diag(inv_mass) p`.

use crate::error::Error;
use crate::model::Posterior;
use crate::rng::Xoshiro256PlusPlus;

/// Energy error beyond which a trajectory is declared divergent (Stan's
/// `max_deltaH`).
pub const DIVERGENCE_THRESHOLD: f64 = 1000.0;

/// A phase-space point with cached log density and gradient.
#[derive(Debug, Clone)]
pub struct State {
    pub q: Vec<f64>,
    pub p: Vec<f64>,
    pub logp: f64,
    pub grad: Vec<f64>,
}

pub struct Hamiltonian<'a> {
    posterior: &'a Posterior,
    pub inv_mass: Vec<f64>,
}

impl<'a> Hamiltonian<'a> {
    pub fn new(posterior: &'a Posterior, inv_mass: Vec<f64>) -> Hamiltonian<'a> {
        assert_eq!(inv_mass.len(), posterior.n_params());
        Hamiltonian {
            posterior,
            inv_mass,
        }
    }

    pub fn init_state(&self, q: Vec<f64>) -> Result<State, Error> {
        let (logp, grad) = self.posterior.logp_grad(&q)?;
        Ok(State {
            p: vec![0.0; q.len()],
            q,
            logp,
            grad,
        })
    }

    /// Sample momentum p ~ N(0, M) with M = diag(1 / inv_mass).
    pub fn sample_momentum(&self, state: &mut State, rng: &mut Xoshiro256PlusPlus) {
        for (p, &im) in state.p.iter_mut().zip(&self.inv_mass) {
            *p = rng.standard_normal() / im.sqrt();
        }
    }

    /// Total energy H = -logp + 0.5 p^T M^{-1} p (+inf for non-finite logp).
    pub fn energy(&self, state: &State) -> f64 {
        let kinetic: f64 = state
            .p
            .iter()
            .zip(&self.inv_mass)
            .map(|(&p, &im)| 0.5 * p * p * im)
            .sum();
        let h = -state.logp + kinetic;
        if h.is_nan() {
            f64::INFINITY
        } else {
            h
        }
    }

    /// One leapfrog step with (signed) step size `eps`.
    pub fn leapfrog(&self, state: &State, eps: f64) -> Result<State, Error> {
        let p_half: Vec<f64> = state
            .p
            .iter()
            .zip(&state.grad)
            .map(|(&p, &g)| p + 0.5 * eps * g)
            .collect();
        let q: Vec<f64> = state
            .q
            .iter()
            .zip(p_half.iter().zip(&self.inv_mass))
            .map(|(&q, (&ph, &im))| q + eps * im * ph)
            .collect();
        let (logp, grad) = self.posterior.logp_grad(&q)?;
        let p: Vec<f64> = p_half
            .iter()
            .zip(&grad)
            .map(|(&ph, &g)| ph + 0.5 * eps * g)
            .collect();
        Ok(State { q, p, logp, grad })
    }

    /// dtau/dp: the "sharp" momentum used by the U-turn criterion.
    fn p_sharp(&self, p: &[f64]) -> Vec<f64> {
        p.iter()
            .zip(&self.inv_mass)
            .map(|(&p, &im)| p * im)
            .collect()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// Generalized U-turn criterion: continue while the trajectory increment
/// `rho` points forward at both sharp endpoints.
fn criterion(p_sharp_a: &[f64], p_sharp_b: &[f64], rho: &[f64]) -> bool {
    dot(p_sharp_a, rho) > 0.0 && dot(p_sharp_b, rho) > 0.0
}

fn log_sum_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// A built subtree: edge states in stepping order plus merge accumulators.
struct Subtree {
    /// Proposal drawn multinomially from the subtree.
    proposal_q: Vec<f64>,
    proposal_logp: f64,
    proposal_grad: Vec<f64>,
    /// Momentum / sharp momentum at the edge nearest the existing tree.
    p_beg: Vec<f64>,
    p_sharp_beg: Vec<f64>,
    /// Momentum / sharp momentum at the far edge.
    p_end: Vec<f64>,
    p_sharp_end: Vec<f64>,
    /// Sum of momenta over the subtree.
    rho: Vec<f64>,
    /// log sum of multinomial weights exp(h0 - h).
    log_sum_weight: f64,
}

/// Per-transition statistics for adaptation and diagnostics.
#[derive(Debug, Clone)]
pub struct Transition {
    pub q: Vec<f64>,
    pub logp: f64,
    pub grad: Vec<f64>,
    pub depth: usize,
    pub n_leapfrog: usize,
    pub divergent: bool,
    /// Average Metropolis acceptance over the trajectory (dual-averaging
    /// statistic).
    pub accept_prob: f64,
}

struct TreeStats {
    n_leapfrog: usize,
    sum_metro_prob: f64,
    divergent: bool,
}

fn build_tree(
    ham: &Hamiltonian<'_>,
    rng: &mut Xoshiro256PlusPlus,
    depth: usize,
    z: &mut State,
    h0: f64,
    eps: f64,
    stats: &mut TreeStats,
) -> Result<Option<Subtree>, Error> {
    if depth == 0 {
        *z = ham.leapfrog(z, eps)?;
        stats.n_leapfrog += 1;
        let h = ham.energy(z);
        let delta = h0 - h; // log weight of this point
        if -delta > DIVERGENCE_THRESHOLD {
            stats.divergent = true;
            return Ok(None);
        }
        stats.sum_metro_prob += delta.exp().min(1.0);
        let p_sharp = ham.p_sharp(&z.p);
        return Ok(Some(Subtree {
            proposal_q: z.q.clone(),
            proposal_logp: z.logp,
            proposal_grad: z.grad.clone(),
            p_beg: z.p.clone(),
            p_sharp_beg: p_sharp.clone(),
            p_end: z.p.clone(),
            p_sharp_end: p_sharp,
            rho: z.p.clone(),
            log_sum_weight: delta,
        }));
    }

    let Some(mut left) = build_tree(ham, rng, depth - 1, z, h0, eps, stats)? else {
        return Ok(None);
    };
    let Some(right) = build_tree(ham, rng, depth - 1, z, h0, eps, stats)? else {
        return Ok(None);
    };

    // Multinomial selection between the two halves.
    let log_sum_weight = log_sum_exp(left.log_sum_weight, right.log_sum_weight);
    let take_right = rng.uniform().ln() < right.log_sum_weight - log_sum_weight;
    let (proposal_q, proposal_logp, proposal_grad) = if take_right {
        (right.proposal_q, right.proposal_logp, right.proposal_grad)
    } else {
        (left.proposal_q, left.proposal_logp, left.proposal_grad)
    };

    let rho = add(&left.rho, &right.rho);
    // U-turn across the merged subtree, plus the across-join checks.
    let mut ok = criterion(&left.p_sharp_beg, &right.p_sharp_end, &rho);
    let rho_left_join = add(&left.rho, &right.p_beg);
    ok = ok && criterion(&left.p_sharp_beg, &right.p_sharp_beg, &rho_left_join);
    let rho_right_join = add(&right.rho, &left.p_end);
    ok = ok && criterion(&left.p_sharp_end, &right.p_sharp_end, &rho_right_join);
    if !ok {
        return Ok(None);
    }

    left.proposal_q = proposal_q;
    left.proposal_logp = proposal_logp;
    left.proposal_grad = proposal_grad;
    left.p_end = right.p_end;
    left.p_sharp_end = right.p_sharp_end;
    left.rho = rho;
    left.log_sum_weight = log_sum_weight;
    Ok(Some(left))
}

/// One NUTS transition from `state` (whose momentum is resampled here).
pub fn transition(
    ham: &Hamiltonian<'_>,
    rng: &mut Xoshiro256PlusPlus,
    mut state: State,
    step_size: f64,
    max_depth: usize,
) -> Result<Transition, Error> {
    ham.sample_momentum(&mut state, rng);
    let h0 = ham.energy(&state);

    let p_sharp0 = ham.p_sharp(&state.p);
    let mut z_minus = state.clone();
    let mut z_plus = state.clone();
    let mut p_sharp_minus = p_sharp0.clone();
    let mut p_sharp_plus = p_sharp0;
    let mut rho = state.p.clone();

    let mut sample_q = state.q.clone();
    let mut sample_logp = state.logp;
    let mut sample_grad = state.grad.clone();
    let mut log_sum_weight = 0.0; // weight of the initial point
    let mut stats = TreeStats {
        n_leapfrog: 0,
        sum_metro_prob: 0.0,
        divergent: false,
    };
    let mut depth = 0usize;

    while depth < max_depth {
        let forward = rng.uniform() < 0.5;
        let eps = if forward { step_size } else { -step_size };
        let mut z = if forward {
            z_plus.clone()
        } else {
            z_minus.clone()
        };
        let subtree = build_tree(ham, rng, depth, &mut z, h0, eps, &mut stats)?;
        let Some(subtree) = subtree else { break };

        depth += 1;

        // Biased progressive sampling toward the new subtree.
        if subtree.log_sum_weight > log_sum_weight
            || rng.uniform().ln() < subtree.log_sum_weight - log_sum_weight
        {
            sample_q = subtree.proposal_q.clone();
            sample_logp = subtree.proposal_logp;
            sample_grad = subtree.proposal_grad.clone();
        }
        log_sum_weight = log_sum_exp(log_sum_weight, subtree.log_sum_weight);

        let rho_total = add(&rho, &subtree.rho);
        // Old-tree edge on the side we extended from.
        let (old_edge_p, old_edge_sharp) = if forward {
            (z_plus.p.clone(), p_sharp_plus.clone())
        } else {
            (z_minus.p.clone(), p_sharp_minus.clone())
        };
        // Far-side edge of the whole trajectory before extension.
        let far_side_sharp = if forward {
            &p_sharp_minus
        } else {
            &p_sharp_plus
        };

        // Whole-trajectory U-turn check plus the two across-join checks.
        let mut persist = criterion(far_side_sharp, &subtree.p_sharp_end, &rho_total);
        let rho_join_old = add(&rho, &subtree.p_beg);
        persist = persist && criterion(far_side_sharp, &subtree.p_sharp_beg, &rho_join_old);
        let rho_join_new = add(&subtree.rho, &old_edge_p);
        persist = persist && criterion(&old_edge_sharp, &subtree.p_sharp_end, &rho_join_new);

        // Commit the extension.
        if forward {
            z_plus = z;
            p_sharp_plus = subtree.p_sharp_end.clone();
        } else {
            z_minus = z;
            p_sharp_minus = subtree.p_sharp_end.clone();
        }
        rho = rho_total;

        if !persist {
            break;
        }
    }

    let accept_prob = if stats.n_leapfrog > 0 {
        stats.sum_metro_prob / stats.n_leapfrog as f64
    } else {
        0.0
    };
    Ok(Transition {
        q: sample_q,
        logp: sample_logp,
        grad: sample_grad,
        depth,
        n_leapfrog: stats.n_leapfrog,
        divergent: stats.divergent,
        accept_prob,
    })
}
