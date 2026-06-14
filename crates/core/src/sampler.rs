//! Chain orchestration: warmup with Stan-style adaptation, then sampling.
//!
//! `sample` is a pure function of (model, data, settings, seed, chain_id):
//! no threads, no filesystem, no clock, no OS entropy. Callers own
//! parallelism (one thread or web worker per chain).

use crate::adapt::{DualAveraging, WarmupSchedule, Welford};
use crate::error::{Error, ErrorKind};
use crate::model::Posterior;
use crate::nuts::{transition, Hamiltonian, State};
use crate::rng::Xoshiro256PlusPlus;

#[derive(Debug, Clone)]
pub struct Settings {
    pub num_warmup: usize,
    pub num_draws: usize,
    pub max_treedepth: usize,
    pub target_accept: f64,
    pub initial_step_size: f64,
}

impl Default for Settings {
    fn default() -> Settings {
        Settings {
            num_warmup: 1000,
            num_draws: 1000,
            max_treedepth: 10,
            target_accept: 0.8,
            initial_step_size: 1.0,
        }
    }
}

impl Settings {
    fn validate(&self) -> Result<(), Error> {
        if self.num_draws == 0 {
            return Err(Error::new(
                ErrorKind::InvalidSettings,
                "num_draws must be at least 1",
            ));
        }
        if !(0.0..1.0).contains(&self.target_accept) || self.target_accept <= 0.0 {
            return Err(Error::new(
                ErrorKind::InvalidSettings,
                "target_accept must be in (0, 1)",
            ));
        }
        if self.max_treedepth == 0 || self.max_treedepth > 20 {
            return Err(Error::new(
                ErrorKind::InvalidSettings,
                "max_treedepth must be in 1..=20",
            ));
        }
        if !self.initial_step_size.is_finite() || self.initial_step_size <= 0.0 {
            return Err(Error::new(
                ErrorKind::InvalidSettings,
                "initial_step_size must be positive and finite",
            ));
        }
        Ok(())
    }
}

/// Post-warmup draws and per-chain diagnostics.
#[derive(Debug, Clone)]
pub struct ChainDraws {
    /// Unconstrained draws, one `q` per kept iteration.
    pub draws: Vec<Vec<f64>>,
    pub logp: Vec<f64>,
    pub divergences: usize,
    /// Tree-depth histogram over kept draws (index = depth).
    pub treedepth_histogram: Vec<usize>,
    /// Step size after adaptation.
    pub step_size: f64,
    /// Adapted inverse metric diagonal.
    pub inv_mass: Vec<f64>,
    /// Mean trajectory acceptance over kept draws.
    pub mean_accept: f64,
}

/// Stan-style coarse step-size search: double or halve until the one-step
/// acceptance crosses 0.5.
fn find_initial_step_size(
    ham: &Hamiltonian<'_>,
    rng: &mut Xoshiro256PlusPlus,
    q: &[f64],
    mut eps: f64,
) -> Result<f64, Error> {
    let mut state = ham.init_state(q.to_vec())?;
    ham.sample_momentum(&mut state, rng);
    let h0 = ham.energy(&state);
    let delta_h = |s: &State, e: f64| -> Result<f64, Error> {
        let stepped = ham.leapfrog(s, e)?;
        Ok(h0 - ham.energy(&stepped))
    };
    let target = 0.8_f64.ln();
    let mut dh = delta_h(&state, eps)?;
    let direction = if dh > target { 1 } else { -1 };
    for _ in 0..100 {
        ham.sample_momentum(&mut state, rng);
        let h0 = ham.energy(&state);
        let stepped = ham.leapfrog(&state, eps)?;
        dh = h0 - ham.energy(&stepped);
        if (direction == 1 && dh <= target) || (direction == -1 && dh >= target) {
            break;
        }
        eps = if direction == 1 { 2.0 * eps } else { 0.5 * eps };
        if !(1e-16..=1e7).contains(&eps) {
            return Err(Error::new(
                ErrorKind::NonFiniteDensity,
                "could not find a workable step size; the posterior may be degenerate",
            ));
        }
    }
    Ok(eps)
}

/// Run one chain: warmup with window adaptation, then `num_draws` draws.
pub fn sample(
    posterior: &Posterior,
    settings: &Settings,
    seed: u64,
    chain_id: u64,
) -> Result<ChainDraws, Error> {
    settings.validate()?;
    let dim = posterior.n_params();
    let mut rng = Xoshiro256PlusPlus::for_chain(seed, chain_id);

    // Stan-style init: uniform(-2, 2) on the unconstrained scale, retrying
    // until the density is finite.
    let mut q = vec![0.0; dim];
    let mut init_ok = false;
    for _ in 0..100 {
        for qi in q.iter_mut() {
            *qi = 4.0 * rng.uniform() - 2.0;
        }
        if posterior.logp(&q)?.is_finite() {
            init_ok = true;
            break;
        }
    }
    if !init_ok {
        return Err(Error::new(
            ErrorKind::NonFiniteDensity,
            "could not find an initial point with finite log density after 100 tries; \
             check the model's constraints and data",
        ));
    }

    let mut inv_mass = vec![1.0; dim];
    let mut state_q = q;

    // Warmup with windowed adaptation.
    let schedule = WarmupSchedule::new(settings.num_warmup);
    let mut step_size = settings.initial_step_size;
    let frozen_step_size;
    {
        let mut ham = Hamiltonian::new(posterior, inv_mass.clone());
        if settings.num_warmup > 0 {
            step_size = find_initial_step_size(&ham, &mut rng, &state_q, step_size)?;
        }
        let mut da = DualAveraging::new(step_size, settings.target_accept);
        let mut welford = Welford::new(dim);
        let mut state = ham.init_state(state_q.clone())?;

        for iter in 0..settings.num_warmup {
            let result = transition(
                &ham,
                &mut rng,
                state.clone(),
                da.step_size(),
                settings.max_treedepth,
            )?;
            state.q = result.q;
            state.logp = result.logp;
            state.grad = result.grad;
            da.update(result.accept_prob);

            let window = schedule.step(iter);
            if window.accumulate {
                welford.push(&state.q);
            }
            if window.close_window && welford.count() > 1 {
                inv_mass = welford.regularized_variance();
                ham = Hamiltonian::new(posterior, inv_mass.clone());
                welford = Welford::new(dim);
                // Re-find a workable step size under the new metric and
                // restart dual averaging around it.
                let eps = find_initial_step_size(&ham, &mut rng, &state.q, da.step_size())?;
                da = DualAveraging::new(eps, settings.target_accept);
            }
        }
        frozen_step_size = if settings.num_warmup > 0 {
            da.averaged_step_size()
        } else {
            settings.initial_step_size
        };
        state_q = state.q;
    }

    // Sampling with frozen adaptation.
    let ham = Hamiltonian::new(posterior, inv_mass.clone());
    let mut state = ham.init_state(state_q)?;
    let mut draws = Vec::with_capacity(settings.num_draws);
    let mut logp = Vec::with_capacity(settings.num_draws);
    let mut divergences = 0usize;
    let mut treedepth_histogram = vec![0usize; settings.max_treedepth + 1];
    let mut accept_sum = 0.0;
    for _ in 0..settings.num_draws {
        let result = transition(
            &ham,
            &mut rng,
            state.clone(),
            frozen_step_size,
            settings.max_treedepth,
        )?;
        state.q = result.q.clone();
        state.logp = result.logp;
        state.grad = result.grad;
        if result.divergent {
            divergences += 1;
        }
        treedepth_histogram[result.depth] += 1;
        accept_sum += result.accept_prob;
        draws.push(result.q);
        logp.push(result.logp);
    }

    Ok(ChainDraws {
        draws,
        logp,
        divergences,
        treedepth_histogram,
        step_size: frozen_step_size,
        inv_mass,
        mean_accept: accept_sum / settings.num_draws as f64,
    })
}
