//! Stan-style warmup adaptation: dual-averaging step size and windowed
//! diagonal mass-matrix estimation (Welford accumulators).
//!
//! Constants and schedule follow the Stan Reference Manual's HMC
//! adaptation description: delta target 0.8 by default; gamma=0.05,
//! t0=10, kappa=0.75, mu=log(10*eps); init buffer 75, term buffer 50,
//! base window 25 with doubling, and the short-warmup proportional
//! adjustment (15% / 75% / 10%).

/// Nesterov dual averaging on log(step size).
#[derive(Debug, Clone)]
pub struct DualAveraging {
    mu: f64,
    log_eps: f64,
    log_eps_bar: f64,
    h_bar: f64,
    count: u64,
    pub target_accept: f64,
}

const GAMMA: f64 = 0.05;
const T0: f64 = 10.0;
const KAPPA: f64 = 0.75;

impl DualAveraging {
    pub fn new(initial_step_size: f64, target_accept: f64) -> DualAveraging {
        DualAveraging {
            mu: (10.0 * initial_step_size).ln(),
            log_eps: initial_step_size.ln(),
            log_eps_bar: 0.0,
            h_bar: 0.0,
            count: 0,
            target_accept,
        }
    }

    pub fn step_size(&self) -> f64 {
        self.log_eps.exp()
    }

    /// The averaged step size to freeze after a window completes.
    pub fn averaged_step_size(&self) -> f64 {
        self.log_eps_bar.exp()
    }

    pub fn update(&mut self, accept_prob: f64) {
        self.count += 1;
        let m = self.count as f64;
        let eta = 1.0 / (m + T0);
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target_accept - accept_prob);
        self.log_eps = self.mu - (m.sqrt() / GAMMA) * self.h_bar;
        let weight = m.powf(-KAPPA);
        self.log_eps_bar = weight * self.log_eps + (1.0 - weight) * self.log_eps_bar;
    }
}

/// Welford running mean/variance per coordinate.
#[derive(Debug, Clone)]
pub struct Welford {
    count: u64,
    mean: Vec<f64>,
    m2: Vec<f64>,
}

impl Welford {
    pub fn new(dim: usize) -> Welford {
        Welford {
            count: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
        }
    }

    pub fn push(&mut self, x: &[f64]) {
        self.count += 1;
        let n = self.count as f64;
        for ((mean, m2), &xi) in self.mean.iter_mut().zip(self.m2.iter_mut()).zip(x) {
            let delta = xi - *mean;
            *mean += delta / n;
            *m2 += delta * (xi - *mean);
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    /// Regularized sample variance, Stan's shrinkage toward 1e-3:
    /// `(n / (n + 5)) * var + 1e-3 * (5 / (n + 5))`.
    pub fn regularized_variance(&self) -> Vec<f64> {
        let n = self.count as f64;
        self.m2
            .iter()
            .map(|&m2| {
                let var = if self.count > 1 { m2 / (n - 1.0) } else { 1.0 };
                (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            })
            .collect()
    }
}

/// What the warmup loop should do at one warmup iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowStep {
    /// Accumulate this draw into the mass-matrix estimator.
    pub accumulate: bool,
    /// This iteration closes a mass window: update the metric and restart
    /// step-size adaptation.
    pub close_window: bool,
}

/// Stan's three-phase warmup schedule.
#[derive(Debug, Clone)]
pub struct WarmupSchedule {
    pub init_buffer: usize,
    pub term_buffer: usize,
    window_ends: Vec<usize>,
    num_warmup: usize,
}

impl WarmupSchedule {
    pub fn new(num_warmup: usize) -> WarmupSchedule {
        let (init_buffer, term_buffer, base_window) = if num_warmup < 75 + 50 + 25 {
            // Short warmup: 15% init, 10% term, remainder windowed.
            let init = (0.15 * num_warmup as f64) as usize;
            let term = (0.10 * num_warmup as f64) as usize;
            (init, term, num_warmup.saturating_sub(init + term))
        } else {
            (75, 50, 25)
        };
        // Doubling windows from init_buffer to num_warmup - term_buffer;
        // the last window absorbs the remainder.
        let mut window_ends = Vec::new();
        let adapt_end = num_warmup.saturating_sub(term_buffer);
        let mut window_start = init_buffer;
        let mut window_size = base_window.max(1);
        while window_start < adapt_end {
            let mut end = window_start + window_size;
            // If the next doubling would not fit, extend to the boundary.
            if end + 2 * window_size > adapt_end {
                end = adapt_end;
            }
            window_ends.push(end.min(adapt_end));
            window_start = end;
            window_size *= 2;
        }
        WarmupSchedule {
            init_buffer,
            term_buffer,
            window_ends,
            num_warmup,
        }
    }

    /// Behavior of warmup iteration `iter` (0-based).
    pub fn step(&self, iter: usize) -> WindowStep {
        let adapt_end = self.num_warmup.saturating_sub(self.term_buffer);
        let in_windows = iter >= self.init_buffer && iter < adapt_end;
        let close = self.window_ends.iter().any(|&end| iter + 1 == end);
        WindowStep {
            accumulate: in_windows,
            close_window: close,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_schedule_matches_stan_windows() {
        // 1000 warmup: windows close at 100, 150, 250, 450, 950 (Stan's
        // documented schedule: 75 init, then 25/50/100/200/500*, 50 term;
        // the 400-draw window extends to the 950 boundary because the next
        // doubling would not fit).
        let schedule = WarmupSchedule::new(1000);
        assert_eq!(schedule.init_buffer, 75);
        assert_eq!(schedule.term_buffer, 50);
        let closes: Vec<usize> = (0..1000)
            .filter(|&i| schedule.step(i).close_window)
            .map(|i| i + 1)
            .collect();
        assert_eq!(closes, vec![100, 150, 250, 450, 950]);
        assert!(!schedule.step(50).accumulate); // init buffer
        assert!(schedule.step(100).accumulate);
        assert!(!schedule.step(960).accumulate); // term buffer
    }

    #[test]
    fn short_warmup_scales_buffers() {
        let schedule = WarmupSchedule::new(100);
        assert_eq!(schedule.init_buffer, 15);
        assert_eq!(schedule.term_buffer, 10);
        let closes: Vec<usize> = (0..100)
            .filter(|&i| schedule.step(i).close_window)
            .map(|i| i + 1)
            .collect();
        assert_eq!(closes, vec![90]);
    }

    #[test]
    fn welford_matches_two_pass_variance() {
        let xs = [
            vec![1.0, -2.0],
            vec![2.5, 0.5],
            vec![-0.5, 3.0],
            vec![4.0, 1.0],
        ];
        let mut w = Welford::new(2);
        for x in &xs {
            w.push(x);
        }
        let n = xs.len() as f64;
        for dim in 0..2 {
            let mean: f64 = xs.iter().map(|x| x[dim]).sum::<f64>() / n;
            let var: f64 = xs.iter().map(|x| (x[dim] - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let want = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0));
            let got = w.regularized_variance()[dim];
            assert!((got - want).abs() < 1e-12, "dim {dim}: {got} vs {want}");
        }
    }

    #[test]
    fn dual_averaging_raises_step_size_when_accepting() {
        let mut da = DualAveraging::new(0.1, 0.8);
        for _ in 0..50 {
            da.update(1.0); // always accepting: step size should grow
        }
        assert!(da.step_size() > 0.1);
        let mut da_low = DualAveraging::new(0.1, 0.8);
        for _ in 0..50 {
            da_low.update(0.0); // always rejecting: step size should shrink
        }
        assert!(da_low.step_size() < 0.1);
    }
}
