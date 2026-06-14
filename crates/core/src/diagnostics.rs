//! Split R-hat and effective sample size, matching
//! `blackjax.diagnostics` (which `jaxstanv5.diagnostics` wraps) value for
//! value.
//!
//! The autocovariance is computed directly (O(N^2)) instead of via FFT;
//! the estimator is identical, only the rounding differs (the fixture
//! test pins agreement to ~1e-8 relative). Geyer's initial positive and
//! monotone sequence rules follow the blackjax implementation including
//! its JAX indexing edge cases (out-of-bounds scatter drops, gather
//! clamps).

/// Per-chain series: `chains[c][i]` is draw `i` of chain `c`.
pub type Chains<'a> = &'a [Vec<f64>];

fn chain_mean(chain: &[f64]) -> f64 {
    chain.iter().sum::<f64>() / chain.len() as f64
}

fn variance_ddof1(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = chain_mean(values);
    values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0)
}

/// Gelman-Rubin potential scale reduction over two or more chains.
pub fn potential_scale_reduction(chains: Chains<'_>) -> f64 {
    assert!(
        chains.len() > 1,
        "potential_scale_reduction needs two or more chains"
    );
    let num_samples = chains[0].len() as f64;
    let means: Vec<f64> = chains.iter().map(|c| chain_mean(c)).collect();
    let vars: Vec<f64> = chains.iter().map(|c| variance_ddof1(c)).collect();
    let between = num_samples * variance_ddof1(&means);
    let within = chain_mean(&vars);
    ((between / within + num_samples - 1.0) / num_samples).sqrt()
}

/// Split R-hat as `jaxstanv5.diagnostics.rhat` computes it: a single chain
/// is split into halves first.
pub fn split_rhat(chains: Chains<'_>) -> f64 {
    if chains.len() > 1 {
        return potential_scale_reduction(chains);
    }
    let chain = &chains[0];
    let half = chain.len() / 2;
    let split = vec![chain[..half].to_vec(), chain[half..2 * half].to_vec()];
    potential_scale_reduction(&split)
}

/// Effective sample size (Geyer initial monotone sequence, Stan-style),
/// mirroring `blackjax.diagnostics.effective_sample_size`.
pub fn effective_sample_size(chains: Chains<'_>) -> f64 {
    let num_chains = chains.len();
    let num_samples = chains[0].len();
    assert!(num_samples > 1, "ess needs at least 2 samples");

    // Biased per-chain autocovariance (divides by N, like the FFT path).
    let mut mean_autocov = vec![0.0f64; num_samples];
    let mut chain_means = Vec::with_capacity(num_chains);
    for chain in chains {
        let mean = chain_mean(chain);
        chain_means.push(mean);
        let centered: Vec<f64> = chain.iter().map(|&x| x - mean).collect();
        for (t, slot) in mean_autocov.iter_mut().enumerate() {
            let mut acc = 0.0;
            for i in 0..num_samples - t {
                acc += centered[i] * centered[i + t];
            }
            *slot += acc / num_samples as f64;
        }
    }
    for slot in mean_autocov.iter_mut() {
        *slot /= num_chains as f64;
    }

    let n = num_samples as f64;
    let mean_var0 = mean_autocov[0] * n / (n - 1.0);
    let mut weighted_var = mean_var0 * (n - 1.0) / n;
    if num_chains > 1 {
        weighted_var += variance_ddof1(&chain_means);
    }

    // rho_hat[0] = 1; rho_hat[t] = 1 - (mean_var0 - autocov[t]) / weighted_var.
    let num_samples_even = num_samples - num_samples % 2;
    let mut rho_hat = Vec::with_capacity(num_samples_even);
    rho_hat.push(1.0);
    for &autocov in &mean_autocov[1..num_samples_even] {
        rho_hat.push(1.0 - (mean_var0 - autocov) / weighted_var);
    }
    let pairs = num_samples_even / 2;
    let even: Vec<f64> = (0..pairs).map(|k| rho_hat[2 * k]).collect();
    let odd: Vec<f64> = (0..pairs).map(|k| rho_hat[2 * k + 1]).collect();

    // Geyer initial positive sequence: prefix of pairs with positive sums.
    let mut mask = vec![false; pairs];
    let mut carry = true;
    let mut max_t = 0usize;
    for k in 0..pairs {
        carry = carry && (even[k] + odd[k] > 0.0);
        mask[k] = carry;
        if carry {
            max_t = k;
        }
    }
    let indices = max_t + 1;

    let odd_masked: Vec<f64> = (0..pairs)
        .map(|k| if mask[k] { odd[k] } else { 0.0 })
        .collect();
    // mask_even = mask with [indices] set to (even[indices] > 0);
    // out-of-bounds scatter is dropped, as in JAX.
    let mut mask_even = mask.clone();
    if indices < pairs {
        mask_even[indices] = even[indices] > 0.0;
    }
    let even_masked: Vec<f64> = (0..pairs)
        .map(|k| if mask_even[k] { even[k] } else { 0.0 })
        .collect();

    // Geyer initial monotone sequence: clamp pair sums to a running minimum.
    let sums: Vec<f64> = (0..pairs).map(|k| even_masked[k] + odd_masked[k]).collect();
    let mut even_final = even_masked.clone();
    let mut odd_final = odd_masked.clone();
    let mut running_min = sums[0];
    for k in 0..pairs {
        if sums[k] > running_min {
            even_final[k] = running_min / 2.0;
            odd_final[k] = running_min / 2.0;
        } else {
            running_min = sums[k];
        }
    }

    let ess_raw = (num_chains * num_samples) as f64;
    // Gather clamps out-of-bounds, as in JAX.
    let even_at_indices = even_final[indices.min(pairs - 1)];
    let total: f64 = even_final.iter().zip(&odd_final).map(|(e, o)| e + o).sum();
    let mut tau_hat = -1.0 + 2.0 * total - even_at_indices;
    tau_hat = tau_hat.max(1.0 / ess_raw.log10());
    ess_raw / tau_hat
}
