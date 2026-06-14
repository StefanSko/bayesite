//! Deterministic PRNG: splitmix64 for seeding/stream derivation,
//! xoshiro256++ as the generator, Marsaglia polar for standard normals.
//!
//! Reimplemented from the public-domain (CC0) reference implementations by
//! David Blackman and Sebastiano Vigna (https://prng.di.unimi.it/); see
//! NOTICE. Test vectors come from those reference programs (as
//! committed in the rand_xoshiro crate's reference tests).
//!
//! No OS entropy anywhere: seeds are explicit arguments (a settled decision
//! — the library must be a pure function, wasm-safe).

const PHI: u64 = 0x9e37_79b9_7f4a_7c15;

/// SplitMix64: a fixed-increment Weyl sequence through a 64-bit mixer.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> SplitMix64 {
        SplitMix64 { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(PHI);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// xoshiro256++ with an explicit 256-bit state.
#[derive(Debug, Clone)]
pub struct Xoshiro256PlusPlus {
    s: [u64; 4],
    /// Cached second normal from the polar method.
    spare_normal: Option<f64>,
}

impl Xoshiro256PlusPlus {
    pub fn from_state(s: [u64; 4]) -> Xoshiro256PlusPlus {
        assert!(s.iter().any(|&w| w != 0), "xoshiro state must be nonzero");
        Xoshiro256PlusPlus {
            s,
            spare_normal: None,
        }
    }

    /// Seed from a u64 by filling the state with splitmix64 outputs
    /// (Vigna's recommended initialization).
    pub fn seed_from_u64(seed: u64) -> Xoshiro256PlusPlus {
        let mut sm = SplitMix64::new(seed);
        Xoshiro256PlusPlus::from_state([sm.next_u64(), sm.next_u64(), sm.next_u64(), sm.next_u64()])
    }

    /// The per-chain stream. Derivation (part of the output contract):
    /// `base = splitmix64(seed).next()`, then the xoshiro state is seeded
    /// from `splitmix64(base XOR chain_id)`. Distinct chain ids give
    /// distinct, decorrelated streams for any fixed seed.
    pub fn for_chain(seed: u64, chain_id: u64) -> Xoshiro256PlusPlus {
        let mut sm = SplitMix64::new(seed);
        let base = sm.next_u64();
        Xoshiro256PlusPlus::seed_from_u64(base ^ chain_id)
    }

    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via the Marsaglia polar method (no trig, no tables).
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(spare) = self.spare_normal.take() {
            return spare;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s >= 1.0 || s == 0.0 {
                continue;
            }
            let factor = (-2.0 * s.ln() / s).sqrt();
            self.spare_normal = Some(v * factor);
            return u * factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference outputs from Vigna's splitmix64.c (seed 1477776061723855037),
    /// as committed in rand_xoshiro's reference test.
    #[test]
    fn splitmix64_matches_reference_vector() {
        let mut rng = SplitMix64::new(1477776061723855037);
        let expected: [u64; 10] = [
            1985237415132408290,
            2979275885539914483,
            13511426838097143398,
            8488337342461049707,
            15141737807933549159,
            17093170987380407015,
            16389528042912955399,
            13177319091862933652,
            10841969400225389492,
            17094824097954834098,
        ];
        for want in expected {
            assert_eq!(rng.next_u64(), want);
        }
    }

    /// Reference outputs from Vigna's xoshiro256plusplus.c with state
    /// [1, 2, 3, 4], as committed in rand_xoshiro's reference test.
    #[test]
    fn xoshiro256pp_matches_reference_vector() {
        let mut rng = Xoshiro256PlusPlus::from_state([1, 2, 3, 4]);
        let expected: [u64; 10] = [
            41943041,
            58720359,
            3588806011781223,
            3591011842654386,
            9228616714210784205,
            9973669472204895162,
            14011001112246962877,
            12406186145184390807,
            15849039046786891736,
            10450023813501588000,
        ];
        for want in expected {
            assert_eq!(rng.next_u64(), want);
        }
    }

    #[test]
    fn uniform_is_in_unit_interval() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        for _ in 0..10_000 {
            let u = rng.uniform();
            assert!((0.0..1.0).contains(&u));
        }
    }

    #[test]
    fn chain_streams_differ_and_are_deterministic() {
        let mut a = Xoshiro256PlusPlus::for_chain(7, 0);
        let mut b = Xoshiro256PlusPlus::for_chain(7, 1);
        let mut a2 = Xoshiro256PlusPlus::for_chain(7, 0);
        let xs: Vec<u64> = (0..4).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..4).map(|_| b.next_u64()).collect();
        let xs2: Vec<u64> = (0..4).map(|_| a2.next_u64()).collect();
        assert_ne!(xs, ys);
        assert_eq!(xs, xs2);
    }

    #[test]
    fn standard_normal_moments_are_sane() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(123);
        let n = 200_000;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let z = rng.standard_normal();
            sum += z;
            sum_sq += z * z;
        }
        let mean = sum / n as f64;
        let var = sum_sq / n as f64 - mean * mean;
        // MC error at n=2e5 is ~1/sqrt(n) ~ 0.0022; 5 sigma margins.
        assert!(mean.abs() < 0.012, "mean {mean}");
        assert!((var - 1.0).abs() < 0.02, "var {var}");
    }
}
