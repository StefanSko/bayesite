//! Special functions: hand-rolled ports, no libm crate.
//!
//! Provenance (see also NOTICE):
//! - `gammaln`/`digamma`: Lanczos approximation, g=7, n=9 ("Godfrey"
//!   coefficients), the same scheme XLA uses for `lgamma`/`digamma`;
//!   reimplemented from the published algorithm (Lanczos 1964, SIAM J.
//!   Numerical Analysis B 1:86-96; coefficient set as tabulated by Godfrey).
//! - `erf`/`erfc`/`ndtr`/`ndtri`: ported from Cephes (Stephen L. Moshier,
//!   netlib.org/cephes, `cprob/ndtr.c` and `cprob/ndtri.c`). Moshier granted
//!   permission to redistribute Cephes-derived work under Apache-2.0-like
//!   terms (see the correspondence quoted in JAX's `jax/_src/scipy/special.py`).
//!
//! All functions are checked against a committed 400-digit mpmath table
//! (`tests/data/special_fn_table.json`).

// Coefficient tables keep the upstream sources' digits verbatim; rustc
// rounds them to the nearest f64 exactly as the C compiler did.
#![allow(clippy::excessive_precision)]

/// ln(2^1024): exp underflow threshold used by Cephes (MAXLOG).
const MAXLOG: f64 = 7.09782712893383996843E2;

const SQRT_2PI: f64 = 2.50662827463100050242E0;

/// Lanczos g.
const LANCZOS_G: f64 = 7.0;
/// Lanczos base coefficient (c0).
const LANCZOS_BASE: f64 = 0.99999999999980993227684700473478;
/// Lanczos series coefficients c1..c8 for g=7, n=9.
const LANCZOS_COEFFS: [f64; 8] = [
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.32342877765313131249651754343,
    -176.61502916214059906584551354002,
    12.507343278686904814458936853317,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7,
];

/// Log-gamma for real arguments (reflection for x < 0.5).
pub fn gammaln(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.5 {
        // Reflection: lgamma(x) = log(pi / |sin(pi x)|) - lgamma(1 - x).
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x == 0.0 {
            return f64::INFINITY; // poles at non-positive integers
        }
        return std::f64::consts::PI.ln() - sin_pi_x.abs().ln() - gammaln(1.0 - x);
    }
    let z = x - 1.0;
    let mut series = LANCZOS_BASE;
    for (i, c) in LANCZOS_COEFFS.iter().enumerate() {
        series += c / (z + (i as f64) + 1.0);
    }
    let g_half = LANCZOS_G + 0.5;
    let t = z + g_half;
    // log(t) via log1p keeps relative precision for large z.
    let log_t = g_half.ln() + (z / g_half).ln_1p();
    SQRT_2PI.ln() + (z + 0.5) * log_t - t + series.ln()
}

/// Digamma (psi) via the derivative of the Lanczos series.
pub fn digamma(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x < 0.5 {
        // Reflection: psi(x) = psi(1 - x) - pi / tan(pi x).
        let tan_pi_x = (std::f64::consts::PI * x).tan();
        if tan_pi_x == 0.0 {
            return f64::NAN; // poles at non-positive integers
        }
        return digamma(1.0 - x) - std::f64::consts::PI / tan_pi_x;
    }
    let z = x - 1.0;
    let mut series = LANCZOS_BASE;
    let mut series_deriv = 0.0;
    for (i, c) in LANCZOS_COEFFS.iter().enumerate() {
        let denom = z + (i as f64) + 1.0;
        series += c / denom;
        series_deriv -= c / (denom * denom);
    }
    let g_half = LANCZOS_G + 0.5;
    let t = z + g_half;
    let log_t = g_half.ln() + (z / g_half).ln_1p();
    log_t - LANCZOS_G / t + series_deriv / series
}

// Cephes erf: |x| < 1, erf(x) = x * T(x^2) / U(x^2).
const ERF_T: [f64; 5] = [
    9.60497373987051638749E0,
    9.00260197203842689217E1,
    2.23200534594684319226E3,
    7.00332514112805075473E3,
    5.55923013010394962768E4,
];
const ERF_U: [f64; 5] = [
    // leading 1.0 implicit (p1evl)
    3.35617141647503099647E1,
    5.21357949780152679795E2,
    4.59432382970980127987E3,
    2.26290000613890934246E4,
    4.92673942608635921086E4,
];

// Cephes erfc: 1 <= x < 8, erfc(x) = exp(-x^2) P(x) / Q(x).
const ERFC_P: [f64; 9] = [
    2.46196981473530512524E-10,
    5.64189564831068821977E-1,
    7.46321056442269912687E0,
    4.86371970985681366614E1,
    1.96520832956077098242E2,
    5.26445194995477358631E2,
    9.34528527171957607540E2,
    1.02755188689515710272E3,
    5.57535335369399327526E2,
];
const ERFC_Q: [f64; 8] = [
    // leading 1.0 implicit (p1evl)
    1.32281951154744992508E1,
    8.67072140885989742329E1,
    3.54937778887819891062E2,
    9.75708501743205489753E2,
    1.82390916687909736289E3,
    2.24633760818710981792E3,
    1.65666309194161350182E3,
    5.57535340817727675546E2,
];

// Cephes erfc: x >= 8, erfc(x) = exp(-x^2) R(x) / S(x).
const ERFC_R: [f64; 6] = [
    5.64189583547755073984E-1,
    1.27536670759978104416E0,
    5.01905042251180477414E0,
    6.16021097993053585195E0,
    7.40974269950448939160E0,
    2.97886665372100240670E0,
];
const ERFC_S: [f64; 6] = [
    // leading 1.0 implicit (p1evl)
    2.26052863220117276590E0,
    9.39603524938001434673E0,
    1.20489539808096656605E1,
    1.70814450747565897222E1,
    9.60896809063285878198E0,
    3.36907645100081516050E0,
];

/// Horner evaluation, coefficients from the highest degree down.
fn polevl(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 0.0;
    for c in coeffs {
        acc = acc * x + c;
    }
    acc
}

/// Horner evaluation with an implicit leading coefficient of 1.
fn p1evl(x: f64, coeffs: &[f64]) -> f64 {
    let mut acc = 1.0;
    for c in coeffs {
        acc = acc * x + c;
    }
    acc
}

/// Error function (Cephes `erf`).
pub fn erf(x: f64) -> f64 {
    if x.abs() > 1.0 {
        return 1.0 - erfc(x);
    }
    let z = x * x;
    x * polevl(z, &ERF_T) / p1evl(z, &ERF_U)
}

/// Complementary error function (Cephes `erfc`).
pub fn erfc(a: f64) -> f64 {
    let x = a.abs();
    if x < 1.0 {
        return 1.0 - erf(a);
    }
    let z = -a * a;
    if z < -MAXLOG {
        // exp underflows: erfc saturates.
        return if a < 0.0 { 2.0 } else { 0.0 };
    }
    let z = z.exp();
    let (p, q) = if x < 8.0 {
        (polevl(x, &ERFC_P), p1evl(x, &ERFC_Q))
    } else {
        (polevl(x, &ERFC_R), p1evl(x, &ERFC_S))
    };
    let mut y = (z * p) / q;
    if a < 0.0 {
        y = 2.0 - y;
    }
    y
}

/// Standard normal CDF (Cephes `ndtr`).
pub fn ndtr(a: f64) -> f64 {
    let x = a * std::f64::consts::FRAC_1_SQRT_2;
    let z = x.abs();
    if z < 1.0 {
        0.5 + 0.5 * erf(x)
    } else {
        let y = 0.5 * erfc(z);
        if x > 0.0 {
            1.0 - y
        } else {
            y
        }
    }
}

// Cephes ndtri rational approximations.
// Central region: x = w + w^3 P0(w^2)/Q0(w^2), scaled by sqrt(2 pi).
const NDTRI_P0: [f64; 5] = [
    -5.99633501014107895267E1,
    9.80010754185999661536E1,
    -5.66762857469070293439E1,
    1.39312609387279679503E1,
    -1.23916583867381258016E0,
];
const NDTRI_Q0: [f64; 8] = [
    // leading 1.0 implicit (p1evl)
    1.95448858338141759834E0,
    4.67627912898881538453E0,
    8.63602421390890590575E1,
    -2.25462687854119370527E2,
    2.00260212380060660359E2,
    -8.20372256168333339912E1,
    1.59056225126211695515E1,
    -1.18331621121330003142E0,
];
// Tail region exp(-2) >= p >= exp(-32): z = sqrt(-2 ln p).
const NDTRI_P1: [f64; 9] = [
    4.05544892305962419923E0,
    3.15251094599893866154E1,
    5.71628192246421288162E1,
    4.40805073893200834700E1,
    1.46849561928858024014E1,
    2.18663306850790267539E0,
    -1.40256079171354495875E-1,
    -3.50424626827848203418E-2,
    -8.57456785154685413611E-4,
];
const NDTRI_Q1: [f64; 8] = [
    // leading 1.0 implicit (p1evl)
    1.57799883256466749731E1,
    4.53907635128879210584E1,
    4.13172038254672030440E1,
    1.50425385692907503408E1,
    2.50464946208309415979E0,
    -1.42182922854787788574E-1,
    -3.80806407691578277194E-2,
    -9.33259480895457427372E-4,
];
// Far tail p < exp(-32).
const NDTRI_P2: [f64; 9] = [
    3.23774891776946035970E0,
    6.91522889068984211695E0,
    3.93881025292474443415E0,
    1.33303460815807542389E0,
    2.01485389549179081538E-1,
    1.23716634817820021358E-2,
    3.01581553508235416007E-4,
    2.65806974686737550832E-6,
    6.23974539184983293730E-9,
];
const NDTRI_Q2: [f64; 8] = [
    // leading 1.0 implicit (p1evl)
    6.02427039364742014255E0,
    3.67983563856160859403E0,
    1.37702099489081330271E0,
    2.16236993594496635890E-1,
    1.34204006088543189037E-2,
    3.28014464682127739104E-4,
    2.89247864745380683936E-6,
    6.79019408009981274425E-9,
];

/// Inverse standard normal CDF (Cephes `ndtri`).
pub fn ndtri(p: f64) -> f64 {
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    let exp_neg_two = (-2.0f64).exp();
    let mut y = p;
    let mut negate = true;
    if y > 1.0 - exp_neg_two {
        y = 1.0 - y;
        negate = false;
    }
    let x = if y > exp_neg_two {
        let w = y - 0.5;
        let ww = w * w;
        let x = w + w * ww * (polevl(ww, &NDTRI_P0) / p1evl(ww, &NDTRI_Q0));
        // Central expansion yields the quantile of (0.5 + w); the final
        // negation below flips it for the lower half.
        -(x * SQRT_2PI)
    } else {
        let z = (-2.0 * y.ln()).sqrt();
        let first = z - z.ln() / z;
        let zr = 1.0 / z;
        let second = if z < 8.0 {
            zr * polevl(zr, &NDTRI_P1) / p1evl(zr, &NDTRI_Q1)
        } else {
            zr * polevl(zr, &NDTRI_P2) / p1evl(zr, &NDTRI_Q2)
        };
        first - second
    };
    if negate {
        -x
    } else {
        x
    }
}

/// x * ln(y) with the convention xlogy(0, y) = 0.
pub fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x * y.ln()
    }
}

/// Numerically stable logistic sigmoid.
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Numerically stable softplus log(1 + exp(x)) = max(x, 0) + log1p(exp(-|x|)).
pub fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}
