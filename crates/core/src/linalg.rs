//! Minimal dense linear algebra: Cholesky and triangular solves.
//!
//! Matrices are row-major `n x n` slices. This is all `MultivariateNormal`
//! needs: a lower-triangular solve for the density (and its transpose for
//! the gradient), plus a Cholesky factorization for callers that build
//! scale factors from covariances (demo/test tooling).

use crate::error::{Error, ErrorKind};

/// Lower-triangular Cholesky factor of a symmetric positive-definite matrix.
pub fn cholesky(n: usize, a: &[f64]) -> Result<Vec<f64>, Error> {
    assert_eq!(a.len(), n * n);
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(Error::new(
                        ErrorKind::NonFiniteDensity,
                        "matrix is not positive definite; cannot factorize",
                    ));
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Ok(l)
}

/// Solve `L x = b` by forward substitution (lower triangular `L`).
pub fn solve_lower(n: usize, l: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Solve `L^T x = b` by back substitution (lower triangular `L`).
pub fn solve_lower_transpose(n: usize, l: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(l.len(), n * n);
    assert_eq!(b.len(), n);
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: [f64; 9] = [4.0, 2.0, -2.0, 2.0, 10.0, 2.0, -2.0, 2.0, 5.0];

    #[test]
    fn cholesky_recomposes() {
        let l = cholesky(3, &A).unwrap();
        // upper triangle is zero
        assert_eq!(l[1], 0.0);
        assert_eq!(l[2], 0.0);
        assert_eq!(l[5], 0.0);
        for i in 0..3 {
            for j in 0..3 {
                let mut acc = 0.0;
                for k in 0..3 {
                    acc += l[i * 3 + k] * l[j * 3 + k];
                }
                assert!((acc - A[i * 3 + j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn cholesky_rejects_indefinite() {
        let bad = [1.0, 2.0, 2.0, 1.0];
        assert!(cholesky(2, &bad).is_err());
    }

    #[test]
    fn solves_invert_each_other() {
        let l = cholesky(3, &A).unwrap();
        let b = [1.0, -2.0, 3.0];
        let x = solve_lower(3, &l, &b);
        // L x == b
        for i in 0..3 {
            let mut acc = 0.0;
            for k in 0..=i {
                acc += l[i * 3 + k] * x[k];
            }
            assert!((acc - b[i]).abs() < 1e-12);
        }
        let y = solve_lower_transpose(3, &l, &b);
        for i in 0..3 {
            let mut acc = 0.0;
            for k in i..3 {
                acc += l[k * 3 + i] * y[k];
            }
            assert!((acc - b[i]).abs() < 1e-12);
        }
    }
}
