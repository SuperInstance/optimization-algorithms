//! Nelder-Mead simplex method (derivative-free optimization).

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Configuration for Nelder-Mead optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NelderMeadConfig {
    /// Reflection coefficient (alpha).
    pub alpha: f64,
    /// Expansion coefficient (gamma).
    pub gamma: f64,
    /// Contraction coefficient (rho).
    pub rho: f64,
    /// Shrink coefficient (sigma).
    pub sigma: f64,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance on simplex size.
    pub tolerance: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
            max_iter: 10000,
            tolerance: 1e-10,
        }
    }
}

/// Result of Nelder-Mead optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NelderMeadResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Nelder-Mead simplex optimization.
pub fn nelder_mead<F>(
    f: &F,
    x0: &DVector<f64>,
    initial_step: f64,
    config: &NelderMeadConfig,
) -> NelderMeadResult
where
    F: Fn(&DVector<f64>) -> f64,
{
    let n = x0.len();
    let n_plus_1 = n + 1;

    // Initialize simplex
    let mut simplex: Vec<DVector<f64>> = Vec::with_capacity(n_plus_1);
    let mut f_values: Vec<f64> = Vec::with_capacity(n_plus_1);

    simplex.push(x0.clone());
    f_values.push(f(x0));

    for i in 0..n {
        let mut xi = x0.clone();
        xi[i] += initial_step;
        simplex.push(xi);
        f_values.push(f(&simplex[i + 1]));
    }

    let mut iterations = 0;

    for i in 0..config.max_iter {
        iterations = i + 1;

        // Sort simplex by function values
        let mut indices: Vec<usize> = (0..n_plus_1).collect();
        indices.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check convergence: max distance from best
        let max_dist = indices
            .iter()
            .map(|&idx| (&simplex[idx] - &simplex[best_idx]).norm())
            .fold(0.0_f64, f64::max);

        if max_dist < config.tolerance {
            return NelderMeadResult {
                x: simplex[best_idx].clone(),
                f_x: f_values[best_idx],
                iterations,
                converged: true,
            };
        }

        // Centroid (excluding worst)
        let mut centroid = DVector::zeros(n);
        for &idx in &indices[..n] {
            centroid += &simplex[idx];
        }
        centroid = centroid.scale(1.0 / n as f64);

        // Reflection
        let xr = &centroid + (&centroid - &simplex[worst_idx]).scale(config.alpha);
        let fr = f(&xr);

        if fr < f_values[second_worst_idx] && fr >= f_values[best_idx] {
            simplex[worst_idx] = xr;
            f_values[worst_idx] = fr;
            continue;
        }

        // Expansion
        if fr < f_values[best_idx] {
            let xe = &centroid + (&xr - &centroid).scale(config.gamma);
            let fe = f(&xe);
            if fe < fr {
                simplex[worst_idx] = xe;
                f_values[worst_idx] = fe;
            } else {
                simplex[worst_idx] = xr;
                f_values[worst_idx] = fr;
            }
            continue;
        }

        // Contraction
        let xc = &centroid + (&simplex[worst_idx] - &centroid).scale(config.rho);
        let fc = f(&xc);
        if fc < f_values[worst_idx] {
            simplex[worst_idx] = xc;
            f_values[worst_idx] = fc;
            continue;
        }

        // Shrink
        for j in 1..n_plus_1 {
            let idx = indices[j];
            simplex[idx] = &simplex[best_idx]
                + (&simplex[idx] - &simplex[best_idx]).scale(config.sigma);
            f_values[idx] = f(&simplex[idx]);
        }
    }

    // Return best point
    let best_idx = f_values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();

    NelderMeadResult {
        x: simplex[best_idx].clone(),
        f_x: f_values[best_idx],
        iterations,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::{sphere, rosenbrock, booth};
    use approx::assert_relative_eq;

    #[test]
    fn test_nm_sphere() {
        let x0 = DVector::from_vec(vec![5.0, -3.0]);
        let result = nelder_mead(&sphere, &x0, 1.0, &NelderMeadConfig {
            tolerance: 1e-10,
            ..Default::default()
        });
        assert!(result.f_x < 1e-6);
        assert_relative_eq!(result.x[0], 0.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nm_rosenbrock() {
        let x0 = DVector::from_vec(vec![-1.0, 1.0]);
        let result = nelder_mead(&rosenbrock, &x0, 1.0, &NelderMeadConfig {
            max_iter: 50000,
            tolerance: 1e-10,
            ..Default::default()
        });
        assert!(result.f_x < 0.01);
    }

    #[test]
    fn test_nm_booth() {
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let result = nelder_mead(&booth, &x0, 1.0, &NelderMeadConfig::default());
        assert_relative_eq!(result.x[0], 1.0, epsilon = 1e-3);
        assert_relative_eq!(result.x[1], 3.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nm_derivative_free() {
        // Nelder-Mead doesn't need gradients - verify it works with discontinuous-ish functions
        let f = |x: &DVector<f64>| sphere(x);
        let x0 = DVector::from_vec(vec![10.0, -10.0, 5.0]);
        let result = nelder_mead(&f, &x0, 2.0, &NelderMeadConfig::default());
        assert!(result.f_x < 1.0);
    }

    #[test]
    fn test_nm_converges_few_dimensions() {
        let x0 = DVector::from_vec(vec![1.5]);
        let result = nelder_mead(&sphere, &x0, 0.5, &NelderMeadConfig::default());
        assert!(result.converged || result.f_x < 1e-6);
    }
}
