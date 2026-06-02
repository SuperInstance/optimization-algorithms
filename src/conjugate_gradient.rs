//! Conjugate gradient methods: Fletcher-Reeves and Polak-Ribiere.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::line_search::backtracking_armijo;

/// Configuration for conjugate gradient optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConjugateGradientConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on gradient norm.
    pub tolerance: f64,
    /// Armijo condition parameter c1.
    pub c1: f64,
    /// Backtracking shrink factor.
    pub rho: f64,
    /// Restart every N iterations (0 = no forced restart).
    pub restart_every: usize,
}

impl Default for ConjugateGradientConfig {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tolerance: 1e-8,
            c1: 1e-4,
            rho: 0.5,
            restart_every: 0,
        }
    }
}

/// Result of conjugate gradient optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CGResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub grad_norm: f64,
    pub converged: bool,
}

/// Conjugate gradient method type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CGMethod {
    /// Fletcher-Reeves: beta = (g_{k+1}^T g_{k+1}) / (g_k^T g_k)
    FletcherReeves,
    /// Polak-Ribiere: beta = (g_{k+1}^T (g_{k+1} - g_k)) / (g_k^T g_k)
    PolakRibiere,
}

/// Nonlinear conjugate gradient method.
pub fn conjugate_gradient<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    method: CGMethod,
    config: &ConjugateGradientConfig,
) -> CGResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut g = grad(&x);
    let mut d = -g.clone();
    let mut iterations = 0;
    let _n = x.len();

    for i in 0..config.max_iter {
        iterations = i + 1;
        let grad_norm = g.norm();

        if grad_norm < config.tolerance {
            return CGResult {
                f_x: f(&x),
                x,
                iterations,
                grad_norm,
                converged: true,
            };
        }

        // Line search
        let ls = backtracking_armijo(f, &g, &x, &d, config.c1, config.rho, 100);
        x = &x + &d.scale(ls.alpha);

        let g_new = grad(&x);
        let beta = match method {
            CGMethod::FletcherReeves => g_new.dot(&g_new) / g.dot(&g),
            CGMethod::PolakRibiere => {
                let beta_pr = g_new.dot(&(&g_new - &g)) / g.dot(&g);
                // Polak-Ribiere with restart: max(0, beta)
                beta_pr.max(0.0)
            }
        };

        // Restart check
        let should_restart = config.restart_every > 0 && (i + 1) % config.restart_every == 0;
        let d_new = if should_restart || beta.is_nan() || beta.is_infinite() {
            -g_new.clone()
        } else {
            -&g_new + d.scale(beta)
        };

        d = d_new;
        g = g_new;
    }

    let grad_norm = g.norm();
    CGResult {
        f_x: f(&x),
        x,
        iterations,
        grad_norm,
        converged: grad_norm < config.tolerance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::{sphere, sphere_grad, rosenbrock, rosenbrock_grad};
    use approx::assert_relative_eq;

    #[test]
    fn test_fr_sphere_converges() {
        let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0]);
        let config = ConjugateGradientConfig {
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = conjugate_gradient(&sphere, &sphere_grad, &x0, CGMethod::FletcherReeves, &config);
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_pr_sphere_converges() {
        let x0 = DVector::from_vec(vec![4.0, -2.0]);
        let config = ConjugateGradientConfig {
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = conjugate_gradient(&sphere, &sphere_grad, &x0, CGMethod::PolakRibiere, &config);
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_fr_rosenbrock() {
        let x0 = DVector::from_vec(vec![-1.0, 1.0]);
        let config = ConjugateGradientConfig {
            max_iter: 20000,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = conjugate_gradient(&rosenbrock, &rosenbrock_grad, &x0, CGMethod::FletcherReeves, &config);
        assert!(result.f_x < 1.0, "Should make progress on Rosenbrock");
    }

    #[test]
    fn test_pr_rosenbrock() {
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let config = ConjugateGradientConfig {
            max_iter: 20000,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = conjugate_gradient(&rosenbrock, &rosenbrock_grad, &x0, CGMethod::PolakRibiere, &config);
        assert!(result.f_x < 1.0, "Should make progress on Rosenbrock");
    }

    #[test]
    fn test_cg_with_restart() {
        let x0 = DVector::from_vec(vec![5.0, 5.0]);
        let config = ConjugateGradientConfig {
            max_iter: 10000,
            tolerance: 1e-8,
            restart_every: 5,
            ..Default::default()
        };
        let result = conjugate_gradient(&sphere, &sphere_grad, &x0, CGMethod::FletcherReeves, &config);
        assert!(result.converged);
    }
}
