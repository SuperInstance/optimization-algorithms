//! Quasi-Newton methods: BFGS and L-BFGS.

use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};
use crate::line_search::backtracking_armijo;

/// Configuration for BFGS optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFGSConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on gradient norm.
    pub tolerance: f64,
    /// Armijo c1 parameter.
    pub c1: f64,
    /// Backtracking shrink factor.
    pub rho: f64,
}

impl Default for BFGSConfig {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tolerance: 1e-8,
            c1: 1e-4,
            rho: 0.5,
        }
    }
}

/// Result of BFGS optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFGSResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub grad_norm: f64,
    pub converged: bool,
    /// The final approximate inverse Hessian.
    pub h_inv: DMatrix<f64>,
}

/// BFGS quasi-Newton optimization.
pub fn bfgs<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    config: &BFGSConfig,
) -> BFGSResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let n = x0.len();
    let mut x = x0.clone();
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = grad(&x);
    let mut iterations = 0;

    for i in 0..config.max_iter {
        iterations = i + 1;
        let grad_norm = g.norm();

        if grad_norm < config.tolerance {
            return BFGSResult {
                f_x: f(&x),
                x,
                iterations,
                grad_norm,
                converged: true,
                h_inv,
            };
        }

        // Search direction: d = -H^{-1} g
        let d = -&h_inv * &g;

        // Line search
        let ls = backtracking_armijo(f, &g, &x, &d, config.c1, config.rho, 100);
        let s = d.scale(ls.alpha);
        let x_new = &x + &s;
        let g_new = grad(&x_new);
        let y = &g_new - &g;

        // BFGS update: H_{k+1}^{-1} = (I - rho*s*y^T) H (I - rho*y*s^T) + rho*s*s^T
        let ys = y.dot(&s);
        if ys > 1e-20 {
            let rho_bfgs = 1.0 / ys;
            let i_n = DMatrix::identity(n, n);
            let a = &i_n - rho_bfgs * &s * &y.transpose();
            let b = &i_n - rho_bfgs * &y * &s.transpose();
            h_inv = &a * &h_inv * &b + rho_bfgs * &s * &s.transpose();
        }

        x = x_new;
        g = g_new;
    }

    let grad_norm = g.norm();
    BFGSResult {
        f_x: f(&x),
        x,
        iterations,
        grad_norm,
        converged: grad_norm < config.tolerance,
        h_inv,
    }
}

/// Configuration for L-BFGS optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LBFGSConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on gradient norm.
    pub tolerance: f64,
    /// Number of corrections to store (memory parameter m).
    pub memory_size: usize,
    /// Armijo c1 parameter.
    pub c1: f64,
    /// Backtracking shrink factor.
    pub rho: f64,
}

impl Default for LBFGSConfig {
    fn default() -> Self {
        Self {
            max_iter: 10000,
            tolerance: 1e-8,
            memory_size: 10,
            c1: 1e-4,
            rho: 0.5,
        }
    }
}

/// Result of L-BFGS optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LBFGSResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub grad_norm: f64,
    pub converged: bool,
}

/// Two-loop recursion for L-BFGS direction computation.
fn lbfgs_two_loop(
    grad: &DVector<f64>,
    s_history: &[DVector<f64>],
    y_history: &[DVector<f64>],
    rho_history: &[f64],
) -> DVector<f64> {
    let m = s_history.len();
    if m == 0 {
        return -grad.clone();
    }

    let mut q = grad.clone();
    let mut alpha = vec![0.0; m];

    // First loop: backward
    for i in (0..m).rev() {
        alpha[i] = rho_history[i] * s_history[i].dot(&q);
        q = q - y_history[i].scale(alpha[i]);
    }

    // Initial Hessian approximation (scaled identity)
    let gamma = s_history[m - 1].dot(&y_history[m - 1])
        / y_history[m - 1].dot(&y_history[m - 1]);
    let mut d = q.scale(gamma);

    // Second loop: forward
    for i in 0..m {
        let beta = rho_history[i] * y_history[i].dot(&d);
        d = d + s_history[i].scale(alpha[i] - beta);
    }

    -d
}

/// L-BFGS (Limited-memory BFGS) optimization.
pub fn lbfgs<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    config: &LBFGSConfig,
) -> LBFGSResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut g = grad(&x);
    let mut iterations = 0;

    let mut s_history: Vec<DVector<f64>> = Vec::new();
    let mut y_history: Vec<DVector<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for i in 0..config.max_iter {
        iterations = i + 1;
        let grad_norm = g.norm();

        if grad_norm < config.tolerance {
            return LBFGSResult {
                f_x: f(&x),
                x,
                iterations,
                grad_norm,
                converged: true,
            };
        }

        // Compute direction using two-loop recursion
        let d = lbfgs_two_loop(&g, &s_history, &y_history, &rho_history);

        // Line search
        let ls = backtracking_armijo(f, &g, &x, &d, config.c1, config.rho, 100);
        let s = d.scale(ls.alpha);
        let x_new = &x + &s;
        let g_new = grad(&x_new);
        let y = &g_new - &g;

        let ys = y.dot(&s);
        if ys > 1e-20 {
            if s_history.len() >= config.memory_size {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(1.0 / ys);
        }

        x = x_new;
        g = g_new;
    }

    let grad_norm = g.norm();
    LBFGSResult {
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
    fn test_bfgs_sphere_converges() {
        let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0]);
        let result = bfgs(&sphere, &sphere_grad, &x0, &BFGSConfig::default());
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_bfgs_positive_definite_hessian() {
        let x0 = DVector::from_vec(vec![2.0, 3.0]);
        let result = bfgs(&sphere, &sphere_grad, &x0, &BFGSConfig::default());
        // The inverse Hessian should be positive definite (all eigenvalues > 0)
        let eig = result.h_inv.symmetric_eigenvalues();
        for i in 0..eig.len() {
            assert!(eig[i] > 0.0, "H_inv should be positive definite");
        }
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let x0 = DVector::from_vec(vec![-1.0, 1.0]);
        let result = bfgs(&rosenbrock, &rosenbrock_grad, &x0, &BFGSConfig {
            max_iter: 50000,
            ..Default::default()
        });
        assert!(result.f_x < 0.01, "BFGS should converge on Rosenbrock, got f={}", result.f_x);
    }

    #[test]
    fn test_lbfgs_sphere_converges() {
        let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0]);
        let result = lbfgs(&sphere, &sphere_grad, &x0, &LBFGSConfig::default());
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn test_lbfgs_rosenbrock() {
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let result = lbfgs(&rosenbrock, &rosenbrock_grad, &x0, &LBFGSConfig {
            max_iter: 50000,
            ..Default::default()
        });
        assert!(result.f_x < 0.01, "L-BFGS should converge on Rosenbrock, got f={}", result.f_x);
    }

    #[test]
    fn test_lbfgs_high_dimensional() {
        // 10D sphere
        let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0, -1.0, 4.0, -2.0, 3.0, -4.0, 1.0, -5.0]);
        let result = lbfgs(&sphere, &sphere_grad, &x0, &LBFGSConfig {
            memory_size: 5,
            ..Default::default()
        });
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_bfgs_preserves_convergence_with_restarts() {
        let x0 = DVector::from_vec(vec![10.0, -10.0]);
        let config = BFGSConfig {
            max_iter: 50000,
            ..Default::default()
        };
        let result = bfgs(&sphere, &sphere_grad, &x0, &config);
        assert!(result.converged);
        assert!(result.x.norm() < 0.01);
    }
}
