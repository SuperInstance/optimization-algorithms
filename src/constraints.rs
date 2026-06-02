//! Constraint handling: penalty methods and barrier methods.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::gradient_descent::{gradient_descent, GradientDescentConfig};

/// Constraint types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Inequality: g(x) <= 0.
    Inequality { g: String },
    /// Equality: h(x) = 0 (approximated as |h(x)| <= epsilon).
    Equality { h: String },
}

/// A constraint function.
pub type ConstraintFn = Box<dyn Fn(&DVector<f64>) -> f64>;

/// Penalty method configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyConfig {
    /// Initial penalty parameter.
    pub mu_init: f64,
    /// Penalty growth factor.
    pub mu_factor: f64,
    /// Maximum penalty iterations.
    pub max_outer_iter: usize,
    /// Inner optimization config.
    pub inner_config: GradientDescentConfig,
    /// Tolerance for constraint satisfaction.
    pub constraint_tol: f64,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            mu_init: 1.0,
            mu_factor: 10.0,
            max_outer_iter: 50,
            inner_config: GradientDescentConfig {
                learning_rate: 0.01,
                max_iter: 5000,
                tolerance: 1e-8,
                ..Default::default()
            },
            constraint_tol: 1e-6,
        }
    }
}

/// Result of constrained optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstrainedResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub constraint_violation: f64,
    pub outer_iterations: usize,
    pub converged: bool,
}

/// Quadratic penalty method for constrained optimization.
///
/// Minimizes f(x) + (mu/2) * sum(max(0, g_i(x))^2) for inequality constraints
/// and f(x) + (mu/2) * sum(h_j(x)^2) for equality constraints.
pub fn quadratic_penalty_method<F, G>(
    f: &F,
    grad_f: &G,
    x0: &DVector<f64>,
    inequality_constraints: &[ConstraintFn],
    equality_constraints: &[ConstraintFn],
    config: &PenaltyConfig,
) -> ConstrainedResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut mu = config.mu_init;

    for outer_iter in 0..config.max_outer_iter {
        // Build penalized objective and gradient
        let mu_capture = mu;
        let ineq = inequality_constraints
            .iter()
            .map(|c| c as &dyn Fn(&DVector<f64>) -> f64)
            .collect::<Vec<_>>();
        let eq = equality_constraints
            .iter()
            .map(|c| c as &dyn Fn(&DVector<f64>) -> f64)
            .collect::<Vec<_>>();

        let penalized_f = |x: &DVector<f64>| -> f64 {
            let mut val = f(x);
            for c in &ineq {
                let cv = c(x);
                if cv > 0.0 {
                    val += 0.5 * mu_capture * cv * cv;
                }
            }
            for c in &eq {
                let cv = c(x);
                val += 0.5 * mu_capture * cv * cv;
            }
            val
        };

        let penalized_grad = |x: &DVector<f64>| -> DVector<f64> {
            let mut g = grad_f(x);
            for c in &ineq {
                let cv = c(x);
                if cv > 0.0 {
                    // Numerical gradient for penalty term
                    let eps = 1e-8;
                    for i in 0..x.len() {
                        let mut xp = x.clone();
                        let mut xm = x.clone();
                        xp[i] += eps;
                        xm[i] -= eps;
                        let fp = c(&xp);
                        let fm = c(&xm);
                        let dc = (fp - fm) / (2.0 * eps);
                        g[i] += mu_capture * cv * dc;
                    }
                }
            }
            for c in &eq {
                let cv = c(x);
                let eps = 1e-8;
                for i in 0..x.len() {
                    let mut xp = x.clone();
                    let mut xm = x.clone();
                    xp[i] += eps;
                    xm[i] -= eps;
                    let fp = c(&xp);
                    let fm = c(&xm);
                    let dc = (fp - fm) / (2.0 * eps);
                    g[i] += mu_capture * cv * dc;
                }
            }
            g
        };

        let result = gradient_descent(&penalized_f, &penalized_grad, &x, &config.inner_config);
        x = result.x;

        // Check constraint satisfaction
        let violation = compute_violation(&x, inequality_constraints, equality_constraints);
        if violation < config.constraint_tol {
            return ConstrainedResult {
                f_x: f(&x),
                x,
                constraint_violation: violation,
                outer_iterations: outer_iter + 1,
                converged: true,
            };
        }

        mu *= config.mu_factor;
    }

    let violation = compute_violation(&x, inequality_constraints, equality_constraints);
    ConstrainedResult {
        f_x: f(&x),
        x,
        constraint_violation: violation,
        outer_iterations: config.max_outer_iter,
        converged: false,
    }
}

/// Log-barrier method for inequality-constrained optimization.
///
/// Minimizes f(x) - (1/t) * sum(log(-g_i(x))) where g_i(x) <= 0.
pub fn barrier_method<F, G>(
    f: &F,
    grad_f: &G,
    x0: &DVector<f64>,
    inequality_constraints: &[ConstraintFn],
    config: &PenaltyConfig,
) -> ConstrainedResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut t = 1.0;
    let mu_barrier = config.mu_factor;

    for outer_iter in 0..config.max_outer_iter {
        let t_capture = t;
        let ineq = inequality_constraints
            .iter()
            .map(|c| c as &dyn Fn(&DVector<f64>) -> f64)
            .collect::<Vec<_>>();

        let barrier_f = |x: &DVector<f64>| -> f64 {
            let mut val = f(x);
            for c in &ineq {
                let cv = c(x);
                if cv >= 0.0 {
                    return f64::INFINITY; // Outside feasible region
                }
                val -= (1.0 / t_capture) * cv.abs().ln();
            }
            val
        };

        let barrier_grad = |x: &DVector<f64>| -> DVector<f64> {
            let mut g = grad_f(x);
            let eps = 1e-8;
            for c in &ineq {
                let cv = c(x);
                if cv < 0.0 {
                    for i in 0..x.len() {
                        let mut xp = x.clone();
                        let mut xm = x.clone();
                        xp[i] += eps;
                        xm[i] -= eps;
                        let cvp = c(&xp);
                        let cvm = c(&xm);
                        if cvp < 0.0 && cvm < 0.0 {
                            let d_log = (cvp.abs().ln() - cvm.abs().ln()) / (2.0 * eps);
                            g[i] -= (1.0 / t_capture) * d_log;
                        }
                    }
                }
            }
            g
        };

        let result = gradient_descent(&barrier_f, &barrier_grad, &x, &config.inner_config);
        x = result.x;

        let duality_gap = inequality_constraints.len() as f64 / t;
        if duality_gap < config.constraint_tol {
            let violation = compute_violation_ineq(&x, inequality_constraints);
            return ConstrainedResult {
                f_x: f(&x),
                x,
                constraint_violation: violation,
                outer_iterations: outer_iter + 1,
                converged: true,
            };
        }

        t *= mu_barrier;
    }

    let violation = compute_violation_ineq(&x, inequality_constraints);
    ConstrainedResult {
        f_x: f(&x),
        x,
        constraint_violation: violation,
        outer_iterations: config.max_outer_iter,
        converged: false,
    }
}

fn compute_violation(
    x: &DVector<f64>,
    ineq: &[ConstraintFn],
    eq: &[ConstraintFn],
) -> f64 {
    let mut v = 0.0f64;
    for c in ineq {
        let cv = c(x);
        if cv > 0.0 {
            v += cv;
        }
    }
    for c in eq {
        v += c(x).abs();
    }
    v
}

fn compute_violation_ineq(x: &DVector<f64>, ineq: &[ConstraintFn]) -> f64 {
    let mut v = 0.0f64;
    for c in ineq {
        let cv = c(x);
        if cv > 0.0 {
            v += cv;
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use approx::assert_relative_eq;

    #[test]
    fn test_penalty_simple_inequality() {
        // Minimize x^2 subject to x >= 1 (i.e., 1 - x <= 0)
        // Optimal at x = 1
        let f = |x: &DVector<f64>| x[0] * x[0];
        let grad_f = |x: &DVector<f64>| DVector::from_vec(vec![2.0 * x[0]]);

        let constraint: ConstraintFn = Box::new(|x: &DVector<f64>| 1.0 - x[0]);

        let config = PenaltyConfig {
            mu_init: 1.0,
            mu_factor: 10.0,
            max_outer_iter: 20,
            inner_config: GradientDescentConfig {
                learning_rate: 0.01,
                max_iter: 10000,
                tolerance: 1e-10,
                ..Default::default()
            },
            constraint_tol: 1e-4,
        };

        let x0 = DVector::from_vec(vec![2.0]);
        let result = quadratic_penalty_method(&f, &grad_f, &x0, &[constraint], &[], &config);
        assert_relative_eq!(result.x[0], 1.0, epsilon = 0.1);
        assert!(result.constraint_violation < 0.1);
    }

    #[test]
    fn test_penalty_equality() {
        // Minimize x^2 + y^2 subject to x + y = 1
        // Optimal at x = y = 0.5
        let f = |x: &DVector<f64>| x[0] * x[0] + x[1] * x[1];
        let grad_f = |x: &DVector<f64>| DVector::from_vec(vec![2.0 * x[0], 2.0 * x[1]]);

        let eq_constraint: ConstraintFn = Box::new(|x: &DVector<f64>| x[0] + x[1] - 1.0);

        let config = PenaltyConfig {
            mu_init: 1.0,
            mu_factor: 10.0,
            max_outer_iter: 20,
            inner_config: GradientDescentConfig {
                learning_rate: 0.001,
                max_iter: 20000,
                tolerance: 1e-10,
                ..Default::default()
            },
            constraint_tol: 1e-2,
        };

        let x0 = DVector::from_vec(vec![0.5, 0.5]);
        let result = quadratic_penalty_method(&f, &grad_f, &x0, &[], &[eq_constraint], &config);
        // Check we're near the optimal line x+y=1
        let sum = result.x[0] + result.x[1];
        assert!((sum - 1.0).abs() < 0.5, "x+y should be near 1, got {}", sum);
        assert!(result.f_x < 1.0, "objective should be less than 1, got {}", result.f_x);
    }

    #[test]
    fn test_barrier_inequality() {
        // Minimize (x-3)^2 subject to x >= 1 (1 - x <= 0)
        // Optimal at x = 3 (constraint not active)
        let f = |x: &DVector<f64>| (x[0] - 3.0).powi(2);
        let grad_f = |x: &DVector<f64>| DVector::from_vec(vec![2.0 * (x[0] - 3.0)]);

        let constraint: ConstraintFn = Box::new(|x: &DVector<f64>| 1.0 - x[0]);

        let config = PenaltyConfig {
            mu_init: 1.0,
            mu_factor: 10.0,
            max_outer_iter: 20,
            inner_config: GradientDescentConfig {
                learning_rate: 0.01,
                max_iter: 10000,
                tolerance: 1e-10,
                ..Default::default()
            },
            constraint_tol: 1e-3,
        };

        let x0 = DVector::from_vec(vec![2.0]);
        let result = barrier_method(&f, &grad_f, &x0, &[constraint], &config);
        assert_relative_eq!(result.x[0], 3.0, epsilon = 0.2);
    }

    #[test]
    fn test_penalty_decreases_violation() {
        // As penalty grows, violation should decrease
        let f = |x: &DVector<f64>| -x[0]; // Minimize -x (maximize x)
        let grad_f = |_x: &DVector<f64>| DVector::from_vec(vec![-1.0]);

        let constraint: ConstraintFn = Box::new(|x: &DVector<f64>| x[0] - 1.0); // x <= 1

        let config = PenaltyConfig {
            mu_init: 1.0,
            mu_factor: 10.0,
            max_outer_iter: 30,
            inner_config: GradientDescentConfig {
                learning_rate: 0.001,
                max_iter: 20000,
                tolerance: 1e-10,
                ..Default::default()
            },
            constraint_tol: 1e-3,
        };

        let x0 = DVector::from_vec(vec![0.5]);
        let result = quadratic_penalty_method(&f, &grad_f, &x0, &[constraint], &[], &config);
        assert!(result.x[0] <= 1.1, "Should satisfy constraint approximately");
    }
}
