//! Line search methods: golden section, backtracking Armijo, Wolfe conditions.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Result of a line search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineSearchResult {
    /// Step size alpha that satisfies the search conditions.
    pub alpha: f64,
    /// Number of function evaluations.
    pub func_evals: usize,
}

/// Golden section search for minimizing a unimodal function on [a, b].
/// Returns the step size that minimizes f(x0 + alpha * d).
pub fn golden_section<F>(
    f: &F,
    x0: &DVector<f64>,
    d: &DVector<f64>,
    a: f64,
    b: f64,
    tol: f64,
) -> LineSearchResult
where
    F: Fn(&DVector<f64>) -> f64,
{
    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = a;
    let mut b = b;
    let mut func_evals = 0;

    let mut x1 = b - golden_ratio * (b - a);
    let mut x2 = a + golden_ratio * (b - a);

    let mut f1 = f(&(x0 + &d.scale(x1)));
    let mut f2 = f(&(x0 + &d.scale(x2)));
    func_evals += 2;

    while (b - a).abs() > tol {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - golden_ratio * (b - a);
            f1 = f(&(x0 + &d.scale(x1)));
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + golden_ratio * (b - a);
            f2 = f(&(x0 + &d.scale(x2)));
        }
        func_evals += 1;
    }

    let alpha = (a + b) / 2.0;
    LineSearchResult { alpha, func_evals }
}

/// Backtracking line search with Armijo (sufficient decrease) condition.
///
/// Starts with alpha = 1.0 and shrinks by factor rho until
/// f(x + alpha * d) <= f(x) + c1 * alpha * grad^T * d.
pub fn backtracking_armijo<F>(
    f: &F,
    grad: &DVector<f64>,
    x: &DVector<f64>,
    d: &DVector<f64>,
    c1: f64,
    rho: f64,
    max_iter: usize,
) -> LineSearchResult
where
    F: Fn(&DVector<f64>) -> f64,
{
    let mut alpha = 1.0;
    let fx = f(x);
    let slope = grad.dot(d);
    let mut func_evals = 1;

    for _ in 0..max_iter {
        let x_new = x + &d.scale(alpha);
        let f_new = f(&x_new);
        func_evals += 1;

        if f_new <= fx + c1 * alpha * slope {
            return LineSearchResult { alpha, func_evals };
        }
        alpha *= rho;
    }

    LineSearchResult { alpha, func_evals }
}

/// Wolfe conditions line search.
///
/// Satisfies both:
/// - Armijo: f(x + alpha*d) <= f(x) + c1 * alpha * grad^T * d
/// - Curvature: |grad(x + alpha*d)^T * d| <= c2 * |grad^T * d|
pub fn wolfe_conditions<F, G>(
    f: &F,
    grad_fn: &G,
    x: &DVector<f64>,
    d: &DVector<f64>,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> LineSearchResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let fx = f(x);
    let grad0 = grad_fn(x);
    let slope0 = grad0.dot(d);
    let mut func_evals = 1;

    // Start with backtracking to satisfy Armijo
    let mut alpha = 1.0;
    let beta_max = 1e10f64;
    let mut _beta_prev = 0.0;
    let mut alpha_prev = 0.0;
    let mut f_prev = fx;

    for _ in 0..max_iter {
        let x_new = x + &d.scale(alpha);
        let f_new = f(&x_new);
        func_evals += 1;

        if f_new > fx + c1 * alpha * slope0 || (f_new >= f_prev && func_evals > 2) {
            return zoom(
                f, grad_fn, x, d, c1, c2, alpha_prev, alpha, f_prev, f_new,
                slope0, &mut func_evals, max_iter,
            );
        }

        let grad_new = grad_fn(&x_new);
        func_evals += 1; // approximate
        let slope_new = grad_new.dot(d);

        if slope_new.abs() <= -c2 * slope0 {
            return LineSearchResult { alpha, func_evals };
        }

        if slope_new >= 0.0 {
            return zoom(
                f, grad_fn, x, d, c1, c2, alpha, alpha_prev, f_new, f_prev,
                slope0, &mut func_evals, max_iter,
            );
        }

        f_prev = f_new;
        alpha_prev = alpha;
        alpha = (alpha + beta_max) / 2.0;
    }

    LineSearchResult { alpha, func_evals }
}

fn zoom<F, G>(
    f: &F,
    grad_fn: &G,
    x: &DVector<f64>,
    d: &DVector<f64>,
    c1: f64,
    c2: f64,
    mut lo: f64,
    mut hi: f64,
    f_lo: f64,
    _f_hi: f64,
    slope0: f64,
    func_evals: &mut usize,
    max_iter: usize,
) -> LineSearchResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let fx = f(x);

    for _ in 0..max_iter {
        let alpha = (lo + hi) / 2.0;
        let x_new = x + &d.scale(alpha);
        let f_new = f(&x_new);
        *func_evals += 1;

        if f_new > fx + c1 * alpha * slope0 || f_new >= f_lo {
            hi = alpha;
        } else {
            let grad_new = grad_fn(&x_new);
            let slope_new = grad_new.dot(d);

            if slope_new.abs() <= -c2 * slope0 {
                return LineSearchResult { alpha, func_evals: *func_evals };
            }

            if slope_new * (hi - lo) >= 0.0 {
                hi = lo;
            }
            lo = alpha;
        }

        if (hi - lo).abs() < 1e-14 {
            break;
        }
    }

    let alpha = (lo + hi) / 2.0;
    LineSearchResult { alpha, func_evals: *func_evals }
}

/// Strong Wolfe conditions line search (curvature condition uses absolute value).
pub fn strong_wolfe<F, G>(
    f: &F,
    grad_fn: &G,
    x: &DVector<f64>,
    d: &DVector<f64>,
    c1: f64,
    c2: f64,
    max_iter: usize,
) -> LineSearchResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    // Use simple backtracking + curvature check approach
    let fx = f(x);
    let grad0 = grad_fn(x);
    let slope0 = grad0.dot(d);
    let mut alpha = 1.0;
    let mut func_evals = 2;

    for _ in 0..max_iter {
        let x_new = x + &d.scale(alpha);
        let f_new = f(&x_new);
        func_evals += 1;

        // Check Armijo
        if f_new > fx + c1 * alpha * slope0 {
            alpha *= 0.5;
            continue;
        }

        // Check strong curvature
        let grad_new = grad_fn(&x_new);
        func_evals += 1;
        let slope_new = grad_new.dot(d);

        if slope_new.abs() <= c2 * slope0.abs() {
            return LineSearchResult { alpha, func_evals };
        }

        alpha *= 0.5;
    }

    LineSearchResult { alpha, func_evals }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::{sphere, sphere_grad};
    use approx::assert_relative_eq;

    #[test]
    fn test_golden_section_sphere() {
        let x0 = DVector::from_vec(vec![2.0]);
        let d = DVector::from_vec(vec![-1.0]); // descent direction
        let result = golden_section(&sphere, &x0, &d, 0.0, 4.0, 1e-10);
        assert_relative_eq!(result.alpha, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_golden_section_2d() {
        let x0 = DVector::from_vec(vec![3.0, 4.0]);
        let d = DVector::from_vec(vec![-3.0, -4.0]);
        let result = golden_section(&sphere, &x0, &d, 0.0, 2.0, 1e-10);
        assert_relative_eq!(result.alpha, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_backtracking_armijo_sufficient_decrease() {
        let x0 = DVector::from_vec(vec![5.0]);
        let grad = sphere_grad(&x0);
        let d = -&grad;
        let result = backtracking_armijo(&sphere, &grad, &x0, &d, 1e-4, 0.5, 100);

        let x_new = &x0 + &d.scale(result.alpha);
        let f_new = sphere(&x_new);
        let f0 = sphere(&x0);
        let slope = grad.dot(&d);
        assert!(f_new <= f0 + 1e-4 * result.alpha * slope, "Armijo condition violated");
    }

    #[test]
    fn test_backtracking_armijo_converges() {
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let grad = sphere_grad(&x0);
        let d = -&grad;
        let result = backtracking_armijo(&sphere, &grad, &x0, &d, 1e-4, 0.5, 100);
        let x_new = &x0 + &d.scale(result.alpha);
        assert!(sphere(&x_new) < sphere(&x0));
    }

    #[test]
    fn test_wolfe_conditions_satisfied() {
        let x0 = DVector::from_vec(vec![3.0, -2.0]);
        let grad = sphere_grad(&x0);
        let d = -&grad;
        let result = wolfe_conditions(
            &sphere, &sphere_grad, &x0, &d, 1e-4, 0.9, 100,
        );

        let x_new = &x0 + &d.scale(result.alpha);
        let f_new = sphere(&x_new);
        let f0 = sphere(&x0);
        let slope0 = grad.dot(&d);

        // Armijo
        assert!(f_new <= f0 + 1e-4 * result.alpha * slope0);

        // Curvature
        let grad_new = sphere_grad(&x_new);
        let slope_new = grad_new.dot(&d);
        assert!(slope_new.abs() <= 0.9 * slope0.abs() + 1e-6);
    }

    #[test]
    fn test_strong_wolfe_conditions() {
        let x0 = DVector::from_vec(vec![2.0, 3.0]);
        let grad = sphere_grad(&x0);
        let d = -&grad;
        let result = strong_wolfe(
            &sphere, &sphere_grad, &x0, &d, 1e-4, 0.9, 100,
        );

        let x_new = &x0 + &d.scale(result.alpha);
        let grad_new = sphere_grad(&x_new);
        let slope_new = grad_new.dot(&d);
        assert!(slope_new.abs() <= 0.9 * grad.dot(&d).abs() + 1e-6);
    }

    #[test]
    fn test_golden_section_few_evals() {
        let x0 = DVector::from_vec(vec![1.0]);
        let d = DVector::from_vec(vec![-1.0]);
        let result = golden_section(&sphere, &x0, &d, 0.0, 3.0, 1e-6);
        assert!(result.func_evals < 100);
    }
}
