//! Test functions for optimization benchmarks.

use nalgebra::DVector;

/// Sphere function: f(x) = sum(x_i^2). Minimum at origin with value 0.
pub fn sphere(x: &DVector<f64>) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Sphere gradient.
pub fn sphere_grad(x: &DVector<f64>) -> DVector<f64> {
    2.0 * x
}

/// Rosenbrock function: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2).
/// Minimum at (1, 1, ..., 1) with value 0.
pub fn rosenbrock(x: &DVector<f64>) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() - 1 {
        let xi = x[i];
        let xi1 = x[i + 1];
        sum += 100.0 * (xi1 - xi * xi).powi(2) + (1.0 - xi).powi(2);
    }
    sum
}

/// Rosenbrock gradient.
pub fn rosenbrock_grad(x: &DVector<f64>) -> DVector<f64> {
    let n = x.len();
    let mut grad = DVector::zeros(n);
    for i in 0..n {
        if i < n - 1 {
            let xi = x[i];
            let xi1 = x[i + 1];
            grad[i] += -400.0 * xi * (xi1 - xi * xi) - 2.0 * (1.0 - xi);
        }
        if i > 0 {
            let xi_prev = x[i - 1];
            let xi = x[i];
            grad[i] += 200.0 * (xi - xi_prev * xi_prev);
        }
    }
    grad
}

/// Rastrigin function: f(x) = 10*n + sum(x_i^2 - 10*cos(2*pi*x_i)).
/// Minimum at origin with value 0.
pub fn rastrigin(x: &DVector<f64>) -> f64 {
    let n = x.len() as f64;
    10.0 * n + x.iter().map(|&xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos()).sum::<f64>()
}

/// Rastrigin gradient.
pub fn rastrigin_grad(x: &DVector<f64>) -> DVector<f64> {
    x.map(|xi| 2.0 * xi + 20.0 * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin())
}

/// Booth function: 2D only. Minimum at (1, 3) with value 0.
pub fn booth(x: &DVector<f64>) -> f64 {
    let (x1, x2) = (x[0], x[1]);
    (x1 + 2.0 * x2 - 7.0).powi(2) + (2.0 * x1 + x2 - 5.0).powi(2)
}

/// Quadratic function: f(x) = 0.5 * x^T A x - b^T x. Minimum at A^{-1}b.
pub fn quadratic(x: &DVector<f64>, a: &nalgebra::DMatrix<f64>, b: &DVector<f64>) -> f64 {
    let v: f64 = (0.5 * x.transpose() * a * x - b.transpose() * x)[(0, 0)];
    v
}

/// Ackley function. Minimum at origin with value 0.
pub fn ackley(x: &DVector<f64>) -> f64 {
    let n = x.len() as f64;
    let sum_sq: f64 = x.iter().map(|xi| xi * xi).sum();
    let sum_cos: f64 = x.iter().map(|&xi| (2.0 * std::f64::consts::PI * xi).cos()).sum();
    -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp()
        - (sum_cos / n).exp()
        + 20.0
        + std::f64::consts::E
}
