//! Gradient descent variants: vanilla, stochastic, mini-batch, momentum, Nesterov.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};

/// Configuration for gradient descent optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientDescentConfig {
    /// Learning rate / step size.
    pub learning_rate: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance on gradient norm.
    pub tolerance: f64,
    /// Momentum coefficient (0.0 = no momentum).
    pub momentum: f64,
    /// Use Nesterov accelerated gradient.
    pub nesterov: bool,
    /// Verbosity level (0 = silent).
    pub verbose: usize,
}

impl Default for GradientDescentConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            max_iter: 10000,
            tolerance: 1e-8,
            momentum: 0.0,
            nesterov: false,
            verbose: 0,
        }
    }
}

/// Result of gradient descent optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimal point found.
    pub x: DVector<f64>,
    /// Function value at optimal point.
    pub f_x: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Gradient norm at termination.
    pub grad_norm: f64,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Standard (batch) gradient descent with optional momentum and Nesterov.
pub fn gradient_descent<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    config: &GradientDescentConfig,
) -> OptimizationResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut velocity = DVector::zeros(x.len());
    let mut iterations = 0;

    for i in 0..config.max_iter {
        iterations = i + 1;

        let eval_point = if config.nesterov && config.momentum > 0.0 {
            &x + &velocity.scale(config.momentum)
        } else {
            x.clone()
        };

        let g = grad(&eval_point);
        let grad_norm = g.norm();

        if grad_norm < config.tolerance {
            return OptimizationResult {
                f_x: f(&x),
                x,
                iterations,
                grad_norm,
                converged: true,
            };
        }

        if config.momentum > 0.0 {
            velocity = velocity.scale(config.momentum) - g.scale(config.learning_rate);
            x = &x + &velocity;
        } else {
            x = x - g.scale(config.learning_rate);
        }
    }

    let g = grad(&x);
    OptimizationResult {
        f_x: f(&x),
        x,
        iterations,
        grad_norm: g.norm(),
        converged: false,
    }
}

/// Stochastic gradient descent using sample-based gradients.
pub fn stochastic_gradient_descent<F, G>(
    f: &F,
    grad_fn: &G,
    x0: &DVector<f64>,
    config: &GradientDescentConfig,
    num_samples: usize,
) -> OptimizationResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>, usize) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut velocity = DVector::zeros(x.len());
    let mut iterations = 0;
    let mut rng = rand::thread_rng();

    for i in 0..config.max_iter {
        iterations = i + 1;
        let sample_idx = rand::Rng::gen_range(&mut rng, 0..num_samples);
        let g = grad_fn(&x, sample_idx);

        if config.nesterov && config.momentum > 0.0 {
            velocity = velocity.scale(config.momentum) - g.scale(config.learning_rate);
            x = &x + &velocity;
        } else if config.momentum > 0.0 {
            velocity = velocity.scale(config.momentum) - g.scale(config.learning_rate);
            x = &x + &velocity;
        } else {
            x = x - g.scale(config.learning_rate);
        }
    }

    let g_full = grad_fn(&x, 0); // Use sample 0 as proxy for full gradient norm
    OptimizationResult {
        f_x: f(&x),
        x,
        iterations,
        grad_norm: g_full.norm(),
        converged: g_full.norm() < config.tolerance,
    }
}

/// Mini-batch gradient descent.
pub fn mini_batch_gradient_descent<F, G>(
    f: &F,
    grad_fn: &G,
    x0: &DVector<f64>,
    config: &GradientDescentConfig,
    num_samples: usize,
    batch_size: usize,
) -> OptimizationResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>, &[usize]) -> DVector<f64>,
{
    let mut x = x0.clone();
    let mut velocity = DVector::zeros(x.len());
    let mut iterations = 0;
    let mut rng = rand::thread_rng();

    for i in 0..config.max_iter {
        iterations = i + 1;

        // Sample mini-batch
        let mut batch: Vec<usize> = (0..num_samples).collect();
        use rand::seq::SliceRandom;
        batch.shuffle(&mut rng);
        batch.truncate(batch_size.min(num_samples));

        let g = grad_fn(&x, &batch);

        if config.momentum > 0.0 {
            velocity = velocity.scale(config.momentum) - g.scale(config.learning_rate);
            x = &x + &velocity;
        } else {
            x = x - g.scale(config.learning_rate);
        }
    }

    let all_indices: Vec<usize> = (0..num_samples).collect();
    let g_full = grad_fn(&x, &all_indices);
    OptimizationResult {
        f_x: f(&x),
        x,
        iterations,
        grad_norm: g_full.norm(),
        converged: g_full.norm() < config.tolerance,
    }
}

/// Momentum gradient descent (alias for gradient_descent with momentum).
pub fn momentum_gradient_descent<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    learning_rate: f64,
    momentum: f64,
    max_iter: usize,
    tolerance: f64,
) -> OptimizationResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let config = GradientDescentConfig {
        learning_rate,
        max_iter,
        tolerance,
        momentum,
        nesterov: false,
        verbose: 0,
    };
    gradient_descent(f, grad, x0, &config)
}

/// Nesterov accelerated gradient descent.
pub fn nesterov_gradient_descent<F, G>(
    f: &F,
    grad: &G,
    x0: &DVector<f64>,
    learning_rate: f64,
    momentum: f64,
    max_iter: usize,
    tolerance: f64,
) -> OptimizationResult
where
    F: Fn(&DVector<f64>) -> f64,
    G: Fn(&DVector<f64>) -> DVector<f64>,
{
    let config = GradientDescentConfig {
        learning_rate,
        max_iter,
        tolerance,
        momentum,
        nesterov: true,
        verbose: 0,
    };
    gradient_descent(f, grad, x0, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::{sphere, sphere_grad, rosenbrock, rosenbrock_grad};
    use approx::assert_relative_eq;

    #[test]
    fn test_gd_converges_sphere() {
        let x0 = DVector::from_vec(vec![5.0, -3.0, 2.0]);
        let config = GradientDescentConfig {
            learning_rate: 0.1,
            max_iter: 5000,
            ..Default::default()
        };
        let result = gradient_descent(&sphere, &sphere_grad, &x0, &config);
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-4);
        for i in 0..3 {
            assert_relative_eq!(result.x[i], 0.0, epsilon = 1e-2);
        }
    }

    #[test]
    fn test_gd_decreases_objective() {
        let x0 = DVector::from_vec(vec![10.0, 10.0]);
        let config = GradientDescentConfig {
            learning_rate: 0.1,
            max_iter: 100,
            ..Default::default()
        };
        let result = gradient_descent(&sphere, &sphere_grad, &x0, &config);
        assert!(result.f_x < sphere(&x0));
    }

    #[test]
    fn test_momentum_converges() {
        let x0 = DVector::from_vec(vec![5.0, -5.0]);
        let result = momentum_gradient_descent(
            &sphere, &sphere_grad, &x0, 0.1, 0.9, 5000, 1e-8,
        );
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_nesterov_converges() {
        let x0 = DVector::from_vec(vec![3.0, 4.0]);
        let result = nesterov_gradient_descent(
            &sphere, &sphere_grad, &x0, 0.1, 0.9, 5000, 1e-8,
        );
        assert!(result.converged);
        assert_relative_eq!(result.f_x, 0.0, epsilon = 1e-3);
    }

    #[test]
    fn test_gd_rosenbrock() {
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let config = GradientDescentConfig {
            learning_rate: 0.001,
            max_iter: 100000,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = gradient_descent(&rosenbrock, &rosenbrock_grad, &x0, &config);
        assert!(result.f_x < 0.1, "Should get close to Rosenbrock minimum");
    }

    #[test]
    fn test_sgd_converges() {
        // Simple SGD on sphere: each "sample" gradient is just 2*x_i
        let f = |x: &DVector<f64>| sphere(x);
        let grad_fn = |x: &DVector<f64>, _idx: usize| sphere_grad(x);

        let x0 = DVector::from_vec(vec![5.0, 3.0]);
        let config = GradientDescentConfig {
            learning_rate: 0.01,
            max_iter: 5000,
            tolerance: 1e-6,
            ..Default::default()
        };
        let result = stochastic_gradient_descent(&f, &grad_fn, &x0, &config, 10);
        assert!(result.f_x < sphere(&x0));
    }

    #[test]
    fn test_mini_batch_converges() {
        let f = |x: &DVector<f64>| sphere(x);
        let grad_fn = |x: &DVector<f64>, _batch: &[usize]| sphere_grad(x);

        let x0 = DVector::from_vec(vec![4.0, -2.0]);
        let config = GradientDescentConfig {
            learning_rate: 0.1,
            max_iter: 5000,
            tolerance: 1e-8,
            ..Default::default()
        };
        let result = mini_batch_gradient_descent(&f, &grad_fn, &x0, &config, 10, 5);
        assert!(result.f_x < sphere(&x0));
    }
}
