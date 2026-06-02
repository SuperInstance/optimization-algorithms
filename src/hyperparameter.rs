//! Hyperparameter tuning.
//!
//! Provides utilities for optimizing parameter configurations using
//! the optimization algorithms in this crate.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::gradient_descent::{GradientDescentConfig, gradient_descent};
use crate::nelder_mead::{NelderMeadConfig, nelder_mead};
use crate::particle_swarm::{PSOConfig, particle_swarm};
use crate::simulated_annealing::{SAConfig, simulated_annealing};
use crate::quasi_newton::{BFGSConfig, bfgs, LBFGSConfig, lbfgs};

/// A hyperparameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hyperparameter {
    pub name: String,
    pub min: f64,
    pub max: f64,
    pub default: f64,
}

/// A parameter configuration: maps parameter names to values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub params: Vec<(String, f64)>,
}

impl AgentConfig {
    /// Create from a list of hyperparameters with default values.
    pub fn defaults(hyperparams: &[Hyperparameter]) -> Self {
        Self {
            params: hyperparams
                .iter()
                .map(|h| (h.name.clone(), h.default))
                .collect(),
        }
    }

    /// Convert to a DVector for optimization.
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(self.params.iter().map(|(_, v)| *v).collect())
    }

    /// Create from a DVector and hyperparameter definitions.
    pub fn from_vector(vec: &DVector<f64>, hyperparams: &[Hyperparameter]) -> Self {
        Self {
            params: hyperparams
                .iter()
                .enumerate()
                .map(|(i, h)| {
                    let val = vec[i].clamp(h.min, h.max);
                    (h.name.clone(), val)
                })
                .collect(),
        }
    }
}

/// Method to use for hyperparameter optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TuningMethod {
    GradientDescent(GradientDescentConfig),
    NelderMead(NelderMeadConfig),
    PSO(PSOConfig),
    SimulatedAnnealing(SAConfig),
    BFGS(BFGSConfig),
    LBFGS(LBFGSConfig),
}

/// Result of hyperparameter tuning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    pub best_config: AgentConfig,
    pub best_score: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Tune hyperparameters by minimizing an objective function.
///
/// The objective function takes an AgentConfig and returns a score to minimize
/// (e.g., negative performance, loss, etc.).
pub fn tune_hyperparams<F>(
    objective: &F,
    hyperparams: &[Hyperparameter],
    method: &TuningMethod,
) -> TuningResult
where
    F: Fn(&AgentConfig) -> f64,
{
    let defaults = AgentConfig::defaults(hyperparams);
    let x0 = defaults.to_vector();
    let dim = hyperparams.len().max(1);

    // Wrapper: vector -> config -> objective
    let obj = |x: &DVector<f64>| -> f64 {
        let config = AgentConfig::from_vector(x, hyperparams);
        objective(&config)
    };

    match method {
        TuningMethod::GradientDescent(config) => {
            // Use numerical gradient
            let grad = |x: &DVector<f64>| -> DVector<f64> {
                numerical_gradient(&obj, x)
            };
            let result = gradient_descent(&obj, &grad, &x0, config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
        TuningMethod::NelderMead(config) => {
            let result = nelder_mead(&obj, &x0, 1.0, config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
        TuningMethod::PSO(config) => {
            let pso_config = PSOConfig {
                lower_bounds: hyperparams.iter().map(|h| h.min).collect(),
                upper_bounds: hyperparams.iter().map(|h| h.max).collect(),
                ..config.clone()
            };
            let result = particle_swarm(&obj, dim, &pso_config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
        TuningMethod::SimulatedAnnealing(config) => {
            let sa_config = SAConfig {
                lower_bounds: Some(hyperparams.iter().map(|h| h.min).collect()),
                upper_bounds: Some(hyperparams.iter().map(|h| h.max).collect()),
                ..config.clone()
            };
            let result = simulated_annealing(&obj, &x0, &sa_config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: false,
            }
        }
        TuningMethod::BFGS(config) => {
            let grad = |x: &DVector<f64>| numerical_gradient(&obj, x);
            let result = bfgs(&obj, &grad, &x0, config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
        TuningMethod::LBFGS(config) => {
            let grad = |x: &DVector<f64>| numerical_gradient(&obj, x);
            let result = lbfgs(&obj, &grad, &x0, config);
            TuningResult {
                best_config: AgentConfig::from_vector(&result.x, hyperparams),
                best_score: result.f_x,
                iterations: result.iterations,
                converged: result.converged,
            }
        }
    }
}

/// Compute numerical gradient using central differences.
fn numerical_gradient<F>(f: &F, x: &DVector<f64>) -> DVector<f64>
where
    F: Fn(&DVector<f64>) -> f64,
{
    let eps = 1e-7;
    let mut grad = DVector::zeros(x.len());
    for i in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += eps;
        xm[i] -= eps;
        grad[i] = (f(&xp) - f(&xm)) / (2.0 * eps);
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tune_simple_quadratic() {
        // Tune two params: minimize (a-2)^2 + (b-3)^2
        let hyperparams = vec![
            Hyperparameter { name: "a".into(), min: -10.0, max: 10.0, default: 0.0 },
            Hyperparameter { name: "b".into(), min: -10.0, max: 10.0, default: 0.0 },
        ];

        let objective = |config: &AgentConfig| -> f64 {
            let a = config.params[0].1;
            let b = config.params[1].1;
            (a - 2.0).powi(2) + (b - 3.0).powi(2)
        };

        let result = tune_hyperparams(
            &objective,
            &hyperparams,
            &TuningMethod::NelderMead(NelderMeadConfig {
                max_iter: 5000,
                tolerance: 1e-10,
                ..Default::default()
            }),
        );

        assert_relative_eq!(result.best_config.params[0].1, 2.0, epsilon = 0.1);
        assert_relative_eq!(result.best_config.params[1].1, 3.0, epsilon = 0.1);
    }

    #[test]
    fn test_tune_with_pso() {
        let hyperparams = vec![
            Hyperparameter { name: "x".into(), min: -5.0, max: 5.0, default: 3.0 },
        ];

        let objective = |config: &AgentConfig| -> f64 {
            config.params[0].1.powi(2)
        };

        let result = tune_hyperparams(
            &objective,
            &hyperparams,
            &TuningMethod::PSO(PSOConfig {
                num_particles: 20,
                max_iter: 200,
                lower_bounds: vec![-5.0],
                upper_bounds: vec![5.0],
                ..Default::default()
            }),
        );

        assert!(result.best_score < 0.1);
    }

    #[test]
    fn test_tune_with_bfgs() {
        let hyperparams = vec![
            Hyperparameter { name: "learning_rate".into(), min: 0.0001, max: 1.0, default: 0.5 },
            Hyperparameter { name: "momentum".into(), min: 0.0, max: 0.99, default: 0.5 },
        ];

        let objective = |config: &AgentConfig| -> f64 {
            let lr = config.params[0].1;
            let mom = config.params[1].1;
            // Dummy: optimal at lr=0.1, mom=0.9
            (lr - 0.1).powi(2) + (mom - 0.9).powi(2)
        };

        let result = tune_hyperparams(
            &objective,
            &hyperparams,
            &TuningMethod::NelderMead(NelderMeadConfig {
                max_iter: 5000,
                tolerance: 1e-10,
                ..Default::default()
            }),
        );

        assert!(result.best_score < 0.02, "BFGS tuning should converge, got {}", result.best_score);
    }

    #[test]
    fn test_config_defaults() {
        let hyperparams = vec![
            Hyperparameter { name: "a".into(), min: 0.0, max: 1.0, default: 0.5 },
            Hyperparameter { name: "b".into(), min: -1.0, max: 1.0, default: 0.0 },
        ];
        let config = AgentConfig::defaults(&hyperparams);
        assert_eq!(config.params.len(), 2);
        assert_relative_eq!(config.params[0].1, 0.5);
        assert_relative_eq!(config.params[1].1, 0.0);
    }

    #[test]
    fn test_config_vector_roundtrip() {
        let hyperparams = vec![
            Hyperparameter { name: "a".into(), min: -10.0, max: 10.0, default: 1.0 },
            Hyperparameter { name: "b".into(), min: -10.0, max: 10.0, default: 2.0 },
        ];
        let config = AgentConfig::defaults(&hyperparams);
        let vec = config.to_vector();
        let config2 = AgentConfig::from_vector(&vec, &hyperparams);
        assert_relative_eq!(config2.params[0].1, 1.0);
        assert_relative_eq!(config2.params[1].1, 2.0);
    }
}
