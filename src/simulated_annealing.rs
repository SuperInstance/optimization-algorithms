//! Simulated annealing optimization.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Temperature schedule for simulated annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemperatureSchedule {
    /// Exponential cooling: T(k) = T0 * cooling_rate^k
    Exponential { cooling_rate: f64 },
    /// Linear cooling: T(k) = T0 - k * rate
    Linear { rate: f64 },
    /// Logarithmic cooling: T(k) = T0 / (1 + k)
    Logarithmic,
}

/// Configuration for simulated annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAConfig {
    /// Initial temperature.
    pub initial_temp: f64,
    /// Minimum temperature (stopping condition).
    pub min_temp: f64,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Number of iterations at each temperature.
    pub iter_per_temp: usize,
    /// Temperature schedule.
    pub schedule: TemperatureSchedule,
    /// Step size for neighbor generation.
    pub step_size: f64,
    /// Lower bounds for each dimension (optional).
    pub lower_bounds: Option<Vec<f64>>,
    /// Upper bounds for each dimension (optional).
    pub upper_bounds: Option<Vec<f64>>,
}

impl Default for SAConfig {
    fn default() -> Self {
        Self {
            initial_temp: 100.0,
            min_temp: 1e-8,
            max_iter: 100000,
            iter_per_temp: 10,
            schedule: TemperatureSchedule::Exponential { cooling_rate: 0.99 },
            step_size: 1.0,
            lower_bounds: None,
            upper_bounds: None,
        }
    }
}

/// Result of simulated annealing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SAResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub final_temp: f64,
}

/// Compute temperature at iteration k.
fn temperature(schedule: &TemperatureSchedule, t0: f64, k: usize) -> f64 {
    match schedule {
        TemperatureSchedule::Exponential { cooling_rate } => {
            t0 * cooling_rate.powi(k as i32)
        }
        TemperatureSchedule::Linear { rate } => {
            (t0 - k as f64 * rate).max(0.0)
        }
        TemperatureSchedule::Logarithmic => {
            t0 / (1.0 + k as f64)
        }
    }
}

/// Compute acceptance probability for a worse solution.
pub fn acceptance_probability(current_energy: f64, new_energy: f64, temperature: f64) -> f64 {
    if new_energy < current_energy {
        1.0
    } else if temperature > 0.0 {
        (-(new_energy - current_energy) / temperature).exp()
    } else {
        0.0
    }
}

/// Simulated annealing optimization.
pub fn simulated_annealing<F>(
    f: &F,
    x0: &DVector<f64>,
    config: &SAConfig,
) -> SAResult
where
    F: Fn(&DVector<f64>) -> f64,
{
    let mut rng = rand::thread_rng();
    let n = x0.len();

    let mut current_x = x0.clone();
    let mut current_f = f(&current_x);
    let mut best_x = current_x.clone();
    let mut best_f = current_f;
    let mut temp = config.initial_temp;
    let mut iterations = 0;
    let mut k = 0;

    while temp > config.min_temp && iterations < config.max_iter {
        for _ in 0..config.iter_per_temp {
            iterations += 1;

            // Generate neighbor
            let mut neighbor = current_x.clone();
            for i in 0..n {
                let delta: f64 = rng.gen_range(-1.0..1.0) * config.step_size;
                neighbor[i] += delta;
                // Clamp to bounds
                if let Some(ref lb) = config.lower_bounds {
                    neighbor[i] = neighbor[i].max(lb[i]);
                }
                if let Some(ref ub) = config.upper_bounds {
                    neighbor[i] = neighbor[i].min(ub[i]);
                }
            }

            let neighbor_f = f(&neighbor);
            let prob = acceptance_probability(current_f, neighbor_f, temp);

            if rng.gen::<f64>() < prob {
                current_x = neighbor;
                current_f = neighbor_f;

                if current_f < best_f {
                    best_x = current_x.clone();
                    best_f = current_f;
                }
            }
        }

        k += 1;
        temp = temperature(&config.schedule, config.initial_temp, k);
    }

    SAResult {
        x: best_x,
        f_x: best_f,
        iterations,
        final_temp: temp,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::sphere;
    use approx::assert_relative_eq;

    #[test]
    fn test_sa_sphere() {
        let x0 = DVector::from_vec(vec![5.0, -5.0]);
        let config = SAConfig {
            initial_temp: 100.0,
            min_temp: 1e-6,
            max_iter: 100000,
            iter_per_temp: 20,
            step_size: 1.0,
            schedule: TemperatureSchedule::Exponential { cooling_rate: 0.999 },
            ..Default::default()
        };
        let result = simulated_annealing(&sphere, &x0, &config);
        assert!(result.f_x < 1.0, "SA should find near minimum, got {}", result.f_x);
    }

    #[test]
    fn test_sa_acceptance_probability_improvement() {
        let prob = acceptance_probability(10.0, 5.0, 1.0);
        assert_relative_eq!(prob, 1.0);
    }

    #[test]
    fn test_sa_acceptance_probability_worse() {
        let prob = acceptance_probability(5.0, 10.0, 1.0);
        assert!(prob < 1.0);
        assert!(prob > 0.0);
    }

    #[test]
    fn test_sa_acceptance_probability_zero_temp() {
        let prob = acceptance_probability(5.0, 10.0, 0.0);
        assert_relative_eq!(prob, 0.0);
    }

    #[test]
    fn test_sa_acceptance_probability_high_temp() {
        let prob_high = acceptance_probability(5.0, 10.0, 100.0);
        let prob_low = acceptance_probability(5.0, 10.0, 1.0);
        assert!(prob_high > prob_low, "Higher temp should give higher acceptance");
    }

    #[test]
    fn test_sa_with_bounds() {
        let x0 = DVector::from_vec(vec![3.0, 3.0]);
        let config = SAConfig {
            initial_temp: 50.0,
            min_temp: 1e-4,
            max_iter: 50000,
            step_size: 0.5,
            lower_bounds: Some(vec![-5.0, -5.0]),
            upper_bounds: Some(vec![5.0, 5.0]),
            schedule: TemperatureSchedule::Exponential { cooling_rate: 0.999 },
            ..Default::default()
        };
        let result = simulated_annealing(&sphere, &x0, &config);
        assert!(result.f_x < 5.0);
    }

    #[test]
    fn test_sa_temperature_schedules() {
        // Verify schedule functions
        let t = temperature(&TemperatureSchedule::Exponential { cooling_rate: 0.99 }, 100.0, 0);
        assert_relative_eq!(t, 100.0);

        let t = temperature(&TemperatureSchedule::Exponential { cooling_rate: 0.99 }, 100.0, 1);
        assert_relative_eq!(t, 99.0);

        let t = temperature(&TemperatureSchedule::Linear { rate: 1.0 }, 100.0, 50);
        assert_relative_eq!(t, 50.0);

        let t = temperature(&TemperatureSchedule::Logarithmic, 100.0, 9);
        assert_relative_eq!(t, 10.0);
    }
}
