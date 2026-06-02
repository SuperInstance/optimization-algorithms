//! Particle swarm optimization (PSO).

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Configuration for particle swarm optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PSOConfig {
    /// Number of particles.
    pub num_particles: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Cognitive coefficient (personal best pull).
    pub c1: f64,
    /// Social coefficient (global best pull).
    pub c2: f64,
    /// Inertia weight (velocity damping).
    pub w: f64,
    /// Convergence tolerance on best value improvement.
    pub tolerance: f64,
    /// Lower bounds.
    pub lower_bounds: Vec<f64>,
    /// Upper bounds.
    pub upper_bounds: Vec<f64>,
    /// Maximum velocity as fraction of search range.
    pub v_max_fraction: f64,
}

impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            num_particles: 30,
            max_iter: 1000,
            c1: 2.0,
            c2: 2.0,
            w: 0.7,
            tolerance: 1e-10,
            lower_bounds: vec![-10.0],
            upper_bounds: vec![10.0],
            v_max_fraction: 0.5,
        }
    }
}

/// Result of PSO optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PSOResult {
    pub x: DVector<f64>,
    pub f_x: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// Particle in the swarm.
struct Particle {
    position: DVector<f64>,
    velocity: DVector<f64>,
    best_position: DVector<f64>,
    best_value: f64,
}

/// Particle swarm optimization.
pub fn particle_swarm<F>(
    f: &F,
    dim: usize,
    config: &PSOConfig,
) -> PSOResult
where
    F: Fn(&DVector<f64>) -> f64,
{
    let mut rng = rand::thread_rng();
    let n = config.num_particles;

    // Build bounds vectors
    let lb = if config.lower_bounds.len() == 1 {
        DVector::from_element(dim, config.lower_bounds[0])
    } else {
        DVector::from_vec(config.lower_bounds.clone())
    };
    let ub = if config.upper_bounds.len() == 1 {
        DVector::from_element(dim, config.upper_bounds[0])
    } else {
        DVector::from_vec(config.upper_bounds.clone())
    };
    let range = &ub - &lb;
    let v_max = range.scale(config.v_max_fraction);

    // Initialize particles
    let mut particles: Vec<Particle> = Vec::with_capacity(n);
    let mut global_best_pos = DVector::zeros(dim);
    let mut global_best_val = f64::INFINITY;

    for _ in 0..n {
        let pos = DVector::from_fn(dim, |i, _| {
            rng.gen_range(lb[i]..ub[i])
        });
        let val = f(&pos);
        let vel = DVector::from_fn(dim, |i, _| {
            rng.gen_range(-v_max[i]..v_max[i])
        });

        if val < global_best_val {
            global_best_val = val;
            global_best_pos = pos.clone();
        }

        particles.push(Particle {
            position: pos,
            velocity: vel,
            best_position: particles.last().map(|_| global_best_pos.clone()).unwrap_or_default(),
            best_value: val,
        });
        // Fix: set best_position correctly
        particles.last_mut().unwrap().best_position = particles.last().unwrap().position.clone();
        particles.last_mut().unwrap().best_value = val;
    }

    let mut iterations = 0;
    let mut prev_best = f64::INFINITY;

    for i in 0..config.max_iter {
        iterations = i + 1;

        for particle in &mut particles {
            // Update velocity
            let r1: f64 = rng.gen::<f64>();
            let r2: f64 = rng.gen::<f64>();

            let cognitive = particle.best_position.clone() - &particle.position;
            let social = &global_best_pos - &particle.position;

            particle.velocity = particle.velocity.scale(config.w)
                + cognitive.scale(config.c1 * r1)
                + social.scale(config.c2 * r2);

            // Clamp velocity
            for j in 0..dim {
                particle.velocity[j] = particle.velocity[j].clamp(-v_max[j], v_max[j]);
            }

            // Update position
            particle.position += &particle.velocity;

            // Clamp position to bounds
            for j in 0..dim {
                particle.position[j] = particle.position[j].clamp(lb[j], ub[j]);
            }

            // Evaluate
            let val = f(&particle.position);
            if val < particle.best_value {
                particle.best_value = val;
                particle.best_position = particle.position.clone();
            }
            if val < global_best_val {
                global_best_val = val;
                global_best_pos = particle.position.clone();
            }
        }

        // Convergence check
        if (prev_best - global_best_val).abs() < config.tolerance && iterations > 10 {
            return PSOResult {
                x: global_best_pos,
                f_x: global_best_val,
                iterations,
                converged: true,
            };
        }
        prev_best = global_best_val;
    }

    PSOResult {
        x: global_best_pos,
        f_x: global_best_val,
        iterations,
        converged: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_functions::{sphere, rosenbrock, rastrigin};
    

    #[test]
    fn test_pso_sphere() {
        let config = PSOConfig {
            num_particles: 50,
            max_iter: 1000,
            lower_bounds: vec![-10.0],
            upper_bounds: vec![10.0],
            ..Default::default()
        };
        let result = particle_swarm(&sphere, 2, &config);
        assert!(result.f_x < 0.1, "PSO should find sphere minimum, got {}", result.f_x);
    }

    #[test]
    fn test_pso_rosenbrock() {
        let config = PSOConfig {
            num_particles: 50,
            max_iter: 2000,
            lower_bounds: vec![-5.0],
            upper_bounds: vec![5.0],
            ..Default::default()
        };
        let result = particle_swarm(&rosenbrock, 2, &config);
        assert!(result.f_x < 1.0, "PSO should find good Rosenbrock solution, got {}", result.f_x);
    }

    #[test]
    fn test_pso_rastrigin() {
        let config = PSOConfig {
            num_particles: 50,
            max_iter: 1000,
            lower_bounds: vec![-5.12],
            upper_bounds: vec![5.12],
            ..Default::default()
        };
        let result = particle_swarm(&rastrigin, 2, &config);
        // Rastrigin has many local minima; just check we get something reasonable
        assert!(result.f_x < 10.0, "PSO should find decent Rastrigin solution, got {}", result.f_x);
    }

    #[test]
    fn test_pso_high_dimensional() {
        let config = PSOConfig {
            num_particles: 60,
            max_iter: 2000,
            lower_bounds: vec![-10.0],
            upper_bounds: vec![10.0],
            ..Default::default()
        };
        let result = particle_swarm(&sphere, 10, &config);
        assert!(result.f_x < 5.0);
    }

    #[test]
    fn test_pso_respects_bounds() {
        let config = PSOConfig {
            num_particles: 20,
            max_iter: 200,
            lower_bounds: vec![-1.0, -1.0],
            upper_bounds: vec![1.0, 1.0],
            ..Default::default()
        };
        let result = particle_swarm(&sphere, 2, &config);
        for i in 0..2 {
            assert!(result.x[i] >= -1.0 && result.x[i] <= 1.0);
        }
    }
}
