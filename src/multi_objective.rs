//! Multi-objective optimization: Pareto fronts, NSGA-II basics.

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use rand::Rng;

/// A solution in multi-objective space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Decision variables.
    pub x: DVector<f64>,
    /// Objective function values.
    pub objectives: Vec<f64>,
}

/// Check if solution `a` dominates solution `b` (Pareto dominance).
/// a dominates b if a is <= b in all objectives and strictly < in at least one.
pub fn dominates(a: &Solution, b: &Solution) -> bool {
    let mut at_least_one_better = false;
    for (oa, ob) in a.objectives.iter().zip(&b.objectives) {
        if oa > ob {
            return false;
        }
        if oa < ob {
            at_least_one_better = true;
        }
    }
    at_least_one_better
}

/// Extract the Pareto front (non-dominated set) from a population.
pub fn pareto_front(solutions: &[Solution]) -> Vec<usize> {
    let n = solutions.len();
    let mut dominated = vec![false; n];

    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[j] {
                continue;
            }
            if dominates(&solutions[j], &solutions[i]) {
                dominated[i] = true;
                break;
            }
        }
    }

    (0..n).filter(|&i| !dominated[i]).collect()
}

/// Crowding distance for NSGA-II.
fn crowding_distance(front: &[usize], solutions: &[Solution], num_objectives: usize) -> Vec<f64> {
    let n = front.len();
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let mut distances = vec![0.0; n];

    for obj in 0..num_objectives {
        // Sort front by objective value
        let mut sorted: Vec<usize> = front.to_vec();
        sorted.sort_by(|&a, &b| {
            solutions[a].objectives[obj]
                .partial_cmp(&solutions[b].objectives[obj])
                .unwrap()
        });

        let min_val = solutions[sorted[0]].objectives[obj];
        let max_val = solutions[sorted[n - 1]].objectives[obj];
        let range = max_val - min_val;

        // Find index mapping
        let mut idx_map = vec![0usize; n];
        for (rank, &sol_idx) in sorted.iter().enumerate() {
            let pos_in_front = front.iter().position(|&x| x == sol_idx).unwrap();
            idx_map[pos_in_front] = rank;
        }

        distances[0] = f64::INFINITY;
        distances[n - 1] = f64::INFINITY;

        if range > 0.0 {
            for i in 1..n - 1 {
                let prev_idx = sorted[i - 1];
                let next_idx = sorted[i + 1];
                distances[idx_map[i]] +=
                    (solutions[next_idx].objectives[obj] - solutions[prev_idx].objectives[obj])
                        / range;
            }
        }
    }

    distances
}

/// Fast non-dominated sorting. Returns fronts as vectors of indices.
pub fn fast_non_dominated_sort(solutions: &[Solution]) -> Vec<Vec<usize>> {
    let n = solutions.len();
    let mut domination_count = vec![0usize; n];
    let mut dominated_set: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(&solutions[i], &solutions[j]) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(&solutions[j], &solutions[i]) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
        if domination_count[i] == 0 {
            current_front.push(i);
        }
    }

    fronts.push(current_front.clone());

    let mut front_idx = 0;
    while !fronts[front_idx].is_empty() {
        let mut next_front = Vec::new();
        for &i in &fronts[front_idx] {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front);
        front_idx += 1;
    }

    fronts
}

/// Configuration for NSGA-II.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NSGA2Config {
    /// Population size.
    pub pop_size: usize,
    /// Number of generations.
    pub num_generations: usize,
    /// Crossover probability.
    pub crossover_prob: f64,
    /// Mutation probability.
    pub mutation_prob: f64,
    /// Mutation step size.
    pub mutation_step: f64,
    /// Lower bounds.
    pub lower_bounds: Vec<f64>,
    /// Upper bounds.
    pub upper_bounds: Vec<f64>,
}

impl Default for NSGA2Config {
    fn default() -> Self {
        Self {
            pop_size: 100,
            num_generations: 200,
            crossover_prob: 0.9,
            mutation_prob: 0.1,
            mutation_step: 0.1,
            lower_bounds: vec![-10.0],
            upper_bounds: vec![10.0],
        }
    }
}

/// Result of NSGA-II optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NSGA2Result {
    /// Final Pareto front solutions.
    pub pareto_front: Vec<Solution>,
    /// All fronts.
    pub fronts: Vec<Vec<usize>>,
    /// Final population.
    pub population: Vec<Solution>,
}

/// NSGA-II multi-objective optimization.
pub fn nsga2<F>(
    objectives: &[F],
    dim: usize,
    config: &NSGA2Config,
) -> NSGA2Result
where
    F: Fn(&DVector<f64>) -> f64,
{
    let mut rng = rand::thread_rng();
    let num_obj = objectives.len();

    let lb = if config.lower_bounds.len() == 1 {
        vec![config.lower_bounds[0]; dim]
    } else {
        config.lower_bounds.clone()
    };
    let ub = if config.upper_bounds.len() == 1 {
        vec![config.upper_bounds[0]; dim]
    } else {
        config.upper_bounds.clone()
    };

    // Initialize population
    let mut population: Vec<Solution> = (0..config.pop_size)
        .map(|_| {
            let x = DVector::from_fn(dim, |i, _| rng.gen_range(lb[i]..ub[i]));
            let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&x)).collect();
            Solution { x, objectives: obj_vals }
        })
        .collect();

    for _gen in 0..config.num_generations {
        // Create offspring
        let mut offspring = Vec::new();
        while offspring.len() < config.pop_size {
            let p1 = rng.gen_range(0..population.len());
            let p2 = rng.gen_range(0..population.len());

            // Simulated binary crossover
            let mut child_x = if rng.gen::<f64>() < config.crossover_prob {
                let mut child = DVector::zeros(dim);
                for i in 0..dim {
                    if rng.gen::<f64>() < 0.5 {
                        child[i] = population[p1].x[i];
                    } else {
                        child[i] = population[p2].x[i];
                    }
                }
                child
            } else {
                population[p1].x.clone()
            };

            // Mutation
            for i in 0..dim {
                if rng.gen::<f64>() < config.mutation_prob {
                    child_x[i] += rng.gen_range(-1.0..1.0) * config.mutation_step;
                    child_x[i] = child_x[i].clamp(lb[i], ub[i]);
                }
            }

            let obj_vals: Vec<f64> = objectives.iter().map(|f| f(&child_x)).collect();
            offspring.push(Solution { x: child_x, objectives: obj_vals });
        }

        // Combine parent + offspring
        let mut combined = population.clone();
        combined.extend(offspring);

        // Non-dominated sorting
        let fronts = fast_non_dominated_sort(&combined);

        // Select new population using crowding distance
        let mut new_pop = Vec::new();
        let mut selected = 0;

        for front in &fronts {
            if selected + front.len() <= config.pop_size {
                for &idx in front {
                    new_pop.push(combined[idx].clone());
                }
                selected += front.len();
            } else {
                // Need to select some from this front using crowding distance
                let remaining = config.pop_size - selected;
                let distances = crowding_distance(front, &combined, num_obj);

                let mut indexed: Vec<(usize, f64)> = front
                    .iter()
                    .enumerate()
                    .map(|(i, &sol_idx)| (sol_idx, distances[i]))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                for (sol_idx, _) in indexed.iter().take(remaining) {
                    new_pop.push(combined[*sol_idx].clone());
                }
                break;
            }
        }

        population = new_pop;
    }

    let fronts = fast_non_dominated_sort(&population);
    let pareto_indices = &fronts[0];
    let pareto_solutions: Vec<Solution> = pareto_indices
        .iter()
        .map(|&i| population[i].clone())
        .collect();

    NSGA2Result {
        pareto_front: pareto_solutions,
        fronts,
        population,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_dominance() {
        let a = Solution {
            x: DVector::from_vec(vec![1.0]),
            objectives: vec![1.0, 2.0],
        };
        let b = Solution {
            x: DVector::from_vec(vec![2.0]),
            objectives: vec![2.0, 3.0],
        };
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_non_dominated() {
        let a = Solution {
            x: DVector::from_vec(vec![1.0]),
            objectives: vec![1.0, 3.0],
        };
        let b = Solution {
            x: DVector::from_vec(vec![2.0]),
            objectives: vec![3.0, 1.0],
        };
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn test_pareto_front_extraction() {
        let solutions = vec![
            Solution { x: DVector::from_vec(vec![0.0]), objectives: vec![1.0, 4.0] },
            Solution { x: DVector::from_vec(vec![1.0]), objectives: vec![2.0, 2.0] },
            Solution { x: DVector::from_vec(vec![2.0]), objectives: vec![4.0, 1.0] },
            Solution { x: DVector::from_vec(vec![3.0]), objectives: vec![3.0, 3.0] }, // dominated
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front.len(), 3);
        assert!(!front.contains(&3), "Dominated solution should not be in Pareto front");
    }

    #[test]
    fn test_fast_non_dominated_sort() {
        let solutions = vec![
            Solution { x: DVector::from_vec(vec![0.0]), objectives: vec![1.0, 4.0] },
            Solution { x: DVector::from_vec(vec![1.0]), objectives: vec![2.0, 2.0] },
            Solution { x: DVector::from_vec(vec![2.0]), objectives: vec![4.0, 1.0] },
            Solution { x: DVector::from_vec(vec![3.0]), objectives: vec![3.0, 3.0] },
        ];
        let fronts = fast_non_dominated_sort(&solutions);
        assert!(fronts[0].len() >= 3);
    }

    #[test]
    fn test_nsga2_runs() {
        // Two conflicting objectives: minimize x1 and minimize -x1 (i.e., maximize x1)
        let obj1 = |x: &DVector<f64>| x[0];
        let obj2 = |x: &DVector<f64>| -x[0];
        let _objectives: Vec<Box<dyn Fn(&DVector<f64>) -> f64>> = vec![
            Box::new(obj1),
            Box::new(obj2),
        ];

        // NSGA2 expects &[F] where F: Fn — use a Vec of closures with same type via wrapper
        // Instead, test with two simple objectives
        fn run_nsga() -> NSGA2Result {
            let obj1 = |x: &DVector<f64>| x[0]; // minimize x1
            let obj2 = |x: &DVector<f64>| (x[0] - 1.0).powi(2); // minimize (x1-1)^2
            let objectives = [obj1, obj2];
            nsga2(&objectives, 1, &NSGA2Config {
                pop_size: 30,
                num_generations: 50,
                lower_bounds: vec![-5.0],
                upper_bounds: vec![5.0],
                ..Default::default()
            })
        }
        let result = run_nsga();
        assert!(!result.pareto_front.is_empty());
        // Pareto front should have solutions with different tradeoffs
        let obj_vals: Vec<(f64, f64)> = result.pareto_front.iter()
            .map(|s| (s.objectives[0], s.objectives[1]))
            .collect();
        // Verify no solution in front dominates another
        for i in 0..obj_vals.len() {
            for j in 0..obj_vals.len() {
                if i != j {
                    let si = &result.pareto_front[i];
                    let sj = &result.pareto_front[j];
                    assert!(!dominates(si, sj), "Pareto front should have no dominated solutions");
                }
            }
        }
    }

    #[test]
    fn test_pareto_front_single_solution() {
        let solutions = vec![
            Solution { x: DVector::from_vec(vec![0.0]), objectives: vec![0.0, 0.0] },
        ];
        let front = pareto_front(&solutions);
        assert_eq!(front, vec![0]);
    }
}
