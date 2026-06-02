//! # optimization-algorithms
//!
//! General optimization algorithms for finding minima of objective functions.
//!
//! ## Modules
//! - **line_search**: Golden section, backtracking Armijo, Wolfe conditions
//! - **gradient_descent**: SGD, mini-batch, momentum, Nesterov
//! - **conjugate_gradient**: Fletcher-Reeves, Polak-Ribiere
//! - **quasi_newton**: BFGS, L-BFGS
//! - **nelder_mead**: Derivative-free simplex method
//! - **simulated_annealing**: Probabilistic global optimization
//! - **particle_swarm**: PSO metaheuristic
//! - **multi_objective**: Pareto fronts, NSGA-II basics
//! - **constraints**: Penalty and barrier methods
//! - **hyperparameter**: Agent configuration tuning

pub mod line_search;
pub mod gradient_descent;
pub mod conjugate_gradient;
pub mod quasi_newton;
pub mod nelder_mead;
pub mod simulated_annealing;
pub mod particle_swarm;
pub mod multi_objective;
pub mod constraints;
pub mod hyperparameter;

pub mod test_functions;
