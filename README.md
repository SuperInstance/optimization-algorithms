# optimization-algorithms

Every optimization algorithm you learned in grad school, in Rust.

## What's inside

| Algorithm | Module |
|-----------|--------|
| Gradient descent (SGD, momentum, Adam) | `gradient_descent` |
| Conjugate gradient (Fletcher–Reeves, Polak–Ribière) | `conjugate_gradient` |
| Quasi-Newton (BFGS, L-BFGS) | `quasi_newton` |
| Nelder-Mead simplex | `nelder_mead` |
| Simulated annealing | `simulated_annealing` |
| Particle swarm optimization | `particle_swarm` |
| Line search (backtracking, Wolfe) | `line_search` |
| Multi-objective (weighted sum, NSGA-II) | `multi_objective` |
| Constraint handling (penalty, barrier) | `constraints` |
| Hyperparameter tuning | `hyperparameter` |
| Test functions (Rosenbrock, Rastrigin, Ackley, …) | `test_functions` |

## Quick start

```toml
[dependencies]
optimization-algorithms = "0.1"
```

```rust
use optimization_algorithms::test_functions::{rosenbrock, rosenbrock_gradient};
use optimization_algorithms::quasi_newton::lbfgs;

let result = lbfgs(
    &rosenbrock,
    &rosenbrock_gradient,
    &DVector::from_vec(vec![-1.0, 1.0]),
    &Default::default(),
);
println!("Minimum at {:?}, f = {}", result.x, result.f_x);
```

## License

MIT OR Apache-2.0
