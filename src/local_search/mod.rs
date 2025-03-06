//! # Local Search Module
//!
//! This module provides tools for integrating local search techniques with genetic algorithms.
//! Local search can significantly improve the performance of genetic algorithms by refining
//! solutions through systematic exploration of the neighborhood around promising individuals.
//!
//! ## Key Components
//!
//! - `LocalSearch` trait: Defines the interface for local search algorithms
//! - `HillClimbing`: A simple hill climbing algorithm
//! - `SimulatedAnnealing`: A simulated annealing algorithm
//! - `TabuSearch`: A tabu search algorithm
//!
//! ## Example
//!
//! ```rust
//! use genalg::local_search::{LocalSearch, HillClimbing};
//! use genalg::evolution::Challenge;
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//!
//! #[derive(Clone, Debug)]
//! struct MySolution {
//!     value: f64,
//! }
//!
//! impl Phenotype for MySolution {
//!     fn crossover(&mut self, other: &Self) {
//!         self.value = (self.value + other.value) / 2.0;
//!     }
//!
//!     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
//!         let values = rng.fetch_uniform(-0.1, 0.1, 1);
//!         let delta = values.front().unwrap();
//!         self.value += *delta as f64;
//!     }
//! }
//!
//! #[derive(Debug)]
//! struct MyChallenge {
//!     target: f64,
//! }
//!
//! impl Challenge<MySolution> for MyChallenge {
//!     fn score(&self, phenotype: &MySolution) -> f64 {
//!         // Higher score is better (inverse of distance to target)
//!         1.0 / (phenotype.value - self.target).abs().max(0.001)
//!     }
//! }
//!
//! // Create a hill climbing algorithm
//! let hill_climbing = HillClimbing::new(10); // 10 iterations
//!
//! // Apply local search to a solution
//! let challenge = MyChallenge { target: 42.0 };
//! let mut solution = MySolution { value: 40.0 };
//! let improved = hill_climbing.search(&mut solution, &challenge);
//! ```

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

/// Trait for local search algorithms.
///
/// Local search algorithms improve solutions by exploring the neighborhood
/// around a current solution and moving to better solutions.
pub trait LocalSearch<P, C>: Debug + Send + Sync
where
    P: Phenotype,
    C: Challenge<P>,
{
    /// Applies the local search algorithm to the given phenotype.
    ///
    /// Returns `true` if the phenotype was improved, `false` otherwise.
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool;

    /// Applies the local search algorithm to the given phenotype using a random number generator.
    ///
    /// This method is useful for local search algorithms that require randomness.
    ///
    /// Returns `true` if the phenotype was improved, `false` otherwise.
    fn search_with_rng(&self, phenotype: &mut P, challenge: &C, rng: &mut RandomNumberGenerator) -> bool;
}

/// A simple hill climbing algorithm.
///
/// Hill climbing is a local search algorithm that iteratively moves to better
/// neighboring solutions until no better solution can be found.
#[derive(Debug, Clone)]
pub struct HillClimbing {
    /// The maximum number of iterations to perform
    max_iterations: usize,
    /// The maximum number of neighbors to evaluate per iteration
    max_neighbors: usize,
}

impl HillClimbing {
    /// Creates a new hill climbing algorithm with the given maximum number of iterations.
    ///
    /// The maximum number of neighbors to evaluate per iteration is set to 10.
    pub fn new(max_iterations: usize) -> Self {
        Self {
            max_iterations,
            max_neighbors: 10,
        }
    }

    /// Creates a new hill climbing algorithm with the given maximum number of iterations
    /// and maximum number of neighbors to evaluate per iteration.
    pub fn with_neighbors(max_iterations: usize, max_neighbors: usize) -> Self {
        Self {
            max_iterations,
            max_neighbors,
        }
    }
}

impl<P, C> LocalSearch<P, C> for HillClimbing
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut rng = RandomNumberGenerator::new();
        self.search_with_rng(phenotype, challenge, &mut rng)
    }

    fn search_with_rng(&self, phenotype: &mut P, challenge: &C, rng: &mut RandomNumberGenerator) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_score = initial_score;
        let mut improved = false;

        for _ in 0..self.max_iterations {
            let mut best_neighbor = phenotype.clone();
            let mut best_neighbor_score = current_score;
            let mut found_better = false;

            // Evaluate neighbors
            for _ in 0..self.max_neighbors {
                let mut neighbor = phenotype.clone();
                neighbor.mutate(rng);
                let neighbor_score = challenge.score(&neighbor);

                if neighbor_score > best_neighbor_score {
                    best_neighbor = neighbor;
                    best_neighbor_score = neighbor_score;
                    found_better = true;
                }
            }

            // If a better neighbor was found, move to it
            if found_better {
                *phenotype = best_neighbor;
                current_score = best_neighbor_score;
                improved = true;
            } else {
                // If no better neighbor was found, stop
                break;
            }
        }

        improved
    }
}

/// A simulated annealing algorithm.
///
/// Simulated annealing is a probabilistic local search algorithm that allows
/// moves to worse solutions with a probability that decreases over time.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing {
    /// The maximum number of iterations to perform
    max_iterations: usize,
    /// The initial temperature
    initial_temperature: f64,
    /// The cooling rate
    cooling_rate: f64,
}

impl SimulatedAnnealing {
    /// Creates a new simulated annealing algorithm with the given parameters.
    pub fn new(max_iterations: usize, initial_temperature: f64, cooling_rate: f64) -> Self {
        Self {
            max_iterations,
            initial_temperature,
            cooling_rate,
        }
    }
}

impl<P, C> LocalSearch<P, C> for SimulatedAnnealing
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut rng = RandomNumberGenerator::new();
        self.search_with_rng(phenotype, challenge, &mut rng)
    }

    fn search_with_rng(&self, phenotype: &mut P, challenge: &C, rng: &mut RandomNumberGenerator) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_score = initial_score;
        let mut current_solution = phenotype.clone();
        let mut temperature = self.initial_temperature;
        let mut improved = false;

        for _ in 0..self.max_iterations {
            // Generate a neighbor
            let mut neighbor = current_solution.clone();
            neighbor.mutate(rng);
            let neighbor_score = challenge.score(&neighbor);

            // Decide whether to move to the neighbor
            let accept = if neighbor_score > current_score {
                // Always accept better solutions
                true
            } else {
                // Accept worse solutions with a probability that decreases with temperature
                let delta = neighbor_score - current_score;
                let probability = (delta / temperature).exp();
                let uniform = rng.fetch_uniform(0.0, 1.0, 1);
                let random = uniform.front().unwrap();
                *random < probability as f32
            };

            if accept {
                current_solution = neighbor;
                current_score = neighbor_score;

                // Update best solution if improved
                if current_score > initial_score {
                    improved = true;
                }
            }

            // Cool down
            temperature *= self.cooling_rate;
        }

        // Update the phenotype with the best solution found
        if improved {
            *phenotype = current_solution;
        }

        improved
    }
}

/// A tabu search algorithm.
///
/// Tabu search is a local search algorithm that maintains a list of recently
/// visited solutions (the tabu list) to avoid cycling and to escape local optima.
#[derive(Debug, Clone)]
pub struct TabuSearch<P>
where
    P: Phenotype + Eq,
{
    /// The maximum number of iterations to perform
    max_iterations: usize,
    /// The maximum number of neighbors to evaluate per iteration
    max_neighbors: usize,
    /// The maximum size of the tabu list
    tabu_list_size: usize,
    /// Phantom data for the phenotype type
    _marker: PhantomData<P>,
}

impl<P> TabuSearch<P>
where
    P: Phenotype + Eq,
{
    /// Creates a new tabu search algorithm with the given parameters.
    pub fn new(max_iterations: usize, max_neighbors: usize, tabu_list_size: usize) -> Self {
        Self {
            max_iterations,
            max_neighbors,
            tabu_list_size,
            _marker: PhantomData,
        }
    }
}

impl<P, C> LocalSearch<P, C> for TabuSearch<P>
where
    P: Phenotype + Eq,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut rng = RandomNumberGenerator::new();
        self.search_with_rng(phenotype, challenge, &mut rng)
    }

    fn search_with_rng(&self, phenotype: &mut P, challenge: &C, rng: &mut RandomNumberGenerator) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_solution = phenotype.clone();
        let mut current_score = initial_score;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut tabu_list = Vec::new();
        let mut improved = false;

        for _ in 0..self.max_iterations {
            let mut best_neighbor = current_solution.clone();
            let mut best_neighbor_score = f64::NEG_INFINITY;
            let mut found_better = false;

            // Evaluate neighbors
            for _ in 0..self.max_neighbors {
                let mut neighbor = current_solution.clone();
                neighbor.mutate(rng);
                let neighbor_score = challenge.score(&neighbor);

                // Check if the neighbor is not in the tabu list
                if !tabu_list.contains(&neighbor) && neighbor_score > best_neighbor_score {
                    best_neighbor = neighbor;
                    best_neighbor_score = neighbor_score;
                    found_better = true;
                }
            }

            // If a better neighbor was found, move to it
            if found_better {
                current_solution = best_neighbor;
                current_score = best_neighbor_score;

                // Update best solution if improved
                if current_score > best_score {
                    best_solution = current_solution.clone();
                    best_score = current_score;
                    improved = true;
                }

                // Add to tabu list
                tabu_list.push(current_solution.clone());
                if tabu_list.len() > self.tabu_list_size {
                    tabu_list.remove(0);
                }
            } else {
                // If no better neighbor was found, stop
                break;
            }
        }

        // Update the phenotype with the best solution found
        if improved {
            *phenotype = best_solution;
        }

        improved
    }
}

/// A hybrid local search algorithm that combines multiple local search algorithms.
///
/// This algorithm applies multiple local search algorithms in sequence.
#[derive(Debug, Clone)]
pub struct HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    /// The local search algorithms to apply
    algorithms: Vec<Arc<dyn LocalSearch<P, C>>>,
}

impl<P, C> HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    /// Creates a new empty hybrid local search algorithm.
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
        }
    }

    /// Adds a local search algorithm to the hybrid.
    pub fn add_algorithm<L>(&mut self, algorithm: L) -> &mut Self
    where
        L: LocalSearch<P, C> + 'static,
    {
        self.algorithms.push(Arc::new(algorithm));
        self
    }
}

impl<P, C> LocalSearch<P, C> for HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut improved = false;

        for algorithm in &self.algorithms {
            if algorithm.search(phenotype, challenge) {
                improved = true;
            }
        }

        improved
    }

    fn search_with_rng(&self, phenotype: &mut P, challenge: &C, rng: &mut RandomNumberGenerator) -> bool {
        let mut improved = false;

        for algorithm in &self.algorithms {
            if algorithm.search_with_rng(phenotype, challenge, rng) {
                improved = true;
            }
        }

        improved
    }
}

impl<P, C> Default for HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    fn default() -> Self {
        Self::new()
    }
} 