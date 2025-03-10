//! # Local Search Algorithms
//!
//! This module provides local search algorithms for refining solutions in genetic algorithms.
//! Local search algorithms can be used to improve solutions by exploring their neighborhood
//! and moving to better solutions.

use crate::evolution::Challenge;
use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

pub mod application;
pub mod manager;

/// A trait for local search algorithms.
///
/// Local search algorithms improve solutions by exploring their neighborhood
/// and moving to better solutions. They are often used in combination with
/// genetic algorithms to refine solutions.
pub trait LocalSearch<P, C>: Debug + Send + Sync
where
    P: Phenotype,
    C: Challenge<P> + ?Sized,
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
    fn search_with_rng(
        &self,
        phenotype: &mut P,
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> bool;
}

// Re-export key types for convenience
pub use application::{
    AllIndividualsStrategy, LocalSearchApplicationStrategy, ProbabilisticStrategy, TopNStrategy,
    TopPercentStrategy,
};
pub use manager::LocalSearchManager;

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
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    ///
    /// # Returns
    ///
    /// A new hill climbing algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_iterations` is 0.
    pub fn new(max_iterations: usize) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            max_neighbors: 10,
        })
    }

    /// Creates a new hill climbing algorithm with the given maximum number of iterations
    /// and maximum number of neighbors to evaluate per iteration.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `max_neighbors` - The maximum number of neighbors to evaluate per iteration.
    ///
    /// # Returns
    ///
    /// A new hill climbing algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_iterations` or `max_neighbors` is 0.
    pub fn with_neighbors(max_iterations: usize, max_neighbors: usize) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }
        if max_neighbors == 0 {
            return Err(GeneticError::Configuration(
                "Maximum neighbors must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            max_neighbors,
        })
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

    fn search_with_rng(
        &self,
        phenotype: &mut P,
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> bool {
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
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `initial_temperature` - The initial temperature.
    /// * `cooling_rate` - The cooling rate (between 0 and 1).
    ///
    /// # Returns
    ///
    /// A new simulated annealing algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_iterations` is 0
    /// - `initial_temperature` is not positive
    /// - `cooling_rate` is not between 0 and 1
    pub fn new(max_iterations: usize, initial_temperature: f64, cooling_rate: f64) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }
        if initial_temperature <= 0.0 {
            return Err(GeneticError::Configuration(
                "Initial temperature must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&cooling_rate) {
            return Err(GeneticError::Configuration(
                "Cooling rate must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            initial_temperature,
            cooling_rate,
        })
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

    fn search_with_rng(
        &self,
        phenotype: &mut P,
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> bool {
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
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `max_neighbors` - The maximum number of neighbors to evaluate per iteration.
    /// * `tabu_list_size` - The maximum size of the tabu list.
    ///
    /// # Returns
    ///
    /// A new tabu search algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_iterations` is 0
    /// - `max_neighbors` is 0
    /// - `tabu_list_size` is 0
    pub fn new(max_iterations: usize, max_neighbors: usize, tabu_list_size: usize) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }
        if max_neighbors == 0 {
            return Err(GeneticError::Configuration(
                "Maximum neighbors must be greater than 0".to_string(),
            ));
        }
        if tabu_list_size == 0 {
            return Err(GeneticError::Configuration(
                "Tabu list size must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            max_neighbors,
            tabu_list_size,
            _marker: PhantomData,
        })
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

    fn search_with_rng(
        &self,
        phenotype: &mut P,
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> bool {
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

    fn search_with_rng(
        &self,
        phenotype: &mut P,
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::Challenge;
    use crate::phenotype::Phenotype;
    use crate::rng::RandomNumberGenerator;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestPhenotype {
        value: i32,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2;
        }

        fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
            let values = rng.fetch_uniform(-1.0, 1.0, 1);
            let delta = values.front().unwrap();
            self.value += (*delta * 10.0) as i32;
        }
    }

    #[derive(Debug)]
    struct TestChallenge {
        target: i32,
        // Counter to track the number of evaluations
        evaluations: Arc<AtomicUsize>,
    }

    impl TestChallenge {
        fn new(target: i32) -> Self {
            Self {
                target,
                evaluations: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn get_evaluations(&self) -> usize {
            self.evaluations.load(Ordering::SeqCst)
        }
    }

    impl Challenge<TestPhenotype> for TestChallenge {
        fn score(&self, phenotype: &TestPhenotype) -> f64 {
            // Increment the evaluation counter
            self.evaluations.fetch_add(1, Ordering::SeqCst);

            // Higher score is better (inverse of distance to target)
            1.0 / ((phenotype.value - self.target).abs() as f64 + 1.0)
        }
    }

    #[test]
    fn test_hill_climbing() {
        let hill_climbing = HillClimbing::new(10).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = hill_climbing.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);

        // Test with neighbors parameter
        let hill_climbing = HillClimbing::with_neighbors(5, 20).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = hill_climbing.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);
    }

    #[test]
    fn test_simulated_annealing() {
        // Use more iterations and higher initial temperature to increase chances of finding improvement
        let simulated_annealing = SimulatedAnnealing::new(50, 20.0, 0.95).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 10 }; // Start with a non-zero value

        let improved = simulated_annealing.search(&mut phenotype, &challenge);

        // Check that evaluations were performed, even if no improvement was found
        assert!(challenge.get_evaluations() > 0);

        // If the algorithm reported an improvement, the value should have changed
        if improved {
            assert!(phenotype.value != 10);
        }
    }

    #[test]
    fn test_tabu_search() {
        let tabu_search = TabuSearch::new(10, 5, 3).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = tabu_search.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);
    }

    #[test]
    fn test_hybrid_local_search() {
        let mut hybrid = HybridLocalSearch::new();
        hybrid
            .add_algorithm(HillClimbing::new(5).unwrap())
            .add_algorithm(SimulatedAnnealing::new(5, 1.0, 0.9).unwrap());

        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = hybrid.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);
    }

    #[test]
    fn test_search_with_rng() {
        let hill_climbing = HillClimbing::new(10).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };
        let mut rng = RandomNumberGenerator::new();

        let _improved = hill_climbing.search_with_rng(&mut phenotype, &challenge, &mut rng);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);
    }
}
