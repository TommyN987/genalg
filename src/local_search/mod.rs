//! # Local Search Algorithms
//!
//! This module provides local search algorithms for refining solutions in genetic algorithms.
//! Local search algorithms can be used to improve solutions by exploring their neighborhood
//! and moving to better solutions.

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use std::fmt::Debug;

pub mod application_stategy;
pub mod hill_climbing;
pub mod hybrid;
pub mod manager;
pub mod simulated_annealing;
pub mod tabu_search;

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
}

pub use application_stategy::{
    AllIndividualsStrategy, LocalSearchApplicationStrategy, ProbabilisticStrategy, TopNStrategy,
    TopPercentStrategy,
};
pub use hill_climbing::HillClimbing;
pub use hybrid::HybridLocalSearch;
pub use manager::LocalSearchManager;
pub use simulated_annealing::SimulatedAnnealing;
pub use tabu_search::TabuSearch;

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
        let hill_climbing = HillClimbing::new(10, 10).unwrap();
        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = hill_climbing.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);

        let hill_climbing = HillClimbing::new(5, 20).unwrap();
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
            .add_algorithm(HillClimbing::new(5, 10).unwrap())
            .add_algorithm(SimulatedAnnealing::new(5, 1.0, 0.9).unwrap());

        let challenge = TestChallenge::new(50);
        let mut phenotype = TestPhenotype { value: 0 };

        let _improved = hybrid.search(&mut phenotype, &challenge);

        // The phenotype should be closer to the target
        assert!(phenotype.value != 0);
        assert!(challenge.get_evaluations() > 0);
    }
}
