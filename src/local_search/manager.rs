//! # Local Search Manager
//!
//! This module provides a manager for coordinating the application of local search
//! algorithms during the evolutionary process. The manager determines which individuals
//! should undergo local search and applies the appropriate algorithm.

use crate::error::{GeneticError, Result};
use crate::evolution::Challenge;
use crate::local_search::application::LocalSearchApplicationStrategy;
use crate::local_search::LocalSearch;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use std::fmt::Debug;
use std::marker::PhantomData;

/// A manager for coordinating the application of local search during evolution.
///
/// The `LocalSearchManager` determines which individuals should undergo local search
/// based on a configurable application strategy, and applies the specified local search
/// algorithm to those individuals.
#[derive(Debug, Clone)]
pub struct LocalSearchManager<P, L, A, C>
where
    P: Phenotype,
    L: LocalSearch<P, C> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
    C: Challenge<P> + Clone,
{
    /// The local search algorithm to apply.
    algorithm: L,
    /// The strategy for selecting individuals to apply local search to.
    application_strategy: A,
    /// Phantom data for the phenotype and challenge types.
    _marker: PhantomData<(P, C)>,
}

impl<P, L, A, C> LocalSearchManager<P, L, A, C>
where
    P: Phenotype,
    L: LocalSearch<P, C> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
    C: Challenge<P> + Clone,
{
    /// Creates a new local search manager with the specified algorithm and application strategy.
    ///
    /// # Arguments
    ///
    /// * `algorithm` - The local search algorithm to apply.
    /// * `application_strategy` - The strategy for selecting individuals to apply local search to.
    pub fn new(algorithm: L, application_strategy: A) -> Self {
        Self {
            algorithm,
            application_strategy,
            _marker: PhantomData,
        }
    }

    /// Applies local search to selected individuals in the population using a random number generator.
    ///
    /// # Arguments
    ///
    /// * `population` - The population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    /// * `challenge` - The challenge used to evaluate fitness.
    /// * `rng` - The random number generator to use.
    ///
    /// # Returns
    ///
    /// A vector of booleans indicating which individuals were improved by local search.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    /// - The selection process encounters an error (e.g., random number generation fails)
    pub fn apply_with_rng(
        &self,
        population: &mut [P],
        fitness: &[f64],
        challenge: &C,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<bool>> {
        // Validate inputs
        self.validate_inputs(population, fitness)?;

        // Select individuals for local search
        let indices = self.application_strategy.select_for_local_search(
            population,
            fitness,
            Some(rng),
        )?;

        // Initialize the improvements vector
        let mut improvements = vec![false; population.len()];

        // Apply local search to selected individuals
        for idx in indices {
            improvements[idx] = self.algorithm.search_with_rng(
                &mut population[idx],
                challenge,
                rng,
            );
        }

        Ok(improvements)
    }

    /// Applies local search to selected individuals in the population without using a random number generator.
    ///
    /// # Arguments
    ///
    /// * `population` - The population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    /// * `challenge` - The challenge used to evaluate fitness.
    ///
    /// # Returns
    ///
    /// A vector of booleans indicating which individuals were improved by local search.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    /// - The selection process requires randomness but no RNG is provided
    /// - The selection process encounters an error
    pub fn apply_without_rng(
        &self,
        population: &mut [P],
        fitness: &[f64],
        challenge: &C,
    ) -> Result<Vec<bool>> {
        // Validate inputs
        self.validate_inputs(population, fitness)?;

        // Select individuals for local search
        let indices = self.application_strategy.select_for_local_search(
            population,
            fitness,
            None,
        )?;

        // Initialize the improvements vector
        let mut improvements = vec![false; population.len()];

        // Apply local search to selected individuals
        for idx in indices {
            improvements[idx] = self.algorithm.search(
                &mut population[idx],
                challenge,
            );
        }

        Ok(improvements)
    }

    /// Validates the inputs for the apply methods.
    ///
    /// # Arguments
    ///
    /// * `population` - The population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    ///
    /// # Returns
    ///
    /// Ok(()) if the inputs are valid, an error otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    fn validate_inputs(&self, population: &[P], fitness: &[f64]) -> Result<()> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        if population.len() != fitness.len() {
            return Err(GeneticError::Configuration(format!(
                "Population size ({}) does not match fitness vector size ({})",
                population.len(),
                fitness.len()
            )));
        }
        Ok(())
    }

    /// Applies local search to selected individuals in the population.
    ///
    /// # Arguments
    ///
    /// * `population` - The population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    /// * `challenge` - The challenge used to evaluate fitness.
    /// * `rng` - An optional random number generator for strategies and algorithms that use randomness.
    ///
    /// # Returns
    ///
    /// A vector of booleans indicating which individuals were improved by local search.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    /// - The selection process requires randomness but `rng` is `None`
    /// - The selection process encounters an error (e.g., random number generation fails)
    pub fn apply(
        &self,
        population: &mut [P],
        fitness: &[f64],
        challenge: &C,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Vec<bool>> {
        match rng {
            Some(rng) => self.apply_with_rng(population, fitness, challenge, rng),
            None => self.apply_without_rng(population, fitness, challenge),
        }
    }

    /// Returns a reference to the local search algorithm.
    pub fn algorithm(&self) -> &L {
        &self.algorithm
    }

    /// Returns a reference to the application strategy.
    pub fn application_strategy(&self) -> &A {
        &self.application_strategy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_search::application::{AllIndividualsStrategy, TopNStrategy};
    use crate::local_search::HillClimbing;
    use crate::rng::RandomNumberGenerator;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        value: f64,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2.0;
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            self.value += 0.1;
        }
    }

    #[derive(Clone, Debug)]
    struct TestChallenge {
        target: f64,
        // Counter to track the number of evaluations
        evaluations: Arc<AtomicUsize>,
    }

    impl TestChallenge {
        fn new(target: f64) -> Self {
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
            self.evaluations.fetch_add(1, Ordering::SeqCst);
            -((phenotype.value - self.target).abs()) // Negative distance (higher is better)
        }
    }

    #[test]
    fn test_local_search_manager_all_individuals() {
        let mut population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![-1.0, -2.0, -3.0]; // Distance from target=0
        let challenge = TestChallenge::new(0.0);
        let hill_climbing = HillClimbing::new(10).unwrap();
        let all_strategy = AllIndividualsStrategy::new();
        let manager = LocalSearchManager::new(hill_climbing, all_strategy);
        let mut rng = RandomNumberGenerator::from_seed(42);

        let result = manager.apply(&mut population, &fitness, &challenge, Some(&mut rng)).unwrap();
        
        // All individuals should have been selected for local search
        assert_eq!(result.len(), 3);
        
        // The challenge should have been evaluated multiple times
        assert!(challenge.get_evaluations() > 0);
    }

    #[test]
    fn test_local_search_manager_top_n() {
        let mut population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![-1.0, -2.0, -3.0]; // Distance from target=0, lower is better
        let challenge = TestChallenge::new(0.0);
        let hill_climbing = HillClimbing::new(10).unwrap();
        let top_strategy = TopNStrategy::new_minimizing(1); // Only the best individual
        let manager = LocalSearchManager::new(hill_climbing, top_strategy);
        let mut rng = RandomNumberGenerator::from_seed(42);

        let result = manager.apply(&mut population, &fitness, &challenge, Some(&mut rng)).unwrap();
        
        // All individuals should have a result
        assert_eq!(result.len(), 3);
        
        // The challenge should have been evaluated multiple times
        assert!(challenge.get_evaluations() > 0);
    }

    #[test]
    fn test_local_search_manager_empty_population() {
        let mut population: Vec<TestPhenotype> = vec![];
        let fitness: Vec<f64> = vec![];
        let challenge = TestChallenge::new(0.0);
        let hill_climbing = HillClimbing::new(10).unwrap();
        let all_strategy = AllIndividualsStrategy::new();
        let manager = LocalSearchManager::new(hill_climbing, all_strategy);
        let mut rng = RandomNumberGenerator::from_seed(42);

        let result = manager.apply(&mut population, &fitness, &challenge, Some(&mut rng));
        
        // Should return an error for empty population
        assert!(result.is_err());
    }

    #[test]
    fn test_local_search_manager_mismatched_lengths() {
        let mut population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
        ];
        let fitness = vec![-1.0, -2.0, -3.0]; // One more than population
        let challenge = TestChallenge::new(0.0);
        let hill_climbing = HillClimbing::new(10).unwrap();
        let all_strategy = AllIndividualsStrategy::new();
        let manager = LocalSearchManager::new(hill_climbing, all_strategy);
        let mut rng = RandomNumberGenerator::from_seed(42);

        let result = manager.apply(&mut population, &fitness, &challenge, Some(&mut rng));
        
        // Should return an error for mismatched lengths
        assert!(result.is_err());
    }
} 