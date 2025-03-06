//! # Combinatorial Breeding Strategies
//!
//! This module provides specialized breeding strategies for combinatorial optimization problems.
//! These strategies are designed to handle discrete solution spaces and constraints effectively.

use crate::constraints::ConstraintManager;
use crate::error::{GeneticError, Result};
use crate::evolution::options::EvolutionOptions;
use crate::local_search::LocalSearch;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use crate::strategy::BreedStrategy;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct CombinatorialBreedConfig {
    /// The probability of applying local search to offspring
    pub local_search_probability: f64,
    /// The probability of repairing invalid solutions
    pub repair_probability: f64,
    /// The maximum number of repair attempts
    pub max_repair_attempts: usize,
    /// Whether to use elitism (preserving the best solutions)
    pub use_elitism: bool,
    /// The number of elite solutions to preserve
    pub num_elites: usize,
}

impl Default for CombinatorialBreedConfig {
    fn default() -> Self {
        Self {
            local_search_probability: 0.1,
            repair_probability: 0.5,
            max_repair_attempts: 10,
            use_elitism: true,
            num_elites: 1,
        }
    }
}

#[derive(Debug, Default)]
pub struct CombinatorialBreedConfigBuilder {
    local_search_probability: Option<f64>,
    repair_probability: Option<f64>,
    max_repair_attempts: Option<usize>,
    use_elitism: Option<bool>,
    num_elites: Option<usize>,
}

impl CombinatorialBreedConfigBuilder {
    pub fn local_search_probability(mut self, value: f64) -> Self {
        self.local_search_probability = Some(value);
        self
    }

    pub fn repair_probability(mut self, value: f64) -> Self {
        self.repair_probability = Some(value);
        self
    }

    pub fn max_repair_attempts(mut self, value: usize) -> Self {
        self.max_repair_attempts = Some(value);
        self
    }

    pub fn use_elitism(mut self, value: bool) -> Self {
        self.use_elitism = Some(value);
        self
    }

    pub fn num_elites(mut self, value: usize) -> Self {
        self.num_elites = Some(value);
        self
    }

    pub fn build(self) -> CombinatorialBreedConfig {
        let default = CombinatorialBreedConfig::default();
        CombinatorialBreedConfig {
            local_search_probability: self
                .local_search_probability
                .unwrap_or(default.local_search_probability),
            repair_probability: self
                .repair_probability
                .unwrap_or(default.repair_probability),
            max_repair_attempts: self
                .max_repair_attempts
                .unwrap_or(default.max_repair_attempts),
            use_elitism: self.use_elitism.unwrap_or(default.use_elitism),
            num_elites: self.num_elites.unwrap_or(default.num_elites),
        }
    }
}

impl CombinatorialBreedConfig {
    pub fn builder() -> CombinatorialBreedConfigBuilder {
        CombinatorialBreedConfigBuilder::default()
    }
}

/// A breeding strategy for combinatorial optimization problems.
///
/// This strategy is designed to handle discrete solution spaces and constraints effectively.
/// It supports:
/// - Constraint-based repair of invalid solutions
/// - Integration with local search algorithms
/// - Elitism (preserving the best solutions)
/// - Specialized crossover and mutation operators
#[derive(Debug, Clone)]
pub struct CombinatorialBreedStrategy<P, L, Ch>
where
    P: Phenotype,
    L: LocalSearch<P, Ch> + Clone,
    Ch: crate::evolution::Challenge<P> + Clone + std::fmt::Debug,
{
    config: CombinatorialBreedConfig,
    constraint_manager: ConstraintManager<P>,
    local_search: Option<L>,
    challenge: Ch,
}

impl<P, L, Ch> CombinatorialBreedStrategy<P, L, Ch>
where
    P: Phenotype,
    L: LocalSearch<P, Ch> + Clone,
    Ch: crate::evolution::Challenge<P> + Clone + std::fmt::Debug,
{
    /// Creates a new combinatorial breed strategy with the given configuration.
    pub fn new(config: CombinatorialBreedConfig, challenge: Ch) -> Self {
        Self {
            config,
            constraint_manager: ConstraintManager::new(),
            local_search: None,
            challenge,
        }
    }

    /// Creates a new combinatorial breed strategy with the default configuration.
    pub fn default_config(challenge: Ch) -> Self {
        Self::new(CombinatorialBreedConfig::default(), challenge)
    }

    /// Adds a constraint to the strategy.
    pub fn add_constraint<T>(&mut self, constraint: T) -> &mut Self
    where
        T: crate::constraints::Constraint<P> + 'static,
    {
        self.constraint_manager.add_constraint(constraint);
        self
    }

    /// Sets the local search algorithm to use.
    pub fn with_local_search(&mut self, local_search: L) -> &mut Self {
        self.local_search = Some(local_search);
        self
    }

    /// Attempts to repair a solution that violates constraints.
    fn repair_solution(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // Check if there are any violations
        if self.constraint_manager.is_valid(phenotype) {
            return true;
        }

        // Try to repair the solution
        for _ in 0..self.config.max_repair_attempts {
            if self.constraint_manager.repair_all_with_rng(phenotype, rng) {
                return true;
            }
        }

        false
    }

    /// Applies local search to a solution if configured.
    fn apply_local_search(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        if let Some(local_search) = &self.local_search {
            let uniform = rng.fetch_uniform(0.0, 1.0, 1);
            match uniform.front() {
                Some(val) if *val < self.config.local_search_probability as f32 => {
                    return local_search.search_with_rng(phenotype, &self.challenge, rng);
                }
                None => {
                    // If random generation fails, just skip local search
                    return false;
                }
                _ => {}
            }
        }

        false
    }

    /// Performs tournament selection on the parents.
    ///
    /// Returns the index of the selected parent based on fitness comparison.
    /// In tournament selection, two individuals are randomly chosen and the one
    /// with higher fitness is selected.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The random number generation fails
    /// - The parents array is empty
    fn tournament_selection(
        &self,
        parents: &[P],
        rng: &mut RandomNumberGenerator,
    ) -> Result<usize> {
        if parents.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        // Generate random indices for two parents
        let uniform1 = rng.fetch_uniform(0.0, parents.len() as f32, 1);
        let idx1 = match uniform1.front() {
            Some(val) => ((val * parents.len() as f32) as usize) % parents.len(),
            None => {
                return Err(GeneticError::RandomGeneration(
                    "Failed to generate random value for tournament selection".to_string(),
                ))
            }
        };

        let uniform2 = rng.fetch_uniform(0.0, parents.len() as f32, 1);
        let idx2 = match uniform2.front() {
            Some(val) => ((val * parents.len() as f32) as usize) % parents.len(),
            None => {
                return Err(GeneticError::RandomGeneration(
                    "Failed to generate random value for tournament selection".to_string(),
                ))
            }
        };

        // Compare fitness and return the index of the better parent
        let fitness1 = self.challenge.score(&parents[idx1]);
        let fitness2 = self.challenge.score(&parents[idx2]);

        // Higher fitness is better
        if fitness1 >= fitness2 {
            Ok(idx1)
        } else {
            Ok(idx2)
        }
    }
}

impl<P, L, Ch> BreedStrategy<P> for CombinatorialBreedStrategy<P, L, Ch>
where
    P: Phenotype,
    L: LocalSearch<P, Ch> + Clone + Send + Sync,
    Ch: crate::evolution::Challenge<P> + Clone + Send + Sync + std::fmt::Debug,
{
    fn breed(
        &self,
        parents: &[P],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<P>> {
        if parents.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        let mut children = Vec::with_capacity(evol_options.get_num_offspring());

        // Add elite parents if configured
        if self.config.use_elitism {
            let num_elites = self.config.num_elites.min(parents.len());
            for parent in parents.iter().take(num_elites) {
                children.push(parent.clone());
            }
        }

        // Generate remaining children
        while children.len() < evol_options.get_num_offspring() {
            // Select parents using tournament selection
            let parent1_idx = self.tournament_selection(parents, rng)?;
            let parent2_idx = self.tournament_selection(parents, rng)?;

            let parent1 = &parents[parent1_idx];
            let parent2 = &parents[parent2_idx];

            // Create child through crossover
            let mut child = parent1.clone();
            child.crossover(parent2);

            // Mutate child
            child.mutate(rng);

            // Repair if invalid and configured to do so
            let random = match rng.fetch_uniform(0.0, 1.0, 1).front() {
                Some(val) => *val,
                None => {
                    return Err(GeneticError::RandomGeneration(
                        "Failed to generate random value for repair probability".to_string(),
                    ))
                }
            };

            if random < self.config.repair_probability as f32
                && !self.constraint_manager.is_valid(&child)
            {
                self.repair_solution(&mut child, rng);
            }

            // Apply local search if configured
            self.apply_local_search(&mut child, rng);

            children.push(child);
        }

        Ok(children)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{Constraint, ConstraintViolation};
    use crate::evolution::Challenge;
    use crate::local_search::HillClimbing;
    use crate::phenotype::Phenotype;
    use crate::rng::RandomNumberGenerator;
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct TestPhenotype {
        values: Vec<usize>,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            if !other.values.is_empty() && !self.values.is_empty() {
                self.values[0] = other.values[0];
            }
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            if !self.values.is_empty() {
                self.values[0] += 1;
            }
        }
    }

    impl AsRef<Vec<usize>> for TestPhenotype {
        fn as_ref(&self) -> &Vec<usize> {
            &self.values
        }
    }

    impl AsMut<Vec<usize>> for TestPhenotype {
        fn as_mut(&mut self) -> &mut Vec<usize> {
            &mut self.values
        }
    }

    #[derive(Debug, Clone)]
    struct TestChallenge {
        target: usize,
        // Counter to track the number of evaluations
        evaluations: Arc<AtomicUsize>,
    }

    #[allow(dead_code)]
    impl TestChallenge {
        fn new(target: usize) -> Self {
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

            // Calculate the sum of values
            let sum: usize = phenotype.values.iter().sum();

            // Higher score is better (inverse of distance to target)
            1.0 / ((sum as isize - self.target as isize).abs() as f64 + 1.0)
        }
    }

    #[derive(Debug)]
    struct UniqueValuesConstraint;

    impl Constraint<TestPhenotype> for UniqueValuesConstraint {
        fn check(&self, phenotype: &TestPhenotype) -> Vec<ConstraintViolation> {
            let mut seen = HashSet::new();
            let mut violations = Vec::new();

            for (idx, &value) in phenotype.values.iter().enumerate() {
                if !seen.insert(value) {
                    violations.push(ConstraintViolation::new(
                        "UniqueValues",
                        format!("Duplicate value {} at position {}", value, idx),
                    ));
                }
            }

            violations
        }

        fn repair(&self, phenotype: &mut TestPhenotype) -> bool {
            let mut seen = HashSet::new();
            let mut changed = false;

            for i in 0..phenotype.values.len() {
                let mut value = phenotype.values[i];
                while !seen.insert(value) {
                    value += 1;
                    changed = true;
                }
                phenotype.values[i] = value;
            }

            changed
        }

        fn repair_with_rng(
            &self,
            phenotype: &mut TestPhenotype,
            _rng: &mut RandomNumberGenerator,
        ) -> bool {
            self.repair(phenotype)
        }
    }

    #[test]
    fn test_combinatorial_breed_config() {
        // Test default config
        let config = CombinatorialBreedConfig::default();
        assert_eq!(config.local_search_probability, 0.1);
        assert_eq!(config.repair_probability, 0.5);
        assert_eq!(config.max_repair_attempts, 10);
        assert!(config.use_elitism);
        assert_eq!(config.num_elites, 1);

        // Test builder
        let config = CombinatorialBreedConfig::builder()
            .local_search_probability(0.2)
            .repair_probability(0.9)
            .max_repair_attempts(20)
            .use_elitism(false)
            .num_elites(0)
            .build();

        assert_eq!(config.local_search_probability, 0.2);
        assert_eq!(config.repair_probability, 0.9);
        assert_eq!(config.max_repair_attempts, 20);
        assert!(!config.use_elitism);
        assert_eq!(config.num_elites, 0);
    }

    #[test]
    fn test_combinatorial_breed_strategy() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let challenge = TestChallenge::new(15);
        let mut strategy =
            CombinatorialBreedStrategy::<TestPhenotype, HillClimbing, TestChallenge>::new(
                config, challenge,
            );

        // Add a constraint
        strategy.add_constraint(UniqueValuesConstraint);

        // Create parents
        let parent1 = TestPhenotype {
            values: vec![1, 2, 3, 4, 5],
        };
        let parent2 = TestPhenotype {
            values: vec![6, 7, 8, 9, 10],
        };
        let parents = vec![parent1.clone(), parent2];

        // Create evolution options
        let mut options = crate::evolution::options::EvolutionOptions::default();
        options.set_num_offspring(5);

        // Create RNG
        let mut rng = RandomNumberGenerator::new();

        // Breed
        let result = strategy.breed(&parents, &options, &mut rng);

        // Check result
        assert!(result.is_ok());
        let children = result.unwrap();
        assert_eq!(children.len(), 5);

        // First child should be the elite parent
        assert_eq!(children[0], parent1);
    }

    #[test]
    fn test_repair_solution() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let challenge = TestChallenge::new(15);
        let mut strategy =
            CombinatorialBreedStrategy::<TestPhenotype, HillClimbing, TestChallenge>::new(
                config, challenge,
            );

        // Add a constraint
        strategy.add_constraint(UniqueValuesConstraint);

        // Create an invalid phenotype
        let mut phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };

        // Create RNG
        let mut rng = RandomNumberGenerator::new();

        // Repair
        let repaired = strategy.repair_solution(&mut phenotype, &mut rng);

        // Check result
        assert!(repaired);
        assert_eq!(phenotype.values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_tournament_selection() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let challenge = TestChallenge::new(15);
        let strategy =
            CombinatorialBreedStrategy::<TestPhenotype, HillClimbing, TestChallenge>::new(
                config.clone(),
                challenge,
            );

        // Create parents with different fitness values
        // The sum of parent1 values is 15, and parent2 is 40
        let parent1 = TestPhenotype {
            values: vec![1, 2, 3, 4, 5], // Sum = 15
        };
        let parent2 = TestPhenotype {
            values: vec![6, 7, 8, 9, 10], // Sum = 40
        };
        let parents = vec![parent1.clone(), parent2.clone()];

        // Create RNG with fixed seed for deterministic testing
        let mut rng = RandomNumberGenerator::from_seed(42);

        // Run tournament selection multiple times
        let mut selected_parent1_count = 0;
        let mut selected_parent2_count = 0;

        for _ in 0..100 {
            // Select
            let idx = strategy
                .tournament_selection(&parents, &mut rng)
                .expect("Tournament selection should not fail");

            // Count which parent was selected
            if idx == 0 {
                selected_parent1_count += 1;
            } else {
                selected_parent2_count += 1;
            }
        }

        // Since parent1 has better fitness (sum=15, target=15), it should be selected more often
        assert!(selected_parent1_count > selected_parent2_count);

        // Now test with a different target to make parent2 have better fitness
        let challenge = TestChallenge::new(40);
        let strategy =
            CombinatorialBreedStrategy::<TestPhenotype, HillClimbing, TestChallenge>::new(
                config, challenge,
            );

        // Reset counters
        selected_parent1_count = 0;
        selected_parent2_count = 0;

        // Reset RNG for deterministic testing
        let mut rng = RandomNumberGenerator::from_seed(42);

        for _ in 0..100 {
            // Select
            let idx = strategy
                .tournament_selection(&parents, &mut rng)
                .expect("Tournament selection should not fail");

            // Count which parent was selected
            if idx == 0 {
                selected_parent1_count += 1;
            } else {
                selected_parent2_count += 1;
            }
        }

        // Now parent2 should be selected more often (sum=40, target=40)
        assert!(selected_parent2_count > selected_parent1_count);

        // Test error handling with empty parents
        let empty_parents: Vec<TestPhenotype> = Vec::new();
        let result = strategy.tournament_selection(&empty_parents, &mut rng);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GeneticError::EmptyPopulation));
    }

    #[test]
    fn test_breed_empty_parents() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let challenge = TestChallenge::new(15);
        let strategy =
            CombinatorialBreedStrategy::<TestPhenotype, HillClimbing, TestChallenge>::new(
                config, challenge,
            );

        // Create empty parents
        let parents: Vec<TestPhenotype> = Vec::new();

        // Create evolution options
        let options = crate::evolution::options::EvolutionOptions::default();

        // Create RNG
        let mut rng = RandomNumberGenerator::new();

        // Breed
        let result = strategy.breed(&parents, &options, &mut rng);

        // Check result
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GeneticError::EmptyPopulation));
    }
}
