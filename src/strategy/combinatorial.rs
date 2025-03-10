//! # Combinatorial Breeding Strategies
//!
//! This module provides specialized breeding strategies for combinatorial optimization problems.
//! These strategies are designed to handle discrete solution spaces and constraints effectively.

use crate::constraints::ConstraintManager;
use crate::error::{GeneticError, Result};
use crate::evolution::options::EvolutionOptions;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use crate::strategy::BreedStrategy;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct CombinatorialBreedConfig {
    /// The probability of repairing invalid solutions
    pub repair_probability: f64,
    /// The maximum number of repair attempts
    pub max_repair_attempts: usize,
}

impl Default for CombinatorialBreedConfig {
    fn default() -> Self {
        Self {
            repair_probability: 0.5,
            max_repair_attempts: 10,
        }
    }
}

impl CombinatorialBreedConfig {
    pub fn builder() -> CombinatorialBreedConfigBuilder {
        CombinatorialBreedConfigBuilder::default()
    }
}

#[derive(Debug, Default)]
pub struct CombinatorialBreedConfigBuilder {
    repair_probability: Option<f64>,
    max_repair_attempts: Option<usize>,
}

impl CombinatorialBreedConfigBuilder {
    pub fn repair_probability(mut self, value: f64) -> Self {
        self.repair_probability = Some(value);
        self
    }

    pub fn max_repair_attempts(mut self, value: usize) -> Self {
        self.max_repair_attempts = Some(value);
        self
    }

    pub fn build(self) -> CombinatorialBreedConfig {
        let default = CombinatorialBreedConfig::default();
        CombinatorialBreedConfig {
            repair_probability: self
                .repair_probability
                .unwrap_or(default.repair_probability),
            max_repair_attempts: self
                .max_repair_attempts
                .unwrap_or(default.max_repair_attempts),
        }
    }
}

/// A breeding strategy for combinatorial optimization problems.
///
/// This strategy is designed to handle discrete solution spaces and constraints effectively.
/// It supports:
/// - Constraint-based repair of invalid solutions
/// - Specialized crossover and mutation operators
#[derive(Debug, Clone)]
pub struct CombinatorialBreedStrategy<P>
where
    P: Phenotype,
{
    config: CombinatorialBreedConfig,
    constraint_manager: ConstraintManager<P>,
}

impl<P> CombinatorialBreedStrategy<P>
where
    P: Phenotype,
{
    /// Creates a new combinatorial breed strategy with the given configuration.
    pub fn new(config: CombinatorialBreedConfig) -> Self {
        Self {
            config,
            constraint_manager: ConstraintManager::new(),
        }
    }

    /// Creates a new combinatorial breed strategy with the default configuration.
    pub fn default_config() -> Self {
        Self::new(CombinatorialBreedConfig::default())
    }

    /// Adds a constraint to the strategy.
    pub fn add_constraint<T>(&mut self, constraint: T) -> &mut Self
    where
        T: crate::constraints::Constraint<P> + 'static,
    {
        self.constraint_manager.add_constraint(constraint);
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
}

impl<P> BreedStrategy<P> for CombinatorialBreedStrategy<P>
where
    P: Phenotype,
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

        let num_offspring = evol_options.get_num_offspring();
        let mut offspring = Vec::with_capacity(num_offspring);

        // Generate offspring
        for _ in 0..num_offspring {
            // Randomly select two parents from the provided parents
            let parent1_idx = match rng.fetch_uniform(0.0, parents.len() as f32, 1).front() {
                Some(val) => (*val as usize) % parents.len(),
                None => {
                    return Err(GeneticError::RandomGeneration(
                        "Failed to generate random value for parent selection".to_string(),
                    ))
                }
            };
            
            let mut parent2_idx = match rng.fetch_uniform(0.0, parents.len() as f32, 1).front() {
                Some(val) => (*val as usize) % parents.len(),
                None => {
                    return Err(GeneticError::RandomGeneration(
                        "Failed to generate random value for parent selection".to_string(),
                    ))
                }
            };
            
            // Ensure parent2 is different from parent1
            while parent2_idx == parent1_idx && parents.len() > 1 {
                parent2_idx = match rng.fetch_uniform(0.0, parents.len() as f32, 1).front() {
                    Some(val) => (*val as usize) % parents.len(),
                    None => {
                        return Err(GeneticError::RandomGeneration(
                            "Failed to generate random value for parent selection".to_string(),
                        ))
                    }
                };
            }

            // Create child through crossover
            let mut child = parents[parent1_idx].clone();
            child.crossover(&parents[parent2_idx]);

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

            offspring.push(child);
        }

        Ok(offspring)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{Constraint, ConstraintViolation};
    use crate::evolution::Challenge;
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

    struct TestChallenge {
        target: usize,
        // Counter to track the number of evaluations
        evaluations: Arc<AtomicUsize>,
    }

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
            self.evaluations.fetch_add(1, Ordering::SeqCst);
            let sum: usize = phenotype.values.iter().sum();
            -((sum as isize - self.target as isize).abs() as f64)
        }
    }

    #[derive(Debug)]
    struct UniqueValuesConstraint;

    impl Constraint<TestPhenotype> for UniqueValuesConstraint {
        fn check(&self, phenotype: &TestPhenotype) -> Vec<ConstraintViolation> {
            let mut seen = HashSet::new();
            let mut violations = Vec::new();

            for (i, &value) in phenotype.values.iter().enumerate() {
                if !seen.insert(value) {
                    violations.push(ConstraintViolation::new(
                        "UniqueValuesConstraint",
                        format!("Duplicate value {} at index {}", value, i),
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
        assert_eq!(config.repair_probability, 0.5);
        assert_eq!(config.max_repair_attempts, 10);

        // Test builder
        let config = CombinatorialBreedConfig::builder()
            .repair_probability(0.9)
            .max_repair_attempts(20)
            .build();

        assert_eq!(config.repair_probability, 0.9);
        assert_eq!(config.max_repair_attempts, 20);
    }

    #[test]
    fn test_combinatorial_breed_strategy() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let mut strategy = CombinatorialBreedStrategy::<TestPhenotype>::new(config);

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
    }

    #[test]
    fn test_repair_solution() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let mut strategy = CombinatorialBreedStrategy::<TestPhenotype>::new(config);

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
    fn test_breed_empty_parents() {
        // Create a strategy
        let config = CombinatorialBreedConfig::default();
        let strategy = CombinatorialBreedStrategy::<TestPhenotype>::new(config);

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
