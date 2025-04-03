//! # OrdinaryStrategy
//!
//! The `OrdinaryStrategy` struct represents a basic breeding strategy where the first
//! parent is considered as the winner of the previous generation, and the remaining
//! parents are used to create new individuals through crossover and mutation.
use std::fmt::Debug;

use rayon::prelude::*;

use super::BreedStrategy;
use crate::{
    error::{GeneticError, Result},
    phenotype::Phenotype,
};

/// # OrdinaryStrategy
///
/// The `OrdinaryStrategy` struct represents a basic breeding strategy where the first
/// parent is considered as the winner of the previous generation, and the remaining
/// parents are used to create new individuals through crossover and mutation.
#[derive(Debug, Clone)]
pub struct OrdinaryStrategy {
    // No fields needed
}

impl OrdinaryStrategy {
    /// Creates a new `OrdinaryStrategy` instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Creates a new `OrdinaryStrategy` instance with a custom parallel threshold.
    ///
    /// # Note
    ///
    /// This constructor is maintained for backward compatibility.
    /// The parallel threshold should now be set in `EvolutionOptions`.
    ///
    /// # Arguments
    ///
    /// * `parallel_threshold` - This parameter is ignored. Set the threshold in `EvolutionOptions` instead.
    ///
    /// # Returns
    ///
    /// A new `OrdinaryStrategy` instance.
    #[deprecated(
        since = "0.1.0",
        note = "Set parallel_threshold in EvolutionOptions instead"
    )]
    pub fn new_with_threshold(_parallel_threshold: usize) -> Self {
        Self {}
    }
}

impl Default for OrdinaryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl<Pheno> BreedStrategy<Pheno> for OrdinaryStrategy
where
    Pheno: Phenotype + Send + Sync,
{
    /// Breeds new individuals based on a set of parent individuals and evolution options.
    ///
    /// The `breed` method is a basic breeding strategy where the first parent is considered
    /// as the winner of the previous generation, and the remaining parents are used to create
    /// new individuals through crossover and mutation.
    ///
    /// ## Parameters
    ///
    /// - `parents`: A slice containing the parent individuals.
    /// - `evol_options`: A reference to the evolution options specifying algorithm parameters.
    /// - `rng`: A mutable reference to the random number generator used for generating
    ///   random values during breeding.
    ///
    /// ## Returns
    ///
    /// A Result with vector containing the newly bred individuals.
    /// This strategy always returns a vector of individuals.
    ///
    /// ## Errors
    ///
    /// This method will return an error if the parents slice is empty.
    ///
    /// ## Performance
    ///
    /// This method uses parallel processing for creating offspring when the number of
    /// offspring is large enough to benefit from parallelism. Each offspring is created
    /// in parallel using Rayon's parallel iterator.
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &crate::evolution::options::EvolutionOptions,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Result<Vec<Pheno>> {
        // Check if parents slice is empty
        if parents.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        let winner_previous_generation = parents[0].clone();
        let mut children = Vec::with_capacity(evol_options.get_num_offspring());

        // Always include the winner of the previous generation
        children.push(winner_previous_generation.clone());

        // Prepare the breeding operations
        let crossover_parents: Vec<&Pheno> = parents.iter().skip(1).collect();
        let num_mutation_only = evol_options
            .get_num_offspring()
            .saturating_sub(parents.len());

        // Determine if we should use parallel processing
        let total_offspring = crossover_parents.len() + num_mutation_only;
        let parallel_threshold = evol_options.get_parallel_threshold();

        if total_offspring >= parallel_threshold {
            // Process crossover parents in parallel
            if !crossover_parents.is_empty() {
                let crossover_children: Vec<Pheno> = crossover_parents
                    .into_par_iter()
                    .map(|parent| {
                        let mut child = winner_previous_generation.clone();
                        child.crossover(parent);

                        child.mutate_thread_local();

                        child
                    })
                    .collect();

                children.extend(crossover_children);
            }

            // Process mutation-only children in parallel
            if num_mutation_only > 0 {
                let mutation_children: Vec<Pheno> = (0..num_mutation_only)
                    .into_par_iter()
                    .map(|_| {
                        let mut child = winner_previous_generation.clone();
                        child.mutate_thread_local();

                        child
                    })
                    .collect();

                children.extend(mutation_children);
            }
        } else {
            // Sequential processing for small populations
            // Process crossover parents
            for parent in crossover_parents {
                let mut child = winner_previous_generation.clone();
                child.crossover(parent);
                child.mutate(rng);
                children.push(child);
            }

            // Process mutation-only children
            for _ in 0..num_mutation_only {
                let mut child = winner_previous_generation.clone();
                child.mutate(rng);
                children.push(child);
            }
        }

        Ok(children)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        breeding::BreedStrategy, evolution::options::EvolutionOptions, phenotype::Phenotype,
        rng::RandomNumberGenerator,
    };

    #[allow(unused)]
    #[test]
    fn test_breed() {
        let mut rng = RandomNumberGenerator::new();
        let evol_options = EvolutionOptions::default();
        let strategy = super::OrdinaryStrategy::default();

        #[derive(Clone, Copy, Debug)]
        struct MockPhenotype;

        impl Phenotype for MockPhenotype {
            fn crossover(&mut self, _other: &Self) {}
            fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
        }

        let mut parents = Vec::<MockPhenotype>::new();

        parents.extend((0..5).map(|_| MockPhenotype));

        let children = strategy.breed(&parents, &evol_options, &mut rng).unwrap();

        assert_eq!(children.len(), evol_options.get_num_offspring());
    }

    #[test]
    fn test_breed_empty_parents() {
        let mut rng = RandomNumberGenerator::new();
        let evol_options = EvolutionOptions::default();
        let strategy = super::OrdinaryStrategy::default();

        #[derive(Clone, Copy, Debug)]
        struct MockPhenotype;

        impl Phenotype for MockPhenotype {
            fn crossover(&mut self, _other: &Self) {}
            fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
        }

        let parents = Vec::<MockPhenotype>::new();

        let result = strategy.breed(&parents, &evol_options, &mut rng);
        assert!(result.is_err());

        match result {
            Err(crate::error::GeneticError::EmptyPopulation) => (),
            _ => panic!("Expected EmptyPopulation error"),
        }
    }
}
