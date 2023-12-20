//! # OrdinaryStrategy
//!
//! The `OrdinaryStrategy` struct represents a basic breeding strategy where the first
//! parent is considered as the winner of the previous generation, and the remaining
//! parents are used to create new individuals through crossover and mutation.
use super::BreedStrategy;
use crate::phenotype::Phenotype;
use std::{cell::RefCell, fmt::Error};

/// # OrdinaryStrategy
///
/// The `OrdinaryStrategy` struct represents a basic breeding strategy where the first
/// parent is considered as the winner of the previous generation, and the remaining
/// parents are used to create new individuals through crossover and mutation.
#[derive(Debug, Clone)]
pub struct OrdinaryStrategy;

impl<Pheno> BreedStrategy<Pheno> for OrdinaryStrategy
where
    Pheno: Phenotype,
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
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &crate::evolution::options::EvolutionOptions,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error> {
        let mut children: Vec<Pheno> = Vec::new();
        let mut winner_previous_generation = RefCell::new(parents[0]);
        children.push(*winner_previous_generation.get_mut());

        parents.iter().skip(1).for_each(|parent| {
            let mut child = *winner_previous_generation.get_mut();
            child.crossover(parent);
            child.mutate(rng);
            children.push(child);
        });

        children.extend((parents.len()..evol_options.get_num_offspring()).map(|_| {
            let mut child = *winner_previous_generation.get_mut();
            child.mutate(rng);
            child
        }));

        Ok(children)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        evolution::options::EvolutionOptions, phenotype::Phenotype, rng::RandomNumberGenerator,
        strategy::BreedStrategy,
    };

    #[allow(unused)]
    #[test]
    fn test_breed() {
        let mut rng = RandomNumberGenerator::new();
        let evol_options = EvolutionOptions::default();
        let strategy = super::OrdinaryStrategy;

        #[derive(Clone, Copy, Debug)]
        struct MockPhenotype;

        impl Phenotype for MockPhenotype {
            fn crossover(&mut self, other: &Self) {}
            fn mutate(&mut self, rng: &mut RandomNumberGenerator) {}
        }

        let mut parents = Vec::<MockPhenotype>::new();

        parents.extend((0..5).into_iter().map(|value| {
            let child = MockPhenotype;
            child
        }));

        let children = strategy.breed(&parents, &evol_options, &mut rng).unwrap();

        assert_eq!(children.len(), evol_options.get_num_offspring());
    }
}
