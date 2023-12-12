use super::BreedStrategy;
use crate::phenotype::Phenotype;
use std::cell::RefCell;

/// # OrdinaryStrategy
///
/// The `OrdinaryStrategy` struct represents a basic breeding strategy where the first
/// parent is considered as the winner of the previous generation, and the remaining
/// parents are used to create new individuals through crossover and mutation.
///
/// ## Implementing the Trait
///
/// To use the `OrdinaryStrategy`, implement the `BreedStrategy` trait for it.
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
    /// A vector containing the newly bred individuals.
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &crate::evol_options::EvolutionOptions,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Vec<Pheno> {
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

        children
    }
}
