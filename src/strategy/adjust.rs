//! # AdjustStrategy
//!
//! The `AdjustStrategy` struct represents a breeding strategy where the first
//! parent is considered as the winner of the previous generation, and the remaining
//! parents are used to create new individuals through crossover and mutation.
//! Furthermore depending on how well the fitness of the winner has increased compared to the previous generation, the number of mutate calls is adjusted.
//! If the increase was significant the number of mutate calls is decreased, otherwise it is increased.
use super::BreedStrategy;
use crate::phenotype::Phenotype;
use std::{fmt::Error, marker::PhantomData};

/// # AdjustStrategy
///
/// The `AdjustStrategy` struct represents a breeding strategy where the first
/// parent is considered as the winner of the previous generation, and the remaining
/// parents are used to create new individuals through crossover and mutation.
/// Furthermore depending on how well the fitness of the winner has increased compared to the previous generation, the number of mutate calls is adjusted.
/// If the increase was significant the number of mutate calls is decreased, otherwise it is increased.
#[derive(Debug, Clone)]
pub struct AdjustStrategy<Pheno>
where
    Pheno: Phenotype,
{
    _marker: PhantomData<Pheno>,
}

impl<Pheno> Default for AdjustStrategy<Pheno>
where
    Pheno: Phenotype,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Pheno> BreedStrategy<Pheno> for AdjustStrategy<Pheno>
where
    Pheno: Phenotype,
{
    /// Breeds offspring from a set of parent phenotypes
    ///
    /// This method uses a winner-takes-all approach, selecting the first parent
    /// as the winner and evolving it to create offspring.
    ///
    /// # Arguments
    ///
    /// * `parents` - A slice of parent phenotypes.
    /// * `evol_options` - Evolution options controlling the breeding process.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of offspring phenotypes or an `Error` if breeding fails.
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &crate::evolution::options::EvolutionOptions,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error> {
        let mut children: Vec<Pheno> = Vec::new();
        let winner_previous_generation = parents[0].clone();

        children.push(winner_previous_generation.clone());

        parents
            .iter()
            .skip(1)
            .try_for_each(|parent| -> Result<(), Error> {
                let mut child = winner_previous_generation.clone();
                child.crossover(parent);
                let mutated_child = self.develop(child, rng)?;
                children.push(mutated_child);
                Ok(())
            })?;

        (parents.len()..evol_options.get_num_offspring()).try_for_each(
            |_| -> Result<(), Error> {
                let child = winner_previous_generation.clone();
                let mutated_child = self.develop(child, rng)?;
                children.push(mutated_child);
                Ok(())
            },
        )?;

        Ok(children)
    }
}

impl<Pheno> AdjustStrategy<Pheno>
where
    Pheno: Phenotype,
{
    /// Develops a phenotype with consideration of the fitness increase of the previous generation
    ///
    /// # Arguments
    ///
    /// * `pheno` - The initial phenotype to be developed.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the developed phenotype
    ///
    /// # Details
    ///
    /// This method develops a phenotype.
    /// The development process involves repeated mutations as often as calculated
    /// calculated by the method calculate_number_of_mutations.
    fn develop(
        &self,
        pheno: Pheno,
        rng: &mut crate::rng::RandomNumberGenerator,
    ) -> Result<Pheno, Error> {
        let mut phenotype = pheno;
        phenotype.mutate(rng);
        let number_of_mutations = self.calculate_number_of_mutations();
        // call mutate number_of_mutations times
        for _ in 0..number_of_mutations {
            phenotype.mutate(rng);
        }
        Ok(phenotype)
    }

    /// Calculates the number of mutations to be performed
    fn calculate_number_of_mutations(&self) -> usize {
        100
    }
}
