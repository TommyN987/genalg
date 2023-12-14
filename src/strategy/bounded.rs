//! # BoundedBreedStrategy
//!
//! Similarly to `OrdinaryStrategy`, the `BoundedBreedStrategy` struct represents
//! a breeding strategy where the first parent is considered the winner
//! of the previous generation, and the remaining parents are used to create
//! new individuals through crossover and mutation.
//!
//! However, the `BoundedBreedStrategy` imposes bounds on the phenotypes during evolution.
//! The algorithm develops a phenotype within the specified bounds, ensuring that the resulting
//! phenotype satisfies the constraints set up by the `Magnitude` trait.
use std::{cell::RefCell, fmt::Error, marker::PhantomData};

use crate::{
    evolution::options::EvolutionOptions, phenotype::Phenotype, rng::RandomNumberGenerator,
};

use super::BreedStrategy;

pub trait Magnitude<Pheno: Phenotype> {
    fn magnitude(&self) -> f64;
    fn min_magnitude(&self) -> f64;
    fn max_magnitude(&self) -> f64;
}

/// # BoundedBreedStrategy
///
/// Similarly to `OrdinaryStrategy`, the `BoundedBreedStrategy` struct represents
/// a breeding strategy where the first parent is considered the winner
/// of the previous generation, and the remaining parents are used to create
/// new individuals through crossover and mutation.
///
/// However, the `BoundedBreedStrategy` imposes bounds on the phenotypes during evolution.
/// The algorithm develops a phenotype within the specified bounds, ensuring that the resulting
/// phenotype satisfies the constraints set up by the `Magnitude` trait.
pub struct BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    _marker: PhantomData<Pheno>,
}

impl<Pheno> Default for BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<Pheno> BreedStrategy<Pheno> for BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    /// Breeds offspring from a set of parent phenotypes, ensuring the offspring
    /// stays within the specified phenotype bounds.
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
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error> {
        let mut children: Vec<Pheno> = Vec::new();
        let mut winner_previous_generation = RefCell::new(parents[0]);

        children.push(self.develop(*winner_previous_generation.get_mut(), rng, false)?);

        parents
            .iter()
            .skip(1)
            .try_for_each(|parent| -> Result<(), Error> {
                let mut child = *winner_previous_generation.get_mut();
                child.crossover(parent);
                let mutated_child = self.develop(child, rng, true)?;
                children.push(mutated_child);
                Ok(())
            })?;

        (parents.len()..evol_options.get_num_offspring()).try_for_each(
            |_| -> Result<(), Error> {
                let child = *winner_previous_generation.get_mut();
                let mutated_child = self.develop(child, rng, true)?;
                children.push(mutated_child);
                Ok(())
            },
        )?;

        Ok(children)
    }
}

impl<Pheno> BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    /// Develops a phenotype within the specified bounds, ensuring that the resulting
    /// phenotype satisfies the magnitude constraints.
    ///
    /// # Arguments
    ///
    /// * `pheno` - The initial phenotype to be developed.
    /// * `rng` - A random number generator for introducing randomness.
    /// * `initial_mutate` - A flag indicating whether to apply initial mutation.
    ///
    /// # Returns
    ///
    /// A `Result` containing the developed phenotype or an `Error` if development fails.
    ///
    /// # Details
    ///
    /// This method attempts to develop a phenotype within the specified magnitude bounds.
    /// If `initial_mutate` is true, an initial mutation is applied to the input phenotype.
    /// The development process involves repeated mutation attempts until a phenotype
    /// within the specified bounds is achieved. If after 1000 attempts, a valid phenotype
    /// is not obtained, an error is returned.
    fn develop(
        &self,
        pheno: Pheno,
        rng: &mut RandomNumberGenerator,
        initial_mutate: bool,
    ) -> Result<Pheno, Error> {
        let mut phenotype = pheno;

        if initial_mutate {
            phenotype.mutate(rng);
        }

        let pheno_type_in_range = |ph: &Pheno| -> bool {
            ph.magnitude() >= ph.min_magnitude() && ph.magnitude() <= ph.max_magnitude()
        };

        let mut advance_pheno_n_times = |ph: Pheno, n: usize| -> Pheno {
            let mut phenotype = ph;
            for _ in 0..n {
                phenotype.mutate(rng);
                if pheno_type_in_range(&phenotype) {
                    break;
                }
            }
            phenotype
        };

        for _ in 0..1000 {
            if pheno_type_in_range(&phenotype) {
                return Ok(phenotype);
            }
            phenotype = advance_pheno_n_times(phenotype, 1000);
        }

        Err(Error)
    }
}
