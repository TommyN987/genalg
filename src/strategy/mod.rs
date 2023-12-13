//! # BreedStrategy
//!
//! The `BreedStrategy` trait defines the interface for strategies responsible for breeding
//! new individuals (phenotypes) based on a set of parent individuals and evolution options.
pub mod bounded;
pub mod ordinary;

use std::fmt::Error;

use crate::{
    evolution::options::EvolutionOptions, phenotype::Phenotype, rng::RandomNumberGenerator,
};

/// # BreedStrategy
///
/// The `BreedStrategy` trait defines the interface for strategies responsible for breeding
/// new individuals (phenotypes) based on a set of parent individuals and evolution options.
pub trait BreedStrategy<Pheno: Phenotype> {
    /// Breeds new individuals based on a set of parent individuals and evolution options.
    /// The `breed` method is responsible for generating a new population of individuals
    /// based on a set of parent individuals, evolution options, and a random number generator.
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
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error>;
}

pub use bounded::{BoundedBreedStrategy, Magnitude};
pub use ordinary::OrdinaryStrategy;
