use std::{fmt::Error, marker::PhantomData};

use super::{
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::{phenotype::Phenotype, rng::RandomNumberGenerator, strategy::BreedStrategy};

/// Represents the result of an evolution, containing a phenotype and its associated score.
#[derive(Clone, Debug)]
pub struct EvolutionResult<Pheno: Phenotype> {
    pub pheno: Pheno,
    pub score: f64,
}

/// Manages the evolution process using a specified breeding strategy and challenge.
pub struct EvolutionLauncher<Pheno, Strategy, Chall>
where
    Pheno: Phenotype,
    Chall: Challenge<Pheno>,
    Strategy: BreedStrategy<Pheno>,
{
    strategy: Strategy,
    challenge: Chall,
    _marker: PhantomData<Pheno>,
}

impl<Pheno, Strategy, Chall> EvolutionLauncher<Pheno, Strategy, Chall>
where
    Pheno: Phenotype + Challenge<Pheno>,
    Chall: Challenge<Pheno>,
    Strategy: BreedStrategy<Pheno>,
{
    /// Creates a new `EvolutionLauncher` instance with the specified breeding strategy and challenge.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The breeding strategy used for generating offspring during evolution.
    /// * `challenge` - The challenge used to evaluate the fitness of phenotypes.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncher` instance.
    pub fn new(strategy: Strategy, challenge: Chall) -> Self {
        Self {
            strategy,
            challenge,
            _marker: PhantomData,
        }
    }
    /// Evolves a population of phenotypes over multiple generations.
    ///
    /// # Arguments
    ///
    /// * `options` - Evolution options controlling the evolution process.
    /// * `starting_value` - The initial phenotype from which evolution begins.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best-evolved phenotype and its associated score, or an `Error` if evolution fails.
    pub fn evolve(
        &self,
        options: &EvolutionOptions,
        starting_value: Pheno,
        rng: &mut RandomNumberGenerator,
    ) -> Result<EvolutionResult<Pheno>, Error> {
        let mut candidates: Vec<Pheno> = Vec::new();
        let mut fitness: Vec<EvolutionResult<Pheno>> = Vec::new();
        let mut parents: Vec<Pheno> = vec![starting_value];

        for generation in 0..options.get_num_generations() {
            candidates.clear();
            candidates.extend(self.strategy.breed(&parents, options, rng)?);

            fitness.clear();

            candidates.iter().for_each(|candidate| {
                let score = self.challenge.score(candidate);
                fitness.push(EvolutionResult {
                    pheno: *candidate,
                    score,
                })
            });

            fitness.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

            match options.get_log_level() {
                LogLevel::Minimal => println!("Generation: {}", generation),
                LogLevel::Verbose => {
                    fitness.iter().for_each(|result| {
                        println!("Generation: {} \n", generation);
                        println!("Phenotype: {:?} \n Score: {}", result.pheno, result.score);
                    });
                }
                LogLevel::None => {}
            }

            parents.clear();
            fitness
                .iter()
                .take(options.get_population_size())
                .for_each(|fitness_result| parents.push(fitness_result.pheno));
        }

        Ok(fitness[0].clone())
    }
}
