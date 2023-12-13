use std::{fmt::Error, marker::PhantomData};

use super::{
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::{phenotype::Phenotype, rng::RandomNumberGenerator, strategy::BreedStrategy};

#[derive(Clone, Debug)]
pub struct EvolutionResult<Pheno: Phenotype> {
    pub pheno: Pheno,
    pub score: f64,
}

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
    pub fn new(strategy: Strategy, challenge: Chall) -> Self {
        Self {
            strategy,
            challenge,
            _marker: PhantomData,
        }
    }

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
