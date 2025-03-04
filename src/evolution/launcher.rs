use std::marker::PhantomData;

use super::{
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::{
    error::{GeneticError, OptionExt, Result},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::BreedStrategy,
};
use rayon::prelude::*;

/// Represents the result of an evolution, containing a phenotype and its associated score.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EvolutionResult<Pheno: Phenotype> {
    /// The evolved phenotype.
    pub pheno: Pheno,
    /// The fitness score of the phenotype.
    pub score: f64,
}

/// Manages the evolution process using a specified breeding strategy and challenge.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
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
    Pheno: Phenotype + Send + Sync,
    Chall: Challenge<Pheno> + Send + Sync,
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

    /// Configures the evolution process with the provided options and starting value.
    ///
    /// This method returns an `EvolutionProcess` that can be further configured
    /// before running the evolution.
    ///
    /// # Arguments
    ///
    /// * `options` - Evolution options controlling the evolution process.
    /// * `starting_value` - The initial phenotype from which evolution begins.
    ///
    /// # Returns
    ///
    /// An `EvolutionProcess` that can be used to run the evolution.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     strategy::OrdinaryStrategy,
    /// # };
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) { self.value = (self.value + other.value) / 2.0; }
    /// #     fn mutate(&mut self, rng: &mut RandomNumberGenerator) { 
    /// #         let values = rng.fetch_uniform(-0.1, 0.1, 1);
    /// #         let delta = values.front().unwrap();
    /// #         self.value += *delta as f64; 
    /// #     }
    /// # }
    /// # struct MyChallenge { target: f64 }
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { 1.0 / (phenotype.value - self.target).abs().max(0.001) }
    /// # }
    /// # let strategy = OrdinaryStrategy::default();
    /// # let challenge = MyChallenge { target: 42.0 };
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// let launcher = EvolutionLauncher::new(strategy, challenge);
    /// let result = launcher
    ///     .configure(options, starting_value)
    ///     .with_seed(42)  // Optional: Set a specific seed
    ///     .run();
    /// ```
    pub fn configure(&self, options: EvolutionOptions, starting_value: Pheno) -> EvolutionProcess<'_, Pheno, Strategy, Chall> {
        EvolutionProcess {
            launcher: self,
            options,
            starting_value,
            seed: None,
        }
    }

    /// Evolves a population of phenotypes over multiple generations.
    ///
    /// This is an internal method used by `EvolutionProcess::run()`.
    /// For public use, use the `configure()` method instead.
    ///
    /// # Arguments
    ///
    /// * `options` - Evolution options controlling the evolution process.
    /// * `starting_value` - The initial phenotype from which evolution begins.
    /// * `rng` - A random number generator for introducing randomness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best-evolved phenotype and its associated score,
    /// or a `GeneticError` if evolution fails.
    fn evolve(
        &self,
        options: &EvolutionOptions,
        starting_value: Pheno,
        rng: &mut RandomNumberGenerator,
    ) -> Result<EvolutionResult<Pheno>> {
        if options.get_population_size() == 0 {
            return Err(GeneticError::Configuration(
                "Population size cannot be zero".to_string(),
            ));
        }

        if options.get_num_offspring() == 0 {
            return Err(GeneticError::Configuration(
                "Number of offspring cannot be zero".to_string(),
            ));
        }

        let mut candidates: Vec<Pheno> = Vec::new();
        let mut fitness: Vec<EvolutionResult<Pheno>> = Vec::new();
        let mut parents: Vec<Pheno> = vec![starting_value];

        for generation in 0..options.get_num_generations() {
            candidates.clear();

            match self.strategy.breed(&parents, options, rng) {
                Ok(bred_candidates) => candidates.extend(bred_candidates),
                Err(e) => {
                    return Err(GeneticError::Breeding(format!(
                        "Failed to breed candidates in generation {}: {}",
                        generation, e
                    )));
                }
            }

            // Only use parallelism if we have enough candidates to make it worthwhile
            let parallel_threshold = options.get_parallel_threshold();

            if candidates.len() >= parallel_threshold {
                // Parallel fitness evaluation
                let parallel_fitness: Result<Vec<_>> = candidates
                    .par_iter()
                    .map(|candidate| {
                        let score = self.challenge.score(candidate);

                        // Check for invalid fitness scores
                        if !score.is_finite() {
                            return Err(GeneticError::FitnessCalculation(format!(
                                "Non-finite fitness score encountered: {}",
                                score
                            )));
                        }

                        Ok(EvolutionResult {
                            pheno: candidate.clone(),
                            score,
                        })
                    })
                    .collect();

                // Handle any errors from parallel evaluation
                fitness = parallel_fitness?;
            } else {
                // Sequential fitness evaluation for small populations
                fitness.clear();
                for candidate in &candidates {
                    let score = self.challenge.score(candidate);

                    // Check for invalid fitness scores
                    if !score.is_finite() {
                        return Err(GeneticError::FitnessCalculation(format!(
                            "Non-finite fitness score encountered: {}",
                            score
                        )));
                    }

                    fitness.push(EvolutionResult {
                        pheno: candidate.clone(),
                        score,
                    });
                }
            }

            // Sort by fitness (descending)
            fitness.sort_by(|a, b| {
                b.score.partial_cmp(&a.score).unwrap_or_else(|| {
                    // Handle NaN values by considering them less than any other value
                    if b.score.is_nan() {
                        std::cmp::Ordering::Less
                    } else if a.score.is_nan() {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
            });

            // Log progress if requested
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

            // Ensure we have at least one fitness result
            if fitness.is_empty() {
                return Err(GeneticError::Evolution(format!(
                    "No viable candidates produced in generation {}",
                    generation
                )));
            }

            // Select parents for the next generation
            parents.clear();
            fitness
                .iter()
                .take(options.get_population_size())
                .for_each(|fitness_result| parents.push(fitness_result.pheno.clone()));
        }

        // Return the best result
        fitness.first().cloned().ok_or_else_genetic(|| {
            GeneticError::Evolution(
                "Evolution completed but no viable candidates were produced".to_string(),
            )
        })
    }
}

/// Represents a configured evolution process that can be run.
///
/// This struct is created by the `configure` method on `EvolutionLauncher`
/// and provides a fluent interface for running the evolution process.
pub struct EvolutionProcess<'a, Pheno, Strategy, Chall>
where
    Pheno: Phenotype,
    Chall: Challenge<Pheno>,
    Strategy: BreedStrategy<Pheno>,
{
    launcher: &'a EvolutionLauncher<Pheno, Strategy, Chall>,
    options: EvolutionOptions,
    starting_value: Pheno,
    seed: Option<u64>,
}

impl<'a, Pheno, Strategy, Chall> EvolutionProcess<'a, Pheno, Strategy, Chall>
where
    Pheno: Phenotype + Send + Sync,
    Chall: Challenge<Pheno> + Send + Sync,
    Strategy: BreedStrategy<Pheno>,
{
    /// Sets a specific seed for the random number generator.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed value for the random number generator.
    ///
    /// # Returns
    ///
    /// The `EvolutionProcess` with the seed configured.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Runs the evolution process.
    ///
    /// This is the main entry point for executing the evolution after configuration.
    /// It internally calls the private `evolve` method on the launcher.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best-evolved phenotype and its associated score,
    /// or a `GeneticError` if evolution fails.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The population size in options is zero
    /// - The number of offspring in options is zero
    /// - The breeding process fails
    /// - No viable candidates are produced in any generation
    ///
    /// # Performance
    ///
    /// This method uses parallel processing for fitness evaluation when the population
    /// size is large enough to benefit from parallelism. The fitness of each candidate
    /// is evaluated in parallel using Rayon's parallel iterator.
    pub fn run(self) -> Result<EvolutionResult<Pheno>> {
        let mut rng = match self.seed {
            Some(seed) => RandomNumberGenerator::from_seed(seed),
            None => RandomNumberGenerator::new(),
        };

        self.launcher.evolve(&self.options, self.starting_value, &mut rng)
    }
}
