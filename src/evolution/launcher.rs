use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{debug, info};

use super::{
    builder::EvolutionLauncherBuilder,
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::{
    error::{GeneticError, Result},
    local_search::{LocalSearchApplicationStrategy, LocalSearchManager},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::SelectionStrategy,
    strategy::BreedStrategy,
    LocalSearch, OptionExt,
};

/// Represents the result of an evolution, containing a phenotype and its associated score.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EvolutionResult<Pheno: Phenotype> {
    /// The evolved phenotype.
    pub pheno: Pheno,
    /// The fitness score of the phenotype.
    pub score: f64,
}

/// Manages the evolution process using a specified breeding strategy, selection strategy, and challenge.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EvolutionLauncher<P, B, S, LS, F, A>
where
    P: Phenotype,
    B: BreedStrategy<P>,
    S: SelectionStrategy<P>,
    LS: LocalSearch<P, F> + Clone,
    F: Challenge<P> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
{
    breed_strategy: B,
    selection_strategy: S,
    local_search_manager: Option<LocalSearchManager<P, LS, A, F>>,
    challenge: F,
}

impl<P, B, S, LS, F, A> EvolutionLauncher<P, B, S, LS, F, A>
where
    P: Phenotype + Send + Sync,
    LS: LocalSearch<P, F> + Clone + Send + Sync,
    F: Challenge<P> + Clone + Send + Sync,
    A: LocalSearchApplicationStrategy<P> + Clone + Send + Sync,
    B: BreedStrategy<P> + Clone + Send + Sync,
    S: SelectionStrategy<P> + Clone + Send + Sync,
{
    /// Creates a new `EvolutionLauncher` instance with the specified breeding strategy, selection strategy, and challenge.
    ///
    /// # Arguments
    ///
    /// * `breed_strategy` - The breeding strategy used for generating offspring during evolution.
    /// * `selection_strategy` - The selection strategy used for selecting parents for the next generation.
    /// * `challenge` - The challenge used to evaluate the fitness of phenotypes.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncher` instance.
    pub fn new(
        breed_strategy: B,
        selection_strategy: S,
        local_search_manager: Option<LocalSearchManager<P, LS, A, F>>,
        challenge: F,
    ) -> Self {
        Self {
            breed_strategy,
            selection_strategy,
            local_search_manager,
            challenge,
        }
    }

    pub fn builder() -> EvolutionLauncherBuilder<P, B, S, LS, F, A> {
        EvolutionLauncherBuilder::new()
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
    /// #     selection::ElitistSelection,
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
    /// # let breed_strategy = OrdinaryStrategy::default();
    /// # let selection_strategy = ElitistSelection::default();
    /// # let challenge = MyChallenge { target: 42.0 };
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// let launcher = EvolutionLauncher::new(breed_strategy, selection_strategy, challenge);
    /// let result = launcher
    ///     .configure(options, starting_value)
    ///     .with_seed(42)  // Optional: Set a specific seed
    ///     .run();
    /// ```
    pub fn configure(
        &self,
        options: EvolutionOptions,
        starting_value: P,
    ) -> EvolutionProcess<P, B, S, LS, F, A> {
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
        rng: &mut RandomNumberGenerator,
        starting_value: P,
    ) -> Result<EvolutionResult<P>> {
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

        let mut candidates: Vec<P> = Vec::new();
        let mut fitness: Vec<EvolutionResult<P>> = Vec::new();
        let mut parents: Vec<P> = vec![starting_value];

        for generation in 0..options.get_num_generations() {
            candidates.clear();

            match self.breed_strategy.breed(&parents, options, rng) {
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

            // Log progress if requested
            match options.get_log_level() {
                LogLevel::Info => info!(generation, "Evolution progress"),
                LogLevel::Debug => {
                    fitness.iter().for_each(|result| {
                        debug!(
                            generation,
                            phenotype = ?result.pheno,
                            score = result.score,
                            "Evolution detailed progress"
                        );
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

            // Extract phenotypes and scores for selection
            let population: Vec<P> = candidates.to_vec();
            let scores: Vec<f64> = fitness.iter().map(|f| f.score).collect();

            // Use the selection strategy to select parents for the next generation
            parents = self.selection_strategy.select(
                &population,
                &scores,
                options.get_population_size(),
                Some(rng),
            )?;
        }

        // Return the best result
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
/// It is not meant to be constructed directly by users.
pub struct EvolutionProcess<'a, P, B, S, LS, F, A>
where
    P: Phenotype,
    B: BreedStrategy<P>,
    S: SelectionStrategy<P>,
    LS: LocalSearch<P, F> + Clone,
    F: Challenge<P> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
{
    launcher: &'a EvolutionLauncher<P, B, S, LS, F, A>,
    options: EvolutionOptions,
    starting_value: P,
    seed: Option<u64>,
}

impl<P, B, S, LS, F, A> EvolutionProcess<'_, P, B, S, LS, F, A>
where
    P: Phenotype + Send + Sync,
    LS: LocalSearch<P, F> + Clone + Send + Sync,
    F: Challenge<P> + Clone + Send + Sync,
    A: LocalSearchApplicationStrategy<P> + Clone + Send + Sync,
    B: BreedStrategy<P> + Clone + Send + Sync,
    S: SelectionStrategy<P> + Clone + Send + Sync,
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
    pub fn run(self) -> Result<EvolutionResult<P>> {
        let mut rng = match self.seed {
            Some(seed) => RandomNumberGenerator::from_seed(seed),
            None => RandomNumberGenerator::new(),
        };

        self.launcher
            .evolve(&self.options, &mut rng, self.starting_value)
    }
}
