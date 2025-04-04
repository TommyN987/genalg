use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use tracing::{debug, info};

use super::{
    builder::EvolutionLauncherBuilder,
    challenge::Challenge,
    options::{EvolutionOptions, LogLevel},
};
use crate::{
    breeding::BreedStrategy,
    error::{GeneticError, Result},
    local_search::{LocalSearchApplicationStrategy, LocalSearchManager},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::SelectionStrategy,
    LocalSearch, OptionExt,
};

/// Represents the result of an evolution, containing a phenotype and its associated score.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EvolutionResult<Pheno: Phenotype> {
    /// The evolved phenotype.
    pub pheno: Pheno,
    /// The fitness score of the phenotype.
    pub score: f64,
}

/// Manages the evolution process using a specified breeding strategy, selection strategy, and challenge.
///
/// The `EvolutionLauncher` is the central component of the genetic algorithm framework.
/// It coordinates the entire evolutionary process, including:
/// - Population initialization
/// - Fitness evaluation
/// - Parent selection
/// - Breeding new individuals
/// - Local search (optional)
/// - Tracking the best solution
///
/// # Type Parameters
///
/// * `P` - The phenotype type that represents individuals in the population
/// * `B` - The breeding strategy type
/// * `S` - The selection strategy type
/// * `LS` - The local search algorithm type
/// * `F` - The challenge (fitness function) type
/// * `A` - The local search application strategy type
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
    /// Creates a new `EvolutionLauncher` instance with the specified breeding strategy, selection strategy, local search manager, and challenge.
    ///
    /// # Arguments
    ///
    /// * `breed_strategy` - The breeding strategy used for generating offspring during evolution.
    /// * `selection_strategy` - The selection strategy used for selecting parents for the next generation.
    /// * `local_search_manager` - An optional local search manager for refining solutions. Use `None` if local search is not needed.
    /// * `challenge` - The challenge used to evaluate the fitness of phenotypes.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncher` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy},
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// let breed_strategy = OrdinaryStrategy::default();
    /// let selection_strategy = ElitistSelection::default();
    /// let challenge = MyChallenge;
    ///
    /// // Create a launcher without local search
    /// let launcher: EvolutionLauncher<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// > = EvolutionLauncher::new(
    ///     breed_strategy,
    ///     selection_strategy,
    ///     None, // No local search
    ///     challenge
    /// );
    /// ```
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

    /// Returns a builder for creating an `EvolutionLauncher` instance.
    ///
    /// The builder pattern provides a more flexible way to create an `EvolutionLauncher`
    /// by allowing you to set components one at a time.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncherBuilder` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy},
    /// #     error::Result,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// fn create_launcher() -> Result<EvolutionLauncher<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >> {
    ///     let breed_strategy = OrdinaryStrategy::default();
    ///     let selection_strategy = ElitistSelection::default();
    ///     let challenge = MyChallenge;
    ///     
    ///     // Create a local search strategy and application strategy
    ///     let hill_climbing = HillClimbing::new(10, 10)?;
    ///     let application_strategy = AllIndividualsStrategy::new();
    ///     
    ///     // Use the builder pattern
    ///     EvolutionLauncher::builder()
    ///         .with_breed_strategy(breed_strategy)
    ///         .with_selection_strategy(selection_strategy)
    ///         .with_local_search_manager(hill_climbing, application_strategy)
    ///         .with_challenge(challenge)
    ///         .build()
    /// }
    /// ```
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
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy},
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
    /// # #[derive(Clone)]
    /// # struct MyChallenge { target: f64 }
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { 1.0 / (phenotype.value - self.target).abs().max(0.001) }
    /// # }
    /// # let breed_strategy = OrdinaryStrategy::default();
    /// # let selection_strategy = ElitistSelection::default();
    /// # let challenge = MyChallenge { target: 42.0 };
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// let launcher: EvolutionLauncher<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// > = EvolutionLauncher::new(breed_strategy, selection_strategy, None, challenge);
    ///
    /// // Configure the evolution process
    /// let process = launcher.configure(options, starting_value);
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
            use_local_search: false,
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
    /// * `rng` - A random number generator for introducing randomness.
    /// * `starting_value` - The initial phenotype from which evolution begins.
    /// * `use_local_search` - Whether to apply local search during evolution.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best-evolved phenotype and its associated score,
    /// or a `GeneticError` if evolution fails.
    ///
    /// # Process
    ///
    /// The evolution process follows these steps for each generation:
    /// 1. Evaluate the fitness of all candidates
    /// 2. Select parents for the next generation based on fitness
    /// 3. Breed new candidates from the selected parents
    /// 4. Apply local search to refine solutions (if enabled)
    /// 5. Repeat until the maximum number of generations is reached
    fn evolve(
        &self,
        options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
        starting_value: P,
        use_local_search: bool,
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

        let mut fitness: Vec<EvolutionResult<P>> = Vec::new();

        // Initialize the population
        let mut candidates: Vec<P> = Vec::new();
        let mut parents: Vec<P> = vec![starting_value];
        self.breed(&mut candidates, &parents, options, rng, 0)?;

        for generation in 0..options.get_num_generations() {
            let population: Vec<P> = candidates.to_vec();

            // 1. Evaluation
            if candidates.len() >= options.get_parallel_threshold() {
                fitness = self.fitness_parallel(&candidates, &self.challenge)?;
            } else {
                fitness = self.fitness_sequential(&candidates, &self.challenge)?;
            }

            // Extract scores from the newly calculated fitness
            let scores: Vec<f64> = fitness.iter().map(|f| f.score).collect();

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

            // 2. Selection
            parents = self.selection_strategy.select(
                &population,
                &scores,
                options.get_population_size(),
            )?;

            // 3. Breeding
            self.breed(&mut candidates, &parents, options, rng, generation)?;

            // 4. Local Search
            if use_local_search {
                match &self.local_search_manager {
                    Some(manager) => {
                        manager.apply(&mut candidates, &scores, &self.challenge)?;
                    }
                    None => {
                        return Err(GeneticError::Configuration(
                            "Local search algorithm not provided".to_string(),
                        ))
                    }
                }
            }

            if fitness.is_empty() {
                return Err(GeneticError::Evolution(format!(
                    "No viable candidates produced in generation {}",
                    generation
                )));
            }
        }

        fitness.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or_else(|| {
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

    /// Breeds new candidates using the breeding strategy.
    ///
    /// This internal method is called during the evolution process to generate
    /// new candidates from the selected parents.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The vector to store the newly bred candidates.
    /// * `parents` - The selected parents for breeding.
    /// * `options` - Evolution options controlling the breeding process.
    /// * `rng` - A random number generator for introducing randomness.
    /// * `generation` - The current generation number (for error reporting).
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the breeding process.
    fn breed(
        &self,
        candidates: &mut Vec<P>,
        parents: &[P],
        options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
        generation: usize,
    ) -> std::result::Result<(), GeneticError> {
        match self.breed_strategy.breed(parents, options, rng) {
            Ok(bred_candidates) => {
                candidates.extend(bred_candidates);
                Ok(())
            }
            Err(e) => Err(GeneticError::Breeding(format!(
                "Failed to breed candidates in generation {}: {}",
                generation, e
            ))),
        }
    }

    /// Evaluates the fitness of candidates in parallel.
    ///
    /// This method is used when the number of candidates exceeds the parallel threshold.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The candidates to evaluate.
    /// * `challenge` - The challenge used to evaluate fitness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the evaluated candidates with their fitness scores.
    fn fitness_parallel(&self, candidates: &[P], challenge: &F) -> Result<Vec<EvolutionResult<P>>> {
        candidates
            .par_iter()
            .map(|candidate| {
                let score = challenge.score(candidate);

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
            .collect()
    }

    /// Evaluates the fitness of candidates sequentially.
    ///
    /// This method is used when the number of candidates is below the parallel threshold.
    ///
    /// # Arguments
    ///
    /// * `candidates` - The candidates to evaluate.
    /// * `challenge` - The challenge used to evaluate fitness.
    ///
    /// # Returns
    ///
    /// A `Result` containing the evaluated candidates with their fitness scores.
    fn fitness_sequential(
        &self,
        candidates: &[P],
        challenge: &F,
    ) -> Result<Vec<EvolutionResult<P>>> {
        candidates
            .iter()
            .map(|candidate| {
                let score = challenge.score(candidate);

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
            .collect()
    }
}

/// Process for running an evolution with the specified configuration.
///
/// The `EvolutionProcess` struct is a builder for an evolution run.
/// It allows configuring various aspects of the evolution process
/// before running it.
///
/// # Type Parameters
///
/// * `P` - The phenotype type that represents individuals in the population
/// * `B` - The breeding strategy type
/// * `S` - The selection strategy type
/// * `LS` - The local search algorithm type
/// * `F` - The challenge (fitness function) type
/// * `A` - The local search application strategy type
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
    use_local_search: bool,
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
    /// Sets a seed for the random number generator.
    ///
    /// Using a seed ensures reproducible evolution runs.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed for the random number generator.
    ///
    /// # Returns
    ///
    /// The `EvolutionProcess` with the seed set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy, LocalSearchManager},
    /// # };
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// # let hill_climbing = HillClimbing::new(10, 10).unwrap();
    /// # let application_strategy = AllIndividualsStrategy::new();
    /// # let local_search_manager = LocalSearchManager::new(hill_climbing, application_strategy);
    /// # let breed_strategy = OrdinaryStrategy::default();
    /// # let selection_strategy = ElitistSelection::default();
    /// # let challenge = MyChallenge;
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// # let launcher: EvolutionLauncher<MyPhenotype, OrdinaryStrategy, ElitistSelection, HillClimbing, MyChallenge, AllIndividualsStrategy> = EvolutionLauncher::new(breed_strategy, selection_strategy, Some(local_search_manager), challenge);
    ///
    /// // Configure the evolution process and set a specific seed
    /// let process = launcher
    ///     .configure(options, starting_value)
    ///     .with_seed(42);
    /// ```
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enables local search in the evolution process.
    ///
    /// When local search is enabled, the evolution process will apply
    /// the configured local search algorithm to refine solutions.
    ///
    /// # Returns
    ///
    /// The `EvolutionProcess` with local search enabled.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy, LocalSearchManager},
    /// # };
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// # let hill_climbing = HillClimbing::new(10, 10).unwrap();
    /// # let application_strategy = AllIndividualsStrategy::new();
    /// # let local_search_manager = LocalSearchManager::new(hill_climbing, application_strategy);
    /// # let breed_strategy = OrdinaryStrategy::default();
    /// # let selection_strategy = ElitistSelection::default();
    /// # let challenge = MyChallenge;
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// # let launcher: EvolutionLauncher<MyPhenotype, OrdinaryStrategy, ElitistSelection, HillClimbing, MyChallenge, AllIndividualsStrategy> = EvolutionLauncher::new(breed_strategy, selection_strategy, Some(local_search_manager), challenge);
    ///
    /// // Configure the evolution process with local search enabled
    /// let process = launcher
    ///     .configure(options, starting_value)
    ///     .with_local_search();
    /// ```
    pub fn with_local_search(mut self) -> Self {
        self.use_local_search = true;
        self
    }

    /// Runs the evolution process.
    ///
    /// This method runs the evolution process with the configured options
    /// and returns the best individual found.
    ///
    /// # Returns
    ///
    /// A `Result` containing the best individual found and its fitness score,
    /// or an error if the evolution process failed.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::{
    /// #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// #     breeding::OrdinaryStrategy,
    /// #     selection::ElitistSelection,
    /// #     local_search::{HillClimbing, AllIndividualsStrategy},
    /// # };
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// # let breed_strategy = OrdinaryStrategy::default();
    /// # let selection_strategy = ElitistSelection::default();
    /// # let challenge = MyChallenge;
    /// # let options = EvolutionOptions::default();
    /// # let starting_value = MyPhenotype { value: 0.0 };
    /// # let launcher: EvolutionLauncher<MyPhenotype, OrdinaryStrategy, ElitistSelection, HillClimbing, MyChallenge, AllIndividualsStrategy> = EvolutionLauncher::new(breed_strategy, selection_strategy, None, challenge);
    ///
    /// // Run the evolution process
    /// let result = launcher
    ///     .configure(options, starting_value)
    ///     .run();
    /// ```
    pub fn run(self) -> Result<EvolutionResult<P>> {
        let mut rng = match self.seed {
            Some(seed) => RandomNumberGenerator::from_seed(seed),
            None => RandomNumberGenerator::new(),
        };

        // Use local search if specified
        let use_local_search = if self.launcher.local_search_manager.is_some() {
            self.use_local_search
        } else {
            false
        };

        // Run the evolution without caching
        self.launcher.evolve(
            &self.options,
            &mut rng,
            self.starting_value,
            use_local_search,
        )
    }
}
