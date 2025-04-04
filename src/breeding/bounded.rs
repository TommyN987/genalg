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
use std::{fmt::Debug, marker::PhantomData};

use rayon::prelude::*;
use tracing::{debug, info};

use crate::{
    breeding::BreedStrategy,
    error::{GeneticError, Result},
    evolution::options::EvolutionOptions,
    evolution::options::LogLevel,
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
};

/// Trait for phenotypes that have a measurable magnitude with defined bounds.
///
/// This trait is used by the `BoundedBreedStrategy` to ensure that phenotypes
/// stay within specified bounds during evolution.
pub trait Magnitude<Pheno: Phenotype> {
    /// Returns the current magnitude of the phenotype.
    fn magnitude(&self) -> f64;

    /// Returns the minimum allowed magnitude for the phenotype.
    fn min_magnitude(&self) -> f64;

    /// Returns the maximum allowed magnitude for the phenotype.
    fn max_magnitude(&self) -> f64;

    /// Checks if the phenotype's magnitude is within the allowed bounds.
    fn is_within_bounds(&self) -> bool {
        let mag = self.magnitude();
        mag >= self.min_magnitude() && mag <= self.max_magnitude() && mag.is_finite()
    }
}

/// Configuration for the `BoundedBreedStrategy`.
///
/// This struct holds the configuration parameters for the `BoundedBreedStrategy`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BoundedBreedConfig {
    /// The maximum number of attempts to develop a phenotype within bounds.
    pub max_development_attempts: usize,
}

impl Default for BoundedBreedConfig {
    fn default() -> Self {
        Self {
            max_development_attempts: 1000,
        }
    }
}

/// Builder for `BoundedBreedConfig`.
///
/// Provides a fluent interface for constructing `BoundedBreedConfig` instances.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct BoundedBreedConfigBuilder {
    max_development_attempts: Option<usize>,
}

impl BoundedBreedConfigBuilder {
    /// Sets the maximum number of development attempts.
    pub fn max_development_attempts(mut self, value: usize) -> Self {
        self.max_development_attempts = Some(value);
        self
    }

    /// Builds the `BoundedBreedConfig` instance.
    pub fn build(self) -> BoundedBreedConfig {
        BoundedBreedConfig {
            max_development_attempts: self.max_development_attempts.unwrap_or(1000),
        }
    }
}

impl BoundedBreedConfig {
    /// Returns a builder for creating a `BoundedBreedConfig` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use  genalg::breeding::{BoundedBreedConfig, BoundedBreedConfigBuilder};
    ///
    /// let config = BoundedBreedConfig::builder()
    ///     .max_development_attempts(2000)
    ///     .build();
    /// ```
    pub fn builder() -> BoundedBreedConfigBuilder {
        BoundedBreedConfigBuilder::default()
    }
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
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    _marker: PhantomData<Pheno>,
    config: BoundedBreedConfig,
}

impl<Pheno> Default for BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    fn default() -> Self {
        Self {
            _marker: PhantomData,
            config: BoundedBreedConfig::default(),
        }
    }
}

impl<Pheno> BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno>,
{
    /// Creates a new `BoundedBreedStrategy` instance with the specified maximum development attempts.
    ///
    /// # Arguments
    ///
    /// * `max_development_attempts` - The maximum number of attempts to develop a phenotype within bounds.
    ///
    /// # Returns
    ///
    /// A new `BoundedBreedStrategy` instance.
    pub fn new(max_development_attempts: usize) -> Self {
        Self {
            _marker: PhantomData,
            config: BoundedBreedConfig {
                max_development_attempts,
            },
        }
    }

    /// Creates a new `BoundedBreedStrategy` instance with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration for the strategy.
    ///
    /// # Returns
    ///
    /// A new `BoundedBreedStrategy` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use  genalg::breeding::{BoundedBreedStrategy, BoundedBreedConfig, Magnitude};
    /// use genalg::phenotype::Phenotype;
    /// use genalg::rng::RandomNumberGenerator;
    ///
    /// #[derive(Clone, Debug)]
    /// struct MyPhenotype {
    ///     value: f64,
    /// }
    ///
    /// impl Phenotype for MyPhenotype {
    ///     fn crossover(&mut self, other: &Self) {
    ///         self.value = (self.value + other.value) / 2.0;
    ///     }
    ///
    ///     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
    ///         let values = rng.fetch_uniform(-0.1, 0.1, 1);
    ///         let delta = values.front().unwrap();
    ///         self.value += *delta as f64;
    ///     }
    /// }
    ///
    /// impl Magnitude<MyPhenotype> for MyPhenotype {
    ///     fn magnitude(&self) -> f64 {
    ///         self.value.abs()
    ///     }
    ///
    ///     fn min_magnitude(&self) -> f64 {
    ///         0.0
    ///     }
    ///
    ///     fn max_magnitude(&self) -> f64 {
    ///         100.0
    ///     }
    /// }
    ///
    /// let config = BoundedBreedConfig::builder()
    ///     .max_development_attempts(2000)
    ///     .build();
    ///
    /// let strategy = BoundedBreedStrategy::<MyPhenotype>::with_config(config);
    /// ```
    pub fn with_config(config: BoundedBreedConfig) -> Self {
        Self {
            _marker: PhantomData,
            config,
        }
    }

    /// Creates a new `BoundedBreedStrategy` instance with custom parameters.
    ///
    /// # Note
    ///
    /// This constructor is maintained for backward compatibility.
    /// The parallel threshold should now be set in `EvolutionOptions`.
    ///
    /// # Arguments
    ///
    /// * `max_development_attempts` - The maximum number of attempts to develop a phenotype within bounds.
    /// * `parallel_threshold` - This parameter is ignored. Set the threshold in `EvolutionOptions` instead.
    ///
    /// # Returns
    ///
    /// A new `BoundedBreedStrategy` instance.
    #[deprecated(
        since = "0.1.0",
        note = "Set parallel_threshold in EvolutionOptions instead"
    )]
    pub fn new_with_params(max_development_attempts: usize, _parallel_threshold: usize) -> Self {
        Self::new(max_development_attempts)
    }
}

impl<Pheno> BreedStrategy<Pheno> for BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude<Pheno> + Send + Sync,
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
    /// A `Result` containing a vector of offspring phenotypes or a `GeneticError` if breeding fails.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The parents slice is empty
    /// - A phenotype cannot be developed within the specified bounds
    ///
    /// # Performance
    ///
    /// This method uses parallel processing for developing phenotypes when the number of
    /// offspring is large enough to benefit from parallelism. Each phenotype is developed
    /// in parallel using Rayon's parallel iterator.
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<Pheno>> {
        // Check if parents slice is empty
        if parents.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        let winner_previous_generation = parents[0].clone();

        // Prepare the children to be developed
        let mut children_to_develop: Vec<(Pheno, bool)> = Vec::new();

        // Add the winner of the previous generation (no initial mutation)
        children_to_develop.push((winner_previous_generation.clone(), false));

        // Create children through crossover with other parents
        for parent in parents.iter().skip(1) {
            let mut child = winner_previous_generation.clone();
            child.crossover(parent);
            children_to_develop.push((child, true));
        }

        // Create additional children through mutation only
        for _ in parents.len()..evol_options.get_num_offspring() {
            children_to_develop.push((winner_previous_generation.clone(), true));
        }

        // Develop all children (in parallel if there are enough)
        let parallel_threshold = evol_options.get_parallel_threshold();
        let log_level = evol_options.get_log_level();

        if children_to_develop.len() >= parallel_threshold {
            // Parallel development
            children_to_develop
                .into_par_iter()
                .map(|(pheno, initial_mutate)| {
                    self.develop_thread_local(pheno, initial_mutate, log_level)
                })
                .collect()
        } else {
            // Sequential development for small populations
            let mut developed_children = Vec::with_capacity(children_to_develop.len());

            for (pheno, initial_mutate) in children_to_develop {
                developed_children.push(self.develop(pheno, rng, initial_mutate, log_level)?);
            }

            Ok(developed_children)
        }
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
    /// * `log_level` - The log level for development progress logging.
    ///
    /// # Returns
    ///
    /// A `Result` containing the developed phenotype or a `GeneticError` if development fails.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The phenotype cannot be developed within the specified bounds after the maximum number of attempts
    /// - The phenotype's magnitude is not a finite number
    ///
    /// # Details
    ///
    /// This method attempts to develop a phenotype within the specified magnitude bounds.
    /// If `initial_mutate` is true, an initial mutation is applied to the input phenotype.
    /// The development process involves repeated mutation attempts until a phenotype
    /// within the specified bounds is achieved. If after the maximum number of attempts,
    /// a valid phenotype is not obtained, an error is returned.
    fn develop(
        &self,
        pheno: Pheno,
        rng: &mut RandomNumberGenerator,
        initial_mutate: bool,
        log_level: &LogLevel,
    ) -> Result<Pheno> {
        let mut phenotype = pheno;

        // Apply initial mutation if requested
        if initial_mutate {
            phenotype.mutate(rng);
        }

        // Check if the phenotype is already within bounds
        if phenotype.is_within_bounds() {
            return Ok(phenotype);
        }

        // Check if the magnitude is a valid number
        let mag = phenotype.magnitude();
        if !mag.is_finite() {
            return Err(GeneticError::InvalidNumericValue(format!(
                "Phenotype magnitude is not a finite number: {}",
                mag
            )));
        }

        // Try to develop the phenotype within bounds
        for attempt in 1..=self.config.max_development_attempts {
            phenotype.mutate(rng);

            if phenotype.is_within_bounds() {
                return Ok(phenotype);
            }

            // Check for NaN or infinity after mutation
            let mag = phenotype.magnitude();
            if !mag.is_finite() {
                return Err(GeneticError::InvalidNumericValue(format!(
                    "Phenotype magnitude became non-finite during development: {}",
                    mag
                )));
            }

            // If we've tried many times without success, log the progress
            if attempt % (self.config.max_development_attempts / 10) == 0 {
                match log_level {
                    LogLevel::Debug => {
                        debug!(
                            attempt,
                            max_attempts = self.config.max_development_attempts,
                            magnitude = phenotype.magnitude(),
                            min_bound = phenotype.min_magnitude(),
                            max_bound = phenotype.max_magnitude(),
                            "Development attempt progress"
                        );
                    }
                    LogLevel::Info => {
                        if attempt == self.config.max_development_attempts / 2 {
                            info!(
                                progress = "50%",
                                magnitude = phenotype.magnitude(),
                                min_bound = phenotype.min_magnitude(),
                                max_bound = phenotype.max_magnitude(),
                                "Development halfway complete"
                            );
                        }
                    }
                    LogLevel::None => {}
                }
            }
        }

        // If we've exhausted all attempts, return an error
        Err(GeneticError::MaxAttemptsReached(format!(
            "Failed to develop phenotype within bounds after {} attempts. Current magnitude: {}, min: {}, max: {}",
            self.config.max_development_attempts,
            phenotype.magnitude(),
            phenotype.min_magnitude(),
            phenotype.max_magnitude()
        )))
    }

    /// Develops a phenotype within the specified bounds, ensuring that the resulting
    /// phenotype satisfies the magnitude constraints.
    ///
    /// # Arguments
    ///
    /// * `pheno` - The initial phenotype to be developed.
    /// * `initial_mutate` - A flag indicating whether to apply initial mutation.
    /// * `log_level` - The log level for development progress logging.
    ///
    /// # Returns
    ///
    /// A `Result` containing the developed phenotype or a `GeneticError` if development fails.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The phenotype cannot be developed within the specified bounds after the maximum number of attempts
    /// - The phenotype's magnitude is not a finite number
    ///
    /// # Details
    ///
    /// This method attempts to develop a phenotype within the specified magnitude bounds.
    /// If `initial_mutate` is true, an initial mutation is applied to the input phenotype.
    /// The development process involves repeated mutation attempts until a phenotype
    /// within the specified bounds is achieved. If after the maximum number of attempts,
    /// a valid phenotype is not obtained, an error is returned.
    fn develop_thread_local(
        &self,
        pheno: Pheno,
        initial_mutate: bool,
        log_level: &LogLevel,
    ) -> Result<Pheno> {
        let mut phenotype = pheno;

        // Apply initial mutation if requested
        if initial_mutate {
            phenotype.mutate_thread_local();
        }

        // Check if the phenotype is already within bounds
        if phenotype.is_within_bounds() {
            return Ok(phenotype);
        }

        // Check if the magnitude is a valid number
        let mag = phenotype.magnitude();
        if !mag.is_finite() {
            return Err(GeneticError::InvalidNumericValue(format!(
                "Phenotype magnitude is not a finite number: {}",
                mag
            )));
        }

        // Try to develop the phenotype within bounds
        for attempt in 1..=self.config.max_development_attempts {
            phenotype.mutate_thread_local();

            if phenotype.is_within_bounds() {
                return Ok(phenotype);
            }

            // Check for NaN or infinity after mutation
            let mag = phenotype.magnitude();
            if !mag.is_finite() {
                return Err(GeneticError::InvalidNumericValue(format!(
                    "Phenotype magnitude became non-finite during development: {}",
                    mag
                )));
            }

            // If we've tried many times without success, log the progress
            if attempt % (self.config.max_development_attempts / 10) == 0 {
                match log_level {
                    LogLevel::Debug => {
                        debug!(
                            attempt,
                            max_attempts = self.config.max_development_attempts,
                            magnitude = phenotype.magnitude(),
                            min_bound = phenotype.min_magnitude(),
                            max_bound = phenotype.max_magnitude(),
                            "Development attempt progress"
                        );
                    }
                    LogLevel::Info => {
                        if attempt == self.config.max_development_attempts / 2 {
                            info!(
                                progress = "50%",
                                magnitude = phenotype.magnitude(),
                                min_bound = phenotype.min_magnitude(),
                                max_bound = phenotype.max_magnitude(),
                                "Development halfway complete"
                            );
                        }
                    }
                    LogLevel::None => {}
                }
            }
        }

        // If we've exhausted all attempts, return an error
        Err(GeneticError::MaxAttemptsReached(format!(
            "Failed to develop phenotype within bounds after {} attempts. Current magnitude: {}, min: {}, max: {}",
            self.config.max_development_attempts,
            phenotype.magnitude(),
            phenotype.min_magnitude(),
            phenotype.max_magnitude()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        evolution::options::{EvolutionOptions, LogLevel},
        phenotype::Phenotype,
        rng::RandomNumberGenerator,
    };

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        value: f64,
        min_bound: f64,
        max_bound: f64,
    }

    impl TestPhenotype {
        fn new(value: f64, min_bound: f64, max_bound: f64) -> Self {
            Self {
                value,
                min_bound,
                max_bound,
            }
        }
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2.0;
        }

        fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
            let delta = *rng.fetch_uniform(-0.1, 0.1, 1).front().unwrap() as f64;
            self.value += delta;
        }
    }

    impl Magnitude<TestPhenotype> for TestPhenotype {
        fn magnitude(&self) -> f64 {
            self.value.abs()
        }

        fn min_magnitude(&self) -> f64 {
            self.min_bound
        }

        fn max_magnitude(&self) -> f64 {
            self.max_bound
        }
    }

    #[test]
    fn test_develop_within_bounds() {
        let mut rng = RandomNumberGenerator::new();
        let strategy = BoundedBreedStrategy::default();
        let pheno = TestPhenotype::new(5.0, 4.0, 6.0);

        let result = strategy.develop(pheno, &mut rng, false, &LogLevel::None);
        assert!(result.is_ok());

        let developed = result.unwrap();
        assert!(developed.magnitude() >= developed.min_magnitude());
        assert!(developed.magnitude() <= developed.max_magnitude());
    }

    #[test]
    fn test_develop_outside_bounds() {
        let mut rng = RandomNumberGenerator::new();
        // Use a small number of attempts to make the test faster
        let strategy = BoundedBreedStrategy::new(10);
        // Create a phenotype that's very unlikely to mutate into the valid range
        let pheno = TestPhenotype::new(100.0, 0.0, 0.1);

        let result = strategy.develop(pheno, &mut rng, false, &LogLevel::None);
        assert!(result.is_err());

        match result {
            Err(GeneticError::MaxAttemptsReached(_)) => (),
            _ => panic!("Expected MaxAttemptsReached error"),
        }
    }

    #[test]
    fn test_breed_empty_parents() {
        let mut rng = RandomNumberGenerator::new();
        let evol_options = EvolutionOptions::default();
        let strategy = BoundedBreedStrategy::<TestPhenotype>::default();

        let parents = Vec::<TestPhenotype>::new();

        let result = strategy.breed(&parents, &evol_options, &mut rng);
        assert!(result.is_err());

        match result {
            Err(GeneticError::EmptyPopulation) => (),
            _ => panic!("Expected EmptyPopulation error"),
        }
    }
}
