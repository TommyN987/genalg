//! # EvolutionOptions
//!
//! The `EvolutionOptions` struct represents the configuration options for an evolutionary
//! algorithm. It includes parameters such as the number of generations, logging level,
//! population size, and the number of offsprings.
//!
//! ## Example
//!
//! ```rust
//! use genalg::evolution::options::{EvolutionOptions, LogLevel};
//!
//! // Create a new EvolutionOptions instance with custom parameters
//! let custom_options = EvolutionOptions::new(200, LogLevel::Verbose, 50, 10);
//!
//! // Create a new EvolutionOptions instance with default parameters
//! let default_options = EvolutionOptions::default();
//! ```
//!
//! ## Structs
//!
//! ### `EvolutionOptions`
//!
//! A struct representing the configuration options for an evolutionary algorithm.
//!
//! #### Fields
//!
//! - `num_generations`: The number of generations for the evolutionary algorithm.
//! - `log_level`: The logging level for the algorithm, represented by the `LogLevel` enum.
//! - `population_size`: The size of the population in each generation.
//! - `num_offsprings`: The number of offsprings generated in each generation.
//! - `parallel_threshold`: The minimum number of items to process in parallel.
//!
//! ### `LogLevel`
//!
//! An enum representing different logging levels for the evolutionary algorithm.
//!
//! #### Variants
//!
//! - `Verbose`: Provides detailed logging information.
//! - `Minimal`: Provides minimal logging information.
//! - `None`: Disables logging.
//!
//! ## Methods
//!
//! ### `EvolutionOptions::new(num_generations: usize, log_level: LogLevel, population_size: usize, num_offsprings: usize) -> Self`
//!
//! Creates a new `EvolutionOptions` instance with the specified parameters.
//!
//! ### `EvolutionOptions::new_with_threshold(num_generations: usize, log_level: LogLevel, population_size: usize, num_offsprings: usize, parallel_threshold: usize) -> Self`
//!
//! Creates a new `EvolutionOptions` instance with all parameters specified.
//!
//! ### `EvolutionOptions::default() -> Self`
//!
//! Creates a new `EvolutionOptions` instance with default parameters.

#[derive(Debug, Clone)]
pub enum LogLevel {
    Verbose,
    Minimal,
    None,
}

#[derive(Debug, Clone)]
pub struct EvolutionOptions {
    num_generations: usize,
    log_level: LogLevel,
    population_size: usize,
    num_offsprings: usize,
    /// Minimum number of items to process in parallel
    parallel_threshold: usize,
}

impl EvolutionOptions {
    pub fn new(
        num_generations: usize,
        log_level: LogLevel,
        population_size: usize,
        num_offsprings: usize,
    ) -> Self {
        Self {
            num_generations,
            log_level,
            population_size,
            num_offsprings,
            parallel_threshold: 1000, // Default parallel threshold
        }
    }

    /// Creates a new `EvolutionOptions` instance with all parameters specified.
    ///
    /// # Arguments
    ///
    /// * `num_generations` - The number of generations for the evolutionary algorithm.
    /// * `log_level` - The logging level for the algorithm.
    /// * `population_size` - The size of the population in each generation.
    /// * `num_offsprings` - The number of offsprings generated in each generation.
    /// * `parallel_threshold` - The minimum number of items to process in parallel.
    pub fn new_with_threshold(
        num_generations: usize,
        log_level: LogLevel,
        population_size: usize,
        num_offsprings: usize,
        parallel_threshold: usize,
    ) -> Self {
        Self {
            num_generations,
            log_level,
            population_size,
            num_offsprings,
            parallel_threshold,
        }
    }

    pub fn get_num_generations(&self) -> usize {
        self.num_generations
    }

    pub fn get_log_level(&self) -> &LogLevel {
        &self.log_level
    }

    pub fn get_population_size(&self) -> usize {
        self.population_size
    }

    pub fn get_num_offspring(&self) -> usize {
        self.num_offsprings
    }

    /// Returns the minimum number of items to process in parallel.
    pub fn get_parallel_threshold(&self) -> usize {
        self.parallel_threshold
    }

    /// Sets the number of generations.
    pub fn set_num_generations(&mut self, num_generations: usize) {
        self.num_generations = num_generations;
    }

    /// Sets the log level.
    pub fn set_log_level(&mut self, log_level: LogLevel) {
        self.log_level = log_level;
    }

    /// Sets the population size.
    pub fn set_population_size(&mut self, population_size: usize) {
        self.population_size = population_size;
    }

    /// Sets the number of offspring.
    pub fn set_num_offspring(&mut self, num_offsprings: usize) {
        self.num_offsprings = num_offsprings;
    }

    /// Sets the parallel threshold.
    pub fn set_parallel_threshold(&mut self, threshold: usize) {
        self.parallel_threshold = threshold;
    }

    /// Returns a builder for creating an `EvolutionOptions` instance.
    ///
    /// This provides a more flexible way to configure evolution options
    /// with a fluent interface.
    ///
    /// # Example
    ///
    /// ```rust
    /// use genalg::evolution::options::{EvolutionOptions, LogLevel};
    ///
    /// let options = EvolutionOptions::builder()
    ///     .num_generations(200)
    ///     .log_level(LogLevel::Minimal)
    ///     .population_size(50)
    ///     .num_offspring(100)
    ///     .parallel_threshold(500)
    ///     .build();
    /// ```
    pub fn builder() -> EvolutionOptionsBuilder {
        EvolutionOptionsBuilder::default()
    }
}

impl Default for EvolutionOptions {
    fn default() -> Self {
        Self {
            num_generations: 100,
            log_level: LogLevel::None,
            population_size: 2,
            num_offsprings: 20,
            parallel_threshold: 1000, // Default parallel threshold
        }
    }
}

/// Builder for `EvolutionOptions`.
///
/// Provides a fluent interface for constructing `EvolutionOptions` instances.
#[derive(Debug, Clone)]
pub struct EvolutionOptionsBuilder {
    num_generations: Option<usize>,
    log_level: Option<LogLevel>,
    population_size: Option<usize>,
    num_offsprings: Option<usize>,
    parallel_threshold: Option<usize>,
}

impl Default for EvolutionOptionsBuilder {
    fn default() -> Self {
        Self {
            num_generations: None,
            log_level: None,
            population_size: None,
            num_offsprings: None,
            parallel_threshold: None,
        }
    }
}

impl EvolutionOptionsBuilder {
    /// Sets the number of generations.
    pub fn num_generations(mut self, value: usize) -> Self {
        self.num_generations = Some(value);
        self
    }

    /// Sets the log level.
    pub fn log_level(mut self, value: LogLevel) -> Self {
        self.log_level = Some(value);
        self
    }

    /// Sets the population size.
    pub fn population_size(mut self, value: usize) -> Self {
        self.population_size = Some(value);
        self
    }

    /// Sets the number of offspring.
    pub fn num_offspring(mut self, value: usize) -> Self {
        self.num_offsprings = Some(value);
        self
    }

    /// Sets the parallel threshold.
    pub fn parallel_threshold(mut self, value: usize) -> Self {
        self.parallel_threshold = Some(value);
        self
    }

    /// Builds the `EvolutionOptions` instance.
    pub fn build(self) -> EvolutionOptions {
        EvolutionOptions {
            num_generations: self.num_generations.unwrap_or(100),
            log_level: self.log_level.unwrap_or(LogLevel::None),
            population_size: self.population_size.unwrap_or(2),
            num_offsprings: self.num_offsprings.unwrap_or(20),
            parallel_threshold: self.parallel_threshold.unwrap_or(1000),
        }
    }
}