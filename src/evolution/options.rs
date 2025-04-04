//! # Evolution Options
//!
//! This module provides the `EvolutionOptions` struct, which is used to configure
//! the evolution process.
//!
//! ## Example
//!
//! ```
//! use genalg::evolution::options::{EvolutionOptions, LogLevel};
//!
//! // Create a custom options instance
//! let custom_options = EvolutionOptions::new(200, LogLevel::Debug, 50, 10);
//!
//! // Or use the builder pattern
//! let builder_options = EvolutionOptions::builder()
//!     .num_generations(200)
//!     .log_level(LogLevel::Info)
//!     .population_size(50)
//!     .num_offspring(100)
//!     .build();
//! ```
//!
//! ## Structs
//!
//! ### `EvolutionOptions`
//!
//! The `EvolutionOptions` struct contains configuration parameters for the evolution process:
//!
//! - `num_generations`: The number of generations to evolve.
//! - `log_level`: The logging level for the algorithm, represented by the `LogLevel` enum.
//! - `population_size`: The size of the population in each generation.
//! - `num_offsprings`: The number of offspring to generate in each generation.
//! - `parallel_threshold`: The minimum number of items to process in parallel.
//! - `use_caching`: Whether to enable caching of fitness evaluations.
//! - `cache_type`: The type of cache to use.
//!
//! ### `LogLevel`
//!
//! The `LogLevel` enum defines the level of logging for the evolution process:
//!
//! #### Variants
//!
//! - `Debug`: Detailed logging including all phenotypes and scores (maps to tracing::debug!)
//! - `Info`: Basic progress information (maps to tracing::info!)
//! - `None`: No logging output
//!
//! ## Methods
//!
//! ### `EvolutionOptions::new(num_generations: usize, log_level: LogLevel, population_size: usize, num_offsprings: usize) -> Self`
//!
//! Creates a new `EvolutionOptions` instance with the specified parameters.
//!
//! ### `EvolutionOptions::new_with_threshold(num_generations: usize, log_level: LogLevel, population_size: usize, num_offsprings: usize, parallel_threshold: usize) -> Self`
//!
//! Creates a new `EvolutionOptions` instance with the specified parameters, including a custom parallel threshold.
//!
//! ### `EvolutionOptions::default() -> Self`
//!
//! Creates a new `EvolutionOptions` instance with default parameters.

/// Defines the level of logging for the evolution process.
///
/// This enum aligns with the standard tracing log levels:
/// - `Debug`: Detailed information for debugging (maps to tracing::debug!)
/// - `Info`: General information about progress (maps to tracing::info!)
/// - `None`: No logging output
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LogLevel {
    /// Detailed logging including all phenotypes and scores (maps to tracing::debug!)
    Debug,
    /// Basic progress information (maps to tracing::info!)
    Info,
    /// No logging output
    None,
}

/// Defines the type of caching to use for fitness evaluations.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheType {
    /// Use a global cache protected by a mutex.
    /// Better for small populations or when evaluations are extremely expensive.
    #[default]
    Global,

    /// Use thread-local caches for better parallel performance.
    /// Better for large populations with parallel processing.
    ThreadLocal,
}

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct EvolutionOptions {
    num_generations: usize,
    log_level: LogLevel,
    population_size: usize,
    num_offsprings: usize,
    /// Minimum number of items to process in parallel
    parallel_threshold: usize,
    /// Whether to enable caching of fitness evaluations
    use_caching: bool,
    /// The type of cache to use
    cache_type: CacheType,
}

impl EvolutionOptions {
    /// Creates a new `EvolutionOptions` instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `num_generations` - The number of generations to evolve.
    /// * `log_level` - The logging level for the algorithm.
    /// * `population_size` - The size of the population in each generation.
    /// * `num_offsprings` - The number of offspring to generate in each generation.
    ///
    /// # Returns
    ///
    /// A new `EvolutionOptions` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use genalg::evolution::options::{EvolutionOptions, LogLevel};
    ///
    /// let options = EvolutionOptions::new(100, LogLevel::Info, 10, 50);
    /// ```
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
            use_caching: false,       // Caching disabled by default
            cache_type: CacheType::Global,
        }
    }

    /// Creates a new `EvolutionOptions` instance with the specified parameters,
    /// including a custom parallel threshold.
    ///
    /// # Arguments
    ///
    /// * `num_generations` - The number of generations to evolve.
    /// * `log_level` - The logging level for the algorithm.
    /// * `population_size` - The size of the population in each generation.
    /// * `num_offsprings` - The number of offspring to generate in each generation.
    /// * `parallel_threshold` - The minimum number of items to process in parallel.
    ///
    /// # Returns
    ///
    /// A new `EvolutionOptions` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use genalg::evolution::options::{EvolutionOptions, LogLevel};
    ///
    /// let options = EvolutionOptions::new_with_threshold(100, LogLevel::Info, 10, 50, 1000);
    /// ```
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
            use_caching: false, // Caching disabled by default
            cache_type: CacheType::Global,
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

    /// Returns whether caching is enabled.
    pub fn get_use_caching(&self) -> bool {
        self.use_caching
    }

    /// Returns the type of cache to use.
    pub fn get_cache_type(&self) -> CacheType {
        self.cache_type
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

    /// Sets whether to use caching.
    pub fn set_use_caching(&mut self, use_caching: bool) {
        self.use_caching = use_caching;
    }

    /// Sets the type of cache to use.
    pub fn set_cache_type(&mut self, cache_type: CacheType) {
        self.cache_type = cache_type;
    }

    /// Creates a builder for constructing an `EvolutionOptions` instance.
    ///
    /// This method returns an `EvolutionOptionsBuilder` that can be used to
    /// construct an `EvolutionOptions` instance using the builder pattern.
    ///
    /// # Returns
    ///
    /// An `EvolutionOptionsBuilder` instance.
    ///
    /// # Example
    ///
    /// ```
    /// use genalg::evolution::options::{EvolutionOptions, LogLevel};
    ///
    /// let options = EvolutionOptions::builder()
    ///     .num_generations(200)
    ///     .log_level(LogLevel::Info)
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
            use_caching: false,       // Caching disabled by default
            cache_type: CacheType::Global,
        }
    }
}

/// Builder for `EvolutionOptions`.
///
/// Provides a fluent interface for constructing `EvolutionOptions` instances.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Default)]
pub struct EvolutionOptionsBuilder {
    num_generations: Option<usize>,
    log_level: Option<LogLevel>,
    population_size: Option<usize>,
    num_offsprings: Option<usize>,
    parallel_threshold: Option<usize>,
    use_caching: Option<bool>,
    cache_type: Option<CacheType>,
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

    /// Sets whether to use caching.
    pub fn use_caching(mut self, value: bool) -> Self {
        self.use_caching = Some(value);
        self
    }

    /// Sets the type of cache to use.
    pub fn cache_type(mut self, value: CacheType) -> Self {
        self.cache_type = Some(value);
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
            use_caching: self.use_caching.unwrap_or(false),
            cache_type: self.cache_type.unwrap_or(CacheType::Global),
        }
    }
}
