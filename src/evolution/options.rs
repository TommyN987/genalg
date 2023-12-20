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
}

impl Default for EvolutionOptions {
    fn default() -> Self {
        Self {
            num_generations: 100,
            log_level: LogLevel::None,
            population_size: 2,
            num_offsprings: 20,
        }
    }
}
