//! # Error Types
//!
//! This module defines custom error types for the genetic algorithm library.
//! It provides specific error variants for different failure scenarios that
//! may occur during the evolution process.
//!
//! ## Examples
//!
//! Using the `Result` type:
//!
//! ```rust
//! use genalg::error::{GeneticError, Result};
//!
//! fn some_function() -> Result<()> {
//!     // Function implementation
//!     Ok(())
//! }
//!
//! fn caller() {
//!     match some_function() {
//!         Ok(_) => println!("Success!"),
//!         Err(e) => println!("Error: {}", e),
//!     }
//! }
//! ```
//!
//! Using the `ResultExt` trait to add context to errors:
//!
//! ```rust
//! use genalg::error::{Result, ResultExt};
//! use std::fs::File;
//!
//! fn read_config_file(path: &str) -> Result<()> {
//!     File::open(path).context("Failed to open config file")
//!         .and_then(|_file| {
//!             // Read file contents
//!             Ok(())
//!         })
//! }
//! ```
//!
//! Using the `OptionExt` trait to convert `Option` to `Result`:
//!
//! ```rust
//! use genalg::error::{GeneticError, OptionExt};
//!
//! fn find_best_candidate(candidates: &[i32]) -> genalg::error::Result<i32> {
//!     candidates.iter().max().cloned().ok_or_else_genetic(||
//!         GeneticError::EmptyPopulation
//!     )
//! }
//! ```
//!
//! Using the `?` operator with automatic error conversion:
//!
//! ```rust
//! use genalg::error::Result;
//! use std::fs::File;
//! use std::io::Read;
//!
//! fn read_config(path: &str) -> Result<String> {
//!     let mut file = File::open(path)?; // io::Error automatically converts to GeneticError
//!     let mut contents = String::new();
//!     file.read_to_string(&mut contents)?; // io::Error automatically converts to GeneticError
//!     Ok(contents)
//! }
//! ```
//!
//! ## Comprehensive Error Handling Example
//!
//! Here's a more comprehensive example showing how to handle various error scenarios
//! in a genetic algorithm application:
//!
//! ```rust
//! use genalg::{
//!     error::{GeneticError, Result, ResultExt, OptionExt},
//!     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
//!     phenotype::Phenotype,
//!     breeding::OrdinaryStrategy,
//!     selection::ElitistSelection,
//! };
//! use std::fs::File;
//! use std::io::{self, Read};
//!
//! // Custom phenotype and challenge implementations omitted for brevity
//! # #[derive(Clone, Debug)]
//! # struct MyPhenotype { value: f64 }
//! # impl Phenotype for MyPhenotype {
//! #     fn crossover(&mut self, other: &Self) { self.value = (self.value + other.value) / 2.0; }
//! #     fn mutate(&mut self, _rng: &mut genalg::rng::RandomNumberGenerator) { }
//! # }
//! # #[derive(Clone)]
//! # struct MyChallenge { target: f64 }
//! # impl Challenge<MyPhenotype> for MyChallenge {
//! #     fn score(&self, phenotype: &MyPhenotype) -> f64 { 1.0 / (phenotype.value - self.target).abs().max(0.001) }
//! # }
//!
//! fn load_initial_phenotype(path: &str) -> Result<MyPhenotype> {
//!     // Handle IO errors with context
//!     let mut file = File::open(path)
//!         .context(format!("Failed to open initial phenotype file: {}", path))?;
//!     
//!     let mut contents = String::new();
//!     file.read_to_string(&mut contents)
//!         .context("Failed to read phenotype data")?;
//!     
//!     // Parse the value, handling potential format errors
//!     let value = contents.trim().parse::<f64>()
//!         .map_err(|e| GeneticError::Other(format!("Invalid phenotype value: {}", e)))?;
//!     
//!     // Validate the value
//!     if !value.is_finite() {
//!         return Err(GeneticError::InvalidNumericValue(
//!             "Initial phenotype value must be finite".to_string()
//!         ));
//!     }
//!     
//!     Ok(MyPhenotype { value })
//! }
//!
//! fn run_evolution(config_path: &str, phenotype_path: &str) -> Result<()> {
//!     // Load the initial phenotype, propagating any errors
//!     let starting_value = load_initial_phenotype(phenotype_path)?;
//!     
//!     // Create evolution components
//!     let options = EvolutionOptions::builder()
//!         .num_generations(100)
//!         .log_level(LogLevel::Info)
//!         .population_size(10)
//!         .num_offspring(50)
//!         .build();
//!     
//!     // Validate configuration
//!     if options.get_population_size() == 0 {
//!         return Err(GeneticError::Configuration(
//!             "Population size cannot be zero".to_string()
//!         ));
//!     }
//!     
//!     let challenge = MyChallenge { target: 42.0 };
//!     let strategy = OrdinaryStrategy::default();
//!     
//!     // Run the evolution, handling potential errors
//!     let selection_strategy = ElitistSelection::default();
//!     let launcher: EvolutionLauncher<
//!         MyPhenotype,
//!         OrdinaryStrategy,
//!         ElitistSelection,
//!         genalg::local_search::HillClimbing,
//!         MyChallenge,
//!         genalg::local_search::AllIndividualsStrategy
//!     > = EvolutionLauncher::new(strategy, selection_strategy, None, challenge);
//!     let result = launcher
//!         .configure(options, starting_value)
//!         .run()?;
//!     
//!     println!("Evolution successful! Best fitness: {}", result.score);
//!     Ok(())
//! }
//!
//! fn main() {
//!     match run_evolution("config.txt", "phenotype.txt") {
//!         Ok(_) => println!("Evolution completed successfully"),
//!         Err(e) => match e {
//!             GeneticError::Configuration(msg) => eprintln!("Configuration error: {}", msg),
//!             GeneticError::EmptyPopulation => eprintln!("Error: Empty population"),
//!             GeneticError::InvalidNumericValue(msg) => eprintln!("Numeric error: {}", msg),
//!             GeneticError::Io(io_err) => eprintln!("I/O error: {}", io_err),
//!             _ => eprintln!("Unexpected error: {}", e),
//!         }
//!     }
//! }
//! ```

use std::error::Error as StdError;
use std::fmt;
use thiserror::Error;

/// Represents errors that can occur in the genetic algorithm library.
///
/// This enum provides specific error variants for different failure scenarios
/// that may occur during the evolution process.
#[derive(Error, Debug)]
pub enum GeneticError {
    /// Error that occurs when a breeding operation fails.
    #[error("Breeding error: {0}")]
    Breeding(String),

    /// Error that occurs when a phenotype cannot be developed within constraints.
    #[error("Development error: {0}")]
    Development(String),

    /// Error that occurs when an evolution process fails.
    #[error("Evolution error: {0}")]
    Evolution(String),

    /// Error that occurs when an invalid configuration is provided.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Error that occurs when an empty population is encountered.
    #[error("Empty population error: Cannot operate on an empty population")]
    EmptyPopulation,

    /// Error that occurs when a fitness calculation fails.
    #[error("Fitness calculation error: {0}")]
    FitnessCalculation(String),

    /// Error that occurs when a selection operation fails.
    #[error("Selection error: {0}")]
    Selection(String),

    /// Error that occurs when a random number generation fails.
    #[error("Random generation error: {0}")]
    RandomGeneration(String),

    /// Error that occurs when a phenotype is outside of valid bounds.
    #[error("Bounds error: Phenotype is outside of valid bounds - {0}")]
    OutOfBounds(String),

    /// Error that occurs when a maximum number of attempts is reached.
    #[error("Maximum attempts reached: {0}")]
    MaxAttemptsReached(String),

    /// Error that occurs when NaN or infinity values are encountered.
    #[error("Invalid numeric value: {0}")]
    InvalidNumericValue(String),

    /// Error that occurs when an I/O operation fails.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A generic error with a custom message.
    #[error("{0}")]
    Other(String),
}

// Implement From for specific error types
// This allows automatic conversion from std::io::Error to GeneticError
// Additional From implementations can be added for other error types as needed

/// A specialized Result type for genetic algorithm operations.
///
/// This type is a convenience wrapper around `std::result::Result` with the error type
/// fixed to `GeneticError`.
///
/// ## Examples
///
/// ```rust
/// use genalg::error::{GeneticError, Result};
///
/// fn may_fail() -> Result<i32> {
///     // Some operation that might fail
///     Ok(42)
/// }
/// ```
pub type Result<T> = std::result::Result<T, GeneticError>;

/// Extension trait for Result to add context to errors.
///
/// This trait provides a convenient way to add context to errors when
/// converting from one error type to `GeneticError`.
///
/// ## Examples
///
/// ```rust
/// use genalg::error::ResultExt;
/// use std::fs::File;
///
/// fn read_file(path: &str) -> genalg::error::Result<()> {
///     File::open(path).context("Failed to open file")?;
///     Ok(())
/// }
/// ```
pub trait ResultExt<T, E> {
    /// Adds context to an error.
    ///
    /// This method converts the error to a `GeneticError` with the provided context.
    ///
    /// ## Arguments
    ///
    /// * `context` - A string providing context for the error.
    ///
    /// ## Returns
    ///
    /// A `Result<T, GeneticError>` with the original value or a contextualized error.
    fn context<C>(self, context: C) -> Result<T>
    where
        C: fmt::Display + Send + Sync + 'static;
}

impl<T, E> ResultExt<T, E> for std::result::Result<T, E>
where
    E: StdError + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> Result<T>
    where
        C: fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|e| GeneticError::Other(format!("{}: {}", context, e)))
    }
}

/// Extension trait for Option to convert to Result with a custom error message.
///
/// This trait provides a convenient way to convert an `Option` to a `Result`
/// with a custom error message.
///
/// ## Examples
///
/// ```rust
/// use genalg::error::{GeneticError, OptionExt};
///
/// fn find_best_candidate(candidates: &[i32]) -> genalg::error::Result<i32> {
///     candidates.iter().max().cloned().ok_or_else_genetic(||
///         GeneticError::EmptyPopulation
///     )
/// }
/// ```
pub trait OptionExt<T> {
    /// Converts an Option to a Result with a custom error message.
    ///
    /// This method converts an `Option<T>` to a `Result<T, GeneticError>` using
    /// a closure to generate the error.
    ///
    /// ## Arguments
    ///
    /// * `err_fn` - A closure that returns a `GeneticError`.
    ///
    /// ## Returns
    ///
    /// A `Result<T, GeneticError>` with the original value or the generated error.
    fn ok_or_else_genetic<F>(self, err_fn: F) -> Result<T>
    where
        F: FnOnce() -> GeneticError;
}

impl<T> OptionExt<T> for Option<T> {
    fn ok_or_else_genetic<F>(self, err_fn: F) -> Result<T>
    where
        F: FnOnce() -> GeneticError,
    {
        self.ok_or_else(err_fn)
    }
}

/// Utility function to convert a standard error to a GeneticError with context.
///
/// This function is useful when you want to convert a standard error to a `GeneticError`
/// with additional context.
///
/// ## Arguments
///
/// * `error` - The error to convert.
/// * `context` - A string providing context for the error.
///
/// ## Returns
///
/// A `GeneticError` with the context and error message.
///
/// ## Examples
///
/// ```rust
/// use genalg::error::to_genetic_error;
/// use std::io;
///
/// fn example() -> genalg::error::Result<()> {
///     let io_error = io::Error::new(io::ErrorKind::NotFound, "File not found");
///     Err(to_genetic_error(io_error, "Failed to read configuration"))
/// }
/// ```
pub fn to_genetic_error<E: StdError>(error: E, context: &str) -> GeneticError {
    GeneticError::Other(format!("{}: {}", context, error))
}
