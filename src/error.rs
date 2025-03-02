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
