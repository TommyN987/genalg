pub mod evolution;
pub mod phenotype;
pub mod rng;
pub mod strategy;
pub mod error;

// Re-export commonly used types for convenience
pub use error::{GeneticError, Result, ResultExt, OptionExt};
