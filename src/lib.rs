pub mod error;
pub mod evolution;
pub mod phenotype;
pub mod rng;
pub mod strategy;

// Re-export commonly used types for convenience
pub use error::{GeneticError, OptionExt, Result, ResultExt};
