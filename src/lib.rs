//! # GenAlg
//! 
//! A flexible, high-performance genetic algorithm library written in Rust.
//! 
//! ## Overview
//! 
//! GenAlg is a modern, thread-safe genetic algorithm framework designed for flexibility, 
//! performance, and ease of use. It provides a robust foundation for implementing 
//! evolutionary algorithms to solve optimization problems across various domains.
//! 
//! ## Key Features
//! 
//! - **Thread-safe**: Designed for parallel processing with `Send` and `Sync` traits
//! - **High Performance**: Optimized for speed with thread-local random number generation
//! - **Flexible**: Adaptable to a wide range of optimization problems
//! - **Extensible**: Easy to implement custom phenotypes, fitness functions, and breeding strategies
//! - **Parallel Processing**: Automatic parallelization for large populations using Rayon
//! 
//! ## Core Components
//! 
//! ### Phenotype
//! 
//! The [`Phenotype`] trait defines the interface for types that represent individuals 
//! in an evolutionary algorithm. It provides methods for crossover and mutation.
//! 
//! ```rust
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//! 
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     value: f64,
//! }
//! 
//! impl Phenotype for MyPhenotype {
//!     fn crossover(&mut self, other: &Self) {
//!         // Combine genetic material with another individual
//!         self.value = (self.value + other.value) / 2.0;
//!     }
//! 
//!     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
//!         // Introduce random changes
//!         let values = rng.fetch_uniform(-0.1, 0.1, 1);
//!         let delta = values.front().unwrap();
//!         self.value += *delta as f64;
//!     }
//!     
//!     // Optional: Override for better performance in parallel contexts
//!     fn mutate_thread_local(&mut self) {
//!         // Custom implementation using thread-local RNG
//!         use genalg::rng::ThreadLocalRng;
//!         let delta = ThreadLocalRng::gen_range(-0.1..0.1);
//!         self.value += delta;
//!     }
//! }
//! ```
//! 
//! ### Challenge
//! 
//! The [`Challenge`] trait defines how to evaluate the fitness of phenotypes:
//! 
//! ```rust
//! use genalg::evolution::Challenge;
//! use genalg::phenotype::Phenotype;
//! 
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     value: f64,
//! }
//! 
//! // Implementation of Phenotype trait omitted for brevity
//! # impl Phenotype for MyPhenotype {
//! #     fn crossover(&mut self, other: &Self) {}
//! #     fn mutate(&mut self, rng: &mut genalg::rng::RandomNumberGenerator) {}
//! # }
//! 
//! struct MyChallenge {
//!     target: f64,
//! }
//! 
//! impl Challenge<MyPhenotype> for MyChallenge {
//!     fn score(&self, phenotype: &MyPhenotype) -> f64 {
//!         // Calculate and return fitness score (higher is better)
//!         1.0 / (phenotype.value - self.target).abs().max(0.001)
//!     }
//! }
//! ```
//! 
//! ### Breeding Strategies
//! 
//! GenAlg provides two built-in breeding strategies:
//! 
//! 1. [`OrdinaryStrategy`]: A basic breeding strategy where the first parent is considered 
//!    the winner of the previous generation.
//! 
//! 2. [`BoundedBreedStrategy`]: Similar to `OrdinaryStrategy` but imposes bounds on 
//!    phenotypes during evolution using the [`Magnitude`] trait.
//! 
//! ### Evolution Launcher
//! 
//! The [`EvolutionLauncher`] manages the evolution process using a specified breeding 
//! strategy and challenge:
//! 
//! ```rust
//! use genalg::{
//!     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
//!     phenotype::Phenotype,
//!     rng::RandomNumberGenerator,
//!     strategy::OrdinaryStrategy,
//! };
//! 
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     value: f64,
//! }
//! 
//! // Implementation of Phenotype trait omitted for brevity
//! # impl Phenotype for MyPhenotype {
//! #     fn crossover(&mut self, other: &Self) {}
//! #     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {}
//! # }
//! 
//! struct MyChallenge {
//!     target: f64,
//! }
//! 
//! // Implementation of Challenge trait omitted for brevity
//! # impl Challenge<MyPhenotype> for MyChallenge {
//! #     fn score(&self, phenotype: &MyPhenotype) -> f64 { 0.0 }
//! # }
//! 
//! // Initialize components
//! let starting_value = MyPhenotype { value: 0.0 };
//! let options = EvolutionOptions::builder()
//!     .num_generations(100)
//!     .log_level(LogLevel::Minimal)
//!     .population_size(10)
//!     .num_offspring(50)
//!     .build();
//! let challenge = MyChallenge { target: 42.0 };
//! let strategy = OrdinaryStrategy::default();
//! 
//! // Create and run the evolution
//! let launcher = EvolutionLauncher::new(strategy, challenge);
//! let result = launcher
//!     .configure(options, starting_value)
//!     .with_seed(42)  // Optional: Set a specific seed
//!     .run()
//!     .unwrap();
//! 
//! println!("Best solution: {:?}, Fitness: {}", result.pheno, result.score);
//! ```
//! 
//! ### Thread-Local Random Number Generation
//! 
//! For optimal performance in parallel contexts, GenAlg provides thread-local random 
//! number generation through the [`ThreadLocalRng`] struct:
//! 
//! ```rust
//! use genalg::rng::ThreadLocalRng;
//! 
//! // Generate a random number in a range
//! let value = ThreadLocalRng::gen_range(0.0..1.0);
//! 
//! // Generate multiple random numbers
//! let numbers = ThreadLocalRng::fetch_uniform(0.0, 1.0, 5);
//! ```
//! 
//! ## Parallel Processing
//! 
//! GenAlg automatically uses parallel processing for fitness evaluation and breeding 
//! when the population size exceeds the parallel threshold. Configure this in your 
//! [`EvolutionOptions`]:
//! 
//! ```rust
//! use genalg::evolution::options::{EvolutionOptions, LogLevel};
//! 
//! // Using the builder pattern
//! let options = EvolutionOptions::builder()
//!     .num_generations(100)
//!     .log_level(LogLevel::Minimal)
//!     .population_size(10)
//!     .num_offspring(50)
//!     .parallel_threshold(500) // Use parallel processing when population >= 500
//!     .build();
//! 
//! // Or using setter methods
//! let mut options = EvolutionOptions::default();
//! options.set_parallel_threshold(500);
//! ```
//! 
//! ## Error Handling
//! 
//! GenAlg provides a comprehensive error handling system through the [`error`] module:
//! 
//! ```rust
//! use genalg::error::{GeneticError, Result, ResultExt, OptionExt};
//! 
//! fn my_function() -> Result<()> {
//!     // Return specific errors
//!     if false {
//!         return Err(GeneticError::Configuration("Invalid parameter".to_string()));
//!     }
//!     
//!     // Convert Option to Result with custom error
//!     let candidates = vec![1, 2, 3];
//!     let best = candidates.iter().max().ok_or_else_genetic(|| 
//!         GeneticError::EmptyPopulation
//!     )?;
//!     
//!     Ok(())
//! }
//! ```
//! 
//! ## Modules
//! 
//! - [`error`]: Error types and utilities
//! - [`evolution`]: Evolution process management
//! - [`phenotype`]: Phenotype trait definition
//! - [`rng`]: Random number generation utilities
//! - [`strategy`]: Breeding strategy implementations
//! 
//! [`Phenotype`]: phenotype::Phenotype
//! [`Challenge`]: evolution::Challenge
//! [`OrdinaryStrategy`]: strategy::OrdinaryStrategy
//! [`BoundedBreedStrategy`]: strategy::bounded::BoundedBreedStrategy
//! [`Magnitude`]: strategy::bounded::Magnitude
//! [`EvolutionLauncher`]: evolution::EvolutionLauncher
//! [`EvolutionOptions`]: evolution::options::EvolutionOptions
//! [`ThreadLocalRng`]: rng::ThreadLocalRng

pub mod error;
pub mod evolution;
pub mod phenotype;
pub mod rng;
pub mod strategy;

// Re-export commonly used types for convenience
pub use error::Result;
pub use evolution::{Challenge, EvolutionLauncher, EvolutionOptions, EvolutionResult};
pub use phenotype::Phenotype;
pub use strategy::{bounded::BoundedBreedStrategy, ordinary::OrdinaryStrategy, BreedStrategy};
