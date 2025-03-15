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
//! - **Local Search**: Integrated local search algorithms to refine solutions
//! - **Constraint Handling**: Support for combinatorial optimization with constraint management
//! - **Fitness Caching**: Efficient caching for expensive fitness evaluations
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
//! #[derive(Clone)]
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
//! GenAlg provides several built-in breeding strategies:
//!
//! 1. [`OrdinaryStrategy`]: A basic breeding strategy where the first parent is considered
//!    the winner of the previous generation.
//!
//! 2. [`BoundedBreedStrategy`]: Similar to `OrdinaryStrategy` but imposes bounds on
//!    phenotypes during evolution using the [`Magnitude`] trait.
//!
//! 3. [`CombinatorialBreedStrategy`]: Specialized for combinatorial optimization problems,
//!    supporting constraint handling and penalty-based fitness adjustment.
//!
//! ### Selection Strategies
//!
//! GenAlg provides several built-in selection strategies for choosing parents based on their fitness:
//!
//! 1. [`ElitistSelection`]: Selects the best individuals based on fitness scores.
//!
//! 2. [`TournamentSelection`]: Selects individuals through tournament selection.
//!
//! 3. [`RouletteWheelSelection`]: Selects individuals with probability proportional to fitness.
//!
//! 4. [`RankBasedSelection`]: Selects individuals based on their rank in the population.
//!
//! ### Local Search Algorithms
//!
//! GenAlg integrates local search algorithms to refine solutions during the evolutionary process:
//!
//! 1. [`HillClimbing`]: A simple hill climbing algorithm that iteratively moves to better neighboring solutions.
//!
//! 2. [`SimulatedAnnealing`]: A probabilistic algorithm that allows moves to worse solutions with decreasing probability.
//!
//! 3. [`TabuSearch`]: A metaheuristic that maintains a list of recently visited solutions to avoid cycling.
//!
//! 4. [`HybridLocalSearch`]: Combines multiple local search algorithms for more effective refinement.
//!
//! Local search can be applied to selected individuals using various strategies:
//!
//! 1. [`AllIndividualsStrategy`]: Applies local search to all individuals in the population.
//!
//! 2. [`TopNStrategy`]: Applies local search to the top N individuals based on fitness.
//!
//! 3. [`TopPercentStrategy`]: Applies local search to a percentage of the top individuals.
//!
//! 4. [`ProbabilisticStrategy`]: Applies local search to individuals with a certain probability.
//!
//! ### Constraint Handling
//!
//! For combinatorial optimization problems, GenAlg provides constraint handling capabilities:
//!
//! 1. [`Constraint`]: Trait for defining constraints on phenotypes.
//!
//! 2. [`ConstraintManager`]: Manages multiple constraints and provides methods for checking and repairing solutions.
//!
//! 3. [`PenaltyAdjustedChallenge`]: Wraps a challenge to adjust fitness scores based on constraint violations.
//!
//! 4. Built-in constraints for combinatorial problems like [`UniqueElementsConstraint`], [`CompleteAssignmentConstraint`], etc.
//!
//! ### Evolution Launcher
//!
//! The [`EvolutionLauncher`] manages the evolution process using a specified breeding
//! strategy, selection strategy, local search manager, and challenge. It now requires four parameters:
//! a breeding strategy, a selection strategy, an optional local search manager, and a challenge.
//!
//! ```rust
//! use genalg::{
//!     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
//!     phenotype::Phenotype,
//!     rng::RandomNumberGenerator,
//!     strategy::OrdinaryStrategy,
//!     selection::ElitistSelection,
//!     local_search::{HillClimbing, AllIndividualsStrategy},
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
//! #[derive(Clone)]
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
//!
//! // Create components for evolution
//! let breed_strategy = OrdinaryStrategy::default();
//! let selection_strategy = ElitistSelection::default();
//! let challenge = MyChallenge { target: 42.0 };
//! let options = EvolutionOptions::default();
//! let starting_value = MyPhenotype { value: 0.0 };
//!
//! // Create launcher with breeding, selection, and challenge
//! // Note: The third parameter (local_search_manager) can be None if local search is not needed
//! let launcher: EvolutionLauncher<
//!     MyPhenotype,
//!     OrdinaryStrategy,
//!     ElitistSelection,
//!     HillClimbing,
//!     MyChallenge,
//!     AllIndividualsStrategy
//! > = EvolutionLauncher::new(
//!     breed_strategy, 
//!     selection_strategy, 
//!     None, // No local search
//!     challenge
//! );
//!
//! // Configure and run the evolution
//! let result = launcher
//!     .configure(options, starting_value)
//!     .with_seed(42)  // Optional: Set a specific seed
//!     .run();
//! ```
//!
//! If you want to use local search, you can create a local search manager and pass it to the launcher:
//!
//! ```rust
//! # use genalg::{
//! #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
//! #     phenotype::Phenotype,
//! #     rng::RandomNumberGenerator,
//! #     strategy::OrdinaryStrategy,
//! #     selection::ElitistSelection,
//! #     local_search::{HillClimbing, AllIndividualsStrategy, LocalSearchManager},
//! # };
//! # 
//! # #[derive(Clone, Debug)]
//! # struct MyPhenotype {
//! #     value: f64,
//! # }
//! # 
//! # impl Phenotype for MyPhenotype {
//! #     fn crossover(&mut self, other: &Self) {}
//! #     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {}
//! # }
//! # 
//! # #[derive(Clone)]
//! # struct MyChallenge {
//! #     target: f64,
//! # }
//! # 
//! # impl Challenge<MyPhenotype> for MyChallenge {
//! #     fn score(&self, phenotype: &MyPhenotype) -> f64 {
//! #         1.0 / (phenotype.value - self.target).abs().max(0.001)
//! #     }
//! # }
//! # 
//! # let breed_strategy = OrdinaryStrategy::default();
//! # let selection_strategy = ElitistSelection::default();
//! # let challenge = MyChallenge { target: 42.0 };
//! # let options = EvolutionOptions::default();
//! # let starting_value = MyPhenotype { value: 0.0 };
//! 
//! // Create a local search manager
//! let hill_climbing = HillClimbing::new(10).unwrap();
//! let application_strategy = AllIndividualsStrategy::new();
//! let local_search_manager = Some(
//!     LocalSearchManager::new(hill_climbing, application_strategy)
//! );
//!
//! // Create launcher with breeding, selection, local search, and challenge
//! let launcher: EvolutionLauncher<
//!     MyPhenotype,
//!     OrdinaryStrategy,
//!     ElitistSelection,
//!     HillClimbing,
//!     MyChallenge,
//!     AllIndividualsStrategy
//! > = EvolutionLauncher::new(
//!     breed_strategy, 
//!     selection_strategy, 
//!     local_search_manager, 
//!     challenge
//! );
//!
//! // Configure and run the evolution with local search
//! let result = launcher
//!     .configure(options, starting_value)
//!     .with_seed(42)
//!     .with_local_search()  // Enable local search
//!     .run();
//! ```
//!
//! You can also use the builder pattern to create an `EvolutionLauncher`:
//!
//! ```rust
//! # use genalg::{
//! #     evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
//! #     phenotype::Phenotype,
//! #     rng::RandomNumberGenerator,
//! #     strategy::OrdinaryStrategy,
//! #     selection::ElitistSelection,
//! #     local_search::{HillClimbing, AllIndividualsStrategy},
//! #     error::Result,
//! # };
//! # 
//! # #[derive(Clone, Debug)]
//! # struct MyPhenotype {
//! #     value: f64,
//! # }
//! # 
//! # impl Phenotype for MyPhenotype {
//! #     fn crossover(&mut self, other: &Self) {}
//! #     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {}
//! # }
//! # 
//! # #[derive(Clone)]
//! # struct MyChallenge {
//! #     target: f64,
//! # }
//! # 
//! # impl Challenge<MyPhenotype> for MyChallenge {
//! #     fn score(&self, phenotype: &MyPhenotype) -> f64 {
//! #         1.0 / (phenotype.value - self.target).abs().max(0.001)
//! #     }
//! # }
//! 
//! fn create_launcher() -> Result<EvolutionLauncher<
//!     MyPhenotype,
//!     OrdinaryStrategy,
//!     ElitistSelection,
//!     HillClimbing,
//!     MyChallenge,
//!     AllIndividualsStrategy
//! >> {
//!     let breed_strategy = OrdinaryStrategy::default();
//!     let selection_strategy = ElitistSelection::default();
//!     let challenge = MyChallenge { target: 42.0 };
//!     
//!     // Create a local search strategy and application strategy
//!     let hill_climbing = HillClimbing::new(10)?;
//!     let application_strategy = AllIndividualsStrategy::new();
//!     
//!     // Use the builder pattern
//!     EvolutionLauncher::builder()
//!         .with_breed_strategy(breed_strategy)
//!         .with_selection_strategy(selection_strategy)
//!         .with_local_search_manager(hill_climbing, application_strategy)
//!         .with_challenge(challenge)
//!         .build()
//! }
//! ```
//!
//! ### Fitness Caching
//!
//! For expensive fitness evaluations, GenAlg provides caching mechanisms:
//!
//! ```rust
//! use genalg::{
//!     caching::{CacheKey, CachedChallenge},
//!     evolution::Challenge,
//!     phenotype::Phenotype,
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
//! #     fn mutate(&mut self, rng: &mut genalg::rng::RandomNumberGenerator) {}
//! # }
//!
//! // Implement CacheKey to enable caching
//! impl CacheKey for MyPhenotype {
//!     // Use i32 as the key type since it implements Hash and Eq
//!     type Key = i32;
//!     
//!     fn cache_key(&self) -> Self::Key {
//!         // Convert the f64 value to an i32 for caching
//!         // In a real implementation, you might want to use a more sophisticated
//!         // conversion that preserves more precision
//!         self.value as i32
//!     }
//! }
//!
//! #[derive(Clone)]
//! struct MyChallenge {
//!     target: f64,
//! }
//!
//! impl Challenge<MyPhenotype> for MyChallenge {
//!     fn score(&self, phenotype: &MyPhenotype) -> f64 {
//!         // Expensive calculation
//!         1.0 / (phenotype.value - self.target).abs().max(0.001)
//!     }
//! }
//!
//! // Create a cached version of the challenge
//! let challenge = MyChallenge { target: 42.0 };
//! let cached_challenge = CachedChallenge::new(challenge);
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
//!     .log_level(LogLevel::Info)
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
//! ### Performance Characteristics
//!
//! GenAlg is optimized for parallel processing with larger populations:
//!
//! - **Parallel operations** show significant performance improvements for larger populations
//! - **Thread-local RNG** eliminates mutex contention in parallel contexts
//! - **Automatic parallelization** occurs when population size exceeds the parallel threshold
//! - **Optimal threshold** depends on your specific hardware and problem complexity
//!
//! For best performance:
//! - Use `mutate_thread_local()` in your phenotype implementations
//! - Set an appropriate parallel threshold based on your hardware
//! - Consider using the `OrdinaryStrategy` for very large populations
//! - Use fitness caching for expensive evaluations
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
//! - [`selection`]: Selection strategy implementations
//! - [`local_search`]: Local search algorithms and application strategies
//! - [`constraints`]: Constraint handling for combinatorial optimization
//! - [`caching`]: Fitness caching for expensive evaluations
//!
//! [`Phenotype`]: phenotype::Phenotype
//! [`Challenge`]: evolution::Challenge
//! [`OrdinaryStrategy`]: strategy::ordinary::OrdinaryStrategy
//! [`BoundedBreedStrategy`]: strategy::bounded::BoundedBreedStrategy
//! [`CombinatorialBreedStrategy`]: strategy::combinatorial::CombinatorialBreedStrategy
//! [`Magnitude`]: strategy::bounded::Magnitude
//! [`EvolutionLauncher`]: evolution::EvolutionLauncher
//! [`EvolutionOptions`]: evolution::options::EvolutionOptions
//! [`ThreadLocalRng`]: rng::ThreadLocalRng
//! [`ElitistSelection`]: selection::ElitistSelection
//! [`TournamentSelection`]: selection::TournamentSelection
//! [`RouletteWheelSelection`]: selection::RouletteWheelSelection
//! [`RankBasedSelection`]: selection::RankBasedSelection
//! [`HillClimbing`]: local_search::HillClimbing
//! [`SimulatedAnnealing`]: local_search::SimulatedAnnealing
//! [`TabuSearch`]: local_search::TabuSearch
//! [`HybridLocalSearch`]: local_search::HybridLocalSearch
//! [`AllIndividualsStrategy`]: local_search::application::AllIndividualsStrategy
//! [`TopNStrategy`]: local_search::application::TopNStrategy
//! [`TopPercentStrategy`]: local_search::application::TopPercentStrategy
//! [`ProbabilisticStrategy`]: local_search::application::ProbabilisticStrategy
//! [`Constraint`]: constraints::Constraint
//! [`ConstraintManager`]: constraints::ConstraintManager
//! [`PenaltyAdjustedChallenge`]: constraints::PenaltyAdjustedChallenge
//! [`UniqueElementsConstraint`]: constraints::combinatorial::UniqueElementsConstraint
//! [`CompleteAssignmentConstraint`]: constraints::combinatorial::CompleteAssignmentConstraint

pub mod caching;
pub mod constraints;
pub mod error;
pub mod evolution;
pub mod local_search;
pub mod phenotype;
pub mod rng;
pub mod selection;
pub mod strategy;

// Re-export commonly used types for convenience
pub use caching::{CacheKey, CachedChallenge, ThreadLocalCachedChallenge};
pub use constraints::{Constraint, ConstraintManager, ConstraintViolation};
pub use error::{GeneticError, OptionExt, Result, ResultExt};
pub use evolution::{Challenge, EvolutionLauncher, EvolutionOptions, EvolutionResult, LogLevel};
pub use local_search::{HillClimbing, LocalSearch, SimulatedAnnealing};
pub use phenotype::Phenotype;
pub use rng::ThreadLocalRng;
pub use selection::{
    ElitistSelection, RankBasedSelection, RouletteWheelSelection, SelectionStrategy,
    TournamentSelection,
};
pub use strategy::combinatorial::{CombinatorialBreedConfig, CombinatorialBreedStrategy};
pub use strategy::{
    bounded::{BoundedBreedConfig, BoundedBreedStrategy, Magnitude},
    ordinary::OrdinaryStrategy,
    BreedStrategy,
};
