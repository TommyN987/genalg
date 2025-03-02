//! # Phenotype Trait
//!
//! The `Phenotype` trait defines the interface for types that represent individuals
//! in an evolutionary algorithm. It provides methods for crossover and mutation.
//!
//! ## Example
//!
//! ```rust
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//!
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     // ... fields and methods for your specific phenotype ...
//! }
//!
//! impl Phenotype for MyPhenotype {
//!     fn crossover(&mut self, other: &Self) {
//!         // Implementation of crossover for MyPhenotype
//!         // ...
//!     }
//!
//!     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
//!         // Implementation of mutation for MyPhenotype
//!         // ...
//!     }
//!
//!     // The mutate_thread_local method has a default implementation
//!     // that calls mutate with a thread-local RNG
//! }
//! ```
//!
//! ## Trait
//!
//! ### `Phenotype`
//!
//! The `Phenotype` trait defines methods for crossover and mutation.
//!
//! ## Methods
//!
//! ### `crossover(&self, other: &Self)`
//!
//! Performs crossover with another individual of the same type.
//!
//! ### `mutate(&mut self, rng: &mut RandomNumberGenerator)`
//!
//! Performs mutation on the individual using the provided random number generator.
//!
//! ### `mutate_thread_local(&mut self)`
//!
//! Performs mutation on the individual using a thread-local random number generator.
//! This method has a default implementation that calls `mutate` with a thread-local RNG.
//!
//! ## Implementing the Trait
//!
//! To use the `Phenotype` trait, implement it for your custom phenotype type.
//!
//! ```rust
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//!
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     // ... fields and methods for your specific phenotype ...
//! }
//!
//! impl Phenotype for MyPhenotype {
//!     fn crossover(&mut self, other: &Self) {
//!         // Implementation of crossover for MyPhenotype
//!         // ...
//!     }
//!
//!     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
//!         // Implementation of mutation for MyPhenotype
//!         // ...
//!     }
//!
//!     // The mutate_thread_local method has a default implementation
//!     // that calls mutate with a thread-local RNG, but you can override it
//!     // for better performance if needed
//! }
//! ```

use std::fmt::Debug;

use crate::rng::RandomNumberGenerator;

/// Trait for types that represent individuals in an evolutionary algorithm.
///
/// This trait defines the core operations that must be implemented by any type
/// that represents an individual in the genetic algorithm. It requires methods
/// for crossover (combining genetic material with another individual) and
/// mutation (introducing random changes).
///
/// Types implementing this trait must also implement `Clone`, `Debug`, `Send`, and `Sync`
/// to enable parallel processing and debugging.
pub trait Phenotype: Clone + Debug + Send + Sync {
    /// Performs crossover with another individual of the same type.
    ///
    /// The `crossover` method is responsible for combining the genetic material of the current
    /// individual (`self`) with another individual (`other`). This process is a fundamental
    /// operation in evolutionary algorithms and is used to create new individuals by exchanging
    /// genetic information.
    ///
    /// ## Parameters
    ///
    /// - `self`: A reference to the current individual.
    /// - `other`: A reference to the other individual for crossover.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use genalg::phenotype::Phenotype;
    /// use genalg::rng::RandomNumberGenerator;
    ///
    /// #[derive(Clone, Debug)]
    /// struct MyPhenotype {
    ///     // ... fields for your specific phenotype ...
    /// }
    ///
    /// impl Phenotype for MyPhenotype {
    ///     fn crossover(&mut self, other: &Self) {
    ///         // Implementation of crossover for MyPhenotype
    ///         // ...
    ///     }
    ///
    ///     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
    ///         // Implementation of mutation for MyPhenotype
    ///         // ...
    ///     }
    /// }
    /// ```
    fn crossover(&mut self, other: &Self);

    /// Performs mutation on the individual using the provided random number generator.
    ///
    /// The `mutate` method introduces random changes to the genetic material of the individual.
    /// Mutation is essential in maintaining diversity within the population and exploring new
    /// regions of the solution space.
    ///
    /// ## Parameters
    ///
    /// - `self`: A mutable reference to the current individual.
    /// - `rng`: A mutable reference to the random number generator used for generating
    ///   random values during mutation.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use genalg::phenotype::Phenotype;
    /// use genalg::rng::RandomNumberGenerator;
    ///
    /// #[derive(Clone, Debug)]
    /// struct MyPhenotype {
    ///     // ... fields and methods for your specific phenotype ...
    /// }
    ///
    /// impl Phenotype for MyPhenotype {
    ///     fn crossover(&mut self, other: &Self) {
    ///         // Implementation of crossover for MyPhenotype
    ///         // ...
    ///     }
    ///
    ///     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
    ///         // Implementation of mutation for MyPhenotype
    ///         // ...
    ///     }
    /// }
    /// ```
    fn mutate(&mut self, rng: &mut RandomNumberGenerator);

    /// Performs mutation on the individual using a thread-local random number generator.
    ///
    /// This method is particularly useful in parallel contexts where each thread needs
    /// its own random number generator. It uses a thread-local RNG to avoid synchronization
    /// overhead.
    ///
    /// By default, this method uses the `ThreadLocalRng` to generate random numbers and
    /// calls the regular `mutate` method with a temporary RNG. You can override this method
    /// for better performance if your mutation operation can be optimized for thread-local use.
    ///
    /// ## Parameters
    ///
    /// - `self`: A mutable reference to the current individual.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use genalg::phenotype::Phenotype;
    /// use genalg::rng::{RandomNumberGenerator, ThreadLocalRng};
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
    ///         let result = rng.fetch_uniform(-0.1, 0.1, 1);
    ///         let delta = result.front().unwrap();
    ///         self.value += *delta as f64;
    ///     }
    ///
    ///     // Optional: Override the default implementation for better performance
    ///     fn mutate_thread_local(&mut self) {
    ///         let delta = ThreadLocalRng::gen_range(-0.1..0.1);
    ///         self.value += delta;
    ///     }
    /// }
    /// ```
    fn mutate_thread_local(&mut self) {
        // Default implementation uses ThreadLocalRng to generate random numbers
        // and calls the regular mutate method with a temporary RNG
        let mut temp_rng = RandomNumberGenerator::new();
        self.mutate(&mut temp_rng);
    }
}
