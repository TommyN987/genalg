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
//! #[derive(Copy, Clone)]
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
//! ## Implementing the Trait
//!
//! To use the `Phenotype` trait, implement it for your custom phenotype type.
//!
//! ```rust
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//!
//! #[derive(Copy, Clone)]
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
//! }
//! ```

use crate::rng::RandomNumberGenerator;

pub trait Phenotype
where
    Self: Copy + Clone,
{
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
    /// #[derive(Copy, Clone)]
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
    /// #[derive(Copy, Clone)]
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
}
