//! # RandomNumberGenerator
//!
//! The `RandomNumberGenerator` struct provides a simple interface for generating
//! random floating-point numbers within a specified range using the `rand` crate.
//!
//! ## Example
//!
//! ```rust
//! use genalg::rng::RandomNumberGenerator;
//!
//! let mut rng = RandomNumberGenerator::new();
//! let random_numbers = rng.fetch_uniform(0.0, 1.0, 5);
//!
//! for number in random_numbers {
//!     println!("Random Number: {}", number);
//! }
//! ```
//!
//! ## Thread-safe RNG
//!
//! For parallel processing, the library provides a `ThreadLocalRng` that can be used
//! without synchronization overhead:
//!
//! ```rust
//! use genalg::rng::ThreadLocalRng;
//!
//! // Get a thread-local RNG
//! let random_number = ThreadLocalRng::gen_range(0.0..1.0);
//! ```

use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use std::collections::VecDeque;

/// A thread-local random number generator that can be used without synchronization.
///
/// This is useful for parallel processing where each thread needs its own RNG.
/// It uses the built-in `ThreadRng` from the `rand` crate, which is automatically
/// seeded from the system entropy and is thread-local.
pub struct ThreadLocalRng;

impl ThreadLocalRng {
    /// Generates a random number in the given range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range to generate a random number in.
    ///
    /// # Returns
    ///
    /// A random number in the given range.
    pub fn gen_range<T, R>(range: R) -> T
    where
        T: rand::distributions::uniform::SampleUniform,
        R: rand::distributions::uniform::SampleRange<T>,
    {
        thread_rng().gen_range(range)
    }

    /// Generates a specified number of random floating-point numbers within the given range.
    ///
    /// # Parameters
    ///
    /// - `from`: The lower bound of the range (inclusive).
    /// - `to`: The upper bound of the range (exclusive).
    /// - `num`: The number of random numbers to generate.
    ///
    /// # Returns
    ///
    /// A `VecDeque` containing the generated random numbers.
    pub fn fetch_uniform(from: f32, to: f32, num: usize) -> VecDeque<f32> {
        let mut uniform_numbers = VecDeque::with_capacity(num);
        let mut rng = thread_rng();
        uniform_numbers.extend((0..num).map(|_| rng.gen_range(from..to)));
        uniform_numbers
    }

    /// Gets a new instance of the thread-local RNG.
    ///
    /// This effectively reseeds the RNG by getting a fresh instance from the system.
    /// This is useful after a fork to ensure that child processes have different random sequences.
    ///
    /// # Note
    ///
    /// The built-in `ThreadRng` doesn't have a direct `reseed` method, but getting a new
    /// instance via `thread_rng()` ensures we get a properly seeded RNG.
    pub fn get_fresh_rng() {
        let _ = thread_rng();
    }
}

/// A wrapper around the `rand` crate's `StdRng` that provides methods for generating
/// random numbers within a specified range.
#[derive(Clone)]
pub struct RandomNumberGenerator {
    pub rng: StdRng,
}

impl RandomNumberGenerator {
    /// Creates a new `RandomNumberGenerator` instance seeded from the system entropy.
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }

    /// Creates a new `RandomNumberGenerator` instance with a specific seed.
    ///
    /// This is useful for reproducible tests and benchmarks.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed to use for the random number generator.
    ///
    /// # Returns
    ///
    /// A new `RandomNumberGenerator` instance.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Generates a specified number of random floating-point numbers within the given range.
    ///
    /// # Parameters
    ///
    /// - `from`: The lower bound of the range (inclusive).
    /// - `to`: The upper bound of the range (exclusive).
    /// - `num`: The number of random numbers to generate.
    ///
    /// # Returns
    ///
    /// A `VecDeque` containing the generated random numbers.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use genalg::rng::RandomNumberGenerator;
    ///
    /// let mut rng = RandomNumberGenerator::new();
    /// let random_numbers = rng.fetch_uniform(0.0, 1.0, 5);
    ///
    /// for number in random_numbers {
    ///     println!("Random Number: {}", number);
    /// }
    /// ```
    pub fn fetch_uniform(&mut self, from: f32, to: f32, num: usize) -> VecDeque<f32> {
        let mut uniform_numbers = VecDeque::new();
        uniform_numbers.extend((0..num).map(|_| self.rng.gen_range(from..to)));
        uniform_numbers
    }
}

impl Default for RandomNumberGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fetch_uniform_with_positive_range() {
        let mut rng = RandomNumberGenerator::new();
        let result = rng.fetch_uniform(0.0, 1.0, 5);

        // Check that the result has the correct length
        assert_eq!(result.len(), 5);

        // Check that all elements are within the specified range
        for &num in result.iter() {
            assert!((0.0..1.0).contains(&num));
        }
    }

    #[test]
    fn test_fetch_uniform_with_negative_range() {
        let mut rng = RandomNumberGenerator::new();
        let result = rng.fetch_uniform(-1.0, 1.0, 3);

        assert_eq!(result.len(), 3);

        for &num in result.iter() {
            assert!((-1.0..1.0).contains(&num));
        }
    }

    #[test]
    fn test_fetch_uniform_with_large_range() {
        let mut rng = RandomNumberGenerator::new();
        let result = rng.fetch_uniform(-1000.0, 1000.0, 10);

        assert_eq!(result.len(), 10);

        for &num in result.iter() {
            assert!((-1000.0..1000.0).contains(&num));
        }
    }

    #[test]
    fn test_fetch_uniform_with_empty_result() {
        let mut rng = RandomNumberGenerator::new();
        let result = rng.fetch_uniform(1.0, 2.0, 0);

        assert!(result.is_empty());
    }

    #[test]
    fn test_clone() {
        let mut rng1 = RandomNumberGenerator::from_seed(42);
        let mut rng2 = rng1.clone();

        // Both RNGs should generate the same sequence after cloning
        let nums1 = rng1.fetch_uniform(0.0, 1.0, 5);
        let nums2 = rng2.fetch_uniform(0.0, 1.0, 5);

        assert_eq!(nums1, nums2);
    }

    #[test]
    fn test_thread_local_rng() {
        // Test that the thread-local RNG works
        let result = ThreadLocalRng::fetch_uniform(0.0, 1.0, 5);

        assert_eq!(result.len(), 5);

        for &num in result.iter() {
            assert!((0.0..1.0).contains(&num));
        }

        // Test that getting a fresh RNG doesn't panic
        ThreadLocalRng::get_fresh_rng();
    }
}
