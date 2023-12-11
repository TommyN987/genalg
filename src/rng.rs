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
//!     let mut rng = RandomNumberGenerator::new();
//!     let random_numbers = rng.fetch_uniform(0.0, 1.0, 5);
//!
//!     for number in random_numbers {
//!         println!("Random Number: {}", number);
//!     }
//! ```

use rand::{rngs::ThreadRng, Rng};
use std::collections::VecDeque;
pub struct RandomNumberGenerator {
    pub rng: ThreadRng,
}

impl RandomNumberGenerator {
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
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

mod tests {

    #[test]
    fn test_fetch_uniform_with_positive_range() {
        let mut rng = crate::RandomNumberGenerator::new();
        let result = rng.fetch_uniform(0.0, 1.0, 5);

        // Check that the result has the correct length
        assert_eq!(result.len(), 5);

        // Check that all elements are within the specified range
        for &num in result.iter() {
            assert!(num >= 0.0 && num < 1.0);
        }
    }

    #[test]
    fn test_fetch_uniform_with_negative_range() {
        let mut rng = crate::RandomNumberGenerator::new();
        let result = rng.fetch_uniform(-1.0, 1.0, 3);

        assert_eq!(result.len(), 3);

        for &num in result.iter() {
            assert!(num >= -1.0 && num < 1.0);
        }
    }

    #[test]
    fn test_fetch_uniform_with_large_range() {
        let mut rng = crate::RandomNumberGenerator::new();
        let result = rng.fetch_uniform(-1000.0, 1000.0, 10);

        assert_eq!(result.len(), 10);

        for &num in result.iter() {
            assert!(num >= -1000.0 && num < 1000.0);
        }
    }

    #[test]
    fn test_fetch_uniform_with_empty_result() {
        let mut rng = crate::RandomNumberGenerator::new();
        let result = rng.fetch_uniform(1.0, 2.0, 0);

        assert!(result.is_empty());
    }
}
