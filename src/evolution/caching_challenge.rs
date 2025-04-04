use std::marker::PhantomData;

use crate::{
    caching::{CacheKey, CachedChallenge, ThreadLocalCachedChallenge},
    evolution::{CacheType, Challenge},
    phenotype::Phenotype,
};

/// Trait for wrapping a challenge with caching functionality.
///
/// This trait is automatically implemented for all types that implement the `Challenge` trait,
/// allowing any challenge to be easily enhanced with caching capabilities without modifying
/// its core implementation. Caching can significantly improve performance by avoiding redundant
/// fitness evaluations, which is especially valuable when fitness calculations are expensive.
///
/// To use caching, the phenotype type must implement the `CacheKey` trait, which defines
/// how to generate a unique identifier for each phenotype. This key is used as the lookup
/// value in the cache.
///
/// # Performance Considerations
///
/// - **Cache Growth**: The cache grows unbounded by default. For long-running evolutions or
///   large populations, consider clearing the cache periodically.
///
/// - **Thread Contention**: The global cache is protected by a mutex, which may become a
///   bottleneck in highly parallel workloads. For these cases, consider the thread-local
///   caching option.
///
/// - **Memory Usage**: Monitor memory usage if you're caching a very large number of phenotypes,
///   as each cache entry consumes memory.
///
/// # Type Parameters
///
/// * `P` - The phenotype type
pub trait CachingChallenge<P: Phenotype>: Challenge<P> + Sized + Clone {
    /// Wraps this challenge with a global cache.
    ///
    /// This method creates a new `CachedChallenge` that wraps the current challenge,
    /// adding a mutex-protected cache shared across all threads. Each unique phenotype
    /// (according to its `cache_key()`) is evaluated only once, and subsequent evaluations
    /// use the cached fitness value.
    ///
    /// # Performance Characteristics
    ///
    /// - **Advantages**: Ensures maximum cache reuse across all threads.
    /// - **Limitations**: May create contention when many threads try to access the cache
    ///   simultaneously. This can become a bottleneck in highly parallel scenarios.
    ///
    /// # Type Constraints
    ///
    /// The phenotype type `P` must implement the `CacheKey` trait, which defines how to
    /// generate a unique identifier for each phenotype.
    ///
    /// # Returns
    ///
    /// A `CachedChallenge` that wraps this challenge with a global cache.
    ///
    /// # Example
    ///
    /// ```
    /// # use genalg::{
    /// #     evolution::{Challenge, caching_challenge::CachingChallenge},
    /// #     phenotype::Phenotype,
    /// #     caching::CacheKey,
    /// #     rng::RandomNumberGenerator,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// #
    /// # let challenge = MyChallenge;
    /// let cached_challenge = challenge.with_global_cache();
    ///
    /// // Use the cached challenge like any other challenge
    /// let phenotype = MyPhenotype { value: 42.0 };
    /// let fitness = cached_challenge.score(&phenotype); // First evaluation is computed
    /// let fitness_again = cached_challenge.score(&phenotype); // Uses cached value
    /// ```
    fn with_global_cache(&self) -> CachedChallenge<P, Self>
    where
        P: CacheKey;

    /// Wraps this challenge with a thread-local cache.
    ///
    /// This method creates a new `ThreadLocalCachedChallenge` that wraps the current challenge,
    /// adding a separate cache for each thread. This approach avoids mutex contention in
    /// highly parallel workloads by giving each thread its own independent cache.
    ///
    /// # Performance Characteristics
    ///
    /// - **Advantages**: Eliminates mutex contention, improving parallel performance.
    /// - **Limitations**: May result in redundant evaluations across threads, as the same
    ///   phenotype evaluated in different threads will have separate cache entries.
    ///   Also uses more memory as each thread maintains its own cache.
    ///
    /// # Type Constraints
    ///
    /// The phenotype type `P` must implement the `CacheKey` trait, which defines how to
    /// generate a unique identifier for each phenotype.
    ///
    /// # Returns
    ///
    /// A `ThreadLocalCachedChallenge` that wraps this challenge with a thread-local cache.
    ///
    /// # Example
    ///
    /// ```
    /// # use genalg::{
    /// #     evolution::{Challenge, caching_challenge::CachingChallenge},
    /// #     phenotype::Phenotype,
    /// #     caching::CacheKey,
    /// #     rng::RandomNumberGenerator,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// #
    /// # let challenge = MyChallenge;
    /// let thread_local_cached_challenge = challenge.with_thread_local_cache();
    ///
    /// // Ideal for parallel processing with rayon or other threading libraries
    /// use rayon::prelude::*;
    /// let phenotypes = vec![MyPhenotype { value: 1.0 }, MyPhenotype { value: 2.0 }];
    /// let scores: Vec<f64> = phenotypes.par_iter()
    ///     .map(|p| thread_local_cached_challenge.score(p))
    ///     .collect();
    /// ```
    fn with_thread_local_cache(&self) -> ThreadLocalCachedChallenge<P, Self>
    where
        P: CacheKey;

    /// Wraps this challenge with a cache of the specified type.
    ///
    /// This method provides a convenient way to choose between global and thread-local
    /// caching based on runtime configuration.
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The type of cache to use (Global or ThreadLocal).
    ///
    /// # Returns
    ///
    /// A boxed `Challenge` that wraps this challenge with the specified cache type.
    /// The returned value is boxed to provide type erasure, allowing the client code
    /// to handle different cache implementations uniformly.
    ///
    /// # Type Constraints
    ///
    /// The phenotype type `P` must implement the `CacheKey` trait and have a `'static`
    /// lifetime. The challenge implementation must also have a `'static` lifetime.
    ///
    /// # Example
    ///
    /// ```
    /// # use genalg::{
    /// #     evolution::{Challenge, CacheType, caching_challenge::CachingChallenge},
    /// #     phenotype::Phenotype,
    /// #     caching::CacheKey,
    /// #     rng::RandomNumberGenerator,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// #
    /// # let challenge = MyChallenge;
    /// // Select caching type based on configuration or runtime conditions
    /// let use_thread_local = std::thread::available_parallelism().unwrap().get() > 4;
    /// let cache_type = if use_thread_local {
    ///     CacheType::ThreadLocal
    /// } else {
    ///     CacheType::Global
    /// };
    ///
    /// let cached_challenge = challenge.with_cache(cache_type);
    /// ```
    fn with_cache(&self, cache_type: CacheType) -> Box<dyn Challenge<P>>
    where
        P: CacheKey + 'static,
        Self: 'static,
    {
        match cache_type {
            CacheType::Global => Box::new(self.with_global_cache()),
            CacheType::ThreadLocal => Box::new(self.with_thread_local_cache()),
        }
    }
}

impl<P, C> CachingChallenge<P> for C
where
    P: Phenotype,
    C: Challenge<P> + Clone,
{
    fn with_global_cache(&self) -> CachedChallenge<P, Self>
    where
        P: CacheKey,
    {
        CachedChallenge::new(self.clone())
    }

    fn with_thread_local_cache(&self) -> ThreadLocalCachedChallenge<P, Self>
    where
        P: CacheKey,
    {
        ThreadLocalCachedChallenge::new(self.clone())
    }
}

/// A challenge adapter that can conditionally enable caching.
///
/// This adapter wraps a challenge and provides the same functionality as the wrapped challenge,
/// allowing caching to be configured at runtime. Currently, it simply delegates to the inner
/// challenge without implementing actual caching - it serves as a placeholder for future
/// conditional caching functionality.
///
/// # Type Parameters
///
/// * `P` - The phenotype type
/// * `C` - The inner challenge type
///
/// # Note
///
/// Important: In the current implementation, the `CachingChallengeSwitcher` does not actually
/// implement caching functionality. It always calls the inner challenge's score method directly,
/// regardless of whether a cache type is set. It is intended as a structural pattern for
/// future enhancements.
///
/// For active caching, use the `CachingChallenge` trait methods directly:
/// - `challenge.with_global_cache()`
/// - `challenge.with_thread_local_cache()`
/// - `challenge.with_cache(cache_type)`
#[derive(Debug, Clone)]
pub struct CachingChallengeSwitcher<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    inner: C,
    cache_type: Option<CacheType>,
    _phantom: PhantomData<P>,
}

impl<P, C> CachingChallengeSwitcher<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    /// Creates a new `CachingChallengeSwitcher` with the specified challenge.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to wrap
    ///
    /// # Returns
    ///
    /// A new `CachingChallengeSwitcher` instance with caching disabled by default.
    ///
    /// # Example
    ///
    /// ```
    /// # use genalg::{
    /// #     evolution::{Challenge, caching_challenge::CachingChallengeSwitcher},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// #
    /// # let challenge = MyChallenge;
    /// let switcher = CachingChallengeSwitcher::new(challenge);
    /// ```
    pub fn new(challenge: C) -> Self {
        Self {
            inner: challenge,
            cache_type: None,
            _phantom: PhantomData,
        }
    }

    /// Configures this switcher to use caching.
    ///
    /// # Arguments
    ///
    /// * `cache_type` - The type of cache to use
    ///
    /// # Returns
    ///
    /// This switcher configured to use caching.
    ///
    /// # Note
    ///
    /// In the current implementation, this setting does not actually enable caching functionality.
    /// The switcher still calls the inner challenge's score method directly. This method is
    /// intended for future enhancements to the conditional caching functionality.
    ///
    /// # Example
    ///
    /// ```
    /// # use genalg::{
    /// #     evolution::{Challenge, CacheType, caching_challenge::CachingChallengeSwitcher},
    /// #     phenotype::Phenotype,
    /// #     rng::RandomNumberGenerator,
    /// # };
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    /// #
    /// # let challenge = MyChallenge;
    /// let switcher = CachingChallengeSwitcher::new(challenge)
    ///     .with_cache(CacheType::Global);
    /// ```
    pub fn with_cache(mut self, cache_type: CacheType) -> Self {
        self.cache_type = Some(cache_type);
        self
    }

    /// Returns the inner challenge.
    ///
    /// # Returns
    ///
    /// A reference to the inner challenge.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    /// Unwraps this switcher, returning the inner challenge.
    ///
    /// # Returns
    ///
    /// The inner challenge.
    pub fn unwrap(self) -> C {
        self.inner
    }
}

impl<P, C> Challenge<P> for CachingChallengeSwitcher<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn score(&self, phenotype: &P) -> f64 {
        // Currently, always use the inner challenge directly
        // In a future implementation, this could conditionally use caching
        // based on the cache_type field
        self.inner.score(phenotype)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use super::*;
    use crate::rng::RandomNumberGenerator;

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        value: i32,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2;
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            self.value += 1;
        }
    }

    impl CacheKey for TestPhenotype {
        type Key = i32;

        fn cache_key(&self) -> Self::Key {
            self.value
        }
    }

    #[derive(Clone)]
    struct TestChallenge {
        evaluations: Arc<AtomicUsize>,
    }

    impl TestChallenge {
        fn new() -> Self {
            Self {
                evaluations: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn get_evaluations(&self) -> usize {
            self.evaluations.load(Ordering::SeqCst)
        }
    }

    impl Challenge<TestPhenotype> for TestChallenge {
        fn score(&self, phenotype: &TestPhenotype) -> f64 {
            // Increment evaluation counter
            self.evaluations.fetch_add(1, Ordering::SeqCst);

            // Simple fitness function: smaller values are better
            -(phenotype.value as f64)
        }
    }

    #[test]
    fn test_with_global_cache() {
        let challenge = TestChallenge::new();
        let cached_challenge = challenge.with_global_cache();

        let phenotype1 = TestPhenotype { value: 5 };
        let phenotype2 = TestPhenotype { value: 10 };

        // First evaluation should compute
        let score1 = cached_challenge.score(&phenotype1);
        assert_eq!(score1, -5.0);
        assert_eq!(challenge.get_evaluations(), 1);

        // Second evaluation of the same phenotype should use cache
        let score1_again = cached_challenge.score(&phenotype1);
        assert_eq!(score1_again, -5.0);
        assert_eq!(challenge.get_evaluations(), 1); // Still 1

        // Different phenotype should compute
        let score2 = cached_challenge.score(&phenotype2);
        assert_eq!(score2, -10.0);
        assert_eq!(challenge.get_evaluations(), 2);
    }

    #[test]
    fn test_with_thread_local_cache() {
        let challenge = TestChallenge::new();
        let cached_challenge = challenge.with_thread_local_cache();

        let phenotype1 = TestPhenotype { value: 5 };
        let phenotype2 = TestPhenotype { value: 10 };

        // First evaluation should compute
        let score1 = cached_challenge.score(&phenotype1);
        assert_eq!(score1, -5.0);
        assert_eq!(challenge.get_evaluations(), 1);

        // Second evaluation of the same phenotype should use cache
        let score1_again = cached_challenge.score(&phenotype1);
        assert_eq!(score1_again, -5.0);
        assert_eq!(challenge.get_evaluations(), 1); // Still 1

        // Different phenotype should compute
        let score2 = cached_challenge.score(&phenotype2);
        assert_eq!(score2, -10.0);
        assert_eq!(challenge.get_evaluations(), 2);
    }

    #[test]
    fn test_with_cache() {
        let challenge = TestChallenge::new();

        // Test with global cache
        let cached_challenge = challenge.clone().with_cache(CacheType::Global);

        let phenotype = TestPhenotype { value: 5 };

        let score1 = cached_challenge.score(&phenotype);
        assert_eq!(score1, -5.0);

        let score2 = cached_challenge.score(&phenotype);
        assert_eq!(score2, -5.0);

        // The evaluation count would be 1, but we can't access it through the dyn trait

        // Test with thread local cache
        let cached_challenge = challenge.clone().with_cache(CacheType::ThreadLocal);

        let score1 = cached_challenge.score(&phenotype);
        assert_eq!(score1, -5.0);

        let score2 = cached_challenge.score(&phenotype);
        assert_eq!(score2, -5.0);
    }

    #[test]
    fn test_caching_challenge_switcher() {
        let challenge = TestChallenge::new();

        // Without caching
        let switcher = CachingChallengeSwitcher::new(challenge.clone());

        let phenotype = TestPhenotype { value: 5 };

        let score1 = switcher.score(&phenotype);
        assert_eq!(score1, -5.0);
        assert_eq!(challenge.get_evaluations(), 1);

        let score2 = switcher.score(&phenotype);
        assert_eq!(score2, -5.0);
        assert_eq!(challenge.get_evaluations(), 2); // No caching, so 2 evaluations

        // With caching - note that the switcher currently ignores the cache type
        // and just delegates to the inner challenge directly
        let challenge = TestChallenge::new();
        let switcher =
            CachingChallengeSwitcher::new(challenge.clone()).with_cache(CacheType::Global);

        let score1 = switcher.score(&phenotype);
        assert_eq!(score1, -5.0);
        assert_eq!(challenge.get_evaluations(), 1);

        let score2 = switcher.score(&phenotype);
        assert_eq!(score2, -5.0);
        assert_eq!(challenge.get_evaluations(), 2); // The switcher doesn't implement caching yet
    }
}
