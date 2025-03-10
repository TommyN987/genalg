//! # Caching Module
//!
//! This module provides caching mechanisms for fitness evaluations in genetic algorithms.
//! Caching can significantly improve performance by avoiding redundant calculations,
//! especially when fitness evaluations are computationally expensive.
//!
//! ## Overview
//!
//! In genetic algorithms, the same phenotype may be evaluated multiple times during evolution.
//! By caching fitness scores, we can avoid recalculating values we've already computed.
//! This module provides two main caching implementations:
//!
//! - `CachedChallenge`: A thread-safe cache using a mutex
//! - `ThreadLocalCachedChallenge`: A thread-local cache for better parallel performance
//!
//! ## Usage
//!
//! To use caching, you need to:
//!
//! 1. Implement the `CacheKey` trait for your phenotype
//! 2. Wrap your challenge in one of the cache decorators
//!
//! ```rust
//! use genalg::caching::{CacheKey, CachedChallenge};
//! use genalg::evolution::Challenge;
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//! use std::fmt::Debug;
//!
//! #[derive(Clone, Debug)]
//! struct MyPhenotype {
//!     values: Vec<i32>,
//! }
//!
//! impl Phenotype for MyPhenotype {
//!     fn crossover(&mut self, other: &Self) {
//!         // Implementation omitted for brevity
//!     }
//!
//!     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
//!         // Implementation omitted for brevity
//!     }
//! }
//!
//! // Implement CacheKey to enable caching
//! impl CacheKey for MyPhenotype {
//!     // Use a type that uniquely identifies phenotypes with the same fitness
//!     type Key = Vec<i32>;
//!
//!     fn cache_key(&self) -> Self::Key {
//!         self.values.clone()
//!     }
//! }
//!
//! // Your challenge implementation
//! struct MyChallenge;
//!
//! impl Challenge<MyPhenotype> for MyChallenge {
//!     fn score(&self, phenotype: &MyPhenotype) -> f64 {
//!         // Expensive computation...
//!         phenotype.values.iter().sum::<i32>() as f64
//!     }
//! }
//!
//! // Create a cached version of your challenge
//! let challenge = MyChallenge;
//! let cached_challenge = CachedChallenge::new(challenge);
//!
//! // Use the cached challenge just like the original
//! let phenotype = MyPhenotype { values: vec![1, 2, 3] };
//! let score = cached_challenge.score(&phenotype); // Computed and cached
//! let score_again = cached_challenge.score(&phenotype); // Retrieved from cache
//! ```
//!
//! ## Choosing a Cache Implementation
//!
//! - `CachedChallenge`: Good for single-threaded or low-contention scenarios
//! - `ThreadLocalCachedChallenge`: Better for highly parallel workloads
//!
//! ## Performance Considerations
//!
//! - Cache lookups are generally much faster than fitness evaluations for complex problems
//! - The cache grows unbounded, so consider calling `clear_cache()` periodically for long-running evolutions
//! - Thread contention on the mutex can become a bottleneck in `CachedChallenge` with many threads
//! - `ThreadLocalCachedChallenge` avoids mutex contention but may use more memory with many threads
//!
//! ## Memory Usage
//!
//! Both cache implementations store fitness values indefinitely until `clear_cache()` is called.
//! For long-running evolutions or large populations, monitor memory usage and clear the cache
//! when appropriate.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;

/// A trait for phenotypes that can be cached.
///
/// This trait extends the `Phenotype` trait and adds the ability to generate a cache key.
/// The cache key is used to identify phenotypes that would have the same fitness score,
/// allowing the caching system to avoid redundant fitness evaluations.
///
/// # Implementation Guidelines
///
/// When implementing `CacheKey`, consider the following:
///
/// 1. The `Key` type should uniquely identify phenotypes with the same fitness
/// 2. The `Key` must be hashable, comparable, and clonable
/// 3. The `cache_key()` method should be efficient, as it's called for every fitness evaluation
/// 4. The `Key` should capture all aspects of the phenotype that affect its fitness
///
/// # Examples
///
/// ```
/// use genalg::caching::CacheKey;
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
/// use std::fmt::Debug;
///
/// #[derive(Clone, Debug)]
/// struct Vector2D {
///     x: f64,
///     y: f64,
/// }
///
/// impl Phenotype for Vector2D {
///     fn crossover(&mut self, other: &Self) {
///         self.x = (self.x + other.x) / 2.0;
///         self.y = (self.y + other.y) / 2.0;
///     }
///
///     fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
///         // Implementation omitted for brevity
///     }
/// }
///
/// impl CacheKey for Vector2D {
///     // For this example, we'll use a tuple of rounded values as the key
///     // This allows for small floating-point differences to map to the same key
///     type Key = (i64, i64);
///
///     fn cache_key(&self) -> Self::Key {
///         // Round to nearest integer to handle floating-point imprecision
///         let x_rounded = (self.x * 1000.0).round() as i64;
///         let y_rounded = (self.y * 1000.0).round() as i64;
///         (x_rounded, y_rounded)
///     }
/// }
/// ```
///
/// In this example, we round the floating-point values to handle small differences
/// that wouldn't affect the fitness score significantly.
pub trait CacheKey: Phenotype {
    /// The type of the cache key.
    type Key: Eq + Hash + Clone + Debug + Send + Sync;

    /// Generates a cache key for this phenotype.
    ///
    /// The cache key should uniquely identify the phenotype for fitness evaluation purposes.
    /// Phenotypes that would have the same fitness score should have the same cache key.
    fn cache_key(&self) -> Self::Key;
}

/// A challenge wrapper that caches fitness evaluations.
///
/// This struct wraps a challenge and caches the results of fitness evaluations
/// to avoid redundant calculations. It uses a thread-safe mutex-protected cache,
/// making it suitable for both single-threaded and multi-threaded environments.
///
/// # Performance Considerations
///
/// - The cache is protected by a mutex, which can become a bottleneck in highly parallel scenarios
/// - Consider using `ThreadLocalCachedChallenge` for better parallel performance
/// - The cache grows unbounded, so consider calling `clear_cache()` periodically for long-running evolutions
///
/// # Examples
///
/// ```
/// use genalg::caching::{CacheKey, CachedChallenge};
/// use genalg::evolution::Challenge;
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
///
/// // Define a phenotype
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
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         self.value += 0.1;
///     }
/// }
///
/// // Implement CacheKey for the phenotype
/// impl CacheKey for MyPhenotype {
///     type Key = i64;
///
///     fn cache_key(&self) -> Self::Key {
///         (self.value * 1000.0).round() as i64
///     }
/// }
///
/// // Define a challenge
/// struct ExpensiveChallenge;
///
/// impl Challenge<MyPhenotype> for ExpensiveChallenge {
///     fn score(&self, phenotype: &MyPhenotype) -> f64 {
///         // Simulate an expensive computation
///         std::thread::sleep(std::time::Duration::from_millis(10));
///         (phenotype.value - 5.0).powi(2)
///     }
/// }
///
/// // Create a cached version of the challenge
/// let challenge = ExpensiveChallenge;
/// let cached_challenge = CachedChallenge::new(challenge);
///
/// // First evaluation (computed and cached)
/// let phenotype = MyPhenotype { value: 3.0 };
/// let score1 = cached_challenge.score(&phenotype);
///
/// // Second evaluation of the same phenotype (retrieved from cache)
/// let score2 = cached_challenge.score(&phenotype);
///
/// // The scores should be identical
/// assert_eq!(score1, score2);
///
/// // Check the cache size
/// assert_eq!(cached_challenge.cache_size(), 1);
///
/// // Clear the cache if needed
/// cached_challenge.clear_cache();
/// assert_eq!(cached_challenge.cache_size(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct CachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    /// The wrapped challenge
    challenge: C,
    /// The cache of fitness evaluations
    cache: Arc<Mutex<HashMap<P::Key, f64>>>,
    /// Phantom data for the phenotype type
    _marker: PhantomData<P>,
}

impl<P, C> CachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    /// Creates a new cached challenge wrapper.
    ///
    /// This constructor creates a new cached challenge with an empty cache.
    /// The cache will be populated as fitness evaluations are performed.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to wrap with caching functionality.
    ///
    /// # Returns
    ///
    /// A new `CachedChallenge` instance that wraps the provided challenge.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = CachedChallenge::new(challenge);
    /// ```
    pub fn new(challenge: C) -> Self {
        Self {
            challenge,
            cache: Arc::new(Mutex::new(HashMap::new())),
            _marker: PhantomData,
        }
    }

    /// Creates a new cached challenge with a pre-populated cache.
    ///
    /// This constructor allows you to initialize the cache with pre-computed values,
    /// which can be useful for resuming a previous evolution or for testing.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to wrap with caching functionality.
    /// * `cache` - A HashMap containing pre-computed fitness values.
    ///
    /// # Returns
    ///
    /// A new `CachedChallenge` instance with the provided cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// // Create a pre-populated cache
    /// let mut cache = HashMap::new();
    /// cache.insert(3000, 4.0); // Key for value 3.0, pre-computed score 4.0
    ///
    /// let challenge = MyChallenge;
    /// let cached_challenge = CachedChallenge::with_cache(challenge, cache);
    ///
    /// // The pre-computed value is already in the cache
    /// let phenotype = MyPhenotype { value: 3.0 };
    /// let score = cached_challenge.score(&phenotype);
    /// assert_eq!(score, 4.0);
    /// ```
    pub fn with_cache(challenge: C, cache: HashMap<P::Key, f64>) -> Self {
        Self {
            challenge,
            cache: Arc::new(Mutex::new(cache)),
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the wrapped challenge.
    ///
    /// This method provides access to the underlying challenge, which can be useful
    /// for inspecting or modifying its properties.
    ///
    /// # Returns
    ///
    /// A reference to the wrapped challenge.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge { difficulty: f64 }
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value * self.difficulty }
    /// # }
    /// #
    /// let challenge = MyChallenge { difficulty: 2.0 };
    /// let cached_challenge = CachedChallenge::new(challenge);
    ///
    /// // Access the inner challenge
    /// assert_eq!(cached_challenge.inner().difficulty, 2.0);
    /// ```
    pub fn inner(&self) -> &C {
        &self.challenge
    }

    /// Returns the current size of the cache.
    ///
    /// This method returns the number of entries in the cache, which can be useful
    /// for monitoring cache usage.
    ///
    /// # Returns
    ///
    /// The number of entries in the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = CachedChallenge::new(challenge);
    ///
    /// // Initially, the cache is empty
    /// assert_eq!(cached_challenge.cache_size(), 0);
    ///
    /// // After scoring a phenotype, the cache has one entry
    /// let phenotype = MyPhenotype { value: 3.0 };
    /// cached_challenge.score(&phenotype);
    /// assert_eq!(cached_challenge.cache_size(), 1);
    /// ```
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Clears the cache, removing all entries.
    ///
    /// This method removes all entries from the cache, which can be useful for
    /// freeing memory or resetting the cache after a significant change in the
    /// fitness landscape.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = CachedChallenge::new(challenge);
    ///
    /// // Add some entries to the cache
    /// let phenotype1 = MyPhenotype { value: 3.0 };
    /// let phenotype2 = MyPhenotype { value: 4.0 };
    /// cached_challenge.score(&phenotype1);
    /// cached_challenge.score(&phenotype2);
    /// assert_eq!(cached_challenge.cache_size(), 2);
    ///
    /// // Clear the cache
    /// cached_challenge.clear_cache();
    /// assert_eq!(cached_challenge.cache_size(), 0);
    /// ```
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Returns a copy of the current cache.
    ///
    /// This method provides access to the current state of the cache, which can be useful
    /// for inspecting the cached values, saving the cache state for later use, or
    /// transferring the cache to another `CachedChallenge` instance.
    ///
    /// # Returns
    ///
    /// A `HashMap` containing the cache keys and their corresponding fitness values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, CachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = CachedChallenge::new(challenge);
    ///
    /// // Add some entries to the cache
    /// let phenotype1 = MyPhenotype { value: 3.0 };
    /// let phenotype2 = MyPhenotype { value: 4.0 };
    /// cached_challenge.score(&phenotype1);
    /// cached_challenge.score(&phenotype2);
    ///
    /// // Get the current cache state
    /// let cache = cached_challenge.get_cache();
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(*cache.get(&3000).unwrap(), 3.0); // Key is rounded value * 1000
    /// assert_eq!(*cache.get(&4000).unwrap(), 4.0);
    ///
    /// // The cache can be used to create a new CachedChallenge with the same cache state
    /// let new_challenge = MyChallenge;
    /// let new_cached_challenge = CachedChallenge::with_cache(new_challenge, cache);
    /// ```
    pub fn get_cache(&self) -> HashMap<P::Key, f64> {
        self.cache.lock().unwrap().clone()
    }
}

impl<P, C> Challenge<P> for CachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    fn score(&self, phenotype: &P) -> f64 {
        let key = phenotype.cache_key();

        // Try to get the score from the cache
        let mut cache = self.cache.lock().unwrap();

        if let Some(score) = cache.get(&key) {
            return *score;
        }

        // If not in cache, calculate the score and cache it
        let score = self.challenge.score(phenotype);
        cache.insert(key, score);

        score
    }
}

/// A thread-local cache for storing key-value pairs.
///
/// This cache maintains separate storage for each thread, eliminating the need for
/// synchronization when accessed from multiple threads. This makes it particularly
/// useful for parallel workloads.
///
/// # Performance Characteristics
///
/// - No synchronization overhead for cache access
/// - Each thread maintains its own cache, which can lead to higher memory usage
/// - The cache grows unbounded within each thread until `clear()` is called
///
/// # Examples
///
/// ```
/// use genalg::caching::{CacheKey, ThreadLocalCache};
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
///
/// // Define a phenotype and implement CacheKey
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
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         self.value += 0.1;
///     }
/// }
///
/// impl CacheKey for MyPhenotype {
///     type Key = i64;
///
///     fn cache_key(&self) -> Self::Key {
///         (self.value * 1000.0).round() as i64
///     }
/// }
///
/// // Create a thread-local cache
/// let cache = ThreadLocalCache::<MyPhenotype>::new();
///
/// // Insert some values
/// cache.insert(1000, 1.0);
/// cache.insert(2000, 2.0);
///
/// // Retrieve values
/// assert_eq!(cache.get(&1000), Some(1.0));
/// assert_eq!(cache.get(&2000), Some(2.0));
/// assert_eq!(cache.get(&3000), None);
///
/// // Check cache size
/// assert_eq!(cache.len(), 2);
/// assert!(!cache.is_empty());
///
/// // Clear the cache
/// cache.clear();
/// assert_eq!(cache.len(), 0);
/// assert!(cache.is_empty());
/// ```
///
/// In multi-threaded scenarios, each thread has its own separate cache:
///
/// ```no_run
/// use genalg::caching::{CacheKey, ThreadLocalCache};
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
/// use std::thread;
/// use std::sync::Arc;
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
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         self.value += 0.1;
///     }
/// }
///
/// impl CacheKey for MyPhenotype {
///     type Key = i64;
///
///     fn cache_key(&self) -> Self::Key {
///         (self.value * 1000.0).round() as i64
///     }
/// }
///
/// let cache = Arc::new(ThreadLocalCache::<MyPhenotype>::new());
///
/// // Spawn threads that each have their own cache
/// let handles: Vec<_> = (0..4).map(|i| {
///     let cache = cache.clone();
///     thread::spawn(move || {
///         // Each thread inserts into its own cache
///         cache.insert(i as i64, i as f64);
///         assert_eq!(cache.len(), 1);
///     })
/// }).collect();
///
/// // Wait for all threads to complete
/// for handle in handles {
///     handle.join().unwrap();
/// }
///
/// // Main thread's cache is still empty
/// assert_eq!(cache.len(), 0);
/// ```
#[derive(Debug)]
pub struct ThreadLocalCache<P>
where
    P: CacheKey,
{
    /// The cache of fitness evaluations
    cache: thread_local::ThreadLocal<RefCell<HashMap<P::Key, f64>>>,
}

impl<P> ThreadLocalCache<P>
where
    P: CacheKey,
{
    /// Creates a new empty thread-local cache.
    pub fn new() -> Self {
        Self {
            cache: thread_local::ThreadLocal::new(),
        }
    }

    /// Gets a cached fitness value if available.
    ///
    /// This method attempts to retrieve a cached fitness value for the given key.
    /// It will only return a value if the key exists in the cache for the current thread.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up in the cache.
    ///
    /// # Returns
    ///
    /// * `Some(f64)` - The cached fitness value if found.
    /// * `None` - If the key is not in the cache for the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Initially, the cache is empty
    /// let key = 42;
    /// assert_eq!(cache.get(&key), None);
    ///
    /// // After inserting a value, we can retrieve it
    /// cache.insert(key, 3.14);
    /// assert_eq!(cache.get(&key), Some(3.14));
    /// ```
    pub fn get(&self, key: &P::Key) -> Option<f64> {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .and_then(|cache| cache.get(key).copied())
    }

    /// Inserts a fitness value into the cache.
    ///
    /// This method adds or updates a key-value pair in the thread-local cache.
    /// If the key already exists, its value will be updated.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert or update.
    /// * `value` - The fitness value to associate with the key.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Insert a new key-value pair
    /// let key = 42;
    /// cache.insert(key, 3.14);
    /// assert_eq!(cache.get(&key), Some(3.14));
    ///
    /// // Update an existing key
    /// cache.insert(key, 2.71);
    /// assert_eq!(cache.get(&key), Some(2.71));
    /// ```
    pub fn insert(&self, key: P::Key, value: f64) {
        if let Some(cell) = self.cache.get() {
            if let Ok(mut cache) = cell.try_borrow_mut() {
                cache.insert(key, value);
                return;
            }
        }

        // If we couldn't get the existing cache, create a new one
        let mut new_cache = HashMap::new();
        new_cache.insert(key, value);
        let cell = RefCell::new(new_cache);
        self.cache.get_or(|| cell);
    }

    /// Clears the cache for the current thread.
    ///
    /// This method removes all entries from the cache for the current thread,
    /// which can be useful for freeing memory or resetting the cache.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Add some entries to the cache
    /// cache.insert(1, 1.0);
    /// cache.insert(2, 2.0);
    /// assert_eq!(cache.len(), 2);
    ///
    /// // Clear the cache
    /// cache.clear();
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn clear(&self) {
        if let Some(cell) = self.cache.get() {
            if let Ok(mut cache) = cell.try_borrow_mut() {
                cache.clear();
            }
        }
    }

    /// Returns the number of cached fitness evaluations for the current thread.
    ///
    /// This method returns the number of entries in the cache for the current thread,
    /// which can be useful for monitoring cache usage.
    ///
    /// # Returns
    ///
    /// The number of entries in the cache for the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Initially, the cache is empty
    /// assert_eq!(cache.len(), 0);
    ///
    /// // After adding entries, the size increases
    /// cache.insert(1, 1.0);
    /// cache.insert(2, 2.0);
    /// assert_eq!(cache.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .map_or(0, |cache| cache.len())
    }

    /// Returns `true` if the cache for the current thread is empty.
    ///
    /// This method checks if the cache for the current thread contains any entries.
    ///
    /// # Returns
    ///
    /// `true` if the cache is empty, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Initially, the cache is empty
    /// assert!(cache.is_empty());
    ///
    /// // After adding an entry, it's no longer empty
    /// cache.insert(1, 1.0);
    /// assert!(!cache.is_empty());
    ///
    /// // After clearing, it's empty again
    /// cache.clear();
    /// assert!(cache.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .map_or(true, |cache| cache.is_empty())
    }

    /// Returns a copy of the current thread-local cache.
    ///
    /// This method provides access to the current state of the cache for the calling thread,
    /// which can be useful for inspecting the cached values or saving the cache state for later use.
    ///
    /// # Returns
    ///
    /// A `HashMap` containing the cache keys and their corresponding fitness values for the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCache};
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// let cache = ThreadLocalCache::<MyPhenotype>::new();
    ///
    /// // Add some entries to the cache
    /// cache.insert(1000, 1.0);
    /// cache.insert(2000, 2.0);
    ///
    /// // Get the current cache state for this thread
    /// let cache_copy = cache.get_cache();
    /// assert_eq!(cache_copy.len(), 2);
    /// assert_eq!(*cache_copy.get(&1000).unwrap(), 1.0);
    /// assert_eq!(*cache_copy.get(&2000).unwrap(), 2.0);
    /// ```
    pub fn get_cache(&self) -> HashMap<P::Key, f64> {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .map_or_else(HashMap::new, |cache| cache.clone())
    }
}

impl<P> Default for ThreadLocalCache<P>
where
    P: CacheKey,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A challenge wrapper that caches fitness evaluations using thread-local storage.
///
/// This struct wraps a challenge and caches the results of fitness evaluations
/// to avoid redundant calculations. It uses thread-local storage for the cache,
/// which avoids mutex contention in highly parallel scenarios.
///
/// # Performance Considerations
///
/// - Each thread maintains its own cache, which eliminates mutex contention
/// - May use more memory than `CachedChallenge` when using many threads
/// - Particularly effective for parallel evolution with many threads
/// - The cache grows unbounded, so consider calling `clear_cache()` periodically
///
/// # When to Use
///
/// Use `ThreadLocalCachedChallenge` when:
/// - You're running evolution in parallel with many threads
/// - You're experiencing contention on the mutex in `CachedChallenge`
/// - Each thread tends to evaluate similar phenotypes
///
/// # Examples
///
/// ```
/// use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
/// use genalg::evolution::Challenge;
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
/// use std::thread;
///
/// // Define a phenotype
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
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         self.value += 0.1;
///     }
/// }
///
/// // Implement CacheKey for the phenotype
/// impl CacheKey for MyPhenotype {
///     type Key = i64;
///
///     fn cache_key(&self) -> Self::Key {
///         (self.value * 1000.0).round() as i64
///     }
/// }
///
/// // Define a challenge
/// #[derive(Clone)]
/// struct ExpensiveChallenge;
///
/// impl Challenge<MyPhenotype> for ExpensiveChallenge {
///     fn score(&self, phenotype: &MyPhenotype) -> f64 {
///         // Simulate an expensive computation
///         std::thread::sleep(std::time::Duration::from_millis(10));
///         (phenotype.value - 5.0).powi(2)
///     }
/// }
///
/// // Create a cached version of the challenge
/// let challenge = ExpensiveChallenge;
/// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
/// let cached_challenge = std::sync::Arc::new(cached_challenge);
///
/// // Spawn multiple threads to demonstrate thread-local caching
/// let mut handles = vec![];
/// for _ in 0..4 {
///     let cached_challenge = cached_challenge.clone();
///     let handle = thread::spawn(move || {
///         // Each thread evaluates the same phenotype twice
///         let phenotype = MyPhenotype { value: 3.0 };
///         
///         // First evaluation (computed and cached in this thread's local cache)
///         let score1 = cached_challenge.score(&phenotype);
///         
///         // Second evaluation (retrieved from this thread's local cache)
///         let score2 = cached_challenge.score(&phenotype);
///         
///         // The scores should be identical
///         assert_eq!(score1, score2);
///     });
///     handles.push(handle);
/// }
///
/// // Wait for all threads to complete
/// for handle in handles {
///     handle.join().unwrap();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ThreadLocalCachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    /// The wrapped challenge
    challenge: C,
    /// The thread-local cache of fitness evaluations
    cache: Arc<ThreadLocalCache<P>>,
    /// Phantom data for the phenotype type
    _marker: PhantomData<P>,
}

impl<P, C> ThreadLocalCachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    /// Creates a new thread-local cached challenge wrapper.
    ///
    /// This constructor creates a new cached challenge that uses thread-local storage
    /// for the cache. Each thread will maintain its own cache, which eliminates
    /// synchronization overhead in parallel scenarios.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to wrap with caching functionality.
    ///
    /// # Returns
    ///
    /// A new `ThreadLocalCachedChallenge` instance that wraps the provided challenge.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
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
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
    /// ```
    pub fn new(challenge: C) -> Self {
        Self {
            challenge,
            cache: Arc::new(ThreadLocalCache::new()),
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the wrapped challenge.
    ///
    /// This method provides access to the underlying challenge, which can be useful
    /// for inspecting or modifying its properties.
    ///
    /// # Returns
    ///
    /// A reference to the wrapped challenge.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge { difficulty: f64 }
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value * self.difficulty }
    /// # }
    /// #
    /// let challenge = MyChallenge { difficulty: 2.0 };
    /// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
    ///
    /// // Access the inner challenge
    /// assert_eq!(cached_challenge.inner().difficulty, 2.0);
    /// ```
    pub fn inner(&self) -> &C {
        &self.challenge
    }

    /// Returns the current size of the thread-local cache for the calling thread.
    ///
    /// This method returns the number of entries in the cache for the current thread,
    /// which can be useful for monitoring cache usage.
    ///
    /// # Returns
    ///
    /// The number of entries in the cache for the current thread.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
    ///
    /// // Initially, the cache is empty
    /// assert_eq!(cached_challenge.cache_size(), 0);
    ///
    /// // After scoring a phenotype, the cache has one entry
    /// let phenotype = MyPhenotype { value: 3.0 };
    /// cached_challenge.score(&phenotype);
    /// assert_eq!(cached_challenge.cache_size(), 1);
    ///
    /// // Note: This only shows the cache size for the current thread
    /// // Other threads will have their own separate caches
    /// ```
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clears the thread-local cache for the calling thread, removing all entries.
    ///
    /// This method removes all entries from the cache for the current thread, which can be useful
    /// for freeing memory or resetting the cache after a significant change in the fitness landscape.
    ///
    /// # Note
    ///
    /// This method only clears the cache for the current thread. Other threads will still have
    /// their caches intact.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
    ///
    /// // Add some entries to the cache
    /// let phenotype1 = MyPhenotype { value: 3.0 };
    /// let phenotype2 = MyPhenotype { value: 4.0 };
    /// cached_challenge.score(&phenotype1);
    /// cached_challenge.score(&phenotype2);
    /// assert_eq!(cached_challenge.cache_size(), 2);
    ///
    /// // Clear the cache for the current thread
    /// cached_challenge.clear_cache();
    /// assert_eq!(cached_challenge.cache_size(), 0);
    ///
    /// // Note: This only clears the cache for the current thread
    /// // Other threads will still have their caches intact
    /// ```
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Returns a copy of the current thread-local cache.
    ///
    /// This method provides access to the current state of the cache for the calling thread,
    /// which can be useful for inspecting the cached values or saving the cache state for later use.
    ///
    /// # Returns
    ///
    /// A `HashMap` containing the cache keys and their corresponding fitness values for the current thread.
    ///
    /// # Note
    ///
    /// This method only returns the cache for the current thread. Other threads will have
    /// their own separate caches that are not included in the result.
    ///
    /// # Examples
    ///
    /// ```
    /// # use genalg::caching::{CacheKey, ThreadLocalCachedChallenge};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use std::collections::HashMap;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// #
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, _other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # impl CacheKey for MyPhenotype {
    /// #     type Key = i64;
    /// #     fn cache_key(&self) -> Self::Key { (self.value * 1000.0).round() as i64 }
    /// # }
    /// #
    /// # struct MyChallenge;
    /// #
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, phenotype: &MyPhenotype) -> f64 { phenotype.value }
    /// # }
    /// #
    /// let challenge = MyChallenge;
    /// let cached_challenge = ThreadLocalCachedChallenge::new(challenge);
    ///
    /// // Add some entries to the cache
    /// let phenotype1 = MyPhenotype { value: 3.0 };
    /// let phenotype2 = MyPhenotype { value: 4.0 };
    /// cached_challenge.score(&phenotype1);
    /// cached_challenge.score(&phenotype2);
    ///
    /// // Get the current cache state for this thread
    /// let cache = cached_challenge.get_cache();
    /// assert_eq!(cache.len(), 2);
    /// assert_eq!(*cache.get(&3000).unwrap(), 3.0); // Key is rounded value * 1000
    /// assert_eq!(*cache.get(&4000).unwrap(), 4.0);
    /// ```
    pub fn get_cache(&self) -> HashMap<P::Key, f64> {
        (*self.cache).get_cache()
    }
}

impl<P, C> Challenge<P> for ThreadLocalCachedChallenge<P, C>
where
    P: CacheKey,
    C: Challenge<P>,
{
    fn score(&self, phenotype: &P) -> f64 {
        let key = phenotype.cache_key();

        // Try to get the score from the cache
        if let Some(score) = self.cache.get(&key) {
            return score;
        }

        // If not in cache, calculate the score and cache it
        let score = self.challenge.score(phenotype);
        self.cache.insert(key, score);

        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::Challenge;
    use crate::phenotype::Phenotype;
    use crate::rng::RandomNumberGenerator;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        value: i32,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2;
        }

        fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
            let values = rng.fetch_uniform(-1.0, 1.0, 1);
            let delta = values.front().unwrap();
            self.value += (*delta * 10.0) as i32;
        }
    }

    impl CacheKey for TestPhenotype {
        type Key = i32;

        fn cache_key(&self) -> Self::Key {
            self.value
        }
    }

    #[derive(Debug, Clone)]
    struct TestChallenge {
        target: i32,
        // Counter to track the number of evaluations
        evaluations: Arc<AtomicUsize>,
    }

    impl TestChallenge {
        fn new(target: i32) -> Self {
            Self {
                target,
                evaluations: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn get_evaluations(&self) -> usize {
            self.evaluations.load(Ordering::SeqCst)
        }
    }

    impl Challenge<TestPhenotype> for TestChallenge {
        fn score(&self, phenotype: &TestPhenotype) -> f64 {
            // Increment the evaluation counter
            self.evaluations.fetch_add(1, Ordering::SeqCst);

            // Higher score is better (inverse of distance to target)
            1.0 / ((phenotype.value - self.target).abs() as f64 + 1.0)
        }
    }

    #[test]
    fn test_cached_challenge() {
        let challenge = TestChallenge::new(50);
        let cached_challenge = CachedChallenge::new(challenge.clone());

        // First evaluation should calculate the score
        let phenotype1 = TestPhenotype { value: 10 };
        let score1 = cached_challenge.score(&phenotype1);
        assert_eq!(challenge.get_evaluations(), 1);

        // Second evaluation of the same phenotype should use the cache
        let phenotype2 = TestPhenotype { value: 10 };
        let score2 = cached_challenge.score(&phenotype2);
        assert_eq!(challenge.get_evaluations(), 1); // Still 1, not 2
        assert_eq!(score1, score2);

        // Different phenotype should calculate a new score
        let phenotype3 = TestPhenotype { value: 20 };
        let score3 = cached_challenge.score(&phenotype3);
        assert_eq!(challenge.get_evaluations(), 2);
        assert_ne!(score1, score3);

        // Test cache size
        assert_eq!(cached_challenge.cache_size(), 2);

        // Test clear cache
        cached_challenge.clear_cache();
        assert_eq!(cached_challenge.cache_size(), 0);

        // After clearing, should calculate again
        let _score4 = cached_challenge.score(&phenotype1);
        assert_eq!(challenge.get_evaluations(), 3);
    }

    #[test]
    fn test_thread_local_cached_challenge() {
        let challenge = TestChallenge::new(50);
        let cached_challenge = ThreadLocalCachedChallenge::new(challenge.clone());

        // First evaluation should calculate the score
        let phenotype1 = TestPhenotype { value: 10 };
        let score1 = cached_challenge.score(&phenotype1);
        assert_eq!(challenge.get_evaluations(), 1);

        // Second evaluation of the same phenotype should use the cache
        let phenotype2 = TestPhenotype { value: 10 };
        let score2 = cached_challenge.score(&phenotype2);
        assert_eq!(challenge.get_evaluations(), 1); // Still 1, not 2
        assert_eq!(score1, score2);

        // Different phenotype should calculate a new score
        let phenotype3 = TestPhenotype { value: 20 };
        let score3 = cached_challenge.score(&phenotype3);
        assert_eq!(challenge.get_evaluations(), 2);
        assert_ne!(score1, score3);

        // Test cache size
        assert_eq!(cached_challenge.cache_size(), 2);

        // Test clear cache
        cached_challenge.clear_cache();
        assert_eq!(cached_challenge.cache_size(), 0);

        // After clearing, should calculate again
        let _score4 = cached_challenge.score(&phenotype1);
        assert_eq!(challenge.get_evaluations(), 3);
    }

    #[test]
    fn test_with_cache() {
        let challenge = TestChallenge::new(50);

        // Create a pre-populated cache
        let mut cache = HashMap::new();
        cache.insert(10, 0.5);

        let cached_challenge = CachedChallenge::with_cache(challenge.clone(), cache);

        // Should use the pre-populated cache value
        let phenotype = TestPhenotype { value: 10 };
        let score = cached_challenge.score(&phenotype);
        assert_eq!(score, 0.5);
        assert_eq!(challenge.get_evaluations(), 0); // No evaluation needed

        // Test get_cache
        let cache = cached_challenge.get_cache();
        assert_eq!(cache.len(), 1);
        assert_eq!(*cache.get(&10).unwrap(), 0.5);
    }
}
