//! # Caching Module
//!
//! This module provides caching mechanisms for fitness evaluations to improve performance.
//! Caching is particularly useful for expensive fitness functions or when the same phenotypes
//! are evaluated multiple times during the evolution process.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;

/// A trait for phenotypes that can be used as cache keys.
///
/// This trait is required for phenotypes that will be used with the `CachedChallenge`.
/// It provides a way to generate a cache key for a phenotype, which is used to look up
/// cached fitness values.
pub trait CacheKey: Phenotype {
    /// The type of the cache key.
    type Key: Eq + Hash + Clone + Debug + Send + Sync;

    /// Generates a cache key for this phenotype.
    ///
    /// The cache key should uniquely identify the phenotype for fitness evaluation purposes.
    /// Phenotypes that would have the same fitness score should have the same cache key.
    fn cache_key(&self) -> Self::Key;
}

/// A wrapper around a challenge that caches fitness evaluations.
///
/// This wrapper caches the results of fitness evaluations to avoid redundant calculations.
/// It is particularly useful for expensive fitness functions or when the same phenotypes
/// are evaluated multiple times during the evolution process.
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
    /// Creates a new cached challenge wrapping the given challenge.
    pub fn new(challenge: C) -> Self {
        Self {
            challenge,
            cache: Arc::new(Mutex::new(HashMap::new())),
            _marker: PhantomData,
        }
    }

    /// Creates a new cached challenge with a pre-populated cache.
    pub fn with_cache(challenge: C, cache: HashMap<P::Key, f64>) -> Self {
        Self {
            challenge,
            cache: Arc::new(Mutex::new(cache)),
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the wrapped challenge.
    pub fn inner(&self) -> &C {
        &self.challenge
    }

    /// Returns the number of cached fitness evaluations.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Returns a copy of the cache.
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

/// A thread-local cache for fitness evaluations.
///
/// This cache is designed to be used in parallel contexts where each thread
/// has its own cache to avoid contention.
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
    pub fn get(&self, key: &P::Key) -> Option<f64> {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .and_then(|cache| cache.get(key).copied())
    }

    /// Inserts a fitness value into the cache.
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
    pub fn clear(&self) {
        if let Some(cell) = self.cache.get() {
            if let Ok(mut cache) = cell.try_borrow_mut() {
                cache.clear();
            }
        }
    }

    /// Returns the number of cached fitness evaluations for the current thread.
    pub fn len(&self) -> usize {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .map_or(0, |cache| cache.len())
    }

    /// Returns `true` if the cache for the current thread is empty.
    pub fn is_empty(&self) -> bool {
        self.cache
            .get()
            .and_then(|cell| cell.try_borrow().ok())
            .map_or(true, |cache| cache.is_empty())
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

/// A wrapper around a challenge that uses a thread-local cache for fitness evaluations.
///
/// This wrapper is designed for parallel contexts where each thread has its own cache
/// to avoid contention.
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
    /// Creates a new thread-local cached challenge wrapping the given challenge.
    pub fn new(challenge: C) -> Self {
        Self {
            challenge,
            cache: Arc::new(ThreadLocalCache::new()),
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the wrapped challenge.
    pub fn inner(&self) -> &C {
        &self.challenge
    }

    /// Clears the cache for the current thread.
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// Returns the number of cached fitness evaluations for the current thread.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
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
