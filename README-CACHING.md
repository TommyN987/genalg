# Fitness Caching in GenAlg

This document explains how to use the fitness caching functionality in the GenAlg library.

## Overview

Fitness caching can significantly improve performance by avoiding redundant fitness evaluations. This is especially useful when:

- Fitness evaluations are computationally expensive
- Similar phenotypes appear frequently during evolution
- You're working with a large population size
- You're running many generations

## How Caching Works

When caching is enabled, the library stores the fitness scores of evaluated phenotypes in a hash map. When a phenotype with the same "cache key" is encountered again, the cached score is returned instead of recalculating it.

The library provides two types of caching:

1. **Global Cache**: A single cache shared across all threads, protected by a mutex.
2. **Thread-Local Cache**: A separate cache for each thread, which can avoid contention in highly parallel scenarios.

## Enabling Caching

There are two ways to enable caching:

### 1. Using `CachingLauncherFactory`

For phenotypes that implement `CacheKey`, you can use the `CachingLauncherFactory` trait:

```rust
use genalg::{
    evolution::{Challenge, CachingLauncherFactory, EvolutionOptions},
    phenotype::Phenotype,
    caching::CacheKey,
    breeding::OrdinaryStrategy,
    selection::ElitistSelection,
    local_search::{HillClimbing, AllIndividualsStrategy},
};

// Define your phenotype with CacheKey implementation
#[derive(Clone, Debug)]
struct MyPhenotype {
    values: Vec<f64>,
}

impl Phenotype for MyPhenotype {
    // Implementation...
}

impl CacheKey for MyPhenotype {
    type Key = Vec<i32>;
    
    fn cache_key(&self) -> Self::Key {
        // Convert floats to integers for more reliable caching
        self.values.iter().map(|v| (*v * 1000.0) as i32).collect()
    }
}

// Your challenge
#[derive(Clone)]
struct MyChallenge;

impl Challenge<MyPhenotype> for MyChallenge {
    fn score(&self, phenotype: &MyPhenotype) -> f64 {
        // Expensive computation...
    }
}

// Create components
let breed_strategy = OrdinaryStrategy::default();
let selection_strategy = ElitistSelection::default();
let hill_climbing = HillClimbing::new(10, 10)?;
let app_strategy = AllIndividualsStrategy::new();
let challenge = MyChallenge;

// Create a launcher with global caching
let launcher = CachingLauncherFactory::create_with_global_cache(
    breed_strategy,
    selection_strategy,
    Some(hill_climbing),
    app_strategy,
    challenge,
)?;

// Run evolution
let options = EvolutionOptions::default();
let starting_value = MyPhenotype { values: vec![0.0, 0.0, 0.0] };
let result = launcher.configure(options, starting_value).run()?;
```

### 2. For Specialized Use Cases

For more specialized use cases, you can directly create cached challenges:

```rust
use genalg::{
    evolution::{Challenge, EvolutionLauncher},
    caching::{CacheKey, CachedChallenge},
};

// Your challenge
let challenge = MyChallenge;

// Create a cached version
let cached_challenge = CachedChallenge::new(challenge);

// Create a launcher with the cached challenge
let launcher = EvolutionLauncher::new(
    breed_strategy,
    selection_strategy,
    local_search_manager,
    cached_challenge,
);
```

## Implementing `CacheKey`

To use caching, your phenotype must implement the `CacheKey` trait:

```rust
use genalg::caching::CacheKey;

impl CacheKey for MyPhenotype {
    type Key = Vec<i32>;
    
    fn cache_key(&self) -> Self::Key {
        // Generate a key that uniquely identifies equivalent phenotypes
        // (phenotypes that would have the same fitness score)
        self.values.iter().map(|v| (*v * 1000.0) as i32).collect()
    }
}
```

Guidelines for implementing `cache_key()`:

1. The key should uniquely identify phenotypes with identical fitness scores
2. The key must be `Eq + Hash + Clone + Debug + Send + Sync`
3. For floating-point values, consider rounding or quantizing to handle precision issues
4. Keep the key generation efficient, as it's called for every fitness evaluation
5. Balance precision (to avoid false cache hits) with flexibility (to maximize cache hits)

## Performance Considerations

1. **Cache Size**: The cache grows unbounded by default. For long-running evolutions or large populations, consider calling `clear_cache()` periodically.

2. **Thread Contention**: For highly parallel workloads, consider using `ThreadLocalCachedChallenge` to avoid mutex contention.

3. **Key Generation**: Make sure your `cache_key()` implementation is efficient, as it's called for every fitness evaluation.

4. **Memory Usage**: Monitor memory usage if you're caching a very large number of phenotypes.

## Choosing the Right Cache Type

- **Global Cache**: Good for most use cases, especially when fitness evaluation is very expensive.
- **Thread-Local Cache**: Better for highly parallel workloads with many threads, but may use more memory.

## Example: Clearing the Cache

You can access and clear the cache as needed:

```rust
// Access the cached challenge from the launcher
let cached_challenge = launcher.get_challenge();

// Clear the cache
cached_challenge.clear_cache();

// Or get the cache size
let cache_size = cached_challenge.cache_size();
println!("Cache contains {} entries", cache_size);
```

## Advanced: Cache Warm-up

For some applications, you might want to pre-populate the cache with known good solutions:

```rust
use std::collections::HashMap;

// Create initial cache
let mut initial_cache = HashMap::new();
initial_cache.insert(known_phenotype.cache_key(), known_score);

// Create cached challenge with pre-populated cache
let cached_challenge = CachedChallenge::with_cache(challenge, initial_cache);
```

This can be useful for:
- Resuming an evolution from a checkpoint
- Starting with known good solutions
- Sharing evaluated solutions across multiple evolution runs 