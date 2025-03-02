# GenAlg

A flexible, high-performance genetic algorithm library written in Rust.

## Overview

GenAlg is a modern, thread-safe genetic algorithm framework designed for flexibility, performance, and ease of use. It provides a robust foundation for implementing evolutionary algorithms to solve optimization problems across various domains.

### Key Features

- **Thread-safe**: Designed for parallel processing with `Send` and `Sync` traits
- **High Performance**: Optimized for speed with thread-local random number generation
- **Flexible**: Adaptable to a wide range of optimization problems
- **Extensible**: Easy to implement custom phenotypes, fitness functions, and breeding strategies

## Quick Start

Here's a simple example that optimizes a function to find a target value:

```rust
use genalg::{
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::OrdinaryStrategy,
};

// Define your phenotype (the solution representation)
#[derive(Clone, Debug)]
struct MyPhenotype {
    value: f64,
}

impl Phenotype for MyPhenotype {
    fn crossover(&mut self, other: &Self) {
        // Implement crossover (e.g., averaging values)
        self.value = (self.value + other.value) / 2.0;
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Implement mutation (e.g., adding random noise)
        let delta = rng.fetch_uniform(-1.0, 1.0, 1).front().unwrap();
        self.value += *delta as f64 / 10.0;
    }
}

// Define your fitness function
struct MyChallenge {
    target: f64,
}

impl Challenge<MyPhenotype> for MyChallenge {
    fn score(&self, phenotype: &MyPhenotype) -> f64 {
        // Higher score is better (inverse of distance to target)
        1.0 / (phenotype.value - self.target).abs().max(0.001)
    }
}

fn main() {
    // Initialize components
    let mut rng = RandomNumberGenerator::new();
    let starting_value = MyPhenotype { value: 0.0 };
    let options = EvolutionOptions::default();
    let challenge = MyChallenge { target: 42.0 };
    let strategy = OrdinaryStrategy::default();
    
    // Create and run the evolution
    let launcher = EvolutionLauncher::new(strategy, challenge);
    let result = launcher.evolve(&options, starting_value, &mut rng).unwrap();
    
    println!("Best solution: {:?}, Fitness: {}", result.pheno, result.score);
}
```

## Core Concepts

### Phenotype

The `Phenotype` trait defines the interface for types that represent individuals in the evolutionary algorithm. Implement this trait for your custom solution representation:

```rust
impl Phenotype for MyPhenotype {
    fn crossover(&mut self, other: &Self) {
        // Combine genetic material with another individual
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Introduce random changes
    }
    
    // Optional: Override for better performance in parallel contexts
    fn mutate_thread_local(&mut self) {
        // Implement mutation using thread-local RNG
    }
}
```

### Challenge

The `Challenge` trait defines how to evaluate the fitness of phenotypes:

```rust
impl Challenge<MyPhenotype> for MyChallenge {
    fn score(&self, phenotype: &MyPhenotype) -> f64 {
        // Calculate and return fitness score (higher is better)
    }
}
```

### Breeding Strategies

GenAlg provides two built-in breeding strategies:

1. **OrdinaryStrategy**: A basic breeding strategy where the first parent is considered the winner of the previous generation.

2. **BoundedBreedStrategy**: Similar to `OrdinaryStrategy` but imposes bounds on phenotypes during evolution.

## Advanced Usage

### Bounded Evolution

For problems with constraints, use the `BoundedBreedStrategy` with the `Magnitude` trait:

```rust
impl Magnitude<MyPhenotype> for MyPhenotype {
    fn magnitude(&self) -> f64 {
        // Return the current magnitude
        self.value.abs()
    }

    fn min_magnitude(&self) -> f64 {
        // Return the minimum allowed magnitude
        0.0
    }

    fn max_magnitude(&self) -> f64 {
        // Return the maximum allowed magnitude
        100.0
    }
}

// Then use BoundedBreedStrategy
let strategy = BoundedBreedStrategy::default();
```

### Parallel Processing

GenAlg automatically uses parallel processing for fitness evaluation and breeding when the population size is large enough. You can customize the threshold:

```rust
let strategy = OrdinaryStrategy::new_with_threshold(500); // Use parallel processing when offspring count >= 500
```

## Performance Optimization

For best performance in parallel contexts, implement the `mutate_thread_local` method in your `Phenotype` implementation:

```rust
fn mutate_thread_local(&mut self) {
    // Use ThreadLocalRng for thread-local mutation
    let delta = ThreadLocalRng::gen_range(-0.1..0.1);
    self.value += delta;
}
```

## Benchmarking

GenAlg includes benchmarks to measure performance:

```bash
cargo bench
``` 