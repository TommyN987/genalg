# GenAlg

A flexible, high-performance genetic algorithm library written in Rust.

## Overview

GenAlg is a modern, thread-safe genetic algorithm framework designed for flexibility, performance, and ease of use. It provides a robust foundation for implementing evolutionary algorithms to solve optimization problems across various domains.

### Key Features

- **Thread-safe**: Designed for parallel processing with `Send` and `Sync` traits
- **High Performance**: Optimized for speed with thread-local random number generation
- **Flexible**: Adaptable to a wide range of optimization problems
- **Extensible**: Easy to implement custom phenotypes, fitness functions, and breeding strategies
- **Parallel Processing**: Automatic parallelization for large populations using Rayon

## API Stability

GenAlg is currently at version 0.1.0, which indicates that the API is still evolving. While we strive to maintain backward compatibility, breaking changes may occur in minor version updates until we reach version 1.0.0. After reaching 1.0.0, we will follow semantic versioning principles:

- **Major version** changes (1.0.0 → 2.0.0) may include breaking API changes
- **Minor version** changes (1.0.0 → 1.1.0) will be backward compatible but may add new functionality
- **Patch version** changes (1.0.0 → 1.0.1) will include bug fixes and performance improvements

We recommend pinning to a specific version in your `Cargo.toml` if API stability is critical for your project.

## Installation

Add GenAlg to your `Cargo.toml`:

```toml
[dependencies]
genalg = "0.1.0"
```

## Quick Start

Here's a simple example that optimizes a function to find a target value:

```rust
use genalg::{
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
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
        let values = rng.fetch_uniform(-1.0, 1.0, 1);
        let delta = values.front().unwrap();
        self.value += *delta as f64 / 10.0;
    }
    
    // Optional: Override for better performance in parallel contexts
    fn mutate_thread_local(&mut self) {
        // Custom implementation using thread-local RNG
        use genalg::rng::ThreadLocalRng;
        let delta = ThreadLocalRng::gen_range(-0.1..0.1);
        self.value += delta;
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
    let starting_value = MyPhenotype { value: 0.0 };
    let options = EvolutionOptions::builder()
        .num_generations(100)
        .log_level(LogLevel::Minimal)
        .population_size(10)
        .num_offspring(50)
        .parallel_threshold(1000)
        .build();
    let challenge = MyChallenge { target: 42.0 };
    let strategy = OrdinaryStrategy::default();
    
    // Create and run the evolution
    let launcher = EvolutionLauncher::new(strategy, challenge);
    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)  // Optional: Set a specific seed
        .run()
        .unwrap();
    
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
        // Introduce random changes using the provided RNG
    }
    
    // Optional: Override for better performance in parallel contexts
    fn mutate_thread_local(&mut self) {
        // Implement mutation using thread-local RNG for better parallel performance
        // Default implementation calls mutate() with a temporary RNG
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

### Evolution Options

Configure the evolution process with `EvolutionOptions`:

```rust
let options = EvolutionOptions::new(
    100,                // Number of generations
    LogLevel::Minimal,  // Logging level
    10,                 // Population size
    50,                 // Number of offspring per generation
);

// Or with parallel threshold
let options = EvolutionOptions::new_with_threshold(
    100,                // Number of generations
    LogLevel::Minimal,  // Logging level
    10,                 // Population size
    50,                 // Number of offspring per generation
    1000,               // Parallel threshold (min items to process in parallel)
);
```

### Breeding Strategies

GenAlg provides two built-in breeding strategies:

1. **OrdinaryStrategy**: A basic breeding strategy where the first parent is considered the winner of the previous generation.

2. **BoundedBreedStrategy**: Similar to `OrdinaryStrategy` but imposes bounds on phenotypes during evolution.

## Advanced Usage

### Bounded Evolution

For problems with constraints, use the `BoundedBreedStrategy` with the `Magnitude` trait:

```rust
use genalg::strategy::{BoundedBreedStrategy, Magnitude};

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
let strategy = BoundedBreedStrategy::new(1000); // 1000 max development attempts
```

### Parallel Processing

GenAlg automatically uses parallel processing for fitness evaluation and breeding when the population size exceeds the parallel threshold. Configure this in your `EvolutionOptions`:

```rust
let mut options = EvolutionOptions::default();
options.set_parallel_threshold(500); // Use parallel processing when population >= 500
```

### Thread-Local Random Number Generation

For optimal performance in parallel contexts, GenAlg provides thread-local random number generation:

```rust
use genalg::rng::ThreadLocalRng;

// Generate a random number in a range
let value = ThreadLocalRng::gen_range(0.0..1.0);

// Generate multiple random numbers
let numbers = ThreadLocalRng::fetch_uniform(0.0, 1.0, 5);
```

The `Phenotype` trait includes a default implementation of `mutate_thread_local()` that uses this feature, but you can override it for better performance:

```rust
fn mutate_thread_local(&mut self) {
    // Custom implementation using ThreadLocalRng
    let delta = ThreadLocalRng::gen_range(-0.1..0.1);
    self.value += delta;
}
```

## Error Handling

GenAlg provides a comprehensive error handling system:

```rust
use genalg::error::{GeneticError, Result, ResultExt, OptionExt};

fn my_function() -> Result<()> {
    // Return specific errors
    if something_wrong {
        return Err(GeneticError::Configuration("Invalid parameter".to_string()));
    }
    
    // Convert standard errors with context
    std::fs::File::open("config.txt").context("Failed to open config file")?;
    
    // Convert Option to Result with custom error
    let best = candidates.iter().max().ok_or_else_genetic(|| 
        GeneticError::EmptyPopulation
    )?;
    
    Ok(())
}
```

## Performance Optimization Tips

1. **Implement `mutate_thread_local()`** for your phenotypes to avoid mutex overhead in parallel processing. This can provide significant performance improvements for large populations.

2. **Tune the parallel threshold** in `EvolutionOptions` based on your specific problem and hardware. The optimal threshold depends on:
   - CPU core count and performance
   - Memory bandwidth
   - Complexity of your fitness function and mutation operations
   - Size of your phenotype data structures

3. **Use efficient data structures** in your phenotype implementation to minimize memory usage and improve cache locality.

4. **Profile your fitness function** as it's often the performance bottleneck in genetic algorithms.

5. **Consider strategy selection** based on your problem characteristics:
   - `OrdinaryStrategy` generally performs better for very large populations
   - `BoundedBreedStrategy` adds constraints but may have higher overhead for certain problems

6. **Benchmark your specific use case** to find the optimal configuration. Use the built-in benchmarks as a starting point:

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench bench_parallel
```

## License

This project is licensed under the [MIT license](http://opensource.org/licenses/MIT)

### Contribution

Contributions are welcome! Please open an issue or submit a pull request.