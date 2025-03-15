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
- **Local Search Integration**: Enhance solutions with integrated local search algorithms

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
    selection::ElitistSelection,
    local_search::{HillClimbing, AllIndividualsStrategy},
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
#[derive(Clone)]
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
        .log_level(LogLevel::Info)
        .population_size(10)
        .num_offspring(50)
        .parallel_threshold(1000)
        .build();
    let challenge = MyChallenge { target: 42.0 };
    let breed_strategy = OrdinaryStrategy::default();
    let selection_strategy = ElitistSelection::default();
    
    // Create and run the evolution without local search
    let launcher = EvolutionLauncher::new(
        breed_strategy, 
        selection_strategy, 
        None, // No local search
        challenge
    );
    
    let result = launcher
        .configure(options.clone(), starting_value.clone())
        .with_seed(42)  // Optional: Set a specific seed
        .run()
        .unwrap();
    
    println!("Best solution without local search: {:?}, Fitness: {}", result.pheno, result.score);
    
    // Alternatively, use the builder pattern
    let launcher = EvolutionLauncher::builder()
        .with_breed_strategy(OrdinaryStrategy::default())
        .with_selection_strategy(ElitistSelection::default())
        .with_challenge(MyChallenge { target: 42.0 })
        .build()
        .unwrap();
        
    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run()
        .unwrap();
        
    println!("Best solution using builder: {:?}, Fitness: {}", result.pheno, result.score);
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
    LogLevel::Info,     // Logging level
    10,                 // Population size
    50,                 // Number of offspring per generation
);

// Or with parallel threshold
let options = EvolutionOptions::new_with_threshold(
    100,                // Number of generations
    LogLevel::Info,     // Logging level
    10,                 // Population size
    50,                 // Number of offspring per generation
    1000,               // Parallel threshold (min items to process in parallel)
);

// Or use the builder pattern
let options = EvolutionOptions::builder()
    .num_generations(100)
    .log_level(LogLevel::Info)
    .population_size(10)
    .num_offspring(50)
    .parallel_threshold(1000)
    .build();
```

### Breeding Strategies

GenAlg provides two built-in breeding strategies:

1. **OrdinaryStrategy**: A basic breeding strategy where the first parent is considered the winner of the previous generation.

2. **BoundedBreedStrategy**: Similar to `OrdinaryStrategy` but imposes bounds on phenotypes during evolution.

### Selection Strategies

GenAlg provides several built-in selection strategies for choosing parents based on their fitness:

1. **ElitistSelection**: Selects the best individuals based on fitness scores. This strategy implements elitism, which ensures that the best solutions are preserved across generations.

   ```rust
   use genalg::selection::ElitistSelection;
   
   // Default: higher fitness is better, no duplicates allowed
   let selection = ElitistSelection::default();
   
   // For minimization problems (lower fitness is better)
   let selection = ElitistSelection::with_options(false, false);
   
   // Allow duplicates in selection
   let selection = ElitistSelection::with_duplicates(true);
   ```

2. **TournamentSelection**: Randomly selects groups of individuals and picks the best from each group. This provides a balance between exploration and exploitation.

   ```rust
   use genalg::selection::TournamentSelection;
   
   // Tournament size determines selection pressure
   // Larger size = more pressure toward best individuals
   let selection = TournamentSelection::new(3);
   
   // For minimization problems
   let selection = TournamentSelection::with_options(3, false, false);
   ```

3. **RouletteWheelSelection**: Selects individuals with probability proportional to their fitness. Also known as fitness proportionate selection.

   ```rust
   use genalg::selection::RouletteWheelSelection;
   
   // Default: higher fitness is better, no duplicates
   let selection = RouletteWheelSelection::default();
   
   // For minimization problems
   let selection = RouletteWheelSelection::with_options(false, false);
   ```

4. **RankBasedSelection**: Selects individuals based on their rank in the population rather than their absolute fitness. This helps prevent premature convergence when there are a few individuals with much higher fitness.

   ```rust
   use genalg::selection::RankBasedSelection;
   
   // Default: selection pressure of 1.5, higher is better
   let selection = RankBasedSelection::default();
   
   // Custom selection pressure (between 1.0 and 2.0)
   let selection = RankBasedSelection::with_pressure(1.8).unwrap();
   
   // For minimization problems
   let selection = RankBasedSelection::with_options(1.5, false, false).unwrap();
   ```

You can also implement your own selection strategy by implementing the `SelectionStrategy` trait:

```rust
use genalg::{
    error::Result,
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::SelectionStrategy,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
struct MyCustomSelection {
    // Your selection configuration fields
}

impl<P: Phenotype> SelectionStrategy<P> for MyCustomSelection {
    fn select(
        &self,
        population: &[P],
        fitness: &[f64],
        num_to_select: usize,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Vec<P>> {
        // Implement your custom selection logic here
        // ...
    }
}
```

## Advanced Usage

### Local Search Integration

GenAlg supports integrating local search algorithms to enhance the quality of solutions:

```rust
use genalg::{
    local_search::{HillClimbing, AllIndividualsStrategy, LocalSearchManager},
    evolution::EvolutionLauncher,
};

// Create a local search algorithm
let hill_climbing = HillClimbing::new(10).unwrap(); // 10 iterations max

// Create a local search application strategy
let application_strategy = AllIndividualsStrategy::new();

// Create a local search manager
let local_search_manager = LocalSearchManager::new(
    hill_climbing,
    application_strategy
);

// Create the launcher with local search
let launcher = EvolutionLauncher::new(
    breed_strategy,
    selection_strategy,
    Some(local_search_manager),
    challenge
);

// Or enable local search during configuration
let result = launcher
    .configure(options, starting_value)
    .with_local_search() // Enable local search
    .run()
    .unwrap();
```

Available local search algorithms:

1. **HillClimbing**: A simple hill climbing algorithm that iteratively makes small improvements.

   ```rust
   use genalg::local_search::HillClimbing;
   
   // Basic hill climbing with default neighbors
   let hill_climbing = HillClimbing::new(10).unwrap();
   
   // With custom number of neighbors to evaluate
   let hill_climbing = HillClimbing::with_neighbors(10, 5).unwrap();
   ```

2. **SimulatedAnnealing**: Allows accepting worse solutions with decreasing probability over time.

   ```rust
   use genalg::local_search::SimulatedAnnealing;
   
   // Parameters: max iterations, initial temperature, cooling rate
   let simulated_annealing = SimulatedAnnealing::new(100, 1.0, 0.95).unwrap();
   ```

3. **TabuSearch**: Prevents revisiting recently explored solutions.

   ```rust
   use genalg::local_search::TabuSearch;
   
   // Parameters: max iterations, max neighbors, tabu list size
   let tabu_search = TabuSearch::new(50, 10, 20).unwrap();
   ```

4. **HybridLocalSearch**: Combines multiple local search algorithms.

   ```rust
   use genalg::local_search::{HybridLocalSearch, HillClimbing, SimulatedAnnealing};
   
   let mut hybrid = HybridLocalSearch::new();
   hybrid.add_algorithm(HillClimbing::new(5).unwrap())
         .add_algorithm(SimulatedAnnealing::new(10, 0.5, 0.9).unwrap());
   ```

Local search application strategies:

1. **AllIndividualsStrategy**: Applies local search to all individuals.

   ```rust
   use genalg::local_search::AllIndividualsStrategy;
   
   let strategy = AllIndividualsStrategy::new();
   ```

2. **TopNStrategy**: Applies local search to the top N individuals.

   ```rust
   use genalg::local_search::TopNStrategy;
   
   // Apply to top 3 individuals (maximizing)
   let strategy = TopNStrategy::new_maximizing(3);
   
   // Apply to top 3 individuals (minimizing)
   let strategy = TopNStrategy::new_minimizing(3);
   ```

3. **TopPercentStrategy**: Applies local search to a percentage of top individuals.

   ```rust
   use genalg::local_search::TopPercentStrategy;
   
   // Apply to top 20% of individuals (maximizing)
   let strategy = TopPercentStrategy::new_maximizing(0.2).unwrap();
   ```

4. **ProbabilisticStrategy**: Applies local search with a certain probability.

   ```rust
   use genalg::local_search::ProbabilisticStrategy;
   
   // 30% chance of applying local search to each individual
   let strategy = ProbabilisticStrategy::new(0.3).unwrap();
   ```

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

### Implementing Custom Breeding Strategies

One of the key design features of GenAlg is the ability to implement your own custom breeding strategies. This allows you to tailor the evolutionary process to your specific problem domain.

To create a custom breeding strategy, implement the `BreedStrategy` trait:

```rust
use genalg::{
    error::Result,
    evolution::options::EvolutionOptions,
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::BreedStrategy,
};

#[derive(Debug, Clone)]
struct MyCustomStrategy {
    // Your strategy configuration fields
    crossover_rate: f64,
    mutation_rate: f64,
}

impl MyCustomStrategy {
    pub fn new(crossover_rate: f64, mutation_rate: f64) -> Self {
        Self { crossover_rate, mutation_rate }
    }
}

impl<Pheno: Phenotype> BreedStrategy<Pheno> for MyCustomStrategy {
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<Pheno>> {
        // Implement your custom breeding logic here
        // This could include:
        // - Selection mechanisms (tournament, roulette wheel, etc.)
        // - Custom crossover operations
        // - Specialized mutation rates
        // - Elitism strategies
        // - Adaptive parameter adjustments
        
        // Example implementation (simplified):
        let mut children = Vec::with_capacity(evol_options.get_num_offspring());
        
        if parents.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        
        // Use the first parent as a template
        let template = &parents[0];
        
        // Generate offspring
        for _ in 0..evol_options.get_num_offspring() {
            // Select parents using your custom selection method
            let parent1 = select_parent(parents, rng);
            let parent2 = select_parent(parents, rng);
            
            // Create child from parent1
            let mut child = parent1.clone();
            
            // Apply crossover with some probability
            if rng.fetch_uniform(0.0, 1.0, 1).front().unwrap() < &(self.crossover_rate as f32) {
                child.crossover(&parent2);
            }
            
            // Apply mutation with some probability
            if rng.fetch_uniform(0.0, 1.0, 1).front().unwrap() < &(self.mutation_rate as f32) {
                child.mutate(rng);
            }
            
            children.push(child);
        }
        
        Ok(children)
    }
}

// Helper function for parent selection
fn select_parent<Pheno: Phenotype>(parents: &[Pheno], rng: &mut RandomNumberGenerator) -> &Pheno {
    // Simple random selection for this example
    let idx = (rng.fetch_uniform(0.0, parents.len() as f32, 1).front().unwrap() 
               * parents.len() as f32) as usize;
    &parents[idx % parents.len()]
}
```

Once you've implemented your custom strategy, you can use it with the `EvolutionLauncher` just like the built-in strategies:

```rust
// Create your custom strategy
let breed_strategy = MyCustomStrategy::new(0.8, 0.2);
let selection_strategy = ElitistSelection::default();

// Create the launcher with your strategies
let launcher = EvolutionLauncher::new(
    breed_strategy, 
    selection_strategy, 
    None, // No local search
    challenge
);

// Configure and run the evolution
let result = launcher
    .configure(options, starting_value)
    .run()
    .unwrap();
```

This flexibility allows you to implement specialized breeding approaches such as:

- **Island Models**: Evolve multiple sub-populations with occasional migration
- **Age-Based Selection**: Consider the age of individuals in the selection process
- **Niching Methods**: Maintain diversity by promoting solutions in different regions
- **Adaptive Parameter Control**: Dynamically adjust mutation and crossover rates
- **Multi-objective Optimization**: Handle multiple competing objectives

### Combinatorial Optimization

For combinatorial optimization problems, GenAlg provides specialized components:

#### Constraint Handling

Define constraints that solutions must satisfy:

```rust
use genalg::constraints::{Constraint, ConstraintManager, ConstraintViolation};

// Define a constraint
#[derive(Debug, Clone)]
struct UniqueElementsConstraint;

impl<P> Constraint<P> for UniqueElementsConstraint
where
    P: Phenotype + AsRef<Vec<usize>>,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let elements = phenotype.as_ref();
        let mut seen = HashSet::new();
        let mut violations = Vec::new();

        for (idx, element) in elements.iter().enumerate() {
            if !seen.insert(element) {
                violations.push(ConstraintViolation::new(
                    "UniqueElements",
                    format!("Duplicate element {:?} at position {}", element, idx),
                ));
            }
        }

        violations
    }
    
    // Optionally implement repair methods
    fn repair(&self, phenotype: &mut P) -> bool {
        // Repair logic
        false
    }
    
    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // Repair logic with randomness
        false
    }
}

// Use the constraint manager
let mut constraint_manager = ConstraintManager::new();
constraint_manager.add_constraint(UniqueElementsConstraint);

// Check if a solution is valid
let is_valid = constraint_manager.is_valid(&solution);

// Get all constraint violations
let violations = constraint_manager.check_all(&solution);

// Try to repair an invalid solution
let repaired = constraint_manager.repair_all(&solution);
```

#### Combinatorial Breeding Strategy

Use the specialized breeding strategy for combinatorial problems:

```rust
use genalg::strategy::combinatorial::{CombinatorialBreedStrategy, CombinatorialBreedConfig};
use genalg::constraints::{Constraint, ConstraintManager};

// Configure the breeding strategy
let config = CombinatorialBreedConfig::builder()
    .repair_probability(0.8)
    .max_repair_attempts(20)
    .use_penalties(true)
    .penalty_weight(2.5)
    .build();

// Create the breeding strategy
let mut strategy = CombinatorialBreedStrategy::new(config);

// Add constraints
strategy.add_constraint(MyConstraint);

// Use with the evolution launcher
let launcher = EvolutionLauncher::new(
    strategy, 
    selection_strategy, 
    local_search_manager, // Optional
    challenge
);
```

#### Fitness Caching

Improve performance by caching fitness evaluations:

```rust
use genalg::caching::{CacheKey, CachedChallenge, ThreadLocalCachedChallenge};

// Implement CacheKey for your phenotype
impl CacheKey for MyPhenotype {
    type Key = i32;
    
    fn cache_key(&self) -> Self::Key {
        // Generate a unique key for this phenotype
        self.value as i32
    }
}

// Wrap your challenge with caching
let cached_challenge = CachedChallenge::new(challenge);

// Or use thread-local caching for parallel contexts
let thread_local_cached_challenge = ThreadLocalCachedChallenge::new(challenge);

// Use with the evolution launcher
let launcher = EvolutionLauncher::new(
    strategy, 
    selection_strategy, 
    local_search_manager, // Optional
    cached_challenge
);
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

### Logging with Tracing

GenAlg uses the `tracing` crate for structured logging. To enable logging in your application, you need to set up a tracing subscriber:

```rust
use tracing_subscriber::{fmt, EnvFilter};

// Initialize the default subscriber with an environment filter
fn setup_logging() {
    fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
}

// Or with a specific level
fn setup_logging_with_level() {
    fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
}

fn main() {
    // Set up logging before running the evolution
    setup_logging();
    
    // ... rest of your code ...
}
```

You can control the verbosity of GenAlg's logging by:

1. Setting the `LogLevel` in `EvolutionOptions`:
   - `LogLevel::Debug`: Detailed logging including phenotypes and scores
   - `LogLevel::Info`: Basic progress information
   - `LogLevel::None`: No logging output

2. Setting the tracing level through the environment variable:
   ```bash
   RUST_LOG=debug cargo run
   ```

3. Setting the tracing level programmatically:
   ```rust
   tracing_subscriber::fmt()
       .with_max_level(tracing::Level::DEBUG)
       .init();
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

7. **Consider disabling local search** for performance-critical applications where solution quality can be traded for speed.

8. **Use the appropriate local search application strategy** to balance solution quality and performance:
   - `TopNStrategy` or `TopPercentStrategy` for applying local search selectively
   - `ProbabilisticStrategy` for a randomized approach that can be tuned

## License

This project is licensed under the [MIT license](http://opensource.org/licenses/MIT)

### Contribution

Contributions are welcome! Please open an issue or submit a pull request.