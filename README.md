# GenAlg

A flexible, high-performance genetic algorithm library written in Rust.

## Overview

GenAlg is a modern, thread-safe genetic algorithm framework designed for flexibility, performance, and ease of use. It provides a robust foundation for implementing evolutionary algorithms to solve optimization problems across various domains.

### Key Features

- **High Performance**: Optimized for speed with thread-local random number generation
- **Flexible**: Adaptable to a wide range of optimization problems
- **Extensible**: Easy to implement custom phenotypes, fitness functions, and breeding strategies
- **Local Search Integration**: Enhance solutions with integrated local search algorithms
- **Parallel Processing**: Automatic parallelization for large populations using Rayon

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
    breeding::OrdinaryStrategy,
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

GenAlg provides three built-in breeding strategies:

1. **OrdinaryStrategy**: A basic breeding strategy where the first parent is considered the winner of the previous generation.

2. **BoundedBreedStrategy**: Similar to `OrdinaryStrategy` but imposes bounds on phenotypes during evolution.

3. **CombinatorialBreedStrategy**: Specialized strategy for combinatorial optimization problems, supporting constraint handling through repair and penalties.
   ```rust
   use genalg::breeding::combinatorial::{CombinatorialBreedStrategy, CombinatorialBreedConfig};
   
   // Create a configuration with repair and penalties
   let config = CombinatorialBreedConfig::builder()
       .repair_probability(0.8)
       .max_repair_attempts(20)
       .use_penalties(true)
       .penalty_weight(5.0)
       .build();
       
   // Create the strategy
   let mut breed_strategy = CombinatorialBreedStrategy::new(config);
   
   // Add constraints (see the Combinatorial Breeding Strategy section for details)
   ```

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
let hill_climbing = HillClimbing::new(10, 10).unwrap(); // 10 iterations max, 10 neighbors to evaluate

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

1. **HillClimbing**: A random restart hill climbing algorithm that iteratively makes small improvements.

   ```rust
   use genalg::local_search::HillClimbing;
   
   // Basic hill climbing with default neighbors
   let hill_climbing = HillClimbing::new(10, 10).unwrap();
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
   hybrid.add_algorithm(HillClimbing::new(5, 10).unwrap())
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
use  genalg::breeding::{BoundedBreedStrategy, Magnitude};

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
    breeding::BreedStrategy,
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
        // ...
    }
}
```

### Combinatorial Breeding Strategy

GenAlg provides a specialized breeding strategy for combinatorial optimization problems through the `CombinatorialBreedStrategy`. This strategy is designed to handle discrete solution spaces with complex constraints effectively.

#### What are Combinatorial Optimization Problems?

Combinatorial optimization problems involve finding an optimal arrangement, grouping, ordering, or selection of discrete objects. Examples include:

- **Assignment Problems**: Assigning resources to tasks
- **Routing Problems**: Finding optimal paths (e.g., Traveling Salesman Problem)
- **Scheduling Problems**: Arranging tasks in time with dependencies
- **Packing Problems**: Fitting items into containers with capacity constraints

#### Using CombinatorialBreedStrategy

The `CombinatorialBreedStrategy` offers powerful constraint-handling capabilities:

```rust
use genalg::{
    breeding::combinatorial::CombinatorialBreedStrategy,
    breeding::combinatorial::CombinatorialBreedConfig,
    constraints::{Constraint, ConstraintViolation},
    phenotype::Phenotype,
};

// Create a configuration
let config = CombinatorialBreedConfig::builder()
    .repair_probability(0.8)
    .max_repair_attempts(10)
    .use_penalties(true)
    .penalty_weight(5.0)
    .build();

// Create the breeding strategy
let mut breed_strategy = CombinatorialBreedStrategy::new(config);

// Add constraints
breed_strategy.add_constraint(MyConstraint::new());
```

#### Configuration Options

The `CombinatorialBreedStrategy` can be configured with the following options:

```rust
// Default configuration
let default_config = CombinatorialBreedConfig::default();

// Custom configuration with builder pattern
let config = CombinatorialBreedConfig::builder()
    .repair_probability(0.8)     // Probability of attempting repair (default: 0.5)
    .max_repair_attempts(20)     // Maximum repair attempts (default: 10)
    .use_penalties(true)         // Whether to use penalties (default: false)
    .penalty_weight(5.0)         // Weight applied to penalties (default: 1.0)
    .build();
```

#### Constraint Handling Approaches

The strategy supports two complementary approaches to handling constraints:

1. **Repair-based Approach**: Invalid solutions are repaired during breeding
   - Controlled by `repair_probability` and `max_repair_attempts`
   - Attempts to fix invalid solutions by applying constraint repair operations

2. **Penalty-based Approach**: Invalid solutions are penalized during fitness evaluation
   - Controlled by `use_penalties` and `penalty_weight`
   - Reduces the fitness of solutions that violate constraints

These approaches can be used separately or in combination:

```rust
// Repair-only approach
let repair_config = CombinatorialBreedConfig::builder()
    .repair_probability(0.9)
    .max_repair_attempts(20)
    .use_penalties(false)
    .build();

// Penalty-only approach
let penalty_config = CombinatorialBreedConfig::builder()
    .repair_probability(0.0)
    .use_penalties(true)
    .penalty_weight(10.0)
    .build();

// Combined approach
let combined_config = CombinatorialBreedConfig::builder()
    .repair_probability(0.7)
    .max_repair_attempts(15)
    .use_penalties(true)
    .penalty_weight(3.0)
    .build();
```

#### Adding Constraints

To add constraints to your breeding strategy:

```rust
// Create a constraint
struct UniqueValuesConstraint;

impl Constraint<MyPhenotype> for UniqueValuesConstraint {
    fn check(&self, phenotype: &MyPhenotype) -> Vec<ConstraintViolation> {
        // Check if the phenotype violates the constraint
        // Return a list of violations or empty list if valid
    }
    
    fn repair(&self, phenotype: &mut MyPhenotype) -> bool {
        // Try to repair the phenotype
        // Return true if successful, false otherwise
    }
    
    fn repair_with_rng(
        &self,
        phenotype: &mut MyPhenotype,
        rng: &mut RandomNumberGenerator,
    ) -> bool {
        // Try to repair using randomization
        // Return true if successful, false otherwise
    }
}

// Add the constraint to the strategy
let mut strategy = CombinatorialBreedStrategy::new(config);
strategy.add_constraint(UniqueValuesConstraint);
```

#### Built-in Combinatorial Constraints

GenAlg provides several built-in constraints for common combinatorial problems:

1. **UniqueElementsConstraint**: Ensures all elements in a collection are unique
   
   ```rust
   use genalg::constraints::combinatorial::UniqueElementsConstraint;
   
   // Create a constraint that ensures a Vec<usize> contains unique values
   let unique_constraint = UniqueElementsConstraint::new(
       "UniqueValues",
       |phenotype: &MyPhenotype| phenotype.as_ref().clone()
   ).unwrap();
   
   breed_strategy.add_constraint(unique_constraint);
   ```

2. **CompleteAssignmentConstraint**: Ensures all required keys have a value
   
   ```rust
   use genalg::constraints::combinatorial::CompleteAssignmentConstraint;
   use std::collections::{HashMap, HashSet};
   
   // Required keys that must be assigned
   let required_keys: HashSet<usize> = (0..10).collect();
   
   // Create a constraint that ensures all tasks are assigned
   let assignment_constraint = CompleteAssignmentConstraint::new(
       "AllTasksAssigned",
       |phenotype: &MyAssignmentPhenotype| phenotype.get_assignments(),
       required_keys
   ).unwrap();
   
   breed_strategy.add_constraint(assignment_constraint);
   ```

3. **CapacityConstraint**: Ensures bins don't exceed their capacity
   
   ```rust
   use genalg::constraints::combinatorial::CapacityConstraint;
   
   // Create a constraint that ensures bins don't exceed capacity
   let capacity_constraint = CapacityConstraint::new(
       "BinCapacity",
       |phenotype: &MyBinPackingPhenotype| phenotype.get_bin_assignments(),
       |bin| bin.capacity
   ).unwrap();
   
   breed_strategy.add_constraint(capacity_constraint);
   ```

4. **DependencyConstraint**: Ensures dependencies between elements are respected
   
   ```rust
   use genalg::constraints::combinatorial::DependencyConstraint;
   
   // Define task dependencies (before, after) pairs
   let dependencies = vec![(1, 2), (2, 3), (1, 4)];
   
   // Create a constraint that ensures task dependencies are respected
   let dependency_constraint = DependencyConstraint::new(
       "TaskDependencies",
       |phenotype: &MySchedulingPhenotype| phenotype.get_task_sequence(),
       dependencies
   ).unwrap();
   
   breed_strategy.add_constraint(dependency_constraint);
   ```

#### Using Penalty-Adjusted Challenges

When using the penalty-based approach, you need to create a penalty-adjusted challenge:

```rust
use genalg::constraints::PenaltyAdjustedChallenge;

// Original challenge
let original_challenge = MyChallenge::new();

// Create a penalty-adjusted challenge
let penalty_challenge = breed_strategy.create_penalty_challenge(original_challenge);

// Use the penalty-adjusted challenge in the evolution launcher
let launcher = EvolutionLauncher::new(
    breed_strategy,
    selection_strategy,
    local_search_manager,
    penalty_challenge  // Use penalty-adjusted challenge
);
```

#### Complete Example: Traveling Salesman Problem

Here's a complete example showing how to use the combinatorial breeding strategy for a Traveling Salesman Problem:

```rust
use genalg::{
    breeding::combinatorial::{CombinatorialBreedStrategy, CombinatorialBreedConfig},
    constraints::combinatorial::UniqueElementsConstraint,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::TournamentSelection,
};
use std::fmt::Debug;

// Define a phenotype representing a route
#[derive(Clone, Debug)]
struct Route {
    cities: Vec<usize>,
}

impl Phenotype for Route {
    fn crossover(&mut self, other: &Self) {
        // Order crossover (OX)
        if self.cities.is_empty() || other.cities.is_empty() {
            return;
        }
        
        // Select a random segment from other parent
        let start = rand::random::<usize>() % self.cities.len();
        let end = start + (rand::random::<usize>() % (self.cities.len() - start));
        
        // Copy the segment
        let segment: Vec<usize> = other.cities[start..=end].to_vec();
        
        // Create a list of cities not in the segment
        let remaining: Vec<usize> = self.cities.iter()
            .filter(|&city| !segment.contains(city))
            .cloned()
            .collect();
        
        // Create the new route
        let mut new_route = Vec::with_capacity(self.cities.len());
        let mut remaining_idx = 0;
        
        for i in 0..self.cities.len() {
            if i >= start && i <= end {
                new_route.push(segment[i - start]);
            } else {
                new_route.push(remaining[remaining_idx]);
                remaining_idx += 1;
            }
        }
        
        self.cities = new_route;
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Swap mutation
        if self.cities.len() < 2 {
            return;
        }
        
        let idx1 = (rng.fetch_uniform(0.0, self.cities.len() as f32, 1)[0] as usize) % self.cities.len();
        let mut idx2 = (rng.fetch_uniform(0.0, self.cities.len() as f32, 1)[0] as usize) % self.cities.len();
        
        while idx1 == idx2 {
            idx2 = (rng.fetch_uniform(0.0, self.cities.len() as f32, 1)[0] as usize) % self.cities.len();
        }
        
        self.cities.swap(idx1, idx2);
    }
}

// Fitness function calculating total distance
#[derive(Clone)]
struct TSPChallenge {
    distances: Vec<Vec<f64>>,
}

impl Challenge<Route> for TSPChallenge {
    fn score(&self, phenotype: &Route) -> f64 {
        if phenotype.cities.is_empty() {
            return f64::NEG_INFINITY;
        }
        
        let mut total_distance = 0.0;
        for i in 0..phenotype.cities.len() {
            let from = phenotype.cities[i];
            let to = phenotype.cities[(i + 1) % phenotype.cities.len()];
            total_distance += self.distances[from][to];
        }
        
        // Return negative distance (higher score is better)
        -total_distance
    }
}

fn main() -> genalg::error::Result<()> {
    // Create distance matrix for 5 cities
    let distances = vec![
        vec![0.0, 10.0, 15.0, 20.0, 25.0],
        vec![10.0, 0.0, 35.0, 25.0, 30.0],
        vec![15.0, 35.0, 0.0, 30.0, 10.0],
        vec![20.0, 25.0, 30.0, 0.0, 15.0],
        vec![25.0, 30.0, 10.0, 15.0, 0.0],
    ];
    
    // Create the challenge
    let challenge = TSPChallenge { distances };
    
    // Create initial solution
    let initial_route = Route { cities: vec![0, 1, 2, 3, 4] };
    
    // Create breeding strategy with constraints
    let config = CombinatorialBreedConfig::builder()
        .repair_probability(0.9)
        .max_repair_attempts(20)
        .build();
        
    let mut breed_strategy = CombinatorialBreedStrategy::new(config);
    
    // Add constraint to ensure each city appears exactly once
    let unique_constraint = UniqueElementsConstraint::new(
        "UniqueCities",
        |route: &Route| route.cities.clone()
    )?;
    
    breed_strategy.add_constraint(unique_constraint);
    
    // Create selection strategy
    let selection = TournamentSelection::new(3);
    
    // Create launcher
    let launcher = EvolutionLauncher::new(
        breed_strategy,
        selection,
        None,  // No local search
        challenge
    );
    
    // Configure and run evolution
    let options = EvolutionOptions::builder()
        .num_generations(100)
        .population_size(50)
        .num_offspring(100)
        .log_level(LogLevel::Info)
        .build();
        
    let result = launcher
        .configure(options, initial_route)
        .run()?;
        
    println!("Best route: {:?}", result.pheno.cities);
    println!("Total distance: {}", -result.score);
    
    Ok(())
}
```

### Fitness Caching

GenAlg provides built-in caching functionality to improve performance by avoiding redundant fitness evaluations. This is particularly useful in scenarios with:

- Expensive fitness evaluations
- Frequent occurrences of similar phenotypes
- Large populations or many generations

#### How Caching Works

Caching in GenAlg is implemented as a wrapper around your challenge:

1. When a phenotype is evaluated, the system first checks if its fitness value is in the cache
2. If found, the cached value is returned without recalculating
3. If not found, the actual fitness function is called and the result is stored in the cache

#### Enabling Caching

There are two ways to enable caching:

1. **Via Evolution Options** (simplest):

```rust
let options = EvolutionOptions::builder()
    .num_generations(100)
    .population_size(50)
    .use_caching(true)  // Enable caching
    .cache_type(CacheType::Global)  // Choose cache type
    .build();
    
let result = launcher
    .configure(options, starting_value)
    .run()
    .unwrap();
```

2. **Direct Caching Challenge Creation** (for more control):

```rust
use genalg::evolution::caching_challenge::CachingChallenge;

// Original challenge
let challenge = MyChallenge { target: 42.0 };

// With global cache (thread-safe, shared across all threads)
let global_cached_challenge = challenge.with_global_cache();

// Or with thread-local cache (separate cache per thread)
let thread_local_cached_challenge = challenge.with_thread_local_cache();

// Or at runtime based on configuration
let cache_type = CacheType::Global; // or CacheType::ThreadLocal
let cached_challenge = challenge.with_cache(cache_type);
```

#### Implementing the CacheKey Trait

For caching to work, your phenotype must implement the `CacheKey` trait:

```rust
use genalg::caching::CacheKey;

impl CacheKey for MyPhenotype {
    // Choose an appropriate type that uniquely identifies phenotypes with the same fitness
    type Key = Vec<f64>;  // Could be any Eq + Hash + Clone + Debug + Send + Sync type

    fn cache_key(&self) -> Self::Key {
        // Return a value that uniquely identifies phenotypes with the same fitness
        // For simple phenotypes, this could be the direct representation
        self.values.clone()
        
        // For floating-point values, consider rounding to handle precision issues
        // self.values.iter().map(|v| (v * 1000.0).round() as i64).collect()
    }
}
```

Guidelines for implementing `CacheKey`:

1. The key must uniquely identify phenotypes that would have the same fitness score
2. The key generation should be computationally efficient
3. For floating-point values, consider rounding to handle precision issues
4. For complex phenotypes, only include the parts that affect fitness

#### Cache Types

GenAlg provides two cache implementations:

1. **Global Cache** (`CacheType::Global`):
   - Single cache shared across all threads
   - Protected by a mutex
   - Maximizes cache reuse
   - Good for problems with low thread contention or single-threaded use

2. **Thread-Local Cache** (`CacheType::ThreadLocal`):
   - Separate cache for each thread
   - No mutex contention
   - Better performance for highly parallel workloads
   - May have some redundant evaluations across threads

#### Performance Considerations

- **Cache Growth**: The cache grows unbounded by default. For long-running evolutions, consider clearing it periodically.
- **Thread Contention**: For highly parallel workloads, thread-local caching may outperform global caching due to reduced mutex contention.
- **Key Generation**: Ensure the `cache_key()` method is efficient, as it's called for every fitness evaluation.
- **Memory Usage**: Monitor memory usage if caching a very large number of phenotypes.

#### Example: Advanced Caching

```rust
use genalg::{
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, CacheType},
    phenotype::Phenotype,
    caching::CacheKey,
    rng::RandomNumberGenerator,
};

// Define a phenotype with CacheKey implementation
#[derive(Clone, Debug)]
struct Vector {
    values: Vec<f64>,
}

impl Phenotype for Vector {
    fn crossover(&mut self, other: &Self) {
        // Implementation omitted for brevity
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Implementation omitted for brevity
    }
}

impl CacheKey for Vector {
    type Key = Vec<i64>;

    fn cache_key(&self) -> Self::Key {
        // Round to nearest 0.001 to handle floating-point imprecision
        self.values.iter()
            .map(|v| (v * 1000.0).round() as i64)
            .collect()
    }
}

// Expensive challenge that benefits from caching
#[derive(Clone)]
struct ExpensiveChallenge;

impl Challenge<Vector> for ExpensiveChallenge {
    fn score(&self, phenotype: &Vector) -> f64 {
        // Simulate a computationally expensive fitness function
        std::thread::sleep(std::time::Duration::from_millis(10));
        phenotype.values.iter().sum::<f64>()
    }
}

// Run evolution with caching
fn main() {
    let starting_vector = Vector { values: vec![0.0, 0.0, 0.0] };
    let challenge = ExpensiveChallenge;
    
    let options = EvolutionOptions::builder()
        .num_generations(100)
        .population_size(20)
        .num_offspring(40)
        .use_caching(true)
        .cache_type(CacheType::ThreadLocal) // Better for parallel execution
        .build();
        
    let launcher = EvolutionLauncher::builder()
        .with_challenge(challenge)
        .build()
        .unwrap();
        
    let result = launcher
        .configure(options, starting_vector)
        .run()
        .unwrap();
        
    println!("Best solution: {:?}, Fitness: {}", result.pheno, result.score);
}
```

### Serialization Support

GenAlg provides optional serialization support using `serde`.

#### Enabling Serialization

To use serialization, you need to enable the `serde` feature in your `Cargo.toml`:

```toml
[dependencies]
genalg = { version = "0.1.0", features = ["serde"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"  # Or any other serde format you prefer
```

#### Serializable Types

The following GenAlg types support serialization when the feature is enabled:

- `EvolutionResult`
- `EvolutionOptions`
- `LogLevel`
- `CacheType`
- `ConstraintViolation`
- `ConstraintError`

#### Making Your Phenotypes Serializable

To make your phenotypes serializable, implement the `Serialize` and `Deserialize` traits from serde:

```rust
use genalg::phenotype::Phenotype;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MyPhenotype {
    value: f64,
}

impl Phenotype for MyPhenotype {
    // ... implementations of crossover and mutate ...
}

// The SerializablePhenotype trait is automatically implemented
// for any type that implements both Phenotype and the serde traits
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