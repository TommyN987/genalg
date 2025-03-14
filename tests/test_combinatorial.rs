use genalg::{
    caching::{CacheKey, CachedChallenge},
    constraints::{Constraint, ConstraintViolation},
    error::{GeneticError, Result},
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{AllIndividualsStrategy, HillClimbing},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
    strategy::combinatorial::{CombinatorialBreedConfig, CombinatorialBreedStrategy},
};

// Define a phenotype for the test
#[derive(Clone, Debug, PartialEq, Eq)]
struct AssignmentPhenotype {
    // Each design item (index) is assigned to a part option (value)
    assignments: Vec<usize>,
    // The available part options for each design item
    available_options: Vec<Vec<usize>>,
}

impl AssignmentPhenotype {
    fn new(available_options: Vec<Vec<usize>>) -> Self {
        // Initially assign each design item to its first available option
        let assignments = available_options
            .iter()
            .map(|options| options.first().copied().unwrap_or(0))
            .collect();

        Self {
            assignments,
            available_options,
        }
    }

    fn get_total_quantity(&self, part_option: usize) -> usize {
        self.assignments
            .iter()
            .filter(|&&assignment| assignment == part_option)
            .count()
    }

    fn is_valid_assignment(&self) -> bool {
        self.assignments
            .iter()
            .enumerate()
            .all(|(idx, &assignment)| self.available_options[idx].contains(&assignment))
    }
}

impl Phenotype for AssignmentPhenotype {
    fn crossover(&mut self, other: &Self) {
        // Crossover: take some assignments from the other parent
        for i in 0..self.assignments.len() {
            if i % 2 == 0 && self.available_options[i].contains(&other.assignments[i]) {
                self.assignments[i] = other.assignments[i];
            }
        }
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Mutate: randomly change some assignments
        if self.assignments.is_empty() {
            return;
        }

        // Choose a random design item to mutate
        let idx = (rng
            .fetch_uniform(0.0, self.assignments.len() as f32, 1)
            .front()
            .unwrap()
            * self.assignments.len() as f32) as usize
            % self.assignments.len();

        // Choose a random available option
        let options = &self.available_options[idx];
        if options.is_empty() {
            return;
        }

        let option_idx = (rng
            .fetch_uniform(0.0, options.len() as f32, 1)
            .front()
            .unwrap()
            * options.len() as f32) as usize
            % options.len();

        // Assign the new option
        self.assignments[idx] = options[option_idx];
    }
}

impl CacheKey for AssignmentPhenotype {
    type Key = Vec<usize>;

    fn cache_key(&self) -> Self::Key {
        self.assignments.clone()
    }
}

// Define a constraint for valid assignments
#[derive(Debug, Clone)]
struct ValidAssignmentConstraint;

impl Constraint<AssignmentPhenotype> for ValidAssignmentConstraint {
    fn check(&self, phenotype: &AssignmentPhenotype) -> Vec<ConstraintViolation> {
        let mut violations = Vec::new();

        for (idx, &assignment) in phenotype.assignments.iter().enumerate() {
            if !phenotype.available_options[idx].contains(&assignment) {
                violations.push(ConstraintViolation::new(
                    "ValidAssignment",
                    format!("Invalid assignment {} for design item {}", assignment, idx),
                ));
            }
        }

        violations
    }

    fn repair(&self, phenotype: &mut AssignmentPhenotype) -> bool {
        let mut modified = false;

        for idx in 0..phenotype.assignments.len() {
            if !phenotype.available_options[idx].contains(&phenotype.assignments[idx]) {
                // Assign the first available option
                if let Some(&first_option) = phenotype.available_options[idx].first() {
                    phenotype.assignments[idx] = first_option;
                    modified = true;
                }
            }
        }

        modified
    }
}

// Define a challenge for the assignment problem
#[derive(Debug, Clone)]
struct AssignmentChallenge {
    // Price breaks for each part option: (quantity, price)
    price_breaks: Vec<Vec<(usize, f64)>>,
}

impl AssignmentChallenge {
    fn new(price_breaks: Vec<Vec<(usize, f64)>>) -> Self {
        Self { price_breaks }
    }

    fn calculate_cost(&self, phenotype: &AssignmentPhenotype) -> f64 {
        let mut total_cost = 0.0;

        // Calculate the total quantity for each part option
        let mut quantities = vec![0; self.price_breaks.len()];
        for &assignment in &phenotype.assignments {
            if assignment < quantities.len() {
                quantities[assignment] += 1;
            }
        }

        // Calculate the cost for each part option
        for (part_option, quantity) in quantities.iter().enumerate() {
            if part_option < self.price_breaks.len() && *quantity > 0 {
                let breaks = &self.price_breaks[part_option];

                // Find the applicable price break
                let mut price = 0.0;
                for &(break_quantity, break_price) in breaks {
                    if *quantity >= break_quantity {
                        price = break_price;
                    } else {
                        break;
                    }
                }

                // Add to total cost
                total_cost += price * (*quantity as f64);
            }
        }

        total_cost
    }
}

impl Challenge<AssignmentPhenotype> for AssignmentChallenge {
    fn score(&self, phenotype: &AssignmentPhenotype) -> f64 {
        // Check if the phenotype is valid
        if !phenotype.is_valid_assignment() {
            return 0.0;
        }

        // Calculate the cost (lower is better)
        let cost = self.calculate_cost(phenotype);

        // Return the inverse (higher is better)
        1_000_000.0 / (cost + 1.0)
    }
}

#[test]
fn test_combinatorial_optimization() -> Result<()> {
    // Define the problem
    let available_options = vec![
        vec![0, 1, 2], // Design item 0 can use part options 0, 1, or 2
        vec![1, 2],    // Design item 1 can use part options 1 or 2
        vec![0, 2],    // Design item 2 can use part options 0 or 2
        vec![0, 1],    // Design item 3 can use part options 0 or 1
        vec![1, 2],    // Design item 4 can use part options 1 or 2
    ];

    // Define price breaks: (quantity, price)
    let price_breaks = vec![
        vec![(1, 10.0), (3, 9.0), (5, 8.0)],   // Part option 0
        vec![(1, 12.0), (3, 11.0), (5, 10.0)], // Part option 1
        vec![(1, 8.0), (3, 7.5), (5, 7.0)],    // Part option 2
    ];

    // Create the initial phenotype
    let initial_phenotype = AssignmentPhenotype::new(available_options.clone());

    // Create the challenge
    let challenge = AssignmentChallenge::new(price_breaks);

    // Wrap with caching
    let cached_challenge =
        CachedChallenge::<AssignmentPhenotype, AssignmentChallenge>::new(challenge);

    // Create the constraint
    let constraint = ValidAssignmentConstraint;

    // Create the breeding strategy
    let config = CombinatorialBreedConfig::builder()
        .repair_probability(1.0) // Always repair
        .max_repair_attempts(10)
        .build();

    let mut strategy = CombinatorialBreedStrategy::<AssignmentPhenotype>::new(config);
    strategy.add_constraint(constraint);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);
    options.set_num_generations(20);
    options.set_log_level(LogLevel::None);

    // Create the selection strategy
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        AssignmentPhenotype,
        CombinatorialBreedStrategy<AssignmentPhenotype>,
        ElitistSelection,
        HillClimbing,
        CachedChallenge<AssignmentPhenotype, AssignmentChallenge>,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, cached_challenge);

    // Run the evolution with a fixed seed
    let result = launcher
        .configure(options, initial_phenotype)
        .with_seed(42)
        .run()?;

    // Check the result
    assert!(result.score > 0.0, "Result score should be positive");

    let result_pheno = result.pheno;
    assert!(
        result_pheno.is_valid_assignment(),
        "Result should be a valid assignment"
    );

    // The optimal solution should use part option 2 as much as possible
    let optimal_count = result_pheno.get_total_quantity(2);
    assert!(
        optimal_count >= 3,
        "At least 3 out of 5 should use part option 2, got {}",
        optimal_count
    );

    Ok(())
}

#[derive(Clone, Debug)]
struct PermutationPhenotype {
    values: Vec<usize>,
}

impl PermutationPhenotype {
    fn new(values: Vec<usize>) -> Self {
        Self { values }
    }

    fn get_values(&self) -> &Vec<usize> {
        &self.values
    }
}

impl Phenotype for PermutationPhenotype {
    fn crossover(&mut self, other: &Self) {
        // Simple order crossover (OX)
        let n = self.values.len();
        let mut rng = RandomNumberGenerator::new();

        // Select two random crossover points
        let points = rng.fetch_uniform(0.0, (n - 1) as f32, 2);
        let mut point1 = points[0] as usize;
        let mut point2 = points[1] as usize;

        // Ensure point1 <= point2
        if point1 > point2 {
            std::mem::swap(&mut point1, &mut point2);
        }

        // Create a copy of the current values
        let original_values = self.values.clone();

        // Copy the segment between crossover points from self
        let segment: Vec<usize> = original_values[point1..=point2].to_vec();

        // Create a list of values from other that are not in the segment
        let remaining: Vec<usize> = other
            .values
            .iter()
            .filter(|&x| !segment.contains(x))
            .cloned()
            .collect();

        // Fill in the values
        let mut remaining_iter = remaining.iter();
        for i in 0..n {
            if i >= point1 && i <= point2 {
                // Keep the segment from self
                continue;
            } else {
                // Fill with values from other that are not in the segment
                if let Some(&val) = remaining_iter.next() {
                    self.values[i] = val;
                }
            }
        }
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        // Swap mutation - swap two random positions
        let n = self.values.len();
        let positions = rng.fetch_uniform(0.0, (n - 1) as f32, 2);
        let pos1 = positions[0] as usize;
        let pos2 = positions[1] as usize;

        self.values.swap(pos1, pos2);
    }
}

#[derive(Clone, Debug)]
struct TSPChallenge {
    // Distance matrix for a simple TSP problem
    distances: Vec<Vec<f64>>,
}

impl TSPChallenge {
    fn new() -> Self {
        // Create a simple distance matrix for testing
        // This represents distances between 5 cities
        let distances = vec![
            vec![0.0, 10.0, 15.0, 20.0, 25.0],
            vec![10.0, 0.0, 35.0, 25.0, 30.0],
            vec![15.0, 35.0, 0.0, 30.0, 10.0],
            vec![20.0, 25.0, 30.0, 0.0, 15.0],
            vec![25.0, 30.0, 10.0, 15.0, 0.0],
        ];

        Self { distances }
    }
}

impl Challenge<PermutationPhenotype> for TSPChallenge {
    fn score(&self, phenotype: &PermutationPhenotype) -> f64 {
        let route = phenotype.get_values();
        let n = route.len();

        // Calculate total distance of the route
        let mut total_distance = 0.0;
        for i in 0..n {
            let from = route[i];
            let to = route[(i + 1) % n]; // Wrap around to the start

            total_distance += self.distances[from][to];
        }

        // Convert to a score (higher is better)
        1000.0 / (1.0 + total_distance)
    }
}

#[test]
fn test_combinatorial() {
    // Create a starting permutation
    let starting_value = PermutationPhenotype::new(vec![0, 1, 2, 3, 4]);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);
    options.set_num_generations(20);
    options.set_log_level(LogLevel::None);

    // Create the TSP challenge
    let challenge = TSPChallenge::new();

    // Create the combinatorial breed strategy with default options
    let combinatorial_options = CombinatorialBreedConfig::default();
    let strategy = CombinatorialBreedStrategy::new(combinatorial_options);
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        PermutationPhenotype,
        CombinatorialBreedStrategy<PermutationPhenotype>,
        ElitistSelection,
        HillClimbing,
        TSPChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, challenge);

    // Run the evolution with a fixed seed
    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run();

    assert!(
        result.is_ok(),
        "Evolution failed: {:?}",
        result
            .err()
            .unwrap_or_else(|| GeneticError::Other("Unknown error".to_string()))
    );

    let winner = result.unwrap();

    // Check that the result is a valid permutation
    let values = winner.pheno.get_values();
    assert_eq!(values.len(), 5, "Result should have 5 values");

    // Check that all values 0-4 are present exactly once
    for i in 0..5 {
        assert!(
            values.contains(&i),
            "Value {} is missing from the result",
            i
        );
        assert_eq!(
            values.iter().filter(|&&x| x == i).count(),
            1,
            "Value {} appears more than once in the result",
            i
        );
    }
}

#[test]
fn test_combinatorial_with_custom_options() {
    // Create a starting permutation
    let starting_value = PermutationPhenotype::new(vec![0, 1, 2, 3, 4]);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);
    options.set_num_generations(20);
    options.set_log_level(LogLevel::None);

    // Create the TSP challenge
    let challenge = TSPChallenge::new();

    // Create the combinatorial breed strategy with custom options using the builder pattern
    let combinatorial_options = CombinatorialBreedConfig::builder()
        .repair_probability(0.8)
        .use_penalties(true)
        .penalty_weight(2.0)
        .max_repair_attempts(20)
        .build();

    let strategy = CombinatorialBreedStrategy::new(combinatorial_options);
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        PermutationPhenotype,
        CombinatorialBreedStrategy<PermutationPhenotype>,
        ElitistSelection,
        HillClimbing,
        TSPChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, challenge);

    // Run the evolution with a fixed seed
    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run();

    assert!(
        result.is_ok(),
        "Evolution failed: {:?}",
        result
            .err()
            .unwrap_or_else(|| GeneticError::Other("Unknown error".to_string()))
    );

    let winner = result.unwrap();

    // Check that the result is a valid permutation
    let values = winner.pheno.get_values();
    assert_eq!(values.len(), 5, "Result should have 5 values");

    // Check that all values 0-4 are present exactly once
    for i in 0..5 {
        assert!(
            values.contains(&i),
            "Value {} is missing from the result",
            i
        );
        assert_eq!(
            values.iter().filter(|&&x| x == i).count(),
            1,
            "Value {} appears more than once in the result",
            i
        );
    }
}
