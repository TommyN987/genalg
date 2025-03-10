use genalg::{
    caching::{CacheKey, CachedChallenge},
    constraints::{Constraint, ConstraintViolation},
    error::Result,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{
        application::TopPercentStrategy, HillClimbing, LocalSearchManager,
    },
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::TournamentSelection,
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
    let initial_phenotype = AssignmentPhenotype::new(available_options);

    // Create the challenge
    let challenge = AssignmentChallenge::new(price_breaks);

    // Wrap with caching
    let cached_challenge = CachedChallenge::new(challenge);

    // Create the constraint
    let constraint = ValidAssignmentConstraint;

    // Create the breeding strategy
    let config = CombinatorialBreedConfig::builder()
        .repair_probability(1.0) // Always repair
        .max_repair_attempts(10)
        .build();

    let mut strategy = CombinatorialBreedStrategy::<AssignmentPhenotype>::new(config);
    strategy.add_constraint(constraint);

    // Create the selection strategy
    let selection_strategy = TournamentSelection::new(3);

    // Create the local search algorithm and manager
    let hill_climbing = HillClimbing::new(10);
    let top_percent_strategy = TopPercentStrategy::new_maximizing(0.2); // Apply to top 20%
    let local_search_manager = LocalSearchManager::new(hill_climbing, top_percent_strategy);

    // Create the evolution options
    let options = EvolutionOptions::builder()
        .num_generations(20)
        .log_level(LogLevel::None)
        .population_size(10)
        .num_offspring(20)
        .build();

    // Create the launcher
    let launcher = EvolutionLauncher::new(
        strategy,
        selection_strategy,
        cached_challenge.clone(),
    );

    // Run the evolution
    let result = launcher.configure(options, initial_phenotype).run()?;

    // Check the result
    assert!(result.score > 0.0);
    assert!(result.pheno.is_valid_assignment());

    // The optimal solution should use part option 2 as much as possible
    let optimal_count = result.pheno.get_total_quantity(2);
    assert!(optimal_count >= 3); // At least 3 out of 5 should use part option 2

    Ok(())
}
