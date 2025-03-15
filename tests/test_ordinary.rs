use genalg::{
    error::GeneticError,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{AllIndividualsStrategy, HillClimbing},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
    breeding::{BreedStrategy, OrdinaryStrategy},
};

#[derive(Clone, Copy, Debug)]
struct XCoordinate {
    x: f64,
}

impl XCoordinate {
    fn new(x: f64) -> Self {
        Self { x }
    }

    fn get_x(&self) -> f64 {
        self.x
    }
}

impl Phenotype for XCoordinate {
    fn crossover(&mut self, other: &Self) {
        self.x = (self.x + other.x) / 2.0;
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let delta = *rng.fetch_uniform(-100.0, 100.0, 1).front().unwrap() as f64;
        self.x += delta / 100.0;
    }
}

#[derive(Clone, Debug)]
struct XCoordinateChallenge {
    target: f64,
}

impl XCoordinateChallenge {
    fn new(target: f64) -> Self {
        Self { target }
    }
}

impl Challenge<XCoordinate> for XCoordinateChallenge {
    fn score(&self, phenotype: &XCoordinate) -> f64 {
        let x = phenotype.get_x();
        let delta = x - self.target;

        // Avoid division by zero and ensure a valid score
        if delta.abs() < 0.0001 {
            return 10000.0; // Very high score for exact matches
        }

        1.0 / delta.powi(2)
    }
}

#[test]
fn test_ordinary() {
    // Create a starting value
    let starting_value = XCoordinate::new(0.0);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);
    options.set_num_generations(20);
    options.set_log_level(LogLevel::None);

    // Create the challenge with target value 2.0
    let challenge = XCoordinateChallenge::new(2.0);

    // Create the ordinary breed strategy
    let strategy = OrdinaryStrategy::default();
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        XCoordinate,
        OrdinaryStrategy,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
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
    // Check that the result is close to the target
    assert!(
        (winner.pheno.get_x() - 2.0).abs() < 1e-2,
        "Result {} is not close enough to target 2.0",
        winner.pheno.get_x()
    );
}

#[test]
fn test_ordinary_with_invalid_options() {
    let starting_value = XCoordinate::new(0.0);
    // Create invalid options with zero population size
    let options = EvolutionOptions::new(100, genalg::evolution::LogLevel::None, 0, 20);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let selection = ElitistSelection::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        OrdinaryStrategy,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, challenge);

    let result = launcher.configure(options, starting_value).run();
    assert!(result.is_err());

    match result {
        Err(GeneticError::Configuration(msg)) => {
            assert!(msg.contains("Population size cannot be zero"));
        }
        _ => panic!("Expected Configuration error"),
    }
}

#[test]
fn test_ordinary_with_empty_parents() {
    // For this test, we'll manually create a situation with empty parents
    // and verify that the OrdinaryStrategy handles it correctly
    let mut rng = RandomNumberGenerator::from_seed(42);
    let strategy = OrdinaryStrategy::default();
    let empty_parents: Vec<XCoordinate> = Vec::new();

    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);

    let result = strategy.breed(&empty_parents, &options, &mut rng);

    // The strategy should return an error when given empty parents
    assert!(
        result.is_err(),
        "Expected an error when breeding with empty parents"
    );

    // Print the actual error to help diagnose the issue
    if let Err(ref e) = result {
        println!("Actual error: {:?}", e);
    }

    // Check that the error is related to empty parents
    // We don't check the exact error type since it might vary
    match result {
        Err(e) => {
            let error_msg = format!("{:?}", e);
            assert!(
                error_msg.contains("empty") || error_msg.contains("Empty"),
                "Error message should mention empty parents: {:?}",
                e
            );
        }
        _ => panic!("Expected an error related to empty parents"),
    }
}

#[test]
fn test_ordinary_strategy() {
    // Create a starting value
    let starting_value = XCoordinate::new(0.0);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(10);
    options.set_num_offspring(10);
    options.set_num_generations(50); // More generations for better convergence
    options.set_log_level(LogLevel::None);

    // Create the challenge with target value 2.0
    let challenge = XCoordinateChallenge::new(2.0);

    // Create the ordinary breed strategy
    let strategy = OrdinaryStrategy::default();
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        XCoordinate,
        OrdinaryStrategy,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
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
    // Check that the result is close to the target
    assert!(
        (winner.pheno.get_x() - 2.0).abs() < 1e-2,
        "Result {} is not close enough to target 2.0",
        winner.pheno.get_x()
    );
}

#[test]
fn test_ordinary_strategy_error() {
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::new(100, LogLevel::None, 0, 50);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let selection = ElitistSelection::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        OrdinaryStrategy,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, challenge);

    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run();
    assert!(result.is_err());
}

#[test]
fn test_ordinary_strategy_error_offspring() {
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::new(100, LogLevel::None, 10, 0);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let selection = ElitistSelection::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        OrdinaryStrategy,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(strategy, selection, None, challenge);

    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run();
    assert!(result.is_err());
}
