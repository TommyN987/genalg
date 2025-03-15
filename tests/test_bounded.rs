use genalg::{
    error::GeneticError,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{AllIndividualsStrategy, HillClimbing},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
    breeding::{BoundedBreedStrategy, Magnitude},
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
        let delta = *rng.fetch_uniform(-0.5, 0.5, 1).front().unwrap() as f64;
        self.x += delta;

        // Ensure we stay within bounds after mutation
        if self.x < 3.0 {
            self.x = 3.0;
        } else if self.x > 10.0 {
            self.x = 10.0;
        }
    }
}

impl Magnitude<XCoordinate> for XCoordinate {
    fn magnitude(&self) -> f64 {
        self.x.abs()
    }

    fn min_magnitude(&self) -> f64 {
        3.0
    }

    fn max_magnitude(&self) -> f64 {
        10.0
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

        // Simple scoring function - higher is better when closer to target
        let delta = (x - self.target).abs();
        if delta < 0.0001 {
            return 10000.0; // Very high score for exact matches
        }

        1.0 / (1.0 + delta)
    }
}

#[test]
fn test_bounded() {
    // Start with a value within bounds
    let starting_value = XCoordinate::new(5.0);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(5);
    options.set_num_offspring(5);
    options.set_num_generations(10);
    options.set_log_level(LogLevel::None);

    // Target value within bounds
    let challenge = XCoordinateChallenge::new(4.0);

    // Create the bounded breed strategy
    let strategy = BoundedBreedStrategy::default();
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        XCoordinate,
        BoundedBreedStrategy<XCoordinate>,
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
    // Check that the result is within bounds
    assert!(
        winner.pheno.get_x() >= 3.0 && winner.pheno.get_x() <= 10.0,
        "Result {} is not within bounds [3.0, 10.0]",
        winner.pheno.get_x()
    );

    // Check that it's close to the target
    assert!(
        (winner.pheno.get_x() - 4.0).abs() < 1.0,
        "Result {} is not close enough to target 4.0",
        winner.pheno.get_x()
    );
}

#[test]
fn test_bounded_with_custom_attempts() {
    // Start with a value within bounds
    let starting_value = XCoordinate::new(5.0);

    // Create evolution options
    let mut options = EvolutionOptions::default();
    options.set_population_size(5);
    options.set_num_offspring(5);
    options.set_num_generations(10);
    options.set_log_level(LogLevel::None);

    // Target value within bounds
    let challenge = XCoordinateChallenge::new(4.0);

    // Create the bounded breed strategy with custom attempts
    let strategy = BoundedBreedStrategy::new(500);
    let selection = ElitistSelection::default();

    // Create the launcher
    let launcher: EvolutionLauncher<
        XCoordinate,
        BoundedBreedStrategy<XCoordinate>,
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
    // Check that the result is within bounds
    assert!(
        winner.pheno.get_x() >= 3.0 && winner.pheno.get_x() <= 10.0,
        "Result {} is not within bounds [3.0, 10.0]",
        winner.pheno.get_x()
    );

    // Check that it's close to the target
    assert!(
        (winner.pheno.get_x() - 4.0).abs() < 1.0,
        "Result {} is not close enough to target 4.0",
        winner.pheno.get_x()
    );
}

#[test]
fn test_bounded_with_invalid_options() {
    let starting_value = XCoordinate::new(5.0);
    // Create invalid options with zero population size
    let options = EvolutionOptions::new(10, LogLevel::None, 0, 5);
    let challenge = XCoordinateChallenge::new(4.0);
    let strategy = BoundedBreedStrategy::<XCoordinate>::default();
    let selection = ElitistSelection::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        BoundedBreedStrategy<XCoordinate>,
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
