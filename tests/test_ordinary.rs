use genalg::{
    error::GeneticError,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::OrdinaryStrategy,
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
        1.0 / delta.powi(2)
    }
}

#[test]
fn test_ordinary() {
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::default();
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);
    let winner = launcher.configure(options, starting_value).run().unwrap();
    assert!((winner.pheno.get_x() - 2.0).abs() < 1e-2);
}

#[test]
fn test_ordinary_with_invalid_options() {
    let starting_value = XCoordinate::new(0.0);
    // Create invalid options with zero population size
    let options = EvolutionOptions::new(100, genalg::evolution::LogLevel::None, 0, 20);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);

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
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::default();

    // Create a challenge that will produce an empty population
    struct EmptyChallenge;

    impl Challenge<XCoordinate> for EmptyChallenge {
        fn score(&self, _: &XCoordinate) -> f64 {
            // Return a negative score to ensure no candidates are selected
            -1.0
        }
    }

    let challenge = EmptyChallenge;
    let strategy = OrdinaryStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, EmptyChallenge> =
        EvolutionLauncher::new(strategy, challenge);

    // This should not panic, but return an error
    let result = launcher.configure(options, starting_value).run();
    assert!(
        result.is_ok(),
        "Evolution with empty challenge should succeed"
    );
}

#[test]
fn test_ordinary_strategy() {
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::new(100, LogLevel::None, 10, 50);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);
    let winner = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run()
        .unwrap();
    assert!((winner.pheno.get_x() - 2.0).abs() < 1e-2);
}

#[test]
fn test_ordinary_strategy_error() {
    let starting_value = XCoordinate::new(0.0);
    let options = EvolutionOptions::new(100, LogLevel::None, 0, 50);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = OrdinaryStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);
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
    let launcher: EvolutionLauncher<XCoordinate, OrdinaryStrategy, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);
    let result = launcher
        .configure(options, starting_value)
        .with_seed(42)
        .run();
    assert!(result.is_err());
}
