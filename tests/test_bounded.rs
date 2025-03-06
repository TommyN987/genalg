use genalg::{
    error::GeneticError,
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::{BoundedBreedStrategy, Magnitude},
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
fn test_bounded() {
    let starting_value = XCoordinate::new(7.0);
    let options = EvolutionOptions::default();
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = BoundedBreedStrategy::default();
    let launcher = EvolutionLauncher::with_default_selection(strategy, challenge);
    let winner = launcher.configure(options, starting_value).run().unwrap();
    assert!((winner.pheno.get_x() - 3.0).abs() < 1e-2);
}

#[test]
fn test_bounded_with_custom_attempts() {
    let starting_value = XCoordinate::new(7.0);
    let options = EvolutionOptions::default();
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = BoundedBreedStrategy::new(500); // Use fewer attempts
    let launcher = EvolutionLauncher::with_default_selection(strategy, challenge);
    let winner = launcher.configure(options, starting_value).run().unwrap();
    assert!((winner.pheno.get_x() - 3.0).abs() < 1e-2);
}

#[test]
fn test_bounded_with_invalid_options() {
    let starting_value = XCoordinate::new(7.0);
    // Create invalid options with zero population size
    let options = EvolutionOptions::new(100, genalg::evolution::LogLevel::None, 0, 20);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = BoundedBreedStrategy::default();
    let launcher = EvolutionLauncher::with_default_selection(strategy, challenge);

    let result = launcher.configure(options, starting_value).run();
    assert!(result.is_err());

    match result {
        Err(GeneticError::Configuration(msg)) => {
            assert!(msg.contains("Population size cannot be zero"));
        }
        _ => panic!("Expected Configuration error"),
    }
}
