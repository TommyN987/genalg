use genalg::{
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    strategy::AdjustStrategy,
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
    let mut rng = RandomNumberGenerator::new();
    let starting_value = XCoordinate::new(100000.0);
    let options = EvolutionOptions::default();
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = AdjustStrategy::default();
    let launcher: EvolutionLauncher<XCoordinate, AdjustStrategy<XCoordinate>, XCoordinateChallenge> =
        EvolutionLauncher::new(strategy, challenge);
    let winner = launcher.evolve(&options, starting_value, &mut rng).unwrap();
    assert!((winner.pheno.get_x() - 2.0).abs() < 1e-2);
}
