use criterion::{black_box, criterion_group, criterion_main, Criterion};

use genalg::{
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
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

fn bench_bounded(c: &mut Criterion) {
    let starting_value = XCoordinate::new(7.0);
    let mut rng = RandomNumberGenerator::new();
    let evol_options = EvolutionOptions::new(100, LogLevel::None, 2, 20);
    let challenge = XCoordinateChallenge::new(2.0);
    let strategy = BoundedBreedStrategy::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        BoundedBreedStrategy<XCoordinate>,
        XCoordinateChallenge,
    > = EvolutionLauncher::new(strategy, challenge);

    c.bench_function("bounded_evolution", |b| {
        b.iter(|| {
            let result = launcher.evolve(
                black_box(&evol_options),
                black_box(starting_value),
                black_box(&mut rng),
            );
            assert!(result.is_ok());
            result.unwrap()
        })
    });
}

criterion_group!(benches, bench_bounded);
criterion_main!(benches);
