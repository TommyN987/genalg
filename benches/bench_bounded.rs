use criterion::{black_box, criterion_group, criterion_main, Criterion};

use genalg::{
    breeding::{BoundedBreedStrategy, Magnitude},
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{AllIndividualsStrategy, HillClimbing},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
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

#[derive(Clone)]
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
    let evol_options = EvolutionOptions::new(100, LogLevel::None, 2, 20);
    let challenge = XCoordinateChallenge::new(2.0);
    let breed_strategy = BoundedBreedStrategy::default();
    let selection_strategy = ElitistSelection::default();

    let launcher: EvolutionLauncher<
        XCoordinate,
        BoundedBreedStrategy<XCoordinate>,
        ElitistSelection,
        HillClimbing,
        XCoordinateChallenge,
        AllIndividualsStrategy,
    > = EvolutionLauncher::new(breed_strategy, selection_strategy, None, challenge);

    c.bench_function("bounded_evolution", |b| {
        b.iter(|| {
            let result = launcher
                .configure(black_box(evol_options.clone()), black_box(starting_value))
                .with_seed(42)
                .run();
            assert!(result.is_ok());
            result.unwrap()
        })
    });
}

criterion_group!(benches, bench_bounded);
criterion_main!(benches);
