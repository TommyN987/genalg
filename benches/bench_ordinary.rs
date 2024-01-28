use criterion::{black_box, criterion_group, criterion_main, Criterion};

use genalg::{
    evolution::{EvolutionOptions, LogLevel},
    rng::RandomNumberGenerator,
    strategy::{BreedStrategy, OrdinaryStrategy},
};

#[allow(unused)]
fn bench_ordinary(c: &mut Criterion) {
    let strategy = OrdinaryStrategy;

    let mut rng = RandomNumberGenerator::new();
    let evol_options = EvolutionOptions::new(10000, LogLevel::Minimal, 2, 20);

    #[derive(Debug, Copy, Clone)]
    struct MockPhenotype;

    impl genalg::phenotype::Phenotype for MockPhenotype {
        fn crossover(&mut self, other: &Self) {}
        fn mutate(&mut self, rng: &mut RandomNumberGenerator) {}
    }

    let mut parents = Vec::<MockPhenotype>::new();

    parents.extend((0..5).into_iter().map(|value| {
        let child = MockPhenotype;
        child
    }));

    c.bench_function("breed", |b| {
        b.iter(|| {
            strategy.breed(
                black_box(&parents),
                black_box(&evol_options),
                black_box(&mut rng),
            )
        })
    });
}

criterion_group!(benches, bench_ordinary);
criterion_main!(benches);
