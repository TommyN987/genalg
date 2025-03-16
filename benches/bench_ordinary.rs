use criterion::{black_box, criterion_group, criterion_main, Criterion};
use genalg::{
    breeding::{BreedStrategy, OrdinaryStrategy},
    evolution::{EvolutionOptions, LogLevel},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
};

fn bench_ordinary(c: &mut Criterion) {
    let strategy = OrdinaryStrategy::default();
    let mut rng = RandomNumberGenerator::new();

    let mut group = c.benchmark_group("ordinary_breeding");
    for size in [10, 100, 1000, 10000].iter() {
        group.bench_function(&format!("ordinary_breeding_{}", size), |b| {
            b.iter(|| {
                let options = EvolutionOptions::new(10, LogLevel::None, 5, *size);
                let parents = vec![
                    TestPhenotype { value: 1.0 },
                    TestPhenotype { value: 2.0 },
                    TestPhenotype { value: 3.0 },
                    TestPhenotype { value: 4.0 },
                    TestPhenotype { value: 5.0 },
                ];

                let result = strategy.breed(
                    black_box(&parents),
                    black_box(&options),
                    black_box(&mut rng),
                );
                assert!(result.is_ok());
            })
        });
    }
    group.finish();
}

#[derive(Clone, Debug)]
struct TestPhenotype {
    value: f64,
}

impl Phenotype for TestPhenotype {
    fn crossover(&mut self, other: &Self) {
        self.value = (self.value + other.value) / 2.0;
    }

    fn mutate(&mut self, rng: &mut RandomNumberGenerator) {
        let delta = *rng.fetch_uniform(-1.0, 1.0, 1).front().unwrap() as f64;
        self.value += delta;
    }

    fn mutate_thread_local(&mut self) {
        use genalg::rng::ThreadLocalRng;
        let delta = ThreadLocalRng::gen_range(-1.0..1.0);
        self.value += delta;
    }
}

criterion_group!(benches, bench_ordinary);
criterion_main!(benches);
