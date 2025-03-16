use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use genalg::{
    breeding::{BoundedBreedStrategy, Magnitude, OrdinaryStrategy},
    evolution::{Challenge, EvolutionLauncher, EvolutionOptions, LogLevel},
    local_search::{AllIndividualsStrategy, HillClimbing},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
};
use rayon::prelude::*;

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

    fn mutate_thread_local(&mut self) {
        use genalg::rng::ThreadLocalRng;
        let delta = ThreadLocalRng::gen_range(-100.0..100.0);
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

#[derive(Debug)]
pub struct XCoordinateChallenge {
    target: f64,
}

impl Clone for XCoordinateChallenge {
    fn clone(&self) -> Self {
        Self {
            target: self.target,
        }
    }
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

fn evaluate_fitness_sequential(
    candidates: &[XCoordinate],
    challenge: &XCoordinateChallenge,
) -> Vec<f64> {
    candidates
        .iter()
        .map(|candidate| challenge.score(candidate))
        .collect()
}

fn evaluate_fitness_parallel(
    candidates: &[XCoordinate],
    challenge: &XCoordinateChallenge,
) -> Vec<f64> {
    candidates
        .par_iter()
        .map(|candidate| challenge.score(candidate))
        .collect()
}

fn breed_sequential(
    parents: &[XCoordinate],
    winner: &XCoordinate,
    num_offspring: usize,
    rng: &mut RandomNumberGenerator,
) -> Vec<XCoordinate> {
    let mut children = Vec::with_capacity(num_offspring);
    children.push(winner.clone());

    for parent in parents.iter().skip(1) {
        let mut child = winner.clone();
        child.crossover(parent);
        child.mutate(rng);
        children.push(child);
    }

    let num_mutation_only = num_offspring.saturating_sub(parents.len());
    for _ in 0..num_mutation_only {
        let mut child = winner.clone();
        child.mutate(rng);
        children.push(child);
    }

    children
}

fn breed_parallel(
    parents: &[XCoordinate],
    winner: &XCoordinate,
    num_offspring: usize,
    _rng: &mut RandomNumberGenerator, // Not used directly anymore
) -> Vec<XCoordinate> {
    let mut children = Vec::with_capacity(num_offspring);
    children.push(winner.clone());

    let crossover_parents: Vec<&XCoordinate> = parents.iter().skip(1).collect();
    let num_mutation_only = num_offspring.saturating_sub(parents.len());

    if !crossover_parents.is_empty() {
        let crossover_children: Vec<XCoordinate> = crossover_parents
            .into_par_iter()
            .map(|parent| {
                let mut child = winner.clone();
                child.crossover(parent);

                child.mutate_thread_local();

                child
            })
            .collect();

        children.extend(crossover_children);
    }

    if num_mutation_only > 0 {
        let mutation_children: Vec<XCoordinate> = (0..num_mutation_only)
            .into_par_iter()
            .map(|_| {
                let mut child = winner.clone();

                child.mutate_thread_local();

                child
            })
            .collect();

        children.extend(mutation_children);
    }

    children
}

fn bench_fitness_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("fitness_evaluation");
    let challenge = XCoordinateChallenge::new(2.0);

    for size in [10, 100, 1000, 10000].iter() {
        let candidates: Vec<XCoordinate> = (0..*size).map(|i| XCoordinate::new(i as f64)).collect();

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &candidates,
            |b, candidates| {
                b.iter(|| evaluate_fitness_sequential(black_box(candidates), black_box(&challenge)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &candidates,
            |b, candidates| {
                b.iter(|| evaluate_fitness_parallel(black_box(candidates), black_box(&challenge)))
            },
        );
    }

    group.finish();
}

fn bench_breeding(c: &mut Criterion) {
    let mut group = c.benchmark_group("breeding");
    let mut rng = RandomNumberGenerator::new();
    let winner = XCoordinate::new(5.0);

    for size in [10, 100, 1000].iter() {
        let parents: Vec<XCoordinate> = (0..*size).map(|i| XCoordinate::new(i as f64)).collect();
        let num_offspring = *size * 2; // Double the parents for offspring

        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            &parents,
            |b, parents| {
                b.iter(|| {
                    breed_sequential(
                        black_box(parents),
                        black_box(&winner),
                        black_box(num_offspring),
                        black_box(&mut rng),
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel", size),
            &parents,
            |b, parents| {
                b.iter(|| {
                    breed_parallel(
                        black_box(parents),
                        black_box(&winner),
                        black_box(num_offspring),
                        black_box(&mut rng),
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_evolution_strategies(c: &mut Criterion) {
    let sizes = [50, 500];
    let mut group = c.benchmark_group("evolution_strategies");

    for size in sizes.iter() {
        let starting_value = XCoordinate::new(5.0);
        let evol_options = EvolutionOptions::new(10, LogLevel::None, 10, *size);
        let challenge = XCoordinateChallenge::new(2.0);
        let selection_strategy = ElitistSelection::default();

        let ordinary_strategy = OrdinaryStrategy::default();
        let ordinary_launcher: EvolutionLauncher<
            XCoordinate,
            OrdinaryStrategy,
            ElitistSelection,
            HillClimbing,
            XCoordinateChallenge,
            AllIndividualsStrategy,
        > = EvolutionLauncher::new(
            ordinary_strategy,
            selection_strategy.clone(),
            None,
            challenge.clone(),
        );

        group.bench_with_input(
            BenchmarkId::new("ordinary", size),
            &evol_options,
            |b, options| {
                b.iter(|| {
                    let result = ordinary_launcher
                        .configure(black_box(options.clone()), black_box(starting_value))
                        .with_seed(42)
                        .run();
                    assert!(result.is_ok());
                })
            },
        );

        let bounded_strategy = BoundedBreedStrategy::default();
        let bounded_launcher: EvolutionLauncher<
            XCoordinate,
            BoundedBreedStrategy<XCoordinate>,
            ElitistSelection,
            HillClimbing,
            XCoordinateChallenge,
            AllIndividualsStrategy,
        > = EvolutionLauncher::new(
            bounded_strategy,
            selection_strategy.clone(),
            None,
            challenge.clone(),
        );

        group.bench_with_input(
            BenchmarkId::new("bounded", size),
            &evol_options,
            |b, options| {
                b.iter(|| {
                    let result = bounded_launcher
                        .configure(black_box(options.clone()), black_box(starting_value))
                        .with_seed(42)
                        .run();
                    assert!(result.is_ok());
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fitness_evaluation,
    bench_breeding,
    bench_evolution_strategies
);
criterion_main!(benches);
