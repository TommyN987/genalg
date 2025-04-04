use genalg::{
    breeding::OrdinaryStrategy,
    caching::CacheKey,
    evolution::{
        caching_challenge::CachingChallenge, Challenge, EvolutionLauncher, EvolutionOptions,
    },
    local_search::{AllIndividualsStrategy, HillClimbing, LocalSearchManager},
    phenotype::Phenotype,
    rng::RandomNumberGenerator,
    selection::ElitistSelection,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

// Define a simple phenotype that can be used for testing
#[derive(Clone, Debug, PartialEq)]
struct TestPhenotype {
    value: i32,
}

impl Phenotype for TestPhenotype {
    fn crossover(&mut self, other: &Self) {
        self.value = (self.value + other.value) / 2;
    }

    fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
        self.value += 1;
    }
}

// Implement CacheKey for TestPhenotype
impl CacheKey for TestPhenotype {
    type Key = i32;

    fn cache_key(&self) -> Self::Key {
        self.value
    }
}

// Define a challenge that tracks the number of evaluations
#[derive(Clone)]
struct CostlyChallenge {
    // Use Arc<AtomicUsize> to track evaluations across clones
    evaluations: Arc<AtomicUsize>,
}

impl CostlyChallenge {
    fn new() -> Self {
        Self {
            evaluations: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn get_evaluations(&self) -> usize {
        self.evaluations.load(Ordering::SeqCst)
    }
}

impl Challenge<TestPhenotype> for CostlyChallenge {
    fn score(&self, phenotype: &TestPhenotype) -> f64 {
        // Increment evaluation counter
        self.evaluations.fetch_add(1, Ordering::SeqCst);

        // Simulate an expensive computation
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Simple fitness function: smaller values are better
        -(phenotype.value as f64)
    }
}

#[test]
fn test_direct_caching() {
    let challenge = CostlyChallenge::new();

    // Create a cached version of the challenge
    let cached_challenge = challenge.with_global_cache();

    let phenotype = TestPhenotype { value: 42 };

    // First evaluation should compute
    let score1 = cached_challenge.score(&phenotype);
    assert_eq!(score1, -42.0);
    assert_eq!(challenge.get_evaluations(), 1);

    // Second evaluation should use cache
    let score2 = cached_challenge.score(&phenotype);
    assert_eq!(score2, -42.0);
    assert_eq!(challenge.get_evaluations(), 1); // Still 1, used cache

    // Different phenotype should compute
    let different_phenotype = TestPhenotype { value: 43 };
    let score3 = cached_challenge.score(&different_phenotype);
    assert_eq!(score3, -43.0);
    assert_eq!(challenge.get_evaluations(), 2);
}

#[test]
fn test_thread_local_caching() {
    let challenge = CostlyChallenge::new();

    // Create a thread-local cached version of the challenge
    let cached_challenge = challenge.with_thread_local_cache();

    let phenotype = TestPhenotype { value: 42 };

    // First evaluation should compute
    let score1 = cached_challenge.score(&phenotype);
    assert_eq!(score1, -42.0);
    assert_eq!(challenge.get_evaluations(), 1);

    // Second evaluation should use cache
    let score2 = cached_challenge.score(&phenotype);
    assert_eq!(score2, -42.0);
    assert_eq!(challenge.get_evaluations(), 1); // Still 1, used cache

    // Test across threads
    let thread_challenge = cached_challenge.clone();
    let thread_phenotype = phenotype.clone();
    let handle = std::thread::spawn(move || {
        let score = thread_challenge.score(&thread_phenotype);
        assert_eq!(score, -42.0);
    });

    handle.join().unwrap();

    // The thread should have its own cache, so we should have another evaluation
    assert_eq!(challenge.get_evaluations(), 2);
}

#[test]
fn test_evolution_with_caching() {
    // Create components for evolution
    let breed_strategy = OrdinaryStrategy::default();
    let selection_strategy = ElitistSelection::default();
    let hill_climbing = HillClimbing::new(5, 3).unwrap();
    let app_strategy = AllIndividualsStrategy::new();
    let challenge = CostlyChallenge::new();

    // Create options for a small evolution
    let options = EvolutionOptions::builder()
        .num_generations(5)
        .population_size(10)
        .num_offspring(15)
        .build();

    // Create a starting phenotype with a negative score
    let starting_value = TestPhenotype { value: 100 }; // Negative score of -100

    // Run without caching
    let launcher = EvolutionLauncher::new(
        breed_strategy.clone(),
        selection_strategy.clone(),
        Some(LocalSearchManager::new(
            hill_climbing.clone(),
            app_strategy.clone(),
        )),
        challenge.clone(),
    );

    // Run evolution without caching
    let without_cache = launcher
        .configure(options.clone(), starting_value.clone())
        .run()
        .unwrap();

    // Count the number of evaluations
    let evaluations_without_cache = challenge.get_evaluations();

    // Reset the challenge for the next test
    let challenge = CostlyChallenge::new();

    // Create a cached challenge
    let cached_challenge = challenge.with_global_cache();

    // Create a launcher with global caching
    let launcher_with_cache = EvolutionLauncher::new(
        breed_strategy.clone(),
        selection_strategy.clone(),
        Some(LocalSearchManager::new(
            hill_climbing.clone(),
            app_strategy.clone(),
        )),
        cached_challenge,
    );

    // Run evolution with caching
    let with_cache = launcher_with_cache
        .configure(options.clone(), starting_value.clone())
        .run()
        .unwrap();

    // Count the number of evaluations
    let evaluations_with_cache = challenge.get_evaluations();

    // The version with caching should have fewer actual evaluations
    println!(
        "Evaluations without cache: {}, with cache: {}",
        evaluations_without_cache, evaluations_with_cache
    );
    assert!(
        evaluations_with_cache < evaluations_without_cache,
        "Caching should reduce the number of evaluations"
    );

    // Check that scores are negative (for minimization)
    println!(
        "Score without cache: {}, with cache: {}",
        without_cache.score, with_cache.score
    );
    assert!(
        without_cache.score < 0.0,
        "Without cache score should be negative"
    );
    assert!(
        with_cache.score < 0.0,
        "With cache score should be negative"
    );
}

#[test]
fn test_cache_sharing() {
    let challenge = CostlyChallenge::new();

    // In the current implementation, each with_global_cache call creates
    // a new, separate cache instance. They don't share the same global cache.
    let cached_challenge1 = challenge.with_global_cache();

    let phenotype = TestPhenotype { value: 42 };

    // First evaluation should compute
    let score1 = cached_challenge1.score(&phenotype);
    assert_eq!(score1, -42.0);
    assert_eq!(challenge.get_evaluations(), 1);

    // If we create a second challenge with the same inner challenge,
    // it will have a separate cache
    let cached_challenge2 = challenge.with_global_cache();

    // This will trigger a new evaluation since it's a different cache
    let score2 = cached_challenge2.score(&phenotype);
    assert_eq!(score2, -42.0);
    assert_eq!(challenge.get_evaluations(), 2); // Now 2, as caches are not shared
}
