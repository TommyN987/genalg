use std::collections::HashSet;

use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use crate::selection::selection_strategy::SelectionStrategy;

/// A selection strategy that selects individuals through tournament selection.
///
/// Tournament selection works by randomly selecting a small group of individuals
/// (the tournament size) and then choosing the best one from that group. This process
/// is repeated until the desired number of individuals are selected.
///
/// Tournament selection provides a balance between exploration and exploitation:
/// - Smaller tournament sizes lead to more exploration (more random selection)
/// - Larger tournament sizes lead to more exploitation (more focus on the best individuals)
///
/// # Examples
///
/// ```
/// use genalg::selection::tournament::TournamentSelection;
/// use genalg::selection::SelectionStrategy;
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
/// use genalg::error::Result;
///
/// #[derive(Clone, Debug)]
/// struct MyPhenotype {
///     value: f64,
/// }
///
/// impl Phenotype for MyPhenotype {
///     fn crossover(&mut self, other: &Self) {
///         self.value = (self.value + other.value) / 2.0;
///     }
///
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         self.value += 0.1;
///     }
/// }
///
/// fn main() -> Result<()> {
///     let population = vec![
///         MyPhenotype { value: 1.0 },
///         MyPhenotype { value: 2.0 },
///         MyPhenotype { value: 3.0 },
///         MyPhenotype { value: 4.0 },
///         MyPhenotype { value: 5.0 },
///     ];
///     
///     let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];
///     
///     let selection = TournamentSelection::default();
///     let selected = selection.select(&population, &fitness, 3)?;
///     
///     assert_eq!(selected.len(), 3);
///     
///     Ok(())
/// }
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct TournamentSelection {
    tournament_size: usize,
    allow_duplicates: bool,
    higher_is_better: bool,
}

impl TournamentSelection {
    /// Creates a new TournamentSelection strategy with the specified tournament size.
    ///
    /// By default, duplicates are not allowed in the selected individuals,
    /// and higher fitness is considered better.
    ///
    /// # Arguments
    ///
    /// * `tournament_size` - The number of individuals that participate in each tournament.
    ///   Must be at least 1. A tournament size of 1 is equivalent to random selection.
    /// * `allow_duplicates` - Whether to allow duplicates in the selected individuals.
    /// * `higher_is_better` - Whether higher fitness is better (true) or lower fitness is better (false).
    ///
    /// # Returns
    ///
    /// A new TournamentSelection strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if `tournament_size` is 0.
    pub fn new(
        tournament_size: usize,
        allow_duplicates: bool,
        higher_is_better: bool,
    ) -> Result<Self> {
        if tournament_size < 1 {
            return Err(GeneticError::Configuration(
                "Tournament size must be at least 1".to_string(),
            ));
        }

        Ok(Self {
            tournament_size,
            allow_duplicates,
            higher_is_better,
        })
    }

    pub fn with_tournament_size(mut self, tournament_size: usize) -> Result<Self> {
        if tournament_size < 1 {
            return Err(GeneticError::Configuration(
                "Tournament size must be at least 1".to_string(),
            ));
        }
        self.tournament_size = tournament_size;
        Ok(self)
    }

    pub fn with_duplicates(mut self) -> Self {
        self.allow_duplicates = true;
        self
    }

    pub fn with_lower_is_better(mut self) -> Self {
        self.higher_is_better = false;
        self
    }

    /// Runs a single tournament and returns the index of the winner.
    ///
    /// # Arguments
    ///
    /// * `fitness` - The fitness scores of all individuals.
    /// * `rng` - A random number generator.
    /// * `excluded` - A set of indices that should be excluded from the tournament.
    ///
    /// # Returns
    ///
    /// The index of the tournament winner.
    ///
    /// # Errors
    ///
    /// Returns an error if random number generation fails or if all individuals are excluded.
    fn run_tournament(
        &self,
        fitness: &[f64],
        rng: &mut RandomNumberGenerator,
        excluded: &HashSet<usize>,
    ) -> Result<usize> {
        let population_size = fitness.len();

        if excluded.len() >= population_size {
            return Err(GeneticError::Configuration(
                "All individuals are excluded from tournament selection".to_string(),
            ));
        }

        let eligible: Vec<usize> = (0..population_size)
            .filter(|i| !excluded.contains(i))
            .collect();

        if eligible.is_empty() {
            return Err(GeneticError::Configuration(
                "No eligible individuals for tournament selection".to_string(),
            ));
        }

        let mut participants = Vec::with_capacity(self.tournament_size);
        for _ in 0..self.tournament_size {
            let uniform = rng.fetch_uniform(0.0, eligible.len() as f32, 1);
            let idx = match uniform.front() {
                Some(val) => (*val as usize) % eligible.len(),
                None => {
                    return Err(GeneticError::RandomGeneration(
                        "Failed to generate random value for tournament selection".to_string(),
                    ))
                }
            };
            participants.push(eligible[idx]);
        }

        let mut best_idx = participants[0];
        let mut best_fitness = fitness[best_idx];

        for &idx in &participants[1..] {
            let current_fitness = fitness[idx];
            let is_better = if self.higher_is_better {
                current_fitness > best_fitness
            } else {
                current_fitness < best_fitness
            };

            if is_better {
                best_idx = idx;
                best_fitness = current_fitness;
            }
        }

        Ok(best_idx)
    }
}

impl Default for TournamentSelection {
    fn default() -> Self {
        // Safe to unwrap because we know the default values are valid
        Self::new(2, false, true).unwrap()
    }
}

impl<P> SelectionStrategy<P> for TournamentSelection
where
    P: Phenotype,
{
    fn select(&self, population: &[P], fitness: &[f64], num_to_select: usize) -> Result<Vec<P>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        if fitness.len() != population.len() {
            return Err(GeneticError::Configuration(format!(
                "Fitness vector length ({}) doesn't match population length ({})",
                fitness.len(),
                population.len()
            )));
        }

        let mut rng = RandomNumberGenerator::new();

        let mut selected = Vec::with_capacity(num_to_select);
        let mut selected_indices = HashSet::new();

        while selected.len() < num_to_select {
            if !self.allow_duplicates && selected_indices.len() >= population.len() {
                break;
            }

            let winner_idx = self.run_tournament(fitness, &mut rng, &selected_indices)?;

            if self.allow_duplicates || selected_indices.insert(winner_idx) {
                selected.push(population[winner_idx].clone());
            }
        }

        if self.allow_duplicates && selected.len() < num_to_select {
            let empty_set = HashSet::new();
            while selected.len() < num_to_select {
                let winner_idx = self.run_tournament(fitness, &mut rng, &empty_set)?;
                selected.push(population[winner_idx].clone());
            }
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phenotype::Phenotype;
    use crate::rng::RandomNumberGenerator;

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        value: f64,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            self.value = (self.value + other.value) / 2.0;
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            self.value += 0.1;
        }
    }

    #[test]
    fn test_tournament_selection() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with default parameters (tournament size 2)
        let selection = TournamentSelection::default();
        let selected = selection.select(&population, &fitness, 3).unwrap();

        // Should select 3 individuals
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_tournament_selection_with_different_sizes() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with tournament size 1 (equivalent to random selection)
        let selection = TournamentSelection::new(1, false, true).unwrap();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);

        // Test with tournament size equal to population size (equivalent to elitist selection)
        let selection = TournamentSelection::new(5, false, true).unwrap();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);

        // Test with large tournament size (greater than population size)
        let selection = TournamentSelection::new(10, false, true).unwrap();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_tournament_selection_lower_is_better() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1]; // Lower values are better

        // Test with lower is better
        let selection = TournamentSelection::new(3, false, false).unwrap();
        let selected = selection.select(&population, &fitness, 10).unwrap();

        assert_eq!(selected.len(), 5); // Should select all 5 individuals (no duplicates)
    }

    #[test]
    fn test_tournament_selection_with_duplicates() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];

        // Test with duplicates allowed
        let selection = TournamentSelection::default().with_duplicates();
        let selected = selection.select(&population, &fitness, 10).unwrap();

        assert_eq!(selected.len(), 10); // Should select 10 individuals with duplicates
    }

    #[test]
    fn test_tournament_selection_empty_population() {
        let population: Vec<TestPhenotype> = Vec::new();
        let fitness: Vec<f64> = Vec::new();

        let selection = TournamentSelection::default();
        let result = selection.select(&population, &fitness, 3);

        assert!(result.is_err());
    }

    #[test]
    fn test_tournament_selection_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];

        let fitness = vec![0.5];

        let selection = TournamentSelection::default();
        let result = selection.select(&population, &fitness, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_tournament_selection_invalid_size() {
        // Tournament size must be at least 1
        let selection = TournamentSelection::default().with_tournament_size(0);

        assert!(selection.is_err());
    }

    #[test]
    fn test_run_tournament() {
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];
        let mut rng = RandomNumberGenerator::from_seed(42);
        let excluded = HashSet::new();

        // Test with higher is better
        let selection = TournamentSelection::default();
        let winner = selection
            .run_tournament(&fitness, &mut rng, &excluded)
            .unwrap();

        // With the fixed seed, we should get a deterministic result
        assert!(winner < fitness.len());

        // Test with lower is better
        let selection = TournamentSelection::default().with_lower_is_better();
        let winner = selection
            .run_tournament(&fitness, &mut rng, &excluded)
            .unwrap();

        // With the fixed seed, we should get a deterministic result
        assert!(winner < fitness.len());
    }

    #[test]
    fn test_run_tournament_with_excluded() {
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];
        let mut rng = RandomNumberGenerator::from_seed(42);

        // Exclude all but one individual
        let mut excluded = HashSet::new();
        excluded.insert(0);
        excluded.insert(1);
        excluded.insert(2);
        excluded.insert(4);

        let selection = TournamentSelection::default();
        let winner = selection
            .run_tournament(&fitness, &mut rng, &excluded)
            .unwrap();

        // Only index 3 is not excluded
        assert_eq!(winner, 3);

        // Exclude all individuals
        let mut excluded = HashSet::new();
        for i in 0..fitness.len() {
            excluded.insert(i);
        }

        let result = selection.run_tournament(&fitness, &mut rng, &excluded);
        assert!(result.is_err());
    }
}
