use std::collections::HashSet;

use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use crate::selection::selection_strategy::SelectionStrategy;

/// A selection strategy that selects individuals based on their rank in the population.
///
/// Rank-based selection assigns a selection probability to each individual based on its
/// rank in the population, rather than its absolute fitness value. This helps prevent
/// premature convergence when there are a few individuals with much higher fitness than
/// the rest of the population.
///
/// This strategy works well for both maximization and minimization problems, and can
/// handle negative fitness values.
///
/// # Examples
///
/// ```
/// use genalg::selection::rank::RankBasedSelection;
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
///     // Create a rank-based selection with default parameters
///     let selection = RankBasedSelection::new(1.5, false, true)?;
///     let selected = selection.select(&population, &fitness, 3)?;
///     
///     assert_eq!(selected.len(), 3);
///     
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RankBasedSelection {
    /// Higher values increase selection pressure.
    selection_pressure: f64,
    allow_duplicates: bool,
    higher_is_better: bool,
}

impl RankBasedSelection {
    /// Creates a new RankBasedSelection strategy with default parameters.
    ///
    /// By default:
    /// - Selection pressure is set to 1.5 (a balanced middle ground between 1.0 and 2.0)
    /// - Duplicates are not allowed in the selected individuals
    /// - Higher fitness is considered better
    pub fn new(
        selection_pressure: f64,
        allow_duplicates: bool,
        higher_is_better: bool,
    ) -> Result<Self> {
        if !(1.0..=2.0).contains(&selection_pressure) {
            return Err(GeneticError::Configuration(
                "Selection pressure must be in the range [1.0, 2.0]".to_string(),
            ));
        }

        Ok(Self {
            selection_pressure,
            allow_duplicates,
            higher_is_better,
        })
    }

    /// Creates a new RankBasedSelection strategy with the specified selection pressure.
    ///
    /// # Arguments
    ///
    /// * `selection_pressure` - The selection pressure parameter. Must be in the range [1.0, 2.0].
    ///   Higher values increase selection pressure.
    ///   - At 1.0, all individuals have equal selection probability (no selection pressure)
    ///   - At 2.0, selection pressure is at its maximum
    ///   - The default value of 1.5 provides a balanced selection pressure
    ///
    /// # Returns
    ///
    /// A Result containing the RankBasedSelection instance, or an error if the selection pressure
    /// is outside the valid range.
    ///
    /// # Errors
    ///
    /// Returns a `GeneticError::Configuration` error if `selection_pressure` is not in the range [1.0, 2.0].
    pub fn with_pressure(mut self, selection_pressure: f64) -> Result<Self> {
        if !(1.0..=2.0).contains(&selection_pressure) {
            return Err(GeneticError::Configuration(
                "Selection pressure must be in the range [1.0, 2.0]".to_string(),
            ));
        }

        self.selection_pressure = selection_pressure;

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

    /// Calculates the selection probabilities for each individual based on their rank.
    ///
    /// # Arguments
    ///
    /// * `fitness` - The fitness scores of all individuals.
    ///
    /// # Returns
    ///
    /// A vector of cumulative probabilities for each individual.
    fn calculate_probabilities(&self, fitness: &[f64]) -> Vec<f64> {
        let population_size = fitness.len();

        let mut indices: Vec<usize> = (0..population_size).collect();

        if self.higher_is_better {
            indices.sort_by(|&a, &b| {
                let fa = fitness[a];
                let fb = fitness[b];

                if fa.is_nan() {
                    return std::cmp::Ordering::Less;
                }
                if fb.is_nan() {
                    return std::cmp::Ordering::Greater;
                }

                fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indices.sort_by(|&a, &b| {
                let fa = fitness[a];
                let fb = fitness[b];

                if fa.is_nan() {
                    return std::cmp::Ordering::Greater;
                }
                if fb.is_nan() {
                    return std::cmp::Ordering::Less;
                }

                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        let mut rank_map = vec![0; population_size];
        for (rank, &idx) in indices.iter().enumerate() {
            rank_map[idx] = rank;
        }

        let mut probs: Vec<f64> = Vec::with_capacity(population_size);
        let mut cumulative = 0.0;

        for &rank in rank_map.iter() {
            let rank = rank as f64;
            let prob = (2.0 - self.selection_pressure) / population_size as f64
                + (2.0 * rank * (self.selection_pressure - 1.0))
                    / (population_size as f64 * (population_size as f64 - 1.0));

            cumulative += prob;
            probs.push(cumulative);
        }

        if let Some(last) = probs.last() {
            if *last > 0.0 && (*last - 1.0).abs() > f64::EPSILON {
                let last = *last;
                for prob in &mut probs {
                    *prob /= last;
                }
            }
        }

        if let Some(last) = probs.last_mut() {
            *last = 1.0;
        }

        probs
    }

    /// Selects an individual using rank-based selection.
    ///
    /// # Arguments
    ///
    /// * `cumulative_probs` - The cumulative probabilities for each individual.
    /// * `rng` - A random number generator.
    ///
    /// # Returns
    ///
    /// The index of the selected individual.
    ///
    /// # Errors
    ///
    /// Returns an error if random number generation fails.
    fn select_individual(
        &self,
        cumulative_probs: &[f64],
        rng: &mut RandomNumberGenerator,
    ) -> Result<usize> {
        let uniform = rng.fetch_uniform(0.0, 1.0, 1);
        let r = match uniform.front() {
            Some(val) => *val as f64,
            None => {
                return Err(GeneticError::RandomGeneration(
                    "Failed to generate random value for rank-based selection".to_string(),
                ))
            }
        };

        for (i, &prob) in cumulative_probs.iter().enumerate() {
            if r <= prob {
                return Ok(i);
            }
        }

        Ok(cumulative_probs.len() - 1)
    }
}

impl Default for RankBasedSelection {
    fn default() -> Self {
        Self {
            selection_pressure: 1.5,
            allow_duplicates: false,
            higher_is_better: true,
        }
    }
}

impl<P> SelectionStrategy<P> for RankBasedSelection
where
    P: Phenotype,
{
    fn select(&self, population: &[P], fitness: &[f64], num_to_select: usize) -> Result<Vec<P>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }

        if fitness.len() != population.len() {
            return Err(GeneticError::Selection(format!(
                "Fitness vector length ({}) doesn't match population length ({})",
                fitness.len(),
                population.len()
            )));
        }

        let cumulative_probs = self.calculate_probabilities(fitness);

        let mut selected = Vec::with_capacity(num_to_select);
        let mut selected_indices = HashSet::new();
        let mut rng = RandomNumberGenerator::new();

        while selected.len() < num_to_select {
            if !self.allow_duplicates && selected_indices.len() >= population.len() {
                break;
            }

            let idx = self.select_individual(&cumulative_probs, &mut rng)?;

            if self.allow_duplicates || selected_indices.insert(idx) {
                selected.push(population[idx].clone());
            }
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use crate::phenotype::Phenotype;
    use crate::rng::RandomNumberGenerator;

    use super::*;

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
    fn test_rank_based_selection() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with default parameters
        let selection = RankBasedSelection::default();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);

        // Test with custom selection pressure
        let selection = RankBasedSelection::default().with_pressure(1.8).unwrap();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);

        // Test with duplicates allowed
        let selection = RankBasedSelection::default().with_duplicates();
        let selected = selection.select(&population, &fitness, 10).unwrap();
        assert_eq!(selected.len(), 10);

        // Test with lower is better and no duplicates
        // When duplicates are not allowed, we can only select up to population.len() individuals
        let selection = RankBasedSelection::new(1.2, false, true).unwrap();
        let selected = selection.select(&population, &fitness, 10).unwrap();
        assert_eq!(selected.len(), 5); // Only 5 unique individuals are available
    }

    #[test]
    fn test_rank_based_selection_empty_population() {
        let population: Vec<TestPhenotype> = Vec::new();
        let fitness: Vec<f64> = Vec::new();

        let selection = RankBasedSelection::default();
        let result = selection.select(&population, &fitness, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_based_selection_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];
        let fitness = vec![0.5];

        let selection = RankBasedSelection::default();
        let result = selection.select(&population, &fitness, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_rank_based_selection_with_nan() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![0.5, f64::NAN, 0.3];

        let selection = RankBasedSelection::default();
        let selected = selection.select(&population, &fitness, 2).unwrap();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_rank_based_selection_minimization() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];
        // Lower values are better in this test
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with minimization (lower is better)
        let selection = RankBasedSelection::new(1.5, false, false).unwrap();
        let selected = selection.select(&population, &fitness, 3).unwrap();
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_invalid_selection_pressure() {
        // Test with selection pressure below minimum
        let result = RankBasedSelection::default().with_pressure(0.5);
        assert!(result.is_err());

        // Test with selection pressure above maximum
        let result = RankBasedSelection::default().with_pressure(2.5);
        assert!(result.is_err());

        // Test with options and invalid selection pressure
        let result = RankBasedSelection::new(2.5, true, false);
        assert!(result.is_err());
    }
}
