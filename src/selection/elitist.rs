use std::cmp::Ordering;

use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::selection::selection_strategy::SelectionStrategy;

/// A selection strategy that selects the best individuals based on fitness.
///
/// This strategy sorts individuals by their fitness scores and selects the top N individuals.
/// It implements elitism, which is a common technique in genetic algorithms to ensure that
/// the best solutions are preserved across generations.
///
/// # Examples
///
/// ```
/// use genalg::selection::elitist::ElitistSelection;
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
///     ];
///     
///     let fitness = vec![0.5, 0.8, 0.3];
///     
///     // By default, higher fitness is better
///     let selection = ElitistSelection::default();
///     let selected = selection.select(&population, &fitness, 2)?;
///     
///     assert_eq!(selected.len(), 2);
///     assert_eq!(selected[0].value, 2.0); // Highest fitness (0.8)
///     assert_eq!(selected[1].value, 1.0); // Second highest fitness (0.5)
///     
///     // For minimization problems, lower fitness is better
///     let selection = ElitistSelection::new(false, false);
///     let selected = selection.select(&population, &fitness, 2)?;
///     
///     assert_eq!(selected.len(), 2);
///     assert_eq!(selected[0].value, 3.0); // Lowest fitness (0.3)
///     assert_eq!(selected[1].value, 1.0); // Second lowest fitness (0.5)
///     
///     Ok(())
/// }
/// ```
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct ElitistSelection {
    /// Whether to allow duplicates in the selected individuals.
    allow_duplicates: bool,
    /// Whether higher fitness is better (true) or lower fitness is better (false).
    higher_is_better: bool,
}

impl ElitistSelection {
    /// Creates a new ElitistSelection strategy with the specified options.
    ///
    /// # Arguments
    ///
    /// * `higher_is_better` - Whether higher fitness is better (true) or lower fitness is better (false).
    /// * `allow_duplicates` - Whether to allow duplicates in the selected individuals.
    pub fn new(higher_is_better: bool, allow_duplicates: bool) -> Self {
        Self {
            allow_duplicates,
            higher_is_better,
        }
    }

    pub fn with_duplicates(mut self) -> Self {
        self.allow_duplicates = true;
        self
    }

    pub fn with_lower_is_better(mut self) -> Self {
        self.higher_is_better = false;
        self
    }
}

impl Default for ElitistSelection {
    /// Creates a new ElitistSelection strategy.
    ///
    /// By default, duplicates are not allowed in the selected individuals,
    /// and higher fitness is considered better.
    fn default() -> Self {
        Self {
            allow_duplicates: false,
            higher_is_better: true,
        }
    }
}

impl<P> SelectionStrategy<P> for ElitistSelection
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

        let mut indexed_fitness: Vec<(usize, f64)> = fitness
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        indexed_fitness.sort_by(|a, b| {
            let cmp = a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                if a.1.is_nan() {
                    Ordering::Less
                } else if b.1.is_nan() {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            });

            if self.higher_is_better {
                cmp.reverse()
            } else {
                cmp
            }
        });

        let mut selected = Vec::with_capacity(num_to_select);
        let mut selected_indices = std::collections::HashSet::new();

        for (idx, _) in indexed_fitness.iter() {
            if !self.allow_duplicates && !selected_indices.insert(*idx) {
                continue;
            }

            selected.push(population[*idx].clone());

            if selected.len() >= num_to_select {
                break;
            }
        }

        // If we need more individuals and duplicates are allowed, cycle through the best ones
        if self.allow_duplicates && selected.len() < num_to_select && !indexed_fitness.is_empty() {
            let mut idx = 0;
            while selected.len() < num_to_select {
                selected.push(population[indexed_fitness[idx % indexed_fitness.len()].0].clone());
                idx += 1;
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
    fn test_elitist_selection() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        // Fitness values (not in order of population)
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with default parameters (higher is better)
        let selection = ElitistSelection::default();
        let selected = selection.select(&population, &fitness, 3).unwrap();

        // Should select individuals with highest fitness
        assert_eq!(selected.len(), 3);
        // Check that the best individuals were selected (indices 3, 1, 0 with fitness 0.9, 0.8, 0.5)
        assert!((selected[0].value - 4.0).abs() < f64::EPSILON); // index 3, fitness 0.9
        assert!((selected[1].value - 2.0).abs() < f64::EPSILON); // index 1, fitness 0.8
        assert!((selected[2].value - 1.0).abs() < f64::EPSILON); // index 0, fitness 0.5
    }

    #[test]
    fn test_elitist_selection_lower_is_better() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        // Fitness values (not in order of population)
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with lower is better
        let selection = ElitistSelection::new(false, false);
        let selected = selection.select(&population, &fitness, 3).unwrap();

        // Should select individuals with lowest fitness
        assert_eq!(selected.len(), 3);
        // Check that the best individuals were selected (indices 4, 2, 0 with fitness 0.1, 0.3, 0.5)
        assert!((selected[0].value - 5.0).abs() < f64::EPSILON); // index 4, fitness 0.1
        assert!((selected[1].value - 3.0).abs() < f64::EPSILON); // index 2, fitness 0.3
        assert!((selected[2].value - 1.0).abs() < f64::EPSILON); // index 0, fitness 0.5
    }

    #[test]
    fn test_elitist_selection_with_duplicates() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];

        // Test with duplicates allowed and lower_is_better=false
        // This means that lower fitness values are considered better
        let selection = ElitistSelection::new(false, true);
        let selected = selection.select(&population, &fitness, 5).unwrap();

        // Should select 5 individuals, with duplicates
        assert_eq!(selected.len(), 5);

        // For lower_is_better=false, lower fitness values are better
        // So the order should be: index 2 (fitness 0.3), index 0 (fitness 0.5), index 1 (fitness 0.8)
        assert!((selected[0].value - 3.0).abs() < f64::EPSILON); // index 2, fitness 0.3
        assert!((selected[1].value - 1.0).abs() < f64::EPSILON); // index 0, fitness 0.5
        assert!((selected[2].value - 2.0).abs() < f64::EPSILON); // index 1, fitness 0.8

        // The remaining 2 should be duplicates of the best individuals
        assert!((selected[3].value - 3.0).abs() < f64::EPSILON); // duplicate of index 2
        assert!((selected[4].value - 1.0).abs() < f64::EPSILON); // duplicate of index 0
    }

    #[test]
    fn test_elitist_selection_without_duplicates() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];

        // Test without duplicates
        let selection = ElitistSelection::new(true, false);
        let selected = selection.select(&population, &fitness, 5).unwrap();

        // Should only select 3 individuals (no duplicates)
        assert_eq!(selected.len(), 3);
        // Should be in order of fitness
        assert!((selected[0].value - 2.0).abs() < f64::EPSILON); // index 1, fitness 0.8
        assert!((selected[1].value - 1.0).abs() < f64::EPSILON); // index 0, fitness 0.5
        assert!((selected[2].value - 3.0).abs() < f64::EPSILON); // index 2, fitness 0.3
    }

    #[test]
    fn test_elitist_selection_empty_population() {
        let population: Vec<TestPhenotype> = Vec::new();
        let fitness: Vec<f64> = Vec::new();

        let selection = ElitistSelection::default();
        let result = selection.select(&population, &fitness, 3);

        assert!(result.is_err());
    }

    #[test]
    fn test_elitist_selection_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];

        let fitness = vec![0.5];

        let selection = ElitistSelection::default();
        let result = selection.select(&population, &fitness, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_elitist_selection_with_nan() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, f64::NAN, 0.3];

        // Test with NaN values
        let selection = ElitistSelection::default();
        let selected = selection.select(&population, &fitness, 3).unwrap();

        // NaN values should be sorted last
        assert_eq!(selected.len(), 3);
        assert!((selected[0].value - 1.0).abs() < f64::EPSILON); // index 0, fitness 0.5
        assert!((selected[1].value - 3.0).abs() < f64::EPSILON); // index 2, fitness 0.3
        assert!((selected[2].value - 2.0).abs() < f64::EPSILON); // index 1, fitness NaN
    }
}
