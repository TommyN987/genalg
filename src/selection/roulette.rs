use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use crate::selection::selection_strategy::SelectionStrategy;
use std::collections::HashSet;

/// A selection strategy that selects individuals through roulette wheel selection.
///
/// Roulette wheel selection (also known as fitness proportionate selection) selects
/// individuals with probability proportional to their fitness. Individuals with higher
/// fitness have a higher chance of being selected.
///
/// This strategy requires all fitness values to be non-negative. If you have negative
/// fitness values, consider using rank-based selection instead.
///
/// # Examples
///
/// ```
/// use genalg::selection::roulette::RouletteWheelSelection;
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
///     // Create a roulette wheel selection with default parameters (no duplicates, higher is better)
///     let selection = RouletteWheelSelection::new(false, true);
///     let selected = selection.select(&population, &fitness, 3)?;
///     
///     assert_eq!(selected.len(), 3);
///     
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RouletteWheelSelection {
    allow_duplicates: bool,
    higher_is_better: bool,
}

impl RouletteWheelSelection {
    /// Creates a new RouletteWheelSelection strategy.
    ///
    /// By default, duplicates are not allowed in the selected individuals,
    /// and higher fitness is considered better.
    pub fn new(allow_duplicates: bool, higher_is_better: bool) -> Self {
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

    /// Calculates the selection probabilities for each individual based on their fitness.
    ///
    /// # Arguments
    ///
    /// * `fitness` - The fitness scores of all individuals.
    ///
    /// # Returns
    ///
    /// A vector of cumulative probabilities for each individual.
    ///
    /// # Errors
    ///
    /// Returns an error if all fitness values are zero or if any fitness value is negative.
    fn calculate_probabilities(&self, fitness: &[f64]) -> Result<Vec<f64>> {
        let population_size = fitness.len();
        let mut probs: Vec<f64> = Vec::with_capacity(population_size);

        if fitness.iter().any(|&f| f < 0.0) {
            return Err(GeneticError::Selection(
                "Roulette wheel selection requires non-negative fitness values".to_string(),
            ));
        }

        let adjusted_fitness: Vec<f64> = if self.higher_is_better {
            fitness.to_vec()
        } else {
            let max_fitness = match fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) {
                f if f <= 0.0 => {
                    return Err(GeneticError::Selection(
                        "Cannot invert fitness values for minimization when all values are non-positive".to_string(),
                    ));
                }
                f => f + 1.0, // Add 1.0 to ensure all values are positive after inversion
            };

            fitness.iter().map(|&f| max_fitness - f).collect()
        };

        let sum: f64 = adjusted_fitness.iter().sum();

        if sum == 0.0 {
            return Err(GeneticError::Selection(
                "Roulette wheel selection requires at least one individual with non-zero fitness"
                    .to_string(),
            ));
        }

        let mut cumulative = 0.0;
        for &fitness in &adjusted_fitness {
            cumulative += fitness / sum;
            probs.push(cumulative);
        }

        if let Some(last) = probs.last_mut() {
            *last = 1.0;
        }

        Ok(probs)
    }

    /// Selects an individual using roulette wheel selection.
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
                    "Failed to generate random value for roulette wheel selection".to_string(),
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

impl Default for RouletteWheelSelection {
    fn default() -> Self {
        Self::new(false, true)
    }
}

impl<P> SelectionStrategy<P> for RouletteWheelSelection
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

        let cumulative_probs = self.calculate_probabilities(fitness)?;

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
    fn test_roulette_wheel_selection() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with default parameters
        let selection = RouletteWheelSelection::default();
        let selected = selection.select(&population, &fitness, 3).unwrap();

        // Should select 3 individuals
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_roulette_wheel_selection_lower_is_better() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1]; // Lower values are better

        // Test with lower is better
        let selection = RouletteWheelSelection::default().with_lower_is_better();
        let selected = selection.select(&population, &fitness, 10).unwrap();

        assert_eq!(selected.len(), 5); // Should select all 5 individuals (no duplicates)
    }

    #[test]
    fn test_roulette_wheel_selection_with_duplicates() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];

        // Test with duplicates allowed
        let selection = RouletteWheelSelection::default().with_duplicates();
        let selected = selection.select(&population, &fitness, 10).unwrap();

        assert_eq!(selected.len(), 10); // Should select 10 individuals with duplicates
    }

    #[test]
    fn test_roulette_wheel_selection_empty_population() {
        let population: Vec<TestPhenotype> = Vec::new();
        let fitness: Vec<f64> = Vec::new();

        let selection = RouletteWheelSelection::default();
        let result = selection.select(&population, &fitness, 3);

        assert!(result.is_err());
    }

    #[test]
    fn test_roulette_wheel_selection_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];

        let fitness = vec![0.5];

        let selection = RouletteWheelSelection::default();
        let result = selection.select(&population, &fitness, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_roulette_wheel_selection_negative_fitness() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, -0.8, 0.3]; // Contains negative fitness

        let selection = RouletteWheelSelection::default();
        let result = selection.select(&population, &fitness, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_roulette_wheel_selection_zero_fitness() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.0, 0.0, 0.0]; // All zero fitness

        let selection = RouletteWheelSelection::default();
        let result = selection.select(&population, &fitness, 1);

        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_probabilities() {
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with higher is better
        let selection = RouletteWheelSelection::default();
        let probs = selection.calculate_probabilities(&fitness).unwrap();

        // Should have the same length as fitness
        assert_eq!(probs.len(), fitness.len());

        // Probabilities should be cumulative and the last one should be 1.0
        assert!((probs[probs.len() - 1] - 1.0).abs() < f64::EPSILON);

        // Probabilities should be in ascending order
        for i in 1..probs.len() {
            assert!(probs[i] >= probs[i - 1]);
        }

        // Test with lower is better
        let selection = RouletteWheelSelection::default().with_lower_is_better();
        let probs = selection.calculate_probabilities(&fitness).unwrap();

        // Should have the same length as fitness
        assert_eq!(probs.len(), fitness.len());

        // Probabilities should be cumulative and the last one should be 1.0
        assert!((probs[probs.len() - 1] - 1.0).abs() < f64::EPSILON);

        // Probabilities should be in ascending order
        for i in 1..probs.len() {
            assert!(probs[i] >= probs[i - 1]);
        }
    }

    #[test]
    fn test_select_individual() {
        let probs = vec![0.2, 0.5, 0.8, 1.0];
        let mut rng = RandomNumberGenerator::from_seed(42);

        let selection = RouletteWheelSelection::default();
        let idx = selection.select_individual(&probs, &mut rng).unwrap();

        // With the fixed seed, we should get a deterministic result
        assert!(idx < probs.len());
    }
}
