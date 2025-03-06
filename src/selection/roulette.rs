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
///     let mut rng = RandomNumberGenerator::new();
///     
///     let selection = RouletteWheelSelection::new();
///     let selected = selection.select(&population, &fitness, 3, Some(&mut rng))?;
///     
///     assert_eq!(selected.len(), 3);
///     
///     Ok(())
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RouletteWheelSelection {
    /// Whether to allow duplicates in the selected individuals.
    allow_duplicates: bool,
    /// Whether higher fitness is better (true) or lower fitness is better (false).
    higher_is_better: bool,
}

impl RouletteWheelSelection {
    /// Creates a new RouletteWheelSelection strategy.
    ///
    /// By default, duplicates are not allowed in the selected individuals,
    /// and higher fitness is considered better.
    pub fn new() -> Self {
        Self {
            allow_duplicates: false,
            higher_is_better: true,
        }
    }

    /// Creates a new RouletteWheelSelection strategy with the specified duplicate policy.
    ///
    /// # Arguments
    ///
    /// * `allow_duplicates` - Whether to allow duplicates in the selected individuals.
    pub fn with_duplicates(allow_duplicates: bool) -> Self {
        Self {
            allow_duplicates,
            higher_is_better: true,
        }
    }

    /// Creates a new RouletteWheelSelection strategy with the specified options.
    ///
    /// # Arguments
    ///
    /// * `higher_is_better` - Whether higher fitness is better (true) or lower fitness is better (false).
    /// * `allow_duplicates` - Whether to allow duplicates in the selected individuals.
    pub fn with_options(higher_is_better: bool, allow_duplicates: bool) -> Self {
        Self {
            allow_duplicates,
            higher_is_better,
        }
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

        // Check for negative fitness values
        if fitness.iter().any(|&f| f < 0.0) {
            return Err(GeneticError::Configuration(
                "Roulette wheel selection requires non-negative fitness values".to_string(),
            ));
        }

        // If higher is better, use fitness directly; if lower is better, invert
        let adjusted_fitness: Vec<f64> = if self.higher_is_better {
            fitness.to_vec()
        } else {
            // For minimization problems, we need to invert the fitness
            // Find the maximum fitness value
            let max_fitness = match fitness.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) {
                f if f <= 0.0 => {
                    return Err(GeneticError::Configuration(
                        "Cannot invert fitness values for minimization when all values are non-positive".to_string(),
                    ));
                }
                f => f + 1.0, // Add 1.0 to ensure all values are positive after inversion
            };

            // Invert the fitness: max - fitness
            fitness.iter().map(|&f| max_fitness - f).collect()
        };

        // Calculate the sum of all fitness values
        let sum: f64 = adjusted_fitness.iter().sum();

        // If the sum is zero, we can't perform roulette wheel selection
        if sum == 0.0 {
            return Err(GeneticError::Configuration(
                "Roulette wheel selection requires at least one individual with non-zero fitness"
                    .to_string(),
            ));
        }

        // Calculate cumulative probabilities
        let mut cumulative = 0.0;
        for &fitness in &adjusted_fitness {
            cumulative += fitness / sum;
            probs.push(cumulative);
        }

        // Ensure the last probability is exactly 1.0 to avoid floating-point errors
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
        // Generate a random value between 0 and 1
        let uniform = rng.fetch_uniform(0.0, 1.0, 1);
        let r = match uniform.front() {
            Some(val) => *val as f64,
            None => {
                return Err(GeneticError::RandomGeneration(
                    "Failed to generate random value for roulette wheel selection".to_string(),
                ))
            }
        };

        // Find the first individual whose cumulative probability is greater than r
        for (i, &prob) in cumulative_probs.iter().enumerate() {
            if r <= prob {
                return Ok(i);
            }
        }

        // If we get here, it's due to floating-point errors, so return the last individual
        Ok(cumulative_probs.len() - 1)
    }
}

impl Default for RouletteWheelSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> SelectionStrategy<P> for RouletteWheelSelection
where
    P: Phenotype,
{
    fn select(
        &self,
        population: &[P],
        fitness: &[f64],
        num_to_select: usize,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Vec<P>> {
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

        // Roulette wheel selection requires randomness
        let rng = match rng {
            Some(rng) => rng,
            None => {
                return Err(GeneticError::Configuration(
                    "Roulette wheel selection requires a random number generator".to_string(),
                ))
            }
        };

        // Calculate selection probabilities
        let cumulative_probs = self.calculate_probabilities(fitness)?;

        let mut selected = Vec::with_capacity(num_to_select);
        let mut selected_indices = HashSet::new();

        // Select individuals until we have enough
        while selected.len() < num_to_select {
            // If we've selected all individuals and duplicates are not allowed, break
            if !self.allow_duplicates && selected_indices.len() >= population.len() {
                break;
            }

            // Select an individual
            let idx = self.select_individual(&cumulative_probs, rng)?;

            // Add the individual to the selected individuals if it's not already selected
            // or if duplicates are allowed
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
        let mut rng = RandomNumberGenerator::from_seed(42); // Use fixed seed for deterministic testing

        // Test with default parameters
        let selection = RouletteWheelSelection::new();
        let selected = selection
            .select(&population, &fitness, 3, Some(&mut rng))
            .unwrap();

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
        let mut rng = RandomNumberGenerator::from_seed(42);

        // Test with lower is better
        let selection = RouletteWheelSelection::with_options(false, false);
        let selected = selection
            .select(&population, &fitness, 10, Some(&mut rng))
            .unwrap();

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
        let mut rng = RandomNumberGenerator::from_seed(42);

        // Test with duplicates allowed
        let selection = RouletteWheelSelection::with_duplicates(true);
        let selected = selection
            .select(&population, &fitness, 10, Some(&mut rng))
            .unwrap();

        assert_eq!(selected.len(), 10); // Should select 10 individuals with duplicates
    }

    #[test]
    fn test_roulette_wheel_selection_without_duplicates() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];
        let mut rng = RandomNumberGenerator::from_seed(42);

        // Test without duplicates
        let selection = RouletteWheelSelection::with_duplicates(false);
        let selected = selection
            .select(&population, &fitness, 10, Some(&mut rng))
            .unwrap();

        assert_eq!(selected.len(), 3); // Should only select 3 individuals (no duplicates)
    }

    #[test]
    fn test_roulette_wheel_selection_empty_population() {
        let population: Vec<TestPhenotype> = Vec::new();
        let fitness: Vec<f64> = Vec::new();
        let mut rng = RandomNumberGenerator::new();

        let selection = RouletteWheelSelection::new();
        let result = selection.select(&population, &fitness, 3, Some(&mut rng));

        assert!(result.is_err());
    }

    #[test]
    fn test_roulette_wheel_selection_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];

        let fitness = vec![0.5];
        let mut rng = RandomNumberGenerator::new();

        let selection = RouletteWheelSelection::new();
        let result = selection.select(&population, &fitness, 1, Some(&mut rng));

        assert!(result.is_err());
    }

    #[test]
    fn test_roulette_wheel_selection_without_rng() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];

        let fitness = vec![0.5, 0.8, 0.3];

        // Roulette wheel selection requires an RNG
        let selection = RouletteWheelSelection::new();
        let result = selection.select(&population, &fitness, 1, None);

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
        let mut rng = RandomNumberGenerator::new();

        let selection = RouletteWheelSelection::new();
        let result = selection.select(&population, &fitness, 1, Some(&mut rng));

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
        let mut rng = RandomNumberGenerator::new();

        let selection = RouletteWheelSelection::new();
        let result = selection.select(&population, &fitness, 1, Some(&mut rng));

        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_probabilities() {
        let fitness = vec![0.5, 0.8, 0.3, 0.9, 0.1];

        // Test with higher is better
        let selection = RouletteWheelSelection::new();
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
        let selection = RouletteWheelSelection::with_options(false, false);
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

        let selection = RouletteWheelSelection::new();
        let idx = selection.select_individual(&probs, &mut rng).unwrap();

        // With the fixed seed, we should get a deterministic result
        assert!(idx < probs.len());
    }
}
