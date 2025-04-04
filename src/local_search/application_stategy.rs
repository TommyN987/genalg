use crate::error::{GeneticError, Result};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;
use std::fmt::Debug;

/// A strategy for selecting individuals from a population to apply local search to.
///
/// This trait allows for different approaches to determining which individuals should
/// undergo local search during the evolutionary process. Implementations can select
/// individuals based on various criteria such as fitness, diversity, or random chance.
pub trait LocalSearchApplicationStrategy<P>: Debug + Send + Sync
where
    P: Phenotype,
{
    /// Selects individuals from the population to apply local search to.
    ///
    /// # Arguments
    ///
    /// * `population` - The current population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    ///
    /// # Returns
    ///
    /// A vector of indices of selected individuals.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    /// - The selection process requires randomness but `rng` is `None`
    /// - The selection process encounters an error (e.g., random number generation fails)
    fn select_for_local_search(&self, population: &[P], fitness: &[f64]) -> Result<Vec<usize>>;
}

/// A strategy that applies local search to all individuals in the population.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct AllIndividualsStrategy;

impl AllIndividualsStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AllIndividualsStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> LocalSearchApplicationStrategy<P> for AllIndividualsStrategy
where
    P: Phenotype,
{
    fn select_for_local_search(&self, population: &[P], fitness: &[f64]) -> Result<Vec<usize>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        if population.len() != fitness.len() {
            return Err(GeneticError::Configuration(format!(
                "Population size ({}) does not match fitness vector size ({})",
                population.len(),
                fitness.len()
            )));
        }

        Ok((0..population.len()).collect())
    }
}

/// A strategy that applies local search to the top N individuals by fitness.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct TopNStrategy {
    n: usize,
    higher_is_better: bool,
}

impl TopNStrategy {
    /// Creates a new strategy that applies local search to the top N individuals.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of top individuals to select.
    /// * `higher_is_better` - Whether higher fitness is better (true) or lower fitness is better (false).
    pub fn new(n: usize, higher_is_better: bool) -> Self {
        Self {
            n,
            higher_is_better,
        }
    }

    /// Creates a new strategy that applies local search to the top N individuals,
    /// assuming higher fitness is better.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of top individuals to select.
    pub fn new_maximizing(n: usize) -> Self {
        Self::new(n, true)
    }

    /// Creates a new strategy that applies local search to the top N individuals,
    /// assuming lower fitness is better.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of top individuals to select.
    pub fn new_minimizing(n: usize) -> Self {
        Self::new(n, false)
    }
}

impl<P> LocalSearchApplicationStrategy<P> for TopNStrategy
where
    P: Phenotype,
{
    fn select_for_local_search(&self, population: &[P], fitness: &[f64]) -> Result<Vec<usize>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        if population.len() != fitness.len() {
            return Err(GeneticError::Configuration(format!(
                "Population size ({}) does not match fitness vector size ({})",
                population.len(),
                fitness.len()
            )));
        }

        // Adjust n if it's larger than the population
        let n = self.n.min(population.len());

        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            let ordering = fitness[a].partial_cmp(&fitness[b]).unwrap_or_else(|| {
                if fitness[a].is_nan() {
                    std::cmp::Ordering::Greater
                } else if fitness[b].is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });

            if self.higher_is_better {
                ordering.reverse()
            } else {
                ordering
            }
        });

        Ok(indices.into_iter().take(n).collect())
    }
}

/// A strategy that applies local search to a percentage of the top individuals by fitness.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct TopPercentStrategy {
    percent: f64,
    higher_is_better: bool,
}

impl TopPercentStrategy {
    /// Creates a new strategy that applies local search to a percentage of the top individuals.
    ///
    /// # Arguments
    ///
    /// * `percent` - The percentage of top individuals to select (0.0 to 1.0).
    /// * `higher_is_better` - Whether higher fitness is better (true) or lower fitness is better (false).
    ///
    /// # Errors
    ///
    /// Returns an error if `percent` is not in the range [0.0, 1.0].
    pub fn new(percent: f64, higher_is_better: bool) -> Result<Self> {
        if !(0.0..=1.0).contains(&percent) {
            return Err(GeneticError::Configuration(
                "Percentage must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self {
            percent,
            higher_is_better,
        })
    }

    /// Creates a new strategy that applies local search to a percentage of the top individuals,
    /// assuming higher fitness is better.
    ///
    /// # Arguments
    ///
    /// * `percent` - The percentage of top individuals to select (0.0 to 1.0).
    ///
    /// # Errors
    ///
    /// Returns an error if `percent` is not in the range [0.0, 1.0].
    pub fn new_maximizing(percent: f64) -> Result<Self> {
        Self::new(percent, true)
    }

    /// Creates a new strategy that applies local search to a percentage of the top individuals,
    /// assuming lower fitness is better.
    ///
    /// # Arguments
    ///
    /// * `percent` - The percentage of top individuals to select (0.0 to 1.0).
    ///
    /// # Errors
    ///
    /// Returns an error if `percent` is not in the range [0.0, 1.0].
    pub fn new_minimizing(percent: f64) -> Result<Self> {
        Self::new(percent, false)
    }
}

impl<P> LocalSearchApplicationStrategy<P> for TopPercentStrategy
where
    P: Phenotype,
{
    fn select_for_local_search(&self, population: &[P], fitness: &[f64]) -> Result<Vec<usize>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        if population.len() != fitness.len() {
            return Err(GeneticError::Configuration(format!(
                "Population size ({}) does not match fitness vector size ({})",
                population.len(),
                fitness.len()
            )));
        }

        let n = (population.len() as f64 * self.percent).ceil() as usize;

        let mut indices: Vec<usize> = (0..population.len()).collect();
        indices.sort_by(|&a, &b| {
            let ordering = fitness[a].partial_cmp(&fitness[b]).unwrap_or_else(|| {
                if fitness[a].is_nan() {
                    std::cmp::Ordering::Greater
                } else if fitness[b].is_nan() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            });

            if self.higher_is_better {
                ordering.reverse()
            } else {
                ordering
            }
        });

        Ok(indices.into_iter().take(n).collect())
    }
}

/// A strategy that applies local search to individuals with a certain probability.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy)]
pub struct ProbabilisticStrategy {
    /// The probability of applying local search to each individual (0.0 to 1.0).
    probability: f64,
}

impl ProbabilisticStrategy {
    /// Creates a new strategy that applies local search to individuals with a certain probability.
    ///
    /// # Arguments
    ///
    /// * `probability` - The probability of applying local search to each individual (0.0 to 1.0).
    ///
    /// # Errors
    ///
    /// Returns an error if `probability` is not in the range [0.0, 1.0].
    pub fn new(probability: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&probability) {
            return Err(GeneticError::Configuration(
                "Probability must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self { probability })
    }
}

impl<P> LocalSearchApplicationStrategy<P> for ProbabilisticStrategy
where
    P: Phenotype,
{
    fn select_for_local_search(&self, population: &[P], fitness: &[f64]) -> Result<Vec<usize>> {
        if population.is_empty() {
            return Err(GeneticError::EmptyPopulation);
        }
        if population.len() != fitness.len() {
            return Err(GeneticError::Configuration(format!(
                "Population size ({}) does not match fitness vector size ({})",
                population.len(),
                fitness.len()
            )));
        }

        let mut rng = RandomNumberGenerator::new();

        let random_values = rng.fetch_uniform(0.0, 1.0, population.len());
        if random_values.len() != population.len() {
            return Err(GeneticError::RandomGeneration(
                "Failed to generate sufficient random values for probabilistic selection"
                    .to_string(),
            ));
        }

        let mut selected = Vec::new();
        for (i, &random) in random_values.iter().enumerate() {
            if random < self.probability as f32 {
                selected.push(i);
            }
        }

        Ok(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_all_individuals_strategy() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![1.0, 2.0, 3.0];
        let strategy = AllIndividualsStrategy::new();

        let result = strategy
            .select_for_local_search(&population, &fitness)
            .unwrap();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_top_n_strategy_maximizing() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![1.0, 2.0, 3.0];
        let strategy = TopNStrategy::new_maximizing(2);

        let result = strategy
            .select_for_local_search(&population, &fitness)
            .unwrap();
        assert_eq!(result, vec![2, 1]); // Indices of highest fitness first
    }

    #[test]
    fn test_top_n_strategy_minimizing() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![1.0, 2.0, 3.0];
        let strategy = TopNStrategy::new_minimizing(2);

        let result = strategy
            .select_for_local_search(&population, &fitness)
            .unwrap();
        assert_eq!(result, vec![0, 1]); // Indices of lowest fitness first
    }

    #[test]
    fn test_top_percent_strategy() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
            TestPhenotype { value: 4.0 },
            TestPhenotype { value: 5.0 },
        ];
        let fitness = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let strategy = TopPercentStrategy::new_maximizing(0.4).unwrap(); // 40% = 2 individuals

        let result = strategy
            .select_for_local_search(&population, &fitness)
            .unwrap();
        assert_eq!(result, vec![4, 3]); // Indices of highest fitness first
    }

    #[test]
    fn test_probabilistic_strategy() {
        let population = vec![
            TestPhenotype { value: 1.0 },
            TestPhenotype { value: 2.0 },
            TestPhenotype { value: 3.0 },
        ];
        let fitness = vec![1.0, 2.0, 3.0];
        let strategy = ProbabilisticStrategy::new(1.0).unwrap(); // 100% probability

        let result = strategy
            .select_for_local_search(&population, &fitness)
            .unwrap();
        assert_eq!(result.len(), 3); // All individuals selected with 100% probability
    }

    #[test]
    fn test_empty_population() {
        let population: Vec<TestPhenotype> = vec![];
        let fitness: Vec<f64> = vec![];
        let strategy = AllIndividualsStrategy::new();

        let result = strategy.select_for_local_search(&population, &fitness);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), GeneticError::EmptyPopulation));
    }

    #[test]
    fn test_mismatched_lengths() {
        let population = vec![TestPhenotype { value: 1.0 }, TestPhenotype { value: 2.0 }];
        let fitness = vec![1.0, 2.0, 3.0]; // One more than population
        let strategy = AllIndividualsStrategy::new();

        let result = strategy.select_for_local_search(&population, &fitness);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GeneticError::Configuration(_)
        ));
    }
}
