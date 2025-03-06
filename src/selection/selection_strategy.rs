use std::fmt::Debug;

use crate::error::Result;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

/// Trait for selection strategies in genetic algorithms.
///
/// Selection strategies are responsible for choosing individuals from a population
/// based on their fitness scores. Different selection strategies can be used to
/// achieve different evolutionary behaviors.
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
///     let mut rng = RandomNumberGenerator::new();
///     
///     let selection = ElitistSelection::new();
///     let selected = selection.select(&population, &fitness, 2, Some(&mut rng))?;
///     
///     assert_eq!(selected.len(), 2);
///     
///     Ok(())
/// }
/// ```
pub trait SelectionStrategy<P>: Debug + Send + Sync
where
    P: Phenotype,
{
    /// Selects individuals from the population based on their fitness scores.
    ///
    /// # Arguments
    ///
    /// * `population` - The current population of individuals.
    /// * `fitness` - The fitness scores corresponding to each individual in the population.
    /// * `num_to_select` - The number of individuals to select.
    /// * `rng` - An optional random number generator for selection strategies that use randomness.
    ///           If a strategy requires randomness but `rng` is `None`, an error will be returned.
    ///
    /// # Returns
    ///
    /// A vector of selected individuals.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The population is empty
    /// - The fitness vector length doesn't match the population length
    /// - The selection process requires randomness but `rng` is `None`
    /// - The selection process encounters an error (e.g., random number generation fails)
    fn select(
        &self,
        population: &[P],
        fitness: &[f64],
        num_to_select: usize,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Vec<P>>;
}
