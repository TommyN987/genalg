use crate::error::{GeneticError, Result};
use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

use super::LocalSearch;

/// A simulated annealing algorithm.
///
/// Simulated annealing is a probabilistic local search algorithm that allows
/// moves to worse solutions with a probability that decreases over time.
#[derive(Debug, Clone)]
pub struct SimulatedAnnealing {
    max_iterations: usize,
    initial_temperature: f64,
    cooling_rate: f64,
}

impl SimulatedAnnealing {
    /// Creates a new simulated annealing algorithm with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `initial_temperature` - The initial temperature.
    /// * `cooling_rate` - The cooling rate (between 0 and 1).
    ///
    /// # Returns
    ///
    /// A new simulated annealing algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_iterations` is 0
    /// - `initial_temperature` is not positive
    /// - `cooling_rate` is not between 0 and 1
    pub fn new(max_iterations: usize, initial_temperature: f64, cooling_rate: f64) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }
        if initial_temperature <= 0.0 {
            return Err(GeneticError::Configuration(
                "Initial temperature must be positive".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&cooling_rate) {
            return Err(GeneticError::Configuration(
                "Cooling rate must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            initial_temperature,
            cooling_rate,
        })
    }
}

impl<P, C> LocalSearch<P, C> for SimulatedAnnealing
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut current_solution = phenotype.clone();
        let mut best_solution = current_solution.clone();
        let mut current_score = challenge.score(&current_solution);
        let mut best_score = current_score;
        let mut temperature = self.initial_temperature;
        let mut rng = RandomNumberGenerator::new();

        for _ in 0..self.max_iterations {
            let mut neighbor = current_solution.clone();
            neighbor.mutate(&mut rng);
            let neighbor_score = challenge.score(&neighbor);

            let accept = if neighbor_score > current_score {
                true
            } else {
                let delta = neighbor_score - current_score;
                let probability = (delta / temperature).exp();
                let random: f64 = rng.fetch_uniform(0.0, 1.0, 1)[0] as f64;
                random < probability
            };

            if accept {
                current_solution = neighbor;
                current_score = neighbor_score;

                if current_score > best_score {
                    best_solution = current_solution.clone();
                    best_score = current_score;
                }
            }

            // Prevent temperature from reaching 0 too soon
            temperature *= self.cooling_rate.max(0.0001);
        }

        if best_score > challenge.score(phenotype) {
            *phenotype = best_solution;
            return true;
        }

        false
    }
}
