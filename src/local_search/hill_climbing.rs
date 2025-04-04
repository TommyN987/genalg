use crate::error::{GeneticError, Result};
use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

use super::LocalSearch;

/// A simple hill climbing algorithm.
///
/// Random Restart Hill climbing is a local search algorithm that iteratively moves to better
/// neighboring solutions until no better solution can be found. The random restart is used to
/// avoid local optima.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone)]
pub struct HillClimbing {
    max_iterations: usize,
    max_neighbors: usize,
    restart_probability: f64,
}

impl HillClimbing {
    /// Creates a new hill climbing algorithm with the given maximum number of iterations.
    /// Restart probability is set to 0.05.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `max_neighbors` - The maximum number of neighbors to evaluate per iteration.
    ///
    /// # Returns
    ///
    /// A new hill climbing algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if `max_iterations` or `max_neighbors` is 0.
    pub fn new(max_iterations: usize, max_neighbors: usize) -> Result<Self> {
        if max_iterations == 0 {
            return Err(GeneticError::Configuration(
                "Maximum iterations must be greater than 0".to_string(),
            ));
        }

        if max_neighbors == 0 {
            return Err(GeneticError::Configuration(
                "Maximum neighbors must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            max_iterations,
            max_neighbors,
            restart_probability: 0.05,
        })
    }

    /// Overrides the default restart probability.
    ///
    /// # Arguments
    ///
    /// * `restart_probability` - The probability of restarting the search.
    ///
    /// # Returns
    ///
    /// A new hill climbing algorithm with the given restart probability.
    ///
    /// # Errors
    ///
    /// Returns an error if the restart probability is not between 0.0 and 1.0.
    pub fn with_restart_probability(mut self, restart_probability: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&restart_probability) {
            return Err(GeneticError::Configuration(
                "Restart probability must be between 0.0 and 1.0".to_string(),
            ));
        }

        self.restart_probability = restart_probability;
        Ok(self)
    }
}

impl<P, C> LocalSearch<P, C> for HillClimbing
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_solution = phenotype.clone();
        let mut current_score = initial_score;
        let mut improved = false;
        let mut rng = RandomNumberGenerator::new();

        for _ in 0..self.max_iterations {
            let mut best_neighbor = None;
            let mut best_neighbor_score = current_score;

            for _ in 0..self.max_neighbors {
                let mut neighbor = current_solution.clone();
                neighbor.mutate(&mut rng);
                let neighbor_score = challenge.score(&neighbor);

                if neighbor_score > best_neighbor_score {
                    best_neighbor = Some(neighbor);
                    best_neighbor_score = neighbor_score;
                }
            }

            if let Some(best) = best_neighbor {
                current_solution = best;
                current_score = best_neighbor_score;
                improved = true;
            } else {
                let should_restart: f64 = match rng.fetch_uniform(0.0, 1.0, 1).front() {
                    Some(value) => *value as f64,
                    None => {
                        return false;
                    }
                };
                if should_restart < self.restart_probability {
                    current_solution.mutate(&mut rng);
                    current_score = challenge.score(&current_solution);
                } else {
                    break;
                }
            }
        }

        if improved {
            *phenotype = current_solution;
        }

        improved
    }
}
