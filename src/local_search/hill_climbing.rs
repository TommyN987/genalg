use crate::error::{GeneticError, Result};
use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

use super::LocalSearch;

/// A simple hill climbing algorithm.
///
/// Hill climbing is a local search algorithm that iteratively moves to better
/// neighboring solutions until no better solution can be found.
#[derive(Debug, Clone)]
pub struct HillClimbing {
    max_iterations: usize,
    max_neighbors: usize,
}

impl HillClimbing {
    /// Creates a new hill climbing algorithm with the given maximum number of iterations.
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
        })
    }
}

impl<P, C> LocalSearch<P, C> for HillClimbing
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_score = initial_score;
        let mut improved = false;
        let mut rng = RandomNumberGenerator::new();

        for _ in 0..self.max_iterations {
            let mut best_neighbor = phenotype.clone();
            let mut best_neighbor_score = current_score;
            let mut found_better = false;

            for _ in 0..self.max_neighbors {
                let mut neighbor = phenotype.clone();
                neighbor.mutate(&mut rng);
                let neighbor_score = challenge.score(&neighbor);

                if neighbor_score > best_neighbor_score {
                    best_neighbor = neighbor;
                    best_neighbor_score = neighbor_score;
                    found_better = true;
                }
            }

            if found_better {
                *phenotype = best_neighbor;
                current_score = best_neighbor_score;
                improved = true;
            } else {
                break;
            }
        }

        improved
    }
}
