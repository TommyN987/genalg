use std::collections::VecDeque;
use std::marker::PhantomData;

use crate::error::{GeneticError, Result};
use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

use super::LocalSearch;

/// A tabu search algorithm.
///
/// Tabu search is a local search algorithm that maintains a list of recently
/// visited solutions (the tabu list) to avoid cycling and to escape local optima.
#[derive(Debug, Clone)]
pub struct TabuSearch<P>
where
    P: Phenotype + Eq,
{
    /// The maximum number of iterations to perform
    max_iterations: usize,
    /// The maximum number of neighbors to evaluate per iteration
    max_neighbors: usize,
    /// The maximum size of the tabu list
    tabu_list_size: usize,
    _marker: PhantomData<P>,
}

impl<P> TabuSearch<P>
where
    P: Phenotype + Eq,
{
    /// Creates a new tabu search algorithm with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `max_iterations` - The maximum number of iterations to perform.
    /// * `max_neighbors` - The maximum number of neighbors to evaluate per iteration.
    /// * `tabu_list_size` - The maximum size of the tabu list.
    ///
    /// # Returns
    ///
    /// A new tabu search algorithm.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `max_iterations` is 0
    /// - `max_neighbors` is 0
    /// - `tabu_list_size` is 0
    pub fn new(max_iterations: usize, max_neighbors: usize, tabu_list_size: usize) -> Result<Self> {
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
        if tabu_list_size == 0 {
            return Err(GeneticError::Configuration(
                "Tabu list size must be greater than 0".to_string(),
            ));
        }
        Ok(Self {
            max_iterations,
            max_neighbors,
            tabu_list_size,
            _marker: PhantomData,
        })
    }
}

impl<P, C> LocalSearch<P, C> for TabuSearch<P>
where
    P: Phenotype + Eq,
    C: Challenge<P>,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let initial_score = challenge.score(phenotype);
        let mut current_solution = phenotype.clone();
        let mut current_score = initial_score;
        let mut best_solution = current_solution.clone();
        let mut best_score = current_score;
        let mut tabu_list = VecDeque::with_capacity(self.tabu_list_size);
        let mut improved = false;
        let mut rng = RandomNumberGenerator::new();

        for _ in 0..self.max_iterations {
            let mut best_neighbor = None;
            let mut best_neighbor_score = f64::NEG_INFINITY;

            for _ in 0..self.max_neighbors {
                let mut neighbor = current_solution.clone();
                neighbor.mutate(&mut rng);
                let neighbor_score = challenge.score(&neighbor);

                // Aspiration criteria: Allow tabu moves if they're better than the global best
                let is_better_than_best = neighbor_score > best_score;
                let is_not_tabu = !tabu_list.contains(&neighbor);

                if (is_better_than_best || is_not_tabu) && neighbor_score > best_neighbor_score {
                    best_neighbor = Some(neighbor);
                    best_neighbor_score = neighbor_score;
                }
            }

            if let Some(neighbor) = best_neighbor {
                current_solution = neighbor.clone();
                current_score = best_neighbor_score;

                if current_score > best_score {
                    best_solution = current_solution.clone();
                    best_score = current_score;
                    improved = true;
                }

                tabu_list.push_back(current_solution.clone());
                if tabu_list.len() > self.tabu_list_size {
                    tabu_list.pop_front();
                }
            } else {
                break;
            }
        }

        if improved {
            *phenotype = best_solution;
        }

        improved
    }
}
