use std::fmt::Debug;
use std::sync::Arc;

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;

use super::LocalSearch;

/// A hybrid local search algorithm that combines multiple local search algorithms.
///
/// This algorithm applies multiple local search algorithms in sequence.
#[derive(Debug, Clone)]
pub struct HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    algorithms: Vec<Arc<dyn LocalSearch<P, C>>>,
}

impl<P, C> HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    /// Creates a new empty hybrid local search algorithm.
    pub fn new() -> Self {
        Self {
            algorithms: Vec::new(),
        }
    }

    /// Adds a local search algorithm to the hybrid.
    pub fn add_algorithm<L>(&mut self, algorithm: L) -> &mut Self
    where
        L: LocalSearch<P, C> + 'static,
    {
        self.algorithms.push(Arc::new(algorithm));
        self
    }
}

impl<P, C> LocalSearch<P, C> for HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    fn search(&self, phenotype: &mut P, challenge: &C) -> bool {
        let mut improved = false;

        for algorithm in &self.algorithms {
            if algorithm.search(phenotype, challenge) {
                improved = true;
            }
        }

        improved
    }
}

impl<P, C> Default for HybridLocalSearch<P, C>
where
    P: Phenotype,
    C: Challenge<P> + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}
