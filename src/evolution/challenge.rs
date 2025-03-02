use crate::phenotype::Phenotype;

/// Trait for evaluating the fitness of phenotypes.
///
/// This trait defines the interface for evaluating the fitness of phenotypes
/// in an evolutionary algorithm. It requires a method for calculating a fitness
/// score for a given phenotype.
///
/// Types implementing this trait should be thread-safe (implement `Send` and `Sync`)
/// to enable parallel fitness evaluation.
pub trait Challenge<Pheno: Phenotype>: Send + Sync {
    /// Calculates the fitness score of a given phenotype.
    ///
    /// # Arguments
    ///
    /// * `phenotype` - The phenotype to be evaluated.
    ///
    /// # Returns
    ///
    /// The fitness score of the phenotype as a floating-point number.
    fn score(&self, phenotype: &Pheno) -> f64;
}
