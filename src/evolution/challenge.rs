use crate::phenotype::Phenotype;

pub trait Challenge<Pheno: Phenotype> {
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
