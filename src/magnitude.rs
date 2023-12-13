use crate::phenotype::Phenotype;

pub trait Magnitude<Pheno: Phenotype> {
    fn magnitude(&self) -> f64;
    fn min_magnitude(&self) -> f64;
    fn max_magnitude(&self) -> f64;
}
