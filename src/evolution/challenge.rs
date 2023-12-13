use crate::phenotype::Phenotype;

pub trait Challenge<Pheno: Phenotype> {
    fn score(&self, phenotype: &Pheno) -> f64;
    fn get_target(&self) -> f64;
}
