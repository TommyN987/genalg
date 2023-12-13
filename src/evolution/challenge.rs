use crate::phenotype::Phenotype;

pub trait Challenge<Pheno: Phenotype> {
    fn score(&self, phenotype: &Pheno) -> f64;
}
