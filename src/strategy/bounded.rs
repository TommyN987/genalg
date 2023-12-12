use std::{cell::RefCell, fmt::Error, marker::PhantomData};

use crate::{
    evol_options::EvolutionOptions, magnitude::Magnitude, phenotype::Phenotype,
    rng::RandomNumberGenerator,
};

use super::BreedStrategy;

pub struct BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude,
{
    _marker: PhantomData<Pheno>,
}

impl<Pheno> BreedStrategy<Pheno> for BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude,
{
    fn breed(
        &self,
        parents: &[Pheno],
        evol_options: &EvolutionOptions,
        rng: &mut RandomNumberGenerator,
    ) -> Result<Vec<Pheno>, Error> {
        let mut children: Vec<Pheno> = Vec::new();
        let mut winner_previous_generation = RefCell::new(parents[0]);

        children.push(self.develop(*winner_previous_generation.get_mut(), rng, false)?);

        parents
            .iter()
            .skip(1)
            .try_for_each(|parent| -> Result<(), Error> {
                let mut child = *winner_previous_generation.get_mut();
                child.crossover(parent);
                let mutated_child = self.develop(child, rng, true)?;
                children.push(mutated_child);
                Ok(())
            })?;

        (parents.len()..evol_options.get_num_offspring()).try_for_each(
            |_| -> Result<(), Error> {
                let child = *winner_previous_generation.get_mut();
                let mutated_child = self.develop(child, rng, true)?;
                children.push(mutated_child);
                Ok(())
            },
        )?;

        Ok(children)
    }
}

impl<Pheno> BoundedBreedStrategy<Pheno>
where
    Pheno: Phenotype + Magnitude,
{
    fn develop(
        &self,
        pheno: Pheno,
        rng: &mut RandomNumberGenerator,
        initial_mutate: bool,
    ) -> Result<Pheno, Error> {
        let mut phenotype = pheno;

        if initial_mutate {
            phenotype.mutate(rng);
        }

        let pheno_type_in_range = |ph: &Pheno| -> bool {
            ph.magnitude() >= ph.min_magnitude() && ph.magnitude() <= ph.max_magnitude()
        };

        let mut advance_pheno_n_times = |ph: Pheno, n: usize| -> Pheno {
            let mut phenotype = ph;
            for _ in 0..n {
                phenotype.mutate(rng);
                if pheno_type_in_range(&phenotype) {
                    break;
                }
            }
            phenotype
        };

        for _ in 0..1000 {
            if pheno_type_in_range(&phenotype) {
                return Ok(phenotype);
            }
            phenotype = advance_pheno_n_times(phenotype, 1000);
        }

        Err(Error)
    }
}
