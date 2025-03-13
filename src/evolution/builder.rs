use crate::{
    error::{GeneticError, Result},
    local_search::{LocalSearchApplicationStrategy, LocalSearchManager},
    BreedStrategy, LocalSearch, Phenotype, SelectionStrategy,
};

use super::{Challenge, EvolutionLauncher};

pub struct EvolutionLauncherBuilder<P, B, S, LS, F, A>
where
    P: Phenotype,
    B: BreedStrategy<P>,
    S: SelectionStrategy<P>,
    LS: LocalSearch<P, F> + Clone,
    F: Challenge<P> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
{
    breed_strategy: Option<B>,
    selection_strategy: Option<S>,
    local_search_manager: Option<LocalSearchManager<P, LS, A, F>>,
    challenge: Option<F>,
}

impl<P, B, S, LS, F, A> EvolutionLauncherBuilder<P, B, S, LS, F, A>
where
    P: Phenotype,
    B: BreedStrategy<P> + Clone,
    S: SelectionStrategy<P> + Clone,
    LS: LocalSearch<P, F> + Clone,
    F: Challenge<P> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
{
    pub fn new() -> Self {
        Self {
            breed_strategy: None,
            selection_strategy: None,
            local_search_manager: None,
            challenge: None,
        }
    }

    pub fn with_breed_strategy(mut self, breed_strategy: B) -> Self {
        self.breed_strategy = Some(breed_strategy);
        self
    }

    pub fn with_selection_strategy(mut self, selection_strategy: S) -> Self {
        self.selection_strategy = Some(selection_strategy);
        self
    }

    pub fn with_local_search_manager(
        mut self,
        local_search_manager: LocalSearchManager<P, LS, A, F>,
    ) -> Self {
        self.local_search_manager = Some(local_search_manager);
        self
    }

    pub fn with_challenge(mut self, challenge: F) -> Self {
        self.challenge = Some(challenge);
        self
    }

    pub fn build(self) -> Result<EvolutionLauncher<P, B, S, LS, F, A>> {
        let breed_strategy = self.breed_strategy.ok_or_else(|| {
            GeneticError::Configuration("Breeding strategy not specified".to_string())
        })?;

        let selection_strategy = self.selection_strategy.ok_or_else(|| {
            GeneticError::Configuration("Selection strategy not specified".to_string())
        })?;

        let challenge = self
            .challenge
            .ok_or_else(|| GeneticError::Configuration("Challenge not specified".to_string()))?;

        Ok(EvolutionLauncher::new(
            breed_strategy,
            selection_strategy,
            self.local_search_manager,
            challenge,
        ))
    }
}

impl<P, B, S, LS, F, A> Default for EvolutionLauncherBuilder<P, B, S, LS, F, A>
where
    P: Phenotype,
    B: BreedStrategy<P> + Clone,
    S: SelectionStrategy<P> + Clone,
    LS: LocalSearch<P, F> + Clone,
    F: Challenge<P> + Clone,
    A: LocalSearchApplicationStrategy<P> + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}
