use crate::{
    error::{GeneticError, Result},
    local_search::{LocalSearchApplicationStrategy, LocalSearchManager},
    BreedStrategy, LocalSearch, Phenotype, SelectionStrategy,
};

use super::{Challenge, EvolutionLauncher};

/// Builder for creating an `EvolutionLauncher` instance.
///
/// This builder provides a fluent interface for constructing an `EvolutionLauncher`
/// with all required components. It allows setting each component individually
/// and validates that all required components are provided before building.
///
/// # Type Parameters
///
/// * `P` - The phenotype type that represents individuals in the population
/// * `B` - The breeding strategy type
/// * `S` - The selection strategy type
/// * `LS` - The local search algorithm type
/// * `F` - The challenge (fitness function) type
/// * `A` - The local search application strategy type
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
    /// Creates a new `EvolutionLauncherBuilder` instance.
    ///
    /// # Returns
    ///
    /// A new `EvolutionLauncherBuilder` with no components set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// // Create a new builder
    /// let builder = EvolutionLauncher::<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >::builder();
    /// ```
    pub fn new() -> Self {
        Self {
            breed_strategy: None,
            selection_strategy: None,
            local_search_manager: None,
            challenge: None,
        }
    }

    /// Sets the breeding strategy for the launcher.
    ///
    /// # Arguments
    ///
    /// * `breed_strategy` - The breeding strategy to use.
    ///
    /// # Returns
    ///
    /// The builder with the breeding strategy set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// let breed_strategy = OrdinaryStrategy::default();
    ///
    /// let builder = EvolutionLauncher::<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >::builder()
    ///     .with_breed_strategy(breed_strategy);
    /// ```
    pub fn with_breed_strategy(mut self, breed_strategy: B) -> Self {
        self.breed_strategy = Some(breed_strategy);
        self
    }

    /// Sets the selection strategy for the launcher.
    ///
    /// # Arguments
    ///
    /// * `selection_strategy` - The selection strategy to use.
    ///
    /// # Returns
    ///
    /// The builder with the selection strategy set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// let selection_strategy = ElitistSelection::default();
    ///
    /// let builder = EvolutionLauncher::<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >::builder()
    ///     .with_selection_strategy(selection_strategy);
    /// ```
    pub fn with_selection_strategy(mut self, selection_strategy: S) -> Self {
        self.selection_strategy = Some(selection_strategy);
        self
    }

    /// Sets the local search manager for the launcher.
    ///
    /// This method creates a `LocalSearchManager` from the provided local search
    /// algorithm and application strategy.
    ///
    /// # Arguments
    ///
    /// * `local_search_strategy` - The local search algorithm to use.
    /// * `local_search_application_strategy` - The strategy for selecting individuals for local search.
    ///
    /// # Returns
    ///
    /// The builder with the local search manager set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::error::Result;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// fn create_builder() -> Result<EvolutionLauncher<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >> {
    ///     let hill_climbing = HillClimbing::new(10)?;
    ///     let application_strategy = AllIndividualsStrategy::new();
    ///     
    ///     EvolutionLauncher::builder()
    ///         .with_breed_strategy(OrdinaryStrategy::default())
    ///         .with_selection_strategy(ElitistSelection::default())
    ///         .with_local_search_manager(hill_climbing, application_strategy)
    ///         .with_challenge(MyChallenge)
    ///         .build()
    /// }
    /// ```
    pub fn with_local_search_manager(
        mut self,
        local_search_strategy: LS,
        local_search_application_strategy: A,
    ) -> Self {
        self.local_search_manager = Some(LocalSearchManager::new(
            local_search_strategy,
            local_search_application_strategy,
        ));
        self
    }

    /// Sets the challenge for the launcher.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to use for evaluating phenotypes.
    ///
    /// # Returns
    ///
    /// The builder with the challenge set.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// let challenge = MyChallenge;
    ///
    /// let builder = EvolutionLauncher::<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >::builder()
    ///     .with_challenge(challenge);
    /// ```
    pub fn with_challenge(mut self, challenge: F) -> Self {
        self.challenge = Some(challenge);
        self
    }

    /// Builds an `EvolutionLauncher` from the configured components.
    ///
    /// This method validates that all required components are set and creates
    /// an `EvolutionLauncher` instance.
    ///
    /// # Returns
    ///
    /// A `Result` containing the `EvolutionLauncher` if all required components
    /// are set, or a `GeneticError` if any required component is missing.
    ///
    /// # Errors
    ///
    /// This method will return an error if:
    /// - The breeding strategy is not set
    /// - The selection strategy is not set
    /// - The challenge is not set
    ///
    /// # Example
    ///
    /// ```rust
    /// # use genalg::evolution::EvolutionLauncher;
    /// # use genalg::phenotype::Phenotype;
    /// # use genalg::rng::RandomNumberGenerator;
    /// # use  genalg::breeding::OrdinaryStrategy;
    /// # use genalg::selection::ElitistSelection;
    /// # use genalg::local_search::{HillClimbing, AllIndividualsStrategy};
    /// # use genalg::evolution::Challenge;
    /// # use genalg::error::Result;
    /// #
    /// # #[derive(Clone, Debug)]
    /// # struct MyPhenotype { value: f64 }
    /// # impl Phenotype for MyPhenotype {
    /// #     fn crossover(&mut self, other: &Self) {}
    /// #     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {}
    /// # }
    /// #
    /// # #[derive(Clone)]
    /// # struct MyChallenge;
    /// # impl Challenge<MyPhenotype> for MyChallenge {
    /// #     fn score(&self, _phenotype: &MyPhenotype) -> f64 { 0.0 }
    /// # }
    ///
    /// fn create_launcher() -> Result<EvolutionLauncher<
    ///     MyPhenotype,
    ///     OrdinaryStrategy,
    ///     ElitistSelection,
    ///     HillClimbing,
    ///     MyChallenge,
    ///     AllIndividualsStrategy
    /// >> {
    ///     EvolutionLauncher::builder()
    ///         .with_breed_strategy(OrdinaryStrategy::default())
    ///         .with_selection_strategy(ElitistSelection::default())
    ///         .with_challenge(MyChallenge)
    ///         // Local search is optional
    ///         .build()
    /// }
    /// ```
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
