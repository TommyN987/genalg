//! # Constraints Module
//!
//! This module provides a framework for defining and enforcing constraints in genetic algorithms.
//! Constraints are particularly useful for combinatorial optimization problems where solutions
//! must satisfy specific requirements to be valid.
//!
//! ## Overview
//!
//! In genetic algorithms, constraints define requirements that valid solutions must satisfy.
//! This module provides tools for:
//!
//! - Defining constraints through the `Constraint` trait
//! - Representing constraint violations with the `ConstraintViolation` struct
//! - Managing multiple constraints with the `ConstraintManager`
//! - Applying common constraints for combinatorial problems
//!
//! ## Key Components
//!
//! - `Constraint` trait: Defines the interface for constraints that can be applied to phenotypes
//! - `ConstraintViolation`: Represents a specific violation of a constraint
//! - `ConstraintManager`: Manages multiple constraints and evaluates them against phenotypes
//! - Combinatorial constraints: Pre-built constraints for common optimization problems
//!
//! ## Basic Usage
//!
//! ```rust
//! use genalg::constraints::{Constraint, ConstraintManager, ConstraintViolation};
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//! use std::fmt::Debug;
//!
//! #[derive(Clone, Debug)]
//! struct MySolution {
//!     values: Vec<usize>,
//! }
//!
//! impl Phenotype for MySolution {
//!     fn crossover(&mut self, other: &Self) {
//!         // Implementation for example
//!         if !other.values.is_empty() && !self.values.is_empty() {
//!             self.values[0] = other.values[0];
//!         }
//!     }
//!
//!     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
//!         // Implementation for example
//!         if !self.values.is_empty() {
//!             self.values[0] += 1;
//!         }
//!     }
//! }
//!
//! // Define a constraint that requires all values to be unique
//! #[derive(Debug)]
//! struct UniqueValuesConstraint;
//!
//! impl<P> Constraint<P> for UniqueValuesConstraint
//! where
//!     P: Phenotype + AsRef<Vec<usize>>,
//! {
//!     fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
//!         let values = phenotype.as_ref();
//!         let mut violations = Vec::new();
//!         
//!         // Check for duplicates
//!         for i in 0..values.len() {
//!             for j in i+1..values.len() {
//!                 if values[i] == values[j] {
//!                     violations.push(ConstraintViolation::new(
//!                         "UniqueValues",
//!                         format!("Duplicate value {} at positions {} and {}", values[i], i, j)
//!                     ));
//!                 }
//!             }
//!         }
//!         
//!         violations
//!     }
//!     
//!     fn repair(&self, phenotype: &mut P) -> bool {
//!         // Implementation of repair logic
//!         false
//!     }
//! }
//!
//! // Example usage
//! impl AsRef<Vec<usize>> for MySolution {
//!     fn as_ref(&self) -> &Vec<usize> {
//!         &self.values
//!     }
//! }
//!
//! let mut constraint_manager = ConstraintManager::new();
//! constraint_manager.add_constraint(UniqueValuesConstraint);
//!
//! let solution = MySolution { values: vec![1, 2, 3, 2, 5] };
//! let violations = constraint_manager.check_all(&solution);
//! assert_eq!(violations.len(), 1); // One duplicate value
//! ```
//!
//! ## Using Combinatorial Constraints
//!
//! The module provides a framework for defining and enforcing constraints:
//!
//! ```rust
//! use genalg::constraints::{Constraint, ConstraintManager, ConstraintViolation};
//! use genalg::phenotype::Phenotype;
//! use genalg::rng::RandomNumberGenerator;
//! use std::collections::HashSet;
//! use std::fmt::Debug;
//!
//! #[derive(Clone, Debug)]
//! struct PermutationSolution {
//!     sequence: Vec<usize>,
//! }
//!
//! impl Phenotype for PermutationSolution {
//!     fn crossover(&mut self, other: &Self) {
//!         // Implementation omitted for brevity
//!     }
//!
//!     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
//!         // Implementation omitted for brevity
//!     }
//! }
//!
//! // Define a custom constraint that ensures all elements in the sequence are unique
//! #[derive(Debug)]
//! struct UniqueValuesConstraint;
//!
//! impl Constraint<PermutationSolution> for UniqueValuesConstraint {
//!     fn check(&self, solution: &PermutationSolution) -> Vec<ConstraintViolation> {
//!         let mut seen = HashSet::new();
//!         let mut violations = Vec::new();
//!         
//!         for (i, &value) in solution.sequence.iter().enumerate() {
//!             if !seen.insert(value) {
//!                 violations.push(ConstraintViolation::new(
//!                     "UniqueValues",
//!                     format!("Duplicate value {} at position {}", value, i)
//!                 ));
//!             }
//!         }
//!         
//!         violations
//!     }
//! }
//!
//! // Add the constraint to a manager
//! let mut manager = ConstraintManager::new();
//! manager.add_constraint(UniqueValuesConstraint);
//!
//! // Check a solution
//! let solution = PermutationSolution { sequence: vec![1, 2, 3, 4, 5] };
//! assert!(manager.is_valid(&solution));
//!
//! let invalid_solution = PermutationSolution { sequence: vec![1, 2, 3, 2, 5] };
//! assert!(!manager.is_valid(&invalid_solution));
//! ```

use std::error::Error;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::evolution::Challenge;
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

pub mod combinatorial;

// Re-export combinatorial constraints for easier access
pub use combinatorial::{
    CapacityConstraint, CompleteAssignmentConstraint, DependencyConstraint,
    UniqueElementsConstraint,
};

/// Represents a violation of a constraint.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// The name of the constraint that was violated
    constraint_name: String,
    /// A description of the violation
    description: String,
    /// An optional severity score (higher means more severe)
    severity: Option<f64>,
}

impl ConstraintViolation {
    /// Creates a new constraint violation with the given name and description.
    pub fn new<S: Into<String>, D: Into<String>>(constraint_name: S, description: D) -> Self {
        Self {
            constraint_name: constraint_name.into(),
            description: description.into(),
            severity: None,
        }
    }

    /// Creates a new constraint violation with the given name, description, and severity.
    pub fn with_severity<S: Into<String>, D: Into<String>>(
        constraint_name: S,
        description: D,
        severity: f64,
    ) -> Self {
        Self {
            constraint_name: constraint_name.into(),
            description: description.into(),
            severity: Some(severity),
        }
    }

    /// Returns the name of the constraint that was violated.
    pub fn constraint_name(&self) -> &str {
        &self.constraint_name
    }

    /// Returns the description of the violation.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the severity of the violation, if specified.
    pub fn severity(&self) -> Option<f64> {
        self.severity
    }
}

impl Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Constraint '{}' violated: {}{}",
            self.constraint_name(),
            self.description(),
            self.severity()
                .map(|s| format!(" (severity: {})", s))
                .unwrap_or_default()
        )
    }
}

/// Trait for defining constraints that can be applied to phenotypes.
///
/// Constraints check whether a phenotype satisfies certain requirements and
/// optionally provide mechanisms to repair invalid phenotypes.
pub trait Constraint<P>: Debug + Send + Sync
where
    P: Phenotype,
{
    /// Checks if the phenotype satisfies this constraint.
    ///
    /// Returns a vector of constraint violations. An empty vector indicates
    /// that the phenotype satisfies the constraint.
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation>;

    /// Attempts to repair the phenotype to satisfy this constraint.
    ///
    /// Returns `true` if the phenotype was successfully repaired, `false` otherwise.
    ///
    /// The default implementation does not perform any repair.
    fn repair(&self, _phenotype: &mut P) -> bool {
        false
    }

    /// Attempts to repair the phenotype using a random number generator.
    ///
    /// This method is useful for repair strategies that require randomness.
    ///
    /// Returns `true` if the phenotype was successfully repaired, `false` otherwise.
    ///
    /// The default implementation does not perform any repair.
    fn repair_with_rng(&self, _phenotype: &mut P, _rng: &mut RandomNumberGenerator) -> bool {
        false
    }

    /// Returns a penalty score for the given constraint violations.
    ///
    /// This score can be used to adjust the fitness of phenotypes that violate
    /// the constraint. Higher values indicate more severe violations.
    ///
    /// The default implementation sums the severity of all violations, or counts
    /// the number of violations if severity is not specified.
    fn penalty_score(&self, violations: &[ConstraintViolation]) -> f64 {
        violations.iter().map(|v| v.severity.unwrap_or(1.0)).sum()
    }
}

/// Manages multiple constraints and evaluates them against phenotypes.
#[derive(Debug, Clone)]
pub struct ConstraintManager<P>
where
    P: Phenotype,
{
    constraints: Vec<Arc<dyn Constraint<P>>>,
    _marker: PhantomData<P>,
}

impl<P> ConstraintManager<P>
where
    P: Phenotype,
{
    /// Creates a new empty constraint manager.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Creates a new constraint manager builder.
    pub fn builder() -> ConstraintManagerBuilder<P> {
        ConstraintManagerBuilder::new()
    }

    /// Adds a constraint to the manager.
    pub fn add_constraint<C>(&mut self, constraint: C) -> &mut Self
    where
        C: Constraint<P> + 'static,
    {
        self.constraints.push(Arc::new(constraint));
        self
    }

    /// Checks if the phenotype satisfies all constraints.
    ///
    /// Returns a vector of constraint violations. An empty vector indicates
    /// that the phenotype satisfies all constraints.
    pub fn check_all(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        self.constraints
            .iter()
            .flat_map(|c| c.check(phenotype))
            .collect()
    }

    /// Attempts to repair the phenotype to satisfy all constraints.
    ///
    /// Returns `true` if the phenotype was successfully repaired for all constraints,
    /// `false` if any constraint could not be repaired.
    pub fn repair_all(&self, phenotype: &mut P) -> bool {
        let mut all_repaired = true;

        for constraint in &self.constraints {
            if !constraint.repair(phenotype) {
                all_repaired = false;
            }
        }

        all_repaired
    }

    /// Attempts to repair the phenotype using a random number generator.
    ///
    /// Returns `true` if the phenotype was successfully repaired for all constraints,
    /// `false` if any constraint could not be repaired.
    pub fn repair_all_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        let mut all_repaired = true;

        for constraint in &self.constraints {
            if !constraint.repair_with_rng(phenotype, rng) {
                all_repaired = false;
            }
        }

        all_repaired
    }

    /// Calculates a total penalty score for all constraint violations.
    ///
    /// This score can be used to adjust the fitness of phenotypes that violate
    /// constraints. Higher values indicate more severe violations.
    pub fn total_penalty_score(&self, phenotype: &P) -> f64 {
        let mut total_score = 0.0;
        for constraint in &self.constraints {
            let violations = constraint.check(phenotype);
            if !violations.is_empty() {
                total_score += constraint.penalty_score(&violations);
            }
        }
        total_score
    }

    /// Checks if the phenotype is valid (satisfies all constraints).
    pub fn is_valid(&self, phenotype: &P) -> bool {
        self.check_all(phenotype).is_empty()
    }

    /// Returns the number of constraints in the manager.
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Returns `true` if the manager has no constraints.
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }
}

impl<P> Default for ConstraintManager<P>
where
    P: Phenotype,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating a constraint manager with a fluent API.
#[derive(Debug, Clone)]
pub struct ConstraintManagerBuilder<P>
where
    P: Phenotype,
{
    constraints: Vec<Arc<dyn Constraint<P>>>,
}

impl<P> ConstraintManagerBuilder<P>
where
    P: Phenotype,
{
    /// Creates a new empty constraint manager builder.
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
        }
    }

    /// Adds a constraint to the manager.
    pub fn with_constraint<C>(mut self, constraint: C) -> Self
    where
        C: Constraint<P> + 'static,
    {
        self.constraints.push(Arc::new(constraint));
        self
    }

    /// Builds the constraint manager.
    pub fn build(self) -> ConstraintManager<P> {
        ConstraintManager {
            constraints: self.constraints,
            _marker: PhantomData,
        }
    }
}

impl<P> Default for ConstraintManagerBuilder<P>
where
    P: Phenotype,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for constraint-related operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintError {
    /// The constraint name is empty.
    EmptyName,
    /// A required collection is empty.
    EmptyCollection(String),
    /// Other constraint-related errors.
    Other(String),
}

impl Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintError::EmptyName => write!(f, "Constraint name cannot be empty"),
            ConstraintError::EmptyCollection(name) => write!(f, "{} cannot be empty", name),
            ConstraintError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for ConstraintError {}

/// A challenge wrapper that adjusts fitness scores based on constraint violations.
///
/// This wrapper applies penalties to the fitness score based on constraint violations,
/// making it useful for handling soft constraints in optimization problems.
///
/// # Examples
///
/// ```
/// use genalg::constraints::{Constraint, ConstraintManager, ConstraintViolation, PenaltyAdjustedChallenge};
/// use genalg::evolution::Challenge;
/// use genalg::phenotype::Phenotype;
/// use genalg::rng::RandomNumberGenerator;
/// use std::fmt::Debug;
///
/// #[derive(Clone, Debug)]
/// struct MySolution {
///     values: Vec<usize>,
/// }
///
/// impl Phenotype for MySolution {
///     fn crossover(&mut self, other: &Self) {
///         // Implementation for example
///         if !other.values.is_empty() && !self.values.is_empty() {
///             self.values[0] = other.values[0];
///         }
///     }
///
///     fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
///         // Implementation for example
///         if !self.values.is_empty() {
///             self.values[0] += 1;
///         }
///     }
/// }
///
/// // Define a constraint that requires all values to be unique
/// #[derive(Debug)]
/// struct UniqueValuesConstraint;
///
/// impl Constraint<MySolution> for UniqueValuesConstraint {
///     fn check(&self, phenotype: &MySolution) -> Vec<ConstraintViolation> {
///         let mut seen = std::collections::HashSet::new();
///         let mut violations = Vec::new();
///         
///         for (i, &value) in phenotype.values.iter().enumerate() {
///             if !seen.insert(value) {
///                 violations.push(ConstraintViolation::new(
///                     "UniqueValues",
///                     format!("Duplicate value {} at position {}", value, i)
///                 ));
///             }
///         }
///         
///         violations
///     }
/// }
///
/// // Define a challenge
/// struct MyChallenge;
///
/// impl Challenge<MySolution> for MyChallenge {
///     fn score(&self, phenotype: &MySolution) -> f64 {
///         // Simple scoring function
///         phenotype.values.iter().sum::<usize>() as f64
///     }
/// }
///
/// // Create a constraint manager
/// let mut constraint_manager = ConstraintManager::new();
/// constraint_manager.add_constraint(UniqueValuesConstraint);
///
/// // Create a penalty-adjusted challenge
/// let challenge = MyChallenge;
/// let penalty_challenge = PenaltyAdjustedChallenge::new(challenge, constraint_manager, 10.0);
///
/// // Create a solution with a constraint violation
/// let solution = MySolution { values: vec![1, 2, 3, 2, 5] };
///
/// // Score will be reduced by the penalty
/// let base_score = 1 + 2 + 3 + 2 + 5; // = 13
/// let penalty = 10.0; // One violation with weight 10.0
/// let expected_score = base_score as f64 - penalty; // = 3.0
/// assert_eq!(penalty_challenge.score(&solution), expected_score);
/// ```
#[derive(Debug, Clone)]
pub struct PenaltyAdjustedChallenge<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    /// The wrapped challenge
    challenge: C,
    /// The constraint manager
    constraint_manager: ConstraintManager<P>,
    /// The weight to apply to penalties
    penalty_weight: f64,
    /// Phantom data for the phenotype type
    _marker: PhantomData<P>,
}

impl<P, C> PenaltyAdjustedChallenge<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    /// Creates a new penalty-adjusted challenge.
    ///
    /// # Arguments
    ///
    /// * `challenge` - The challenge to wrap.
    /// * `constraint_manager` - The constraint manager to use for calculating penalties.
    /// * `penalty_weight` - The weight to apply to penalties when adjusting fitness.
    ///
    /// # Returns
    ///
    /// A new penalty-adjusted challenge.
    pub fn new(
        challenge: C,
        constraint_manager: ConstraintManager<P>,
        penalty_weight: f64,
    ) -> Self {
        Self {
            challenge,
            constraint_manager,
            penalty_weight,
            _marker: PhantomData,
        }
    }

    /// Returns a reference to the wrapped challenge.
    pub fn inner(&self) -> &C {
        &self.challenge
    }

    /// Returns a reference to the constraint manager.
    pub fn constraint_manager(&self) -> &ConstraintManager<P> {
        &self.constraint_manager
    }

    /// Returns the penalty weight.
    pub fn penalty_weight(&self) -> f64 {
        self.penalty_weight
    }
}

impl<P, C> Challenge<P> for PenaltyAdjustedChallenge<P, C>
where
    P: Phenotype,
    C: Challenge<P>,
{
    fn score(&self, phenotype: &P) -> f64 {
        let base_score = self.challenge.score(phenotype);
        let penalty = self.constraint_manager.total_penalty_score(phenotype) * self.penalty_weight;

        // Adjust score based on penalty (lower is worse)
        base_score - penalty
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RandomNumberGenerator;
    use std::collections::HashSet;

    #[derive(Clone, Debug)]
    struct TestPhenotype {
        values: Vec<usize>,
    }

    impl Phenotype for TestPhenotype {
        fn crossover(&mut self, other: &Self) {
            if !other.values.is_empty() && !self.values.is_empty() {
                self.values[0] = other.values[0];
            }
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            if !self.values.is_empty() {
                self.values[0] += 1;
            }
        }
    }

    impl AsRef<Vec<usize>> for TestPhenotype {
        fn as_ref(&self) -> &Vec<usize> {
            &self.values
        }
    }

    impl AsMut<Vec<usize>> for TestPhenotype {
        fn as_mut(&mut self) -> &mut Vec<usize> {
            &mut self.values
        }
    }

    #[derive(Debug)]
    struct UniqueValuesConstraint;

    impl Constraint<TestPhenotype> for UniqueValuesConstraint {
        fn check(&self, phenotype: &TestPhenotype) -> Vec<ConstraintViolation> {
            let values = phenotype.as_ref();
            let mut seen = HashSet::new();
            let mut violations = Vec::new();

            for (idx, &value) in values.iter().enumerate() {
                if !seen.insert(value) {
                    violations.push(ConstraintViolation::new(
                        "UniqueValues",
                        format!("Duplicate value {} at position {}", value, idx),
                    ));
                }
            }

            violations
        }

        fn repair(&self, phenotype: &mut TestPhenotype) -> bool {
            let values = phenotype.as_mut();
            let mut seen = HashSet::new();
            let mut modified = false;

            for i in 0..values.len() {
                let mut value = values[i];
                while !seen.insert(value) {
                    value += 1;
                    modified = true;
                }
                values[i] = value;
            }

            modified
        }
    }

    #[test]
    fn test_constraint_violation() {
        let violation = ConstraintViolation::new("Test", "Test violation");
        assert_eq!(violation.constraint_name(), "Test");
        assert_eq!(violation.description(), "Test violation");
        assert!(violation.severity().is_none());

        let violation_with_severity =
            ConstraintViolation::with_severity("Test", "Test violation", 2.0);
        assert_eq!(violation_with_severity.constraint_name(), "Test");
        assert_eq!(violation_with_severity.description(), "Test violation");
        assert_eq!(violation_with_severity.severity(), Some(2.0));
    }

    #[test]
    fn test_constraint_check() {
        let constraint = UniqueValuesConstraint;

        // Valid phenotype
        let valid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 4, 5],
        };
        let violations = constraint.check(&valid_phenotype);
        assert!(violations.is_empty());

        // Invalid phenotype
        let invalid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };
        let violations = constraint.check(&invalid_phenotype);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].constraint_name(), "UniqueValues");
    }

    #[test]
    fn test_constraint_repair() {
        let constraint = UniqueValuesConstraint;

        // Invalid phenotype
        let mut invalid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };
        let repaired = constraint.repair(&mut invalid_phenotype);

        assert!(repaired);
        assert_eq!(invalid_phenotype.values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_constraint_manager() {
        let mut manager = ConstraintManager::new();
        manager.add_constraint(UniqueValuesConstraint);

        // Valid phenotype
        let valid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 4, 5],
        };
        assert!(manager.is_valid(&valid_phenotype));

        // Invalid phenotype
        let invalid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };
        assert!(!manager.is_valid(&invalid_phenotype));

        let violations = manager.check_all(&invalid_phenotype);
        assert_eq!(violations.len(), 1);

        // Test repair
        let mut repairable_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };
        let repaired = manager.repair_all(&mut repairable_phenotype);
        assert!(repaired);
        assert!(manager.is_valid(&repairable_phenotype));
    }

    #[test]
    fn test_penalty_score() {
        let constraint = UniqueValuesConstraint;

        // Create violations with different severities
        let violations = vec![
            ConstraintViolation::with_severity("Test", "Violation 1", 1.0),
            ConstraintViolation::with_severity("Test", "Violation 2", 2.0),
        ];

        let score = constraint.penalty_score(&violations);
        assert_eq!(score, 3.0);

        // Test with no severity
        let violations = vec![
            ConstraintViolation::new("Test", "Violation 1"),
            ConstraintViolation::new("Test", "Violation 2"),
        ];

        let score = constraint.penalty_score(&violations);
        assert_eq!(score, 2.0);
    }

    #[test]
    fn test_constraint_manager_builder() {
        // Create a constraint manager using the builder pattern
        let manager = ConstraintManager::<TestPhenotype>::builder()
            .with_constraint(UniqueValuesConstraint)
            .build();

        // Valid phenotype
        let valid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 4, 5],
        };
        assert!(manager.is_valid(&valid_phenotype));

        // Invalid phenotype
        let invalid_phenotype = TestPhenotype {
            values: vec![1, 2, 3, 2, 5],
        };
        assert!(!manager.is_valid(&invalid_phenotype));

        // Test with multiple constraints
        let manager = ConstraintManager::<TestPhenotype>::builder()
            .with_constraint(UniqueValuesConstraint)
            .with_constraint(UniqueValuesConstraint) // Adding the same constraint twice for testing
            .build();

        assert_eq!(manager.len(), 2);
        assert!(!manager.is_empty());
    }

    #[test]
    fn test_constraint_manager_builder_default() {
        // Test that the default builder creates an empty manager
        let builder = ConstraintManagerBuilder::<TestPhenotype>::default();
        let manager = builder.build();

        assert_eq!(manager.len(), 0);
        assert!(manager.is_empty());
    }

    #[test]
    fn test_penalty_adjusted_challenge() {
        // Create a phenotype with a constraint violation
        let phenotype = PenaltyTestPhenotype {
            values: vec![1, 2, 3, 2, 5], // Duplicate value 2
        };

        // Create a constraint manager with a constraint
        let mut constraint_manager = ConstraintManager::new();
        constraint_manager.add_constraint(PenaltyUniqueValuesConstraint);

        // Create a challenge
        let challenge = PenaltyTestChallenge::default();

        // Create a penalty-adjusted challenge
        let penalty_challenge = PenaltyAdjustedChallenge::new(challenge, constraint_manager, 10.0);

        // Calculate expected score
        let base_score = phenotype.values.iter().sum::<usize>() as f64; // = 13.0
        let penalty = 10.0; // One violation with weight 10.0
        let expected_score = base_score - penalty; // = 3.0

        // Check that the score is adjusted by the penalty
        assert_eq!(penalty_challenge.score(&phenotype), expected_score);
    }

    #[test]
    fn test_penalty_adjusted_challenge_accessors() {
        // Create a constraint manager
        let constraint_manager = ConstraintManager::new();

        // Create a challenge
        let challenge = PenaltyTestChallenge::default();

        // Create a penalty-adjusted challenge
        let penalty_challenge = PenaltyAdjustedChallenge::new(challenge, constraint_manager, 5.0);

        // Check accessors
        assert_eq!(penalty_challenge.penalty_weight(), 5.0);
        assert_eq!(penalty_challenge.constraint_manager().len(), 0);
        assert_eq!(penalty_challenge.inner().target, 0);
    }

    #[test]
    fn test_penalty_adjusted_challenge_no_violations() {
        // Create a phenotype with no constraint violations
        let phenotype = PenaltyTestPhenotype {
            values: vec![1, 2, 3, 4, 5],
        };

        // Create a constraint manager with a constraint
        let mut constraint_manager = ConstraintManager::new();
        constraint_manager.add_constraint(PenaltyUniqueValuesConstraint);

        // Create a challenge
        let challenge = PenaltyTestChallenge::default();

        // Create a penalty-adjusted challenge
        let penalty_challenge = PenaltyAdjustedChallenge::new(challenge, constraint_manager, 10.0);

        // Calculate expected score (no penalty)
        let expected_score = phenotype.values.iter().sum::<usize>() as f64; // = 15.0

        // Check that the score is not adjusted
        assert_eq!(penalty_challenge.score(&phenotype), expected_score);
    }

    // Test phenotype for PenaltyAdjustedChallenge tests
    #[derive(Clone, Debug)]
    struct PenaltyTestPhenotype {
        values: Vec<usize>,
    }

    impl Phenotype for PenaltyTestPhenotype {
        fn crossover(&mut self, other: &Self) {
            if !other.values.is_empty() && !self.values.is_empty() {
                self.values[0] = other.values[0];
            }
        }

        fn mutate(&mut self, _rng: &mut RandomNumberGenerator) {
            if !self.values.is_empty() {
                self.values[0] += 1;
            }
        }
    }

    // Test constraint for PenaltyAdjustedChallenge tests
    #[derive(Debug)]
    struct PenaltyUniqueValuesConstraint;

    impl Constraint<PenaltyTestPhenotype> for PenaltyUniqueValuesConstraint {
        fn check(&self, phenotype: &PenaltyTestPhenotype) -> Vec<ConstraintViolation> {
            let mut seen = HashSet::new();
            let mut violations = Vec::new();

            for (i, &value) in phenotype.values.iter().enumerate() {
                if !seen.insert(value) {
                    violations.push(ConstraintViolation::new(
                        "UniqueValues",
                        format!("Duplicate value {} at position {}", value, i),
                    ));
                }
            }

            violations
        }
    }

    // Test challenge for PenaltyAdjustedChallenge tests
    #[derive(Debug, Clone)]
    struct PenaltyTestChallenge {
        target: usize,
    }

    impl Default for PenaltyTestChallenge {
        fn default() -> Self {
            Self { target: 0 }
        }
    }

    impl Challenge<PenaltyTestPhenotype> for PenaltyTestChallenge {
        fn score(&self, phenotype: &PenaltyTestPhenotype) -> f64 {
            phenotype.values.iter().sum::<usize>() as f64
        }
    }
}
