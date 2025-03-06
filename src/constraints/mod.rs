//! # Constraints Module
//!
//! This module provides a framework for defining and enforcing constraints in genetic algorithms.
//! Constraints are particularly useful for combinatorial optimization problems where solutions
//! must satisfy specific requirements to be valid.
//!
//! ## Key Components
//!
//! - `Constraint` trait: Defines the interface for constraints that can be applied to phenotypes
//! - `ConstraintViolation`: Represents a specific violation of a constraint
//! - `ConstraintManager`: Manages multiple constraints and evaluates them against phenotypes
//!
//! ## Example
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

use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::sync::Arc;

use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

/// Represents a violation of a constraint.
#[derive(Debug, Clone)]
pub struct ConstraintViolation {
    /// The name of the constraint that was violated
    pub constraint_name: String,
    /// A description of the violation
    pub description: String,
    /// An optional severity score (higher means more severe)
    pub severity: Option<f64>,
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
}

impl Display for ConstraintViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Constraint '{}' violated: {}{}",
            self.constraint_name,
            self.description,
            self.severity
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
        violations
            .iter()
            .map(|v| v.severity.unwrap_or(1.0))
            .sum()
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
        self.constraints
            .iter()
            .map(|c| {
                let violations = c.check(phenotype);
                c.penalty_score(&violations)
            })
            .sum()
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