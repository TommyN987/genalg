//! # Combinatorial Constraints
//!
//! This module provides common constraints for combinatorial optimization problems.
//! These constraints are particularly useful for problems involving selection, assignment,
//! or sequencing of discrete elements.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;

use crate::constraints::{Constraint, ConstraintViolation};
use crate::phenotype::Phenotype;
use crate::rng::RandomNumberGenerator;

/// Ensures that all elements in a collection are unique.
///
/// This constraint is useful for problems where each element can only be used once,
/// such as assignment problems or permutation problems.
#[derive(Debug, Clone)]
pub struct UniqueElementsConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the elements to check for uniqueness
    extractor: F,
    /// Phantom data for the element type
    _marker: PhantomData<(P, T)>,
}

impl<P, T, F> UniqueElementsConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    /// Creates a new unique elements constraint with the given name and extractor function.
    ///
    /// The extractor function is used to extract the elements to check for uniqueness
    /// from the phenotype.
    pub fn new<S: Into<String>>(name: S, extractor: F) -> Self {
        Self {
            name: name.into(),
            extractor,
            _marker: PhantomData,
        }
    }
}

impl<P, T, F> Constraint<P> for UniqueElementsConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let elements = (self.extractor)(phenotype);
        let mut seen = HashSet::new();
        let mut violations = Vec::new();

        for (idx, element) in elements.iter().enumerate() {
            if !seen.insert(element) {
                violations.push(ConstraintViolation::new(
                    &self.name,
                    format!("Duplicate element {:?} at position {}", element, idx),
                ));
            }
        }

        violations
    }

    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // This is a generic implementation that might not work for all phenotypes
        // It relies on the phenotype's mutate method to potentially fix the uniqueness issue

        // Check if there are any violations
        let violations = self.check(phenotype);
        if violations.is_empty() {
            return false; // No violations to repair
        }

        // Try to repair by mutating the phenotype
        phenotype.mutate(rng);

        // Check if repair was successful
        let new_violations = self.check(phenotype);
        new_violations.is_empty()
    }
}

/// Ensures that all required keys are assigned a value.
///
/// This constraint is useful for assignment problems where each key must be assigned
/// a value from a set of possible values.
#[derive(Debug, Clone)]
pub struct CompleteAssignmentConstraint<P, K, V, F>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<K, V> + Send + Sync + Debug,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the assignments from the phenotype
    extractor: F,
    /// The set of keys that must be assigned
    required_keys: HashSet<K>,
    /// Phantom data for the value type
    _marker: PhantomData<P>,
}

impl<P, K, V, F> CompleteAssignmentConstraint<P, K, V, F>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<K, V> + Send + Sync + Debug,
{
    /// Creates a new complete assignment constraint with the given name, extractor function,
    /// and set of required keys.
    pub fn new<S: Into<String>>(name: S, extractor: F, required_keys: HashSet<K>) -> Self {
        Self {
            name: name.into(),
            extractor,
            required_keys,
            _marker: PhantomData,
        }
    }
}

impl<P, K, V, F> Constraint<P> for CompleteAssignmentConstraint<P, K, V, F>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<K, V> + Send + Sync + Debug,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let assignments = (self.extractor)(phenotype);
        let mut violations = Vec::new();

        for key in &self.required_keys {
            if !assignments.contains_key(key) {
                violations.push(ConstraintViolation::new(
                    &self.name,
                    format!("Missing assignment for key {:?}", key),
                ));
            }
        }

        violations
    }

    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // This is a generic implementation that might not work for all phenotypes
        // It relies on the phenotype's mutate method to potentially fix the assignment issue

        // Check if there are any violations
        let violations = self.check(phenotype);
        if violations.is_empty() {
            return false; // No violations to repair
        }

        // Try to repair by mutating the phenotype
        phenotype.mutate(rng);

        // Check if repair was successful
        let new_violations = self.check(phenotype);
        new_violations.is_empty()
    }
}

/// Ensures that assignments satisfy capacity constraints.
///
/// This constraint is useful for bin packing or resource allocation problems
/// where each bin or resource has a limited capacity.
#[derive(Debug, Clone)]
pub struct CapacityConstraint<P, K, V, F, G>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<V, Vec<K>> + Send + Sync + Debug,
    G: Fn(&V) -> usize + Send + Sync + Debug,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the assignments from the phenotype
    extractor: F,
    /// Function to get the capacity of each bin
    capacity_fn: G,
    /// Phantom data for the key and value types
    _marker: PhantomData<P>,
}

impl<P, K, V, F, G> CapacityConstraint<P, K, V, F, G>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<V, Vec<K>> + Send + Sync + Debug,
    G: Fn(&V) -> usize + Send + Sync + Debug,
{
    /// Creates a new capacity constraint with the given name, extractor function,
    /// and capacity function.
    pub fn new<S: Into<String>>(name: S, extractor: F, capacity_fn: G) -> Self {
        Self {
            name: name.into(),
            extractor,
            capacity_fn,
            _marker: PhantomData,
        }
    }
}

impl<P, K, V, F, G> Constraint<P> for CapacityConstraint<P, K, V, F, G>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone + Send + Sync,
    V: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> HashMap<V, Vec<K>> + Send + Sync + Debug,
    G: Fn(&V) -> usize + Send + Sync + Debug,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let assignments = (self.extractor)(phenotype);
        let mut violations = Vec::new();

        for (bin, items) in assignments.iter() {
            let capacity = (self.capacity_fn)(bin);
            if items.len() > capacity {
                violations.push(ConstraintViolation::new(
                    &self.name,
                    format!(
                        "Bin {:?} has {} items but capacity is {}",
                        bin,
                        items.len(),
                        capacity
                    ),
                ));
            }
        }

        violations
    }

    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // This is a generic implementation that might not work for all phenotypes
        // It relies on the phenotype's mutate method to potentially fix the capacity issue

        // Check if there are any violations
        let violations = self.check(phenotype);
        if violations.is_empty() {
            return false; // No violations to repair
        }

        // Try to repair by mutating the phenotype
        phenotype.mutate(rng);

        // Check if repair was successful
        let new_violations = self.check(phenotype);
        new_violations.is_empty()
    }
}

/// Ensures that dependencies between elements are satisfied.
///
/// This constraint is useful for problems where some elements must come before
/// others, such as scheduling or sequencing problems.
#[derive(Debug, Clone)]
pub struct DependencyConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the sequence from the phenotype
    extractor: F,
    /// The set of dependencies (before, after) pairs
    dependencies: Vec<(T, T)>,
    /// Phantom data for the phenotype type
    _marker: PhantomData<P>,
}

impl<P, T, F> DependencyConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    /// Creates a new dependency constraint with the given name, extractor function,
    /// and dependencies.
    pub fn new<S: Into<String>>(name: S, extractor: F, dependencies: Vec<(T, T)>) -> Self {
        Self {
            name: name.into(),
            extractor,
            dependencies,
            _marker: PhantomData,
        }
    }
}

impl<P, T, F> Constraint<P> for DependencyConstraint<P, T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone + Send + Sync,
    F: Fn(&P) -> Vec<T> + Send + Sync + Debug,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let sequence = (self.extractor)(phenotype);
        let mut violations = Vec::new();

        // Build a map of element to position
        let mut positions = HashMap::new();
        for (pos, element) in sequence.iter().enumerate() {
            positions.insert(element, pos);
        }

        // Check each dependency
        for (before, after) in &self.dependencies {
            if let (Some(&before_pos), Some(&after_pos)) =
                (positions.get(before), positions.get(after))
            {
                if before_pos >= after_pos {
                    violations.push(ConstraintViolation::new(
                        &self.name,
                        format!(
                            "Dependency violation: {:?} must come before {:?}",
                            before, after
                        ),
                    ));
                }
            }
        }

        violations
    }

    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // This is a generic implementation that might not work for all phenotypes
        // It relies on the phenotype's mutate method to potentially fix the dependency issue

        // Check if there are any violations
        let violations = self.check(phenotype);
        if violations.is_empty() {
            return false; // No violations to repair
        }

        // Try to repair by mutating the phenotype
        phenotype.mutate(rng);

        // Check if repair was successful
        let new_violations = self.check(phenotype);
        new_violations.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test the constraint violation struct
    #[test]
    fn test_constraint_violation() {
        let violation = ConstraintViolation::new("TestConstraint", "Test violation");
        assert_eq!(violation.constraint_name, "TestConstraint");
        assert_eq!(violation.description, "Test violation");
        assert!(violation.severity.is_none());

        let violation_with_severity =
            ConstraintViolation::with_severity("TestConstraint", "Test violation", 2.5);
        assert_eq!(violation_with_severity.constraint_name, "TestConstraint");
        assert_eq!(violation_with_severity.description, "Test violation");
        assert_eq!(violation_with_severity.severity, Some(2.5));
    }

    // Test the constraint module documentation examples
    #[test]
    fn test_constraint_documentation_examples() {
        // This test verifies that the examples in the module documentation compile
        // and work as expected. It doesn't directly test the constraints themselves,
        // but ensures that the API is usable as documented.

        // Example of creating a constraint violation
        let violation = ConstraintViolation::new("UniqueValues", "Duplicate value 2 at position 3");
        assert_eq!(violation.constraint_name, "UniqueValues");
        assert_eq!(violation.description, "Duplicate value 2 at position 3");

        // Example of creating a constraint violation with severity
        let violation = ConstraintViolation::with_severity(
            "CapacityConstraint",
            "Bin 1 exceeds capacity by 3 items",
            3.0,
        );
        assert_eq!(violation.severity, Some(3.0));

        // Example of formatting a constraint violation
        let violation = ConstraintViolation::new("TestConstraint", "Test violation");
        let formatted = format!("{}", violation);
        assert!(formatted.contains("TestConstraint"));
        assert!(formatted.contains("Test violation"));
    }
}
