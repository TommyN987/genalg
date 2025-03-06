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
use crate::rng::{RandomNumberGenerator, ThreadLocalRng};

/// Ensures that all elements in a collection are unique.
///
/// This constraint is useful for problems where each element can only be used once,
/// such as assignment problems or permutation problems.
#[derive(Debug, Clone)]
pub struct UniqueElementsConstraint<T, F>
where
    T: Eq + Hash + Debug + Clone,
    F: Fn(&T) -> Vec<T>,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the elements to check for uniqueness
    extractor: F,
    /// Phantom data for the element type
    _marker: PhantomData<T>,
}

impl<T, F> UniqueElementsConstraint<T, F>
where
    T: Eq + Hash + Debug + Clone,
    F: Fn(&T) -> Vec<T>,
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

impl<P, T, F> Constraint<P> for UniqueElementsConstraint<T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone,
    F: Fn(&P) -> Vec<T> + Send + Sync,
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
        // This is a simple repair strategy that might not work for all problems
        // A more sophisticated repair would depend on the specific problem
        
        // Extract elements
        let elements = (self.extractor)(phenotype);
        
        // Find duplicates
        let mut seen = HashSet::new();
        let mut duplicates = Vec::new();
        
        for (idx, element) in elements.iter().enumerate() {
            if !seen.insert(element) {
                duplicates.push(idx);
            }
        }
        
        // If no duplicates, no repair needed
        if duplicates.is_empty() {
            return true;
        }
        
        // Try to repair by mutating the phenotype
        // This is a generic approach that might not work for all phenotypes
        phenotype.mutate(rng);
        
        // Check if repair was successful
        let new_elements = (self.extractor)(phenotype);
        let mut new_seen = HashSet::new();
        
        for element in new_elements {
            if !new_seen.insert(element) {
                return false;
            }
        }
        
        true
    }
}

/// Ensures that all required assignments are made.
///
/// This constraint is useful for assignment problems where each item must be assigned
/// to exactly one category or position.
#[derive(Debug, Clone)]
pub struct CompleteAssignmentConstraint<K, V, F>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone,
    F: Fn(&K) -> Vec<V>,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the assignments from the phenotype
    extractor: F,
    /// The set of keys that must be assigned
    required_keys: HashSet<K>,
    /// Phantom data for the value type
    _marker: PhantomData<V>,
}

impl<K, V, F> CompleteAssignmentConstraint<K, V, F>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone,
    F: Fn(&K) -> Vec<V>,
{
    /// Creates a new complete assignment constraint with the given name, extractor function,
    /// and set of required keys.
    ///
    /// The extractor function is used to extract the assignments from the phenotype.
    pub fn new<S: Into<String>>(name: S, extractor: F, required_keys: HashSet<K>) -> Self {
        Self {
            name: name.into(),
            extractor,
            required_keys,
            _marker: PhantomData,
        }
    }
}

impl<P, K, V, F> Constraint<P> for CompleteAssignmentConstraint<K, V, F>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone,
    F: Fn(&P) -> HashMap<K, V> + Send + Sync,
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
        // This is a simple repair strategy that might not work for all problems
        // A more sophisticated repair would depend on the specific problem
        
        // Extract assignments
        let assignments = (self.extractor)(phenotype);
        
        // Find missing assignments
        let mut missing = Vec::new();
        
        for key in &self.required_keys {
            if !assignments.contains_key(key) {
                missing.push(key.clone());
            }
        }
        
        // If no missing assignments, no repair needed
        if missing.is_empty() {
            return true;
        }
        
        // Try to repair by mutating the phenotype
        // This is a generic approach that might not work for all phenotypes
        phenotype.mutate(rng);
        
        // Check if repair was successful
        let new_assignments = (self.extractor)(phenotype);
        
        for key in &self.required_keys {
            if !new_assignments.contains_key(key) {
                return false;
            }
        }
        
        true
    }
}

/// Ensures that assignments satisfy capacity constraints.
///
/// This constraint is useful for bin packing or resource allocation problems
/// where each bin or resource has a limited capacity.
#[derive(Debug, Clone)]
pub struct CapacityConstraint<K, V, F, G>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone,
    F: Fn(&K) -> HashMap<V, usize>,
    G: Fn(&V) -> usize,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the assignments from the phenotype
    extractor: F,
    /// Function to get the capacity of each bin
    capacity_fn: G,
    /// Phantom data for the key and value types
    _marker: PhantomData<(K, V)>,
}

impl<K, V, F, G> CapacityConstraint<K, V, F, G>
where
    K: Eq + Hash + Debug + Clone,
    V: Debug + Clone,
    F: Fn(&K) -> HashMap<V, usize>,
    G: Fn(&V) -> usize,
{
    /// Creates a new capacity constraint with the given name, extractor function,
    /// and capacity function.
    ///
    /// The extractor function is used to extract the assignments from the phenotype.
    /// The capacity function is used to get the capacity of each bin.
    pub fn new<S: Into<String>>(name: S, extractor: F, capacity_fn: G) -> Self {
        Self {
            name: name.into(),
            extractor,
            capacity_fn,
            _marker: PhantomData,
        }
    }
}

impl<P, K, V, F, G> Constraint<P> for CapacityConstraint<K, V, F, G>
where
    P: Phenotype,
    K: Eq + Hash + Debug + Clone,
    V: Eq + Hash + Debug + Clone,
    F: Fn(&P) -> HashMap<V, Vec<K>> + Send + Sync,
    G: Fn(&V) -> usize + Send + Sync,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let assignments = (self.extractor)(phenotype);
        let mut violations = Vec::new();

        for (bin, items) in &assignments {
            let capacity = (self.capacity_fn)(bin);
            
            if items.len() > capacity {
                violations.push(ConstraintViolation::new(
                    &self.name,
                    format!(
                        "Bin {:?} exceeds capacity: {} items assigned, capacity is {}",
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
        // This is a simple repair strategy that might not work for all problems
        // A more sophisticated repair would depend on the specific problem
        
        // Extract assignments
        let assignments = (self.extractor)(phenotype);
        
        // Find capacity violations
        let mut violations = false;
        
        for (bin, items) in &assignments {
            let capacity = (self.capacity_fn)(bin);
            
            if items.len() > capacity {
                violations = true;
                break;
            }
        }
        
        // If no violations, no repair needed
        if !violations {
            return true;
        }
        
        // Try to repair by mutating the phenotype
        // This is a generic approach that might not work for all phenotypes
        phenotype.mutate(rng);
        
        // Check if repair was successful
        let new_assignments = (self.extractor)(phenotype);
        
        for (bin, items) in &new_assignments {
            let capacity = (self.capacity_fn)(bin);
            
            if items.len() > capacity {
                return false;
            }
        }
        
        true
    }
}

/// Ensures that a solution satisfies dependency constraints.
///
/// This constraint is useful for scheduling or sequencing problems where some
/// tasks must be completed before others.
#[derive(Debug, Clone)]
pub struct DependencyConstraint<T, F>
where
    T: Eq + Hash + Debug + Clone,
    F: Fn(&T) -> Vec<(T, T)>,
{
    /// Name of the constraint for error messages
    name: String,
    /// Function to extract the sequence from the phenotype
    extractor: F,
    /// The set of dependencies (before, after) pairs
    dependencies: Vec<(T, T)>,
}

impl<T, F> DependencyConstraint<T, F>
where
    T: Eq + Hash + Debug + Clone,
    F: Fn(&T) -> Vec<(T, T)>,
{
    /// Creates a new dependency constraint with the given name, extractor function,
    /// and set of dependencies.
    ///
    /// The extractor function is used to extract the sequence from the phenotype.
    /// Each dependency is a pair (before, after) indicating that the 'before' element
    /// must appear before the 'after' element in the sequence.
    pub fn new<S: Into<String>>(name: S, extractor: F, dependencies: Vec<(T, T)>) -> Self {
        Self {
            name: name.into(),
            extractor,
            dependencies,
        }
    }
}

impl<P, T, F> Constraint<P> for DependencyConstraint<T, F>
where
    P: Phenotype,
    T: Eq + Hash + Debug + Clone,
    F: Fn(&P) -> Vec<T> + Send + Sync,
{
    fn check(&self, phenotype: &P) -> Vec<ConstraintViolation> {
        let sequence = (self.extractor)(phenotype);
        let mut violations = Vec::new();

        // Create a map of element to position
        let mut positions = HashMap::new();
        for (idx, element) in sequence.iter().enumerate() {
            positions.insert(element, idx);
        }

        // Check each dependency
        for (before, after) in &self.dependencies {
            if let (Some(&before_pos), Some(&after_pos)) = (positions.get(before), positions.get(after)) {
                if before_pos >= after_pos {
                    violations.push(ConstraintViolation::new(
                        &self.name,
                        format!(
                            "Dependency violation: {:?} must appear before {:?}, but found at positions {} and {}",
                            before,
                            after,
                            before_pos,
                            after_pos
                        ),
                    ));
                }
            } else {
                // If either element is missing, that's a different kind of violation
                if !positions.contains_key(before) {
                    violations.push(ConstraintViolation::new(
                        &self.name,
                        format!("Missing element {:?} in sequence", before),
                    ));
                }
                if !positions.contains_key(after) {
                    violations.push(ConstraintViolation::new(
                        &self.name,
                        format!("Missing element {:?} in sequence", after),
                    ));
                }
            }
        }

        violations
    }

    fn repair_with_rng(&self, phenotype: &mut P, rng: &mut RandomNumberGenerator) -> bool {
        // This is a simple repair strategy that might not work for all problems
        // A more sophisticated repair would depend on the specific problem
        
        // Extract sequence
        let sequence = (self.extractor)(phenotype);
        
        // Create a map of element to position
        let mut positions = HashMap::new();
        for (idx, element) in sequence.iter().enumerate() {
            positions.insert(element, idx);
        }
        
        // Check for violations
        let mut violations = false;
        
        for (before, after) in &self.dependencies {
            if let (Some(&before_pos), Some(&after_pos)) = (positions.get(before), positions.get(after)) {
                if before_pos >= after_pos {
                    violations = true;
                    break;
                }
            } else {
                // If either element is missing, that's a violation
                violations = true;
                break;
            }
        }
        
        // If no violations, no repair needed
        if !violations {
            return true;
        }
        
        // Try to repair by mutating the phenotype
        // This is a generic approach that might not work for all phenotypes
        phenotype.mutate(rng);
        
        // Check if repair was successful
        let new_sequence = (self.extractor)(phenotype);
        
        // Create a map of element to position
        let mut new_positions = HashMap::new();
        for (idx, element) in new_sequence.iter().enumerate() {
            new_positions.insert(element, idx);
        }
        
        // Check for violations
        for (before, after) in &self.dependencies {
            if let (Some(&before_pos), Some(&after_pos)) = (new_positions.get(before), new_positions.get(after)) {
                if before_pos >= after_pos {
                    return false;
                }
            } else {
                // If either element is missing, that's a violation
                return false;
            }
        }
        
        true
    }
} 