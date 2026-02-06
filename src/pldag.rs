use crate::error::{PldagError, Result};
use crate::storage::{InMemoryStore, NodeStore, NodeStoreTrait};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Represents a bound with minimum and maximum values.
/// Used to specify the allowed range for variables and constraints.
pub type Bound = (i32, i32);

/// Type alias for node identifiers in the DAG.
pub type ID = String;

/// Creates a hash from a vector of string-integer pairs and a standalone integer.
///
/// This function is used to generate unique IDs for constraints by hashing their
/// coefficients and bias together.
///
/// # Arguments
/// * `data` - Vector of (variable_name, coefficient) pairs
/// * `num` - Additional integer value to include in the hash (typically the bias)
///
/// # Returns
/// A 64-bit hash value representing the input data
fn create_hash(data: &Vec<(String, i32)>, num: i32) -> u64 {
    // Create a new hasher
    let mut hasher = DefaultHasher::new();

    // Hash the vector
    for (s, i) in data {
        s.hash(&mut hasher);
        i.hash(&mut hasher);
    }

    // Hash the standalone i32 value
    num.hash(&mut hasher);

    // Return the final hash value
    hasher.finish()
}

/// Adds two bounds together element-wise.
///
/// # Arguments
/// * `b1` - First bound (min1, max1)
/// * `b2` - Second bound (min2, max2)
///
/// # Returns
/// A new bound (min1 + min2, max1 + max2)
fn bound_add(b1: Bound, b2: Bound) -> Bound {
    (b1.0 + b2.0, b1.1 + b2.1)
}

/// Multiplies a bound by a scalar coefficient.
///
/// When the coefficient is negative, the min and max values are swapped
/// to maintain the correct bound ordering.
///
/// # Arguments
/// * `k` - Scalar coefficient to multiply by
/// * `b` - Bound to multiply (min, max)
///
/// # Returns
/// A new bound with the multiplication applied
fn bound_multiply(k: i32, b: Bound) -> Bound {
    if k < 0 {
        (k * b.1, k * b.0)
    } else {
        (k * b.0, k * b.1)
    }
}

/// Integer floor division that is correct for negative numbers.
fn div_floor(a: i32, b: i32) -> i32 {
    assert!(b != 0);
    let (q, r) = (a / b, a % b);
    if r != 0 && (r > 0) != (b > 0) {
        q - 1
    } else {
        q
    }
}

/// Integer ceil division that is correct for negative numbers.
fn div_ceil(a: i32, b: i32) -> i32 {
    assert!(b != 0);
    let (q, r) = (a / b, a % b);
    if r != 0 && (r > 0) == (b > 0) {
        q + 1
    } else {
        q
    }
}

/// Intersect two bounds (component-wise).
fn intersect_bounds(a: Bound, b: Bound) -> Bound {
    (a.0.max(b.0), a.1.min(b.1))
}

/// Tighten variable bounds assuming a constraint of the form:
///
///   sum(a_i * x_i) + bias >= 0
///
/// is **TRUE**.
///
/// This uses the ">= b" bound tightening logic on:
///
///   sum(a_i * x_i) >= -bias_min
///
/// where bias_min = bias.0.
///
/// Returns true if any bound was changed.
fn tighten_constraint_true(constraint: &Constraint, values: &mut IndexMap<String, Bound>) -> bool {
    let mut changed = false;

    // Transform: sum(a_i * x_i) + bias >= 0  ->  sum(a_i * x_i) >= -bias_min
    let b = -constraint.bias.0;
    let coeffs = &constraint.coefficients;

    for (var_k, a_k) in coeffs.iter() {
        if *a_k == 0 {
            continue;
        }

        // Get current bound for x_k
        let (mut l_k, mut u_k) = values
            .get(var_k)
            .cloned()
            .unwrap_or((i32::MIN / 2, i32::MAX / 2));

        // Compute best help from other variables
        let mut big_b = 0i32;
        for (var_i, a_i) in coeffs.iter() {
            if var_i == var_k {
                continue;
            }
            let (l_i, u_i) = values
                .get(var_i)
                .cloned()
                .unwrap_or((i32::MIN / 2, i32::MAX / 2));

            if *a_i > 0 {
                big_b += a_i * u_i;
            } else if *a_i < 0 {
                big_b += a_i * l_i;
            }
        }

        // a_k * x_k + B >= b   ->   solve for x_k
        if *a_k > 0 {
            let num = b - big_b;
            let new_l = div_ceil(num, *a_k);
            if new_l > l_k {
                l_k = new_l;
                changed = true;
            }
        } else {
            // a_k < 0
            let num = b - big_b;
            let new_u = div_floor(num, *a_k);
            if new_u < u_k {
                u_k = new_u;
                changed = true;
            }
        }

        // Write back updated bound for x_k
        values.insert(var_k.clone(), (l_k, u_k));
    }

    changed
}

/// Sparse representation of an integer matrix.
///
/// Stores only non-zero elements using coordinate format (COO):
/// - `rows\[i\]`, `cols\[i\]`, `vals\[i\]` represent a non-zero element at position (rows\[i\], cols\[i\]) with value vals\[i\]
#[derive(Hash, Clone)]
pub struct SparseIntegerMatrix {
    /// Row indices of non-zero elements
    pub rows: Vec<usize>,
    /// Column indices of non-zero elements  
    pub cols: Vec<usize>,
    /// Values of non-zero elements
    pub vals: Vec<i32>,
    /// Matrix dimensions: (number_of_rows, number_of_columns)
    pub shape: (usize, usize),
}

impl fmt::Display for SparseIntegerMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut dense_matrix = DenseIntegerMatrix::new(self.shape.0, self.shape.1);
        for ((&row, &col), &val) in self.rows.iter().zip(&self.cols).zip(&self.vals) {
            dense_matrix.data[row][col] = val;
        }

        dense_matrix.fmt(f)
    }
}

/// Dense representation of an integer matrix.
///
/// Stores all elements in a 2D vector structure.
#[derive(Clone)]
pub struct DenseIntegerMatrix {
    /// Matrix data stored as a vector of rows
    pub data: Vec<Vec<i32>>,
    /// Matrix dimensions: (number_of_rows, number_of_columns)
    pub shape: (usize, usize),
}

impl fmt::Display for DenseIntegerMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.data {
            for val in row {
                write!(f, "{:>3} ", val)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl DenseIntegerMatrix {
    /// Creates a new dense integer matrix filled with zeros.
    ///
    /// # Arguments
    /// * `rows` - Number of rows in the matrix
    /// * `cols` - Number of columns in the matrix
    ///
    /// # Returns
    /// A new `DenseIntegerMatrix` with all elements initialized to zero
    pub fn new(rows: usize, cols: usize) -> DenseIntegerMatrix {
        DenseIntegerMatrix {
            data: vec![vec![0; cols]; rows],
            shape: (rows, cols),
        }
    }

    /// Computes the matrix-vector dot product.
    ///
    /// Multiplies this matrix by a vector and returns the resulting vector.
    /// The input vector length must match the number of columns in the matrix.
    ///
    /// # Arguments
    /// * `vector` - Input vector to multiply with
    ///
    /// # Returns
    /// A vector representing the matrix-vector product
    ///
    /// # Panics
    /// May panic if the vector length doesn't match the matrix column count
    pub fn dot_product(&self, vector: &[i32]) -> Vec<i32> {
        self.data
            .iter()
            .map(|row| {
                row.iter()
                    .zip(vector.iter())
                    .map(|(a, b)| a * b)
                    .sum()
            })
            .collect()
    }
}

/// Dense representation of a polyhedron defined by linear constraints.
///
/// Represents the constraint system Ax >= b where:
/// - A is the constraint matrix
/// - b is the right-hand side vector
/// - columns maps matrix columns to variable names
#[derive(Clone)]
pub struct DensePolyhedron {
    /// Constraint matrix A
    pub a: DenseIntegerMatrix,
    /// Right-hand side vector b
    pub b: Vec<i32>,
    /// Variable names corresponding to matrix columns
    pub columns: Vec<String>,
    /// Column bounds
    pub column_bounds: Vec<Bound>,
}

impl fmt::Display for DensePolyhedron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let space = 4;
        for col_name in &self.columns {
            write!(
                f,
                "{:>space$} ",
                &col_name.chars().take(3).collect::<String>()
            )?;
        }
        writeln!(f)?;
        for (ir, row) in self.a.data.iter().enumerate() {
            for val in row {
                write!(f, "{:>space$} ", val)?;
            }
            write!(f, ">= {:>space$}", self.b[ir])?;
            writeln!(f)?;
        }
        Ok(())
    }
}

impl DensePolyhedron {
    /// Converts a variable assignment map to a vector ordered by the polyhedron's columns.
    ///
    /// # Arguments
    /// * `from_assignments` - Map of variable names to their assigned values
    ///
    /// # Returns
    /// A vector where each position corresponds to a column in the polyhedron,
    /// with values from the assignment map or 0 if not assigned
    pub fn to_vector(&self, from_assignments: &HashMap<String, i32>) -> Vec<i32> {
        let mut vector: Vec<i32> = vec![0; self.columns.len()];
        for (index, v) in from_assignments.iter().filter_map(|(k, v)| {
            self.columns
                .iter()
                .position(|col| col == k)
                .map(|index| (index, v))
        }) {
            vector[index] = *v;
        }
        vector
    }

    /// Creates a new polyhedron by fixing certain variables to specific values.
    ///
    /// This operation eliminates the specified variables from the polyhedron by
    /// substituting their fixed values into the constraints and removing their columns.
    ///
    /// # Arguments
    /// * `values` - Map of variable names to their fixed values
    ///
    /// # Returns
    /// A new `DensePolyhedron` with the specified variables eliminated
    pub fn assume(&self, values: &HashMap<String, i32>) -> DensePolyhedron {
        // 1) Make mutable copies of everything
        let mut new_a_data = self.a.data.clone(); // Vec<Vec<i32>>
        let mut new_b = self.b.clone(); // Vec<i32>
        let mut new_columns = self.columns.clone(); // Vec<String>

        // 2) Find which columns we’re going to remove, along with their assigned values.
        //    We capture (idx_in_matrix, column_name, assigned_value).
        let mut to_remove: Vec<(usize, String, i32)> = values
            .iter()
            .filter_map(|(name, &val)| {
                // look up current index of `name` in self.columns
                self.columns
                    .iter()
                    .position(|col| col == name)
                    .map(|idx| (idx, name.clone(), val))
            })
            .collect();

        // 3) Remove from highest index to lowest so earlier removals
        //    don’t shift the positions of later ones.
        to_remove.sort_by(|a, b| b.0.cmp(&a.0));

        // 4) For each (idx, name, val) do:
        //      - b := b - A[:, idx] * val
        //      - remove column idx from every row of A
        //      - remove from columns and integer_columns
        for (col_idx, _, fixed_val) in to_remove {
            for row in 0..new_a_data.len() {
                // subtract A[row][col_idx] * fixed_val from b[row]
                new_b[row] -= new_a_data[row][col_idx] * fixed_val;
                // now remove that column entry
                new_a_data[row].remove(col_idx);
            }
            // drop the column name
            new_columns.remove(col_idx);
        }

        // 5) Rebuild the DenseIntegerMatrix with updated shape
        let new_shape = (new_a_data.len(), new_columns.len());
        let new_a = DenseIntegerMatrix {
            data: new_a_data,
            shape: new_shape,
        };

        // 6) Return the shrunken polyhedron
        DensePolyhedron {
            a: new_a,
            b: new_b,
            columns: new_columns,
            column_bounds: self
                .column_bounds
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    if !values.contains_key(&self.columns[i]) {
                        Some(*b)
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Evaluates the polyhedron constraints against variable bounds.
    ///
    /// Tests whether the lower and upper bounds of the given variables
    /// satisfy all constraints in the polyhedron.
    ///
    /// # Arguments
    /// * `assignments` - Map of variable names to their bounds (min, max)
    ///
    /// # Returns
    /// A bound where:
    /// - .0 is 1 if lower bounds satisfy all constraints, 0 otherwise
    /// - .1 is 1 if upper bounds satisfy all constraints, 0 otherwise
    pub fn evaluate(&self, assignments: &IndexMap<String, Bound>) -> Bound {
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        for (key, bound) in assignments {
            lower_bounds.insert(key.clone(), bound.0);
            upper_bounds.insert(key.clone(), bound.1);
        }

        let lower_result = self
            .a
            .dot_product(&self.to_vector(&lower_bounds))
            .iter()
            .zip(&self.b)
            .all(|(a, b)| a >= b);

        let upper_result = self
            .a
            .dot_product(&self.to_vector(&upper_bounds))
            .iter()
            .zip(&self.b)
            .all(|(a, b)| a >= b);

        (lower_result as i32, upper_result as i32)
    }
}

impl From<SparseIntegerMatrix> for DenseIntegerMatrix {
    fn from(sparse: SparseIntegerMatrix) -> DenseIntegerMatrix {
        let mut dense = DenseIntegerMatrix::new(sparse.shape.0, sparse.shape.1);
        for ((&row, &col), &val) in sparse.rows.iter().zip(&sparse.cols).zip(&sparse.vals) {
            dense.data[row][col] = val;
        }
        dense
    }
}

impl From<DenseIntegerMatrix> for SparseIntegerMatrix {
    fn from(dense: DenseIntegerMatrix) -> SparseIntegerMatrix {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in dense.data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(val);
                }
            }
        }
        SparseIntegerMatrix {
            rows,
            cols,
            vals,
            shape: dense.shape,
        }
    }
}

impl SparseIntegerMatrix {
    /// Creates a new empty sparse integer matrix.
    ///
    /// # Returns
    /// A new `SparseIntegerMatrix` with no entries and shape (0, 0)
    pub fn new() -> SparseIntegerMatrix {
        SparseIntegerMatrix {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            shape: (0, 0),
        }
    }
}

/// Sparse representation of a polyhedron defined by linear constraints.
///
/// Represents the constraint system Ax >= b in sparse format for memory efficiency.
#[derive(Hash, Clone)]
pub struct SparsePolyhedron {
    /// Sparse constraint matrix A
    pub a: SparseIntegerMatrix,
    /// Right-hand side vector b
    pub b: Vec<i32>,
    /// Variable names corresponding to matrix columns
    pub columns: Vec<String>,
    /// Column bounds
    pub column_bounds: Vec<Bound>,
}

impl From<SparsePolyhedron> for DensePolyhedron {
    fn from(sparse: SparsePolyhedron) -> DensePolyhedron {
        let mut dense_matrix = DenseIntegerMatrix::new(sparse.a.shape.0, sparse.a.shape.1);
        for ((&row, &col), &val) in sparse.a.rows.iter().zip(&sparse.a.cols).zip(&sparse.a.vals) {
            dense_matrix.data[row][col] = val;
        }
        DensePolyhedron {
            a: dense_matrix,
            b: sparse.b,
            columns: sparse.columns,
            column_bounds: sparse.column_bounds,
        }
    }
}

impl From<DensePolyhedron> for SparsePolyhedron {
    fn from(dense: DensePolyhedron) -> SparsePolyhedron {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in dense.a.data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(val);
                }
            }
        }
        SparsePolyhedron {
            a: SparseIntegerMatrix {
                rows,
                cols,
                vals,
                shape: dense.a.shape,
            },
            b: dense.b,
            columns: dense.columns,
            column_bounds: dense.column_bounds,
        }
    }
}

/// Represents a coefficient in a linear constraint: (variable_name, coefficient_value).
pub type Coefficient = (String, i32);

/// Maps node IDs to their bound values.
pub type Assignment = IndexMap<ID, Bound>;

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Hash)]
/// Represents a linear constraint in the form: sum(coeff_i * var_i) + bias >= 0.
pub struct Constraint {
    /// Vector of (variable_name, coefficient) pairs
    pub coefficients: Vec<Coefficient>,
    /// Bias/constant term with potential bounds
    pub bias: Bound,
}

impl Constraint {
    /// Computes the dot product of the constraint coefficients with variable bounds.
    ///
    /// This calculates the range of possible values for the linear combination
    /// of variables in this constraint, excluding the bias term.
    ///
    /// # Arguments
    /// * `values` - Map of variable names to their bounds
    ///
    /// # Returns
    /// A bound representing the min and max possible values of the dot product
    pub fn dot(&self, values: &IndexMap<String, Bound>) -> Bound {
        self.coefficients.iter().fold((0, 0), |acc, (key, coeff)| {
            let bound = values.get(key).unwrap_or(&(0, 0));
            let (min, max) = bound_multiply(*coeff, *bound);
            (acc.0 + min, acc.1 + max)
        })
    }

    /// Evaluates the constraint against variable bounds.
    ///
    /// Computes whether the constraint (dot product + bias >= 0) is satisfied
    /// for the given variable bounds.
    ///
    /// # Arguments
    /// * `values` - Map of variable names to their bounds
    ///
    /// # Returns
    /// A bound where:
    /// - .0 is 1 if the constraint is satisfied with lower bounds, 0 otherwise
    /// - .1 is 1 if the constraint is satisfied with upper bounds, 0 otherwise
    pub fn evaluate(&self, values: &IndexMap<String, Bound>) -> Bound {
        let bound = self.dot(values);
        return (
            (bound.0 + self.bias.0 >= 0) as i32,
            (bound.1 + self.bias.1 >= 0) as i32,
        );
    }

    /// Creates the negation of this constraint.
    ///
    /// Transforms the constraint from (ax + b >= 0) to (-ax - b - 1 >= 0),
    /// which is equivalent to (ax + b < 0).
    ///
    /// # Returns
    /// A new `Constraint` representing the negation of this constraint
    pub fn negate(&self) -> Constraint {
        Constraint {
            coefficients: self
                .coefficients
                .iter()
                .map(|(key, val)| (key.clone(), -val))
                .collect(),
            bias: (-self.bias.0 - 1, -self.bias.1 - 1),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Hash)]
/// Represents different types of boolean expressions in the DAG.
pub enum Node {
    /// A composite node representing a linear constraint
    Composite(Constraint),
    /// A primitive (leaf) node with a bound on its value
    Primitive(Bound),
}

/// A Primitive Logic Directed Acyclic Graph (PL-DAG).
///
/// The PL-DAG represents a logical system where:
/// - Primitive nodes are leaf variables with bounds
/// - Composite nodes represent logical constraints over other nodes
/// - Each node has an associated coefficient for accumulation operations
///
/// The DAG structure ensures no cycles and enables efficient bottom-up propagation.
pub struct Pldag {
    /// Store for mapping node IDs to their corresponding nodes, supporting multiple access patterns
    pub storage: Box<dyn NodeStoreTrait>,
}

impl Pldag {
    /// Creates a new empty PL-DAG.
    ///
    /// # Returns
    /// A new `Pldag` instance with no nodes
    pub fn new() -> Pldag {
        Pldag {
            storage: Box::new(NodeStore::new(Arc::new(InMemoryStore::new()))),
        }
    }

    pub fn new_custom(storage: Box<dyn NodeStoreTrait>) -> Pldag {
        Pldag { storage }
    }

    /// Estimates an upper bound on the count of feasible assignments in the DAG.
    ///
    /// # Arguments
    /// * `dag` - mapping from node name to Node (Primitive / Composite)
    /// * `assumptions` - mapping from node name to assumed bound
    ///
    /// # Returns
    /// An estimated upper bound on the count of feasible assignments
    pub fn estimate_upper_bound_count(
        dag: &HashMap<String, Node>,
        assumptions: &HashMap<String, Bound>,
    ) -> Result<usize> {
        let tightened = Pldag::tighten(dag, assumptions)?;

        let mut count = 1usize;
        for (_name, node) in dag.iter() {
            match node {
                Node::Primitive(_) => {
                    // Primitive node: multiply by the size of its bound
                    let bound = tightened.get(_name).unwrap_or(&(0, 0));
                    let size = (bound.1 - bound.0 + 1) as usize;
                    count *= size;
                }
                // Composite nodes do not contribute to count, only restricts from the primitives
                Node::Composite(_) => continue,
            }
        }

        Ok(count)
    }

    /// Full tightening over the DAG given initial assumptions.
    ///
    /// - `dag`: mapping from node name to Node (Primitive / Composite)
    /// - `assumptions`: mapping from node name to assumed bound,
    ///    e.g. "A" -> (1,1) means boolean node A is TRUE.
    ///
    /// Returns an IndexMap of final bounds for all nodes (primitives + composite booleans).
    pub fn tighten(
        dag: &HashMap<String, Node>,
        assumptions: &HashMap<String, Bound>,
    ) -> Result<IndexMap<String, Bound>> {
        // 1. Initialize bounds for all nodes.
        //
        // - Primitive: use its bound.
        // - Composite: treat as boolean in [0,1] if not in assumptions.
        let mut values: IndexMap<String, Bound> = IndexMap::new();
        for (name, node) in dag.iter() {
            let initial = match node {
                Node::Primitive(b) => *b,
                Node::Composite(_) => (0, 1), // boolean: unknown in [0,1]
            };
            values.insert(name.clone(), initial);
        }

        // 2. Apply assumptions by intersecting bounds.
        for (name, assumed) in assumptions.iter() {
            let entry = values.entry(name.clone()).or_insert(*assumed);
            *entry = intersect_bounds(*entry, *assumed);
        }

        // 3. Fixed-point iteration: propagate until no more changes.
        let max_iters = 100;
        let mut iter = 0;

        loop {
            iter += 1;
            if iter > max_iters {
                return Err(PldagError::MaxIterationsExceeded { max_iters });
            }

            let mut changed = false;

            // For each composite node:
            //  1) tighten its boolean bound using current variable bounds
            //  2) if boolean is forced to TRUE or FALSE, tighten variable bounds.
            for (name, node) in dag.iter() {
                let constraint = match node {
                    Node::Composite(c) => c,
                    Node::Primitive(_) => continue,
                };

                // Current boolean bound of this constraint node
                let bool_bound = values.get(name).cloned().unwrap_or((0, 1));
                let old_bool_bound = bool_bound;

                // (a) Evaluate constraint and intersect with current boolean bound.
                //
                // evaluate() returns:
                //   (1,1) => definitely true
                //   (0,0) => definitely false
                //   (0,1) => unknown
                let eval = constraint.evaluate(&values);
                let new_bool_bound = intersect_bounds(bool_bound, eval);
                if new_bool_bound != old_bool_bound {
                    values.insert(name.clone(), new_bool_bound);
                    changed = true;
                }

                // (b) If now forced TRUE, propagate on the constraint.
                let (lb, ub) = new_bool_bound;
                if lb == 1 && ub == 1 {
                    // Constraint is TRUE: sum(a_i * x_i) + bias >= 0
                    if tighten_constraint_true(constraint, &mut values) {
                        changed = true;
                    }
                } else if lb == 0 && ub == 0 {
                    // Constraint is FALSE:
                    // use the negated constraint, which is also of the form >= 0.
                    let neg = constraint.negate();
                    if tighten_constraint_true(&neg, &mut values) {
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        Ok(values)
    }

    pub fn reduce(
        dag: &HashMap<ID, Node>,
        fixed: &HashMap<String, i32>,
    ) -> Result<HashMap<ID, Node>> {
        let mut reduced: HashMap<ID, Node> = HashMap::new();

        'nodes: for (node_id, node) in dag.iter() {
            // Drop nodes that are fixed
            let node_key = node_id.to_string(); // assumes this matches `fixed` keys
            if fixed.contains_key(&node_key) {
                continue 'nodes;
            }

            match node {
                Node::Primitive(bound) => {
                    reduced.insert(node_id.clone(), Node::Primitive(*bound));
                }

                Node::Composite(constraint) => {
                    let mut new_coefficients: Vec<(String, i32)> = Vec::new();
                    let mut new_bias = constraint.bias;

                    for (var_name, coeff) in constraint.coefficients.iter() {
                        if let Some(&fixed_val) = fixed.get(var_name) {
                            let contribution = bound_multiply(*coeff, (fixed_val, fixed_val));
                            new_bias = bound_add(new_bias, contribution);
                        } else {
                            new_coefficients.push((var_name.clone(), *coeff));
                        }
                    }

                    // If constant after substitution, drop it too (it's fixed now).
                    if new_coefficients.is_empty() {
                        let (lb, ub) = new_bias;
                        // constraint is bias >= 0
                        if lb >= 0 || ub < 0 {
                            continue 'nodes;
                        }
                        // If ambiguous interval, keep it (rare; depends on your bias type).
                    }

                    reduced.insert(
                        node_id.clone(),
                        Node::Composite(Constraint {
                            coefficients: new_coefficients,
                            bias: new_bias,
                        }),
                    );
                }
            }
        }

        Ok(reduced)
    }

    /// Static propagation function that works on any DAG without requiring storage.
    ///
    /// This is useful when you need to propagate bounds through a sub-DAG or
    /// a DAG that is not stored in the main Pldag storage.
    ///
    /// # Arguments
    /// * `dag` - HashMap mapping node IDs to their corresponding nodes
    /// * `assignments` - Initial assignment of bounds to variables
    ///
    /// # Returns
    /// Complete assignment including bounds for all reachable nodes
    pub fn propagate_dag<K>(
        dag: &HashMap<ID, Node>,
        assignments: impl IntoIterator<Item = (K, Bound)>,
    ) -> Result<Assignment>
    where
        K: ToString,
    {
        // Build reverse dependency map (parent_ids) from the dag
        let mut parent_ids: HashMap<String, Vec<String>> = HashMap::new();
        for (node_id, node) in dag.iter() {
            if let Node::Composite(constraint) = node {
                for (input_id, _) in constraint.coefficients.iter() {
                    parent_ids
                        .entry(input_id.clone())
                        .or_insert_with(Vec::new)
                        .push(node_id.clone());
                }
            }
        }

        // Convert assignments into IndexMap<String, Bound>
        let assignments_map: IndexMap<String, Bound> = assignments
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        // Initialize results with the provided assignments
        let mut results: IndexMap<String, Bound> = IndexMap::new();

        // Extract all keys from the initial assignments
        let mut queue: Vec<String> = assignments_map.keys().cloned().collect();

        // Keep track of visited nodes to avoid reprocessing
        let mut visited = HashSet::new();

        while !queue.is_empty() {
            let mut next_batch: Vec<String> = Vec::new();
            let mut processed_this_batch: Vec<String> = Vec::new();

            // Loop over all nodes in queue
            while let Some(node_id) = queue.pop() {
                if visited.contains(&node_id) {
                    continue; // Already processed this node
                }

                let node = match dag.get(&node_id) {
                    Some(n) => n,
                    None => {
                        return Err(PldagError::NodeNotFound {
                            node_id: node_id.clone(),
                        })
                    }
                };

                match node {
                    Node::Primitive(primitive) => {
                        // If this node is in the initial assignments, use that value
                        if let Some(bound) = assignments_map.get(&node_id) {
                            // However, if the assigned bound is looser than the primitive's inherent bound,
                            // we return an error since it is not allowed.
                            if bound.0 < primitive.0 || bound.1 > primitive.1 {
                                return Err(PldagError::NodeOutOfBounds {
                                    node_id: node_id.clone(),
                                    got_bound: *bound,
                                    expected_bound: *primitive,
                                });
                            }

                            results.insert(node_id.to_string(), *bound);
                        } else {
                            // Otherwise, use the primitive's inherent bound
                            results.insert(node_id.to_string(), *primitive);
                        }
                        visited.insert(node_id.clone());
                        processed_this_batch.push(node_id.clone());
                    }
                    Node::Composite(constraint) => {
                        // Filter coefficients and calculate bias
                        let bias: i32 = constraint.bias.0; // Using lower bound for bias
                        let coefficients = &constraint.coefficients;

                        // Check if all input variables are in results
                        let all_inputs_available = coefficients
                            .iter()
                            .all(|(input_id, _)| results.contains_key(input_id));

                        if !all_inputs_available {
                            // Put the missing coefficients back to the queue
                            for (input_id, _) in coefficients.iter() {
                                if !results.contains_key(input_id) && !next_batch.contains(input_id)
                                {
                                    next_batch.push(input_id.clone());
                                }
                            }
                            // Not all inputs are ready, push this node back to the queue
                            next_batch.push(node_id.clone());
                            continue;
                        }

                        // Get coefficient values from results
                        let mut coef_vals = HashMap::new();
                        for (input_id, _) in coefficients.iter() {
                            if let Some(val) = results.get(input_id) {
                                coef_vals.insert(input_id.clone(), val.clone());
                            }
                        }

                        // Calculate result
                        // Instead of allocating a Vec, accumulate directly.
                        let summed = coefficients.iter().fold((0, 0), |acc, (input_id, coef)| {
                            let bound = coef_vals.get(input_id).unwrap();
                            let prod = bound_multiply(*coef, *bound);
                            bound_add(acc, prod)
                        });
                        let biased = bound_add(summed, (bias, bias));

                        results.insert(
                            node_id.to_string(),
                            ((biased.0 >= 0) as i32, (biased.1 >= 0) as i32),
                        );
                        visited.insert(node_id.clone());
                        processed_this_batch.push(node_id.clone());
                    }
                }
            }

            // Add dependent nodes to next batch using our built parent_ids map
            for node_id in processed_this_batch {
                if let Some(parents) = parent_ids.get(&node_id) {
                    for dependent in parents {
                        if !visited.contains(dependent) && !next_batch.contains(dependent) {
                            next_batch.push(dependent.clone());
                        }
                    }
                }
            }

            if !next_batch.is_empty() {
                queue = next_batch;
            }
        }

        Ok(results)
    }

    /// Propagates bounds through the DAG bottom-up.
    ///
    /// Starting from the given variable assignments, this method computes bounds
    /// for all composite nodes by propagating constraints upward through the DAG.
    ///
    /// # Arguments
    /// * `assignment` - Initial assignment of bounds to variables
    /// * `to_root` - Optional root node to end propagation at
    ///
    /// # Returns
    /// Complete assignment including bounds for all reachable nodes
    pub fn propagate<K>(&self, assignments: impl IntoIterator<Item = (K, Bound)>, to_root: Option<K>) -> Result<Assignment>
    where
        K: ToString,
    {
        // Convert assignments into IndexMap<String, Bound>
        let assignments_map: IndexMap<String, Bound> = assignments
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();

        // Initialize results with the provided assignments
        let mut results: IndexMap<String, Bound> = IndexMap::new();

        // Extract all keys from the initial assignments
        let mut queue: Vec<String> = assignments_map.keys().cloned().collect();

        // Keep track of visited nodes to avoid reprocessing1
        let mut visited = HashSet::new();
        while (queue.len() > 0) || (to_root.is_some() && !visited.contains(&to_root.as_ref().unwrap().to_string())) {
            let mut next_batch: Vec<String> = Vec::new();
            let mut processed_this_batch: Vec<String> = Vec::new();
            let batch_incoming = self.storage.get_nodes(&queue);

            // Loop over all nodes in queue
            while let Some(node_id) = queue.pop() {
                if visited.contains(&node_id) {
                    continue; // Already processed this node
                }

                let node = match batch_incoming.get(&node_id) {
                    Some(n) => n,
                    None => {
                        return Err(PldagError::NodeNotFound {
                            node_id: node_id.clone(),
                        })
                    }
                };

                match node {
                    Node::Primitive(primitive) => {
                        // If this node is in the initial assignments, use that value
                        if let Some(bound) = assignments_map.get(&node_id) {

                            // However, if the assigned bound is looser than the primitive's inherent bound,
                            // we return an error since it is not allowed.
                            if bound.0 < primitive.0 || bound.1 > primitive.1 {
                                return Err(PldagError::NodeOutOfBounds {
                                    node_id: node_id.clone(),
                                    got_bound: *bound,
                                    expected_bound: *primitive,
                                });
                            }

                            results.insert(node_id.to_string(), *bound);
                        } else {
                            // Otherwise, use the primitive's inherent bound
                            results.insert(node_id.to_string(), *primitive);
                        }
                        visited.insert(node_id.clone());
                        processed_this_batch.push(node_id.clone());
                    }
                    Node::Composite(constraint) => {
                        // Filter coefficients and calculate bias
                        let bias: i32 = constraint.bias.0; // Using lower bound for bias
                        let coefficients = &constraint.coefficients;

                        // Check if all input variables are in results
                        let all_inputs_available = coefficients
                            .iter()
                            .all(|(input_id, _)| results.contains_key(input_id));

                        if !all_inputs_available {
                            // Put the missing coefficients back to the queue
                            for (input_id, _) in coefficients.iter() {
                                if !results.contains_key(input_id) && !next_batch.contains(input_id)
                                {
                                    next_batch.push(input_id.clone());
                                }
                            }
                            // Not all inputs are ready, push this node back to the queue
                            next_batch.push(node_id.clone());
                            continue;
                        }

                        // Get coefficient values from results
                        let mut coef_vals = HashMap::new();
                        for (input_id, _) in coefficients.iter() {
                            if let Some(val) = results.get(input_id) {
                                coef_vals.insert(input_id.clone(), val.clone());
                            }
                        }

                        // Calculate result
                        // Instead of allocating a Vec, accumulate directly.
                        let summed = coefficients.iter().fold((0, 0), |acc, (input_id, coef)| {
                            let bound = coef_vals.get(input_id).unwrap();
                            let prod = bound_multiply(*coef, *bound);
                            bound_add(acc, prod)
                        });
                        let biased = bound_add(summed, (bias, bias));

                        results.insert(
                            node_id.to_string(),
                            ((biased.0 >= 0) as i32, (biased.1 >= 0) as i32),
                        );
                        visited.insert(node_id.clone());
                        processed_this_batch.push(node_id.clone());
                    }
                }
            }

            // Add dependent nodes to next batch
            let batch_outgoing = self.storage.get_parent_ids(&processed_this_batch);
            for outgoing in batch_outgoing.values() {
                for dependent in outgoing.into_iter() {
                    if !visited.contains(dependent) && !next_batch.contains(&dependent) {
                        next_batch.push(dependent.clone());
                    }
                }
            }

            if !next_batch.is_empty() {
                queue = next_batch;
            }
        }

        Ok(results)
    }

    /// Propagates bounds using default primitive variable bounds.
    ///
    /// Convenience method that calls `propagate` with the default bounds
    /// of all primitive variables as defined in the DAG.
    ///
    /// # Returns
    /// Complete assignments with bounds for all nodes
    pub fn propagate_default(&self) -> Result<Assignment> {
        let primitives = self.get_primitives(vec![]);
        self.propagate(primitives, None)
    }

    /// Computes ranks for all nodes in the DAG.
    ////
    /// Ranks represent the longest distance from any root node to each node.
    ///// # Arguments
    /// * `dag` - mapping from node name to Node (Primitive / Composite)
    ///
    /// # Returns
    /// A HashMap of node IDs to their corresponding ranks
    pub fn ranks(dag: &HashMap<ID, Node>) -> Result<HashMap<ID, usize>> {
        // Compute ranks given the resolved DAG
        let dependencies = Self::dependency_map(dag);
        let topo = Self::topological_sort(&dag, &dependencies)?;

        // Compute ranks via reverse topological order
        let mut ranks: HashMap<String, usize> = HashMap::new();
        for node_id in topo.iter().rev() {
            if let Some(child_ids) = dependencies.get(node_id) {
                if child_ids.is_empty() {
                    ranks.insert(node_id.clone(), 0);
                } else {
                    let max_child_rank = child_ids
                        .iter()
                        .filter_map(|child_id| ranks.get(child_id))
                        .max()
                        .unwrap_or(&0);
                    ranks.insert(node_id.clone(), 1 + max_child_rank);
                }
            } else {
                ranks.insert(node_id.clone(), 0);
            }
        }

        Ok(ranks)
    }

    pub fn topological_sort(
        dag: &HashMap<ID, Node>,
        dependency_map: &HashMap<ID, Vec<ID>>,
    ) -> Result<Vec<ID>> {
        let mut in_degree: HashMap<String, usize> =
            dag.keys().map(|node_id| (node_id.clone(), 0)).collect();

        for node_id in dag.keys() {
            if let Some(child_ids) = dependency_map.get(node_id) {
                for child_id in child_ids {
                    *in_degree.entry(child_id.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut queue: Vec<String> = in_degree
            .iter()
            .filter_map(|(node_id, &deg)| {
                if deg == 0 {
                    Some(node_id.clone())
                } else {
                    None
                }
            })
            .collect();
        let mut result: Vec<String> = Vec::new();

        while !queue.is_empty() {
            let node_id = queue.pop().unwrap();
            result.push(node_id.clone());
            if let Some(child_ids) = dependency_map.get(&node_id) {
                for child_id in child_ids {
                    if let Some(deg) = in_degree.get_mut(child_id) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push(child_id.clone());
                        }
                    }
                }
            }
        }

        debug_assert_eq!(result.len(), dag.len());
        Ok(result)
    }

    pub fn dependency_map(dag: &HashMap<ID, Node>) -> HashMap<ID, Vec<ID>> {
        dag.iter()
            .map(|(node_id, node)| {
                let child_ids = match node {
                    Node::Composite(constraint) => constraint
                        .coefficients
                        .iter()
                        .map(|(child_id, _)| child_id.clone())
                        .collect::<Vec<String>>(),
                    _ => Vec::new(),
                };
                (node_id.clone(), child_ids)
            })
            .collect()
    }

    #[cfg(feature = "glpk")]
    #[cfg_attr(feature = "trace", tracing::instrument(skip_all))]
    /// Solve the supplied objectives in-process with GLPK.
    /// Only available when the crate is compiled with `--features glpk`
    ///
    /// # Arguments
    /// * `roots` - Vector of root node IDs to define the sub-DAG for solving. If empty, uses the entire DAG.
    /// * `objectives` - Vector of ID to value mapping representing different objective functions to solve
    /// * `assume` - Fixed variable assignments to apply before solving
    /// * `maximize` - If true, maximizes the objective; if false, minimizes it
    ///
    /// # Returns
    /// Vector of optional valued assignments, one for each objective. None if infeasible.
    pub fn solve(
        dag: &HashMap<ID, Node>,
        objectives: Vec<HashMap<&str, f64>>,
        assume: HashMap<&str, Bound>,
        maximize: bool,
    ) -> Result<Vec<Option<Assignment>>> {
        use glpk_rust::{
            solve_ilps, IntegerSparseMatrix, Solution, SparseLEIntegerPolyhedron, Status, Variable,
        };

        // Convert the PL-DAG to a polyhedron representation
        let polyhedron = Self::to_sparse_polyhedron(dag, true)?;

        // Validate assume that the bounds does not override column bounds
        for (key, bound) in assume.iter() {
            if let Some(idx) = polyhedron.columns.iter().position(|col| col == key) {
                let col_bound = polyhedron.column_bounds[idx];
                if bound.0 < col_bound.0 || bound.1 > col_bound.1 {
                    return Err(PldagError::NodeOutOfBounds {
                        node_id: key.to_string(),
                        got_bound: *bound,
                        expected_bound: col_bound,
                    });
                }
            }
        }

        // Convert sparse matrix to the format expected by glpk-rust
        // NOTE: As soon as the polyhedron is made, the order of the columns are vital.
        // Therefore always use polyhedron.columns to get the variable names in the correct order.
        let mut glpk_matrix = SparseLEIntegerPolyhedron {
            a: IntegerSparseMatrix {
                rows: polyhedron.a.rows.iter().map(|&x| x as i32).collect(),
                cols: polyhedron.a.cols.iter().map(|&x| x as i32).collect(),
                vals: polyhedron.a.vals.iter().map(|&x| -1 * x).collect(),
            },
            b: polyhedron.b.iter().map(|&x| (0, -1 * x)).collect(),
            variables: polyhedron
                .columns
                .iter()
                .zip(polyhedron.column_bounds.iter())
                .map(|(key, bound)| Variable {
                    id: key.as_str(),
                    bound: *assume.get(key.as_str()).unwrap_or(&(bound.0, bound.1)),
                })
                .collect(),
            double_bound: false,
        };

        // If there are no constraints, insert a dummy row
        if glpk_matrix.a.rows.is_empty() {
            for i in 0..polyhedron.columns.len() {
                glpk_matrix.a.rows.push(0);
                glpk_matrix.a.cols.push(i as i32);
                glpk_matrix.a.vals.push(0);
            }
            glpk_matrix.b.push((0, 0));
        }

        let solutions: Vec<Solution>;
        #[cfg(feature = "trace")]
        {
            let span = tracing::span!(tracing::Level::INFO, "solve_ilps");
            solutions = span.in_scope(|| solve_ilps(&mut glpk_matrix, objectives, maximize, false));
        }
        #[cfg(not(feature = "trace"))]
        {
            solutions = solve_ilps(&mut glpk_matrix, objectives, maximize, false);
        }

        return Ok(solutions
            .iter()
            .map(|solution| {
                if solution.status == Status::Optimal {
                    let mut assignment: Assignment = IndexMap::new();
                    for col_name in polyhedron.columns.iter() {
                        let value = solution.solution.get(col_name).unwrap_or(&0);
                        assignment.insert(col_name.clone(), (*value, *value));
                    }
                    Some(assignment)
                } else {
                    None
                }
            })
            .collect());
    }

    /// Extracts a sub-DAG containing all nodes reachable from the given roots.
    /// NOTE: if roots is empty, returns the entire DAG.
    ///
    /// # Arguments
    /// * `roots` - Vector of root node IDs to start the sub-DAG extraction
    ///
    /// # Returns
    /// A HashMap of node IDs to their corresponding nodes in the sub-DAG
    pub fn sub_dag(&self, roots: Vec<ID>) -> Result<HashMap<ID, Node>> {
        // By default use Rust implementation
        // If no roots, return empty tree
        if roots.is_empty() {
            return Ok(self.dag());
        }

        let mut queue: Vec<String> = roots;

        // Accumulate nodes for larger batches
        let mut sub_dag: HashMap<String, Node> = HashMap::new();

        while queue.len() > 0 {
            // Batch fetch incoming edges for current batch
            let all_incoming = self.storage.get_nodes(&queue);

            // Check that we got all nodes from queue in all_coming. 
            // Else we return a NodeNotFound error.
            for node_id in queue.iter() {
                if !all_incoming.contains_key(node_id) {
                    return Err(PldagError::NodeNotFound { node_id: node_id.to_string() });
                }
            }

            let mut next_batch = Vec::new();

            for (input_id, incoming) in all_incoming.iter() {

                // Add this node to the sub_dag
                sub_dag.insert(input_id.clone(), incoming.clone());

                // If it is a composite, enqueue its coefficients if not already present
                match incoming {
                    Node::Primitive(_) => {}
                    Node::Composite(constraint) => {
                        // Enqueue all coefficient variable IDs
                        for (coef_id, _) in constraint.coefficients.iter() {
                            if sub_dag.contains_key(coef_id) {
                                continue;
                            }
                            if !next_batch.contains(coef_id) {
                                next_batch.push(coef_id.clone());
                            }
                        }
                    }
                }
            }
            queue = next_batch;
        }

        Ok(sub_dag)
    }

    pub fn dag(&self) -> HashMap<ID, Node> {
        self.storage.get_all_nodes()
    }

    /// Converts the PL-DAG to a sparse polyhedron for ILP solving.
    ///
    /// Transforms the logical constraints in the DAG into a system of linear
    /// inequalities suitable for integer linear programming solvers.
    ///
    /// # Arguments
    /// * `dag` - mapping from node ID to Node (Primitive / Composite)
    /// * `double_binding` - If true, creates bidirectional implications for composite nodes
    ///
    /// # Returns
    /// A `SparsePolyhedron` representing the DAG constraints
    pub fn to_sparse_polyhedron(dag: &HashMap<ID, Node>, double_binding: bool) -> Result<SparsePolyhedron> {
        // Create a new sparse matrix
        let mut a_matrix = SparseIntegerMatrix::new();
        let mut b_vector: Vec<i32> = Vec::new();

        // Filter out all Nodes that are primitives
        let primitives: IndexMap<&String, Bound> = dag
            .iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .filter_map(|(key, node)| {
                if let Node::Primitive(bound) = &node {
                    Some((key, *bound))
                } else {
                    None
                }
            })
            .collect();

        // Filter out all Nodes that are composites
        let composites: IndexMap<&String, &Constraint> = dag
            .iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .filter_map(|(key, node)| {
                if let Node::Composite(constraint) = &node {
                    Some((key, constraint))
                } else {
                    None
                }
            })
            .collect();

        // Create a index mapping for all columns
        let column_names_map: IndexMap<String, usize> = primitives
            .keys()
            .chain(composites.keys())
            .enumerate()
            .map(|(i, key)| ((*key).clone(), i))
            .collect();

        // Keep track of the current row index
        let mut row_i: usize = 0;

        for (key, composite) in composites {
            // Get the index of the current key
            let ki = *column_names_map.get(key).unwrap();

            // Construct the inner bound of the composite
            let mut coef_bounds: IndexMap<String, Bound> = IndexMap::new();
            for (coef_key, _) in composite.coefficients.iter() {
                if let Some(node) = dag.get(coef_key) {
                    match node {
                        Node::Primitive(b) => {
                            coef_bounds.insert(coef_key.clone(), *b);
                        }
                        _ => {
                            coef_bounds.insert(coef_key.clone(), (0, 1));
                        }
                    }
                } else {
                    return Err(PldagError::NodeNotFound {
                        node_id: coef_key.clone(),
                    });
                }
            }

            // An inner bound $\text{ib}(\phi)$ of a
            // linear inequality constraint $\phi$ is the sum of all variable's
            // step bounds, excl bias and before evaluated with the $\geq$ operator.
            // For instance, the inner bound of the linear inequality $-2x + y + z \geq 0$,
            // where $x,y,z \in \{0,1\}^3$, is $[-2, 2]$, since the lowest value the
            // sum can be is $-2$ (given from the combination $x=1, y=0, z=0$) and the
            // highest value is $2$ (from $x=0, y=1, z=1$).
            let ib_phi = composite.dot(&coef_bounds);

            // 1. Let $d = max(|ib(phi)|) + |bias|$.
            let d_pi = std::cmp::max(ib_phi.0.abs(), ib_phi.1.abs()) + composite.bias.0.abs();

            // Push values for pi variable coefficient
            a_matrix.rows.push(row_i);
            a_matrix.cols.push(ki);
            a_matrix.vals.push(-d_pi);

            // Push values for phi coefficients
            for (coef_key, coef_val) in composite.coefficients.iter() {
                let ck_index: usize = *column_names_map.get(coef_key).unwrap();
                a_matrix.rows.push(row_i);
                a_matrix.cols.push(ck_index);
                a_matrix.vals.push(*coef_val);
            }

            // Push bias value
            let b_phi = composite.bias.0 + d_pi;
            b_vector.push(-1 * b_phi);

            if double_binding {
                // ## Transforming $\phi \rightarrow \pi$
                // First note that $\phi \rightarrow \pi$ is equivilant to $\neg \phi \lor \pi$.

                // 1. Calculate $\phi' = \neg \phi$
                // 2. Let $d = \text{max}(|\text{ib}(\phi')|)$
                // 3. Append $(d - \text{bias}(\phi'))\pi$ to left side of $\phi'$.
                // 4. Let bias be as is

                let phi_prim = composite.negate();
                let phi_prim_ib = phi_prim.dot(&coef_bounds);
                let d_phi_prim = std::cmp::max(phi_prim_ib.0.abs(), phi_prim_ib.1.abs());
                let pi_coef = d_phi_prim - phi_prim.bias.0;

                // Push values for pi variable coefficient
                a_matrix.rows.push(row_i + 1);
                a_matrix.cols.push(ki);
                a_matrix.vals.push(pi_coef);

                // Push values for phi coefficients
                for (phi_coef_key, phi_coef_val) in phi_prim.coefficients.iter() {
                    let ck_index: usize = *column_names_map.get(phi_coef_key).unwrap();
                    a_matrix.rows.push(row_i + 1);
                    a_matrix.cols.push(ck_index);
                    a_matrix.vals.push(*phi_coef_val);
                }

                // Push bias value
                b_vector.push(-1 * phi_prim.bias.0);

                // Increment one more row when double binding
                row_i += 1;
            }

            // Increment the row index
            row_i += 1;
        }

        // Set the shape of the A matrix
        a_matrix.shape = (row_i, column_names_map.len());

        // Create the polyhedron
        let polyhedron = SparsePolyhedron {
            a: a_matrix,
            b: b_vector,
            columns: column_names_map.keys().cloned().collect(),
            column_bounds: column_names_map
                .keys()
                .map(|key| {
                    match dag.get(key) {
                        Some(Node::Primitive(bound)) => *bound,
                        _ => (0, 1),
                    }
                })
                .collect(),
        };

        return Ok(polyhedron);
    }

    /// Converts the PL-DAG to a sparse polyhedron with default settings.
    ///
    /// Convenience method that calls `to_sparse_polyhedron` with all options enabled:
    /// double_binding=true, integer_constraints=true, fixed_constraints=true.
    ///
    /// # Returns
    /// A `SparsePolyhedron` with full constraint encoding
    pub fn to_sparse_polyhedron_default(dag: &HashMap<ID, Node>) -> Result<SparsePolyhedron> {
        Self::to_sparse_polyhedron(dag, true)
    }

    /// Converts the PL-DAG to a dense polyhedron.
    ///
    /// # Arguments
    /// * `double_binding` - If true, creates bidirectional implications
    ///
    /// # Returns
    /// A `DensePolyhedron` representing the DAG constraints
    pub fn to_dense_polyhedron(dag: &HashMap<ID, Node>, double_binding: bool) -> Result<DensePolyhedron> {
        // Convert to sparse polyhedron first
        let sparse_polyhedron = Self::to_sparse_polyhedron(dag, double_binding)?;
        // Convert sparse to dense polyhedron
        Ok(sparse_polyhedron.into())
    }

    /// Converts the PL-DAG to a dense polyhedron with default settings.
    ///
    /// # Returns
    /// A `DensePolyhedron` with all constraint options enabled
    pub fn to_dense_polyhedron_default(dag: &HashMap<ID, Node>) -> Result<DensePolyhedron> {
        Self::to_dense_polyhedron(dag, true)
    }

    /// Retrieves all primitive variables from the given PL-DAG roots.
    ///
    /// # Returns
    /// An `IndexMap` mapping variable IDs to their corresponding `Bound` objects
    pub fn get_primitives(&self, roots: Vec<String>) -> IndexMap<String, Bound> {
        let sub_dag = self.sub_dag(roots).unwrap();
        sub_dag
            .iter()
            .filter_map(|(key, node)| {
                if let Node::Primitive(bound) = node {
                    Some((key.clone(), *bound))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Retrieves all composite constraints from the PL-DAG.
    ///
    /// # Returns
    /// An `IndexMap` mapping constraint IDs to their corresponding `Constraint` objects
    pub fn get_composites(&self, roots: Vec<String>) -> IndexMap<String, Constraint> {
        let sub_dag = self.sub_dag(roots).unwrap();
        sub_dag
            .iter()
            .filter_map(|(key, node)| {
                if let Node::Composite(constraint) = node {
                    Some((key.clone(), constraint.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Retrieves a node by its ID.
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the node to retrieve
    /// # Returns
    /// An `Option<Node>` which is Some(Node) if found, or None if not found
    pub fn get_node(&self, id: &str) -> Option<Node> {
        self.storage.get_nodes(&[id.to_string()]).get(id).cloned()
    }

    /// Deletes a node from the PL-DAG by its ID.
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the node to delete
    /// 
    /// Note: 
    ///
    /// Returns nothing.
    pub fn delete_node(&mut self, id: &str) -> Result<()> {
        let parents = self.storage.get_parent_ids(&[id.to_string()]);
        if let Some(parents) = parents.get(id) {
            if !parents.is_empty() {
                return Err(PldagError::NodeReferenced {
                    node_id: id.to_string(),
                    referencing_nodes: parents.clone(),
                });
            }
        }
        self.storage.delete(id);
        return Ok(());
    }

    /// Creates a primitive (leaf) variable with the specified bounds.
    ///
    /// Primitive variables represent the base variables in the DAG and have
    /// no dependencies on other nodes.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the variable
    /// * `bound` - The allowed range (min, max) for this variable
    pub fn set_primitive(&mut self, id: &str, bound: Bound) -> Result<ID> {
        self.storage.set_node(id, Node::Primitive(bound));
        Ok(id.to_string())
    }

    /// Creates multiple primitive variables with the same bounds.
    ///
    /// Convenience method to create several primitive variables at once.
    /// Duplicate IDs are automatically filtered out.
    ///
    /// # Arguments
    /// * `ids` - Iterator of unique identifiers for the variables
    /// * `bound` - The common bound to apply to all variables
    pub fn set_primitives<K>(&mut self, ids: impl IntoIterator<Item = K>, bound: Bound) -> Vec<Result<ID>>
    where
        K: ToString,
    {
        let unique_ids: IndexSet<String> = ids.into_iter().map(|k| k.to_string()).collect();
        let results: Vec<Result<ID>> = unique_ids
            .iter()
            .map(|id| self.set_primitive(id, bound))
            .collect();
        results
    }

    /// Creates a general linear inequality constraint.
    ///
    /// Creates a constraint of the form: sum(coeff_i * var_i) + bias >= 0.
    /// The constraint is automatically assigned a unique ID based on its content.
    ///
    /// # Arguments
    /// * `coefficient_variables` - Iterator of (variable_id, coefficient) pairs
    /// * `bias` - Constant bias term
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any coefficient ID doesn't exist
    pub fn set_gelineq<K>(
        &mut self,
        coefficient_variables: impl IntoIterator<Item = (K, i32)>,
        bias: i32,
    ) -> Result<ID>
    where
        K: ToString,
    {
        // Ensure coefficients have unique keys by summing duplicate values
        let mut unique_coefficients: IndexMap<ID, i32> = IndexMap::new();
        for (key, value) in coefficient_variables {
            *unique_coefficients.entry(key.to_string()).or_insert(0) += value;
        }

        // Check that all coefficient IDs exist in storage
        for coef_id in unique_coefficients.keys() {
            if !self.storage.node_exists(coef_id) {
                return Err(PldagError::NodeNotFound {
                    node_id: coef_id.clone(),
                });
            }
        }

        let coefficient_variables: Vec<Coefficient> = unique_coefficients
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .collect();

        // Create a hash from the input data
        let hash = create_hash(&coefficient_variables, bias);

        // Return the hash as a string
        let id = hash.to_string();
        let constraint = Constraint {
            coefficients: coefficient_variables.clone(),
            bias: (bias, bias),
        };

        // Insert the constraint as a node
        self.storage.set_node(&id, Node::Composite(constraint));

        Ok(id.to_string())
    }

    /// Creates an "at least" constraint: sum(variables) >= value.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to sum
    /// * `value` - Minimum required sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_atleast<K>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> Result<ID>
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, 1)), -value)
    }

    pub fn set_atleast_ref<K, V>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> Result<ID>
    where
        K: ToString,
        V: ToString,
    {
        self.set_gelineq(
            references
                .into_iter()
                .map(|x| (x.to_string(), 1))
                .chain([(value.to_string(), -1)]),
            0,
        )
    }

    /// Creates an "at most" constraint: sum(variables) <= value.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to sum
    /// * `value` - Maximum allowed sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_atmost<K>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> Result<ID>
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, -1)), value)
    }

    pub fn set_atmost_ref<K, V>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> Result<ID>
    where
        K: ToString,
        V: ToString,
    {
        self.set_gelineq(
            references
                .into_iter()
                .map(|x| (x.to_string(), -1))
                .chain([(value.to_string(), 1)]),
            0,
        )
    }

    /// Creates an equality constraint: sum(variables) == value.
    ///
    /// Implemented as the conjunction of "at least" and "at most" constraints.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to sum (must be clonable)
    /// * `value` - Required exact sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_equal<K, I>(
        &mut self,
        references: I,
        value: i32,
    ) -> Result<ID>
    where
        K: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast(references.clone(), value)?;
        let lb = self.set_atmost(references, value)?;
        self.set_and(vec![ub, lb])
    }

    pub fn set_equal_ref<K, V, I>(
        &mut self,
        references: I,
        value: V,
    ) -> Result<ID>
    where
        K: ToString,
        V: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast_ref(references.clone(), value.to_string())?;
        let lb = self.set_atmost_ref(references, value)?;
        self.set_and(vec![ub, lb])
    }

    /// Creates a logical AND constraint.
    ///
    /// Returns true if and only if ALL referenced variables are true.
    /// Implemented as: sum(variables) >= count(variables).
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to AND together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_and<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let length = unique_references.len();
        self.set_atleast(unique_references.iter().map(|x| x.as_str()), length as i32)
    }

    /// Creates a logical OR constraint.
    ///
    /// Returns true if AT LEAST ONE of the referenced variables is true.
    /// Implemented as: sum(variables) >= 1.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to OR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_or<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atleast(unique_references.iter().map(|x| x.as_str()), 1)
    }

    /// Creates a logical OPTIONAL constraint.
    ///
    /// Returns true no matter the referenced variables are true or false.
    /// Implemented as: sum(references) <= len(references).
    ///
    /// # Arguments
    /// * `references` - Variable IDs to make optional
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_optional<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atmost(unique_references.iter().map(|x| x.as_str()), unique_references.len() as i32)
    }

    /// Creates a logical NAND constraint.
    ///
    /// Returns true if NOT ALL of the referenced variables are true.
    /// Implemented as: sum(variables) <= count(variables) - 1.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to NAND together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_nand<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let length = unique_references.len();
        self.set_atmost(
            unique_references.iter().map(|x| x.as_str()),
            length as i32 - 1,
        )
    }

    /// Creates a logical NOR constraint.
    ///
    /// Returns true if NONE of the referenced variables are true.
    /// Implemented as: sum(variables) <= 0.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to NOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_nor<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0)
    }

    /// Creates a logical NOT constraint.
    ///
    /// Returns true if NONE of the referenced variables are true.
    /// Functionally equivalent to NOR. Implemented as: sum(variables) <= 0.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to negate
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_not<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0)
    }

    /// Creates a logical XOR constraint.
    ///
    /// Returns true if EXACTLY ONE of the referenced variables is true.
    /// Implemented as the conjunction of OR and "at most 1" constraints.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to XOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_xor<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let atleast = self.set_or(unique_references.iter().map(|x| x.as_str()))?;
        let atmost = self.set_atmost(unique_references.iter().map(|x| x.as_str()), 1)?;
        self.set_and(vec![atleast, atmost])
    }

    /// Creates a logical XNOR constraint.
    ///
    /// Returns true if an EVEN NUMBER of the referenced variables are true
    /// (including zero). Implemented as: (sum >= 2) OR (sum <= 0).
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to XNOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_xnor<K>(&mut self, references: impl IntoIterator<Item = K>) -> Result<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let atleast = self.set_atleast(unique_references.iter().map(|x| x.as_str()), 2)?;
        let atmost = self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0)?;
        self.set_or(vec![atleast, atmost])
    }

    /// Creates a logical IMPLICATION constraint: condition -> consequence.
    ///
    /// Returns true if the condition is false OR the consequence is true.
    /// Implemented as: NOT(condition) OR consequence.
    ///
    /// # Arguments
    /// * `condition` - The condition variable ID
    /// * `consequence` - The consequence variable ID
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_imply<C, Q>(&mut self, condition: C, consequence: Q) -> Result<ID>
    where
        C: ToString,
        Q: ToString,
    {
        let not_condition = self.set_not(vec![condition.to_string()])?;
        self.set_or(vec![not_condition, consequence.to_string()])
    }

    /// Creates a logical EQUIVALENCE constraint: lhs <-> rhs.
    ///
    /// Returns true if both variables have the same truth value.
    /// Implemented as: (lhs -> rhs) AND (rhs -> lhs).
    ///
    /// # Arguments
    /// * `lhs` - The left-hand side variable ID
    /// * `rhs` - The right-hand side variable ID
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub fn set_equiv<L, R>(&mut self, lhs: L, rhs: R) -> Result<ID>
    where
        L: ToString,
        R: ToString,
    {
        // Convert to strings first to avoid type mismatches
        let lhs_str = lhs.to_string();
        let rhs_str = rhs.to_string();

        let imply_lr = self.set_and(vec![lhs_str.clone(), rhs_str.clone()])?;
        let imply_rl = self.set_not(vec![rhs_str, lhs_str])?;
        self.set_or(vec![imply_lr, imply_rl])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a helper function that generates all primitive combinations
    // for a given PLDAG model, propagates them, and compares against the
    // corresponding polyhedron evaluations.
    fn primitive_combinations(model: &Pldag) -> Vec<IndexMap<String, i32>> {
        let tree = model.dag();
        let primitives: Vec<&String> = tree
            .iter()
            .filter_map(|(key, node)| {
                if let Node::Primitive(_) = node {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        let mut combinations: Vec<IndexMap<String, i32>> = Vec::new();

        let num_primitives = primitives.len();
        let num_combinations = 1 << num_primitives; // 2^n combinations

        for i in 0..num_combinations {
            let mut combo = IndexMap::new();
            for (j, &prim) in primitives.iter().enumerate() {
                let value = if (i & (1 << j)) != 0 { 1 } else { 0 };
                combo.insert(prim.clone(), value);
            }
            combinations.push(combo);
        }

        combinations
    }

    /// Helper: for every primitive combination,
    ///   1) run `propagate` on the PLDAG model  
    ///   2) build the corresponding assignments  
    ///   3) run `assume(root=1)` on the polyhedron  
    ///   4) evaluate the shrunken polyhedron on the same assignments  
    ///   5) assert they agree at `root`.
    fn evaluate_model_polyhedron(model: &Pldag, poly: &DensePolyhedron, root: &String) {
        for combo in primitive_combinations(model) {
            // build an IndexMap<&str,Bound> as propagate expects
            let interp = combo
                .iter()
                .map(|(k, &v)| (k.as_str(), (v, v)))
                .collect::<IndexMap<&str, Bound>>();

            // what the DAG says the root can be
            let prop = model.propagate(interp, Some(&root.clone())).unwrap();
            let model_root_val = *prop.get(root).unwrap();

            // now shrink the polyhedron by assuming root=1
            let mut assumption = HashMap::new();
            assumption.insert(root.clone(), 1);
            let shrunk = poly.assume(&assumption);

            // and evaluate that shrunk system on the same propagated bounds
            let poly_val = shrunk.evaluate(&prop);
            assert_eq!(
                poly_val, model_root_val,
                "Disagreement on {:?}: model={:?}, poly={:?}",
                combo, model_root_val, poly_val
            );
        }
    }

    /// Helper: create a primitive node with a simple [min, max] bound.
    fn prim(min: i32, max: i32) -> Node {
        Node::Primitive((min, max))
    }

    /// Helper: build a constraint: sum(coeffs) + bias >= 0
    fn cons(coeffs: Vec<(&str, i32)>, bias: i32) -> Node {
        let coefficients = coeffs
            .into_iter()
            .map(|(name, c)| (name.to_string(), c))
            .collect::<Vec<_>>();
        Node::Composite(Constraint {
            coefficients,
            bias: (bias, bias),
        })
    }

    #[test]
    fn test_propagate() {
        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]).unwrap();

        let result = model.propagate_default().unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        let result = model.propagate(assignments, None).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]).unwrap();
        let result = model.propagate_default().unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(assignments, None).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(assignments, None).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 0));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (0, 0));
        let result = model.propagate(assignments, None).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        // Test propagation to specific root only and check that the others are not included in the result
        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let or_1 = model.set_or(vec!["x", "z"]).unwrap();
        let or_2 = model.set_or(vec!["y", "z"]).unwrap();
        let or_3 = model.set_or(vec!["x", "y"]).unwrap();
        let root = model.set_and(vec![or_1.clone(), or_2.clone(), or_3.clone()]).unwrap();
        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        let result = model.propagate(assignments, Some(&or_1)).unwrap();
        assert_eq!(result.get("x").unwrap(), &(1, 1));
        assert_eq!(result.get(&or_1).unwrap(), &(1, 1));
        assert!(result.get(&or_2).is_none());
        assert!(result.get(&or_3).is_none());
        assert!(result.get(&root).is_none());
    }

    /// XOR already covered; test the OR gate
    #[test]
    fn test_propagate_or_gate() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let or_root = model.set_or(vec!["a", "b"]).unwrap();

        // No assignment: both inputs full [0,1], output [0,1]
        let res = model.propagate_default().unwrap();
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (1, 1));
        let res = model.propagate(interp, Some(&or_root.clone())).unwrap();
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (0, 0));
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp, Some(&or_root.clone())).unwrap();
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp, Some(&or_root.clone())).unwrap();
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[test]
    fn test_propagate_not_gate() {
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        let not_root = model.set_not(vec!["p"]).unwrap();

        // no assignment ⇒ [0,1]
        let res = model.propagate_default().unwrap();
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = IndexMap::<String, Bound>::new();
        interp.insert("p".into(), (0, 0));
        let res = model.propagate(interp, Some(not_root.clone())).unwrap();
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (1, 1));
        let res = model.propagate(interp, Some(&not_root.clone())).unwrap();
        assert_eq!(res[&not_root], (0, 0));
    }

    #[test]
    fn test_to_polyhedron_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        let root = m.set_and(vec!["x", "y"]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_or() {
        let mut m = Pldag::new();
        m.set_primitive("a".into(), (0, 1));
        m.set_primitive("b".into(), (0, 1));
        m.set_primitive("c".into(), (0, 1));
        let root = m.set_or(vec!["a", "b", "c"]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_not() {
        let mut m = Pldag::new();
        m.set_primitive("p".into(), (0, 1));
        let root = m.set_not(vec!["p"]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_xor() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        m.set_primitive("z".into(), (0, 1));
        let root = m.set_xor(vec!["x", "y", "z"]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_nested() {
        // Build a small two‐level circuit:
        //   w = AND(x,y),  v = OR(w, NOT(z))
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        m.set_primitive("z".into(), (0, 1));

        let w = m.set_and(vec!["x", "y"]).unwrap();
        let nz = m.set_not(vec!["z"]).unwrap();
        let v = m.set_or(vec![w.clone(), nz.clone()]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &v);
    }

    /// Nested/composed AND then XOR:
    ///   w = AND(x,y);  v = XOR(w,z)
    #[test]
    fn test_propagate_nested_composite() {
        let mut model = Pldag::new();
        model.set_primitive("x".into(), (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z".into(), (0, 1));

        let w = model.set_and(vec!["x", "y"]).unwrap();
        let v = model.set_xor(vec![w.clone(), "z".into()]).unwrap();

        // no assignment: everything [0,1]
        let res = model.propagate_default().unwrap();
        for var in &["x", "y", "z"] {
            assert_eq!(res[*var], (0, 1), "{}", var);
        }
        assert_eq!(res[&w], (0, 1));
        assert_eq!(res[&v], (0, 1));

        // x=1,y=1,z=0 ⇒ w=1,v=1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (1, 1));
        interp.insert("y", (1, 1));
        interp.insert("z", (0, 0));
        let res = model.propagate(interp, None).unwrap();
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (1, 1));
        interp.insert("z", (1, 1));
        let res = model.propagate(interp, None).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (0, 0));
        interp.insert("z", (0, 0));
        let res = model.propagate(interp, None).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (0, 0));
    }

    /// If you ever get an inconsistent assignment (out‐of‐bounds for a primitive),
    /// propagate should leave it as given (or you could choose to clamp / panic)—here
    /// we simply check that nothing blows up.
    #[test]
    fn test_propagate_out_of_bounds_should_crash() {
        let mut model = Pldag::new();
        model.set_primitive("u".into(), (0, 1));

        let mut interp = IndexMap::<&str, Bound>::new();
        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        interp.insert("u".into(), (5, 5));
        let res = model.propagate(interp, None);

        // Assert that we did get an error
        assert!(res.is_err());
    }

    #[test]
    fn test_to_polyhedron() {
        fn evaluate_model_polyhedron(model: &Pldag, polyhedron: &DensePolyhedron, root: &String) {
            for combination in primitive_combinations(model) {
                let assignments = combination
                    .iter()
                    .map(|(k, &v)| (k.as_str(), (v, v)))
                    .collect::<IndexMap<&str, Bound>>();
                let model_prop = model.propagate(assignments, None).unwrap();
                let model_eval = *model_prop.get(root).unwrap();
                let mut assumption = HashMap::new();
                assumption.insert(root.clone(), 1);
                let assumed_polyhedron = polyhedron.assume(&assumption);
                let assumed_poly_eval = assumed_polyhedron.evaluate(&model_prop);
                assert_eq!(assumed_poly_eval, model_eval);
            }
        }

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]).unwrap();
        let polyhedron: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]).unwrap();
        let polyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]).unwrap();
        let polyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);
    }

    /// Single‐operand composites should act as identity: root == operand
    #[test]
    fn test_to_polyhedron_single_operand_identity() {
        // AND(x) == x
        {
            let mut m = Pldag::new();
            m.set_primitive("x".into(), (0, 1));
            let root = m.set_and::<&str>(vec!["x"]).unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // OR(y) == y
        {
            let mut m = Pldag::new();
            m.set_primitive("y", (0, 1));
            let root = m.set_or(vec!["y"]).unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // XOR(z) == z
        {
            let mut m = Pldag::new();
            m.set_primitive("z".into(), (0, 1));
            let root = m.set_xor(vec!["z"]).unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[test]
    fn test_to_polyhedron_duplicate_operands_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        let root = m.set_and(vec!["x", "x"]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    /// Deeply nested 5‐level chain:
    ///    w1 = AND(a,b)
    ///    w2 = OR(w1,c)
    ///    w3 = XOR(w2,d)
    ///    root = NOT(w3)
    #[test]
    fn test_to_polyhedron_deeply_nested_chain() {
        let mut m = Pldag::new();
        // primitives a,b,c,d,e  (e unused but shows extra var)
        for &v in &["a", "b", "c", "d", "e"] {
            m.set_primitive(v.into(), (0, 1));
        }
        let a = "a";
        let b = "b";
        let c = "c";
        let d = "d";

        let w1 = m.set_and(vec![a, b]).unwrap();
        let w2 = m.set_or(vec![w1.clone(), c.to_string()]).unwrap();
        let w3 = m.set_xor(vec![w2.clone(), d.to_string()]).unwrap();
        let root = m.set_not(vec![w3.clone()]).unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_print_dense_matrix() {
        let mut matrix = DenseIntegerMatrix::new(3, 3);
        matrix.data[0][0] = 1;
        matrix.data[0][2] = 2;
        matrix.data[1][0] = 3;
        matrix.data[2][2] = 4;

        let output = format!("{}", matrix);
        let expected = "  1   0   2 \n  3   0   0 \n  0   0   4 \n";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_print_sparse_matrix() {
        let matrix = SparseIntegerMatrix {
            rows: vec![0, 0, 1, 2],
            cols: vec![0, 2, 0, 2],
            vals: vec![1, 2, 3, 4],
            shape: (3, 3),
        };

        let output = format!("{}", matrix);
        let expected = "  1   0   2 \n  3   0   0 \n  0   0   4 \n";
        assert_eq!(output, expected);
    }

    #[test]
    fn test_equiv() {
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let equiv = model.set_equiv("p", "q").unwrap();
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 1));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));

        model.set_primitive("p", (0, 0));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));
    }

    #[test]
    fn test_imply() {
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let equiv = model.set_imply("p", "q").unwrap();
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default().unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));
    }

    #[test]
    fn test_node_out_of_bounds_error() {
        // If we propagate a primitive with a bound that is outside its predefined range,
        // we should get a NodeOutOfBounds error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (2, 2)); // Out of bounds
        let result = model.propagate(interp, None);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (-1, 2)); // Out of bounds
        let result = model.propagate(interp, None);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (-1, -1)); // Out of bounds
        let result = model.propagate(interp, None);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (1, 1)); // Not out of bounds
        let result = model.propagate(interp, None);
        assert!(matches!(result, Ok(_)));
    }

    #[test]
    fn test_node_not_found_error_when_propagate() {
        // If we propagate a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]);
        assert!(matches!(result, Err(PldagError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[test]
    fn test_node_not_found_error_when_sub_dag() {
        // If we create a sub-dag with a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]);
        assert!(matches!(result, Err(PldagError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[test]
    fn test_node_not_found_error_when_to_polyhedron() {
        // If we convert to a polyhedron with a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]);
        assert!(matches!(result, Err(PldagError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[test]
    fn binary_cardinality_all_forced_to_one_when_true() {
        // x, y, z in [0,1]
        // A: x + y + z - 3 >= 0  <=>  x + y + z >= 3
        // A is assumed TRUE → x = y = z = 1

        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("z".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1), ("z", 1)], -3));

        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));

        let values = Pldag::tighten(&dag, &assumptions).unwrap();

        assert_eq!(values.get("x"), Some(&(1, 1)));
        assert_eq!(values.get("y"), Some(&(1, 1)));
        assert_eq!(values.get("z"), Some(&(1, 1)));
        assert_eq!(values.get("A"), Some(&(1, 1)));
    }

    #[test]
    fn binary_cardinality_false_does_not_tighten() {
        // x, y, z in [0,1]
        // A: x + y + z >= 3
        // A is FALSE → x + y + z <= 2
        // With [0,1] for all, this does NOT force any individual variable.

        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("z".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1), ("z", 1)], -3));

        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (0, 0)); // A forced FALSE

        let values = Pldag::tighten(&dag, &assumptions).unwrap();

        assert_eq!(values.get("x"), Some(&(0, 1)));
        assert_eq!(values.get("y"), Some(&(0, 1)));
        assert_eq!(values.get("z"), Some(&(0, 1)));
        assert_eq!(values.get("A"), Some(&(0, 0)));
    }

    #[test]
    fn chained_constraints_do_not_tighten_in_this_case() {
        // x, y, z ∈ [0,3]
        // A: x + y - 3 >= 0  <=>  x + y >= 3
        // B: y + z - 3 >= 0  <=>  y + z >= 3
        // Assume A = TRUE and B = TRUE.
        //
        // Interval reasoning alone cannot tighten x, y, or z here.

        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 3));
        dag.insert("y".into(), prim(0, 3));
        dag.insert("z".into(), prim(0, 3));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -3));
        dag.insert("B".into(), cons(vec![("y", 1), ("z", 1)], -3));

        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));
        assumptions.insert("B".into(), (1, 1));

        let values = Pldag::tighten(&dag, &assumptions).unwrap();

        let x = values.get("x").unwrap();
        let y = values.get("y").unwrap();
        let z = values.get("z").unwrap();

        // No tightening should happen on x, y, z with this propagation strength.
        assert_eq!(*x, (0, 3));
        assert_eq!(*y, (0, 3));
        assert_eq!(*z, (0, 3));

        // A and B must be true.
        assert_eq!(values.get("A"), Some(&(1, 1)));
        assert_eq!(values.get("B"), Some(&(1, 1)));
    }

    #[test]
    fn composite_as_boolean_in_another_constraint() {
        // A: x + y - 3 >= 0   (x + y >= 3), boolean node A
        // D: 5*A + z - 6 >= 0   (5*A + z >= 6)
        //
        // x,y,z ∈ [0,5]
        // Assume D is TRUE, but A is not explicitly assumed.
        //
        // From D:
        //  - If A were 0, then z >= 6 impossible (since z ≤ 5)
        //  -> so A must be 1
        //  -> D being TRUE forces A TRUE, then A TRUE forces x + y >= 3.

        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 5));
        dag.insert("y".into(), prim(0, 5));
        dag.insert("z".into(), prim(0, 5));

        // A: x + y - 3 >= 0
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -3));

        // D: 5*A + z - 6 >= 0  (A is treated as variable in [0,1])
        dag.insert("D".into(), cons(vec![("A", 5), ("z", 1)], -6));

        let mut assumptions = HashMap::new();
        assumptions.insert("D".into(), (1, 1)); // D must be true

        let values = Pldag::tighten(&dag, &assumptions).unwrap();

        let a = values.get("A").unwrap();
        let z = values.get("z").unwrap();
        let x = values.get("x").unwrap();
        let y = values.get("y").unwrap();

        // D true should force A = 1 (because with A=0, z >= 6 impossible)
        assert_eq!(*a, (1, 1), "expected A to be forced to TRUE by D");

        // With A = 1, D becomes: 5*1 + z - 6 >= 0 => z >= 1
        assert!(z.0 >= 1, "expected z lower bound >= 1, got {:?}", z);

        // x and y are not tightened by pure interval propagation
        assert_eq!(*x, (0, 5));
        assert_eq!(*y, (0, 5));
    }

    #[test]
    fn test_tighten_bounds_on_an_xor() {
        // A = B + C >= 2
        // B = x + y + z >= 1
        // C = -x -y -z >= -1

        // Assume A is TRUE, and x = (1, 1) then y and z must be (0, 0)
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("z".into(), prim(0, 1));
        dag.insert("B".into(), cons(vec![("x", 1), ("y", 1), ("z", 1)], -1));
        dag.insert("C".into(), cons(vec![("x", -1), ("y", -1), ("z", -1)], 1));
        dag.insert("A".into(), cons(vec![("B", 1), ("C", 1)], -2));
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));
        assumptions.insert("x".into(), (1, 1));
        let values = Pldag::tighten(&dag, &assumptions).unwrap();
        assert_eq!(values.get("y"), Some(&(0, 0)));
        assert_eq!(values.get("z"), Some(&(0, 0)));
    }

    #[test]
    fn test_estimate_count_upper_bounds_on_an_xor() {
        // A = B + C >= 2
        // B = x + y + z >= 1
        // C = -x -y -z >= -1
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("z".into(), prim(0, 1));
        dag.insert("B".into(), cons(vec![("x", 1), ("y", 1), ("z", 1)], -1));
        dag.insert("C".into(), cons(vec![("x", -1), ("y", -1), ("z", -1)], 1));
        dag.insert("A".into(), cons(vec![("B", 1), ("C", 1)], -2)); // A equiv xor(x,y,z)
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));
        assumptions.insert("x".into(), (1, 1));
        let ub_count = Pldag::estimate_upper_bound_count(&dag, &assumptions).unwrap();
        assert_eq!(ub_count, 1);
    }

    #[test]
    fn test_estimate_count_upper_bounds_on_an_and() {
        // A = x + y >= 2 when A is TRUE
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -2)); // A equiv and(x,y)
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));
        let ub_count = Pldag::estimate_upper_bound_count(&dag, &assumptions).unwrap();
        assert_eq!(ub_count, 1);

        // A = x + y >= 2 when A is FALSE
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -2)); // A equiv and(x,y)
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (0, 1));
        let ub_count = Pldag::estimate_upper_bound_count(&dag, &assumptions).unwrap();
        assert_eq!(ub_count, 4);
    }

    #[test]
    fn test_estimate_count_upper_bounds_on_an_or() {
        // A = x + y >= 1 when A is TRUE
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -1)); // A equiv or(x,y)
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (1, 1));
        let ub_count = Pldag::estimate_upper_bound_count(&dag, &assumptions).unwrap();
        assert_eq!(ub_count, 4);   
        
        // A = x + y >= 1 when A is FALSE
        let mut dag = HashMap::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("A".into(), cons(vec![("x", 1), ("y", 1)], -1)); // A equiv or(x,y)
        let mut assumptions = HashMap::new();
        assumptions.insert("A".into(), (0, 0));
        let ub_count = Pldag::estimate_upper_bound_count(&dag, &assumptions).unwrap();
        assert_eq!(ub_count, 1);   
    }

    #[test]
    fn test_simple_sub_dag_with_xor() {
        let mut model = Pldag::new();
        model.set_primitive("x".into(), (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z".into(), (0, 1));
        let root = model.set_xor(vec!["x", "y", "z"]).unwrap();
        let sub_dag = model.sub_dag(vec![root.clone()]).unwrap();
        assert!(sub_dag.get(&root).is_some());
    }

    #[test]
    fn test_delete_node_should_succeed() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let and_node = model.set_and(vec!["a", "b"]).unwrap();
        let delete_result = model.delete_node(&and_node);
        assert!(delete_result.is_ok());
        assert!(model.get_node(&and_node).is_none());
    }

    #[test]
    fn test_delete_primitives_should_succeed() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let delete_result_a = model.delete_node(&"a");
        let delete_result_b = model.delete_node(&"b");
        assert!(delete_result_a.is_ok());
        assert!(delete_result_b.is_ok());
        assert!(model.get_node(&"a").is_none());
        assert!(model.get_node(&"b").is_none());
    }

    #[test]
    fn test_delete_node_should_fail_for_still_having_references() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let and_node = model.set_and(vec!["a", "b"]).unwrap();
        model.set_or(vec![and_node.clone(), "a".into()]).unwrap();
        let delete_result = model.delete_node(&and_node);
        assert!(delete_result.is_err());
    }

    #[test]
    fn test_compute_ranks() {

        // Simple case: a and b are rank 0, and (a AND b) is rank 1
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let and_node = model.set_and(vec!["a", "b"]).unwrap();
        model.set_or(vec![and_node.clone(), "a".into()]).unwrap();
        let ranks = Pldag::ranks(&model.sub_dag(vec![]).unwrap()).unwrap();
        assert_eq!(ranks.get("a"), Some(&0));
        assert_eq!(ranks.get("b"), Some(&0));
        assert_eq!(ranks.get(&and_node), Some(&1));

        // More complex case 1
        let mut model = Pldag::new();
        model.set_primitive("x".into(), (0, 1));
        model.set_primitive("y".into(), (0, 1));
        let and_node = model.set_and(vec!["x", "y"]).unwrap();
        let not_node = model.set_not(vec![and_node.clone()]).unwrap();
        model.set_xor(vec![not_node.clone(), "x".into()]).unwrap();
        let ranks = Pldag::ranks(&model.sub_dag(vec![]).unwrap()).unwrap();
        assert_eq!(ranks.get("x"), Some(&0));
        assert_eq!(ranks.get("y"), Some(&0));
        assert_eq!(ranks.get(&and_node), Some(&1));
        assert_eq!(ranks.get(&not_node), Some(&2));

        // More complex case 2
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        model.set_primitive("q".into(), (0, 1));
        let equiv_node = model.set_equiv("p", "q").unwrap();
        let imply_node = model.set_imply("p", "q").unwrap();
        model.set_or(vec![equiv_node.clone(), imply_node.clone()]).unwrap();
        let ranks = Pldag::ranks(&model.sub_dag(vec![]).unwrap()).unwrap();
        assert_eq!(ranks.get("p"), Some(&0));
        assert_eq!(ranks.get("q"), Some(&0));
        assert_eq!(ranks.get(&equiv_node), Some(&2));
        assert_eq!(ranks.get(&imply_node), Some(&2));
    }

    #[test]
    fn test_reduce() {
        // A = B + C + D >= 3
        // B = x + y >= 2
        // C = y + z >= 2
        // D = a >= 1

        // and we give that a = 1, D = 1
        // This should return a new DAG like:

        // A = B + C >= 2
        // B = x + y >= 2
        // C = y + z >= 2
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("x".into(), (0, 1));
        model.set_primitive("y".into(), (0, 1));
        model.set_primitive("z".into(), (0, 1));
        let D = model.set_and(vec!["a"]).unwrap();
        let C = model.set_and(vec!["y", "z"]).unwrap();
        let B = model.set_and(vec!["x", "y"]).unwrap();
        let A = model.set_and(vec![B.clone(), C.clone(), D.clone()]).unwrap();
        let mut fixed = HashMap::new();
        fixed.insert("a".into(), 1);
        fixed.insert(D.clone(), 1);
        let dag = model.sub_dag(vec![A.clone()]).unwrap();
        let reduced_dag = Pldag::reduce(&dag, &fixed).unwrap();
        // Check that D and a is not in reduced DAG
        assert!(reduced_dag.get(&D).is_none());
        assert!(reduced_dag.get(&"a".to_string()).is_none());
        // Check that A, B, C, x, y, z are in reduced DAG
        assert!(reduced_dag.get(&A).is_some());
        assert!(reduced_dag.get(&B).is_some());
        assert!(reduced_dag.get(&C).is_some());
        assert!(reduced_dag.get(&"x".to_string()).is_some());
        assert!(reduced_dag.get(&"y".to_string()).is_some());
        assert!(reduced_dag.get(&"z".to_string()).is_some());

        // Propagate the reduced DAG with x = 1, y = 1, z = 1, which should satisfy A = 1, B = 1, C = 1
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), (1, 1));
        assignments.insert("y".to_string(), (1, 1));
        assignments.insert("z".to_string(), (1, 1));
        let propagated = Pldag::propagate_dag(&reduced_dag, assignments).unwrap();
        assert_eq!(propagated.get(&A).unwrap(), &(1, 1));
        assert_eq!(propagated.get(&B).unwrap(), &(1, 1));
        assert_eq!(propagated.get(&C).unwrap(), &(1, 1));
    }
}