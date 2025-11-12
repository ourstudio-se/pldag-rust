use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::collections::HashSet;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Index, RangeInclusive};

/// Represents a bound with minimum and maximum values.
/// Used to specify the allowed range for variables and constraints.
pub type Bound = (i64, i64);

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
fn create_hash(data: &Vec<(String, i64)>, num: i64) -> u64 {
    // Create a new hasher
    let mut hasher = DefaultHasher::new();

    // Hash the vector
    for (s, i) in data {
        s.hash(&mut hasher);
        i.hash(&mut hasher);
    }

    // Hash the standalone i64 value
    num.hash(&mut hasher);

    // Return the final hash value
    hasher.finish()
}

/// Checks if a bound represents a fixed value (min equals max).
///
/// # Arguments
/// * `b` - A bound tuple (min, max)
///
/// # Returns
/// `true` if the bound represents a single fixed value, `false` otherwise
fn bound_fixed(b: Bound) -> bool {
    b.0 == b.1
}

/// Checks if a bound represents a boolean range [0, 1].
///
/// # Arguments
/// * `b` - A bound tuple (min, max)
///
/// # Returns
/// `true` if the bound represents a boolean value (0 to 1), `false` otherwise
fn bound_bool(b: Bound) -> bool {
    b.0 == 0 && b.1 == 1
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
    return (b1.0 + b2.0, b1.1 + b2.1);
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
fn bound_multiply(k: i64, b: Bound) -> Bound {
    if k < 0 {
        return (k * b.1, k * b.0);
    } else {
        return (k * b.0, k * b.1);
    }
}

/// Calculates the span (range) of a bound.
///
/// # Arguments
/// * `b` - A bound tuple (min, max)
///
/// # Returns
/// The absolute difference between max and min values
fn bound_span(b: Bound) -> i64 {
    return (b.1 - b.0).abs();
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
    pub vals: Vec<i64>,
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
    pub data: Vec<Vec<i64>>,
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
    pub fn dot_product(&self, vector: &Vec<i64>) -> Vec<i64> {
        let mut result = vec![0; self.shape.0];
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                result[i] += self.data[i][j] * vector[j];
            }
        }
        result
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
    pub A: DenseIntegerMatrix,
    /// Right-hand side vector b
    pub b: Vec<i64>,
    /// Variable names corresponding to matrix columns
    pub columns: Vec<String>,
    /// Subset of columns that represent integer variables
    pub integer_columns: Vec<String>,
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
        for (ir, row) in self.A.data.iter().enumerate() {
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
    pub fn to_vector(&self, from_assignments: &HashMap<String, i64>) -> Vec<i64> {
        let mut vector: Vec<i64> = vec![0; self.columns.len()];
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
    pub fn assume(&self, values: &HashMap<String, i64>) -> DensePolyhedron {
        // 1) Make mutable copies of everything
        let mut new_A_data = self.A.data.clone(); // Vec<Vec<i64>>
        let mut new_b = self.b.clone(); // Vec<i64>
        let mut new_columns = self.columns.clone(); // Vec<String>
        let mut new_int_cols = self.integer_columns.clone(); // Vec<String>

        // 2) Find which columns we’re going to remove, along with their assigned values.
        //    We capture (idx_in_matrix, column_name, assigned_value).
        let mut to_remove: Vec<(usize, String, i64)> = values
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
        for (col_idx, col_name, fixed_val) in to_remove {
            for row in 0..new_A_data.len() {
                // subtract A[row][col_idx] * fixed_val from b[row]
                new_b[row] -= new_A_data[row][col_idx] * fixed_val;
                // now remove that column entry
                new_A_data[row].remove(col_idx);
            }
            // drop the column name
            new_columns.remove(col_idx);
            // if it was an integer column, drop it too
            new_int_cols.retain(|c| c != &col_name);
        }

        // 5) Rebuild the DenseIntegerMatrix with updated shape
        let new_shape = (new_A_data.len(), new_columns.len());
        let new_A = DenseIntegerMatrix {
            data: new_A_data,
            shape: new_shape,
        };

        // 6) Return the shrunken polyhedron
        DensePolyhedron {
            A: new_A,
            b: new_b,
            columns: new_columns,
            integer_columns: new_int_cols,
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
            .A
            .dot_product(&self.to_vector(&lower_bounds))
            .iter()
            .zip(&self.b)
            .all(|(a, b)| a >= b);

        let upper_result = self
            .A
            .dot_product(&self.to_vector(&upper_bounds))
            .iter()
            .zip(&self.b)
            .all(|(a, b)| a >= b);

        (lower_result as i64, upper_result as i64)
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
    pub A: SparseIntegerMatrix,
    /// Right-hand side vector b
    pub b: Vec<i64>,
    /// Variable names corresponding to matrix columns
    pub columns: Vec<String>,
    /// Subset of columns that represent integer variables
    pub integer_columns: Vec<String>,
}

impl SparsePolyhedron {
    fn get_hash(&self) -> u64 {
        let mut state = DefaultHasher::new();
        // Hash the SparseIntegerMatrix A
        self.hash(&mut state);
        state.finish()
    }
}

impl From<SparsePolyhedron> for DensePolyhedron {
    fn from(sparse: SparsePolyhedron) -> DensePolyhedron {
        let mut dense_matrix = DenseIntegerMatrix::new(sparse.A.shape.0, sparse.A.shape.1);
        for ((&row, &col), &val) in sparse.A.rows.iter().zip(&sparse.A.cols).zip(&sparse.A.vals) {
            dense_matrix.data[row][col] = val;
        }
        DensePolyhedron {
            A: dense_matrix,
            b: sparse.b,
            columns: sparse.columns,
            integer_columns: sparse.integer_columns,
        }
    }
}

impl From<DensePolyhedron> for SparsePolyhedron {
    fn from(dense: DensePolyhedron) -> SparsePolyhedron {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut vals = Vec::new();
        for (i, row) in dense.A.data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0 {
                    rows.push(i);
                    cols.push(j);
                    vals.push(val);
                }
            }
        }
        SparsePolyhedron {
            A: SparseIntegerMatrix {
                rows,
                cols,
                vals,
                shape: dense.A.shape,
            },
            b: dense.b,
            columns: dense.columns,
            integer_columns: dense.integer_columns,
        }
    }
}

/// Represents a coefficient in a linear constraint: (variable_name, coefficient_value).
pub type Coefficient = (String, i64);

/// Maps node IDs to their bound values.
pub type Assignment = IndexMap<ID, Bound>;

/// Represents bounds for floating-point coefficient values.
pub type VBound = (f64, f64);

/// Combines integer bounds with floating-point coefficient bounds.
pub type MultiBound = (Bound, VBound);

/// Maps node IDs to their bounds and accumulated coefficients.
pub type ValuedAssignment = IndexMap<ID, MultiBound>;

pub struct Presolved {
    pub tightened: Pldag,  // the new graph
    pub fixed: Assignment, // id → (v,v)   (0..1 vars only)
}

#[derive(Clone, PartialEq, Debug, Hash)]
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
            (bound.0 + self.bias.0 >= 0) as i64,
            (bound.1 + self.bias.1 >= 0) as i64,
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

#[derive(Clone, PartialEq, Debug, Hash)]
/// Represents different types of boolean expressions in the DAG.
pub enum BoolExpression {
    /// A composite node representing a linear constraint
    Composite(Constraint),
    /// A primitive (leaf) node with a bound on its value
    Primitive(Bound),
}

/// Represents a node in the PL-DAG containing both logical expression and coefficient.
pub struct Node {
    /// The logical expression (either primitive or composite)
    pub expression: BoolExpression,
    /// Coefficient associated with this node for accumulation
    pub coefficient: f64,
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
    /// Map from node IDs to their corresponding nodes
    pub nodes: IndexMap<ID, Node>,
}

impl Hash for Pldag {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the nodes in a consistent order
        for (id, node) in self.nodes.iter().sorted_by(|a, b| a.0.cmp(b.0)) {
            id.hash(state);
            node.coefficient.to_bits().hash(state);
            match &node.expression {
                BoolExpression::Composite(c) => {
                    c.hash(state);
                }
                BoolExpression::Primitive(b) => {
                    b.hash(state);
                }
            }
        }
    }
}

impl Pldag {
    /// Creates a new empty PL-DAG.
    ///
    /// # Returns
    /// A new `Pldag` instance with no nodes
    pub fn new() -> Pldag {
        Pldag {
            nodes: IndexMap::new(),
        }
    }

    pub fn get_hash(&self) -> u64 {
        let mut state = DefaultHasher::new();
        self.hash(&mut state);
        state.finish()
    }

    fn get_coef_bounds(&self, composite: &Constraint) -> IndexMap<String, Bound> {
        let mut coef_bounds: IndexMap<String, Bound> = IndexMap::new();
        for (coef_key, _) in composite.coefficients.iter() {
            let coef_node = self
                .nodes
                .get(coef_key.as_str())
                .unwrap_or_else(|| panic!("Coefficient key '{}' not found in nodes", coef_key));
            match &coef_node.expression {
                BoolExpression::Primitive(bound) => {
                    coef_bounds.insert(coef_key.clone(), *bound);
                }
                _ => {
                    coef_bounds.insert(coef_key.clone(), (0, 1));
                }
            }
        }
        return coef_bounds;
    }

    /// Validates the PL-DAG structure.
    ///
    /// Checks that all composite nodes only reference existing child nodes.
    /// # Returns
    /// * **`Ok(())`** &nbsp; if the validation passes
    /// * **`Err(String)`** &nbsp; if any validation fails, with an error message
    pub fn validate(&self) -> Result<(), String> {
        for (key, node) in &self.nodes {
            // Check that each nodes references exists among the nodes
            if let BoolExpression::Composite(constraint) = &node.expression {
                for (child_id, _) in &constraint.coefficients {
                    if !self.nodes.contains_key(child_id) {
                        return Err(format!(
                            "Node '{}' references missing child node '{}'",
                            key, child_id
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Computes the transitive dependency closure for all nodes.
    ///
    /// For each node in the DAG, this method calculates all nodes that can be
    /// reached by following dependency edges (the set of all descendants).
    ///
    /// # Returns
    /// A map from each node ID to the set of all node IDs it transitively depends on
    pub fn transitive_dependencies(&self) -> HashMap<ID, HashSet<ID>> {
        // memo: node -> its already-computed reachables
        let mut memo: HashMap<String, HashSet<String>> = HashMap::new();
        let mut result: HashMap<String, HashSet<String>> = HashMap::new();

        for key in self.nodes.keys() {
            // compute (or fetch) and store
            let deps = self._collect_deps(key, &mut memo);
            result.insert(key.clone(), deps);
        }

        result
    }

    /// Helper function to recursively collect dependencies with memoization.
    ///
    /// # Arguments
    /// * `node` - The node to collect dependencies for
    /// * `memo` - Memoization cache to avoid recomputing dependencies
    ///
    /// # Returns
    /// The set of all nodes that the given node transitively depends on
    fn _collect_deps(&self, node: &ID, memo: &mut HashMap<ID, HashSet<ID>>) -> HashSet<ID> {
        // if we’ve already done this node, just clone the result
        if let Some(cached) = memo.get(node) {
            return cached.clone();
        }

        let mut deps = HashSet::new();

        if let Some(node_data) = self.nodes.get(node) {
            if let BoolExpression::Composite(constraint) = &node_data.expression {
                for (child_id, _) in &constraint.coefficients {
                    // direct edge
                    deps.insert(child_id.clone());
                    // transitive edges
                    let sub = self._collect_deps(child_id, memo);
                    deps.extend(sub);
                }
            }
            // if Primitive, we leave deps empty
        }

        // memoize before returning
        memo.insert(node.clone(), deps.clone());
        deps
    }

    /// child-id → parent-ids  (computed once)
    fn _build_parent_map(&self) -> IndexMap<String, Vec<String>> {
        let mut rev = IndexMap::<String, Vec<String>>::new();
        for (pid, node) in &self.nodes {
            if let BoolExpression::Composite(c) = &node.expression {
                for (cid, _) in &c.coefficients {
                    rev.entry(cid.clone()).or_default().push(pid.clone());
                }
            }
        }
        rev
    }

    /// Runs the **presolve / bound-propagation** phase on this `Pldag`
    /// **without mutating it** and returns a *new* graph in which:
    ///
    /// * every **primitive** variable carries the *tightest* interval that can be
    ///   proven from the current model and the optional `seed` assignments;
    /// * every **composite** node (linear row) whose truth-value collapses to
    ///   **always true** `(1, 1)` or **always false** `(0, 0)` is rewritten as a
    ///   primitive with that fixed bound;
    /// * all other composites are copied unchanged, but may reference the
    ///   tightened bounds of their primitives.
    ///
    /// The original DAG is left untouched so you can keep it for explanations
    /// or alternative presolve strategies.
    ///
    /// # Parameters
    ///
    /// * **`seed`** – an optional map `id → (min, max)` that fixes or tightens
    ///   the bounds of some primitives *before* propagation starts.  
    ///   For Boolean variables use `(0,0)` to force **false** and `(1,1)` to
    ///   force **true**.
    ///
    /// # Returns
    ///
    /// * **`Ok(Presolved)`** &nbsp;  
    ///   *`Presolved.tightened`* is the new, tightened `Pldag`.  
    ///   *`Presolved.fixed`* collects every node whose bound collapsed to a
    ///   single value `(v,v)` during presolve.  
    ///   Use `fixed` ⊕ `solve(tightened)` to reconstruct a full solution.
    ///
    /// * **`Err(String)`** &nbsp;  
    ///   The string is the **ID of the first composite row found infeasible**
    ///   (its upper bound fell below 0, i.e. the row is violated under *all*
    ///   possible assignments).
    ///
    /// # Algorithm (short version)
    ///
    /// 1. Build a **reverse adjacency** (child → parents) once.
    /// 2. Seed a FIFO queue with every node whose bounds are already fixed.
    /// 3. While the queue is not empty  
    ///    &nbsp;&nbsp;* re-evaluate each parent’s bound;  
    ///    &nbsp;&nbsp;* detect `(0,0)` ⇒ **unsat**;  
    ///    &nbsp;&nbsp;* if a parent’s bound tightens to `(v,v)` push it.  
    ///    This is classic *arc consistency* for ≥-constraints and runs in
    ///    **O(#edges)** time.
    /// 4. Re-build a fresh `Pldag` from the final bounds table.
    ///
    /// # Panics
    ///
    /// This function **never panics** on well-formed DAGs (acyclic and with
    /// every coefficient variable present in `self.nodes`). Malformed inputs
    /// may trigger `unwrap()` panics inside the implementation.
    pub fn presolve(&self, seed: &IndexMap<&str, Bound>) -> Result<Presolved, String> {
        // ------------ Phase 0: initial bounds table ------------------------
        let mut bounds: Assignment = seed.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        for (id, node) in &self.nodes {
            bounds
                .entry(id.clone())
                .or_insert_with(|| match node.expression {
                    BoolExpression::Primitive(b) => b,
                    _ => (0, 1), // composite starts open
                });
        }

        // ------------ Phase 1: propagate (same loop as previous answer) ----
        let parents = self._build_parent_map();
        let mut q: VecDeque<String> = bounds
            .iter()
            .filter(|(_, &b)| b.0 == b.1)
            .map(|(id, _)| id.clone())
            .collect();

        while let Some(cid) = q.pop_front() {
            if let Some(pvec) = parents.get(&cid) {
                for pid in pvec {
                    let BoolExpression::Composite(row) = &self.nodes[pid].expression else {
                        continue;
                    };
                    let new = row.evaluate(&bounds);

                    if new == (0, 0) {
                        // UNSAT row
                        return Err(pid.clone());
                    }
                    let ent = bounds.entry(pid.clone()).or_insert(new);
                    if *ent != new {
                        *ent = new;
                        if new.0 == new.1 {
                            // became fixed
                            q.push_back(pid.clone());
                        }
                    }
                }
            }
        }

        // ------------ Phase 2: build the tightened DAG ---------------------
        let mut tightened_nodes = IndexMap::<String, Node>::new();
        let mut fixed_map = Assignment::new();

        for (id, node) in &self.nodes {
            let b = bounds[id];
            if b.0 == b.1 {
                // fixed variable / row
                fixed_map.insert(id.clone(), b);

                // Represent it as a primitive with that bound.
                tightened_nodes.insert(
                    id.clone(),
                    Node {
                        expression: BoolExpression::Primitive(b),
                        coefficient: node.coefficient,
                    },
                );
            } else {
                // Unfixed: keep original but *write back* tightened bounds
                // for primitives so later phases know their narrowed range.
                let new_expr = match &node.expression {
                    BoolExpression::Primitive(_) => BoolExpression::Primitive(b),
                    other => other.clone(),
                };
                tightened_nodes.insert(
                    id.clone(),
                    Node {
                        expression: new_expr,
                        coefficient: node.coefficient,
                    },
                );
            }
        }

        Ok(Presolved {
            tightened: Pldag {
                nodes: tightened_nodes,
            },
            fixed: fixed_map,
        })
    }

    /// Generates all possible combinations of primitive variable assignments.
    ///
    /// Enumerates the Cartesian product of all primitive variable bounds,
    /// yielding every possible complete assignment of values to primitive variables.
    ///
    /// # Returns
    /// An iterator over all possible variable assignments as HashMaps
    pub fn primitive_combinations(&self) -> impl Iterator<Item = HashMap<ID, i64>> {
        // 1. Pull out [(var_name, (low, high)), …]
        let primitives: Vec<(String, (i64, i64))> = self
            .nodes
            .iter()
            .filter_map(|(key, node)| {
                if let BoolExpression::Primitive(bound) = &node.expression {
                    Some((key.clone(), *bound))
                } else {
                    None
                }
            })
            .collect();

        // 2. Extract names in order
        let keys: Vec<String> = primitives.iter().map(|(k, _)| k.clone()).collect();

        // 3. Turn each bound into an inclusive range  low..=high
        let ranges: Vec<RangeInclusive<i64>> = primitives
            .iter()
            .map(|(_, (low, high))| *low..=*high)
            .collect();

        // 4. For each range, collect it into a Vec<i64>, then do the cartesian product
        ranges
            .into_iter()
            .map(|r| r.collect::<Vec<_>>())
            .multi_cartesian_product()
            // 5. Zip back with keys to build each assignment map
            .map(move |values| {
                keys.iter()
                    .cloned()
                    .zip(values.into_iter())
                    .collect::<HashMap<String, i64>>()
            })
    }

    /// Propagates bounds through the DAG bottom-up.
    ///
    /// Starting from the given variable assignments, this method computes bounds
    /// for all composite nodes by propagating constraints upward through the DAG.
    ///
    /// # Arguments
    /// * `assignment` - Initial assignment of bounds to variables
    ///
    /// # Returns
    /// Complete assignment including bounds for all reachable nodes
    pub fn propagate(&self, assignment: &IndexMap<&str, Bound>) -> Assignment {
        // Convert &str keys to String keys for internal storage
        let mut result: Assignment = assignment
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        // Fill result with the primitive variable bounds
        for (key, node) in self.nodes.iter() {
            if !result.contains_key(key) {
                if let BoolExpression::Primitive(bound) = &node.expression {
                    result.insert(key.clone(), *bound);
                }
            }
        }

        // S = All composites that
        // (1) have only primitive variables as input and
        // (2) are not present in `result/assignments`
        let mut S: VecDeque<String> = self
            .nodes
            .iter()
            .filter(|(key, node)| match &node.expression {
                BoolExpression::Composite(composite) => {
                    composite
                        .coefficients
                        .iter()
                        .all(|x| match self.nodes.get(&x.0) {
                            Some(node_data) => {
                                matches!(node_data.expression, BoolExpression::Primitive(_))
                            }
                            _ => false,
                        })
                        && !result.contains_key(key.as_str())
                }
                BoolExpression::Primitive(_) => false,
            })
            .map(|(key, _)| key.clone())
            .collect();

        // Add a set to keep track of the visited nodes
        // This is to avoid infinite loops in case of cycles
        // in the graph
        let mut visited = HashSet::new();

        // Iterate until S is empty
        while let Some(s) = S.pop_front() {
            // If we have already visited this node, skip it
            if visited.contains(&s) {
                panic!("Cycle detected in the graph");
            }

            // Mark the current node as visited
            visited.insert(s.clone());

            // Start by propagating the bounds from current_id's children
            // to current_id
            match self.nodes.get(&s) {
                Some(node_data) => {
                    if let BoolExpression::Composite(composite) = &node_data.expression {
                        result.insert(s.clone(), composite.evaluate(&result));

                        // Now find all composites having `current_id` as its input.
                        // And if that composite's children are in `result`, then add to S.
                        let incoming = self
                            .nodes
                            .iter()
                            .filter(|(key, node)| {
                                !result.contains_key(key.as_str())
                                    && match &node.expression {
                                        BoolExpression::Composite(sub_composite) => {
                                            sub_composite.coefficients.iter().any(|x| x.0 == s)
                                        }
                                        _ => false,
                                    }
                            })
                            .map(|(key, _)| key.clone())
                            .collect::<Vec<String>>();

                        // Add the incoming composites to S without any checks
                        for incoming_id in incoming {
                            if !S.contains(&incoming_id) {
                                S.push_back(incoming_id);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        return result;
    }

    /// Propagates bounds using default primitive variable bounds.
    ///
    /// Convenience method that calls `propagate` with the default bounds
    /// of all primitive variables as defined in the DAG.
    ///
    /// # Returns
    /// Complete assignment with bounds for all nodes
    pub fn propagate_default(&self) -> Assignment {
        let assignments: IndexMap<&str, Bound> = self
            .nodes
            .iter()
            .filter_map(|(key, node)| {
                if let BoolExpression::Primitive(bound) = &node.expression {
                    Some((key.as_str(), *bound))
                } else {
                    None
                }
            })
            .collect();
        self.propagate(&assignments)
    }

    /// Propagates bounds and accumulates coefficients for multiple assignments.
    ///
    /// For each input assignment, this method:
    /// 1. Propagates bounds through the DAG
    /// 2. Accumulates coefficients where each parent's coefficient is the sum
    ///    of its children's coefficients plus its own coefficient
    ///
    /// # Arguments
    /// * `assignments` - Vector of assignments to process
    ///
    /// # Returns
    /// Vector of valued assignments, each containing bounds and accumulated coefficients
    pub fn propagate_many_coefs(
        &self,
        assignments: Vec<&IndexMap<&str, Bound>>,
    ) -> Vec<ValuedAssignment> {
        // Calculate transitive dependencies
        let transitive_deps = self.transitive_dependencies();

        let mut assignment_results: Vec<ValuedAssignment> = Vec::new();

        for assignment in assignments {
            // Start with the propagated bounds
            let result = self.propagate(assignment);

            let mut valued_assigment: ValuedAssignment = IndexMap::new();
            // Calculate the accumulated coefficients with the new bound result
            for (key, deps) in transitive_deps.iter() {
                let mut coef_sum: VBound = (0.0, 0.0);
                if let Some(node) = self.nodes.get(key) {
                    coef_sum = (node.coefficient, node.coefficient);
                }
                for dep in deps {
                    let dep_res = result.get(dep).unwrap_or(&(0, 1));
                    if let Some(node) = self.nodes.get(dep) {
                        if node.coefficient < 0.0 {
                            coef_sum.0 += node.coefficient * dep_res.1 as f64;
                            coef_sum.1 += node.coefficient * dep_res.0 as f64;
                        } else {
                            coef_sum.0 += node.coefficient * dep_res.0 as f64;
                            coef_sum.1 += node.coefficient * dep_res.1 as f64;
                        }
                    }
                }
                if let Some(bound) = result.get(key) {
                    valued_assigment.insert(key.clone(), (*bound, coef_sum));
                }
            }

            // Return the accumulated coefficients
            assignment_results.push(valued_assigment);
        }

        return assignment_results;
    }

    /// Propagates bounds and accumulates coefficients for a single assignment.
    ///
    /// Convenience method that calls `propagate_many_coefs` with a single assignment.
    ///
    /// # Arguments
    /// * `assignment` - The assignment to propagate
    ///
    /// # Returns
    /// A valued assignment containing bounds and accumulated coefficients for all nodes
    ///
    /// # Panics
    /// Panics if no assignments are returned (should not happen under normal circumstances)
    pub fn propagate_coefs(&self, assignment: &IndexMap<&str, Bound>) -> ValuedAssignment {
        // Propagate the assignment through the Pldag
        return self
            .propagate_many_coefs(vec![assignment])
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                panic!(
                    "No assignments found after propagation with {:?}",
                    assignment
                )
            });
    }

    /// Propagates bounds and coefficients using default primitive bounds.
    ///
    /// Convenience method that calls `propagate_coefs` with the default bounds
    /// of all primitive variables.
    ///
    /// # Returns
    /// A valued assignment with bounds and coefficients for all nodes
    pub fn propagate_coefs_default(&self) -> ValuedAssignment {
        // Propagate the default assignment through the Pldag
        let assignments: IndexMap<&str, Bound> = self
            .nodes
            .iter()
            .filter_map(|(key, node)| {
                if let BoolExpression::Primitive(bound) = &node.expression {
                    Some((key.as_str(), *bound))
                } else {
                    None
                }
            })
            .collect();
        self.propagate_coefs(&assignments)
    }

    #[cfg(feature = "glpk")]
    /// Solve the supplied objectives in-process with GLPK.
    /// Only available when the crate is compiled with `--features glpk`
    ///
    /// # Arguments
    /// * `objectives` - Vector of ID to value mapping representing different objective functions to solve
    /// * `assume` - Fixed variable assignments to apply before solving
    /// * `maximize` - If true, maximizes the objective; if false, minimizes it
    ///
    /// # Returns
    /// Vector of optional valued assignments, one for each objective. None if infeasible.
    pub fn solve(
        &self,
        objectives: Vec<HashMap<&str, f64>>,
        assume: HashMap<&str, Bound>,
        maximize: bool,
    ) -> Result<Vec<Option<Assignment>>, String> {
        use glpk_rust::{
            solve_ilps, IntegerSparseMatrix, Solution, SparseLEIntegerPolyhedron, Status, Variable,
        };

        // Convert the PL-DAG to a polyhedron representation
        let polyhedron = self.to_sparse_polyhedron(true, true, true)?;

        // Check that all assumptions are valid
        for (key, bound) in &assume {
            if let Some(variable) = self.nodes.get(&key.to_string()) {
                match &variable.expression {
                    BoolExpression::Primitive(b) => {
                        // Check that the assumption is within the bounds of the variable
                        if bound.0 < b.0 || bound.1 > b.1 {
                            // Return infeasible solution for all objectives if any assumption is out of bounds
                            return Ok(vec![None; objectives.len()]);
                        }
                    }
                    BoolExpression::Composite(_) => {
                        // Composite nodes do not have bounds, so we assume (0, 1) by default
                        if bound.0 < 0 || bound.1 > 1 {
                            // Return infeasible solution for all objectives if any assumption is out of bounds
                            return Ok(vec![None; objectives.len()]);
                        }
                    } // Default for composite nodes
                };
            }
        }

        // Convert sparse matrix to the format expected by glpk-rust
        // NOTE: As soon as the polyhedron is made, the order of the columns are vital.
        // Therefore always use polyhedron.columns to get the variable names in the correct order.
        let mut glpk_matrix = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: polyhedron.A.rows.iter().map(|&x| x as i32).collect(),
                cols: polyhedron.A.cols.iter().map(|&x| x as i32).collect(),
                vals: polyhedron.A.vals.iter().map(|&x| -1 * x as i32).collect(),
            },
            b: polyhedron.b.iter().map(|&x| (0, -1 * x as i32)).collect(),
            variables: polyhedron
                .columns
                .iter()
                .map(|key| {
                    let node = self.nodes.get(key).unwrap();
                    Variable {
                        id: key.as_str(),
                        bound: match &node.expression {
                            BoolExpression::Primitive(bound) => {
                                let bound_to_use = assume.get(key.as_str()).unwrap_or(bound);
                                (bound_to_use.0 as i32, bound_to_use.1 as i32)
                            }
                            BoolExpression::Composite(_) => {
                                let bound_to_use = assume.get(key.as_str()).unwrap_or(&(0, 1));
                                (bound_to_use.0 as i32, bound_to_use.1 as i32)
                            }
                        },
                    }
                })
                .collect(),
            double_bound: false,
        };

        // If there are no constraints, insert a dummy row
        if glpk_matrix.A.rows.is_empty() {
            for i in 0..self.nodes.len() {
                glpk_matrix.A.rows.push(i as i32);
                glpk_matrix.A.cols.push(i as i32);
                glpk_matrix.A.vals.push(0);
            }
            glpk_matrix.b.push((0, 0));
        }

        let solutions: Vec<Solution> = solve_ilps(&mut glpk_matrix, objectives, maximize, false);

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

    /// Retrieves the objective function coefficients from all primitive nodes.
    ///
    /// Collects the coefficients of all primitive (leaf) nodes in the DAG,
    /// which represent the objective function values for ILP optimization.
    /// Composite nodes are excluded from the result.
    ///
    /// # Returns
    /// A map from primitive variable names to their objective coefficients
    pub fn get_objective(&self) -> IndexMap<String, f64> {
        // Collect objective coefficients from primitive nodes
        self.nodes
            .iter()
            .filter_map(|(key, node)| {
                if let BoolExpression::Primitive(_) = &node.expression {
                    Some((key.clone(), node.coefficient))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Converts the PL-DAG to a sparse polyhedron for ILP solving.
    ///
    /// Transforms the logical constraints in the DAG into a system of linear
    /// inequalities suitable for integer linear programming solvers.
    ///
    /// # Arguments
    /// * `double_binding` - If true, creates bidirectional implications for composite nodes
    /// * `integer_constraints` - If true, adds bounds constraints for integer variables
    /// * `fixed_constraints` - If true, adds equality constraints for fixed primitive variables
    ///
    /// # Returns
    /// A `SparsePolyhedron` representing the DAG constraints
    pub fn to_sparse_polyhedron(
        &self,
        double_binding: bool,
        integer_constraints: bool,
        fixed_constraints: bool,
    ) -> Result<SparsePolyhedron, String> {
        // Validate the PL-DAG before conversion
        self.validate()?;

        // Create a new sparse matrix
        let mut A_matrix = SparseIntegerMatrix::new();
        let mut b_vector: Vec<i64> = Vec::new();

        // Filter out all BoolExpressions that are primitives
        let primitives: IndexMap<&String, Bound> = self
            .nodes
            .iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .filter_map(|(key, node)| {
                if let BoolExpression::Primitive(bound) = &node.expression {
                    Some((key, *bound))
                } else {
                    None
                }
            })
            .collect();

        // Filter out all BoolExpressions that are composites
        let composites: IndexMap<&String, &Constraint> = self
            .nodes
            .iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .filter_map(|(key, node)| {
                if let BoolExpression::Composite(constraint) = &node.expression {
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

            let coef_bounds = self.get_coef_bounds(composite);
            // An inner bound $\text{ib}(\phi)$ of a
            // linear inequality constraint $\phi$ is the sum of all variable's
            // step bounds, excl bias and before evaluated with the $\geq$ operator.
            // For instance, the inner bound of the linear inequality $-2x + y + z \geq 0$,
            // where $x,y,z \in \{0,1\}^3$, is $[-2, 2]$, since the lowest value the
            // sum can be is $-2$ (given from the combination $x=1, y=0, z=0$) and the
            // highest value is $2$ (from $x=0, y=1, z=1$).
            let ib_phi = composite.dot(&coef_bounds);

            // 1. Let $d = max(|ib(phi)|)$.
            let d_pi = std::cmp::max(ib_phi.0.abs(), ib_phi.1.abs());

            // Push values for pi variable coefficient
            A_matrix.rows.push(row_i);
            A_matrix.cols.push(ki);
            A_matrix.vals.push(-d_pi);

            // Push values for phi coefficients
            for (coef_key, coef_val) in composite.coefficients.iter() {
                let ck_index: usize = *column_names_map.get(coef_key).unwrap();
                A_matrix.rows.push(row_i);
                A_matrix.cols.push(ck_index);
                A_matrix.vals.push(*coef_val);
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
                A_matrix.rows.push(row_i + 1);
                A_matrix.cols.push(ki);
                A_matrix.vals.push(pi_coef);

                // Push values for phi coefficients
                for (phi_coef_key, phi_coef_val) in phi_prim.coefficients.iter() {
                    let ck_index: usize = *column_names_map.get(phi_coef_key).unwrap();
                    A_matrix.rows.push(row_i + 1);
                    A_matrix.cols.push(ck_index);
                    A_matrix.vals.push(*phi_coef_val);
                }

                // Push bias value
                b_vector.push(-1 * phi_prim.bias.0);

                // Increment one more row when double binding
                row_i += 1;
            }

            // Increment the row index
            row_i += 1;
        }

        if fixed_constraints {
            // Add the bounds for the primitive variables that are fixed.
            // We start by creating a grouping on the lower and upper bounds of the primitive variables
            let mut fixed_bound_map: IndexMap<i64, Vec<usize>> = IndexMap::new();
            for (key, bound) in primitives.iter().filter(|(_, bound)| bound_fixed(**bound)) {
                fixed_bound_map
                    .entry(bound.0)
                    .or_insert_with(Vec::new)
                    .push(*column_names_map.get(key.as_str()).unwrap());
            }

            for (v, primitive_ids) in fixed_bound_map.iter() {
                let b = *v * primitive_ids.len() as i64;
                for i in vec![-1, 1] {
                    for primitive_id in primitive_ids {
                        A_matrix.rows.push(row_i);
                        A_matrix.cols.push(*primitive_id);
                        A_matrix.vals.push(i);
                    }
                    b_vector.push(i * b);
                    row_i += 1;
                }
            }
        }

        // Collect all integer variables
        let mut integer_variables: Vec<String> = Vec::new();

        // Restrain integer bounds
        for (p_key, p_bound) in primitives
            .iter()
            .filter(|(_, bound)| bound.0 < 0 || bound.1 > 1)
        {
            // Add the variable to the integer variables list
            integer_variables.push((*p_key).clone());

            if integer_constraints {
                // Get the index of the current key
                let pi = *column_names_map.get(p_key.as_str()).unwrap();

                A_matrix.rows.push(row_i);
                A_matrix.cols.push(pi);
                A_matrix.vals.push(1);
                b_vector.push(p_bound.0);
                row_i += 1;

                A_matrix.rows.push(row_i);
                A_matrix.cols.push(pi);
                A_matrix.vals.push(-1);
                b_vector.push(-1 * p_bound.1);
                row_i += 1;
            }
        }

        // Set the shape of the A matrix
        A_matrix.shape = (row_i, column_names_map.len());

        // Create the polyhedron
        let polyhedron = SparsePolyhedron {
            A: A_matrix,
            b: b_vector,
            columns: column_names_map.keys().cloned().collect(),
            integer_columns: integer_variables,
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
    pub fn to_sparse_polyhedron_default(&self) -> Result<SparsePolyhedron, String> {
        self.to_sparse_polyhedron(true, true, true)
    }

    /// Converts the PL-DAG to a dense polyhedron.
    ///
    /// # Arguments
    /// * `double_binding` - If true, creates bidirectional implications
    /// * `integer_constraints` - If true, adds integer bounds constraints
    /// * `fixed_constraints` - If true, adds fixed variable constraints
    ///
    /// # Returns
    /// A `DensePolyhedron` representing the DAG constraints
    pub fn to_dense_polyhedron(
        &self,
        double_binding: bool,
        integer_constraints: bool,
        fixed_constraints: bool,
    ) -> Result<DensePolyhedron, String> {
        // Convert to sparse polyhedron first
        let sparse_polyhedron =
            self.to_sparse_polyhedron(double_binding, integer_constraints, fixed_constraints)?;
        // Convert sparse to dense polyhedron
        Ok(sparse_polyhedron.into())
    }

    /// Converts the PL-DAG to a dense polyhedron with default settings.
    ///
    /// # Returns
    /// A `DensePolyhedron` with all constraint options enabled
    pub fn to_dense_polyhedron_default(&self) -> Result<DensePolyhedron, String> {
        self.to_dense_polyhedron(true, true, true)
    }

    /// Sets the coefficient for a specific node.
    ///
    /// Updates the coefficient value associated with the given node ID.
    /// If the node doesn't exist, this method has no effect.
    ///
    /// # Arguments
    /// * `id` - The node ID to update
    /// * `coefficient` - The new coefficient value
    pub fn set_coef(&mut self, id: &str, coefficient: f64) -> bool {
        // Update the coefficient for the given node
        if let Some(node) = self.nodes.get_mut(id) {
            node.coefficient = coefficient;
            return true;
        }
        false
    }

    /// Retrieves the coefficient for a specific node.
    ///
    /// # Arguments
    /// * `id` - The node ID to query
    ///
    /// # Returns
    /// The coefficient value for the node, or 0.0 if the node doesn't exist
    pub fn get_coef(&self, id: &str) -> f64 {
        // Get the coefficient for the given node
        self.nodes
            .get(id)
            .map(|node| node.coefficient)
            .unwrap_or(0.0)
    }

    /// Creates a primitive (leaf) variable with the specified bounds.
    ///
    /// Primitive variables represent the base variables in the DAG and have
    /// no dependencies on other nodes.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the variable
    /// * `bound` - The allowed range (min, max) for this variable
    pub fn set_primitive(&mut self, id: &str, bound: Bound) {
        // Insert the primitive variable as a node
        self.nodes.insert(
            id.to_string(),
            Node {
                expression: BoolExpression::Primitive(bound),
                coefficient: 0.0,
            },
        );
    }

    /// Creates multiple primitive variables with the same bounds.
    ///
    /// Convenience method to create several primitive variables at once.
    /// Duplicate IDs are automatically filtered out.
    ///
    /// # Arguments
    /// * `ids` - Vector of unique identifiers for the variables
    /// * `bound` - The common bound to apply to all variables
    pub fn set_primitives(&mut self, ids: Vec<&str>, bound: Bound) {
        let unique_ids: IndexSet<_> = ids.into_iter().collect();
        for id in unique_ids {
            self.set_primitive(id, bound);
        }
    }

    /// Creates a general linear inequality constraint.
    ///
    /// Creates a constraint of the form: sum(coeff_i * var_i) + bias >= 0.
    /// The constraint is automatically assigned a unique ID based on its content.
    ///
    /// # Arguments
    /// * `coefficient_variables` - Vector of (variable_id, coefficient) pairs
    /// * `bias` - Constant bias term
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_gelineq<'a>(
        &mut self,
        coefficient_variables: impl IntoIterator<Item = (&'a str, i64)>,
        bias: i64,
    ) -> ID {
        // Ensure coefficients have unique keys by summing duplicate values
        let mut unique_coefficients: IndexMap<ID, i64> = IndexMap::new();
        for (key, value) in coefficient_variables {
            *unique_coefficients.entry(key.to_string()).or_insert(0) += value;
        }
        let coefficient_variables: Vec<Coefficient> = unique_coefficients
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .collect();

        // Create a hash from the input data
        let hash = create_hash(&coefficient_variables, bias);

        // Return the hash as a string
        let id = hash;

        let constraint = Constraint {
            coefficients: coefficient_variables,
            bias: (bias, bias),
        };

        // If the composite is either a contradiction or tautology, we create a primitive instead
        // and fix it to 0 or 1 respectively.
        let coef_bounds = self.get_coef_bounds(&constraint);
        let ib_phi = constraint.dot(&coef_bounds);
        if (ib_phi.0 + bias) >= 0 {
            // Tautology: always true
            self.nodes.insert(
                id.to_string(),
                Node {
                    expression: BoolExpression::Primitive((1, 1)),
                    coefficient: 0.0,
                },
            );
            return id.to_string();
        } else if (ib_phi.1 + bias) < 0 {
            // Contradiction: always false
            self.nodes.insert(
                id.to_string(),
                Node {
                    expression: BoolExpression::Primitive((0, 0)),
                    coefficient: 0.0,
                },
            );
            return id.to_string();
        } else {
            // Insert the constraint as a node
            self.nodes.insert(
                id.to_string(),
                Node {
                    expression: BoolExpression::Composite(constraint),
                    coefficient: 0.0,
                },
            );
        }

        id.to_string()
    }

    /// Creates an "at least" constraint: sum(variables) >= value.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to sum
    /// * `value` - Minimum required sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_atleast<'a>(&mut self, references: impl IntoIterator<Item = &'a str>, value: i64) -> ID {
        self.set_gelineq(references.into_iter().map(|x| (x, 1)), -value)
    }

    pub fn set_atleast_ref<'a>(
        &mut self,
        references: impl IntoIterator<Item = &'a str>,
        value: &str,
    ) -> ID {
        self.set_gelineq(
            references.into_iter().map(|x| (x, 1)).chain([(value, -1)]),
            0,
        )
    }

    /// Creates an "at most" constraint: sum(variables) <= value.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to sum
    /// * `value` - Maximum allowed sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_atmost<'a>(&mut self, references: impl IntoIterator<Item = &'a str>, value: i64) -> ID {
        self.set_gelineq(references.into_iter().map(|x| (x, -1)), value)
    }

    pub fn set_atmost_ref<'a>(
        &mut self,
        references: impl IntoIterator<Item = &'a str>,
        value: &str,
    ) -> ID {
        self.set_gelineq(
            references.into_iter().map(|x| (x, -1)).chain([(value, 1)]),
            0,
        )
    }

    /// Creates an equality constraint: sum(variables) == value.
    ///
    /// Implemented as the conjunction of "at least" and "at most" constraints.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to sum
    /// * `value` - Required exact sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_equal<'a >(&mut self, references: impl IntoIterator<Item = &'a str> + Clone, value: i64) -> ID {
        let ub = self.set_atleast(references.clone(), value);
        let lb = self.set_atmost(references, value);
        self.set_and(vec![ub, lb])
    }

    pub fn set_equal_ref<'a >(&mut self, references: impl IntoIterator<Item = &'a str> + Clone, value: &str) -> ID {
        let ub = self.set_atleast_ref(references.clone(), value);
        let lb = self.set_atmost_ref(references, value);
        self.set_and(vec![ub, lb])
    }

    /// Creates a logical AND constraint.
    ///
    /// Returns true if and only if ALL referenced variables are true.
    /// Implemented as: sum(variables) >= count(variables).
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to AND together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_and<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        let length = unique_references.len();
        self.set_atleast(
            unique_references.iter().map(|x| x.as_str()),
            length as i64,
        )
    }

    /// Creates a logical OR constraint.
    ///
    /// Returns true if AT LEAST ONE of the referenced variables is true.
    /// Implemented as: sum(variables) >= 1.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to OR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_or<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        self.set_atleast(unique_references.iter().map(|x| x.as_str()), 1)
    }

    /// Creates a logical NAND constraint.
    ///
    /// Returns true if NOT ALL of the referenced variables are true.
    /// Implemented as: sum(variables) <= count(variables) - 1.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to NAND together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_nand<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        let length = unique_references.len();
        self.set_atmost(
            unique_references.iter().map(|x| x.as_str()),
            length as i64 - 1,
        )
    }

    /// Creates a logical NOR constraint.
    ///
    /// Returns true if NONE of the referenced variables are true.
    /// Implemented as: sum(variables) <= 0.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to NOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_nor<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0)
    }

    /// Creates a logical NOT constraint.
    ///
    /// Returns true if NONE of the referenced variables are true.
    /// Functionally equivalent to NOR. Implemented as: sum(variables) <= 0.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to negate
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_not<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0)
    }

    /// Creates a logical XOR constraint.
    ///
    /// Returns true if EXACTLY ONE of the referenced variables is true.
    /// Implemented as the conjunction of OR and "at most 1" constraints.
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to XOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_xor<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        let atleast = self.set_or(unique_references.iter().map(|x| x.as_str()).collect());
        let atmost = self.set_atmost(unique_references.iter().map(|x| x.as_str()), 1);
        self.set_and(vec![atleast, atmost])
    }

    /// Creates a logical XNOR constraint.
    ///
    /// Returns true if an EVEN NUMBER of the referenced variables are true
    /// (including zero). Implemented as: (sum >= 2) OR (sum <= 0).
    ///
    /// # Arguments
    /// * `references` - Vector of variable IDs to XNOR together
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_xnor<T>(&mut self, references: Vec<T>) -> ID
    where
        T: Into<String>,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.into()).collect();
        let atleast = self.set_atleast(unique_references.iter().map(|x| x.as_str()), 2);
        let atmost = self.set_atmost(unique_references.iter().map(|x| x.as_str()), 0);
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_imply<T, U>(&mut self, condition: T, consequence: U) -> ID
    where
        T: Into<String>,
        U: Into<String>,
    {
        let not_condition = self.set_not(vec![condition.into()]);
        self.set_or(vec![not_condition, consequence.into()])
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_equiv<T, U>(&mut self, lhs: T, rhs: U) -> ID
    where
        T: Into<String> + Clone,
        U: Into<String> + Clone,
    {
        // Convert to strings first to avoid type mismatches
        let lhs_str: String = lhs.into();
        let rhs_str: String = rhs.into();

        let imply_lr = self.set_and(vec![lhs_str.clone(), rhs_str.clone()]);
        let imply_rl = self.set_not(vec![rhs_str, lhs_str]);
        self.set_or(vec![imply_lr, imply_rl])
    }

    pub fn set_bounds(&mut self, id: &str, bound: Bound) -> bool {
        if let Some(node) = self.nodes.get_mut(id) {
            if let BoolExpression::Primitive(b) = &mut node.expression {
                *b = bound;
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: for every primitive combination,
    ///   1) run `propagate` on the PLDAG model  
    ///   2) build the corresponding assignments  
    ///   3) run `assume(root=1)` on the polyhedron  
    ///   4) evaluate the shrunken polyhedron on the same assignments  
    ///   5) assert they agree at `root`.
    fn evaluate_model_polyhedron(model: &Pldag, poly: &DensePolyhedron, root: &String) {
        for combo in model.primitive_combinations() {
            // build an IndexMap<&str,Bound> as propagate expects
            let interp = combo
                .iter()
                .map(|(k, &v)| (k.as_str(), (v, v)))
                .collect::<IndexMap<&str, Bound>>();

            // what the DAG says the root can be
            let prop = model.propagate(&interp);
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

    #[test]
    fn test_propagate() {
        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]);

        let result = model.propagate(&IndexMap::new());
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        let result = model.propagate(&assignments);
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let result = model.propagate(&IndexMap::new());
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(&assignments);
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(&assignments);
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 0));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (0, 0));
        let result = model.propagate(&assignments);
        assert_eq!(result.get(&root).unwrap(), &(1, 1));
    }

    /// XOR already covered; test the OR gate
    #[test]
    fn test_propagate_or_gate() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let or_root = model.set_or(vec!["a", "b"]);

        // No assignment: both inputs full [0,1], output [0,1]
        let res = model.propagate(&IndexMap::new());
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = IndexMap::new();
        interp.insert("a".into(), (1, 1));
        let res = model.propagate(&interp);
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = IndexMap::new();
        interp.insert("a".into(), (0, 0));
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(&interp);
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = IndexMap::new();
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(&interp);
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[test]
    fn test_propagate_not_gate() {
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        let not_root = model.set_not(vec!["p"]);

        // no assignment ⇒ [0,1]
        let res = model.propagate(&IndexMap::new());
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = IndexMap::new();
        interp.insert("p".into(), (0, 0));
        let res = model.propagate(&interp);
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = IndexMap::new();
        interp.insert("p".into(), (1, 1));
        let res = model.propagate(&interp);
        assert_eq!(res[&not_root], (0, 0));
    }

    #[test]
    fn test_to_polyhedron_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        let root = m.set_and(vec!["x", "y"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_or() {
        let mut m = Pldag::new();
        m.set_primitive("a".into(), (0, 1));
        m.set_primitive("b".into(), (0, 1));
        m.set_primitive("c".into(), (0, 1));
        let root = m.set_or(vec!["a", "b", "c"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_not() {
        let mut m = Pldag::new();
        m.set_primitive("p".into(), (0, 1));
        let root = m.set_not(vec!["p"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_xor() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        m.set_primitive("z".into(), (0, 1));
        let root = m.set_xor(vec!["x", "y", "z"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
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

        let w = m.set_and(vec!["x", "y"]);
        let nz = m.set_not(vec!["z"]);
        let v = m.set_or(vec![w.clone(), nz.clone()]);

        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
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

        let w = model.set_and(vec!["x", "y"]);
        let v = model.set_xor(vec![w.clone(), "z".into()]);

        // no assignment: everything [0,1]
        let res = model.propagate(&IndexMap::new());
        for var in &["x", "y", "z"] {
            assert_eq!(res[*var], (0, 1), "{}", var);
        }
        assert_eq!(res[&w], (0, 1));
        assert_eq!(res[&v], (0, 1));

        // x=1,y=1,z=0 ⇒ w=1,v=1
        let mut interp = IndexMap::new();
        interp.insert("x", (1, 1));
        interp.insert("y", (1, 1));
        interp.insert("z", (0, 0));
        let res = model.propagate(&interp);
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = IndexMap::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (1, 1));
        interp.insert("z", (1, 1));
        let res = model.propagate(&interp);
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = IndexMap::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (0, 0));
        interp.insert("z", (0, 0));
        let res = model.propagate(&interp);
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (0, 0));
    }

    /// If you ever get an inconsistent assignment (out‐of‐bounds for a primitive),
    /// propagate should leave it as given (or you could choose to clamp / panic)—here
    /// we simply check that nothing blows up.
    #[test]
    fn test_propagate_out_of_bounds_does_not_crash() {
        let mut model = Pldag::new();
        model.set_primitive("u".into(), (0, 1));
        let root = model.set_not(vec!["u"]);

        let mut interp = IndexMap::new();
        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        interp.insert("u".into(), (5, 5));
        let res = model.propagate(&interp);

        // we expect propagate to return exactly (5,5) for "u" and compute root = negate(5)
        assert_eq!(res["u"], (5, 5));
        // Depending on your semantic for negate, it might be
        //   bound_multiply(-1,(5,5)) + bias
        // so just check it didn’t panic:
        let _ = res[&root];
    }

    #[test]
    fn test_to_polyhedron() {
        fn evaluate_model_polyhedron(model: &Pldag, polyhedron: &DensePolyhedron, root: &String) {
            for combination in model.primitive_combinations() {
                let assignments = combination
                    .iter()
                    .map(|(k, &v)| (k.as_str(), (v, v)))
                    .collect::<IndexMap<&str, Bound>>();
                let model_prop = model.propagate(&assignments);
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
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let polyhedron: DensePolyhedron = model.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]);
        let polyhedron = model.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let polyhedron = model.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);
    }

    /// Single‐operand composites should act as identity: root == operand
    #[test]
    fn test_to_polyhedron_single_operand_identity() {
        // AND(x) == x
        {
            let mut m = Pldag::new();
            m.set_primitive("x".into(), (0, 1));
            let root = m.set_and::<&str>(vec!["x"]);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // OR(y) == y
        {
            let mut m = Pldag::new();
            m.set_primitive("y", (0, 1));
            let root = m.set_or(vec!["y"]);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // XOR(z) == z
        {
            let mut m = Pldag::new();
            m.set_primitive("z".into(), (0, 1));
            let root = m.set_xor(vec!["z"]);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[test]
    fn test_to_polyhedron_duplicate_operands_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        let root = m.set_and(vec!["x", "x"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
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

        let w1 = m.set_and(vec![a, b]);
        let w2 = m.set_or(vec![w1.clone(), c.to_string()]);
        let w3 = m.set_xor(vec![w2.clone(), d.to_string()]);
        let root = m.set_not(vec![w3.clone()]);

        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn make_simple_dag() {
        let mut pldag = Pldag::new();
        pldag.set_primitive("b".into(), (0, 1));
        pldag.set_primitive("d".into(), (0, 1));
        pldag.set_primitive("e".into(), (0, 1));
        let c = pldag.set_or(vec!["d", "e"]);
        let a = pldag.set_or(vec!["b".into(), c.to_string()]);
        let deps = pldag.transitive_dependencies();
        let expect = |xs: &[&str]| xs.iter().cloned().map(String::from).collect::<HashSet<_>>();
        assert_eq!(deps.get(&a), Some(&expect(&["b", &c, "d", "e"])));
        assert_eq!(deps.get(&c), Some(&expect(&["d", "e"])));
        assert_eq!(deps.get("b"), Some(&expect(&[])));
        assert_eq!(deps.get("d"), Some(&expect(&[])));
        assert_eq!(deps.get("e"), Some(&expect(&[])));
    }

    #[test]
    fn test_chain_dag() {
        // x → [y], y → [z], z → []
        let mut pldag = Pldag::new();
        pldag.set_primitive("z", (0, 1));
        let y = pldag.set_or(vec!["z"]);
        let x = pldag.set_or(vec![y.clone()]);
        let deps = pldag.transitive_dependencies();

        let expect = |xs: &[&str]| xs.iter().cloned().map(String::from).collect::<HashSet<_>>();

        assert_eq!(deps.get(&x), Some(&expect(&[&y, "z"])));
        assert_eq!(deps.get(&y), Some(&expect(&["z"])));
        assert_eq!(deps.get("z"), Some(&expect(&[])));
    }

    #[test]
    fn test_all_primitives() {
        // no Composite at all
        let mut pldag = Pldag::new();
        for &name in &["p", "q", "r"] {
            pldag.nodes.insert(
                name.into(),
                Node {
                    expression: BoolExpression::Primitive((1, 5)),
                    coefficient: 0.0,
                },
            );
        }
        let deps = pldag.transitive_dependencies();

        for &name in &["p", "q", "r"] {
            assert!(
                deps.get(name).unwrap().is_empty(),
                "{} should have no deps",
                name
            );
        }
    }

    #[test]
    fn test_propagate_weighted() {
        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]);

        // Set coefficients for nodes
        model.set_coef("x", 2.0);
        model.set_coef("y", 3.0);

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));

        let propagated = model.propagate_coefs(&assignments);

        // Check the results: bounds should be (1,1) and coefficients should be accumulated
        assert_eq!(propagated.get("x").unwrap().0, (1, 1)); // bounds
        assert_eq!(propagated.get("x").unwrap().1, (2.0, 2.0)); // coefficients
        assert_eq!(propagated.get("y").unwrap().0, (1, 1)); // bounds
        assert_eq!(propagated.get("y").unwrap().1, (3.0, 3.0)); // coefficients
        assert_eq!(propagated.get(&root).unwrap().0, (1, 1)); // bounds
        assert_eq!(propagated.get(&root).unwrap().1, (5.0, 5.0)); // accumulated coefficients
    }

    #[test]
    fn test_readme_example() {
        // Build your PL-DAG (omitting details)...
        // For example, we create a model of three boolean variables x, y and z.
        // We bind them to an xor constraint.
        let mut pldag: Pldag = Pldag::new();

        // First setup the primitive variables
        pldag.set_primitive("x", (0, 1));
        pldag.set_primitive("y", (0, 1));
        pldag.set_primitive("z", (0, 1));

        // A reference ID is returned
        let root = pldag.set_or(vec!["x", "y", "z"]);

        // 1. Validate a combination:
        let mut inputs: IndexMap<&str, Bound> = IndexMap::new();
        let validited = pldag.propagate(&inputs);
        // Since nothing is given, and all other variable inplicitly is (0, 1) from the pldag model,
        // the root will be (0,1) since there's not enough information to evalute the root `or` node.
        println!("Root valid? {}", *validited.get(&root).unwrap() == (1, 1)); // This will be False

        // If we however fix x to be zero, then the root is false
        inputs.insert("x", (0, 0));
        let revalidited = pldag.propagate(&inputs);
        println!("Root valid? {}", *revalidited.get(&root).unwrap() == (1, 1)); // This will be false

        // However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
        inputs.insert("y", (1, 1));
        inputs.insert("z", (1, 1));
        let revalidited = pldag.propagate(&inputs);
        println!("Root valid? {}", *revalidited.get(&root).unwrap() == (1, 1)); // This will be true

        // 2. Score a configuration:
        // We can score a configuration by setting coefficients on nodes.
        pldag.set_coef("x", 1.0);
        pldag.set_coef("y", 2.0);
        pldag.set_coef("z", 3.0);
        // Add a discount value if the root is true
        pldag.set_coef(&root, -1.0);
        let scores = pldag.propagate_coefs(&inputs);
        println!("Total score: {:?}", scores.get(&root).unwrap().1); // .1 is the coefficient part

        // And notice what will happen if we remove the x value (i.e. x being (0,1))
        inputs.insert("x", (0, 1));
        let scores = pldag.propagate_coefs(&inputs);
        // The root will return bounds with coefficient range meaning the value is between bounds with not enough information to
        // determine the exact value.
        println!("Total score: {:?}", scores.get(&root).unwrap().1); // .1 is the coefficient part

        // .. and if we set x to be 0, then the root will be definitely determined.
        inputs.insert("x", (0, 0));
        let scores = pldag.propagate_coefs(&inputs);
        println!("Total score: {:?}", scores.get(&root).unwrap().1); // .1 is the coefficient part

        // .. and if we set y and z to be 0, then the root will be 0.
        inputs.insert("y", (0, 0));
        inputs.insert("z", (0, 0));
        let scores = pldag.propagate_coefs(&inputs);
        println!("Total score: {:?}", scores.get(&root).unwrap().1); // .1 is the coefficient part
    }

    #[test]
    fn test_get_objective() {
        let mut model = Pldag::new();

        // Create primitive nodes with different coefficients
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));

        // Create a composite node (should not appear in coefficients)
        let root = model.set_and(vec!["x", "y"]);

        // Initially, all objective coefficients should be 0.0 (default)
        let coeffs = model.get_objective();
        assert_eq!(coeffs.len(), 3); // Only primitives should be included
        assert_eq!(coeffs.get("x"), Some(&0.0));
        assert_eq!(coeffs.get("y"), Some(&0.0));
        assert_eq!(coeffs.get("z"), Some(&0.0));
        assert_eq!(coeffs.get(&root), None); // Composite nodes should not be included

        // Set some coefficients
        model.set_coef("x", 2.5);
        model.set_coef("y", -1.0);
        model.set_coef("z", 3.14);
        model.set_coef(&root, 10.0); // This should not appear in get_objective

        // Test that get_objective returns the updated values for primitives only
        let coeffs = model.get_objective();
        assert_eq!(coeffs.len(), 3); // Still only primitives
        assert_eq!(coeffs.get("x"), Some(&2.5));
        assert_eq!(coeffs.get("y"), Some(&-1.0));
        assert_eq!(coeffs.get("z"), Some(&3.14));
        assert_eq!(coeffs.get(&root), None); // Composite still not included

        // Verify the order is preserved (IndexMap should maintain insertion order)
        let keys: Vec<&String> = coeffs.keys().collect();
        assert_eq!(keys, vec!["x", "y", "z"]);

        // Test with a model that has only composite nodes
        let mut model_no_primitives = Pldag::new();
        model_no_primitives.set_primitive("a", (0, 1));
        model_no_primitives.set_primitive("b", (0, 1));
        let composite1 = model_no_primitives.set_and(vec!["a", "b".into()]);
        let composite2 = model_no_primitives.set_or(vec![composite1.clone()]);

        // Remove the primitives by replacing them with composites
        // (This is a bit artificial, but tests the filtering logic)
        let coeffs_mixed = model_no_primitives.get_objective();
        assert_eq!(coeffs_mixed.len(), 2); // Should only include "a" and "b" (primitives)
        assert!(coeffs_mixed.contains_key("a"));
        assert!(coeffs_mixed.contains_key("b"));
        assert!(!coeffs_mixed.contains_key(&composite1));
        assert!(!coeffs_mixed.contains_key(&composite2));
    }

    #[test]
    fn test_presolve_with_xor() {
        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));

        // Create an XOR constraint
        let root = model.set_xor(vec!["x", "y", "z".into()]);

        let mut seed_assignments = IndexMap::new();
        seed_assignments.insert("x", (1, 1)); // x is true
        seed_assignments.insert("y", (0, 0)); // y is false
        seed_assignments.insert("z", (0, 0)); // z is false
        match model.presolve(&seed_assignments) {
            Ok(p) => {
                assert_eq!(p.fixed[&root], (1, 1)); // root should be true
                assert_eq!(p.fixed["x"], (1, 1)); // x should be true
                assert_eq!(p.fixed["y"], (0, 0)); // y should be false
                assert_eq!(p.fixed["z"], (0, 0)); // z
            }
            Err(e) => panic!("Presolve failed: {}", e),
        }

        let mut seed_assignments = IndexMap::new();
        seed_assignments.insert("x", (1, 1)); // x is true
        seed_assignments.insert("y", (0, 1)); // y is f/t meaning we cannot decide
        seed_assignments.insert("z", (0, 0)); // z is false
        match model.presolve(&seed_assignments) {
            Ok(p) => {
                // Root shouldn't be in fixed since y is undecided (0,1)
                assert!(!p.fixed.contains_key(&root));
                assert_eq!(p.fixed["x"], (1, 1)); // x should be true
                assert!(!p.fixed.contains_key("y")); // y is undecided
                assert_eq!(p.fixed["z"], (0, 0)); // z
            }
            Err(e) => panic!("Presolve failed: {}", e),
        }

        // let mut model = Pldag::new();
        // model.set_primitive("x", (0, 1));
        // model.set_primitive("y", (0, 1));
        // model.set_primitive("z", (0, 1));

        // // Create an XOR constraint
        // let root = model.set_and(vec!["x", "y", "z".into()]);
        // let mut seed_assignments = IndexMap::new();
        // seed_assignments.insert(root, (1, 1)); // x
        // match model.presolve(&seed_assignments) {
        //     Ok(p) => {
        //         assert_eq!(p.fixed["x"], (0, 1));
        //         assert_eq!(p.fixed["y"], (0, 1));
        //         assert_eq!(p.fixed["z"], (0, 1));
        //     },
        //     Err(e) => panic!("Presolve failed: {}", e),
        // }
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
    fn test_composite_turns_to_primitive() {
        // When a composite node is either a contradiction or a tautology, it will be simplified to a fixed primitive node (0 if contradiction, 1 if tautology).
        let mut model = Pldag::new();
        let a = model.set_and(Vec::<String>::new());
        let b = model.set_or(Vec::<String>::new());
        assert_eq!(
            model.nodes.get(&a).unwrap().expression,
            BoolExpression::Primitive((1, 1))
        );
        assert_eq!(
            model.nodes.get(&b).unwrap().expression,
            BoolExpression::Primitive((0, 0))
        );
    }

    #[test]
    fn test_equiv() {
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let equiv = model.set_equiv("p", "q");
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 1));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));

        model.set_primitive("p", (0, 0));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));
    }

    #[test]
    fn test_imply() {
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let equiv = model.set_imply("p", "q");
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate(&IndexMap::new());
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));
    }

    #[test]
    fn test_pldag_hash_function() {
        let mut model1 = Pldag::new();
        model1.set_primitive("x", (0, 1));
        model1.set_primitive("y", (0, 1));
        model1.set_and(vec!["x", "y"]);

        let mut model2 = Pldag::new();
        model2.set_primitive("y", (0, 1));
        model2.set_primitive("x", (0, 1));
        model2.set_and(vec!["y", "x"]);

        // Check that hash of model1 and model2 are the same
        assert_eq!(model1.get_hash(), model2.get_hash());
    }

    #[test]
    fn test_sparse_polyhedron_from_pldag_hash_function() {
        let mut model1 = Pldag::new();
        model1.set_primitive("x", (0, 1));
        model1.set_primitive("y", (0, 1));
        model1.set_and(vec!["x", "y"]);
        let polyhash1 = model1.to_sparse_polyhedron_default().unwrap().get_hash();

        let mut model2 = Pldag::new();
        model2.set_primitive("y", (0, 1));
        model2.set_primitive("x", (0, 1));
        model2.set_and(vec!["y", "x"]);
        let polyhash2 = model2.to_sparse_polyhedron_default().unwrap().get_hash();
        assert_eq!(polyhash1, polyhash2);
    }
}
