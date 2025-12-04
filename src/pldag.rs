use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::collections::HashSet;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::iter::empty;
use crate::storage::KeyValueStore;
use serde::{Serialize, Deserialize};

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
            column_bounds: self.column_bounds.iter().enumerate().filter_map(|(i, b)| {
                if !values.contains_key(&self.columns[i]) {
                    Some(*b)
                } else {
                    None
                }
            }).collect(),
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
    /// Column bounds
    pub column_bounds: Vec<Bound>,
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
            column_bounds: sparse.column_bounds,
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
            column_bounds: dense.column_bounds,
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

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Hash)]
/// Represents different types of boolean expressions in the DAG.
pub enum Node {
    /// A composite node representing a linear constraint
    Composite(Constraint),
    /// A primitive (leaf) node with a bound on its value
    Primitive(Bound),
}

/// A list of references to other node IDs.
type References = Vec<String>;

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
    pub storage: Box<dyn KeyValueStore>,
}

impl Pldag {
    /// Creates a new empty PL-DAG.
    ///
    /// # Returns
    /// A new `Pldag` instance with no nodes
    pub fn new() -> Pldag {
        Pldag {
            storage: Box::new(crate::storage::InMemoryStore::new()),
        }
    }

    pub fn new_custom(storage: Box<dyn KeyValueStore>) -> Pldag {
        Pldag {
            storage,
        }
    }

    fn get_incoming(&self, ids: Vec<&str>) -> Vec<Node> {
        let results = self.storage
            .mget(&ids.iter().map(|s| s.to_string()).collect::<Vec<String>>());

        // Preserve the order of the input ids by looking up each key in the HashMap
        ids.iter()
            .filter_map(|id| {
                results.get(&id.to_string())
                    .and_then(|v| serde_json::from_value::<Node>(v.clone()).ok())
            })
            .collect()
    }

    fn get_outgoing_ids(&self, ids: Vec<&str>) -> Vec<References> {
        let keys: Vec<String> = ids.iter()
            .map(|s| format!("__outgoing__{}", s))
            .collect();
        let results = self.storage.mget(&keys);

        // Preserve the order of the input ids by looking up each key in the HashMap
        keys.iter()
            .map(|key| {
                results.get(key)
                    .and_then(|v| serde_json::from_value::<References>(v.clone()).ok())
                    .unwrap_or_default()
            })
            .collect()
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
    pub fn propagate<K>(&self, assignments: impl IntoIterator<Item = (K, Bound)>) -> Assignment
    where
        K: ToString,
    {
        let mut results = assignments.into_iter().map(|(k, v)| (k.to_string(), v)).collect::<Assignment>();

        // Extract all keys from the initial assignments
        let batch_keys: Vec<String> = results.keys().cloned().collect();
        let batch: Vec<String> = batch_keys.iter().map(|k| k.to_string()).collect();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();
        queue.push_back(batch.clone());

        let mut visited = HashSet::new();
        while let Some(current_batch) = queue.pop_front() {
            visited.extend(current_batch.iter().cloned());

            let mut next_batch: Vec<String> = Vec::new();
            let batch_incoming = self.get_incoming(current_batch.iter().map(|s| s.as_str()).collect());

            for (i, node) in batch_incoming.iter().enumerate() {
                let node_id = current_batch[i].clone();
                if results.contains_key(&node_id) {
                    continue; // Already have result for this node
                }
                match node {
                    Node::Primitive(primitive) => {
                        // Primitive nodes already have their bounds set
                        results.insert(
                            node_id.to_string(),
                            *primitive,
                        );
                    }
                    Node::Composite(constraint) => {
                        // Filter coefficients and calculate bias
                        let coefficients: HashMap<String, i64> = constraint
                                                    .coefficients
                                                    .iter()
                                                    .cloned()
                                                    .collect();

                        let bias: i64 = constraint.bias.0; // Using lower bound for bias

                        // Check if all input variables are in results
                        let all_inputs_available = coefficients
                            .keys()
                            .all(|input_id| results.contains_key(input_id));

                        if !all_inputs_available {
                            // Put the missing coefficients back to the queue
                            for input_id in coefficients.keys() {
                                if !results.contains_key(input_id) && !next_batch.contains(input_id) {
                                    next_batch.push(input_id.clone());
                                }
                            }
                            // Not all inputs are ready, push this node back to the queue
                            next_batch.push(node_id.clone());
                            continue;
                        }

                        // Get coefficient values from results
                        let mut coef_vals = HashMap::new();
                        for input_id in coefficients.keys() {
                            if let Some(val) = results.get(input_id) {
                                coef_vals.insert(input_id.clone(), val.clone());
                            }
                        }

                        // Calculate result
                        let multiplied: Vec<(i64, i64)> = coefficients
                            .iter()
                            .map(|(input_id, coef)| {
                                bound_multiply(*coef, *coef_vals.get(input_id).unwrap())
                            })
                            .collect();

                        let summed = multiplied
                            .iter()
                            .fold((0, 0), |acc, b| bound_add(acc, *b));
                        let biased = bound_add(summed, (bias, bias));

                        results.insert(node_id.to_string(), ((biased.0 >= 0) as i64, (biased.1 >= 0) as i64));
                    }
                }
            }

            // Add dependent nodes to next batch
            let batch_outgoing = self.get_outgoing_ids(current_batch.iter().map(|s| s.as_str()).collect());
            for outgoing in batch_outgoing.into_iter() {
                for dependent in outgoing.into_iter() {
                    if !visited.contains(dependent.as_str()) && !next_batch.contains(&dependent) {
                        next_batch.push(dependent);
                    }
                }
            }

            if !next_batch.is_empty() {
                queue.push_back(next_batch);
            }
        }

        results
    }

    /// Propagates bounds using default primitive variable bounds.
    ///
    /// Convenience method that calls `propagate` with the default bounds
    /// of all primitive variables as defined in the DAG.
    ///
    /// # Returns
    /// Complete assignments with bounds for all nodes
    pub fn propagate_default(&self) -> Assignment {
        let primitives = self.get_primitives(vec![]);
        self.propagate(primitives)
    }

    #[cfg(feature = "glpk")]
    #[cfg_attr(feature = "trace", tracing::instrument(skip_all))]
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
        let polyhedron = self.to_sparse_polyhedron(
            assume.iter().map(|(k, v)| (k.to_string(), *v)).collect::<HashMap<String, Bound>>(),
            true,
            true,
            true,
        )?;

        println!("Polyhedron for ILP solving:\n{}", DensePolyhedron::from(polyhedron.clone()));

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
                .zip(polyhedron.column_bounds.iter())
                .map(|(key, bound)| {
                    Variable {
                        id: key.as_str(),
                        bound: (bound.0 as i32, bound.1 as i32),
                    }
                })
                .collect(),
            double_bound: false,
        };

        // If there are no constraints, insert a dummy row
        if glpk_matrix.A.rows.is_empty() {
            for i in 0..polyhedron.columns.len() {
                glpk_matrix.A.rows.push(i as i32);
                glpk_matrix.A.cols.push(i as i32);
                glpk_matrix.A.vals.push(0);
            }
            glpk_matrix.b.push((0, 0));
        }

        let mut solutions: Vec<Solution> = Vec::default();
        #[cfg(feature = "trace")] {
            let span = tracing::span!(tracing::Level::INFO, "solve_ilps");
            solutions = span.in_scope(|| solve_ilps(&mut glpk_matrix, objectives, maximize, false));
        }
        #[cfg(not(feature = "trace"))] {
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

    pub fn sub_tree(&self, roots: Vec<String>) -> HashMap<String, Node> {
        // By default use Rust implementation
        self.sub_tree_rust(roots)
    }

    pub fn tree(&self) -> HashMap<String, Node> {
        // Get all root nodes (nodes with no incoming edges)
        let keys = self
            .storage
            .keys("*")
            .into_iter()
            .filter(|k| !k.starts_with("__outgoing__"))
            .collect::<Vec<_>>();

        let data = self.get_incoming(keys.iter().map(|s| s.as_str()).collect());
        keys.into_iter()
            .zip(data.into_iter())
            .collect::<HashMap<String, Node>>()
    }

    /// Rust-based subtree traversal (fallback for non-Redis stores)
    /// Returns: HashMap<node_id, (incoming_edges, domain)>
    fn sub_tree_rust(&self, roots: Vec<String>) -> HashMap<String, Node> {

        // If no roots, return empty tree
        if roots.is_empty() {
            return self.tree();
        }
        
        // Use a single visited set for O(1) lookups
        let mut visited = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();
        queue.push_back(roots.iter().map(|s| s.to_string()).collect());

        // Accumulate nodes for larger batches
        let mut sub_tree: HashMap<String, Node> = HashMap::new();

        while let Some(current_batch) = queue.pop_front() {
            // Batch fetch incoming edges for current batch
            let all_incoming = self.get_incoming(current_batch.iter().map(|s| s.as_str()).collect());
            let mut next_batch = Vec::new();

            for (input_id, incoming) in current_batch.iter().zip(all_incoming.iter()) {
                sub_tree.insert(input_id.clone(), incoming.clone());
                // HashSet.insert returns false if already present
                if visited.insert(input_id.clone()) {
                    match incoming {
                        Node::Primitive(_) => {}
                        Node::Composite(constraint) => {
                            // Enqueue all coefficient variable IDs
                            for (coef_id, _) in constraint.coefficients.iter() {
                                if !visited.contains(coef_id.as_str()) && !next_batch.contains(coef_id) {
                                    next_batch.push(coef_id.clone());
                                }
                            }
                        }
                    }
                    next_batch.push(input_id.clone());
                }
            }

            if !next_batch.is_empty() {
                queue.push_back(next_batch);
            }
        }

        sub_tree
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
        assume: HashMap<String, Bound>,
        double_binding: bool,
        integer_constraints: bool,
        fixed_constraints: bool,
    ) -> Result<SparsePolyhedron, String> {

        // Create a new sparse matrix
        let mut A_matrix = SparseIntegerMatrix::new();
        let mut b_vector: Vec<i64> = Vec::new();

        // Get the sub tree from the roots
        let sub_dag = self.sub_tree(assume.keys().cloned().collect());

        // Check that the assume keys are all in the sub_dag. If not, return an error.
        for key in assume.keys() {
            if !sub_dag.contains_key(key) {
                return Err(format!("Assume key '{}' not found in PL-DAG", key));
            }
        }

        // Filter out all Nodes that are primitives
        let primitives: IndexMap<&String, Bound> = sub_dag
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
        let composites: IndexMap<&String, &Constraint> = sub_dag
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
            let coef_bounds = composite.coefficients.iter().map(|(coef_key, _)| {
                            (coef_key.clone(), match sub_dag.get(coef_key) {
                                Some(Node::Primitive(bound)) => *bound,
                                _ => (0, 1),
                            })
                        }).collect::<IndexMap<String, Bound>>();

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

        // Set an lower and upper bound for the variables in assume
        for (a_key, a_bound) in assume.iter() {
            // Get the index of the current key
            let ai = *column_names_map.get(a_key.as_str()).unwrap();

            A_matrix.rows.push(row_i);
            A_matrix.cols.push(ai);
            A_matrix.vals.push(1);
            b_vector.push(a_bound.0);
            row_i += 1;

            A_matrix.rows.push(row_i);
            A_matrix.cols.push(ai);
            A_matrix.vals.push(-1);
            b_vector.push(-1 * a_bound.1);
            row_i += 1;
        }

        // Set the shape of the A matrix
        A_matrix.shape = (row_i, column_names_map.len());

        // Create the polyhedron
        let polyhedron = SparsePolyhedron {
            A: A_matrix,
            b: b_vector,
            columns: column_names_map.keys().cloned().collect(),
            integer_columns: integer_variables,
            column_bounds: column_names_map
                .keys()
                .map(|key| (match sub_dag.get(key) {
                                Some(Node::Primitive(bound)) => *bound,
                                _ => (0, 1),
                            }))
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
    pub fn to_sparse_polyhedron_default(&self, assume: HashMap<String, Bound>) -> Result<SparsePolyhedron, String> {
        self.to_sparse_polyhedron(assume, true, true, true)
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
        assume: HashMap<String, Bound>,
        double_binding: bool,
        integer_constraints: bool,
        fixed_constraints: bool,
    ) -> Result<DensePolyhedron, String> {
        // Convert to sparse polyhedron first
        let sparse_polyhedron =
            self.to_sparse_polyhedron(assume, double_binding, integer_constraints, fixed_constraints)?;
        // Convert sparse to dense polyhedron
        Ok(sparse_polyhedron.into())
    }

    /// Converts the PL-DAG to a dense polyhedron with default settings.
    ///
    /// # Returns
    /// A `DensePolyhedron` with all constraint options enabled
    pub fn to_dense_polyhedron_default(&self, assume: HashMap<String, Bound>) -> Result<DensePolyhedron, String> {
        self.to_dense_polyhedron(assume, true, true, true)
    }

    /// Retrieves all primitive variables from the given PL-DAG roots.
    /// ///
    /// # Returns
    /// An `IndexMap` mapping variable IDs to their corresponding `Bound` objects
    pub fn get_primitives(&self, roots: Vec<String>) -> IndexMap<String, Bound> {
        let sub_dag = self.sub_tree(roots);
        sub_dag.iter().filter_map(|(key, node)| {
            if let Node::Primitive(bound) = node {
                Some((key.clone(), *bound))
            } else {
                None
            }
        }).collect()
    }

    /// Retrieves all composite constraints from the PL-DAG.
    /// ///
    /// # Returns
    /// An `IndexMap` mapping constraint IDs to their corresponding `Constraint` objects
    pub fn get_composites(&self, roots: Vec<String>) -> IndexMap<String, Constraint> {
        let sub_dag = self.sub_tree(roots);
        sub_dag.iter().filter_map(|(key, node)| {
            if let Node::Composite(constraint) = node {
                Some((key.clone(), constraint.clone()))
            } else {
                None
            }
        }).collect()
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
        self.storage.set(
            id,
            serde_json::to_value(Node::Primitive(bound)).unwrap(),
        );
        // Set empty outgoing references for this primitive if not already present
        if !self.storage.exists(&format!("__outgoing__{}", id)) {
            self.storage.set(
                &format!("__outgoing__{}", id),
                serde_json::to_value(Vec::<String>::new()).unwrap(),
            );
        }
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
        let id = hash.to_string();
        if self.storage.exists(&id) {
            // Constraint already exists, return existing ID
            return id.to_string();
        }

        // Set empty outgoing references for this constraint, if not already present
        self.storage.set(
            &format!("__outgoing__{}", id),
            serde_json::to_value(Vec::<String>::new()).unwrap(),
        );

        let constraint = Constraint {
            coefficients: coefficient_variables.clone(),
            bias: (bias, bias),
        };

        // Insert the constraint as a node
        self.storage.set(
            &id,
            serde_json::to_value(Node::Composite(constraint)).unwrap(),
        );

        // For each coefficient variable, add this id as an incoming reference
        // Use mget to batch fetch all incoming references
        let coef_ids: Vec<&str> = coefficient_variables
            .iter()
            .map(|(coef_id, _)| coef_id.as_str())
            .collect();
        let coefficient_current_references: Vec<References> = self.get_outgoing_ids(coef_ids.clone());

        for (coef_id, mut current_references) in coef_ids.iter().zip(coefficient_current_references) {
            if !current_references.contains(&id.to_string()) {
                current_references.push(id.to_string());
                self.storage.set(
                    &format!("__outgoing__{}", coef_id),
                    serde_json::to_value(current_references).unwrap(),
                );
            }
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
        println!("not_condition ID: {}", not_condition);
        let id = self.set_or(vec![not_condition, consequence.into()]);
        println!("imply ID: {}", id);
        id
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
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a helper function that generates all primitive combinations
    // for a given PLDAG model, propagates them, and compares against the
    // corresponding polyhedron evaluations.
    fn primitive_combinations(model: &Pldag) -> Vec<IndexMap<String, i64>> {
        let tree = model.tree();
        let primitives: Vec<&String> = tree.iter()
            .filter_map(|(key, node)| {
                if let Node::Primitive(_) = node {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        let mut combinations: Vec<IndexMap<String, i64>> = Vec::new();

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
            let prop = model.propagate(interp);
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

        let result = model.propagate_default();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        let result = model.propagate(assignments);
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        // let mut model = Pldag::new();
        // model.set_primitive("x", (0, 1));
        // model.set_primitive("y", (0, 1));
        // model.set_primitive("z", (0, 1));
        // let root = model.set_xor(vec!["x", "y", "z".into()]);
        // let result = model.propagate_default();
        // assert_eq!(result.get("x").unwrap(), &(0, 1));
        // assert_eq!(result.get("y").unwrap(), &(0, 1));
        // assert_eq!(result.get("z").unwrap(), &(0, 1));
        // assert_eq!(result.get(&root).unwrap(), &(0, 1));

        // let mut assignments = IndexMap::new();
        // assignments.insert("x", (1, 1));
        // assignments.insert("y", (1, 1));
        // assignments.insert("z", (1, 1));
        // let result = model.propagate(assignments);
        // assert_eq!(result.get(&root).unwrap(), &(0, 0));

        // let mut assignments = IndexMap::new();
        // assignments.insert("x", (0, 1));
        // assignments.insert("y", (1, 1));
        // assignments.insert("z", (1, 1));
        // let result = model.propagate(assignments);
        // assert_eq!(result.get(&root).unwrap(), &(0, 0));

        // let mut assignments = IndexMap::new();
        // assignments.insert("x", (0, 0));
        // assignments.insert("y", (1, 1));
        // assignments.insert("z", (0, 0));
        // let result = model.propagate(assignments);
        // assert_eq!(result.get(&root).unwrap(), &(1, 1));
    }

    /// XOR already covered; test the OR gate
    #[test]
    fn test_propagate_or_gate() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let or_root = model.set_or(vec!["a", "b"]);

        // No assignment: both inputs full [0,1], output [0,1]
        let res = model.propagate_default();
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (1, 1));
        let res = model.propagate(interp);
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (0, 0));
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp);
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp);
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[test]
    fn test_propagate_not_gate() {
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        let not_root = model.set_not(vec!["p"]);

        // no assignment ⇒ [0,1]
        let res = model.propagate_default();
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = IndexMap::<String, Bound>::new();
        interp.insert("p".into(), (0, 0));
        let res = model.propagate(interp);
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (1, 1));
        let res = model.propagate(interp);
        assert_eq!(res[&not_root], (0, 0));
    }

    #[test]
    fn test_to_polyhedron_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        let root = m.set_and(vec!["x", "y"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_or() {
        let mut m = Pldag::new();
        m.set_primitive("a".into(), (0, 1));
        m.set_primitive("b".into(), (0, 1));
        m.set_primitive("c".into(), (0, 1));
        let root = m.set_or(vec!["a", "b", "c"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_not() {
        let mut m = Pldag::new();
        m.set_primitive("p".into(), (0, 1));
        let root = m.set_not(vec!["p"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_xor() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        m.set_primitive("z".into(), (0, 1));
        let root = m.set_xor(vec!["x", "y", "z"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
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

        let mut assume = HashMap::new();
        assume.insert(v.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
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
        let res = model.propagate_default();
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
        let res = model.propagate(interp);
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (1, 1));
        interp.insert("z", (1, 1));
        let res = model.propagate(interp);
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (0, 0));
        interp.insert("z", (0, 0));
        let res = model.propagate(interp);
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

        let mut interp = IndexMap::<&str, Bound>::new();
        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        interp.insert("u".into(), (5, 5));
        let res = model.propagate(interp);

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
            for combination in primitive_combinations(model) {
                let assignments = combination
                    .iter()
                    .map(|(k, &v)| (k.as_str(), (v, v)))
                    .collect::<IndexMap<&str, Bound>>();
                let model_prop = model.propagate(assignments);
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
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let polyhedron: DensePolyhedron = model.to_sparse_polyhedron_default(assume).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let polyhedron = model.to_sparse_polyhedron_default(assume).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let polyhedron = model.to_sparse_polyhedron_default(assume).unwrap().into();
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
            let mut assume = HashMap::new();
            assume.insert(root.clone(), (1, 1));
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // OR(y) == y
        {
            let mut m = Pldag::new();
            m.set_primitive("y", (0, 1));
            let root = m.set_or(vec!["y"]);
            let mut assume = HashMap::new();
            assume.insert(root.clone(), (1, 1));
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // XOR(z) == z
        {
            let mut m = Pldag::new();
            m.set_primitive("z".into(), (0, 1));
            let root = m.set_xor(vec!["z"]);
            let mut assume = HashMap::new();
            assume.insert(root.clone(), (1, 1));
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[test]
    fn test_to_polyhedron_duplicate_operands_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        let root = m.set_and(vec!["x", "x"]);
        let mut assume = HashMap::new();
        assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(assume).unwrap().into();
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
        let mut root_assume = HashMap::new();
        root_assume.insert(root.clone(), (1, 1));
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(root_assume).unwrap().into();
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
        let equiv = model.set_equiv("p", "q");
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 1));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));

        model.set_primitive("p", (0, 0));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));
    }

    #[test]
    fn test_imply() {
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let equiv = model.set_imply("p", "q");
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (1, 1));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        model.set_primitive("p", (1, 1));
        model.set_primitive("q", (0, 0));
        let propagated = model.propagate_default();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));
    }

    // Commented out: test_pldag_hash_function uses get_hash() which no longer exists
    // #[test]
    // fn test_pldag_hash_function() {
    //     let mut model1 = Pldag::new();
    //     model1.set_primitive("x", (0, 1));
    //     model1.set_primitive("y", (0, 1));
    //     model1.set_and(vec!["x", "y"]);
    //
    //     let mut model2 = Pldag::new();
    //     model2.set_primitive("y", (0, 1));
    //     model2.set_primitive("x", (0, 1));
    //     model2.set_and(vec!["y", "x"]);
    //
    //     // Check that hash of model1 and model2 are the same
    //     assert_eq!(model1.get_hash(), model2.get_hash());
    // }

    // Commented out: test_sparse_polyhedron_from_pldag_hash_function uses get_hash() and undefined variable root
    // #[test]
    // fn test_sparse_polyhedron_from_pldag_hash_function() {
    //     let mut model1 = Pldag::new();
    //     model1.set_primitive("x", (0, 1));
    //     model1.set_primitive("y", (0, 1));
    //     model1.set_and(vec!["x", "y"]);
    //     let polyhash1 = model1.to_sparse_polyhedron_default(vec![root.clone()]).unwrap().get_hash();
    //
    //     let mut model2 = Pldag::new();
    //     model2.set_primitive("y", (0, 1));
    //     model2.set_primitive("x", (0, 1));
    //     model2.set_and(vec!["y", "x"]);
    //     let polyhash2 = model2.to_sparse_polyhedron_default(vec![root.clone()]).unwrap().get_hash();
    //     assert_eq!(polyhash1, polyhash2);
    // }
}
