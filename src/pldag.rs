use crate::error::{PldagError, Result};
use crate::storage::{InMemoryStore, NodeStore, NodeStoreTrait};
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::{hash_map::DefaultHasher, HashMap};
use std::fmt;
use std::hash::{Hash, Hasher};

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
            storage: Box::new(NodeStore::new(Box::new(InMemoryStore::new()))),
        }
    }

    pub fn new_custom(storage: Box<dyn NodeStoreTrait>) -> Pldag {
        Pldag { storage }
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
    pub fn propagate<K>(&self, assignments: impl IntoIterator<Item = (K, Bound)>) -> Result<Assignment>
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
        while queue.len() > 0 {
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
        self.propagate(primitives)
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
        &self,
        roots: Vec<ID>,
        objectives: Vec<HashMap<&str, f64>>,
        assume: HashMap<&str, Bound>,
        maximize: bool,
    ) -> Vec<Option<Assignment>> {
        use glpk_rust::{
            solve_ilps, IntegerSparseMatrix, Solution, SparseLEIntegerPolyhedron, Status, Variable,
        };

        // Convert the PL-DAG to a polyhedron representation
        let polyhedron = self.to_sparse_polyhedron(roots, true);

        // Validate assume that the bounds does not override column bounds
        for (key, bound) in assume.iter() {
            if let Some(idx) = polyhedron.columns.iter().position(|col| col == key) {
                let col_bound = polyhedron.column_bounds[idx];
                if bound.0 < col_bound.0 || bound.1 > col_bound.1 {
                    return vec![None; objectives.len()];
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

        return solutions
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
            .collect();
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
                match incoming {
                    Node::Primitive(_) => {}
                    Node::Composite(constraint) => {
                        // Enqueue all coefficient variable IDs
                        for (coef_id, _) in constraint.coefficients.iter() {
                            if sub_dag.contains_key(coef_id) {
                                return Err(PldagError::CycleDetected { node_id: coef_id.to_string() });
                            }
                            if !next_batch.contains(coef_id) {
                                next_batch.push(coef_id.clone());
                            }
                        }
                    }
                }
                next_batch.push(input_id.clone());
                sub_dag.insert(input_id.clone(), incoming.clone());
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
    /// * `double_binding` - If true, creates bidirectional implications for composite nodes
    ///
    /// # Returns
    /// A `SparsePolyhedron` representing the DAG constraints
    pub fn to_sparse_polyhedron(&self, roots: Vec<ID>, double_binding: bool) -> SparsePolyhedron {
        // Create a new sparse matrix
        let mut a_matrix = SparseIntegerMatrix::new();
        let mut b_vector: Vec<i32> = Vec::new();

        // Get the sub tree from the roots
        let sub_dag = self.sub_dag(roots.clone()).unwrap();

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
            let coef_bounds = composite
                .coefficients
                .iter()
                .map(|(coef_key, _)| {
                    (
                        coef_key.clone(),
                        match sub_dag.get(coef_key) {
                            Some(Node::Primitive(bound)) => *bound,
                            _ => (0, 1),
                        },
                    )
                })
                .collect::<IndexMap<String, Bound>>();

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
                    match sub_dag.get(key) {
                        Some(Node::Primitive(bound)) => *bound,
                        _ => (0, 1),
                    }
                })
                .collect(),
        };

        return polyhedron;
    }

    /// Converts the PL-DAG to a sparse polyhedron with default settings.
    ///
    /// Convenience method that calls `to_sparse_polyhedron` with all options enabled:
    /// double_binding=true, integer_constraints=true, fixed_constraints=true.
    ///
    /// # Returns
    /// A `SparsePolyhedron` with full constraint encoding
    pub fn to_sparse_polyhedron_default(&self, roots: Vec<ID>) -> SparsePolyhedron {
        self.to_sparse_polyhedron(roots, true)
    }

    /// Converts the PL-DAG to a dense polyhedron.
    ///
    /// # Arguments
    /// * `double_binding` - If true, creates bidirectional implications
    ///
    /// # Returns
    /// A `DensePolyhedron` representing the DAG constraints
    pub fn to_dense_polyhedron(&self, roots: Vec<ID>, double_binding: bool) -> DensePolyhedron {
        // Convert to sparse polyhedron first
        let sparse_polyhedron = self.to_sparse_polyhedron(roots, double_binding);
        // Convert sparse to dense polyhedron
        sparse_polyhedron.into()
    }

    /// Converts the PL-DAG to a dense polyhedron with default settings.
    ///
    /// # Returns
    /// A `DensePolyhedron` with all constraint options enabled
    pub fn to_dense_polyhedron_default(&self, roots: Vec<ID>) -> DensePolyhedron {
        self.to_dense_polyhedron(roots, true)
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

    /// Creates a primitive (leaf) variable with the specified bounds.
    ///
    /// Primitive variables represent the base variables in the DAG and have
    /// no dependencies on other nodes.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the variable
    /// * `bound` - The allowed range (min, max) for this variable
    pub fn set_primitive(&mut self, id: &str, bound: Bound) {
        self.storage.set_node(id, Node::Primitive(bound));
    }

    /// Creates multiple primitive variables with the same bounds.
    ///
    /// Convenience method to create several primitive variables at once.
    /// Duplicate IDs are automatically filtered out.
    ///
    /// # Arguments
    /// * `ids` - Iterator of unique identifiers for the variables
    /// * `bound` - The common bound to apply to all variables
    pub fn set_primitives<K>(&mut self, ids: impl IntoIterator<Item = K>, bound: Bound)
    where
        K: ToString,
    {
        let unique_ids: IndexSet<String> = ids.into_iter().map(|k| k.to_string()).collect();
        for id in unique_ids {
            self.set_primitive(&id, bound);
        }
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_gelineq<K>(
        &mut self,
        coefficient_variables: impl IntoIterator<Item = (K, i32)>,
        bias: i32,
    ) -> ID
    where
        K: ToString,
    {
        // Ensure coefficients have unique keys by summing duplicate values
        let mut unique_coefficients: IndexMap<ID, i32> = IndexMap::new();
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
        let constraint = Constraint {
            coefficients: coefficient_variables.clone(),
            bias: (bias, bias),
        };

        // Insert the constraint as a node
        self.storage.set_node(&id, Node::Composite(constraint));

        id.to_string()
    }

    /// Creates an "at least" constraint: sum(variables) >= value.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to sum
    /// * `value` - Minimum required sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_atleast<K>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> ID
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, 1)), -value)
    }

    pub fn set_atleast_ref<K, V>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_atmost<K>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> ID
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, -1)), value)
    }

    pub fn set_atmost_ref<K, V>(
        &mut self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_equal<K, I>(
        &mut self,
        references: I,
        value: i32,
    ) -> ID
    where
        K: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast(references.clone(), value);
        let lb = self.set_atmost(references, value);
        self.set_and(vec![ub, lb])
    }

    pub fn set_equal_ref<K, V, I>(
        &mut self,
        references: I,
        value: V,
    ) -> ID
    where
        K: ToString,
        V: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast_ref(references.clone(), value.to_string());
        let lb = self.set_atmost_ref(references, value);
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_and<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_or<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atleast(unique_references.iter().map(|x| x.as_str()), 1)
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_nand<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_nor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_not<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_xor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let atleast = self.set_or(unique_references.iter().map(|x| x.as_str()));
        let atmost = self.set_atmost(unique_references.iter().map(|x| x.as_str()), 1);
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_xnor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
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
    pub fn set_imply<C, Q>(&mut self, condition: C, consequence: Q) -> ID
    where
        C: ToString,
        Q: ToString,
    {
        let not_condition = self.set_not(vec![condition.to_string()]);
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
    /// The unique ID assigned to this constraint OR None if it failed to create the constraint
    pub fn set_equiv<L, R>(&mut self, lhs: L, rhs: R) -> ID
    where
        L: ToString,
        R: ToString,
    {
        // Convert to strings first to avoid type mismatches
        let lhs_str = lhs.to_string();
        let rhs_str = rhs.to_string();

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
            let prop = model.propagate(interp).unwrap();
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

        let result = model.propagate_default().unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        let result = model.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let result = model.propagate_default().unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = model.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = IndexMap::new();
        assignments.insert("x", (0, 0));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (0, 0));
        let result = model.propagate(assignments).unwrap();
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
        let res = model.propagate_default().unwrap();
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (1, 1));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("a".into(), (0, 0));
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("b".into(), (0, 0));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[test]
    fn test_propagate_not_gate() {
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        let not_root = model.set_not(vec!["p"]);

        // no assignment ⇒ [0,1]
        let res = model.propagate_default().unwrap();
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = IndexMap::<String, Bound>::new();
        interp.insert("p".into(), (0, 0));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (1, 1));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&not_root], (0, 0));
    }

    #[test]
    fn test_to_polyhedron_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        let root = m.set_and(vec!["x", "y"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_or() {
        let mut m = Pldag::new();
        m.set_primitive("a".into(), (0, 1));
        m.set_primitive("b".into(), (0, 1));
        m.set_primitive("c".into(), (0, 1));
        let root = m.set_or(vec!["a", "b", "c"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_not() {
        let mut m = Pldag::new();
        m.set_primitive("p".into(), (0, 1));
        let root = m.set_not(vec!["p"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_xor() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        m.set_primitive("y", (0, 1));
        m.set_primitive("z".into(), (0, 1));
        let root = m.set_xor(vec!["x", "y", "z"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
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
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
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
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (1, 1));
        interp.insert("z", (1, 1));
        let res = model.propagate(interp).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (0, 0));
        interp.insert("z", (0, 0));
        let res = model.propagate(interp).unwrap();
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
        let res = model.propagate(interp).unwrap();

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
                let model_prop = model.propagate(assignments).unwrap();
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
        let polyhedron: DensePolyhedron = model.to_sparse_polyhedron_default(vec![]).into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        let root = model.set_and(vec!["x", "y"]);
        let polyhedron = model.to_sparse_polyhedron_default(vec![]).into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x", (0, 1));
        model.set_primitive("y", (0, 1));
        model.set_primitive("z", (0, 1));
        let root = model.set_xor(vec!["x", "y", "z".into()]);
        let polyhedron = model.to_sparse_polyhedron_default(vec![]).into();
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
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // OR(y) == y
        {
            let mut m = Pldag::new();
            m.set_primitive("y", (0, 1));
            let root = m.set_or(vec!["y"]);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // XOR(z) == z
        {
            let mut m = Pldag::new();
            m.set_primitive("z".into(), (0, 1));
            let root = m.set_xor(vec!["z"]);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[test]
    fn test_to_polyhedron_duplicate_operands_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0, 1));
        let root = m.set_and(vec!["x", "x"]);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
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
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default(vec![]).into();
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
        let equiv = model.set_imply("p", "q");
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
        let result = model.propagate(interp);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (-1, 2)); // Out of bounds
        let result = model.propagate(interp);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (-1, -1)); // Out of bounds
        let result = model.propagate(interp);
        assert!(matches!(result, Err(PldagError::NodeOutOfBounds { .. })));
        
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("p".into(), (1, 1)); // Not out of bounds
        let result = model.propagate(interp);
        assert!(matches!(result, Ok(_)));
    }

    #[test]
    fn test_node_not_found_error_when_propagate() {
        // If we propagate a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        model.set_and(vec!["p", "q", "r"]); // 'r' does not exist
        let mut interp = IndexMap::<&str, Bound>::new();
        interp.insert("q".into(), (0, 1));
        let result = model.propagate(interp);
        assert!(matches!(result, Err(PldagError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[test]
    fn test_node_not_found_error_when_sub_dag() {
        // If we create a sub-dag with a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let mut model = Pldag::new();
        model.set_primitive("p", (0, 1));
        model.set_primitive("q", (0, 1));
        let root = model.set_and(vec!["p", "q", "r"]);
        let result = model.sub_dag(vec![root]);
        assert!(matches!(result, Err(PldagError::NodeNotFound { node_id } ) if node_id == "r"));
    }
}