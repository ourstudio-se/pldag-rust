use crate::error::{ComputeError, ComputeResult, ModelError, ModelResult};
use crate::storage::{InMemoryStore, NodeStore, NodeStoreTrait};
use indexmap::IndexSet;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
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
#[allow(dead_code)]
fn tighten_constraint_true(constraint: &Constraint, values: &mut HashMap<String, Bound>) -> bool {
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

/// Evaluate a constraint from CompiledDag: returns (0,0), (1,1), or (0,1).
fn evaluate_constraint(coefs: &[Coef], bias_lo: i32, values: &[Bound], _dag: &CompiledDag) -> Bound {
    let mut sum = (0, 0);
    for c in coefs.iter() {
        let input_bound = values[c.input as usize];
        sum = bound_add(sum, bound_multiply(c.coef, input_bound));
    }
    let biased = bound_add(sum, (bias_lo, bias_lo));
    ((biased.0 >= 0) as i32, (biased.1 >= 0) as i32)
}

/// Tighten variable bounds for CompiledDag assuming constraint is TRUE.
fn tighten_constraint_true_compiled(coefs: &[Coef], bias_lo: i32, values: &mut [Bound]) -> bool {
    let mut changed = false;
    let b = -bias_lo;

    for (k, coef_k) in coefs.iter().enumerate() {
        let a_k = coef_k.coef;
        if a_k == 0 {
            continue;
        }

        let var_k_idx = coef_k.input as usize;
        let (mut l_k, mut u_k) = values[var_k_idx];

        // Compute best help from other variables
        let mut big_b = 0i32;
        for (i, coef_i) in coefs.iter().enumerate() {
            if i == k {
                continue;
            }
            let a_i = coef_i.coef;
            let var_i_idx = coef_i.input as usize;
            let (l_i, u_i) = values[var_i_idx];

            if a_i > 0 {
                big_b += a_i * u_i;
            } else if a_i < 0 {
                big_b += a_i * l_i;
            }
        }

        // a_k * x_k + B >= b   ->   solve for x_k
        if a_k > 0 {
            let num = b - big_b;
            let new_l = div_ceil(num, a_k);
            if new_l > l_k {
                l_k = new_l;
                changed = true;
            }
        } else {
            // a_k < 0
            let num = b - big_b;
            let new_u = div_floor(num, a_k);
            if new_u < u_k {
                u_k = new_u;
                changed = true;
            }
        }

        // Write back updated bound
        values[var_k_idx] = (l_k, u_k);
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
    pub fn evaluate(&self, assignments: &HashMap<String, Bound>) -> Bound {
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

impl Default for SparseIntegerMatrix {
    fn default() -> Self {
        SparseIntegerMatrix::new()
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
pub type Assignment = HashMap<ID, Bound>;

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
    pub fn dot(&self, values: &HashMap<String, Bound>) -> Bound {
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
    pub fn evaluate(&self, values: &HashMap<String, Bound>) -> Bound {
        let bound = self.dot(values);
        (
            (bound.0 + self.bias.0 >= 0) as i32,
            (bound.1 + self.bias.1 >= 0) as i32,
        )
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

/// The kind of a node in a [`CompiledDag`].
///
/// Each node is either a leaf (a primitive variable with declared bounds) or
/// a composite (a linear-combination constraint over other nodes).
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub enum Kind {
    /// A leaf variable with a declared inherent bound `(lower, upper)`.
    Primitive {
        /// The intrinsic bound the variable is allowed to take.
        inherent: Bound,
    },
    /// A composite (linear-combination) constraint.
    ///
    /// Evaluates as `sum(coef_i * value_i) + bias_lo >= 0`, where the
    /// `(coef_i, value_i)` pairs are the slice `coefs[start..end]` of the
    /// owning [`CompiledDag`].
    Composite {
        /// Constant additive term applied before the `>= 0` test.
        bias_lo: i32,
        /// Half-open `[start, end)` range into the parent DAG's flat
        /// [`coefs`](CompiledDag::coefs) vector.
        coef_range: (usize, usize),
    },
}

/// A single `(input_index, coefficient)` term inside a composite constraint.
///
/// Composite nodes in a [`CompiledDag`] reference these via a half-open
/// [`Kind::Composite::coef_range`] into the flat [`CompiledDag::coefs`] array.
#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq)]
pub struct Coef {
    /// Dense index of the input node in the parent [`CompiledDag`].
    pub input: u32,
    /// The coefficient applied to that input.
    pub coef: i32,
}

/// A compact, indexed snapshot of a [`Pldag`], optimised for fast propagation.
///
/// Build one with [`Pldag::dag`] (or directly with [`CompiledDag::compile`])
/// and then call [`CompiledDag::propagate`] (or
/// [`CompiledDag::propagate_with_scratch`] in hot loops).
///
/// The fields are public to support advanced consumers (custom traversals,
/// custom polyhedron encodings) but should be treated as a read-only,
/// internally-consistent representation. Mutating them out-of-band may
/// produce undefined behaviour at the API level.
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CompiledDag {
    /// Map from external string id to the dense `u32` index used internally.
    pub id_to_ix: HashMap<String, u32>,
    /// Reverse of [`id_to_ix`](Self::id_to_ix); used to render outputs back as strings.
    pub ix_to_id: Vec<String>,

    /// The [`Kind`] of each node, indexed by dense id.
    pub kind: Vec<Kind>,
    /// Flat backing storage for all composite-node coefficient terms.
    /// Each composite slices into this via its [`Kind::Composite::coef_range`].
    pub coefs: Vec<Coef>,

    /// Reverse-dependency lists: for each node, the dense ids of composites
    /// that consume it as an input. Used by propagation to wake parents.
    pub parents: Vec<Vec<u32>>,

    /// For each composite node, the number of inputs it has. Indexed by dense
    /// id; entries for primitive nodes are `0`.
    pub input_count: Vec<u32>,
}

/// Reusable scratch buffers for [`CompiledDag::propagate_with_scratch`].
///
/// Allocate once with [`Scratch::new`] and reuse across many calls — including
/// across DAGs of different sizes — to amortise the per-call allocation cost.
/// Backing capacity grows as needed and is never shrunk.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    known: Vec<bool>,
    values: Vec<Bound>,
    missing: Vec<u32>,
    assigned: Vec<bool>,
    queue: VecDeque<u32>,
}

impl Scratch {
    /// Creates a new empty scratch buffer set. The first call to
    /// [`CompiledDag::propagate_with_scratch`] will grow it to the DAG size.
    pub fn new() -> Self {
        Self::default()
    }

    /// Resize all buffers to length `n` and reset their contents. Backing
    /// capacity is reused; never shrunk.
    fn prepare(&mut self, n: usize) {
        self.known.clear();
        self.known.resize(n, false);
        self.values.clear();
        self.values.resize(n, (0, 0));
        self.missing.clear();
        self.missing.resize(n, 0);
        self.assigned.clear();
        self.assigned.resize(n, false);
        self.queue.clear();
        if self.queue.capacity() < n {
            self.queue.reserve(n - self.queue.capacity());
        }
    }
}

impl Default for CompiledDag {
    fn default() -> Self {
        Self::new()
    }
}

impl CompiledDag {
    /// Creates a new empty CompiledDag.
    pub fn new() -> Self {
        Self {
            id_to_ix: HashMap::new(),
            ix_to_id: Vec::new(),
            kind: Vec::new(),
            coefs: Vec::new(),
            parents: Vec::new(),
            input_count: Vec::new(),
        }
    }

    /// Inserts a node into the CompiledDag.
    ///
    /// Note: This is less efficient than batch compilation via `compile()` or `compile_optimized()`.
    /// For building large DAGs, prefer collecting nodes first and using `compile_optimized()`.
    pub fn insert(&mut self, id: String, node: Node) {
        // Check if node already exists
        if self.id_to_ix.contains_key(&id) {
            // For simplicity, we'll panic. A more sophisticated implementation could update in place.
            panic!("Node '{}' already exists in CompiledDag. Use compile_optimized for batch operations.", id);
        }

        let idx = self.kind.len() as u32;
        self.id_to_ix.insert(id.clone(), idx);
        self.ix_to_id.push(id);

        match node {
            Node::Primitive(bound) => {
                self.kind.push(Kind::Primitive { inherent: bound });
                self.parents.push(Vec::new());
                self.input_count.push(0);
            }
            Node::Composite(constraint) => {
                let start = self.coefs.len();
                let mut cnt = 0u32;

                for (input_id, coef) in constraint.coefficients.iter() {
                    if let Some(&input_ix) = self.id_to_ix.get(input_id) {
                        self.coefs.push(Coef { input: input_ix, coef: *coef });
                        self.parents[input_ix as usize].push(idx);
                        cnt += 1;
                    } else {
                        // Dependency doesn't exist yet - this is a limitation of incremental insertion
                        panic!("Dependency '{}' not found. Insert dependencies before composites.", input_id);
                    }
                }

                let end = self.coefs.len();
                self.kind.push(Kind::Composite {
                    bias_lo: constraint.bias.0,
                    coef_range: (start, end),
                });
                self.parents.push(Vec::new());
                self.input_count.push(cnt);
            }
        }
    }

    /// Optimized compilation from a Vec of (String, Node) pairs.
    /// This avoids redundant HashMap lookups and enables single-pass optimization.
    pub fn compile(mut nodes: Vec<(String, Node)>) -> Self {
        nodes.sort_by(|(a, _), (b, _)| a.cmp(b));
        let n = nodes.len();

        // Pre-allocate all structures with exact capacity
        let mut id_to_ix = HashMap::with_capacity(n);
        let mut ix_to_id = Vec::with_capacity(n);
        let mut kind = Vec::with_capacity(n);
        let mut coefs = Vec::new();
        let mut parents = vec![Vec::new(); n];
        let mut input_count = vec![0; n];

        // First pass: assign indices and populate primitives
        // We need indices known before we can resolve composite dependencies
        for (idx, (id, node)) in nodes.iter().enumerate() {
            id_to_ix.insert(id.clone(), idx as u32);
            ix_to_id.push(id.clone());

            match node {
                Node::Primitive(bound) => {
                    kind.push(Kind::Primitive { inherent: *bound });
                }
                Node::Composite(_) => {
                    // Placeholder, will fill in second pass
                    kind.push(Kind::Primitive { inherent: (0, 0) });
                }
            }
        }

        // Second pass: resolve composite dependencies now that all indices are known
        for (node_ix, (_id, node)) in nodes.iter().enumerate() {
            if let Node::Composite(c) = node {
                let start = coefs.len();
                let mut cnt = 0u32;

                for (input_id, coef) in c.coefficients.iter() {
                    if let Some(&input_ix) = id_to_ix.get(input_id) {
                        coefs.push(Coef { input: input_ix, coef: *coef });
                        parents[input_ix as usize].push(node_ix as u32);
                        cnt += 1;
                    }
                }

                let end = coefs.len();
                kind[node_ix] = Kind::Composite {
                    bias_lo: c.bias.0,
                    coef_range: (start, end),
                };
                input_count[node_ix] = cnt;
            }
        }

        Self {
            id_to_ix,
            ix_to_id,
            kind,
            coefs,
            parents,
            input_count,
        }
    }

    /// Converts the CompiledDag back to a HashMap<String, Node> representation.
    /// This is useful for legacy code that still expects the HashMap format.
    pub fn to_hashmap(&self) -> HashMap<String, Node> {
        let mut map = HashMap::with_capacity(self.kind.len());

        for (i, kind) in self.kind.iter().enumerate() {
            let id = self.ix_to_id[i].clone();
            let node = match kind {
                Kind::Primitive { inherent } => Node::Primitive(*inherent),
                Kind::Composite { bias_lo, coef_range } => {
                    let (start, end) = coef_range;
                    let coefficients: Vec<Coefficient> = self.coefs[*start..*end]
                        .iter()
                        .map(|coef| {
                            let input_id = self.ix_to_id[coef.input as usize].clone();
                            (input_id, coef.coef)
                        })
                        .collect();

                    Node::Composite(Constraint {
                        coefficients,
                        bias: (*bias_lo, *bias_lo),
                    })
                }
            };
            map.insert(id, node);
        }

        map
    }

    /// Gets the bound for a specific node ID.
    ///
    /// # Arguments
    /// * `id` - The node ID to look up
    ///
    /// # Returns
    /// `Some(Bound)` if the node exists (inherent bound for primitives, (0,1) for composites),
    /// `None` if the node doesn't exist in the DAG
    pub fn get(&self, id: &str) -> Option<Bound> {
        self.id_to_ix.get(id).map(|&idx| {
            let i = idx as usize;
            match self.kind[i] {
                Kind::Primitive { inherent } => inherent,
                Kind::Composite { .. } => (0, 1), // Composite nodes default to binary bounds
            }
        })
    }

    /// Propagate with new assignments.
    ///
    /// Pure compute over the compiled topology — does not mutate `self`.
    /// Allocates fresh scratch buffers each call; for hot loops, prefer
    /// [`CompiledDag::propagate_with_scratch`] to reuse buffers.
    pub fn propagate<K>(
        &self,
        assignments: impl IntoIterator<Item = (K, Bound)>,
    ) -> ComputeResult<HashMap<String, Bound>>
    where
        K: ToString,
    {
        let mut scratch = Scratch::new();
        self.propagate_with_scratch(assignments, &mut scratch)
    }

    /// Propagate with new assignments, reusing the storage in `scratch`.
    ///
    /// Equivalent to [`CompiledDag::propagate`], but reuses the buffers in
    /// `scratch` instead of allocating fresh ones. The same `Scratch` may be
    /// reused across many calls, including across DAGs of different sizes —
    /// it is grown as needed and reset on entry.
    pub fn propagate_with_scratch<K>(
        &self,
        assignments: impl IntoIterator<Item = (K, Bound)>,
        scratch: &mut Scratch,
    ) -> ComputeResult<HashMap<String, Bound>>
    where
        K: ToString,
    {
        let n = self.kind.len();
        scratch.prepare(n);

        let Scratch { known, values, missing, assigned, queue } = scratch;

        // missing starts as input_count for composites, 0 for primitives
        for i in 0..n {
            missing[i] = match self.kind[i] {
                Kind::Composite { .. } => self.input_count[i],
                Kind::Primitive { .. } => 0,
            };
        }

        for (k, b) in assignments.into_iter() {
            let s = k.to_string();
            if let Some(&ix) = self.id_to_ix.get(&s) {
                let i = ix as usize;
                values[i] = b;
                assigned[i] = true;
                queue.push_back(ix);
            }
        }

        // Enqueue all primitives (assigned or not) to start propagation
        for (i, _) in assigned.iter().enumerate().take(n) {
            if matches!(self.kind[i], Kind::Primitive { .. })
                && !assigned[i] {
                    queue.push_back(i as u32);
                }
        }

        // Main loop
        while let Some(ix) = queue.pop_front() {
            let i = ix as usize;
            if known[i] {
                continue;
            }

            match self.kind[i] {
                Kind::Primitive { inherent } => {
                    let out = if assigned[i] {
                        let b = values[i];
                        if b.0 < inherent.0 || b.1 > inherent.1 {
                            return Err(ComputeError::NodeOutOfBounds {
                                node_id: self.ix_to_id[i].clone(),
                                got_bound: b,
                                expected_bound: inherent,
                            });
                        }
                        b
                    } else {
                        inherent
                    };

                    values[i] = out;
                    known[i] = true;

                    // notify parents
                    for &p in &self.parents[i] {
                        let pi = p as usize;
                        if missing[pi] > 0 {
                            missing[pi] -= 1;
                            if missing[pi] == 0 {
                                queue.push_back(p);
                            }
                        }
                    }
                }

                Kind::Composite { bias_lo, coef_range: (start, end) } => {
                    // If this composite is ready, all its inputs should already be known.
                    // Compute quickly from flat coef array.
                    let mut sum: Bound = (0, 0);

                    for k in start..end {
                        let c = self.coefs[k];
                        let inp = c.input as usize;
                        if !known[inp] {
                            // not ready; queue its input and self again (rare if missing is correct)
                            queue.push_back(c.input);
                            queue.push_back(ix);
                            sum = (0, 0);
                            break;
                        }
                        sum = bound_add(sum, bound_multiply(c.coef, values[inp]));
                    }

                    // If we bailed out due to missing input, skip for now
                    // (use a flag instead of sum==(0,0) because that can be real)
                    let mut ok = true;
                    for k in start..end {
                        if !known[self.coefs[k].input as usize] {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }

                    let biased = bound_add(sum, (bias_lo, bias_lo));
                    let out = ((biased.0 >= 0) as i32, (biased.1 >= 0) as i32);

                    values[i] = out;
                    known[i] = true;

                    // notify parents
                    for &p in &self.parents[i] {
                        let pi = p as usize;
                        if missing[pi] > 0 {
                            missing[pi] -= 1;
                            if missing[pi] == 0 {
                                queue.push_back(p);
                            }
                        }
                    }
                }
            }
        }

        // Build output map (string ids -> bounds) for nodes that were computed.
        // If you only need the bounds for a subset, you can return something else.
        let mut out = HashMap::with_capacity(n);
        for i in 0..n {
            if known[i] {
                out.insert(self.ix_to_id[i].clone(), values[i]);
            }
        }
        Ok(out)
    }

    /// Propagate many assignment sets against the same DAG, reusing a single
    /// internal [`Scratch`] across iterations.
    ///
    /// Equivalent to calling [`CompiledDag::propagate`] in a loop, but allocates
    /// the working buffers once instead of per call. Fails fast: returns
    /// `Err` on the first set that produces a [`ComputeError`], discarding
    /// any earlier results.
    pub fn propagate_many<K, I, J>(
        &self,
        assignment_sets: J,
    ) -> ComputeResult<Vec<HashMap<String, Bound>>>
    where
        K: ToString,
        I: IntoIterator<Item = (K, Bound)>,
        J: IntoIterator<Item = I>,
    {
        let mut scratch = Scratch::new();
        assignment_sets
            .into_iter()
            .map(|a| self.propagate_with_scratch(a, &mut scratch))
            .collect()
    }
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
    pub storage: Arc<dyn NodeStoreTrait>,
    validate_coeffs: bool,
    allow_empty_constraints: bool,
}

impl Default for Pldag {
    fn default() -> Self {
        Self::new()
    }
}

impl Pldag {
    /// Creates a new empty PL-DAG.
    ///
    /// # Returns
    /// A new `Pldag` instance with no nodes
    pub fn new() -> Pldag {
        Pldag {
            storage: Arc::new(NodeStore::new(Arc::new(InMemoryStore::new()))),
            validate_coeffs: true,
            allow_empty_constraints: false,
        }
    }

    /// Creates a new PL-DAG backed by a caller-supplied [`NodeStoreTrait`].
    ///
    /// Use this to plug in a database-backed or otherwise-customised storage
    /// layer instead of the default in-memory store.
    pub fn new_custom(storage: Arc<dyn NodeStoreTrait>) -> Pldag {
        Pldag {
            storage,
            validate_coeffs: true,
            allow_empty_constraints: false,
        }
    }

    /// Sets whether to validate that coefficients exists on insertion, guaranteeing a valid DAG, at the
    /// cost of extra lookups on insertion.
    /// Default value is true.
    pub fn set_validate_coeffs(mut self, validate_coeffs: bool) -> Self {
        self.validate_coeffs = validate_coeffs;
        self
    }

    /// Sets whether to allow constraints with no coefficient variables.
    ///
    /// An empty constraint reduces to the constant `bias >= 0`, which is either
    /// a tautology (when `bias >= 0`) or unsatisfiable (when `bias < 0`). Such
    /// constraints carry no information about other variables and are typically
    /// the result of an upstream bug (e.g. accidentally passing an empty
    /// reference list to `set_and` / `set_atleast`), so this is disabled by
    /// default and `set_gelineq` returns [`ModelError::EmptyConstraint`].
    ///
    /// Enable this if you intentionally rely on the old behaviour of building
    /// degenerate constraints (e.g. `set_and(vec![])` as a tautology).
    /// Default value is false.
    pub fn set_allow_empty_constraints(mut self, allow_empty_constraints: bool) -> Self {
        self.allow_empty_constraints = allow_empty_constraints;
        self
    }

    /// Full tightening over the DAG given initial assumptions.
    ///
    /// - `dag`: mapping from node name to Node (Primitive / Composite)
    /// - `assumptions`: mapping from node name to assumed bound,
    ///   e.g. "A" -> (1,1) means boolean node A is TRUE.
    ///
    /// Returns an HashMap of final bounds for all nodes (primitives + composite booleans).
    pub fn tighten(
        dag: &CompiledDag,
        assumptions: &HashMap<String, Bound>,
    ) -> ComputeResult<HashMap<String, Bound>> {
        let n = dag.kind.len();

        // 1. Initialize bounds for all nodes
        let mut values: Vec<Bound> = Vec::with_capacity(n);
        for kind in dag.kind.iter() {
            let initial = match kind {
                Kind::Primitive { inherent } => *inherent,
                Kind::Composite { .. } => (0, 1), // boolean: unknown in [0,1]
            };
            values.push(initial);
        }

        // 2. Apply assumptions by intersecting bounds
        for (name, assumed) in assumptions.iter() {
            if let Some(&idx) = dag.id_to_ix.get(name) {
                let i = idx as usize;
                values[i] = intersect_bounds(values[i], *assumed);
            }
        }

        // 3. Fixed-point iteration: propagate until no more changes
        let max_iters = 100;
        let mut iter = 0;

        loop {
            iter += 1;
            if iter > max_iters {
                return Err(ComputeError::MaxIterationsExceeded { max_iters });
            }

            let mut changed = false;

            // For each composite node
            for (node_idx, kind) in dag.kind.iter().enumerate() {
                let (bias_lo, coef_range) = match kind {
                    Kind::Composite { bias_lo, coef_range } => (bias_lo, coef_range),
                    Kind::Primitive { .. } => continue,
                };

                // Current boolean bound of this constraint node
                let bool_bound = values[node_idx];
                let old_bool_bound = bool_bound;

                // (a) Evaluate constraint and intersect with current boolean bound
                let (start, end) = coef_range;
                let eval = evaluate_constraint(&dag.coefs[*start..*end], *bias_lo, &values, dag);
                let new_bool_bound = intersect_bounds(bool_bound, eval);

                if new_bool_bound != old_bool_bound {
                    values[node_idx] = new_bool_bound;
                    changed = true;
                }

                // (b) If now forced TRUE or FALSE, propagate
                let (lb, ub) = new_bool_bound;
                if lb == 1 && ub == 1 {
                    // Constraint is TRUE
                    if tighten_constraint_true_compiled(&dag.coefs[*start..*end], *bias_lo, &mut values) {
                        changed = true;
                    }
                } else if lb == 0 && ub == 0 {
                    // Constraint is FALSE: use negated constraint
                    let neg_bias = -bias_lo - 1;
                    let neg_coefs: Vec<Coef> = dag.coefs[*start..*end]
                        .iter()
                        .map(|c| Coef { input: c.input, coef: -c.coef })
                        .collect();
                    if tighten_constraint_true_compiled(&neg_coefs, neg_bias, &mut values) {
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Convert Vec<Bound> to HashMap<String, Bound>
        let mut result = HashMap::with_capacity(n);
        for (i, bound) in values.into_iter().enumerate() {
            result.insert(dag.ix_to_id[i].clone(), bound);
        }

        Ok(result)
    }

    /// Returns a smaller [`CompiledDag`] with the given variables substituted by constants.
    ///
    /// Each node listed in `fixed` is removed from the DAG, and every composite
    /// that referenced it has the substituted contribution folded into its
    /// `bias_lo` term. Use this to specialise a generic model for a specific
    /// scenario before propagating or solving.
    ///
    /// # Arguments
    /// * `dag` — the source DAG to reduce.
    /// * `fixed` — mapping from node id to the integer value to substitute.
    pub fn reduce(
        dag: &CompiledDag,
        fixed: &HashMap<String, i32>,
    ) -> ComputeResult<CompiledDag> {
        let mut nodes: Vec<(String, Node)> = Vec::new();

        'nodes: for (node_idx, kind) in dag.kind.iter().enumerate() {
            let node_id = &dag.ix_to_id[node_idx];

            // Drop nodes that are fixed
            if fixed.contains_key(node_id) {
                continue 'nodes;
            }

            match kind {
                Kind::Primitive { inherent } => {
                    nodes.push((node_id.clone(), Node::Primitive(*inherent)));
                }
                Kind::Composite { bias_lo, coef_range } => {
                    let (start, end) = coef_range;
                    let mut new_coefficients: Vec<(String, i32)> = Vec::new();
                    let mut new_bias = (*bias_lo, *bias_lo);

                    for coef in dag.coefs[*start..*end].iter() {
                        let var_name = &dag.ix_to_id[coef.input as usize];

                        if let Some(&fixed_val) = fixed.get(var_name) {
                            // Substitute fixed value into bias
                            let contribution = bound_multiply(coef.coef, (fixed_val, fixed_val));
                            new_bias = bound_add(new_bias, contribution);
                        } else {
                            // Keep variable in constraint
                            new_coefficients.push((var_name.clone(), coef.coef));
                        }
                    }

                    // If constant after substitution, drop it too (it's fixed now)
                    if new_coefficients.is_empty() {
                        let (lb, ub) = new_bias;
                        // constraint is bias >= 0
                        if lb >= 0 || ub < 0 {
                            continue 'nodes;
                        }
                        // If ambiguous interval, keep it (rare)
                    }

                    nodes.push((
                        node_id.clone(),
                        Node::Composite(Constraint {
                            coefficients: new_coefficients,
                            bias: new_bias,
                        }),
                    ));
                }
            }
        }

        Ok(CompiledDag::compile(nodes))
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
        dag: &CompiledDag,
        assignments: impl IntoIterator<Item = (K, Bound)>,
    ) -> ComputeResult<Assignment>
    where
        K: ToString,
    {
        dag.propagate(assignments)
    }

    /// Computes ranks for all nodes in the DAG.
    ////
    /// Ranks represent the longest distance from any root node to each node.
    ///// # Arguments
    /// * `dag` - mapping from node name to Node (Primitive / Composite)
    ///
    /// # Returns
    /// A HashMap of node IDs to their corresponding ranks
    pub fn ranks(cd: &CompiledDag) -> ComputeResult<HashMap<ID, usize>> {
        let n = cd.kind.len();
        let mut ranks: Vec<usize> = vec![0; n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Calculate in-degrees (how many parents each node has)
        for i in 0..n {
            for &parent_idx in &cd.parents[i] {
                in_degree[parent_idx as usize] += 1;
            }
        }

        // Topological sort using Kahn's algorithm with rank calculation
        let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

        // Start with nodes that have no parents (in-degree = 0)
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
                ranks[i] = 0;
            }
        }

        let mut processed = 0;

        while let Some(node_idx) = queue.pop_front() {
            processed += 1;

            // For each parent of this node
            for &parent_idx in &cd.parents[node_idx] {
                let parent = parent_idx as usize;

                // Update parent's rank to be max of (current rank, child rank + 1)
                ranks[parent] = ranks[parent].max(ranks[node_idx] + 1);

                // Decrease in-degree
                in_degree[parent] -= 1;

                // If all children have been processed, add parent to queue
                if in_degree[parent] == 0 {
                    queue.push_back(parent);
                }
            }
        }

        // Check for cycles
        if processed != n {
            // Find a node that wasn't processed (part of the cycle)
            let cycle_node_idx = in_degree.iter().position(|&deg| deg > 0).unwrap_or(0);
            return Err(ComputeError::CycleDetected {
                node_id: cd.ix_to_id[cycle_node_idx].clone(),
            });
        }

        // Convert Vec<usize> to HashMap<String, usize>
        let mut result = HashMap::with_capacity(n);
        for (i, rank) in ranks.into_iter().enumerate() {
            result.insert(cd.ix_to_id[i].clone(), rank);
        }

        Ok(result)
    }

    /// Returns the node ids of `dag` in topological order.
    ///
    /// Producers (primitives, plus composites whose inputs are already settled)
    /// appear before their consumers. The pre-built `dependency_map` —
    /// typically obtained from [`Pldag::dependency_map`] — is supplied
    /// separately to avoid recomputing it across calls.
    pub fn topological_sort(
        dag: &HashMap<ID, Node>,
        dependency_map: &HashMap<ID, Vec<ID>>,
    ) -> ComputeResult<Vec<ID>> {
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

        while let Some(node_id) = queue.pop() {
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

    /// Builds the child-id map for a raw `(id -> Node)` view of a DAG.
    ///
    /// For each node, the returned map lists the ids of nodes it depends on:
    /// composite nodes list their coefficient inputs; primitive nodes map to
    /// an empty list. This is the input expected by [`Pldag::topological_sort`].
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


    /// Extracts a sub-DAG containing all nodes reachable from the given roots.
    /// NOTE: if roots is empty, returns the entire DAG.
    ///
    /// # Arguments
    /// * `roots` - Vector of root node IDs to start the sub-DAG extraction
    ///
    /// # Returns
    /// A HashMap of node IDs to their corresponding nodes in the sub-DAG
    pub async fn sub_dag(&self, roots: Vec<ID>) -> ModelResult<CompiledDag> {
        // If no roots, return entire DAG
        if roots.is_empty() {
            return self.dag().await;
        }

        let mut queue: Vec<String> = roots;

        // Use a HashSet for visited tracking (faster than HashMap::contains_key)
        let mut visited: HashSet<String> = HashSet::new();

        // Accumulate nodes in order of discovery - this preserves some locality
        let mut nodes: Vec<(String, Node)> = Vec::new();

        while !queue.is_empty() {
            // Batch fetch incoming edges for current batch
            let all_incoming = self.storage.get_nodes(&queue).await?;

            // Check that we got all nodes from queue
            for node_id in queue.iter() {
                if !all_incoming.contains_key(node_id) {
                    return Err(ModelError::NodeNotFound { node_id: node_id.to_string() });
                }
            }

            let mut next_batch = Vec::new();

            for (input_id, incoming) in all_incoming.into_iter() {
                // Skip if already visited
                if !visited.insert(input_id.clone()) {
                    continue;
                }

                // Add node directly to our ordered list
                nodes.push((input_id.clone(), incoming.clone()));

                // If composite, enqueue its dependencies
                if let Node::Composite(constraint) = &incoming {
                    for (coef_id, _) in constraint.coefficients.iter() {
                        if !visited.contains(coef_id) && !next_batch.contains(coef_id) {
                            next_batch.push(coef_id.clone());
                        }
                    }
                }
            }
            queue = next_batch;
        }

        // Use optimized compilation directly from the ordered node list
        Ok(CompiledDag::compile(nodes))
    }

    /// Compiles the entire model into a [`CompiledDag`].
    ///
    /// This is the recommended starting point for evaluation: build your
    /// model with `set_*` methods, call `dag()` once, then propagate or
    /// solve against the resulting compact representation as many times as
    /// you like. See [`Pldag::sub_dag`] to compile only a subset.
    pub async fn dag(&self) -> ModelResult<CompiledDag> {
        let all_nodes = self.storage.get_all_nodes().await?.into_iter().collect::<Vec<_>>();
        Ok(CompiledDag::compile(all_nodes))
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
    pub fn to_sparse_polyhedron(
        cd: &CompiledDag,
        double_binding: bool,
    ) -> ComputeResult<SparsePolyhedron> {
        let ncols = cd.kind.len();

        // Pre-count composites + NNZ to reserve capacity
        let mut comp_count = 0usize;
        let mut nnz = 0usize;

        for k in &cd.kind {
            if let Kind::Composite { coef_range, .. } = *k {
                comp_count += 1;
                let inputs = coef_range.1 - coef_range.0;
                nnz += 1 + inputs; // row for phi -> pi
                if double_binding {
                    nnz += 1 + inputs; // row for pi -> phi (via neg phi OR pi)
                }
            }
        }

        let nrows = comp_count * if double_binding { 2 } else { 1 };

        let mut a_matrix = SparseIntegerMatrix::new();
        a_matrix.rows.reserve(nnz);
        a_matrix.cols.reserve(nnz);
        a_matrix.vals.reserve(nnz);

        let mut b_vector: Vec<i32> = Vec::with_capacity(nrows);

        let mut row_i: usize = 0;

        for (ix, k) in cd.kind.iter().enumerate() {
            let Kind::Composite { bias_lo, coef_range } = *k else { continue };

            let ki = ix; // pi column index is the node index itself
            let (start, end) = coef_range;

            // --- Compute ib_phi = dot(bounds_of_inputs) excluding bias ---
            // bounds_of_inputs: primitive -> inherent; composite -> (0,1)
            let mut ib: Bound = (0, 0);
            for j in start..end {
                let c = cd.coefs[j];
                let inp = c.input as usize;

                let bnd = match cd.kind[inp] {
                    Kind::Primitive { inherent } => inherent,
                    Kind::Composite { .. } => (0, 1),
                };

                let prod = bound_multiply(c.coef, bnd);
                ib = bound_add(ib, prod);
            }

            // d_pi = max(|ib(phi)|) + |bias|
            let d_pi = std::cmp::max(ib.0.abs(), ib.1.abs()) + bias_lo.abs();

            // Row: -d_pi*pi + sum(coef_i * x_i) >= -(bias + d_pi)
            a_matrix.rows.push(row_i);
            a_matrix.cols.push(ki);
            a_matrix.vals.push(-d_pi);

            for j in start..end {
                let c = cd.coefs[j];
                a_matrix.rows.push(row_i);
                a_matrix.cols.push(c.input as usize);
                a_matrix.vals.push(c.coef);
            }

            let b_phi = bias_lo + d_pi;
            b_vector.push(-b_phi);

            if double_binding {
                // Avoid building negate(phi) and avoid extra dot:
                //
                // phi_prim.bias0 = -bias_lo - 1
                // d_phi_prim = max(|ib(phi)|)   (same as max(|ib(neg phi)|))
                // pi_coef = d_phi_prim - phi_prim.bias0 = d_phi_prim + bias_lo + 1

                let d_phi_prim = std::cmp::max(ib.0.abs(), ib.1.abs());
                let pi_coef = d_phi_prim + bias_lo + 1;

                a_matrix.rows.push(row_i + 1);
                a_matrix.cols.push(ki);
                a_matrix.vals.push(pi_coef);

                // negate coefficients
                for j in start..end {
                    let c = cd.coefs[j];
                    a_matrix.rows.push(row_i + 1);
                    a_matrix.cols.push(c.input as usize);
                    a_matrix.vals.push(-c.coef);
                }

                let phi_prim_bias0 = -bias_lo - 1;
                b_vector.push(-phi_prim_bias0);

                row_i += 1;
            }

            row_i += 1;
        }

        a_matrix.shape = (row_i, ncols);

        Ok(SparsePolyhedron {
            a: a_matrix,
            b: b_vector,
            // columns are already in index order
            columns: cd.ix_to_id.clone(),
            // primitives have inherent bounds; composites are boolean
            column_bounds: cd
                .kind
                .iter()
                .map(|k| match *k {
                    Kind::Primitive { inherent } => inherent,
                    Kind::Composite { .. } => (0, 1),
                })
                .collect(),
        })
    }

    /// Converts the PL-DAG to a sparse polyhedron with default settings.
    ///
    /// Convenience method that calls `to_sparse_polyhedron` with all options enabled:
    /// double_binding=true, integer_constraints=true, fixed_constraints=true.
    ///
    /// # Returns
    /// A `SparsePolyhedron` with full constraint encoding
    pub fn to_sparse_polyhedron_default(cd: &CompiledDag) -> ComputeResult<SparsePolyhedron> {
        Self::to_sparse_polyhedron(cd, true)
    }

    /// Converts the PL-DAG to a dense polyhedron.
    ///
    /// # Arguments
    /// * `double_binding` - If true, creates bidirectional implications
    ///
    /// # Returns
    /// A `DensePolyhedron` representing the DAG constraints
    pub fn to_dense_polyhedron(cd: &CompiledDag, double_binding: bool) -> ComputeResult<DensePolyhedron> {
        // Convert to sparse polyhedron first
        let sparse_polyhedron = Self::to_sparse_polyhedron(cd, double_binding)?;
        // Convert sparse to dense polyhedron
        Ok(sparse_polyhedron.into())
    }

    /// Converts the PL-DAG to a dense polyhedron with default settings.
    ///
    /// # Returns
    /// A `DensePolyhedron` with all constraint options enabled
    pub fn to_dense_polyhedron_default(cd: &CompiledDag) -> ComputeResult<DensePolyhedron> {
        Self::to_dense_polyhedron(cd, true)
    }

    /// Retrieves all primitive variables from the given PL-DAG roots.
    ///
    /// # Returns
    /// An `HashMap` mapping variable IDs to their corresponding `Bound` objects
    pub fn get_primitives(dag: &CompiledDag) -> Vec<String> {
        dag
            .kind
            .iter()
            .enumerate()
            .filter_map(|(i, kind)| {
                if let Kind::Primitive { inherent: _ } = kind {
                    Some(dag.ix_to_id[i].clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Retrieves all composite constraints from the PL-DAG.
    ///
    /// # Returns
    /// An `HashMap` mapping constraint IDs to their corresponding `Constraint` objects
    pub fn get_composites(dag: &CompiledDag) -> Vec<String> {
        dag
            .kind
            .iter()
            .enumerate()
            .filter_map(|(i, kind)| {
                if let Kind::Composite { bias_lo:_, coef_range:_ } = kind {
                    Some(dag.ix_to_id[i].clone())
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
    pub async fn get_node(&self, id: &str) -> ModelResult<Option<Node>> {
        Ok(self.storage.get_nodes(&[id.to_string()]).await?.get(id).cloned())
    }

    /// Retrieves multiple nodes by their IDs.
    /// If a requested ID does not exist, it will simply be omitted from the result.
    ///
    /// # Arguments
    /// * `ids` - A slice of unique identifiers for the nodes to retrieve
    /// # Returns
    /// A `HashMap<String, Node>` mapping each requested ID to its corresponding Node.
    pub async fn get_nodes(&self, ids: &[String]) -> ModelResult<HashMap<String, Node>> {
        Ok(self.storage.get_nodes(ids).await?)
    }

    /// Deletes a node from the PL-DAG by its ID.
    ///
    /// # Arguments
    /// * `id` - The unique identifier of the node to delete
    pub async fn delete_node(&self, id: &str) -> ModelResult<()> {
        let parents = self.storage.get_parent_ids(&[id.to_string()]).await?;
        if let Some(parents) = parents.get(id) {
            if !parents.is_empty() {
                return Err(ModelError::NodeReferenced {
                    node_id: id.to_string(),
                    referencing_nodes: parents.clone(),
                });
            }
        }
        self.storage.delete(id).await?;
        Ok(())
    }

    /// Creates a primitive (leaf) variable with the specified bounds.
    ///
    /// Primitive variables represent the base variables in the DAG and have
    /// no dependencies on other nodes.
    ///
    /// # Arguments
    /// * `id` - Unique identifier for the variable
    /// * `bound` - The allowed range (min, max) for this variable
    pub async fn set_primitive(&self, id: &str, bound: Bound) -> ModelResult<ID> {
        self.storage.set_node(id, Node::Primitive(bound)).await?;
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
    pub async fn set_primitives<K>(&self, ids: impl IntoIterator<Item = K>, bound: Bound) -> ModelResult<Vec<ID>>
    where
        K: ToString,
    {
        let unique_ids: IndexSet<String> = ids.into_iter().map(|k| k.to_string()).collect();
        let primitives: Vec<(&str, &Bound)> = unique_ids
            .iter()
            .map(|id| (id.as_str(), &bound))
            .collect();

        self.storage.set_primitives(&primitives).await?;

        Ok(unique_ids.into_iter().collect())
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
    pub async fn set_gelineq<K>(
        &self,
        coefficient_variables: impl IntoIterator<Item = (K, i32)>,
        bias: i32,
    ) -> ModelResult<ID>
    where
        K: ToString,
    {   
        // Ensure coefficients have unique keys by summing duplicate values
        let mut unique_coefficients: HashMap<ID, i32> = HashMap::new();
        for (key, value) in coefficient_variables {
            *unique_coefficients.entry(key.to_string()).or_insert(0) += value;
        }

        // Drop entries whose summed coefficient is zero: they contribute
        // nothing to the inequality, would leak irrelevant variables into the
        // node hash, and would otherwise bypass the empty-constraint guard
        // below.
        unique_coefficients.retain(|_, coef| *coef != 0);

        // Require at least one coefficient to prevent empty constraints, unless
        // the model has been configured to allow them (the old behaviour).
        if unique_coefficients.is_empty() && !self.allow_empty_constraints {
            return Err(ModelError::EmptyConstraint);
        }

        // Check that all coefficient IDs exist in storage
        if self.validate_coeffs {
            for coef_id in unique_coefficients.keys() {
                if !self.storage.node_exists(coef_id).await? {
                    return Err(ModelError::NodeNotFound {
                        node_id: coef_id.clone(),
                    });
                }
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
        self.storage.set_node(&id, Node::Composite(constraint)).await?;

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
    pub async fn set_atleast<K>(
        &self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> ModelResult<ID>
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, 1)), -value).await
    }

    /// Like [`Pldag::set_atleast`], but the threshold is itself a node reference.
    ///
    /// Encodes `sum(references) >= value`, where `value` is the id of an
    /// existing node whose current bound is used as the threshold. Useful
    /// for expressing data-driven constraints where the right-hand side is
    /// not known statically.
    pub async fn set_atleast_ref<K, V>(
        &self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> ModelResult<ID>
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
        ).await
    }

    /// Creates an "at most" constraint: sum(variables) <= value.
    ///
    /// # Arguments
    /// * `references` - Iterator of variable IDs to sum
    /// * `value` - Maximum allowed sum
    ///
    /// # Returns
    /// The unique ID assigned to this constraint, or an error if any reference doesn't exist
    pub async fn set_atmost<K>(
        &self,
        references: impl IntoIterator<Item = K>,
        value: i32,
    ) -> ModelResult<ID>
    where
        K: ToString,
    {
        self.set_gelineq(references.into_iter().map(|x| (x, -1)), value).await
    }

    /// Like [`Pldag::set_atmost`], but the cap is itself a node reference.
    ///
    /// Encodes `sum(references) <= value`, where `value` is the id of an
    /// existing node whose current bound is used as the cap.
    pub async fn set_atmost_ref<K, V>(
        &self,
        references: impl IntoIterator<Item = K>,
        value: V,
    ) -> ModelResult<ID>
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
        ).await
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
    pub async fn set_equal<K, I>(
        &self,
        references: I,
        value: i32,
    ) -> ModelResult<ID>
    where
        K: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast(references.clone(), value).await?;
        let lb = self.set_atmost(references, value).await?;
        self.set_and(vec![ub, lb]).await
    }

    /// Like [`Pldag::set_equal`], but the target sum is itself a node reference.
    ///
    /// Encodes `sum(references) == value` by conjoining `set_atleast_ref` and
    /// `set_atmost_ref` against the same `value` node.
    pub async fn set_equal_ref<K, V, I>(
        &self,
        references: I,
        value: V,
    ) -> ModelResult<ID>
    where
        K: ToString,
        V: ToString,
        I: IntoIterator<Item = K> + Clone,
    {
        let ub = self.set_atleast_ref(references.clone(), value.to_string()).await?;
        let lb = self.set_atmost_ref(references, value).await?;
        self.set_and(vec![ub, lb]).await
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
    pub async fn set_and<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let length = unique_references.len();
        self.set_atleast(unique_references, length as i32).await
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
    pub async fn set_or<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atleast(unique_references, 1).await
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
    pub async fn set_optional<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let len = unique_references.len() as i32;
        self.set_atmost(unique_references, len).await
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
    pub async fn set_nand<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let length = unique_references.len();
        self.set_atmost(unique_references, length as i32 - 1).await
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
    pub async fn set_nor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atmost(unique_references, 0).await
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
    pub async fn set_not<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        self.set_atmost(unique_references, 0).await
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
    pub async fn set_xor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let atleast = self.set_or(unique_references.clone()).await?;
        let atmost = self.set_atmost(unique_references, 1).await?;
        self.set_and(vec![atleast, atmost]).await
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
    pub async fn set_xnor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>
    where
        K: ToString,
    {
        let unique_references: IndexSet<String> =
            references.into_iter().map(|x| x.to_string()).collect();
        let atleast = self.set_atleast(unique_references.clone(), 2).await?;
        let atmost = self.set_atmost(unique_references, 0).await?;
        self.set_or(vec![atleast, atmost]).await
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
    pub async fn set_imply<C, Q>(&self, condition: C, consequence: Q) -> ModelResult<ID>
    where
        C: ToString,
        Q: ToString,
    {
        let not_condition = self.set_not(vec![condition.to_string()]).await?;
        self.set_or(vec![not_condition, consequence.to_string()]).await
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
    pub async fn set_equiv<L, R>(&self, lhs: L, rhs: R) -> ModelResult<ID>
    where
        L: ToString,
        R: ToString,
    {
        // Convert to strings first to avoid type mismatches
        let lhs_str = lhs.to_string();
        let rhs_str = rhs.to_string();

        let imply_lr = self.set_and(vec![lhs_str.clone(), rhs_str.clone()]).await?;
        let imply_rl = self.set_not(vec![rhs_str, lhs_str]).await?;
        self.set_or(vec![imply_lr, imply_rl]).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Create a helper function that generates all primitive combinations
    // for a given PLDAG model, propagates them, and compares against the
    // corresponding polyhedron evaluations.
    async fn primitive_combinations(model: &Pldag) -> Vec<HashMap<String, i32>> {
        let dag = model.dag().await.unwrap();
        let primitives = Pldag::get_primitives(&dag);
        let mut combinations: Vec<HashMap<String, i32>> = Vec::new();

        let num_primitives = primitives.len();
        let num_combinations = 1 << num_primitives; // 2^n combinations

        for i in 0..num_combinations {
            let mut combo = HashMap::new();
            for (j, prim_name) in primitives.iter().enumerate() {
                let value = if (i & (1 << j)) != 0 { 1 } else { 0 };
                combo.insert(prim_name.clone(), value);
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
    async fn evaluate_model_polyhedron(model: &Pldag, poly: &DensePolyhedron, root: &String) {
        for combo in primitive_combinations(model).await {
            // build an HashMap<&str,Bound> as propagate expects
            let interp = combo
                .iter()
                .map(|(k, &v)| (k.as_str(), (v, v)))
                .collect::<HashMap<&str, Bound>>();

            // what the DAG says the root can be
            let prop = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
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

    #[tokio::test]
    async fn test_compiled_dag_sorts(){
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let id = model.set_and(vec!["x", "y", "z"]).await.unwrap();

        let ids: Vec<_> = ["x", "y", "z", &id].iter().map(|s| s.to_string()).collect();
        let nodes_result = model.get_nodes(&ids).await;
        assert!(nodes_result.is_ok(), "Failed to get nodes: {:?}", nodes_result.err());
        let nodes_first: Vec<_> = nodes_result.unwrap().into_iter().collect();
        let nodes_second = vec![nodes_first[1].clone(), nodes_first[0].clone(), nodes_first[3].clone(), nodes_first[2].clone()]; // shuffle the order

        let dag_first = CompiledDag::compile(nodes_first);
        let dag_second = CompiledDag::compile(nodes_second);
        
        assert_eq!(dag_first.kind, dag_second.kind, "Compiled DAGs kind list differ");
        assert_eq!(dag_first.ix_to_id, dag_second.ix_to_id, "Compiled DAGs ix_to_id list differ");
        assert_eq!(dag_first.coefs, dag_second.coefs, "Compiled DAGs coefficients differ");
    }

    #[tokio::test]
    async fn test_propagate() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let root = model.set_and(vec!["x", "y"]).await.unwrap();
        let dag = model.dag().await.unwrap();
        let result = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = HashMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_xor(vec!["x", "y", "z"]).await.unwrap();
        let dag = model.dag().await.unwrap();
        let result = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut assignments = HashMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = HashMap::new();
        assignments.insert("x", (0, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let mut assignments = HashMap::new();
        assignments.insert("x", (0, 0));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (0, 0));
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        // Test propagation to specific root only and check that the others are not included in the result
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let or_1 = model.set_or(vec!["x", "z"]).await.unwrap();
        let or_2 = model.set_or(vec!["y", "z"]).await.unwrap();
        let or_3 = model.set_or(vec!["x", "y"]).await.unwrap();
        let root = model.set_and(vec![or_1.clone(), or_2.clone(), or_3.clone()]).await.unwrap();
        let mut assignments = HashMap::new();
        assignments.insert("x", (1, 1));
        let sub_dag = model.sub_dag(vec![or_1.clone()]).await.unwrap();
        let result = Pldag::propagate_dag(&sub_dag, assignments).unwrap();
        assert_eq!(result.get("x").unwrap(), &(1, 1));
        assert_eq!(result.get(&or_1).unwrap(), &(1, 1));
        assert!(!result.contains_key(&or_2));
        assert!(!result.contains_key(&or_3));
        assert!(!result.contains_key(&root));
    }

    #[tokio::test]
    async fn test_propagate_with_scratch_reuse_across_dags() {
        // Build two DAGs of different sizes and run propagate_with_scratch
        // against the same Scratch buffer; results must match a fresh
        // CompiledDag::propagate call. This exercises the grow-only resize
        // path and verifies no state leaks between calls.
        let small = Pldag::new();
        let _ = small.set_primitive("a", (0, 1)).await;
        let _ = small.set_primitive("b", (0, 1)).await;
        let small_root = small.set_and(vec!["a", "b"]).await.unwrap();
        let small_dag = small.dag().await.unwrap();

        let large = Pldag::new();
        let _ = large.set_primitive("p", (0, 1)).await;
        let _ = large.set_primitive("q", (0, 1)).await;
        let _ = large.set_primitive("r", (0, 1)).await;
        let _ = large.set_primitive("s", (0, 1)).await;
        let or_pq = large.set_or(vec!["p", "q"]).await.unwrap();
        let or_rs = large.set_or(vec!["r", "s"]).await.unwrap();
        let large_root = large.set_and(vec![or_pq, or_rs]).await.unwrap();
        let large_dag = large.dag().await.unwrap();

        let mut scratch = Scratch::new();

        // First call: small DAG, both primitives = 1.
        let mut a1 = HashMap::new();
        a1.insert("a", (1, 1));
        a1.insert("b", (1, 1));
        let with = small_dag
            .propagate_with_scratch(a1.clone(), &mut scratch)
            .unwrap();
        let baseline = small_dag.propagate(a1).unwrap();
        assert_eq!(with, baseline);
        assert_eq!(with.get(&small_root).unwrap(), &(1, 1));

        // Second call: larger DAG (grows the buffers), partial assignment.
        let mut a2 = HashMap::new();
        a2.insert("p", (1, 1));
        a2.insert("r", (1, 1));
        let with = large_dag
            .propagate_with_scratch(a2.clone(), &mut scratch)
            .unwrap();
        let baseline = large_dag.propagate(a2).unwrap();
        assert_eq!(with, baseline);
        assert_eq!(with.get(&large_root).unwrap(), &(1, 1));

        // Third call: back to the small DAG with different assignment.
        // This exercises the case where buffer capacity exceeds the DAG
        // size — the prefix must be cleanly reset.
        let mut a3 = HashMap::new();
        a3.insert("a", (0, 0));
        a3.insert("b", (1, 1));
        let with = small_dag
            .propagate_with_scratch(a3.clone(), &mut scratch)
            .unwrap();
        let baseline = small_dag.propagate(a3).unwrap();
        assert_eq!(with, baseline);
        assert_eq!(with.get(&small_root).unwrap(), &(0, 0));
    }

    #[tokio::test]
    async fn test_propagate_many_matches_repeated_propagate() {
        // propagate_many must produce the same result, in order, as repeated
        // standalone propagate calls — and propagate the first error eagerly.
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let root = model.set_or(vec!["x", "y"]).await.unwrap();
        let dag = model.dag().await.unwrap();

        let sets: Vec<Vec<(&str, Bound)>> = vec![
            vec![("x", (1, 1))],
            vec![("x", (0, 0)), ("y", (0, 0))],
            vec![("y", (1, 1))],
        ];

        let many = dag.propagate_many(sets.clone()).unwrap();
        let one_by_one: Vec<_> = sets
            .iter()
            .map(|s| dag.propagate(s.clone()).unwrap())
            .collect();
        assert_eq!(many, one_by_one);
        assert_eq!(many[0][&root], (1, 1));
        assert_eq!(many[1][&root], (0, 0));
        assert_eq!(many[2][&root], (1, 1));

        // Fail-fast: an out-of-bounds assignment surfaces as Err on that set,
        // and downstream sets are not returned.
        let bad_sets: Vec<Vec<(&str, Bound)>> = vec![
            vec![("x", (1, 1))],
            vec![("x", (2, 2))], // out of (0, 1)
            vec![("y", (1, 1))],
        ];
        let err = dag.propagate_many(bad_sets).unwrap_err();
        assert!(matches!(err, ComputeError::NodeOutOfBounds { .. }));
    }

    /// XOR already covered; test the OR gate
    #[tokio::test]
    async fn test_propagate_or_gate() {
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let or_root = model.set_or(vec!["a", "b"]).await.unwrap();

        // No assignment: both inputs full [0,1], output [0,1]
        let dag = model.dag().await.unwrap();
        let res = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("a", (1, 1));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("a", (0, 0));
        interp.insert("b", (0, 0));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("b", (0, 0));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[tokio::test]
    async fn test_propagate_not_gate() {
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let not_root = model.set_not(vec!["p"]).await.unwrap();

        // no assignment ⇒ [0,1]
        let dag = model.dag().await.unwrap();
        let res = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (0, 0));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (1, 1));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&not_root], (0, 0));
    }

    #[tokio::test]
    async fn test_to_polyhedron_and() {
        let m = Pldag::new();
        let _ = m.set_primitive("x", (0, 1)).await;
        let _ = m.set_primitive("y", (0, 1)).await;
        let root = m.set_and(vec!["x", "y"]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    #[tokio::test]
    async fn test_to_polyhedron_or() {
        let m = Pldag::new();
        let _ = m.set_primitive("a", (0, 1)).await;
        let _ = m.set_primitive("b", (0, 1)).await;
        let _ = m.set_primitive("c", (0, 1)).await;
        let root = m.set_or(vec!["a", "b", "c"]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    #[tokio::test]
    async fn test_to_polyhedron_not() {
        let m = Pldag::new();
        let _ = m.set_primitive("p", (0, 1)).await;
        let root = m.set_not(vec!["p"]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    #[tokio::test]
    async fn test_to_polyhedron_xor() {
        let m = Pldag::new();
        let _ = m.set_primitive("x", (0, 1)).await;
        let _ = m.set_primitive("y", (0, 1)).await;
        let _ = m.set_primitive("z", (0, 1)).await;
        let root = m.set_xor(vec!["x", "y", "z"]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    #[tokio::test]
    async fn test_to_polyhedron_nested() {
        // Build a small two‐level circuit:
        //   w = AND(x,y),  v = OR(w, NOT(z))
        let m = Pldag::new();
        let _ = m.set_primitive("x", (0, 1)).await;
        let _ = m.set_primitive("y", (0, 1)).await;
        let _ = m.set_primitive("z", (0, 1)).await;

        let w = m.set_and(vec!["x", "y"]).await.unwrap();
        let nz = m.set_not(vec!["z"]).await.unwrap();
        let v = m.set_or(vec![w.clone(), nz.clone()]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &v).await;
    }

    /// Nested/composed AND then XOR:
    ///   w = AND(x,y);  v = XOR(w,z)
    #[tokio::test]
    async fn test_propagate_nested_composite() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;

        let w = model.set_and(vec!["x", "y"]).await.unwrap();
        let v = model.set_xor(vec![w.clone(), "z".into()]).await.unwrap();

        // no assignment: everything [0,1]
        let dag = model.dag().await.unwrap();
        let res = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        for var in &["x", "y", "z"] {
            assert_eq!(res[*var], (0, 1), "{}", var);
        }
        assert_eq!(res[&w], (0, 1));
        assert_eq!(res[&v], (0, 1));

        // x=1,y=1,z=0 ⇒ w=1,v=1
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("x", (1, 1));
        interp.insert("y", (1, 1));
        interp.insert("z", (0, 0));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (1, 1));
        interp.insert("z", (1, 1));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("x", (0, 0));
        interp.insert("y", (0, 0));
        interp.insert("z", (0, 0));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (0, 0));
    }

    /// If you ever get an inconsistent assignment (out‐of‐bounds for a primitive),
    /// propagate should leave it as given (or you could choose to clamp / panic)—here
    /// we simply check that nothing blows up.
    #[tokio::test]
    async fn test_propagate_out_of_bounds_should_crash() {
        let model = Pldag::new();
        let _ = model.set_primitive("u", (0, 1)).await;

        let mut interp = HashMap::<&str, Bound>::new();
        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        interp.insert("u", (5, 5));
        let res = Pldag::propagate_dag(&model.dag().await.unwrap(), interp);

        // Assert that we did get an error
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_to_polyhedron() {
        async fn evaluate_model_polyhedron(model: &Pldag, polyhedron: &DensePolyhedron, root: &String) {
            for combination in primitive_combinations(model).await {
                let assignments = combination
                    .iter()
                    .map(|(k, &v)| (k.as_str(), (v, v)))
                    .collect::<HashMap<&str, Bound>>();
                let model_prop = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
                let model_eval = *model_prop.get(root).unwrap();
                let mut assumption = HashMap::new();
                assumption.insert(root.clone(), 1);
                let assumed_polyhedron = polyhedron.assume(&assumption);
                let assumed_poly_eval = assumed_polyhedron.evaluate(&model_prop);
                assert_eq!(assumed_poly_eval, model_eval);
            }
        }

        let model: Pldag = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_xor(vec!["x", "y", "z"]).await.unwrap();
        let polyhedron: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root).await;

        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let root = model.set_and(vec!["x", "y"]).await.unwrap();
        let polyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root).await;

        let model: Pldag = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_xor(vec!["x", "y", "z"]).await.unwrap();
        let polyhedron = Pldag::to_sparse_polyhedron_default(&model.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root).await;
    }

    /// Single‐operand composites should act as identity: root == operand
    #[tokio::test]
    async fn test_to_polyhedron_single_operand_identity() {
        // AND(x) == x
        {
            let m = Pldag::new();
            let _ = m.set_primitive("x", (0, 1)).await;
            let root = m.set_and::<&str>(vec!["x"]).await.unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root).await;
        }
        // OR(y) == y
        {
            let m = Pldag::new();
            let _ = m.set_primitive("y", (0, 1)).await;
            let root = m.set_or(vec!["y"]).await.unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root).await;
        }
        // XOR(z) == z
        {
            let m = Pldag::new();
            let _ = m.set_primitive("z", (0, 1)).await;
            let root = m.set_xor(vec!["z"]).await.unwrap();
            let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
            evaluate_model_polyhedron(&m, &poly, &root).await;
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[tokio::test]
    async fn test_to_polyhedron_duplicate_operands_and() {
        let m = Pldag::new();
        let _ = m.set_primitive("x", (0, 1)).await;
        let root = m.set_and(vec!["x", "x"]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    /// Deeply nested 5‐level chain:
    ///    w1 = AND(a,b)
    ///    w2 = OR(w1,c)
    ///    w3 = XOR(w2,d)
    ///    root = NOT(w3)
    #[tokio::test]
    async fn test_to_polyhedron_deeply_nested_chain() {
        let m = Pldag::new();
        // primitives a,b,c,d,e  (e unused but shows extra var)
        for &v in &["a", "b", "c", "d", "e"] {
            let _ = m.set_primitive(v, (0, 1)).await;
        }
        let a = "a";
        let b = "b";
        let c = "c";
        let d = "d";

        let w1 = m.set_and(vec![a, b]).await.unwrap();
        let w2 = m.set_or(vec![w1.clone(), c.to_string()]).await.unwrap();
        let w3 = m.set_xor(vec![w2.clone(), d.to_string()]).await.unwrap();
        let root = m.set_not(vec![w3.clone()]).await.unwrap();
        let poly: DensePolyhedron = Pldag::to_sparse_polyhedron_default(&m.sub_dag(vec![]).await.unwrap()).unwrap().into();
        evaluate_model_polyhedron(&m, &poly, &root).await;
    }

    #[tokio::test]
    async fn test_print_dense_matrix() {
        let mut matrix = DenseIntegerMatrix::new(3, 3);
        matrix.data[0][0] = 1;
        matrix.data[0][2] = 2;
        matrix.data[1][0] = 3;
        matrix.data[2][2] = 4;

        let output = format!("{}", matrix);
        let expected = "  1   0   2 \n  3   0   0 \n  0   0   4 \n";
        assert_eq!(output, expected);
    }

    #[tokio::test]
    async fn test_print_sparse_matrix() {
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

    #[tokio::test]
    async fn test_equiv() {
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        let equiv = model.set_equiv("p", "q").await.unwrap();
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        let _ = model.set_primitive("p", (1, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        let _ = model.set_primitive("p", (1, 1)).await;
        let _ = model.set_primitive("q", (0, 0)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));

        let _ = model.set_primitive("p", (0, 0)).await;
        let _ = model.set_primitive("q", (0, 0)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        let _ = model.set_primitive("p", (1, 1)).await;
        let _ = model.set_primitive("q", (1, 1)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));
    }

    #[tokio::test]
    async fn test_imply() {
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        let equiv = model.set_imply("p", "q").await.unwrap();
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 1));

        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (1, 1)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(1, 1));

        let _ = model.set_primitive("p", (1, 1)).await;
        let _ = model.set_primitive("q", (0, 0)).await;
        let dag = model.dag().await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(propagated.get(&equiv).unwrap(), &(0, 0));
    }

    #[tokio::test]
    async fn test_node_out_of_bounds_error() {
        // If we propagate a primitive with a bound that is outside its predefined range,
        // we should get a NodeOutOfBounds error.
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (2, 2)); // Out of bounds
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), interp);
        assert!(matches!(result, Err(ComputeError::NodeOutOfBounds { .. })));
        
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (-1, 2)); // Out of bounds
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), interp);
        assert!(matches!(result, Err(ComputeError::NodeOutOfBounds { .. })));
        
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (-1, -1)); // Out of bounds
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), interp);
        assert!(matches!(result, Err(ComputeError::NodeOutOfBounds { .. })));
        
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let mut interp = HashMap::<&str, Bound>::new();
        interp.insert("p", (1, 1)); // Not out of bounds
        let result = Pldag::propagate_dag(&model.dag().await.unwrap(), interp);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_node_not_found_error_when_propagate() {
        // If we propagate a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]).await;
        assert!(matches!(result, Err(ModelError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[tokio::test]
    async fn test_node_not_found_error_when_sub_dag() {
        // If we create a sub-dag with a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]).await;
        assert!(matches!(result, Err(ModelError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[tokio::test]
    async fn test_node_not_found_error_when_to_polyhedron() {
        // If we convert to a polyhedron with a variable that does not exist in the model,
        // we should get a NodeNotFound error.
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        // set_and will return an error when 'r' does not exist
        let result = model.set_and(vec!["p", "q", "r"]).await;
        assert!(matches!(result, Err(ModelError::NodeNotFound { node_id } ) if node_id == "r"));
    }

    #[tokio::test]
    async fn test_empty_constraint_error_when_set_gelineq_has_no_coefficients() {
        // set_gelineq with no coefficient/variable pairs must reject the call,
        // since an empty linear inequality is not a meaningful constraint.
        let model = Pldag::new();
        let result = model.set_gelineq(Vec::<(&str, i32)>::new(), 0).await;
        assert!(matches!(result, Err(ModelError::EmptyConstraint)));
    }

    #[tokio::test]
    async fn test_zero_summed_coefficients_treated_as_empty() {
        // Duplicate coefficients that sum to zero (e.g. (x, 1) + (x, -1)) must
        // collapse to an empty coefficient set so the empty-constraint guard
        // fires and irrelevant variables don't leak into the node hash.
        let model = Pldag::new();
        model.set_primitive("x", (0, 1)).await.unwrap();

        let result = model
            .set_gelineq(vec![("x", 1), ("x", -1)], 0)
            .await;
        assert!(
            matches!(result, Err(ModelError::EmptyConstraint)),
            "coefficients summing to zero must be filtered before the empty check",
        );
    }

    #[tokio::test]
    async fn test_zero_summed_coefficients_do_not_affect_hash() {
        // After filtering out zero-valued entries, a constraint built with
        // (x, 1) + (x, -1) + (y, 1) must hash to the same id as one built with
        // just (y, 1) — the cancelled variable should not leak into the id.
        let model = Pldag::new();
        model.set_primitive("x", (0, 1)).await.unwrap();
        model.set_primitive("y", (0, 1)).await.unwrap();

        let with_cancel = model
            .set_gelineq(vec![("x", 1), ("x", -1), ("y", 1)], 0)
            .await
            .expect("constraint with cancelling coefficients should succeed");
        let without_cancel = model
            .set_gelineq(vec![("y", 1)], 0)
            .await
            .expect("plain constraint should succeed");

        assert_eq!(
            with_cancel, without_cancel,
            "zero-summed coefficients must not influence the constraint id",
        );
    }

    #[tokio::test]
    async fn test_empty_constraint_allowed_when_configured() {
        // When the model is configured to allow empty constraints, set_gelineq
        // should accept zero coefficients and produce a constant constraint.
        let model = Pldag::new().set_allow_empty_constraints(true);

        // bias >= 0 → tautology: the resulting node propagates to (1, 1).
        let taut_id = model
            .set_gelineq(Vec::<(&str, i32)>::new(), 0)
            .await
            .expect("empty constraint should be allowed when configured");

        let dag = model.dag().await.expect("dag should compile");
        let values = Pldag::tighten(&dag, &HashMap::new()).expect("tighten");
        assert_eq!(values.get(&taut_id), Some(&(1, 1)));

        // bias < 0 → unsatisfiable: the resulting node propagates to (0, 0).
        let contra_id = model
            .set_gelineq(Vec::<(&str, i32)>::new(), -1)
            .await
            .expect("empty constraint should be allowed when configured");

        let dag = model.dag().await.expect("dag should compile");
        let values = Pldag::tighten(&dag, &HashMap::new()).expect("tighten");
        assert_eq!(values.get(&contra_id), Some(&(0, 0)));
    }

    #[tokio::test]
    async fn binary_cardinality_all_forced_to_one_when_true() {
        // x, y, z in [0,1]
        // A: x + y + z - 3 >= 0  <=>  x + y + z >= 3
        // A is assumed TRUE → x = y = z = 1

        let mut dag = CompiledDag::new();
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

    #[tokio::test]
    async fn binary_cardinality_false_does_not_tighten() {
        // x, y, z in [0,1]
        // A: x + y + z >= 3
        // A is FALSE → x + y + z <= 2
        // With [0,1] for all, this does NOT force any individual variable.

        let mut dag = CompiledDag::new();
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

    #[tokio::test]
    async fn chained_constraints_do_not_tighten_in_this_case() {
        // x, y, z ∈ [0,3]
        // A: x + y - 3 >= 0  <=>  x + y >= 3
        // B: y + z - 3 >= 0  <=>  y + z >= 3
        // Assume A = TRUE and B = TRUE.
        //
        // Interval reasoning alone cannot tighten x, y, or z here.

        let mut dag = CompiledDag::new();
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

    #[tokio::test]
    async fn composite_as_boolean_in_another_constraint() {
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

        let mut dag = CompiledDag::new();
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

    #[tokio::test]
    async fn test_tighten_bounds_on_an_xor() {
        // A = B + C >= 2
        // B = x + y + z >= 1
        // C = -x -y -z >= -1

        // Assume A is TRUE, and x = (1, 1) then y and z must be (0, 0)
        let mut dag = CompiledDag::new();
        dag.insert("x".into(), prim(0, 1));
        dag.insert("y".into(), prim(0, 1));
        dag.insert("z".into(), prim(0, 1));
        dag.insert("B".into(), cons(vec![("x", 1), ("y", 1), ("z", 1)], -1));
        dag.insert("C".into(), cons(vec![("x", -1), ("y", -1), ("z", -1)], 1));
        dag.insert("A".into(), cons(vec![("B", 1), ("C", 1)], -2));
        let mut assumptions: HashMap<String, (i32, i32)> = HashMap::new();
        assumptions.insert("A".to_string(), (1, 1));
        assumptions.insert("x".to_string(), (1, 1));
        let values = Pldag::tighten(&dag, &assumptions).unwrap();
        assert_eq!(values.get("y"), Some(&(0, 0)));
        assert_eq!(values.get("z"), Some(&(0, 0)));
    }

    #[tokio::test]
    async fn test_simple_sub_dag_with_xor() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_xor(vec!["x", "y", "z"]).await.unwrap();
        let sub_dag = model.sub_dag(vec![root.clone()]).await.unwrap();
        assert!(sub_dag.get(&root).is_some());
    }

    #[tokio::test]
    async fn test_delete_node_should_succeed() {
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let and_node = model.set_and(vec!["a", "b"]).await.unwrap();
        let delete_result = model.delete_node(&and_node).await;
        assert!(delete_result.is_ok());
        assert!(model.get_node(&and_node).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_primitives_should_succeed() {
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let delete_result_a = model.delete_node("a").await;
        let delete_result_b = model.delete_node("b").await;
        assert!(delete_result_a.is_ok());
        assert!(delete_result_b.is_ok());
        assert!(model.get_node("a").await.unwrap().is_none());
        assert!(model.get_node("b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_composite_then_delete_primitives_should_succeed() {
        let model = Pldag::new();
        let _ = model.set_primitive("a".into(), (0, 1)).await;
        let _ = model.set_primitive("b".into(), (0, 1)).await;
        let and_node = model.set_and(vec!["a", "b"]).await.unwrap();
        let delete_result_composite = model.delete_node(&and_node).await;
        let delete_result_a = model.delete_node(&"a").await;
        let delete_result_b = model.delete_node(&"b").await;
        assert!(delete_result_composite.is_ok());
        assert!(delete_result_a.is_ok());
        assert!(delete_result_b.is_ok());
        assert!(model.get_node(&and_node).await.unwrap().is_none());
        assert!(model.get_node(&"a").await.unwrap().is_none());
        assert!(model.get_node(&"b").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_delete_node_should_fail_for_still_having_references() {
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let and_node = model.set_and(vec!["a", "b"]).await.unwrap();
        model.set_or(vec![and_node.clone(), "a".into()]).await.unwrap();
        let delete_result = model.delete_node(&and_node).await;
        assert!(delete_result.is_err());
    }

    #[tokio::test]
    async fn test_compute_ranks() {

        // Simple case: a and b are rank 0, and (a AND b) is rank 1
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let and_node = model.set_and(vec!["a", "b"]).await.unwrap();
        model.set_or(vec![and_node.clone(), "a".into()]).await.unwrap();
        let ranks = Pldag::ranks(&model.dag().await.unwrap()).unwrap();
        assert_eq!(ranks.get("a"), Some(&0));
        assert_eq!(ranks.get("b"), Some(&0));
        assert_eq!(ranks.get(&and_node), Some(&1));

        // More complex case 1
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let and_node = model.set_and(vec!["x", "y"]).await.unwrap();
        let not_node = model.set_not(vec![and_node.clone()]).await.unwrap();
        model.set_xor(vec![not_node.clone(), "x".into()]).await.unwrap();
        let ranks = Pldag::ranks(&model.sub_dag(vec![]).await.unwrap()).unwrap();
        assert_eq!(ranks.get("x"), Some(&0));
        assert_eq!(ranks.get("y"), Some(&0));
        assert_eq!(ranks.get(&and_node), Some(&1));
        assert_eq!(ranks.get(&not_node), Some(&2));

        // More complex case 2
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        let equiv_node = model.set_equiv("p", "q").await.unwrap();
        let imply_node = model.set_imply("p", "q").await.unwrap();
        model.set_or(vec![equiv_node.clone(), imply_node.clone()]).await.unwrap();
        let ranks = Pldag::ranks(&model.sub_dag(vec![]).await.unwrap()).unwrap();
        assert_eq!(ranks.get("p"), Some(&0));
        assert_eq!(ranks.get("q"), Some(&0));
        assert_eq!(ranks.get(&equiv_node), Some(&2));
        assert_eq!(ranks.get(&imply_node), Some(&2));
    }

    #[tokio::test]
    async fn test_reduce() {
        // A = B + C + D >= 3
        // B = x + y >= 2
        // C = y + z >= 2
        // D = a >= 1

        // and we give that a = 1, D = 1
        // This should return a new DAG like:

        // A = B + C >= 2
        // B = x + y >= 2
        // C = y + z >= 2
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let d = model.set_and(vec!["a"]).await.unwrap();
        let c = model.set_and(vec!["y", "z"]).await.unwrap();
        let b = model.set_and(vec!["x", "y"]).await.unwrap();
        let a = model.set_and(vec![b.clone(), c.clone(), d.clone()]).await.unwrap();
        let mut fixed: HashMap<String, i32> = HashMap::new();
        fixed.insert("a".to_string(), 1);
        fixed.insert(d.to_string(), 1);
        let dag = model.sub_dag(vec![a.clone()]).await.unwrap();
        let reduced_dag = Pldag::reduce(&dag, &fixed).unwrap();
        // check that d and a is not in reduced daG
        assert!(reduced_dag.get(&d).is_none());
        assert!(reduced_dag.get("a").is_none());
        // check that a, b, c, x, y, z are in reduced DaG
        assert!(reduced_dag.get(&a).is_some());
        assert!(reduced_dag.get(&b).is_some());
        assert!(reduced_dag.get(&c).is_some());
        assert!(reduced_dag.get("x").is_some());
        assert!(reduced_dag.get("y").is_some());
        assert!(reduced_dag.get("z").is_some());

        // Propagate the reduced DAG with x = 1, y = 1, z = 1, which should satisfy A = 1, B = 1, C = 1
        let mut assignments = HashMap::new();
        assignments.insert("x", (1, 1));
        assignments.insert("y", (1, 1));
        assignments.insert("z", (1, 1));
        let propagated = Pldag::propagate_dag(&reduced_dag, assignments).unwrap();
        assert_eq!(propagated.get(&a).unwrap(), &(1, 1));
        assert_eq!(propagated.get(&b).unwrap(), &(1, 1));
        assert_eq!(propagated.get(&c).unwrap(), &(1, 1));
    }

    #[tokio::test]
    async fn test_propagte_with_id_not_in_dag_shoul_pass() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_and(vec!["x", "y", "z"]).await.unwrap();
        let mut assignments = HashMap::new();
        assignments.insert("x".to_string(), (1, 1));
        assignments.insert("y".to_string(), (1, 1));
        assignments.insert("z".to_string(), (1, 1));
        let dag = model.sub_dag(vec![root.clone()]).await.unwrap();
        let propagated = Pldag::propagate_dag(&dag, assignments.clone()).unwrap();
        assert_eq!(propagated.get(&root).unwrap(), &(1, 1));

        let propagated = Pldag::propagate_dag(&model.dag().await.unwrap(), assignments).unwrap();
        assert_eq!(propagated.get(&root).unwrap(), &(1, 1));
    }

    // ========================================================================
    // CompiledDag propagate tests
    // ========================================================================

    #[tokio::test]
    async fn test_compiled_dag_propagate() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let root = model.set_and(vec!["x", "y"]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        let result = compiled.propagate(Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let assignments = vec![("x", (1, 1)), ("y", (1, 1))];
        let result = compiled.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let root = model.set_xor(vec!["x", "y", "z"]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        let result = compiled.propagate(Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let assignments = vec![("x", (1, 1)), ("y", (1, 1)), ("z", (1, 1))];
        let result = compiled.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let assignments = vec![("x", (0, 1)), ("y", (1, 1)), ("z", (1, 1))];
        let result = compiled.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(0, 0));

        let assignments = vec![("x", (0, 0)), ("y", (1, 1)), ("z", (0, 0))];
        let result = compiled.propagate(assignments).unwrap();
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        // Test propagation to specific root only and check that the others are not included in the result
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;
        let or_1 = model.set_or(vec!["x", "z"]).await.unwrap();
        let or_2 = model.set_or(vec!["y", "z"]).await.unwrap();
        let or_3 = model.set_or(vec!["x", "y"]).await.unwrap();
        let root = model.set_and(vec![or_1.clone(), or_2.clone(), or_3.clone()]).await.unwrap();

        let sub_dag = model.sub_dag(vec![or_1.clone()]).await.unwrap();

        let assignments = vec![("x", (1, 1))];
        let result = sub_dag.propagate(assignments).unwrap();
        assert_eq!(result.get("x").unwrap(), &(1, 1));
        assert_eq!(result.get(&or_1).unwrap(), &(1, 1));
        assert!(!result.contains_key(&or_2));
        assert!(!result.contains_key(&or_3));
        assert!(!result.contains_key(&root));
    }

    #[tokio::test]
    async fn test_compiled_dag_propagate_or_gate() {
        let model = Pldag::new();
        let _ = model.set_primitive("a", (0, 1)).await;
        let _ = model.set_primitive("b", (0, 1)).await;
        let or_root = model.set_or(vec!["a", "b"]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        // No assignment: both inputs full [0,1], output [0,1]
        let res = compiled.propagate(Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let assignments = vec![("a", (1, 1))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let assignments = vec![("a", (0, 0)), ("b", (0, 0))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let assignments = vec![("b", (0, 0))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&or_root], (0, 1));
    }

    #[tokio::test]
    async fn test_compiled_dag_propagate_not_gate() {
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let not_root = model.set_not(vec!["p"]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        // no assignment ⇒ [0,1]
        let res = compiled.propagate(Vec::<(&str, Bound)>::new()).unwrap();
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let assignments = vec![("p", (0, 0))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let assignments = vec![("p", (1, 1))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&not_root], (0, 0));
    }

    #[tokio::test]
    async fn test_compiled_dag_propagate_nested_composite() {
        let model = Pldag::new();
        let _ = model.set_primitive("x", (0, 1)).await;
        let _ = model.set_primitive("y", (0, 1)).await;
        let _ = model.set_primitive("z", (0, 1)).await;

        let w = model.set_and(vec!["x", "y"]).await.unwrap();
        let v = model.set_xor(vec![w.clone(), "z".into()]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        // no assignment: everything [0,1]
        let res = compiled.propagate(Vec::<(&str, Bound)>::new()).unwrap();
        for var in &["x", "y", "z"] {
            assert_eq!(res[*var], (0, 1), "{}", var);
        }
        assert_eq!(res[&w], (0, 1));
        assert_eq!(res[&v], (0, 1));

        // x=1,y=1,z=0 ⇒ w=1,v=1
        let assignments = vec![("x", (1, 1)), ("y", (1, 1)), ("z", (0, 0))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&w], (1, 1));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let assignments = vec![("x", (0, 0)), ("y", (1, 1)), ("z", (1, 1))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (1, 1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let assignments = vec![("x", (0, 0)), ("y", (0, 0)), ("z", (0, 0))];
        let res = compiled.propagate(assignments).unwrap();
        assert_eq!(res[&w], (0, 0));
        assert_eq!(res[&v], (0, 0));
    }

    #[tokio::test]
    async fn test_compiled_dag_propagate_out_of_bounds_should_crash() {
        let model = Pldag::new();
        let _ = model.set_primitive("u", (0, 1)).await;

        let compiled = model.sub_dag(vec![]).await.unwrap();

        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        let assignments = vec![("u", (5, 5))];
        let res = compiled.propagate(assignments);

        // Assert that we did get an error
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_compiled_dag_propagate_node_not_found_error_when_propagate() {
        // If we propagate a variable that does not exist in the compiled dag,
        // the assignment should just be ignored (it won't crash, but won't affect anything)
        let model = Pldag::new();
        let _ = model.set_primitive("p", (0, 1)).await;
        let _ = model.set_primitive("q", (0, 1)).await;
        let root = model.set_and(vec!["p", "q"]).await.unwrap();

        let compiled = model.sub_dag(vec![]).await.unwrap();

        // Propagate with a nonexistent variable "r"
        let assignments = vec![("p", (1, 1)), ("q", (1, 1)), ("r", (1, 1))];
        let result = compiled.propagate(assignments).unwrap();

        // Should still work, just ignoring "r"
        assert_eq!(result.get(&root).unwrap(), &(1, 1));
    }
}