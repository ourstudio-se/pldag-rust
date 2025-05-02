use std::collections::HashSet;
use std::collections::{HashMap, VecDeque, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};
use bimap::BiMap;
use std::ops::RangeInclusive;
use itertools::Itertools;
use indexmap::{IndexMap, IndexSet};

type Bound = (i64, i64);
type ID = String;

// Function to create a hash from a reference to Vec<(String, i64)> and an i64
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

fn bound_fixed(b: Bound) -> bool {
    // Check if the bound is fixed
    b.0 == b.1
}

fn bound_bool(b: Bound) -> bool {
    // Check if the bound is boolean
    b.0 == 0 && b.1 == 1
}

fn bound_add(b1: Bound, b2: Bound) -> Bound {
    return (b1.0 + b2.0, b1.1 + b2.1);
}

fn bound_multiply(k: i64, b: Bound) -> Bound {
    if k < 0 {
        return (k*b.1, k*b.0);
    } else {
        return (k*b.0, k*b.1);
    }
}

fn bound_span(b: Bound) -> i64 {
    // Calculate the span of the bound
    return (b.1 - b.0).abs();
}

struct SparseIntegerMatrix {
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<i64>,
    shape: (usize, usize),
}

struct DenseIntegerMatrix {
    data: Vec<Vec<i64>>,
    shape: (usize, usize),
}

impl DenseIntegerMatrix {
    fn new(rows: usize, cols: usize) -> DenseIntegerMatrix {
        DenseIntegerMatrix {
            data: vec![vec![0; cols]; rows],
            shape: (rows, cols),
        }
    }

    fn dot_product(&self, vector: &Vec<i64>) -> Vec<i64> {
        let mut result = vec![0; self.shape.0];
        for i in 0..self.shape.0 {
            for j in 0..self.shape.1 {
                result[i] += self.data[i][j] * vector[j];
            }
        }
        result
    }
}

struct DensePolyhedron {
    A: DenseIntegerMatrix,
    b: Vec<i64>,
    columns: Vec<String>,
    integer_columns: Vec<String>,
}

impl DensePolyhedron {
    fn to_vector(&self, from_interpretation: &HashMap<String, i64>) -> Vec<i64> {
        let mut vector: Vec<i64> = vec![0; self.columns.len()];
        for (index, v) in from_interpretation
            .iter()
            .filter_map(|(k, v)| self.columns.iter().position(|col| col == k).map(|index| (index, v)))
        {
            vector[index] = *v;
        }
        vector
    }

    fn assume(&self, values: &HashMap<String, i64>) -> DensePolyhedron {
        // 1) Make mutable copies of everything
        let mut new_A_data   = self.A.data.clone();     // Vec<Vec<i64>>
        let mut new_b        = self.b.clone();          // Vec<i64>
        let mut new_columns  = self.columns.clone();    // Vec<String>
        let mut new_int_cols = self.integer_columns.clone();  // Vec<String>

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

    fn evaluate(&self, interpretation: &IndexMap<String, Bound>) -> Bound {
        let mut lower_bounds = HashMap::new();
        let mut upper_bounds = HashMap::new();
        for (key, bound) in interpretation {
            lower_bounds.insert(key.clone(), bound.0);
            upper_bounds.insert(key.clone(), bound.1);
        }

        let lower_result = self.A.dot_product(&self.to_vector(&lower_bounds))
            .iter()
            .zip(&self.b)
            .all(|(a, b)| a >= b);

        let upper_result = self.A.dot_product(&self.to_vector(&upper_bounds))
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
    fn new() -> SparseIntegerMatrix {
        SparseIntegerMatrix {
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
            shape: (0, 0),
        }
    }
}

struct SparsePolyhedron {
    // ...
    A: SparseIntegerMatrix,
    b: Vec<i64>,
    columns: Vec<String>,
    integer_columns: Vec<String>,
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

type Coefficient = (String, i64);
struct Constraint {
    coefficients: Vec<Coefficient>,
    bias: Bound,
}

impl Constraint {

    fn dot(&self, values: &IndexMap<String, Bound>) -> Bound {
        self.coefficients.iter().fold((0, 0), |acc, (key, coeff)| {
            let bound = values.get(key).unwrap_or(&(0, 0));
            let (min, max) = bound_multiply(*coeff, *bound);
            (acc.0 + min, acc.1 + max)
        })
    }

    fn evaluate(&self, values: &IndexMap<String, Bound>) -> Bound {
        let bound = self.dot(values);
        return (
            (bound.0 + self.bias.0 >= 0) as i64,
            (bound.1 + self.bias.1 >= 0) as i64
        )
    }

    fn negate(&self) -> Constraint {
        Constraint {
            coefficients: self.coefficients.iter().map(|(key, val)| {
                (key.clone(), -val)
            }).collect(),
            bias: (
                -self.bias.0-1,
                 -self.bias.1-1
            ),
        }
    }
}
enum BoolExpression {
    Composite(Constraint),
    Primitive(Bound),
}

struct Pldag {
    // Contraints matrix map
    _amat: IndexMap<String, BoolExpression>,
    // Aliases map
    _amap: BiMap<String, String>,
}

impl Pldag {

    fn new() -> Pldag {
        Pldag {
            _amat: IndexMap::new(),
            _amap: BiMap::new(),
        }
    }

    /// Compute for each node the full set of nodes reachable from it.
    pub fn transitive_dependencies(&self) -> HashMap<ID, HashSet<ID>> {
        // memo: node -> its already-computed reachables
        let mut memo: HashMap<String, HashSet<String>> = HashMap::new();
        let mut result: HashMap<String, HashSet<String>> = HashMap::new();

        for key in self._amat.keys() {
            // compute (or fetch) and store
            let deps = self._collect_deps(key, &mut memo);
            result.insert(key.clone(), deps);
        }

        result
    }

    /// Helper: return (and memoize) the set of all nodes reachable from `node`.
    fn _collect_deps(&self, node: &ID, memo: &mut HashMap<ID, HashSet<ID>>) -> HashSet<ID> {
        // if we’ve already done this node, just clone the result
        if let Some(cached) = memo.get(node) {
            return cached.clone();
        }

        let mut deps = HashSet::new();

        if let Some(expr) = self._amat.get(node) {
            if let BoolExpression::Composite(constraint) = expr {
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

    fn primitive_combinations(&self) -> impl Iterator<Item = HashMap<ID, i64>> {
        // 1. Pull out [(var_name, (low, high)), …]
        let primitives: Vec<(String, (i64, i64))> = self._amat
            .iter()
            .filter_map(|(key, expr)| {
                if let BoolExpression::Primitive(bound) = expr {
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

    pub fn get_id(&self, from_alias: &String) -> ID {
        // Check if the alias is present in the aliases map
        if let Some(id) = self._amap.get_by_left(from_alias) {
            return id.clone();
        }
        // If not, return the alias itself
        return from_alias.clone();
    }

    pub fn get_alias(&self, from_id: &ID) -> String {
        // Check if the id is present in the aliases map
        if let Some(alias) = self._amap.get_by_right(from_id) {
            return alias.clone();
        }
        // If not, return the id itself
        return from_id.clone();
    }

    pub fn check_combination(&self, interpretation: &IndexMap<ID, Bound>) -> IndexMap<ID, Bound> {

        let mut result= interpretation.clone();

        // Fill result with the primitive variable bounds
        for (key, value) in self._amat.iter() {
            if !result.contains_key(key) {
                if let BoolExpression::Primitive(bound) = value {
                    result.insert(key.clone(), *bound);
                }
            }
        }

        // S = All composites that 
        // (1) have only primitive variables as input and
        // (2) are not present in `result/interpretation`
        let mut S: VecDeque<String> = self._amat
            .iter()
            .filter(|(key, constraint)| {
                match constraint {
                    BoolExpression::Composite(composite) => {
                    composite.coefficients.iter().all(|x| {
                        match self._amat.get(&x.0) {
                        Some(BoolExpression::Primitive(_)) => true,
                        _ => false
                        }
                    }) && !result.contains_key(&key.to_string())
                    },
                    BoolExpression::Primitive(_) => false,
                }
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
            match self._amat.get(&s) {
                Some(BoolExpression::Composite(composite)) => {
                    result.insert(
                        s.clone(), 
                        composite.evaluate(&result)
                    );
        
                    // Now find all composites having `current_id` as its input.
                    // And if that composite's children are in `result`, then add to S.
                    let incoming = self._amat
                        .iter()
                        .filter(|(key, sub_constraint)| {
                            !result.contains_key(&key.to_string()) && match sub_constraint {
                                BoolExpression::Composite(sub_composite) => {
                                    sub_composite.coefficients.iter().any(|x| x.0 == s)
                                },
                                _ => false
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
                },
                _ => {}
            }
        }

        return result;
    }
    
    pub fn check_combination_default(&self) -> IndexMap<ID, Bound> {
        let interpretation: IndexMap<String, Bound> = self._amap.iter().filter_map(|(key, value)| {
            if let Some(bound) = self._amat.get(key) {
                if let BoolExpression::Primitive(bound) = bound {
                    Some((value.clone(), *bound))
                } else {
                    None
                }
            } else {
                None
            }
        }).collect();
        self.check_combination(&interpretation)
    }
    
    pub fn score_combination_batch(&self, interpretation: &IndexMap<ID, Bound>, weight_sets: &Vec<&IndexMap<ID, f64>>) -> Vec<HashMap<ID, (f64, f64)>> {
        let trans_deps = self.transitive_dependencies();
        let mut result = Vec::new();
        for weights in weight_sets {
            let mut local_result = HashMap::new();
    
            for (variable, dependencies) in trans_deps.iter() {
                let variable_bounds = interpretation.get(variable.as_str()).unwrap_or(&(0, 1));
                if dependencies.len() > 0 {
                    let dependency_weighted_bound = dependencies.iter()
                        .filter_map(|dep| interpretation.get(dep).map(|bound| {
                            let weight = weights.get(dep).unwrap_or(&0.0);
                            (
                                (bound.0 as f64) * weight,
                                (bound.1 as f64) * weight,
                            )
                        }))
                        .fold((0.0, 0.0), |acc, (low, high)| {
                            (acc.0 + low, acc.1 + high)
                        });
                        local_result.insert(variable.clone(), (dependency_weighted_bound.0 * variable_bounds.0 as f64, dependency_weighted_bound.1 * variable_bounds.1 as f64));
                } else {
                    let weight = weights.get(variable.as_str()).unwrap_or(&0.0);
                    local_result.insert(variable.clone(), (variable_bounds.0 as f64 * weight, variable_bounds.1 as f64 * weight));
                }
            }

            // Add the local result to the global result
            result.push(local_result);
        }
        return result;
    }

    pub fn score_combination(&self, interpretation: &IndexMap<ID, Bound>, weights: &IndexMap<ID, f64>) -> HashMap<ID, (f64, f64)> {
        return self.score_combination_batch(interpretation, &vec![weights]).get(0).unwrap().clone();
    }
    
    pub fn check_and_score(&self, interpretation: &IndexMap<ID, Bound>, weights: &IndexMap<ID, f64>) -> HashMap<ID, (f64, f64)> {
        self.score_combination(&self.check_combination(interpretation), weights)
    }

    pub fn check_and_score_default(&self, weights: &IndexMap<ID, f64>) -> HashMap<ID, (f64, f64)> {
        self.score_combination(&self.check_combination_default(), weights)
    }

    pub fn to_sparse_polyhedron(&self, double_binding: bool) -> SparsePolyhedron {

        fn get_coef_bounds(composite: &Constraint, amat: &IndexMap<String, BoolExpression>) -> IndexMap<String, Bound> {
            let mut coef_bounds: IndexMap<String, Bound> = IndexMap::new();
            for (coef_key, _) in composite.coefficients.iter() {
                let coef_exp = amat.get(&coef_key.to_string())
                    .unwrap_or_else(|| panic!("Coefficient key '{}' not found in _amat", coef_key));
                match coef_exp {
                    BoolExpression::Primitive(bound) => {
                        coef_bounds.insert(coef_key.to_string(), *bound);
                    },
                    _ => {coef_bounds.insert(coef_key.to_string(), (0,1));}
                }
            }
            return coef_bounds;
        }

        // Create a new sparse matrix
        let mut A_matrix = SparseIntegerMatrix::new();
        let mut b_vector: Vec<i64> = Vec::new();

        // Filter out all BoolExpressions that are primitives
        let primitives: HashMap<&String, Bound> = self._amat.iter()
            .filter_map(|(key, value)| {
                if let BoolExpression::Primitive(bound) = value {
                    Some((key, *bound))
                } else {
                    None
                }
            })
            .collect();

        // Filter out all BoolExpressions that are composites
        let composites: HashMap<&String, &Constraint> = self._amat.iter()
            .filter_map(|(key, value)| {
                if let BoolExpression::Composite(constraint) = value {
                    Some((key, constraint))
                } else {
                    None
                }
            })
            .collect();

        // Create a index mapping for all columns
        let column_names_map: IndexMap<String, usize> = primitives.keys().chain(composites.keys()).enumerate().map(|(i, key)| (key.to_string(), i)).collect();

        // Keep track of the current row index
        let mut row_i: usize = 0;

        for (key, composite) in composites {

            // Get the index of the current key
            let ki = *column_names_map.get(key).unwrap();

            // Construct the inner bound of the composite
            let coef_bounds = get_coef_bounds(composite, &self._amat);

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

        // Add the bounds for the primitive variables that are fixed.
        // We start by creating a grouping on the lower and upper bounds of the primitive variables
        let mut fixed_bound_map: HashMap<i64, Vec<usize>> = HashMap::new();
        for (key, bound) in primitives.iter().filter(|(_, bound)| bound_fixed(**bound)) {
            fixed_bound_map.entry(bound.0).or_insert_with(Vec::new).push(*column_names_map.get(&key.to_string()).unwrap());
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

        // Collect all integer variables
        let mut integer_variables: Vec<String> = Vec::new();

        // Restrain integer bounds
        for (p_key, p_bound) in primitives.iter().filter(|(_, bound)| bound.0 < 0 || bound.1 > 1) {
            
            // Add the variable to the integer variables list
            integer_variables.push(p_key.to_string());
            
            // Get the index of the current key
            let pi = *column_names_map.get(&p_key.to_string()).unwrap();
            
            if p_bound.0 < 0 {
                A_matrix.rows.push(row_i);
                A_matrix.cols.push(pi);
                A_matrix.vals.push(-1);
                b_vector.push(-1 * p_bound.0);
                row_i += 1;
            }

            if p_bound.1 > 1 {
                A_matrix.rows.push(row_i);
                A_matrix.cols.push(pi);
                A_matrix.vals.push(1);
                b_vector.push(p_bound.1);
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

        return polyhedron;
    }

    pub fn to_sparse_polyhedron_default(&self) -> SparsePolyhedron {
        self.to_sparse_polyhedron(true)
    }

    pub fn to_dense_polyhedron(&self, double_binding: bool) -> DensePolyhedron {
        DensePolyhedron::from(self.to_sparse_polyhedron(double_binding))
    }

    pub fn to_dense_polyhedron_default(&self) -> DensePolyhedron {
        self.to_dense_polyhedron(true)
    }
    
    pub fn set_primitive(&mut self, id: ID, bound: Bound) {
        // Insert the primitive variable into the constraints hashmap
        self._amat.insert(id.clone(), BoolExpression::Primitive(bound));
        
        // Insert the primitive variable into the aliases hashmap
        self._amap.insert(id.clone(), id.clone());
    }

    pub fn set_primitives(&mut self, ids: Vec<ID>, bound: Bound) {
        let unique_ids: IndexSet<_> = ids.into_iter().collect();
        for id in unique_ids {
            self.set_primitive(id, bound);
        }
    }

    pub fn set_gelineq(&mut self, coefficient_variables: Vec<Coefficient>, bias: i64, alias: Option<ID>) -> ID {
        // Ensure coefficients have unique keys by summing duplicate values
        let mut unique_coefficients: IndexMap<ID, i64> = IndexMap::new();
        for (key, value) in coefficient_variables {
            *unique_coefficients.entry(key).or_insert(0) += value;
        }
        let coefficient_variables: Vec<Coefficient> = unique_coefficients.into_iter().collect();

        // Create a hash from the input data
        let hash = create_hash(&coefficient_variables, bias);
        
        // Return the hash as a string
        let id = hash.to_string();

        // Insert the constraint into the constraints hashmap
        self._amat.insert(id.clone(), BoolExpression::Composite(Constraint { coefficients: coefficient_variables, bias: (bias, bias) }));

        // Set the alias if provided
        if let Some(alias) = alias {
            self._amap.insert(id.clone(), alias);
        }

        return id;
    }

    pub fn set_atleast(&mut self, references: Vec<ID>, value: i64, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        self.set_gelineq(unique_references.into_iter().map(|x| (x, 1)).collect(), -value, alias)
    }

    pub fn set_atmost(&mut self, references: Vec<ID>, value: i64, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        self.set_gelineq(unique_references.into_iter().map(|x| (x, -1)).collect(), value, alias)
    }

    pub fn set_equal(&mut self, references: Vec<ID>, value: i64, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        let ub = self.set_atleast(unique_references.clone().into_iter().collect(), value, None);
        let lb = self.set_atmost(unique_references.into_iter().collect(), value, None);
        self.set_and(vec![ub, lb], alias)
    }

    pub fn set_and(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        let length = unique_references.len();
        self.set_atleast(unique_references.into_iter().collect(), length as i64, alias)
    }

    pub fn set_or(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        self.set_atleast(unique_references.into_iter().collect(), 1, alias)
    }

    pub fn set_nand(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        let length = unique_references.len();
        self.set_atmost(unique_references.into_iter().collect(), length as i64 - 1, alias)
    }
    
    pub fn set_nor(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        self.set_atmost(unique_references.into_iter().collect(), 0, alias)
    }

    pub fn set_not(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        self.set_atmost(unique_references.into_iter().collect(), 0, alias)
    }

    pub fn set_xor(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        let atleast = self.set_or(unique_references.clone().into_iter().collect(), None);
        let atmost = self.set_atmost(unique_references.into_iter().collect(), 1, None);
        self.set_and(vec![atleast, atmost], alias)
    }

    pub fn set_xnor(&mut self, references: Vec<ID>, alias: Option<ID>) -> ID {
        let unique_references: IndexSet<_> = references.into_iter().collect();
        let atleast = self.set_atleast(unique_references.clone().into_iter().collect(), 2, None);
        let atmost = self.set_atmost(unique_references.into_iter().collect(), 0, None);
        self.set_or(vec![atleast, atmost], alias)
    }

    pub fn set_imply(&mut self, condition: ID, consequence: ID, alias: Option<ID>) -> ID {
        let not_condition = self.set_not(vec![condition], None);
        self.set_or(vec![not_condition, consequence], alias)
    }

    pub fn set_equiv(&mut self, lhs: ID, rhs: ID, alias: Option<ID>) -> ID {
        let imply_lr = self.set_imply(lhs.clone(), rhs.clone(), None);
        let imply_rl = self.set_imply(rhs.clone(), lhs.clone(), None);
        self.set_and(vec![imply_lr, imply_rl], alias)
    }

    pub fn set_alias(&mut self, id: ID, alias: String) {
        // Insert the alias into the aliases hashmap
        self._amap.insert(id.clone(), alias.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: for every primitive combination,
    ///   1) run `propagate` on the PLDAG model  
    ///   2) build the corresponding interpretation  
    ///   3) run `assume(root=1)` on the polyhedron  
    ///   4) evaluate the shrunken polyhedron on the same interpretation  
    ///   5) assert they agree at `root`.
    fn evaluate_model_polyhedron(
        model: &Pldag,
        poly: &DensePolyhedron,
        root: &String
    ) {
        for combo in model.primitive_combinations() {
            // build an IndexMap<String,Bound> as propagate expects
            let interp = combo.iter()
                .map(|(k,&v)| (k.clone(), (v,v)))
                .collect::<IndexMap<String,Bound>>();

            // what the DAG says the root can be
            let prop = model.check_combination(&interp);
            let model_root_val = *prop.get(root).unwrap();

            // now shrink the polyhedron by assuming root=1
            let mut assumption = HashMap::new();
            assumption.insert(root.clone(), 1);
            let shrunk = poly.assume(&assumption);

            // and evaluate that shrunk system on the same propagated bounds
            let poly_val = shrunk.evaluate(&prop);
            assert_eq!(
                poly_val,
                model_root_val,
                "Disagreement on {:?}: model={:?}, poly={:?}",
                combo,
                model_root_val,
                poly_val
            );
        }
    }

    #[test]
    fn test_propagate() {
        let mut model = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        let root = model.set_and(vec!["x".to_string(), "y".to_string()], None);

        let result = model.check_combination(&IndexMap::new());
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut interpretation = IndexMap::new();
        interpretation.insert("x".to_string(), (1, 1));
        interpretation.insert("y".to_string(), (1, 1));
        let result = model.check_combination(&interpretation);
        assert_eq!(result.get(&root).unwrap(), &(1, 1));

        let mut model = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        model.set_primitive("z".to_string(), (0, 1));
        let root = model.set_xor(vec!["x".to_string(), "y".to_string(), "z".to_string()], None);
        let result = model.check_combination(&IndexMap::new());
        assert_eq!(result.get("x").unwrap(), &(0, 1));
        assert_eq!(result.get("y").unwrap(), &(0, 1));
        assert_eq!(result.get("z").unwrap(), &(0, 1));
        assert_eq!(result.get(&root).unwrap(), &(0, 1));

        let mut interpretation = IndexMap::new();
        interpretation.insert("x".to_string(), (1, 1));
        interpretation.insert("y".to_string(), (1, 1));
        interpretation.insert("z".to_string(), (1, 1));
        let result = model.check_combination(&interpretation);
        assert_eq!(result.get(&root).unwrap(), &(0, 0));
        
        let mut interpretation = IndexMap::new();
        interpretation.insert("x".to_string(), (0, 1));
        interpretation.insert("y".to_string(), (1, 1));
        interpretation.insert("z".to_string(), (1, 1));
        let result = model.check_combination(&interpretation);
        assert_eq!(result.get(&root).unwrap(), &(0, 0));
        
        let mut interpretation = IndexMap::new();
        interpretation.insert("x".to_string(), (0, 0));
        interpretation.insert("y".to_string(), (1, 1));
        interpretation.insert("z".to_string(), (0, 0));
        let result = model.check_combination(&interpretation);
        assert_eq!(result.get(&root).unwrap(), &(1, 1));
    }

    /// XOR already covered; test the OR gate
    #[test]
    fn test_propagate_or_gate() {
        let mut model = Pldag::new();
        model.set_primitive("a".into(), (0, 1));
        model.set_primitive("b".into(), (0, 1));
        let or_root = model.set_or(vec!["a".into(), "b".into()], None);

        // No assignment: both inputs full [0,1], output [0,1]
        let res = model.check_combination(&IndexMap::new());
        assert_eq!(res["a"], (0, 1));
        assert_eq!(res["b"], (0, 1));
        assert_eq!(res[&or_root], (0, 1));

        // a=1 ⇒ output must be 1
        let mut interp = IndexMap::new();
        interp.insert("a".into(), (1, 1));
        let res = model.check_combination(&interp);
        assert_eq!(res[&or_root], (1, 1));

        // both zero ⇒ output zero
        let mut interp = IndexMap::new();
        interp.insert("a".into(), (0, 0));
        interp.insert("b".into(), (0, 0));
        let res = model.check_combination(&interp);
        assert_eq!(res[&or_root], (0, 0));

        // partial: a=[0,1], b=0 ⇒ output=[0,1]
        let mut interp = IndexMap::new();
        interp.insert("b".into(), (0, 0));
        let res = model.check_combination(&interp);
        assert_eq!(res[&or_root], (0, 1));
    }

    /// Test the NOT gate (negation)
    #[test]
    fn test_propagate_not_gate() {
        let mut model = Pldag::new();
        model.set_primitive("p".into(), (0, 1));
        let not_root = model.set_not(vec!["p".into()], None);

        // no assignment ⇒ [0,1]
        let res = model.check_combination(&IndexMap::new());
        assert_eq!(res["p"], (0, 1));
        assert_eq!(res[&not_root], (0, 1));

        // p = 0 ⇒ root = 1
        let mut interp = IndexMap::new();
        interp.insert("p".into(), (0, 0));
        let res = model.check_combination(&interp);
        assert_eq!(res[&not_root], (1, 1));

        // p = 1 ⇒ root = 0
        let mut interp = IndexMap::new();
        interp.insert("p".into(), (1, 1));
        let res = model.check_combination(&interp);
        assert_eq!(res[&not_root], (0, 0));
    }

    #[test]
    fn test_to_polyhedron_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0,1));
        m.set_primitive("y".into(), (0,1));
        let root = m.set_and(vec!["x".into(), "y".into()], None);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_or() {
        let mut m = Pldag::new();
        m.set_primitive("a".into(), (0,1));
        m.set_primitive("b".into(), (0,1));
        m.set_primitive("c".into(), (0,1));
        let root = m.set_or(vec!["a".into(), "b".into(), "c".into()], None);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_not() {
        let mut m = Pldag::new();
        m.set_primitive("p".into(), (0,1));
        let root = m.set_not(vec!["p".into()], None);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_xor() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0,1));
        m.set_primitive("y".into(), (0,1));
        m.set_primitive("z".into(), (0,1));
        let root = m.set_xor(vec!["x".into(), "y".into(), "z".into()], None);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    #[test]
    fn test_to_polyhedron_nested() {
        // Build a small two‐level circuit:
        //   w = AND(x,y),  v = OR(w, NOT(z))
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0,1));
        m.set_primitive("y".into(), (0,1));
        m.set_primitive("z".into(), (0,1));

        let w = m.set_and(vec!["x".into(), "y".into()], None);
        let nz = m.set_not(vec!["z".into()], None);
        let v = m.set_or(vec![w.clone(), nz.clone()], None);

        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &v);
    }

    /// Nested/composed AND then XOR: 
    ///   w = AND(x,y);  v = XOR(w,z)
    #[test]
    fn test_propagate_nested_composite() {
        let mut model = Pldag::new();
        model.set_primitive("x".into(), (0, 1));
        model.set_primitive("y".into(), (0, 1));
        model.set_primitive("z".into(), (0, 1));

        let w = model.set_and(vec!["x".into(), "y".into()], None);
        let v = model.set_xor(vec![w.clone(), "z".into()], None);

        // no assignment: everything [0,1]
        let res = model.check_combination(&IndexMap::new());
        for var in &["x","y","z"] {
            assert_eq!(res[*var], (0,1), "{}", var);
        }
        assert_eq!(res[&w], (0,1));
        assert_eq!(res[&v], (0,1));

        // x=1,y=1,z=0 ⇒ w=1,v=1
        let mut interp = IndexMap::new();
        interp.insert("x".into(), (1,1));
        interp.insert("y".into(), (1,1));
        interp.insert("z".into(), (0,0));
        let res = model.check_combination(&interp);
        assert_eq!(res[&w], (1,1));
        assert_eq!(res[&v], (1,1));

        // x=0,y=1,z=1 ⇒ w=0,v=1
        let mut interp = IndexMap::new();
        interp.insert("x".into(), (0,0));
        interp.insert("y".into(), (1,1));
        interp.insert("z".into(), (1,1));
        let res = model.check_combination(&interp);
        assert_eq!(res[&w], (0,0));
        assert_eq!(res[&v], (1,1));

        // x=0,y=0,z=0 ⇒ w=0,v=0
        let mut interp = IndexMap::new();
        interp.insert("x".into(), (0,0));
        interp.insert("y".into(), (0,0));
        interp.insert("z".into(), (0,0));
        let res = model.check_combination(&interp);
        assert_eq!(res[&w], (0,0));
        assert_eq!(res[&v], (0,0));
    }

    /// If you ever get an inconsistent assignment (out‐of‐bounds for a primitive),
    /// propagate should leave it as given (or you could choose to clamp / panic)—here
    /// we simply check that nothing blows up.
    #[test]
    fn test_propagate_out_of_bounds_does_not_crash() {
        let mut model = Pldag::new();
        model.set_primitive("u".into(), (0, 1));
        let root = model.set_not(vec!["u".into()], None);

        let mut interp = IndexMap::new();
        // ← deliberately illegal: u ∈ {0,1} but we assign 5
        interp.insert("u".into(), (5,5));
        let res = model.check_combination(&interp);

        // we expect propagate to return exactly (5,5) for "u" and compute root = negate(5)
        assert_eq!(res["u"], (5,5));
        // Depending on your semantic for negate, it might be
        //   bound_multiply(-1,(5,5)) + bias
        // so just check it didn’t panic:
        let _ = res[&root];
    }

    #[test]
    fn test_to_polyhedron() {

        fn evaluate_model_polyhedron(model: &Pldag, polyhedron: &DensePolyhedron, root: &String) {
            for combination in model.primitive_combinations() {
                let interpretation = combination
                    .iter()
                    .map(|(k, &v)| (k.clone(), (v, v)))
                    .collect::<IndexMap<String, Bound>>();
                let model_prop = model.check_combination(&interpretation);
                let model_eval = *model_prop.get(root).unwrap();
                let mut assumption = HashMap::new();
                assumption.insert(root.clone(), 1);
                let assumed_polyhedron = polyhedron.assume(&assumption);
                let assumed_poly_eval = assumed_polyhedron.evaluate(&model_prop);
                assert_eq!(assumed_poly_eval, model_eval);
            }
        }

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        model.set_primitive("z".to_string(), (0, 1));
        let root = model.set_xor(vec!["x".to_string(), "y".to_string(), "z".to_string()], None);
        let polyhedron: DensePolyhedron = model.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        let root = model.set_and(vec!["x".to_string(), "y".to_string()], None);
        let polyhedron = model.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);

        let mut model: Pldag = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        model.set_primitive("z".to_string(), (0, 1));
        let root = model.set_xor(vec!["x".to_string(), "y".to_string(), "z".to_string()], None);
        let polyhedron = model.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&model, &polyhedron, &root);
    }

    /// Single‐operand composites should act as identity: root == operand
    #[test]
    fn test_to_polyhedron_single_operand_identity() {
        // AND(x) == x
        {
            let mut m = Pldag::new();
            m.set_primitive("x".into(), (0,1));
            let root = m.set_and(vec!["x".into()], None);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // OR(y) == y
        {
            let mut m = Pldag::new();
            m.set_primitive("y".into(), (0,1));
            let root = m.set_or(vec!["y".into()], None);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
        // XOR(z) == z
        {
            let mut m = Pldag::new();
            m.set_primitive("z".into(), (0,1));
            let root = m.set_xor(vec!["z".into()], None);
            let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
            evaluate_model_polyhedron(&m, &poly, &root);
        }
    }

    /// Duplicate‐operand AND(x,x) should also behave like identity(x)
    #[test]
    fn test_to_polyhedron_duplicate_operands_and() {
        let mut m = Pldag::new();
        m.set_primitive("x".into(), (0,1));
        let root = m.set_and(vec!["x".into(), "x".into()], None);
        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
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
        for &v in &["a","b","c","d","e"] {
            m.set_primitive(v.into(), (0,1));
        }
        let a = "a".to_string();
        let b = "b".to_string();
        let c = "c".to_string();
        let d = "d".to_string();

        let w1 = m.set_and(vec![a.clone(), b.clone()], None);
        let w2 = m.set_or(vec![w1.clone(), c.clone()], None);
        let w3 = m.set_xor(vec![w2.clone(), d.clone()], None);
        let root = m.set_not(vec![w3.clone()], None);

        let poly: DensePolyhedron = m.to_sparse_polyhedron_default().into();
        evaluate_model_polyhedron(&m, &poly, &root);
    }

    /// Helper to build the “a → [b,c], b → [], c → [d,e]” example.
    fn make_simple_dag() -> Pldag {
        let mut pldag = Pldag::new();
        pldag.set_primitive("b".into(), (0, 1));
        pldag.set_primitive("d".into(), (0, 1));
        pldag.set_primitive("e".into(), (0, 1));
        let c = pldag.set_or(vec!["d".into(), "e".into()], None);
        let a = pldag.set_or(vec!["b".into(), c.clone()], None);
        pldag.set_alias("a".into(), a.clone());
        pldag.set_alias("c".into(), c.clone());
        return pldag;
    }

    #[test]
    fn test_simple_dag() {
        let pldag = make_simple_dag();
        let deps = pldag.transitive_dependencies();

        let expect = |xs: &[&str]| {
            xs.iter().cloned().map(String::from).collect::<HashSet<_>>()
        };

        let a = pldag.get_id(&"a".to_string());
        let c = pldag.get_id(&"c".to_string());
        assert_eq!(deps.get(&a), Some(&expect(&["b", &c.clone(), "d", "e"])));
        assert_eq!(deps.get("b"), Some(&expect(&[])));
        assert_eq!(deps.get(&c), Some(&expect(&["d", "e"])));
        assert_eq!(deps.get("d"), Some(&expect(&[])));
        assert_eq!(deps.get("e"), Some(&expect(&[])));
    }

    #[test]
    fn test_chain_dag() {
        // x → [y], y → [z], z → []
        let mut pldag = Pldag::new();
        pldag.set_primitive("z".into(), (0, 0));;
        let y = pldag.set_or(vec!["z".into()], None);
        let x = pldag.set_or(vec![y.clone()], None);
        let deps = pldag.transitive_dependencies();

        let expect = |xs: &[&str]| {
            xs.iter().cloned().map(String::from).collect::<HashSet<_>>()
        };

        assert_eq!(deps.get(&x), Some(&expect(&[&y.to_string(), "z"])));
        assert_eq!(deps.get(&y), Some(&expect(&["z"])));
        assert_eq!(deps.get("z"), Some(&expect(&[])));
    }

    #[test]
    fn test_all_primitives() {
        // no Composite at all
        let mut amat = IndexMap::new();
        for &name in &["p", "q", "r"] {
            amat.insert(name.into(), BoolExpression::Primitive((1, 5)));
        }
        let mut pldag = Pldag::new();
        pldag._amat = amat;
        let deps = pldag.transitive_dependencies();

        for &name in &["p", "q", "r"] {
            assert!(deps.get(name).unwrap().is_empty(), "{} should have no deps", name);
        }
    }

    #[test]
    fn test_propagate_weighted() {
        let mut model = Pldag::new();
        model.set_primitive("x".to_string(), (0, 1));
        model.set_primitive("y".to_string(), (0, 1));
        let root = model.set_and(vec!["x".to_string(), "y".to_string()], None);
        let mut interpretation = IndexMap::new();
        interpretation.insert("x".to_string(), (1, 1));
        interpretation.insert("y".to_string(), (1, 1));
        let mut weights: IndexMap<String, f64> = IndexMap::new();
        weights.insert("x".to_string(), 2.0);
        weights.insert("y".to_string(), 3.0);
        let propagated = model.check_and_score(&interpretation, &weights);
        assert_eq!(propagated.get("x").unwrap(), &(2.0, 2.0));
        assert_eq!(propagated.get("y").unwrap(), &(3.0, 3.0));
        assert_eq!(propagated.get(&root).unwrap(), &(5.0, 5.0));
    }

    #[test]
    fn test_readme_example() {
        // Build your PL-DAG (omitting details)...
        // For example, we create a model of three boolean variables x, y and z.
        // We bind them to an xor constraint.
        let mut pldag: Pldag = Pldag::new();

        // First setup the primitive variables
        pldag.set_primitive("x".to_string(), (0, 1));
        pldag.set_primitive("y".to_string(), (0, 1));
        pldag.set_primitive("z".to_string(), (0, 1));

        // A reference ID is returned
        let root = pldag.set_or(vec![
            "x".to_string(),
            "y".to_string(),
            "z".to_string(),
        ], None);

        // 1. Validate a combination:
        let mut inputs: IndexMap<String, Bound> = IndexMap::new();
        let validited = pldag.check_combination(&inputs);
        // Since nothing is given, and all other variable inplicitly is (0, 1) from the pldag model,
        // the root will be (0,1) since there's not enough information to evalute the root `or` node.
        println!("Root valid? {}", *validited.get(&root).unwrap() == (1, 1)); // This will be False

        // If we however for instance fix x to be zero, then the root is false
        inputs.insert("x".to_string(), (0,0));
        let revalidited = pldag.check_combination(&inputs);
        println!("Root valid? {}", *revalidited.get(&root).unwrap() == (1, 1)); // This will be false

        // However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
        inputs.insert("y".to_string(), (1,1));
        inputs.insert("z".to_string(), (1,1));
        let revalidited = pldag.check_combination(&inputs);
        println!("Root valid? {}", *revalidited.get(&root).unwrap() == (1, 1)); // This will be true

        // 2. Score a configuration:
        // We can score a configuration by using the score_combination function.
        let mut weights: IndexMap<String, f64> = IndexMap::new();
        weights.insert("x".to_string(), 1.0);
        weights.insert("y".to_string(), 2.0);
        weights.insert("z".to_string(), 3.0);
        // Add a discount value if the root is true
        weights.insert(root.clone(), -1.0);
        let scores = pldag.check_and_score(&inputs, &weights);
        println!("Total score: {:?}", scores.get(&root).unwrap());

        // And notice what will happen if we remove the x value (i.e. x being (0,1))
        inputs.insert("x".to_string(), (0,1));
        let scores = pldag.check_and_score(&inputs, &weights);
        // The root will return (5,6) meaning its value is between 5 and 6 with not enough information to
        // determine the exact value. 
        println!("Total score: {:?}", scores.get(&root).unwrap());

        // .. and if we set x to be 0, then the root will be definitely 5.
        inputs.insert("x".to_string(), (0,0));
        let scores = pldag.check_and_score(&inputs, &weights);
        println!("Total score: {:?}", scores.get(&root).unwrap());

        // .. and if we set y and z to be 0, then the root will be 0.
        inputs.insert("y".to_string(), (0,0));
        inputs.insert("z".to_string(), (0,0));
        let scores = pldag.check_and_score(&inputs, &weights);
        println!("Total score: {:?}", scores.get(&root).unwrap());
    }
}
