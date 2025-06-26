# Primitive Logic Directed Acyclic Graph (PL-DAG)

A **Primitive Logic Directed Acyclic Graph** (PL-DAG) is a DAG in which every node encodes a logical operation and every leaf represents a literal. Interior nodes freely express arbitrary Boolean combinations of their predecessors—for example, an AND-node evaluates to true only if _all_ of its incoming nodes (or leaves) evaluate to true. This "freedom" to assemble logical relationships with minimal ceremony makes the PL-DAG both powerful and easy to work with.

---

## Install
```bash
cargo add pldag-rust
```

## Core Routines

The PL-DAG provides these core APIs for working with combinations and evaluations:

### 1. `propagate`

```rust
fn propagate(
    &self,
    assignment: &Assignment, // IndexMap<String, Bound>
) -> Assignment;
```

- **Purpose:** Takes a map of node-IDs → `Bound` values, propagates bounds up the graph bottom-up, and returns the resulting bounds for each node. A `Bound` is a tuple `(i64, i64)` representing min and max values.

---

### 2. `propagate_coefs`

```rust
fn propagate_coefs(
    &self,
    assignment: &Assignment,
) -> ValuedAssignment; // IndexMap<String, MultiBound>
```

- **Purpose:** Propagates both bounds and coefficients through the DAG. Returns a `ValuedAssignment` where each node maps to a `MultiBound = (Bound, VBound)`. The coefficient for a parent node is the sum of its children's coefficients plus its own coefficient.

---

### 3. `propagate_default` and `propagate_coefs_default`

```rust
fn propagate_default(&self) -> Assignment;
fn propagate_coefs_default(&self) -> ValuedAssignment;
```

- **Purpose:** Convenience methods that propagate using the default primitive bounds defined in the DAG.

---

### 4. `to_sparse_polyhedron`

```rust
fn to_sparse_polyhedron(&self, double_binding: bool, integer_constraints: bool, fixed_constraints: bool) -> SparsePolyhedron;
```

- **Purpose:** Exports the PL-DAG as a sparse polyhedral system—perfect for dropping straight into an ILP solver when you need to optimize or solve over the same logical constraints.

---

### 5. Node Management

```rust
fn set_coef(&mut self, id: ID, coefficient: f64);
fn get_coef(&self, id: &ID) -> f64;
fn get_objective(&self) -> IndexMap<String, f64>;
fn set_primitive(&mut self, id: ID, bound: Bound);
```

- **Purpose:** Manage coefficients and primitive bounds on nodes. Each node stores both its logical expression and an associated coefficient. The `get_objective()` method retrieves objective function coefficients for ILP optimization.

---

## Example Usage

```rust
use indexmap::IndexMap;
use pldag::{Pldag, Bound};

// Build your PL-DAG
// For example, we create a model of three boolean variables x, y and z.
// We bind them to an OR constraint.
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
]);

// 1. Validate a combination:
let mut inputs: IndexMap<String, Bound> = IndexMap::new();
let validated = pldag.propagate(&inputs);
// Since nothing is given, and all other variables implicitly have bounds (0, 1) from the pldag model,
// the root will be (0,1) since there's not enough information to evaluate the root `or` node.
println!("Root valid? {}", *validated.get(&root).unwrap() == (1, 1)); // This will be false

// If we however fix x to be zero, then we can check the result
inputs.insert("x".to_string(), (0,0));
let revalidated = pldag.propagate(&inputs);
println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be false

// However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
inputs.insert("y".to_string(), (1,1));
inputs.insert("z".to_string(), (1,1));
let revalidated = pldag.propagate(&inputs);
println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be true

// 2. Score a configuration:
// We can score a configuration by setting coefficients on nodes.
pldag.set_coef("x".to_string(), 1.0);
pldag.set_coef("y".to_string(), 2.0);
pldag.set_coef("z".to_string(), 3.0);
// Add a discount value if the root is true
pldag.set_coef(root.clone(), -1.0);

// Use propagate_coefs to get both bounds and accumulated coefficients
let scores = pldag.propagate_coefs(&inputs);
// The result contains (bounds, coefficients) for each node
let root_result = scores.get(&root).unwrap();
println!("Root bounds: {:?}, Total score: {:?}", root_result.0, root_result.1);

// And notice what will happen if we remove the x value (i.e. x being (0,1))
inputs.insert("x".to_string(), (0,1));
let scores = pldag.propagate_coefs(&inputs);
// The coefficients will reflect the range of possible values
let root_result = scores.get(&root).unwrap();
println!("Root bounds: {:?}, Score range: {:?}", root_result.0, root_result.1);

// .. and if we set x to be 0, then the score will be more constrained.
inputs.insert("x".to_string(), (0,0));
let scores = pldag.propagate_coefs(&inputs);
let root_result = scores.get(&root).unwrap();
println!("Root bounds: {:?}, Score: {:?}", root_result.0, root_result.1);

// .. and if we set y and z to be 0, then the root will be 0.
inputs.insert("y".to_string(), (0,0));
inputs.insert("z".to_string(), (0,0));
let scores = pldag.propagate_coefs(&inputs);
let root_result = scores.get(&root).unwrap();
println!("Root bounds: {:?}, Score: {:?}", root_result.0, root_result.1);
```

Enjoy building and evaluating logical models with the PL-DAG!
