# Primitive Logic Directed Acyclic Graph (PL-DAG)

A **Primitive Logic Directed Acyclic Graph** (PL-DAG) is a DAG in which every node encodes a logical operation and every leaf represents a literal. Interior nodes freely express arbitrary Boolean combinations of their predecessors—for example, an AND-node evaluates to true only if _all_ of its incoming nodes (or leaves) evaluate to true. This “freedom” to assemble logical relationships with minimal ceremony makes the PL-DAG both powerful and easy to work with.

---

## Install
```bash
cargo add pldag-rust
```

## Core Routines

The PL-DAG provides these core APIs for working with combinations and evaluations:

### 1. `check_combination`

```rust
fn check_combination(
    &self,
    inputs: &HashMap<String, Bound>,
) -> HashMap<String, bool>;
```

- **Purpose:** Takes a map of leaf-IDs → `Bound` values, propagates Boolean truth-assignments up the graph, and returns for each node whether it ends up `true` or `false`. Typically, you’ll inspect the root to see if a given combination is valid.

---

### 2. `score_combination`

```rust
fn score_combination(
    &self,
    inputs: &HashMap<String, Bound>,
    weights: &HashMap<String, f64>,
) -> HashMap<String, f64>;
```

- **Purpose:** After assigning `Bound` values to leaves, apply a single map of node-IDs → `f64` weights. Each node’s score is computed by combining its own weight with the scores of its children. Ideal for “pricing” or weighting a single configuration.

---

### 3. `check_and_score`

```rust
fn check_and_score(
    &self,
    inputs: &HashMap<String, Bound>,
    weights: &HashMap<String, f64>,
) -> (HashMap<String, bool>, HashMap<String, f64>);
```

- **Purpose:** A one-stop helper that first runs `check_combination` and then feeds the resulting Booleans into `score_combination`. Returns both the Boolean map and the per-node scores in one call.

---

### 4. `to_sparse_polyhedron`

```rust
fn to_sparse_polyhedron(&self) -> SparsePolyhedron;
```

- **Purpose:** Exports the PL-DAG as a sparse polyhedral system—perfect for dropping straight into an ILP solver when you need to optimize or solve over the same logical constraints.

---

### 5. `batch_score_combinations`

```rust
fn batch_score_combinations(
    &self,
    inputs: &HashMap<String, Bound>,
    weight_sets: &HashMap<String, HashMap<String, f64>>,
) -> HashMap<String, HashMap<String, f64>>;
```

- **Purpose:** Score a single combination against multiple weight maps at once. Returns, for each named weight set, a per-node score map.

---

## Example Usage

```rust
use indexmap::IndexMap;
use std::collections::HashMap;
use pldag::{Pldag, Bound};

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
```

Enjoy building and evaluating logical models with the PL-DAG!

