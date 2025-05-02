# Primitive Logic Directed Acyclic Graph (PL-DAG)

A **Primitive Logic Directed Acyclic Graph** (PL-DAG) is a DAG in which every node encodes a logical operation and every leaf represents a literal. Interior nodes freely express arbitrary Boolean combinations of their predecessors—for example, an AND-node evaluates to true only if _all_ of its incoming nodes (or leaves) evaluate to true. This “freedom” to assemble logical relationships with minimal ceremony makes the PL-DAG both powerful and easy to work with.

---

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

// Build your PL-DAG (omitting details)...
let pldag: Pldag = /* ... */;

// 1. Validate a combination:
let inputs: HashMap<String, Bound> = /* leaf assignments */;
let validity = pldag.check_combination(&inputs);
println!("Root valid? {}", validity.get("root").copied().unwrap_or(false));

// 2. Score a configuration:
let weights: HashMap<String, f64> = /* per-node weights */;
let scores = pldag.score_combination(&inputs, &weights);
println!("Total score: {}", scores.get("root").unwrap());

// 3. Validate and score in one go:
let (valid_map, score_map) = pldag.check_and_score(&inputs, &weights);

// 4. Export for ILP:
let poly = pldag.to_sparse_polyhedron();
// feed `poly` to your ILP solver

// 5. Batch scoring:
let multi_scores = pldag.batch_score_combinations(&inputs, &multi_weights);
```

Enjoy building and evaluating logical models with the PL-DAG!

