# Propositional Logic Directed Acyclic Graph (PL-DAG)

A PL-DAG is a directed acyclic graph for representing many related discrete logic models in one shared structure. Each leaf defines a discrete integer variable domain, and each internal node defines a reusable derived variable constrained by a linear relation over its input variables.

Instead of building a one-off optimization model, solving it, and then throwing it away or caching the result, a PL-DAG lets you keep the underlying logical structure alive as a modular graph. Large domains can be represented once, shared across use cases, and sliced into smaller computation-ready subgraphs when needed.

This makes PL-DAG useful as an abstraction layer above traditional ILP modelling. You can store commercial rules, technical rules, product baselines, feature dependencies, capacity constraints, and other logical representations in the same graph without having to compute over the whole thing every time.

The central operation is `sub_dag`: given one or more nodes of interest, it extracts only the transitive dependency graph needed for that computation. That slice can then be used for bound propagation or exported to a polyhedral form your favourite ILP solver can consume.

PL-DAGs are especially useful when a domain contains many overlapping logical models:

- many variants of the same model that differ along independent dimensions (region, segment, version, time period, …)
- commercial and technical/BOM layers that need to coexist
- feature toggles and dependency systems shared across many products
- capacity, compatibility, and selection rules reused in many contexts
- cross-model questions that are hard to ask when each model is built and solved in isolation

By breaking logic into small reusable nodes, a single PL-DAG can hold millions of nodes spanning a whole domain. Ordinary computations still operate on small extracted subgraphs, while broader questions can cross boundaries that would otherwise live in separate models.

---

## ✨ Key Features

| Area              | What you get                                                                                              |
| ----------------- | --------------------------------------------------------------------------------------------------------- |
| **Modelling**     | Build Boolean / linear constraint systems in a single graph representation.                               |
| **Async storage** | Mutating and storage-touching methods are `async fn`, so you can plug in any backend (Postgres, Redis, …).|
| **Analysis**      | Efficient bound propagation over compiled DAG snapshots.                                                  |
| **Export**        | `to_sparse_polyhedron()` emits a polyhedral ILP model ready for any external ILP solver.|

---

## Typical Workflow

You keep **one big graph** that can hold millions of nodes representing all sorts of things in your domain — products, rules, capacities, feature flags — and pull out just the slice you need, whenever you need it.

The everyday loop is three steps:

1. **Populate** — grow the DAG using `set_primitive`, `set_or`, `set_atleast`, … Do it once, or keep adding to it as your domain evolves.
2. **Sub-DAG** — pick a node you care about and call `pldag.sub_dag(vec![node_id]).await?`. You get back a self-contained `CompiledDag` of just the subgraph rooted at that node.
3. **Compute** — run `propagate`, `to_sparse_polyhedron`, or any other analysis on that snapshot.

```rust
// 1. Populate (once, or as your domain evolves)
pldag.set_primitive("x", (0, 1)).await?;
pldag.set_primitive("y", (0, 1)).await?;
let root = pldag.set_or(vec!["x", "y"]).await?;

// 2. Pick a node of interest, get its sub-DAG
let dag = pldag.sub_dag(vec![root.clone()]).await?;

// 3. Compute on it
let bounds = dag.propagate([("x", (1, 1))])?;
```

That's it. The same PL-DAG can back any number of these slices, so a single shared model serves many use cases at once.

---

## Install

```bash
cargo add pldag
```

You will also need an async runtime to drive the model-building calls. The examples below use [`tokio`](https://crates.io/crates/tokio):

```bash
cargo add tokio --features rt-multi-thread,macros
```

---

## Architecture

A `Pldag` is a thin wrapper around a pluggable storage backend. The storage holds the DAG topology; the `Pldag` exposes the modelling API on top of it.

```
┌──────────────┐       ┌────────────────┐       ┌──────────────────┐
│    Pldag     │ ──►   │ NodeStoreTrait │ ──►   │  KeyValueStore   │
│  (modelling) │       │   (DAG ops)    │       │ (bytes / values) │
└──────────────┘       └────────────────┘       └──────────────────┘
                                                         │
                                                         ▼
                                          InMemoryStore  /  Your DB
```

- `KeyValueStore` is the low-level async key/value contract.
- `NodeStoreTrait` layers DAG-specific operations (parents, children, primitives) on top.
- `InMemoryStore` is the default backend (`std::sync::RwLock` under the hood). For a custom backend, implement the two traits and pass it via `Pldag::new_custom`.

Because the traits are async, you can back the DAG with any database that has an async client.

---

## Core Routines

### Building the model

```rust
async fn set_primitive(&self, id: &str, bound: Bound) -> ModelResult<ID>;
async fn set_primitives<K>(&self, ids: impl IntoIterator<Item = K>, bound: Bound) -> ModelResult<Vec<ID>>
where K: ToString;
```

Create primitive (leaf) variables with specified bounds.

```rust
async fn set_gelineq<K>(
    &self,
    coefficient_variables: impl IntoIterator<Item = (K, i32)>,
    bias: i32,
) -> ModelResult<ID>
where K: ToString;
```

Creates a general linear inequality: `sum(coeff_i * var_i) + bias >= 0`.

```rust
async fn set_atleast<K>(&self, references: impl IntoIterator<Item = K>, value: i32) -> ModelResult<ID>;
async fn set_atmost<K>(&self, references: impl IntoIterator<Item = K>, value: i32) -> ModelResult<ID>;
async fn set_equal<K, I>(&self, references: I, value: i32) -> ModelResult<ID>;
```

`sum(vars) >= value`, `sum(vars) <= value`, `sum(vars) == value`.

#### Logical constraints

```rust
async fn set_and<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_or<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_not<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_xor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_nand<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_nor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_xnor<K>(&self, references: impl IntoIterator<Item = K>) -> ModelResult<ID>;
async fn set_imply<C, Q>(&self, condition: C, consequence: Q) -> ModelResult<ID>;
async fn set_equiv<L, R>(&self, lhs: L, rhs: R) -> ModelResult<ID>;
```

- `set_and`: True iff all input variables are true.
- `set_or`: True iff at least one input variable is true.
- `set_not`: True iff the input variable is false.
- `set_xor`: True iff exactly one input variable is true.
- `set_nand`: True iff not all input variables are true.
- `set_nor`: True iff no input variables are true.
- `set_xnor`: True iff not exactly one input variable is true.
- `set_imply`: True iff the condition being true requires the consequence to be true. Equivalent to `!condition OR consequence`.
- `set_equiv`: True iff `lhs` and `rhs` have the same Boolean value.

#### Inspection / removal

```rust
async fn get_node(&self, id: &str) -> ModelResult<Option<Node>>;
async fn get_nodes(&self, ids: &[String]) -> ModelResult<HashMap<String, Node>>;
async fn delete_node(&self, id: &str) -> ModelResult<()>;
```

`delete_node` errors with `ModelError::NodeReferenced` if other nodes still depend on the target.

### Snapshotting & analysis

```rust
async fn dag(&self) -> ModelResult<CompiledDag>;
async fn sub_dag(&self, roots: Vec<ID>) -> ModelResult<CompiledDag>;
```

`dag` snapshots the entire DAG; `sub_dag` snapshots only the subgraph reachable from `roots` (or the entire DAG if `roots` is empty).

Once you hold a `CompiledDag`, all further analyses are synchronous and return [`ComputeResult`]:

```rust
fn CompiledDag::propagate<K>(
    &self,
    assignments: impl IntoIterator<Item = (K, Bound)>,
) -> ComputeResult<HashMap<String, Bound>>;

fn Pldag::propagate_dag<K>(
    dag: &CompiledDag,
    assignments: impl IntoIterator<Item = (K, Bound)>,
) -> ComputeResult<Assignment>;
```

Snapshot then compute: `let dag = pldag.dag().await?; let bounds = dag.propagate(assignments)?;`.

### Export

```rust
fn Pldag::to_sparse_polyhedron(cd: &CompiledDag, double_binding: bool) -> ComputeResult<SparsePolyhedron>;
fn Pldag::to_dense_polyhedron(cd: &CompiledDag, double_binding: bool) -> ComputeResult<DensePolyhedron>;
```

Both `SparsePolyhedron` and `DensePolyhedron` implement `serde::Serialize`, so you can ship them over HTTP to a remote ILP service or hand them to any in-process solver of your choice.

---

## Errors

Fallible operations return one of two disjoint result types so each function's signature describes its actual failure modes:

- **`ModelResult<T>` / `ModelError`** — operations that touch storage (`set_*`, `delete_node`, `get_node(s)`, `dag`, `sub_dag`). Variants: `EmptyConstraint`, `NodeNotFound`, `NodeReferenced`, and `Backend(StorageError)` for underlying backend failures.
- **`ComputeResult<T>` / `ComputeError`** — pure in-memory operations on a `CompiledDag` (propagate, tighten, ranks, polyhedron export). Variants: `NodeOutOfBounds`, `MaxIterationsExceeded`, `CycleDetected`.

`StorageError` is the lower-level backend error returned by `KeyValueStore` / `NodeStoreTrait` implementations; it flows into `ModelError::Backend` via `?`.

---

## Quick Example

```rust
use std::collections::HashMap;
use pldag::{Bound, Pldag};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build a simple OR‑of‑three model
    let pldag = Pldag::new();
    pldag.set_primitive("x", (0, 1)).await?;
    pldag.set_primitive("y", (0, 1)).await?;
    pldag.set_primitive("z", (0, 1)).await?;
    let root = pldag.set_or(vec!["x", "y", "z"]).await?;

    // Snapshot the DAG once, then analyse synchronously.
    let dag = pldag.dag().await?;

    // 1. With no assignments, the OR's bound is undetermined: (0, 1).
    let bounds = dag.propagate(Vec::<(&str, Bound)>::new())?;
    assert_eq!(bounds[&root], (0, 1));

    // 2. Pinning y = 1 forces the OR to true.
    let mut assignments: HashMap<&str, Bound> = HashMap::new();
    assignments.insert("y", (1, 1));
    let bounds = dag.propagate(assignments)?;
    assert_eq!(bounds[&root], (1, 1));

    Ok(())
}
```

---

## Custom Storage Backend

Plug in your own database by implementing the two storage traits:

```rust
use async_trait::async_trait;
use pldag::{KeyValueStore, NodeStore, Pldag, StorageResult};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

struct MyDbStore { /* connection pool, etc. */ }

#[async_trait]
impl KeyValueStore for MyDbStore {
    async fn get(&self, key: &str) -> StorageResult<Option<Value>> { /* … */ }
    async fn set(&self, key: &str, value: Value) -> StorageResult<()> { /* … */ }
    // … the rest of the trait
}

#[tokio::main]
async fn main() {
    let kv: Arc<dyn KeyValueStore> = Arc::new(MyDbStore { /* … */ });
    let pldag = Pldag::new_custom(Arc::new(NodeStore::new(kv)));
    // build, query, snapshot — all backed by your DB.
}
```

`NodeStore` is the default `NodeStoreTrait` implementation layered on top of any `KeyValueStore`; if you want to override the DAG-level behaviour as well, implement `NodeStoreTrait` directly.

---

## Notes

- All `set_*` functions accept any iterable type whose items implement `ToString`.
- `Pldag::new()` returns a model backed by an in-memory store. Use `Pldag::new_custom(store)` to provide your own backend.
- Storage backend failures surface as `ModelError::Backend(StorageError)` and bubble up from `?` automatically via the `From<StorageError> for ModelError` impl.

## License

MIT.
