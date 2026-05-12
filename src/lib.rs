//! # pldag — Primitive Logic Directed Acyclic Graphs
//!
//! `pldag` lets you build, evaluate, and reason about boolean logic expressed
//! as a DAG of primitive variables and linear-combination constraints.
//!
//! It is useful for modelling configuration rules, feature flags with
//! dependencies, product configurators, or any domain where the question
//! _"given these inputs, which outputs are forced to true / false / unknown?"_
//! must be answered repeatedly and quickly.
//!
//! ## Concepts
//!
//! - **[`Pldag`]** — the editable model. Add primitive variables and logical
//!   constraints (AND, OR, XOR, NOT, at-least, at-most, equal, …). Backed by
//!   a pluggable async [`NodeStoreTrait`] so the same model can live in memory
//!   or in a database.
//! - **[`CompiledDag`]** — a compact, immutable, indexed snapshot of a `Pldag`,
//!   produced by [`Pldag::dag`]. Optimised for fast propagation.
//! - **Propagation** — given assignments for some variables,
//!   [`CompiledDag::propagate`] tightens the bounds of every reachable node.
//!   For hot loops, [`CompiledDag::propagate_with_scratch`] reuses buffers via
//!   a [`Scratch`].
//! - **Polyhedra** — [`SparsePolyhedron`] and [`DensePolyhedron`] turn the DAG
//!   into a system of linear inequalities suitable for ILP solvers.
//!
//! ## Quick start
//!
//! ```no_run
//! use pldag::{Pldag, Scratch};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = Pldag::new();
//! model.set_primitive("a", (0, 1)).await?;
//! model.set_primitive("b", (0, 1)).await?;
//! let root = model.set_and(vec!["a", "b"]).await?;
//!
//! // Compile once, propagate many times.
//! let dag = model.dag().await?;
//! let mut scratch = Scratch::new();
//!
//! let result = dag.propagate_with_scratch(
//!     [("a", (1, 1)), ("b", (1, 1))],
//!     &mut scratch,
//! )?;
//! assert_eq!(result.get(&root).unwrap(), &(1, 1));
//! # Ok(())
//! # }
//! ```
//!
//! ## Errors
//!
//! Fallible operations return one of two precise result types:
//!
//! - [`ComputeResult<T>`] — pure in-memory operations on a [`CompiledDag`]
//!   (propagation, tightening, ranks, polyhedron conversion). Errors are
//!   modelled by [`ComputeError`].
//! - [`ModelResult<T>`] — operations that read or mutate the model's storage
//!   (`set_*`, `delete_node`, `dag`, `sub_dag`, `get_node`). Errors are
//!   modelled by [`ModelError`], which wraps backend-level [`StorageError`]s.
//!
//! The two enums are disjoint: no variant is shared, so each function signature
//! precisely describes its failure modes.

#![warn(missing_docs)]

mod error;
mod pldag;
mod storage;

pub use error::{ComputeError, ComputeResult, StorageError, StorageResult, ModelError, ModelResult};
pub use pldag::{ID, Bound, Pldag, Node, Constraint, SparsePolyhedron, DensePolyhedron, CompiledDag, Kind, Coef, Scratch};
pub use storage::{InMemoryStore, KeyValueStore, NodeStore, NodeStoreTrait};
