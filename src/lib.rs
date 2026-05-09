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
//! # async fn run() -> pldag::Result<()> {
//! let model = Pldag::new();
//! model.set_primitive("a", (0, 1)).await;
//! model.set_primitive("b", (0, 1)).await;
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
//! All fallible operations return [`Result<T, PldagError>`]. Storage backends
//! return [`StorageResult<T, StorageError>`], which is automatically converted
//! into [`PldagError::Storage`] when bubbled up.
//!
//! [`Result<T, PldagError>`]: Result
//! [`StorageResult<T, StorageError>`]: StorageResult

#![warn(missing_docs)]

mod error;
mod pldag;
mod storage;

pub use error::{PldagError, Result, StorageError, StorageResult};
pub use pldag::{ID, Bound, Pldag, Node, Constraint, SparsePolyhedron, DensePolyhedron, CompiledDag, Kind, Coef, Scratch};
pub use storage::{InMemoryStore, KeyValueStore, NodeStore, NodeStoreTrait};
