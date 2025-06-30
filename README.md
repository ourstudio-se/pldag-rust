# Primitive Logic Directed Acyclic Graph (PL‑DAG)

A **Primitive Logic Directed Acyclic Graph** (PL‑DAG) is a DAG in which every node encodes a logical operation and every leaf represents a literal. Interior nodes freely express arbitrary Boolean combinations of their predecessors—for example, an AND‑node evaluates to `true` only if *all* of its incoming nodes (or leaves) evaluate to `true`. This flexibility makes the PL‑DAG both powerful and easy to work with.

---

## ✨ Key Features

| Area                   | What you get                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Modelling**          | Build Boolean/linear constraint systems in a single graph representation.                                   |
| **Analysis**           | Fast bound‑propagation (`propagate*`) and coefficient accumulation (`propagate_coefs*`).                    |
| **Export**             | `to_sparse_polyhedron()` generates a polyhedral ILP model ready for any solver.                             |
| **🧩 Optional solver** | Turn on the `glpk` feature to link against [GLPK](https://www.gnu.org/software/glpk/) and solve in‑process. |

---

## Install

### 1  — Modelling‑only (MIT licence)

```bash
cargo add pldag-rust
```

This pulls *no* GPL code; you can ship the resulting binary under any licence compatible with MIT.

### 2  — Modelling **+** in‑process GLPK solver (GPL v3+ applies)

```bash
cargo add pldag-rust --features glpk
```

Enabling the `glpk` feature links to the GNU Linear Programming Kit (GLPK). If you **distribute** a binary built with this feature you must meet the requirements of the GPL‑3.0‑or‑later.

> **Heads‑up:** Leaving the feature off keeps *all* code MIT‑licensed. The choice is completely under your control at `cargo build` time.

---

## Core Routines

### 1. `propagate`

```rust
fn propagate(
    &self,
    assignment: &Assignment, // IndexMap<String, Bound>
) -> Assignment;
```

*Propagates bounds bottom‑up through the DAG and returns a map of node → bound (`(min, max)`).*

### 2. `propagate_coefs`

```rust
fn propagate_coefs(
    &self,
    assignment: &Assignment,
) -> ValuedAssignment; // IndexMap<String, MultiBound>
```

*Propagates both bounds **and** linear coefficients (`MultiBound = (Bound, VBound)`).*

### 3. Convenience variants

```rust
fn propagate_default(&self) -> Assignment;
fn propagate_coefs_default(&self) -> ValuedAssignment;
```

### 4. `to_sparse_polyhedron`

```rust
fn to_sparse_polyhedron(
    &self,
    double_binding: bool,
    integer_constraints: bool,
    fixed_constraints: bool,
) -> SparsePolyhedron;
```

*Emits a sparse polyhedral representation suitable for ILP solvers (GLPK, CPLEX, Gurobi, …).*
`SparsePolyhedron` implements `serde::Serialize`, so you can also ship it over HTTP to a remote solver service if you prefer.

### 5. Node management helpers

```rust
fn set_coef(&mut self, id: ID, coefficient: f64);
fn get_coef(&self, id: &ID) -> f64;
fn get_objective(&self) -> IndexMap<String, f64>;
fn set_primitive(&mut self, id: ID, bound: Bound);
```

---

## Quick Example

```rust
use indexmap::IndexMap;
use pldag::{Pldag, Bound};

// Build a simple OR‑of‑three model
let mut pldag = Pldag::new();
pldag.set_primitive("x", (0, 1));
pldag.set_primitive("y", (0, 1));
pldag.set_primitive("z", (0, 1));
let root = pldag.set_or(["x", "y", "z"]);

// 1. Validate a combination
let validated = pldag.propagate_default();
println!("root bound = {:?}", validated[&root]);

// 2. Optimise with coefficients
pldag.set_coef("x", 1.0);
pldag.set_coef("y", 2.0);
pldag.set_coef("z", 3.0);
pldag.set_coef(&root, -1.0);
let scored = pldag.propagate_coefs_default();
println!("root value = {:?}", scored[&root].1);
```

---

## License

* **Library code:** MIT (permissive).
* **Optional solver:** If you build with `--features glpk`, you link against GLPK, which is **GPL‑3.0‑or‑later**. Distributing such a binary triggers the GPL’s obligations.

You choose the trade‑off: leave the feature off for a fully permissive dependency tree, or enable it for a batteries‑included ILP solver.

---

Enjoy building and evaluating logical models with PL‑DAG! 🎉
