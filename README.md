# Propositional Logic Directed Acyclic Graph (PL‑DAG)

A PL-DAG is a directed acyclic graph where each leaf defines a discrete integer domain (e.g., x ∈ [-5, 3]) and each internal node defines a linear inequality over its predecessors. The graph therefore represents a structured system of discrete constraints, allowing arbitrary compositions of integer ranges and linear relations while naturally sharing repeated sub-expressions.

PL-DAGs are especially well suited for describing and solving discrete optimization problems—problems where you want to make the best possible choice under a set of rules.
Typical examples include:

- Choosing the best product configuration given technical constraints

- Optimizing costs or performance while respecting limits

- Selecting combinations of components that must work together

- Modeling resource limits, capacities, or integer-valued decisions

- Exploring what combinations of values are feasible or optimal

Because the PL-DAG breaks everything into small, reusable pieces, it becomes easy to build complex models from simple parts and to solve them with efficient algorithms.

---

## ✨ Key Features

| Area                   | What you get                                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| **Modelling**          | Build Boolean/linear constraint systems in a single graph representation.                                   |
| **Analysis**           | Fast bound‑propagation (`propagate*`).                    |
| **Export**             | `to_sparse_polyhedron()` generates a polyhedral ILP model ready for any solver.                             |
| **🧩 Optional solver** | Turn on the `glpk` feature to link against [GLPK](https://www.gnu.org/software/glpk/) and solve in‑process. |

---

## Install

### 1  — Modelling‑only (MIT licence)

```bash
cargo add pldag
```

This pulls *no* GPL code; you can ship the resulting binary under any licence compatible with MIT.

### 2  — Modelling **+** in‑process GLPK solver (GPL v3+ applies)

```bash
cargo add pldag --features glpk
```

Enabling the `glpk` feature links to the GNU Linear Programming Kit (GLPK). If you **distribute** a binary built with this feature you must meet the requirements of the GPL‑3.0‑or‑later.

> **Heads‑up:** Leaving the feature off keeps *all* code MIT‑licensed. The choice is completely under your control at `cargo build` time.

---

## Core Routines

### Analysis & Propagation

#### `propagate`

```rust
fn propagate<K>(
    &self,
    assignment: impl IntoIterator<Item = (K, Bound)>
) -> Result<Assignment>
where K: ToString;
```

*Propagates bounds bottom‑up through the DAG and returns a map of node → bound (`(min, max)`).*

#### `propagate_default`

```rust
fn propagate_default(&self) -> Result<Assignment>;
```

*Convenience method that propagates using default bounds of all primitive variables.*

### Building the Model

#### Primitive Variables

```rust
fn set_primitive(&mut self, id: &str, bound: Bound);
fn set_primitives<K>(&mut self, ids: impl IntoIterator<Item = K>, bound: Bound)
where K: ToString;
```

*Create primitive (leaf) variables with specified bounds. `set_primitives` creates multiple variables with the same bounds.*

#### Linear Constraints

```rust
fn set_gelineq<K>(
    &mut self,
    coefficient_variables: impl IntoIterator<Item = (K, i32)>,
    bias: i32
) -> ID
where K: ToString;
```

*Creates a general linear inequality: `sum(coeff_i * var_i) + bias >= 0`.*

```rust
fn set_atleast<K>(&mut self, references: impl IntoIterator<Item = K>, value: i32) -> ID
where K: ToString;

fn set_atmost<K>(&mut self, references: impl IntoIterator<Item = K>, value: i32) -> ID
where K: ToString;

fn set_equal<K, I>(&mut self, references: I, value: i32) -> ID
where K: ToString, I: IntoIterator<Item = K> + Clone;
```

*`set_atleast`: Creates `sum(variables) >= value`*
*`set_atmost`: Creates `sum(variables) <= value`*
*`set_equal`: Creates `sum(variables) == value`*

#### Logical Constraints

```rust
fn set_and<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_or<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_not<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_xor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_nand<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_nor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;

fn set_xnor<K>(&mut self, references: impl IntoIterator<Item = K>) -> ID
where K: ToString;
```

*Standard logical operations:*
- `set_and`: ALL variables must be true
- `set_or`: AT LEAST ONE variable must be true
- `set_not`: NONE of the variables can be true
- `set_xor`: EXACTLY ONE variable must be true
- `set_nand`: NOT ALL variables can be true
- `set_nor`: NONE of the variables can be true
- `set_xnor`: EVEN NUMBER of variables must be true (including zero)

```rust
fn set_imply<C, Q>(&mut self, condition: C, consequence: Q) -> ID
where C: ToString, Q: ToString;

fn set_equiv<L, R>(&mut self, lhs: L, rhs: R) -> ID
where L: ToString, R: ToString;
```

*`set_imply`: Creates `condition → consequence` (implication)*
*`set_equiv`: Creates `lhs ↔ rhs` (equivalence/biconditional)*

### Export & Solving

#### `to_sparse_polyhedron`

```rust
fn to_sparse_polyhedron(
    &self,
    roots: Vec<ID>,
    double_binding: bool
) -> SparsePolyhedron;
```

*Emits a sparse polyhedral representation suitable for ILP solvers (GLPK, CPLEX, Gurobi, …).*
*`SparsePolyhedron` implements `serde::Serialize`, so you can ship it over HTTP to a remote solver service if you prefer.*

#### `solve` (Optional GLPK Feature)

```rust
#[cfg(feature = "glpk")]
fn solve(
    &self,
    roots: Vec<ID>,
    objectives: Vec<HashMap<&str, f64>>,
    assume: HashMap<&str, Bound>,
    maximize: bool
) -> Vec<Option<Assignment>>;
```

*Solves integer linear programming problems using GLPK. Takes multiple objective functions, fixed variable assumptions, and returns optimal assignments.*

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
let root = pldag.set_or(vec!["x", "y", "z"]);

// Validate a combination
let validated = pldag.propagate_default().unwrap();
println!("root bound = {:?}", validated[&root]);
```

### 3. Solving with GLPK (Optional Feature)

When the `glpk` feature is enabled, you can solve optimization problems directly:

```rust
#[cfg(feature = "glpk")]
use std::collections::HashMap;
use pldag::{Pldag, Bound};

// Build a simple problem: maximize x + 2y + 3z subject to x ∨ y ∨ z
let mut pldag = Pldag::new();
pldag.set_primitive("x", (0, 1));
pldag.set_primitive("y", (0, 1));
pldag.set_primitive("z", (0, 1));
let root = pldag.set_or(vec!["x", "y", "z"]);

// Set up the objective function: maximize x + 2y + 3z
let mut objective = HashMap::new();
objective.insert("x", 1.0);
objective.insert("y", 2.0);
objective.insert("z", 3.0);

// Constraints: require that the OR constraint is satisfied
let mut assumptions = HashMap::new();
assumptions.insert(&root, (1, 1)); // root must be true

// Solve the optimization problem
let solutions = pldag.solve(vec![root.clone()], vec![objective], assumptions, true);

if let Some(solution) = &solutions[0] {
    println!("Optimal solution found:");
    println!("x = {:?}", solution.get("x"));
    println!("y = {:?}", solution.get("y"));
    println!("z = {:?}", solution.get("z"));
    println!("root = {:?}", solution.get(&root));
} else {
    println!("No feasible solution found");
}
```

This example demonstrates:
- **Problem setup**: Creating boolean variables and logical constraints
- **Objective function**: Defining what to optimize (maximize x + 2y + 3z)
- **Assumptions**: Fixing certain variables or constraints (root must be true)
- **Solving**: Using GLPK to find the optimal solution
- **Result interpretation**: Extracting variable values from the solution

---

## Notes

- All `set_*` functions accept any iterable type that can be converted to strings via the `ToString` trait, providing maximum flexibility.

## License

* **Library code:** MIT (permissive).
* **Optional solver:** If you build with `--features glpk`, you link against GLPK, which is **GPL‑3.0‑or‑later**. Distributing such a binary triggers the GPL’s obligations.

You choose the trade‑off: leave the feature off for a fully permissive dependency tree, or enable it for a batteries‑included ILP solver.

---

Enjoy building and evaluating logical models with PL‑DAG! 🎉
