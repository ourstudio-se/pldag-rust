# Propositional Logic Directed Acyclic Graph (PL‑DAG)

A **Propositional Logic Directed Acyclic Graph** (PL‑DAG) is a DAG in which every node encodes a logical operation and every leaf represents a literal. Interior nodes freely express arbitrary Boolean combinations of their predecessors—for example, an AND‑node evaluates to `true` only if *all* of its incoming nodes (or leaves) evaluate to `true`. This flexibility makes the PL‑DAG both powerful and easy to work with.

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

### 1. `propagate`

```rust
fn propagate(
    &self,
    assignment: &IndexMap<&str, Bound>, // Assignment = IndexMap<String, Bound>
) -> Assignment;
```

*Propagates bounds bottom‑up through the DAG and returns a map of node → bound (`(min, max)`).*

### 2. `propagate_coefs`

```rust
fn propagate_coefs(
    &self,
    assignment: &IndexMap<&str, Bound>,
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
fn set_coef(&mut self, id: &str, coefficient: f64);
fn get_coef(&self, id: &str) -> f64;
fn get_objective(&self) -> IndexMap<String, f64>;
fn set_primitive(&mut self, id: &str, bound: Bound);
```

### 6. `solve` (Optional GLPK Feature)

```rust
#[cfg(feature = "glpk")]
fn solve(
    &self,
    objectives: Vec<HashMap<&str, f64>>,
    assume: HashMap<&str, Bound>,
    maximize: bool,
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
let root = pldag.set_or(vec!["x", "y", "z"]).unwrap();

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
let root = pldag.set_or(vec!["x", "y", "z"]).unwrap();

// Set up the objective function: maximize x + 2y + 3z
let mut objective = HashMap::new();
objective.insert("x", 1.0);
objective.insert("y", 2.0);
objective.insert("z", 3.0);

// Constraints: require that the OR constraint is satisfied
let mut assumptions = HashMap::new();
assumptions.insert(&root, (1, 1)); // root must be true

// Solve the optimization problem
let solutions = pldag.solve(vec![objective], assumptions, true);

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
- <i>Please note that when a composite is either a tautology (always true) or a contradition (always false), these are automatically transformed into a primitive with fixed bounds set to (0,0) if contradition and (1,1) if tautology.</i>

## License

* **Library code:** MIT (permissive).
* **Optional solver:** If you build with `--features glpk`, you link against GLPK, which is **GPL‑3.0‑or‑later**. Distributing such a binary triggers the GPL’s obligations.

You choose the trade‑off: leave the feature off for a fully permissive dependency tree, or enable it for a batteries‑included ILP solver.

---

Enjoy building and evaluating logical models with PL‑DAG! 🎉
