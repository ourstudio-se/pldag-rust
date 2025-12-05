use indexmap::IndexMap;
use pldag::{Bound, Pldag};

fn main() {
    // Build your PL-DAG
    // For example, we create a model of three boolean variables x, y and z.
    // We bind them to an OR constraint.
    let mut pldag: Pldag = Pldag::new();

    // First setup the primitive variables
    pldag.set_primitive("x", (0, 1));
    pldag.set_primitive("y", (0, 1));
    pldag.set_primitive("z", (0, 1));

    // A reference ID is returned
    let root = pldag.set_or(vec!["x", "y", "z"]);

    // 1. Validate a combination:
    let mut inputs: IndexMap<&str, Bound> = IndexMap::new();
    let validated = pldag.propagate_default();
    // Since nothing is given, and all other variables implicitly have bounds (0, 1) from the pldag model,
    // the root will be (0,1) since there's not enough information to evaluate the root `or` node.
    println!("Root valid? {}", *validated.get(&root).unwrap() == (1, 1)); // This will be false

    // If we however fix x to be zero, then we can check the result
    inputs.insert("x", (0, 0));
    let revalidated = pldag.propagate(inputs.clone());
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be false

    // However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
    inputs.insert("y", (1, 1));
    inputs.insert("z", (1, 1));
    let revalidated = pldag.propagate(inputs.clone());
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be true

    // Build a simple OR‑of‑three model
    let mut pldag = Pldag::new();
    pldag.set_primitive("x", (0, 1));
    pldag.set_primitive("y", (0, 1));
    pldag.set_primitive("z", (0, 1));
    let root = pldag.set_or(vec!["x", "y", "z"]);

    // 1. Validate a combination
    let validated = pldag.propagate_default();
    println!("root bound = {:?}", validated[&root]);
}
