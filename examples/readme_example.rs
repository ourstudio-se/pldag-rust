use std::collections::HashMap;
use pldag::{Bound, Pldag};

#[tokio::main]
async fn main() {
    // Build your PL-DAG
    // For example, we create a model of three boolean variables x, y and z.
    // We bind them to an OR constraint.
    let pldag: Pldag = Pldag::new();

    // First setup the primitive variables
    pldag.set_primitive("x", (0, 1)).await.unwrap();
    pldag.set_primitive("y", (0, 1)).await.unwrap();
    pldag.set_primitive("z", (0, 1)).await.unwrap();

    // A reference ID is returned
    let root = pldag.set_or(vec!["x", "y", "z"]).await.unwrap();

    // Export a sub dag
    let dag = pldag.sub_dag(vec![root.clone()]).await.unwrap();

    // 1. Validate a combination:
    let mut inputs: HashMap<&str, Bound> = HashMap::new();
    let validated = Pldag::propagate_dag(&dag, inputs.clone()).unwrap();
    // Since nothing is given, and all other variables implicitly have bounds (0, 1) from the pldag model,
    // the root will be (0,1) since there's not enough information to evaluate the root `or` node.
    println!("Root valid? {}", *validated.get(&root).unwrap() == (1, 1)); // This will be false

    // If we however fix x to be zero, then we can check the result
    inputs.insert("x", (0, 0));
    let revalidated = pldag.propagate(inputs.clone()).await.unwrap();
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be false

    // However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
    inputs.insert("y", (1, 1));
    inputs.insert("z", (1, 1));
    let revalidated = pldag.propagate(inputs.clone()).await.unwrap();
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be true

    // Build a simple OR‑of‑three model
    let pldag = Pldag::new();
    pldag.set_primitive("x", (0, 1)).await.unwrap();
    pldag.set_primitive("y", (0, 1)).await.unwrap();
    pldag.set_primitive("z", (0, 1)).await.unwrap();
    let root = pldag.set_or(vec!["x", "y", "z"]).await.unwrap();
    let dag = pldag.sub_dag(vec![root.clone()]).await.unwrap();

    // 1. Validate a combination
    let validated = dag.propagate(inputs).unwrap();
    println!("root bound = {:?}", validated[&root]);
}
