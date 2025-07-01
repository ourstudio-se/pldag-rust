use indexmap::IndexMap;
use pldag::{Pldag, Bound};

fn main() {
    // Build your PL-DAG
    // For example, we create a model of three boolean variables x, y and z.
    // We bind them to an OR constraint.
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
    ]);

    // 1. Validate a combination:
    let mut inputs: IndexMap<String, Bound> = IndexMap::new();
    let validated = pldag.propagate(&inputs);
    // Since nothing is given, and all other variables implicitly have bounds (0, 1) from the pldag model,
    // the root will be (0,1) since there's not enough information to evaluate the root `or` node.
    println!("Root valid? {}", *validated.get(&root).unwrap() == (1, 1)); // This will be false

    // If we however fix x to be zero, then we can check the result
    inputs.insert("x".to_string(), (0,0));
    let revalidated = pldag.propagate(&inputs);
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be false

    // However, fixing y and z to 1 will yield the root node to be true (since the root will be true if any of x, y or z is true).
    inputs.insert("y".to_string(), (1,1));
    inputs.insert("z".to_string(), (1,1));
    let revalidated = pldag.propagate(&inputs);
    println!("Root valid? {}", *revalidated.get(&root).unwrap() == (1, 1)); // This will be true

    // 2. Score a configuration:
    // We can score a configuration by setting coefficients on nodes.
    pldag.set_coef("x", 1.0);
    pldag.set_coef("y", 2.0);
    pldag.set_coef("z", 3.0);
    // Add a discount value if the root is true
    pldag.set_coef(&root, -1.0);

    // Use propagate_coefs to get both bounds and accumulated coefficients
    let scores = pldag.propagate_coefs(&inputs);
    // The result contains (bounds, coefficients) for each node
    let root_result = scores.get(&root).unwrap();
    println!("Root bounds: {:?}, Total score: {:?}", root_result.0, root_result.1);

    // And notice what will happen if we remove the x value (i.e. x being (0,1))
    inputs.insert("x".to_string(), (0,1));
    let scores = pldag.propagate_coefs(&inputs);
    // The coefficients will reflect the range of possible values
    let root_result = scores.get(&root).unwrap();
    println!("Root bounds: {:?}, Score range: {:?}", root_result.0, root_result.1);

    // .. and if we set x to be 0, then the score will be more constrained.
    inputs.insert("x".to_string(), (0,0));
    let scores = pldag.propagate_coefs(&inputs);
    let root_result = scores.get(&root).unwrap();
    println!("Root bounds: {:?}, Score: {:?}", root_result.0, root_result.1);

    // .. and if we set y and z to be 0, then the root will be 0.
    inputs.insert("y".to_string(), (0,0));
    inputs.insert("z".to_string(), (0,0));
    let scores = pldag.propagate_coefs(&inputs);
    let root_result = scores.get(&root).unwrap();
    println!("Root bounds: {:?}, Score: {:?}", root_result.0, root_result.1);

    // Build a simple OR‑of‑three model
    let mut pldag = Pldag::new();
    pldag.set_primitive("x".to_string(), (0, 1));
    pldag.set_primitive("y".to_string(), (0, 1));
    pldag.set_primitive("z".to_string(), (0, 1));
    let root = pldag.set_or(vec!["x".to_string(), "y".to_string(), "z".to_string()]);

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
}