#[cfg(feature = "glpk")]
mod glpk_tests {
    use pldag::{Pldag, Bound};
    use std::collections::HashMap;

    #[test]
    fn or_three_vars_is_feasible() {
        //   x ∨ y ∨ z  with all vars boolean
        let mut dag = Pldag::new();
        dag.set_primitive("x".to_string(), (0, 1));
        dag.set_primitive("y".to_string(), (0, 1));
        dag.set_primitive("z".to_string(), (0, 1));
        let _root = dag.set_or(vec!["x".to_string(), "y".to_string(), "z".to_string()]);

        let objective = HashMap::<String, f64>::new();  // dummy objective
        let mut assume    = HashMap::<String, Bound>::new();   // no fixed vars
        assume.insert(_root.clone(), (1,1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "model should be feasible");
        assert!(*solns[0].as_ref().expect("").get(&_root).unwrap() == (1,1), "solution should be (1,1) for root node");
    }
}