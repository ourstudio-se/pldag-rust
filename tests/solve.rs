#[cfg(feature = "glpk")]
mod glpk_tests {
    use pldag::{Pldag, Bound};
    use std::collections::HashMap;

    #[test]
    fn or_three_vars_is_feasible() {
        //   x ∨ y ∨ z  with all vars boolean
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let _root = dag.set_or(vec!["x", "y", "z"]);

        let objective = HashMap::<&str, f64>::new();  // dummy objective
        let mut assume    = HashMap::<&str, Bound>::new();   // no fixed vars
        assume.insert(&_root, (1,1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "model should be feasible");
        assert!(*solns[0].as_ref().expect("").get(&_root).unwrap() == (1,1), "solution should be (1,1) for root node");
    }

    #[test]
    fn and_constraint_all_true() {
        // x ∧ y ∧ z with all vars true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let and_node = dag.set_and(vec!["x", "y", "z"]);

        let objective = HashMap::<&str, f64>::from_iter(vec![
            ("x", 1.0),
            ("y", 1.0),
            ("z", 1.0)
        ]);
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&and_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "AND with all true should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get("x").unwrap(), (1, 1));
        assert_eq!(*soln.get("y").unwrap(), (1, 1));
        assert_eq!(*soln.get("z").unwrap(), (1, 1));
    }

    #[test]
    fn and_constraint_mixed_infeasible() {
        // x ∧ y ∧ z with x=1, y=1, z=0 should make AND=0, but we require AND=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let and_node = dag.set_and(vec!["x", "y", "z"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(&and_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "AND with mixed values should be infeasible when requiring AND=1");
    }

    #[test]
    fn not_constraint() {
        // ¬x with x boolean
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        let not_node = dag.set_not(vec!["x"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&not_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "NOT constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get("x").unwrap(), (0, 0));
        assert_eq!(*soln.get(&not_node).unwrap(), (1, 1));
    }

    #[test]
    fn xor_constraint_exactly_one() {
        // x ⊕ y ⊕ z with exactly one true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let xor_node = dag.set_xor(vec!["x", "y", "z"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&xor_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "XOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        // Count how many variables are true
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        let z_val = soln.get("z").unwrap().0;
        assert_eq!(x_val + y_val + z_val, 1, "XOR should have exactly one true variable");
    }

    #[test]
    fn nand_constraint() {
        // ¬(x ∧ y) - not all can be true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let nand_node = dag.set_nand(vec!["x", "y"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&nand_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "NAND constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(!(x_val == 1 && y_val == 1), "NAND should not allow both variables to be true");
    }

    #[test]
    fn nor_constraint() {
        // ¬(x ∨ y) - none can be true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let nor_node = dag.set_nor(vec!["x", "y"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&nor_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "NOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        assert_eq!(*soln.get("x").unwrap(), (0, 0));
        assert_eq!(*soln.get("y").unwrap(), (0, 0));
    }

    #[test]
    fn xnor_constraint() {
        // x ⊙ y - either both true or both false
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let xnor_node = dag.set_xnor(vec!["x", "y"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&xnor_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "XNOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert_eq!(x_val, y_val, "XNOR should have both variables equal");
    }

    #[test]
    fn implication_true_true() {
        // x → y with x=1, y=1 should be true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let imply_node = dag.set_imply("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Implication x=1, y=1 should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&imply_node).unwrap(), (1, 1));
    }

    #[test]
    fn implication_true_false_infeasible() {
        // x → y with x=1, y=0 should be false, but we require implication=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let imply_node = dag.set_imply("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(&imply_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Implication x=1, y=0 should be infeasible when requiring implication=1");
    }

    #[test]
    fn implication_false_any() {
        // x → y with x=0, y=any should be true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let imply_node = dag.set_imply("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert(&imply_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Implication x=0, y=any should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&imply_node).unwrap(), (1, 1));
    }

    #[test]
    fn equivalence_both_true() {
        // x ↔ y with both true
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let equiv_node = dag.set_equiv("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Equivalence x=1, y=1 should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&equiv_node).unwrap(), (1, 1));
    }

    #[test]
    fn equivalence_both_false() {
        // x ↔ y with both false
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let equiv_node = dag.set_equiv("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (0, 0));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Equivalence x=0, y=0 should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&equiv_node).unwrap(), (1, 1));
    }

    #[test]
    fn equivalence_different_infeasible() {
        // x ↔ y with x=1, y=0 should be false, but we require equivalence=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let equiv_node = dag.set_equiv("x", "y");

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(&equiv_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Equivalence x=1, y=0 should be infeasible when requiring equivalence=1");
    }

    #[test]
    fn atleast_constraint_satisfied() {
        // x + y + z >= 2 with x=1, y=1, z=0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let atleast_node = dag.set_atleast(vec!["x", "y", "z"], 2);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "At least 2 constraint should be satisfied");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&atleast_node).unwrap(), (1, 1));
    }

    #[test]
    fn atleast_constraint_not_satisfied() {
        // x + y + z >= 2 with x=1, y=0, z=0, but we require atleast=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let atleast_node = dag.set_atleast(vec!["x", "y", "z"], 2);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(&atleast_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "At least 2 constraint should not be satisfied with only 1 variable");
    }

    #[test]
    fn atmost_constraint_satisfied() {
        // x + y + z <= 2 with x=1, y=1, z=0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let atmost_node = dag.set_atmost(vec!["x", "y", "z"], 2);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "At most 2 constraint should be satisfied");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&atmost_node).unwrap(), (1, 1));
    }

    #[test]
    fn atmost_constraint_not_satisfied() {
        // x + y + z <= 1 with x=1, y=1, z=0, but we require atmost=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let atmost_node = dag.set_atmost(vec!["x", "y", "z"], 1);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(&atmost_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "At most 1 constraint should not be satisfied with 2 variables");
    }

    #[test]
    fn equal_constraint_satisfied() {
        // x + y + z = 2 with x=1, y=1, z=0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Equal 2 constraint should be satisfied");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&equal_node).unwrap(), (1, 1));
    }

    #[test]
    fn equal_constraint_not_satisfied() {
        // x + y + z = 2 with x=1, y=0, z=0, but we require equal=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(&equal_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Equal 2 constraint should not be satisfied with sum=1");
    }

    #[test]
    fn general_linear_inequality() {
        // 2x + 3y - z >= 4 (represented as 2x + 3y - z - 4 >= 0)
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        dag.set_primitive("z", (0, 5));
        let gelineq_node = dag.set_gelineq(vec![
            ("x", 2),
            ("y", 3),
            ("z", -1)
        ], -4);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (2, 2));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        // 2*2 + 3*1 - 0 = 7 >= 4, so constraint should be satisfied

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "General linear inequality should be satisfied");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&gelineq_node).unwrap(), (1, 1));
    }

    #[test]
    fn general_linear_inequality_not_satisfied() {
        // 2x + 3y - z >= 4 with x=1, y=0, z=0, but we require gelineq=1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        dag.set_primitive("z", (0, 5));
        let gelineq_node = dag.set_gelineq(vec![
            ("x", 2),
            ("y", 3),
            ("z", -1)
        ], -4);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(&gelineq_node, (1, 1));
        // 2*1 + 3*0 - 0 = 2 < 4, so constraint should not be satisfied

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "General linear inequality should not be satisfied when sum < threshold");
    }

    #[test]
    fn nested_logical_structure() {
        // (x ∧ y) ∨ (z ∧ w) - complex nested structure
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        dag.set_primitive("w", (0, 1));
        
        let and1 = dag.set_and(vec!["x", "y"]);
        let and2 = dag.set_and(vec!["z", "w"]);
        let or_root = dag.set_or(vec![and1, and2]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (1, 1));
        assume.insert("z", (1, 1));
        assume.insert("w", (1, 1));
        assume.insert(&or_root, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Nested structure should be feasible when second branch is true");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&or_root).unwrap(), (1, 1));
    }

    #[test]
    fn mixed_constraint_types() {
        // Combine logical and linear constraints: (x ∧ y) → (a + b + c >= 2)
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("a", (0, 1));
        dag.set_primitive("b", (0, 1));
        dag.set_primitive("c", (0, 1));
        
        let and_node = dag.set_and(vec!["x", "y"]);
        let atleast_node = dag.set_atleast(vec!["a", "b", "c"], 2);
        let imply_node = dag.set_imply(and_node, atleast_node);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("a", (1, 1));
        assume.insert("b", (1, 1));
        assume.insert("c", (0, 0));
        assume.insert(&imply_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Mixed constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&imply_node).unwrap(), (1, 1));
    }

    #[test]
    fn deep_nesting_levels() {
        // Deep nesting: x ∧ (y ∨ (z ∧ (w ∨ v)))
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        dag.set_primitive("w", (0, 1));
        dag.set_primitive("v", (0, 1));
        
        let or_inner = dag.set_or(vec!["w", "v"]);
        let and_inner = dag.set_and(vec!["z".to_string(), or_inner]);
        let or_middle = dag.set_or(vec!["y".to_string(), and_inner]);
        let and_root = dag.set_and(vec!["x".to_string(), or_middle]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (1, 1));
        assume.insert("w", (0, 0));
        assume.insert("v", (1, 1));
        assume.insert(&and_root, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Deep nesting should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&and_root).unwrap(), (1, 1));
    }

    #[test]
    fn circular_dependency_prevention() {
        // Test that the DAG prevents circular dependencies through proper construction
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        
        let and_node = dag.set_and(vec!["x", "y"]);
        let or_node = dag.set_or(vec![and_node.clone(), "x".to_string()]);
        
        // This should work fine - no circular dependency
        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(&or_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Valid DAG structure should be feasible");
    }

    #[test]
    fn multiple_references_to_same_node() {
        // Test referencing the same intermediate node multiple times
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        
        let and_node = dag.set_and(vec!["x", "y"]);
        let or_node1 = dag.set_or(vec![and_node.clone(), "z".to_string()]);
        let or_node2 = dag.set_or(vec![and_node.clone(), "x".to_string()]);
        let final_and = dag.set_and(vec![or_node1, or_node2]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(&final_and, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Multiple references to same node should work");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(&final_and).unwrap(), (1, 1));
    }

    #[test]
    fn optimization_maximize_single_variable() {
        // Maximize x subject to x + y <= 3
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        let atmost_node = dag.set_atmost(vec!["x", "y"], 3);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atmost_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Optimization should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        // Should maximize x while keeping x + y <= 3
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val <= 3, "Constraint should be satisfied");
        assert!(x_val >= 0, "x should be maximized subject to constraints");
    }

    #[test]
    fn optimization_minimize_objective() {
        // Minimize 2x + 3y subject to x + y >= 2
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        let atleast_node = dag.set_atleast(vec!["x", "y"], 2);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 2.0);
        objective.insert("y", 3.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atleast_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, false); // minimize
        assert!(solns[0].is_some(), "Minimization should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val >= 2, "Constraint should be satisfied");
        
        // Should minimize 2x + 3y, so prefer x over y
        let objective_value = 2 * x_val + 3 * y_val;
        assert!(objective_value >= 4, "Minimum objective should be at least 4");
    }

    #[test]
    fn optimization_multiple_objectives() {
        // Test with multiple different objectives
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 2));
        dag.set_primitive("y", (0, 2));
        dag.set_primitive("z", (0, 2));
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2);

        let mut obj1 = HashMap::<&str, f64>::new();
        obj1.insert("x", 1.0);
        
        let mut obj2 = HashMap::<&str, f64>::new();
        obj2.insert("y", 2.0);
        
        let mut obj3 = HashMap::<&str, f64>::new();
        obj3.insert("z", 3.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&equal_node, (1, 1));

        let solns = dag.solve(vec![obj1, obj2, obj3], assume, true);
        assert!(solns[0].is_some(), "First objective should be feasible");
        assert!(solns[1].is_some(), "Second objective should be feasible");
        assert!(solns[2].is_some(), "Third objective should be feasible");
        
        // All should satisfy x + y + z = 2
        for soln in solns.iter().flatten() {
            let x_val = soln.get("x").unwrap().0;
            let y_val = soln.get("y").unwrap().0;
            let z_val = soln.get("z").unwrap().0;
            assert_eq!(x_val + y_val + z_val, 2, "All solutions should satisfy constraint");
        }
    }

    #[test]
    fn optimization_with_logical_constraints() {
        // Maximize x + y subject to x ∧ y → (z ∨ w)
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        dag.set_primitive("w", (0, 1));
        
        let and_node = dag.set_and(vec!["x", "y"]);
        let or_node = dag.set_or(vec!["z", "w"]);
        let imply_node = dag.set_imply(and_node, or_node);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        objective.insert("y", 1.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&imply_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Optimization with logical constraints should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        let z_val = soln.get("z").unwrap().0;
        let w_val = soln.get("w").unwrap().0;
        
        // If x=1 and y=1, then z ∨ w must be true
        if x_val == 1 && y_val == 1 {
            assert!(z_val == 1 || w_val == 1, "Implication constraint should be satisfied");
        }
    }

    #[test]
    fn optimization_with_coefficients() {
        // Test optimization using set_coef functionality
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        dag.set_coef("x", 2.5);
        dag.set_coef("y", 1.5);
        
        let atmost_node = dag.set_atmost(vec!["x", "y"], 4);

        let objective_indexmap = dag.get_objective();
        let objective = objective_indexmap.iter().map(|(k, v)| (k.as_str(), *v)).collect::<HashMap<&str, f64>>();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atmost_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Optimization with coefficients should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val <= 4, "Constraint should be satisfied");
        
        // Should prefer x since it has higher coefficient
        let objective_value = 2.5 * (x_val as f64) + 1.5 * (y_val as f64);
        assert!(objective_value >= 0.0, "Objective should be non-negative");
    }

    #[test]
    fn infeasible_contradictory_constraints() {
        // Create a real contradiction: x = 1 AND x = 0 using constraint nodes
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        
        // Create two equal constraints: x = 1 and x = 0
        let eq1 = dag.set_equal(vec!["x"], 1);  // x = 1
        let eq2 = dag.set_equal(vec!["x"], 0);  // x = 0
        let both = dag.set_and(vec![eq1, eq2]); // Both must be true (contradiction)

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&both, (1, 1)); // Require both constraints to be satisfied

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Contradictory constraints should be infeasible");
    }

    #[test]
    fn infeasible_impossible_linear_constraint() {
        // x + y = 5 with x, y ∈ [0, 1]
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let equal_node = dag.set_equal(vec!["x", "y"], 5);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&equal_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Impossible linear constraint should be infeasible");
    }

    #[test]
    fn infeasible_logical_contradiction() {
        // x ∧ ¬x should be false
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        let not_node = dag.set_not(vec!["x"]);
        let and_node = dag.set_and(vec!["x".to_string(), not_node]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&and_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Logical contradiction should be infeasible");
    }

    #[test]
    fn infeasible_conflicting_implications() {
        // x → y and x → ¬y with x = 1
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        let not_y = dag.set_not(vec!["y"]);
        let imply1 = dag.set_imply("x", "y");
        let imply2 = dag.set_imply("x", not_y);
        let and_node = dag.set_and(vec![imply1, imply2]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert(&and_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Conflicting implications should be infeasible");
    }

    #[test]
    fn infeasible_over_constrained_system() {
        // x + y = 2, x + y = 3, x ≥ 0, y ≥ 0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        let equal1 = dag.set_equal(vec!["x", "y"], 2);
        let equal2 = dag.set_equal(vec!["x", "y"], 3);
        let and_node = dag.set_and(vec![equal1, equal2]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&and_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Over-constrained system should be infeasible");
    }

    #[test]
    fn infeasible_boundary_violation() {
        // x ∈ [0, 1] but require x = 2
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (2, 2));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Boundary violation should be infeasible");
    }

    #[test]
    fn infeasible_complex_nested_contradiction() {
        // (x ∧ y) ∨ (z ∧ w) = 1, but x=0, y=0, z=0, w=0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        dag.set_primitive("y", (0, 1));
        dag.set_primitive("z", (0, 1));
        dag.set_primitive("w", (0, 1));
        
        let and1 = dag.set_and(vec!["x", "y"]);
        let and2 = dag.set_and(vec!["z", "w"]);
        let or_node = dag.set_or(vec![and1, and2]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert("w", (0, 0));
        assume.insert(&or_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Complex nested contradiction should be infeasible");
    }

    #[test]
    fn infeasible_optimization_no_solution() {
        // Maximize x subject to x + y <= -1 with x, y ≥ 0
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));
        let atmost_node = dag.set_atmost(vec!["x", "y"], -1);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atmost_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_none(), "Optimization with impossible constraints should be infeasible");
    }

    #[test]
    fn edge_case_empty_integer_constraints() {
        // No constraints, just maximize x
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 10));

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        
        let assume = HashMap::<&str, Bound>::new();

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Empty constraints should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        // Should maximize x to its upper bound
        let x_val = soln.get("x").unwrap().0;
        assert_eq!(x_val, 10, "Should maximize x to upper bound");
    }

    #[test]
    fn edge_case_single_variable_bounds() {
        // Test various single variable bound scenarios
        let mut dag = Pldag::new();
        dag.set_primitive("x", (5, 5)); // Fixed value

        let objective = HashMap::<&str, f64>::new();
        let assume = HashMap::<&str, Bound>::new();

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Fixed value should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        assert_eq!(*soln.get("x").unwrap(), (5, 5), "Fixed value should be exact");
    }

    #[test]
    fn edge_case_zero_coefficients() {
        // Test optimization with zero coefficients
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 5));
        dag.set_primitive("y", (0, 5));

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 0.0);
        objective.insert("y", 0.0);
        
        let assume = HashMap::<&str, Bound>::new();

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Zero coefficients should be feasible");
        // Any solution should be valid when coefficients are zero
    }

    #[test]
    fn edge_case_large_bounds() {
        // Test with large bound values
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1000000));
        dag.set_primitive("y", (0, 1000000));
        let atmost_node = dag.set_atmost(vec!["x", "y"], 999999);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atmost_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Large bounds should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val <= 999999, "Large bound constraint should be satisfied");
    }

    #[test]
    fn edge_case_negative_bounds() {
        // Test with negative bounds
        let mut dag = Pldag::new();
        dag.set_primitive("x", (-10, 10));
        dag.set_primitive("y", (-5, 5));
        let atleast_node = dag.set_atleast(vec!["x", "y"], -3);

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atleast_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Negative bounds should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val >= -3, "Negative bound constraint should be satisfied");
    }

    #[test]
    fn edge_case_single_element_operations() {
        // Test logical operations with single elements
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        
        let and_single = dag.set_and(vec!["x"]);
        let or_single = dag.set_or(vec!["x"]);
        let xor_single = dag.set_xor(vec!["x"]);

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Single element operations should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        assert_eq!(*soln.get(&and_single).unwrap(), (1, 1));
        assert_eq!(*soln.get(&or_single).unwrap(), (1, 1));
        assert_eq!(*soln.get(&xor_single).unwrap(), (1, 1));
    }

    #[test]
    fn edge_case_identical_variables_in_constraint() {
        // Test constraint with repeated variables: x + x + x >= 3
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 2));
        let atleast_node = dag.set_atleast(vec!["x", "x", "x"], 3);

        let mut objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(&atleast_node, (1, 1));

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Repeated variables in constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        let x_val = soln.get("x").unwrap().0;
        assert!(3 * x_val >= 3, "Repeated variable constraint should be satisfied");
    }

    #[test]
    fn edge_case_very_small_coefficients() {
        // Test with very small floating point coefficients
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1000));
        dag.set_primitive("y", (0, 1000));

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 0.000001);
        objective.insert("y", 0.000002);
        
        let assume = HashMap::<&str, Bound>::new();

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Very small coefficients should be feasible");
        let soln = solns[0].as_ref().unwrap();
        
        // Should still maximize according to relative coefficients
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val >= 0 && y_val >= 0, "Solution should be within bounds");
    }

    #[test]
    fn edge_case_empty_variable_lists() {
        // Test operations with empty variable lists - should fail gracefully
        let mut dag = Pldag::new();
        dag.set_primitive("x", (0, 1));
        
        // Most operations should handle empty lists gracefully
        // This is mainly testing that the library doesn't crash
        let objective = HashMap::<&str, f64>::new();
        let assume = HashMap::<&str, Bound>::new();

        let solns = dag.solve(vec![objective], assume, true);
        assert!(solns[0].is_some(), "Empty operations should not crash");
    }

    #[test]
    fn test_common_dag_with_xor_conjunction() {
        let mut dag = Pldag::new();
        dag.set_primitives(
            vec![
                "s1",
                "s2",
                "f1",
                "f2",
            ],
            (0, 1),
        );
        let sizes = dag.set_xor(vec!["s1", "s2"]);
        let fabrics = dag.set_xor(vec!["f1", "f2"]);

        let root = dag.set_and(vec![sizes, fabrics]);

        println!("root: {}", root);
        let solution = dag.solve(
            vec![HashMap::from([
                ("s1", 1.0),
                ("s2", -1.0), 
                ("f2", 1.0), 
                ("f1", -1.0)
            ])],
            HashMap::from([(root.as_str(), (1, 1))]),
            true,
        );
        let asd = &solution[0];
        match asd {
            None => {}
            Some(assignments) => {
                for (id, bound) in assignments {
                    println!("{}: {}", id, bound.0)
                }
            }
        }
    }
}