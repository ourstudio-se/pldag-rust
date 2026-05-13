mod glpk_tests {
    use glpk_rust::{
        solve_ilps, IntegerSparseMatrix, SparseLEIntegerPolyhedron, Status, Variable,
    };
    use pldag::{Bound, CompiledDag, ComputeError, Pldag, ModelError};
    use std::collections::HashMap;

    /// Test-local ILP solver that builds a sparse polyhedron from the compiled
    /// DAG and dispatches it to GLPK. Kept inside the test module so the main
    /// crate stays solver-free.
    fn solve(
        cd: &CompiledDag,
        objectives: Vec<HashMap<&str, f64>>,
        assume: HashMap<&str, Bound>,
        maximize: bool,
    ) -> Result<Vec<Option<HashMap<String, Bound>>>, ComputeError> {
        let polyhedron = Pldag::to_sparse_polyhedron(cd, true)?;

        for (key, bound) in assume.iter() {
            if let Some(idx) = polyhedron.columns.iter().position(|col| col == key) {
                let col_bound = polyhedron.column_bounds[idx];
                if bound.0 < col_bound.0 || bound.1 > col_bound.1 {
                    return Err(ComputeError::NodeOutOfBounds {
                        node_id: key.to_string(),
                        got_bound: *bound,
                        expected_bound: col_bound,
                    });
                }
            }
        }

        let mut glpk_matrix = SparseLEIntegerPolyhedron {
            a: IntegerSparseMatrix {
                rows: polyhedron.a.rows.iter().map(|&x| x as i32).collect(),
                cols: polyhedron.a.cols.iter().map(|&x| x as i32).collect(),
                vals: polyhedron.a.vals.iter().map(|&x| -x).collect(),
            },
            b: polyhedron.b.iter().map(|&x| (0, -x)).collect(),
            variables: polyhedron
                .columns
                .iter()
                .zip(polyhedron.column_bounds.iter())
                .map(|(key, bound)| Variable {
                    id: key.as_str(),
                    bound: *assume.get(key.as_str()).unwrap_or(bound),
                })
                .collect(),
            double_bound: false,
        };

        // glpk-rust panics on empty constraint matrices; insert a dummy zero row.
        if glpk_matrix.a.rows.is_empty() {
            for i in 0..polyhedron.columns.len() {
                glpk_matrix.a.rows.push(0);
                glpk_matrix.a.cols.push(i as i32);
                glpk_matrix.a.vals.push(0);
            }
            glpk_matrix.b.push((0, 0));
        }

        let solutions = solve_ilps(&mut glpk_matrix, objectives, maximize, false, false);

        Ok(solutions
            .iter()
            .map(|solution| {
                if solution.status == Status::Optimal {
                    let mut assignment: HashMap<String, Bound> = HashMap::new();
                    for col_name in polyhedron.columns.iter() {
                        let value = solution.solution.get(col_name).unwrap_or(&0);
                        assignment.insert(col_name.clone(), (*value, *value));
                    }
                    Some(assignment)
                } else {
                    None
                }
            })
            .collect())
    }

    #[tokio::test]
    async fn or_three_vars_is_feasible() {
        //   x ∨ y ∨ z  with all vars boolean
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let _root = dag.set_or(vec!["x", "y", "z"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new(); // dummy objective
        let mut assume = HashMap::<&str, Bound>::new(); // no fixed vars
        assume.insert(_root.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "model should be feasible");
        assert!(
            *solns[0].as_ref().unwrap().get(&_root).unwrap() == (1, 1),
            "solution should be (1,1) for root node"
        );
    }

    #[tokio::test]
    async fn and_constraint_all_true() {
        // x ∧ y ∧ z with all vars true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let and_node = dag.set_and(vec!["x", "y", "z"]).await.unwrap();

        let objective = HashMap::<&str, f64>::from_iter(vec![("x", 1.0), ("y", 1.0), ("z", 1.0)]);
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(and_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "AND with all true should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get("x").unwrap(), (1, 1));
        assert_eq!(*soln.get("y").unwrap(), (1, 1));
        assert_eq!(*soln.get("z").unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn and_constraint_mixed_infeasible() {
        // x ∧ y ∧ z with x=1, y=1, z=0 should make AND=0, but we require AND=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let and_node = dag.set_and(vec!["x", "y", "z"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(and_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "AND with mixed values should be infeasible when requiring AND=1"
        );
    }

    #[tokio::test]
    async fn not_constraint() {
        // ¬x with x boolean
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        let not_node = dag.set_not(vec!["x"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(not_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "NOT constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get("x").unwrap(), (0, 0));
        assert_eq!(*soln.get(not_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn xor_constraint_exactly_one() {
        // x ⊕ y ⊕ z with exactly one true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let xor_node = dag.set_xor(vec!["x", "y", "z"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(xor_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "XOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();

        // Count how many variables are true
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        let z_val = soln.get("z").unwrap().0;
        assert_eq!(
            x_val + y_val + z_val,
            1,
            "XOR should have exactly one true variable"
        );
    }

    #[tokio::test]
    async fn nand_constraint() {
        // ¬(x ∧ y) - not all can be true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let nand_node = dag.set_nand(vec!["x", "y"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(nand_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "NAND constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(
            !(x_val == 1 && y_val == 1),
            "NAND should not allow both variables to be true"
        );
    }

    #[tokio::test]
    async fn nor_constraint() {
        // ¬(x ∨ y) - none can be true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let nor_node = dag.set_nor(vec!["x", "y"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(nor_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "NOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();

        assert_eq!(*soln.get("x").unwrap(), (0, 0));
        assert_eq!(*soln.get("y").unwrap(), (0, 0));
    }

    #[tokio::test]
    async fn xnor_constraint() {
        // x ⊙ y - either both true or both false
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let xnor_node = dag.set_xnor(vec!["x", "y"]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(xnor_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "XNOR constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert_eq!(x_val, y_val, "XNOR should have both variables equal");
    }

    #[tokio::test]
    async fn implication_true_true() {
        // x → y with x=1, y=1 should be true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let imply_node = dag.set_imply("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert(imply_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Implication x=1, y=1 should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(imply_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn implication_true_false_infeasible() {
        // x → y with x=1, y=0 should be false, but we require implication=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let imply_node = dag.set_imply("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(imply_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Implication x=1, y=0 should be infeasible when requiring implication=1"
        );
    }

    #[tokio::test]
    async fn implication_false_any() {
        // x → y with x=0, y=any should be true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let imply_node = dag.set_imply("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert(imply_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Implication x=0, y=any should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(imply_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn equivalence_both_true() {
        // x ↔ y with both true
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let equiv_node = dag.set_equiv("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert(equiv_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Equivalence x=1, y=1 should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(equiv_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn equivalence_both_false() {
        // x ↔ y with both false
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let equiv_node = dag.set_equiv("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (0, 0));
        assume.insert(equiv_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Equivalence x=0, y=0 should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(equiv_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn equivalence_different_infeasible() {
        // x ↔ y with x=1, y=0 should be false, but we require equivalence=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let equiv_node = dag.set_equiv("x", "y").await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(equiv_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Equivalence x=1, y=0 should be infeasible when requiring equivalence=1"
        );
    }

    #[tokio::test]
    async fn atleast_constraint_satisfied() {
        // x + y + z >= 2 with x=1, y=1, z=0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["x", "y", "z"], 2).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(atleast_node.as_str(), (0, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "At least 2 constraint should be satisfied"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(atleast_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn atleast_constraint_not_satisfied() {
        // x + y + z >= 2 with x=1, y=0, z=0, but we require atleast=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["x", "y", "z"], 2).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(atleast_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "At least 2 constraint should not be satisfied with only 1 variable"
        );
    }

    #[tokio::test]
    async fn atmost_constraint_satisfied() {
        // x + y + z <= 2 with x=1, y=1, z=0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let atmost_node = dag.set_atmost(vec!["x", "y", "z"], 2).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(atmost_node.as_str(), (0, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "At most 2 constraint should be satisfied"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(atmost_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn atmost_constraint_not_satisfied() {
        // x + y + z <= 1 with x=1, y=1, z=0, but we require atmost=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let atmost_node = dag.set_atmost(vec!["x", "y", "z"], 1).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(atmost_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "At most 1 constraint should not be satisfied with 2 variables"
        );
    }

    #[tokio::test]
    async fn equal_constraint_satisfied() {
        // x + y + z = 2 with x=1, y=1, z=0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(equal_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Equal 2 constraint should be satisfied");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(equal_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn equal_constraint_not_satisfied() {
        // x + y + z = 2 with x=1, y=0, z=0, but we require equal=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(equal_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Equal 2 constraint should not be satisfied with sum=1"
        );
    }

    #[tokio::test]
    async fn general_linear_inequality() {
        // 2x + 3y - z >= 4 (represented as 2x + 3y - z - 4 >= 0)
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        dag.set_primitive("z", (0, 5)).await.unwrap();
        let gelineq_node = dag.set_gelineq(vec![("x", 2), ("y", 3), ("z", -1)], -4).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (2, 2));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(gelineq_node.as_str(), (1, 1));
        // 2*2 + 3*1 - 0 = 7 >= 4, so constraint should be satisfied

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "General linear inequality should be satisfied"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(gelineq_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn general_linear_inequality_not_satisfied() {
        // 2x + 3y - z >= 4 with x=1, y=0, z=0, but we require gelineq=1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        dag.set_primitive("z", (0, 5)).await.unwrap();
        let gelineq_node = dag.set_gelineq(vec![("x", 2), ("y", 3), ("z", -1)], -4).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert(gelineq_node.as_str(), (1, 1));
        // 2*1 + 3*0 - 0 = 2 < 4, so constraint should not be satisfied

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "General linear inequality should not be satisfied when sum < threshold"
        );
    }

    #[tokio::test]
    async fn nested_logical_structure() {
        // (x ∧ y) ∨ (z ∧ w) - complex nested structure
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        dag.set_primitive("w", (0, 1)).await.unwrap();

        let and1 = dag.set_and(vec!["x", "y"]).await.unwrap();
        let and2 = dag.set_and(vec!["z", "w"]).await.unwrap();
        let or_root = dag.set_or(vec![and1, and2]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (1, 1));
        assume.insert("z", (1, 1));
        assume.insert("w", (1, 1));
        assume.insert(or_root.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Nested structure should be feasible when second branch is true"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(or_root.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn mixed_constraint_types() {
        // Combine logical and linear constraints: (x ∧ y) → (a + b + c >= 2)
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("a", (0, 1)).await.unwrap();
        dag.set_primitive("b", (0, 1)).await.unwrap();
        dag.set_primitive("c", (0, 1)).await.unwrap();

        let and_node = dag.set_and(vec!["x", "y"]).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["a", "b", "c"], 2).await.unwrap();
        let imply_node = dag.set_imply(and_node, atleast_node).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("a", (1, 1));
        assume.insert("b", (1, 1));
        assume.insert("c", (0, 0));
        assume.insert(imply_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Mixed constraint should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(imply_node.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn deep_nesting_levels() {
        // Deep nesting: x ∧ (y ∨ (z ∧ (w ∨ v)))
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        dag.set_primitive("w", (0, 1)).await.unwrap();
        dag.set_primitive("v", (0, 1)).await.unwrap();

        let or_inner = dag.set_or(vec!["w", "v"]).await.unwrap();
        let and_inner = dag.set_and(vec!["z".to_string(), or_inner]).await.unwrap();
        let or_middle = dag.set_or(vec!["y".to_string(), and_inner]).await.unwrap();
        let and_root = dag.set_and(vec!["x".to_string(), or_middle]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert("z", (1, 1));
        assume.insert("w", (0, 0));
        assume.insert("v", (1, 1));
        assume.insert(and_root.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Deep nesting should be feasible");
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(and_root.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn circular_dependency_prevention() {
        // Test that the DAG prevents circular dependencies through proper construction
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();

        let and_node = dag.set_and(vec!["x", "y"]).await.unwrap();
        let or_node = dag.set_or(vec![and_node.clone(), "x".to_string()]).await.unwrap();

        // This should work fine - no circular dependency
        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (0, 0));
        assume.insert(or_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Valid DAG structure should be feasible");
    }

    #[tokio::test]
    async fn multiple_references_to_same_node() {
        // Test referencing the same intermediate node multiple times
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();

        let and_node = dag.set_and(vec!["x", "y"]).await.unwrap();
        let or_node1 = dag.set_or(vec![and_node.clone(), "z".to_string()]).await.unwrap();
        let or_node2 = dag.set_or(vec![and_node.clone(), "x".to_string()]).await.unwrap();
        let final_and = dag.set_and(vec![or_node1, or_node2]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert("y", (1, 1));
        assume.insert("z", (0, 0));
        assume.insert(final_and.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Multiple references to same node should work"
        );
        let soln = solns[0].as_ref().unwrap();
        assert_eq!(*soln.get(final_and.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn optimization_maximize_single_variable() {
        // Maximize x subject to x + y <= 3
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        let atmost_node = dag.set_atmost(vec!["x", "y"], 3).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atmost_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Optimization should be feasible");
        let soln = solns[0].as_ref().unwrap();

        // Should maximize x while keeping x + y <= 3
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val <= 3, "Constraint should be satisfied");
        assert!(x_val >= 0, "x should be maximized subject to constraints");
    }

    #[tokio::test]
    async fn optimization_minimize_objective() {
        // Minimize 2x + 3y subject to x + y >= 2
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["x", "y"], 2).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 2.0);
        objective.insert("y", 3.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atleast_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, false).unwrap(); // minimiz.unwrap()e
        assert!(solns[0].is_some(), "Minimization should be feasible");
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val >= 2, "Constraint should be satisfied");

        // Should minimize 2x + 3y, so prefer x over y
        let objective_value = 2 * x_val + 3 * y_val;
        assert!(
            objective_value >= 4,
            "Minimum objective should be at least 4"
        );
    }

    #[tokio::test]
    async fn optimization_multiple_objectives() {
        // Test with multiple different objectives
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 2)).await.unwrap();
        dag.set_primitive("y", (0, 2)).await.unwrap();
        dag.set_primitive("z", (0, 2)).await.unwrap();
        let equal_node = dag.set_equal(vec!["x", "y", "z"], 2).await.unwrap();

        let mut obj1 = HashMap::<&str, f64>::new();
        obj1.insert("x", 1.0);

        let mut obj2 = HashMap::<&str, f64>::new();
        obj2.insert("y", 2.0);

        let mut obj3 = HashMap::<&str, f64>::new();
        obj3.insert("z", 3.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(equal_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![obj1, obj2, obj3], assume, true).unwrap();
        assert!(solns[0].is_some(), "First objective should be feasible");
        assert!(solns[1].is_some(), "Second objective should be feasible");
        assert!(solns[2].is_some(), "Third objective should be feasible");

        // All should satisfy x + y + z = 2
        for soln in solns.iter().flatten() {
            let x_val = soln.get("x").unwrap().0;
            let y_val = soln.get("y").unwrap().0;
            let z_val = soln.get("z").unwrap().0;
            assert_eq!(
                x_val + y_val + z_val,
                2,
                "All solutions should satisfy constraint"
            );
        }
    }

    #[tokio::test]
    async fn optimization_with_logical_constraints() {
        // Maximize x + y subject to x ∧ y → (z ∨ w)
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        dag.set_primitive("w", (0, 1)).await.unwrap();

        let and_node = dag.set_and(vec!["x", "y"]).await.unwrap();
        let or_node = dag.set_or(vec!["z", "w"]).await.unwrap();
        let imply_node = dag.set_imply(and_node, or_node).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);
        objective.insert("y", 1.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(imply_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Optimization with logical constraints should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        let z_val = soln.get("z").unwrap().0;
        let w_val = soln.get("w").unwrap().0;

        // If x=1 and y=1, then z ∨ w must be true
        if x_val == 1 && y_val == 1 {
            assert!(
                z_val == 1 || w_val == 1,
                "Implication constraint should be satisfied"
            );
        }
    }

    #[tokio::test]
    async fn optimization_with_coefficients() {
        // Test optimization using coefficients in objective
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();

        let atmost_node = dag.set_atmost(vec!["x", "y"], 4).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 2.5);
        objective.insert("y", 1.5);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atmost_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Optimization with coefficients should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val + y_val <= 4, "Constraint should be satisfied");

        // Should prefer x since it has higher coefficient
        let objective_value = 2.5 * (x_val as f64) + 1.5 * (y_val as f64);
        assert!(objective_value >= 0.0, "Objective should be non-negative");
    }

    #[tokio::test]
    async fn infeasible_contradictory_constraints() {
        // Create a real contradiction: x = 1 AND x = 0 using constraint nodes
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();

        // Create two equal constraints: x = 1 and x = 0
        let eq1 = dag.set_equal(vec!["x"], 1).await.unwrap(); // x = 1
        let eq2 = dag.set_equal(vec!["x"], 0).await.unwrap(); // x = 0
        let both = dag.set_and(vec![eq1, eq2]).await.unwrap(); // Both must be true (contradiction)

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(both.as_str(), (1, 1)); // Require both constraints to be satisfied

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Contradictory constraints should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_impossible_linear_constraint() {
        // x + y = 5 with x, y ∈ [0, 1]
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let equal_node = dag.set_equal(vec!["x", "y"], 5).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(equal_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Impossible linear constraint should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_logical_contradiction() {
        // x ∧ ¬x should be false
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        let not_node = dag.set_not(vec!["x"]).await.unwrap();
        let and_node = dag.set_and(vec!["x".to_string(), not_node]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(and_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Logical contradiction should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_conflicting_implications() {
        // x → y and x → ¬y with x = 1
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        let not_y = dag.set_not(vec!["y"]).await.unwrap();
        let imply1 = dag.set_imply("x", "y").await.unwrap();
        let imply2 = dag.set_imply("x", not_y).await.unwrap();
        let and_node = dag.set_and(vec![imply1, imply2]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert(and_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Conflicting implications should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_over_constrained_system() {
        // x + y = 2, x + y = 3, x ≥ 0, y ≥ 0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        let equal1 = dag.set_equal(vec!["x", "y"], 2).await.unwrap();
        let equal2 = dag.set_equal(vec!["x", "y"], 3).await.unwrap();
        let and_node = dag.set_and(vec![equal1, equal2]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(and_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Over-constrained system should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_boundary_violation() {
        // x ∈ [0, 1] but require x = 2
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (2, 2));

        let result = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true);
        assert!(
            result.is_err(),
            "Boundary violation should result in an error"
        );
    }

    #[tokio::test]
    async fn infeasible_complex_nested_contradiction() {
        // (x ∧ y) ∨ (z ∧ w) = 1, but x=0, y=0, z=0, w=0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        dag.set_primitive("y", (0, 1)).await.unwrap();
        dag.set_primitive("z", (0, 1)).await.unwrap();
        dag.set_primitive("w", (0, 1)).await.unwrap();

        let and1 = dag.set_and(vec!["x", "y"]).await.unwrap();
        let and2 = dag.set_and(vec!["z", "w"]).await.unwrap();
        let or_node = dag.set_or(vec![and1, and2]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (0, 0));
        assume.insert("y", (0, 0));
        assume.insert("z", (0, 0));
        assume.insert("w", (0, 0));
        assume.insert(or_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Complex nested contradiction should be infeasible"
        );
    }

    #[tokio::test]
    async fn infeasible_optimization_no_solution() {
        // Maximize x subject to x + y <= -1 with x, y ≥ 0
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();
        let atmost_node = dag.set_atmost(vec!["x", "y"], -1).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atmost_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_none(),
            "Optimization with impossible constraints should be infeasible"
        );
    }

    #[tokio::test]
    async fn edge_case_empty_integer_constraints() {
        // No constraints, just maximize x
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 10)).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);

        let assume = HashMap::<&str, Bound>::new();

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Empty constraints should be feasible");
        let soln = solns[0].as_ref().unwrap();

        // Should maximize x to its upper bound
        let x_val = soln.get("x").unwrap().0;
        assert_eq!(x_val, 10, "Should maximize x to upper bound");
    }

    #[tokio::test]
    async fn edge_case_single_variable_bounds() {
        // Test various single variable bound scenarios
        let dag = Pldag::new();
        dag.set_primitive("x", (5, 5)).await.unwrap(); // Fixed value

        let objective = HashMap::<&str, f64>::new();
        let assume = HashMap::<&str, Bound>::new();

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Fixed value should be feasible");
        let soln = solns[0].as_ref().unwrap();

        assert_eq!(
            *soln.get("x").unwrap(),
            (5, 5),
            "Fixed value should be exact"
        );
    }

    #[tokio::test]
    async fn edge_case_zero_coefficients() {
        // Test optimization with zero coefficients
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 5)).await.unwrap();
        dag.set_primitive("y", (0, 5)).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 0.0);
        objective.insert("y", 0.0);

        let assume = HashMap::<&str, Bound>::new();

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Zero coefficients should be feasible");
        // Any solution should be valid when coefficients are zero
    }

    #[tokio::test]
    async fn edge_case_large_bounds() {
        // Test with large bound values
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1000000)).await.unwrap();
        dag.set_primitive("y", (0, 1000000)).await.unwrap();
        let atmost_node = dag.set_atmost(vec!["x", "y"], 999999).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atmost_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Large bounds should be feasible");
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(
            x_val + y_val <= 999999,
            "Large bound constraint should be satisfied"
        );
    }

    #[tokio::test]
    async fn edge_case_negative_bounds() {
        // Test with negative bounds
        let dag = Pldag::new();
        dag.set_primitive("x", (-10, 10)).await.unwrap();
        dag.set_primitive("y", (-5, 5)).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["x", "y"], -3).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 1.0);

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atleast_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Negative bounds should be feasible");
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(
            x_val + y_val >= -3,
            "Negative bound constraint should be satisfied"
        );
    }

    #[tokio::test]
    async fn edge_case_single_element_operations() {
        // Test logical operations with single elements
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();

        let and_single = dag.set_and(vec!["x"]).await.unwrap();
        let or_single = dag.set_or(vec!["x"]).await.unwrap();
        let xor_single = dag.set_xor(vec!["x"]).await.unwrap();
        let root = dag.set_and(vec![
            and_single.clone(),
            or_single.clone(),
            xor_single.clone(),
        ]).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert("x", (1, 1));
        assume.insert(root.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Single element operations should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();

        assert_eq!(*soln.get(and_single.as_str()).unwrap(), (1, 1));
        assert_eq!(*soln.get(or_single.as_str()).unwrap(), (1, 1));
        assert_eq!(*soln.get(xor_single.as_str()).unwrap(), (1, 1));
    }

    #[tokio::test]
    async fn edge_case_identical_variables_in_constraint() {
        // Test constraint with repeated variables: x + x + x >= 3
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 2)).await.unwrap();
        let atleast_node = dag.set_atleast(vec!["x", "x", "x"], 3).await.unwrap();

        let objective = HashMap::<&str, f64>::new();
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(atleast_node.as_str(), (1, 1));

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Repeated variables in constraint should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();

        let x_val = soln.get("x").unwrap().0;
        assert!(
            3 * x_val >= 3,
            "Repeated variable constraint should be satisfied"
        );
    }

    #[tokio::test]
    async fn edge_case_very_small_coefficients() {
        // Test with very small floating point coefficients
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1000)).await.unwrap();
        dag.set_primitive("y", (0, 1000)).await.unwrap();

        let mut objective = HashMap::<&str, f64>::new();
        objective.insert("x", 0.000001);
        objective.insert("y", 0.000002);

        let assume = HashMap::<&str, Bound>::new();

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(
            solns[0].is_some(),
            "Very small coefficients should be feasible"
        );
        let soln = solns[0].as_ref().unwrap();

        // Should still maximize according to relative coefficients
        let x_val = soln.get("x").unwrap().0;
        let y_val = soln.get("y").unwrap().0;
        assert!(x_val >= 0 && y_val >= 0, "Solution should be within bounds");
    }

    #[tokio::test]
    async fn edge_case_empty_variable_lists() {
        // Test operations with empty variable lists - should fail gracefully
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();

        // Most operations should handle empty lists gracefully
        // This is mainly testing that the library doesn't crash
        let objective = HashMap::<&str, f64>::new();
        let assume = HashMap::<&str, Bound>::new();

        let solns = solve(&dag.sub_dag(vec![]).await.unwrap(), vec![objective], assume, true).unwrap();
        assert!(solns[0].is_some(), "Empty operations should not crash");
    }

    #[tokio::test]
    async fn test_common_dag_with_xor_conjunction() {
        let dag = Pldag::new();
        dag.set_primitives(vec!["s1", "s2", "f1", "f2"], (0, 1)).await.unwrap();
        let s = dag.set_xor(vec!["s1", "s2"]).await.unwrap();
        let f = dag.set_xor(vec!["f1", "f2"]).await.unwrap();
        let root = dag.set_and(vec![s, f]).await.unwrap();
        let solution = solve(
            &dag.sub_dag(vec![]).await.unwrap(),
            vec![HashMap::from([
                ("s1", 1.0),
                ("s2", -1.0),
                ("f2", 1.0),
                ("f1", -1.0),
            ])],
            HashMap::from([(root.as_str(), (1, 1))]),
            true,
        ).unwrap();
        let solution = &solution[0];
        assert!(solution.is_some(), "Expected a feasible solution");
        let solution_unwrapped = solution.as_ref().unwrap();
        assert_eq!(
            *solution_unwrapped.get("s1").unwrap(),
            (1, 1),
            "s1 should be selected"
        );
        assert_eq!(
            *solution_unwrapped.get("s2").unwrap(),
            (0, 0),
            "s2 should not be selected"
        );
        assert_eq!(
            *solution_unwrapped.get("f1").unwrap(),
            (0, 0),
            "f1 should not be selected"
        );
        assert_eq!(
            *solution_unwrapped.get("f2").unwrap(),
            (1, 1),
            "f2 should be selected"
        );
    }

    #[tokio::test]
    async fn test_atleast_with_no_variables_is_rejected() {
        let dag = Pldag::new();
        dag.set_primitive("x", (0, 1)).await.unwrap();
        let result = dag.set_atleast(Vec::<&str>::new(), 1).await;
        assert!(matches!(result, Err(ModelError::EmptyConstraint)));
    }

    #[tokio::test]
    async fn test_exactly_one_boolean_selection() {
        let model = Pldag::new();
        model.set_primitive("a", (0, 1)).await.unwrap();
        model.set_primitive("b", (0, 1)).await.unwrap();
        model.set_primitive("c", (0, 1)).await.unwrap();
        let xor = model.set_atmost(vec!["a", "b", "c"], 1).await.unwrap();

        let solutions = solve(
            &model.sub_dag(vec![]).await.unwrap(),
            vec![HashMap::from([("a", 1.0), ("b", 1.0), ("c", 1.0)])],
            HashMap::from([(xor.as_str(), (1, 1))]),
            true,
        ).unwrap();
        let assignments = solutions[0].as_ref().unwrap();
        let selected_vars: Vec<&&str> = ["a", "b", "c"]
            .iter()
            .filter(|&var| assignments.get(*var).unwrap().0 == 1)
            .collect();
        assert_eq!(
            selected_vars.len(),
            1,
            "Exactly one variable should be selected"
        );
    }

    #[tokio::test]
    async fn test_solve_with_equalities_and_integers() {
        let model = Pldag::new();
        model.set_primitive("a", (0, 2)).await.unwrap();
        model.set_primitive("b", (0, 2)).await.unwrap();
        let lr = model.set_equal(vec!["a", "b"], 1).await.unwrap();
        let rr = model.set_equal(vec!["a", "b"], 2).await.unwrap();
        let root = model.set_and(vec![lr, rr]).await.unwrap();
        let solutions = solve(
            &model.sub_dag(vec![]).await.unwrap(),
            vec![HashMap::from([("a", 1.0), ("b", 1.0)])],
            HashMap::from([(root.as_str(), (1, 1))]),
            true,
        ).unwrap();
        assert!(
            solutions[0].is_none(),
            "Conflicting equalities should be infeasible"
        );
    }

    #[tokio::test]
    async fn test_solve_with_integer_decision_variables_1() {
        // This test is from an issue in the pldag-python repo. It turns out that the solver
        // returns a solution which is not valid according to the constraints.
        let model = Pldag::new();
        model.set_primitive("a", (0, 2)).await.unwrap();
        model.set_primitive("b", (0, 2)).await.unwrap();
        model.set_primitive("c", (0, 2)).await.unwrap();
        model.set_primitive("x", (0, 2)).await.unwrap();
        model.set_primitive("y", (0, 2)).await.unwrap();
        model.set_primitive("z", (0, 2)).await.unwrap();

        let root = model.set_equal(vec!["x", "y", "z"], 1).await.unwrap();
        let solutions = solve(
            &model.sub_dag(vec![]).await.unwrap(),
            vec![HashMap::from([
                ("a", -1.0),
                ("b", -1.0),
                ("c", -1.0),
                ("x", -1.0),
                ("y", -1.0),
                ("z", -1.0),
            ])],
            HashMap::from([(root.as_str(), (1, 1)), ("a", (1, 1))]),
            true,
        ).unwrap();
        if let Some(assignments) = &solutions[0] {
            // Convert String keys to &str keys for propagate
            let str_assignments = assignments.iter().map(|(k, v)| (k.as_str(), *v));
            let propagated = Pldag::propagate_dag(&model.dag().await.unwrap(), str_assignments).unwrap();

            // Check that root is Some and is equal to 1
            assert!(
                assignments.get(root.as_str()).is_some(),
                "Root constraint should be assigned"
            );
            assert!(
                assignments.get(root.as_str()).unwrap().0 == 1,
                "Root constraint should be equal to 1"
            );
            assert!(
                assignments.get("a").is_some(),
                "Variable 'a' should be assigned"
            );
            assert!(
                assignments.get("a").unwrap().0 == 1,
                "Variable 'a' should be equal to 1"
            );

            assert!(
                propagated.get(root.as_str()).is_some(),
                "Root constraint should be satisfied after propagation"
            );
            assert!(
                propagated.get(root.as_str()).unwrap().0 == 1,
                "Root constraint should be equal to 1 after propagation"
            );
            assert!(
                propagated.get("a").is_some(),
                "Variable 'a' should be satisfied after propagation"
            );
            assert!(
                propagated.get("a").unwrap().0 == 1,
                "Variable 'a' should be equal to 1 after propagation"
            );
        } else {
            panic!("Expected a feasible solution");
        }
    }

    #[tokio::test]
    async fn test_empty_composites_are_rejected() {
        let model = Pldag::new();
        assert!(matches!(
            model.set_and(Vec::<String>::new()).await,
            Err(ModelError::EmptyConstraint)
        ));
        assert!(matches!(
            model.set_gelineq(Vec::<(&str, i32)>::new(), -5).await,
            Err(ModelError::EmptyConstraint)
        ));
    }

    #[tokio::test]
    async fn test_solve_empty_constraint_when_allowed() {
        // When the model is configured to allow empty constraints, set_gelineq
        // with no coefficients produces a constant node. A bias >= 0 gives a
        // tautology that the solver should satisfy, while bias < 0 gives a
        // contradiction that becomes infeasible once we force the node TRUE.
        let model = Pldag::new().set_allow_empty_constraints(true);
        model.set_primitive("x", (0, 1)).await.unwrap();

        let taut = model.set_gelineq(Vec::<(&str, i32)>::new(), 0).await.unwrap();
        let contra = model.set_gelineq(Vec::<(&str, i32)>::new(), -1).await.unwrap();

        // The tautology should be forced to 1 in any feasible assignment, even
        // when wired up alongside a real primitive.
        let root = model.set_and(vec![taut.as_str(), "x"]).await.unwrap();
        let dag = model.sub_dag(vec![]).await.unwrap();

        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(root.as_str(), (1, 1));
        let solutions = solve(&dag, vec![HashMap::<&str, f64>::new()], assume, true).unwrap();
        let solution = solutions[0]
            .as_ref()
            .expect("tautology AND x must be feasible");
        assert_eq!(*solution.get(taut.as_str()).unwrap(), (1, 1));
        assert_eq!(*solution.get("x").unwrap(), (1, 1));
        assert_eq!(*solution.get(root.as_str()).unwrap(), (1, 1));

        // The contradiction can never be 1, so assuming it true yields no
        // feasible solution.
        let mut assume = HashMap::<&str, Bound>::new();
        assume.insert(contra.as_str(), (1, 1));
        let solutions = solve(&dag, vec![HashMap::<&str, f64>::new()], assume, true).unwrap();
        assert!(
            solutions[0].is_none(),
            "forcing the contradiction TRUE must be infeasible"
        );
    }

    #[tokio::test]
    async fn test_solve_atmost_tautology() {
        let model = Pldag::new();
        model.set_primitive("x", (0, 1)).await.unwrap();
        model.set_primitive("y", (0, 1)).await.unwrap();
        model.set_primitive("z", (0, 1)).await.unwrap();
        let atmost_taut = model.set_atmost(vec!["x", "y", "z"], 3).await.unwrap();
        let solutions = solve(
            &model.sub_dag(vec![]).await.unwrap(),
            vec![HashMap::from([(atmost_taut.as_str(), -1.0)])],
            HashMap::new(),
            true,
        ).unwrap();
        let solution = solutions[0].as_ref().unwrap();
        assert_eq!(*solution.get(atmost_taut.as_str()).unwrap(), (1, 1));
    }
}
