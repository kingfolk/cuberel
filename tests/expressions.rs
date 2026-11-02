use std::thread;

use cuberel::{
    Batch, ColumnData, DataType, Error, Expr, ExprId, ExprNode, InputKind, KernelProgram,
    Parameters, PrepareLimits, ScalarOp, Schema, col, param, prepare, prepare_with_limits,
};

fn schema() -> Schema {
    Schema::new(
        &[("a", DataType::F32), ("b", DataType::F32)],
        &[("scale", DataType::F32), ("bias", DataType::F32)],
    )
    .unwrap()
}

#[test]
fn composite_expression_rebinds_two_same_type_parameters() {
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema(),
    )
    .unwrap();
    let a = [1.0, 2.0, -3.0];
    let b = [10.0, 20.0, 30.0];
    // Deliberately bind columns and parameters in the opposite order to the schema.
    let batch = Batch::new(3)
        .with_column("b", ColumnData::F32(&b))
        .unwrap()
        .with_column("a", ColumnData::F32(&a))
        .unwrap();
    let first = Parameters::new()
        .with("bias", 1.0_f32)
        .unwrap()
        .with("scale", 2.0_f32)
        .unwrap();
    let second = Parameters::new()
        .with("scale", -1.0_f32)
        .unwrap()
        .with("bias", 0.0_f32)
        .unwrap();
    assert_eq!(
        program.evaluate_reference(&batch, &first).unwrap(),
        [13.0, 25.0, 25.0]
    );
    assert_eq!(
        program.evaluate_reference(&batch, &second).unwrap(),
        [9.0, 18.0, 33.0]
    );
    assert_eq!(program.output_type(), DataType::F32);
    assert!(
        program
            .nodes()
            .iter()
            .all(|node| node.data_type() == DataType::F32)
    );
}

#[test]
fn prepared_program_owns_its_data_and_can_cross_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KernelProgram>();
    let program = {
        let schema = schema();
        let expr = col("a") * param("scale");
        prepare(&expr, &schema).unwrap()
    };
    let result = thread::spawn(move || {
        let a = [3.0, -4.0];
        let batch = Batch::new(2).with_column("a", ColumnData::F32(&a)).unwrap();
        let parameters = Parameters::new().with("scale", 2.0_f32).unwrap();
        program
            .clone()
            .evaluate_reference(&batch, &parameters)
            .unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(result, [6.0, -8.0]);
}

#[test]
fn preserves_parentheses_and_float_evaluation_order() {
    let schema = schema();
    let left = prepare(&((col("a") + param("bias")) + col("b")), &schema).unwrap();
    let right = prepare(&(col("a") + (param("bias") + col("b"))), &schema).unwrap();
    let a = [16_777_216.0_f32];
    let b = [-16_777_216.0_f32];
    let batch = Batch::new(1)
        .with_column("a", ColumnData::F32(&a))
        .unwrap()
        .with_column("b", ColumnData::F32(&b))
        .unwrap();
    let parameters = Parameters::new().with("bias", 1.0_f32).unwrap();
    assert_eq!(left.evaluate_reference(&batch, &parameters).unwrap(), [0.0]);
    assert_eq!(
        right.evaluate_reference(&batch, &parameters).unwrap(),
        [1.0]
    );
}

#[test]
fn named_operands_have_separate_namespaces_and_stable_schema_ids() {
    let schema = Schema::new(
        &[("b", DataType::F32), ("a", DataType::F32)],
        &[("a", DataType::F32)],
    )
    .unwrap();
    let program = prepare(&(col("a") + param("a")), &schema).unwrap();
    let column_id = schema.column_id("a").unwrap();
    let parameter_id = schema.parameter_id("a").unwrap();
    assert_eq!(column_id.index(), 1);
    assert_eq!(parameter_id.index(), 0);
    assert_eq!(program.nodes()[0].op(), ScalarOp::Column(column_id));
    assert_eq!(program.nodes()[1].op(), ScalarOp::Parameter(parameter_id));
    let a = [3.0];
    let batch = Batch::new(1).with_column("a", ColumnData::F32(&a)).unwrap();
    let parameters = Parameters::new().with("a", 7.0_f32).unwrap();
    assert_eq!(
        program.evaluate_reference(&batch, &parameters).unwrap(),
        [10.0]
    );
}

#[test]
fn raw_dag_can_share_an_intermediate() {
    let expr = Expr::from_nodes(
        vec![
            ExprNode::Column("a".into()),
            ExprNode::Parameter("scale".into()),
            ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
            ExprNode::Add(ExprId::new(2), ExprId::new(2)),
        ],
        ExprId::new(3),
    );
    let program = prepare(&expr, &schema()).unwrap();
    let a = [2.0, -3.0];
    let batch = Batch::new(2).with_column("a", ColumnData::F32(&a)).unwrap();
    let parameters = Parameters::new().with("scale", 4.0_f32).unwrap();
    assert_eq!(
        program.evaluate_reference(&batch, &parameters).unwrap(),
        [16.0, -24.0]
    );
}

#[test]
fn evaluates_parameter_only_program_using_explicit_row_count() {
    let program = prepare(&(param("scale") * param("scale")), &schema()).unwrap();
    let parameters = Parameters::new().with("scale", 3.0_f32).unwrap();
    assert_eq!(
        program
            .evaluate_reference(&Batch::new(4), &parameters)
            .unwrap(),
        [9.0; 4]
    );
}

#[test]
fn empty_batch_still_validates_inputs() {
    let program = prepare(&(col("a") + param("bias")), &schema()).unwrap();
    let batch = Batch::new(0)
        .with_column("a", ColumnData::F32(&[]))
        .unwrap();
    let parameters = Parameters::new().with("bias", 1.0_f32).unwrap();
    assert_eq!(
        program.evaluate_reference(&batch, &parameters).unwrap(),
        Vec::<f32>::new()
    );
    assert_eq!(
        program
            .evaluate_reference(&batch, &Parameters::new())
            .unwrap_err(),
        Error::MissingInput {
            kind: InputKind::Parameter,
            name: "bias".into()
        }
    );
}

#[test]
fn missing_referenced_input_is_an_error_but_unused_fields_are_not_required() {
    let program = prepare(&col("a"), &schema()).unwrap();
    assert_eq!(
        program
            .evaluate_reference(&Batch::new(1), &Parameters::new())
            .unwrap_err(),
        Error::MissingInput {
            kind: InputKind::Column,
            name: "a".into()
        }
    );
    let a = [4.0];
    let batch = Batch::new(1)
        .with_column("extra", ColumnData::I32(&[7]))
        .unwrap()
        .with_column("a", ColumnData::F32(&a))
        .unwrap();
    let parameters = Parameters::new().with("extra", 8_i32).unwrap();
    assert_eq!(
        program.evaluate_reference(&batch, &parameters).unwrap(),
        [4.0]
    );
}

#[test]
fn mismatched_column_and_parameter_types_are_rejected() {
    let column_program = prepare(&col("a"), &schema()).unwrap();
    let batch = Batch::new(1)
        .with_column("a", ColumnData::I32(&[7]))
        .unwrap();
    assert_eq!(
        column_program
            .evaluate_reference(&batch, &Parameters::new())
            .unwrap_err(),
        Error::TypeMismatch {
            kind: InputKind::Column,
            name: "a".into(),
            expected: DataType::F32,
            actual: DataType::I32
        }
    );
    let parameter_program = prepare(&param("scale"), &schema()).unwrap();
    let parameters = Parameters::new().with("scale", 2_i32).unwrap();
    assert_eq!(
        parameter_program
            .evaluate_reference(&Batch::new(1), &parameters)
            .unwrap_err(),
        Error::TypeMismatch {
            kind: InputKind::Parameter,
            name: "scale".into(),
            expected: DataType::F32,
            actual: DataType::I32
        }
    );
}

#[test]
fn mismatched_column_lengths_are_rejected_at_batch_construction() {
    assert_eq!(
        Batch::new(2)
            .with_column("a", ColumnData::F32(&[1.0]))
            .unwrap_err(),
        Error::ColumnLengthMismatch {
            name: "a".into(),
            expected: 2,
            actual: 1
        }
    );
}

#[test]
fn duplicate_and_empty_names_are_rejected() {
    assert_eq!(
        Schema::new(&[("a", DataType::F32), ("a", DataType::F32)], &[]).unwrap_err(),
        Error::DuplicateName {
            kind: InputKind::Column,
            name: "a".into()
        }
    );
    assert_eq!(
        Schema::new(&[], &[("p", DataType::F32), ("p", DataType::F32)]).unwrap_err(),
        Error::DuplicateName {
            kind: InputKind::Parameter,
            name: "p".into()
        }
    );
    assert_eq!(
        Schema::new(&[("", DataType::F32)], &[]).unwrap_err(),
        Error::EmptyName {
            kind: InputKind::Column
        }
    );
    assert_eq!(
        Parameters::new().with("", 1.0_f32).unwrap_err(),
        Error::EmptyName {
            kind: InputKind::Parameter
        }
    );
    assert_eq!(
        Parameters::new()
            .with("p", 1.0_f32)
            .unwrap()
            .with("p", 2.0_f32)
            .unwrap_err(),
        Error::DuplicateName {
            kind: InputKind::Parameter,
            name: "p".into()
        }
    );
    assert_eq!(
        Batch::new(0)
            .with_column("a", ColumnData::F32(&[]))
            .unwrap()
            .with_column("a", ColumnData::F32(&[]))
            .unwrap_err(),
        Error::DuplicateName {
            kind: InputKind::Column,
            name: "a".into()
        }
    );
}

#[test]
fn prepare_rejects_unknown_columns_and_parameters() {
    for (expr, kind) in [
        (col("missing"), InputKind::Column),
        (param("missing"), InputKind::Parameter),
    ] {
        assert_eq!(
            prepare(&expr, &schema()).unwrap_err(),
            Error::UnknownName {
                kind,
                name: "missing".into()
            }
        );
    }
}

#[test]
fn prepare_rejects_unsupported_referenced_types() {
    let schema = Schema::new(&[("a", DataType::I32)], &[("p", DataType::I32)]).unwrap();
    for (expr, kind, name) in [
        (col("a"), InputKind::Column, "a"),
        (param("p"), InputKind::Parameter, "p"),
    ] {
        assert_eq!(
            prepare(&expr, &schema).unwrap_err(),
            Error::UnsupportedType {
                kind,
                name: name.into(),
                data_type: DataType::I32
            }
        );
    }
}

#[test]
fn prepare_rejects_empty_arenas_and_invalid_roots() {
    assert_eq!(
        prepare(&Expr::from_nodes(vec![], ExprId::new(0)), &schema()).unwrap_err(),
        Error::EmptyExpression
    );
    let expr = Expr::from_nodes(vec![ExprNode::Column("a".into())], ExprId::new(1));
    assert_eq!(
        prepare(&expr, &schema()).unwrap_err(),
        Error::InvalidRoot {
            root: ExprId::new(1),
            node_count: 1
        }
    );
}

#[test]
fn prepare_rejects_forward_self_and_out_of_range_references() {
    for bad in [1, 2, usize::MAX] {
        let expr = Expr::from_nodes(
            vec![
                ExprNode::Column("a".into()),
                ExprNode::Add(ExprId::new(0), ExprId::new(bad)),
            ],
            ExprId::new(1),
        );
        assert_eq!(
            prepare(&expr, &schema()).unwrap_err(),
            Error::InvalidReference {
                node: ExprId::new(1),
                operand: ExprId::new(bad)
            }
        );
    }
    let cycle = Expr::from_nodes(
        vec![
            ExprNode::Add(ExprId::new(1), ExprId::new(1)),
            ExprNode::Add(ExprId::new(0), ExprId::new(0)),
        ],
        ExprId::new(1),
    );
    assert!(matches!(
        prepare(&cycle, &schema()),
        Err(Error::InvalidReference { .. })
    ));
}

#[test]
fn composing_invalid_raw_expressions_cannot_heal_their_roots() {
    let bad = Expr::from_nodes(vec![ExprNode::Column("a".into())], ExprId::new(1));
    assert!(matches!(
        prepare(&(bad.clone() + param("bias")), &schema()),
        Err(Error::InvalidRoot { .. })
    ));
    assert!(matches!(
        prepare(&(col("a") * bad), &schema()),
        Err(Error::InvalidRoot { .. })
    ));
    let bad_operand = Expr::from_nodes(
        vec![ExprNode::Mul(ExprId::new(usize::MAX), ExprId::new(0))],
        ExprId::new(0),
    );
    assert!(matches!(
        prepare(&(col("a") + bad_operand), &schema()),
        Err(Error::InvalidReference { .. })
    ));
}

#[test]
fn prepare_checks_unreachable_raw_nodes_too() {
    let expr = Expr::from_nodes(
        vec![
            ExprNode::Column("a".into()),
            ExprNode::Parameter("missing".into()),
        ],
        ExprId::new(0),
    );
    assert!(matches!(
        prepare(&expr, &schema()),
        Err(Error::UnknownName { .. })
    ));
}

#[test]
fn expression_limits_have_inclusive_boundaries() {
    let expr = (col("a") + param("bias")) * col("b");
    let schema = schema();
    assert!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 5,
                max_depth: 3
            }
        )
        .is_ok()
    );
    assert_eq!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 4,
                max_depth: 3
            }
        )
        .unwrap_err(),
        Error::ExpressionTooLarge {
            node_count: 5,
            limit: 4
        }
    );
    assert_eq!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 5,
                max_depth: 2
            }
        )
        .unwrap_err(),
        Error::ExpressionTooDeep {
            node: ExprId::new(4),
            depth: 3,
            limit: 2
        }
    );
}

#[test]
fn deeply_nested_raw_input_is_rejected_without_recursive_traversal_or_drop() {
    let mut nodes = vec![ExprNode::Column("a".into())];
    for index in 1..10_000 {
        nodes.push(ExprNode::Add(ExprId::new(index - 1), ExprId::new(0)));
    }
    let expr = Expr::from_nodes(nodes, ExprId::new(9999));
    assert_eq!(
        prepare_with_limits(
            &expr,
            &schema(),
            PrepareLimits {
                max_nodes: 10_000,
                max_depth: 64
            }
        )
        .unwrap_err(),
        Error::ExpressionTooDeep {
            node: ExprId::new(64),
            depth: 65,
            limit: 64
        }
    );
}

#[test]
fn multiple_rows_match_an_independent_arithmetic_formula() {
    let a: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) / 8.0).collect();
    let b: Vec<f32> = (0..257).map(|i| (i % 13) as f32 / 4.0).collect();
    let program = prepare(
        &((col("a") * param("scale") + col("b")) * (param("scale") + param("bias"))),
        &schema(),
    )
    .unwrap();
    let batch = Batch::new(a.len())
        .with_column("a", ColumnData::F32(&a))
        .unwrap()
        .with_column("b", ColumnData::F32(&b))
        .unwrap();
    for (scale, bias) in [(1.5_f32, -0.25_f32), (-2.0, 0.5), (0.0, 1.0)] {
        let parameters = Parameters::new()
            .with("scale", scale)
            .unwrap()
            .with("bias", bias)
            .unwrap();
        let expected: Vec<f32> = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a * scale + b) * (scale + bias))
            .collect();
        assert_eq!(
            program.evaluate_reference(&batch, &parameters).unwrap(),
            expected
        );
    }
}
