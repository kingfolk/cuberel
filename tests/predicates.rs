use cuberel::{
    Comparison, DataType, Error, Expr, ExprId, ExprNode, InputKind, PrepareLimits, ScalarOp,
    Schema,
    abi::{BindingError, BindingLayout, BufferDescriptor, PhysicalType},
    col, param, prepare, prepare_with_limits,
};

fn schema() -> Schema {
    Schema::new(
        &[("a", DataType::F32), ("b", DataType::F32)],
        &[("p", DataType::F32)],
    )
    .unwrap()
}

#[test]
fn mixed_arithmetic_and_predicates_infer_types_and_preserve_root() {
    let expr = (col("a") * param("p"))
        .gt(col("b"))
        .and(!col("a").eq_expr(col("b")));
    let program = prepare(&expr, &schema()).unwrap();
    assert_eq!(program.output_type(), DataType::Bool);
    assert!(
        program
            .nodes()
            .iter()
            .any(|n| n.data_type() == DataType::F32)
    );
    assert!(
        program
            .nodes()
            .iter()
            .any(|n| matches!(n.op(), ScalarOp::Not(_)))
    );
    assert!(matches!(
        program.nodes()[program.root().index()].op(),
        ScalarOp::And(..)
    ));
    let program = prepare(
        &Expr::from_nodes(
            vec![
                ExprNode::Column("a".into()),
                ExprNode::Column("b".into()),
                ExprNode::Compare(Comparison::LessThan, ExprId::new(0), ExprId::new(1)),
                ExprNode::Not(ExprId::new(2)),
            ],
            ExprId::new(2),
        ),
        &schema(),
    )
    .unwrap();
    assert_eq!(program.root(), ExprId::new(2));
    assert_eq!(program.output_type(), DataType::Bool);
}

#[test]
fn predicate_types_are_checked_even_in_unreachable_nodes() {
    let predicate = col("a").gt(col("b"));
    for expr in [
        predicate.clone() + col("a"),
        col("a") * predicate.clone(),
        predicate.clone().lt(col("a")),
        predicate.clone().eq_expr(predicate.clone()),
        col("a").and(predicate.clone()),
        predicate.or(col("a")),
        !col("a"),
    ] {
        assert!(matches!(
            prepare(&expr, &schema()),
            Err(Error::OperandTypeMismatch { .. })
        ));
    }
    let expr = Expr::from_nodes(
        vec![ExprNode::Column("a".into()), ExprNode::Not(ExprId::new(0))],
        ExprId::new(0),
    );
    assert_eq!(
        prepare(&expr, &schema()).unwrap_err(),
        Error::OperandTypeMismatch {
            node: ExprId::new(1),
            operand: ExprId::new(0),
            expected: DataType::Bool,
            actual: DataType::F32,
        }
    );
    // A bool schema declaration is descriptive; bool host inputs are a later feature.
    let bool_schema = Schema::new(&[("a", DataType::Bool)], &[]).unwrap();
    assert!(matches!(
        prepare(&col("a"), &bool_schema),
        Err(Error::UnsupportedType {
            kind: InputKind::Column,
            data_type: DataType::Bool,
            ..
        })
    ));
}

#[test]
fn unary_and_binary_predicates_validate_graphs_and_limits() {
    for nodes in [
        vec![ExprNode::Not(ExprId::new(0))],
        vec![
            ExprNode::Column("a".into()),
            ExprNode::Compare(Comparison::Equal, ExprId::new(0), ExprId::new(2)),
        ],
    ] {
        assert!(matches!(
            prepare(&Expr::from_nodes(nodes, ExprId::new(0)), &schema()),
            Err(Error::InvalidReference { .. })
        ));
    }
    let bad = Expr::from_nodes(vec![ExprNode::Column("a".into())], ExprId::new(7));
    for expr in [!bad.clone(), bad.clone().lt(col("a")), col("a").gt(bad)] {
        assert!(matches!(
            prepare(&expr, &schema()),
            Err(Error::InvalidRoot { .. })
        ));
    }
    let expr = !col("a").gt(param("p"));
    assert!(matches!(
        prepare_with_limits(
            &expr,
            &schema(),
            PrepareLimits {
                max_nodes: 10,
                max_depth: 2
            }
        ),
        Err(Error::ExpressionTooDeep { depth: 3, .. })
    ));
    assert!(matches!(
        prepare_with_limits(
            &expr,
            &schema(),
            PrepareLimits {
                max_nodes: 3,
                max_depth: 10
            }
        ),
        Err(Error::ExpressionTooLarge { .. })
    ));
}

#[test]
fn mask_layout_requires_u32_output_and_validates_it_as_such() {
    let schema = schema();
    let mask = BindingLayout::new(&prepare(&col("a").lt(param("p")), &schema).unwrap());
    let arithmetic = BindingLayout::new(&prepare(&(col("a") + param("p")), &schema).unwrap());
    assert_ne!(mask, arithmetic);
    assert_eq!(mask.buffers()[0].element_type(), PhysicalType::F32);
    assert_eq!(mask.buffers()[1].element_type(), PhysicalType::U32);
    let mut buffers = vec![
        BufferDescriptor {
            element_type: PhysicalType::F32,
            logical_len: 3,
            allocation_bytes: 12,
            byte_offset: 0,
            end_padding_bytes: 0,
        },
        BufferDescriptor {
            element_type: PhysicalType::U32,
            logical_len: 3,
            allocation_bytes: 12,
            byte_offset: 0,
            end_padding_bytes: 0,
        },
    ];
    assert_eq!(mask.validate_buffers(3, &buffers).unwrap().byte_len(), 12);
    buffers[1].element_type = PhysicalType::F32;
    assert!(matches!(
        mask.validate_buffers(3, &buffers),
        Err(BindingError::BufferType {
            index: 1,
            expected: PhysicalType::U32,
            ..
        })
    ));
    buffers[1].element_type = PhysicalType::U32;
    buffers[1].allocation_bytes = 8;
    assert!(matches!(
        mask.validate_buffers(3, &buffers),
        Err(BindingError::BufferTooSmall { index: 1, .. })
    ));
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod runtime {
    use cubecl::Device;
    use cuberel::{
        Batch, ColumnData, Comparison, DataType, Error, ExecutionError, Executor, Expr, ExprId,
        ExprNode, Parameters, Schema, abi::BindingError, col, param, prepare,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn batch<'a>(a: &'a [f32], b: &'a [f32]) -> Batch<'a> {
        Batch::new(a.len())
            .with_column("a", ColumnData::F32(a))
            .unwrap()
            .with_column("b", ColumnData::F32(b))
            .unwrap()
    }

    fn exercise(device: Device) {
        let executor = Executor::new(device.client()).unwrap();
        let schema = super::schema();
        let input = batch(
            &[
                f32::NEG_INFINITY,
                -2.0,
                -0.0,
                0.0,
                2.0,
                f32::INFINITY,
                f32::NAN,
                1.0,
            ],
            &[
                f32::NEG_INFINITY,
                -1.0,
                0.0,
                -0.0,
                1.0,
                f32::INFINITY,
                0.0,
                f32::NAN,
            ],
        );
        let comparisons = [
            (col("a").eq_expr(col("b")), [1, 0, 1, 1, 0, 1, 0, 0]),
            (col("a").ne_expr(col("b")), [0, 1, 0, 0, 1, 0, 1, 1]),
            (col("a").lt(col("b")), [0, 1, 0, 0, 0, 0, 0, 0]),
            (col("a").le(col("b")), [1, 1, 1, 1, 0, 1, 0, 0]),
            (col("a").gt(col("b")), [0, 0, 0, 0, 1, 0, 0, 0]),
            (col("a").ge(col("b")), [1, 0, 1, 1, 1, 1, 0, 0]),
        ];
        for (expr, expected) in comparisons {
            let program = prepare(&expr, &schema).unwrap();
            assert_eq!(
                executor
                    .execute_mask(&program, &input, &Parameters::new())
                    .unwrap(),
                expected
            );
        }
        // A shared operand must not cause x == x to fold true for NaN.
        let shared = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column("a".into()),
                    ExprNode::Compare(Comparison::Equal, ExprId::new(0), ExprId::new(0)),
                ],
                ExprId::new(1),
            ),
            &schema,
        )
        .unwrap();
        assert_eq!(
            executor
                .execute_mask(&shared, &input, &Parameters::new())
                .unwrap(),
            [1, 1, 1, 1, 1, 1, 0, 1]
        );
        // All four two-valued truth-table rows, and not of a compound expression.
        let input = batch(&[1.0, 1.0, -1.0, -1.0], &[1.0, -1.0, 1.0, -1.0]);
        let x = col("a").gt(param("p"));
        let y = col("b").gt(param("p"));
        let parameters = Parameters::new().with("p", 0.0_f32).unwrap();
        for (expr, expected) in [
            (x.clone().and(y.clone()), [1, 0, 0, 0]),
            (x.clone().or(y.clone()), [1, 1, 1, 0]),
            (!x.clone(), [0, 0, 1, 1]),
            (!(x.clone().and(y.clone())), [0, 1, 1, 1]),
        ] {
            let program = prepare(&expr, &schema).unwrap();
            assert_eq!(
                executor
                    .execute_mask(&program, &input, &parameters)
                    .unwrap(),
                expected
            );
        }
        // Predicate operands are evaluated eagerly; there are no effects/NULLs.
        let nan_params = Parameters::new().with("p", f32::NAN).unwrap();
        for (expr, expected) in [
            (col("a").eq_expr(param("p")), [0; 4]),
            (col("a").ne_expr(param("p")), [1; 4]),
        ] {
            assert_eq!(
                executor
                    .execute_mask(&prepare(&expr, &schema).unwrap(), &input, &nan_params)
                    .unwrap(),
                expected
            );
        }

        // Tail guards, all/none matches, arithmetic feeding comparisons, and rebinding.
        let predicate = prepare(&(col("a") * param("p")).gt(col("b")), &schema).unwrap();
        let size = executor.cube_size() as usize;
        for rows in [0, 1, size - 1, size, size + 1, size * 3 + 7, 4099] {
            let a: Vec<f32> = (0..rows).map(|i| [1.0, 2.0, 3.0, 4.0][i % 4]).collect();
            let b = vec![4.0; rows];
            for (p, pattern) in [
                (0.0_f32, [0, 0, 0, 0]),
                (2.0, [0, 0, 1, 1]),
                (8.0, [1, 1, 1, 1]),
            ] {
                let expected: Vec<u32> = (0..rows).map(|i| pattern[i % 4]).collect();
                assert_eq!(
                    executor
                        .execute_mask(
                            &predicate,
                            &batch(&a, &b),
                            &Parameters::new().with("p", p).unwrap()
                        )
                        .unwrap(),
                    expected
                );
            }
        }
        // Parameter-only expressions broadcast without requiring any input buffers.
        let broadcast = prepare(&param("p").ge(param("p")), &schema).unwrap();
        assert_eq!(
            executor
                .execute_mask(&broadcast, &Batch::new(3), &parameters)
                .unwrap(),
            [1; 3]
        );
        assert_eq!(
            executor
                .execute_mask(&broadcast, &Batch::new(3), &nan_params)
                .unwrap(),
            [0; 3]
        );

        // Wrong execution API and invalid bindings fail before definition, even for empty input.
        let count = executor.define_count();
        assert!(matches!(
            executor.execute(&predicate, &Batch::new(0), &parameters),
            Err(ExecutionError::OutputType {
                expected: DataType::F32,
                actual: DataType::Bool
            })
        ));
        let arithmetic = prepare(&col("a"), &schema).unwrap();
        assert!(matches!(
            executor.execute_mask(&arithmetic, &Batch::new(0), &parameters),
            Err(ExecutionError::OutputType {
                expected: DataType::Bool,
                actual: DataType::F32
            })
        ));
        assert!(matches!(
            executor.execute_mask(&predicate, &batch(&[], &[]), &Parameters::new()),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::MissingInput { .. }
            )))
        ));
        assert!(matches!(
            executor.execute_mask(&predicate, &Batch::new(0), &parameters),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::MissingInput { .. }
            )))
        ));
        let wrong = Parameters::new().with("p", 1_i32).unwrap();
        assert!(matches!(
            executor.execute_mask(&predicate, &batch(&[], &[]), &wrong),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::TypeMismatch { .. }
            )))
        ));
        assert_eq!(executor.define_count(), count);

        // A unique cold key isolates cache assertions from persisted user caches.
        let name = format!(
            "predicate_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let schema = Schema::new(
            &[(&name, DataType::F32)],
            &[
                ("scale", DataType::F32),
                ("limit", DataType::F32),
                ("upper", DataType::F32),
            ],
        )
        .unwrap();
        let expr = (col(&name) * param("scale"))
            .gt(param("limit"))
            .and(!col(&name).ge(param("upper")));
        let program = prepare(&expr, &schema).unwrap();
        let params = |scale, limit, upper| {
            Parameters::new()
                .with("scale", scale)
                .unwrap()
                .with("limit", limit)
                .unwrap()
                .with("upper", upper)
                .unwrap()
        };
        let data = [1.0, 3.0, 5.0];
        let input = Batch::new(3)
            .with_column(&name, ColumnData::F32(&data))
            .unwrap();
        let before = executor.define_count();
        assert_eq!(
            executor
                .execute_mask(&program, &input, &params(2.0_f32, 4.0_f32, 5.0_f32))
                .unwrap(),
            [0, 1, 0]
        );
        assert_eq!(executor.define_count(), before + 1);
        let data = [2.0, 4.0];
        let input = Batch::new(2)
            .with_column(&name, ColumnData::F32(&data))
            .unwrap();
        let reparsed = prepare(&expr, &schema).unwrap();
        assert_eq!(executor.kernel_id(&program), executor.kernel_id(&reparsed));
        assert_eq!(
            executor
                .execute_mask(&reparsed, &input, &params(3.0_f32, 3.0_f32, 9.0_f32))
                .unwrap(),
            [1, 1]
        );
        assert_eq!(executor.define_count(), before + 1);
        let mut ids = Vec::new();
        for expr in [
            col(&name).lt(param("limit")),
            col(&name).le(param("limit")),
            col(&name).gt(param("limit")),
            col(&name).ge(param("limit")),
            col(&name).eq_expr(param("limit")),
            col(&name).ne_expr(param("limit")),
        ] {
            let id = executor.kernel_id(&prepare(&expr, &schema).unwrap());
            assert!(!ids.contains(&id));
            ids.push(id);
        }
        assert_ne!(
            executor.kernel_id(
                &prepare(
                    &col(&name)
                        .lt(param("limit"))
                        .and(col(&name).gt(param("scale"))),
                    &schema
                )
                .unwrap()
            ),
            executor.kernel_id(
                &prepare(
                    &col(&name)
                        .lt(param("limit"))
                        .or(col(&name).gt(param("scale"))),
                    &schema
                )
                .unwrap()
            )
        );
        assert_ne!(
            executor.kernel_id(&program),
            executor.kernel_id(&prepare(&!expr, &schema).unwrap())
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_predicate_conformance() {
        exercise(Device::cpu().expect("real CPU required"));
    }

    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_predicate_conformance() {
        exercise(Device::metal(Default::default()).expect("real native Metal required"));
    }
}
