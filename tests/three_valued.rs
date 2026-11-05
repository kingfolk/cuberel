use cuberel::{
    DataType, Error, Expr, ExprId, ExprNode, Field, PrepareLimits, Schema,
    abi::{
        BINDING_ABI_VERSION, BindingError, BindingLayout, BufferDescriptor, BufferSource,
        PhysicalType,
    },
    col, param, prepare, prepare_with_limits,
};

fn schema() -> Schema {
    Schema::from_fields(
        &[
            Field::new("a", DataType::F32).with_nullable(true),
            Field::new("b", DataType::F32).with_nullable(true),
            Field::new("plain", DataType::F32),
        ],
        &[
            Field::new("p", DataType::F32).with_nullable(true),
            Field::new("zero", DataType::F32),
        ],
    )
    .unwrap()
}

#[test]
fn predicates_propagate_nullability_and_is_null_is_always_known() {
    let schema = schema();
    let x = col("a").gt(param("zero"));
    for expr in [
        x.clone(),
        !x.clone(),
        x.clone().and(col("plain").lt(param("zero"))),
        x.clone().or(col("b").gt(param("zero"))),
    ] {
        let program = prepare(&expr, &schema).unwrap();
        assert_eq!(program.output_type(), DataType::Bool);
        assert!(program.output_nullable());
    }
    for expr in [
        col("a").is_null(),
        x.clone().is_null(),
        (!x).is_null(),
        (col("a") * param("p")).is_null(),
        col("a").is_null().is_null(),
        col("plain").is_null(),
    ] {
        let program = prepare(&expr, &schema).unwrap();
        assert_eq!(program.output_type(), DataType::Bool);
        assert!(!program.output_nullable());
    }
    assert!(matches!(
        prepare(&(col("a").is_null() + col("plain")), &schema),
        Err(Error::OperandTypeMismatch {
            expected: DataType::F32,
            actual: DataType::Bool,
            ..
        })
    ));
}

#[test]
fn is_null_composition_checks_raw_roots_references_and_depth() {
    let schema = schema();
    let bad = Expr::from_nodes(vec![ExprNode::Column("a".into())], ExprId::new(1));
    assert!(matches!(
        prepare(&bad.is_null(), &schema),
        Err(Error::InvalidRoot { .. })
    ));
    for nodes in [
        vec![ExprNode::IsNull(ExprId::new(0))],
        vec![
            ExprNode::Column("a".into()),
            ExprNode::IsNull(ExprId::new(2)),
        ],
    ] {
        assert!(matches!(
            prepare(&Expr::from_nodes(nodes, ExprId::new(0)), &schema),
            Err(Error::InvalidReference { .. })
        ));
    }
    let expr = !col("a").is_null().is_null();
    assert!(matches!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 4,
                max_depth: 3
            }
        ),
        Err(Error::ExpressionTooDeep { depth: 4, .. })
    ));
    assert!(matches!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 3,
                max_depth: 4
            }
        ),
        Err(Error::ExpressionTooLarge { .. })
    ));
    assert!(
        prepare_with_limits(
            &expr,
            &schema,
            PrepareLimits {
                max_nodes: 4,
                max_depth: 4
            }
        )
        .is_ok()
    );
    // Right-hand unary nodes must be rebased when arenas are combined.
    assert!(
        !prepare(
            &col("a").is_null().and((col("b") + param("p")).is_null()),
            &schema
        )
        .unwrap()
        .output_nullable()
    );
}

#[test]
fn nullable_bool_uses_existing_u32_value_validity_abi() {
    let schema = schema();
    let program = prepare(&col("a").eq_expr(param("p")), &schema).unwrap();
    let layout = BindingLayout::new(&program);
    assert_eq!(layout.version(), BINDING_ABI_VERSION);
    assert_eq!(
        layout
            .buffers()
            .iter()
            .map(|s| s.source())
            .collect::<Vec<_>>(),
        [
            BufferSource::Column(schema.column_id("a").unwrap()),
            BufferSource::ColumnValidity(schema.column_id("a").unwrap()),
            BufferSource::Output,
            BufferSource::OutputValidity
        ]
    );
    assert_eq!(layout.buffers()[2].element_type(), PhysicalType::U32);
    assert_eq!(layout.buffers()[3].element_type(), PhysicalType::U32);
    let mut descriptors: Vec<_> = layout
        .buffers()
        .iter()
        .map(|s| BufferDescriptor {
            element_type: s.element_type(),
            logical_len: 2,
            allocation_bytes: 8,
            byte_offset: 0,
            end_padding_bytes: 0,
        })
        .collect();
    assert!(layout.validate_buffers(2, &descriptors).is_ok());
    descriptors[2].element_type = PhysicalType::F32;
    assert!(matches!(
        layout.validate_buffers(2, &descriptors),
        Err(BindingError::BufferType {
            index: 2,
            expected: PhysicalType::U32,
            ..
        })
    ));
    let is_null = BindingLayout::new(&prepare(&col("a").is_null(), &schema).unwrap());
    assert_eq!(is_null.buffers().len(), 3);
    assert_eq!(is_null.buffers()[2].source(), BufferSource::Output);
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod runtime {
    use cubecl::Device;
    use cuberel::{
        Batch, ColumnData, Comparison, DataType, Error, ExecutionError, Executor, Expr, ExprId,
        ExprNode, Field, InputKind, NullableBool, Parameters, Schema, abi::BindingError, col,
        param, prepare,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn input<'a>(a: &'a [f32], av: &'a [u32], b: &'a [f32], bv: &'a [u32]) -> Batch<'a> {
        Batch::new(a.len())
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: a,
                    validity: av,
                },
            )
            .unwrap()
            .with_column(
                "b",
                ColumnData::NullableF32 {
                    values: b,
                    validity: bv,
                },
            )
            .unwrap()
    }
    fn assert_predicate(output: &NullableBool, expected: &[Option<bool>]) {
        assert_eq!(output.len(), expected.len());
        assert_eq!(output.is_empty(), expected.is_empty());
        assert_eq!(output.iter().collect::<Vec<_>>(), expected);
        for ((&value, &valid), expected) in
            output.values().iter().zip(output.validity()).zip(expected)
        {
            assert_eq!(value, u32::from(*expected == Some(true)));
            assert_eq!(valid, u32::from(expected.is_some()));
        }
    }
    fn exercise(device: Device) {
        let executor = Executor::new(device.client()).unwrap();
        let schema = super::schema();
        let parameters = Parameters::new().with("zero", 0.0_f32).unwrap();
        let x = col("a").gt(param("zero"));
        let y = col("b").gt(param("zero"));
        let t = Some(true);
        let f = Some(false);
        let u = None;
        // Cartesian product of true, false, unknown in row-major order.
        let batch = input(
            &[1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 99.0, 99.0, 99.0],
            &[1, 1, 1, 1, 1, 1, 0, 0, 0],
            &[1.0, -1.0, 99.0, 1.0, -1.0, 99.0, 1.0, -1.0, 99.0],
            &[1, 1, 0, 1, 1, 0, 1, 1, 0],
        );
        for (expr, expected) in [
            (x.clone().and(y.clone()), [t, f, u, f, f, f, u, f, u]),
            (x.clone().or(y.clone()), [t, t, t, t, f, u, t, u, u]),
            (!x.clone(), [f, f, f, t, t, t, u, u, u]),
            (x.clone().is_null(), [f, f, f, f, f, f, t, t, t]),
            (col("a").is_null(), [f, f, f, f, f, f, t, t, t]),
            (x.clone().is_null().is_null(), [f; 9]),
            // Nested truth tables exercise validity, not just bool payloads.
            (!(x.clone().and(y.clone())), [f, t, u, t, t, t, u, t, u]),
            ((!x.clone()).or(!y.clone()), [f, t, u, t, t, t, u, t, u]),
            (
                x.clone().or(y.clone()).is_null(),
                [f, f, f, f, f, t, f, t, t],
            ),
            (
                x.clone().and(y.clone().is_null()),
                [f, f, t, f, f, f, f, f, u],
            ),
        ] {
            let program = prepare(&expr, &schema).unwrap();
            assert_predicate(
                &executor
                    .execute_predicate(&program, &batch, &parameters)
                    .unwrap(),
                &expected,
            );
            let keep: Vec<u32> = expected
                .iter()
                .map(|value| u32::from(*value == t))
                .collect();
            assert_eq!(
                executor
                    .execute_mask(&program, &batch, &parameters)
                    .unwrap(),
                keep
            );
        }
        // Nullable vs non-nullable operands: known false dominates AND, known
        // true dominates OR. Static nullability remains conservative.
        let batch = batch
            .with_column("plain", ColumnData::F32(&[-1.0; 9]))
            .unwrap();
        let false_and = prepare(&col("plain").gt(param("zero")).and(x.clone()), &schema).unwrap();
        assert!(false_and.output_nullable());
        assert_predicate(
            &executor
                .execute_predicate(&false_and, &batch, &parameters)
                .unwrap(),
            &[f; 9],
        );
        let true_or = prepare(&col("plain").lt(param("zero")).or(x.clone()), &schema).unwrap();
        assert_predicate(
            &executor
                .execute_predicate(&true_or, &batch, &parameters)
                .unwrap(),
            &[t; 9],
        );

        // A statically known IS NULL still requires its declared input even if
        // the optimizer folds away the load.
        let known_null_test = prepare(&col("plain").is_null(), &schema).unwrap();
        assert_predicate(
            &executor
                .execute_predicate(&known_null_test, &batch, &parameters)
                .unwrap(),
            &[f; 9],
        );
        assert!(matches!(
            executor.execute_mask(&known_null_test, &Batch::new(0), &parameters),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::MissingInput {
                    kind: InputKind::Column,
                    ..
                }
            )))
        ));

        let batch = input(
            &[1.0, 2.0, 99.0, 4.0, 99.0, f32::NAN, f32::INFINITY, -0.0],
            &[1, 1, 0, 1, 0, 1, 1, 1],
            &[2.0, 2.0, 3.0, 99.0, 99.0, f32::NAN, f32::INFINITY, 0.0],
            &[1, 1, 1, 0, 0, 1, 1, 1],
        );
        for (expr, expected) in [
            (col("a").eq_expr(col("b")), [f, t, u, u, u, f, t, t]),
            (col("a").ne_expr(col("b")), [t, f, u, u, u, t, f, f]),
            (col("a").lt(col("b")), [t, f, u, u, u, f, f, f]),
            (col("a").le(col("b")), [t, t, u, u, u, f, t, t]),
            (col("a").gt(col("b")), [f, f, u, u, u, f, f, f]),
            (col("a").ge(col("b")), [f, t, u, u, u, f, t, t]),
            ((col("a") + col("b")).is_null(), [f, f, t, t, t, f, f, f]),
        ] {
            let program = prepare(&expr, &schema).unwrap();
            assert_predicate(
                &executor
                    .execute_predicate(&program, &batch, &parameters)
                    .unwrap(),
                &expected,
            );
        }
        // A shared SSA operand must preserve NULL and valid NaN distinctions.
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
        assert_predicate(
            &executor
                .execute_predicate(&shared, &batch, &Parameters::new())
                .unwrap(),
            &[t, t, u, t, u, f, t, t],
        );

        // Unknown equality can otherwise have a true payload (sanitized 0 == 0).
        let equality = prepare(&col("a").eq_expr(param("zero")), &schema).unwrap();
        let null = Batch::new(3)
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: &[0.0, f32::NAN, f32::INFINITY],
                    validity: &[0, 0, 0],
                },
            )
            .unwrap();
        assert_predicate(
            &executor
                .execute_predicate(&equality, &null, &parameters)
                .unwrap(),
            &[u; 3],
        );
        assert_eq!(
            executor
                .execute_mask(&equality, &null, &parameters)
                .unwrap(),
            [0; 3]
        );
        let is_null = prepare(&col("a").is_null(), &schema).unwrap();
        assert_eq!(
            executor
                .execute_mask(&is_null, &null, &Parameters::new())
                .unwrap(),
            [1; 3]
        );

        // Tails, empty, all-valid host slices, and mixed nullable parameters.
        let predicate = prepare(&(col("a") * param("p")).gt(param("zero")), &schema).unwrap();
        let size = executor.cube_size() as usize;
        for rows in [0, 1, size - 1, size, size + 1, size * 3 + 7, 4099] {
            let a: Vec<f32> = (0..rows).map(|i| [2.0, -2.0, 99.0][i % 3]).collect();
            let validity: Vec<u32> = (0..rows).map(|i| [1, 1, 0][i % 3]).collect();
            let batch = Batch::new(rows)
                .with_column(
                    "a",
                    ColumnData::NullableF32 {
                        values: &a,
                        validity: &validity,
                    },
                )
                .unwrap();
            for p in [Some(2.0_f32), Some(-1.0), None] {
                let expected: Vec<_> = (0..rows)
                    .map(|i| match p {
                        Some(2.0) => [t, f, u][i % 3],
                        Some(_) => [f, t, u][i % 3],
                        None => u,
                    })
                    .collect();
                let params = Parameters::new()
                    .with("p", p)
                    .unwrap()
                    .with("zero", 0.0_f32)
                    .unwrap();
                assert_predicate(
                    &executor
                        .execute_predicate(&predicate, &batch, &params)
                        .unwrap(),
                    &expected,
                );
                assert_eq!(
                    executor.execute_mask(&predicate, &batch, &params).unwrap(),
                    expected
                        .iter()
                        .map(|v| u32::from(*v == t))
                        .collect::<Vec<_>>()
                );
            }
        }
        let broadcast = prepare(&param("p").eq_expr(param("p")), &schema).unwrap();
        for (p, expected) in [(Some(1.0_f32), t), (Some(f32::NAN), f), (None, u)] {
            assert_predicate(
                &executor
                    .execute_predicate(
                        &broadcast,
                        &Batch::new(4),
                        &Parameters::new().with("p", p).unwrap(),
                    )
                    .unwrap(),
                &[expected; 4],
            );
        }
        let before = executor.define_count();
        let empty = Batch::new(0)
            .with_column("a", ColumnData::F32(&[]))
            .unwrap();
        assert!(matches!(
            executor.execute_predicate(&predicate, &empty, &parameters),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::MissingInput {
                    kind: InputKind::Parameter,
                    ..
                }
            )))
        ));
        let wrong = Parameters::new()
            .with("p", 1_i32)
            .unwrap()
            .with("zero", 0.0_f32)
            .unwrap();
        assert!(matches!(
            executor.execute_mask(&predicate, &empty, &wrong),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::TypeMismatch { .. }
            )))
        ));
        let wrong = Parameters::new()
            .with("p", None::<f32>)
            .unwrap()
            .with("zero", None::<f32>)
            .unwrap();
        assert!(matches!(
            executor.execute_mask(&predicate, &empty, &wrong),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::UnexpectedNull { .. }
            )))
        ));
        let arithmetic = prepare(&col("a"), &schema).unwrap();
        assert!(matches!(
            executor.execute_predicate(&arithmetic, &empty, &parameters),
            Err(ExecutionError::OutputType {
                expected: DataType::Bool,
                actual: DataType::F32
            })
        ));
        assert!(matches!(
            executor.execute_nullable(&predicate, &empty, &parameters),
            Err(ExecutionError::OutputType { .. })
        ));
        assert!(matches!(
            executor.execute(&predicate, &empty, &parameters),
            Err(ExecutionError::OutputType { .. })
        ));
        assert_eq!(executor.define_count(), before);
        // Rejected invocations leave subsequent valid executions usable.
        assert_eq!(
            executor
                .execute_mask(&is_null, &null, &Parameters::new())
                .unwrap(),
            [1; 3]
        );

        // Unique cold identity: null flags/values and row count remain arguments.
        let name = format!(
            "three_valued_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let schema = Schema::from_fields(
            &[Field::new(&name, DataType::F32).with_nullable(true)],
            &[Field::new("p", DataType::F32).with_nullable(true)],
        )
        .unwrap();
        let expression = !col(&name).gt(param("p"));
        let program = prepare(&expression, &schema).unwrap();
        let batch = Batch::new(3)
            .with_column(
                &name,
                ColumnData::NullableF32 {
                    values: &[1.0, 99.0, 3.0],
                    validity: &[1, 0, 1],
                },
            )
            .unwrap();
        let before = executor.define_count();
        assert_predicate(
            &executor
                .execute_predicate(
                    &program,
                    &batch,
                    &Parameters::new().with("p", Some(2.0_f32)).unwrap(),
                )
                .unwrap(),
            &[t, u, f],
        );
        assert_eq!(executor.define_count(), before + 1);
        let batch = Batch::new(2)
            .with_column(&name, ColumnData::F32(&[4.0, 1.0]))
            .unwrap();
        for (p, expected) in [(None, [u, u]), (Some(2.0_f32), [f, t])] {
            let parameters = Parameters::new().with("p", p).unwrap();
            assert_predicate(
                &executor
                    .execute_predicate(&program, &batch, &parameters)
                    .unwrap(),
                &expected,
            );
            assert_eq!(
                executor
                    .execute_mask(&program, &batch, &parameters)
                    .unwrap(),
                expected
                    .iter()
                    .map(|v| u32::from(*v == t))
                    .collect::<Vec<_>>()
            );
            assert_eq!(executor.define_count(), before + 1);
        }
        let shared = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column(name.clone()),
                    ExprNode::Parameter("p".into()),
                    ExprNode::Compare(Comparison::GreaterThan, ExprId::new(0), ExprId::new(1)),
                    ExprNode::Not(ExprId::new(2)),
                    ExprNode::IsNull(ExprId::new(3)),
                ],
                ExprId::new(4),
            ),
            &schema,
        )
        .unwrap();
        let repeated = prepare(&expression.clone().is_null(), &schema).unwrap();
        assert_eq!(executor.kernel_id(&shared), executor.kernel_id(&repeated));
        let expected = [f, f];
        let parameters = Parameters::new().with("p", Some(2.0_f32)).unwrap();
        assert_predicate(
            &executor
                .execute_predicate(&shared, &batch, &parameters)
                .unwrap(),
            &expected,
        );
        let count = executor.define_count();
        assert_eq!(
            executor
                .execute_mask(&repeated, &batch, &parameters)
                .unwrap(),
            [0, 0]
        );
        assert_eq!(executor.define_count(), count);
        assert_ne!(executor.kernel_id(&program), executor.kernel_id(&shared));
        assert_ne!(
            executor.kernel_id(&shared),
            executor.kernel_id(&prepare(&!expression.clone().is_null(), &schema).unwrap())
        );
        // IsNull in the right subtree is remapped and keyed even with DAG sharing.
        let repeated = prepare(
            &expression
                .clone()
                .is_null()
                .or(expression.clone().is_null()),
            &schema,
        )
        .unwrap();
        let shared = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column(name),
                    ExprNode::Parameter("p".into()),
                    ExprNode::Compare(Comparison::GreaterThan, ExprId::new(0), ExprId::new(1)),
                    ExprNode::Not(ExprId::new(2)),
                    ExprNode::IsNull(ExprId::new(3)),
                    ExprNode::Or(ExprId::new(4), ExprId::new(4)),
                ],
                ExprId::new(5),
            ),
            &schema,
        )
        .unwrap();
        assert_eq!(executor.kernel_id(&shared), executor.kernel_id(&repeated));
        assert_eq!(
            executor
                .execute_mask(
                    &repeated,
                    &batch,
                    &Parameters::new().with("p", None::<f32>).unwrap()
                )
                .unwrap(),
            [1, 1]
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_three_valued_conformance() {
        exercise(Device::cpu().expect("real CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_three_valued_conformance() {
        exercise(Device::metal(Default::default()).expect("real native Metal required"));
    }
}
