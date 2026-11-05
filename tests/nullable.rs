use cuberel::{
    Batch, ColumnData, DataType, Error, Expr, ExprId, ExprNode, Field, InputKind, Parameters,
    Schema,
    abi::{
        BindingError, BindingLayout, BufferDescriptor, BufferSource, PhysicalType, ScalarArgument,
        ScalarSource,
    },
    col, param, prepare,
};

fn field(name: &str, nullable: bool) -> Field {
    Field::new(name, DataType::F32).with_nullable(nullable)
}
fn schema() -> Schema {
    Schema::from_fields(
        &[field("a", true), field("b", false)],
        &[field("scale", true), field("bias", true)],
    )
    .unwrap()
}

#[test]
fn nullability_is_bound_and_propagated_without_changing_numeric_type() {
    let schema = schema();
    assert!(schema.columns()[0].is_nullable());
    assert!(!schema.columns()[1].is_nullable());
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )
    .unwrap();
    assert_eq!(program.output_type(), DataType::F32);
    assert!(program.output_nullable());
    assert!(program.nodes()[0].is_nullable());
    assert!(program.nodes()[2].is_nullable());
    let program = prepare(&col("b"), &schema).unwrap();
    assert!(!program.output_nullable());
    assert_eq!(
        Schema::new(&[("a", DataType::F32)], &[]).unwrap().columns(),
        &[field("a", false)]
    );
    assert!(matches!(
        Schema::from_fields(&[field("a", true), field("a", false)], &[]),
        Err(Error::DuplicateName { .. })
    ));
    assert!(matches!(
        Schema::from_fields(&[], &[field("", true)]),
        Err(Error::EmptyName { .. })
    ));
    let invalid =
        Schema::from_fields(&[Field::new("a", DataType::I32).with_nullable(true)], &[]).unwrap();
    assert!(matches!(
        prepare(&col("a"), &invalid),
        Err(Error::UnsupportedType { .. })
    ));
}

#[test]
fn nullable_comparisons_are_rejected_before_three_valued_logic_exists() {
    let schema = schema();
    for expr in [
        col("a").gt(col("b")),
        (col("b") * param("scale")).eq_expr(col("b")),
    ] {
        assert!(matches!(
            prepare(&expr, &schema),
            Err(Error::UnsupportedNullableOperation { .. })
        ));
    }
    assert!(
        !prepare(&col("b").lt(col("b")), &schema)
            .unwrap()
            .output_nullable()
    );
    // Unreachable inputs still bind, but their nullability does not taint the root.
    let program = prepare(
        &Expr::from_nodes(
            vec![ExprNode::Column("b".into()), ExprNode::Column("a".into())],
            ExprId::new(0),
        ),
        &schema,
    )
    .unwrap();
    assert!(!program.output_nullable());
    assert!(program.nodes()[1].is_nullable());
}

#[test]
fn validity_views_reject_wrong_lengths_and_non_boolean_flags() {
    assert!(matches!(
        Batch::new(2).with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 2.0],
                validity: &[1]
            }
        ),
        Err(Error::ValidityLengthMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
    assert!(matches!(
        Batch::new(2).with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 2.0],
                validity: &[1, 2]
            }
        ),
        Err(Error::InvalidValidity {
            index: 1,
            value: 2,
            ..
        })
    ));
    assert!(matches!(
        Batch::new(1).with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 2.0],
                validity: &[1, 1]
            }
        ),
        Err(Error::ColumnLengthMismatch { .. })
    ));
    assert!(
        Batch::new(0)
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: &[],
                    validity: &[]
                }
            )
            .is_ok()
    );
}

#[test]
fn nullable_layout_orders_value_validity_planes_and_per_type_scalars() {
    let schema = schema();
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )
    .unwrap();
    let layout = BindingLayout::new(&program);
    let a = schema.column_id("a").unwrap();
    let b = schema.column_id("b").unwrap();
    assert_eq!(
        layout
            .buffers()
            .iter()
            .map(|s| s.source())
            .collect::<Vec<_>>(),
        [
            BufferSource::Column(a),
            BufferSource::ColumnValidity(a),
            BufferSource::Column(b),
            BufferSource::Output,
            BufferSource::OutputValidity
        ]
    );
    assert_eq!(
        layout
            .buffers()
            .iter()
            .map(|s| s.element_type())
            .collect::<Vec<_>>(),
        [
            PhysicalType::F32,
            PhysicalType::U32,
            PhysicalType::F32,
            PhysicalType::F32,
            PhysicalType::U32
        ]
    );
    assert_eq!(
        layout
            .scalars()
            .iter()
            .map(|s| (s.element_type(), s.ordinal_within_type()))
            .collect::<Vec<_>>(),
        [
            (PhysicalType::U32, 0),
            (PhysicalType::F32, 0),
            (PhysicalType::U32, 1),
            (PhysicalType::F32, 1),
            (PhysicalType::U32, 2)
        ]
    );
    assert_eq!(
        layout.scalars()[2].source(),
        ScalarSource::ParameterValidity(schema.parameter_id("scale").unwrap())
    );
    let batch = Batch::new(2)
        .with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 99.0],
                validity: &[1, 0],
            },
        )
        .unwrap()
        .with_column("b", ColumnData::F32(&[2.0, 3.0]))
        .unwrap();
    let params = Parameters::new()
        .with("bias", None::<f32>)
        .unwrap()
        .with("scale", 2.0_f32)
        .unwrap();
    let host = layout.bind_host(&batch, &params).unwrap();
    assert_eq!(host.column_validities(), &[Some([1, 0].as_slice()), None]);
    assert_eq!(
        host.scalars(),
        [
            ScalarArgument::U32(2),
            ScalarArgument::F32(2.0),
            ScalarArgument::U32(1),
            ScalarArgument::F32(0.0),
            ScalarArgument::U32(0)
        ]
    );
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
    assert_eq!(
        layout.validate_buffers(2, &descriptors).unwrap().byte_len(),
        8
    );
    descriptors[4].element_type = PhysicalType::F32;
    assert!(matches!(
        layout.validate_buffers(2, &descriptors),
        Err(BindingError::BufferType {
            index: 4,
            expected: PhysicalType::U32,
            ..
        })
    ));
    descriptors[4].element_type = PhysicalType::U32;
    descriptors[4].allocation_bytes = 4;
    assert!(matches!(
        layout.validate_buffers(2, &descriptors),
        Err(BindingError::BufferTooSmall { index: 4, .. })
    ));
    assert!(matches!(
        layout.validate_buffers(2, &descriptors[..4]),
        Err(BindingError::BufferCount { .. })
    ));
}

#[test]
fn bindings_accept_all_valid_inputs_and_enforce_non_null_contracts() {
    let nullable = BindingLayout::new(&prepare(&(col("a") * param("scale")), &schema()).unwrap());
    let all_valid = Batch::new(2)
        .with_column("a", ColumnData::F32(&[1.0, 2.0]))
        .unwrap();
    assert_eq!(
        nullable
            .bind_host(
                &all_valid,
                &Parameters::new().with("scale", Some(3.0_f32)).unwrap()
            )
            .unwrap()
            .column_validities(),
        &[None]
    );
    let strict_schema = Schema::new(&[("a", DataType::F32)], &[("scale", DataType::F32)]).unwrap();
    let strict =
        BindingLayout::new(&prepare(&(col("a") * param("scale")), &strict_schema).unwrap());
    let nulls = Batch::new(2)
        .with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 0.0],
                validity: &[1, 0],
            },
        )
        .unwrap();
    let params = Parameters::new().with("scale", 2.0_f32).unwrap();
    assert!(matches!(
        strict.bind_host(&nulls, &params),
        Err(BindingError::Input(Error::UnexpectedNull {
            kind: InputKind::Column,
            ..
        }))
    ));
    assert!(matches!(
        strict.bind_host(
            &all_valid,
            &Parameters::new().with("scale", None::<f32>).unwrap()
        ),
        Err(BindingError::Input(Error::UnexpectedNull {
            kind: InputKind::Parameter,
            ..
        }))
    ));
    let valid = Batch::new(2)
        .with_column(
            "a",
            ColumnData::NullableF32 {
                values: &[1.0, 2.0],
                validity: &[1, 1],
            },
        )
        .unwrap();
    assert!(strict.bind_host(&valid, &params).is_ok());
    let empty = Batch::new(0)
        .with_column("a", ColumnData::F32(&[]))
        .unwrap();
    assert!(matches!(
        nullable.bind_host(&empty, &Parameters::new()),
        Err(BindingError::Input(Error::MissingInput {
            kind: InputKind::Parameter,
            ..
        }))
    ));
    assert!(matches!(
        nullable.bind_host(&empty, &Parameters::new().with("scale", 1_i32).unwrap()),
        Err(BindingError::Input(Error::TypeMismatch { .. }))
    ));
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod runtime {
    use cubecl::{
        Device,
        prelude::{AddressType, CubeDim, ElemType, ExecutionMode, KernelSettings},
    };
    use cuberel::{
        Batch, ColumnData, DataType, Error, ExecutionError, Executor, Expr, ExprId, ExprNode,
        Field, InputKind, NullableF32, Parameters, Schema,
        abi::{BindingError, BindingLayout, BufferSource, PhysicalType},
        col,
        cubecl_abi::DeviceBuffer,
        param, prepare,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn assert_projection(result: &NullableF32, expected: &[Option<f32>]) {
        assert_eq!(result.len(), expected.len());
        assert_eq!(result.is_empty(), expected.is_empty());
        assert_eq!(result.iter().collect::<Vec<_>>(), expected);
        assert_eq!(
            result
                .validity()
                .iter()
                .map(|v| *v == 1)
                .collect::<Vec<_>>(),
            expected.iter().map(Option::is_some).collect::<Vec<_>>()
        );
        for (&payload, &valid) in result.values().iter().zip(result.validity()) {
            if valid == 0 {
                assert_eq!(payload.to_bits(), 0);
            }
        }
    }

    fn exercise(device: Device) {
        let client = device.client();
        let executor = Executor::new(client.clone()).unwrap();
        let schema = Schema::from_fields(
            &[super::field("a", true), super::field("b", true)],
            &[super::field("scale", true), super::field("bias", true)],
        )
        .unwrap();
        let program = prepare(
            &(col("a") * param("scale") + col("b") + param("bias")),
            &schema,
        )
        .unwrap();
        let params = |scale: Option<f32>, bias: Option<f32>| {
            Parameters::new()
                .with("scale", scale)
                .unwrap()
                .with("bias", bias)
                .unwrap()
        };
        // Both nullable parameters have separate U32 validity ordinals; bindings
        // preserve them through IR declarations and actual launches.
        let declaration = BindingLayout::new(&program)
            .declare(
                &client,
                KernelSettings::new(
                    CubeDim::new_1d(executor.cube_size()).into(),
                    ExecutionMode::Checked,
                    AddressType::U32,
                ),
            )
            .unwrap();
        assert!(
            declaration
                .column_validity(schema.column_id("a").unwrap())
                .is_some()
        );
        assert!(
            declaration
                .parameter_validity(schema.parameter_id("bias").unwrap())
                .is_some()
        );
        assert_ne!(declaration.output(), declaration.output_validity().unwrap());
        let info = declaration.build().info;
        assert_eq!(
            info.scalars
                .iter()
                .find(|f| f.ty == AddressType::U32.unsigned_type())
                .unwrap()
                .count,
            3
        );
        assert_eq!(
            info.scalars
                .iter()
                .find(|f| f.ty == ElemType::Float(cubecl::ir::FloatKind::F32))
                .unwrap()
                .count,
            2
        );

        let size = executor.cube_size() as usize;
        for rows in [0, 1, size - 1, size, size + 1, size * 3 + 7, 4099] {
            let a: Vec<f32> = (0..rows)
                .map(|i| [1.0, f32::NAN, 3.0, 4.0][i % 4])
                .collect();
            let b: Vec<f32> = (0..rows)
                .map(|i| [10.0, 20.0, f32::INFINITY, 40.0][i % 4])
                .collect();
            let av: Vec<u32> = (0..rows).map(|i| [1, 0, 1, 1][i % 4]).collect();
            let bv: Vec<u32> = (0..rows).map(|i| [1, 1, 0, 1][i % 4]).collect();
            let batch = Batch::new(rows)
                .with_column(
                    "a",
                    ColumnData::NullableF32 {
                        values: &a,
                        validity: &av,
                    },
                )
                .unwrap()
                .with_column(
                    "b",
                    ColumnData::NullableF32 {
                        values: &b,
                        validity: &bv,
                    },
                )
                .unwrap();
            let expected: Vec<_> = (0..rows)
                .map(|i| [Some(13.0), None, None, Some(49.0)][i % 4])
                .collect();
            assert_projection(
                &executor
                    .execute_nullable(&program, &batch, &params(Some(2.0), Some(1.0)))
                    .unwrap(),
                &expected,
            );
            for bindings in [
                params(None, Some(1.0)),
                params(Some(2.0), None),
                params(None, None),
            ] {
                assert_projection(
                    &executor
                        .execute_nullable(&program, &batch, &bindings)
                        .unwrap(),
                    &vec![None; rows],
                );
            }
        }
        // Explicit all-null input including arbitrary initialized NaN/Inf payloads.
        let copy = prepare(&col("a"), &schema).unwrap();
        let nulls = Batch::new(3)
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: &[f32::NAN, f32::INFINITY, -123.0],
                    validity: &[0, 0, 0],
                },
            )
            .unwrap();
        assert_projection(
            &executor
                .execute_nullable(&copy, &nulls, &Parameters::new())
                .unwrap(),
            &[None; 3],
        );
        let data = [-0.0, f32::NAN, f32::INFINITY];
        let all_valid = Batch::new(3)
            .with_column("a", ColumnData::F32(&data))
            .unwrap();
        let output = executor
            .execute_nullable(&copy, &all_valid, &Parameters::new())
            .unwrap();
        assert_eq!(output.validity(), [1; 3]);
        assert_eq!(output.values()[0].to_bits(), (-0.0_f32).to_bits());
        assert!(output.values()[1].is_nan());
        assert_eq!(output.values()[2], f32::INFINITY);
        let mut iter = output.iter();
        assert!(iter.next().unwrap().is_some());
        assert!(iter.next().unwrap().unwrap().is_nan());
        // No algebraic null erasure: NULL * 0 remains NULL.
        let mul = prepare(&(col("a") * param("scale")), &schema).unwrap();
        assert_projection(
            &executor
                .execute_nullable(&mul, &nulls, &params(Some(0.0), Some(0.0)))
                .unwrap(),
            &[None; 3],
        );

        // Nullable lowering retains the existing separate-rounding contract.
        let rounding = prepare(&(col("a") * param("scale") + param("bias")), &schema).unwrap();
        let a = [4097.0];
        let batch = Batch::new(1)
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: &a,
                    validity: &[1],
                },
            )
            .unwrap();
        assert_projection(
            &executor
                .execute_nullable(
                    &rounding,
                    &batch,
                    &params(Some(4097.0), Some(-16_785_408.0)),
                )
                .unwrap(),
            &[Some(0.0)],
        );
        // Parameter-only nullable expressions broadcast both value and validity.
        let broadcast = prepare(&(param("scale") + param("bias")), &schema).unwrap();
        assert_projection(
            &executor
                .execute_nullable(&broadcast, &Batch::new(3), &params(Some(2.0), Some(3.0)))
                .unwrap(),
            &[Some(5.0); 3],
        );
        assert_projection(
            &executor
                .execute_nullable(&broadcast, &Batch::new(3), &params(Some(2.0), None))
                .unwrap(),
            &[None; 3],
        );
        // Shared DAG, root before last node, and unreachable nullable input.
        let raw = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column("a".into()),
                    ExprNode::Mul(ExprId::new(0), ExprId::new(0)),
                    ExprNode::Parameter("bias".into()),
                ],
                ExprId::new(1),
            ),
            &schema,
        )
        .unwrap();
        assert_projection(
            &executor
                .execute_nullable(&raw, &batch, &params(Some(0.0), None))
                .unwrap(),
            &[Some(16_785_408.0)],
        );
        let strict_schema = Schema::new(&[("a", DataType::F32)], &[]).unwrap();
        let strict = prepare(&col("a"), &strict_schema).unwrap();
        assert_projection(
            &executor
                .execute_nullable(&strict, &batch, &Parameters::new())
                .unwrap(),
            &[Some(4097.0)],
        );

        // Empty invocations still check required scalars and schema contracts.
        let before = executor.define_count();
        let empty = Batch::new(0)
            .with_column(
                "a",
                ColumnData::NullableF32 {
                    values: &[],
                    validity: &[],
                },
            )
            .unwrap();
        assert_projection(
            &executor
                .execute_nullable(&copy, &empty, &Parameters::new())
                .unwrap(),
            &[],
        );
        assert!(matches!(
            executor.execute(&copy, &empty, &Parameters::new()),
            Err(ExecutionError::NullableOutput)
        ));
        assert!(matches!(
            executor.execute_nullable(&mul, &empty, &Parameters::new()),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::MissingInput {
                    kind: InputKind::Parameter,
                    ..
                }
            )))
        ));
        assert!(matches!(
            executor.execute_nullable(
                &mul,
                &empty,
                &Parameters::new().with("scale", 1_i32).unwrap()
            ),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::TypeMismatch { .. }
            )))
        ));
        assert!(matches!(
            executor.execute(&strict, &nulls, &Parameters::new()),
            Err(ExecutionError::Binding(BindingError::Input(
                Error::UnexpectedNull { .. }
            )))
        ));
        let predicate = prepare(&col("a").gt(col("a")), &strict_schema).unwrap();
        assert!(matches!(
            executor.execute_nullable(&predicate, &empty, &Parameters::new()),
            Err(ExecutionError::OutputType {
                expected: DataType::F32,
                actual: DataType::Bool
            })
        ));
        assert_eq!(executor.define_count(), before);
        assert_projection(
            &executor
                .execute_nullable(&copy, &batch, &Parameters::new())
                .unwrap(),
            &[Some(4097.0)],
        );

        // Separate output buffers must not alias each other or either input plane.
        let layout = BindingLayout::new(&copy);
        let payload = client.create_from_slice(&1.0_f32.to_ne_bytes());
        let valid = client.create_from_slice(&1_u32.to_ne_bytes());
        let out = client.empty(4);
        let mut buffers = vec![
            unsafe { DeviceBuffer::from_raw_parts(payload.clone(), PhysicalType::F32, 1) },
            unsafe { DeviceBuffer::from_raw_parts(valid.clone(), PhysicalType::U32, 1) },
            unsafe { DeviceBuffer::from_raw_parts(out.clone(), PhysicalType::F32, 1) },
            unsafe { DeviceBuffer::from_raw_parts(out, PhysicalType::U32, 1) },
        ];
        assert!(matches!(
            layout.bind_device(&client, 1, &buffers, &Parameters::new()),
            Err(BindingError::AliasedOutputs {
                first: 2,
                second: 3
            })
        ));
        buffers[3] = unsafe { DeviceBuffer::from_raw_parts(valid, PhysicalType::U32, 1) };
        assert!(matches!(
            layout.bind_device(&client, 1, &buffers, &Parameters::new()),
            Err(BindingError::AliasedOutput { input_index: 1 })
        ));
        buffers[3] = unsafe { DeviceBuffer::from_raw_parts(client.empty(4), PhysicalType::U32, 1) };
        assert!(
            layout
                .bind_device(&client, 1, &buffers, &Parameters::new())
                .is_ok()
        );
        assert_eq!(layout.buffers()[3].source(), BufferSource::OutputValidity);

        // Static nullability changes identity; dynamic validity and NULL scalar
        // values rebind without compiling a new kernel.
        let name = format!(
            "nullable_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let schema = Schema::from_fields(
            &[Field::new(&name, DataType::F32).with_nullable(true)],
            &[super::field("scale", true)],
        )
        .unwrap();
        let expr = col(&name) * param("scale");
        let program = prepare(&expr, &schema).unwrap();
        let before = executor.define_count();
        let a = [1.0, 99.0, 3.0];
        let batch = Batch::new(3)
            .with_column(
                &name,
                ColumnData::NullableF32 {
                    values: &a,
                    validity: &[1, 0, 1],
                },
            )
            .unwrap();
        assert_projection(
            &executor
                .execute_nullable(
                    &program,
                    &batch,
                    &Parameters::new().with("scale", 2.0_f32).unwrap(),
                )
                .unwrap(),
            &[Some(2.0), None, Some(6.0)],
        );
        assert_eq!(executor.define_count(), before + 1);
        for scale in [Some(3.0_f32), None] {
            let a = [2.0, 4.0];
            let batch = Batch::new(2)
                .with_column(&name, ColumnData::F32(&a))
                .unwrap();
            let expected = match scale {
                Some(_) => [Some(6.0), Some(12.0)],
                None => [None, None],
            };
            assert_projection(
                &executor
                    .execute_nullable(
                        &program,
                        &batch,
                        &Parameters::new().with("scale", scale).unwrap(),
                    )
                    .unwrap(),
                &expected,
            );
            assert_eq!(executor.define_count(), before + 1);
        }
        let strict_schema =
            Schema::new(&[(&name, DataType::F32)], &[("scale", DataType::F32)]).unwrap();
        assert_ne!(
            executor.kernel_id(&program),
            executor.kernel_id(&prepare(&expr, &strict_schema).unwrap())
        );
        let repeated = prepare(
            &((col(&name) * param("scale")) + (col(&name) * param("scale"))),
            &schema,
        )
        .unwrap();
        let shared = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column(name.clone()),
                    ExprNode::Parameter("scale".into()),
                    ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
                    ExprNode::Add(ExprId::new(2), ExprId::new(2)),
                ],
                ExprId::new(3),
            ),
            &schema,
        )
        .unwrap();
        assert_eq!(executor.kernel_id(&repeated), executor.kernel_id(&shared));
        assert_projection(
            &executor
                .execute_nullable(
                    &shared,
                    &batch,
                    &Parameters::new().with("scale", Some(2.0_f32)).unwrap(),
                )
                .unwrap(),
            &[Some(4.0), None, Some(12.0)],
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_nullable_projection_conformance() {
        exercise(Device::cpu().expect("real CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_nullable_projection_conformance() {
        exercise(Device::metal(Default::default()).expect("real native Metal required"));
    }
}
