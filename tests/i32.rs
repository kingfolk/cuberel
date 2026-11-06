use cuberel::{
    Batch, ColumnData, DataType, Error, Expr, ExprId, ExprNode, Field, Parameters, Schema,
    abi::{
        BindingError, BindingLayout, BufferDescriptor, BufferSource, PhysicalType, ScalarArgument,
    },
    col, param, prepare,
};

fn field(name: &str, ty: DataType, nullable: bool) -> Field {
    Field::new(name, ty).with_nullable(nullable)
}
#[test]
fn numeric_types_are_exact_and_nullable_i32_is_supported() {
    let schema = Schema::from_fields(
        &[
            field("i", DataType::I32, true),
            field("f", DataType::F32, false),
        ],
        &[field("p", DataType::I32, true)],
    )
    .unwrap();
    let program = prepare(&(col("i") * param("p")), &schema).unwrap();
    assert_eq!(program.output_type(), DataType::I32);
    assert!(program.output_nullable());
    assert!(program.has_integer_arithmetic());
    for expr in [
        col("i") + col("f"),
        col("f") * col("i"),
        col("i").gt(col("f")),
    ] {
        assert!(matches!(
            prepare(&expr, &schema),
            Err(Error::OperandTypeMismatch { .. })
        ));
    }
    assert!(
        prepare(&col("i").gt(param("p")), &schema)
            .unwrap()
            .output_nullable()
    );
    assert!(
        !prepare(&col("i").is_null(), &schema)
            .unwrap()
            .output_nullable()
    );
}
#[test]
fn mixed_payload_abi_keeps_scalar_ordinals_separate_and_adds_error_plane() {
    let schema = Schema::from_fields(
        &[
            field("f", DataType::F32, false),
            field("i", DataType::I32, true),
        ],
        &[
            field("p", DataType::I32, true),
            field("fp", DataType::F32, false),
            field("q", DataType::I32, true),
        ],
    )
    .unwrap();
    let program = prepare(
        &(col("i") + param("p"))
            .gt(param("q"))
            .and(col("f").gt(param("fp"))),
        &schema,
    )
    .unwrap();
    let layout = BindingLayout::new(&program);
    assert_eq!(
        layout
            .buffers()
            .iter()
            .map(|s| s.element_type())
            .collect::<Vec<_>>(),
        [
            PhysicalType::F32,
            PhysicalType::I32,
            PhysicalType::U32,
            PhysicalType::U32,
            PhysicalType::U32,
            PhysicalType::U32
        ]
    );
    assert_eq!(
        layout.buffers().last().unwrap().source(),
        BufferSource::OutputError
    );
    assert_eq!(
        layout
            .scalars()
            .iter()
            .map(|s| (s.element_type(), s.ordinal_within_type()))
            .collect::<Vec<_>>(),
        [
            (PhysicalType::U32, 0),
            (PhysicalType::I32, 0),
            (PhysicalType::U32, 1),
            (PhysicalType::F32, 0),
            (PhysicalType::I32, 1),
            (PhysicalType::U32, 2)
        ]
    );
    let batch = Batch::new(1)
        .with_column(
            "i",
            ColumnData::NullableI32 {
                values: &[2],
                validity: &[1],
            },
        )
        .unwrap()
        .with_column("f", ColumnData::F32(&[3.0]))
        .unwrap();
    let params = Parameters::new()
        .with("q", None::<i32>)
        .unwrap()
        .with("fp", 4.0_f32)
        .unwrap()
        .with("p", Some(3_i32))
        .unwrap();
    let bound = layout.bind_host(&batch, &params).unwrap();
    assert_eq!(
        bound.columns(),
        [ColumnData::F32(&[3.0]), ColumnData::I32(&[2])]
    );
    assert_eq!(
        bound.scalars(),
        [
            ScalarArgument::U32(1),
            ScalarArgument::I32(3),
            ScalarArgument::U32(1),
            ScalarArgument::F32(4.0),
            ScalarArgument::I32(0),
            ScalarArgument::U32(0)
        ]
    );
    let mut descriptors: Vec<_> = layout
        .buffers()
        .iter()
        .map(|s| BufferDescriptor {
            element_type: s.element_type(),
            logical_len: 1,
            allocation_bytes: 4,
            byte_offset: 0,
            end_padding_bytes: 0,
        })
        .collect();
    assert!(layout.validate_buffers(1, &descriptors).is_ok());
    descriptors[5].element_type = PhysicalType::I32;
    assert!(matches!(
        layout.validate_buffers(1, &descriptors),
        Err(BindingError::BufferType {
            index: 5,
            expected: PhysicalType::U32,
            ..
        })
    ));
}
#[test]
fn i32_validity_and_non_null_bindings_reject_invalid_inputs() {
    assert!(matches!(
        Batch::new(2).with_column(
            "i",
            ColumnData::NullableI32 {
                values: &[1, 2],
                validity: &[1]
            }
        ),
        Err(Error::ValidityLengthMismatch { .. })
    ));
    assert!(matches!(
        Batch::new(1).with_column(
            "i",
            ColumnData::NullableI32 {
                values: &[1],
                validity: &[2]
            }
        ),
        Err(Error::InvalidValidity { .. })
    ));
    let schema = Schema::new(&[("i", DataType::I32)], &[("p", DataType::I32)]).unwrap();
    let layout = BindingLayout::new(&prepare(&(col("i") + param("p")), &schema).unwrap());
    let batch = Batch::new(1)
        .with_column(
            "i",
            ColumnData::NullableI32 {
                values: &[i32::MAX],
                validity: &[0],
            },
        )
        .unwrap();
    assert!(matches!(
        layout.bind_host(&batch, &Parameters::new().with("p", 1_i32).unwrap()),
        Err(BindingError::Input(Error::UnexpectedNull { .. }))
    ));
    let batch = Batch::new(0)
        .with_column("i", ColumnData::I32(&[]))
        .unwrap();
    assert!(matches!(
        layout.bind_host(&batch, &Parameters::new().with("p", None::<i32>).unwrap()),
        Err(BindingError::Input(Error::UnexpectedNull { .. }))
    ));
    assert!(matches!(
        layout.bind_host(&batch, &Parameters::new().with("p", 1.0_f32).unwrap()),
        Err(BindingError::Input(Error::TypeMismatch {
            expected: DataType::I32,
            ..
        }))
    ));
}
#[test]
fn error_plane_tracks_root_dependencies_not_unused_arena_nodes() {
    let schema = Schema::new(&[("i", DataType::I32)], &[("p", DataType::I32)]).unwrap();
    let expr = Expr::from_nodes(
        vec![
            ExprNode::Column("i".into()),
            ExprNode::Parameter("p".into()),
            ExprNode::Add(ExprId::new(0), ExprId::new(1)),
        ],
        ExprId::new(0),
    );
    let program = prepare(&expr, &schema).unwrap();
    assert!(!program.has_integer_arithmetic());
    assert!(
        !BindingLayout::new(&program)
            .buffers()
            .iter()
            .any(|s| s.source() == BufferSource::OutputError)
    );
    assert!(
        !prepare(&col("i").gt(param("p")), &schema)
            .unwrap()
            .has_integer_arithmetic()
    );
    assert!(
        prepare(&(col("i") + param("p")).is_null(), &schema)
            .unwrap()
            .has_integer_arithmetic()
    );
}

#[cfg(feature = "arrow")]
#[test]
fn arrow_schema_maps_int32_and_rejects_numeric_type_mismatch() {
    use arrow_array::{Float32Array, Int32Array};
    use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema};
    use cuberel::arrow::{ArrowAdapterError, ArrowBatch, schema_from_arrow};
    let schema = schema_from_arrow(
        &ArrowSchema::new(vec![ArrowField::new("i", ArrowType::Int32, true)]),
        &[],
    )
    .unwrap();
    assert_eq!(schema.columns()[0].data_type(), DataType::I32);
    let program = prepare(&col("i"), &schema).unwrap();
    let int = Int32Array::from(vec![Some(1), None]);
    assert!(ArrowBatch::from_columns(&program, 2, &[("i", &int)]).is_ok());
    let float = Float32Array::from(vec![1.0, 2.0]);
    assert!(matches!(
        ArrowBatch::from_columns(&program, 2, &[("i", &float)]),
        Err(ArrowAdapterError::TypeMismatch {
            expected: DataType::I32,
            actual: DataType::F32,
            ..
        })
    ));
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod runtime {
    use cubecl::Device;
    use cuberel::{
        Batch, ColumnData, DataType, ExecutionError, Executor, Expr, ExprId, ExprNode, Parameters,
        Schema,
        abi::{BindingError, BindingLayout, PhysicalType},
        col,
        cubecl_abi::DeviceBuffer,
        param, prepare,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    fn pair<'a>(a: &'a [i32], b: &'a [i32]) -> Batch<'a> {
        Batch::new(a.len())
            .with_column("a", ColumnData::I32(a))
            .unwrap()
            .with_column("b", ColumnData::I32(b))
            .unwrap()
    }
    fn exercise(device: Device) {
        let client = device.client();
        let executor = Executor::new(client.clone()).unwrap();
        let schema = Schema::new(
            &[("a", DataType::I32), ("b", DataType::I32)],
            &[("p", DataType::I32), ("q", DataType::I32)],
        )
        .unwrap();
        let add = prepare(&(col("a") + col("b")), &schema).unwrap();
        let mul = prepare(&(col("a") * col("b")), &schema).unwrap();
        // Rust's checked integer primitives provide an independent exact oracle.
        let boundary = [
            i32::MIN,
            i32::MIN + 1,
            -65536,
            -46341,
            -46340,
            -2,
            -1,
            0,
            1,
            2,
            46340,
            46341,
            65536,
            i32::MAX - 1,
            i32::MAX,
        ];
        for (program, multiply) in [(&add, false), (&mul, true)] {
            let mut good_a = Vec::new();
            let mut good_b = Vec::new();
            let mut expected = Vec::new();
            for &a in &boundary {
                for &b in &boundary {
                    let result = if multiply {
                        a.checked_mul(b)
                    } else {
                        a.checked_add(b)
                    };
                    if let Some(v) = result {
                        good_a.push(a);
                        good_b.push(b);
                        expected.push(v);
                    } else {
                        let actual = executor.execute_i32(
                            program,
                            &pair(&[1, a], &[1, b]),
                            &Parameters::new(),
                        );
                        assert!(
                            matches!(actual, Err(ExecutionError::IntegerOverflow { row: 1 })),
                            "a={a} b={b} mul={multiply}: {actual:?}"
                        );
                    }
                }
            }
            assert_eq!(
                executor
                    .execute_i32(program, &pair(&good_a, &good_b), &Parameters::new())
                    .unwrap(),
                expected
            );
        }
        for (expr, expected) in [
            (col("a").eq_expr(col("b")), [0, 0, 1, 0, 0]),
            (col("a").ne_expr(col("b")), [1, 1, 0, 1, 1]),
            (col("a").lt(col("b")), [1, 1, 0, 0, 0]),
            (col("a").le(col("b")), [1, 1, 1, 0, 0]),
            (col("a").gt(col("b")), [0, 0, 0, 1, 1]),
            (col("a").ge(col("b")), [0, 0, 1, 1, 1]),
        ] {
            assert_eq!(
                executor
                    .execute_mask(
                        &prepare(&expr, &schema).unwrap(),
                        &pair(&[i32::MIN, -1, 0, 1, i32::MAX], &[0; 5]),
                        &Parameters::new()
                    )
                    .unwrap(),
                expected
            );
        }
        let grouping = prepare(&((col("a") + param("p")) + param("q")), &schema).unwrap();
        assert!(matches!(
            executor.execute_i32(
                &grouping,
                &pair(&[i32::MAX], &[0]),
                &Parameters::new()
                    .with("p", 1_i32)
                    .unwrap()
                    .with("q", -1_i32)
                    .unwrap()
            ),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        let eager = prepare(
            &col("b")
                .lt(col("b"))
                .and((col("a") + param("p")).gt(col("b"))),
            &schema,
        )
        .unwrap();
        assert!(matches!(
            executor.execute_mask(
                &eager,
                &pair(&[i32::MAX], &[0]),
                &Parameters::new().with("p", 1_i32).unwrap()
            ),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        let is_null = prepare(&(col("a") + param("p")).is_null(), &schema).unwrap();
        assert!(matches!(
            executor.execute_mask(
                &is_null,
                &pair(&[i32::MAX], &[0]),
                &Parameters::new().with("p", 1_i32).unwrap()
            ),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        let raw = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column("a".into()),
                    ExprNode::Parameter("p".into()),
                    ExprNode::Add(ExprId::new(0), ExprId::new(1)),
                ],
                ExprId::new(0),
            ),
            &schema,
        )
        .unwrap();
        assert_eq!(
            executor
                .execute_i32(
                    &raw,
                    &pair(&[i32::MAX], &[0]),
                    &Parameters::new().with("p", 1_i32).unwrap()
                )
                .unwrap(),
            [i32::MAX]
        );

        let schema = Schema::from_fields(
            &[
                super::field("a", DataType::I32, true),
                super::field("b", DataType::I32, true),
                super::field("f", DataType::F32, false),
            ],
            &[
                super::field("p", DataType::I32, true),
                super::field("q", DataType::I32, false),
                super::field("fp", DataType::F32, false),
            ],
        )
        .unwrap();
        let combined = prepare(&(col("a") * param("p") + col("b") + param("q")), &schema).unwrap();
        let predicate = prepare(
            &(col("a") * param("p"))
                .gt(param("q"))
                .and(col("f").gt(param("fp"))),
            &schema,
        )
        .unwrap();
        let params = |p: Option<i32>| {
            Parameters::new()
                .with("p", p)
                .unwrap()
                .with("q", 1_i32)
                .unwrap()
                .with("fp", 2.0_f32)
                .unwrap()
        };
        let size = executor.cube_size() as usize;
        for rows in [0, 1, size - 1, size, size + 1, size * 3 + 7, 4099] {
            let a: Vec<_> = (0..rows).map(|i| [1, i32::MAX, 3, 4][i % 4]).collect();
            let b: Vec<_> = (0..rows).map(|i| [10, 20, i32::MIN, 40][i % 4]).collect();
            let av: Vec<_> = (0..rows).map(|i| [1, 0, 1, 1][i % 4]).collect();
            let bv: Vec<_> = (0..rows).map(|i| [1, 1, 0, 1][i % 4]).collect();
            let floats = vec![3.0; rows];
            let batch = Batch::new(rows)
                .with_column(
                    "a",
                    ColumnData::NullableI32 {
                        values: &a,
                        validity: &av,
                    },
                )
                .unwrap()
                .with_column(
                    "b",
                    ColumnData::NullableI32 {
                        values: &b,
                        validity: &bv,
                    },
                )
                .unwrap()
                .with_column("f", ColumnData::F32(&floats))
                .unwrap();
            let output = executor
                .execute_nullable_i32(&combined, &batch, &params(Some(2)))
                .unwrap();
            assert_eq!(
                output.iter().collect::<Vec<_>>(),
                (0..rows)
                    .map(|i| [Some(13), None, None, Some(49)][i % 4])
                    .collect::<Vec<_>>()
            );
            for (&v, &valid) in output.values().iter().zip(output.validity()) {
                if valid == 0 {
                    assert_eq!(v, 0);
                }
            }
            assert_eq!(
                executor
                    .execute_nullable_i32(&combined, &batch, &params(None))
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                vec![None; rows]
            );
            assert_eq!(
                executor
                    .execute_predicate(&predicate, &batch, &params(Some(2)))
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                (0..rows)
                    .map(|i| [Some(true), None, Some(true), Some(true)][i % 4])
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                executor
                    .execute_mask(&predicate, &batch, &params(None))
                    .unwrap(),
                vec![0; rows]
            );
        }
        // Valid intermediate overflow is still an error when a parent is NULL;
        // NULL operands suppress overflow only for that individual operation.
        let null_schema = Schema::from_fields(
            &[super::field("a", DataType::I32, true)],
            &[
                super::field("p", DataType::I32, false),
                super::field("q", DataType::I32, true),
            ],
        )
        .unwrap();
        let expr = prepare(&((col("a") + param("p")) + param("q")), &null_schema).unwrap();
        let params = Parameters::new()
            .with("p", 1_i32)
            .unwrap()
            .with("q", None::<i32>)
            .unwrap();
        let null = Batch::new(1)
            .with_column(
                "a",
                ColumnData::NullableI32 {
                    values: &[i32::MAX],
                    validity: &[0],
                },
            )
            .unwrap();
        assert_eq!(
            executor
                .execute_nullable_i32(&expr, &null, &params)
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [None]
        );
        let valid = Batch::new(1)
            .with_column("a", ColumnData::I32(&[i32::MAX]))
            .unwrap();
        assert!(matches!(
            executor.execute_nullable_i32(&expr, &valid, &params),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        assert!(matches!(
            executor.execute(&expr, &null, &params),
            Err(ExecutionError::OutputType { .. })
        ));
        assert!(matches!(
            executor.execute_i32(&expr, &null, &params),
            Err(ExecutionError::NullableOutput)
        ));
        // Recovery and shared cache identity across successful/overflow/NULL calls.
        let name = format!(
            "checked_i32_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let schema = Schema::from_fields(
            &[super::field(&name, DataType::I32, true)],
            &[super::field("p", DataType::I32, true)],
        )
        .unwrap();
        let program = prepare(&(col(&name) * param("p")), &schema).unwrap();
        let before = executor.define_count();
        let batch = Batch::new(2)
            .with_column(&name, ColumnData::I32(&[2, 3]))
            .unwrap();
        assert_eq!(
            executor
                .execute_nullable_i32(
                    &program,
                    &batch,
                    &Parameters::new().with("p", 4_i32).unwrap()
                )
                .unwrap()
                .values(),
            [8, 12]
        );
        assert_eq!(executor.define_count(), before + 1);
        let overflow = Batch::new(1)
            .with_column(&name, ColumnData::I32(&[i32::MAX]))
            .unwrap();
        assert!(matches!(
            executor.execute_nullable_i32(
                &program,
                &overflow,
                &Parameters::new().with("p", 2_i32).unwrap()
            ),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        assert_eq!(
            executor
                .execute_nullable_i32(
                    &program,
                    &overflow,
                    &Parameters::new().with("p", None::<i32>).unwrap()
                )
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [None]
        );
        assert_eq!(
            executor
                .execute_nullable_i32(
                    &program,
                    &batch,
                    &Parameters::new().with("p", 3_i32).unwrap()
                )
                .unwrap()
                .values(),
            [6, 9]
        );
        assert_eq!(executor.define_count(), before + 1);
        let repeated = prepare(
            &((col(&name) * param("p")) + (col(&name) * param("p"))),
            &schema,
        )
        .unwrap();
        let shared = prepare(
            &Expr::from_nodes(
                vec![
                    ExprNode::Column(name.clone()),
                    ExprNode::Parameter("p".into()),
                    ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
                    ExprNode::Add(ExprId::new(2), ExprId::new(2)),
                ],
                ExprId::new(3),
            ),
            &schema,
        )
        .unwrap();
        assert_eq!(executor.kernel_id(&repeated), executor.kernel_id(&shared));
        assert_eq!(
            executor
                .execute_nullable_i32(
                    &repeated,
                    &batch,
                    &Parameters::new().with("p", 2_i32).unwrap()
                )
                .unwrap()
                .values(),
            [8, 12]
        );
        let count = executor.define_count();
        assert!(matches!(
            executor.execute_nullable_i32(
                &shared,
                &overflow,
                &Parameters::new().with("p", 1_i32).unwrap()
            ),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        assert_eq!(executor.define_count(), count);
        let layout = BindingLayout::new(&program);
        let handles = [
            client.create_from_slice(&1_i32.to_ne_bytes()),
            client.create_from_slice(&1_u32.to_ne_bytes()),
            client.empty(4),
            client.empty(4),
            client.empty(4),
        ];
        let mut buffers: Vec<_> = layout
            .buffers()
            .iter()
            .zip(&handles)
            .map(|(slot, handle)| unsafe {
                DeviceBuffer::from_raw_parts(handle.clone(), slot.element_type(), 1)
            })
            .collect();
        buffers[4] =
            unsafe { DeviceBuffer::from_raw_parts(handles[2].clone(), PhysicalType::U32, 1) };
        assert!(matches!(
            layout.bind_device(
                &client,
                1,
                &buffers,
                &Parameters::new().with("p", 2_i32).unwrap()
            ),
            Err(BindingError::AliasedOutputs {
                first: 2,
                second: 4
            })
        ));
        #[cfg(feature = "arrow")]
        arrow(&executor);
    }
    #[cfg(feature = "arrow")]
    fn arrow(executor: &Executor) {
        use arrow_array::{Array, Int32Array, RecordBatch};
        use arrow_buffer::{NullBuffer, ScalarBuffer};
        use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema};
        use cuberel::arrow::{ArrowAdapterError, ArrowBatch, schema_from_arrow};
        use std::sync::Arc;
        let fields = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "i",
            ArrowType::Int32,
            true,
        )]));
        let schema = schema_from_arrow(&fields, &[super::field("p", DataType::I32, true)]).unwrap();
        let array = Int32Array::new(
            ScalarBuffer::from(vec![-9, 1, i32::MAX, 3]).slice(1, 3),
            Some(
                NullBuffer::from(vec![true, false, true, false, true, true, false, true])
                    .slice(5, 3),
            ),
        );
        let input = RecordBatch::try_new(fields.clone(), vec![Arc::new(array)]).unwrap();
        let program = prepare(&(col("i") * param("p")), &schema).unwrap();
        let params = Parameters::new().with("p", 2_i32).unwrap();
        let output = executor.execute_arrow(&program, &input, &params).unwrap();
        let output = output.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(output.iter().collect::<Vec<_>>(), [Some(2), None, Some(6)]);
        let sliced = input.slice(1, 2);
        let output = executor.execute_arrow(&program, &sliced, &params).unwrap();
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some(6)]
        );
        let adapter = ArrowBatch::try_new(&program, &input).unwrap();
        let result = executor
            .execute_nullable_i32(&program, &adapter.as_batch().unwrap(), &params)
            .unwrap();
        let ptr = result.values().as_ptr();
        let result = result.into_arrow();
        assert_eq!(ptr, result.values().as_ptr());
        let overflow =
            RecordBatch::try_new(fields, vec![Arc::new(Int32Array::from(vec![i32::MAX]))]).unwrap();
        assert!(matches!(
            executor.execute_arrow(&program, &overflow, &params),
            Err(ArrowAdapterError::Execution(
                ExecutionError::IntegerOverflow { row: 0 }
            ))
        ));
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_checked_i32_conformance() {
        exercise(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_checked_i32_conformance() {
        exercise(Device::metal(Default::default()).expect("Metal required"));
    }
}
