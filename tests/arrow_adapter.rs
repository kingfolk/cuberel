#![cfg(feature = "arrow")]

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, RecordBatch, RecordBatchOptions, StringArray,
};
use arrow_buffer::{NullBuffer, ScalarBuffer};
use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema};
use cuberel::{
    ColumnData, DataType, Error, Field, Parameters, Schema,
    abi::{BindingError, BindingLayout},
    arrow::{ArrowAdapterError, ArrowBatch, schema_from_arrow},
    col, param, prepare,
};

fn schema(nullable: bool) -> Schema {
    Schema::from_fields(
        &[Field::new("a", DataType::F32).with_nullable(nullable)],
        &[],
    )
    .unwrap()
}
fn records(fields: Vec<ArrowField>, arrays: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), arrays).unwrap()
}
fn empty_columns(rows: usize) -> RecordBatch {
    RecordBatch::try_new_with_options(
        Arc::new(ArrowSchema::empty()),
        vec![],
        &RecordBatchOptions::new().with_row_count(Some(rows)),
    )
    .unwrap()
}

#[test]
fn arrow_schema_preserves_names_nullability_and_explicit_parameter_fields() {
    let arrow = ArrowSchema::new(vec![
        ArrowField::new("a", ArrowType::Float32, true),
        ArrowField::new("b", ArrowType::Float32, false),
    ]);
    let parameters = [Field::new("scale", DataType::F32).with_nullable(true)];
    let schema = schema_from_arrow(&arrow, &parameters).unwrap();
    assert_eq!(
        schema.columns(),
        [
            Field::new("a", DataType::F32).with_nullable(true),
            Field::new("b", DataType::F32)
        ]
    );
    assert_eq!(schema.parameters(), parameters);
    for ty in [ArrowType::Float64, ArrowType::Boolean, ArrowType::Utf8] {
        assert!(
            matches!(schema_from_arrow(&ArrowSchema::new(vec![ArrowField::new("bad", ty.clone(), true)]), &[]), Err(ArrowAdapterError::UnsupportedType { data_type, .. }) if data_type == ty)
        );
    }
    assert!(matches!(
        schema_from_arrow(
            &ArrowSchema::new(vec![
                ArrowField::new("a", ArrowType::Float32, true),
                ArrowField::new("a", ArrowType::Float32, false)
            ]),
            &[]
        ),
        Err(ArrowAdapterError::Input(Error::DuplicateName { .. }))
    ));
    assert!(matches!(
        schema_from_arrow(
            &ArrowSchema::new(vec![ArrowField::new("", ArrowType::Float32, true)]),
            &[]
        ),
        Err(ArrowAdapterError::Input(Error::EmptyName { .. }))
    ));
}

#[test]
fn independent_value_and_bitmap_offsets_are_borrowed_once() {
    let values = ScalarBuffer::from(vec![-99.0, -98.0, 1.0, 777.0, 3.0, 4.0, -97.0]).slice(2, 4);
    let nulls = NullBuffer::from(vec![
        false, true, false, true, false, true, false, true, true, false,
    ])
    .slice(5, 4);
    assert_eq!(nulls.inner().offset(), 5);
    let array = Float32Array::new(values, Some(nulls));
    let program = prepare(&col("a"), &schema(true)).unwrap();
    let layout = BindingLayout::new(&program);
    let input = ArrowBatch::from_columns(&program, 4, &[("a", &array)]).unwrap();
    let batch = input.as_batch().unwrap();
    let host = layout.bind_host(&batch, &Parameters::new()).unwrap();
    assert_eq!(
        host.columns()[0].as_f32().unwrap().as_ptr(),
        array.values().as_ptr()
    );
    assert_eq!(host.columns()[0].as_f32().unwrap(), [1.0, 777.0, 3.0, 4.0]);
    assert_eq!(host.column_validities()[0].unwrap(), [1, 0, 1, 1]);
    let slice = array.slice(1, 3).slice(1, 2);
    let input = ArrowBatch::from_columns(&program, 2, &[("a", &slice)]).unwrap();
    let batch = input.as_batch().unwrap();
    let host = layout.bind_host(&batch, &Parameters::new()).unwrap();
    assert_eq!(
        host.columns()[0].as_f32().unwrap().as_ptr(),
        slice.values().as_ptr()
    );
    assert_eq!(host.columns()[0].as_f32().unwrap(), [3.0, 4.0]);
    assert_eq!(host.column_validities(), &[None]);
}

#[test]
fn all_bitmap_bit_offsets_and_empty_nested_slices_match_logical_rows() {
    let logical = vec![
        Some(-99.0),
        None,
        Some(1.0),
        Some(2.0),
        None,
        Some(4.0),
        None,
        Some(6.0),
        Some(7.0),
        None,
        Some(9.0),
        None,
        Some(11.0),
        Some(12.0),
        None,
        Some(14.0),
        Some(15.0),
        None,
        Some(17.0),
        Some(-98.0),
    ];
    let array = Float32Array::from(logical.clone());
    let program = prepare(&col("a"), &schema(true)).unwrap();
    let layout = BindingLayout::new(&program);
    for offset in 0..10 {
        for len in [0, 1, 3, 9] {
            let sliced = array.slice(offset, len + 1).slice(1, len);
            let input = ArrowBatch::from_columns(&program, len, &[("a", &sliced)]).unwrap();
            assert_eq!(input.row_count(), len);
            let batch = input.as_batch().unwrap();
            let host = layout.bind_host(&batch, &Parameters::new()).unwrap();
            assert_eq!(
                host.columns()[0].as_f32().unwrap().as_ptr(),
                sliced.values().as_ptr()
            );
            let actual: Vec<_> = host.columns()[0]
                .as_f32()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if host.column_validities()[0].is_none_or(|valid| valid[i] == 1) {
                        Some(v)
                    } else {
                        None
                    }
                })
                .collect();
            assert_eq!(actual, logical[offset + 1..offset + 1 + len]);
        }
    }
}

#[test]
fn direct_arrays_validate_names_lengths_and_required_columns() {
    let schema = Schema::new(&[("b", DataType::F32), ("a", DataType::F32)], &[]).unwrap();
    let program = prepare(&(col("a") + col("b")), &schema).unwrap();
    let a = Float32Array::from(vec![1.0, 2.0]);
    let b = Float32Array::from(vec![10.0, 20.0]);
    let short = Float32Array::from(vec![1.0]);
    let input = ArrowBatch::from_columns(&program, 2, &[("a", &a), ("b", &b)]).unwrap();
    let batch = input.as_batch().unwrap();
    let bound = BindingLayout::new(&program)
        .bind_host(&batch, &Parameters::new())
        .unwrap();
    assert_eq!(
        bound.columns(),
        &[
            ColumnData::F32(b.values().as_ref()),
            ColumnData::F32(a.values().as_ref())
        ]
    );
    assert!(matches!(
        ArrowBatch::from_columns(&program, 2, &[("a", &a), ("a", &b)]),
        Err(ArrowAdapterError::Input(Error::DuplicateName { .. }))
    ));
    assert!(matches!(
        ArrowBatch::from_columns(&program, 2, &[("", &a)]),
        Err(ArrowAdapterError::Input(Error::EmptyName { .. }))
    ));
    assert!(
        matches!(ArrowBatch::from_columns(&program,2,&[("a",&a)]), Err(ArrowAdapterError::Input(Error::MissingInput { name, .. })) if name == "b")
    );
    assert!(
        matches!(ArrowBatch::from_columns(&program,2,&[("a",&a),("b",&b),("extra",&short)]), Err(ArrowAdapterError::Input(Error::ColumnLengthMismatch { name, .. })) if name == "extra")
    );
}

#[test]
fn record_batch_matches_names_ignores_unreferenced_types_and_rejects_ambiguity() {
    let program = prepare(&col("a"), &schema(true)).unwrap();
    let input = records(
        vec![
            ArrowField::new("extra", ArrowType::Utf8, false),
            ArrowField::new("a", ArrowType::Float32, true),
        ],
        vec![
            Arc::new(StringArray::from(vec!["x", "y"])),
            Arc::new(Float32Array::from(vec![Some(1.0), None])),
        ],
    );
    let adapter = ArrowBatch::try_new(&program, &input).unwrap();
    let batch = adapter.as_batch().unwrap();
    assert_eq!(
        BindingLayout::new(&program)
            .bind_host(&batch, &Parameters::new())
            .unwrap()
            .column_validities()[0]
            .unwrap(),
        [1, 0]
    );
    // The convenience schema conversion is deliberately all-F32; explicit
    // Cuberel schemas can instead select supported columns of mixed batches.
    assert!(schema_from_arrow(&input.schema(), &[]).is_err());
    let duplicate = records(
        vec![
            ArrowField::new("a", ArrowType::Float32, false),
            ArrowField::new("a", ArrowType::Float32, false),
        ],
        vec![
            Arc::new(Float32Array::from(vec![1.0])),
            Arc::new(Float32Array::from(vec![2.0])),
        ],
    );
    assert!(matches!(
        ArrowBatch::try_new(&program, &duplicate),
        Err(ArrowAdapterError::Input(Error::DuplicateName { .. }))
    ));
    assert!(matches!(
        ArrowBatch::try_new(&program, &empty_columns(0)),
        Err(ArrowAdapterError::Input(Error::MissingInput { .. }))
    ));
}

#[test]
fn arrow_type_and_actual_null_contracts_are_checked_before_execution() {
    let nullable = prepare(&col("a"), &schema(true)).unwrap();
    let wrong = records(
        vec![ArrowField::new("a", ArrowType::Float64, false)],
        vec![Arc::new(Float64Array::from(vec![1.0]))],
    );
    assert!(matches!(
        ArrowBatch::try_new(&nullable, &wrong),
        Err(ArrowAdapterError::UnsupportedType {
            data_type: ArrowType::Float64,
            ..
        })
    ));
    let strict = prepare(&col("a"), &schema(false)).unwrap();
    let nulls = Float32Array::from(vec![Some(1.0), None]);
    assert!(matches!(
        ArrowBatch::from_columns(&strict, 2, &[("a", &nulls)]),
        Err(ArrowAdapterError::Input(Error::UnexpectedNull { .. }))
    ));
    let all_valid = Float32Array::new(vec![1.0, 2.0].into(), Some(NullBuffer::new_valid(2)));
    assert!(ArrowBatch::from_columns(&strict, 2, &[("a", &all_valid)]).is_ok());
    let empty = Float32Array::new_null(0);
    assert!(ArrowBatch::from_columns(&strict, 0, &[("a", &empty)]).is_ok());
}

#[test]
fn parameter_only_batches_preserve_explicit_rows_and_check_abi_before_allocation() {
    let schema = Schema::new(&[], &[("p", DataType::F32)]).unwrap();
    let program = prepare(&param("p"), &schema).unwrap();
    let input = empty_columns(7);
    let adapter = ArrowBatch::try_new(&program, &input).unwrap();
    assert_eq!(adapter.as_batch().unwrap().row_count(), 7);
    let too_many = (u64::from(u32::MAX) / 4 + 1) as usize;
    assert!(matches!(
        ArrowBatch::from_columns(&program, too_many, &[]),
        Err(ArrowAdapterError::Binding(
            BindingError::AddressOverflow { .. }
        ))
    ));
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod runtime {
    use arrow_array::{Array, BooleanArray, Float32Array, Float64Array};
    use arrow_buffer::{NullBuffer, ScalarBuffer};
    use arrow_schema::{DataType as ArrowType, Field as ArrowField};
    use cubecl::Device;
    use cuberel::{
        DataType, Error, ExecutionError, Executor, Field, InputKind, Parameters, Schema,
        abi::BindingError,
        arrow::{ArrowAdapterError, ArrowBatch, schema_from_arrow},
        col, param, prepare,
    };
    use std::{
        sync::Arc,
        time::{SystemTime, UNIX_EPOCH},
    };

    fn exercise(device: Device) {
        let executor = Executor::new(device.client()).unwrap();
        let fields = vec![
            ArrowField::new("a", ArrowType::Float32, true),
            ArrowField::new("b", ArrowType::Float32, true),
        ];
        let arrow_schema = arrow_schema::Schema::new(fields.clone());
        let schema = schema_from_arrow(
            &arrow_schema,
            &[
                Field::new("scale", DataType::F32).with_nullable(true),
                Field::new("bias", DataType::F32),
                Field::new("limit", DataType::F32),
            ],
        )
        .unwrap();
        let arithmetic = prepare(
            &(col("a") * param("scale") + col("b") + param("bias")),
            &schema,
        )
        .unwrap();
        let predicate = prepare(&col("a").gt(param("limit")), &schema).unwrap();
        let params = Parameters::new()
            .with("scale", Some(2.0_f32))
            .unwrap()
            .with("bias", 1.0_f32)
            .unwrap()
            .with("limit", 2.0_f32)
            .unwrap();
        let size = executor.cube_size() as usize;
        for rows in [0, 1, size - 1, size, size + 1, size * 3 + 7, 4099] {
            for offset in [1, 5, 9] {
                let a: Vec<Option<f32>> = (0..offset + rows + 3)
                    .map(|i| {
                        if i < offset || i >= offset + rows {
                            Some(-999.0)
                        } else {
                            [Some(1.0), None, Some(3.0), Some(4.0)][(i - offset) % 4]
                        }
                    })
                    .collect();
                let b: Vec<Option<f32>> = (0..offset + rows + 3)
                    .map(|i| {
                        if i < offset || i >= offset + rows {
                            Some(-888.0)
                        } else {
                            [Some(10.0), Some(20.0), None, Some(40.0)][(i - offset) % 4]
                        }
                    })
                    .collect();
                // RecordBatch::slice independently slices both arrays/bitmaps.
                let input = super::records(
                    fields.clone(),
                    vec![
                        Arc::new(Float32Array::from(a)),
                        Arc::new(Float32Array::from(b)),
                    ],
                )
                .slice(offset, rows);
                let output = executor
                    .execute_arrow(&arithmetic, &input, &params)
                    .unwrap();
                let output = output.as_any().downcast_ref::<Float32Array>().unwrap();
                assert_eq!(
                    output.iter().collect::<Vec<_>>(),
                    (0..rows)
                        .map(|i| [Some(13.0), None, None, Some(49.0)][i % 4])
                        .collect::<Vec<_>>()
                );
                for i in 0..rows {
                    if output.is_null(i) {
                        assert_eq!(output.value(i).to_bits(), 0);
                    }
                }
                let output = executor.execute_arrow(&predicate, &input, &params).unwrap();
                let output = output.as_any().downcast_ref::<BooleanArray>().unwrap();
                assert_eq!(
                    output.iter().collect::<Vec<_>>(),
                    (0..rows)
                        .map(|i| [Some(false), None, Some(true), Some(true)][i % 4])
                        .collect::<Vec<_>>()
                );
                let mask = executor
                    .execute_arrow_mask(&predicate, &input, &params)
                    .unwrap();
                assert!(mask.nulls().is_none());
                assert_eq!(
                    mask.iter().collect::<Vec<_>>(),
                    (0..rows)
                        .map(|i| Some([false, false, true, true][i % 4]))
                        .collect::<Vec<_>>()
                );
            }
        }
        // Payload offsets and bitmap offsets do not need to share an origin.
        let a = Float32Array::new(
            ScalarBuffer::from(vec![-10.0, -20.0, 1.0, f32::NAN, 3.0]).slice(2, 3),
            Some(
                NullBuffer::from(vec![true, false, true, false, true, true, false, true])
                    .slice(5, 3),
            ),
        );
        let copy = prepare(&col("a"), &schema).unwrap();
        let input = super::records(vec![fields[0].clone()], vec![Arc::new(a)]);
        let adapter = ArrowBatch::try_new(&copy, &input).unwrap();
        let batch = adapter.as_batch().unwrap();
        let owned = executor
            .execute_nullable(&copy, &batch, &Parameters::new())
            .unwrap();
        let pointer = owned.values().as_ptr();
        let arrow = owned.into_arrow();
        assert_eq!(arrow.values().as_ptr(), pointer);
        assert_eq!(
            arrow.iter().collect::<Vec<_>>(),
            [Some(1.0), None, Some(3.0)]
        );
        let sliced = input.slice(1, 2);
        let output = executor
            .execute_arrow(&copy, &sliced, &Parameters::new())
            .unwrap();
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [None, Some(3.0)]
        );
        let all_null = super::records(
            vec![fields[0].clone()],
            vec![Arc::new(Float32Array::new_null(17))],
        );
        let output = executor
            .execute_arrow(&copy, &all_null, &Parameters::new())
            .unwrap();
        assert_eq!(output.null_count(), 17);
        let output = executor
            .execute_arrow(
                &prepare(&col("a").is_null(), &schema).unwrap(),
                &all_null,
                &Parameters::new(),
            )
            .unwrap();
        let output = output.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert!(output.nulls().is_none());
        assert_eq!(output.iter().collect::<Vec<_>>(), [Some(true); 17]);
        // Valid NaN is retained as a value, independently of Arrow validity.
        let special = super::records(
            vec![fields[0].clone()],
            vec![Arc::new(Float32Array::from(vec![
                -0.0,
                f32::NAN,
                f32::INFINITY,
            ]))],
        );
        let output = executor
            .execute_arrow(&copy, &special, &Parameters::new())
            .unwrap();
        let output = output.as_any().downcast_ref::<Float32Array>().unwrap();
        assert!(output.nulls().is_none());
        assert_eq!(output.value(0).to_bits(), (-0.0_f32).to_bits());
        assert!(output.value(1).is_nan());
        assert_eq!(output.value(2), f32::INFINITY);

        let parameter_schema =
            Schema::from_fields(&[], &[Field::new("p", DataType::F32).with_nullable(true)])
                .unwrap();
        let broadcast = prepare(&param("p"), &parameter_schema).unwrap();
        let empty_cols = super::empty_columns(4);
        let output = executor
            .execute_arrow(
                &broadcast,
                &empty_cols,
                &Parameters::new().with("p", None::<f32>).unwrap(),
            )
            .unwrap();
        assert_eq!(output.null_count(), 4);
        let output = executor
            .execute_arrow(
                &broadcast,
                &empty_cols,
                &Parameters::new().with("p", 3.0_f32).unwrap(),
            )
            .unwrap();
        assert_eq!(
            output
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some(3.0); 4]
        );
        // Invalid and empty invocations preserve preflight and API errors.
        let empty = super::records(
            fields.clone(),
            vec![
                Arc::new(Float32Array::new_null(0)),
                Arc::new(Float32Array::new_null(0)),
            ],
        );
        let before = executor.define_count();
        assert!(matches!(
            executor.execute_arrow(&arithmetic, &empty, &Parameters::new()),
            Err(ArrowAdapterError::Execution(ExecutionError::Binding(
                BindingError::Input(Error::MissingInput {
                    kind: InputKind::Parameter,
                    ..
                })
            )))
        ));
        assert!(matches!(
            executor.execute_arrow_mask(&arithmetic, &empty, &params),
            Err(ArrowAdapterError::Execution(
                ExecutionError::OutputType { .. }
            ))
        ));
        let wrong = super::records(
            vec![ArrowField::new("a", ArrowType::Float64, false)],
            vec![Arc::new(Float64Array::from(vec![1.0]))],
        );
        assert!(matches!(
            executor.execute_arrow(&copy, &wrong, &Parameters::new()),
            Err(ArrowAdapterError::UnsupportedType { .. })
        ));
        assert_eq!(executor.define_count(), before);
        assert_eq!(
            executor
                .execute_arrow(&copy, &empty, &Parameters::new())
                .unwrap()
                .len(),
            0
        );

        // Shared core/Arrow compilation identity; values, schema order, row count
        // and NULL rebinding do not become specialization constants.
        let name = format!(
            "arrow_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let fields = vec![ArrowField::new(&name, ArrowType::Float32, true)];
        let schema = schema_from_arrow(
            &arrow_schema::Schema::new(fields.clone()),
            &[
                Field::new("scale", DataType::F32).with_nullable(true),
                Field::new("bias", DataType::F32),
            ],
        )
        .unwrap();
        let program = prepare(&(col(&name) * param("scale") + param("bias")), &schema).unwrap();
        let params = |scale: Option<f32>| {
            Parameters::new()
                .with("scale", scale)
                .unwrap()
                .with("bias", 1.0_f32)
                .unwrap()
        };
        let input = super::records(
            fields.clone(),
            vec![Arc::new(Float32Array::from(vec![
                Some(1.0),
                None,
                Some(3.0),
            ]))],
        );
        let before = executor.define_count();
        let result = executor
            .execute_arrow(&program, &input, &params(Some(2.0)))
            .unwrap();
        assert_eq!(
            result
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some(3.0), None, Some(7.0)]
        );
        assert_eq!(executor.define_count(), before + 1);
        let adapter = ArrowBatch::try_new(&program, &input).unwrap();
        assert_eq!(
            executor
                .execute_nullable(&program, &adapter.as_batch().unwrap(), &params(Some(2.0)))
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some(3.0), None, Some(7.0)]
        );
        assert_eq!(executor.define_count(), before + 1);
        let input = super::records(
            vec![
                ArrowField::new("extra", ArrowType::Float32, false),
                fields[0].clone(),
            ],
            vec![
                Arc::new(Float32Array::from(vec![999.0, 999.0])),
                Arc::new(Float32Array::from(vec![4.0, 1.0])),
            ],
        );
        for (scale, expected) in [(None, [None, None]), (Some(3.0), [Some(13.0), Some(4.0)])] {
            let output = executor
                .execute_arrow(&program, &input, &params(scale))
                .unwrap();
            assert_eq!(
                output
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(executor.define_count(), before + 1);
        }
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_arrow_conformance() {
        exercise(Device::cpu().expect("real CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_arrow_conformance() {
        exercise(Device::metal(Default::default()).expect("real Metal required"));
    }
}
