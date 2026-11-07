#![cfg(all(
    feature = "arrow",
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
use arrow_array::{Array, ArrayRef, BooleanArray, Float32Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema};
use cubecl::Device;
use cuberel::{
    DataType, Executor, Field, Parameters, Schema, arrow::ArrowBatch, col, param, prepare,
};
use std::sync::Arc;

fn export(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[
            Field::new("x", DataType::I32).with_nullable(true),
            Field::new("f", DataType::F32).with_nullable(true),
        ],
        &[
            Field::new("limit", DataType::I32),
            Field::new("scale", DataType::F32),
        ],
    )
    .unwrap();
    // Include both columns in the host adapter, then reuse that one upload for
    // different programs. Slice starts within a validity byte and payload array.
    let union = prepare(&col("x").is_null().or(col("f").is_null()), &schema).unwrap();
    let predicate = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let f = prepare(&(col("f") * param("scale")), &schema).unwrap();
    let b = prepare(&col("f").gt(param("scale")), &schema).unwrap();
    let constant = prepare(&param("scale"), &schema).unwrap();
    let input = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("x", ArrowType::Int32, true),
            ArrowField::new("f", ArrowType::Float32, true),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(-9),
                None,
                Some(-9),
                Some(10),
                None,
                Some(30),
                Some(40),
                Some(-9),
            ])) as ArrayRef,
            Arc::new(Float32Array::from(vec![
                Some(-9.0),
                None,
                Some(-9.0),
                Some(1.0),
                Some(2.0),
                None,
                Some(4.0),
                Some(-9.0),
            ])) as ArrayRef,
        ],
    )
    .unwrap()
    .slice(3, 4);
    let batch = {
        let adapter = ArrowBatch::try_new(&union, &input).unwrap();
        e.upload(&adapter.as_batch().unwrap()).unwrap()
    };
    drop(input);
    for threshold in [30, 100, -1] {
        let p = Parameters::new()
            .with("limit", threshold)
            .unwrap()
            .with("scale", 2.0_f32)
            .unwrap();
        let s = e.select_resident(&predicate, &batch, &p).unwrap();
        let projection = e
            .project_selected(
                &batch,
                &s,
                &[
                    ("integer", &x),
                    ("scaled", &f),
                    ("bool", &b),
                    ("constant", &constant),
                ],
                &p,
            )
            .unwrap();
        let before = e.stats();
        let output = projection.materialize_arrow().unwrap();
        let expected_x = match threshold {
            30 => vec![30, 40],
            100 => vec![],
            _ => vec![10, 30, 40],
        };
        let expected_f = match threshold {
            30 => vec![None, Some(8.0)],
            100 => vec![],
            _ => vec![Some(2.0), None, Some(8.0)],
        };
        let expected_b = match threshold {
            30 => vec![None, Some(true)],
            100 => vec![],
            _ => vec![Some(false), None, Some(true)],
        };
        assert_eq!(output.num_rows(), expected_x.len());
        let arrays = output.columns();
        assert_eq!(
            arrays[0]
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .as_ref(),
            expected_x
        );
        assert_eq!(
            arrays[1]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected_f
        );
        assert_eq!(
            arrays[2]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            expected_b
        );
        assert_eq!(
            arrays[3]
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .as_ref(),
            vec![2.0; expected_x.len()]
        );
        let fields = output.schema();
        assert_eq!(
            fields
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            ["integer", "scaled", "bool", "constant"]
        );
        assert_eq!(fields.field(1).data_type(), &ArrowType::Float32);
        assert_eq!(fields.field(2).data_type(), &ArrowType::Boolean);
        assert!(fields.field(0).is_nullable());
        assert!(fields.field(1).is_nullable());
        assert!(!fields.field(3).is_nullable());
        // Four projections: I32 values/validity, F32 values/validity,
        // Bool values/validity, constant values + one count. No tail reads.
        assert_eq!(
            e.stats().readback_bytes - before.readback_bytes,
            4 + expected_x.len() as u64 * 28
        );
        assert_eq!(e.stats().upload_bytes, before.upload_bytes);
    }
    // Select the NULL row and verify Arrow receives a real NULL, not capacity
    // padding or a canonical payload mistaken for a valid zero.
    let nulls = prepare(&col("x").is_null(), &schema).unwrap();
    let s = e
        .select_resident(&nulls, &batch, &Parameters::new())
        .unwrap();
    let output = e
        .project_selected(&batch, &s, &[("null", &x)], &Parameters::new())
        .unwrap()
        .materialize_arrow()
        .unwrap();
    assert_eq!(output.num_rows(), 1);
    assert_eq!(output.column(0).null_count(), 1);
    assert!(output.column(0).is_null(0));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_compact_arrow() {
    export(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_compact_arrow() {
    export(Device::metal(Default::default()).expect("Metal required"));
}
