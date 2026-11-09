#![cfg(all(
    feature = "arrow",
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
use arrow_array::{Array, Float32Array, Int64Array, UInt64Array};
use arrow_schema::DataType as ArrowType;
use cubecl::Device;
use cuberel::{
    Aggregate, Batch, ColumnData, DataType, Executor, Field, Parameters, Schema,
    arrow::{ArrowAdapterError, schema_from_arrow},
    col, prepare,
};
fn export(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[
            Field::new("x", DataType::I32).with_nullable(true),
            Field::new("f", DataType::F32).with_nullable(true),
        ],
        &[],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let f = prepare(&col("f"), &schema).unwrap();
    let requests = [
        ("wide", Aggregate::Sum(x)),
        ("float", Aggregate::Sum(f)),
        ("rows", Aggregate::CountStar),
    ];
    for n in [0, 3] {
        let batch = e
            .upload(
                &Batch::new(n)
                    .with_column("x", ColumnData::I32(&[i32::MAX; 3][..n]))
                    .unwrap()
                    .with_column(
                        "f",
                        ColumnData::NullableF32 {
                            values: &[1.0, 99.0, 2.0][..n],
                            validity: &[1, 0, 1][..n],
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let out = e
            .aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap();
        assert_eq!(out.stats().partial_state_bytes, if n == 0 { 0 } else { 20 });
        let output = out.materialize_arrow().unwrap();
        assert_eq!(output.num_rows(), 1);
        assert_eq!(output.schema().field(0).name(), "wide");
        assert_eq!(output.schema().field(0).data_type(), &ArrowType::Int64);
        assert!(output.schema().field(0).is_nullable());
        assert_eq!(output.schema().field(1).data_type(), &ArrowType::Float32);
        assert!(!output.schema().field(2).is_nullable());
        let wide = output
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let float = output
            .column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert_eq!(
            wide.iter().collect::<Vec<_>>(),
            [if n == 0 {
                None
            } else {
                Some(i64::from(i32::MAX) * 3)
            }]
        );
        assert_eq!(
            float.iter().collect::<Vec<_>>(),
            [if n == 0 { None } else { Some(3.0) }]
        );
        assert_eq!(
            output
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            n as u64
        );
        // Int64 is an aggregate output type, not a newly supported input type.
        assert!(matches!(
            schema_from_arrow(output.schema().as_ref(), &[]),
            Err(ArrowAdapterError::UnsupportedType {
                data_type: ArrowType::Int64,
                ..
            })
        ));
    }
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_sum_arrow() {
    export(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_sum_arrow() {
    export(Device::metal(Default::default()).expect("Metal required"));
}
