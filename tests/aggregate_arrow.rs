#![cfg(all(
    feature = "arrow",
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
use arrow_array::{Array, Float32Array, Int32Array, UInt64Array};
use arrow_schema::DataType as ArrowType;
use cubecl::Device;
use cuberel::{
    Aggregate, Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, prepare,
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
        ("rows", Aggregate::CountStar),
        ("valid", Aggregate::Count(x.clone())),
        ("min", Aggregate::Min(x)),
        ("max", Aggregate::Max(f)),
    ];
    for n in [0, 3] {
        let batch = e
            .upload(
                &Batch::new(n)
                    .with_column(
                        "x",
                        ColumnData::NullableI32 {
                            values: &[10, 0, -5][..n],
                            validity: &[1, 0, 1][..n],
                        },
                    )
                    .unwrap()
                    .with_column(
                        "f",
                        ColumnData::F32(&[f32::from_bits(0xffa0_0001), 2.0, 9.0][..n]),
                    )
                    .unwrap(),
            )
            .unwrap();
        let output = e
            .aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap()
            .materialize_arrow()
            .unwrap();
        assert_eq!(output.num_rows(), 1);
        assert_eq!(
            output
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            ["rows", "valid", "min", "max"]
        );
        assert_eq!(output.schema().field(0).data_type(), &ArrowType::UInt64);
        assert!(!output.schema().field(0).is_nullable());
        assert!(output.schema().field(2).is_nullable());
        assert_eq!(
            output
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            n as u64
        );
        assert_eq!(
            output
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .value(0),
            if n == 0 { 0 } else { 2 }
        );
        let min = output
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let max = output
            .column(3)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        if n == 0 {
            assert!(min.is_null(0));
            assert!(max.is_null(0));
        } else {
            assert_eq!(min.value(0), -5);
            assert_eq!(max.value(0).to_bits(), 0x7fc0_0000);
        }
    }
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_aggregate_arrow() {
    export(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_aggregate_arrow() {
    export(Device::metal(Default::default()).expect("Metal required"));
}
