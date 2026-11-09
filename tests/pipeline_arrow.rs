#![cfg(feature = "arrow")]
use cuberel::{AggregateResult, AggregateValue, PipelineResult};
#[test]
fn completed_scalar_branch_has_distinct_arrow_shape() {
    let result = PipelineResult {
        row_count: 0,
        projections: Vec::new(),
        aggregates: vec![AggregateResult {
            name: "sum".into(),
            value: AggregateValue::I64(None),
        }],
    }
    .into_arrow()
    .unwrap();
    assert!(result.projections.is_none());
    assert_eq!(result.row_count, 0);
    let aggregates = result.aggregates.unwrap();
    assert_eq!(aggregates.num_rows(), 1);
    assert_eq!(aggregates.column(0).null_count(), 1);
}
#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn exercise(device: cubecl::Device) {
    use cuberel::{
        Aggregate, Batch, ColumnData, DataType, Executor, Parameters, Pipeline, Schema, col,
        prepare,
    };
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let none = prepare(&col("x").ne_expr(col("x")), &schema).unwrap();
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[1, 2]))
                .unwrap(),
        )
        .unwrap();
    let query = Pipeline::new()
        .project("x", x.clone())
        .aggregate("sum", Aggregate::Sum(x));
    let out = e
        .execute_pipeline(&query, &batch, &Parameters::new())
        .unwrap()
        .materialize_arrow()
        .unwrap();
    assert_eq!(out.projections.unwrap().num_rows(), 2);
    assert_eq!(out.aggregates.unwrap().num_rows(), 1);
    let out = e
        .execute_pipeline(&query.filter(none), &batch, &Parameters::new())
        .unwrap()
        .materialize_arrow()
        .unwrap();
    assert_eq!(out.row_count, 0);
    assert_eq!(out.projections.unwrap().num_rows(), 0);
    let sum = out.aggregates.unwrap();
    assert_eq!(sum.num_rows(), 1);
    assert_eq!(sum.column(0).null_count(), 1);
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_pipeline_arrow() {
    exercise(cubecl::Device::cpu().unwrap());
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_pipeline_arrow() {
    exercise(cubecl::Device::metal(Default::default()).unwrap());
}
