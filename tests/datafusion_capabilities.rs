#![cfg(feature = "datafusion")]
#[cfg(any(feature = "cpu", feature = "metal"))]
use arrow_array::Float32Array;
use arrow_array::{Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::datafusion::translate_expression;
use datafusion::{
    common::ScalarValue,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CastExpr, Column, IsNullExpr, Literal},
    },
};
use std::sync::Arc;

fn column(index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new("duplicate", index))
}
fn lit(v: ScalarValue) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(v))
}
fn binary(
    a: Arc<dyn PhysicalExpr>,
    op: Operator,
    b: Arc<dyn PhysicalExpr>,
) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(a, op, b))
}
fn schema() -> Schema {
    Schema::new(vec![
        Field::new("duplicate", DataType::Int32, true),
        Field::new("duplicate", DataType::Float32, true),
    ])
}

#[test]
fn positions_types_nulls_and_literal_values_have_separate_contracts() {
    let s = schema();
    for index in [0, 1] {
        let t = translate_expression(&column(index), &s).unwrap();
        assert_eq!(t.input_columns, vec![index]);
        assert_eq!(t.program.schema().columns()[0].name(), format!("c{index}"));
    }
    let a = translate_expression(
        &binary(column(0), Operator::Gt, lit(ScalarValue::Int32(Some(1)))),
        &s,
    )
    .unwrap();
    let b = translate_expression(
        &binary(column(0), Operator::Gt, lit(ScalarValue::Int32(Some(99)))),
        &s,
    )
    .unwrap();
    assert_eq!(
        a.program.nodes(),
        b.program.nodes(),
        "values are bindings, not program identity"
    );
    assert!(a.program.output_nullable());
    let null = translate_expression(&lit(ScalarValue::Int32(None)), &s).unwrap();
    assert!(null.program.output_nullable());
    assert!(translate_expression(&lit(ScalarValue::Int64(Some(1))), &s).is_err());
    let cast: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(column(0), DataType::Int64, None));
    assert_eq!(
        translate_expression(&cast, &s).unwrap_err().code,
        "unsupported_cast"
    );
}

#[test]
fn overflow_short_circuit_and_float_order_cannot_be_silently_changed() {
    let s = schema();
    let wrapping = binary(column(0), Operator::Plus, lit(ScalarValue::Int32(Some(1))));
    assert_eq!(
        translate_expression(&wrapping, &s).unwrap_err().code,
        "semantic_mismatch"
    );
    let checked: Arc<dyn PhysicalExpr> = Arc::new(
        BinaryExpr::new(column(0), Operator::Plus, lit(ScalarValue::Int32(Some(1))))
            .with_fail_on_overflow(true),
    );
    assert!(
        translate_expression(&checked, &s)
            .unwrap()
            .program
            .has_integer_arithmetic()
    );
    let right = binary(checked, Operator::Gt, lit(ScalarValue::Int32(Some(0))));
    let left = binary(column(0), Operator::Lt, lit(ScalarValue::Int32(Some(0))));
    let short_circuit = binary(left, Operator::And, right);
    assert!(
        translate_expression(&short_circuit, &s)
            .unwrap_err()
            .detail
            .contains("short-circuit")
    );
    let floats = binary(column(1), Operator::Eq, column(1));
    assert_eq!(
        translate_expression(&floats, &s).unwrap_err().code,
        "semantic_mismatch"
    );
    let null: Arc<dyn PhysicalExpr> = Arc::new(IsNullExpr::new(column(1)));
    assert!(translate_expression(&null, &s).is_ok());
}

#[tokio::test]
async fn cpu_sql_is_the_wrapping_oracle_and_unsupported_plans_still_run() {
    let ctx = datafusion::prelude::SessionContext::new();
    ctx.register_batch(
        "t",
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![Some(i32::MAX), None]))],
        )
        .unwrap(),
    )
    .unwrap();
    let result = ctx
        .sql("SELECT x + CAST(1 AS INT) AS y FROM t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let a = result[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(a.value(0), i32::MIN);
    assert!(a.is_null(1));
}

#[cfg(any(feature = "cpu", feature = "metal"))]
fn conformance(executor: cuberel::Executor) {
    let s = schema();
    let input = RecordBatch::try_new(
        Arc::new(s.clone()),
        vec![
            Arc::new(Int32Array::from(vec![
                Some(1),
                None,
                Some(-2),
                Some(0),
                Some(i32::MAX),
            ])),
            Arc::new(Float32Array::from(vec![
                Some(f32::NAN),
                None,
                Some(-0.0),
                Some(f32::INFINITY),
                Some(f32::from_bits(1)),
            ])),
        ],
    )
    .unwrap();
    let cmp = binary(column(0), Operator::Gt, lit(ScalarValue::Int32(Some(0))));
    let null: Arc<dyn PhysicalExpr> = Arc::new(IsNullExpr::new(column(1)));
    let expressions = vec![
        column(0),
        column(1),
        cmp.clone(),
        binary(cmp, Operator::Or, null.clone()),
        null,
    ];
    for expr in expressions {
        let translated = translate_expression(&expr, &s).unwrap();
        let renamed = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("c0", DataType::Int32, true),
                Field::new("c1", DataType::Float32, true),
            ])),
            input.columns().to_vec(),
        )
        .unwrap();
        let actual = executor
            .execute_arrow(&translated.program, &renamed, &translated.parameters)
            .unwrap();
        let expected = expr
            .evaluate(&input)
            .unwrap()
            .into_array(input.num_rows())
            .unwrap();
        assert_eq!(actual.to_data(), expected.to_data());
    }
    let checked: Arc<dyn PhysicalExpr> = Arc::new(
        BinaryExpr::new(column(0), Operator::Plus, lit(ScalarValue::Int32(Some(1))))
            .with_fail_on_overflow(true),
    );
    let translated = translate_expression(&checked, &s).unwrap();
    let renamed = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("c0", DataType::Int32, true)])),
        vec![input.column(0).clone()],
    )
    .unwrap();
    assert!(checked.evaluate(&input).is_err());
    assert!(
        executor
            .execute_arrow(&translated.program, &renamed, &translated.parameters)
            .is_err()
    );
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_ir_matches_bound_datafusion_expressions() {
    conformance(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
    );
}
#[cfg(feature = "metal")]
#[test]
fn metal_ir_matches_bound_datafusion_expressions() {
    conformance(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(cubecl::metal::MetalDevice::default())
                .unwrap()
                .client(),
            64,
        )
        .unwrap(),
    );
}
