use std::{env, error::Error, sync::Arc};

use arrow_array::{Array, BooleanArray, Float32Array, RecordBatch};
use arrow_schema::{DataType as ArrowType, Field as ArrowField, Schema as ArrowSchema};
use cubecl::Device;
use cuberel::{
    DataType, Executor, Field, Parameters, arrow::schema_from_arrow, col, param, prepare,
};

fn main() -> Result<(), Box<dyn Error>> {
    let device = match env::args().nth(1).as_deref() {
        Some("cpu") => Device::cpu()?,
        Some("metal") => Device::metal(Default::default())?,
        _ => return Err("select backend: arrow_expr cpu (or metal)".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        ArrowType::Float32,
        true,
    )]));
    let padded = Float32Array::from(vec![
        Some(-999.0),
        Some(1.0_f32),
        None,
        Some(3.0),
        Some(-888.0),
    ]);
    let input = RecordBatch::try_new(schema.clone(), vec![Arc::new(padded)])?.slice(1, 3);
    let schema = schema_from_arrow(
        &schema,
        &[
            Field::new("scale", DataType::F32).with_nullable(true),
            Field::new("bias", DataType::F32),
        ],
    )?;
    let parameters = Parameters::new()
        .with("scale", Some(2.0_f32))?
        .with("bias", 1.0_f32)?;
    let program = prepare(&(col("a") * param("scale") + param("bias")), &schema)?;
    let output = executor.execute_arrow(&program, &input, &parameters)?;
    let values = output
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or("expected F32 output")?
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(values, [Some(3.0), None, Some(7.0)]);
    println!("a * scale + bias: {values:?}");
    let program = prepare(&col("a").gt(param("bias")), &schema)?;
    let output = executor.execute_arrow(&program, &input, &parameters)?;
    let values = output
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or("expected Bool output")?
        .iter()
        .collect::<Vec<_>>();
    assert_eq!(values, [Some(false), None, Some(true)]);
    println!("a > bias: {values:?}");
    let before = executor.define_count();
    let mask = executor.execute_arrow_mask(&program, &input, &parameters)?;
    assert_eq!(
        mask.iter().collect::<Vec<_>>(),
        [Some(false), Some(false), Some(true)]
    );
    assert!(mask.nulls().is_none());
    assert_eq!(executor.define_count(), before);
    println!("WHERE mask: {:?}", mask.iter().collect::<Vec<_>>());
    Ok(())
}
