use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, ExecutionError, Executor, Field, Parameters, Schema, col, param,
    prepare,
};
use std::{env, error::Error};

fn main() -> Result<(), Box<dyn Error>> {
    let device = match env::args().nth(1).as_deref() {
        Some("cpu") => Device::cpu()?,
        Some("metal") => Device::metal(Default::default())?,
        _ => return Err("select backend: checked_i32 cpu (or metal)".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("a", DataType::I32).with_nullable(true)],
        &[
            Field::new("scale", DataType::I32),
            Field::new("bias", DataType::I32),
        ],
    )?;
    let expression = col("a") * param("scale") + param("bias");
    let program = prepare(&expression, &schema)?;
    let parameters = Parameters::new()
        .with("scale", 2_i32)?
        .with("bias", 1_i32)?;
    let batch = Batch::new(3).with_column(
        "a",
        ColumnData::NullableI32 {
            values: &[1, i32::MAX, 3],
            validity: &[1, 0, 1],
        },
    )?;
    let output = executor.execute_nullable_i32(&program, &batch, &parameters)?;
    assert_eq!(output.iter().collect::<Vec<_>>(), [Some(3), None, Some(7)]);
    println!("projection: {:?}", output.iter().collect::<Vec<_>>());
    let predicate = prepare(&expression.gt(param("bias")), &schema)?;
    let mask = executor.execute_mask(&predicate, &batch, &parameters)?;
    assert_eq!(mask, [1, 0, 1]);
    println!("WHERE mask: {mask:?}");
    let overflow = Batch::new(1).with_column("a", ColumnData::I32(&[i32::MAX]))?;
    match executor.execute_nullable_i32(&program, &overflow, &parameters) {
        Err(ExecutionError::IntegerOverflow { row: 0 }) => {
            println!("overflow: checked error at row 0")
        }
        other => return Err(format!("unexpected overflow outcome: {other:?}").into()),
    }
    let again = executor.execute_nullable_i32(&program, &batch, &parameters)?;
    assert_eq!(again.iter().collect::<Vec<_>>(), [Some(3), None, Some(7)]);
    Ok(())
}
