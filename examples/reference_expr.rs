use std::error::Error;

use cuberel::{Batch, ColumnData, DataType, Parameters, Schema, col, param, prepare};

fn main() -> Result<(), Box<dyn Error>> {
    let schema = Schema::new(
        &[("a", DataType::F32), ("b", DataType::F32)],
        &[("scale", DataType::F32), ("bias", DataType::F32)],
    )?;
    let expression = col("a") * param("scale") + col("b") + param("bias");
    let program = prepare(&expression, &schema)?;
    let a = [1.0_f32, 2.0, 3.0];
    let b = [10.0_f32, 20.0, 30.0];
    let batch = Batch::new(a.len())
        .with_column("a", ColumnData::F32(&a))?
        .with_column("b", ColumnData::F32(&b))?;
    for scale in [2.0_f32, -1.0] {
        let parameters = Parameters::new()
            .with("scale", scale)?
            .with("bias", 1.0_f32)?;
        let result = program.evaluate_reference(&batch, &parameters)?;
        println!("host reference, scale={scale}: {result:?}");
    }
    Ok(())
}
