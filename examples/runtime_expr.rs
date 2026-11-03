use std::{env, error::Error};

use cubecl::Device;
use cuberel::{Batch, ColumnData, DataType, Executor, Parameters, Schema, col, param, prepare};

fn main() -> Result<(), Box<dyn Error>> {
    let backend = env::args()
        .nth(1)
        .ok_or("select backend: runtime_expr cpu (or metal)")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("expected cpu or metal".into()),
    };
    let client = device.client();
    println!("backend={:?} device={}", device.runtime(), client.name());
    let executor = Executor::new(client)?;
    println!("cube_size={}", executor.cube_size());
    let schema = Schema::new(
        &[("a", DataType::F32), ("b", DataType::F32)],
        &[("scale", DataType::F32), ("bias", DataType::F32)],
    )?;
    let a = [1.0_f32, 2.0, 3.0];
    let b = [10.0_f32, 20.0, 30.0];
    let batch = Batch::new(3)
        .with_column("a", ColumnData::F32(&a))?
        .with_column("b", ColumnData::F32(&b))?;
    // These different expression structures are built and executed in one process.
    for (name, expr, expected) in [
        ("add", col("a") + param("scale"), [3.0, 4.0, 5.0]),
        ("mul", col("a") * param("scale"), [2.0, 4.0, 6.0]),
        (
            "combined",
            col("a") * param("scale") + col("b") + param("bias"),
            [13.0, 25.0, 37.0],
        ),
    ] {
        let program = prepare(&expr, &schema)?;
        let parameters = Parameters::new()
            .with("scale", 2.0_f32)?
            .with("bias", 1.0_f32)?;
        let result = executor.execute(&program, &batch, &parameters)?;
        assert_eq!(result, expected);
        println!("{name}: {result:?}");
    }
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )?;
    for (scale, expected) in [(2.0_f32, [13.0, 25.0, 37.0]), (-1.0, [10.0, 19.0, 28.0])] {
        let parameters = Parameters::new()
            .with("scale", scale)?
            .with("bias", 1.0_f32)?;
        let result = executor.execute(&program, &batch, &parameters)?;
        assert_eq!(result, expected);
        println!("rebound scale={scale}: {result:?}");
    }
    Ok(())
}
