use std::{env, error::Error};

use cubecl::Device;
use cuberel::{Batch, ColumnData, DataType, Executor, Parameters, Schema, col, param, prepare};

fn main() -> Result<(), Box<dyn Error>> {
    let device = match env::args().nth(1).as_deref() {
        Some("cpu") => Device::cpu()?,
        Some("metal") => Device::metal(Default::default())?,
        _ => return Err("select backend: predicate_mask cpu (or metal)".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Schema::new(
        &[("a", DataType::F32)],
        &[
            ("scale", DataType::F32),
            ("limit", DataType::F32),
            ("upper", DataType::F32),
        ],
    )?;
    let expression = (col("a") * param("scale"))
        .gt(param("limit"))
        .and(col("a").lt(param("upper")));
    let program = prepare(&expression, &schema)?;
    let data = [1.0_f32, 3.0, 5.0];
    let batch = Batch::new(data.len()).with_column("a", ColumnData::F32(&data))?;
    for (scale, limit, upper, expected) in [
        (2.0_f32, 4.0_f32, 5.0_f32, [0, 1, 0]),
        (1.0_f32, 0.0_f32, 6.0_f32, [1, 1, 1]),
    ] {
        let parameters = Parameters::new()
            .with("scale", scale)?
            .with("limit", limit)?
            .with("upper", upper)?;
        let before = executor.define_count();
        let mask = executor.execute_mask(&program, &batch, &parameters)?;
        assert_eq!(mask, expected);
        println!(
            "scale={scale} limit={limit} upper={upper}: {mask:?}; new definitions={}",
            executor.define_count() - before
        );
    }
    Ok(())
}
