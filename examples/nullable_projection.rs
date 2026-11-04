use std::{env, error::Error};

use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param, prepare,
};

fn main() -> Result<(), Box<dyn Error>> {
    let device = match env::args().nth(1).as_deref() {
        Some("cpu") => Device::cpu()?,
        Some("metal") => Device::metal(Default::default())?,
        _ => return Err("select backend: nullable_projection cpu (or metal)".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("a", DataType::F32).with_nullable(true)],
        &[
            Field::new("scale", DataType::F32).with_nullable(true),
            Field::new("bias", DataType::F32),
        ],
    )?;
    let program = prepare(&(col("a") * param("scale") + param("bias")), &schema)?;
    // NULL payloads are initialized F32 values. Validity, rather than NaN,
    // determines which rows are NULL.
    let data = [1.0_f32, f32::NAN, 3.0];
    let validity = [1_u32, 0, 1];
    let batch = Batch::new(data.len()).with_column(
        "a",
        ColumnData::NullableF32 {
            values: &data,
            validity: &validity,
        },
    )?;
    for (scale, expected) in [
        (Some(2.0_f32), [Some(3.0_f32), None, Some(7.0_f32)]),
        (None, [None, None, None]),
        (Some(3.0_f32), [Some(4.0_f32), None, Some(10.0_f32)]),
    ] {
        let parameters = Parameters::new()
            .with("scale", scale)?
            .with("bias", 1.0_f32)?;
        let before = executor.define_count();
        let output = executor.execute_nullable(&program, &batch, &parameters)?;
        let logical = output.iter().collect::<Vec<_>>();
        assert_eq!(logical, expected);
        println!(
            "scale={scale:?}: {logical:?}; validity={:?}; new definitions={}",
            output.validity(),
            executor.define_count() - before
        );
    }
    Ok(())
}
