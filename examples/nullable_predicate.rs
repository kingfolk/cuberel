use std::{env, error::Error};

use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param, prepare,
};

fn main() -> Result<(), Box<dyn Error>> {
    let device = match env::args().nth(1).as_deref() {
        Some("cpu") => Device::cpu()?,
        Some("metal") => Device::metal(Default::default())?,
        _ => return Err("select backend: nullable_predicate cpu (or metal)".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("a", DataType::F32).with_nullable(true)],
        &[Field::new("limit", DataType::F32)],
    )?;
    let batch = Batch::new(3).with_column(
        "a",
        ColumnData::NullableF32 {
            values: &[1.0_f32, 99.0, 3.0],
            validity: &[1_u32, 0, 1],
        },
    )?;
    let parameters = Parameters::new().with("limit", 2.0_f32)?;
    let comparison = col("a").gt(param("limit"));
    for (name, expr, expected, keep) in [
        (
            "a > limit",
            comparison.clone(),
            [Some(false), None, Some(true)],
            [0, 0, 1],
        ),
        (
            "a IS NULL",
            col("a").is_null(),
            [Some(false), Some(true), Some(false)],
            [0, 1, 0],
        ),
        (
            "NOT(a > limit)",
            !comparison.clone(),
            [Some(true), None, Some(false)],
            [1, 0, 0],
        ),
        (
            "a IS NULL OR a > limit",
            col("a").is_null().or(comparison),
            [Some(false), Some(true), Some(true)],
            [0, 1, 1],
        ),
    ] {
        let program = prepare(&expr, &schema)?;
        let output = executor.execute_predicate(&program, &batch, &parameters)?;
        let logical = output.iter().collect::<Vec<_>>();
        assert_eq!(logical, expected);
        let before = executor.define_count();
        let mask = executor.execute_mask(&program, &batch, &parameters)?;
        assert_eq!(mask, keep);
        assert_eq!(executor.define_count(), before);
        println!("{name}: {logical:?}; WHERE mask={mask:?}");
    }
    Ok(())
}
