//! Explicit device selection and completion; timing includes completion/readback.
use std::{error::Error, time::Instant};

use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param, prepare,
};

fn main() -> Result<(), Box<dyn Error>> {
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("requested backend is not enabled; use cpu or native metal".into()),
    };
    let executor = Executor::new(device.client())?;
    println!("{backend}: tile={}", executor.cube_size());
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let batch = Batch::new(4).with_column(
        "x",
        ColumnData::NullableI32 {
            values: &[10, 0, 30, 40],
            validity: &[1, 0, 1, 1],
        },
    )?;
    let parameters = Parameters::new().with("limit", 30_i32)?;
    let selected = executor.select(&predicate, &batch, &parameters)?;
    assert_eq!(selected.materialize()?, [2, 3]);
    println!("{backend}: x=[10,NULL,30,40], WHERE x>=30 -> indices=[2,3]");

    // Enough rows to exercise recursively scanned block counts with default tile.
    let mask: Vec<_> = (0..17003).map(|i| u32::from(i % 97 < 3)).collect();
    for run in ["first", "reuse"] {
        let start = Instant::now();
        let selection = executor.compact_mask(&mask)?;
        // Submission has not requested a count read. Scatter consumed it on device.
        let submit = start.elapsed();
        let start = Instant::now();
        let count = selection.selected_count()?;
        let count_completion = start.elapsed();
        let start = Instant::now();
        let indices = selection.materialize()?;
        let materialize = start.elapsed();
        assert_eq!(count as usize, indices.len());
        assert_eq!(
            indices,
            mask.iter()
                .enumerate()
                .filter_map(|(i, &v)| (v == 1).then_some(i as u32))
                .collect::<Vec<_>>()
        );
        println!(
            "{run}: count={count}, stats={:?}, submit={submit:?}, complete+count={count_completion:?}, materialize(rechecks_count)={materialize:?}, defines={}",
            selection.stats(),
            executor.define_count()
        );
    }
    Ok(())
}
