use std::error::Error;

use cuberel::{
    Batch, ColumnData, DataType, Parameters, Schema, abi::BindingLayout, col, param, prepare,
};

fn main() -> Result<(), Box<dyn Error>> {
    let schema = Schema::new(
        &[("b", DataType::F32), ("a", DataType::F32)],
        &[("bias", DataType::F32), ("scale", DataType::F32)],
    )?;
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )?;
    let layout = BindingLayout::new(&program);
    for slot in layout.buffers() {
        println!(
            "buffer {}: {} {:?} {:?}",
            slot.index(),
            slot.name(),
            slot.access(),
            slot.element_type()
        );
    }
    for slot in layout.scalars() {
        println!(
            "scalar {}: {:?}, ordinal within type {}",
            slot.name(),
            slot.element_type(),
            slot.ordinal_within_type()
        );
    }
    let a = [1.0_f32, 2.0, 3.0];
    let b = [10.0_f32, 20.0, 30.0];
    let batch = Batch::new(3)
        .with_column("a", ColumnData::F32(&a))?
        .with_column("b", ColumnData::F32(&b))?;
    let parameters = Parameters::new()
        .with("scale", 2.0_f32)?
        .with("bias", 1.0_f32)?;
    let bound = layout.bind_host(&batch, &parameters)?;
    println!(
        "rows={}, bytes per buffer={}, scalars={:?}",
        bound.extent().row_count(),
        bound.extent().byte_len(),
        bound.scalars()
    );
    Ok(())
}
