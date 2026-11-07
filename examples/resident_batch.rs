//! Explicit upload, resident queries and blocking host materialization.
use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Executor, Field, HostColumn, Parameters, Schema, col, param,
    prepare,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args()
        .nth(1)
        .ok_or("usage: resident_batch cpu|metal")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("requested backend is not enabled on this platform".into()),
    };
    let executor = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let projection = prepare(&(col("x") * param("limit")), &schema)?;
    let batch = {
        let values = [10, 0, 30, 40];
        let validity = [1, 0, 1, 1];
        executor.upload(&Batch::new(4).with_column(
            "x",
            ColumnData::NullableI32 {
                values: &values,
                validity: &validity,
            },
        )?)?
    };
    let selected = executor.select_resident(
        &predicate,
        &batch,
        &Parameters::new().with("limit", 30_i32)?,
    )?;
    // This is a full-batch projection; selected projection belongs to F3.
    let projected = executor.execute_resident(
        &projection,
        &batch,
        &Parameters::new().with("limit", 2_i32)?,
    )?;
    let submitted = executor.stats();
    assert_eq!(submitted.upload_bytes, 32);
    assert_eq!(submitted.readback_calls, 0);
    assert_eq!(submitted.count_readbacks, 0);
    assert!(selected.belongs_to(&batch));
    drop(batch);
    assert_eq!(selected.materialize()?, [2, 3]);
    let columns = projected.materialize()?;
    let HostColumn::I32(column) = &columns[0] else {
        return Err("expected I32 output".into());
    };
    assert_eq!(
        column.iter().collect::<Vec<_>>(),
        [Some(20), None, Some(60), Some(80)]
    );
    println!(
        "backend={backend} indices=[2, 3] projection={:?}",
        column.iter().collect::<Vec<_>>()
    );
    println!("before materialize: {submitted:?}");
    println!("after materialize: {:?}", executor.stats());
    Ok(())
}
