//! Resident selection, nested filter, selected projection and compact export.
use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param, prepare,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args()
        .nth(1)
        .ok_or("usage: selected_projection cpu|metal")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("select an enabled cpu or native metal backend".into()),
    };
    let e = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("low", DataType::I32),
            Field::new("high", DataType::I32),
            Field::new("two", DataType::I32),
        ],
    )?;
    let first = prepare(&col("x").ge(param("low")), &schema)?;
    let second = prepare(&col("x").lt(param("high")), &schema)?;
    let twice = prepare(&(col("x") * param("two")), &schema)?;
    let greater = prepare(&col("x").gt(param("low")), &schema)?;
    let batch = e.upload(&Batch::new(5).with_column(
        "x",
        ColumnData::NullableI32 {
            values: &[10, 0, 30, 40, 100],
            validity: &[1, 0, 1, 1, 1],
        },
    )?)?;
    let p = Parameters::new()
        .with("low", 30_i32)?
        .with("high", 50_i32)?
        .with("two", 2_i32)?;
    let selected = e.select_resident(&first, &batch, &p)?;
    let selected = e.filter_selected(&second, &batch, &selected, &p)?;
    let output = e.project_selected(
        &batch,
        &selected,
        &[("twice", &twice), ("greater", &greater)],
        &p,
    )?;
    assert!(selected.belongs_to(&batch));
    drop(batch);
    drop(selected);
    let submitted = e.stats();
    assert_eq!(submitted.upload_bytes, 40);
    assert_eq!(submitted.readback_calls, 0);
    assert_eq!(submitted.count_readbacks, 0);
    #[cfg(feature = "arrow")]
    {
        let compact = output.materialize_arrow()?;
        assert_eq!(compact.num_rows(), 2);
        println!("backend={backend} compact Arrow={compact:?}");
    }
    #[cfg(not(feature = "arrow"))]
    println!(
        "backend={backend} compact columns={:?}",
        output.materialize()?
    );
    println!("before export: {submitted:?}");
    println!("after export: {:?}", e.stats());
    Ok(())
}
