//! Resident selected global COUNT/MIN/MAX, with explicit scalar export.
use cubecl::Device;
use cuberel::{
    Aggregate, Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param,
    prepare,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args()
        .nth(1)
        .ok_or("usage: global_aggregates cpu|metal")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("select cpu or native metal".into()),
    };
    let e = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )?;
    let x = prepare(&col("x"), &schema)?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let batch = e.upload(&Batch::new(4).with_column(
        "x",
        ColumnData::NullableI32 {
            values: &[10, 0, 30, 40],
            validity: &[1, 0, 1, 1],
        },
    )?)?;
    let p = Parameters::new().with("limit", 30_i32)?;
    let s = e.select_resident(&predicate, &batch, &p)?;
    let requests = [
        ("rows", Aggregate::CountStar),
        ("valid", Aggregate::Count(x.clone())),
        ("min", Aggregate::Min(x.clone())),
        ("max", Aggregate::Max(x)),
    ];
    let out = e.aggregate_selected(&batch, &s, &requests, &Parameters::new())?;
    drop(batch);
    drop(s);
    let submitted = e.stats();
    assert_eq!(submitted.upload_bytes, 32);
    assert_eq!(submitted.readback_calls, 0);
    assert_eq!(submitted.count_readbacks, 0);
    #[cfg(feature = "arrow")]
    {
        let row = out.materialize_arrow()?;
        assert_eq!(row.num_rows(), 1);
        println!("backend={backend} {row:?}");
    }
    #[cfg(not(feature = "arrow"))]
    println!("backend={backend} {:?}", out.materialize()?);
    println!("reduction={:?}", out.stats());
    println!("before export: {submitted:?}");
    println!("after export: {:?}", e.stats());
    Ok(())
}
