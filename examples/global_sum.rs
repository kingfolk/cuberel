//! Resident SUM with I32->I64 widening and F32 output.
use cubecl::Device;
use cuberel::{
    Aggregate, Batch, ColumnData, DataType, Executor, Field, Parameters, Schema, col, param,
    prepare,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args()
        .nth(1)
        .ok_or("usage: global_sum cpu|metal")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("select cpu or native metal".into()),
    };
    let e = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[
            Field::new("x", DataType::I32).with_nullable(true),
            Field::new("f", DataType::F32).with_nullable(true),
        ],
        &[Field::new("limit", DataType::I32)],
    )?;
    let x = prepare(&col("x"), &schema)?;
    let f = prepare(&col("f"), &schema)?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let batch = e.upload(
        &Batch::new(4)
            .with_column(
                "x",
                ColumnData::NullableI32 {
                    values: &[10, 0, i32::MAX, i32::MAX],
                    validity: &[1, 0, 1, 1],
                },
            )?
            .with_column(
                "f",
                ColumnData::NullableF32 {
                    values: &[1.0, 0.0, 2.5, 3.5],
                    validity: &[1, 0, 1, 1],
                },
            )?,
    )?;
    let selected = e.select_resident(
        &predicate,
        &batch,
        &Parameters::new().with("limit", 30_i32)?,
    )?;
    let out = e.aggregate_selected(
        &batch,
        &selected,
        &[
            ("wide_sum", Aggregate::Sum(x)),
            ("float_sum", Aggregate::Sum(f)),
        ],
        &Parameters::new(),
    )?;
    drop(batch);
    drop(selected);
    let submitted = e.stats();
    assert_eq!(submitted.upload_bytes, 64);
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
