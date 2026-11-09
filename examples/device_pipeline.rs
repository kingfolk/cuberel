//! Inspect a reference/fused graph on an explicitly selected real runtime.
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, Executor, Field, FusionMode,
    Parameters, Pipeline, Schema, col, param, prepare,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = std::env::args()
        .nth(1)
        .ok_or("usage: device_pipeline cpu|metal")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("select cpu or native metal".into()),
    };
    let e = Executor::new(device.client())?;
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32),
            Field::new("two", DataType::I32),
        ],
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
    let parameters = Parameters::new()
        .with("limit", 30_i32)?
        .with("two", 2_i32)?;
    let query = Pipeline::new()
        .filter(predicate.clone())
        .aggregate("rows", Aggregate::CountStar)
        .aggregate("count", Aggregate::Count(x.clone()))
        .aggregate("sum", Aggregate::Sum(x.clone()))
        .aggregate("min", Aggregate::Min(x.clone()))
        .aggregate("max", Aggregate::Max(x.clone()));
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        let before = e.stats();
        let pending = e.execute_pipeline(&query.clone().fusion(mode), &batch, &parameters)?;
        let submitted = e.stats();
        assert_eq!(submitted.readback_calls, before.readback_calls);
        assert_eq!(submitted.upload_calls, before.upload_calls);
        println!(
            "{backend} {mode:?}: {} launches, 0 uploads, 0 reads during submission",
            pending.graph().kernel_launches()
        );
        for (id, stage) in pending.graph().stages().iter().enumerate() {
            println!("  {id}: {:?} <- {:?}", stage.kind, stage.dependencies);
        }
        let result = pending.materialize()?;
        assert_eq!(result.row_count, 2);
        assert_eq!(result.aggregates[2].value, AggregateValue::I64(Some(70)));
        println!("  {:?}", result.aggregates);
    }
    let projection = prepare(&(col("x") * param("two")), &schema)?;
    let both = Pipeline::new()
        .filter(predicate)
        .project("doubled", projection)
        .aggregate("sum", Aggregate::Sum(x));
    let result = e
        .execute_pipeline(&both, &batch, &parameters)?
        .materialize()?;
    println!("shared selection projection: {:?}", result.projections);
    #[cfg(feature = "arrow")]
    {
        let arrow = result.into_arrow()?;
        assert_eq!(arrow.projections.unwrap().num_rows(), 2);
        assert_eq!(arrow.aggregates.unwrap().num_rows(), 1);
    }
    Ok(())
}
