//! Global SQL aggregation over multiple memory batches/partitions.
#[cfg(any(feature = "cpu", feature = "metal"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use cuberel::{
        Executor,
        datafusion::{AdapterOptions, register},
    };
    use datafusion::{
        datasource::MemTable,
        execution::session_state::SessionStateBuilder,
        physical_plan::{collect, displayable},
        prelude::{SessionConfig, SessionContext},
    };
    use std::sync::Arc;
    let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
    let device = match backend.as_str() {
        #[cfg(feature = "cpu")]
        "cpu" => cubecl::Device::cpu()?,
        #[cfg(feature = "metal")]
        "metal" => cubecl::Device::metal(Default::default())?,
        _ => return Err(format!("backend {backend} is not enabled").into()),
    };
    let executor =
        Executor::with_cube_size(device.client(), if backend == "cpu" { 8 } else { 64 })?;
    let config = SessionConfig::new().with_target_partitions(2);
    let state = register(
        SessionStateBuilder::new_with_default_features().with_config(config),
        AdapterOptions::with_executor(executor.clone()),
    )
    .build();
    let ctx = SessionContext::new_with_state(state);
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
    let input = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
        ]))],
    )?;
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(
            schema,
            vec![
                vec![input.slice(0, 2), input.slice(0, 0)],
                vec![input.slice(2, 2)],
            ],
        )?),
    )?;
    let query = "SELECT count(*) AS n, count(x) AS valid, min(x) AS lo, max(x) AS hi FROM t WHERE x >= CAST(30 AS INT)";
    let plan = ctx.sql(query).await?.create_physical_plan().await?;
    println!("{query}\n{}", displayable(plan.as_ref()).indent(true));
    let result = collect(plan, ctx.task_ctx()).await?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].num_rows(), 1);
    println!("{result:#?}\n{:?}", executor.stats());
    Ok(())
}

#[cfg(not(any(feature = "cpu", feature = "metal")))]
fn main() {
    eprintln!("Enable a CubeCL cpu or metal backend to execute SQL.");
}
