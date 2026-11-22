//! Inspect placement before executing, then print actual fragment metrics.
#[cfg(any(feature = "cpu", feature = "metal"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use cuberel::{
        Executor,
        datafusion::{AdapterOptions, PlacementMode, SharedResources, placement_report, register},
    };
    use datafusion::{
        execution::session_state::SessionStateBuilder,
        physical_plan::{collect, display::DisplayableExecutionPlan, displayable},
        prelude::{SessionConfig, SessionContext},
    };
    use std::sync::Arc;
    let mut args = std::env::args().skip(1);
    let backend = args.next().unwrap_or_else(|| "cpu".into());
    let mode = match args.next().as_deref() {
        None | Some("opt-in") => PlacementMode::OptIn,
        Some("cpu-only") => PlacementMode::CpuOnly,
        Some("require") => PlacementMode::RequireCandidates,
        Some(m) => return Err(format!("unknown mode {m}").into()),
    };
    let device = match backend.as_str() {
        #[cfg(feature = "cpu")]
        "cpu" => cubecl::Device::cpu()?,
        #[cfg(feature = "metal")]
        "metal" => cubecl::Device::metal(Default::default())?,
        _ => return Err(format!("backend {backend} not enabled").into()),
    };
    let executor =
        Executor::with_cube_size(device.client(), if backend == "cpu" { 8 } else { 64 })?;
    let resources = SharedResources::default();
    let options = AdapterOptions::with_resources(executor.clone(), resources.clone())
        .with_placement_mode(mode);
    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config(config.clone());
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
        ]))],
    )?;
    ctx.register_batch("t", batch)?;
    let sql = "SELECT x AS value,x IS NULL AS missing FROM t WHERE x >= CAST(30 AS INT)";
    let original = ctx.sql(sql).await?.create_physical_plan().await?;
    let before = executor.stats();
    println!("{sql}\n{}", placement_report(original.clone(), &options)?);
    assert_eq!(before, executor.stats());
    let state = register(
        SessionStateBuilder::new_with_default_features().with_config(config),
        options,
    )
    .build();
    let plan = state
        .physical_optimizers()
        .iter()
        .find(|r| r.name() == "cuberel")
        .ok_or("missing adapter rule")?
        .optimize(original, state.config_options())?;
    let plan = state
        .physical_optimizers()
        .last()
        .ok_or("missing sanity check")?
        .optimize(plan, state.config_options())?;
    println!("{}", displayable(plan.as_ref()).indent(true));
    let result = collect(plan.clone(), ctx.task_ctx()).await?;
    assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
    println!(
        "{result:#?}\n{}",
        DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
    );
    drop(result);
    println!(
        "group={:?}\nexecutor={:?}",
        resources.snapshot(),
        executor.stats()
    );
    Ok(())
}
#[cfg(not(any(feature = "cpu", feature = "metal")))]
fn main() {
    eprintln!(
        "Enable cpu or metal for the execution example; placement_report also works without a runtime."
    );
}
