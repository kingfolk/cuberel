#[cfg(any(feature = "cpu", feature = "metal"))]
mod runtime {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use cuberel::{
        Executor,
        datafusion::{AdapterOptions, register},
    };
    use datafusion::{
        execution::session_state::SessionStateBuilder,
        physical_plan::{collect, displayable},
        prelude::{SessionConfig, SessionContext},
    };
    use std::{error::Error, sync::Arc};

    pub async fn run() -> Result<(), Box<dyn Error>> {
        let backend = std::env::args().nth(1).unwrap_or_else(|| "cpu".into());
        let device = match backend.as_str() {
            #[cfg(feature = "cpu")]
            "cpu" => cubecl::Device::cpu()?,
            #[cfg(feature = "metal")]
            "metal" => cubecl::Device::metal(Default::default())?,
            _ => return Err(format!("backend {backend} is not enabled").into()),
        };
        let executor = Executor::new(device.client())?;
        let ctx = SessionContext::new_with_state(
            register(
                SessionStateBuilder::new_with_default_features()
                    .with_config(SessionConfig::new().with_target_partitions(1)),
                AdapterOptions::with_executor(executor.clone()),
            )
            .build(),
        );
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
        let plan = ctx
            .sql("SELECT x AS value, x IS NULL AS missing FROM t WHERE x >= CAST(30 AS INT)")
            .await?
            .create_physical_plan()
            .await?;
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(text.contains("CubeExec: filter=true"), "{text}");
        println!("{text}");
        let result = collect(plan, ctx.task_ctx()).await?;
        assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        for batch in result {
            println!("{batch:?}");
        }
        println!("backend={backend} stats={:?}", executor.stats());
        Ok(())
    }
}

#[cfg(any(feature = "cpu", feature = "metal"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    runtime::run().await
}

#[cfg(not(any(feature = "cpu", feature = "metal")))]
fn main() {
    eprintln!("Enable cpu or metal to run this example");
    std::process::exit(2);
}
