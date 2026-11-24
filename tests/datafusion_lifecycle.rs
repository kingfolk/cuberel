#![cfg(all(feature = "datafusion", any(feature = "cpu", feature = "metal")))]
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::{
    Executor,
    datafusion::{AdapterOptions, SharedLimits, SharedResources, register},
};
use datafusion::{
    datasource::MemTable,
    execution::session_state::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use futures::TryStreamExt;
use std::sync::Arc;

async fn plan(
    executor: &Executor,
    resources: &SharedResources,
) -> (SessionContext, Arc<dyn ExecutionPlan>) {
    let ctx = SessionContext::new_with_state(
        register(
            SessionStateBuilder::new_with_default_features()
                .with_config(SessionConfig::new().with_target_partitions(1)),
            AdapterOptions::with_resources(executor.clone(), resources.clone()),
        )
        .build(),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(
            (0..129)
                .map(|i| if i == 1 { None } else { Some(i) })
                .collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch.clone(), batch]]).unwrap()),
    )
    .unwrap();
    let p = ctx
        .sql("SELECT x AS value,x IS NULL AS missing FROM t WHERE x > CAST(3 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    (ctx, p)
}
async fn exercise(executor: Executor) {
    let executor = executor.with_aggregate_readback_mode(cuberel::AggregateReadbackMode::Batched);
    let resources = SharedResources::default();
    let (ctx, p) = plan(&executor, &resources).await;
    let before = executor.stats();
    drop(p.execute(0, ctx.task_ctx()).unwrap());
    assert_eq!(before, executor.stats());
    assert_eq!(resources.snapshot().active_streams, 0);
    let mut stream = p.execute(0, ctx.task_ctx()).unwrap();
    let first = stream.try_next().await.unwrap().unwrap();
    assert_eq!(first.num_rows(), 125);
    assert_eq!(resources.snapshot().active_requests, 0);
    assert_eq!(resources.snapshot().device_bytes, 0);
    let calls = executor.stats().upload_calls;
    tokio::task::yield_now().await;
    tokio::task::yield_now().await;
    assert_eq!(
        calls,
        executor.stats().upload_calls,
        "slow consumer must not prefetch"
    );
    let view = first.column(0).slice(3, 5);
    drop(first);
    drop(stream);
    assert_eq!(resources.snapshot().active_streams, 0);
    assert!(
        resources.snapshot().host_bytes > 0,
        "consumer-held Arrow view retains output reservation"
    );
    assert_eq!(
        view.as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .as_ref(),
        [7, 8, 9, 10, 11]
    );
    drop(view);
    assert_eq!(resources.snapshot().host_bytes, 0);
    assert!(resources.snapshot().peak_device_bytes > 0);

    let limited = SharedResources::new(SharedLimits {
        max_streams: 1,
        ..SharedLimits::default()
    });
    let (ctx, p) = plan(&executor, &limited).await;
    let mut slow = p.execute(0, ctx.task_ctx()).unwrap();
    let first = slow.try_next().await.unwrap().unwrap();
    // Separately registered options explicitly share the same controller.
    let (other_ctx, other) = plan(&executor, &limited).await;
    let before = executor.stats();
    assert!(
        collect(other, other_ctx.task_ctx())
            .await
            .unwrap_err()
            .to_string()
            .contains("streams")
    );
    assert_eq!(before.upload_calls, executor.stats().upload_calls);
    drop(slow);
    drop(first);
    assert_eq!(limited.snapshot().host_bytes, 0);
    let result = collect(
        datafusion::physical_plan::execution_plan::reset_plan_states(p).unwrap(),
        ctx.task_ctx(),
    )
    .await
    .unwrap();
    assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 250);
    drop(result);
    assert_eq!(limited.snapshot().active_streams, 0);
    assert_eq!(limited.snapshot().host_bytes, 0);

    for (limits, reason) in [
        (
            SharedLimits {
                max_device_bytes: 0,
                ..SharedLimits::default()
            },
            "device_bytes",
        ),
        (
            SharedLimits {
                max_host_bytes: 0,
                ..SharedLimits::default()
            },
            "host_bytes",
        ),
    ] {
        let pool = SharedResources::new(limits);
        let (ctx, p) = plan(&executor, &pool).await;
        let before = executor.stats();
        let error = collect(p, ctx.task_ctx()).await.unwrap_err();
        assert!(error.to_string().contains(reason));
        fn typed(error: &datafusion::common::DataFusionError) -> bool {
            match error {
                datafusion::common::DataFusionError::Context(_, inner) => typed(inner),
                datafusion::common::DataFusionError::External(inner) => {
                    inner.is::<cuberel::datafusion::SharedResourceError>()
                }
                _ => false,
            }
        }
        assert!(
            typed(&error),
            "shared resource errors retain structured causes: {error:?}"
        );
        assert_eq!(before.upload_calls, executor.stats().upload_calls);
        assert_eq!(pool.snapshot().host_bytes, 0);
        assert_eq!(pool.snapshot().device_bytes, 0);
        assert_eq!(pool.snapshot().active_requests, 0);
        assert!(pool.snapshot().rejected_reservations > 0);
    }
    // Aggregate scalar state and its final output use the same resource group.
    let (ctx, _) = plan(&executor, &resources).await;
    let query = ctx
        .sql("SELECT sum(x),count(*) FROM t WHERE x > CAST(3 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let result = collect(query, ctx.task_ctx()).await.unwrap();
    assert_eq!(result[0].num_rows(), 1);
    drop(result);
    assert_eq!(resources.snapshot().host_bytes, 0);
    assert_eq!(resources.snapshot().active_streams, 0);
}
#[cfg(feature = "cpu")]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_shared_lifecycle() {
    exercise(Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap()).await;
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[tokio::test(flavor = "multi_thread")]
async fn metal_shared_lifecycle() {
    exercise(
        Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    )
    .await;
}
