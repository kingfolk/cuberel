#![cfg(feature = "datafusion")]
#[path = "../examples/support/sql_acceptance.rs"]
mod support;

use cuberel::datafusion::{AdapterOptions, CubeAggregateExec, SharedResources, register};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect},
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;
use support::{Case, Fixture, Workload, compare};

fn aggregate(plan: &Arc<dyn ExecutionPlan>) -> Option<&Arc<dyn ExecutionPlan>> {
    if plan.is::<CubeAggregateExec>() {
        Some(plan)
    } else {
        plan.children().into_iter().find_map(aggregate)
    }
}
fn count(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    plan.metrics()
        .unwrap()
        .iter()
        .filter(|m| m.value().name() == name)
        .map(|m| m.value().as_usize())
        .sum()
}
async fn exercise(executor: Option<cuberel::Executor>) {
    let workload = Workload::parse("aggregate").unwrap();
    for case in [
        Case {
            rows: 16385,
            pass: 50,
            nulls: 20,
            batches: 1,
            partitions: 1,
        },
        Case {
            rows: 65,
            pass: 100,
            nulls: 100,
            batches: 3,
            partitions: 2,
        },
        Case {
            rows: 0,
            pass: 50,
            nulls: 20,
            batches: 1,
            partitions: 1,
        },
    ] {
        let fixture = Fixture::new(case).unwrap();
        let expected = fixture.expected(workload).unwrap();
        let mut requests = Vec::new();
        for size in [8192, 65536] {
            let resources = SharedResources::default();
            let options = executor
                .as_ref()
                .map(|e| AdapterOptions::with_resources(e.clone(), resources.clone()))
                .unwrap_or_default();
            let ctx = SessionContext::new_with_state(
                register(
                    SessionStateBuilder::new_with_default_features().with_config(
                        SessionConfig::new()
                            .with_target_partitions(case.partitions)
                            .with_batch_size(size),
                    ),
                    options,
                )
                .build(),
            );
            fixture.register(&ctx).unwrap();
            let plan = ctx
                .sql(&workload.sql(case.pass))
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            assert_eq!(aggregate(&plan).is_some(), executor.is_some());
            let output = collect(plan.clone(), ctx.task_ctx()).await.unwrap();
            compare(&plan.schema(), &output, &expected, false).unwrap();
            if let Some(node) = aggregate(&plan) {
                requests.push(count(node, "requests_completed"));
            }
            drop(output);
            let snapshot = resources.snapshot();
            assert_eq!(
                (
                    snapshot.active_requests,
                    snapshot.active_streams,
                    snapshot.host_bytes,
                    snapshot.device_bytes
                ),
                (0, 0, 0, 0)
            );
            assert!(!snapshot.poisoned);
        }
        if executor.is_some() && case.rows == 16385 {
            assert!(
                requests[0] > requests[1],
                "source setting must change actual request granularity: {requests:?}"
            );
        }
    }
}

#[tokio::test]
async fn cpu_oracle_across_session_batch_sizes() {
    exercise(None).await;
}

#[cfg(feature = "cpu")]
#[tokio::test]
async fn cubecl_cpu_aggregate_across_session_batch_sizes() {
    exercise(Some(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
    ))
    .await;
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[tokio::test]
async fn metal_aggregate_across_session_batch_sizes() {
    exercise(Some(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    ))
    .await;
}
