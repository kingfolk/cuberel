#![cfg(feature = "datafusion")]
#[path = "../examples/support/sql_acceptance.rs"]
mod support;
use cuberel::datafusion::{AdapterOptions, CubeAggregateExec, CubeExec, SharedResources, register};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    physical_plan::{ExecutionPlan, collect, displayable},
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;
use support::{Case, Fixture, Workload, compare};

fn cases() -> Vec<Case> {
    // Pairwise coverage around both CPU/Metal tiles, exact empty/all-NULL semantics,
    // selection extremes, sliced buffers and single/multi-batch/partition plans.
    [0, 1, 7, 8, 9, 63, 64, 65, 257, 1024, 65, 65]
        .into_iter()
        .enumerate()
        .map(|(i, rows)| Case {
            rows,
            pass: [100, 0, 50, 1, 100, 50, 0, 100, 50, 1, 100, 100][i],
            nulls: [0, 20, 100, 0, 20, 0, 20, 20, 20, 0, 100, 0][i],
            batches: if i % 2 == 0 { 1 } else { 3 },
            partitions: if i % 3 == 0 { 2 } else { 1 },
        })
        .collect()
}
fn cubes(plan: &Arc<dyn ExecutionPlan>) -> usize {
    usize::from(plan.is::<CubeExec>() || plan.is::<CubeAggregateExec>())
        + plan.children().into_iter().map(cubes).sum::<usize>()
}
fn has_global(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<CubeAggregateExec>() || plan.children().into_iter().any(has_global)
}
async fn exercise(executor: Option<cuberel::Executor>) {
    for case in cases() {
        let f = Fixture::new(case).unwrap();
        for w in [Workload::Project, Workload::Aggregate, Workload::Mixed] {
            let resources = SharedResources::default();
            let options = executor
                .as_ref()
                .map(|e| AdapterOptions::with_resources(e.clone(), resources.clone()))
                .unwrap_or_default();
            let config = SessionConfig::new().with_target_partitions(case.partitions);
            let ctx = SessionContext::new_with_state(
                register(
                    SessionStateBuilder::new_with_default_features().with_config(config),
                    options,
                )
                .build(),
            );
            f.register(&ctx).unwrap();
            let sql = w.sql(case.pass);
            let plan = ctx
                .sql(&sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            assert_eq!(
                cubes(&plan) > 0,
                executor.is_some(),
                "{sql}: {}",
                displayable(plan.as_ref()).indent(true)
            );
            if executor.is_some() && w == Workload::Aggregate {
                assert!(
                    has_global(&plan),
                    "global aggregate must execute on the selected backend"
                );
            }
            let out = collect(plan.clone(), ctx.task_ctx()).await.unwrap();
            compare(
                &plan.schema(),
                &out,
                &f.expected(w).unwrap(),
                w == Workload::Project && case.partitions > 1,
            )
            .unwrap_or_else(|e| panic!("{case:?} {sql}: {e}"));
            drop(out);
            let s = resources.snapshot();
            assert_eq!(
                (
                    s.active_requests,
                    s.active_streams,
                    s.host_bytes,
                    s.device_bytes
                ),
                (0, 0, 0, 0)
            );
            assert!(!s.poisoned);
        }
    }
}
#[tokio::test]
async fn datafusion_cpu_acceptance_oracle() {
    exercise(None).await;
}
#[cfg(feature = "cpu")]
#[tokio::test]
async fn cubecl_cpu_acceptance() {
    exercise(Some(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
    ))
    .await;
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[tokio::test]
async fn native_metal_acceptance() {
    exercise(Some(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    ))
    .await;
}
#[test]
fn independent_expected_values_and_schema_rejections() {
    use arrow_array::{Int32Array, Int64Array, RecordBatch};
    assert_eq!(Workload::parse("project").unwrap(), Workload::Project);
    assert!(Workload::parse("invalid").is_err());
    let f = Fixture::new(Case {
        rows: 4,
        pass: 50,
        nulls: 0,
        batches: 1,
        partitions: 1,
    })
    .unwrap();
    let p = f.expected(Workload::Project).unwrap();
    assert_eq!(
        p.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .as_ref(),
        &[0, 1, 3]
    );
    let a = f.expected(Workload::Aggregate).unwrap();
    assert_eq!(
        a.column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        -2876
    );
    assert_eq!(
        a.column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        3
    );
    let wrong = RecordBatch::try_new(
        a.schema(),
        vec![
            Arc::new(Int64Array::from(vec![None::<i64>])),
            a.column(1).clone(),
            a.column(2).clone(),
            a.column(3).clone(),
            a.column(4).clone(),
        ],
    )
    .unwrap();
    assert!(compare(&a.schema(), &[wrong], &a, false).is_err());
    assert!(compare(&p.schema(), std::slice::from_ref(&a), &a, false).is_err());
    assert!(
        Fixture::new(Case {
            rows: 1,
            pass: 101,
            nulls: 0,
            batches: 1,
            partitions: 1
        })
        .is_err()
    );
}
