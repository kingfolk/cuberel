#![cfg(feature = "datafusion")]
#[path = "../examples/support/sql_acceptance.rs"]
mod support;

use cuberel::datafusion::{
    AdapterOptions, BatchPolicy, BatchPolicyError, placement_report, register,
};
#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cuberel::datafusion::{CubeAggregateExec, CubeExec, SharedResources};
#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use datafusion::physical_plan::{ExecutionPlan, displayable};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    physical_plan::collect,
    prelude::{SessionConfig, SessionContext},
};
#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use std::sync::Arc;
use support::{Case, Fixture, Workload, compare};

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn sql_context(fixture: &Fixture, options: AdapterOptions, batch_size: usize) -> SessionContext {
    let ctx = SessionContext::new_with_state(
        register(
            SessionStateBuilder::new_with_default_features().with_config(
                SessionConfig::new()
                    .with_target_partitions(fixture.case.partitions)
                    .with_batch_size(batch_size),
            ),
            options,
        )
        .build(),
    );
    fixture.register(&ctx).unwrap();
    ctx
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn runtime_node(plan: &Arc<dyn ExecutionPlan>, aggregate: bool) -> Option<Arc<dyn ExecutionPlan>> {
    if (aggregate && plan.is::<CubeAggregateExec>()) || (!aggregate && plan.is::<CubeExec>()) {
        return Some(plan.clone());
    }
    plan.children()
        .into_iter()
        .find_map(|child| runtime_node(child, aggregate))
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    plan.metrics()
        .map(|metrics| {
            metrics
                .iter()
                .filter(|metric| metric.value().name() == name)
                .map(|metric| metric.value().as_usize())
                .sum()
        })
        .unwrap_or(0)
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn max_metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    let own = plan
        .metrics()
        .map(|metrics| {
            metrics
                .iter()
                .filter(|metric| metric.value().name() == name)
                .map(|metric| metric.value().as_usize())
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0);
    plan.children()
        .into_iter()
        .map(|child| max_metric(child, name))
        .fold(own, usize::max)
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn assert_drained(resources: &SharedResources) {
    let snapshot = resources.snapshot();
    assert_eq!(
        (
            snapshot.active_requests,
            snapshot.active_streams,
            snapshot.device_bytes,
            snapshot.host_bytes,
        ),
        (0, 0, 0, 0),
        "resources must drain: {snapshot:?}"
    );
    assert!(!snapshot.poisoned);
    assert_eq!(snapshot.rejected_reservations, 0);
}

#[test]
fn batch_policy_rejects_invalid_limits() {
    for (source_rows, max_request_rows) in [(0, 1), (1, 0), (17, 16)] {
        assert!(matches!(
            BatchPolicy::new(source_rows, max_request_rows),
            Err(BatchPolicyError::InvalidSizes { .. })
        ));
    }
    if let Ok(too_large) = usize::try_from(u64::from(u32::MAX) + 1) {
        assert!(matches!(
            BatchPolicy::new(1, too_large),
            Err(BatchPolicyError::InvalidSizes { .. })
        ));
    }
    let valid = BatchPolicy::new(8, 16).unwrap();
    assert_eq!(valid.source_rows(), 8);
    assert_eq!(valid.max_request_rows(), 16);
}

#[tokio::test]
async fn policy_registration_changes_only_batch_size_and_stays_planning_only() {
    let policy = BatchPolicy::new(8, 16).unwrap();
    let original_config = SessionConfig::new()
        .with_target_partitions(3)
        .with_batch_size(8192)
        .with_coalesce_batches(false)
        .with_repartition_joins(false);

    let mut registered = register(
        SessionStateBuilder::new_with_default_features().with_config(original_config.clone()),
        AdapterOptions::planning_only().with_batch_policy(policy),
    );
    let configured = registered.config().as_ref().unwrap();
    assert_eq!(configured.target_partitions(), 3);
    assert_eq!(configured.batch_size(), 8);
    assert!(!configured.options().execution.coalesce_batches);
    assert!(!configured.repartition_joins());

    let fixture = Fixture::new(Case {
        rows: 65,
        pass: 50,
        nulls: 20,
        batches: 1,
        partitions: 3,
    })
    .unwrap();
    let ctx = SessionContext::new_with_state(registered.build());
    fixture.register(&ctx).unwrap();
    let plan = ctx
        .sql(&Workload::parse("aggregate").unwrap().sql(50))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let report = placement_report(
        plan.clone(),
        &AdapterOptions::planning_only().with_batch_policy(policy),
    )
    .unwrap();
    assert_eq!(report.device, None);
    assert_eq!(report.batch_policy, Some(policy));
    let workload = Workload::parse("aggregate").unwrap();
    let expected = fixture.expected(workload).unwrap();
    let schema = plan.schema();
    let output = collect(plan, ctx.task_ctx()).await.unwrap();
    compare(&schema, &output, &expected, false).unwrap();

    let mut disabled = register(
        SessionStateBuilder::new_with_default_features().with_config(original_config),
        AdapterOptions::default().with_batch_policy(policy),
    );
    let disabled_config = disabled.config().as_ref().unwrap();
    assert_eq!(disabled_config.target_partitions(), 3);
    assert_eq!(disabled_config.batch_size(), 8192);
    assert!(!disabled_config.options().execution.coalesce_batches);
    assert!(!disabled_config.repartition_joins());

    let disabled_ctx = SessionContext::new_with_state(disabled.build());
    fixture.register(&disabled_ctx).unwrap();
    let disabled_plan = disabled_ctx
        .sql(&Workload::parse("aggregate").unwrap().sql(50))
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let disabled_report = placement_report(
        disabled_plan,
        &AdapterOptions::default().with_batch_policy(policy),
    )
    .unwrap();
    assert_eq!(disabled_report.batch_policy, None);
    assert_eq!(disabled_report.device, None);
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
async fn exercise_bounded_workloads(executor: cuberel::Executor) {
    let executor = executor.with_aggregate_readback_mode(cuberel::AggregateReadbackMode::Batched);
    let policy = BatchPolicy::new(8, 16).unwrap();
    for workload in [
        Workload::parse("aggregate").unwrap(),
        Workload::parse("project").unwrap(),
    ] {
        let fixture = Fixture::new(Case {
            rows: 65,
            pass: 50,
            nulls: 20,
            batches: 1,
            partitions: 1,
        })
        .unwrap();
        let resources = SharedResources::default();
        let options = AdapterOptions::with_resources(executor.clone(), resources.clone())
            .with_batch_policy(policy);
        let ctx = sql_context(&fixture, options.clone(), 8192);
        let plan = ctx
            .sql(&workload.sql(fixture.case.pass))
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let aggregate = workload == Workload::Aggregate;
        let node = runtime_node(&plan, aggregate).expect("policy must retain device placement");
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(text.contains("source_rows: 8"));
        assert!(text.contains("max_request_rows: 16"));
        let report = placement_report(plan.clone(), &options).unwrap();
        assert_eq!(report.batch_policy, Some(policy));

        let output_schema = plan.schema();
        let output = collect(plan, ctx.task_ctx()).await.unwrap();
        let expected = fixture.expected(workload).unwrap();
        compare(
            &output_schema,
            &output,
            &expected,
            workload == Workload::Project,
        )
        .unwrap();
        assert!(metric(&node, "requests_completed") > 0);
        let actual_max_rows = max_metric(&node, "max_input_batch_rows");
        assert!(actual_max_rows > 0);
        assert!(
            actual_max_rows <= 8,
            "session source target should keep runtime input batches at most 8; observed {actual_max_rows}"
        );
        drop(output);
        assert_drained(&resources);
    }
}

#[cfg(feature = "cpu")]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_batch_policy_bounds_aggregate_and_project() {
    exercise_bounded_workloads(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
    )
    .await;
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[tokio::test(flavor = "multi_thread")]
async fn metal_batch_policy_bounds_aggregate_and_project() {
    exercise_bounded_workloads(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    )
    .await;
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
async fn exercise_cap_rejection_and_clone_isolation(executor: cuberel::Executor) {
    let executor = executor.with_aggregate_readback_mode(cuberel::AggregateReadbackMode::Batched);
    let policy = BatchPolicy::new(8, 16).unwrap();
    for workload in [
        Workload::parse("aggregate").unwrap(),
        Workload::parse("project").unwrap(),
    ] {
        let oversized = Fixture::new(Case {
            rows: 65,
            pass: 50,
            nulls: 20,
            batches: 1,
            partitions: 1,
        })
        .unwrap();
        // This plan retains the original 8192-row DataFusion source setting and
        // its single 65-row source batch. The adapter rule is applied afterward.
        let original_ctx = sql_context(&oversized, AdapterOptions::default(), 8192);
        let original_plan = original_ctx
            .sql(&workload.sql(oversized.case.pass))
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();

        let resources = SharedResources::default();
        let base_options = AdapterOptions::with_resources(executor.clone(), resources.clone());
        let capped_options = base_options.clone().with_batch_policy(policy);
        let state = register(
            SessionStateBuilder::new_with_default_features(),
            capped_options.clone(),
        )
        .build();
        let rule = state
            .physical_optimizers()
            .iter()
            .find(|rule| rule.name() == "cuberel")
            .expect("registered Cuberel rule");
        let capped_plan = rule
            .optimize(original_plan.clone(), state.config_options())
            .unwrap();
        let before = executor.stats();
        let error = collect(capped_plan, original_ctx.task_ctx())
            .await
            .expect_err("65-row request must exceed the explicit cap");
        assert!(error.to_string().contains("max_request_rows"), "{error}");
        assert_eq!(
            before,
            executor.stats(),
            "rejection must precede device work"
        );
        assert_drained(&resources);

        // The same policy engine remains usable after a rejected request.
        let small = Fixture::new(Case {
            rows: 8,
            pass: 50,
            nulls: 20,
            batches: 1,
            partitions: 1,
        })
        .unwrap();
        let small_ctx = sql_context(&small, capped_options.clone(), 8192);
        let small_plan = small_ctx
            .sql(&workload.sql(small.case.pass))
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let small_output = collect(small_plan.clone(), small_ctx.task_ctx())
            .await
            .expect("capped engine should be reusable");
        compare(
            &small_plan.schema(),
            &small_output,
            &small.expected(workload).unwrap(),
            workload == Workload::Project,
        )
        .unwrap();
        drop(small_output);
        assert_drained(&resources);

        // with_batch_policy must copy-on-write the engine: the original options
        // clone has no cap and can still execute the original 65-row plan.
        let uncapped_state = register(
            SessionStateBuilder::new_with_default_features(),
            base_options,
        )
        .build();
        let uncapped_rule = uncapped_state
            .physical_optimizers()
            .iter()
            .find(|rule| rule.name() == "cuberel")
            .unwrap();
        let uncapped_plan = uncapped_rule
            .optimize(original_plan, uncapped_state.config_options())
            .unwrap();
        let uncapped_output = collect(uncapped_plan.clone(), original_ctx.task_ctx())
            .await
            .expect("uncapped options clone must remain uncapped");
        compare(
            &uncapped_plan.schema(),
            &uncapped_output,
            &oversized.expected(workload).unwrap(),
            workload == Workload::Project,
        )
        .unwrap();
        drop(uncapped_output);
        assert_drained(&resources);
    }
}

#[cfg(feature = "cpu")]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_cap_rejection_precedes_reservation_and_options_clones_are_isolated() {
    exercise_cap_rejection_and_clone_isolation(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
    )
    .await;
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[tokio::test(flavor = "multi_thread")]
async fn metal_cap_rejection_precedes_reservation_and_options_clones_are_isolated() {
    exercise_cap_rejection_and_clone_isolation(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    )
    .await;
}
