#![cfg(feature = "datafusion")]
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::datafusion::{AdapterOptions, Placement, PlacementMode, placement_report, register};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    physical_plan::ExecutionPlan,
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;

async fn source() -> SessionContext {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let b = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
        ]))],
    )
    .unwrap();
    ctx.register_batch("t", b).unwrap();
    ctx
}
fn rewrite(
    plan: Arc<dyn ExecutionPlan>,
    options: AdapterOptions,
) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
    let state = register(SessionStateBuilder::new_with_default_features(), options).build();
    let plan = state
        .physical_optimizers()
        .iter()
        .find(|r| r.name() == "cuberel")
        .unwrap()
        .optimize(plan, state.config_options())?;
    state
        .physical_optimizers()
        .last()
        .unwrap()
        .optimize(plan, state.config_options())
}
#[tokio::test]
async fn placement_explains_disabled_and_semantic_rejection_without_runtime() {
    let ctx = source().await;
    let p = ctx
        .sql("SELECT x AS value FROM t WHERE x >= CAST(30 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let options = AdapterOptions::planning_only();
    let report = placement_report(p.clone(), &options).unwrap();
    assert_eq!(report.decisions.len(), 1);
    let d = &report.decisions[0];
    assert_eq!(d.fragment_id, "root");
    assert_eq!(d.reason.code, "disabled");
    assert_eq!(d.placement, Placement::Cpu);
    assert_eq!(d.input_rows_upper_bound, Some(4));
    assert!(d.estimated_device_bytes.is_none());
    assert!(report.device.is_none());
    assert!(
        rewrite(
            p,
            options.with_placement_mode(PlacementMode::RequireCandidates)
        )
        .unwrap_err()
        .to_string()
        .contains("disabled")
    );
    let p = ctx
        .sql("SELECT x + CAST(1 AS INT) AS value FROM t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let report = placement_report(p, &AdapterOptions::planning_only()).unwrap();
    assert!(
        report
            .decisions
            .iter()
            .any(|d| d.reason.code == "semantic_mismatch")
    );
}

#[cfg(any(feature = "cpu", feature = "metal"))]
mod runtime {
    use super::{rewrite, source};
    use cuberel::{
        Executor,
        datafusion::{
            AdapterOptions, CubeAggregateExec, CubeExec, Placement, PlacementMode, SharedLimits,
            SharedResources, placement_report,
        },
    };
    use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
    use futures::TryStreamExt;
    use std::sync::Arc;
    fn cube(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        if plan.is::<CubeExec>() || plan.is::<CubeAggregateExec>() {
            return plan.clone();
        }
        cube(plan.children()[0])
    }
    fn metric(p: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        let m = p.metrics().unwrap();
        m.iter()
            .filter(|m| m.value().name() == name)
            .map(|m| m.value().as_usize())
            .sum()
    }
    fn assert_phase_scopes(plan: &Arc<dyn ExecutionPlan>) {
        let adapt = metric(plan, "arrow_adapt");
        let preflight = metric(plan, "resource_preflight");
        let submit = metric(plan, "upload_submit");
        let complete = metric(plan, "complete_export");
        for name in [
            "arrow_adapt",
            "resource_preflight",
            "upload_submit",
            "complete_export",
        ] {
            assert!(
                plan.metrics()
                    .unwrap()
                    .iter()
                    .any(|m| m.value().name() == name)
            );
        }
        assert!(adapt + preflight <= metric(plan, "prepare"));
        assert!(submit + complete <= metric(plan, "execute_export"));
        assert!(adapt + preflight + submit + complete <= metric(plan, "worker"));
    }
    async fn exercise(executor: Executor) {
        let ctx = source().await;
        let sql = "SELECT x AS value,x IS NULL AS missing FROM t WHERE x >= CAST(30 AS INT)";
        let original = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let options = AdapterOptions::with_executor(executor.clone());
        let before = executor.stats();
        let report = placement_report(original.clone(), &options).unwrap();
        assert_eq!(before, executor.stats());
        // Restore the full Partial/Final contract on a real multi-partition plan.
        {
            let multi = datafusion::prelude::SessionContext::new_with_config(
                datafusion::prelude::SessionConfig::new().with_target_partitions(2),
            );
            let b = arrow_array::RecordBatch::try_new(
                Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
                    "x",
                    arrow_schema::DataType::Int32,
                    true,
                )])),
                vec![Arc::new(arrow_array::Int32Array::from(vec![
                    Some(10),
                    None,
                    Some(30),
                    Some(40),
                ]))],
            )
            .unwrap();
            multi
                .register_table(
                    "t",
                    Arc::new(
                        datafusion::datasource::MemTable::try_new(
                            b.schema(),
                            vec![vec![b.slice(0, 2)], vec![b.slice(2, 2)]],
                        )
                        .unwrap(),
                    ),
                )
                .unwrap();
            let original = multi
                .sql("SELECT sum(x),count(*),min(x) FROM t WHERE x>=CAST(30 AS INT)")
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            assert!(
                displayable(original.as_ref())
                    .indent(true)
                    .to_string()
                    .contains("mode=Final")
            );
            let opts = AdapterOptions::with_executor(executor.clone());
            let device = rewrite(original, opts.clone()).unwrap();
            let expected = collect(device.clone(), multi.task_ctx()).await.unwrap();
            assert_phase_scopes(&cube(&device));
            let cpu = rewrite(device, opts.with_placement_mode(PlacementMode::CpuOnly)).unwrap();
            assert!(
                !displayable(cpu.as_ref())
                    .indent(true)
                    .to_string()
                    .contains("Cube")
            );
            let before = executor.stats();
            let actual = collect(
                datafusion::physical_plan::execution_plan::reset_plan_states(cpu).unwrap(),
                multi.task_ctx(),
            )
            .await
            .unwrap();
            assert_eq!(expected, actual);
            assert_eq!(before, executor.stats());
        }
        let before = executor.stats();
        assert_eq!(report.decisions.len(), 1);
        assert_eq!(report.decisions[0].placement, Placement::Device);
        let device = rewrite(
            original.clone(),
            options
                .clone()
                .with_placement_mode(PlacementMode::RequireCandidates),
        )
        .unwrap();
        let text = displayable(device.as_ref()).indent(true).to_string();
        assert!(text.contains("fragment_id=root"));
        assert!(text.contains("per_batch_estimate=runtime"));
        let again = rewrite(device.clone(), options.clone()).unwrap();
        assert_eq!(text, displayable(again.as_ref()).indent(true).to_string());
        let node = cube(&device);
        let defines = executor.define_count();
        let result = collect(device.clone(), ctx.task_ctx()).await.unwrap();
        assert_eq!(result[0].num_rows(), 2);
        let after = executor.stats();
        assert_eq!(metric(&node, "input_rows"), 4);
        assert_eq!(
            metric(&node, "define_calls"),
            executor.define_count() - defines
        );
        assert_eq!(metric(&node, "output_rows"), 2);
        assert_eq!(node.metrics().unwrap().output_rows(), Some(2));
        assert_eq!(metric(&node, "input_batches"), 1);
        assert_eq!(metric(&node, "output_batches"), 1);
        assert_eq!(metric(&node, "streams_completed"), 1);
        assert_eq!(metric(&node, "active_streams"), 0);
        assert_eq!(
            metric(&node, "observed_upload_bytes") as u64,
            after.upload_bytes - before.upload_bytes
        );
        assert_eq!(
            metric(&node, "observed_launches") as u64,
            after.kernel_launches - before.kernel_launches
        );
        assert_phase_scopes(&node);
        assert!(metric(&node, "max_estimated_device_bytes") > 0);
        assert!(metric(&node, "unfused_requests") > 0);
        assert!(
            node.metrics()
                .unwrap()
                .iter()
                .any(|m| m.value().name() == "fusion_fallback")
        );
        for name in [
            "prepare",
            "worker",
            "queue_wait",
            "execute_export",
            "stream_wall",
        ] {
            assert!(
                node.metrics()
                    .unwrap()
                    .iter()
                    .any(|m| m.value().name() == name)
            );
        }
        let registrations = node.metrics().unwrap().iter().count();
        for _ in 0..8 {
            drop(node.execute(0, ctx.task_ctx()).unwrap());
        }
        assert_eq!(
            node.metrics().unwrap().iter().count(),
            registrations,
            "unpolled streams do not grow metric registrations"
        );
        drop(result);
        // Restore CPU operators even if the input plan was previously rewritten.
        let before = executor.stats();
        let cpu = rewrite(
            device,
            options.clone().with_placement_mode(PlacementMode::CpuOnly),
        )
        .unwrap();
        assert!(
            !displayable(cpu.as_ref())
                .indent(true)
                .to_string()
                .contains("CubeExec")
        );
        let result = collect(cpu, ctx.task_ctx()).await.unwrap();
        assert_eq!(result[0].num_rows(), 2);
        assert_eq!(before, executor.stats());
        // Incompatible candidates fail diagnostically; CPU sources/sorts/limits do not.
        let p = ctx
            .sql("SELECT x+CAST(1 AS INT) FROM t")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert!(
            rewrite(
                p,
                options
                    .clone()
                    .with_placement_mode(PlacementMode::RequireCandidates)
            )
            .unwrap_err()
            .to_string()
            .contains("semantic_mismatch")
        );
        let p = ctx
            .sql("SELECT x FROM t WHERE x >= CAST(30 AS INT) ORDER BY x LIMIT 1")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let p = rewrite(
            p,
            options
                .clone()
                .with_placement_mode(PlacementMode::RequireCandidates),
        )
        .unwrap();
        assert_eq!(collect(p, ctx.task_ctx()).await.unwrap()[0].num_rows(), 1);
        // A canceled stream and a rejected request have separate observations.
        let p = rewrite(original.clone(), options.clone()).unwrap();
        let node = cube(&p);
        let mut stream = p.execute(0, ctx.task_ctx()).unwrap();
        let batch = stream.try_next().await.unwrap().unwrap();
        drop(batch);
        drop(stream);
        assert_eq!(metric(&node, "streams_canceled"), 1);
        assert_eq!(metric(&node, "active_streams"), 0);
        let limits = SharedResources::new(SharedLimits {
            max_device_bytes: 0,
            ..SharedLimits::default()
        });
        let p = rewrite(
            original,
            AdapterOptions::with_resources(executor.clone(), limits.clone()),
        )
        .unwrap();
        let node = cube(&p);
        let before = executor.stats();
        assert!(collect(p, ctx.task_ctx()).await.is_err());
        assert_eq!(metric(&node, "streams_failed"), 1);
        assert_eq!(metric(&node, "request_errors"), 1);
        assert_eq!(metric(&node, "observed_upload_bytes"), 0);
        assert!(metric(&node, "max_estimated_device_bytes") > 0);
        assert_eq!(before.upload_calls, executor.stats().upload_calls);
        assert_eq!(limits.snapshot().host_bytes, 0);
        // Global plans expose the same metrics and can restore complete CPU finality.
        let p = ctx
            .sql("SELECT sum(x),count(*),min(x) FROM t WHERE x >= CAST(30 AS INT)")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let p = rewrite(p, options.clone()).unwrap();
        let node = cube(&p);
        let result = collect(p.clone(), ctx.task_ctx()).await.unwrap();
        assert_eq!(result[0].num_rows(), 1);
        assert_eq!(metric(&node, "output_rows"), 1);
        assert_eq!(metric(&node, "input_rows"), 4);
        assert!(metric(&node, "observed_launches") > 0);
        drop(result);
        let before = executor.stats();
        let cpu = rewrite(p, options.with_placement_mode(PlacementMode::CpuOnly)).unwrap();
        assert_eq!(collect(cpu, ctx.task_ctx()).await.unwrap()[0].num_rows(), 1);
        assert_eq!(before, executor.stats());
    }
    #[cfg(feature = "cpu")]
    #[tokio::test(flavor = "multi_thread")]
    async fn cpu_placement_and_metrics() {
        exercise(Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap())
            .await;
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_placement_and_metrics() {
        exercise(
            Executor::with_cube_size(
                cubecl::Device::metal(Default::default()).unwrap().client(),
                64,
            )
            .unwrap(),
        )
        .await;
    }
}
