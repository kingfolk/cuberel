#![cfg(feature = "datafusion")]
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::datafusion::inspect_global_aggregate;
use datafusion::{
    datasource::MemTable,
    physical_plan::{ExecutionPlan, aggregates::AggregateExec},
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;

fn batch(values: Vec<Option<i32>>) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(values))],
    )
    .unwrap()
}
async fn plan(
    parts: Vec<Vec<RecordBatch>>,
    sql: &str,
    partitions: usize,
) -> (SessionContext, Arc<dyn ExecutionPlan>) {
    let ctx =
        SessionContext::new_with_config(SessionConfig::new().with_target_partitions(partitions));
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(batch(vec![]).schema(), parts).unwrap()),
    )
    .unwrap();
    let p = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    (ctx, p)
}
fn aggregate(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.is::<AggregateExec>() {
        return plan.clone();
    }
    for c in plan.children() {
        if c.is::<AggregateExec>() || !c.children().is_empty() {
            return aggregate(c);
        }
    }
    panic!("missing aggregate: {plan:?}")
}

#[tokio::test]
async fn global_placement_contract_without_runtime() {
    for partitions in [1, 2] {
        let (_, p) = plan(
            vec![vec![batch(vec![Some(1), None])], vec![batch(vec![Some(3)])]],
            "SELECT count(*), count(x), min(x), max(x) FROM t WHERE x > CAST(0 AS INT)",
            partitions,
        )
        .await;
        let p = aggregate(&p);
        let info = inspect_global_aggregate(p.clone()).unwrap_or_else(|e| panic!("{e}: {p:?}"));
        assert_eq!(info.input_upper_bound, 3);
        assert_eq!(info.aggregates, 4);
        // Intermediate accumulator states must never be mistaken for raw rows.
        if let Some(a) = p.downcast_ref::<AggregateExec>() {
            let single = datafusion::physical_plan::aggregates::AggregateExec::try_new(
                datafusion::physical_plan::aggregates::AggregateMode::Single,
                a.group_expr().clone(),
                a.aggr_expr().to_vec(),
                a.filter_expr().to_vec(),
                Arc::new(datafusion::physical_plan::empty::EmptyExec::new(
                    a.input_schema(),
                )),
                a.input_schema(),
            )
            .unwrap();
            assert_eq!(
                inspect_global_aggregate(Arc::new(single)).unwrap_err().code,
                "unknown_aggregate_bound"
            );
        }
    }
    for sql in [
        "SELECT min(x) FROM t GROUP BY x",
        "SELECT count(DISTINCT x), min(x) FROM t",
        "SELECT min(x) FILTER (WHERE x > 0) FROM t",
    ] {
        let (_, p) = plan(vec![vec![batch(vec![Some(1), None])]], sql, 1).await;
        assert!(inspect_global_aggregate(aggregate(&p)).is_err());
    }
}

#[cfg(any(feature = "cpu", feature = "metal"))]
mod runtime {
    use super::{aggregate, batch, plan};
    use arrow_array::RecordBatch;
    use cuberel::{
        Executor, ResourceBudget,
        datafusion::{AdapterOptions, CubeAggregateExec, register},
    };
    use datafusion::{
        execution::session_state::SessionStateBuilder,
        physical_plan::{ExecutionPlan, collect, displayable},
    };
    use futures::TryStreamExt;
    use std::{sync::Arc, time::Duration};

    fn rewrite(p: Arc<dyn ExecutionPlan>, executor: &Executor) -> Arc<dyn ExecutionPlan> {
        let state = register(
            SessionStateBuilder::new_with_default_features(),
            AdapterOptions::with_executor(executor.clone()),
        )
        .build();
        let rule = state
            .physical_optimizers()
            .iter()
            .find(|r| r.name() == "cuberel")
            .unwrap();
        let p = rule.optimize(p, state.config_options()).unwrap();
        let again = rule.optimize(p.clone(), state.config_options()).unwrap();
        assert_eq!(
            displayable(p.as_ref()).indent(true).to_string(),
            displayable(again.as_ref()).indent(true).to_string()
        );
        state
            .physical_optimizers()
            .last()
            .unwrap()
            .optimize(p, state.config_options())
            .unwrap()
    }
    fn count(p: &Arc<dyn ExecutionPlan>) -> usize {
        usize::from(p.is::<CubeAggregateExec>())
            + p.children().into_iter().map(count).sum::<usize>()
    }
    fn has_aggregate(p: &Arc<dyn ExecutionPlan>) -> bool {
        p.is::<datafusion::physical_plan::aggregates::AggregateExec>()
            || p.children().into_iter().any(has_aggregate)
    }
    fn values(batches: &[RecordBatch]) -> Vec<String> {
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);
        batches[0]
            .columns()
            .iter()
            .map(|a| {
                datafusion::arrow::util::display::array_value_to_string(a.as_ref(), 0).unwrap()
            })
            .collect()
    }
    async fn exercise(executor: Executor) {
        let values_input = batch(vec![
            Some(-5),
            Some(10),
            None,
            Some(30),
            Some(40),
            Some(100),
        ]);
        for partitions in [1, 2] {
            for parts in [
                vec![
                    vec![values_input.slice(0, 0), values_input.slice(0, 3)],
                    vec![values_input.slice(3, 3)],
                ],
                vec![vec![], vec![]],
                vec![vec![batch(vec![])], vec![batch(vec![None, None])]],
            ] {
                for suffix in [
                    "",
                    " WHERE x >= CAST(30 AS INT)",
                    " WHERE x > CAST(200 AS INT)",
                ] {
                    let sql = format!(
                        "SELECT count(*) AS n, count(x) AS valid, min(x) AS lo, max(x) AS hi FROM t{suffix}"
                    );
                    let (ctx, p) = plan(parts.clone(), &sql, partitions).await;
                    let stats = executor.stats();
                    let device = rewrite(p.clone(), &executor);
                    assert_eq!(executor.stats(), stats, "planning cannot submit");
                    assert_eq!(
                        count(&device),
                        usize::from(has_aggregate(&p)),
                        "{}",
                        displayable(device.as_ref()).indent(true)
                    );
                    let expected = collect(p, ctx.task_ctx()).await.unwrap();
                    let actual = collect(device, ctx.task_ctx()).await.unwrap();
                    assert_eq!(actual[0].schema(), expected[0].schema());
                    assert_eq!(values(&actual), values(&expected));
                }
            }
        }
        // DataFusion can fold an empty SQL aggregate to constants before our
        // rule. Explicit real physical fixtures also exercise device EOF/NULL.
        let (ctx, template) = plan(
            vec![vec![batch(vec![Some(1)])]],
            "SELECT count(*),count(x),min(x),max(x) FROM t",
            1,
        )
        .await;
        let template = aggregate(&template);
        let a = template
            .downcast_ref::<datafusion::physical_plan::aggregates::AggregateExec>()
            .unwrap();
        for batches in [vec![], vec![batch(vec![])], vec![batch(vec![None, None])]] {
            let input = datafusion::datasource::memory::MemorySourceConfig::try_new_exec(
                &[batches],
                batch(vec![]).schema(),
                None,
            )
            .unwrap();
            let original: Arc<dyn ExecutionPlan> = Arc::new(
                datafusion::physical_plan::aggregates::AggregateExec::try_new(
                    datafusion::physical_plan::aggregates::AggregateMode::Single,
                    a.group_expr().clone(),
                    a.aggr_expr().to_vec(),
                    a.filter_expr().to_vec(),
                    input.clone(),
                    input.schema(),
                )
                .unwrap(),
            );
            let p = rewrite(original.clone(), &executor);
            assert_eq!(count(&p), 1);
            let expected = collect(original, ctx.task_ctx()).await.unwrap();
            let actual = collect(p, ctx.task_ctx()).await.unwrap();
            assert_eq!(actual[0].schema(), expected[0].schema());
            assert_eq!(values(&actual), values(&expected));
        }
        // Unsupported SQL aggregation remains complete DataFusion aggregation.
        for sql in [
            "SELECT count(DISTINCT x), min(x) FROM t",
            "SELECT min(x) FILTER (WHERE x > 0) FROM t",
            "SELECT min(x), max(x) FROM t GROUP BY x",
        ] {
            let (ctx, p) = plan(vec![vec![values_input.clone()]], sql, 1).await;
            let device = rewrite(p.clone(), &executor);
            assert_eq!(count(&device), 0);
            assert_eq!(
                collect(p, ctx.task_ctx()).await.unwrap(),
                collect(device, ctx.task_ctx()).await.unwrap()
            );
        }
        let (ctx, p) = plan(
            vec![vec![values_input]],
            "SELECT count(*),count(CAST(NULL AS INT)),min(x) FROM t",
            1,
        )
        .await;
        let device = rewrite(p.clone(), &executor);
        assert_eq!(count(&device), 1);
        assert_eq!(
            values(&collect(device.clone(), ctx.task_ctx()).await.unwrap()),
            ["6", "0", "-5"]
        );
        // Constructing/dropping an unpolled aggregate stream performs no upload.
        let before = executor.stats();
        drop(device.execute(0, ctx.task_ctx()).unwrap());
        assert_eq!(before, executor.stats());
        // Same-source/schema child replacement revalidates the complete contract.
        let a = aggregate(&p);
        let direct = rewrite(a, &executor);
        let child = direct.children()[0].clone();
        let replaced = direct
            .replace_children(
                vec![child],
                datafusion::physical_plan::ReplaceChildrenOptions::new(
                    datafusion::physical_plan::ChildrenPropertiesMode::Recompute,
                ),
            )
            .unwrap();
        assert_eq!(
            values(&collect(replaced, ctx.task_ctx()).await.unwrap()),
            ["6", "0", "-5"]
        );
        // Resource rejection occurs before any upload and never publishes a row.
        let limited = executor.clone().with_budget(ResourceBudget {
            max_device_bytes: 0,
            ..ResourceBudget::UNLIMITED
        });
        let p = rewrite(p, &limited);
        let before = limited.stats();
        assert!(
            collect(p, ctx.task_ctx())
                .await
                .unwrap_err()
                .to_string()
                .contains("budget")
        );
        assert_eq!(before.upload_calls, limited.stats().upload_calls);
        // A later checked argument error must not publish a partial aggregate.
        let (failed_ctx, source) = plan(
            vec![vec![batch(vec![Some(1)]), batch(vec![Some(i32::MAX)])]],
            "SELECT * FROM t",
            1,
        )
        .await;
        let checked: Arc<dyn datafusion::physical_expr::PhysicalExpr> = Arc::new(
            datafusion::physical_expr::expressions::BinaryExpr::new(
                Arc::new(datafusion::physical_expr::expressions::Column::new("x", 0)),
                datafusion::logical_expr::Operator::Plus,
                Arc::new(datafusion::physical_expr::expressions::Literal::new(
                    datafusion::common::ScalarValue::Int32(Some(1)),
                )),
            )
            .with_fail_on_overflow(true),
        );
        let arg = datafusion::physical_expr::aggregate::AggregateExprBuilder::new(
            datafusion::functions_aggregate::min_max::min_udaf(),
            vec![checked],
        )
        .schema(source.schema())
        .alias("checked_min")
        .build()
        .unwrap();
        let failed: Arc<dyn ExecutionPlan> = Arc::new(
            datafusion::physical_plan::aggregates::AggregateExec::try_new(
                datafusion::physical_plan::aggregates::AggregateMode::Single,
                datafusion::physical_plan::aggregates::PhysicalGroupBy::new_single(vec![]),
                vec![Arc::new(arg)],
                vec![None],
                source.clone(),
                source.schema(),
            )
            .unwrap(),
        );
        let failed = rewrite(failed, &executor);
        assert_eq!(count(&failed), 1);
        let mut failed_stream = failed.execute(0, failed_ctx.task_ctx()).unwrap();
        assert!(
            failed_stream
                .try_next()
                .await
                .unwrap_err()
                .to_string()
                .contains("overflow")
        );
        assert!(failed_stream.try_next().await.unwrap().is_none());
        // Slow/unpolled output has no producer. Dropping a polled future is safe;
        // a fresh plan remains usable after the bounded worker drains.
        let (ctx, p) = plan(
            vec![vec![batch((0..129).map(Some).collect())]],
            "SELECT min(x),count(x) FROM t",
            1,
        )
        .await;
        let p = rewrite(p, &executor);
        let mut stream = p.execute(0, ctx.task_ctx()).unwrap();
        let _ = tokio::time::timeout(Duration::from_nanos(1), stream.try_next()).await;
        drop(stream);
        let p = datafusion::physical_plan::execution_plan::reset_plan_states(p).unwrap();
        assert_eq!(
            values(
                &tokio::time::timeout(Duration::from_secs(30), collect(p, ctx.task_ctx()))
                    .await
                    .unwrap()
                    .unwrap()
            ),
            ["0", "129"]
        );
    }
    #[cfg(feature = "cpu")]
    #[tokio::test(flavor = "multi_thread")]
    async fn cpu_global_count_min_max() {
        exercise(Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap())
            .await;
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_global_count_min_max() {
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
