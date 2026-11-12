#![cfg(all(feature = "datafusion", any(feature = "cpu", feature = "metal")))]
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use cuberel::{
    Executor,
    datafusion::{AdapterOptions, CubeExec, inspect_fragment, register},
};
use datafusion::{
    common::ScalarValue,
    execution::session_state::SessionStateBuilder,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, Column, IsNullExpr, Literal},
    },
    physical_plan::{
        ExecutionPlan, collect, displayable,
        filter::{FilterExec, FilterExecBuilder},
        projection::ProjectionExec,
    },
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;

fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name, index))
}
fn lit(value: i32) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(ScalarValue::Int32(Some(value))))
}
fn bin(a: Arc<dyn PhysicalExpr>, op: Operator, b: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
    Arc::new(BinaryExpr::new(a, op, b))
}
fn project(
    input: Arc<dyn ExecutionPlan>,
    exprs: Vec<(Arc<dyn PhysicalExpr>, &str)>,
) -> Arc<dyn ExecutionPlan> {
    Arc::new(
        ProjectionExec::try_new(
            exprs
                .into_iter()
                .map(|(e, n)| (e, n.to_string()))
                .collect::<Vec<_>>(),
            input,
        )
        .unwrap(),
    )
}
fn filter(
    input: Arc<dyn ExecutionPlan>,
    predicate: Arc<dyn PhysicalExpr>,
) -> Arc<dyn ExecutionPlan> {
    Arc::new(FilterExec::try_new(predicate, input).unwrap())
}
fn count(plan: &Arc<dyn ExecutionPlan>) -> usize {
    usize::from(plan.is::<CubeExec>()) + plan.children().into_iter().map(count).sum::<usize>()
}
fn rewrite(plan: Arc<dyn ExecutionPlan>, executor: &Executor) -> Arc<dyn ExecutionPlan> {
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
    let rewritten = rule.optimize(plan, state.config_options()).unwrap();
    let again = rule
        .optimize(rewritten.clone(), state.config_options())
        .unwrap();
    assert_eq!(
        displayable(again.as_ref()).indent(true).to_string(),
        displayable(rewritten.as_ref()).indent(true).to_string(),
        "rewrite must be idempotent"
    );
    state
        .physical_optimizers()
        .last()
        .unwrap()
        .optimize(rewritten, state.config_options())
        .unwrap()
}
fn rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    batches
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .map(|i| {
                    b.columns()
                        .iter()
                        .map(|a| {
                            datafusion::arrow::util::display::array_value_to_string(a.as_ref(), i)
                                .unwrap()
                        })
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
async fn source(values: Vec<Option<i32>>, extra: bool) -> Arc<dyn ExecutionPlan> {
    let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let mut fields = vec![Field::new("x", DataType::Int32, true)];
    let mut arrays: Vec<arrow_array::ArrayRef> = vec![Arc::new(Int32Array::from(values.clone()))];
    if extra {
        fields.push(Field::new("y", DataType::Int32, false));
        fields.push(Field::new("unused", DataType::Utf8, false));
        arrays.push(Arc::new(Int32Array::from(
            (1..=values.len() as i32).collect::<Vec<_>>(),
        )));
        arrays.push(Arc::new(StringArray::from(vec!["host only"; values.len()])));
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let split = batch.num_rows() / 2;
    let parts = vec![vec![
        batch.slice(0, 0),
        batch.slice(0, split),
        batch.slice(split, batch.num_rows() - split),
    ]];
    ctx.register_table(
        "t",
        Arc::new(datafusion::datasource::MemTable::try_new(batch.schema(), parts).unwrap()),
    )
    .unwrap();
    ctx.sql("SELECT * FROM t")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap()
}
async fn compare(
    original: Arc<dyn ExecutionPlan>,
    optimized: Arc<dyn ExecutionPlan>,
) -> Vec<Vec<String>> {
    let ctx = SessionContext::new();
    let expected = collect(original, ctx.task_ctx()).await.unwrap();
    let actual = collect(optimized, ctx.task_ctx()).await.unwrap();
    assert_eq!(rows(&actual), rows(&expected));
    if let (Some(a), Some(e)) = (actual.first(), expected.first()) {
        assert_eq!(a.schema(), e.schema());
    }
    rows(&actual)
}

async fn chains(executor: &Executor) {
    let base = source(vec![Some(10), None, Some(30), Some(40), Some(100)], true).await;
    let lower = filter(base, bin(col("x", 0), Operator::GtEq, lit(30)));
    let rename = project(lower, vec![(col("y", 1), "k"), (col("x", 0), "renamed")]);
    let upper = filter(rename, bin(col("renamed", 1), Operator::Lt, lit(50)));
    let plan = project(
        upper,
        vec![
            (col("renamed", 1), "value"),
            (Arc::new(IsNullExpr::new(col("renamed", 1))), "missing"),
            (col("k", 0), "key"),
        ],
    );
    let before = executor.stats();
    let inspection = inspect_fragment(plan.clone()).unwrap();
    assert_eq!(inspection.physical_nodes, 4);
    assert_eq!(inspection.filter_count, 2);
    assert_eq!(inspection.input_columns, vec![0, 1]);
    assert_eq!(inspection.programs.len(), 5);
    assert!(inspection.normalized_expressions[1].contains("x@0"));
    let device = rewrite(plan.clone(), executor);
    assert_eq!(
        executor.stats(),
        before,
        "discovery is device-side-effect free"
    );
    assert_eq!(count(&device), 1);
    let text = displayable(device.as_ref()).indent(true).to_string();
    assert!(
        text.contains("filters=2") && text.contains("nodes=4"),
        "{text}"
    );
    assert_eq!(
        compare(plan, device).await,
        vec![vec!["30", "false", "3"], vec!["40", "false", "4"]]
    );
    assert_eq!(
        executor.stats().upload_bytes - before.upload_bytes,
        5 * 8 * 2,
        "referenced columns upload once, text stays on host"
    );

    // FilterExec's own output projection changes the next predicate's positions.
    let base = source(vec![Some(10), None, Some(30), Some(40), Some(100)], true).await;
    let projected: Arc<dyn ExecutionPlan> = Arc::new(
        FilterExecBuilder::new(bin(col("x", 0), Operator::GtEq, lit(30)), base)
            .apply_projection(Some(vec![1, 0]))
            .unwrap()
            .build()
            .unwrap(),
    );
    let plan = project(
        filter(projected, bin(col("x", 1), Operator::Lt, lit(50))),
        vec![(col("x", 1), "value"), (col("y", 0), "key")],
    );
    let device = rewrite(plan.clone(), executor);
    assert_eq!(count(&device), 1);
    assert_eq!(
        compare(plan, device).await,
        vec![vec!["30", "3"], vec!["40", "4"]]
    );
}

async fn cpu_boundaries(executor: &Executor) {
    let base = source(
        vec![Some(10), None, Some(30), Some(40), Some(i32::MAX)],
        false,
    )
    .await;
    let lower = filter(base, bin(col("x", 0), Operator::GtEq, lit(30)));
    // Default wrapping arithmetic is not supported by Cuberel. Device execution
    // must resume below and above this CPU computation without changing overflow.
    let cpu = project(lower, vec![(bin(col("x", 0), Operator::Plus, lit(1)), "y")]);
    let upper = project(
        filter(cpu, bin(col("y", 0), Operator::Lt, lit(50))),
        vec![(col("y", 0), "result")],
    );
    let device = rewrite(upper.clone(), executor);
    assert_eq!(
        count(&device),
        2,
        "{}",
        displayable(device.as_ref()).indent(true)
    );
    let actual = compare(upper, device).await;
    assert_eq!(actual, vec![vec!["31"], vec!["41"], vec!["-2147483648"]]);

    // CPU output alone can consume a lower device region.
    let base = source(vec![Some(30), Some(i32::MAX)], false).await;
    let cpu = project(
        filter(base, bin(col("x", 0), Operator::GtEq, lit(30))),
        vec![(bin(col("x", 0), Operator::Plus, lit(1)), "y")],
    );
    let device = rewrite(cpu.clone(), executor);
    assert_eq!(count(&device), 1);
    compare(cpu, device).await;
}

async fn errors_and_limits(executor: &Executor) {
    let checked: Arc<dyn PhysicalExpr> =
        Arc::new(BinaryExpr::new(col("x", 0), Operator::Plus, lit(1)).with_fail_on_overflow(true));
    let base = source(vec![Some(i32::MAX), Some(1)], false).await;
    let plan = filter(
        filter(base.clone(), bin(col("x", 0), Operator::Lt, lit(10))),
        bin(checked.clone(), Operator::Gt, lit(0)),
    );
    let device = rewrite(plan.clone(), executor);
    assert_eq!(count(&device), 1);
    assert_eq!(compare(plan, device).await, vec![vec!["1"]]);
    let plan = filter(
        filter(base.clone(), bin(checked.clone(), Operator::Gt, lit(0))),
        bin(col("x", 0), Operator::Lt, lit(10)),
    );
    let device = rewrite(plan.clone(), executor);
    let ctx = SessionContext::new();
    assert!(collect(plan, ctx.task_ctx()).await.is_err());
    assert!(
        collect(device, ctx.task_ctx()).await.is_err(),
        "later filter cannot hide an earlier overflow"
    );
    let intermediate = project(base, vec![(col("x", 0), "x"), (checked, "unused_checked")]);
    let plan = project(
        filter(intermediate, bin(col("x", 0), Operator::Lt, lit(10))),
        vec![(col("x", 0), "x")],
    );
    let device = rewrite(plan.clone(), executor);
    assert!(collect(plan, ctx.task_ctx()).await.is_err());
    assert!(
        collect(device, ctx.task_ctx()).await.is_err(),
        "cannot drop or delay an unused checked projection"
    );

    let mut plan = source(vec![Some(1), None, Some(3)], false).await;
    for i in 0..17 {
        plan = filter(plan, bin(col("x", 0), Operator::GtEq, lit(-i)));
    }
    let device = rewrite(plan.clone(), executor);
    assert_eq!(count(&device), 2);
    assert!(
        displayable(device.as_ref())
            .indent(true)
            .to_string()
            .contains("fragment_limit")
    );
    compare(plan, device).await;
}

async fn sql(executor: &Executor) {
    let options = AdapterOptions::with_executor(executor.clone());
    let gpu = SessionContext::new_with_state(
        register(
            SessionStateBuilder::new_with_default_features()
                .with_config(SessionConfig::new().with_target_partitions(1)),
            options,
        )
        .build(),
    );
    let cpu = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
        ]))],
    )
    .unwrap();
    gpu.register_batch("t", batch.clone()).unwrap();
    cpu.register_batch("t", batch).unwrap();
    for query in [
        "SELECT v AS result FROM (SELECT x AS v FROM t WHERE x >= CAST(30 AS INT)) q WHERE v < CAST(50 AS INT)",
        "SELECT x + CAST(1 AS INT) AS result FROM t WHERE x >= CAST(30 AS INT)",
        "SELECT x FROM t WHERE x > CAST(10 AS INT) ORDER BY x DESC LIMIT 1",
    ] {
        let plan = gpu
            .sql(query)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        assert!(count(&plan) > 0);
        let actual = collect(plan, gpu.task_ctx()).await.unwrap();
        let expected = cpu.sql(query).await.unwrap().collect().await.unwrap();
        assert_eq!(rows(&actual), rows(&expected));
    }
}
async fn exercise(executor: Executor) {
    chains(&executor).await;
    cpu_boundaries(&executor).await;
    errors_and_limits(&executor).await;
    sql(&executor).await;
    barriers_and_empty(&executor).await;
}

async fn barriers_and_empty(executor: &Executor) {
    use datafusion::physical_plan::{Partitioning, repartition::RepartitionExec};
    let base = source(vec![Some(10), None, Some(30), Some(40)], false).await;
    let lower = filter(base, bin(col("x", 0), Operator::GtEq, lit(30)));
    let exchange: Arc<dyn ExecutionPlan> =
        Arc::new(RepartitionExec::try_new(lower, Partitioning::RoundRobinBatch(2)).unwrap());
    let plan = project(exchange, vec![(col("x", 0), "result")]);
    let device = rewrite(plan.clone(), executor);
    assert_eq!(count(&device), 2);
    let text = displayable(device.as_ref()).indent(true).to_string();
    assert!(text.contains("partition_boundary") && text.contains("RepartitionExec"));
    let ctx = SessionContext::new();
    let mut expected = rows(&collect(plan, ctx.task_ctx()).await.unwrap());
    let mut actual = rows(&collect(device, ctx.task_ctx()).await.unwrap());
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected);

    let base = source(vec![Some(10), None, Some(30)], false).await;
    let fetch: Arc<dyn ExecutionPlan> = Arc::new(
        FilterExecBuilder::new(bin(col("x", 0), Operator::GtEq, lit(10)), base)
            .with_fetch(Some(1))
            .build()
            .unwrap(),
    );
    assert_eq!(
        inspect_fragment(fetch.clone()).unwrap_err().code,
        "fetch_boundary"
    );
    let device = rewrite(fetch.clone(), executor);
    assert_eq!(count(&device), 0);
    compare(fetch, device).await;

    let base = source(vec![Some(10), None, Some(30)], false).await;
    let wide = project(
        base,
        (0..33)
            .map(|i| (col("x", 0), format!("out{i}")))
            .collect::<Vec<_>>()
            .iter()
            .map(|(e, n)| (e.clone(), n.as_str()))
            .collect(),
    );
    assert_eq!(
        inspect_fragment(wide.clone()).unwrap_err().code,
        "fragment_limit"
    );
    let device = rewrite(wide.clone(), executor);
    assert_eq!(count(&device), 0);
    compare(wide, device).await;

    for values in [vec![], vec![None, None], vec![Some(1)], vec![Some(30)]] {
        let base = source(values, false).await;
        let plan = filter(
            filter(base, bin(col("x", 0), Operator::GtEq, lit(30))),
            bin(col("x", 0), Operator::Lt, lit(50)),
        );
        let device = rewrite(plan.clone(), executor);
        assert_eq!(count(&device), 1);
        compare(plan, device).await;
    }
}
#[cfg(feature = "cpu")]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_contiguous_fragments() {
    exercise(Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap()).await;
}
#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_contiguous_fragments() {
    exercise(
        Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
    )
    .await;
}
