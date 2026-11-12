#![cfg(all(feature = "datafusion", any(feature = "cpu", feature = "metal")))]
use arrow_array::{Float32Array, Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use cuberel::{
    Executor, ResourceBudget,
    datafusion::{AdapterOptions, CubeExec, register},
};
use datafusion::{
    datasource::MemTable,
    execution::session_state::SessionStateBuilder,
    physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, ReplaceChildrenOptions,
        collect, displayable,
    },
    prelude::SessionContext,
};
use futures::TryStreamExt;
use std::sync::Arc;

fn context(executor: Option<Executor>) -> SessionContext {
    let options = executor
        .map(AdapterOptions::with_executor)
        .unwrap_or_default();
    SessionContext::new_with_state(
        register(SessionStateBuilder::new_with_default_features(), options).build(),
    )
}
fn input(rows: usize) -> RecordBatch {
    let s = Arc::new(
        Schema::new(vec![
            Field::new("x", DataType::Int32, true)
                .with_metadata([("unit".into(), "items".into())].into()),
            Field::new("f", DataType::Float32, true),
        ])
        .with_metadata([("source".into(), "fixture".into())].into()),
    );
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(Int32Array::from(
                (0..rows)
                    .map(|i| (i % 5 != 0).then_some(i as i32 - 3))
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..rows)
                    .map(|i| (i % 3 != 0).then_some(if i % 2 == 0 { -0.0 } else { f32::INFINITY }))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}
fn register_table(ctx: &SessionContext, partitions: Vec<Vec<RecordBatch>>, schema: Arc<Schema>) {
    ctx.register_table(
        "t",
        Arc::new(MemTable::try_new(schema, partitions).unwrap()),
    )
    .unwrap();
}
fn ordered_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    for batch in batches {
        for row in 0..batch.num_rows() {
            rows.push(
                batch
                    .columns()
                    .iter()
                    .map(|a| {
                        datafusion::arrow::util::display::array_value_to_string(a.as_ref(), row)
                            .unwrap()
                    })
                    .collect(),
            );
        }
    }
    rows
}
fn flattened(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let mut rows = ordered_rows(batches);
    rows.sort();
    rows
}

async fn exercise(executor: Executor) {
    let cpu = context(None);
    let gpu = context(Some(executor.clone()));
    let batch = input(131);
    let partitions = vec![
        vec![batch.slice(0, 0), batch.slice(0, 65), batch.slice(65, 66)],
        vec![batch.slice(0, 1)],
    ];
    register_table(&cpu, partitions.clone(), batch.schema());
    register_table(&gpu, partitions, batch.schema());
    for sql in [
        "SELECT x, f FROM t WHERE x >= CAST(3 AS INT)",
        "SELECT x AS same, x AS copy, x IS NULL AS missing FROM t WHERE x < CAST(5 AS INT)",
        "SELECT f FROM t WHERE x > CAST(1000 AS INT)",
        "SELECT x, (x > CAST(3 AS INT)) OR (x IS NULL) AS keep FROM t",
        "SELECT CAST(2 AS INT) AS constant FROM t",
        "SELECT CAST(NULL AS INT) AS missing FROM t WHERE x < CAST(0 AS INT)",
        "SELECT x FROM t WHERE x > CAST(2 AS INT) ORDER BY x DESC LIMIT 5",
    ] {
        let expected_frame = cpu.sql(sql).await.unwrap();
        let expected_schema = expected_frame.schema().clone();
        let expected = expected_frame.collect().await.unwrap();
        let frame = gpu.sql(sql).await.unwrap();
        assert_eq!(frame.schema(), &expected_schema);
        let before = executor.stats();
        let plan = frame.create_physical_plan().await.unwrap();
        let text = displayable(plan.as_ref()).indent(true).to_string();
        assert!(text.contains("CubeExec"), "{sql}: {text}");
        assert_eq!(executor.stats(), before, "planning must not submit work");
        let actual = collect(plan.clone(), gpu.task_ctx()).await.unwrap();
        assert_eq!(flattened(&actual), flattened(&expected), "{sql}: {text}");
        if sql.contains("ORDER BY") {
            assert_eq!(ordered_rows(&actual), ordered_rows(&expected));
        }
        for batch in &actual {
            assert_eq!(batch.schema().as_ref(), expected_schema.as_arrow());
        }
    }
    // The pinned SQL binder rejects duplicate projection names in both modes.
    for ctx in [&cpu, &gpu] {
        assert!(ctx.sql("SELECT x AS same, x AS same FROM t").await.is_err());
    }
    // Device SQL placement is opt-in; wrapping arithmetic remains a CPU node.
    let sql = "SELECT x + CAST(1 AS INT) AS y FROM t";
    let before = executor.stats();
    let actual = gpu.sql(sql).await.unwrap().collect().await.unwrap();
    assert_eq!(executor.stats(), before);
    assert_eq!(
        flattened(&actual),
        flattened(&cpu.sql(sql).await.unwrap().collect().await.unwrap())
    );
    let text_schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int32, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let text = RecordBatch::try_new(
        text_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["one", "two"])),
        ],
    )
    .unwrap();
    let texts = context(Some(executor.clone()));
    register_table(&texts, vec![vec![text]], text_schema);
    let before = executor.stats();
    let text_result = texts
        .sql("SELECT label FROM t WHERE x > CAST(1 AS INT)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(
        text_result[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "two"
    );
    assert_eq!(
        executor.stats(),
        before,
        "unsupported pass-through must remain CPU"
    );
    // A failed request cannot become a successful CPU retry, and budget checks
    // happen before upload. A healthy request on the shared device still works.
    let limited = executor.clone().with_budget(ResourceBudget {
        max_device_bytes: 0,
        ..ResourceBudget::UNLIMITED
    });
    let ctx = context(Some(limited));
    register_table(&ctx, vec![vec![batch.clone()]], batch.schema());
    let before = executor.stats();
    let failure = ctx
        .sql("SELECT x FROM t WHERE x > CAST(3 AS INT)")
        .await
        .unwrap()
        .collect()
        .await;
    assert!(failure.is_err());
    assert_eq!(
        executor.stats(),
        before,
        "budget rejection must precede allocation"
    );
    assert!(
        gpu.sql("SELECT x FROM t WHERE x > CAST(3 AS INT)")
            .await
            .unwrap()
            .collect()
            .await
            .is_ok()
    );
}

async fn lifecycle(executor: Executor) {
    let ctx = SessionContext::new_with_state(
        register(
            SessionStateBuilder::new_with_default_features()
                .with_config(datafusion::prelude::SessionConfig::new().with_target_partitions(1)),
            AdapterOptions::with_executor(executor.clone()),
        )
        .build(),
    );
    let batch = input(32);
    register_table(
        &ctx,
        vec![vec![batch.clone(), batch.clone(), batch.clone()]],
        batch.schema(),
    );
    let plan = ctx
        .sql("SELECT x AS y, x IS NULL AS missing FROM t WHERE x >= CAST(1 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    // Unpolled stream/drop has no submissions or eager prefetch.
    let before = executor.stats();
    let stream = plan.execute(0, ctx.task_ctx()).unwrap();
    drop(stream);
    assert_eq!(executor.stats(), before);
    let mut stream = plan.execute(0, ctx.task_ctx()).unwrap();
    let first = stream.try_next().await.unwrap().unwrap();
    let after_first = executor.stats();
    assert!(first.num_rows() > 0);
    assert_eq!(
        after_first.upload_bytes - before.upload_bytes,
        32 * 8,
        "only x value/validity uploaded once; unused f stays on host"
    );
    assert_eq!(after_first.count_readbacks - before.count_readbacks, 1);
    tokio::task::yield_now().await;
    assert_eq!(
        executor.stats(),
        after_first,
        "no prefetch while consumer waits"
    );
    drop(stream);
    // Rewriting a child recomputes properties and rejects incompatible schemas.
    let cube = find_cube(&plan).expect("CubeExec in plan");
    let child = cube.children()[0].clone();
    let rebuilt = cube
        .clone()
        .replace_children(
            vec![child],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    assert_eq!(rebuilt.schema(), cube.schema());
    assert_eq!(
        rebuilt.output_partitioning().partition_count(),
        cube.output_partitioning().partition_count()
    );
    let sort = datafusion::physical_expr::LexOrdering::new(vec![
        datafusion::physical_expr::PhysicalSortExpr::new(
            Arc::new(datafusion::physical_expr::expressions::Column::new("x", 0)),
            datafusion::arrow::compute::SortOptions::default(),
        ),
    ])
    .unwrap();
    let sorted_child = Arc::new(datafusion::physical_plan::sorts::sort::SortExec::new(
        sort,
        cube.children()[0].clone(),
    ));
    let sorted = cube
        .clone()
        .replace_children(
            vec![sorted_child],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
        .unwrap();
    assert!(
        sorted.output_ordering().is_some(),
        "properties must be recomputed for the changed child"
    );
    assert!(
        cube.clone()
            .replace_children(
                vec![],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute)
            )
            .is_err()
    );
}
fn find_cube(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
    if plan.is::<CubeExec>() {
        return Some(plan.clone());
    }
    plan.children().into_iter().find_map(find_cube)
}

#[cfg(feature = "cpu")]
#[tokio::test(flavor = "multi_thread")]
async fn cpu_sql_fragments_and_stream_lifetimes() {
    let executor = Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap();
    exercise(executor.clone()).await;
    lifecycle(executor).await;
}
#[cfg(feature = "metal")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_sql_fragments_and_stream_lifetimes() {
    let executor = Executor::with_cube_size(
        cubecl::Device::metal(Default::default()).unwrap().client(),
        64,
    )
    .unwrap();
    exercise(executor.clone()).await;
    lifecycle(executor).await;
}
