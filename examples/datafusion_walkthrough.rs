//! Print actual SQL plans, normalized programs, a controlled F4 chain and costs.
#[cfg(any(feature = "cpu", feature = "metal"))]
mod runtime {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
    use cuberel::{
        Executor,
        abi::BindingLayout,
        datafusion::{AdapterOptions, inspect_fragment, register},
    };
    use datafusion::{
        common::ScalarValue,
        execution::session_state::{SessionState, SessionStateBuilder},
        logical_expr::Operator,
        physical_expr::{
            PhysicalExpr,
            expressions::{BinaryExpr, Column, IsNullExpr, Literal},
        },
        physical_plan::{
            ExecutionPlan, collect, displayable, filter::FilterExec, projection::ProjectionExec,
        },
        prelude::{SessionConfig, SessionContext},
    };
    use std::{
        error::Error,
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        sync::Arc,
    };
    type StudyResult<T> = Result<T, Box<dyn Error>>;
    fn col(name: &str, index: usize) -> Arc<dyn PhysicalExpr> {
        Arc::new(Column::new(name, index))
    }
    fn compare(column: Arc<dyn PhysicalExpr>, op: Operator, value: i32) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(
            column,
            op,
            Arc::new(Literal::new(ScalarValue::Int32(Some(value)))),
        ))
    }
    fn project(
        input: Arc<dyn ExecutionPlan>,
        exprs: Vec<(Arc<dyn PhysicalExpr>, String)>,
    ) -> StudyResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
    }
    fn snapshot(out: &Option<PathBuf>, name: &str, text: impl AsRef<str>) -> StudyResult<()> {
        println!("\n--- {name} ---\n{}", text.as_ref());
        if let Some(out) = out {
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join(name))?
                .write_all(text.as_ref().as_bytes())?;
        }
        Ok(())
    }
    fn rewrite(
        state: &SessionState,
        plan: Arc<dyn ExecutionPlan>,
    ) -> StudyResult<Arc<dyn ExecutionPlan>> {
        let rule = state
            .physical_optimizers()
            .iter()
            .find(|r| r.name() == "cuberel")
            .ok_or("missing rule")?;
        let plan = rule.optimize(plan, state.config_options())?;
        Ok(state
            .physical_optimizers()
            .last()
            .ok_or("missing sanity rule")?
            .optimize(plan, state.config_options())?)
    }
    async fn run_plan(
        label: &str,
        original: Arc<dyn ExecutionPlan>,
        state: &SessionState,
        ctx: &SessionContext,
        executor: &Executor,
        out: &Option<PathBuf>,
    ) -> StudyResult<()> {
        snapshot(
            out,
            &format!("{label}-before.txt"),
            displayable(original.as_ref()).indent(true).to_string(),
        )?;
        let inspection = inspect_fragment(original.clone())?;
        let mut programs = format!(
            "nodes={} filters={} inputs={:?}\nboundary={}\nnormalized={:#?}\n",
            inspection.physical_nodes,
            inspection.filter_count,
            inspection.input_columns,
            inspection.boundary,
            inspection.normalized_expressions
        );
        for (index, program) in inspection.programs.iter().enumerate() {
            programs.push_str(&format!("\nprogram {index} (filter if index < {}):\n{program:#?}\nbase binding layout (selected launches add indices/count):\n{:#?}\n",inspection.filter_count,BindingLayout::new(program)));
        }
        snapshot(out, &format!("{label}-programs.txt"), programs)?;
        let before = executor.stats();
        let device = rewrite(state, original.clone())?;
        assert_eq!(executor.stats(), before, "planning cannot submit work");
        snapshot(
            out,
            &format!("{label}-after.txt"),
            displayable(device.as_ref()).indent(true).to_string(),
        )?;
        let expected = collect(original, ctx.task_ctx()).await?;
        let defines = executor.define_count();
        let actual = collect(device.clone(), ctx.task_ctx()).await?;
        let first = executor.stats();
        assert_eq!(actual.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        let values: Vec<_> = actual
            .iter()
            .flat_map(|b| {
                b.column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .expect("Int32 value")
                    .values()
                    .iter()
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(values, [30, 40]);
        assert_eq!(expected.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        let result = format!(
            "{actual:#?}\nfirst_define_delta={}\nstats_before={before:?}\nstats_after={first:?}\n",
            executor.define_count() - defines
        );
        snapshot(out, &format!("{label}-result.txt"), result)?;
        let warm_defines = executor.define_count();
        let _ = collect(device, ctx.task_ctx()).await?;
        snapshot(
            out,
            &format!("{label}-warm.txt"),
            format!(
                "warm_define_delta={}\nstats={:?}\n",
                executor.define_count() - warm_defines,
                executor.stats()
            ),
        )?;
        Ok(())
    }
    fn configure_export(out: &Path, backend: &str) -> StudyResult<()> {
        fs::create_dir_all(out)?;
        if out.join("environment.sqlite").exists() {
            return Err("export destination already has environment.sqlite".into());
        }
        let mut config = CubeClRuntimeConfig::default();
        config.compilation.cache = true;
        config.environment.name = format!("cuberel-study-{backend}-{}", std::process::id());
        config.environment.records.level = cubecl::records::RecordLevel::Full;
        CubeClRuntimeConfig::set(config);
        cubecl::environment::set_root(out.join("cache"));
        Ok(())
    }
    pub async fn run() -> StudyResult<()> {
        let mut args = std::env::args().skip(1);
        let backend = args.next().unwrap_or_else(|| "cpu".into());
        let out = args.next().map(PathBuf::from);
        if let Some(out) = &out {
            configure_export(out, &backend)?;
        }
        let device = match backend.as_str() {
            #[cfg(feature = "cpu")]
            "cpu" => cubecl::Device::cpu()?,
            #[cfg(feature = "metal")]
            "metal" => cubecl::Device::metal(Default::default())?,
            _ => return Err(format!("backend {backend} not enabled").into()),
        };
        let tile = if backend == "cpu" { 8 } else { 64 };
        let executor = Executor::with_cube_size(device.client(), tile)?;
        let config = SessionConfig::new().with_target_partitions(1);
        let ctx = SessionContext::new_with_config(config.clone());
        let state = register(
            SessionStateBuilder::new_with_default_features().with_config(config),
            AdapterOptions::with_executor(executor.clone()),
        )
        .build();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![
                    Some(10),
                    None,
                    Some(30),
                    Some(40),
                    Some(100),
                ])),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            ],
        )?;
        ctx.register_batch("t", batch)?;
        let sql = "SELECT value, value IS NULL AS missing, tag FROM (SELECT x AS value, y AS tag FROM t WHERE x >= CAST(30 AS INT)) q WHERE value < CAST(50 AS INT)";
        snapshot(&out, "query.sql", sql)?;
        let frame = ctx.sql(sql).await?;
        snapshot(
            &out,
            "logical.txt",
            frame.logical_plan().display_indent_schema().to_string(),
        )?;
        let optimized = ctx.state().optimize(frame.logical_plan())?;
        snapshot(
            &out,
            "optimized-logical.txt",
            optimized.display_indent_schema().to_string(),
        )?;
        run_plan(
            "sql",
            frame.create_physical_plan().await?,
            &state,
            &ctx,
            &executor,
            &out,
        )
        .await?;

        // A real DataFusion physical fixture keeps the four F4 nodes explicit:
        // SQL optimization is allowed to combine equivalent pure predicates.
        let source = ctx.table("t").await?.create_physical_plan().await?;
        let lower: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(
            compare(col("x", 0), Operator::GtEq, 30),
            source,
        )?);
        let renamed = project(
            lower,
            vec![(col("y", 1), "k".into()), (col("x", 0), "renamed".into())],
        )?;
        let upper: Arc<dyn ExecutionPlan> = Arc::new(FilterExec::try_new(
            compare(col("renamed", 1), Operator::Lt, 50),
            renamed,
        )?);
        let plan = project(
            upper,
            vec![
                (col("renamed", 1), "value".into()),
                (
                    Arc::new(IsNullExpr::new(col("renamed", 1))),
                    "missing".into(),
                ),
                (col("k", 0), "tag".into()),
            ],
        )?;
        run_plan("chain", plan, &state, &ctx, &executor, &out).await?;
        if let Some(out) = out {
            if !cubecl::environment::namespaces()
                .iter()
                .any(|n| n.namespace == "records/v1/compilation" && n.entries != 0)
            {
                snapshot(
                    &Some(out),
                    "records-unavailable.txt",
                    "No durable compilation records from this backend/build. Text traces are complete. Metal with cubecl/persistence supports the full IR/MSL export on the pinned revision; CPU currently does not use that recording path.",
                )?;
                return Ok(());
            }
            cubecl::environment::bundle().save(
                out.join("environment.sqlite"),
                cubecl::bundle::BundleFormat::Sqlite,
            )?;
            println!(
                "full compilation records exported to {}",
                out.join("environment.sqlite").display()
            );
        }
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
    eprintln!("Enable cpu or metal to execute this walkthrough");
    std::process::exit(2);
}
