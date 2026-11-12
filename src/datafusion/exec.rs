//! One bounded filter/project region. No SQL aggregation or cross-query cache.
use ::datafusion::{
    common::{
        DataFusionError, Result,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    execution::TaskContext,
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
        ExecutionPlanProperties, PlanProperties, ReplaceChildrenOptions, SendableRecordBatchStream,
        execution_plan::Boundedness,
        filter::FilterExec,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        projection::ProjectionExec,
        stream::RecordBatchStreamAdapter,
    },
};
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};
use futures::TryStreamExt;
use std::{
    collections::BTreeSet,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Semaphore;

use super::{Unsupported, capability::translate_named};
use crate::{
    Executor, Expr, ExprId, ExprNode, Field, KernelProgram, Parameters, Pipeline, Schema,
    arrow::ArrowBatch, prepare,
};

/// An adapter instance serializes its device requests across partitions/queries.
/// The permit belongs to the blocking worker through completion, even after drop.
pub(crate) struct Engine {
    pub(crate) executor: Executor,
    gate: Arc<Semaphore>,
}
impl Engine {
    pub(crate) fn new(executor: Executor) -> Self {
        Self {
            executor,
            gate: Arc::new(Semaphore::new(1)),
        }
    }
}
impl fmt::Debug for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Engine")
            .field("cube_size", &self.executor.cube_size())
            .field("budget", &self.executor.budget())
            .finish()
    }
}
fn execution(error: impl fmt::Display) -> DataFusionError {
    DataFusionError::Execution(format!("Cuberel: {error}"))
}
fn reject(code: &'static str, detail: impl Into<String>) -> Unsupported {
    Unsupported {
        code,
        detail: detail.into(),
    }
}

#[derive(Clone, Debug)]
struct Fragment {
    pipeline: Pipeline,
    parameters: Parameters,
    input_columns: Vec<usize>,
    input_program: KernelProgram,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
}
impl Fragment {
    fn compile(
        region: &Arc<dyn ExecutionPlan>,
    ) -> std::result::Result<(Self, Arc<dyn ExecutionPlan>, bool), Unsupported> {
        let (projection, mut input) = if let Some(p) = region.downcast_ref::<ProjectionExec>() {
            (Some(p), p.input().clone())
        } else {
            (None, region.clone())
        };
        let filter = input.downcast_ref::<FilterExec>();
        if projection.is_none() && filter.is_none() {
            return Err(reject("unsupported_plan", region.name()));
        }
        if let Some(filter) = filter {
            if filter.fetch().is_some() {
                return Err(reject("unsupported_plan", "filter fetch is a CPU boundary"));
            }
            if projection.is_some() && filter.projection().is_some() {
                return Err(reject(
                    "unsupported_plan",
                    "projected filter requires field remapping",
                ));
            }
        }
        let predicate = filter.map(|f| f.predicate().clone());
        let output_indices = filter.and_then(|f| f.projection().clone());
        if let Some(filter) = filter {
            input = filter.input().clone();
        }
        if input.boundedness() != Boundedness::Bounded {
            return Err(reject("unbounded_input", "finite input required"));
        }
        let outputs: Vec<Arc<dyn PhysicalExpr>> = if let Some(projection) = projection {
            projection.expr().iter().map(|e| e.expr.clone()).collect()
        } else {
            let indices: Vec<_> = output_indices
                .map(|p| p.as_ref().to_vec())
                .unwrap_or_else(|| (0..input.schema().fields().len()).collect());
            indices
                .into_iter()
                .map(|i| {
                    Arc::new(Column::new(input.schema().field(i).name(), i))
                        as Arc<dyn PhysicalExpr>
                })
                .collect()
        };
        if outputs.is_empty() || outputs.len() > 32 {
            return Err(reject(
                "expression_limit",
                "one to 32 projected columns required",
            ));
        }
        let mut pipeline = Pipeline::new();
        let mut parameters = Parameters::new();
        let mut columns = BTreeSet::new();
        let mut first_program = None;
        for (i, expr) in predicate.iter().chain(outputs.iter()).enumerate() {
            let t = translate_named(expr, input.schema().as_ref(), &format!("e{i}"))?;
            for (name, value) in t.literal_bindings() {
                parameters = parameters
                    .with(name, *value)
                    .map_err(|e| reject("binding", e.to_string()))?;
            }
            columns.extend(t.input_columns);
            first_program.get_or_insert_with(|| t.program.clone());
            if predicate.is_some() && i == 0 {
                pipeline = pipeline.filter(t.program);
            } else {
                pipeline = pipeline.project(format!("out{i}"), t.program);
            }
        }
        let input_columns: Vec<_> = columns.into_iter().collect();
        let mut fields = Vec::new();
        let mut own_fields = Vec::new();
        let mut nodes = Vec::new();
        for &index in &input_columns {
            let field = input
                .schema()
                .field(index)
                .clone()
                .with_name(format!("c{index}"));
            let ty = match field.data_type() {
                arrow_schema::DataType::Int32 => crate::DataType::I32,
                arrow_schema::DataType::Float32 => crate::DataType::F32,
                _ => return Err(reject("unsupported_type", "only numeric fragment inputs")),
            };
            own_fields.push(Field::new(field.name(), ty).with_nullable(field.is_nullable()));
            nodes.push(ExprNode::Column(field.name().clone()));
            fields.push(field);
        }
        let input_program = if nodes.is_empty() {
            first_program.expect("nonempty output")
        } else {
            prepare(
                &Expr::from_nodes(nodes, ExprId::new(0)),
                &Schema::from_fields(&own_fields, &[])
                    .map_err(|e| reject("binding", e.to_string()))?,
            )
            .map_err(|e| reject("binding", e.to_string()))?
        };
        Ok((
            Self {
                pipeline,
                parameters,
                input_columns,
                input_program,
                input_schema: Arc::new(ArrowSchema::new(fields)),
                output_schema: region.schema(),
            },
            input,
            predicate.is_some(),
        ))
    }
    fn run(&self, executor: &Executor, input: &RecordBatch) -> Result<RecordBatch> {
        let columns = self
            .input_columns
            .iter()
            .map(|&i| input.column(i).clone())
            .collect();
        let renamed = RecordBatch::try_new_with_options(
            self.input_schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(input.num_rows())),
        )
        .map_err(execution)?;
        let adapter = ArrowBatch::try_new(&self.input_program, &renamed).map_err(execution)?;
        let host = adapter.as_batch().map_err(execution)?;
        let result = executor
            .execute_pipeline_host(&self.pipeline, &host, &self.parameters)
            .map_err(execution)?;
        let materialized = result
            .materialize_arrow()
            .map_err(execution)?
            .projections
            .ok_or_else(|| execution("missing projection"))?;
        RecordBatch::try_new(self.output_schema.clone(), materialized.columns().to_vec())
            .map_err(execution)
    }
}

/// A bounded SQL filter/project fragment on one explicitly selected executor.
/// Its output schema/properties come from the equivalent DataFusion region.
#[derive(Debug)]
pub struct CubeExec {
    input: Arc<dyn ExecutionPlan>,
    region: Arc<dyn ExecutionPlan>,
    fragment: Arc<Fragment>,
    engine: Arc<Engine>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
    has_filter: bool,
}
impl CubeExec {
    fn try_new(
        region: Arc<dyn ExecutionPlan>,
        engine: Arc<Engine>,
    ) -> std::result::Result<Self, Unsupported> {
        let (fragment, input, has_filter) = Fragment::compile(&region)?;
        Ok(Self {
            input,
            properties: region.properties().clone(),
            region,
            fragment: Arc::new(fragment),
            engine,
            metrics: ExecutionPlanMetricsSet::new(),
            has_filter,
        })
    }
}
impl DisplayAs for CubeExec {
    fn fmt_as(&self, _kind: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CubeExec: filter={}, outputs={}, semantic_profile=df55-i32-v1, max_in_flight=1",
            self.has_filter,
            self.schema().fields().len()
        )
    }
}
impl ExecutionPlan for CubeExec {
    fn name(&self) -> &'static str {
        "CubeExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let result = self.region.apply_expressions(f)?;
        if result == TreeNodeRecursion::Stop {
            return Ok(result);
        }
        if self.has_filter
            && let Some(projection) = self.region.downcast_ref::<ProjectionExec>()
        {
            return projection.input().apply_expressions(f);
        }
        Ok(result)
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }
    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(execution("CubeExec needs one child"));
        }
        let input = children.remove(0);
        if input.schema() != self.input.schema() {
            return Err(execution("replacement child schema changed"));
        }
        let options = ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute);
        let region = if let Some(projection) = self.region.downcast_ref::<ProjectionExec>() {
            let child = if self.has_filter {
                projection
                    .input()
                    .clone()
                    .replace_children(vec![input], options)?
            } else {
                input
            };
            self.region.clone().replace_children(
                vec![child],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?
        } else {
            self.region.clone().replace_children(vec![input], options)?
        };
        Ok(Arc::new(
            Self::try_new(region, self.engine.clone()).map_err(execution)?,
        ))
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let stream = self.input.execute(partition, context)?;
        let state = StreamState {
            input: stream,
            engine: self.engine.clone(),
            fragment: self.fragment.clone(),
            cancellation: CancelOnDrop(Arc::new(AtomicBool::new(false))),
            batch: 0,
            partition,
            rows: MetricBuilder::new(&self.metrics).counter("input_rows", partition),
            output: MetricBuilder::new(&self.metrics).counter("output_rows", partition),
        };
        let stream = futures::stream::try_unfold(state, |mut state| async move {
            let Some(batch) = state.input.try_next().await? else {
                return Ok(None);
            };
            let permit = state
                .engine
                .gate
                .clone()
                .acquire_owned()
                .await
                .map_err(execution)?;
            let engine = state.engine.clone();
            let fragment = state.fragment.clone();
            let cancelled = state.cancellation.0.clone();
            let rows = batch.num_rows();
            let result = tokio::task::spawn_blocking(move || {
                let _permit = permit;
                if cancelled.load(Ordering::Acquire) {
                    return Err(execution("cancelled before submission"));
                }
                // The permit and all input/device ownership survive cancellation.
                // A submitted kernel is drained by materialize, never host-retried.
                fragment.run(&engine.executor, &batch)
            })
            .await
            .map_err(execution)?
            .map_err(|e| {
                execution(format!(
                    "partition={} batch={}: {e}",
                    state.partition, state.batch
                ))
            })?;
            state.rows.add(rows);
            state.output.add(result.num_rows());
            state.batch += 1;
            Ok(Some((result, state)))
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }
}
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}
struct StreamState {
    input: SendableRecordBatchStream,
    engine: Arc<Engine>,
    fragment: Arc<Fragment>,
    cancellation: CancelOnDrop,
    partition: usize,
    batch: usize,
    rows: ::datafusion::physical_plan::metrics::Count,
    output: ::datafusion::physical_plan::metrics::Count,
}

pub(crate) fn rewrite(
    plan: Arc<dyn ExecutionPlan>,
    engine: &Arc<Engine>,
) -> Result<Arc<dyn ExecutionPlan>> {
    if plan.is::<CubeExec>() {
        return Ok(plan);
    }
    if let Ok(exec) = CubeExec::try_new(plan.clone(), engine.clone()) {
        return Ok(Arc::new(exec));
    }
    Ok(plan
        .map_children(|child| rewrite(child, engine).map(Transformed::yes))?
        .data)
}

#[cfg(all(test, any(feature = "cpu", feature = "metal")))]
mod tests {
    use super::{Engine, rewrite};
    use crate::Executor;
    use ::datafusion::{physical_plan::collect, prelude::SessionContext};
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use std::{sync::Arc, time::Duration};

    async fn cancellation(executor: Executor) {
        let engine = Arc::new(Engine::new(executor.clone()));
        let ctx = SessionContext::new();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from((0..8192).collect::<Vec<i32>>()))],
        )
        .unwrap();
        ctx.register_batch("t", batch).unwrap();
        let original = ctx
            .sql("SELECT x AS y FROM t WHERE x > CAST(3 AS INT)")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let plan = rewrite(original, &engine).unwrap();
        let held = engine.gate.clone().acquire_owned().await.unwrap();
        let before = executor.stats();
        let mut stream = plan.execute(0, ctx.task_ctx()).unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(10), stream.try_next())
                .await
                .is_err()
        );
        drop(stream);
        drop(held);
        tokio::task::yield_now().await;
        assert_eq!(
            executor.stats(),
            before,
            "cancelled waiter cannot upload later"
        );

        let query = plan.clone();
        let task_ctx = ctx.task_ctx();
        let task = tokio::spawn(async move { collect(query, task_ctx).await });
        tokio::time::timeout(Duration::from_secs(30), async {
            while executor.stats().kernel_launches == before.kernel_launches {
                assert!(
                    !task.is_finished(),
                    "query ended without a device submission"
                );
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        task.abort();
        let _ = task.await;
        // This permit is returned only by the worker after completion/error,
        // never merely by dropping the async JoinHandle.
        let permit =
            tokio::time::timeout(Duration::from_secs(30), engine.gate.clone().acquire_owned())
                .await
                .unwrap()
                .unwrap();
        drop(permit);
        assert!(
            collect(plan, ctx.task_ctx()).await.is_ok(),
            "device reusable after cancellation"
        );
    }
    #[cfg(feature = "cpu")]
    #[tokio::test(flavor = "multi_thread")]
    async fn cpu_cancel_waiting_and_submitted_work() {
        cancellation(Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap())
            .await;
    }
    #[cfg(feature = "metal")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_cancel_waiting_and_submitted_work() {
        cancellation(
            Executor::with_cube_size(
                cubecl::Device::metal(Default::default()).unwrap().client(),
                64,
            )
            .unwrap(),
        )
        .await;
    }
}
