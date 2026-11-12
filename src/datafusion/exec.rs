//! Bounded filter/project execution and shared SQL device admission.
use ::datafusion::{
    common::{
        DataFusionError, Result,
        tree_node::{Transformed, TreeNode, TreeNodeRecursion},
    },
    execution::TaskContext,
    physical_expr::PhysicalExpr,
    physical_plan::{
        ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
        ReplaceChildrenOptions, SendableRecordBatchStream,
        metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet},
        stream::RecordBatchStreamAdapter,
    },
};
use futures::TryStreamExt;
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Semaphore;

use super::{
    Unsupported,
    fragment::{Fragment, Region},
};
use crate::Executor;

/// An adapter instance serializes its device requests across partitions/queries.
/// The permit belongs to the blocking worker through completion, even after drop.
pub(crate) struct Engine {
    pub(crate) executor: Executor,
    pub(crate) gate: Arc<Semaphore>,
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
pub(crate) fn execution(error: impl fmt::Display) -> DataFusionError {
    DataFusionError::Execution(format!("Cuberel: {error}"))
}
/// A bounded SQL filter/project fragment on one explicitly selected executor.
/// Its output schema/properties come from the equivalent DataFusion region.
#[derive(Debug)]
pub struct CubeExec {
    input: Arc<dyn ExecutionPlan>,
    region: Region,
    fragment: Arc<Fragment>,
    engine: Arc<Engine>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}
impl CubeExec {
    fn try_new(
        region: Arc<dyn ExecutionPlan>,
        engine: Arc<Engine>,
    ) -> std::result::Result<Self, Unsupported> {
        let (region, fragment) = Region::discover(region)?;
        Ok(Self {
            input: region.input.clone(),
            properties: region.root().properties().clone(),
            region,
            fragment: Arc::new(fragment),
            engine,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}
impl DisplayAs for CubeExec {
    fn fmt_as(&self, _kind: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CubeExec: filter={}, filters={}, outputs={}, nodes={}, boundary=[{}], semantic_profile=df55-i32-v1, max_in_flight=1",
            self.fragment.filter_count != 0,
            self.fragment.filter_count,
            self.schema().fields().len(),
            self.region.nodes.len(),
            self.region.boundary
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
        for root in &self.fragment.roots {
            if f(root)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
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
        let region = self.region.rebuild(input)?;
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
pub(crate) struct CancelOnDrop(pub(crate) Arc<AtomicBool>);
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
    if plan.is::<CubeExec>() || plan.is::<super::global::CubeAggregateExec>() {
        return Ok(plan);
    }
    if let Ok(exec) = super::global::CubeAggregateExec::try_new(plan.clone(), engine.clone()) {
        let original_input = exec.children()[0].clone();
        let input = rewrite(original_input.clone(), engine)?;
        let exec = Arc::new(exec);
        return if Arc::ptr_eq(&input, &original_input) {
            Ok(exec)
        } else {
            exec.replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
        };
    }
    if let Ok(exec) = CubeExec::try_new(plan.clone(), engine.clone()) {
        let input = rewrite(exec.input.clone(), engine)?;
        if Arc::ptr_eq(&input, &exec.input) {
            return Ok(Arc::new(exec));
        }
        let root = exec.region.rebuild(input)?;
        let mut rebuilt = CubeExec::try_new(root, engine.clone()).map_err(execution)?;
        rebuilt.region.boundary = exec.region.boundary;
        return Ok(Arc::new(rebuilt));
    }
    Ok(plan
        .map_children(|child| rewrite(child, engine).map(Transformed::yes))?
        .data)
}

#[cfg(all(test, any(feature = "cpu", feature = "metal")))]
mod tests {
    use super::{Engine, rewrite};
    use crate::Executor;
    use ::datafusion::{
        physical_plan::{collect, execution_plan::reset_plan_states},
        prelude::{SessionConfig, SessionContext},
    };
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use futures::TryStreamExt;
    use std::{sync::Arc, time::Duration};

    async fn cancellation(executor: Executor) {
        let engine = Arc::new(Engine::new(executor.clone()));
        // Keep this admission/worker test independent of DataFusion's optional
        // repartition producers; exchange cancellation is tested separately.
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from((0..129).collect::<Vec<i32>>()))],
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

        let query = reset_plan_states(plan.clone()).unwrap();
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
            collect(reset_plan_states(plan).unwrap(), ctx.task_ctx())
                .await
                .is_ok(),
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

    async fn exchange_cancellation(executor: Executor) {
        use ::datafusion::{
            common::ScalarValue,
            logical_expr::Operator,
            physical_expr::{
                PhysicalExpr,
                expressions::{BinaryExpr, Column, Literal},
            },
            physical_plan::{
                ExecutionPlan, Partitioning, displayable, filter::FilterExec,
                projection::ProjectionExec, repartition::RepartitionExec,
            },
        };
        let engine = Arc::new(Engine::new(executor.clone()));
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(1));
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)])),
            vec![Arc::new(Int32Array::from((0..129).collect::<Vec<i32>>()))],
        )
        .unwrap();
        ctx.register_batch("t", batch).unwrap();
        let source = ctx
            .table("t")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let column: Arc<dyn PhysicalExpr> = Arc::new(Column::new("x", 0));
        let predicate = Arc::new(BinaryExpr::new(
            column.clone(),
            Operator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(3)))),
        ));
        let lower: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(predicate, source).unwrap());
        let exchange: Arc<dyn ExecutionPlan> =
            Arc::new(RepartitionExec::try_new(lower, Partitioning::RoundRobinBatch(2)).unwrap());
        let original: Arc<dyn ExecutionPlan> =
            Arc::new(ProjectionExec::try_new(vec![(column, "y".to_string())], exchange).unwrap());
        let plan = rewrite(original, &engine).unwrap();
        assert_eq!(
            displayable(plan.as_ref())
                .indent(true)
                .to_string()
                .matches("CubeExec:")
                .count(),
            2
        );
        let held = engine.gate.clone().acquire_owned().await.unwrap();
        let before = executor.stats();
        let query = plan.clone();
        let task_ctx = ctx.task_ctx();
        let mut task = tokio::spawn(async move { collect(query, task_ctx).await });
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut task)
                .await
                .is_err()
        );
        task.abort();
        let _ = task.await;
        let fresh = reset_plan_states(plan.clone()).unwrap();
        drop(plan);
        drop(held);
        let permit =
            tokio::time::timeout(Duration::from_secs(30), engine.gate.clone().acquire_owned())
                .await
                .unwrap()
                .unwrap();
        assert_eq!(
            executor.stats(),
            before,
            "cancelled exchange producers cannot submit after admission resumes"
        );
        drop(permit);
        let output = tokio::time::timeout(Duration::from_secs(30), collect(fresh, ctx.task_ctx()))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 125);
        assert_eq!(
            executor.stats().upload_bytes - before.upload_bytes,
            (129 + 125) * 8
        );
    }
    #[cfg(feature = "cpu")]
    #[tokio::test(flavor = "multi_thread")]
    async fn cpu_cancel_exchange_between_fragments() {
        exchange_cancellation(
            Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
        )
        .await;
    }
    #[cfg(feature = "metal")]
    #[tokio::test(flavor = "multi_thread")]
    async fn metal_cancel_exchange_between_fragments() {
        exchange_cancellation(
            Executor::with_cube_size(
                cubecl::Device::metal(Default::default()).unwrap().client(),
                64,
            )
            .unwrap(),
        )
        .await;
    }
}
