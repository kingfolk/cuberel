//! Partition-scoped observations. Executor deltas are not driver compile counts.
use crate::{ExecutionStats, Executor, ResourceEstimate, StageGraph};
use ::datafusion::{
    common::Result,
    physical_plan::{
        RecordBatchStream, SendableRecordBatchStream,
        metrics::{Count, ExecutionPlanMetricsSet, Gauge, MetricBuilder, Time},
    },
};
use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use futures::Stream;
use std::{
    collections::BTreeMap,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll},
    time::Instant,
};

pub(crate) type ObservationCache = Mutex<BTreeMap<usize, Arc<Observation>>>;
#[derive(Debug)]
pub(crate) struct Observation {
    counts: BTreeMap<&'static str, Count>,
    pub(crate) queue: Time,
    pub(crate) prepare: Time,
    pub(crate) execute_export: Time,
    worker: Time,
    wall: Time,
    active: Gauge,
    device: Gauge,
    workspace: Gauge,
    stages: Gauge,
    output_capacity: Gauge,
    set: ExecutionPlanMetricsSet,
    partition: usize,
    fallback: Mutex<BTreeMap<&'static str, Count>>,
}
impl Observation {
    pub(crate) fn cached(
        cache: &ObservationCache,
        set: &ExecutionPlanMetricsSet,
        partition: usize,
    ) -> Arc<Self> {
        cache
            .lock()
            .unwrap()
            .entry(partition)
            .or_insert_with(|| Self::new(set, partition))
            .clone()
    }
    pub(crate) fn new(set: &ExecutionPlanMetricsSet, partition: usize) -> Arc<Self> {
        let names = [
            "input_rows",
            "input_batches",
            "output_rows",
            "output_batches",
            "streams_completed",
            "streams_failed",
            "streams_canceled",
            "requests_completed",
            "request_errors",
            "observed_upload_calls",
            "observed_upload_bytes",
            "observed_readback_calls",
            "observed_readback_bytes",
            "observed_launches",
            "observed_count_reads",
            "observed_alloc_calls",
            "observed_alloc_bytes",
            "define_calls",
            "fused_requests",
            "unfused_requests",
        ];
        let counts = names
            .into_iter()
            .map(|n| {
                (
                    n,
                    match n {
                        "output_rows" => MetricBuilder::new(set).output_rows(partition),
                        "output_batches" => MetricBuilder::new(set).output_batches(partition),
                        _ => MetricBuilder::new(set).counter(n, partition),
                    },
                )
            })
            .collect();
        Arc::new(Self {
            counts,
            queue: MetricBuilder::new(set).subset_time("queue_wait", partition),
            prepare: MetricBuilder::new(set).subset_time("prepare", partition),
            execute_export: MetricBuilder::new(set).subset_time("execute_export", partition),
            worker: MetricBuilder::new(set).subset_time("worker", partition),
            wall: MetricBuilder::new(set).subset_time("stream_wall", partition),
            active: MetricBuilder::new(set).gauge("active_streams", partition),
            device: MetricBuilder::new(set).gauge("max_estimated_device_bytes", partition),
            workspace: MetricBuilder::new(set).gauge("max_requested_workspace_bytes", partition),
            stages: MetricBuilder::new(set).gauge("max_estimated_stages", partition),
            output_capacity: MetricBuilder::new(set).gauge("max_output_capacity", partition),
            set: set.clone(),
            partition,
            fallback: Mutex::new(BTreeMap::new()),
        })
    }
    pub(crate) fn add(&self, name: &'static str, value: u64) {
        self.counts[name].add(usize::try_from(value).unwrap_or(usize::MAX));
    }
    pub(crate) fn input(&self, rows: usize) {
        self.add("input_rows", rows as u64);
        self.add("input_batches", 1);
    }
    pub(crate) fn workspace(&self, bytes: u64) {
        self.workspace
            .set_max(usize::try_from(bytes).unwrap_or(usize::MAX));
    }
    pub(crate) fn estimate(&self, e: &ResourceEstimate) {
        self.device.set_max(e.device_bytes as usize);
        self.stages.set_max(e.stages as usize);
        self.output_capacity.set_max(e.output_rows as usize);
    }
    pub(crate) fn fusion(&self, graph: &StageGraph) {
        self.add(
            if graph.is_fused() {
                "fused_requests"
            } else {
                "unfused_requests"
            },
            1,
        );
        if let Some(reason) = graph.fusion_reason() {
            let mut reasons = self.fallback.lock().unwrap();
            let c = reasons.entry(reason).or_insert_with(|| {
                MetricBuilder::new(&self.set)
                    .with_new_label("reason", reason)
                    .counter("fusion_fallback", self.partition)
            });
            c.add(1);
        }
    }
    pub(crate) fn worker(self: &Arc<Self>, executor: &Executor) -> WorkerObservation {
        WorkerObservation {
            observer: self.clone(),
            executor: executor.clone(),
            stats: executor.stats(),
            defines: executor.define_count(),
            start: Instant::now(),
        }
    }
    pub(crate) fn stream(
        self: &Arc<Self>,
        inner: SendableRecordBatchStream,
    ) -> SendableRecordBatchStream {
        Box::pin(ObservedStream {
            inner,
            observer: self.clone(),
            start: None,
            done: false,
        })
    }
}
pub(crate) struct WorkerObservation {
    observer: Arc<Observation>,
    executor: Executor,
    stats: ExecutionStats,
    defines: usize,
    start: Instant,
}
impl Drop for WorkerObservation {
    fn drop(&mut self) {
        let now = self.executor.stats();
        let old = self.stats;
        for (name, new, old) in [
            ("observed_upload_calls", now.upload_calls, old.upload_calls),
            ("observed_upload_bytes", now.upload_bytes, old.upload_bytes),
            (
                "observed_readback_calls",
                now.readback_calls,
                old.readback_calls,
            ),
            (
                "observed_readback_bytes",
                now.readback_bytes,
                old.readback_bytes,
            ),
            (
                "observed_launches",
                now.kernel_launches,
                old.kernel_launches,
            ),
            (
                "observed_count_reads",
                now.count_readbacks,
                old.count_readbacks,
            ),
            (
                "observed_alloc_calls",
                now.allocation_calls,
                old.allocation_calls,
            ),
            (
                "observed_alloc_bytes",
                now.allocation_bytes,
                old.allocation_bytes,
            ),
        ] {
            self.observer.add(name, new.saturating_sub(old));
        }
        self.observer.add(
            "define_calls",
            self.executor.define_count().saturating_sub(self.defines) as u64,
        );
        self.observer.worker.add_elapsed(self.start);
    }
}
struct ObservedStream {
    inner: SendableRecordBatchStream,
    observer: Arc<Observation>,
    start: Option<Instant>,
    done: bool,
}
impl ObservedStream {
    fn stop(&mut self) {
        if let Some(start) = self.start.take() {
            self.observer.wall.add_elapsed(start);
            self.observer.active.sub(1);
        }
    }
}
impl RecordBatchStream for ObservedStream {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}
impl Stream for ObservedStream {
    type Item = Result<RecordBatch>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done {
            return Poll::Ready(None);
        }
        if this.start.is_none() {
            this.start = Some(Instant::now());
            this.observer.active.add(1);
        }
        let next = this.inner.as_mut().poll_next(cx);
        match &next {
            Poll::Ready(Some(Ok(batch))) => {
                this.observer.add("output_rows", batch.num_rows() as u64);
                this.observer.add("output_batches", 1);
            }
            Poll::Ready(Some(Err(_))) => {
                this.observer.add("streams_failed", 1);
                this.done = true;
                this.stop();
            }
            Poll::Ready(None) => {
                this.observer.add("streams_completed", 1);
                this.done = true;
                this.stop();
            }
            Poll::Pending => {}
        }
        next
    }
}
impl Drop for ObservedStream {
    fn drop(&mut self) {
        if !self.done && self.start.is_some() {
            self.observer.add("streams_canceled", 1);
        }
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::Observation;
    use datafusion::{
        common::DataFusionError,
        physical_plan::{metrics::ExecutionPlanMetricsSet, stream::RecordBatchStreamAdapter},
    };
    use futures::TryStreamExt;
    use std::sync::Arc;
    #[tokio::test]
    async fn pending_drop_and_terminal_error_are_observed_once() {
        let set = ExecutionPlanMetricsSet::new();
        let o = Observation::new(&set, 0);
        let mut stream = o.stream(Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(arrow_schema::Schema::empty()),
            futures::stream::pending(),
        )));
        assert!(futures::poll!(stream.try_next()).is_pending());
        assert_eq!(o.active.value(), 1);
        drop(stream);
        assert_eq!(o.counts["streams_canceled"].value(), 1);
        assert_eq!(o.active.value(), 0);
        let o = Observation::new(&set, 1);
        let mut stream = o.stream(Box::pin(RecordBatchStreamAdapter::new(
            Arc::new(arrow_schema::Schema::empty()),
            futures::stream::once(async { Err(DataFusionError::Execution("safe failure".into())) }),
        )));
        assert!(stream.try_next().await.is_err());
        assert!(stream.try_next().await.unwrap().is_none());
        drop(stream);
        assert_eq!(o.counts["streams_failed"].value(), 1);
        assert_eq!(o.counts["streams_completed"].value(), 0);
        assert_eq!(o.counts["streams_canceled"].value(), 0);
        assert_eq!(o.active.value(), 0);
    }
}
