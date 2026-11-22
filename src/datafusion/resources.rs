//! Shared logical reservations. No waits while retaining unadmitted host batches.
use ::datafusion::{
    arrow::array::ArrayData,
    common::{DataFusionError, Result},
};
use arrow_array::{RecordBatch, RecordBatchOptions, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use std::{
    collections::BTreeMap,
    error::Error,
    fmt,
    ptr::NonNull,
    sync::{Arc, Mutex},
};
use tokio::sync::Semaphore;

/// Logical adapter limits, excluding upstream caches and backend/driver caches.
/// Admission fails rather than waiting for bytes held by a downstream consumer.
#[derive(Clone, Copy, Debug)]
pub struct SharedLimits {
    pub max_device_bytes: u64,
    pub max_host_bytes: u64,
    pub max_requests: u64,
    pub max_streams: u64,
}
impl Default for SharedLimits {
    fn default() -> Self {
        Self {
            max_device_bytes: 256 * 1024 * 1024,
            max_host_bytes: 512 * 1024 * 1024,
            max_requests: 16,
            max_streams: 64,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SharedResourceError {
    Exhausted {
        resource: &'static str,
        required: u64,
        limit: u64,
    },
    EstimateOverflow,
    Poisoned,
}
impl fmt::Display for SharedResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted {
                resource,
                required,
                limit,
            } => write!(f, "shared {resource} needs {required}, limit is {limit}"),
            Self::EstimateOverflow => write!(f, "shared resource estimate overflow"),
            Self::Poisoned => write!(
                f,
                "shared resource group is poisoned after failed backend drain"
            ),
        }
    }
}
impl Error for SharedResourceError {}
fn error(e: SharedResourceError) -> DataFusionError {
    DataFusionError::External(Box::new(e))
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| error(SharedResourceError::EstimateOverflow))
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| error(SharedResourceError::EstimateOverflow))
}
fn check(resource: &'static str, required: u64, limit: u64) -> Result<()> {
    if required > limit {
        Err(error(SharedResourceError::Exhausted {
            resource,
            required,
            limit,
        }))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceSnapshot {
    pub active_streams: u64,
    pub active_requests: u64,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub peak_device_bytes: u64,
    pub peak_host_bytes: u64,
    pub peak_requests: u64,
    pub completed_requests: u64,
    pub canceled_streams: u64,
    pub failed_streams: u64,
    pub rejected_reservations: u64,
    pub poisoned: bool,
}
#[derive(Default)]
struct State {
    snapshot: ResourceSnapshot,
    scratch: u64,
    buffers: BTreeMap<usize, (u64, u64)>,
}
impl State {
    fn host(&self) -> u64 {
        self.scratch + self.buffers.values().map(|(bytes, _)| *bytes).sum::<u64>()
    }
    fn update(&mut self) {
        self.snapshot.host_bytes = self.host();
        self.snapshot.peak_host_bytes = self.snapshot.peak_host_bytes.max(self.snapshot.host_bytes);
        self.snapshot.peak_device_bytes = self
            .snapshot
            .peak_device_bytes
            .max(self.snapshot.device_bytes);
        self.snapshot.peak_requests = self
            .snapshot
            .peak_requests
            .max(self.snapshot.active_requests);
    }
    fn extra(&self, buffers: &BTreeMap<usize, u64>) -> Result<u64> {
        buffers.iter().try_fold(0, |sum, (id, bytes)| {
            add(
                sum,
                bytes.saturating_sub(self.buffers.get(id).map_or(0, |e| e.0)),
            )
        })
    }
    fn retain(&mut self, buffers: &BTreeMap<usize, u64>) {
        for (id, bytes) in buffers {
            let entry = self.buffers.entry(*id).or_default();
            entry.0 = entry.0.max(*bytes);
            entry.1 += 1;
        }
    }
    fn release(&mut self, ids: &[usize]) {
        for id in ids {
            let entry = self.buffers.get_mut(id).expect("live reservation");
            entry.1 -= 1;
            if entry.1 == 0 {
                self.buffers.remove(id);
            }
        }
    }
}
struct Inner {
    limits: SharedLimits,
    state: Mutex<State>,
    gate: Arc<Semaphore>,
}
/// Clone this controller to share bytes/admission across separately registered
/// adapters. Options clones always share it. Independently created groups do not.
#[derive(Clone)]
pub struct SharedResources(Arc<Inner>);
impl fmt::Debug for SharedResources {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedResources")
            .field("limits", &self.limits())
            .field("snapshot", &self.snapshot())
            .finish()
    }
}
impl Default for SharedResources {
    fn default() -> Self {
        Self::new(SharedLimits::default())
    }
}
impl SharedResources {
    pub fn new(limits: SharedLimits) -> Self {
        Self(Arc::new(Inner {
            limits,
            state: Mutex::new(State::default()),
            gate: Arc::new(Semaphore::new(1)),
        }))
    }
    pub fn limits(&self) -> SharedLimits {
        self.0.limits
    }
    pub fn snapshot(&self) -> ResourceSnapshot {
        self.0.state.lock().unwrap().snapshot
    }
    pub(crate) fn gate(&self) -> Arc<Semaphore> {
        self.0.gate.clone()
    }
    pub(crate) fn poison(&self) {
        self.0.state.lock().unwrap().snapshot.poisoned = true;
    }
    pub(crate) fn ensure_healthy(&self) -> Result<()> {
        if self.snapshot().poisoned {
            Err(error(SharedResourceError::Poisoned))
        } else {
            Ok(())
        }
    }
    fn reserve(&self, buffers: &BTreeMap<usize, u64>, scratch: u64, stream: bool) -> Result<()> {
        let mut state = self.0.state.lock().unwrap();
        let result = (|| {
            if state.snapshot.poisoned {
                return Err(error(SharedResourceError::Poisoned));
            }
            if stream {
                check(
                    "streams",
                    add(state.snapshot.active_streams, 1)?,
                    self.0.limits.max_streams,
                )?;
            } else {
                check(
                    "requests",
                    add(state.snapshot.active_requests, 1)?,
                    self.0.limits.max_requests,
                )?;
            }
            check(
                "host_bytes",
                add(add(state.host(), state.extra(buffers)?)?, scratch)?,
                self.0.limits.max_host_bytes,
            )
        })();
        if result.is_err() {
            state.snapshot.rejected_reservations += 1;
            return result;
        }
        state.retain(buffers);
        state.scratch += scratch;
        if stream {
            state.snapshot.active_streams += 1;
        } else {
            state.snapshot.active_requests += 1;
        }
        state.update();
        Ok(())
    }
    pub(crate) fn stream(&self, outputs: usize) -> Result<StreamLease> {
        let scratch = add(4096, mul(outputs as u64, 128)?)?;
        self.reserve(&BTreeMap::new(), scratch, true)?;
        Ok(StreamLease {
            resources: self.clone(),
            scratch,
            finished: false,
            failed: false,
        })
    }
    pub(crate) fn request(&self, batch: &RecordBatch, scratch: u64) -> Result<RequestLease> {
        let buffers = batch_buffers(batch);
        self.reserve(&buffers, scratch, false)?;
        Ok(RequestLease {
            resources: self.clone(),
            scratch,
            ids: buffers.keys().copied().collect(),
            device: 0,
            completed: false,
        })
    }
    fn output(&self, batch: RecordBatch, scratch: &mut u64) -> Result<RecordBatch> {
        let buffers = batch_buffers(&batch);
        {
            let mut state = self.0.state.lock().unwrap();
            let extra = state.extra(&buffers)?;
            // Convert existing workspace credit atomically into output ownership.
            if let Err(e) = check("output_estimate", extra, *scratch) {
                state.snapshot.rejected_reservations += 1;
                return Err(e);
            }
            *scratch -= extra;
            state.scratch -= extra;
            state.retain(&buffers);
            state.update();
        }
        let lease = Arc::new(OutputLease {
            resources: self.clone(),
            ids: buffers.keys().copied().collect(),
        });
        let arrays = batch
            .columns()
            .iter()
            .map(|a| wrap_data(a.to_data(), &lease).map(make_array))
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new_with_options(
            batch.schema(),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
        )
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
    }
}
/// Conservative adapter-owned working planes: bitmap expansion, upload copies,
/// readback/packing plus small metadata. This is not a physical driver peak.
pub(crate) fn workspace(rows: usize, inputs: usize, outputs: usize) -> Result<u64> {
    add(
        add(4096, mul(add(inputs as u64, outputs as u64)?, 1024)?)?,
        add(
            mul(mul(rows as u64, inputs as u64)?, 32)?,
            mul(mul(rows as u64, outputs as u64)?, 32)?,
        )?,
    )
}
pub(crate) struct StreamLease {
    resources: SharedResources,
    scratch: u64,
    finished: bool,
    failed: bool,
}
impl StreamLease {
    pub(crate) fn finish(&mut self, failed: bool) {
        self.finished = true;
        self.failed = failed;
    }
    pub(crate) fn output(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        self.resources.output(batch, &mut self.scratch)
    }
}
impl Drop for StreamLease {
    fn drop(&mut self) {
        let mut state = self.resources.0.state.lock().unwrap();
        state.scratch -= self.scratch;
        state.snapshot.active_streams -= 1;
        if !self.finished {
            state.snapshot.canceled_streams += 1;
        }
        if self.failed {
            state.snapshot.failed_streams += 1;
        }
        state.update();
    }
}
pub(crate) struct RequestLease {
    resources: SharedResources,
    scratch: u64,
    ids: Vec<usize>,
    device: u64,
    completed: bool,
}
impl RequestLease {
    pub(crate) fn device(&mut self, bytes: u64) -> Result<()> {
        let mut state = self.resources.0.state.lock().unwrap();
        let required = add(state.snapshot.device_bytes - self.device, bytes)?;
        if let Err(e) = check(
            "device_bytes",
            required,
            self.resources.0.limits.max_device_bytes,
        ) {
            state.snapshot.rejected_reservations += 1;
            return Err(e);
        }
        state.snapshot.device_bytes = required;
        self.device = bytes;
        state.update();
        Ok(())
    }
    pub(crate) fn output(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        self.resources.output(batch, &mut self.scratch)
    }
    pub(crate) fn complete(&mut self) {
        self.completed = true;
    }
}
impl Drop for RequestLease {
    fn drop(&mut self) {
        let mut state = self.resources.0.state.lock().unwrap();
        state.scratch -= self.scratch;
        state.release(&self.ids);
        state.snapshot.device_bytes -= self.device;
        state.snapshot.active_requests -= 1;
        if self.completed {
            state.snapshot.completed_requests += 1;
        }
        state.update();
    }
}
struct OutputLease {
    resources: SharedResources,
    ids: Vec<usize>,
}
impl Drop for OutputLease {
    fn drop(&mut self) {
        let mut state = self.resources.0.state.lock().unwrap();
        state.release(&self.ids);
        state.update();
    }
}

fn batch_buffers(batch: &RecordBatch) -> BTreeMap<usize, u64> {
    let mut buffers = BTreeMap::new();
    let mut pending = batch
        .columns()
        .iter()
        .map(|a| a.to_data())
        .collect::<Vec<_>>();
    while let Some(data) = pending.pop() {
        for b in data
            .buffers()
            .iter()
            .chain(data.nulls().map(|n| n.inner().inner()))
        {
            let bytes = b.capacity().max(b.len() + b.ptr_offset()) as u64;
            if bytes != 0 {
                let entry = buffers
                    .entry(b.data_ptr().as_ptr() as usize)
                    .or_insert(0_u64);
                *entry = (*entry).max(bytes);
            }
        }
        pending.extend(data.child_data().iter().cloned());
    }
    buffers
}
// The owner pins both original allocation and reservation. Rebuilding ArrayData
// preserves concrete Arrow array types, offsets and validity without copies.
struct BufferOwner {
    _buffer: Buffer,
    _lease: Arc<OutputLease>,
}
fn wrap_buffer(buffer: Buffer, lease: &Arc<OutputLease>) -> Buffer {
    let ptr: NonNull<u8> = buffer.data_ptr();
    let len = buffer.len();
    let offset = buffer.ptr_offset();
    let owner = Arc::new(BufferOwner {
        _buffer: buffer,
        _lease: lease.clone(),
    });
    // SAFETY: the owner retains the immutable original Buffer for the entire
    // custom allocation lifetime. The original Bytes contains the initialized
    // prefix through offset+len; preserving its base also preserves dedup IDs.
    unsafe { Buffer::from_custom_allocation(ptr, offset + len, owner) }
        .slice_with_length(offset, len)
}
fn wrap_data(data: ArrayData, lease: &Arc<OutputLease>) -> Result<ArrayData> {
    let buffers = data
        .buffers()
        .iter()
        .map(|b| wrap_buffer(b.clone(), lease))
        .collect();
    let nulls = data.nulls().map(|n| {
        NullBuffer::new(BooleanBuffer::new(
            wrap_buffer(n.inner().inner().clone(), lease),
            n.offset(),
            n.len(),
        ))
    });
    let children = data
        .child_data()
        .iter()
        .cloned()
        .map(|d| wrap_data(d, lease))
        .collect::<Result<Vec<_>>>()?;
    data.into_builder()
        .buffers(buffers)
        .nulls(nulls)
        .child_data(children)
        .build()
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))
}

#[cfg(test)]
mod tests {
    use super::{SharedLimits, SharedResources, batch_buffers, workspace};
    use arrow_array::{Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;
    fn batch() -> RecordBatch {
        let a = Arc::new(Int32Array::from(vec![Some(8), None, Some(30), Some(40)]));
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ])),
            vec![a.clone(), a],
        )
        .unwrap()
        .slice(1, 3)
    }
    #[test]
    fn dedup_and_arrow_output_ownership_survive_slices_and_clones() {
        let resources = SharedResources::default();
        let b = batch();
        let bytes = batch_buffers(&b).values().sum::<u64>();
        let mut first = resources.request(&b, 4096).unwrap();
        let second = resources.request(&b, 0).unwrap();
        assert_eq!(resources.snapshot().host_bytes, bytes + 4096);
        let result = first.output(b.clone()).unwrap();
        let nested = resources.request(&result, 0).unwrap();
        assert_eq!(
            resources.snapshot().host_bytes,
            bytes + 4096,
            "rebound sliced outputs retain allocation identity"
        );
        drop(nested);
        let slice = result.column(0).slice(1, 2);
        assert_eq!(
            slice
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some(30), Some(40)]
        );
        assert!(result.column(0).is_null(0));
        drop(result);
        drop(first);
        drop(second);
        drop(b);
        assert_eq!(resources.snapshot().host_bytes, bytes);
        drop(slice);
        assert_eq!(resources.snapshot().host_bytes, 0);
        assert_eq!(resources.snapshot().active_requests, 0);
    }
    #[test]
    fn failed_reservations_are_atomic_and_do_not_release_other_requests() {
        let resources = SharedResources::new(SharedLimits {
            max_device_bytes: 100,
            max_host_bytes: 8192,
            max_requests: 2,
            max_streams: 1,
        });
        let b = batch();
        let mut first = resources.request(&b, 0).unwrap();
        first.device(80).unwrap();
        let mut second = resources.request(&b, 0).unwrap();
        assert!(second.device(30).is_err());
        assert_eq!(resources.snapshot().device_bytes, 80);
        assert!(resources.request(&b, 0).is_err());
        assert!(resources.request(&b, 9000).is_err());
        drop(first);
        assert_eq!(resources.snapshot().active_requests, 1);
        second.device(100).unwrap();
        drop(second);
        assert_eq!(resources.snapshot().device_bytes, 0);
        assert_eq!(resources.snapshot().host_bytes, 0);
        let stream = resources.stream(1).unwrap();
        assert!(resources.stream(1).is_err());
        drop(stream);
        assert_eq!(resources.snapshot().active_streams, 0);
        assert!(workspace(usize::MAX, usize::MAX, 32).is_err());
    }
}
