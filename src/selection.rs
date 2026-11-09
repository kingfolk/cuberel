//! Stable row selection with an explicit device-to-host completion boundary.

use std::sync::Arc;

use crate::completion::{ExitContext, synchronize};
use cubecl::{prelude::Client, server::Handle};

use crate::{
    Batch, DataType, DeviceBatch, ExecutionError, Executor, KernelProgram, Parameters,
    abi::{BindingError, BindingLayout},
    executor::SubmittedExpression,
    resident::ExecutionCounters,
    selection_kernel::{SelectionKernel, Stage},
};

/// Logical allocation costs, excluding runtime metadata, allocator caches and
/// expression inputs/outputs. Not a driver memory-peak measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SelectionStats {
    pub scan_levels: usize,
    /// Scan, offset propagation and scatter launches; excludes the predicate.
    pub kernel_launches: usize,
    pub workspace_bytes: u64,
    pub indices_capacity_bytes: u64,
}

/// Owned, pending stable selection. Capacity is the source row count, not the
/// selected length. Only [`Self::materialize`] exposes completed host indices.
/// Clones share immutable buffers and the same source identity.
/// No raw handles or uninitialized capacity tail are exposed by the safe API.
#[derive(Clone)]
pub struct DeviceSelection {
    client: Client,
    counters: Arc<ExecutionCounters>,
    capacity: u32,
    source: Arc<()>,
    completion_id: Arc<()>,
    indices: Option<Handle>,
    count: Option<Handle>,
    // Also retain every intermediate and input until completion or runtime-safe
    // drop. CubeCL's queued bindings own their memory if this object is dropped.
    retained: Vec<Handle>,
    predicate: Option<Arc<SubmittedExpression>>,
    dependency: Option<Arc<crate::DeviceProjection>>,
    stats: SelectionStats,
}

impl DeviceSelection {
    pub(crate) fn indices_handle(&self) -> Handle {
        self.indices.clone().expect("nonempty selection")
    }
    pub(crate) fn count_handle(&self) -> Handle {
        self.count.clone().expect("nonempty selection")
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn stats(&self) -> SelectionStats {
        self.stats
    }

    /// Selections from the same resident snapshot (and selection clones) share
    /// a source. Separate host select/compact calls and uploads have fresh identities.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source, &other.source)
    }

    /// Whether this selection uses the original row coordinates of this batch.
    pub fn belongs_to(&self, batch: &DeviceBatch) -> bool {
        Arc::ptr_eq(&self.source, &batch.source)
    }

    /// Complete all stages and predicate error checks, then read the device
    /// count. This is an explicit blocking boundary, without reading indices.
    /// Each call repeats synchronization/count/error readback.
    pub fn selected_count(&self) -> Result<u32, ExecutionError> {
        synchronize(&self.client, &self.handles())?;
        ExitContext::default().count(self)
    }
    pub(crate) fn handles(&self) -> Vec<Handle> {
        self.retained.clone()
    }
    pub(crate) fn count_in(&self, exit: &mut ExitContext) -> Result<u32, ExecutionError> {
        let key = Arc::as_ptr(&self.completion_id) as usize;
        if let Some((_, count)) = exit.counts.iter().find(|(id, _)| *id == key) {
            return Ok(*count);
        }
        if self.capacity == 0 {
            return Ok(0);
        }
        if let Some(dependency) = &self.dependency {
            dependency.complete_in(exit)?;
        }
        if let Some(predicate) = &self.predicate {
            predicate.check_errors(&self.client, &self.counters)?;
        }
        self.counters
            .count_reads
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bytes =
            self.counters
                .read(&self.client, self.count.clone().expect("nonempty count"), 4)?;
        let count = u32::from_ne_bytes(bytes.as_chunks::<4>().0[0]);
        if count > self.capacity {
            return Err(ExecutionError::InvalidSelectionCount {
                count,
                capacity: self.capacity,
            });
        }
        exit.counts.push((key, count));
        Ok(count)
    }

    /// Wait for every stage (including predicate error/validity planes), check
    /// checked-I32 errors, then read count and only the initialized index prefix.
    /// Repeated calls are valid but repeat synchronization and readback.
    /// CubeCL allocation/transport APIs may panic, as in the existing executor.
    pub fn materialize(&self) -> Result<Vec<u32>, ExecutionError> {
        let count = self.selected_count()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let handle = self.indices.clone().expect("nonempty indices");
        // Narrow the logical handle, never read the uninitialized capacity tail.
        let handle = handle.offset_end(u64::from(self.capacity - count) * 4);
        let bytes = self
            .counters
            .read(&self.client, handle, count as usize * 4)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_ne_bytes(*b))
            .collect())
    }
}

impl Executor {
    /// Execute a Bool predicate with WHERE semantics and compact it on device.
    /// NULL/false are discarded; selected original row IDs are strictly ordered.
    /// Submission does not read predicate values, indices or count to the host.
    /// I32 predicate overflow and backend failures surface at materialize().
    pub fn select(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<DeviceSelection, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        let layout = BindingLayout::new(program);
        let host = layout.bind_host(batch, parameters)?;
        let plan = SelectionPlan::new(&self.client, host.extent().row_count(), self.cube_size())?;
        let predicate = Arc::new(self.submit_expression(program, batch, parameters)?);
        self.selection_from_predicate(plan, predicate, Arc::new(()))
    }

    /// Select from an existing immutable snapshot without uploading columns or
    /// reading predicate values/count. Selections preserve the batch identity.
    pub fn select_resident(
        &self,
        program: &KernelProgram,
        batch: &DeviceBatch,
        parameters: &Parameters,
    ) -> Result<DeviceSelection, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        batch.validate(&self.client, &BindingLayout::new(program), parameters)?;
        let plan = SelectionPlan::new(&self.client, batch.row_count(), self.cube_size())?;
        let predicate = Arc::new(self.submit_resident_expression(program, batch, parameters)?);
        self.selection_from_predicate(plan, predicate, batch.source.clone())
    }

    fn selection_from_predicate(
        &self,
        plan: SelectionPlan,
        predicate: Arc<SubmittedExpression>,
        source: Arc<()>,
    ) -> Result<DeviceSelection, ExecutionError> {
        // The expression emitter canonicalizes invalid Bool payload to false,
        // so its value plane is already exactly the WHERE mask.
        let mut selection = self.submit_selection(
            plan,
            predicate.values.clone(),
            predicate.retained.clone(),
            Some(predicate),
            None,
        )?;
        selection.source = source;
        Ok(selection)
    }

    /// Upload a validated 0/1 mask and produce a pending device selection.
    /// Invalid input is rejected before any allocation or launch.
    pub fn compact_mask(&self, mask: &[u32]) -> Result<DeviceSelection, ExecutionError> {
        let rows = checked_rows(mask.len() as u64)?;
        let plan = SelectionPlan::new(&self.client, rows, self.cube_size())?;
        for (index, &value) in mask.iter().enumerate() {
            if value > 1 {
                return Err(ExecutionError::InvalidMaskValue { index, value });
            }
        }
        let handle = (rows != 0).then(|| {
            let bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_ne_bytes()).collect();
            self.upload_bytes(&bytes)
        });
        let retained = handle.iter().cloned().collect();
        self.submit_selection(plan, handle, retained, None, None)
    }

    /// Apply another WHERE only to currently selected rows. Results keep stable
    /// original batch row IDs; parent predicate/backend failures remain observable.
    pub fn filter_selected(
        &self,
        program: &KernelProgram,
        batch: &DeviceBatch,
        selected: &DeviceSelection,
        parameters: &Parameters,
    ) -> Result<DeviceSelection, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        let plan = SelectionPlan::new(&self.client, batch.row_count(), self.cube_size())?;
        if batch.row_count() != 0 && self.client.properties().hardware.max_bindings < 7 {
            return Err(ExecutionError::UnsupportedDevice {
                reason: "nested selection needs 7 buffer bindings including metadata",
            });
        }
        let predicate = Arc::new(self.project_selected(
            batch,
            selected,
            &[("_predicate", program)],
            parameters,
        )?);
        let submitted = &predicate.outputs[0].submitted;
        let mut retained = selected.retained.clone();
        retained.extend(submitted.retained.iter().cloned());
        let mut selection = self.submit_selection(
            plan,
            submitted.values.clone(),
            retained,
            None,
            Some(selected),
        )?;
        selection.source = batch.source.clone();
        selection.dependency = Some(predicate);
        Ok(selection)
    }

    fn submit_selection(
        &self,
        plan: SelectionPlan,
        mask: Option<Handle>,
        mut retained: Vec<Handle>,
        predicate: Option<Arc<SubmittedExpression>>,
        selected: Option<&DeviceSelection>,
    ) -> Result<DeviceSelection, ExecutionError> {
        let mut selection = DeviceSelection {
            client: self.client.clone(),
            counters: self.counters.clone(),
            capacity: plan.rows,
            source: Arc::new(()),
            completion_id: Arc::new(()),
            indices: None,
            count: None,
            retained: Vec::new(),
            predicate,
            dependency: None,
            stats: plan.stats,
        };
        if plan.rows == 0 {
            return Ok(selection);
        }
        let mask = mask.expect("validated nonempty mask");
        let mut levels: Vec<(Handle, Handle)> = Vec::with_capacity(plan.levels.len());
        let mut input = mask.clone();
        for (level, &len) in plan.levels.iter().enumerate() {
            let ranks = self.client.empty(len as usize * 4);
            let totals = self
                .client
                .empty(len.div_ceil(self.cube_size()) as usize * 4);
            if level == 0
                && let Some(selected) = selected
            {
                self.selection_launch(
                    Stage::ScanSelected,
                    len,
                    &[
                        input,
                        ranks.clone(),
                        totals.clone(),
                        selected.count_handle(),
                    ],
                );
            } else {
                self.selection_launch(Stage::Scan, len, &[input, ranks.clone(), totals.clone()]);
            }
            retained.extend([ranks.clone(), totals.clone()]);
            input = totals.clone();
            levels.push((ranks, totals));
        }
        // Each parent rank becomes global before its child consumes it.
        for level in (0..levels.len() - 1).rev() {
            self.selection_launch(
                Stage::AddOffsets,
                plan.levels[level],
                &[levels[level + 1].0.clone(), levels[level].0.clone()],
            );
        }
        let count = levels.last().expect("scan levels").1.clone();
        let indices = self.client.empty(plan.rows as usize * 4);
        // Scatter is a device consumer of count: no host count read is needed
        // to schedule it or to bounds-check its selected output positions.
        if let Some(selected) = selected {
            self.selection_launch(
                Stage::ScatterSelected,
                plan.rows,
                &[
                    mask,
                    levels[0].0.clone(),
                    count.clone(),
                    indices.clone(),
                    selected.indices_handle(),
                    selected.count_handle(),
                ],
            );
        } else {
            self.selection_launch(
                Stage::Scatter,
                plan.rows,
                &[mask, levels[0].0.clone(), count.clone(), indices.clone()],
            );
        }
        retained.push(indices.clone());
        selection.indices = Some(indices);
        selection.count = Some(count);
        selection.retained = retained;
        Ok(selection)
    }

    pub(crate) fn selection_plan(&self, rows: u32) -> Result<SelectionStats, ExecutionError> {
        Ok(SelectionPlan::new(&self.client, rows, self.cube_size())?.stats)
    }
    fn selection_launch(&self, stage: Stage, rows: u32, buffers: &[Handle]) {
        let kernel = SelectionKernel::new(
            stage,
            self.client.clone(),
            self.settings(),
            self.define_count.clone(),
        );
        // All layouts, grids and capabilities were preflighted before upload.
        // Handles are private, initialized inputs/fresh outputs, same service,
        // with exact lengths prescribed by the stage layout.
        kernel.launch(rows, buffers);
        self.counters
            .launches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

struct SelectionPlan {
    rows: u32,
    levels: Vec<u32>,
    stats: SelectionStats,
}

impl SelectionPlan {
    fn new(client: &Client, rows: u32, tile: u32) -> Result<Self, ExecutionError> {
        checked_rows(u64::from(rows))?;
        let mut levels = Vec::new();
        let mut stats = SelectionStats::default();
        if rows != 0 {
            let hw = &client.properties().hardware;
            if tile < 2 || tile as usize * 4 > hw.max_shared_memory_size || hw.max_bindings < 5 {
                return Err(ExecutionError::UnsupportedDevice {
                    reason: "selection needs >=2 lanes, tile*4 shared bytes and 5 buffer bindings",
                });
            }
            let mut len = rows;
            loop {
                let blocks = len.div_ceil(tile);
                if blocks > hw.max_cube_count.0
                    || hw.max_cube_count.1 == 0
                    || hw.max_cube_count.2 == 0
                {
                    return Err(ExecutionError::GridTooLarge {
                        required: blocks,
                        max: hw.max_cube_count.0,
                    });
                }
                levels.push(len);
                stats.workspace_bytes += (u64::from(len) + u64::from(blocks)) * 4;
                if blocks == 1 {
                    break;
                }
                len = blocks;
            }
            stats.scan_levels = levels.len();
            stats.kernel_launches = levels.len() * 2;
            stats.indices_capacity_bytes = u64::from(rows) * 4;
        }
        Ok(Self {
            rows,
            levels,
            stats,
        })
    }
}

fn checked_rows(rows: u64) -> Result<u32, ExecutionError> {
    if rows > u64::from(u32::MAX) / 4 {
        return Err(BindingError::AddressOverflow {
            field: "selection_byte_length",
            value: rows.saturating_mul(4),
        }
        .into());
    }
    Ok(rows as u32)
}

#[cfg(test)]
mod tests {
    use super::checked_rows;

    #[test]
    fn selection_extent_is_checked_without_large_allocations() {
        let max = u64::from(u32::MAX) / 4;
        assert_eq!(checked_rows(max).unwrap(), max as u32);
        assert!(checked_rows(max + 1).is_err());
        assert!(checked_rows(u64::MAX).is_err());
    }
}

// Backend failure fixture is intentionally rejected before lowering. It never
// runs malformed instructions or reads uninitialized/OOB device memory.
#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
pub(crate) mod completion_tests {
    use crate::{ExecutionError, Executor, cubecl_abi::LauncherGuard};
    use cubecl::{
        Device,
        ir::{UIntKind, dialect::branch::ReturnOp},
        prelude::{
            BufferArg, BufferIOAttr, Client, CubeCount, CubeKernel, ElemType, KernelBuilder,
            KernelDefinition, KernelId, KernelLauncher, KernelMetadata, KernelSettings,
        },
    };

    struct RejectedOutput {
        client: Client,
        settings: KernelSettings,
    }
    impl KernelMetadata for RejectedOutput {
        fn id(&self) -> KernelId {
            KernelId::new::<Self>()
                .cube_dim(self.settings.cube_dim)
                .address_type(self.settings.address_type)
                .mode(self.settings.execution_mode)
        }
        fn address_type(&self) -> ElemType {
            self.settings.address_type.unsigned_type()
        }
    }
    impl CubeKernel for RejectedOutput {
        fn define(&self) -> KernelDefinition {
            let mut builder = KernelBuilder::new(self.settings.clone());
            builder.runtime_properties(self.client.target_properties().clone());
            builder.device_properties(self.client.properties());
            let ty = ElemType::UInt(UIntKind::U32).to_type(builder.scope.ctx());
            builder.buffer(ty);
            builder
                .scope
                .push_error("intentional selection completion fixture");
            builder
                .scope
                .register(&ReturnOp::new(builder.scope.ctx_mut()));
            builder.build()
        }
    }
    pub(crate) fn rejected_handle(
        client: &Client,
        settings: KernelSettings,
    ) -> cubecl::server::Handle {
        let marker = client.empty(4);
        reject_output(client, settings, marker.clone());
        marker
    }
    pub(crate) fn reject_output(
        client: &Client,
        settings: KernelSettings,
        marker: cubecl::server::Handle,
    ) {
        let kernel = RejectedOutput {
            client: client.clone(),
            settings: settings.clone(),
        };
        let mut guard = LauncherGuard(Some(KernelLauncher::new(settings)));
        let launcher = guard.0.as_mut().unwrap();
        launcher.declare_io(BufferIOAttr::WriteOnly);
        // SAFETY: fresh write-only U32 allocation of exactly one element.
        launcher.register_buffer(unsafe { BufferArg::from_raw_parts(marker.clone(), 1) }, 4);
        guard
            .0
            .take()
            .unwrap()
            .launch(CubeCount::Static(1, 1, 1), kernel, client);
    }
    fn exercise(device: Device) {
        let client = device.client();
        let executor = Executor::new(client.clone()).unwrap();
        let mut selection = executor.compact_mask(&[1, 0, 1]).unwrap();
        // Both count and indices really succeeded. A different retained output
        // then fails, so observing just count cannot establish overall success.
        assert_eq!(selection.materialize().unwrap(), [0, 2]);
        let mut bad_count = selection.clone();
        let counter = client.create_from_slice(&4_u32.to_ne_bytes());
        bad_count.count = Some(counter.clone());
        bad_count.retained.push(counter);
        assert!(matches!(
            bad_count.materialize(),
            Err(ExecutionError::InvalidSelectionCount {
                count: 4,
                capacity: 3
            })
        ));
        let marker = rejected_handle(&client, executor.settings());
        selection.retained.push(marker);
        assert!(matches!(
            selection.selected_count(),
            Err(ExecutionError::Backend(_))
        ));
        assert!(matches!(
            selection.materialize(),
            Err(ExecutionError::Backend(_))
        ));
        assert_eq!(
            executor
                .compact_mask(&[0, 1])
                .unwrap()
                .materialize()
                .unwrap(),
            [1]
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_retained_failure_blocks_success() {
        exercise(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_retained_failure_blocks_success() {
        exercise(Device::metal(Default::default()).expect("native Metal required"));
    }
}
