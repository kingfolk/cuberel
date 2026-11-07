//! Stable row selection with an explicit device-to-host completion boundary.

use std::sync::Arc;

use cubecl::{future::block_on, prelude::Client, server::Handle};

use crate::{
    Batch, DataType, ExecutionError, Executor, KernelProgram, Parameters,
    abi::{BindingError, BindingLayout},
    executor::{SubmittedExpression, read_plane},
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
    capacity: u32,
    source: Arc<()>,
    indices: Option<Handle>,
    count: Option<Handle>,
    // Also retain every intermediate and input until completion or runtime-safe
    // drop. CubeCL's queued bindings own their memory if this object is dropped.
    retained: Vec<Handle>,
    predicate: Option<Arc<SubmittedExpression>>,
    stats: SelectionStats,
}

impl DeviceSelection {
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn stats(&self) -> SelectionStats {
        self.stats
    }

    /// Equal only for clones of the same source snapshot. Two separately
    /// uploaded equal masks/batches are distinct sources.
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source, &other.source)
    }

    /// Complete all stages and predicate error checks, then read the device
    /// count. This is an explicit blocking boundary, without reading indices.
    /// Each call repeats synchronization/count/error readback.
    pub fn selected_count(&self) -> Result<u32, ExecutionError> {
        if self.capacity == 0 {
            return Ok(0);
        }
        block_on(self.client.sync_buffers(self.retained.iter()))
            .map_err(ExecutionError::Backend)?;
        if let Some(predicate) = &self.predicate {
            predicate.check_errors(&self.client)?;
        }
        let bytes = read_plane(&self.client, self.count.clone().expect("nonempty count"), 4)?;
        let count = u32::from_ne_bytes(bytes.as_chunks::<4>().0[0]);
        if count > self.capacity {
            return Err(ExecutionError::InvalidSelectionCount {
                count,
                capacity: self.capacity,
            });
        }
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
        let bytes = read_plane(&self.client, handle, count as usize * 4)?;
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
        // The expression emitter canonicalizes invalid Bool payload to false,
        // so its value plane is already exactly the WHERE mask.
        self.submit_selection(
            plan,
            predicate.values.clone(),
            predicate.retained.clone(),
            Some(predicate),
        )
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
            self.client.create_from_slice(&bytes)
        });
        let retained = handle.iter().cloned().collect();
        self.submit_selection(plan, handle, retained, None)
    }

    fn submit_selection(
        &self,
        plan: SelectionPlan,
        mask: Option<Handle>,
        mut retained: Vec<Handle>,
        predicate: Option<Arc<SubmittedExpression>>,
    ) -> Result<DeviceSelection, ExecutionError> {
        let mut selection = DeviceSelection {
            client: self.client.clone(),
            capacity: plan.rows,
            source: Arc::new(()),
            indices: None,
            count: None,
            retained: Vec::new(),
            predicate,
            stats: plan.stats,
        };
        if plan.rows == 0 {
            return Ok(selection);
        }
        let mask = mask.expect("validated nonempty mask");
        let mut levels: Vec<(Handle, Handle)> = Vec::with_capacity(plan.levels.len());
        let mut input = mask.clone();
        for &len in &plan.levels {
            let ranks = self.client.empty(len as usize * 4);
            let totals = self
                .client
                .empty(len.div_ceil(self.cube_size()) as usize * 4);
            self.selection_launch(Stage::Scan, len, &[input, ranks.clone(), totals.clone()]);
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
        self.selection_launch(
            Stage::Scatter,
            plan.rows,
            &[mask, levels[0].0.clone(), count.clone(), indices.clone()],
        );
        retained.push(indices.clone());
        selection.indices = Some(indices);
        selection.count = Some(count);
        selection.retained = retained;
        Ok(selection)
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
mod completion_tests {
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
        let marker = client.empty(4);
        let kernel = RejectedOutput {
            client: client.clone(),
            settings: executor.settings(),
        };
        let mut guard = LauncherGuard(Some(KernelLauncher::new(executor.settings())));
        let launcher = guard.0.as_mut().unwrap();
        launcher.declare_io(BufferIOAttr::WriteOnly);
        // SAFETY: fresh write-only U32 allocation of exactly one element.
        launcher.register_buffer(unsafe { BufferArg::from_raw_parts(marker.clone(), 1) }, 4);
        guard
            .0
            .take()
            .unwrap()
            .launch(CubeCount::Static(1, 1, 1), kernel, &client);
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
