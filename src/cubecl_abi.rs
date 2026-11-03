//! CubeCL adapter for the versioned layout. This module declares and binds
//! arguments, but does not lower query expressions or cache query kernels.

use cubecl::{
    ir::{
        FloatKind, IntKind, UIntKind,
        dialect::general::{CastOp, ReadScalarOp},
        pliron::value::Value,
    },
    prelude::{
        BufferArg, BufferIOAttr, Client, CubeCount, CubeKernel, ElemType, KernelBuilder,
        KernelDefinition, KernelLauncher, KernelSettings, Scope,
    },
    server::Handle,
};

use crate::{
    ColumnId, ParameterId, Parameters,
    abi::{
        BindingError, BindingLayout, BufferAccess, BufferDescriptor, BufferExtent, BufferSource,
        PhysicalType, ScalarArgument, ScalarSource,
    },
};

impl PhysicalType {
    fn elem_type(self) -> ElemType {
        match self {
            Self::F32 => FloatKind::F32.into(),
            Self::U32 => UIntKind::U32.into(),
            Self::I32 => IntKind::I32.into(),
        }
    }
}

/// Context-local declaration. Do not store its values in a persistent program.
pub struct KernelDeclaration {
    layout: BindingLayout,
    builder: KernelBuilder,
    buffers: Vec<Value>,
    parameters: Vec<(ParameterId, Value)>,
    row_count_raw: Value,
    row_count: Value,
}

impl KernelDeclaration {
    pub fn layout(&self) -> &BindingLayout {
        &self.layout
    }
    pub fn scope(&self) -> &Scope {
        &self.builder.scope
    }
    pub fn column(&self, id: ColumnId) -> Option<Value> {
        self.layout
            .buffers()
            .iter()
            .zip(&self.buffers)
            .find(|(slot, _)| slot.source() == BufferSource::Column(id))
            .map(|(_, value)| *value)
    }
    pub fn parameter(&self, id: ParameterId) -> Option<Value> {
        self.parameters
            .iter()
            .find(|(parameter, _)| *parameter == id)
            .map(|(_, value)| *value)
    }
    pub fn output(&self) -> Value {
        *self.buffers.last().expect("layout always has an output")
    }
    /// Abstract IndexType value, suitable for index comparisons and memory operations.
    pub fn row_count(&self) -> Value {
        self.row_count
    }
    /// Physical U32 read; its explicit conversion produces row_count().
    pub fn row_count_raw(&self) -> Value {
        self.row_count_raw
    }
    pub fn build(self) -> KernelDefinition {
        self.builder.build()
    }
}

/// A typed view of an opaque device allocation. The raw import is unsafe because
/// handles do not encode the actual contents' element type or initialization.
#[derive(Clone, Debug)]
pub struct DeviceBuffer {
    handle: Handle,
    element_type: PhysicalType,
    logical_len: u64,
}

impl DeviceBuffer {
    /// Import a runtime-owned allocation/view. Range/alignment/device checks are
    /// performed by bind_device, not assumed from these raw parts.
    ///
    /// # Safety
    /// The handle must represent real live runtime memory with an appropriately
    /// aligned allocation base. Its readable elements must be initialized and
    /// represented by element_type. For a write-only output, initialization is
    /// not required. Metadata alone cannot prove these conditions.
    pub unsafe fn from_raw_parts(
        handle: Handle,
        element_type: PhysicalType,
        logical_len: u64,
    ) -> Self {
        Self {
            handle,
            element_type,
            logical_len,
        }
    }
    pub fn descriptor(&self) -> BufferDescriptor {
        BufferDescriptor {
            element_type: self.element_type,
            logical_len: self.logical_len,
            allocation_bytes: self.handle.size(),
            byte_offset: self.handle.offset_start.unwrap_or(0),
            end_padding_bytes: self.handle.offset_end.unwrap_or(0),
        }
    }
}

impl BindingLayout {
    /// Build in a fresh context; callers cannot accidentally reuse scalar counters
    /// from a builder containing a different ABI. Values remain context-local.
    pub fn declare(
        &self,
        client: &Client,
        settings: KernelSettings,
    ) -> Result<KernelDeclaration, BindingError> {
        validate_settings(self, &settings)?;
        let mut builder = KernelBuilder::new(settings);
        builder.runtime_properties(client.target_properties().clone());
        builder.device_properties(client.properties());
        let mut buffers = Vec::new();
        for slot in self.buffers() {
            let ty = slot.element_type().elem_type().to_type(builder.scope.ctx());
            buffers.push(builder.buffer(ty));
        }
        let mut parameters = Vec::new();
        let mut raw_count = None;
        for slot in self.scalars() {
            let elem = slot.element_type().elem_type();
            let id = builder.scalar(elem);
            assert_eq!(
                id,
                slot.ordinal_within_type(),
                "fresh builder scalar ordinal"
            );
            let scope = &builder.scope;
            let ty = elem.to_type(scope.ctx());
            let value = scope.register_with_result(&ReadScalarOp::new(scope.ctx_mut(), ty, id));
            match slot.source() {
                ScalarSource::RowCount => raw_count = Some(value),
                ScalarSource::Parameter(id) => parameters.push((id, value)),
            }
        }
        let raw_count = raw_count.expect("layout always declares row_count");
        let scope = &builder.scope;
        let index_ty = ElemType::Index.to_type(scope.ctx());
        let count = scope.register_with_result(&CastOp::new(scope.ctx_mut(), index_ty, raw_count));
        Ok(KernelDeclaration {
            layout: self.clone(),
            builder,
            buffers,
            parameters,
            row_count_raw: raw_count,
            row_count: count,
        })
    }

    /// Fully validate before any thread-local launcher state is touched. Buffers
    /// follow layout order, including the final write-only output.
    pub fn bind_device(
        &self,
        client: &Client,
        row_count: u64,
        buffers: &[DeviceBuffer],
        parameters: &Parameters,
    ) -> Result<ValidatedBindings, BindingError> {
        let descriptors: Vec<_> = buffers.iter().map(DeviceBuffer::descriptor).collect();
        let extent = self.validate_buffers(row_count, &descriptors)?;
        let scalars = self.bind_scalars(extent, parameters)?;
        for (index, buffer) in buffers.iter().enumerate() {
            if buffer.handle.service != client.service_id() {
                return Err(BindingError::ForeignDeviceBuffer { index });
            }
        }
        let output_id = buffers
            .last()
            .expect("validated buffer count")
            .handle
            .memory
            .clone()
            .binding()
            .id();
        for (index, buffer) in buffers[..buffers.len() - 1].iter().enumerate() {
            if buffer.handle.memory.clone().binding().id() == output_id {
                return Err(BindingError::AliasedOutput { input_index: index });
            }
        }
        let handles = buffers
            .iter()
            .map(|buffer| {
                let mut handle = buffer.handle.clone();
                // Validation proved this subtraction is in range. CubeCL derives
                // metadata length from this view's bytes, not BufferArg's len field.
                handle.offset_end = Some(
                    handle.size() - handle.offset_start.unwrap_or(0) - extent.byte_len() as u64,
                );
                handle
            })
            .collect();
        Ok(ValidatedBindings {
            layout: self.clone(),
            extent,
            handles,
            scalars,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Submission {
    Empty,
    Enqueued,
}

/// Owned, normalized buffers and scalar snapshots. Registration is only exposed
/// through a consuming launch, never a partially populated KernelLauncher.
#[derive(Debug)]
pub struct ValidatedBindings {
    layout: BindingLayout,
    extent: BufferExtent,
    handles: Vec<Handle>,
    scalars: Vec<ScalarArgument>,
}

impl ValidatedBindings {
    pub fn layout(&self) -> &BindingLayout {
        &self.layout
    }
    pub fn extent(&self) -> BufferExtent {
        self.extent
    }
    pub fn scalars(&self) -> &[ScalarArgument] {
        &self.scalars
    }

    /// Register all arguments contiguously and submit. Enqueued is not successful
    /// execution: callers must check read_one/sync_buffers at completion.
    ///
    /// # Safety
    /// The kernel must implement exactly this layout, use the same settings and
    /// honor buffer access, logical lengths and synchronization rules. Metadata
    /// checks cannot prove that an arbitrary CubeKernel body obeys its ABI.
    /// Do not interleave another CubeCL launcher's registration on this thread.
    pub unsafe fn launch<K: CubeKernel>(
        self,
        client: &Client,
        settings: KernelSettings,
        kernel: K,
        count: CubeCount,
    ) -> Result<Submission, BindingError> {
        validate_settings(&self.layout, &settings)?;
        let id = kernel.id();
        if id.address_type != settings.address_type
            || kernel.address_type() != settings.address_type.unsigned_type()
        {
            return Err(BindingError::SettingsMismatch {
                field: "kernel address_type",
            });
        }
        if id.cube_dim != settings.cube_dim || id.mode != settings.execution_mode {
            return Err(BindingError::SettingsMismatch {
                field: "kernel dimensions/execution_mode",
            });
        }
        for (index, handle) in self.handles.iter().enumerate() {
            if handle.service != client.service_id() {
                return Err(BindingError::ForeignDeviceBuffer { index });
            }
        }
        if self.extent.row_count() == 0 {
            return Ok(Submission::Empty);
        }
        if !matches!(&count, CubeCount::Static(x, y, z) if *x > 0 && *y > 0 && *z > 0) {
            return Err(BindingError::SettingsMismatch {
                field: "nonempty static cube count",
            });
        }
        let mut guard = LauncherGuard(Some(KernelLauncher::new(settings)));
        let launcher = guard.0.as_mut().expect("fresh launcher");
        for (slot, handle) in self.layout.buffers().iter().zip(self.handles) {
            launcher.declare_io(match slot.access() {
                BufferAccess::ReadOnly => BufferIOAttr::ReadOnly,
                BufferAccess::WriteOnly => BufferIOAttr::WriteOnly,
            });
            // SAFETY: bind_device checked and narrowed the handle to exactly the
            // logical extent. ABI v1 has scalar F32 buffers, not vector lanes.
            let buffer =
                unsafe { BufferArg::from_raw_parts(handle, self.extent.row_count() as usize) };
            launcher.register_buffer(buffer, slot.element_type().size_bytes() as usize);
        }
        for scalar in self.scalars {
            match scalar {
                ScalarArgument::U32(value) => launcher.register_scalar(value),
                ScalarArgument::F32(value) => launcher.register_scalar(value),
            }
        }
        guard
            .0
            .take()
            .expect("registered launcher")
            .launch(count, kernel, client);
        Ok(Submission::Enqueued)
    }
}

struct LauncherGuard(Option<KernelLauncher>);
impl Drop for LauncherGuard {
    fn drop(&mut self) {
        if let Some(launcher) = self.0.take() {
            launcher.discard();
        }
    }
}

fn validate_settings(
    layout: &BindingLayout,
    settings: &KernelSettings,
) -> Result<(), BindingError> {
    if settings.address_type != layout.address_type() {
        return Err(BindingError::SettingsMismatch {
            field: "address_type",
        });
    }
    let dim = settings.cube_dim;
    if dim.x == 0
        || dim.y == 0
        || dim.z == 0
        || dim
            .x
            .checked_mul(dim.y)
            .and_then(|xy| xy.checked_mul(dim.z))
            .is_none()
    {
        return Err(BindingError::SettingsMismatch { field: "cube_dim" });
    }
    Ok(())
}
