//! Versioned, value-independent kernel argument layout and host-side preflight.

use std::{collections::BTreeSet, error::Error as StdError, fmt};

use cubecl::prelude::AddressType;

use crate::{
    Batch, ColumnData, ColumnId, DataType, Error, InputKind, KernelProgram, ParameterId,
    Parameters, ScalarOp, ScalarValue,
};

pub const BINDING_ABI_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicalType {
    F32,
    U32,
    I32,
}

impl PhysicalType {
    pub const fn size_bytes(self) -> u64 {
        4
    }
    pub const fn alignment(self) -> u64 {
        4
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferSource {
    Column(ColumnId),
    Output,
}

/// A scalar F32 buffer, with a logical length equal to the invocation row count.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferSlot {
    index: usize,
    source: BufferSource,
    name: String,
    access: BufferAccess,
}

impl BufferSlot {
    pub fn index(&self) -> usize {
        self.index
    }
    pub fn source(&self) -> BufferSource {
        self.source
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn access(&self) -> BufferAccess {
        self.access
    }
    pub fn element_type(&self) -> PhysicalType {
        PhysicalType::F32
    }
    pub fn alignment(&self) -> u64 {
        self.element_type().alignment()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarSource {
    RowCount,
    Parameter(ParameterId),
}

/// The ordinal is within its physical type, not a position in a packed struct.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScalarSlot {
    source: ScalarSource,
    name: String,
    element_type: PhysicalType,
    ordinal_within_type: usize,
}

impl ScalarSlot {
    pub fn source(&self) -> ScalarSource {
        self.source
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn element_type(&self) -> PhysicalType {
        self.element_type
    }
    pub fn ordinal_within_type(&self) -> usize {
        self.ordinal_within_type
    }
}

/// Does not contain parameter values, row counts, device handles or expression ops.
/// Equality identifies an ABI, not the complete kernel compilation identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindingLayout {
    version: u32,
    address_type: AddressType,
    buffers: Vec<BufferSlot>,
    scalars: Vec<ScalarSlot>,
}

impl BindingLayout {
    pub fn new(program: &KernelProgram) -> Self {
        let mut columns = BTreeSet::new();
        let mut parameters = BTreeSet::new();
        for node in program.nodes() {
            match node.op() {
                ScalarOp::Column(id) => {
                    columns.insert(id.index());
                }
                ScalarOp::Parameter(id) => {
                    parameters.insert(id.index());
                }
                _ => {}
            }
        }
        let mut buffers = Vec::new();
        for index in columns {
            buffers.push(BufferSlot {
                index: buffers.len(),
                source: BufferSource::Column(ColumnId(index)),
                name: program.schema().columns()[index].name().into(),
                access: BufferAccess::ReadOnly,
            });
        }
        buffers.push(BufferSlot {
            index: buffers.len(),
            source: BufferSource::Output,
            name: "output".into(),
            access: BufferAccess::WriteOnly,
        });
        let mut scalars = vec![ScalarSlot {
            source: ScalarSource::RowCount,
            name: "row_count".into(),
            element_type: PhysicalType::U32,
            ordinal_within_type: 0,
        }];
        for (ordinal, index) in parameters.into_iter().enumerate() {
            scalars.push(ScalarSlot {
                source: ScalarSource::Parameter(ParameterId(index)),
                name: program.schema().parameters()[index].name().into(),
                element_type: PhysicalType::F32,
                ordinal_within_type: ordinal,
            });
        }
        Self {
            version: BINDING_ABI_VERSION,
            address_type: AddressType::U32,
            buffers,
            scalars,
        }
    }

    pub fn with_address_type(
        program: &KernelProgram,
        address_type: AddressType,
    ) -> Result<Self, BindingError> {
        if address_type != AddressType::U32 {
            return Err(BindingError::UnsupportedAddressType(address_type));
        }
        Ok(Self::new(program))
    }
    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn address_type(&self) -> AddressType {
        self.address_type
    }
    pub fn buffers(&self) -> &[BufferSlot] {
        &self.buffers
    }
    pub fn scalars(&self) -> &[ScalarSlot] {
        &self.scalars
    }

    /// Check active lengths and byte extents without allocating a device buffer.
    /// ABI v1 conservatively limits both row count and active byte ends to U32.
    pub fn extent(&self, row_count: u64) -> Result<BufferExtent, BindingError> {
        let rows = u32::try_from(row_count).map_err(|_| BindingError::AddressOverflow {
            field: "row_count",
            value: row_count,
        })?;
        let bytes = row_count * PhysicalType::F32.size_bytes();
        if bytes > u64::from(u32::MAX) {
            return Err(BindingError::AddressOverflow {
                field: "byte_length",
                value: bytes,
            });
        }
        let byte_len = usize::try_from(bytes).map_err(|_| BindingError::AddressOverflow {
            field: "host_byte_length",
            value: bytes,
        })?;
        Ok(BufferExtent {
            row_count: rows,
            byte_len,
        })
    }

    /// Descriptors describe full allocations plus a start offset and end padding.
    /// Extra visible capacity is allowed; the device adapter narrows it at bind time.
    pub fn validate_buffers(
        &self,
        row_count: u64,
        descriptors: &[BufferDescriptor],
    ) -> Result<BufferExtent, BindingError> {
        let extent = self.extent(row_count)?;
        if descriptors.len() != self.buffers.len() {
            return Err(BindingError::BufferCount {
                expected: self.buffers.len(),
                actual: descriptors.len(),
            });
        }
        for (slot, descriptor) in self.buffers.iter().zip(descriptors) {
            let index = slot.index;
            if descriptor.element_type != slot.element_type() {
                return Err(BindingError::BufferType {
                    index,
                    expected: slot.element_type(),
                    actual: descriptor.element_type,
                });
            }
            if descriptor.logical_len != row_count {
                return Err(BindingError::BufferLength {
                    index,
                    expected: row_count,
                    actual: descriptor.logical_len,
                });
            }
            let used_offsets = descriptor
                .byte_offset
                .checked_add(descriptor.end_padding_bytes)
                .filter(|sum| *sum <= descriptor.allocation_bytes)
                .ok_or(BindingError::InvalidBufferRange { index })?;
            if descriptor.byte_offset % slot.alignment() != 0 {
                return Err(BindingError::BufferAlignment {
                    index,
                    offset: descriptor.byte_offset,
                    required: slot.alignment(),
                });
            }
            let available = descriptor.allocation_bytes - used_offsets;
            let required = extent.byte_len as u64;
            if available < required {
                return Err(BindingError::BufferTooSmall {
                    index,
                    required,
                    available,
                });
            }
            let end = descriptor
                .byte_offset
                .checked_add(required)
                .ok_or(BindingError::InvalidBufferRange { index })?;
            if end > u64::from(u32::MAX) {
                return Err(BindingError::AddressOverflow {
                    field: "buffer_byte_end",
                    value: end,
                });
            }
        }
        Ok(extent)
    }

    /// Validate named host input before any upload or launcher registration.
    pub fn bind_host<'a>(
        &self,
        batch: &Batch<'a>,
        parameters: &Parameters,
    ) -> Result<HostBindings<'a>, BindingError> {
        let extent = self.extent(batch.row_count() as u64)?;
        let mut columns = Vec::new();
        for slot in &self.buffers {
            if let BufferSource::Column(_) = slot.source {
                let kind = InputKind::Column;
                let data = batch
                    .column(&slot.name)
                    .ok_or_else(|| Error::MissingInput {
                        kind,
                        name: slot.name.clone(),
                    })?;
                let ColumnData::F32(values) = data else {
                    return Err(Error::TypeMismatch {
                        kind,
                        name: slot.name.clone(),
                        expected: DataType::F32,
                        actual: data.data_type(),
                    }
                    .into());
                };
                columns.push(values);
            }
        }
        let scalars = self.bind_scalars(extent, parameters)?;
        Ok(HostBindings {
            layout: self.clone(),
            extent,
            columns,
            scalars,
        })
    }

    pub(crate) fn bind_scalars(
        &self,
        extent: BufferExtent,
        parameters: &Parameters,
    ) -> Result<Vec<ScalarArgument>, BindingError> {
        self.scalars
            .iter()
            .map(|slot| match slot.source {
                ScalarSource::RowCount => Ok(ScalarArgument::U32(extent.row_count)),
                ScalarSource::Parameter(_) => {
                    let kind = InputKind::Parameter;
                    let value =
                        parameters
                            .value(&slot.name)
                            .ok_or_else(|| Error::MissingInput {
                                kind,
                                name: slot.name.clone(),
                            })?;
                    let ScalarValue::F32(value) = value else {
                        return Err(Error::TypeMismatch {
                            kind,
                            name: slot.name.clone(),
                            expected: DataType::F32,
                            actual: value.data_type(),
                        }
                        .into());
                    };
                    Ok(ScalarArgument::F32(value))
                }
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferExtent {
    row_count: u32,
    byte_len: usize,
}
impl BufferExtent {
    pub fn row_count(self) -> u32 {
        self.row_count
    }
    pub fn byte_len(self) -> usize {
        self.byte_len
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub element_type: PhysicalType,
    pub logical_len: u64,
    pub allocation_bytes: u64,
    pub byte_offset: u64,
    pub end_padding_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ScalarArgument {
    U32(u32),
    F32(f32),
}

/// Values are in layout order. Column slices exclude the write-only output slot.
#[derive(Clone, Debug)]
pub struct HostBindings<'a> {
    layout: BindingLayout,
    extent: BufferExtent,
    columns: Vec<&'a [f32]>,
    scalars: Vec<ScalarArgument>,
}
impl<'a> HostBindings<'a> {
    pub fn layout(&self) -> &BindingLayout {
        &self.layout
    }
    pub fn extent(&self) -> BufferExtent {
        self.extent
    }
    pub fn columns(&self) -> &[&'a [f32]] {
        &self.columns
    }
    pub fn scalars(&self) -> &[ScalarArgument] {
        &self.scalars
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingError {
    Input(Error),
    UnsupportedAddressType(AddressType),
    AddressOverflow {
        field: &'static str,
        value: u64,
    },
    BufferCount {
        expected: usize,
        actual: usize,
    },
    BufferType {
        index: usize,
        expected: PhysicalType,
        actual: PhysicalType,
    },
    BufferLength {
        index: usize,
        expected: u64,
        actual: u64,
    },
    InvalidBufferRange {
        index: usize,
    },
    BufferAlignment {
        index: usize,
        offset: u64,
        required: u64,
    },
    BufferTooSmall {
        index: usize,
        required: u64,
        available: u64,
    },
    ForeignDeviceBuffer {
        index: usize,
    },
    AliasedOutput {
        input_index: usize,
    },
    SettingsMismatch {
        field: &'static str,
    },
}

impl From<Error> for BindingError {
    fn from(error: Error) -> Self {
        Self::Input(error)
    }
}
impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => error.fmt(f),
            Self::UnsupportedAddressType(ty) => {
                write!(f, "ABI v1 does not support address type {ty}")
            }
            Self::AddressOverflow { field, value } => {
                write!(f, "{field}={value} exceeds the ABI v1 address range")
            }
            Self::BufferCount { expected, actual } => {
                write!(f, "expected {expected} buffers, got {actual}")
            }
            Self::BufferType {
                index,
                expected,
                actual,
            } => write!(f, "buffer {index}: expected {expected:?}, got {actual:?}"),
            Self::BufferLength {
                index,
                expected,
                actual,
            } => write!(f, "buffer {index}: expected {expected} rows, got {actual}"),
            Self::InvalidBufferRange { index } => {
                write!(f, "buffer {index}: offsets are outside the allocation")
            }
            Self::BufferAlignment {
                index,
                offset,
                required,
            } => write!(
                f,
                "buffer {index}: offset {offset} is not aligned to {required} bytes"
            ),
            Self::BufferTooSmall {
                index,
                required,
                available,
            } => write!(
                f,
                "buffer {index}: requires {required} bytes, has {available}"
            ),
            Self::ForeignDeviceBuffer { index } => {
                write!(f, "buffer {index} belongs to another device service")
            }
            Self::AliasedOutput { input_index } => write!(
                f,
                "output shares an allocation with input buffer {input_index}"
            ),
            Self::SettingsMismatch { field } => write!(f, "kernel settings mismatch: {field}"),
        }
    }
}
impl StdError for BindingError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            _ => None,
        }
    }
}
