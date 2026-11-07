//! Versioned, value-independent kernel argument layout and host-side preflight.

use std::{collections::BTreeSet, error::Error as StdError, fmt};

use cubecl::prelude::AddressType;

use crate::{
    Batch, ColumnData, ColumnId, DataType, Error, InputKind, KernelProgram, ParameterId,
    Parameters, ScalarOp, ScalarValue,
};

pub const BINDING_ABI_VERSION: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PhysicalType {
    F32,
    U32,
    I32,
}

impl PhysicalType {
    fn for_numeric(ty: DataType) -> Self {
        match ty {
            DataType::F32 => Self::F32,
            DataType::I32 => Self::I32,
            _ => unreachable!("prepared numeric input"),
        }
    }

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
    ColumnValidity(ColumnId),
    Output,
    OutputValidity,
    OutputError,
}

/// A scalar F32/I32 payload or U32 validity/mask/error buffer.
/// Logical length equals the invocation row count.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferSlot {
    index: usize,
    source: BufferSource,
    name: String,
    access: BufferAccess,
    element_type: PhysicalType,
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
        self.element_type
    }
    pub fn alignment(&self) -> u64 {
        self.element_type().alignment()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarSource {
    RowCount,
    Parameter(ParameterId),
    ParameterValidity(ParameterId),
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
                element_type: PhysicalType::for_numeric(
                    program.schema().columns()[index].data_type(),
                ),
            });
            if program.schema().columns()[index].is_nullable() {
                buffers.push(BufferSlot {
                    index: buffers.len(),
                    source: BufferSource::ColumnValidity(ColumnId(index)),
                    name: program.schema().columns()[index].name().into(),
                    access: BufferAccess::ReadOnly,
                    element_type: PhysicalType::U32,
                });
            }
        }
        buffers.push(BufferSlot {
            index: buffers.len(),
            source: BufferSource::Output,
            name: "output".into(),
            access: BufferAccess::WriteOnly,
            element_type: match program.output_type() {
                DataType::F32 => PhysicalType::F32,
                DataType::Bool => PhysicalType::U32,
                DataType::I32 => PhysicalType::I32,
            },
        });
        if program.output_nullable() {
            buffers.push(BufferSlot {
                index: buffers.len(),
                source: BufferSource::OutputValidity,
                name: "output_validity".into(),
                access: BufferAccess::WriteOnly,
                element_type: PhysicalType::U32,
            });
        }
        if program.has_integer_arithmetic() {
            buffers.push(BufferSlot {
                index: buffers.len(),
                source: BufferSource::OutputError,
                name: "output_error".into(),
                access: BufferAccess::WriteOnly,
                element_type: PhysicalType::U32,
            });
        }
        let mut scalars = vec![ScalarSlot {
            source: ScalarSource::RowCount,
            name: "row_count".into(),
            element_type: PhysicalType::U32,
            ordinal_within_type: 0,
        }];
        let mut validity_ordinal = 1;
        let mut float_ordinal = 0;
        let mut int_ordinal = 0;
        for index in parameters {
            let ty = PhysicalType::for_numeric(program.schema().parameters()[index].data_type());
            let ordinal = match ty {
                PhysicalType::F32 => {
                    let n = float_ordinal;
                    float_ordinal += 1;
                    n
                }
                PhysicalType::I32 => {
                    let n = int_ordinal;
                    int_ordinal += 1;
                    n
                }
                _ => unreachable!(),
            };
            scalars.push(ScalarSlot {
                source: ScalarSource::Parameter(ParameterId(index)),
                name: program.schema().parameters()[index].name().into(),
                element_type: ty,
                ordinal_within_type: ordinal,
            });
            if program.schema().parameters()[index].is_nullable() {
                scalars.push(ScalarSlot {
                    source: ScalarSource::ParameterValidity(ParameterId(index)),
                    name: program.schema().parameters()[index].name().into(),
                    element_type: PhysicalType::U32,
                    ordinal_within_type: validity_ordinal,
                });
                validity_ordinal += 1;
            }
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
    /// ABI v4 conservatively limits both row count and active byte ends to U32.
    pub fn extent(&self, row_count: u64) -> Result<BufferExtent, BindingError> {
        let rows = u32::try_from(row_count).map_err(|_| BindingError::AddressOverflow {
            field: "row_count",
            value: row_count,
        })?;
        // Every supported buffer slot is four bytes per row in ABI v4.
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
        let mut column_validities = Vec::new();
        for slot in &self.buffers {
            if let BufferSource::Column(id) = slot.source {
                let kind = InputKind::Column;
                let data = batch
                    .column(&slot.name)
                    .ok_or_else(|| Error::MissingInput {
                        kind,
                        name: slot.name.clone(),
                    })?;
                let expected = match slot.element_type {
                    PhysicalType::F32 => DataType::F32,
                    PhysicalType::I32 => DataType::I32,
                    _ => unreachable!(),
                };
                if data.data_type() != expected {
                    return Err(Error::TypeMismatch {
                        kind,
                        name: slot.name.clone(),
                        expected,
                        actual: data.data_type(),
                    }
                    .into());
                }
                let validity = data.validity();
                let values = match data {
                    ColumnData::F32(v) | ColumnData::NullableF32 { values: v, .. } => {
                        ColumnData::F32(v)
                    }
                    ColumnData::I32(v) | ColumnData::NullableI32 { values: v, .. } => {
                        ColumnData::I32(v)
                    }
                };
                if !self
                    .buffers
                    .iter()
                    .any(|s| s.source == BufferSource::ColumnValidity(id))
                    && validity.is_some_and(|v| v.contains(&0))
                {
                    return Err(Error::UnexpectedNull {
                        kind,
                        name: slot.name.clone(),
                    }
                    .into());
                }
                column_validities.push(validity);
                columns.push(values);
            }
        }
        let scalars = self.bind_scalars(extent, parameters)?;
        Ok(HostBindings {
            layout: self.clone(),
            extent,
            columns,
            column_validities,
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
                ScalarSource::Parameter(id) | ScalarSource::ParameterValidity(id) => {
                    let kind = InputKind::Parameter;
                    let value =
                        parameters
                            .value(&slot.name)
                            .ok_or_else(|| Error::MissingInput {
                                kind,
                                name: slot.name.clone(),
                            })?;
                    let payload_type = self
                        .scalars
                        .iter()
                        .find(|s| s.source == ScalarSource::Parameter(id))
                        .expect("parameter payload")
                        .element_type;
                    let expected = match payload_type {
                        PhysicalType::F32 => DataType::F32,
                        PhysicalType::I32 => DataType::I32,
                        _ => unreachable!(),
                    };
                    if value.data_type() != expected {
                        return Err(Error::TypeMismatch {
                            kind,
                            name: slot.name.clone(),
                            expected,
                            actual: value.data_type(),
                        }
                        .into());
                    }
                    let (payload, valid) = match value {
                        ScalarValue::F32(v) => (ScalarArgument::F32(v), true),
                        ScalarValue::NullableF32(v) => {
                            (ScalarArgument::F32(v.unwrap_or(0.0)), v.is_some())
                        }
                        ScalarValue::I32(v) => (ScalarArgument::I32(v), true),
                        ScalarValue::NullableI32(v) => {
                            (ScalarArgument::I32(v.unwrap_or(0)), v.is_some())
                        }
                    };
                    if !valid
                        && !self
                            .scalars
                            .iter()
                            .any(|s| s.source == ScalarSource::ParameterValidity(id))
                    {
                        return Err(Error::UnexpectedNull {
                            kind,
                            name: slot.name.clone(),
                        }
                        .into());
                    }
                    Ok(match slot.source {
                        ScalarSource::Parameter(_) => payload,
                        ScalarSource::ParameterValidity(_) => ScalarArgument::U32(u32::from(valid)),
                        _ => unreachable!(),
                    })
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
    I32(i32),
}

/// Column value slices are in schema-ID order, excluding validity/output slots.
/// column_validities() supplies a matching optional slice for each value slice.
#[derive(Clone, Debug)]
pub struct HostBindings<'a> {
    layout: BindingLayout,
    extent: BufferExtent,
    columns: Vec<ColumnData<'a>>,
    column_validities: Vec<Option<&'a [u32]>>,
    scalars: Vec<ScalarArgument>,
}
impl<'a> HostBindings<'a> {
    pub fn layout(&self) -> &BindingLayout {
        &self.layout
    }
    pub fn extent(&self) -> BufferExtent {
        self.extent
    }
    pub fn columns(&self) -> &[ColumnData<'a>] {
        &self.columns
    }
    /// Dense validity slices aligned with columns(); None means all valid.
    pub fn column_validities(&self) -> &[Option<&'a [u32]>] {
        &self.column_validities
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
    AliasedOutputs {
        first: usize,
        second: usize,
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
                write!(f, "ABI v4 does not support address type {ty}")
            }
            Self::AddressOverflow { field, value } => {
                write!(f, "{field}={value} exceeds the ABI v4 address range")
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
            Self::AliasedOutputs { first, second } => {
                write!(f, "outputs {first} and {second} share an allocation")
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
