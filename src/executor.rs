//! Synchronous execution of prepared arithmetic and predicate programs on a selected CubeCL client.

use std::{
    error::Error as StdError,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cubecl::{
    bytes::Bytes,
    prelude::{
        AddressType, Client, CubeCount, CubeDim, ExecutionMode, KernelId, KernelMetadata,
        KernelSettings,
    },
    server::ServerError,
};

use crate::{
    Batch, DataType, KernelProgram, Parameters,
    abi::{BindingError, BindingLayout, BufferSource},
    cubecl_abi::DeviceBuffer,
    emitter::{ExpressionIdentity, ExpressionKernel},
};

/// A selected device and a fixed one-dimensional workgroup size. No CPU fallback.
/// Programs remain independent of the executor and can be reused with new inputs.
#[derive(Clone)]
pub struct Executor {
    client: Client,
    cube_size: u32,
    define_count: Arc<AtomicUsize>,
}

impl Executor {
    pub fn new(client: Client) -> Result<Self, ExecutionError> {
        let hw = &client.properties().hardware;
        let size = 64.min(hw.max_units_per_cube).min(hw.max_cube_dim.0);
        Self::with_cube_size(client, size)
    }

    pub fn with_cube_size(client: Client, cube_size: u32) -> Result<Self, ExecutionError> {
        let hw = &client.properties().hardware;
        if cube_size == 0
            || cube_size > hw.max_units_per_cube
            || cube_size > hw.max_cube_dim.0
            || hw.max_cube_dim.1 == 0
            || hw.max_cube_dim.2 == 0
        {
            return Err(ExecutionError::UnsupportedDevice {
                reason: "cube size must be positive and fit the device limits",
            });
        }
        Ok(Self {
            client,
            cube_size,
            define_count: Arc::new(AtomicUsize::new(0)),
        })
    }

    pub fn cube_size(&self) -> u32 {
        self.cube_size
    }

    /// IR definition attempts for kernels submitted by this executor and its
    /// clones. Counts failed definitions too; not driver compilations or disk hits.
    /// Observe after checked readback. Other executors have independent counters,
    /// but can share CubeCL cache entries and therefore define nothing themselves.
    pub fn define_count(&self) -> usize {
        self.define_count.load(Ordering::Relaxed)
    }

    /// Return the `KernelId` that will be used for compilation and caching
    /// of this program on this executor's device and settings.
    pub fn kernel_id(&self, program: &KernelProgram) -> KernelId {
        let kernel = ExpressionKernel::new(
            program.clone(),
            self.client.clone(),
            self.settings(),
            self.define_count.clone(),
        );
        kernel.id()
    }

    /// Return the structural compilation identity for this program.
    pub fn kernel_identity(&self, program: &KernelProgram) -> ExpressionIdentity {
        let kernel = ExpressionKernel::new(
            program.clone(),
            self.client.clone(),
            self.settings(),
            self.define_count.clone(),
        );
        kernel.identity()
    }

    fn settings(&self) -> KernelSettings {
        KernelSettings::new(
            CubeDim::new_1d(self.cube_size).into(),
            ExecutionMode::Checked,
            AddressType::U32,
        )
    }

    /// Validate before uploading; return output only after a checked device read.
    /// Zero rows validate bindings and return without allocation, definition or launch.
    /// CubeCL allocation and transport APIs may panic; backend read/compile failures
    /// reported by CubeCL are preserved in ExecutionError::Backend.
    pub fn execute(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<Vec<f32>, ExecutionError> {
        self.require_output_type(program, DataType::F32)?;
        self.require_non_nullable(program)?;
        let bytes = self.execute_bytes(program, batch, parameters)?.values;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect())
    }

    /// Execute a Bool predicate with WHERE semantics as a dense row-aligned mask.
    /// Only valid true keeps a row (1); false and NULL produce 0. No compaction.
    pub fn execute_mask(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<Vec<u32>, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        let output = self.execute_bytes(program, batch, parameters)?;
        // Unknown Bool payloads are already false on device. Check both output
        // encodings without a host truth-table evaluation or another kernel.
        if let Some(validity) = output.validity {
            decode_validity(&validity)?;
        }
        decode_validity(&output.values)
    }

    /// Execute an F32 projection with dense 0/1 validity. Non-nullable
    /// programs are accepted and return all-valid flags.
    pub fn execute_nullable(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<NullableF32, ExecutionError> {
        self.require_output_type(program, DataType::F32)?;
        let output = self.execute_bytes(program, batch, parameters)?;
        let values: Vec<_> = output
            .values
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        let validity = match output.validity {
            Some(bytes) => decode_validity(&bytes)?,
            None => vec![1; values.len()],
        };
        Ok(NullableF32 { values, validity })
    }

    /// Execute a Bool projection, preserving true/false/NULL as value/validity.
    /// Non-nullable Bool programs are accepted and return all-valid flags.
    pub fn execute_predicate(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<NullableBool, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        let output = self.execute_bytes(program, batch, parameters)?;
        let values = decode_validity(&output.values)?;
        let validity = match output.validity {
            Some(bytes) => decode_validity(&bytes)?,
            None => vec![1; values.len()],
        };
        Ok(NullableBool { values, validity })
    }

    fn require_non_nullable(&self, program: &KernelProgram) -> Result<(), ExecutionError> {
        if program.output_nullable() {
            return Err(ExecutionError::NullableOutput);
        }
        Ok(())
    }

    fn require_output_type(
        &self,
        program: &KernelProgram,
        expected: DataType,
    ) -> Result<(), ExecutionError> {
        let actual = program.output_type();
        if actual != expected {
            return Err(ExecutionError::OutputType { expected, actual });
        }
        Ok(())
    }

    fn execute_bytes(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<OutputBytes, ExecutionError> {
        let layout = BindingLayout::new(program);
        let host = layout.bind_host(batch, parameters)?;
        let rows = host.extent().row_count();
        if rows == 0 {
            return Ok(OutputBytes {
                values: Bytes::from_bytes_vec(Vec::new()),
                validity: program
                    .output_nullable()
                    .then(|| Bytes::from_bytes_vec(Vec::new())),
            });
        }
        let hw = &self.client.properties().hardware;
        // Each buffer plus the packed scalar/metadata info buffer needs a binding.
        if layout.buffers().len() as u64 + 1 > u64::from(hw.max_bindings) {
            return Err(ExecutionError::UnsupportedDevice {
                reason: "program exceeds the device buffer binding limit",
            });
        }
        let cubes = rows.div_ceil(self.cube_size);
        if cubes > hw.max_cube_count.0 || hw.max_cube_count.1 == 0 || hw.max_cube_count.2 == 0 {
            return Err(ExecutionError::GridTooLarge {
                required: cubes,
                max: hw.max_cube_count.0,
            });
        }
        let mut buffers = Vec::with_capacity(layout.buffers().len());
        let mut output = None;
        let mut output_validity = None;
        for slot in layout.buffers() {
            let handle = match slot.source() {
                BufferSource::Column(id) => {
                    let (column, _) = host.column_data(id);
                    let bytes: Vec<u8> = column
                        .iter()
                        .flat_map(|value| value.to_ne_bytes())
                        .collect();
                    self.client.create_from_slice(&bytes)
                }
                BufferSource::ColumnValidity(id) => {
                    let (_, validity) = host.column_data(id);
                    let bytes: Vec<u8> = (0..rows as usize)
                        .flat_map(|i| validity.map_or(1, |v| v[i]).to_ne_bytes())
                        .collect();
                    self.client.create_from_slice(&bytes)
                }
                BufferSource::Output => {
                    let handle = self.client.empty(host.extent().byte_len());
                    output = Some(handle.clone());
                    handle
                }
                BufferSource::OutputValidity => {
                    let handle = self.client.empty(host.extent().byte_len());
                    output_validity = Some(handle.clone());
                    handle
                }
            };
            // SAFETY: input bytes come from initialized typed slices and checked
            // validity. Outputs are distinct fresh write-only allocations. The
            // emitter initializes both output planes for every in-range row.
            buffers.push(unsafe {
                DeviceBuffer::from_raw_parts(handle, slot.element_type(), u64::from(rows))
            });
        }
        let bound = layout.bind_device(&self.client, u64::from(rows), &buffers, parameters)?;
        let settings = self.settings();
        let kernel = ExpressionKernel::new(
            program.clone(),
            self.client.clone(),
            settings.clone(),
            self.define_count.clone(),
        );
        // SAFETY: this private emitter uses the same immutable layout/settings,
        // guards every load/store by row_count and has no barriers or aliasing.
        unsafe {
            bound.launch(
                &self.client,
                settings,
                kernel,
                CubeCount::Static(cubes, 1, 1),
            )?;
        }
        let read = |handle| {
            let bytes = self
                .client
                .read_one(handle)
                .map_err(ExecutionError::Backend)?;
            if bytes.len() != host.extent().byte_len() {
                return Err(ExecutionError::OutputLength {
                    expected: host.extent().byte_len(),
                    actual: bytes.len(),
                });
            }
            Ok(bytes)
        };
        // No result is exposed unless every declared output has completed.
        let values = read(output.expect("layout output"))?;
        let validity = output_validity.map(read).transpose()?;
        Ok(OutputBytes { values, validity })
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Binding(BindingError),
    NullableOutput,
    OutputType {
        expected: DataType,
        actual: DataType,
    },
    InvalidMaskValue {
        index: usize,
        value: u32,
    },
    UnsupportedDevice {
        reason: &'static str,
    },
    GridTooLarge {
        required: u32,
        max: u32,
    },
    Backend(ServerError),
    OutputLength {
        expected: usize,
        actual: usize,
    },
}

impl From<BindingError> for ExecutionError {
    fn from(value: BindingError) -> Self {
        Self::Binding(value)
    }
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NullableOutput => f.write_str("nullable F32 output requires execute_nullable"),
            Self::OutputType { expected, actual } => {
                write!(f, "expected {expected} output, program produces {actual}")
            }
            Self::InvalidMaskValue { index, value } => {
                write!(f, "mask row {index}: expected 0 or 1, got {value}")
            }
            Self::Binding(error) => write!(f, "invalid query binding: {error}"),
            Self::UnsupportedDevice { reason } => {
                write!(f, "unsupported execution settings: {reason}")
            }
            Self::GridTooLarge { required, max } => write!(
                f,
                "query needs {required} cubes on x; device limit is {max}"
            ),
            Self::Backend(error) => write!(f, "device execution failed: {error}"),
            Self::OutputLength { expected, actual } => {
                write!(f, "device returned {actual} bytes; expected {expected}")
            }
        }
    }
}

struct OutputBytes {
    values: Bytes,
    validity: Option<Bytes>,
}

/// Owned F32 projection with one 0/1 validity flag per row.
/// Invalid payloads are canonical +0; use iter() to observe SQL NULLs.
#[derive(Clone, Debug, PartialEq)]
pub struct NullableF32 {
    values: Vec<f32>,
    validity: Vec<u32>,
}
impl NullableF32 {
    #[cfg(feature = "arrow")]
    pub(crate) fn into_parts(self) -> (Vec<f32>, Vec<u32>) {
        (self.values, self.validity)
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }
    pub fn validity(&self) -> &[u32] {
        &self.validity
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<f32>> + '_ {
        self.values
            .iter()
            .zip(&self.validity)
            .map(|(&value, &valid)| (valid == 1).then_some(value))
    }
}

/// Owned Bool projection. Unknown payloads are canonical 0 (false).
/// iter() preserves SQL unknown as None; execute_mask() applies WHERE semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NullableBool {
    values: Vec<u32>,
    validity: Vec<u32>,
}
impl NullableBool {
    #[cfg(feature = "arrow")]
    pub(crate) fn into_parts(self) -> (Vec<u32>, Vec<u32>) {
        (self.values, self.validity)
    }

    pub fn values(&self) -> &[u32] {
        &self.values
    }
    pub fn validity(&self) -> &[u32] {
        &self.validity
    }
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<bool>> + '_ {
        self.values
            .iter()
            .zip(&self.validity)
            .map(|(&value, &valid)| (valid == 1).then_some(value == 1))
    }
}

fn decode_validity(bytes: &[u8]) -> Result<Vec<u32>, ExecutionError> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
        .map(|(index, b)| {
            let value = u32::from_ne_bytes(*b);
            if value > 1 {
                Err(ExecutionError::InvalidMaskValue { index, value })
            } else {
                Ok(value)
            }
        })
        .collect()
}

impl StdError for ExecutionError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Binding(error) => Some(error),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}
