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
    abi::{BindingError, BindingLayout, PhysicalType},
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
        let bytes = self.execute_bytes(program, batch, parameters)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect())
    }

    /// Execute a non-nullable predicate as a dense row-aligned mask.
    /// Every entry is exactly 0 (false) or 1 (true); rows are not compacted.
    pub fn execute_mask(
        &self,
        program: &KernelProgram,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<Vec<u32>, ExecutionError> {
        self.require_output_type(program, DataType::Bool)?;
        let bytes = self.execute_bytes(program, batch, parameters)?;
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
    ) -> Result<Bytes, ExecutionError> {
        let layout = BindingLayout::new(program);
        let host = layout.bind_host(batch, parameters)?;
        let rows = host.extent().row_count();
        if rows == 0 {
            return Ok(Bytes::from_bytes_vec(Vec::new()));
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
        for column in host.columns() {
            let bytes: Vec<u8> = column
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();
            let handle = self.client.create_from_slice(&bytes);
            // SAFETY: the runtime owns the initialized allocation uploaded above;
            // bytes represent scalar F32, aligned by the runtime allocator.
            buffers.push(unsafe {
                DeviceBuffer::from_raw_parts(handle, PhysicalType::F32, u64::from(rows))
            });
        }
        let output = self.client.empty(host.extent().byte_len());
        // SAFETY: a fresh, runtime-aligned allocation of the exact required size.
        // This layout's output is write-only; the emitter writes every logical row.
        buffers.push(unsafe {
            DeviceBuffer::from_raw_parts(
                output.clone(),
                layout.buffers().last().expect("output slot").element_type(),
                u64::from(rows),
            )
        });
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
        let bytes = self
            .client
            .read_one(output)
            .map_err(ExecutionError::Backend)?;
        if bytes.len() != host.extent().byte_len() {
            return Err(ExecutionError::OutputLength {
                expected: host.extent().byte_len(),
                actual: bytes.len(),
            });
        }
        Ok(bytes)
    }
}

#[derive(Debug)]
pub enum ExecutionError {
    Binding(BindingError),
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

impl StdError for ExecutionError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            Self::Binding(error) => Some(error),
            Self::Backend(error) => Some(error),
            _ => None,
        }
    }
}
