//! Immutable device snapshots and explicit blocking completion/materialization.

use crate::{
    Batch, DataType, Error, ExecutionError, Executor, Field, InputKind, KernelProgram,
    NullableBool, NullableF32, NullableI32, Parameters,
    abi::{BindingError, BindingLayout, BufferSource, PhysicalType},
    executor::{SubmittedExpression, decode_validity, read_plane},
};
use cubecl::{bytes::Bytes, future::block_on, prelude::Client, server::Handle};
use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Cumulative explicit Cuberel transfers and submissions, shared by executor
/// clones. Excludes runtime metadata, implicit backend copies and allocator caches.
/// Calls count attempts; bytes count successful readbacks or uploaded payloads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutionStats {
    pub upload_calls: u64,
    pub upload_bytes: u64,
    pub readback_calls: u64,
    pub readback_bytes: u64,
    pub kernel_launches: u64,
    pub count_readbacks: u64,
}

#[derive(Default)]
pub(crate) struct ExecutionCounters {
    uploads: AtomicU64,
    upload_bytes: AtomicU64,
    reads: AtomicU64,
    read_bytes: AtomicU64,
    pub(crate) launches: AtomicU64,
    pub(crate) count_reads: AtomicU64,
}
impl ExecutionCounters {
    pub(crate) fn read(
        &self,
        client: &Client,
        handle: Handle,
        expected: usize,
    ) -> Result<Bytes, ExecutionError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let bytes = read_plane(client, handle, expected)?;
        self.read_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        Ok(bytes)
    }
    fn snapshot(&self) -> ExecutionStats {
        ExecutionStats {
            upload_calls: self.uploads.load(Ordering::Relaxed),
            upload_bytes: self.upload_bytes.load(Ordering::Relaxed),
            readback_calls: self.reads.load(Ordering::Relaxed),
            readback_bytes: self.read_bytes.load(Ordering::Relaxed),
            kernel_launches: self.launches.load(Ordering::Relaxed),
            count_readbacks: self.count_reads.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
struct ResidentColumn {
    field: Field,
    has_null: bool,
    values: Option<Handle>,
    validity: Option<Handle>,
}

/// An owned, immutable upload of exactly the columns present in a host Batch.
/// No host borrows, raw imports, implicit address cache or mutable buffers.
/// Independently uploaded snapshots have distinct identities, even if equal.
#[derive(Clone)]
pub struct DeviceBatch {
    client: Client,
    rows: u32,
    pub(crate) source: Arc<()>,
    columns: BTreeMap<String, ResidentColumn>,
}
impl DeviceBatch {
    pub fn row_count(&self) -> u32 {
        self.rows
    }
    pub fn columns(&self) -> impl ExactSizeIterator<Item = &Field> {
        self.columns.values().map(|c| &c.field)
    }
    pub fn same_source(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.source, &other.source)
    }
    pub(crate) fn validate(
        &self,
        client: &Client,
        layout: &BindingLayout,
        parameters: &Parameters,
    ) -> Result<(), ExecutionError> {
        if self.client.service_id() != client.service_id() {
            return Err(ExecutionError::ForeignDeviceBatch);
        }
        let extent = layout.extent(u64::from(self.rows))?;
        layout.bind_scalars(extent, parameters)?;
        for slot in layout.buffers() {
            if let BufferSource::Column(id) = slot.source() {
                let column = self.columns.get(slot.name()).ok_or_else(|| {
                    BindingError::from(Error::MissingInput {
                        kind: InputKind::Column,
                        name: slot.name().into(),
                    })
                })?;
                let expected = match slot.element_type() {
                    PhysicalType::F32 => DataType::F32,
                    _ => DataType::I32,
                };
                if column.field.data_type() != expected {
                    return Err(BindingError::from(Error::TypeMismatch {
                        kind: InputKind::Column,
                        name: slot.name().into(),
                        expected,
                        actual: column.field.data_type(),
                    })
                    .into());
                }
                if column.has_null
                    && !layout
                        .buffers()
                        .iter()
                        .any(|s| s.source() == BufferSource::ColumnValidity(id))
                {
                    return Err(BindingError::from(Error::UnexpectedNull {
                        kind: InputKind::Column,
                        name: slot.name().into(),
                    })
                    .into());
                }
            }
        }
        Ok(())
    }
    pub(crate) fn value_handle(&self, name: &str) -> Handle {
        self.columns[name]
            .values
            .clone()
            .expect("nonempty validated batch")
    }
    pub(crate) fn validity_handle(&self, name: &str) -> Handle {
        self.columns[name]
            .validity
            .clone()
            .expect("nonempty validated batch")
    }
}

/// A completed column. Validity is always exposed, including all-valid programs.
#[derive(Clone, Debug, PartialEq)]
pub enum HostColumn {
    F32(NullableF32),
    I32(NullableI32),
    Bool(NullableBool),
}

/// One or more pending expression outputs, with a common completion boundary.
/// Clones share immutable outputs. Completion checks every retained resource and
/// every error plane before materialize returns any host columns.
#[derive(Clone)]
pub struct DeviceResult {
    client: Client,
    counters: Arc<ExecutionCounters>,
    outputs: Arc<Vec<(DataType, bool, SubmittedExpression)>>,
}
impl DeviceResult {
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }
    /// Explicit blocking completion. Checked I32 error planes require readback;
    /// value and validity planes remain on device. Repeated calls repeat checks.
    pub fn complete(&self) -> Result<(), ExecutionError> {
        let handles = self.outputs.iter().flat_map(|(_, _, o)| o.retained.iter());
        if self.outputs.iter().any(|(_, _, o)| o.rows != 0) {
            block_on(self.client.sync_buffers(handles)).map_err(ExecutionError::Backend)?;
        }
        for (_, _, output) in self.outputs.iter() {
            output.check_errors(&self.client, &self.counters)?;
        }
        Ok(())
    }
    /// Blocking readback after completion; no partial vector is returned on error.
    /// Output order matches the program slice passed to execute_many_resident.
    pub fn materialize(&self) -> Result<Vec<HostColumn>, ExecutionError> {
        self.complete()?;
        self.outputs
            .iter()
            .map(|(ty, _, output)| {
                let bytes = output.read_bytes(&self.client, &self.counters)?;
                let validity = match bytes.validity {
                    Some(v) => decode_validity(&v)?,
                    None => vec![1; output.rows as usize],
                };
                Ok(match ty {
                    DataType::F32 => HostColumn::F32(NullableF32::new(
                        bytes
                            .values
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| f32::from_ne_bytes(*v))
                            .collect(),
                        validity,
                    )),
                    DataType::I32 => HostColumn::I32(NullableI32::new(
                        bytes
                            .values
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .map(|v| i32::from_ne_bytes(*v))
                            .collect(),
                        validity,
                    )),
                    DataType::Bool => HostColumn::Bool(NullableBool::new(
                        decode_validity(&bytes.values)?,
                        validity,
                    )),
                })
            })
            .collect()
    }
    pub fn output_type(&self, index: usize) -> Option<DataType> {
        self.outputs.get(index).map(|o| o.0)
    }
    pub fn output_nullable(&self, index: usize) -> Option<bool> {
        self.outputs.get(index).map(|o| o.1)
    }
}

impl Executor {
    pub fn stats(&self) -> ExecutionStats {
        self.counters.snapshot()
    }
    pub(crate) fn upload_bytes(&self, bytes: &[u8]) -> Handle {
        let handle = self.client.create_from_slice(bytes);
        self.counters.uploads.fetch_add(1, Ordering::Relaxed);
        self.counters
            .upload_bytes
            .fetch_add(bytes.len() as u64, Ordering::Relaxed);
        handle
    }
    /// Upload exactly the supplied columns once. To upload a subset, construct a
    /// Batch containing that subset. Missing query inputs fail; no implicit upload.
    /// Each column has a dense validity plane (all ones when non-nullable), so
    /// later nullable programs need no new column uploads. Zero rows allocate none.
    /// CubeCL allocation/transport APIs can panic, as in synchronous execution.
    pub fn upload(&self, batch: &Batch<'_>) -> Result<DeviceBatch, ExecutionError> {
        // Independent of any program; use the common ABI's conservative limits.
        let rows = batch.row_count() as u64;
        if rows > u64::from(u32::MAX) / 4 {
            return Err(BindingError::AddressOverflow {
                field: "batch_byte_length",
                value: rows.saturating_mul(4),
            }
            .into());
        }
        let mut columns = BTreeMap::new();
        for (name, data) in batch.columns() {
            let validity = data.validity();
            let has_null = validity.is_some_and(|v| v.contains(&0));
            let (values, flags) = if rows == 0 {
                (None, None)
            } else {
                let values = self.upload_bytes(&data.to_bytes());
                let flags: Vec<u8> = (0..batch.row_count())
                    .flat_map(|i| validity.map_or(1, |v| v[i]).to_ne_bytes())
                    .collect();
                (Some(values), Some(self.upload_bytes(&flags)))
            };
            columns.insert(
                name.into(),
                ResidentColumn {
                    field: Field::new(name, data.data_type()).with_nullable(validity.is_some()),
                    has_null,
                    values,
                    validity: flags,
                },
            );
        }
        Ok(DeviceBatch {
            client: self.client.clone(),
            rows: rows as u32,
            source: Arc::new(()),
            columns,
        })
    }
    /// Submit one program using resident inputs. Does not complete or read back.
    pub fn execute_resident(
        &self,
        program: &KernelProgram,
        batch: &DeviceBatch,
        parameters: &Parameters,
    ) -> Result<DeviceResult, ExecutionError> {
        self.execute_many_resident(std::slice::from_ref(program), batch, parameters)
    }
    /// Validate all bindings before submitting any program. Parameters are rebound
    /// on every call; snapshots and program identities remain independent.
    pub fn execute_many_resident(
        &self,
        programs: &[KernelProgram],
        batch: &DeviceBatch,
        parameters: &Parameters,
    ) -> Result<DeviceResult, ExecutionError> {
        // Also reject a foreign empty batch / empty program list.
        if self.client.service_id() != batch.client.service_id() {
            return Err(ExecutionError::ForeignDeviceBatch);
        }
        for program in programs {
            let layout = BindingLayout::new(program);
            batch.validate(&self.client, &layout, parameters)?;
            self.preflight_expression(&layout, u64::from(batch.rows))?;
        }
        let mut outputs = Vec::with_capacity(programs.len());
        for program in programs {
            outputs.push((
                program.output_type(),
                program.output_nullable(),
                self.submit_resident_expression(program, batch, parameters)?,
            ));
        }
        Ok(DeviceResult {
            client: self.client.clone(),
            counters: self.counters.clone(),
            outputs: Arc::new(outputs),
        })
    }
}

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod tests {
    use super::DeviceResult;
    use crate::{
        Batch, ColumnData, DataType, ExecutionError, Executor, Parameters, Schema, col, prepare,
    };
    use cubecl::Device;
    use std::sync::Arc;
    fn completion(device: Device) {
        let client = device.client();
        let executor = Executor::new(client.clone()).unwrap();
        let schema = Schema::new(&[("x", DataType::F32)], &[]).unwrap();
        let program = prepare(&col("x"), &schema).unwrap();
        let batch = executor
            .upload(
                &Batch::new(2)
                    .with_column("x", ColumnData::F32(&[1.0, 2.0]))
                    .unwrap(),
            )
            .unwrap();
        let mut result: DeviceResult = executor
            .execute_many_resident(
                &[program.clone(), program.clone()],
                &batch,
                &Parameters::new(),
            )
            .unwrap();
        // First prove both real outputs succeeded. Then a separate retained
        // resource fails verification; neither primary output carries its error.
        result.complete().unwrap();
        let marker =
            crate::selection::completion_tests::rejected_handle(&client, executor.settings());
        Arc::get_mut(&mut result.outputs).unwrap()[1]
            .2
            .retained
            .push(marker);
        let before = executor.stats();
        assert!(matches!(result.complete(), Err(ExecutionError::Backend(_))));
        assert!(matches!(
            result.materialize(),
            Err(ExecutionError::Backend(_))
        ));
        assert_eq!(executor.stats().readback_calls, before.readback_calls);
        executor
            .execute_resident(&program, &batch, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap();
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_multi_output_completion_checks_all_resources() {
        completion(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_multi_output_completion_checks_all_resources() {
        completion(Device::metal(Default::default()).expect("native Metal required"));
    }
}
