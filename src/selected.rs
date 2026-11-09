//! Count-guarded selected projection and compact, owned host columns.
use crate::{
    DeviceBatch, DeviceSelection, ExecutionError, Executor, Field, HostColumn, KernelProgram,
    Parameters,
    abi::BindingLayout,
    executor::{OutputBytes, SubmittedExpression},
    resident::{ExecutionCounters, decode_host_column},
};
use cubecl::{bytes::Bytes, future::block_on, prelude::Client, server::Handle};
use std::{collections::BTreeSet, sync::Arc};

/// A compact completed column with its output name, type and nullability.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedColumn {
    pub(crate) field: Field,
    pub(crate) data: HostColumn,
}
impl SelectedColumn {
    pub fn field(&self) -> &Field {
        &self.field
    }
    pub fn data(&self) -> &HostColumn {
        &self.data
    }
    pub fn into_data(self) -> HostColumn {
        self.data
    }
}

pub(crate) struct ProjectionOutput {
    pub(crate) field: Field,
    pub(crate) submitted: SubmittedExpression,
}

/// Pending selected expressions. Allocations use source capacity; only device
/// count rows are initialized. No count or payload is read during submission.
/// Clones share immutable outputs and all selection/error dependencies.
#[derive(Clone)]
pub struct DeviceProjection {
    client: Client,
    counters: Arc<ExecutionCounters>,
    pub(crate) selection: DeviceSelection,
    pub(crate) outputs: Arc<Vec<ProjectionOutput>>,
}
impl DeviceProjection {
    pub fn capacity(&self) -> u32 {
        self.selection.capacity()
    }
    pub fn fields(&self) -> impl ExactSizeIterator<Item = &Field> {
        self.outputs.iter().map(|o| &o.field)
    }
    /// Block, check all dependencies, read device count and initialized error
    /// prefixes. Returns the logical row count, never allocation capacity.
    /// Overflow diagnostics use the smallest original row across all outputs.
    pub fn complete(&self) -> Result<u32, ExecutionError> {
        if self.capacity() != 0 {
            block_on(
                self.client.sync_buffers(
                    self.outputs
                        .iter()
                        .flat_map(|o| o.submitted.retained.iter()),
                ),
            )
            .map_err(ExecutionError::Backend)?;
        }
        let count = self.selection.selected_count()?;
        let mut first = None;
        if count != 0 {
            for output in self.outputs.iter() {
                if let Some(error) = &output.submitted.error {
                    let bytes = self.read_prefix(error, count)?;
                    for bytes in bytes.as_chunks::<4>().0 {
                        let row = u32::from_ne_bytes(*bytes);
                        if row != 0 {
                            first = Some(first.map_or(row - 1, |old: u32| old.min(row - 1)));
                        }
                    }
                }
            }
        }
        if let Some(row) = first {
            return Err(ExecutionError::IntegerOverflow { row: row as usize });
        }
        Ok(count)
    }
    /// Export only initialized count prefixes. Values and validity are packed
    /// into owned host columns only after every dependency/error check succeeds.
    pub fn materialize(&self) -> Result<Vec<SelectedColumn>, ExecutionError> {
        let count = self.complete()?;
        self.outputs
            .iter()
            .map(|output| {
                let values = self.read_optional(&output.submitted.values, count)?;
                let validity = if output.field.is_nullable() {
                    Some(self.read_optional(&output.submitted.validity, count)?)
                } else {
                    None
                };
                let data = decode_host_column(
                    output.field.data_type(),
                    OutputBytes { values, validity },
                    count as usize,
                )?;
                Ok(SelectedColumn {
                    field: output.field.clone(),
                    data,
                })
            })
            .collect()
    }
    fn read_optional(&self, handle: &Option<Handle>, count: u32) -> Result<Bytes, ExecutionError> {
        if count == 0 {
            return Ok(Bytes::from_bytes_vec(Vec::new()));
        }
        self.read_prefix(handle.as_ref().expect("nonempty projection plane"), count)
    }
    fn read_prefix(&self, handle: &Handle, count: u32) -> Result<Bytes, ExecutionError> {
        let handle = handle
            .clone()
            .offset_end(u64::from(self.capacity() - count) * 4);
        self.counters.read(&self.client, handle, count as usize * 4)
    }
}

impl Executor {
    /// Compute named expressions only on selected original rows. All bindings,
    /// names, device/source identity and launch limits validate before submission.
    /// Each expression has an independent kernel, sharing the input selection.
    pub fn project_selected(
        &self,
        batch: &DeviceBatch,
        selection: &DeviceSelection,
        programs: &[(&str, &KernelProgram)],
        parameters: &Parameters,
    ) -> Result<DeviceProjection, ExecutionError> {
        if !selection.belongs_to(batch) {
            return Err(ExecutionError::SelectionSourceMismatch);
        }
        if programs.is_empty() {
            return Err(ExecutionError::EmptyProjection);
        }
        let mut names = BTreeSet::new();
        for &(name, program) in programs {
            if name.is_empty() || !names.insert(name) {
                return Err(ExecutionError::InvalidProjectionName);
            }
            let layout = BindingLayout::new(program);
            batch.validate(&self.client, &layout, parameters)?;
            self.preflight_expression(&layout, u64::from(batch.row_count()))?;
            if batch.row_count() != 0
                && layout.buffers().len() as u64 + 3
                    > u64::from(self.client.properties().hardware.max_bindings)
            {
                return Err(ExecutionError::UnsupportedDevice {
                    reason: "selected expression exceeds buffer binding limit",
                });
            }
        }
        let mut outputs = Vec::with_capacity(programs.len());
        for &(name, program) in programs {
            outputs.push(ProjectionOutput {
                field: Field::new(name, program.output_type())
                    .with_nullable(program.output_nullable()),
                submitted: self
                    .submit_selected_expression(program, batch, parameters, selection)?,
            });
        }
        Ok(DeviceProjection {
            client: self.client.clone(),
            counters: self.counters.clone(),
            selection: selection.clone(),
            outputs: Arc::new(outputs),
        })
    }
}

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod tests {
    use crate::{
        Batch, ColumnData, DataType, ExecutionError, Executor, Parameters, Schema, col,
        emitter::ExpressionKernel, param, prepare,
    };
    use cubecl::{Device, prelude::KernelMetadata};
    use std::sync::Arc;
    fn completion(device: Device) {
        let client = device.client();
        let e = Executor::new(client.clone()).unwrap();
        let schema = Schema::new(&[("x", DataType::F32)], &[("limit", DataType::F32)]).unwrap();
        let copy = prepare(&col("x"), &schema).unwrap();
        let predicate = prepare(&col("x").gt(param("limit")), &schema).unwrap();
        let plain = ExpressionKernel::new(
            copy.clone(),
            client.clone(),
            e.settings(),
            e.define_count.clone(),
        );
        let selected = ExpressionKernel::selected(
            copy.clone(),
            client.clone(),
            e.settings(),
            e.define_count.clone(),
        );
        assert_ne!(plain.id(), selected.id());
        let batch = e
            .upload(
                &Batch::new(3)
                    .with_column("x", ColumnData::F32(&[1.0, 2.0, 3.0]))
                    .unwrap(),
            )
            .unwrap();
        let s = e
            .select_resident(
                &predicate,
                &batch,
                &Parameters::new().with("limit", 1.0_f32).unwrap(),
            )
            .unwrap();
        let mut out = e
            .project_selected(
                &batch,
                &s,
                &[("first", &copy), ("second", &copy)],
                &Parameters::new(),
            )
            .unwrap();
        assert_eq!(out.complete().unwrap(), 2);
        // Successful value planes do not make an independently failed retained
        // resource successful. The fixture is rejected before device lowering.
        let marker = crate::selection::completion_tests::rejected_handle(&client, e.settings());
        Arc::get_mut(&mut out.outputs).unwrap()[1]
            .submitted
            .retained
            .push(marker);
        let before = e.stats();
        assert!(matches!(out.materialize(), Err(ExecutionError::Backend(_))));
        assert_eq!(e.stats(), before);
        assert_eq!(
            e.project_selected(&batch, &s, &[("fresh", &copy)], &Parameters::new())
                .unwrap()
                .complete()
                .unwrap(),
            2
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_selected_completion_checks_all_resources() {
        completion(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_selected_completion_checks_all_resources() {
        completion(Device::metal(Default::default()).expect("Metal required"));
    }
}
