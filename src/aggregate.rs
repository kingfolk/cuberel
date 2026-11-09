//! Single-batch global COUNT/MIN/MAX with explicit device completion.
use crate::{
    DataType, DeviceBatch, DeviceProjection, DeviceSelection, ExecutionError, Executor,
    KernelProgram, Parameters,
    abi::BindingLayout,
    executor::SubmittedExpression,
    reduction_kernel::{ReductionKernel, ReductionKind},
    resident::ExecutionCounters,
};
use cubecl::{future::block_on, prelude::Client, server::Handle};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::Ordering},
};

/// Owned aggregate specification. MIN/MAX accept F32/I32; COUNT accepts every
/// current expression result type. No implicit numeric casts or host evaluation.
#[derive(Clone, Debug)]
pub enum Aggregate {
    CountStar,
    Count(KernelProgram),
    Min(KernelProgram),
    Max(KernelProgram),
}
impl Aggregate {
    fn program(&self) -> Option<&KernelProgram> {
        match self {
            Self::CountStar => None,
            Self::Count(p) | Self::Min(p) | Self::Max(p) => Some(p),
        }
    }
    fn kind(&self) -> Result<Option<ReductionKind>, ExecutionError> {
        Ok(Some(match self {
            Self::CountStar => return Ok(None),
            Self::Count(_) => ReductionKind::Count,
            Self::Min(p) if p.output_type() == DataType::I32 => ReductionKind::MinI32,
            Self::Max(p) if p.output_type() == DataType::I32 => ReductionKind::MaxI32,
            Self::Min(p) if p.output_type() == DataType::F32 => ReductionKind::MinF32,
            Self::Max(p) if p.output_type() == DataType::F32 => ReductionKind::MaxF32,
            _ => {
                return Err(ExecutionError::InvalidAggregation {
                    reason: "MIN/MAX require I32 or F32 arguments",
                });
            }
        }))
    }
}

/// COUNT is host U64, but the device count/row ABI remains bounded U32.
/// Empty/all-NULL MIN/MAX produce None; F32 valid NaNs have canonical bits.
#[derive(Clone, Debug, PartialEq)]
pub enum AggregateValue {
    Count(u64),
    I32(Option<i32>),
    F32(Option<f32>),
}
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateResult {
    pub name: String,
    pub value: AggregateValue,
}

/// Logical reduction allocations/submissions, excluding expression, selection,
/// input buffers, runtime staging and allocator caches; not physical peak memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AggregateStats {
    pub reduction_levels: usize,
    pub reduction_launches: usize,
    pub partial_state_bytes: u64,
}
struct Entry {
    name: String,
    kind: Option<ReductionKind>,
    state: Option<(Handle, Handle)>,
}
enum Dependency {
    Dense(Arc<SubmittedExpression>),
    Selected(Arc<DeviceProjection>),
}

/// Pending global aggregates over one resident batch/selection. Keeps every
/// intermediate and argument/error dependency alive. No implicit CPU fallback.
#[derive(Clone)]
pub struct DeviceAggregates {
    client: Client,
    counters: Arc<ExecutionCounters>,
    rows: u32,
    selection: Option<DeviceSelection>,
    entries: Arc<Vec<Entry>>,
    dependencies: Arc<Vec<Dependency>>,
    retained: Arc<Vec<Handle>>,
    stats: AggregateStats,
}
impl DeviceAggregates {
    pub fn stats(&self) -> AggregateStats {
        self.stats
    }
    /// Block and check all retained resources and expression/selection errors.
    /// Returns participating row count; values remain on device. Checked I32
    /// arguments and selected dependencies may require error/count readback.
    pub fn complete(&self) -> Result<u32, ExecutionError> {
        if !self.retained.is_empty() {
            block_on(self.client.sync_buffers(self.retained.iter()))
                .map_err(ExecutionError::Backend)?;
        }
        let rows = match &self.selection {
            Some(s) => s.selected_count()?,
            None => self.rows,
        };
        let mut first_error = None;
        for dependency in self.dependencies.iter() {
            let checked = match dependency {
                Dependency::Dense(input) => input.check_errors(&self.client, &self.counters),
                Dependency::Selected(input) => input.complete().map(|_| ()),
            };
            match checked {
                Err(ExecutionError::IntegerOverflow { row }) => {
                    first_error = Some(first_error.map_or(row, |old: usize| old.min(row)))
                }
                Err(error) => return Err(error),
                Ok(()) => {}
            }
        }
        if let Some(row) = first_error {
            return Err(ExecutionError::IntegerOverflow { row });
        }
        Ok(rows)
    }
    /// Return one named scalar per aggregate, in request order, only after all
    /// completion/error checks succeed. Repeated calls repeat checks/readback.
    pub fn materialize(&self) -> Result<Vec<AggregateResult>, ExecutionError> {
        let rows = self.complete()?;
        self.entries
            .iter()
            .map(|entry| {
                let value = match entry.kind {
                    None => AggregateValue::Count(u64::from(rows)),
                    Some(kind) => {
                        let (bits, count) = match &entry.state {
                            Some((value, count)) => {
                                let count = self.read_word(count)?;
                                if count > rows {
                                    return Err(ExecutionError::InvalidAggregateCount {
                                        count,
                                        rows,
                                    });
                                }
                                let bits = if count != 0 && kind != ReductionKind::Count {
                                    self.read_word(value)?
                                } else {
                                    0
                                };
                                (bits, count)
                            }
                            None => (0, 0),
                        };
                        match kind {
                            ReductionKind::Count => AggregateValue::Count(u64::from(count)),
                            ReductionKind::MinI32 | ReductionKind::MaxI32 => {
                                AggregateValue::I32((count != 0).then_some(bits as i32))
                            }
                            ReductionKind::MinF32 | ReductionKind::MaxF32 => {
                                AggregateValue::F32((count != 0).then_some(f32::from_bits(bits)))
                            }
                        }
                    }
                };
                Ok(AggregateResult {
                    name: entry.name.clone(),
                    value,
                })
            })
            .collect()
    }
    fn read_word(&self, handle: &Handle) -> Result<u32, ExecutionError> {
        let bytes = self.counters.read(&self.client, handle.clone(), 4)?;
        Ok(u32::from_ne_bytes(bytes.as_chunks::<4>().0[0]))
    }
}

impl Executor {
    /// Submit named global aggregates over every logical row of a resident batch.
    pub fn aggregate_resident(
        &self,
        batch: &DeviceBatch,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<DeviceAggregates, ExecutionError> {
        self.aggregate_impl(batch, None, aggregates, parameters)
    }
    /// Submit named global aggregates only over selected original rows. No count
    /// is read on submission; arguments retain F3 active-row/error semantics.
    pub fn aggregate_selected(
        &self,
        batch: &DeviceBatch,
        selection: &DeviceSelection,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<DeviceAggregates, ExecutionError> {
        self.aggregate_impl(batch, Some(selection), aggregates, parameters)
    }
    fn aggregate_impl(
        &self,
        batch: &DeviceBatch,
        selection: Option<&DeviceSelection>,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<DeviceAggregates, ExecutionError> {
        batch.validate_client(&self.client)?;
        if selection.is_some_and(|s| !s.belongs_to(batch)) {
            return Err(ExecutionError::SelectionSourceMismatch);
        }
        if aggregates.is_empty() {
            return Err(ExecutionError::InvalidAggregation {
                reason: "at least one named aggregate is required",
            });
        }
        let mut names = BTreeSet::new();
        let mut expressions = 0usize;
        for &(name, ref aggregate) in aggregates {
            if name.is_empty() || !names.insert(name) {
                return Err(ExecutionError::InvalidAggregation {
                    reason: "aggregate names must be nonempty and unique",
                });
            }
            aggregate.kind()?;
            if let Some(program) = aggregate.program() {
                expressions += 1;
                let layout = BindingLayout::new(program);
                batch.validate(&self.client, &layout, parameters)?;
                self.preflight_expression(&layout, u64::from(batch.row_count()))?;
                if selection.is_some()
                    && batch.row_count() != 0
                    && layout.buffers().len() as u64 + 3
                        > u64::from(self.client.properties().hardware.max_bindings)
                {
                    return Err(ExecutionError::UnsupportedDevice {
                        reason: "selected aggregate argument exceeds binding limit",
                    });
                }
            }
        }
        let plan = ReductionPlan::new(
            &self.client,
            batch.row_count(),
            self.cube_size(),
            expressions,
        )?;
        let mut entries = Vec::with_capacity(aggregates.len());
        let mut retained = Vec::new();
        let mut dependencies = Vec::new();
        for &(name, ref aggregate) in aggregates {
            let kind = aggregate.kind()?;
            let mut state = None;
            if let Some(program) = aggregate.program() {
                let (values, validity) = if let Some(selection) = selection {
                    let input = Arc::new(self.project_selected(
                        batch,
                        selection,
                        &[(name, program)],
                        parameters,
                    )?);
                    let submitted = &input.outputs[0].submitted;
                    retained.extend(submitted.retained.iter().cloned());
                    let planes = (submitted.values.clone(), submitted.validity.clone());
                    dependencies.push(Dependency::Selected(input));
                    planes
                } else {
                    let input =
                        Arc::new(self.submit_resident_expression(program, batch, parameters)?);
                    retained.extend(input.retained.iter().cloned());
                    let planes = (input.values.clone(), input.validity.clone());
                    dependencies.push(Dependency::Dense(input));
                    planes
                };
                if let Some(mut values) = values {
                    let mut counts = validity.unwrap_or_else(|| values.clone());
                    for (level, &(rows, blocks)) in plan.levels.iter().enumerate() {
                        let out_value = self.client.empty(blocks as usize * 4);
                        let out_count = self.client.empty(blocks as usize * 4);
                        let selected_count = if level == 0 {
                            selection
                                .map(|s| s.count_handle())
                                .unwrap_or_else(|| values.clone())
                        } else {
                            values.clone()
                        };
                        let kernel = ReductionKernel::new(
                            kind.expect("expression kind"),
                            level != 0,
                            program.output_nullable(),
                            selection.is_some() && level == 0,
                            self.client.clone(),
                            self.settings(),
                            self.define_count.clone(),
                        );
                        kernel.launch(
                            rows,
                            &[
                                values,
                                counts,
                                selected_count,
                                out_value.clone(),
                                out_count.clone(),
                            ],
                        );
                        self.counters.launches.fetch_add(1, Ordering::Relaxed);
                        retained.extend([out_value.clone(), out_count.clone()]);
                        values = out_value;
                        counts = out_count;
                    }
                    state = Some((values, counts));
                }
            }
            entries.push(Entry {
                name: name.into(),
                kind,
                state,
            });
        }
        Ok(DeviceAggregates {
            client: self.client.clone(),
            counters: self.counters.clone(),
            rows: batch.row_count(),
            selection: selection.cloned(),
            entries: Arc::new(entries),
            dependencies: Arc::new(dependencies),
            retained: Arc::new(retained),
            stats: plan.stats,
        })
    }
}
struct ReductionPlan {
    levels: Vec<(u32, u32)>,
    stats: AggregateStats,
}
impl ReductionPlan {
    fn new(
        client: &Client,
        rows: u32,
        tile: u32,
        expressions: usize,
    ) -> Result<Self, ExecutionError> {
        let mut levels = Vec::new();
        let mut bytes = 0u64;
        if rows != 0 && expressions != 0 {
            let hw = &client.properties().hardware;
            if tile < 2
                || tile.checked_next_power_of_two().is_none()
                || (tile as usize)
                    .checked_mul(8)
                    .is_none_or(|n| n > hw.max_shared_memory_size)
                || hw.max_bindings < 6
            {
                return Err(ExecutionError::UnsupportedDevice {
                    reason: "reduction needs >=2 lanes, tile*8 shared bytes and 6 bindings",
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
                let padded = u64::from(blocks) * u64::from(tile);
                if padded > u64::from(u32::MAX) {
                    return Err(crate::abi::BindingError::AddressOverflow {
                        field: "reduction_padded_grid",
                        value: padded,
                    }
                    .into());
                }
                bytes = bytes.checked_add(u64::from(blocks) * 8).ok_or(
                    ExecutionError::InvalidAggregation {
                        reason: "partial-state byte sum overflow",
                    },
                )?;
                levels.push((len, blocks));
                if blocks == 1 {
                    break;
                }
                len = blocks;
            }
        }
        let stats = AggregateStats {
            reduction_levels: levels.len(),
            reduction_launches: levels.len().checked_mul(expressions).ok_or(
                ExecutionError::InvalidAggregation {
                    reason: "reduction launch count overflow",
                },
            )?,
            partial_state_bytes: bytes.checked_mul(expressions as u64).ok_or(
                ExecutionError::InvalidAggregation {
                    reason: "partial-state request size overflow",
                },
            )?,
        };
        Ok(Self { levels, stats })
    }
}

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod tests {
    use crate::{
        Aggregate, AggregateValue, Batch, ColumnData, DataType, ExecutionError, Executor,
        Parameters, Schema, col, prepare,
    };
    use cubecl::Device;
    use std::sync::Arc;
    fn completion(device: Device) {
        let client = device.client();
        let e = Executor::new(client.clone()).unwrap();
        let schema = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
        let x = prepare(&col("x"), &schema).unwrap();
        let requests = [
            ("valid", Aggregate::Count(x)),
            ("rows", Aggregate::CountStar),
        ];
        let batch = e
            .upload(
                &Batch::new(3)
                    .with_column("x", ColumnData::I32(&[1, 2, 3]))
                    .unwrap(),
            )
            .unwrap();
        let mut out = e
            .aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap();
        assert_eq!(
            out.materialize().unwrap()[0].value,
            AggregateValue::Count(3)
        );
        // Corrupt only an initialized counter, never device addresses or loads.
        let counter = client.create_from_slice(&4_u32.to_ne_bytes());
        Arc::get_mut(&mut out.entries).unwrap()[0]
            .state
            .as_mut()
            .unwrap()
            .1 = counter.clone();
        Arc::get_mut(&mut out.retained).unwrap().push(counter);
        assert!(matches!(
            out.materialize(),
            Err(ExecutionError::InvalidAggregateCount { count: 4, rows: 3 })
        ));
        let mut out = e
            .aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap();
        out.complete().unwrap();
        let marker = crate::selection::completion_tests::rejected_handle(&client, e.settings());
        Arc::get_mut(&mut out.retained).unwrap().push(marker);
        let before = e.stats();
        assert!(matches!(out.materialize(), Err(ExecutionError::Backend(_))));
        assert_eq!(e.stats(), before);
        assert_eq!(
            e.aggregate_resident(&batch, &requests, &Parameters::new())
                .unwrap()
                .materialize()
                .unwrap()[0]
                .value,
            AggregateValue::Count(3)
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_aggregate_completion_checks_all_resources() {
        completion(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_aggregate_completion_checks_all_resources() {
        completion(Device::metal(Default::default()).expect("Metal required"));
    }
}
