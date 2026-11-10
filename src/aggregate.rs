//! Single-batch global COUNT/MIN/MAX/SUM with explicit device completion.
use crate::completion::{ExitContext, synchronize};
use crate::{
    DataType, DeviceBatch, DeviceProjection, DeviceSelection, ExecutionError, Executor,
    KernelProgram, Parameters,
    abi::BindingLayout,
    executor::SubmittedExpression,
    reduction_kernel::{ReductionKernel, ReductionKind},
    resident::ExecutionCounters,
};
use cubecl::{prelude::Client, server::Handle};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::Ordering},
};

/// Owned aggregate specification. MIN/MAX accept F32/I32; COUNT accepts every
/// current expression result type. SUM widens I32 to I64; F32 stays F32.
#[derive(Clone, Debug)]
pub enum Aggregate {
    CountStar,
    Count(KernelProgram),
    Min(KernelProgram),
    Max(KernelProgram),
    Sum(KernelProgram),
}
impl Aggregate {
    pub(crate) fn program(&self) -> Option<&KernelProgram> {
        match self {
            Self::CountStar => None,
            Self::Count(p) | Self::Min(p) | Self::Max(p) | Self::Sum(p) => Some(p),
        }
    }
    pub(crate) fn kind(&self) -> Result<Option<ReductionKind>, ExecutionError> {
        Ok(Some(match self {
            Self::CountStar => return Ok(None),
            Self::Count(_) => ReductionKind::Count,
            Self::Min(p) if p.output_type() == DataType::I32 => ReductionKind::MinI32,
            Self::Max(p) if p.output_type() == DataType::I32 => ReductionKind::MaxI32,
            Self::Min(p) if p.output_type() == DataType::F32 => ReductionKind::MinF32,
            Self::Max(p) if p.output_type() == DataType::F32 => ReductionKind::MaxF32,
            Self::Sum(p) if p.output_type() == DataType::I32 => ReductionKind::SumI64,
            Self::Sum(p) if p.output_type() == DataType::F32 => ReductionKind::SumF32,
            _ => {
                return Err(ExecutionError::InvalidAggregation {
                    reason: "MIN/MAX/SUM require I32 or F32 arguments",
                });
            }
        }))
    }
}

/// COUNT is host U64, but the device count/row ABI remains bounded U32.
/// Empty/all-NULL MIN/MAX/SUM produce None; F32 valid NaNs have canonical bits.
#[derive(Clone, Debug, PartialEq)]
pub enum AggregateValue {
    Count(u64),
    I32(Option<i32>),
    I64(Option<i64>),
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
struct AggregateState {
    low: Handle,
    high: Option<Handle>,
    count: Handle,
}
struct Entry {
    name: String,
    kind: Option<ReductionKind>,
    state: Option<AggregateState>,
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
    predicate_checks: Arc<Vec<SubmittedExpression>>,
    row_count_state: Option<Handle>,
}
impl DeviceAggregates {
    pub fn stats(&self) -> AggregateStats {
        self.stats
    }
    /// Block and check all retained resources and expression/selection errors.
    /// Returns participating row count; values remain on device. Checked I32
    /// arguments and selected dependencies may require error/count readback.
    pub fn complete(&self) -> Result<u32, ExecutionError> {
        synchronize(&self.client, &self.handles())?;
        self.complete_in(&mut ExitContext::default())
    }
    pub(crate) fn handles(&self) -> Vec<Handle> {
        let mut handles = self.retained.as_ref().clone();
        if let Some(selection) = &self.selection {
            handles.extend(selection.handles());
        }
        handles
    }
    pub(crate) fn complete_in(&self, exit: &mut ExitContext) -> Result<u32, ExecutionError> {
        for predicate in self.predicate_checks.iter() {
            predicate.check_errors(&self.client, &self.counters)?;
        }
        let rows = match &self.selection {
            Some(s) => exit.count(s)?,
            None => {
                if let Some(count) = &self.row_count_state {
                    let count = self.read_word(count)?;
                    if count > self.rows {
                        return Err(ExecutionError::InvalidAggregateCount {
                            count,
                            rows: self.rows,
                        });
                    }
                    count
                } else {
                    self.rows
                }
            }
        };
        let mut first_error = None;
        for dependency in self.dependencies.iter() {
            let checked = match dependency {
                Dependency::Dense(input) => input.check_errors(&self.client, &self.counters),
                Dependency::Selected(input) => input.complete_in(exit).map(|_| ()),
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
        self.read_results(self.complete()?)
    }
    pub(crate) fn read_results(&self, rows: u32) -> Result<Vec<AggregateResult>, ExecutionError> {
        self.entries
            .iter()
            .map(|entry| {
                let value = match entry.kind {
                    None => AggregateValue::Count(u64::from(rows)),
                    Some(kind) => {
                        let (bits, high, count) = match &entry.state {
                            Some(state) => {
                                let count = self.read_word(&state.count)?;
                                if count > rows {
                                    return Err(ExecutionError::InvalidAggregateCount {
                                        count,
                                        rows,
                                    });
                                }
                                let bits = if count != 0 && kind != ReductionKind::Count {
                                    self.read_word(&state.low)?
                                } else {
                                    0
                                };
                                let high = if count != 0 && kind.needs_high() {
                                    self.read_word(state.high.as_ref().expect("wide state"))?
                                } else {
                                    0
                                };
                                (bits, high, count)
                            }
                            None => (0, 0, 0),
                        };
                        match kind {
                            ReductionKind::Count => AggregateValue::Count(u64::from(count)),
                            ReductionKind::MinI32 | ReductionKind::MaxI32 => {
                                AggregateValue::I32((count != 0).then_some(bits as i32))
                            }
                            ReductionKind::SumI64 => AggregateValue::I64(
                                (count != 0)
                                    .then_some(((u64::from(high) << 32) | u64::from(bits)) as i64),
                            ),
                            ReductionKind::MinF32
                            | ReductionKind::MaxF32
                            | ReductionKind::SumF32 => {
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
    pub(crate) fn preflight_aggregates(
        &self,
        batch: &DeviceBatch,
        selected: bool,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<ReductionPlan, ExecutionError> {
        batch.validate_client(&self.client)?;
        if aggregates.is_empty() {
            return Err(ExecutionError::InvalidAggregation {
                reason: "at least one named aggregate is required",
            });
        }
        let mut names = BTreeSet::new();
        let mut expressions = 0usize;
        let mut state_words = 0u64;
        let mut max_state_words = 0usize;
        for &(name, ref aggregate) in aggregates {
            if name.is_empty() || !names.insert(name) {
                return Err(ExecutionError::InvalidAggregation {
                    reason: "aggregate names must be nonempty and unique",
                });
            }
            if let Some(kind) = aggregate.kind()? {
                let words = kind.state_words();
                state_words = state_words.checked_add(words as u64).ok_or(
                    ExecutionError::InvalidAggregation {
                        reason: "request state width overflow",
                    },
                )?;
                max_state_words = max_state_words.max(words);
            }
            if let Some(program) = aggregate.program() {
                expressions += 1;
                let layout = BindingLayout::new(program);
                batch.validate(&self.client, &layout, parameters)?;
                self.preflight_expression(&layout, u64::from(batch.row_count()))?;
                if selected
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
        ReductionPlan::new(
            &self.client,
            batch.row_count(),
            self.cube_size(),
            expressions,
            state_words,
            max_state_words,
        )
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
        let plan = self.preflight_aggregates(batch, selection.is_some(), aggregates, parameters)?;
        let mut account = crate::budget::Account::new(
            u64::from(batch.row_count()),
            crate::budget::inputs(batch, selection)?,
            1,
        );
        account.aggregates(aggregates, &plan)?;
        self.enforce(account.finish()?)?;
        let engine = self.without_budget();
        let mut entries = Vec::with_capacity(aggregates.len());
        let mut retained = Vec::new();
        let mut dependencies = Vec::new();
        for &(name, ref aggregate) in aggregates {
            let kind = aggregate.kind()?;
            let mut state = None;
            if let Some(program) = aggregate.program() {
                let (values, validity) = if let Some(selection) = selection {
                    let input = Arc::new(engine.project_selected(
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
                        Arc::new(engine.submit_resident_expression(program, batch, parameters)?);
                    retained.extend(input.retained.iter().cloned());
                    let planes = (input.values.clone(), input.validity.clone());
                    dependencies.push(Dependency::Dense(input));
                    planes
                };
                if let Some(mut values) = values {
                    let kind = kind.expect("expression kind");
                    let mut high = None;
                    let mut counts = validity.unwrap_or_else(|| values.clone());
                    for (level, &(rows, blocks)) in plan.levels.iter().enumerate() {
                        let out_value = self.allocate(blocks as usize * 4);
                        let out_count = self.allocate(blocks as usize * 4);
                        let out_high = kind
                            .needs_high()
                            .then(|| self.allocate(blocks as usize * 4));
                        let selected_count = if level == 0 {
                            selection
                                .map(|s| s.count_handle())
                                .unwrap_or_else(|| values.clone())
                        } else {
                            values.clone()
                        };
                        let kernel = ReductionKernel::new(
                            kind,
                            level != 0,
                            program.output_nullable(),
                            selection.is_some() && level == 0,
                            self.client.clone(),
                            self.settings(),
                            self.define_count.clone(),
                        );
                        let input_high = high.clone().unwrap_or_else(|| values.clone());
                        let mut buffers = vec![
                            values,
                            counts,
                            selected_count,
                            out_value.clone(),
                            out_count.clone(),
                        ];
                        if let Some(out_high) = &out_high {
                            buffers.extend([input_high, out_high.clone()]);
                        }
                        kernel.launch(rows, &buffers);
                        self.counters.launches.fetch_add(1, Ordering::Relaxed);
                        retained.extend([out_value.clone(), out_count.clone()]);
                        retained.extend(out_high.iter().cloned());
                        high = out_high;
                        values = out_value;
                        counts = out_count;
                    }
                    state = Some(AggregateState {
                        low: values,
                        high,
                        count: counts,
                    });
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
            predicate_checks: Arc::new(Vec::new()),
            row_count_state: None,
        })
    }
}
pub(crate) struct ReductionPlan {
    pub(crate) levels: Vec<(u32, u32)>,
    pub(crate) stats: AggregateStats,
}
impl ReductionPlan {
    pub(crate) fn new(
        client: &Client,
        rows: u32,
        tile: u32,
        expressions: usize,
        state_words: u64,
        max_state_words: usize,
    ) -> Result<Self, ExecutionError> {
        let mut levels = Vec::new();
        let mut bytes = 0u64;
        if rows != 0 && expressions != 0 {
            let hw = &client.properties().hardware;
            if tile < 2
                || tile.checked_next_power_of_two().is_none()
                || (tile as usize)
                    .checked_mul(max_state_words * 4)
                    .is_none_or(|n| n > hw.max_shared_memory_size)
                || hw.max_bindings < if max_state_words == 3 { 8 } else { 6 }
            {
                return Err(ExecutionError::UnsupportedDevice {
                    reason: "reduction needs >=2 lanes and sufficient shared memory/bindings",
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
                bytes = bytes.checked_add(u64::from(blocks) * 4).ok_or(
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
            partial_state_bytes: bytes.checked_mul(state_words).ok_or(
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
            .count = counter.clone();
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

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod sum_completion_tests {
    use crate::{
        Aggregate, AggregateValue, Batch, ColumnData, DataType, ExecutionError, Executor,
        Parameters, Schema, col, prepare,
    };
    use cubecl::Device;
    fn exercise(device: Device) {
        let client = device.client();
        let e = Executor::new(client.clone()).unwrap();
        let schema = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
        let x = prepare(&col("x"), &schema).unwrap();
        let req = [("sum", Aggregate::Sum(x))];
        let batch = e
            .upload(
                &Batch::new(3)
                    .with_column("x", ColumnData::I32(&[i32::MAX; 3]))
                    .unwrap(),
            )
            .unwrap();
        let out = e
            .aggregate_resident(&batch, &req, &Parameters::new())
            .unwrap();
        let before = e.stats();
        assert_eq!(
            out.materialize().unwrap()[0].value,
            AggregateValue::I64(Some(i64::from(i32::MAX) * 3))
        );
        assert_eq!(e.stats().readback_bytes - before.readback_bytes, 12); // count, low and high
        let high = out.entries[0]
            .state
            .as_ref()
            .unwrap()
            .high
            .as_ref()
            .unwrap()
            .clone();
        // Only the high word is tainted by a verifier-rejected write. The low
        // and count outputs really succeeded; completion must check all three.
        crate::selection::completion_tests::reject_output(&client, e.settings(), high);
        let before = e.stats();
        assert!(matches!(
            out.clone().complete(),
            Err(ExecutionError::Backend(_))
        ));
        assert!(matches!(out.materialize(), Err(ExecutionError::Backend(_))));
        assert_eq!(e.stats(), before);
        assert_eq!(
            e.aggregate_resident(&batch, &req, &Parameters::new())
                .unwrap()
                .materialize()
                .unwrap()[0]
                .value,
            AggregateValue::I64(Some(i64::from(i32::MAX) * 3))
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_sum_high_word_failure_is_observed() {
        exercise(Device::cpu().expect("CPU required"));
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_sum_high_word_failure_is_observed() {
        exercise(Device::metal(Default::default()).expect("Metal required"));
    }
}

#[path = "aggregate/fused.rs"]
pub(crate) mod fused;
