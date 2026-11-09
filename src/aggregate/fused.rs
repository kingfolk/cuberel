//! Aggregate-only fusion. Predicate evaluation and argument evaluation retain
//! separate error planes and active-row boundaries; merge kernels stay unchanged.
use super::{AggregateState, Dependency, DeviceAggregates, Entry, ReductionPlan};
use crate::{
    Aggregate, DeviceBatch, ExecutionError, Executor, ExprId, KernelProgram, Parameters,
    abi::{BindingLayout, BufferAccess, BufferSource, ScalarArgument},
    cubecl_abi::LauncherGuard,
    emitter::{ExpressionKernel, emit_scalar},
    executor::SubmittedExpression,
    expr::merge_programs,
    ir_support::{barrier, branch, builtin, load, store, uint},
    reduction_kernel::{ReductionKernel, ReductionKind, canonical_float, combine, combine_i64},
};
use cubecl::{
    ir::{
        Builtin, UIntKind,
        dialect::{
            branch::ReturnOp,
            cmp::{IEqualOp, UGreaterThanOp, ULessThanOp},
            general::{BoolAndOp, CastOp, ReinterpretCastOp, SelectOp},
            math::{IAddOp, URemOp},
            memory::{LoadOp, StoreOp},
        },
        pliron::{op::Op, operation::verify_operation},
        types::ArrayType,
    },
    prelude::{
        BufferArg, BufferIOAttr, Client, CubeCount, CubeKernel, ElemType, KernelDefinition,
        KernelId, KernelLauncher, KernelMetadata, KernelSettings,
    },
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Spec {
    kind: ReductionKind,
    root: Option<ExprId>,
}
pub(crate) struct FusedPlan {
    union: KernelProgram,
    predicate: Option<KernelProgram>,
    arguments: Option<KernelProgram>,
    specs: Vec<Spec>,
    entry_states: Vec<Option<usize>>,
    row_state: Option<usize>,
    predicate_error: bool,
    argument_error: bool,
    reduction: ReductionPlan,
}
impl FusedPlan {
    pub(crate) fn prepare(
        executor: &Executor,
        batch: &DeviceBatch,
        predicate: Option<&KernelProgram>,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<Self, &'static str> {
        if predicate.is_some()
            && aggregates.iter().any(
                |(_, a)| matches!(a, Aggregate::Sum(p) if p.output_type() == crate::DataType::F32),
            )
        {
            return Err("filtered F32 SUM requires the compact selection reduction tree");
        }
        let args: Vec<_> = aggregates.iter().filter_map(|(_, a)| a.program()).collect();
        let mut all = args.clone();
        all.extend(predicate);
        let (union, _) = merge_programs(&all)?;
        let (arguments, roots) = if args.is_empty() {
            (None, Vec::new())
        } else {
            let (program, roots) = merge_programs(&args)?;
            (Some(program.rebind_schema(union.schema())), roots)
        };
        let predicate = predicate.map(|p| p.rebind_schema(union.schema()));
        let has_error = |p: &KernelProgram| {
            BindingLayout::new(p)
                .buffers()
                .iter()
                .any(|s| s.source() == BufferSource::OutputError)
        };
        let predicate_error = predicate.as_ref().is_some_and(has_error);
        let argument_error = arguments.as_ref().is_some_and(has_error);
        let mut specs = Vec::new();
        let mut entry_states = Vec::new();
        let mut roots = roots.into_iter();
        for (_, aggregate) in aggregates {
            if let Some(kind) = aggregate.kind().map_err(|_| "unsupported aggregate")? {
                entry_states.push(Some(specs.len()));
                specs.push(Spec {
                    kind,
                    root: Some(roots.next().expect("argument root")),
                });
            } else {
                entry_states.push(None);
            }
        }
        let row_state = predicate.as_ref().map(|_| {
            let index = specs.len();
            specs.push(Spec {
                kind: ReductionKind::Count,
                root: None,
            });
            index
        });
        if specs.is_empty() {
            return Err("metadata-only COUNT needs no fused kernel");
        }
        let layout = BindingLayout::new(&union);
        batch
            .validate(&executor.client, &layout, parameters)
            .map_err(|_| "fused input layout unsupported")?;
        executor
            .preflight_expression(&layout, u64::from(batch.row_count()))
            .map_err(|_| "fused expression limits exceeded")?;
        let words: usize = specs.iter().map(|s| s.kind.state_words()).sum();
        let inputs = layout
            .buffers()
            .iter()
            .filter(|s| s.access() == BufferAccess::ReadOnly)
            .count();
        let hw = &executor.client.properties().hardware;
        if inputs + words + usize::from(predicate_error) + usize::from(argument_error) + 1
            > hw.max_bindings as usize
        {
            return Err("fused buffer binding limit exceeded");
        }
        if specs.iter().any(|s| s.kind.needs_high()) && hw.max_bindings < 8 {
            return Err("wide aggregate merge binding limit exceeded");
        }
        let reduction = ReductionPlan::new(
            &executor.client,
            batch.row_count(),
            executor.cube_size(),
            specs.len(),
            words as u64,
            words,
        )
        .map_err(|_| "fused shared memory or reduction limits exceeded")?;
        Ok(Self {
            union,
            predicate,
            arguments,
            specs,
            entry_states,
            row_state,
            predicate_error,
            argument_error,
            reduction,
        })
    }
    pub(crate) fn launches(&self) -> (usize, usize) {
        let levels = self.reduction.levels.len();
        (
            usize::from(levels != 0),
            levels.saturating_sub(1) * self.specs.len(),
        )
    }
    pub(crate) fn submit(
        self,
        executor: &Executor,
        batch: &DeviceBatch,
        aggregates: &[(&str, Aggregate)],
        parameters: &Parameters,
    ) -> Result<DeviceAggregates, ExecutionError> {
        let layout = BindingLayout::new(&self.union);
        let mut retained = Vec::new();
        let mut buffers = Vec::new();
        if batch.row_count() != 0 {
            for slot in layout
                .buffers()
                .iter()
                .filter(|s| s.access() == BufferAccess::ReadOnly)
            {
                buffers.push(match slot.source() {
                    BufferSource::Column(_) => batch.value_handle(slot.name()),
                    BufferSource::ColumnValidity(_) => batch.validity_handle(slot.name()),
                    _ => unreachable!("read-only input"),
                });
            }
        }
        let input_count = buffers.len();
        let mut states = Vec::new();
        if let Some(&(_, blocks)) = self.reduction.levels.first() {
            for spec in &self.specs {
                let low = executor.client.empty(blocks as usize * 4);
                let count = executor.client.empty(blocks as usize * 4);
                let high = spec
                    .kind
                    .needs_high()
                    .then(|| executor.client.empty(blocks as usize * 4));
                buffers.extend([low.clone(), count.clone()]);
                buffers.extend(high.iter().cloned());
                states.push(AggregateState { low, high, count });
            }
        }
        let pred_error = (self.predicate_error && batch.row_count() != 0)
            .then(|| executor.client.empty(batch.row_count() as usize * 4));
        let arg_error = (self.argument_error && batch.row_count() != 0)
            .then(|| executor.client.empty(batch.row_count() as usize * 4));
        buffers.extend(pred_error.iter().cloned());
        buffers.extend(arg_error.iter().cloned());
        retained.extend(buffers.iter().cloned());
        if !self.reduction.levels.is_empty() {
            let kernel = FusedKernel {
                union: self.union,
                predicate: self.predicate,
                arguments: self.arguments,
                specs: self.specs.clone(),
                predicate_error: self.predicate_error,
                argument_error: self.argument_error,
                client: executor.client.clone(),
                settings: executor.settings(),
                definitions: executor.define_count.clone(),
            };
            let mut guard = LauncherGuard(Some(KernelLauncher::new(executor.settings())));
            let launcher = guard.0.as_mut().expect("fresh launcher");
            for (i, buffer) in buffers.into_iter().enumerate() {
                launcher.declare_io(if i < input_count {
                    BufferIOAttr::ReadOnly
                } else {
                    BufferIOAttr::WriteOnly
                });
                // SAFETY: validated immutable inputs and exact private U32 state/error planes.
                let len = buffer.size_in_used() as usize / 4;
                launcher.register_buffer(unsafe { BufferArg::from_raw_parts(buffer, len) }, 4);
            }
            for scalar in
                layout.bind_scalars(layout.extent(u64::from(batch.row_count()))?, parameters)?
            {
                match scalar {
                    ScalarArgument::U32(v) => launcher.register_scalar(v),
                    ScalarArgument::F32(v) => launcher.register_scalar(v),
                    ScalarArgument::I32(v) => launcher.register_scalar(v),
                }
            }
            guard.0.take().expect("registered launcher").launch(
                CubeCount::Static(self.reduction.levels[0].1, 1, 1),
                kernel,
                &executor.client,
            );
            executor.counters.launches.fetch_add(1, Ordering::Relaxed);
            for &(rows, blocks) in self.reduction.levels.iter().skip(1) {
                for (state, spec) in states.iter_mut().zip(&self.specs) {
                    let low = executor.client.empty(blocks as usize * 4);
                    let count = executor.client.empty(blocks as usize * 4);
                    let high = spec
                        .kind
                        .needs_high()
                        .then(|| executor.client.empty(blocks as usize * 4));
                    let mut buffers = vec![
                        state.low.clone(),
                        state.count.clone(),
                        state.low.clone(),
                        low.clone(),
                        count.clone(),
                    ];
                    if let Some(high) = &high {
                        buffers.extend([state.high.clone().expect("wide input"), high.clone()]);
                    }
                    ReductionKernel::new(
                        spec.kind,
                        true,
                        true,
                        false,
                        executor.client.clone(),
                        executor.settings(),
                        executor.define_count.clone(),
                    )
                    .launch(rows, &buffers);
                    executor.counters.launches.fetch_add(1, Ordering::Relaxed);
                    retained.extend([low.clone(), count.clone()]);
                    retained.extend(high.iter().cloned());
                    *state = AggregateState { low, high, count };
                }
            }
        }
        let row_count_state = self
            .row_state
            .and_then(|i| states.get(i).map(|s| s.count.clone()));
        let entries = aggregates
            .iter()
            .zip(self.entry_states)
            .map(|((name, a), index)| Entry {
                name: (*name).into(),
                kind: a.kind().expect("preflight"),
                state: index.and_then(|i| {
                    states.get(i).map(|s| AggregateState {
                        low: s.low.clone(),
                        high: s.high.clone(),
                        count: s.count.clone(),
                    })
                }),
            })
            .collect();
        let checked = |error| SubmittedExpression {
            rows: batch.row_count(),
            values: None,
            validity: None,
            error,
            retained: retained.clone(),
        };
        let dependencies = arg_error
            .map(|h| vec![Dependency::Dense(Arc::new(checked(Some(h))))])
            .unwrap_or_default();
        let predicate_checks = pred_error
            .map(|h| vec![checked(Some(h))])
            .unwrap_or_default();
        let mut stats = self.reduction.stats;
        stats.reduction_launches = usize::from(!self.reduction.levels.is_empty())
            + self.reduction.levels.len().saturating_sub(1) * self.specs.len();
        Ok(DeviceAggregates {
            client: executor.client.clone(),
            counters: executor.counters.clone(),
            rows: batch.row_count(),
            selection: None,
            entries: Arc::new(entries),
            dependencies: Arc::new(dependencies),
            retained: Arc::new(retained),
            stats,
            predicate_checks: Arc::new(predicate_checks),
            row_count_state,
        })
    }
}
struct FusedKernel {
    union: KernelProgram,
    predicate: Option<KernelProgram>,
    arguments: Option<KernelProgram>,
    specs: Vec<Spec>,
    predicate_error: bool,
    argument_error: bool,
    client: Client,
    settings: KernelSettings,
    definitions: Arc<AtomicUsize>,
}
impl KernelMetadata for FusedKernel {
    fn id(&self) -> KernelId {
        let identity = |p: &KernelProgram| {
            ExpressionKernel::new(
                p.clone(),
                self.client.clone(),
                self.settings.clone(),
                self.definitions.clone(),
            )
            .identity()
        };
        KernelId::new::<Self>()
            .info((
                1u32,
                "fused-active-pairwise-sum-v1",
                identity(&self.union),
                self.predicate.as_ref().map(identity),
                self.arguments.as_ref().map(identity),
                self.specs.clone(),
            ))
            .cube_dim(self.settings.cube_dim)
            .address_type(self.settings.address_type)
            .mode(self.settings.execution_mode)
    }
    fn address_type(&self) -> ElemType {
        self.settings.address_type.unsigned_type()
    }
}
impl CubeKernel for FusedKernel {
    fn define(&self) -> KernelDefinition {
        self.definitions.fetch_add(1, Ordering::Relaxed);
        let layout = BindingLayout::new(&self.union);
        let mut declaration = layout
            .declare_inputs(
                &self.client,
                self.settings
                    .clone()
                    .kernel_name(self.id().entrypoint_name("cuberel_fused")),
            )
            .expect("trusted fused layout");
        let outputs: Vec<_> = self
            .specs
            .iter()
            .map(|spec| {
                let low = declaration.append_buffer(UIntKind::U32.into());
                let count = declaration.append_buffer(UIntKind::U32.into());
                let high = spec
                    .kind
                    .needs_high()
                    .then(|| declaration.append_buffer(UIntKind::U32.into()));
                (low, count, high)
            })
            .collect();
        let pred_error = self
            .predicate_error
            .then(|| declaration.append_buffer(UIntKind::U32.into()));
        let arg_error = self
            .argument_error
            .then(|| declaration.append_buffer(UIntKind::U32.into()));
        let s = declaration.scope();
        s.state_mut().modes.fp_math_mode = Default::default();
        let ty = ElemType::UInt(UIntKind::U32).to_type(s.ctx());
        let gid = builtin(s, Builtin::AbsolutePos);
        let lid = builtin(s, Builtin::UnitPos);
        let block = builtin(s, Builtin::CubePos);
        let zero = uint(s, 0);
        let index_ty = ElemType::Index.to_type(s.ctx());
        let source_index = s.register_with_result(&CastOp::new(s.ctx_mut(), index_ty, gid));
        let locals: Vec<_> = self
            .specs
            .iter()
            .map(|spec| {
                let low = s.create_local_mut(ty, None);
                let count = s.create_local_mut(ty, None);
                let high = spec.kind.needs_high().then(|| s.create_local_mut(ty, None));
                for v in [Some(low), Some(count), high].into_iter().flatten() {
                    s.register(&StoreOp::new(s.ctx_mut(), v, zero));
                }
                (low, count, high)
            })
            .collect();
        let in_range = s.register_with_result(&ULessThanOp::new(
            s.ctx_mut(),
            gid,
            declaration.row_count_raw(),
        ));
        branch(s, in_range, |body| {
            if let Some(error) = arg_error {
                store(body, error, gid, zero);
            }
            let evaluate = |active: &cubecl::prelude::Scope| {
                let args = self
                    .arguments
                    .as_ref()
                    .map(|p| emit_scalar(p, &declaration, active, source_index));
                if let Some(error) = arg_error {
                    let flag = active.register_with_result(&CastOp::new(
                        active.ctx_mut(),
                        ty,
                        args.as_ref().expect("arguments").overflow,
                    ));
                    store(active, error, gid, flag);
                }
                for (spec, &(low, count, high)) in self.specs.iter().zip(&locals) {
                    let (value, valid) = if let Some(root) = spec.root {
                        let args = args.as_ref().expect("root arguments");
                        let valid = active.register_with_result(&CastOp::new(
                            active.ctx_mut(),
                            ty,
                            args.validities[root.index()],
                        ));
                        let value = if spec.kind == ReductionKind::Count {
                            uint(active, 0)
                        } else {
                            active.register_with_result(&ReinterpretCastOp::new(
                                active.ctx_mut(),
                                ty,
                                args.values[root.index()],
                            ))
                        };
                        (value, valid)
                    } else {
                        (uint(active, 0), uint(active, 1))
                    };
                    let value = if matches!(
                        spec.kind,
                        ReductionKind::MinF32 | ReductionKind::MaxF32 | ReductionKind::SumF32
                    ) {
                        canonical_float(active, value)
                    } else {
                        value
                    };
                    active.register(&StoreOp::new(active.ctx_mut(), low, value));
                    active.register(&StoreOp::new(active.ctx_mut(), count, valid));
                    if let Some(high) = high {
                        let sign = uint(active, 0x7fff_ffff);
                        let ones = uint(active, u32::MAX);
                        let negative = active.register_with_result(&UGreaterThanOp::new(
                            active.ctx_mut(),
                            value,
                            sign,
                        ));
                        let bits = active.register_with_result(&SelectOp::new(
                            active.ctx_mut(),
                            negative,
                            ones,
                            zero,
                        ));
                        active.register(&StoreOp::new(active.ctx_mut(), high, bits));
                    }
                }
            };
            if let Some(predicate) = &self.predicate {
                let pred = emit_scalar(predicate, &declaration, body, source_index);
                if let Some(error) = pred_error {
                    let flag =
                        body.register_with_result(&CastOp::new(body.ctx_mut(), ty, pred.overflow));
                    store(body, error, gid, flag);
                }
                let root = predicate.root().index();
                let active = body.register_with_result(&BoolAndOp::new(
                    body.ctx_mut(),
                    pred.values[root],
                    pred.validities[root],
                ));
                branch(body, active, evaluate);
            } else {
                evaluate(body);
            }
        });
        let array = ArrayType::get(s.ctx(), ty, self.settings.cube_dim.x as usize);
        let shared: Vec<_> = locals
            .iter()
            .map(|&(low, count, high)| {
                let a = s.create_shared(array, None);
                let c = s.create_shared(array, None);
                let h = high.map(|_| s.create_shared(array, None));
                for (local, shared) in [(Some(low), Some(a)), (Some(count), Some(c)), (high, h)] {
                    if let (Some(local), Some(shared)) = (local, shared) {
                        let v = s.register_with_result(&LoadOp::new(s.ctx_mut(), local));
                        store(s, shared, lid, v);
                    }
                }
                (a, c, h)
            })
            .collect();
        barrier(s);
        let mut distance = 1;
        while distance < self.settings.cube_dim.x {
            let width = uint(s, distance * 2);
            let d = uint(s, distance);
            let tile = uint(s, self.settings.cube_dim.x);
            let rem = s.register_with_result(&URemOp::new(s.ctx_mut(), lid, width));
            let left = s.register_with_result(&IEqualOp::new(s.ctx_mut(), rem, zero));
            let right = s.register_with_result(&IAddOp::new(s.ctx_mut(), lid, d));
            let exists = s.register_with_result(&ULessThanOp::new(s.ctx_mut(), right, tile));
            let active = s.register_with_result(&BoolAndOp::new(s.ctx_mut(), left, exists));
            branch(s, active, |body| {
                for (spec, &(v, c, h)) in self.specs.iter().zip(&shared) {
                    let a = load(body, v, lid);
                    let b = load(body, v, right);
                    let ca = load(body, c, lid);
                    let cb = load(body, c, right);
                    let value = if let Some(h) = h {
                        let ah = load(body, h, lid);
                        let bh = load(body, h, right);
                        let (value, high) = combine_i64(body, (a, ah, ca), (b, bh, cb));
                        store(body, h, lid, high);
                        value
                    } else {
                        combine(body, spec.kind, a, b, ca, cb)
                    };
                    let count = body.register_with_result(&IAddOp::new(body.ctx_mut(), ca, cb));
                    store(body, v, lid, value);
                    store(body, c, lid, count);
                }
            });
            barrier(s);
            distance *= 2;
        }
        let first = s.register_with_result(&IEqualOp::new(s.ctx_mut(), lid, zero));
        branch(s, first, |body| {
            for (&(v, c, h), &(ov, oc, oh)) in shared.iter().zip(&outputs) {
                let value = load(body, v, zero);
                let count = load(body, c, zero);
                store(body, ov, block, value);
                store(body, oc, block, count);
                if let (Some(h), Some(oh)) = (h, oh) {
                    let high = load(body, h, zero);
                    store(body, oh, block, high);
                }
            }
        });
        s.register(&ReturnOp::new(s.ctx_mut()));
        verify_operation(s.state().module.get_operation(), s.ctx())
            .expect("invalid fused reduction IR");
        declaration.build()
    }
}

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod tests {
    use super::{FusedKernel, FusedPlan};
    use crate::{
        Aggregate, Batch, ColumnData, DataType, ExecutionError, Executor, Parameters, Schema, col,
        prepare,
    };
    use cubecl::{Device, prelude::KernelMetadata};
    fn check(device: Device) {
        let e = Executor::new(device.client()).unwrap();
        let schema = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
        let x = prepare(&col("x"), &schema).unwrap();
        let batch = e
            .upload(
                &Batch::new(2)
                    .with_column("x", ColumnData::I32(&[-1, -1]))
                    .unwrap(),
            )
            .unwrap();
        let args = [("sum", Aggregate::Sum(x.clone()))];
        let plan = FusedPlan::prepare(&e, &batch, None, &args, &Parameters::new()).unwrap();
        let id = |plan: &FusedPlan| {
            FusedKernel {
                union: plan.union.clone(),
                predicate: plan.predicate.clone(),
                arguments: plan.arguments.clone(),
                specs: plan.specs.clone(),
                predicate_error: plan.predicate_error,
                argument_error: plan.argument_error,
                client: e.client.clone(),
                settings: e.settings(),
                definitions: e.define_count.clone(),
            }
            .id()
        };
        let sum_id = id(&plan);
        let renamed = FusedPlan::prepare(
            &e,
            &batch,
            None,
            &[("renamed", Aggregate::Sum(x.clone()))],
            &Parameters::new(),
        )
        .unwrap();
        assert_eq!(sum_id, id(&renamed));
        let count = FusedPlan::prepare(
            &e,
            &batch,
            None,
            &[("count", Aggregate::Count(x))],
            &Parameters::new(),
        )
        .unwrap();
        assert_ne!(sum_id, id(&count));
        let out = plan.submit(&e, &batch, &args, &Parameters::new()).unwrap();
        out.complete().unwrap();
        let high = out.entries[0]
            .state
            .as_ref()
            .unwrap()
            .high
            .as_ref()
            .unwrap()
            .clone();
        crate::selection::completion_tests::reject_output(&e.client, e.settings(), high);
        let before = e.stats();
        assert!(matches!(out.materialize(), Err(ExecutionError::Backend(_))));
        assert_eq!(e.stats(), before);
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_fused_identity_and_high_word_dependency() {
        check(Device::cpu().unwrap());
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_fused_identity_and_high_word_dependency() {
        check(Device::metal(Default::default()).unwrap());
    }
}
