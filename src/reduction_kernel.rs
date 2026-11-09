//! Fixed, portable block reduction over (value bits, valid count) states.
use crate::{
    cubecl_abi::LauncherGuard,
    ir_support::{barrier, branch, builtin, load, store, uint},
};
use cubecl::{
    ir::{
        Builtin, FloatKind, IntKind, UIntKind,
        attributes::FloatAttr,
        dialect::{
            bitwise::{BitwiseAndOp, BitwiseOrOp, BitwiseXorOp},
            branch::ReturnOp,
            cmp::{IEqualOp, SGreaterThanOp, SLessThanOp, UGreaterThanOp, ULessThanOp},
            general::{BoolAndOp, BoolOrOp, CastOp, ReadScalarOp, ReinterpretCastOp, SelectOp},
            math::{FmaOp, IAddOp, URemOp},
            memory::{LoadOp, StoreOp},
        },
        pliron::{builtin::ops::ConstantOp, op::Op, operation::verify_operation, value::Value},
        types::ArrayType,
    },
    prelude::{
        BufferArg, BufferIOAttr, Client, CubeCount, CubeKernel, ElemType, KernelBuilder,
        KernelDefinition, KernelId, KernelLauncher, KernelMetadata, KernelSettings, Scope,
    },
    server::Handle,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ReductionKind {
    Count,
    MinI32,
    MaxI32,
    MinF32,
    MaxF32,
    SumF32,
    SumI64,
}

impl ReductionKind {
    pub(crate) fn needs_high(self) -> bool {
        self == Self::SumI64
    }
    pub(crate) fn state_words(self) -> usize {
        if self.needs_high() { 3 } else { 2 }
    }
}

pub(crate) struct ReductionKernel {
    kind: ReductionKind,
    merge: bool,
    nullable: bool,
    selected: bool,
    client: Client,
    settings: KernelSettings,
    definitions: Arc<AtomicUsize>,
}
impl ReductionKernel {
    pub(crate) fn new(
        kind: ReductionKind,
        merge: bool,
        nullable: bool,
        selected: bool,
        client: Client,
        settings: KernelSettings,
        definitions: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            kind,
            merge,
            nullable,
            selected,
            client,
            settings,
            definitions,
        }
    }
    // Private ABI: values, validity/partial counts, selected count/dummy input,
    // output values, output counts; wide SUM additionally appends input/output
    // high words. One U32 scalar carries logical input capacity.
    pub(crate) fn launch(self, rows: u32, buffers: &[Handle]) {
        assert_eq!(buffers.len(), if self.kind.needs_high() { 7 } else { 5 });
        let mut guard = LauncherGuard(Some(KernelLauncher::new(self.settings.clone())));
        let launcher = guard.0.as_mut().expect("fresh launcher");
        for (i, handle) in buffers.iter().enumerate() {
            assert_eq!(handle.service, self.client.service_id());
            launcher.declare_io(if i < 3 || i == 5 {
                BufferIOAttr::ReadOnly
            } else {
                BufferIOAttr::WriteOnly
            });
            // SAFETY: private planner owns exact U32-word planes. First-stage
            // selected reads are count-guarded; merge inputs are fully initialized.
            // Value words preserve F32/I32 bits, not a host numeric conversion.
            launcher.register_buffer(
                unsafe {
                    BufferArg::from_raw_parts(handle.clone(), handle.size_in_used() as usize / 4)
                },
                4,
            );
        }
        launcher.register_scalar(rows);
        let client = self.client.clone();
        let blocks = rows.div_ceil(self.settings.cube_dim.x);
        guard.0.take().expect("registered launcher").launch(
            CubeCount::Static(blocks, 1, 1),
            self,
            &client,
        );
    }
}
impl KernelMetadata for ReductionKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((
                2u32,
                "ieee-bits-minmax;sum-pairwise-empty-identity-fma-one-canonical-nan-v1",
                self.kind,
                self.merge,
                self.nullable,
                self.selected,
                self.settings.clone(),
                self.client.name().to_string(),
                self.client.properties().checksum(),
            ))
            .cube_dim(self.settings.cube_dim)
            .address_type(self.settings.address_type)
            .mode(self.settings.execution_mode)
    }
    fn address_type(&self) -> ElemType {
        self.settings.address_type.unsigned_type()
    }
}
impl CubeKernel for ReductionKernel {
    fn define(&self) -> KernelDefinition {
        self.definitions.fetch_add(1, Ordering::Relaxed);
        let mut builder = KernelBuilder::new(
            self.settings
                .clone()
                .kernel_name(self.id().entrypoint_name("cuberel_reduce")),
        );
        builder.runtime_properties(self.client.target_properties().clone());
        builder.device_properties(self.client.properties());
        let ty = ElemType::UInt(UIntKind::U32).to_type(builder.scope.ctx());
        let buffers: Vec<_> = (0..if self.kind.needs_high() { 7 } else { 5 })
            .map(|_| builder.buffer(ty))
            .collect();
        let ordinal = builder.scalar(UIntKind::U32.into());
        let s = &builder.scope;
        s.state_mut().modes.fp_math_mode = Default::default();
        let rows = s.register_with_result(&ReadScalarOp::new(s.ctx_mut(), ty, ordinal));
        let gid = builtin(s, Builtin::AbsolutePos);
        let lid = builtin(s, Builtin::UnitPos);
        let block = builtin(s, Builtin::CubePos);
        let zero = uint(s, 0);
        let local_value = s.create_local_mut(ty, None);
        let local_count = s.create_local_mut(ty, None);
        let local_high = self.kind.needs_high().then(|| s.create_local_mut(ty, None));
        if let Some(high) = local_high {
            s.register(&StoreOp::new(s.ctx_mut(), high, zero));
        }
        s.register(&StoreOp::new(s.ctx_mut(), local_value, zero));
        s.register(&StoreOp::new(s.ctx_mut(), local_count, zero));
        let in_range = s.register_with_result(&ULessThanOp::new(s.ctx_mut(), gid, rows));
        branch(s, in_range, |in_range| {
            let read = |active: &Scope| {
                let count = if self.merge || self.nullable {
                    load(active, buffers[1], gid)
                } else {
                    uint(active, 1)
                };
                let value = if self.kind == ReductionKind::Count {
                    uint(active, 0)
                } else {
                    load(active, buffers[0], gid)
                };
                let value = if matches!(
                    self.kind,
                    ReductionKind::MinF32 | ReductionKind::MaxF32 | ReductionKind::SumF32
                ) {
                    canonical_float(active, value)
                } else {
                    value
                };
                if let Some(high) = local_high {
                    let bits = if self.merge {
                        load(active, buffers[5], gid)
                    } else {
                        let sign = uint(active, 0x7fff_ffff);
                        let negative = active.register_with_result(&UGreaterThanOp::new(
                            active.ctx_mut(),
                            value,
                            sign,
                        ));
                        let ones = uint(active, u32::MAX);
                        active.register_with_result(&SelectOp::new(
                            active.ctx_mut(),
                            negative,
                            ones,
                            zero,
                        ))
                    };
                    active.register(&StoreOp::new(active.ctx_mut(), high, bits));
                }
                active.register(&StoreOp::new(active.ctx_mut(), local_count, count));
                active.register(&StoreOp::new(active.ctx_mut(), local_value, value));
            };
            if self.selected {
                let count = load(in_range, buffers[2], zero);
                let active = in_range.register_with_result(&ULessThanOp::new(
                    in_range.ctx_mut(),
                    gid,
                    count,
                ));
                branch(in_range, active, read);
            } else {
                read(in_range);
            }
        });
        let array = ArrayType::get(s.ctx(), ty, self.settings.cube_dim.x as usize);
        let shared_values = s.create_shared(array, None);
        let shared_counts = s.create_shared(array, None);
        let shared_high = self.kind.needs_high().then(|| s.create_shared(array, None));
        if let Some(shared_high) = shared_high {
            let high =
                s.register_with_result(&LoadOp::new(s.ctx_mut(), local_high.expect("wide local")));
            store(s, shared_high, lid, high);
        }
        let value = s.register_with_result(&LoadOp::new(s.ctx_mut(), local_value));
        let count = s.register_with_result(&LoadOp::new(s.ctx_mut(), local_count));
        store(s, shared_values, lid, value);
        store(s, shared_counts, lid, count);
        barrier(s);
        let mut distance = 1;
        while distance < self.settings.cube_dim.x {
            let d = uint(s, distance);
            let width = uint(s, distance * 2);
            let rem = s.register_with_result(&URemOp::new(s.ctx_mut(), lid, width));
            let left = s.register_with_result(&IEqualOp::new(s.ctx_mut(), rem, zero));
            let right = s.register_with_result(&IAddOp::new(s.ctx_mut(), lid, d));
            let tile = uint(s, self.settings.cube_dim.x);
            let exists = s.register_with_result(&ULessThanOp::new(s.ctx_mut(), right, tile));
            let active = s.register_with_result(&BoolAndOp::new(s.ctx_mut(), left, exists));
            branch(s, active, |body| {
                let a = load(body, shared_values, lid);
                let b = load(body, shared_values, right);
                let ca = load(body, shared_counts, lid);
                let cb = load(body, shared_counts, right);
                let value = if let Some(shared_high) = shared_high {
                    let ah = load(body, shared_high, lid);
                    let bh = load(body, shared_high, right);
                    let (value, high) = combine_i64(body, (a, ah, ca), (b, bh, cb));
                    store(body, shared_high, lid, high);
                    value
                } else {
                    combine(body, self.kind, a, b, ca, cb)
                };
                let count = body.register_with_result(&IAddOp::new(body.ctx_mut(), ca, cb));
                // Pairs are disjoint within each round: right lanes never write.
                store(body, shared_values, lid, value);
                store(body, shared_counts, lid, count);
            });
            barrier(s); // every lane, including inactive and tail lanes
            distance *= 2;
        }
        let first = s.register_with_result(&IEqualOp::new(s.ctx_mut(), lid, zero));
        branch(s, first, |body| {
            let value = load(body, shared_values, zero);
            let count = load(body, shared_counts, zero);
            store(body, buffers[3], block, value);
            store(body, buffers[4], block, count);
            if let Some(shared_high) = shared_high {
                let high = load(body, shared_high, zero);
                store(body, buffers[6], block, high);
            }
        });
        s.register(&ReturnOp::new(s.ctx_mut()));
        verify_operation(s.state().module.get_operation(), s.ctx()).expect("invalid reduction IR");
        builder.build()
    }
}
pub(crate) fn canonical_float(s: &Scope, bits: Value) -> Value {
    let abs = magnitude(s, bits);
    let inf = uint(s, 0x7f80_0000);
    let nan = s.register_with_result(&UGreaterThanOp::new(s.ctx_mut(), abs, inf));
    let canonical = uint(s, 0x7fc0_0000);
    s.register_with_result(&SelectOp::new(s.ctx_mut(), nan, canonical, bits))
}
fn magnitude(s: &Scope, bits: Value) -> Value {
    let mask = uint(s, 0x7fff_ffff);
    s.register_with_result(&BitwiseAndOp::new(s.ctx_mut(), bits, mask))
}
pub(crate) fn combine(
    s: &Scope,
    kind: ReductionKind,
    a: Value,
    b: Value,
    ca: Value,
    cb: Value,
) -> Value {
    let zero = uint(s, 0);
    if kind == ReductionKind::Count {
        return zero;
    }
    let value = match kind {
        ReductionKind::MinI32 | ReductionKind::MaxI32 => {
            let ty = ElemType::Int(IntKind::I32).to_type(s.ctx());
            let av = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), ty, a));
            let bv = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), ty, b));
            let choose_a = if kind == ReductionKind::MinI32 {
                s.register_with_result(&SLessThanOp::new(s.ctx_mut(), av, bv))
            } else {
                s.register_with_result(&SGreaterThanOp::new(s.ctx_mut(), av, bv))
            };
            s.register_with_result(&SelectOp::new(s.ctx_mut(), choose_a, a, b))
        }
        ReductionKind::MinF32 | ReductionKind::MaxF32 => {
            // IEEE bit ordering avoids backend denormal flushing in FP compares.
            // It naturally orders -0 before +0; all valid NaNs override below.
            let av = float_order_key(s, a);
            let bv = float_order_key(s, b);
            let choose_a = if kind == ReductionKind::MinF32 {
                s.register_with_result(&ULessThanOp::new(s.ctx_mut(), av, bv))
            } else {
                s.register_with_result(&UGreaterThanOp::new(s.ctx_mut(), av, bv))
            };
            let regular = s.register_with_result(&SelectOp::new(s.ctx_mut(), choose_a, a, b));
            let ma = magnitude(s, a);
            let mb = magnitude(s, b);
            let inf = uint(s, 0x7f80_0000);
            let na = s.register_with_result(&UGreaterThanOp::new(s.ctx_mut(), ma, inf));
            let nb = s.register_with_result(&UGreaterThanOp::new(s.ctx_mut(), mb, inf));
            let any_nan = s.register_with_result(&BoolOrOp::new(s.ctx_mut(), na, nb));
            let canonical = uint(s, 0x7fc0_0000);
            s.register_with_result(&SelectOp::new(s.ctx_mut(), any_nan, canonical, regular))
        }
        ReductionKind::SumF32 => {
            let ty = ElemType::Float(FloatKind::F32).to_type(s.ctx());
            let av = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), ty, a));
            let bv = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), ty, b));
            // fma(1,a,b) is a single rounded add. Avoid CubeCL's unconditional
            // FMul->FAdd contraction pattern, as in the expression emitter.
            let one = FloatAttr::from_f64(s.ctx(), ty, 1.0);
            let one = s.register_with_result(&ConstantOp::new(s.ctx_mut(), Box::new(one)));
            let sum = s.register_with_result(&FmaOp::new(s.ctx_mut(), one, av, bv));
            let bits_ty = ElemType::UInt(UIntKind::U32).to_type(s.ctx());
            let bits = s.register_with_result(&ReinterpretCastOp::new(s.ctx_mut(), bits_ty, sum));
            canonical_float(s, bits)
        }
        ReductionKind::Count | ReductionKind::SumI64 => unreachable!(),
    };
    let empty_a = s.register_with_result(&IEqualOp::new(s.ctx_mut(), ca, zero));
    let empty_b = s.register_with_result(&IEqualOp::new(s.ctx_mut(), cb, zero));
    let value = s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_b, a, value));
    s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_a, b, value))
}

fn float_order_key(s: &Scope, bits: Value) -> Value {
    let sign = uint(s, 0x8000_0000);
    let abs_mask = uint(s, 0x7fff_ffff);
    let negative = s.register_with_result(&UGreaterThanOp::new(s.ctx_mut(), bits, abs_mask));
    let all = uint(s, u32::MAX);
    let reversed = s.register_with_result(&BitwiseXorOp::new(s.ctx_mut(), bits, all));
    let positive = s.register_with_result(&BitwiseOrOp::new(s.ctx_mut(), bits, sign));
    s.register_with_result(&SelectOp::new(s.ctx_mut(), negative, reversed, positive))
}

// All word arithmetic is U32 wrapping. Under the batch byte bound, every
// mathematical partial sum fits i64; no native device I64 instructions needed.
pub(crate) fn combine_i64(
    s: &Scope,
    a: (Value, Value, Value),
    b: (Value, Value, Value),
) -> (Value, Value) {
    let (al, ah, ca) = a;
    let (bl, bh, cb) = b;
    let low = s.register_with_result(&IAddOp::new(s.ctx_mut(), al, bl));
    let carry = s.register_with_result(&ULessThanOp::new(s.ctx_mut(), low, al));
    let ty = ElemType::UInt(UIntKind::U32).to_type(s.ctx());
    let carry = s.register_with_result(&CastOp::new(s.ctx_mut(), ty, carry));
    let high = s.register_with_result(&IAddOp::new(s.ctx_mut(), ah, bh));
    let high = s.register_with_result(&IAddOp::new(s.ctx_mut(), high, carry));
    let zero = uint(s, 0);
    let empty_a = s.register_with_result(&IEqualOp::new(s.ctx_mut(), ca, zero));
    let empty_b = s.register_with_result(&IEqualOp::new(s.ctx_mut(), cb, zero));
    let low = s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_b, al, low));
    let low = s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_a, bl, low));
    let high = s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_b, ah, high));
    let high = s.register_with_result(&SelectOp::new(s.ctx_mut(), empty_a, bh, high));
    (low, high)
}
