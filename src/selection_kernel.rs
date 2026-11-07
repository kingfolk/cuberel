//! Runtime IR templates for stable U32 selection. No frontend #[cube] expansion.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use cubecl::{
    ir::{
        Builtin, ConstantValue, OpInserter, UIntKind,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::{IEqualOp, UGreaterThanOrEqualOp, ULessThanOp},
            general::{CastOp, ReadBuiltinOp, ReadScalarOp, SelectOp},
            math::{IAddOp, ISubOp},
            memory::{IndexOp, LoadOp, StoreOp},
            synchronization::{SyncOp, SyncScope, SyncScopeAttr},
        },
        pliron::{
            builtin::{attributes::IntegerAttr, ops::ConstantOp},
            op::Op,
            operation::verify_operation,
            value::Value,
        },
        types::ArrayType,
    },
    prelude::{
        BufferArg, BufferIOAttr, Client, CubeCount, CubeKernel, ElemType, KernelBuilder,
        KernelDefinition, KernelId, KernelLauncher, KernelMetadata, KernelSettings, Scope,
    },
    server::Handle,
};

use crate::cubecl_abi::LauncherGuard;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Stage {
    Scan,
    AddOffsets,
    Scatter,
}

impl Stage {
    // Shared declaration/launch resource order. Scalar 0 is logical row count.
    fn access(self) -> &'static [BufferIOAttr] {
        use BufferIOAttr::{ReadOnly, ReadWrite, WriteOnly};
        match self {
            Self::Scan => &[ReadOnly, WriteOnly, WriteOnly],
            Self::AddOffsets => &[ReadOnly, ReadWrite],
            Self::Scatter => &[ReadOnly, ReadOnly, ReadOnly, WriteOnly],
        }
    }
}

pub(crate) struct SelectionKernel {
    stage: Stage,
    client: Client,
    settings: KernelSettings,
    define_count: Arc<AtomicUsize>,
}

impl SelectionKernel {
    pub(crate) fn new(
        stage: Stage,
        client: Client,
        settings: KernelSettings,
        define_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            stage,
            client,
            settings,
            define_count,
        }
    }

    pub(crate) fn launch(self, rows: u32, buffers: &[Handle]) {
        assert_eq!(buffers.len(), self.stage.access().len());
        let tile = self.settings.cube_dim.x;
        let mut guard = LauncherGuard(Some(KernelLauncher::new(self.settings.clone())));
        let launcher = guard.0.as_mut().expect("fresh launcher");
        for (handle, &access) in buffers.iter().zip(self.stage.access()) {
            assert_eq!(handle.service, self.client.service_id());
            launcher.declare_io(access);
            // SAFETY: private stage layouts use scalar U32 arrays; the planner
            // allocates exact initialized input / fresh output ranges. Only the
            // scatter output's count prefix is initialized and ever read.
            let arg = unsafe {
                BufferArg::from_raw_parts(handle.clone(), handle.size_in_used() as usize / 4)
            };
            launcher.register_buffer(arg, 4);
        }
        launcher.register_scalar(rows);
        let client = self.client.clone();
        guard.0.take().expect("registered launcher").launch(
            CubeCount::Static(rows.div_ceil(tile), 1, 1),
            self,
            &client,
        );
    }
}

impl KernelMetadata for SelectionKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((
                1u32,
                self.stage,
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

impl CubeKernel for SelectionKernel {
    fn define(&self) -> KernelDefinition {
        self.define_count.fetch_add(1, Ordering::Relaxed);
        let settings = self
            .settings
            .clone()
            .kernel_name(self.id().entrypoint_name("cuberel_select"));
        let mut builder = KernelBuilder::new(settings);
        builder.runtime_properties(self.client.target_properties().clone());
        builder.device_properties(self.client.properties());
        let ty = ElemType::UInt(UIntKind::U32).to_type(builder.scope.ctx());
        let buffers: Vec<_> = self
            .stage
            .access()
            .iter()
            .map(|_| builder.buffer(ty))
            .collect();
        let ordinal = builder.scalar(UIntKind::U32.into());
        let s = &builder.scope;
        let rows = s.register_with_result(&ReadScalarOp::new(s.ctx_mut(), ty, ordinal));
        let gid = builtin(s, Builtin::AbsolutePos);
        let lid = builtin(s, Builtin::UnitPos);
        let block = builtin(s, Builtin::CubePos);
        let zero = uint(s, 0);
        let in_range = s.register_with_result(&ULessThanOp::new(s.ctx_mut(), gid, rows));
        match self.stage {
            Stage::Scan => {
                // Initialize every lane, including out-of-range tail lanes.
                let local = s.create_local_mut(ty, None);
                s.register(&StoreOp::new(s.ctx_mut(), local, zero));
                branch(s, in_range, |body| {
                    let value = load(body, buffers[0], gid);
                    body.register(&StoreOp::new(body.ctx_mut(), local, value));
                });
                let value = s.register_with_result(&LoadOp::new(s.ctx_mut(), local));
                let array = ArrayType::get(s.ctx(), ty, self.settings.cube_dim.x as usize);
                let shared = s.create_shared(array, None);
                store(s, shared, lid, value);
                barrier(s);
                // Hillis-Steele inclusive scan. Every lane participates in
                // both barriers; no read/write races between rounds.
                let mut distance = 1;
                while distance < self.settings.cube_dim.x {
                    let d = uint(s, distance);
                    let active =
                        s.register_with_result(&UGreaterThanOrEqualOp::new(s.ctx_mut(), lid, d));
                    let previous = s.register_with_result(&ISubOp::new(s.ctx_mut(), lid, d));
                    let previous =
                        s.register_with_result(&SelectOp::new(s.ctx_mut(), active, previous, zero));
                    let left = load(s, shared, previous);
                    let left =
                        s.register_with_result(&SelectOp::new(s.ctx_mut(), active, left, zero));
                    let current = load(s, shared, lid);
                    let sum = s.register_with_result(&IAddOp::new(s.ctx_mut(), current, left));
                    barrier(s); // finish reads before any lane overwrites shared
                    store(s, shared, lid, sum);
                    barrier(s); // make writes visible to the next round
                    distance *= 2;
                }
                let inclusive = load(s, shared, lid);
                let rank = s.register_with_result(&ISubOp::new(s.ctx_mut(), inclusive, value));
                branch(s, in_range, |body| store(body, buffers[1], gid, rank));
                let last = uint(s, self.settings.cube_dim.x - 1);
                let last = s.register_with_result(&IEqualOp::new(s.ctx_mut(), lid, last));
                branch(s, last, |body| store(body, buffers[2], block, inclusive));
            }
            Stage::AddOffsets => branch(s, in_range, |body| {
                let parent = load(body, buffers[0], block);
                let rank = load(body, buffers[1], gid);
                let rank = body.register_with_result(&IAddOp::new(body.ctx_mut(), rank, parent));
                store(body, buffers[1], gid, rank);
            }),
            Stage::Scatter => branch(s, in_range, |body| {
                let keep = load(body, buffers[0], gid);
                let one = uint(body, 1);
                let keep = body.register_with_result(&IEqualOp::new(body.ctx_mut(), keep, one));
                branch(body, keep, |selected| {
                    let rank = load(selected, buffers[1], gid);
                    let count = load(selected, buffers[2], zero);
                    let valid = selected.register_with_result(&ULessThanOp::new(
                        selected.ctx_mut(),
                        rank,
                        count,
                    ));
                    branch(selected, valid, |out| store(out, buffers[3], rank, gid));
                });
            }),
        }
        s.register(&ReturnOp::new(s.ctx_mut()));
        verify_operation(s.state().module.get_operation(), s.ctx()).expect("invalid selection IR");
        builder.build()
    }
}

fn branch(scope: &Scope, condition: Value, f: impl FnOnce(&Scope)) {
    let op = IfOp::new(scope.ctx_mut(), condition);
    let body = scope.branch_child(OpInserter::new_at_block_end(op.then_block(scope.ctx())));
    f(&body);
    body.terminate_yield();
    let other = scope.branch_child(OpInserter::new_at_block_end(op.else_block(scope.ctx())));
    other.terminate_yield();
    scope.register(&op);
    scope.update_flags_after_branch(&[body, other]);
}

fn barrier(scope: &Scope) {
    scope.register(&SyncOp::new(
        scope.ctx_mut(),
        SyncScopeAttr(SyncScope::Cube),
    ));
}

fn builtin(scope: &Scope, builtin: Builtin) -> Value {
    let index = ElemType::Index.to_type(scope.ctx());
    let raw = scope.register_with_result(&ReadBuiltinOp::new(scope.ctx_mut(), index, builtin));
    let ty = ElemType::UInt(UIntKind::U32).to_type(scope.ctx());
    scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, raw))
}

fn ptr(scope: &Scope, buffer: Value, index: Value) -> Value {
    let ty = ElemType::Index.to_type(scope.ctx());
    let index = scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, index));
    scope.register_with_result(&IndexOp::maybe_checked(
        scope.ctx_mut(),
        buffer,
        index,
        true,
    ))
}

fn load(scope: &Scope, buffer: Value, index: Value) -> Value {
    let ptr = ptr(scope, buffer, index);
    scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))
}

fn store(scope: &Scope, buffer: Value, index: Value, value: Value) {
    let ptr = ptr(scope, buffer, index);
    scope.register(&StoreOp::new(scope.ctx_mut(), ptr, value));
}

fn uint(scope: &Scope, value: u32) -> Value {
    let attr =
        ConstantValue::UInt(u64::from(value)).as_attribute(scope.ctx(), UIntKind::U32.into());
    scope.register_with_result(&ConstantOp::new(
        scope.ctx_mut(),
        Box::new(
            attr.downcast_ref::<IntegerAttr>()
                .expect("U32 attribute")
                .clone(),
        ),
    ))
}
