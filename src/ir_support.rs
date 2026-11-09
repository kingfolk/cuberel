//! Small IR construction helpers shared by fixed selection/reduction templates.
use cubecl::{
    ir::{
        Builtin, ConstantValue, OpInserter, UIntKind,
        dialect::{
            branch::IfOp,
            general::{CastOp, ReadBuiltinOp},
            memory::{IndexOp, LoadOp, StoreOp},
            synchronization::{SyncOp, SyncScope, SyncScopeAttr},
        },
        pliron::{
            builtin::{attributes::IntegerAttr, ops::ConstantOp},
            value::Value,
        },
    },
    prelude::{ElemType, Scope},
};

pub(crate) fn branch(scope: &Scope, condition: Value, f: impl FnOnce(&Scope)) {
    let op = IfOp::new(scope.ctx_mut(), condition);
    let body = scope.branch_child(OpInserter::new_at_block_end(op.then_block(scope.ctx())));
    f(&body);
    body.terminate_yield();
    let other = scope.branch_child(OpInserter::new_at_block_end(op.else_block(scope.ctx())));
    other.terminate_yield();
    scope.register(&op);
    scope.update_flags_after_branch(&[body, other]);
}

pub(crate) fn barrier(scope: &Scope) {
    scope.register(&SyncOp::new(
        scope.ctx_mut(),
        SyncScopeAttr(SyncScope::Cube),
    ));
}

pub(crate) fn builtin(scope: &Scope, builtin: Builtin) -> Value {
    let index = ElemType::Index.to_type(scope.ctx());
    let raw = scope.register_with_result(&ReadBuiltinOp::new(scope.ctx_mut(), index, builtin));
    let ty = ElemType::UInt(UIntKind::U32).to_type(scope.ctx());
    scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, raw))
}

pub(crate) fn ptr(scope: &Scope, buffer: Value, index: Value) -> Value {
    let ty = ElemType::Index.to_type(scope.ctx());
    let index = scope.register_with_result(&CastOp::new(scope.ctx_mut(), ty, index));
    scope.register_with_result(&IndexOp::maybe_checked(
        scope.ctx_mut(),
        buffer,
        index,
        true,
    ))
}

pub(crate) fn load(scope: &Scope, buffer: Value, index: Value) -> Value {
    let ptr = ptr(scope, buffer, index);
    scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))
}

pub(crate) fn store(scope: &Scope, buffer: Value, index: Value, value: Value) {
    let ptr = ptr(scope, buffer, index);
    scope.register(&StoreOp::new(scope.ctx_mut(), ptr, value));
}

pub(crate) fn uint(scope: &Scope, value: u32) -> Value {
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
