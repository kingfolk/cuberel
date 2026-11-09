use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cubecl::{
    ir::{
        Builtin, ConstantValue, FloatKind, IntKind, OpInserter, UIntKind,
        attributes::{BoolAttr, FloatAttr},
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::{
                FEqualOp, FGreaterThanOp, FGreaterThanOrEqualOp, FLessThanOp, FLessThanOrEqualOp,
                FNotEqualOp, IEqualOp, INotEqualOp, SGreaterThanOp, SGreaterThanOrEqualOp,
                SLessThanOp, SLessThanOrEqualOp, UGreaterThanOp, ULessThanOp,
            },
            general::{
                BoolAndOp, BoolNotOp, BoolOrOp, CastOp, ReadBuiltinOp, ReinterpretCastOp, SelectOp,
            },
            math::{FMulOp, FmaOp, IAddOp, IMulOp, ISubOp, UDivOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{
            builtin::{attributes::IntegerAttr, ops::ConstantOp},
            op::Op,
            operation::verify_operation,
            value::Value,
        },
    },
    prelude::{
        Client, CubeKernel, ElemType, KernelDefinition, KernelId, KernelMetadata, KernelSettings,
        Scope,
    },
};

use crate::{
    Comparison, DataType, ExprId, KernelProgram, ScalarOp, TypedExpr, abi::BindingLayout,
    cubecl_abi::KernelDeclaration,
};

// Change when emitted semantics or instruction policy changes.
pub const CODEGEN_VERSION: u32 = 9;

/// Structural identity for compilation and caching of an expression kernel.
/// Dynamic values, handles, and row counts are absent.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExpressionIdentity {
    pub version: u32,
    pub normalization_version: u32,
    pub settings: KernelSettings,
    pub target_name: String,
    pub nodes: Vec<TypedExpr>,
    pub root: ExprId,
    pub layout: BindingLayout,
    pub device_checksum: u64,
}

pub(crate) struct ExpressionKernel {
    program: KernelProgram,
    selected: bool,
    layout: BindingLayout,
    client: Client,
    settings: KernelSettings,
    define_count: Arc<AtomicUsize>,
}

impl ExpressionKernel {
    pub(crate) fn new(
        program: KernelProgram,
        client: Client,
        settings: KernelSettings,
        define_count: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            selected: false,
            layout: BindingLayout::new(&program),
            program,
            client,
            settings,
            define_count,
        }
    }

    pub(crate) fn selected(
        program: KernelProgram,
        client: Client,
        settings: KernelSettings,
        define_count: Arc<AtomicUsize>,
    ) -> Self {
        let mut kernel = Self::new(program, client, settings, define_count);
        kernel.selected = true;
        kernel
    }
    pub(crate) fn identity(&self) -> ExpressionIdentity {
        let mut nodes = Vec::new();
        let mut remapped: Vec<ExprId> = Vec::with_capacity(self.program.nodes().len());
        let mut interned = HashMap::new();
        // Remove duplicate pure nodes; preserve operand order and floating-point
        // grouping. All arena nodes, including dead inputs, still participate.
        for node in self.program.nodes() {
            let op = match node.op() {
                ScalarOp::Add(a, b) => ScalarOp::Add(remapped[a.index()], remapped[b.index()]),
                ScalarOp::Mul(a, b) => ScalarOp::Mul(remapped[a.index()], remapped[b.index()]),
                ScalarOp::Compare(op, a, b) => {
                    ScalarOp::Compare(op, remapped[a.index()], remapped[b.index()])
                }
                ScalarOp::And(a, b) => ScalarOp::And(remapped[a.index()], remapped[b.index()]),
                ScalarOp::Or(a, b) => ScalarOp::Or(remapped[a.index()], remapped[b.index()]),
                ScalarOp::Not(a) => ScalarOp::Not(remapped[a.index()]),
                ScalarOp::IsNull(a) => ScalarOp::IsNull(remapped[a.index()]),
                leaf => leaf,
            };
            let canonical = node.with_op(op);
            let id = *interned.entry(canonical.clone()).or_insert_with(|| {
                let id = ExprId::new(nodes.len());
                nodes.push(canonical);
                id
            });
            remapped.push(id);
        }
        ExpressionIdentity {
            version: CODEGEN_VERSION,
            normalization_version: 1,
            nodes,
            root: remapped[self.program.root().index()],
            layout: self.layout.clone(),
            device_checksum: self.client.properties().checksum(),
            target_name: self.client.name().into(),
            settings: self.settings.clone(),
        }
    }
}

impl KernelMetadata for ExpressionKernel {
    fn id(&self) -> KernelId {
        let id = KernelId::new::<Self>();
        let id = if self.selected {
            id.info((self.identity(), "selected-prefix-v1"))
        } else {
            id.info(self.identity())
        };
        id.cube_dim(self.settings.cube_dim)
            .address_type(self.settings.address_type)
            .mode(self.settings.execution_mode)
    }

    fn address_type(&self) -> ElemType {
        self.settings.address_type.unsigned_type()
    }
}

impl CubeKernel for ExpressionKernel {
    fn define(&self) -> KernelDefinition {
        self.define_count.fetch_add(1, Ordering::Relaxed);
        let settings = self
            .settings
            .clone()
            .kernel_name(self.id().entrypoint_name("cuberel_expr"));
        let declaration = if self.selected {
            self.layout.declare_selected(&self.client, settings)
        } else {
            self.layout.declare(&self.client, settings)
        }
        .expect("trusted emitter settings");
        let scope = declaration.scope();
        // No contraction, reassociation, or assumptions about finite inputs.
        scope.state_mut().modes.fp_math_mode = Default::default();
        let index_ty = ElemType::Index.to_type(scope.ctx());
        let index = scope.register_with_result(&ReadBuiltinOp::new(
            scope.ctx_mut(),
            index_ty,
            Builtin::AbsolutePos,
        ));
        let in_range = scope.register_with_result(&ULessThanOp::new(
            scope.ctx_mut(),
            index,
            declaration.row_count(),
        ));
        branch(scope, in_range, |outer| {
            if let Some((indices, count)) = declaration.selected() {
                let zero = literal(outer, ConstantValue::UInt(0), UIntKind::U32.into());
                let zero =
                    outer.register_with_result(&CastOp::new(outer.ctx_mut(), index_ty, zero));
                let ptr = outer.register_with_result(&IndexOp::maybe_checked(
                    outer.ctx_mut(),
                    count,
                    zero,
                    true,
                ));
                let count = outer.register_with_result(&LoadOp::new(outer.ctx_mut(), ptr));
                let count =
                    outer.register_with_result(&CastOp::new(outer.ctx_mut(), index_ty, count));
                let active =
                    outer.register_with_result(&ULessThanOp::new(outer.ctx_mut(), index, count));
                branch(outer, active, |active| {
                    let ptr = active.register_with_result(&IndexOp::maybe_checked(
                        active.ctx_mut(),
                        indices,
                        index,
                        true,
                    ));
                    let row = active.register_with_result(&LoadOp::new(active.ctx_mut(), ptr));
                    let row =
                        active.register_with_result(&CastOp::new(active.ctx_mut(), index_ty, row));
                    let valid_row = active.register_with_result(&ULessThanOp::new(
                        active.ctx_mut(),
                        row,
                        declaration.row_count(),
                    ));
                    branch(active, valid_row, |body| {
                        self.emit_row(&declaration, body, row, index)
                    });
                });
            } else {
                self.emit_row(&declaration, outer, index, index);
            }
        });
        scope.register(&ReturnOp::new(scope.ctx_mut()));
        verify_operation(scope.state().module.get_operation(), scope.ctx())
            .expect("Cuberel emitted invalid IR");
        declaration.build()
    }
}

impl ExpressionKernel {
    fn emit_row(
        &self,
        declaration: &KernelDeclaration,
        body: &Scope,
        index: Value,
        output_index: Value,
    ) {
        let scope = body;
        let f32_ty = ElemType::Float(FloatKind::F32).to_type(scope.ctx());
        let one = FloatAttr::from_f64(scope.ctx(), f32_ty, 1.0);
        let one = scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(one)));
        let zero = FloatAttr::from_f64(scope.ctx(), f32_ty, 0.0);
        let zero = scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(zero)));
        let valid_true =
            scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(BoolAttr(true))));
        let int_zero = literal(scope, ConstantValue::Int(0), ElemType::Int(IntKind::I32));
        let mut overflow_any = scope
            .register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(BoolAttr(false))));
        let bool_ty = ElemType::Bool.to_type(body.ctx());
        let mut validities: Vec<Value> = Vec::with_capacity(self.program.nodes().len());
        let mut values: Vec<Value> = Vec::with_capacity(self.program.nodes().len());
        // This match runs once while constructing IR, not once per row on a device.
        // Arena order preserves DAG sharing and the original arithmetic grouping.
        for (node_index, node) in self.program.nodes().iter().enumerate() {
            let mut overflow = None;
            let value = match node.op() {
                ScalarOp::Column(id) => {
                    let buffer = declaration.column(id).expect("prepared column in layout");
                    let ptr = body.register_with_result(&IndexOp::maybe_checked(
                        body.ctx_mut(),
                        buffer,
                        index,
                        true,
                    ));
                    body.register_with_result(&LoadOp::new(body.ctx_mut(), ptr))
                }
                ScalarOp::Parameter(id) => declaration
                    .parameter(id)
                    .expect("prepared parameter in layout"),
                // Pinned CubeCL unconditionally fuses FMul -> FAdd despite
                // fp_math_mode. fma(1,a,b) is exactly one rounded addition
                // (1*a is exact), including signed zeros. It avoids that
                // pattern and never fuses arithmetic across expression nodes.
                ScalarOp::Add(a, b) | ScalarOp::Mul(a, b) if node.data_type() == DataType::I32 => {
                    let (value, flag) = checked_i32(
                        body,
                        values[a.index()],
                        values[b.index()],
                        matches!(node.op(), ScalarOp::Mul(..)),
                    );
                    overflow = Some(flag);
                    value
                }
                ScalarOp::Add(a, b) => body.register_with_result(&FmaOp::new(
                    body.ctx_mut(),
                    one,
                    values[a.index()],
                    values[b.index()],
                )),
                ScalarOp::Mul(a, b) => body.register_with_result(&FMulOp::new(
                    body.ctx_mut(),
                    values[a.index()],
                    values[b.index()],
                )),
                ScalarOp::Compare(comparison, a, b) => {
                    let integer = self.program.nodes()[a.index()].data_type() == DataType::I32;
                    let (a, b) = (values[a.index()], values[b.index()]);
                    if integer {
                        match comparison {
                            Comparison::Equal => {
                                body.register_with_result(&IEqualOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::NotEqual => {
                                body.register_with_result(&INotEqualOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::LessThan => {
                                body.register_with_result(&SLessThanOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::LessThanOrEqual => body.register_with_result(
                                &SLessThanOrEqualOp::new(body.ctx_mut(), a, b),
                            ),
                            Comparison::GreaterThan => body
                                .register_with_result(&SGreaterThanOp::new(body.ctx_mut(), a, b)),
                            Comparison::GreaterThanOrEqual => body.register_with_result(
                                &SGreaterThanOrEqualOp::new(body.ctx_mut(), a, b),
                            ),
                        }
                    } else {
                        match comparison {
                            Comparison::Equal => {
                                body.register_with_result(&FEqualOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::NotEqual => {
                                body.register_with_result(&FNotEqualOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::LessThan => {
                                body.register_with_result(&FLessThanOp::new(body.ctx_mut(), a, b))
                            }
                            Comparison::LessThanOrEqual => body.register_with_result(
                                &FLessThanOrEqualOp::new(body.ctx_mut(), a, b),
                            ),
                            Comparison::GreaterThan => body
                                .register_with_result(&FGreaterThanOp::new(body.ctx_mut(), a, b)),
                            Comparison::GreaterThanOrEqual => body.register_with_result(
                                &FGreaterThanOrEqualOp::new(body.ctx_mut(), a, b),
                            ),
                        }
                    }
                }
                ScalarOp::And(a, b) => body.register_with_result(&BoolAndOp::new(
                    body.ctx_mut(),
                    values[a.index()],
                    values[b.index()],
                )),
                ScalarOp::Or(a, b) => body.register_with_result(&BoolOrOp::new(
                    body.ctx_mut(),
                    values[a.index()],
                    values[b.index()],
                )),
                ScalarOp::Not(a) => {
                    body.register_with_result(&BoolNotOp::new(body.ctx_mut(), values[a.index()]))
                }
                ScalarOp::IsNull(a) => body
                    .register_with_result(&BoolNotOp::new(body.ctx_mut(), validities[a.index()])),
            };
            let valid = match node.op() {
                ScalarOp::Column(id) if node.is_nullable() => {
                    let buffer = declaration
                        .column_validity(id)
                        .expect("nullable column validity");
                    let ptr = body.register_with_result(&IndexOp::maybe_checked(
                        body.ctx_mut(),
                        buffer,
                        index,
                        true,
                    ));
                    let raw = body.register_with_result(&LoadOp::new(body.ctx_mut(), ptr));
                    body.register_with_result(&CastOp::new(body.ctx_mut(), bool_ty, raw))
                }
                ScalarOp::Parameter(id) if node.is_nullable() => {
                    let raw = declaration
                        .parameter_validity(id)
                        .expect("nullable parameter validity");
                    body.register_with_result(&CastOp::new(body.ctx_mut(), bool_ty, raw))
                }
                ScalarOp::Add(a, b) | ScalarOp::Mul(a, b) | ScalarOp::Compare(_, a, b)
                    if node.is_nullable() =>
                {
                    body.register_with_result(&BoolAndOp::new(
                        body.ctx_mut(),
                        validities[a.index()],
                        validities[b.index()],
                    ))
                }
                ScalarOp::And(a, b) | ScalarOp::Or(a, b) if node.is_nullable() => {
                    // Payloads of unknown Bool operands are canonical false.
                    // AND is known when true or either operand is known false.
                    // OR is known when true or both operands are known false.
                    let not_a = body
                        .register_with_result(&BoolNotOp::new(body.ctx_mut(), values[a.index()]));
                    let not_b = body
                        .register_with_result(&BoolNotOp::new(body.ctx_mut(), values[b.index()]));
                    let a_false = body.register_with_result(&BoolAndOp::new(
                        body.ctx_mut(),
                        validities[a.index()],
                        not_a,
                    ));
                    let b_false = body.register_with_result(&BoolAndOp::new(
                        body.ctx_mut(),
                        validities[b.index()],
                        not_b,
                    ));
                    let known_false = match node.op() {
                        ScalarOp::And(..) => body.register_with_result(&BoolOrOp::new(
                            body.ctx_mut(),
                            a_false,
                            b_false,
                        )),
                        ScalarOp::Or(..) => body.register_with_result(&BoolAndOp::new(
                            body.ctx_mut(),
                            a_false,
                            b_false,
                        )),
                        _ => unreachable!(),
                    };
                    body.register_with_result(&BoolOrOp::new(body.ctx_mut(), value, known_false))
                }
                ScalarOp::Not(a) if node.is_nullable() => validities[a.index()],
                _ => valid_true,
            };
            // Every payload load is initialized. Canonicalize invalid F32 to +0
            // and invalid Bool to false before consumers use them.
            let value = if node.is_nullable() {
                match node.data_type() {
                    DataType::F32 => body.register_with_result(&SelectOp::new(
                        body.ctx_mut(),
                        valid,
                        value,
                        zero,
                    )),
                    DataType::Bool => {
                        body.register_with_result(&BoolAndOp::new(body.ctx_mut(), valid, value))
                    }
                    DataType::I32 => body.register_with_result(&SelectOp::new(
                        body.ctx_mut(),
                        valid,
                        value,
                        int_zero,
                    )),
                }
            } else {
                value
            };
            if self.program.reachable()[node_index]
                && let Some(flag) = overflow
            {
                let flag = body.register_with_result(&BoolAndOp::new(body.ctx_mut(), valid, flag));
                overflow_any =
                    body.register_with_result(&BoolOrOp::new(body.ctx_mut(), overflow_any, flag));
            }
            validities.push(valid);
            values.push(value);
        }
        let ptr = body.register_with_result(&IndexOp::maybe_checked(
            body.ctx_mut(),
            declaration.output(),
            output_index,
            true,
        ));
        let result = values[self.program.root().index()];
        // Device booleans stay in SSA. The portable buffer ABI stores 0/1 U32,
        // avoiding backend-dependent bool sizes and host bool bit patterns.
        let result = if self.program.output_type() == DataType::Bool {
            let ty = ElemType::UInt(UIntKind::U32).to_type(body.ctx());
            body.register_with_result(&CastOp::new(body.ctx_mut(), ty, result))
        } else {
            result
        };
        body.register(&StoreOp::new(body.ctx_mut(), ptr, result));
        if let Some(buffer) = declaration.output_validity() {
            let ptr = body.register_with_result(&IndexOp::maybe_checked(
                body.ctx_mut(),
                buffer,
                output_index,
                true,
            ));
            let ty = ElemType::UInt(UIntKind::U32).to_type(body.ctx());
            let valid = body.register_with_result(&CastOp::new(
                body.ctx_mut(),
                ty,
                validities[self.program.root().index()],
            ));
            body.register(&StoreOp::new(body.ctx_mut(), ptr, valid));
        }
        if let Some(buffer) = declaration.output_error() {
            let ptr = body.register_with_result(&IndexOp::maybe_checked(
                body.ctx_mut(),
                buffer,
                output_index,
                true,
            ));
            let ty = ElemType::UInt(UIntKind::U32).to_type(body.ctx());
            let flag = if self.selected {
                let row = body.register_with_result(&CastOp::new(body.ctx_mut(), ty, index));
                let one = literal(body, ConstantValue::UInt(1), UIntKind::U32.into());
                let row = body.register_with_result(&IAddOp::new(body.ctx_mut(), row, one));
                let zero = literal(body, ConstantValue::UInt(0), UIntKind::U32.into());
                body.register_with_result(&SelectOp::new(body.ctx_mut(), overflow_any, row, zero))
            } else {
                body.register_with_result(&CastOp::new(body.ctx_mut(), ty, overflow_any))
            };
            body.register(&StoreOp::new(body.ctx_mut(), ptr, flag));
        }
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

fn literal(scope: &Scope, value: ConstantValue, ty: ElemType) -> Value {
    let attribute = value.as_attribute(scope.ctx(), ty);
    scope.register_with_result(&ConstantOp::new(
        scope.ctx_mut(),
        Box::new(
            attribute
                .downcast_ref::<IntegerAttr>()
                .expect("integer literal")
                .clone(),
        ),
    ))
}

// All low-word arithmetic is U32 modulo 2^32. No signed overflow, I64,
// signed division, zero division or host-side expression evaluation is used.
fn checked_i32(scope: &Scope, a: Value, b: Value, multiply: bool) -> (Value, Value) {
    let u32_ty = ElemType::UInt(UIntKind::U32).to_type(scope.ctx());
    let i32_ty = ElemType::Int(IntKind::I32).to_type(scope.ctx());
    let zero = literal(scope, ConstantValue::UInt(0), UIntKind::U32.into());
    let int_zero = literal(scope, ConstantValue::Int(0), IntKind::I32.into());
    let ua = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), u32_ty, a));
    let ub = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), u32_ty, b));
    let low = if multiply {
        scope.register_with_result(&IMulOp::new(scope.ctx_mut(), ua, ub))
    } else {
        scope.register_with_result(&IAddOp::new(scope.ctx_mut(), ua, ub))
    };
    let result = scope.register_with_result(&ReinterpretCastOp::new(scope.ctx_mut(), i32_ty, low));
    let neg_a = scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), a, int_zero));
    let neg_b = scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), b, int_zero));
    let same_sign = bool_equal(scope, neg_a, neg_b);
    let overflow = if multiply {
        let neg_ua = scope.register_with_result(&ISubOp::new(scope.ctx_mut(), zero, ua));
        let neg_ub = scope.register_with_result(&ISubOp::new(scope.ctx_mut(), zero, ub));
        let magnitude_a =
            scope.register_with_result(&SelectOp::new(scope.ctx_mut(), neg_a, neg_ua, ua));
        let magnitude_b =
            scope.register_with_result(&SelectOp::new(scope.ctx_mut(), neg_b, neg_ub, ub));
        let positive_limit = literal(scope, ConstantValue::UInt(2147483647), UIntKind::U32.into());
        let negative_limit = literal(scope, ConstantValue::UInt(2147483648), UIntKind::U32.into());
        let limit = scope.register_with_result(&SelectOp::new(
            scope.ctx_mut(),
            same_sign,
            positive_limit,
            negative_limit,
        ));
        let one = literal(scope, ConstantValue::UInt(1), UIntKind::U32.into());
        let b_zero = scope.register_with_result(&IEqualOp::new(scope.ctx_mut(), magnitude_b, zero));
        let denominator =
            scope.register_with_result(&SelectOp::new(scope.ctx_mut(), b_zero, one, magnitude_b));
        let quotient =
            scope.register_with_result(&UDivOp::new(scope.ctx_mut(), limit, denominator));
        let too_large = scope.register_with_result(&UGreaterThanOp::new(
            scope.ctx_mut(),
            magnitude_a,
            quotient,
        ));
        let b_nonzero = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), b_zero));
        scope.register_with_result(&BoolAndOp::new(scope.ctx_mut(), too_large, b_nonzero))
    } else {
        let negative_result =
            scope.register_with_result(&SLessThanOp::new(scope.ctx_mut(), result, int_zero));
        let same_result_sign = bool_equal(scope, neg_a, negative_result);
        let changed_sign =
            scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), same_result_sign));
        scope.register_with_result(&BoolAndOp::new(scope.ctx_mut(), same_sign, changed_sign))
    };
    (result, overflow)
}

// Pinned Metal cannot emit cmp.bool_equal; equivalent supported Bool ops.
fn bool_equal(scope: &Scope, a: Value, b: Value) -> Value {
    let both = scope.register_with_result(&BoolAndOp::new(scope.ctx_mut(), a, b));
    let na = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), a));
    let nb = scope.register_with_result(&BoolNotOp::new(scope.ctx_mut(), b));
    let neither = scope.register_with_result(&BoolAndOp::new(scope.ctx_mut(), na, nb));
    scope.register_with_result(&BoolOrOp::new(scope.ctx_mut(), both, neither))
}
