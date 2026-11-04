use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use cubecl::{
    ir::{
        Builtin, FloatKind, OpInserter, UIntKind,
        attributes::FloatAttr,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::{
                FEqualOp, FGreaterThanOp, FGreaterThanOrEqualOp, FLessThanOp, FLessThanOrEqualOp,
                FNotEqualOp, ULessThanOp,
            },
            general::{BoolAndOp, BoolNotOp, BoolOrOp, CastOp, ReadBuiltinOp},
            math::{FMulOp, FmaOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{builtin::ops::ConstantOp, op::Op, operation::verify_operation, value::Value},
    },
    prelude::{
        Client, CubeKernel, ElemType, KernelDefinition, KernelId, KernelMetadata, KernelSettings,
    },
};

use crate::{Comparison, DataType, ExprId, KernelProgram, ScalarOp, TypedExpr, abi::BindingLayout};

// Change when emitted semantics or instruction policy changes.
pub const CODEGEN_VERSION: u32 = 4;

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
            layout: BindingLayout::new(&program),
            program,
            client,
            settings,
            define_count,
        }
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
        KernelId::new::<Self>()
            .info(self.identity())
            .cube_dim(self.settings.cube_dim)
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
        let declaration = self
            .layout
            .declare(&self.client, settings)
            .expect("trusted emitter settings");
        let scope = declaration.scope();
        // No contraction, reassociation, or assumptions about finite inputs.
        scope.state_mut().modes.fp_math_mode = Default::default();
        let f32_ty = ElemType::Float(FloatKind::F32).to_type(scope.ctx());
        let one = FloatAttr::from_f64(scope.ctx(), f32_ty, 1.0);
        let one = scope.register_with_result(&ConstantOp::new(scope.ctx_mut(), Box::new(one)));
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
        let branch = IfOp::new(scope.ctx_mut(), in_range);
        let body = scope.branch_child(OpInserter::new_at_block_end(branch.then_block(scope.ctx())));
        let mut values: Vec<Value> = Vec::with_capacity(self.program.nodes().len());
        // This match runs once while constructing IR, not once per row on a device.
        // Arena order preserves DAG sharing and the original arithmetic grouping.
        for node in self.program.nodes() {
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
                    let (a, b) = (values[a.index()], values[b.index()]);
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
                        Comparison::LessThanOrEqual => body
                            .register_with_result(&FLessThanOrEqualOp::new(body.ctx_mut(), a, b)),
                        Comparison::GreaterThan => {
                            body.register_with_result(&FGreaterThanOp::new(body.ctx_mut(), a, b))
                        }
                        Comparison::GreaterThanOrEqual => body.register_with_result(
                            &FGreaterThanOrEqualOp::new(body.ctx_mut(), a, b),
                        ),
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
            };
            values.push(value);
        }
        let ptr = body.register_with_result(&IndexOp::maybe_checked(
            body.ctx_mut(),
            declaration.output(),
            index,
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
        body.terminate_yield();
        let other =
            scope.branch_child(OpInserter::new_at_block_end(branch.else_block(scope.ctx())));
        other.terminate_yield();
        scope.register(&branch);
        scope.update_flags_after_branch(&[body, other]);
        // Verify a complete module: both regions yield and the entry function
        // returns. KernelBuilder::build alone does not add this root terminator.
        scope.register(&ReturnOp::new(scope.ctx_mut()));
        verify_operation(scope.state().module.get_operation(), scope.ctx())
            .expect("Cuberel emitted invalid IR");
        // CubeCL catches define panics as CompilationError::ExpansionPanicked;
        // the output read in Executor observes that failure instead of stale data.
        declaration.build()
    }
}
