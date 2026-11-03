use cubecl::{
    ir::{
        Builtin, FloatKind, OpInserter,
        attributes::FloatAttr,
        dialect::{
            branch::{IfOp, ReturnOp},
            cmp::ULessThanOp,
            general::ReadBuiltinOp,
            math::{FMulOp, FmaOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{builtin::ops::ConstantOp, op::Op, operation::verify_operation, value::Value},
    },
    prelude::{
        Client, CubeKernel, ElemType, KernelDefinition, KernelId, KernelMetadata, KernelSettings,
    },
};

use crate::{ExprId, KernelProgram, ScalarOp, TypedExpr, abi::BindingLayout};

// Change when emitted semantics or instruction policy changes.
const CODEGEN_VERSION: u32 = 3;

// Minimal structural identity required for correct execution. Normalization and
// cache instrumentation are a separate milestone item. Values/handles are absent.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExpressionIdentity {
    version: u32,
    nodes: Vec<TypedExpr>,
    root: ExprId,
    layout: BindingLayout,
    device_checksum: u64,
}

pub(crate) struct ExpressionKernel {
    program: KernelProgram,
    layout: BindingLayout,
    client: Client,
    settings: KernelSettings,
}

impl ExpressionKernel {
    pub(crate) fn new(program: KernelProgram, client: Client, settings: KernelSettings) -> Self {
        Self {
            layout: BindingLayout::new(&program),
            program,
            client,
            settings,
        }
    }
}

impl KernelMetadata for ExpressionKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info(ExpressionIdentity {
                version: CODEGEN_VERSION,
                nodes: self.program.nodes().to_vec(),
                root: self.program.root(),
                layout: self.layout.clone(),
                device_checksum: self.client.properties().checksum(),
            })
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
            };
            values.push(value);
        }
        let ptr = body.register_with_result(&IndexOp::maybe_checked(
            body.ctx_mut(),
            declaration.output(),
            index,
            true,
        ));
        body.register(&StoreOp::new(
            body.ctx_mut(),
            ptr,
            values[self.program.root().index()],
        ));
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
