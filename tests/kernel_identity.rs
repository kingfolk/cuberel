//! Real-device identity/reuse and isolated definition/backend failure checks.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use std::time::{SystemTime, UNIX_EPOCH};

use cubecl::{
    CompilationError, Device,
    ir::{
        dialect::{
            branch::ReturnOp,
            memory::{IndexOp, StoreOp},
        },
        pliron::{op::Op, operation::verify_operation},
    },
    prelude::{
        AddressType, Client, CubeCount, CubeDim, CubeKernel, ElemType, ExecutionMode,
        KernelDefinition, KernelId, KernelMetadata, KernelSettings,
    },
    server::{LaunchError, ServerError},
};
use cuberel::{
    Batch, CODEGEN_VERSION, ColumnData, DataType, Error, ExecutionError, Executor, Expr, ExprId,
    ExprNode, Parameters, Schema,
    abi::{BindingError, BindingLayout, PhysicalType},
    col,
    cubecl_abi::{DeviceBuffer, Submission},
    param, prepare,
};

fn batch<'a>(name: &str, a: &'a [f32], b: &'a [f32]) -> Batch<'a> {
    Batch::new(a.len())
        .with_column(name, ColumnData::F32(a))
        .unwrap()
        .with_column("b", ColumnData::F32(b))
        .unwrap()
}
fn params(scale: f32, bias: f32) -> Parameters {
    Parameters::new()
        .with("scale", scale)
        .unwrap()
        .with("bias", bias)
        .unwrap()
}

fn exercise_identity_and_reuse(device: Device) {
    let client = device.client();
    let executor = Executor::new(client.clone()).unwrap();
    // Unique schema name guarantees a cold key without clearing user caches or
    // assuming disk caching is disabled. Names are already part of BindingLayout.
    let a_name = format!(
        "a_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let schema = Schema::new(
        &[(&a_name, DataType::F32), ("b", DataType::F32)],
        &[("scale", DataType::F32), ("bias", DataType::F32)],
    )
    .unwrap();
    let add = prepare(&(col(&a_name) + param("scale")), &schema).unwrap();
    let mul = prepare(&(col(&a_name) * param("scale")), &schema).unwrap();
    let combined = prepare(
        &(col(&a_name) * param("scale") + col("b") + param("bias")),
        &schema,
    )
    .unwrap();
    let id = executor.kernel_id(&add);
    let identity = executor.kernel_identity(&add);
    assert_eq!(identity.version, CODEGEN_VERSION);
    assert_eq!(identity.normalization_version, 1);
    assert_eq!(identity.device_checksum, client.properties().checksum());
    assert_eq!(identity.target_name, client.name());
    assert_ne!(id, executor.kernel_id(&mul));
    assert_ne!(id, executor.kernel_id(&combined));
    assert_ne!(identity, executor.kernel_identity(&mul));
    assert_eq!(identity.layout, executor.kernel_identity(&mul).layout);
    let again = prepare(&(col(&a_name) + param("scale")), &schema).unwrap();
    assert_eq!(id, executor.kernel_id(&again));
    assert_eq!(identity, executor.kernel_identity(&again));
    // Settings are part of the public identity as well as CubeCL's KernelId.
    let other_size = if executor.cube_size() == 1 { 2 } else { 1 };
    let other = Executor::with_cube_size(client.clone(), other_size).unwrap();
    assert_ne!(id, other.kernel_id(&add));
    assert_ne!(identity, other.kernel_identity(&add));
    // No algebraic commutation/reassociation: operand order remains significant.
    let reversed = prepare(&(param("scale") + col(&a_name)), &schema).unwrap();
    assert_ne!(id, executor.kernel_id(&reversed));

    let a1 = [1.0, 2.0, 3.0];
    let b1 = [10.0, 20.0, 30.0];
    // Empty and invalid invocations of a cold key must not define anything.
    assert_eq!(
        executor
            .execute(&combined, &batch(&a_name, &[], &[]), &params(2.0, 1.0))
            .unwrap(),
        Vec::<f32>::new()
    );
    assert!(matches!(
        executor.execute(&combined, &batch(&a_name, &a1, &b1), &Parameters::new()),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::MissingInput { .. }
        )))
    ));
    let wrong = batch(&a_name, &a1, &b1);
    assert!(matches!(
        executor.execute(
            &combined,
            &wrong,
            &Parameters::new().with("scale", 2_i32).unwrap()
        ),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::TypeMismatch { .. }
        )))
    ));
    assert_eq!(executor.define_count(), 0);

    assert_eq!(
        executor
            .execute(&combined, &batch(&a_name, &a1, &b1), &params(2.0, 1.0))
            .unwrap(),
        [13.0, 25.0, 37.0]
    );
    assert_eq!(
        executor.define_count(),
        1,
        "cold key must define once; this is not a driver compile count"
    );
    assert_eq!(
        executor
            .execute(&combined, &batch(&a_name, &a1, &b1), &params(3.0, -5.0))
            .unwrap(),
        [8.0, 21.0, 34.0]
    );
    assert_eq!(executor.define_count(), 1);
    let a2 = [4.0, 5.0];
    let b2 = [100.0, 200.0];
    // New addresses AND row count do not specialize the kernel.
    assert_eq!(
        executor
            .execute(&combined, &batch(&a_name, &a2, &b2), &params(1.0, 0.0))
            .unwrap(),
        [104.0, 205.0]
    );
    assert_eq!(executor.define_count(), 1);
    assert_eq!(
        executor
            .execute(&mul, &batch(&a_name, &a1, &b1), &params(5.0, 0.0))
            .unwrap(),
        [5.0, 10.0, 15.0]
    );
    assert_eq!(executor.define_count(), 2);
    assert_eq!(
        executor
            .clone()
            .execute(&mul, &batch(&a_name, &a2, &b2), &params(-2.0, 0.0))
            .unwrap(),
        [-8.0, -10.0]
    );
    assert_eq!(executor.define_count(), 2);
    assert_eq!(
        executor
            .execute(&combined, &batch(&a_name, &a1, &b1), &params(10.0, 2.0))
            .unwrap(),
        [22.0, 42.0, 62.0]
    );
    assert_eq!(executor.define_count(), 2);
    assert_eq!(
        executor
            .execute(&add, &batch(&a_name, &a1, &b1), &params(2.0, 0.0))
            .unwrap(),
        [3.0, 4.0, 5.0]
    );
    assert_eq!(executor.define_count(), 3);
    assert_eq!(
        executor
            .execute(&again, &batch(&a_name, &a2, &b2), &params(3.0, 0.0))
            .unwrap(),
        [7.0, 8.0]
    );
    assert_eq!(executor.define_count(), 3);

    // Shared DAG and duplicated DSL nodes normalize to the same pure structure.
    let repeated = prepare(
        &((col(&a_name) * param("scale")) + (col(&a_name) * param("scale"))),
        &schema,
    )
    .unwrap();
    let shared = prepare(
        &Expr::from_nodes(
            vec![
                ExprNode::Column(a_name.clone()),
                ExprNode::Parameter("scale".into()),
                ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
                ExprNode::Add(ExprId::new(2), ExprId::new(2)),
            ],
            ExprId::new(3),
        ),
        &schema,
    )
    .unwrap();
    assert_eq!(executor.kernel_id(&repeated), executor.kernel_id(&shared));
    assert_eq!(executor.kernel_identity(&repeated).nodes.len(), 4);
    assert_eq!(
        executor
            .execute(&repeated, &batch(&a_name, &a1, &b1), &params(2.0, 0.0))
            .unwrap(),
        [4.0, 8.0, 12.0]
    );
    assert_eq!(executor.define_count(), 4);
    assert_eq!(
        executor
            .execute(&shared, &batch(&a_name, &a2, &b2), &params(3.0, 0.0))
            .unwrap(),
        [24.0, 30.0]
    );
    assert_eq!(executor.define_count(), 4);
    // A separately constructed executor shares runtime cache, not observations.
    let independent = Executor::with_cube_size(client, executor.cube_size()).unwrap();
    assert_eq!(
        independent
            .execute(&combined, &batch(&a_name, &a1, &b1), &params(2.0, 1.0))
            .unwrap(),
        [13.0, 25.0, 37.0]
    );
    assert_eq!(independent.define_count(), 0);
    assert_eq!(executor.define_count(), 4);
    // Identical nodes and ABI but a different output root must not reuse code.
    let leaf_root = prepare(
        &Expr::from_nodes(
            vec![
                ExprNode::Column(a_name.clone()),
                ExprNode::Parameter("scale".into()),
                ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
                ExprNode::Add(ExprId::new(2), ExprId::new(2)),
            ],
            ExprId::new(0),
        ),
        &schema,
    )
    .unwrap();
    assert_ne!(executor.kernel_id(&shared), executor.kernel_id(&leaf_root));
    let cloned = executor.clone();
    assert_eq!(
        cloned
            .execute(&leaf_root, &batch(&a_name, &a2, &b2), &params(3.0, 0.0))
            .unwrap(),
        a2
    );
    assert_eq!(executor.define_count(), 5);
    assert_eq!(cloned.define_count(), 5);
    assert_eq!(
        other
            .execute(&add, &batch(&a_name, &a1, &b1), &params(2.0, 0.0))
            .unwrap(),
        [3.0, 4.0, 5.0]
    );
    assert_eq!(other.define_count(), 1);
    assert_eq!(executor.define_count(), 5);
    println!(
        "{:?}: primary definitions=5; changed-settings definitions=1; rebinding/reprepare/shared-DAG reuse added zero definitions",
        device.runtime()
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FailureKind {
    DefinitionVerifier,
    BackendValidation,
}

// Test-only, with the EXACT production ABI. Malformed operations remain invalid
// until rejected; no invalid device code may execute and no OOB probe is launched.
struct IllegalIrKernel {
    client: Client,
    layout: BindingLayout,
    settings: KernelSettings,
    kind: FailureKind,
}
impl KernelMetadata for IllegalIrKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((self.layout.clone(), self.kind))
            .cube_dim(self.settings.cube_dim)
            .address_type(self.settings.address_type)
            .mode(self.settings.execution_mode)
    }
    fn address_type(&self) -> ElemType {
        self.settings.address_type.unsigned_type()
    }
}
impl CubeKernel for IllegalIrKernel {
    fn define(&self) -> KernelDefinition {
        let settings = self
            .settings
            .clone()
            .kernel_name(self.id().entrypoint_name("cuberel_illegal"));
        let declaration = self.layout.declare(&self.client, settings).unwrap();
        let scope = declaration.scope();
        let zero = scope.const_usize(0);
        let ptr = scope.register_with_result(&IndexOp::maybe_checked(
            scope.ctx_mut(),
            declaration.output(),
            zero,
            true,
        ));
        // This invalid store must be rejected: IndexType is not F32.
        scope.register(&StoreOp::new(scope.ctx_mut(), ptr, zero));
        scope.register(&ReturnOp::new(scope.ctx_mut()));
        let error = verify_operation(scope.state().module.get_operation(), scope.ctx())
            .expect_err("test fixture must actually contain invalid IR");
        if self.kind == FailureKind::DefinitionVerifier {
            panic!("intentional test-only IR verification failure: {error}");
        }
        // Forward the verifier diagnostic through CubeCL's supported error list.
        // Both backends refuse this before lowering. Do not rely on lowerers to
        // rediscover ill-typed stores: LLVM opaque pointers can erase that mismatch.
        scope.push_error(error.to_string());
        declaration.build()
    }
}

fn is_compilation_error(error: &ServerError, definition: bool) -> bool {
    match error {
        ServerError::Launch(LaunchError::CompilationError(
            CompilationError::ExpansionPanicked { .. },
        )) => definition,
        ServerError::Launch(LaunchError::CompilationError(CompilationError::Validation {
            ..
        })) => !definition,
        ServerError::Unwritten { root, .. } => is_compilation_error(root, definition),
        ServerError::Several { errors, .. } => {
            !errors.is_empty() && errors.iter().all(|e| is_compilation_error(e, definition))
        }
        _ => false,
    }
}

fn exercise_illegal_ir_error_path(device: Device) {
    let client = device.client();
    let executor = Executor::new(client.clone()).unwrap();
    let schema = Schema::new(&[("a", DataType::F32)], &[("scale", DataType::F32)]).unwrap();
    let program = prepare(&(col("a") + param("scale")), &schema).unwrap();
    let layout = BindingLayout::new(&program);
    for kind in [
        FailureKind::DefinitionVerifier,
        FailureKind::BackendValidation,
    ] {
        let input = client.create_from_slice(&1.0_f32.to_ne_bytes());
        let output = client.create_from_slice(&(-999.0_f32).to_ne_bytes());
        // SAFETY: runtime-owned aligned allocations of initialized scalar F32.
        let buffers = unsafe {
            [
                DeviceBuffer::from_raw_parts(input.clone(), PhysicalType::F32, 1),
                DeviceBuffer::from_raw_parts(output.clone(), PhysicalType::F32, 1),
            ]
        };
        let bound = layout
            .bind_device(&client, 1, &buffers, &params(2.0, 0.0))
            .unwrap();
        let settings = KernelSettings::new(
            CubeDim::new_1d(1).into(),
            ExecutionMode::Checked,
            AddressType::U32,
        );
        let kernel = IllegalIrKernel {
            client: client.clone(),
            layout: layout.clone(),
            settings: settings.clone(),
            kind,
        };
        // SAFETY: exact ABI and valid allocations; intentionally ill-typed IR
        // cannot reach execution: both cases assert verifier rejection in define.
        // A backend failing to reject it is a defect, never an acceptable result.
        let submitted =
            unsafe { bound.launch(&client, settings, kernel, CubeCount::Static(1, 1, 1)) }.unwrap();
        assert_eq!(submitted, Submission::Enqueued);
        let error = client
            .read_one(output)
            .expect_err("failed output must not return the sentinel as successful data");
        assert!(
            is_compilation_error(&error, kind == FailureKind::DefinitionVerifier),
            "wrong error category: {error:?}"
        );
        assert_eq!(&client.read_one(input).unwrap()[..], &1.0_f32.to_ne_bytes());
        assert_eq!(
            executor
                .execute(
                    &program,
                    &batch("a", &[5.0, 6.0], &[0.0, 0.0]),
                    &params(2.0, 0.0)
                )
                .unwrap(),
            [7.0, 8.0]
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_kernel_identity_and_reuse() {
    exercise_identity_and_reuse(Device::cpu().expect("real CPU runtime required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_illegal_ir_error_path() {
    exercise_illegal_ir_error_path(Device::cpu().expect("real CPU runtime required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_kernel_identity_and_reuse() {
    exercise_identity_and_reuse(
        Device::metal(Default::default()).expect("real Metal runtime required"),
    );
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_illegal_ir_error_path() {
    exercise_illegal_ir_error_path(
        Device::metal(Default::default()).expect("real Metal runtime required"),
    );
}

#[cfg(all(feature = "cpu", feature = "metal", target_vendor = "apple"))]
#[test]
fn cpu_and_metal_have_distinct_target_identities() {
    let schema = Schema::new(&[("a", DataType::F32)], &[]).unwrap();
    let program = prepare(&col("a"), &schema).unwrap();
    let cpu = Executor::new(Device::cpu().unwrap().client()).unwrap();
    let metal = Executor::new(Device::metal(Default::default()).unwrap().client()).unwrap();
    assert_ne!(
        cpu.kernel_identity(&program),
        metal.kernel_identity(&program)
    );
    assert_ne!(cpu.kernel_id(&program), metal.kernel_id(&program));
}
