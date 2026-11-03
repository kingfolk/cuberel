//! Hand-written ABI probe, not the product expression-to-kernel compiler.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use cubecl::{
    Device,
    ir::{
        Builtin, FloatKind, OpInserter,
        dialect::{
            branch::IfOp,
            cmp::ULessThanOp,
            general::{BufferLenOp, CastOp, ReadBuiltinOp},
            math::{FAddOp, FMulOp},
            memory::{IndexOp, LoadOp, StoreOp},
        },
        pliron::{r#type::Typed, value::Value},
    },
    prelude::{
        AddressType, Client, CubeCount, CubeDim, CubeKernel, ElemType, ExecutionMode,
        KernelDefinition, KernelId, KernelMetadata, KernelSettings, Scope,
    },
    server::Handle,
};

use cuberel::{
    Batch, ColumnData, ColumnId, DataType, KernelProgram, ParameterId, Parameters, Schema,
    abi::{BindingError, BindingLayout, PhysicalType},
    col,
    cubecl_abi::{DeviceBuffer, Submission},
    param, prepare,
};

fn program() -> KernelProgram {
    let schema = Schema::new(
        &[
            ("b", DataType::F32),
            ("unused", DataType::I32),
            ("a", DataType::F32),
        ],
        &[
            ("bias", DataType::F32),
            ("unused", DataType::I32),
            ("scale", DataType::F32),
        ],
    )
    .unwrap();
    prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )
    .unwrap()
}

fn settings() -> KernelSettings {
    KernelSettings::new(
        CubeDim::new_1d(4).into(),
        ExecutionMode::Checked,
        AddressType::U32,
    )
}

struct ProbeKernel {
    client: Client,
    layout: BindingLayout,
    settings: KernelSettings,
    a: ColumnId,
    b: ColumnId,
    scale: ParameterId,
    bias: ParameterId,
    must_not_define: bool,
}

impl ProbeKernel {
    fn new(client: &Client, program: &KernelProgram, must_not_define: bool) -> Self {
        Self {
            client: client.clone(),
            layout: BindingLayout::new(program),
            settings: settings(),
            a: program.schema().column_id("a").unwrap(),
            b: program.schema().column_id("b").unwrap(),
            scale: program.schema().parameter_id("scale").unwrap(),
            bias: program.schema().parameter_id("bias").unwrap(),
            must_not_define,
        }
    }
}

impl KernelMetadata for ProbeKernel {
    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
            .info((self.layout.clone(), self.must_not_define))
            .cube_dim(self.settings.cube_dim)
            .address_type(self.settings.address_type)
            .mode(self.settings.execution_mode)
    }
    fn address_type(&self) -> ElemType {
        self.settings.address_type.unsigned_type()
    }
}

fn load(scope: &Scope, buffer: Value, index: Value) -> Value {
    let ptr = scope.register_with_result(&IndexOp::maybe_checked(
        scope.ctx_mut(),
        buffer,
        index,
        true,
    ));
    scope.register_with_result(&LoadOp::new(scope.ctx_mut(), ptr))
}

impl CubeKernel for ProbeKernel {
    fn define(&self) -> KernelDefinition {
        assert!(
            !self.must_not_define,
            "an empty invocation must not compile or launch"
        );
        let settings = self
            .settings
            .clone()
            .kernel_name(self.id().entrypoint_name("abi_probe"));
        let declaration = self.layout.declare(&self.client, settings).unwrap();
        let scope = declaration.scope();
        let index_ty = ElemType::Index.to_type(scope.ctx());
        let f32_ty = ElemType::Float(FloatKind::F32).to_type(scope.ctx());
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
        let a = load(&body, declaration.column(self.a).unwrap(), index);
        let b = load(&body, declaration.column(self.b).unwrap(), index);
        let scale = declaration.parameter(self.scale).unwrap();
        let bias = declaration.parameter(self.bias).unwrap();
        let product = body.register_with_result(&FMulOp::new(body.ctx_mut(), a, scale));
        let sum = body.register_with_result(&FAddOp::new(body.ctx_mut(), product, b));
        let sum = body.register_with_result(&FAddOp::new(body.ctx_mut(), sum, bias));
        // Observe both the U32 scalar and buffer metadata, independently. A
        // capacity-derived length or wrong scalar ordinal changes the result.
        let count = body.register_with_result(&CastOp::new(
            body.ctx_mut(),
            f32_ty,
            declaration.row_count(),
        ));
        let length = body.register_with_result(&BufferLenOp::new(body.ctx_mut(), 0));
        let length = body.register_with_result(&CastOp::new(body.ctx_mut(), f32_ty, length));
        let sum = body.register_with_result(&FAddOp::new(body.ctx_mut(), sum, count));
        let sum = body.register_with_result(&FAddOp::new(body.ctx_mut(), sum, length));
        let ptr = body.register_with_result(&IndexOp::maybe_checked(
            body.ctx_mut(),
            declaration.output(),
            index,
            true,
        ));
        body.register(&StoreOp::new(body.ctx_mut(), ptr, sum));
        body.terminate_yield();
        let other =
            scope.branch_child(OpInserter::new_at_block_end(branch.else_block(scope.ctx())));
        other.terminate_yield();
        scope.register(&branch);
        scope.update_flags_after_branch(&[body, other]);
        declaration.build()
    }
}

fn upload(client: &Client, values: &[f32]) -> Handle {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    client.create_from_slice(&bytes)
}

fn import(handle: Handle, rows: u64) -> DeviceBuffer {
    // SAFETY: all fixtures originate from upload() of initialized F32 values.
    // Logical range, offsets, service identity and aliasing are checked at bind.
    unsafe { DeviceBuffer::from_raw_parts(handle, PhysicalType::F32, rows) }
}

fn exercise_abi(device: Device) {
    let client = device.client();
    let program = program();
    let layout = BindingLayout::new(&program);
    let declaration = layout.declare(&client, settings()).unwrap();
    let scope = declaration.scope();
    assert_eq!(
        declaration.row_count_raw().get_type(scope.ctx()),
        AddressType::U32.unsigned_type().to_type(scope.ctx())
    );
    assert_eq!(
        declaration.row_count().get_type(scope.ctx()),
        ElemType::Index.to_type(scope.ctx())
    );
    let info = declaration.build().info;
    assert_eq!(info.metadata.num_meta(), 3);
    assert_eq!(
        info.scalars
            .iter()
            .find(|field| field.ty == ElemType::Float(FloatKind::F32))
            .unwrap()
            .count,
        2
    );
    assert_eq!(
        info.scalars
            .iter()
            .find(|field| field.ty == AddressType::U32.unsigned_type())
            .unwrap()
            .count,
        1
    );
    assert!(info.scalars.iter().all(|field| field.offset % 8 == 0));

    let a = [1.0, 2.0, -3.0, 4.0, 5.0];
    let b = [10.0, 20.0, 30.0, 40.0, 50.0];
    let rows = a.len() as u64;
    let a_padded = [111.0, 1.0, 2.0, -3.0, 4.0, 5.0, 222.0, 333.0];
    let b_padded = [111.0, 10.0, 20.0, 30.0, 40.0, 50.0, 222.0, 333.0];
    let input_a = import(upload(&client, &a_padded).offset_start(4), rows);
    let input_b = import(upload(&client, &b_padded).offset_start(4), rows);
    let output = upload(&client, &[-999.0; 7]);
    // Layout order is b, a, output, despite the opposite expression order.
    let buffers = [input_b.clone(), input_a, import(output.clone(), rows)];
    let parameters = Parameters::new()
        .with("scale", 2.0_f32)
        .unwrap()
        .with("bias", 7.0_f32)
        .unwrap();

    assert!(matches!(
        layout.bind_device(&client, rows, &buffers, &Parameters::new()),
        Err(BindingError::Input(_))
    ));
    let aliased = [buffers[0].clone(), buffers[1].clone(), buffers[0].clone()];
    assert!(matches!(
        layout.bind_device(&client, rows, &aliased, &parameters),
        Err(BindingError::AliasedOutput { .. })
    ));
    let bad_settings = settings().address_type(AddressType::U64);
    assert!(matches!(
        layout.declare(&client, bad_settings.clone()),
        Err(BindingError::SettingsMismatch { .. })
    ));
    let bound = layout
        .bind_device(&client, rows, &buffers, &parameters)
        .unwrap();
    // SAFETY: the probe uses exactly layout.declare(), checks the logical bound,
    // and writes only the distinct output. Invalid settings fail before registration.
    assert!(matches!(
        unsafe {
            bound.launch(
                &client,
                bad_settings,
                ProbeKernel::new(&client, &program, false),
                CubeCount::Static(2, 1, 1),
            )
        },
        Err(BindingError::SettingsMismatch { .. })
    ));

    let batch = Batch::new(5)
        .with_column("a", ColumnData::F32(&a))
        .unwrap()
        .with_column("b", ColumnData::F32(&b))
        .unwrap();
    for (scale, bias) in [(2.0_f32, 7.0_f32), (-1.0, 3.0)] {
        let parameters = Parameters::new()
            .with("scale", scale)
            .unwrap()
            .with("bias", bias)
            .unwrap();
        let bound = layout
            .bind_device(&client, rows, &buffers, &parameters)
            .unwrap();
        let kernel = ProbeKernel::new(&client, &program, false);
        // SAFETY: same layout, scalar F32 buffers, bounded writes and no barriers.
        let submitted =
            unsafe { bound.launch(&client, settings(), kernel, CubeCount::Static(2, 1, 1)) }
                .unwrap();
        assert_eq!(submitted, Submission::Enqueued);
        let bytes = client
            .read_one(output.clone())
            .expect("probe must execute successfully");
        let actual: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_ne_bytes(*chunk))
            .collect();
        let expected: Vec<f32> = program
            .evaluate_reference(&batch, &parameters)
            .unwrap()
            .into_iter()
            .map(|value| value + rows as f32 + rows as f32)
            .collect();
        assert_eq!(&actual[..5], expected.as_slice());
        assert_eq!(&actual[5..], &[-999.0, -999.0]);
    }

    // An empty invocation validates arguments but must not register, compile,
    // dispatch a zero grid, or alter an existing output allocation.
    let empty_buffers = [
        import(upload(&client, &[1.0]), 0),
        import(upload(&client, &[2.0]), 0),
        import(output.clone(), 0),
    ];
    let before = client.read_one(output.clone()).unwrap();
    let bound = layout
        .bind_device(&client, 0, &empty_buffers, &parameters)
        .unwrap();
    // SAFETY: no execution is permitted for a zero-row invocation.
    assert_eq!(
        unsafe {
            bound.launch(
                &client,
                settings(),
                ProbeKernel::new(&client, &program, true),
                CubeCount::Static(0, 1, 1),
            )
        }
        .unwrap(),
        Submission::Empty
    );
    assert_eq!(&client.read_one(output).unwrap()[..], &before[..]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_binding_declaration_and_launch_agree() {
    exercise_abi(Device::cpu().expect("CPU runtime must be available"));
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_binding_declaration_and_launch_agree() {
    exercise_abi(Device::metal(Default::default()).expect("native Metal device must be available"));
}

#[cfg(all(feature = "cpu", feature = "metal", target_vendor = "apple"))]
#[test]
fn foreign_device_handles_are_rejected_before_registration() {
    let cpu = Device::cpu().unwrap().client();
    let metal = Device::metal(Default::default()).unwrap().client();
    let program = program();
    let layout = BindingLayout::new(&program);
    let buffers = [
        import(upload(&cpu, &[1.0]), 1),
        import(upload(&metal, &[2.0]), 1),
        import(upload(&metal, &[0.0]), 1),
    ];
    let parameters = Parameters::new()
        .with("scale", 2.0_f32)
        .unwrap()
        .with("bias", 1.0_f32)
        .unwrap();
    assert!(matches!(
        layout.bind_device(&metal, 1, &buffers, &parameters),
        Err(BindingError::ForeignDeviceBuffer { index: 0 })
    ));
}
