#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Error, ExecutionError, Executor, Expr, ExprId, ExprNode,
    InputKind, Parameters, Schema, abi::BindingError, col, param, prepare,
};

fn schema() -> Schema {
    Schema::new(
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
    .unwrap()
}

fn params(scale: f32, bias: f32) -> Parameters {
    Parameters::new()
        .with("scale", scale)
        .unwrap()
        .with("bias", bias)
        .unwrap()
}

fn batch<'a>(a: &'a [f32], b: &'a [f32]) -> Batch<'a> {
    Batch::new(a.len())
        .with_column("a", ColumnData::F32(a))
        .unwrap()
        .with_column("b", ColumnData::F32(b))
        .unwrap()
}

fn exercise(device: Device) {
    let client = device.client();
    println!("runtime={:?} device={}", device.runtime(), client.name());
    assert!(Executor::with_cube_size(client.clone(), 0).is_err());
    assert!(Executor::with_cube_size(client.clone(), u32::MAX).is_err());
    let executor = Executor::new(client).unwrap();
    let schema = schema();
    let combined = prepare(
        &(col("a") * param("scale") + col("b") + param("bias")),
        &schema,
    )
    .unwrap();
    let size = executor.cube_size() as usize;
    // Logical row guards cover the last partial cube and multiple complete cubes.
    for rows in [0, 1, size - 1, size, size + 1, size * 3, size * 3 + 7, 4099] {
        let a: Vec<f32> = (0..rows)
            .map(|i| (i as i32 % 101 - 50) as f32 / 8.0)
            .collect();
        let b: Vec<f32> = (0..rows).map(|i| (i % 13) as f32 / 4.0).collect();
        for (scale, bias) in [(2.0_f32, 1.0_f32), (-1.0, 3.0)] {
            let expected: Vec<f32> = a
                .iter()
                .zip(&b)
                .map(|(a, b)| a * scale + b + bias)
                .collect();
            assert_eq!(
                executor
                    .execute(&combined, &batch(&a, &b), &params(scale, bias))
                    .unwrap(),
                expected
            );
        }
    }

    let a = [1.0, 2.0, -3.0];
    let b = [10.0, 20.0, 30.0];
    let input = batch(&a, &b);
    // Same buffer/scalar ABI and Rust kernel type, different runtime expression.
    let add = prepare(&(col("a") + param("scale")), &schema).unwrap();
    let mul = prepare(&(col("a") * param("scale")), &schema).unwrap();
    for _ in 0..3 {
        assert_eq!(
            executor.execute(&add, &input, &params(2.0, 1.0)).unwrap(),
            [3.0, 4.0, -1.0]
        );
        assert_eq!(
            executor.execute(&mul, &input, &params(2.0, 1.0)).unwrap(),
            [2.0, 4.0, -6.0]
        );
        assert_eq!(
            executor
                .execute(&combined, &input, &params(2.0, 1.0))
                .unwrap(),
            [13.0, 25.0, 25.0]
        );
    }
    assert_eq!(
        executor.execute(&mul, &input, &params(-1.0, 0.0)).unwrap(),
        [-1.0, -2.0, 3.0]
    );

    let copy = prepare(&col("a"), &schema).unwrap();
    // Ordinary multiplication retains negative zero on both selected backends.
    let zeros = executor
        .execute(&mul, &batch(&[-0.0, 0.0], &[0.0, 0.0]), &params(2.0, 0.0))
        .unwrap();
    assert_eq!(
        zeros.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        [(-0.0_f32).to_bits(), 0.0_f32.to_bits()]
    );
    // Unused schema fields/parameters need no data; unused supplied I32 is allowed.
    let input = Batch::new(3)
        .with_column("extra", ColumnData::I32(&[1, 2, 3]))
        .unwrap()
        .with_column("a", ColumnData::F32(&a))
        .unwrap();
    assert_eq!(
        executor.execute(&copy, &input, &Parameters::new()).unwrap(),
        a
    );
    let broadcast = prepare(&(param("scale") * param("scale")), &schema).unwrap();
    let too_many_rows = (u64::from(u32::MAX) / 4 + 1) as usize;
    assert!(matches!(
        executor.execute(&broadcast, &Batch::new(too_many_rows), &params(3.0, 0.0)),
        Err(ExecutionError::Binding(
            BindingError::AddressOverflow { .. }
        ))
    ));
    assert_eq!(
        executor
            .execute(&broadcast, &Batch::new(4), &params(3.0, 0.0))
            .unwrap(),
        [9.0; 4]
    );

    let dag = Expr::from_nodes(
        vec![
            ExprNode::Column("a".into()),
            ExprNode::Parameter("scale".into()),
            ExprNode::Mul(ExprId::new(0), ExprId::new(1)),
            ExprNode::Add(ExprId::new(2), ExprId::new(2)),
            // Root need not be the last node; all declared inputs still validate.
            ExprNode::Parameter("bias".into()),
        ],
        ExprId::new(3),
    );
    let dag = prepare(&dag, &schema).unwrap();
    assert_eq!(
        executor.execute(&dag, &input, &params(4.0, 999.0)).unwrap(),
        [8.0, 16.0, -24.0]
    );

    // Arithmetic grouping survives IR lowering. Reassociation changes these results.
    let left = prepare(&((col("a") + param("bias")) + col("b")), &schema).unwrap();
    let right = prepare(&(col("a") + (param("bias") + col("b"))), &schema).unwrap();
    let input = batch(&[16_777_216.0], &[-16_777_216.0]);
    assert_eq!(
        executor.execute(&left, &input, &params(0.0, 1.0)).unwrap(),
        [0.0]
    );
    assert_eq!(
        executor.execute(&right, &input, &params(0.0, 1.0)).unwrap(),
        [1.0]
    );
    // A fused multiply-add would produce 1 rather than 0 for this exact fixture.
    let no_fma = prepare(&(col("a") * param("scale") + param("bias")), &schema).unwrap();
    assert_eq!(
        executor
            .execute(
                &no_fma,
                &batch(&[4097.0], &[0.0]),
                &params(4097.0, -16_785_408.0)
            )
            .unwrap(),
        [0.0]
    );

    // Non-binary-exact finite values use an explicit absolute/relative tolerance.
    let a: Vec<f32> = (0..257).map(|i| (i as f32 - 128.0) / 7.0).collect();
    let b: Vec<f32> = (0..257).map(|i| (i % 13) as f32 / 3.0).collect();
    let expression = prepare(
        &((col("a") * param("scale") + col("b")) * (param("scale") + param("bias"))),
        &schema,
    )
    .unwrap();
    let actual = executor
        .execute(&expression, &batch(&a, &b), &params(1.3, -0.2))
        .unwrap();
    for ((actual, a), b) in actual.iter().zip(&a).zip(&b) {
        let expected = (a * 1.3 + b) * (1.3 - 0.2);
        assert!((actual - expected).abs() <= 1e-6 + 1e-5 * expected.abs());
    }

    // Validation still runs for empty input, before any upload or kernel launch.
    let empty = batch(&[], &[]);
    assert!(matches!(
        executor.execute(&combined, &empty, &Parameters::new()),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::MissingInput {
                kind: InputKind::Parameter,
                ..
            }
        )))
    ));
    assert!(matches!(
        executor.execute(&copy, &Batch::new(1), &Parameters::new()),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::MissingInput {
                kind: InputKind::Column,
                ..
            }
        )))
    ));
    let wrong = Batch::new(1)
        .with_column("a", ColumnData::I32(&[1]))
        .unwrap();
    assert!(matches!(
        executor.execute(&copy, &wrong, &Parameters::new()),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::TypeMismatch { .. }
        )))
    ));
    let wrong = Parameters::new().with("scale", 3_i32).unwrap();
    assert!(matches!(
        executor.execute(&broadcast, &Batch::new(0), &wrong),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::TypeMismatch { .. }
        )))
    ));
    // A rejected invocation cannot poison launcher state for the next valid call.
    assert_eq!(
        executor
            .execute(&copy, &batch(&[7.0], &[0.0]), &Parameters::new())
            .unwrap(),
        [7.0]
    );

    let schema = Schema::new(&[("a", DataType::F32)], &[("a", DataType::F32)]).unwrap();
    let program = prepare(&(col("a") + param("a")), &schema).unwrap();
    let input = Batch::new(1)
        .with_column("a", ColumnData::F32(&[3.0]))
        .unwrap();
    assert_eq!(
        executor
            .execute(
                &program,
                &input,
                &Parameters::new().with("a", 7.0_f32).unwrap()
            )
            .unwrap(),
        [10.0]
    );
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime_expression_conformance() {
    exercise(Device::cpu().expect("real CPU runtime required"));
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_runtime_expression_conformance() {
    exercise(Device::metal(Default::default()).expect("real native Metal device required"));
}
