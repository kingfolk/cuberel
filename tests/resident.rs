//! Snapshot reuse, completion, identity and transfer accounting on real devices.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, Error, ExecutionError, Executor, Field, HostColumn, Parameters,
    Schema, abi::BindingError, col, param, prepare,
};

fn reuse(device: Device) {
    let client = device.client();
    let executor = Executor::new(client.clone()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("p", DataType::I32)],
    )
    .unwrap();
    let add = prepare(&(col("x") + param("p")), &schema).unwrap();
    let predicate = prepare(&col("x").ge(param("p")), &schema).unwrap();
    let id = executor.kernel_id(&add);
    let batch = {
        let values = vec![10, 0, 30, 40];
        let flags = vec![1, 0, 1, 1];
        executor
            .upload(
                &Batch::new(4)
                    .with_column(
                        "x",
                        ColumnData::NullableI32 {
                            values: &values,
                            validity: &flags,
                        },
                    )
                    .unwrap(),
            )
            .unwrap()
    };
    assert_eq!(batch.row_count(), 4);
    assert_eq!(
        batch
            .columns()
            .map(|f| (f.name(), f.data_type(), f.is_nullable()))
            .collect::<Vec<_>>(),
        [("x", DataType::I32, true)]
    );
    let cloned = batch.clone();
    assert!(cloned.same_source(&batch));
    let uploaded = executor.stats();
    assert_eq!((uploaded.upload_calls, uploaded.upload_bytes), (2, 32));
    assert_eq!(uploaded.readback_calls, 0);
    let p = Parameters::new().with("p", 2_i32).unwrap();
    let first = executor.execute_resident(&add, &batch, &p).unwrap();
    let second = executor
        .execute_resident(&add, &batch, &Parameters::new().with("p", 5_i32).unwrap())
        .unwrap();
    assert_eq!(executor.kernel_id(&add), id);
    let selection = executor
        .select_resident(
            &predicate,
            &batch,
            &Parameters::new().with("p", 30_i32).unwrap(),
        )
        .unwrap();
    assert!(selection.belongs_to(&batch));
    let other = Executor::with_cube_size(client, 1).unwrap();
    let cross_executor = other.execute_resident(&predicate, &batch, &p).unwrap();
    assert_eq!(executor.stats().upload_bytes, 32);
    assert_eq!(executor.stats().readback_calls, 0);
    assert_eq!(
        executor.stats().kernel_launches,
        3 + selection.stats().kernel_launches as u64
    );
    assert_eq!(executor.stats().count_readbacks, 0);
    drop(batch);
    drop(cloned);
    let first_clone = first.clone();
    drop(first);
    let result = std::thread::spawn(move || first_clone.materialize().unwrap())
        .join()
        .unwrap();
    let HostColumn::I32(result) = &result[0] else {
        panic!("expected I32")
    };
    assert_eq!(
        result.iter().collect::<Vec<_>>(),
        [Some(12), None, Some(32), Some(42)]
    );
    let HostColumn::I32(result) = &second.materialize().unwrap()[0] else {
        panic!("expected I32")
    };
    assert_eq!(
        result.iter().collect::<Vec<_>>(),
        [Some(15), None, Some(35), Some(45)]
    );
    assert_eq!(selection.materialize().unwrap(), [2, 3]);
    assert_eq!(executor.stats().count_readbacks, 1);
    let HostColumn::Bool(result) = &cross_executor.materialize().unwrap()[0] else {
        panic!("expected Bool")
    };
    assert_eq!(
        result.iter().collect::<Vec<_>>(),
        [Some(true), None, Some(true), Some(true)]
    );

    // A changed host allocation requires a fresh snapshot, even at the same address.
    let mut values = [1.0_f32, 2.0];
    let snapshot = executor
        .upload(
            &Batch::new(2)
                .with_column("f", ColumnData::F32(&values))
                .unwrap(),
        )
        .unwrap();
    values[0] = 9.0;
    let fresh = executor
        .upload(
            &Batch::new(2)
                .with_column("f", ColumnData::F32(&values))
                .unwrap(),
        )
        .unwrap();
    assert!(!snapshot.same_source(&fresh));
    let fs =
        Schema::from_fields(&[Field::new("f", DataType::F32).with_nullable(true)], &[]).unwrap();
    let program = prepare(&col("f"), &fs).unwrap();
    let old = executor
        .execute_resident(&program, &snapshot, &Parameters::new())
        .unwrap();
    old.complete().unwrap();
    let HostColumn::F32(result) = &old.materialize().unwrap()[0] else {
        panic!("expected F32")
    };
    assert_eq!(result.iter().collect::<Vec<_>>(), [Some(1.0), Some(2.0)]);
    let HostColumn::F32(result) = &executor
        .execute_resident(&program, &fresh, &Parameters::new())
        .unwrap()
        .materialize()
        .unwrap()[0]
    else {
        panic!("expected F32")
    };
    assert_eq!(result.values(), [9.0, 2.0]);
    assert!(result.validity().iter().all(|&v| v == 1));
}

fn errors_and_empty(device: Device) {
    let executor = Executor::new(device.client()).unwrap();
    let schema = Schema::new(
        &[("x", DataType::I32), ("missing", DataType::I32)],
        &[("p", DataType::I32)],
    )
    .unwrap();
    let add = prepare(&(col("x") + param("p")), &schema).unwrap();
    let identity = prepare(&col("x"), &schema).unwrap();
    let missing = prepare(&col("missing"), &schema).unwrap();
    let batch = executor
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1]))
                .unwrap(),
        )
        .unwrap();
    let params = Parameters::new().with("p", 1_i32).unwrap();
    let before = executor.stats();
    // Preflight all outputs, so missing later input never launches earlier output.
    assert!(matches!(
        executor.execute_many_resident(&[identity.clone(), missing], &batch, &params),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::MissingInput { .. }
        )))
    ));
    assert!(
        executor
            .execute_resident(&add, &batch, &Parameters::new())
            .is_err()
    );
    assert!(
        executor
            .execute_resident(&add, &batch, &Parameters::new().with("p", 1.0_f32).unwrap())
            .is_err()
    );
    assert_eq!(executor.stats(), before);
    let results = executor
        .execute_many_resident(&[identity.clone(), add.clone()], &batch, &params)
        .unwrap();
    assert_eq!(results.output_count(), 2);
    assert_eq!(results.output_type(0), Some(DataType::I32));
    assert_eq!(results.output_nullable(1), Some(false));
    assert!(matches!(
        results.complete(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    assert!(matches!(
        results.materialize(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    // Error plane only: no value/validity payload returned, even for first output.
    assert_eq!(executor.stats().readback_bytes - before.readback_bytes, 16);
    drop(
        executor
            .execute_resident(&identity, &batch, &params)
            .unwrap(),
    );
    let safe = executor
        .execute_resident(&add, &batch, &Parameters::new().with("p", 0_i32).unwrap())
        .unwrap();
    let HostColumn::I32(result) = &safe.materialize().unwrap()[0] else {
        panic!("expected I32")
    };
    assert_eq!(result.values(), [i32::MAX, 1]);
    let nulls = executor
        .upload(
            &Batch::new(2)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &[1, 2],
                        validity: &[1, 0],
                    },
                )
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        executor.execute_resident(&identity, &nulls, &params),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::UnexpectedNull { .. }
        )))
    ));
    let wrong = executor
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::F32(&[1.0, 2.0]))
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        executor.execute_resident(&identity, &wrong, &params),
        Err(ExecutionError::Binding(BindingError::Input(
            Error::TypeMismatch { .. }
        )))
    ));

    let before = executor.stats();
    let empty = executor
        .upload(
            &Batch::new(0)
                .with_column("x", ColumnData::I32(&[]))
                .unwrap(),
        )
        .unwrap();
    assert!(
        executor
            .execute_resident(&add, &empty, &Parameters::new())
            .is_err()
    );
    let result = executor.execute_resident(&add, &empty, &params).unwrap();
    result.complete().unwrap();
    let HostColumn::I32(result) = &result.materialize().unwrap()[0] else {
        panic!("expected I32")
    };
    assert!(result.is_empty());
    assert_eq!(executor.stats(), before);
    assert!(executor.upload(&Batch::new(u32::MAX as usize)).is_err());
    assert_eq!(executor.stats(), before);
    // Parameter-only programs use explicit row count without column allocations.
    let scalar = prepare(&param("p"), &schema).unwrap();
    let no_columns = executor.upload(&Batch::new(3)).unwrap();
    let HostColumn::I32(result) = &executor
        .execute_resident(&scalar, &no_columns, &params)
        .unwrap()
        .materialize()
        .unwrap()[0]
    else {
        panic!("expected I32")
    };
    assert_eq!(result.values(), [1, 1, 1]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_resident_reuse() {
    reuse(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_resident_errors_and_empty() {
    errors_and_empty(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_resident_reuse() {
    reuse(Device::metal(Default::default()).expect("native Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_resident_errors_and_empty() {
    errors_and_empty(Device::metal(Default::default()).expect("native Metal required"));
}
#[cfg(all(feature = "cpu", feature = "metal", target_vendor = "apple"))]
#[test]
fn foreign_resident_batch_rejected_even_when_empty() {
    let cpu = Executor::new(Device::cpu().unwrap().client()).unwrap();
    let metal = Executor::new(Device::metal(Default::default()).unwrap().client()).unwrap();
    let schema = Schema::new(&[], &[("p", DataType::I32)]).unwrap();
    let program = prepare(&param("p"), &schema).unwrap();
    let params = Parameters::new().with("p", 1_i32).unwrap();
    for n in [0, 3] {
        let batch = cpu.upload(&Batch::new(n)).unwrap();
        assert!(matches!(
            metal.execute_resident(&program, &batch, &params),
            Err(ExecutionError::ForeignDeviceBatch)
        ));
        assert!(matches!(
            metal.execute_many_resident(&[], &batch, &params),
            Err(ExecutionError::ForeignDeviceBatch)
        ));
    }
    assert_eq!(metal.stats(), Default::default());
}
