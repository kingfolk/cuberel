//! Explicit scalar/batched host read scheduling over real backends.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use cubecl::Device;
use cuberel::{
    Aggregate, AggregateReadbackMode, AggregateResult, AggregateValue, Batch, ColumnData, DataType,
    ExecutionError, Executor, Field, FusionMode, Parameters, Pipeline, Schema, col, param, prepare,
};

fn requests(x: &cuberel::KernelProgram) -> [(&'static str, Aggregate); 5] {
    [
        ("COUNT*", Aggregate::CountStar),
        ("COUNT(x)", Aggregate::Count(x.clone())),
        ("SUM(x)", Aggregate::Sum(x.clone())),
        ("MIN(x)", Aggregate::Min(x.clone())),
        ("MAX(x)", Aggregate::Max(x.clone())),
    ]
}

fn i32_batch(executor: &Executor) -> cuberel::DeviceBatch {
    let mut values = vec![0_i32; 65];
    values[..4].copy_from_slice(&[i32::MAX, i32::MAX, i32::MIN, i32::MIN]);
    let mut validity = vec![0_u32; 65];
    validity[..4].fill(1);
    executor
        .upload(
            &Batch::new(65)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &values,
                        validity: &validity,
                    },
                )
                .unwrap(),
        )
        .unwrap()
}

fn schema() -> Schema {
    Schema::from_fields(&[Field::new("x", DataType::I32).with_nullable(true)], &[]).unwrap()
}

fn names(results: &[AggregateResult]) -> Vec<&str> {
    results.iter().map(|r| r.name.as_str()).collect()
}

fn integer_readback(device: Device) {
    let scalar = Executor::new(device.client()).unwrap();
    assert_eq!(
        scalar.aggregate_readback_mode(),
        AggregateReadbackMode::Scalar
    );
    let batch = i32_batch(&scalar);
    let x = prepare(&col("x"), &schema()).unwrap();
    let req = requests(&x);
    let expected = [
        AggregateValue::Count(65),
        AggregateValue::Count(4),
        AggregateValue::I64(Some(-2)),
        AggregateValue::I32(Some(i32::MIN)),
        AggregateValue::I32(Some(i32::MAX)),
    ];
    for (mode, calls) in [
        (AggregateReadbackMode::Scalar, 8),
        (AggregateReadbackMode::Batched, 2),
    ] {
        let executor = scalar.clone().with_aggregate_readback_mode(mode);
        assert_eq!(executor.aggregate_readback_mode(), mode);
        let pending = executor
            .aggregate_resident(&batch, &req, &Parameters::new())
            .unwrap();
        let before = executor.stats(); // Exclude the upload and submission.
        let before_buffers = executor.readback_buffers();
        let result = pending.materialize().unwrap();
        let after = executor.stats();
        assert_eq!(
            names(&result),
            ["COUNT*", "COUNT(x)", "SUM(x)", "MIN(x)", "MAX(x)"]
        );
        assert_eq!(
            result.iter().map(|r| r.value.clone()).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(after.readback_calls - before.readback_calls, calls);
        assert_eq!(after.readback_bytes - before.readback_bytes, 32);
        assert_eq!(executor.readback_buffers() - before_buffers, 8);
        assert_eq!(after.upload_calls, before.upload_calls);
        assert_eq!(after.upload_bytes, before.upload_bytes);
        assert_eq!(after.kernel_launches, before.kernel_launches);
        let repeated = pending.materialize().unwrap();
        assert_eq!(repeated, result);
        assert_eq!(
            executor.stats().readback_calls - after.readback_calls,
            calls
        );
        assert_eq!(executor.stats().readback_bytes - after.readback_bytes, 32);
    }

    // The pending object captures its mode; clone configuration affects only new submissions.
    let pending = scalar
        .aggregate_resident(&batch, &req, &Parameters::new())
        .unwrap();
    let batched_clone = scalar
        .clone()
        .with_aggregate_readback_mode(AggregateReadbackMode::Batched);
    let before = scalar.stats();
    pending.materialize().unwrap();
    let once = scalar.stats();
    assert_eq!(once.readback_calls - before.readback_calls, 8);
    pending.materialize().unwrap();
    let twice = scalar.stats();
    assert_eq!(twice.readback_calls - once.readback_calls, 8);
    assert_eq!(twice.readback_bytes - once.readback_bytes, 32);
    assert_eq!(
        batched_clone.aggregate_readback_mode(),
        AggregateReadbackMode::Batched
    );
    assert_eq!(scalar.kernel_id(&x), batched_clone.kernel_id(&x));
}

fn pipeline_readback(device: Device) {
    let base = Executor::new(device.client()).unwrap();
    let batch = i32_batch(&base);
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("zero", DataType::I32)],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let filter = prepare(&col("x").ge(param("zero")), &schema).unwrap();
    let query = Pipeline::new()
        .filter(filter.clone())
        .aggregate("COUNT*", Aggregate::CountStar)
        .aggregate("COUNT(x)", Aggregate::Count(x.clone()))
        .aggregate("SUM(x)", Aggregate::Sum(x.clone()))
        .aggregate("MIN(x)", Aggregate::Min(x.clone()))
        .aggregate("MAX(x)", Aggregate::Max(x.clone()))
        .fusion(FusionMode::Required);
    let params = Parameters::new().with("zero", 0_i32).unwrap();
    for (mode, calls) in [
        (AggregateReadbackMode::Scalar, 9),
        (AggregateReadbackMode::Batched, 3),
    ] {
        let executor = base.clone().with_aggregate_readback_mode(mode);
        let pending = executor.execute_pipeline(&query, &batch, &params).unwrap();
        assert!(pending.graph().is_fused());
        let before = executor.stats();
        let before_buffers = executor.readback_buffers();
        let result = pending.materialize().unwrap();
        let after = executor.stats();
        assert_eq!(result.row_count, 2);
        assert_eq!(
            names(&result.aggregates),
            ["COUNT*", "COUNT(x)", "SUM(x)", "MIN(x)", "MAX(x)"]
        );
        assert_eq!(
            result
                .aggregates
                .iter()
                .map(|r| r.value.clone())
                .collect::<Vec<_>>(),
            [
                AggregateValue::Count(2),
                AggregateValue::Count(2),
                AggregateValue::I64(Some(4_294_967_294)),
                AggregateValue::I32(Some(i32::MAX)),
                AggregateValue::I32(Some(i32::MAX)),
            ]
        );
        assert_eq!(after.readback_calls - before.readback_calls, calls);
        assert_eq!(after.readback_bytes - before.readback_bytes, 36);
        assert_eq!(executor.readback_buffers() - before_buffers, 9);
        assert_eq!(after.upload_calls, before.upload_calls);
        assert_eq!(after.upload_bytes, before.upload_bytes);
        assert_eq!(after.kernel_launches, before.kernel_launches);
    }

    for (mode, calls) in [
        (AggregateReadbackMode::Scalar, 9),
        (AggregateReadbackMode::Batched, 3),
    ] {
        let executor = base.clone().with_aggregate_readback_mode(mode);
        let selected = executor.select_resident(&filter, &batch, &params).unwrap();
        let pending = executor
            .aggregate_selected(&batch, &selected, &requests(&x), &Parameters::new())
            .unwrap();
        let before = executor.stats();
        let before_buffers = executor.readback_buffers();
        let result = pending.materialize().unwrap();
        let after = executor.stats();
        assert_eq!(
            result.iter().map(|r| r.value.clone()).collect::<Vec<_>>(),
            [
                AggregateValue::Count(2),
                AggregateValue::Count(2),
                AggregateValue::I64(Some(4_294_967_294)),
                AggregateValue::I32(Some(i32::MAX)),
                AggregateValue::I32(Some(i32::MAX)),
            ]
        );
        assert_eq!(after.readback_calls - before.readback_calls, calls);
        assert_eq!(after.readback_bytes - before.readback_bytes, 36);
        assert_eq!(executor.readback_buffers() - before_buffers, 9);
    }
}

fn empty_and_null(device: Device) {
    let base = Executor::new(device.client()).unwrap();
    let x = prepare(&col("x"), &schema()).unwrap();
    let req = requests(&x);
    let values = vec![0_i32; 65];
    let validity = vec![0_u32; 65];
    let null_batch = base
        .upload(
            &Batch::new(65)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &values,
                        validity: &validity,
                    },
                )
                .unwrap(),
        )
        .unwrap();
    let empty = base
        .upload(
            &Batch::new(0)
                .with_column("x", ColumnData::I32(&[]))
                .unwrap(),
        )
        .unwrap();
    for (mode, calls) in [
        (AggregateReadbackMode::Scalar, 4),
        (AggregateReadbackMode::Batched, 1),
    ] {
        let executor = base.clone().with_aggregate_readback_mode(mode);
        let pending = executor
            .aggregate_resident(&null_batch, &req, &Parameters::new())
            .unwrap();
        let before = executor.stats();
        let before_buffers = executor.readback_buffers();
        let result = pending.materialize().unwrap();
        let after = executor.stats();
        assert_eq!(
            result.iter().map(|r| r.value.clone()).collect::<Vec<_>>(),
            [
                AggregateValue::Count(65),
                AggregateValue::Count(0),
                AggregateValue::I64(None),
                AggregateValue::I32(None),
                AggregateValue::I32(None),
            ]
        );
        assert_eq!(after.readback_calls - before.readback_calls, calls);
        assert_eq!(after.readback_bytes - before.readback_bytes, 16);
        assert_eq!(executor.readback_buffers() - before_buffers, 4);

        let pending = executor
            .aggregate_resident(&empty, &req, &Parameters::new())
            .unwrap();
        let before = executor.stats();
        let before_buffers = executor.readback_buffers();
        let result = pending.materialize().unwrap();
        let after = executor.stats();
        assert_eq!(result[0].value, AggregateValue::Count(0));
        assert_eq!(after.readback_calls, before.readback_calls);
        assert_eq!(after.readback_bytes, before.readback_bytes);
        assert_eq!(executor.readback_buffers(), before_buffers);
    }
}

fn floats(device: Device) {
    let base = Executor::new(device.client()).unwrap();
    let schema =
        Schema::from_fields(&[Field::new("f", DataType::F32).with_nullable(false)], &[]).unwrap();
    let f = prepare(&col("f"), &schema).unwrap();
    let req = [
        ("n", Aggregate::Count(f.clone())),
        ("sum", Aggregate::Sum(f.clone())),
        ("min", Aggregate::Min(f.clone())),
        ("max", Aggregate::Max(f)),
    ];
    let finite = base
        .upload(
            &Batch::new(3)
                .with_column("f", ColumnData::F32(&[1.0e20, -1.0e20, 3.0]))
                .unwrap(),
        )
        .unwrap();
    let exceptional = base
        .upload(
            &Batch::new(3)
                .with_column(
                    "f",
                    ColumnData::F32(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]),
                )
                .unwrap(),
        )
        .unwrap();
    let mut exceptional_sum_bits = None;
    for mode in [
        AggregateReadbackMode::Scalar,
        AggregateReadbackMode::Batched,
    ] {
        let executor = base.clone().with_aggregate_readback_mode(mode);
        let result = executor
            .aggregate_resident(&finite, &req, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(result[0].value, AggregateValue::Count(3));
        assert_eq!(result[1].value, AggregateValue::F32(Some(3.0)));
        assert_eq!(result[2].value, AggregateValue::F32(Some(-1.0e20)));
        assert_eq!(result[3].value, AggregateValue::F32(Some(1.0e20)));
        let result = executor
            .aggregate_resident(&exceptional, &req, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(result[0].value, AggregateValue::Count(3));
        let bits = |index: usize| match &result[index].value {
            AggregateValue::F32(Some(value)) => value.to_bits(),
            _ => panic!("expected a present F32 result"),
        };
        assert_eq!(bits(2), 0x7fc0_0000); // NaN propagates through extrema.
        assert_eq!(bits(3), 0x7fc0_0000);
        let AggregateValue::F32(Some(sum)) = &result[1].value else {
            panic!("valid NaN sum")
        };
        assert!(sum.is_nan());
        let sum_bits = sum.to_bits();
        assert_eq!(sum_bits, 0x7fc0_0000);
        if let Some(previous) = exceptional_sum_bits {
            assert_eq!(sum_bits, previous);
        }
        exceptional_sum_bits = Some(sum_bits);
    }
}

fn errors_and_filter(device: Device) {
    let base = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32)],
        &[
            Field::new("one", DataType::I32),
            Field::new("zero", DataType::I32),
        ],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let plus = prepare(&(col("x") + param("one")), &schema).unwrap();
    let pred = prepare(&col("x").lt(param("zero")), &schema).unwrap();
    let batch = base
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1]))
                .unwrap(),
        )
        .unwrap();
    for mode in [
        AggregateReadbackMode::Scalar,
        AggregateReadbackMode::Batched,
    ] {
        let executor = base.clone().with_aggregate_readback_mode(mode);
        let params = Parameters::new()
            .with("one", 1_i32)
            .unwrap()
            .with("zero", 0_i32)
            .unwrap();
        assert!(matches!(
            executor
                .aggregate_resident(&batch, &[("sum", Aggregate::Sum(plus.clone()))], &params)
                .unwrap()
                .materialize(),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
        let good = executor
            .aggregate_resident(&batch, &[("sum", Aggregate::Sum(x.clone()))], &params)
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(
            good[0].value,
            AggregateValue::I64(Some(i64::from(i32::MAX) + 1))
        );
        let query = Pipeline::new()
            .filter(pred.clone())
            .aggregate("sum", Aggregate::Sum(plus.clone()))
            .fusion(FusionMode::Required);
        let empty = executor
            .execute_pipeline(&query, &batch, &params)
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(empty.row_count, 0);
        assert_eq!(empty.aggregates[0].value, AggregateValue::I64(None));
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_aggregate_readback_modes() {
    let device = Device::cpu().expect("CPU runtime required");
    integer_readback(device.clone());
    pipeline_readback(device.clone());
    empty_and_null(device.clone());
    floats(device.clone());
    errors_and_filter(device);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_aggregate_readback_modes() {
    let device = Device::metal(Default::default()).expect("native Metal runtime required");
    integer_readback(device.clone());
    pipeline_readback(device.clone());
    empty_and_null(device.clone());
    floats(device.clone());
    errors_and_filter(device);
}
