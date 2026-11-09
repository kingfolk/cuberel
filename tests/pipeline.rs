//! Fused and reference plans exercise actual selected runtimes, never fallback.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, ExecutionError, Executor, Field,
    FusionMode, HostColumn, Parameters, Pipeline, Schema, StageKind, col, param, prepare,
};
fn exercise(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32),
            Field::new("one", DataType::I32),
        ],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let add = prepare(&(col("x") + param("one")), &schema).unwrap();
    let pred = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let query = Pipeline::new()
        .filter(pred.clone())
        .aggregate("rows", Aggregate::CountStar)
        .aggregate("n", Aggregate::Count(add.clone()))
        .aggregate("sum", Aggregate::Sum(add.clone()))
        .aggregate("min", Aggregate::Min(x.clone()))
        .aggregate("max", Aggregate::Max(x.clone()));
    let params = Parameters::new()
        .with("limit", 0_i32)
        .unwrap()
        .with("one", 1_i32)
        .unwrap();
    for n in [
        0,
        1,
        e.cube_size() as usize - 1,
        e.cube_size() as usize,
        e.cube_size() as usize + 1,
        e.cube_size() as usize * e.cube_size() as usize + 1,
    ] {
        let input: Vec<_> = (0..n).map(|i| (i as i32 % 43) - 20).collect();
        let flags: Vec<_> = (0..n).map(|i| u32::from(i % 5 != 0)).collect();
        let batch = e
            .upload(
                &Batch::new(n)
                    .with_column(
                        "x",
                        ColumnData::NullableI32 {
                            values: &input,
                            validity: &flags,
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let mut results = Vec::new();
        for mode in [FusionMode::Unfused, FusionMode::Required] {
            let before = e.stats();
            let pending = e
                .execute_pipeline(&query.clone().fusion(mode), &batch, &params)
                .unwrap();
            assert_eq!(e.stats().readback_calls, before.readback_calls);
            assert_eq!(e.stats().upload_calls, before.upload_calls);
            assert_eq!(
                e.stats().kernel_launches - before.kernel_launches,
                pending.graph().kernel_launches() as u64
            );
            for (i, stage) in pending.graph().stages().iter().enumerate() {
                assert!(stage.dependencies.iter().all(|&d| d < i));
            }
            assert_eq!(
                pending.graph().is_fused(),
                mode == FusionMode::Required && n != 0
            );
            results.push(pending.materialize().unwrap());
        }
        assert_eq!(results[0], results[1]);
        let kept: Vec<_> = input
            .iter()
            .zip(&flags)
            .filter_map(|(&v, &f)| (f == 1 && v >= 0).then_some(v))
            .collect();
        assert_eq!(results[1].row_count, kept.len() as u32);
        assert_eq!(
            results[1].aggregates[2].value,
            AggregateValue::I64(
                (!kept.is_empty()).then(|| kept.iter().map(|&v| i64::from(v) + 1).sum())
            )
        );
    }
    // Shared selection feeds projection and aggregates, with one count read per exit.
    let batch = e
        .upload(
            &Batch::new(4)
                .with_column("x", ColumnData::I32(&[-1, 4, 7, 12]))
                .unwrap(),
        )
        .unwrap();
    let both = Pipeline::new()
        .filter(pred.clone())
        .project("plus", add.clone())
        .aggregate("sum", Aggregate::Sum(x.clone()));
    let before = e.stats();
    let pending = e.execute_pipeline(&both, &batch, &params).unwrap();
    assert_eq!(
        pending
            .graph()
            .stages()
            .iter()
            .filter(|s| s.kind == StageKind::Scatter)
            .count(),
        1
    );
    assert_eq!(e.stats().readback_calls, before.readback_calls);
    let result = pending.materialize().unwrap();
    assert_eq!(e.stats().count_readbacks - before.count_readbacks, 1);
    assert_eq!(result.aggregates[0].value, AggregateValue::I64(Some(23)));
    let HostColumn::I32(column) = result.projections[0].data() else {
        panic!("I32")
    };
    assert_eq!(column.values(), &[5, 8, 13]);
    // A second filter only sees parent's active original rows.
    let second = prepare(&col("x").lt(param("limit")), &schema).unwrap();
    let chain = Pipeline::new()
        .filter(pred.clone())
        .filter(second)
        .project("x", x.clone());
    let before = e.stats();
    let pending = e.execute_pipeline(&chain, &batch, &params).unwrap();
    assert_eq!(
        e.stats().kernel_launches - before.kernel_launches,
        pending.graph().kernel_launches() as u64
    );
    assert_eq!(pending.materialize().unwrap().row_count, 0);
    // Missing bindings/names/required unsupported patterns fail before launches.
    for invalid in [
        both.clone().project("sum", x.clone()),
        both.clone().fusion(FusionMode::Required),
        Pipeline::new(),
    ] {
        let before = e.stats();
        assert!(e.execute_pipeline(&invalid, &batch, &params).is_err());
        assert_eq!(e.stats(), before);
    }
    let before = e.stats();
    assert!(
        e.execute_pipeline(&query, &batch, &Parameters::new())
            .is_err()
    );
    assert_eq!(e.stats(), before);
    // Parameter changes reuse structural kernels; values do not enter the key.
    let only = Pipeline::new()
        .filter(pred)
        .aggregate("sum", Aggregate::Sum(x));
    let a = e.execute_pipeline(&only, &batch, &params).unwrap();
    assert_eq!(a.materialize().unwrap().row_count, 3);
    let definitions = e.define_count();
    let b = e
        .execute_pipeline(
            &only,
            &batch,
            &Parameters::new().with("limit", 10_i32).unwrap(),
        )
        .unwrap();
    assert_eq!(b.materialize().unwrap().row_count, 1);
    assert_eq!(e.define_count(), definitions);
}
fn errors(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::new(
        &[("x", DataType::I32), ("y", DataType::I32)],
        &[("limit", DataType::I32), ("one", DataType::I32)],
    )
    .unwrap();
    let plus = prepare(&(col("x") + param("one")), &schema).unwrap();
    let pred = prepare(&col("x").lt(param("limit")), &schema).unwrap();
    let params = Parameters::new()
        .with("limit", 10_i32)
        .unwrap()
        .with("one", 1_i32)
        .unwrap();
    let batch = e
        .upload(
            &Batch::new(3)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1, 2]))
                .unwrap()
                .with_column("y", ColumnData::I32(&[1, 1, 1]))
                .unwrap(),
        )
        .unwrap();
    let query = Pipeline::new()
        .filter(pred)
        .aggregate("sum", Aggregate::Sum(plus.clone()));
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        let output = e
            .execute_pipeline(&query.clone().fusion(mode), &batch, &params)
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(output.aggregates[0].value, AggregateValue::I64(Some(5)));
    }
    // Predicate error cannot be hidden by WHERE (or an argument error).
    let pred = prepare(&(col("x") + param("one")).lt(param("limit")), &schema).unwrap();
    let y = prepare(&(col("y") + param("one")), &schema).unwrap();
    let query = Pipeline::new()
        .filter(pred)
        .aggregate("sum", Aggregate::Sum(y.clone()));
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        assert!(matches!(
            e.execute_pipeline(&query.clone().fusion(mode), &batch, &params)
                .unwrap()
                .materialize(),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
    }
    // An earlier argument error never overrides a later predicate-stage error.
    let priority_batch = e
        .upload(
            &Batch::new(3)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1, 1]))
                .unwrap()
                .with_column("y", ColumnData::I32(&[1, 1, i32::MAX]))
                .unwrap(),
        )
        .unwrap();
    let priority_pred = prepare(&(col("y") + param("one")).lt(param("limit")), &schema).unwrap();
    let priority = Pipeline::new()
        .filter(priority_pred)
        .aggregate("sum", Aggregate::Sum(plus.clone()));
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        assert!(matches!(
            e.execute_pipeline(&priority.clone().fusion(mode), &priority_batch, &params)
                .unwrap()
                .materialize(),
            Err(ExecutionError::IntegerOverflow { row: 2 })
        ));
    }
    // Multiple argument branches report the earliest original row.
    let batch = e
        .upload(
            &Batch::new(4)
                .with_column("x", ColumnData::I32(&[1, 1, 1, i32::MAX]))
                .unwrap()
                .with_column("y", ColumnData::I32(&[1, i32::MAX, 1, 1]))
                .unwrap(),
        )
        .unwrap();
    let query = Pipeline::new()
        .aggregate("x", Aggregate::Sum(plus.clone()))
        .aggregate("y", Aggregate::Sum(y.clone()));
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        let pending = e
            .execute_pipeline(&query.clone().fusion(mode), &batch, &params)
            .unwrap();
        assert!(matches!(
            pending.materialize(),
            Err(ExecutionError::IntegerOverflow { row: 1 })
        ));
    }
    let both = Pipeline::new()
        .project("x", plus)
        .aggregate("y", Aggregate::Sum(y));
    assert!(matches!(
        e.execute_pipeline(&both, &batch, &params)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 1 })
    ));
}
fn floats(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("f", DataType::F32).with_nullable(true)],
        &[Field::new("limit", DataType::F32)],
    )
    .unwrap();
    let f = prepare(&col("f"), &schema).unwrap();
    let query = Pipeline::new()
        .aggregate("sum", Aggregate::Sum(f.clone()))
        .aggregate("min", Aggregate::Min(f.clone()))
        .aggregate("max", Aggregate::Max(f.clone()));
    let n = e.cube_size() as usize * e.cube_size() as usize + 1;
    let input: Vec<_> = (0..n)
        .map(|i| [1e20, -1e20, 1.0, -0.0, 0.0, 3.5][i % 6])
        .collect();
    let flags: Vec<_> = (0..n).map(|i| u32::from(i % 7 != 0)).collect();
    let batch = e
        .upload(
            &Batch::new(n)
                .with_column(
                    "f",
                    ColumnData::NullableF32 {
                        values: &input,
                        validity: &flags,
                    },
                )
                .unwrap(),
        )
        .unwrap();
    let mut expected = None;
    for mode in [FusionMode::Unfused, FusionMode::Required] {
        let out = e
            .execute_pipeline(&query.clone().fusion(mode), &batch, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap();
        let bits: Vec<_> = out
            .aggregates
            .iter()
            .map(|a| match a.value {
                AggregateValue::F32(v) => v.map(f32::to_bits),
                _ => panic!("F32"),
            })
            .collect();
        if let Some(expected) = &expected {
            assert_eq!(&bits, expected);
        } else {
            expected = Some(bits);
        }
    }
    let predicate = prepare(&col("f").ge(param("limit")), &schema).unwrap();
    let filtered = query.clone().filter(predicate);
    let params = Parameters::new().with("limit", 0.0_f32).unwrap();
    let auto = e.execute_pipeline(&filtered, &batch, &params).unwrap();
    assert!(!auto.graph().is_fused());
    assert!(auto.graph().fusion_reason().unwrap().contains("F32 SUM"));
    let expected = e
        .execute_pipeline(
            &filtered.clone().fusion(FusionMode::Unfused),
            &batch,
            &params,
        )
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(auto.materialize().unwrap(), expected);
    let before = e.stats();
    assert!(
        e.execute_pipeline(&filtered.fusion(FusionMode::Required), &batch, &params)
            .is_err()
    );
    assert_eq!(e.stats(), before);
    // Canonical NaN and signed-zero ordering also survive fused partials.
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("f", ColumnData::F32(&[f32::from_bits(0x7fa01234), -0.0]))
                .unwrap(),
        )
        .unwrap();
    let out = e
        .execute_pipeline(
            &query.fusion(FusionMode::Required),
            &batch,
            &Parameters::new(),
        )
        .unwrap()
        .materialize()
        .unwrap();
    for a in out.aggregates {
        let AggregateValue::F32(Some(v)) = a.value else {
            panic!("float")
        };
        assert_eq!(v.to_bits(), 0x7fc00000);
    }
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_pipeline() {
    exercise(Device::cpu().unwrap());
    errors(Device::cpu().unwrap());
    floats(Device::cpu().unwrap());
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_pipeline() {
    exercise(Device::metal(Default::default()).unwrap());
    errors(Device::metal(Default::default()).unwrap());
    floats(Device::metal(Default::default()).unwrap());
}
fn edge_cases(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema =
        Schema::from_fields(&[Field::new("x", DataType::I32).with_nullable(true)], &[]).unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let boolean = prepare(&col("x").is_null(), &schema).unwrap();
    for (values, flags) in [
        (&[i32::MIN, i32::MAX, -1, 2][..], &[1, 1, 1, 1][..]),
        (&[17, 29][..], &[0, 0][..]),
    ] {
        let batch = e
            .upload(
                &Batch::new(values.len())
                    .with_column(
                        "x",
                        ColumnData::NullableI32 {
                            values,
                            validity: flags,
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let query = Pipeline::new()
            .aggregate("s", Aggregate::Sum(x.clone()))
            .aggregate("c", Aggregate::Count(boolean.clone()));
        let fused = e
            .execute_pipeline(
                &query.clone().fusion(FusionMode::Required),
                &batch,
                &Parameters::new(),
            )
            .unwrap()
            .materialize()
            .unwrap();
        let reference = e
            .execute_pipeline(
                &query.fusion(FusionMode::Unfused),
                &batch,
                &Parameters::new(),
            )
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(fused, reference);
        let count = Pipeline::new()
            .filter(boolean.clone())
            .aggregate("rows", Aggregate::CountStar)
            .fusion(FusionMode::Required);
        let result = e
            .execute_pipeline(&count, &batch, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap();
        assert_eq!(
            result.row_count,
            flags.iter().filter(|&&v| v == 0).count() as u32
        );
    }
    // Fused inputs survive scope exit, cloning and completion on another thread.
    let pending = {
        let source = e
            .upload(
                &Batch::new(2)
                    .with_column("x", ColumnData::I32(&[i32::MAX, i32::MAX]))
                    .unwrap(),
            )
            .unwrap();
        let query = Pipeline::new()
            .aggregate("sum", Aggregate::Sum(x.clone()))
            .fusion(FusionMode::Required);
        e.execute_pipeline(&query, &source, &Parameters::new())
            .unwrap()
    };
    let clone = pending.clone();
    drop(pending);
    let output = std::thread::spawn(move || clone.materialize())
        .join()
        .unwrap()
        .unwrap();
    assert_eq!(
        output.aggregates[0].value,
        AggregateValue::I64(Some(2 * i64::from(i32::MAX)))
    );
    // Independently prepared schemas with different column IDs share by name.
    let ab = Schema::new(&[("a", DataType::I32), ("b", DataType::I32)], &[]).unwrap();
    let ba = Schema::new(&[("b", DataType::I32), ("a", DataType::I32)], &[]).unwrap();
    let a = prepare(&col("a"), &ab).unwrap();
    let b = prepare(&col("b"), &ba).unwrap();
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("a", ColumnData::I32(&[10, 20]))
                .unwrap()
                .with_column("b", ColumnData::I32(&[1, 2]))
                .unwrap(),
        )
        .unwrap();
    let query = Pipeline::new()
        .aggregate("a", Aggregate::Sum(a))
        .aggregate("b", Aggregate::Sum(b))
        .fusion(FusionMode::Required);
    let out = e
        .execute_pipeline(&query, &batch, &Parameters::new())
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(out.aggregates[0].value, AggregateValue::I64(Some(30)));
    assert_eq!(out.aggregates[1].value, AggregateValue::I64(Some(3)));
    // Valid reference requests can have differing declared nullability: fusion
    // declines the conflicting union, Auto executes the device reference path.
    let nn = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
    let xx = prepare(&col("x"), &nn).unwrap();
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[1, 2]))
                .unwrap(),
        )
        .unwrap();
    let query = Pipeline::new()
        .aggregate("n", Aggregate::Sum(xx))
        .aggregate("nullable", Aggregate::Sum(x));
    let pending = e
        .execute_pipeline(&query, &batch, &Parameters::new())
        .unwrap();
    assert!(!pending.graph().is_fused());
    assert!(
        pending
            .graph()
            .fusion_reason()
            .unwrap()
            .contains("consistent")
    );
    assert_eq!(
        pending.materialize().unwrap().aggregates[0].value,
        AggregateValue::I64(Some(3))
    );
    let before = e.stats();
    assert!(
        e.execute_pipeline(
            &query.fusion(FusionMode::Required),
            &batch,
            &Parameters::new()
        )
        .is_err()
    );
    assert_eq!(e.stats(), before);
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_pipeline_edges() {
    edge_cases(Device::cpu().unwrap());
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_pipeline_edges() {
    edge_cases(Device::metal(Default::default()).unwrap());
}
