//! Real backend aggregation with fixed expected scalars and Rust slice primitives.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, ExecutionError, Executor, Field,
    Parameters, Schema, col, param, prepare,
};
fn values(result: &cuberel::DeviceAggregates) -> Vec<AggregateValue> {
    result
        .materialize()
        .unwrap()
        .into_iter()
        .map(|v| v.value)
        .collect()
}
fn integers(device: Device) {
    let client = device.client();
    let e = Executor::new(client.clone()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let predicate = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let requests = [
        ("rows", Aggregate::CountStar),
        ("valid", Aggregate::Count(x.clone())),
        ("min", Aggregate::Min(x.clone())),
        ("max", Aggregate::Max(x.clone())),
    ];
    let batch = {
        let x = [30, 0, -5, 0, i32::MAX, i32::MIN];
        let validity = [1, 0, 1, 1, 1, 1];
        e.upload(
            &Batch::new(6)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &x,
                        validity: &validity,
                    },
                )
                .unwrap(),
        )
        .unwrap()
    };
    let out = e
        .aggregate_resident(&batch, &requests, &Parameters::new())
        .unwrap();
    assert_eq!(e.stats().readback_calls, 0);
    assert_eq!(
        values(&out),
        [
            AggregateValue::Count(6),
            AggregateValue::Count(5),
            AggregateValue::I32(Some(i32::MIN)),
            AggregateValue::I32(Some(i32::MAX))
        ]
    );
    assert_eq!(
        out.materialize()
            .unwrap()
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>(),
        ["rows", "valid", "min", "max"]
    );
    let p = Parameters::new().with("limit", 0_i32).unwrap();
    let selected = e.select_resident(&predicate, &batch, &p).unwrap();
    let before = e.stats();
    let out = e
        .aggregate_selected(&batch, &selected, &requests, &Parameters::new())
        .unwrap();
    assert_eq!(e.stats().readback_calls, before.readback_calls);
    assert_eq!(e.stats().upload_bytes, before.upload_bytes);
    assert_eq!(
        values(&out),
        [
            AggregateValue::Count(3),
            AggregateValue::Count(3),
            AggregateValue::I32(Some(0)),
            AggregateValue::I32(Some(i32::MAX))
        ]
    );
    let nulls = prepare(&col("x").is_null(), &schema).unwrap();
    let selected = e
        .select_resident(&nulls, &batch, &Parameters::new())
        .unwrap();
    assert_eq!(
        values(
            &e.aggregate_selected(&batch, &selected, &requests, &Parameters::new())
                .unwrap()
        ),
        [
            AggregateValue::Count(1),
            AggregateValue::Count(0),
            AggregateValue::I32(None),
            AggregateValue::I32(None)
        ]
    );
    let none = prepare(&col("x").gt(param("limit")), &schema).unwrap();
    let selected = e
        .select_resident(
            &none,
            &batch,
            &Parameters::new().with("limit", i32::MAX).unwrap(),
        )
        .unwrap();
    assert_eq!(
        values(
            &e.aggregate_selected(&batch, &selected, &requests, &Parameters::new())
                .unwrap()
        ),
        [
            AggregateValue::Count(0),
            AggregateValue::Count(0),
            AggregateValue::I32(None),
            AggregateValue::I32(None)
        ]
    );
    let empty = e
        .upload(
            &Batch::new(0)
                .with_column("x", ColumnData::I32(&[]))
                .unwrap(),
        )
        .unwrap();
    let before = e.stats();
    assert_eq!(
        values(
            &e.aggregate_resident(&empty, &requests, &Parameters::new())
                .unwrap()
        ),
        [
            AggregateValue::Count(0),
            AggregateValue::Count(0),
            AggregateValue::I32(None),
            AggregateValue::I32(None)
        ]
    );
    assert_eq!(e.stats(), before);
    // Recursive merging, validity counts >1, signed values and a padded tail.
    let n = e.cube_size() as usize * e.cube_size() as usize + 1;
    let input: Vec<_> = (0..n).map(|i| (i * 73 % 997) as i32 - 498).collect();
    let valid: Vec<_> = (0..n).map(|i| u32::from(i % 5 != 0)).collect();
    let expected: Vec<_> = input
        .iter()
        .zip(&valid)
        .filter_map(|(&v, &valid)| (valid == 1).then_some(v))
        .collect();
    let batch = e
        .upload(
            &Batch::new(n)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &input,
                        validity: &valid,
                    },
                )
                .unwrap(),
        )
        .unwrap();
    let out = e
        .aggregate_resident(&batch, &requests, &Parameters::new())
        .unwrap();
    assert!(out.stats().reduction_levels >= 3);
    assert_eq!(
        out.stats().reduction_launches,
        out.stats().reduction_levels * 3
    );
    assert_eq!(
        values(&out),
        [
            AggregateValue::Count(n as u64),
            AggregateValue::Count(expected.len() as u64),
            AggregateValue::I32(expected.iter().copied().min()),
            AggregateValue::I32(expected.iter().copied().max())
        ]
    );
    let odd_values: Vec<i32> = (0..29).map(|i| i * 7 - 50).collect();
    let odd = Executor::with_cube_size(client, 3).unwrap();
    let batch = odd
        .upload(
            &Batch::new(29)
                .with_column("x", ColumnData::I32(&odd_values))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        values(
            &odd.aggregate_resident(&batch, &requests, &Parameters::new())
                .unwrap()
        ),
        [
            AggregateValue::Count(29),
            AggregateValue::Count(29),
            AggregateValue::I32(odd_values.iter().copied().min()),
            AggregateValue::I32(odd_values.iter().copied().max())
        ]
    );
}
// Input, validity, COUNT, and exact MIN/MAX result bits.
type FloatCase<'a> = (&'a [f32], &'a [u32], u64, Option<u32>, Option<u32>);
fn floats(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema =
        Schema::from_fields(&[Field::new("f", DataType::F32).with_nullable(true)], &[]).unwrap();
    let f = prepare(&col("f"), &schema).unwrap();
    let requests = [
        ("count", Aggregate::Count(f.clone())),
        ("min", Aggregate::Min(f.clone())),
        ("max", Aggregate::Max(f)),
    ];
    // Explicit bit expectations: signed zero, canonical NaN, infinities, true
    // numeric zero versus no valid values, and a single valid NaN without merge.
    let cases: &[FloatCase<'_>] = &[
        (
            &[-7.5, 2.25],
            &[1, 1],
            2,
            Some((-7.5_f32).to_bits()),
            Some(2.25_f32.to_bits()),
        ),
        (&[-0.0], &[1], 1, Some(0x8000_0000), Some(0x8000_0000)),
        (
            &[f32::from_bits(1), f32::from_bits(2)],
            &[1, 1],
            2,
            Some(1),
            Some(2),
        ),
        (
            &[f32::from_bits(0x8000_0001), f32::from_bits(1)],
            &[1, 1],
            2,
            Some(0x8000_0001),
            Some(1),
        ),
        (&[0.0, -0.0], &[1, 1], 2, Some(0x8000_0000), Some(0)),
        (
            &[f32::INFINITY, -2.0, f32::NEG_INFINITY],
            &[1, 1, 1],
            3,
            Some(0xff80_0000),
            Some(0x7f80_0000),
        ),
        (
            &[1.0, f32::from_bits(0xffa0_0017), 9.0],
            &[1, 1, 1],
            3,
            Some(0x7fc0_0000),
            Some(0x7fc0_0000),
        ),
        (
            &[f32::from_bits(0x7f80_0001)],
            &[1],
            1,
            Some(0x7fc0_0000),
            Some(0x7fc0_0000),
        ),
        (&[f32::NAN, 0.0], &[0, 1], 1, Some(0), Some(0)),
        (&[9.0, -7.0], &[0, 0], 0, None, None),
    ];
    for &(input, flags, count, min, max) in cases {
        let batch = e
            .upload(
                &Batch::new(input.len())
                    .with_column(
                        "f",
                        ColumnData::NullableF32 {
                            values: input,
                            validity: flags,
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let out = e
            .aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap();
        for _ in 0..2 {
            let actual = values(&out);
            assert_eq!(actual[0], AggregateValue::Count(count));
            let AggregateValue::F32(a) = actual[1] else {
                panic!("F32 expected")
            };
            let AggregateValue::F32(b) = actual[2] else {
                panic!("F32 expected")
            };
            assert_eq!(a.map(f32::to_bits), min);
            assert_eq!(b.map(f32::to_bits), max);
        }
    }
    // NaN and both zero signs survive more than one partial merge level.
    let n = e.cube_size() as usize * e.cube_size() as usize + 1;
    let mut input = vec![1.0; n];
    input[n - 1] = f32::from_bits(0xffc0_0033);
    let batch = e
        .upload(
            &Batch::new(n)
                .with_column("f", ColumnData::F32(&input))
                .unwrap(),
        )
        .unwrap();
    let actual = values(
        &e.aggregate_resident(&batch, &requests, &Parameters::new())
            .unwrap(),
    );
    for value in &actual[1..] {
        let AggregateValue::F32(Some(v)) = value else {
            panic!("NaN expected")
        };
        assert_eq!(v.to_bits(), 0x7fc0_0000);
    }
}
fn errors(device: Device) {
    let client = device.client();
    let e = Executor::new(client.clone()).unwrap();
    let schema = Schema::new(
        &[("x", DataType::I32)],
        &[("one", DataType::I32), ("limit", DataType::I32)],
    )
    .unwrap();
    let p = Parameters::new()
        .with("one", 1_i32)
        .unwrap()
        .with("limit", 10_i32)
        .unwrap();
    let add = prepare(&(col("x") + param("one")), &schema).unwrap();
    let predicate = prepare(&col("x").lt(param("limit")), &schema).unwrap();
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1]))
                .unwrap(),
        )
        .unwrap();
    let count = [("count", Aggregate::Count(add.clone()))];
    let out = e.aggregate_resident(&batch, &count, &p).unwrap();
    assert!(matches!(
        out.materialize(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    let selected = e.select_resident(&predicate, &batch, &p).unwrap();
    let pending = e.aggregate_selected(&batch, &selected, &count, &p).unwrap();
    let clone = pending.clone();
    drop(pending);
    assert_eq!(
        std::thread::spawn(move || values(&clone)).join().unwrap(),
        [AggregateValue::Count(1)]
    );
    let failing = prepare(&(col("x") + param("one")).gt(param("limit")), &schema).unwrap();
    let failed = e.select_resident(&failing, &batch, &p).unwrap();
    assert!(matches!(
        e.aggregate_selected(&batch, &failed, &[("rows", Aggregate::CountStar)], &p)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    let before = e.stats();
    assert!(e.aggregate_resident(&batch, &[], &p).is_err());
    assert!(
        e.aggregate_resident(&batch, &[("", Aggregate::CountStar)], &p)
            .is_err()
    );
    assert!(
        e.aggregate_resident(
            &batch,
            &[("a", Aggregate::CountStar), ("a", Aggregate::CountStar)],
            &p
        )
        .is_err()
    );
    assert!(
        e.aggregate_resident(
            &batch,
            &[
                ("a", Aggregate::CountStar),
                ("bad", Aggregate::Min(predicate.clone()))
            ],
            &p
        )
        .is_err()
    );
    assert!(
        e.aggregate_resident(&batch, &count, &Parameters::new())
            .is_err()
    );
    assert_eq!(e.stats(), before);
    {
        let schema = Schema::new(
            &[("a", DataType::I32), ("b", DataType::I32)],
            &[("one", DataType::I32), ("zero", DataType::I32)],
        )
        .unwrap();
        let late = prepare(&(col("a") + param("one")), &schema).unwrap();
        let early = prepare(&(col("b") + param("one")), &schema).unwrap();
        let keep = prepare(&col("a").ge(param("zero")), &schema).unwrap();
        let input = e
            .upload(
                &Batch::new(4)
                    .with_column("a", ColumnData::I32(&[-1, 0, -1, i32::MAX]))
                    .unwrap()
                    .with_column("b", ColumnData::I32(&[0, i32::MAX, 0, 0]))
                    .unwrap(),
            )
            .unwrap();
        let p = Parameters::new()
            .with("one", 1_i32)
            .unwrap()
            .with("zero", 0_i32)
            .unwrap();
        let requests = [
            ("late", Aggregate::Count(late)),
            ("early", Aggregate::Min(early)),
        ];
        let selected = e.select_resident(&keep, &input, &p).unwrap();
        for out in [
            e.aggregate_resident(&input, &requests, &p).unwrap(),
            e.aggregate_selected(&input, &selected, &requests, &p)
                .unwrap(),
        ] {
            assert!(matches!(
                out.materialize(),
                Err(ExecutionError::IntegerOverflow { row: 1 })
            ));
        }
    }
    // COUNT(false) counts non-NULL expressions, not true mask values.
    assert_eq!(
        values(
            &e.aggregate_resident(&batch, &[("bool", Aggregate::Count(predicate.clone()))], &p)
                .unwrap()
        ),
        [AggregateValue::Count(2)]
    );
    let other = e
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[1, 2]))
                .unwrap(),
        )
        .unwrap();
    assert!(matches!(
        e.aggregate_selected(&other, &selected, &count, &p),
        Err(ExecutionError::SelectionSourceMismatch)
    ));
    drop(batch);
    drop(selected);
    let single = Executor::with_cube_size(client, 1).unwrap();
    assert!(single.aggregate_resident(&other, &count, &p).is_err());
    assert_eq!(
        values(
            &single
                .aggregate_resident(&other, &[("rows", Aggregate::CountStar)], &p)
                .unwrap()
        ),
        [AggregateValue::Count(2)]
    );
    assert_eq!(
        values(&e.aggregate_resident(&other, &count, &p).unwrap()),
        [AggregateValue::Count(2)]
    );
    drop(e.aggregate_resident(&other, &count, &p).unwrap());
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_integer_aggregates() {
    integers(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_float_aggregate_policy() {
    floats(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_aggregate_errors() {
    errors(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_integer_aggregates() {
    integers(Device::metal(Default::default()).expect("Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_float_aggregate_policy() {
    floats(Device::metal(Default::default()).expect("Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_aggregate_errors() {
    errors(Device::metal(Default::default()).expect("Metal required"));
}

#[cfg(all(feature = "cpu", feature = "metal", target_vendor = "apple"))]
#[test]
fn aggregate_count_star_checks_service_without_expression_bindings() {
    let client = Device::cpu().unwrap().client();
    let cpu = Executor::new(client.clone()).unwrap();
    let same_service = Executor::new(client).unwrap();
    let metal = Executor::new(Device::metal(Default::default()).unwrap().client()).unwrap();
    let request = [("rows", Aggregate::CountStar)];
    for n in [0, 3] {
        let batch = cpu.upload(&Batch::new(n)).unwrap();
        assert!(matches!(
            metal.aggregate_resident(&batch, &request, &Parameters::new()),
            Err(ExecutionError::ForeignDeviceBatch)
        ));
        assert_eq!(
            values(
                &same_service
                    .aggregate_resident(&batch, &request, &Parameters::new())
                    .unwrap()
            ),
            [AggregateValue::Count(n as u64)]
        );
    }
    assert_eq!(metal.stats(), Default::default());
}
