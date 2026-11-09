//! SUM primitive acceptance: exact wide integers and specified F32 trees.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, ExecutionError, Executor, Field,
    Parameters, Schema, col, param, prepare,
};
fn scalar(out: &cuberel::DeviceAggregates) -> AggregateValue {
    out.materialize().unwrap().remove(0).value
}
fn wide(device: Device) {
    let client = device.client();
    let e = Executor::new(client.clone()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let req = [("sum", Aggregate::Sum(x.clone()))];
    let cases: &[&[i32]] = &[
        &[],
        &[0],
        &[i32::MAX, i32::MAX],
        &[i32::MIN, i32::MIN],
        &[i32::MAX, 1],
        &[i32::MAX, i32::MAX, 2],
        &[-1, -1, 2],
        &[i32::MAX, i32::MAX, i32::MAX, i32::MIN, i32::MIN, -1],
    ];
    for &input in cases {
        let batch = e
            .upload(
                &Batch::new(input.len())
                    .with_column("x", ColumnData::I32(input))
                    .unwrap(),
            )
            .unwrap();
        let before = e.stats();
        let out = e
            .aggregate_resident(&batch, &req, &Parameters::new())
            .unwrap();
        assert_eq!(e.stats().readback_calls, before.readback_calls);
        let expected = (!input.is_empty()).then(|| input.iter().map(|&v| i64::from(v)).sum());
        assert_eq!(scalar(&out), AggregateValue::I64(expected));
        if input.is_empty() {
            assert_eq!(e.stats(), before);
        }
    }
    // Three merge levels, signed extension, low-word carry and NULL holes.
    let n = e.cube_size() as usize * e.cube_size() as usize + 1;
    let input: Vec<i32> = (0..n)
        .map(|i| match i % 4 {
            0 => i32::MAX,
            1 => i32::MIN,
            2 => -1,
            _ => 17,
        })
        .collect();
    let flags: Vec<u32> = (0..n).map(|i| u32::from(i % 5 != 0)).collect();
    let expected: i64 = input
        .iter()
        .zip(&flags)
        .filter_map(|(&v, &valid)| (valid == 1).then_some(i64::from(v)))
        .sum();
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
    let out = e
        .aggregate_resident(&batch, &req, &Parameters::new())
        .unwrap();
    assert!(out.stats().reduction_levels >= 3);
    let blocks: u64 = {
        let mut len = n as u64;
        let mut total = 0;
        loop {
            len = len.div_ceil(u64::from(e.cube_size()));
            total += len;
            if len == 1 {
                break;
            }
        }
        total
    };
    assert_eq!(out.stats().partial_state_bytes, blocks * 12);
    assert_eq!(scalar(&out), AggregateValue::I64(Some(expected)));
    let keep = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let p = Parameters::new().with("limit", 0_i32).unwrap();
    let selected = e.select_resident(&keep, &batch, &p).unwrap();
    let before = e.stats();
    let out = e
        .aggregate_selected(&batch, &selected, &req, &Parameters::new())
        .unwrap();
    assert_eq!(e.stats().readback_calls, before.readback_calls);
    assert_eq!(e.stats().upload_bytes, before.upload_bytes);
    let expected: i64 = input
        .iter()
        .zip(&flags)
        .filter_map(|(&v, &valid)| (valid == 1 && v >= 0).then_some(i64::from(v)))
        .sum();
    drop(selected);
    drop(batch);
    let moved = out.clone();
    drop(out);
    assert_eq!(
        std::thread::spawn(move || scalar(&moved)).join().unwrap(),
        AggregateValue::I64(Some(expected))
    );
    // All-NULL arguments and empty selections produce NULL, distinct from zero.
    let null = e
        .upload(
            &Batch::new(2)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &[i32::MAX, i32::MIN],
                        validity: &[0, 0],
                    },
                )
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        scalar(
            &e.aggregate_resident(&null, &req, &Parameters::new())
                .unwrap()
        ),
        AggregateValue::I64(None)
    );
    let selected = e.select_resident(&keep, &null, &p).unwrap();
    assert_eq!(
        scalar(
            &e.aggregate_selected(&null, &selected, &req, &Parameters::new())
                .unwrap()
        ),
        AggregateValue::I64(None)
    );
    let odd = Executor::with_cube_size(client, 3).unwrap();
    let values = vec![i32::MAX; 29];
    let batch = odd
        .upload(
            &Batch::new(29)
                .with_column("x", ColumnData::I32(&values))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        scalar(
            &odd.aggregate_resident(&batch, &req, &Parameters::new())
                .unwrap()
        ),
        AggregateValue::I64(Some(i64::from(i32::MAX) * 29))
    );
}
type FloatCase<'a> = (&'a [f32], &'a [u32], Option<u32>);
fn floats(device: Device) {
    let client = device.client();
    let e = Executor::with_cube_size(client.clone(), 4).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("f", DataType::F32).with_nullable(true)],
        &[Field::new("limit", DataType::F32)],
    )
    .unwrap();
    let f = prepare(&col("f"), &schema).unwrap();
    let req = [("sum", Aggregate::Sum(f.clone()))];
    let cases: &[FloatCase<'_>] = &[
        (&[], &[], None),
        (&[9.0], &[0], None),
        (&[-0.0], &[1], Some(0x8000_0000)),
        (&[-0.0, -0.0], &[1, 1], Some(0x8000_0000)),
        (&[-0.0, 0.0], &[1, 1], Some(0)),
        (&[1.5, -0.5, 2.0], &[1, 1, 1], Some(3.0_f32.to_bits())),
        (&[f32::from_bits(0xff80_0001)], &[1], Some(0x7fc0_0000)),
        (
            &[f32::INFINITY, f32::NEG_INFINITY],
            &[1, 1],
            Some(0x7fc0_0000),
        ),
        (&[f32::INFINITY, 5.0], &[1, 1], Some(0x7f80_0000)),
        (&[f32::NEG_INFINITY, -5.0], &[1, 1], Some(0xff80_0000)),
        (&[f32::NAN, 2.0], &[0, 1], Some(2.0_f32.to_bits())),
        // Pair tree: (large+1) + (-large+1) = 0; left fold would be 1.
        (&[1.0e20, 1.0, -1.0e20, 1.0], &[1, 1, 1, 1], Some(0)),
    ];
    for &(input, valid, expected) in cases {
        let batch = e
            .upload(
                &Batch::new(input.len())
                    .with_column(
                        "f",
                        ColumnData::NullableF32 {
                            values: input,
                            validity: valid,
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let AggregateValue::F32(result) = scalar(
            &e.aggregate_resident(&batch, &req, &Parameters::new())
                .unwrap(),
        ) else {
            panic!("F32 expected")
        };
        assert_eq!(result.map(f32::to_bits), expected);
    }
    // Tile 3 fixes a different grouping: ((large+1)+-large) + 1 = 1.
    let input = [1.0e20, 1.0, -1.0e20, 1.0];
    let batch = e
        .upload(
            &Batch::new(4)
                .with_column("f", ColumnData::F32(&input))
                .unwrap(),
        )
        .unwrap();
    let odd = Executor::with_cube_size(client, 3).unwrap();
    assert_eq!(
        scalar(
            &odd.aggregate_resident(&batch, &req, &Parameters::new())
                .unwrap()
        ),
        AggregateValue::F32(Some(1.0))
    );
    // Finite multi-level accuracy against exact input-f32 values summed in f64.
    let input: Vec<f32> = (0..1001)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
        .collect();
    let expected: f64 = input.iter().map(|&v| f64::from(v)).sum();
    let batch = e
        .upload(
            &Batch::new(input.len())
                .with_column("f", ColumnData::F32(&input))
                .unwrap(),
        )
        .unwrap();
    let mut previous = None;
    for _ in 0..3 {
        let AggregateValue::F32(Some(result)) = scalar(
            &e.aggregate_resident(&batch, &req, &Parameters::new())
                .unwrap(),
        ) else {
            panic!("F32 expected")
        };
        assert!((f64::from(result) - expected).abs() <= 1e-3 + 1e-5 * expected.abs());
        if let Some(bits) = previous {
            assert_eq!(result.to_bits(), bits);
        }
        previous = Some(result.to_bits());
    }
    let keep = prepare(&col("f").ge(param("limit")), &schema).unwrap();
    let selected = e
        .select_resident(
            &keep,
            &batch,
            &Parameters::new().with("limit", 0.0_f32).unwrap(),
        )
        .unwrap();
    let expected: f64 = input
        .iter()
        .filter(|&&v| v >= 0.0)
        .map(|&v| f64::from(v))
        .sum();
    let AggregateValue::F32(Some(result)) = scalar(
        &e.aggregate_selected(&batch, &selected, &req, &Parameters::new())
            .unwrap(),
    ) else {
        panic!("F32 expected")
    };
    assert!((f64::from(result) - expected).abs() <= 1e-3 + 1e-5 * expected.abs());
}
fn errors(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::new(
        &[("x", DataType::I32)],
        &[("one", DataType::I32), ("limit", DataType::I32)],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let add = prepare(&(col("x") + param("one")), &schema).unwrap();
    let predicate = prepare(&col("x").lt(param("limit")), &schema).unwrap();
    let p = Parameters::new()
        .with("one", 1_i32)
        .unwrap()
        .with("limit", 10_i32)
        .unwrap();
    let batch = e
        .upload(
            &Batch::new(2)
                .with_column("x", ColumnData::I32(&[i32::MAX, 1]))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        scalar(
            &e.aggregate_resident(&batch, &[("sum", Aggregate::Sum(x))], &p)
                .unwrap()
        ),
        AggregateValue::I64(Some(i64::from(i32::MAX) + 1))
    );
    let req = [("sum", Aggregate::Sum(add))];
    assert!(matches!(
        e.aggregate_resident(&batch, &req, &p)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    let selected = e.select_resident(&predicate, &batch, &p).unwrap();
    assert_eq!(
        scalar(&e.aggregate_selected(&batch, &selected, &req, &p).unwrap()),
        AggregateValue::I64(Some(2))
    );
    let failing = prepare(&(col("x") + param("one")).gt(param("limit")), &schema).unwrap();
    let failed = e.select_resident(&failing, &batch, &p).unwrap();
    assert!(matches!(
        e.aggregate_selected(&batch, &failed, &req, &p)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    let before = e.stats();
    assert!(
        e.aggregate_resident(&batch, &[("bad", Aggregate::Sum(predicate))], &p)
            .is_err()
    );
    assert!(
        e.aggregate_resident(&batch, &req, &Parameters::new())
            .is_err()
    );
    assert_eq!(e.stats(), before);
    assert_eq!(
        scalar(&e.aggregate_selected(&batch, &selected, &req, &p).unwrap()),
        AggregateValue::I64(Some(2))
    );
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_wide_sum() {
    wide(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_float_sum() {
    floats(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_sum_errors() {
    errors(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_wide_sum() {
    wide(Device::metal(Default::default()).expect("Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_float_sum() {
    floats(Device::metal(Default::default()).expect("Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_sum_errors() {
    errors(Device::metal(Default::default()).expect("Metal required"));
}
