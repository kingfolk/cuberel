//! Stable selection on real devices; expected indices use ordinary Rust slices.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, ExecutionError, Executor, Field, Parameters, Schema, col, param,
    prepare,
};

fn expected(mask: &[u32]) -> Vec<u32> {
    mask.iter()
        .enumerate()
        .filter_map(|(i, &v)| (v == 1).then_some(i as u32))
        .collect()
}

fn masks(device: Device) {
    let client = device.client();
    let executor = Executor::new(client.clone()).unwrap();
    let tile = executor.cube_size() as usize;
    for n in [
        0,
        1,
        tile - 1,
        tile,
        tile + 1,
        tile * tile - 1,
        tile * tile,
        tile * tile + 1,
        17003,
    ] {
        for pattern in 0..4 {
            let mask: Vec<_> = (0..n)
                .map(|i| match pattern {
                    0 => 0,
                    1 => 1,
                    2 => (i % 2) as u32,
                    _ => u32::from((i * 17 + 31) % 97 < 3),
                })
                .collect();
            let selection = executor.compact_mask(&mask).unwrap();
            assert_eq!(selection.capacity(), n as u32);
            let stats = selection.stats();
            assert_eq!(stats.indices_capacity_bytes, n as u64 * 4);
            assert_eq!(stats.kernel_launches, stats.scan_levels * 2);
            if n > tile * tile {
                assert!(stats.scan_levels >= 3);
            }
            if n == 0 {
                assert_eq!(stats.workspace_bytes, 0);
                assert_eq!(stats.kernel_launches, 0);
            }
            assert_eq!(
                selection.materialize().unwrap(),
                expected(&mask),
                "n={n}, pattern={pattern}"
            );
        }
    }
    // Dynamic masks, lengths and number of recursive levels do not enter keys.
    let mask = vec![1; tile * tile + 3];
    executor.compact_mask(&mask).unwrap().materialize().unwrap();
    let defined = executor.define_count();
    executor
        .compact_mask(&[0, 1, 0])
        .unwrap()
        .materialize()
        .unwrap();
    assert_eq!(executor.define_count(), defined);
    let a = executor.compact_mask(&[0, 1]).unwrap();
    let b = a.clone();
    let c = executor.compact_mask(&[0, 1]).unwrap();
    assert!(a.same_source(&b));
    assert!(!a.same_source(&c));
    drop(a);
    assert_eq!(b.materialize().unwrap(), [1]);
    assert_eq!(b.materialize().unwrap(), [1]);
    // Queued resources remain valid even when an unobserved selection is dropped.
    drop(executor.compact_mask(&vec![1; tile * 3 + 1]).unwrap());
    assert_eq!(
        executor
            .compact_mask(&[1, 0, 1])
            .unwrap()
            .materialize()
            .unwrap(),
        [0, 2]
    );
    assert!(matches!(
        executor.compact_mask(&[0, 2, 1]),
        Err(ExecutionError::InvalidMaskValue { index: 1, value: 2 })
    ));
    assert_eq!(executor.define_count(), defined);

    // Non power-of-two and minimal supported tile sizes exercise the same scan.
    for size in [2, 3, 8] {
        let other = Executor::with_cube_size(client.clone(), size).unwrap();
        let mask: Vec<_> = (0..137).map(|i| u32::from(i % 3 == 1)).collect();
        assert_eq!(
            other.compact_mask(&mask).unwrap().materialize().unwrap(),
            expected(&mask)
        );
    }
    let single = Executor::with_cube_size(client, 1).unwrap();
    assert!(
        single
            .compact_mask(&[])
            .unwrap()
            .materialize()
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        single.compact_mask(&[1]),
        Err(ExecutionError::UnsupportedDevice { .. })
    ));
}

fn predicates(device: Device) {
    let executor = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32).with_nullable(true),
            Field::new("bias", DataType::I32),
        ],
    )
    .unwrap();
    let predicate = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let selected = {
        let values = [10, 0, 30, 40];
        let validity = [1, 0, 1, 1];
        let batch = Batch::new(4)
            .with_column(
                "x",
                ColumnData::NullableI32 {
                    values: &values,
                    validity: &validity,
                },
            )
            .unwrap();
        let params = Parameters::new().with("limit", Some(30_i32)).unwrap();
        executor.select(&predicate, &batch, &params).unwrap()
    }; // all host borrows and parameters can be dropped before completion
    let moved = selected.clone();
    assert_eq!(
        std::thread::spawn(move || moved.materialize().unwrap())
            .join()
            .unwrap(),
        [2, 3]
    );
    assert_eq!(selected.selected_count().unwrap(), 2);
    assert_eq!(selected.materialize().unwrap(), [2, 3]);
    let values = [10, 0, 30, 40];
    let validity = [1, 0, 1, 1];
    let batch = Batch::new(4)
        .with_column(
            "x",
            ColumnData::NullableI32 {
                values: &values,
                validity: &validity,
            },
        )
        .unwrap();
    let null = Parameters::new().with("limit", None::<i32>).unwrap();
    let none = executor.select(&predicate, &batch, &null).unwrap();
    assert!(none.materialize().unwrap().is_empty());
    let all = Parameters::new().with("limit", -1_i32).unwrap();
    assert_eq!(
        executor
            .select(&predicate, &batch, &all)
            .unwrap()
            .materialize()
            .unwrap(),
        [0, 2, 3]
    );
    let is_null = prepare(&col("x").is_null(), &schema).unwrap();
    assert_eq!(
        executor
            .select(&is_null, &batch, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap(),
        [1]
    );
    let numeric = prepare(&(col("x") + param("bias")), &schema).unwrap();
    assert!(matches!(
        executor.select(&numeric, &batch, &Parameters::new()),
        Err(ExecutionError::OutputType { .. })
    ));
    assert!(matches!(
        executor.select(&predicate, &batch, &Parameters::new()),
        Err(ExecutionError::Binding(_))
    ));
    let empty = Batch::new(0)
        .with_column("x", ColumnData::I32(&[]))
        .unwrap();
    let before = executor.define_count();
    assert!(
        executor
            .select(&predicate, &empty, &all)
            .unwrap()
            .materialize()
            .unwrap()
            .is_empty()
    );
    assert_eq!(executor.define_count(), before);

    // Predicate arithmetic errors survive even when WHERE would reject all rows.
    let overflow = prepare(&(col("x") + param("bias")).gt(param("limit")), &schema).unwrap();
    let input = Batch::new(3)
        .with_column("x", ColumnData::I32(&[i32::MAX, 0, i32::MAX]))
        .unwrap();
    let params = Parameters::new()
        .with("bias", 1_i32)
        .unwrap()
        .with("limit", i32::MAX)
        .unwrap();
    let failed = executor.select(&overflow, &input, &params).unwrap();
    for _ in 0..2 {
        assert!(matches!(
            failed.materialize(),
            Err(ExecutionError::IntegerOverflow { row: 0 })
        ));
    }
    assert_eq!(
        executor
            .select(&predicate, &batch, &all)
            .unwrap()
            .materialize()
            .unwrap(),
        [0, 2, 3]
    );

    // F32 predicates share exactly the same compaction path.
    let schema = Schema::new(&[("x", DataType::F32)], &[("limit", DataType::F32)]).unwrap();
    let predicate = prepare(&col("x").gt(param("limit")), &schema).unwrap();
    let input = Batch::new(5)
        .with_column(
            "x",
            ColumnData::F32(&[f32::NAN, -0.0, 1.0, f32::INFINITY, -1.0]),
        )
        .unwrap();
    let params = Parameters::new().with("limit", 0.0_f32).unwrap();
    assert_eq!(
        executor
            .select(&predicate, &input, &params)
            .unwrap()
            .materialize()
            .unwrap(),
        [2, 3]
    );
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_stable_masks() {
    masks(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_selected_predicates() {
    predicates(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_stable_masks() {
    masks(Device::metal(Default::default()).expect("native Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_selected_predicates() {
    predicates(Device::metal(Default::default()).expect("native Metal required"));
}
