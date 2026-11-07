//! Real-device selected pipelines with explicit expected rows and values.
#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Batch, ColumnData, DataType, ExecutionError, Executor, Field, HostColumn, Parameters, Schema,
    col, param, prepare,
};

fn pipeline(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32),
            Field::new("two", DataType::I32),
        ],
    )
    .unwrap();
    let predicate = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let twice = prepare(&(col("x") * param("two")), &schema).unwrap();
    let copy = prepare(&col("x"), &schema).unwrap();
    let bools = prepare(&col("x").gt(param("limit")), &schema).unwrap();
    let is_null = prepare(&col("x").is_null(), &schema).unwrap();
    let p = Parameters::new()
        .with("limit", 30_i32)
        .unwrap()
        .with("two", 2_i32)
        .unwrap();
    let batch = {
        let values = [10, 0, 30, 40];
        let validity = [1, 0, 1, 1];
        e.upload(
            &Batch::new(4)
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
    };
    let first = e.select_resident(&predicate, &batch, &p).unwrap();
    let nested = e.filter_selected(&bools, &batch, &first, &p).unwrap();
    let result = e
        .project_selected(
            &batch,
            &first,
            &[("twice", &twice), ("greater", &bools)],
            &p,
        )
        .unwrap();
    let nested_result = e
        .project_selected(&batch, &nested, &[("original", &copy)], &p)
        .unwrap();
    assert_eq!(e.stats().readback_calls, 0);
    assert_eq!(e.stats().count_readbacks, 0);
    assert_eq!(e.stats().upload_bytes, 32);
    assert!(nested.same_source(&first));
    assert!(nested.belongs_to(&batch));
    assert_eq!(
        result.fields().map(|f| f.name()).collect::<Vec<_>>(),
        ["twice", "greater"]
    );
    assert_eq!(result.capacity(), 4);
    let nulls = e.select_resident(&is_null, &batch, &p).unwrap();
    let null_result = e
        .project_selected(&batch, &nulls, &[("x", &copy)], &p)
        .unwrap();
    drop(batch);
    drop(first);
    drop(nested);
    let clone = result.clone();
    drop(result);
    let columns = std::thread::spawn(move || clone.materialize().unwrap())
        .join()
        .unwrap();
    let HostColumn::I32(v) = columns[0].data() else {
        panic!("I32 expected")
    };
    assert_eq!(v.iter().collect::<Vec<_>>(), [Some(60), Some(80)]);
    assert_eq!(
        columns[0].field(),
        &Field::new("twice", DataType::I32).with_nullable(true)
    );
    let HostColumn::Bool(v) = columns[1].data() else {
        panic!("Bool expected")
    };
    assert_eq!(v.iter().collect::<Vec<_>>(), [Some(false), Some(true)]);
    let HostColumn::I32(v) = nested_result.materialize().unwrap().remove(0).into_data() else {
        panic!("I32 expected")
    };
    assert_eq!(v.values(), [40]);
    let HostColumn::I32(v) = null_result.materialize().unwrap().remove(0).into_data() else {
        panic!("I32 expected")
    };
    assert_eq!(v.iter().collect::<Vec<_>>(), [None]);

    // Representative prefix boundaries: empty source, no match, all match, and
    // recursive scan with a sparse parent. No cartesian test matrix is needed.
    let tile = e.cube_size() as usize;
    for (n, threshold) in [
        (0, 0),
        (1, -1),
        (tile + 1, 100),
        (tile * tile + 1, 9),
        (65, -1),
    ] {
        let values: Vec<i32> = (0..n).map(|i| (i % 13) as i32).collect();
        let p = Parameters::new()
            .with("limit", threshold)
            .unwrap()
            .with("two", 2_i32)
            .unwrap();
        let batch = e
            .upload(
                &Batch::new(n)
                    .with_column("x", ColumnData::I32(&values))
                    .unwrap(),
            )
            .unwrap();
        let s = e.select_resident(&predicate, &batch, &p).unwrap();
        let child = e.filter_selected(&bools, &batch, &s, &p).unwrap();
        let before = e.stats();
        let out = e
            .project_selected(&batch, &child, &[("x", &twice)], &p)
            .unwrap();
        assert_eq!(e.stats().readback_calls, before.readback_calls);
        let expected: Vec<_> = values
            .iter()
            .filter(|&&v| v > threshold)
            .map(|v| v * 2)
            .collect();
        let HostColumn::I32(v) = out.materialize().unwrap().remove(0).into_data() else {
            panic!("I32 expected")
        };
        assert_eq!(v.values(), expected, "n={n}, threshold={threshold}");
        let rows: Vec<_> = values
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v > threshold).then_some(i as u32))
            .collect();
        assert_eq!(child.materialize().unwrap(), rows);
        if n == 0 {
            assert_eq!(e.stats(), before);
        }
        drop(
            e.project_selected(&batch, &s, &[("drop", &twice)], &p)
                .unwrap(),
        );
    }
}

fn errors(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::new(
        &[("x", DataType::I32), ("y", DataType::I32)],
        &[("one", DataType::I32), ("limit", DataType::I32)],
    )
    .unwrap();
    let p = Parameters::new()
        .with("one", 1_i32)
        .unwrap()
        .with("limit", 10_i32)
        .unwrap();
    let predicate = prepare(&col("x").lt(param("limit")), &schema).unwrap();
    let add = prepare(&(col("x") + param("one")), &schema).unwrap();
    let host = Batch::new(2)
        .with_column("x", ColumnData::I32(&[i32::MAX, 1]))
        .unwrap();
    let batch = e.upload(&host).unwrap();
    let s = e.select_resident(&predicate, &batch, &p).unwrap();
    let out = e
        .project_selected(&batch, &s, &[("plus", &add)], &p)
        .unwrap();
    let before = e.stats();
    let cols = out.materialize().unwrap();
    let HostColumn::I32(v) = cols[0].data() else {
        panic!("I32 expected")
    };
    assert_eq!(v.values(), [2]); // skipped MAX never produces projection overflow
    assert_eq!(e.stats().readback_bytes - before.readback_bytes, 12); // count + 1 error + 1 value; no capacity tail
    assert!(matches!(
        e.execute_i32(&add, &host, &p),
        Err(ExecutionError::IntegerOverflow { row: 0 })
    ));
    let other = e.upload(&host).unwrap();
    let before = e.stats();
    assert!(matches!(
        e.project_selected(&other, &s, &[("x", &add)], &p),
        Err(ExecutionError::SelectionSourceMismatch)
    ));
    assert!(matches!(
        e.filter_selected(&predicate, &other, &s, &p),
        Err(ExecutionError::SelectionSourceMismatch)
    ));
    assert!(matches!(
        e.project_selected(&batch, &s, &[], &p),
        Err(ExecutionError::EmptyProjection)
    ));
    assert!(matches!(
        e.project_selected(&batch, &s, &[("x", &add), ("x", &add)], &p),
        Err(ExecutionError::InvalidProjectionName)
    ));
    assert!(e.project_selected(&batch, &s, &[("", &add)], &p).is_err());
    let missing = prepare(&col("y"), &schema).unwrap();
    assert!(
        e.project_selected(&batch, &s, &[("ok", &add), ("missing", &missing)], &p)
            .is_err()
    );
    assert!(
        e.project_selected(&batch, &s, &[("x", &add)], &Parameters::new())
            .is_err()
    );
    assert_eq!(e.stats(), before);

    // Different outputs fail at different source rows; report min original row,
    // even when that row belongs to the later output, never compact slot zero.
    let host = Batch::new(4)
        .with_column("x", ColumnData::I32(&[-1, 0, -1, i32::MAX]))
        .unwrap()
        .with_column("y", ColumnData::I32(&[0, i32::MAX, 0, 0]))
        .unwrap();
    let batch = e.upload(&host).unwrap();
    let p = Parameters::new()
        .with("one", 1_i32)
        .unwrap()
        .with("limit", 0_i32)
        .unwrap();
    let keep = prepare(&col("x").ge(param("limit")), &schema).unwrap();
    let y = prepare(&(col("y") + param("one")), &schema).unwrap();
    let s = e.select_resident(&keep, &batch, &p).unwrap();
    let out = e
        .project_selected(&batch, &s, &[("late", &add), ("early", &y)], &p)
        .unwrap();
    assert!(matches!(
        out.materialize(),
        Err(ExecutionError::IntegerOverflow { row: 1 })
    ));
    // A failing nested predicate cannot be hidden by a later empty filter.
    let failing = prepare(&(col("y") + param("one")).lt(param("limit")), &schema).unwrap();
    let child = e.filter_selected(&failing, &batch, &s, &p).unwrap();
    let none = e.filter_selected(&predicate, &batch, &child, &p).unwrap();
    assert!(matches!(
        none.materialize(),
        Err(ExecutionError::IntegerOverflow { row: 1 })
    ));
    // A failing first predicate also remains observable after projection.
    let failed = e.select_resident(&failing, &batch, &p).unwrap();
    assert!(matches!(
        e.project_selected(&batch, &failed, &[("x", &add)], &p)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 1 })
    ));
    // Recover with the same snapshot/selection; no mutation or sticky error.
    let copy = prepare(&col("x"), &schema).unwrap();
    let HostColumn::I32(v) = e
        .project_selected(&batch, &s, &[("x", &copy)], &p)
        .unwrap()
        .materialize()
        .unwrap()
        .remove(0)
        .into_data()
    else {
        panic!("I32 expected")
    };
    assert_eq!(v.values(), [0, i32::MAX]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_selected_pipeline() {
    pipeline(Device::cpu().expect("CPU required"));
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_selected_errors() {
    errors(Device::cpu().expect("CPU required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_selected_pipeline() {
    pipeline(Device::metal(Default::default()).expect("Metal required"));
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_selected_errors() {
    errors(Device::metal(Default::default()).expect("Metal required"));
}
