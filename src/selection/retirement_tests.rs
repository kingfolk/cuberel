use cubecl::{
    prelude::{Client, KernelSettings},
    server::Handle,
};
use std::cell::{Cell, RefCell};
type OwnershipProbe = Box<dyn Fn(&[Handle]) -> bool>;
struct Probe {
    role: &'static str,
    released: Box<dyn Fn() -> bool>,
    owned: OwnershipProbe,
}
thread_local! {
    static PROBES: RefCell<Option<Vec<Probe>>> = const { RefCell::new(None) };
    static FAIL_MASK: Cell<usize> = const { Cell::new(0) };
}
fn probe(role: &'static str, handle: &Handle) {
    PROBES.with(|trace| {
        if let Some(trace) = trace.borrow_mut().as_mut() {
            let weak = handle.memory.clone().binding().downgrade();
            let id = handle.memory.clone().binding().id();
            trace.push(Probe {
                role,
                released: Box::new(move || weak.upgrade().is_none()),
                owned: Box::new(move |handles| {
                    handles
                        .iter()
                        .any(|h| h.memory.clone().binding().id() == id)
                }),
            });
        }
    });
}
pub(super) fn mask(client: &Client, settings: KernelSettings, handle: &Handle) {
    probe("mask", handle);
    FAIL_MASK.with(|count| {
        let remaining = count.get();
        if remaining != 0 {
            count.set(remaining - 1);
            if remaining == 1 {
                super::completion_tests::reject_output(client, settings, handle.clone());
            }
        }
    });
}
pub(super) fn validity(handle: Option<&Handle>) {
    if let Some(handle) = handle {
        probe("validity", handle);
    }
}
pub(super) fn workspace(rank: &Handle, totals: &Handle, final_count: bool) {
    probe("rank", rank);
    if !final_count {
        probe("totals", totals);
    }
}
#[cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
fn exercise(device: cubecl::Device) {
    use crate::{
        Aggregate, Batch, ColumnData, DataType, ExecutionError, Executor, Field, HostColumn,
        Parameters, Schema, col, param, prepare,
    };
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("lo", DataType::I32),
            Field::new("hi", DataType::I32),
            Field::new("one", DataType::I32),
        ],
    )
    .unwrap();
    let first = prepare(&col("x").ge(param("lo")), &schema).unwrap();
    let next = prepare(&col("x").lt(param("hi")), &schema).unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let params = Parameters::new()
        .with("lo", 30_i32)
        .unwrap()
        .with("hi", 50_i32)
        .unwrap();
    // Reference clones/branches keep their own indices, never private masks.
    PROBES.with(|p| *p.borrow_mut() = Some(Vec::new()));
    let batch = e
        .upload(
            &Batch::new(5)
                .with_column(
                    "x",
                    ColumnData::NullableI32 {
                        values: &[10, 0, 30, 40, 100],
                        validity: &[1, 0, 1, 1, 1],
                    },
                )
                .unwrap(),
        )
        .unwrap();
    let before = e.stats();
    let parent = e.select_resident(&first, &batch, &params).unwrap();
    let parent_clone = parent.clone();
    let parent_indices = parent.indices_handle().memory.binding().downgrade();
    let branch = e
        .project_selected(&batch, &parent, &[("x", &x)], &params)
        .unwrap();
    let child = e.filter_selected(&next, &batch, &parent, &params).unwrap();
    assert_eq!(e.stats().readback_calls, before.readback_calls);
    assert_eq!(e.stats().count_readbacks, before.count_readbacks);
    let probes = PROBES.with(|p| p.borrow_mut().take().unwrap());
    let owned = child.handles();
    for p in &probes {
        assert!(!(p.owned)(&owned), "{} remains reachable", p.role);
    }
    assert_eq!(probes.iter().filter(|p| p.role == "mask").count(), 2);
    assert_eq!(probes.iter().filter(|p| p.role == "validity").count(), 2);
    let residency = child.residency();
    assert_eq!(residency.indices_bytes, 20);
    assert_eq!(residency.count_bytes, 8);
    assert_eq!(residency.error_bytes, 0);
    assert_eq!(residency.dependency_bytes, 40);
    assert_eq!(child.materialize().unwrap(), [2, 3]);
    assert_eq!(parent_clone.materialize().unwrap(), [2, 3, 4]);
    let HostColumn::I32(column) = branch.materialize().unwrap().remove(0).into_data() else {
        panic!("I32")
    };
    assert_eq!(column.values(), [30, 40, 100]);
    for p in &probes {
        assert!(
            (p.released)(),
            "{} still pinned after consumer completion",
            p.role
        );
    }
    drop(parent);
    drop(parent_clone);
    drop(branch);
    assert!(
        parent_indices.upgrade().is_none(),
        "child retains ancestor indices"
    );
    // Aggregate depends on the compact completion history, not old masks.
    let a = e
        .aggregate_selected(
            &batch,
            &child,
            &[("sum", Aggregate::Sum(x.clone()))],
            &params,
        )
        .unwrap();
    drop(child);
    drop(batch);
    let clone = a.clone();
    drop(a);
    assert_eq!(
        std::thread::spawn(move || clone.materialize())
            .join()
            .unwrap()
            .unwrap()[0]
            .value,
        crate::AggregateValue::I64(Some(70))
    );
    // Explicit Bool output keeps its own values/validity unchanged.
    let batch = e
        .upload(
            &Batch::new(3)
                .with_column("x", ColumnData::I32(&[10, 30, 40]))
                .unwrap(),
        )
        .unwrap();
    let projected = e.execute_resident(&first, &batch, &params).unwrap();
    let s = e.select_resident(&first, &batch, &params).unwrap();
    drop(s);
    let HostColumn::Bool(bool_values) = projected.materialize().unwrap().remove(0) else {
        panic!("Bool")
    };
    assert_eq!(bool_values.values(), [0, 1, 1]);
    // Deep chains release ancestors even before export; include tails, all-NULL,
    // all-match, zero input and dropping unfinished intermediate selections.
    for (n, all_null) in [
        (0, false),
        (1, false),
        (e.cube_size() as usize + 1, false),
        (5, true),
    ] {
        let values: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let flags = vec![u32::from(!all_null); n];
        let batch = e
            .upload(
                &Batch::new(n)
                    .with_column(
                        "x",
                        ColumnData::NullableI32 {
                            values: &values,
                            validity: &flags,
                        },
                    )
                    .unwrap(),
            )
            .unwrap();
        let params = Parameters::new().with("lo", -1_i32).unwrap();
        PROBES.with(|p| *p.borrow_mut() = Some(Vec::new()));
        drop(e.select_resident(&first, &batch, &params).unwrap());
        let before = e.stats();
        let mut current = e.select_resident(&first, &batch, &params).unwrap();
        for _ in 1..16 {
            current = e
                .filter_selected(&first, &batch, &current, &params)
                .unwrap();
        }
        assert_eq!(e.stats().readback_calls, before.readback_calls);
        let residency = current.residency();
        assert_eq!(residency.stages, 16);
        assert_eq!(residency.indices_bytes, n as u64 * 4);
        assert_eq!(residency.count_bytes, if n == 0 { 0 } else { 64 });
        let probes = PROBES.with(|p| p.borrow_mut().take().unwrap());
        for probe in &probes {
            assert!(!(probe.owned)(&current.handles()));
        }
        let expected: Vec<u32> = if all_null {
            Vec::new()
        } else {
            (0..n as u32).collect()
        };
        assert_eq!(current.materialize().unwrap(), expected);
        for probe in probes {
            assert!((probe.released)(), "{} kept by deep history", probe.role);
        }
        assert_eq!(
            e.stats().count_readbacks - before.count_readbacks,
            if n == 0 { 0 } else { 16 }
        );
    }
    // Last parent's active slot overflows, although the child count is zero.
    let batch = e
        .upload(
            &Batch::new(5)
                .with_column("x", ColumnData::I32(&[0, 1, 2, 3, i32::MAX]))
                .unwrap(),
        )
        .unwrap();
    let params = Parameters::new()
        .with("lo", 2_i32)
        .unwrap()
        .with("hi", 0_i32)
        .unwrap()
        .with("one", 1_i32)
        .unwrap();
    let checked = prepare(&(col("x") + param("one")).lt(param("hi")), &schema).unwrap();
    let root = e.select_resident(&first, &batch, &params).unwrap();
    let child = e.filter_selected(&checked, &batch, &root, &params).unwrap();
    let grandchild = e.filter_selected(&next, &batch, &child, &params).unwrap();
    assert_eq!(grandchild.residency().error_bytes, 20);
    for _ in 0..2 {
        assert!(matches!(
            grandchild.clone().materialize(),
            Err(ExecutionError::IntegerOverflow { row: 4 })
        ));
    }
    assert!(matches!(
        e.project_selected(&batch, &grandchild, &[("x", &x)], &params)
            .unwrap()
            .materialize(),
        Err(ExecutionError::IntegerOverflow { row: 4 })
    ));
    // A failed retired mask still taints scatter's retained indices.
    for target in [1, 2] {
        FAIL_MASK.with(|f| f.set(target));
        let parent = e.select_resident(&first, &batch, &params).unwrap();
        let child = e.filter_selected(&next, &batch, &parent, &params).unwrap();
        drop(parent);
        let before = e.stats();
        assert!(matches!(
            child.materialize(),
            Err(ExecutionError::Backend(_))
        ));
        assert!(matches!(
            child.clone().materialize(),
            Err(ExecutionError::Backend(_))
        ));
        assert_eq!(
            e.stats(),
            before,
            "backend failure must precede any count/error reads"
        );
    }
    // Independent parent witnesses remain checked after dropping the parent.
    let mut parent = e.select_resident(&first, &batch, &params).unwrap();
    parent
        .retained
        .push(super::completion_tests::rejected_handle(
            &e.client,
            e.settings(),
        ));
    let child = e.filter_selected(&next, &batch, &parent, &params).unwrap();
    drop(parent);
    assert!(matches!(
        child.materialize(),
        Err(ExecutionError::Backend(_))
    ));
    assert_eq!(
        e.select_resident(&first, &batch, &params)
            .unwrap()
            .materialize()
            .unwrap(),
        [2, 3, 4]
    );
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_selection_retirement() {
    exercise(cubecl::Device::cpu().unwrap());
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_selection_retirement() {
    exercise(cubecl::Device::metal(Default::default()).unwrap());
}
