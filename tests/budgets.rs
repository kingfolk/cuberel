#![cfg(any(feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, BudgetError, BudgetResource, ColumnData, DataType,
    ExecutionError, Executor, Field, FusionMode, Parameters, Pipeline, ResourceBudget, Schema,
    StageKind, col, param, prepare,
};
fn exercise(device: Device) {
    let e = Executor::new(device.client()).unwrap();
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("lo", DataType::I32),
            Field::new("one", DataType::I32),
        ],
    )
    .unwrap();
    let x = prepare(&col("x"), &schema).unwrap();
    let plus = prepare(&(col("x") + param("one")), &schema).unwrap();
    let p = prepare(&col("x").ge(param("lo")), &schema).unwrap();
    let params = Parameters::new()
        .with("lo", 4_i32)
        .unwrap()
        .with("one", 1_i32)
        .unwrap();
    for n in [
        0,
        1,
        e.cube_size() as usize + 1,
        e.cube_size() as usize * e.cube_size() as usize + 1,
    ] {
        let values: Vec<i32> = (0..n).map(|i| (i % 10) as i32).collect();
        let flags: Vec<u32> = (0..n).map(|i| u32::from(i % 5 != 0)).collect();
        let host = Batch::new(n)
            .with_column(
                "x",
                ColumnData::NullableI32 {
                    values: &values,
                    validity: &flags,
                },
            )
            .unwrap();
        let batch = e.upload(&host).unwrap();
        let mixed = Pipeline::new()
            .filter(p.clone())
            .filter(p.clone())
            .project("plus", plus.clone())
            .aggregate("sum", Aggregate::Sum(x.clone()));
        let fused = Pipeline::new()
            .filter(p.clone())
            .aggregate("sum", Aggregate::Sum(plus.clone()))
            .aggregate("rows", Aggregate::CountStar)
            .fusion(FusionMode::Required);
        for (query, has_projection) in [(mixed, true), (fused, false)] {
            let before = e.stats();
            let estimate = e.estimate_pipeline(&query, &batch, &params).unwrap();
            assert_eq!(
                e.stats(),
                before,
                "estimation must not allocate/upload/read/launch"
            );
            assert_eq!(estimate.input_bytes, n as u64 * 8);
            assert_eq!(
                estimate.output_rows,
                if has_projection { (n as u64).max(1) } else { 1 }
            );
            assert!(estimate.retained_bytes <= estimate.device_bytes);
            let limits = ResourceBudget {
                max_device_bytes: estimate.device_bytes,
                max_output_rows: estimate.output_rows,
                max_stages: estimate.stages,
            };
            let bounded = e.clone().with_budget(limits);
            assert_eq!(bounded.kernel_id(&plus), e.kernel_id(&plus));
            let pending = bounded.execute_pipeline(&query, &batch, &params).unwrap();
            assert_eq!(pending.estimate(), estimate);
            assert_eq!(pending.graph().stages().len() as u64, estimate.stages);
            assert_eq!(
                e.stats().allocation_bytes - before.allocation_bytes,
                estimate.new_allocation_bytes()
            );
            assert_eq!(
                e.stats().kernel_launches - before.kernel_launches,
                estimate.stages - 2
            );
            assert_eq!(e.stats().readback_calls, before.readback_calls);
            assert_eq!(pending.residency_bytes(), estimate.retained_bytes);
            let result = pending.materialize().unwrap();
            let kept: Vec<_> = values
                .iter()
                .zip(&flags)
                .filter_map(|(&v, &f)| (f == 1 && v >= 4).then_some(v))
                .collect();
            assert_eq!(result.row_count, kept.len() as u32);
            if result.projections.is_empty() {
                assert_eq!(
                    result.aggregates[0].value,
                    AggregateValue::I64(
                        (!kept.is_empty()).then(|| kept.iter().map(|&v| i64::from(v) + 1).sum())
                    )
                );
            } else {
                assert_eq!(
                    result.aggregates[0].value,
                    AggregateValue::I64(
                        (!kept.is_empty()).then(|| kept.iter().map(|&v| i64::from(v)).sum())
                    )
                );
            }
            // Every budget boundary is inclusive; one less rejects with no side effects.
            for (resource, short) in [
                (
                    BudgetResource::DeviceBytes,
                    ResourceBudget {
                        max_device_bytes: estimate.device_bytes.saturating_sub(1),
                        ..limits
                    },
                ),
                (
                    BudgetResource::Stages,
                    ResourceBudget {
                        max_stages: estimate.stages - 1,
                        ..limits
                    },
                ),
            ] {
                if resource == BudgetResource::DeviceBytes && estimate.device_bytes == 0 {
                    continue;
                }
                let before = e.stats();
                assert!(
                    matches!(e.clone().with_budget(short).execute_pipeline(&query,&batch,&params),Err(ExecutionError::Budget(BudgetError::Exceeded {resource:r,..})) if r==resource)
                );
                assert_eq!(e.stats(), before);
            }
            if estimate.output_rows > 0 {
                let before = e.stats();
                assert!(matches!(
                    e.clone()
                        .with_budget(ResourceBudget {
                            max_output_rows: estimate.output_rows - 1,
                            ..limits
                        })
                        .execute_pipeline(&query, &batch, &params),
                    Err(ExecutionError::Budget(BudgetError::Exceeded {
                        resource: BudgetResource::OutputRows,
                        ..
                    }))
                ));
                assert_eq!(e.stats(), before);
            }
            let host_estimate = bounded
                .estimate_pipeline_host(&query, &host, &params)
                .unwrap();
            assert_eq!(host_estimate, estimate);
            let before = e.stats();
            let pending = bounded
                .execute_pipeline_host(&query, &host, &params)
                .unwrap();
            assert_eq!(pending.graph().stages()[0].kind, StageKind::Upload);
            assert_eq!(
                e.stats().allocation_bytes - before.allocation_bytes,
                estimate.device_bytes
            );
            assert_eq!(e.stats().readback_calls, before.readback_calls);
            pending.complete().unwrap();
        }
    }
    // Parent selection inputs and source columns are deduplicated, not counted twice.
    let host = Batch::new(4)
        .with_column("x", ColumnData::I32(&[1, 4, 7, 9]))
        .unwrap();
    let batch = e.upload(&host).unwrap();
    let parent = e.select_resident(&p, &batch, &params).unwrap();
    let bounded = e.clone().with_budget(ResourceBudget {
        max_device_bytes: 0,
        ..ResourceBudget::UNLIMITED
    });
    macro_rules! rejects {
        ($call:expr) => {{
            let before = e.stats();
            assert!(matches!($call, Err(ExecutionError::Budget(_))));
            assert_eq!(e.stats(), before);
        }};
    }
    rejects!(bounded.upload(&host));
    rejects!(bounded.execute_nullable_i32(&plus, &host, &params));
    rejects!(bounded.select(&p, &host, &params));
    rejects!(bounded.compact_mask(&[1, 0, 1]));
    rejects!(bounded.execute_many_resident(&[x.clone(), plus.clone()], &batch, &params));
    rejects!(bounded.select_resident(&p, &batch, &params));
    rejects!(bounded.filter_selected(&p, &batch, &parent, &params));
    rejects!(bounded.project_selected(&batch, &parent, &[("x", &x)], &params));
    rejects!(bounded.aggregate_resident(&batch, &[("sum", Aggregate::Sum(x.clone()))], &params));
    rejects!(bounded.aggregate_selected(
        &batch,
        &parent,
        &[("sum", Aggregate::Sum(x.clone()))],
        &params
    ));
    let query = Pipeline::new()
        .filter(p.clone())
        .project("plus", plus.clone())
        .aggregate("sum", Aggregate::Sum(x.clone()));
    rejects!(bounded.execute_pipeline_host(&query, &host, &params));
    rejects!(bounded.execute_pipeline(&query, &batch, &params));
    let before = e.stats();
    assert!(
        e.execute_pipeline_host(&query, &host, &Parameters::new())
            .is_err()
    );
    assert_eq!(e.stats(), before);
    // A scalar row budget must not constrain intermediate capacity-N arrays.
    let scalar = e.clone().with_budget(ResourceBudget {
        max_output_rows: 1,
        ..ResourceBudget::UNLIMITED
    });
    assert_eq!(
        scalar
            .aggregate_selected(
                &batch,
                &parent,
                &[("sum", Aggregate::Sum(x.clone()))],
                &params
            )
            .unwrap()
            .materialize()
            .unwrap()[0]
            .value,
        AggregateValue::I64(Some(20))
    );
    let query = Pipeline::new()
        .filter(p)
        .aggregate("sum", Aggregate::Sum(x))
        .fusion(FusionMode::Unfused);
    assert_eq!(
        scalar
            .execute_pipeline_host(&query, &host, &params)
            .unwrap()
            .materialize()
            .unwrap()
            .aggregates[0]
            .value,
        AggregateValue::I64(Some(20))
    );
    // Exact direct selected-operation footprints include shared parent inputs once.
    for limit in [99, 100] {
        let executor = e.clone().with_budget(ResourceBudget {
            max_device_bytes: limit,
            ..ResourceBudget::UNLIMITED
        });
        let before = e.stats();
        let output = executor.project_selected(&batch, &parent, &[("plus", &plus)], &params);
        if limit == 99 {
            assert!(matches!(output, Err(ExecutionError::Budget(_))));
            assert_eq!(e.stats(), before);
        } else {
            output.unwrap().complete().unwrap();
            assert_eq!(e.stats().allocation_bytes - before.allocation_bytes, 48);
        }
    }
    // Metadata-only aggregate output still has one row, with no device buffers.
    let empty_source = Batch::new(1000);
    let metadata = Pipeline::new().aggregate("n", Aggregate::CountStar);
    let zero_memory = e.clone().with_budget(ResourceBudget {
        max_device_bytes: 0,
        max_output_rows: 1,
        max_stages: 2,
    });
    let before = e.stats();
    assert_eq!(
        zero_memory
            .execute_pipeline_host(&metadata, &empty_source, &Parameters::new())
            .unwrap()
            .materialize()
            .unwrap()
            .aggregates[0]
            .value,
        AggregateValue::Count(1000)
    );
    assert_eq!(e.stats(), before);
    // Filtered F32 SUM must budget the reference selection plan in Auto mode.
    let fs = Schema::from_fields(
        &[Field::new("f", DataType::F32).with_nullable(true)],
        &[Field::new("lo", DataType::F32)],
    )
    .unwrap();
    let f = prepare(&col("f"), &fs).unwrap();
    let pred = prepare(&col("f").ge(param("lo")), &fs).unwrap();
    let host = Batch::new(3)
        .with_column("f", ColumnData::F32(&[1., 2., 3.]))
        .unwrap();
    let parameters = Parameters::new().with("lo", 2_f32).unwrap();
    let query = Pipeline::new()
        .filter(pred)
        .aggregate("sum", Aggregate::Sum(f));
    let estimate = e
        .estimate_pipeline_host(&query, &host, &parameters)
        .unwrap();
    assert!(estimate.indices_bytes > 0 && estimate.scan_bytes > 0);
    let before = e.stats();
    let pending = e
        .clone()
        .with_budget(ResourceBudget {
            max_device_bytes: estimate.device_bytes,
            ..ResourceBudget::UNLIMITED
        })
        .execute_pipeline_host(&query, &host, &parameters)
        .unwrap();
    assert!(!pending.graph().is_fused());
    assert_eq!(
        e.stats().allocation_bytes - before.allocation_bytes,
        estimate.device_bytes
    );
    pending.materialize().unwrap();
}
#[cfg(feature = "cpu")]
#[test]
fn cpu_request_budgets() {
    exercise(Device::cpu().unwrap());
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
fn metal_request_budgets() {
    exercise(Device::metal(Default::default()).unwrap());
}
