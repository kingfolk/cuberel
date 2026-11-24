#![cfg(feature = "datafusion")]

use arrow_array::Int32Array;
#[cfg(not(any(feature = "cpu", all(feature = "metal", target_os = "macos"))))]
use arrow_array::RecordBatch;
#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
use arrow_array::{Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::datafusion::{AdapterOptions, ProfilingMode, placement_report, register};
#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    prelude::{SessionConfig, SessionContext},
};
use std::sync::Arc;

fn input() -> RecordBatch {
    let padded = Int32Array::from(vec![
        Some(-3),
        Some(-2),
        Some(-1),
        Some(10),
        None,
        Some(30),
        Some(40),
    ]);
    let x = padded.slice(3, 4);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
        vec![Arc::new(x)],
    )
    .unwrap()
}

fn context(options: AdapterOptions, batch: RecordBatch) -> SessionContext {
    let state = register(
        SessionStateBuilder::new_with_default_features()
            .with_config(SessionConfig::new().with_target_partitions(1)),
        options,
    )
    .build();
    let ctx = SessionContext::new_with_state(state);
    ctx.register_batch("t", batch).unwrap();
    ctx
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
fn find_cube(plan: &Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
    if plan.name().contains("Cube") {
        return plan.clone();
    }
    for child in plan.children() {
        if let Some(found) = child.name().contains("Cube").then(|| child.clone()) {
            return found;
        }
        if !child.children().is_empty() {
            return find_cube(child);
        }
    }
    panic!(
        "missing Cube node: {}",
        displayable(plan.as_ref()).indent(true)
    );
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
fn metric(plan: &Arc<dyn ExecutionPlan>, name: &str) -> Option<usize> {
    let values: Vec<_> = plan
        .metrics()?
        .iter()
        .filter(|m| m.value().name() == name)
        .map(|m| m.value().as_usize())
        .collect();
    (!values.is_empty()).then(|| values.into_iter().sum())
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
fn rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.num_rows()).map(move |row| {
                batch
                    .columns()
                    .iter()
                    .map(|column| {
                        datafusion::arrow::util::display::array_value_to_string(column, row)
                            .unwrap()
                    })
                    .collect()
            })
        })
        .collect()
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
fn stats_delta(a: cuberel::ExecutionStats, b: cuberel::ExecutionStats) -> cuberel::ExecutionStats {
    cuberel::ExecutionStats {
        upload_calls: a.upload_calls - b.upload_calls,
        upload_bytes: a.upload_bytes - b.upload_bytes,
        readback_calls: a.readback_calls - b.readback_calls,
        readback_bytes: a.readback_bytes - b.readback_bytes,
        kernel_launches: a.kernel_launches - b.kernel_launches,
        count_readbacks: a.count_readbacks - b.count_readbacks,
        allocation_calls: a.allocation_calls - b.allocation_calls,
        allocation_bytes: a.allocation_bytes - b.allocation_bytes,
    }
}

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
const PROFILE: &str = "profile_validity_expand_ns profile_validity_expand_bytes profile_value_encode_ns profile_value_encode_bytes profile_validity_encode_ns profile_validity_encode_bytes profile_runtime_upload_ns profile_runtime_allocate_ns profile_pipeline_prepare_ns profile_pipeline_submit_ns profile_completion_sync_ns profile_readback_api_ns profile_scalar_decode_ns profile_arrow_export_ns profile_device_window_ns profile_device_windows profile_device_unavailable profile_device_failed profile_device_boundary_ns profile_graph_aggregate_partial profile_graph_aggregate_merge profile_graph_other_kernels profile_global_merge_ns profile_global_output_ns";

#[cfg(any(feature = "cpu", all(feature = "metal", target_os = "macos")))]
async fn exercise(executor: cuberel::Executor, has_device_timing: bool) {
    let queries = [
        (
            "SELECT x AS value FROM t WHERE x >= CAST(30 AS INT)",
            vec![vec![String::from("30")], vec![String::from("40")]],
        ),
        (
            "SELECT sum(x) AS total,count(*) AS n,count(x) AS valid,min(x) AS lo,max(x) AS hi FROM t WHERE x >= CAST(30 AS INT)",
            vec![
                vec!["70", "2", "2", "30", "40"]
                    .into_iter()
                    .map(String::from)
                    .collect(),
            ],
        ),
    ];
    for (sql, expected) in queries {
        let mut baseline = None;
        for mode in [
            ProfilingMode::Off,
            ProfilingMode::Host,
            ProfilingMode::Device,
        ] {
            let ctx = context(
                AdapterOptions::with_executor(executor.clone()).with_profiling_mode(mode),
                input(),
            );
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let node = find_cube(&plan);
            let before = executor.stats();
            let actual = collect(plan, ctx.task_ctx()).await.unwrap();
            let delta = stats_delta(executor.stats(), before);
            assert_eq!(rows(&actual), expected, "{sql}, {mode:?}");
            let result_schema = actual[0].schema();
            let fields = result_schema.fields();
            if sql.starts_with("SELECT x AS value") {
                assert_eq!(
                    (
                        fields[0].name().as_str(),
                        fields[0].data_type(),
                        fields[0].is_nullable()
                    ),
                    ("value", &DataType::Int32, true)
                );
            } else {
                assert_eq!(
                    fields.iter().map(|f| f.name().as_str()).collect::<Vec<_>>(),
                    ["total", "n", "valid", "lo", "hi"]
                );
                assert_eq!(
                    fields.iter().map(|f| f.data_type()).collect::<Vec<_>>(),
                    [
                        &DataType::Int64,
                        &DataType::Int64,
                        &DataType::Int64,
                        &DataType::Int32,
                        &DataType::Int32
                    ]
                );
                assert_eq!(
                    fields.iter().map(|f| f.is_nullable()).collect::<Vec<_>>(),
                    [true, false, false, true, true]
                );
            }
            if let Some((schema, stats)) = &baseline {
                assert_eq!(&actual[0].schema(), schema, "{sql}, {mode:?}");
                assert_eq!(&delta, stats, "{sql}, {mode:?}");
            } else {
                baseline = Some((actual[0].schema(), delta));
            }
            let found = PROFILE
                .split_whitespace()
                .filter(|&name| metric(&node, name).is_some())
                .count();
            if mode == ProfilingMode::Off {
                assert_eq!(found, 0);
            } else {
                assert_eq!(found, PROFILE.split_whitespace().count());
                for (profile, coarse) in [
                    ("profile_validity_expand_ns", "arrow_adapt"),
                    ("profile_pipeline_prepare_ns", "upload_submit"),
                    ("profile_readback_api_ns", "complete_export"),
                    ("profile_completion_sync_ns", "complete_export"),
                ] {
                    assert!(metric(&node, profile).unwrap() <= metric(&node, coarse).unwrap());
                }
                for name in [
                    "profile_value_encode_ns",
                    "profile_pipeline_submit_ns",
                    "profile_readback_api_ns",
                    "profile_completion_sync_ns",
                ] {
                    assert!(metric(&node, name).unwrap() > 0, "{name}, {sql}");
                }
                assert_eq!(
                    metric(&node, "profile_validity_expand_bytes"),
                    Some(16),
                    "{sql}, {mode:?}, plan={} ",
                    datafusion::physical_plan::display::DisplayableExecutionPlan::with_metrics(
                        node.as_ref()
                    )
                    .indent(true)
                );
                assert_eq!(metric(&node, "profile_value_encode_bytes"), Some(16));
                assert_eq!(metric(&node, "profile_validity_encode_bytes"), Some(16));
                assert_eq!(metric(&node, "profile_device_failed"), Some(0));
                assert_eq!(
                    metric(&node, "profile_graph_aggregate_partial").unwrap()
                        + metric(&node, "profile_graph_aggregate_merge").unwrap()
                        + metric(&node, "profile_graph_other_kernels").unwrap(),
                    metric(&node, "observed_launches").unwrap()
                );
                if sql.starts_with("SELECT x AS value") {
                    assert!(metric(&node, "profile_arrow_export_ns").unwrap() > 0);
                } else {
                    assert!(metric(&node, "profile_global_merge_ns").unwrap() > 0);
                    assert!(metric(&node, "profile_global_output_ns").unwrap() > 0);
                }
                if mode == ProfilingMode::Host {
                    assert_eq!(metric(&node, "profile_device_windows"), Some(0));
                    assert_eq!(metric(&node, "profile_device_window_ns"), Some(0));
                } else if has_device_timing {
                    assert_eq!(metric(&node, "profile_device_windows"), Some(1));
                    assert!(metric(&node, "profile_device_window_ns").unwrap() > 0);
                } else {
                    assert_eq!(metric(&node, "profile_device_unavailable"), Some(1));
                    assert_eq!(metric(&node, "profile_device_windows"), Some(0));
                    assert_eq!(metric(&node, "profile_device_window_ns"), Some(0));
                }
            }
        }
    }

    for mode in [ProfilingMode::Host, ProfilingMode::Device] {
        let ctx = context(
            AdapterOptions::with_executor(executor.clone()).with_profiling_mode(mode),
            input().slice(0, 0),
        );
        let plan = ctx
            .sql("SELECT count(*) AS n,sum(x) AS total FROM t")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let node = find_cube(&plan);
        let result = collect(plan, ctx.task_ctx()).await.unwrap();
        assert_eq!(
            result[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            0
        );
        assert!(result[0].column(1).is_null(0));
        for name in [
            "profile_device_windows",
            "profile_device_unavailable",
            "profile_device_failed",
            "profile_value_encode_bytes",
            "profile_validity_encode_bytes",
        ] {
            assert_eq!(metric(&node, name), Some(0), "{name}, {mode:?}");
        }
    }

    let opts = AdapterOptions::with_executor(executor).with_profiling_mode(ProfilingMode::Host);
    let ctx = context(opts, input());
    let plan = ctx
        .sql("SELECT x AS value FROM t WHERE x >= CAST(30 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    assert!(
        displayable(plan.as_ref())
            .indent(true)
            .to_string()
            .contains("profiling=Host")
    );
}

#[tokio::test]
async fn planning_only_has_no_profiling_mode_or_runtime() {
    let options = AdapterOptions::planning_only().with_profiling_mode(ProfilingMode::Host);
    let ctx = context(options.clone(), input());
    let plan = ctx
        .sql("SELECT x AS value FROM t WHERE x >= CAST(30 AS INT)")
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let report = placement_report(plan, &options).unwrap();
    assert_eq!(report.profiling_mode, None);
    assert!(report.to_string().contains("profiling=None"));
}

#[cfg(feature = "cpu")]
#[tokio::test]
async fn cpu_profiles_all_modes_with_host_results_and_stats() {
    exercise(
        cuberel::Executor::with_cube_size(cubecl::Device::cpu().unwrap().client(), 8).unwrap(),
        false,
    )
    .await;
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[tokio::test]
async fn metal_profiles_all_modes_with_host_results_and_stats() {
    exercise(
        cuberel::Executor::with_cube_size(
            cubecl::Device::metal(Default::default()).unwrap().client(),
            64,
        )
        .unwrap(),
        true,
    )
    .await;
}
