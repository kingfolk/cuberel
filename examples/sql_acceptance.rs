//! One fresh-process F8 case. stdout=CSV; stderr=plans/placement/metrics.
#[path = "support/sql_acceptance.rs"]
mod support;
use cuberel::{
    datafusion::{register, AdapterOptions, CubeAggregateExec, CubeExec, SharedResources},
    ExecutionStats, Executor,
};
use datafusion::{
    execution::session_state::SessionStateBuilder,
    physical_plan::{collect, display::DisplayableExecutionPlan, displayable, ExecutionPlan},
    prelude::{SessionConfig, SessionContext},
};
use std::{sync::Arc, time::Instant};
use support::{compare, Case, Fixture, Result, Workload};

#[cfg(any(feature = "cpu", feature = "metal"))]
fn executor(backend: &str) -> Result<Option<Executor>> {
    let device = match backend {
        "datafusion" => return Ok(None),
        #[cfg(feature = "cpu")]
        "cpu" => cubecl::Device::cpu()?,
        #[cfg(feature = "metal")]
        "metal" => cubecl::Device::metal(Default::default())?,
        _ => {
            return Err(format!(
                "backend {backend} unavailable; no backend substitution is allowed"
            )
            .into());
        }
    };
    Ok(Some(Executor::with_cube_size(
        device.client(),
        if backend == "cpu" { 8 } else { 64 },
    )?))
}
#[cfg(not(any(feature = "cpu", feature = "metal")))]
fn executor(backend: &str) -> Result<Option<Executor>> {
    if backend == "datafusion" {
        Ok(None)
    } else {
        Err(format!("backend {backend} unavailable; no backend substitution is allowed").into())
    }
}
fn cube_nodes(p: &Arc<dyn ExecutionPlan>) -> usize {
    usize::from(p.is::<CubeExec>() || p.is::<CubeAggregateExec>())
        + p.children().iter().map(|c| cube_nodes(c)).sum::<usize>()
}
fn has_global(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<CubeAggregateExec>() || plan.children().into_iter().any(has_global)
}
fn metric(p: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    let local = if p.is::<CubeExec>() || p.is::<CubeAggregateExec>() {
        p.metrics()
            .map(|m| {
                m.iter()
                    .filter(|m| m.value().name() == name)
                    .map(|m| m.value().as_usize())
                    .sum()
            })
            .unwrap_or(0)
    } else {
        0
    };
    local + p.children().iter().map(|c| metric(c, name)).sum::<usize>()
}
fn optional_metric(p: &Arc<dyn ExecutionPlan>, name: &str) -> Option<usize> {
    let local = if p.is::<CubeExec>() || p.is::<CubeAggregateExec>() {
        p.metrics().and_then(|m| {
            let values = m
                .iter()
                .filter(|m| m.value().name() == name)
                .map(|m| m.value().as_usize())
                .collect::<Vec<_>>();
            (!values.is_empty()).then(|| values.into_iter().sum())
        })
    } else {
        None
    };
    let mut total = local.unwrap_or(0);
    let mut found = local.is_some();
    for child in p.children() {
        if let Some(value) = optional_metric(child, name) {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}
fn max_metric(p: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    let local = if p.is::<CubeExec>() || p.is::<CubeAggregateExec>() {
        p.metrics()
            .map(|m| {
                m.iter()
                    .filter(|m| m.value().name() == name)
                    .map(|m| m.value().as_usize())
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    } else {
        0
    };
    p.children()
        .into_iter()
        .map(|c| max_metric(c, name))
        .fold(local, usize::max)
}
fn delta(a: ExecutionStats, b: ExecutionStats) -> ExecutionStats {
    ExecutionStats {
        upload_calls: b.upload_calls - a.upload_calls,
        upload_bytes: b.upload_bytes - a.upload_bytes,
        readback_calls: b.readback_calls - a.readback_calls,
        readback_bytes: b.readback_bytes - a.readback_bytes,
        kernel_launches: b.kernel_launches - a.kernel_launches,
        count_readbacks: b.count_readbacks - a.count_readbacks,
        allocation_calls: b.allocation_calls - a.allocation_calls,
        allocation_bytes: b.allocation_bytes - a.allocation_bytes,
    }
}
fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.
}

// The direct DSL uses host input every query, exports Arrow and merges batch scalars.
// It has no DataFusion scheduling or SharedResources controller; peaks remain unavailable.
fn dsl(
    f: &Fixture,
    w: Workload,
    e: &Executor,
) -> Result<(Vec<arrow_array::RecordBatch>, f64, f64, u64)> {
    use arrow_array::{Int32Array, Int64Array, RecordBatch};
    use cuberel::{
        arrow::ArrowBatch, col, param, prepare, Aggregate, AggregateValue, DataType, Field,
        Parameters, Pipeline, Schema,
    };
    let start = Instant::now();
    let schema = Schema::from_fields(
        &[
            Field::new("id", DataType::I32),
            Field::new("x", DataType::I32).with_nullable(true),
            Field::new("y", DataType::I32).with_nullable(true),
        ],
        &[Field::new("limit", DataType::I32)],
    )?;
    let pred = prepare(&col("x").lt(param("limit")), &schema)?;
    let y = prepare(&col("y"), &schema)?;
    // Binding-only union of referenced columns for Arrow adaptation; never executed.
    let binding = prepare(
        &if w == Workload::Aggregate {
            col("x") + col("y")
        } else {
            col("id") + col("x") + col("y")
        },
        &schema,
    )?;
    let mut query = Pipeline::new().filter(pred);
    if w == Workload::Aggregate {
        query = query
            .aggregate("total", Aggregate::Sum(y.clone()))
            .aggregate("n", Aggregate::CountStar)
            .aggregate("valid", Aggregate::Count(y.clone()))
            .aggregate("lo", Aggregate::Min(y.clone()))
            .aggregate("hi", Aggregate::Max(y));
    } else {
        query = query
            .project("id", prepare(&col("id"), &schema)?)
            .project("value", prepare(&col("x"), &schema)?)
            .project("payload", y)
            .project("missing", prepare(&col("y").is_null(), &schema)?);
    }
    let params = Parameters::new().with("limit", f.case.pass as i32)?;
    let prepare_ms = ms(start);
    let start = Instant::now();
    let mut output = vec![];
    let (mut total, mut n, mut valid, mut lo, mut hi) =
        (None::<i64>, 0_i64, 0_i64, None::<i32>, None::<i32>);
    let mut peak = 0;
    for b in f.parts.iter().flatten() {
        let adapted = ArrowBatch::try_new(&binding, b)?;
        let pending = e.execute_pipeline_host(&query, &adapted.as_batch()?, &params)?;
        peak = peak.max(pending.estimate().device_bytes);
        if w == Workload::Aggregate {
            let r = pending.materialize()?;
            match (
                &r.aggregates[0].value,
                &r.aggregates[1].value,
                &r.aggregates[2].value,
                &r.aggregates[3].value,
                &r.aggregates[4].value,
            ) {
                (
                    AggregateValue::I64(sum),
                    AggregateValue::Count(rows),
                    AggregateValue::Count(count),
                    AggregateValue::I32(min),
                    AggregateValue::I32(max),
                ) => {
                    if let Some(s) = sum {
                        total = Some(
                            total
                                .unwrap_or(0)
                                .checked_add(*s)
                                .ok_or("DSL sum overflow")?,
                        );
                    }
                    n = n
                        .checked_add(i64::try_from(*rows)?)
                        .ok_or("DSL count overflow")?;
                    valid = valid
                        .checked_add(i64::try_from(*count)?)
                        .ok_or("DSL count overflow")?;
                    if let Some(v) = min {
                        lo = Some(lo.map_or(*v, |old| old.min(*v)));
                    }
                    if let Some(v) = max {
                        hi = Some(hi.map_or(*v, |old| old.max(*v)));
                    }
                }
                _ => return Err("unexpected DSL aggregate shape".into()),
            }
        } else if let Some(b) = pending.materialize_arrow()?.projections {
            output.push(b);
        }
    }
    if w == Workload::Aggregate {
        // Explicit SQL shape conversion: DSL COUNT is UInt64; SQL COUNT is Int64.
        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("total", arrow_schema::DataType::Int64, true),
            arrow_schema::Field::new("n", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("valid", arrow_schema::DataType::Int64, false),
            arrow_schema::Field::new("lo", arrow_schema::DataType::Int32, true),
            arrow_schema::Field::new("hi", arrow_schema::DataType::Int32, true),
        ]));
        output.push(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![total])),
                Arc::new(Int64Array::from(vec![n])),
                Arc::new(Int64Array::from(vec![valid])),
                Arc::new(Int32Array::from(vec![lo])),
                Arc::new(Int32Array::from(vec![hi])),
            ],
        )?);
    }
    let output = output
        .into_iter()
        .map(|b| {
            let schema = Arc::new(
                b.schema()
                    .as_ref()
                    .clone()
                    .with_metadata(f.input.schema().metadata().clone()),
            );
            arrow_array::RecordBatch::try_new(schema, b.columns().to_vec())
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok((output, prepare_ms, ms(start), peak))
}
#[tokio::main]
async fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 9 && args.len() != 10 {
        return Err("usage: sql_acceptance datafusion|cpu|metal sql|dsl project|aggregate|mixed rows pass_percent null_percent batches partitions warm_samples [session_batch_size]".into());
    }
    let backend = &args[0];
    let path = &args[1];
    let workload = Workload::parse(&args[2])?;
    if path != "sql" && path != "dsl" {
        return Err("path must be sql or dsl".into());
    }
    if path == "dsl" && (backend == "datafusion" || workload == Workload::Mixed) {
        return Err("DSL requires a device and project/aggregate workload".into());
    }
    let requested_batch_size = args
        .get(9)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if requested_batch_size == Some(0) {
        return Err("session_batch_size must be positive".into());
    }
    if path == "dsl" && requested_batch_size.is_some() {
        return Err("session_batch_size applies only to SQL; omit it for DSL".into());
    }
    let session_batch_size = requested_batch_size.unwrap_or(8192);
    let case = Case {
        rows: args[3].parse()?,
        pass: args[4].parse()?,
        nulls: args[5].parse()?,
        batches: args[6].parse()?,
        partitions: args[7].parse()?,
    };
    let samples = args[8].parse::<usize>()?;
    if samples == 0 {
        return Err("at least one warm sample required".into());
    }
    let f = Fixture::new(case)?;
    let expected = f.expected(workload)?;
    let start = Instant::now();
    let e = executor(backend)?;
    let init_ms = ms(start);
    let sql = workload.sql(case.pass);
    eprintln!("sql={sql}\ncase={case:?}\nbackend={backend} path={path} init_ms={init_ms:.6}");
    println!(
        "backend,path,workload,rows,pass_percent,null_percent,batches,partitions,session_batch_size,phase,sample,init_ms,prepare_ms,execute_ms,total_ms,output_rows,upload_calls,upload_bytes,readback_calls,readback_bytes,launches,count_reads,allocation_calls,allocation_bytes,define_calls,queue_wait_ns,adapter_prepare_ns,execute_export_ns,worker_ns,peak_device_bytes,peak_host_bytes,max_request_device_bytes,input_batches,completed_requests,actual_fused_requests,actual_unfused_requests"
    );
    for sample in 0..=samples {
        let before = e.as_ref().map(Executor::stats).unwrap_or_default();
        let defs = e.as_ref().map(Executor::define_count).unwrap_or(0);
        let (out, schema, prepare_ms, execute_ms, resources, plan, max_request) = if path == "dsl" {
            let (out, p, x, peak) = dsl(&f, workload, e.as_ref().unwrap())?;
            let schema = out
                .first()
                .map(|b| b.schema())
                .unwrap_or_else(|| expected.schema());
            (out, schema, p, x, None, None, peak)
        } else {
            let resources = SharedResources::default();
            let options = e
                .as_ref()
                .map(|e| AdapterOptions::with_resources(e.clone(), resources.clone()))
                .unwrap_or_default();
            let start = Instant::now();
            let builder = SessionStateBuilder::new_with_default_features().with_config(
                SessionConfig::new()
                    .with_target_partitions(case.partitions)
                    .with_batch_size(session_batch_size),
            );
            let ctx = SessionContext::new_with_state(register(builder, options.clone()).build());
            f.register(&ctx)?;
            let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
            let p = ms(start);
            if (e.is_some() && cube_nodes(&plan) == 0) || (e.is_none() && cube_nodes(&plan) != 0) {
                return Err(
                    "unexpected placement; refusing to label CPU fallback as device execution"
                        .into(),
                );
            }
            if e.is_some() && workload == Workload::Aggregate && !has_global(&plan) {
                return Err("aggregate workload did not select CubeAggregateExec".into());
            }
            if sample == 0 {
                eprintln!("{}", displayable(plan.as_ref()).indent(true));
            }
            let start = Instant::now();
            let out = collect(plan.clone(), ctx.task_ctx()).await?;
            let x = ms(start);
            let peak = max_metric(&plan, "max_estimated_device_bytes") as u64;
            (out, plan.schema(), p, x, Some(resources), Some(plan), peak)
        };
        let stats = delta(before, e.as_ref().map(Executor::stats).unwrap_or_default());
        let definitions = e.as_ref().map(Executor::define_count).unwrap_or(0) - defs;
        // The snapshot is per sample; output is still owned here and charged.
        let snapshot = resources.as_ref().map(SharedResources::snapshot);
        if let Some(plan) = &plan {
            for (name, value) in [
                ("observed_upload_bytes", stats.upload_bytes),
                ("observed_readback_bytes", stats.readback_bytes),
                ("observed_launches", stats.kernel_launches),
            ] {
                if metric(plan, name) as u64 != value {
                    return Err(format!("metric mismatch: {name}").into());
                }
            }
            if sample == samples {
                eprintln!(
                    "{}",
                    DisplayableExecutionPlan::with_metrics(plan.as_ref()).indent(true)
                );
            }
        }
        compare(&schema, &out, &expected, workload == Workload::Project)?;
        let output_rows = out.iter().map(|b| b.num_rows()).sum::<usize>();
        drop(out);
        if let Some(resources) = &resources {
            let s = resources.snapshot();
            if s.active_requests != 0
                || s.active_streams != 0
                || s.device_bytes != 0
                || s.host_bytes != 0
                || s.poisoned
                || s.rejected_reservations != 0
            {
                return Err(format!("resources did not drain: {s:?}").into());
            }
        }
        let times = ["queue_wait", "prepare", "execute_export", "worker"]
            .map(|n| plan.as_ref().map(|p| metric(p, n)).unwrap_or(0));
        let peak_device = snapshot
            .map(|s| s.peak_device_bytes.to_string())
            .unwrap_or_default();
        let peak_host = snapshot
            .map(|s| s.peak_host_bytes.to_string())
            .unwrap_or_default();
        let runtime_metrics = plan.as_ref().map(|plan| {
            [
                "input_batches",
                "requests_completed",
                "fused_requests",
                "unfused_requests",
            ]
            .map(|name| {
                optional_metric(plan, name)
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            })
        });
        let runtime_metrics = runtime_metrics.unwrap_or_default();
        println!(
            "{backend},{path},{},{},{},{},{},{},{},{},{sample},{:.6},{prepare_ms:.6},{execute_ms:.6},{:.6},{output_rows},{},{},{},{},{},{},{},{},{definitions},{},{},{},{},{peak_device},{peak_host},{max_request},{},{},{},{}",
            args[2],
            case.rows,
            case.pass,
            case.nulls,
            case.batches,
            case.partitions,
            if path == "sql" { session_batch_size.to_string() } else { String::new() },
            if sample == 0 { "first" } else { "warm" },
            if sample == 0 { init_ms } else { 0. },
            prepare_ms + execute_ms,
            stats.upload_calls,
            stats.upload_bytes,
            stats.readback_calls,
            stats.readback_bytes,
            stats.kernel_launches,
            stats.count_readbacks,
            stats.allocation_calls,
            stats.allocation_bytes,
            times[0],
            times[1],
            times[2],
            times[3],
            runtime_metrics[0],
            runtime_metrics[1],
            runtime_metrics[2],
            runtime_metrics[3]
        );
    }
    Ok(())
}
