//! Fresh-process M2 acceptance case. Args: cpu|metal rows pass_percent null_percent [samples].
use cubecl::{Device, MemoryPoolsReport, MemoryScope};
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, DeviceBatch, ExecutionStats, Executor,
    Field, HostColumn, Parameters, Pipeline, ResourceBudget, ResourceEstimate, Schema, col, param,
    prepare,
};
use std::{error::Error, hint::black_box, time::Instant};
struct Output {
    values: Vec<i32>,
    valid: Vec<u32>,
    rows: u64,
    sum: Option<i64>,
}
struct Observation {
    ms: f64,
    stats: ExecutionStats,
    defines: usize,
    estimate: Option<ResourceEstimate>,
    resident: u64,
}
struct Workload<'a> {
    source: &'a [i32],
    host: Batch<'a>,
    query: Pipeline,
    predicate: cuberel::KernelProgram,
    projection: cuberel::KernelProgram,
    x: cuberel::KernelProgram,
    params: Parameters,
    expected: Output,
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
fn check(out: Output, expected: &Output) {
    assert_eq!(out.values, expected.values);
    assert_eq!(out.valid, expected.valid);
    assert_eq!(out.rows, expected.rows);
    assert_eq!(out.sum, expected.sum);
    black_box(out);
}
fn query(
    e: &Executor,
    w: &Workload<'_>,
    batch: Option<&DeviceBatch>,
) -> Result<Observation, Box<dyn Error>> {
    let before = e.stats();
    let definitions = e.define_count();
    let start = Instant::now();
    let pending = if let Some(batch) = batch {
        e.execute_pipeline(&w.query, batch, &w.params)?
    } else {
        e.execute_pipeline_host(&w.query, &w.host, &w.params)?
    };
    let result = pending.materialize()?;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    let estimate = pending.estimate();
    let resident = pending.residency_bytes();
    let HostColumn::I32(column) = result.projections.into_iter().next().unwrap().into_data() else {
        panic!("I32 projection")
    };
    let AggregateValue::Count(rows) = result.aggregates[0].value else {
        panic!("count")
    };
    let AggregateValue::I64(sum) = result.aggregates[1].value else {
        panic!("sum")
    };
    check(
        Output {
            values: column.values().to_vec(),
            valid: column.validity().to_vec(),
            rows,
            sum,
        },
        &w.expected,
    );
    let stats = delta(before, e.stats());
    assert_eq!(
        stats.allocation_bytes,
        if batch.is_some() {
            estimate.new_allocation_bytes()
        } else {
            estimate.device_bytes
        }
    );
    assert!(resident <= estimate.retained_bytes);
    Ok(Observation {
        ms,
        stats,
        defines: e.define_count() - definitions,
        estimate: Some(estimate),
        resident,
    })
}
fn synchronous(e: &Executor, w: &Workload<'_>) -> Result<Observation, Box<dyn Error>> {
    let before = e.stats();
    let definitions = e.define_count();
    let start = Instant::now();
    let mask = e.execute_mask(&w.predicate, &w.host, &w.params)?;
    let original = w.source;
    let kept: Vec<i32> = original
        .iter()
        .zip(mask)
        .filter_map(|(&v, keep)| (keep == 1).then_some(v))
        .collect();
    let host = Batch::new(kept.len()).with_column("x", ColumnData::I32(&kept))?;
    let projected = e.execute_nullable_i32(&w.projection, &host, &w.params)?;
    let batch = e.upload(&host)?;
    let results = e
        .aggregate_resident(
            &batch,
            &[
                ("rows", Aggregate::CountStar),
                ("sum", Aggregate::Sum(w.x.clone())),
            ],
            &w.params,
        )?
        .materialize()?;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    let AggregateValue::Count(rows) = results[0].value else {
        panic!("count")
    };
    let AggregateValue::I64(sum) = results[1].value else {
        panic!("sum")
    };
    check(
        Output {
            values: projected.values().to_vec(),
            valid: projected.validity().to_vec(),
            rows,
            sum,
        },
        &w.expected,
    );
    Ok(Observation {
        ms,
        stats: delta(before, e.stats()),
        defines: e.define_count() - definitions,
        estimate: None,
        resident: 0,
    })
}
fn rust_primitive(values: &[i32], flags: &[u32], limit: i32, expected: &Output) -> Observation {
    let start = Instant::now();
    let (values, sum, rows) = values
        .iter()
        .zip(flags)
        .filter(|(v, flag)| **flag == 1 && **v >= limit)
        .fold(
            (Vec::new(), 0_i64, 0_u64),
            |(mut out, sum, rows), (&v, _)| {
                out.push(v.checked_add(1).expect("bounded benchmark domain"));
                (out, sum + i64::from(v), rows + 1)
            },
        );
    let valid = vec![1; values.len()];
    let ms = start.elapsed().as_secs_f64() * 1000.;
    check(
        Output {
            values,
            valid,
            rows,
            sum: (rows != 0).then_some(sum),
        },
        expected,
    );
    Observation {
        ms,
        stats: ExecutionStats::default(),
        defines: 0,
        estimate: None,
        resident: 0,
    }
}
fn print(mode: &str, iteration: usize, o: Observation) {
    let (bound, retained, stages) = o
        .estimate
        .map(|e| {
            (
                e.device_bytes.to_string(),
                e.retained_bytes.to_string(),
                e.stages.to_string(),
            )
        })
        .unwrap_or_default();
    println!(
        "{mode},{iteration},{:.6},{},{},{},{},{},{},{},{},{},{bound},{retained},{stages},{}",
        o.ms,
        o.stats.upload_calls,
        o.stats.upload_bytes,
        o.stats.readback_calls,
        o.stats.readback_bytes,
        o.stats.kernel_launches,
        o.stats.count_readbacks,
        o.stats.allocation_calls,
        o.stats.allocation_bytes,
        o.defines,
        o.resident
    );
}
fn peak(p: &MemoryPoolsReport) -> u64 {
    p.dynamic
        .iter()
        .chain([&p.persistent, &p.dedicated])
        .map(|p| p.pages_peak)
        .sum()
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let backend = args
        .get(1)
        .map(String::as_str)
        .ok_or("cpu|metal rows pass_percent null_percent [samples]")?;
    let n: usize = args.get(2).ok_or("rows")?.parse()?;
    let pass: i32 = args.get(3).ok_or("pass percent")?.parse()?;
    let null: u32 = args.get(4).ok_or("null percent")?.parse()?;
    let repeats: usize = args.get(5).map(String::as_str).unwrap_or("3").parse()?;
    if !(0..=100).contains(&pass) || null > 100 || repeats < 3 || repeats.is_multiple_of(2) {
        return Err("invalid percentage or sample count".into());
    }
    let values: Vec<i32> = (0..n).map(|i| (i % 1000) as i32).collect();
    let flags: Vec<u32> = (0..n)
        .map(|i| u32::from(i as u32 % 100 < 100 - null))
        .collect();
    let limit = 1000 - pass * 10;
    let kept: Vec<_> = values
        .iter()
        .zip(&flags)
        .filter_map(|(&v, &valid)| (valid == 1 && v >= limit).then_some(v))
        .collect();
    let expected = Output {
        values: kept.iter().map(|v| v + 1).collect(),
        valid: vec![1; kept.len()],
        rows: kept.len() as u64,
        sum: (!kept.is_empty()).then(|| kept.iter().map(|&v| i64::from(v)).sum()),
    };
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32),
            Field::new("one", DataType::I32),
        ],
    )?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let projection = prepare(&(col("x") + param("one")), &schema)?;
    let x = prepare(&col("x"), &schema)?;
    let workload = Workload {
        source: &values,
        host: Batch::new(n).with_column(
            "x",
            ColumnData::NullableI32 {
                values: &values,
                validity: &flags,
            },
        )?,
        query: Pipeline::new()
            .filter(predicate.clone())
            .project("plus", projection.clone())
            .aggregate("rows", Aggregate::CountStar)
            .aggregate("sum", Aggregate::Sum(x.clone())),
        predicate,
        projection,
        x,
        params: Parameters::new().with("limit", limit)?.with("one", 1_i32)?,
        expected,
    };
    if args.get(6).map(String::as_str) == Some("--rust-only") {
        println!(
            "# backend={backend},rows={n},pass={pass},null={null},tile=0,samples={repeats},debug={},init_ms=0,rust_only=true",
            cfg!(debug_assertions)
        );
        println!(
            "mode,iteration,ms,upload_calls,upload_bytes,reads,read_bytes,launches,count_reads,allocations,allocation_bytes,defines,device_bound,retained_bound,stages,result_residency"
        );
        for i in 0..repeats {
            print(
                "rust",
                i,
                rust_primitive(&values, &flags, limit, &workload.expected),
            );
        }
        return Ok(());
    }
    let start = Instant::now();
    let device = match backend {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("cpu or native metal required".into()),
    };
    let client = device.client();
    let e = Executor::new(client.clone())?.with_budget(ResourceBudget {
        max_device_bytes: 512 * 1024 * 1024,
        max_output_rows: 4 * 1024 * 1024,
        max_stages: 4096,
    });
    let init = start.elapsed().as_secs_f64() * 1000.;
    println!(
        "# backend={backend},rows={n},pass={pass},null={null},tile={},samples={repeats},debug={},init_ms={init:.6}",
        e.cube_size(),
        cfg!(debug_assertions)
    );
    println!(
        "mode,iteration,ms,upload_calls,upload_bytes,reads,read_bytes,launches,count_reads,allocations,allocation_bytes,defines,device_bound,retained_bound,stages,result_residency"
    );
    let mut first = query(&e, &workload, None)?;
    first.ms += init;
    print("first_query", 0, first);
    synchronous(&e, &workload)?;
    for i in 0..repeats {
        print("synchronous", i, synchronous(&e, &workload)?);
    }
    query(&e, &workload, None)?;
    for i in 0..repeats {
        let o = query(&e, &workload, None)?;
        assert_eq!(o.defines, 0, "warm query defined a new kernel");
        print("warm_host", i, o);
    }
    let before = e.stats();
    let definitions = e.define_count();
    let start = Instant::now();
    let batch = e.upload(&workload.host)?;
    let upload_ms = start.elapsed().as_secs_f64() * 1000.;
    print(
        "resident_upload",
        0,
        Observation {
            ms: upload_ms,
            stats: delta(before, e.stats()),
            defines: e.define_count() - definitions,
            estimate: None,
            resident: 0,
        },
    );
    query(&e, &workload, Some(&batch))?;
    for i in 0..repeats {
        let o = query(&e, &workload, Some(&batch))?;
        assert_eq!(o.stats.upload_calls, 0);
        assert_eq!(o.defines, 0);
        print("resident", i, o);
    }
    for i in 0..repeats {
        print(
            "rust",
            i,
            rust_primitive(&values, &flags, limit, &workload.expected),
        );
    }
    let report = client.memory_report(MemoryScope::CurrentStream);
    let usage = report.usage();
    let peaks: u64 = report
        .streams
        .iter()
        .map(|s| peak(&s.pools) + s.auxiliary.iter().map(|a| peak(&a.pools)).sum::<u64>())
        .sum();
    eprintln!(
        "runtime_in_use={},runtime_reserved={},sum_pool_page_peaks={peaks}\n{report:#?}",
        usage.bytes_in_use, usage.bytes_reserved
    );
    Ok(())
}
