//! Four-filter residency/latency report. Run each case in a fresh process.
use cubecl::{Device, MemoryPoolsReport, MemoryScope};
use cuberel::{
    Batch, ColumnData, DataType, DeviceSelection, Executor, Field, Parameters, Schema, col, param,
    prepare,
};
use std::{error::Error, hint::black_box, time::Instant};
fn owned_bytes(selection: &DeviceSelection) -> u64 {
    selection.residency().total_bytes()
}
fn peak_pages(pools: &MemoryPoolsReport) -> u64 {
    pools
        .dynamic
        .iter()
        .chain([&pools.persistent, &pools.dedicated])
        .map(|p| p.pages_peak)
        .sum()
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let backend = args
        .get(1)
        .map(String::as_str)
        .ok_or("usage: selection_residency cpu|metal [limit] [rows] [samples] [depth]")?;
    let device = match backend {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("select cpu or native metal".into()),
    };
    let client = device.client();
    let e = Executor::new(client.clone())?;
    let limit: i32 = args.get(2).map(String::as_str).unwrap_or("990").parse()?;
    let n: usize = args.get(3).map(String::as_str).unwrap_or("65536").parse()?;
    let samples: usize = args.get(4).map(String::as_str).unwrap_or("7").parse()?;
    let depth: usize = args.get(5).map(String::as_str).unwrap_or("4").parse()?;
    if samples < 3 || samples.is_multiple_of(2) || depth == 0 {
        return Err("use positive depth and odd samples >=3".into());
    }
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[Field::new("limit", DataType::I32)],
    )?;
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let x: Vec<i32> = (0..n).map(|i| (i % 1000) as i32).collect();
    let valid: Vec<u32> = (0..n).map(|i| u32::from(i % 10 != 9)).collect();
    let expected = x
        .iter()
        .zip(&valid)
        .filter(|(x, valid)| **valid == 1 && **x >= limit)
        .count();
    let batch = e.upload(&Batch::new(n).with_column(
        "x",
        ColumnData::NullableI32 {
            values: &x,
            validity: &valid,
        },
    )?)?;
    let params = Parameters::new().with("limit", limit)?;
    let mut times = Vec::new();
    let mut last = None;
    let mut delta = (0, 0, 0, 0, 0, 0);
    println!(
        "# {backend}, tile={}, release={}, samples={samples}, warmups=2",
        e.cube_size(),
        !cfg!(debug_assertions)
    );
    for sample in 0..samples + 2 {
        drop(last.take());
        let before = e.stats();
        let start = Instant::now();
        let mut selection = e.select_resident(&predicate, &batch, &params)?;
        for _ in 1..depth {
            selection = e.filter_selected(&predicate, &batch, &selection, &params)?;
        }
        let submitted = e.stats();
        assert_eq!(submitted.readback_calls, before.readback_calls);
        assert_eq!(submitted.count_readbacks, before.count_readbacks);
        let count = selection.selected_count()?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        assert_eq!(count as usize, expected);
        let after = e.stats();
        delta = (
            after.kernel_launches - before.kernel_launches,
            after.upload_calls - before.upload_calls,
            after.readback_calls - before.readback_calls,
            after.readback_bytes - before.readback_bytes,
            after.count_readbacks - before.count_readbacks,
            owned_bytes(&selection),
        );
        if sample >= 2 {
            times.push(elapsed);
            eprintln!("sample,{backend},{n},{depth},{limit},{sample},{elapsed:.6}");
        }
        last = Some(selection);
        black_box(count);
    }
    times.sort_by(f64::total_cmp);
    println!(
        "backend,rows,depth,limit,selected,owned_bytes,launches,uploads,reads,read_bytes,count_reads,median_ms"
    );
    println!(
        "{backend},{n},{depth},{limit},{expected},{},{},{},{},{},{},{:.6}",
        delta.5,
        delta.0,
        delta.1,
        delta.2,
        delta.3,
        delta.4,
        times[samples / 2]
    );
    // Diagnostic snapshot after the timed exit. Never queried between stages.
    // Per-pool page high-water marks are not a simultaneous physical-byte peak.
    let report = client.memory_report(MemoryScope::CurrentStream);
    let usage = report.usage();
    let peaks: u64 = report
        .streams
        .iter()
        .map(|s| {
            peak_pages(&s.pools)
                + s.auxiliary
                    .iter()
                    .map(|a| peak_pages(&a.pools))
                    .sum::<u64>()
        })
        .sum();
    println!(
        "# runtime_in_use={},runtime_reserved={},sum_pool_page_peaks={peaks}",
        usage.bytes_in_use, usage.bytes_reserved
    );
    eprintln!("memory_report={report:#?}");
    black_box(last);
    Ok(())
}
