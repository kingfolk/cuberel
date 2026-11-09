//! Warm resident end-to-end benchmark, including completion and scalar export.
//! Usage: bench_pipeline_fusion cpu|metal [rows_csv] [samples] [keep_percent] [plain|checked]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateResult, AggregateValue, Batch, ColumnData, DataType, DeviceBatch, Executor,
    Field, FusionMode, Parameters, Pipeline, Schema, col, param, prepare,
};
use std::{error::Error, hint::black_box, time::Instant};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}
fn run(
    e: &Executor,
    query: &Pipeline,
    batch: &DeviceBatch,
    params: &Parameters,
    expected: &[AggregateResult],
    expected_rows: u32,
) -> Result<f64, Box<dyn Error>> {
    // Time submission, planning, allocation, execution, sync, checks and scalar
    // readback. Exclude source upload, query preparation and result assertions.
    let start = Instant::now();
    let output = e.execute_pipeline(query, batch, params)?.materialize()?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(output.row_count, expected_rows);
    assert_eq!(output.aggregates, expected);
    black_box(output);
    Ok(elapsed)
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let backend=args.get(1).ok_or("usage: bench_pipeline_fusion cpu|metal [rows_csv] [samples] [keep_percent] [plain|checked]")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("choose cpu or native metal".into()),
    };
    let client = device.client();
    let e = Executor::new(client.clone())?;
    let rows: Vec<usize> = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("1024,65536,1048576")
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let samples: usize = args.get(3).map(String::as_str).unwrap_or("7").parse()?;
    let keep: i32 = args.get(4).map(String::as_str).unwrap_or("50").parse()?;
    let checked = match args.get(5).map(String::as_str).unwrap_or("plain") {
        "plain" => false,
        "checked" => true,
        _ => return Err("choose plain or checked".into()),
    };
    if samples < 3 || samples.is_multiple_of(2) || !(1..=100).contains(&keep) {
        return Err("samples must be odd and >=3; keep_percent must be 1..100".into());
    }
    println!(
        "# backend={backend}, device={}, tile={}, samples={samples}, warmup_pairs=2, keep_percent={keep}, checked={checked}, build_debug={}",
        client.name(),
        e.cube_size(),
        cfg!(debug_assertions)
    );
    let schema = Schema::from_fields(
        &[Field::new("x", DataType::I32).with_nullable(true)],
        &[
            Field::new("limit", DataType::I32),
            Field::new("one", DataType::I32),
        ],
    )?;
    let argument = if checked {
        prepare(&(col("x") + param("one")), &schema)?
    } else {
        prepare(&col("x"), &schema)?
    };
    let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
    let query = Pipeline::new()
        .filter(predicate)
        .aggregate("rows", Aggregate::CountStar)
        .aggregate("count", Aggregate::Count(argument.clone()))
        .aggregate("sum", Aggregate::Sum(argument.clone()))
        .aggregate("min", Aggregate::Min(argument.clone()))
        .aggregate("max", Aggregate::Max(argument));
    let off = query.clone().fusion(FusionMode::Unfused);
    let on = query.fusion(FusionMode::Required);
    let limit = 1000 - keep * 10;
    let params = Parameters::new().with("limit", limit)?.with("one", 1_i32)?;
    println!(
        "backend,rows,keep_percent,checked,selected,off_median_ms,on_median_ms,speedup,off_min_ms,off_max_ms,on_min_ms,on_max_ms,off_launches,on_launches,off_read_bytes,on_read_bytes"
    );
    for n in rows {
        let values: Vec<i32> = (0..n).map(|i| (i % 1000) as i32).collect();
        // Deterministic 10% NULLs, evenly distributed within each x bucket range.
        let flags: Vec<u32> = (0..n).map(|i| u32::from(i % 10 != 9)).collect();
        let kept: Vec<i32> = values
            .iter()
            .zip(&flags)
            .filter_map(|(&v, &valid)| (valid == 1 && v >= limit).then_some(v + i32::from(checked)))
            .collect();
        let count = kept.len() as u32;
        let expected = vec![
            AggregateResult {
                name: "rows".into(),
                value: AggregateValue::Count(u64::from(count)),
            },
            AggregateResult {
                name: "count".into(),
                value: AggregateValue::Count(u64::from(count)),
            },
            AggregateResult {
                name: "sum".into(),
                value: AggregateValue::I64(
                    (!kept.is_empty()).then(|| kept.iter().map(|&v| i64::from(v)).sum()),
                ),
            },
            AggregateResult {
                name: "min".into(),
                value: AggregateValue::I32(kept.iter().copied().min()),
            },
            AggregateResult {
                name: "max".into(),
                value: AggregateValue::I32(kept.iter().copied().max()),
            },
        ];
        let batch = e.upload(&Batch::new(n).with_column(
            "x",
            ColumnData::NullableI32 {
                values: &values,
                validity: &flags,
            },
        )?)?;
        for _ in 0..2 {
            run(&e, &off, &batch, &params, &expected, count)?;
            run(&e, &on, &batch, &params, &expected, count)?;
        }
        let definitions = e.define_count();
        let mut timings = [Vec::new(), Vec::new()];
        let mut metrics = [(0, 0); 2];
        for sample in 0..samples {
            // Alternate order to limit systematic cache/thermal/order bias.
            let order = if sample % 2 == 0 { [0, 1] } else { [1, 0] };
            for mode in order {
                let before = e.stats();
                let time = run(
                    &e,
                    if mode == 0 { &off } else { &on },
                    &batch,
                    &params,
                    &expected,
                    count,
                )?;
                let after = e.stats();
                assert_eq!(after.upload_calls, before.upload_calls);
                metrics[mode] = (
                    after.kernel_launches - before.kernel_launches,
                    after.readback_bytes - before.readback_bytes,
                );
                timings[mode].push(time);
                eprintln!("sample,{backend},{n},{keep},{checked},{sample},{mode},{time:.6}");
            }
        }
        assert_eq!(
            e.define_count(),
            definitions,
            "timed samples unexpectedly defined a kernel"
        );
        let off_median = median(&mut timings[0]);
        let on_median = median(&mut timings[1]);
        println!(
            "{backend},{n},{keep},{checked},{count},{off_median:.6},{on_median:.6},{:.4},{:.6},{:.6},{:.6},{:.6},{},{},{},{}",
            off_median / on_median,
            timings[0][0],
            timings[0][samples - 1],
            timings[1][0],
            timings[1][samples - 1],
            metrics[0].0,
            metrics[1].0,
            metrics[0].1,
            metrics[1].1
        );
    }
    Ok(())
}
