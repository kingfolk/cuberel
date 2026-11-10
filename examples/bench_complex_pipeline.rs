//! Filtered wide queries, including exact fusion eligibility and checked Rust baseline.
//! Args: cpu|metal [rows_csv] [samples] [keep_percent] [widths_csv]
use cubecl::Device;
use cuberel::{
    Aggregate, AggregateValue, Batch, ColumnData, DataType, DeviceBatch, Executor, Field,
    FusionMode, HostColumn, Parameters, Pipeline, ResourceBudget, Schema, col, param, prepare,
};
use std::{error::Error, hint::black_box, time::Instant};

struct Fixture {
    x: Vec<i32>,
    y: Vec<i32>,
    z: Vec<i32>,
    xv: Vec<u32>,
    yv: Vec<u32>,
}
#[derive(Debug, PartialEq)]
struct Answer {
    rows: u32,
    values: Vec<Vec<i32>>,
    valid: Vec<Vec<u32>>,
    aggregates: Vec<AggregateValue>,
}
struct Case<'a> {
    fixture: &'a Fixture,
    batch: &'a DeviceBatch,
    query: Pipeline,
    params: Parameters,
    expected: Answer,
    width: usize,
    mixed: bool,
    limit: i32,
}
fn arg(x: i32, y: i32, z: i32, j: usize) -> i32 {
    let base = x.checked_add(y.checked_mul(j as i32 + 1).unwrap()).unwrap();
    base.checked_mul(base)
        .unwrap()
        .checked_add(z.checked_mul(j as i32 + 3).unwrap())
        .unwrap()
        .checked_add(y.checked_mul(z).unwrap())
        .unwrap()
        .checked_add(x.checked_mul(z).unwrap())
        .unwrap()
}
// Separate column-oriented oracle; the timed Rust path below uses a single row loop.
fn expected(f: &Fixture, width: usize, mixed: bool, limit: i32) -> Answer {
    let selected: Vec<_> = (0..f.x.len())
        .filter(|&i| f.xv[i] == 1 && f.x[i] >= limit && f.z[i] < 24)
        .collect();
    let columns: Vec<Vec<Option<i32>>> = (0..width)
        .map(|j| {
            selected
                .iter()
                .map(|&i| (f.yv[i] == 1).then(|| arg(f.x[i], f.y[i], f.z[i], j)))
                .collect()
        })
        .collect();
    let mut aggregates = vec![
        AggregateValue::Count(selected.len() as u64),
        AggregateValue::Count(columns[0].iter().flatten().count() as u64),
    ];
    for column in &columns {
        let sum = column
            .iter()
            .flatten()
            .map(|&v| i64::from(v))
            .reduce(|a, b| a + b);
        aggregates.push(AggregateValue::I64(sum));
        aggregates.push(AggregateValue::I32(column.iter().flatten().copied().max()));
    }
    Answer {
        rows: selected.len() as u32,
        values: if mixed {
            columns
                .iter()
                .map(|c| c.iter().map(|v| v.unwrap_or(0)).collect())
                .collect()
        } else {
            vec![]
        },
        valid: if mixed {
            columns
                .iter()
                .map(|c| c.iter().map(|v| u32::from(v.is_some())).collect())
                .collect()
        } else {
            vec![]
        },
        aggregates,
    }
}
fn rust(c: &Case<'_>) -> Answer {
    let f = c.fixture;
    let mut values = if c.mixed {
        vec![Vec::new(); c.width]
    } else {
        vec![]
    };
    let mut valid = if c.mixed {
        vec![Vec::new(); c.width]
    } else {
        vec![]
    };
    let mut sums = vec![0_i64; c.width];
    let mut maxima = vec![i32::MIN; c.width];
    let mut rows = 0;
    let mut count = 0_u64;
    for i in 0..f.x.len() {
        let (x, y, z) = (f.x[i], f.y[i], f.z[i]);
        if f.xv[i] == 0 || x < c.limit || z >= 24 {
            continue;
        }
        rows += 1;
        let good = f.yv[i] == 1;
        count += u64::from(good);
        for j in 0..c.width {
            let v = if good { arg(x, y, z, j) } else { 0 };
            if good {
                sums[j] += i64::from(v);
                maxima[j] = maxima[j].max(v);
            }
            if c.mixed {
                values[j].push(v);
                valid[j].push(u32::from(good));
            }
        }
    }
    let mut aggregates = vec![
        AggregateValue::Count(u64::from(rows)),
        AggregateValue::Count(count),
    ];
    for j in 0..c.width {
        aggregates.push(AggregateValue::I64((count != 0).then_some(sums[j])));
        aggregates.push(AggregateValue::I32((count != 0).then_some(maxima[j])));
    }
    Answer {
        rows,
        values,
        valid,
        aggregates,
    }
}
fn run_device(
    e: &Executor,
    c: &Case<'_>,
    mode: FusionMode,
) -> Result<(f64, bool, u64, u64), Box<dyn Error>> {
    let before = e.stats();
    let defines = e.define_count();
    let query = c.query.clone().fusion(mode);
    let start = Instant::now();
    let pending = e.execute_pipeline(&query, c.batch, &c.params)?;
    let result = pending.materialize()?;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    let fused = pending.graph().is_fused();
    if c.mixed {
        assert!(!fused);
        if mode == FusionMode::Auto {
            assert_eq!(
                pending.graph().fusion_reason(),
                Some("fusion supports aggregate-only fragments with at most one predicate")
            );
        }
    } else {
        assert_eq!(
            fused,
            mode == FusionMode::Required && c.batch.row_count() != 0
        );
    }
    assert_eq!(result.row_count, c.expected.rows);
    assert_eq!(
        result
            .aggregates
            .iter()
            .map(|a| a.value.clone())
            .collect::<Vec<_>>(),
        c.expected.aggregates
    );
    assert_eq!(result.projections.len(), c.expected.values.len());
    for (j, column) in result.projections.iter().enumerate() {
        let HostColumn::I32(column) = column.data() else {
            panic!("I32 output");
        };
        assert_eq!(column.validity(), c.expected.valid[j]);
        for (i, &v) in column.values().iter().enumerate() {
            if column.validity()[i] == 1 {
                assert_eq!(v, c.expected.values[j][i]);
            }
        }
    }
    black_box(result);
    let after = e.stats();
    assert_eq!(before.upload_calls, after.upload_calls);
    assert_eq!(
        after.allocation_bytes - before.allocation_bytes,
        pending.estimate().new_allocation_bytes()
    );
    Ok((
        ms,
        fused,
        e.define_count() as u64 - defines as u64,
        after.kernel_launches - before.kernel_launches,
    ))
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let backend = args
        .get(1)
        .ok_or("cpu|metal [rows_csv] [samples] [keep_percent] [widths_csv]")?;
    let device = match backend.as_str() {
        "cpu" => Device::cpu()?,
        "metal" => Device::metal(Default::default())?,
        _ => return Err("choose cpu or metal".into()),
    };
    let e = Executor::new(device.client())?.with_budget(ResourceBudget {
        max_device_bytes: 1024 * 1024 * 1024,
        max_output_rows: 4_194_304,
        max_stages: 4096,
    });
    let rows: Vec<usize> = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("65536,1048576")
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let samples: usize = args.get(3).map(String::as_str).unwrap_or("5").parse()?;
    let keep: i32 = args.get(4).map(String::as_str).unwrap_or("50").parse()?;
    let widths: Vec<usize> = args
        .get(5)
        .map(String::as_str)
        .unwrap_or("3,4,8")
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    if samples < 3
        || samples.is_multiple_of(2)
        || !(0..=100).contains(&keep)
        || widths.iter().any(|w| ![3, 4, 8].contains(w))
    {
        return Err("samples odd >=3, keep 0..100, width 3, 4 or 8".into());
    }
    println!(
        "# backend={backend},device={},tile={},samples={samples},warmups=1,debug={}",
        device.client().name(),
        e.cube_size(),
        cfg!(debug_assertions)
    );
    println!(
        "backend,rows,keep,width,projections,aggregates,selected,mode,sample,ms,fused,launches,reads,read_bytes,count_reads,allocation_bytes,device_bound,retained_bound,defines"
    );
    for n in rows {
        let f = Fixture {
            x: (0..n).map(|i| (i % 1000) as i32).collect(),
            y: (0..n).map(|i| ((i * 17) % 97) as i32).collect(),
            z: (0..n).map(|i| ((i * 7) % 31) as i32).collect(),
            xv: (0..n).map(|i| u32::from(i % 10 != 9)).collect(),
            yv: (0..n).map(|i| u32::from(i % 7 != 3)).collect(),
        };
        let host = Batch::new(n)
            .with_column(
                "x",
                ColumnData::NullableI32 {
                    values: &f.x,
                    validity: &f.xv,
                },
            )?
            .with_column(
                "y",
                ColumnData::NullableI32 {
                    values: &f.y,
                    validity: &f.yv,
                },
            )?
            .with_column("z", ColumnData::I32(&f.z))?;
        let batch = e.upload(&host)?;
        for &width in &widths {
            let mut fields = vec![
                Field::new("limit", DataType::I32),
                Field::new("zlimit", DataType::I32),
            ];
            let mut params = Parameters::new()
                .with("limit", 1000 - keep * 10)?
                .with("zlimit", 24_i32)?;
            for j in 0..width {
                fields.push(Field::new(format!("a{j}"), DataType::I32));
                fields.push(Field::new(format!("b{j}"), DataType::I32));
                params = params
                    .with(format!("a{j}"), j as i32 + 1)?
                    .with(format!("b{j}"), j as i32 + 3)?;
            }
            let schema = Schema::from_fields(
                &[
                    Field::new("x", DataType::I32).with_nullable(true),
                    Field::new("y", DataType::I32).with_nullable(true),
                    Field::new("z", DataType::I32),
                ],
                &fields,
            )?;
            let predicate = prepare(
                &col("x")
                    .ge(param("limit"))
                    .and(col("z").lt(param("zlimit"))),
                &schema,
            )?;
            let programs: Vec<_> = (0..width)
                .map(|j| {
                    let base = col("x") + col("y") * param(format!("a{j}"));
                    prepare(
                        &(base.clone() * base
                            + col("z") * param(format!("b{j}"))
                            + col("y") * col("z")
                            + col("x") * col("z")),
                        &schema,
                    )
                })
                .collect::<Result<_, _>>()?;
            for mixed in [false, true] {
                let mut query = Pipeline::new()
                    .filter(predicate.clone())
                    .aggregate("rows", Aggregate::CountStar)
                    .aggregate("valid", Aggregate::Count(programs[0].clone()));
                for (j, p) in programs.iter().enumerate() {
                    query = query
                        .aggregate(format!("sum{j}"), Aggregate::Sum(p.clone()))
                        .aggregate(format!("max{j}"), Aggregate::Max(p.clone()));
                    if mixed {
                        query = query.project(format!("p{j}"), p.clone());
                    }
                }
                let c = Case {
                    fixture: &f,
                    batch: &batch,
                    query,
                    params: params.clone(),
                    expected: expected(&f, width, mixed, 1000 - keep * 10),
                    width,
                    mixed,
                    limit: 1000 - keep * 10,
                };
                let eligible = !mixed
                    && e.estimate_pipeline(
                        &c.query.clone().fusion(FusionMode::Required),
                        &batch,
                        &c.params,
                    )
                    .is_ok();
                let modes = [
                    FusionMode::Unfused,
                    if eligible {
                        FusionMode::Required
                    } else {
                        FusionMode::Auto
                    },
                ];
                if !eligible {
                    eprintln!(
                        "eligibility,{backend},{n},{width},mixed={mixed},{:?}",
                        e.estimate_pipeline(
                            &c.query.clone().fusion(FusionMode::Required),
                            &batch,
                            &c.params
                        )
                        .unwrap_err()
                    );
                }
                // Required must reject unsupported outputs atomically, without fallback.
                if !eligible {
                    let before = e.stats();
                    assert!(
                        e.execute_pipeline(
                            &c.query.clone().fusion(FusionMode::Required),
                            &batch,
                            &c.params
                        )
                        .is_err()
                    );
                    assert_eq!(before, e.stats());
                }
                for mode in modes {
                    run_device(&e, &c, mode)?;
                }
                assert_eq!(rust(&c), c.expected);
                for sample in 0..samples {
                    // Rotate all three modes, including Rust, to reduce order bias.
                    for k in 0..3 {
                        let mode = (k + sample) % 3;
                        let before = e.stats();
                        let (ms, fused, defines, launches, label) = if mode == 2 {
                            let start = Instant::now();
                            let result = rust(black_box(&c));
                            let ms = start.elapsed().as_secs_f64() * 1000.;
                            assert_eq!(result, c.expected);
                            black_box(result);
                            (ms, false, 0, 0, "rust")
                        } else {
                            let (ms, fused, defines, launches) = run_device(&e, &c, modes[mode])?;
                            assert_eq!(defines, 0, "timed kernel definition");
                            (
                                ms,
                                fused,
                                defines,
                                launches,
                                if mode == 0 {
                                    "unfused"
                                } else if !eligible {
                                    "auto_fallback"
                                } else {
                                    "fused"
                                },
                            )
                        };
                        let after = e.stats();
                        let estimate = e.estimate_pipeline(
                            &c.query
                                .clone()
                                .fusion(if mode == 2 { modes[0] } else { modes[mode] }),
                            &batch,
                            &c.params,
                        )?;
                        println!(
                            "{backend},{n},{keep},{width},{},{},{},{label},{sample},{ms:.6},{fused},{launches},{},{},{},{},{},{},{defines}",
                            if mixed { width } else { 0 },
                            width * 2 + 2,
                            c.expected.rows,
                            after.readback_calls - before.readback_calls,
                            after.readback_bytes - before.readback_bytes,
                            after.count_readbacks - before.count_readbacks,
                            after.allocation_bytes - before.allocation_bytes,
                            if mode == 2 { 0 } else { estimate.device_bytes },
                            if mode == 2 {
                                0
                            } else {
                                estimate.retained_bytes
                            }
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
