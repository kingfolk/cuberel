# Complex filtered pipeline benchmark — 2026-11-10

This extends the F7 simple-query acceptance workload without changing the engine.
Library base: `05223ca`; example `bench_complex_pipeline.rs` is a local addition.
Apple M1 / 16 GiB / macOS 15.5, Rust 1.99.0, release, pinned CubeCL
`5037c0639ce3b1205392bdbe43dcf25e2ea09e29`. CPU tile 8, native Metal tile 64.

## Workload and a supported fusion example

Three columns: x[i]=i%1000, y[i]=(17*i)%97, z[i]=(7*i)%31.
x is NULL when i%10==9; y is NULL when i%7==3; z is non-null.
The single prepared predicate is `x >= 1000 - 10*keep AND z < 24`.
`keep` is a nominal x threshold percentage, not final selectivity: both NULLs
and the z condition affect output. At N=1,048,576 and keep=50, 365,162 rows pass.

For j from zero to width-1, each checked I32 expression is:

```text
q_j = (x + (j+1)*y) * (x + (j+1)*y) + (j+3)*z + y*z + x*z
```

Expressions stay safely inside I32 for this bounded fixture; SUM widens to I64.
The aggregate list is COUNT(*), COUNT(q0), then SUM(q_j), MAX(q_j) for each j.
The mixed query additionally exports every q_j as a compact nullable projection.

**A supported Metal fusion case has width 3 and eight scalar aggregates:**

```text
WHERE x >= 500 AND z < 24
COUNT(*), COUNT(q0), SUM(q0), MAX(q0),
                    SUM(q1), MAX(q1), SUM(q2), MAX(q2)
```

It runs with `FusionMode::Required`; the benchmark asserts the graph is actually
fused for nonempty input. The three expressions share subexpressions, so fusion
can also share argument work rather than materializing repeated arrays.

For widths 4/8 (10/18 aggregates), native Metal rejects Required with
`fused buffer binding limit exceeded`; Auto runs the same reference shape as
Unfused. CPU supports those aggregate-only fused shapes. This is not a universal
"eight aggregates maximum": input value/validity planes, checked error planes,
aggregate state widths and hidden row-count state all consume bindings. Different
queries can hit the limit earlier or later. No binding ABI or fusion capability
was expanded for this benchmark.

All mixed projection/aggregate shapes reject Required because current fusion
supports aggregate-only fragments. The Auto mixed timings below are explicitly
fallback measurements, not fusion gains.

## Method

Each case uploads its source once and prepares the query before timing. One
warm-up per device mode precedes samples. Metal uses five samples per mode, CPU
three. Rotate the order of Unfused, Required/Auto and Rust on successive samples.
Timer covers planning, allocations, submission, completion, error checks and
all host results, excluding source upload, query cloning, assertions and metadata
diagnostics. Rust uses one row loop with checked arithmetic, shared selection,
scalar states and (when needed) projected value/validity vectors. It is a direct
Rust baseline, not DataFusion or an assertion of the best possible CPU code.

An independent column-oriented host oracle validates every aggregate, selected
row count, output validity and each valid projected payload. Undefined NULL
payload bytes are not compared. Every measured query has zero uploads and zero
new kernel definitions, and requested allocation bytes exactly match its estimate.
No kernel-only timing or physical device peak is claimed.

## Metal, nominal keep 50% (completed-query median)

| N | Expressions | Projections | Aggregates | Unfused ms | Fused / Auto ms | Actual fusion | Rust ms |
|---:|---:|---:|---:|---:|---:|---|---:|
| 65,536 | 3 | 0 | 8 | 1.909 | 1.494 | yes | 0.310 |
| 65,536 | 3 | 3 | 8 | 2.759 | 2.722 | no (fallback) | 0.384 |
| 65,536 | 4 | 0 | 10 | 2.920 | 2.497 | no (fallback) | 0.369 |
| 65,536 | 4 | 4 | 10 | 3.341 | 3.459 | no (fallback) | 0.498 |
| 65,536 | 8 | 0 | 18 | 4.202 | 4.470 | no (fallback) | 0.610 |
| 65,536 | 8 | 8 | 18 | 5.890 | 5.776 | no (fallback) | 1.419 |
| 1,048,576 | 3 | 0 | 8 | 8.600 | 6.105 | yes | 4.817 |
| 1,048,576 | 3 | 3 | 8 | 13.164 | 12.673 | no (fallback) | 6.980 |
| 1,048,576 | 4 | 0 | 10 | 10.855 | 10.944 | no (fallback) | 5.805 |
| 1,048,576 | 4 | 4 | 10 | 16.234 | 15.989 | no (fallback) | 8.792 |
| 1,048,576 | 8 | 0 | 18 | 17.904 | 17.844 | no (fallback) | 9.641 |
| 1,048,576 | 8 | 8 | 18 | 29.978 | 29.183 | no (fallback) | 29.602 |

## Metal, nominal keep 1%

| N | Expressions | Projections | Aggregates | Unfused ms | Fused / Auto ms | Actual fusion | Rust ms |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1,048,576 | 3 | 0 | 8 | 5.153 | 5.071 | yes | 1.533 |
| 1,048,576 | 3 | 3 | 8 | 5.950 | 5.810 | no (fallback) | 1.568 |
| 1,048,576 | 4 | 0 | 10 | 6.812 | 6.254 | no (fallback) | 1.531 |
| 1,048,576 | 4 | 4 | 10 | 7.178 | 6.436 | no (fallback) | 1.572 |

## Metal, nominal keep 100%

| N | Expressions | Projections | Aggregates | Unfused ms | Fused / Auto ms | Actual fusion | Rust ms |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1,048,576 | 3 | 0 | 8 | 14.750 | 5.119 | yes | 8.080 |
| 1,048,576 | 3 | 3 | 8 | 28.029 | 30.324 | no (fallback) | 13.599 |
| 1,048,576 | 4 | 0 | 10 | 16.610 | 15.941 | no (fallback) | 10.011 |
| 1,048,576 | 4 | 4 | 10 | 28.974 | 29.998 | no (fallback) | 16.348 |

## CPU reference runtime, nominal keep 50%

| N | Expressions | Projections | Aggregates | Unfused ms | Fused / Auto ms | Actual fusion | Rust ms |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1,024 | 4 | 0 | 10 | 68.455 | 31.358 | yes | 0.008 |
| 1,024 | 4 | 4 | 10 | 68.308 | 92.320 | no (fallback) | 0.016 |
| 1,024 | 8 | 0 | 18 | 153.347 | 61.536 | yes | 0.012 |
| 1,024 | 8 | 8 | 18 | 152.315 | 144.175 | no (fallback) | 0.033 |
| 65,536 | 4 | 0 | 10 | 969.075 | 261.767 | yes | 0.366 |
| 65,536 | 4 | 4 | 10 | 1021.201 | 968.238 | no (fallback) | 0.509 |

## Interpretation

At 1M rows, the supported eight-aggregate Metal query improves from 8.600 ms
unfused to 6.105 ms fused (1.41x), with 44 to 25 launches. Direct Rust is still
faster at 4.817 ms. More arithmetic and aggregation do not automatically make
this GPU path preferable once completion and readback are included.

The widest mixed query (8 projections, 18 aggregates) at 1M rows is close to Rust:
Auto fallback 29.183 ms, Unfused 29.978 ms, Rust 29.602 ms. Those two device modes
have the same physical plan; their difference is run variability, not fusion.
A small difference around parity is not evidence of a robust speed advantage.
The CPU CubeCL path remains much slower than direct Rust despite fusion reducing
its own cost. CPU tests deliberately cover 1K/64K rather than a full 1M matrix.

These are short local measurements on correlated synthetic data, not confidence
intervals or deployment thresholds. Cross-width comparisons also come from
separate processes and different run times. The per-mode min/max and raw samples
are retained; only matched query shapes should be used for fusion comparisons.

## Reproduce and correctness boundaries

```sh
CARGO_TARGET_DIR=/private/tmp/cuberel-complex cargo build --release --offline --all-features --example bench_complex_pipeline
/private/tmp/cuberel-complex/release/examples/bench_complex_pipeline metal 65536,1048576 5 50 3,4,8
/private/tmp/cuberel-complex/release/examples/bench_complex_pipeline metal 1048576 5 1 3,4
/private/tmp/cuberel-complex/release/examples/bench_complex_pipeline metal 1048576 5 100 3,4
/private/tmp/cuberel-complex/release/examples/bench_complex_pipeline cpu 1024 3 50 4,8
/private/tmp/cuberel-complex/release/examples/bench_complex_pipeline cpu 65536 3 50 4
```

Separate real Metal checks cover N=0,1,1025 with keep=0, widths 3/4/8; Metal and CPU
also cover N=0,1,1025 with keep=50, width 3. These exercise empty, all-filtered,
tails, NULL arguments, empty aggregates, multilevel work and Required rejection
without side effects. Empty supported queries need no fused kernel launch.
Strict example Clippy with warnings denied and `cargo fmt --check` pass.

- [Individual performance samples](complex-pipeline-samples.csv)
- [Medians, ranges, transfer and allocation counters](complex-pipeline.csv)
- [Boundary verification observations](complex-pipeline-boundaries.csv)
- [Runtime metadata and explicit fusion rejection reasons](complex-pipeline-eligibility.txt)
