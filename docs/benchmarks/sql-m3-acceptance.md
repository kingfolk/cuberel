# M3 SQL acceptance: CPU/native Metal, 2026-11-22

F8 verifies M3 as an experimental, explicitly enabled SQL integration within
its published semantic/resource limits. **Keep the measured workloads on DataFusion
CPU for performance.** Metal SQL won none of the 26 matched warm-p50 comparisons
on this host/configuration. This evidence supports retaining explicit placement;
it does not establish an automatic threshold or justify expanding into general
group-by/join/top-k yet.

## Evidence and reproduction

Apple M1, 16 GiB, macOS 15.5 (24F74), Rust/Cargo 1.99.0, release profile. Library
base `227f98348f3b8a65906d54c54ed06f07f9a155ec`; the F8 harness sources are identified
by SHA-256 in the environment record. CubeCL remains pinned to
`5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, DataFusion 55.2.0 and Arrow 59.2.0.
CPU cube size is 8, native Metal 64. The declared Rust 1.95 minimum was not tested.

98 sequential fresh processes produced 588 verified samples: 78 SQL processes
(26 cases on three backends) and 20 direct DSL processes. Each process recorded
one first query and five warm queries. Builds/tests had finished before formal
sampling. No OS/driver cache reset or concurrent-query throughput measurement was
performed. Five-sample nearest-rank p95 is the observed maximum, not a production
tail estimate. Comparison/schema checks and fixture generation are outside timing.

```sh
cargo build --locked --offline --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/cuberel-f8-measurements --warm 5
```

Use a new output directory on rerun. The [measurement contract](../sql-acceptance.md)
defines fixtures, SQL, batch layout, exact result checks, feature commands and
counter/timer limitations. All cases returned correct values/NULLs, exact SQL
schemas/metadata and ordering where required; every device case selected actual
Cube operators (including CubeAggregateExec for global aggregates). Per-case
observed transfers/launches matched executor deltas and shared reservations drained
to zero after output release. No backend was substituted or silently retried.

Archived CSV uses LF record delimiters; parsed cells are unchanged from the
original capture. The environment record identifies this formatting normalization.

Raw evidence:

- [All 588 samples](sql-m3-2026-11-22-samples.csv).
- [All 98 summaries](sql-m3-2026-11-22-summary.csv).
- [Every command, optimized plan and final fragment metrics](sql-m3-2026-11-22-plans.txt).
- [Hardware, toolchain, binary/source hashes and invocation](sql-m3-2026-11-22-environment.json).
- [Completion marker](sql-m3-2026-11-22-complete.txt).

## Matched SQL size sweep

Times below are **prepare + execute/export**, in milliseconds, p50 / p95. Prepare
constructs the session, registers input, parses SQL and builds a fresh physical
plan each query. Device initialization is separate. Input is uploaded again on
every execution. Predicate threshold is 50%, requested NULL density 20%, one
fixture chunk/partition; actual passing rows are approximately 40% of input.
The projection returns four columns; aggregation returns one global row containing
SUM(y), COUNT(*), COUNT(y), MIN(y), MAX(y).

| Workload | Rows | DataFusion CPU | CubeCL CPU SQL | Native Metal SQL |
|---|---:|---:|---:|---:|
| project | 0 | 0.368 / 0.496 | 0.546 / 0.620 | 0.786 / 0.810 |
| project | 1,024 | 0.414 / 0.511 | 3.148 / 5.489 | 1.039 / 1.512 |
| project | 65,536 | 0.599 / 0.644 | 133.405 / 148.620 | 7.903 / 8.777 |
| project | 1,048,576 | 4.379 / 4.653 | 2559.467 / 3272.467 | 117.082 / 119.353 |
| project | 2,097,152 | 8.389 / 8.606 | 4664.880 / 4738.989 | 237.153 / 237.695 |
| aggregate | 0 | 0.754 / 0.793 | 1.009 / 1.106 | 1.043 / 1.298 |
| aggregate | 1,024 | 0.766 / 0.807 | 17.033 / 19.460 | 1.559 / 2.079 |
| aggregate | 65,536 | 0.943 / 1.017 | 288.033 / 333.083 | 8.916 / 10.796 |
| aggregate | 1,048,576 | 4.010 / 4.180 | 7853.011 / 13656.374 | 127.712 / 184.814 |
| aggregate | 2,097,152 | 6.846 / 7.168 | 8725.408 / 18818.094 | 246.017 / 249.001 |

The CPU-backend variance at large aggregation sizes is visible in the raw samples;
these observations are insufficient for a fitted cost model. DataFusion CPU also
wins the empty/small-input cases in this run.

## First process query and initialization

The 2M-row case below separates device initialization from SQL preparation and
first execution/export (ms). “First” may benefit from existing OS/driver caches;
it is not an isolated driver compilation measurement.

| Workload | Backend | Init | First prepare | First execute/export | First query total |
|---|---|---:|---:|---:|---:|
| project | datafusion | 0.000 | 1.524 | 7.903 | 9.427 |
| project | cpu | 0.419 | 1.669 | 4935.321 | 4936.991 |
| project | metal | 54.268 | 1.816 | 341.474 | 343.289 |
| aggregate | datafusion | 0.000 | 2.189 | 5.922 | 8.111 |
| aggregate | cpu | 0.464 | 2.176 | 9081.995 | 9084.171 |
| aggregate | metal | 52.649 | 2.669 | 390.430 | 393.099 |

Add init to first query total when assessing device startup plus a query. Process
startup/Tokio startup, data generation, validation and destruction are excluded.
Warm frontend define attempts were zero in all nonempty SQL size-sweep samples;
this establishes frontend reuse, not absence of driver compilation.

## Batching, transfers and observed costs

SQL retains DataFusion's default **8192-row execution batches** for all backends.
A 64K fixture chunk is delivered as eight nonempty batches plus an empty batch;
2M rows become 256 nonempty batches plus an empty batch. The aggregate remains
one global SQL answer. Larger fixture chunk count and source partitions in CSV
are input layout parameters, not a promise of runtime batch count.

The following are actual counters/group peaks from the final warm native Metal
sample at 2M rows; MiB uses 2^20 bytes. The group peaks are logical reservations,
not RSS or driver/cache memory. They exclude the independent oracle and upstream
DataFusion memory management.

| Workload | Upload MiB | Readback MiB | Launches | Selection count reads | Peak device MiB | Peak host MiB |
|---|---:|---:|---:|---:|---:|---:|
| project | 48.000 | 19.201 | 2816 | 256 | 0.501 | 36.046 |
| aggregate | 32.000 | 0.009 | 2816 | 0 | 0.130 | 18.265 |

Small aggregate readback does not imply few submissions: its metrics record
256 fused requests but 2816 actual launches, including device partial merges.
For both templates, the last sample spends most worker time in execute/export;
queue wait totals are below 0.03 ms without contention. Separate upload and
backend compilation timers are unavailable. Queue/prepare/worker/export timers
overlap, so they must not be added to reconstruct elapsed query time.

Direct DSL consumes each original fixture chunk without DataFusion's 8192-row
splitting. These diagnostics include host upload, Arrow adaptation/export and
checked scalar merging; they are **not** an isolated adapter-overhead comparison.
The shared-controller peaks are unavailable for DSL and left blank in CSV.

| Workload | Rows | DataFusion CPU SQL p50 | Metal SQL p50 | Direct Metal DSL p50 | Direct CubeCL CPU DSL p50 |
|---|---:|---:|---:|---:|---:|
| project | 65,536 | 0.599 | 7.903 | 1.380 | 133.899 |
| project | 1,048,576 | 4.379 | 117.082 | 22.543 | 2191.584 |
| project | 2,097,152 | 8.389 | 237.153 | 111.666 | 4368.792 |
| aggregate | 65,536 | 0.943 | 8.916 | 1.256 | 237.310 |
| aggregate | 1,048,576 | 4.010 | 127.712 | 13.433 | 1905.479 |
| aggregate | 2,097,152 | 6.846 | 246.017 | 25.153 | 3971.352 |

Fewer larger DSL requests can be faster than default SQL device execution,
but neither direct Metal template beats the matched DataFusion CPU SQL baseline
at the three nonempty sizes in this table. This does not predict the effect of changing
SQL batch size: that requires a new matched experiment with its resource/latency
tradeoffs measured explicitly.

## Selection, NULL, layout and CPU-boundary cases

All rows below use 65,536 input rows. Timings are ms. `chunks/parts` describes
the source fixture; the memory source may split chunks. `mixed` retains CPU
Sort/Limit above device projection and returns at most 17 ordered rows.

| Workload | Threshold % | NULL % | Chunks/parts | DataFusion CPU p50 | Metal SQL p50 | Metal SQL p95 |
|---|---:|---:|---|---:|---:|---:|
| project | 0 | 20 | 1/1 | 0.425 | 5.445 | 6.934 |
| project | 1 | 20 | 1/1 | 0.414 | 7.897 | 9.107 |
| project | 100 | 20 | 1/1 | 0.868 | 8.775 | 9.900 |
| project | 50 | 0 | 1/1 | 0.504 | 8.286 | 9.052 |
| project | 50 | 100 | 1/1 | 0.403 | 6.204 | 6.751 |
| project | 50 | 20 | 4/1 | 0.692 | 8.232 | 8.672 |
| project | 50 | 20 | 4/2 | 0.634 | 8.198 | 9.869 |
| aggregate | 0 | 20 | 1/1 | 0.798 | 7.013 | 8.155 |
| aggregate | 1 | 20 | 1/1 | 0.780 | 8.374 | 9.086 |
| aggregate | 100 | 20 | 1/1 | 1.098 | 8.071 | 8.949 |
| aggregate | 50 | 0 | 1/1 | 0.811 | 5.901 | 6.868 |
| aggregate | 50 | 100 | 1/1 | 0.885 | 7.156 | 8.039 |
| aggregate | 50 | 20 | 4/1 | 0.935 | 8.566 | 9.829 |
| aggregate | 50 | 20 | 4/2 | 0.992 | 8.829 | 9.737 |
| mixed | 50 | 20 | 1/1 | 0.950 | 8.103 | 8.718 |
| mixed | 50 | 20 | 4/2 | 1.003 | 8.869 | 9.384 |

## M3 verification

| Configuration/check | Result |
|---|---|
| `cargo test --locked --offline --all-features -- --test-threads=1` | 165 tests passed on actual CPU/native Metal |
| `cargo test --locked --offline --no-default-features --features datafusion` | 63 tests passed without linked runtime |
| `cargo test --locked --offline --no-default-features --features datafusion,cpu --test datafusion_acceptance` | 3 tests passed; 36 SQL scenarios on CPU plus oracle/fixed checks |
| `cargo test --locked --offline --no-default-features --features datafusion,metal --test datafusion_acceptance` | 3 tests passed on native Metal |
| `cargo test --locked --offline --no-default-features --features arrow` | 49 tests passed |
| `cargo check --locked --offline --no-default-features --all-targets` | Passed |
| `cargo clippy --locked --offline --all-features --all-targets -- -D warnings` | Passed |
| `RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --all-features --no-deps` | Passed |
| `cargo fmt --all -- --check`, `git diff --check` | Passed |
| Release quick/full driver | 18/98 processes; 36/588 verified samples |

Checks used `/private/tmp/cuberel-build.hmThUI` as `--target-dir`; release used
the separate path in the reproduction command. The existing nonfatal macOS
unwind-table linker warning remains. An earlier test run concurrent with heavy
dependency compilation exceeded an existing 30-second CPU cancellation deadline.
The final complete serial run passed with that test/deadline unchanged. Existing
lifecycle/diagnostic suites cover cancellation, drain, shared budgets and errors;
the release timings do not measure concurrent-query contention.

## Enablement boundary and next evidence

| Category | Decision |
|---|---|
| Suitable for explicit enablement | Bounded semantic/backend-integration experiments within the published allow-list; no measured production speedup candidate here |
| Continue DataFusion CPU | All 26 matched SQL fixture/configuration cases on this M1, including low selectivity, all-NULL, multi-batch/partition and CPU Sort/Limit |
| Not measured | Larger SQL batch-size configurations, other hardware, concurrent throughput/tails and application-specific longer supported fragments |

The next experiment should compare **matched DataFusion SQL batch sizes** while
holding SQL, schema, data and backend fixed, then profile the Arrow adaptation,
submission/completion and export costs identified here. Keep shared-budget,
backpressure and error boundaries intact; the adapter must not silently rebatch
input to hide resource failures. Pick a real benefiting workload before choosing
an M4 operator. General group-by/join/top-k, broader SQL types/casts and cross-query
residency retain their separate scope rather than becoming hidden M3 tail work.
