# SQL acceptance and end-to-end measurements

M3's SQL adapter remains experimental and explicitly enabled. The F8 harness
compares DataFusion CPU with the pinned CubeCL CPU and native Metal backends,
using the same SQL, Arrow input, session settings and complete host-input/export
path. It also measures equivalent direct DSL pipelines. No automatic placement
threshold is installed. See the [measured report](benchmarks/sql-m3-acceptance.md)
for the tested workload boundary and raw evidence.

## Reproduce

```sh
cargo build --locked --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/sql-m3-measurements --warm 5

# Short smoke matrix; use a new output directory on each run.
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/sql-m3-smoke --quick --warm 1

# A single process: backend, path, workload, rows, threshold %, NULL %, chunks,
# source partitions, number of additional warm samples.
/private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  metal sql aggregate 65536 50 20 4 2 5

cargo test --locked --all-features --test datafusion_acceptance
cargo test --locked --no-default-features --features datafusion \
  --test datafusion_acceptance
```

The runner uses only Python's standard library, creates a new output directory,
and fails on an unavailable backend or a failed case. Restrict `--backends` when
intentionally measuring a subset. The binary also runs DataFusion alone with
`--no-default-features --features datafusion`; device requests then fail rather
than silently changing backends. Native Metal needs actual Apple GPU access.

Each process writes sample CSV to stdout and its SQL, optimized physical plan
and final per-fragment metrics to stderr. The runner records every command,
hardware/toolchain/environment, per-process CSV/log, combined `samples.csv`,
`summary.csv`, and a `COMPLETE` marker only after every case succeeds. Keep the
full directory when collecting new evidence. The checked-in report includes
combined samples, summary and all per-case plan/metrics logs.

## Fixtures and correctness

The two main SQL workloads are:

```sql
SELECT id,x AS value,y AS payload,y IS NULL AS missing
FROM t WHERE x < CAST(50 AS INT);

SELECT sum(y) AS total,count(*) AS n,count(y) AS valid,min(y) AS lo,max(y) AS hi
FROM t WHERE x < CAST(50 AS INT);
```

The mixed workload adds `ORDER BY payload ASC NULLS LAST,id LIMIT 17` to the
projection. Sort/Limit stay in DataFusion and consume the device fragment's
Arrow output. Multi-partition aggregation must still return one global SQL row.

`id` is a unique non-null Int32 row number. For row `i`, `x=(37*i)%100` and
`y=(31*i)%2001-1000`. The x validity score is
`(53*i+19*(i/100)+17)%100`; y uses `(29*i+11*(i/100)+7)%100` (integer division).
A score below the requested NULL percentage means NULL. The predicate threshold
is a nominal selection percentage before NULL filtering, not a promise of an
exact output count. CSV records actual output rows. All numeric values and SUM
bounds fit the adapter's supported types/ranges.

Every fixture starts three values into padded arrays, preserving nonzero value
and validity offsets. Requested chunks are assigned round-robin to source
partitions; an explicit empty batch follows each chunk. `batches` in CSV means
non-padding chunks, so the source contains twice that many RecordBatches.
Input schema metadata is retained. Generation and independent expected-result
construction happen outside the measured query.

The SQL sessions retain DataFusion's default `batch_size=8192`. Its memory source
splits larger fixture chunks at execution, equally for DataFusion CPU and both
SQL device backends. The direct DSL consumes fixture chunks unchanged. For
example, the 64K/one-chunk SQL case actually pulls eight nonempty 8192-row batches
plus one empty batch; DSL pulls one nonempty batch plus one empty batch. Inspect
`input_batches`/`output_batches` in the saved per-case metrics. SQL-versus-DSL
differences include this upstream batching and scheduling, and are not an isolated
measurement of adapter overhead. Larger SQL batch sizes require a separate
matched experiment; this harness deliberately retains the ordinary defaults.

Each sample checks exact output schema (including names, order, types,
nullability and metadata), values, NULLs and row counts against an independent
host calculation. The SQL baseline goes through the same check. Projection
comparison sorts by unique ID only when order is unspecified; the mixed SQL
checks its specified ordering directly. Integration tests additionally check
single-partition stable projection order and fixed hand-computed aggregates.
Device samples require actual CubeExec/CubeAggregateExec nodes and compare
observed transfer/launch metrics with the dedicated executor counters. After
releasing consumer output, all shared host/device/request/stream reservations
must be zero, with no rejected reservations or poisoned group.

## Measurement contract

- Each backend/path/case starts a fresh process. The `first` sample includes
  first kernel definition/compilation encountered in that process, but OS and
  driver/disk caches are **not cleared**. It is not a guaranteed cold driver
  compile. Device initialization is reported separately as `init_ms`.
- Each sample constructs a new session, registers the same host table, parses
  SQL and creates a new optimized physical plan. `prepare_ms` includes these
  steps. New plans avoid reused exchange streams and accumulated plan metrics.
  A dedicated executor is reused within the process for warm kernel reuse.
- `execute_ms` covers collection through completed CPU/Metal execution and host
  Arrow output, including upload, count/error/state readbacks, asynchronous
  worker admission and SQL host scalar merging. There is no cross-query input
  residency. `total_ms=prepare_ms+execute_ms`; add first `init_ms` for device
  startup plus query cost. Process startup, fixture generation, comparison,
  logging and output destruction are excluded.
- DSL timings include program preparation, per-batch Arrow adaptation, ordinary
  host upload, completed device execution, Arrow export and checked scalar merge.
  DSL COUNT's UInt64 is explicitly converted to SQL Int64 and input schema
  metadata is restored for comparison. DSL executes batches sequentially and
  has no DataFusion scheduler or shared controller; it is a diagnostic
  comparison, not the SQL baseline. Its shared peaks are blank, and its maximum
  per-request logical estimate is separate.
- Upload/readback bytes, calls, launches, selection-count reads, requested
  allocation bytes and define attempts are actual executor deltas. Define
  attempts are not backend compilation counts. CPU baseline has zero device
  counters. F7 queue/prepare/execute-export/worker times overlap and cannot be
  added to infer elapsed time. Separate upload/driver-compile timers are
  unavailable; complete export time includes their costs.
- Fresh shared controllers provide per-query logical group peaks, including
  retained consumer output. These exclude RSS, upstream exchanges, driver
  metadata and runtime allocator caches. `max_request_device_bytes` takes the
  maximum request estimate across all fragments/partitions, not their sum.
- Warm p50/p95 use nearest rank and preserve all samples. With five warm samples,
  p95 is their maximum and is only a small-sample observation, not a reliable
  production tail estimate. Runs are sequential without concurrent queries;
  queue times do not characterize contention. Shared-budget/cancellation tests
  cover concurrency separately.

## M3 regression evidence

F8 complements the existing semantic, property and lifecycle suites; it does not
replace them. The public contracts and their executable evidence are:

| Contract | Evidence |
|---|---|
| Runtime-free registration, pinned dependencies, disabled fallback | `datafusion_registration`, SQL-only feature run |
| Typed allow-list, NULL/3VL, checked effects, rejected casts/types | `datafusion_capabilities`, `datafusion_execution` |
| Fragment composition, CPU boundaries, child property rebuild, rule re-entry | `datafusion_fragments`, `datafusion_execution` |
| Global shape, empty/all-NULL, COUNT types, widening SUM, bounds/multi-partition final | `datafusion_global`, `datafusion_acceptance` |
| Slow consumers, detached workers, cancel/drop/admission, failure drain and reuse | `datafusion_lifecycle`, library exec/resources/global tests |
| Explicit policies, observations, bounded registration and restored CPU plans | `datafusion_diagnostics` |
| CPU oracle/native CPU/Metal, tile edges, sliced arrays, mixed Sort/Limit | `datafusion_acceptance` plus the release matrix |

F32 SQL aggregation, arbitrary casts, general I64/F64 inputs, host pass-through
gather and cross-query residency remain outside this adapter's support. Failure
or rejection never authorizes silently retrying a device query on CPU. Future
operator work should start from a measured workload, not from a kernel-only
speedup or this fixture alone.
