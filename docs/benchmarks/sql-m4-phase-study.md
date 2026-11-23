# M4 worker phase attribution: 2026-11-22

Completion/export is the largest observed worker phase for the primary 1M-row
filtered integer aggregate at 64K source batches. This justifies testing batched
scalar-state reads while retaining the scalar reference. It does not reopen
cross-batch device accumulation or establish a GPU kernel bottleneck.

## Reproduce and measurement contract

The [predeclared contract](../sql-phase-profiling.md) selects 1,048,576 rows,
threshold 50%, NULL density 20%, one source chunk/partition and session batches
64K/256K/1M. Eighteen sequential fresh processes (two backends, three settings,
three repeats) produced 198 verified samples. Every process has one first and ten
warm queries, rotating backend order across repeats. The fixture and exact
independent schema/NULL/value oracle are unchanged. All samples validated, actual
device placement/transfer counters matched, and shared reservations drained.

Apple M1/16 GiB, macOS 15.5, Rust/Cargo 1.99, release, base main 3b62f41 plus the
phase harness/core changes identified in source/binary metadata. Dependencies
remain pinned. Build/tests stopped before timing. Driver/OS caches were not
cleared. No application trace, concurrency throughput or hardware timing claim.

## Native Metal host-wall intervals (ms)

Query is the median of three process warm-p50s, with their min–max in parentheses.
Other columns are medians of the thirty individual warm metric observations.
They must not be added: medians come from potentially different observations,
and worker is a parent interval. The single input partition avoids summing
overlapping partition walls in this fixture. The complete/export interval can
include GPU work submitted earlier; it is not a readback-only timer.

| Source rows | Whole query (process-p50 range) | Worker | Arrow adapt | Resource preflight | Upload/submit | Complete/export |
|---|---:|---:|---:|---:|---:|---:|
| 65536 | 20.632 (20.558–21.211) | 19.562 | 2.895 | 0.246 | 3.635 | 12.613 |
| 262144 | 14.689 (14.641–14.736) | 13.432 | 2.774 | 0.080 | 4.443 | 6.020 |
| 1048576 | 12.992 (12.883–13.040) | 11.840 | 3.108 | 0.040 | 5.373 | 3.124 |

At 64K source rows, complete/export is about 64% of the worker median, comfortably
above the predeclared 10% investigation gate. This is evidence to investigate the
boundary, not a causal allocation of all that time to host reads. The phase also
contains backend completion, error checks, scalar reads and decoding. At larger
batches its relative cost falls while Arrow adaptation and upload/submit remain.

The pinned CubeCL Metal implementation resolves all descriptors in a `read`
request, flushes/waits once for the request, then copies its requested resources.
Its public client supports a vector of handles. This is an implementation basis
for grouping small scalar reads, not evidence that all buffers become one
physical copy or that every backend has the same cost. Preserve the executable
scalar mode and record both read API calls and buffer descriptors.

The proposed next layer reads counts together, checks them against participating
rows, then reads only valid payloads together. Row-count/error completion checks
retain their existing order; NULL payloads are not fetched speculatively. No new
device kernel, aggregate tree, input cache or per-stream accumulator is required.
Independent CPU comparisons and a same-binary scalar/batched matrix determine
whether the optimization meets the 10% whole-query recommendation criterion.

Whole-query times from this run must not be compared with an earlier run to
claim the profiler itself improved performance. Instrumentation has overhead;
the optimized/reference comparison must share it. CPU-only phase fields are
unavailable, not zero; its independent query time remains in the raw files.

## Verification and raw evidence

Real CPU/native Metal diagnostics tests pass, including the nesting invariants:
adapt+preflight is contained by prepare; submit+complete by execute/export; all
four children are contained by worker. Existing metric identity/placement and
CPU-only behavior remain covered. The release matrix validated every result.

- [198 samples](sql-m4-phases-2026-11-22-samples.csv).
- [18 process summaries](sql-m4-phases-2026-11-22-summary.csv).
- [Commands, plans and final metrics](sql-m4-phases-2026-11-22-plans.txt).
- [Environment and source/binary identity](sql-m4-phases-2026-11-22-environment.json).
- [Completion marker](sql-m4-phases-2026-11-22-complete.txt).
