# M4 opt-in scalar readback batching: 2026-11-22

At the predeclared 1M-row/64K-batch fixture, Batched reduces median completed-query
time by about 13% versus Scalar, meeting the 10% recommendation gate in all three
process comparisons. The benefit shrinks at larger batches; the 1M-batch result
is not reliable across repeats. Scalar stays the default. DataFusion CPU remains
faster at every measured source size, so M4's CPU-speedup goal remains unmet.

## Same-binary contract and environment

This follows the [predeclared phase contract](../sql-phase-profiling.md) and
[phase evidence](sql-m4-phase-study.md). The existing synthetic filtered integer
aggregate has 1,048,576 rows, 50% predicate threshold, 20% requested NULL density,
one source chunk/partition and exact independent schema/NULL/value checks. Each
source size (64K/256K/1M) runs DataFusion CPU once and Metal Scalar/Batched in each
of three fresh-process repeats, with ten warm samples plus one first query.
Backend and mode order rotate. All 27 processes/297 samples succeeded, selected
actual CubeAggregateExec for Metal, matched explicit counters and drained shared
reservations. No compilation/test workload ran during performance sampling.

Apple M1/16 GiB, macOS 15.5, Rust/Cargo 1.99, release, fixed dependencies. Both
Metal modes use the same binary, host inputs, pipeline, controller and timing
scopes; mode changes only aggregate host read scheduling. Source/binary hashes
and an exact source patch against base `e6b9995` are archived for reproducibility
of the measured uncommitted implementation. Apply it to a clean checkout of that
base with `git apply --unidiff-zero <source.patch>` when reproducing the recorded
binary sources. The final code relocates only the counter accessor into its
existing impl for lint cleanliness; the exact measured layout remains in the patch. Driver/OS caches were not
cleared. First/init costs are preserved but excluded from warm query time.

## Complete-query results (ms)

Each cell is the median of three per-process warm p50s, not a pooled percentile.
Parentheses show their min–max. Percent is the decrease in whole-query time from
Scalar to Batched; it includes session/plan preparation, host upload, completed
execution, global host merge and final export. It excludes input/oracle creation,
validation and initialization. Ten-sample p95 remains a small-sample maximum.

| Source rows | CPU | Metal Scalar (range) | Metal Batched (range) | Time decrease |
|---|---:|---:|---:|---:|
| 65,536 | 3.423 | 21.114 (20.648–21.191) | 18.353 (18.167–18.564) | 13.1% |
| 262,144 | 3.396 | 14.633 (14.621–14.719) | 13.634 (13.625–13.672) | 6.8% |
| 1,048,576 | 3.809 | 12.899 (12.361–12.966) | 12.677 (12.588–12.687) | 1.7% |

At 64K, each repeat improves by more than 10% (about 12–13%). At 256K, improvement
is consistently about 7%, below the predeclared recommendation threshold. At 1M,
the first repeat regresses and process-p50 ranges overlap; the ~2% median
difference is not promoted as reliable benefit. Do not compare this CPU baseline
with prior runs to fit a placement threshold. These are sequential-query results
on one host, not concurrency or other-hardware predictions.

## Identical buffers/bytes/kernels, fewer API calls

The final warm counters agree across repeats. Matched modes have identical
output, uploads, readback buffer descriptors/bytes, allocation calls/bytes,
logical host/device peaks, kernel launches and selection-count reads.

| Source rows | Read API calls Scalar → Batched | Buffer descriptors (both) | Read bytes (both) | Launches (both) |
|---|---:|---:|---:|---:|
| 65,536 | 144 → 48 | 144 | 576 | 176 |
| 262,144 | 36 → 12 | 36 | 144 | 44 |
| 1,048,576 | 9 → 3 | 9 | 36 | 16 |

Batched keeps the existing participating-row/error completion checks, reads count
planes together, validates them, and then reads only valid payload planes
together. It does not pack data into a new device buffer, add a kernel, cache a
result or maintain cross-batch device state. Each call can still read several
physical resources. API-call reduction must not be described as fewer buffers or
physical copies. The phase timers include pending GPU work and decoding, not
only API overhead; use the whole-query comparison as the outcome evidence.

## Enablement and remaining work

Use [the explicit mode API](../aggregate-readback.md) for experiments matching
the demonstrated boundary. Default Scalar and all current SQL type/shape rules
remain unchanged. F32 tests cover the direct DSL, not new SQL F32 support. The
M4 source-size memory tradeoff still applies. The full query remains slower than
CPU; remaining Arrow/encoding, submit/merge launches and application fragments
need independent evidence. Cross-batch device accumulation stays deferred.

## Reproduce and raw evidence

```sh
cargo build --locked --offline --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/cuberel-m4-readback-study --batch-study \
  --study-rows 1048576 --batch-sizes 65536 262144 1048576 \
  --backends datafusion metal --readback-modes scalar batched --repeats 3 --warm 10
```

Use a new output directory. The default F8 runner still uses Scalar and retains
its SQL/DSL matrix; nondefault readback modes require the explicit study mode.

- [297 samples](sql-m4-readback-2026-11-22-samples.csv).
- [27 process summaries](sql-m4-readback-2026-11-22-summary.csv).
- [Commands/plans/metrics](sql-m4-readback-2026-11-22-plans.txt).
- [Environment and all library-source hashes](sql-m4-readback-2026-11-22-environment.json).
- [Exact measured source patch](sql-m4-readback-2026-11-22-source.patch).
- [Completion marker](sql-m4-readback-2026-11-22-complete.txt).

## Final verification (2026-11-23)

The final stack passed 176 all-feature tests on actual CPU/native Metal and 66
SQL-only tests. New fixed-result regressions cover resident, selected and fused
aggregates, exact names/values/call-byte-buffer counts, empty/all-NULL payload
suppression, F32 canonical bits, checked overflow/filter suppression, repeated
reads and captured mode/kernel identity. Existing malformed-count, high-word
backend failure, shape validation, SQL globality/late errors and lifecycle
contracts also ran with Batched. No-feature all-target check, strict all-target
Clippy/rustdoc, fmt/diff passed. The original default SQL/DSL quick workflow
completed 11 processes and 22 verified samples with the final release build.
The known nonfatal macOS unwind-table linker warning remains. The measured source
patch was reapplied in an isolated temporary checkout and matched all archived
library/harness/manifest hashes exactly.
