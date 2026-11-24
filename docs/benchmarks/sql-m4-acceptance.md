# M4 SQL execution evaluation: 2026-11-23

M4 improves the measured Metal execution path, but its CPU-performance gate is
**not met**. In this extended study DataFusion CPU wins all 11 matched workload /
session-size comparisons. Increasing total input to 8M rows does not reverse the
result. Keep the measured SQL on CPU for performance; device placement, source
batch tuning and aggregate readback mode remain explicit experimental choices.
This is a completed evaluation of the selected M4 implementation, not a claim
that every milestone exit condition passed.

## Scope and reproducibility

The [saved profile contract](../sql-acceptance-profiles.md) was committed before
sampling at `e69eaf183ef1857585b76e67167d797dbf828b84`, based on merged M4 #34/#35
main `62071d0`. The library is unchanged from that merged implementation. The
binary and all library/harness/manifest hashes are archived; every recorded
source hash was checked against that commit. No uncommitted source patch is
needed. Dates use America/New_York; the archived UTC run began on November 24.

Apple M1 / 16 GiB, macOS 15.5, Rust/Cargo 1.99, release. CubeCL fork remains pinned
to `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, DataFusion 55.2.0, Arrow 59.2.0.
The declared Rust 1.95 minimum was not independently tested. No OS/driver caches
were cleared, and no builds/tests or competing device work ran during sampling.
The shared adapter caps are fixed at 256 MiB host / 64 MiB device, with default
request/stream limits. Source/oracle memory, RSS, upstream exchanges and runtime
caches are outside these logical caps.

99 sequential fresh processes produced 1,089 verified samples: three process
repeats of DataFusion CPU and Metal Scalar/Batched, one first query plus ten warm
queries each. Backends and modes rotate between repeats. The CPU baseline runs
once per case/size/repeat. Timings include session construction, SQL/physical-plan
preparation, host input/upload, completed execution, host global merging and
Arrow export; they exclude initialization, fixture/oracle construction, checking
and output destruction. First/init costs are retained in raw evidence.

All fixtures remain **synthetic**. There is no real application trace, latency
SLO or general-purpose SQL speedup claim. Every query uses the existing fixed
SQL, schema metadata, sliced arrays, inserted empty batches and independent
exact-result oracle. Workloads are:

| Case ID | Input and boundary |
|---|---|
| primary | 1,048,576 rows, threshold 50%, requested NULL 20%, one source chunk/partition |
| scale8m | 8,388,608 rows; otherwise primary; total input changes independently of request size |
| layout-two-partitions | 1M rows split into eight chunks / two source partitions; global result still one row |
| full-select-no-null | 1M rows, threshold 100%, NULL 0% |
| low-select | 1M rows, threshold 1%, NULL 20% |
| project-output | 65,536 rows, threshold 50%, NULL 20%; full four-column host output |
| mixed-cpu-boundary | Same projection with CPU Sort/Limit, at most 17 ordered output rows |

## Complete-query comparison

Times are milliseconds, each the median of three **per-process warm p50s**.
The last column is the time decrease from Scalar to Batched at the same size;
it does not compare against CPU. Ten-sample p95 is an observed maximum, not a
production tail estimate. Session size is an upstream setting, not a guarantee
that a smaller source chunk becomes a larger device request.

| Case | Session rows | DataFusion CPU | Metal Scalar | Metal Batched | Time decrease |
|---|---:|---:|---:|---:|---:|
| primary | 8,192 | 3.616 | 125.688 | 90.960 | 27.6% |
| primary | 65,536 | 3.493 | 23.377 | 19.609 | 16.1% |
| primary | 262,144 | 3.693 | 15.223 | 14.489 | 4.8% |
| scale8m | 65,536 | 21.353 | 171.143 | 143.912 | 15.9% |
| scale8m | 262,144 | 22.839 | 114.869 | 106.801 | 7.0% |
| layout-two-partitions | 65,536 | 2.348 | 24.499 | 20.311 | 17.1% |
| layout-two-partitions | 262,144 | 2.618 | 17.332 | 15.414 | 11.1% |
| full-select-no-null | 65,536 | 1.506 | 18.347 | 14.854 | 19.0% |
| low-select | 65,536 | 1.071 | 23.871 | 19.482 | 18.4% |
| project-output | 65,536 | 0.561 | 2.278 | 2.263 | 0.7% |
| mixed-cpu-boundary | 65,536 | 0.930 | 2.663 | 2.681 | -0.7% |

The primary 64K setting passes the predeclared 10% recommendation gate again:
paired process improvements are 17.1%, 16.8%, 14.6%. Its 16.1% median decrease
is a new same-run observation; it does not replace the earlier study's 13.1%.
The primary 256K setting improves only 4.8%. The two-partition 256K median improves
11.1%, but two process pairs are below 10%; do not promote this as uniformly
passing that gate. At 8M, 64K improves 15.9% and 256K 7.0% relative to Scalar,
while CPU remains faster at both sizes.

For the primary fixture, changing the current-library M3 execution configuration
(8K/Scalar, 125.688 ms) to 64K/Batched gives 19.609 ms (6.41x); 256K/Batched gives
14.489 ms (8.67x). This is a same-binary configuration comparison, not a rebuild
of historical M3. Logical host peaks rise from 10.02 to 22.27 / 64.27 MiB. The
original-code before/after evidence remains in the [batching study](sql-m4-batch-study.md).

CPU's best primary configuration here is 64K, 3.493 ms. The fastest tested Metal
configuration, 256K/Batched, is about 4.15x slower. At 8M, the fastest Metal is
106.801 ms versus the best tested CPU 21.353 ms, about 5.00x slower. Larger total
input alone therefore does not establish a crossover in this range. These
comparisons include ordinary host input and output, without resident-input reuse.

The readback option does not change projection's execution path. Its ~0.7%
movement and mixed's ~0.7% median regression are observations with overlapping
process ranges, not a causal speedup/regression claim for that option. Full
materialized output and the CPU Sort/Limit boundary still need their own costs.

All process-p50 ranges are disclosed below. The primary 256K CPU process at
9.413 ms is retained without exclusion or replacement; it does not change the
CPU superiority conclusion.

| Case | Session rows | CPU p50 range | Scalar p50 range | Batched p50 range |
|---|---:|---:|---:|---:|
| primary | 8,192 | 3.600–3.618 | 125.212–126.285 | 89.989–91.222 |
| primary | 65,536 | 3.454–3.511 | 23.318–23.656 | 19.391–19.966 |
| primary | 262,144 | 3.649–9.413 | 15.067–15.382 | 14.414–14.520 |
| scale8m | 65,536 | 21.001–22.033 | 169.693–171.802 | 142.910–144.404 |
| scale8m | 262,144 | 22.776–23.158 | 114.685–115.370 | 106.344–106.840 |
| layout-two-partitions | 65,536 | 2.337–2.352 | 24.318–24.809 | 19.799–20.647 |
| layout-two-partitions | 262,144 | 2.600–2.692 | 16.884–17.367 | 15.259–15.611 |
| full-select-no-null | 65,536 | 1.495–1.543 | 18.135–18.613 | 14.655–14.883 |
| low-select | 65,536 | 1.062–1.080 | 23.543–23.910 | 19.022–19.979 |
| project-output | 65,536 | 0.560–0.567 | 2.131–2.311 | 2.178–2.320 |
| mixed-cpu-boundary | 65,536 | 0.895–0.938 | 2.654–2.758 | 2.674–2.962 |

## Work, requests and logical memory

Across each matched Scalar/Batched pair, exact results and observed output rows,
input/completed/fused/unfused request counts, uploads, descriptors, successful
read bytes, launches, count reads and allocation calls/bytes agree. Counters
below are last-warm observations; all repeats agree on the listed work counts.
Host peak is the maximum across the six Metal processes. MiB uses 2^20 bytes.
Calls are runtime API invocations, not physical copies or driver submissions.

| Case | Session rows | Read calls Scalar → Batched | Descriptors (both) | Read bytes (both) | Launches (both) | Peak logical host MiB |
|---|---:|---:|---:|---:|---:|---:|
| primary | 8,192 | 1152 → 384 | 1152 | 4608 | 1408 | 10.02 |
| primary | 65,536 | 144 → 48 | 144 | 576 | 176 | 22.27 |
| primary | 262,144 | 36 → 12 | 36 | 144 | 44 | 64.27 |
| scale8m | 65,536 | 1152 → 384 | 1152 | 4608 | 1408 | 80.02 |
| scale8m | 262,144 | 288 → 96 | 288 | 1152 | 352 | 122.02 |
| layout-two-partitions | 65,536 | 144 → 48 | 144 | 576 | 176 | 22.27 |
| layout-two-partitions | 262,144 | 72 → 24 | 72 | 288 | 88 | 36.27 |
| full-select-no-null | 65,536 | 144 → 48 | 144 | 576 | 176 | 22.02 |
| low-select | 65,536 | 144 → 48 | 144 | 576 | 176 | 22.27 |
| project-output | 65,536 | 7 → 7 | 7 | 629140 | 11 | 14.78 |
| mixed-cpu-boundary | 65,536 | 7 → 7 | 7 | 629140 | 11 | 14.78 |

The two-partition fixture contains 128K chunks. Setting session size to 256K
still produces eight nonempty device aggregate requests, rather than four;
the adapter does not concatenate chunks. At 64K it produces sixteen. Global
merging and all input partitions are included in the one-row SQL result. CPU
retains the same source layout and target-partition setting, including its
ordinary parallelism; device global aggregation still uses bounded coalescing.

The 8M/256K case reserves at most 122.02 MiB host and 4.17 MiB device in these
observations, below the fixed caps. No successful sample had a reservation
rejection or poisoned group. After releasing output, request/stream/host/device
reservations all drained to zero. Actual worker counter deltas match plan
observations; no fallback was labeled device execution.

Two deliberate failures were checked separately, not dropped from an attempted
successful matrix or counted as successful timings:

- 8M input / 1M request / Batched requires 304,103,192 host bytes (about 290.02 MiB)
  against 268,435,456 (256 MiB). It fails on the first request's admission.
- 1M input / 64K request with a one-byte device cap requires 1,094,380 device
  bytes and fails device admission.

Both runs return nonzero with the correct resource error, preserve logs and
manifest, and produce no COMPLETE marker. Caps were not raised to get a timing.
Failure drain/reuse and cancellation are additionally checked by the existing
lifecycle/resource suites; the early-exiting benchmark process alone does not
prove those lifecycle properties.

## Remaining phase costs

These are medians of the three **last-warm metric observations**, in ms, not
per-process query p50s or GPU timings. They are nested host-wall scopes; separate
medians must not be summed or subtracted to reconstruct query time.

| Batched case / session rows | Arrow adaptation | Resource preflight | Upload/submit | Complete/export | Worker |
|---|---:|---:|---:|---:|---:|
| primary / 64K | 3.588 | 0.255 | 3.750 | 9.346 | 17.631 |
| primary / 256K | 3.444 | 0.080 | 4.688 | 5.043 | 13.476 |
| scale8m / 256K | 27.538 | 0.529 | 36.244 | 40.872 | 106.020 |

Completion includes pending GPU work/checks/decoding; upload/submit includes
encoding/allocations and possible preparation. They do not isolate driver
compilation, hardware utilization or transfer time. Significant work remains
in several phases after read batching. This evidence does not identify a single
additional optimization that would close the CPU gap, establish zero-copy or
justify cross-batch device state. A future change needs a new bounded proposal
and complete-query comparison.

## Validation and stage decision

The boundary profile completed 50 fresh CPU/native Metal/DataFusion processes
with 100 exact-result samples, covering empty/one-row, 63/64/65 and 256/257 tile
boundaries, all-NULL/no-match, nonzero value/bitmap offsets, multiple source
chunks/partitions and mixed CPU Sort/Limit. These one-warm-sample runs support
correctness only. The original default SQL/DSL quick path completed 11 processes
and 22 verified samples. All processes used the final release binary.

177 all-feature tests and 67 SQL-only tests passed. This includes fixed integer
and F32 DSL readback values/canonical bits, malformed count/word/backend results,
checked overflow/filter suppression, SQL NULL/globality/schema/properties,
shared-budget cancellation, late errors, drain/poison and reuse. No-feature
all-target check, strict all-target Clippy/rustdoc and Python's eight focused
manifest/output-validation tests passed. After two style fixes, the affected five
acceptance tests, strict checks and final release build passed again. The existing
nonfatal macOS unwind-table linker warning remains.

| Gate | Result |
|---|---|
| Goal A: reproducible improvement of the selected original Metal path | Supported for the declared synthetic aggregate configurations, with memory tradeoffs |
| Explicit use, semantic/resource regressions, diagnostics and reproducible evidence | Passed within the selected support boundary |
| Primary batched-readback ≥10% across process comparisons | Passed in this replication; weaker generalization disclosed |
| Goal B: a matched full-query win over reasonable DataFusion CPU | **Not passed: 0 of 11 comparisons** |
| Real application workload / latency SLO | Not supplied; synthetic identity retained |
| F3 cross-batch device accumulation | Deferred; not implemented or claimed complete |

The selected M4 work and expanded evidence can be reviewed as delivered while
its performance-value exit condition remains open. Keep Scalar default and
explicit placement; do not install a size threshold, widen the SQL allow-list or
continue stacking speculative changes on this fixture. Further M4 performance
work should start from a concrete application requirement and a measured cost
hypothesis. The next capability work is M5's bounded group-by / unique-build
join contracts and fixtures; those operators need their own independent CPU
baseline and acceptance and are not blocked by simple M4 SQL losing to CPU.

## Reproduce and evidence

Build and run the two profiles using the commands in the
[profile workflow](../sql-acceptance-profiles.md), choosing new output directories.
For this report use the pinned source commit above and the exact archived
manifests. Three repeats / ten warm samples are required for the performance
comparison; the separate correctness profile uses one repeat / one warm sample.

- Evaluation: [1,089 samples](sql-m4-evaluation-2026-11-23-samples.csv),
  [99 summaries](sql-m4-evaluation-2026-11-23-summary.csv),
  [every command/plan/metric](sql-m4-evaluation-2026-11-23-plans.txt),
  [source/binary/environment identity](sql-m4-evaluation-2026-11-23-environment.json),
  [exact manifest](sql-m4-evaluation-2026-11-23-manifest.json),
  [completion](sql-m4-evaluation-2026-11-23-complete.txt).
- Boundaries: [100 samples](sql-m4-boundaries-2026-11-23-samples.csv),
  [50 summaries](sql-m4-boundaries-2026-11-23-summary.csv),
  [every command/plan/metric](sql-m4-boundaries-2026-11-23-plans.txt),
  [identity](sql-m4-boundaries-2026-11-23-environment.json),
  [exact manifest](sql-m4-boundaries-2026-11-23-manifest.json),
  [completion](sql-m4-boundaries-2026-11-23-complete.txt).
- [Expected budget failures: manifests, commands, logs, partial stdout and identity](sql-m4-rejections-2026-11-23.json).
- Prior matched [batch-size](sql-m4-batch-study.md), [phase](sql-m4-phase-study.md)
  and [same-binary readback](sql-m4-readback-study.md) studies retain their own
  raw evidence; do not pool samples or use cross-run CPU times as a new baseline.
