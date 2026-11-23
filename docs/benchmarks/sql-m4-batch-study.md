# M4-F1 matched SQL batching: 2026-11-22

Larger native source batches substantially reduce the completed Metal query cost
for the existing filtered integer aggregate. They do **not** make this workload
faster than DataFusion CPU. Keep placement explicit and proceed with a bounded,
explicit source-batch configuration; do not add internal adapter concatenation.

## Contract and environment

The predeclared [study contract](../sql-batch-study.md) selects the existing M3
synthetic aggregate: SUM(y), COUNT(*), COUNT(y), MIN(y), MAX(y), with WHERE x < 50,
requested NULL density 20%, one immutable source chunk/partition, and actual
Arrow value/validity offsets. The independent host oracle validates every result
and exact schema/metadata; the device plan must contain CubeAggregateExec.
Source batch sizes are session settings shared by matched CPU and Metal runs.
Total input remains fixed while the setting changes. This is not an application
trace, an isolated kernel benchmark or an automatic placement threshold.

Apple M1, 16 GiB, macOS 15.5, Rust/Cargo 1.99.0, release. Dependencies remain
CubeCL fork 5037c0639ce3b1205392bdbe43dcf25e2ea09e29, DataFusion 55.2.0 and Arrow
59.2.0. Base is merged M3-F8 main 6ce1953 plus the F1 harness identified by source
hashes in the archived environment. Tests and builds finished before sampling.

72 fresh processes (3 sizes × 4 settings × 2 backends × 3 repeats) produced 792
verified samples, each with one first query and ten warm queries. Processes run
sequentially; backend order rotates across repeats. OS/driver caches are not
cleared. There is no concurrent-query or other-hardware claim.

## Matched complete-query results

Times include session/SQL/physical-plan preparation, host input, completed
execution, scalar merges and export, excluding device initialization, input/oracle
generation and validation. Each time below is the median of three **per-process
warm p50s**, not a percentile pooled over thirty samples. Metal's parentheses
show the three process-p50 min–max. Full per-process p95s and first/init costs are
preserved in the raw files; ten-sample nearest-rank p95 is the observed maximum.
Logical host peaks are the maximum of the last warm observations, not RSS.

| Rows | Session batch rows | CPU ms | Metal ms (process-p50 range) | Metal improvement vs 8K | Peak logical host MiB |
|---|---:|---:|---:|---:|---:|
| 65,536 | 8,192 | 1.512 | 9.756 (9.665–9.875) | 1.00x | 2.28 |
| 65,536 | 65,536 | 1.451 | 3.343 (3.264–3.359) | 2.92x | 14.53 |
| 65,536 | 262,144 | 1.453 | 3.292 (3.233–3.388) | 2.96x | 14.53 |
| 65,536 | 1,048,576 | 1.488 | 3.317 (3.299–3.338) | 2.94x | 14.53 |
| 1,048,576 | 8,192 | 5.972 | 131.536 (127.981–136.857) | 1.00x | 10.02 |
| 1,048,576 | 65,536 | 5.685 | 28.306 (28.166–30.655) | 4.65x | 22.27 |
| 1,048,576 | 262,144 | 5.936 | 19.703 (19.595–19.706) | 6.68x | 64.27 |
| 1,048,576 | 1,048,576 | 6.315 | 18.735 (18.537–18.835) | 7.02x | 232.27 |
| 2,097,152 | 8,192 | 10.927 | 286.983 (286.507–287.721) | 1.00x | 18.27 |
| 2,097,152 | 65,536 | 10.026 | 55.627 (53.132–55.962) | 5.16x | 30.52 |
| 2,097,152 | 262,144 | 10.492 | 37.186 (37.184–38.019) | 7.72x | 72.52 |
| 2,097,152 | 1,048,576 | 11.207 | 35.180 (35.008–35.407) | 8.16x | 240.52 |

At the primary 1M-row fixture, 64K improves Metal by 4.65x and 256K by 6.68x,
exceeding the predeclared 2x continuation criterion across all three process
repeats. CPU's best observed configuration is 64K at 5.685 ms; the fastest Metal
setting is 1M at 18.735 ms, about 3.30x slower. At 2M rows, CPU's best configuration
is also 64K (10.026 ms) versus Metal's 1M setting (35.180 ms). These CPU baselines
are from this matched run; do not compare the new Metal times only with CPU times
from an earlier run under different conditions.

For 64K total rows, 64K/256K/1M source settings all produce one nonempty batch;
their small timing differences are variation, not evidence of additional batching
benefit. The fastest setting is not automatically the preferred memory/latency
tradeoff. At 1M input, moving 256K → 1M gives only about 5% lower Metal p50 while
raising the logical host peak from 64.27 to 232.27 MiB.

## Actual request and transfer evidence

The following are the last warm counts in each primary-fixture process; all three
repeats agree on these counters. The source inserts one explicit empty batch,
which completes a worker request without launching a kernel. Every query uploads
the same 16 MiB of referenced input; ordinary SQL does not reuse a resident input.

| Session rows | Input batches / completed requests | Kernel launches | Readback calls | Readback bytes |
|---|---:|---:|---:|---:|
| 8,192 | 129 / 129 | 1,408 | 1,152 | 4,608 |
| 65,536 | 17 / 17 | 176 | 144 | 576 |
| 262,144 | 5 / 5 | 44 | 36 | 144 |
| 1,048,576 | 2 / 2 | 16 | 9 | 36 |

Fewer request/launch/completion/readback boundaries accompany the measured
improvement. This supports granularity as an optimization direction; it does not
provide an isolated causal time allocation. Large batches change recursive merge
depth as well as request count. No separate GPU kernel, driver compilation or
upload timer is available, and overlapping worker/prepare/export metrics are not
summed into query elapsed time. Frontend define attempts do not measure driver
compilation. Memory is unified but the API still adapts/encodes buffers; zero-copy
and hardware saturation are not established by these counters.

All cases passed the independent result/schema checks, observed transfer/launch
counts matched executor deltas, and shared reservations returned to zero after
output release. No backend substitution, budget rejection or CPU retry occurred.

## Decision and next layer

M4 goal A (improving the original Metal path) has supporting evidence for this
fixture; goal B (beating a reasonable CPU baseline) remains **not achieved**.
Proceed with native source batching plus a separately reviewable opt-in request
row ceiling and policy diagnostics. Preserve default session behavior and all
existing byte/stage limits. The larger-source API target cannot guarantee every
operator emits that size, so cap actual requests before upload and fail explicitly
rather than splitting/replaying input to hide a resource error.

Do not add a concatenation buffer, scheduler or implicit resident cache for this
result. F3 cross-batch device state is deferred: at the largest setting there is
only one nonempty input and nine scalar readback calls, so this experiment does
not justify treating cross-batch state as the next dominant optimization. Further
packing/submission/export attribution and application-specific fragments remain
M4 follow-up work. This report does not close all of M4 or validate broader SQL.

## Reproduce and raw evidence

Use the build and study command in the [study contract](../sql-batch-study.md),
with a new output directory. Archived files preserve every repetition:

- [792 individual samples](sql-m4-batches-2026-11-22-samples.csv).
- [72 process summaries](sql-m4-batches-2026-11-22-summary.csv).
- [All commands, plans and final metrics](sql-m4-batches-2026-11-22-plans.txt).
- [Environment, invocation and source/binary hashes](sql-m4-batches-2026-11-22-environment.json).
- [Completion marker](sql-m4-batches-2026-11-22-complete.txt).

## F1 verification

The new `datafusion_batch_sizes` suite passed three tests on actual CubeCL CPU,
native Metal and the DataFusion-only oracle; the runtime-free feature run passed
its oracle test. It checks exact aggregates across source batch sizes, sliced
nullable data, empty/all-NULL/multiple-partition input, reduced actual request
counts and zero reservations after output release. Strict Clippy for the changed
example/test, formatting and diff checks passed. The original quick SQL/DSL
workflow also completed 11 DataFusion/Metal processes and 22 verified samples.
The existing nonfatal macOS unwind-table linker warning remains.
