# SQL profiling attribution and overhead: 2026-11-24

The new profiler narrows the next investigation to **runtime input creation and
existing completion synchronization, with Arrow validity expansion also material**.
Host scalar merging, output construction and explicit pipeline-submit work are
small in these observations. This selects investigations, not a demonstrated
optimization or a claim that the GPU kernels are optimal. CPU still wins the
three Off-mode comparisons.

## Fixed contract and measurement identity

The [profiling contract](../sql-detailed-profiling.md) and three-fixture manifest
were committed before sampling at `8db1d08f87fde61810acaa0c6432bb2c13001a28`, based
on merged F5 main `2866375`. All archived library/harness hashes were verified
against that commit; binary and exact manifest hashes are preserved. One release
binary supplies Off, Host and Device modes. No source patch or dependency upgrade
is needed. Hardware is Apple M1 / 16 GiB, macOS 15.5, Rust/Cargo 1.99, pinned
CubeCL fork `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, DataFusion 55.2.0, Arrow 59.2.0.
Rust 1.95 minimum was not independently tested.

36 sequential fresh processes produced 396 verified samples. Each fixture runs
three fresh-process repeats, one first query plus ten warm queries, with mode and
backend order rotating. DataFusion CPU runs once per fixture/repeat. Builds/tests
finish before timing; no competing agent device work ran. OS/driver caches are
not cleared. All fixtures remain synthetic, ordinary host-input SQL without
cross-query residency, one source chunk/partition, threshold 50%, requested NULL
20%, Batched aggregate readback, 256 MiB adapter host / 64 MiB device logical caps.

- `primary`: 1,048,576 input rows, session batch 262,144, integer global aggregate.
- `scale8m`: 8,388,608 rows, session batch 262,144, same SQL.
- `project-output`: 65,536 rows, session batch 65,536, full four-column host projection.

Complete query times include session/SQL/physical-plan preparation, ordinary input,
completed execution/merge and Arrow export. Initialization, fixture/oracle creation,
validation and output destruction are excluded. First/init and every warm sample
remain available. Nearest-rank p95 from ten warm samples is only an observed maximum.

## Same-binary collection overhead

Each time is the median of three per-process warm query p50s, in ms. Percent is
relative to Off's median. Negative deltas are measurement variation, not profiler
speedups. This measures collection modes against the **instrumented Off mode**;
it does not isolate dormant-hook overhead against pre-instrumentation code.

| Fixture | CPU | Off | Host | Host delta | Device | Device delta |
|---|---:|---:|---:|---:|---:|---:|
| primary | 3.670 | 15.162 | 15.312 | +1.0% | 15.770 | +4.0% |
| scale8m | 23.005 | 117.995 | 117.691 | -0.3% | 116.228 | -1.5% |
| project-output | 0.562 | 2.272 | 2.493 | +9.7% | 2.654 | +16.8% |

| Fixture | Off process-p50 range | Host range | Device range |
|---|---:|---:|---:|
| primary | 14.666–15.558 | 14.784–15.624 | 15.314–15.906 |
| scale8m | 111.587–118.198 | 114.147–118.164 | 116.204–117.583 |
| project-output | 2.149–2.482 | 2.365–3.047 | 2.417–2.672 |

Primary Host paired-repeat changes are -5.0%, +6.5%, +1.0%; scale8m changes are
+5.5%, +0.1%, -3.4%. A small median difference does not establish a fixed per-query
instrumentation cost. In projection, the original Host median +9.7% exceeds the
predeclared 5% goal, and paired changes are +34.1%, +0.4%, +10.0%. Device adds
boundary synchronization and has no low-overhead guarantee; projection's original
+16.8% makes it unsuitable as an ordinary production-latency measurement mode.

Because projection exceeded the goal, a narrow follow-up was declared before its
sampling: preserve the original, repeat only projection with three fresh processes,
**50 warm queries** each and the same three modes/CPU baseline. Those 12 processes /
612 samples yielded Off 2.321 ms, Host 2.334 ms (+0.6%), Device 2.499 ms (+7.7%).
Off process p50s are 2.321/2.274/2.630 ms, Host 2.334/2.357/2.246, Device
2.470/2.529/2.499. The range still prevents a stable sub-5% overhead claim. Both
studies are archived separately, without exclusion, pooling or replacement of
unfavorable samples. Use Off for end-to-end recommendations and profile modes
for bounded diagnosis, reporting their perturbation.

## Host attribution

Values below are medians of the three **last-warm Host metric observations**, in
ms. They are not query p50s and are not additive: allocation is nested in pipeline
submission, and runtime calls can include queue/device wait. Coarse parent phases
remain in the raw data. A zero means that instrumented operation did not occur.

| Metric after `profile_` | primary | scale8m | project-output |
|---|---:|---:|---:|
| validity_expand | 2.708 | 21.960 | 0.174 |
| value_encode | 0.475 | 3.762 | 0.020 |
| validity_encode | 0.279 | 2.284 | 0.023 |
| runtime_upload | 3.410 | 27.860 | 0.208 |
| runtime_allocate | 0.121 | 0.921 | 0.008 |
| pipeline_prepare | 0.054 | 0.337 | 0.012 |
| pipeline_submit | 0.210 | 1.561 | 0.068 |
| completion_sync | 4.531 | 36.832 | 0.834 |
| readback_api | 0.539 | 4.045 | 0.335 |
| scalar_decode | 0.002 | 0.018 | 0.000 |
| arrow_export | 0.000 | 0.000 | 0.160 |
| global_merge | 0.001 | 0.006 | 0.000 |
| global_output | 0.011 | 0.015 | 0.000 |

The new distinctions matter:

- Validity expansion alone accounts for about 2.71 / 21.96 ms in the two aggregate
  observations. This is the existing Arrow bitmask-to-dense-U32 path. It is a
  concrete host loop to investigate, with offset/NULL semantics retained.
- Runtime input creation takes about 3.41 / 27.86 ms, while explicit value and
  validity encoding are much smaller. `runtime_upload_ns` wraps
  `create_from_slice`; this includes runtime allocation/queue/copy work and does
  **not** establish physical bus bandwidth or which internal operation dominates.
- Existing completion synchronization takes about 4.53 / 36.83 ms. It includes
  unfinished GPU work and runtime/driver/host waiting; read API calls are measured
  separately. Neither interval is pure copy time.
- Host global merge/output are around microseconds to tens of microseconds.
  Moving only that host merge does not have evidence for closing the CPU gap.
  Pipeline submission and explicit intermediate allocation are also smaller
  observed host components, even though aggregate merge launch counts are high.

The 1M Host observations are several ms in multiple components against the
same-run CPU's 3.670 ms **whole query**. Eliminating one host loop alone is therefore
insufficient evidence of a CPU win. Do not subtract independently summarized
metrics to predict a precise optimized total or guaranteed speedup.

## Device spans and boundaries

These are Device-mode last-warm metric medians, in ms. The GPU column is the sum
of valid **per-request earliest-start to latest-end command-buffer spans**, not
kernel busy-time sums. It includes gaps within a window and excludes host gaps
between request windows. All nonempty windows resolved positively, with zero
unavailable/failed counts in these Metal samples.

| Fixture | Valid windows/query | Sum GPU window spans | Host window-boundary time | Existing completion-sync time |
|---|---:|---:|---:|---:|
| primary | 4 | 1.174 | 0.295 | 5.168 |
| scale8m | 32 | 13.475 | 2.412 | 42.679 |
| project-output | 1 | 0.098 | 0.109 | 0.859 |

The device spans are substantially smaller than whole-query wall times in these
profiled executions. Together with the host scopes, this supports investigating
host/runtime/completion paths before assuming the reduction shader is the main
limit. It does not measure occupancy, utilization or per-kernel execution, nor
prove that kernels cannot improve. In particular, do **not** subtract the primary
1.174 ms span from its 5.168 ms sync median and call the difference driver overhead:
they have different boundaries and overlap with host submission/read activity.
Host and Device modes are also differently perturbed executions.

The pinned CPU runtime advertises Device capability but actually resolves System
timing. The implementation only accepts verified native `metal` plus Device
capability, checks the returned method, and exposes no GPU number for CPU.
Four CPU smoke processes/eight exact-result samples verify nonempty unavailable
and empty/no-window paths. Raw GPU counters are diagnostic totals; CSV GPU time
is blank whenever no valid window exists or any window was unavailable/failed.

The Metal server invalidates open windows on other-stream backend faults. The
real checked-failure/abandon/reopen regression is process-isolated from unrelated
intentional failure tests; concurrent external client use is outside the timing
isolation guarantee. Device mode adds backend start/end synchronization. Host
mode adds no sync/read/launch/device-allocation API.

## Unchanged work and correctness

Every sample checks independent exact values, aliases/schema/metadata/NULLs,
actual device placement and observed executor deltas. Shared caps are respected
and all request/stream/host/device reservations drain after output release.
Across matched Off/Host/Device samples, input/completed/fused/unfused request
counts, result rows, upload/read calls and bytes/descriptors, allocation calls /
bytes, count reads and launches agree. Stage counts sum to observed launches in
successful profiled requests. Encoding byte counters agree with uploaded payloads.

| Fixture | Expanded validity bytes | Encoded value bytes | Encoded validity bytes | Aggregate partial stages | Merge stages | Other kernel stages |
|---|---:|---:|---:|---:|---:|---:|
| primary | 8388608 | 8388608 | 8388608 | 4 | 40 | 0 |
| scale8m | 67108864 | 67108864 | 67108864 | 32 | 320 | 0 |
| project-output | 524288 | 786432 | 786432 | 0 | 0 | 11 |

The byte counts are generated API payloads, not RSS/physical traffic or compressed
transfer sizes. Runtime/driver caches, profiling command-buffer retention and
upstream/oracle allocations are outside adapter logical memory observations.

Validation passed: 183 all-feature tests on real CPU/native Metal, 70 SQL-only
tests, 13 Python runner tests, no-feature all-target check, strict all-target
Clippy/rustdoc and fmt/diff. Fixed-result profiles cover Off/Host/Device equality,
byte/stage counts, timer nesting, empty input, unavailable CPU windows and valid
Metal windows. Private tests cover nested/unwind scope restoration and real
checked-overflow window abandonment/reuse. Original default SQL/DSL quick passed
11 processes/22 samples on the final binary. The existing nonfatal macOS
unwind-table linker warning remains. Luna handled the bounded harness/regression
work; actual token usage is unavailable from the tools.

## Next bounded investigations

1. Split the runtime input-creation and completion APIs internally into allocation,
   memory copy, server queue and synchronization/event wait. These are the largest
   remaining mixed intervals; this needs verified backend instrumentation, not
   host subtraction or a dependency upgrade assumed to be safe.
2. Measure a correctness-preserving improvement to validity expansion on this
   pinned ABI first. Bit-packed device validity or zero-copy would be separate
   representation/ownership proposals with independent budgets and NULL/offset
   proofs, not conclusions of this profile.
3. Revisit stage merging/device-resident accumulation only with those costs
   isolated and a declared full-query benefit hypothesis. Counts alone do not
   prioritize a new kernel, and tiny host merge costs do not justify it alone.

No optimization, default-mode change, broader SQL placement or CPU-performance
claim is shipped by this report. M4's CPU-win gate remains unmet. M5's relational
contracts remain separate capability work and can use these observations when
choosing a device execution design.

## Reproduce and raw evidence

Use the build and three-mode commands in the [profile contract](../sql-detailed-profiling.md)
at the pinned source commit with a new output directory. Follow-up and CPU smoke
manifests below preserve their exact settings; use three repeats / 50 warm for
the projection follow-up and one repeat / one warm with backends cpu and modes
host/device for the CPU smoke. Original study uses three repeats / ten warm.

- Original 36-process study: [samples](sql-profile-attribution-2026-11-24-samples.csv), [summaries](sql-profile-attribution-2026-11-24-summary.csv), [all commands/plans/metrics](sql-profile-attribution-2026-11-24-plans.txt), [identity](sql-profile-attribution-2026-11-24-environment.json), [exact manifest](sql-profile-attribution-2026-11-24-manifest.json), [completion](sql-profile-attribution-2026-11-24-complete.txt).
- 12-process projection follow-up: [samples](sql-profile-projection-followup-2026-11-24-samples.csv), [summaries](sql-profile-projection-followup-2026-11-24-summary.csv), [all commands/plans/metrics](sql-profile-projection-followup-2026-11-24-plans.txt), [identity](sql-profile-projection-followup-2026-11-24-environment.json), [exact manifest](sql-profile-projection-followup-2026-11-24-manifest.json), [completion](sql-profile-projection-followup-2026-11-24-complete.txt).
- 4-process CPU availability smoke: [samples](sql-profile-cpu-smoke-2026-11-24-samples.csv), [summaries](sql-profile-cpu-smoke-2026-11-24-summary.csv), [all commands/plans/metrics](sql-profile-cpu-smoke-2026-11-24-plans.txt), [identity](sql-profile-cpu-smoke-2026-11-24-environment.json), [exact manifest](sql-profile-cpu-smoke-2026-11-24-manifest.json), [completion](sql-profile-cpu-smoke-2026-11-24-complete.txt).
