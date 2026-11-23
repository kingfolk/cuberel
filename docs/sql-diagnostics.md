# SQL placement reports and fragment observations

`placement_report(plan, &options)` explains candidates using the same region and
aggregate compilers as execution. It does not poll inputs, allocate device buffers
or compile/launch kernels. Reports work with `planning_only()` and no linked
runtime. Each candidate includes its traversal/fragment ID, selected CPU/device
placement and primary reason, lower boundary, exact input/output schema, input
partition contract, filter/aggregate/node counts and proven row bounds when known.
The report header records the target, semantic profile, per-request budget and
shared limits. No guessed per-batch byte estimate is substituted for missing data:
`None` means unknown; actual estimates are exposed at runtime.

Supported chains are one candidate and lower CPU boundaries are traversed again.
Ordinary sources, sort, limit and exchanges are not separately rejected candidate
expressions. Rejected filter/project/aggregate nodes retain their structured
`Unsupported` reason, such as `semantic_mismatch`, `unsupported_type`,
`fragment_limit`, `unknown_aggregate_bound`, `aggregate_range` or
`partition_contract`. Disabled/CPU-only placement is explicit. A selected candidate
is semantically eligible; runtime resource/device preflight can still reject it.

## Explicit policy modes

```rust,ignore
use cuberel::datafusion::{AdapterOptions, PlacementMode, placement_report};
let options = AdapterOptions::with_executor(executor.clone())
    .with_placement_mode(PlacementMode::RequireCandidates);
let report = placement_report(original_plan.clone(), &options)?;
println!("{report}");
```

- `OptIn` keeps the existing explicit supported-fragment policy. Rejected
  candidates stay in DataFusion and lower supported regions can still accelerate.
- `CpuOnly` retains/restores DataFusion operators, including if passed an already
  rewritten CubeExec/CubeAggregateExec plan. No device execution is performed.
- `RequireCandidates` rejects planning when a filter/project/global-aggregate
  candidate cannot be placed, or no candidates exist. It does not demand that
  CPU sources, exchanges, sort/limit disappear. This is a diagnostic requirement,
  not a CPU fallback switch. An executor must still be supplied/enabled.

Default AdapterOptions remain disabled. Selecting a mode alone does not enable
registration. No automatic workload threshold or measured cost model is installed;
those require F8 evidence. Placement policy and M2 aggregate FusionMode differ:
an unfused device pipeline is still device placement.

## EXPLAIN and EXPLAIN ANALYZE

Cube operators display fragment ID, device, semantic profile, shape/partition
contract, shared limits, and `per_batch_estimate=runtime`. Filter/project nodes
retain their lower-boundary reason. `placement_report` supplies reasons for CPU
candidates which ordinary EXPLAIN leaves as DataFusion nodes. IDs follow the
adapter traversal: fused regions collapse to one edge before descending to their
input; they are plan-local and do not specialize kernel cache identities.

Per-partition metrics are attached to the ordinary DataFusion metrics interface
and displayed by EXPLAIN ANALYZE / `DisplayableExecutionPlan::with_metrics`:

| Metrics | Meaning |
|---|---|
| input/output rows and batches | Pulled input, returned output; output uses canonical DataFusion counters |
| streams_completed/failed/canceled, active_streams | First-poll through EOF/error/drop; an unpolled drop is not a canceled run |
| requests_completed/request_errors | Successful worker completions and failed admission/worker attempts |
| observed_upload/readback calls/bytes | Explicit executor counter deltas while holding shared admission |
| observed_launches/count_reads/alloc_calls/alloc_bytes | Actual explicit submissions/readbacks/logical allocation attempts |
| define_calls | IR definition attempts; not driver compilation counts or disk-cache hits |
| max_estimated_device_bytes/stages, max_output_capacity | Maximum per-request preflight estimates, including rejected budgets |
| max_requested_workspace_bytes | Conservative host workspace request; not live/physical memory |
| fused_requests/unfused_requests, fusion_fallback{reason} | Actual M2 stage-graph choice, with bounded static fallback reasons |
| queue_wait | Waiting for shared device admission |
| prepare | Arrow adaptation and typed/resource preflight |
| execute_export | Device pipeline preparation/submission/completion and Arrow or scalar export |
| worker | Blocking worker lifetime including failure drain |
| stream_wall | First poll through EOF/error/drop, including child/consumer waiting |

Times are wall durations and overlap; do not add them or sum partition wall time
to claim query elapsed time. Runtime IR definition and backend compilation occur
inside execution; a separate driver compilation timer is unavailable and is not
invented. Counts include failed requests where the underlying operation occurred.
Selection count reads are distinct from aggregate state count-word readbacks.
`max_*` gauges are per-partition request maxima. DataFusion's aggregated display
may sum partition maxima; that is not an instantaneous group memory peak. Only
SharedResources snapshots report the group's deduplicated live/peak reservations.

Transfer/launch deltas are isolated across adapters sharing the controller gate.
Concurrent direct Executor calls or separate groups sharing its counters can
contaminate observations; these are explicit executor observations, not driver
telemetry. Use a dedicated executor/controller when attribution matters.
Metrics follow the physical plan execution lifetime; reset/rebuild plan states
before a new independent execution, especially with exchanges.
Registrations are cached per plan partition, so repeated/unpolled execute calls
do not append a fresh set of counters. Cleanup metrics can finish after stream
cancellation; wait for the group's active requests to drain before taking a final
observation.

SharedResources snapshots remain **group** live/peak logical reservations and
rejections. They are not copied into every partition metric, which would double
count shared allocations. Consumer-held output remains charged after stream wall
time ends. See [resource ownership and cancellation](sql-lifecycle.md).

```sh
cargo run --locked --all-features --example datafusion_diagnostics -- cpu
cargo run --locked --all-features --example datafusion_diagnostics -- metal require
cargo run --locked --all-features --example datafusion_diagnostics -- cpu cpu-only
cargo test --locked --all-features --test datafusion_diagnostics
```

The example uses `[10,NULL,30,40]`, returns value `[30,40]` and missing
`[false,false]`, and prints both placement and actual observations. Tests compare
observed transfers with executor counters on actual CPU/native Metal, inspect
disabled/semantic rejections without a runtime, check policy failures, restored
CPU execution, cancellation/failure metrics and global-aggregate observations.
This establishes diagnostics, not SQL performance improvement.

Verified on 2026-11-20 with Rust 1.99 and the pinned dependencies: 161 all-feature
tests passed on actual CPU/native Metal, 61 SQL-only tests passed without a
linked runtime, and strict Clippy/rustdoc, no-feature all-target, fmt/diff checks
passed. Both device examples observed 32 upload bytes, 28 readback bytes and
5 launches; CPU-only mode observed zero device work. Output release returned
all shared host/device reservations to zero. The known macOS unwind-table linker
warning remains nonfatal.

The [explicit batch policy](sql-batch-policy.md) is shown in the placement report
and Cube EXPLAIN nodes. `max_input_batch_rows` records a per-partition actual
maximum; the configured source target is not a prediction of every request size.

## Nested host-wall phase scopes

Filter/project and global-aggregate workers now expose four disjoint child scopes:

| Metric | Boundary |
|---|---|
| `arrow_adapt` | Referenced-column mapping, Arrow/schema adaptation, dense validity expansion and host Batch view |
| `resource_preflight` | Pipeline estimate, budget and shared device-byte reservation |
| `upload_submit` | Existing host-pipeline API: repeated validated preparation, encoding/upload, allocations, IR/compilation where observable, and submissions |
| `complete_export` | Completion/error checks, scalar or column readbacks, decoding/Arrow packing and projection output reservation |

The first pair is nested under `prepare`; the second under `execute_export`. All
are inside `worker`. They are host wall intervals, not GPU/kernel/driver timers.
Submission may execute or compile inline; completion can include outstanding GPU
work. Global host scalar merging and final SQL-row construction occur outside the
worker. Sibling scopes do not cover every scheduler/metadata cost. Times from
parallel partitions still overlap, and medians cannot be added to reconstruct a
query. Use the independent end-to-end timer for performance claims.

Failures count time in the phase actually entered; unentered registered phases
remain zero. CPU-only DataFusion and direct DSL have no Cube phase observations;
the acceptance CSV marks those fields unavailable rather than inventing times.
