# Opt-in SQL performance profiling

`AdapterOptions::with_profiling_mode(ProfilingMode::Host)` exposes fine-grained
host wall scopes in executed Cube nodes. `Off` is the default and retains the
existing coarse metrics. `Device` adds one coarse device window per nonempty
request on the verified native Metal backend. It is a diagnostic mode:
opening/closing the pinned Metal window synchronizes the stream. No per-kernel
synchronization or kernel/ABI change is added.

```rust,ignore
use cuberel::datafusion::{AdapterOptions, ProfilingMode};
let options = AdapterOptions::with_executor(executor)
    .with_profiling_mode(ProfilingMode::Host);
```

Like batch policy, profiling is captured in the rewritten engine. Rebuild a
query to change it; a setter on another options clone does not modify existing
plans. Planning and placement reports do not execute or open a device window.
CpuOnly/restored plans have no Cube profiling observations. EXPLAIN and placement
reports show the requested mode, without implying device timing is available.

## Host scopes and counters

Profile metrics are cumulative integer nanoseconds (`_ns`) or bytes (`_bytes`),
scoped to the executed node/partition. Off does not register these metrics or
read fine-grained clocks; its low-level hooks only check an empty thread-local
slot. Host installs a collector only inside the synchronous blocking SQL worker.
Nested scopes and error/panic paths restore the previous collector, so reused
worker threads and other queries cannot inherit its data. No additional device
sync, read, launch or allocation API is called by Host collection.

| Metric after `profile_` | Boundary / interpretation |
|---|---|
| `validity_expand_ns`, `validity_expand_bytes` | Arrow NULL bitmap to dense U32 validity; bytes generated, only columns with actual NULLs |
| `value_encode_ns`, `value_encode_bytes` | Existing per-value host byte encoding before input creation |
| `validity_encode_ns`, `validity_encode_bytes` | Existing dense U32 validity byte encoding, including all-ones planes |
| `runtime_upload_ns` | `client.create_from_slice`; may include runtime allocation/queue work, not physical transfer time |
| `runtime_allocate_ns` | Explicit `client.empty` calls; excludes input creation and runtime internal/cache allocations |
| `pipeline_prepare_ns` | Host pipeline descriptor/prepare/budget validation inside execute_pipeline_host; separate from adapter preflight |
| `pipeline_submit_ns` | Existing stage submission, including intermediate allocations and kernel preparation/runtime calls; not isolated driver time |
| `completion_sync_ns` | Existing checked completion synchronization calls; no new sync in Host mode |
| `readback_api_ns` | Existing single-/multi-buffer read API invocations, including pending work/runtime wait; not pure copy time |
| `scalar_decode_ns` | Aggregate U32 word decoding; Batched also includes returned shape/count validation, not projection payload decoding |
| `arrow_export_ns` | Arrow construction/bitmap packing inside pipeline materialization; SQL global aggregate final output uses the metric below |
| `global_merge_ns` | SQL host scalar merge after each completed request, outside worker |
| `global_output_ns` | Final SQL aggregate Arrow construction and consumer output reservation, outside worker |

These are nested host-wall observations, not an additive partition of query time.
For example, runtime allocation is nested inside pipeline submission; device
work can proceed during host submit, and read API time can include waiting.
Existing `arrow_adapt`, `upload_submit`, `complete_export`, `worker` remain the
parents where applicable. Device window boundary overhead is outside the existing
upload/complete scopes but inside worker/query time. Do not subtract separate
phase medians or a GPU span from host wall time to manufacture a CPU-only cost.

`profile_graph_aggregate_partial`, `profile_graph_aggregate_merge` and
`profile_graph_other_kernels` count the successfully returned stage graph's
kernel stages, excluding upload/input/materialize boundaries. Their sum matches
observed launches for successful supported SQL requests; they are not GPU times.
Errors partway through submission can leave observed launches without a returned
graph, so use both observations when investigating failures.

## Device window semantics and failure handling

The fixed revision provides native Metal command-buffer timing. Its CPU runtime
advertises Device timing but returns System timing from its fallback profiler, so
the generic capability flag alone is insufficient. This integration allows only
the verified `metal` runtime plus Device capability, and checks the resolved
method again. Device mode skips the backend window on CPU, increments
`profile_device_unavailable`, and keeps GPU time unavailable. It does not relabel
system timing as GPU timing. Empty requests open no window.

On Metal, the window begins after adapter preflight, before ordinary host pipeline
execution, and ends after checked materialization. The backend collects
work-bearing command buffers and reports **latest GPU end minus earliest GPU
start**. This is a GPU wall span across those command buffers, including gaps;
it is not the sum of kernel busy times, per-kernel timestamps, occupancy or
bandwidth. It also cannot separate shader time from scheduling between buffers.
The collector does not preserve individual command-buffer timestamps.

- `profile_device_window_ns`: sum of valid request window spans.
- `profile_device_windows`: number of positive, successfully resolved windows.
- `profile_device_unavailable`: nonempty requests on a backend without device timing.
- `profile_device_failed`: failed/refused/abandoned/nonpositive windows.
- `profile_device_boundary_ns`: host time in start/end/abandon calls, including
  additional boundary synchronization. It is nested diagnostic overhead, not GPU time.

The backend can report zero on invalid timestamps. Conservatively treat a
nonempty zero-duration window as failed instead of presenting it as fast work.
A normal query error or panic abandons its token before the worker releases its
permit/resources; it never publishes a success timing. Profiling errors mark
the diagnostic failed and do not replace an existing query error or authorize
CPU retry. Requests still use the existing drain/poison and completion checks.

CSV exposes device time only if at least one window succeeded and no window was
unavailable/failed for that sample. Otherwise it is blank, even if some partial
windows succeeded. Raw plan counters retain the diagnostic totals. Off and
runtime-free/DSL profile fields are blank; Host has host metrics but blank GPU
time. A zero counter means an instrumented operation did not occur; a blank means
unavailable or not collected.

The pinned Metal server invalidates open profile windows when it observes a
backend failure, including a failure on another stream. Such timings are failed
diagnostics, even when the measured request itself returned valid results. The
positive window/recovery test therefore uses its own process to avoid unrelated
intentional backend-failure tests.

Use a dedicated engine/client and avoid concurrent external work or another
CubeCL profiling observer during Device sampling. The SQL gate isolates this
engine's requests, not arbitrary external uses/streams of the same client;
the backend cannot establish whole-device exclusivity. Device synchronization
changes batching and may perturb latency, so performance recommendations must
use Off samples, with Host/Device overhead reported alongside them.

## Harness and focused attribution contract

```sh
cargo build --locked --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
# One direct SQL device process:
/private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  metal sql aggregate 1048576 50 20 1 1 10 262144 batched --profiling host
# Same-binary off/host/device comparison, one CPU baseline per fixture/repeat:
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/cuberel-profile-attribution \
  --manifest benchmarks/sql-profile-attribution.json \
  --profiling-modes off host device --backends datafusion metal --repeats 3 --warm 10
```

The existing v1 manifest schema stays unchanged; `--profiling-modes` is an
explicit study/manifest dimension. Device mode order rotates across process
repeats. Original default SQL/DSL filenames/matrix and default Scalar mode remain;
DSL has no SQL worker collector and rejects measurement flags. Invalid/duplicate
profiling flags fail explicitly; non-Off profiling requires SQL device execution.
The runner validates requested mode, metric presence/integer values and GPU
availability before accepting samples.

Before sampling, fix three synthetic cases: 1M/256K aggregate, 8M/256K aggregate,
and 64K/64K projection, all threshold 50% / requested NULL 20%, one source
chunk/partition. Batched readback and 256 MiB host / 64 MiB device caps stay fixed.
Run three fresh-process repeats with ten warm samples; builds/tests finish first.
Archive all raw samples, plans, mode-dependent overhead, source/binary hashes
and exact result/byte/launch/resource checks. Host overhead should ideally stay
below 5%; report larger overhead or repeat variation without removing samples.
This same-binary comparison quantifies collection overhead relative to the new
Off mode; it does not isolate the cost of dormant hooks versus a pre-instrumentation
binary. Device overhead has no predeclared acceptance threshold because the backend
adds synchronization. Do not call a faster profiled sample an optimization.

The output should rank the observed host components and device span with their
coverage/overlap limits, select a bounded next investigation, and state the
remaining unknowns. It must not claim an optimization's speedup from attribution
alone or require a speculative new kernel/backend dependency.
