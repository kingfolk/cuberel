# Explicit bounded SQL batch policy

`AdapterOptions::with_batch_policy` pairs an explicit DataFusion source batch-size
setting with a hard row ceiling on actual device-worker requests. It implements
the native configuration route selected by the [matched M4 study](benchmarks/sql-m4-batch-study.md),
without adding adapter concatenation, prefetch, splitting or device-state caching.

```rust,ignore
use cuberel::datafusion::{AdapterOptions, BatchPolicy, register};
use datafusion::execution::session_state::SessionStateBuilder;

let policy = BatchPolicy::new(65_536, 65_536)?;
let options = AdapterOptions::with_executor(executor).with_batch_policy(policy);
let state = register(
    SessionStateBuilder::new_with_default_features().with_config(session_config),
    options,
).build();
```

Sizes satisfy `0 < source_rows <= max_request_rows <= u32::MAX`. The constructor
returns `BatchPolicyError` for invalid sizes. Accessors expose both values.
No policy is installed by default: existing session sizes, support boundaries and
execution behavior are preserved. A policy alone does not enable disabled
options. With `planning_only()` or `CpuOnly`, the explicit source configuration
can also be used for CPU comparisons without device execution.

## Source configuration and actual requests

Enabled registration changes only the builder's DataFusion session batch size;
other session settings, including target partitions, remain intact. This setting
applies to the session's CPU operators/sources as well as device fragments. It is
not an aggregate-only rewrite, a guarantee that all sources coalesce to that size,
or a promise about physical operators' output batch sizes.

Every batch entering CubeExec or CubeAggregateExec's worker is checked against
`max_request_rows` before adapter host-workspace reservation, Arrow adaptation,
device allocation or upload. Oversized input returns the typed
`BatchPolicyError::RequestTooLarge`, with row count and ceiling; it is never
automatically sliced, replayed or retried on CPU. Existing shared host/device
byte limits and per-request stage/output limits still apply, so a batch within
the row ceiling may still fail byte admission. Row counts are not byte estimates.

Options can be cloned and assigned different policies without changing the
original clone's policy. They still share the same controller/device gate and
byte limits. Plan child replacement retains its engine policy. Already-rewritten
Cube plans keep their original engine; rebuild a query with the new options to
apply a new policy rather than reusing an uncapped physical plan. This adds no new
buffer owner or device state; existing output reservations and cancellation/drain
lifetimes remain in force. Earlier filter/project output may have been consumed
before a later error; a global aggregate never publishes a partial answer.

## Diagnostics and measured tradeoffs

`placement_report` exposes the configured policy without a runtime or input
polling. Cube operators include it in EXPLAIN. `max_input_batch_rows` records
actual pulled input rows as a per-partition maximum, including a subsequently
rejected batch. It is distinct from the source target, input batch count and
completed request count. DataFusion may sum partition gauges in its aggregated
display; that sum is not the largest single request. `request_errors` includes
row-cap failure, while reservation-rejection counters describe actual admission
attempts. Shared-resource live/peak snapshots remain group-scoped.

The M1 study's 1M-row synthetic aggregate improved from 131.536 ms at 8K source
batches to 28.306 ms at 64K and 19.703 ms at 256K. Logical host peaks increased
from 10.02 MiB to 22.27 and 64.27 MiB. DataFusion CPU was still faster; these are
explicit configuration tradeoffs, not automatic thresholds or a production
speedup recommendation. Changing only total row count does not reproduce the
source-granularity experiment. See the report for repeated samples, CPU's best
matched configuration and the larger 1M-batch memory cost.

```sh
# Tiny correctness/EXPLAIN example, not a performance measurement.
cargo run --locked --all-features --example datafusion_aggregates -- metal 65536
cargo test --locked --all-features --test datafusion_batch_policy
cargo test --locked --no-default-features --features datafusion \
  --test datafusion_batch_policy
```

Ordinary F8 harness batch-size selection uses native SessionConfig directly for
matched baselines; this product API additionally makes the row ceiling explicit.
The policy does not imply cross-query residency or that the remaining launch,
packing, completion and export costs have been eliminated.

## Verification

On 2026-11-22 the final stack passed 174 all-feature tests on actual CubeCL CPU
and native Metal, plus 66 SQL-only tests without a linked device runtime. The
new suite verifies invalid sizes, preserved session settings, disabled/planning
behavior, exact aggregate/project results and schemas, measured input-row maxima,
pre-upload cap rejection, reuse after failure, and isolated options-clone policies.
No-feature all-target check, strict all-target Clippy/rustdoc, fmt and diff checks
passed. The Metal policy example returned `(70,2,2,30,40)`. The existing nonfatal
macOS unwind-table linker warning remains. Performance evidence is the matched
native-source study linked above, not a claim of a new reduction algorithm.
