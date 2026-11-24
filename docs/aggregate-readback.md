# Opt-in batched aggregate scalar readback

`Executor::with_aggregate_readback_mode(AggregateReadbackMode::Batched)` groups
aggregate scalar reads through CubeCL's existing multi-buffer API. `Scalar`
remains the default and an executable reference. This is a host read-scheduling
option; it changes no kernel, reduction tree, input residency or SQL allow-list.

```rust,ignore
use cuberel::{AggregateReadbackMode, Executor};
use cuberel::datafusion::{AdapterOptions, BatchPolicy};

let executor = Executor::new(device.client())?
    .with_aggregate_readback_mode(AggregateReadbackMode::Batched);
let options = AdapterOptions::with_executor(executor)
    .with_batch_policy(BatchPolicy::new(65_536, 65_536)?);
```

The executor getter exposes the chosen mode. A pending aggregate captures its
mode when submitted; changing another executor clone affects only subsequent
results. Clones still share device/counters as before. Kernel identities do not
include this host-only choice. Repeated materialization repeats checks and reads;
there is no persistent success cache.

## Ordered checks and payload suppression

Existing synchronization over retained resources, predicate/error checks,
selection/participating row count and checked argument dependencies happen first.
For fused filtered aggregates, the existing row-count read remains separate.
After completion succeeds, Batched performs:

1. Read all present aggregate count planes in one API request. Validate every
   count against participating rows before requesting numeric payloads.
2. Read only the valid non-COUNT low/high payload planes in one API request.
   Empty/all-NULL states never fetch unused numeric payloads. No handles means
   no API call. Decode the same named values in request order.

Plane count and four-byte word lengths are checked before decoding or returning
a result. Any failure returns an error without partial results or a CPU retry.
SUM's high word, NULL versus zero, F32 canonical bits, overflow and error-order
contracts remain intact. No asynchronous worker, device allocation, packing
kernel or cross-batch accumulator is added; the original worker permit, input
and output ownership/drain rules remain in force.

## Calls, buffers and bytes

`ExecutionStats::readback_calls` counts explicit runtime read API invocations
(including failures), which can now contain multiple buffers. It does not count
physical copies, command buffers or driver submissions. The separate
`Executor::readback_buffers()` counts attempted buffer descriptors, including
failures and shared with clones. Existing ExecutionStats struct literals remain
compatible. `readback_bytes` counts validated successful payloads; a malformed
batched response contributes no successful bytes. Implicit driver transfers are
outside these observations.

SQL exposes `observed_readback_buffers` alongside observed calls/bytes, and shows
the configured mode in EXPLAIN/placement reports. No executor means no configured
device mode; a displayed mode does not imply an unsupported aggregate was placed.
The harness preserves mode and buffer deltas and verifies them against actual
plan counters. Count/error plane reads remain included; selection-count reads
retain their existing separate definition.

For the existing five-output filtered integer aggregate, each nonempty result
still has nine four-byte descriptors. Scalar makes nine calls; Batched makes
three (participating-row count, aggregate counts, valid payloads). For a plain
resident result with host-known row count, eight calls become two. Bytes and
kernel launches remain unchanged. This is not transfer compression.

## Performance boundary

The [same-binary study](benchmarks/sql-m4-readback-study.md) tests the option after
[phase attribution](benchmarks/sql-m4-phase-study.md). At 1M input/64K source rows,
the median of three process p50s improved from 21.114 to 18.353 ms (about 13% less
time). At 256K the improvement was about 7%; at 1M source rows the small difference
was not reliable across repeats. DataFusion CPU remained faster in every tested
configuration. Keep mode selection explicit and account for source-size/memory
tradeoffs; no automatic threshold or general GPU speedup is installed.

```sh
# One existing SQL fixture, including all checks and host input/export:
cargo run --locked --release --all-features --example sql_acceptance -- \
  metal sql aggregate 1048576 50 20 1 1 10 65536 batched
cargo test --locked --all-features --test aggregate_readback
```

F32 behavior is tested through the existing direct DSL; this does not extend SQL
F32 aggregation support. Further GPU launch/merge and Arrow/encoding costs remain
separate work, and cross-batch device accumulation remains deferred.

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
