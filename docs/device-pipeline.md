# Resident stage graphs and safe fusion

`Executor::execute_pipeline` submits a bounded query fragment over an existing
`DeviceBatch`. `Pipeline` collects ordered Bool filters, named projections and
named global aggregates. Expressions bind original source columns; output aliases
cannot feed other outputs. The limits are 16 filters and 32 total outputs, with
nonempty, unique output names across both branches. This is a fixed stage planner,
not SQL planning, a cost optimizer or a general asynchronous scheduler.

```rust
use cuberel::{Aggregate, FusionMode, Pipeline};
// x and predicate are prepared KernelPrograms; batch is an uploaded DeviceBatch.
let query = Pipeline::new()
    .filter(predicate)
    .aggregate("rows", Aggregate::CountStar)
    .aggregate("sum", Aggregate::Sum(x))
    .fusion(FusionMode::Auto);
let pending = executor.execute_pipeline(&query, &batch, &parameters)?;
println!("{:?}", pending.graph().stages());
let result = pending.materialize()?;
```

All filter, projection and aggregate names, types, bindings, capacities and device
limits validate before any stage is submitted. Submission does not upload source
columns again, read a selected count, or read intermediate results. Sources and
parameters may be dropped afterwards: pending objects retain immutable buffers
and scalar snapshots. The executor's selected device is always used.

## Plans and fusion boundaries

`FusionMode::Unfused` is the executable reference plan. A filter becomes predicate,
scan levels, offset propagation and stable scatter. All terminal projections and
aggregate arguments share that selection chain. Aggregates retain their existing
partial and merge kernels. Projection outputs still have independent kernels;
sharing selection does not imply a single projection kernel.

`Auto` fuses aggregate-only fragments with zero or one predicate when layouts and
device limits allow it. The first kernel evaluates the predicate, evaluates
arguments inside its valid-true guard, and builds all aggregate partial states in
one source scan. Common argument expression nodes are interned by typed structure,
without changing operand order or arithmetic grouping. It avoids mask, selection
indices and full argument projection arrays on this path. Later levels reuse the
existing merge kernels, one per aggregate state. A filtered fused fragment also
reduces a hidden participating-row count, including for COUNT(*).

Unsupported patterns, conflicting input declarations or insufficient fused shared
memory/bindings take the reference device path in Auto. `Required` instead returns
`InvalidPipeline` before launch. Metadata-only unfiltered COUNT(*) has no kernel;
it uses the reference path in Auto. There is no fallback to CPU after device or
semantic failure.

Filtered F32 SUM always uses compact selection: fusing a predicate would change
which rows are paired by the specified F32 reduction tree. Auto reports this
reason; Required rejects it. Unfiltered F32 SUM preserves the original pairwise
tree bit for bit on the selected backend. COUNT/MIN/MAX, exact widening I32 SUM,
nullable states, canonical NaNs and signed-zero MIN/MAX use the existing policy.
This is not a cross-backend reproducibility claim for general F32 SUM.

`StageGraph::stages()` is topologically ordered, with earlier dependency indices.
It includes resident-input/materialization boundaries and actual kernel stages;
`kernel_launches()` excludes boundaries. Merge dependencies conservatively follow
submission order. Empty input has no kernel stages. `is_fused()` describes actual
launches, so it is false on empty input. `fusion_reason()` explains reference-plan
selection; it is absent when an eligible fused plan was selected, including empty
input. Kernel identities include stage/version, expression/ABI structure,
aggregate kinds and roots, predicate boundary, device and tile settings. They
exclude row count, parameter values, addresses, output aliases and payload data.

## Completion, errors and export

`DevicePipeline::complete()` synchronizes all retained branch resources together,
then validates dependencies and errors and returns the participating row count.
`materialize()` additionally reads compact projection prefixes and scalar results.
An exit-local context reads each shared selection's count once and avoids
rechecking its predicates across branches. Each new completion/export repeats
checks; no persistent success cache hides a later backend failure.

Earlier filter-stage failures take precedence. Terminal expression failures
report the smallest original source row across projections and aggregate
arguments. Fused predicates retain a separate error plane; false/NULL predicate
rows cannot hide predicate errors. Checked I32 arguments only execute for active
rows. Backend failures are checked before count/error/payload reads; no partial
host result is returned if any branch fails. Checked arithmetic still needs error
readback at this explicit exit, and filtered execution needs a count read there.

`PipelineResult` separates `projections` (participating rows) and `aggregates`
(one named scalar each); aggregates are not broadcast across projection rows.
With `arrow`, `materialize_arrow()` or the completed result's `into_arrow()` returns
`arrow::PipelineArrowResult`: separate optional projection and one-row aggregate
RecordBatches plus `row_count`. Absent output branches are `None`; an empty
projection still has its schema and zero rows. Arrow packing happens on host.

The reference compaction path now uses [selection-only residency](selection-residency.md):
private masks/validity/workspace retire after enqueue of their final consumers,
while count/error checks remain. This is separate from avoiding allocations through
fusion; neither change adds a custom allocator.

## Reproduce

```sh
cargo run --locked --example device_pipeline -- cpu
cargo run --locked --no-default-features --features metal --example device_pipeline -- metal
cargo test --locked --all-features --test pipeline
```

For `x=[10,NULL,30,40]`, `WHERE x>=30`, COUNT(*), COUNT(x), SUM(x), MIN(x), MAX(x):
reference submission uses 11 launches; fused submission uses 1. Both return two
participating rows and scalars `[2,2,70,30,40]`. Both submit with zero additional
uploads/readbacks. Adding `x*2` as a projection uses one shared selection and
returns `[60,80]` alongside the scalar SUM 70. These are logical launch/transfer
counts, not latency or physical allocator-memory measurements.

Validation on 2026-11-09 used the pinned CubeCL revision and Arrow 60:
122 all-feature tests passed on real CPU/native Metal (107 integration, 15 unit),
39 tests without runtimes and 48 with Arrow only. Strict Clippy, rustdoc and format
checks passed. The suite compares reference/fused values and errors, multi-level
tails, nullable counts, wide carries, F32 trees, original-row errors, retained
resource failures, and clone/drop/cross-thread ownership.

A warmed release comparison with timings, raw samples and reproduction commands
is available in the [Apple M1 fusion benchmark](benchmarks/fusion-m1-2026-11-09.md).
