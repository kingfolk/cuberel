# Selected projection, nested filters and compact Arrow output

F3 builds on [device batches](device-batch.md) and [stable selection](stable-selection.md).
A selection remains a stable list of **original batch row IDs**, with a device
count and source capacity. It is not a materialized smaller batch.

```rust
// Given an Executor, a resident DeviceBatch and prepared programs:
let first = executor.select_resident(&predicate, &batch, &params)?;
let second = executor.filter_selected(&next_predicate, &batch, &first, &params)?;
let pending = executor.project_selected(
    &batch, &second, &[("amount", &amount), ("condition", &condition)], &params,
)?;
// No intermediate count, mask or indices have been read back.
let compact_columns = pending.materialize()?;
// With feature `arrow`, use pending.materialize_arrow()? instead.
```

`project_selected` requires one or more named programs, validates every name and
binding before allocating/submitting any output, and launches one kernel per
expression. Names must be unique and nonempty. Output order follows the program
slice. F32, checked I32 and Bool results retain their prepared nullability.
`SelectedColumn` pairs an owned `HostColumn` with an owned output `Field`.

`DeviceProjection::capacity` is the original row-count allocation upper bound.
`complete` returns the actual selected count after all backend/dependency/error
checks; `materialize` exports only that many values and validity flags. Output
clones share immutable device buffers; host snapshots, source batch and selection
may be dropped before completion. All bound buffers and upstream dependencies
remain owned until completion or runtime-safe drop.

## Device execution and active-row semantics

The private selected expression ABI preserves the ordinary bindings, appending
read-only U32 selection-indices and count buffers. The host launches using source
capacity. IR first guards the global slot by capacity, loads the device count,
guards the slot by count, then loads the original row index and checks it against
source rows. Only inside these guards does it load source values/validity and
compute the expression. Results are stored at the compact output slot. No host
gather, expanded input copy, host expression evaluator or query `#[cube]` is used.

`filter_selected` evaluates its predicate only over the incoming selected rows.
The first scan level reads predicate masks only below the parent's device count;
other lanes contribute zero and still participate in barriers. Recursive counts
and offset propagation use the established scan. Selected scatter stores
`parent_indices[slot]`, preserving original coordinates and stable ordering.
There is no host count read between these stages.

Checked expressions retain eager, reachable-node semantics **within active rows**.
For `x=[I32_MAX,1]`, `WHERE x<10`, projection `x+1` returns `[2]`. The unselected
MAX row does not produce a projection error. The ordinary full-batch expression
API still reports its overflow. The first predicate checks all original rows;
subsequent predicates check their incoming selected rows. A failed ancestor
predicate is retained and cannot be hidden by a later empty filter/projection.

Selected I32 error slots encode `original_row+1`, with zero meaning success.
The conservative U32 byte bound proves this addition cannot overflow. Completion
checks all output error prefixes and reports the smallest original row across
those outputs. Earlier predicate-stage errors take precedence over downstream
results. Backend failures on any retained resource prevent returning partial
columns, even if primary output handles succeeded.

A selection from a different upload is rejected before submission, including an
equal-length/equal-value batch. A foreign runtime service is rejected by the
existing resident binding checks. No API accepts arbitrary indices, reordered
selections, duplicate rows, in-place gather or raw handle imports.

## Compact readback and Arrow

`materialize_arrow` returns an Arrow 60 `RecordBatch` whose row count equals the
device count. Names/order/types/prepared nullability are preserved. Bool payloads
and validity are packed into Arrow bitmaps on host; numeric owned vectors move
into Arrow arrays. This is not zero-copy GPU Arrow. Existing `ArrowBatch` adapters
continue to normalize sliced payload/validity offsets before explicit upload.

Every readback narrows the logical handle to the initialized prefix, including
selected error planes. A zero selection reads its count/dependency checks but
reads no output/indices tails. A zero-source batch validates inputs and names,
then allocates/launches/reads nothing. Both real CPU and Metal tests verify the
logical readback-byte counts; backend staging/internal copies remain outside
Cuberel's counters. The pinned runtime's `Client::read_async` derives copy shape
from `Handle::size_in_used`; CPU `read_async` uses that visible length, and native
Metal `read` passes the narrowed shape to `read_pitched` over shared buffer
contents. Thus neither backend exposes an uninitialized full-capacity tail here;
these logical bytes are not a measurement of GPU bus traffic.

Completion is explicitly blocking. It synchronizes output resources, checks the
selection and its ancestors, reads counts and error prefixes, then exports value
and validity prefixes. Shared ancestor counts/checks are memoized within each explicit exit; repeated
exports repeat checks. Counters are charged to the executor owning each pending
stage. F6 provides bounded pipelines; general request budgets remain F7.

Each output allocates 4N value bytes, another 4N for nullable validity and another
4N for checked error planes when required. Nested filters keep compact completion histories rather than ancestor masks,
projections and scan workspace. External parent readers keep their own buffers
valid; see [selection-only residency](selection-residency.md). All sizes retain
the checked U32-byte bounds.
Extra selected-expression bindings are preflighted; nested compaction needs six
buffers plus runtime metadata (seven bindings). Unified request budgets remain F7.

The shared expression emitter now uses codegen **v9**, with a distinct
`selected-prefix-v1` identity tag. Selection primitive identity is **v2** because
its stage ABI now includes selected scan/scatter; this also prevents enum-tag
collisions with old cached stages. The public flat binding ABI remains v4.
Neither counts, parameter values nor selection contents enter structural keys.

## Reproduction and local evidence

```sh
cargo test --offline --locked --all-features
cargo check --offline --locked --no-default-features --all-targets
cargo test --offline --locked --no-default-features
cargo test --offline --locked --no-default-features --features arrow
cargo clippy --offline --locked --all-features --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --offline --locked --all-features --no-deps
cargo run --offline --locked --all-features --example selected_projection -- cpu
cargo run --offline --locked --all-features --example selected_projection -- metal
cargo fmt --all -- --check
```

Local acceptance on 2026-11-07: macOS 15.5, Rust 1.99.0, the pinned CubeCL
revision and Arrow 60.0.0; separate Cargo target `/private/tmp/cuberel-build.hmThUI`.

- CPU + real native Metal + Arrow: 90 tests passed (83 integration, 7 unit),
  zero failed or ignored. F3 adds 4 pipeline/error tests, 2 sliced Arrow tests
  and 2 safe retained-resource/identity tests, with no generated cross-product.
- No-runtime all-target check passed; no-runtime 39 tests and Arrow-only 47 tests
  passed. Full legacy expression, binding, I32, nullable and selection regressions
  remain passing. Selected overflows do not change eager full-batch behavior.
- Clippy all features/targets and rustdoc passed with warnings denied. fmt/diff
  checks passed. Dependencies remain pinned and unchanged.

The example runs
`x=[10,NULL,30,40,100]`, `WHERE x>=30`, then `WHERE x<50`, and projects
`twice=[60,80]`, `greater=[false,true]` into two compact rows. It explicitly
asserts no uploads beyond the initial snapshot and no pre-export readbacks/count
reads. No performance speedup is claimed.

Both explicit CPU/Metal example runs produced the same two-row Arrow result and
reported 2 uploads / 40 bytes, 8 launches, and 0 pre-export readbacks/count reads.
At export, each reported 7 reads / 48 bytes, including two 4-byte ancestor/current
counts, an 8-byte selected error prefix, and four 8-byte value/validity prefixes.
The input was never re-uploaded, and no selection-index prefix needed host export.
