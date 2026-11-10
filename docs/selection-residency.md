# Selection-only residency and temporary retirement

Selections now keep the current original-row indices, stage counts, checked
predicate error planes and independent completion dependencies. Private WHERE
masks, unused Bool-validity outputs, scan workspace and unused ancestor indices
are released from Cuberel ownership after their final consumers are enqueued.
The runtime owns work still in flight; this does not force driver deallocation.

For `x = [10, NULL, 30, 40, 100]`:

```text
WHERE x >= 30 -> temporary mask -> selection [2,3,4], device count 3
WHERE x < 50  -> temporary mask -> selection [2,3],   device count 2
                                      |                     |
                                no ancestor mask       current indices
```

Each filter may still allocate its own capacity-N mask. Its selection does not
keep that mask after compaction is submitted. Indices still reserve `4*N` bytes,
not `4*selected_count`: this change does not add indirect dispatch or shrink
allocations after a count read. F6 aggregate fusion already avoids these masks
entirely on eligible paths; F6a applies to the ordinary compaction path.

## Completion history

Nested filters share a flat history of immutable per-stage checks, rather than
retaining ancestor `DeviceProjection`/`DeviceSelection` objects. Each check stores
its count, optional error plane and non-mask completion witnesses. A child no
longer retains its parent's indices merely for future error checks.

At an explicit exit, all retained witnesses and current outputs are synchronized.
Checks run oldest first. Dense predicate error flags use source capacity; selected
predicate flags encode original-row-plus-one and use the **parent's count** for
the initialized prefix. Using the child's count would hide errors on rows later
filtered out, especially when the child count is zero. F6's exit-local cache still
reads/checks a shared stage once per exit; every new exit checks again.

For example, select original rows `[2,3,4]` from `[0,1,2,3,I32_MAX]`, then evaluate
`WHERE x+1 < 0`. The child has no matching rows, but its checked error at original
row 4 must be reported. Discarding masks must not turn that query into success.

Counts and error planes are deliberately retained. With checked arithmetic,
capacity-sized error planes can still grow with filter depth; this feature does
not implement minimum-error-row reduction. [F7 budgets](resource-budgets.md)
account for those retained error planes. Source
buffers retained as completion witnesses are reported separately from indices,
counts and errors. External batches/outputs may also own the same allocations.

## Why dropping references is safe

The pinned CubeCL revision was audited locally:

- Queued CPU tasks own managed resources/bindings until their work is done.
- Native Metal resolves and encodes buffer resources on ordered streams; encoded
  command buffers and the runtime memory manager govern reuse/deallocation.
- Backend enqueue/verification failures propagate from read allocations to written
  allocations. Mask/scan/rank dependencies feed retained count or indices; a
  failed retired mask therefore still fails the selection's completion.
- Predicate launch failures also affect its consumed mask, while semantic checked
  errors remain in their own error plane. Independent witnesses are preserved
  instead of assuming every failure must belong to the main dataflow.

Sources at the pinned revision: [failure model](https://github.com/kingfolk/cubecl/blob/5037c0639ce3b1205392bdbe43dcf25e2ea09e29/ERROR_HANDLING.md),
[CPU task resources](https://github.com/kingfolk/cubecl/blob/5037c0639ce3b1205392bdbe43dcf25e2ea09e29/crates/cubecl-cpu/src/compute/server.rs),
[Metal enqueue](https://github.com/kingfolk/cubecl/blob/5037c0639ce3b1205392bdbe43dcf25e2ea09e29/crates/cubecl-metal/src/compute/server.rs).

Cuberel adds no per-stage sync, flush, background completion thread, dummy read,
count readback or custom buffer pool for retirement. Submission is still distinct
from successful execution. The change removes unnecessary application references;
CubeCL controls when queued pins can end and when cached storage can be reused.

Real CPU/Metal tests audit mask, validity and workspace allocations with weak
bindings. They check that no selection ownership path reaches those allocations
and, after consumer completion, that no extra strong descriptor owner survives.
This is an allocation-ownership check, not proof that the driver returned memory
to the OS. Tests also cover 16-stage chains, empty/all-NULL/full/tail inputs,
unfinished drops, cloning, cross-thread completion, original-row errors, retired
mask failures and independent retained-resource failures.

## Shared readers and public outputs

A caller holding a parent selection or a projection branch keeps that object's
own indices/outputs valid. Creating a child does not mutate or forcibly reclaim
external readers. Public Bool projection and explicit mask APIs keep their
existing output semantics; only internal WHERE predicate outputs are retired.
Projection/aggregate consumers continue to guard device-count prefixes and retain
the completion checks they require.

## Inspect and reproduce

`DeviceSelection::residency()` performs no device operation. It returns
`SelectionResidency`: stage count, current index bytes, history count bytes,
checked-error bytes and other dependency bytes, deduplicated by allocation.
`total_bytes()` sums those fields. Reports from different externally held objects
must not simply be added: they can share allocations.

`SelectionStats::workspace_bytes` still describes requested compaction allocations,
not retained bytes or a physical peak. Runtime in-flight ownership, staging,
shared memory, allocator padding/caches and external owners are outside the
Cuberel-owned residency metric.

```sh
cargo run --locked --release --all-features --example selection_residency -- cpu 990 65536 7 4
cargo run --locked --release --all-features --example selection_residency -- metal 990 65536 7 4
cargo test --locked --all-features --lib selection_retirement
```

The example measures submission through `selected_count()` completion, validates
counts independently and checks zero reads during submission. It records runtime
pool usage/high-water pages only after the timed exit. Full before/after evidence
is in the [F6a residency report](benchmarks/selection-residency-m1-2026-11-09.md).

Validation: 124 all-feature tests (17 unit + 107 integration) passed on real CPU
and native Metal with Arrow enabled; 39 tests passed without runtimes and 48 with
Arrow only. Strict all-target Clippy, rustdoc with warnings denied, formatting
and diff checks passed. The release example and baseline adapter produced the
recorded measurements without adding production-stage synchronization.

F7 [resource budgets](resource-budgets.md) include these lifecycle differences:
admission conservatively counts all requested buffers that may be in flight,
while the retention bound credits retired private masks/workspace/ancestor indices.
