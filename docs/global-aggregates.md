# Single-batch global COUNT / MIN / MAX

F4 builds on [resident batches](device-batch.md) and
[selected execution](selected-projection.md). Global means all logical rows in
one input batch or its stable selection; it does not mean distributed, streaming
or cross-partition aggregation. SUM is added by the [F5 extension](global-sum.md); GROUP BY and SQL integration remain separate.

```rust
use cuberel::Aggregate;
// x and predicate are prepared KernelPrograms; batch is a DeviceBatch.
let selected = executor.select_resident(&predicate, &batch, &params)?;
let request = [
    ("rows", Aggregate::CountStar),
    ("valid", Aggregate::Count(x.clone())),
    ("min", Aggregate::Min(x.clone())),
    ("max", Aggregate::Max(x.clone())),
];
let pending = executor.aggregate_selected(&batch, &selected, &request, &params)?;
// Or aggregate_resident(&batch, &request, &params) for every batch row.
let scalars = pending.materialize()?;
// With feature arrow: pending.materialize_arrow()? returns one RecordBatch row.
```

Names must be nonempty and unique; requests must be nonempty. All types, input
bindings, names, service/source identity, grids and reduction capabilities are
validated before argument kernels/allocations. MIN/MAX accept I32/F32; COUNT
accepts current numeric and Bool expression results. `COUNT(false)` counts one
valid value, not zero true flags. No implicit casts or generic U64 inputs are added.

| Operation | NULL handling | Empty / all NULL | Host output |
| --- | --- | --- | --- |
| COUNT(*) | Counts every participating row | 0 | `AggregateValue::Count(u64)` |
| COUNT(expr) | Ignores NULL results | 0 | `AggregateValue::Count(u64)` |
| MIN/MAX(I32) | Ignores NULL results | None | `AggregateValue::I32(Option<i32>)` |
| MIN/MAX(F32) | Ignores NULL results | None | `AggregateValue::F32(Option<f32>)` |

COUNT(*) uses existing row metadata (or the selection's device count) and needs
no reduction kernel. The device expression/partial-state count ABI remains U32,
with the existing floor(U32_MAX/4) row bound. Converting a final count to host U64
does not imply U64 device indexing or support for more rows.

## Partial-state execution

Each argument executes through the existing checked full-batch or selected IR
path. First-stage block reduction consumes `(value bits, validity)` and produces
`(value bits, valid count)` per block. Merge stages recursively consume partial
counts (not 0/1 flags) until one state remains. Count and payload use two U32
planes, independent of the logical I32/F32 type. Missing/non-nullable validity and
static lengths use compile-time modes, with private initialized dummy read-only
bindings that are never read. Selected first stages guard source capacity and
device count before any argument-prefix loads.

Every lane initializes shared memory, including NULLs and padded tails. Fixed
pairwise tree rounds update disjoint lane pairs and synchronize all lanes after
each round. There are no per-row global atomics, cross-block barriers, spin waits,
host value evaluation or query `#[cube]` bodies. Each merge is an independent
launch, using the pinned runtime's buffer dependency tracking. Tile >=2 is required
for argument reduction; COUNT(*) does not need that capability. Non-power-of-two
tiles are supported. Two shared U32 arrays cost tile*8 logical shared bytes.

All partial allocations, summed request bytes, padded grids and launch counts are
checked before submission. `AggregateStats` reports maximum reduction depth,
reduction launch count and all logical partial-state bytes (including final
states). It excludes argument/selection buffers, runtime metadata/staging,
allocator caches and driver peak memory. Every argument currently has its own
expression and reduction chain; shared scans/fusion are F6, budgets/performance F7.

## Floating-point and error contract

Any valid NaN produces canonical `0x7fc00000`, regardless of its original payload,
sign or signaling bit. MIN(-0,+0) is -0; MAX(-0,+0) is +0. Infinities retain their
usual order. Comparison uses ordered IEEE bit keys, so subnormal ordering and zero
signs do not depend on backend floating-point comparison/flush modes. The first
stage canonicalizes valid NaN payloads; every merge propagates them. NULL states
are ignored using counts, never payload sentinels. No F32 summation/reassociation
policy is introduced. This profile has not been claimed equivalent to DataFusion;
that compatibility check belongs to the later integration milestone.

Checked I32 argument nodes must still execute. `COUNT(x+1)` cannot discard an
overflow merely because it only counts results. Selected arguments only evaluate
active rows, with original-row diagnostics; skipped overflows remain skipped.
Ancestor predicate failures remain observable, including for COUNT(*) and empty
selections. Across aggregate arguments, the smallest original overflow row wins;
earlier predicate-stage failures take precedence. No host results are returned
until all dependencies/errors pass; backend failure never falls back to CPU.

## Completion, lifetime and Arrow

`DeviceAggregates` retains every input/argument/error/partial buffer and selected
dependency. Host inputs and parameters, batch and selection may be dropped after
submission. Clones share immutable results. `complete` is explicitly blocking and
returns participating row count after synchronization and error checks.
`materialize` reads final count/value words and returns named scalars in request
order; empty/all-NULL MIN/MAX never require reading a value word. Zero source rows
validate bindings but allocate, launch and read nothing. Repeated exports repeat
checks/readback. Selected arguments currently repeat upstream count/error checks
at export; no count is read during stage submission. CubeCL allocation/transport
APIs may still panic, as in existing execution APIs.

With `arrow`, output always contains one row: COUNT columns are non-nullable
UInt64, and MIN/MAX are nullable Int32/Float32 (NULL for empty input). Input Arrow
allow-lists remain unchanged. UInt64 output is not a claim of equivalence to a
future DataFusion COUNT return type; that boundary needs explicit conversion.

Reduction keys include primitive version, IEEE NaN/zero bit-order policy, kind,
first/merge/nullable/selected modes, settings/tile and target/device identity.
Values, parameter values, addresses, selection contents and row counts are absent.
Existing expression codegen and selection identities are unchanged by F4.

## Reproduction

```sh
cargo test --offline --locked --all-features
cargo check --offline --locked --no-default-features --all-targets
cargo test --offline --locked --no-default-features
cargo test --offline --locked --no-default-features --features arrow
cargo clippy --offline --locked --all-features --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --offline --locked --all-features --no-deps
cargo run --offline --locked --all-features --example global_aggregates -- cpu
cargo run --offline --locked --all-features --example global_aggregates -- metal
cargo fmt --all -- --check
```

## Local acceptance (2026-11-08)

macOS 15.5, Rust 1.99.0, pinned CubeCL revision and Arrow 60.0.0; separate Cargo
target `/private/tmp/cuberel-build.hmThUI`:

- CPU + real native Metal + Arrow: 101 tests passed (92 integration, 9 unit),
  zero failures or ignored tests. F4 adds 7 runtime tests, 2 Arrow export tests
  and 2 safe retained-resource/count-corruption unit tests. No generated
  cross-product or host query interpreter was added.
- Submission-time recheck against merged F3: 18 unit/aggregate/Arrow tests passed.
- No-runtime all-target check passed; no-runtime 39 tests and Arrow-only 47 tests
  passed. All existing expression, selection, residency and projection tests pass.
- Clippy all features/targets and rustdoc passed with warnings denied; fmt and
  diff checks passed. No dependency, old expression or selection identity change.
- Integer cases cover recursive merges, nullable counts >1 per partial,
  non-power-of-two tile 3, padded tails, extremes, zero and empty selections.
  Float cases check exact NaN/zero/Inf/finite/subnormal bits and repeated export.
  Failure checks cover COUNT argument overflow, original-row minimum across
  arguments, retained upstream failures, independent failed resources, invalid
  final count rejection, failed-request recovery, clone/drop and cross-thread use.

Both CPU/Metal examples returned the one-row Arrow result
`rows=2, valid=2, min=30, max=40`. Each uploaded 32 bytes once (two planes), submitted
9 kernels and read no count/payload before export. Reduction-only statistics were
one level, three launches and 24 partial-state bytes. Export reported nine reads /
36 bytes, including four repeated upstream count checks and five final scalar
words. These are logical counters, not GPU bus/driver peak/timing measurements;
no speedup is claimed.

F4 was developed and accepted locally, then prepared for PR submission at the
user's request. Its base includes merged F3 PR #13. Local milestone/roadmap
planning files are excluded from the product change.

[F5 SUM](global-sum.md) adds nullable F32 and widening I32→I64 totals, extends
partial-state accounting by word width, and advances reducer identity to v2.
