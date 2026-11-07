# Device batches and explicit completion

`Executor::upload` creates an immutable `DeviceBatch` snapshot of exactly the
columns present in a borrowed host `Batch`. Host values, validity and parameters
may be released after submission. Changing host data never mutates a snapshot;
call upload again. Clones share the snapshot and its identity. There is no host
address cache, mutable device column API, or safe raw-handle import.

```rust
use cuberel::{Batch, ColumnData, DataType, Executor, Parameters, Schema, col, param, prepare};
# fn run(executor: &Executor) -> Result<(), Box<dyn std::error::Error>> {
let schema = Schema::new(&[("x", DataType::I32)], &[("bias", DataType::I32)])?;
let program = prepare(&(col("x") + param("bias")), &schema)?;
let batch = executor.upload(&Batch::new(3).with_column("x", ColumnData::I32(&[1, 2, 3]))?)?;
let a = executor.execute_resident(&program, &batch, &Parameters::new().with("bias", 10_i32)?)?;
let b = executor.execute_resident(&program, &batch, &Parameters::new().with("bias", 20_i32)?)?;
// No value, validity, error or count readbacks have happened yet.
assert_eq!(executor.stats().readback_calls, 0);
a.complete()?; // blocking completion and checked-I32 error-plane readback
let a_columns = a.materialize()?;
let b_columns = b.materialize()?;
# let _ = (a_columns, b_columns);
# Ok(()) }
```

`execute_resident` returns one pending column; `execute_many_resident` returns
columns in program order and validates all bindings/settings before any launch.
`DeviceResult::complete` blocks using CubeCL `sync_buffers` over **all** retained
inputs, outputs, validity and error planes, then checks checked-I32 errors.
`materialize` completes first, reads values/validity, and returns an owned
`Vec<HostColumn>` only if every operation succeeds. F32/I32/Bool columns expose
dense validity, including all-valid outputs; output type and declared nullability
remain available on the device result. Repeated completion/materialization is
valid and repeats synchronization/error reads, rather than caching success.

An I32 overflow in any output prevents returning all outputs. Backend failures,
unsupported settings and semantic errors remain distinct. A verifier-rejected
retained resource also prevents success even when primary outputs succeeded.
There is no CPU fallback, dummy readback per stage, general async scheduler or
promise of overlap. CubeCL's upload/allocation APIs can still panic.

`select_resident` compacts a predicate directly from resident columns, without
reading values/count. `DeviceSelection::belongs_to` checks the original batch
identity. Two selections of the same snapshot have the same source, regardless
of predicates or parameter values; independent uploads are different sources.
Existing `select`/`compact_mask` retain their unique-source behavior. Selected
projection, nested filters and Arrow compact export are F3, not F2 capabilities.

## Binding and ownership

- Every snapshot records row count, column names/types/nullability, private owned
  handles, runtime service identity and a unique batch identity. Columns enumerate
  in name order. Construct a host Batch containing a subset for on-demand upload;
  a query referencing a missing column fails before allocation or submission.
- Every column receives a value plane and dense U32 validity plane, including
  all-one validity for non-nullable columns. This deliberate extra 4N bytes lets
  nullable programs reuse non-nullable uploads without additional transfers.
  Nullable data with actual NULLs fails when bound to a non-nullable program;
  all-valid nullable data may bind, matching the synchronous API.
- Parameters bind per call. Snapshot identity and parameter values do not enter
  the expression compilation key. The existing synchronous expression APIs and
  selection API upload their referenced columns and share the resident submission
  core. Non-nullable convenience calls also upload the all-valid plane.
- Two executors can share a batch only if `Client::service_id()` is equal, the
  same rule as the existing device ABI. Backend/type names or hardware checksums
  alone are insufficient. Foreign services are rejected even for zero rows.
  Different cube sizes on the same service are allowed. No implicit migration.
- Result handles retain every bound input until completion/drop. Clones share
  immutable buffers; output allocations are distinct from inputs and each other.
  Dropping an unobserved result relies on the pinned CubeCL queued-binding
  ownership contract, as in F1. No custom resource pool is introduced.
- Empty input validates names/types/parameters and identity, with no buffers,
  launches or readback. Parameter-only programs still use explicit row count.
  The ABI retains the checked U32 active-byte bound (floor(U32_MAX/4) rows per
  plane). This is not U64 addressing or a unified F7 device-memory budget.

## Costs and reproduction

`Executor::stats()` reports cumulative explicit upload/readback calls and bytes,
kernel submission count, and selection-count reads. Executor clones share the
counters; independent executors have separate counters even on the same service.
Readbacks triggered by a result are charged to its submitting executor. Calls
count attempts, bytes count successfully transferred logical payloads. Concurrent
snapshots of counters are observational, not atomic transactions. Counters exclude
CubeCL scalar/metadata traffic, internal staging/copies, allocation costs, driver
compilations and allocator caches. They are not timing or peak-memory metrics.

Run the example on explicitly selected devices:

```sh
cargo run --offline --locked --all-features --example resident_batch -- cpu
cargo run --offline --locked --all-features --example resident_batch -- metal
cargo test --offline --locked --all-features
cargo test --offline --locked --no-default-features
cargo test --offline --locked --no-default-features --features arrow
cargo clippy --offline --locked --all-features --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --offline --locked --all-features --no-deps
cargo fmt --all -- --check
```

The example uploads one nullable I32 column (32 bytes in 2 planes) and submits a
predicate/selection plus a separate full-batch projection. Before materialization:
0 readbacks, 0 count reads, 4 launches (predicate, scan, scatter, projection).
This demonstrates shared input residency, not a selected projection or speedup.

## Local acceptance (2026-11-07)

On the development Apple host, Rust 1.99.0, pinned CubeCL revision and Arrow
60.0.0, using `/private/tmp/cuberel-build.hmThUI` as the separate Cargo target:

- CPU + native Metal + Arrow: 77 integration tests and 5 unit tests passed;
  no failures or ignored tests. Includes 5 resident integration tests and 2 new
  unit tests using the existing safe verifier-rejection fixture.
- No runtime: all-target check and 39 tests passed. Arrow-only: 47 tests passed.
- Clippy all features/targets and rustdoc passed with warnings denied; fmt/diff
  checks passed. Dependencies and the kernel ABI/codegen identity did not change.
- Both explicit CPU/Metal example runs returned indices `[2,3]` and full-batch
  projection `[20,NULL,60,80]`. Each reported upload 2 calls / 32 bytes, 4 launches,
  and **zero readback/count calls before materialization**. After materialization:
  5 reads / 60 bytes, including one 4-byte count, an 8-byte index prefix, and
  16-byte value, validity and I32 error planes. There was no second column upload.

No speedup or asynchronous overlap claim follows from these transfer counts.
F3 selected projection/gather and F7 unified budgets/performance remain separate.
