# Binding layout and CubeCL adapter

Milestone 01 work item 3 provides a versioned ABI shared by IR declarations and
runtime argument registration. It is product code for the later query emitter.
The device tests use a manually written IR probe. Automatic expression lowering
is now provided by [the executor](runtime-emitter.md); work item 5 adds
[normalized identity and scoped definition observation](kernel-identity.md).

## Layout and host binding

```rust
use cuberel::abi::BindingLayout;

let layout = BindingLayout::new(&program);
let inputs = layout.bind_host(&batch, &parameters)?;
```

See `examples/binding_layout.rs` for a complete runnable example. Layout creation
deduplicates used columns and parameters, then sorts each by schema ID. It
includes references in every validated arena node, even nodes unreachable from
the root, matching preparation. Unused schema fields
do not occupy slots. Input buffers precede the output planes. Inputs are read-only and outputs are
write-only. ABI v3 adds dense U32 validity beside nullable F32 inputs and outputs;
non-nullable Bool [predicate masks](predicates.md) retain U32 value output. See
[nullable projection ABI](nullable-projection.md) for exact ordering and contracts.
Column names, parameter names and internal slot names are separate namespaces.

The first scalar is physical U32 row count, ordinal 0 within U32. Used F32
parameter payloads follow in schema order, with ordinals 0, 1, and so on within F32.
Nullable parameters additionally have U32 validity ordinals starting at 1.
These ordinals are per type, not positions in one mixed scalar array. The caller's
insertion order does not affect binding. Bound parameter values are copied into
a snapshot; host column slices are borrowed. Parameter-only programs still have
a row count and output planes.

Layout equality/hash includes ABI version, address width and binding structure.
It excludes input addresses and parameter values. Two different computations
can have the same ABI: layout equality is not sufficient as a kernel cache key.

## CubeCL declaration and submission

`layout.declare(client, settings)` creates a fresh `KernelBuilder`, injects runtime
and device properties, and declares buffers and scalars in layout order. It
returns a context-local `KernelDeclaration` exposing the scope and values by
column/parameter ID. Its row count accessor returns IR IndexType after an
explicit cast from the physical U32 scalar; `row_count_raw()` exposes the U32
read. `build()` consumes the declaration. Never put these context-local values
into a persistent `KernelProgram`.

`DeviceBuffer::from_raw_parts(handle, element_type, logical_len)` imports an
opaque runtime handle. This is unsafe because metadata cannot prove that the
allocation is live, its base is aligned, or readable bytes contain initialized
values of the claimed type. Readable payloads, including NULL slots, must be initialized. Validity flags are
0/1; a write-only output need not be initialized.

`layout.bind_device(client, row_count, buffers, parameters)` checks all arguments
before touching a thread-local launcher. Buffers must be supplied in layout order,
including all output planes. It returns owned `ValidatedBindings` that retain handles
and scalar snapshots. Consuming `launch(client, settings, kernel, cube_count)`
registers the complete argument list without awaits or exposing a partially
populated launcher. A guard discards launcher state if registration unwinds.

Launch is unsafe because an arbitrary `CubeKernel` body must obey the supplied
layout, settings, bounds and synchronization rules. The adapter checks kernel
metadata address width, cube dimensions and execution mode, but metadata cannot
prove a kernel body accesses memory correctly. `Executor` wraps this boundary
with the private trusted expression emitter. Do not interleave third-party
launcher registration on the same thread.

`Submission::Enqueued` means submission, not successful execution. The caller
must observe the result of output read or synchronization. Zero rows still
validate arguments, then return `Submission::Empty` without compilation or
dispatch. Nonempty submissions currently require positive static grid dimensions.

## Range and ownership validation

The pure `validate_buffers` API checks descriptors without allocating device
memory. The device binding path also applies these checks:

- Exact buffer count, declared physical element type and logical length equal to row count.
- Checked start/end offsets, four-byte alignment and sufficient visible capacity.
- U32 row count, byte length and active byte endpoint; U64 addressing is rejected.
- Every handle belongs to the selected client service.
- Output does not share an allocation with any input, including disjoint views.
  Read-only inputs may alias each other.
- Required named scalars exist and have the expected type. Extra bindings are
  permitted, as with host input binding.

Extra allocation capacity is allowed, but each registered handle is narrowed to
exactly the logical byte length. In the pinned CubeCL revision, launcher metadata
comes from the handle's visible bytes, not `BufferArg`'s nominal length field.
Without narrowing, a padded allocation could expose the wrong length to IR.
Allocation size itself may exceed U32 if the active view's endpoint fits; the
adapter does not promise support for U64 addresses. Strides, vectors, NULLs and
in-place output are outside ABI version 3.

## Verification

At this work item's completion on 2026-11-01, the combined suite passed 35 tests.
After temporary-interpreter removal, work item 4 passed 29 integration tests:
12 frontend, 10 layout, 3 ABI device, 2 backend buffer and 2 expression-device
conformance tests. The ABI probes still execute real CPU and native Metal kernels
and read results. Work item 5 expands the full suite to 34 tests; see
`kernel-identity.md` for the current validation matrix.

The probes deliberately use schema order different from expression and supplied
argument order, two distinct F32 parameters, offset input views, spare allocation
capacity, and non-divisible row counts with guarded tail threads. They compare
results against a use-case-specific arithmetic formula and independently read IR
buffer-length metadata and the physical U32 count. They verify that output
padding stays untouched, parameters can be rebound, invalid bindings precede a
successful launch, empty invocations do not define a kernel, and a CPU handle
cannot be bound to a Metal client.

The pure tests cover canonical order, deduplication, parameter-only expressions,
namespace collisions, binding errors, length/capacity/alignment/range overflow,
empty batches, U32 limits and unsupported U64. These tests specifically verify ABI
transport; expression execution is tested separately in `tests/runtime_expr.rs`.

```sh
cargo test --locked --no-default-features
cargo run --locked --no-default-features --example binding_layout
cargo test --locked --all-features
cargo clippy --locked --all-features --all-targets -- -D warnings
cargo rustdoc --locked --no-default-features --lib -- -D warnings
cargo fmt --all -- --check
```

The verification used `/private/tmp/cuberel-build.hmThUI` as `--target-dir` and
`--offline` with already downloaded dependencies. CPU and Metal tests require
actual device access; a GPU sandbox initialization failure is not a passing skip.
