# Runtime expression emission and execution

Milestone 01 work item 4 compiles a prepared expression to CubeCL IR and executes
it on a selected client. `Executor::execute(program, batch, parameters)` is a safe,
synchronous API returning `Result<Vec<f32>, ExecutionError>`. It supports column
and scalar leaves, addition and multiplication of scalar F32 values.
[Nullable F32 projection](nullable-projection.md) now uses explicit value/validity
planes and `Executor::execute_nullable` for nullable roots.
[Predicate execution](predicates.md) now adds F32 comparisons and boolean
composition through `Executor::execute_mask`, returning dense U32 flags.
No Rust source or `#[cube]` kernel is generated, and query execution does not call
rustc. See `examples/runtime_expr.rs` for add, multiply, combined expressions and
parameter rebinding in one process.

## Execution boundary

`Executor::new(client)` chooses `min(64, max_units_per_cube, max_cube_dim.x)` threads
per cube. `with_cube_size(client, size)` accepts an explicit positive size within
device limits. The executor uses a checked U32 kernel with a one-dimensional
static grid. It rejects a grid exceeding the device x limit instead of silently
truncating it or distributing it over other dimensions.

Every invocation first creates the value-independent `BindingLayout`, validates
named host columns/parameters and U32 extents, and checks device binding/grid
limits. It uploads only required input columns, allocates a distinct output,
validates device handles and submits through the same layout. Host inputs are
serialized as native F32 bytes. This API uploads and reads back on every call;
it is not a device-resident batch or asynchronous stream API.

For zero rows, required input validation still runs. Then execution returns an
empty vector without allocation, definition or launch. A parameter-only expression
broadcasts over the batch's explicit row count. Every validated arena node
participates in binding, including nodes unreachable from the root; no dead-code
frontend optimization is promised.

The executor returns output only after `client.read_one(output)` succeeds and
the returned byte length is checked. Backend compilation/execution/read errors
reported by CubeCL retain their `ServerError` in `ExecutionError::Backend`, with
a standard error source. There is no host fallback or unchecked read. CubeCL's
infallible allocation/transport methods can still panic; this API does not claim
to turn allocator failure, process abort or every transport panic into a Result.

## IR construction and ownership

The private `ExpressionKernel` holds immutable program data, the layout, client
and settings. Its `CubeKernel::define` creates a fresh builder/declaration and
injects target/device properties. A host-side loop traverses the topological arena
once, mapping expression IDs to context-local SSA values. Shared DAG references
reuse the emitted value; nodes are not recursively compiled.

The kernel reads AbsolutePos and the physical U32 row-count scalar converted to
IR IndexType. A structured `index < row_count` branch contains all column loads,
arithmetic and the root store. Both regions terminate with yield; the entry
function explicitly returns. The complete module is verified before builder
finalization. A verifier assertion inside define becomes CubeCL's
`CompilationError::ExpansionPanicked`, observed through checked output read.
Backend lowering performs its own subsequent verification.

No Value, Scope or pliron Context escapes into `KernelProgram`. The safe executor
is the owner of the unsafe ABI import/launch boundary: it controls allocation,
buffer types, logical lengths, access modes, body generation and grid settings.

## Floating-point lowering

Fast-math flags are empty. The pinned CubeCL `InstCombinePass` nevertheless
unconditionally merges FMul followed by FAdd into FMA on CPU and Metal, changing
a separately rounded expression. Direct emission exposed this in the fixture
`4097 * 4097 + (-16785408)`: separate operations yield 0; fused operations yield 1.

To preserve the node boundary without modifying the dependency, addition lowers
to `FmaOp(1.0, lhs, rhs)` and multiplication to `FMulOp(lhs, rhs)`. Multiplication
by exactly 1 is exact, so this FMA implements one rounded addition. It does not
combine multiplication from a preceding expression node, and does not match
CubeCL's FMul/FAdd fusion pattern. Existing grouping, cancellation and signed-zero
multiplication fixtures run on both actual backends. This workaround is explicit
in the emitter and codegen version; it can be simplified when CubeCL provides a
reliable strict floating-point policy. Its performance has not been benchmarked.

Finite normal arithmetic is the tested contract. Non-exact finite cases use
`abs_error <= 1e-6 + 1e-5 * abs(expected)`; binary-exact fixtures use exact asserts.
Nullable add/mul semantics are defined by [nullable projection](nullable-projection.md),
and comparison edges by [predicates](predicates.md). Overflow rules, denormal
handling and general cross-device bitwise identity remain outside this feature.

## Identity and remaining work

Correct execution already requires different operations with the same ABI to
have different kernel IDs. The minimal identity includes codegen version, typed
arena nodes/root, BindingLayout and device-properties checksum; CubeCL's KernelId
also includes dimensions, address type and execution mode. Input handles, row
count and parameter values are arguments, not specialization constants.

This uses CubeCL's existing compilation cache and does not add a Cuberel cache.
Work item 5 now provides [normalized identity, scoped define observation and
isolated failure tests](kernel-identity.md). Correct repeated outputs alone are
not a cache-hit measurement; define counts do not count driver compilations.

## Temporary interpreter removal

As required for this work item, the temporary host evaluator and its example have
been removed. Shared Batch, ColumnData, Parameters and ScalarValue moved from the
former reference module to `input.rs` with their root exports preserved. Preparation
tests remain device independent; numerical, grouping and DAG cases now execute
real kernels and assert explicit expected arrays or case-specific formulas. The
ABI probe also uses a direct formula. No generic AST interpreter remains in tests.

## Verification report

Verified 2026-11-01 on macOS/arm64, Rust/Cargo 1.99.0, using the unchanged pinned
CubeCL `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`. The local CubeCL checkout was clean;
the Cargo dependency remains a fixed git revision, without a local path override.
This work item was subsequently merged in PR #4 (implementation commit 3b084a4).

At work item 4 completion, the combined suite had 29 tests: 12 frontend, 10 layout, 3 ABI-device, 2 backend
buffer and 2 runtime-expression conformance tests. Each conformance test checks
many expression shapes, parameter sets and lengths, including 0, 1, B-1, B, B+1,
3B, 3B+7 and 4099 rows. Cases cover schema/binding order differences, two F32
parameters, repeated shape switching, parameter-only broadcasting, column copying,
shared DAG operands, a root before the last arena node, arithmetic grouping,
non-fused product/addition rounding, signed zero, finite-value tolerance, U32
overflow rejection without allocation, invalid bindings and a subsequent valid
invocation. The ABI probe retains offset/padded-allocation and untouched-output
padding checks. No backend substitutes for a missing GPU.

```sh
cargo test --locked --offline --all-features
cargo test --locked --offline --no-default-features
cargo run --locked --offline --example runtime_expr -- cpu
cargo run --locked --offline --no-default-features --features metal --example runtime_expr -- metal
cargo clippy --locked --offline --all-features --all-targets -- -D warnings
cargo rustdoc --locked --offline --no-default-features --lib -- -D warnings
cargo fmt --all -- --check
```

Verification uses `--target-dir /private/tmp/cuberel-build.hmThUI`. The executable
prints the selected runtime, client name and cube size. Metal checks run with
actual GPU access outside the restricted command sandbox.

All listed checks passed. The all-feature suite passed 29 tests, none ignored;
the no-runtime suite passed 22 frontend/layout tests. Clippy and rustdoc passed
with warnings denied; formatting and diff whitespace checks passed. The selected
client names were `cpu` and `metal`; default cube sizes were 8 and 64 respectively.
Both standalone examples printed `[13.0, 25.0, 37.0]` for the combined expression
and `[10.0, 19.0, 28.0]` after rebinding scale to -1 with bias 1.
