# Checked I32 expressions and Milestone 02 scope

I32 columns/parameters now support copy, add/mul, six signed comparisons, nullable
propagation, IS NULL, three-valued logic and WHERE masks. F32 and I32 operands
must match exactly; there is no implicit numeric cast or integer division API.
Bool input declarations remain unsupported. This completes the scoped M1 typed
numeric/nullable predicate and Arrow boundary work; performance superiority,
compaction and device residency are not part of this stage's acceptance.

## API example

```rust
let schema = Schema::from_fields(
    &[Field::new("a", DataType::I32).with_nullable(true)],
    &[Field::new("scale", DataType::I32), Field::new("bias", DataType::I32)],
)?;
let program = prepare(&(col("a") * param("scale") + param("bias")), &schema)?;
let batch = Batch::new(3).with_column("a", ColumnData::NullableI32 {
    values: &[1, i32::MAX, 3], validity: &[1, 0, 1],
})?;
let parameters = Parameters::new().with("scale", 2_i32)?.with("bias", 1_i32)?;
let output = executor.execute_nullable_i32(&program, &batch, &parameters)?;
assert_eq!(output.iter().collect::<Vec<_>>(), [Some(3), None, Some(7)]);
```

`execute_i32` returns Vec<i32> for a non-nullable root; `execute_nullable_i32`
returns NullableI32 for either static nullability. NullableI32 exposes immutable
values/validity and an Option<i32> iterator; invalid payloads are canonical zero.
Parameters accept i32 or Option<i32>. F32 methods keep their existing signatures
and reject an I32 root, and vice versa. See the complete
[`checked_i32` example](../examples/checked_i32.rs).

## Overflow and evaluation contract

Add/mul use checked signed I32 semantics at every reachable arithmetic node. If
both operands are valid and the mathematical result is outside MIN..=MAX, the
entire invocation returns `ExecutionError::IntegerOverflow { row }`, identifying
the first failing logical row. No partial result is exposed. With a NULL operand,
that operation is NULL and its potential overflow is suppressed.

Checks preserve expression grouping: `(MAX + 1) + (-1)` fails despite the final
mathematical value fitting. Reachable operations are eager, including operands of
AND/OR and IS NULL. An already overflowing valid child still fails if its parent
also has a NULL operand, or if a logical parent could otherwise determine a result
without it. No short-circuit/fallible CASE contract is promised. All input names
and types in the arena still validate, but overflow in a node unreachable from the
output root is ignored. Raw graph roots need not be the last arena node.

An overflow is a successful kernel completion with a query error, distinct from
backend compilation/launch failure. Both categories return Result errors. A later
valid invocation on the same executor works, with the same cached query artifact.

## Portable arithmetic IR

Low-word addition/multiplication use U32 arithmetic modulo 2^32 and reinterpret
the resulting bit pattern as I32. The wrapped word is internal and is never a
successful public result on overflow. This avoids invoking C++/Metal signed
overflow or relying on optimization flags to define it.

Addition overflow is detected from operand and result signs. For multiplication,
compute unsigned magnitudes (including magnitude 2^31 for MIN), then choose a
limit of MAX for a non-negative product or 2^31 for a negative product. Overflow
is `abs(a) > limit / abs(b)` when b is nonzero. A selected denominator of at least
one makes the emitted U32 division safe even for zero or NULL inputs. There is no
I64 requirement, signed division, MIN/-1 instruction, device atomics, runtime
Rust compilation or host expression interpreter.

The pinned native Metal backend cannot emit cmp.bool_equal. Sign equality uses
equivalent AND/OR/NOT operations, verified on CPU and Metal; the dependency is not
modified. F32's existing strict-add legalization remains intact.

## ABI, diagnostics and cache

ABI v4 records I32 payload buffers and scalars. Row count and validity keep their
U32 ordinals; F32 and I32 parameters have separate per-type ordinal sequences.
All planes remain four bytes per logical row with the existing U32 extent checks.
HostBindings::columns() now returns typed ColumnData payload views; use as_f32()
or as_i32() instead of assuming every buffer is F32. Validity remains separately
aligned with these views.

Programs with reachable I32 add/mul append a write-only U32 OutputError buffer.
Each in-range row stores 0 or 1 after OR-ing valid overflow evidence for reachable
nodes. Value, optional validity and error buffers are separate allocations; all
outputs participate in alias/range checks. Queries without reachable integer
arithmetic omit this plane. Every output store stays within the row guard.

The executor checks every readback and length before interpreting overflow flags.
Any overflow rejects the invocation before a result/Arrow array is returned.
The additional error plane costs 4*N device bytes plus checked readback and a host
scan for these programs. This simple per-row diagnostic is an initial correctness
choice, not a throughput optimization.

Codegen v9 and ABI v4 invalidate incompatible cached artifacts. Typed node types,
nullability and the presence of the error plane participate in identity; runtime
values/null flags do not. The ordinary full-batch error output contains row flags, avoiding raw
ExprId diagnostics that would be inconsistent when structurally duplicated and
shared DAGs normalize to the same cache identity.

## Arrow integration

The optional Arrow 59.2 adapter accepts Float32 and Int32. Direct named-array
binding uses `&dyn Array`, permitting mixed numeric columns. RecordBatch binding
matches each referenced field against the program's exact type. Unsupported Arrow
types and a supported numeric type mismatch have distinct structured errors.

ArrowBatch borrows typed values and expands validity with independent offsets, as
before. `execute_arrow` returns Int32Array for an I32 root; NullableI32::into_arrow
moves the payload Vec and packs validity. Overflow propagates through
ArrowAdapterError::Execution, with no array exposed. Arrow copies, bitmap packing,
GPU upload and readback costs remain as described by the Arrow boundary document,
plus the error plane when needed. No new dependency is introduced.

## Verification

Dedicated tests cover exact typing, mixed F32/I32/U32 ABI ordinals, nullable bindings,
root reachability and error-plane validation. Native CPU/Metal fixtures compare
boundary combinations against Rust checked_add/checked_mul, including MIN/MAX,
signs, zero, 46340/46341 and 65536. They also cover signed comparisons, grouped
overflow, NULL suppression, nullable projection and mixed-type predicate tails,
post-error recovery, cache reuse and error-output alias rejection. Arrow fixtures
include sliced Int32 arrays with independently offset validity and moved output
pointer identity. Existing F32/three-valued/Arrow regressions remain in the suite.

```sh
cargo test --locked --all-features
cargo test --locked --no-default-features --features arrow
cargo test --locked --no-default-features
cargo run --locked --example checked_i32 -- cpu
cargo run --locked --no-default-features --features metal --example checked_i32 -- metal
```

Verified locally on 2026-11-05 with macOS/arm64 and the existing Rust/Cargo 1.99.0
and pinned CubeCL revision: 68 all-feature integration tests (zero ignored),
46 Arrow-only/no-runtime tests, and 38 tests without optional features passed.
All-target/all-feature Clippy and library rustdoc passed with warnings denied;
formatting/diff checks passed. CPU and native Metal examples produced [3,NULL,7],
WHERE [1,0,1], reported a row-0 overflow and then executed successfully again.
The device adapter implementation moved to src/device_abi.rs while preserving
its public cubecl_abi module name. No dependency changes were made.

Selected execution encodes original row+1 in compact error prefixes and checks
only active rows. See [selected projection](selected-projection.md); the existing
eager full-batch arithmetic semantics remain unchanged.
