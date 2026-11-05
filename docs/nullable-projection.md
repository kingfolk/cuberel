# Nullable F32 projections

F32 column and parameter declarations can now carry nullability. Prepared add/mul
programs propagate validity and execute through the existing runtime IR emitter
on CPU or native Metal. The host receives an owned `NullableF32` with a values
slice and one dense U32 validity flag per row.

```rust
let schema = Schema::from_fields(
    &[Field::new("a", DataType::F32).with_nullable(true)],
    &[Field::new("scale", DataType::F32).with_nullable(true),
      Field::new("bias", DataType::F32)],
)?;
let program = prepare(&(col("a") * param("scale") + param("bias")), &schema)?;
let batch = Batch::new(3).with_column("a", ColumnData::NullableF32 {
    values: &[1.0, f32::NAN, 3.0], validity: &[1, 0, 1],
})?;
let parameters = Parameters::new().with("scale", Some(2.0_f32))?.with("bias", 1.0_f32)?;
let output = executor.execute_nullable(&program, &batch, &parameters)?;
assert_eq!(output.iter().collect::<Vec<_>>(), [Some(3.0), None, Some(7.0)]);
assert_eq!(output.values(), [3.0, 0.0, 7.0]);
assert_eq!(output.validity(), [1, 0, 1]);
```

See [`examples/nullable_projection.rs`](../examples/nullable_projection.rs) for
imports and device setup. Rebinding `scale` to `None::<f32>` yields three NULLs.
Rebinding back to a value restores the corresponding non-NULL rows without a new
kernel identity. Data and validity changes are runtime arguments.

## Schema, input and result contracts

`Schema::new` still declares non-nullable fields. `Field::new(...).with_nullable(true)`
and `Schema::from_fields` declare nullable columns or parameters. Preparation records
nullability on each `TypedExpr`; arithmetic output is nullable if either input is
nullable. `KernelProgram::output_nullable()` exposes that static property.

`ColumnData::NullableF32 { values, validity }` borrows equal-length initialized
slices. Each validity flag must be exactly 0 or 1, checked during batch construction.
All supplied columns, including extra unused ones, obey these shape/flag checks.
`Parameters::with` accepts `Option<f32>` as a typed nullable F32 scalar.
Schema declaration and invocation values are distinct: nullable fields accept
ordinary F32 slices/scalars as all valid; non-nullable fields accept nullable
wrappers only if they actually contain no NULL. Violations fail before upload,
including NULL scalar parameters on an empty batch. Required inputs are validated
even when the row count is zero, after which no buffers are allocated or launched.

`execute_nullable` accepts F32 programs of either static nullability. For a
non-nullable root it synthesizes all-valid host flags and uses the existing single
device output. `execute` rejects a nullable root with `NullableOutput`, preventing
validity loss. [Nullable predicate APIs](three-valued-predicates.md) now preserve
Bool validity with `execute_predicate`, and apply WHERE semantics with `execute_mask`. Result fields
are private and exposed as immutable slices or an `ExactSizeIterator<Option<f32>>`.

NULL is determined only by validity. A valid NaN remains a valid NaN. An invalid
payload can be any initialized F32 input, including NaN or infinity; it is never
an uninitialized hole in the slice. All nullable SSA values are sanitized to +0
when invalid, at leaves before their consumers and after arithmetic:

```text
out.valid = a.valid AND b.valid
out.value = select(out.valid, arithmetic(a.value, b.value), +0)
```

`NULL * 0` remains NULL. Every invalid output payload is canonical positive zero,
which makes the public values slice deterministic without reading undefined
memory. This does not authorize reading uninitialized payloads through the raw
device API. Current arithmetic is pure F32 add/mul and can evaluate eagerly; no
evaluation contract for future fallible operations is established here. Existing
arithmetic grouping and separate-rounding policy remain in effect.

## Binding ABI v3 and IR

Buffers remain scalar four-byte elements with U32 addressing. In schema-ID order,
each referenced column has its F32 value buffer, followed by a U32 validity buffer
if declared nullable. Value output follows all inputs; nullable output adds a
separate write-only U32 validity buffer. Both have the invocation's row count.
`BufferSource::{ColumnValidity, OutputValidity}` identifies planes explicitly;
`KernelDeclaration::output()` finds the value output regardless of its position.

Row count retains U32 ordinal 0. Referenced parameter payloads have their existing
F32 ordinals; each nullable parameter adds a U32 validity scalar using ordinals
1, 2, and so on. `ScalarSource::ParameterValidity` distinguishes it even when
names overlap column or internal names. NULL scalar payloads bind as +0. The same
layout drives declarations, host preflight, descriptor validation and launch.

The emitter reads validity, casts to Bool SSA, combines it alongside values, and
stores output validity as U32 0/1. Every load and both stores stay within the row
guard. A complete-module verifier runs before lowering. Both output allocations
are fresh, and the raw adapter rejects output/input or output/output aliasing,
including disjoint views of a common allocation. Raw validity imports require
initialized 0/1 values; descriptor checks cannot inspect opaque device contents.

The executor reads value and validity outputs through checked CubeCL completion
calls and checks each byte length. It validates returned validity before exposing
any result; failure from either read returns an error. Readback is synchronous
and currently uses two read calls for nullable outputs. There is no device-resident
result or async interface, and this feature makes no performance improvement claim.

ABI version 3 describes value/validity planes; current codegen version 6 adds
nullable predicate semantics without changing their physical representation. Static
nullability participates in typed nodes and binding identity; dynamic flags,
NULL/Some scalar state, row counts and addresses do not. Duplicate pure nullable
subexpressions retain the existing normalized DAG identity policy.

## Boundaries and validation

Nullable comparisons, IS NULL, nullable Bool/three-valued logic and WHERE keep
semantics are now implemented by [nullable predicates](three-valued-predicates.md). The optional [Arrow adapter](arrow-adapter.md) now handles F32 bitmap/offset
conversion; integer arithmetic remains later work. Validity is currently dense U32, not an Arrow
bitmap, and no host expression interpreter is introduced.

Seven tests cover schema/nullability inference and input validation, ABI planes,
per-type scalar ordinals, strict/non-null binding rules, and actual CPU/Metal
execution. Runtime fixtures cover empty/tail/multi-workgroup inputs, all NULL,
mixed validity, two nullable parameters and broadcasts, NaN payloads, valid signed
zero/NaN/infinity, strict rounding, DAG roots and sharing, output aliasing, and
reuse after null/scalar/row-count/address changes. Existing regressions also run.

```sh
cargo run --locked --example nullable_projection -- cpu
cargo run --locked --no-default-features --features metal --example nullable_projection -- metal
cargo test --locked --all-features
cargo test --locked --no-default-features
```

Validation passed against the unchanged pinned CubeCL revision, using the existing
macOS/arm64 Rust/Cargo 1.99.0 toolchain and temporary target directory:
47 all-feature integration tests (zero ignored), 31 no-runtime tests, Clippy for
all features/targets and library rustdoc with warnings denied, formatting and diff
checks. The standalone example ran on actual CPU and native Metal; rebinding
`Some(2)` to NULL and back to `Some(3)` added zero IR definitions after the first
invocation. No Cargo dependency or lockfile changes were needed.
