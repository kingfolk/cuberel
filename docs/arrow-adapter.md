# Arrow host boundary adapters

Enable `arrow` to use Arrow input/output with the existing runtime expression
compiler. `arrow-array`, `arrow-buffer` and `arrow-schema` are pinned to 60.0.0.
This optional feature is independent of CPU/Metal runtime features and is off by
default. The CubeCL revision remains unchanged. [Checked I32](checked-i32.md) extends
this adapter under binding ABI v4 and codegen v8.

```toml
cuberel = { path = "...", features = ["arrow"] }
arrow-array = "=60.0.0"
arrow-schema = "=60.0.0"
```

I32 columns, nullable scalar bindings and Int32Array output are now supported;
see [checked I32 semantics](checked-i32.md). Numeric field types must match exactly.
An overflowing I32 expression returns an error rather than an Arrow array.

## Workflow

```rust
let schema = schema_from_arrow(&input.schema(), &[
    Field::new("scale", DataType::F32).with_nullable(true),
    Field::new("bias", DataType::F32),
])?;
let program = prepare(&(col("a") * param("scale") + param("bias")), &schema)?;
let result: ArrayRef = executor.execute_arrow(&program, &input, &parameters)?;
```

For an Arrow column `a=[1, NULL, 3]`, `scale=2`, `bias=1`, the output is a
Float32Array `[3, NULL, 7]`. A Bool root returns a BooleanArray preserving unknown:
`a > bias` returns `[false, NULL, true]`. `execute_arrow_mask` instead returns a
BooleanArray `[false, false, true]` with no null bitmap, applying WHERE semantics.
It preserves row positions and does not compact or return a filtered batch.

The input is a RecordBatch; the output is one array because KernelProgram still
describes one expression. To form an output RecordBatch, use Arrow's ordinary
`RecordBatch::try_new` with a field matching `program.output_type()` and
`program.output_nullable()`. The adapter adds no relational query planner.

See [`examples/arrow_expr.rs`](../examples/arrow_expr.rs) for complete setup and a
sliced RecordBatch. Parameters keep the existing named `Parameters` API, including
`Option<f32>`; this feature does not add Arrow scalar parameter bindings.

## Names, types and nullability

`schema_from_arrow` retains field order, exact names and nullability, while accepting
only Float32 and Int32 fields. It validates duplicates and empty names through Schema. Other
Arrow types return `ArrowAdapterError::UnsupportedType`; metadata and extension
semantics are not translated into query operations.

When a RecordBatch contains unrelated columns (for example UTF8), construct an
explicit Cuberel schema for the supported fragment. `ArrowBatch::try_new` and the
Executor convenience methods select only referenced columns by exact name, including
references in unreachable arena nodes. Unreferenced types do not need conversion.
Duplicate/empty RecordBatch field names are rejected even when unreferenced, so a
name can never silently choose the first of several columns. Missing names and
referenced unsupported or mismatched numeric columns are errors. Input array order need not match the
program schema order.

Actual NULLs in a non-nullable Cuberel field fail before device work. A nullable
Arrow field with no actual NULLs may bind to a non-nullable Cuberel field, as with
the existing host wrappers. Nullable Cuberel fields accept all-valid arrays, and
their declared ABI still has validity slots. Static schema/nullability determines
the kernel key; actual null flags and buffer addresses never specialize it.

## Borrowed input and offsets

`ArrowBatch` retains borrowed F32 value slices and owns dense U32 validity for
referenced arrays with NULLs. `as_batch()` creates a borrowed ordinary Batch view;
the adapter and source value buffers must outlive that view and execution. No
self-referential struct or unsafe Arrow memory import is used.

```rust
let adapter = ArrowBatch::from_columns(&program, array.len(), &[("a", &array)])?;
let batch = adapter.as_batch()?;
let result = executor.execute_nullable(&program, &batch, &parameters)?.into_arrow();
```

`from_columns` accepts named Float32Array/Int32Array references as `&dyn Array` and validates all
supplied lengths/names, including extras. An empty column list plus explicit row
count supports parameter-only programs. Existing U32 row/byte bounds are checked
before bitmap expansion, without allocating large buffers to test the boundary.

For each logical array view:

- `values()` already contains the ScalarBuffer's sliced value range. No second
  array offset is added.
- `is_valid(i)` separately honors the NullBuffer's bit offset. Value and validity
  buffers can have different origins, and bit offsets need not be byte aligned.
- Dense flags cover exactly the logical row count. Tail bits and payload padding
  outside the view never become input rows.

All-valid arrays need no retained dense validity; the existing executor synthesizes
ones if the program's nullable ABI requires them. Arrow null payloads remain
initialized valid F32 bit patterns, with no logical meaning. The emitter sanitizes
invalid payloads as before; valid NaN/infinity/signed zero remain values. Arrow's
[array documentation](https://docs.rs/arrow-array/60.0.0/arrow_array/) describes
the typed values and zero-copy slicing APIs used here.

## Output ownership and costs

`NullableF32::into_arrow` consumes the result and moves its F32 Vec into an Arrow
ScalarBuffer without copying the payload. It packs dense validity into a bitmap,
omitting the bitmap when every row is valid. `NullableBool::into_arrow` packs both
dense U32 value and validity planes into independent bitmaps. Returned arrays own
their buffers and do not borrow the input or adapter. No `Vec<Option<T>>` output
intermediate is constructed by the adapter.

Memory/transfer costs remain explicit:

- Arrow input F32 values are borrowed at the adapter boundary. Each referenced
  nullable input with actual NULLs allocates `4*N` host bytes for dense validity;
  adapter reuse can avoid repeating that expansion. Building a Batch view also
  validates flags. Executor convenience calls rebuild the adapter each invocation.
- The current executor still encodes/uploads required values and validity on every
  call, then reads output planes. This is host adaptation, not device zero-copy or
  persistent device residency.
- Numeric output packing needs up to `ceil(N/8)` bytes of validity bitmap in addition
  to the moved `4*N` payload. Bool output needs up to two `ceil(N/8)` bitmaps while
  consuming the dense readback vectors. Buffer allocators may reserve extra capacity.
- Nullable WHERE execution currently checks both output readbacks before packing
  its all-valid mask; it does not skip the validity transfer. Packing is O(N).

No expression lowering, kernel-cache policy or backend error handling is replaced.
Core execution failures remain available through `ArrowAdapterError::Execution`;
no CPU fallback or Arrow arithmetic interpreter is introduced.

## Verification

Seven host tests cover schema mapping, all bitmap bit offsets, nested/empty slices,
independently offset value/validity buffers, borrowed-pointer identity, named binding,
duplicate/empty/missing names, heterogeneous batches, null/type contracts, and row
bounds before allocation. Two conformance tests run real CPU and native Metal:
projection, nullable Bool and WHERE outputs at empty/tail/multiple workgroups,
all NULL, signed zero/NaN/infinity, parameter-only batches, error preflight, numeric
output pointer preservation, and shared core/Arrow kernel reuse after rebinding.

```sh
cargo test --locked --all-features
cargo test --locked --no-default-features --features arrow
cargo test --locked --no-default-features
cargo run --locked --features arrow --example arrow_expr -- cpu
cargo run --locked --no-default-features --features arrow,metal --example arrow_expr -- metal
```

Arrow IPC/FFI/device interfaces, Bool/other input types, multi-expression projection,
DataFusion planning and compaction remain separate later features.

Verified on 2026-11-04 on macOS/arm64 with the existing Rust/Cargo 1.99.0 toolchain:
61 all-feature integration tests passed (zero ignored), 41 passed with only the
Arrow feature and no runtime, and 34 passed with all optional features disabled.
All-feature/all-target Clippy and library rustdoc passed with warnings denied;
formatting and diff checks passed. The example ran on actual CPU and native Metal,
producing `[3, NULL, 7]`, `[false, NULL, true]` and `[false, false, true]` for its
numeric, Bool and WHERE outputs. Commands used `--locked --offline` and the existing
temporary target directory. The lockfile adds Arrow's dependency graph while
retaining every previously locked package version.
