# F32 predicates and dense masks

This feature starts Milestone 02 (design M1). Prepared F32 computations can now
feed comparisons, boolean composition and a real CPU/native Metal predicate
kernel. Every input row has one output flag; matching rows keep their original
positions. Output length always equals the batch row count.

```rust
let expression = (col("a") * param("scale"))
    .gt(param("limit"))
    .and(col("a").lt(param("upper")));
let program = prepare(&expression, &schema)?;
let mask: Vec<u32> = executor.execute_mask(&program, &batch, &parameters)?;
```

With `a = [1, 3, 5]`, `scale = 2`, `limit = 4`, `upper = 5`, this returns
`[0, 1, 0]`. Rebinding to `scale = 1`, `limit = 0`, `upper = 6` returns
`[1, 1, 1]` through the same kernel identity. See
[`examples/predicate_mask.rs`](../examples/predicate_mask.rs) for complete setup.

## Expression and type contract

| API | Operand types | Output |
| --- | --- | --- |
| `+`, `*` | F32, F32 | F32 |
| `eq_expr`, `ne_expr`, `lt`, `le`, `gt`, `ge` | F32, F32 | Bool |
| `and`, `or` | Bool, Bool | Bool |
| `!expression` | Bool | Bool |

`eq_expr`/`ne_expr` construct query comparisons. Rust's `==`/`!=` continue to
compare the host `Expr` structures. `Comparison` and the matching `ExprNode` /
`ScalarOp` variants are available to programmatic frontends. All arena nodes,
including unreachable ones, are checked for valid references and operand types.
Type errors report the consuming node, operand ID, expected type and actual type.
Unary nodes participate in preparation depth/node bounds and malformed-root checks.

Comparison operands must remain non-nullable F32; nullable comparisons are
rejected. Bool is currently an expression result type; a referenced Bool input declaration is explicitly
unsupported. There are no implicit numeric/bool conversions or bool comparisons.
Boolean composition evaluates its operands eagerly. All current operations are
pure and safe to evaluate on every in-range row; this does not establish an
evaluation policy for future fallible functions.

Comparisons follow IEEE floating-point predicates: NaN makes equality and ordered
comparisons false, and inequality true; `+0` equals `-0`; infinities compare in
their usual order. NaN is a value, with no NULL semantics. This comparison
contract does not broaden the existing arithmetic contract to guarantee NaN
payloads, denormal behavior or cross-device bitwise arithmetic identity.

## IR and ABI

Comparison nodes lower to CubeCL's F32 comparison operations; AND/OR/NOT lower
to boolean operations. Bool values live in SSA. Only the root is converted to
physical U32 before storing, producing exactly 0 or 1. This avoids a shared
buffer ABI depending on the size of a backend's native boolean representation.
The root type controls the output declaration, imported device-buffer type,
readback interpretation and validation. All loads, operations and stores remain
inside the logical row-count guard. Complete IR is verified before build.

Binding ABI version 3 (extended by [nullable projection](nullable-projection.md))
records each buffer's physical type. Predicate operands use F32 value inputs;
nullable F32 projections add U32 validity planes. Arithmetic output is F32 and
predicate output is U32. All currently supported
slots are four bytes per row, so existing extent/alignment/U32 byte-end checks
remain applicable. Buffer validation rejects an F32 allocation descriptor for
a predicate's U32 output, even though their byte sizes match.

`Executor::execute` retains `Vec<f32>` output and rejects Bool programs.
`Executor::execute_mask` returns `Vec<u32>` and rejects F32 programs. Both reject
wrong result types before allocation/launch; empty batches still validate all
required inputs. Readback uses the same checked completion path, then validates
mask values are 0/1. It does not reinterpret arbitrary bytes as Rust bools.

Codegen version is now 5 after nullable projection support. Identity includes
comparison kind, boolean operations,
typed nodes, remapped operands/root and the updated physical layout. Duplicate
pure predicate nodes normalize with the existing structural interning. Parameter
values, row counts and buffer addresses remain outside the key. CPU/Metal reuse
tests verify that a nested predicate defines once for a unique cold key, then
reuses the artifact after parameters, input addresses and row counts change.

## Scope and verification

This produces a non-nullable dense mask. F32 projection validity is implemented
separately; nullable predicates, SQL three-valued logic,
`IS NULL`, Arrow arrays/offsets, integer expressions, row compaction and relation
operators remain later features. No host expression interpreter is introduced.

Six new tests cover mixed-type preparation, invalid/unreachable operands,
unary/binary reference and depth limits, output descriptor checks, and actual
CPU/native Metal execution. Runtime fixtures include six comparison operators,
NaN on either operand and scalar parameters, infinities, signed zero, shared
operand comparisons, every two-valued AND/OR truth-table row, NOT, nested
arithmetic, parameter broadcasts, empty/partial/multiple workgroups, all/none
matches, invalid bindings, wrong execution APIs and cache identity/reuse.

```sh
cargo run --locked --example predicate_mask -- cpu
cargo run --locked --no-default-features --features metal --example predicate_mask -- metal
cargo test --locked --all-features
cargo test --locked --no-default-features
```

Verified against the unchanged pinned CubeCL revision on macOS/arm64, with the
existing Rust/Cargo 1.99.0 toolchain and shared temporary target directory:
40 all-feature integration tests and 26 no-runtime tests passed, with zero
ignored tests. Clippy (all features/targets) and library rustdoc passed with
warnings denied; formatting and diff checks passed. The standalone example ran
on CPU and native Metal, both producing `[0, 1, 0]` then `[1, 1, 1]`, with zero
additional definitions on the second invocation. No new dependency was added.
