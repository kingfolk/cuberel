# Nullable predicates and SQL three-valued logic

Nullable F32 arithmetic can now feed all six comparisons. Bool expressions carry
`(value, valid)` through AND/OR/NOT. `is_null()` produces a known Bool for either
F32 or Bool expressions. Inputs still use F32 payloads and dense U32 validity;
Bool host columns/scalars remain unsupported.

For `a = [1, NULL, 3]` and `limit = 2`:

| Expression | Predicate projection | WHERE mask |
| --- | --- | --- |
| `a > limit` | `[false, NULL, true]` | `[0, 0, 1]` |
| `a IS NULL` | `[false, true, false]` | `[0, 1, 0]` |
| `NOT(a > limit)` | `[true, NULL, false]` | `[1, 0, 0]` |
| `a IS NULL OR a > limit` | `[false, true, true]` | `[0, 1, 1]` |

```rust
let program = prepare(&col("a").gt(param("limit")), &schema)?;
let output = executor.execute_predicate(&program, &batch, &parameters)?;
assert_eq!(output.iter().collect::<Vec<_>>(), [Some(false), None, Some(true)]);
let keep = executor.execute_mask(&program, &batch, &parameters)?;
assert_eq!(keep, [0, 0, 1]);
```

[`examples/nullable_predicate.rs`](../examples/nullable_predicate.rs) contains
complete schema, input and device setup. Both calls use the same kernel identity;
requesting a WHERE mask does not compile or launch a separate mask-conversion
kernel. The expression still executes on each invocation.

## Logical rules

A comparison is NULL if either operand is NULL, even for `NULL = NULL` or
`NULL != NULL`. With valid operands, existing IEEE F32 comparison behavior is
preserved: NaN is a value, not NULL. A valid NaN compared for equality is false,
inequality is true, and `NaN IS NULL` is false.

| a | b | a AND b | a OR b |
| --- | --- | --- | --- |
| true | true | true | true |
| true | false | false | true |
| true | NULL | NULL | true |
| false | true | false | true |
| false | false | false | false |
| false | NULL | false | NULL |
| NULL | true | NULL | true |
| NULL | false | false | NULL |
| NULL | NULL | NULL | NULL |

NOT swaps true/false and preserves NULL. IS NULL is true exactly when its operand
is invalid, and its result is never NULL. It works on arithmetic, comparisons,
logical combinations and another IS NULL. Rust `!` builds logical NOT; Rust
`==`/`!=` still compare host Expr structures, so use `eq_expr`/`ne_expr` for queries.

Preparation conservatively marks comparison/AND/OR results nullable if either
operand is nullable, and NOT inherits its operand's nullability. IS NULL is
non-nullable. This static property describes possible nullability without proving
relations between expressions: `a IS NULL OR a > limit` can have an all-valid
runtime result while still being statically nullable. All arena nodes continue
to participate in input validation, including unreachable inputs and operands of
an IS NULL that could be folded to a constant.

## IR construction

For nullable comparisons, validity is `a.valid AND b.valid`. For boolean logic,
let `aT = a.valid AND a.value`, `aF = a.valid AND NOT a.value`, and likewise for b:

```text
AND: true = aT AND bT; false = aF OR bF
OR:  true = aT OR bT;  false = aF AND bF
valid = true OR false
value = true
NOT: valid = a.valid; value = NOT a.value
IS NULL: valid = true; value = NOT a.valid
```

Unknown Bool payloads are canonical false after every node. The emitter uses that
invariant to compute true bits directly from Bool payloads, and combines known
false evidence to derive validity for AND/OR. NOT is gated by its inherited
validity before consumers use it. This avoids incorrectly applying ordinary
`a.valid AND b.valid` to every logical operator. The implementation emits these
operations into CubeCL IR once during definition; there is no row-wise AST
interpreter in host or device code.

ABI v4 already describes U32 values and separate U32 validity slots, so no new
physical ABI version is required. Nullable Bool roots use both planes. F32 values
still use F32 output plus optional validity; IS NULL uses a single known Bool
output. All buffer accesses and both stores stay inside the logical row guard,
and full-module verification remains in place. Codegen version 6 separates these
artifacts from the previous predicate semantics. IsNull operands are remapped
during identity normalization; runtime null flags, values and row counts remain
outside the key.

## Output and completion contracts

`execute_predicate()` accepts any Bool root and returns an owned `NullableBool`.
Its immutable `values()` and `validity()` slices contain 0/1 U32 words, and `iter()`
returns `Option<bool>`. Non-nullable roots receive all-valid host flags.

`execute_mask()` explicitly applies WHERE semantics: true keeps the row; false
and NULL reject it. The returned mask is the canonical Bool payload produced on
device (`keep = valid AND value`), with no host truth-table evaluation or row
compaction. Both output readbacks/lengths are checked, and returned binary flags
are validated. Nullable masks currently retain a validity output and its checked
readback; avoiding that transfer is a separate optimization. No result is exposed
if either read fails. The existing synchronous upload/readback model is unchanged.

`execute()` and `execute_nullable()` remain F32-only. Using a mismatched API fails
before upload. Empty invocations still validate input declarations and parameters,
then return empty output without an invalid zero-grid launch. Operators are pure
and evaluate eagerly; this does not add a short-circuit/fallible-function contract.
The optional [Arrow adapter](arrow-adapter.md) now handles host F32 arrays and
bitmap/offset conversion. Host Bool bindings and compacted selection remain later work.

## Verification

Five new tests cover inference, unary raw graph/root/depth bounds, and nullable
Bool output layout plus actual CPU and native Metal conformance. Fixtures assert
the full 3x3 AND/OR truth tables, NOT, nested logic, F32/Bool IS NULL, six nullable
comparisons, NULL versus NaN/infinity/signed zero, scalar broadcasts, tails/empty,
invalid bindings, output API checks, shared DAG identities and rebinding reuse.
The former unsupported-nullable-comparison fixture now checks supported inference.

```sh
cargo run --locked --example nullable_predicate -- cpu
cargo run --locked --no-default-features --features metal --example nullable_predicate -- metal
cargo test --locked --all-features
cargo test --locked --no-default-features
```

Verified on 2026-11-04 using macOS/arm64, the existing Rust/Cargo 1.99.0 toolchain
and unchanged pinned CubeCL revision: 52 all-feature integration tests (zero
ignored), 34 no-runtime tests, all-feature/all-target Clippy and library rustdoc
with warnings denied, formatting and diff checks. The standalone example ran on
real CPU and native Metal and matched all four projected results/WHERE masks in
the table above. Validation used `--locked --offline` and the existing temporary
target directory; no new dependency was added.

Current numeric extension: [checked I32](checked-i32.md) adds typed payloads and
a per-row overflow error plane; I32 and F32 operands do not implicitly mix.
