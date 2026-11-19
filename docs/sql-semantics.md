# SQL capability profile: df55-i32-v1

Capabilities are checked on bound DataFusion **55.2.0 physical expressions**,
including explicit coercions, not SQL text. Rejected expressions retain the
original DataFusion plan and semantics. No session arithmetic option is changed.

| Expression | Initial mapping |
|---|---|
| Int32/Float32 column | Physical position → unique `cN`; nullable retained |
| Typed Int32/Float32 literal/NULL | Immutable scalar binding, not a kernel constant |
| Int32 comparisons | Six signed comparisons; NULL propagates |
| AND/OR/NOT, IS NULL/IS NOT NULL | 3VL for pure supported subtrees |
| Checked Int32 add/multiply | Only BinaryExpr explicitly configured to fail on overflow |
| Default wrapping arithmetic | CPU: `semantic_mismatch` |
| Checked operations under AND/OR | CPU: DataFusion short-circuits and the DSL is eager |
| Float arithmetic/comparisons | CPU: rounding/total-order/NaN compatibility is unverified |
| Cast | Metadata-free identity only; otherwise `unsupported_cast` |
| Boolean inputs/literals, Int64/Float64, other functions | CPU: unsupported type/operator |

The checked flag is part of DataFusion BinaryExpr equality. The translator
compares the expression against the same children/operator with checked mode
enabled; it never infers this flag from display strings. Nullable arithmetic and
checked overflow are validated against evaluated DataFusion expressions on the
actual CPU/Metal backends. Keeping wrapping operations on CPU is required even
if a particular fixture's small values happen not to overflow.

Physical column positions are authoritative. Duplicate SQL names do not collapse
two columns; the adapter uses its own names and the SQL execution boundary
restores the original output schema. Extension input semantics are rejected.
Translation has depth/node bounds and ordinary Cuberel preparation limits.

`translate_expression` returns an owned program, input positions and scalar
bindings. Rebinding the same typed literal shape changes values without changing
the program. SQL placeholder substitution/coercion/constant folding still occurs
in DataFusion: different optimized SQL shapes do not promise the same kernel.
Execution must never silently rerun a failed kernel on the host.

The conformance fixtures compare DataFusion CPU with direct Cuberel execution,
including NULL, all-valid values, I32 extrema, NaN, infinity, signed zero,
subnormal payloads and checked failure. Full SQL stream/fragment tests additionally
cover placement, batches/partitions and lifecycle. SQL global aggregates remained on CPU in F1–F3; the
[bounded integer aggregate contract](sql-global-aggregates.md) now defines
supported COUNT, MIN/MAX and widening SUM regions, row proofs and host-state merge.
