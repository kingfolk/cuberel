# Typed expressions and preparation

Milestone 01 work item 2 introduced runtime expression construction and preparation.
Work item 4 adds [device execution](runtime-emitter.md) and removes the temporary
host evaluator. This document describes the retained frontend and input APIs.

## Public workflow

1. `Schema::new(columns, parameters)` declares non-nullable names/types;
   `Schema::from_fields` accepts explicit nullable fields.
2. `col("a") * param("scale") + col("b")` builds an owned expression graph.
3. `prepare(&expression, &schema)` checks the graph and binds names to typed IDs.
4. `Batch::new(row_count).with_column(...)` supplies borrowed host columns.
5. `Parameters::new().with(...)` supplies named scalars.
6. `executor.execute(&program, &batch, &parameters)` runs a non-nullable F32 kernel;
   `execute_nullable` returns F32 values and validity;
   `execute_predicate` preserves nullable Bool; `execute_mask` produces WHERE flags.

See `examples/runtime_expr.rs` for selected CPU/Metal execution and parameter
rebinding. Frontend preparation and host binding work with no runtime feature.
The prepared program owns schema and expression data through immutable Arcs. It
can outlive its source expression/schema, be cheaply cloned and shared across
threads. It stores no CubeCL Scope, pliron Value, device handle or client.

## Types and names

ColumnId and ParameterId are positions within a program's schema, not globally
unique handles. ExprId is local to an expression/program arena. IDs from different
programs do not identify the same binding merely because their indexes match.

Nullable F32 arithmetic and nullable Bool predicates are implemented.
See [nullable projections](nullable-projection.md) for schema/input/result
contracts. Bool is an expression result; Bool input declarations remain unsupported. See
[predicates](predicates.md) for comparison and boolean APIs.
DataType::I32, ColumnData::I32
and ScalarValue::I32 describe unsupported/mismatched input and produce structured
errors; they do not enable integer arithmetic. Used I32 declarations fail at
preparation, and I32 data for an F32 declaration fails before device upload.
There are no implicit casts; give float parameters an explicit `f32` suffix.

Columns and parameters may have the same name. Empty or duplicate names within
a namespace are rejected; names are exact and case-sensitive. Name resolution
happens during preparation and binding, not once per device row. Input insertion
order is irrelevant. Unused schema fields and extra supplied inputs are permitted.
Every supplied batch column must still match the explicit row count.

## Graph validation and bounds

Expressions use a flat arena instead of recursively owned boxes. Preparation and
drop do not recurse through the graph. Binary DSL operations merge arenas and
rebase the right operand's IDs. Construction preserves parentheses and does not do CSE,
reassociation or device compilation.

`Expr::from_nodes(nodes, root)` is a raw frontend entry point. Preparation checks
all nodes, including ones unreachable from the root, rejecting:

- Empty arenas or an out-of-range root.
- Out-of-range, forward or self references; operands must precede their parent,
  which also rejects cycles.
- Unknown names, unsupported used types or mismatched operand types.
- Expressions exceeding the size or depth limits.

Defaults are 4096 nodes and depth 128. A leaf has depth one; bounds are inclusive.
Use PrepareLimits and prepare_with_limits to change them. Node count is checked
before allocating the prepared arena. This is not a budget for memory already
allocated while constructing an expression or for device output memory.

The binding layout also includes input references from unreachable nodes. Raw
frontends should supply a pruned arena when they want dead inputs removed. Shared
DAG references are valid; the emitter reuses their SSA values. A root may precede
other nodes.

## Execution semantics and validation

Parameter-only programs broadcast over the batch row count. Zero rows still
validate required inputs, then produce empty output without allocation or launch.
The original arithmetic grouping is retained. See `runtime-emitter.md` for the
strict rounding workaround required by the pinned CubeCL optimizer and numerical
limits; general bitwise identity is not defined here. Nullable add/mul propagates
validity. Nullable comparisons, three-valued logic and IS NULL are defined in
[three-valued predicates](three-valued-predicates.md). F32 comparison
behavior for NaN, infinity and signed zero is defined in [predicates](predicates.md).

Twelve frontend tests cover names, types, ownership/Send/Sync, malformed arenas,
raw composition, unreachable nodes, inclusive bounds and a 10,000-node deep graph.
Ten separate layout tests exercise binding and address checks. Numerical cases
run real CPU/Metal kernels and assert explicit expected results rather than
maintaining a second AST execution engine.

```sh
cargo test --locked --no-default-features --test expressions
cargo test --locked --no-default-features --test binding_layout
cargo test --locked --features metal --test runtime_expr
```
