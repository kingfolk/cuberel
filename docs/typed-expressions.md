# Typed expressions and host reference evaluation

This implements milestone 01's second work item. It provides product expression
and preparation APIs plus a host reference evaluator for later differential
testing of device kernels. It does not emit CubeCL IR, define a kernel ABI, or
launch a query on a device.

## Public workflow

1. `Schema::new(columns, parameters)` declares two separate name/type namespaces.
2. `col("a") * param("scale") + col("b")` builds an owned expression graph.
3. `prepare(&expression, &schema)` checks the graph and binds names to typed IDs.
4. `Batch::new(row_count).with_column(...)` supplies borrowed host columns.
5. `Parameters::new().with(...)` supplies named scalars.
6. `program.evaluate_reference(&batch, &parameters)` returns `Result<Vec<f32>, Error>`.

See `examples/reference_expr.rs` for a runnable example that reuses a prepared
program with two different parameter sets. No device feature is needed to run it.

The program owns its schema and expression data via immutable `Arc`s; it can
outlive the input expression/schema, be cheaply cloned, and be shared across
threads. No CubeCL `Scope`, pliron `Value`, buffer handle or device client is
stored in the program.

## Types and names

`ColumnId` and `ParameterId` are positions within a program's schema, not globally
unique handles. `ExprId` is local to an expression/program arena. IDs from two
programs should not be interpreted as the same binding merely because their
numeric indexes match.

Only non-nullable `f32` arithmetic is implemented. `DataType::I32`,
`ColumnData::I32` and `ScalarValue::I32` can describe unsupported or mismatched
inputs so callers receive structured errors. They do not enable integer
arithmetic. Used I32 declarations fail during preparation; I32 data supplied for
an F32 declaration fails before reference output is evaluated. No implicit casts
are performed. Give float parameters an explicit `f32` suffix.

Columns and parameters may have the same name because they are different
namespaces. Within a namespace, empty and duplicate names are rejected. Names
are exact and case-sensitive. Names are resolved during prepare; evaluation
does not look up column/parameter names once per row. Binding order is irrelevant.
Unused schema fields and extra supplied inputs are permitted. Every supplied
batch column must still have the declared batch row count.

## Graph validation and bounds

Expressions use a flat arena, not recursively owned boxed nodes. Preparation,
evaluation and drop do not recurse through the graph. `+` and `*` merge arenas
and rebase the right operand's expression IDs. They preserve parentheses and do
not perform CSE, reassociation or compilation.

`Expr::from_nodes(nodes, root)` is a raw frontend entry point. Before the graph
becomes a `KernelProgram`, prepare rejects:

- Empty arenas and root indexes outside the arena.
- Out-of-range, forward and self references; operands must precede their parent.
  This topological-order requirement also rejects cycles.
- Unknown column or parameter names.
- Unsupported used input types.
- Expressions larger or deeper than the configured bounds.

All arena nodes are checked, including nodes unreachable from the root. Reference
evaluation likewise validates inputs for every node. Raw frontends should supply
a pruned arena when they want unused graph inputs removed.

Defaults are 4096 nodes and depth 128. A leaf has depth one; bounds are inclusive.
Use `prepare_with_limits` and `PrepareLimits` to change them. Node count is
checked before allocating the prepared arena. These limits constrain preparation,
not memory already allocated by a caller while constructing a source expression,
and do not constitute an output-memory budget.

## Reference semantics

For each row, the host interpreter evaluates validated nodes in arena order and
returns the root value. It uses one reusable scratch vector; intermediate columns
are not materialized. Runtime is proportional to rows times node count, and
scratch space is proportional to node count, in addition to output and binding
storage. This is a correctness reference, not an optimized CPU query engine.

A batch has an explicit row count. Parameter-only programs broadcast over this
count, and zero rows produce an empty result. Missing/type-invalid inputs still
fail for a zero-row batch: empty output cannot hide a binding error.

Arithmetic uses Rust `f32` addition and multiplication with the expression's
original grouping. No explicit `mul_add` or algebraic reassociation is used.
SQL NULL, NaN ordering, overflow policy and cross-device bitwise equivalence are
not defined here. Nonfinite inputs/results are not separately rejected by this
host evaluator; they follow the underlying Rust floating-point operations.

## Verification

The integration suite covers parameter rebinding, input order, separate name
namespaces, shared DAG operands, empty/parameter-only batches, missing/type-invalid
bindings, length mismatches, source ownership/Send/Sync, malformed graphs,
inclusive limits, a 10,000-node over-depth input, and numerical grouping that
distinguishes reassociated float expressions. A multi-row case compares results
with a separately written arithmetic formula.

```sh
cargo test --locked --no-default-features --test expressions
cargo run --locked --no-default-features --example reference_expr
```

On 2026-11-01, all 20 expression integration tests passed both without any runtime
feature and in the combined CPU/Metal build. The existing 2 device buffer tests
also passed in the combined build. The reference example produced:

```text
host reference, scale=2: [13.0, 25.0, 37.0]
host reference, scale=-1: [10.0, 19.0, 28.0]
```

Clippy with all features/targets and `-D warnings`, and rustdoc without runtime
features and with `-D warnings`, passed using the toolchain and temporary target
directory documented in the build baseline.

The subsequent [binding layout feature](binding-layout.md) adds a single device
ABI consumed by CubeCL declarations and launcher registration. The reference
interpreter remains independent of device execution.
