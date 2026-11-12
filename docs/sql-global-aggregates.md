# Bounded global SQL aggregates

The explicit DataFusion executor adapter supports built-in COUNT, Int32 MIN/MAX,
with no GROUP BY, DISTINCT, aggregate FILTER/ORDER BY or aggregate modifiers.
COUNT(*) counts rows; COUNT(expr) counts non-NULL Int32 results. Constant COUNT
arguments include COUNT(NULL). SQL COUNT is non-null Int64; MIN/MAX are nullable
Int32. The exact original SQL field names, order and metadata are retained.
SUM and floating aggregation are not part of this first aggregate layer.

```sql
SELECT count(*) AS n, count(x) AS valid, min(x) AS lo, max(x) AS hi
FROM t WHERE x >= CAST(30 AS INT);
```

For `[10,NULL,30,40]`, this returns one row `(2,2,30,40)`, regardless of how the
rows are split across accepted batches/partitions. Empty streams, zero-row batches,
all-NULL inputs and empty selections return one row at EOF: COUNT=0 and
MIN/MAX=NULL. A global result is never emitted after just one input batch.

## Placement and globality

`CubeAggregateExec` accepts a single-partition `Single` AggregateExec, or a
validated complete `Final -> [CoalescePartitions] -> Partial` region. Both
aggregate nodes must have matching built-in expressions and original schemas,
and independently pass all modifier/group checks. It consumes the original raw
rows, merging their partitions through DataFusion's CoalescePartitionsExec when
needed. It never treats Partial states as raw rows or replaces Final alone.
Other partial/final shapes remain DataFusion aggregation. Supported filter/project
regions beneath a rejected aggregate can still accelerate.

Pure column mappings and supported filters directly below the raw aggregate are
normalized into its resident pipeline, preserving filter and checked-error order.
Computed projections and exchanges retain their boundaries. Only referenced
numeric columns upload; COUNT(*) can use row counts without uploading a column.
At most 16 filters and 32 aggregate outputs are accepted.

Placement requires a proven total row upper bound from an immutable
MemorySourceConfig snapshot. Explicit known non-expanding filters, projections,
repartitions/coalescing and existing CubeExecs can propagate that proof. Estimated
statistics, arbitrary providers/operators and unbounded streams are not proofs;
they retain CPU aggregation with `unknown_aggregate_bound` inspection rejection.
`inspect_global_aggregate` exposes the same compiler's bound, counts and merge
contract without polling a stream or allocating device buffers.

## Execution and failures

Each batch uses the existing budgeted M2 aggregation pipeline. The worker completes
and checks device operations, reads small typed scalar states, then a fixed host
reducer merges checked counts and nullable Int32 extrema. This is not an
expression interpreter and does not collect raw rows. Multiple batches require
multiple small-state readbacks. Runtime input counts are checked against the
proven bound; SQL COUNT conversions are checked against Int64.

The stream is lazy and processes one batch at a time. Device admission is shared
with filter/project requests through the same adapter engine. The blocking worker
owns the permit and inputs through completion even if the stream is dropped.
Cancellation stops future pulling/submission; submitted kernels are not preempted.
No final row is emitted before EOF, or after a later batch/checked argument error.
Resource rejection happens before upload and runtime errors never retry on CPU.
Broader host-byte reservations/fairness remain the separate lifecycle feature.

```sh
cargo run --locked --all-features --example datafusion_aggregates -- cpu
cargo run --locked --all-features --example datafusion_aggregates -- metal
cargo test --locked --all-features --test datafusion_global
cargo test --locked --no-default-features --features datafusion --test datafusion_global
```

The integration tests compare exact schema/NULL values with DataFusion across
multiple batches/partitions, slices, empty inputs and selections; exercise unsafe
shape rejection, child replacement, budgets, cancellation and late checked errors.
No SQL performance improvement is claimed.
