# Opt-in SQL filter/project execution

The `datafusion` feature supplies `register`, `AdapterOptions` and `CubeExec`.
Construct an explicit CubeCL CPU or native Metal executor and pass
`AdapterOptions::with_executor(executor)` to registration. Default options do not
install the adapter; `planning_only()` installs validation without device work.
Cloned options share one admission semaphore. Separately constructed options
are independent: this is not a global device-memory reservation service.

## Supported regions

F4 recognizes a contiguous chain of filters and pure column projections ending
in a supported projection or filter. Rename/reorder/drop mappings and a
FilterExec's output projection are composed by field position. Predicates retain
their bottom-to-top execution order, and all terminal outputs are evaluated only
after those filters. Inputs must be finite. All expressions in a node must pass
the [capability checks](sql-semantics.md); there is no partial expression extraction.
An intermediate computed projection stays a boundary, even when its output is
unused above: moving or deleting it could change checked errors. Filters with
fetch stay on CPU. The current limits are 16 filters, 32 terminal outputs and
64 physical nodes per region. If a longer chain fails translation or a limit,
the longest valid root prefix is used, with its lower boundary kept intact.
DataFusion may insert Repartition between filter and projection; that remains a
CPU/Arrow boundary. The example uses `target_partitions=1` to demonstrate one
combined filter/project fragment, while tests also retain default/multi-partition
plans. The adapter does not silently change the caller's partition settings.
Global aggregates, sort, limit, functions and unsupported types retain their
DataFusion operators. Discovery continues recursively below a region's input,
so both sides of a CPU or exchange boundary can contain independent CubeExecs.
Adjacent CubeExecs separated by an intermediate computation still have an Arrow
boundary; F4 does not inline that computation into a later filter.

Each input batch uploads referenced numeric columns once, executes the M2
pipeline with device selection/counts, then completes and exports host Arrow.
Projection output uses the exact SQL schema, field order, names, nullability and
metadata. Internal `cN`/`outN` names are private. Unreferenced columns are not
uploaded. Unsupported pass-through output columns (for example text selected
after a numeric WHERE) keep that region on CPU; they are never appended without
applying the selection. Bool is an expression output, not a supported input.

The equivalent original DataFusion region provides ordering, partitioning and
equivalence properties. Execution preserves per-partition input order and does
not rebatch across partitions. Replacing a child requires the same schema and
rebuilds the original region's properties before retranslation. The final
`SanityCheckPlan` checks distribution/ordering after the custom rule. Statistics
default to unknown rather than advertising unsupported precision.

`EXPLAIN` identifies CubeExec, filter count, physical node count, output count,
lower-boundary reason, semantic profile and the one-request admission limit.
`inspect_fragment(plan)` returns the same normalized expressions/programs and
field positions without device work, or a structured rejection when the root
cannot be placed. This also explains fully rejected roots (for example a
33-column projection), which remain ordinary CPU nodes in EXPLAIN. Basic
input/output row metrics are available. Whole-query placement reports and full
transfer/timing metrics are F7 work.
One logical export may involve multiple buffer/count/error readbacks; inspect
the executor's actual counters rather than assuming one read per fragment.

## Backpressure, cancellation and errors

`execute()` creates a lazy stream. No batch is uploaded before polling. Each
stream pulls one child batch on demand and never prefetches while its output is
waiting for a consumer. A semaphore shared by the adapter's queries/partitions
permits one device request at a time. Waiting partitions may retain one host
batch each; comprehensive host-byte reservations and fairness tuning remain F6.

Blocking upload/complete/materialize run in Tokio's blocking pool. The worker
owns the admission permit and input buffers until completion, including when
the output stream or awaiting future is dropped. Drop sets a cancellation flag;
a worker that has not submitted work can stop. Already submitted kernels are
not preempted: the bounded worker completes/error-checks and releases its own
resources. Dropping a JoinHandle is not treated as GPU completion or as permit
release. There are no background batch producers or unbounded result queues.

The existing per-request ResourceBudget is checked before upload. Runtime OOM,
kernel errors, checked errors and readback errors propagate as a failed stream
with partition/batch context; they are never retried on CPU. Earlier successful
batches may already have been consumed before a later error. Planning rejection
retains CPU placement; runtime budget rejection returns an error. Cross-query
host-pointer caches and adaptive rebatching are not implemented.

## Run

```sh
cargo run --locked --features datafusion --example datafusion_sql -- cpu
cargo run --locked --no-default-features --features datafusion,metal --example datafusion_sql -- metal
cargo test --locked --all-features --test datafusion_execution
```

The tests compare SQL/schema/NULL results with DataFusion CPU, exercise multiple
batches/partitions, empty selections, aliases, CPU boundaries, ordered limit,
budget rejection before allocation, lazy drop and no-prefetch behavior. A device
is explicitly required for its test; an unavailable GPU never passes as CPU.

Verified on 2026-11-10 with Rust 1.99, DataFusion 55.2.0, Arrow 59.2.0 and the
pinned CubeCL revision: 137 all-feature tests on actual CPU/native Metal and 53
SQL-only tests without a CubeCL runtime passed. No-feature all-target check,
all-feature/all-target Clippy and rustdoc with warnings denied passed. The debug
macOS linker reports a large unwind-table warning; linking and execution succeed.
No end-to-end SQL speedup is claimed; workload placement measurements remain F8.

F4 adds controlled physical fixtures for multiple filters, rename/reorder and
FilterExec projection maps, device→CPU→device execution, exchange/fetch limits,
shape splitting, empty/NULL/sliced batches and checked error order. Actual SQL
tests accompany these fixtures because DataFusion may simplify an equivalent
SQL subquery chain before physical placement.

F4 verification on 2026-11-12 passed 141 all-feature tests on actual CPU/native
Metal, 53 SQL-only tests, the no-feature all-target check, strict Clippy/rustdoc,
formatting and diff checks. Cancellation covers both a single region and two
regions separated by an exchange; canceled plans are reset with DataFusion's
`reset_plan_states` before re-execution. The walkthrough ran on both backends;
the native Metal run also exported full IR/MSL compilation records.

The `datafusion_walkthrough` example prints logical/default physical/rewritten
plans, normalized KernelPrograms, base ABI layouts, results and warm define
counts. On Metal, an optional fresh output directory exports an isolated CubeCL
environment with full compilation records; include the upstream
`cubecl/persistence` feature for that export. The pinned CPU backend writes the
textual traces and a records-unavailable notice instead. Default product features
and dependency sources remain intact.
