# Per-request resource budgets

`Executor::with_budget(ResourceBudget)` limits logical device bytes, output row
capacity and stage count. Defaults are unlimited, preserving existing behavior;
fixed shape limits (16 pipeline filters / 32 outputs) and device/ABI limits still
apply. Clones share counters/client but can have independent budget settings.

```rust
use cuberel::{Executor, ResourceBudget};
let executor = Executor::new(device.client())?.with_budget(ResourceBudget {
    max_device_bytes: 64 * 1024 * 1024,
    max_output_rows: 1_000_000,
    max_stages: 128,
});
// query is a Pipeline; host is a Batch; parameters are already prepared.
let estimate = executor.estimate_pipeline_host(&query, &host, &parameters)?;
println!("{estimate:?}");
let pending = executor.execute_pipeline_host(&query, &host, &parameters)?;
let result = pending.materialize()?;
```

The complete host fragment validates bindings, chooses its fusion/reference plan
and checks the budget **before the first upload/allocation/launch**. A missing
parameter in a late aggregate or an over-budget merge cannot leave earlier
filters partially submitted. The private allocation-free descriptor used for
host planning is never exposed/submitted as a resident batch.

`execute_pipeline` does the same for an existing `DeviceBatch`.
`estimate_pipeline` and `estimate_pipeline_host` validate the plan and return its
estimate without enforcing the budget, so callers can inspect an over-budget
plan. They allocate host metadata but perform no device calls or count reads.
The selected estimate is also available from `DevicePipeline::estimate()`.

## Admission bound and retention

`ResourceEstimate::device_bytes` is a conservative sum of supplied device inputs
plus **all new requested buffer capacities**. It does not assume that asynchronous
work has finished early enough for temporary storage to be reused. F6a does not
justify admitting against only the smaller post-completion footprint.

The estimate separates:

| Field | Counted logical bytes |
|---|---|
| input_bytes | Full supplied resident batch, or all columns of the host Batch to upload; standalone selected operations also include deduplicated selection/error dependencies |
| value_bytes | New expression value planes, including WHERE masks and aggregate argument arrays |
| validity_bytes | New nullable output validity planes |
| error_bytes | New checked expression error planes, including separate fused predicate/argument planes |
| indices_bytes | Capacity-N indices for every new compaction |
| scan_bytes | Every scan rank/total level, including final counts |
| partial_bytes | All partial/merge levels; widening SUM high words and fused hidden row-count states included |
| retained_bytes | Logical upper bound after private mask/validity/scan scratch and unused ancestor indices retire; includes supplied inputs and still-needed count/error/aggregate buffers |

`new_allocation_bytes()` excludes supplied inputs from the admission bound.
`DevicePipeline::residency_bytes()` observes currently result-owned logical bytes
by allocation identity without a device query. It can be smaller than the
retention upper bound when supplied inputs are unused or externally owned.
Externally held selections/batches/branches may share allocations, so adding
separate objects' residency numbers is not a global memory measurement.

For example, two filters use two masks and two indices allocations in the
admission bound. After completion only the current private indices and both
stages' required counts/errors remain. Caller-supplied parent selection storage
is not credited as free: external readers may still use it. Source columns and
shared dependencies are deduplicated by allocation identity for standalone
selected operations. Checked error planes can still grow with filter depth.

All byte products, totals and stage additions use checked arithmetic.
`BudgetError::Overflow` reports an unrepresentable estimate. `Exceeded` identifies
`BudgetResource::{DeviceBytes, OutputRows, Stages}` plus required and limit values.
Limits are inclusive: exactly meeting a bound succeeds. Rejections occur before
allocation attempts and do not trigger fallback to another backend or plan.

## Output rows and stages

`max_output_rows` bounds the **capacity per output column**, not total cells,
source participation count or actual selected count. Projection/selection capacity
is N; aggregate output is one row even for empty input. A mixed empty projection
and aggregate therefore still needs a one-row output budget. Width is accounted
in bytes. Zero-size device planes request zero bytes.

A scalar-output budget of one row permits capacity-N intermediate argument arrays;
those arrays are charged to device bytes, not misclassified as final row output.
A direct `upload` is a separate request returning an N-row batch and uses an
N-row output bound. Use `execute_pipeline_host` to budget upload and scalar query
as one fragment when that distinction matters.

A query's stage count includes its logical source and export boundaries plus
physical kernel launches. `StageGraph::stages().len()` matches it. Host pipeline
sources have kind `Upload`, resident ones `ResidentInput`; both are non-kernel
boundaries. Upload alone has one stage. Empty queries have no kernel launches but
still have source/exit boundaries. Transfers, allocations and count reads are
reported separately, not treated as additional graph kernels.

## Coverage and limits

Budgets also cover the existing upload, synchronous expression, resident multi-
expression, host/resident selection, compaction, nested filter, selected projection
and aggregate APIs. Each public call is one request. A globally checked composite
fragment uses the same client/counters for internal operations and does not apply
final-output limits to internal arrays. Existing binding, ownership and semantic
checks continue to run.

The byte budget excludes driver padding, scalar/metadata buffers, workgroup shared
memory, runtime staging/caches, compiler memory and host result/Arrow allocations.
Shared-memory/binding/grid capabilities still have their existing preflight
checks. This is not a device-wide reservation, concurrent-query limiter or OOM
guarantee. Backend allocation APIs may still panic; execution failures continue
through the existing completion/error contracts. No custom pool is introduced.

`ExecutionStats::{allocation_calls, allocation_bytes}` count explicit logical
allocation attempts through Cuberel, including uploaded planes. They exclude
runtime-private allocation traffic. They allow checking that admission rejects
without even allocating a temporary plane and auditing estimates against actual
requested bytes; they do not measure physical allocation or peak device memory.
Budgets and estimates do not enter compilation identities.

## Validation and M2 exit

The resource suite exercises real CPU/native Metal, inclusive and one-less byte/
row/stage limits, tails and multilevel reductions, fused/reference plans, shared
selection inputs, atomic host rejection, checked planes and filtered F32 SUM
fallback. Every accepted pipeline's measured requested allocation bytes match its
estimate exactly (after excluding already-resident input for resident calls).
Final verification: 127 tests with all features (real CPU and native Metal),
40 without default features, and 49 with Arrow alone. Strict all-target
Clippy, warnings-as-errors rustdoc and formatting checks pass.
Existing NULL, I32 overflow, original-row, ownership and deferred-failure suites
remain required. No generic query interpreter or new dependency was added.

The [M2 acceptance report](benchmarks/milestone-03-acceptance.md) compares first
query, synchronous calls, warm host, resident reuse and direct Rust primitives,
with raw timings/transfers/allocation bounds and runtime pool snapshots. This
closes the bounded M2 device-fragment stage; it does not establish general SQL
acceleration or cross-backend F32 equivalence. DataFusion integration remains M3.
