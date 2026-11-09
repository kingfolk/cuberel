# Stable device selection

A WHERE mask retains input positions; selection compacts the kept original row
IDs. `Executor::select` now connects the existing runtime predicate emitter to
runtime IR scan/scatter kernels without a host mask round trip.

```rust
use cubecl::Device;
use cuberel::{Batch, ColumnData, DataType, Executor, Parameters, Schema, col, param, prepare};

fn main() -> Result<(), Box<dyn std::error::Error>> {
let executor = Executor::new(Device::cpu()?.client())?;
let schema = Schema::new(&[("x", DataType::I32)], &[("limit", DataType::I32)])?;
let predicate = prepare(&col("x").ge(param("limit")), &schema)?;
let batch = Batch::new(4).with_column("x", ColumnData::I32(&[10, 20, 30, 40]))?;
let parameters = Parameters::new().with("limit", 30_i32)?;
let selection = executor.select(&predicate, &batch, &parameters)?;
assert_eq!(selection.capacity(), 4); // allocation upper bound, not selected length
assert_eq!(selection.materialize()?, [2, 3]);

let selection = executor.compact_mask(&[0, 1, 0, 1])?;
assert_eq!(selection.selected_count()?, 2); // explicit completion + count read
assert_eq!(selection.materialize()?, [1, 3]); // repeats completion/count check
Ok(())
}
```

## Public contract

- `select` accepts a prepared Bool program; true keeps a row, false/NULL discards
  it. Invalid Bool payloads from the existing emitter are already canonical false.
  There is no extra validity download or host predicate evaluation.
- `compact_mask` validates every host flag is 0/1 before upload. No safe raw
  device-mask import is exposed. F32/I32 predicates use the same U32 algorithm.
- Submission returns a pending owned `DeviceSelection`; it does not request any
  readback. Submission latency is not kernel execution time and does not promise
  that backend enqueue/compilation is nonblocking.
- `selected_count()` first checks completion of all retained buffers, including
  current outputs and completion/error witnesses, then checks semantic errors
  and reads the four-byte count. It does not read indices.
- `materialize()` performs that same boundary, then reads exactly `4*count` bytes
  of indices. Empty selection returns an empty Vec without reading the unwritten
  capacity tail. Zero input validates bindings, allocates/launches nothing.
- Indices strictly increase, contain no duplicates and address the original
  input batch. Host parameters and input slices can be dropped after submission;
  uploaded buffers, scalars and intermediate handles have independent ownership.
- Clones share immutable buffers/source identity. `same_source` compares that
  identity; two separately submitted equal inputs are distinct snapshots. This
  is a minimal ownership contract, not a host-pointer-based residency cache.
- Repeated reads recheck completion/count/errors. There is no implicit readback
  memoization. Dropping without reading does not report errors; queued CubeCL
  bindings own resources until their queued work can safely release them.

## Algorithm and runtime IR

For workgroup size T, scan every block of T values using shared U32 memory and
an exclusive rank per logical row. Every lane initializes its shared slot;
out-of-range tail lanes supply zero but still participate in barriers.
The fixed template emits a Hillis–Steele inclusive scan and subtracts the lane's
original value to get an exclusive rank. Each round finishes all reads before
any shared write, then orders the writes before the next round.

```text
mask[N]
  -> scan -> local ranks[N], totals[ceil(N/T)]
  -> scan totals recursively until one block -> device count[1]
  -> propagate parent offsets from top to bottom -> global ranks[N]
  -> scatter kept original row IDs -> indices[capacity=N]
```

Offset propagation updates each rank in place, with exactly one writer per row.
Scatter consumes the root device count and writes only kept rows with
`rank < count`. It is scheduled from the known input capacity, without reading
count on the host. Distinct launches provide global ordering; there is no
cross-block spin wait, global barrier assumption or atomic append ordering.

For L scan levels there are L scans, L−1 offset launches and one scatter: 2L
selection launches. The optional predicate adds one expression launch.
The template supports non-power-of-two tiles. A nonempty selection currently
requires T>=2, T*4 shared bytes and at least five runtime bindings (four scatter
buffers plus metadata). Single-lane executors can still execute expressions;
nonempty selection returns `UnsupportedDevice` instead of failing to terminate
recursive planning.

`selection_kernel.rs` uses `KernelBuilder`, `Scope`, `ArrayType`, `SyncOp` and
ordinary arithmetic/control-flow/memory IR operations. Both conditional regions
yield, the function returns, and the complete module is verified before build.
No query `#[cube]`, generated Rust, runtime rustc or host expression interpreter
is involved. Stage resource order is shared by declaration and launcher, including
read/write attributes and unwind-safe launcher discard.

### Equivalent CubeCL Rust for the block scan

The following is a readable CubeCL Rust rendering of `Stage::Scan` in
`src/selection_kernel.rs`. It illustrates the algorithm; it is not an additional
production kernel or a dump of the generated IR. Launch setup is omitted:
use a one-dimensional workgroup with exactly `tile` lanes, matching the runtime
IR settings, and allocate one block count per launched workgroup.

```rust
use cubecl::prelude::{ABSOLUTE_POS, CUBE_POS, Shared, UNIT_POS, cube, sync_cube};

#[cube(launch)]
fn block_scan(
    input: &[u32],
    ranks: &mut [u32],
    block_counts: &mut [u32],
    row_count: u32,
    #[comptime] tile: usize,
) {
    let gid = ABSOLUTE_POS as usize;
    let lid = UNIT_POS as usize;
    let block = CUBE_POS as usize;

    // Tail lanes contribute zero but must still reach every barrier.
    let mut value = 0u32;
    if gid < row_count as usize {
        value = input[gid];
    }

    let mut shared = Shared::<[u32]>::new_slice(tile);
    shared[lid] = value;
    sync_cube();

    let mut distance = 1usize;
    while distance < tile {
        let current = shared[lid];
        let mut left = 0u32;
        if lid >= distance {
            left = shared[lid - distance];
        }
        let sum = current + left;

        // All lanes finish reading the old round before any overwrite.
        sync_cube();
        shared[lid] = sum;
        // Make this round's writes visible before the next round reads.
        sync_cube();
        distance *= 2;
    }

    let inclusive = shared[lid];
    if gid < row_count as usize {
        ranks[gid] = inclusive - value;
    }
    if lid == tile - 1 {
        block_counts[block] = inclusive;
    }
}
```

For `tile=4`, one block evolves as follows. Each round reads the previous
round's values, not partially updated values from its neighbours:

```text
input / initial shared:  [0, 1, 0, 1]
distance = 1:           [0, 1, 1, 1]
distance = 2:           [0, 1, 1, 2]  <- inclusive prefix sum
subtract original:     [0, 0, 1, 1]  <- exclusive block-local rank
block_counts[block] = 2
```

A rank counts selected rows strictly before the current row within its block.
Only selected rows use that rank as an output position; unselected rows may
share ranks without causing conflicting writes. The last lane's inclusive sum
includes every input in the block, so only that lane writes its block count.
Tail padding is zero, so this remains correct for a partial final block.
Never return early for an out-of-range lane before the barriers.

| CubeCL Rust operation | Runtime IR construction |
|---|---|
| `Shared::new_slice(tile)` | `ArrayType` and `Scope::create_shared` |
| Read `shared[index]` | `IndexOp` and `LoadOp` |
| `current + left` | `IAddOp` |
| `sync_cube()` | `SyncOp(SyncScope::Cube)` |
| Array assignment | `IndexOp` and `StoreOp` |
| `inclusive - value` | `ISubOp` |

The production emitter unfolds `distance=1,2,4,...` while constructing IR on
the host; its Rust loop is not a device loop. It also uses a clamped index and
`SelectOp` for the left-hand read, instead of the conditional read shown above.
Both contribute zero when `lid < distance`. All actual data reads, sums, barriers
and output writes execute on the selected device. The `#[cube]` attribute above
is explanatory syntax only; the production path continues to build IR directly.

## Errors, limits and identity

Predicate I32 arithmetic keeps its existing eager checked semantics. Even if
an overflowed predicate's mask happens to be all false, `selected_count` and
`materialize` return `IntegerOverflow` with the first original failing row.
No indices are exposed as success before that check. This feature does not yet
evaluate projection expressions on selected rows.

`sync_buffers` checks all retained resources, rather than assuming a successful
count proves every output succeeded. Backend failures stay `ExecutionError::Backend`;
failed computation does not fall back to CPU. A test-only output is deliberately
rejected by CubeCL's validation diagnostic path to verify delayed failure blocks
success even when count/indices independently succeeded. It executes no malformed
memory operations. Allocation/transport APIs can still panic as in the existing
executor; this is not a guarantee of recovery from arbitrary device faults.

The existing conservative U32 active-byte bound remains: capacity*4 must fit U32,
so the current maximum is floor(u32::MAX/4). All recursion sizes and grids are
planned before upload. This is not a claim that CubeCL intrinsically requires a
4 GiB byte limit. A smaller actual allocation/grid/shared-memory capability can
still limit the operation. Unified request memory budgets are a later feature.

Selection primitives have their own versioned identity (currently version 2, extended by F3), stage, full
settings/tile, target name and device checksum. Counts, row lengths, number of
recursion levels, values and addresses do not enter the primitive key. Expression
ABI v4 is unchanged; shared expression codegen is now v9. `Executor::define_count()` counts expression and
selection IR definition attempts, not driver compilation or disk-cache hits.

## Memory and readback costs

`SelectionStats` reports requested logical workspace bytes, index capacity bytes,
scan level count and selection launch count. These exclude the source mask or
predicate input/output planes, runtime scalar/metadata buffers, workgroup shared
memory and allocator caches. They are not a driver memory-peak report.

Workspace is `4 * sum(level_len + ceil(level_len/T))`; indices reserve `4*N` bytes.
Private masks/validity/workspace retire from Cuberel ownership after their final
consumers are enqueued; runtime bindings protect in-flight use. Counts and checked
error dependencies remain available for repeated materialization. See
[selection-only residency](selection-residency.md) for lifetime guarantees and metrics.

For N=17003, T=64, L=3, workspace is 70184 bytes and indices capacity 68012 bytes.
A host-mask call additionally uploads 68012 bytes, giving 206208 requested device
buffer bytes before runtime metadata/cache costs. There are six selection launches.
Count remains on device through every stage.

At completion the runtime performs a checked synchronization, then a four-byte
count read; a nonempty result reads only its initialized indices prefix. Checked
integer predicates additionally read their existing `4*N` error plane. Nullable
predicate validity is checked for backend completion without downloading its bytes.
No throughput or GPU speedup is claimed from launch counts or allocation estimates.

Run `stable_selection cpu` or `stable_selection metal` for an executable example.
It prints submit time, completion+count time and materialization time (which repeats
count/error checks), plus memory/launch statistics and define attempts. Timings
include host/runtime overhead and are not GPU event kernel timings. “first” is the
first measured run in that process, not a promise of an empty driver/disk cache.

## Validation and next steps

Real CPU/native Metal tests cover empty/single/tail input, full/empty/alternating/
sparse masks, multiple recursion levels, tiles 2/3/8/default, all-NULL predicates,
I32 and F32 predicates, checked errors, rebinding/reuse, source identity, repeated
reads, drop before completion and delayed backend errors. Extent arithmetic is
tested without giant allocations. Expected masks use Rust iterator filtering;
there is no generic reference interpreter.

This feature returns original row IDs; it does not gather data columns, export a
compacted Arrow RecordBatch, implement reusable DeviceBatch or expose a general
async scheduler. Those capabilities will extend these owned device results and
submission/completion boundaries in subsequent M2 features.

## Local verification record

Verified with Rust 1.99.0 on Apple M1 (MacBookPro17,1), using the repository's
locked dependencies and debug profile. All-feature validation passed 72 integration
tests and 3 unit tests on real CPU/native Metal, with zero ignored tests.
No-runtime validation passed 38 integration tests plus the extent unit test;
Arrow-only passed 46 integration tests plus that unit test. Strict all-target
Clippy, warnings-as-errors rustdoc and formatting passed. Additional focused
selection checks cover cross-thread completion and over-capacity count rejection.

One development sample of the 17003-row mask example (528 selected rows):

| Runtime / run | Tile | Scan levels / launches | Submit | Completion + count | Materialize, repeating count |
|---|---:|---:|---:|---:|---:|
| CPU / first | 8 | 5 / 10 | 2.042 ms | 2180.695 ms | 0.068 ms |
| CPU / reuse | 8 | 5 / 10 | 1.404 ms | 291.493 ms | 0.123 ms |
| Metal / first | 64 | 3 / 6 | 1.605 ms | 14.432 ms | 0.336 ms |
| Metal / reuse | 64 | 3 / 6 | 1.654 ms | 0.736 ms | 0.253 ms |

CPU workspace was 87464 bytes; Metal workspace was 70184 bytes. Both reserved
68012 index bytes, uploaded 68012 mask bytes and downloaded a 2112-byte index
prefix. Each count request reads four bytes. Define attempts stayed at four
through the reuse run (including the earlier expression demonstration).
These are single debug observations, not a statistically controlled benchmark
or CPU/GPU speedup claim. In particular, the barrier-heavy CPU template has
substantial overhead; backend-specific scan optimization remains future work.

[F3 selected execution](selected-projection.md) adds count-guarded scan/scatter,
nested filters in original row coordinates and compact projection/Arrow output.
