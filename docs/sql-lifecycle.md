# Shared SQL resources and stream lifecycle

`AdapterOptions::with_executor` creates one shared resource group for all its
clones, queries and partitions. Its defaults are 256 MiB logical device bytes,
512 MiB logical host bytes, 16 in-flight/waiting requests and 64 polled streams.
The device gate is FIFO and permits one blocking request at a time. To share the
same limits across independently registered adapters, pass a cloned controller:

```rust,ignore
use cuberel::datafusion::{AdapterOptions, SharedLimits, SharedResources};
let resources = SharedResources::new(SharedLimits {
    max_device_bytes: 128 * 1024 * 1024,
    max_host_bytes: 256 * 1024 * 1024,
    max_requests: 8,
    max_streams: 32,
});
let options = AdapterOptions::with_resources(executor.clone(), resources.clone());
// Register options.clone() in each participating session.
let snapshot = resources.snapshot();
```

Separately created controllers are independent, even for the same device. Direct
Executor calls and other applications are outside this accounting domain; this
does not reserve all physical device memory or replace DataFusion's memory pool.

## Admission, accounting and backpressure

An unpolled stream reserves nothing and submits nothing. On first poll it reserves
a stream slot and a small bounded scalar/metadata workspace. It pulls one child
batch, then atomically reserves input retention, conservative host workspace and
a request slot before waiting for the device gate. There is no prefetch producer.
The next child batch is not pulled while the current output awaits the consumer.
Aggregate streams similarly retain only bounded scalar state between requests.

Byte/slot exhaustion returns `SharedResourceError` immediately, with the resource,
required amount and limit. It does not retain an unadmitted batch in an unbounded
wait queue. Byte waits while holding inputs/outputs could deadlock nested device
fragments or a collecting consumer. Caller-visible failure also preserves batch
and checked-error boundaries: no automatic CPU retry, splitting or rebatching.
Nested child streams are polled before acquiring the device gate.

Input Arrow allocations are deduplicated by allocation base within the group;
clones/slices and repeated columns do not multiply the charge. A slice retains
its original allocation capacity. Externally owned Arrow buffers expose only
their visible extent, so that is the available conservative logical charge.
Host workspace reserves 4096 bytes plus 1024 bytes per input/output column and
32 bytes per input/output column per input row, covering bounded adapter
metadata, plane expansion, upload/readback copies and packing.
This intentionally overestimates many aggregate-only workloads.

After typed preflight, M2's conservative ResourceEstimate reserves shared device
bytes before upload. Per-executor ResourceBudget still applies independently.
The estimate includes input, validity, masks, scan/selection, errors, aggregate
states and output capacity, without credit for asynchronous temporary retirement.
Waiting requests hold host reservations but have not allocated device buffers.

Ready/delivered output uses a different lifetime: its actual Arrow buffer charge
is transferred atomically from workspace credit and attached to immutable buffer
owners. Dropping the stream does not release bytes still retained by a consumer's
batch/array clone or slice. Concrete Arrow types, offsets, metadata and validity
are preserved without copying the output values. A reservation conservatively
covers its output batch until the last attached buffer owner disappears.

Upstream source caches, CPU operators/exchanges, allocator bookkeeping, backend
staging/caches, driver metadata and physical memory peaks are not included. The
group bounds adapter-owned retained buffers/workspaces, not whole-process RSS.

## Cancellation and errors

Drop stops pulling input and cancels admission waits. A queued worker checks the
cancellation flag before work. Once running, it owns the input, byte reservations
and gate until complete/materialize finishes. Dropping/aborting its awaiting future
does not release those resources or preempt GPU kernels. A bounded detached worker
finishes cleanup; Drop does not synchronously wait on the async executor thread.

Worker errors and unwinding panics drain the backend before releasing admission.
If the drain itself fails/panics, the group is poisoned and rejects future work;
it cannot claim that unsafe capacity is reusable. A hung backend can retain its
bounded worker/reservations indefinitely; no fixed cancellation deadline is
promised. Ordinary checked/budget failures release resources and permit reuse.
Aborting panics/process OOM are outside Rust unwind recovery.

No global aggregate row is emitted before EOF or after a later batch error.
Filter/project consumers can already have received earlier successful batches.
`snapshot()` exposes live/peak logical host/device/request counts, polled streams,
successful completed requests, canceled/failed streams, rejections and poison state.
An abandoned polled stream counts as canceled even if it had consumed its last
batch without polling EOF. These are group counters, not per-query driver metrics.

Tests include allocation deduplication, sliced nullable output ownership,
pre-upload byte rejection, separate-adapter sharing, slow consumers, canceled
waiters, latched detached workers, panic drain/reuse, and real CPU/native Metal
filter/global/exchange paths. A worker latch verifies ownership without relying
on a particular kernel duration or unsafe memory access.

An optional [batch policy](sql-batch-policy.md) checks actual worker input rows
before workspace reservation/upload. It does not rebatch input or weaken byte
admission, output ownership or cancellation/drain guarantees.
