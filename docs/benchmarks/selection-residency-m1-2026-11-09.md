# F6a residency comparison, Apple M1, 2026-11-09

F6a reduces persistent selection-owned buffers without changing compaction kernels,
launch counts or host-count synchronization. It does not consistently improve
latency in this local run; memory/lifetime behavior is the feature being delivered.

## Method

Apple M1, 16 GiB RAM, macOS 15.5; Rust 1.99.0; pinned CubeCL revision
`5037c0639ce3b1205392bdbe43dcf25e2ea09e29`. Release builds, default tiles CPU=8 and
native Metal=64. F6 reference is merged commit `342b5f9`; F6a uses the accompanying
implementation and `examples/selection_residency.rs`.

Each fresh process uploads one resident nullable I32 column: 65,536 rows,
`x[i]=i%1000`, `valid[i]=(i%10!=9)`. It applies four filters with the same
`x >= limit` predicate. Thresholds 0, 990 and 1001 cover high selectivity, sparse
selection and no matches. Each case warms twice, then records seven samples.
Versions run sequentially, not concurrently. Timings cover selection submission
through `selected_count()` completion, excluding upload, first compilation,
residency diagnostics and runtime memory reporting. They do not include exporting
the indices payload. Every count is checked against a host iterator.

The reference was built from a temporary archived checkout. Its Cargo package
name was changed to `cuberel-f6a-baseline` with `[lib] name="cuberel"` to isolate
build fingerprints while sharing cached dependencies; compaction source and
kernel settings stayed unchanged. An observation-only
[reference audit adapter](f6a-reference-audit.rs.txt) measures retained allocations
on that old private implementation. The benchmark harness replaces
`selection.residency().total_bytes()` with `selection.legacy_owned_bytes()` there.

## Application-owned logical buffers

These values deduplicate allocation identities reachable from the final selection
and its completion history, with no separately retained parent selections.

| Resource | F6 CPU | F6a CPU | F6 Metal | F6a Metal |
|---|---:|---:|---:|---:|
| Private WHERE masks | 1,048,576 | 0 | 1,048,576 | 0 |
| Private Bool validity | 1,048,576 | 0 | 1,048,576 | 0 |
| Indices (current + retained ancestors) | 1,048,576 | 262,144 | 1,048,576 | 262,144 |
| Scan workspace excluding retained final counts | 1,348,160 | 0 | 1,081,856 | 0 |
| Stage counts | 16 | 16 | 16 | 16 |
| Source dependencies | 524,288 | 524,288 | 524,288 | 524,288 |
| **Total bytes** | **5,018,192** | **786,448** | **4,751,888** | **786,448** |

Private mask ownership drops from four to zero. Total reachable logical bytes
fall by 84.3% on CPU and 83.4% on Metal. Current indices still reserve `4*N` bytes;
no readback is used to shrink them. Source buffers are shared with the batch and
counted once here. Checked arithmetic would additionally keep capacity-sized
error planes, which this simple predicate does not need.

Reference mask counts and totals were also audited directly in the old private
selection tree. New runtime-backed weak-binding tests check that masks, validity
and workspace have no remaining selection-owned path and no unnecessary strong
binding after consumer completion. External parent readers/branches are tested
separately; their legitimately live outputs are not forcibly released.

## Runtime pools and high-water marks

All three selectivities gave the same pool snapshot for a given version/runtime.
Reports are queried only after the final timed exit, never between filters.

| Runtime | Version | Runtime bytes in use at exit | Reserved/cache bytes at exit | Sum of per-pool peak pages |
|---|---|---:|---:|---:|
| CPU | F6 | 5,018,880 | 14,680,064 | 4 |
| CPU | F6a | 786,688 | 12,582,912 | 3 |
| Metal | F6 | 4,754,432 | 14,680,064 | 4 |
| Metal | F6a | 787,456 | 10,485,760 | 2 |

Runtime usage includes allocator padding and runtime metadata; application logical
bytes above do not. Reserved storage includes caches and may stay large after
references are dropped. `pages_peak` is the runtime's per-pool high-water mark;
adding peaks from different pools is **not** a simultaneous global byte peak or
an OS/driver-resident-memory measurement. Full pool kinds, sizes and marks are
preserved in the [raw runtime reports](selection-residency-m1-2026-11-09-pools.txt).

## Launches, transfers and latency

Both versions use 52 launches per CPU query and 28 per Metal query. Each submits
with zero uploads/readbacks/count reads; its explicit exit reads four counts
(16 bytes, four reads). No dummy reads or retirement-specific host synchronization
were added. Primitive kernel versions, layouts and allocation stats are unchanged.

| Runtime | Threshold | Selected | F6 median ms | F6a median ms | Change |
|---|---:|---:|---:|---:|---:|
| cpu | 0 | 58,983 | 441.600 | 468.273 | +6.0% |
| cpu | 990 | 585 | 423.804 | 418.623 | -1.2% |
| cpu | 1001 | 0 | 434.057 | 418.910 | -3.5% |
| metal | 0 | 58,983 | 0.554 | 0.556 | +0.5% |
| metal | 990 | 585 | 0.550 | 0.525 | -4.6% |
| metal | 1001 | 0 | 0.546 | 0.533 | -2.3% |

These sequential local samples do not establish a consistent speedup. The high
selectivity CPU median is about 6% slower in F6a; other cases are about 0.5% slower
to 4.6% faster. Raw ranges overlap and background activity/thermals were not
controlled. Keep this as memory/lifetime evidence, not an end-to-end acceleration
claim. The workload, hardware and timing boundary differ from the F6 fusion
benchmark; the two reports are not directly comparable.

## Reproduce

```sh
CARGO_TARGET_DIR=/private/tmp/cuberel-f6a-target \
  cargo build --locked --release --all-features --example selection_residency
/private/tmp/cuberel-f6a-target/release/examples/selection_residency cpu 0 65536 7 4
/private/tmp/cuberel-f6a-target/release/examples/selection_residency metal 990 65536 7 4
# Repeat for each runtime and thresholds 0, 990, 1001, one process at a time.
```

For the reference, archive `342b5f9` into a temporary directory, apply the package
identity isolation described above, append the observation adapter to its
`src/selection.rs`, and copy the example with the single owned-byte accessor
substitution. Build/run from that snapshot; do not overwrite the active checkout.
Use distinct target directories or isolated package identities to avoid stale
shared-target aliases. Raw summaries include the directly audited legacy bytes.

[Summary CSV](selection-residency-m1-2026-11-09.csv) includes logical/runtime bytes,
launches, transfers and latency ranges. [Raw timings](selection-residency-m1-2026-11-09-samples.csv)
include all 84 timed observations. Runtime pool reports are retained separately.
