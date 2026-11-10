# M2 / F7 acceptance: bounded resident fragments

Measured on Apple M1, 16 GiB unified memory, macOS 15.5, aarch64, Rust 1.99.0,
release profile, CubeCL fork revision `5037c0639ce3b1205392bdbe43dcf25e2ea09e29`.
CPU and native Metal run sequentially on the same machine. These are local
observations with three samples per warm phase, not stable placement thresholds.

## Query and timing contract

The fixture has nullable I32 x[i] = i % 1000; validity is
(i % 100) < (100 - NULL percent). Filter x >= 1000 - pass_percent * 10,
project checked x + 1, return COUNT(*) and widening I64 SUM(x).
The pass percentage is the nominal pre-NULL predicate rate over a full value cycle;
NULLs correlate with values, so it is not the final selected percentage.
All paths validate identical output values, validity, selected rows and aggregates.
Mixed projection plus aggregate currently uses the reference path (no fusion).

- First query: fresh process device/client/executor initialization plus completed
  host query. Input construction and expected-result calculation are outside it.
  Disk compiler caches were not deleted, so this is process-first, not guaranteed
  cold compilation. Initialization time is recorded separately in the CSV.
- Synchronous: existing mask API, host selection/gather, nullable projection,
  then selected-input upload and aggregate APIs. Includes all completion/readback.
- Warm host: full host pipeline including upload and materialization, after warmup.
- Resident: same query repeatedly against one uploaded batch; upload is separately
  recorded as `resident_upload`. No query upload or new kernel definitions.
- Rust: standard iterator filter/fold, checked I32 addition, output validity and
  widening sum on the CPU. Checked Rust samples were recaptured in separate fresh
  processes after device measurements; `rust_separate` records this. The earlier
  unchecked primitive timings are excluded from published data.

Every timer covers completion and host results, not asynchronous submit alone.
Correctness comparisons and residency diagnostics run after the timer. No verified
per-kernel hardware timing is exposed here. N=0 Rust timings round to zero at this
clock precision; do not derive a speedup from them. First query has one sample;
other reported query medians have three, with min/max in the CSV.

## Core matrix (milliseconds)

| Backend | N | Pass % | NULL % | First query | Sync | Warm host | Resident | Rust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 0 | 50 | 10 | 4.712418 | 0.010167 | 0.016083 | 0.014791 | 0.000000 |
| cpu | 1 | 100 | 0 | 186.523125 | 10.117791 | 13.912250 | 20.048000 | 0.000084 |
| cpu | 1024 | 50 | 10 | 250.335749 | 31.767875 | 70.192375 | 117.805125 | 0.002666 |
| cpu | 65536 | 50 | 10 | 509.887251 | 363.551458 | 1754.129167 | 1712.297250 | 0.111875 |
| cpu | 1048576 | 50 | 10 | 14195.441792 | 641.497417 | 3379.484292 | 3392.044625 | 2.121167 |
| cpu | 2097152 | 50 | 10 | 12195.915209 | 6579.300083 | 16666.003875 | 6506.117125 | 4.213584 |
| metal | 0 | 50 | 10 | 63.104376 | 0.013375 | 0.018875 | 0.017209 | 0.000041 |
| metal | 1 | 100 | 0 | 224.980667 | 0.781916 | 0.424542 | 0.397084 | 0.000042 |
| metal | 1024 | 50 | 10 | 170.717875 | 0.860584 | 0.447792 | 0.427250 | 0.002583 |
| metal | 65536 | 50 | 10 | 153.057458 | 1.230666 | 0.623791 | 0.555000 | 0.111834 |
| metal | 1048576 | 50 | 10 | 165.403500 | 15.170666 | 9.245250 | 4.105000 | 2.057458 |
| metal | 2097152 | 50 | 10 | 164.200791 | 33.429625 | 14.966875 | 7.333750 | 3.969875 |

## Selectivity / NULL matrix at 64K (milliseconds)

| Backend | N | Pass % | NULL % | First query | Sync | Warm host | Resident | Rust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | 65536 | 0 | 10 | 324.371250 | 0.522208 | 207.215708 | 208.266583 | 0.040583 |
| cpu | 65536 | 1 | 10 | 319.178584 | 0.451167 | 343.079542 | 184.138916 | 0.086750 |
| cpu | 65536 | 50 | 0 | 247.732543 | 36.153167 | 310.793083 | 297.984208 | 0.125375 |
| cpu | 65536 | 50 | 10 | 509.887251 | 363.551458 | 1754.129167 | 1712.297250 | 0.111875 |
| cpu | 65536 | 50 | 100 | 324.050666 | 0.486084 | 263.336792 | 250.802208 | 0.023791 |
| cpu | 65536 | 100 | 10 | 327.817376 | 67.151750 | 203.479333 | 227.426083 | 0.224542 |
| metal | 65536 | 0 | 10 | 159.356334 | 0.579584 | 0.522292 | 0.412958 | 0.039583 |
| metal | 65536 | 1 | 10 | 163.559542 | 0.546542 | 0.481542 | 0.358917 | 0.038333 |
| metal | 65536 | 50 | 0 | 147.636959 | 1.194541 | 0.641625 | 0.677625 | 0.122875 |
| metal | 65536 | 50 | 10 | 153.057458 | 1.230666 | 0.623791 | 0.555000 | 0.111834 |
| metal | 65536 | 50 | 100 | 151.098083 | 0.609666 | 0.477375 | 0.356000 | 0.023458 |
| metal | 65536 | 100 | 10 | 149.936499 | 1.436000 | 0.665875 | 0.747333 | 0.185084 |

## Resource evidence and reuse

The harness sets 512 MiB logical bytes, 4M output rows and 4096 stages. Every
pipeline's measured explicit allocation bytes equal the admission bound minus
already-resident inputs. Estimates include input, mask, validity, checked error,
indices, all scan levels and all aggregate partial levels, without credit for
asynchronous temporary retirement. All accepted cases fit these limits.

At Metal N=65,536, 50% nominal pass, 10% NULL, the admission bound is 2,904,400
bytes, of which input is 524,288 bytes and resident-query new allocations are
2,380,112 bytes. Result-owned residency and estimated retained bytes are both
2,109,648. There are 12 launches / 14 graph stages, one count read, and seven
readbacks totaling 351,448 bytes. Warm host uploads two planes / 524,288 bytes;
resident queries upload zero. Runtime-specific tile sizes change scan/partial
sizes; the per-case CSV records those differences.

The 2,097,152-row admission bound is 98,266,544 bytes on CPU and 92,940,464 on
Metal. Reuse means one source upload and repeated query execution; output and
scratch buffers are still newly requested on each query. Zero warm definitions
means compiled kernel identities are reused, not that allocation or execution is
free. Raw samples include upload/readback calls and bytes, launches, count reads,
definitions, allocation attempts/bytes, retained and admission bounds.

Pool reports are collected after all query modes have completed and their outputs
have dropped; the resident input remains alive. `runtime_in_use` and
`runtime_reserved` are that final runtime snapshot. `sum_pool_page_peaks` sums
individual pools' page high-water counters, whose maxima need not coincide and
whose page sizes differ. It is not a global physical peak in bytes. Raw reports
preserve page sizes, padding and per-pool peaks. Driver/process/unified-memory
physical peak and kernel duration were not measured, and cache reservations are
not presented as those metrics.

## Placement conclusion and stage exit

Metal resident execution improves on the composed synchronous device path for
nonempty core cases, and avoids repeat input transfers. At 1M rows it measures
4.105 ms versus 15.171 ms synchronous and 9.245 ms warm host. Direct checked Rust
is still faster on this simple query. The CubeCL CPU reference path is especially
slow and variable for scan/reduction capacity work; it is a correctness and
portability path, not the preferred implementation for this CPU workload.

M2 therefore exits as an **experimental bounded resident device-fragment API**,
with measured transfer reuse and explicit rejection limits. These results do not
justify broad SQL acceleration or automatic GPU placement. M3 must retain an
explicit opt-in and compare DataFusion CPU end-to-end before expanding placement.
F7 does not add cross-batch aggregation, spill, concurrent global reservations or
new fusion eligibility. The existing NULL, checked-overflow, original-row,
ownership and deferred-completion contracts remain required.

## Reproduce and raw artifacts

```sh
CARGO_TARGET_DIR=/private/tmp/cuberel-m2-acceptance cargo build --release --offline --all-features --example milestone_acceptance
/private/tmp/cuberel-m2-acceptance/release/examples/milestone_acceptance metal 65536 50 10 3
/private/tmp/cuberel-m2-acceptance/release/examples/milestone_acceptance cpu 65536 50 10 3
/private/tmp/cuberel-m2-acceptance/release/examples/milestone_acceptance cpu 65536 50 10 3 --rust-only
```

Run one process per backend/case; capture stdout CSV and stderr pool reports.
The 11 cases per backend are (N, pass %, NULL %): (0,50,10), (1,100,0),
(1024,50,10), (65536,50,10), (1048576,50,10), (65536,0,10), (65536,1,10),
(65536,100,10), (65536,50,0), (65536,50,100), (2097152,50,10).
`--rust-only` skips device initialization and emits only the checked Rust baseline.

- [308 individual observations](milestone-03-acceptance-samples.csv)
- [132 phase summaries with ranges and counters](milestone-03-acceptance.csv)
- [22 runtime cases and pool snapshot totals](milestone-03-runtime-cases.csv)
- [Full pool reports](milestone-03-runtime-pools.txt)
- [Budget contract](../resource-budgets.md)
