# Reproducible SQL workload profiles

`sql_acceptance.py --manifest PATH` runs a saved, bounded list of existing SQL
fixtures. Each case fixes its source layout, session sizes, readback modes and
logical adapter memory caps. It avoids adding unrelated combinations when
validating an optimization. These are synthetic fixtures, not new SQL support or
an application latency guarantee.

```sh
cargo build --locked --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
# CPU is the CubeCL reference; datafusion is the SQL performance baseline.
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/sql-m4-boundaries --manifest benchmarks/sql-m4-boundaries.json \
  --backends datafusion cpu metal --warm 1
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/sql-m4-evaluation --manifest benchmarks/sql-m4-evaluation.json \
  --backends datafusion metal --repeats 3 --warm 10
```

The boundary profile covers empty/one-row, tile edges, no matches, all NULLs,
sliced Arrow buffers, multiple chunks/partitions and the CPU Sort/Limit boundary.
The evaluation profile extends the primary 1M-row aggregate to 8M rows, two source
partitions, selection/NULL extremes and materialized/mixed outputs. The baseline
at 8K uses the current library's Scalar mode with the M3 execution configuration;
it is not a rebuild of historical M3 code. Previous reports preserve that code's
measurements separately.

## Manifest contract

Version 1 has exactly `version`, `description` and `cases`. Each case requires:

| Field | Contract |
|---|---|
| `id` | Unique lowercase letters/digits/hyphens; starts with letter/digit, at most 64 characters |
| `workload` | `project`, `aggregate` or `mixed`, using the existing fixed SQL/oracle |
| `rows` | 0 through I32::MAX |
| `pass_percent`, `null_percent` | Integers from 0 through 100; pass is a nominal threshold before NULL filtering |
| `batches`, `partitions` | Positive source chunk/partition counts through I32::MAX; explicit empty chunks remain |
| `batch_sizes` | Nonempty, unique positive session sizes through I32::MAX |
| `readback_modes` | Nonempty, unique subset of `scalar`, `batched` |
| `shared_host_bytes`, `shared_device_bytes` | Positive U64 logical adapter caps, in bytes |

Unknown/missing/duplicate JSON keys, duplicate IDs and boolean numeric fields
fail before starting a case or creating output. A manifest cannot be combined
with `--quick`, `--batch-study`, `--study-rows` or nondefault command-line sizes
and modes. `--backends`, `--repeats` and `--warm` select execution coverage.
Manifest jobs are SQL only; default and batch-study workflows retain their
previous behavior. Each DataFusion baseline runs once per case/size/repeat;
device backends run each mode. Backend/mode order rotates between repeats.

The output directory archives the exact manifest bytes and hash alongside
source/binary identity, invocation, raw CSV, plans/metrics and summary. Combined
CSV and summary carry `case_id`. Every process must return the requested case,
layout, size, mode, sample sequence, caps and finite nonnegative timings; a
mislabeled/incomplete result cannot produce a successful COMPLETE marker.

## Budget and measurement boundary

The binary accepts trailing `--shared-host-bytes N` and
`--shared-device-bytes N` for SQL. Omit them to retain existing 512 MiB host /
256 MiB device defaults. DSL has no shared controller and rejects these flags.
For example:

```sh
/private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  metal sql aggregate 1048576 50 20 1 1 10 65536 batched \
  --shared-host-bytes 268435456 --shared-device-bytes 67108864
```

These configure the existing `SharedResources` admission rules; they do not
rebatch, retry, change device estimates or limit DataFusion's upstream memory.
Both modes and CPU baseline receive the same declared caps, but DataFusion alone
has no adapter allocations to enforce. Adapter caps and peak requests are
recorded in CSV, and device peaks must fit their caps. They exclude source/oracle
memory, RSS, backend/driver caches and DataFusion exchanges. In particular,
source Arrow slices may keep the full fixture allocation alive; a smaller
request does not guarantee a smaller retained host allocation.

Every sample checks exact schema/metadata/NULLs/values and actual device placement,
compares observed upload/read/allocate/launch counters to executor deltas, then
requires shared reservations to drain after consumer output is released. No
backend substitution or CPU retry is permitted. A failure leaves partial logs
and no COMPLETE marker. Error/cancellation/poison recovery remains covered by the
existing integration tests, rather than being inferred from successful timings.

The [original measurement contract](sql-acceptance.md) still defines timing,
fixture and phase semantics. Compare complete query time with same-run CPU and
Scalar references; never sum overlapping timers. Performance claims require
three process repeats with ten warm samples, and improvement above observed
variation. At the primary 64K aggregate retain the earlier 10% recommendation
gate; other cases test generalization and must disclose regressions. Larger total
input and larger request size are separate variables. CPU superiority remains an
independent, unfulfilled gate until a full matched query demonstrates otherwise.

The 8M-row sweep fixes requests at 64K/256K. A 1M-row request retains the full
source backing and reserves roughly 224 MiB workspace alone, so it is expected
to exceed the 256 MiB adapter host cap. Check that rejection separately; do not
increase the cap after seeing a failed timing or include a rejected case as a
successful measurement:

```sh
/private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  metal sql aggregate 8388608 50 20 1 1 1 1048576 batched \
  --shared-host-bytes 268435456 --shared-device-bytes 67108864
```
