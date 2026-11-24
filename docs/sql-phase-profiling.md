# SQL worker phase attribution (M4 follow-up)

The previous [matched batching study](benchmarks/sql-m4-batch-study.md) established
a useful explicit source-configuration path but did not isolate the remaining
worker cost. This follow-up exposes four nested host-wall scopes described in
[SQL diagnostics](sql-diagnostics.md): `arrow_adapt`, `resource_preflight`,
`upload_submit`, and `complete_export`. It changes no execution order or SQL
support. None is a hardware kernel, pure transfer, or driver compilation timer.

The primary fixture remains the existing synthetic filtered integer aggregate:
1,048,576 rows, threshold 50%, NULL density 20%, one source chunk/partition, exact
host oracle. Explicit session batches are 65,536, 262,144 and 1,048,576. Each
CPU/Metal configuration runs in three fresh processes with ten warm samples;
backend order rotates. Preserve commands, raw samples, plans and environment.
Builds/tests finish before timing. This is a focused attribution matrix, not a
repeat of every already-tested layout/size combination.

Use the independent whole-query timer for before/after conclusions. Phase values
in a CSV sample are the same execution's per-partition metric sums; the single
partition avoids concurrent-partition wall overlap in this study. First/init and
warm costs remain separate; OS/driver caches are not cleared. Unavailable
CPU-only/DSL phase observations remain blank. Never sum phase medians, their
parents, or worker/stream times to manufacture query elapsed time.

Before sampling, select the next implementation by evidence: investigate batched
scalar-state readback only if complete/export is at least 10% of worker time at
the primary 64K setting. Otherwise prioritize the measured larger phase. This
does not reopen cross-batch device accumulation: within-result scalar readback
can reuse CubeCL's existing multi-buffer read without a new device state or IR.
Any candidate keeps an executable scalar reference and is benchmarked with exact
results, unchanged upload/launch/value bytes and failure/cancellation checks.
For this fixture, require at least 10% whole-query improvement across repeated
process medians to recommend an optimized mode; report weaker/inconsistent
results as such. CPU superiority is a separate unresolved goal.

```sh
cargo build --locked --offline --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/cuberel-m4-phase-study --batch-study \
  --study-rows 1048576 --batch-sizes 65536 262144 1048576 \
  --backends datafusion metal --repeats 3 --warm 10
```

Further 2M/8M or application-specific work is conditional on the identified
bottleneck and budget, not an automatic full matrix. The existing global
COUNT/NULL/widening-SUM and schema/error/resource contracts remain mandatory.
