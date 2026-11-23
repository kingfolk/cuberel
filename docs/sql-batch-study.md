# Matched SQL batch-size study (M4-F1)

This extends the [M3 acceptance harness](sql-acceptance.md) without changing the
adapter or its supported SQL semantics. DataFusion CPU and native Metal receive
the same session batch size, SQL, fixture, source partitions and expected result.
CubeCL CPU is a correctness/reference backend, not the optimized CPU baseline.

## Experiment contract

The primary fixture is the existing filtered integer global aggregate: 1,048,576
rows, predicate threshold 50%, requested NULL density 20%, one source chunk and
one source partition. The matched study also includes 65,536 and 2,097,152 rows.
Session batch sizes are 8192, 65536, 262144 and 1048576. These are explicit input
configuration choices, not automatic placement thresholds or adapter rebatching.
The source can still deliver smaller batches and empty batches; record actual
input batches and completed worker requests rather than inferring them from the
session setting. Empty requests can complete without launching device work.

Before sampling, the decision criteria are:

- Continue a larger-source-batch configuration path if the primary Metal query
  repeatedly improves by at least 2x against the original 8192-row configuration,
  with matching exact results and documented logical memory peaks. Confirm the
  selected comparison in three independent processes with ten warm samples each.
- Separately compare Metal against matched DataFusion CPU configurations and the
  best CPU configuration in the study. An improvement against the old Metal path
  does not establish a win against CPU. No production placement threshold is fit.
- If a native session setting is sufficient, do not introduce adapter-side
  concatenation, staging or a new scheduler. Keep byte/stage admission, error
  checks and cancellation behavior intact. Device state accumulation needs a
  separate decision based on the remaining costs.

Each fresh process records its first query and additional warm samples. Device
initialization is separate. Query timing includes session/plan preparation,
ordinary host input, completed execution, scalar-state merges and result export.
Fixture/oracle construction and validation are outside timing. Driver/OS caches
are not cleared. With ten warm samples, nearest-rank p95 is still the observed
maximum, not a reliable production tail estimate.

Performance jobs run sequentially after compilation/tests have stopped. Rotate
backend order across process repeats. Preserve every command, actual physical
plan, raw sample, environment/source/binary identity and completion marker.
Repeated source layouts that produce the same actual batch sizes are not
independent evidence of a batching improvement. Resource rejection is a recorded
failure, not permission to split/retry the query or substitute another backend.

## Commands and output

```sh
cargo build --locked --offline --release --all-features --example sql_acceptance \
  --target-dir /private/tmp/cuberel-f8-release
python3 scripts/sql_acceptance.py \
  --binary /private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  --out /private/tmp/cuberel-m4-batch-study \
  --batch-study --backends datafusion metal \
  --batch-sizes 8192 65536 262144 1048576 --warm 10 --repeats 3
```

Use a new output directory each run. A direct binary invocation can append a SQL
session batch size to the original nine arguments; omitting it retains 8192:

```sh
/private/tmp/cuberel-f8-release/release/examples/sql_acceptance \
  metal sql aggregate 1048576 50 20 1 1 10 65536
```

The ordinary F8 runner defaults are retained. The study mode uses SQL only;
direct DSL has no DataFusion session batch-size setting and is not labeled as a
matched batch-size experiment. Requested source chunk count, session batch size,
observed input batches, completed requests and fused/unfused requests are distinct.
Do not add overlapping worker/prepare/export wall metrics to derive query time;
use the independent query timer. Requested memory peaks are logical reservations,
not physical process/driver memory.

The synthetic fixture is a diagnostic workload. Conclusions apply to the tested
hardware/configurations, not all SQL or applications. Measurement results and the
next implementation decision are recorded separately with the raw evidence.
