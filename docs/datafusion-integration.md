# Optional DataFusion integration

Enable `datafusion` to use the opt-in SQL adapter. It implies `arrow`, but no
CubeCL runtime: `--no-default-features --features datafusion` supports DataFusion
CPU SQL and planning. The default DSL build does not depend on DataFusion/Tokio.

## Dependency contract

DataFusion is pinned to **55.2.0** with default features disabled and SQL enabled.
Its release uses Arrow 59.2, so the public Arrow adapter moves from 60.0.0 to
**59.2.0**. Consumers must use the same Arrow major; there is no dual-version
RecordBatch bridge, unsafe conversion or serialization round trip. The alignment
is a public Rust type dependency change, not a device ABI or semantic change.
The CubeCL git revision and declared Rust 1.95 minimum are unchanged; validation
uses the installed Rust 1.99 toolchain, not an untested MSRV claim.

## Registration

```rust
use cuberel::datafusion::{register, AdapterOptions};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::SessionContext;

let state = register(
    SessionStateBuilder::new_with_default_features(),
    AdapterOptions::planning_only(),
).build();
let context = SessionContext::new_with_state(state);
```

Disabled options leave the builder unchanged. Enabled registration appends a
`cuberel` physical rule and a final `SanityCheckPlan` to the existing rules,
preserving DataFusion's correctness-critical distribution/ordering enforcement.
The final sanity check is deliberate: appended custom rules must also be checked.
`planning_only()` preserves the original CPU plan. Use
`AdapterOptions::with_executor(executor)` to explicitly enable the supported
filter/project device path. SQL parsing, catalog, binding and optimization remain
DataFusion responsibilities. Planner execution never uploads or launches work.
See [SQL semantics](sql-semantics.md) and [execution/lifetime limits](sql-execution.md).

## Checks

```sh
cargo test --locked --no-default-features --features datafusion --test datafusion_registration
cargo test --locked --no-default-features --features arrow
cargo test --locked --all-features
cargo tree --locked --no-default-features -e normal
```

The SQL feature is optional. Arrow regression tests cover independently sliced
values/validity, schema checks, nullable/Bool packing and checked integer errors.

Verified for the F1 layer on 2026-11-10: SQL registration/CPU execution passed
without CubeCL runtimes; the Arrow-only suite passed after alignment. The normal
dependency tree without default features contains neither DataFusion nor Tokio.
Historical M1/M2 reports retain the Arrow 60 version actually used for those runs.
