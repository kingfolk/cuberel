# Cuberel

A Rust query compiler targeting portable compute kernels through CubeCL.

The canonical development directory is `/Users/devilhill/Codes/cuberel`.
Product source, tests, configuration and documentation belong in this repository;
temporary build directories may live under `/private/tmp`.

The library supports runtime expression construction, schema/type validation,
and host reference evaluation of non-nullable `f32` addition and multiplication.
A shared binding layout now drives CubeCL buffer/scalar declarations and argument
registration, with real CPU and Metal kernel tests. General expression-to-IR
lowering, SQL and Arrow integration are not yet implemented.
The backend smoke example separately initializes a real device and round-trips
a buffer; it does **not** compile or launch a compute kernel.

## Expressions and host reference evaluation

```rust
use cuberel::{Batch, ColumnData, DataType, Parameters, Schema, col, param, prepare};

let schema = Schema::new(
    &[("a", DataType::F32), ("b", DataType::F32)],
    &[("scale", DataType::F32), ("bias", DataType::F32)],
)?;
let expression = col("a") * param("scale") + col("b") + param("bias");
let program = prepare(&expression, &schema)?;
let a = [1.0_f32, 2.0, 3.0];
let b = [10.0_f32, 20.0, 30.0];
let batch = Batch::new(a.len())
    .with_column("a", ColumnData::F32(&a))?
    .with_column("b", ColumnData::F32(&b))?;
let parameters = Parameters::new()
    .with("scale", 2.0_f32)?
    .with("bias", 1.0_f32)?;
let result = program.evaluate_reference(&batch, &parameters)?;
assert_eq!(result, [13.0, 25.0, 37.0]);
```

A prepared program can be evaluated again with new named parameters or columns.
`evaluate_reference` is an independent host interpreter, not a CubeCL CPU kernel,
and does not initialize a device. It works with no runtime feature enabled:

```sh
cargo run --locked --no-default-features --example reference_expr
cargo test --locked --no-default-features --test expressions
```

See [the expression API and semantics](docs/typed-expressions.md) for preparation
limits, error handling, raw node graphs and the boundaries of this feature.

## Kernel binding ABI

`cuberel::abi::BindingLayout::new(&program)` determines input/output buffer order,
physical scalar types and per-type scalar ordinals. The same layout declares
CubeCL IR arguments and validates/registers runtime arguments through
`cuberel::cubecl_abi`. It checks lengths, capacity, alignment, U32 address bounds,
device ownership and output aliasing before launcher registration. Empty batches
skip submission. The device adapter is a low-level interface; an arbitrary kernel
body still requires an unsafe caller contract.

```sh
cargo run --locked --no-default-features --example binding_layout
cargo test --locked --no-default-features --test binding_layout
cargo test --locked --features metal --test binding_device
```

See [the binding ABI and validation report](docs/binding-layout.md). Its device
tests build a small probe kernel directly with CubeCL IR APIs, without `#[cube]`.
They verify argument transport; the general query emitter is the next work item.

## Dependency baseline

- Rust edition 2024; declared minimum Rust version: 1.95 (not yet tested at the minimum).
- Development toolchain: `rustc 1.99.0 (b940084d7 2026-09-28)` and Cargo 1.99.0.
- CubeCL: [kingfolk/cubecl](https://github.com/kingfolk/cubecl), pinned to
  `5037c0639ce3b1205392bdbe43dcf25e2ea09e29` (workspace version 0.11.0).
- This is the fork revision used by the original design investigation, not a
  claim that the same commit exists in the upstream repository.
- `Cargo.lock` is part of the project and should be committed. Use `--locked`
  for repeatable dependency resolution. No absolute local dependency paths or
  prebuilt cubeql artifacts are required.

## Backends

| Cargo feature | Runtime | Requirements |
|---|---|---|
| `cpu` (default) | CubeCL CPU / LLVM | Supported native toolchain, C/C++ linker and CubeCL's LLVM bundle |
| `metal` | Native Metal | macOS, a Metal-capable device and Xcode Command Line Tools |
| none | No runtime linked | Compile-time library check only; no execution backend |

`metal` forwards to CubeCL's `metal-native`, not its wgpu-based `metal` alias.
Features are additive; use `--no-default-features --features metal` for a
Metal-only build. The example names the requested backend explicitly and does
not silently fall back to another device.

The first build needs network access to fetch the pinned git revision and
registry dependencies. The CPU build uses `tracel-llvm-bundler` to locate or
download LLVM; a manual system LLVM installation is not the default setup.
On macOS, install the Apple command-line tools (`xcode-select --install`) if
the linker/SDK is missing. Native LLVM linking may also require `zstd` and
other platform libraries; preserve the build diagnostics when investigating
missing native libraries rather than replacing the pinned dependencies.

## Build and check

```sh
cargo build --locked
cargo test --locked
cargo run --locked --example backend_smoke -- cpu

# macOS / native Metal, with no CPU runtime
cargo build --locked --no-default-features --features metal
cargo test --locked --no-default-features --features metal
cargo run --locked --no-default-features --features metal --example backend_smoke -- metal

# Both backends, and compile-only with neither runtime
cargo test --locked --features metal
cargo check --locked --no-default-features --all-targets
cargo fmt --all -- --check
```

`--features metal` on a non-Apple host does not establish Metal runtime support.
The Metal tests run only on Apple targets; a skipped test is not GPU validation.

## Clean build verification

Use a fresh target directory instead of deleting an existing build:

```sh
validation_target=$(mktemp -d /private/tmp/cuberel-build.XXXXXX)
cargo build --locked --target-dir "$validation_target"
cargo test --locked --target-dir "$validation_target"
cargo run --locked --target-dir "$validation_target" --example backend_smoke -- cpu
```

Sharing downloaded source packages or an LLVM toolchain cache is fine. This
check must compile Rust dependencies from source into the fresh target; it
must not link old cubeql `.rlib` files. The temporary target can be retained
for follow-up validation.

The verified environment, feature matrix and reproduction commands are recorded
in [the build baseline report](docs/build-baseline.md). In a restricted macOS
sandbox, Metal can compile successfully while device initialization fails with
`No Metal device found`. The Metal tests must also run in an environment that
allows GPU access; do not replace them with a CPU fallback or a passing skip.

## Local planning documents

`DESIGN.md`, `MILESTONE-01.md` and `AGENTS.md` are local planning documents and
agent instructions intentionally excluded by exact `.gitignore` entries.
Product source, tests, this README and
`Cargo.lock` remain eligible for version control.
