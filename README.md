# Cuberel

A Rust query compiler targeting portable compute kernels through CubeCL.

The canonical development directory is `/Users/devilhill/Codes/cuberel`.
Product source, tests, configuration and documentation belong in this repository;
temporary build directories may live under `/private/tmp`.

The library builds and validates runtime expressions, lowers non-nullable `f32`
arithmetic and boolean predicates to CubeCL IR, and executes on a selected CPU or native
Metal client. No `#[cube]`, Rust source generation or runtime rustc invocation is
used by this query path. SQL and Arrow integration are not yet implemented.
The backend smoke example separately initializes a real device and round-trips
a buffer; it does **not** compile or launch a compute kernel.

## Runtime expression execution

```rust
use cubecl::Device;
use cuberel::{Batch, ColumnData, DataType, Executor, Parameters, Schema, col, param, prepare};

let executor = Executor::new(Device::cpu()?.client())?;

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
let result = executor.execute(&program, &batch, &parameters)?;
assert_eq!(result, [13.0, 25.0, 37.0]);
```

A prepared program can execute again with new named parameters or columns. Backend
selection is explicit; execution checks the output read before returning a result.
Empty batches validate inputs but allocate and launch nothing. The temporary host
interpreter has been removed; numerical tests assert independent expected results.

```sh
cargo run --locked --example runtime_expr -- cpu
cargo run --locked --no-default-features --features metal --example runtime_expr -- metal
cargo test --locked --features metal --test runtime_expr
# Frontend construction/validation still works without a linked runtime.
cargo test --locked --no-default-features --test expressions
```

See [the expression API and semantics](docs/typed-expressions.md) for preparation
limits and raw node graphs. See [device execution and its limits](docs/runtime-emitter.md)
for IR construction, floating-point policy and the verification report.

## Predicate masks

Compare F32 expressions with `eq_expr`, `ne_expr`, `lt`, `le`, `gt`, `ge`;
compose predicates with `and`, `or` and `!`. Execute a Bool program with
`executor.execute_mask(&program, &batch, &parameters)` to obtain one U32 flag
per input row (0 or 1), retaining row positions. F32 programs use `execute`.

For `(a * scale) > limit AND a < upper`, input `[1, 3, 5]` with parameters
`scale=2`, `limit=4`, `upper=5` produces `[0, 1, 0]`. See
[predicate types, semantics and ABI](docs/predicates.md).

```sh
cargo run --locked --example predicate_mask -- cpu
cargo run --locked --no-default-features --features metal --example predicate_mask -- metal
```

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
They verify argument transport separately from the query emitter in `Executor`.

## Kernel identity and reuse

`executor.kernel_identity(&program)` exposes the normalized typed structure,
binding ABI, codegen policy, settings and target description;
`executor.kernel_id(&program)` returns the actual CubeCL cache identity.
Parameters, row counts and buffer addresses do not specialize the key. Repeated
pure nodes and shared DAG references normalize to the same structure, while
operand order and floating-point grouping remain significant.

`executor.define_count()` counts IR definition attempts submitted by that executor
and its clones. It does not count driver compilations or disk-cache hits. Tests
verify real CPU/Metal reuse and that definition/compiler failures are reported
when reading output, followed by successful valid execution on the same client.
See [identity, observation and failure guarantees](docs/kernel-identity.md).

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

`DESIGN.md`, `MILESTONE-01.md`, `MILESTONE-02.md`, `TODO.md` and `AGENTS.md` are local
planning documents and agent instructions intentionally excluded by exact `.gitignore` entries.
Product source, tests, this README and
`Cargo.lock` remain eligible for version control.
