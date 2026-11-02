# Build baseline: milestone 01, work item 1

Verified on 2026-11-01. Status: complete for the independent package and backend
build baseline. This does not complete milestone 01's query compiler or its
kernel execution requirements.

The canonical repository is `/Users/devilhill/Codes/cuberel`. The initial
implementation was migrated from an assistant workspace on 2026-11-01; all ten
transferred files were compared with their originals before the old copies were
removed. Source, configuration, documentation and future feature work belong in
this repository. The existing repository's Git history was preserved.

## Source and toolchain

| Component | Verified baseline |
|---|---|
| Cuberel | Independent edition-2024 library package, version 0.1.0 |
| Rust | 1.99.0, commit `b940084d7eb6a299eb4bfeb8e34901bc051e7ac4` |
| Cargo | 1.99.0 (`5f94df478`, 2026-08-27) |
| Host | `aarch64-apple-darwin`, macOS 15.5 (24F74) |
| Apple SDK tools | `/Library/Developer/CommandLineTools` |
| CubeCL | `https://github.com/kingfolk/cubecl.git` |
| CubeCL revision | `5037c0639ce3b1205392bdbe43dcf25e2ea09e29` |
| CubeCL workspace version | 0.11.0 |
| LLVM bundler | `tracel-llvm-bundler` 23.1.0-4, resolved in Cargo.lock |

The Cargo git checkout was checked against the pinned revision. It had no
tracked modifications; Cargo's generated `.cargo-ok` stamp was the only
untracked file. The separate user checkout at `/Users/devilhill/Codes/cubecl`
was not edited or switched.

Rust 1.95 is the package's declared minimum, inherited as a baseline requirement
from CubeCL's workspace. It was **not** tested in this work item. The development
toolchain version above is recorded, not enforced by a floating toolchain alias.

## Clean build and feature matrix

The initial CPU build used a newly created, empty target directory:
`/private/tmp/cuberel-build.hmThUI`. The subsequent feature configurations used
that same directory, compiling their additional or differently configured Rust
dependencies there. Downloaded source packages and native LLVM toolchain caches
were allowed; no prebuilt Rust artifacts from cubeql were linked.

| Configuration | Result | Scope |
|---|---|---|
| Default `cpu`: build | Pass | Independent build from pinned git source |
| Default `cpu`: test | Pass | 1 integration test: real CPU buffer round-trip |
| Metal only: test | Pass | 1 integration test: real native Metal buffer round-trip |
| CPU + Metal: test | Pass | Both integration tests passed, no ignored tests |
| No runtime: check all targets | Pass | Compile-only; no backend execution claimed |
| CPU smoke example | Pass | Explicit CPU selection, 20 bytes uploaded/read |
| Metal smoke example | Pass | Explicit Metal selection, 20 bytes uploaded/read |
| rustfmt | Pass | All project Rust files |
| Clippy, all features/targets | Pass | `-D warnings` for project code |

The integration tests compare all 67 bytes of a non-aligned-length payload.
The device runtime identity is checked before the transfer, preventing a GPU
request from silently turning into a CPU success. Runtime initialization is not
lazy query evaluation: these checks intentionally launch **no compute kernel**.

Observed example output:

```text
backend=Cpu device=cpu buffer_round_trip=20 bytes; no kernel launched
backend=Metal device=metal buffer_round_trip=20 bytes; no kernel launched
```

The runtime labels above are not hardware model names.

## Reproduce

Run from the repository root. Initial source/LLVM downloads require network
access; subsequent checks can use `--offline` once the cache is populated.

```sh
build_dir=$(mktemp -d /private/tmp/cuberel-build.XXXXXX)

cargo build --locked --target-dir "$build_dir"
cargo test --locked --target-dir "$build_dir"
cargo test --locked --no-default-features --features metal --target-dir "$build_dir"
cargo test --locked --features metal --target-dir "$build_dir"
cargo check --locked --no-default-features --all-targets --target-dir "$build_dir"

cargo run --locked --features metal --target-dir "$build_dir" --example backend_smoke -- cpu
cargo run --locked --features metal --target-dir "$build_dir" --example backend_smoke -- metal

cargo fmt --all -- --check
cargo clippy --locked --all-features --all-targets --target-dir "$build_dir" -- -D warnings
```

Metal runtime checks require an Apple host with GPU access. In this session,
the same Metal-only test first failed inside the command sandbox with
`No Metal device found`, then passed outside it with unchanged source and lockfile.
The combined CPU/Metal tests and Metal example also passed outside the sandbox.
This is an execution-permissions prerequisite, not a skipped GPU test or a
substitute for diagnosing future failures on other systems.

## Remaining milestone work

At the time of this build-baseline work item, typed expressions/reference
evaluation, binding ABI, runtime CubeCL IR emission, query execution, kernel cache
identity and kernel failure propagation were unimplemented. The subsequent
[typed-expression work item](typed-expressions.md) adds expression preparation
and host reference evaluation. No SQL, Arrow data processing, JIT query
correctness or performance claim follows from the build and transfer checks here.
