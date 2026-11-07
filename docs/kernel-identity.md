# Kernel identity, definition reuse and failure completion

Milestone 01 work item 5 completes the identity and error-path WIP after PR #4.
The implementation was merged in [PR #5](https://github.com/kingfolk/cuberel/pull/5).
[Predicate support](predicates.md) extends these identities to boolean operations
and typed mask outputs. [Nullable projection](nullable-projection.md) adds
static node nullability and separate validity slots; dynamic null flags never
specialize the key. [Three-valued predicates](three-valued-predicates.md) adds
IsNull operand normalization and nullable Bool artifacts under codegen v6.

## Inspecting the key

```rust
let identity = executor.kernel_identity(&program);
let kernel_id = executor.kernel_id(&program);
let before = executor.define_count();
let result = executor.execute(&program, &batch, &parameters)?;
let new_definitions = executor.define_count() - before;
```

The first two methods inspect exactly the identity used by submission; they do
not initialize a launcher or invoke define. ExpressionIdentity is an owned,
inspectable snapshot, not a caller-supplied key that can override execution.
It includes:

- Codegen version (currently 9) and normalization format version (currently 1).
- Normalized typed nodes and the remapped output root.
- BindingLayout, including ABI version, physical types and address width.
- Full executor KernelSettings, including dimensions and execution mode.
- Client target name and the device-properties checksum.

CubeCL KernelId additionally identifies the Rust kernel type and records address
width, dimensions and execution mode. The IR's hashed entrypoint name is derived
from that id. Parameter values, input contents, device handles/addresses, row
counts and observation counters are excluded. The exact 1.0 constant used by
strict-add legalization is part of the versioned codegen policy; future user
literals or other baked constants must be included in the key before support is
added. A layout alone never identifies a computation.

For the current F32 emitter, no matrix target properties or autotuned target
choices affect emitted instructions. Device properties and target properties are
still injected during definition. CubeCL owns the per-service in-memory cache
and its persistent backend/architecture fingerprint namespace (see pinned
`cubecl-server/src/compiler/store.rs`). The Cuberel identity does not replace
that namespace or claim portable executable binaries. Additional target-dependent
lowering must extend this identity/version policy. Cuberel adds no disk cache or
cross-version serialization contract.

## Conservative structure normalization

The key traverses all validated arena nodes in topological order, interns identical
typed pure nodes, and remaps operand/root IDs. This makes a DSL expression that
duplicates `a * scale` equivalent to a raw DAG that shares that same intermediate.
The lookup HashMap is never iterated to determine output order, so randomized
hashing does not change the identity.

Operand order and floating-point grouping are retained. This is not algebraic
simplification, reassociation or full semantic equivalence. Different independent
node orders, names/schema IDs, or extra unreachable computations can produce
different keys. Unreachable inputs still participate in binding, as before;
normalization does not silently remove their validation. These conservative
misses are permitted. The emitter can retain the original arena: both forms have
identical pure computation and ABI under the current arithmetic contract.

## What the counter measures

`Executor::define_count()` counts entries to ExpressionKernel::define for kernels
submitted by that executor. Clones share an Arc/atomic counter; independently
constructed executors have separate counters. Counters have no reset operation
and do not affect keys. Observe a delta after checked readback; concurrent calls
through the same executor contribute to the same delta, so it is not per-query
attribution. Failed definitions count as attempts as well.

This is an IR-definition count, not a driver compilation count, timing metric or
disk-cache hit counter. A reused artifact can avoid define, and so can a warm
artifact compiled by another executor. The test uses a unique schema name for
its cold key without clearing user caches or changing global CubeCL settings.
After a cold key defines once, parameter, input address and row-count changes
must add zero definitions while still producing explicitly expected results.

The previous WIP's resettable process-global counter was replaced because it
would mix unrelated executors and parallel CPU/Metal tests.

## Invalid IR and completion errors

The isolated test kernel uses the exact production layout and distinct, valid
one-element F32 allocations. It constructs an IndexType-to-F32 store mismatch
and asserts that the module verifier rejects it. Two separate paths then run:

1. A definition-verification panic is converted by CubeCL into
   CompilationError::ExpansionPanicked.
2. The verifier diagnostic is forwarded with Scope::push_error; CPU and Metal
   compilers return CompilationError::Validation before lowering.

Both submissions return Enqueued, but reading the declared output must return
the typed error, including when wrapped by Unwritten/Several. The initialized
sentinel must never be returned as successful output. The read-only input stays
readable, and the next valid query on the same client/executor must succeed.
There is no deliberate out-of-bounds access or hardware-fault injection.

An exploratory test found that handing the malformed store directly to the
pinned CPU lowerer can lose the mismatch and succeed; this is not accepted as
correct rejection. That is why production verifies complete IR before build,
and why the diagnostic negative path refuses it before lowering. These tests
prove expansion/compiler-error propagation and recovery, not that every backend
automatically rejects every malformed IR or recovers from poisoned hardware.
The lowerer verification gap is tracked separately in local TODO.md.

## Verification

On 2026-11-03, the pinned dependency remained
`5037c0639ce3b1205392bdbe43dcf25e2ea09e29`, Rust/Cargo 1.99.0 on macOS/arm64.
At Milestone 01 completion the combined suite had 34 integration tests, including five new identity
and failure tests. Real CPU and native Metal are both required, without fallback
or ignored tests. The no-runtime suite retains 22 frontend/layout tests.

Each backend's reuse fixture observes five definitions on the primary executor
(combined, mul, add, repeated/shared DAG, different output root), and one on an
executor with different cube dimensions. Rebinding, different input lengths and
addresses, re-preparation, shared DAG normalization and reuse from another
executor introduce no extra definitions for an already compiled key. Empty and
invalid cold invocations define nothing. Counters are validated after readback.

```sh
cargo test --locked --offline --all-features
cargo test --locked --offline --all-features --test kernel_identity -- --nocapture
cargo test --locked --offline --no-default-features
cargo clippy --locked --offline --all-features --all-targets -- -D warnings
cargo rustdoc --locked --offline --no-default-features --lib -- -D warnings
cargo fmt --all -- --check
```

Validation uses `--target-dir /private/tmp/cuberel-build.hmThUI`; Metal runs with
GPU access outside the restricted sandbox. The expected test-only caught panic
may print a panic-hook message with nocapture; a passing test asserts its typed
completion error rather than treating that log as successful execution.

At Milestone 01 completion all listed checks passed: 34 combined integration tests (zero ignored), 22
no-runtime tests, Clippy/rustdoc with warnings denied, formatting and diff checks.
Milestone 01 is verified within its stated CPU/native Metal scope.

Current numeric extension: [checked I32](checked-i32.md) adds typed payloads and
a per-row overflow error plane; I32 and F32 operands do not implicitly mix.

Selected execution uses a distinct `selected-prefix-v1` tag with codegen v9.
The selection primitive uses v2 for its extended stage ABI. See
[selected projection](selected-projection.md) for guard, prefix and error contracts.
