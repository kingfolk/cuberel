//! Cuberel compiles runtime query expressions into portable CubeCL kernels.
//!
//! The initial package establishes the dependency and backend build baseline.
//! Query expressions, parameter bindings and IR lowering are not implemented yet.
//!
//! Backend features:
//! - `cpu` (default): CubeCL's LLVM-based CPU runtime.
//! - `metal`: CubeCL's native Metal runtime on Apple platforms.
//!
//! Both features can be enabled together. Backend selection must be explicit in
//! examples and tests, so a missing GPU cannot silently fall back to the CPU.
