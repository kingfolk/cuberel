//! Cuberel compiles runtime query expressions into portable CubeCL kernels.
//!
//! Build expressions with [`col`] and [`param`], bind them with [`prepare`], and
//! execute them on a selected CubeCL client with [`Executor::execute`].
//! Expressions lower directly to IR at runtime, without generating Rust source.
//! Supports nullable `f32` arithmetic and non-nullable boolean predicates; use
//! [`Executor::execute_nullable`] for value/validity projection output and
//! [`Executor::execute_mask`] for dense 0/1 predicate output.
//!
//! Backend features:
//! - `cpu` (default): CubeCL's LLVM-based CPU runtime.
//! - `metal`: CubeCL's native Metal runtime on Apple platforms.
//!
//! Both features can be enabled together. Backend selection must be explicit in
//! examples and tests, so a missing GPU cannot silently fall back to the CPU.

pub mod abi;
pub mod cubecl_abi;

mod emitter;
mod error;
mod executor;
mod expr;
mod input;
mod schema;

pub use emitter::{CODEGEN_VERSION, ExpressionIdentity};
pub use error::{Error, InputKind};
pub use executor::{ExecutionError, Executor, NullableF32};
pub use expr::{
    Comparison, Expr, ExprId, ExprNode, KernelProgram, PrepareLimits, ScalarOp, TypedExpr, col,
    param, prepare, prepare_with_limits,
};
pub use input::{Batch, ColumnData, Parameters, ScalarValue};
pub use schema::{ColumnId, DataType, Field, ParameterId, Schema};
