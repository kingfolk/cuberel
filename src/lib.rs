//! Cuberel compiles runtime query expressions into portable CubeCL kernels.
//!
//! Build expressions with [`col`] and [`param`], bind them with [`prepare`], and
//! evaluate them on the host with [`KernelProgram::evaluate_reference`].
//! This reference path does not initialize a device or emit CubeCL IR.
//! Only non-nullable `f32` arithmetic is currently supported.
//!
//! Backend features:
//! - `cpu` (default): CubeCL's LLVM-based CPU runtime.
//! - `metal`: CubeCL's native Metal runtime on Apple platforms.
//!
//! Both features can be enabled together. Backend selection must be explicit in
//! examples and tests, so a missing GPU cannot silently fall back to the CPU.

pub mod abi;
pub mod cubecl_abi;

mod error;
mod expr;
mod reference;
mod schema;

pub use error::{Error, InputKind};
pub use expr::{
    Expr, ExprId, ExprNode, KernelProgram, PrepareLimits, ScalarOp, TypedExpr, col, param, prepare,
    prepare_with_limits,
};
pub use reference::{Batch, ColumnData, Parameters, ScalarValue};
pub use schema::{ColumnId, DataType, Field, ParameterId, Schema};
