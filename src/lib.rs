//! Cuberel compiles runtime query expressions into portable CubeCL kernels.
//!
//! Build expressions with [`col`] and [`param`], bind them with [`prepare`], and
//! execute them on a selected CubeCL client with [`Executor::execute`].
//! Expressions lower directly to IR at runtime, without generating Rust source.
//! Supports nullable `f32` and checked `i32` arithmetic and nullable boolean predicates; use
//! [`Executor::execute_nullable`] for value/validity projection output and
//! [`Executor::execute_predicate`] for three-valued Bool projections, or
//! [`Executor::execute_mask`] for dense WHERE flags, or [`Executor::select`]
//! for stable device selection indices and a device-side count.
//! [`Executor::upload`] creates an immutable [`DeviceBatch`] for repeated
//! [`Executor::execute_resident`] / [`Executor::select_resident`] calls;
//! [`DeviceResult::complete`] and materialize expose explicit blocking boundaries.
//! [`Executor::filter_selected`] refines a selection in original row coordinates;
//! [`Executor::project_selected`] computes only its active rows and returns a
//! [`DeviceProjection`] for compact host or Arrow export.
//! [`Executor::aggregate_resident`] and [`Executor::aggregate_selected`] return
//! pending single-batch COUNT/MIN/MAX/SUM scalars through [`DeviceAggregates`].
//!
//! [`Executor::execute_pipeline`] combines resident filters, projections and
//! aggregates with an inspectable [`StageGraph`], safe aggregate fusion and one
//! explicit completion/export boundary through [`DevicePipeline`].
//!
//! [`ResourceBudget`] limits request bytes, output capacity and stages before
//! allocation. [`Executor::estimate_pipeline`] exposes conservative plan costs;
//! [`Executor::execute_pipeline_host`] budgets upload and execution atomically.
//!
//! Backend features:
//! - `cpu` (default): CubeCL's LLVM-based CPU runtime.
//! - `metal`: CubeCL's native Metal runtime on Apple platforms.
//! - `arrow`: Arrow 60 host array/RecordBatch adapters, independent of runtimes.
//!
//! Both features can be enabled together. Backend selection must be explicit in
//! examples and tests, so a missing GPU cannot silently fall back to the CPU.

/// Host Arrow 60 adapters, available with the `arrow` feature.
#[cfg(feature = "arrow")]
pub mod arrow;

pub mod abi;
#[path = "device_abi.rs"]
pub mod cubecl_abi;

mod aggregate;
mod budget;
mod completion;
mod emitter;
mod error;
mod executor;
mod expr;
mod input;
mod ir_support;
mod pipeline;
mod reduction_kernel;
mod resident;
mod schema;
mod selected;
mod selection;
mod selection_kernel;

pub use aggregate::{Aggregate, AggregateResult, AggregateStats, AggregateValue, DeviceAggregates};
pub use budget::{BudgetError, BudgetResource, ResourceBudget, ResourceEstimate};
pub use emitter::{CODEGEN_VERSION, ExpressionIdentity};
pub use error::{Error, InputKind};
pub use executor::{ExecutionError, Executor, NullableBool, NullableF32, NullableI32};
pub use expr::{
    Comparison, Expr, ExprId, ExprNode, KernelProgram, PrepareLimits, ScalarOp, TypedExpr, col,
    param, prepare, prepare_with_limits,
};
pub use input::{Batch, ColumnData, Parameters, ScalarValue};
pub use pipeline::{
    DevicePipeline, FusionMode, Pipeline, PipelineResult, PipelineStage, StageGraph, StageKind,
};
pub use resident::{DeviceBatch, DeviceResult, ExecutionStats, HostColumn};
pub use schema::{ColumnId, DataType, Field, ParameterId, Schema};
pub use selected::{DeviceProjection, SelectedColumn};
pub use selection::{DeviceSelection, SelectionResidency, SelectionStats};
