//! Optional DataFusion integration, pinned to DataFusion 55.2 / Arrow 59.2.
//!
//! Registration preserves DataFusion's default planning pipeline. Only an
//! explicitly supplied executor enables supported device fragments.

mod batch_policy;
mod capability;
mod diagnostics;
mod exec;
mod fragment;
mod global;
mod observation;
mod resources;
pub use crate::profiling::ProfilingMode;
pub use batch_policy::{BatchPolicy, BatchPolicyError};
pub use capability::{TranslatedExpression, Unsupported, translate_expression};
pub use diagnostics::{
    Placement, PlacementDecision, PlacementMode, PlacementReport, placement_report,
};
pub use exec::CubeExec;
pub use fragment::{FragmentInspection, inspect_fragment};
pub use global::{CubeAggregateExec, GlobalAggregateInspection, inspect_global_aggregate};
pub use resources::{ResourceSnapshot, SharedLimits, SharedResourceError, SharedResources};

use std::sync::Arc;

use ::datafusion::{
    common::{Result, config::ConfigOptions},
    execution::session_state::SessionStateBuilder,
    physical_optimizer::{PhysicalOptimizerRule, sanity_checker::SanityCheckPlan},
    physical_plan::ExecutionPlan,
};

/// SQL adapter configuration. Disabled by default; existing sessions are unchanged.
#[derive(Clone, Debug, Default)]
pub struct AdapterOptions {
    pub enabled: bool,
    pub(crate) profiling_mode: ProfilingMode,
    pub(crate) batch_policy: Option<BatchPolicy>,
    pub(crate) engine: Option<Arc<exec::Engine>>,
    pub(crate) placement_mode: PlacementMode,
}

impl AdapterOptions {
    /// Register CPU-only SQL planning without device placement.
    pub fn planning_only() -> Self {
        Self {
            enabled: true,
            profiling_mode: ProfilingMode::Off,
            batch_policy: None,
            engine: None,
            placement_mode: PlacementMode::OptIn,
        }
    }

    /// Explicit opt-in. Clones share one in-flight device request across queries.
    pub fn with_executor(executor: crate::Executor) -> Self {
        Self::with_resources(executor, SharedResources::default())
    }

    /// Share one admission gate and logical byte limits with other adapters.
    pub fn with_resources(executor: crate::Executor, resources: SharedResources) -> Self {
        Self {
            enabled: true,
            profiling_mode: ProfilingMode::Off,
            batch_policy: None,
            engine: Some(Arc::new(exec::Engine::with_resources(executor, resources))),
            placement_mode: PlacementMode::OptIn,
        }
    }
    /// Explicitly configure source granularity and cap actual device-worker input rows.
    /// Does not enable a disabled adapter or coalesce/split any input internally.
    /// Registration changes the session batch size, including CPU operators/sources.
    /// Rebuild queries to apply it; already-rewritten plans retain their engine.
    pub fn with_batch_policy(mut self, policy: BatchPolicy) -> Self {
        self.batch_policy = Some(policy);
        if let Some(engine) = &mut self.engine {
            Arc::make_mut(engine).batch_policy = Some(policy);
        }
        self
    }
    /// Enable fine-grained SQL worker diagnostics. Host adds no synchronization;
    /// Device adds coarse backend windows and boundary synchronization. Existing
    /// rewritten plans retain their engine; rebuild queries to change this mode.
    pub fn with_profiling_mode(mut self, mode: ProfilingMode) -> Self {
        self.profiling_mode = mode;
        if let Some(engine) = &mut self.engine {
            Arc::make_mut(engine).profiling_mode = mode;
        }
        self
    }
    pub fn with_placement_mode(mut self, mode: PlacementMode) -> Self {
        self.placement_mode = mode;
        self
    }
}

/// Add the adapter after the builder's standard physical optimizer rules, then
/// validate the resulting properties. Does not replace the default rule set.
pub fn register(mut builder: SessionStateBuilder, options: AdapterOptions) -> SessionStateBuilder {
    if !options.enabled {
        return builder;
    }
    if let Some(policy) = options.batch_policy {
        let config = builder.config().take().unwrap_or_default();
        builder = builder.with_config(config.with_batch_size(policy.source_rows()));
    }
    builder
        .with_physical_optimizer_rule(Arc::new(CubeRule { options }))
        .with_physical_optimizer_rule(Arc::new(SanityCheckPlan::new()))
}

#[derive(Debug)]
struct CubeRule {
    options: AdapterOptions,
}

impl PhysicalOptimizerRule for CubeRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if self.options.placement_mode == PlacementMode::CpuOnly {
            return exec::restore_cpu(plan);
        }
        if self.options.placement_mode == PlacementMode::RequireCandidates {
            diagnostics::require(&placement_report(plan.clone(), &self.options)?)?;
        }
        match &self.options.engine {
            Some(engine) => exec::rewrite(plan, engine),
            None => Ok(plan),
        }
    }

    fn name(&self) -> &str {
        "cuberel"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
