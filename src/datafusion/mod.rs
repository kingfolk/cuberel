//! Optional DataFusion integration, pinned to DataFusion 55.2 / Arrow 59.2.
//!
//! Registration preserves DataFusion's default planning pipeline. Only an
//! explicitly supplied executor enables supported filter/project placement.

mod capability;
mod exec;
mod fragment;
pub use capability::{TranslatedExpression, Unsupported, translate_expression};
pub use exec::CubeExec;
pub use fragment::{FragmentInspection, inspect_fragment};

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
    engine: Option<Arc<exec::Engine>>,
}

impl AdapterOptions {
    /// Register CPU-only SQL planning without device placement.
    pub fn planning_only() -> Self {
        Self {
            enabled: true,
            engine: None,
        }
    }

    /// Explicit opt-in. Clones share one in-flight device request across queries.
    pub fn with_executor(executor: crate::Executor) -> Self {
        Self {
            enabled: true,
            engine: Some(Arc::new(exec::Engine::new(executor))),
        }
    }
}

/// Add the adapter after the builder's standard physical optimizer rules, then
/// validate the resulting properties. Does not replace the default rule set.
pub fn register(builder: SessionStateBuilder, options: AdapterOptions) -> SessionStateBuilder {
    if !options.enabled {
        return builder;
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
