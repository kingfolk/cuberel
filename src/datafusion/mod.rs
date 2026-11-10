//! Optional DataFusion integration, pinned to DataFusion 55.2 / Arrow 59.2.
//!
//! Registration preserves DataFusion's default planning pipeline. Device placement
//! is introduced separately; registering this initial adapter never launches work.

mod capability;
pub use capability::{TranslatedExpression, Unsupported, translate_expression};

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
        let _ = &self.options;
        Ok(plan)
    }

    fn name(&self) -> &str {
        "cuberel"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
