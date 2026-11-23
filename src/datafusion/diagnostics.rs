//! Explicit placement decisions. No device work, stream polling or cost guessing.
use super::{
    AdapterOptions, CubeAggregateExec, CubeExec, Unsupported,
    fragment::{Region, reject},
    global::{GlobalPlan, row_bound},
};
use ::datafusion::{
    common::{DataFusionError, Result},
    physical_plan::{
        ExecutionPlan, aggregates::AggregateExec, filter::FilterExec, projection::ProjectionExec,
    },
};
use arrow_schema::SchemaRef;
use std::{fmt, sync::Arc};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlacementMode {
    #[default]
    OptIn,
    CpuOnly,
    /// Every filter/project/global-aggregate candidate must be placeable.
    /// Sources, exchanges, sort/limit and other CPU operators are not candidates.
    RequireCandidates,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    Device,
    Cpu,
}
#[derive(Clone, Debug)]
pub struct PlacementDecision {
    pub fragment_id: String,
    pub node: String,
    pub placement: Placement,
    pub reason: Unsupported,
    pub boundary: Option<Unsupported>,
    pub input_schema: SchemaRef,
    pub output_schema: SchemaRef,
    pub input_partitions: usize,
    pub physical_nodes: usize,
    pub filters: usize,
    pub aggregates: usize,
    pub input_rows_upper_bound: Option<u64>,
    pub output_rows_upper_bound: Option<u64>,
    /// No static per-batch size is assumed. Runtime metrics record the estimate.
    pub estimated_device_bytes: Option<u64>,
    pub fusion: String,
}
#[derive(Clone, Debug)]
pub struct PlacementReport {
    pub mode: PlacementMode,
    pub batch_policy: Option<super::BatchPolicy>,
    pub device: Option<String>,
    pub semantic_profile: &'static str,
    pub per_request_budget: Option<crate::ResourceBudget>,
    pub shared_limits: Option<super::SharedLimits>,
    pub decisions: Vec<PlacementDecision>,
}
impl fmt::Display for PlacementReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "placement={:?}, device={:?}, semantic_profile={}, per_request_budget={:?}, shared_limits={:?}, batch_policy={:?}",
            self.mode,
            self.device,
            self.semantic_profile,
            self.per_request_budget,
            self.shared_limits,
            self.batch_policy
        )?;
        for d in &self.decisions {
            writeln!(
                f,
                "{} {}: {:?} [{}], boundary={:?}, nodes={}, filters={}, aggregates={}, input_partitions={}, input_rows_upper_bound={:?}, output_rows_upper_bound={:?}, per_batch_device_bytes={:?}, fusion={}; input={:?}; output={:?}",
                d.fragment_id,
                d.node,
                d.placement,
                d.reason,
                d.boundary,
                d.physical_nodes,
                d.filters,
                d.aggregates,
                d.input_partitions,
                d.input_rows_upper_bound,
                d.output_rows_upper_bound,
                d.estimated_device_bytes,
                d.fusion,
                d.input_schema,
                d.output_schema
            )?;
        }
        Ok(())
    }
}
pub fn placement_report(
    plan: Arc<dyn ExecutionPlan>,
    options: &AdapterOptions,
) -> Result<PlacementReport> {
    let mut report = PlacementReport {
        mode: options.placement_mode,
        batch_policy: options.enabled.then_some(options.batch_policy).flatten(),
        device: options
            .engine
            .as_ref()
            .map(|e| e.executor.client.name().to_string()),
        semantic_profile: "df55-i32-v1",
        per_request_budget: options.engine.as_ref().map(|e| e.executor.budget()),
        shared_limits: options.engine.as_ref().map(|e| e.resources.limits()),
        decisions: Vec::new(),
    };
    walk(plan, "root".into(), options, &mut report)?;
    Ok(report)
}
fn walk(
    plan: Arc<dyn ExecutionPlan>,
    id: String,
    options: &AdapterOptions,
    report: &mut PlacementReport,
) -> Result<()> {
    let selected = options.enabled
        && options.engine.is_some()
        && options.placement_mode != PlacementMode::CpuOnly;
    let disabled = reject(
        "disabled",
        if options.placement_mode == PlacementMode::CpuOnly {
            "CPU-only placement"
        } else {
            "no enabled executor"
        },
    );
    let mut boundary = None;
    let candidate = if let Some(exec) = plan.downcast_ref::<CubeAggregateExec>() {
        Some(Ok((
            exec.plan.input.clone(),
            exec.plan.nodes.len(),
            exec.plan.inspection.filters,
            exec.plan.inspection.aggregates,
            Some(exec.plan.inspection.input_upper_bound),
        )))
    } else if let Some(exec) = plan.downcast_ref::<CubeExec>() {
        boundary = Some(exec.region.boundary.clone());
        Some(Ok((
            exec.input.clone(),
            exec.region.nodes.len(),
            exec.fragment_filters(),
            0,
            row_bound(&exec.input, 0).ok(),
        )))
    } else if plan.is::<AggregateExec>() {
        Some(GlobalPlan::compile(plan.clone()).map(|p| {
            (
                p.input,
                p.nodes.len(),
                p.inspection.filters,
                p.inspection.aggregates,
                Some(p.inspection.input_upper_bound),
            )
        }))
    } else if plan.is::<FilterExec>() || plan.is::<ProjectionExec>() {
        Some(Region::discover(plan.clone()).map(|(r, p)| {
            boundary = Some(r.boundary);
            let bound = row_bound(&r.input, 0).ok();
            (r.input, r.nodes.len(), p.filter_count, 0, bound)
        }))
    } else {
        None
    };
    if let Some(candidate) = candidate {
        match candidate {
            Ok((input, nodes, filters, aggregates, bound)) => {
                use ::datafusion::physical_plan::ExecutionPlanProperties;
                report.decisions.push(PlacementDecision {
                    fragment_id: id.clone(),
                    node: plan.name().into(),
                    placement: if selected {
                        Placement::Device
                    } else {
                        Placement::Cpu
                    },
                    reason: if selected {
                        reject(
                            "explicit_opt_in",
                            "eligible; runtime capacity/capabilities still checked",
                        )
                    } else {
                        disabled
                    },
                    boundary,
                    input_schema: input.schema(),
                    output_schema: plan.schema(),
                    input_partitions: input.output_partitioning().partition_count(),
                    physical_nodes: nodes,
                    filters,
                    aggregates,
                    input_rows_upper_bound: bound,
                    output_rows_upper_bound: if aggregates > 0 { Some(1) } else { bound },
                    estimated_device_bytes: None,
                    fusion: if aggregates > 0 {
                        "M2 Auto; runtime preflight records actual fusion/fallback".into()
                    } else {
                        "unfused filter/project pipeline".into()
                    },
                });
                return walk(input, format!("{id}.0"), options, report);
            }
            Err(reason) => {
                use ::datafusion::physical_plan::ExecutionPlanProperties;
                report.decisions.push(PlacementDecision {
                    fragment_id: id.clone(),
                    node: plan.name().into(),
                    placement: Placement::Cpu,
                    reason: if !options.enabled || options.placement_mode == PlacementMode::CpuOnly
                    {
                        disabled
                    } else {
                        reason
                    },
                    boundary: None,
                    input_schema: plan
                        .children()
                        .first()
                        .map_or_else(|| plan.schema(), |p| p.schema()),
                    output_schema: plan.schema(),
                    input_partitions: plan
                        .children()
                        .first()
                        .map_or(0, |p| p.output_partitioning().partition_count()),
                    physical_nodes: 1,
                    filters: usize::from(plan.is::<FilterExec>()),
                    aggregates: 0,
                    input_rows_upper_bound: None,
                    output_rows_upper_bound: None,
                    estimated_device_bytes: None,
                    fusion: "CPU".into(),
                });
            }
        }
    }
    for (i, child) in plan.children().into_iter().enumerate() {
        walk(child.clone(), format!("{id}.{i}"), options, report)?;
    }
    Ok(())
}
pub(crate) fn require(report: &PlacementReport) -> Result<()> {
    if let Some(d) = report
        .decisions
        .iter()
        .find(|d| d.placement == Placement::Cpu)
    {
        return Err(DataFusionError::Context(
            format!("Cuberel required candidate {} ({})", d.fragment_id, d.node),
            Box::new(DataFusionError::External(Box::new(d.reason.clone()))),
        ));
    }
    if report.decisions.is_empty() {
        return Err(DataFusionError::Plan(
            "Cuberel required candidates: no candidate fragments".into(),
        ));
    }
    Ok(())
}
