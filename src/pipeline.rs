//! Bounded, acyclic resident query fragments with an executable reference path.
use crate::{
    Aggregate, AggregateResult, DataType, DeviceAggregates, DeviceBatch, DeviceProjection,
    DeviceResult, DeviceSelection, ExecutionError, Executor, Field, KernelProgram, Parameters,
    SelectedColumn,
    abi::BindingLayout,
    aggregate::fused::FusedPlan,
    completion::{ExitContext, check_all, synchronize},
};
use cubecl::prelude::Client;
use std::collections::BTreeSet;

/// Auto falls back to the device reference plan when fusion cannot preserve
/// semantics or fit the device. Required reports that condition before launch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FusionMode {
    Unfused,
    #[default]
    Auto,
    Required,
}
/// One resident source, up to 16 ordered filters and 32 named outputs.
/// Output expressions always bind original source columns, not output aliases.
#[derive(Clone, Debug, Default)]
pub struct Pipeline {
    filters: Vec<KernelProgram>,
    projections: Vec<(String, KernelProgram)>,
    aggregates: Vec<(String, Aggregate)>,
    fusion: FusionMode,
}
impl Pipeline {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn filter(mut self, predicate: KernelProgram) -> Self {
        self.filters.push(predicate);
        self
    }
    pub fn project(mut self, name: impl Into<String>, program: KernelProgram) -> Self {
        self.projections.push((name.into(), program));
        self
    }
    pub fn aggregate(mut self, name: impl Into<String>, aggregate: Aggregate) -> Self {
        self.aggregates.push((name.into(), aggregate));
        self
    }
    pub fn fusion(mut self, mode: FusionMode) -> Self {
        self.fusion = mode;
        self
    }
}
/// Physical stages. ResidentInput and Materialize are boundaries, not launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    ResidentInput,
    Predicate,
    Scan,
    AddOffsets,
    Scatter,
    Projection,
    AggregatePartial,
    AggregateMerge,
    FusedAggregatePartial,
    Materialize,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineStage {
    pub kind: StageKind,
    pub dependencies: Vec<usize>,
}
/// Topologically ordered stages. Every dependency index is strictly smaller
/// than its stage index. The graph describes this submission, including tails.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StageGraph {
    stages: Vec<PipelineStage>,
    fusion_reason: Option<&'static str>,
}
impl StageGraph {
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }
    pub fn fusion_reason(&self) -> Option<&'static str> {
        self.fusion_reason
    }
    pub fn kernel_launches(&self) -> usize {
        self.stages
            .iter()
            .filter(|s| !matches!(s.kind, StageKind::ResidentInput | StageKind::Materialize))
            .count()
    }
    pub fn is_fused(&self) -> bool {
        self.stages
            .iter()
            .any(|s| s.kind == StageKind::FusedAggregatePartial)
    }
    fn push(&mut self, kind: StageKind, dependencies: Vec<usize>) -> usize {
        let id = self.stages.len();
        debug_assert!(dependencies.iter().all(|&i| i < id));
        self.stages.push(PipelineStage { kind, dependencies });
        id
    }
}
/// A pending query fragment. All branches complete and pass error checks before
/// any host result is exposed. No implicit host reads happen during submission.
#[derive(Clone)]
pub struct DevicePipeline {
    client: Client,
    rows: u32,
    selection: Option<DeviceSelection>,
    dense: Option<DeviceResult>,
    selected: Option<DeviceProjection>,
    aggregates: Option<DeviceAggregates>,
    fields: Vec<Field>,
    graph: StageGraph,
}
/// Compact projection columns and one scalar per aggregate. These are separate
/// shapes: aggregates are not broadcast to the projection row count.
#[derive(Clone, Debug, PartialEq)]
pub struct PipelineResult {
    pub row_count: u32,
    pub projections: Vec<SelectedColumn>,
    pub aggregates: Vec<AggregateResult>,
}
impl DevicePipeline {
    pub fn graph(&self) -> &StageGraph {
        &self.graph
    }
    fn finish(&self) -> Result<(u32, u32), ExecutionError> {
        let mut handles = Vec::new();
        if let Some(s) = &self.selection {
            handles.extend(s.handles());
        }
        if let Some(p) = &self.dense {
            handles.extend(p.handles());
        }
        if let Some(p) = &self.selected {
            handles.extend(p.handles());
        }
        if let Some(a) = &self.aggregates {
            handles.extend(a.handles());
        }
        synchronize(&self.client, &handles)?;
        let mut exit = ExitContext::default();
        // Filter-stage errors take precedence over terminal expressions.
        let rows = if let Some(s) = &self.selection {
            exit.count(s)?
        } else {
            self.rows
        };
        let mut checks = Vec::new();
        if let Some(p) = &self.dense {
            checks.push(p.complete_in(&mut exit));
        }
        if let Some(p) = &self.selected {
            checks.push(p.complete_in(&mut exit).map(|_| ()));
        }
        let mut aggregate_rows = rows;
        if let Some(a) = &self.aggregates {
            checks.push(a.complete_in(&mut exit).map(|count| {
                aggregate_rows = count;
            }));
        }
        check_all(checks)?;
        Ok((
            if self.graph.is_fused() {
                aggregate_rows
            } else {
                rows
            },
            aggregate_rows,
        ))
    }
    pub fn complete(&self) -> Result<u32, ExecutionError> {
        self.finish().map(|(rows, _)| rows)
    }
    pub fn materialize(&self) -> Result<PipelineResult, ExecutionError> {
        let (rows, aggregate_rows) = self.finish()?;
        let projections = if let Some(p) = &self.selected {
            p.read_columns(rows)?
        } else if let Some(p) = &self.dense {
            p.read_columns()?
                .into_iter()
                .zip(&self.fields)
                .map(|(data, field)| SelectedColumn {
                    field: field.clone(),
                    data,
                })
                .collect()
        } else {
            Vec::new()
        };
        let aggregates = self
            .aggregates
            .as_ref()
            .map(|a| a.read_results(aggregate_rows))
            .transpose()?
            .unwrap_or_default();
        Ok(PipelineResult {
            row_count: rows,
            projections,
            aggregates,
        })
    }
}
impl Executor {
    /// Validate the complete fragment before any allocation/launch. Auto and
    /// Unfused both run on this executor's explicitly chosen device.
    pub fn execute_pipeline(
        &self,
        pipeline: &Pipeline,
        batch: &DeviceBatch,
        parameters: &Parameters,
    ) -> Result<DevicePipeline, ExecutionError> {
        batch.validate_client(&self.client)?;
        let invalid = |reason| ExecutionError::InvalidPipeline { reason };
        if pipeline.filters.len() > 16
            || pipeline.projections.len() + pipeline.aggregates.len() > 32
        {
            return Err(invalid(
                "pipeline supports at most 16 filters and 32 outputs",
            ));
        }
        if pipeline.projections.is_empty() && pipeline.aggregates.is_empty() {
            return Err(invalid("pipeline needs a projection or aggregate output"));
        }
        let mut names = BTreeSet::new();
        for name in pipeline
            .projections
            .iter()
            .map(|(n, _)| n)
            .chain(pipeline.aggregates.iter().map(|(n, _)| n))
        {
            if name.is_empty() || !names.insert(name) {
                return Err(invalid("output names must be nonempty and unique"));
            }
        }
        for (index, predicate) in pipeline.filters.iter().enumerate() {
            self.require_output_type(predicate, DataType::Bool)?;
            let layout = BindingLayout::new(predicate);
            batch.validate(&self.client, &layout, parameters)?;
            self.preflight_expression(&layout, u64::from(batch.row_count()))?;
            if index > 0
                && batch.row_count() != 0
                && self.client.properties().hardware.max_bindings < 7
            {
                return Err(ExecutionError::UnsupportedDevice {
                    reason: "nested selection needs 7 buffer bindings including metadata",
                });
            }
            if index > 0 {
                self.preflight_projection(batch, true, &[("predicate", predicate)], parameters)?;
            }
            self.selection_plan(batch.row_count())?;
        }
        let projection_refs: Vec<_> = pipeline
            .projections
            .iter()
            .map(|(n, p)| (n.as_str(), p))
            .collect();
        if !projection_refs.is_empty() {
            self.preflight_projection(
                batch,
                !pipeline.filters.is_empty(),
                &projection_refs,
                parameters,
            )?;
        }
        let aggregate_refs: Vec<_> = pipeline
            .aggregates
            .iter()
            .map(|(n, a)| (n.as_str(), a.clone()))
            .collect();
        let reduction = if !aggregate_refs.is_empty() {
            Some(self.preflight_aggregates(
                batch,
                !pipeline.filters.is_empty(),
                &aggregate_refs,
                parameters,
            )?)
        } else {
            None
        };
        let mut graph = StageGraph::default();
        let source = graph.push(StageKind::ResidentInput, Vec::new());
        let prepared = if pipeline.fusion == FusionMode::Unfused {
            Err("fusion disabled by caller")
        } else if !pipeline.projections.is_empty() || pipeline.filters.len() > 1 {
            Err("fusion supports aggregate-only fragments with at most one predicate")
        } else {
            FusedPlan::prepare(
                self,
                batch,
                pipeline.filters.first(),
                &aggregate_refs,
                parameters,
            )
        };
        let mut dense = None;
        let mut selected = None;
        let mut selection = None;
        let mut terminals = Vec::new();
        let aggregates = match prepared {
            Ok(plan) => {
                let (first, merges) = plan.launches();
                let mut last = source;
                if first != 0 {
                    last = graph.push(StageKind::FusedAggregatePartial, vec![source]);
                }
                for _ in 0..merges {
                    last = graph.push(StageKind::AggregateMerge, vec![last]);
                }
                terminals.push(last);
                Some(plan.submit(self, batch, &aggregate_refs, parameters)?)
            }
            Err(reason) => {
                if pipeline.fusion == FusionMode::Required {
                    return Err(invalid(reason));
                }
                graph.fusion_reason = Some(reason);
                let mut current = source;
                for predicate in &pipeline.filters {
                    let next = if let Some(parent) = &selection {
                        self.filter_selected(predicate, batch, parent, parameters)?
                    } else {
                        self.select_resident(predicate, batch, parameters)?
                    };
                    if batch.row_count() != 0 {
                        current = graph.push(StageKind::Predicate, vec![current]);
                        let levels = next.stats().scan_levels;
                        for _ in 0..levels {
                            current = graph.push(StageKind::Scan, vec![current]);
                        }
                        for _ in 1..levels {
                            current = graph.push(StageKind::AddOffsets, vec![current]);
                        }
                        current = graph.push(StageKind::Scatter, vec![current]);
                    }
                    selection = Some(next);
                }
                if !projection_refs.is_empty() {
                    if let Some(s) = &selection {
                        selected =
                            Some(self.project_selected(batch, s, &projection_refs, parameters)?);
                    } else {
                        dense = Some(
                            self.execute_many_resident(
                                &pipeline
                                    .projections
                                    .iter()
                                    .map(|(_, p)| p.clone())
                                    .collect::<Vec<_>>(),
                                batch,
                                parameters,
                            )?,
                        );
                    }
                    for _ in &pipeline.projections {
                        terminals.push(if batch.row_count() != 0 {
                            graph.push(StageKind::Projection, vec![current])
                        } else {
                            current
                        });
                    }
                }
                if aggregate_refs.is_empty() {
                    None
                } else {
                    for (_, a) in &aggregate_refs {
                        let mut last = current;
                        if a.program().is_some() && batch.row_count() != 0 {
                            last = graph.push(StageKind::Projection, vec![last]);
                            for level in 0..reduction
                                .as_ref()
                                .expect("reduction plan")
                                .stats
                                .reduction_levels
                            {
                                last = graph.push(
                                    if level == 0 {
                                        StageKind::AggregatePartial
                                    } else {
                                        StageKind::AggregateMerge
                                    },
                                    vec![last],
                                );
                            }
                        }
                        terminals.push(last);
                    }
                    Some(if let Some(s) = &selection {
                        self.aggregate_selected(batch, s, &aggregate_refs, parameters)?
                    } else {
                        self.aggregate_resident(batch, &aggregate_refs, parameters)?
                    })
                }
            }
        };
        graph.push(StageKind::Materialize, terminals);
        Ok(DevicePipeline {
            client: self.client.clone(),
            rows: batch.row_count(),
            selection,
            dense,
            selected,
            aggregates,
            fields: pipeline
                .projections
                .iter()
                .map(|(n, p)| Field::new(n, p.output_type()).with_nullable(p.output_nullable()))
                .collect(),
            graph,
        })
    }
}

#[cfg(all(
    test,
    any(feature = "cpu", all(feature = "metal", target_vendor = "apple"))
))]
mod tests {
    use super::{FusionMode, Pipeline};
    use crate::{
        Aggregate, Batch, ColumnData, DataType, ExecutionError, Executor, Parameters, Schema, col,
        prepare,
    };
    use cubecl::Device;
    use std::sync::Arc;
    fn failure(device: Device) {
        let e = Executor::new(device.client()).unwrap();
        let schema = Schema::new(&[("x", DataType::I32)], &[]).unwrap();
        let x = prepare(&col("x"), &schema).unwrap();
        let pred = prepare(&col("x").eq_expr(col("x")), &schema).unwrap();
        let batch = e
            .upload(
                &Batch::new(2)
                    .with_column("x", ColumnData::I32(&[1, 2]))
                    .unwrap(),
            )
            .unwrap();
        let query = Pipeline::new()
            .filter(pred)
            .project("x", x.clone())
            .aggregate("sum", Aggregate::Sum(x))
            .fusion(FusionMode::Unfused);
        let mut out = e
            .execute_pipeline(&query, &batch, &Parameters::new())
            .unwrap();
        out.complete().unwrap();
        let marker = crate::selection::completion_tests::rejected_handle(&e.client, e.settings());
        Arc::get_mut(&mut out.selected.as_mut().unwrap().outputs).unwrap()[0]
            .submitted
            .retained
            .push(marker);
        let before = e.stats();
        assert!(matches!(out.materialize(), Err(ExecutionError::Backend(_))));
        assert_eq!(
            e.stats(),
            before,
            "sync every resource before reading count/errors/payloads"
        );
    }
    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_pipeline_checks_every_branch() {
        failure(Device::cpu().unwrap());
    }
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    #[test]
    fn metal_pipeline_checks_every_branch() {
        failure(Device::metal(Default::default()).unwrap());
    }
}
