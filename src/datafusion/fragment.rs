//! Normalize a contiguous physical region without moving effects or partitions.
use std::{collections::BTreeSet, sync::Arc};

use ::datafusion::{
    common::{
        Result,
        tree_node::{Transformed, TreeNode},
    },
    physical_expr::{PhysicalExpr, expressions::Column},
    physical_plan::{
        ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, ReplaceChildrenOptions,
        execution_plan::Boundedness, filter::FilterExec, projection::ProjectionExec,
    },
};
use arrow_array::{RecordBatch, RecordBatchOptions};
use arrow_schema::{Schema as ArrowSchema, SchemaRef};

use super::{
    Unsupported,
    capability::{check_expression_limits, translate_named},
};
use crate::{
    Executor, Expr, ExprId, ExprNode, Field, KernelProgram, Parameters, Pipeline, Schema,
    arrow::ArrowBatch, prepare,
};

const MAX_REGION_NODES: usize = 64;
const MAX_FILTERS: usize = 16;
const MAX_OUTPUTS: usize = 32;

/// Inspect a candidate without device allocations. Normalized expressions use
/// source field positions; programs bind `cN` through their own local schemas.
#[derive(Clone, Debug)]
pub struct FragmentInspection {
    pub physical_nodes: usize,
    pub filter_count: usize,
    pub input_columns: Vec<usize>,
    pub normalized_expressions: Vec<String>,
    pub programs: Vec<KernelProgram>,
    pub boundary: Unsupported,
}

/// Explain the accepted prefix, or return a structured CPU-placement reason.
/// This runs the same translator as execution without uploading or launching.
pub fn inspect_fragment(
    plan: Arc<dyn ExecutionPlan>,
) -> std::result::Result<FragmentInspection, Unsupported> {
    let (region, fragment) = Region::discover(plan)?;
    let programs = fragment
        .roots
        .iter()
        .enumerate()
        .map(|(index, expr)| {
            translate_named(expr, &region.input.schema(), &format!("e{index}")).map(|t| t.program)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(FragmentInspection {
        physical_nodes: region.nodes.len(),
        filter_count: fragment.filter_count,
        input_columns: fragment.input_columns,
        normalized_expressions: fragment.roots.iter().map(ToString::to_string).collect(),
        programs,
        boundary: region.boundary,
    })
}

pub(crate) fn reject(code: &'static str, detail: impl Into<String>) -> Unsupported {
    Unsupported {
        code,
        detail: detail.into(),
    }
}

/// Original nodes, ordered root to leaf, retain DataFusion property semantics.
#[derive(Clone, Debug)]
pub(crate) struct Region {
    pub(crate) nodes: Vec<Arc<dyn ExecutionPlan>>,
    pub(crate) input: Arc<dyn ExecutionPlan>,
    pub(crate) boundary: Unsupported,
}
impl Region {
    pub(crate) fn root(&self) -> &Arc<dyn ExecutionPlan> {
        &self.nodes[0]
    }

    pub(crate) fn rebuild(
        &self,
        mut input: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        for node in self.nodes.iter().rev() {
            input = node.clone().replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?;
        }
        Ok(input)
    }

    /// Try longest prefixes first. A rejected lower node stays outside the region;
    /// no predicate/output is silently discarded when a shape limit is reached.
    pub(crate) fn discover(
        root: Arc<dyn ExecutionPlan>,
    ) -> std::result::Result<(Self, Fragment), Unsupported> {
        let mut nodes = Vec::new();
        let mut cursor = root;
        let boundary = loop {
            if nodes.len() == MAX_REGION_NODES {
                break reject("fragment_limit", "64 physical nodes per region");
            }
            if let Some(filter) = cursor.downcast_ref::<FilterExec>() {
                if filter.fetch().is_some() {
                    break reject(
                        "fetch_boundary",
                        "FilterExec fetch retains CPU evaluation boundary",
                    );
                }
                nodes.push(cursor.clone());
                cursor = filter.input().clone();
            } else if let Some(projection) = cursor.downcast_ref::<ProjectionExec>() {
                if !nodes.is_empty() && !projection.expr().iter().all(|e| e.expr.is::<Column>()) {
                    break reject(
                        "computed_projection",
                        "intermediate computations retain their evaluation boundary",
                    );
                }
                nodes.push(cursor.clone());
                cursor = projection.input().clone();
            } else {
                let code = match cursor.name() {
                    "RepartitionExec" | "CoalescePartitionsExec" | "SortPreservingMergeExec" => {
                        "partition_boundary"
                    }
                    "SortExec" | "GlobalLimitExec" | "LocalLimitExec" => "order_limit_boundary",
                    "CubeExec" => "existing_fragment",
                    _ => "cpu_boundary",
                };
                break reject(code, cursor.name());
            }
        };
        let mut failure = boundary.clone();
        for len in (1..=nodes.len()).rev() {
            let region = Self {
                nodes: nodes[..len].to_vec(),
                input: nodes[len - 1].children()[0].clone(),
                boundary: if len == nodes.len() {
                    boundary.clone()
                } else {
                    failure.clone()
                },
            };
            match Fragment::compile(&region) {
                Ok(fragment) => return Ok((region, fragment)),
                Err(reason) => failure = reason,
            }
        }
        Err(failure)
    }
}

/// Rebind physical column positions through pure projection/filter mappings.
/// Operations, overflow flags and operand order are retained by with_new_children.
fn remap(
    expr: &Arc<dyn PhysicalExpr>,
    positions: &[usize],
    local: &ArrowSchema,
    source: &ArrowSchema,
) -> std::result::Result<Arc<dyn PhysicalExpr>, Unsupported> {
    check_expression_limits(expr)?;
    let mapped = expr
        .clone()
        .transform_up(|node| {
            if let Some(column) = node.downcast_ref::<Column>() {
                let index = column.index();
                let old = local.fields().get(index).ok_or_else(|| {
                    ::datafusion::common::DataFusionError::Plan(
                        "column outside local schema".into(),
                    )
                })?;
                if old
                    .metadata()
                    .keys()
                    .any(|k| k.starts_with("ARROW:extension:"))
                {
                    return Err(::datafusion::common::DataFusionError::Plan(
                        "extension column semantics".into(),
                    ));
                }
                let index = *positions.get(index).ok_or_else(|| {
                    ::datafusion::common::DataFusionError::Plan(
                        "column outside projection map".into(),
                    )
                })?;
                let field = source.fields().get(index).ok_or_else(|| {
                    ::datafusion::common::DataFusionError::Plan(
                        "column outside source schema".into(),
                    )
                })?;
                return Ok(Transformed::yes(
                    Arc::new(Column::new(field.name(), index)) as Arc<dyn PhysicalExpr>
                ));
            }
            Ok(Transformed::no(node))
        })
        .map_err(|e| reject("binding", e.to_string()))?
        .data;
    if expr
        .data_type(local)
        .map_err(|e| reject("binding", e.to_string()))?
        != mapped
            .data_type(source)
            .map_err(|e| reject("binding", e.to_string()))?
        || expr
            .nullable(local)
            .map_err(|e| reject("binding", e.to_string()))?
            != mapped
                .nullable(source)
                .map_err(|e| reject("binding", e.to_string()))?
    {
        return Err(reject(
            "semantic_mismatch",
            "field remapping changed type/nullability",
        ));
    }
    Ok(mapped)
}

#[derive(Clone, Debug)]
pub(crate) struct Fragment {
    pipeline: Pipeline,
    parameters: Parameters,
    input_columns: Vec<usize>,
    input_program: KernelProgram,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    pub(crate) roots: Vec<Arc<dyn PhysicalExpr>>,
    pub(crate) filter_count: usize,
}
impl Fragment {
    fn compile(region: &Region) -> std::result::Result<Self, Unsupported> {
        if region.input.boundedness() != Boundedness::Bounded {
            return Err(reject("unbounded_input", "finite input required"));
        }
        let source = region.input.schema();
        let mut positions: Vec<_> = (0..source.fields().len()).collect();
        let mut predicates = Vec::new();
        let mut outputs = None;
        for (index, node) in region.nodes.iter().enumerate().rev() {
            let local = node.children()[0].schema();
            if let Some(filter) = node.downcast_ref::<FilterExec>() {
                predicates.push(remap(filter.predicate(), &positions, &local, &source)?);
                if let Some(projection) = filter.projection() {
                    positions = projection
                        .iter()
                        .map(|&i| {
                            positions
                                .get(i)
                                .copied()
                                .ok_or_else(|| reject("binding", "invalid filter projection index"))
                        })
                        .collect::<std::result::Result<_, _>>()?;
                }
            } else if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
                if index == 0 {
                    outputs = Some(
                        projection
                            .expr()
                            .iter()
                            .map(|e| remap(&e.expr, &positions, &local, &source))
                            .collect::<std::result::Result<Vec<_>, _>>()?,
                    );
                } else {
                    positions = projection
                        .expr()
                        .iter()
                        .map(|e| {
                            let column = e.expr.downcast_ref::<Column>().ok_or_else(|| {
                                reject("computed_projection", "only pure internal column mappings")
                            })?;
                            positions
                                .get(column.index())
                                .copied()
                                .ok_or_else(|| reject("binding", "invalid projection index"))
                        })
                        .collect::<std::result::Result<_, _>>()?;
                }
            }
        }
        let outputs = outputs.unwrap_or_else(|| {
            positions
                .iter()
                .map(|&i| Arc::new(Column::new(source.field(i).name(), i)) as Arc<dyn PhysicalExpr>)
                .collect()
        });
        if predicates.len() > MAX_FILTERS || outputs.is_empty() || outputs.len() > MAX_OUTPUTS {
            return Err(reject(
                "fragment_limit",
                "at most 16 filters and one to 32 outputs",
            ));
        }
        let filter_count = predicates.len();
        let roots: Vec<_> = predicates.into_iter().chain(outputs).collect();
        let output_schema = region.root().schema();
        let mut pipeline = Pipeline::new();
        let mut parameters = Parameters::new();
        let mut columns = BTreeSet::new();
        let mut first_program = None;
        for (index, expr) in roots.iter().enumerate() {
            let t = translate_named(expr, &source, &format!("e{index}"))?;
            if index >= filter_count {
                let field = &output_schema.fields()[index - filter_count];
                if field.data_type()
                    != &expr
                        .data_type(&source)
                        .map_err(|e| reject("binding", e.to_string()))?
                    || field.is_nullable() != t.program.output_nullable()
                    || field
                        .metadata()
                        .keys()
                        .any(|k| k.starts_with("ARROW:extension:"))
                {
                    return Err(reject(
                        "semantic_mismatch",
                        "SQL output field type/nullability/extension differs",
                    ));
                }
            }
            for (name, value) in t.literal_bindings() {
                parameters = parameters
                    .with(name, *value)
                    .map_err(|e| reject("binding", e.to_string()))?;
            }
            columns.extend(t.input_columns);
            first_program.get_or_insert_with(|| t.program.clone());
            if index < filter_count {
                pipeline = pipeline.filter(t.program);
            } else {
                pipeline = pipeline.project(format!("out{index}"), t.program);
            }
        }
        let input_columns: Vec<_> = columns.into_iter().collect();
        let mut fields = Vec::new();
        let mut own_fields = Vec::new();
        let mut nodes = Vec::new();
        for &index in &input_columns {
            let field = source.field(index).clone().with_name(format!("c{index}"));
            let ty = match field.data_type() {
                arrow_schema::DataType::Int32 => crate::DataType::I32,
                arrow_schema::DataType::Float32 => crate::DataType::F32,
                _ => return Err(reject("unsupported_type", "only numeric fragment inputs")),
            };
            own_fields.push(Field::new(field.name(), ty).with_nullable(field.is_nullable()));
            nodes.push(ExprNode::Column(field.name().clone()));
            fields.push(field);
        }
        let input_program = if nodes.is_empty() {
            first_program.expect("nonempty output")
        } else {
            prepare(
                &Expr::from_nodes(nodes, ExprId::new(0)),
                &Schema::from_fields(&own_fields, &[])
                    .map_err(|e| reject("binding", e.to_string()))?,
            )
            .map_err(|e| reject("binding", e.to_string()))?
        };
        Ok(Self {
            pipeline,
            parameters,
            input_columns,
            input_program,
            input_schema: Arc::new(ArrowSchema::new(fields)),
            output_schema: region.root().schema(),
            roots,
            filter_count,
        })
    }
    pub(crate) fn run(&self, executor: &Executor, input: &RecordBatch) -> Result<RecordBatch> {
        let error = |e| super::exec::execution(e);
        let columns = self
            .input_columns
            .iter()
            .map(|&i| input.column(i).clone())
            .collect();
        let renamed = RecordBatch::try_new_with_options(
            self.input_schema.clone(),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(input.num_rows())),
        )
        .map_err(error)?;
        let adapter =
            ArrowBatch::try_new(&self.input_program, &renamed).map_err(super::exec::execution)?;
        let host = adapter.as_batch().map_err(super::exec::execution)?;
        let result = executor
            .execute_pipeline_host(&self.pipeline, &host, &self.parameters)
            .map_err(super::exec::execution)?;
        let materialized = result
            .materialize_arrow()
            .map_err(super::exec::execution)?
            .projections
            .ok_or_else(|| super::exec::execution("missing projection"))?;
        RecordBatch::try_new(self.output_schema.clone(), materialized.columns().to_vec())
            .map_err(super::exec::execution)
    }
}
