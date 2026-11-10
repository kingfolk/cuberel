//! Per-request logical bounds, independent of runtime caches and other requests.
use crate::{
    Aggregate, DeviceBatch, DeviceSelection, ExecutionError, Executor, KernelProgram,
    SelectionStats,
    abi::{BindingLayout, BufferAccess, BufferSource},
    aggregate::ReductionPlan,
};
use std::{collections::HashSet, error::Error, fmt};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetResource {
    DeviceBytes,
    OutputRows,
    Stages,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    Exceeded {
        resource: BudgetResource,
        required: u64,
        limit: u64,
    },
    Overflow {
        component: &'static str,
    },
}
impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exceeded {
                resource,
                required,
                limit,
            } => write!(
                f,
                "request {resource:?} needs {required}, budget is {limit}"
            ),
            Self::Overflow { component } => write!(f, "resource estimate overflow: {component}"),
        }
    }
}
impl Error for BudgetError {}
impl From<BudgetError> for ExecutionError {
    fn from(value: BudgetError) -> Self {
        Self::Budget(value)
    }
}

/// Immutable per-executor request limits. No global reservation or OOM guarantee.
/// Default preserves the existing unlimited policy and fixed pipeline shape caps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub max_device_bytes: u64,
    /// Maximum logical rows per output column (aggregate-only output is one row).
    pub max_output_rows: u64,
    /// Physical kernel stages plus logical source/exit boundaries. Upload alone is one.
    pub max_stages: u64,
}
impl ResourceBudget {
    pub const UNLIMITED: Self = Self {
        max_device_bytes: u64::MAX,
        max_output_rows: u64::MAX,
        max_stages: u64::MAX,
    };
    pub fn check(&self, estimate: &ResourceEstimate) -> Result<(), BudgetError> {
        for (resource, required, limit) in [
            (
                BudgetResource::DeviceBytes,
                estimate.device_bytes,
                self.max_device_bytes,
            ),
            (
                BudgetResource::OutputRows,
                estimate.output_rows,
                self.max_output_rows,
            ),
            (BudgetResource::Stages, estimate.stages, self.max_stages),
        ] {
            if required > limit {
                return Err(BudgetError::Exceeded {
                    resource,
                    required,
                    limit,
                });
            }
        }
        Ok(())
    }
}
impl Default for ResourceBudget {
    fn default() -> Self {
        Self::UNLIMITED
    }
}

/// Conservative logical device-byte bound: existing inputs plus every newly
/// requested allocation, with no credit for asynchronous temporary retirement.
/// Category bytes exclude driver padding, metadata, staging, shared memory/cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceEstimate {
    pub input_bytes: u64,
    /// Expression value planes, including internal masks and aggregate arguments.
    pub value_bytes: u64,
    pub validity_bytes: u64,
    pub error_bytes: u64,
    pub indices_bytes: u64,
    /// Scan ranks and totals, including final count words.
    pub scan_bytes: u64,
    pub partial_bytes: u64,
    pub device_bytes: u64,
    /// Logical upper bound after private selection scratch/ancestor indices retire.
    /// Includes supplied inputs; it is not a driver/global physical-memory peak.
    pub retained_bytes: u64,
    pub output_rows: u64,
    pub stages: u64,
}
impl ResourceEstimate {
    pub fn new_allocation_bytes(&self) -> u64 {
        self.device_bytes - self.input_bytes
    }
}
pub(crate) fn add(a: u64, b: u64, component: &'static str) -> Result<u64, BudgetError> {
    a.checked_add(b).ok_or(BudgetError::Overflow { component })
}
pub(crate) fn mul(a: u64, b: u64, component: &'static str) -> Result<u64, BudgetError> {
    a.checked_mul(b).ok_or(BudgetError::Overflow { component })
}
pub(crate) fn upload_bytes(rows: u64, columns: usize) -> Result<u64, BudgetError> {
    mul(
        mul(rows, 8, "upload column planes")?,
        columns as u64,
        "upload columns",
    )
}
pub(crate) fn inputs(
    batch: &DeviceBatch,
    selection: Option<&DeviceSelection>,
) -> Result<u64, BudgetError> {
    let mut bytes = batch.logical_bytes()?;
    if let Some(selection) = selection {
        let mut seen: HashSet<_> = batch
            .handles()
            .iter()
            .map(|h| h.memory.clone().binding().id())
            .collect();
        for handle in selection.handles() {
            if seen.insert(handle.memory.clone().binding().id()) {
                bytes = add(bytes, handle.size_in_used(), "shared selection inputs")?;
            }
        }
    }
    Ok(bytes)
}

pub(crate) struct Account {
    pub(crate) e: ResourceEstimate,
    rows: u64,
    private_indices: u64,
}
impl Account {
    pub(crate) fn new(rows: u64, input_bytes: u64, output_rows: u64) -> Self {
        Self {
            e: ResourceEstimate {
                input_bytes,
                retained_bytes: input_bytes,
                output_rows,
                stages: 2,
                ..ResourceEstimate::default()
            },
            rows,
            private_indices: 0,
        }
    }
    pub(crate) fn expression(
        &mut self,
        program: &KernelProgram,
        retain: bool,
    ) -> Result<(), BudgetError> {
        let bytes = mul(self.rows, 4, "expression plane")?;
        let mut kept = 0;
        for slot in BindingLayout::new(program)
            .buffers()
            .iter()
            .filter(|s| s.access() == BufferAccess::WriteOnly)
        {
            let field = match slot.source() {
                BufferSource::Output => &mut self.e.value_bytes,
                BufferSource::OutputValidity => &mut self.e.validity_bytes,
                BufferSource::OutputError => &mut self.e.error_bytes,
                _ => unreachable!("private output slot"),
            };
            *field = add(*field, bytes, "expression output planes")?;
            if retain || slot.source() == BufferSource::OutputError {
                kept = add(kept, bytes, "retained expression planes")?;
            }
        }
        self.e.retained_bytes = add(self.e.retained_bytes, kept, "retained expression")?;
        self.stages(u64::from(self.rows != 0))
    }
    pub(crate) fn selection(&mut self, stats: SelectionStats) -> Result<(), BudgetError> {
        self.e.indices_bytes = add(
            self.e.indices_bytes,
            stats.indices_capacity_bytes,
            "selection indices",
        )?;
        self.e.scan_bytes = add(
            self.e.scan_bytes,
            stats.workspace_bytes,
            "selection workspace",
        )?;
        // All selection stages in a pipeline are private. Supplied external
        // selections are already in input_bytes and cannot be credited as freed.
        self.e.retained_bytes -= self.private_indices;
        self.private_indices = stats.indices_capacity_bytes;
        self.e.retained_bytes = add(
            self.e.retained_bytes,
            self.private_indices,
            "retained indices",
        )?;
        if self.rows != 0 {
            self.e.retained_bytes = add(self.e.retained_bytes, 4, "retained count")?;
        }
        self.stages(stats.kernel_launches as u64)
    }
    pub(crate) fn partials(&mut self, bytes: u64, launches: u64) -> Result<(), BudgetError> {
        self.e.partial_bytes = add(self.e.partial_bytes, bytes, "partial states")?;
        self.e.retained_bytes = add(self.e.retained_bytes, bytes, "retained partial states")?;
        self.stages(launches)
    }
    pub(crate) fn errors(&mut self, planes: u64) -> Result<(), BudgetError> {
        let bytes = mul(
            mul(self.rows, 4, "fused error plane")?,
            planes,
            "fused error planes",
        )?;
        self.e.error_bytes = add(self.e.error_bytes, bytes, "fused error planes")?;
        self.e.retained_bytes = add(self.e.retained_bytes, bytes, "retained errors")?;
        Ok(())
    }
    pub(crate) fn stages(&mut self, stages: u64) -> Result<(), BudgetError> {
        self.e.stages = add(self.e.stages, stages, "stage count")?;
        Ok(())
    }
    pub(crate) fn aggregates(
        &mut self,
        requests: &[(&str, Aggregate)],
        plan: &ReductionPlan,
    ) -> Result<(), BudgetError> {
        for (_, aggregate) in requests {
            if let Some(program) = aggregate.program() {
                self.expression(program, true)?;
            }
        }
        self.partials(
            plan.stats.partial_state_bytes,
            plan.stats.reduction_launches as u64,
        )
    }
    pub(crate) fn finish(mut self) -> Result<ResourceEstimate, BudgetError> {
        self.e.device_bytes = self.e.input_bytes;
        for bytes in [
            self.e.value_bytes,
            self.e.validity_bytes,
            self.e.error_bytes,
            self.e.indices_bytes,
            self.e.scan_bytes,
            self.e.partial_bytes,
        ] {
            self.e.device_bytes = add(self.e.device_bytes, bytes, "total device bytes")?;
        }
        debug_assert!(self.e.retained_bytes <= self.e.device_bytes);
        Ok(self.e)
    }
}
impl Executor {
    /// Set per-request limits. Clones share counters and keep independent budgets.
    pub fn with_budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }
    pub fn budget(&self) -> ResourceBudget {
        self.budget
    }
    pub(crate) fn without_budget(&self) -> Self {
        self.clone().with_budget(ResourceBudget::UNLIMITED)
    }
    pub(crate) fn enforce(&self, estimate: ResourceEstimate) -> Result<(), ExecutionError> {
        self.budget.check(&estimate)?;
        Ok(())
    }
    pub(crate) fn expression_estimate(
        &self,
        batch: &DeviceBatch,
        selection: Option<&DeviceSelection>,
        programs: &[&KernelProgram],
    ) -> Result<ResourceEstimate, ExecutionError> {
        let mut a = Account::new(
            u64::from(batch.row_count()),
            inputs(batch, selection)?,
            if programs.is_empty() {
                0
            } else {
                u64::from(batch.row_count())
            },
        );
        for program in programs {
            a.expression(program, true)?;
        }
        Ok(a.finish()?)
    }
    pub(crate) fn select_estimate(
        &self,
        rows: u32,
        input: u64,
        program: Option<&KernelProgram>,
        stats: SelectionStats,
    ) -> Result<ResourceEstimate, ExecutionError> {
        let mut a = Account::new(u64::from(rows), input, u64::from(rows));
        if let Some(program) = program {
            a.expression(program, false)?;
        } else {
            a.e.retained_bytes = 0;
        } // uploaded host mask retires after scatter
        a.selection(stats)?;
        Ok(a.finish()?)
    }
}
#[cfg(test)]
mod tests {
    use super::{Account, BudgetError, BudgetResource, ResourceBudget, mul};
    #[test]
    fn checked_arithmetic_and_inclusive_limits() {
        assert!(matches!(
            mul(u64::MAX, 4, "test"),
            Err(BudgetError::Overflow { .. })
        ));
        assert!(Account::new(0, u64::MAX, 0).finish().is_ok());
        let mut a = Account::new(0, u64::MAX, 0);
        a.e.value_bytes = 1;
        assert!(matches!(a.finish(), Err(BudgetError::Overflow { .. })));
        let e = Account::new(0, 16, 1).finish().unwrap();
        let b = ResourceBudget {
            max_device_bytes: 16,
            max_output_rows: 1,
            max_stages: 2,
        };
        assert!(b.check(&e).is_ok());
        assert!(matches!(
            ResourceBudget { max_stages: 1, ..b }.check(&e),
            Err(BudgetError::Exceeded {
                resource: BudgetResource::Stages,
                required: 2,
                limit: 1
            })
        ));
    }
}
