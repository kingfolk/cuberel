//! One blocking boundary and per-exit count memoization. No persistent success cache.
use crate::{DeviceSelection, ExecutionError};
use cubecl::{future::block_on, prelude::Client, server::Handle};
#[derive(Default)]
pub(crate) struct ExitContext {
    pub(crate) counts: Vec<(usize, u32)>,
}
impl ExitContext {
    pub(crate) fn count(&mut self, selection: &DeviceSelection) -> Result<u32, ExecutionError> {
        selection.count_in(self)
    }
}
pub(crate) fn synchronize(client: &Client, handles: &[Handle]) -> Result<(), ExecutionError> {
    if !handles.is_empty() {
        block_on(client.sync_buffers(handles.iter())).map_err(ExecutionError::Backend)?;
    }
    Ok(())
}
pub(crate) fn check_all(
    checks: impl IntoIterator<Item = Result<(), ExecutionError>>,
) -> Result<(), ExecutionError> {
    let mut first = None;
    for result in checks {
        match result {
            Err(ExecutionError::IntegerOverflow { row }) => {
                first = Some(first.map_or(row, |old: usize| old.min(row)))
            }
            Err(error) => return Err(error),
            Ok(()) => {}
        }
    }
    if let Some(row) = first {
        Err(ExecutionError::IntegerOverflow { row })
    } else {
        Ok(())
    }
}
