//! Explicit source granularity and a hard adapter request-row ceiling.
use std::{error::Error, fmt};

/// An explicit session configuration, not an internal coalescing/splitting rule.
///
/// `source_rows` configures DataFusion's session batch size. It is a target for
/// participating sources, not a promise about every physical operator's output.
/// Every batch entering a device worker is checked against `max_request_rows`
/// before host workspace reservation or upload. Byte/stage limits still apply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPolicy {
    source_rows: usize,
    max_request_rows: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchPolicyError {
    InvalidSizes {
        source_rows: usize,
        max_request_rows: usize,
    },
    RequestTooLarge {
        rows: usize,
        max_request_rows: usize,
    },
}
impl fmt::Display for BatchPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSizes {
                source_rows,
                max_request_rows,
            } => write!(
                f,
                "batch policy requires 0 < source_rows ({source_rows}) <= max_request_rows ({max_request_rows}) <= u32::MAX"
            ),
            Self::RequestTooLarge {
                rows,
                max_request_rows,
            } => write!(
                f,
                "batch has {rows} rows, exceeding explicit max_request_rows {max_request_rows}; no automatic split or CPU retry"
            ),
        }
    }
}
impl Error for BatchPolicyError {}
impl BatchPolicy {
    pub fn new(source_rows: usize, max_request_rows: usize) -> Result<Self, BatchPolicyError> {
        if source_rows == 0
            || source_rows > max_request_rows
            || max_request_rows > u32::MAX as usize
        {
            return Err(BatchPolicyError::InvalidSizes {
                source_rows,
                max_request_rows,
            });
        }
        Ok(Self {
            source_rows,
            max_request_rows,
        })
    }
    pub fn source_rows(self) -> usize {
        self.source_rows
    }
    pub fn max_request_rows(self) -> usize {
        self.max_request_rows
    }
    pub(crate) fn check(self, rows: usize) -> Result<(), BatchPolicyError> {
        if rows > self.max_request_rows {
            Err(BatchPolicyError::RequestTooLarge {
                rows,
                max_request_rows: self.max_request_rows,
            })
        } else {
            Ok(())
        }
    }
}
