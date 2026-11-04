use std::collections::BTreeMap;

use crate::{DataType, Error, InputKind};

/// Borrowed initialized host columns; this is not an Arrow or device buffer.
#[derive(Clone, Copy, Debug)]
pub enum ColumnData<'a> {
    F32(&'a [f32]),
    /// Dense 0/1 validity, one initialized payload per row (including NULLs).
    NullableF32 {
        values: &'a [f32],
        validity: &'a [u32],
    },
    I32(&'a [i32]),
}

impl ColumnData<'_> {
    pub fn data_type(self) -> DataType {
        match self {
            Self::F32(_) | Self::NullableF32 { .. } => DataType::F32,
            Self::I32(_) => DataType::I32,
        }
    }
    pub fn len(self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::NullableF32 { values, .. } => values.len(),
            Self::I32(v) => v.len(),
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// Explicit row count also permits parameter-only programs and empty batches.
#[derive(Clone, Debug)]
pub struct Batch<'a> {
    row_count: usize,
    columns: BTreeMap<String, ColumnData<'a>>,
}

impl<'a> Batch<'a> {
    pub fn new(row_count: usize) -> Self {
        Self {
            row_count,
            columns: BTreeMap::new(),
        }
    }
    pub fn row_count(&self) -> usize {
        self.row_count
    }
    pub(crate) fn column(&self, name: &str) -> Option<ColumnData<'a>> {
        self.columns.get(name).copied()
    }
    pub fn with_column(
        mut self,
        name: impl Into<String>,
        data: ColumnData<'a>,
    ) -> Result<Self, Error> {
        let name = name.into();
        validate_new_name(&self.columns, &name, InputKind::Column)?;
        if data.len() != self.row_count {
            return Err(Error::ColumnLengthMismatch {
                name,
                expected: self.row_count,
                actual: data.len(),
            });
        }
        if let ColumnData::NullableF32 { values, validity } = data {
            if validity.len() != values.len() {
                return Err(Error::ValidityLengthMismatch {
                    name,
                    expected: values.len(),
                    actual: validity.len(),
                });
            }
            if let Some((index, &value)) =
                validity.iter().enumerate().find(|(_, value)| **value > 1)
            {
                return Err(Error::InvalidValidity { name, index, value });
            }
        }
        self.columns.insert(name, data);
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ScalarValue {
    F32(f32),
    NullableF32(Option<f32>),
    I32(i32),
}

impl ScalarValue {
    pub fn data_type(self) -> DataType {
        match self {
            Self::F32(_) | Self::NullableF32(_) => DataType::F32,
            Self::I32(_) => DataType::I32,
        }
    }
}

impl From<f32> for ScalarValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}
impl From<Option<f32>> for ScalarValue {
    fn from(value: Option<f32>) -> Self {
        Self::NullableF32(value)
    }
}
impl From<i32> for ScalarValue {
    fn from(value: i32) -> Self {
        Self::I32(value)
    }
}

/// Named host scalars. Construct a new set to rebind a prepared program.
#[derive(Clone, Debug, Default)]
pub struct Parameters {
    values: BTreeMap<String, ScalarValue>,
}

impl Parameters {
    pub fn new() -> Self {
        Self::default()
    }
    pub(crate) fn value(&self, name: &str) -> Option<ScalarValue> {
        self.values.get(name).copied()
    }
    pub fn with(
        mut self,
        name: impl Into<String>,
        value: impl Into<ScalarValue>,
    ) -> Result<Self, Error> {
        let name = name.into();
        validate_new_name(&self.values, &name, InputKind::Parameter)?;
        self.values.insert(name, value.into());
        Ok(self)
    }
}

fn validate_new_name<V>(
    values: &BTreeMap<String, V>,
    name: &str,
    kind: InputKind,
) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::EmptyName { kind });
    }
    if values.contains_key(name) {
        return Err(Error::DuplicateName {
            kind,
            name: name.into(),
        });
    }
    Ok(())
}
