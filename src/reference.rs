use std::collections::BTreeMap;

use crate::{DataType, Error, InputKind, KernelProgram, ScalarOp};

/// Borrowed non-nullable host columns; this is not an Arrow or device buffer.
#[derive(Clone, Copy, Debug)]
pub enum ColumnData<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
}

impl ColumnData<'_> {
    pub fn data_type(self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::I32(_) => DataType::I32,
        }
    }
    pub fn len(self) -> usize {
        match self {
            Self::F32(v) => v.len(),
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
        self.columns.insert(name, data);
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ScalarValue {
    F32(f32),
    I32(i32),
}

impl ScalarValue {
    pub fn data_type(self) -> DataType {
        match self {
            Self::F32(_) => DataType::F32,
            Self::I32(_) => DataType::I32,
        }
    }
}

impl From<f32> for ScalarValue {
    fn from(value: f32) -> Self {
        Self::F32(value)
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

impl KernelProgram {
    /// Evaluate each validated node in arena order for each host row.
    ///
    /// All node inputs are validated before producing output, even for zero rows.
    /// Extra batch columns and scalar bindings are ignored. Input binding order
    /// does not matter. Arithmetic uses Rust f32 add/multiply without explicitly
    /// fusing multiply-add or reassociating the expression. SQL NaN/overflow rules
    /// are not defined by this reference API.
    pub fn evaluate_reference(
        &self,
        batch: &Batch<'_>,
        parameters: &Parameters,
    ) -> Result<Vec<f32>, Error> {
        let schema = self.schema();
        let mut columns = vec![None; schema.columns().len()];
        let mut scalars = vec![None; schema.parameters().len()];
        for node in self.nodes() {
            match node.op() {
                ScalarOp::Column(id) if columns[id.index()].is_none() => {
                    let name = schema.columns()[id.index()].name();
                    let kind = InputKind::Column;
                    let data = batch.columns.get(name).ok_or_else(|| Error::MissingInput {
                        kind,
                        name: name.into(),
                    })?;
                    let ColumnData::F32(values) = data else {
                        return Err(Error::TypeMismatch {
                            kind,
                            name: name.into(),
                            expected: DataType::F32,
                            actual: data.data_type(),
                        });
                    };
                    columns[id.index()] = Some(*values);
                }
                ScalarOp::Parameter(id) if scalars[id.index()].is_none() => {
                    let name = schema.parameters()[id.index()].name();
                    let kind = InputKind::Parameter;
                    let value = parameters
                        .values
                        .get(name)
                        .ok_or_else(|| Error::MissingInput {
                            kind,
                            name: name.into(),
                        })?;
                    let ScalarValue::F32(value) = value else {
                        return Err(Error::TypeMismatch {
                            kind,
                            name: name.into(),
                            expected: DataType::F32,
                            actual: value.data_type(),
                        });
                    };
                    scalars[id.index()] = Some(*value);
                }
                _ => {}
            }
        }
        let mut output = Vec::with_capacity(batch.row_count);
        let mut values = vec![0.0_f32; self.nodes().len()];
        for row in 0..batch.row_count {
            for (index, node) in self.nodes().iter().enumerate() {
                values[index] = match node.op() {
                    ScalarOp::Column(id) => columns[id.index()].expect("validated column")[row],
                    ScalarOp::Parameter(id) => scalars[id.index()].expect("validated parameter"),
                    ScalarOp::Add(a, b) => values[a.index()] + values[b.index()],
                    ScalarOp::Mul(a, b) => values[a.index()] * values[b.index()],
                };
            }
            output.push(values[self.root().index()]);
        }
        Ok(output)
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
