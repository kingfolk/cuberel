use std::{collections::BTreeSet, fmt};

use crate::{Error, InputKind};

/// Logical input types. Only F32 is supported by prepared arithmetic programs.
/// I32 can describe an input so unsupported use or mismatched bindings are reported.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    I32,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::F32 => "f32",
            Self::I32 => "i32",
        })
    }
}

/// Position of a column in a program's schema, not a global column identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColumnId(pub(crate) usize);

impl ColumnId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// Position of a parameter in a program's schema, not a global parameter identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParameterId(pub(crate) usize);

impl ParameterId {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    name: String,
    data_type: DataType,
}

impl Field {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

/// Owned declarations for the column and scalar-parameter namespaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Schema {
    columns: Vec<Field>,
    parameters: Vec<Field>,
}

impl Schema {
    pub fn new(
        columns: &[(&str, DataType)],
        parameters: &[(&str, DataType)],
    ) -> Result<Self, Error> {
        Ok(Self {
            columns: fields(columns, InputKind::Column)?,
            parameters: fields(parameters, InputKind::Parameter)?,
        })
    }

    pub fn columns(&self) -> &[Field] {
        &self.columns
    }
    pub fn parameters(&self) -> &[Field] {
        &self.parameters
    }
    pub fn column_id(&self, name: &str) -> Option<ColumnId> {
        self.columns
            .iter()
            .position(|f| f.name == name)
            .map(ColumnId)
    }
    pub fn parameter_id(&self, name: &str) -> Option<ParameterId> {
        self.parameters
            .iter()
            .position(|f| f.name == name)
            .map(ParameterId)
    }
}

fn fields(declarations: &[(&str, DataType)], kind: InputKind) -> Result<Vec<Field>, Error> {
    let mut names = BTreeSet::new();
    let mut fields = Vec::with_capacity(declarations.len());
    for &(name, data_type) in declarations {
        if name.is_empty() {
            return Err(Error::EmptyName { kind });
        }
        if !names.insert(name) {
            return Err(Error::DuplicateName {
                kind,
                name: name.into(),
            });
        }
        fields.push(Field {
            name: name.into(),
            data_type,
        });
    }
    Ok(fields)
}
