use std::{error::Error as StdError, fmt};

use crate::{DataType, ExprId};

/// Which input namespace a name belongs to. Columns and parameters are separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Column,
    Parameter,
}

impl fmt::Display for InputKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Column => "column",
            Self::Parameter => "parameter",
        })
    }
}

/// Errors from expression preparation and host reference evaluation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    EmptyName {
        kind: InputKind,
    },
    DuplicateName {
        kind: InputKind,
        name: String,
    },
    UnknownName {
        kind: InputKind,
        name: String,
    },
    UnsupportedType {
        kind: InputKind,
        name: String,
        data_type: DataType,
    },
    EmptyExpression,
    InvalidRoot {
        root: ExprId,
        node_count: usize,
    },
    InvalidReference {
        node: ExprId,
        operand: ExprId,
    },
    ExpressionTooLarge {
        node_count: usize,
        limit: usize,
    },
    ExpressionTooDeep {
        node: ExprId,
        depth: usize,
        limit: usize,
    },
    MissingInput {
        kind: InputKind,
        name: String,
    },
    TypeMismatch {
        kind: InputKind,
        name: String,
        expected: DataType,
        actual: DataType,
    },
    ColumnLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyName { kind } => write!(f, "{kind} name must not be empty"),
            Self::DuplicateName { kind, name } => write!(f, "duplicate {kind} {name:?}"),
            Self::UnknownName { kind, name } => write!(f, "unknown {kind} {name:?} in schema"),
            Self::UnsupportedType {
                kind,
                name,
                data_type,
            } => write!(
                f,
                "{kind} {name:?} has unsupported type {data_type}; expected f32"
            ),
            Self::EmptyExpression => f.write_str("expression has no nodes"),
            Self::InvalidRoot { root, node_count } => write!(
                f,
                "root {} is outside {node_count} expression nodes",
                root.index()
            ),
            Self::InvalidReference { node, operand } => write!(
                f,
                "node {} references {}; operands must refer to earlier nodes",
                node.index(),
                operand.index()
            ),
            Self::ExpressionTooLarge { node_count, limit } => {
                write!(f, "expression has {node_count} nodes; limit is {limit}")
            }
            Self::ExpressionTooDeep { node, depth, limit } => write!(
                f,
                "node {} has depth {depth}; limit is {limit}",
                node.index()
            ),
            Self::MissingInput { kind, name } => write!(f, "missing {kind} {name:?}"),
            Self::TypeMismatch {
                kind,
                name,
                expected,
                actual,
            } => write!(f, "{kind} {name:?} has type {actual}; expected {expected}"),
            Self::ColumnLengthMismatch {
                name,
                expected,
                actual,
            } => write!(f, "column {name:?} has {actual} rows; batch has {expected}"),
        }
    }
}

impl StdError for Error {}
