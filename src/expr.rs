use std::{
    ops::{Add, Mul},
    sync::Arc,
};

use crate::{ColumnId, DataType, Error, InputKind, ParameterId, Schema};

/// An expression-local node index. Raw indexes are validated by prepare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExprId(usize);

impl ExprId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }
    pub const fn index(self) -> usize {
        self.0
    }
}

/// An unbound node. Binary operands must precede their parent in the arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprNode {
    Column(String),
    Parameter(String),
    Add(ExprId, ExprId),
    Mul(ExprId, ExprId),
}

impl ExprNode {
    fn rebased(self, offset: usize) -> Self {
        // Keep malformed raw indexes invalid instead of overflowing during DSL composition.
        let rebase = |id: ExprId| ExprId(id.0.saturating_add(offset));
        match self {
            Self::Add(a, b) => Self::Add(rebase(a), rebase(b)),
            Self::Mul(a, b) => Self::Mul(rebase(a), rebase(b)),
            leaf => leaf,
        }
    }
}

/// An owned, flat expression arena. Construction and drop are non-recursive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expr {
    nodes: Vec<ExprNode>,
    root: ExprId,
}

impl Expr {
    /// Construct a raw graph for programmatic frontends. All nodes, including
    /// unreachable nodes, are checked during preparation. Forward references and
    /// cycles are rejected; operands must refer to earlier nodes.
    pub fn from_nodes(nodes: Vec<ExprNode>, root: ExprId) -> Self {
        Self { nodes, root }
    }
    pub fn nodes(&self) -> &[ExprNode] {
        &self.nodes
    }
    pub fn root(&self) -> ExprId {
        self.root
    }

    fn combine(mut self, rhs: Self, op: fn(ExprId, ExprId) -> ExprNode) -> Self {
        let offset = self.nodes.len();
        let invalid_root = self.root.0 >= offset || rhs.root.0 >= rhs.nodes.len();
        let rhs_root = ExprId(rhs.root.0.saturating_add(offset));
        self.nodes
            .extend(rhs.nodes.into_iter().map(|node| node.rebased(offset)));
        let root = ExprId(self.nodes.len());
        self.nodes.push(op(self.root, rhs_root));
        // Appending another arena must not accidentally make a bad raw root valid.
        self.root = if invalid_root {
            ExprId(usize::MAX)
        } else {
            root
        };
        self
    }
}

pub fn col(name: impl Into<String>) -> Expr {
    Expr::from_nodes(vec![ExprNode::Column(name.into())], ExprId(0))
}

pub fn param(name: impl Into<String>) -> Expr {
    Expr::from_nodes(vec![ExprNode::Parameter(name.into())], ExprId(0))
}

impl Add for Expr {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        self.combine(rhs, ExprNode::Add)
    }
}

impl Mul for Expr {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.combine(rhs, ExprNode::Mul)
    }
}

/// Bounds checked before allocating the prepared arena. Leaf depth is one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrepareLimits {
    pub max_nodes: usize,
    pub max_depth: usize,
}

impl Default for PrepareLimits {
    fn default() -> Self {
        Self {
            max_nodes: 4096,
            max_depth: 128,
        }
    }
}

/// Bound operations. These indexes refer to this program's schema and arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    Column(ColumnId),
    Parameter(ParameterId),
    Add(ExprId, ExprId),
    Mul(ExprId, ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypedExpr {
    op: ScalarOp,
    data_type: DataType,
}

impl TypedExpr {
    pub fn op(&self) -> ScalarOp {
        self.op
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

/// Immutable, bound arithmetic program, independent of CubeCL contexts/devices.
/// Clones share owned data. IDs are local to this program, not cross-program handles.
#[derive(Clone, Debug)]
pub struct KernelProgram {
    schema: Arc<Schema>,
    nodes: Arc<[TypedExpr]>,
    root: ExprId,
}

impl KernelProgram {
    pub fn schema(&self) -> &Schema {
        &self.schema
    }
    pub fn nodes(&self) -> &[TypedExpr] {
        &self.nodes
    }
    pub fn root(&self) -> ExprId {
        self.root
    }
    pub fn output_type(&self) -> DataType {
        self.nodes[self.root.0].data_type
    }
}

pub fn prepare(expr: &Expr, schema: &Schema) -> Result<KernelProgram, Error> {
    prepare_with_limits(expr, schema, PrepareLimits::default())
}

/// Validate the entire topologically ordered arena and resolve names to IDs.
/// Only referenced input declarations must be F32. No reassociation, constant
/// folding or device compilation is performed.
pub fn prepare_with_limits(
    expr: &Expr,
    schema: &Schema,
    limits: PrepareLimits,
) -> Result<KernelProgram, Error> {
    let node_count = expr.nodes.len();
    if node_count == 0 {
        return Err(Error::EmptyExpression);
    }
    if node_count > limits.max_nodes {
        return Err(Error::ExpressionTooLarge {
            node_count,
            limit: limits.max_nodes,
        });
    }
    if expr.root.0 >= node_count {
        return Err(Error::InvalidRoot {
            root: expr.root,
            node_count,
        });
    }
    let mut nodes = Vec::with_capacity(node_count);
    let mut depths = Vec::<usize>::with_capacity(node_count);
    for (index, node) in expr.nodes.iter().enumerate() {
        let node_id = ExprId(index);
        let (op, depth) = match node {
            ExprNode::Column(name) => {
                let kind = InputKind::Column;
                let id = schema.column_id(name).ok_or_else(|| Error::UnknownName {
                    kind,
                    name: name.clone(),
                })?;
                require_f32(kind, name, schema.columns()[id.index()].data_type())?;
                (ScalarOp::Column(id), 1)
            }
            ExprNode::Parameter(name) => {
                let kind = InputKind::Parameter;
                let id = schema
                    .parameter_id(name)
                    .ok_or_else(|| Error::UnknownName {
                        kind,
                        name: name.clone(),
                    })?;
                require_f32(kind, name, schema.parameters()[id.index()].data_type())?;
                (ScalarOp::Parameter(id), 1)
            }
            ExprNode::Add(a, b) | ExprNode::Mul(a, b) => {
                for &operand in [a, b] {
                    if operand.0 >= index {
                        return Err(Error::InvalidReference {
                            node: node_id,
                            operand,
                        });
                    }
                }
                let depth = depths[a.0].max(depths[b.0]) + 1;
                let op = match node {
                    ExprNode::Add(..) => ScalarOp::Add(*a, *b),
                    _ => ScalarOp::Mul(*a, *b),
                };
                (op, depth)
            }
        };
        if depth > limits.max_depth {
            return Err(Error::ExpressionTooDeep {
                node: node_id,
                depth,
                limit: limits.max_depth,
            });
        }
        nodes.push(TypedExpr {
            op,
            data_type: DataType::F32,
        });
        depths.push(depth);
    }
    Ok(KernelProgram {
        schema: Arc::new(schema.clone()),
        nodes: nodes.into(),
        root: expr.root,
    })
}

fn require_f32(kind: InputKind, name: &str, data_type: DataType) -> Result<(), Error> {
    if data_type != DataType::F32 {
        return Err(Error::UnsupportedType {
            kind,
            name: name.into(),
            data_type,
        });
    }
    Ok(())
}
