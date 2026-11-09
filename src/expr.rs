use std::{
    ops::{Add, Mul, Not},
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

/// Same-type numeric comparisons. F32 uses IEEE semantics: unordered comparisons are false except
/// NotEqual, which is true. NaN is a floating-point value, not SQL NULL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparison {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

/// An unbound node. Binary operands must precede their parent in the arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExprNode {
    Column(String),
    Parameter(String),
    Add(ExprId, ExprId),
    Mul(ExprId, ExprId),
    Compare(Comparison, ExprId, ExprId),
    And(ExprId, ExprId),
    Or(ExprId, ExprId),
    Not(ExprId),
    IsNull(ExprId),
}

impl ExprNode {
    fn rebased(self, offset: usize) -> Self {
        // Keep malformed raw indexes invalid instead of overflowing during DSL composition.
        let rebase = |id: ExprId| ExprId(id.0.saturating_add(offset));
        match self {
            Self::Add(a, b) => Self::Add(rebase(a), rebase(b)),
            Self::Mul(a, b) => Self::Mul(rebase(a), rebase(b)),
            Self::Compare(op, a, b) => Self::Compare(op, rebase(a), rebase(b)),
            Self::And(a, b) => Self::And(rebase(a), rebase(b)),
            Self::Or(a, b) => Self::Or(rebase(a), rebase(b)),
            Self::Not(a) => Self::Not(rebase(a)),
            Self::IsNull(a) => Self::IsNull(rebase(a)),
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

    /// Compare same-type F32 or I32 expressions. Use eq_expr/ne_expr to distinguish query
    /// construction from Rust's structural PartialEq implementation.
    pub fn eq_expr(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::Equal)
    }
    pub fn ne_expr(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::NotEqual)
    }
    pub fn lt(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::LessThan)
    }
    pub fn le(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::LessThanOrEqual)
    }
    pub fn gt(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::GreaterThan)
    }
    pub fn ge(self, rhs: Self) -> Self {
        self.compare(rhs, Comparison::GreaterThanOrEqual)
    }
    pub fn and(self, rhs: Self) -> Self {
        self.combine(rhs, ExprNode::And)
    }
    pub fn or(self, rhs: Self) -> Self {
        self.combine(rhs, ExprNode::Or)
    }

    /// Construct a non-nullable Bool expression testing numeric or Bool validity.
    pub fn is_null(self) -> Self {
        self.unary(ExprNode::IsNull)
    }

    fn unary(mut self, op: fn(ExprId) -> ExprNode) -> Self {
        let invalid_root = self.root.index() >= self.nodes.len();
        let root = ExprId(self.nodes.len());
        self.nodes.push(op(self.root));
        self.root = if invalid_root {
            ExprId(usize::MAX)
        } else {
            root
        };
        self
    }

    fn compare(self, rhs: Self, comparison: Comparison) -> Self {
        self.combine(rhs, |a, b| ExprNode::Compare(comparison, a, b))
    }

    fn combine(mut self, rhs: Self, op: impl FnOnce(ExprId, ExprId) -> ExprNode) -> Self {
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

impl Not for Expr {
    type Output = Self;
    fn not(self) -> Self {
        self.unary(ExprNode::Not)
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
    Compare(Comparison, ExprId, ExprId),
    And(ExprId, ExprId),
    Or(ExprId, ExprId),
    Not(ExprId),
    IsNull(ExprId),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TypedExpr {
    op: ScalarOp,
    data_type: DataType,
    nullable: bool,
}

impl TypedExpr {
    pub fn is_nullable(&self) -> bool {
        self.nullable
    }
    pub(crate) fn with_op(&self, op: ScalarOp) -> Self {
        Self {
            op,
            data_type: self.data_type,
            nullable: self.nullable,
        }
    }
    pub fn op(&self) -> ScalarOp {
        self.op
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
}

/// Immutable, bound arithmetic or predicate program, independent of CubeCL contexts/devices.
/// Clones share owned data. IDs are local to this program, not cross-program handles.
#[derive(Clone, Debug)]
pub struct KernelProgram {
    schema: Arc<Schema>,
    nodes: Arc<[TypedExpr]>,
    root: ExprId,
    reachable: Arc<[bool]>,
}

impl KernelProgram {
    pub(crate) fn rebind_schema(&self, schema: &Schema) -> Self {
        let nodes = self
            .nodes()
            .iter()
            .map(|node| {
                node.with_op(match node.op() {
                    ScalarOp::Column(id) => ScalarOp::Column(
                        schema
                            .column_id(self.schema().columns()[id.index()].name())
                            .expect("union column"),
                    ),
                    ScalarOp::Parameter(id) => ScalarOp::Parameter(
                        schema
                            .parameter_id(self.schema().parameters()[id.index()].name())
                            .expect("union parameter"),
                    ),
                    other => other,
                })
            })
            .collect::<Vec<_>>();
        Self {
            schema: Arc::new(schema.clone()),
            nodes: nodes.into(),
            root: self.root,
            reachable: self.reachable.clone(),
        }
    }
    pub(crate) fn reachable(&self) -> &[bool] {
        &self.reachable
    }
    /// Whether the output depends on checked I32 add/mul (requires an error plane).
    pub fn has_integer_arithmetic(&self) -> bool {
        self.nodes
            .iter()
            .zip(self.reachable.iter())
            .any(|(node, &used)| {
                used && node.data_type == DataType::I32
                    && matches!(node.op, ScalarOp::Add(..) | ScalarOp::Mul(..))
            })
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }
    pub fn nodes(&self) -> &[TypedExpr] {
        &self.nodes
    }
    pub fn root(&self) -> ExprId {
        self.root
    }
    pub fn output_nullable(&self) -> bool {
        self.nodes[self.root.0].nullable
    }
    pub fn output_type(&self) -> DataType {
        self.nodes[self.root.0].data_type
    }
}

pub fn prepare(expr: &Expr, schema: &Schema) -> Result<KernelProgram, Error> {
    prepare_with_limits(expr, schema, PrepareLimits::default())
}

/// Validate the entire topologically ordered arena and resolve names to IDs.
/// Referenced inputs must be F32 or I32; operands must match exactly. No reassociation, constant
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
    let mut nodes: Vec<TypedExpr> = Vec::with_capacity(node_count);
    let mut depths = Vec::<usize>::with_capacity(node_count);
    for (index, node) in expr.nodes.iter().enumerate() {
        let node_id = ExprId(index);
        let (op, depth, data_type) = match node {
            ExprNode::Column(name) => {
                let kind = InputKind::Column;
                let id = schema.column_id(name).ok_or_else(|| Error::UnknownName {
                    kind,
                    name: name.clone(),
                })?;
                let ty = schema.columns()[id.index()].data_type();
                require_numeric(kind, name, ty)?;
                (ScalarOp::Column(id), 1, ty)
            }
            ExprNode::Parameter(name) => {
                let kind = InputKind::Parameter;
                let id = schema
                    .parameter_id(name)
                    .ok_or_else(|| Error::UnknownName {
                        kind,
                        name: name.clone(),
                    })?;
                let ty = schema.parameters()[id.index()].data_type();
                require_numeric(kind, name, ty)?;
                (ScalarOp::Parameter(id), 1, ty)
            }
            ExprNode::Add(a, b)
            | ExprNode::Mul(a, b)
            | ExprNode::Compare(_, a, b)
            | ExprNode::And(a, b)
            | ExprNode::Or(a, b) => {
                let expected = match node {
                    ExprNode::And(..) | ExprNode::Or(..) => DataType::Bool,
                    _ => {
                        let ty = nodes
                            .get(a.index())
                            .ok_or(Error::InvalidReference {
                                node: node_id,
                                operand: *a,
                            })?
                            .data_type;
                        if ty == DataType::Bool {
                            return Err(Error::OperandTypeMismatch {
                                node: node_id,
                                operand: *a,
                                expected: DataType::F32,
                                actual: ty,
                            });
                        }
                        ty
                    }
                };
                for &operand in [a, b] {
                    validate_operand(&nodes, node_id, operand, expected)?;
                }
                let depth = depths[a.0].max(depths[b.0]) + 1;
                let (op, ty) = match node {
                    ExprNode::Add(..) => (ScalarOp::Add(*a, *b), expected),
                    ExprNode::Mul(..) => (ScalarOp::Mul(*a, *b), expected),
                    ExprNode::Compare(cmp, ..) => (ScalarOp::Compare(*cmp, *a, *b), DataType::Bool),
                    ExprNode::And(..) => (ScalarOp::And(*a, *b), DataType::Bool),
                    ExprNode::Or(..) => (ScalarOp::Or(*a, *b), DataType::Bool),
                    _ => unreachable!(),
                };
                (op, depth, ty)
            }
            ExprNode::Not(a) => {
                validate_operand(&nodes, node_id, *a, DataType::Bool)?;
                (ScalarOp::Not(*a), depths[a.0] + 1, DataType::Bool)
            }
            ExprNode::IsNull(a) => {
                if a.index() >= index {
                    return Err(Error::InvalidReference {
                        node: node_id,
                        operand: *a,
                    });
                }
                (ScalarOp::IsNull(*a), depths[a.0] + 1, DataType::Bool)
            }
        };
        if depth > limits.max_depth {
            return Err(Error::ExpressionTooDeep {
                node: node_id,
                depth,
                limit: limits.max_depth,
            });
        }
        let nullable = match op {
            ScalarOp::Column(id) => schema.columns()[id.index()].is_nullable(),
            ScalarOp::Parameter(id) => schema.parameters()[id.index()].is_nullable(),
            ScalarOp::Add(a, b)
            | ScalarOp::Mul(a, b)
            | ScalarOp::Compare(_, a, b)
            | ScalarOp::And(a, b)
            | ScalarOp::Or(a, b) => nodes[a.index()].nullable || nodes[b.index()].nullable,
            ScalarOp::Not(a) => nodes[a.index()].nullable,
            ScalarOp::IsNull(_) => false,
        };
        nodes.push(TypedExpr {
            op,
            data_type,
            nullable,
        });
        depths.push(depth);
    }
    let mut reachable = vec![false; nodes.len()];
    reachable[expr.root.index()] = true;
    for index in (0..nodes.len()).rev() {
        if !reachable[index] {
            continue;
        }
        match nodes[index].op {
            ScalarOp::Add(a, b)
            | ScalarOp::Mul(a, b)
            | ScalarOp::Compare(_, a, b)
            | ScalarOp::And(a, b)
            | ScalarOp::Or(a, b) => {
                reachable[a.index()] = true;
                reachable[b.index()] = true;
            }
            ScalarOp::Not(a) | ScalarOp::IsNull(a) => reachable[a.index()] = true,
            _ => {}
        }
    }
    Ok(KernelProgram {
        reachable: reachable.into(),
        schema: Arc::new(schema.clone()),
        nodes: nodes.into(),
        root: expr.root,
    })
}

fn require_numeric(kind: InputKind, name: &str, data_type: DataType) -> Result<(), Error> {
    if !matches!(data_type, DataType::F32 | DataType::I32) {
        return Err(Error::UnsupportedType {
            kind,
            name: name.into(),
            data_type,
        });
    }
    Ok(())
}

fn validate_operand(
    nodes: &[TypedExpr],
    node: ExprId,
    operand: ExprId,
    expected: DataType,
) -> Result<(), Error> {
    let value = nodes
        .get(operand.index())
        .ok_or(Error::InvalidReference { node, operand })?;
    if value.data_type != expected {
        return Err(Error::OperandTypeMismatch {
            node,
            operand,
            expected,
            actual: value.data_type,
        });
    }
    Ok(())
}

/// Rebind programs to one schema and intern identical typed nodes. No algebraic
/// reassociation. Conflicting declarations decline fusion rather than changing
/// the independently prepared programs' input contracts.
pub(crate) fn merge_programs(
    programs: &[&KernelProgram],
) -> Result<(KernelProgram, Vec<ExprId>), &'static str> {
    use std::collections::{BTreeMap, HashMap};
    let mut columns = BTreeMap::new();
    let mut parameters = BTreeMap::new();
    for program in programs {
        for node in program.nodes() {
            let (map, field) = match node.op() {
                ScalarOp::Column(id) => (&mut columns, &program.schema().columns()[id.index()]),
                ScalarOp::Parameter(id) => {
                    (&mut parameters, &program.schema().parameters()[id.index()])
                }
                _ => continue,
            };
            if let Some(old) = map.insert(field.name().to_owned(), field.clone())
                && old != *field
            {
                return Err("fusion requires consistent input declarations");
            }
        }
    }
    let schema = Schema::from_fields(
        &columns.into_values().collect::<Vec<_>>(),
        &parameters.into_values().collect::<Vec<_>>(),
    )
    .map_err(|_| "invalid fused schema")?;
    let mut nodes = Vec::new();
    let mut interned = HashMap::new();
    let mut roots = Vec::new();
    for program in programs {
        let mut remap: Vec<ExprId> = Vec::new();
        for node in program.nodes() {
            let op = match node.op() {
                ScalarOp::Column(id) => ScalarOp::Column(
                    schema
                        .column_id(program.schema().columns()[id.index()].name())
                        .expect("merged column"),
                ),
                ScalarOp::Parameter(id) => ScalarOp::Parameter(
                    schema
                        .parameter_id(program.schema().parameters()[id.index()].name())
                        .expect("merged parameter"),
                ),
                ScalarOp::Add(a, b) => ScalarOp::Add(remap[a.index()], remap[b.index()]),
                ScalarOp::Mul(a, b) => ScalarOp::Mul(remap[a.index()], remap[b.index()]),
                ScalarOp::Compare(op, a, b) => {
                    ScalarOp::Compare(op, remap[a.index()], remap[b.index()])
                }
                ScalarOp::And(a, b) => ScalarOp::And(remap[a.index()], remap[b.index()]),
                ScalarOp::Or(a, b) => ScalarOp::Or(remap[a.index()], remap[b.index()]),
                ScalarOp::Not(a) => ScalarOp::Not(remap[a.index()]),
                ScalarOp::IsNull(a) => ScalarOp::IsNull(remap[a.index()]),
            };
            let canonical = node.with_op(op);
            let id = *interned.entry(canonical.clone()).or_insert_with(|| {
                let id = ExprId::new(nodes.len());
                nodes.push(canonical);
                id
            });
            remap.push(id);
        }
        roots.push(remap[program.root().index()]);
    }
    if roots.is_empty() {
        return Err("fusion needs a predicate or argument");
    }
    // An artificial Bool root marks all real roots reachable. It is not exported.
    let mut root = None;
    for &id in &roots {
        let null = ExprId::new(nodes.len());
        nodes.push(TypedExpr {
            op: ScalarOp::IsNull(id),
            data_type: DataType::Bool,
            nullable: false,
        });
        if let Some(old) = root {
            let id = ExprId::new(nodes.len());
            nodes.push(TypedExpr {
                op: ScalarOp::Or(old, null),
                data_type: DataType::Bool,
                nullable: false,
            });
            root = Some(id);
        } else {
            root = Some(null);
        }
    }
    if nodes.len() > PrepareLimits::default().max_nodes {
        return Err("fused arena exceeds node limit");
    }
    let root = root.expect("roots");
    let mut reachable = vec![false; nodes.len()];
    reachable[root.index()] = true;
    for i in (0..nodes.len()).rev() {
        if !reachable[i] {
            continue;
        }
        match nodes[i].op() {
            ScalarOp::Add(a, b)
            | ScalarOp::Mul(a, b)
            | ScalarOp::Compare(_, a, b)
            | ScalarOp::And(a, b)
            | ScalarOp::Or(a, b) => {
                reachable[a.index()] = true;
                reachable[b.index()] = true;
            }
            ScalarOp::Not(a) | ScalarOp::IsNull(a) => reachable[a.index()] = true,
            _ => {}
        }
    }
    Ok((
        KernelProgram {
            schema: Arc::new(schema),
            nodes: nodes.into(),
            root,
            reachable: reachable.into(),
        },
        roots,
    ))
}
