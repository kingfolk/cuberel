//! Conservative translation of bound physical expressions, never SQL text.
use std::{collections::BTreeMap, fmt, sync::Arc};

use ::datafusion::{
    common::ScalarValue as SqlScalar,
    logical_expr::Operator,
    physical_expr::{
        PhysicalExpr,
        expressions::{BinaryExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NotExpr},
    },
};
use arrow_schema::{DataType as ArrowType, Schema as ArrowSchema};

use crate::{
    DataType, Expr, Field, KernelProgram, Parameters, ScalarValue, Schema, col, param, prepare,
};

/// A stable reason for keeping an expression in the original CPU plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Unsupported {
    pub code: &'static str,
    pub detail: String,
}
impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for Unsupported {}
fn reject(code: &'static str, detail: impl Into<String>) -> Unsupported {
    Unsupported {
        code,
        detail: detail.into(),
    }
}

/// One typed expression and immutable scalar bindings. Input columns use `cN`
/// names tied to physical field positions, independent of SQL aliases.
#[derive(Clone, Debug)]
pub struct TranslatedExpression {
    pub program: KernelProgram,
    pub parameters: Parameters,
    pub input_columns: Vec<usize>,
    pub(crate) bindings: Vec<(String, ScalarValue)>,
}

impl TranslatedExpression {
    /// Literal values are execution bindings, separate from the typed program.
    pub fn literal_bindings(&self) -> &[(String, ScalarValue)] {
        &self.bindings
    }
}

/// Translate one expression without allocation on a device or changing SQL settings.
/// Wrapping I32 arithmetic and unverified floating operations stay on CPU.
pub fn translate_expression(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &ArrowSchema,
) -> Result<TranslatedExpression, Unsupported> {
    translate_named(expr, schema, "expr")
}

pub(crate) fn check_expression_limits(expr: &Arc<dyn PhysicalExpr>) -> Result<(), Unsupported> {
    // Bound traversal before invoking recursive DataFusion type/nullability APIs.
    let mut pending = vec![(expr, 0)];
    let mut nodes = 0;
    while let Some((node, depth)) = pending.pop() {
        nodes += 1;
        if depth > 64 || nodes > 1024 {
            return Err(reject("expression_limit", "translation depth/node limit"));
        }
        pending.extend(node.children().into_iter().map(|child| (child, depth + 1)));
    }
    Ok(())
}

pub(crate) fn translate_named(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &ArrowSchema,
    namespace: &str,
) -> Result<TranslatedExpression, Unsupported> {
    check_expression_limits(expr)?;
    let mut translator = Translator {
        schema,
        namespace,
        columns: BTreeMap::new(),
        bindings: Vec::new(),
        fields: Vec::new(),
        visited: 0,
    };
    let (expression, _) = translator.visit(expr, 0)?;
    let columns: Vec<_> = translator.columns.values().cloned().collect();
    let own_schema = Schema::from_fields(&columns, &translator.fields)
        .map_err(|e| reject("binding", e.to_string()))?;
    let program =
        prepare(&expression, &own_schema).map_err(|e| reject("expression_limit", e.to_string()))?;
    let sql_type = expr
        .data_type(schema)
        .map_err(|e| reject("binding", e.to_string()))?;
    let output = match program.output_type() {
        DataType::I32 => ArrowType::Int32,
        DataType::F32 => ArrowType::Float32,
        DataType::Bool => ArrowType::Boolean,
    };
    if sql_type != output
        || expr
            .nullable(schema)
            .map_err(|e| reject("binding", e.to_string()))?
            != program.output_nullable()
    {
        return Err(reject(
            "semantic_mismatch",
            "output type/nullability differs",
        ));
    }
    let mut parameters = Parameters::new();
    for (name, value) in &translator.bindings {
        parameters = parameters
            .with(name, *value)
            .map_err(|e| reject("binding", e.to_string()))?;
    }
    Ok(TranslatedExpression {
        program,
        parameters,
        input_columns: translator.columns.keys().copied().collect(),
        bindings: translator.bindings,
    })
}

struct Translator<'a> {
    schema: &'a ArrowSchema,
    namespace: &'a str,
    columns: BTreeMap<usize, Field>,
    bindings: Vec<(String, ScalarValue)>,
    fields: Vec<Field>,
    visited: usize,
}
impl Translator<'_> {
    // The bool tracks possible checked errors. DataFusion short-circuits boolean
    // RHS evaluation, while the DSL is eager: those compositions are rejected.
    fn visit(
        &mut self,
        expr: &Arc<dyn PhysicalExpr>,
        depth: usize,
    ) -> Result<(Expr, bool), Unsupported> {
        self.visited += 1;
        if depth > 64 || self.visited > 1024 {
            return Err(reject("expression_limit", "translation depth/node limit"));
        }
        if expr
            .return_field(self.schema)
            .map_err(|e| reject("binding", e.to_string()))?
            .metadata()
            .keys()
            .any(|k| k.starts_with("ARROW:extension:"))
        {
            return Err(reject(
                "semantic_mismatch",
                "extension expression semantics",
            ));
        }
        if let Some(column) = expr.downcast_ref::<Column>() {
            let index = column.index();
            let field = self
                .schema
                .fields()
                .get(index)
                .ok_or_else(|| reject("binding", "column index outside schema"))?;
            if field
                .metadata()
                .keys()
                .any(|k| k.starts_with("ARROW:extension:"))
            {
                return Err(reject("semantic_mismatch", "extension column semantics"));
            }
            let ty = numeric_type(field.data_type())?;
            self.columns.insert(
                index,
                Field::new(format!("c{index}"), ty).with_nullable(field.is_nullable()),
            );
            return Ok((col(format!("c{index}")), false));
        }
        if let Some(literal) = expr.downcast_ref::<Literal>() {
            let (ty, value, nullable) = match literal.value() {
                SqlScalar::Int32(v) => (DataType::I32, ScalarValue::NullableI32(*v), v.is_none()),
                SqlScalar::Float32(v) => (DataType::F32, ScalarValue::NullableF32(*v), v.is_none()),
                _ => {
                    return Err(reject(
                        "unsupported_type",
                        format!("literal {:?}", literal.value().data_type()),
                    ));
                }
            };
            let name = format!("p{}_{}", self.namespace, self.bindings.len());
            self.fields
                .push(Field::new(&name, ty).with_nullable(nullable));
            self.bindings.push((name.clone(), value));
            return Ok((param(name), false));
        }
        if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
            let op = *binary.op();
            if matches!(op, Operator::Plus | Operator::Multiply) {
                if expr
                    .data_type(self.schema)
                    .map_err(|e| reject("binding", e.to_string()))?
                    != ArrowType::Int32
                {
                    return Err(reject(
                        "semantic_mismatch",
                        "floating arithmetic profile is not verified",
                    ));
                }
                let checked = BinaryExpr::new(binary.left().clone(), op, binary.right().clone())
                    .with_fail_on_overflow(true);
                if binary != &checked {
                    return Err(reject(
                        "semantic_mismatch",
                        "wrapping I32 is not checked I32",
                    ));
                }
            }
            if !matches!(
                op,
                Operator::Plus
                    | Operator::Multiply
                    | Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
                    | Operator::And
                    | Operator::Or
            ) {
                return Err(reject("unsupported_operator", op.to_string()));
            }
            if !matches!(
                op,
                Operator::And | Operator::Or | Operator::Plus | Operator::Multiply
            ) && binary
                .left()
                .data_type(self.schema)
                .map_err(|e| reject("binding", e.to_string()))?
                != ArrowType::Int32
            {
                return Err(reject(
                    "semantic_mismatch",
                    "float total-order/NaN and boolean comparisons are not IEEE I32 comparisons",
                ));
            }
            let (left, le) = self.visit(binary.left(), depth + 1)?;
            let (right, re) = self.visit(binary.right(), depth + 1)?;
            if matches!(op, Operator::And | Operator::Or) && (le || re) {
                return Err(reject(
                    "semantic_mismatch",
                    "short-circuit checked effects differ from eager DSL",
                ));
            }
            let result = match op {
                Operator::Plus => left + right,
                Operator::Multiply => left * right,
                Operator::Eq => left.eq_expr(right),
                Operator::NotEq => left.ne_expr(right),
                Operator::Lt => left.lt(right),
                Operator::LtEq => left.le(right),
                Operator::Gt => left.gt(right),
                Operator::GtEq => left.ge(right),
                Operator::And => left.and(right),
                Operator::Or => left.or(right),
                _ => unreachable!(),
            };
            return Ok((
                result,
                le || re || matches!(op, Operator::Plus | Operator::Multiply),
            ));
        }
        if let Some(unary) = expr.downcast_ref::<IsNullExpr>() {
            let (arg, effect) = self.visit(unary.arg(), depth + 1)?;
            return Ok((arg.is_null(), effect));
        }
        if let Some(unary) = expr.downcast_ref::<IsNotNullExpr>() {
            let (arg, effect) = self.visit(unary.arg(), depth + 1)?;
            return Ok((!arg.is_null(), effect));
        }
        if let Some(unary) = expr.downcast_ref::<NotExpr>() {
            let (arg, effect) = self.visit(unary.arg(), depth + 1)?;
            return Ok((!arg, effect));
        }
        if let Some(cast) = expr.downcast_ref::<CastExpr>() {
            if cast
                .expr()
                .data_type(self.schema)
                .map_err(|e| reject("binding", e.to_string()))?
                == expr
                    .data_type(self.schema)
                    .map_err(|e| reject("binding", e.to_string()))?
                && expr
                    .return_field(self.schema)
                    .map_err(|e| reject("binding", e.to_string()))?
                    .metadata()
                    .is_empty()
            {
                return self.visit(cast.expr(), depth + 1);
            }
            return Err(reject(
                "unsupported_cast",
                "only metadata-free identity casts",
            ));
        }
        Err(reject("unsupported_operator", expr.to_string()))
    }
}
fn numeric_type(ty: &ArrowType) -> Result<DataType, Unsupported> {
    match ty {
        ArrowType::Int32 => Ok(DataType::I32),
        ArrowType::Float32 => Ok(DataType::F32),
        _ => Err(reject("unsupported_type", format!("input {ty:?}"))),
    }
}
