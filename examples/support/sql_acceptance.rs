//! Deterministic host fixtures and an independent, typed SQL oracle.
use arrow_array::{Array, ArrayRef, BooleanArray, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion::{
    arrow::compute::{SortColumn, SortOptions, concat_batches, lexsort_to_indices, take},
    datasource::MemTable,
    prelude::SessionContext,
};
use std::{collections::HashMap, error::Error, sync::Arc};
pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    Project,
    Aggregate,
    Mixed,
}
impl Workload {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "project" => Ok(Self::Project),
            "aggregate" => Ok(Self::Aggregate),
            "mixed" => Ok(Self::Mixed),
            _ => Err(format!("unknown workload: {s}").into()),
        }
    }
    pub fn sql(self, pass: usize) -> String {
        let predicate = format!("x < CAST({pass} AS INT)");
        match self {
            Self::Project => format!(
                "SELECT id,x AS value,y AS payload,y IS NULL AS missing FROM t WHERE {predicate}"
            ),
            Self::Aggregate => format!(
                "SELECT sum(y) AS total,count(*) AS n,count(y) AS valid,min(y) AS lo,max(y) AS hi FROM t WHERE {predicate}"
            ),
            Self::Mixed => format!(
                "SELECT id,x AS value,y AS payload,y IS NULL AS missing FROM t WHERE {predicate} ORDER BY payload ASC NULLS LAST,id LIMIT 17"
            ),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub rows: usize,
    pub pass: usize,
    pub nulls: usize,
    pub batches: usize,
    pub partitions: usize,
}
impl Case {
    pub fn validate(self) -> Result<()> {
        if self.rows > i32::MAX as usize
            || self.pass > 100
            || self.nulls > 100
            || self.batches == 0
            || self.partitions == 0
        {
            return Err("rows must fit I32; percentages must be 0..100; batches/partitions must be positive".into());
        }
        Ok(())
    }
}
pub struct Fixture {
    pub case: Case,
    pub input: RecordBatch,
    pub parts: Vec<Vec<RecordBatch>>,
}
impl Fixture {
    pub fn new(case: Case) -> Result<Self> {
        case.validate()?;
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int32, false),
                Field::new("x", DataType::Int32, true),
                Field::new("y", DataType::Int32, true),
            ],
            HashMap::from([("fixture".into(), "sql-acceptance-v1".into())]),
        ));
        // Padding plus slice forces nonzero value and bitmap offsets, including for empty input.
        let id = Int32Array::from_iter_values((0..case.rows + 3).map(|i| (i as i64 - 3) as i32));
        let x = Int32Array::from(
            (0..case.rows + 3)
                .map(|j| {
                    let i = j.saturating_sub(3) as u64;
                    (((i * 53 + (i / 100) * 19 + 17) % 100) >= case.nulls as u64)
                        .then_some(((i * 37) % 100) as i32)
                })
                .collect::<Vec<_>>(),
        );
        let y = Int32Array::from(
            (0..case.rows + 3)
                .map(|j| {
                    let i = j.saturating_sub(3) as u64;
                    (((i * 29 + (i / 100) * 11 + 7) % 100) >= case.nulls as u64)
                        .then_some(((i * 31) % 2001) as i32 - 1000)
                })
                .collect::<Vec<_>>(),
        );
        let input = RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(x), Arc::new(y)])?
            .slice(3, case.rows);
        let mut parts = vec![vec![]; case.partitions];
        for b in 0..case.batches {
            let start = case.rows * b / case.batches;
            let end = case.rows * (b + 1) / case.batches;
            parts[b % case.partitions].push(input.slice(start, end - start));
            // Explicit empty batches between chunks exercise empty input without removing the source schema.
            parts[b % case.partitions].push(input.slice(start, 0));
        }
        Ok(Self { case, input, parts })
    }
    pub fn register(&self, ctx: &SessionContext) -> Result<()> {
        ctx.register_table(
            "t",
            Arc::new(MemTable::try_new(self.input.schema(), self.parts.clone())?),
        )?;
        Ok(())
    }
    pub fn expected(&self, workload: Workload) -> Result<RecordBatch> {
        let x = self
            .input
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let y = self
            .input
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let mut rows = (0..self.case.rows)
            .filter(|&i| x.is_valid(i) && x.value(i) < self.case.pass as i32)
            .collect::<Vec<_>>();
        if workload == Workload::Aggregate {
            let values = rows
                .iter()
                .filter(|&&i| y.is_valid(i))
                .map(|&i| y.value(i))
                .collect::<Vec<_>>();
            let total =
                (!values.is_empty()).then(|| values.iter().map(|&v| i64::from(v)).sum::<i64>());
            let schema = Arc::new(Schema::new_with_metadata(
                vec![
                    Field::new("total", DataType::Int64, true),
                    Field::new("n", DataType::Int64, false),
                    Field::new("valid", DataType::Int64, false),
                    Field::new("lo", DataType::Int32, true),
                    Field::new("hi", DataType::Int32, true),
                ],
                self.input.schema().metadata().clone(),
            ));
            return Ok(RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![total])),
                    Arc::new(Int64Array::from(vec![rows.len() as i64])),
                    Arc::new(Int64Array::from(vec![values.len() as i64])),
                    Arc::new(Int32Array::from(vec![values.iter().copied().min()])),
                    Arc::new(Int32Array::from(vec![values.iter().copied().max()])),
                ],
            )?);
        }
        if workload == Workload::Mixed {
            rows.sort_by_key(|&i| (y.is_null(i), if y.is_valid(i) { y.value(i) } else { 0 }, i));
            rows.truncate(17);
        }
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("id", DataType::Int32, false),
                Field::new("value", DataType::Int32, true),
                Field::new("payload", DataType::Int32, true),
                Field::new("missing", DataType::Boolean, false),
            ],
            self.input.schema().metadata().clone(),
        ));
        Ok(RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from_iter_values(rows.iter().map(|&i| i as i32))),
                Arc::new(Int32Array::from(
                    rows.iter().map(|&i| Some(x.value(i))).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    rows.iter()
                        .map(|&i| y.is_valid(i).then(|| y.value(i)))
                        .collect::<Vec<_>>(),
                )),
                Arc::new(BooleanArray::from(
                    rows.iter().map(|&i| y.is_null(i)).collect::<Vec<_>>(),
                )),
            ],
        )?)
    }
}
pub fn compare(
    schema: &SchemaRef,
    output: &[RecordBatch],
    expected: &RecordBatch,
    unordered: bool,
) -> Result<()> {
    if schema.as_ref() != expected.schema().as_ref() || output.iter().any(|b| b.schema() != *schema)
    {
        return Err(format!(
            "schema mismatch: actual {schema:?}, expected {:?}",
            expected.schema()
        )
        .into());
    }
    let mut actual = concat_batches(schema, output)?;
    if unordered && actual.num_rows() > 0 {
        let indices = lexsort_to_indices(
            &[SortColumn {
                values: actual.column(0).clone(),
                options: Some(SortOptions::default()),
            }],
            None,
        )?;
        let arrays = actual
            .columns()
            .iter()
            .map(|a| take(a.as_ref(), &indices, None))
            .collect::<std::result::Result<Vec<ArrayRef>, _>>()?;
        actual = RecordBatch::try_new(schema.clone(), arrays)?;
    }
    if actual != *expected {
        return Err(format!(
            "result mismatch (actual {} rows, expected {} rows)",
            actual.num_rows(),
            expected.num_rows()
        )
        .into());
    }
    Ok(())
}
