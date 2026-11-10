#![cfg(feature = "datafusion")]

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use cuberel::datafusion::{AdapterOptions, register};
use datafusion::{execution::session_state::SessionStateBuilder, prelude::SessionContext};

#[tokio::test]
async fn optional_registration_preserves_sql_and_default_rules() {
    for enabled in [false, true] {
        let state = register(
            SessionStateBuilder::new_with_default_features(),
            AdapterOptions { enabled },
        )
        .build();
        let names: Vec<_> = state
            .physical_optimizers()
            .iter()
            .map(|r| r.name())
            .collect();
        assert!(names.len() > 5, "default optimizer pipeline was lost");
        assert_eq!(
            names.iter().filter(|&&n| n == "cuberel").count(),
            usize::from(enabled)
        );
        if enabled {
            assert_eq!(names[names.len() - 2], "cuberel");
            assert_eq!(names.last().copied(), Some("SanityCheckPlan"));
        }
        let ctx = SessionContext::new_with_state(state);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]))],
        )
        .unwrap();
        ctx.register_batch("t", batch).unwrap();
        let result = ctx
            .sql("SELECT x + CAST(2 AS INT) AS y FROM t WHERE x > CAST(1 AS INT)")
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert_eq!(result.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
        assert_eq!(
            result[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(0),
            5
        );
    }
}
