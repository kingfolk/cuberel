use cubecl::prelude::AddressType;

use cuberel::{
    Batch, ColumnData, DataType, Error, InputKind, Parameters, Schema,
    abi::{
        BINDING_ABI_VERSION, BindingError, BindingLayout, BufferAccess, BufferDescriptor,
        BufferSource, PhysicalType, ScalarArgument, ScalarSource,
    },
    col, param, prepare,
};

fn layout() -> BindingLayout {
    let schema = Schema::new(
        &[
            ("b", DataType::F32),
            ("unused", DataType::I32),
            ("a", DataType::F32),
        ],
        &[
            ("bias", DataType::F32),
            ("unused", DataType::I32),
            ("scale", DataType::F32),
        ],
    )
    .unwrap();
    let program = prepare(
        &(col("a") * param("scale") + col("b") + param("bias") + param("scale")),
        &schema,
    )
    .unwrap();
    BindingLayout::new(&program)
}

fn descriptors(layout: &BindingLayout, rows: u64) -> Vec<BufferDescriptor> {
    layout
        .buffers()
        .iter()
        .map(|_| BufferDescriptor {
            element_type: PhysicalType::F32,
            logical_len: rows,
            allocation_bytes: rows * 4,
            byte_offset: 0,
            end_padding_bytes: 0,
        })
        .collect()
}

#[test]
fn canonical_layout_deduplicates_references_and_preserves_schema_order() {
    let layout = layout();
    assert_eq!(layout.version(), BINDING_ABI_VERSION);
    assert_eq!(layout.address_type(), AddressType::U32);
    assert_eq!(
        layout
            .buffers()
            .iter()
            .map(|slot| slot.name())
            .collect::<Vec<_>>(),
        ["b", "a", "output"]
    );
    for (index, slot) in layout.buffers().iter().enumerate() {
        assert_eq!(slot.index(), index);
        assert_eq!(slot.element_type(), PhysicalType::F32);
        assert_eq!(slot.alignment(), 4);
    }
    assert!(matches!(layout.buffers()[0].source(), BufferSource::Column(id) if id.index() == 0));
    assert!(matches!(layout.buffers()[1].source(), BufferSource::Column(id) if id.index() == 2));
    assert_eq!(layout.buffers()[0].access(), BufferAccess::ReadOnly);
    assert_eq!(layout.buffers()[2].source(), BufferSource::Output);
    assert_eq!(layout.buffers()[2].access(), BufferAccess::WriteOnly);
    assert_eq!(
        layout
            .scalars()
            .iter()
            .map(|slot| slot.name())
            .collect::<Vec<_>>(),
        ["row_count", "bias", "scale"]
    );
    assert_eq!(layout.scalars()[0].source(), ScalarSource::RowCount);
    assert_eq!(layout.scalars()[0].element_type(), PhysicalType::U32);
    assert_eq!(layout.scalars()[0].ordinal_within_type(), 0);
    assert_eq!(layout.scalars()[1].element_type(), PhysicalType::F32);
    assert_eq!(layout.scalars()[1].ordinal_within_type(), 0);
    assert_eq!(layout.scalars()[2].ordinal_within_type(), 1);
}

#[test]
fn host_binding_orders_same_type_scalars_and_snapshots_values() {
    let layout = layout();
    let a = [1.0, 2.0, 3.0];
    let b = [10.0, 20.0, 30.0];
    let batch = Batch::new(3)
        .with_column("a", ColumnData::F32(&a))
        .unwrap()
        .with_column("b", ColumnData::F32(&b))
        .unwrap();
    let parameters = Parameters::new()
        .with("scale", 2.0_f32)
        .unwrap()
        .with("bias", 7.0_f32)
        .unwrap();
    let bound = layout.bind_host(&batch, &parameters).unwrap();
    assert_eq!(bound.layout(), &layout);
    assert_eq!(bound.extent().row_count(), 3);
    assert_eq!(bound.extent().byte_len(), 12);
    assert_eq!(bound.columns(), &[b.as_slice(), a.as_slice()]);
    assert_eq!(
        bound.scalars(),
        &[
            ScalarArgument::U32(3),
            ScalarArgument::F32(7.0),
            ScalarArgument::F32(2.0)
        ]
    );
    let rebound = layout
        .bind_host(
            &batch,
            &Parameters::new()
                .with("bias", -1.0_f32)
                .unwrap()
                .with("scale", 4.0_f32)
                .unwrap(),
        )
        .unwrap();
    assert_eq!(bound.layout(), rebound.layout());
    assert_eq!(bound.scalars()[1], ScalarArgument::F32(7.0));
    assert_eq!(rebound.scalars()[1], ScalarArgument::F32(-1.0));
}

#[test]
fn layouts_identify_abi_not_expression_semantics_or_invocation_values() {
    let schema = Schema::new(&[("a", DataType::F32)], &[("p", DataType::F32)]).unwrap();
    let add = BindingLayout::new(&prepare(&(col("a") + param("p")), &schema).unwrap());
    let mul = BindingLayout::new(&prepare(&(col("a") * param("p")), &schema).unwrap());
    assert_eq!(add, mul);
    let batch = Batch::new(0)
        .with_column("a", ColumnData::F32(&[]))
        .unwrap();
    let bound = add
        .bind_host(&batch, &Parameters::new().with("p", 3.0_f32).unwrap())
        .unwrap();
    assert_eq!(bound.layout(), &mul);
    assert_eq!(bound.extent().byte_len(), 0);
}

#[test]
fn parameter_only_layout_has_output_and_count_without_input_buffers() {
    let schema = Schema::new(&[], &[("p", DataType::F32)]).unwrap();
    let layout = BindingLayout::new(&prepare(&(param("p") * param("p")), &schema).unwrap());
    assert_eq!(layout.buffers().len(), 1);
    let bound = layout
        .bind_host(
            &Batch::new(9),
            &Parameters::new().with("p", 2.0_f32).unwrap(),
        )
        .unwrap();
    assert!(bound.columns().is_empty());
    assert_eq!(
        bound.scalars(),
        &[ScalarArgument::U32(9), ScalarArgument::F32(2.0)]
    );
}

#[test]
fn host_validation_reports_missing_and_mistyped_inputs_even_for_empty_batches() {
    let layout = layout();
    assert!(matches!(
        layout.bind_host(&Batch::new(0), &Parameters::new()),
        Err(BindingError::Input(Error::MissingInput {
            kind: InputKind::Column,
            ..
        }))
    ));
    let wrong = Batch::new(0)
        .with_column("a", ColumnData::I32(&[]))
        .unwrap()
        .with_column("b", ColumnData::F32(&[]))
        .unwrap();
    assert!(matches!(
        layout.bind_host(&wrong, &Parameters::new()),
        Err(BindingError::Input(Error::TypeMismatch {
            kind: InputKind::Column,
            ..
        }))
    ));
    let batch = Batch::new(0)
        .with_column("a", ColumnData::F32(&[]))
        .unwrap()
        .with_column("b", ColumnData::F32(&[]))
        .unwrap();
    assert!(matches!(
        layout.bind_host(&batch, &Parameters::new()),
        Err(BindingError::Input(Error::MissingInput {
            kind: InputKind::Parameter,
            ..
        }))
    ));
    let parameters = Parameters::new()
        .with("bias", 1_i32)
        .unwrap()
        .with("scale", 2.0_f32)
        .unwrap();
    assert!(matches!(
        layout.bind_host(&batch, &parameters),
        Err(BindingError::Input(Error::TypeMismatch {
            kind: InputKind::Parameter,
            ..
        }))
    ));
}

#[test]
fn buffer_count_type_and_logical_length_are_checked() {
    let layout = layout();
    let original = descriptors(&layout, 5);
    assert_eq!(
        layout.validate_buffers(5, &original[..2]).unwrap_err(),
        BindingError::BufferCount {
            expected: 3,
            actual: 2
        }
    );
    let mut bad = original.clone();
    bad[0].element_type = PhysicalType::I32;
    assert_eq!(
        layout.validate_buffers(5, &bad).unwrap_err(),
        BindingError::BufferType {
            index: 0,
            expected: PhysicalType::F32,
            actual: PhysicalType::I32
        }
    );
    bad = original;
    bad[1].logical_len = 4;
    assert_eq!(
        layout.validate_buffers(5, &bad).unwrap_err(),
        BindingError::BufferLength {
            index: 1,
            expected: 5,
            actual: 4
        }
    );
}

#[test]
fn validates_offsets_capacity_alignment_and_checked_ranges() {
    let layout = layout();
    let mut views = descriptors(&layout, 3);
    for view in &mut views {
        view.allocation_bytes = 40;
        view.byte_offset = 4;
        view.end_padding_bytes = 8;
    }
    assert_eq!(layout.validate_buffers(3, &views).unwrap().byte_len(), 12);
    views[2].allocation_bytes = 23;
    assert_eq!(
        layout.validate_buffers(3, &views).unwrap_err(),
        BindingError::BufferTooSmall {
            index: 2,
            required: 12,
            available: 11
        }
    );
    views[2].allocation_bytes = 40;
    views[1].byte_offset = 1;
    assert!(matches!(
        layout.validate_buffers(3, &views),
        Err(BindingError::BufferAlignment { index: 1, .. })
    ));
    views[1].byte_offset = u64::MAX;
    assert_eq!(
        layout.validate_buffers(3, &views).unwrap_err(),
        BindingError::InvalidBufferRange { index: 1 }
    );
    views[1].byte_offset = 36;
    assert_eq!(
        layout.validate_buffers(3, &views).unwrap_err(),
        BindingError::InvalidBufferRange { index: 1 }
    );
}

#[test]
fn addresses_check_bytes_not_only_row_counts_without_allocating_large_buffers() {
    let layout = layout();
    let last_safe_rows = u64::from(u32::MAX) / 4;
    assert_eq!(
        layout.extent(last_safe_rows).unwrap().byte_len() as u64,
        last_safe_rows * 4
    );
    assert!(matches!(
        layout.extent(last_safe_rows + 1),
        Err(BindingError::AddressOverflow {
            field: "byte_length",
            ..
        })
    ));
    assert!(matches!(
        layout.extent(u64::from(u32::MAX) + 1),
        Err(BindingError::AddressOverflow {
            field: "row_count",
            ..
        })
    ));
    assert!(matches!(
        layout.extent(u64::MAX),
        Err(BindingError::AddressOverflow {
            field: "row_count",
            ..
        })
    ));
    let mut views = descriptors(&layout, 1);
    views[0].byte_offset = u64::from(u32::MAX) - 3;
    views[0].allocation_bytes = u64::from(u32::MAX) + 9;
    assert!(matches!(
        layout.validate_buffers(1, &views),
        Err(BindingError::AddressOverflow {
            field: "buffer_byte_end",
            ..
        })
    ));
}

#[test]
fn internal_slot_names_do_not_reserve_user_column_or_parameter_names() {
    let schema = Schema::new(
        &[("output", DataType::F32)],
        &[("row_count", DataType::F32)],
    )
    .unwrap();
    let layout =
        BindingLayout::new(&prepare(&(col("output") + param("row_count")), &schema).unwrap());
    let batch = Batch::new(1)
        .with_column("output", ColumnData::F32(&[2.0]))
        .unwrap();
    let parameters = Parameters::new().with("row_count", 7.0_f32).unwrap();
    let bound = layout.bind_host(&batch, &parameters).unwrap();
    assert_eq!(
        bound.scalars(),
        &[ScalarArgument::U32(1), ScalarArgument::F32(7.0)]
    );
    assert!(matches!(
        layout.buffers()[0].source(),
        BufferSource::Column(_)
    ));
    assert_eq!(layout.buffers()[1].source(), BufferSource::Output);
}

#[test]
fn zero_rows_are_valid_and_u64_is_explicitly_unsupported() {
    let layout = layout();
    assert_eq!(
        layout
            .validate_buffers(0, &descriptors(&layout, 0))
            .unwrap()
            .row_count(),
        0
    );
    let schema = Schema::new(&[("a", DataType::F32)], &[]).unwrap();
    let program = prepare(&col("a"), &schema).unwrap();
    assert_eq!(
        BindingLayout::with_address_type(&program, AddressType::U64).unwrap_err(),
        BindingError::UnsupportedAddressType(AddressType::U64)
    );
}
