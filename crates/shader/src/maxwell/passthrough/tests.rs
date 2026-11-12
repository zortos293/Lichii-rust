use super::*;

fn program(n: usize) -> (Program, Vec<BlockId>) {
    let mut prog = Program::new();
    let blocks: Vec<BlockId> = (0..n).map(|_| prog.new_block()).collect();
    prog.block_order = blocks.clone();
    prog.syntax_list = blocks.iter().map(|&b| SyntaxNode::Block(b)).collect();
    (prog, blocks)
}

fn set_reg(prog: &mut Program, block: BlockId, reg: Reg) {
    prog.append(
        block,
        Opcode::SetRegister,
        &[Value::Reg(reg), Value::U32(0)],
        0,
    );
}

fn set_attr(prog: &mut Program, block: BlockId, attr: Attribute, value: Value) -> InstId {
    prog.append(
        block,
        Opcode::SetAttribute,
        &[Value::Attribute(attr), value, Value::U32(0)],
        0,
    )
}

fn get_attr(prog: &mut Program, block: BlockId, attr: Attribute) -> InstId {
    prog.append(
        block,
        Opcode::GetAttribute,
        &[Value::Attribute(attr), Value::U32(0)],
        0,
    )
}

fn ops(prog: &Program, block: BlockId) -> Vec<Opcode> {
    prog.insts_of(block).map(|id| prog.op(id)).collect()
}

fn attribute_stores(prog: &Program, block: BlockId) -> Vec<(Attribute, Value)> {
    prog.insts_of(block)
        .filter(|&id| prog.op(id) == Opcode::SetAttribute)
        .map(|id| match prog.arg(id, 0) {
            Value::Attribute(attr) => (attr, prog.arg(id, 1)),
            other => panic!("{other:?} is not an attribute"),
        })
        .collect()
}

fn varyings(attributes: &[Attribute]) -> VaryingState {
    let mut state = VaryingState::default();
    for &attr in attributes {
        state.set(attr, true);
    }
    state
}

#[test]
fn free_register_is_the_highest_unused_one() {
    let (mut prog, b) = program(2);
    assert_eq!(find_free_register(&prog), Some(Reg::r(254)));
    set_reg(&mut prog, b[0], Reg::r(254));
    prog.append(b[1], Opcode::GetRegister, &[Value::Reg(Reg::r(253))], 0);
    set_reg(&mut prog, b[1], Reg::ur(3));
    assert_eq!(find_free_register(&prog), Some(Reg::r(252)));
    for index in 0..=252 {
        set_reg(&mut prog, b[0], Reg::r(index));
    }
    assert_eq!(find_free_register(&prog), None);
}

#[test]
fn viewport_mask_stores_become_register_writes() {
    let (mut prog, b) = program(2);
    prog.append(b[0], Opcode::Prologue, &[], 0);
    let value = prog.append(b[1], Opcode::GetCbufF32, &[Value::U32(0), Value::U32(0)], 0);
    let store = set_attr(
        &mut prog,
        b[1],
        Attribute::VIEWPORT_MASK,
        Value::Inst(value),
    );

    let reg = lower_viewport_mask(&mut prog);
    assert_eq!(reg, Some(Reg::r(254)));
    assert_eq!(prog.op(store), Opcode::Void);
    assert_eq!(
        ops(&prog, b[0]),
        vec![Opcode::SetRegister, Opcode::Prologue]
    );
    let init = prog.block(b[0]).first().unwrap();
    assert_eq!(
        prog.inst(init).args(),
        &[Value::Reg(Reg::r(254)), Value::U32(1)]
    );
    assert_eq!(
        ops(&prog, b[1]),
        vec![
            Opcode::GetCbufF32,
            Opcode::BitCastU32F32,
            Opcode::SetRegister
        ]
    );
    let bitcast = prog.inst(prog.block(b[1]).first().unwrap()).next().unwrap();
    assert_eq!(prog.arg(bitcast, 0), Value::Inst(value));
    let write = prog.block(b[1]).last().unwrap();
    assert_eq!(
        prog.inst(write).args(),
        &[Value::Reg(Reg::r(254)), Value::Inst(bitcast)]
    );
}

#[test]
fn programs_without_a_viewport_mask_keep_their_stores() {
    let (mut prog, b) = program(1);
    set_attr(&mut prog, b[0], Attribute::LAYER, Value::F32(1.0));
    assert_eq!(lower_viewport_mask(&mut prog), None);
    assert_eq!(ops(&prog, b[0]), vec![Opcode::SetAttribute]);
}

#[test]
fn passthrough_copies_generics_and_position_before_each_epilogue() {
    let (mut prog, b) = program(1);
    prog.append(b[0], Opcode::Epilogue, &[], 0);
    let info = ShaderInfo {
        passthrough: varyings(&[Attribute::generic(2, 1), Attribute::POSITION_X]),
        ..ShaderInfo::default()
    };
    let host = HostTranslateInfo {
        support_viewport_mask: true,
        ..HostTranslateInfo::default()
    };

    lower_geometry_passthrough(&mut prog, &info, 2, &host);
    let all = ops(&prog, b[0]);
    assert_eq!(all.last(), Some(&Opcode::Epilogue));
    assert_eq!(all[all.len() - 2], Opcode::EndPrimitive);
    assert_eq!(
        all.iter().filter(|&&op| op == Opcode::EmitVertex).count(),
        2
    );
    let stores = attribute_stores(&prog, b[0]);
    let expected: Vec<Attribute> = (0..2)
        .flat_map(|_| {
            (0..4)
                .map(|c| Attribute::generic(2, c))
                .chain((0..4).map(|c| Attribute::POSITION_X.offset(c)))
        })
        .collect();
    assert_eq!(stores.iter().map(|s| s.0).collect::<Vec<_>>(), expected);
    for (index, (attr, value)) in stores.iter().enumerate() {
        let load = value.inst().unwrap();
        assert_eq!(prog.op(load), Opcode::GetAttribute);
        assert_eq!(prog.arg(load, 0), Value::Attribute(*attr));
        assert_eq!(prog.arg(load, 1), Value::U32(index as u32 / 8));
    }
}

#[test]
fn passthrough_without_host_viewport_masks_culls_and_picks_the_lowest_viewport() {
    let (mut prog, b) = program(1);
    let mask = prog.append(b[0], Opcode::GetCbufF32, &[Value::U32(0), Value::U32(0)], 0);
    set_attr(&mut prog, b[0], Attribute::VIEWPORT_MASK, Value::Inst(mask));
    prog.append(b[0], Opcode::Epilogue, &[], 0);
    let info = ShaderInfo {
        passthrough: varyings(&[Attribute::POSITION_X]),
        ..ShaderInfo::default()
    };

    lower_geometry_passthrough(&mut prog, &info, 1, &HostTranslateInfo::default());
    let stores = attribute_stores(&prog, b[0]);
    assert_eq!(
        stores.iter().map(|s| s.0).collect::<Vec<_>>(),
        vec![
            Attribute::POSITION_X,
            Attribute::POSITION_Y,
            Attribute::POSITION_Z,
            Attribute::POSITION_W,
            Attribute::VIEWPORT_INDEX,
        ]
    );
    let w = stores[3].1.inst().unwrap();
    assert_eq!(prog.op(w), Opcode::SelectF32);
    assert_eq!(prog.arg(w, 1), Value::F32(1.0));
    let culled = prog.arg(w, 0).inst().unwrap();
    assert_eq!(prog.op(culled), Opcode::IEqual);
    let mask_read = prog.arg(culled, 0).inst().unwrap();
    assert_eq!(prog.op(mask_read), Opcode::GetRegister);
    assert_eq!(prog.arg(mask_read, 0), Value::Reg(Reg::r(254)));
    let index = prog.resolve(stores[4].1).inst().unwrap();
    assert_eq!(prog.op(index), Opcode::BitCastF32U32);
    let viewport = prog.arg(index, 0).inst().unwrap();
    assert_eq!(prog.op(viewport), Opcode::SelectU32);
    assert_eq!(
        prog.op(prog.arg(viewport, 2).inst().unwrap()),
        Opcode::FindUMsb32
    );
}

#[test]
fn legacy_stores_and_loads_move_to_free_generics() {
    let (mut prog, b) = program(1);
    let diffuse_g = Attribute::COLOR_FRONT_DIFFUSE_R.offset(1);
    let store = set_attr(&mut prog, b[0], diffuse_g, Value::F32(0.5));
    let generic_store = set_attr(&mut prog, b[0], Attribute::generic(0, 0), Value::F32(1.0));
    let color_load = get_attr(&mut prog, b[0], diffuse_g);
    let fog_load = get_attr(&mut prog, b[0], Attribute::FOG_COORDINATE);
    let position_load = get_attr(&mut prog, b[0], Attribute::POSITION_X);
    crate::opt::compute_post_order(&mut prog);
    let mut info = ShaderInfo {
        stores: varyings(&[diffuse_g, Attribute::generic(0, 0)]),
        loads: varyings(&[
            diffuse_g,
            Attribute::FOG_COORDINATE,
            Attribute::POSITION_X,
            Attribute::generic(1, 0),
        ]),
        ..ShaderInfo::default()
    };
    let mut runtime = RuntimeInfo {
        previous_stage_stores: varyings(&[Attribute::generic(1, 0)]),
        previous_stage_legacy_stores_mapping: (0..4)
            .map(|c| {
                (
                    Attribute::COLOR_FRONT_DIFFUSE_R.offset(c),
                    Attribute::generic(5, c),
                )
            })
            .collect(),
        ..RuntimeInfo::default()
    };
    runtime.generic_input_types[0] = AttributeType::Disabled;

    convert_legacy_to_generic(&mut prog, &mut info, &runtime);
    assert_eq!(
        info.legacy_stores_mapping,
        (0..4)
            .map(|c| (
                Attribute::COLOR_FRONT_DIFFUSE_R.offset(c),
                Attribute::generic(1, c)
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        prog.arg(store, 0),
        Value::Attribute(Attribute::generic(1, 1))
    );
    assert!(info.stores.get(Attribute::generic(1, 1)));
    assert_eq!(
        prog.arg(generic_store, 0),
        Value::Attribute(Attribute::generic(0, 0))
    );
    assert_eq!(
        prog.arg(color_load, 0),
        Value::Attribute(Attribute::generic(5, 1))
    );
    assert_eq!(
        prog.arg(fog_load, 0),
        Value::Attribute(Attribute::generic(0, 0))
    );
    assert!(info.loads.get(Attribute::generic(5, 1)));
    assert!(info.loads.get(Attribute::generic(0, 0)));
    assert_eq!(
        prog.arg(position_load, 0),
        Value::Attribute(Attribute::POSITION_X)
    );
}

#[test]
fn programs_without_legacy_attributes_are_unchanged() {
    let (mut prog, b) = program(1);
    let store = set_attr(&mut prog, b[0], Attribute::generic(3, 0), Value::F32(1.0));
    crate::opt::compute_post_order(&mut prog);
    let mut info = ShaderInfo {
        stores: varyings(&[Attribute::generic(3, 0)]),
        ..ShaderInfo::default()
    };
    let before = info.clone();
    convert_legacy_to_generic(&mut prog, &mut info, &RuntimeInfo::default());
    assert_eq!(info, before);
    assert_eq!(
        prog.arg(store, 0),
        Value::Attribute(Attribute::generic(3, 0))
    );
}

#[test]
fn legacy_attributes_past_the_free_generics_stay_unmapped() {
    let state = varyings(&[
        Attribute::COLOR_FRONT_DIFFUSE_R,
        Attribute::FOG_COORDINATE,
        Attribute::FIXED_FNC_TEXTURE_0_S.offset(4),
    ]);
    let mapping = generate_legacy_to_generic_mappings(&state, [Attribute::generic(7, 0)], &[]);
    assert_eq!(
        mapping,
        (0..4)
            .map(|c| (
                Attribute::COLOR_FRONT_DIFFUSE_R.offset(c),
                Attribute::generic(7, c)
            ))
            .collect::<Vec<_>>()
    );
    let mapping = generate_legacy_to_generic_mappings(
        &state,
        [
            Attribute::generic(7, 0),
            Attribute::generic(8, 0),
            Attribute::generic(9, 0),
        ],
        &[],
    );
    assert_eq!(
        mapping[4],
        (Attribute::FOG_COORDINATE, Attribute::generic(8, 0))
    );
    assert_eq!(
        mapping[5],
        (
            Attribute::FIXED_FNC_TEXTURE_0_S.offset(4),
            Attribute::generic(9, 0)
        )
    );
    assert_eq!(mapping.len(), 9);
}

#[test]
fn generated_passthrough_writes_the_emulated_layer() {
    let layer = Attribute::generic(3, 0);
    let source = ShaderInfo {
        stores: varyings(&[Attribute::POSITION_X, Attribute::generic(0, 2), layer]),
        emulated_layer: Some(layer),
        ..ShaderInfo::default()
    };

    let (prog, info) = generate_geometry_passthrough(&source, OutputTopology::TriangleStrip);
    assert_eq!(info.loads, source.stores);
    assert!(info.stores.get(Attribute::LAYER));
    assert!(!info.stores.get(layer));
    assert!(info.stores.generic(0));
    assert_eq!(prog.block_order.len(), 2);
    assert_eq!(prog.syntax_list.last(), Some(&SyntaxNode::Return));
    let (body, ret) = (prog.block_order[0], prog.block_order[1]);
    assert_eq!(prog.block(body).successors.as_slice(), &[ret]);
    assert_eq!(ops(&prog, ret), vec![Opcode::Epilogue]);
    let all = ops(&prog, body);
    assert_eq!(
        all.iter().filter(|&&op| op == Opcode::EmitVertex).count(),
        3
    );
    assert_eq!(all.last(), Some(&Opcode::EndPrimitive));
    let stores = attribute_stores(&prog, body);
    let per_vertex: Vec<Attribute> = (0..4)
        .map(|c| Attribute::generic(0, c))
        .chain((0..4).map(|c| Attribute::POSITION_X.offset(c)))
        .chain([Attribute::LAYER])
        .collect();
    assert_eq!(
        stores.iter().map(|s| s.0).collect::<Vec<_>>(),
        [per_vertex.clone(), per_vertex.clone(), per_vertex].concat()
    );
    let layer_load = stores[8].1.inst().unwrap();
    assert_eq!(prog.arg(layer_load, 0), Value::Attribute(layer));
    assert_eq!(output_topology_vertices(OutputTopology::PointList), 1);
    assert_eq!(output_topology_vertices(OutputTopology::LineStrip), 2);
}
