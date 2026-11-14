//! Geometry passthrough lowering and legacy varying rewriting (translate_program.cpp).
//!
//! A passthrough geometry program copies each input vertex to its output. Hosts without
//! passthrough geometry shaders get the copy as explicit IR, and a vertex program that writes a
//! layer the host cannot write before geometry gets a generated geometry program that does it.

use crate::info::{ShaderInfo, VaryingState};
use crate::ir::{
    Attribute, BlockId, Emitter, ICompare, InstId, Opcode, Program, Reg, SyntaxNode, Value,
};
use crate::maxwell::header::OutputTopology;
use crate::opt::{compute_post_order, ssa_rewrite};
use crate::profile::HostTranslateInfo;
use crate::runtime::{AttributeType, RuntimeInfo};

const CULLED_POSITION: [f32; 4] = [2.0, 2.0, 2.0, 1.0];

/// Vertices in one primitive of a geometry program's output topology.
pub const fn output_topology_vertices(topology: OutputTopology) -> u32 {
    match topology {
        OutputTopology::PointList => 1,
        OutputTopology::LineStrip => 2,
        OutputTopology::TriangleStrip => 3,
    }
}

/// Emits one primitive that copies `output_vertices` input vertices to the output. With
/// `viewport_mask_reg`, the lowest viewport the mask selects becomes the viewport index and an
/// empty mask moves the position outside the clip volume.
fn emit_geometry_passthrough(
    e: &mut Emitter,
    output_vertices: u32,
    passthrough_mask: &VaryingState,
    passthrough_position: bool,
    passthrough_layer: Option<Attribute>,
    viewport_mask_reg: Option<Reg>,
) {
    let mut culled = e.imm1(false);
    let mut viewport = e.imm32(0);
    if let Some(reg) = viewport_mask_reg {
        let mask = e.get_reg(reg);
        let negated = e.ineg(mask);
        let lowest_bit = e.bitwise_and(mask, negated);
        culled = e.icompare(ICompare::Equal, mask, Value::U32(0));
        let msb = e.find_u_msb(lowest_bit);
        viewport = e.select(culled, Value::U32(0), msb);
    }
    for vertex in 0..output_vertices {
        let vertex = Value::U32(vertex);
        for generic in (0..Attribute::NUM_GENERICS).filter(|&g| passthrough_mask.generic(g)) {
            for component in 0..4 {
                let attr = Attribute::generic(generic, component);
                let value = e.get_attribute(attr, vertex);
                e.set_attribute(attr, value, Value::U32(0));
            }
        }
        if passthrough_position {
            for (component, culled_value) in (0..4).zip(CULLED_POSITION) {
                let attr = Attribute::POSITION_X.offset(component);
                let mut value = e.get_attribute(attr, vertex);
                if viewport_mask_reg.is_some() {
                    value = e.select(culled, Value::F32(culled_value), value);
                }
                e.set_attribute(attr, value, Value::U32(0));
            }
        }
        if viewport_mask_reg.is_some() {
            let index = e.bitcast_f32(viewport);
            e.set_attribute(Attribute::VIEWPORT_INDEX, index, Value::U32(0));
        }
        if let Some(layer) = passthrough_layer {
            let value = e.get_attribute(layer, Value::U32(0));
            e.set_attribute(Attribute::LAYER, value, Value::U32(0));
        }
        e.emit_vertex(Value::U32(0));
    }
    e.end_primitive(Value::U32(0));
}

fn insts(prog: &Program) -> impl Iterator<Item = InstId> + '_ {
    prog.block_order.iter().flat_map(|&b| prog.insts_of(b))
}

/// The highest general purpose register the program neither reads nor writes.
fn find_free_register(prog: &Program) -> Option<Reg> {
    let mut used = [false; Reg::RZ.0 as usize];
    for id in insts(prog) {
        if matches!(prog.op(id), Opcode::GetRegister | Opcode::SetRegister)
            && let Value::Reg(reg) = prog.arg(id, 0)
            && let Some(used) = used.get_mut(reg.index())
        {
            *used = true;
        }
    }
    (0..used.len())
        .rev()
        .find(|&index| !used[index])
        .map(|index| Reg(index as u16))
}

/// Turns the program's viewport mask stores into writes of a free register, which starts at
/// viewport 0's bit, and returns the register; none when the program stores no mask.
fn lower_viewport_mask(prog: &mut Program) -> Option<Reg> {
    let stores: Vec<InstId> = insts(prog)
        .filter(|&id| {
            prog.op(id) == Opcode::SetAttribute
                && prog.arg(id, 0) == Value::Attribute(Attribute::VIEWPORT_MASK)
        })
        .collect();
    if stores.is_empty() {
        return None;
    }
    let Some(reg) = find_free_register(prog) else {
        crate::fallback::note("geometry", "viewport_mask_no_free_register");
        return None;
    };
    let &entry = prog.block_order.first()?;
    for id in stores {
        let value = prog.arg(id, 1);
        let mut e = Emitter::before(prog, id);
        let mask = e.bitcast_u32(value);
        e.set_reg(reg, mask);
        prog.remove(id);
    }
    let mut e = match prog.block(entry).first() {
        Some(first) => Emitter::before(prog, first),
        None => Emitter::new(prog, entry),
    };
    e.set_reg(reg, Value::U32(1));
    Some(reg)
}

/// Lowers a passthrough geometry program for a host without passthrough geometry shaders:
/// copies the attributes `info.passthrough` names to the output before every epilogue, as
/// `output_vertices` vertices. Run it before the SSA rewrite.
pub fn lower_geometry_passthrough(
    prog: &mut Program,
    info: &ShaderInfo,
    output_vertices: u32,
    host: &HostTranslateInfo,
) {
    let viewport_mask_reg = if host.support_viewport_mask {
        None
    } else {
        lower_viewport_mask(prog)
    };
    let epilogues: Vec<InstId> = insts(prog)
        .filter(|&id| prog.op(id) == Opcode::Epilogue)
        .collect();
    let passthrough_position = info.passthrough.any_component(Attribute::POSITION_X);
    for id in epilogues {
        emit_geometry_passthrough(
            &mut Emitter::before(prog, id),
            output_vertices,
            &info.passthrough,
            passthrough_position,
            None,
            viewport_mask_reg,
        );
    }
}

/// Maps the legacy attributes `state` uses to generics, in attribute order: to the generic the
/// previous stage mapped it to, or else to the next of `unused_generics`. An attribute left
/// without a generic stays unmapped.
fn generate_legacy_to_generic_mappings(
    state: &VaryingState,
    unused_generics: impl IntoIterator<Item = Attribute>,
    previous_stage_mapping: &[(Attribute, Attribute)],
) -> Vec<(Attribute, Attribute)> {
    let mut unused_generics = unused_generics.into_iter();
    let previous = |attr: Attribute| {
        previous_stage_mapping
            .iter()
            .find(|&&(legacy, _)| legacy == attr)
            .map(|&(_, generic)| generic)
    };
    let colors = (0..4).map(|i| (Attribute::COLOR_FRONT_DIFFUSE_R.offset(i * 4), 4));
    let fog = std::iter::once((Attribute::FOG_COORDINATE, 1));
    let textures = (0..Attribute::NUM_FIXED_FNC_TEXTURES)
        .map(|i| (Attribute::FIXED_FNC_TEXTURE_0_S.offset(i * 4), 4));
    let mut mapping = Vec::new();
    for (attr, count) in colors.chain(fog).chain(textures) {
        let used = if count == 1 {
            state.get(attr)
        } else {
            state.any_component(attr)
        };
        if !used {
            continue;
        }
        if previous(attr).is_some() {
            mapping.extend((0..count).filter_map(|i| {
                let generic = previous(attr.offset(i));
                if generic.is_none() {
                    crate::fallback::note("attribute", "legacy_component_unmapped");
                }
                Some((attr.offset(i), generic?))
            }));
        } else if let Some(generic) = unused_generics.next() {
            mapping.extend((0..count).map(|i| (attr.offset(i), generic.offset(i))));
        } else {
            crate::fallback::note("attribute", "legacy_no_free_generic");
        }
    }
    mapping
}

/// Rewrites every `op` access of a mapped legacy attribute to its generic, marking the generic
/// in `state`. `mapping` is sorted by legacy attribute.
fn remap_legacy_accesses(
    prog: &mut Program,
    op: Opcode,
    mapping: &[(Attribute, Attribute)],
    state: &mut VaryingState,
) {
    for i in 0..prog.post_order.len() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            if prog.op(id) != op {
                continue;
            }
            let Value::Attribute(attr) = prog.arg(id, 0) else {
                continue;
            };
            if !attr.is_legacy() {
                continue;
            }
            let Ok(index) = mapping.binary_search_by_key(&attr, |&(legacy, _)| legacy) else {
                continue;
            };
            let generic = mapping[index].1;
            state.set(generic, true);
            prog.set_arg(id, 0, Value::Attribute(generic));
        }
    }
}

/// Moves the legacy attributes (colours, fog, fixed function texture coordinates) a program
/// stores and loads to free generics, for hosts that have no legacy varyings. Stores take
/// generics the program does not store and record the mapping in
/// `info.legacy_stores_mapping`; loads follow the previous stage's mapping.
pub fn convert_legacy_to_generic(prog: &mut Program, info: &mut ShaderInfo, runtime: &RuntimeInfo) {
    if info.stores.legacy() {
        let stores = &info.stores;
        let unused = (0..Attribute::NUM_GENERICS)
            .filter(|&i| !stores.generic(i))
            .map(|i| Attribute::generic(i, 0));
        info.legacy_stores_mapping = generate_legacy_to_generic_mappings(stores, unused, &[]);
        remap_legacy_accesses(
            prog,
            Opcode::SetAttribute,
            &info.legacy_stores_mapping,
            &mut info.stores,
        );
    }
    if info.loads.legacy() {
        let loads = &info.loads;
        let unused = (0..Attribute::NUM_GENERICS)
            .filter(|&i| {
                !runtime.previous_stage_stores.generic(i)
                    || !loads.generic(i)
                    || runtime.generic_input_types[usize::from(i)] == AttributeType::Disabled
            })
            .map(|i| Attribute::generic(i, 0));
        let mapping = generate_legacy_to_generic_mappings(
            loads,
            unused,
            &runtime.previous_stage_legacy_stores_mapping,
        );
        remap_legacy_accesses(prog, Opcode::GetAttribute, &mapping, &mut info.loads);
    }
}

/// Builds a geometry program that passes every vertex of `source`'s outputs through and writes
/// the layer from `source`'s emulated layer attribute, for hosts that cannot write the layer
/// before the geometry stage. Its output vertex count is
/// `output_topology_vertices(output_topology)`.
pub fn generate_geometry_passthrough(
    source: &ShaderInfo,
    output_topology: OutputTopology,
) -> (Program, ShaderInfo) {
    let mut info = ShaderInfo {
        loads: source.stores,
        stores: source.stores,
        ..ShaderInfo::default()
    };
    info.stores.set(Attribute::LAYER, true);
    if let Some(layer) = source.emulated_layer {
        info.stores.set(layer, false);
    }
    let mut prog = Program::new();
    let body: BlockId = prog.new_block();
    let ret = prog.new_block();
    emit_geometry_passthrough(
        &mut Emitter::new(&mut prog, body),
        output_topology_vertices(output_topology),
        &info.stores,
        true,
        source.emulated_layer,
        None,
    );
    Emitter::new(&mut prog, ret).epilogue();
    prog.add_branch(body, ret);
    prog.block_order = vec![body, ret];
    prog.syntax_list = vec![
        SyntaxNode::Block(body),
        SyntaxNode::Block(ret),
        SyntaxNode::Return,
    ];
    compute_post_order(&mut prog);
    ssa_rewrite(&mut prog);
    (prog, info)
}

#[cfg(test)]
mod tests;
