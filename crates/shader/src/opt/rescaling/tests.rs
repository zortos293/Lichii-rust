use super::*;
use crate::ir::{BlockId, SyntaxNode};
use crate::opt::compute_post_order;

fn program() -> (Program, BlockId) {
    let mut prog = Program::new();
    let block = prog.new_block();
    prog.block_order = vec![block];
    prog.syntax_list = vec![SyntaxNode::Block(block)];
    compute_post_order(&mut prog);
    (prog, block)
}

fn reference(prog: &mut Program, block: BlockId, value: InstId) -> InstId {
    prog.append(block, Opcode::Reference, &[Value::Inst(value)], 0)
}

fn used(prog: &Program, reference: InstId) -> InstId {
    prog.resolve(prog.arg(reference, 0))
        .inst()
        .expect("an instruction")
}

fn arg_inst(prog: &Program, id: InstId, index: usize) -> InstId {
    prog.resolve(prog.arg(id, index))
        .inst()
        .expect("an instruction")
}

fn texture_flags(ty: TextureType, descriptor_index: u16) -> u32 {
    TextureInstInfo {
        ty,
        descriptor_index,
        ..TextureInstInfo::default()
    }
    .to_flags()
}

fn frag_coord(prog: &mut Program, block: BlockId) -> InstId {
    prog.append(
        block,
        Opcode::GetAttribute,
        &[Value::Attribute(Attribute::POSITION_X), Value::U32(0)],
        0,
    )
}

#[test]
fn frag_coord_scales_down_in_fragment_programs() {
    let (mut prog, b) = program();
    let get = frag_coord(&mut prog, b);
    let r = reference(&mut prog, b, get);
    rescaling(&mut prog, ShaderStage::Fragment, ResolutionScale::X2);

    let scaled = used(&prog, r);
    assert_eq!(prog.op(scaled), Opcode::FPMul32);
    let coord = arg_inst(&prog, scaled, 0);
    assert_eq!(prog.op(coord), Opcode::GetAttribute);
    assert_eq!(prog.arg(coord, 0), Value::Attribute(Attribute::POSITION_X));
    assert_eq!(
        prog.op(arg_inst(&prog, scaled, 1)),
        Opcode::ResolutionDownFactor
    );

    let (mut prog, b) = program();
    let get = frag_coord(&mut prog, b);
    let r = reference(&mut prog, b, get);
    rescaling(&mut prog, ShaderStage::VertexB, ResolutionScale::X2);
    assert_eq!(used(&prog, r), get);
}

#[test]
fn point_size_scales_up() {
    let (mut prog, b) = program();
    let set = prog.append(
        b,
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POINT_SIZE),
            Value::F32(4.0),
            Value::U32(0),
        ],
        0,
    );
    rescaling(&mut prog, ShaderStage::VertexB, ResolutionScale::X2);
    let scaled = arg_inst(&prog, set, 1);
    assert_eq!(prog.op(scaled), Opcode::FPMul32);
    assert_eq!(prog.arg(scaled, 0), Value::F32(4.0));
    let up = arg_inst(&prog, scaled, 1);
    assert_eq!(prog.op(up), Opcode::FPRecip32);
    assert_eq!(
        prog.op(arg_inst(&prog, up, 0)),
        Opcode::ResolutionDownFactor
    );
}

fn query(prog: &mut Program, block: BlockId, ty: TextureType) -> InstId {
    prog.append(
        block,
        Opcode::ImageQueryDimensions,
        &[Value::Void, Value::U32(0), Value::U1(false)],
        texture_flags(ty, 3),
    )
}

#[test]
fn size_query_reports_the_guest_size() {
    let (mut prog, b) = program();
    let q = query(&mut prog, b, TextureType::Color2D);
    let r = reference(&mut prog, b, q);
    rescaling(&mut prog, ShaderStage::Compute, ResolutionScale::X2);

    let size = used(&prog, r);
    assert_eq!(prog.op(size), Opcode::CompositeConstructU32x4);
    let width = arg_inst(&prog, size, 0);
    assert_eq!(prog.op(width), Opcode::SelectU32);
    let test = arg_inst(&prog, width, 0);
    assert_eq!(prog.op(test), Opcode::IsTextureScaled);
    assert_eq!(prog.arg(test, 0), Value::U32(3));
    let divided = arg_inst(&prog, width, 1);
    assert_eq!(prog.op(divided), Opcode::UDiv32);
    assert_eq!(prog.arg(divided, 1), Value::U32(2));
    let extract = arg_inst(&prog, divided, 0);
    assert_eq!(prog.arg(extract, 1), Value::U32(0));
    assert_eq!(
        prog.op(arg_inst(&prog, extract, 0)),
        Opcode::ImageQueryDimensions
    );
    assert_eq!(prog.arg(width, 2), Value::Inst(extract));

    let (mut prog, b) = program();
    let q = query(&mut prog, b, TextureType::Color2D);
    let r = reference(&mut prog, b, q);
    rescaling(&mut prog, ShaderStage::Compute, ResolutionScale::X1_2);
    let width = arg_inst(&prog, used(&prog, r), 0);
    let shifted = arg_inst(&prog, width, 1);
    assert_eq!(prog.op(shifted), Opcode::ShiftLeftLogical32);
    assert_eq!(prog.arg(shifted, 1), Value::U32(1));

    let (mut prog, b) = program();
    let q = query(&mut prog, b, TextureType::Color3D);
    let r = reference(&mut prog, b, q);
    rescaling(&mut prog, ShaderStage::Compute, ResolutionScale::X2);
    assert_eq!(used(&prog, r), q);
}

fn fetch(prog: &mut Program, block: BlockId, ty: TextureType) -> InstId {
    let x = prog.append(block, Opcode::LaneId, &[], 0);
    let coords = prog.append(
        block,
        Opcode::CompositeConstructU32x2,
        &[Value::Inst(x), Value::U32(3)],
        0,
    );
    prog.append(
        block,
        Opcode::ImageFetch,
        &[
            Value::Void,
            Value::Inst(coords),
            Value::Void,
            Value::U32(0),
            Value::Void,
        ],
        texture_flags(ty, 1),
    )
}

#[test]
fn texel_fetch_scales_up_outside_fragment_programs() {
    let (mut prog, b) = program();
    let f = fetch(&mut prog, b, TextureType::Color2D);
    rescaling(&mut prog, ShaderStage::Compute, ResolutionScale::X2);

    let coords = arg_inst(&prog, f, 1);
    assert_eq!(prog.op(coords), Opcode::CompositeConstructU32x2);
    let x = arg_inst(&prog, coords, 0);
    assert_eq!(prog.op(x), Opcode::SelectU32);
    assert_eq!(prog.op(arg_inst(&prog, x, 0)), Opcode::IsTextureScaled);
    let multiplied = arg_inst(&prog, x, 1);
    assert_eq!(prog.op(multiplied), Opcode::IMul32);
    assert_eq!(prog.arg(multiplied, 1), Value::U32(2));
    assert_eq!(prog.arg(f, 2), Value::Void);

    let (mut prog, b) = program();
    let f = fetch(&mut prog, b, TextureType::Color1D);
    let coords = prog.arg(f, 1);
    rescaling(&mut prog, ShaderStage::Compute, ResolutionScale::X2);
    assert_eq!(prog.arg(f, 1), coords);
}

#[test]
fn fragment_texel_fetch_follows_the_fragment() {
    let (mut prog, b) = program();
    let f = fetch(&mut prog, b, TextureType::Color2D);
    rescaling(&mut prog, ShaderStage::Fragment, ResolutionScale::X2);

    let coords = arg_inst(&prog, f, 1);
    let y = arg_inst(&prog, coords, 1);
    assert_eq!(prog.op(y), Opcode::SelectU32);
    let texel = arg_inst(&prog, y, 1);
    assert_eq!(prog.op(texel), Opcode::ConvertU32F32);
    let deviation = arg_inst(&prog, texel, 0);
    assert_eq!(prog.op(deviation), Opcode::FPAdd32);
    let base = arg_inst(&prog, deviation, 0);
    assert_eq!(prog.arg(base, 1), Value::F32(2.0));
    let within = arg_inst(&prog, deviation, 1);
    let coord = arg_inst(&prog, within, 0);
    assert_eq!(prog.arg(coord, 0), Value::Attribute(Attribute::POSITION_Y));
}

#[test]
fn shuffled_frag_coord_scales_after_the_shuffle() {
    let (mut prog, b) = program();
    let get = frag_coord(&mut prog, b);
    let bits = prog.append(b, Opcode::BitCastU32F32, &[Value::Inst(get)], 0);
    let shuffle = prog.append(
        b,
        Opcode::ShuffleIndex,
        &[
            Value::Inst(bits),
            Value::U32(0),
            Value::U32(31),
            Value::U32(0),
        ],
        0,
    );
    let r = reference(&mut prog, b, shuffle);
    rescaling(&mut prog, ShaderStage::Fragment, ResolutionScale::X2);

    assert_eq!(prog.op(get), Opcode::GetAttribute);
    assert_eq!(prog.arg(bits, 0), Value::Inst(get));
    let result = used(&prog, r);
    assert_eq!(prog.op(result), Opcode::BitCastU32F32);
    let scaled = arg_inst(&prog, result, 0);
    assert_eq!(prog.op(scaled), Opcode::FPMul32);
    let value = arg_inst(&prog, scaled, 0);
    assert_eq!(prog.op(value), Opcode::BitCastF32U32);
    let copy = arg_inst(&prog, value, 0);
    assert_eq!(prog.op(copy), Opcode::ShuffleIndex);
    assert_eq!(prog.arg(copy, 0), Value::Inst(bits));
}

#[test]
fn unrelated_ir_is_left_alone() {
    let (mut prog, b) = program();
    let x = prog.append(b, Opcode::LaneId, &[], 0);
    let sum = prog.append(b, Opcode::IAdd32, &[Value::Inst(x), Value::U32(1)], 0);
    let color = prog.append(
        b,
        Opcode::GetAttribute,
        &[Value::Attribute(Attribute::generic(0, 0)), Value::U32(0)],
        0,
    );
    reference(&mut prog, b, sum);
    reference(&mut prog, b, color);
    let count = prog.insts_of(b).count();
    rescaling(&mut prog, ShaderStage::Fragment, ResolutionScale::X2);
    assert_eq!(prog.insts_of(b).count(), count);
    assert_eq!(prog.op(sum), Opcode::IAdd32);
    assert_eq!(prog.op(color), Opcode::GetAttribute);
}
