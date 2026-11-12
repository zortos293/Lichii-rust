//! Vertex position and blit parameter rewrites that depend on the render area, ported from
//! `position_pass.cpp`.

use crate::info::ShaderInfo;
use crate::ir::{Attribute, Emitter, FpControl, InstId, Opcode, Program, Value};
use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

/// Constant bank the Switch 2 blit parameter block lands in (bank 24, lowered to 17).
const BLIT_BANK: u32 = 17;

fn render_area_extent(ir: &mut Emitter<'_>, element: u32) -> Value {
    let area = ir.inst(Opcode::RenderArea, &[]);
    ir.composite_extract(area, element)
}

/// With the viewport transform disabled, a vertex program writes window coordinates; maps its
/// X and Y position outputs to normalized device coordinates over the render area.
pub fn position(env: &dyn ShaderEnvironment, prog: &mut Program, info: &mut ShaderInfo) {
    if env.stage() != ShaderStage::VertexB || env.viewport_transform_state() != 0 {
        return;
    }
    info.uses_render_area = true;
    let position_stores: Vec<(InstId, u32)> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| prog.op(id) == Opcode::SetAttribute)
        .filter_map(|id| match prog.arg(id, 0) {
            Value::Attribute(Attribute::POSITION_X) => Some((id, 0)),
            Value::Attribute(Attribute::POSITION_Y) => Some((id, 1)),
            _ => None,
        })
        .collect();
    for (id, element) in position_stores {
        let value = prog.arg(id, 1);
        let mut ir = Emitter::before(prog, id);
        let extent = render_area_extent(&mut ir, element);
        let recip = ir.fp_recip(extent);
        let scale = ir.fp_mul(recip, Value::F32(2.0), FpControl::default());
        let ndc = ir.fp_fma(value, scale, Value::F32(-1.0), FpControl::default());
        prog.set_arg(id, 1, ndc);
    }
}

fn in_blit_block(offset: u32) -> bool {
    offset < 0x20 || (0x30..0x4c).contains(&offset) || (0x50..0x60).contains(&offset)
}

fn is_nonzero_blit_word(offset: u32) -> bool {
    offset < 0x20 || matches!(offset, 0x38 | 0x3c | 0x58 | 0x5c)
}

/// The value the console's blit parameter block holds at `offset` for a blit covering the
/// whole render target.
fn blit_word(ir: &mut Emitter<'_>, offset: u32) -> Value {
    match offset {
        0x0 | 0x10 | 0x4 | 0x14 => {
            let extent = render_area_extent(ir, (offset & 4) / 4);
            let recip = ir.fp_recip(extent);
            ir.bitcast_u32(recip)
        }
        0x8 | 0xc => {
            let extent = render_area_extent(ir, (offset - 0x8) / 4);
            ir.bitcast_u32(extent)
        }
        0x18 | 0x1c | 0x38 | 0x3c => Value::U32(1.0f32.to_bits()),
        0x58 | 0x5c => {
            let extent = render_area_extent(ir, (offset - 0x58) / 4);
            let last = ir.fp_add(extent, Value::F32(-1.0), FpControl::default());
            ir.convert_f_to_i(32, false, last)
        }
        _ => Value::U32(0),
    }
}

/// True when every value reaching `value`, through up to four phis, is the bit pattern of a
/// float.
fn is_float_bits(prog: &Program, value: Value, depth: u32) -> bool {
    let Some(id) = prog.resolve(value).inst() else {
        return false;
    };
    match prog.op(id) {
        Opcode::BitCastU32F32 => true,
        Opcode::Phi if depth < 4 => prog
            .inst(id)
            .args()
            .iter()
            .all(|&arg| is_float_bits(prog, arg, depth + 1)),
        _ => false,
    }
}

/// Native fragment programs add an integer dither bias from words 0x10-0x18 of the blit bank to
/// their output bits. The console leaves those zero, but the synthesized block holds the texel
/// size and UV scale there, so the bias is dropped.
fn drop_dither_bias(prog: &mut Program) {
    let casts: Vec<InstId> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| prog.op(id) == Opcode::BitCastF32U32)
        .collect();
    for cast in casts {
        let Some(arithmetic) = prog.resolve(prog.arg(cast, 0)).inst() else {
            continue;
        };
        let op = prog.op(arithmetic);
        if !matches!(op, Opcode::IAdd32 | Opcode::ISub32) {
            continue;
        }
        let first_bias_arg = usize::from(op == Opcode::ISub32);
        for bias_arg in first_bias_arg..2 {
            let float_bits = prog.arg(arithmetic, 1 - bias_arg);
            let Some(bias) = prog.resolve(prog.arg(arithmetic, bias_arg)).inst() else {
                continue;
            };
            if prog.op(bias) == Opcode::GetCbufU32
                && prog.arg(bias, 0).imm_u32() == Some(BLIT_BANK)
                && matches!(prog.arg(bias, 1).imm_u32(), Some(0x10 | 0x14 | 0x18))
                && is_float_bits(prog, float_bits, 0)
            {
                prog.set_arg(cast, 0, float_bits);
                break;
            }
        }
    }
}

/// Switch 2 fullscreen blit programs read a parameter block the console provides from constant
/// bank 24 (lowered to 17, which games never bind there). Replaces those reads with the values
/// the block holds for a blit covering the render area. A program reading bank 17 outside the
/// block, or only words the block leaves zero, reads a table the game bound and is left alone.
pub fn blit_constants(env: &dyn ShaderEnvironment, prog: &mut Program, info: &mut ShaderInfo) {
    let stage = env.stage();
    if !matches!(stage, ShaderStage::VertexB | ShaderStage::Fragment) {
        return;
    }
    let reads: Vec<(InstId, u32)> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| {
            matches!(
                prog.op(id),
                Opcode::GetCbufU32 | Opcode::GetCbufF32 | Opcode::GetCbufU32x2
            ) && prog.arg(id, 0).imm_u32() == Some(BLIT_BANK)
        })
        .filter_map(|id| Some((id, prog.arg(id, 1).imm_u32()?)))
        .collect();
    let mut reads_nonzero_word = false;
    for &(id, offset) in &reads {
        let pair = prog.op(id) == Opcode::GetCbufU32x2;
        if !in_blit_block(offset) || (pair && !in_blit_block(offset + 4)) {
            return;
        }
        reads_nonzero_word |=
            is_nonzero_blit_word(offset) || (pair && is_nonzero_blit_word(offset + 4));
    }
    if !reads_nonzero_word {
        return;
    }
    if stage == ShaderStage::Fragment && env.arch() == ShaderArch::Ampere {
        drop_dither_bias(prog);
    }
    for (id, offset) in reads {
        let op = prog.op(id);
        let mut ir = Emitter::before(prog, id);
        let value = match op {
            Opcode::GetCbufU32x2 => {
                let low = blit_word(&mut ir, offset);
                let high = blit_word(&mut ir, offset + 4);
                ir.composite_construct(&[low, high])
            }
            Opcode::GetCbufF32 => {
                let word = blit_word(&mut ir, offset);
                ir.bitcast_f32(word)
            }
            _ => blit_word(&mut ir, offset),
        };
        prog.replace_uses_with(id, value);
    }
    info.uses_render_area = true;
}
