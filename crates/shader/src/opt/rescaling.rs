//! Resolution rescaling (rescaling_pass.cpp): makes a program rendering to a rescaled target
//! see guest-sized coordinates. Fragment coordinates scale down, point sizes up, size queries of
//! rescaled textures report guest sizes, and integer texel coordinates scale up (or, in
//! fragment programs, follow the fragment within the scaled texel).

use lichii_common::settings::ResolutionScale;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use crate::ShaderStage;
use crate::ir::{
    Attribute, Emitter, FpControl, InstId, Opcode, Program, TextureInstInfo, TextureType, Value,
};

/// The scale as `up_scale >> down_shift` (the C++ resolution info).
#[derive(Clone, Copy)]
struct Scale {
    up_scale: u32,
    down_shift: u32,
    up_factor: f32,
    down_factor: f32,
}

impl Scale {
    fn new(scale: ResolutionScale) -> Scale {
        let (num, den) = scale.ratio();
        Scale {
            up_scale: num,
            down_shift: den.trailing_zeros(),
            up_factor: num as f32 / den as f32,
            down_factor: den as f32 / num as f32,
        }
    }
}

fn is_rescalable(ty: TextureType) -> bool {
    matches!(
        ty,
        TextureType::Color2D | TextureType::ColorArray2D | TextureType::Color2DRect
    )
}

fn flags(prog: &Program, id: InstId) -> TextureInstInfo {
    TextureInstInfo::from_flags(prog.inst(id).flags)
}

fn attribute(prog: &Program, id: InstId) -> Option<Attribute> {
    match prog.arg(id, 0) {
        Value::Attribute(attribute) => Some(attribute),
        _ => None,
    }
}

fn copy_before(prog: &mut Program, id: InstId) -> Value {
    let inst = prog.inst(id);
    let (op, flags) = (inst.op, inst.flags);
    let args: SmallVec<[Value; 5]> = SmallVec::from_slice(inst.args());
    Value::Inst(prog.insert_before(id, op, &args, flags))
}

/// Rescales the program for rendering at `scale`; run it only when the scale is not 1x.
pub fn rescaling(prog: &mut Program, stage: ShaderStage, scale: ResolutionScale) {
    let scale = Scale::new(scale);
    let is_fragment = stage == ShaderStage::Fragment;
    let mut shuffled_frag_coords = FxHashSet::default();
    if is_fragment {
        for_each_inst(prog, |prog, id| {
            mark_shuffled_frag_coord(prog, id, &mut shuffled_frag_coords)
        });
    }
    for_each_inst(prog, |prog, id| {
        visit(prog, id, is_fragment, scale, &shuffled_frag_coords)
    });
}

fn for_each_inst(prog: &mut Program, mut f: impl FnMut(&mut Program, InstId)) {
    for i in 0..prog.post_order.len() {
        let mut cur = prog.block(prog.post_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            f(prog, id);
        }
    }
}

/// A fragment coordinate shuffled across lanes as integer bits stays unscaled until after the
/// shuffle, which then scales its result up.
fn mark_shuffled_frag_coord(prog: &mut Program, id: InstId, marked: &mut FxHashSet<InstId>) {
    if !matches!(
        prog.op(id),
        Opcode::ShuffleIndex | Opcode::ShuffleUp | Opcode::ShuffleDown | Opcode::ShuffleButterfly
    ) {
        return;
    }
    let Some(bitcast) = prog.resolve(prog.arg(id, 0)).inst() else {
        return;
    };
    if prog.op(bitcast) != Opcode::BitCastU32F32 {
        return;
    }
    let Some(source) = prog.resolve(prog.arg(bitcast, 0)).inst() else {
        return;
    };
    if prog.op(source) != Opcode::GetAttribute
        || !matches!(
            attribute(prog, source),
            Some(Attribute::POSITION_X | Attribute::POSITION_Y)
        )
    {
        return;
    }
    marked.insert(source);
    let shuffle = copy_before(prog, id);
    let fp = FpControl::default();
    let mut ir = Emitter::before(prog, id);
    let value = ir.inst(Opcode::BitCastF32U32, &[shuffle]);
    let down_factor = ir.inst(Opcode::ResolutionDownFactor, &[]);
    let up_factor = ir.fp_recip(down_factor);
    let scaled = ir.fp_mul(value, up_factor, fp);
    let bits = ir.inst(Opcode::BitCastU32F32, &[scaled]);
    prog.replace_uses_with(id, bits);
}

fn visit(
    prog: &mut Program,
    id: InstId,
    is_fragment: bool,
    scale: Scale,
    shuffled_frag_coords: &FxHashSet<InstId>,
) {
    let fp = FpControl::default();
    match prog.op(id) {
        Opcode::GetAttribute
            if is_fragment
                && !shuffled_frag_coords.contains(&id)
                && matches!(
                    attribute(prog, id),
                    Some(Attribute::POSITION_X | Attribute::POSITION_Y)
                ) =>
        {
            let attribute = prog.arg(id, 0);
            let mut ir = Emitter::before(prog, id);
            let down_factor = ir.inst(Opcode::ResolutionDownFactor, &[]);
            let frag_coord = ir.inst(Opcode::GetAttribute, &[attribute, Value::U32(0)]);
            let downscaled = ir.fp_mul(frag_coord, down_factor, fp);
            prog.replace_uses_with(id, downscaled);
        }
        Opcode::InterpolateAtOffset if is_fragment => {
            let offset = [prog.arg(id, 1), prog.arg(id, 2)];
            let mut ir = Emitter::before(prog, id);
            let down_factor = ir.inst(Opcode::ResolutionDownFactor, &[]);
            let up_factor = ir.fp_recip(down_factor);
            let [x, y] = offset.map(|v| ir.fp_mul(v, up_factor, fp));
            prog.set_arg(id, 1, x);
            prog.set_arg(id, 2, y);
        }
        Opcode::SetAttribute if attribute(prog, id) == Some(Attribute::POINT_SIZE) => {
            let point_size = prog.arg(id, 1);
            let mut ir = Emitter::before(prog, id);
            let down_factor = ir.inst(Opcode::ResolutionDownFactor, &[]);
            let up_factor = ir.fp_recip(down_factor);
            let upscaled = ir.fp_mul(point_size, up_factor, fp);
            prog.set_arg(id, 1, upscaled);
        }
        Opcode::ImageQueryDimensions => patch_query_dimensions(prog, id, scale),
        Opcode::ImageFetch => {
            let ty = flags(prog, id).ty;
            if !is_rescalable(ty) {
                return;
            }
            let is_scaled = scaled_test(prog, id, Opcode::IsTextureScaled);
            if is_fragment {
                sub_scale_coords(prog, id, is_scaled, scale);
            } else {
                scale_coords(
                    prog,
                    id,
                    1,
                    is_scaled,
                    scale,
                    ty == TextureType::ColorArray2D,
                );
            }
            scale_coords(prog, id, 2, is_scaled, scale, false);
        }
        Opcode::ImageRead => {
            let ty = flags(prog, id).ty;
            if !is_rescalable(ty) {
                return;
            }
            let is_scaled = scaled_test(prog, id, Opcode::IsImageScaled);
            if is_fragment {
                sub_scale_coords(prog, id, is_scaled, scale);
            } else {
                scale_coords(
                    prog,
                    id,
                    1,
                    is_scaled,
                    scale,
                    ty == TextureType::ColorArray2D,
                );
            }
        }
        _ => {}
    }
}

/// `IsTextureScaled` or `IsImageScaled` of the instruction's descriptor, before it.
fn scaled_test(prog: &mut Program, id: InstId, test: Opcode) -> Value {
    let index = u32::from(flags(prog, id).descriptor_index);
    Emitter::before(prog, id).inst(test, &[Value::U32(index)])
}

fn upscale(ir: &mut Emitter<'_>, is_scaled: Value, value: Value, scale: Scale) -> Value {
    let mut scaled = value;
    if scale.up_scale != 1 {
        scaled = ir.imul(scaled, Value::U32(scale.up_scale));
    }
    if scale.down_shift != 0 {
        scaled = ir.shift_right_arithmetic(scaled, Value::U32(scale.down_shift));
    }
    ir.select(is_scaled, scaled, value)
}

fn downscale(ir: &mut Emitter<'_>, is_scaled: Value, value: Value, scale: Scale) -> Value {
    let mut scaled = value;
    if scale.down_shift != 0 {
        scaled = ir.shift_left_logical(scaled, Value::U32(scale.down_shift));
    }
    if scale.up_scale != 1 {
        scaled = ir.inst(Opcode::UDiv32, &[scaled, Value::U32(scale.up_scale)]);
    }
    ir.select(is_scaled, scaled, value)
}

/// The scaled texel the fragment at `attribute` falls in, from guest texel `value`.
fn sub_scale(
    ir: &mut Emitter<'_>,
    is_scaled: Value,
    value: Value,
    attribute: Attribute,
    scale: Scale,
) -> Value {
    let fp = FpControl::default();
    let up_factor = Value::F32(scale.up_factor);
    let texel = ir.convert_i_to_f(32, 32, false, value, fp);
    let base = ir.fp_mul(texel, up_factor, fp);
    let frag_coord = ir.inst(
        Opcode::GetAttribute,
        &[Value::Attribute(attribute), Value::U32(0)],
    );
    let guest_coord = ir.fp_mul(frag_coord, Value::F32(scale.down_factor), fp);
    let guest_floor = ir.fp_floor(guest_coord, fp);
    let floor = ir.fp_mul(up_factor, guest_floor, fp);
    let neg_floor = ir.fp_neg(floor);
    let within = ir.fp_add(frag_coord, neg_floor, fp);
    let deviation = ir.fp_add(base, within, fp);
    let scaled = ir.convert_f_to_i(32, false, deviation);
    ir.select(is_scaled, scaled, value)
}

fn patch_query_dimensions(prog: &mut Program, id: InstId, scale: Scale) {
    if !is_rescalable(flags(prog, id).ty) {
        return;
    }
    let is_scaled = scaled_test(prog, id, Opcode::IsTextureScaled);
    let query = copy_before(prog, id);
    let mut ir = Emitter::before(prog, id);
    let width = ir.composite_extract(query, 0);
    let width = downscale(&mut ir, is_scaled, width, scale);
    let height = ir.composite_extract(query, 1);
    let height = downscale(&mut ir, is_scaled, height, scale);
    let depth = ir.composite_extract(query, 2);
    let mips = ir.composite_extract(query, 3);
    let replacement = ir.composite_construct(&[width, height, depth, mips]);
    prog.replace_uses_with(id, replacement);
}

/// Scales X and Y of the integer coordinates (or offset) at `arg`, keeping the third component
/// (an array layer) when `has_layer`.
fn scale_coords(
    prog: &mut Program,
    id: InstId,
    arg: usize,
    is_scaled: Value,
    scale: Scale,
    has_layer: bool,
) {
    let composite = prog.arg(id, arg);
    if composite == Value::Void {
        return;
    }
    let mut ir = Emitter::before(prog, id);
    let x = ir.composite_extract(composite, 0);
    let x = upscale(&mut ir, is_scaled, x, scale);
    let y = ir.composite_extract(composite, 1);
    let y = upscale(&mut ir, is_scaled, y, scale);
    let scaled = if has_layer {
        let layer = ir.composite_extract(composite, 2);
        ir.composite_construct(&[x, y, layer])
    } else {
        ir.composite_construct(&[x, y])
    };
    prog.set_arg(id, arg, scaled);
}

fn sub_scale_coords(prog: &mut Program, id: InstId, is_scaled: Value, scale: Scale) {
    let ty = flags(prog, id).ty;
    let coords = prog.arg(id, 1);
    let mut ir = Emitter::before(prog, id);
    let x = ir.composite_extract(coords, 0);
    let x = sub_scale(&mut ir, is_scaled, x, Attribute::POSITION_X, scale);
    let y = ir.composite_extract(coords, 1);
    let y = sub_scale(&mut ir, is_scaled, y, Attribute::POSITION_Y, scale);
    let scaled = if ty == TextureType::ColorArray2D {
        let layer = ir.composite_extract(coords, 2);
        ir.composite_construct(&[x, y, layer])
    } else {
        ir.composite_construct(&[x, y])
    };
    prog.set_arg(id, 1, scaled);
}

#[cfg(test)]
mod tests;
