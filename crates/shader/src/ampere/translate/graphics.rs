//! Attributes, interpolation, internal stage buffers, geometry output, special registers and barriers.

use super::{TResult, Translator, control};
use crate::ampere::decode::Form;
use crate::ampere::opcode::Opcode;
use crate::ir::{Attribute, FpControl, ICompare, Opcode as Ir, Patch, Value};
use crate::maxwell::header::PixelImap;
use crate::{ShaderStage, TranslateError};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Ald => ald(t),
        Ast => ast(t),
        Ipa => ipa(t),
        Isberd => isberd(t),
        Isbewr => isbewr(t),
        Out => out(t),
        Pixld => pixld(t),
        Al2p => al2p(t),
        S2r | S2ur => s2r(t),
        Cs2r => cs2r(t),
        Csmtest => csmtest(t),
        Lepc => lepc(t),
        Bar => bar(t),
        _ => Err(t.unsupported()),
    }
}

/// The patch for word "index", if it names one.
fn patch(t: &Translator<'_>, index: u32) -> Result<Patch, TranslateError> {
    if index >= u32::from(Patch::COMPONENT_0.0 + Patch::NUM_COMPONENTS) {
        return Err(t.unsupported());
    }
    Ok(Patch(index as u8))
}

/// An ALD/AST access: the first attribute word (address at 40..50), the element count (74..76)
/// and whether it names patch words (76), which only tessellation programs have. Physical
/// addressing (77) needs no translation: AL2P maps attribute addresses onto themselves.
fn attribute_access(t: &Translator<'_>) -> Result<(u32, u16, bool), TranslateError> {
    let address = t.insn.bits(40, 10) as u32;
    let is_patch = t.insn.bit(76);
    let tessellation = matches!(
        t.env.stage(),
        ShaderStage::TessellationControl | ShaderStage::TessellationEval
    );
    if !address.is_multiple_of(4) || (is_patch && !tessellation) {
        return Err(t.unsupported());
    }
    Ok((address / 4, t.insn.bits(74, 2) as u16 + 1, is_patch))
}

/// Byte address of element "element" of an indexed access: the index register plus the
/// instruction's attribute address.
fn indexed_address(t: &mut Translator<'_>, base: Value, word: u32, element: u16) -> Value {
    let offset = (word + u32::from(element)) * 4;
    if offset == 0 {
        base
    } else {
        t.ir.iadd(base, Value::U32(offset))
    }
}

/// ALD: attributes of the vertex in R[32..40] into R[16..24] onwards. An output read (.O, 79)
/// is a tessellation control program's read of its own patch words, or a tessellation
/// evaluation program's read of its tessellation coordinates.
fn ald(t: &mut Translator<'_>) -> TResult {
    let (word, count, is_patch) = attribute_access(t)?;
    let output = t.insn.bit(79);
    let stage = t.env.stage();
    let tess_coord = stage == ShaderStage::TessellationEval
        && word >= u32::from(Attribute::TESS_EVAL_POINT_U.0)
        && word + u32::from(count) <= u32::from(Attribute::TESS_EVAL_POINT_V.0) + 1;
    let valid_output = if is_patch {
        output == (stage == ShaderStage::TessellationControl)
    } else {
        !output || tess_coord
    };
    let index_reg = t.insn.src0();
    if !valid_output || (is_patch && !index_reg.is_zero()) {
        return Err(t.unsupported());
    }
    let dest = t.insn.dst();
    let vertex = t.x(t.insn.src1());
    if index_reg.is_zero() {
        for element in 0..count {
            let index = word + u32::from(element);
            let value = if is_patch {
                let patch = patch(t, index)?;
                t.ir.get_patch(patch)
            } else {
                t.ir.get_attribute(Attribute(index as u16), vertex)
            };
            t.set_f(dest.offset(element), value);
        }
        return Ok(());
    }
    let base = t.x(index_reg);
    for element in 0..count {
        let address = indexed_address(t, base, word, element);
        let value = t.ir.get_attribute_indexed(address, vertex);
        t.set_f(dest.offset(element), value);
    }
    Ok(())
}

/// AST: R[32..40] onwards into attributes of the vertex in R[64..72]. Patch words are written by
/// tessellation control programs only. A lone store to a[0x5c], a per-primitive shading rate
/// hint with no Maxwell meaning, is dropped, which keeps full-rate shading.
fn ast(t: &mut Translator<'_>) -> TResult {
    const SHADING_RATE_HINT: u32 = 0x5c / 4;
    let (word, count, is_patch) = attribute_access(t)?;
    let index_reg = t.insn.src0();
    if t.insn.bit(79)
        || (is_patch && (t.env.stage() != ShaderStage::TessellationControl || !index_reg.is_zero()))
    {
        return Err(t.unsupported());
    }
    if !is_patch && word == SHADING_RATE_HINT && count == 1 && index_reg.is_zero() {
        return Ok(());
    }
    let src = t.insn.src1();
    let vertex = t.x(t.insn.src2());
    if index_reg.is_zero() {
        for element in 0..count {
            let index = word + u32::from(element);
            let value = t.f(src.offset(element));
            if is_patch {
                let patch = patch(t, index)?;
                t.ir.set_patch(patch, value);
            } else {
                t.ir.set_attribute(Attribute(index as u16), value, vertex);
            }
        }
        return Ok(());
    }
    let base = t.x(index_reg);
    for element in 0..count {
        let address = indexed_address(t, base, word, element);
        let value = t.f(src.offset(element));
        t.ir.set_attribute_indexed(address, value, vertex);
    }
    Ok(())
}

/// IPA: the attribute word at 64..72, the sample location at 76..78 (default, centroid or
/// .OFFSET by Rb at 32..40) and the frequency at 78..80 (pass or constant; NAK never emits the
/// third, state). The indexed form adds the uniform register at 24..30 to the attribute's byte
/// address.
///
/// IPA returns a perspective input divided by the clip-space w, and programs multiply it by the
/// reciprocal of IPA's position w themselves, so the value is multiplied by the position's w as
/// Maxwell's IPA does. `InterpolateAtOffset` divides by w at the offset itself. A constant input
/// does not vary, so .OFFSET reads it as it is.
fn ipa(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let (location, frequency) = (insn.bits(76, 2), insn.bits(78, 2));
    if location > 2 || frequency > 1 || !insn.pred_dst0().is_true() {
        return Err(t.unsupported());
    }
    let word = insn.bits(64, 8) as u16;
    let attribute = Attribute(word);
    let index = insn.usrc0();
    let indexed = t.form == Form::UregIndexed && !index.is_zero();
    let value = if location == 2 && frequency == 0 {
        if indexed || t.env.stage() != ShaderStage::Fragment {
            return Err(t.unsupported());
        }
        let offset = t.x(insn.src1());
        interpolate_at_offset(t, attribute, offset)?
    } else if indexed {
        let base = t.x(index);
        let address = t.ir.iadd(base, Value::U32(u32::from(word) * 4));
        let value = t.ir.get_attribute_indexed(address, Value::U32(0));
        divide_indexed_by_w(t, address, value)
    } else {
        let value = t.ir.get_attribute(attribute, Value::U32(0));
        if interpolates_in_perspective(t, attribute) {
            let w = t.ir.get_attribute(Attribute::POSITION_W, Value::U32(0));
            t.ir.fp_mul(value, w, FpControl::default())
        } else {
            value
        }
    };
    t.set_f(insn.dst(), value);
    Ok(())
}

/// Divides an indexed read by w when the attribute `address` selects at run time interpolates
/// in perspective, from a mask of the program's 64 16-byte attribute slots.
fn divide_indexed_by_w(t: &mut Translator<'_>, address: Value, value: Value) -> Value {
    let mut mask = 0u64;
    for slot in 0..64u16 {
        if (0..4).any(|c| interpolates_in_perspective(t, Attribute(slot * 4 + c))) {
            mask |= 1 << slot;
        }
    }
    if mask == 0 {
        return value;
    }
    let w = t.ir.get_attribute(Attribute::POSITION_W, Value::U32(0));
    let slot = t.ir.shift_right_logical(address, Value::U32(4));
    let slot = t.ir.bitwise_and(slot, Value::U32(63));
    let low = t.ir.icompare(ICompare::ULessThan, slot, Value::U32(32));
    let half = t.ir.select(
        low,
        Value::U32(mask as u32),
        Value::U32((mask >> 32) as u32),
    );
    let bit = t.ir.bitwise_and(slot, Value::U32(31));
    let bit = t.ir.bit_field_extract(half, bit, Value::U32(1), false);
    let perspective = t.ir.icompare(ICompare::NotEqual, bit, Value::U32(0));
    let factor = t.ir.select(perspective, w, Value::F32(1.0));
    t.ir.fp_mul(value, factor, FpControl::default())
}

/// Whether IPA returns `attribute` divided by the clip-space w: a generic the program header
/// maps as perspective (or leaves unused), or a legacy input. The host interpolates it
/// perspective-correct, so the translation multiplies it by `gl_FragCoord.w` (1 / w).
fn interpolates_in_perspective(t: &Translator<'_>, attribute: Attribute) -> bool {
    let Some(generic) = attribute.generic_index() else {
        return attribute.is_legacy();
    };
    let imap = control::program_header(t)
        .unwrap_or_default()
        .ps()
        .generic_input_map(usize::from(generic))
        .into_iter()
        .find(|&imap| imap != PixelImap::Unused)
        .unwrap_or(PixelImap::Unused);
    matches!(imap, PixelImap::Perspective | PixelImap::Unused)
}

/// IPA.OFFSET's offset holds x in the low and y in the high halfword, signed with 12 fraction
/// bits (NAK's `load_barycentric_offset`). Generic inputs interpolate at it, the position's x
/// and y move by it, and system values that are constant over the primitive read as they are.
/// The position's z and w and the point coordinate would need the hardware's interpolation.
fn interpolate_at_offset(
    t: &mut Translator<'_>,
    attribute: Attribute,
    offset: Value,
) -> Result<Value, TranslateError> {
    let half = |t: &mut Translator<'_>, shift: u32| {
        let half =
            t.ir.bit_field_extract(offset, Value::U32(shift), Value::U32(16), true);
        let half =
            t.ir.convert_i_to_f(32, 32, true, half, FpControl::default());
        t.ir.fp_mul(half, Value::F32(1.0 / 4096.0), FpControl::default())
    };
    let shift = match attribute {
        _ if attribute.is_generic() => {
            let (x, y) = (half(t, 0), half(t, 16));
            return Ok(t.ir.inst(
                Ir::InterpolateAtOffset,
                &[Value::Attribute(attribute), x, y],
            ));
        }
        Attribute::POSITION_X => 0,
        Attribute::POSITION_Y => 16,
        Attribute::PRIMITIVE_ID
        | Attribute::LAYER
        | Attribute::VIEWPORT_INDEX
        | Attribute::FRONT_FACE => return Ok(t.ir.get_attribute(attribute, Value::U32(0))),
        _ => return Err(t.unsupported()),
    };
    let delta = half(t, shift);
    let value = t.ir.get_attribute(attribute, Value::U32(0));
    Ok(t.ir.fp_add(value, delta, FpControl::default()))
}

fn is_task_or_mesh(t: &Translator<'_>) -> bool {
    matches!(t.env.stage(), ShaderStage::Task | ShaderStage::Mesh)
}

/// Bits 64..96 a task or mesh stage buffer access may set: .32 (75), the region (76..78, 0 the
/// map, 3 the attributes), .SKEW (78) and .O (79).
const ISBE_MODIFIERS: u64 = 0xf800;

/// Byte address of a task or mesh stage buffer access: Ra (24..32) plus the offset at 40..64.
/// Addresses are linear bytes in the region; .SKEW needs no translation.
fn isbe_address(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let region = t.insn.bits(76, 2);
    if t.insn.bits(64, 32) & !ISBE_MODIFIERS != 0 || (region != 0 && region != 3) {
        return Err(t.unsupported());
    }
    let base = t.x(t.insn.src0());
    Ok(t.ir.iadd(base, Value::U32(t.insn.bits(40, 24) as u32)))
}

/// ISBERD. Task and mesh programs read their input attribute region with it. Geometry and
/// tessellation programs use the plain form to turn a vertex index into the attribute handle
/// ALD takes, which is the index itself.
fn isberd(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if !is_task_or_mesh(t) {
        let vertex_indexed = matches!(
            t.env.stage(),
            ShaderStage::Geometry
                | ShaderStage::TessellationControl
                | ShaderStage::TessellationEval
        );
        if !vertex_indexed || insn.bits(32, 64) != 0 || insn.src0().is_zero() {
            return Err(t.unsupported());
        }
        let value = t.x(insn.src0());
        t.set_x(insn.dst(), value);
        return Ok(());
    }
    if insn.bits(76, 2) != 3 || insn.bit(79) {
        return Err(t.unsupported());
    }
    let address = isbe_address(t)?;
    let value = t.ir.inst(Ir::IsbeRead, &[address]);
    t.set_x(insn.dst(), value);
    Ok(())
}

/// ISBEWR: a task or mesh program writes R[32..40] to its output attribute region (32-bit) or
/// its map region (32-bit or U8).
fn isbewr(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if !is_task_or_mesh(t) || !insn.bit(79) {
        return Err(t.unsupported());
    }
    let address = isbe_address(t)?;
    let op = match (insn.bits(76, 2) == 3, insn.bit(75)) {
        (true, true) => Ir::IsbeWriteAttribute,
        (true, false) => return Err(t.unsupported()),
        (false, true) => Ir::IsbeWriteMapU32,
        (false, false) => Ir::IsbeWriteMapU8,
    };
    let value = t.x(insn.src1());
    t.ir.inst(op, &[address, value]);
    Ok(())
}

/// OUT: geometry vertex emission (type 1 at 78..80), primitive cut (2) or both (3) on the
/// stream in the second source; type 0 finishes the output and needs nothing. The result is
/// the next output handle, which the IR does not model, so it reads 0.
fn out(t: &mut Translator<'_>) -> TResult {
    if t.env.stage() != ShaderStage::Geometry {
        return Err(t.unsupported());
    }
    let kind = t.insn.bits(78, 2);
    if kind != 0 {
        let stream = t.src_b()?;
        let stream = t.ir.bitwise_and(stream, Value::U32(0b11));
        if kind & 1 != 0 {
            t.ir.emit_vertex(stream);
        }
        if kind & 2 != 0 {
            t.ir.end_primitive(stream);
        }
    }
    t.set_x(t.insn.dst(), Value::U32(0));
    Ok(())
}

/// PIXLD: only the sample index (.MY_INDEX, 3 at 78..81) without a predicate output.
fn pixld(t: &mut Translator<'_>) -> TResult {
    const MY_INDEX: u64 = 3;
    if t.insn.bits(78, 3) != MY_INDEX || !t.insn.pred_dst0().is_true() {
        return Err(t.unsupported());
    }
    let sample = t.ir.inst(Ir::SampleId, &[]);
    t.set_x(t.insn.dst(), sample);
    Ok(())
}

/// AL2P: the physical address of an attribute, which is the attribute address itself: Ra plus
/// the signed offset at 40..51. Only the single-word form (74..76 zero) is supported.
fn al2p(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(74, 2) != 0 {
        return Err(t.unsupported());
    }
    let offset = Value::U32(t.insn.sbits(40, 11) as u32);
    let index = t.x(t.insn.src0());
    let result = t.ir.iadd(index, offset);
    t.set_x(t.insn.dst(), result);
    Ok(())
}

/// Special register "sr", numbered as on Maxwell. Registers the IR does not model (clocks,
/// performance counters, machine ids) read as 0, as Maxwell's S2R does. SR_VARIABLE_RATE
/// (132) reads as a 1x1 fragment: without variable-rate shading, width and height are 1.
fn special_register(t: &mut Translator<'_>, sr: u64) -> Value {
    const VARIABLE_RATE_1X1: u32 = 0x0001_0100;
    match sr {
        0 => t.ir.lane_id(),
        17 => t.ir.inst(Ir::InvocationId, &[]),
        18 => {
            let y = t.ir.inst(Ir::YDirection, &[]);
            t.ir.bitcast_u32(y)
        }
        19 => {
            let helper = t.ir.inst(Ir::IsHelperInvocation, &[]);
            t.ir.select(helper, Value::U32(u32::MAX), Value::U32(0))
        }
        29 => t.ir.inst(Ir::InvocationInfo, &[]),
        30 => t.ir.inst(Ir::WScaleFactorXY, &[]),
        31 => t.ir.inst(Ir::WScaleFactorZ, &[]),
        32 => {
            let tid = t.ir.local_invocation_id();
            let [x, y, z] = [0, 1, 2].map(|i| t.ir.composite_extract(tid, i));
            let xy = t.ir.bit_field_insert(x, y, Value::U32(16), Value::U32(8));
            t.ir.bit_field_insert(xy, z, Value::U32(26), Value::U32(6))
        }
        33..=35 => {
            let tid = t.ir.local_invocation_id();
            t.ir.composite_extract(tid, (sr - 33) as u32)
        }
        37..=39 => {
            let ctaid = t.ir.workgroup_id();
            t.ir.composite_extract(ctaid, (sr - 37) as u32)
        }
        56 => t.ir.inst(Ir::SubgroupEqMask, &[]),
        57 => t.ir.inst(Ir::SubgroupLtMask, &[]),
        58 => t.ir.inst(Ir::SubgroupLeMask, &[]),
        59 => t.ir.inst(Ir::SubgroupGtMask, &[]),
        60 => t.ir.inst(Ir::SubgroupGeMask, &[]),
        132 => Value::U32(VARIABLE_RATE_1X1),
        _ => Value::U32(0),
    }
}

/// S2R and S2UR: the special register at 72..80 into a register or a uniform register.
fn s2r(t: &mut Translator<'_>) -> TResult {
    let value = special_register(t, t.insn.bits(72, 8));
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// CS2R: the special register at 72..80 (255 is SRZ, zero) into a register, or with .64 (80)
/// that register and the next into a pair.
fn cs2r(t: &mut Translator<'_>) -> TResult {
    const SRZ: u64 = 255;
    let (sr, dest) = (t.insn.bits(72, 8), t.insn.dst());
    let registers: &[u64] = if t.insn.bit(80) { &[0, 1] } else { &[0] };
    for &i in registers {
        let value = if sr == SRZ {
            Value::U32(0)
        } else {
            special_register(t, sr + i)
        };
        t.set_x(dest.offset(i as u16), value);
    }
    Ok(())
}

/// CSMTEST tests coarse shading state the emulator does not model: both predicate outputs
/// (81..84, 84..87) read false, so the "@P EXIT" that follows never leaves early.
fn csmtest(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(87, 4) != 7 || t.insn.bits(16, 16) != 0 {
        return Err(t.unsupported());
    }
    for pred in [t.insn.pred_dst0(), t.insn.pred_dst1()] {
        t.ir.set_pred(pred, Value::U1(false));
    }
    Ok(())
}

/// LEPC: the address of this instruction into a register pair.
fn lepc(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(24, 8) != 0 || t.insn.bits(32, 64) != 0 {
        return Err(t.unsupported());
    }
    let dest = t.insn.dst();
    let address = t.base + u64::from(t.pc);
    t.ir.prog.reads_start_address = true;
    t.set_x(dest, Value::U32(address as u32));
    t.set_x(dest.offset(1), Value::U32((address >> 32) as u32));
    Ok(())
}

/// BAR.SYNC 0 over the whole workgroup; .DEFER_BLOCKING (80) changes nothing here. Named
/// barriers, thread counts, arrivals and reductions are not supported.
fn bar(t: &mut Translator<'_>) -> TResult {
    const DEFER_BLOCKING: u64 = 1 << 16;
    if t.form != Form::Fixed
        || t.insn.bits(32, 32) != 0
        || t.insn.bits(64, 32) & !DEFER_BLOCKING != 0
    {
        return Err(t.unsupported());
    }
    t.ir.barrier();
    Ok(())
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::ResolutionScale;

    use super::super::tests::{TestEnv, encode, translate, translate_stage};
    use crate::ShaderStage::{self, *};
    use crate::ampere::decode::Form;
    use crate::ampere::instruction::Instruction;
    use crate::ampere::opcode::Opcode;
    use crate::backend::Bindings;
    use crate::backend::spirv::tests::validate;
    use crate::profile::HostTranslateInfo;
    use crate::translate::TranslateOptions;
    use crate::{Profile, RuntimeInfo};

    const RZ: u128 = 0xff;
    const PT_DST: u128 = 7 << 81;

    fn dst(reg: u128) -> u128 {
        reg << 16
    }

    fn ra(reg: u128) -> u128 {
        reg << 24
    }

    fn rb(reg: u128) -> u128 {
        reg << 32
    }

    fn rc(reg: u128) -> u128 {
        reg << 64
    }

    fn fixed(op: Opcode, fields: u128) -> Instruction {
        encode(op, Form::Fixed, fields)
    }

    fn run(stage: ShaderStage, inst: Instruction) -> String {
        translate_stage(stage, &[inst]).unwrap()
    }

    fn rejects(stage: ShaderStage, inst: Instruction) {
        assert!(
            translate_stage(stage, &[inst]).is_err(),
            "{inst:x?} in {stage:?}"
        );
    }

    /// ALD with a[address] (40..50) and "count" components.
    fn ald(address: u128, count: u128, fields: u128) -> Instruction {
        fixed(Opcode::Ald, (address << 40) | ((count - 1) << 74) | fields)
    }

    /// AST with a[address] (40..50) and "count" components.
    fn ast(address: u128, count: u128, fields: u128) -> Instruction {
        fixed(Opcode::Ast, (address << 40) | ((count - 1) << 74) | fields)
    }

    #[test]
    fn ald_reads_vertex_attributes() {
        let inst = ald(0x80, 2, dst(4) | ra(RZ) | rb(3));
        assert_eq!(
            run(Geometry, inst),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = GetAttribute Generic0X, %0\n\
             \x20 %2 = BitCastU32F32 %1\n\
             \x20 SetRegister R4, %2\n\
             \x20 %3 = GetAttribute Generic0Y, %0\n\
             \x20 %4 = BitCastU32F32 %3\n\
             \x20 SetRegister R5, %4\n"
        );
    }

    #[test]
    fn ald_indexed_adds_the_attribute_address() {
        let inst = ald(0x80, 2, dst(1) | ra(2) | rb(RZ));
        assert_eq!(
            run(VertexB, inst),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x80\n\
             \x20 %2 = GetAttributeIndexed %1, #0x0\n\
             \x20 %3 = BitCastU32F32 %2\n\
             \x20 SetRegister R1, %3\n\
             \x20 %4 = IAdd32 %0, #0x84\n\
             \x20 %5 = GetAttributeIndexed %4, #0x0\n\
             \x20 %6 = BitCastU32F32 %5\n\
             \x20 SetRegister R2, %6\n"
        );
        let physical = ald(0, 1, dst(1) | ra(2) | rb(RZ) | (1 << 77));
        assert_eq!(
            run(TessellationEval, physical),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetAttributeIndexed %0, #0x0\n\
             \x20 %2 = BitCastU32F32 %1\n\
             \x20 SetRegister R1, %2\n"
        );
    }

    #[test]
    fn ald_patches_and_tessellation_coordinates() {
        let patch = 1 << 76;
        let output = 1 << 79;
        let control = ald(0x18, 1, dst(1) | ra(RZ) | rb(RZ) | patch | output);
        assert_eq!(
            run(TessellationControl, control),
            "{0}:\n  %0 = GetPatch Patch0\n  %1 = BitCastU32F32 %0\n  SetRegister R1, %1\n"
        );
        let eval = ald(0x1c, 1, dst(1) | ra(RZ) | rb(RZ) | patch);
        assert!(run(TessellationEval, eval).contains("GetPatch Patch1"));
        let coords = ald(0x2f0, 2, dst(1) | ra(RZ) | rb(RZ) | output);
        assert_eq!(
            run(TessellationEval, coords),
            "{0}:\n\
             \x20 %0 = GetAttribute TessCoordU, #0x0\n\
             \x20 %1 = BitCastU32F32 %0\n\
             \x20 SetRegister R1, %1\n\
             \x20 %2 = GetAttribute TessCoordV, #0x0\n\
             \x20 %3 = BitCastU32F32 %2\n\
             \x20 SetRegister R2, %3\n"
        );
    }

    #[test]
    fn ald_invalid_forms_are_unsupported() {
        let plain = ra(RZ) | rb(RZ);
        let patch = 1 << 76;
        let output = 1 << 79;
        rejects(VertexB, ald(0x82, 1, plain));
        rejects(VertexB, ald(0x18, 1, plain | patch));
        rejects(TessellationEval, ald(0x18, 1, ra(2) | rb(RZ) | patch));
        rejects(TessellationControl, ald(0x18, 1, plain | patch));
        rejects(TessellationEval, ald(0x18, 1, plain | patch | output));
        rejects(TessellationControl, ald(0x3fc, 1, plain | patch | output));
        rejects(VertexB, ald(0x80, 1, plain | output));
        rejects(TessellationEval, ald(0x2f4, 2, plain | output));
    }

    #[test]
    fn ast_writes_attributes_and_patches() {
        let position = ast(0x7c, 1, ra(RZ) | rb(1) | rc(RZ));
        assert_eq!(
            run(VertexB, position),
            "{0}:\n\
             \x20 %0 = GetRegister R1\n\
             \x20 %1 = BitCastF32U32 %0\n\
             \x20 SetAttribute PositionW, %1, #0x0\n"
        );
        let patch = ast(4, 1, ra(RZ) | rb(2) | rc(RZ) | (1 << 76));
        assert_eq!(
            run(TessellationControl, patch),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = BitCastF32U32 %0\n\
             \x20 SetPatch TessLod1, %1\n"
        );
        let per_vertex = ast(0x80, 1, ra(RZ) | rb(2) | rc(3));
        assert!(run(TessellationControl, per_vertex).contains("SetAttribute Generic0X, %2, %0"));
    }

    #[test]
    fn ast_indexed_and_shading_rate_hint() {
        let indexed = ast(0x10, 2, ra(4) | rb(RZ) | rc(RZ));
        assert_eq!(
            run(VertexB, indexed),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = IAdd32 %0, #0x10\n\
             \x20 SetAttributeIndexed %1, #0.0f, #0x0\n\
             \x20 %2 = IAdd32 %0, #0x14\n\
             \x20 SetAttributeIndexed %2, #0.0f, #0x0\n"
        );
        let hint = ast(0x5c, 1, ra(RZ) | rb(1) | rc(RZ));
        assert_eq!(run(VertexB, hint), "{0}:\n");
    }

    #[test]
    fn ast_invalid_forms_are_unsupported() {
        let plain = ra(RZ) | rb(1) | rc(RZ);
        rejects(VertexB, ast(0x7c, 1, plain | (1 << 79)));
        rejects(TessellationEval, ast(4, 1, plain | (1 << 76)));
        rejects(
            TessellationControl,
            ast(4, 1, ra(2) | rb(1) | rc(RZ) | (1 << 76)),
        );
        rejects(VertexB, ast(0x7e, 1, plain));
    }

    fn ipa_to(reg: u128, word: u128, fields: u128) -> Instruction {
        fixed(
            Opcode::Ipa,
            dst(reg) | rb(RZ) | (word << 64) | PT_DST | fields,
        )
    }

    fn ipa(word: u128, fields: u128) -> Instruction {
        ipa_to(1, word, fields)
    }

    #[test]
    fn ipa_reads_the_interpolated_value() {
        let expected = "{0}:\n\
                        \x20 %0 = GetAttribute Generic0Y, #0x0\n\
                        \x20 %1 = GetAttribute PositionW, #0x0\n\
                        \x20 %2 = FPMul32 %0, %1\n\
                        \x20 %3 = BitCastU32F32 %2\n\
                        \x20 SetRegister R1, %3\n";
        assert_eq!(run(Fragment, ipa(33, 0)), expected);
        let centroid = 1 << 76;
        let constant = 1 << 78;
        assert_eq!(run(Fragment, ipa(33, centroid)), expected);
        assert_eq!(run(Fragment, ipa(33, constant)), expected);
        assert!(run(Fragment, ipa(31, 0)).contains("GetAttribute PositionW, #0x0"));
        assert!(run(Fragment, ipa(255, 0)).contains("GetAttribute FrontFace, #0x0"));
    }

    #[test]
    fn ipa_indexed_by_a_uniform_register() {
        let fields = dst(1) | ra(5) | rb(RZ) | (32 << 64) | PT_DST;
        let indexed = encode(Opcode::Ipa, Form::UregIndexed, fields);
        assert_eq!(
            run(Fragment, indexed),
            "{0}:\n\
             \x20 %0 = GetRegister UR5\n\
             \x20 %1 = IAdd32 %0, #0x80\n\
             \x20 %2 = GetAttributeIndexed %1, #0x0\n\
             \x20 %3 = GetAttribute PositionW, #0x0\n\
             \x20 %4 = ShiftRightLogical32 %1, #0x4\n\
             \x20 %5 = BitwiseAnd32 %4, #0x3f\n\
             \x20 %6 = ULessThan %5, #0x20\n\
             \x20 %7 = SelectU32 %6, #0xffffff00, #0x3ff4fff\n\
             \x20 %8 = BitwiseAnd32 %5, #0x1f\n\
             \x20 %9 = BitFieldUExtract %7, %8, #0x1\n\
             \x20 %10 = INotEqual %9, #0x0\n\
             \x20 %11 = SelectF32 %10, %3, #1.0f\n\
             \x20 %12 = FPMul32 %2, %11\n\
             \x20 %13 = BitCastU32F32 %12\n\
             \x20 SetRegister R1, %13\n"
        );
        let urz = encode(Opcode::Ipa, Form::UregIndexed, fields | ra(63));
        assert!(run(Fragment, urz).starts_with("{0}:\n  %0 = GetAttribute Generic0X, #0x0\n"));
    }

    fn ipa_at_offset_to(reg: u128, word: u128, fields: u128) -> Instruction {
        fixed(
            Opcode::Ipa,
            dst(reg) | rb(4) | (word << 64) | PT_DST | (2 << 76) | fields,
        )
    }

    fn ipa_at_offset(word: u128, fields: u128) -> Instruction {
        ipa_at_offset_to(1, word, fields)
    }

    #[test]
    fn ipa_offset_interpolates_generic_inputs() {
        assert_eq!(
            run(Fragment, ipa_at_offset(33, 0)),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = BitFieldSExtract %0, #0x0, #0x10\n\
             \x20 %2 = ConvertF32S32 %1\n\
             \x20 %3 = FPMul32 %2, #0.00024414063f\n\
             \x20 %4 = BitFieldSExtract %0, #0x10, #0x10\n\
             \x20 %5 = ConvertF32S32 %4\n\
             \x20 %6 = FPMul32 %5, #0.00024414063f\n\
             \x20 %7 = InterpolateAtOffset Generic0Y, %3, %6\n\
             \x20 %8 = BitCastU32F32 %7\n\
             \x20 SetRegister R1, %8\n"
        );
    }

    #[test]
    fn ipa_offset_moves_the_position_by_the_offset() {
        assert_eq!(
            run(Fragment, ipa_at_offset(29, 0)),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = BitFieldSExtract %0, #0x10, #0x10\n\
             \x20 %2 = ConvertF32S32 %1\n\
             \x20 %3 = FPMul32 %2, #0.00024414063f\n\
             \x20 %4 = GetAttribute PositionY, #0x0\n\
             \x20 %5 = FPAdd32 %4, %3\n\
             \x20 %6 = BitCastU32F32 %5\n\
             \x20 SetRegister R1, %6\n"
        );
        assert!(run(Fragment, ipa_at_offset(28, 0)).contains("BitFieldSExtract %0, #0x0, #0x10"));
    }

    #[test]
    fn ipa_offset_reads_flat_system_values_as_they_are() {
        for (word, name) in [
            (24, "PrimitiveId"),
            (25, "Layer"),
            (26, "ViewportIndex"),
            (255, "FrontFace"),
        ] {
            assert_eq!(
                run(Fragment, ipa_at_offset(word, 0)),
                format!(
                    "{{0}}:\n\
                     \x20 %0 = GetRegister R4\n\
                     \x20 %1 = GetAttribute {name}, #0x0\n\
                     \x20 %2 = BitCastU32F32 %1\n\
                     \x20 SetRegister R1, %2\n"
                )
            );
        }
    }

    #[test]
    fn ipa_offset_reads_constant_inputs_as_they_are() {
        assert_eq!(
            run(Fragment, ipa_at_offset(33, 1 << 78)),
            run(Fragment, ipa(33, 0))
        );
    }

    #[test]
    fn ipa_other_modes_are_unsupported() {
        rejects(Fragment, ipa(32, 3 << 76));
        rejects(VertexB, ipa_at_offset(32, 0));
        for word in [30, 31, 184] {
            rejects(Fragment, ipa_at_offset(word, 0));
        }
        let indexed = dst(1) | ra(5) | rb(4) | (32 << 64) | PT_DST | (2 << 76);
        rejects(Fragment, encode(Opcode::Ipa, Form::UregIndexed, indexed));
        rejects(Fragment, ipa(32, 2 << 78));
        rejects(Fragment, ipa(32, 3 << 78));
        rejects(Fragment, Instruction(ipa(32, 0).0 ^ PT_DST));
    }

    const ATTR: u128 = 3 << 76;
    const B32: u128 = 1 << 75;
    const OUT: u128 = 1 << 79;
    const SKEW: u128 = 1 << 78;

    fn isberd(fields: u128) -> Instruction {
        fixed(Opcode::Isberd, dst(1) | fields)
    }

    fn isbewr(fields: u128) -> Instruction {
        fixed(Opcode::Isbewr, rb(3) | OUT | B32 | fields)
    }

    #[test]
    fn isberd_reads_the_task_and_mesh_attribute_region() {
        let read = isberd(ra(2) | (0x10 << 40) | ATTR | B32);
        assert_eq!(
            run(Mesh, read),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x10\n\
             \x20 %2 = IsbeRead %1\n\
             \x20 SetRegister R1, %2\n"
        );
        let workgroup = isberd(ra(RZ) | ATTR | B32 | SKEW);
        assert_eq!(
            run(Task, workgroup),
            "{0}:\n  %0 = IAdd32 #0x0, #0x0\n  %1 = IsbeRead %0\n  SetRegister R1, %1\n"
        );
    }

    #[test]
    fn isberd_is_the_vertex_index_in_geometry_and_tessellation() {
        for stage in [Geometry, TessellationControl, TessellationEval] {
            assert_eq!(
                run(stage, isberd(ra(2))),
                "{0}:\n  %0 = GetRegister R2\n  SetRegister R1, %0\n"
            );
        }
    }

    #[test]
    fn isberd_invalid_forms_are_unsupported() {
        rejects(Fragment, isberd(ra(2) | ATTR | B32));
        rejects(Compute, isberd(ra(2)));
        rejects(Geometry, isberd(ra(RZ)));
        rejects(Geometry, isberd(ra(2) | ATTR));
        rejects(Mesh, isberd(ra(2) | B32));
        rejects(Mesh, isberd(ra(2) | ATTR | B32 | OUT));
        rejects(Mesh, isberd(ra(2) | ATTR | (1 << 74)));
        rejects(Mesh, isberd(ra(2) | ATTR | B32 | (1 << 87)));
    }

    #[test]
    fn isbewr_writes_attributes_and_the_map() {
        let attribute = isbewr(ra(2) | (4 << 40) | ATTR);
        assert_eq!(
            run(Task, attribute),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x4\n\
             \x20 %2 = GetRegister R3\n\
             \x20 IsbeWriteAttribute %1, %2\n"
        );
        let count = isbewr(ra(RZ) | (3 << 40));
        assert_eq!(
            run(Mesh, count),
            "{0}:\n\
             \x20 %0 = IAdd32 #0x0, #0x3\n\
             \x20 %1 = GetRegister R3\n\
             \x20 IsbeWriteMapU32 %0, %1\n"
        );
        let index = Instruction(isbewr(ra(4) | (5 << 40)).0 ^ B32);
        assert!(run(Mesh, index).contains("IsbeWriteMapU8 %1, %2"));
    }

    #[test]
    fn isbewr_invalid_forms_are_unsupported() {
        rejects(Fragment, isbewr(ra(2) | ATTR));
        rejects(Geometry, isbewr(ra(2) | ATTR));
        rejects(Mesh, Instruction(isbewr(ra(2) | ATTR).0 ^ OUT));
        rejects(Mesh, Instruction(isbewr(ra(2) | ATTR).0 ^ B32));
        rejects(Mesh, isbewr(ra(2) | (1 << 76)));
    }

    #[test]
    fn out_emits_and_cuts() {
        let both = encode(
            Opcode::Out,
            Form::RegReg,
            dst(1) | ra(2) | rb(4) | (3 << 78),
        );
        assert_eq!(
            run(Geometry, both),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = BitwiseAnd32 %0, #0x3\n\
             \x20 EmitVertex %1\n\
             \x20 EndPrimitive %1\n\
             \x20 SetRegister R1, #0x0\n"
        );
        let emit = encode(
            Opcode::Out,
            Form::ImmReg,
            dst(3) | ra(2) | (1 << 32) | (1 << 78),
        );
        assert_eq!(
            run(Geometry, emit),
            "{0}:\n  %0 = BitwiseAnd32 #0x1, #0x3\n  EmitVertex %0\n  SetRegister R3, #0x0\n"
        );
        let cut = encode(
            Opcode::Out,
            Form::RegReg,
            dst(RZ) | ra(2) | rb(RZ) | (2 << 78),
        );
        assert_eq!(
            run(Geometry, cut),
            "{0}:\n  %0 = BitwiseAnd32 #0x0, #0x3\n  EndPrimitive %0\n"
        );
        let finish = encode(Opcode::Out, Form::RegReg, dst(RZ) | ra(2) | rb(RZ));
        assert_eq!(run(Geometry, finish), "{0}:\n");
        rejects(VertexB, both);
    }

    #[test]
    fn pixld_reads_the_sample_index() {
        let my_index = fixed(Opcode::Pixld, dst(6) | (3 << 78) | PT_DST);
        assert_eq!(
            run(Fragment, my_index),
            "{0}:\n  %0 = SampleId\n  SetRegister R6, %0\n"
        );
        rejects(Fragment, fixed(Opcode::Pixld, dst(6) | (1 << 78) | PT_DST));
        rejects(Fragment, fixed(Opcode::Pixld, dst(6) | (3 << 78)));
    }

    #[test]
    fn al2p_adds_the_signed_offset() {
        let inst = fixed(Opcode::Al2p, dst(2) | ra(5) | (0x7fc << 40));
        assert_eq!(
            run(TessellationControl, inst),
            "{0}:\n  %0 = GetRegister R5\n  %1 = IAdd32 %0, #0xfffffffc\n  SetRegister R2, %1\n"
        );
        let output = fixed(Opcode::Al2p, dst(2) | ra(5) | (0x80 << 40) | (1 << 79));
        assert!(run(TessellationControl, output).contains("IAdd32 %0, #0x80"));
        rejects(TessellationControl, fixed(Opcode::Al2p, (1 << 74) | ra(5)));
    }

    fn s2r(sr: u128) -> String {
        run(Compute, fixed(Opcode::S2r, dst(1) | (sr << 72)))
    }

    #[test]
    fn s2r_simple_registers() {
        for (sr, op) in [
            (0, "LaneId"),
            (17, "InvocationId"),
            (29, "InvocationInfo"),
            (30, "WScaleFactorXY"),
            (31, "WScaleFactorZ"),
            (56, "SubgroupEqMask"),
            (57, "SubgroupLtMask"),
            (58, "SubgroupLeMask"),
            (59, "SubgroupGtMask"),
            (60, "SubgroupGeMask"),
        ] {
            assert_eq!(
                s2r(sr),
                format!("{{0}}:\n  %0 = {op}\n  SetRegister R1, %0\n")
            );
        }
    }

    #[test]
    fn s2r_thread_and_workgroup_ids() {
        assert_eq!(
            s2r(32),
            "{0}:\n\
             \x20 %0 = LocalInvocationId\n\
             \x20 %1 = CompositeExtractU32x3 %0, #0x0\n\
             \x20 %2 = CompositeExtractU32x3 %0, #0x1\n\
             \x20 %3 = CompositeExtractU32x3 %0, #0x2\n\
             \x20 %4 = BitFieldInsert %1, %2, #0x10, #0x8\n\
             \x20 %5 = BitFieldInsert %4, %3, #0x1a, #0x6\n\
             \x20 SetRegister R1, %5\n"
        );
        assert_eq!(
            s2r(34),
            "{0}:\n  %0 = LocalInvocationId\n  %1 = CompositeExtractU32x3 %0, #0x1\n  SetRegister R1, %1\n"
        );
        assert_eq!(
            s2r(39),
            "{0}:\n  %0 = WorkgroupId\n  %1 = CompositeExtractU32x3 %0, #0x2\n  SetRegister R1, %1\n"
        );
    }

    #[test]
    fn s2r_fragment_state_and_constants() {
        assert_eq!(
            s2r(19),
            "{0}:\n\
             \x20 %0 = IsHelperInvocation\n\
             \x20 %1 = SelectU32 %0, #0xffffffff, #0x0\n\
             \x20 SetRegister R1, %1\n"
        );
        assert_eq!(
            s2r(18),
            "{0}:\n  %0 = YDirection\n  %1 = BitCastU32F32 %0\n  SetRegister R1, %1\n"
        );
        assert_eq!(s2r(132), "{0}:\n  SetRegister R1, #0x10100\n");
        assert_eq!(s2r(80), "{0}:\n  SetRegister R1, #0x0\n");
        assert_eq!(s2r(28), "{0}:\n  SetRegister R1, #0x0\n");
    }

    #[test]
    fn s2ur_writes_a_uniform_register() {
        let inst = fixed(Opcode::S2ur, dst(4) | (37 << 72));
        assert_eq!(
            run(Compute, inst),
            "{0}:\n  %0 = WorkgroupId\n  %1 = CompositeExtractU32x3 %0, #0x0\n  SetRegister UR4, %1\n"
        );
    }

    #[test]
    fn cs2r_zero_pair_and_single_register() {
        let srz = fixed(Opcode::Cs2r, dst(2) | (0xff << 72) | (1 << 80));
        assert_eq!(
            run(Compute, srz),
            "{0}:\n  SetRegister R2, #0x0\n  SetRegister R3, #0x0\n"
        );
        let mask = fixed(Opcode::Cs2r, dst(2) | (57 << 72));
        assert_eq!(
            run(Compute, mask),
            "{0}:\n  %0 = SubgroupLtMask\n  SetRegister R2, %0\n"
        );
        let clock = fixed(Opcode::Cs2r, dst(4) | (80 << 72) | (1 << 80));
        assert_eq!(
            run(Compute, clock),
            "{0}:\n  SetRegister R4, #0x0\n  SetRegister R5, #0x0\n"
        );
    }

    #[test]
    fn csmtest_reads_false() {
        let inst = fixed(Opcode::Csmtest, (2 << 81) | (7 << 84) | (7 << 87));
        assert_eq!(run(VertexB, inst), "{0}:\n  SetPred P2, #false\n");
        let both = fixed(Opcode::Csmtest, (1 << 84) | (7 << 87));
        assert_eq!(
            run(VertexB, both),
            "{0}:\n  SetPred P0, #false\n  SetPred P1, #false\n"
        );
        rejects(
            VertexB,
            fixed(Opcode::Csmtest, (7 << 81) | (7 << 84) | (3 << 87)),
        );
        rejects(VertexB, fixed(Opcode::Csmtest, dst(1) | (7 << 87)));
    }

    #[test]
    fn lepc_loads_the_instruction_address() {
        let nop = fixed(Opcode::Nop, 0);
        let lepc = fixed(Opcode::Lepc, dst(4));
        let address = TestEnv::BASE + 0x10;
        assert_eq!(
            translate(&[nop, lepc]).unwrap(),
            format!(
                "{{0}}:\n  SetRegister R4, #{:#x}\n  SetRegister R5, #{:#x}\n",
                address as u32,
                address >> 32
            )
        );
        rejects(Geometry, fixed(Opcode::Lepc, dst(4) | ra(1)));
        assert!(
            run(Geometry, lepc).contains("SetRegister R4"),
            "a geometry program's LEPC without the at-exit setup is an ordinary LEPC"
        );
    }

    #[test]
    fn bar_sync_is_a_workgroup_barrier() {
        let defer_blocking = 1 << 80;
        assert_eq!(
            translate(&[fixed(Opcode::Bar, defer_blocking)]).unwrap(),
            "{0}:\n  Barrier\n"
        );
        assert_eq!(
            translate(&[fixed(Opcode::Bar, 0)]).unwrap(),
            "{0}:\n  Barrier\n"
        );
        rejects(Compute, fixed(Opcode::Bar, 1 << 54));
        rejects(Compute, fixed(Opcode::Bar, 1 << 77));
        rejects(Compute, encode(Opcode::Bar, Form::RegReg, 0));
    }

    #[test]
    fn ldtram_is_unsupported() {
        rejects(Fragment, fixed(Opcode::Ldtram, dst(1) | ra(63) | (1 << 91)));
    }

    fn exit() -> Instruction {
        fixed(Opcode::Exit, 7 << 87)
    }

    fn whole_program(stage: ShaderStage, code: &[Instruction], header: &[(usize, u32)]) -> TestEnv {
        let mut env = TestEnv::new(code);
        env.stage = stage;
        for &(word, value) in header {
            env.header[word] = value;
        }
        env
    }

    /// Translates and compiles the program "env" describes and checks the module with
    /// spirv-val; returns the translated IR.
    fn compile(name: &str, env: &TestEnv) -> String {
        let profile = Profile::default();
        let host = HostTranslateInfo::default();
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let ir = crate::translate::translate(env, options)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .ir
            .dump();
        let runtime = RuntimeInfo::default();
        let module = crate::module::compile(env, options, &runtime, &mut Bindings::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        validate(name, &profile, &module.spirv);
        ir
    }

    #[test]
    fn geometry_program_inside_the_exit_envelope_emits() {
        let body = [
            ald(0x80, 4, ra(RZ) | rb(RZ)),
            ast(0x70, 4, ra(RZ) | rb(0) | rc(RZ)),
            encode(
                Opcode::Out,
                Form::RegReg,
                dst(RZ) | ra(RZ) | rb(RZ) | (1 << 78),
            ),
        ];
        let header = [
            (0, 1 | (3 << 5) | (4 << 10)),
            (2, 1 << 24),
            (3, 7 << 24),
            (4, 3),
            (6, 0xf),
            (13, 0xf0 << 8),
        ];
        let code = crate::ampere::geometry_exit::envelope(&body);
        let env = whole_program(Geometry, &code, &header);
        let ir = compile("ampere_geometry_exit", &env);
        assert!(ir.contains("EmitVertex"), "{ir}");
        assert!(ir.contains("SetAttribute PositionX"), "{ir}");
        let (profile, host) = (Profile::default(), HostTranslateInfo::default());
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let translated = crate::translate::translate(&env, options).unwrap();
        assert!(
            !translated.info.reads_start_address,
            "the setup's LEPC is skipped"
        );

        let mut broken = code.clone();
        broken[code.len() - 2] = fixed(Opcode::Nop, 0);
        let error =
            crate::translate::translate(&whole_program(Geometry, &broken, &header), options)
                .unwrap_err()
                .to_string();
        assert!(error.contains("native exit handler body"), "{error}");
    }

    #[test]
    fn blit_programs_read_the_parameter_block_from_root_table_slot_24() {
        let code = [
            encode(
                Opcode::Ldc,
                Form::Bound,
                dst(0) | ra(RZ) | (0x8 << 38) | (24 << 54) | (4 << 73),
            ),
            ast(0x70, 1, ra(RZ) | rb(0) | rc(RZ)),
            exit(),
        ];
        let env = whole_program(
            VertexB,
            &code,
            &[(0, 1 | (3 << 5) | (1 << 10)), (13, 0xf0 << 8)],
        );
        let ir = compile("ampere_blit_slot_24", &env);
        assert!(!ir.contains("GetCbuf"), "{ir}");
        let (profile, host) = (Profile::default(), HostTranslateInfo::default());
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let translated = crate::translate::translate(&env, options).unwrap();
        assert!(translated.info.uses_render_area);
        assert_eq!(translated.info.constant_buffer_mask, 0);
    }

    #[test]
    fn vertex_program_copies_an_attribute_to_the_position() {
        let code = [
            ald(0x80, 4, ra(RZ) | rb(RZ)),
            ast(0x70, 4, ra(RZ) | rb(0) | rc(RZ)),
            ast(0x5c, 1, ra(RZ) | rb(RZ) | rc(RZ)),
            exit(),
        ];
        let env = whole_program(
            VertexB,
            &code,
            &[(0, 1 | (3 << 5) | (1 << 10)), (6, 0xf), (13, 0xf0 << 8)],
        );
        let ir = compile("ampere_vertex", &env);
        for attribute in ["Generic0X", "Generic0W"] {
            assert!(ir.contains(&format!("GetAttribute {attribute}")), "{ir}");
        }
        for attribute in ["PositionX", "PositionW"] {
            assert!(ir.contains(&format!("SetAttribute {attribute}")), "{ir}");
        }
    }

    #[test]
    fn fragment_program_writes_interpolated_inputs() {
        let code = [
            ipa_to(0, 32, 0),
            ipa_to(1, 33, 1 << 76),
            ipa_to(2, 34, 0),
            ipa_to(3, 31, 0),
            exit(),
        ];
        let env = whole_program(
            Fragment,
            &code,
            &[
                (0, 2 | (3 << 5) | (5 << 10)),
                (5, 0x80 << 24),
                (6, 0x2a),
                (18, 0xf),
            ],
        );
        let ir = compile("ampere_fragment", &env);
        assert!(ir.contains("SetFragColor"), "{ir}");
        assert_eq!(ir.matches("FPMul").count(), 3, "{ir}");
    }

    /// An indexed IPA divides by w only when the attribute it selects at run time is
    /// perspective: the mask has a bit per 16-byte slot, generic n at slot 8 + n.
    #[test]
    fn indexed_ipa_masks_the_perspective_slots() {
        let fields = dst(1) | ra(5) | rb(RZ) | (32 << 64) | PT_DST;
        let code = [encode(Opcode::Ipa, Form::UregIndexed, fields), exit()];
        let perspective_constant_screen_linear = 0xaa | (0x55 << 8) | (0xff << 16);
        let mut env = whole_program(Fragment, &code, &[(6, perspective_constant_screen_linear)]);
        env.header[7] = 0x5555_5555;
        let mut prog = crate::ir::Program::new();
        let block = prog.new_block();
        super::super::translate_block(&mut prog, block, &env, 0, env.end() - 16).unwrap();
        let ir = prog.dump();
        let low = !(0xff | (0b110 << 8) | (0xf << 12));
        assert!(
            ir.contains(&format!("SelectU32 %6, #{low:#x}, #0x3ff4fff")),
            "{ir}"
        );
    }

    /// IPA returns a perspective input divided by w, which programs undo with the reciprocal of
    /// IPA's position w; constant and screen-linear inputs come back as they are.
    #[test]
    fn ipa_divides_only_perspective_inputs_by_w() {
        let code = [
            ipa_to(0, 32, 0),
            ipa_to(1, 36, 0),
            ipa_to(2, 40, 0),
            ipa_to(3, 31, 0),
            exit(),
        ];
        let constant_screen_linear_perspective = 0x55 | (0xff << 8) | (0xaa << 16);
        let env = whole_program(
            Fragment,
            &code,
            &[
                (0, 2 | (3 << 5) | (5 << 10)),
                (5, 0x80 << 24),
                (6, constant_screen_linear_perspective),
                (18, 0xf),
            ],
        );
        let ir = compile("ampere_fragment_imap", &env);
        let multiplied: Vec<&str> = ir.lines().filter(|line| line.contains("FPMul")).collect();
        assert_eq!(multiplied.len(), 1, "{ir}");
        let z = ir
            .lines()
            .find(|line| line.contains("GetAttribute Generic2X"))
            .and_then(|line| line.split_whitespace().next())
            .expect("Generic2X is read");
        assert!(multiplied[0].contains(&format!("FPMul32 {z},")), "{ir}");
    }

    #[test]
    fn fragment_program_interpolates_at_an_offset() {
        let code = [
            ipa_to(4, 34, 0),
            ipa_at_offset_to(0, 32, 0),
            ipa_at_offset_to(1, 33, 0),
            ipa_at_offset_to(2, 28, 0),
            ipa_at_offset_to(3, 255, 0),
            exit(),
        ];
        let env = whole_program(
            Fragment,
            &code,
            &[
                (0, 2 | (3 << 5) | (5 << 10)),
                (5, 0x80 << 24),
                (6, 0x2a),
                (18, 0xf),
            ],
        );
        let ir = compile("ampere_fragment_offset", &env);
        assert_eq!(ir.matches("InterpolateAtOffset").count(), 2, "{ir}");
        assert!(!ir.contains("DPdx"), "{ir}");
    }

    #[test]
    fn compute_program_reads_ids_and_synchronizes() {
        let code = [
            fixed(Opcode::S2r, dst(0) | (33 << 72)),
            fixed(Opcode::S2ur, dst(1) | (37 << 72)),
            fixed(Opcode::Cs2r, dst(2) | (0xff << 72) | (1 << 80)),
            fixed(Opcode::Bar, 1 << 80),
            exit(),
        ];
        let ir = compile("ampere_compute", &whole_program(Compute, &code, &[]));
        assert!(ir.contains("Barrier"), "{ir}");
    }
}
