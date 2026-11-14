//! LDL, STL, LDS and STS: local and shared memory.

use super::load_store_memory::is_aligned;
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{ICompare, Opcode, Value};

/// Byte offset: `Ra` plus a signed 24-bit immediate, or the unsigned immediate when `Ra` is RZ.
fn offset(t: &mut Translator<'_>) -> Value {
    let reg = t.insn.src_a_reg();
    if reg.is_zero() {
        return Value::U32(t.insn.bits(20, 24) as u32);
    }
    let base = t.x(reg);
    t.ir.iadd(base, Value::U32(t.insn.signed_bits(20, 24) as u32))
}

/// The word index and byte offset of a local memory access.
fn word_offset(t: &mut Translator<'_>) -> (Value, Value) {
    let offset = offset(t);
    let word = match offset {
        Value::U32(bytes) => Value::U32(bytes / 4),
        _ => t.ir.shift_right_arithmetic(offset, Value::U32(2)),
    };
    (word, offset)
}

/// Access size in bits and signedness (bits 48..51).
fn size(t: &Translator<'_>) -> Result<(u16, bool), TranslateError> {
    Ok(match t.insn.bits(48, 3) {
        0 => (8, false),
        1 => (8, true),
        2 => (16, false),
        3 => (16, true),
        4 => (32, false),
        5 => (64, false),
        6 => (128, false),
        _ => return Err(t.unsupported()),
    })
}

/// Local memory size in bytes, from the environment (the program header or the compute launch).
fn local_memory_size(t: &Translator<'_>) -> Result<u32, TranslateError> {
    Ok(t.env.local_memory_size())
}

/// Bit position of the byte (`mask` 24) or half word (`mask` 16) at `offset` in its word.
fn sub_word_bit(t: &mut Translator<'_>, offset: Value, mask: u32) -> Value {
    let bits = t.ir.shift_left_logical(offset, Value::U32(3));
    t.ir.bitwise_and(bits, Value::U32(mask))
}

/// Loads a local memory word; reads past the end return zero.
fn load_local(t: &mut Translator<'_>, word: Value, offset: Value, size: u32) -> Value {
    let in_bounds = t.ir.icompare(ICompare::ULessThan, offset, Value::U32(size));
    let value = t.ir.inst(Opcode::LoadLocal, &[word]);
    t.ir.select(in_bounds, value, Value::U32(0))
}

pub(super) fn ldl(t: &mut Translator<'_>) -> TResult {
    let dest = t.insn.dest_reg();
    let (bits, signed) = size(t)?;
    let local_size = local_memory_size(t)?;
    let count = (bits / 32).max(1);
    if !is_aligned(dest, count) {
        return Err(t.unsupported());
    }
    let (word, offset) = word_offset(t);
    if local_size == 0 || matches!(offset, Value::U32(bytes) if bytes >= local_size) {
        crate::fallback::note("LDL", "local_read_past_size");
    }
    let value = load_local(t, word, offset, local_size);
    if bits < 32 {
        let bit = sub_word_bit(t, offset, if bits == 8 { 24 } else { 16 });
        let field =
            t.ir.bit_field_extract(value, bit, Value::U32(u32::from(bits)), signed);
        t.set_x(dest, field);
        return Ok(());
    }
    t.set_x(dest, value);
    for i in 1..count {
        let sub_word = t.ir.iadd(word, Value::U32(u32::from(i)));
        let sub_offset = t.ir.iadd(offset, Value::U32(u32::from(i) * 4));
        let value = load_local(t, sub_word, sub_offset, local_size);
        t.set_x(dest.offset(i), value);
    }
    Ok(())
}

pub(super) fn stl(t: &mut Translator<'_>) -> TResult {
    let reg = t.insn.dest_reg();
    let (bits, _) = size(t)?;
    let count = (bits / 32).max(1);
    if !is_aligned(reg, count) {
        return Err(t.unsupported());
    }
    let (word, offset) = word_offset(t);
    if let Value::U32(bytes) = offset
        && bytes >= local_memory_size(t)?
    {
        crate::fallback::note("STL", "local_write_past_size");
        return Ok(());
    }
    let src = t.x(reg);
    if bits < 32 {
        let bit = sub_word_bit(t, offset, if bits == 8 { 24 } else { 16 });
        let old = t.ir.inst(Opcode::LoadLocal, &[word]);
        let value =
            t.ir.bit_field_insert(old, src, bit, Value::U32(u32::from(bits)));
        t.ir.inst(Opcode::WriteLocal, &[word, value]);
        return Ok(());
    }
    t.ir.inst(Opcode::WriteLocal, &[word, src]);
    for i in 1..count {
        let sub_word = t.ir.iadd(word, Value::U32(u32::from(i)));
        let value = t.x(reg.offset(i));
        t.ir.inst(Opcode::WriteLocal, &[sub_word, value]);
    }
    Ok(())
}

pub(super) fn lds(t: &mut Translator<'_>) -> TResult {
    let dest = t.insn.dest_reg();
    let (bits, signed) = size(t)?;
    let (op, count) = match (bits, signed) {
        (8, false) => (Opcode::LoadSharedU8, 1),
        (8, true) => (Opcode::LoadSharedS8, 1),
        (16, false) => (Opcode::LoadSharedU16, 1),
        (16, true) => (Opcode::LoadSharedS16, 1),
        (32, _) => (Opcode::LoadSharedU32, 1),
        (64, _) => (Opcode::LoadSharedU64, 2),
        _ => (Opcode::LoadSharedU128, 4),
    };
    if !is_aligned(dest, count) {
        return Err(t.unsupported());
    }
    let offset = offset(t);
    let value = t.ir.inst(op, &[offset]);
    if count == 1 {
        t.set_x(dest, value);
        return Ok(());
    }
    for i in 0..count {
        let element = t.ir.composite_extract(value, u32::from(i));
        t.set_x(dest.offset(i), element);
    }
    Ok(())
}

pub(super) fn sts(t: &mut Translator<'_>) -> TResult {
    let reg = t.insn.dest_reg();
    let (op, count) = match size(t)?.0 {
        8 => (Opcode::WriteSharedU8, 1),
        16 => (Opcode::WriteSharedU16, 1),
        32 => (Opcode::WriteSharedU32, 1),
        64 => (Opcode::WriteSharedU64, 2),
        _ => (Opcode::WriteSharedU128, 4),
    };
    if !is_aligned(reg, count.min(2)) {
        return Err(t.unsupported());
    }
    let offset = offset(t);
    let value = match count {
        1 => t.x(reg),
        2 => {
            let words = [t.x(reg), t.x(reg.offset(1))];
            t.ir.composite_construct(&words)
        }
        _ => {
            let words = [
                t.x(reg),
                t.x(reg.offset(1)),
                t.x(reg.offset(2)),
                t.x(reg.offset(3)),
            ];
            t.ir.composite_construct(&words)
        }
    };
    t.ir.inst(op, &[offset, value]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::exit_program::tests::translate_stage;
    use super::super::tests::{TestEnv, encode, translate};
    use super::super::translate_block;
    use crate::ShaderStage;
    use crate::ir::Program;
    use crate::maxwell::opcode::Opcode;

    /// A vertex program header with `low` bytes of local memory and `crs` bytes of CRS stack.
    fn header(low: u32, crs: u32) -> [u32; 20] {
        let mut header = [0; 20];
        header[1] = low;
        header[3] = crs;
        header
    }

    fn vertex(code: u64, local: u32) -> String {
        translate_stage(&[code], ShaderStage::VertexB, header(local, 0x10)).unwrap()
    }

    #[test]
    fn ldl_u8_register_offset() {
        let insn = encode(Opcode::Ldl, (2 << 20) | (3 << 8) | 1);
        assert_eq!(
            vertex(insn, 0x30),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = IAdd32 %0, #0x2\n\
             \x20 %2 = ShiftRightArithmetic32 %1, #0x2\n\
             \x20 %3 = ULessThan %1, #0x40\n\
             \x20 %4 = LoadLocal %2\n\
             \x20 %5 = SelectU32 %3, %4, #0x0\n\
             \x20 %6 = ShiftLeftLogical32 %1, #0x3\n\
             \x20 %7 = BitwiseAnd32 %6, #0x18\n\
             \x20 %8 = BitFieldUExtract %5, %7, #0x8\n\
             \x20 SetRegister R1, %8\n"
        );
    }

    #[test]
    fn ldl_b64_immediate_offset() {
        let insn = encode(Opcode::Ldl, (5 << 48) | (8 << 20) | (0xff << 8) | 4);
        assert_eq!(
            vertex(insn, 0x30),
            "{0}:\n\
             \x20 %0 = ULessThan #0x8, #0x40\n\
             \x20 %1 = LoadLocal #0x2\n\
             \x20 %2 = SelectU32 %0, %1, #0x0\n\
             \x20 SetRegister R4, %2\n\
             \x20 %3 = IAdd32 #0x2, #0x1\n\
             \x20 %4 = IAdd32 #0x8, #0x4\n\
             \x20 %5 = ULessThan %4, #0x40\n\
             \x20 %6 = LoadLocal %3\n\
             \x20 %7 = SelectU32 %5, %6, #0x0\n\
             \x20 SetRegister R5, %7\n"
        );
    }

    #[test]
    fn ldl_in_compute_takes_the_launch_local_memory_size() {
        let mut env = TestEnv::new(&[encode(Opcode::Ldl, (4 << 48) | (0xff << 8))]);
        env.local_memory = 0x80;
        let mut prog = Program::new();
        let block = prog.new_block();
        translate_block(&mut prog, block, &env, 0, env.end()).unwrap();
        let dump = prog.dump();
        assert!(dump.contains("ULessThan #0x0, #0x80"), "{dump}");
    }

    #[test]
    fn stl_s16_and_out_of_bounds_drop() {
        let insn = encode(Opcode::Stl, (3 << 48) | (6 << 20) | (0xff << 8) | 2);
        assert_eq!(
            vertex(insn, 0x30),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ShiftLeftLogical32 #0x6, #0x3\n\
             \x20 %2 = BitwiseAnd32 %1, #0x10\n\
             \x20 %3 = LoadLocal #0x1\n\
             \x20 %4 = BitFieldInsert %3, %0, %2, #0x10\n\
             \x20 WriteLocal #0x1, %4\n"
        );
        let past_end = encode(Opcode::Stl, (4 << 48) | (0x40 << 20) | (0xff << 8) | 2);
        assert_eq!(vertex(past_end, 0x30), "{0}:\n");
    }

    #[test]
    fn stl_b128_register_offset_in_compute() {
        let insn = encode(Opcode::Stl, (6 << 48) | (5 << 8) | 8);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R5\n\
             \x20 %1 = IAdd32 %0, #0x0\n\
             \x20 %2 = ShiftRightArithmetic32 %1, #0x2\n\
             \x20 %3 = GetRegister R8\n\
             \x20 WriteLocal %2, %3\n\
             \x20 %4 = IAdd32 %2, #0x1\n\
             \x20 %5 = GetRegister R9\n\
             \x20 WriteLocal %4, %5\n\
             \x20 %6 = IAdd32 %2, #0x2\n\
             \x20 %7 = GetRegister R10\n\
             \x20 WriteLocal %6, %7\n\
             \x20 %8 = IAdd32 %2, #0x3\n\
             \x20 %9 = GetRegister R11\n\
             \x20 WriteLocal %8, %9\n"
        );
    }

    #[test]
    fn lds_s8_and_b128() {
        let s8 = encode(Opcode::Lds, (1 << 48) | (0xffffff << 20) | (2 << 8) | 1);
        assert_eq!(
            translate(&[s8]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0xffffffff\n\
             \x20 %2 = LoadSharedS8 %1\n\
             \x20 SetRegister R1, %2\n"
        );
        let b128 = encode(Opcode::Lds, (6 << 48) | (0x20 << 20) | (0xff << 8) | 4);
        assert_eq!(
            translate(&[b128]).unwrap(),
            "{0}:\n\
             \x20 %0 = LoadSharedU128 #0x20\n\
             \x20 %1 = CompositeExtractU32x4 %0, #0x0\n\
             \x20 SetRegister R4, %1\n\
             \x20 %2 = CompositeExtractU32x4 %0, #0x1\n\
             \x20 SetRegister R5, %2\n\
             \x20 %3 = CompositeExtractU32x4 %0, #0x2\n\
             \x20 SetRegister R6, %3\n\
             \x20 %4 = CompositeExtractU32x4 %0, #0x3\n\
             \x20 SetRegister R7, %4\n"
        );
    }

    #[test]
    fn sts_u16_and_b64() {
        let u16 = encode(Opcode::Sts, (2 << 48) | (4 << 20) | (0xff << 8) | 3);
        assert_eq!(
            translate(&[u16]).unwrap(),
            "{0}:\n  %0 = GetRegister R3\n  WriteSharedU16 #0x4, %0\n"
        );
        let b64 = encode(Opcode::Sts, (5 << 48) | (6 << 8) | 2);
        assert_eq!(
            translate(&[b64]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R6\n\
             \x20 %1 = IAdd32 %0, #0x0\n\
             \x20 %2 = GetRegister R2\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = CompositeConstructU32x2 %2, %3\n\
             \x20 WriteSharedU64 %1, %4\n"
        );
    }

    #[test]
    fn misaligned_and_reserved_sizes_are_unsupported() {
        assert!(translate(&[encode(Opcode::Lds, (5 << 48) | 1)]).is_err());
        assert!(translate(&[encode(Opcode::Sts, (6 << 48) | 1)]).is_err());
        assert!(translate(&[encode(Opcode::Stl, (6 << 48) | 2)]).is_err());
        assert!(translate(&[encode(Opcode::Lds, 7 << 48)]).is_err());
    }
}
