//! LEA: scaled address add, low and high halves, ported from load_effective_address.cpp.

use super::{TResult, Translator};
use crate::ir::{Pred, Value};

fn lea_hi(
    t: &mut Translator<'_>,
    base: Value,
    offset_hi: Value,
    scale: u32,
    neg: bool,
    x: bool,
) -> TResult {
    let insn = t.insn;
    if x || !Pred::p(insn.bits(48, 3) as u8).is_true() || insn.bit(47) {
        return Err(t.unsupported());
    }
    let offset_lo = t.x(insn.src_a_reg());
    let pair = t.ir.composite_construct(&[offset_lo, offset_hi]);
    let packed_offset = t.ir.pack_uint_2x32(pair);
    let offset = if neg {
        t.ir.ineg(packed_offset)
    } else {
        packed_offset
    };
    let scaled_offset = t.ir.shift_right_logical(offset, Value::U32(32 - scale));
    let unpacked = t.ir.unpack_uint_2x32(scaled_offset);
    let scaled_offset_w0 = t.ir.composite_extract(unpacked, 0);
    let result = t.ir.iadd(base, scaled_offset_w0);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

fn lea_lo(t: &mut Translator<'_>, base: Value) -> TResult {
    let insn = t.insn;
    if insn.bit(46) || !Pred::p(insn.bits(48, 3) as u8).is_true() || insn.bit(47) {
        return Err(t.unsupported());
    }
    let offset_lo = t.x(insn.src_a_reg());
    let offset = if insn.bit(45) {
        t.ir.ineg(offset_lo)
    } else {
        offset_lo
    };
    let scale = insn.bits(39, 5) as u32;
    let scaled_offset = t.ir.shift_left_logical(offset, Value::U32(scale));
    let result = t.ir.iadd(base, scaled_offset);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn lea_hi_reg(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let base = t.get_reg20();
    let offset_hi = t.get_reg39();
    lea_hi(
        t,
        base,
        offset_hi,
        insn.bits(28, 5) as u32,
        insn.bit(37),
        insn.bit(38),
    )
}

pub(super) fn lea_hi_cbuf(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let base = t.get_cbuf()?;
    let offset_hi = t.get_reg39();
    lea_hi(
        t,
        base,
        offset_hi,
        insn.bits(51, 5) as u32,
        insn.bit(56),
        insn.bit(57),
    )
}

pub(super) fn lea_lo_reg(t: &mut Translator<'_>) -> TResult {
    let base = t.get_reg20();
    lea_lo(t, base)
}

pub(super) fn lea_lo_cbuf(t: &mut Translator<'_>) -> TResult {
    let base = t.get_cbuf()?;
    lea_lo(t, base)
}

pub(super) fn lea_lo_imm(t: &mut Translator<'_>) -> TResult {
    let base = t.get_imm20();
    lea_lo(t, base)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn lea_lo_reg_neg() {
        let insn = encode(
            Opcode::LeaLoReg,
            (1 << 8) | (2 << 20) | (3 << 39) | (1 << 45) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = INeg32 %1
  %3 = ShiftLeftLogical32 %2, #0x3
  %4 = IAdd32 %0, %3
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lea_lo_imm() {
        let insn = encode(
            Opcode::LeaLoImm,
            (1 << 8) | (5 << 20) | (2 << 39) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = ShiftLeftLogical32 %0, #0x2
  %2 = IAdd32 #0x5, %1
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lea_lo_cbuf() {
        let insn = encode(
            Opcode::LeaLoCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (2 << 39) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = ShiftLeftLogical32 %1, #0x2
  %3 = IAdd32 %0, %2
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lea_lo_predicate() {
        let insn = encode(Opcode::LeaLoReg, 1 << 48);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn lea_hi_reg_neg() {
        let insn = encode(
            Opcode::LeaHiReg,
            (1 << 8) | (2 << 20) | (3 << 39) | (4 << 28) | (1 << 37) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = CompositeConstructU32x2 %2, %1
  %4 = PackUint2x32 %3
  %5 = INeg64 %4
  %6 = ShiftRightLogical64 %5, #0x1c
  %7 = UnpackUint2x32 %6
  %8 = CompositeExtractU32x2 %7, #0x0
  %9 = IAdd32 %0, %8
  SetRegister R0, %9
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lea_hi_cbuf() {
        let insn = encode(
            Opcode::LeaHiCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (4 << 51) | (7 << 48),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = CompositeConstructU32x2 %2, %1
  %4 = PackUint2x32 %3
  %5 = ShiftRightLogical64 %4, #0x1c
  %6 = UnpackUint2x32 %5
  %7 = CompositeExtractU32x2 %6, #0x0
  %8 = IAdd32 %0, %7
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn lea_hi_x() {
        let insn = encode(Opcode::LeaHiReg, (1 << 38) | (7 << 48));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
