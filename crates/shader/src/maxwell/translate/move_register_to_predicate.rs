//! R2P: unpack register bits into predicates or condition codes, ported from
//! move_register_to_predicate.cpp.

use super::{TResult, Translator};
use crate::ir::{Pred, Value};

fn r2p(t: &mut Translator<'_>, mask: Value) -> TResult {
    let insn = t.insn;
    let src = t.x(insn.src_a_reg());
    let count = Value::U32(1);
    let pr_mode = !insn.bit(40);
    let num_items = if pr_mode { 7 } else { 4 };
    let offset_base = insn.bits(41, 2) as u32 * 8;
    for index in 0..num_items {
        let src_field =
            t.ir.bit_field_extract(src, Value::U32(offset_base + index), count, false);
        let src_zero = t.ir.get_zero_from_op(src_field);
        let src_bit = t.ir.logical_not(src_zero);
        let mask_bfe =
            t.ir.bit_field_extract(mask, Value::U32(index), count, false);
        let inv_mask_bit = t.ir.get_zero_from_op(mask_bfe);
        if pr_mode {
            let pred = Pred::p(index as u8);
            let old = t.ir.get_pred(pred, false);
            let value = t.ir.select(inv_mask_bit, old, src_bit);
            t.ir.set_pred(pred, value);
            continue;
        }
        let old = match index {
            0 => t.ir.get_z_flag(),
            1 => t.ir.get_s_flag(),
            2 => t.ir.get_c_flag(),
            _ => t.ir.get_o_flag(),
        };
        let value = t.ir.select(inv_mask_bit, old, src_bit);
        match index {
            0 => t.ir.set_z_flag(value),
            1 => t.ir.set_s_flag(value),
            2 => t.ir.set_c_flag(value),
            _ => t.ir.set_o_flag(value),
        }
    }
    Ok(())
}

pub(super) fn r2p_cbuf(t: &mut Translator<'_>) -> TResult {
    let mask = t.get_cbuf()?;
    r2p(t, mask)
}

pub(super) fn r2p_reg(t: &mut Translator<'_>) -> TResult {
    let mask = t.get_reg20();
    r2p(t, mask)
}

pub(super) fn r2p_imm(t: &mut Translator<'_>) -> TResult {
    let mask = t.get_imm20();
    r2p(t, mask)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn r2p_reg_condition_codes() {
        let insn = encode(Opcode::R2pReg, (1 << 8) | (2 << 20) | (1 << 40) | (1 << 41));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = BitFieldUExtract %1, #0x8, #0x1
  %3 = GetZeroFromOp %2
  %4 = LogicalNot %3
  %5 = BitFieldUExtract %0, #0x0, #0x1
  %6 = GetZeroFromOp %5
  %7 = GetZFlag
  %8 = SelectU1 %6, %7, %4
  SetZFlag %8
  %9 = BitFieldUExtract %1, #0x9, #0x1
  %10 = GetZeroFromOp %9
  %11 = LogicalNot %10
  %12 = BitFieldUExtract %0, #0x1, #0x1
  %13 = GetZeroFromOp %12
  %14 = GetSFlag
  %15 = SelectU1 %13, %14, %11
  SetSFlag %15
  %16 = BitFieldUExtract %1, #0xa, #0x1
  %17 = GetZeroFromOp %16
  %18 = LogicalNot %17
  %19 = BitFieldUExtract %0, #0x2, #0x1
  %20 = GetZeroFromOp %19
  %21 = GetCFlag
  %22 = SelectU1 %20, %21, %18
  SetCFlag %22
  %23 = BitFieldUExtract %1, #0xb, #0x1
  %24 = GetZeroFromOp %23
  %25 = LogicalNot %24
  %26 = BitFieldUExtract %0, #0x3, #0x1
  %27 = GetZeroFromOp %26
  %28 = GetOFlag
  %29 = SelectU1 %27, %28, %25
  SetOFlag %29
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn r2p_imm_predicates() {
        let insn = encode(Opcode::R2pImm, (1 << 8) | (0x3 << 20));
        let dump = translate(&[insn]).unwrap();
        let head = "\
{0}:
  %0 = GetRegister R1
  %1 = BitFieldUExtract %0, #0x0, #0x1
  %2 = GetZeroFromOp %1
  %3 = LogicalNot %2
  %4 = BitFieldUExtract #0x3, #0x0, #0x1
  %5 = GetZeroFromOp %4
  %6 = GetPred P0
  %7 = SelectU1 %5, %6, %3
  SetPred P0, %7
  %8 = BitFieldUExtract %0, #0x1, #0x1
";
        assert!(dump.starts_with(head), "{dump}");
        assert_eq!(dump.lines().count(), 58);
    }

    #[test]
    fn r2p_cbuf() {
        let insn = encode(
            Opcode::R2pCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (1 << 40),
        );
        let dump = translate(&[insn]).unwrap();
        let head = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = BitFieldUExtract %1, #0x0, #0x1
";
        assert!(dump.starts_with(head), "{dump}");
        assert_eq!(dump.lines().count(), 35);
    }
}
