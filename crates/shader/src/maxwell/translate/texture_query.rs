//! TXQ and TXQ (b): texture queries.

use super::texture_fetch::image;
use super::{TResult, Translator};
use crate::ir::Value;

fn txq_impl(t: &mut Translator<'_>, cbuf_offset: Option<u32>) -> TResult {
    let insn = t.insn;
    let mut src = insn.src_a_reg();
    let handle = match cbuf_offset {
        Some(offset) => Value::U32(offset),
        None => {
            let handle = t.x(src);
            src = src.offset(1);
            handle
        }
    };
    let mask = insn.bits(31, 4);
    if insn.bits(22, 3) != 1 {
        return Err(t.unsupported());
    }
    let skip_mips = Value::U1((mask >> 3) & 1 == 0);
    let lod = t.x(src);
    let query = image::query_dimension(&mut t.ir, handle, lod, skip_mips);

    let mut dest = insn.dest_reg();
    for element in 0..4 {
        if (mask >> element) & 1 == 0 {
            continue;
        }
        let value = t.ir.composite_extract(query, element);
        t.set_x(dest, value);
        dest = dest.offset(1);
    }
    Ok(())
}

pub(super) fn txq(t: &mut Translator<'_>) -> TResult {
    let cbuf_offset = t.insn.bits(36, 13) as u32 * 4;
    txq_impl(t, Some(cbuf_offset))
}

pub(super) fn txq_b(t: &mut Translator<'_>) -> TResult {
    txq_impl(t, None)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn txq_dimensions_with_mips() {
        let insn = encode(
            Opcode::Txq,
            (3 << 36) | (0b1001 << 31) | (1 << 22) | (2 << 8) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = BoundImageQueryDimensions #0xc, %0, #false
  %2 = CompositeExtractU32x4 %1, #0x0
  SetRegister R4, %2
  %3 = CompositeExtractU32x4 %1, #0x3
  SetRegister R5, %3
"
        );
    }

    #[test]
    fn txq_b_bindless_skips_mips() {
        let insn = encode(Opcode::TxqB, (0b0010 << 31) | (1 << 22) | (2 << 8) | 4);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = BindlessImageQueryDimensions %0, %1, #true
  %3 = CompositeExtractU32x4 %2, #0x1
  SetRegister R4, %3
"
        );
    }

    #[test]
    fn txq_rejects_other_modes() {
        assert!(translate(&[encode(Opcode::Txq, 2 << 22)]).is_err());
        assert!(translate(&[encode(Opcode::TxqB, 5 << 22)]).is_err());
    }
}
