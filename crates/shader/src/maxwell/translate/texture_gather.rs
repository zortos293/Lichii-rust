//! TLD4 and TLD4 (b): four-texel gathers.

use super::texture_fetch::{
    Dim, float_coords, image, packed_offset, set_sparse_pred, store_masked,
};
use super::{TResult, Translator};
use crate::ir::{Reg, TextureInstInfo, Value};

fn ptp_offsets(t: &mut Translator<'_>, reg: &mut Reg) -> (Value, Value) {
    let mut vector = |t: &mut Translator<'_>| {
        let value = t.x(*reg);
        *reg = reg.offset(1);
        let mut c = [Value::Void; 4];
        for (i, c) in c.iter_mut().enumerate() {
            *c =
                t.ir.bit_field_extract(value, Value::U32(i as u32 * 8), Value::U32(6), true);
        }
        t.ir.composite_construct(&c)
    };
    let first = vector(t);
    (first, vector(t))
}

fn tld4_impl(t: &mut Translator<'_>, component: u64, offset_type: u64, bindless: bool) -> TResult {
    let insn = t.insn;
    let dim = Dim::decode(t)?;
    let is_depth = insn.bit(50);
    let coords = float_coords(t, insn.src_a_reg(), dim);

    let mut meta = insn.src_b_reg();
    let handle = if bindless {
        let handle = t.x(meta);
        meta = meta.offset(1);
        handle
    } else {
        Value::U32(insn.bits(36, 13) as u32 * 4)
    };
    let (offset, offset2) = match offset_type {
        0 => (Value::Void, Value::Void),
        1 => (packed_offset(t, &mut meta, dim, 6, 8)?, Value::Void),
        2 => ptp_offsets(t, &mut meta),
        _ => return Err(t.unsupported()),
    };
    let info = TextureInstInfo {
        ty: dim.ty(),
        is_depth,
        gather_component: component as u8,
        ..Default::default()
    };
    let sample = if is_depth {
        let dref = t.f(meta);
        image::gather_dref(&mut t.ir, handle, coords, offset, offset2, dref, info)
    } else {
        image::gather(&mut t.ir, handle, coords, offset, offset2, info)
    };
    store_masked(t, insn.dest_reg(), sample);
    set_sparse_pred(t, sample);
    Ok(())
}

pub(super) fn tld4_b(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    tld4_impl(t, insn.bits(38, 2), insn.bits(36, 2), true)
}

pub(super) fn tld4(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    tld4_impl(t, insn.bits(56, 2), insn.bits(54, 2), false)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tld4_aoffi_green_component() {
        let insn = encode(
            Opcode::Tld4,
            (1 << 56)
                | (1 << 54)
                | (7 << 51)
                | (3 << 36)
                | (0b0001 << 31)
                | (2 << 28)
                | (6 << 20)
                | (255 << 8)
                | 9,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructF32x2 #0.0f, #0.0f
  %1 = GetRegister R6
  %2 = BitFieldSExtract %1, #0x0, #0x6
  %3 = BitFieldSExtract %1, #0x8, #0x6
  %4 = CompositeConstructU32x2 %2, %3
  %5 = BoundImageGather #0xc, %0, %4, void (flags 0x408000)
  %6 = CompositeExtractF32x4 %5, #0x0
  %7 = BitCastU32F32 %6
  SetRegister R9, %7
"
        );
    }

    #[test]
    fn tld4_b_ptp_depth_sparse() {
        let insn = encode(
            Opcode::Tld4B,
            (3 << 38) | (2 << 36) | (2 << 51) | (1 << 50) | (2 << 20) | (255 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = BitFieldSExtract %1, #0x0, #0x6
  %3 = BitFieldSExtract %1, #0x8, #0x6
  %4 = BitFieldSExtract %1, #0x10, #0x6
  %5 = BitFieldSExtract %1, #0x18, #0x6
  %6 = CompositeConstructU32x4 %2, %3, %4, %5
  %7 = GetRegister R4
  %8 = BitFieldSExtract %7, #0x0, #0x6
  %9 = BitFieldSExtract %7, #0x8, #0x6
  %10 = BitFieldSExtract %7, #0x10, #0x6
  %11 = BitFieldSExtract %7, #0x18, #0x6
  %12 = CompositeConstructU32x4 %8, %9, %10, %11
  %13 = GetRegister R5
  %14 = BitCastF32U32 %13
  %15 = BindlessImageGatherDref %0, #0.0f, %6, %12, %14 (flags 0xc40000)
  %16 = GetSparseFromOp %15
  %17 = LogicalNot %16
  SetPred P2, %17
"
        );
    }

    #[test]
    fn tld4_rejects_invalid_offset_type() {
        assert!(translate(&[encode(Opcode::Tld4, 3 << 54)]).is_err());
        assert!(translate(&[encode(Opcode::Tld4B, 3 << 36)]).is_err());
    }
}
