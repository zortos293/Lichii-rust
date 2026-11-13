//! TLD4S: 2D gathers with swizzled destinations.

use super::texture_fetch::image;
use super::texture_fetch_swizzled::check_alignment;
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Reg, TextureInstInfo, TextureType, Value};

fn sample(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let insn = t.insn;
    let handle = Value::U32(insn.bits(36, 13) as u32 * 4);
    let a = insn.src_a_reg();
    let b = insn.src_b_reg();
    let is_depth = insn.bit(50);
    let info = TextureInstInfo {
        ty: TextureType::Color2D,
        relaxed_precision: insn.bit(55),
        gather_component: insn.bits(52, 2) as u8,
        is_depth,
        ..Default::default()
    };
    let void = Value::Void;
    if insn.bit(51) {
        check_alignment(t, a, 2)?;
        if is_depth {
            check_alignment(t, b, 2)?;
        }
        let c = [t.f(a), t.f(a.offset(1))];
        let coords = t.ir.composite_construct(&c);
        let packed = t.x(b);
        let o = [
            t.ir.bit_field_extract(packed, Value::U32(0), Value::U32(6), true),
            t.ir.bit_field_extract(packed, Value::U32(8), Value::U32(6), true),
        ];
        let offset = t.ir.composite_construct(&o);
        if is_depth {
            let dref = t.f(b.offset(1));
            return Ok(image::gather_dref(
                &mut t.ir, handle, coords, offset, void, dref, info,
            ));
        }
        return Ok(image::gather(&mut t.ir, handle, coords, offset, void, info));
    }
    if is_depth {
        check_alignment(t, a, 2)?;
        let c = [t.f(a), t.f(a.offset(1))];
        let coords = t.ir.composite_construct(&c);
        let dref = t.f(b);
        return Ok(image::gather_dref(
            &mut t.ir, handle, coords, void, void, dref, info,
        ));
    }
    let c = [t.f(a), t.f(b)];
    let coords = t.ir.composite_construct(&c);
    Ok(image::gather(&mut t.ir, handle, coords, void, void, info))
}

pub(super) fn tld4s(t: &mut Translator<'_>) -> TResult {
    let sample = sample(t)?;
    let insn = t.insn;
    let dest_a = insn.dest_reg();
    let dest_b = Reg(insn.bits(28, 8) as u16);
    let mut components = [Value::Void; 4];
    for (i, c) in components.iter_mut().enumerate() {
        *c = t.ir.composite_extract(sample, i as u32);
    }
    if insn.bit(55) {
        for (dest, pair) in [(dest_a, 0), (dest_b, 2)] {
            let v =
                t.ir.composite_construct(&[components[pair], components[pair + 1]]);
            let packed = t.ir.pack_half_2x16(v);
            t.set_x(dest, packed);
        }
        return Ok(());
    }
    check_alignment(t, dest_a, 2)?;
    check_alignment(t, dest_b, 2)?;
    let dests = [dest_a, dest_a.offset(1), dest_b, dest_b.offset(1)];
    for (dest, value) in dests.into_iter().zip(components) {
        t.set_f(dest, value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tld4s_f32_blue() {
        let insn = encode(
            Opcode::Tld4s,
            (2 << 52) | (1 << 36) | (2 << 28) | (255 << 20) | (255 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructF32x2 #0.0f, #0.0f
  %1 = BoundImageGather #0x4, %0, void, void (flags 0x808000)
  %2 = CompositeExtractF32x4 %1, #0x0
  %3 = CompositeExtractF32x4 %1, #0x1
  %4 = CompositeExtractF32x4 %1, #0x2
  %5 = CompositeExtractF32x4 %1, #0x3
  %6 = BitCastU32F32 %2
  SetRegister R0, %6
  %7 = BitCastU32F32 %3
  SetRegister R1, %7
  %8 = BitCastU32F32 %4
  SetRegister R2, %8
  %9 = BitCastU32F32 %5
  SetRegister R3, %9
"
        );
    }

    #[test]
    fn tld4s_f16_aoffi_depth() {
        let insn = encode(
            Opcode::Tld4s,
            (1 << 55) | (1 << 51) | (1 << 50) | (255 << 28) | (4 << 20) | (255 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructF32x2 #0.0f, #0.0f
  %1 = GetRegister R4
  %2 = BitFieldSExtract %1, #0x0, #0x6
  %3 = BitFieldSExtract %1, #0x8, #0x6
  %4 = CompositeConstructU32x2 %2, %3
  %5 = GetRegister R5
  %6 = BitCastF32U32 %5
  %7 = BoundImageGatherDref #0x0, %0, %4, void, %6 (flags 0x248000)
  %8 = CompositeExtractF32x4 %7, #0x0
  %9 = CompositeExtractF32x4 %7, #0x1
  %10 = CompositeExtractF32x4 %7, #0x2
  %11 = CompositeExtractF32x4 %7, #0x3
  %12 = CompositeConstructF32x2 %8, %9
  %13 = PackHalf2x16 %12
  SetRegister R1, %13
  %14 = CompositeConstructF32x2 %10, %11
  %15 = PackHalf2x16 %14
"
        );
    }

    #[test]
    fn tld4s_rejects_unaligned() {
        assert!(translate(&[encode(Opcode::Tld4s, (1 << 51) | (3 << 8))]).is_err());
        assert!(translate(&[encode(Opcode::Tld4s, (255 << 8) | (255 << 20) | 1)]).is_err());
    }
}
