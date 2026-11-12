//! TMML and TMML (b): level-of-detail queries.

use super::texture_fetch::{Dim, image};
use super::{TResult, Translator};
use crate::ir::{FpControl, Reg, TextureInstInfo, Value};

/// Query coordinates; the array layer the ISA reads is not needed and is dropped.
fn coords(t: &mut Translator<'_>, reg: Reg, dim: Dim) -> Value {
    let (first, count) = match dim {
        Dim::D1 => (0, 1),
        Dim::Array1D => (1, 1),
        Dim::D2 => (0, 2),
        Dim::Array2D => (1, 2),
        Dim::D3 | Dim::Cube => (0, 3),
        Dim::ArrayCube => (1, 3),
    };
    let mut c = [Value::Void; 3];
    for (i, c) in c.iter_mut().take(count).enumerate() {
        *c = t.f(reg.offset(first + i as u16));
    }
    if count == 1 {
        c[0]
    } else {
        t.ir.composite_construct(&c[..count])
    }
}

fn tmml_impl(t: &mut Translator<'_>, bindless: bool) -> TResult {
    let insn = t.insn;
    let mask = insn.bits(31, 4);
    if mask & 0b1100 != 0 {
        return Err(t.unsupported());
    }
    let dim = Dim::decode(t)?;
    let coords = coords(t, insn.src_a_reg(), dim);
    let handle = if bindless {
        t.x(insn.src_b_reg())
    } else {
        Value::U32(insn.bits(36, 13) as u32 * 4)
    };
    let info = TextureInstInfo {
        ty: dim.ty(),
        ..Default::default()
    };
    let sample = image::query_lod(&mut t.ir, handle, coords, info);

    let mut dest = insn.dest_reg();
    for element in 0..2 {
        if (mask >> element) & 1 == 0 {
            continue;
        }
        let value = t.ir.composite_extract(sample, element ^ 1);
        let scaled = t.ir.fp_mul(value, Value::F32(256.0), FpControl::default());
        let fixed = t.ir.convert_f_to_i(32, element == 0, scaled);
        let masked = t.ir.bitwise_and(fixed, Value::U32(0xffff));
        t.set_x(dest, masked);
        dest = dest.offset(1);
    }
    Ok(())
}

pub(super) fn tmml(t: &mut Translator<'_>) -> TResult {
    tmml_impl(t, false)
}

pub(super) fn tmml_b(t: &mut Translator<'_>) -> TResult {
    tmml_impl(t, true)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tmml_array_1d_both_results() {
        let insn = encode(
            Opcode::Tmml,
            (2 << 36) | (0b0011 << 31) | (1 << 28) | (4 << 8) | 6,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R5
  %1 = BitCastF32U32 %0
  %2 = BoundImageQueryLod #0x8, %1 (flags 0x4000)
  %3 = CompositeExtractF32x4 %2, #0x1
  %4 = FPMul32 %3, #256.0f
  %5 = ConvertS32F32 %4
  %6 = BitwiseAnd32 %5, #0xffff
  SetRegister R6, %6
  %7 = CompositeExtractF32x4 %2, #0x0
  %8 = FPMul32 %7, #256.0f
  %9 = ConvertU32F32 %8
  %10 = BitwiseAnd32 %9, #0xffff
  SetRegister R7, %10
"
        );
    }

    #[test]
    fn tmml_b_bindless_2d() {
        let insn = encode(Opcode::TmmlB, (2 << 28) | (3 << 20) | (255 << 8));
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructF32x2 #0.0f, #0.0f
  %1 = GetRegister R3
  %2 = BindlessImageQueryLod %1, %0 (flags 0x8000)
"
        );
    }

    #[test]
    fn tmml_rejects_ba_results() {
        assert!(translate(&[encode(Opcode::Tmml, 0b0100 << 31)]).is_err());
        assert!(translate(&[encode(Opcode::TmmlB, 5 << 28)]).is_err());
    }
}
