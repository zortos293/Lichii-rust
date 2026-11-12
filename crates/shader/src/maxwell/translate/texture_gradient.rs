//! TXD and TXD (b): sampling with explicit derivatives.

use super::texture_fetch::{image, set_sparse_pred, store_masked};
use super::{TResult, Translator};
use crate::ir::{TextureInstInfo, TextureType, Value};

fn txd_impl(t: &mut Translator<'_>, bindless: bool) -> TResult {
    let insn = t.insn;
    if insn.bit(50) {
        return Err(t.unsupported());
    }
    let (ty, num_derivatives, is_array) = match insn.bits(28, 3) {
        0 => (TextureType::Color1D, 1, false),
        1 => (TextureType::ColorArray1D, 1, true),
        2 => (TextureType::Color2D, 2, false),
        3 => (TextureType::ColorArray2D, 2, true),
        _ => return Err(t.unsupported()),
    };

    let mut base = insn.src_a_reg();
    let handle = if bindless {
        let handle = t.x(base);
        base = base.offset(1);
        handle
    } else {
        Value::U32(insn.bits(36, 13) as u32 * 4)
    };
    let last = base.offset(num_derivatives);
    let mut c = [Value::Void; 3];
    for (i, c) in c.iter_mut().take(num_derivatives as usize).enumerate() {
        *c = t.f(base.offset(i as u16));
    }
    let mut len = num_derivatives as usize;
    if is_array {
        let packed = t.x(last);
        let index =
            t.ir.bit_field_extract(packed, Value::U32(0), Value::U32(16), false);
        c[len] =
            t.ir.convert_i_to_f(32, 16, false, index, Default::default());
        len += 1;
    }
    let coords = if len == 1 {
        c[0]
    } else {
        t.ir.composite_construct(&c[..len])
    };

    let derivative = insn.src_b_reg();
    let mut d = [Value::Void; 4];
    let num_components = num_derivatives as usize * 2;
    for (i, d) in d.iter_mut().take(num_components).enumerate() {
        *d = t.f(derivative.offset(i as u16));
    }
    let derivatives = t.ir.composite_construct(&d[..num_components]);

    let offset = if insn.bit(35) {
        let packed = t.x(last);
        let o = [
            t.ir.bit_field_extract(packed, Value::U32(16), Value::U32(4), true),
            t.ir.bit_field_extract(packed, Value::U32(20), Value::U32(4), true),
        ];
        t.ir.composite_construct(&o)
    } else {
        Value::Void
    };

    let info = TextureInstInfo {
        ty,
        num_derivatives: num_derivatives as u8,
        ..Default::default()
    };
    let sample = image::gradient(
        &mut t.ir,
        handle,
        coords,
        derivatives,
        offset,
        Value::Void,
        info,
    );
    store_masked(t, insn.dest_reg(), sample);
    set_sparse_pred(t, sample);
    Ok(())
}

pub(super) fn txd(t: &mut Translator<'_>) -> TResult {
    txd_impl(t, false)
}

pub(super) fn txd_b(t: &mut Translator<'_>) -> TResult {
    txd_impl(t, true)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn txd_1d() {
        let insn = encode(
            Opcode::Txd,
            (5 << 36) | (7 << 51) | (0b0001 << 31) | (4 << 20) | (2 << 8) | 8,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = BitCastF32U32 %0
  %2 = GetRegister R4
  %3 = BitCastF32U32 %2
  %4 = GetRegister R5
  %5 = BitCastF32U32 %4
  %6 = CompositeConstructF32x2 %3, %5
  %7 = BoundImageGradient #0x14, %1, %6, void, void (flags 0x1000000)
  %8 = CompositeExtractF32x4 %7, #0x0
  %9 = BitCastU32F32 %8
  SetRegister R8, %9
"
        );
    }

    #[test]
    fn txd_b_array_2d_with_offset() {
        let insn = encode(
            Opcode::TxdB,
            (7 << 51) | (1 << 35) | (3 << 28) | (255 << 20) | (1 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = BitCastF32U32 %1
  %3 = GetRegister R3
  %4 = BitCastF32U32 %3
  %5 = GetRegister R4
  %6 = BitFieldUExtract %5, #0x0, #0x10
  %7 = ConvertF32U16 %6
  %8 = CompositeConstructF32x3 %2, %4, %7
  %9 = CompositeConstructF32x4 #0.0f, #0.0f, #0.0f, #0.0f
  %10 = GetRegister R4
  %11 = BitFieldSExtract %10, #0x10, #0x4
  %12 = BitFieldSExtract %10, #0x14, #0x4
  %13 = CompositeConstructU32x2 %11, %12
  %14 = BindlessImageGradient %0, %8, %9, %13, void (flags 0x200c000)
"
        );
    }

    #[test]
    fn txd_rejects_lc_and_3d() {
        assert!(translate(&[encode(Opcode::Txd, 1 << 50)]).is_err());
        assert!(translate(&[encode(Opcode::Txd, 4 << 28)]).is_err());
        assert!(translate(&[encode(Opcode::TxdB, 6 << 28)]).is_err());
    }
}
