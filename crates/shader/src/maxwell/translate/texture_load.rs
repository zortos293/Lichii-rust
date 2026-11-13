//! TLD and TLD (b): texel fetches with integer coordinates.

use super::texture_fetch::{Dim, image, packed_offset, set_sparse_pred, store_masked};
use super::{TResult, Translator};
use crate::ir::{Reg, TextureInstInfo, Value};

fn integer_coords(t: &mut Translator<'_>, reg: Reg, dim: Dim) -> Value {
    let layer = |t: &mut Translator<'_>| {
        let v = t.x(reg);
        t.ir.bit_field_extract(v, Value::U32(0), Value::U32(16), false)
    };
    match dim {
        Dim::D1 => t.x(reg),
        Dim::Array1D => {
            let c = [t.x(reg.offset(1)), layer(t)];
            t.ir.composite_construct(&c)
        }
        Dim::D2 => {
            let c = [t.x(reg), t.x(reg.offset(1))];
            t.ir.composite_construct(&c)
        }
        Dim::Array2D => {
            let c = [t.x(reg.offset(1)), t.x(reg.offset(2)), layer(t)];
            t.ir.composite_construct(&c)
        }
        Dim::D3 | Dim::Cube => {
            let c = [t.x(reg), t.x(reg.offset(1)), t.x(reg.offset(2))];
            t.ir.composite_construct(&c)
        }
        Dim::ArrayCube => {
            let c = [
                t.x(reg.offset(1)),
                t.x(reg.offset(2)),
                t.x(reg.offset(3)),
                layer(t),
            ];
            t.ir.composite_construct(&c)
        }
    }
}

fn tld_impl(t: &mut Translator<'_>, bindless: bool) -> TResult {
    let insn = t.insn;
    let dim = Dim::decode(t)?;
    let coords = integer_coords(t, insn.src_a_reg(), dim);

    let mut meta = insn.src_b_reg();
    let mut next = |t: &mut Translator<'_>| {
        let v = t.x(meta);
        meta = meta.offset(1);
        v
    };
    let handle = if bindless {
        next(t)
    } else {
        Value::U32(insn.bits(36, 13) as u32 * 4)
    };
    let lod = if insn.bit(55) { next(t) } else { Value::U32(0) };
    let offset = if insn.bit(35) {
        packed_offset(t, &mut meta, dim, 4, 4)?
    } else {
        Value::Void
    };
    let multisample = if insn.bit(50) { t.x(meta) } else { Value::Void };
    if insn.bit(54) {
        return Err(t.unsupported());
    }
    let info = TextureInstInfo {
        ty: dim.ty(),
        ..Default::default()
    };
    let sample = image::fetch(&mut t.ir, handle, coords, offset, lod, multisample, info);
    store_masked(t, insn.dest_reg(), sample);
    set_sparse_pred(t, sample);
    Ok(())
}

pub(super) fn tld(t: &mut Translator<'_>) -> TResult {
    tld_impl(t, false)
}

pub(super) fn tld_b(t: &mut Translator<'_>) -> TResult {
    tld_impl(t, true)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tld_2d_lod_offset() {
        let insn = encode(
            Opcode::Tld,
            (1 << 55)
                | (7 << 51)
                | (4 << 36)
                | (1 << 35)
                | (0b0100 << 31)
                | (2 << 28)
                | (6 << 20)
                | (2 << 8)
                | 3,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = CompositeConstructU32x2 %0, %1
  %3 = GetRegister R6
  %4 = GetRegister R7
  %5 = BitFieldSExtract %4, #0x0, #0x4
  %6 = BitFieldSExtract %4, #0x4, #0x4
  %7 = CompositeConstructU32x2 %5, %6
  %8 = BoundImageFetch #0x10, %2, %7, %3, void (flags 0x8000)
  %9 = CompositeExtractF32x4 %8, #0x2
  %10 = BitCastU32F32 %9
  SetRegister R3, %10
"
        );
    }

    #[test]
    fn tld_b_array_1d_multisample_sparse() {
        let insn = encode(
            Opcode::TldB,
            (1 << 50) | (3 << 51) | (1 << 28) | (8 << 20) | (2 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R3
  %1 = GetRegister R2
  %2 = BitFieldUExtract %1, #0x0, #0x10
  %3 = CompositeConstructU32x2 %0, %2
  %4 = GetRegister R8
  %5 = GetRegister R9
  %6 = BindlessImageFetch %4, %3, void, #0x0, %5 (flags 0x4000)
  %7 = GetSparseFromOp %6
  %8 = LogicalNot %7
  SetPred P3, %8
"
        );
    }

    #[test]
    fn tld_rejects_clamp_and_cube_offset() {
        assert!(translate(&[encode(Opcode::Tld, 1 << 54)]).is_err());
        assert!(translate(&[encode(Opcode::Tld, (1 << 35) | (6 << 28))]).is_err());
    }
}
