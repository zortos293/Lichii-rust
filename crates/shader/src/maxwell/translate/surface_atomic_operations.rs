//! SUATOM and SURED: 32-bit surface atomics.

use super::surface_load_store::{SurfaceDim, surface_coords, surface_handle};
use super::texture_fetch::image;
use super::{TResult, Translator};
use crate::ir::{ImageFormat, Opcode, Reg, TextureInstInfo};

/// (bound, bindless) opcode pairs of the atomic operations, indexed by the encoded op; MIN and
/// MAX have signed and unsigned entries.
const fn atomic_ops(op: u64, signed: bool) -> Option<(Opcode, Opcode)> {
    use Opcode::*;
    Some(match (op, signed) {
        (0, _) => (BoundImageAtomicIAdd32, BindlessImageAtomicIAdd32),
        (1, true) => (BoundImageAtomicSMin32, BindlessImageAtomicSMin32),
        (1, false) => (BoundImageAtomicUMin32, BindlessImageAtomicUMin32),
        (2, true) => (BoundImageAtomicSMax32, BindlessImageAtomicSMax32),
        (2, false) => (BoundImageAtomicUMax32, BindlessImageAtomicUMax32),
        (3, _) => (BoundImageAtomicInc32, BindlessImageAtomicInc32),
        (4, _) => (BoundImageAtomicDec32, BindlessImageAtomicDec32),
        (5, _) => (BoundImageAtomicAnd32, BindlessImageAtomicAnd32),
        (6, _) => (BoundImageAtomicOr32, BindlessImageAtomicOr32),
        (7, _) => (BoundImageAtomicXor32, BindlessImageAtomicXor32),
        (8, _) => (BoundImageAtomicExchange32, BindlessImageAtomicExchange32),
        _ => return None,
    })
}

fn image_atom_op(
    t: &mut Translator<'_>,
    dest: Option<Reg>,
    operand: Reg,
    op: u64,
    clamp: u64,
    size: u64,
    bindless: bool,
) -> TResult {
    if clamp != 0 {
        return Err(t.unsupported());
    }
    let signed = match size {
        0 | 6 => false,
        1 => true,
        _ => return Err(t.unsupported()),
    };
    let Some(ops) = atomic_ops(op, signed) else {
        return Err(t.unsupported());
    };
    let dim = SurfaceDim::decode(t)?;
    let coords = surface_coords(t, t.insn.src_a_reg(), dim);
    let handle = surface_handle(t, !bindless);
    let info = TextureInstInfo {
        ty: dim.ty(),
        image_format: ImageFormat::R32Uint,
        ..Default::default()
    };
    let value = t.x(operand);
    let result = image::atomic(&mut t.ir, ops, handle, coords, value, info);
    if let Some(dest) = dest {
        t.set_x(dest, result);
    }
    Ok(())
}

pub(super) fn sured(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    image_atom_op(
        t,
        None,
        insn.dest_reg(),
        insn.bits(24, 3),
        insn.bits(49, 2),
        insn.bits(20, 3),
        !insn.bit(51),
    )
}

pub(super) fn suatom(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    image_atom_op(
        t,
        Some(insn.dest_reg()),
        insn.src_b_reg(),
        insn.bits(29, 4),
        insn.bits(49, 2),
        insn.bits(51, 3),
        insn.bit(54),
    )
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn suatom_signed_min_bound_2d() {
        let insn = encode(
            Opcode::Suatom,
            (1 << 51) | (4 << 36) | (3 << 33) | (1 << 29) | (7 << 20) | (2 << 8) | 5,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = CompositeConstructU32x2 %0, %1
  %3 = GetRegister R7
  %4 = BoundImageAtomicSMin32 #0x10, %2, %3 (flags 0x14008000)
  SetRegister R5, %4
"
        );
    }

    #[test]
    fn suatom_bindless_exchange_buffer() {
        let insn = encode(
            Opcode::Suatom,
            (1 << 54) | (6 << 51) | (10 << 39) | (1 << 33) | (8 << 29) | (7 << 20) | (2 << 8) | 5,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R10
  %2 = GetRegister R7
  %3 = BindlessImageAtomicExchange32 %1, %0, %2 (flags 0x1401c000)
  SetRegister R5, %3
"
        );
    }

    #[test]
    fn sured_unsigned_max_discards_result() {
        let insn = encode(
            Opcode::Sured,
            (1 << 51) | (2 << 36) | (2 << 24) | (2 << 8) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R4
  %2 = BoundImageAtomicUMax32 #0x8, %0, %1 (flags 0x14000000)
"
        );
    }

    #[test]
    fn surface_atomics_reject_unsupported_fields() {
        assert!(translate(&[encode(Opcode::Suatom, 2 << 51)]).is_err());
        assert!(translate(&[encode(Opcode::Suatom, 9 << 29)]).is_err());
        assert!(translate(&[encode(Opcode::Suatom, 1 << 49)]).is_err());
        assert!(translate(&[encode(Opcode::Suatom, 6 << 33)]).is_err());
        assert!(translate(&[encode(Opcode::Sured, 4 << 20)]).is_err());
    }
}
