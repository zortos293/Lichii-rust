//! MUFU.

use super::{TResult, Translator};
use crate::ShaderArch;
use crate::ir::Opcode;

/// Whether the program is lowered native (Ampere) code, which may use encodings Maxwell lacks.
pub(super) fn has_native_instructions(t: &Translator<'_>) -> bool {
    t.env.arch() == ShaderArch::Ampere
}

pub(super) fn mufu(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let op = match i.bits(20, 4) {
        0 => Opcode::FPCos,
        1 => Opcode::FPSin,
        2 => Opcode::FPExp2,
        3 => Opcode::FPLog2,
        4 => Opcode::FPRecip32,
        5 => Opcode::FPRecipSqrt32,
        8 => Opcode::FPSqrt,
        9 if has_native_instructions(t) => Opcode::FPTanh,
        _ => return Err(t.unsupported()),
    };
    let src = t.f(i.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src, i.bit(46), i.bit(48));
    let mut value = t.ir.fp_unary32(op, op_a);
    if i.bit(50) {
        value = t.ir.fp_saturate(value);
    }
    t.set_f(i.dest_reg(), value);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, check_native, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn mufu_rsq_abs_sat() {
        check(
            encode(Opcode::Mufu, 2 | 4 << 8 | 5 << 20 | 1 << 46 | 1 << 50),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAbs32 %1",
                "%3 = FPRecipSqrt32 %2",
                "%4 = FPSaturate32 %3",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn mufu_sin_negated() {
        check(
            encode(Opcode::Mufu, 2 | 4 << 8 | 1 << 20 | 1 << 48),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = FPSin %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn mufu_rejects_64_bit_and_native_only_ops() {
        rejects(encode(Opcode::Mufu, 2 | 4 << 8 | 6 << 20));
        rejects(encode(Opcode::Mufu, 2 | 4 << 8 | 7 << 20));
        rejects(encode(Opcode::Mufu, 2 | 4 << 8 | 9 << 20));
        rejects(encode(Opcode::Mufu, 2 | 4 << 8 | 10 << 20));
    }

    #[test]
    fn mufu_tanh_in_native_programs() {
        check_native(
            encode(Opcode::Mufu, 2 | 4 << 8 | 9 << 20),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPTanh %1",
                "%3 = BitCastU32F32 %2",
                "SetRegister R2, %3",
            ],
        );
    }
}
