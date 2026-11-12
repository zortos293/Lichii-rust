//! FSWZADD.

use super::floating_point_add::{fp_rounding, ftz_mode};
use super::{TResult, Translator};
use crate::ir::{FpControl, Opcode, Value};

pub(super) fn fswzadd(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    if i.bit(38) {
        tracing::warn!("(STUBBED) FSWZADD - NDV mode");
    }
    let src_a = t.get_float_reg8();
    let src_b = t.get_float_reg20();
    let swizzle = Value::U32(i.bits(28, 8) as u32);
    let control = FpControl {
        no_contraction: false,
        rounding: fp_rounding(i.bits(39, 2)),
        fmz_mode: ftz_mode(i.bit(44)),
    };
    let result = t.ir.inst_flags(
        Opcode::FSwizzleAdd,
        control.to_flags(),
        &[src_a, src_b, swizzle],
    );
    t.set_f(i.dest_reg(), result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fswzadd_ftz_rounding() {
        check(
            encode(
                Opcode::Fswzadd,
                2 | 4 << 8 | 6 << 20 | 0xa5 << 28 | 1 << 39 | 1 << 44,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = FSwizzleAdd %1, %3, #0xa5 (flags 0x14)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fswzadd_rejects_cc() {
        rejects(encode(Opcode::Fswzadd, 2 | 4 << 8 | 6 << 20 | 1 << 47));
    }
}
