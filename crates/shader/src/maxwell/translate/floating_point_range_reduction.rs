//! RRO: range reduction ahead of MUFU, a no-op beyond its modifiers on the host.

use super::{TResult, Translator};
use crate::ir::Value;

fn rro(t: &mut Translator<'_>, src: Value) -> TResult {
    let i = t.insn;
    let value = t.ir.fp_abs_neg(src, i.bit(49), i.bit(45));
    t.set_f(i.dest_reg(), value);
    Ok(())
}

pub(super) fn rro_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_float_cbuf()?;
    rro(t, src)
}

pub(super) fn rro_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_float_reg20();
    rro(t, src)
}

pub(super) fn rro_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn rro_reg_abs_neg() {
        check(
            encode(Opcode::RroReg, 2 | 6 << 20 | 1 << 45 | 1 << 49),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAbs32 %1",
                "%3 = FPNeg32 %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn rro_cbuf() {
        check(
            encode(Opcode::RroCbuf, 2 | 1 << 34 | 3 << 20),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = BitCastU32F32 %0",
                "SetRegister R2, %1",
            ],
        );
    }

    #[test]
    fn rro_imm_is_unsupported() {
        rejects(encode(Opcode::RroImm, 2));
    }
}
