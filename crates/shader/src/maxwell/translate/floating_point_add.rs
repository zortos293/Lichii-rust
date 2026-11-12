//! FADD and FADD32I.

use super::{TResult, Translator};
use crate::ir::{FmzMode, FpControl, FpRounding, Value};

/// Maxwell's 2-bit rounding field (RN, RM, RP, RZ).
pub(super) const fn fp_rounding(bits: u64) -> FpRounding {
    match bits & 3 {
        0 => FpRounding::Rn,
        1 => FpRounding::Rm,
        2 => FpRounding::Rp,
        _ => FpRounding::Rz,
    }
}

pub(super) const fn ftz_mode(ftz: bool) -> FmzMode {
    if ftz { FmzMode::Ftz } else { FmzMode::None }
}

struct Fadd {
    sat: bool,
    ftz: bool,
    rounding: FpRounding,
    abs_a: bool,
    neg_a: bool,
    abs_b: bool,
    neg_b: bool,
}

fn fadd(t: &mut Translator<'_>, src_b: Value, m: Fadd) -> TResult {
    let src_a = t.f(t.insn.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src_a, m.abs_a, m.neg_a);
    let op_b = t.ir.fp_abs_neg(src_b, m.abs_b, m.neg_b);
    let control = FpControl {
        no_contraction: true,
        rounding: m.rounding,
        fmz_mode: ftz_mode(m.ftz),
    };
    let mut value = t.ir.fp_add(op_a, op_b, control);
    if m.sat {
        value = t.ir.fp_saturate(value);
    }
    t.set_f(t.insn.dest_reg(), value);
    Ok(())
}

fn fadd_common(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    if i.bit(47) {
        return Err(t.unsupported());
    }
    let m = Fadd {
        sat: i.bit(50),
        ftz: i.bit(44),
        rounding: fp_rounding(i.bits(39, 2)),
        abs_a: i.bit(46),
        neg_a: i.bit(48),
        abs_b: i.bit(49),
        neg_b: i.bit(45),
    };
    fadd(t, src_b, m)
}

pub(super) fn fadd_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    fadd_common(t, src_b)
}

pub(super) fn fadd_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    fadd_common(t, src_b)
}

pub(super) fn fadd_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    fadd_common(t, src_b)
}

pub(super) fn fadd32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if i.bit(52) {
        return Err(t.unsupported());
    }
    let m = Fadd {
        sat: false,
        ftz: i.bit(55),
        rounding: FpRounding::Rn,
        abs_a: i.bit(54),
        neg_a: i.bit(56),
        abs_b: i.bit(57),
        neg_b: i.bit(53),
    };
    let src_b = t.get_float_imm32();
    fadd(t, src_b, m)
}

#[cfg(test)]
pub(super) mod tests {
    use lichii_common::GpuVAddr;

    use super::super::tests::{TestEnv, encode, translate};
    use super::super::translate_block;
    use crate::ir::Program;
    use crate::maxwell::opcode::Opcode;
    use crate::{ShaderArch, ShaderEnvironment, ShaderStage};

    /// Asserts the trimmed dump lines of `insn`, without the block header.
    pub(in super::super) fn check(insn: u64, expected: &[&str]) {
        let dump = translate(&[insn]).unwrap();
        let got: Vec<&str> = dump.lines().skip(1).map(str::trim).collect();
        assert_eq!(got, expected);
    }

    pub(in super::super) fn rejects(insn: u64) {
        assert!(translate(&[insn]).is_err());
    }

    /// A test environment that reports lowered native (Ampere) code.
    struct NativeEnv(TestEnv);

    impl ShaderEnvironment for NativeEnv {
        fn arch(&self) -> ShaderArch {
            ShaderArch::Ampere
        }
        fn stage(&self) -> ShaderStage {
            self.0.stage()
        }
        fn start_address(&self) -> GpuVAddr {
            self.0.start_address()
        }
        fn read_u64(&self, address: GpuVAddr) -> u64 {
            self.0.read_u64(address)
        }
        fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
            self.0.read_cbuf(index, offset)
        }
        fn shared_memory_size(&self) -> u32 {
            self.0.shared_memory_size()
        }
        fn workgroup_size(&self) -> [u32; 3] {
            self.0.workgroup_size()
        }
    }

    /// Like [`check`], for an instruction in a lowered native program.
    pub(in super::super) fn check_native(insn: u64, expected: &[&str]) {
        let env = NativeEnv(TestEnv::new(&[insn]));
        let mut prog = Program::new();
        let block = prog.new_block();
        translate_block(&mut prog, block, &env, 0, env.0.end()).unwrap();
        let dump = prog.dump();
        let got: Vec<&str> = dump.lines().skip(1).map(str::trim).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn fadd_reg_sat_ftz_abs() {
        check(
            encode(
                Opcode::FaddReg,
                2 | 4 << 8 | 6 << 20 | 1 << 44 | 1 << 46 | 1 << 50 | 3 << 39,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPAbs32 %3",
                "%5 = FPAdd32 %4, %1 (flags 0x19)",
                "%6 = FPSaturate32 %5",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fadd_cbuf() {
        check(
            encode(
                Opcode::FaddCbuf,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 1 << 45 | 1 << 48,
            ),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = FPNeg32 %2",
                "%4 = FPNeg32 %0",
                "%5 = FPAdd32 %3, %4 (flags 0x33)",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn fadd_imm() {
        check(
            encode(Opcode::FaddImm, 2 | 4 << 8 | 0x3f800 << 20 | 1 << 56),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAdd32 %1, #-1.0f (flags 0x33)",
                "%3 = BitCastU32F32 %2",
                "SetRegister R2, %3",
            ],
        );
    }

    #[test]
    fn fadd32i() {
        check(
            encode(
                Opcode::Fadd32i,
                2 | 4 << 8 | 0x3f80_0000 << 20 | 1 << 55 | 1 << 56 | 1 << 57,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = FPAbs32 #1.0f",
                "%4 = FPAdd32 %2, %3 (flags 0x13)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fadd_rejects_cc() {
        rejects(encode(Opcode::FaddReg, 2 | 4 << 8 | 6 << 20 | 1 << 47));
        rejects(encode(Opcode::Fadd32i, 2 | 4 << 8 | 1 << 52));
        rejects(encode(Opcode::FaddCbuf, 2 | 4 << 8 | 18 << 34));
    }
}
