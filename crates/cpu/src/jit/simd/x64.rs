//! x86-64 code for `Op::Simd`.

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use dynasmrt::{DynasmApi, dynasm};

use super::super::ir::Value;
use super::super::x64::Emitter;
use super::ir::{Fp, SimdOp};

impl Emitter<'_> {
    /// Moves the low element of `xmm src` into `rax`.
    fn low_to_rax(&mut self, fp: Fp, src: u8) {
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; movd eax, Rx(src)),
            Fp::D => dynasm!(self.asm ; .arch x64 ; movq rax, Rx(src)),
        }
    }

    /// Writes the low element in `rax` to `xmm dst`, zeroing the rest of it.
    fn rax_to_low(&mut self, fp: Fp, dst: u8) {
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; movd Rx(dst), eax),
            Fp::D => dynasm!(self.asm ; .arch x64 ; movq Rx(dst), rax),
        }
    }

    pub(in super::super) fn simd(&mut self, i: usize, op: SimdOp) {
        let v = Value(i as u32);
        match op {
            SimdOp::ToGpr(fp, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; movd Rd(d), Rx(a)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; movq Rq(d), Rx(a)),
                }
            }
            SimdOp::FromGpr(fp, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; movd Rx(d), Rd(a)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rq(a)),
                }
            }
            SimdOp::Move(fp, a) | SimdOp::Abs(fp, a) | SimdOp::Neg(fp, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                self.low_to_rax(fp, a);
                match (op, fp) {
                    (SimdOp::Abs(..), Fp::S) => {
                        dynasm!(self.asm ; .arch x64 ; and eax, 0x7fff_ffff)
                    }
                    (SimdOp::Abs(..), Fp::D) => dynasm!(self.asm ; .arch x64 ; btr rax, 63),
                    (SimdOp::Neg(..), Fp::S) => {
                        dynasm!(self.asm ; .arch x64 ; xor eax, 0x8000_0000u32 as i32)
                    }
                    (SimdOp::Neg(..), Fp::D) => dynasm!(self.asm ; .arch x64 ; btc rax, 63),
                    _ => {}
                }
                self.rax_to_low(fp, d);
            }
            SimdOp::Select(fp, c, a, b) => {
                let (c, a, b) = (self.reg(c), self.reg(a), self.reg(b));
                let d = self.def(v);
                self.low_to_rax(fp, b);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; movd ecx, Rx(a)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; movq rcx, Rx(a)),
                }
                dynasm!(self.asm
                    ; .arch x64
                    ; test Rd(c), Rd(c)
                    ; cmovnz rax, rcx
                );
                self.rax_to_low(fp, d);
            }
        }
    }
}
