//! x86-64 code for `Op::Simd`.
//!
//! A checked op computes on the host only where the host's result is the Arm ARM's: FPCR selects
//! round to nearest without flush to zero, FPSR.IXC is already set when the op can be inexact,
//! and the result is not a NaN, an infinity or (for ops that can underflow) below the normal
//! range. Any other case takes the op's slow path, which runs the instruction through
//! `CallSimd` and reloads its result from the context.

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, dynasm};

use super::super::ir::{Loc, Ty, Value};
use super::super::x64::{CTX_FPCR, CTX_FPSR, CTX_PSTATE, Emitter, loc_offset};
use super::ir::{Arith, Fp, Fused, SimdOp};

/// FPCR.RMode and FPCR.FZ.
pub(super) const HOST_FPCR: i32 = 3 << 22 | 1 << 24;
/// FPSR.IXC.
pub(super) const IXC: i8 = 0x10;
/// An H value with exponent 31 (an infinity or a NaN, or under FPCR.AHP a normal number) falls
/// back; every other H value means the same in both formats and converts exactly, so FPCR is not
/// checked at all.
///
/// 2^112 as an S value and 2^1008 as a D value: the exponent bias differences from H. A finite H
/// value's magnitude bits, shifted into the S or D position, times this scale is the exact
/// converted value, subnormals included.
#[repr(align(16))]
struct HalfScale([u32; 4]);

static HALF_SCALE: HalfScale = HalfScale([0x7780_0000, 0, 0, 0x7ef0_0000]);

#[derive(Default)]
pub(in super::super) struct State {
    slow: Vec<Slow>,
}

/// The out-of-line path of a checked op: runs `word` through `CallSimd` and reloads the result.
struct Slow {
    label: DynamicLabel,
    resume: DynamicLabel,
    word: u32,
    dst: u8,
    ty: Ty,
    saved: [u16; 2],
}

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

    /// Starts a checked op and returns its slow-path label. The host computes only under round to
    /// nearest without flush to zero, and only reports inexact results by leaving FPSR.IXC alone,
    /// so an op that can be inexact also falls back while IXC is clear.
    pub(super) fn checked(&mut self, inexact: bool) -> DynamicLabel {
        let slow = self.asm.new_dynamic_label();
        dynasm!(self.asm
            ; .arch x64
            ; test DWORD [r15 + CTX_FPCR], HOST_FPCR
            ; jnz =>slow
        );
        if inexact {
            dynasm!(self.asm
                ; .arch x64
                ; test BYTE [r15 + CTX_FPSR], IXC
                ; jz =>slow
            );
        }
        slow
    }

    /// Jumps to `slow` unless the FP result in `xmm15` is finite and, when `normal`, a normal
    /// number: a zero or subnormal result may have underflowed, and the Arm ARM detects tininess
    /// before rounding, so a result of exactly the smallest normal falls back too. A zero result
    /// stays native when one of `zero_factors` is zero, because a product or quotient with a zero
    /// factor or dividend is exactly zero. Leaves the result's bits in `rax`.
    fn check_result(&mut self, fp: Fp, normal: bool, zero_factors: &[u8], slow: DynamicLabel) {
        match fp {
            Fp::S => dynasm!(self.asm
                ; .arch x64
                ; movd eax, xmm15
                ; mov ecx, eax
                ; and ecx, 0x7fff_ffff
            ),
            Fp::D => dynasm!(self.asm
                ; .arch x64
                ; movq rax, xmm15
                ; mov rcx, rax
                ; btr rcx, 63
            ),
        }
        if !normal {
            match fp {
                Fp::S => dynasm!(self.asm ; .arch x64 ; cmp ecx, 0x7f7f_ffff ; ja =>slow),
                Fp::D => dynasm!(self.asm
                    ; .arch x64
                    ; mov rdx, QWORD 0x7fef_ffff_ffff_ffff
                    ; cmp rcx, rdx
                    ; ja =>slow
                ),
            }
            return;
        }
        let (zero, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
        let below = if zero_factors.is_empty() { slow } else { zero };
        match fp {
            Fp::S => dynasm!(self.asm
                ; .arch x64
                ; sub ecx, 0x0080_0001
                ; cmp ecx, 0x7f7f_ffff - 0x0080_0001
                ; jbe =>done
                ; cmp ecx, -0x0080_0001
                ; je =>below
                ; jmp =>slow
            ),
            Fp::D => dynasm!(self.asm
                ; .arch x64
                ; mov rdx, QWORD 0x0010_0000_0000_0001
                ; sub rcx, rdx
                ; mov rdx, QWORD 0x7fef_ffff_ffff_ffff - 0x0010_0000_0000_0001
                ; cmp rcx, rdx
                ; jbe =>done
                ; mov rdx, QWORD -0x0010_0000_0000_0001
                ; cmp rcx, rdx
                ; je =>below
                ; jmp =>slow
            ),
        }
        if let Some((&last, rest)) = zero_factors.split_last() {
            dynasm!(self.asm ; .arch x64 ; =>zero ; xorps xmm15, xmm15);
            for &f in rest {
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; ucomiss xmm15, Rx(f)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; ucomisd xmm15, Rx(f)),
                }
                dynasm!(self.asm ; .arch x64 ; je =>done);
            }
            match fp {
                Fp::S => dynasm!(self.asm ; .arch x64 ; ucomiss xmm15, Rx(last)),
                Fp::D => dynasm!(self.asm ; .arch x64 ; ucomisd xmm15, Rx(last)),
            }
            dynasm!(self.asm ; .arch x64 ; jne =>slow);
        }
        dynasm!(self.asm ; .arch x64 ; =>done);
    }

    /// `FMAX`, `FMIN`, `FMAXNM` or `FMINNM` of `xmm15` (a copy of `a`) and `b` into `xmm15`. Once a NaN
    /// input has jumped to `slow`, every result is one of the inputs, exact and without flags.
    /// Equal inputs are combined bitwise, because x86 returns the second operand for +0 and -0
    /// where Arm returns +0 for the maximum and -0 for the minimum.
    fn min_max(&mut self, max: bool, fp: Fp, a: u8, b: u8, slow: DynamicLabel) {
        let (different, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; ucomiss Rx(a), Rx(b)),
            Fp::D => dynasm!(self.asm ; .arch x64 ; ucomisd Rx(a), Rx(b)),
        }
        dynasm!(self.asm ; .arch x64 ; jp =>slow ; jne =>different);
        if max {
            dynasm!(self.asm ; .arch x64 ; andps xmm15, Rx(b));
        } else {
            dynasm!(self.asm ; .arch x64 ; orps xmm15, Rx(b));
        }
        dynasm!(self.asm ; .arch x64 ; jmp =>done ; =>different);
        match (max, fp) {
            (true, Fp::S) => dynasm!(self.asm ; .arch x64 ; maxss xmm15, Rx(b)),
            (true, Fp::D) => dynasm!(self.asm ; .arch x64 ; maxsd xmm15, Rx(b)),
            (false, Fp::S) => dynasm!(self.asm ; .arch x64 ; minss xmm15, Rx(b)),
            (false, Fp::D) => dynasm!(self.asm ; .arch x64 ; minsd xmm15, Rx(b)),
        }
        dynasm!(self.asm ; .arch x64 ; =>done);
    }

    /// The NZCV of an ordered compare of `xmm a` with `xmm b` in `d`; unordered jumps to `slow`.
    /// Clobbers `rax`, `rcx` and `rdx`.
    fn compare_flags(&mut self, fp: Fp, a: u8, b: u8, d: u8, slow: DynamicLabel) {
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; ucomiss Rx(a), Rx(b)),
            Fp::D => dynasm!(self.asm ; .arch x64 ; ucomisd Rx(a), Rx(b)),
        }
        dynasm!(self.asm
            ; .arch x64
            ; jp =>slow
            ; setb al
            ; setz cl
            ; movzx eax, al
            ; movzx ecx, cl
            ; lea edx, [rax * 8]
            ; lea edx, [rdx + rcx * 4]
            ; xor eax, 1
            ; lea Rd(d), [rdx + rax * 2]
        );
    }

    pub(super) fn finish_checked(
        &mut self,
        i: usize,
        slow: DynamicLabel,
        word: u32,
        dst: u8,
        ty: Ty,
    ) {
        let resume = self.asm.new_dynamic_label();
        dynasm!(self.asm ; .arch x64 ; =>resume);
        let saved = self.live_across(i);
        self.simd.slow.push(Slow {
            label: slow,
            resume,
            word,
            dst,
            ty,
            saved,
        });
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
            SimdOp::Select(fp, c, a, b) if self.fused[c.index()] => {
                let (a, b) = (self.reg(a), self.reg(b));
                let d = self.def(v);
                let cc = self.compare(c);
                self.low_to_rax(fp, b);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; movd ecx, Rx(a)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; movq rcx, Rx(a)),
                }
                self.asm.extend([0x48, 0x0f, 0x40 | cc, 0xc1]);
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
            SimdOp::Arith { op, fp, a, b, word } => {
                let (a, b) = (self.reg(a), self.reg(b));
                let d = self.def(v);
                let extreme = matches!(op, Arith::Max | Arith::Min);
                let slow = self.checked(!extreme);
                dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(a));
                match (op, fp) {
                    (Arith::Max | Arith::Min, _) => self.min_max(op == Arith::Max, fp, a, b, slow),
                    (Arith::Add, Fp::S) => dynasm!(self.asm ; .arch x64 ; addss xmm15, Rx(b)),
                    (Arith::Add, Fp::D) => dynasm!(self.asm ; .arch x64 ; addsd xmm15, Rx(b)),
                    (Arith::Sub, Fp::S) => dynasm!(self.asm ; .arch x64 ; subss xmm15, Rx(b)),
                    (Arith::Sub, Fp::D) => dynasm!(self.asm ; .arch x64 ; subsd xmm15, Rx(b)),
                    (Arith::Mul, Fp::S) => dynasm!(self.asm ; .arch x64 ; mulss xmm15, Rx(b)),
                    (Arith::Mul, Fp::D) => dynasm!(self.asm ; .arch x64 ; mulsd xmm15, Rx(b)),
                    (Arith::Div, Fp::S) => dynasm!(self.asm ; .arch x64 ; divss xmm15, Rx(b)),
                    (Arith::Div, Fp::D) => dynasm!(self.asm ; .arch x64 ; divsd xmm15, Rx(b)),
                    (Arith::Nmul, Fp::S) => dynasm!(self.asm ; .arch x64 ; mulss xmm15, Rx(b)),
                    (Arith::Nmul, Fp::D) => dynasm!(self.asm ; .arch x64 ; mulsd xmm15, Rx(b)),
                }
                match op {
                    Arith::Add | Arith::Sub => self.check_result(fp, false, &[], slow),
                    Arith::Mul | Arith::Nmul => self.check_result(fp, true, &[a, b], slow),
                    Arith::Div => self.check_result(fp, true, &[a], slow),
                    Arith::Max | Arith::Min => self.low_to_rax(fp, 15),
                }
                match (op, fp) {
                    (Arith::Nmul, Fp::S) => {
                        dynasm!(self.asm ; .arch x64 ; xor eax, 0x8000_0000u32 as i32)
                    }
                    (Arith::Nmul, Fp::D) => dynasm!(self.asm ; .arch x64 ; btc rax, 63),
                    _ => {}
                }
                self.rax_to_low(fp, d);
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::Fused {
                op,
                fp,
                n,
                m,
                a,
                word,
            } => {
                let (n, m, a) = (self.reg(n), self.reg(m), self.reg(a));
                let d = self.def(v);
                let slow = self.checked(true);
                if self.target.fma {
                    dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(a));
                    match (op, fp) {
                        (Fused::Madd, Fp::S) => {
                            dynasm!(self.asm ; .arch x64 ; vfmadd231ss xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Madd, Fp::D) => {
                            dynasm!(self.asm ; .arch x64 ; vfmadd231sd xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Msub, Fp::S) => {
                            dynasm!(self.asm ; .arch x64 ; vfnmadd231ss xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Msub, Fp::D) => {
                            dynasm!(self.asm ; .arch x64 ; vfnmadd231sd xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Nmadd, Fp::S) => {
                            dynasm!(self.asm ; .arch x64 ; vfnmsub231ss xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Nmadd, Fp::D) => {
                            dynasm!(self.asm ; .arch x64 ; vfnmsub231sd xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Nmsub, Fp::S) => {
                            dynasm!(self.asm ; .arch x64 ; vfmsub231ss xmm15, Rx(n), Rx(m))
                        }
                        (Fused::Nmsub, Fp::D) => {
                            dynasm!(self.asm ; .arch x64 ; vfmsub231sd xmm15, Rx(n), Rx(m))
                        }
                    }
                    self.check_result(fp, true, &[n, m], slow);
                    self.rax_to_low(fp, d);
                } else {
                    dynasm!(self.asm ; .arch x64 ; jmp =>slow);
                }
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::Sqrt { fp, a, word } => {
                let a = self.reg(a);
                let d = self.def(v);
                let slow = self.checked(true);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; sqrtss xmm15, Rx(a)),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; sqrtsd xmm15, Rx(a)),
                }
                self.check_result(fp, false, &[], slow);
                self.rax_to_low(fp, d);
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::FromHalf { fp, a, word } => {
                let a = self.reg(a);
                let d = self.def(v);
                let slow = self.asm.new_dynamic_label();
                dynasm!(self.asm
                    ; .arch x64
                    ; movd eax, Rx(a)
                    ; mov ecx, eax
                    ; and ecx, 0x7c00
                    ; cmp ecx, 0x7c00
                    ; je =>slow
                    ; mov ecx, eax
                    ; and ecx, 0x8000
                    ; and eax, 0x7fff
                    ; mov rdx, QWORD HALF_SCALE.0.as_ptr() as i64
                );
                match fp {
                    Fp::S => dynasm!(self.asm
                        ; .arch x64
                        ; shl eax, 13
                        ; movd xmm15, eax
                        ; mulss xmm15, [rdx]
                        ; movd eax, xmm15
                        ; shl ecx, 16
                        ; or eax, ecx
                    ),
                    Fp::D => dynasm!(self.asm
                        ; .arch x64
                        ; shl rax, 42
                        ; movq xmm15, rax
                        ; mulsd xmm15, [rdx + 8]
                        ; movq rax, xmm15
                        ; shl rcx, 48
                        ; or rax, rcx
                    ),
                }
                self.rax_to_low(fp, d);
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::Convert { fp, a, word } => {
                let a = self.reg(a);
                let d = self.def(v);
                let narrowing = fp == Fp::D;
                let slow = self.checked(narrowing);
                let to = match fp {
                    Fp::S => {
                        dynasm!(self.asm ; .arch x64 ; cvtss2sd xmm15, Rx(a));
                        Fp::D
                    }
                    Fp::D => {
                        dynasm!(self.asm ; .arch x64 ; cvtsd2ss xmm15, Rx(a));
                        Fp::S
                    }
                };
                self.check_result(to, narrowing, &[], slow);
                self.rax_to_low(to, d);
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::FromInt {
                fp,
                signed,
                wide,
                x,
                word,
            } => {
                let x = self.reg(x);
                let d = self.def(v);
                let slow = self.checked(true);
                match (signed, wide) {
                    (true, true) => dynasm!(self.asm ; .arch x64 ; mov rax, Rq(x)),
                    (true, false) => dynasm!(self.asm ; .arch x64 ; movsxd rax, Rd(x)),
                    (false, false) => dynasm!(self.asm ; .arch x64 ; mov eax, Rd(x)),
                    (false, true) => dynasm!(self.asm
                        ; .arch x64
                        ; mov rax, Rq(x)
                        ; test rax, rax
                        ; js =>slow
                    ),
                }
                dynasm!(self.asm ; .arch x64 ; xorps xmm15, xmm15);
                match fp {
                    Fp::S => dynasm!(self.asm ; .arch x64 ; cvtsi2ss xmm15, rax),
                    Fp::D => dynasm!(self.asm ; .arch x64 ; cvtsi2sd xmm15, rax),
                }
                self.low_to_rax(fp, 15);
                self.rax_to_low(fp, d);
                self.finish_checked(i, slow, word, d, Ty::U128);
            }
            SimdOp::ToInt {
                fp,
                signed,
                wide,
                a,
                word,
            } => {
                let a = self.reg(a);
                let d = self.def(v);
                let slow = self.checked(true);
                match (fp, signed && !wide) {
                    (Fp::S, true) => dynasm!(self.asm ; .arch x64 ; cvttss2si eax, Rx(a)),
                    (Fp::D, true) => dynasm!(self.asm ; .arch x64 ; cvttsd2si eax, Rx(a)),
                    (Fp::S, false) => dynasm!(self.asm ; .arch x64 ; cvttss2si rax, Rx(a)),
                    (Fp::D, false) => dynasm!(self.asm ; .arch x64 ; cvttsd2si rax, Rx(a)),
                }
                match (signed, wide) {
                    (true, false) => dynasm!(self.asm
                        ; .arch x64
                        ; cmp eax, 0x8000_0000u32 as i32
                        ; je =>slow
                    ),
                    (true, true) => dynasm!(self.asm
                        ; .arch x64
                        ; mov rcx, rax
                        ; rol rcx, 1
                        ; cmp rcx, 1
                        ; je =>slow
                    ),
                    (false, false) => dynasm!(self.asm
                        ; .arch x64
                        ; mov rcx, rax
                        ; shr rcx, 32
                        ; jnz =>slow
                    ),
                    (false, true) => dynasm!(self.asm ; .arch x64 ; test rax, rax ; js =>slow),
                }
                match wide {
                    true => dynasm!(self.asm ; .arch x64 ; mov Rq(d), rax),
                    false => dynasm!(self.asm ; .arch x64 ; mov Rd(d), eax),
                }
                self.finish_checked(i, slow, word, d, Ty::U64);
            }
            SimdOp::Compare { fp, a, b, word } => {
                let a = self.reg(a);
                let b = match b {
                    Some(b) => self.reg(b),
                    None => {
                        dynasm!(self.asm ; .arch x64 ; xorps xmm15, xmm15);
                        15
                    }
                };
                let d = self.def(v);
                let slow = self.checked(false);
                self.compare_flags(fp, a, b, d, slow);
                self.finish_checked(i, slow, word, d, Ty::Nzcv);
            }
            SimdOp::CondCompare {
                fp,
                c,
                a,
                b,
                nzcv,
                word,
            } => {
                let (c, a, b) = (self.reg(c), self.reg(a), self.reg(b));
                let d = self.def(v);
                let (skip, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
                dynasm!(self.asm ; .arch x64 ; test Rd(c), Rd(c) ; jz =>skip);
                let slow = self.checked(false);
                self.compare_flags(fp, a, b, d, slow);
                dynasm!(self.asm
                    ; .arch x64
                    ; jmp =>done
                    ; =>skip
                    ; mov Rd(d), nzcv as i32
                    ; =>done
                );
                self.finish_checked(i, slow, word, d, Ty::Nzcv);
            }
            SimdOp::SatNarrow { size, n, low, .. } => self.sat_narrow(i, size, n, low),
            SimdOp::Vec(op) => self.vector(i, op),
            SimdOp::Lanes(lanes) => self.lanes(i, lanes),
        }
    }

    /// Emits the slow paths of every checked op in the block.
    pub(in super::super) fn simd_slow_paths(&mut self) {
        for slow in std::mem::take(&mut self.simd.slow) {
            dynasm!(self.asm ; .arch x64 ; =>slow.label);
            if slow.ty == Ty::Nzcv {
                self.materialize_nzcv();
            }
            self.save(slow.saved);
            dynasm!(self.asm
                ; .arch x64
                ; mov rdi, r15
                ; mov esi, slow.word as i32
                ; mov rdx, r13
                ; xor ecx, ecx
                ; mov rax, QWORD self.target.simd_slow as i64
                ; sub rsp, 8
                ; call rax
                ; add rsp, 8
            );
            self.restore(slow.saved);
            let (d, rd) = (slow.dst, (slow.word & 31) as u8);
            match slow.ty {
                Ty::U128 => {
                    let offset = loc_offset(Loc::V(rd));
                    dynasm!(self.asm ; .arch x64 ; movdqu Rx(d), [r15 + offset]);
                }
                Ty::Nzcv => dynasm!(self.asm
                    ; .arch x64
                    ; mov Rd(d), DWORD [r15 + CTX_PSTATE]
                    ; shr Rd(d), 28
                ),
                _ if rd == 31 => dynasm!(self.asm ; .arch x64 ; xor Rd(d), Rd(d)),
                _ => {
                    let offset = loc_offset(Loc::X(rd));
                    dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r15 + offset]);
                }
            }
            dynasm!(self.asm ; .arch x64 ; jmp =>slow.resume);
        }
    }
}
