//! Advanced SIMD floating-point arithmetic on S and D lanes, compiled as checked ops: every lane is
//! computed on the host, and the whole instruction falls back to `CallSimd` unless each lane the
//! instruction reads passes the scalar FPCR.FZ input check and each lane it writes passes the
//! scalar result checks (finite, and above the smallest normal or an exact zero for the ops that
//! can underflow).

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use super::super::ir::{Loc, Op, Ty, Value};
use super::super::translate::Translator;
use super::super::x64::Emitter;
use super::ir::{Fp, SimdOp};
use super::x64::Inputs;
use crate::interpreter::simd::softfloat::{recip_estimate_u9, recip_sqrt_estimate_u9};

const SCRATCH: u8 = 15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LaneOp {
    Add,
    /// `FADDP`: sums of adjacent lane pairs, those of `n` in the low lanes and those of `m` above.
    AddPairwise,
    Sub,
    Mul,
    /// Fused `acc + n * m`.
    Mla,
    /// Fused `acc - n * m`.
    Mls,
    /// `FRECPS`: fused `2 - n * m`.
    RecipStep,
    /// `FRSQRTS`: fused `(3 - n * m) / 2`.
    RsqrtStep,
    /// `SCVTF` (vector) on 32-bit lanes of `n`.
    FromInt,
    /// `FCMEQ`: all ones where `n == m`.
    CompareEq,
    /// `FCMGE`: all ones where `n >= m`.
    CompareGe,
    /// `FCMGT`: all ones where `n > m`.
    CompareGt,
    /// `FRECPE` on S lanes of `n`.
    RecipEstimate,
    /// `FRSQRTE` on S lanes of `n`.
    RsqrtEstimate,
}

/// A lane-wise op on `n` and `m`, or on `n` and lane `index` of `m` for the by-element forms. It
/// writes the low `lanes` lanes and zeroes the rest; one lane is the scalar by-element form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lanes {
    pub op: LaneOp,
    pub fp: Fp,
    pub lanes: u8,
    pub n: Value,
    pub m: Value,
    pub index: Option<u8>,
    /// The accumulator of `Mla` and `Mls`.
    pub acc: Option<Value>,
    pub word: u32,
}

impl Lanes {
    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        f(&mut self.n);
        f(&mut self.m);
        if let Some(acc) = &mut self.acc {
            f(acc);
        }
    }
}

/// 2.0, 3.0 and 0.5 in every lane, as S then as D: the `FRECPS` and `FRSQRTS` constants.
#[repr(align(16))]
struct Steps([u32; 24]);

static STEPS: Steps = Steps([
    0x4000_0000,
    0x4000_0000,
    0x4000_0000,
    0x4000_0000,
    0x4040_0000,
    0x4040_0000,
    0x4040_0000,
    0x4040_0000,
    0x3f00_0000,
    0x3f00_0000,
    0x3f00_0000,
    0x3f00_0000,
    0,
    0x4000_0000,
    0,
    0x4000_0000,
    0,
    0x4008_0000,
    0,
    0x4008_0000,
    0,
    0x3fe0_0000,
    0,
    0x3fe0_0000,
]);

/// The constants of [`Emitter::estimate`], 16 bytes each.
#[repr(align(16))]
struct EstimateConstants([u32; 36]);

const ABS: i32 = 0;
const MIN_NORMAL: i32 = 16;
const SIGN: i32 = 32;
const RSQRT_BOUND: i32 = 48;
const RECIP_BOUND: i32 = 64;
const RSQRT_EXP: i32 = 80;
const RECIP_EXP: i32 = 96;
const ONES: i32 = 112;
const EXP: i32 = 128;

static ESTIMATE: EstimateConstants = EstimateConstants({
    let values = [
        0x7fff_ffff,
        0x0080_0000,
        0x8000_0000,
        0x7f00_0000 ^ 0x8000_0000,
        0x7e00_0000 ^ 0x8000_0000,
        381,
        254,
        0xffff_ffff,
        0xff,
    ];
    let mut table = [0; 36];
    let mut i = 0;
    while i < 36 {
        table[i] = values[i / 4];
        i += 1;
    }
    table
});

/// The 8-bit `FRSQRTE` fraction of a positive normal S value, indexed by its bits 15..24: the low
/// exponent bit and the top 8 fraction bits.
static RSQRT_FRACTIONS: [u8; 512] = {
    let mut table = [0; 512];
    let mut i = 0;
    while i < 512 {
        let scaled = if i < 256 {
            256 | i
        } else {
            128 | (i & 0xff) >> 1
        };
        table[i] = recip_sqrt_estimate_u9(scaled as u64) as u8;
        i += 1;
    }
    table
};

/// The 8-bit `FRECPE` fraction of a normal S value, indexed by its top 8 fraction bits.
static RECIP_FRACTIONS: [u8; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = recip_estimate_u9(256 | i as u64) as u8;
        i += 1;
    }
    table
};

/// The absolute-value mask, infinity, the smallest normal, zero and a sum's input floor (see
/// [`Inputs`]) in every lane, for S then D.
#[repr(align(16))]
struct Bounds([u32; 40]);

static BOUNDS: Bounds = Bounds([
    0x7fff_ffff,
    0x7fff_ffff,
    0x7fff_ffff,
    0x7fff_ffff,
    0x7f80_0000,
    0x7f80_0000,
    0x7f80_0000,
    0x7f80_0000,
    0x0080_0000,
    0x0080_0000,
    0x0080_0000,
    0x0080_0000,
    0,
    0,
    0,
    0,
    0x0c00_0000,
    0x0c00_0000,
    0x0c00_0000,
    0x0c00_0000,
    0xffff_ffff,
    0x7fff_ffff,
    0xffff_ffff,
    0x7fff_ffff,
    0x0000_0000,
    0x7ff0_0000,
    0x0000_0000,
    0x7ff0_0000,
    0x0000_0000,
    0x0010_0000,
    0x0000_0000,
    0x0010_0000,
    0,
    0,
    0,
    0,
    0,
    0x0350_0000,
    0,
    0x0350_0000,
]);

/// The `BOUNDS` rows for `fp`.
fn bounds(fp: Fp) -> i64 {
    BOUNDS.0.as_ptr() as i64 + if fp == Fp::D { 80 } else { 0 }
}

const fn field(word: u32, low: u32, bits: u32) -> u8 {
    ((word >> low) & ((1 << bits) - 1)) as u8
}

/// Emits the vector FP instruction `word` as a checked op and returns true, or emits nothing and
/// returns false.
pub(super) fn lower(t: &mut Translator, word: u32) -> bool {
    let fp = if word >> 22 & 1 != 0 { Fp::D } else { Fp::S };
    let vector_lanes = match (fp, word >> 30 & 1 != 0) {
        (Fp::S, true) => 4,
        (Fp::S, false) | (Fp::D, true) => 2,
        (Fp::D, false) => 0,
    };
    let three_same = match word & 0xbfa0_fc00 {
        0x0e20_d400 => Some(LaneOp::Add),
        0x2e20_d400 => Some(LaneOp::AddPairwise),
        0x0ea0_d400 => Some(LaneOp::Sub),
        0x2e20_dc00 => Some(LaneOp::Mul),
        0x0e20_cc00 => Some(LaneOp::Mla),
        0x0ea0_cc00 => Some(LaneOp::Mls),
        0x0e20_fc00 => Some(LaneOp::RecipStep),
        0x0ea0_fc00 => Some(LaneOp::RsqrtStep),
        0x0e20_e400 => Some(LaneOp::CompareEq),
        0x2e20_e400 => Some(LaneOp::CompareGe),
        0x2ea0_e400 => Some(LaneOp::CompareGt),
        _ => None,
    };
    let by_element = |masked: u32| match masked {
        0x0f80_1000 => Some(LaneOp::Mla),
        0x0f80_5000 => Some(LaneOp::Mls),
        0x0f80_9000 => Some(LaneOp::Mul),
        _ => None,
    };
    let unary = |masked: u32| match masked {
        0x0e21_d800 => Some(LaneOp::FromInt),
        0x0ea1_d800 => Some(LaneOp::RecipEstimate),
        0x2ea1_d800 => Some(LaneOp::RsqrtEstimate),
        _ => None,
    };
    let (op, element, lanes) = if let Some(op) = unary(word & 0xbfff_fc00) {
        (op, false, vector_lanes)
    } else if let Some(op) =
        unary(word & 0xffff_fc00 ^ 0x5000_0000).filter(|&op| op != LaneOp::FromInt)
    {
        (op, false, 1)
    } else if let Some(op) = three_same {
        (op, false, vector_lanes)
    } else if let Some(op) = by_element(word & 0xbf80_f400) {
        (op, true, vector_lanes)
    } else if let Some(op) = by_element(word & 0xff80_f400 ^ 0x5000_0000) {
        (op, true, 1)
    } else {
        return false;
    };
    if lanes == 0 {
        return false;
    }
    let index = if element {
        let (h, l) = (field(word, 11, 1), field(word, 21, 1));
        match fp {
            Fp::S => Some(h << 1 | l),
            Fp::D if l == 0 => Some(h),
            Fp::D => return false,
        }
    } else {
        None
    };
    let (rd, rn, rm) = (field(word, 0, 5), field(word, 5, 5), field(word, 16, 5));
    let vec = |t: &mut Translator, r: u8| t.op(Op::Get(Loc::V(r)));
    let n = vec(t, rn);
    let m = if matches!(
        op,
        LaneOp::FromInt | LaneOp::RecipEstimate | LaneOp::RsqrtEstimate
    ) {
        n
    } else {
        vec(t, rm)
    };
    let acc = matches!(op, LaneOp::Mla | LaneOp::Mls).then(|| vec(t, rd));
    let r = t.op(Op::Simd(SimdOp::Lanes(Lanes {
        op,
        fp,
        lanes,
        n,
        m,
        index,
        acc,
        word,
    })));
    t.set(Loc::V(rd), r);
    true
}

impl Emitter<'_> {
    pub(super) fn lanes(&mut self, i: usize, lanes: Lanes) {
        let Lanes {
            op,
            fp,
            lanes,
            n,
            m,
            index,
            acc,
            word,
        } = lanes;
        let (n, m) = (self.reg(n), self.reg(m));
        let acc = acc.map(|a| self.reg(a));
        let d = self.def(Value(i as u32));
        let compare = matches!(
            op,
            LaneOp::CompareEq | LaneOp::CompareGe | LaneOp::CompareGt
        );
        let estimate = matches!(op, LaneOp::RecipEstimate | LaneOp::RsqrtEstimate);
        let written = (1i32 << lanes) - 1;
        let element = index.map_or(written, |index| 1 << index);
        let inputs = [(n, written), (m, element), (acc.unwrap_or(n), written)];
        let inputs = Inputs::lanes(
            fp,
            &inputs[..if acc.is_some() { 3 } else { 2 }],
            matches!(op, LaneOp::Add | LaneOp::Sub | LaneOp::AddPairwise),
        );
        let slow = match op {
            _ if estimate => self.asm.new_dynamic_label(),
            LaneOp::FromInt => self.checked(true, None),
            _ => self.checked(!compare, Some(inputs)),
        };
        let fused = matches!(
            op,
            LaneOp::Mla | LaneOp::Mls | LaneOp::RecipStep | LaneOp::RsqrtStep
        );
        if fused && !self.target.fma {
            dynasm!(self.asm ; .arch x64 ; jmp =>slow);
            self.finish_checked(i, slow, word, d, Ty::U128);
            return;
        }
        let b = match (index, fp) {
            (None, _) => m,
            (Some(index), Fp::S) => {
                let lane = (index * 0x55) as i8;
                dynasm!(self.asm ; .arch x64 ; pshufd xmm15, Rx(m), lane);
                SCRATCH
            }
            (Some(index), Fp::D) => {
                let lane = if index == 0 { 0x44 } else { 0xeeu8 as i8 };
                dynasm!(self.asm ; .arch x64 ; pshufd xmm15, Rx(m), lane);
                SCRATCH
            }
        };
        if compare {
            let written = (1i32 << lanes) - 1;
            dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(n));
            match fp {
                Fp::S => {
                    dynasm!(self.asm ; .arch x64 ; cmpps xmm15, Rx(m), 3 ; movmskps eax, xmm15)
                }
                Fp::D => {
                    dynasm!(self.asm ; .arch x64 ; cmppd xmm15, Rx(m), 3 ; movmskpd eax, xmm15)
                }
            }
            dynasm!(self.asm ; .arch x64 ; test eax, written ; jnz =>slow);
        }
        match op {
            LaneOp::FromInt | LaneOp::RecipEstimate | LaneOp::RsqrtEstimate => {}
            LaneOp::CompareGe | LaneOp::CompareGt => {
                dynasm!(self.asm ; .arch x64 ; movaps Rx(d), Rx(m))
            }
            LaneOp::RecipStep | LaneOp::RsqrtStep => {
                let constants = STEPS.0.as_ptr() as i64 + if fp == Fp::D { 48 } else { 0 };
                let offset = if op == LaneOp::RsqrtStep { 16 } else { 0 };
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD constants
                    ; movaps Rx(d), [rax + offset]
                );
            }
            _ => dynasm!(self.asm ; .arch x64 ; movaps Rx(d), Rx(acc.unwrap_or(n))),
        }
        match (op, fp) {
            (LaneOp::FromInt, _) => dynasm!(self.asm ; .arch x64 ; cvtdq2ps Rx(d), Rx(n)),
            (LaneOp::RecipEstimate | LaneOp::RsqrtEstimate, _) => {
                self.estimate(op == LaneOp::RsqrtEstimate, lanes, d, n, slow)
            }
            (LaneOp::CompareEq, Fp::S) => dynasm!(self.asm ; .arch x64 ; cmpps Rx(d), Rx(m), 0),
            (LaneOp::CompareEq, Fp::D) => dynasm!(self.asm ; .arch x64 ; cmppd Rx(d), Rx(m), 0),
            (LaneOp::CompareGe, Fp::S) => dynasm!(self.asm ; .arch x64 ; cmpps Rx(d), Rx(n), 2),
            (LaneOp::CompareGe, Fp::D) => dynasm!(self.asm ; .arch x64 ; cmppd Rx(d), Rx(n), 2),
            (LaneOp::CompareGt, Fp::S) => dynasm!(self.asm ; .arch x64 ; cmpps Rx(d), Rx(n), 1),
            (LaneOp::CompareGt, Fp::D) => dynasm!(self.asm ; .arch x64 ; cmppd Rx(d), Rx(n), 1),
            (LaneOp::RecipStep, Fp::S) => {
                dynasm!(self.asm ; .arch x64 ; vfnmadd231ps Rx(d), Rx(n), Rx(m))
            }
            (LaneOp::RsqrtStep, Fp::S) => dynasm!(self.asm
                ; .arch x64
                ; vfnmadd231ps Rx(d), Rx(n), Rx(m)
                ; mulps Rx(d), [rax + 32]
            ),
            (LaneOp::RecipStep, Fp::D) => self.fma231pd(0xbc, d, n, m),
            (LaneOp::RsqrtStep, Fp::D) => {
                self.fma231pd(0xbc, d, n, m);
                dynasm!(self.asm ; .arch x64 ; mulpd Rx(d), [rax + 32]);
            }
            (LaneOp::AddPairwise, Fp::S) if lanes == 4 => dynasm!(self.asm
                ; .arch x64
                ; shufps Rx(d), Rx(m), 0x88u8 as i8
                ; movaps xmm15, Rx(n)
                ; shufps xmm15, Rx(m), 0xddu8 as i8
                ; addps Rx(d), xmm15
            ),
            (LaneOp::AddPairwise, Fp::S) => dynasm!(self.asm
                ; .arch x64
                ; unpcklps Rx(d), Rx(m)
                ; movhlps xmm15, Rx(d)
                ; addps Rx(d), xmm15
            ),
            (LaneOp::AddPairwise, Fp::D) => dynasm!(self.asm
                ; .arch x64
                ; unpcklpd Rx(d), Rx(m)
                ; movaps xmm15, Rx(n)
                ; unpckhpd xmm15, Rx(m)
                ; addpd Rx(d), xmm15
            ),
            (LaneOp::Add, Fp::S) => dynasm!(self.asm ; .arch x64 ; addps Rx(d), Rx(b)),
            (LaneOp::Add, Fp::D) => dynasm!(self.asm ; .arch x64 ; addpd Rx(d), Rx(b)),
            (LaneOp::Sub, Fp::S) => dynasm!(self.asm ; .arch x64 ; subps Rx(d), Rx(b)),
            (LaneOp::Sub, Fp::D) => dynasm!(self.asm ; .arch x64 ; subpd Rx(d), Rx(b)),
            (LaneOp::Mul, Fp::S) => dynasm!(self.asm ; .arch x64 ; mulps Rx(d), Rx(b)),
            (LaneOp::Mul, Fp::D) => dynasm!(self.asm ; .arch x64 ; mulpd Rx(d), Rx(b)),
            (LaneOp::Mla, Fp::S) => dynasm!(self.asm ; .arch x64 ; vfmadd231ps Rx(d), Rx(n), Rx(b)),
            (LaneOp::Mla, Fp::D) => self.fma231pd(0xb8, d, n, b),
            (LaneOp::Mls, Fp::S) => {
                dynasm!(self.asm ; .arch x64 ; vfnmadd231ps Rx(d), Rx(n), Rx(b))
            }
            (LaneOp::Mls, Fp::D) => self.fma231pd(0xbc, d, n, b),
        }
        if op != LaneOp::FromInt && !compare && !estimate {
            self.check_lanes(op, fp, lanes, d, n, m, index, slow);
        }
        match (fp, lanes) {
            (Fp::S, 4) | (Fp::D, 2) => {}
            (Fp::S, 1) => dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d) ; movd Rx(d), eax),
            _ => dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(d)),
        }
        self.finish_checked(i, slow, word, d, Ty::U128);
    }

    /// `FRSQRTE` (`sqrt`) or `FRECPE` of the S lanes of `xmm n` into `xmm d`, as the Arm ARM computes
    /// them: an 8-bit fraction from a table and an exponent from the input's. Jumps to `slow` unless
    /// every written lane is a positive normal number (`FRSQRTE`) or a normal number with a normal
    /// reciprocal (`FRECPE`). The estimate of those raises no flag under any FPCR, so the op needs
    /// no FPCR or IXC check. Clobbers `xmm15`, `rax`, `rcx` and `rdx`.
    fn estimate(&mut self, sqrt: bool, lanes: u8, d: u8, n: u8, slow: dynasmrt::DynamicLabel) {
        let constants = ESTIMATE.0.as_ptr() as i64;
        let written = (1i32 << lanes) - 1;
        let (table, index_mask, bound) = if sqrt {
            (RSQRT_FRACTIONS.as_ptr() as i64, 0x1ff, RSQRT_BOUND)
        } else {
            (RECIP_FRACTIONS.as_ptr() as i64, 0xff, RECIP_BOUND)
        };
        dynasm!(self.asm
            ; .arch x64
            ; mov rax, QWORD constants
            ; movaps xmm15, Rx(n)
        );
        if !sqrt {
            dynasm!(self.asm ; .arch x64 ; andps xmm15, [rax + ABS]);
        }
        dynasm!(self.asm
            ; .arch x64
            ; psubd xmm15, [rax + MIN_NORMAL]
            ; pxor xmm15, [rax + SIGN]
            ; movaps Rx(d), [rax + bound]
            ; pcmpgtd Rx(d), xmm15
            ; movmskps ecx, Rx(d)
            ; and ecx, written
            ; cmp ecx, written
            ; jne =>slow
            ; mov rcx, QWORD table
        );
        for k in 0..lanes {
            if k == 0 {
                dynasm!(self.asm ; .arch x64 ; movd eax, Rx(n));
            } else {
                dynasm!(self.asm ; .arch x64 ; pshufd xmm15, Rx(n), (k * 0x55) as i8 ; movd eax, xmm15);
            }
            dynasm!(self.asm
                ; .arch x64
                ; shr eax, 15
                ; and eax, index_mask
                ; movzx eax, BYTE [rcx + rax]
            );
            if k & 1 == 0 {
                dynasm!(self.asm ; .arch x64 ; mov edx, eax);
            } else {
                dynasm!(self.asm ; .arch x64 ; shl rax, 32 ; or rdx, rax);
            }
            if k & 1 == 1 || k + 1 == lanes {
                if k < 2 {
                    dynasm!(self.asm ; .arch x64 ; movq Rx(d), rdx);
                } else {
                    dynasm!(self.asm ; .arch x64 ; movq xmm15, rdx ; punpcklqdq Rx(d), xmm15);
                }
            }
        }
        dynasm!(self.asm
            ; .arch x64
            ; pslld Rx(d), 15
            ; mov rax, QWORD constants
            ; movaps xmm15, Rx(n)
            ; psrld xmm15, 23
        );
        if sqrt {
            dynasm!(self.asm
                ; .arch x64
                ; psubd xmm15, [rax + RSQRT_EXP]
                ; pxor xmm15, [rax + ONES]
                ; psrld xmm15, 1
                ; pslld xmm15, 23
                ; por Rx(d), xmm15
            );
        } else {
            dynasm!(self.asm
                ; .arch x64
                ; pand xmm15, [rax + EXP]
                ; psubd xmm15, [rax + RECIP_EXP]
                ; pxor xmm15, [rax + ONES]
                ; pslld xmm15, 23
                ; por Rx(d), xmm15
                ; movaps xmm15, Rx(n)
                ; pand xmm15, [rax + SIGN]
                ; por Rx(d), xmm15
            );
        }
    }

    /// Jumps to `slow` if a lane of an input `xmm` that its mask selects is below the smallest
    /// normal, or for a `sum` below its floor (see [`Inputs`]), and not a zero. Clobbers `xmm15`,
    /// `rax` and `rcx`.
    pub(super) fn flushed_lanes(
        &mut self,
        fp: Fp,
        inputs: &[(u8, i32)],
        sum: bool,
        slow: dynasmrt::DynamicLabel,
    ) {
        const LT: i8 = 1;
        const NEQ: i8 = 4;
        const RCX: u8 = 1;
        dynasm!(self.asm ; .arch x64 ; mov rax, QWORD bounds(fp));
        for &(src, mask) in inputs {
            dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(src) ; andps xmm15, [rax]);
            self.compare_bound(fp, if sum { 64 } else { 32 }, LT);
            dynasm!(self.asm ; .arch x64 ; andps xmm15, Rx(src));
            self.compare_bound(fp, 48, NEQ);
            self.lane_mask(fp, RCX);
            dynasm!(self.asm ; .arch x64 ; test ecx, mask ; jnz =>slow);
        }
    }

    /// Jumps to `slow` unless every written lane of the result in `xmm d` is finite and, for the
    /// ops that can underflow, above the smallest normal or a zero whose product had a zero
    /// factor. Clobbers `xmm15`, `rax`, `rcx` and `rdx`.
    #[allow(clippy::too_many_arguments)]
    fn check_lanes(
        &mut self,
        op: LaneOp,
        fp: Fp,
        lanes: u8,
        d: u8,
        n: u8,
        m: u8,
        index: Option<u8>,
        slow: dynasmrt::DynamicLabel,
    ) {
        let written = (1i32 << lanes) - 1;
        const LT: i8 = 1;
        const NLE: i8 = 6;
        const RAX: u8 = 0;
        const RCX: u8 = 1;
        const RDX: u8 = 2;
        dynasm!(self.asm
            ; .arch x64
            ; mov rax, QWORD bounds(fp)
            ; movaps xmm15, Rx(d)
            ; andps xmm15, [rax]
        );
        self.compare_bound(fp, 16, LT);
        self.lane_mask(fp, RCX);
        if matches!(op, LaneOp::Mul | LaneOp::Mla | LaneOp::Mls) {
            dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(d) ; andps xmm15, [rax]);
            self.compare_bound(fp, 32, NLE);
            self.lane_mask(fp, RDX);
            dynasm!(self.asm ; .arch x64 ; and ecx, edx);
            self.zero_lanes(fp, n, RAX);
            self.zero_lanes(fp, m, RDX);
            if let Some(index) = index {
                dynasm!(self.asm
                    ; .arch x64
                    ; shr edx, index as i8
                    ; and edx, 1
                    ; neg edx
                );
            }
            dynasm!(self.asm ; .arch x64 ; or eax, edx);
            self.zero_lanes(fp, d, RDX);
            dynasm!(self.asm ; .arch x64 ; and eax, edx ; or ecx, eax);
        }
        dynasm!(self.asm
            ; .arch x64
            ; and ecx, written
            ; cmp ecx, written
            ; jne =>slow
        );
    }

    /// `VFMADD231PD` (`0xb8`) or `VFNMADD231PD` (`0xbc`) `xmm d, xmm n, xmm b`, encoded by hand
    /// because dynasm-rs encodes the PD forms with VEX.W clear, which is the PS instruction.
    fn fma231pd(&mut self, opcode: u8, d: u8, n: u8, b: u8) {
        let r = (!d >> 3 & 1) << 7;
        let rb = (!b >> 3 & 1) << 5;
        self.asm.extend([
            0xc4,
            r | 1 << 6 | rb | 0x02,
            0x80 | (!n & 0xf) << 3 | 0x01,
            opcode,
            0xc0 | (d & 7) << 3 | (b & 7),
        ]);
    }

    /// Compares `xmm15` with the bound at `[rax + offset]` lane by lane.
    fn compare_bound(&mut self, fp: Fp, offset: i32, predicate: i8) {
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; cmpps xmm15, [rax + offset], predicate),
            Fp::D => dynasm!(self.asm ; .arch x64 ; cmppd xmm15, [rax + offset], predicate),
        }
    }

    /// The sign bits of `xmm15`'s lanes in the low bits of `gpr`.
    fn lane_mask(&mut self, fp: Fp, gpr: u8) {
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; movmskps Rd(gpr), xmm15),
            Fp::D => dynasm!(self.asm ; .arch x64 ; movmskpd Rd(gpr), xmm15),
        }
    }

    /// A bit in `gpr` for each lane of `xmm src` that is a zero of either sign.
    fn zero_lanes(&mut self, fp: Fp, src: u8, gpr: u8) {
        dynasm!(self.asm ; .arch x64 ; xorps xmm15, xmm15);
        match fp {
            Fp::S => dynasm!(self.asm ; .arch x64 ; cmpps xmm15, Rx(src), 0),
            Fp::D => dynasm!(self.asm ; .arch x64 ; cmppd xmm15, Rx(src), 0),
        }
        self.lane_mask(fp, gpr);
    }
}
