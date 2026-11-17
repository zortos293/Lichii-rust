//! Advanced SIMD integer arithmetic, element moves and `LD1`/`ST1` (multiple structures), compiled
//! inline with SSE2, and SSE4.1/4.2 where the host has them. Each lowering gives the interpreter's
//! result bit for bit; encodings it does not take stay on `Op::CallSimd` or the interpreter.

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use dynasmrt::{DynasmApi, DynasmLabelApi, dynasm};

use super::super::ir::{Loc, Op, Ty, Value, W};
use super::super::translate::Translator;
use super::super::x64::{CTX_FPSR, Emitter, ST_SCRATCH};
use super::ir::{Fp, SimdOp};
use crate::decoder::{Inst, PostIndex, Reg};

const SCRATCH: u8 = 15;
/// FPSR.QC.
const QC: i32 = 1 << 27;

/// Byte masks for `CNT`: `0x55`, `0x33` and `0x0f` in every byte.
#[repr(align(16))]
struct Masks([u8; 48]);

static CNT_MASKS: Masks = {
    let mut m = [0; 48];
    let mut i = 0;
    while i < 16 {
        m[i] = 0x55;
        m[16 + i] = 0x33;
        m[32 + i] = 0x0f;
        i += 1;
    }
    Masks(m)
};

/// `0x70` in every byte: added with unsigned saturation to a `TBL` index, it keeps indices below
/// 16 below `0x80` with the same low nibble and sends every other one to `0x80` or above, which
/// `PSHUFB` turns into a zero byte.
#[repr(align(16))]
struct TableBias([u8; 16]);

static TABLE_BIAS: TableBias = TableBias([0x70; 16]);

/// For `USHL` on S lanes and on D lanes: a mask of each lane's low byte, then 1 in each lane.
#[repr(align(16))]
struct ShiftMasks([u32; 8], [u64; 4]);

static SHIFT_MASKS: ShiftMasks =
    ShiftMasks([0xff, 0xff, 0xff, 0xff, 1, 1, 1, 1], [0xff, 0xff, 1, 1]);

/// Lane-wise operations on two vectors. Comparisons give all ones in a lane where they hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Sub,
    And,
    /// `a & !b`.
    Bic,
    Orr,
    /// `a | !b`.
    Orn,
    Eor,
    CmEq,
    /// Signed `a > b`.
    CmGt,
    /// Signed `a >= b`.
    CmGe,
    /// Unsigned `a > b`.
    CmHi,
    /// Unsigned `a >= b`.
    CmHs,
    Mul,
    /// Sums of adjacent lane pairs, those of `a` in the low half and those of `b` in the high
    /// half.
    Addp,
    SMax,
    SMin,
    UMax,
    UMin,
    /// All ones in a lane where `a & b` is nonzero.
    CmTst,
    /// `(a + b) >> 1` without overflow, signed or unsigned.
    Hadd(bool),
    /// `(a + b + 1) >> 1` without overflow, signed or unsigned.
    Rhadd(bool),
    /// `|a - b|`, signed or unsigned.
    Abd(bool),
    /// `SMAXP`, `SMINP`, `UMAXP` or `UMINP`: the larger (or, unless `max`, the smaller) of
    /// each adjacent lane pair, signed or unsigned, those of `a` in the low half and those of `b`
    /// in the high half.
    MinMaxP {
        max: bool,
        signed: bool,
    },
}

/// The bitwise selects, each `base ^ ((n ^ base) & mask)` on the destination's old value `old`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Select {
    /// `BSL`: `m` where `old` is clear, `n` where it is set.
    Bsl,
    /// `BIT`: `n` where `m` is set, else `old`.
    Bit,
    /// `BIF`: `n` where `m` is clear, else `old`.
    Bif,
}

/// The arithmetic of `VecOp::Long`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Long {
    /// `SADDL`: `n + m`.
    Add,
    /// `SSUBL`: `n - m`.
    Sub,
    /// `SADDW`: `n`, already wide, plus `m`.
    AddWide,
    /// `SSUBW`: `n`, already wide, minus `m`.
    SubWide,
    /// `SABDL`: `|n - m|`.
    Abd,
    /// `SMULL`: `n * m`.
    Mul,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shift {
    Shl,
    Ushr,
    Sshr,
}

/// Lanes of `8 << size` bits filling 128 bits, or only the low 64 with the high half zeroed when
/// `q` is clear.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arrangement {
    pub size: u8,
    pub q: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VecOp {
    Binary(Binary, Arrangement, Value, Value),
    /// Every lane shifted by an immediate: up to the lane width for right shifts, below it for
    /// `Shl`.
    Shift(Shift, Arrangement, u8, Value),
    /// The 64-bit constant in the low half, and in the high half too when `q` is set.
    Imm(u64, bool),
    /// The number of set bits in each byte, over 128 bits when `q` is set and else the low 64.
    Cnt(bool, Value),
    /// The low lane-sized bits of a general-purpose value in every lane.
    Dup(Arrangement, Value),
    /// `vector` with lane `index` of `8 << size` bits replaced by the low bits of `value`.
    Insert {
        size: u8,
        index: u8,
        vector: Value,
        value: Value,
    },
    /// Lane `index` as a general-purpose value: zero-extended, or sign-extended to `signed`.
    Extract {
        size: u8,
        index: u8,
        signed: Option<W>,
        vector: Value,
    },
    /// Lane `index` of `8 << size` bits (H or S) of `vector` in every lane.
    DupLane {
        size: u8,
        index: u8,
        vector: Value,
    },
    /// Scalar `DUP` (`MOV` from an element): lane `index` of `8 << size` bits, with every other
    /// bit zero.
    ScalarDup {
        size: u8,
        index: u8,
        vector: Value,
    },
    /// `EXT`: bytes `index` onwards of `m:n` (with `n` in the low bytes), 16 of them when `q` is
    /// set and else 8 from the low halves.
    Ext {
        q: bool,
        index: u8,
        n: Value,
        m: Value,
    },
    /// `ZIP1` (`high` clear) or `ZIP2`: the lanes of the low or high halves of `n` and `m`,
    /// interleaved with `n`'s first.
    Zip {
        high: bool,
        arrangement: Arrangement,
        n: Value,
        m: Value,
    },
    /// `BSL`, `BIT` or `BIF`, over 128 bits when `q` is set and else the low 64.
    Select {
        op: Select,
        q: bool,
        old: Value,
        n: Value,
        m: Value,
    },
    /// An integer three-different op (the `2` form when `high`) on the low (or high) half of the
    /// `8 << size`-bit lanes of `m`, and of `n` too unless the op is a wide one, sign- or
    /// zero-extended to `16 << size` bits. With `acc`, the result is added to the accumulator, or
    /// subtracted from it when its flag is set (`SABAL`, `SMLAL`, `SMLSL` and the unsigned forms).
    Long {
        op: Long,
        signed: bool,
        size: u8,
        high: bool,
        n: Value,
        m: Value,
        acc: Option<(Value, bool)>,
    },
    /// `SSHLL`/`USHLL` (and the `2` forms when `high`): the low (or high) half of `a`'s
    /// `8 << size`-bit lanes, sign- or zero-extended and shifted left by `shift`, up to `8 << size`
    /// (`SHLL`).
    ShiftLong {
        signed: bool,
        size: u8,
        high: bool,
        shift: u8,
        a: Value,
    },
    /// `SRSHR`/`URSHR` by `shift` (1 to the lane width): each lane of `a` shifted right,
    /// arithmetically when `signed`, rounding to nearest with halves up; `SRSRA`/`URSRA` when
    /// `acc` is given, which adds the result to it.
    RoundShift {
        signed: bool,
        arrangement: Arrangement,
        shift: u8,
        acc: Option<Value>,
        a: Value,
    },
    /// `MLA` (`acc + n * m`) or `MLS` (`acc - n * m`, when `sub`) on H or S lanes.
    MulAcc {
        sub: bool,
        arrangement: Arrangement,
        acc: Value,
        n: Value,
        m: Value,
    },
    /// `USHL` on S or D lanes: each lane of `n` shifted by the signed low byte of `m`'s lane, left
    /// when it is positive and logically right when negative, giving zero from the lane width on.
    Ushl(Arrangement, Value, Value),
    /// `ADDV`: the sum of every lane, modulo the lane width, in the low lane and zero above.
    AddAcross(Arrangement, Value),
    /// `XTN`: the low halves of `a`'s lanes of `16 << size` bits, in the low 64 bits with zero
    /// above, or for `XTN2` in the high 64 bits above `low`'s low 64.
    Narrow {
        size: u8,
        a: Value,
        low: Option<Value>,
    },
    /// `UZP1`/`UZP2` (`transpose` clear) or `TRN1`/`TRN2` on S or D lanes: the even lanes of
    /// `n` and `m`, or the odd ones when `odd`, concatenated or interleaved.
    Permute {
        transpose: bool,
        odd: bool,
        arrangement: Arrangement,
        n: Value,
        m: Value,
    },
    /// `SMAXV`, `SMINV`, `UMAXV` or `UMINV` (`op` the matching `Binary`) on H or S lanes: the
    /// extreme lane in the low lane and zero above.
    MinMaxAcross(Binary, Arrangement, Value),
    /// `REV64` on H or S lanes: the lanes of each 64-bit half in reverse order.
    Rev64(Arrangement, Value),
    /// `TBL` with one table register: byte `i` is byte `index[i]` of `table`, or zero from 16 on,
    /// over 16 bytes when `q` is set and else the low 8.
    Table {
        q: bool,
        table: Value,
        index: Value,
    },
}

impl VecOp {
    pub fn ty(&self) -> Ty {
        match self {
            VecOp::Extract { .. } => Ty::U64,
            _ => Ty::U128,
        }
    }

    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            VecOp::Binary(_, _, a, b)
            | VecOp::Insert {
                vector: a,
                value: b,
                ..
            }
            | VecOp::Ext { n: a, m: b, .. }
            | VecOp::Zip { n: a, m: b, .. }
            | VecOp::Ushl(_, a, b)
            | VecOp::Permute { n: a, m: b, .. }
            | VecOp::Table {
                table: a, index: b, ..
            } => {
                f(a);
                f(b);
            }
            VecOp::Select { old, n, m, .. } | VecOp::MulAcc { acc: old, n, m, .. } => {
                f(old);
                f(n);
                f(m);
            }
            VecOp::Shift(_, _, _, a)
            | VecOp::ScalarDup { vector: a, .. }
            | VecOp::DupLane { vector: a, .. }
            | VecOp::Dup(_, a)
            | VecOp::Cnt(_, a)
            | VecOp::Extract { vector: a, .. }
            | VecOp::ShiftLong { a, .. }
            | VecOp::AddAcross(_, a)
            | VecOp::MinMaxAcross(_, _, a)
            | VecOp::Rev64(_, a) => f(a),
            VecOp::Narrow { a, low, .. } => {
                f(a);
                if let Some(low) = low {
                    f(low);
                }
            }
            VecOp::Long { n, m, acc, .. } => {
                if let Some((acc, _)) = acc {
                    f(acc);
                }
                f(n);
                f(m);
            }
            VecOp::RoundShift { acc, a, .. } => {
                if let Some(acc) = acc {
                    f(acc);
                }
                f(a);
            }
            VecOp::Imm(..) => {}
        }
    }
}

const fn field(word: u32, low: u32, bits: u32) -> u8 {
    ((word >> low) & ((1 << bits) - 1)) as u8
}

/// `AdvSIMDExpandImm` for the integer `cmode`s (all but `0b1111`).
fn expand_imm(op: bool, cmode: u8, imm8: u64) -> u64 {
    let words = |x: u64| x << 32 | x;
    match cmode >> 1 {
        0..=3 => words(imm8 << (8 * (cmode >> 1))),
        4 | 5 => (imm8 << (8 * (cmode >> 1 & 1))) * 0x0001_0001_0001_0001,
        6 if cmode & 1 == 0 => words(imm8 << 8 | 0xff),
        6 => words(imm8 << 16 | 0xffff),
        _ if op => (0..8).fold(0, |r, i| r | (((imm8 >> i) & 1) * 0xff) << (8 * i)),
        _ => imm8 * 0x0101_0101_0101_0101,
    }
}

/// Whether the host has the instructions `op` needs on lanes of `8 << size` bits.
fn supported(t: &Translator, op: Binary, size: u8) -> bool {
    match op {
        Binary::Add
        | Binary::Sub
        | Binary::And
        | Binary::Bic
        | Binary::Orr
        | Binary::Orn
        | Binary::Eor
        | Binary::Addp => true,
        Binary::CmEq => size < 3 || t.sse41,
        Binary::CmGt | Binary::CmGe => size < 3 || t.sse42,
        Binary::CmHi | Binary::CmHs => size == 0 || (size < 3 && t.sse41),
        Binary::Mul => size == 1 || (size == 2 && t.sse41),
        Binary::SMax | Binary::SMin => size == 1 || (size < 3 && t.sse41),
        Binary::UMax | Binary::UMin => size == 0 || (size < 3 && t.sse41),
        Binary::CmTst => size < 3 || t.sse41,
        Binary::Hadd(_) | Binary::Rhadd(true) => size == 1 || size == 2,
        Binary::Rhadd(false) => size < 3,
        Binary::Abd(signed) | Binary::MinMaxP { signed, .. } => {
            supported(t, if signed { Binary::SMax } else { Binary::UMax }, size)
        }
    }
}

/// Emits the vector instruction `word` as native ops and returns true, or emits nothing and
/// returns false.
pub(super) fn lower(t: &mut Translator, word: u32) -> bool {
    let (rd, rn, rm) = (field(word, 0, 5), field(word, 5, 5), field(word, 16, 5));
    let (q, u) = (word >> 30 & 1 != 0, word >> 29 & 1 != 0);
    let size = field(word, 22, 2);
    let arrangement = Arrangement { size, q };
    let vec = |t: &mut Translator, r: u8| t.op(Op::Get(Loc::V(r)));
    let emit = |t: &mut Translator, op: VecOp| t.op(Op::Simd(SimdOp::Vec(op)));
    if word & 0x9f00_0400 == 0x0f00_0000 && by_element(t, word) {
        return true;
    }
    if word & 0x9f20_fc00 == 0x0e20_b400 && size == 1 && t.sse41 {
        let (a, b) = (vec(t, rn), vec(t, rm));
        let r = t.op(Op::Simd(SimdOp::SatMulHigh {
            round: u,
            arrangement,
            a,
            b,
            word,
        }));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbfe0_fc00 == 0x0e00_0000 && t.sse41 {
        let (table, index) = (vec(t, rn), vec(t, rm));
        let r = emit(t, VecOp::Table { q, table, index });
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf20_fc00 == 0x2e20_4400 && t.avx2 && (size == 2 || (size == 3 && q)) {
        let (n, m) = (vec(t, rn), vec(t, rm));
        let r = emit(t, VecOp::Ushl(arrangement, n, m));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_fc00 == 0x0e20_9400 && supported(t, Binary::Mul, size) {
        let (acc, n, m) = (vec(t, rd), vec(t, rn), vec(t, rm));
        let r = emit(
            t,
            VecOp::MulAcc {
                sub: u,
                arrangement,
                acc,
                n,
                m,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x0e31_b800 && (size < 2 || (size == 2 && q)) {
        let a = vec(t, rn);
        let r = emit(t, VecOp::AddAcross(arrangement, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x0e21_2800 && size < 3 {
        let a = vec(t, rn);
        let low = q.then(|| vec(t, rd));
        let r = emit(t, VecOp::Narrow { size, a, low });
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf20_8c00 == 0x0e00_0800
        && matches!(field(word, 12, 3), 1 | 2 | 5 | 6)
        && (size == 2 || (size == 3 && q))
    {
        let (n, m) = (vec(t, rn), vec(t, rm));
        let r = emit(
            t,
            VecOp::Permute {
                transpose: word & 0x2000 != 0,
                odd: word & 0x4000 != 0,
                arrangement,
                n,
                m,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f3e_fc00 == 0x0e30_a800 && (size == 1 || (size == 2 && q)) {
        let op = match (u, word & 0x1_0000 != 0) {
            (false, false) => Binary::SMax,
            (false, true) => Binary::SMin,
            (true, false) => Binary::UMax,
            (true, true) => Binary::UMin,
        };
        if supported(t, op, size) {
            let a = vec(t, rn);
            let r = emit(t, VecOp::MinMaxAcross(op, arrangement, a));
            t.set(Loc::V(rd), r);
            return true;
        }
    }
    if word & 0xbf3f_fc00 == 0x2e21_3800 && size < 3 {
        let a = vec(t, rn);
        let r = emit(
            t,
            VecOp::ShiftLong {
                signed: false,
                size,
                high: q,
                shift: 8 << size,
                a,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x2e21_2800 && (size == 0 || (size == 1 && t.sse41)) {
        let n = vec(t, rn);
        let low = q.then(|| vec(t, rd));
        let r = t.op(Op::Simd(SimdOp::SatNarrow { size, n, low, word }));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x0e20_0800 && matches!(size, 1 | 2) {
        let a = vec(t, rn);
        let r = emit(t, VecOp::Rev64(arrangement, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x2e20_b800 && (q || size < 3) {
        let a = vec(t, rn);
        let zero = emit(t, VecOp::Imm(0, true));
        let r = emit(t, VecOp::Binary(Binary::Sub, arrangement, zero, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbfff_fc00 == 0x2e20_5800 {
        let a = vec(t, rn);
        let zero = emit(t, VecOp::Imm(0, true));
        let r = emit(t, VecOp::Binary(Binary::Orn, arrangement, zero, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9ff8_fc00 == 0x0f00_f400 && (!u || q) {
        let imm8 = field(word, 16, 3) << 5 | field(word, 5, 5);
        let imm = match u {
            false => super::translate::expand_imm(Fp::S, imm8) * 0x1_0000_0001,
            true => super::translate::expand_imm(Fp::D, imm8),
        };
        let r = emit(t, VecOp::Imm(imm, q));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_fc00 == 0x0e20_1c00 && u && size != 0 {
        let op = [Select::Bsl, Select::Bit, Select::Bif][size as usize - 1];
        let (old, n, m) = (vec(t, rd), vec(t, rn), vec(t, rm));
        let r = emit(t, VecOp::Select { op, q, old, n, m });
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_dc00 == 0x0e20_0c00 && size < 2 {
        let (a, b) = (vec(t, rn), vec(t, rm));
        let r = t.op(Op::Simd(SimdOp::SatArith {
            arrangement,
            a,
            b,
            word,
        }));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_0400 == 0x0e20_0400 {
        let opcode = field(word, 11, 5);
        let op = match (opcode, u, size) {
            (0b10000, false, _) => Binary::Add,
            (0b10000, true, _) => Binary::Sub,
            (0b00011, false, 0) => Binary::And,
            (0b00011, false, 1) => Binary::Bic,
            (0b00011, false, 2) => Binary::Orr,
            (0b00011, false, _) => Binary::Orn,
            (0b00011, true, 0) => Binary::Eor,
            (0b10001, true, _) => Binary::CmEq,
            (0b00110, false, _) => Binary::CmGt,
            (0b00111, false, _) => Binary::CmGe,
            (0b00110, true, _) => Binary::CmHi,
            (0b00111, true, _) => Binary::CmHs,
            (0b10011, false, _) => Binary::Mul,
            (0b10111, false, _) => Binary::Addp,
            (0b01100, false, _) => Binary::SMax,
            (0b01101, false, _) => Binary::SMin,
            (0b01100, true, _) => Binary::UMax,
            (0b01101, true, _) => Binary::UMin,
            (0b10001, false, _) => Binary::CmTst,
            (0b00000, _, _) => Binary::Hadd(!u),
            (0b00010, _, _) => Binary::Rhadd(!u),
            (0b01110, _, _) => Binary::Abd(!u),
            (0b10100, _, _) => Binary::MinMaxP {
                max: true,
                signed: !u,
            },
            (0b10101, _, _) => Binary::MinMaxP {
                max: false,
                signed: !u,
            },
            _ => return false,
        };
        if opcode != 0b00011 && ((size == 3 && !q) || !supported(t, op, size)) {
            return false;
        }
        let (a, b) = (vec(t, rn), vec(t, rm));
        let r = emit(t, VecOp::Binary(op, arrangement, a, b));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf20_bc00 == 0x0e00_3800 && (q || size < 3) {
        let (n, m) = (vec(t, rn), vec(t, rm));
        let high = word & 0x4000 != 0;
        let r = emit(
            t,
            VecOp::Zip {
                high,
                arrangement,
                n,
                m,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf20_0c00 == 0x0e20_0000
        && matches!(field(word, 12, 4), 0b1001 | 0b1011 | 0b1101)
        && (size == 1 || (size == 2 && t.sse41))
    {
        let acc = match field(word, 12, 4) {
            0b1001 => Some(false),
            0b1011 => Some(true),
            _ => None,
        };
        let acc = acc.map(|sub| (vec(t, rd), sub));
        let (n, m) = (vec(t, rn), vec(t, rm));
        let r = t.op(Op::Simd(SimdOp::SatMulLong {
            size,
            high: q,
            n,
            m,
            acc,
            word,
        }));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_dc00 == 0x0e20_4000 && size < 3 {
        let (n, m) = (vec(t, rn), vec(t, rm));
        let wide = Arrangement {
            size: size + 1,
            q: true,
        };
        let op = if word & 0x2000 != 0 {
            Binary::Sub
        } else {
            Binary::Add
        };
        let mut x = emit(t, VecOp::Binary(op, wide, n, m));
        if u {
            let half = 1u64 << ((8 << size) - 1);
            let lanes = match size {
                0 => half * 0x0001_0001_0001_0001,
                1 => half * 0x0000_0001_0000_0001,
                _ => half,
            };
            let round = emit(t, VecOp::Imm(lanes, true));
            x = emit(t, VecOp::Binary(Binary::Add, wide, x, round));
        }
        let high = emit(t, VecOp::Shift(Shift::Ushr, wide, 8 << size, x));
        let low = q.then(|| vec(t, rd));
        let r = emit(t, VecOp::Narrow { size, a: high, low });
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f20_0c00 == 0x0e20_0000 && size < 3 {
        let signed = !u;
        let (op, acc) = match field(word, 12, 4) {
            0b0000 => (Long::Add, None),
            0b0001 => (Long::AddWide, None),
            0b0010 => (Long::Sub, None),
            0b0011 => (Long::SubWide, None),
            0b0101 => (Long::Abd, Some(false)),
            0b0111 => (Long::Abd, None),
            0b1000 => (Long::Mul, Some(false)),
            0b1010 => (Long::Mul, Some(true)),
            0b1100 => (Long::Mul, None),
            _ => return false,
        };
        let native = match op {
            Long::Abd => supported(t, Binary::Abd(signed), size),
            Long::Mul => size < 2 || !signed || t.sse41,
            _ => true,
        };
        if !native {
            return false;
        }
        let acc = acc.map(|sub| (vec(t, rd), sub));
        let (n, m) = (vec(t, rn), vec(t, rm));
        let r = emit(
            t,
            VecOp::Long {
                op,
                signed,
                size,
                high: q,
                n,
                m,
                acc,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f3f_ec00 == 0x0e20_8800 || word & 0xbf3f_fc00 == 0x0e20_a800 {
        let (op, swap) = match (field(word, 12, 2), u) {
            (0, false) => (Binary::CmGt, false),
            (0, true) => (Binary::CmGe, false),
            (1, false) => (Binary::CmEq, false),
            (1, true) => (Binary::CmGe, true),
            _ => (Binary::CmGt, true),
        };
        if (size == 3 && !q) || !supported(t, op, size) {
            return false;
        }
        let a = vec(t, rn);
        let zero = emit(t, VecOp::Imm(0, true));
        let (a, b) = if swap { (zero, a) } else { (a, zero) };
        let r = emit(t, VecOp::Binary(op, arrangement, a, b));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9fbf_fc00 == 0x0ea0_f800 && (q || size == 2) {
        let sign = if size == 3 {
            1 << 63
        } else {
            0x8000_0000_8000_0000
        };
        let a = vec(t, rn);
        let mask = emit(t, VecOp::Imm(sign, true));
        let op = if u { Binary::Eor } else { Binary::Bic };
        let r = emit(t, VecOp::Binary(op, arrangement, a, mask));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbfff_fc00 == 0x0e20_5800 {
        let a = vec(t, rn);
        let r = emit(t, VecOp::Cnt(q, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9ff8_0400 == 0x0f00_0400 {
        let cmode = field(word, 12, 4);
        if word & 0x800 != 0 || cmode == 0b1111 {
            return false;
        }
        let imm8 = (field(word, 16, 3) as u64) << 5 | field(word, 5, 5) as u64;
        let imm = expand_imm(u, cmode, imm8);
        let r = if cmode < 0b1100 && cmode & 1 == 1 {
            let old = vec(t, rd);
            let imm = emit(t, VecOp::Imm(imm, q));
            let op = if u { Binary::Bic } else { Binary::Orr };
            emit(t, VecOp::Binary(op, arrangement, old, imm))
        } else if u && cmode < 0b1110 {
            emit(t, VecOp::Imm(!imm, q))
        } else {
            emit(t, VecOp::Imm(imm, q))
        };
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9f80_0400 == 0x0f00_0400 {
        let size = 7 - field(word, 19, 4).leading_zeros() as u8;
        let esize = 8u8 << size;
        let immhb = field(word, 16, 7);
        if field(word, 11, 5) == 0b10100 && size < 3 {
            let a = vec(t, rn);
            let r = emit(
                t,
                VecOp::ShiftLong {
                    signed: !u,
                    size,
                    high: q,
                    shift: immhb - esize,
                    a,
                },
            );
            t.set(Loc::V(rd), r);
            return true;
        }
        let opcode = field(word, 11, 5);
        if matches!(opcode, 0b00100 | 0b00110) && size != 0 && (size < 3 || (u && q)) {
            let a = vec(t, rn);
            let acc = (opcode == 0b00110).then(|| vec(t, rd));
            let r = emit(
                t,
                VecOp::RoundShift {
                    signed: !u,
                    arrangement: Arrangement { size, q },
                    shift: 2 * esize - immhb,
                    acc,
                    a,
                },
            );
            t.set(Loc::V(rd), r);
            return true;
        }
        let right = 2 * esize - immhb;
        let shiftable = |op: Shift| size != 0 && (size < 3 || (q && op != Shift::Sshr));
        if opcode == 0b00010 {
            let op = if u { Shift::Ushr } else { Shift::Sshr };
            if !shiftable(op) {
                return false;
            }
            let (acc, a) = (vec(t, rd), vec(t, rn));
            let arrangement = Arrangement { size, q };
            let shifted = emit(t, VecOp::Shift(op, arrangement, right, a));
            let r = emit(t, VecOp::Binary(Binary::Add, arrangement, acc, shifted));
            t.set(Loc::V(rd), r);
            return true;
        }
        if u && matches!(opcode, 0b01000 | 0b01010) {
            let (op, amount) = if opcode == 0b01000 {
                (Shift::Ushr, right)
            } else {
                (Shift::Shl, immhb - esize)
            };
            if !shiftable(op) {
                return false;
            }
            let ones = u64::MAX >> (64 - esize);
            let mask = match op {
                Shift::Shl => ones << amount & ones,
                _ => ones.checked_shr(amount as u32).unwrap_or(0),
            };
            let lanes = match size {
                1 => mask * 0x0001_0001_0001_0001,
                2 => mask * 0x0000_0001_0000_0001,
                _ => mask,
            };
            let arrangement = Arrangement { size, q };
            let (old, a) = (vec(t, rd), vec(t, rn));
            let shifted = emit(t, VecOp::Shift(op, arrangement, amount, a));
            let mask = emit(t, VecOp::Imm(lanes, q));
            let kept = emit(t, VecOp::Binary(Binary::Bic, arrangement, old, mask));
            let r = emit(t, VecOp::Binary(Binary::Orr, arrangement, kept, shifted));
            t.set(Loc::V(rd), r);
            return true;
        }
        if opcode & 0b11100 == 0b10000 && size < 3 {
            let (round, saturate) = (opcode & 1 != 0, opcode & 0b10 != 0 || u);
            let sat_narrow = !u || size == 0 || (size == 1 && t.sse41);
            if (opcode & 0b10 != 0 && u) || (saturate && (size == 2 || !sat_narrow)) {
                return false;
            }
            let wide = Arrangement {
                size: size + 1,
                q: true,
            };
            let a = vec(t, rn);
            let shifted = if round {
                emit(
                    t,
                    VecOp::RoundShift {
                        signed: saturate,
                        arrangement: wide,
                        shift: right,
                        acc: None,
                        a,
                    },
                )
            } else {
                let op = if saturate { Shift::Sshr } else { Shift::Ushr };
                emit(t, VecOp::Shift(op, wide, right, a))
            };
            let low = q.then(|| vec(t, rd));
            let r = if saturate {
                t.op(Op::Simd(SimdOp::SatNarrow {
                    size,
                    n: shifted,
                    low,
                    word,
                }))
            } else {
                emit(
                    t,
                    VecOp::Narrow {
                        size,
                        a: shifted,
                        low,
                    },
                )
            };
            t.set(Loc::V(rd), r);
            return true;
        }
        let (op, amount) = match (opcode, u) {
            (0b00000, false) => (Shift::Sshr, 2 * esize - immhb),
            (0b00000, true) => (Shift::Ushr, 2 * esize - immhb),
            (0b01010, false) => (Shift::Shl, immhb - esize),
            _ => return false,
        };
        if size == 0 || (size == 3 && (!q || op == Shift::Sshr)) {
            return false;
        }
        let a = vec(t, rn);
        let r = emit(t, VecOp::Shift(op, Arrangement { size, q }, amount, a));
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0x9fe0_8400 == 0x0e00_0400 {
        return copy(t, word);
    }
    if word & 0xffe0_fc00 == 0x5e00_0400 {
        let imm5 = field(word, 16, 5);
        let size = imm5.trailing_zeros() as u8;
        if size > 3 {
            return false;
        }
        let vector = vec(t, rn);
        let index = imm5 >> (size + 1);
        let r = emit(
            t,
            VecOp::ScalarDup {
                size,
                index,
                vector,
            },
        );
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbfe0_8400 == 0x2e00_0000 {
        let index = field(word, 11, 4);
        if !q && index >= 8 {
            return false;
        }
        let (n, m) = (vec(t, rn), vec(t, rm));
        let r = emit(t, VecOp::Ext { q, index, n, m });
        t.set(Loc::V(rd), r);
        return true;
    }
    if word & 0xbf3f_fc00 == 0x0e21_4800 && size < 2 {
        let n = vec(t, rn);
        let low = q.then(|| vec(t, rd));
        let r = t.op(Op::Simd(SimdOp::SatNarrow { size, n, low, word }));
        t.set(Loc::V(rd), r);
        return true;
    }
    false
}

/// The integer multiplies by element: `MUL`, `MLA`, `MLS`, the long `SMULL`, `SMLAL` and
/// `SMLSL` with their unsigned forms, and `SQDMULH`/`SQRDMULH` on H lanes.
fn by_element(t: &mut Translator, word: u32) -> bool {
    let (rd, rn) = (field(word, 0, 5), field(word, 5, 5));
    let (q, u, size) = (word >> 30 & 1 != 0, word >> 29 & 1 != 0, field(word, 22, 2));
    let (index, rm) = match size {
        1 => (
            field(word, 11, 1) << 2 | field(word, 20, 2),
            field(word, 16, 4),
        ),
        2 => (
            field(word, 11, 1) << 1 | field(word, 21, 1),
            field(word, 16, 5),
        ),
        _ => return false,
    };
    let arrangement = Arrangement { size, q };
    let opcode = field(word, 12, 4);
    let vec = |t: &mut Translator, r: u8| t.op(Op::Get(Loc::V(r)));
    let emit = |t: &mut Translator, op: VecOp| t.op(Op::Simd(SimdOp::Vec(op)));
    let lane = |t: &mut Translator| {
        let vector = vec(t, rm);
        emit(
            t,
            VecOp::DupLane {
                size,
                index,
                vector,
            },
        )
    };
    let r = match (u, opcode) {
        (false, 0b1000) if supported(t, Binary::Mul, size) => {
            let n = vec(t, rn);
            let m = lane(t);
            emit(t, VecOp::Binary(Binary::Mul, arrangement, n, m))
        }
        (true, 0b0000 | 0b0100) if supported(t, Binary::Mul, size) => {
            let (acc, n) = (vec(t, rd), vec(t, rn));
            let m = lane(t);
            let sub = opcode == 0b0100;
            emit(
                t,
                VecOp::MulAcc {
                    sub,
                    arrangement,
                    acc,
                    n,
                    m,
                },
            )
        }
        (_, 0b0010 | 0b0110 | 0b1010) if size == 1 || u || t.sse41 => {
            let acc = match opcode {
                0b0010 => Some(false),
                0b0110 => Some(true),
                _ => None,
            };
            let acc = acc.map(|sub| (vec(t, rd), sub));
            let n = vec(t, rn);
            let m = lane(t);
            emit(
                t,
                VecOp::Long {
                    op: Long::Mul,
                    signed: !u,
                    size,
                    high: q,
                    n,
                    m,
                    acc,
                },
            )
        }
        (false, 0b0011 | 0b0111 | 0b1011) if size == 1 || t.sse41 => {
            let acc = match opcode {
                0b0011 => Some(false),
                0b0111 => Some(true),
                _ => None,
            };
            let acc = acc.map(|sub| (vec(t, rd), sub));
            let n = vec(t, rn);
            let m = lane(t);
            t.op(Op::Simd(SimdOp::SatMulLong {
                size,
                high: q,
                n,
                m,
                acc,
                word,
            }))
        }
        (false, 0b1100 | 0b1101) if size == 1 && t.sse41 => {
            let a = vec(t, rn);
            let b = lane(t);
            t.op(Op::Simd(SimdOp::SatMulHigh {
                round: opcode == 0b1101,
                arrangement,
                a,
                b,
                word,
            }))
        }
        _ => return false,
    };
    t.set(Loc::V(rd), r);
    true
}

/// DUP, INS, SMOV and UMOV.
fn copy(t: &mut Translator, word: u32) -> bool {
    let (rd, rn) = (field(word, 0, 5), field(word, 5, 5));
    let (q, op) = (word >> 30 & 1 != 0, word >> 29 & 1 != 0);
    let (imm5, imm4) = (field(word, 16, 5), field(word, 11, 4));
    let size = imm5.trailing_zeros() as u8;
    if size > 3 {
        return false;
    }
    let index = imm5 >> (size + 1);
    let insertable = size == 1 || size == 3 || t.sse41;
    let emit = |t: &mut Translator, op: VecOp| t.op(Op::Simd(SimdOp::Vec(op)));
    let extract = |t: &mut Translator, index: u8, signed: Option<W>| {
        let vector = t.op(Op::Get(Loc::V(rn)));
        emit(
            t,
            VecOp::Extract {
                size,
                index,
                signed,
                vector,
            },
        )
    };
    let insert = |t: &mut Translator, value: Value| {
        let vector = t.op(Op::Get(Loc::V(rd)));
        let r = emit(
            t,
            VecOp::Insert {
                size,
                index,
                vector,
                value,
            },
        );
        t.set(Loc::V(rd), r);
    };
    match (op, imm4) {
        (false, 0b0000 | 0b0001) if size < 3 || q => {
            let value = if imm4 == 0 {
                extract(t, index, None)
            } else {
                t.x(Reg(rn))
            };
            let r = emit(t, VecOp::Dup(Arrangement { size, q }, value));
            t.set(Loc::V(rd), r);
        }
        (false, 0b0011) if q && insertable => {
            let value = t.x(Reg(rn));
            insert(t, value);
        }
        (false, 0b0101) if size < 2 || (q && size == 2) => {
            let r = extract(t, index, Some(W::of(q)));
            t.set_x(Reg(rd), r);
        }
        (false, 0b0111) if (q && size == 3) || (!q && size < 3) => {
            let r = extract(t, index, None);
            t.set_x(Reg(rd), r);
        }
        (true, _) if q && insertable => {
            let value = extract(t, imm4 >> size, None);
            insert(t, value);
        }
        _ => return false,
    }
    true
}

/// Emits `LD1`..`LD4`/`ST1`..`ST4` (multiple structures) and returns true, or emits nothing and
/// returns false when an interleaved form needs SSE4.1 lane inserts the host lacks.
///
/// `LD1`/`ST1` make one vector access per register, in register order. `LD2`..`LD4`/`ST2`..`ST4`
/// access the interleaved structures as 16-byte chunks (and an 8-byte tail) in address order, and
/// move each element between its chunk lane and its register lane. Either way a load writes the
/// registers only after every access succeeded and the base is written back last, so a fault
/// leaves the interpreter's state: no register or base written, and every byte before the
/// faulting one stored.
pub(in super::super) fn load_store_multiple(t: &mut Translator, inst: &Inst, pc: u64) -> bool {
    let Inst::VecMulti {
        load,
        q,
        size: element,
        rpt,
        selem,
        vt,
        rn,
        post,
    } = *inst
    else {
        unreachable!("only VecMulti reaches here");
    };
    if selem > 1 {
        return interleaved(t, load, q, element, selem, vt, rn, post, pc);
    }
    let size = if q { 4 } else { 3 };
    let base = t.x(rn);
    let mut loaded = [Value(0); 4];
    for r in 0..rpt {
        let addr = if r == 0 {
            base
        } else {
            let offset = t.op(Op::Const((r as u64) << size));
            t.op(Op::Add(W::W64, base, offset))
        };
        if load {
            loaded[r as usize] = t.op(Op::LoadVec { size, addr, pc });
        } else {
            let value = t.op(Op::Get(Loc::V((vt + r) % 32)));
            t.op(Op::StoreVec {
                size,
                addr,
                value,
                pc,
            });
        }
    }
    if load {
        for r in 0..rpt {
            t.set(Loc::V((vt + r) % 32), loaded[r as usize]);
        }
    }
    post_index(t, rn, post, base);
    true
}

#[allow(clippy::too_many_arguments)]
fn interleaved(
    t: &mut Translator,
    load: bool,
    q: bool,
    size: u8,
    selem: u8,
    vt: u8,
    rn: Reg,
    post: PostIndex,
    pc: u64,
) -> bool {
    if size.is_multiple_of(2) && !t.sse41 {
        return false;
    }
    let vector = |t: &mut Translator, op| t.op(Op::Simd(SimdOp::Vec(op)));
    let move_lane =
        |t: &mut Translator, from: Value, from_lane: usize, to: Value, to_lane: usize| {
            let value = vector(
                t,
                VecOp::Extract {
                    size,
                    index: from_lane as u8,
                    signed: None,
                    vector: from,
                },
            );
            vector(
                t,
                VecOp::Insert {
                    size,
                    index: to_lane as u8,
                    vector: to,
                    value,
                },
            )
        };
    let bytes = selem as usize * if q { 16 } else { 8 };
    let lanes = (if q { 16 } else { 8 }) >> size;
    let per_chunk = 16 >> size;
    let base = t.x(rn);
    let mut chunks = [(Value(0), Value(0)); 4];
    for (c, chunk) in chunks.iter_mut().enumerate().take(bytes.div_ceil(16)) {
        chunk.0 = if c == 0 {
            base
        } else {
            let offset = t.op(Op::Const(16 * c as u64));
            t.op(Op::Add(W::W64, base, offset))
        };
    }
    let chunk_size = |c: usize| if 16 * c + 16 <= bytes { 4 } else { 3 };
    let regs: [u8; 4] = std::array::from_fn(|s| (vt + s as u8) % 32);
    if load {
        for (c, (addr, value)) in chunks.iter_mut().enumerate().take(bytes.div_ceil(16)) {
            *value = t.op(Op::LoadVec {
                size: chunk_size(c),
                addr: *addr,
                pc,
            });
        }
        let mut loaded = [Value(0); 4];
        for (s, value) in loaded.iter_mut().enumerate().take(selem as usize) {
            *value = (0..lanes).fold(vector(t, VecOp::Imm(0, true)), |v, e| {
                let k = e * selem as usize + s;
                move_lane(t, chunks[k / per_chunk].1, k % per_chunk, v, e)
            });
        }
        for (s, &value) in loaded.iter().enumerate().take(selem as usize) {
            t.set(Loc::V(regs[s]), value);
        }
    } else {
        let mut source = [Value(0); 4];
        for (s, value) in source.iter_mut().enumerate().take(selem as usize) {
            *value = t.op(Op::Get(Loc::V(regs[s])));
        }
        for (c, &(addr, _)) in chunks.iter().enumerate().take(bytes.div_ceil(16)) {
            let elements = (bytes - 16 * c).min(16) >> size;
            let value = (0..elements).fold(vector(t, VecOp::Imm(0, true)), |v, i| {
                let k = c * per_chunk + i;
                move_lane(t, source[k % selem as usize], k / selem as usize, v, i)
            });
            t.op(Op::StoreVec {
                size: chunk_size(c),
                addr,
                value,
                pc,
            });
        }
    }
    post_index(t, rn, post, base);
    true
}

/// Emits `LD1`..`LD4`/`ST1`..`ST4` (single structure) or `LD1R`..`LD4R` and returns true, or emits
/// nothing and returns false when a lane insert needs SSE4.1 the host lacks. Like the
/// interpreter, each register is written as soon as its element loads, so a fault leaves the
/// earlier registers updated, and the base is written back last.
pub(in super::super) fn load_store_single(t: &mut Translator, inst: &Inst, pc: u64) -> bool {
    let (load, size, selem, lane, q, vt, rn, post) = match *inst {
        Inst::VecSingle {
            load,
            size,
            selem,
            index,
            vt,
            rn,
            post,
        } => (load, size, selem, Some(index), false, vt, rn, post),
        Inst::VecReplicate {
            q,
            size,
            selem,
            vt,
            rn,
            post,
        } => (true, size, selem, None, q, vt, rn, post),
        _ => unreachable!("only VecSingle and VecReplicate reach here"),
    };
    if load && lane.is_some() && size % 2 == 0 && !t.sse41 {
        return false;
    }
    let vector = |t: &mut Translator, op| t.op(Op::Simd(SimdOp::Vec(op)));
    let base = t.x(rn);
    for s in 0..selem {
        let addr = if s == 0 {
            base
        } else {
            let offset = t.op(Op::Const((s as u64) << size));
            t.op(Op::Add(W::W64, base, offset))
        };
        let reg = Loc::V((vt + s) % 32);
        match lane {
            Some(index) if !load => {
                let v = t.op(Op::Get(reg));
                let value = vector(
                    t,
                    VecOp::Extract {
                        size,
                        index,
                        signed: None,
                        vector: v,
                    },
                );
                t.op(Op::Store {
                    size,
                    addr,
                    value,
                    pc,
                });
            }
            Some(index) => {
                let value = t.op(Op::Load { size, addr, pc });
                let v = t.op(Op::Get(reg));
                let r = vector(
                    t,
                    VecOp::Insert {
                        size,
                        index,
                        vector: v,
                        value,
                    },
                );
                t.set(reg, r);
            }
            None => {
                let value = t.op(Op::Load { size, addr, pc });
                let r = vector(t, VecOp::Dup(Arrangement { size, q }, value));
                t.set(reg, r);
            }
        }
    }
    post_index(t, rn, post, base);
    true
}

fn post_index(t: &mut Translator, rn: Reg, post: PostIndex, base: Value) {
    let step = match post {
        PostIndex::None => return,
        PostIndex::Imm(n) => t.op(Op::Const(n as u64)),
        PostIndex::Reg(rm) => t.x(rm),
    };
    let new = t.op(Op::Add(W::W64, base, step));
    t.set_x(rn, new);
}

/// The SSE instructions the lowerings are built from, each `dst = dst op src` on lanes.
#[derive(Clone, Copy)]
enum Sse {
    Add,
    Sub,
    And,
    /// `!dst & src`.
    AndNot,
    Or,
    Xor,
    Eq,
    Gt,
    MaxU,
    MaxS,
    MinU,
    MinS,
    Mul,
    /// Unsigned `(d + s + 1) >> 1`, B and H lanes.
    AvgU,
    /// Saturating add or subtract, signed or unsigned, B and H lanes.
    AddSat(bool),
    SubSat(bool),
}

impl Emitter<'_> {
    fn sse(&mut self, op: Sse, size: u8, d: u8, s: u8) {
        match (op, size) {
            (Sse::Add, 0) => dynasm!(self.asm ; .arch x64 ; paddb Rx(d), Rx(s)),
            (Sse::Add, 1) => dynasm!(self.asm ; .arch x64 ; paddw Rx(d), Rx(s)),
            (Sse::Add, 2) => dynasm!(self.asm ; .arch x64 ; paddd Rx(d), Rx(s)),
            (Sse::Add, _) => dynasm!(self.asm ; .arch x64 ; paddq Rx(d), Rx(s)),
            (Sse::Sub, 0) => dynasm!(self.asm ; .arch x64 ; psubb Rx(d), Rx(s)),
            (Sse::Sub, 1) => dynasm!(self.asm ; .arch x64 ; psubw Rx(d), Rx(s)),
            (Sse::Sub, 2) => dynasm!(self.asm ; .arch x64 ; psubd Rx(d), Rx(s)),
            (Sse::Sub, _) => dynasm!(self.asm ; .arch x64 ; psubq Rx(d), Rx(s)),
            (Sse::And, _) => dynasm!(self.asm ; .arch x64 ; pand Rx(d), Rx(s)),
            (Sse::AndNot, _) => dynasm!(self.asm ; .arch x64 ; pandn Rx(d), Rx(s)),
            (Sse::Or, _) => dynasm!(self.asm ; .arch x64 ; por Rx(d), Rx(s)),
            (Sse::Xor, _) => dynasm!(self.asm ; .arch x64 ; pxor Rx(d), Rx(s)),
            (Sse::Eq, 0) => dynasm!(self.asm ; .arch x64 ; pcmpeqb Rx(d), Rx(s)),
            (Sse::Eq, 1) => dynasm!(self.asm ; .arch x64 ; pcmpeqw Rx(d), Rx(s)),
            (Sse::Eq, 2) => dynasm!(self.asm ; .arch x64 ; pcmpeqd Rx(d), Rx(s)),
            (Sse::Eq, _) => dynasm!(self.asm ; .arch x64 ; pcmpeqq Rx(d), Rx(s)),
            (Sse::Gt, 0) => dynasm!(self.asm ; .arch x64 ; pcmpgtb Rx(d), Rx(s)),
            (Sse::Gt, 1) => dynasm!(self.asm ; .arch x64 ; pcmpgtw Rx(d), Rx(s)),
            (Sse::Gt, 2) => dynasm!(self.asm ; .arch x64 ; pcmpgtd Rx(d), Rx(s)),
            (Sse::Gt, _) => dynasm!(self.asm ; .arch x64 ; pcmpgtq Rx(d), Rx(s)),
            (Sse::MaxU, 0) => dynasm!(self.asm ; .arch x64 ; pmaxub Rx(d), Rx(s)),
            (Sse::MaxU, 1) => dynasm!(self.asm ; .arch x64 ; pmaxuw Rx(d), Rx(s)),
            (Sse::MaxU, _) => dynasm!(self.asm ; .arch x64 ; pmaxud Rx(d), Rx(s)),
            (Sse::MaxS, 0) => dynasm!(self.asm ; .arch x64 ; pmaxsb Rx(d), Rx(s)),
            (Sse::MaxS, 1) => dynasm!(self.asm ; .arch x64 ; pmaxsw Rx(d), Rx(s)),
            (Sse::MaxS, _) => dynasm!(self.asm ; .arch x64 ; pmaxsd Rx(d), Rx(s)),
            (Sse::MinU, 0) => dynasm!(self.asm ; .arch x64 ; pminub Rx(d), Rx(s)),
            (Sse::MinU, 1) => dynasm!(self.asm ; .arch x64 ; pminuw Rx(d), Rx(s)),
            (Sse::MinU, _) => dynasm!(self.asm ; .arch x64 ; pminud Rx(d), Rx(s)),
            (Sse::MinS, 0) => dynasm!(self.asm ; .arch x64 ; pminsb Rx(d), Rx(s)),
            (Sse::MinS, 1) => dynasm!(self.asm ; .arch x64 ; pminsw Rx(d), Rx(s)),
            (Sse::MinS, _) => dynasm!(self.asm ; .arch x64 ; pminsd Rx(d), Rx(s)),
            (Sse::Mul, 1) => dynasm!(self.asm ; .arch x64 ; pmullw Rx(d), Rx(s)),
            (Sse::Mul, _) => dynasm!(self.asm ; .arch x64 ; pmulld Rx(d), Rx(s)),
            (Sse::AvgU, 0) => dynasm!(self.asm ; .arch x64 ; pavgb Rx(d), Rx(s)),
            (Sse::AvgU, _) => dynasm!(self.asm ; .arch x64 ; pavgw Rx(d), Rx(s)),
            (Sse::AddSat(true), 0) => dynasm!(self.asm ; .arch x64 ; paddsb Rx(d), Rx(s)),
            (Sse::AddSat(true), _) => dynasm!(self.asm ; .arch x64 ; paddsw Rx(d), Rx(s)),
            (Sse::AddSat(false), 0) => dynasm!(self.asm ; .arch x64 ; paddusb Rx(d), Rx(s)),
            (Sse::AddSat(false), _) => dynasm!(self.asm ; .arch x64 ; paddusw Rx(d), Rx(s)),
            (Sse::SubSat(true), 0) => dynasm!(self.asm ; .arch x64 ; psubsb Rx(d), Rx(s)),
            (Sse::SubSat(true), _) => dynasm!(self.asm ; .arch x64 ; psubsw Rx(d), Rx(s)),
            (Sse::SubSat(false), 0) => dynasm!(self.asm ; .arch x64 ; psubusb Rx(d), Rx(s)),
            (Sse::SubSat(false), _) => dynasm!(self.asm ; .arch x64 ; psubusw Rx(d), Rx(s)),
        }
    }

    fn invert(&mut self, d: u8) {
        dynasm!(self.asm
            ; .arch x64
            ; pcmpeqd Rx(SCRATCH), Rx(SCRATCH)
            ; pxor Rx(d), Rx(SCRATCH)
        );
    }

    /// Shifts each lane of `8 << size` bits of `xmm d` by `amount`; `Sshr` takes H and S lanes.
    fn shift_imm(&mut self, op: Shift, size: u8, d: u8, amount: u8) {
        let n = amount as i8;
        match (op, size) {
            (Shift::Shl, 1) => dynasm!(self.asm ; .arch x64 ; psllw Rx(d), n),
            (Shift::Shl, 2) => dynasm!(self.asm ; .arch x64 ; pslld Rx(d), n),
            (Shift::Shl, _) => dynasm!(self.asm ; .arch x64 ; psllq Rx(d), n),
            (Shift::Ushr, 1) => dynasm!(self.asm ; .arch x64 ; psrlw Rx(d), n),
            (Shift::Ushr, 2) => dynasm!(self.asm ; .arch x64 ; psrld Rx(d), n),
            (Shift::Ushr, _) => dynasm!(self.asm ; .arch x64 ; psrlq Rx(d), n),
            (Shift::Sshr, 1) => dynasm!(self.asm ; .arch x64 ; psraw Rx(d), n),
            (Shift::Sshr, _) => dynasm!(self.asm ; .arch x64 ; psrad Rx(d), n),
        }
    }

    fn copy(&mut self, d: u8, s: u8) {
        dynasm!(self.asm ; .arch x64 ; movdqa Rx(d), Rx(s));
    }

    /// `x` = `s` with each pair of `8 << size`-bit lanes summed into one lane twice as wide: the
    /// sum zero-extended for bytes, sign-extended for halfwords, and in the high half for words.
    fn pair_sums(&mut self, size: u8, x: u8, s: u8) {
        self.copy(x, s);
        match size {
            0 => dynasm!(self.asm
                ; .arch x64
                ; psllw Rx(x), 8
                ; paddb Rx(x), Rx(s)
                ; psrlw Rx(x), 8
            ),
            1 => dynasm!(self.asm
                ; .arch x64
                ; pslld Rx(x), 16
                ; paddw Rx(x), Rx(s)
                ; psrad Rx(x), 16
            ),
            _ => dynasm!(self.asm
                ; .arch x64
                ; psllq Rx(x), 32
                ; paddd Rx(x), Rx(s)
            ),
        }
    }

    /// Narrows the pair sums `pair_sums` left in `d` and `s` to the low and high half of `d`.
    fn pack_sums(&mut self, size: u8, d: u8, s: u8) {
        match size {
            0 => dynasm!(self.asm ; .arch x64 ; packuswb Rx(d), Rx(s)),
            1 => dynasm!(self.asm ; .arch x64 ; packssdw Rx(d), Rx(s)),
            _ => dynasm!(self.asm ; .arch x64 ; shufps Rx(d), Rx(s), 0xdd_u8 as i8),
        }
    }

    /// `d` = the low (or, when `high`, the high) half of `s`'s `8 << size`-bit lanes, each sign- or
    /// zero-extended to twice its width. Clobbers `xmm15`.
    fn widen(&mut self, signed: bool, size: u8, high: bool, d: u8, s: u8) {
        if high {
            dynasm!(self.asm ; .arch x64 ; pshufd Rx(d), Rx(s), 0xee_u8 as i8);
        } else {
            self.copy(d, s);
        }
        match (size, signed) {
            (0, true) => dynasm!(self.asm ; .arch x64 ; punpcklbw Rx(d), Rx(d) ; psraw Rx(d), 8),
            (0, false) => dynasm!(self.asm ; .arch x64 ; punpcklbw Rx(d), Rx(d) ; psrlw Rx(d), 8),
            (1, true) => dynasm!(self.asm ; .arch x64 ; punpcklwd Rx(d), Rx(d) ; psrad Rx(d), 16),
            (1, false) => dynasm!(self.asm ; .arch x64 ; punpcklwd Rx(d), Rx(d) ; psrld Rx(d), 16),
            (_, true) => dynasm!(self.asm
                ; .arch x64
                ; movdqa Rx(SCRATCH), Rx(d)
                ; psrad Rx(SCRATCH), 31
                ; punpckldq Rx(d), Rx(SCRATCH)
            ),
            (_, false) => dynasm!(self.asm ; .arch x64 ; punpckldq Rx(d), Rx(d) ; psrlq Rx(d), 32),
        }
    }

    /// `d` = `a - d` on lanes of `8 << size` bits.
    fn sub_from(&mut self, size: u8, d: u8, a: u8) {
        self.copy(SCRATCH, a);
        self.sse(Sse::Sub, size, SCRATCH, d);
        self.copy(d, SCRATCH);
    }

    /// `d` = the products of the low (or, when `high`, the high) halves of `n`'s and `m`'s
    /// `8 << size`-bit lanes, each twice as wide, signed or unsigned. Clobbers `xmm15`.
    fn mul_long(&mut self, signed: bool, size: u8, high: bool, d: u8, n: u8, m: u8) {
        match size {
            0 => {
                self.widen(signed, 0, high, d, n);
                self.widen(signed, 0, high, SCRATCH, m);
                self.sse(Sse::Mul, 1, d, SCRATCH);
            }
            1 => {
                self.copy(d, n);
                self.sse(Sse::Mul, 1, d, m);
                self.copy(SCRATCH, n);
                if signed {
                    dynasm!(self.asm ; .arch x64 ; pmulhw Rx(SCRATCH), Rx(m));
                } else {
                    dynasm!(self.asm ; .arch x64 ; pmulhuw Rx(SCRATCH), Rx(m));
                }
                if high {
                    dynasm!(self.asm ; .arch x64 ; punpckhwd Rx(d), Rx(SCRATCH));
                } else {
                    dynasm!(self.asm ; .arch x64 ; punpcklwd Rx(d), Rx(SCRATCH));
                }
            }
            _ => {
                let lanes = if high { 0xfa_u8 } else { 0x50 } as i8;
                dynasm!(self.asm
                    ; .arch x64
                    ; pshufd Rx(d), Rx(n), lanes
                    ; pshufd Rx(SCRATCH), Rx(m), lanes
                );
                if signed {
                    dynasm!(self.asm ; .arch x64 ; pmuldq Rx(d), Rx(SCRATCH));
                } else {
                    dynasm!(self.asm ; .arch x64 ; pmuludq Rx(d), Rx(SCRATCH));
                }
            }
        }
    }

    fn addp(&mut self, size: u8, q: bool, d: u8, a: u8, b: u8) {
        if size == 3 {
            dynasm!(self.asm
                ; .arch x64
                ; movdqa Rx(d), Rx(a)
                ; punpcklqdq Rx(d), Rx(b)
                ; movdqa Rx(SCRATCH), Rx(a)
                ; punpckhqdq Rx(SCRATCH), Rx(b)
                ; paddq Rx(d), Rx(SCRATCH)
            );
        } else if q {
            self.pair_sums(size, d, a);
            self.pair_sums(size, SCRATCH, b);
            self.pack_sums(size, d, SCRATCH);
        } else {
            self.copy(d, a);
            dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(d), Rx(b));
            self.pair_sums(size, SCRATCH, d);
            self.pack_sums(size, SCRATCH, SCRATCH);
            self.copy(d, SCRATCH);
        }
    }

    /// `d` = `op` of the even and odd lanes of `8 << size` bits of `a` followed by `b`, so each lane
    /// pair of `a` lands in the low half and each of `b` in the high half. Without `q`, only the low
    /// halves of `a` and `b` count and the high half of `d` is left undefined. Clobbers `xmm15`.
    fn min_max_pairs(&mut self, op: Sse, size: u8, q: bool, d: u8, a: u8, b: u8) {
        let (a, b) = if q {
            (a, b)
        } else {
            let x = self.temp();
            self.copy(x, a);
            dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(x), Rx(b));
            (x, x)
        };
        match size {
            0 => {
                let t = self.temp();
                dynasm!(self.asm
                    ; .arch x64
                    ; pcmpeqw Rx(t), Rx(t)
                    ; psrlw Rx(t), 8
                    ; movdqa Rx(d), Rx(a)
                    ; pand Rx(d), Rx(t)
                    ; movdqa Rx(SCRATCH), Rx(b)
                    ; pand Rx(SCRATCH), Rx(t)
                    ; packuswb Rx(d), Rx(SCRATCH)
                    ; movdqa Rx(SCRATCH), Rx(a)
                    ; psrlw Rx(SCRATCH), 8
                    ; movdqa Rx(t), Rx(b)
                    ; psrlw Rx(t), 8
                    ; packuswb Rx(SCRATCH), Rx(t)
                );
            }
            1 => {
                let t = self.temp();
                dynasm!(self.asm
                    ; .arch x64
                    ; movdqa Rx(d), Rx(a)
                    ; pslld Rx(d), 16
                    ; psrad Rx(d), 16
                    ; movdqa Rx(SCRATCH), Rx(b)
                    ; pslld Rx(SCRATCH), 16
                    ; psrad Rx(SCRATCH), 16
                    ; packssdw Rx(d), Rx(SCRATCH)
                    ; movdqa Rx(SCRATCH), Rx(a)
                    ; psrad Rx(SCRATCH), 16
                    ; movdqa Rx(t), Rx(b)
                    ; psrad Rx(t), 16
                    ; packssdw Rx(SCRATCH), Rx(t)
                );
            }
            _ => dynasm!(self.asm
                ; .arch x64
                ; movdqa Rx(d), Rx(a)
                ; shufps Rx(d), Rx(b), 0x88_u8 as i8
                ; movdqa Rx(SCRATCH), Rx(a)
                ; shufps Rx(SCRATCH), Rx(b), 0xdd_u8 as i8
            ),
        }
        self.sse(op, size, d, SCRATCH);
    }

    /// `SQADD`, `UQADD`, `SQSUB` or `UQSUB` (by `word`) on B or H lanes, setting FPSR.QC when
    /// any lane of the arrangement saturated, that is when the wrapping result differs.
    pub(super) fn sat_arith(
        &mut self,
        i: usize,
        arrangement: Arrangement,
        a: Value,
        b: Value,
        word: u32,
    ) {
        let Arrangement { size, q } = arrangement;
        let (signed, sub) = (word & 0x2000_0000 == 0, word & 0x2000 != 0);
        let (a, b) = (self.reg(a), self.reg(b));
        let d = self.def(Value(i as u32));
        let (wrap, sat) = match sub {
            false => (Sse::Add, Sse::AddSat(signed)),
            true => (Sse::Sub, Sse::SubSat(signed)),
        };
        self.copy(SCRATCH, a);
        self.sse(wrap, size, SCRATCH, b);
        self.copy(d, a);
        self.sse(sat, size, d, b);
        self.sse(Sse::Eq, size, SCRATCH, d);
        let (exact, all) = (self.asm.new_dynamic_label(), if q { 0xffff } else { 0xff });
        dynasm!(self.asm
            ; .arch x64
            ; pmovmskb eax, Rx(SCRATCH)
            ; and eax, all
            ; cmp eax, all
            ; je =>exact
            ; or DWORD [r15 + CTX_FPSR], QC
            ; =>exact
        );
        if !q {
            dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(d));
        }
    }

    /// `SQDMULH`, or `SQRDMULH` when `round`, on H lanes: the high half of each doubled product,
    /// rounded when `round`. Only `-0x8000 * -0x8000` gives `-0x8000` there, which saturates to `0x7fff`
    /// and sets FPSR.QC.
    pub(super) fn sat_mul_high(
        &mut self,
        i: usize,
        round: bool,
        arrangement: Arrangement,
        a: Value,
        b: Value,
    ) {
        let q = arrangement.q;
        let (a, b) = (self.reg(a), self.reg(b));
        let d = self.def(Value(i as u32));
        self.copy(d, a);
        if round {
            dynasm!(self.asm ; .arch x64 ; pmulhrsw Rx(d), Rx(b));
        } else {
            dynasm!(self.asm
                ; .arch x64
                ; pmulhw Rx(d), Rx(b)
                ; psllw Rx(d), 1
                ; movdqa Rx(SCRATCH), Rx(a)
                ; pmullw Rx(SCRATCH), Rx(b)
                ; psrlw Rx(SCRATCH), 15
                ; por Rx(d), Rx(SCRATCH)
            );
        }
        let (exact, all) = (self.asm.new_dynamic_label(), if q { 0xffff } else { 0xff });
        dynasm!(self.asm
            ; .arch x64
            ; pcmpeqw Rx(SCRATCH), Rx(SCRATCH)
            ; psllw Rx(SCRATCH), 15
            ; pcmpeqw Rx(SCRATCH), Rx(d)
            ; pxor Rx(d), Rx(SCRATCH)
            ; pmovmskb eax, Rx(SCRATCH)
            ; test eax, all
            ; jz =>exact
            ; or DWORD [r15 + CTX_FPSR], QC
            ; =>exact
        );
        if !q {
            dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(d));
        }
    }

    /// Sets FPSR.QC when any byte of `mask` is nonzero.
    fn qc_if_any(&mut self, mask: u8) {
        let exact = self.asm.new_dynamic_label();
        dynasm!(self.asm
            ; .arch x64
            ; pmovmskb eax, Rx(mask)
            ; test eax, eax
            ; jz =>exact
            ; or DWORD [r15 + CTX_FPSR], QC
            ; =>exact
        );
    }

    /// Each lane of `16 << size` bits of `x` replaced by copies of its sign bit (`size` 1 or 2).
    fn lane_signs(&mut self, size: u8, x: u8) {
        dynasm!(self.asm ; .arch x64 ; psrad Rx(x), 31);
        if size == 2 {
            dynasm!(self.asm ; .arch x64 ; pshufd Rx(x), Rx(x), 0xf5_u8 as i8);
        }
    }

    /// `SQDMULL`, `SQDMLAL` or `SQDMLSL` (see `SimdOp::SatMulLong`). Only `MIN * MIN` overflows
    /// the doubling, and it alone gives `MIN` there, so those lanes become `MAX`. The accumulate
    /// overflows where the operands' signs say it must, and saturates towards the accumulator's sign.
    pub(super) fn sat_mul_long(
        &mut self,
        i: usize,
        size: u8,
        high: bool,
        n: Value,
        m: Value,
        acc: Option<(Value, bool)>,
    ) {
        let (n, m) = (self.reg(n), self.reg(m));
        let acc = acc.map(|(acc, sub)| (self.reg(acc), sub));
        let d = self.def(Value(i as u32));
        let (t, u) = (self.temp(), self.temp());
        let wide = size + 1;
        self.mul_long(true, size, high, d, n, m);
        self.sse(Sse::Add, wide, d, d);
        dynasm!(self.asm ; .arch x64 ; pcmpeqd Rx(t), Rx(t));
        if size == 1 {
            dynasm!(self.asm ; .arch x64 ; pslld Rx(t), 31);
        } else {
            dynasm!(self.asm ; .arch x64 ; psllq Rx(t), 63);
        }
        self.sse(Sse::Eq, wide, t, d);
        self.sse(Sse::Xor, wide, d, t);
        self.qc_if_any(t);
        let Some((acc, sub)) = acc else {
            return;
        };
        self.copy(SCRATCH, acc);
        self.copy(t, acc);
        if sub {
            self.sse(Sse::Sub, wide, SCRATCH, d);
            self.sse(Sse::Xor, wide, t, d);
            self.copy(d, acc);
            self.sse(Sse::Xor, wide, d, SCRATCH);
        } else {
            self.sse(Sse::Add, wide, SCRATCH, d);
            self.sse(Sse::Xor, wide, t, SCRATCH);
            self.sse(Sse::Xor, wide, d, SCRATCH);
        }
        self.sse(Sse::And, wide, t, d);
        self.lane_signs(size, t);
        dynasm!(self.asm ; .arch x64 ; pcmpeqd Rx(u), Rx(u));
        if size == 1 {
            dynasm!(self.asm ; .arch x64 ; psrld Rx(u), 1);
        } else {
            dynasm!(self.asm ; .arch x64 ; psrlq Rx(u), 1);
        }
        self.copy(d, acc);
        self.lane_signs(size, d);
        self.sse(Sse::Xor, wide, d, u);
        self.sse(Sse::Xor, wide, d, SCRATCH);
        self.sse(Sse::And, wide, d, t);
        self.sse(Sse::Xor, wide, d, SCRATCH);
        self.qc_if_any(t);
    }

    /// `SQXTN`/`SQXTN2`, or `SQXTUN`/`SQXTUN2` when `unsigned`: packs `n` with signed or unsigned
    /// saturation, and sets FPSR.QC unless sign- or zero-extending the packed lanes again gives
    /// back every lane of `n`.
    pub(super) fn sat_narrow(
        &mut self,
        i: usize,
        size: u8,
        unsigned: bool,
        n: Value,
        low: Option<Value>,
    ) {
        let n = self.reg(n);
        let low = low.map(|low| self.reg(low));
        let d = self.def(Value(i as u32));
        dynasm!(self.asm ; .arch x64 ; movaps xmm15, Rx(n));
        match (size, unsigned) {
            (0, false) => dynasm!(self.asm ; .arch x64 ; packsswb xmm15, xmm15),
            (_, false) => dynasm!(self.asm ; .arch x64 ; packssdw xmm15, xmm15),
            (0, true) => dynasm!(self.asm ; .arch x64 ; packuswb xmm15, xmm15),
            (_, true) => dynasm!(self.asm ; .arch x64 ; packusdw xmm15, xmm15),
        }
        match low {
            Some(low) => dynasm!(self.asm
                ; .arch x64
                ; movaps Rx(d), Rx(low)
                ; punpcklqdq Rx(d), xmm15
            ),
            None => dynasm!(self.asm ; .arch x64 ; movq Rx(d), xmm15),
        }
        match size {
            0 => dynasm!(self.asm ; .arch x64 ; punpcklbw xmm15, xmm15),
            _ => dynasm!(self.asm ; .arch x64 ; punpcklwd xmm15, xmm15),
        }
        match (size, unsigned) {
            (0, false) => dynasm!(self.asm ; .arch x64 ; psraw xmm15, 8 ; pcmpeqw xmm15, Rx(n)),
            (_, false) => dynasm!(self.asm ; .arch x64 ; psrad xmm15, 16 ; pcmpeqd xmm15, Rx(n)),
            (0, true) => dynasm!(self.asm ; .arch x64 ; psrlw xmm15, 8 ; pcmpeqw xmm15, Rx(n)),
            (_, true) => dynasm!(self.asm ; .arch x64 ; psrld xmm15, 16 ; pcmpeqd xmm15, Rx(n)),
        }
        let exact = self.asm.new_dynamic_label();
        dynasm!(self.asm
            ; .arch x64
            ; pmovmskb eax, xmm15
            ; cmp eax, 0xffff
            ; je =>exact
            ; or DWORD [r15 + CTX_FPSR], QC
            ; =>exact
        );
    }

    pub(super) fn vector(&mut self, i: usize, op: VecOp) {
        let v = Value(i as u32);
        let (d, q) = match op {
            VecOp::Binary(op, Arrangement { size, q }, a, b) => {
                let (a, b) = (self.reg(a), self.reg(b));
                let d = self.def(v);
                match op {
                    Binary::Bic => {
                        self.copy(d, b);
                        self.sse(Sse::AndNot, size, d, a);
                    }
                    Binary::Orn => {
                        dynasm!(self.asm ; .arch x64 ; pcmpeqd Rx(SCRATCH), Rx(SCRATCH));
                        self.sse(Sse::Xor, size, SCRATCH, b);
                        self.copy(d, a);
                        self.sse(Sse::Or, size, d, SCRATCH);
                    }
                    Binary::CmGe => {
                        self.copy(d, b);
                        self.sse(Sse::Gt, size, d, a);
                        self.invert(d);
                    }
                    Binary::CmHi => {
                        self.copy(d, a);
                        self.sse(Sse::MaxU, size, d, b);
                        self.sse(Sse::Eq, size, d, b);
                        self.invert(d);
                    }
                    Binary::CmHs => {
                        self.copy(d, a);
                        self.sse(Sse::MaxU, size, d, b);
                        self.sse(Sse::Eq, size, d, a);
                    }
                    Binary::Addp => self.addp(size, q, d, a, b),
                    Binary::MinMaxP { max, signed } => {
                        let op = match (max, signed) {
                            (true, true) => Sse::MaxS,
                            (true, false) => Sse::MaxU,
                            (false, true) => Sse::MinS,
                            (false, false) => Sse::MinU,
                        };
                        self.min_max_pairs(op, size, q, d, a, b);
                    }
                    Binary::CmTst => {
                        self.copy(d, a);
                        self.sse(Sse::And, size, d, b);
                        dynasm!(self.asm ; .arch x64 ; pxor Rx(SCRATCH), Rx(SCRATCH));
                        self.sse(Sse::Eq, size, d, SCRATCH);
                        self.invert(d);
                    }
                    Binary::Rhadd(false) if size < 2 => {
                        self.copy(d, a);
                        self.sse(Sse::AvgU, size, d, b);
                    }
                    Binary::Hadd(signed) | Binary::Rhadd(signed) => {
                        let shift = if signed { Shift::Sshr } else { Shift::Ushr };
                        self.copy(SCRATCH, a);
                        self.sse(Sse::Xor, size, SCRATCH, b);
                        self.shift_imm(shift, size, SCRATCH, 1);
                        self.copy(d, a);
                        if matches!(op, Binary::Hadd(_)) {
                            self.sse(Sse::And, size, d, b);
                            self.sse(Sse::Add, size, d, SCRATCH);
                        } else {
                            self.sse(Sse::Or, size, d, b);
                            self.sse(Sse::Sub, size, d, SCRATCH);
                        }
                    }
                    Binary::Abd(signed) => {
                        let (max, min) = if signed {
                            (Sse::MaxS, Sse::MinS)
                        } else {
                            (Sse::MaxU, Sse::MinU)
                        };
                        self.copy(SCRATCH, a);
                        self.sse(min, size, SCRATCH, b);
                        self.copy(d, a);
                        self.sse(max, size, d, b);
                        self.sse(Sse::Sub, size, d, SCRATCH);
                    }
                    _ => {
                        let op = match op {
                            Binary::Add => Sse::Add,
                            Binary::Sub => Sse::Sub,
                            Binary::And => Sse::And,
                            Binary::Orr => Sse::Or,
                            Binary::Eor => Sse::Xor,
                            Binary::CmEq => Sse::Eq,
                            Binary::CmGt => Sse::Gt,
                            Binary::SMax => Sse::MaxS,
                            Binary::SMin => Sse::MinS,
                            Binary::UMax => Sse::MaxU,
                            Binary::UMin => Sse::MinU,
                            _ => Sse::Mul,
                        };
                        self.copy(d, a);
                        self.sse(op, size, d, b);
                    }
                }
                (d, q)
            }
            VecOp::Shift(op, Arrangement { size, q }, amount, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                self.copy(d, a);
                self.shift_imm(op, size, d, amount);
                (d, q)
            }
            VecOp::RoundShift {
                signed,
                arrangement: Arrangement { size, q },
                shift,
                acc,
                a,
            } => {
                let a = self.reg(a);
                let acc = acc.map(|acc| self.reg(acc));
                let d = self.def(v);
                let esize = 8 << size;
                self.copy(SCRATCH, a);
                self.shift_imm(Shift::Shl, size, SCRATCH, esize - shift);
                self.shift_imm(Shift::Ushr, size, SCRATCH, esize - 1);
                if let Some(acc) = acc {
                    self.sse(Sse::Add, size, SCRATCH, acc);
                }
                self.copy(d, a);
                let op = if signed { Shift::Sshr } else { Shift::Ushr };
                self.shift_imm(op, size, d, shift);
                self.sse(Sse::Add, size, d, SCRATCH);
                (d, q)
            }
            VecOp::ScalarDup {
                size,
                index,
                vector,
            } => {
                let s = self.reg(vector);
                let d = self.def(v);
                self.copy(d, s);
                if index != 0 {
                    let bytes = (index << size) as i8;
                    dynasm!(self.asm ; .arch x64 ; psrldq Rx(d), bytes);
                }
                match size {
                    0 => {
                        dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d) ; movzx eax, al ; movd Rx(d), eax)
                    }
                    1 => {
                        dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d) ; movzx eax, ax ; movd Rx(d), eax)
                    }
                    2 => dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d) ; movd Rx(d), eax),
                    _ => dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(d)),
                }
                return;
            }
            VecOp::Ext { q, index, n, m } => {
                let (n, m) = (self.reg(n), self.reg(m));
                let d = self.def(v);
                let k = index as i8;
                self.copy(d, n);
                if q {
                    dynasm!(self.asm
                        ; .arch x64
                        ; psrldq Rx(d), k
                        ; movdqa Rx(SCRATCH), Rx(m)
                        ; pslldq Rx(SCRATCH), 16 - k
                        ; por Rx(d), Rx(SCRATCH)
                    );
                } else {
                    dynasm!(self.asm
                        ; .arch x64
                        ; punpcklqdq Rx(d), Rx(m)
                        ; psrldq Rx(d), k
                    );
                }
                (d, q)
            }
            VecOp::Zip {
                high,
                arrangement: Arrangement { size, q },
                n,
                m,
            } => {
                let (n, m) = (self.reg(n), self.reg(m));
                let d = self.def(v);
                self.copy(d, n);
                match (size, high && q) {
                    (0, false) => dynasm!(self.asm ; .arch x64 ; punpcklbw Rx(d), Rx(m)),
                    (1, false) => dynasm!(self.asm ; .arch x64 ; punpcklwd Rx(d), Rx(m)),
                    (2, false) => dynasm!(self.asm ; .arch x64 ; punpckldq Rx(d), Rx(m)),
                    (_, false) => dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(d), Rx(m)),
                    (0, true) => dynasm!(self.asm ; .arch x64 ; punpckhbw Rx(d), Rx(m)),
                    (1, true) => dynasm!(self.asm ; .arch x64 ; punpckhwd Rx(d), Rx(m)),
                    (2, true) => dynasm!(self.asm ; .arch x64 ; punpckhdq Rx(d), Rx(m)),
                    (_, true) => dynasm!(self.asm ; .arch x64 ; punpckhqdq Rx(d), Rx(m)),
                }
                if high && !q {
                    dynasm!(self.asm ; .arch x64 ; psrldq Rx(d), 8);
                }
                (d, q)
            }
            VecOp::Select { op, q, old, n, m } => {
                let (old, n, m) = (self.reg(old), self.reg(n), self.reg(m));
                let d = self.def(v);
                let (base, mask) = match op {
                    Select::Bsl => (m, old),
                    Select::Bit | Select::Bif => (old, m),
                };
                self.copy(d, n);
                self.sse(Sse::Xor, 0, d, base);
                if op == Select::Bif {
                    self.copy(SCRATCH, mask);
                    self.sse(Sse::AndNot, 0, SCRATCH, d);
                    self.copy(d, base);
                    self.sse(Sse::Xor, 0, d, SCRATCH);
                } else {
                    self.sse(Sse::And, 0, d, mask);
                    self.sse(Sse::Xor, 0, d, base);
                }
                (d, q)
            }
            VecOp::Long {
                op,
                signed,
                size,
                high,
                n,
                m,
                acc,
            } => {
                let (n, m) = (self.reg(n), self.reg(m));
                let acc = acc.map(|(acc, sub)| (self.reg(acc), sub));
                let d = self.def(v);
                let wide = size + 1;
                match op {
                    Long::Add | Long::Sub => {
                        let t = self.temp();
                        self.widen(signed, size, high, d, n);
                        self.widen(signed, size, high, t, m);
                        let op = if op == Long::Add { Sse::Add } else { Sse::Sub };
                        self.sse(op, wide, d, t);
                    }
                    Long::AddWide => {
                        self.widen(signed, size, high, d, m);
                        self.sse(Sse::Add, wide, d, n);
                    }
                    Long::SubWide => {
                        self.widen(signed, size, high, d, m);
                        self.sub_from(wide, d, n);
                    }
                    Long::Abd => {
                        let (max, min) = if signed {
                            (Sse::MaxS, Sse::MinS)
                        } else {
                            (Sse::MaxU, Sse::MinU)
                        };
                        self.copy(SCRATCH, n);
                        self.sse(min, size, SCRATCH, m);
                        self.copy(d, n);
                        self.sse(max, size, d, m);
                        self.sse(Sse::Sub, size, d, SCRATCH);
                        self.widen(false, size, high, d, d);
                    }
                    Long::Mul => self.mul_long(signed, size, high, d, n, m),
                }
                match acc {
                    Some((acc, true)) => self.sub_from(wide, d, acc),
                    Some((acc, false)) => self.sse(Sse::Add, wide, d, acc),
                    None => {}
                }
                return;
            }
            VecOp::ShiftLong {
                signed,
                size,
                high,
                shift,
                a,
            } => {
                let a = self.reg(a);
                let d = self.def(v);
                self.widen(signed, size, high, d, a);
                let n = shift as i8;
                match (size, shift) {
                    (_, 0) => {}
                    (0, _) => dynasm!(self.asm ; .arch x64 ; psllw Rx(d), n),
                    (1, _) => dynasm!(self.asm ; .arch x64 ; pslld Rx(d), n),
                    _ => dynasm!(self.asm ; .arch x64 ; psllq Rx(d), n),
                }
                return;
            }
            VecOp::Imm(imm, q) => {
                let d = self.def(v);
                if imm == 0 {
                    dynasm!(self.asm ; .arch x64 ; pxor Rx(d), Rx(d));
                    return;
                }
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD imm as i64
                    ; movq Rx(d), rax
                );
                if q {
                    dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(d), Rx(d));
                }
                return;
            }
            VecOp::MulAcc {
                sub,
                arrangement: Arrangement { size, q },
                acc,
                n,
                m,
            } => {
                let (acc, n, m) = (self.reg(acc), self.reg(n), self.reg(m));
                let d = self.def(v);
                self.copy(SCRATCH, n);
                self.sse(Sse::Mul, size, SCRATCH, m);
                self.copy(d, acc);
                self.sse(if sub { Sse::Sub } else { Sse::Add }, size, d, SCRATCH);
                (d, q)
            }
            VecOp::Table { q, table, index } => {
                let (table, index) = (self.reg(table), self.reg(index));
                let d = self.def(v);
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD TABLE_BIAS.0.as_ptr() as i64
                    ; movdqa Rx(SCRATCH), Rx(index)
                    ; paddusb Rx(SCRATCH), [rax]
                );
                self.copy(d, table);
                dynasm!(self.asm ; .arch x64 ; pshufb Rx(d), Rx(SCRATCH));
                (d, q)
            }
            VecOp::Ushl(Arrangement { size, q }, n, m) => {
                let (n, m) = (self.reg(n), self.reg(m));
                let d = self.def(v);
                let masks = match size {
                    2 => SHIFT_MASKS.0.as_ptr() as i64,
                    _ => SHIFT_MASKS.1.as_ptr() as i64,
                };
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD masks
                    ; movdqa Rx(SCRATCH), Rx(m)
                    ; pand Rx(SCRATCH), [rax]
                    ; movdqu [r13 + ST_SCRATCH], Rx(SCRATCH)
                    ; pxor Rx(SCRATCH), [rax]
                );
                match size {
                    2 => dynasm!(self.asm
                        ; .arch x64
                        ; paddd Rx(SCRATCH), [rax + 16]
                        ; vpsrlvd Rx(SCRATCH), Rx(n), Rx(SCRATCH)
                        ; vpsllvd Rx(d), Rx(n), [r13 + ST_SCRATCH]
                    ),
                    _ => dynasm!(self.asm
                        ; .arch x64
                        ; paddq Rx(SCRATCH), [rax + 16]
                        ; vpsrlvq Rx(SCRATCH), Rx(n), Rx(SCRATCH)
                        ; vpsllvq Rx(d), Rx(n), [r13 + ST_SCRATCH]
                    ),
                }
                dynasm!(self.asm ; .arch x64 ; por Rx(d), Rx(SCRATCH));
                (d, q)
            }
            VecOp::AddAcross(Arrangement { size, q }, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match (size, q) {
                    (0, _) => {
                        match q {
                            true => self.copy(d, a),
                            false => dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(a)),
                        }
                        dynasm!(self.asm
                            ; .arch x64
                            ; pxor Rx(SCRATCH), Rx(SCRATCH)
                            ; psadbw Rx(d), Rx(SCRATCH)
                            ; pshufd Rx(SCRATCH), Rx(d), 0x4e
                            ; paddd Rx(d), Rx(SCRATCH)
                            ; movd eax, Rx(d)
                            ; movzx eax, al
                        );
                    }
                    (1, _) => {
                        self.copy(d, a);
                        if q {
                            dynasm!(self.asm
                                ; .arch x64
                                ; pshufd Rx(SCRATCH), Rx(d), 0x4e
                                ; paddw Rx(d), Rx(SCRATCH)
                            );
                        }
                        dynasm!(self.asm
                            ; .arch x64
                            ; pshuflw Rx(SCRATCH), Rx(d), 0x4e
                            ; paddw Rx(d), Rx(SCRATCH)
                            ; pshuflw Rx(SCRATCH), Rx(d), 0xb1u8 as i8
                            ; paddw Rx(d), Rx(SCRATCH)
                            ; movd eax, Rx(d)
                            ; movzx eax, ax
                        );
                    }
                    _ => {
                        self.copy(d, a);
                        dynasm!(self.asm
                            ; .arch x64
                            ; pshufd Rx(SCRATCH), Rx(d), 0x4e
                            ; paddd Rx(d), Rx(SCRATCH)
                            ; pshufd Rx(SCRATCH), Rx(d), 0xb1u8 as i8
                            ; paddd Rx(d), Rx(SCRATCH)
                            ; movd eax, Rx(d)
                        );
                    }
                }
                dynasm!(self.asm ; .arch x64 ; movd Rx(d), eax);
                (d, true)
            }
            VecOp::Narrow { size, a, low } => {
                let a = self.reg(a);
                let low = low.map(|low| self.reg(low));
                let d = self.def(v);
                match size {
                    0 => dynasm!(self.asm
                        ; .arch x64
                        ; movdqa Rx(SCRATCH), Rx(a)
                        ; psllw Rx(SCRATCH), 8
                        ; psrlw Rx(SCRATCH), 8
                        ; packuswb Rx(SCRATCH), Rx(SCRATCH)
                    ),
                    1 => dynasm!(self.asm
                        ; .arch x64
                        ; movdqa Rx(SCRATCH), Rx(a)
                        ; pslld Rx(SCRATCH), 16
                        ; psrad Rx(SCRATCH), 16
                        ; packssdw Rx(SCRATCH), Rx(SCRATCH)
                    ),
                    _ => dynasm!(self.asm ; .arch x64 ; pshufd Rx(SCRATCH), Rx(a), 0x08),
                }
                match low {
                    Some(low) => {
                        self.copy(d, low);
                        dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(d), Rx(SCRATCH));
                    }
                    None => dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(SCRATCH)),
                }
                (d, true)
            }
            VecOp::Permute {
                transpose,
                odd,
                arrangement: Arrangement { size, q },
                n,
                m,
            } => {
                let (n, m) = (self.reg(n), self.reg(m));
                let d = self.def(v);
                self.copy(d, n);
                match (size, q) {
                    (3, _) if odd => dynasm!(self.asm ; .arch x64 ; punpckhqdq Rx(d), Rx(m)),
                    (3, _) => dynasm!(self.asm ; .arch x64 ; punpcklqdq Rx(d), Rx(m)),
                    (_, false) => {
                        dynasm!(self.asm ; .arch x64 ; unpcklps Rx(d), Rx(m));
                        if odd {
                            dynasm!(self.asm ; .arch x64 ; psrldq Rx(d), 8);
                        }
                    }
                    (_, true) => {
                        let lanes = if odd { 0xdd } else { 0x88 };
                        dynasm!(self.asm ; .arch x64 ; shufps Rx(d), Rx(m), lanes as i8);
                        if transpose {
                            dynasm!(self.asm ; .arch x64 ; pshufd Rx(d), Rx(d), 0xd8u8 as i8);
                        }
                    }
                }
                (d, q)
            }
            VecOp::MinMaxAcross(op, Arrangement { size, q }, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                let op = match op {
                    Binary::SMax => Sse::MaxS,
                    Binary::SMin => Sse::MinS,
                    Binary::UMax => Sse::MaxU,
                    _ => Sse::MinU,
                };
                self.copy(d, a);
                if q {
                    dynasm!(self.asm ; .arch x64 ; pshufd Rx(SCRATCH), Rx(d), 0x4e);
                    self.sse(op, size, d, SCRATCH);
                }
                if size == 1 {
                    dynasm!(self.asm ; .arch x64 ; pshuflw Rx(SCRATCH), Rx(d), 0x4e);
                    self.sse(op, size, d, SCRATCH);
                    dynasm!(self.asm ; .arch x64 ; pshuflw Rx(SCRATCH), Rx(d), 0xb1u8 as i8);
                    self.sse(op, size, d, SCRATCH);
                    dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d) ; movzx eax, ax);
                } else {
                    dynasm!(self.asm ; .arch x64 ; pshufd Rx(SCRATCH), Rx(d), 0xb1u8 as i8);
                    self.sse(op, size, d, SCRATCH);
                    dynasm!(self.asm ; .arch x64 ; movd eax, Rx(d));
                }
                dynasm!(self.asm ; .arch x64 ; movd Rx(d), eax);
                (d, true)
            }
            VecOp::Rev64(Arrangement { size, q }, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match size {
                    1 => dynasm!(self.asm
                        ; .arch x64
                        ; pshuflw Rx(d), Rx(a), 0x1b
                        ; pshufhw Rx(d), Rx(d), 0x1b
                    ),
                    _ => dynasm!(self.asm ; .arch x64 ; pshufd Rx(d), Rx(a), 0xb1u8 as i8),
                }
                (d, q)
            }
            VecOp::Cnt(q, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD CNT_MASKS.0.as_ptr() as i64
                    ; movdqa Rx(d), Rx(a)
                    ; movdqa Rx(SCRATCH), Rx(a)
                    ; psrlw Rx(SCRATCH), 1
                    ; pand Rx(SCRATCH), [rax]
                    ; psubb Rx(d), Rx(SCRATCH)
                    ; movdqa Rx(SCRATCH), Rx(d)
                    ; psrlw Rx(SCRATCH), 2
                    ; pand Rx(SCRATCH), [rax + 16]
                    ; pand Rx(d), [rax + 16]
                    ; paddb Rx(d), Rx(SCRATCH)
                    ; movdqa Rx(SCRATCH), Rx(d)
                    ; psrlw Rx(SCRATCH), 4
                    ; paddb Rx(d), Rx(SCRATCH)
                    ; pand Rx(d), [rax + 32]
                );
                (d, q)
            }
            VecOp::DupLane {
                size,
                index,
                vector,
            } => {
                let s = self.reg(vector);
                let d = self.def(v);
                let lanes = ((index & 3) * 0x55) as i8;
                match (size, index >= 4) {
                    (1, false) => dynasm!(self.asm
                        ; .arch x64
                        ; pshuflw Rx(d), Rx(s), lanes
                        ; punpcklqdq Rx(d), Rx(d)
                    ),
                    (1, true) => dynasm!(self.asm
                        ; .arch x64
                        ; pshufhw Rx(d), Rx(s), lanes
                        ; punpckhqdq Rx(d), Rx(d)
                    ),
                    _ => dynasm!(self.asm ; .arch x64 ; pshufd Rx(d), Rx(s), lanes),
                }
                return;
            }
            VecOp::Dup(Arrangement { size, q }, x) => {
                let x = self.reg(x);
                let d = self.def(v);
                match size {
                    0 => dynasm!(self.asm
                        ; .arch x64
                        ; mov eax, Rd(x)
                        ; movzx eax, al
                        ; imul eax, eax, 0x0101_0101
                        ; movd Rx(d), eax
                        ; pshufd Rx(d), Rx(d), 0
                    ),
                    1 => dynasm!(self.asm
                        ; .arch x64
                        ; mov eax, Rd(x)
                        ; movzx eax, ax
                        ; imul eax, eax, 0x0001_0001
                        ; movd Rx(d), eax
                        ; pshufd Rx(d), Rx(d), 0
                    ),
                    2 => dynasm!(self.asm
                        ; .arch x64
                        ; movd Rx(d), Rd(x)
                        ; pshufd Rx(d), Rx(d), 0
                    ),
                    _ => dynasm!(self.asm
                        ; .arch x64
                        ; movq Rx(d), Rq(x)
                        ; punpcklqdq Rx(d), Rx(d)
                    ),
                }
                (d, q)
            }
            VecOp::Insert {
                size,
                index,
                vector,
                value,
            } => {
                let (s, x) = (self.reg(vector), self.reg(value));
                let d = self.def(v);
                self.copy(d, s);
                let n = index as i8;
                match (size, index) {
                    (0, _) => dynasm!(self.asm ; .arch x64 ; pinsrb Rx(d), Rd(x), n),
                    (1, _) => dynasm!(self.asm ; .arch x64 ; pinsrw Rx(d), Rd(x), n),
                    (2, _) => dynasm!(self.asm ; .arch x64 ; pinsrd Rx(d), Rd(x), n),
                    (_, 0) => dynasm!(self.asm
                        ; .arch x64
                        ; movq Rx(SCRATCH), Rq(x)
                        ; movsd Rx(d), Rx(SCRATCH)
                    ),
                    _ => dynasm!(self.asm
                        ; .arch x64
                        ; movq Rx(SCRATCH), Rq(x)
                        ; punpcklqdq Rx(d), Rx(SCRATCH)
                    ),
                }
                return;
            }
            VecOp::Extract {
                size,
                index,
                signed,
                vector,
            } => {
                let s = self.reg(vector);
                let d = self.def(v);
                let n = index as i8;
                match (size, index) {
                    (0, _) => {
                        dynasm!(self.asm ; .arch x64 ; pextrw eax, Rx(s), n >> 1);
                        if index & 1 == 0 {
                            dynasm!(self.asm ; .arch x64 ; movzx eax, al);
                        } else {
                            dynasm!(self.asm ; .arch x64 ; shr eax, 8);
                        }
                    }
                    (1, _) => dynasm!(self.asm ; .arch x64 ; pextrw eax, Rx(s), n),
                    (2, 0) => dynasm!(self.asm ; .arch x64 ; movd eax, Rx(s)),
                    (2, _) => dynasm!(self.asm
                        ; .arch x64
                        ; pshufd Rx(SCRATCH), Rx(s), n
                        ; movd eax, Rx(SCRATCH)
                    ),
                    (_, 0) => dynasm!(self.asm ; .arch x64 ; movq rax, Rx(s)),
                    _ => dynasm!(self.asm
                        ; .arch x64
                        ; pshufd Rx(SCRATCH), Rx(s), 0xee_u8 as i8
                        ; movq rax, Rx(SCRATCH)
                    ),
                }
                match (signed, size) {
                    (None, _) => dynasm!(self.asm ; .arch x64 ; mov Rq(d), rax),
                    (Some(W::W32), 0) => dynasm!(self.asm ; .arch x64 ; movsx Rd(d), al),
                    (Some(W::W32), _) => dynasm!(self.asm ; .arch x64 ; movsx Rd(d), ax),
                    (Some(W::W64), 0) => dynasm!(self.asm ; .arch x64 ; movsx Rq(d), al),
                    (Some(W::W64), 1) => dynasm!(self.asm ; .arch x64 ; movsx Rq(d), ax),
                    (Some(W::W64), _) => dynasm!(self.asm ; .arch x64 ; movsxd Rq(d), eax),
                }
                return;
            }
        };
        if !q {
            dynasm!(self.asm ; .arch x64 ; movq Rx(d), Rx(d));
        }
    }
}
