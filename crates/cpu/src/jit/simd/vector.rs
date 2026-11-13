//! Advanced SIMD integer arithmetic, element moves and `LD1`/`ST1` (multiple structures), compiled
//! inline with SSE2, and SSE4.1/4.2 where the host has them. Each lowering gives the interpreter's
//! result bit for bit; encodings it does not take stay on `Op::CallSimd` or the interpreter.

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use dynasmrt::{DynasmApi, dynasm};

use super::super::ir::{Loc, Op, Ty, Value, W};
use super::super::translate::Translator;
use super::super::x64::Emitter;
use super::ir::SimdOp;
use crate::decoder::{Inst, PostIndex, Reg};

const SCRATCH: u8 = 15;

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
            } => {
                f(a);
                f(b);
            }
            VecOp::Shift(_, _, _, a)
            | VecOp::Dup(_, a)
            | VecOp::Cnt(_, a)
            | VecOp::Extract { vector: a, .. } => f(a),
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
    if word & 0xbf3f_fc00 == 0x0e20_9800 {
        if (size == 3 && !q) || !supported(t, Binary::CmEq, size) {
            return false;
        }
        let a = vec(t, rn);
        let zero = emit(t, VecOp::Imm(0, true));
        let r = emit(t, VecOp::Binary(Binary::CmEq, arrangement, a, zero));
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
        let (op, amount) = match (field(word, 11, 5), u) {
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
    false
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
    Mul,
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
            (Sse::Mul, 1) => dynasm!(self.asm ; .arch x64 ; pmullw Rx(d), Rx(s)),
            (Sse::Mul, _) => dynasm!(self.asm ; .arch x64 ; pmulld Rx(d), Rx(s)),
        }
    }

    fn invert(&mut self, d: u8) {
        dynasm!(self.asm
            ; .arch x64
            ; pcmpeqd Rx(SCRATCH), Rx(SCRATCH)
            ; pxor Rx(d), Rx(SCRATCH)
        );
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
                    _ => {
                        let op = match op {
                            Binary::Add => Sse::Add,
                            Binary::Sub => Sse::Sub,
                            Binary::And => Sse::And,
                            Binary::Orr => Sse::Or,
                            Binary::Eor => Sse::Xor,
                            Binary::CmEq => Sse::Eq,
                            Binary::CmGt => Sse::Gt,
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
                (d, q)
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
