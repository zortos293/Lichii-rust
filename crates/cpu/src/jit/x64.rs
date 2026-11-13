//! x86-64 backend: linear-scan register allocation and code emission for one IR block.
//!
//! Emitted code follows the System V ABI on every host OS. It runs with `r15` pointing at the
//! guest `ThreadContext`, `r13` at the `JitState`, and `r14` at the fastmem arena base. `rax`,
//! `rcx`, `rdx` and `xmm15` are scratch inside one op and never hold a value across ops. Each
//! memory access gets an out-of-line slow path that calls back into Rust; the fastmem fault
//! handler resumes a faulting access at that slow path.

#![allow(
    clippy::useless_conversion,
    reason = "dynasm! converts every register operand with into()"
)]

use std::mem::offset_of;

use dynasmrt::x64::X64Relocation;
use dynasmrt::{DynamicLabel, DynasmApi, DynasmLabelApi, VecAssembler, dynasm};

use lichii_memory::ReservationSlot;

use super::JitState;
use super::ir::{Block, Loc, Op, Terminal, Ty, Value, W};
use crate::decoder::AtomicOp;
use crate::interpreter::condition_holds;
use crate::{ThreadContext, TscScale};

type Asm = VecAssembler<X64Relocation>;

pub(super) const EXIT_CONTINUE: u64 = 0;
pub(super) const EXIT_FAULT: u64 = 2;
/// The `info` operation code of a `CAS` slow path; `LD<op>` uses the `AtomicOp` discriminant.
pub(super) const ATOMIC_CAS: i32 = 15;

/// Allocatable general-purpose registers, callee-saved ones first so slow paths save less.
const GPRS: [u8; 9] = [3, 5, 12, 6, 7, 8, 9, 10, 11];
const RBX: u8 = 3;
/// `rsi rdi r8-r11`.
const CALLER_SAVED: u16 = 0b0000_1111_1100_0000;
const XMMS: u8 = 15;
pub(super) const SPILL_SLOTS: usize = 256;

/// Where translated code finds memory and its slow paths.
pub(super) struct Target {
    /// `Fastmem::size()`, or `None` to send every access to the slow path.
    pub arena_size: Option<u64>,
    pub read: usize,
    pub write: usize,
    pub atomic: usize,
    pub exclusive: usize,
    pub zero: usize,
    pub ticks: usize,
    /// `simd::call`, which runs `Op::CallSimd`.
    pub simd: usize,
    /// The clock's `TscScale`, which counter reads use inline instead of calling `ticks`.
    pub tsc_scale: Option<*const TscScale>,
    /// The host has `cmpxchg16b`.
    pub cmpxchg16b: bool,
}

/// Machine code for one block and its fastmem access sites as `(access, slow path)` offsets.
pub(super) struct Compiled {
    pub code: Vec<u8>,
    pub sites: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Gpr = 0,
    Xmm = 1,
}

#[derive(Clone, Copy)]
enum Src {
    Reg(u8),
    Imm(i32),
}

/// The out-of-line slow path of one memory access: calls
/// `helper(state, addr, pc, info) -> faulted` with the operands in `scratch` and `scratch2`.
struct Stub {
    label: DynamicLabel,
    resume: DynamicLabel,
    sites: Vec<u32>,
    addr: u8,
    pc: u64,
    helper: usize,
    info: i32,
    vec: bool,
    dst: Option<u8>,
    /// Up to four 64-bit operands, or one vector in the first, stored at `scratch` then `scratch2`.
    values: [Option<Src>; 4],
    saved: [u16; 2],
    /// Registers a `cmpxchg16b` borrowed, restored from `cx16_save` before anything else.
    borrowed: [Option<u8>; 2],
}

const fn ctx(offset: usize) -> i32 {
    offset as i32
}

const CTX_X: i32 = ctx(offset_of!(ThreadContext, x));
const CTX_SP: i32 = ctx(offset_of!(ThreadContext, sp));
const CTX_PC: i32 = ctx(offset_of!(ThreadContext, pc));
const CTX_PSTATE: i32 = ctx(offset_of!(ThreadContext, pstate));
const CTX_FPCR: i32 = ctx(offset_of!(ThreadContext, fpcr));
const CTX_FPSR: i32 = ctx(offset_of!(ThreadContext, fpsr));
const CTX_V: i32 = ctx(offset_of!(ThreadContext, v));
const CTX_TPIDR: i32 = ctx(offset_of!(ThreadContext, tpidr_el0));
const CTX_TPIDRRO: i32 = ctx(offset_of!(ThreadContext, tpidrro_el0));
const ST_SCRATCH: i32 = ctx(offset_of!(JitState, scratch));
const ST_SCRATCH2: i32 = ctx(offset_of!(JitState, scratch2));
const ST_SAVE_GPR: i32 = ctx(offset_of!(JitState, save_gpr));
const TSC_MUL: i32 = offset_of!(TscScale, mul) as i32;
const TSC_SHIFT: i32 = offset_of!(TscScale, shift) as i32;
const TSC_OFFSET: i32 = offset_of!(TscScale, offset) as i32;
const ST_HIGH: i32 = ctx(offset_of!(JitState, high));
const ST_CX16_SAVE: i32 = ctx(offset_of!(JitState, cx16_save));
const ST_SAVE_XMM: i32 = ctx(offset_of!(JitState, save_xmm));
const ST_SPILL: i32 = ctx(offset_of!(JitState, spill));
const ST_LINK: i32 = ctx(offset_of!(JitState, link));
const ST_TABLE: i32 = ctx(offset_of!(JitState, table));
const ST_RESERVATION: i32 = ctx(offset_of!(JitState, reservation));
const RES_VALUE_LO: i32 = offset_of!(ReservationSlot, value_lo) as i32;
const RES_VALUE_HI: i32 = offset_of!(ReservationSlot, value_hi) as i32;
const RES_ADDR: i32 = offset_of!(ReservationSlot, addr) as i32;
const RES_SIZE: i32 = offset_of!(ReservationSlot, size) as i32;
const ST_INTERRUPT: i32 = ctx(offset_of!(JitState, interrupt));
/// Entries in the direct-mapped `guest pc -> host code` table indirect branches probe.
pub(super) const TABLE_BITS: u32 = 16;

fn loc_offset(loc: Loc) -> i32 {
    match loc {
        Loc::X(n) => CTX_X + 8 * n as i32,
        Loc::Sp => CTX_SP,
        Loc::Nzcv => CTX_PSTATE,
        Loc::V(n) => CTX_V + 16 * n as i32,
        Loc::Fpcr => CTX_FPCR,
        Loc::Fpsr => CTX_FPSR,
        Loc::Tpidr => CTX_TPIDR,
        Loc::Tpidrro => CTX_TPIDRRO,
    }
}

/// Bit `nzcv` of the result is whether `cond` holds for those flags.
fn cond_table(cond: u8) -> i32 {
    (0..16).fold(0, |mask, flags| {
        mask | (condition_holds(cond, flags) as i32) << flags
    })
}

macro_rules! alu {
    ($a:expr, $op:ident, $w:expr, $d:expr, $src:expr) => {
        match ($w, $src) {
            (W::W32, Src::Reg(s)) => dynasm!($a ; .arch x64 ; $op Rd($d), Rd(s)),
            (W::W64, Src::Reg(s)) => dynasm!($a ; .arch x64 ; $op Rq($d), Rq(s)),
            (W::W32, Src::Imm(i)) => dynasm!($a ; .arch x64 ; $op Rd($d), i),
            (W::W64, Src::Imm(i)) => dynasm!($a ; .arch x64 ; $op Rq($d), i),
        }
    };
}

macro_rules! shift {
    ($a:expr, $op:ident, $w:expr, $d:expr, $src:expr) => {
        match ($w, $src) {
            (W::W32, Src::Reg(s)) => dynasm!($a ; .arch x64 ; mov ecx, Rd(s) ; $op Rd($d), cl),
            (W::W64, Src::Reg(s)) => dynasm!($a ; .arch x64 ; mov ecx, Rd(s) ; $op Rq($d), cl),
            (W::W32, Src::Imm(i)) => dynasm!($a ; .arch x64 ; $op Rd($d), i as i8),
            (W::W64, Src::Imm(i)) => dynasm!($a ; .arch x64 ; $op Rq($d), i as i8),
        }
    };
}

fn mov(a: &mut Asm, w: W, d: u8, src: Src) {
    match (w, src) {
        (W::W32, Src::Reg(s)) => dynasm!(a ; .arch x64 ; mov Rd(d), Rd(s)),
        (W::W64, Src::Reg(s)) => dynasm!(a ; .arch x64 ; mov Rq(d), Rq(s)),
        (W::W32, Src::Imm(i)) => dynasm!(a ; .arch x64 ; mov Rd(d), i),
        (W::W64, Src::Imm(i)) => dynasm!(a ; .arch x64 ; mov Rq(d), i),
    }
}

fn mov_imm64(a: &mut Asm, d: u8, value: u64) {
    if value as i64 == value as i32 as i64 {
        dynasm!(a ; .arch x64 ; mov Rq(d), value as i32);
    } else {
        dynasm!(a ; .arch x64 ; mov Rq(d), QWORD value as i64);
    }
}

/// Packs the host flags of the preceding add or subtract into NZCV in `eax`. `eax` and `ecx`
/// must be zero. A64 subtraction sets C when no borrow happens, the inverse of x86's CF.
fn flags_to_nzcv(a: &mut Asm, inverted_carry: bool) {
    dynasm!(a ; .arch x64 ; seto al);
    if inverted_carry {
        dynasm!(a ; .arch x64 ; setnc cl);
    } else {
        dynasm!(a ; .arch x64 ; setc cl);
    }
    dynasm!(a
        ; .arch x64
        ; lea eax, [rax + rcx * 2]
        ; setz cl
        ; lea eax, [rax + rcx * 4]
        ; sets cl
        ; lea eax, [rax + rcx * 8]
    );
}

struct Emitter<'a> {
    asm: Asm,
    block: &'a Block,
    target: &'a Target,
    holder: [[Option<u32>; 16]; 2],
    reg: Vec<Option<u8>>,
    slot: Vec<Option<u16>>,
    free_slots: Vec<u16>,
    next_slot: u16,
    locked: [u16; 2],
    last_use: Vec<u32>,
    stubs: Vec<Stub>,
    fault_exit: DynamicLabel,
}

impl Emitter<'_> {
    fn class(&self, v: Value) -> Class {
        match self.block.ops[v.index()].ty() {
            Ty::U128 => Class::Xmm,
            _ => Class::Gpr,
        }
    }

    fn candidates(class: Class) -> impl Iterator<Item = u8> {
        let gprs = GPRS.iter().copied().filter(move |_| class == Class::Gpr);
        let xmms = (0..XMMS).filter(move |_| class == Class::Xmm);
        gprs.chain(xmms)
    }

    /// A free register of `class`, spilling the value whose last use is furthest away if none is.
    fn alloc(&mut self, class: Class) -> u8 {
        let c = class as usize;
        let unlocked = |r: u8, locked: u16| locked & (1 << r) == 0;
        if let Some(r) = Self::candidates(class)
            .find(|&r| self.holder[c][r as usize].is_none() && unlocked(r, self.locked[c]))
        {
            return r;
        }
        let victim = Self::candidates(class)
            .filter(|&r| unlocked(r, self.locked[c]))
            .max_by_key(|&r| self.last_use[self.holder[c][r as usize].unwrap() as usize])
            .expect("every register of a class is locked by one op");
        self.spill(class, victim);
        victim
    }

    fn spill(&mut self, class: Class, r: u8) {
        let v = self.holder[class as usize][r as usize]
            .take()
            .expect("spilling an empty register");
        self.reg[v as usize] = None;
        if matches!(self.block.ops[v as usize], Op::Const(_)) || self.slot[v as usize].is_some() {
            return;
        }
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            self.next_slot += 1;
            self.next_slot - 1
        });
        assert!(
            (slot as usize) < SPILL_SLOTS,
            "a block needs too many spill slots"
        );
        self.slot[v as usize] = Some(slot);
        let offset = ST_SPILL + 16 * slot as i32;
        match class {
            Class::Gpr => dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], Rq(r)),
            Class::Xmm => dynasm!(self.asm ; .arch x64 ; movdqu [r13 + offset], Rx(r)),
        }
    }

    fn take(&mut self, v: Value, class: Class, r: u8) {
        self.holder[class as usize][r as usize] = Some(v.0);
        self.reg[v.index()] = Some(r);
        self.locked[class as usize] |= 1 << r;
    }

    /// `v` in a register, locked until the current op is finished.
    fn reg(&mut self, v: Value) -> u8 {
        let class = self.class(v);
        if let Some(r) = self.reg[v.index()] {
            self.locked[class as usize] |= 1 << r;
            return r;
        }
        let r = self.alloc(class);
        if let Op::Const(value) = self.block.ops[v.index()] {
            mov_imm64(&mut self.asm, r, value);
        } else {
            let slot = self.slot[v.index()].expect("a value is used before it is defined");
            let offset = ST_SPILL + 16 * slot as i32;
            match class {
                Class::Gpr => dynasm!(self.asm ; .arch x64 ; mov Rq(r), QWORD [r13 + offset]),
                Class::Xmm => dynasm!(self.asm ; .arch x64 ; movdqu Rx(r), [r13 + offset]),
            }
        }
        self.take(v, class, r);
        r
    }

    /// `v` as an immediate when it is a constant that fits one, else in a register.
    fn src(&mut self, v: Value, w: W) -> Src {
        match (self.block.ops[v.index()], w) {
            (Op::Const(c), W::W32) => Src::Imm(c as u32 as i32),
            (Op::Const(c), W::W64) if c as i64 == c as i32 as i64 => Src::Imm(c as i32),
            _ => Src::Reg(self.reg(v)),
        }
    }

    fn def(&mut self, v: Value) -> u8 {
        let class = self.class(v);
        let r = self.alloc(class);
        self.take(v, class, r);
        r
    }

    fn release(&mut self, v: Value) {
        if let Some(r) = self.reg[v.index()].take() {
            self.holder[self.class(v) as usize][r as usize] = None;
        }
        if let Some(slot) = self.slot[v.index()].take() {
            self.free_slots.push(slot);
        }
    }

    /// Frees every operand whose last use was op `i`, and the op's own value if it is unused.
    fn finish(&mut self, i: usize, op: &Op) {
        for v in op.args().chain([Value(i as u32)]) {
            if self.last_use[v.index()] as usize <= i {
                self.release(v);
            }
        }
        self.locked = [0; 2];
    }

    fn emit(&mut self, i: usize) {
        let v = Value(i as u32);
        let op = self.block.ops[i];
        match op {
            Op::Nop | Op::Const(_) => return,
            Op::Get(loc) => {
                let d = self.def(v);
                let offset = loc_offset(loc);
                match loc {
                    Loc::V(_) => dynasm!(self.asm ; .arch x64 ; movdqu Rx(d), [r15 + offset]),
                    Loc::Nzcv => dynasm!(self.asm
                        ; .arch x64
                        ; mov Rd(d), DWORD [r15 + offset]
                        ; shr Rd(d), 28
                    ),
                    Loc::Fpcr | Loc::Fpsr => {
                        dynasm!(self.asm ; .arch x64 ; mov Rd(d), DWORD [r15 + offset])
                    }
                    _ => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r15 + offset]),
                }
            }
            Op::Set(loc, x) => self.set(loc, x),
            Op::Add(w, a, b)
            | Op::Sub(w, a, b)
            | Op::And(w, a, b)
            | Op::Or(w, a, b)
            | Op::Xor(w, a, b) => {
                let b = self.src(b, w);
                let a = self.src(a, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, a);
                match op {
                    Op::Add(..) => alu!(self.asm, add, w, d, b),
                    Op::Sub(..) => alu!(self.asm, sub, w, d, b),
                    Op::And(..) => alu!(self.asm, and, w, d, b),
                    Op::Or(..) => alu!(self.asm, or, w, d, b),
                    _ => alu!(self.asm, xor, w, d, b),
                }
            }
            Op::Adc(w, a, b, f) | Op::Sbc(w, a, b, f) => {
                let f = self.reg(f);
                let b = self.src(b, w);
                let a = self.src(a, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, a);
                dynasm!(self.asm ; .arch x64 ; bt Rd(f), 1);
                if matches!(op, Op::Adc(..)) {
                    alu!(self.asm, adc, w, d, b);
                } else {
                    dynasm!(self.asm ; .arch x64 ; cmc);
                    alu!(self.asm, sbb, w, d, b);
                }
            }
            Op::AddFlags(w, a, b)
            | Op::SubFlags(w, a, b)
            | Op::AdcFlags(w, a, b, _)
            | Op::SbcFlags(w, a, b, _) => {
                let carry = match op {
                    Op::AdcFlags(_, _, _, f) | Op::SbcFlags(_, _, _, f) => Some(self.reg(f)),
                    _ => None,
                };
                let b = self.src(b, w);
                let a = self.src(a, w);
                let d = self.def(v);
                dynasm!(self.asm ; .arch x64 ; xor eax, eax ; xor ecx, ecx);
                mov(&mut self.asm, w, 2, a);
                if let Some(f) = carry {
                    dynasm!(self.asm ; .arch x64 ; bt Rd(f), 1);
                }
                match op {
                    Op::AddFlags(..) => alu!(self.asm, add, w, 2, b),
                    Op::SubFlags(..) => alu!(self.asm, sub, w, 2, b),
                    Op::AdcFlags(..) => alu!(self.asm, adc, w, 2, b),
                    _ => {
                        dynasm!(self.asm ; .arch x64 ; cmc);
                        alu!(self.asm, sbb, w, 2, b);
                    }
                }
                flags_to_nzcv(
                    &mut self.asm,
                    matches!(op, Op::SubFlags(..) | Op::SbcFlags(..)),
                );
                dynasm!(self.asm ; .arch x64 ; mov Rd(d), eax);
            }
            Op::LogicFlags(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                dynasm!(self.asm ; .arch x64 ; xor eax, eax ; xor ecx, ecx);
                alu!(self.asm, test, w, a, Src::Reg(a));
                dynasm!(self.asm
                    ; .arch x64
                    ; setz cl
                    ; lea eax, [rcx * 4]
                    ; sets cl
                    ; lea eax, [rax + rcx * 8]
                    ; mov Rd(d), eax
                );
            }
            Op::Not(w, a) => {
                let a = self.src(a, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, a);
                match w {
                    W::W32 => dynasm!(self.asm ; .arch x64 ; not Rd(d)),
                    W::W64 => dynasm!(self.asm ; .arch x64 ; not Rq(d)),
                }
            }
            Op::Lsl(w, a, b) | Op::Lsr(w, a, b) | Op::Asr(w, a, b) | Op::Ror(w, a, b) => {
                let amount = match self.block.constant(b) {
                    Some(c) => Src::Imm((c & (w.bits() as u64 - 1)) as i32),
                    None => Src::Reg(self.reg(b)),
                };
                let a = self.src(a, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, a);
                match op {
                    Op::Lsl(..) => shift!(self.asm, shl, w, d, amount),
                    Op::Lsr(..) => shift!(self.asm, shr, w, d, amount),
                    Op::Asr(..) => shift!(self.asm, sar, w, d, amount),
                    _ => shift!(self.asm, ror, w, d, amount),
                }
            }
            Op::ZeroExt(size, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), Rb(a)),
                    1 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), Rw(a)),
                    2 => dynasm!(self.asm ; .arch x64 ; mov Rd(d), Rd(a)),
                    _ => dynasm!(self.asm ; .arch x64 ; mov Rq(d), Rq(a)),
                }
            }
            Op::SignExt(w, size, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match (w, size) {
                    (W::W32, 0) => dynasm!(self.asm ; .arch x64 ; movsx Rd(d), Rb(a)),
                    (W::W32, 1) => dynasm!(self.asm ; .arch x64 ; movsx Rd(d), Rw(a)),
                    (W::W32, _) => dynasm!(self.asm ; .arch x64 ; mov Rd(d), Rd(a)),
                    (W::W64, 0) => dynasm!(self.asm ; .arch x64 ; movsx Rq(d), Rb(a)),
                    (W::W64, 1) => dynasm!(self.asm ; .arch x64 ; movsx Rq(d), Rw(a)),
                    (W::W64, 2) => dynasm!(self.asm ; .arch x64 ; movsxd Rq(d), Rd(a)),
                    (W::W64, _) => dynasm!(self.asm ; .arch x64 ; mov Rq(d), Rq(a)),
                }
            }
            Op::Mul(w, a, b) => {
                let b = self.reg(b);
                let a = self.src(a, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, a);
                match w {
                    W::W32 => dynasm!(self.asm ; .arch x64 ; imul Rd(d), Rd(b)),
                    W::W64 => dynasm!(self.asm ; .arch x64 ; imul Rq(d), Rq(b)),
                }
            }
            Op::UMulH(a, b) | Op::SMulH(a, b) => {
                let b = self.reg(b);
                let a = self.src(a, W::W64);
                let d = self.def(v);
                mov(&mut self.asm, W::W64, 0, a);
                if matches!(op, Op::UMulH(..)) {
                    dynasm!(self.asm ; .arch x64 ; mul Rq(b));
                } else {
                    dynasm!(self.asm ; .arch x64 ; imul Rq(b));
                }
                dynasm!(self.asm ; .arch x64 ; mov Rq(d), rdx);
            }
            Op::UDiv(w, a, b) | Op::SDiv(w, a, b) => {
                let b = self.reg(b);
                let a = self.src(a, w);
                let d = self.def(v);
                let (zero, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
                mov(&mut self.asm, w, 0, a);
                alu!(self.asm, test, w, b, Src::Reg(b));
                dynasm!(self.asm ; .arch x64 ; jz =>zero);
                match (op, w) {
                    (Op::UDiv(..), W::W32) => {
                        dynasm!(self.asm ; .arch x64 ; xor edx, edx ; div Rd(b))
                    }
                    (Op::UDiv(..), W::W64) => {
                        dynasm!(self.asm ; .arch x64 ; xor edx, edx ; div Rq(b))
                    }
                    (_, W::W32) => {
                        let normal = self.asm.new_dynamic_label();
                        dynasm!(self.asm
                        ; .arch x64
                        ; cmp Rd(b), -1
                        ; jne =>normal
                        ; neg eax
                        ; jmp =>done
                        ; =>normal
                        ; cdq
                        ; idiv Rd(b)
                        )
                    }
                    (_, W::W64) => {
                        let normal = self.asm.new_dynamic_label();
                        dynasm!(self.asm
                        ; .arch x64
                        ; cmp Rq(b), -1
                        ; jne =>normal
                        ; neg rax
                        ; jmp =>done
                        ; =>normal
                        ; cqo
                        ; idiv Rq(b)
                        )
                    }
                }
                dynasm!(self.asm
                    ; .arch x64
                    ; jmp =>done
                    ; =>zero
                    ; xor eax, eax
                    ; =>done
                    ; mov Rq(d), rax
                );
            }
            Op::Clz(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match w {
                    W::W32 => dynasm!(self.asm
                        ; .arch x64
                        ; bsr eax, Rd(a)
                        ; mov ecx, 63
                        ; cmovz eax, ecx
                        ; xor eax, 31
                    ),
                    W::W64 => dynasm!(self.asm
                        ; .arch x64
                        ; bsr rax, Rq(a)
                        ; mov ecx, 127
                        ; cmovz eax, ecx
                        ; xor eax, 63
                    ),
                }
                dynasm!(self.asm ; .arch x64 ; mov Rd(d), eax);
            }
            Op::Rbit(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                mov(&mut self.asm, w, d, Src::Reg(a));
                match w {
                    W::W32 => dynasm!(self.asm ; .arch x64 ; bswap Rd(d)),
                    W::W64 => dynasm!(self.asm ; .arch x64 ; bswap Rq(d)),
                }
                for (mask, bits) in [
                    (0x0f0f_0f0f_0f0f_0f0f_u64, 4i8),
                    (0x3333_3333_3333_3333, 2),
                    (0x5555_5555_5555_5555, 1),
                ] {
                    match w {
                        W::W32 => dynasm!(self.asm
                            ; .arch x64
                            ; mov eax, Rd(d)
                            ; shr eax, bits
                            ; and eax, mask as u32 as i32
                            ; and Rd(d), mask as u32 as i32
                            ; shl Rd(d), bits
                            ; or Rd(d), eax
                        ),
                        W::W64 => dynasm!(self.asm
                            ; .arch x64
                            ; mov rax, Rq(d)
                            ; shr rax, bits
                            ; mov rcx, QWORD mask as i64
                            ; and rax, rcx
                            ; and Rq(d), rcx
                            ; shl Rq(d), bits
                            ; or Rq(d), rax
                        ),
                    }
                }
            }
            Op::Rev(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                mov(&mut self.asm, w, d, Src::Reg(a));
                match w {
                    W::W32 => dynasm!(self.asm ; .arch x64 ; bswap Rd(d)),
                    W::W64 => dynasm!(self.asm ; .arch x64 ; bswap Rq(d)),
                }
            }
            Op::Rev16(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                match w {
                    W::W32 => dynasm!(self.asm
                        ; .arch x64
                        ; mov eax, Rd(a)
                        ; shr eax, 8
                        ; and eax, 0x00ff_00ff
                        ; mov Rd(d), Rd(a)
                        ; and Rd(d), 0x00ff_00ff
                        ; shl Rd(d), 8
                        ; or Rd(d), eax
                    ),
                    W::W64 => dynasm!(self.asm
                        ; .arch x64
                        ; mov rax, Rq(a)
                        ; shr rax, 8
                        ; mov rcx, QWORD 0x00ff_00ff_00ff_00ff
                        ; and rax, rcx
                        ; mov Rq(d), Rq(a)
                        ; and Rq(d), rcx
                        ; shl Rq(d), 8
                        ; or Rq(d), rax
                    ),
                }
            }
            Op::Rev32(a) => {
                let a = self.reg(a);
                let d = self.def(v);
                dynasm!(self.asm
                    ; .arch x64
                    ; mov Rq(d), Rq(a)
                    ; bswap Rq(d)
                    ; rol Rq(d), 32
                );
            }
            Op::Select(w, c, a, b) => {
                let c = self.reg(c);
                let a = self.reg(a);
                let b = self.src(b, w);
                let d = self.def(v);
                mov(&mut self.asm, w, d, b);
                dynasm!(self.asm ; .arch x64 ; test Rd(c), Rd(c));
                match w {
                    W::W32 => dynasm!(self.asm ; .arch x64 ; cmovnz Rd(d), Rd(a)),
                    W::W64 => dynasm!(self.asm ; .arch x64 ; cmovnz Rq(d), Rq(a)),
                }
            }
            Op::TestCond(cond, f) => {
                let f = self.reg(f);
                let d = self.def(v);
                let table = cond_table(cond);
                dynasm!(self.asm
                    ; .arch x64
                    ; mov eax, table
                    ; bt eax, Rd(f)
                    ; setc al
                    ; movzx Rd(d), al
                );
            }
            Op::IsZero(w, a) => {
                let a = self.reg(a);
                let d = self.def(v);
                dynasm!(self.asm ; .arch x64 ; xor eax, eax);
                alu!(self.asm, test, w, a, Src::Reg(a));
                dynasm!(self.asm ; .arch x64 ; setz al ; mov Rd(d), eax);
            }
            Op::TestBit(a, bit) => {
                let a = self.reg(a);
                let d = self.def(v);
                dynasm!(self.asm
                    ; .arch x64
                    ; xor eax, eax
                    ; bt Rq(a), bit as i8
                    ; setc al
                    ; mov Rd(d), eax
                );
            }
            Op::Ticks => match self.target.tsc_scale {
                Some(scale) => {
                    let d = self.def(v);
                    dynasm!(self.asm
                        ; .arch x64
                        ; rdtsc
                        ; shl rdx, 32
                        ; or rax, rdx
                        ; mov rcx, QWORD scale as i64
                        ; mul QWORD [rcx + TSC_MUL]
                        ; mov rcx, QWORD [rcx + TSC_SHIFT]
                        ; shrd rax, rdx, cl
                        ; mov rcx, QWORD scale as i64
                        ; add rax, QWORD [rcx + TSC_OFFSET]
                        ; mov Rq(d), rax
                    );
                }
                None => self.call(i, self.target.ticks),
            },
            Op::High => {
                let d = self.def(v);
                dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r13 + ST_HIGH]);
            }
            Op::Fence => dynasm!(self.asm ; .arch x64 ; mfence),
            Op::CallSimd { word, pc } => self.call_simd(i, word, pc),
            _ => self.memory(i, op),
        }
        self.finish(i, &op);
    }

    fn set(&mut self, loc: Loc, x: Value) {
        let offset = loc_offset(loc);
        match loc {
            Loc::V(_) => {
                let r = self.reg(x);
                dynasm!(self.asm ; .arch x64 ; movdqu [r15 + offset], Rx(r));
            }
            Loc::Nzcv => {
                dynasm!(self.asm
                    ; .arch x64
                    ; mov eax, DWORD [r15 + offset]
                    ; and eax, 0x0fff_ffff
                );
                match self.block.constant(x) {
                    Some(c) => dynasm!(self.asm ; .arch x64 ; or eax, ((c as u32) << 28) as i32),
                    None => {
                        let r = self.reg(x);
                        dynasm!(self.asm
                            ; .arch x64
                            ; mov ecx, Rd(r)
                            ; shl ecx, 28
                            ; or eax, ecx
                        );
                    }
                }
                dynasm!(self.asm ; .arch x64 ; mov DWORD [r15 + offset], eax);
            }
            Loc::Fpcr | Loc::Fpsr => match self.src(x, W::W32) {
                Src::Imm(i) => dynasm!(self.asm ; .arch x64 ; mov DWORD [r15 + offset], i),
                Src::Reg(r) => dynasm!(self.asm ; .arch x64 ; mov DWORD [r15 + offset], Rd(r)),
            },
            _ => match self.src(x, W::W64) {
                Src::Imm(i) => dynasm!(self.asm ; .arch x64 ; mov QWORD [r15 + offset], i),
                Src::Reg(r) => dynasm!(self.asm ; .arch x64 ; mov QWORD [r15 + offset], Rq(r)),
            },
        }
    }

    /// Caller-saved registers holding a value that is still needed after op `i`.
    fn live_across(&self, i: usize) -> [u16; 2] {
        let mut saved = [0u16; 2];
        for class in [Class::Gpr, Class::Xmm] {
            for r in 0..16u8 {
                let Some(held) = self.holder[class as usize][r as usize] else {
                    continue;
                };
                let caller_saved = class == Class::Xmm || CALLER_SAVED & (1 << r) != 0;
                if caller_saved && self.last_use[held as usize] as usize > i && held != i as u32 {
                    saved[class as usize] |= 1 << r;
                }
            }
        }
        saved
    }

    fn save(&mut self, saved: [u16; 2]) {
        for r in 0..16u8 {
            if saved[0] & (1 << r) != 0 {
                let offset = ST_SAVE_GPR + 8 * r as i32;
                dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], Rq(r));
            }
            if saved[1] & (1 << r) != 0 {
                let offset = ST_SAVE_XMM + 16 * r as i32;
                dynasm!(self.asm ; .arch x64 ; movdqu [r13 + offset], Rx(r));
            }
        }
    }

    fn restore(&mut self, saved: [u16; 2]) {
        for r in 0..16u8 {
            if saved[0] & (1 << r) != 0 {
                let offset = ST_SAVE_GPR + 8 * r as i32;
                dynasm!(self.asm ; .arch x64 ; mov Rq(r), QWORD [r13 + offset]);
            }
            if saved[1] & (1 << r) != 0 {
                let offset = ST_SAVE_XMM + 16 * r as i32;
                dynasm!(self.asm ; .arch x64 ; movdqu Rx(r), [r13 + offset]);
            }
        }
    }

    /// Calls `helper(state)` from the main path and puts its `u64` result in op `i`'s register.
    fn call(&mut self, i: usize, helper: usize) {
        let d = self.def(Value(i as u32));
        let saved = self.live_across(i);
        self.save(saved);
        dynasm!(self.asm
            ; .arch x64
            ; mov rdi, r13
            ; mov rax, QWORD helper as i64
            ; sub rsp, 8
            ; call rax
            ; add rsp, 8
        );
        self.restore(saved);
        dynasm!(self.asm ; .arch x64 ; mov Rq(d), rax);
    }

    /// Calls `simd(ctx, word, state, pc) -> undefined` from the main path, leaving through the fault
    /// exit when the instruction is undefined.
    fn call_simd(&mut self, i: usize, word: u32, pc: u64) {
        let saved = self.live_across(i);
        self.save(saved);
        dynasm!(self.asm
            ; .arch x64
            ; mov rdi, r15
            ; mov esi, word as i32
            ; mov rdx, r13
            ; mov rcx, QWORD pc as i64
            ; mov rax, QWORD self.target.simd as i64
            ; sub rsp, 8
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jnz =>self.fault_exit
        );
        self.restore(saved);
    }

    fn memory(&mut self, i: usize, op: Op) {
        let (size, addr, pc) = match op {
            Op::Load { size, addr, pc }
            | Op::LoadVec { size, addr, pc }
            | Op::Store { size, addr, pc, .. }
            | Op::StoreVec { size, addr, pc, .. }
            | Op::AtomicRmw { size, addr, pc, .. }
            | Op::Cas { size, addr, pc, .. }
            | Op::LoadExclusive { size, addr, pc }
            | Op::StoreExclusive { size, addr, pc, .. } => (size, addr, pc),
            Op::StoreExclusive128 { addr, pc, .. } | Op::Cas128 { addr, pc, .. } => (4, addr, pc),
            Op::ZeroBlock { addr, pc } => (6, addr, pc),
            _ => unreachable!(),
        };
        let vec = matches!(op, Op::LoadVec { .. } | Op::StoreVec { .. });
        let a = self.reg(addr);
        let stored = match op {
            Op::Store { value, .. } => Some(self.src(value, W::W64)),
            _ => None,
        };
        let mut reg = |v: Value| Some(Src::Reg(self.reg(v)));
        let values = match op {
            Op::Store { .. } => [stored, None, None, None],
            Op::StoreVec { value, .. }
            | Op::AtomicRmw { value, .. }
            | Op::StoreExclusive { value, .. } => [reg(value), None, None, None],
            Op::Cas { expected, new, .. } => [reg(expected), None, reg(new), None],
            Op::StoreExclusive128 { low, high, .. } => [reg(low), reg(high), None, None],
            Op::Cas128 {
                expected_low,
                expected_high,
                new_low,
                new_high,
                ..
            } => [
                reg(expected_low),
                reg(expected_high),
                reg(new_low),
                reg(new_high),
            ],
            _ => [None; 4],
        };
        let borrowed = if matches!(op, Op::StoreExclusive128 { .. } | Op::Cas128 { .. }) {
            self.borrow_for_cx16(a, &values)
        } else {
            [None; 2]
        };
        let dst = (op.ty() != Ty::Void).then(|| self.def(Value(i as u32)));
        let label = self.asm.new_dynamic_label();
        let resume = self.asm.new_dynamic_label();
        let sites = match self.target.arena_size {
            Some(arena) => {
                let aligned = matches!(
                    op,
                    Op::AtomicRmw { .. }
                        | Op::Cas { .. }
                        | Op::LoadExclusive { .. }
                        | Op::StoreExclusive { .. }
                        | Op::StoreExclusive128 { .. }
                        | Op::Cas128 { .. }
                );
                if aligned && size > 0 {
                    let misaligned = (1i32 << size) - 1;
                    dynasm!(self.asm ; .arch x64 ; test Rd(a), misaligned ; jnz =>label);
                }
                let limit = (arena - (1u64 << size)) as i64;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD limit
                    ; cmp Rq(a), rax
                    ; ja =>label
                );
                self.access(op, a, dst, values, borrowed)
            }
            None => {
                dynasm!(self.asm ; .arch x64 ; jmp =>label);
                Vec::new()
            }
        };
        dynasm!(self.asm ; .arch x64 ; =>resume);
        let (helper, info) = match op {
            Op::Load { .. } | Op::LoadVec { .. } => (self.target.read, size as i32),
            Op::Store { .. } | Op::StoreVec { .. } => (self.target.write, size as i32),
            Op::AtomicRmw { op, .. } => (self.target.atomic, size as i32 | (op as i32) << 8),
            Op::Cas { .. } => (self.target.atomic, size as i32 | ATOMIC_CAS << 8),
            Op::LoadExclusive { .. } => (self.target.exclusive, size as i32),
            Op::StoreExclusive { .. } | Op::StoreExclusive128 { .. } => {
                (self.target.exclusive, size as i32 | 1 << 8)
            }
            Op::Cas128 { .. } => (self.target.atomic, size as i32 | ATOMIC_CAS << 8),
            _ => (self.target.zero, size as i32),
        };
        let saved = self.live_across(i);
        self.stubs.push(Stub {
            label,
            resume,
            sites,
            addr: a,
            pc,
            helper,
            info,
            vec,
            dst,
            values,
            saved,
            borrowed,
        });
    }

    /// Saves `rbx`, which `cmpxchg16b` needs for the low half of the new value, and when the
    /// address is in `rbx` also a second register to hold it: one that holds no operand.
    fn borrow_for_cx16(&mut self, a: u8, values: &[Option<Src>; 4]) -> [Option<u8>; 2] {
        let operands: Vec<u8> = values
            .iter()
            .filter_map(|v| match v {
                Some(Src::Reg(r)) => Some(*r),
                _ => None,
            })
            .collect();
        let second = (a == RBX).then(|| {
            *GPRS
                .iter()
                .find(|&&r| r != RBX && !operands.contains(&r))
                .expect("five operands leave a free register")
        });
        for (k, r) in [Some(RBX), second].into_iter().enumerate() {
            if let Some(r) = r {
                let offset = ST_CX16_SAVE + 8 * k as i32;
                dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], Rq(r));
            }
        }
        [Some(RBX), second]
    }

    /// Runs `lock cmpxchg16b` on the guest address in `a` with `rdx:rax` already holding the expected
    /// value: loads the new value into `rcx:rbx`, then gives back the borrowed registers. Returns the
    /// offset of the instruction that can fault.
    fn cmpxchg16b(&mut self, a: u8, low: u8, high: u8, borrowed: [Option<u8>; 2]) -> u32 {
        if let Some(y) = borrowed[1] {
            dynasm!(self.asm ; .arch x64 ; lea Rq(y), [r14 + Rq(a)]);
        }
        dynasm!(self.asm ; .arch x64 ; mov rcx, Rq(high) ; mov rbx, Rq(low));
        let site = self.asm.offset().0 as u32;
        match borrowed[1] {
            Some(y) => dynasm!(self.asm ; .arch x64 ; lock cmpxchg16b [Rq(y)]),
            None => dynasm!(self.asm ; .arch x64 ; lock cmpxchg16b [r14 + Rq(a)]),
        }
        self.give_back(borrowed);
        site
    }

    fn give_back(&mut self, borrowed: [Option<u8>; 2]) {
        for (k, r) in borrowed.into_iter().enumerate() {
            if let Some(r) = r {
                let offset = ST_CX16_SAVE + 8 * k as i32;
                dynasm!(self.asm ; .arch x64 ; mov Rq(r), QWORD [r13 + offset]);
            }
        }
    }

    /// Emits the fastmem access itself and returns the offsets of the instructions that can
    /// fault.
    fn access(
        &mut self,
        op: Op,
        a: u8,
        dst: Option<u8>,
        values: [Option<Src>; 4],
        borrowed: [Option<u8>; 2],
    ) -> Vec<u32> {
        let at = |asm: &Asm| asm.offset().0 as u32;
        let reg = |src: Option<Src>| match src {
            Some(Src::Reg(r)) => r,
            _ => unreachable!("this operand is always in a register"),
        };
        let site;
        match op {
            Op::Load { size, .. } => {
                let d = dst.unwrap();
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), BYTE [r14 + Rq(a)]),
                    1 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), WORD [r14 + Rq(a)]),
                    2 => dynasm!(self.asm ; .arch x64 ; mov Rd(d), DWORD [r14 + Rq(a)]),
                    _ => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r14 + Rq(a)]),
                }
            }
            Op::LoadVec { size, .. } => {
                let d = dst.unwrap();
                site = at(&self.asm);
                match size {
                    0 => {
                        dynasm!(self.asm ; .arch x64 ; movzx eax, BYTE [r14 + Rq(a)] ; movd Rx(d), eax)
                    }
                    1 => {
                        dynasm!(self.asm ; .arch x64 ; movzx eax, WORD [r14 + Rq(a)] ; movd Rx(d), eax)
                    }
                    2 => dynasm!(self.asm ; .arch x64 ; movd Rx(d), DWORD [r14 + Rq(a)]),
                    3 => dynasm!(self.asm ; .arch x64 ; movq Rx(d), QWORD [r14 + Rq(a)]),
                    _ => dynasm!(self.asm ; .arch x64 ; movdqu Rx(d), [r14 + Rq(a)]),
                }
            }
            Op::StoreVec { size, .. } => {
                let r = reg(values[0]);
                if size < 2 {
                    dynasm!(self.asm ; .arch x64 ; movd eax, Rx(r));
                }
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; mov BYTE [r14 + Rq(a)], al),
                    1 => dynasm!(self.asm ; .arch x64 ; mov WORD [r14 + Rq(a)], ax),
                    2 => dynasm!(self.asm ; .arch x64 ; movd DWORD [r14 + Rq(a)], Rx(r)),
                    3 => dynasm!(self.asm ; .arch x64 ; movq QWORD [r14 + Rq(a)], Rx(r)),
                    _ => dynasm!(self.asm ; .arch x64 ; movdqu [r14 + Rq(a)], Rx(r)),
                }
            }
            Op::Store { size, .. } => {
                site = at(&self.asm);
                match (values[0], size) {
                    (Some(Src::Reg(r)), 0) => {
                        dynasm!(self.asm ; .arch x64 ; mov BYTE [r14 + Rq(a)], Rb(r))
                    }
                    (Some(Src::Reg(r)), 1) => {
                        dynasm!(self.asm ; .arch x64 ; mov WORD [r14 + Rq(a)], Rw(r))
                    }
                    (Some(Src::Reg(r)), 2) => {
                        dynasm!(self.asm ; .arch x64 ; mov DWORD [r14 + Rq(a)], Rd(r))
                    }
                    (Some(Src::Reg(r)), _) => {
                        dynasm!(self.asm ; .arch x64 ; mov QWORD [r14 + Rq(a)], Rq(r))
                    }
                    (Some(Src::Imm(i)), 0) => {
                        dynasm!(self.asm ; .arch x64 ; mov BYTE [r14 + Rq(a)], i as i8)
                    }
                    (Some(Src::Imm(i)), 1) => {
                        dynasm!(self.asm ; .arch x64 ; mov WORD [r14 + Rq(a)], i as i16)
                    }
                    (Some(Src::Imm(i)), 2) => {
                        dynasm!(self.asm ; .arch x64 ; mov DWORD [r14 + Rq(a)], i)
                    }
                    (Some(Src::Imm(i)), _) => {
                        dynasm!(self.asm ; .arch x64 ; mov QWORD [r14 + Rq(a)], i)
                    }
                    (None, _) => unreachable!("a store always has a value"),
                }
            }
            Op::AtomicRmw {
                op: AtomicOp::Add | AtomicOp::Swp,
                size,
                ..
            } => {
                let (v, d) = (reg(values[0]), dst.unwrap());
                dynasm!(self.asm ; .arch x64 ; mov Rq(d), Rq(v));
                site = at(&self.asm);
                match (op, size) {
                    (
                        Op::AtomicRmw {
                            op: AtomicOp::Add, ..
                        },
                        0,
                    ) => {
                        dynasm!(self.asm ; .arch x64 ; lock xadd BYTE [r14 + Rq(a)], Rb(d) ; movzx Rd(d), Rb(d))
                    }
                    (
                        Op::AtomicRmw {
                            op: AtomicOp::Add, ..
                        },
                        1,
                    ) => {
                        dynasm!(self.asm ; .arch x64 ; lock xadd WORD [r14 + Rq(a)], Rw(d) ; movzx Rd(d), Rw(d))
                    }
                    (
                        Op::AtomicRmw {
                            op: AtomicOp::Add, ..
                        },
                        2,
                    ) => dynasm!(self.asm ; .arch x64 ; lock xadd DWORD [r14 + Rq(a)], Rd(d)),
                    (
                        Op::AtomicRmw {
                            op: AtomicOp::Add, ..
                        },
                        _,
                    ) => dynasm!(self.asm ; .arch x64 ; lock xadd QWORD [r14 + Rq(a)], Rq(d)),
                    (_, 0) => {
                        dynasm!(self.asm ; .arch x64 ; xchg BYTE [r14 + Rq(a)], Rb(d) ; movzx Rd(d), Rb(d))
                    }
                    (_, 1) => {
                        dynasm!(self.asm ; .arch x64 ; xchg WORD [r14 + Rq(a)], Rw(d) ; movzx Rd(d), Rw(d))
                    }
                    (_, 2) => dynasm!(self.asm ; .arch x64 ; xchg DWORD [r14 + Rq(a)], Rd(d)),
                    (_, _) => dynasm!(self.asm ; .arch x64 ; xchg QWORD [r14 + Rq(a)], Rq(d)),
                }
            }
            Op::AtomicRmw { op: rmw, size, .. } => {
                let (v, d) = (reg(values[0]), dst.unwrap());
                let first = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; movzx eax, BYTE [r14 + Rq(a)]),
                    1 => dynasm!(self.asm ; .arch x64 ; movzx eax, WORD [r14 + Rq(a)]),
                    2 => dynasm!(self.asm ; .arch x64 ; mov eax, DWORD [r14 + Rq(a)]),
                    _ => dynasm!(self.asm ; .arch x64 ; mov rax, QWORD [r14 + Rq(a)]),
                }
                let retry = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch x64 ; =>retry);
                let signed = matches!(rmw, AtomicOp::Smax | AtomicOp::Smin);
                match rmw {
                    AtomicOp::Clr => {
                        dynasm!(self.asm ; .arch x64 ; mov rcx, Rq(v) ; not rcx ; and rcx, rax)
                    }
                    AtomicOp::Eor => dynasm!(self.asm ; .arch x64 ; mov rcx, rax ; xor rcx, Rq(v)),
                    AtomicOp::Set => dynasm!(self.asm ; .arch x64 ; mov rcx, rax ; or rcx, Rq(v)),
                    _ => {
                        match (signed, size) {
                            (true, 0) => {
                                dynasm!(self.asm ; .arch x64 ; movsx rcx, al ; movsx rdx, Rb(v))
                            }
                            (true, 1) => {
                                dynasm!(self.asm ; .arch x64 ; movsx rcx, ax ; movsx rdx, Rw(v))
                            }
                            (true, 2) => {
                                dynasm!(self.asm ; .arch x64 ; movsxd rcx, eax ; movsxd rdx, Rd(v))
                            }
                            (false, 0) => {
                                dynasm!(self.asm ; .arch x64 ; movzx ecx, al ; movzx edx, Rb(v))
                            }
                            (false, 1) => {
                                dynasm!(self.asm ; .arch x64 ; movzx ecx, ax ; movzx edx, Rw(v))
                            }
                            (false, 2) => {
                                dynasm!(self.asm ; .arch x64 ; mov ecx, eax ; mov edx, Rd(v))
                            }
                            _ => dynasm!(self.asm ; .arch x64 ; mov rcx, rax ; mov rdx, Rq(v)),
                        }
                        dynasm!(self.asm ; .arch x64 ; cmp rcx, rdx);
                        match rmw {
                            AtomicOp::Smax => dynasm!(self.asm ; .arch x64 ; cmovl rcx, rdx),
                            AtomicOp::Smin => dynasm!(self.asm ; .arch x64 ; cmovg rcx, rdx),
                            AtomicOp::Umax => dynasm!(self.asm ; .arch x64 ; cmovb rcx, rdx),
                            _ => dynasm!(self.asm ; .arch x64 ; cmova rcx, rdx),
                        }
                    }
                }
                let second = at(&self.asm);
                match size {
                    0 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg BYTE [r14 + Rq(a)], cl ; jnz =>retry ; movzx Rd(d), al)
                    }
                    1 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg WORD [r14 + Rq(a)], cx ; jnz =>retry ; movzx Rd(d), ax)
                    }
                    2 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg DWORD [r14 + Rq(a)], ecx ; jnz =>retry ; mov Rd(d), eax)
                    }
                    _ => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg QWORD [r14 + Rq(a)], rcx ; jnz =>retry ; mov Rq(d), rax)
                    }
                }
                return vec![first, second];
            }
            Op::Cas { size, .. } => {
                let (e, n, d) = (reg(values[0]), reg(values[2]), dst.unwrap());
                dynasm!(self.asm ; .arch x64 ; mov rax, Rq(e));
                site = at(&self.asm);
                match size {
                    0 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg BYTE [r14 + Rq(a)], Rb(n) ; movzx Rd(d), al)
                    }
                    1 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg WORD [r14 + Rq(a)], Rw(n) ; movzx Rd(d), ax)
                    }
                    2 => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg DWORD [r14 + Rq(a)], Rd(n) ; mov Rd(d), eax)
                    }
                    _ => {
                        dynasm!(self.asm ; .arch x64 ; lock cmpxchg QWORD [r14 + Rq(a)], Rq(n) ; mov Rq(d), rax)
                    }
                }
            }
            Op::LoadExclusive { size: 4, .. } => {
                let d = dst.unwrap();
                site = at(&self.asm);
                dynasm!(self.asm
                    ; .arch x64
                    ; movdqu xmm15, [r14 + Rq(a)]
                    ; mov rcx, QWORD [r13 + ST_RESERVATION]
                    ; movdqu [rcx + RES_VALUE_LO], xmm15
                    ; movq Rq(d), xmm15
                    ; mov rax, QWORD [rcx + RES_VALUE_HI]
                    ; mov QWORD [r13 + ST_HIGH], rax
                    ; mov QWORD [rcx + RES_ADDR], Rq(a)
                    ; mov QWORD [rcx + RES_SIZE], 16
                );
            }
            Op::StoreExclusive128 { .. } => {
                let (low, high, d) = (reg(values[0]), reg(values[1]), dst.unwrap());
                let (lost, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rcx, QWORD [r13 + ST_RESERVATION]
                    ; cmp QWORD [rcx + RES_SIZE], 16
                    ; jne =>lost
                    ; cmp QWORD [rcx + RES_ADDR], Rq(a)
                    ; jne =>lost
                    ; mov rax, QWORD [rcx + RES_VALUE_LO]
                    ; mov rdx, QWORD [rcx + RES_VALUE_HI]
                );
                site = self.cmpxchg16b(a, low, high, borrowed);
                dynasm!(self.asm
                    ; .arch x64
                    ; setne al
                    ; movzx Rd(d), al
                    ; jmp =>done
                    ; =>lost
                    ; mov Rd(d), 1
                    ; =>done
                    ; mov rcx, QWORD [r13 + ST_RESERVATION]
                    ; mov QWORD [rcx + RES_SIZE], 0
                );
            }
            Op::Cas128 { .. } => {
                let [el, eh, nl, nh] = values.map(reg);
                let d = dst.unwrap();
                dynasm!(self.asm ; .arch x64 ; mov rax, Rq(el) ; mov rdx, Rq(eh));
                site = self.cmpxchg16b(a, nl, nh, borrowed);
                dynasm!(self.asm
                    ; .arch x64
                    ; mov QWORD [r13 + ST_HIGH], rdx
                    ; mov Rq(d), rax
                );
            }
            Op::LoadExclusive { size, .. } => {
                let d = dst.unwrap();
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), BYTE [r14 + Rq(a)]),
                    1 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), WORD [r14 + Rq(a)]),
                    2 => dynasm!(self.asm ; .arch x64 ; mov Rd(d), DWORD [r14 + Rq(a)]),
                    _ => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r14 + Rq(a)]),
                }
                let bytes = 1i32 << size;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rcx, QWORD [r13 + ST_RESERVATION]
                    ; mov QWORD [rcx + RES_VALUE_LO], Rq(d)
                    ; mov QWORD [rcx + RES_VALUE_HI], 0
                    ; mov QWORD [rcx + RES_ADDR], Rq(a)
                    ; mov QWORD [rcx + RES_SIZE], bytes
                );
            }
            Op::StoreExclusive { size, .. } => {
                let (n, d) = (reg(values[0]), dst.unwrap());
                let (lost, done) = (self.asm.new_dynamic_label(), self.asm.new_dynamic_label());
                let bytes = 1i32 << size;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rcx, QWORD [r13 + ST_RESERVATION]
                    ; cmp QWORD [rcx + RES_SIZE], bytes
                    ; jne =>lost
                    ; cmp QWORD [rcx + RES_ADDR], Rq(a)
                    ; jne =>lost
                    ; mov rax, QWORD [rcx + RES_VALUE_LO]
                );
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; lock cmpxchg BYTE [r14 + Rq(a)], Rb(n)),
                    1 => dynasm!(self.asm ; .arch x64 ; lock cmpxchg WORD [r14 + Rq(a)], Rw(n)),
                    2 => dynasm!(self.asm ; .arch x64 ; lock cmpxchg DWORD [r14 + Rq(a)], Rd(n)),
                    _ => dynasm!(self.asm ; .arch x64 ; lock cmpxchg QWORD [r14 + Rq(a)], Rq(n)),
                }
                dynasm!(self.asm
                    ; .arch x64
                    ; setne al
                    ; movzx Rd(d), al
                    ; jmp =>done
                    ; =>lost
                    ; mov Rd(d), 1
                    ; =>done
                    ; mov QWORD [rcx + RES_SIZE], 0
                );
            }
            Op::ZeroBlock { .. } => {
                dynasm!(self.asm ; .arch x64 ; pxor xmm15, xmm15);
                let mut sites = Vec::new();
                for offset in [0, 16, 32, 48] {
                    sites.push(at(&self.asm));
                    dynasm!(self.asm ; .arch x64 ; movdqu [r14 + Rq(a) + offset], xmm15);
                }
                return sites;
            }
            _ => unreachable!(),
        }
        vec![site]
    }

    fn terminal(&mut self) {
        match self.block.terminal {
            Terminal::Jump(target) => self.link(target),
            Terminal::JumpIndirect(t) => {
                match self.src(t, W::W64) {
                    Src::Imm(i) => dynasm!(self.asm ; .arch x64 ; mov rax, i),
                    Src::Reg(r) => dynasm!(self.asm ; .arch x64 ; mov rax, Rq(r)),
                }
                let miss = self.asm.new_dynamic_label();
                let mask = (1i32 << TABLE_BITS) - 1;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov QWORD [r15 + CTX_PC], rax
                    ; mov ecx, eax
                    ; shr ecx, 2
                    ; and ecx, mask
                    ; shl ecx, 4
                    ; mov rdx, QWORD [r13 + ST_TABLE]
                    ; cmp QWORD [rdx + rcx], rax
                    ; jne =>miss
                    ; jmp QWORD [rdx + rcx + 8]
                    ; =>miss
                    ; xor eax, eax
                    ; ret
                );
            }
            Terminal::Branch {
                cond,
                taken,
                not_taken,
            } => {
                let c = self.reg(cond);
                let other = self.asm.new_dynamic_label();
                dynasm!(self.asm ; .arch x64 ; test Rd(c), Rd(c) ; jz =>other);
                self.link(taken);
                dynasm!(self.asm ; .arch x64 ; =>other);
                self.link(not_taken);
            }
        }
    }

    /// A linkable exit to `target`: a `jmp rel32` that starts out jumping to the code right after
    /// it, which leaves to the dispatcher with the jump's address in `JitState::link`. Linking
    /// points the `jmp` at the target block; unlinking sets its offset back to zero.
    fn link(&mut self, target: u64) {
        let site = self.asm.new_dynamic_label();
        let unlinked = self.asm.new_dynamic_label();
        dynasm!(self.asm
            ; .arch x64
            ; =>site
            ; jmp =>unlinked
            ; =>unlinked
        );
        mov_imm64(&mut self.asm, 0, target);
        dynasm!(self.asm
            ; .arch x64
            ; mov QWORD [r15 + CTX_PC], rax
            ; lea rcx, [=>site]
            ; mov QWORD [r13 + ST_LINK], rcx
            ; xor eax, eax
            ; ret
        );
    }

    fn exit_to(&mut self, pc: u64, code: u64) {
        mov_imm64(&mut self.asm, 0, pc);
        dynasm!(self.asm
            ; .arch x64
            ; mov QWORD [r15 + CTX_PC], rax
            ; mov eax, code as i32
            ; ret
        );
    }

    fn stub(&mut self, stub: &Stub) -> u32 {
        let start = self.asm.offset().0 as u32;
        dynasm!(self.asm ; .arch x64 ; =>stub.label);
        self.give_back(stub.borrowed);
        self.save(stub.saved);
        let offsets = [ST_SCRATCH, ST_SCRATCH + 8, ST_SCRATCH2, ST_SCRATCH2 + 8];
        for (value, offset) in stub.values.iter().zip(offsets) {
            match *value {
                Some(Src::Reg(r)) if stub.vec => {
                    dynasm!(self.asm ; .arch x64 ; movdqu [r13 + offset], Rx(r))
                }
                Some(Src::Reg(r)) => {
                    dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], Rq(r))
                }
                Some(Src::Imm(imm)) => {
                    dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], imm)
                }
                None => {}
            }
        }
        dynasm!(self.asm
            ; .arch x64
            ; mov rsi, Rq(stub.addr)
            ; mov rdi, r13
            ; mov rdx, QWORD stub.pc as i64
            ; mov ecx, stub.info
            ; mov rax, QWORD stub.helper as i64
            ; sub rsp, 8
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jnz =>self.fault_exit
        );
        self.restore(stub.saved);
        match stub.dst {
            Some(d) if stub.vec => dynasm!(self.asm ; .arch x64 ; movdqu Rx(d), [r13 + ST_SCRATCH]),
            Some(d) => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r13 + ST_SCRATCH]),
            None => {}
        }
        dynasm!(self.asm ; .arch x64 ; jmp =>stub.resume);
        start
    }
}

/// Compiles a block that the passes have already run on and that starts at guest `start`.
/// Every block first checks the interrupt flag, so linked blocks that loop still stop.
pub(super) fn compile(block: &Block, start: u64, target: &Target) -> Compiled {
    let n = block.ops.len();
    let mut last_use: Vec<u32> = (0..n as u32).collect();
    for (i, op) in block.ops.iter().enumerate() {
        for v in op.args() {
            last_use[v.index()] = i as u32;
        }
    }
    let mut terminal = block.terminal;
    terminal.for_each_arg(|v| last_use[v.index()] = n as u32);
    let mut asm = Asm::new(0);
    let fault_exit = asm.new_dynamic_label();
    let mut e = Emitter {
        asm,
        block,
        target,
        holder: [[None; 16]; 2],
        reg: vec![None; n],
        slot: vec![None; n],
        free_slots: Vec::new(),
        next_slot: 0,
        locked: [0; 2],
        last_use,
        stubs: Vec::new(),
        fault_exit,
    };
    let interrupted = e.asm.new_dynamic_label();
    dynasm!(e.asm
        ; .arch x64
        ; mov rax, QWORD [r13 + ST_INTERRUPT]
        ; cmp BYTE [rax], 0
        ; jne =>interrupted
    );
    for i in 0..n {
        e.emit(i);
    }
    e.terminal();
    let stubs = std::mem::take(&mut e.stubs);
    let mut sites = Vec::new();
    for stub in &stubs {
        let start = e.stub(stub);
        sites.extend(stub.sites.iter().map(|&site| (site, start)));
    }
    dynasm!(e.asm
        ; .arch x64
        ; =>fault_exit
        ; mov eax, EXIT_FAULT as i32
        ; ret
        ; =>interrupted
    );
    e.exit_to(start, EXIT_CONTINUE);
    let code = e.asm.finalize().expect("every label is defined");
    Compiled { code, sites }
}

/// Code empty indirect-table slots point at: returns to the dispatcher like a table miss, so a
/// guest target equal to the empty key still reaches it.
pub(super) fn miss_stub() -> Vec<u8> {
    let mut asm = Asm::new(0);
    dynasm!(asm
        ; .arch x64
        ; mov eax, EXIT_CONTINUE as i32
        ; ret
    );
    asm.finalize()
        .expect("the miss stub has no unresolved labels")
}

/// `extern "sysv64" fn(ctx, state, arena base, block code) -> exit code`: saves the
/// callee-saved registers, sets up the pinned ones and calls the block.
pub(super) fn trampoline() -> Vec<u8> {
    let mut asm = Asm::new(0);
    dynasm!(asm
        ; .arch x64
        ; push rbx
        ; push rbp
        ; push r12
        ; push r13
        ; push r14
        ; push r15
        ; sub rsp, 8
        ; mov r15, rdi
        ; mov r13, rsi
        ; mov r14, rdx
        ; call rcx
        ; add rsp, 8
        ; pop r15
        ; pop r14
        ; pop r13
        ; pop r12
        ; pop rbp
        ; pop rbx
        ; ret
    );
    asm.finalize().expect("the trampoline has no labels")
}
