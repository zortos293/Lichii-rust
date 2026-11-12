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

use super::JitState;
use super::ir::{Block, Loc, Op, Terminal, Ty, Value, W};
use crate::ThreadContext;
use crate::interpreter::condition_holds;

type Asm = VecAssembler<X64Relocation>;

pub(super) const EXIT_CONTINUE: u64 = 0;
pub(super) const EXIT_INTERPRET: u64 = 1;
pub(super) const EXIT_FAULT: u64 = 2;

/// Allocatable general-purpose registers, callee-saved ones first so slow paths save less.
const GPRS: [u8; 9] = [3, 5, 12, 6, 7, 8, 9, 10, 11];
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

struct Stub {
    label: DynamicLabel,
    resume: DynamicLabel,
    site: Option<u32>,
    addr: u8,
    size: u8,
    pc: u64,
    vec: bool,
    dst: Option<u8>,
    value: Option<Src>,
    saved: [u16; 2],
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
const ST_SAVE_GPR: i32 = ctx(offset_of!(JitState, save_gpr));
const ST_SAVE_XMM: i32 = ctx(offset_of!(JitState, save_xmm));
const ST_SPILL: i32 = ctx(offset_of!(JitState, spill));

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
            Op::Load { .. } | Op::LoadVec { .. } | Op::Store { .. } | Op::StoreVec { .. } => {
                self.memory(i, op)
            }
            Op::Fence => dynasm!(self.asm ; .arch x64 ; mfence),
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

    fn memory(&mut self, i: usize, op: Op) {
        let (size, addr, value, pc, vec) = match op {
            Op::Load { size, addr, pc } => (size, addr, None, pc, false),
            Op::LoadVec { size, addr, pc } => (size, addr, None, pc, true),
            Op::Store {
                size,
                addr,
                value,
                pc,
            } => (size, addr, Some(value), pc, false),
            Op::StoreVec {
                size,
                addr,
                value,
                pc,
            } => (size, addr, Some(value), pc, true),
            _ => unreachable!(),
        };
        let a = self.reg(addr);
        let value = value.map(|x| {
            if vec {
                Src::Reg(self.reg(x))
            } else {
                self.src(x, W::W64)
            }
        });
        let dst = value.is_none().then(|| self.def(Value(i as u32)));
        let label = self.asm.new_dynamic_label();
        let resume = self.asm.new_dynamic_label();
        let site = match self.target.arena_size {
            Some(arena) => {
                let limit = (arena - (1u64 << size)) as i64;
                dynasm!(self.asm
                    ; .arch x64
                    ; mov rax, QWORD limit
                    ; cmp Rq(a), rax
                    ; ja =>label
                );
                Some(self.access(a, size, vec, dst, value))
            }
            None => {
                dynasm!(self.asm ; .arch x64 ; jmp =>label);
                None
            }
        };
        dynasm!(self.asm ; .arch x64 ; =>resume);
        let mut saved = [0u16; 2];
        for class in [Class::Gpr, Class::Xmm] {
            for r in 0..16u8 {
                let Some(held) = self.holder[class as usize][r as usize] else {
                    continue;
                };
                let caller_saved = class == Class::Xmm || CALLER_SAVED & (1 << r) != 0;
                if caller_saved && self.last_use[held as usize] as usize > i && dst != Some(r) {
                    saved[class as usize] |= 1 << r;
                }
            }
        }
        self.stubs.push(Stub {
            label,
            resume,
            site,
            addr: a,
            size,
            pc,
            vec,
            dst,
            value,
            saved,
        });
    }

    /// Emits the fastmem access itself and returns the offset of the instruction that can fault.
    fn access(&mut self, a: u8, size: u8, vec: bool, dst: Option<u8>, value: Option<Src>) -> u32 {
        let at = |asm: &Asm| asm.offset().0 as u32;
        let site;
        match (dst, value, vec) {
            (Some(d), _, false) => {
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), BYTE [r14 + Rq(a)]),
                    1 => dynasm!(self.asm ; .arch x64 ; movzx Rd(d), WORD [r14 + Rq(a)]),
                    2 => dynasm!(self.asm ; .arch x64 ; mov Rd(d), DWORD [r14 + Rq(a)]),
                    _ => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r14 + Rq(a)]),
                }
            }
            (Some(d), _, true) => {
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
            (None, Some(Src::Reg(r)), true) => {
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
            (None, Some(Src::Reg(r)), false) => {
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; mov BYTE [r14 + Rq(a)], Rb(r)),
                    1 => dynasm!(self.asm ; .arch x64 ; mov WORD [r14 + Rq(a)], Rw(r)),
                    2 => dynasm!(self.asm ; .arch x64 ; mov DWORD [r14 + Rq(a)], Rd(r)),
                    _ => dynasm!(self.asm ; .arch x64 ; mov QWORD [r14 + Rq(a)], Rq(r)),
                }
            }
            (None, Some(Src::Imm(imm)), _) => {
                site = at(&self.asm);
                match size {
                    0 => dynasm!(self.asm ; .arch x64 ; mov BYTE [r14 + Rq(a)], imm as i8),
                    1 => dynasm!(self.asm ; .arch x64 ; mov WORD [r14 + Rq(a)], imm as i16),
                    2 => dynasm!(self.asm ; .arch x64 ; mov DWORD [r14 + Rq(a)], imm),
                    _ => dynasm!(self.asm ; .arch x64 ; mov QWORD [r14 + Rq(a)], imm),
                }
            }
            (None, None, _) => unreachable!("a store always has a value"),
        }
        site
    }

    fn terminal(&mut self) {
        match self.block.terminal {
            Terminal::Jump(target) => self.exit_to(target, EXIT_CONTINUE),
            Terminal::Interpret(pc) => self.exit_to(pc, EXIT_INTERPRET),
            Terminal::JumpIndirect(t) => {
                match self.src(t, W::W64) {
                    Src::Imm(i) => dynasm!(self.asm ; .arch x64 ; mov QWORD [r15 + CTX_PC], i),
                    Src::Reg(r) => dynasm!(self.asm ; .arch x64 ; mov QWORD [r15 + CTX_PC], Rq(r)),
                }
                dynasm!(self.asm ; .arch x64 ; xor eax, eax ; ret);
            }
            Terminal::Branch {
                cond,
                taken,
                not_taken,
            } => {
                let c = self.reg(cond);
                mov_imm64(&mut self.asm, 0, not_taken);
                mov_imm64(&mut self.asm, 1, taken);
                dynasm!(self.asm
                    ; .arch x64
                    ; test Rd(c), Rd(c)
                    ; cmovnz rax, rcx
                    ; mov QWORD [r15 + CTX_PC], rax
                    ; xor eax, eax
                    ; ret
                );
            }
        }
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
        for r in 0..16u8 {
            if stub.saved[0] & (1 << r) != 0 {
                let offset = ST_SAVE_GPR + 8 * r as i32;
                dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + offset], Rq(r));
            }
            if stub.saved[1] & (1 << r) != 0 {
                let offset = ST_SAVE_XMM + 16 * r as i32;
                dynasm!(self.asm ; .arch x64 ; movdqu [r13 + offset], Rx(r));
            }
        }
        let helper = match stub.value {
            Some(Src::Reg(r)) if stub.vec => {
                dynasm!(self.asm ; .arch x64 ; movdqu [r13 + ST_SCRATCH], Rx(r));
                self.target.write
            }
            Some(Src::Reg(r)) => {
                dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + ST_SCRATCH], Rq(r));
                self.target.write
            }
            Some(Src::Imm(imm)) => {
                dynasm!(self.asm ; .arch x64 ; mov QWORD [r13 + ST_SCRATCH], imm);
                self.target.write
            }
            None => self.target.read,
        };
        let size = stub.size as i32;
        dynasm!(self.asm
            ; .arch x64
            ; mov rsi, Rq(stub.addr)
            ; mov rdi, r13
            ; mov rdx, QWORD stub.pc as i64
            ; mov ecx, size
            ; mov rax, QWORD helper as i64
            ; sub rsp, 8
            ; call rax
            ; add rsp, 8
            ; test al, al
            ; jnz =>self.fault_exit
        );
        for r in 0..16u8 {
            if stub.saved[0] & (1 << r) != 0 {
                let offset = ST_SAVE_GPR + 8 * r as i32;
                dynasm!(self.asm ; .arch x64 ; mov Rq(r), QWORD [r13 + offset]);
            }
            if stub.saved[1] & (1 << r) != 0 {
                let offset = ST_SAVE_XMM + 16 * r as i32;
                dynasm!(self.asm ; .arch x64 ; movdqu Rx(r), [r13 + offset]);
            }
        }
        match stub.dst {
            Some(d) if stub.vec => dynasm!(self.asm ; .arch x64 ; movdqu Rx(d), [r13 + ST_SCRATCH]),
            Some(d) => dynasm!(self.asm ; .arch x64 ; mov Rq(d), QWORD [r13 + ST_SCRATCH]),
            None => {}
        }
        dynasm!(self.asm ; .arch x64 ; jmp =>stub.resume);
        start
    }
}

/// Compiles a block that the passes have already run on.
pub(super) fn compile(block: &Block, target: &Target) -> Compiled {
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
    for i in 0..n {
        e.emit(i);
    }
    e.terminal();
    let stubs = std::mem::take(&mut e.stubs);
    let mut sites = Vec::new();
    for stub in &stubs {
        let start = e.stub(stub);
        if let Some(site) = stub.site {
            sites.push((site, start));
        }
    }
    dynasm!(e.asm
        ; .arch x64
        ; =>fault_exit
        ; mov eax, EXIT_FAULT as i32
        ; ret
    );
    let code = e.asm.finalize().expect("every label is defined");
    Compiled { code, sites }
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
