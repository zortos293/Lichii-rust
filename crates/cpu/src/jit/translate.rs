//! Translates decoded instructions into IR, one guest block at a time.

use super::ir::{Block, Loc, Op, Terminal, Value, W};
use crate::CpuConfig;
use crate::decoder::*;

/// What translating one instruction did.
pub(super) enum Step {
    /// Fall through to the next instruction.
    Next,
    /// The instruction ended the block.
    End(Terminal),
    /// The JIT does not translate this instruction; nothing was emitted.
    Rejected,
}

pub(super) struct Translator<'a> {
    pub block: Block,
    pub config: &'a CpuConfig,
}

impl Translator<'_> {
    fn op(&mut self, op: Op) -> Value {
        self.block.push(op)
    }

    fn k(&mut self, value: u64) -> Value {
        self.op(Op::Const(value))
    }

    fn x(&mut self, r: Reg) -> Value {
        match r.0 {
            31 => self.k(0),
            32 => self.op(Op::Get(Loc::Sp)),
            n => self.op(Op::Get(Loc::X(n))),
        }
    }

    /// `high:low` as one 64-bit value from the low words of two registers.
    fn join32(&mut self, low: Reg, high: Reg) -> Value {
        let low = self.x(low);
        let low = self.op(Op::ZeroExt(2, low));
        let high = self.x(high);
        let shift = self.k(32);
        let high = self.op(Op::Lsl(W::W64, high, shift));
        self.op(Op::Or(W::W64, high, low))
    }

    /// Writes the low and high words of `value` to two registers, zero-extended.
    fn split32(&mut self, value: Value, low: Reg, high: Reg) {
        let low_word = self.op(Op::ZeroExt(2, value));
        let shift = self.k(32);
        let high_word = self.op(Op::Lsr(W::W64, value, shift));
        self.set_x(low, low_word);
        self.set_x(high, high_word);
    }

    fn set_x(&mut self, r: Reg, value: Value) {
        match r.0 {
            31 => {}
            32 => {
                self.op(Op::Set(Loc::Sp, value));
            }
            n => {
                self.op(Op::Set(Loc::X(n), value));
            }
        }
    }

    fn set(&mut self, loc: Loc, value: Value) {
        self.op(Op::Set(loc, value));
    }

    fn nzcv(&mut self) -> Value {
        self.op(Op::Get(Loc::Nzcv))
    }

    fn cond(&mut self, cond: u8) -> Value {
        if cond >= 14 {
            return self.k(1);
        }
        let flags = self.nzcv();
        self.op(Op::TestCond(cond, flags))
    }

    fn add_sub(&mut self, sf: bool, sub: bool, setflags: bool, rd: Reg, a: Value, b: Value) {
        let w = W::of(sf);
        if setflags {
            let flags = self.op(if sub {
                Op::SubFlags(w, a, b)
            } else {
                Op::AddFlags(w, a, b)
            });
            self.set(Loc::Nzcv, flags);
        }
        let r = self.op(if sub {
            Op::Sub(w, a, b)
        } else {
            Op::Add(w, a, b)
        });
        self.set_x(rd, r);
    }

    fn logical(&mut self, sf: bool, op: LogicOp, rd: Reg, a: Value, b: Value) {
        let w = W::of(sf);
        let r = self.op(match op {
            LogicOp::And | LogicOp::Ands => Op::And(w, a, b),
            LogicOp::Orr => Op::Or(w, a, b),
            LogicOp::Eor => Op::Xor(w, a, b),
        });
        if op == LogicOp::Ands {
            let flags = self.op(Op::LogicFlags(w, r));
            self.set(Loc::Nzcv, flags);
        }
        self.set_x(rd, r);
    }

    fn shift(&mut self, w: W, value: Value, ty: ShiftType, amount: u8) -> Value {
        if amount == 0 {
            return value;
        }
        let amount = self.k(amount as u64);
        self.op(match ty {
            ShiftType::Lsl => Op::Lsl(w, value, amount),
            ShiftType::Lsr => Op::Lsr(w, value, amount),
            ShiftType::Asr => Op::Asr(w, value, amount),
            ShiftType::Ror => Op::Ror(w, value, amount),
        })
    }

    fn extend(&mut self, value: Value, ext: Extend, amount: u8) -> Value {
        let extended = match ext {
            Extend::Uxtb => self.op(Op::ZeroExt(0, value)),
            Extend::Uxth => self.op(Op::ZeroExt(1, value)),
            Extend::Uxtw => self.op(Op::ZeroExt(2, value)),
            Extend::Uxtx | Extend::Sxtx => value,
            Extend::Sxtb => self.op(Op::SignExt(W::W64, 0, value)),
            Extend::Sxth => self.op(Op::SignExt(W::W64, 1, value)),
            Extend::Sxtw => self.op(Op::SignExt(W::W64, 2, value)),
        };
        self.shift(W::W64, extended, ShiftType::Lsl, amount)
    }

    fn load_extend(&mut self, value: Value, size: u8, ext: LoadExt) -> Value {
        match ext {
            LoadExt::Zero => value,
            LoadExt::Sign32 => self.op(Op::SignExt(W::W32, size, value)),
            LoadExt::Sign64 => self.op(Op::SignExt(W::W64, size, value)),
        }
    }

    /// The access address and, for pre/post-indexed forms, the new base.
    fn address(&mut self, rn: Reg, addr: Addr) -> (Value, Option<Value>) {
        let base = self.x(rn);
        match addr {
            Addr::Offset(o) => (self.offset(base, o as i64), None),
            Addr::Pre(o) => {
                let a = self.offset(base, o as i64);
                (a, Some(a))
            }
            Addr::Post(o) => (base, Some(self.offset(base, o as i64))),
            Addr::Reg { rm, extend, shift } => {
                let index = self.x(rm);
                let index = self.extend(index, extend, shift);
                (self.op(Op::Add(W::W64, base, index)), None)
            }
        }
    }

    fn offset(&mut self, base: Value, offset: i64) -> Value {
        if offset == 0 {
            return base;
        }
        let offset = self.k(offset as u64);
        self.op(Op::Add(W::W64, base, offset))
    }

    fn writeback(&mut self, rn: Reg, base: Option<Value>) {
        if let Some(base) = base {
            self.set_x(rn, base);
        }
    }

    /// Translates the instruction at `pc`. A rejected instruction emits nothing.
    pub fn translate(&mut self, inst: &Inst, pc: u64) -> Step {
        let next = pc.wrapping_add(4);
        let target = |offset: i32| pc.wrapping_add(offset as i64 as u64);
        match *inst {
            Inst::Nop | Inst::Hint { .. } => {}
            Inst::Adr { rd, offset } => {
                let v = self.k(target(offset));
                self.set_x(rd, v);
            }
            Inst::Adrp { rd, offset } => {
                let v = self.k((pc & !0xfff).wrapping_add(offset as u64));
                self.set_x(rd, v);
            }
            Inst::AddSubImm {
                sf,
                sub,
                setflags,
                rd,
                rn,
                imm,
            } => {
                let a = self.x(rn);
                let b = self.k(imm as u64);
                self.add_sub(sf, sub, setflags, rd, a, b);
            }
            Inst::LogicalImm {
                sf,
                op,
                rd,
                rn,
                imm,
            } => {
                let a = self.x(rn);
                let b = self.k(imm);
                self.logical(sf, op, rd, a, b);
            }
            Inst::MovWide {
                sf,
                op,
                rd,
                shift,
                imm,
            } => {
                let w = W::of(sf);
                let imm = (imm as u64) << shift;
                let v = match op {
                    MovOp::N => self.k(!imm & w.mask()),
                    MovOp::Z => self.k(imm),
                    MovOp::K => {
                        let old = self.x(rd);
                        let keep = self.k(!(0xffff << shift));
                        let kept = self.op(Op::And(w, old, keep));
                        let imm = self.k(imm);
                        self.op(Op::Or(w, kept, imm))
                    }
                };
                self.set_x(rd, v);
            }
            Inst::Bitfield {
                sf,
                op,
                rd,
                rn,
                immr,
                imms,
            } => {
                let w = W::of(sf);
                let Some((wmask, tmask)) = decode_bit_masks(sf, imms as u32, immr as u32, false)
                else {
                    unreachable!("the decoder only emits valid bitfield immediates");
                };
                let src = self.x(rn);
                let rotated = self.shift(w, src, ShiftType::Ror, immr);
                let v = match op {
                    BitfieldOp::Ubfm => {
                        let mask = self.k(wmask & tmask);
                        self.op(Op::And(w, rotated, mask))
                    }
                    BitfieldOp::Sbfm => {
                        let up = self.k(63 - imms as u64);
                        let top = self.op(Op::Lsl(W::W64, src, up));
                        let down = self.k(63);
                        let top = self.op(Op::Asr(W::W64, top, down));
                        let not_t = self.k(!tmask);
                        let top = self.op(Op::And(W::W64, top, not_t));
                        let mask = self.k(wmask & tmask);
                        let bottom = self.op(Op::And(W::W64, rotated, mask));
                        self.op(Op::Or(w, top, bottom))
                    }
                    BitfieldOp::Bfm => {
                        let dst = self.x(rd);
                        let not_w = self.k(!wmask);
                        let kept = self.op(Op::And(W::W64, dst, not_w));
                        let wm = self.k(wmask);
                        let inserted = self.op(Op::And(W::W64, rotated, wm));
                        let bottom = self.op(Op::Or(W::W64, kept, inserted));
                        let not_t = self.k(!tmask);
                        let top = self.op(Op::And(W::W64, dst, not_t));
                        let tm = self.k(tmask);
                        let bottom = self.op(Op::And(W::W64, bottom, tm));
                        self.op(Op::Or(w, top, bottom))
                    }
                };
                self.set_x(rd, v);
            }
            Inst::Extr {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                let w = W::of(sf);
                let (n, m) = (self.x(rn), self.x(rm));
                let v = if lsb == 0 {
                    let zero = self.k(0);
                    self.op(Op::Or(w, m, zero))
                } else {
                    let low = self.shift(w, m, ShiftType::Lsr, lsb);
                    let high = self.shift(w, n, ShiftType::Lsl, w.bits() as u8 - lsb);
                    self.op(Op::Or(w, low, high))
                };
                self.set_x(rd, v);
            }
            Inst::AddSubShift {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let a = self.x(rn);
                let m = self.x(rm);
                let b = self.shift(W::of(sf), m, shift, amount);
                self.add_sub(sf, sub, setflags, rd, a, b);
            }
            Inst::AddSubExt {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
                extend,
                amount,
            } => {
                let a = self.x(rn);
                let m = self.x(rm);
                let b = self.extend(m, extend, amount);
                self.add_sub(sf, sub, setflags, rd, a, b);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
            } => {
                let w = W::of(sf);
                let (a, b) = (self.x(rn), self.x(rm));
                let flags = self.nzcv();
                if setflags {
                    let new = self.op(if sub {
                        Op::SbcFlags(w, a, b, flags)
                    } else {
                        Op::AdcFlags(w, a, b, flags)
                    });
                    self.set(Loc::Nzcv, new);
                }
                let r = self.op(if sub {
                    Op::Sbc(w, a, b, flags)
                } else {
                    Op::Adc(w, a, b, flags)
                });
                self.set_x(rd, r);
            }
            Inst::LogicalShift {
                sf,
                op,
                invert,
                rd,
                rn,
                rm,
                shift,
                amount,
            } => {
                let w = W::of(sf);
                let a = self.x(rn);
                let m = self.x(rm);
                let mut b = self.shift(w, m, shift, amount);
                if invert {
                    b = self.op(Op::Not(w, b));
                }
                self.logical(sf, op, rd, a, b);
            }
            Inst::CondCompare {
                sf,
                sub,
                rn,
                rhs,
                imm,
                nzcv,
                cond,
            } => {
                let w = W::of(sf);
                let a = self.x(rn);
                let b = if imm {
                    self.k(rhs as u64)
                } else {
                    self.x(Reg(rhs))
                };
                let compared = self.op(if sub {
                    Op::SubFlags(w, a, b)
                } else {
                    Op::AddFlags(w, a, b)
                });
                let holds = self.cond(cond);
                let fixed = self.k(nzcv as u64);
                let flags = self.op(Op::Select(W::W64, holds, compared, fixed));
                self.set(Loc::Nzcv, flags);
            }
            Inst::CondSelect {
                sf,
                op,
                rd,
                rn,
                rm,
                cond,
            } => {
                let w = W::of(sf);
                let (n, m) = (self.x(rn), self.x(rm));
                let other = match op {
                    CselOp::Csel => m,
                    CselOp::Csinc => {
                        let one = self.k(1);
                        self.op(Op::Add(w, m, one))
                    }
                    CselOp::Csinv => self.op(Op::Not(w, m)),
                    CselOp::Csneg => {
                        let zero = self.k(0);
                        self.op(Op::Sub(w, zero, m))
                    }
                };
                let holds = self.cond(cond);
                let v = self.op(Op::Select(w, holds, n, other));
                self.set_x(rd, v);
            }
            Inst::Dp1 { sf, op, rd, rn } => {
                let w = W::of(sf);
                let n = self.x(rn);
                let v = match op {
                    Dp1Op::Rbit => self.op(Op::Rbit(w, n)),
                    Dp1Op::Rev16 => self.op(Op::Rev16(w, n)),
                    Dp1Op::Rev32 if sf => self.op(Op::Rev32(n)),
                    Dp1Op::Rev32 | Dp1Op::Rev64 => self.op(Op::Rev(w, n)),
                    Dp1Op::Clz => self.op(Op::Clz(w, n)),
                    Dp1Op::Cls => {
                        let one = self.k(1);
                        let half = self.op(Op::Asr(w, n, one));
                        let changes = self.op(Op::Xor(w, n, half));
                        let zeros = self.op(Op::Clz(w, changes));
                        self.op(Op::Sub(w, zeros, one))
                    }
                };
                self.set_x(rd, v);
            }
            Inst::Dp2 { sf, op, rd, rn, rm } => {
                let w = W::of(sf);
                let (n, m) = (self.x(rn), self.x(rm));
                let v = self.op(match op {
                    Dp2Op::Udiv => Op::UDiv(w, n, m),
                    Dp2Op::Sdiv => Op::SDiv(w, n, m),
                    Dp2Op::Lslv => Op::Lsl(w, n, m),
                    Dp2Op::Lsrv => Op::Lsr(w, n, m),
                    Dp2Op::Asrv => Op::Asr(w, n, m),
                    Dp2Op::Rorv => Op::Ror(w, n, m),
                });
                self.set_x(rd, v);
            }
            Inst::Dp3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra,
            } => {
                let w = W::of(sf);
                let (n, m, a) = (self.x(rn), self.x(rm), self.x(ra));
                let long = |t: &mut Self, signed: bool| {
                    let (n, m) = if signed {
                        (
                            t.op(Op::SignExt(W::W64, 2, n)),
                            t.op(Op::SignExt(W::W64, 2, m)),
                        )
                    } else {
                        (t.op(Op::ZeroExt(2, n)), t.op(Op::ZeroExt(2, m)))
                    };
                    t.op(Op::Mul(W::W64, n, m))
                };
                let v = match op {
                    Dp3Op::Madd | Dp3Op::Msub => {
                        let product = self.op(Op::Mul(w, n, m));
                        self.op(if op == Dp3Op::Madd {
                            Op::Add(w, a, product)
                        } else {
                            Op::Sub(w, a, product)
                        })
                    }
                    Dp3Op::Smaddl | Dp3Op::Umaddl => {
                        let product = long(self, op == Dp3Op::Smaddl);
                        self.op(Op::Add(W::W64, a, product))
                    }
                    Dp3Op::Smsubl | Dp3Op::Umsubl => {
                        let product = long(self, op == Dp3Op::Smsubl);
                        self.op(Op::Sub(W::W64, a, product))
                    }
                    Dp3Op::Smulh => self.op(Op::SMulH(n, m)),
                    Dp3Op::Umulh => self.op(Op::UMulH(n, m)),
                };
                self.set_x(rd, v);
            }
            Inst::Pacga { rd, rn, rm } => {
                let (n, m) = (self.x(rn), self.x(rm));
                let x = self.op(Op::Xor(W::W64, n, m));
                let mask = self.k(0xffff_ffff_0000_0000);
                let v = self.op(Op::And(W::W64, x, mask));
                self.set_x(rd, v);
            }

            Inst::B { offset } => return Step::End(Terminal::Jump(target(offset))),
            Inst::Bl { offset } => {
                let ret = self.k(next);
                self.set_x(Reg::LR, ret);
                return Step::End(Terminal::Jump(target(offset)));
            }
            Inst::BCond { cond, offset } => {
                if cond >= 14 {
                    return Step::End(Terminal::Jump(target(offset)));
                }
                let holds = self.cond(cond);
                return Step::End(Terminal::Branch {
                    cond: holds,
                    taken: target(offset),
                    not_taken: next,
                });
            }
            Inst::Cbz {
                sf,
                nonzero,
                rt,
                offset,
            } => {
                let t = self.x(rt);
                let zero = self.op(Op::IsZero(W::of(sf), t));
                let (taken, not_taken) = if nonzero {
                    (next, target(offset))
                } else {
                    (target(offset), next)
                };
                return Step::End(Terminal::Branch {
                    cond: zero,
                    taken,
                    not_taken,
                });
            }
            Inst::Tbz {
                nonzero,
                rt,
                bit,
                offset,
            } => {
                let t = self.x(rt);
                let set = self.op(Op::TestBit(t, bit));
                let (taken, not_taken) = if nonzero {
                    (target(offset), next)
                } else {
                    (next, target(offset))
                };
                return Step::End(Terminal::Branch {
                    cond: set,
                    taken,
                    not_taken,
                });
            }
            Inst::Br { rn } | Inst::Ret { rn } => {
                let t = self.x(rn);
                return Step::End(Terminal::JumpIndirect(t));
            }
            Inst::Blr { rn } => {
                let t = self.x(rn);
                let ret = self.k(next);
                self.set_x(Reg::LR, ret);
                return Step::End(Terminal::JumpIndirect(t));
            }

            Inst::Barrier {
                op: BarrierOp::Dmb | BarrierOp::Dsb,
            } => {
                self.op(Op::Fence);
            }
            Inst::Barrier { .. } => {}
            Inst::Mrs { rt, reg } => {
                let v = match reg {
                    SysReg::Nzcv => {
                        let flags = self.nzcv();
                        let shift = self.k(28);
                        self.op(Op::Lsl(W::W64, flags, shift))
                    }
                    SysReg::Fpcr => self.op(Op::Get(Loc::Fpcr)),
                    SysReg::Fpsr => self.op(Op::Get(Loc::Fpsr)),
                    SysReg::TpidrEl0 => self.op(Op::Get(Loc::Tpidr)),
                    SysReg::TpidrroEl0 => self.op(Op::Get(Loc::Tpidrro)),
                    SysReg::CntfrqEl0 => self.k(self.config.clock.frequency()),
                    SysReg::CtrEl0 => self.k(self.config.ctr_el0 as u64),
                    SysReg::DczidEl0 => self.k(self.config.dczid_el0 as u64),
                    SysReg::CntpctEl0 | SysReg::CntvctEl0 => self.op(Op::Ticks),
                };
                self.set_x(rt, v);
            }
            Inst::Msr { rt, reg } => {
                let v = self.x(rt);
                match reg {
                    SysReg::Nzcv => {
                        let shift = self.k(28);
                        let flags = self.op(Op::Lsr(W::W64, v, shift));
                        let nibble = self.k(0xf);
                        let flags = self.op(Op::And(W::W64, flags, nibble));
                        self.set(Loc::Nzcv, flags);
                    }
                    SysReg::Fpcr | SysReg::Fpsr => {
                        let low = self.op(Op::ZeroExt(2, v));
                        let loc = if reg == SysReg::Fpcr {
                            Loc::Fpcr
                        } else {
                            Loc::Fpsr
                        };
                        self.set(loc, low);
                    }
                    SysReg::TpidrEl0 => self.set(Loc::Tpidr, v),
                    _ => unreachable!("the decoder rejects writes to read-only registers"),
                }
            }

            Inst::LoadLiteral {
                size,
                ext,
                rt,
                offset,
            } => {
                let addr = self.k(target(offset));
                let v = self.op(Op::Load { size, addr, pc });
                let v = self.load_extend(v, size, ext);
                self.set_x(rt, v);
            }
            Inst::LoadLiteralVec { size, vt, offset } => {
                let addr = self.k(target(offset));
                let v = self.op(Op::LoadVec { size, addr, pc });
                self.set(Loc::V(vt), v);
            }
            Inst::Load {
                size,
                ext,
                rt,
                rn,
                addr,
                ..
            } => {
                let (addr, wb) = self.address(rn, addr);
                let v = self.op(Op::Load { size, addr, pc });
                self.writeback(rn, wb);
                let v = self.load_extend(v, size, ext);
                self.set_x(rt, v);
            }
            Inst::Store {
                size, rt, rn, addr, ..
            } => {
                let (addr, wb) = self.address(rn, addr);
                let value = self.x(rt);
                self.op(Op::Store {
                    size,
                    addr,
                    value,
                    pc,
                });
                self.writeback(rn, wb);
            }
            Inst::LoadVec { size, vt, rn, addr } => {
                let (addr, wb) = self.address(rn, addr);
                let v = self.op(Op::LoadVec { size, addr, pc });
                self.set(Loc::V(vt), v);
                self.writeback(rn, wb);
            }
            Inst::StoreVec { size, vt, rn, addr } => {
                let (addr, wb) = self.address(rn, addr);
                let value = self.op(Op::Get(Loc::V(vt)));
                self.op(Op::StoreVec {
                    size,
                    addr,
                    value,
                    pc,
                });
                self.writeback(rn, wb);
            }
            Inst::LoadPair {
                size,
                ext,
                rt,
                rt2,
                rn,
                addr,
            } => {
                let (addr, wb) = self.address(rn, addr);
                let addr2 = self.offset(addr, 1 << size);
                let first = self.op(Op::Load { size, addr, pc });
                let second = self.op(Op::Load {
                    size,
                    addr: addr2,
                    pc,
                });
                self.writeback(rn, wb);
                let first = self.load_extend(first, size, ext);
                self.set_x(rt, first);
                let second = self.load_extend(second, size, ext);
                self.set_x(rt2, second);
            }
            Inst::StorePair {
                size,
                rt,
                rt2,
                rn,
                addr,
            } => {
                let (addr, wb) = self.address(rn, addr);
                let addr2 = self.offset(addr, 1 << size);
                let (first, second) = (self.x(rt), self.x(rt2));
                self.op(Op::Store {
                    size,
                    addr,
                    value: first,
                    pc,
                });
                self.op(Op::Store {
                    size,
                    addr: addr2,
                    value: second,
                    pc,
                });
                self.writeback(rn, wb);
            }
            Inst::LoadPairVec {
                size,
                vt,
                vt2,
                rn,
                addr,
            } => {
                let (addr, wb) = self.address(rn, addr);
                let addr2 = self.offset(addr, 1 << size);
                let first = self.op(Op::LoadVec { size, addr, pc });
                let second = self.op(Op::LoadVec {
                    size,
                    addr: addr2,
                    pc,
                });
                self.writeback(rn, wb);
                self.set(Loc::V(vt), first);
                self.set(Loc::V(vt2), second);
            }
            Inst::StorePairVec {
                size,
                vt,
                vt2,
                rn,
                addr,
            } => {
                let (addr, wb) = self.address(rn, addr);
                let addr2 = self.offset(addr, 1 << size);
                let first = self.op(Op::Get(Loc::V(vt)));
                let second = self.op(Op::Get(Loc::V(vt2)));
                self.op(Op::StoreVec {
                    size,
                    addr,
                    value: first,
                    pc,
                });
                self.op(Op::StoreVec {
                    size,
                    addr: addr2,
                    value: second,
                    pc,
                });
                self.writeback(rn, wb);
            }
            Inst::AtomicRmw {
                op,
                size,
                rs,
                rt,
                rn,
                ..
            } => {
                let addr = self.x(rn);
                let value = self.x(rs);
                let old = self.op(Op::AtomicRmw {
                    op,
                    size,
                    addr,
                    value,
                    pc,
                });
                self.set_x(rt, old);
            }
            Inst::Cas {
                size, rs, rt, rn, ..
            } => {
                let addr = self.x(rn);
                let (expected, new) = (self.x(rs), self.x(rt));
                let old = self.op(Op::Cas {
                    size,
                    addr,
                    expected,
                    new,
                    pc,
                });
                self.set_x(rs, old);
            }
            Inst::LoadExclusive {
                size,
                pair,
                rt,
                rt2,
                rn,
                ..
            } if size + pair as u8 <= 3 => {
                let addr = self.x(rn);
                let value = self.op(Op::LoadExclusive {
                    size: size + pair as u8,
                    addr,
                    pc,
                });
                if pair {
                    self.split32(value, rt, rt2);
                } else {
                    self.set_x(rt, value);
                }
            }
            Inst::StoreExclusive {
                size,
                pair,
                rs,
                rt,
                rt2,
                rn,
                ..
            } if size + pair as u8 <= 3 => {
                let addr = self.x(rn);
                let value = if pair {
                    self.join32(rt, rt2)
                } else {
                    self.x(rt)
                };
                let status = self.op(Op::StoreExclusive {
                    size: size + pair as u8,
                    addr,
                    value,
                    pc,
                });
                self.set_x(rs, status);
            }
            Inst::Casp {
                size: 2,
                rs,
                rt,
                rn,
                ..
            } => {
                let addr = self.x(rn);
                let expected = self.join32(Reg(rs), Reg(rs + 1));
                let new = self.join32(Reg(rt), Reg(rt + 1));
                let old = self.op(Op::Cas {
                    size: 3,
                    addr,
                    expected,
                    new,
                    pc,
                });
                self.split32(old, Reg(rs), Reg(rs + 1));
            }
            Inst::DcZva { rt } if 4 << (self.config.dczid_el0 & 0xf) == 64 => {
                let t = self.x(rt);
                let mask = self.k(!63);
                let addr = self.op(Op::And(W::W64, t, mask));
                self.op(Op::ZeroBlock { addr, pc });
            }
            _ => return Step::Rejected,
        }
        if inst.ends_block() {
            Step::End(Terminal::Jump(next))
        } else {
            Step::Next
        }
    }
}
