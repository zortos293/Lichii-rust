//! Block-local SSA IR between the decoder's [`Inst`](crate::decoder::Inst) stream and the host
//! backend.
//!
//! A block is a flat list of [`Op`]s, each defining at most one [`Value`] (its own index), plus
//! one [`Terminal`]. Values only refer to earlier ops. Passes replace the ops they remove with
//! [`Op::Nop`] instead of renumbering.

use super::simd::SimdOp;
use crate::decoder::AtomicOp;

/// The result of the op at this index in [`Block::ops`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Value(pub u32);

impl Value {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Operation width. 32-bit ops ignore the upper half of their inputs and zero-extend their
/// result, as A64 `W` registers do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W {
    W32,
    W64,
}

impl W {
    pub fn of(sf: bool) -> Self {
        if sf { W::W64 } else { W::W32 }
    }

    pub fn bits(self) -> u32 {
        match self {
            W::W32 => 32,
            W::W64 => 64,
        }
    }

    pub fn mask(self) -> u64 {
        match self {
            W::W32 => u32::MAX as u64,
            W::W64 => u64::MAX,
        }
    }
}

/// The host instruction whose flags an [`Op::Compare`] tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp {
    /// The flags of `AddFlags(a, b)`.
    Add,
    /// The flags of `SubFlags(a, b)`.
    Sub,
    /// The flags of `LogicFlags(a & b)`.
    Test,
}

impl Cmp {
    /// Whether one host condition code computes A64 condition `cond` on these flags. Only
    /// subtraction has A64's inverted carry, which `HI` and `LS` need.
    pub fn fuses(self, cond: u8) -> bool {
        cond < 14 && (self == Cmp::Sub || !matches!(cond, 8 | 9))
    }
}

/// What an op produces. Every type but `U128` and `Void` lives in a general-purpose host
/// register; `U128` lives in a vector register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ty {
    /// 0 or 1.
    U1,
    U32,
    U64,
    U128,
    /// NZCV in bits 3..0.
    Nzcv,
    Void,
}

/// Guest state an op can read or write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    /// `X0`..`X30`.
    X(u8),
    Sp,
    Nzcv,
    V(u8),
    Fpcr,
    Fpsr,
    Tpidr,
    Tpidrro,
}

impl Loc {
    pub const COUNT: usize = 69;

    pub fn index(self) -> usize {
        match self {
            Loc::X(n) => n as usize,
            Loc::Sp => 31,
            Loc::Nzcv => 32,
            Loc::V(n) => 33 + n as usize,
            Loc::Fpcr => 65,
            Loc::Fpsr => 66,
            Loc::Tpidr => 67,
            Loc::Tpidrro => 68,
        }
    }

    pub fn ty(self) -> Ty {
        match self {
            Loc::X(_) | Loc::Sp | Loc::Tpidr | Loc::Tpidrro => Ty::U64,
            Loc::Nzcv => Ty::Nzcv,
            Loc::V(_) => Ty::U128,
            Loc::Fpcr | Loc::Fpsr => Ty::U32,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Removed by a pass.
    Nop,
    Const(u64),
    Get(Loc),
    Set(Loc, Value),

    Add(W, Value, Value),
    Sub(W, Value, Value),
    /// `a + b + C`, with C taken from bit 1 of the `Nzcv` operand.
    Adc(W, Value, Value, Value),
    /// `a + !b + C`.
    Sbc(W, Value, Value, Value),
    /// The NZCV of `Add`, `Sub`, `Adc` and `Sbc`.
    AddFlags(W, Value, Value),
    SubFlags(W, Value, Value),
    AdcFlags(W, Value, Value, Value),
    SbcFlags(W, Value, Value, Value),
    /// N and Z of a logical result; C and V clear.
    LogicFlags(W, Value),

    And(W, Value, Value),
    Or(W, Value, Value),
    Xor(W, Value, Value),
    Not(W, Value),
    /// Shifts take the amount modulo the width.
    Lsl(W, Value, Value),
    Lsr(W, Value, Value),
    Asr(W, Value, Value),
    Ror(W, Value, Value),
    /// Zero-extends the low `8 << size` bits.
    ZeroExt(u8, Value),
    /// Sign-extends the low `8 << size` bits to the width.
    SignExt(W, u8, Value),

    Mul(W, Value, Value),
    UMulH(Value, Value),
    SMulH(Value, Value),
    /// A64 division: by zero gives 0, `MIN / -1` gives `MIN`.
    UDiv(W, Value, Value),
    SDiv(W, Value, Value),
    Clz(W, Value),
    Rbit(W, Value),
    /// Reverses the bytes of the whole width.
    Rev(W, Value),
    /// Reverses the bytes of each halfword.
    Rev16(W, Value),
    /// Reverses the bytes of each word of a 64-bit value.
    Rev32(Value),

    /// `if c != 0 { a } else { b }`.
    Select(W, Value, Value, Value),
    /// `ConditionHolds(cond)` on an `Nzcv` value.
    TestCond(u8, Value),
    /// `ConditionHolds(cond)` on the flags the host instruction of `Cmp` sets on `a` and `b`,
    /// for a `cond` that [`Cmp::fuses`].
    Compare(Cmp, u8, W, Value, Value),
    IsZero(W, Value),
    TestBit(Value, u8),

    /// Zero-extended load of `1 << size` bytes (`size` 0..=3).
    Load {
        size: u8,
        addr: Value,
        pc: u64,
    },
    /// Zero-extended `U128` load of `1 << size` bytes (`size` 0..=4).
    LoadVec {
        size: u8,
        addr: Value,
        pc: u64,
    },
    Store {
        size: u8,
        addr: Value,
        value: Value,
        pc: u64,
    },
    StoreVec {
        size: u8,
        addr: Value,
        value: Value,
        pc: u64,
    },
    /// LSE `LD<op>`/`SWP` of `1 << size` bytes: the zero-extended old value. Faults with
    /// `CpuFault::Alignment` unless `addr` is aligned to the size.
    AtomicRmw {
        op: AtomicOp,
        size: u8,
        addr: Value,
        value: Value,
        pc: u64,
    },
    /// `LDXR` and `LDXP` of `1 << size` bytes: the zero-extended value, with a reservation for it
    /// in this core's `ExclusiveMonitor` slot. Aligned like `AtomicRmw`. A 16-byte load gives the low
    /// half and leaves the high half for `High`.
    LoadExclusive {
        size: u8,
        addr: Value,
        pc: u64,
    },
    /// `STXR` and the 32-bit `STXP`: stores `value` when the reservation still covers `addr`
    /// and memory still holds the loaded value, and always drops the reservation. 0 when the store
    /// happened, 1 otherwise.
    StoreExclusive {
        size: u8,
        addr: Value,
        value: Value,
        pc: u64,
    },
    /// The 64-bit `STXP`: `StoreExclusive` of `high:low`.
    StoreExclusive128 {
        addr: Value,
        low: Value,
        high: Value,
        pc: u64,
    },
    /// The 64-bit `CASP`: the low half of the old `high:low` value, leaving the high half for `High`.
    /// Aligned like `AtomicRmw`.
    Cas128 {
        addr: Value,
        expected_low: Value,
        expected_high: Value,
        new_low: Value,
        new_high: Value,
        pc: u64,
    },
    /// The high half of the last 16-byte `LoadExclusive` or `Cas128`, which must come right before it.
    High,
    /// `CAS`: the zero-extended old value, aligned like `AtomicRmw`.
    Cas {
        size: u8,
        addr: Value,
        expected: Value,
        new: Value,
        pc: u64,
    },
    /// `DC ZVA`: zeroes the 64 bytes at `addr`, which is 64-byte aligned.
    ZeroBlock {
        addr: Value,
        pc: u64,
    },
    /// `CNTPCT_EL0`.
    Ticks,
    /// `DMB`/`DSB`.
    Fence,
    /// One instruction of the SIMD/FP data-processing group, run by the interpreter's
    /// `simd::execute` on the live `ThreadContext`. It may read and write any guest state, and
    /// faults with `CpuFault::Undefined` on an unallocated encoding.
    CallSimd {
        word: u32,
        pc: u64,
    },
    /// A SIMD/FP operation the JIT lowers natively, defined by `jit::simd`.
    Simd(SimdOp),
}

impl Op {
    pub fn ty(&self) -> Ty {
        match *self {
            Op::Nop
            | Op::Set(..)
            | Op::Store { .. }
            | Op::StoreVec { .. }
            | Op::ZeroBlock { .. }
            | Op::Fence
            | Op::CallSimd { .. } => Ty::Void,
            Op::Get(loc) => loc.ty(),
            Op::LoadVec { .. } => Ty::U128,
            Op::AddFlags(..)
            | Op::SubFlags(..)
            | Op::AdcFlags(..)
            | Op::SbcFlags(..)
            | Op::LogicFlags(..) => Ty::Nzcv,
            Op::TestCond(..) | Op::Compare(..) | Op::IsZero(..) | Op::TestBit(..) => Ty::U1,
            Op::Simd(s) => s.ty(),
            _ => Ty::U64,
        }
    }

    /// Whether the op can leave the block early, with a guest fault.
    pub fn may_exit(&self) -> bool {
        matches!(
            self,
            Op::Load { .. }
                | Op::LoadVec { .. }
                | Op::Store { .. }
                | Op::StoreVec { .. }
                | Op::AtomicRmw { .. }
                | Op::Cas { .. }
                | Op::LoadExclusive { .. }
                | Op::StoreExclusive { .. }
                | Op::StoreExclusive128 { .. }
                | Op::Cas128 { .. }
                | Op::ZeroBlock { .. }
                | Op::CallSimd { .. }
        )
    }

    /// Whether removing the op when nothing uses its value changes nothing.
    pub fn is_pure(&self) -> bool {
        match self {
            Op::Simd(s) => s.is_pure(),
            _ => !matches!(self, Op::Set(..) | Op::Fence) && !self.may_exit(),
        }
    }

    /// Calls `f` on every operand.
    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Op::Nop
            | Op::Const(_)
            | Op::Get(_)
            | Op::Ticks
            | Op::Fence
            | Op::High
            | Op::CallSimd { .. } => {}
            Op::Simd(s) => s.for_each_arg(&mut f),
            Op::StoreExclusive128 {
                addr, low, high, ..
            } => {
                f(addr);
                f(low);
                f(high);
            }
            Op::Cas128 {
                addr,
                expected_low,
                expected_high,
                new_low,
                new_high,
                ..
            } => {
                f(addr);
                f(expected_low);
                f(expected_high);
                f(new_low);
                f(new_high);
            }
            Op::Set(_, a)
            | Op::LogicFlags(_, a)
            | Op::Not(_, a)
            | Op::ZeroExt(_, a)
            | Op::SignExt(_, _, a)
            | Op::Clz(_, a)
            | Op::Rbit(_, a)
            | Op::Rev(_, a)
            | Op::Rev16(_, a)
            | Op::Rev32(a)
            | Op::TestCond(_, a)
            | Op::IsZero(_, a)
            | Op::TestBit(a, _)
            | Op::Load { addr: a, .. }
            | Op::LoadVec { addr: a, .. }
            | Op::LoadExclusive { addr: a, .. }
            | Op::ZeroBlock { addr: a, .. } => f(a),
            Op::Add(_, a, b)
            | Op::Sub(_, a, b)
            | Op::AddFlags(_, a, b)
            | Op::SubFlags(_, a, b)
            | Op::Compare(_, _, _, a, b)
            | Op::And(_, a, b)
            | Op::Or(_, a, b)
            | Op::Xor(_, a, b)
            | Op::Lsl(_, a, b)
            | Op::Lsr(_, a, b)
            | Op::Asr(_, a, b)
            | Op::Ror(_, a, b)
            | Op::Mul(_, a, b)
            | Op::UMulH(a, b)
            | Op::SMulH(a, b)
            | Op::UDiv(_, a, b)
            | Op::SDiv(_, a, b)
            | Op::Store {
                addr: a, value: b, ..
            }
            | Op::StoreVec {
                addr: a, value: b, ..
            }
            | Op::AtomicRmw {
                addr: a, value: b, ..
            }
            | Op::StoreExclusive {
                addr: a, value: b, ..
            } => {
                f(a);
                f(b);
            }
            Op::Adc(_, a, b, c)
            | Op::Sbc(_, a, b, c)
            | Op::AdcFlags(_, a, b, c)
            | Op::SbcFlags(_, a, b, c)
            | Op::Select(_, a, b, c)
            | Op::Cas {
                addr: a,
                expected: b,
                new: c,
                ..
            } => {
                f(a);
                f(b);
                f(c);
            }
        }
    }

    pub fn args(&self) -> impl Iterator<Item = Value> {
        let mut args = [Value(0); 5];
        let mut len = 0;
        let mut op = *self;
        op.for_each_arg(|v| {
            args[len] = *v;
            len += 1;
        });
        args.into_iter().take(len)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Terminal {
    Jump(u64),
    JumpIndirect(Value),
    /// `if cond != 0 { taken } else { not_taken }`.
    Branch {
        cond: Value,
        taken: u64,
        not_taken: u64,
    },
}

impl Terminal {
    pub fn for_each_arg(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            Terminal::JumpIndirect(v) | Terminal::Branch { cond: v, .. } => f(v),
            Terminal::Jump(_) => {}
        }
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    pub ops: Vec<Op>,
    pub terminal: Terminal,
}

impl Block {
    pub fn push(&mut self, op: Op) -> Value {
        self.ops.push(op);
        Value(self.ops.len() as u32 - 1)
    }

    pub fn constant(&self, v: Value) -> Option<u64> {
        match self.ops[v.index()] {
            Op::Const(c) => Some(c),
            _ => None,
        }
    }
}
