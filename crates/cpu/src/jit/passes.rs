//! IR passes. Each one rewrites a [`Block`] in place and keeps its meaning.

use super::ir::{Block, Cmp, Loc, Op, Terminal, Value, W};
use crate::interpreter::condition_holds;

/// Which passes run; tests turn them off one at a time to check each against the others.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Passes {
    pub get_set: bool,
    pub fold: bool,
    pub fuse: bool,
    pub lazy: bool,
    pub dce: bool,
}

impl Passes {
    pub const ALL: Self = Self {
        get_set: true,
        fold: true,
        fuse: true,
        lazy: true,
        dce: true,
    };

    pub fn run(self, block: &mut Block) {
        if self.get_set {
            get_set_elimination(block);
        }
        if self.fold {
            fold_constants(block);
        }
        if self.fuse {
            fuse_conditions(block);
        }
        if self.lazy {
            lazy_flags(block);
        }
        if self.dce {
            dead_code_elimination(block);
        }
    }
}

/// Rewrites every operand through `replace`, which maps a value to the one that stands for it.
fn resolve(replace: &[Value], v: &mut Value) {
    *v = replace[v.index()];
}

/// Forwards known guest state to later reads and drops writes a later write of the same
/// location makes dead. An op that can exit commits every pending write first, so the context is
/// exact wherever the block can leave. `CallSimd` can also write any location, so reads after it
/// load from the context again.
pub(super) fn get_set_elimination(block: &mut Block) {
    let mut replace: Vec<Value> = (0..block.ops.len() as u32).map(Value).collect();
    let mut known: [Option<Value>; Loc::COUNT] = [None; Loc::COUNT];
    let mut pending: [Option<usize>; Loc::COUNT] = [None; Loc::COUNT];
    for i in 0..block.ops.len() {
        let mut op = block.ops[i];
        op.for_each_arg(|v| resolve(&replace, v));
        match op {
            Op::Get(loc) => match known[loc.index()] {
                Some(v) => {
                    replace[i] = v;
                    op = Op::Nop;
                }
                None => known[loc.index()] = Some(Value(i as u32)),
            },
            Op::Set(loc, v) => {
                if let Some(dead) = pending[loc.index()].replace(i) {
                    block.ops[dead] = Op::Nop;
                }
                known[loc.index()] = Some(v);
            }
            Op::CallSimd { .. } => {
                pending = [None; Loc::COUNT];
                known = [None; Loc::COUNT];
            }
            _ if op.may_exit() => pending = [None; Loc::COUNT],
            _ => {}
        }
        block.ops[i] = op;
    }
    block.terminal.for_each_arg(|v| resolve(&replace, v));
}

const fn sign_extend(value: u64, size: u8) -> u64 {
    let shift = 64 - (8u32 << size);
    (((value << shift) as i64) >> shift) as u64
}

const fn nzcv(n: bool, z: bool, c: bool, v: bool) -> u64 {
    (n as u64) << 3 | (z as u64) << 2 | (c as u64) << 1 | v as u64
}

/// `AddWithCarry` from the ARM ARM: the result and its NZCV.
fn add_with_carry(w: W, a: u64, b: u64, carry: u64) -> (u64, u64) {
    let (a, b) = (a & w.mask(), b & w.mask());
    let wide = a as u128 + b as u128 + carry as u128;
    let r = wide as u64 & w.mask();
    let top = w.bits() - 1;
    let sign = |x: u64| x >> top & 1 != 0;
    let carry_out = wide >> w.bits() != 0;
    let overflow = sign(a) == sign(b) && sign(r) != sign(a);
    (r, nzcv(sign(r), r == 0, carry_out, overflow))
}

/// The value of a pure op whose operands are all known, or `None`.
pub(super) fn evaluate(op: &Op, k: impl Fn(Value) -> Option<u64>) -> Option<u64> {
    let c = |f: u64| f >> 1 & 1;
    let shift = |w: W, b: u64| (b & (w.bits() as u64 - 1)) as u32;
    Some(match *op {
        Op::Add(w, a, b) => k(a)?.wrapping_add(k(b)?) & w.mask(),
        Op::Sub(w, a, b) => k(a)?.wrapping_sub(k(b)?) & w.mask(),
        Op::Adc(w, a, b, f) => add_with_carry(w, k(a)?, k(b)?, c(k(f)?)).0,
        Op::Sbc(w, a, b, f) => add_with_carry(w, k(a)?, !k(b)?, c(k(f)?)).0,
        Op::AddFlags(w, a, b) => add_with_carry(w, k(a)?, k(b)?, 0).1,
        Op::SubFlags(w, a, b) => add_with_carry(w, k(a)?, !k(b)?, 1).1,
        Op::AdcFlags(w, a, b, f) => add_with_carry(w, k(a)?, k(b)?, c(k(f)?)).1,
        Op::SbcFlags(w, a, b, f) => add_with_carry(w, k(a)?, !k(b)?, c(k(f)?)).1,
        Op::LogicFlags(w, a) => {
            let a = k(a)? & w.mask();
            nzcv(a >> (w.bits() - 1) != 0, a == 0, false, false)
        }
        Op::And(w, a, b) => k(a)? & k(b)? & w.mask(),
        Op::Or(w, a, b) => (k(a)? | k(b)?) & w.mask(),
        Op::Xor(w, a, b) => (k(a)? ^ k(b)?) & w.mask(),
        Op::Not(w, a) => !k(a)? & w.mask(),
        Op::Lsl(w, a, b) => (k(a)? << shift(w, k(b)?)) & w.mask(),
        Op::Lsr(w, a, b) => (k(a)? & w.mask()) >> shift(w, k(b)?),
        Op::Asr(w, a, b) => match w {
            W::W32 => ((k(a)? as i32) >> shift(w, k(b)?)) as u32 as u64,
            W::W64 => ((k(a)? as i64) >> shift(w, k(b)?)) as u64,
        },
        Op::Ror(w, a, b) => match w {
            W::W32 => (k(a)? as u32).rotate_right(shift(w, k(b)?)) as u64,
            W::W64 => k(a)?.rotate_right(shift(w, k(b)?)),
        },
        Op::ZeroExt(size, a) => {
            let bits = 8u32 << size;
            if bits == 64 {
                k(a)?
            } else {
                k(a)? & ((1 << bits) - 1)
            }
        }
        Op::SignExt(w, size, a) => sign_extend(k(a)?, size) & w.mask(),
        Op::Mul(w, a, b) => k(a)?.wrapping_mul(k(b)?) & w.mask(),
        Op::UMulH(a, b) => ((k(a)? as u128 * k(b)? as u128) >> 64) as u64,
        Op::SMulH(a, b) => ((k(a)? as i64 as i128 * k(b)? as i64 as i128) >> 64) as u64,
        Op::UDiv(w, a, b) => (k(a)? & w.mask())
            .checked_div(k(b)? & w.mask())
            .unwrap_or(0),
        Op::SDiv(w, a, b) => {
            let (a, b) = (k(a)?, k(b)?);
            match w {
                W::W32 if b as u32 == 0 => 0,
                W::W32 => (a as i32).wrapping_div(b as i32) as u32 as u64,
                W::W64 if b == 0 => 0,
                W::W64 => (a as i64).wrapping_div(b as i64) as u64,
            }
        }
        Op::Clz(w, a) => match w {
            W::W32 => (k(a)? as u32).leading_zeros() as u64,
            W::W64 => k(a)?.leading_zeros() as u64,
        },
        Op::Rbit(w, a) => match w {
            W::W32 => (k(a)? as u32).reverse_bits() as u64,
            W::W64 => k(a)?.reverse_bits(),
        },
        Op::Rev(w, a) => match w {
            W::W32 => (k(a)? as u32).swap_bytes() as u64,
            W::W64 => k(a)?.swap_bytes(),
        },
        Op::Rev16(w, a) => {
            let a = k(a)?;
            (((a & 0x00ff_00ff_00ff_00ff) << 8) | ((a >> 8) & 0x00ff_00ff_00ff_00ff)) & w.mask()
        }
        Op::Rev32(a) => k(a)?.swap_bytes().rotate_left(32),
        Op::Select(w, cond, a, b) => (if k(cond)? != 0 { k(a)? } else { k(b)? }) & w.mask(),
        Op::TestCond(cond, f) => condition_holds(cond, k(f)? as u32) as u64,
        Op::Compare(kind, cond, w, a, b) => {
            condition_holds(cond, flags_of(kind, w, k(a)?, k(b)?) as u32) as u64
        }
        Op::IsZero(w, a) => (k(a)? & w.mask() == 0) as u64,
        Op::TestBit(a, bit) => k(a)? >> bit & 1,
        _ => return None,
    })
}

/// The NZCV the host instruction of `kind` sets on `a` and `b`.
pub(super) fn flags_of(kind: Cmp, w: W, a: u64, b: u64) -> u64 {
    match kind {
        Cmp::Add => add_with_carry(w, a, b, 0).1,
        Cmp::Sub => add_with_carry(w, a, !b, 1).1,
        Cmp::Test => {
            let r = a & b & w.mask();
            nzcv(r >> (w.bits() - 1) != 0, r == 0, false, false)
        }
    }
}

/// Evaluates pure ops on constants, and simplifies the terminal when its operand is known.
pub(super) fn fold_constants(block: &mut Block) {
    for i in 0..block.ops.len() {
        let op = block.ops[i];
        if let Some(value) = evaluate(&op, |v| block.constant(v)) {
            block.ops[i] = Op::Const(value);
        }
    }
    block.terminal = match block.terminal {
        Terminal::JumpIndirect(v) => match block.constant(v) {
            Some(target) => Terminal::Jump(target),
            None => block.terminal,
        },
        Terminal::Branch {
            cond,
            taken,
            not_taken,
        } => match block.constant(cond) {
            Some(c) => Terminal::Jump(if c != 0 { taken } else { not_taken }),
            None => block.terminal,
        },
        t => t,
    };
}

/// Turns each `TestCond` on the flags of an add, a subtract or a logical result into a
/// `Compare` on their operands, which the backend lowers to one host `add`, `cmp` or `test`
/// and a condition code. The flags op stays only while something else, such as the NZCV write
/// at the block's end, still reads it.
pub(super) fn fuse_conditions(block: &mut Block) {
    for i in 0..block.ops.len() {
        let Op::TestCond(cond, f) = block.ops[i] else {
            continue;
        };
        if let Some((kind, w, a, b)) = compared(block, f)
            && kind.fuses(cond)
        {
            block.ops[i] = Op::Compare(kind, cond, w, a, b);
        }
    }
}

/// The host instruction and operands whose flags `f` is, when it is the NZCV of an add, a
/// subtract or a logical result.
fn compared(block: &Block, f: Value) -> Option<(Cmp, W, Value, Value)> {
    Some(match block.ops[f.index()] {
        Op::AddFlags(w, a, b) => (Cmp::Add, w, a, b),
        Op::SubFlags(w, a, b) => (Cmp::Sub, w, a, b),
        Op::LogicFlags(w, r) => match block.ops[r.index()] {
            Op::And(aw, a, b) if aw == w => (Cmp::Test, w, a, b),
            _ => (Cmp::Test, w, r, r),
        },
        _ => return None,
    })
}

/// Writes NZCV lazily: a `Set` of flags that nothing else reads becomes `SetLazyNzcv` on the
/// flag op's operands, so the block stores those instead of packing the flags, and DCE removes
/// the flags op.
pub(super) fn lazy_flags(block: &mut Block) {
    let mut uses = vec![0u32; block.ops.len()];
    block
        .ops
        .iter()
        .flat_map(Op::args)
        .for_each(|v| uses[v.index()] += 1);
    block.terminal.for_each_arg(|v| uses[v.index()] += 1);
    for i in 0..block.ops.len() {
        let Op::Set(Loc::Nzcv, f) = block.ops[i] else {
            continue;
        };
        if uses[f.index()] == 1
            && let Some((kind, w, a, b)) = compared(block, f)
        {
            block.ops[i] = Op::SetLazyNzcv(kind, w, a, b);
        }
    }
}

/// Removes pure ops whose value nothing uses.
pub(super) fn dead_code_elimination(block: &mut Block) {
    let mut used = vec![false; block.ops.len()];
    block.terminal.for_each_arg(|v| used[v.index()] = true);
    for i in (0..block.ops.len()).rev() {
        let op = &mut block.ops[i];
        if op.is_pure() && !used[i] {
            *op = Op::Nop;
            continue;
        }
        op.for_each_arg(|v| used[v.index()] = true);
    }
}
