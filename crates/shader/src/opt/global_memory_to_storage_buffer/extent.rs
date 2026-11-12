//! Proves where a program's stores to one storage buffer can land relative to its pointer word.
//!
//! Anything not provably affine in the invocation and workgroup ids with constant non-negative
//! strides (a value loaded from memory or read from another constant buffer word, a
//! loop-carried index, a signed or 64-bit computation) leaves the extent unknown.

use rustc_hash::{FxHashMap, FxHashSet};

use super::track::{cbuf_word, track_low_address};
use super::{BufferAddr, imm_u32, inst_of, store_size};
use crate::info::StorageWriteExtent;
use crate::ir::{InstId, Opcode, Program, Value};

const CONST_LIMIT: i64 = 1 << 40;
const COEF_LIMIT: u64 = 1 << 32;
const MAX_DEPTH: u32 = 256;

/// An upper bound of an unsigned 32-bit value for every invocation of a dispatch:
/// `c + sum(coef[i] * id[i])` over the local invocation id xyz (0-2) and workgroup id xyz
/// (3-5), plus `base` times the storage buffer's pointer word. An exact bound is the value
/// itself modulo 2^32, so a negative `c` means the value can wrap below zero. An inexact bound
/// is only an upper bound, with `c >= 0`.
#[derive(Clone, Copy, Debug)]
struct IdBound {
    c: i64,
    coef: [u64; 6],
    base: u32,
    exact: bool,
}

const ZERO: IdBound = IdBound {
    c: 0,
    coef: [0; 6],
    base: 0,
    exact: true,
};

impl IdBound {
    fn is_constant(&self) -> bool {
        self.base == 0 && self.coef.iter().all(|&c| c == 0)
    }

    fn has_group_terms(&self) -> bool {
        self.coef[3..].iter().any(|&c| c != 0)
    }

    fn usable(&self) -> bool {
        self.base == 0 && self.c >= 0
    }
}

fn constant(c: i64) -> Option<IdBound> {
    Some(IdBound { c, ..ZERO })
}

fn inexact(mut value: IdBound) -> Option<IdBound> {
    value.exact = false;
    Some(value)
}

fn checked(value: IdBound) -> Option<IdBound> {
    let in_range = (-CONST_LIMIT..=CONST_LIMIT).contains(&value.c)
        && value.base <= 1
        && value.coef.iter().all(|&c| c < COEF_LIMIT);
    in_range.then_some(value)
}

fn sum(lhs: &IdBound, rhs: &IdBound) -> Option<IdBound> {
    if (!lhs.exact && rhs.c < 0) || (!rhs.exact && lhs.c < 0) {
        return None;
    }
    let mut result = IdBound {
        c: lhs.c + rhs.c,
        coef: [0; 6],
        base: lhs.base + rhs.base,
        exact: lhs.exact && rhs.exact,
    };
    for (i, coef) in result.coef.iter_mut().enumerate() {
        *coef = lhs.coef[i] + rhs.coef[i];
    }
    checked(result)
}

/// Component-wise maximum: a bound of either value. A value that can wrap below zero is larger
/// than any bound of the other.
fn max(lhs: &IdBound, rhs: &IdBound) -> Option<IdBound> {
    if lhs.base != rhs.base || lhs.c < 0 || rhs.c < 0 {
        return None;
    }
    let mut result = IdBound {
        c: lhs.c.max(rhs.c),
        coef: [0; 6],
        base: lhs.base,
        exact: false,
    };
    for (i, coef) in result.coef.iter_mut().enumerate() {
        *coef = lhs.coef[i].max(rhs.coef[i]);
    }
    Some(result)
}

/// `value * factor`, where an inexact factor only bounds the multiplier: then the value must
/// not wrap below zero.
fn scale(value: &IdBound, factor: u64, exact_factor: bool) -> Option<IdBound> {
    if value.base != 0 || factor >= COEF_LIMIT || (!exact_factor && value.c < 0) {
        return None;
    }
    let magnitude = value.c.unsigned_abs();
    if magnitude != 0 && factor > CONST_LIMIT as u64 / magnitude {
        return None;
    }
    let mut result = IdBound {
        c: value.c * factor as i64,
        coef: [0; 6],
        base: 0,
        exact: value.exact && exact_factor,
    };
    for (i, coef) in result.coef.iter_mut().enumerate() {
        if value.coef[i] != 0 && factor > (COEF_LIMIT - 1) / value.coef[i] {
            return None;
        }
        *coef = value.coef[i] * factor;
    }
    checked(result)
}

/// `value >> shift`, rounded up term by term.
fn shift_down(value: &IdBound, shift: u32) -> Option<IdBound> {
    if value.base != 0 || value.c < 0 || shift >= 32 {
        return None;
    }
    let up = |x: u64| (x + (1 << shift) - 1) >> shift;
    let mut result = IdBound {
        c: up(value.c as u64) as i64,
        coef: [0; 6],
        base: 0,
        exact: false,
    };
    for (i, coef) in result.coef.iter_mut().enumerate() {
        *coef = up(value.coef[i]);
    }
    Some(result)
}

/// Store extents of one storage buffer, widened by every store added.
pub(super) struct StoreExtentAnalysis {
    buffer: BufferAddr,
    workgroup: [u32; 3],
    memo: FxHashMap<InstId, Option<IdBound>>,
    active: FxHashSet<InstId>,
    guard: Option<IdBound>,
    guard_failed: bool,
}

impl StoreExtentAnalysis {
    pub(super) fn new(buffer: BufferAddr, workgroup: [u32; 3]) -> StoreExtentAnalysis {
        StoreExtentAnalysis {
            buffer,
            workgroup,
            memo: FxHashMap::default(),
            active: FxHashSet::default(),
            guard: None,
            guard_failed: false,
        }
    }

    /// Merges the extent of one store or atomic into `extent`.
    pub(super) fn add_store(
        &mut self,
        prog: &Program,
        inst: InstId,
        extent: &mut StorageWriteExtent,
    ) {
        match self.store_extent(prog, inst) {
            Some(store) => extent.merge(&store),
            None => *extent = StorageWriteExtent::default(),
        }
    }

    fn store_extent(&mut self, prog: &Program, inst: InstId) -> Option<StorageWriteExtent> {
        let addr = track_low_address(prog, inst)?;
        let low = self.evaluate(prog, addr.value, 0)?;
        if low.base != 1 || self.guard_failed {
            return None;
        }
        let start = low.c + i64::from(addr.imm_offset);
        let end = start + i64::from(store_size(prog.op(inst)));
        if start < 0 || end > i64::from(u32::MAX) {
            return None;
        }
        let mut result = StorageWriteExtent {
            bounded: true,
            const_end: end as u32,
            ..StorageWriteExtent::default()
        };
        for i in 0..3 {
            if low.coef[i] >= COEF_LIMIT || low.coef[i + 3] >= COEF_LIMIT {
                return None;
            }
            result.local_stride[i] = low.coef[i] as u32;
            result.group_stride[i] = low.coef[i + 3] as u32;
        }
        if let Some(guard) = self.guard {
            if guard.c > 0xffff {
                return None;
            }
            result.guarded = true;
            result.guard_const = guard.c.max(0) as u32;
            for i in 0..3 {
                if guard.coef[i] > 0xffff || guard.coef[i + 3] > 0xffff {
                    return None;
                }
                result.guard_local_stride[i] = guard.coef[i] as u32;
                result.guard_group_stride[i] = guard.coef[i + 3] as u32;
            }
        }
        Some(result)
    }

    /// The value's bound when only its local-id terms vary (the workgroup size is fixed).
    fn static_max(&self, value: &IdBound) -> Option<u64> {
        if value.base != 0 || value.has_group_terms() || value.c < 0 {
            return None;
        }
        let mut total = value.c as u64;
        for i in 0..3 {
            if self.workgroup[i] == 0 {
                return None;
            }
            total =
                total.wrapping_add(value.coef[i].wrapping_mul(u64::from(self.workgroup[i] - 1)));
        }
        Some(total)
    }

    /// Requires a value to stay below 2^16 for the bound to hold.
    fn guard_value(&mut self, value: &IdBound) {
        if !value.usable() {
            self.guard_failed = true;
            return;
        }
        let merged = match &self.guard {
            None => Some(*value),
            Some(guard) => max(guard, value),
        };
        match merged {
            Some(merged) => self.guard = Some(merged),
            None => self.guard_failed = true,
        }
    }

    fn bit_field_extract(&mut self, prog: &Program, inst: InstId, depth: u32) -> Option<IdBound> {
        let offset = imm_u32(prog, prog.arg(inst, 1))?;
        let count = imm_u32(prog, prog.arg(inst, 2))?;
        if offset >= 32 || count == 0 || count > 32 {
            return None;
        }
        let source = prog.resolve(prog.arg(inst, 0));
        if source.is_immediate() {
            let field = (u64::from(source.imm_u32()?) >> offset) & ((1u64 << count) - 1);
            return constant(field as i64);
        }
        let source = self.evaluate(prog, source, depth + 1);
        let field_max = if count <= 24 {
            inexact(IdBound {
                c: (1i64 << count) - 1,
                ..ZERO
            })
        } else {
            None
        };
        let Some(source) = source.filter(IdBound::usable) else {
            return field_max;
        };
        if let Some(static_max) = self.static_max(&source)
            && static_max >> offset == 0
        {
            return constant(0);
        }
        if offset == 0 {
            return if count == 32 {
                Some(source)
            } else {
                inexact(source)
            };
        }
        if offset >= 16 {
            self.guard_value(&source);
            return constant(0);
        }
        shift_down(&source, offset)
    }

    fn evaluate(&mut self, prog: &Program, value: Value, depth: u32) -> Option<IdBound> {
        let value = prog.resolve(value);
        let Some(id) = value.inst() else {
            return match value {
                Value::U32(v) => constant(i64::from(v as i32)),
                _ => None,
            };
        };
        if depth > MAX_DEPTH {
            return None;
        }
        if let Some(&memo) = self.memo.get(&id) {
            return memo;
        }
        if !self.active.insert(id) {
            return None;
        }
        let result = self.evaluate_inst(prog, id, depth);
        self.active.remove(&id);
        self.memo.insert(id, result);
        result
    }

    fn evaluate_inst(&mut self, prog: &Program, id: InstId, depth: u32) -> Option<IdBound> {
        if let Some(word) = cbuf_word(prog, id) {
            return (word == self.buffer).then_some(IdBound { base: 1, ..ZERO });
        }
        let inst = prog.inst(id);
        let arg = |this: &mut Self, index: usize| this.evaluate(prog, inst.arg(index), depth + 1);
        match inst.op {
            Opcode::CompositeExtractU32x3 => {
                let vector = inst_of(prog, inst.arg(0))?;
                let element = imm_u32(prog, inst.arg(1)).filter(|&e| e <= 2)? as usize;
                let mut result = ZERO;
                match prog.op(vector) {
                    Opcode::LocalInvocationId => result.coef[element] = 1,
                    Opcode::WorkgroupId => result.coef[3 + element] = 1,
                    _ => return None,
                }
                Some(result)
            }
            Opcode::IAdd32 => {
                let lhs = arg(self, 0)?;
                let rhs = arg(self, 1)?;
                sum(&lhs, &rhs)
            }
            Opcode::ISub32 => {
                let lhs = arg(self, 0)?;
                let rhs = arg(self, 1)?;
                if !rhs.is_constant() || !rhs.exact {
                    return None;
                }
                sum(&lhs, &IdBound { c: -rhs.c, ..ZERO })
            }
            Opcode::IMul32 => {
                let lhs = arg(self, 0)?;
                let rhs = arg(self, 1)?;
                if rhs.is_constant() && rhs.c >= 0 {
                    return scale(&lhs, rhs.c as u64, rhs.exact);
                }
                if lhs.is_constant() && lhs.c >= 0 {
                    return scale(&rhs, lhs.c as u64, lhs.exact);
                }
                None
            }
            Opcode::ShiftLeftLogical32 => {
                let shift = imm_u32(prog, inst.arg(1)).filter(|&s| s < 32)?;
                let lhs = arg(self, 0)?;
                scale(&lhs, 1 << shift, true)
            }
            Opcode::ShiftRightLogical32 => {
                let shift = imm_u32(prog, inst.arg(1))?;
                let lhs = arg(self, 0)?;
                shift_down(&lhs, shift)
            }
            Opcode::BitwiseAnd32 => {
                let lhs = arg(self, 0);
                let rhs = arg(self, 1);
                for side in [rhs, lhs].into_iter().flatten() {
                    if side.usable() && side.is_constant() && side.c <= 0xff_ffff {
                        return inexact(side);
                    }
                }
                [lhs, rhs]
                    .into_iter()
                    .flatten()
                    .find(IdBound::usable)
                    .and_then(inexact)
            }
            Opcode::BitwiseOr32 | Opcode::BitwiseXor32 | Opcode::UMax32 => {
                let lhs = arg(self, 0)?;
                let rhs = arg(self, 1)?;
                if !lhs.usable() || !rhs.usable() {
                    return None;
                }
                sum(&lhs, &rhs).and_then(inexact)
            }
            Opcode::UMin32 => {
                let lhs = arg(self, 0);
                let rhs = arg(self, 1);
                [rhs, lhs]
                    .into_iter()
                    .flatten()
                    .find(IdBound::usable)
                    .and_then(inexact)
            }
            Opcode::BitFieldUExtract => self.bit_field_extract(prog, id, depth),
            Opcode::BitFieldInsert => {
                let offset = imm_u32(prog, inst.arg(2)).filter(|&o| o < 32)?;
                imm_u32(prog, inst.arg(3))?;
                let base = arg(self, 0)?;
                let insert = arg(self, 1)?;
                if !base.usable() || !insert.usable() {
                    return None;
                }
                let shifted = scale(&insert, 1 << offset, true)?;
                sum(&base, &shifted).and_then(inexact)
            }
            Opcode::SelectU32 => {
                let lhs = arg(self, 1)?;
                let rhs = arg(self, 2)?;
                max(&lhs, &rhs)
            }
            Opcode::Phi => {
                let mut result: Option<IdBound> = None;
                for index in 0..inst.num_args() {
                    let incoming = arg(self, index)?;
                    result = Some(match result {
                        None => incoming,
                        Some(current) => max(&current, &incoming)?,
                    });
                }
                result
            }
            _ => None,
        }
    }
}
