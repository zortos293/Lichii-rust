//! Handle programs for bindless handles no constant buffer word gives (BuildHandleProgram in
//! texture_pass.cpp): the operations from constant buffer and storage buffer reads to the
//! handle, as a [`HandleProgram`] the renderer evaluates for each draw.
//!
//! Building walks the handle's operands depth first. A value used more than once is kept in a
//! slot the first time and loaded after, so the walk visits each instruction once, and a budget
//! bounds it. Reaching a value again while it is being built means a loop carries it, which no
//! single evaluation can give.

use std::sync::atomic::{AtomicBool, Ordering};

use rustc_hash::FxHashMap;

use super::{Ctx, resolve_conditional_phi};
use crate::handle::{HandleOp, HandleProgram};
use crate::ir::{BlockId, InstId, Opcode, Program, Value};

const BUDGET: u32 = 4096;
const SLOTS: usize = 256;

/// A built program, and the word it takes as the descriptor array element
/// ([`HandleOp::Index`]), if any.
pub(super) struct Built {
    pub program: HandleProgram,
    pub index: Option<InstId>,
}

/// Builds the program computing `value` from constant buffer and storage buffer words; `None`
/// when it depends on anything else, or on the first unsupported operation, which
/// `Err` names.
///
/// With `allow_index`, a handle read from a table at an index each invocation reads for itself
/// (a light index from a per-pixel list) still builds: the index word becomes the descriptor
/// array element. Only a word read at an address no draw-wide value gives, scaled to a byte
/// offset into the address of the handle's own read, qualifies, and one word per program.
pub(super) fn build(
    prog: &Program,
    dominance: &Dominance,
    value: Value,
    ctx: Ctx,
    allow_index: bool,
) -> Result<Built, Option<Opcode>> {
    let mut builder = Builder {
        prog,
        dominance,
        ctx,
        code: Vec::new(),
        budget: BUDGET,
        slots: FxHashMap::default(),
        slot_count: 0,
        allow_index,
        index: None,
        load_depth: 0,
        scaled: 0,
        failure: None,
    };
    if !builder.value(value) {
        return Err(builder.failure);
    }
    while matches!(builder.code.last(), Some(HandleOp::Tee(_))) {
        builder.code.pop();
    }
    let index = builder.index;
    let failure = builder.failure;
    HandleProgram::new(builder.code)
        .map(|program| Built { program, index })
        .ok_or(failure)
}

#[derive(Clone, Copy)]
enum Slot {
    Building,
    Kept(u8),
}

struct Builder<'p> {
    prog: &'p Program,
    dominance: &'p Dominance,
    ctx: Ctx,
    code: Vec<HandleOp>,
    budget: u32,
    slots: FxHashMap<InstId, Slot>,
    slot_count: usize,
    allow_index: bool,
    index: Option<InstId>,
    /// Reads whose address is being built.
    load_depth: u32,
    /// Constant multiplications and shifts around the value being built: an index scaled to a
    /// byte offset.
    scaled: u32,
    failure: Option<Opcode>,
}

impl Builder<'_> {
    fn emit(&mut self, op: HandleOp) -> bool {
        self.code.push(op);
        true
    }

    fn fail(&mut self, op: Opcode) -> bool {
        self.failure.get_or_insert(op);
        false
    }

    fn value(&mut self, value: Value) -> bool {
        let value = self.prog.resolve(value);
        let Value::Inst(inst) = value else {
            return self.node(value);
        };
        match self.slots.get(&inst) {
            Some(Slot::Building) => return self.fail(Opcode::Phi),
            Some(&Slot::Kept(slot)) => return self.emit(HandleOp::Load(slot)),
            None => {}
        }
        self.slots.insert(inst, Slot::Building);
        let (start, first_slot, index) = (self.code.len(), self.slot_count, self.index);
        let built = self.node(value);
        self.slots.remove(&inst);
        if !built {
            self.index = index;
            self.code.truncate(start);
            self.slots
                .retain(|_, slot| !matches!(*slot, Slot::Kept(s) if usize::from(s) >= first_slot));
            self.slot_count = first_slot;
            return false;
        }
        if self.prog.inst(inst).use_count() > 1 && self.slot_count < SLOTS {
            let slot = self.slot_count as u8;
            self.slot_count += 1;
            self.slots.insert(inst, Slot::Kept(slot));
            self.emit(HandleOp::Tee(slot));
        }
        true
    }

    fn arg(&self, inst: InstId, index: usize) -> Value {
        self.prog.arg(inst, index)
    }

    fn args(&mut self, inst: InstId, indices: &[usize]) -> bool {
        indices.iter().all(|&i| self.value(self.arg(inst, i)))
    }

    fn binary(&mut self, inst: InstId, op: HandleOp) -> bool {
        self.args(inst, &[0, 1]) && self.emit(op)
    }

    fn unary(&mut self, inst: InstId, op: HandleOp) -> bool {
        self.args(inst, &[0]) && self.emit(op)
    }

    fn node(&mut self, value: Value) -> bool {
        let Some(budget) = self.budget.checked_sub(1) else {
            return false;
        };
        self.budget = budget;
        let inst = match value {
            Value::Inst(inst) => inst,
            Value::U1(value) => return self.emit(HandleOp::Immediate(u32::from(value))),
            Value::U8(value) => return self.emit(HandleOp::Immediate(u32::from(value))),
            Value::U16(value) => return self.emit(HandleOp::Immediate(u32::from(value))),
            Value::U32(value) => return self.emit(HandleOp::Immediate(value)),
            Value::F32(value) => return self.emit(HandleOp::Immediate(value.to_bits())),
            _ => return false,
        };
        use HandleOp as H;
        let op = self.prog.op(inst);
        match op {
            Opcode::Identity
            | Opcode::ConditionRef
            | Opcode::ShuffleIndex
            | Opcode::VoteAny
            | Opcode::VoteAll
            | Opcode::BitCastU32F32
            | Opcode::BitCastF32U32 => self.args(inst, &[0]),
            Opcode::GetCbufU32 | Opcode::LoadStorage32 => self.load_word(inst, inst, 0),
            Opcode::CompositeExtractU32x2
            | Opcode::CompositeExtractU32x3
            | Opcode::CompositeExtractU32x4 => {
                let source = self.prog.resolve(self.arg(inst, 0)).inst();
                match (source, self.prog.resolve(self.arg(inst, 1)).imm_u32()) {
                    (Some(source), Some(element)) => self.load_word(inst, source, element * 4),
                    _ => self.fail(op),
                }
            }
            Opcode::Phi => self.phi(inst, value),
            Opcode::IAdd32 => self.binary(inst, H::Add),
            Opcode::ISub32 => self.binary(inst, H::Sub),
            Opcode::IMul32 | Opcode::ShiftLeftLogical32 => {
                let is_shift = op == Opcode::ShiftLeftLogical32;
                let scaled = self.prog.resolve(self.arg(inst, 1)).is_immediate()
                    || (!is_shift && self.prog.resolve(self.arg(inst, 0)).is_immediate());
                self.scaled += u32::from(scaled);
                let built = self.args(inst, &[0, 1]);
                self.scaled -= u32::from(scaled);
                built && self.emit(if is_shift { H::Shl } else { H::Mul })
            }
            Opcode::BitwiseAnd32 => self.binary(inst, H::And),
            Opcode::BitwiseOr32 => self.binary(inst, H::Or),
            Opcode::BitwiseXor32 => self.binary(inst, H::Xor),
            Opcode::ShiftRightLogical32 => self.binary(inst, H::Shr),
            Opcode::ShiftRightArithmetic32 => self.binary(inst, H::Sar),
            Opcode::INeg32 => self.unary(inst, H::Neg),
            Opcode::BitwiseNot32 => self.unary(inst, H::Not),
            Opcode::UMin32 => self.binary(inst, H::UMin),
            Opcode::UMax32 => self.binary(inst, H::UMax),
            Opcode::SMin32 => self.binary(inst, H::SMin),
            Opcode::SMax32 => self.binary(inst, H::SMax),
            Opcode::SLessThan => self.binary(inst, H::SLt),
            Opcode::SLessThanEqual => self.binary(inst, H::SLe),
            Opcode::SGreaterThan => self.binary(inst, H::SGt),
            Opcode::SGreaterThanEqual => self.binary(inst, H::SGe),
            Opcode::ULessThan => self.binary(inst, H::ULt),
            Opcode::ULessThanEqual => self.binary(inst, H::ULe),
            Opcode::UGreaterThan => self.binary(inst, H::UGt),
            Opcode::UGreaterThanEqual => self.binary(inst, H::UGe),
            Opcode::IEqual => self.binary(inst, H::Eq),
            Opcode::INotEqual => self.binary(inst, H::Ne),
            Opcode::LogicalAnd => self.binary(inst, H::LAnd),
            Opcode::LogicalOr => self.binary(inst, H::LOr),
            Opcode::LogicalXor => self.binary(inst, H::LXor),
            Opcode::LogicalNot => self.unary(inst, H::LNot),
            Opcode::GetZeroFromOp => {
                self.args(inst, &[0]) && self.emit(H::Immediate(0)) && self.emit(H::Eq)
            }
            Opcode::GetSignFromOp => {
                self.args(inst, &[0]) && self.emit(H::Immediate(0)) && self.emit(H::SLt)
            }
            Opcode::GetCarryFromOp => {
                let add = self.prog.resolve(self.arg(inst, 0)).inst();
                match add.filter(|&add| self.prog.op(add) == Opcode::IAdd32) {
                    Some(add) => {
                        self.value(Value::Inst(add))
                            && self.value(self.arg(add, 0))
                            && self.emit(H::ULt)
                    }
                    None => self.fail(op),
                }
            }
            Opcode::UndefU1 | Opcode::UndefU32 => self.emit(H::Immediate(0)),
            Opcode::BitFieldUExtract => {
                self.args(inst, &[0, 1])
                    && self.emit(H::Shr)
                    && self.emit(H::Immediate(1))
                    && self.args(inst, &[2])
                    && self.emit(H::Shl)
                    && self.emit(H::Immediate(1))
                    && self.emit(H::Sub)
                    && self.emit(H::And)
            }
            Opcode::BitFieldSExtract => {
                self.args(inst, &[0])
                    && self.emit(H::Immediate(32))
                    && self.args(inst, &[1])
                    && self.emit(H::Sub)
                    && self.args(inst, &[2])
                    && self.emit(H::Sub)
                    && self.emit(H::Shl)
                    && self.emit(H::Immediate(32))
                    && self.args(inst, &[2])
                    && self.emit(H::Sub)
                    && self.emit(H::Sar)
            }
            Opcode::BitFieldInsert => {
                let mask = |b: &mut Self| {
                    b.emit(H::Immediate(1))
                        && b.args(inst, &[3])
                        && b.emit(H::Shl)
                        && b.emit(H::Immediate(1))
                        && b.emit(H::Sub)
                        && b.args(inst, &[2])
                        && b.emit(H::Shl)
                };
                self.args(inst, &[0])
                    && mask(self)
                    && self.emit(H::Not)
                    && self.emit(H::And)
                    && self.args(inst, &[1, 2])
                    && self.emit(H::Shl)
                    && mask(self)
                    && self.emit(H::And)
                    && self.emit(H::Or)
            }
            Opcode::UClamp32 => {
                self.binary(inst, H::UMax) && self.args(inst, &[2]) && self.emit(H::UMin)
            }
            Opcode::SClamp32 => {
                self.binary(inst, H::SMax) && self.args(inst, &[2]) && self.emit(H::SMin)
            }
            Opcode::SelectU1 | Opcode::SelectU32 => {
                self.args(inst, &[0, 1, 2]) && self.emit(H::Select)
            }
            _ => self.fail(op),
        }
    }

    /// The value every defined incoming edge carries, looking through the register-to-uniform
    /// move a conditional block makes for a handle. Different values on different paths need
    /// a condition no program has, so they bind nothing. A loop header's phi builds only when
    /// every back edge carries the phi itself: a value the loop changes differs between
    /// iterations.
    fn phi(&mut self, phi: InstId, value: Value) -> bool {
        if self.ctx.key.is_some() {
            let resolved = resolve_conditional_phi(self.prog, value, self.ctx.block);
            if self.prog.resolve(resolved) != Value::Inst(phi) {
                return self.value(resolved);
            }
        }
        let header = self.prog.inst(phi).block();
        let varies = (0..self.prog.inst(phi).num_args()).any(|i| {
            self.dominance
                .is_back_edge(self.prog.inst(phi).phi_block(i), header)
                && self.prog.resolve(self.arg(phi, i)) != Value::Inst(phi)
        });
        if varies {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "a bindless handle changes on each loop iteration: binding no texture for it"
                );
            }
            return self.fail(Opcode::Phi);
        }
        let mut incoming = (0..self.prog.inst(phi).num_args()).filter_map(|i| {
            let value = self.prog.resolve(self.arg(phi, i));
            let Value::Inst(id) = value else {
                return Some(value);
            };
            if id == phi {
                return None;
            }
            match self.prog.op(id) {
                Opcode::UndefU1
                | Opcode::UndefU8
                | Opcode::UndefU16
                | Opcode::UndefU32
                | Opcode::UndefU64 => None,
                Opcode::ShuffleIndex => Some(self.prog.resolve(self.arg(id, 0))),
                _ => Some(value),
            }
        });
        let Some(first) = incoming.next() else {
            return self.fail(Opcode::Phi);
        };
        if incoming.any(|value| value != first) {
            return self.fail(Opcode::Phi);
        }
        self.value(first)
    }

    /// A word of a constant buffer read or a storage buffer load `source`, at byte offset
    /// `+extra`. `node` is the value being built (the read or a word extracted from it).
    fn load_word(&mut self, node: InstId, source: InstId, extra: u32) -> bool {
        let op = self.prog.op(source);
        let (read, address) = match op {
            Opcode::GetCbufU32 | Opcode::GetCbufU32x2 => {
                let Some(index) = self.prog.resolve(self.arg(source, 0)).imm_u32() else {
                    return self.fail(op);
                };
                let offset = self.prog.resolve(self.arg(source, 1));
                if let Some(offset) = offset.imm_u32() {
                    return self.emit(HandleOp::Cbuf {
                        index,
                        offset: offset.wrapping_add(extra),
                    });
                }
                (HandleOp::CbufDynamic { index }, offset)
            }
            Opcode::LoadStorage32 | Opcode::LoadStorage64 | Opcode::LoadStorage128 => {
                let Some(binding) = self.prog.resolve(self.arg(source, 0)).imm_u32() else {
                    return self.fail(op);
                };
                (HandleOp::Storage { binding }, self.arg(source, 1))
            }
            _ => return self.fail(op),
        };
        if !self.address(address) {
            return self.index_word(node);
        }
        if extra != 0 {
            self.emit(HandleOp::Immediate(extra));
            self.emit(HandleOp::Add);
        }
        self.emit(read)
    }

    /// The address of a read. One inside another read's address must be computable on its
    /// own: an index word may only appear in the handle read's own address.
    fn address(&mut self, offset: Value) -> bool {
        let (allow_index, scaled) = (self.allow_index, self.scaled);
        self.allow_index = allow_index && self.load_depth == 0;
        self.scaled = 0;
        self.load_depth += 1;
        let built = self.value(offset);
        self.load_depth -= 1;
        self.scaled = scaled;
        self.allow_index = allow_index;
        built
    }

    /// `node` as the element index of a descriptor array (see [`build`]).
    fn index_word(&mut self, node: InstId) -> bool {
        if !self.allow_index
            || self.load_depth == 0
            || self.scaled == 0
            || self.index.is_some_and(|index| index != node)
        {
            return false;
        }
        self.index = Some(node);
        self.emit(HandleOp::Index)
    }
}

/// Immediate dominators of the program's blocks (Cooper, Harvey and Kennedy over post order
/// numbers), for whether a table's index word is defined wherever the texture instruction runs.
pub(super) struct Dominance {
    /// Post order number of each block, `NONE` for an unreachable one.
    numbers: Vec<usize>,
    idom: Vec<usize>,
}

const NONE: usize = usize::MAX;

impl Dominance {
    pub(super) fn new(prog: &Program) -> Dominance {
        let order = &prog.post_order;
        let mut numbers = vec![NONE; prog.num_blocks()];
        for (number, block) in order.iter().enumerate() {
            numbers[block.0 as usize] = number;
        }
        let mut idom = vec![NONE; order.len()];
        let Some(entry) = order.len().checked_sub(1) else {
            return Dominance { numbers, idom };
        };
        idom[entry] = entry;
        let intersect = |idom: &[usize], mut a: usize, mut b: usize| {
            while a != b {
                while a < b {
                    a = idom[a];
                }
                while b < a {
                    b = idom[b];
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for number in (0..entry).rev() {
                let mut dominator = NONE;
                for predecessor in &prog.block(order[number]).predecessors {
                    let p = numbers[predecessor.0 as usize];
                    if p == NONE || idom[p] == NONE {
                        continue;
                    }
                    dominator = if dominator == NONE {
                        p
                    } else {
                        intersect(&idom, p, dominator)
                    };
                }
                if dominator != idom[number] {
                    idom[number] = dominator;
                    changed = true;
                }
            }
        }
        Dominance { numbers, idom }
    }

    /// Whether the edge from `from` to `to` closes a loop: `to` was entered before `from`
    /// and left after it.
    pub(super) fn is_back_edge(&self, from: BlockId, to: BlockId) -> bool {
        let (from, to) = (self.numbers[from.0 as usize], self.numbers[to.0 as usize]);
        from != NONE && to != NONE && from <= to
    }

    /// Whether `def` runs before `user` on every path to `user`.
    pub(super) fn dominates(&self, prog: &Program, def: InstId, user: InstId) -> bool {
        let (def_block, use_block) = (prog.inst(def).block(), prog.inst(user).block());
        if def_block == use_block {
            return prog.insts_of(def_block).find(|&id| id == def || id == user) == Some(def);
        }
        let target = self.numbers[def_block.0 as usize];
        let mut number = self.numbers[use_block.0 as usize];
        if target == NONE || number == NONE {
            return false;
        }
        while self.idom[number] != NONE && self.idom[number] != number {
            number = self.idom[number];
            if number == target {
                return true;
            }
        }
        false
    }
}
