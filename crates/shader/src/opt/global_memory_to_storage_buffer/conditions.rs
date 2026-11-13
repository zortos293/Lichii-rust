//! Structured control flow facts the pointer-addend search uses to skip phi inputs that cannot
//! reach an access: the `if` scopes around each block and the loop each block is in.

use rustc_hash::FxHashMap;

use super::{imm_u32, inst_of};
use crate::ir::{BlockId, InstId, Opcode, Program, SyntaxNode, Value};

/// A boolean value and the polarity under which it holds.
pub(super) type Predicate = (Value, bool);

type BitKey = (InstId, u32, Option<InstId>, u8);

const NONE: u32 = u32::MAX;
const LITERAL_DEPTH: u32 = 256;

#[derive(Clone, Copy)]
struct Scope {
    condition: Predicate,
    header: Option<BlockId>,
}

pub(super) struct Conditions {
    order: Vec<u32>,
    block_context: Vec<u32>,
    context_scopes: Vec<Vec<Scope>>,
    context_loop: Vec<Option<BlockId>>,
    merges: Vec<Vec<Scope>>,
    bit_cache: FxHashMap<BitKey, Option<Predicate>>,
}

fn constant(predicate: Predicate) -> Option<bool> {
    match predicate.0 {
        Value::U1(value) => Some(value == predicate.1),
        _ => None,
    }
}

impl Conditions {
    pub(super) fn new(prog: &Program) -> Conditions {
        let blocks = prog.num_blocks();
        let mut order = vec![NONE; blocks];
        for (index, block) in prog.block_order.iter().enumerate() {
            order[block.0 as usize] = index as u32;
        }
        let mut conditions = Conditions {
            order,
            block_context: vec![NONE; blocks],
            context_scopes: Vec::new(),
            context_loop: Vec::new(),
            merges: vec![Vec::new(); blocks],
            bit_cache: FxHashMap::default(),
        };
        let mut active: Vec<Scope> = Vec::new();
        let mut loop_stack = Vec::new();
        let mut previous = None;
        let mut context = NONE;
        for node in &prog.syntax_list {
            match *node {
                SyntaxNode::Block(block) => {
                    previous = Some(block);
                    if context == NONE {
                        context = conditions.context_scopes.len() as u32;
                        conditions.context_scopes.push(active.clone());
                        conditions.context_loop.push(loop_stack.last().copied());
                    }
                    conditions.block_context[block.0 as usize] = context;
                }
                SyntaxNode::If { cond, merge, .. } => {
                    let scope = Scope {
                        condition: conditions.literal(prog, cond, LITERAL_DEPTH),
                        header: previous,
                    };
                    conditions.merges[merge.0 as usize].push(scope);
                    active.push(scope);
                    context = NONE;
                }
                SyntaxNode::EndIf { .. } => {
                    active.pop();
                    context = NONE;
                }
                SyntaxNode::Loop { body, .. } => {
                    loop_stack.push(body);
                    context = NONE;
                }
                SyntaxNode::Repeat { .. } => {
                    loop_stack.pop();
                    context = NONE;
                }
                _ => {}
            }
        }
        conditions.bit_cache = FxHashMap::default();
        conditions
    }

    /// Blocks with the same context are inside the same `if` scopes and loop, so
    /// [`Conditions::allows_phi`] answers the same for all of them.
    pub(super) fn context(&self, block: BlockId) -> u32 {
        self.block_context[block.0 as usize]
    }

    pub(super) fn is_back_edge(&self, prog: &Program, phi: InstId, arg: usize) -> bool {
        let inst = prog.inst(phi);
        let owner = self.order[inst.block().0 as usize];
        owner != NONE && self.order[inst.phi_block(arg).0 as usize] >= owner
    }

    /// Whether phi input `arg` can flow into an access in `current`: false when the access sits
    /// in a branch of the same condition that rules that input out.
    pub(super) fn allows_phi(
        &self,
        prog: &Program,
        phi: InstId,
        arg: usize,
        current: BlockId,
    ) -> bool {
        let inst = prog.inst(phi);
        let owner = inst.block();
        let owner_context = self.block_context[owner.0 as usize];
        let current_context = self.block_context[current.0 as usize];
        if owner_context == NONE
            || current_context == NONE
            || self.context_loop[owner_context as usize]
                != self.context_loop[current_context as usize]
        {
            return true;
        }
        let incoming = Some(inst.phi_block(arg));
        for branch in &self.merges[owner.0 as usize] {
            for known in &self.context_scopes[current_context as usize] {
                if branch.condition.0 != known.condition.0 {
                    continue;
                }
                let taken = branch.condition.1 == known.condition.1;
                let bypass = incoming == branch.header;
                if bypass == taken {
                    return false;
                }
            }
        }
        true
    }

    fn bit(
        &mut self,
        prog: &Program,
        value: Value,
        bit: u32,
        depth: u32,
        assumed: Option<Predicate>,
    ) -> Option<Predicate> {
        let Some(inst) = inst_of(prog, value) else {
            return self.bit_impl(prog, value, bit, depth, assumed);
        };
        let kind = match assumed {
            None => 0,
            Some(a) => match constant(a) {
                Some(c) => 4 + c as u8,
                None => 1 + a.1 as u8,
            },
        };
        let key = (inst, bit, assumed.and_then(|a| inst_of(prog, a.0)), kind);
        if let Some(&cached) = self.bit_cache.get(&key) {
            return cached;
        }
        self.bit_cache.insert(key, None);
        let result = self.bit_impl(prog, value, bit, depth, assumed);
        self.bit_cache.insert(key, result);
        result
    }

    fn bit_impl(
        &mut self,
        prog: &Program,
        value: Value,
        bit: u32,
        depth: u32,
        assumed: Option<Predicate>,
    ) -> Option<Predicate> {
        if bit >= 32 || depth == 0 {
            return None;
        }
        let value = prog.resolve(value);
        let Some(id) = value.inst() else {
            return value
                .imm_u32()
                .map(|v| (Value::U1((v >> bit) & 1 != 0), true));
        };
        let inst = prog.inst(id);
        match inst.op {
            Opcode::Phi if inst.num_args() == 2 => {
                let owner = inst.block();
                let merge = &self.merges[owner.0 as usize];
                let owner_order = self.order[owner.0 as usize];
                if merge.len() != 1 || owner_order == NONE {
                    return None;
                }
                let branch = merge[0];
                let bypass = usize::from(Some(inst.phi_block(0)) != branch.header);
                if Some(inst.phi_block(bypass)) != branch.header
                    || self.order[inst.phi_block(1 - bypass).0 as usize] >= owner_order
                {
                    return None;
                }
                if let Some(a) = assumed
                    && a.0 == branch.condition.0
                {
                    let take_body = a.1 == branch.condition.1;
                    let arg = inst.arg(if take_body { 1 - bypass } else { bypass });
                    return self.bit(prog, arg, bit, depth - 1, assumed);
                }
                let false_guard = (branch.condition.0, !branch.condition.1);
                let no = self.bit(prog, inst.arg(bypass), bit, depth - 1, Some(false_guard));
                let yes = self.bit(
                    prog,
                    inst.arg(1 - bypass),
                    bit,
                    depth - 1,
                    Some(branch.condition),
                );
                let (Some(no), Some(yes)) = (no, yes) else {
                    return None;
                };
                if no == yes {
                    return Some(no);
                }
                let (Some(nc), Some(yc)) = (constant(no), constant(yes)) else {
                    return None;
                };
                if nc == yc {
                    return Some((Value::U1(yc), true));
                }
                Some((branch.condition.0, branch.condition.1 == yc))
            }
            Opcode::SelectU32 => {
                let yes = self.bit(prog, inst.arg(1), bit, depth - 1, assumed);
                let no = self.bit(prog, inst.arg(2), bit, depth - 1, assumed);
                let (Some(yes), Some(no)) = (yes, no) else {
                    return None;
                };
                if yes == no {
                    return Some(yes);
                }
                let (Some(yc), Some(nc)) = (constant(yes), constant(no)) else {
                    return None;
                };
                if nc == yc {
                    return Some((Value::U1(yc), true));
                }
                let mut cond = self.literal(prog, inst.arg(0), depth - 1);
                cond.1 = cond.1 == yc;
                Some(cond)
            }
            Opcode::BitwiseOr32 | Opcode::BitwiseAnd32 => {
                let is_or = inst.op == Opcode::BitwiseOr32;
                let lhs = self.bit(prog, inst.arg(0), bit, depth - 1, assumed);
                let rhs = self.bit(prog, inst.arg(1), bit, depth - 1, assumed);
                if let Some(c) = lhs.and_then(constant) {
                    return if c == is_or { lhs } else { rhs };
                }
                if let Some(c) = rhs.and_then(constant) {
                    return if c == is_or { rhs } else { lhs };
                }
                None
            }
            Opcode::BitFieldUExtract => {
                let offset = imm_u32(prog, inst.arg(1))?;
                let count = imm_u32(prog, inst.arg(2))?;
                if bit >= count {
                    return Some((Value::U1(false), true));
                }
                self.bit(
                    prog,
                    inst.arg(0),
                    bit.wrapping_add(offset),
                    depth - 1,
                    assumed,
                )
            }
            _ => None,
        }
    }

    fn literal(&mut self, prog: &Program, value: Value, depth: u32) -> Predicate {
        let value = prog.resolve(value);
        let Some(id) = value.inst() else {
            return (value, true);
        };
        if depth == 0 {
            return (value, true);
        }
        let inst = prog.inst(id);
        match inst.op {
            Opcode::ConditionRef => return self.literal(prog, inst.arg(0), depth - 1),
            Opcode::LogicalNot => {
                let mut result = self.literal(prog, inst.arg(0), depth - 1);
                result.1 = !result.1;
                return result;
            }
            Opcode::SelectU1 => {
                let cond = self.literal(prog, inst.arg(0), depth - 1);
                if let Some(c) = constant(cond) {
                    return self.literal(prog, inst.arg(if c { 1 } else { 2 }), depth - 1);
                }
            }
            Opcode::IEqual | Opcode::INotEqual | Opcode::GetZeroFromOp => {
                let compares_zero =
                    inst.op == Opcode::GetZeroFromOp || imm_u32(prog, inst.arg(1)) == Some(0);
                if compares_zero
                    && let Some(result) = self.zero_test(prog, inst.op, inst.arg(0), depth)
                {
                    return result;
                }
            }
            _ => {}
        }
        (Value::Inst(id), true)
    }

    fn zero_test(
        &mut self,
        prog: &Program,
        op: Opcode,
        input: Value,
        depth: u32,
    ) -> Option<Predicate> {
        let is_zero = op != Opcode::INotEqual;
        let input = prog.resolve(input);
        if let Value::U32(v) = input {
            return Some((Value::U1(v == 0), is_zero));
        }
        let extract = prog.inst(input.inst()?);
        if extract.op == Opcode::BitwiseAnd32 {
            for arg in 0..2 {
                let Some(mask) = imm_u32(prog, extract.arg(arg)) else {
                    continue;
                };
                if !mask.is_power_of_two() {
                    continue;
                }
                let bit = mask.trailing_zeros();
                if let Some(mut result) = self.bit(prog, extract.arg(1 - arg), bit, depth - 1, None)
                {
                    result.1 = result.1 != is_zero;
                    return Some(result);
                }
            }
        }
        if extract.op == Opcode::BitFieldUExtract
            && imm_u32(prog, extract.arg(2)) == Some(1)
            && let Some(mut result) = self.bit(prog, input, 0, depth - 1, None)
        {
            result.1 = result.1 != is_zero;
            return Some(result);
        }
        None
    }
}
