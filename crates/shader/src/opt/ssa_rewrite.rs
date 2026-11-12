//! SSA construction (Braun et al., "Simple and Efficient Construction of Static Single
//! Assignment Form"): register, predicate, flag and goto-variable reads become the reaching
//! definition, with phis where definitions merge.
//!
//! Reads walk predecessors with an explicit stack instead of recursion. A phi in a sealed block
//! stays pending until a cycle reads it or it merges two different values, so most trivial phis
//! are never inserted.

use rustc_hash::FxHashMap;

use crate::ir::{BlockId, InstId, Opcode, Pred, Program, Reg, Value};

const PRED_BASE: u32 = Reg::COUNT as u32;
const Z_FLAG: u32 = PRED_BASE + Pred::COUNT as u32;
const S_FLAG: u32 = Z_FLAG + 1;
const C_FLAG: u32 = Z_FLAG + 2;
const O_FLAG: u32 = Z_FLAG + 3;
const INDIRECT_BRANCH: u32 = Z_FLAG + 4;
const GOTO_BASE: u32 = Z_FLAG + 5;

fn undef_opcode(var: u32) -> Opcode {
    if var < PRED_BASE || var == INDIRECT_BRANCH {
        Opcode::UndefU32
    } else {
        Opcode::UndefU1
    }
}

#[derive(Clone, Copy)]
enum Def {
    Value(Value),
    Pending,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Start,
    SetValue,
    PushPhiArgument,
}

#[derive(Clone, Copy)]
struct Frame {
    block: BlockId,
    result: Value,
    phi: Option<InstId>,
    pred_index: usize,
    operands_begin: usize,
    num_operands: usize,
    pc: Status,
}

impl Frame {
    fn new(block: BlockId) -> Frame {
        Frame {
            block,
            result: Value::Void,
            phi: None,
            pred_index: 0,
            operands_begin: 0,
            num_operands: 0,
            pc: Status::Start,
        }
    }
}

#[derive(Default)]
struct Pass {
    defs: FxHashMap<(u32, u32), Def>,
    incomplete: FxHashMap<u32, Vec<(u32, InstId)>>,
    sealed: Vec<bool>,
    stack: Vec<Frame>,
    pending_operands: Vec<(BlockId, Value)>,
}

fn insert_undef(prog: &mut Program, block: BlockId, op: Opcode) -> Value {
    let mut cur = prog.block(block).first();
    while let Some(id) = cur {
        if prog.op(id) != Opcode::Phi {
            return Value::Inst(prog.insert_before(id, op, &[], 0));
        }
        cur = prog.inst(id).next();
    }
    Value::Inst(prog.append(block, op, &[], 0))
}

fn new_phi(prog: &mut Program, block: BlockId, var: u32) -> InstId {
    prog.prepend(block, Opcode::Phi, &[], undef_opcode(var).ty().bits())
}

fn try_remove_trivial_phi(prog: &mut Program, phi: InstId, block: BlockId, var: u32) -> Value {
    let mut same: Option<Value> = None;
    for i in 0..prog.inst(phi).num_args() {
        let resolved = prog.resolve(prog.arg(phi, i));
        if resolved == Value::Inst(phi) || same.is_some_and(|s| s.same(resolved)) {
            continue;
        }
        if same.is_some() {
            return Value::Inst(phi);
        }
        same = Some(resolved);
    }
    let same = same.unwrap_or_else(|| insert_undef(prog, block, undef_opcode(var)));
    prog.replace_uses_with(phi, same);
    same
}

impl Pass {
    fn write(&mut self, var: u32, block: BlockId, value: Value) {
        self.defs.insert((block.0, var), Def::Value(value));
    }

    fn read(&mut self, prog: &mut Program, var: u32, root: BlockId) -> Value {
        self.stack.clear();
        self.stack.push(Frame::new(root));
        self.stack.push(Frame::new(root));
        while self.stack.len() > 1 {
            let top = self.stack.len() - 1;
            let block = self.stack[top].block;
            match self.stack[top].pc {
                Status::Start => match self.defs.get(&(block.0, var)).copied() {
                    Some(Def::Value(v)) => {
                        self.stack[top].result = v;
                        self.set_value(var);
                    }
                    Some(Def::Pending) => {
                        let phi = self.materialize_phi(prog, self.pending_frame(block), var);
                        self.stack[top].result = Value::Inst(phi);
                        self.set_value(var);
                    }
                    None if !self.sealed[block.0 as usize] => {
                        let phi = new_phi(prog, block, var);
                        self.incomplete.entry(block.0).or_default().push((var, phi));
                        self.stack[top].result = Value::Inst(phi);
                        self.set_value(var);
                    }
                    None if prog.block(block).predecessors.len() == 1 => {
                        let pred = prog.block(block).predecessors[0];
                        self.stack[top].pc = Status::SetValue;
                        self.stack.push(Frame::new(pred));
                    }
                    None => {
                        self.defs.insert((block.0, var), Def::Pending);
                        self.stack[top].operands_begin = self.pending_operands.len();
                        self.prepare_phi_operand(prog, var);
                    }
                },
                Status::SetValue => self.set_value(var),
                Status::PushPhiArgument => {
                    let frame = self.stack[top];
                    let pred = prog.block(block).predecessors[frame.pred_index];
                    match frame.phi {
                        Some(phi) => prog.add_phi_operand(phi, pred, frame.result),
                        None => {
                            self.pending_operands.push((pred, frame.result));
                            self.stack[top].num_operands += 1;
                        }
                    }
                    self.stack[top].pred_index += 1;
                    self.prepare_phi_operand(prog, var);
                }
            }
        }
        self.stack[0].result
    }

    fn set_value(&mut self, var: u32) {
        let frame = self.stack.pop().expect("read stack");
        self.write(var, frame.block, frame.result);
        if let Some(parent) = self.stack.last_mut() {
            parent.result = frame.result;
        }
    }

    fn prepare_phi_operand(&mut self, prog: &mut Program, var: u32) {
        let top = self.stack.len() - 1;
        let frame = self.stack[top];
        let preds = &prog.block(frame.block).predecessors;
        if frame.pred_index < preds.len() {
            let pred = preds[frame.pred_index];
            self.stack[top].pc = Status::PushPhiArgument;
            self.stack.push(Frame::new(pred));
            return;
        }
        let result = match frame.phi {
            Some(phi) => try_remove_trivial_phi(prog, phi, frame.block, var),
            None => self.resolve_pending_phi(prog, top, var),
        };
        self.pending_operands.truncate(frame.operands_begin);
        self.stack.pop();
        if let Some(parent) = self.stack.last_mut() {
            parent.result = result;
        }
        self.write(var, frame.block, result);
    }

    fn pending_frame(&self, block: BlockId) -> usize {
        self.stack
            .iter()
            .rposition(|f| f.block == block && f.pc == Status::PushPhiArgument && f.phi.is_none())
            .expect("pending phi without a frame")
    }

    fn materialize_phi(&mut self, prog: &mut Program, frame: usize, var: u32) -> InstId {
        let Frame {
            block,
            operands_begin,
            num_operands,
            ..
        } = self.stack[frame];
        let phi = new_phi(prog, block, var);
        for &(pred, value) in &self.pending_operands[operands_begin..operands_begin + num_operands]
        {
            prog.add_phi_operand(phi, pred, value);
        }
        self.stack[frame].phi = Some(phi);
        self.write(var, block, Value::Inst(phi));
        phi
    }

    fn resolve_pending_phi(&mut self, prog: &mut Program, frame: usize, var: u32) -> Value {
        let Frame {
            block,
            operands_begin,
            num_operands,
            ..
        } = self.stack[frame];
        let mut same: Option<Value> = None;
        for i in operands_begin..operands_begin + num_operands {
            let op = prog.resolve(self.pending_operands[i].1);
            match same {
                Some(s) if s.same(op) => {}
                Some(_) => return Value::Inst(self.materialize_phi(prog, frame, var)),
                None => same = Some(op),
            }
        }
        same.unwrap_or_else(|| insert_undef(prog, block, undef_opcode(var)))
    }

    fn seal(&mut self, prog: &mut Program, block: BlockId) {
        for (var, phi) in self.incomplete.remove(&block.0).unwrap_or_default() {
            for i in 0..prog.block(block).predecessors.len() {
                let pred = prog.block(block).predecessors[i];
                let value = self.read(prog, var, pred);
                prog.add_phi_operand(phi, pred, value);
            }
            try_remove_trivial_phi(prog, phi, block, var);
        }
        self.sealed[block.0 as usize] = true;
    }
}

fn variable(prog: &Program, id: InstId) -> Option<u32> {
    match prog.arg(id, 0) {
        Value::Reg(r) if !r.is_zero() => Some(u32::from(r.0)),
        Value::Pred(p) if !p.is_true() => Some(PRED_BASE + u32::from(p.0)),
        Value::U32(goto) => Some(GOTO_BASE + goto),
        _ => None,
    }
}

fn flag_variable(op: Opcode) -> Option<u32> {
    match op {
        Opcode::SetZFlag | Opcode::GetZFlag => Some(Z_FLAG),
        Opcode::SetSFlag | Opcode::GetSFlag => Some(S_FLAG),
        Opcode::SetCFlag | Opcode::GetCFlag => Some(C_FLAG),
        Opcode::SetOFlag | Opcode::GetOFlag => Some(O_FLAG),
        Opcode::SetIndirectBranchVariable | Opcode::GetIndirectBranchVariable => {
            Some(INDIRECT_BRANCH)
        }
        _ => None,
    }
}

/// Rewrites every context read into SSA form. Needs `Program::post_order`.
pub fn ssa_rewrite(prog: &mut Program) {
    let mut pass = Pass {
        sealed: vec![false; prog.num_blocks()],
        ..Pass::default()
    };
    let order: Vec<BlockId> = prog.post_order.iter().rev().copied().collect();
    for &block in &order {
        let mut cur = prog.block(block).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            let op = prog.op(id);
            match op {
                Opcode::SetRegister | Opcode::SetPred | Opcode::SetGotoVariable => {
                    if let Some(var) = variable(prog, id) {
                        pass.write(var, block, prog.arg(id, 1));
                    }
                }
                Opcode::GetRegister | Opcode::GetPred | Opcode::GetGotoVariable => {
                    if let Some(var) = variable(prog, id) {
                        let value = pass.read(prog, var, block);
                        prog.replace_uses_with(id, value);
                    }
                }
                _ => match (flag_variable(op), op.ty() == crate::ir::Type::VOID) {
                    (Some(var), true) => pass.write(var, block, prog.arg(id, 0)),
                    (Some(var), false) => {
                        let value = pass.read(prog, var, block);
                        prog.replace_uses_with(id, value);
                    }
                    (None, _) => {}
                },
            }
        }
        pass.seal(prog, block);
    }
    for block in order.into_iter().rev() {
        let mut cur = prog.block(block).last();
        while let Some(id) = cur {
            cur = prog.inst(id).prev();
            if prog.op(id) == Opcode::Identity && !prog.inst(id).has_uses() {
                prog.remove(id);
            }
        }
    }
}
