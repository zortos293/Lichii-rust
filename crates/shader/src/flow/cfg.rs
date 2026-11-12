//! Control flow graph discovery.
//!
//! Blocks are found by following every reachable label from the entry point. A predicated
//! instruction gets its own conditional block behind an empty "virtual" block that holds the
//! condition, so translation never has to predicate individual instructions.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::ops::Bound;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::{FlowInst, FlowIsa, FlowOp, Token};
use crate::ir::{Condition, Reg};
use crate::{ShaderEnvironment, TranslateError};

/// Index of a block in `Cfg::blocks`.
pub type BlockId = u32;
/// Index of a function in `Cfg::functions`. Function 0 is the program entry.
pub type FunctionId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndClass {
    Branch,
    IndirectBranch,
    Call,
    Exit,
    Return,
    Kill,
}

/// Maxwell's divergence stack as seen at one point of the program.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stack(SmallVec<[(Token, u32); 4]>);

impl Stack {
    fn push(&mut self, token: Token, target: u32) {
        self.0.push((token, target));
    }

    fn position(&self, token: Token) -> Option<usize> {
        self.0.iter().rposition(|&(t, _)| t == token)
    }

    pub fn peek(&self, token: Token) -> Option<u32> {
        self.position(token).map(|i| self.0[i].1)
    }

    /// Drops the newest `token` and everything pushed after it.
    fn remove(&self, token: Token) -> Stack {
        match self.position(token) {
            Some(i) => Stack(self.0[..i].iter().copied().collect()),
            None => Stack(SmallVec::new()),
        }
    }

    fn pop(&self, token: Token) -> Option<(u32, Stack)> {
        Some((self.peek(token)?, self.remove(token)))
    }
}

#[derive(Clone, Debug)]
pub struct Block {
    /// First instruction offset. A virtual block holds only a condition and covers nothing.
    pub begin: u32,
    /// One past the last instruction translated in this block.
    pub end: u32,
    pub is_virtual: bool,
    pub end_class: EndClass,
    pub cond: Condition,
    pub stack: Stack,
    pub branch_true: Option<BlockId>,
    pub branch_false: Option<BlockId>,
    pub function_call: FunctionId,
    pub return_block: Option<BlockId>,
    pub branch_reg: Reg,
    pub branch_offset: i32,
    /// Targets of an indirect branch with the guest offset each was reached through.
    pub indirect_branches: Vec<(BlockId, u32)>,
}

impl Block {
    fn new(begin: u32, stack: Stack) -> Block {
        Block {
            begin,
            end: begin,
            is_virtual: false,
            end_class: EndClass::Branch,
            cond: Condition::TRUE,
            stack,
            branch_true: None,
            branch_false: None,
            function_call: 0,
            return_block: None,
            branch_reg: Reg::RZ,
            branch_offset: 0,
            indirect_branches: Vec::new(),
        }
    }

    fn key(&self) -> Key {
        (self.begin, !self.is_virtual)
    }

    fn contains(&self, pc: u32) -> bool {
        !self.is_virtual && pc >= self.begin && pc < self.end
    }
}

/// Blocks sort by offset, and a virtual block sorts just before the real block it guards.
type Key = (u32, bool);

#[derive(Clone, Copy, Debug)]
struct Label {
    address: u32,
    block: BlockId,
}

#[derive(Clone, Debug)]
pub struct Function {
    pub entry: u32,
    blocks: BTreeMap<Key, BlockId>,
    labels: Vec<Label>,
    pending: FxHashMap<u32, BlockId>,
}

impl Function {
    /// Block ids of this function in address order.
    pub fn blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.blocks.values().copied()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Analysis {
    Branch,
    Continue,
}

pub struct Cfg {
    pub functions: Vec<Function>,
    pub blocks: Vec<Block>,
    exits_to_dispatcher: bool,
}

struct Builder<'a> {
    isa: &'a dyn FlowIsa,
    env: &'a dyn ShaderEnvironment,
    program_start: u32,
    dispatch_block: Option<BlockId>,
    cfg: Cfg,
}

impl Cfg {
    /// Discovers every block reachable from `start` (a program offset).
    ///
    /// With `exits_to_dispatcher`, every exit of the entry function instead branches to an empty
    /// block placed after the program, so another program can be appended (VertexA).
    pub fn build(
        isa: &dyn FlowIsa,
        env: &dyn ShaderEnvironment,
        start: u32,
        exits_to_dispatcher: bool,
    ) -> Result<Cfg, TranslateError> {
        let start = isa.align(start);
        let mut builder = Builder {
            isa,
            env,
            program_start: start,
            dispatch_block: None,
            cfg: Cfg {
                functions: Vec::new(),
                blocks: Vec::new(),
                exits_to_dispatcher,
            },
        };
        if exits_to_dispatcher {
            let mut block = Block::new(0, Stack::default());
            block.end_class = EndClass::Exit;
            builder.dispatch_block = Some(builder.new_block(block));
        }
        builder.add_function(start);
        let mut function_id = 0;
        while function_id < builder.cfg.functions.len() {
            while let Some(label) = builder.cfg.functions[function_id].labels.pop() {
                builder.cfg.functions[function_id]
                    .pending
                    .remove(&label.address);
                builder.analyze_label(function_id, label)?;
            }
            function_id += 1;
        }
        if let Some(dispatch) = builder.dispatch_block {
            let last_end = builder.cfg.functions[0]
                .blocks
                .values()
                .next_back()
                .map_or(start, |&id| builder.cfg.blocks[id as usize].end);
            let block = builder.blk(dispatch);
            block.begin = isa.next(last_end);
            block.end = block.begin;
            let key = block.key();
            builder.cfg.functions[0].blocks.insert(key, dispatch);
        }
        Ok(builder.cfg)
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id as usize]
    }

    pub fn exits_to_dispatcher(&self) -> bool {
        self.exits_to_dispatcher
    }

    /// Graphviz rendering for debugging.
    pub fn dot(&self) -> String {
        let name = |id: BlockId| {
            let b = self.block(id);
            if b.is_virtual {
                format!("\"Virtual {:#x}\"", b.begin)
            } else {
                format!("\"{:#x}\"", b.begin)
            }
        };
        let mut dot = String::from("digraph shader {\n");
        let mut uid = 0;
        for function in &self.functions {
            let _ = writeln!(dot, "\tsubgraph cluster_{} {{", function.entry);
            for id in function.blocks() {
                let b = self.block(id);
                let n = name(id);
                let _ = writeln!(dot, "\t\t{n};");
                let mut terminal = |label: &str| {
                    let _ = writeln!(dot, "\t\t{n}->N{uid};");
                    let _ = writeln!(dot, "\t\tN{uid} [label=\"{label}\"][shape=square];");
                    uid += 1;
                };
                match b.end_class {
                    EndClass::Branch => {
                        if let (Some(t), false) = (b.branch_true, b.cond.is_false()) {
                            let _ = writeln!(dot, "\t\t{n}->{} [label=\"{}\"]", name(t), b.cond);
                        }
                        if let (Some(f), false) = (b.branch_false, b.cond.is_true()) {
                            let _ = writeln!(dot, "\t\t{n}->{}", name(f));
                        }
                    }
                    EndClass::IndirectBranch => {
                        for &(t, _) in &b.indirect_branches {
                            let _ = writeln!(dot, "\t\t{n}->{}", name(t));
                        }
                    }
                    EndClass::Call => terminal(&format!("Call {}", b.function_call)),
                    EndClass::Exit => terminal("Exit"),
                    EndClass::Return => terminal("Return"),
                    EndClass::Kill => terminal("Kill"),
                }
            }
            dot.push_str("\t}\n");
        }
        dot.push_str("}\n");
        dot
    }
}

impl Builder<'_> {
    fn new_block(&mut self, block: Block) -> BlockId {
        let id = self.cfg.blocks.len() as BlockId;
        self.cfg.blocks.push(block);
        id
    }

    fn blk(&mut self, id: BlockId) -> &mut Block {
        &mut self.cfg.blocks[id as usize]
    }

    fn stack_of(&self, id: BlockId) -> Stack {
        self.cfg.blocks[id as usize].stack.clone()
    }

    fn add_function(&mut self, entry: u32) -> FunctionId {
        let block = self.new_block(Block::new(entry, Stack::default()));
        let mut pending = FxHashMap::default();
        pending.insert(entry, block);
        self.cfg.functions.push(Function {
            entry,
            blocks: BTreeMap::new(),
            labels: vec![Label {
                address: entry,
                block,
            }],
            pending,
        });
        self.cfg.functions.len() - 1
    }

    fn error(&self, pc: u32) -> TranslateError {
        TranslateError::Unstructured {
            address: self.env.start_address() + u64::from(pc),
        }
    }

    fn analyze_label(&mut self, f: FunctionId, label: Label) -> Result<(), TranslateError> {
        if self.inspect_visited_blocks(f, label)? {
            return Ok(());
        }
        let mut pc = label.address;
        let next = self.cfg.functions[f]
            .blocks
            .range((Bound::Excluded((pc, true)), Bound::Unbounded))
            .next()
            .map(|(_, &id)| id);
        let next_begin = next.map(|id| self.cfg.blocks[id as usize].begin);
        let block = label.block;
        let mut is_branch = false;
        while next_begin.is_none_or(|b| pc < b) {
            if self.analyze_inst(block, f, pc)? == Analysis::Branch {
                is_branch = true;
                break;
            }
            pc = self.isa.next(pc);
        }
        if !is_branch {
            let b = self.blk(block);
            b.end = pc;
            b.cond = Condition::TRUE;
            b.branch_true = next;
            b.branch_false = None;
        }
        let key = self.cfg.blocks[block as usize].key();
        self.cfg.functions[f].blocks.insert(key, block);
        Ok(())
    }

    fn inspect_visited_blocks(
        &mut self,
        f: FunctionId,
        label: Label,
    ) -> Result<bool, TranslateError> {
        let pc = label.address;
        let Some((&key, &visited)) = self.cfg.functions[f]
            .blocks
            .range(..=(pc, true))
            .next_back()
        else {
            return Ok(false);
        };
        if !self.cfg.blocks[visited as usize].contains(pc) {
            return Ok(false);
        }
        if key.0 == pc {
            return Err(self.error(pc));
        }
        self.split(visited, label.block, pc);
        let new_key = self.cfg.blocks[label.block as usize].key();
        self.cfg.functions[f].blocks.insert(new_key, label.block);
        Ok(true)
    }

    fn split(&mut self, old: BlockId, new: BlockId, pc: u32) {
        let old_block = self.cfg.blocks[old as usize].clone();
        let mut head = Block::new(old_block.begin, old_block.stack.clone());
        head.end = pc;
        head.branch_true = Some(new);
        let mut tail = old_block;
        tail.begin = pc;
        tail.is_virtual = false;
        self.cfg.blocks[new as usize] = tail;
        self.cfg.blocks[old as usize] = head;
    }

    fn analyze_inst(
        &mut self,
        block: BlockId,
        f: FunctionId,
        pc: u32,
    ) -> Result<Analysis, TranslateError> {
        let FlowInst { op, cond } = self.isa.classify(self.env, pc)?;
        match op {
            FlowOp::Branch { target } => {
                if !self.analyze_branch(block, f, pc, cond) {
                    return Ok(Analysis::Continue);
                }
                let t = self.add_label(block, self.stack_of(block), target, f);
                let b = self.blk(block);
                b.branch_true = Some(t);
                b.end = pc;
                Ok(Analysis::Branch)
            }
            FlowOp::Return => {
                if !self.analyze_branch(block, f, pc, cond) {
                    return Ok(Analysis::Continue);
                }
                let b = self.blk(block);
                b.end_class = EndClass::Return;
                b.end = pc;
                Ok(Analysis::Branch)
            }
            FlowOp::Pop { token } => {
                if !self.analyze_branch(block, f, pc, cond) {
                    return Ok(Analysis::Continue);
                }
                let Some((target, stack)) = self.cfg.blocks[block as usize].stack.pop(token) else {
                    return Err(self.error(pc));
                };
                let t = self.add_label(block, stack, target, f);
                let b = self.blk(block);
                b.branch_true = Some(t);
                b.end = pc;
                Ok(Analysis::Branch)
            }
            FlowOp::Kill => {
                self.analyze_cond_inst(block, f, pc, EndClass::Kill, cond);
                Ok(Analysis::Branch)
            }
            FlowOp::Push { token, target } => {
                self.blk(block).stack.push(token, target);
                Ok(Analysis::Continue)
            }
            FlowOp::IndirectBranch { absolute } => {
                self.analyze_indirect(block, f, pc, cond, absolute)
            }
            FlowOp::Exit => self.analyze_exit(block, f, pc, cond),
            FlowOp::Call { target } => {
                let call = match self
                    .cfg
                    .functions
                    .iter()
                    .position(|func| func.entry == target)
                {
                    Some(id) => id,
                    None => self.add_function(target),
                };
                let ret = self.add_label(block, self.stack_of(block), self.isa.next(pc), f);
                let b = self.blk(block);
                b.end_class = EndClass::Call;
                b.function_call = call;
                b.return_block = Some(ret);
                b.end = pc;
                Ok(Analysis::Branch)
            }
            FlowOp::Normal => {
                if cond.is_true() || cond.is_false() {
                    return Ok(Analysis::Continue);
                }
                self.analyze_cond_inst(block, f, pc, EndClass::Branch, cond);
                Ok(Analysis::Branch)
            }
        }
    }

    /// Sets the block's condition and fall-through label; false when the branch is never taken.
    fn analyze_branch(&mut self, block: BlockId, f: FunctionId, pc: u32, cond: Condition) -> bool {
        if cond.is_false() {
            return false;
        }
        if cond.is_true() {
            self.blk(block).cond = Condition::TRUE;
        } else {
            let next = self.add_label(block, self.stack_of(block), self.isa.next(pc), f);
            let b = self.blk(block);
            b.cond = cond;
            b.branch_false = Some(next);
        }
        true
    }

    fn analyze_cond_inst(
        &mut self,
        block: BlockId,
        f: FunctionId,
        pc: u32,
        end_class: EndClass,
        cond: Condition,
    ) {
        let stack = self.stack_of(block);
        if self.cfg.blocks[block as usize].begin != pc {
            let t = self.add_label(block, stack, pc, f);
            let b = self.blk(block);
            b.end = pc;
            b.cond = Condition::TRUE;
            b.branch_true = Some(t);
            b.branch_false = None;
            return;
        }
        let mut conditional = self.cfg.blocks[block as usize].clone();
        conditional.end = self.isa.next(pc);
        conditional.end_class = end_class;
        let conditional_id = self.new_block(conditional);
        let mut virtual_block = Block::new(pc, stack.clone());
        virtual_block.is_virtual = true;
        virtual_block.cond = cond;
        virtual_block.branch_true = Some(conditional_id);
        self.cfg.blocks[block as usize] = virtual_block;
        let endif = self.add_label(conditional_id, stack, self.isa.next(pc), f);
        self.blk(block).branch_false = Some(endif);
        if matches!(end_class, EndClass::Branch | EndClass::Kill) {
            let c = self.blk(conditional_id);
            c.cond = Condition::TRUE;
            c.branch_true = Some(endif);
            c.branch_false = None;
        }
        let key = self.cfg.blocks[conditional_id as usize].key();
        self.cfg.functions[f].blocks.insert(key, conditional_id);
    }

    fn analyze_indirect(
        &mut self,
        block: BlockId,
        f: FunctionId,
        pc: u32,
        cond: Condition,
        absolute: bool,
    ) -> Result<Analysis, TranslateError> {
        let address = self.env.start_address() + u64::from(pc);
        let table = self
            .isa
            .track_indirect_branch(self.env, pc, self.program_start)
            .ok_or(TranslateError::UntrackedIndirectBranch { address })?;
        if !cond.is_true() {
            return Err(TranslateError::UntrackedIndirectBranch { address });
        }
        let bias = if absolute { 0 } else { pc };
        let mut targets: Vec<u32> = (0..table.num_entries)
            .map(|i| {
                let entry = self
                    .env
                    .read_cbuf(table.cbuf_index, table.cbuf_offset + i * 4);
                entry
                    .wrapping_add(bias)
                    .wrapping_add(table.branch_offset as u32)
            })
            .collect();
        targets.sort_unstable();
        targets.dedup();
        let stack = self.stack_of(block);
        let branches: Vec<(BlockId, u32)> = targets
            .into_iter()
            .map(|target| (self.add_label(block, stack.clone(), target, f), target))
            .collect();
        let next = self.isa.next(pc);
        let b = self.blk(block);
        b.indirect_branches = branches;
        b.cond = Condition::TRUE;
        b.end = next;
        b.end_class = EndClass::IndirectBranch;
        b.branch_reg = table.reg;
        b.branch_offset = table.branch_offset.wrapping_add(bias as i32);
        Ok(Analysis::Branch)
    }

    fn analyze_exit(
        &mut self,
        block: BlockId,
        f: FunctionId,
        pc: u32,
        cond: Condition,
    ) -> Result<Analysis, TranslateError> {
        if cond.is_false() {
            return Ok(Analysis::Continue);
        }
        let dispatch = self.dispatch_block;
        if dispatch.is_some() && f != 0 {
            return Err(self.error(pc));
        }
        let stack = self.stack_of(block);
        if !cond.is_true() {
            if stack.peek(Token::Pexit).is_some() {
                return Err(self.error(pc));
            }
            if let Some(dispatch) = dispatch {
                let next = self.add_label(block, stack, self.isa.next(pc), f);
                let b = self.blk(block);
                b.end = pc;
                b.end_class = EndClass::Branch;
                b.cond = cond;
                b.branch_true = Some(dispatch);
                b.branch_false = Some(next);
                return Ok(Analysis::Branch);
            }
            self.analyze_cond_inst(block, f, pc, EndClass::Exit, cond);
            return Ok(Analysis::Branch);
        }
        if let Some(exit_pc) = stack.peek(Token::Pexit) {
            let t = self.add_label(block, stack.remove(Token::Pexit), exit_pc, f);
            let b = self.blk(block);
            b.cond = Condition::TRUE;
            b.branch_true = Some(t);
            b.branch_false = None;
            return Ok(Analysis::Branch);
        }
        let next = self.isa.next(pc);
        let b = self.blk(block);
        if let Some(dispatch) = dispatch {
            b.cond = Condition::TRUE;
            b.end = pc;
            b.end_class = EndClass::Branch;
            b.branch_true = Some(dispatch);
            b.branch_false = None;
            return Ok(Analysis::Branch);
        }
        b.end = next;
        b.end_class = EndClass::Exit;
        Ok(Analysis::Branch)
    }

    fn add_label(&mut self, block: BlockId, stack: Stack, pc: u32, f: FunctionId) -> BlockId {
        let current = &self.cfg.blocks[block as usize];
        if current.begin == pc && !current.is_virtual {
            return block;
        }
        let function = &self.cfg.functions[f];
        if let Some(&id) = function.blocks.get(&(pc, true)) {
            return function.blocks.get(&(pc, false)).copied().unwrap_or(id);
        }
        if let Some(&id) = function.pending.get(&pc) {
            return id;
        }
        let id = self.new_block(Block::new(pc, stack));
        let function = &mut self.cfg.functions[f];
        function.labels.push(Label {
            address: pc,
            block: id,
        });
        function.pending.insert(pc, id);
        id
    }
}
