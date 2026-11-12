//! Instructions, blocks and programs.
//!
//! Instructions live in one arena per program and are linked into their block in order, so
//! passes insert and remove in O(1) and never move instructions.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::opcode::Opcode;
use super::types::Type;
use super::value::{Attribute, Patch, Pred, Reg};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

const NONE: u32 = u32::MAX;

/// An instruction argument: an instruction result, an immediate, or a context name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    Void,
    Inst(InstId),
    Reg(Reg),
    Pred(Pred),
    Attribute(Attribute),
    Patch(Patch),
    U1(bool),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Value {
    pub const fn is_immediate(self) -> bool {
        !matches!(self, Value::Void | Value::Inst(_))
    }

    pub const fn inst(self) -> Option<InstId> {
        match self {
            Value::Inst(id) => Some(id),
            _ => None,
        }
    }

    /// Type of an immediate or name; instruction results need [`Program::type_of`].
    pub const fn immediate_type(self) -> Type {
        match self {
            Value::Void => Type::VOID,
            Value::Inst(_) => Type::OPAQUE,
            Value::Reg(_) => Type::REG,
            Value::Pred(_) => Type::PRED,
            Value::Attribute(_) => Type::ATTRIBUTE,
            Value::Patch(_) => Type::PATCH,
            Value::U1(_) => Type::U1,
            Value::U8(_) => Type::U8,
            Value::U16(_) => Type::U16,
            Value::U32(_) => Type::U32,
            Value::U64(_) => Type::U64,
            Value::F32(_) => Type::F32,
            Value::F64(_) => Type::F64,
        }
    }

    /// Raw bits of a 32-bit or narrower immediate.
    pub fn imm_u32(self) -> Option<u32> {
        match self {
            Value::U1(v) => Some(v as u32),
            Value::U8(v) => Some(v as u32),
            Value::U16(v) => Some(v as u32),
            Value::U32(v) => Some(v),
            Value::F32(v) => Some(v.to_bits()),
            _ => None,
        }
    }

    /// Raw bits of any numeric immediate.
    pub fn imm_u64(self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(v),
            Value::F64(v) => Some(v.to_bits()),
            v => v.imm_u32().map(u64::from),
        }
    }

    pub fn imm_bool(self) -> Option<bool> {
        match self {
            Value::U1(v) => Some(v),
            _ => None,
        }
    }

    pub fn imm_f32(self) -> Option<f32> {
        match self {
            Value::F32(v) => Some(v),
            Value::U32(v) => Some(f32::from_bits(v)),
            _ => None,
        }
    }

    pub fn imm_f64(self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(v),
            Value::U64(v) => Some(f64::from_bits(v)),
            _ => None,
        }
    }

    /// Bitwise equality, so NaN immediates compare equal to themselves.
    pub fn same(self, other: Value) -> bool {
        match (self, other) {
            (Value::F32(a), Value::F32(b)) => a.to_bits() == b.to_bits(),
            (Value::F64(a), Value::F64(b)) => a.to_bits() == b.to_bits(),
            (a, b) => a == b,
        }
    }
}

impl From<InstId> for Value {
    fn from(id: InstId) -> Value {
        Value::Inst(id)
    }
}

#[derive(Clone, Debug)]
pub struct Inst {
    pub op: Opcode,
    pub flags: u32,
    args: SmallVec<[Value; 5]>,
    phi_blocks: Vec<BlockId>,
    use_count: u32,
    block: BlockId,
    prev: u32,
    next: u32,
}

impl Inst {
    pub fn args(&self) -> &[Value] {
        &self.args
    }

    pub fn arg(&self, index: usize) -> Value {
        self.args[index]
    }

    pub fn num_args(&self) -> usize {
        self.args.len()
    }

    pub fn use_count(&self) -> u32 {
        self.use_count
    }

    pub fn has_uses(&self) -> bool {
        self.use_count > 0
    }

    pub fn block(&self) -> BlockId {
        self.block
    }

    /// Predecessor block of phi operand `index`.
    pub fn phi_block(&self, index: usize) -> BlockId {
        self.phi_blocks[index]
    }

    pub fn are_all_args_immediates(&self) -> bool {
        self.args.iter().all(|a| a.is_immediate())
    }

    pub fn next(&self) -> Option<InstId> {
        (self.next != NONE).then_some(InstId(self.next))
    }

    pub fn prev(&self) -> Option<InstId> {
        (self.prev != NONE).then_some(InstId(self.prev))
    }
}

#[derive(Clone, Debug, Default)]
pub struct Block {
    first: u32,
    last: u32,
    pub predecessors: SmallVec<[BlockId; 2]>,
    pub successors: SmallVec<[BlockId; 2]>,
}

impl Block {
    fn new() -> Block {
        Block {
            first: NONE,
            last: NONE,
            predecessors: SmallVec::new(),
            successors: SmallVec::new(),
        }
    }

    pub fn first(&self) -> Option<InstId> {
        (self.first != NONE).then_some(InstId(self.first))
    }

    pub fn last(&self) -> Option<InstId> {
        (self.last != NONE).then_some(InstId(self.last))
    }

    pub fn is_empty(&self) -> bool {
        self.first == NONE
    }
}

/// Structured control flow, in program order. Conditions are U1 values kept alive by a
/// `ConditionRef` instruction in the block that computes them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyntaxNode {
    Block(BlockId),
    If {
        cond: Value,
        body: BlockId,
        merge: BlockId,
    },
    EndIf {
        merge: BlockId,
    },
    Loop {
        body: BlockId,
        continue_block: BlockId,
        merge: BlockId,
    },
    Repeat {
        cond: Value,
        loop_header: BlockId,
        merge: BlockId,
    },
    Break {
        cond: Value,
        merge: BlockId,
        skip: BlockId,
    },
    Return,
    Unreachable,
}

/// The arenas behind a program: every instruction and block, with use counts.
#[derive(Clone, Debug, Default)]
pub struct Program {
    insts: Vec<Inst>,
    blocks: Vec<Block>,
    pseudo: FxHashMap<(InstId, Opcode), InstId>,
    /// Blocks in program order.
    pub block_order: Vec<BlockId>,
    /// Blocks in post order of the structured control flow.
    pub post_order: Vec<BlockId>,
    pub syntax_list: Vec<SyntaxNode>,
}

impl Program {
    pub fn new() -> Program {
        Program::default()
    }

    pub fn new_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(Block::new());
        id
    }

    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }

    pub fn block_mut(&mut self, id: BlockId) -> &mut Block {
        &mut self.blocks[id.0 as usize]
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_insts(&self) -> usize {
        self.insts.len()
    }

    pub fn inst(&self, id: InstId) -> &Inst {
        &self.insts[id.0 as usize]
    }

    pub fn arg(&self, id: InstId, index: usize) -> Value {
        self.insts[id.0 as usize].args[index]
    }

    pub fn op(&self, id: InstId) -> Opcode {
        self.insts[id.0 as usize].op
    }

    pub fn set_flags(&mut self, id: InstId, flags: u32) {
        self.insts[id.0 as usize].flags = flags;
    }

    /// Adds an edge between two blocks.
    pub fn add_branch(&mut self, from: BlockId, to: BlockId) {
        self.blocks[from.0 as usize].successors.push(to);
        self.blocks[to.0 as usize].predecessors.push(from);
    }

    fn alloc(&mut self, op: Opcode, args: &[Value], flags: u32, block: BlockId) -> InstId {
        let id = InstId(self.insts.len() as u32);
        self.insts.push(Inst {
            op,
            flags,
            args: SmallVec::from_slice(args),
            phi_blocks: Vec::new(),
            use_count: 0,
            block,
            prev: NONE,
            next: NONE,
        });
        for &arg in args {
            self.use_value(arg);
        }
        if op.is_pseudo()
            && let Some(Value::Inst(producer)) = args.first()
        {
            self.pseudo.insert((*producer, op), id);
        }
        id
    }

    /// Appends an instruction at the end of `block`.
    pub fn append(&mut self, block: BlockId, op: Opcode, args: &[Value], flags: u32) -> InstId {
        let id = self.alloc(op, args, flags, block);
        let b = &mut self.blocks[block.0 as usize];
        let last = b.last;
        b.last = id.0;
        if last == NONE {
            b.first = id.0;
        } else {
            self.insts[last as usize].next = id.0;
            self.insts[id.0 as usize].prev = last;
        }
        id
    }

    /// Inserts an instruction just before `before` in its block.
    pub fn insert_before(
        &mut self,
        before: InstId,
        op: Opcode,
        args: &[Value],
        flags: u32,
    ) -> InstId {
        let block = self.insts[before.0 as usize].block;
        let id = self.alloc(op, args, flags, block);
        let prev = self.insts[before.0 as usize].prev;
        self.insts[id.0 as usize].prev = prev;
        self.insts[id.0 as usize].next = before.0;
        self.insts[before.0 as usize].prev = id.0;
        if prev == NONE {
            self.blocks[block.0 as usize].first = id.0;
        } else {
            self.insts[prev as usize].next = id.0;
        }
        id
    }

    /// Inserts an instruction at the start of `block`.
    pub fn prepend(&mut self, block: BlockId, op: Opcode, args: &[Value], flags: u32) -> InstId {
        match self.block(block).first() {
            Some(first) => self.insert_before(first, op, args, flags),
            None => self.append(block, op, args, flags),
        }
    }

    /// Invalidates an instruction and unlinks it from its block. It must have no uses left.
    pub fn remove(&mut self, id: InstId) {
        self.invalidate(id);
        self.unlink(id);
    }

    /// Unlinks an instruction from its block, keeping its opcode and arguments so values that
    /// still point at it (an identity being forwarded) stay meaningful.
    pub fn unlink(&mut self, id: InstId) {
        let Inst {
            prev, next, block, ..
        } = self.insts[id.0 as usize];
        if prev == NONE {
            self.blocks[block.0 as usize].first = next;
        } else {
            self.insts[prev as usize].next = next;
        }
        if next == NONE {
            self.blocks[block.0 as usize].last = prev;
        } else {
            self.insts[next as usize].prev = prev;
        }
        let inst = &mut self.insts[id.0 as usize];
        inst.prev = NONE;
        inst.next = NONE;
    }

    /// Instruction ids of a block in order. Removing the current instruction while iterating
    /// is fine; removing the next one is not.
    pub fn insts_of(&self, block: BlockId) -> BlockInsts<'_> {
        BlockInsts {
            prog: self,
            cur: self.blocks[block.0 as usize].first,
        }
    }

    fn use_value(&mut self, value: Value) {
        if let Value::Inst(id) = value {
            self.insts[id.0 as usize].use_count += 1;
        }
    }

    fn undo_use(&mut self, value: Value) {
        if let Value::Inst(id) = value {
            let inst = &mut self.insts[id.0 as usize];
            inst.use_count -= 1;
        }
    }

    pub fn set_arg(&mut self, id: InstId, index: usize, value: Value) {
        let old = self.insts[id.0 as usize].args[index];
        self.undo_use(old);
        self.use_value(value);
        self.insts[id.0 as usize].args[index] = value;
    }

    pub fn add_phi_operand(&mut self, phi: InstId, block: BlockId, value: Value) {
        self.use_value(value);
        let inst = &mut self.insts[phi.0 as usize];
        inst.args.push(value);
        inst.phi_blocks.push(block);
    }

    /// Drops every argument and turns the instruction into `Void`.
    pub fn invalidate(&mut self, id: InstId) {
        let args = std::mem::take(&mut self.insts[id.0 as usize].args);
        for &arg in &args {
            self.undo_use(arg);
        }
        let op = self.insts[id.0 as usize].op;
        if op.is_pseudo()
            && let Some(Value::Inst(producer)) = args.first()
        {
            self.pseudo.remove(&(*producer, op));
        }
        let inst = &mut self.insts[id.0 as usize];
        inst.phi_blocks.clear();
        inst.op = Opcode::Void;
    }

    /// Turns the instruction into an `Identity` of `replacement`; uses see the replacement
    /// through [`Program::resolve`] until identity removal rewrites them.
    pub fn replace_uses_with(&mut self, id: InstId, replacement: Value) {
        self.invalidate(id);
        self.use_value(replacement);
        let inst = &mut self.insts[id.0 as usize];
        inst.op = Opcode::Identity;
        inst.args.push(replacement);
    }

    /// Changes the opcode, keeping the arguments.
    pub fn replace_opcode(&mut self, id: InstId, op: Opcode) {
        self.insts[id.0 as usize].op = op;
    }

    /// Follows identities to the value they forward.
    pub fn resolve(&self, mut value: Value) -> Value {
        while let Value::Inst(id) = value {
            let inst = &self.insts[id.0 as usize];
            if inst.op != Opcode::Identity {
                break;
            }
            value = inst.args[0];
        }
        value
    }

    /// Type of a value; a phi's type is stored in its flags.
    pub fn type_of(&self, value: Value) -> Type {
        match self.resolve(value) {
            Value::Inst(id) => {
                let inst = &self.insts[id.0 as usize];
                match inst.op {
                    Opcode::Phi => Type::from_bits_truncate(inst.flags),
                    op => op.ty(),
                }
            }
            v => v.immediate_type(),
        }
    }

    /// The pseudo-instruction (carry, zero, sparse, ...) attached to `producer`, if any.
    pub fn pseudo_op(&self, producer: InstId, op: Opcode) -> Option<InstId> {
        self.pseudo.get(&(producer, op)).copied()
    }
}

pub struct BlockInsts<'a> {
    prog: &'a Program,
    cur: u32,
}

impl Iterator for BlockInsts<'_> {
    type Item = InstId;

    fn next(&mut self) -> Option<InstId> {
        if self.cur == NONE {
            return None;
        }
        let id = InstId(self.cur);
        self.cur = self.prog.insts[self.cur as usize].next;
        Some(id)
    }
}
