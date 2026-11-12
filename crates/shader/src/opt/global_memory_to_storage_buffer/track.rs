//! Tracking a global address back to the constant buffer word that holds its base pointer.
//!
//! Every search here is memoized by the instruction it starts from (and, for the
//! condition-aware search, the block context), and visited sets are dense stamps, so a program
//! with thousands of accesses through a few addresses walks each address once.

use std::collections::VecDeque;

use rustc_hash::{FxHashMap, FxHashSet};

use super::conditions::Conditions;
use super::{BufferAddr, imm_u32, inst_of, is_global_memory};
use crate::ir::{BlockId, InstId, Opcode, Program, Value};

/// The low word of a global address (`value` plus `imm_offset`) and its high word.
#[derive(Clone, Copy, Debug)]
pub(super) struct LowAddr {
    pub value: Value,
    pub high: Value,
    pub imm_offset: i32,
}

/// Splits the address of a global memory instruction, expected as
/// `IAdd64(PackUint2x32(CompositeConstructU32x2(low, high)), imm)` with the add and pack optional.
pub(super) fn track_low_address(prog: &Program, inst: InstId) -> Option<LowAddr> {
    let mut addr = inst_of(prog, prog.arg(inst, 0))?;
    let mut imm_offset = 0;
    if prog.op(addr) == Opcode::IAdd64 {
        let offset = prog.resolve(prog.arg(addr, 1));
        if !offset.is_immediate() {
            return None;
        }
        imm_offset = offset.imm_u64()? as i32;
        addr = inst_of(prog, prog.arg(addr, 0))?;
    }
    if prog.op(addr) == Opcode::PackUint2x32 {
        addr = inst_of(prog, prog.arg(addr, 0))?;
    }
    if prog.op(addr) != Opcode::CompositeConstructU32x2 {
        return None;
    }
    Some(LowAddr {
        value: prog.resolve(prog.arg(addr, 0)),
        high: prog.resolve(prog.arg(addr, 1)),
        imm_offset,
    })
}

/// The constant buffer word an instruction reads, including one half of a 64-bit read.
pub(super) fn cbuf_word(prog: &Program, inst: InstId) -> Option<BufferAddr> {
    match prog.op(inst) {
        Opcode::CompositeExtractU32x2 => {
            let vector = inst_of(prog, prog.arg(inst, 0))?;
            if prog.op(vector) != Opcode::GetCbufU32x2 {
                return None;
            }
            let index = imm_u32(prog, prog.arg(vector, 0))?;
            let offset = imm_u32(prog, prog.arg(vector, 1))?;
            let element = imm_u32(prog, prog.arg(inst, 1))?;
            Some(BufferAddr::new(
                index,
                offset.wrapping_add(element.wrapping_mul(4)),
            ))
        }
        Opcode::GetCbufU32 => Some(BufferAddr::new(
            imm_u32(prog, prog.arg(inst, 0))?,
            imm_u32(prog, prog.arg(inst, 1))?,
        )),
        _ => None,
    }
}

/// A visited set over instruction ids that clears in O(1).
pub(super) struct Marks {
    stamps: Vec<u32>,
    epoch: u32,
}

impl Marks {
    pub(super) fn new(len: usize) -> Marks {
        Marks {
            stamps: vec![0; len],
            epoch: 1,
        }
    }

    pub(super) fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamps.fill(0);
            self.epoch = 1;
        }
    }

    pub(super) fn insert(&mut self, id: InstId) -> bool {
        let stamp = &mut self.stamps[id.0 as usize];
        let fresh = *stamp != self.epoch;
        *stamp = self.epoch;
        fresh
    }

    pub(super) fn contains(&self, id: InstId) -> bool {
        self.stamps[id.0 as usize] == self.epoch
    }
}

/// Bias towards a range of constant buffer offsets when looking for storage buffers.
pub(super) struct Bias {
    pub index: u32,
    pub offset_begin: u32,
    pub offset_end: u32,
    pub alignment: u32,
}

/// NVN puts storage buffer descriptors in this range of constant buffer 0.
pub(super) const NVN_BIAS: Bias = Bias {
    index: 0,
    offset_begin: 0x100,
    offset_end: 0x700,
    alignment: 16,
};

/// The 64-bit constant buffer pointer summed into an address, found by
/// [`Tracker::pointer_addend`]: `strict` needs a single candidate, `nearest` takes the
/// nearest of several unless two tie.
#[derive(Clone, Copy, Default)]
pub(super) struct PointerAddend {
    pub strict: Option<BufferAddr>,
    pub nearest: Option<BufferAddr>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Addend {
    Unknown,
    Zero,
    One(BufferAddr),
    Many,
}

impl Addend {
    fn plus(self, other: Addend) -> Addend {
        match (self, other) {
            (Addend::Zero | Addend::Unknown, x) | (x, Addend::Zero | Addend::Unknown) => x,
            _ => Addend::Many,
        }
    }
}

#[derive(Default)]
struct TrackMemo {
    results: FxHashMap<InstId, Option<BufferAddr>>,
    exhausted: Vec<bool>,
}

/// Memoized searches over a program that does not change while storage buffers are collected.
pub(super) struct Tracker {
    sum_marks: [Marks; 2],
    sum_queue: VecDeque<(InstId, u32, bool)>,
    /// Scratch visited set for one-off walks.
    pub(super) marks: Marks,
    bfs_queue: VecDeque<InstId>,
    bfs_visited: Vec<InstId>,
    pointer_addends: FxHashMap<(InstId, Option<InstId>, u32), PointerAddend>,
    addends: Vec<Addend>,
    addend_stack: Vec<(InstId, bool)>,
    biased: TrackMemo,
    unbiased: TrackMemo,
}

impl Tracker {
    pub(super) fn new(prog: &Program) -> Tracker {
        let n = prog.num_insts();
        Tracker {
            sum_marks: [Marks::new(n), Marks::new(n)],
            sum_queue: VecDeque::new(),
            marks: Marks::new(n),
            bfs_queue: VecDeque::new(),
            bfs_visited: Vec::new(),
            pointer_addends: FxHashMap::default(),
            addends: vec![Addend::Unknown; n],
            addend_stack: Vec::new(),
            biased: TrackMemo {
                exhausted: vec![false; n],
                ..TrackMemo::default()
            },
            unbiased: TrackMemo {
                exhausted: vec![false; n],
                ..TrackMemo::default()
            },
        }
    }

    /// Appends to `out` the constant buffer words summed into `value` with their breadth-first
    /// depth, nearest first, looking through additions, selects and phis. With `conditions`,
    /// phi inputs that cannot reach an access in the given block are skipped, and words past a
    /// loop back edge are no longer filtered.
    pub(super) fn for_each_summed_word(
        &mut self,
        prog: &Program,
        value: Value,
        conditions: Option<(&Conditions, BlockId)>,
        out: &mut Vec<(BufferAddr, u32)>,
    ) {
        let Some(start) = inst_of(prog, value) else {
            return;
        };
        self.sum_marks[0].clear();
        self.sum_marks[1].clear();
        self.sum_queue.clear();
        self.sum_queue.push_back((start, 0, true));
        while let Some((id, depth, same_iteration)) = self.sum_queue.pop_front() {
            if !self.sum_marks[usize::from(same_iteration)].insert(id) {
                continue;
            }
            if let Some(word) = cbuf_word(prog, id) {
                out.push((word, depth));
                continue;
            }
            let op = prog.op(id);
            if !matches!(op, Opcode::IAdd32 | Opcode::Phi | Opcode::SelectU32) {
                continue;
            }
            let first = usize::from(op == Opcode::SelectU32);
            for arg in first..prog.inst(id).num_args() {
                let is_phi = op == Opcode::Phi;
                if is_phi
                    && same_iteration
                    && let Some((conditions, current)) = conditions
                    && !conditions.allows_phi(prog, id, arg, current)
                {
                    continue;
                }
                let next_iteration = same_iteration
                    && !(is_phi && conditions.is_some_and(|(c, _)| c.is_back_edge(prog, id, arg)));
                if let Some(next) = inst_of(prog, prog.arg(id, arg)) {
                    self.sum_queue.push_back((next, depth + 1, next_iteration));
                }
            }
        }
    }

    /// The 64-bit constant buffer pointer summed into an address: its low word is summed into
    /// the low half and its high word into the high half. Iterator steps and 32-bit scalars
    /// (strides, sizes) are skipped.
    pub(super) fn pointer_addend(
        &mut self,
        prog: &Program,
        addr: &LowAddr,
        iterator_words: &FxHashSet<BufferAddr>,
        conditions: &Conditions,
        block: BlockId,
    ) -> PointerAddend {
        let Some(low) = inst_of(prog, addr.value) else {
            return PointerAddend::default();
        };
        let key = (low, inst_of(prog, addr.high), conditions.context(block));
        if let Some(&memo) = self.pointer_addends.get(&key) {
            return memo;
        }
        let mut words = Vec::new();
        self.for_each_summed_word(prog, addr.high, Some((conditions, block)), &mut words);
        let high_words: FxHashSet<BufferAddr> = words.drain(..).map(|(word, _)| word).collect();
        self.for_each_summed_word(prog, addr.value, Some((conditions, block)), &mut words);
        let mut result: Option<(BufferAddr, u32)> = None;
        let mut tied = false;
        let mut candidates = FxHashSet::default();
        for (word, depth) in words {
            if !word.offset.is_multiple_of(8)
                || iterator_words.contains(&word)
                || !high_words.contains(&word.high_word())
            {
                continue;
            }
            candidates.insert(word);
            match result {
                Some((_, nearest)) if depth > nearest => {}
                Some((found, nearest)) if depth == nearest => tied |= found != word,
                _ => {
                    result = Some((word, depth));
                    tied = false;
                }
            }
        }
        let nearest = result.filter(|_| !tied).map(|(word, _)| word);
        let addend = PointerAddend {
            strict: nearest.filter(|_| candidates.len() == 1),
            nearest,
        };
        self.pointer_addends.insert(key, addend);
        addend
    }

    /// The one 8-byte aligned constant buffer word added directly into the low address through
    /// a tree of 32-bit additions; none when several are.
    pub(super) fn track_addend(&mut self, prog: &Program, value: Value) -> Option<BufferAddr> {
        let root = inst_of(prog, value)?;
        self.addend_stack.clear();
        self.addend_stack.push((root, false));
        while let Some((id, expanded)) = self.addend_stack.pop() {
            if !expanded && self.addends[id.0 as usize] != Addend::Unknown {
                continue;
            }
            if prog.op(id) != Opcode::IAdd32 {
                self.addends[id.0 as usize] = addend_leaf(prog, id);
                continue;
            }
            let terms = [
                inst_of(prog, prog.arg(id, 0)),
                inst_of(prog, prog.arg(id, 1)),
            ];
            if expanded {
                self.addends[id.0 as usize] = terms
                    .iter()
                    .map(|term| term.map_or(Addend::Zero, |t| self.addends[t.0 as usize]))
                    .fold(Addend::Zero, Addend::plus);
                continue;
            }
            self.addend_stack.push((id, true));
            for term in terms.into_iter().flatten() {
                if self.addends[term.0 as usize] == Addend::Unknown {
                    self.addend_stack.push((term, false));
                }
            }
        }
        match self.addends[root.0 as usize] {
            Addend::One(word) => Some(word),
            _ => None,
        }
    }

    /// The first constant buffer read, breadth first and rightmost arguments first, that can
    /// hold a storage buffer pointer under `bias`. Pointers loaded from memory are new
    /// addresses, so the search does not walk through global loads.
    pub(super) fn track(
        &mut self,
        prog: &Program,
        value: Value,
        bias: Option<&Bias>,
    ) -> Option<BufferAddr> {
        let start = inst_of(prog, value)?;
        let memo = if bias.is_some() {
            &mut self.biased
        } else {
            &mut self.unbiased
        };
        if let Some(&result) = memo.results.get(&start) {
            return result;
        }
        let result = breadth_first_search(
            prog,
            start,
            bias,
            &mut memo.exhausted,
            &mut self.marks,
            &mut self.bfs_queue,
            &mut self.bfs_visited,
        );
        memo.results.insert(start, result);
        result
    }
}

fn addend_leaf(prog: &Program, id: InstId) -> Addend {
    if prog.op(id) != Opcode::GetCbufU32 {
        return Addend::Zero;
    }
    match (
        imm_u32(prog, prog.arg(id, 0)),
        imm_u32(prog, prog.arg(id, 1)),
    ) {
        (Some(index), Some(offset)) if offset.is_multiple_of(8) => {
            Addend::One(BufferAddr::new(index, offset))
        }
        _ => Addend::Zero,
    }
}

fn bias_match(prog: &Program, id: InstId, bias: Option<&Bias>) -> Option<BufferAddr> {
    if !matches!(prog.op(id), Opcode::GetCbufU32 | Opcode::GetCbufU32x2) {
        return None;
    }
    let addr = BufferAddr::new(
        imm_u32(prog, prog.arg(id, 0))?,
        imm_u32(prog, prog.arg(id, 1))?,
    );
    let alignment = bias.map_or(8, |b| b.alignment);
    if !addr.offset.is_multiple_of(alignment) {
        return None;
    }
    if let Some(b) = bias
        && (addr.index != b.index || addr.offset < b.offset_begin || addr.offset >= b.offset_end)
    {
        return None;
    }
    Some(addr)
}

/// A breadth-first search that skips instructions a failed search already proved fruitless,
/// and marks everything it visited as fruitless when it fails.
fn breadth_first_search(
    prog: &Program,
    start: InstId,
    bias: Option<&Bias>,
    exhausted: &mut [bool],
    marks: &mut Marks,
    queue: &mut VecDeque<InstId>,
    visited: &mut Vec<InstId>,
) -> Option<BufferAddr> {
    let skipped =
        |exhausted: &[bool], id: InstId| exhausted[id.0 as usize] || is_global_memory(prog.op(id));
    if skipped(exhausted, start) {
        return None;
    }
    marks.clear();
    queue.clear();
    visited.clear();
    queue.push_back(start);
    while let Some(id) = queue.pop_front() {
        visited.push(id);
        if let Some(found) = bias_match(prog, id, bias) {
            return Some(found);
        }
        for &arg in prog.inst(id).args().iter().rev() {
            let Some(next) = inst_of(prog, arg) else {
                continue;
            };
            if !skipped(exhausted, next) && marks.insert(next) {
                queue.push_back(next);
            }
        }
    }
    for id in visited.drain(..) {
        exhausted[id.0 as usize] = true;
    }
    None
}
