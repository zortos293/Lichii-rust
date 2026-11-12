//! Program-wide facts that tell base pointers apart from other constant buffer words summed into
//! addresses: strided iterator steps, and pointers that global atomics address.

use std::collections::VecDeque;

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use super::track::{Marks, Tracker, cbuf_word, track_low_address};
use super::{BufferAddr, inst_of, is_global_memory_write};
use crate::ir::{InstId, Opcode, Program, Type, Value};

const NONE: u32 = u32::MAX;
const SUM_BUDGET: u32 = 64;
const READ_BUDGET: usize = 64;

/// Distinct words of a bounded addition tree (at most [`SUM_BUDGET`] nodes).
struct WordSum {
    words: SmallVec<[BufferAddr; 8]>,
    budget: u32,
    reaches_loop: bool,
}

impl WordSum {
    fn new() -> WordSum {
        WordSum {
            words: SmallVec::new(),
            budget: SUM_BUDGET,
            reaches_loop: false,
        }
    }
}

/// Collects the constant buffer words of an addition tree and returns false when another value
/// feeds it. Reaching `loop_phi` is recorded in the sum.
fn sum_words(prog: &Program, value: Value, loop_phi: Option<InstId>, sum: &mut WordSum) -> bool {
    let Some(id) = inst_of(prog, value) else {
        return true;
    };
    if sum.budget == 0 {
        return false;
    }
    sum.budget -= 1;
    if Some(id) == loop_phi {
        sum.reaches_loop = true;
        return false;
    }
    if let Some(word) = cbuf_word(prog, id) {
        if !sum.words.contains(&word) {
            sum.words.push(word);
        }
        return true;
    }
    if prog.op(id) != Opcode::IAdd32 {
        return false;
    }
    let lhs = sum_words(prog, prog.arg(id, 0), loop_phi, sum);
    if !lhs && loop_phi.is_none() {
        return false;
    }
    let rhs = sum_words(prog, prog.arg(id, 1), loop_phi, sum);
    lhs && rhs
}

/// Iterative Tarjan strongly connected components over additions, selects and phis.
struct Tarjan {
    index: Vec<u32>,
    lowlink: Vec<u32>,
    on_stack: Vec<bool>,
    stack: Vec<InstId>,
    calls: Vec<(InstId, usize)>,
    next_index: u32,
}

impl Tarjan {
    fn enter(&mut self, prog: &Program, id: InstId) {
        self.index[id.0 as usize] = self.next_index;
        self.lowlink[id.0 as usize] = self.next_index;
        self.next_index += 1;
        self.stack.push(id);
        self.on_stack[id.0 as usize] = true;
        self.calls.push((id, sum_edges(prog, id).start));
    }
}

fn sum_edges(prog: &Program, id: InstId) -> std::ops::Range<usize> {
    match prog.op(id) {
        Opcode::IAdd32 | Opcode::Phi => 0..prog.inst(id).num_args(),
        Opcode::SelectU32 => 1..prog.inst(id).num_args(),
        _ => 0..0,
    }
}

/// The constant buffer words that step strided iterators (CUTLASS-style kernel parameters):
/// words added to a loop-carried value, and words summed with two or more other words into an
/// offset table. A base pointer is neither.
pub(super) fn collect_iterator_words(
    prog: &Program,
    tracker: &mut Tracker,
) -> FxHashSet<BufferAddr> {
    let mut result = FxHashSet::default();
    let mut phis = Vec::new();
    for &block in &prog.post_order {
        for id in prog.insts_of(block) {
            match prog.op(id) {
                Opcode::Phi => {
                    phis.push(id);
                    for &arg in prog.inst(id).args() {
                        let mut sum = WordSum::new();
                        sum_words(prog, arg, Some(id), &mut sum);
                        if sum.reaches_loop {
                            result.extend(sum.words);
                        }
                    }
                }
                Opcode::IAdd32 => {
                    let mut sum = WordSum::new();
                    if sum_words(prog, Value::Inst(id), None, &mut sum) && sum.words.len() >= 3 {
                        result.extend(sum.words);
                    }
                }
                _ => {}
            }
        }
    }
    let n = prog.num_insts();
    let mut tarjan = Tarjan {
        index: vec![NONE; n],
        lowlink: vec![0; n],
        on_stack: vec![false; n],
        stack: Vec::new(),
        calls: Vec::new(),
        next_index: 0,
    };
    let mut members = Marks::new(n);
    let mut words = Vec::new();
    for root in phis {
        if tarjan.index[root.0 as usize] != NONE {
            continue;
        }
        tarjan.enter(prog, root);
        while let Some(&(id, arg)) = tarjan.calls.last() {
            let v = id.0 as usize;
            if arg < sum_edges(prog, id).end {
                if let Some(call) = tarjan.calls.last_mut() {
                    call.1 += 1;
                }
                let Some(next) = inst_of(prog, prog.arg(id, arg)) else {
                    continue;
                };
                let w = next.0 as usize;
                if tarjan.index[w] == NONE {
                    tarjan.enter(prog, next);
                } else if tarjan.on_stack[w] {
                    tarjan.lowlink[v] = tarjan.lowlink[v].min(tarjan.index[w]);
                }
                continue;
            }
            tarjan.calls.pop();
            if let Some(&(parent, _)) = tarjan.calls.last() {
                let p = parent.0 as usize;
                tarjan.lowlink[p] = tarjan.lowlink[p].min(tarjan.lowlink[v]);
            }
            if tarjan.lowlink[v] != tarjan.index[v] {
                continue;
            }
            let split = tarjan.stack.iter().rposition(|&m| m == id).unwrap_or(0);
            let component: Vec<InstId> = tarjan.stack.drain(split..).collect();
            for member in &component {
                tarjan.on_stack[member.0 as usize] = false;
            }
            if component.len() < 2 {
                continue;
            }
            members.clear();
            for &member in &component {
                members.insert(member);
            }
            for &add in &component {
                if prog.op(add) != Opcode::IAdd32 {
                    continue;
                }
                for &term in prog.inst(add).args() {
                    if inst_of(prog, term).is_some_and(|t| members.contains(t)) {
                        continue;
                    }
                    tracker.for_each_summed_word(prog, term, None, &mut words);
                    result.extend(words.drain(..).map(|(word, _)| word));
                }
            }
        }
    }
    result
}

/// Whether a condition reads a constant buffer word, as the carry out of an addition to that
/// word does.
fn reads_cbuf_word(prog: &Program, condition: Value, word: BufferAddr) -> bool {
    let mut queue = VecDeque::from([condition]);
    let mut visited = FxHashSet::default();
    while visited.len() < READ_BUDGET {
        let Some(value) = queue.pop_front() else {
            break;
        };
        let Some(id) = inst_of(prog, value) else {
            continue;
        };
        if !visited.insert(id) {
            continue;
        }
        if cbuf_word(prog, id) == Some(word) {
            return true;
        }
        queue.extend(prog.inst(id).args().iter().copied());
    }
    false
}

/// The 64-bit constant buffer pointers a global atomic's address is based on: the whole address
/// apart from an immediate offset, or a 64-bit sum whose high half adds the pointer's high word
/// and the carry out of its low word. An atomic has no global fallback, so such a pointer is a
/// buffer even while it is unmapped.
pub(super) fn collect_atomic_pointers(
    prog: &Program,
    tracker: &mut Tracker,
) -> FxHashSet<BufferAddr> {
    let word = |value: Value| inst_of(prog, value).and_then(|id| cbuf_word(prog, id));
    let mut result = FxHashSet::default();
    let mut words = Vec::new();
    for &block in &prog.post_order {
        for id in prog.insts_of(block) {
            let op = prog.op(id);
            if !is_global_memory_write(op) || op.ty() == Type::VOID {
                continue;
            }
            let Some(addr) = track_low_address(prog, id) else {
                continue;
            };
            if let Some(low) = word(addr.value)
                && low.offset.is_multiple_of(8)
                && word(addr.high) == Some(low.high_word())
            {
                result.insert(low);
                continue;
            }
            let mut high_words = FxHashSet::default();
            let mut carries: SmallVec<[Value; 2]> = SmallVec::new();
            let mut pending = vec![addr.high];
            tracker.marks.clear();
            while let Some(term) = pending.pop() {
                let Some(term) = inst_of(prog, term) else {
                    continue;
                };
                if !tracker.marks.insert(term) {
                    continue;
                }
                if let Some(high_word) = cbuf_word(prog, term) {
                    high_words.insert(high_word);
                    continue;
                }
                let inst = prog.inst(term);
                match inst.op {
                    Opcode::IAdd32 | Opcode::Phi => pending.extend(inst.args().iter().copied()),
                    Opcode::SelectU32
                        if prog.resolve(inst.arg(1)).is_immediate()
                            && prog.resolve(inst.arg(2)).is_immediate() =>
                    {
                        carries.push(inst.arg(0));
                    }
                    _ => {}
                }
            }
            if carries.is_empty() {
                continue;
            }
            tracker.for_each_summed_word(prog, addr.value, None, &mut words);
            for (pointer, _) in words.drain(..) {
                if pointer.offset.is_multiple_of(8)
                    && high_words.contains(&pointer.high_word())
                    && carries
                        .iter()
                        .any(|&carry| reads_cbuf_word(prog, carry, pointer))
                {
                    result.insert(pointer);
                }
            }
        }
    }
    result
}
