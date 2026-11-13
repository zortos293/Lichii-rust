//! Gives each geometry primitive its own output slot on hosts whose geometry subgroups have one
//! lane, ported from `geometry_compaction_pass.cpp`.

use rustc_hash::FxHashMap;

use crate::ShaderStage;
use crate::ir::{Attribute, BlockId, Emitter, InstId, Opcode, Program, Value};
use crate::profile::HostTranslateInfo;

const MAX_BALLOT_DEPTH: u8 = 8;

type BallotMemo = FxHashMap<(InstId, u8), Option<InstId>>;

/// The `SubgroupBallot` feeding `value` within `MAX_BALLOT_DEPTH` instructions, not looking
/// through phis. Memoized per instruction and depth, so every search together stays linear.
fn find_ballot(prog: &Program, memo: &mut BallotMemo, value: Value, depth: u8) -> Option<InstId> {
    if depth > MAX_BALLOT_DEPTH {
        return None;
    }
    let id = prog.resolve(value).inst()?;
    if let Some(&found) = memo.get(&(id, depth)) {
        return found;
    }
    let found = match prog.op(id) {
        Opcode::SubgroupBallot => Some(id),
        Opcode::Phi => None,
        _ => (0..prog.inst(id).num_args())
            .find_map(|index| find_ballot(prog, memo, prog.arg(id, index), depth + 1)),
    };
    memo.insert((id, depth), found);
    found
}

fn is_shuffle(op: Opcode) -> bool {
    matches!(
        op,
        Opcode::ShuffleIndex | Opcode::ShuffleUp | Opcode::ShuffleDown | Opcode::ShuffleButterfly
    )
}

/// `amount` times the index of this invocation's primitive among all primitives of the draw.
fn primitive_slot(ir: &mut Emitter<'_>, invocations: u32, amount: Value) -> Value {
    let mut key = ir.get_attribute_u32(Attribute::PRIMITIVE_ID, Value::U32(0));
    if invocations > 1 {
        let scaled = ir.imul(key, Value::U32(invocations));
        let invocation = ir.inst(Opcode::InvocationId, &[]);
        key = ir.iadd(scaled, invocation);
    }
    ir.imul(key, amount)
}

fn merges_atomic(prog: &Program, phi: InstId, block: BlockId, atomic: InstId) -> bool {
    (0..prog.inst(phi).num_args()).any(|index| {
        prog.inst(phi).phi_block(index) == block
            && prog.resolve(prog.arg(phi, index)) == Value::Inst(atomic)
    })
}

/// A phi merging the atomic's result with a shuffle of it (the slot broadcast to the other
/// lanes) takes the primitive slot on the shuffle's path too.
fn rewrite_merged_slots(prog: &mut Program, block: BlockId, atomic: InstId, invocations: u32) {
    let successors = prog.block(block).successors.clone();
    for successor in successors {
        let phis: Vec<InstId> = prog
            .insts_of(successor)
            .filter(|&id| prog.op(id) == Opcode::Phi && merges_atomic(prog, id, block, atomic))
            .collect();
        for phi in phis {
            for index in 0..prog.inst(phi).num_args() {
                let Some(shuffle) = prog.resolve(prog.arg(phi, index)).inst() else {
                    continue;
                };
                if !is_shuffle(prog.op(shuffle)) {
                    continue;
                }
                let amount = prog.arg(shuffle, 0);
                let predecessor = prog.inst(phi).phi_block(index);
                let mut ir = Emitter::new(prog, predecessor);
                let slot = primitive_slot(&mut ir, invocations, amount);
                prog.set_arg(phi, index, slot);
            }
        }
    }
}

/// Replaces the slot allocation with the primitive's own slot, keeping the counter at the end
/// of the highest slot written.
fn rewrite_atomic(prog: &mut Program, atomic: InstId, invocations: u32) {
    let (binding, offset, amount) = (
        prog.arg(atomic, 0),
        prog.arg(atomic, 1),
        prog.arg(atomic, 2),
    );
    let mut ir = Emitter::before(prog, atomic);
    let slot = primitive_slot(&mut ir, invocations, amount);
    let end = ir.iadd(slot, amount);
    ir.inst(Opcode::StorageAtomicUMax32, &[binding, offset, end]);
    prog.replace_uses_with(atomic, slot);
}

/// Geometry programs that compact their output allocate slots with a storage atomic add of a
/// ballot count, then broadcast the slot with a shuffle. With one lane per subgroup every
/// invocation allocates alone, in an order the host does not keep, so each primitive takes
/// the slot its primitive id (and invocation) gives instead, and the ballot's predicate becomes
/// true.
pub fn geometry_compaction(
    prog: &mut Program,
    stage: ShaderStage,
    invocations: u32,
    host_info: &HostTranslateInfo,
) {
    if stage != ShaderStage::Geometry || !host_info.single_lane_geometry_subgroups {
        return;
    }
    let mut memo = BallotMemo::default();
    let mut slot_atomics = Vec::new();
    for &block in &prog.post_order {
        for id in prog.insts_of(block) {
            if prog.op(id) != Opcode::StorageAtomicIAdd32 {
                continue;
            }
            if let Some(ballot) = find_ballot(prog, &mut memo, prog.arg(id, 2), 0) {
                slot_atomics.push((id, ballot));
            }
        }
    }
    for (atomic, ballot) in slot_atomics {
        crate::fallback::note("geometry", "compaction_slot_per_primitive");
        let block = prog.inst(atomic).block();
        rewrite_merged_slots(prog, block, atomic, invocations);
        rewrite_atomic(prog, atomic, invocations);
        if let Some(pred) = prog.resolve(prog.arg(ballot, 0)).inst() {
            prog.replace_uses_with(pred, Value::U1(true));
        }
    }
}
