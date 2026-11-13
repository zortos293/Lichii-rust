//! Switch 2 native programs: every constant buffer pointer a low address can come from.

use rustc_hash::FxHashSet;
use smallvec::SmallVec;

use super::track::{Marks, track_low_address};
use super::{BufferAddr, StorageInfo, imm_u32, inst_of};
use crate::ir::{InstId, Opcode, Program, Value};
use crate::{ShaderEnvironment, ShaderStage};

const MAX_SLOTS: u32 = 16;
const MAX_HOLDERS: usize = 8;

pub(super) type Candidates = SmallVec<[BufferAddr; 4]>;
type Slots = SmallVec<[(u32, u32); 4]>;

/// The first slot and stride of a pointer array read at a dynamic constant buffer offset:
/// `(index << shift) + base` or `index * stride + base`.
fn dynamic_pointer_slots(prog: &Program, offset: Value) -> Option<(u32, u32)> {
    let stride_of = |value: Value| -> Option<u32> {
        let id = inst_of(prog, value)?;
        if !matches!(prog.op(id), Opcode::ShiftLeftLogical32 | Opcode::IMul32) {
            return None;
        }
        match (prog.op(id), imm_u32(prog, prog.arg(id, 1))?) {
            (Opcode::ShiftLeftLogical32, shift) if shift < 16 => Some(1 << shift),
            (Opcode::IMul32, stride) => Some(stride),
            _ => None,
        }
    };
    let mut base = 0;
    let mut stride = stride_of(offset);
    if stride.is_none()
        && let Some(add) = inst_of(prog, offset)
        && prog.op(add) == Opcode::IAdd32
    {
        for arg in 0..2 {
            if stride.is_some() {
                break;
            }
            if let Some(constant) = imm_u32(prog, prog.arg(add, arg ^ 1)) {
                base = constant;
                stride = stride_of(prog.arg(add, arg));
            }
        }
    }
    let stride = stride?;
    (stride >= 8 && stride.is_multiple_of(8)).then_some((base, stride))
}

/// Slots of a pointer array read at a dynamic offset, following phis and selects of the
/// offset. A stride of zero is a single slot.
fn collect_pointer_slots(prog: &Program, offset: Value) -> Slots {
    fn visit(prog: &Program, offset: Value, slots: &mut Slots, visited: &mut FxHashSet<InstId>) {
        let offset = prog.resolve(offset);
        let Some(id) = offset.inst() else {
            if let Some(constant) = offset.imm_u32()
                && !slots.contains(&(constant, 0))
            {
                slots.push((constant, 0));
            }
            return;
        };
        if !visited.insert(id) {
            return;
        }
        if let Some(pattern) = dynamic_pointer_slots(prog, offset) {
            if !slots.contains(&pattern) {
                slots.push(pattern);
            }
            return;
        }
        let inst = prog.inst(id);
        match inst.op {
            Opcode::Phi => {
                for &arg in inst.args() {
                    visit(prog, arg, slots, visited);
                }
            }
            Opcode::SelectU32 => {
                visit(prog, inst.arg(1), slots, visited);
                visit(prog, inst.arg(2), slots, visited);
            }
            _ => {}
        }
    }
    let mut slots = Slots::new();
    visit(prog, offset, &mut slots, &mut FxHashSet::default());
    slots
}

fn slot_addresses(slots: &Slots, index: u32, indirect: u32) -> impl Iterator<Item = BufferAddr> {
    slots.iter().flat_map(move |&(first, stride)| {
        let count = if stride != 0 { MAX_SLOTS } else { 1 };
        (0..count).map(move |slot| BufferAddr {
            index,
            offset: first.wrapping_add(slot.wrapping_mul(stride)),
            indirect,
        })
    })
}

/// Every constant buffer pointer the low address `start` can come from, following additions,
/// selects and phis, and whether any came from a dynamically indexed pointer array or a
/// pointer table. A predicated address assignment becomes a phi whose other input can point
/// into a different buffer, so the search does not stop at the first pointer. A pointer loaded
/// from a table that a constant buffer pointer addresses becomes an indirect candidate.
pub(super) fn native_pointer_candidates(
    prog: &Program,
    start: InstId,
    env: &dyn ShaderEnvironment,
    info: &mut StorageInfo,
    marks: &mut Marks,
) -> (Candidates, bool) {
    let mut candidates = Candidates::new();
    let mut dynamic = false;
    let mut pending = vec![start];
    marks.clear();
    marks.insert(start);
    let mut next = 0;
    while next < pending.len() {
        let id = pending[next];
        next += 1;
        let inst = prog.inst(id);
        let mut enqueue = |value: Value| {
            if let Some(arg) = inst_of(prog, value)
                && marks.insert(arg)
            {
                pending.push(arg);
            }
        };
        match inst.op {
            Opcode::GetCbufU32 | Opcode::GetCbufU32x2 => {
                let Some(index) = imm_u32(prog, inst.arg(0)) else {
                    continue;
                };
                let Some(offset) = imm_u32(prog, inst.arg(1)) else {
                    let slots = collect_pointer_slots(prog, inst.arg(1));
                    for pointer in slot_addresses(&slots, index, BufferAddr::DIRECT) {
                        if pointer.offset.is_multiple_of(8)
                            && !candidates.contains(&pointer)
                            && env.is_cbuf_storage_buffer_pointer(pointer.index, pointer.offset)
                        {
                            candidates.push(pointer);
                            dynamic = true;
                        }
                    }
                    continue;
                };
                let pointer = BufferAddr::new(index, offset);
                if offset.is_multiple_of(8)
                    && !candidates.contains(&pointer)
                    && env.is_cbuf_storage_buffer_pointer(index, offset)
                {
                    candidates.push(pointer);
                }
            }
            Opcode::IAdd32 | Opcode::Phi => {
                for &arg in inst.args().iter().rev() {
                    enqueue(arg);
                }
            }
            Opcode::SelectU32 => {
                enqueue(inst.arg(1));
                enqueue(inst.arg(2));
            }
            Opcode::ShuffleIndex => enqueue(inst.arg(0)),
            Opcode::CompositeExtractU32x2 => {
                if imm_u32(prog, inst.arg(1)) == Some(0) {
                    enqueue(inst.arg(0));
                }
            }
            Opcode::LoadGlobal64 => {
                table_candidates(prog, id, env, info, &mut candidates, &mut dynamic);
            }
            _ => {}
        }
    }
    (candidates, dynamic)
}

/// A pointer loaded from the memory a constant buffer pointer points to (the binding reads it
/// again for every dispatch and draw). The table pointer can also be one of several constant
/// buffer pointers (a phi or select of them), or an entry of a constant buffer array indexed at
/// run time; those are table candidates, bound only in the bindings the other storage buffers
/// leave free.
fn table_candidates(
    prog: &Program,
    load: InstId,
    env: &dyn ShaderEnvironment,
    info: &mut StorageInfo,
    candidates: &mut Candidates,
    dynamic: &mut bool,
) {
    let holder_addr = track_low_address(prog, load);
    let (root, holder_offset) = match holder_addr {
        Some(addr) => {
            let Some(root) = inst_of(prog, addr.value) else {
                return;
            };
            (root, addr.imm_offset)
        }
        None => {
            let mut address = prog.resolve(prog.arg(load, 0));
            let mut offset = 0;
            if let Some(add) = address.inst()
                && prog.op(add) == Opcode::IAdd64
                && let Some(imm) = imm_u64(prog, prog.arg(add, 1))
            {
                offset = imm as i32;
                address = prog.resolve(prog.arg(add, 0));
            }
            let Some(pack) = address.inst() else {
                return;
            };
            if prog.op(pack) != Opcode::PackUint2x32 {
                return;
            }
            let Some(root) = inst_of(prog, prog.arg(pack, 0)) else {
                return;
            };
            (root, offset)
        }
    };
    if holder_offset < 0 || holder_offset % 8 != 0 {
        return;
    }
    let indirect = holder_offset as u32;
    let mut add = |pointer: BufferAddr, table: bool| {
        if candidates.contains(&pointer) {
            return;
        }
        if table && env.stage() != ShaderStage::Compute {
            return;
        }
        if table {
            info.tables_seen = true;
            if !info.tables_enabled {
                return;
            }
            if !info.table_candidates.contains(&pointer) {
                info.table_candidates.push(pointer);
            }
        }
        candidates.push(pointer);
        *dynamic = true;
    };
    let mut holders: SmallVec<[(InstId, bool); MAX_HOLDERS]> = SmallVec::new();
    holders.push((root, false));
    let mut pending_holder = 0;
    while pending_holder < holders.len() {
        let (holder, via_phi) = holders[pending_holder];
        pending_holder += 1;
        let inst = prog.inst(holder);
        let mut enqueue = |value: Value, branch: bool| {
            if let Some(next) = inst_of(prog, value)
                && holders.len() < MAX_HOLDERS
                && !holders.iter().any(|&(h, _)| h == next)
            {
                holders.push((next, via_phi || branch));
            }
        };
        match inst.op {
            Opcode::CompositeExtractU32x2 => {
                if imm_u32(prog, inst.arg(1)) == Some(0) {
                    enqueue(inst.arg(0), false);
                }
                continue;
            }
            Opcode::Phi => {
                for &arg in inst.args() {
                    enqueue(arg, true);
                }
                continue;
            }
            Opcode::SelectU32 => {
                enqueue(inst.arg(1), true);
                enqueue(inst.arg(2), true);
                continue;
            }
            Opcode::GetCbufU32 | Opcode::GetCbufU32x2 => {}
            _ => continue,
        }
        let Some(index) = imm_u32(prog, inst.arg(0)) else {
            continue;
        };
        if let Some(offset) = imm_u32(prog, inst.arg(1)) {
            if offset.is_multiple_of(8) && env.is_cbuf_storage_buffer_pointer(index, offset) {
                add(
                    BufferAddr {
                        index,
                        offset,
                        indirect,
                    },
                    via_phi || holder_addr.is_none(),
                );
            }
            continue;
        }
        let slots = collect_pointer_slots(prog, inst.arg(1));
        for pointer in slot_addresses(&slots, index, indirect) {
            if pointer.offset.is_multiple_of(8)
                && env.is_cbuf_indirect_storage_buffer_pointer(
                    pointer.index,
                    pointer.offset,
                    pointer.indirect,
                )
            {
                add(pointer, true);
            }
        }
    }
}

fn imm_u64(prog: &Program, value: Value) -> Option<u64> {
    let value = prog.resolve(value);
    if value.is_immediate() {
        value.imm_u64()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_operand_offsets_track_no_array() {
        let mut prog = Program::new();
        let block = prog.new_block();
        let lane = prog.append(block, Opcode::LaneId, &[], 0);
        assert_eq!(dynamic_pointer_slots(&prog, Value::Inst(lane)), None);
    }
}
