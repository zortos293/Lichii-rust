//! Global memory to storage buffer: rebases global loads, stores and atomics on the storage
//! buffer whose 64-bit pointer a constant buffer holds, and lists those buffers in
//! [`ShaderInfo::storage_buffers_descriptors`].
//!
//! Each access's address is tracked back to a constant buffer word, trying in order: a 64-bit
//! pointer pair summed into both address halves, a single word added into the low half, the
//! NVN descriptor range of constant buffer 0, and any aligned constant buffer read. Switch 2
//! native programs bind every candidate pointer an address can come from and leave the access
//! global when there are several, so the backend picks the buffer at run time.
//!
//! Every search is memoized per start instruction over IR that does not change while the
//! buffers are collected (NX2_NOTES row 121: per-access searches made one shader take 3.4 s).

mod conditions;
mod extent;
mod heuristics;
mod native;
#[cfg(test)]
mod tests;
mod track;

use std::collections::{BTreeMap, BTreeSet};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use self::conditions::Conditions;
use self::extent::StoreExtentAnalysis;
use self::native::Candidates;
use self::track::{NVN_BIAS, Tracker, track_low_address};
use crate::info::{StorageBufferDescriptor, StorageWriteExtent};
use crate::ir::{BlockId, Emitter, InstId, Opcode, Program, Value};
use crate::{Profile, ShaderEnvironment, ShaderInfo, TranslateError};

/// Address in a constant buffer of a storage buffer pointer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct BufferAddr {
    index: u32,
    offset: u32,
    /// Offset of the buffer pointer in the memory the constant buffer pointer points to.
    indirect: u32,
}

impl BufferAddr {
    const DIRECT: u32 = StorageBufferDescriptor::DIRECT;

    fn new(index: u32, offset: u32) -> BufferAddr {
        BufferAddr {
            index,
            offset,
            indirect: Self::DIRECT,
        }
    }

    fn high_word(self) -> BufferAddr {
        BufferAddr::new(self.index, self.offset.wrapping_add(4))
    }

    fn is_indirect(self) -> bool {
        self.indirect != Self::DIRECT
    }
}

fn inst_of(prog: &Program, value: Value) -> Option<InstId> {
    prog.resolve(value).inst()
}

fn imm_u32(prog: &Program, value: Value) -> Option<u32> {
    let value = prog.resolve(value);
    if value.is_immediate() {
        value.imm_u32()
    } else {
        None
    }
}

/// Collection state, rebuilt from scratch when the pass retries with other binding choices.
#[derive(Default)]
struct StorageInfo {
    set: BTreeSet<BufferAddr>,
    to_replace: Vec<(BufferAddr, InstId)>,
    writes: FxHashSet<BufferAddr>,
    native_candidates: FxHashMap<InstId, Candidates>,
    /// Native accesses whose candidates include slots of a dynamically indexed pointer array.
    native_dynamic: FxHashSet<InstId>,
    /// A native access with several candidate pointers was left global.
    native_runtime_bounds: bool,
    /// Paired accesses bind the loaded pointers that compete with their pair.
    bind_competing_loaded: bool,
    competing_loaded_bound: bool,
    /// Pointer table entries are bound only in the bindings the other buffers leave free.
    tables_enabled: bool,
    tables_seen: bool,
    table_budget: usize,
    table_used: usize,
    /// The buffers bound without table entries; bound again, they take no table budget.
    counted: BTreeSet<BufferAddr>,
    table_candidates: SmallVec<[BufferAddr; 8]>,
    /// Buffers a store or atomic that stayed global may write to, anywhere.
    unbounded_writes: FxHashSet<BufferAddr>,
    /// A store or atomic stayed global without a buffer.
    untracked_writes: bool,
}

struct Collector<'a> {
    prog: &'a Program,
    env: &'a dyn ShaderEnvironment,
    conditions: Conditions,
    tracker: Tracker,
    iterator_words: FxHashSet<BufferAddr>,
    atomic_pointers: FxHashSet<BufferAddr>,
}

impl Collector<'_> {
    fn collect(&mut self, info: &mut StorageInfo) {
        let prog = self.prog;
        for &block in &prog.post_order {
            for inst in prog.insts_of(block) {
                let op = prog.op(inst);
                if !is_global_memory(op) {
                    continue;
                }
                let bound = self.collect_access(block, inst, info);
                info.untracked_writes |= !bound && is_global_memory_write(op);
            }
        }
    }

    /// Finds the storage buffer of one global access. Returns false when the access stays
    /// global without a buffer.
    fn collect_access(&mut self, block: BlockId, inst: InstId, info: &mut StorageInfo) -> bool {
        let (prog, env) = (self.prog, self.env);
        let Some(addr) = track_low_address(prog, inst) else {
            return false;
        };
        let op = prog.op(inst);
        let is_write = is_global_memory_write(op);
        let paired =
            self.tracker
                .pointer_addend(prog, &addr, &self.iterator_words, &self.conditions, block);
        let native = env.is_native_program();
        let mut storage_buffer = paired
            .strict
            .filter(|b| !native || env.is_cbuf_storage_buffer_pointer(b.index, b.offset));
        let paired_found = storage_buffer.is_some();
        if native && let Some(start) = inst_of(prog, addr.value) {
            if !info.native_candidates.contains_key(&start) {
                let (candidates, dynamic) = native::native_pointer_candidates(
                    prog,
                    start,
                    env,
                    info,
                    &mut self.tracker.marks,
                );
                if dynamic {
                    info.native_dynamic.insert(start);
                }
                info.native_candidates.insert(start, candidates);
            }
            let candidates = info.native_candidates[&start].clone();
            let has_loaded_pointer = candidates.iter().any(|c| c.is_indirect());
            let competes_with_pair = !paired_found
                || (has_loaded_pointer && info.bind_competing_loaded && is_runtime_routable(op));
            let several = paired_found
                || candidates.len() > 1
                || (!candidates.is_empty() && info.native_dynamic.contains(&start));
            if competes_with_pair && several {
                if let Some(pair) = storage_buffer {
                    info.competing_loaded_bound = true;
                    info.set.insert(pair);
                    if is_write {
                        info.writes.insert(pair);
                    }
                }
                if bind_candidates(info, &candidates, is_write) {
                    info.native_runtime_bounds = true;
                    return true;
                }
            } else if !paired_found && candidates.len() == 1 {
                storage_buffer = Some(candidates[0]);
            }
        }
        if native && storage_buffer.is_none() {
            match paired.strict {
                Some(pointer) if self.atomic_pointers.contains(&pointer) => {
                    storage_buffer = Some(pointer)
                }
                _ => return false,
            }
        }
        let tracker = &mut self.tracker;
        let found = storage_buffer
            .or(paired.nearest)
            .or_else(|| tracker.track_addend(prog, addr.value))
            .or_else(|| tracker.track(prog, addr.value, Some(&NVN_BIAS)))
            .or_else(|| tracker.track(prog, addr.value, None));
        let Some(storage_buffer) = found else {
            tracing::warn!("storage buffer failed to track, using global memory fallbacks");
            return false;
        };
        if is_write {
            info.writes.insert(storage_buffer);
        }
        info.set.insert(storage_buffer);
        info.to_replace.push((storage_buffer, inst));
        true
    }
}

/// Binds every candidate of a native access that stays global. Table entries take from the
/// table budget and are skipped once it runs out. Returns whether any candidate was bound.
fn bind_candidates(info: &mut StorageInfo, candidates: &Candidates, is_write: bool) -> bool {
    let mut accepted = false;
    for &candidate in candidates {
        let holder = BufferAddr::new(candidate.index, candidate.offset);
        if info.table_candidates.contains(&candidate) {
            let takes_binding =
                |buffer: &BufferAddr| !info.set.contains(buffer) && !info.counted.contains(buffer);
            let added = usize::from(takes_binding(&candidate))
                + usize::from(candidate.is_indirect() && takes_binding(&holder));
            if info.table_used + added > info.table_budget {
                continue;
            }
            info.table_used += added;
        }
        info.set.insert(candidate);
        if is_write {
            info.writes.insert(candidate);
            info.unbounded_writes.insert(candidate);
        }
        if candidate.is_indirect() {
            info.set.insert(holder);
        }
        accepted = true;
    }
    accepted
}

/// Rebases the program's global memory accesses on storage buffers and appends the buffers to
/// `info.storage_buffers_descriptors`. Fails when the program needs more storage buffers than
/// a stage can bind.
pub fn global_memory_to_storage_buffer(
    env: &dyn ShaderEnvironment,
    prog: &mut Program,
    info: &mut ShaderInfo,
    profile: &Profile,
) -> Result<(), TranslateError> {
    let has_global_memory = prog
        .post_order
        .iter()
        .any(|&block| prog.insts_of(block).any(|id| is_global_memory(prog.op(id))));
    if !has_global_memory {
        return Ok(());
    }
    split_address_sums(prog);
    let max_ssbos = storage_buffer_limit(profile);
    let (storage, extents) = {
        let shared: &Program = prog;
        let mut tracker = Tracker::new(shared);
        let iterator_words = heuristics::collect_iterator_words(shared, &mut tracker);
        let atomic_pointers = heuristics::collect_atomic_pointers(shared, &mut tracker);
        let mut collector = Collector {
            prog: shared,
            env,
            conditions: Conditions::new(shared),
            tracker,
            iterator_words,
            atomic_pointers,
        };
        let mut storage = StorageInfo {
            bind_competing_loaded: profile.support_int64,
            ..StorageInfo::default()
        };
        collector.collect(&mut storage);
        if storage.competing_loaded_bound && storage.set.len() > max_ssbos {
            storage = StorageInfo::default();
            collector.collect(&mut storage);
        }
        if storage.tables_seen && storage.set.len() < max_ssbos {
            let mut with_tables = StorageInfo {
                bind_competing_loaded: storage.bind_competing_loaded,
                tables_enabled: true,
                table_budget: max_ssbos - storage.set.len(),
                counted: storage.set.clone(),
                ..StorageInfo::default()
            };
            collector.collect(&mut with_tables);
            if with_tables.set.len() <= max_ssbos {
                storage = with_tables;
            }
        }
        if storage.set.len() > max_ssbos {
            return Err(TranslateError::TooManyStorageBuffers {
                count: storage.set.len(),
                limit: max_ssbos,
            });
        }
        let extents = store_extents(shared, &storage, env.workgroup_size());
        (storage, extents)
    };
    info.native_global_bounds |= storage.native_runtime_bounds;
    let set: Vec<BufferAddr> = storage.set.into_iter().collect();
    for buffer in &set {
        let is_written = storage.writes.contains(buffer);
        let write_extent = match extents.get(buffer) {
            Some(extent) if is_written => *extent,
            _ => StorageWriteExtent::default(),
        };
        info.storage_buffers_descriptors
            .push(StorageBufferDescriptor {
                cbuf_index: buffer.index,
                cbuf_offset: buffer.offset,
                count: 1,
                is_written,
                indirect_offset: buffer.indirect,
                write_extent,
            });
    }
    let alignment = profile.min_ssbo_alignment.clamp(1, 1 << 31) as u32;
    for (buffer, inst) in storage.to_replace {
        let Ok(binding) = set.binary_search(&buffer) else {
            continue;
        };
        let offset = storage_offset(prog, inst, buffer, alignment);
        replace(prog, inst, Value::U32(binding as u32), offset);
    }
    Ok(())
}

/// Rewrites each global address that sums two non-immediate 64-bit terms, such as a 32-bit
/// register plus a 64-bit uniform pointer (`[Ra + UR.64]`), as the pair of 32-bit sums with
/// carry that hosts without `shaderInt64` lower it to, so its halves track to the pointer the
/// same way on every host. An immediate offset added last stays outside the pair.
fn split_address_sums(prog: &mut Program) {
    let accesses: Vec<InstId> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| is_global_memory(prog.op(id)))
        .collect();
    for inst in accesses {
        let mut address = prog.arg(inst, 0);
        if let Some(add) = inst_of(prog, address)
            && prog.op(add) == Opcode::IAdd64
            && prog.resolve(prog.arg(add, 1)).is_immediate()
        {
            address = prog.arg(add, 0);
        }
        address_words(prog, address);
    }
}

/// The low and high words of a 64-bit address term: a packed pair, a zero-extended word, or a
/// sum of two such terms, which is rewritten into a packed pair.
fn address_words(prog: &mut Program, value: Value) -> Option<(Value, Value)> {
    let inst = inst_of(prog, value)?;
    match prog.op(inst) {
        Opcode::PackUint2x32 => {
            let pair = inst_of(prog, prog.arg(inst, 0))?;
            (prog.op(pair) == Opcode::CompositeConstructU32x2).then(|| {
                (
                    prog.resolve(prog.arg(pair, 0)),
                    prog.resolve(prog.arg(pair, 1)),
                )
            })
        }
        Opcode::ConvertU64U32 => Some((prog.resolve(prog.arg(inst, 0)), Value::U32(0))),
        Opcode::IAdd64 => {
            let (a, b) = (prog.arg(inst, 0), prog.arg(inst, 1));
            if prog.resolve(a).is_immediate() || prog.resolve(b).is_immediate() {
                return None;
            }
            let (a_lo, a_hi) = address_words(prog, a)?;
            let (b_lo, b_hi) = address_words(prog, b)?;
            let mut ir = Emitter::before(prog, inst);
            let lo = ir.iadd(a_lo, b_lo);
            let carry = ir.get_carry_from_op(lo);
            let carry = ir.select(carry, Value::U32(1), Value::U32(0));
            let hi = match (a_hi, b_hi) {
                (hi, Value::U32(0)) | (Value::U32(0), hi) => hi,
                _ => ir.iadd(a_hi, b_hi),
            };
            let hi = ir.iadd(hi, carry);
            let pair = ir.composite_construct(&[lo, hi]);
            let packed = ir.pack_uint_2x32(pair);
            prog.replace_uses_with(inst, packed);
            Some((lo, hi))
        }
        _ => None,
    }
}

/// Store extents per written buffer, read from the guest addresses before they are rebased.
/// None while a store stays global: the backend routes it to the closest bound pointer at or
/// below the address, which need not be one of its candidates.
fn store_extents(
    prog: &Program,
    storage: &StorageInfo,
    workgroup: [u32; 3],
) -> BTreeMap<BufferAddr, StorageWriteExtent> {
    let mut extents = BTreeMap::new();
    if storage.untracked_writes || !storage.unbounded_writes.is_empty() {
        return extents;
    }
    let mut analyses: BTreeMap<BufferAddr, StoreExtentAnalysis> = BTreeMap::new();
    for &(buffer, inst) in &storage.to_replace {
        if !is_global_memory_write(prog.op(inst)) {
            continue;
        }
        let analysis = analyses
            .entry(buffer)
            .or_insert_with(|| StoreExtentAnalysis::new(buffer, workgroup));
        let extent = extents.entry(buffer).or_insert(StorageWriteExtent {
            bounded: true,
            ..StorageWriteExtent::default()
        });
        analysis.add_store(prog, inst, extent);
    }
    extents
}

/// The byte offset into the storage buffer: the guest address minus the buffer pointer, with
/// the pointer aligned down to what the host can bind.
fn storage_offset(prog: &mut Program, inst: InstId, buffer: BufferAddr, alignment: u32) -> Value {
    let low = track_low_address(prog, inst);
    let address = prog.arg(inst, 0);
    let mut ir = Emitter::before(prog, inst);
    let offset = match low {
        Some(low) if low.imm_offset != 0 => {
            let imm = ir.imm_s32(low.imm_offset);
            ir.iadd(low.value, imm)
        }
        Some(low) => low.value,
        None => ir.uconvert(32, address),
    };
    let pointer = ir.get_cbuf(Value::U32(buffer.index), Value::U32(buffer.offset));
    let pointer = ir.bitwise_and(pointer, Value::U32(!(alignment - 1)));
    ir.isub(offset, pointer)
}

fn replace(prog: &mut Program, inst: InstId, binding: Value, offset: Value) {
    let op = prog.op(inst);
    let storage_op = global_to_storage(op);
    if !is_global_memory_write(op) {
        let flags = prog.inst(inst).flags;
        let load = prog.insert_before(inst, storage_op, &[binding, offset], flags);
        prog.replace_uses_with(inst, Value::Inst(load));
        return;
    }
    let data = prog.arg(inst, 1);
    if is_global_store(op) {
        prog.insert_before(inst, storage_op, &[binding, offset, data], 0);
        prog.invalidate(inst);
        return;
    }
    let atomic = if matches!(
        op,
        Opcode::GlobalAtomicCompareExchange32 | Opcode::GlobalAtomicCompareExchange64
    ) {
        let comparator = prog.arg(inst, 2);
        prog.insert_before(inst, storage_op, &[binding, offset, data, comparator], 0)
    } else {
        prog.insert_before(inst, storage_op, &[binding, offset, data], 0)
    };
    prog.replace_uses_with(inst, Value::Inst(atomic));
}

/// The most storage buffers one stage can bind on this host.
pub fn storage_buffer_limit(profile: &Profile) -> usize {
    match profile.max_per_stage_storage_buffers {
        0 => ShaderInfo::MAX_SSBOS,
        limit => ShaderInfo::MAX_SSBOS.min(limit as usize),
    }
}

/// Merges `source`'s storage buffers into `base` (two programs that share one stage) and
/// returns, for each of `source`'s descriptors, its binding in `base`. Pass the bindings to
/// [`remap_storage_bindings`] for `source`'s program.
pub fn join_storage_info(
    base: &mut ShaderInfo,
    source: &ShaderInfo,
    limit: usize,
) -> Result<Vec<u32>, TranslateError> {
    base.native_global_bounds |= source.native_global_bounds;
    base.requires_coherent_global_memory |= source.requires_coherent_global_memory;
    let descriptors = &mut base.storage_buffers_descriptors;
    let mut bindings = Vec::with_capacity(source.storage_buffers_descriptors.len());
    for desc in &source.storage_buffers_descriptors {
        let existing = descriptors.iter().position(|e| {
            e.cbuf_index == desc.cbuf_index
                && e.cbuf_offset == desc.cbuf_offset
                && e.count == desc.count
                && e.indirect_offset == desc.indirect_offset
        });
        if let Some(index) = existing {
            let existing = &mut descriptors[index];
            if desc.is_written {
                if existing.is_written {
                    existing.write_extent.merge(&desc.write_extent);
                } else {
                    existing.write_extent = desc.write_extent;
                }
            }
            existing.is_written |= desc.is_written;
            bindings.push(index as u32);
            continue;
        }
        if descriptors.len() >= limit {
            return Err(TranslateError::TooManyStorageBuffers {
                count: descriptors.len() + 1,
                limit,
            });
        }
        bindings.push(descriptors.len() as u32);
        descriptors.push(*desc);
    }
    Ok(bindings)
}

/// Rewrites the binding (first argument) of every storage buffer access through `bindings`.
pub fn remap_storage_bindings(prog: &mut Program, bindings: &[u32]) {
    for i in 0..prog.post_order.len() {
        let block = prog.post_order[i];
        let mut cur = prog.block(block).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            if !is_storage_access(prog.op(id)) {
                continue;
            }
            if let Some(&binding) =
                imm_u32(prog, prog.arg(id, 0)).and_then(|b| bindings.get(b as usize))
            {
                prog.set_arg(id, 0, Value::U32(binding));
            }
        }
    }
}

fn is_global_store(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        WriteGlobalS8
            | WriteGlobalU8
            | WriteGlobalS16
            | WriteGlobalU16
            | WriteGlobal32
            | WriteGlobal64
            | WriteGlobal128
    )
}

fn is_global_load(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        LoadGlobalS8
            | LoadGlobalU8
            | LoadGlobalS16
            | LoadGlobalU16
            | LoadGlobal32
            | LoadGlobal64
            | LoadGlobal128
    )
}

fn is_global_memory(op: Opcode) -> bool {
    is_global_load(op) || is_global_memory_write(op)
}

/// Stores and atomics.
fn is_global_memory_write(op: Opcode) -> bool {
    is_global_store(op) || is_global_atomic(op)
}

fn is_global_atomic(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        GlobalAtomicIAdd32
            | GlobalAtomicSMin32
            | GlobalAtomicUMin32
            | GlobalAtomicSMax32
            | GlobalAtomicUMax32
            | GlobalAtomicInc32
            | GlobalAtomicDec32
            | GlobalAtomicAnd32
            | GlobalAtomicOr32
            | GlobalAtomicXor32
            | GlobalAtomicExchange32
            | GlobalAtomicCompareExchange32
            | GlobalAtomicIAdd64
            | GlobalAtomicSMin64
            | GlobalAtomicUMin64
            | GlobalAtomicSMax64
            | GlobalAtomicUMax64
            | GlobalAtomicAnd64
            | GlobalAtomicOr64
            | GlobalAtomicXor64
            | GlobalAtomicExchange64
            | GlobalAtomicCompareExchange64
            | GlobalAtomicIAdd32x2
            | GlobalAtomicSMin32x2
            | GlobalAtomicUMin32x2
            | GlobalAtomicSMax32x2
            | GlobalAtomicUMax32x2
            | GlobalAtomicAnd32x2
            | GlobalAtomicOr32x2
            | GlobalAtomicXor32x2
            | GlobalAtomicExchange32x2
            | GlobalAtomicAddF32
            | GlobalAtomicAddF16x2
            | GlobalAtomicAddF32x2
            | GlobalAtomicMinF16x2
            | GlobalAtomicMinF32x2
            | GlobalAtomicMaxF16x2
            | GlobalAtomicMaxF32x2
    )
}

/// Whether the backend can pick a global access's buffer among the bound ones at run time.
fn is_runtime_routable(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        LoadGlobal32
            | LoadGlobal64
            | LoadGlobal128
            | WriteGlobal32
            | WriteGlobal64
            | WriteGlobal128
            | GlobalAtomicIAdd32
            | GlobalAtomicUMax32
            | GlobalAtomicCompareExchange32
    )
}

fn is_storage_access(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        LoadStorageU8
            | LoadStorageS8
            | LoadStorageU16
            | LoadStorageS16
            | LoadStorage32
            | LoadStorage64
            | LoadStorage128
            | WriteStorageU8
            | WriteStorageS8
            | WriteStorageU16
            | WriteStorageS16
            | WriteStorage32
            | WriteStorage64
            | WriteStorage128
            | StorageAtomicIAdd32
            | StorageAtomicSMin32
            | StorageAtomicUMin32
            | StorageAtomicSMax32
            | StorageAtomicUMax32
            | StorageAtomicInc32
            | StorageAtomicDec32
            | StorageAtomicAnd32
            | StorageAtomicOr32
            | StorageAtomicXor32
            | StorageAtomicExchange32
            | StorageAtomicCompareExchange32
            | StorageAtomicIAdd64
            | StorageAtomicSMin64
            | StorageAtomicUMin64
            | StorageAtomicSMax64
            | StorageAtomicUMax64
            | StorageAtomicAnd64
            | StorageAtomicOr64
            | StorageAtomicXor64
            | StorageAtomicExchange64
            | StorageAtomicCompareExchange64
            | StorageAtomicIAdd32x2
            | StorageAtomicSMin32x2
            | StorageAtomicUMin32x2
            | StorageAtomicSMax32x2
            | StorageAtomicUMax32x2
            | StorageAtomicAnd32x2
            | StorageAtomicOr32x2
            | StorageAtomicXor32x2
            | StorageAtomicExchange32x2
            | StorageAtomicAddF32
            | StorageAtomicAddF16x2
            | StorageAtomicAddF32x2
            | StorageAtomicMinF16x2
            | StorageAtomicMinF32x2
            | StorageAtomicMaxF16x2
            | StorageAtomicMaxF32x2
    )
}

fn global_to_storage(op: Opcode) -> Opcode {
    use Opcode::*;
    match op {
        LoadGlobalS8 => LoadStorageS8,
        LoadGlobalU8 => LoadStorageU8,
        LoadGlobalS16 => LoadStorageS16,
        LoadGlobalU16 => LoadStorageU16,
        LoadGlobal32 => LoadStorage32,
        LoadGlobal64 => LoadStorage64,
        LoadGlobal128 => LoadStorage128,
        WriteGlobalS8 => WriteStorageS8,
        WriteGlobalU8 => WriteStorageU8,
        WriteGlobalS16 => WriteStorageS16,
        WriteGlobalU16 => WriteStorageU16,
        WriteGlobal32 => WriteStorage32,
        WriteGlobal64 => WriteStorage64,
        WriteGlobal128 => WriteStorage128,
        GlobalAtomicIAdd32 => StorageAtomicIAdd32,
        GlobalAtomicSMin32 => StorageAtomicSMin32,
        GlobalAtomicUMin32 => StorageAtomicUMin32,
        GlobalAtomicSMax32 => StorageAtomicSMax32,
        GlobalAtomicUMax32 => StorageAtomicUMax32,
        GlobalAtomicInc32 => StorageAtomicInc32,
        GlobalAtomicDec32 => StorageAtomicDec32,
        GlobalAtomicAnd32 => StorageAtomicAnd32,
        GlobalAtomicOr32 => StorageAtomicOr32,
        GlobalAtomicXor32 => StorageAtomicXor32,
        GlobalAtomicExchange32 => StorageAtomicExchange32,
        GlobalAtomicCompareExchange32 => StorageAtomicCompareExchange32,
        GlobalAtomicIAdd64 => StorageAtomicIAdd64,
        GlobalAtomicSMin64 => StorageAtomicSMin64,
        GlobalAtomicUMin64 => StorageAtomicUMin64,
        GlobalAtomicSMax64 => StorageAtomicSMax64,
        GlobalAtomicUMax64 => StorageAtomicUMax64,
        GlobalAtomicAnd64 => StorageAtomicAnd64,
        GlobalAtomicOr64 => StorageAtomicOr64,
        GlobalAtomicXor64 => StorageAtomicXor64,
        GlobalAtomicExchange64 => StorageAtomicExchange64,
        GlobalAtomicCompareExchange64 => StorageAtomicCompareExchange64,
        GlobalAtomicIAdd32x2 => StorageAtomicIAdd32x2,
        GlobalAtomicSMin32x2 => StorageAtomicSMin32x2,
        GlobalAtomicUMin32x2 => StorageAtomicUMin32x2,
        GlobalAtomicSMax32x2 => StorageAtomicSMax32x2,
        GlobalAtomicUMax32x2 => StorageAtomicUMax32x2,
        GlobalAtomicAnd32x2 => StorageAtomicAnd32x2,
        GlobalAtomicOr32x2 => StorageAtomicOr32x2,
        GlobalAtomicXor32x2 => StorageAtomicXor32x2,
        GlobalAtomicExchange32x2 => StorageAtomicExchange32x2,
        GlobalAtomicAddF32 => StorageAtomicAddF32,
        GlobalAtomicAddF16x2 => StorageAtomicAddF16x2,
        GlobalAtomicMinF16x2 => StorageAtomicMinF16x2,
        GlobalAtomicMaxF16x2 => StorageAtomicMaxF16x2,
        GlobalAtomicAddF32x2 => StorageAtomicAddF32x2,
        GlobalAtomicMinF32x2 => StorageAtomicMinF32x2,
        GlobalAtomicMaxF32x2 => StorageAtomicMaxF32x2,
        op => unreachable!("{op:?} is not a global memory instruction"),
    }
}

/// Bytes a global store or atomic writes.
fn store_size(op: Opcode) -> u32 {
    use Opcode::*;
    match op {
        WriteGlobalS8 | WriteGlobalU8 => 1,
        WriteGlobalS16 | WriteGlobalU16 => 2,
        WriteGlobal64
        | GlobalAtomicIAdd64
        | GlobalAtomicSMin64
        | GlobalAtomicUMin64
        | GlobalAtomicSMax64
        | GlobalAtomicUMax64
        | GlobalAtomicAnd64
        | GlobalAtomicOr64
        | GlobalAtomicXor64
        | GlobalAtomicExchange64
        | GlobalAtomicCompareExchange64
        | GlobalAtomicIAdd32x2
        | GlobalAtomicSMin32x2
        | GlobalAtomicUMin32x2
        | GlobalAtomicSMax32x2
        | GlobalAtomicUMax32x2
        | GlobalAtomicAnd32x2
        | GlobalAtomicOr32x2
        | GlobalAtomicXor32x2
        | GlobalAtomicExchange32x2
        | GlobalAtomicAddF32x2
        | GlobalAtomicMinF32x2
        | GlobalAtomicMaxF32x2 => 8,
        WriteGlobal128 => 16,
        _ => 4,
    }
}
