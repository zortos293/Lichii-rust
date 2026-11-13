use std::time::{Duration, Instant};

use rustc_hash::FxHashSet;

use super::*;
use crate::ir::{SyntaxNode, Type};
use crate::opt::compute_post_order;
use crate::{ShaderArch, ShaderStage};
use lichii_common::GpuVAddr;

#[derive(Default)]
struct Env {
    native: bool,
    graphics: bool,
    pointers: Option<FxHashSet<(u32, u32)>>,
}

impl ShaderEnvironment for Env {
    test_gpu_state!();
    fn arch(&self) -> ShaderArch {
        ShaderArch::Maxwell
    }
    fn stage(&self) -> ShaderStage {
        if self.graphics {
            ShaderStage::Fragment
        } else {
            ShaderStage::Compute
        }
    }
    fn start_address(&self) -> GpuVAddr {
        0
    }
    fn read_u64(&self, _: GpuVAddr) -> u64 {
        0
    }
    fn read_cbuf(&self, _: u32, _: u32) -> u32 {
        0
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [64, 1, 1]
    }
    fn is_native_program(&self) -> bool {
        self.native
    }
    fn is_cbuf_storage_buffer_pointer(&self, index: u32, offset: u32) -> bool {
        self.pointers
            .as_ref()
            .is_none_or(|p| p.contains(&(index, offset)))
    }
}

struct Builder {
    prog: Program,
    block: BlockId,
}

impl Builder {
    fn new() -> Builder {
        let mut prog = Program::new();
        let block = prog.new_block();
        prog.block_order = vec![block];
        prog.syntax_list = vec![SyntaxNode::Block(block), SyntaxNode::Return];
        Builder { prog, block }
    }

    fn inst(&mut self, op: Opcode, args: &[Value]) -> Value {
        Value::Inst(self.prog.append(self.block, op, args, 0))
    }

    fn cbuf(&mut self, index: u32, offset: u32) -> Value {
        self.inst(Opcode::GetCbufU32, &[Value::U32(index), Value::U32(offset)])
    }

    fn address(&mut self, low: Value, high: Value, offset: u64) -> Value {
        let vector = self.inst(Opcode::CompositeConstructU32x2, &[low, high]);
        let pointer = self.inst(Opcode::PackUint2x32, &[vector]);
        self.inst(Opcode::IAdd64, &[pointer, Value::U64(offset)])
    }

    fn pair_address(&mut self, index: u32, offset: u32, imm: u64) -> Value {
        let low = self.cbuf(index, offset);
        let high = self.cbuf(index, offset + 4);
        self.address(low, high, imm)
    }

    fn local_x(&mut self) -> Value {
        let id = self.inst(Opcode::LocalInvocationId, &[]);
        self.inst(Opcode::CompositeExtractU32x3, &[id, Value::U32(0)])
    }

    fn use_value(&mut self, value: Value) -> InstId {
        self.prog.append(self.block, Opcode::Reference, &[value], 0)
    }

    fn run(&mut self, env: &Env, profile: &Profile) -> Result<ShaderInfo, TranslateError> {
        compute_post_order(&mut self.prog);
        let mut info = ShaderInfo::default();
        global_memory_to_storage_buffer(env, &mut self.prog, &mut info, profile)?;
        Ok(info)
    }
}

fn profile() -> Profile {
    Profile {
        min_ssbo_alignment: 16,
        support_int64: true,
        ..Profile::default()
    }
}

fn descriptor(index: u32, offset: u32, is_written: bool) -> StorageBufferDescriptor {
    StorageBufferDescriptor {
        cbuf_index: index,
        cbuf_offset: offset,
        count: 1,
        is_written,
        indirect_offset: StorageBufferDescriptor::DIRECT,
        write_extent: StorageWriteExtent::default(),
    }
}

/// The storage instruction a rewritten access forwards to.
fn forwarded(prog: &Program, access: InstId) -> InstId {
    assert_eq!(prog.op(access), Opcode::Identity);
    prog.arg(access, 0)
        .inst()
        .expect("forwarded to an instruction")
}

/// Asserts `offset` is `ISub(low (+ imm), BitwiseAnd(GetCbufU32 index offset, !(align - 1)))`.
fn assert_offset(prog: &Program, offset: Value, low: Value, imm: i32, buffer: (u32, u32)) {
    let sub = offset.inst().expect("offset is computed");
    assert_eq!(prog.op(sub), Opcode::ISub32);
    let base = prog.arg(sub, 0);
    if imm == 0 {
        assert_eq!(base, low);
    } else {
        let add = base.inst().expect("immediate offset is added");
        assert_eq!(prog.op(add), Opcode::IAdd32);
        assert_eq!(prog.inst(add).args(), &[low, Value::U32(imm as u32)]);
    }
    let and = prog.arg(sub, 1).inst().expect("pointer is aligned");
    assert_eq!(prog.op(and), Opcode::BitwiseAnd32);
    assert_eq!(prog.arg(and, 1), Value::U32(!15));
    let pointer = prog.arg(and, 0).inst().expect("pointer is read");
    assert_eq!(prog.op(pointer), Opcode::GetCbufU32);
    assert_eq!(
        prog.inst(pointer).args(),
        &[Value::U32(buffer.0), Value::U32(buffer.1)]
    );
}

#[test]
fn paired_pointer_load_becomes_a_storage_load() {
    let mut b = Builder::new();
    let low = b.cbuf(0, 0x110);
    let high = b.cbuf(0, 0x114);
    let address = b.address(low, high, 8);
    let load = b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap();
    let user = b.use_value(Value::Inst(load));
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x110, false)]
    );
    let storage = forwarded(&b.prog, load);
    assert_eq!(b.prog.op(storage), Opcode::LoadStorage32);
    assert_eq!(b.prog.arg(storage, 0), Value::U32(0));
    assert_offset(&b.prog, b.prog.arg(storage, 1), low, 8, (0, 0x110));
    assert_eq!(b.prog.resolve(b.prog.arg(user, 0)), Value::Inst(storage));
}

#[test]
fn bindings_follow_constant_buffer_order() {
    let mut b = Builder::new();
    let late = b.pair_address(0, 0x200, 0);
    let early = b.pair_address(0, 0x120, 0);
    let late_load = b.inst(Opcode::LoadGlobal32, &[late]).inst().unwrap();
    let early_load = b.inst(Opcode::LoadGlobal64, &[early]).inst().unwrap();
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x120, false), descriptor(0, 0x200, false)]
    );
    let early_storage = forwarded(&b.prog, early_load);
    assert_eq!(b.prog.op(early_storage), Opcode::LoadStorage64);
    assert_eq!(b.prog.arg(early_storage, 0), Value::U32(0));
    assert_eq!(b.prog.arg(forwarded(&b.prog, late_load), 0), Value::U32(1));
}

#[test]
fn store_is_replaced_and_records_its_extent() {
    let mut b = Builder::new();
    let pointer = b.cbuf(0, 0x140);
    let high = b.cbuf(0, 0x144);
    let x = b.local_x();
    let scaled = b.inst(Opcode::ShiftLeftLogical32, &[x, Value::U32(2)]);
    let low = b.inst(Opcode::IAdd32, &[pointer, scaled]);
    let address = b.address(low, high, 16);
    let store = b
        .inst(Opcode::WriteGlobal32, &[address, Value::U32(7)])
        .inst()
        .unwrap();
    let info = b.run(&Env::default(), &profile()).unwrap();

    let mut expected = descriptor(0, 0x140, true);
    expected.write_extent = StorageWriteExtent {
        bounded: true,
        const_end: 20,
        local_stride: [4, 0, 0],
        ..StorageWriteExtent::default()
    };
    assert_eq!(info.storage_buffers_descriptors, vec![expected]);
    assert_eq!(b.prog.op(store), Opcode::Void);
    let write = b.prog.inst(store).prev().unwrap();
    assert_eq!(b.prog.op(write), Opcode::WriteStorage32);
    assert_eq!(b.prog.arg(write, 0), Value::U32(0));
    assert_eq!(b.prog.arg(write, 2), Value::U32(7));
    assert_offset(&b.prog, b.prog.arg(write, 1), low, 16, (0, 0x140));
}

#[test]
fn store_through_a_loaded_index_has_an_unknown_extent() {
    let mut b = Builder::new();
    let index_address = b.pair_address(0, 0x150, 0);
    let index = b.inst(Opcode::LoadGlobal32, &[index_address]);
    let pointer = b.cbuf(0, 0x140);
    let high = b.cbuf(0, 0x144);
    let low = b.inst(Opcode::IAdd32, &[pointer, index]);
    let address = b.address(low, high, 0);
    b.inst(Opcode::WriteGlobal32, &[address, Value::U32(1)]);
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x140, true), descriptor(0, 0x150, false)]
    );
}

#[test]
fn atomics_keep_their_operands_and_results() {
    let mut b = Builder::new();
    let address = b.pair_address(0, 0x180, 4);
    let add = b
        .inst(Opcode::GlobalAtomicIAdd32, &[address, Value::U32(1)])
        .inst()
        .unwrap();
    let swap = b
        .inst(
            Opcode::GlobalAtomicCompareExchange32,
            &[address, Value::U32(2), Value::U32(3)],
        )
        .inst()
        .unwrap();
    b.use_value(Value::Inst(add));
    b.use_value(Value::Inst(swap));
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert!(info.storage_buffers_descriptors[0].is_written);
    let add = forwarded(&b.prog, add);
    assert_eq!(b.prog.op(add), Opcode::StorageAtomicIAdd32);
    assert_eq!(b.prog.arg(add, 2), Value::U32(1));
    let swap = forwarded(&b.prog, swap);
    assert_eq!(b.prog.op(swap), Opcode::StorageAtomicCompareExchange32);
    assert_eq!(
        &b.prog.inst(swap).args()[2..],
        &[Value::U32(2), Value::U32(3)]
    );
}

#[test]
fn wide_compare_exchange_keeps_compare_before_value() {
    let mut b = Builder::new();
    let address = b.pair_address(0, 0x180, 8);
    let swap = b
        .inst(
            Opcode::GlobalAtomicCompareExchange64,
            &[address, Value::U64(2), Value::U64(3)],
        )
        .inst()
        .unwrap();
    b.use_value(Value::Inst(swap));
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert!(info.storage_buffers_descriptors[0].is_written);
    let swap = forwarded(&b.prog, swap);
    assert_eq!(b.prog.op(swap), Opcode::StorageAtomicCompareExchange64);
    assert_eq!(
        &b.prog.inst(swap).args()[2..],
        &[Value::U64(2), Value::U64(3)]
    );
}

#[test]
fn single_addend_word_is_the_base_without_a_high_pair() {
    let mut b = Builder::new();
    let pointer = b.cbuf(2, 0x18);
    let stride = b.cbuf(2, 0x24);
    let low = b.inst(Opcode::IAdd32, &[pointer, stride]);
    let high = b.inst(Opcode::UndefU32, &[]);
    let address = b.address(low, high, 0);
    b.inst(Opcode::LoadGlobal32, &[address]);
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(2, 0x18, false)]
    );
}

#[test]
fn nvn_range_wins_over_other_reads() {
    let mut b = Builder::new();
    let nvn = b.cbuf(0, 0x120);
    let other = b.cbuf(3, 0x40);
    let low = b.inst(Opcode::BitwiseOr32, &[nvn, other]);
    let high = b.inst(Opcode::UndefU32, &[]);
    let address = b.address(low, high, 0);
    b.inst(Opcode::LoadGlobal32, &[address]);
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x120, false)]
    );
}

#[test]
fn unbiased_search_is_the_last_resort() {
    let mut b = Builder::new();
    let word = b.cbuf(1, 0x20);
    let x = b.local_x();
    let low = b.inst(Opcode::BitwiseOr32, &[word, x]);
    let high = b.inst(Opcode::UndefU32, &[]);
    let address = b.address(low, high, 0);
    b.inst(Opcode::LoadGlobal32, &[address]);
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(1, 0x20, false)]
    );
}

#[test]
fn untrackable_accesses_stay_global() {
    let mut b = Builder::new();
    let immediate = b
        .inst(Opcode::LoadGlobal32, &[Value::U64(0x1000)])
        .inst()
        .unwrap();
    let x = b.local_x();
    let wide = b.inst(Opcode::ConvertU64U32, &[x]);
    let narrow = b.inst(Opcode::LoadGlobal32, &[wide]).inst().unwrap();
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert!(info.storage_buffers_descriptors.is_empty());
    assert_eq!(b.prog.op(immediate), Opcode::LoadGlobal32);
    assert_eq!(b.prog.op(narrow), Opcode::LoadGlobal32);
}

#[test]
fn program_without_global_memory_is_untouched() {
    let mut b = Builder::new();
    let word = b.cbuf(0, 0x110);
    let sum = b.inst(Opcode::IAdd32, &[word, Value::U32(4)]);
    b.use_value(sum);
    let before = b.prog.num_insts();
    let info = b.run(&Env::default(), &profile()).unwrap();

    assert_eq!(info, ShaderInfo::default());
    assert_eq!(b.prog.num_insts(), before);
}

#[test]
fn native_program_binds_every_candidate_of_a_select() {
    let mut b = Builder::new();
    let first = b.cbuf(0, 0x10);
    let second = b.cbuf(0, 0x20);
    let cond = b.inst(Opcode::UndefU1, &[]);
    let low = b.inst(Opcode::SelectU32, &[cond, first, second]);
    let high = b.inst(Opcode::UndefU32, &[]);
    let address = b.address(low, high, 0);
    let load = b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap();
    let env = Env {
        native: true,
        ..Env::default()
    };
    let info = b.run(&env, &profile()).unwrap();

    assert!(info.native_global_bounds);
    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x10, false), descriptor(0, 0x20, false)]
    );
    assert_eq!(b.prog.op(load), Opcode::LoadGlobal32);
}

#[test]
fn native_program_rebases_on_its_one_mapped_pointer() {
    let mut b = Builder::new();
    let pointer = b.cbuf(0, 0x28);
    let size = b.cbuf(0, 0x30);
    let low = b.inst(Opcode::IAdd32, &[pointer, size]);
    let high = b.cbuf(0, 0x2c);
    let address = b.address(low, high, 0);
    let load = b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap();
    let env = Env {
        native: true,
        pointers: Some([(0, 0x28)].into_iter().collect()),
        ..Env::default()
    };
    let info = b.run(&env, &profile()).unwrap();

    assert!(!info.native_global_bounds);
    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x28, false)]
    );
    assert_eq!(b.prog.op(forwarded(&b.prog, load)), Opcode::LoadStorage32);
}

#[test]
fn native_program_leaves_unmapped_pointers_global() {
    let mut b = Builder::new();
    let address = b.pair_address(0, 0x40, 0);
    let load = b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap();
    let env = Env {
        native: true,
        pointers: Some(FxHashSet::default()),
        ..Env::default()
    };
    let info = b.run(&env, &profile()).unwrap();

    assert!(info.storage_buffers_descriptors.is_empty());
    assert_eq!(b.prog.op(load), Opcode::LoadGlobal32);
}

#[test]
fn native_atomic_pointer_is_bound_while_unmapped() {
    let mut b = Builder::new();
    let address = b.pair_address(0, 0x40, 0);
    let atomic = b
        .inst(Opcode::GlobalAtomicIAdd32, &[address, Value::U32(1)])
        .inst()
        .unwrap();
    let env = Env {
        native: true,
        pointers: Some(FxHashSet::default()),
        ..Env::default()
    };
    let info = b.run(&env, &profile()).unwrap();

    assert_eq!(info.storage_buffers_descriptors.len(), 1);
    assert_eq!(
        b.prog.op(forwarded(&b.prog, atomic)),
        Opcode::StorageAtomicIAdd32
    );
}

#[test]
fn native_pointer_loaded_from_a_table_binds_both_buffers() {
    let mut b = Builder::new();
    let table = b.pair_address(0, 0x50, 8);
    let loaded = b.inst(Opcode::LoadGlobal64, &[table]);
    let low = b.inst(Opcode::CompositeExtractU32x2, &[loaded, Value::U32(0)]);
    let high = b.inst(Opcode::CompositeExtractU32x2, &[loaded, Value::U32(1)]);
    let address = b.address(low, high, 0);
    let load = b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap();
    let env = Env {
        native: true,
        ..Env::default()
    };
    let info = b.run(&env, &profile()).unwrap();

    let mut indirect = descriptor(0, 0x50, false);
    indirect.indirect_offset = 8;
    assert_eq!(
        info.storage_buffers_descriptors,
        vec![indirect, descriptor(0, 0x50, false)]
    );
    assert!(info.native_global_bounds);
    assert_eq!(b.prog.op(load), Opcode::LoadGlobal32);
}

#[test]
fn too_many_buffers_is_an_error() {
    let mut b = Builder::new();
    for i in 0..3 {
        let address = b.pair_address(0, 0x100 + i * 0x10, 0);
        b.inst(Opcode::LoadGlobal32, &[address]);
    }
    let profile = Profile {
        max_per_stage_storage_buffers: 2,
        ..profile()
    };
    let result = b.run(&Env::default(), &profile);
    assert!(matches!(
        result,
        Err(TranslateError::TooManyStorageBuffers { count: 3, limit: 2 })
    ));
}

#[test]
fn join_merges_shared_buffers_and_remaps_bindings() {
    let mut base = ShaderInfo {
        storage_buffers_descriptors: vec![descriptor(0, 0x110, false)],
        ..ShaderInfo::default()
    };
    let source = ShaderInfo {
        storage_buffers_descriptors: vec![descriptor(0, 0x120, false), descriptor(0, 0x110, true)],
        ..ShaderInfo::default()
    };
    let bindings = join_storage_info(&mut base, &source, 64).unwrap();
    assert_eq!(bindings, vec![1, 0]);
    assert_eq!(
        base.storage_buffers_descriptors,
        vec![descriptor(0, 0x110, true), descriptor(0, 0x120, false)]
    );

    let mut b = Builder::new();
    let load = b
        .inst(Opcode::LoadStorage32, &[Value::U32(0), Value::U32(0)])
        .inst()
        .unwrap();
    let other = b
        .inst(Opcode::IAdd32, &[Value::U32(0), Value::U32(1)])
        .inst()
        .unwrap();
    compute_post_order(&mut b.prog);
    remap_storage_bindings(&mut b.prog, &bindings);
    assert_eq!(b.prog.arg(load, 0), Value::U32(1));
    assert_eq!(b.prog.arg(other, 0), Value::U32(0));
}

/// About 5,000 loads through 8 shared addresses, each built on a 1,024-instruction chain of
/// additions. A search per access walks 5,000 chains; the per-start memo walks 8.
#[test]
fn thousands_of_accesses_through_few_bases_track_in_linear_time() {
    const BASES: u32 = 8;
    const CHAIN: usize = 1024;
    const LOADS: u32 = 5000;
    let mut b = Builder::new();
    let x = b.local_x();
    let mut lows = Vec::new();
    for base in 0..BASES {
        let mut low = b.cbuf(0, 0x100 + base * 0x10);
        for k in 0..CHAIN {
            let term = b.inst(Opcode::BitwiseAnd32, &[x, Value::U32(k as u32)]);
            low = b.inst(Opcode::IAdd32, &[low, term]);
        }
        let high = b.cbuf(0, 0x104 + base * 0x10);
        lows.push((low, high));
    }
    let mut loads = Vec::new();
    for i in 0..LOADS {
        let (low, high) = lows[(i % BASES) as usize];
        let address = b.address(low, high, u64::from(i) * 4);
        loads.push(b.inst(Opcode::LoadGlobal32, &[address]).inst().unwrap());
    }
    compute_post_order(&mut b.prog);
    let mut info = ShaderInfo::default();
    let started = Instant::now();
    global_memory_to_storage_buffer(&Env::default(), &mut b.prog, &mut info, &profile()).unwrap();
    let elapsed = started.elapsed();

    assert_eq!(info.storage_buffers_descriptors.len(), BASES as usize);
    assert!(
        loads
            .iter()
            .all(|&load| b.prog.op(forwarded(&b.prog, load)) == Opcode::LoadStorage32)
    );
    assert!(
        elapsed < Duration::from_millis(100),
        "global memory to storage buffer took {elapsed:?}"
    );
}

#[test]
fn loop_carried_strides_are_not_base_pointers() {
    let mut prog = Program::new();
    let entry = prog.new_block();
    let header = prog.new_block();
    let exit = prog.new_block();
    prog.block_order = vec![entry, header, exit];
    prog.syntax_list = vec![
        SyntaxNode::Block(entry),
        SyntaxNode::Loop {
            body: header,
            continue_block: header,
            merge: exit,
        },
        SyntaxNode::Block(header),
        SyntaxNode::Repeat {
            cond: Value::U1(false),
            loop_header: header,
            merge: exit,
        },
        SyntaxNode::Block(exit),
        SyntaxNode::Return,
    ];
    prog.add_branch(entry, header);
    prog.add_branch(header, header);
    prog.add_branch(header, exit);
    let cbuf = |prog: &mut Program, block, offset| {
        Value::Inst(prog.append(
            block,
            Opcode::GetCbufU32,
            &[Value::U32(0), Value::U32(offset)],
            0,
        ))
    };
    let base = cbuf(&mut prog, entry, 0x110);
    let high = cbuf(&mut prog, entry, 0x114);
    let step = cbuf(&mut prog, entry, 0x118);
    let step_high = cbuf(&mut prog, entry, 0x11c);
    let phi = prog.append(header, Opcode::Phi, &[], Type::U32.bits());
    let next = prog.append(header, Opcode::IAdd32, &[Value::Inst(phi), step], 0);
    prog.add_phi_operand(phi, entry, base);
    prog.add_phi_operand(phi, header, Value::Inst(next));
    let high_sum = prog.append(header, Opcode::IAdd32, &[high, step_high], 0);
    let vector = prog.append(
        header,
        Opcode::CompositeConstructU32x2,
        &[Value::Inst(phi), Value::Inst(high_sum)],
        0,
    );
    let pointer = prog.append(header, Opcode::PackUint2x32, &[Value::Inst(vector)], 0);
    prog.append(header, Opcode::LoadGlobal32, &[Value::Inst(pointer)], 0);
    compute_post_order(&mut prog);
    let mut info = ShaderInfo::default();
    global_memory_to_storage_buffer(&Env::default(), &mut prog, &mut info, &profile()).unwrap();

    assert_eq!(
        info.storage_buffers_descriptors,
        vec![descriptor(0, 0x110, false)]
    );
}
