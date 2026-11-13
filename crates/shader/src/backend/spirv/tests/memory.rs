//! Memory, shared memory, atomic and warp family tests.

use super::*;

/// Instructions of `op` in the entry point, leaving out the helper functions.
fn count_entry_ops(module: &dr::Module, op: Op) -> usize {
    let Some(Operand::IdRef(main)) = module.entry_points[0].operands.get(1) else {
        panic!("entry point without a function");
    };
    module
        .functions
        .iter()
        .filter(|f| f.def.as_ref().and_then(|d| d.result_id) == Some(*main))
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| i.class.opcode == op)
        .count()
}

fn storage_descriptor(cbuf_offset: u32) -> StorageBufferDescriptor {
    StorageBufferDescriptor {
        cbuf_index: 0,
        cbuf_offset,
        count: 1,
        is_written: true,
        indirect_offset: StorageBufferDescriptor::DIRECT,
        write_extent: Default::default(),
    }
}

/// A compute block whose first value is `LocalInvocationId.x`, a dynamic offset or operand.
fn compute_block(s: &mut Synthetic) -> (BlockId, Value) {
    let b = s.block();
    let local = s.inst(b, Opcode::LocalInvocationId, &[]);
    let x = s.inst(b, Opcode::CompositeExtractU32x3, &[local, Value::U32(0)]);
    (b, x)
}

fn finish_compute(s: Synthetic, b: BlockId, info: ShaderInfo) -> TranslatedProgram {
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo {
            uses_local_invocation_id: true,
            ..info
        },
    );
    program.workgroup_size = [64, 1, 1];
    program
}

fn storage_memory_program() -> TranslatedProgram {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let binding = Value::U32(0);
    let pairs = [
        (Opcode::LoadStorageU8, Opcode::WriteStorageU8),
        (Opcode::LoadStorageS8, Opcode::WriteStorageS8),
        (Opcode::LoadStorageU16, Opcode::WriteStorageU16),
        (Opcode::LoadStorageS16, Opcode::WriteStorageS16),
        (Opcode::LoadStorage32, Opcode::WriteStorage32),
        (Opcode::LoadStorage64, Opcode::WriteStorage64),
        (Opcode::LoadStorage128, Opcode::WriteStorage128),
    ];
    for offset in [x, Value::U32(6), Value::U32(32)] {
        for (load, write) in pairs {
            let value = s.inst(b, load, &[binding, offset]);
            s.inst(b, write, &[binding, offset, value]);
        }
    }
    let info = ShaderInfo {
        uses_int8: true,
        uses_int16: true,
        used_storage_buffer_types: Type::U8 | Type::U16 | Type::U32 | Type::U32X2 | Type::U32X4,
        storage_buffers_descriptors: vec![storage_descriptor(0x110)],
        ..ShaderInfo::default()
    };
    finish_compute(s, b, info)
}

#[test]
fn storage_loads_and_stores_follow_the_small_integer_and_alignment_support() {
    let program = storage_memory_program();
    let words = emit(
        "storage_words",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&words, Op::BitFieldUExtract), 6);
    assert_eq!(count_ops(&words, Op::BitFieldSExtract), 6);
    assert_eq!(
        count_ops(&words, Op::FunctionCall),
        12,
        "narrow writes go through the CAS loop"
    );

    let small = Profile {
        support_int8: true,
        support_int16: true,
        support_storage_buffer_8bit: true,
        support_storage_buffer_16bit: true,
        min_ssbo_alignment: 4,
        ..Profile::default()
    };
    let direct = emit("storage_small", &small, &RuntimeInfo::default(), &program);
    assert_eq!(count_ops(&direct, Op::BitFieldUExtract), 0);
    assert_eq!(count_ops(&direct, Op::FunctionCall), 0);
    assert_eq!(count_ops(&direct, Op::UConvert), 6);
    assert_eq!(count_ops(&direct, Op::SConvert), 6 + 12);
    assert!(direct.capabilities.iter().any(|c| c.operands
        == [Operand::Capability(
            rspirv::spirv::Capability::StorageBuffer8BitAccess
        )]));

    let no_aliasing = Profile {
        support_descriptor_aliasing: false,
        support_int8: true,
        support_storage_buffer_8bit: true,
        ..Profile::default()
    };
    let module = emit(
        "storage_no_aliasing",
        &no_aliasing,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::BitFieldUExtract), 6);
}

fn global_memory_info(nvn_base: u32) -> ShaderInfo {
    ShaderInfo {
        uses_global_memory: true,
        stores_global_memory: true,
        uses_int64: true,
        uses_global_increment: true,
        uses_global_decrement: true,
        uses_atomic_f32_add: true,
        nvn_buffer_base: nvn_base,
        nvn_buffer_used: 0b1,
        used_constant_buffer_types: Type::U32 | Type::U32X2,
        used_storage_buffer_types: Type::U32 | Type::U32X2 | Type::U32X4,
        constant_buffer_descriptors: vec![ConstantBufferDescriptor { index: 0, count: 1 }],
        storage_buffers_descriptors: vec![storage_descriptor(nvn_base)],
        ..ShaderInfo::default()
    }
}

#[test]
fn global_loads_stores_and_atomics_call_the_routing_functions() {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let address = Value::U64(0x8000_1000);
    for (load, write) in [
        (Opcode::LoadGlobal32, Opcode::WriteGlobal32),
        (Opcode::LoadGlobal64, Opcode::WriteGlobal64),
        (Opcode::LoadGlobal128, Opcode::WriteGlobal128),
    ] {
        let value = s.inst(b, load, &[address]);
        s.inst(b, write, &[address, value]);
    }
    let memory_only = s.ir.clone();
    for op in [
        Opcode::GlobalAtomicIAdd32,
        Opcode::GlobalAtomicSMin32,
        Opcode::GlobalAtomicUMin32,
        Opcode::GlobalAtomicSMax32,
        Opcode::GlobalAtomicUMax32,
        Opcode::GlobalAtomicInc32,
        Opcode::GlobalAtomicDec32,
        Opcode::GlobalAtomicAnd32,
        Opcode::GlobalAtomicOr32,
        Opcode::GlobalAtomicXor32,
        Opcode::GlobalAtomicExchange32,
    ] {
        s.inst(b, op, &[address, x]);
    }
    s.inst(b, Opcode::GlobalAtomicAddF32, &[address, Value::F32(1.0)]);
    s.inst(
        b,
        Opcode::GlobalAtomicCompareExchange32,
        &[address, Value::U32(0), x],
    );
    let info = global_memory_info(0x110);
    let program = finish_compute(s, b, info.clone());
    let profile = Profile {
        support_int64: true,
        min_ssbo_alignment: 4,
        ..Profile::default()
    };
    let module = emit("global_memory", &profile, &RuntimeInfo::default(), &program);
    assert_eq!(count_entry_ops(&module, Op::FunctionCall), 6 + 13);

    let no_int64 = Profile::default();
    let memory_only = finish_compute(Synthetic { ir: memory_only }, b, info);
    let module = emit(
        "global_memory_no_int64",
        &no_int64,
        &RuntimeInfo::default(),
        &memory_only,
    );
    assert_eq!(count_entry_ops(&module, Op::FunctionCall), 0);
    let mut bindings = Bindings::default();
    let error = emit_spirv(&no_int64, &RuntimeInfo::default(), &program, &mut bindings)
        .expect_err("global atomics need 64-bit addresses");
    assert!(matches!(error, EmitError::Unsupported(_)), "{error}");

    for (op, args) in [
        (Opcode::LoadGlobalU8, &[address][..]),
        (Opcode::WriteGlobalS16, &[address, Value::U32(0)]),
        (Opcode::GlobalAtomicIAdd64, &[address, Value::U64(1)]),
        (
            Opcode::GlobalAtomicCompareExchange64,
            &[address, Value::U64(1), Value::U64(2)],
        ),
    ] {
        let error = emit_single(op, args);
        assert!(
            matches!(error, EmitError::Unsupported(_)),
            "{op:?}: {error}"
        );
    }
}

fn storage_atomic_program() -> TranslatedProgram {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let binding = Value::U32(0);
    let f32x2 = s.inst(
        b,
        Opcode::CompositeConstructF32x2,
        &[Value::F32(1.0), Value::F32(2.0)],
    );
    let u32x2 = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[Value::U32(1), Value::U32(2)],
    );
    for offset in [x, Value::U32(8)] {
        for op in [
            Opcode::StorageAtomicIAdd32,
            Opcode::StorageAtomicSMin32,
            Opcode::StorageAtomicUMin32,
            Opcode::StorageAtomicSMax32,
            Opcode::StorageAtomicUMax32,
            Opcode::StorageAtomicInc32,
            Opcode::StorageAtomicDec32,
            Opcode::StorageAtomicAnd32,
            Opcode::StorageAtomicOr32,
            Opcode::StorageAtomicXor32,
            Opcode::StorageAtomicExchange32,
        ] {
            s.inst(b, op, &[binding, offset, x]);
        }
        s.inst(
            b,
            Opcode::StorageAtomicCompareExchange32,
            &[binding, offset, Value::U32(0), x],
        );
        s.inst(
            b,
            Opcode::StorageAtomicAddF32,
            &[binding, offset, Value::F32(1.0)],
        );
        for op in [
            Opcode::StorageAtomicAddF32x2,
            Opcode::StorageAtomicMinF32x2,
            Opcode::StorageAtomicMaxF32x2,
        ] {
            s.inst(b, op, &[binding, offset, f32x2]);
        }
        for op in [
            Opcode::StorageAtomicIAdd64,
            Opcode::StorageAtomicSMin64,
            Opcode::StorageAtomicUMin64,
            Opcode::StorageAtomicSMax64,
            Opcode::StorageAtomicUMax64,
            Opcode::StorageAtomicAnd64,
            Opcode::StorageAtomicOr64,
            Opcode::StorageAtomicXor64,
            Opcode::StorageAtomicExchange64,
        ] {
            s.inst(b, op, &[binding, offset, Value::U64(1)]);
        }
        for op in [
            Opcode::StorageAtomicIAdd32x2,
            Opcode::StorageAtomicSMin32x2,
            Opcode::StorageAtomicUMin32x2,
            Opcode::StorageAtomicSMax32x2,
            Opcode::StorageAtomicUMax32x2,
            Opcode::StorageAtomicAnd32x2,
            Opcode::StorageAtomicOr32x2,
            Opcode::StorageAtomicXor32x2,
            Opcode::StorageAtomicExchange32x2,
        ] {
            s.inst(b, op, &[binding, offset, u32x2]);
        }
    }
    let info = ShaderInfo {
        uses_int64: true,
        uses_int64_bit_atomics: true,
        uses_global_increment: true,
        uses_global_decrement: true,
        uses_atomic_f32_add: true,
        uses_atomic_f32x2_add: true,
        uses_atomic_f32x2_min: true,
        uses_atomic_f32x2_max: true,
        used_storage_buffer_types: Type::U32 | Type::U64 | Type::U32X2,
        storage_buffers_descriptors: vec![storage_descriptor(0x110)],
        ..ShaderInfo::default()
    };
    finish_compute(s, b, info)
}

#[test]
fn storage_atomics_with_and_without_int64_atomics() {
    let program = storage_atomic_program();
    let atomics64 = Profile {
        support_int64: true,
        support_int64_atomics: true,
        ..Profile::default()
    };
    let module = emit(
        "storage_atomics_int64",
        &atomics64,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::AtomicIAdd), 2 * 2);
    assert_eq!(count_ops(&module, Op::AtomicExchange), 2 * 2);
    assert_eq!(
        count_ops(&module, Op::AtomicCompareExchange),
        2 + 6,
        "the compare-exchange and one per CAS loop"
    );
    assert_eq!(count_entry_ops(&module, Op::FunctionCall), 2 * 6);

    let plain64 = Profile {
        support_int64: true,
        ..Profile::default()
    };
    let module = emit(
        "storage_atomics_no_int64_atomics",
        &plain64,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::AtomicIAdd), 2);
    assert_eq!(count_ops(&module, Op::AtomicExchange), 2);

    let no_aliasing = Profile {
        support_descriptor_aliasing: false,
        ..plain64
    };
    emit(
        "storage_atomics_no_aliasing",
        &no_aliasing,
        &RuntimeInfo::default(),
        &program,
    );
}

/// A 64-bit shared and a 64-bit storage compare-exchange, each at a dynamic and a constant
/// offset.
fn wide_compare_exchange_program() -> TranslatedProgram {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let wide = s.inst(b, Opcode::ConvertU64U32, &[x]);
    for offset in [x, Value::U32(8)] {
        s.inst(
            b,
            Opcode::SharedAtomicCompareExchange64,
            &[offset, Value::U64(0), wide],
        );
        s.inst(
            b,
            Opcode::StorageAtomicCompareExchange64,
            &[Value::U32(0), offset, Value::U64(0), wide],
        );
    }
    let info = ShaderInfo {
        uses_int64: true,
        uses_int64_bit_atomics: true,
        used_storage_buffer_types: Type::U64 | Type::U32X2,
        storage_buffers_descriptors: vec![storage_descriptor(0x110)],
        ..ShaderInfo::default()
    };
    let mut program = finish_compute(s, b, info);
    program.shared_memory_size = 1024;
    program
}

#[test]
fn wide_compare_exchanges_with_and_without_int64_atomics() {
    let program = wide_compare_exchange_program();
    let native = Profile {
        supported_spirv: 0x0001_0400,
        support_int64: true,
        support_int64_atomics: true,
        support_shared_int64_atomics: true,
        support_explicit_workgroup_layout: true,
        ..Profile::default()
    };
    let module = emit(
        "wide_cas_native",
        &native,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::AtomicCompareExchange), 2 + 2);
    assert_eq!(count_ops(&module, Op::Select), 0);

    let no_explicit_layout = Profile {
        support_explicit_workgroup_layout: false,
        ..native
    };
    let module = emit(
        "wide_cas_no_explicit_layout",
        &no_explicit_layout,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(
        count_ops(&module, Op::AtomicCompareExchange),
        2,
        "only the storage compare-exchanges stay atomic"
    );
    assert_eq!(count_ops(&module, Op::Select), 2);

    let plain64 = Profile {
        support_int64: true,
        ..Profile::default()
    };
    let module = emit(
        "wide_cas_no_int64_atomics",
        &plain64,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::AtomicCompareExchange), 0);
    assert_eq!(count_ops(&module, Op::Select), 2 + 2);
}

fn shared_memory_program() -> TranslatedProgram {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let u32x2 = s.inst(
        b,
        Opcode::CompositeConstructU32x2,
        &[Value::U32(1), Value::U32(2)],
    );
    for (load, write) in [
        (Opcode::LoadSharedU8, Opcode::WriteSharedU8),
        (Opcode::LoadSharedS8, Opcode::WriteSharedU8),
        (Opcode::LoadSharedU16, Opcode::WriteSharedU16),
        (Opcode::LoadSharedS16, Opcode::WriteSharedU16),
        (Opcode::LoadSharedU32, Opcode::WriteSharedU32),
        (Opcode::LoadSharedU64, Opcode::WriteSharedU64),
        (Opcode::LoadSharedU128, Opcode::WriteSharedU128),
    ] {
        let value = s.inst(b, load, &[x]);
        s.inst(b, write, &[x, value]);
    }
    for op in [
        Opcode::SharedAtomicIAdd32,
        Opcode::SharedAtomicSMin32,
        Opcode::SharedAtomicUMin32,
        Opcode::SharedAtomicSMax32,
        Opcode::SharedAtomicUMax32,
        Opcode::SharedAtomicInc32,
        Opcode::SharedAtomicDec32,
        Opcode::SharedAtomicAnd32,
        Opcode::SharedAtomicOr32,
        Opcode::SharedAtomicXor32,
        Opcode::SharedAtomicExchange32,
    ] {
        s.inst(b, op, &[x, x]);
    }
    s.inst(b, Opcode::SharedAtomicExchange64, &[x, Value::U64(1)]);
    s.inst(b, Opcode::SharedAtomicExchange32x2, &[x, u32x2]);
    s.inst(b, Opcode::SharedAtomicCompareExchange32, &[x, x, x]);
    let info = ShaderInfo {
        uses_int8: true,
        uses_int16: true,
        uses_int64: true,
        uses_int64_bit_atomics: true,
        uses_shared_increment: true,
        uses_shared_decrement: true,
        ..ShaderInfo::default()
    };
    let mut program = finish_compute(s, b, info);
    program.shared_memory_size = 1024;
    program
}

fn workgroup_variables(module: &dr::Module) -> usize {
    module
        .types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .count()
}

#[test]
fn shared_memory_with_and_without_explicit_workgroup_layout() {
    let program = shared_memory_program();
    let words = Profile {
        support_int64: true,
        ..Profile::default()
    };
    let module = emit("shared_words", &words, &RuntimeInfo::default(), &program);
    assert_eq!(workgroup_variables(&module), 1);
    assert_eq!(count_ops(&module, Op::BitFieldUExtract), 2);
    assert_eq!(count_ops(&module, Op::BitFieldSExtract), 2);
    assert_eq!(
        count_entry_ops(&module, Op::FunctionCall),
        4 + 2,
        "narrow stores and increment/decrement go through CAS loops"
    );

    let explicit = Profile {
        supported_spirv: 0x0001_0400,
        support_int8: true,
        support_int16: true,
        support_int64: true,
        support_int64_atomics: true,
        support_shared_int64_atomics: true,
        support_explicit_workgroup_layout: true,
        support_workgroup_layout_8bit_access: true,
        support_workgroup_layout_16bit_access: true,
        ..Profile::default()
    };
    let module = emit(
        "shared_explicit",
        &explicit,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(workgroup_variables(&module), 6);
    assert_eq!(count_ops(&module, Op::BitFieldUExtract), 0);
    assert_eq!(count_entry_ops(&module, Op::FunctionCall), 2);
    assert_eq!(count_ops(&module, Op::AtomicExchange), 2);
}

fn warp_program(stage_ops: bool) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    let pred = Value::U1(true);
    s.inst(b, Opcode::LaneId, &[]);
    for op in [
        Opcode::VoteAll,
        Opcode::VoteAny,
        Opcode::VoteEqual,
        Opcode::SubgroupBallot,
    ] {
        s.inst(b, op, &[pred]);
    }
    for op in [
        Opcode::SubgroupEqMask,
        Opcode::SubgroupLtMask,
        Opcode::SubgroupLeMask,
        Opcode::SubgroupGtMask,
        Opcode::SubgroupGeMask,
    ] {
        s.inst(b, op, &[]);
    }
    let clamp = Value::U32(31);
    let segment = Value::U32(0);
    for (op, index) in [
        (Opcode::ShuffleIndex, x),
        (Opcode::ShuffleIndex, Value::U32(3)),
        (Opcode::ShuffleUp, Value::U32(1)),
        (Opcode::ShuffleDown, Value::U32(1)),
        (Opcode::ShuffleButterfly, Value::U32(1)),
    ] {
        let shuffle = s.inst(b, op, &[x, index, clamp, segment]);
        s.inst(b, Opcode::GetInBoundsFromOp, &[shuffle]);
    }
    if stage_ops {
        s.inst(b, Opcode::QuadBroadcast, &[x, Value::U32(1)]);
        s.inst(b, Opcode::QuadSwap, &[x, Value::U32(0)]);
    }
    s.inst(
        b,
        Opcode::FSwizzleAdd,
        &[Value::F32(1.0), Value::F32(2.0), x],
    );
    let info = ShaderInfo {
        uses_subgroup_invocation_id: true,
        uses_subgroup_vote: true,
        uses_subgroup_mask: true,
        uses_subgroup_shuffles: true,
        uses_quad_shuffles: stage_ops,
        uses_fswzadd: true,
        ..ShaderInfo::default()
    };
    finish_compute(s, b, info)
}

#[test]
fn warp_operations_on_native_and_emulated_subgroups() {
    let program = warp_program(true);
    let native = emit(
        "warp_native",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&native, Op::GroupNonUniformShuffleUp), 1);
    assert_eq!(count_ops(&native, Op::GroupNonUniformShuffleDown), 1);
    assert_eq!(count_ops(&native, Op::GroupNonUniformQuadSwap), 1);
    assert_eq!(count_ops(&native, Op::GroupNonUniformAll), 1);
    assert_eq!(count_ops(&native, Op::VectorExtractDynamic), 2);

    let wide = Profile {
        warp_size_potentially_larger_than_guest: true,
        max_subgroup_size: 64,
        ..Profile::default()
    };
    let module = emit("warp_wide", &wide, &RuntimeInfo::default(), &program);
    assert_eq!(count_ops(&module, Op::GroupNonUniformAll), 0);
    assert!(count_ops(&module, Op::VectorExtractDynamic) > 2);

    let broken = Profile {
        warp_size_potentially_larger_than_guest: true,
        max_subgroup_size: 64,
        has_broken_spirv_subgroup_mask_vector_extract_dynamic: true,
        has_broken_spirv_subgroup_shuffle: true,
        support_shuffle_relative: false,
        support_quad_shuffles: false,
        ..Profile::default()
    };
    let module = emit("warp_broken", &broken, &RuntimeInfo::default(), &program);
    assert_eq!(count_ops(&module, Op::VectorExtractDynamic), 2);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffle), 0);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffleUp), 0);
    assert!(count_ops(&module, Op::GroupNonUniformBroadcast) >= 4 * 64);

    let no_relative = Profile {
        support_shuffle_relative: false,
        ..Profile::default()
    };
    let module = emit(
        "warp_no_relative",
        &no_relative,
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffleUp), 0);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffle), 3);

    let no_subgroups = Profile {
        supported_subgroup_stages: 0,
        ..Profile::default()
    };
    let module = emit(
        "warp_no_subgroups",
        &no_subgroups,
        &RuntimeInfo::default(),
        &warp_program(false),
    );
    let subgroup_ops = module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| format!("{:?}", i.class.opcode).starts_with("GroupNonUniform"))
        .count();
    assert_eq!(subgroup_ops, 0);
}

#[test]
fn quad_shuffles_without_host_quad_support_define_the_invocation_id() {
    let mut s = Synthetic::new();
    let (b, x) = compute_block(&mut s);
    s.inst(b, Opcode::QuadBroadcast, &[x, Value::U32(2)]);
    s.inst(b, Opcode::QuadSwap, &[x, Value::U32(1)]);
    let info = ShaderInfo {
        uses_quad_shuffles: true,
        ..ShaderInfo::default()
    };
    let program = finish_compute(s, b, info);
    let profile = Profile {
        support_quad_shuffles: false,
        ..Profile::default()
    };
    let module = emit("quad_fallback", &profile, &RuntimeInfo::default(), &program);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffle), 1);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffleXor), 1);
}

#[test]
fn fragment_derivatives() {
    let mut s = Synthetic::new();
    let b = s.block();
    let value = s.inst(
        b,
        Opcode::GetAttribute,
        &[Value::Attribute(Attribute::generic(0, 0)), Value::U32(0)],
    );
    for op in [
        Opcode::DPdxFine,
        Opcode::DPdyFine,
        Opcode::DPdxCoarse,
        Opcode::DPdyCoarse,
    ] {
        s.inst(b, op, &[value]);
    }
    let info = ShaderInfo {
        loads: varying(&generic0()),
        uses_derivatives: true,
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Fragment,
        info,
    );
    let module = emit(
        "derivatives",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::DPdxFine), 1);
    assert_eq!(count_ops(&module, Op::DPdyCoarse), 1);
}
