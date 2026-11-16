//! Cooperative-matrix HMMA and the Vulkan memory model it brings to the whole module.

use rspirv::dr::Instruction;
use rspirv::spirv::{MemoryAccess, MemoryModel, MemorySemantics};

use super::*;
use crate::backend::spirv::cooperative_hmma_test;

fn cooperative_profile(supported_spirv: u32) -> Profile {
    Profile {
        supported_spirv,
        support_cooperative_hmma: true,
        support_cooperative_hmma_f32: true,
        ..Profile::default()
    }
}

/// A compute program with HMMA.16816 in both accumulator forms, one HMMA.1688, and shared
/// memory accesses around a barrier.
fn hmma_program(workgroup_x: u32) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let local = s.inst(b, Opcode::LocalInvocationId, &[]);
    let x = s.inst(b, Opcode::CompositeExtractU32x3, &[local, Value::U32(0)]);
    let a = s.inst(b, Opcode::CompositeConstructU32x4, &[x, x, x, x]);
    let pair = s.inst(b, Opcode::CompositeConstructU32x2, &[x, x]);
    let d = s.inst_flags(b, Opcode::TensorMulAdd, &[a, pair, pair], 16);
    s.inst_flags(b, Opcode::TensorMulAddF32, &[a, pair, a], 16);
    s.inst_flags(b, Opcode::TensorMulAdd, &[a, pair, d], 8);
    let word = s.inst(b, Opcode::CompositeExtractU32x2, &[d, Value::U32(0)]);
    s.inst(b, Opcode::WriteSharedU32, &[x, word]);
    s.inst(b, Opcode::WriteSharedU8, &[x, word]);
    s.inst(b, Opcode::WriteSharedU128, &[x, a]);
    s.inst(b, Opcode::Barrier, &[]);
    s.inst(b, Opcode::WorkgroupMemoryBarrier, &[]);
    s.inst(b, Opcode::LoadSharedU32, &[x]);
    s.inst(b, Opcode::LoadSharedU16, &[x]);
    s.inst(b, Opcode::SharedAtomicIAdd32, &[x, x]);
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo {
            uses_local_invocation_id: true,
            uses_int8: true,
            uses_int16: true,
            ..ShaderInfo::default()
        },
    );
    program.workgroup_size = [workgroup_x, 1, 1];
    program.shared_memory_size = 1024;
    program
}

fn memory_model(module: &dr::Module) -> MemoryModel {
    match module.memory_model.as_ref().map(|i| &i.operands[1]) {
        Some(Operand::MemoryModel(model)) => *model,
        other => panic!("memory model operand {other:?}"),
    }
}

fn instructions(module: &dr::Module) -> impl Iterator<Item = &Instruction> {
    module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
}

fn accesses_with(module: &dr::Module, op: Op, access: MemoryAccess) -> usize {
    instructions(module)
        .filter(|i| i.class.opcode == op)
        .filter(|i| {
            i.operands
                .iter()
                .any(|o| matches!(o, Operand::MemoryAccess(a) if a.contains(access)))
        })
        .count()
}

fn barrier_semantics(module: &dr::Module) -> Vec<MemorySemantics> {
    let constant = |id: u32| {
        module
            .types_global_values
            .iter()
            .find(|i| i.result_id == Some(id))
            .and_then(|i| match i.operands.first() {
                Some(Operand::LiteralBit32(bits)) => MemorySemantics::from_bits(*bits),
                _ => None,
            })
            .expect("barrier semantics are a constant")
    };
    instructions(module)
        .filter_map(|i| match (i.class.opcode, i.operands.as_slice()) {
            (Op::ControlBarrier, [_, _, Operand::IdMemorySemantics(id)])
            | (Op::MemoryBarrier, [_, Operand::IdMemorySemantics(id)]) => Some(constant(*id)),
            _ => None,
        })
        .collect()
}

#[test]
fn hmma_16816_runs_on_cooperative_matrices_under_the_vulkan_memory_model() {
    for (name, spirv) in [
        ("coop_hmma_1_3", 0x0001_0300),
        ("coop_hmma_1_6", 0x0001_0600),
    ] {
        let profile = cooperative_profile(spirv);
        let module = emit(name, &profile, &RuntimeInfo::default(), &hmma_program(64));
        assert_eq!(memory_model(&module), MemoryModel::Vulkan, "{name}");
        assert_eq!(
            count_ops(&module, Op::CooperativeMatrixMulAddKHR),
            2,
            "{name}"
        );
        assert!(
            count_ops(&module, Op::FunctionCall) > 0,
            "{name}: HMMA.1688 stays on the software path"
        );
        assert!(
            accesses_with(&module, Op::Load, MemoryAccess::NON_PRIVATE_POINTER) >= 2,
            "{name}: shared loads are non-private"
        );
        assert!(
            accesses_with(&module, Op::Store, MemoryAccess::MAKE_POINTER_AVAILABLE) >= 2,
            "{name}: shared stores make the pointer available"
        );
        let semantics = barrier_semantics(&module);
        assert_eq!(semantics.len(), 2, "{name}");
        for semantics in semantics {
            assert!(
                semantics.contains(MemorySemantics::MAKE_AVAILABLE | MemorySemantics::MAKE_VISIBLE),
                "{name}: {semantics:?}"
            );
        }
    }
}

#[test]
fn hmma_stays_on_subgroup_shuffles_without_whole_warps_or_support() {
    let partial = emit(
        "coop_hmma_partial_warp",
        &cooperative_profile(0x0001_0600),
        &RuntimeInfo::default(),
        &hmma_program(48),
    );
    let unsupported = emit(
        "coop_hmma_unsupported",
        &Profile::default(),
        &RuntimeInfo::default(),
        &hmma_program(64),
    );
    for module in [partial, unsupported] {
        assert_eq!(memory_model(&module), MemoryModel::GLSL450);
        assert_eq!(count_ops(&module, Op::CooperativeMatrixMulAddKHR), 0);
        assert_eq!(
            accesses_with(&module, Op::Load, MemoryAccess::NON_PRIVATE_POINTER),
            0
        );
        for semantics in barrier_semantics(&module) {
            assert!(
                !semantics
                    .intersects(MemorySemantics::MAKE_AVAILABLE | MemorySemantics::MAKE_VISIBLE)
            );
        }
    }
}

#[test]
fn the_device_self_test_shader_is_valid() {
    for spirv in [0x0001_0300, 0x0001_0600] {
        for f32_accumulators in [false, true] {
            let code = cooperative_hmma_test(spirv, f32_accumulators);
            let name = format!("coop_hmma_self_test_{spirv:x}_{f32_accumulators}");
            validate(&name, &cooperative_profile(spirv), &code);
            let module = dr::load_words(&code).expect("rspirv parses the module");
            assert_eq!(count_ops(&module, Op::CooperativeMatrixMulAddKHR), 1);
            assert_eq!(memory_model(&module), MemoryModel::Vulkan);
        }
    }
}
