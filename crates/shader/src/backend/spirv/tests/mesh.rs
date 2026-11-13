//! Task and mesh programs (native `SPV_EXT_mesh_shader` and the compute emulation), the mesh
//! emulation's vertex and distribution modules, and the software TENSOR_MUL_ADD.

use rspirv::dr;
use rspirv::spirv::{BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, Op};

use super::{Synthetic, count_ops, emit, validate};
use crate::backend::Bindings;
use crate::backend::spirv::{
    EmitError, MeshEmulationDistributionConstants, emit_mesh_emulation_distribution,
    emit_mesh_emulation_vertex, emit_spirv, mesh_emulation_block_words,
};
use crate::info::{MeshInfo, StorageBufferDescriptor};
use crate::ir::{Attribute, Opcode, SyntaxNode, Type, Value};
use crate::runtime::{MeshEmulation, MeshEmulationTask};
use crate::translate::TranslatedProgram;
use crate::{Profile, RuntimeInfo, ShaderInfo, ShaderStage};

use Opcode::*;

const TOPOLOGY_POINTS: u32 = 0;
const TOPOLOGY_LINES: u32 = 1;
const TOPOLOGY_TRIANGLES: u32 = 4;

fn mesh_profile() -> Profile {
    Profile {
        supported_spirv: 0x0001_0600,
        support_mesh_shader: true,
        ..Profile::default()
    }
}

fn emulation_profile() -> Profile {
    Profile {
        supported_spirv: 0x0001_0600,
        ..Profile::default()
    }
}

fn line(attribute: Attribute) -> u32 {
    u32::from(attribute.0)
}

fn mesh_info(topology: u32) -> MeshInfo {
    MeshInfo {
        payload_words: 12,
        max_vertices: 40,
        max_primitives: 70,
        topology,
        vertex_lines: vec![
            line(Attribute::POSITION_X),
            line(Attribute::POSITION_Y),
            line(Attribute::POSITION_Z),
            line(Attribute::POSITION_W),
            line(Attribute::POINT_SIZE),
            line(Attribute::generic(0, 0)),
            line(Attribute::generic(0, 3)),
            line(Attribute::generic(2, 1)),
            line(Attribute::PRIMITIVE_ID),
        ],
        primitive_lines: vec![
            line(Attribute::PRIMITIVE_ID),
            line(Attribute::LAYER),
            line(Attribute::VIEWPORT_INDEX),
            line(Attribute::generic(1, 0)),
            line(Attribute::POSITION_X),
        ],
    }
}

fn stage_info(mesh: MeshInfo) -> ShaderInfo {
    ShaderInfo {
        uses_workgroup_id: true,
        uses_local_invocation_id: true,
        mesh,
        ..ShaderInfo::default()
    }
}

/// A task program: reads its workgroup index, writes a mesh workgroup count and payload
/// words, one of them at a per-invocation address.
fn task_program() -> TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Prologue, &[]);
    let index = s.inst(b, IsbeRead, &[Value::U32(0)]);
    let local = s.inst(b, LocalInvocationId, &[]);
    let lane = s.inst(b, CompositeExtractU32x3, &[local, Value::U32(0)]);
    let address = s.inst(b, ShiftLeftLogical32, &[lane, Value::U32(2)]);
    let address = s.inst(b, IAdd32, &[address, Value::U32(0x10)]);
    s.inst(b, IsbeWriteAttribute, &[Value::U32(4), Value::U32(3)]);
    s.inst(b, IsbeWriteAttribute, &[address, index]);
    s.inst(b, Barrier, &[]);
    s.inst(b, Epilogue, &[]);
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Task,
        stage_info(MeshInfo {
            payload_words: 12,
            ..MeshInfo::default()
        }),
    );
    program.workgroup_size = [32, 1, 1];
    program
}

/// A mesh program using every stage buffer op, its writes under a branch whose merge holds the
/// epilogue and a phi.
fn mesh_program(mesh: MeshInfo) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let entry = s.block();
    let then = s.block();
    let merge = s.block();
    s.inst(entry, Prologue, &[]);
    let own = s.inst(entry, IsbeRead, &[Value::U32(0)]);
    let local = s.inst(entry, LocalInvocationId, &[]);
    let lane = s.inst(entry, CompositeExtractU32x3, &[local, Value::U32(0)]);
    let address = s.inst(entry, ShiftLeftLogical32, &[lane, Value::U32(2)]);
    let payload_address = s.inst(entry, IAdd32, &[address, Value::U32(0x10)]);
    let payload = s.inst(entry, IsbeRead, &[payload_address]);
    let first = s.inst(entry, ULessThan, &[lane, Value::U32(16)]);
    let cond = s.inst(entry, ConditionRef, &[first]);
    s.inst(then, IsbeWriteAttribute, &[address, payload]);
    s.inst(then, IsbeWriteAttribute, &[Value::U32(0x80), own]);
    s.inst(then, IsbeWriteMapU32, &[Value::U32(3), Value::U32(5)]);
    s.inst(then, IsbeWriteMapU8, &[Value::U32(4), lane]);
    s.inst(then, IsbeWriteMapU8, &[lane, own]);
    let phi = s.phi(merge, Type::U32);
    s.ir.add_phi_operand(phi, then, payload);
    s.ir.add_phi_operand(phi, entry, Value::U32(0));
    s.inst(
        merge,
        IsbeWriteAttribute,
        &[Value::U32(0x100), Value::Inst(phi)],
    );
    s.inst(merge, Barrier, &[]);
    s.inst(merge, WorkgroupMemoryBarrier, &[]);
    s.inst(merge, Epilogue, &[]);
    s.ir.add_branch(entry, then);
    s.ir.add_branch(entry, merge);
    s.ir.add_branch(then, merge);
    let mut program = s.finish(
        vec![
            SyntaxNode::Block(entry),
            SyntaxNode::If {
                cond,
                body: then,
                merge,
            },
            SyntaxNode::Block(then),
            SyntaxNode::EndIf { merge },
            SyntaxNode::Block(merge),
            SyntaxNode::Return,
        ],
        ShaderStage::Mesh,
        stage_info(mesh),
    );
    program.workgroup_size = [32, 1, 1];
    program
}

fn execution_model(module: &dr::Module) -> ExecutionModel {
    match module.entry_points[0].operands[0] {
        dr::Operand::ExecutionModel(model) => model,
        ref other => panic!("entry point without execution model: {other:?}"),
    }
}

fn has_execution_mode(module: &dr::Module, mode: ExecutionMode) -> bool {
    module
        .execution_modes
        .iter()
        .any(|m| m.operands[1] == dr::Operand::ExecutionMode(mode))
}

fn builtins(module: &dr::Module) -> Vec<BuiltIn> {
    module
        .annotations
        .iter()
        .filter_map(|a| match a.operands.get(2) {
            Some(dr::Operand::BuiltIn(builtin)) => Some(*builtin),
            _ => None,
        })
        .collect()
}

fn per_primitive_decorations(module: &dr::Module) -> usize {
    module
        .annotations
        .iter()
        .filter(|a| {
            a.operands.get(1) == Some(&dr::Operand::Decoration(Decoration::PerPrimitiveEXT))
        })
        .count()
}

fn emit_error(profile: &Profile, runtime: &RuntimeInfo, program: &TranslatedProgram) -> EmitError {
    let mut bindings = Bindings::default();
    emit_spirv(profile, runtime, program, &mut bindings).expect_err("emission fails")
}

#[test]
fn native_task_program_emits_its_mesh_tasks() {
    let module = emit(
        "native_task",
        &mesh_profile(),
        &RuntimeInfo::default(),
        &task_program(),
    );
    assert_eq!(execution_model(&module), ExecutionModel::TaskEXT);
    assert_eq!(count_ops(&module, Op::EmitMeshTasksEXT), 1);
    assert_eq!(count_ops(&module, Op::Return), 0);
    assert!(
        module
            .capabilities
            .iter()
            .any(|c| c.operands[0] == dr::Operand::Capability(Capability::MeshShadingEXT))
    );
    assert!(count_ops(&module, Op::ControlBarrier) >= 3);
}

#[test]
fn native_mesh_program_for_every_topology() {
    for (name, topology, mode, indices) in [
        (
            "native_mesh_triangles",
            TOPOLOGY_TRIANGLES,
            ExecutionMode::OutputTrianglesEXT,
            BuiltIn::PrimitiveTriangleIndicesEXT,
        ),
        (
            "native_mesh_lines",
            TOPOLOGY_LINES,
            ExecutionMode::OutputLinesEXT,
            BuiltIn::PrimitiveLineIndicesEXT,
        ),
        (
            "native_mesh_points",
            TOPOLOGY_POINTS,
            ExecutionMode::OutputPoints,
            BuiltIn::PrimitivePointIndicesEXT,
        ),
    ] {
        let runtime = RuntimeInfo {
            convert_depth_mode: true,
            fixed_state_point_size: Some(2.0),
            ..RuntimeInfo::default()
        };
        let module = emit(
            name,
            &mesh_profile(),
            &runtime,
            &mesh_program(mesh_info(topology)),
        );
        assert_eq!(execution_model(&module), ExecutionModel::MeshEXT, "{name}");
        assert_eq!(count_ops(&module, Op::SetMeshOutputsEXT), 1, "{name}");
        assert!(has_execution_mode(&module, mode), "{name}");
        assert!(
            has_execution_mode(&module, ExecutionMode::OutputVertices),
            "{name}"
        );
        assert!(
            has_execution_mode(&module, ExecutionMode::OutputPrimitivesEXT),
            "{name}"
        );
        let builtins = builtins(&module);
        for builtin in [
            indices,
            BuiltIn::Position,
            BuiltIn::PointSize,
            BuiltIn::PrimitiveId,
            BuiltIn::Layer,
            BuiltIn::ViewportIndex,
        ] {
            assert!(builtins.contains(&builtin), "{name}: {builtin:?}");
        }
        assert_eq!(per_primitive_decorations(&module), 4, "{name}");
    }
}

#[test]
fn mesh_program_without_outputs() {
    let mesh = MeshInfo {
        payload_words: 0,
        max_vertices: 3,
        max_primitives: 1,
        topology: TOPOLOGY_TRIANGLES,
        ..MeshInfo::default()
    };
    emit(
        "native_mesh_empty",
        &mesh_profile(),
        &RuntimeInfo::default(),
        &mesh_program(mesh),
    );
}

#[test]
fn mesh_program_beside_shared_memory() {
    for (name, explicit_layout) in [
        ("native_mesh_shared", false),
        ("native_mesh_explicit_shared", true),
    ] {
        let profile = Profile {
            support_explicit_workgroup_layout: explicit_layout,
            ..mesh_profile()
        };
        let mut program = mesh_program(mesh_info(TOPOLOGY_TRIANGLES));
        program.shared_memory_size = 256;
        emit(name, &profile, &RuntimeInfo::default(), &program);
    }
}

#[test]
fn host_without_mesh_shaders_rejects_native_programs() {
    let runtime = RuntimeInfo::default();
    let task = task_program();
    let mesh = mesh_program(mesh_info(TOPOLOGY_TRIANGLES));
    for program in [&task, &mesh] {
        let error = emit_error(&emulation_profile(), &runtime, program);
        assert!(
            matches!(error, EmitError::UnsupportedStage(stage) if stage == program.stage),
            "{error}"
        );
        let old_spirv = Profile {
            supported_spirv: 0x0001_0300,
            ..mesh_profile()
        };
        let error = emit_error(&old_spirv, &runtime, program);
        assert!(matches!(error, EmitError::Unsupported(_)), "{error}");
    }
}

fn emulation(task: bool) -> MeshEmulation {
    MeshEmulation {
        output_binding: 5,
        task: task.then_some(MeshEmulationTask {
            payload_binding: 6,
            offsets_binding: 7,
            args_binding: 8,
            payload_stride: 8,
        }),
    }
}

#[test]
fn emulated_task_program_writes_the_payload_buffer() {
    let runtime = RuntimeInfo {
        mesh_emulation: Some(emulation(true)),
        ..RuntimeInfo::default()
    };
    let module = emit(
        "emulated_task",
        &emulation_profile(),
        &runtime,
        &task_program(),
    );
    assert_eq!(execution_model(&module), ExecutionModel::GLCompute);
    assert_eq!(count_ops(&module, Op::EmitMeshTasksEXT), 0);
    assert!(module.extensions.iter().all(|e| !matches!(
        &e.operands[0],
        dr::Operand::LiteralString(s) if s == "SPV_EXT_mesh_shader"
    )));

    let without_buffers = RuntimeInfo {
        mesh_emulation: Some(emulation(false)),
        ..RuntimeInfo::default()
    };
    let error = emit_error(&emulation_profile(), &without_buffers, &task_program());
    assert!(matches!(error, EmitError::Unsupported(_)), "{error}");
}

#[test]
fn emulated_mesh_program_with_and_without_a_task() {
    for (name, task) in [("emulated_mesh_task", true), ("emulated_mesh", false)] {
        let runtime = RuntimeInfo {
            mesh_emulation: Some(emulation(task)),
            convert_depth_mode: true,
            ..RuntimeInfo::default()
        };
        let mut program = mesh_program(mesh_info(TOPOLOGY_TRIANGLES));
        program.info.storage_buffers_descriptors = vec![StorageBufferDescriptor {
            cbuf_index: 0,
            cbuf_offset: 0x10,
            count: 1,
            is_written: true,
            indirect_offset: StorageBufferDescriptor::DIRECT,
            write_extent: Default::default(),
        }];
        let module = emit(name, &emulation_profile(), &runtime, &program);
        assert_eq!(
            execution_model(&module),
            ExecutionModel::GLCompute,
            "{name}"
        );
        assert_eq!(count_ops(&module, Op::SetMeshOutputsEXT), 0, "{name}");
        let buffers = module
            .types_global_values
            .iter()
            .filter(|i| {
                i.class.opcode == Op::Variable
                    && i.operands[0]
                        == dr::Operand::StorageClass(rspirv::spirv::StorageClass::StorageBuffer)
            })
            .count();
        assert_eq!(buffers, if task { 5 } else { 2 }, "{name}");
    }
}

#[test]
fn stage_buffer_writes_outside_their_stage_are_rejected() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Prologue, &[]);
    s.inst(b, IsbeWriteMapU8, &[Value::U32(4), Value::U32(1)]);
    s.inst(b, Epilogue, &[]);
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Task,
        stage_info(MeshInfo::default()),
    );
    let error = emit_error(&mesh_profile(), &RuntimeInfo::default(), &program);
    assert!(
        matches!(error, EmitError::UnsupportedOpcode(IsbeWriteMapU8)),
        "{error}"
    );

    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Prologue, &[]);
    s.inst(b, IsbeWriteMapU32, &[Value::U32(7), Value::U32(1)]);
    s.inst(b, Epilogue, &[]);
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Mesh,
        stage_info(mesh_info(TOPOLOGY_TRIANGLES)),
    );
    let error = emit_error(&mesh_profile(), &RuntimeInfo::default(), &program);
    assert!(
        matches!(error, EmitError::UnsupportedOpcode(IsbeWriteMapU32)),
        "{error}"
    );
}

#[test]
fn mesh_emulation_vertex_module() {
    let mesh = mesh_info(TOPOLOGY_TRIANGLES);
    for (name, profile, args, convert_depth) in [
        ("emulation_vertex", emulation_profile(), None, false),
        ("emulation_vertex_task", emulation_profile(), Some(3), true),
        (
            "emulation_vertex_spirv13",
            Profile::default(),
            Some(3),
            true,
        ),
    ] {
        let code = emit_mesh_emulation_vertex(&profile, &mesh, 2, args, convert_depth)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        validate(name, &profile, &code);
        let module = dr::load_words(&code).expect("rspirv parses the module");
        assert_eq!(execution_model(&module), ExecutionModel::Vertex, "{name}");
    }
    assert_eq!(mesh_emulation_block_words(&mesh), 40 * 9 + 70 * 3);
}

#[test]
fn mesh_emulation_distribution_module() {
    for (name, profile) in [
        ("emulation_distribution", emulation_profile()),
        ("emulation_distribution_spirv13", Profile::default()),
    ] {
        let code =
            emit_mesh_emulation_distribution(&profile).unwrap_or_else(|e| panic!("{name}: {e}"));
        validate(name, &profile, &code);
        let module = dr::load_words(&code).expect("rspirv parses the module");
        assert_eq!(
            execution_model(&module),
            ExecutionModel::GLCompute,
            "{name}"
        );
        assert_eq!(count_ops(&module, Op::LoopMerge), 1, "{name}");
    }
    let constants = MeshEmulationDistributionConstants {
        tasks: 1,
        payload_words: 2,
        capacity_words: 3,
        block_words: 4,
        vertices_per_group: 5,
        max_groups: 6,
    };
    assert_eq!(constants.words(), [1, 2, 3, 4, 5, 6]);
}

fn tensor_program(depths: &[u32]) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Prologue, &[]);
    let local = s.inst(b, LocalInvocationId, &[]);
    let x = s.inst(b, CompositeExtractU32x3, &[local, Value::U32(0)]);
    let a = s.inst(b, CompositeConstructU32x4, &[x, x, x, x]);
    let bc = s.inst(b, CompositeConstructU32x2, &[x, Value::U32(0x3c00_3c00)]);
    for &depth in depths {
        let d = s.inst_flags(b, TensorMulAdd, &[a, bc, bc], depth);
        let low = s.inst(b, CompositeExtractU32x2, &[d, Value::U32(0)]);
        s.inst(b, WriteSharedU32, &[Value::U32(0), low]);
    }
    s.inst(b, Epilogue, &[]);
    let info = ShaderInfo {
        uses_local_invocation_id: true,
        ..ShaderInfo::default()
    };
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        info,
    );
    program.workgroup_size = [32, 1, 1];
    program.shared_memory_size = 16;
    program
}

#[test]
fn software_tensor_mul_add() {
    let profile = emulation_profile();
    let module = emit(
        "tensor_mul_add",
        &profile,
        &RuntimeInfo::default(),
        &tensor_program(&[8, 16, 16]),
    );
    assert_eq!(module.functions.len(), 3, "main and one function per depth");
    assert_eq!(count_ops(&module, Op::FunctionCall), 3);
    assert_eq!(count_ops(&module, Op::GroupNonUniformShuffle), 2 * 4 * 4);

    let error = emit_error(&profile, &RuntimeInfo::default(), &tensor_program(&[4]));
    assert!(matches!(error, EmitError::Unsupported(_)), "{error}");
    let no_subgroups = Profile {
        supported_subgroup_stages: 0,
        ..emulation_profile()
    };
    let error = emit_error(
        &no_subgroups,
        &RuntimeInfo::default(),
        &tensor_program(&[8]),
    );
    assert!(matches!(error, EmitError::Unsupported(_)), "{error}");
}
