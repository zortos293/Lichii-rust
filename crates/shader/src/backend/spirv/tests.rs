//! Emission tests. Every module is validated with `spirv-val` when it is installed.

use std::path::PathBuf;
use std::process::Command;

use lichii_common::settings::ResolutionScale;
use rspirv::dr::{self, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass};

use super::{EmitError, emit_spirv};
use crate::backend::Bindings;
use crate::info::{ConstantBufferDescriptor, StorageBufferDescriptor, VaryingState};
use crate::ir::{Attribute, BlockId, InstId, Opcode, Program, SyntaxNode, Type, Value};
use crate::maxwell::header::OutputTopology;
use crate::maxwell::opcode::Opcode as Sass;
use crate::maxwell::translate::tests::{TestEnv, encode};
use crate::module::{DescriptorKind, ResourceLayout};
use crate::opt;
use crate::profile::HostTranslateInfo;
use crate::runtime::{AttributeType, CompareFunction};
use crate::translate::{TranslateOptions, TranslatedProgram, translate};
use crate::{Profile, RuntimeInfo, ShaderInfo, ShaderStage};

const PT: u64 = 7 << 16;

fn target_env(profile: &Profile) -> &'static str {
    match profile.supported_spirv {
        v if v >= 0x0001_0600 => "vulkan1.3",
        v if v >= 0x0001_0500 => "vulkan1.2",
        v if v >= 0x0001_0400 => "vulkan1.1spv1.4",
        _ => "vulkan1.1",
    }
}

/// Runs spirv-val on `code`; a machine without it prints a skip line and passes. Constant
/// buffers are word arrays, so validation assumes `uniformBufferStandardLayout` (Vulkan 1.2 core).
fn validate(name: &str, profile: &Profile, code: &[u32]) {
    let path: PathBuf =
        std::env::temp_dir().join(format!("lichii-spirv-{name}-{}.spv", std::process::id()));
    let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(&path, bytes).expect("write module");
    let output = Command::new("spirv-val")
        .arg("--target-env")
        .arg(target_env(profile))
        .arg("--uniform-buffer-standard-layout")
        .arg(&path)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!("skipping SPIR-V validation of {name}: spirv-val not found");
            let _ = std::fs::remove_file(&path);
            return;
        }
        Err(e) => panic!("running spirv-val: {e}"),
    };
    let disassembly = Command::new("spirv-dis")
        .arg(&path)
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    assert!(
        output.status.success(),
        "{name} failed validation:\n{}{}\n{disassembly}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn emit(
    name: &str,
    profile: &Profile,
    runtime: &RuntimeInfo,
    program: &TranslatedProgram,
) -> dr::Module {
    let mut bindings = Bindings::default();
    let code = emit_spirv(profile, runtime, program, &mut bindings)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    validate(name, profile, &code);
    dr::load_words(&code).expect("rspirv parses the module")
}

fn translate_env(env: &TestEnv) -> TranslatedProgram {
    let host = HostTranslateInfo {
        support_float64: true,
        support_float16: true,
        support_int64: true,
        support_conditional_barrier: true,
        ..HostTranslateInfo::default()
    };
    let options = TranslateOptions {
        profile: &Profile::default(),
        host: &host,
        resolution_scale: ResolutionScale::X1,
    };
    translate(env, options).expect("translate")
}

/// A synthetic program: blocks in program order with their syntax list.
struct Synthetic {
    ir: Program,
}

impl Synthetic {
    fn new() -> Synthetic {
        Synthetic { ir: Program::new() }
    }

    fn block(&mut self) -> BlockId {
        let block = self.ir.new_block();
        self.ir.block_order.push(block);
        block
    }

    fn inst(&mut self, block: BlockId, op: Opcode, args: &[Value]) -> Value {
        Value::Inst(self.ir.append(block, op, args, 0))
    }

    fn inst_flags(&mut self, block: BlockId, op: Opcode, args: &[Value], flags: u32) -> Value {
        Value::Inst(self.ir.append(block, op, args, flags))
    }

    fn phi(&mut self, block: BlockId, ty: Type) -> InstId {
        self.ir.append(block, Opcode::Phi, &[], ty.bits())
    }

    fn finish(
        mut self,
        syntax: Vec<SyntaxNode>,
        stage: ShaderStage,
        info: ShaderInfo,
    ) -> TranslatedProgram {
        self.ir.syntax_list = syntax;
        opt::compute_post_order(&mut self.ir);
        opt::verify(&self.ir).expect("valid synthetic IR");
        TranslatedProgram {
            hash: 1,
            ir: self.ir,
            info,
            stage,
            workgroup_size: [1, 1, 1],
            output_topology: OutputTopology::PointList,
            output_vertices: 0,
            invocations: 0,
            local_memory_size: 0,
            shared_memory_size: 0,
            is_geometry_passthrough: false,
        }
    }
}

fn varying(attributes: &[Attribute]) -> VaryingState {
    let mut state = VaryingState::default();
    for &attribute in attributes {
        state.set(attribute, true);
    }
    state
}

fn generic0() -> [Attribute; 4] {
    std::array::from_fn(|c| Attribute::generic(0, c as u16))
}

fn count_ops(module: &dr::Module, op: Op) -> usize {
    module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| i.class.opcode == op)
        .count()
}

#[test]
fn empty_compute_program_from_the_translator() {
    let env = TestEnv::new(&[encode(Sass::Exit, PT | 0xf)]);
    let program = translate_env(&env);
    let module = emit(
        "empty_compute",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(module.entry_points.len(), 1);
}

#[test]
fn empty_vertex_program_from_the_translator() {
    let mut env = TestEnv::new(&[encode(Sass::Exit, PT | 0xf)]);
    env.stage = ShaderStage::VertexB;
    let program = translate_env(&env);
    let module = emit(
        "empty_vertex",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    let positions = module
        .annotations
        .iter()
        .filter(|a| a.operands.get(2) == Some(&Operand::BuiltIn(rspirv::spirv::BuiltIn::Position)))
        .count();
    assert_eq!(positions, 1, "the vertex program writes Position");
}

#[test]
fn compute_program_with_shared_and_local_memory() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Opcode::Prologue, &[]);
    let local = s.inst(b, Opcode::LocalInvocationId, &[]);
    let group = s.inst(b, Opcode::WorkgroupId, &[]);
    let x = s.inst(b, Opcode::CompositeExtractU32x3, &[local, Value::U32(0)]);
    s.inst(b, Opcode::CompositeExtractU32x3, &[group, Value::U32(1)]);
    s.inst(b, Opcode::WriteLocal, &[Value::U32(3), x]);
    let loaded = s.inst(b, Opcode::LoadLocal, &[Value::U32(3)]);
    s.inst(b, Opcode::WriteLocal, &[Value::U32(0), loaded]);
    s.inst(b, Opcode::Barrier, &[]);
    s.inst(b, Opcode::WorkgroupMemoryBarrier, &[]);
    s.inst(b, Opcode::DeviceMemoryBarrier, &[]);
    s.inst(b, Opcode::Epilogue, &[]);
    let info = ShaderInfo {
        uses_workgroup_id: true,
        uses_local_invocation_id: true,
        uses_local_memory: true,
        uses_shared_increment: true,
        uses_shared_decrement: true,
        ..ShaderInfo::default()
    };
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        info,
    );
    program.workgroup_size = [64, 2, 1];
    program.shared_memory_size = 1024;
    program.local_memory_size = 64;
    let module = emit(
        "compute_shared",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    let workgroup_vars = module
        .types_global_values
        .iter()
        .filter(|i| {
            i.class.opcode == Op::Variable
                && i.operands.first() == Some(&Operand::StorageClass(StorageClass::Workgroup))
        })
        .count();
    assert_eq!(workgroup_vars, 1);
    assert_eq!(
        module.functions.len(),
        5,
        "main and two CAS loops with their bodies"
    );
    assert_eq!(count_ops(&module, Op::AtomicCompareExchange), 2);
}

#[test]
fn fragment_program_with_inputs_outputs_and_alpha_test() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Opcode::Prologue, &[]);
    for (component, attribute) in generic0().into_iter().enumerate() {
        let value = s.inst(
            b,
            Opcode::GetAttribute,
            &[Value::Attribute(attribute), Value::U32(0)],
        );
        s.inst(
            b,
            Opcode::SetFragColor,
            &[Value::U32(0), Value::U32(component as u32), value],
        );
    }
    let y = s.inst(b, Opcode::YDirection, &[]);
    s.inst(b, Opcode::SetFragDepth, &[y]);
    s.inst(b, Opcode::DemoteToHelperInvocation, &[]);
    s.inst(b, Opcode::Epilogue, &[]);
    let info = ShaderInfo {
        loads: varying(&generic0()),
        stores_frag_color: [true, false, false, false, false, false, false, false],
        stores_frag_depth: true,
        uses_demote_to_helper_invocation: true,
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Fragment,
        info,
    );
    let runtime = RuntimeInfo {
        previous_stage_stores: varying(&generic0()),
        alpha_test_func: Some(CompareFunction::Greater),
        alpha_test_reference: 0.5,
        convert_depth_mode: true,
        ..RuntimeInfo::default()
    };
    let module = emit("fragment", &Profile::default(), &runtime, &program);
    assert_eq!(
        count_ops(&module, Op::Kill),
        2,
        "alpha test and demote fallback"
    );
    let demoting = Profile {
        support_demote_to_helper_invocation: true,
        ..Profile::default()
    };
    let module = emit("fragment_demote", &demoting, &runtime, &program);
    assert_eq!(count_ops(&module, Op::DemoteToHelperInvocation), 1);
}

#[test]
fn if_with_phi_at_the_merge() {
    let mut s = Synthetic::new();
    let entry = s.block();
    let then = s.block();
    let merge = s.block();
    s.inst(entry, Opcode::Prologue, &[]);
    let helper = s.inst(entry, Opcode::IsHelperInvocation, &[]);
    let not_helper = s.inst(entry, Opcode::LogicalNot, &[helper]);
    let cond = s.inst(entry, Opcode::ConditionRef, &[not_helper]);
    let y = s.inst(then, Opcode::YDirection, &[]);
    let phi = s.phi(merge, Type::F32);
    s.ir.add_phi_operand(phi, then, y);
    s.ir.add_phi_operand(phi, entry, Value::F32(0.0));
    s.inst(
        merge,
        Opcode::SetFragColor,
        &[Value::U32(0), Value::U32(0), Value::Inst(phi)],
    );
    s.inst(merge, Opcode::Epilogue, &[]);
    s.ir.add_branch(entry, then);
    s.ir.add_branch(entry, merge);
    s.ir.add_branch(then, merge);
    let info = ShaderInfo {
        uses_is_helper_invocation: true,
        stores_frag_color: [true, false, false, false, false, false, false, false],
        ..ShaderInfo::default()
    };
    let program = s.finish(
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
        ShaderStage::Fragment,
        info,
    );
    let module = emit("if", &Profile::default(), &RuntimeInfo::default(), &program);
    assert_eq!(count_ops(&module, Op::SelectionMerge), 1);
    assert_eq!(count_ops(&module, Op::Phi), 1);
}

#[test]
fn loop_with_a_phi_naming_a_later_value() {
    let mut s = Synthetic::new();
    let entry = s.block();
    let header = s.block();
    let body = s.block();
    let continue_block = s.block();
    let merge = s.block();
    s.inst(entry, Opcode::Prologue, &[]);
    let phi = s.phi(header, Type::F32);
    let helper = s.inst(body, Opcode::IsHelperInvocation, &[]);
    let y = s.inst(body, Opcode::YDirection, &[]);
    let next = s.inst(body, Opcode::SelectF32, &[helper, y, Value::Inst(phi)]);
    let again = s.inst(continue_block, Opcode::LogicalNot, &[helper]);
    let cond = s.inst(continue_block, Opcode::ConditionRef, &[again]);
    s.ir.add_phi_operand(phi, entry, Value::F32(1.0));
    s.ir.add_phi_operand(phi, continue_block, next);
    s.inst(
        merge,
        Opcode::SetFragColor,
        &[Value::U32(0), Value::U32(3), Value::Inst(phi)],
    );
    s.inst(merge, Opcode::Epilogue, &[]);
    s.ir.add_branch(entry, header);
    s.ir.add_branch(header, body);
    s.ir.add_branch(body, continue_block);
    s.ir.add_branch(continue_block, header);
    s.ir.add_branch(continue_block, merge);
    let info = ShaderInfo {
        uses_is_helper_invocation: true,
        stores_frag_color: [true, false, false, false, false, false, false, false],
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![
            SyntaxNode::Block(entry),
            SyntaxNode::Block(header),
            SyntaxNode::Loop {
                body,
                continue_block,
                merge,
            },
            SyntaxNode::Block(body),
            SyntaxNode::Block(continue_block),
            SyntaxNode::Repeat {
                cond,
                loop_header: header,
                merge,
            },
            SyntaxNode::Block(merge),
            SyntaxNode::Return,
        ],
        ShaderStage::Fragment,
        info,
    );
    let module = emit(
        "loop",
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
    );
    assert_eq!(count_ops(&module, Op::LoopMerge), 1);
    let phi = module.functions[0].blocks[1].instructions[0].clone();
    assert_eq!(phi.class.opcode, Op::Phi);
    assert!(
        phi.operands.iter().all(|o| o != &Operand::IdRef(0)),
        "every phi value is patched: {phi:?}"
    );
}

#[test]
fn vertex_program_with_indexed_attributes_and_indirect_constant_buffers() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Opcode::Prologue, &[]);
    let vertex_id = s.inst(
        b,
        Opcode::GetAttributeU32,
        &[Value::Attribute(Attribute::VERTEX_ID), Value::U32(0)],
    );
    let word = s.inst(b, Opcode::GetCbufU32, &[vertex_id, Value::U32(16)]);
    s.inst(b, Opcode::GetCbufF32, &[Value::U32(1), vertex_id]);
    s.inst(b, Opcode::GetCbufU8, &[Value::U32(2), Value::U32(5)]);
    s.inst(b, Opcode::GetCbufS16, &[Value::U32(2), vertex_id]);
    s.inst(b, Opcode::GetCbufU32x2, &[Value::U32(0), Value::U32(8)]);
    let value = s.inst(b, Opcode::GetAttributeIndexed, &[word, Value::U32(0)]);
    s.inst(
        b,
        Opcode::SetAttributeIndexed,
        &[word, value, Value::U32(0)],
    );
    s.inst(
        b,
        Opcode::SetAttribute,
        &[
            Value::Attribute(Attribute::POSITION_Y),
            value,
            Value::U32(0),
        ],
    );
    s.inst(b, Opcode::Epilogue, &[]);
    let info = ShaderInfo {
        loads: varying(&[generic0().as_slice(), &[Attribute::VERTEX_ID]].concat()),
        stores: varying(&[generic0().as_slice(), &[Attribute::POSITION_Y]].concat()),
        loads_indexed_attributes: true,
        stores_indexed_attributes: true,
        uses_cbuf_indirect: true,
        used_constant_buffer_types: Type::U32 | Type::F32 | Type::U8 | Type::U16 | Type::U32X2,
        used_indirect_cbuf_types: Type::U32,
        constant_buffer_descriptors: vec![ConstantBufferDescriptor {
            index: 0,
            count: 14,
        }],
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::VertexB,
        info,
    );
    let runtime = RuntimeInfo {
        previous_stage_stores: varying(&generic0()),
        convert_depth_mode: true,
        noperspective_outputs: 1,
        ..RuntimeInfo::default()
    };
    emit("vertex_indexed", &Profile::default(), &runtime, &program);
    let aliasless = Profile {
        support_descriptor_aliasing: false,
        ..Profile::default()
    };
    emit("vertex_indexed_aliasless", &aliasless, &runtime, &program);
}

#[test]
fn tessellation_and_geometry_entry_points() {
    let stages = [
        ShaderStage::TessellationControl,
        ShaderStage::TessellationEval,
        ShaderStage::Geometry,
    ];
    for stage in stages {
        let mut s = Synthetic::new();
        let b = s.block();
        s.inst(b, Opcode::Prologue, &[]);
        let position = s.inst(
            b,
            Opcode::GetAttribute,
            &[Value::Attribute(Attribute::POSITION_X), Value::U32(0)],
        );
        s.inst(
            b,
            Opcode::SetAttribute,
            &[
                Value::Attribute(Attribute::POSITION_X),
                position,
                Value::U32(0),
            ],
        );
        match stage {
            ShaderStage::TessellationControl => {
                s.inst(
                    b,
                    Opcode::SetPatch,
                    &[Value::Patch(crate::ir::Patch::LOD_LEFT), position],
                );
                s.inst(
                    b,
                    Opcode::SetPatch,
                    &[Value::Patch(crate::ir::Patch::LOD_INTERIOR_V), position],
                );
            }
            ShaderStage::TessellationEval => {
                let u = s.inst(
                    b,
                    Opcode::GetAttribute,
                    &[
                        Value::Attribute(Attribute::TESS_EVAL_POINT_U),
                        Value::U32(0),
                    ],
                );
                s.inst(
                    b,
                    Opcode::SetAttribute,
                    &[Value::Attribute(Attribute::POSITION_Y), u, Value::U32(0)],
                );
            }
            _ => {
                s.inst(b, Opcode::EmitVertex, &[Value::U32(0)]);
                s.inst(b, Opcode::EndPrimitive, &[Value::U32(0)]);
            }
        }
        s.inst_flags(b, Opcode::Epilogue, &[], 0);
        let mut loads = varying(&[Attribute::POSITION_X]);
        if stage == ShaderStage::TessellationEval {
            loads.set(Attribute::TESS_EVAL_POINT_U, true);
        }
        let info = ShaderInfo {
            loads,
            stores: varying(&[Attribute::POSITION_X, Attribute::POSITION_Y]),
            uses_invocation_id: stage == ShaderStage::TessellationControl,
            stores_tess_level_outer: true,
            stores_tess_level_inner: true,
            ..ShaderInfo::default()
        };
        let mut program = s.finish(vec![SyntaxNode::Block(b), SyntaxNode::Return], stage, info);
        program.invocations = if stage == ShaderStage::Geometry { 1 } else { 4 };
        program.output_vertices = 3;
        program.output_topology = OutputTopology::TriangleStrip;
        let runtime = RuntimeInfo {
            fixed_state_point_size: (stage == ShaderStage::Geometry).then_some(2.0),
            ..RuntimeInfo::default()
        };
        emit(
            &format!("{stage:?}"),
            &Profile::default(),
            &runtime,
            &program,
        );
    }
}

#[test]
fn global_memory_routing_and_storage_cas_loops() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Opcode::Prologue, &[]);
    s.inst(b, Opcode::Epilogue, &[]);
    let nvn_base = 0x110;
    let storage = |offset, indirect_offset| StorageBufferDescriptor {
        cbuf_index: 0,
        cbuf_offset: offset,
        count: 1,
        is_written: true,
        indirect_offset,
        write_extent: Default::default(),
    };
    let info = ShaderInfo {
        uses_global_memory: true,
        stores_global_memory: true,
        uses_int64: true,
        uses_global_increment: true,
        uses_global_decrement: true,
        uses_atomic_f32_add: true,
        uses_atomic_f32x2_add: true,
        uses_atomic_f32x2_min: true,
        uses_atomic_f32x2_max: true,
        nvn_buffer_base: nvn_base,
        nvn_buffer_used: 0b11,
        used_constant_buffer_types: Type::U32 | Type::U32X2,
        used_storage_buffer_types: Type::U32 | Type::U32X2 | Type::U32X4,
        constant_buffer_descriptors: vec![ConstantBufferDescriptor { index: 0, count: 1 }],
        storage_buffers_descriptors: vec![
            storage(nvn_base, StorageBufferDescriptor::DIRECT),
            storage(nvn_base + 0x10, StorageBufferDescriptor::DIRECT),
            storage(nvn_base, 0x20),
        ],
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        info.clone(),
    );
    let profile = Profile {
        support_int64: true,
        min_ssbo_alignment: 4,
        ..Profile::default()
    };
    emit(
        "global_routing",
        &profile,
        &RuntimeInfo::default(),
        &program,
    );
    let native = TranslatedProgram {
        info: ShaderInfo {
            native_global_bounds: true,
            ..info
        },
        ..program
    };
    emit(
        "global_routing_native",
        &profile,
        &RuntimeInfo::default(),
        &native,
    );
}

#[test]
fn binding_decorations_match_the_resource_layout() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, Opcode::GetCbufU32, &[Value::U32(0), Value::U32(4)]);
    s.inst(b, Opcode::GetCbufF32, &[Value::U32(4), Value::U32(8)]);
    let storage = |offset| StorageBufferDescriptor {
        cbuf_index: 0,
        cbuf_offset: offset,
        count: 1,
        is_written: false,
        indirect_offset: StorageBufferDescriptor::DIRECT,
        write_extent: Default::default(),
    };
    let info = ShaderInfo {
        used_constant_buffer_types: Type::U32 | Type::F32,
        used_storage_buffer_types: Type::U32 | Type::U32X4,
        constant_buffer_descriptors: vec![
            ConstantBufferDescriptor { index: 0, count: 1 },
            ConstantBufferDescriptor { index: 3, count: 2 },
        ],
        storage_buffers_descriptors: vec![storage(0x110), storage(0x120)],
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        info,
    );
    let profile = Profile::default();
    let first_binding = 3;
    let mut bindings = Bindings {
        unified: first_binding,
        ..Bindings::default()
    };
    let code =
        emit_spirv(&profile, &RuntimeInfo::default(), &program, &mut bindings).expect("emit");
    validate("bindings", &profile, &code);
    let module = dr::load_words(&code).expect("parse");
    let mut emitted: Vec<(u32, DescriptorKind)> = Vec::new();
    for var in module
        .types_global_values
        .iter()
        .filter(|i| i.class.opcode == Op::Variable)
    {
        let kind = match var.operands[0] {
            Operand::StorageClass(StorageClass::Uniform) => DescriptorKind::UniformBuffer,
            Operand::StorageClass(StorageClass::StorageBuffer) => DescriptorKind::StorageBuffer,
            _ => continue,
        };
        let id = var.result_id.expect("variable id");
        let binding = module
            .annotations
            .iter()
            .find_map(|a| match a.operands.as_slice() {
                [
                    Operand::IdRef(target),
                    Operand::Decoration(Decoration::Binding),
                    Operand::LiteralBit32(binding),
                ] if *target == id => Some(*binding),
                _ => None,
            })
            .expect("binding decoration");
        if !emitted.contains(&(binding, kind)) {
            emitted.push((binding, kind));
        }
    }
    emitted.sort_by_key(|&(binding, _)| binding);
    let layout = ResourceLayout::for_module(&program.info, first_binding);
    let expected: Vec<(u32, DescriptorKind)> = layout.sets[0]
        .bindings
        .iter()
        .map(|b| (b.binding, b.kind))
        .collect();
    assert_eq!(emitted, expected);
    assert_eq!(bindings.unified, layout.next_binding(first_binding));
}

fn emit_single(op: Opcode, args: &[Value]) -> EmitError {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, op, args);
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo::default(),
    );
    emit_spirv(
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
        &mut Bindings::default(),
    )
    .expect_err("the opcode has no emitter")
}

#[test]
fn unsupported_and_unimplemented_opcodes_are_reported() {
    let error = emit_single(Opcode::IsbeRead, &[Value::U32(0)]);
    assert!(
        matches!(error, EmitError::UnsupportedOpcode(Opcode::IsbeRead)),
        "{error}"
    );
    let error = emit_single(Opcode::IAdd32, &[Value::U32(1), Value::U32(2)]);
    assert!(
        matches!(error, EmitError::Unimplemented(Opcode::IAdd32)),
        "{error}"
    );
}

#[test]
fn mesh_programs_are_unsupported() {
    let s = Synthetic::new();
    let mut program = s.finish(Vec::new(), ShaderStage::Compute, ShaderInfo::default());
    program.stage = ShaderStage::Mesh;
    let error = emit_spirv(
        &Profile::default(),
        &RuntimeInfo::default(),
        &program,
        &mut Bindings::default(),
    )
    .expect_err("mesh");
    assert!(matches!(
        error,
        EmitError::UnsupportedStage(ShaderStage::Mesh)
    ));
}

#[test]
fn generic_input_types_follow_the_vertex_attribute_formats() {
    let mut s = Synthetic::new();
    let b = s.block();
    for index in 0..4u16 {
        let value = s.inst(
            b,
            Opcode::GetAttribute,
            &[
                Value::Attribute(Attribute::generic(index, 1)),
                Value::U32(0),
            ],
        );
        s.inst(
            b,
            Opcode::SetAttribute,
            &[
                Value::Attribute(Attribute::generic(index, 1)),
                value,
                Value::U32(0),
            ],
        );
    }
    let attributes: Vec<Attribute> = (0..4).map(|i| Attribute::generic(i, 1)).collect();
    let info = ShaderInfo {
        loads: varying(&attributes),
        stores: varying(&attributes),
        ..ShaderInfo::default()
    };
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::VertexB,
        info,
    );
    let mut runtime = RuntimeInfo {
        previous_stage_stores: varying(&attributes),
        ..RuntimeInfo::default()
    };
    runtime.generic_input_types[0] = AttributeType::SignedInt;
    runtime.generic_input_types[1] = AttributeType::UnsignedScaled;
    runtime.generic_input_types[2] = AttributeType::SignedScaled;
    runtime.generic_input_types[3] = AttributeType::Disabled;
    emit("generic_inputs", &Profile::default(), &runtime, &program);
}
