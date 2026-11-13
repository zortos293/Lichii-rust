//! Switch 2 task and mesh programs, from the C++ `emit_spirv_mesh.cpp`.
//!
//! The programs talk to the hardware through the stage buffer (ISBE). A task program writes the
//! number of mesh workgroups at byte 4 of its output region and its payload after it; each mesh
//! workgroup reads that region (its own index at byte 0) and writes its vertices and primitives
//! to an attribute region in groups of 32, one 32-word line per attribute, plus the primitive
//! count and 8-bit vertex indices to a map region. Here the task's output region is the task
//! payload, the mesh output regions live in workgroup memory, and the epilogue (reached by every
//! invocation, see `StructureOptions::converge_exits`) hands them to Vulkan: through
//! `SPV_EXT_mesh_shader`, or under [`crate::runtime::MeshEmulation`] to storage buffers from a
//! compute shader.

use rspirv::spirv::{
    BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel, GlslStd450Op, MemorySemantics,
    Scope, SelectionControl, StorageClass, Word,
};

use crate::backend::spirv::context::{EmitContext, StorageElement};
use crate::backend::spirv::mesh_emulation::{
    MESH_EMULATION_ARGS_BASE, MESH_EMULATION_ARGS_TASKS, mesh_emulation_block_words,
};
use crate::backend::spirv::{EmitError, EmitResult};
use crate::info::MeshInfo;
use crate::ir::{Attribute, InstId, Opcode};
use crate::runtime::MeshEmulationTask;
use crate::{Profile, RuntimeInfo, ShaderStage};

/// Steps of the branchless binary search for a mesh workgroup's task: up to 2^17 tasks.
const TASK_SEARCH_STEPS: u32 = 17;

/// The output one attribute line of the stage buffer is copied to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LineTarget {
    variable: Word,
    element: Option<u32>,
    is_float: bool,
}

/// Stage buffer regions and outputs of a task or mesh program.
#[derive(Debug, Default)]
pub(crate) struct MeshState {
    payload: Word,
    payload_u32: Word,
    attributes: Word,
    map: Word,
    primitive_count: Word,
    shared_u32: Word,
    primitive_indices: Word,
    point_size: Word,
    position: Word,
    threads: u32,
    attribute_words: u32,
    map_bytes: u32,
    primitive_base: u32,
    vertex_targets: Vec<Option<LineTarget>>,
    primitive_targets: Vec<Option<LineTarget>>,
    storage_u32: Word,
    output_buffer: Word,
    task_payload_buffer: Word,
    task_offsets_buffer: Word,
    task_args_buffer: Word,
    group_index: Word,
    /// The epilogue ended the entry point with OpEmitMeshTasksEXT, so the return after it emits
    /// nothing.
    pub(crate) terminated: bool,
}

fn is_mesh_stage(stage: ShaderStage) -> bool {
    matches!(stage, ShaderStage::Task | ShaderStage::Mesh)
}

/// Rejects a task or mesh program the host cannot run, before any module is built.
pub(crate) fn check_support(
    profile: &Profile,
    runtime: &RuntimeInfo,
    stage: ShaderStage,
    mesh: &MeshInfo,
) -> EmitResult<()> {
    if !is_mesh_stage(stage) {
        return Ok(());
    }
    let is_mesh = stage == ShaderStage::Mesh;
    match runtime.mesh_emulation {
        Some(emulation) if stage == ShaderStage::Task && emulation.task.is_none() => Err(
            EmitError::Unsupported("task program emulation without task buffers"),
        ),
        Some(_) if is_mesh && mesh.primitive_vertices() != 3 => Err(EmitError::Unsupported(
            "mesh emulation of point and line topologies",
        )),
        Some(_) if is_mesh && !mesh.primitive_lines.is_empty() => Err(EmitError::Unsupported(
            "mesh emulation of per-primitive outputs",
        )),
        Some(_) => Ok(()),
        None if !profile.support_mesh_shader => Err(EmitError::UnsupportedStage(stage)),
        None if profile.supported_spirv < 0x0001_0400 => {
            Err(EmitError::Unsupported("mesh shaders below SPIR-V 1.4"))
        }
        None => Ok(()),
    }
}

fn variable(
    ctx: &mut EmitContext<'_>,
    ty: Word,
    storage_class: StorageClass,
    builtin: Option<BuiltIn>,
    name: &str,
) -> Word {
    let pointer = ctx.type_pointer(storage_class, ty);
    let id = ctx.global_variable(pointer, storage_class, None);
    if let Some(builtin) = builtin {
        ctx.decorate_builtin(id, builtin);
    }
    if !name.is_empty() {
        ctx.b.name(id, name);
    }
    ctx.add_interface(id, storage_class);
    id
}

/// Defines the stage buffer regions and, for native mesh programs, the mesh outputs
/// (the C++ `DefineMesh`).
pub(crate) fn define(ctx: &mut EmitContext<'_>) {
    if !is_mesh_stage(ctx.stage) {
        return;
    }
    let mesh = &ctx.info.mesh;
    let emulated = ctx.runtime.mesh_emulation.is_some();
    if !emulated {
        ctx.add_extension("SPV_EXT_mesh_shader");
        ctx.add_capability(Capability::MeshShadingEXT);
    }
    let (u32_t, u32x3) = (ctx.u32[1], ctx.u32[3]);
    if ctx.workgroup_id == 0 {
        ctx.workgroup_id = variable(
            ctx,
            u32x3,
            StorageClass::Input,
            Some(BuiltIn::WorkgroupId),
            "",
        );
    }
    if ctx.local_invocation_id == 0 {
        ctx.local_invocation_id = variable(
            ctx,
            u32x3,
            StorageClass::Input,
            Some(BuiltIn::LocalInvocationId),
            "",
        );
    }
    ctx.mesh.threads = ctx.translated.workgroup_size[0].max(1);
    let payload_class = if emulated {
        StorageClass::Workgroup
    } else {
        StorageClass::TaskPayloadWorkgroupEXT
    };
    let payload_type = ctx.type_array(u32_t, mesh.payload_words.max(1));
    ctx.mesh.payload = variable(ctx, payload_type, payload_class, None, "task_payload");
    ctx.mesh.payload_u32 = ctx.type_pointer(payload_class, u32_t);
    if ctx.stage == ShaderStage::Task {
        return;
    }
    let vertex_lines = mesh.vertex_lines.len() as u32;
    let primitive_lines = mesh.primitive_lines.len() as u32;
    let primitive_base = mesh.max_vertices.div_ceil(32) * vertex_lines * 32;
    let attribute_words =
        (primitive_base + mesh.max_primitives.div_ceil(32) * primitive_lines * 32).max(1);
    let map_bytes = 4 + mesh.primitive_vertices() * mesh.max_primitives;
    ctx.mesh.primitive_base = primitive_base;
    ctx.mesh.attribute_words = attribute_words;
    ctx.mesh.map_bytes = map_bytes;
    ctx.mesh.shared_u32 = ctx.type_pointer(StorageClass::Workgroup, u32_t);
    let attributes_type = ctx.type_array(u32_t, attribute_words);
    ctx.mesh.attributes = variable(
        ctx,
        attributes_type,
        StorageClass::Workgroup,
        None,
        "isbe_attributes",
    );
    let map_type = ctx.type_array(u32_t, map_bytes);
    ctx.mesh.map = variable(ctx, map_type, StorageClass::Workgroup, None, "isbe_map");
    ctx.mesh.primitive_count = variable(
        ctx,
        u32_t,
        StorageClass::Workgroup,
        None,
        "isbe_primitive_count",
    );
    if emulated {
        return;
    }
    define_outputs(ctx);
}

fn define_outputs(ctx: &mut EmitContext<'_>) {
    let mesh = &ctx.info.mesh;
    let (u32_t, f32_t, f32x4) = (ctx.u32[1], ctx.f32[1], ctx.f32[4]);
    let primitive_vertices = mesh.primitive_vertices();
    let index_type = ctx.u32[primitive_vertices as usize];
    let indices_type = ctx.type_array(index_type, mesh.max_primitives);
    let indices_builtin = match primitive_vertices {
        1 => BuiltIn::PrimitivePointIndicesEXT,
        2 => BuiltIn::PrimitiveLineIndicesEXT,
        _ => BuiltIn::PrimitiveTriangleIndicesEXT,
    };
    ctx.mesh.primitive_indices = variable(
        ctx,
        indices_type,
        StorageClass::Output,
        Some(indices_builtin),
        "",
    );
    let per_vertex_f32 = ctx.type_array(f32_t, mesh.max_vertices);
    let per_vertex_f32x4 = ctx.type_array(f32x4, mesh.max_vertices);
    let per_primitive_f32x4 = ctx.type_array(f32x4, mesh.max_primitives);
    let per_primitive_u32 = ctx.type_array(u32_t, mesh.max_primitives);
    if primitive_vertices == 1 {
        ctx.mesh.point_size = variable(
            ctx,
            per_vertex_f32,
            StorageClass::Output,
            Some(BuiltIn::PointSize),
            "",
        );
    }
    let mut generics = [[0; Attribute::NUM_GENERICS as usize]; 2];
    let mut primitive_builtins = [0; 3];
    let mut target = |ctx: &mut EmitContext<'_>, line: u32, per_primitive: bool| {
        let attribute = Attribute(line as u16);
        if let Some(index) = attribute.generic_index() {
            let slot = &mut generics[per_primitive as usize][index as usize];
            if *slot == 0 {
                let ty = if per_primitive {
                    per_primitive_f32x4
                } else {
                    per_vertex_f32x4
                };
                *slot = variable(ctx, ty, StorageClass::Output, None, "");
                ctx.decorate(*slot, Decoration::Location, &[u32::from(index)]);
                if per_primitive {
                    ctx.decorate(*slot, Decoration::PerPrimitiveEXT, &[]);
                }
            }
            return Some(LineTarget {
                variable: *slot,
                element: Some(u32::from(attribute.component())),
                is_float: true,
            });
        }
        if per_primitive {
            let (slot, builtin) = match attribute {
                Attribute::PRIMITIVE_ID => (0, BuiltIn::PrimitiveId),
                Attribute::LAYER => (1, BuiltIn::Layer),
                Attribute::VIEWPORT_INDEX => (2, BuiltIn::ViewportIndex),
                _ => return None,
            };
            if primitive_builtins[slot] == 0 {
                let id = variable(
                    ctx,
                    per_primitive_u32,
                    StorageClass::Output,
                    Some(builtin),
                    "",
                );
                ctx.decorate(id, Decoration::PerPrimitiveEXT, &[]);
                primitive_builtins[slot] = id;
            }
            return Some(LineTarget {
                variable: primitive_builtins[slot],
                element: None,
                is_float: false,
            });
        }
        match attribute {
            Attribute::POSITION_X
            | Attribute::POSITION_Y
            | Attribute::POSITION_Z
            | Attribute::POSITION_W => {
                if ctx.mesh.position == 0 {
                    ctx.mesh.position = variable(
                        ctx,
                        per_vertex_f32x4,
                        StorageClass::Output,
                        Some(BuiltIn::Position),
                        "",
                    );
                }
                Some(LineTarget {
                    variable: ctx.mesh.position,
                    element: Some(u32::from(attribute.0 - Attribute::POSITION_X.0)),
                    is_float: true,
                })
            }
            Attribute::POINT_SIZE => {
                if ctx.mesh.point_size == 0 {
                    ctx.mesh.point_size = variable(
                        ctx,
                        per_vertex_f32,
                        StorageClass::Output,
                        Some(BuiltIn::PointSize),
                        "",
                    );
                }
                Some(LineTarget {
                    variable: ctx.mesh.point_size,
                    element: None,
                    is_float: true,
                })
            }
            _ => None,
        }
    };
    let vertex_targets = mesh
        .vertex_lines
        .iter()
        .map(|&line| target(ctx, line, false))
        .collect();
    let primitive_targets = mesh
        .primitive_lines
        .iter()
        .map(|&line| target(ctx, line, true))
        .collect();
    ctx.mesh.vertex_targets = vertex_targets;
    ctx.mesh.primitive_targets = primitive_targets;
}

/// Defines the storage buffers of the mesh emulation, after the program's own storage buffers
/// so their u32 block type is shared (the C++ `DefineMeshEmulationOutput`).
pub(crate) fn define_emulation_output(ctx: &mut EmitContext<'_>) {
    let Some(emulation) = ctx.runtime.mesh_emulation else {
        return;
    };
    if !is_mesh_stage(ctx.stage) {
        return;
    }
    ctx.add_extension("SPV_KHR_storage_buffer_storage_class");
    let u32_t = ctx.u32[1];
    let mut block_pointer = ctx.storage_type(StorageElement::U32).array;
    if block_pointer == 0 {
        let array = ctx.b.type_runtime_array(u32_t);
        ctx.decorate(array, Decoration::ArrayStride, &[4]);
        let block = ctx.b.type_struct([array]);
        ctx.decorate(block, Decoration::Block, &[]);
        ctx.member_decorate(block, 0, Decoration::Offset, &[0]);
        block_pointer = ctx.type_pointer(StorageClass::StorageBuffer, block);
    }
    ctx.mesh.storage_u32 = ctx.type_pointer(StorageClass::StorageBuffer, u32_t);
    let define = |ctx: &mut EmitContext<'_>, binding: u32, name: &str| {
        let id = ctx.global_variable(block_pointer, StorageClass::StorageBuffer, None);
        ctx.decorate(id, Decoration::Binding, &[binding]);
        ctx.decorate(id, Decoration::DescriptorSet, &[0]);
        ctx.b.name(id, name);
        ctx.add_interface(id, StorageClass::StorageBuffer);
        id
    };
    if let Some(task) = emulation.task {
        ctx.mesh.task_payload_buffer =
            define(ctx, task.payload_binding, "mesh_emulation_task_payload");
    }
    if ctx.stage == ShaderStage::Task {
        return;
    }
    ctx.mesh.output_buffer = define(ctx, emulation.output_binding, "mesh_emulation_output");
    if let Some(task) = emulation.task {
        ctx.mesh.task_offsets_buffer =
            define(ctx, task.offsets_binding, "mesh_emulation_task_offsets");
        ctx.mesh.task_args_buffer = define(ctx, task.args_binding, "mesh_emulation_task_args");
        ctx.mesh.group_index =
            variable(ctx, u32_t, StorageClass::Private, None, "mesh_group_index");
    }
}

/// Adds the execution modes of a task or mesh entry point and returns its execution model.
pub(crate) fn define_entry_point(ctx: &mut EmitContext<'_>, main: Word) -> ExecutionModel {
    let threads = ctx.mesh.threads;
    ctx.b
        .execution_mode(main, ExecutionMode::LocalSize, [threads, 1, 1]);
    if ctx.runtime.mesh_emulation.is_some() {
        return ExecutionModel::GLCompute;
    }
    if ctx.stage == ShaderStage::Task {
        return ExecutionModel::TaskEXT;
    }
    let mesh = &ctx.info.mesh;
    let topology = match mesh.primitive_vertices() {
        1 => ExecutionMode::OutputPoints,
        2 => ExecutionMode::OutputLinesEXT,
        _ => ExecutionMode::OutputTrianglesEXT,
    };
    ctx.b
        .execution_mode(main, ExecutionMode::OutputVertices, [mesh.max_vertices]);
    ctx.b.execution_mode(
        main,
        ExecutionMode::OutputPrimitivesEXT,
        [mesh.max_primitives],
    );
    ctx.b.execution_mode(main, topology, []);
    ExecutionModel::MeshEXT
}

fn barrier(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let workgroup = ctx.const_u32(Scope::Workgroup as u32);
    let semantics = MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY;
    let semantics = ctx.const_u32(semantics.bits());
    Ok(ctx.b.control_barrier(workgroup, workgroup, semantics)?)
}

fn first_component(ctx: &mut EmitContext<'_>, vector: Word) -> EmitResult<Word> {
    let (u32_t, u32x3) = (ctx.u32[1], ctx.u32[3]);
    let value = ctx.b.load(u32x3, None, vector, None, [])?;
    Ok(ctx.b.composite_extract(u32_t, None, value, [0])?)
}

fn local_invocation(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    first_component(ctx, ctx.local_invocation_id)
}

fn workgroup_x(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    first_component(ctx, ctx.workgroup_id)
}

fn add(ctx: &mut EmitContext<'_>, a: Word, b: Word) -> EmitResult<Word> {
    Ok(ctx.b.i_add(ctx.u32[1], None, a, b)?)
}

fn add_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    add(ctx, a, b)
}

fn mul_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    Ok(ctx.b.i_mul(ctx.u32[1], None, a, b)?)
}

fn shift_right_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    Ok(ctx.b.shift_right_logical(ctx.u32[1], None, a, b)?)
}

fn umin(ctx: &mut EmitContext<'_>, a: Word, b: Word) -> EmitResult<Word> {
    ctx.glsl_inst(ctx.u32[1], GlslStd450Op::UMin, &[a, b])
}

fn umin_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    umin(ctx, a, b)
}

fn less_than_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    Ok(ctx.b.u_less_than(ctx.u1, None, a, b)?)
}

fn load_u32(ctx: &mut EmitContext<'_>, pointer: Word) -> EmitResult<Word> {
    Ok(ctx.b.load(ctx.u32[1], None, pointer, None, [])?)
}

fn store(ctx: &mut EmitContext<'_>, pointer: Word, value: Word) -> EmitResult<()> {
    Ok(ctx.b.store(pointer, value, None, [])?)
}

/// Clamps `index` to the last of `count` (at least one) words.
fn clamped(ctx: &mut EmitContext<'_>, index: Word, count: u32) -> EmitResult<Word> {
    umin_imm(ctx, index, count.max(1) - 1)
}

fn payload_word(ctx: &mut EmitContext<'_>, index: Word) -> EmitResult<Word> {
    let index = clamped(ctx, index, ctx.info.mesh.payload_words)?;
    Ok(ctx
        .b
        .access_chain(ctx.mesh.payload_u32, None, ctx.mesh.payload, [index])?)
}

fn shared_word(ctx: &mut EmitContext<'_>, array: Word, index: Word) -> EmitResult<Word> {
    Ok(ctx
        .b
        .access_chain(ctx.mesh.shared_u32, None, array, [index])?)
}

fn buffer_word(ctx: &mut EmitContext<'_>, buffer: Word, index: Word) -> EmitResult<Word> {
    Ok(ctx.b.access_chain(
        ctx.mesh.storage_u32,
        None,
        buffer,
        [ctx.u32_zero_value, index],
    )?)
}

/// Word of line 0 of element `index` (a vertex or a primitive) in the attribute region:
/// `base + (index / 32) * num_lines * 32 + index % 32`.
fn first_line_word(
    ctx: &mut EmitContext<'_>,
    index: Word,
    base: u32,
    num_lines: u32,
) -> EmitResult<Word> {
    let group = shift_right_imm(ctx, index, 5)?;
    let thirty_one = ctx.const_u32(31);
    let slot = ctx.b.bitwise_and(ctx.u32[1], None, index, thirty_one)?;
    let group_base = mul_imm(ctx, group, num_lines * 32)?;
    let slot = add_imm(ctx, slot, base)?;
    add(ctx, group_base, slot)
}

fn load_line(ctx: &mut EmitContext<'_>, first: Word, line: u32) -> EmitResult<Word> {
    let index = add_imm(ctx, first, line * 32)?;
    let pointer = shared_word(ctx, ctx.mesh.attributes, index)?;
    load_u32(ctx, pointer)
}

/// Runs `body` under `if (condition)`.
fn if_then(
    ctx: &mut EmitContext<'_>,
    condition: Word,
    body: impl FnOnce(&mut EmitContext<'_>) -> EmitResult<()>,
) -> EmitResult<()> {
    let then_label = ctx.b.id();
    let merge_label = ctx.b.id();
    ctx.b.selection_merge(merge_label, SelectionControl::NONE)?;
    ctx.b
        .branch_conditional(condition, then_label, merge_label, [])?;
    ctx.b.begin_block(Some(then_label))?;
    body(ctx)?;
    ctx.b.branch(merge_label)?;
    ctx.b.begin_block(Some(merge_label))?;
    Ok(())
}

/// Runs `body` for elements `0..count`, a workgroup's worth at a time, each invocation
/// taking the element at its local index; a last partial round is guarded.
fn for_each_element(
    ctx: &mut EmitContext<'_>,
    invocation: Word,
    count: u32,
    mut body: impl FnMut(&mut EmitContext<'_>, Word) -> EmitResult<()>,
) -> EmitResult<()> {
    let threads = ctx.mesh.threads;
    for first in (0..count).step_by(threads as usize) {
        let element = add_imm(ctx, invocation, first)?;
        if first + threads > count {
            let inside = less_than_imm(ctx, element, count)?;
            if_then(ctx, inside, |ctx| body(ctx, element))?;
        } else {
            body(ctx, element)?;
        }
    }
    Ok(())
}

/// Copies the lines of element `index` from the attribute region to their outputs.
fn copy_lines(
    ctx: &mut EmitContext<'_>,
    index: Word,
    base: u32,
    per_primitive: bool,
    convert_depth: bool,
) -> EmitResult<()> {
    let targets = if per_primitive {
        ctx.mesh.primitive_targets.clone()
    } else {
        ctx.mesh.vertex_targets.clone()
    };
    let first = first_line_word(ctx, index, base, targets.len() as u32)?;
    let mut vectors: Vec<(Word, [Option<Word>; 4])> = Vec::new();
    for (line, target) in targets.iter().enumerate() {
        let Some(target) = *target else {
            continue;
        };
        let word = load_line(ctx, first, line as u32)?;
        let value = if target.is_float {
            ctx.b.bitcast(ctx.f32[1], None, word)?
        } else {
            word
        };
        let Some(element) = target.element else {
            let pointer_type = if target.is_float {
                ctx.output_f32
            } else {
                ctx.output_u32
            };
            let pointer = ctx
                .b
                .access_chain(pointer_type, None, target.variable, [index])?;
            store(ctx, pointer, value)?;
            continue;
        };
        let at = match vectors.iter().position(|(v, _)| *v == target.variable) {
            Some(at) => at,
            None => {
                vectors.push((target.variable, [None; 4]));
                vectors.len() - 1
            }
        };
        vectors[at].1[element as usize] = Some(value);
    }
    let f32x4 = ctx.f32[4];
    let pointer_type = ctx.type_pointer(StorageClass::Output, f32x4);
    for (variable, mut components) in vectors {
        if convert_depth
            && variable == ctx.mesh.position
            && let [_, _, Some(z), Some(w)] = components
        {
            let sum = ctx.b.f_add(ctx.f32[1], None, z, w)?;
            let half = ctx.const_f32(0.5);
            components[2] = Some(ctx.b.f_mul(ctx.f32[1], None, sum, half)?);
        }
        let pointer = ctx.b.access_chain(pointer_type, None, variable, [index])?;
        let (zero, one) = (ctx.const_f32(0.0), ctx.const_f32(1.0));
        let defaults = [zero, zero, zero, one];
        let value = ctx.b.composite_construct(
            f32x4,
            None,
            [0, 1, 2, 3].map(|i| {
                components[i].unwrap_or_else(|| {
                    crate::fallback::note("mesh", "unwritten_component_default");
                    defaults[i]
                })
            }),
        )?;
        store(ctx, pointer, value)?;
    }
    Ok(())
}

/// Hands the stage buffer to Vulkan: a task program's mesh workgroup count and payload, a
/// mesh program's vertices and primitives.
pub(crate) fn epilogue(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    match (ctx.stage, ctx.runtime.mesh_emulation) {
        (ShaderStage::Task, None) => task_epilogue(ctx),
        (ShaderStage::Task, Some(emulation)) => match emulation.task {
            Some(task) => emulated_task_epilogue(ctx, task),
            None => Err(EmitError::Unsupported(
                "task program emulation without task buffers",
            )),
        },
        (ShaderStage::Mesh, None) => mesh_epilogue(ctx),
        (ShaderStage::Mesh, Some(_)) => emulated_mesh_epilogue(ctx),
        _ => Ok(()),
    }
}

fn task_epilogue(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    barrier(ctx)?;
    let one = ctx.const_u32(1);
    let pointer = payload_word(ctx, one)?;
    let count = load_u32(ctx, pointer)?;
    ctx.b
        .emit_mesh_tasks_ext(count, one, one, Some(ctx.mesh.payload))?;
    ctx.mesh.terminated = true;
    Ok(())
}

fn mesh_epilogue(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let mesh = &ctx.info.mesh;
    barrier(ctx)?;
    let stored = load_u32(ctx, ctx.mesh.primitive_count)?;
    let count = umin_imm(ctx, stored, mesh.max_primitives)?;
    let max_vertices = ctx.const_u32(mesh.max_vertices);
    ctx.b.set_mesh_outputs_ext(max_vertices, count)?;

    let invocation = local_invocation(ctx)?;
    let convert =
        ctx.runtime.convert_depth_mode && !ctx.profile.support_native_ndc && ctx.mesh.position != 0;
    let points = mesh.primitive_vertices() == 1;
    let point_size = ctx.const_f32(ctx.runtime.fixed_state_point_size.unwrap_or(1.0));
    for_each_element(ctx, invocation, mesh.max_vertices, |ctx, vertex| {
        if points {
            let pointer =
                ctx.b
                    .access_chain(ctx.output_f32, None, ctx.mesh.point_size, [vertex])?;
            store(ctx, pointer, point_size)?;
        }
        copy_lines(ctx, vertex, 0, false, convert)
    })?;

    let primitive_vertices = mesh.primitive_vertices();
    let index_type = ctx.u32[primitive_vertices as usize];
    let index_pointer = ctx.type_pointer(StorageClass::Output, index_type);
    let threads = ctx.mesh.threads;
    for first in (0..mesh.max_primitives).step_by(threads as usize) {
        let primitive = add_imm(ctx, invocation, first)?;
        let used = ctx.b.u_less_than(ctx.u1, None, primitive, count)?;
        if_then(ctx, used, |ctx| {
            let scaled = mul_imm(ctx, primitive, primitive_vertices)?;
            let byte = add_imm(ctx, scaled, 4)?;
            let mut indices = [0; 3];
            for (element, index) in indices
                .iter_mut()
                .enumerate()
                .take(primitive_vertices as usize)
            {
                let address = add_imm(ctx, byte, element as u32)?;
                let pointer = shared_word(ctx, ctx.mesh.map, address)?;
                *index = load_u32(ctx, pointer)?;
            }
            let value = if primitive_vertices == 1 {
                indices[0]
            } else {
                ctx.b.composite_construct(
                    index_type,
                    None,
                    indices[..primitive_vertices as usize].iter().copied(),
                )?
            };
            let pointer =
                ctx.b
                    .access_chain(index_pointer, None, ctx.mesh.primitive_indices, [primitive])?;
            store(ctx, pointer, value)?;
            copy_lines(ctx, primitive, ctx.mesh.primitive_base, true, false)
        })?;
    }
    Ok(())
}

/// Mesh emulation: the task workgroup's payload (its mesh workgroup count at word 1) goes to its
/// block of the task payload buffer, for the distribution pass and the mesh workgroups.
fn emulated_task_epilogue(ctx: &mut EmitContext<'_>, task: MeshEmulationTask) -> EmitResult<()> {
    barrier(ctx)?;
    let stride = task.payload_stride;
    let words = ctx.info.mesh.payload_words.min(stride);
    let workgroup = workgroup_x(ctx)?;
    let block = mul_imm(ctx, workgroup, stride)?;
    let invocation = local_invocation(ctx)?;
    for_each_element(ctx, invocation, words, |ctx, word| {
        let source = payload_word(ctx, word)?;
        let value = load_u32(ctx, source)?;
        let index = add(ctx, block, word)?;
        let destination = buffer_word(ctx, ctx.mesh.task_payload_buffer, index)?;
        store(ctx, destination, value)
    })
}

/// Mesh emulation: writes this workgroup's block of the output buffer, laid out as
/// [`mesh_emulation_block_words`] describes, instead of the mesh outputs.
fn emulated_mesh_epilogue(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let mesh = &ctx.info.mesh;
    barrier(ctx)?;
    let vertex_lines = mesh.vertex_lines.len() as u32;
    let vertex_words = mesh.max_vertices * vertex_lines;
    let workgroup = workgroup_x(ctx)?;
    let mut block = mul_imm(ctx, workgroup, mesh_emulation_block_words(mesh))?;
    if ctx.mesh.task_args_buffer != 0 {
        let base = ctx.const_u32(MESH_EMULATION_ARGS_BASE);
        let pointer = buffer_word(ctx, ctx.mesh.task_args_buffer, base)?;
        let base = load_u32(ctx, pointer)?;
        block = add(ctx, block, base)?;
    }
    let stored = load_u32(ctx, ctx.mesh.primitive_count)?;
    let count = umin_imm(ctx, stored, mesh.max_primitives)?;
    let invocation = local_invocation(ctx)?;

    for_each_element(ctx, invocation, mesh.max_vertices, |ctx, vertex| {
        let source = first_line_word(ctx, vertex, 0, vertex_lines)?;
        let offset = mul_imm(ctx, vertex, vertex_lines)?;
        let destination = add(ctx, block, offset)?;
        for line in 0..vertex_lines {
            let word = load_line(ctx, source, line)?;
            let index = add_imm(ctx, destination, line)?;
            let pointer = buffer_word(ctx, ctx.mesh.output_buffer, index)?;
            store(ctx, pointer, word)?;
        }
        Ok(())
    })?;

    let index_base = add_imm(ctx, block, vertex_words)?;
    let map_bytes = ctx.mesh.map_bytes;
    for_each_element(ctx, invocation, mesh.max_primitives, |ctx, primitive| {
        let used = ctx.b.u_less_than(ctx.u1, None, primitive, count)?;
        let scaled = mul_imm(ctx, primitive, 3)?;
        let byte = add_imm(ctx, scaled, 4)?;
        let destination = add(ctx, index_base, scaled)?;
        for element in 0..3 {
            let address = add_imm(ctx, byte, element)?;
            let address = clamped(ctx, address, map_bytes)?;
            let pointer = shared_word(ctx, ctx.mesh.map, address)?;
            let value = load_u32(ctx, pointer)?;
            let value = ctx
                .b
                .select(ctx.u32[1], None, used, value, ctx.u32_zero_value)?;
            let index = add_imm(ctx, destination, element)?;
            let pointer = buffer_word(ctx, ctx.mesh.output_buffer, index)?;
            store(ctx, pointer, value)?;
        }
        Ok(())
    })
}

/// Initialises the stage buffer: the mesh workgroup count of a task program, the primitive
/// count of a mesh program and, under the emulation, the input region and vertex positions.
/// It emits no branches, since the entry block's label is a phi predecessor for later blocks.
pub(crate) fn prologue(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let emulation = ctx.runtime.mesh_emulation;
    match ctx.stage {
        ShaderStage::Task if emulation.is_none() => {
            let one = ctx.const_u32(1);
            let pointer = payload_word(ctx, one)?;
            store(ctx, pointer, ctx.u32_zero_value)?;
            return barrier(ctx);
        }
        ShaderStage::Task => {}
        ShaderStage::Mesh => store(ctx, ctx.mesh.primitive_count, ctx.u32_zero_value)?,
        _ => return Ok(()),
    }
    let Some(emulation) = emulation else {
        return barrier(ctx);
    };
    let invocation = local_invocation(ctx)?;
    match emulation.task {
        Some(task) if ctx.stage == ShaderStage::Mesh => copy_task_payload(ctx, invocation, task)?,
        _ => zero_payload(ctx, invocation)?,
    }
    if ctx.stage == ShaderStage::Mesh {
        clear_positions(ctx, invocation)?;
    }
    barrier(ctx)
}

/// Plain workgroup memory, zeroed so words nobody writes read as zero. Indices past the region
/// are clamped to its last word, which is zeroed as well.
fn zero_payload(ctx: &mut EmitContext<'_>, invocation: Word) -> EmitResult<()> {
    let threads = ctx.mesh.threads;
    for first in (0..ctx.info.mesh.payload_words).step_by(threads as usize) {
        let word = add_imm(ctx, invocation, first)?;
        let pointer = payload_word(ctx, word)?;
        store(ctx, pointer, ctx.u32_zero_value)?;
    }
    Ok(())
}

/// Mesh emulation with a task: finds this workgroup's task and its index within the task's mesh
/// workgroups, and copies the task's payload into the input region. The task is the last one
/// whose first mesh workgroup is not past this one, found by a branchless binary search of the
/// offsets (tasks with no mesh workgroups share the next task's offset and lose to it). Words
/// past the payload are clamped to its last one, so every store writes a real word, and words
/// past the task's stride read zero.
fn copy_task_payload(
    ctx: &mut EmitContext<'_>,
    invocation: Word,
    task_buffers: MeshEmulationTask,
) -> EmitResult<()> {
    let (u1, u32_t) = (ctx.u1, ctx.u32[1]);
    let global = workgroup_x(ctx)?;
    let tasks_index = ctx.const_u32(MESH_EMULATION_ARGS_TASKS);
    let pointer = buffer_word(ctx, ctx.mesh.task_args_buffer, tasks_index)?;
    let tasks = load_u32(ctx, pointer)?;
    let one = ctx.const_u32(1);
    let at_least_one = ctx.glsl_inst(u32_t, GlslStd450Op::UMax, &[tasks, one])?;
    let last_task = ctx.b.i_sub(u32_t, None, at_least_one, one)?;
    let mut task = ctx.u32_zero_value;
    let mut remaining = tasks;
    for _ in 0..TASK_SEARCH_STEPS {
        let half = shift_right_imm(ctx, remaining, 1)?;
        let middle = add(ctx, task, half)?;
        let middle = umin(ctx, middle, last_task)?;
        let pointer = buffer_word(ctx, ctx.mesh.task_offsets_buffer, middle)?;
        let first = load_u32(ctx, pointer)?;
        let not_past = ctx.b.u_less_than_equal(u1, None, first, global)?;
        task = ctx.b.select(u32_t, None, not_past, middle, task)?;
        remaining = ctx.b.i_sub(u32_t, None, remaining, half)?;
    }
    let pointer = buffer_word(ctx, ctx.mesh.task_offsets_buffer, task)?;
    let first_group = load_u32(ctx, pointer)?;
    let group_index = ctx.b.i_sub(u32_t, None, global, first_group)?;
    store(ctx, ctx.mesh.group_index, group_index)?;
    let stride = task_buffers.payload_stride.max(1);
    let block = mul_imm(ctx, task, stride)?;
    let words = ctx.info.mesh.payload_words;
    let threads = ctx.mesh.threads;
    for first in (0..words).step_by(threads as usize) {
        let word = add_imm(ctx, invocation, first)?;
        let word = umin_imm(ctx, word, words - 1)?;
        let source = umin_imm(ctx, word, stride - 1)?;
        let source = add(ctx, block, source)?;
        let pointer = buffer_word(ctx, ctx.mesh.task_payload_buffer, source)?;
        let value = load_u32(ctx, pointer)?;
        let inside = less_than_imm(ctx, word, stride)?;
        let value = ctx
            .b
            .select(u32_t, None, inside, value, ctx.u32_zero_value)?;
        let pointer = payload_word(ctx, word)?;
        store(ctx, pointer, value)?;
    }
    Ok(())
}

/// Workgroup memory keeps whatever the previous workgroup left, and a primitive that uses a
/// vertex the program never wrote would draw a stray triangle, so every vertex position starts
/// as NaN, which the rasterizer drops. Past the last vertex the index is clamped to it.
fn clear_positions(ctx: &mut EmitContext<'_>, invocation: Word) -> EmitResult<()> {
    let mesh = &ctx.info.mesh;
    let num_lines = mesh.vertex_lines.len() as u32;
    let positions = u32::from(Attribute::POSITION_X.0)..=u32::from(Attribute::POSITION_W.0);
    let nan = ctx.const_u32(0x7fc0_0000);
    let threads = ctx.mesh.threads;
    for (line, &attribute) in mesh.vertex_lines.iter().enumerate() {
        if !positions.contains(&attribute) {
            continue;
        }
        crate::fallback::note("mesh", "unwritten_position_nan");
        for first in (0..mesh.max_vertices).step_by(threads as usize) {
            let vertex = add_imm(ctx, invocation, first)?;
            let vertex = umin_imm(ctx, vertex, mesh.max_vertices - 1)?;
            let index = first_line_word(ctx, vertex, line as u32 * 32, num_lines)?;
            let pointer = shared_word(ctx, ctx.mesh.attributes, index)?;
            store(ctx, pointer, nan)?;
        }
    }
    Ok(())
}

/// A task program's input region holds its workgroup index; a mesh program's holds its own
/// index at byte 0 and the task's payload after it.
pub(super) fn isbe_read(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if !is_mesh_stage(ctx.stage) {
        return Err(EmitError::UnsupportedOpcode(Opcode::IsbeRead));
    }
    let workgroup = workgroup_x(ctx)?;
    if ctx.stage == ShaderStage::Task {
        return Ok(workgroup);
    }
    let own_index = if ctx.mesh.group_index != 0 {
        load_u32(ctx, ctx.mesh.group_index)?
    } else {
        workgroup
    };
    let address = ctx.def_arg(inst, 0)?;
    let index = shift_right_imm(ctx, address, 2)?;
    let pointer = payload_word(ctx, index)?;
    let word = load_u32(ctx, pointer)?;
    let is_own_index = ctx.b.i_equal(ctx.u1, None, address, ctx.u32_zero_value)?;
    Ok(ctx
        .b
        .select(ctx.u32[1], None, is_own_index, own_index, word)?)
}

pub(super) fn isbe_write_attribute(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    if !is_mesh_stage(ctx.stage) {
        return Err(EmitError::UnsupportedOpcode(Opcode::IsbeWriteAttribute));
    }
    let address = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let index = shift_right_imm(ctx, address, 2)?;
    let pointer = if ctx.stage == ShaderStage::Task {
        payload_word(ctx, index)?
    } else {
        let index = clamped(ctx, index, ctx.mesh.attribute_words)?;
        shared_word(ctx, ctx.mesh.attributes, index)?
    };
    store(ctx, pointer, value)
}

pub(super) fn isbe_write_map_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    if ctx.stage != ShaderStage::Mesh {
        return Err(EmitError::UnsupportedOpcode(Opcode::IsbeWriteMapU8));
    }
    let address = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let index = clamped(ctx, address, ctx.mesh.map_bytes)?;
    let pointer = shared_word(ctx, ctx.mesh.map, index)?;
    let mask = ctx.const_u32(0xff);
    let byte = ctx.b.bitwise_and(ctx.u32[1], None, value, mask)?;
    store(ctx, pointer, byte)
}

/// The primitive count: the map is written from its end, so its first word is at byte 3.
pub(super) fn isbe_write_map_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    if ctx.stage != ShaderStage::Mesh || ctx.arg(inst, 0).imm_u32() != Some(3) {
        return Err(EmitError::UnsupportedOpcode(Opcode::IsbeWriteMapU32));
    }
    let value = ctx.def_arg(inst, 1)?;
    store(ctx, ctx.mesh.primitive_count, value)
}
