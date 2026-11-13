//! Mesh emulation for hosts without `VK_EXT_mesh_shader`, from the C++ `emit_spirv_mesh.cpp`.
//!
//! Under [`crate::runtime::MeshEmulation`] a mesh program runs as a compute shader writing one
//! block per workgroup to a u32 storage buffer, and the modules here draw it: a vertex shader that
//! reads the blocks as a triangle list and, with a task program, a one-invocation distribution
//! pass between the task and mesh dispatches that turns each task's mesh workgroup count into
//! dispatch and draw arguments.

use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, GlslStd450Op, LoopControl, MemoryModel, StorageClass, Word,
};
use rustc_hash::FxHashMap;

use super::EmitResult;
use crate::Profile;
use crate::info::MeshInfo;
use crate::ir::Attribute;

/// Word of the distribution's argument buffer holding the mesh workgroups the tasks asked for;
/// a `vkCmdDispatchIndirect` command sits at word 0.
pub const MESH_EMULATION_ARGS_REQUESTED: u32 = 3;
/// Byte offset of the `vkCmdDrawIndirect` command in the argument buffer.
pub const MESH_EMULATION_DRAW_ARGS_OFFSET: u32 = 16;
/// Word of the argument buffer holding the draw's first output word in the arena.
pub const MESH_EMULATION_ARGS_BASE: u32 = 8;
/// Word of the argument buffer holding the draw's task workgroup count.
pub const MESH_EMULATION_ARGS_TASKS: u32 = 9;
/// Size of the argument buffer in words.
pub const MESH_EMULATION_ARGS_WORDS: u32 = 10;

/// Words of one mesh workgroup's block in the emulation's output buffer: `max_vertices *
/// vertex_lines` words (vertex `v`, line `l` at `v * vertex_lines + l`), then `max_primitives * 3`
/// vertex indices (0 0 0, a degenerate triangle, past the primitive count).
pub fn mesh_emulation_block_words(mesh: &MeshInfo) -> u32 {
    mesh.max_vertices * mesh.vertex_lines.len() as u32 + mesh.max_primitives * 3
}

/// Push constants of the distribution pass, in member order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct MeshEmulationDistributionConstants {
    /// Task workgroups of the draw.
    pub tasks: u32,
    /// Words per task in the payload buffer (its mesh workgroup count at word 1).
    pub payload_words: u32,
    /// Words of the output arena.
    pub capacity_words: u32,
    /// Output words per mesh workgroup, [`mesh_emulation_block_words`].
    pub block_words: u32,
    /// Draw vertices per mesh workgroup (`max_primitives * 3`).
    pub vertices_per_group: u32,
    /// The device's `maxComputeWorkGroupCount[0]`.
    pub max_groups: u32,
}

impl MeshEmulationDistributionConstants {
    const WORDS: u32 = 6;

    /// The constants as push constant words.
    pub fn words(&self) -> [u32; Self::WORDS as usize] {
        [
            self.tasks,
            self.payload_words,
            self.capacity_words,
            self.block_words,
            self.vertices_per_group,
            self.max_groups,
        ]
    }
}

/// A standalone module under construction with its u32 constants cached.
struct Module {
    b: Builder,
    void_type: Word,
    bool_type: Word,
    u32_type: Word,
    constants: FxHashMap<u32, Word>,
    interfaces: Vec<Word>,
    spirv_version: u32,
}

impl Module {
    fn new(profile: &Profile) -> Module {
        let mut b = Builder::new();
        b.set_version(
            ((profile.supported_spirv >> 16) & 0xff) as u8,
            ((profile.supported_spirv >> 8) & 0xff) as u8,
        );
        b.capability(Capability::Shader);
        b.extension("SPV_KHR_storage_buffer_storage_class");
        b.memory_model(AddressingModel::Logical, MemoryModel::GLSL450);
        let void_type = b.type_void();
        let bool_type = b.type_bool();
        let u32_type = b.type_int(32, 0);
        Module {
            b,
            void_type,
            bool_type,
            u32_type,
            constants: FxHashMap::default(),
            interfaces: Vec::new(),
            spirv_version: profile.supported_spirv,
        }
    }

    fn constant(&mut self, value: u32) -> Word {
        if let Some(&id) = self.constants.get(&value) {
            return id;
        }
        let id = self.b.constant_bit32(self.u32_type, value);
        self.constants.insert(value, id);
        id
    }

    fn decorate(&mut self, target: Word, decoration: Decoration, literal: u32) {
        self.b
            .decorate(target, decoration, [Operand::LiteralBit32(literal)]);
    }

    /// Pointer types to the u32 storage buffer block and to one of its words.
    fn storage_block(&mut self, writable: bool) -> (Word, Word) {
        let array = self.b.type_runtime_array(self.u32_type);
        self.decorate(array, Decoration::ArrayStride, 4);
        let block = self.b.type_struct([array]);
        self.b.decorate(block, Decoration::Block, []);
        self.b
            .member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
        if !writable {
            self.b
                .member_decorate(block, 0, Decoration::NonWritable, []);
        }
        let pointer = self
            .b
            .type_pointer(None, StorageClass::StorageBuffer, block);
        let word = self
            .b
            .type_pointer(None, StorageClass::StorageBuffer, self.u32_type);
        (pointer, word)
    }

    /// A global variable, listed on the entry point where the SPIR-V version requires it.
    fn global(&mut self, pointer: Word, storage_class: StorageClass) -> Word {
        let id = self.b.variable(pointer, None, storage_class, None);
        if matches!(storage_class, StorageClass::Input | StorageClass::Output)
            || self.spirv_version >= 0x0001_0400
        {
            self.interfaces.push(id);
        }
        id
    }

    fn buffer(&mut self, pointer: Word, binding: u32, name: &str) -> Word {
        let id = self.global(pointer, StorageClass::StorageBuffer);
        self.decorate(id, Decoration::Binding, binding);
        self.decorate(id, Decoration::DescriptorSet, 0);
        self.b.name(id, name);
        id
    }

    fn begin_main(&mut self) -> EmitResult<Word> {
        let function_type = self.b.type_function(self.void_type, []);
        let main =
            self.b
                .begin_function(self.void_type, None, FunctionControl::NONE, function_type)?;
        self.b.begin_block(None)?;
        Ok(main)
    }

    fn finish(mut self, model: ExecutionModel, main: Word) -> EmitResult<Vec<u32>> {
        self.b.ret()?;
        self.b.end_function()?;
        self.b.entry_point(model, main, "main", self.interfaces);
        Ok(self.b.module().assemble())
    }

    fn add(&mut self, a: Word, b: Word) -> EmitResult<Word> {
        Ok(self.b.i_add(self.u32_type, None, a, b)?)
    }

    fn mul(&mut self, a: Word, b: Word) -> EmitResult<Word> {
        Ok(self.b.i_mul(self.u32_type, None, a, b)?)
    }

    fn word(&mut self, word_pointer: Word, buffer: Word, index: Word) -> EmitResult<Word> {
        let zero = self.constant(0);
        Ok(self
            .b
            .access_chain(word_pointer, None, buffer, [zero, index])?)
    }

    fn load(&mut self, pointer: Word) -> EmitResult<Word> {
        Ok(self.b.load(self.u32_type, None, pointer, None, [])?)
    }
}

/// The vertex shader drawing the mesh emulation's output (set 0, `binding`) as a triangle list
/// of `groups * max_primitives * 3` vertices with the mesh program's vertex outputs: Position
/// and the generic outputs at their locations. With a task program (`args_binding` set) the
/// blocks start at the arguments' [`MESH_EMULATION_ARGS_BASE`]. `convert_depth` remaps depth
/// from OpenGL clip space as the mesh program's own epilogue would.
pub fn emit_mesh_emulation_vertex(
    profile: &Profile,
    mesh: &MeshInfo,
    binding: u32,
    args_binding: Option<u32>,
    convert_depth: bool,
) -> EmitResult<Vec<u32>> {
    let mut m = Module::new(profile);
    let u32_type = m.u32_type;
    let s32_type = m.b.type_int(32, 1);
    let f32_type = m.b.type_float(32, None);
    let vec4_type = m.b.type_vector(f32_type, 4);
    let zero_f = m.b.constant_bit32(f32_type, 0.0f32.to_bits());
    let one_f = m.b.constant_bit32(f32_type, 1.0f32.to_bits());
    let half_f = m.b.constant_bit32(f32_type, 0.5f32.to_bits());
    let default_vec4 =
        m.b.constant_composite(vec4_type, [zero_f, zero_f, zero_f, one_f]);

    let (block_pointer, buffer_u32) = m.storage_block(false);
    let buffer = m.buffer(block_pointer, binding, "mesh_emulation_output");
    let args =
        args_binding.map(|binding| m.buffer(block_pointer, binding, "mesh_emulation_task_args"));

    let input_s32 = m.b.type_pointer(None, StorageClass::Input, s32_type);
    let vertex_index = m.global(input_s32, StorageClass::Input);
    m.b.decorate(
        vertex_index,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::VertexIndex)],
    );
    let output_vec4 = m.b.type_pointer(None, StorageClass::Output, vec4_type);
    let output_f32 = m.b.type_pointer(None, StorageClass::Output, f32_type);
    let position = m.global(output_vec4, StorageClass::Output);
    m.b.decorate(
        position,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::Position)],
    );
    let mut generics = [0; Attribute::NUM_GENERICS as usize];
    for &line in &mesh.vertex_lines {
        let Some(index) = Attribute(line as u16).generic_index() else {
            continue;
        };
        if generics[index as usize] == 0 {
            let id = m.global(output_vec4, StorageClass::Output);
            m.decorate(id, Decoration::Location, u32::from(index));
            generics[index as usize] = id;
        }
    }

    let main = m.begin_main()?;
    let vertex_lines = mesh.vertex_lines.len() as u32;
    let max_primitives = m.constant(mesh.max_primitives.max(1));
    let three = m.constant(3);
    let draw_vertex = m.b.load(s32_type, None, vertex_index, None, [])?;
    let draw_vertex = m.b.bitcast(u32_type, None, draw_vertex)?;
    let slot = m.b.u_div(u32_type, None, draw_vertex, three)?;
    let corner = m.b.u_mod(u32_type, None, draw_vertex, three)?;
    let workgroup = m.b.u_div(u32_type, None, slot, max_primitives)?;
    let primitive = m.b.u_mod(u32_type, None, slot, max_primitives)?;
    let block_words = m.constant(mesh_emulation_block_words(mesh));
    let mut block = m.mul(workgroup, block_words)?;
    if let Some(args) = args {
        let base_index = m.constant(MESH_EMULATION_ARGS_BASE);
        let pointer = m.word(buffer_u32, args, base_index)?;
        let base = m.load(pointer)?;
        block = m.add(block, base)?;
    }
    let vertex_words = m.constant(mesh.max_vertices * vertex_lines);
    let indices = m.add(block, vertex_words)?;
    let primitive_word = m.mul(primitive, three)?;
    let primitive_word = m.add(primitive_word, corner)?;
    let index_word = m.add(indices, primitive_word)?;
    let pointer = m.word(buffer_u32, buffer, index_word)?;
    let vertex = m.load(pointer)?;
    let last_vertex = m.constant(mesh.max_vertices.max(1) - 1);
    let glsl = m.b.ext_inst_import("GLSL.std.450");
    let vertex = m.b.ext_inst(
        u32_type,
        None,
        glsl,
        GlslStd450Op::UMin as u32,
        [Operand::IdRef(vertex), Operand::IdRef(last_vertex)],
    )?;
    let lines = m.constant(vertex_lines);
    let vertex_offset = m.mul(vertex, lines)?;
    let vertex_base = m.add(block, vertex_offset)?;

    m.b.store(position, default_vec4, None, [])?;
    for &generic in generics.iter().filter(|&&g| g != 0) {
        m.b.store(generic, default_vec4, None, [])?;
    }
    let mut position_z = None;
    let mut position_w = None;
    for (line, &attribute) in mesh.vertex_lines.iter().enumerate() {
        let attribute = Attribute(attribute as u16);
        let (variable, element) = if let Some(index) = attribute.generic_index() {
            (generics[index as usize], u32::from(attribute.component()))
        } else if (Attribute::POSITION_X.0..=Attribute::POSITION_W.0).contains(&attribute.0) {
            (position, u32::from(attribute.0 - Attribute::POSITION_X.0))
        } else {
            continue;
        };
        let line = m.constant(line as u32);
        let word_index = m.add(vertex_base, line)?;
        let pointer = m.word(buffer_u32, buffer, word_index)?;
        let word = m.load(pointer)?;
        let value = m.b.bitcast(f32_type, None, word)?;
        let element_id = m.constant(element);
        let pointer = m.b.access_chain(output_f32, None, variable, [element_id])?;
        m.b.store(pointer, value, None, [])?;
        if variable == position && element == 2 {
            position_z = Some(value);
        } else if variable == position && element == 3 {
            position_w = Some(value);
        }
    }
    if let (true, Some(z), Some(w)) = (convert_depth, position_z, position_w) {
        let sum = m.b.f_add(f32_type, None, z, w)?;
        let depth = m.b.f_mul(f32_type, None, sum, half_f)?;
        let two = m.constant(2);
        let pointer = m.b.access_chain(output_f32, None, position, [two])?;
        m.b.store(pointer, depth, None, [])?;
    }
    m.finish(ExecutionModel::Vertex, main)
}

/// The distribution pass of the mesh emulation with a task program: one invocation between
/// the task and mesh dispatches. Set 0 holds the payload buffer (binding 0), the offsets (1:
/// first mesh workgroup of each task), the arguments (2, [`MESH_EMULATION_ARGS_WORDS`] words)
/// and the arena counter (3: output words allocated so far, kept counting past the arena's end),
/// with [`MeshEmulationDistributionConstants`] as push constants. The draw's output is allocated
/// from the arena every draw of the submission shares; mesh workgroups that do not fit are not
/// dispatched.
pub fn emit_mesh_emulation_distribution(profile: &Profile) -> EmitResult<Vec<u32>> {
    let mut m = Module::new(profile);
    let (bool_type, u32_type) = (m.bool_type, m.u32_type);
    let (block_pointer, buffer_u32) = m.storage_block(true);
    let payload = m.buffer(block_pointer, 0, "payload");
    let offsets = m.buffer(block_pointer, 1, "offsets");
    let args = m.buffer(block_pointer, 2, "args");
    let counter = m.buffer(block_pointer, 3, "arena_counter");

    let members = [u32_type; MeshEmulationDistributionConstants::WORDS as usize];
    let constants_type = m.b.type_struct(members);
    m.b.decorate(constants_type, Decoration::Block, []);
    for member in 0..MeshEmulationDistributionConstants::WORDS {
        m.b.member_decorate(
            constants_type,
            member,
            Decoration::Offset,
            [Operand::LiteralBit32(member * 4)],
        );
    }
    let constants_pointer =
        m.b.type_pointer(None, StorageClass::PushConstant, constants_type);
    let constants = m.global(constants_pointer, StorageClass::PushConstant);
    let constant_u32 = m.b.type_pointer(None, StorageClass::PushConstant, u32_type);
    let function_u32 = m.b.type_pointer(None, StorageClass::Function, u32_type);

    let main = m.begin_main()?;
    let total_var =
        m.b.variable(function_u32, None, StorageClass::Function, None);
    let task_var =
        m.b.variable(function_u32, None, StorageClass::Function, None);
    let push = |m: &mut Module, member: u32| -> EmitResult<Word> {
        let member = m.constant(member);
        let pointer = m.b.access_chain(constant_u32, None, constants, [member])?;
        m.load(pointer)
    };
    let tasks = push(&mut m, 0)?;
    let payload_words = push(&mut m, 1)?;
    let capacity_words = push(&mut m, 2)?;
    let block_words = push(&mut m, 3)?;
    let vertices_per_group = push(&mut m, 4)?;
    let max_groups = push(&mut m, 5)?;
    let zero = m.constant(0);
    let one = m.constant(1);
    m.b.store(total_var, zero, None, [])?;
    m.b.store(task_var, zero, None, [])?;

    let header = m.b.id();
    let condition = m.b.id();
    let body = m.b.id();
    let continue_label = m.b.id();
    let merge = m.b.id();
    m.b.branch(header)?;
    m.b.begin_block(Some(header))?;
    m.b.loop_merge(merge, continue_label, LoopControl::NONE, [])?;
    m.b.branch(condition)?;
    m.b.begin_block(Some(condition))?;
    let task = m.load(task_var)?;
    let more = m.b.u_less_than(bool_type, None, task, tasks)?;
    m.b.branch_conditional(more, body, merge, [])?;
    m.b.begin_block(Some(body))?;
    let total = m.load(total_var)?;
    let task_base = m.mul(task, payload_words)?;
    let count_word = m.add(task_base, one)?;
    let pointer = m.word(buffer_u32, payload, count_word)?;
    let requested = m.load(pointer)?;
    let pointer = m.word(buffer_u32, offsets, task)?;
    m.b.store(pointer, total, None, [])?;
    let total = m.add(total, requested)?;
    m.b.store(total_var, total, None, [])?;
    m.b.branch(continue_label)?;
    m.b.begin_block(Some(continue_label))?;
    let task = m.load(task_var)?;
    let next = m.add(task, one)?;
    m.b.store(task_var, next, None, [])?;
    m.b.branch(header)?;
    m.b.begin_block(Some(merge))?;

    let total = m.load(total_var)?;
    let counter_word = m.word(buffer_u32, counter, zero)?;
    let base = m.load(counter_word)?;
    let allocated = m.mul(total, block_words)?;
    let end = m.add(base, allocated)?;
    m.b.store(counter_word, end, None, [])?;
    let fits = m.b.u_less_than(bool_type, None, base, capacity_words)?;
    let left = m.b.i_sub(u32_type, None, capacity_words, base)?;
    let room = m.b.select(u32_type, None, fits, left, zero)?;
    let room_groups = m.b.u_div(u32_type, None, room, block_words)?;
    let glsl = m.b.ext_inst_import("GLSL.std.450");
    let umin = |m: &mut Module, a: Word, b: Word| -> EmitResult<Word> {
        Ok(m.b.ext_inst(
            u32_type,
            None,
            glsl,
            GlslStd450Op::UMin as u32,
            [Operand::IdRef(a), Operand::IdRef(b)],
        )?)
    };
    let dispatched = umin(&mut m, total, room_groups)?;
    let dispatched = umin(&mut m, dispatched, max_groups)?;
    let draw = MESH_EMULATION_DRAW_ARGS_OFFSET / 4;
    let draw_vertices = m.mul(dispatched, vertices_per_group)?;
    for (word, value) in [
        (0, dispatched),
        (1, one),
        (2, one),
        (MESH_EMULATION_ARGS_REQUESTED, total),
        (draw, draw_vertices),
        (draw + 1, one),
        (draw + 2, zero),
        (draw + 3, zero),
        (MESH_EMULATION_ARGS_BASE, base),
        (MESH_EMULATION_ARGS_TASKS, tasks),
    ] {
        let index = m.constant(word);
        let pointer = m.word(buffer_u32, args, index)?;
        m.b.store(pointer, value, None, [])?;
    }
    m.b.execution_mode(main, ExecutionMode::LocalSize, [1, 1, 1]);
    m.finish(ExecutionModel::GLCompute, main)
}
