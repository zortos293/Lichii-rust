//! Emission state: the module under construction and every type, constant, resource, interface
//! variable and helper function a program needs, ported from the C++ `spirv_emit_context.cpp`.
//!
//! rspirv deduplicates types by scanning every global (constants included) and never
//! deduplicates constants, so this context caches both: emission code takes types from the
//! fields here or from the cached helpers ([`EmitContext::type_pointer`] and friends) and
//! constants from the `const_*` helpers, never from `b.type_*` or `b.constant_*` directly.

use rspirv::dr::{self, Builder, Operand};
use rspirv::spirv::{
    self, BuiltIn, Capability, Decoration, Dim, FunctionControl, GlslStd450Op,
    ImageFormat as SpvFormat, LoopControl, Scope, SelectionControl, StorageClass, Word,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use super::emit::context_get_set::{get_cbuf_u32_at, get_cbuf_u32x2_at};
use super::{
    EmitError, EmitResult, NUM_IMAGE_SCALING_WORDS, NUM_TEXTURE_SCALING_WORDS,
    RENDER_AREA_LAYOUT_OFFSET, RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, RESCALING_LAYOUT_IMAGES_OFFSET,
    RESCALING_LAYOUT_TEXTURES_OFFSET,
};
use crate::backend::Bindings;
use crate::info::{
    ImageBufferDescriptor, ImageDescriptor, Interpolation, StorageBufferDescriptor,
    TextureDescriptor,
};
use crate::ir::{Attribute, BlockId, ImageFormat, InstId, Program, TextureType, Type, Value};
use crate::runtime::AttributeType;
use crate::translate::TranslatedProgram;
use crate::{Profile, RuntimeInfo, ShaderInfo, ShaderStage};

/// The four vector widths of one scalar type, indexed by component count (`f32[1]` is the
/// scalar, `f32[4]` the vec4). Zero when the program does not use the scalar type.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VectorTypes([Word; 4]);

impl VectorTypes {
    fn define(b: &mut Builder, base: Word, name: &str) -> VectorTypes {
        b.name(base, name);
        let mut defs = [base; 4];
        for (i, def) in defs.iter_mut().enumerate().skip(1) {
            *def = b.type_vector(base, i as u32 + 1);
            b.name(*def, format!("{name}x{}", i + 1));
        }
        VectorTypes(defs)
    }
}

impl std::ops::Index<usize> for VectorTypes {
    type Output = Word;

    fn index(&self, size: usize) -> &Word {
        &self.0[size - 1]
    }
}

/// Element type of a constant buffer view; the views alias the same binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CbufElement {
    U8,
    S8,
    U16,
    S16,
    U32,
    F32,
    U32x2,
    U32x4,
}

/// Element type of a storage buffer view; the views alias the same binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StorageElement {
    U8,
    S8,
    U16,
    S16,
    U32,
    F32,
    U64,
    U32x2,
    U32x4,
}

/// Pointer types of one storage buffer view: to the block and to one element.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StorageTypeDefinition {
    pub array: Word,
    pub element: Word,
}

#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "read by the image families, which are stubs in this milestone"
)]
pub(crate) struct TextureDefinition {
    pub id: Word,
    pub sampled_type: Word,
    pub pointer_type: Word,
    pub image_type: Word,
    pub count: u32,
    pub is_multisample: bool,
    pub is_integer: bool,
}

#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "read by the image families, which are stubs in this milestone"
)]
pub(crate) struct TextureBufferDefinition {
    pub id: Word,
    pub count: u32,
}

#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "read by the image families, which are stubs in this milestone"
)]
pub(crate) struct ImageBufferDefinition {
    pub id: Word,
    pub image_type: Word,
    pub pointer_type: Word,
    pub count: u32,
    pub is_integer: bool,
}

#[derive(Clone, Copy, Debug)]
#[allow(
    dead_code,
    reason = "read by the image families, which are stubs in this milestone"
)]
pub(crate) struct ImageDefinition {
    pub id: Word,
    pub image_type: Word,
    pub pointer_type: Word,
    pub count: u32,
    pub is_integer: bool,
}

/// How a generic input's declared type becomes the F32 the IR reads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum InputGenericLoadOp {
    #[default]
    None,
    Bitcast,
    SToF,
    UToF,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct InputGenericInfo {
    pub id: Word,
    pub pointer_type: Word,
    pub component_type: Word,
    pub load_op: InputGenericLoadOp,
}

/// The output variable holding one component of a generic output. Transform feedback splits a
/// generic into several variables; `first_element` is the component the variable starts at.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GenericElementInfo {
    pub id: Word,
    pub first_element: u32,
    pub num_components: u32,
}

/// 32-bit global atomics on addresses the storage buffer pass could not track, routed at run
/// time to the storage buffer holding the address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GlobalAtomic {
    IAdd,
    SMin,
    UMin,
    SMax,
    UMax,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exchange,
    AddF32,
}

impl GlobalAtomic {
    const COUNT: usize = 12;
    const ALL: [GlobalAtomic; GlobalAtomic::COUNT] = [
        GlobalAtomic::IAdd,
        GlobalAtomic::SMin,
        GlobalAtomic::UMin,
        GlobalAtomic::SMax,
        GlobalAtomic::UMax,
        GlobalAtomic::Inc,
        GlobalAtomic::Dec,
        GlobalAtomic::And,
        GlobalAtomic::Or,
        GlobalAtomic::Xor,
        GlobalAtomic::Exchange,
        GlobalAtomic::AddF32,
    ];
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CasOperation {
    Increment,
    Decrement,
    FpAdd,
    FpMin,
    FpMax,
}

/// A phi emitted with placeholder values, patched once every value it names is defined.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DeferredPhi {
    pub block: usize,
    pub index: usize,
    pub inst: InstId,
}

/// Everything emission needs. Family functions read the fields directly and emit through `b`.
#[allow(
    dead_code,
    reason = "fields and helpers the per-opcode families read as they are filled in"
)]
pub(crate) struct EmitContext<'a> {
    pub b: Builder,
    pub profile: &'a Profile,
    pub runtime: &'a RuntimeInfo,
    pub program: &'a Program,
    pub info: &'a ShaderInfo,
    pub translated: &'a TranslatedProgram,
    pub stage: ShaderStage,
    pub requires_coherent_global_memory: bool,
    pub requires_coherent_image_memory: bool,

    defs: Vec<Word>,
    labels: Vec<Word>,
    pub(crate) phis: Vec<DeferredPhi>,
    pub(crate) main_function: usize,
    u32_constants: FxHashMap<u32, Word>,
    s32_constants: FxHashMap<u32, Word>,
    f32_constants: FxHashMap<u32, Word>,
    u64_constants: FxHashMap<u64, Word>,
    f64_constants: FxHashMap<u64, Word>,
    composite_constants: FxHashMap<(Word, SmallVec<[Word; 4]>), Word>,
    null_constants: FxHashMap<Word, Word>,
    pointer_types: FxHashMap<(u32, Word), Word>,
    array_types: FxHashMap<(Word, u32), Word>,
    capabilities: FxHashSet<u32>,
    extensions: FxHashSet<&'static str>,

    /// The GLSL.std.450 extended instruction set.
    pub glsl: Word,
    pub void_id: Word,
    pub u1: Word,
    pub u8: Word,
    pub s8: Word,
    pub u16: Word,
    pub s16: Word,
    pub u64: Word,
    pub f32: VectorTypes,
    pub u32: VectorTypes,
    pub s32: VectorTypes,
    pub f16: VectorTypes,
    pub f64: VectorTypes,

    pub true_value: Word,
    pub false_value: Word,
    pub u32_zero_value: Word,
    pub f32_zero_value: Word,

    uniform_types: [Word; 8],
    storage_types: [StorageTypeDefinition; 9],

    pub private_u32: Word,
    pub shared_u8: Word,
    pub shared_u16: Word,
    pub shared_u32: Word,
    pub shared_u64: Word,
    pub shared_u32x2: Word,
    pub shared_u32x4: Word,
    pub input_f32: Word,
    pub input_u32: Word,
    pub input_s32: Word,
    pub output_f32: Word,
    pub output_u32: Word,

    pub image_buffer_type: Word,
    pub image_u32: Word,

    cbufs: [[Word; 8]; ShaderInfo::MAX_CBUFS],
    ssbos: Vec<[Word; 9]>,
    pub texture_buffers: Vec<TextureBufferDefinition>,
    pub image_buffers: Vec<ImageBufferDefinition>,
    pub textures: Vec<TextureDefinition>,
    pub images: Vec<ImageDefinition>,

    pub workgroup_id: Word,
    pub local_invocation_id: Word,
    pub invocation_id: Word,
    pub patch_vertices_in: Word,
    pub sample_id: Word,
    pub is_helper_invocation: Word,
    pub subgroup_local_invocation_id: Word,
    pub subgroup_mask_eq: Word,
    pub subgroup_mask_lt: Word,
    pub subgroup_mask_le: Word,
    pub subgroup_mask_gt: Word,
    pub subgroup_mask_ge: Word,
    pub instance_id: Word,
    pub instance_index: Word,
    pub base_instance: Word,
    pub vertex_id: Word,
    pub vertex_index: Word,
    pub draw_index: Word,
    pub base_vertex: Word,
    pub front_face: Word,
    pub point_coord: Word,
    pub tess_coord: Word,
    pub clip_distances: Word,
    pub layer: Word,
    pub viewport_index: Word,
    pub viewport_mask: Word,
    pub primitive_id: Word,

    pub fswzadd_lut_a: Word,
    pub fswzadd_lut_b: Word,

    pub indexed_load_func: Word,
    pub indexed_store_func: Word,

    pub rescaling_uniform_constant: Word,
    pub rescaling_push_constants: Word,
    pub rescaling_textures_type: Word,
    pub rescaling_images_type: Word,
    pub rescaling_textures_member_index: u32,
    pub rescaling_images_member_index: u32,
    pub rescaling_downfactor_member_index: u32,
    pub texture_rescaling_index: u32,
    pub image_rescaling_index: u32,

    pub render_area_push_constant: Word,
    pub render_area_member_index: u32,

    pub local_memory: Word,

    pub uses_explicit_workgroup_layout: bool,
    pub shared_memory_u8: Word,
    pub shared_memory_u16: Word,
    pub shared_memory_u32: Word,
    pub shared_memory_u64: Word,
    pub shared_memory_u32x2: Word,
    pub shared_memory_u32x4: Word,
    pub shared_memory_u32_type: Word,

    pub shared_store_u8_func: Word,
    pub shared_store_u16_func: Word,
    pub increment_cas_shared: Word,
    pub increment_cas_ssbo: Word,
    pub decrement_cas_shared: Word,
    pub decrement_cas_ssbo: Word,
    pub f32_add_cas: Word,
    pub f16x2_add_cas: Word,
    pub f16x2_min_cas: Word,
    pub f16x2_max_cas: Word,
    pub f32x2_add_cas: Word,
    pub f32x2_min_cas: Word,
    pub f32x2_max_cas: Word,
    pub write_storage_cas_loop_func: Word,
    pub load_global_func_u32: Word,
    pub load_global_func_u32x2: Word,
    pub load_global_func_u32x4: Word,
    pub write_global_func_u32: Word,
    pub write_global_func_u32x2: Word,
    pub write_global_func_u32x4: Word,
    global_routing: bool,
    global_atomic_funcs: [Word; GlobalAtomic::COUNT],
    atomic_cas_global_func_u32: Word,

    pub need_input_position_indirect: bool,
    pub input_position: Word,
    pub input_generics: [InputGenericInfo; 32],
    pub output_point_size: Word,
    pub output_position: Word,
    pub output_generics: [[GenericElementInfo; 4]; 32],
    pub output_tess_level_outer: Word,
    pub output_tess_level_inner: Word,
    pub patches: [Word; 30],
    pub frag_color: [Word; 8],
    pub sample_mask: Word,
    pub frag_depth: Word,

    /// Variables listed on the entry point.
    pub interfaces: Vec<Word>,
    /// Result ids of shuffles whose opcode is rewritten after assembly (see the warp family).
    pub opcode_rewrites: Vec<(Word, spirv::Op)>,

    pub load_const_func_u8: Word,
    pub load_const_func_u16: Word,
    pub load_const_func_u32: Word,
    pub load_const_func_f32: Word,
    pub load_const_func_u32x2: Word,
    pub load_const_func_u32x4: Word,

    /// Ids decorated NonUniform; non-empty adds the descriptor indexing capabilities.
    pub non_uniform_ids: FxHashSet<Word>,
    pub uses_nonuniform_sampled_image: bool,
    pub uses_nonuniform_storage_image: bool,
    pub uses_nonuniform_uniform_texel_buffer: bool,
    pub uses_nonuniform_storage_texel_buffer: bool,
}

fn uses_nvn_storage_descriptor(info: &ShaderInfo, desc: &StorageBufferDescriptor) -> bool {
    const DESCRIPTOR_SIZE: u32 = 0x10;
    if desc.cbuf_index != 0 || desc.cbuf_offset < info.nvn_buffer_base {
        return false;
    }
    let offset = desc.cbuf_offset - info.nvn_buffer_base;
    if !offset.is_multiple_of(DESCRIPTOR_SIZE) {
        return false;
    }
    let slot = offset / DESCRIPTOR_SIZE;
    slot < 16 && info.nvn_buffer_used & (1 << slot) != 0
}

fn stage_name(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::VertexA => "vs_a",
        ShaderStage::VertexB => "vs",
        ShaderStage::TessellationControl => "tcs",
        ShaderStage::TessellationEval => "tes",
        ShaderStage::Geometry => "gs",
        ShaderStage::Fragment => "fs",
        ShaderStage::Compute => "cs",
        ShaderStage::Task => "ts",
        ShaderStage::Mesh => "ms",
    }
}

fn spv_image_format(format: ImageFormat) -> SpvFormat {
    match format {
        ImageFormat::Typeless => SpvFormat::Unknown,
        ImageFormat::R8Uint => SpvFormat::R8ui,
        ImageFormat::R8Sint => SpvFormat::R8i,
        ImageFormat::R16Uint => SpvFormat::R16ui,
        ImageFormat::R16Sint => SpvFormat::R16i,
        ImageFormat::R32Uint => SpvFormat::R32ui,
        ImageFormat::R32G32Uint => SpvFormat::Rg32ui,
        ImageFormat::R32G32B32A32Uint => SpvFormat::Rgba32ui,
    }
}

fn descriptor_name(
    stage: ShaderStage,
    prefix: &str,
    cbuf_index: u32,
    cbuf_offset: u32,
    count: u32,
) -> String {
    let stage = stage_name(stage);
    if count > 1 {
        format!("{stage}_{prefix}{cbuf_index}_{cbuf_offset:02x}x{count}")
    } else {
        format!("{stage}_{prefix}{cbuf_index}_{cbuf_offset:02x}")
    }
}

#[derive(Clone, Copy)]
enum BindingKind {
    Uniform,
    Storage,
    Texture,
    Image,
}

fn binding_slot(bindings: &mut Bindings, unified: bool, kind: BindingKind) -> &mut u32 {
    if unified {
        return &mut bindings.unified;
    }
    match kind {
        BindingKind::Uniform => &mut bindings.uniform_buffer,
        BindingKind::Storage => &mut bindings.storage_buffer,
        BindingKind::Texture => &mut bindings.texture,
        BindingKind::Image => &mut bindings.image,
    }
}

#[allow(
    dead_code,
    reason = "helpers the per-opcode families call as they are filled in"
)]
impl<'a> EmitContext<'a> {
    /// Builds the module header, types, constants, resources and helper functions of
    /// `translated`, advancing `bindings` past the descriptors it declares.
    pub fn new(
        profile: &'a Profile,
        runtime: &'a RuntimeInfo,
        translated: &'a TranslatedProgram,
        bindings: &mut Bindings,
    ) -> EmitResult<EmitContext<'a>> {
        let mut b = Builder::new();
        b.set_version(
            ((profile.supported_spirv >> 16) & 0xff) as u8,
            ((profile.supported_spirv >> 8) & 0xff) as u8,
        );
        b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);
        let glsl = b.ext_inst_import("GLSL.std.450");
        let program = &translated.ir;
        let info = &translated.info;
        let mut ctx = EmitContext {
            b,
            profile,
            runtime,
            program,
            info,
            translated,
            stage: translated.stage,
            requires_coherent_global_memory: info.requires_coherent_global_memory,
            requires_coherent_image_memory: info.requires_coherent_image_memory,
            defs: vec![0; program.num_insts()],
            labels: vec![0; program.num_blocks()],
            phis: Vec::new(),
            main_function: 0,
            u32_constants: FxHashMap::default(),
            s32_constants: FxHashMap::default(),
            f32_constants: FxHashMap::default(),
            u64_constants: FxHashMap::default(),
            f64_constants: FxHashMap::default(),
            composite_constants: FxHashMap::default(),
            null_constants: FxHashMap::default(),
            pointer_types: FxHashMap::default(),
            array_types: FxHashMap::default(),
            capabilities: FxHashSet::default(),
            extensions: FxHashSet::default(),
            glsl,
            void_id: 0,
            u1: 0,
            u8: 0,
            s8: 0,
            u16: 0,
            s16: 0,
            u64: 0,
            f32: VectorTypes::default(),
            u32: VectorTypes::default(),
            s32: VectorTypes::default(),
            f16: VectorTypes::default(),
            f64: VectorTypes::default(),
            true_value: 0,
            false_value: 0,
            u32_zero_value: 0,
            f32_zero_value: 0,
            uniform_types: [0; 8],
            storage_types: [StorageTypeDefinition::default(); 9],
            private_u32: 0,
            shared_u8: 0,
            shared_u16: 0,
            shared_u32: 0,
            shared_u64: 0,
            shared_u32x2: 0,
            shared_u32x4: 0,
            input_f32: 0,
            input_u32: 0,
            input_s32: 0,
            output_f32: 0,
            output_u32: 0,
            image_buffer_type: 0,
            image_u32: 0,
            cbufs: [[0; 8]; ShaderInfo::MAX_CBUFS],
            ssbos: Vec::new(),
            texture_buffers: Vec::new(),
            image_buffers: Vec::new(),
            textures: Vec::new(),
            images: Vec::new(),
            workgroup_id: 0,
            local_invocation_id: 0,
            invocation_id: 0,
            patch_vertices_in: 0,
            sample_id: 0,
            is_helper_invocation: 0,
            subgroup_local_invocation_id: 0,
            subgroup_mask_eq: 0,
            subgroup_mask_lt: 0,
            subgroup_mask_le: 0,
            subgroup_mask_gt: 0,
            subgroup_mask_ge: 0,
            instance_id: 0,
            instance_index: 0,
            base_instance: 0,
            vertex_id: 0,
            vertex_index: 0,
            draw_index: 0,
            base_vertex: 0,
            front_face: 0,
            point_coord: 0,
            tess_coord: 0,
            clip_distances: 0,
            layer: 0,
            viewport_index: 0,
            viewport_mask: 0,
            primitive_id: 0,
            fswzadd_lut_a: 0,
            fswzadd_lut_b: 0,
            indexed_load_func: 0,
            indexed_store_func: 0,
            rescaling_uniform_constant: 0,
            rescaling_push_constants: 0,
            rescaling_textures_type: 0,
            rescaling_images_type: 0,
            rescaling_textures_member_index: 0,
            rescaling_images_member_index: 0,
            rescaling_downfactor_member_index: 0,
            texture_rescaling_index: bindings.texture_scaling_index,
            image_rescaling_index: bindings.image_scaling_index,
            render_area_push_constant: 0,
            render_area_member_index: 0,
            local_memory: 0,
            uses_explicit_workgroup_layout: false,
            shared_memory_u8: 0,
            shared_memory_u16: 0,
            shared_memory_u32: 0,
            shared_memory_u64: 0,
            shared_memory_u32x2: 0,
            shared_memory_u32x4: 0,
            shared_memory_u32_type: 0,
            shared_store_u8_func: 0,
            shared_store_u16_func: 0,
            increment_cas_shared: 0,
            increment_cas_ssbo: 0,
            decrement_cas_shared: 0,
            decrement_cas_ssbo: 0,
            f32_add_cas: 0,
            f16x2_add_cas: 0,
            f16x2_min_cas: 0,
            f16x2_max_cas: 0,
            f32x2_add_cas: 0,
            f32x2_min_cas: 0,
            f32x2_max_cas: 0,
            write_storage_cas_loop_func: 0,
            load_global_func_u32: 0,
            load_global_func_u32x2: 0,
            load_global_func_u32x4: 0,
            write_global_func_u32: 0,
            write_global_func_u32x2: 0,
            write_global_func_u32x4: 0,
            global_routing: false,
            global_atomic_funcs: [0; GlobalAtomic::COUNT],
            atomic_cas_global_func_u32: 0,
            need_input_position_indirect: false,
            input_position: 0,
            input_generics: [InputGenericInfo::default(); 32],
            output_point_size: 0,
            output_position: 0,
            output_generics: [[GenericElementInfo::default(); 4]; 32],
            output_tess_level_outer: 0,
            output_tess_level_inner: 0,
            patches: [0; 30],
            frag_color: [0; 8],
            sample_mask: 0,
            frag_depth: 0,
            interfaces: Vec::new(),
            opcode_rewrites: Vec::new(),
            load_const_func_u8: 0,
            load_const_func_u16: 0,
            load_const_func_u32: 0,
            load_const_func_f32: 0,
            load_const_func_u32x2: 0,
            load_const_func_u32x4: 0,
            non_uniform_ids: FxHashSet::default(),
            uses_nonuniform_sampled_image: false,
            uses_nonuniform_storage_image: false,
            uses_nonuniform_uniform_texel_buffer: false,
            uses_nonuniform_storage_texel_buffer: false,
        };
        let unified = profile.unified_descriptor_binding;
        ctx.add_capability(Capability::Shader);
        ctx.define_common_types();
        ctx.define_common_constants();
        ctx.define_inputs()?;
        ctx.define_outputs()?;
        ctx.define_local_memory();
        ctx.define_shared_memory()?;
        ctx.define_shared_memory_functions()?;
        ctx.define_constant_buffers(binding_slot(bindings, unified, BindingKind::Uniform));
        ctx.define_constant_buffer_indirect_functions()?;
        ctx.define_storage_buffers(binding_slot(bindings, unified, BindingKind::Storage))?;
        ctx.define_texture_buffers(binding_slot(bindings, unified, BindingKind::Texture))?;
        ctx.define_image_buffers(binding_slot(bindings, unified, BindingKind::Image));
        let texture_binding = binding_slot(bindings, unified, BindingKind::Texture);
        let mut texture_scaling = 0;
        ctx.define_textures(texture_binding, &mut texture_scaling)?;
        bindings.texture_scaling_index += texture_scaling;
        let image_binding = binding_slot(bindings, unified, BindingKind::Image);
        let mut image_scaling = 0;
        ctx.define_images(image_binding, &mut image_scaling)?;
        bindings.image_scaling_index += image_scaling;
        ctx.define_attribute_mem_access()?;
        ctx.define_write_storage_cas_loop_function()?;
        ctx.define_global_memory_functions()?;
        ctx.define_rescaling_input();
        if info.uses_rescaling_uniform && info.uses_render_area && unified {
            return Err(EmitError::Unsupported(
                "rescaling and render-area push constants in one module",
            ));
        }
        ctx.define_render_area();
        Ok(ctx)
    }

    pub fn add_capability(&mut self, capability: Capability) {
        if self.capabilities.insert(capability as u32) {
            self.b.capability(capability);
        }
    }

    pub fn add_extension(&mut self, extension: &'static str) {
        if self.extensions.insert(extension) {
            self.b.extension(extension);
        }
    }

    /// Whether this stage may use subgroup operations.
    pub fn supports_subgroups(&self) -> bool {
        self.profile.supports_subgroup_stage(self.stage)
    }

    /// Label of an IR block.
    pub fn label(&self, block: BlockId) -> Word {
        self.labels[block.0 as usize]
    }

    pub(crate) fn assign_labels(&mut self) {
        for &block in &self.program.block_order {
            self.labels[block.0 as usize] = self.b.id();
        }
    }

    /// Argument `index` of `inst`, with identities followed.
    pub fn arg(&self, inst: InstId, index: usize) -> Value {
        self.program.resolve(self.program.arg(inst, index))
    }

    pub fn flags(&self, inst: InstId) -> u32 {
        self.program.inst(inst).flags
    }

    /// An argument that must be a 32-bit immediate (indices, components, bindings).
    pub fn imm_u32(&self, inst: InstId, index: usize) -> EmitResult<u32> {
        let value = self.arg(inst, index);
        value.imm_u32().ok_or_else(|| {
            EmitError::InvalidIr(format!(
                "argument {index} of {:?} is {value:?}, not an immediate",
                self.program.op(inst)
            ))
        })
    }

    /// The id of an IR value: an immediate becomes a constant, an instruction its result.
    pub fn def(&mut self, value: Value) -> EmitResult<Word> {
        match self.program.resolve(value) {
            Value::Inst(id) => match self.defs[id.0 as usize] {
                0 => Err(EmitError::InvalidIr(format!(
                    "{:?} result used before its definition",
                    self.program.op(id)
                ))),
                def => Ok(def),
            },
            Value::U1(v) => Ok(if v { self.true_value } else { self.false_value }),
            Value::U32(v) => Ok(self.const_u32(v)),
            Value::U64(v) => Ok(self.const_u64(v)),
            Value::F32(v) => Ok(self.const_f32(v)),
            Value::F64(v) => Ok(self.const_f64(v)),
            v => Err(EmitError::InvalidIr(format!(
                "immediate {v:?} has no SPIR-V value"
            ))),
        }
    }

    /// Like [`EmitContext::def`], with `None` for a `Void` argument (an absent optional
    /// operand such as a texture offset).
    pub fn def_opt(&mut self, value: Value) -> EmitResult<Option<Word>> {
        match self.program.resolve(value) {
            Value::Void => Ok(None),
            v => self.def(v).map(Some),
        }
    }

    pub fn def_arg(&mut self, inst: InstId, index: usize) -> EmitResult<Word> {
        let value = self.arg(inst, index);
        self.def(value)
    }

    /// Records the result id of `inst`. Families call it for the pseudo-instructions
    /// (carry, zero, sparse, ...) attached to the instruction they emit.
    pub fn set_def(&mut self, inst: InstId, id: Word) {
        self.defs[inst.0 as usize] = id;
    }

    pub fn is_defined(&self, inst: InstId) -> bool {
        self.defs[inst.0 as usize] != 0
    }

    pub fn const_u32(&mut self, value: u32) -> Word {
        if let Some(&id) = self.u32_constants.get(&value) {
            return id;
        }
        let id = self.b.constant_bit32(self.u32[1], value);
        self.u32_constants.insert(value, id);
        id
    }

    pub fn const_s32(&mut self, value: i32) -> Word {
        let bits = value as u32;
        if let Some(&id) = self.s32_constants.get(&bits) {
            return id;
        }
        let id = self.b.constant_bit32(self.s32[1], bits);
        self.s32_constants.insert(bits, id);
        id
    }

    /// An f32 constant, cached by bit pattern (0.0 and -0.0 stay distinct).
    pub fn const_f32(&mut self, value: f32) -> Word {
        let bits = value.to_bits();
        if let Some(&id) = self.f32_constants.get(&bits) {
            return id;
        }
        let id = self.b.constant_bit32(self.f32[1], bits);
        self.f32_constants.insert(bits, id);
        id
    }

    pub fn const_u64(&mut self, value: u64) -> Word {
        if let Some(&id) = self.u64_constants.get(&value) {
            return id;
        }
        let id = self.b.constant_bit64(self.u64, value);
        self.u64_constants.insert(value, id);
        id
    }

    pub fn const_f64(&mut self, value: f64) -> Word {
        let bits = value.to_bits();
        if let Some(&id) = self.f64_constants.get(&bits) {
            return id;
        }
        let id = self.b.constant_bit64(self.f64[1], bits);
        self.f64_constants.insert(bits, id);
        id
    }

    pub fn const_composite(&mut self, ty: Word, constituents: &[Word]) -> Word {
        let key = (ty, SmallVec::from_slice(constituents));
        if let Some(&id) = self.composite_constants.get(&key) {
            return id;
        }
        let id = self.b.constant_composite(ty, constituents.iter().copied());
        self.composite_constants.insert(key, id);
        id
    }

    /// A `u32` vector constant of one to four components.
    pub fn const_u32_vec(&mut self, values: &[u32]) -> Word {
        let mut ids = SmallVec::<[Word; 4]>::new();
        for &v in values {
            ids.push(self.const_u32(v));
        }
        if ids.len() == 1 {
            return ids[0];
        }
        let ty = self.u32[ids.len()];
        self.const_composite(ty, &ids)
    }

    /// An `s32` vector constant of one to four components.
    pub fn const_s32_vec(&mut self, values: &[i32]) -> Word {
        let mut ids = SmallVec::<[Word; 4]>::new();
        for &v in values {
            ids.push(self.const_s32(v));
        }
        if ids.len() == 1 {
            return ids[0];
        }
        let ty = self.s32[ids.len()];
        self.const_composite(ty, &ids)
    }

    pub fn const_null(&mut self, ty: Word) -> Word {
        if let Some(&id) = self.null_constants.get(&ty) {
            return id;
        }
        let id = self.b.constant_null(ty);
        self.null_constants.insert(ty, id);
        id
    }

    /// A pointer type, cached.
    pub fn type_pointer(&mut self, storage_class: StorageClass, pointee: Word) -> Word {
        let key = (storage_class as u32, pointee);
        if let Some(&id) = self.pointer_types.get(&key) {
            return id;
        }
        let id = self.b.type_pointer(None, storage_class, pointee);
        self.pointer_types.insert(key, id);
        id
    }

    /// An array type of `length` elements, cached.
    pub fn type_array(&mut self, element: Word, length: u32) -> Word {
        if let Some(&id) = self.array_types.get(&(element, length)) {
            return id;
        }
        let length_id = self.const_u32(length);
        let id = self.b.type_array(element, length_id);
        self.array_types.insert((element, length), id);
        id
    }

    /// The IR type's SPIR-V type, for phis and other type-generic instructions.
    pub fn type_id(&self, ty: Type) -> EmitResult<Word> {
        Ok(match ty {
            Type::U1 => self.u1,
            Type::U8 => self.u8,
            Type::U16 => self.u16,
            Type::U32 => self.u32[1],
            Type::U64 => self.u64,
            Type::F16 => self.f16[1],
            Type::F32 => self.f32[1],
            Type::F64 => self.f64[1],
            Type::U32X2 => self.u32[2],
            Type::U32X3 => self.u32[3],
            Type::U32X4 => self.u32[4],
            Type::F16X2 => self.f16[2],
            Type::F16X3 => self.f16[3],
            Type::F16X4 => self.f16[4],
            Type::F32X2 => self.f32[2],
            Type::F32X3 => self.f32[3],
            Type::F32X4 => self.f32[4],
            Type::F64X2 => self.f64[2],
            Type::F64X3 => self.f64[3],
            Type::F64X4 => self.f64[4],
            ty => return Err(EmitError::InvalidIr(format!("no SPIR-V type for {ty}"))),
        })
    }

    /// A GLSL.std.450 instruction.
    pub fn glsl_inst(&mut self, ty: Word, op: GlslStd450Op, operands: &[Word]) -> EmitResult<Word> {
        Ok(self.b.ext_inst(
            ty,
            None,
            self.glsl,
            op as u32,
            operands.iter().map(|&id| Operand::IdRef(id)),
        )?)
    }

    /// The constant buffer view of `element` type bound for guest constant buffer `binding`.
    pub fn cbuf(&self, binding: usize, element: CbufElement) -> Word {
        self.cbufs[binding][element as usize]
    }

    /// Pointer type to one element of a constant buffer view.
    pub fn uniform_type(&self, element: CbufElement) -> Word {
        self.uniform_types[element as usize]
    }

    /// The storage buffer view of `element` type for storage buffer descriptor slot `index`.
    pub fn ssbo(&self, index: usize, element: StorageElement) -> Word {
        self.ssbos[index][element as usize]
    }

    pub fn storage_type(&self, element: StorageElement) -> StorageTypeDefinition {
        self.storage_types[element as usize]
    }

    /// The routing function of an untracked 32-bit global atomic, or `None` when no storage
    /// buffer can be routed to. Its body is emitted after the entry point.
    pub fn global_atomic_function(&mut self, kind: GlobalAtomic) -> Option<Word> {
        if !self.global_routing {
            return None;
        }
        let slot = &mut self.global_atomic_funcs[kind as usize];
        if *slot == 0 {
            *slot = self.b.id();
        }
        Some(*slot)
    }

    /// The routing function of an untracked global compare-exchange, as
    /// [`EmitContext::global_atomic_function`].
    pub fn global_compare_exchange_function(&mut self) -> Option<Word> {
        if !self.global_routing {
            return None;
        }
        if self.atomic_cas_global_func_u32 == 0 {
            self.atomic_cas_global_func_u32 = self.b.id();
        }
        Some(self.atomic_cas_global_func_u32)
    }

    /// Bit offset of the byte `offset` addresses within its word.
    pub fn bit_offset8(&mut self, offset: Value) -> EmitResult<Word> {
        if let Some(offset) = offset.imm_u32() {
            return Ok(self.const_u32((offset % 4) * 8));
        }
        self.bit_offset(offset, 24)
    }

    /// Bit offset of the halfword `offset` addresses within its word.
    pub fn bit_offset16(&mut self, offset: Value) -> EmitResult<Word> {
        if let Some(offset) = offset.imm_u32() {
            return Ok(self.const_u32(((offset / 2) % 2) * 16));
        }
        self.bit_offset(offset, 16)
    }

    fn bit_offset(&mut self, offset: Value, mask: u32) -> EmitResult<Word> {
        let offset = self.def(offset)?;
        let three = self.const_u32(3);
        let mask = self.const_u32(mask);
        let u32_t = self.u32[1];
        let shifted = self.b.shift_left_logical(u32_t, None, offset, three)?;
        Ok(self.b.bitwise_and(u32_t, None, shifted, mask)?)
    }

    /// A module-scope variable, even while a block is open.
    pub fn global_variable(
        &mut self,
        pointer_type: Word,
        storage_class: StorageClass,
        initializer: Option<Word>,
    ) -> Word {
        let id = self.b.id();
        let mut operands = vec![Operand::StorageClass(storage_class)];
        operands.extend(initializer.map(Operand::IdRef));
        self.b
            .module_mut()
            .types_global_values
            .push(dr::Instruction::new(
                spirv::Op::Variable,
                Some(pointer_type),
                Some(id),
                operands,
            ));
        id
    }

    /// Lists a global variable on the entry point: inputs and outputs always, every other
    /// storage class from SPIR-V 1.4 on, where the interface must name them.
    pub fn add_interface(&mut self, variable: Word, storage_class: StorageClass) {
        if matches!(storage_class, StorageClass::Input | StorageClass::Output)
            || self.profile.supported_spirv >= 0x0001_0400
        {
            self.interfaces.push(variable);
        }
    }

    pub fn decorate(&mut self, target: Word, decoration: Decoration, literals: &[u32]) {
        self.b.decorate(
            target,
            decoration,
            literals.iter().map(|&v| Operand::LiteralBit32(v)),
        );
    }

    pub fn decorate_builtin(&mut self, target: Word, builtin: BuiltIn) {
        self.b
            .decorate(target, Decoration::BuiltIn, [Operand::BuiltIn(builtin)]);
    }

    pub fn member_decorate(
        &mut self,
        structure: Word,
        member: u32,
        decoration: Decoration,
        literals: &[u32],
    ) {
        self.b.member_decorate(
            structure,
            member,
            decoration,
            literals.iter().map(|&v| Operand::LiteralBit32(v)),
        );
    }

    fn define_variable(
        &mut self,
        ty: Word,
        builtin: Option<BuiltIn>,
        storage_class: StorageClass,
        initializer: Option<Word>,
    ) -> Word {
        let pointer_type = self.type_pointer(storage_class, ty);
        let id = self.global_variable(pointer_type, storage_class, initializer);
        if let Some(builtin) = builtin {
            self.decorate_builtin(id, builtin);
        }
        self.add_interface(id, storage_class);
        id
    }

    fn define_input(
        &mut self,
        mut ty: Word,
        per_invocation: bool,
        builtin: Option<BuiltIn>,
    ) -> Word {
        if per_invocation {
            match self.stage {
                ShaderStage::TessellationControl | ShaderStage::TessellationEval => {
                    ty = self.type_array(ty, 32);
                }
                ShaderStage::Geometry => {
                    ty = self.type_array(ty, self.runtime.input_topology.vertices());
                }
                _ => {}
            }
        }
        self.define_variable(ty, builtin, StorageClass::Input, None)
    }

    fn define_output(
        &mut self,
        mut ty: Word,
        invocations: Option<u32>,
        builtin: Option<BuiltIn>,
        initializer: Option<Word>,
    ) -> Word {
        if let Some(invocations) = invocations
            && self.stage == ShaderStage::TessellationControl
        {
            ty = self.type_array(ty, invocations);
        }
        self.define_variable(ty, builtin, StorageClass::Output, initializer)
    }

    fn begin_label(&mut self, label: Word) -> EmitResult<()> {
        self.b.begin_block(Some(label))?;
        Ok(())
    }

    fn function_local_u32(&mut self) -> Word {
        let pointer = self.type_pointer(StorageClass::Function, self.u32[1]);
        self.b.variable(pointer, None, StorageClass::Function, None)
    }

    fn define_common_types(&mut self) {
        let info = self.info;
        let profile = self.profile;
        self.void_id = self.b.type_void();
        self.u1 = self.b.type_bool();
        self.b.name(self.u1, "u1");
        let f32_t = self.b.type_float(32, None);
        self.f32 = VectorTypes::define(&mut self.b, f32_t, "f32");
        let u32_t = self.b.type_int(32, 0);
        self.u32 = VectorTypes::define(&mut self.b, u32_t, "u32");
        let s32_t = self.b.type_int(32, 1);
        self.s32 = VectorTypes::define(&mut self.b, s32_t, "s32");
        self.private_u32 = self.type_pointer(StorageClass::Private, u32_t);
        self.input_f32 = self.type_pointer(StorageClass::Input, f32_t);
        self.input_u32 = self.type_pointer(StorageClass::Input, u32_t);
        self.input_s32 = self.type_pointer(StorageClass::Input, s32_t);
        self.output_f32 = self.type_pointer(StorageClass::Output, f32_t);
        self.output_u32 = self.type_pointer(StorageClass::Output, u32_t);
        if info.uses_int8 && profile.support_int8 {
            self.add_capability(Capability::Int8);
            self.u8 = self.b.type_int(8, 0);
            self.s8 = self.b.type_int(8, 1);
        }
        if info.uses_int16 && profile.support_int16 {
            self.add_capability(Capability::Int16);
            self.u16 = self.b.type_int(16, 0);
            self.s16 = self.b.type_int(16, 1);
        }
        if info.uses_int64 && profile.support_int64 {
            self.add_capability(Capability::Int64);
            self.u64 = self.b.type_int(64, 0);
        }
        if info.uses_fp16 {
            self.add_capability(Capability::Float16);
            let f16_t = self.b.type_float(16, None);
            self.f16 = VectorTypes::define(&mut self.b, f16_t, "f16");
        }
        if info.uses_fp64 {
            self.add_capability(Capability::Float64);
            let f64_t = self.b.type_float(64, None);
            self.f64 = VectorTypes::define(&mut self.b, f64_t, "f64");
        }
    }

    fn define_common_constants(&mut self) {
        self.true_value = self.b.constant_true(self.u1);
        self.false_value = self.b.constant_false(self.u1);
        self.u32_zero_value = self.const_u32(0);
        self.f32_zero_value = self.const_f32(0.0);
    }

    fn define_inputs(&mut self) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        let runtime = self.runtime;
        let loads = &info.loads;
        let stage = self.stage;
        let is_fragment = stage == ShaderStage::Fragment;
        let (u32_t, u32x3, u32x4) = (self.u32[1], self.u32[3], self.u32[4]);
        if info.uses_workgroup_id {
            self.workgroup_id = self.define_input(u32x3, false, Some(BuiltIn::WorkgroupId));
        }
        if info.uses_local_invocation_id {
            self.local_invocation_id =
                self.define_input(u32x3, false, Some(BuiltIn::LocalInvocationId));
        }
        if info.uses_invocation_id {
            self.invocation_id = self.define_input(u32_t, false, Some(BuiltIn::InvocationId));
        }
        if info.uses_invocation_info
            && matches!(
                stage,
                ShaderStage::TessellationControl | ShaderStage::TessellationEval
            )
        {
            self.patch_vertices_in = self.define_input(u32_t, false, Some(BuiltIn::PatchVertices));
        }
        if info.uses_sample_id {
            self.sample_id = self.define_input(u32_t, false, Some(BuiltIn::SampleId));
            if is_fragment {
                self.decorate(self.sample_id, Decoration::Flat, &[]);
            }
        }
        if info.uses_is_helper_invocation {
            self.is_helper_invocation =
                self.define_input(self.u1, false, Some(BuiltIn::HelperInvocation));
        }
        if info.uses_subgroup_mask && self.supports_subgroups() {
            self.subgroup_mask_eq = self.define_input(u32x4, false, Some(BuiltIn::SubgroupEqMask));
            self.subgroup_mask_lt = self.define_input(u32x4, false, Some(BuiltIn::SubgroupLtMask));
            self.subgroup_mask_le = self.define_input(u32x4, false, Some(BuiltIn::SubgroupLeMask));
            self.subgroup_mask_gt = self.define_input(u32x4, false, Some(BuiltIn::SubgroupGtMask));
            self.subgroup_mask_ge = self.define_input(u32x4, false, Some(BuiltIn::SubgroupGeMask));
            if is_fragment {
                for mask in [
                    self.subgroup_mask_eq,
                    self.subgroup_mask_lt,
                    self.subgroup_mask_le,
                    self.subgroup_mask_gt,
                    self.subgroup_mask_ge,
                ] {
                    self.decorate(mask, Decoration::Flat, &[]);
                }
            }
        }
        if (info.uses_fswzadd
            || info.uses_subgroup_invocation_id
            || info.uses_subgroup_shuffles
            || (profile.warp_size_potentially_larger_than_guest
                && (info.uses_subgroup_vote || info.uses_subgroup_mask)))
            && self.supports_subgroups()
        {
            self.add_capability(Capability::GroupNonUniform);
            self.subgroup_local_invocation_id =
                self.define_input(u32_t, false, Some(BuiltIn::SubgroupLocalInvocationId));
            if is_fragment {
                self.decorate(self.subgroup_local_invocation_id, Decoration::Flat, &[]);
            }
        }
        if info.uses_fswzadd {
            let one = self.const_f32(1.0);
            let minus_one = self.const_f32(-1.0);
            let zero = self.const_f32(0.0);
            let f32x4 = self.f32[4];
            self.fswzadd_lut_a = self.const_composite(f32x4, &[minus_one, one, minus_one, zero]);
            self.fswzadd_lut_b =
                self.const_composite(f32x4, &[minus_one, minus_one, one, minus_one]);
        }
        if loads.get(Attribute::PRIMITIVE_ID) {
            self.primitive_id = self.define_input(u32_t, false, Some(BuiltIn::PrimitiveId));
            if is_fragment {
                self.add_capability(Capability::Geometry);
                self.decorate(self.primitive_id, Decoration::Flat, &[]);
            }
        }
        if loads.get(Attribute::LAYER) {
            self.add_capability(Capability::Geometry);
            self.layer = self.define_input(u32_t, false, Some(BuiltIn::Layer));
            self.decorate(self.layer, Decoration::Flat, &[]);
        }
        if loads.any_component(Attribute::POSITION_X)
            || info.passthrough.any_component(Attribute::POSITION_X)
        {
            if !is_fragment && profile.has_broken_spirv_position_input {
                self.need_input_position_indirect = true;
                let block = self.b.type_struct([self.f32[4]]);
                self.input_position = self.define_input(block, true, None);
                self.b.member_decorate(
                    block,
                    0,
                    Decoration::BuiltIn,
                    [Operand::BuiltIn(BuiltIn::Position)],
                );
                self.decorate(block, Decoration::Block, &[]);
            } else {
                let builtin = if is_fragment {
                    BuiltIn::FragCoord
                } else {
                    BuiltIn::Position
                };
                self.input_position = self.define_input(self.f32[4], true, Some(builtin));
                if profile.support_geometry_shader_passthrough
                    && info.passthrough.any_component(Attribute::POSITION_X)
                {
                    self.decorate(self.input_position, Decoration::PassthroughNV, &[]);
                }
            }
        }
        if loads.get(Attribute::INSTANCE_ID) {
            if profile.support_vertex_instance_id {
                self.instance_id = self.define_input(u32_t, true, Some(BuiltIn::InstanceId));
                if loads.get(Attribute::BASE_INSTANCE) {
                    self.base_instance = self.define_draw_parameter(BuiltIn::BaseInstance);
                }
            } else {
                self.instance_index = self.define_input(u32_t, true, Some(BuiltIn::InstanceIndex));
                self.base_instance = self.define_draw_parameter(BuiltIn::BaseInstance);
            }
        } else if loads.get(Attribute::BASE_INSTANCE) {
            self.base_instance = self.define_draw_parameter(BuiltIn::BaseInstance);
        }
        if loads.get(Attribute::VERTEX_ID) {
            if profile.support_vertex_instance_id {
                self.vertex_id = self.define_input(u32_t, true, Some(BuiltIn::VertexId));
                if loads.get(Attribute::BASE_VERTEX) {
                    self.base_vertex = self.define_draw_parameter(BuiltIn::BaseVertex);
                }
            } else {
                self.vertex_index = self.define_input(u32_t, true, Some(BuiltIn::VertexIndex));
                self.base_vertex = self.define_draw_parameter(BuiltIn::BaseVertex);
            }
        } else if loads.get(Attribute::BASE_VERTEX) {
            self.base_vertex = self.define_draw_parameter(BuiltIn::BaseVertex);
        }
        if loads.get(Attribute::DRAW_ID) {
            self.draw_index = self.define_draw_parameter(BuiltIn::DrawIndex);
        }
        if loads.get(Attribute::FRONT_FACE) {
            self.front_face = self.define_input(self.u1, true, Some(BuiltIn::FrontFacing));
        }
        if loads.get(Attribute::POINT_SPRITE_S) || loads.get(Attribute::POINT_SPRITE_T) {
            self.point_coord = self.define_input(self.f32[2], true, Some(BuiltIn::PointCoord));
        }
        if loads.get(Attribute::TESS_EVAL_POINT_U) || loads.get(Attribute::TESS_EVAL_POINT_V) {
            self.tess_coord = self.define_input(self.f32[3], false, Some(BuiltIn::TessCoord));
        }
        for index in 0..Attribute::NUM_GENERICS {
            let input_type = runtime.generic_input_types[index as usize];
            if !runtime.previous_stage_stores.generic(index)
                || (!loads.generic(index) && !info.passthrough.generic(index))
                || input_type == AttributeType::Disabled
            {
                continue;
            }
            let ty = self.attribute_type(input_type);
            let id = self.define_input(ty, true, None);
            self.decorate(id, Decoration::Location, &[u32::from(index)]);
            self.b.name(id, format!("in_attr{index}"));
            self.input_generics[index as usize] = self.attribute_info(input_type, id);
            if info.passthrough.generic(index) && profile.support_geometry_shader_passthrough {
                self.decorate(id, Decoration::PassthroughNV, &[]);
            }
            if !is_fragment {
                continue;
            }
            let is_integer = matches!(
                input_type,
                AttributeType::SignedInt | AttributeType::UnsignedInt
            );
            if runtime.per_primitive_generics & (1 << index) != 0 {
                return Err(EmitError::Unsupported("per-primitive fragment inputs"));
            }
            if is_integer {
                self.decorate(id, Decoration::Flat, &[]);
                continue;
            }
            match info.interpolation[index as usize] {
                Interpolation::Smooth => {}
                Interpolation::NoPerspective => self.decorate(id, Decoration::NoPerspective, &[]),
                Interpolation::Flat => self.decorate(id, Decoration::Flat, &[]),
            }
        }
        if stage == ShaderStage::TessellationEval {
            for index in 0..info.uses_patches.len() {
                if !info.uses_patches[index] {
                    continue;
                }
                let id = self.define_input(self.f32[4], false, None);
                self.decorate(id, Decoration::Patch, &[]);
                self.decorate(id, Decoration::Location, &[index as u32]);
                self.patches[index] = id;
            }
        }
        Ok(())
    }

    fn define_draw_parameter(&mut self, builtin: BuiltIn) -> Word {
        self.add_extension("SPV_KHR_shader_draw_parameters");
        self.add_capability(Capability::DrawParameters);
        self.define_input(self.u32[1], true, Some(builtin))
    }

    /// Declared type of a vertex attribute or colour output of `ty`.
    pub fn attribute_type(&mut self, ty: AttributeType) -> Word {
        let scaled_as_float = self.profile.support_scaled_attributes;
        match ty {
            AttributeType::SignedInt => self.s32[4],
            AttributeType::UnsignedInt => self.u32[4],
            AttributeType::SignedScaled if !scaled_as_float => self.s32[4],
            AttributeType::UnsignedScaled if !scaled_as_float => self.u32[4],
            _ => self.f32[4],
        }
    }

    fn attribute_info(&self, ty: AttributeType, id: Word) -> InputGenericInfo {
        let float = InputGenericInfo {
            id,
            pointer_type: self.input_f32,
            component_type: self.f32[1],
            load_op: InputGenericLoadOp::None,
        };
        let scaled_as_float = self.profile.support_scaled_attributes;
        let (pointer_type, component_type, load_op) = match ty {
            AttributeType::Float => return float,
            AttributeType::Disabled => return InputGenericInfo::default(),
            AttributeType::SignedScaled | AttributeType::UnsignedScaled if scaled_as_float => {
                return float;
            }
            AttributeType::UnsignedInt => {
                (self.input_u32, self.u32[1], InputGenericLoadOp::Bitcast)
            }
            AttributeType::SignedInt => (self.input_s32, self.s32[1], InputGenericLoadOp::Bitcast),
            AttributeType::SignedScaled => (self.input_s32, self.s32[1], InputGenericLoadOp::SToF),
            AttributeType::UnsignedScaled => {
                (self.input_u32, self.u32[1], InputGenericLoadOp::UToF)
            }
        };
        InputGenericInfo {
            id,
            pointer_type,
            component_type,
            load_op,
        }
    }

    fn define_outputs(&mut self) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        let runtime = self.runtime;
        let stage = self.stage;
        let stores = &info.stores;
        let invocations = Some(self.translated.invocations);
        let is_fragment = stage == ShaderStage::Fragment;
        if runtime.convert_depth_mode
            || stores.any_component(Attribute::POSITION_X)
            || stage == ShaderStage::VertexB
        {
            self.output_position =
                self.define_output(self.f32[4], invocations, Some(BuiltIn::Position), None);
        }
        if stores.get(Attribute::POINT_SIZE) || runtime.fixed_state_point_size.is_some() {
            if is_fragment {
                return Err(EmitError::Unsupported(
                    "storing PointSize in a fragment program",
                ));
            }
            self.output_point_size =
                self.define_output(self.f32[1], invocations, Some(BuiltIn::PointSize), None);
        }
        if stores.clip_distances() {
            if is_fragment {
                return Err(EmitError::Unsupported(
                    "storing ClipDistance in a fragment program",
                ));
            }
            if profile.max_user_clip_distances > 0 {
                let used = profile.max_user_clip_distances.min(8);
                let ty = self.type_array(self.f32[1], used);
                let zeros = [self.f32_zero_value; 8];
                let initializer = self.const_composite(ty, &zeros[..used as usize]);
                self.clip_distances = self.define_output(
                    ty,
                    invocations,
                    Some(BuiltIn::ClipDistance),
                    Some(initializer),
                );
            }
        }
        let layer_viewport_allowed =
            profile.support_viewport_index_layer_non_geometry || stage == ShaderStage::Geometry;
        if stores.get(Attribute::LAYER) && layer_viewport_allowed {
            if is_fragment {
                return Err(EmitError::Unsupported(
                    "storing Layer in a fragment program",
                ));
            }
            self.layer = self.define_output(self.u32[1], invocations, Some(BuiltIn::Layer), None);
        }
        if stores.get(Attribute::VIEWPORT_INDEX) && layer_viewport_allowed {
            if is_fragment {
                return Err(EmitError::Unsupported(
                    "storing ViewportIndex in a fragment program",
                ));
            }
            self.viewport_index =
                self.define_output(self.u32[1], invocations, Some(BuiltIn::ViewportIndex), None);
        }
        if stores.get(Attribute::VIEWPORT_MASK) && profile.support_viewport_mask {
            let ty = self.type_array(self.u32[1], 1);
            self.viewport_mask = self.define_output(ty, None, Some(BuiltIn::ViewportMaskNV), None);
        }
        for index in 0..Attribute::NUM_GENERICS {
            if stores.generic(index) {
                self.define_generic_output(index, invocations);
            }
        }
        match stage {
            ShaderStage::TessellationControl => {
                if info.stores_tess_level_outer {
                    let ty = self.type_array(self.f32[1], 4);
                    self.output_tess_level_outer =
                        self.define_output(ty, None, Some(BuiltIn::TessLevelOuter), None);
                    self.decorate(self.output_tess_level_outer, Decoration::Patch, &[]);
                }
                if info.stores_tess_level_inner {
                    let ty = self.type_array(self.f32[1], 2);
                    self.output_tess_level_inner =
                        self.define_output(ty, None, Some(BuiltIn::TessLevelInner), None);
                    self.decorate(self.output_tess_level_inner, Decoration::Patch, &[]);
                }
                for index in 0..info.uses_patches.len() {
                    if !info.uses_patches[index] {
                        continue;
                    }
                    let id = self.define_output(self.f32[4], None, None, None);
                    self.decorate(id, Decoration::Patch, &[]);
                    self.decorate(id, Decoration::Location, &[index as u32]);
                    self.patches[index] = id;
                }
            }
            ShaderStage::Fragment => {
                for index in 0..8u32 {
                    let dual_source = runtime.dual_source_blend && index <= 1;
                    if !dual_source
                        && !info.stores_frag_color[index as usize]
                        && !profile.need_declared_frag_colors
                    {
                        continue;
                    }
                    let ty = self.attribute_type(runtime.color_output_types[index as usize]);
                    let id = self.define_output(ty, None, None, None);
                    self.frag_color[index as usize] = id;
                    if dual_source {
                        self.decorate(id, Decoration::Location, &[0]);
                        self.decorate(id, Decoration::Index, &[index]);
                    } else {
                        self.decorate(id, Decoration::Location, &[index]);
                    }
                    self.b.name(id, format!("frag_color{index}"));
                }
                if info.stores_frag_depth {
                    self.frag_depth = self.define_output(self.f32[1], None, None, None);
                    self.decorate_builtin(self.frag_depth, BuiltIn::FragDepth);
                }
                if info.stores_sample_mask {
                    let ty = self.type_array(self.u32[1], 1);
                    self.sample_mask = self.define_output(ty, None, None, None);
                    self.decorate_builtin(self.sample_mask, BuiltIn::SampleMask);
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn define_generic_output(&mut self, index: u16, invocations: Option<u32>) {
        let base_attr = Attribute::generic(index, 0).0 as usize;
        let mut element = 0u32;
        while element < 4 {
            let xfb = self
                .runtime
                .xfb_varyings
                .get(base_attr + element as usize)
                .filter(|v| v.components > 0)
                .copied();
            let num_components = xfb.map_or(4 - element, |v| v.components.min(4 - element));
            let id = self.define_output(self.f32[num_components as usize], invocations, None, None);
            self.decorate(id, Decoration::Location, &[u32::from(index)]);
            if element > 0 {
                self.decorate(id, Decoration::Component, &[element]);
            }
            if let Some(xfb) = xfb {
                self.decorate(id, Decoration::XfbBuffer, &[xfb.buffer]);
                self.decorate(id, Decoration::XfbStride, &[xfb.stride]);
                self.decorate(id, Decoration::Offset, &[xfb.offset]);
                if self.stage == ShaderStage::Geometry && xfb.stream != 0 {
                    self.decorate(id, Decoration::Stream, &[xfb.stream]);
                }
            }
            self.b.name(id, format!("out_attr{index}"));
            let info = GenericElementInfo {
                id,
                first_element: element,
                num_components,
            };
            for slot in &mut self.output_generics[index as usize]
                [element as usize..(element + num_components) as usize]
            {
                *slot = info;
            }
            element += num_components;
        }
    }

    fn define_local_memory(&mut self) {
        let size = self.translated.local_memory_size;
        if size == 0 {
            return;
        }
        let element = self.b.type_struct([self.u32[1]]);
        let ty = self.type_array(element, size.div_ceil(4));
        let pointer = self.type_pointer(StorageClass::Private, ty);
        self.local_memory = self.global_variable(pointer, StorageClass::Private, None);
        self.add_interface(self.local_memory, StorageClass::Private);
    }

    fn define_shared_memory(&mut self) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        self.uses_explicit_workgroup_layout = profile.support_explicit_workgroup_layout
            && (!info.uses_int8 || profile.support_workgroup_layout_8bit_access)
            && (!info.uses_int16 || profile.support_workgroup_layout_16bit_access);
        let size = self.translated.shared_memory_size;
        if size == 0 {
            return Ok(());
        }
        if self.uses_explicit_workgroup_layout {
            self.add_extension("SPV_KHR_workgroup_memory_explicit_layout");
            self.add_capability(Capability::WorkgroupMemoryExplicitLayoutKHR);
            if info.uses_int8 && profile.support_int8 {
                self.add_capability(Capability::WorkgroupMemoryExplicitLayout8BitAccessKHR);
                (self.shared_memory_u8, self.shared_u8, _) = self.explicit_shared(self.u8, 1);
            }
            if info.uses_int16 && profile.support_int16 {
                self.add_capability(Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
                (self.shared_memory_u16, self.shared_u16, _) = self.explicit_shared(self.u16, 2);
            }
            if info.uses_int64 && profile.support_int64 {
                (self.shared_memory_u64, self.shared_u64, _) = self.explicit_shared(self.u64, 8);
            }
            (
                self.shared_memory_u32,
                self.shared_u32,
                self.shared_memory_u32_type,
            ) = self.explicit_shared(self.u32[1], 4);
            (self.shared_memory_u32x2, self.shared_u32x2, _) = self.explicit_shared(self.u32[2], 8);
            (self.shared_memory_u32x4, self.shared_u32x4, _) =
                self.explicit_shared(self.u32[4], 16);
            return Ok(());
        }
        let ty = self.type_array(self.u32[1], size.div_ceil(4));
        self.shared_memory_u32_type = self.type_pointer(StorageClass::Workgroup, ty);
        self.shared_u32 = self.type_pointer(StorageClass::Workgroup, self.u32[1]);
        self.shared_memory_u32 =
            self.global_variable(self.shared_memory_u32_type, StorageClass::Workgroup, None);
        self.add_interface(self.shared_memory_u32, StorageClass::Workgroup);
        if info.uses_int8 {
            self.shared_store_u8_func = self.shared_store_function(24, 8)?;
        }
        if info.uses_int16 {
            self.shared_store_u16_func = self.shared_store_function(16, 16)?;
        }
        Ok(())
    }

    /// A workgroup block of `element` aliasing the other views of shared memory: the variable,
    /// the pointer type to one element and the pointer type to the block.
    fn explicit_shared(&mut self, element: Word, element_size: u32) -> (Word, Word, Word) {
        let count = self.translated.shared_memory_size.div_ceil(element_size);
        let array = self.type_array(element, count);
        self.decorate(array, Decoration::ArrayStride, &[element_size]);
        let block = self.b.type_struct([array]);
        self.member_decorate(block, 0, Decoration::Offset, &[0]);
        self.decorate(block, Decoration::Block, &[]);
        let pointer = self.type_pointer(StorageClass::Workgroup, block);
        let element_pointer = self.type_pointer(StorageClass::Workgroup, element);
        let variable = self.global_variable(pointer, StorageClass::Workgroup, None);
        self.decorate(variable, Decoration::Aliased, &[]);
        self.add_interface(variable, StorageClass::Workgroup);
        (variable, element_pointer, pointer)
    }

    /// Stores the low `size` bits of a value into shared memory at a byte offset with a
    /// compare-exchange loop, for hosts without 8 and 16-bit workgroup access.
    fn shared_store_function(&mut self, mask: u32, size: u32) -> EmitResult<Word> {
        let u32_t = self.u32[1];
        let func_type = self.b.type_function(self.void_id, [u32_t, u32_t]);
        let loop_header = self.b.id();
        let continue_block = self.b.id();
        let merge_block = self.b.id();
        let func = self
            .b
            .begin_function(self.void_id, None, FunctionControl::NONE, func_type)?;
        let offset = self.b.function_parameter(u32_t)?;
        let insert_value = self.b.function_parameter(u32_t)?;
        self.b.begin_block(None)?;
        let expected_var = self.function_local_u32();
        let two = self.const_u32(2);
        let three = self.const_u32(3);
        let mask = self.const_u32(mask);
        let count = self.const_u32(size);
        let one = self.const_u32(1);
        let zero = self.u32_zero_value;
        let word_offset = self.b.shift_right_arithmetic(u32_t, None, offset, two)?;
        let shift_offset = self.b.shift_left_logical(u32_t, None, offset, three)?;
        let bit_offset = self.b.bitwise_and(u32_t, None, shift_offset, mask)?;
        let word_pointer =
            self.b
                .access_chain(self.shared_u32, None, self.shared_memory_u32, [word_offset])?;
        let initial = self.b.load(u32_t, None, word_pointer, None, [])?;
        self.b.store(expected_var, initial, None, [])?;
        self.b.branch(loop_header)?;
        self.begin_label(loop_header)?;
        self.b
            .loop_merge(merge_block, continue_block, LoopControl::NONE, [])?;
        self.b.branch(continue_block)?;
        self.begin_label(continue_block)?;
        let old_value = self.b.load(u32_t, None, expected_var, None, [])?;
        let new_value =
            self.b
                .bit_field_insert(u32_t, None, old_value, insert_value, bit_offset, count)?;
        let result = self.b.atomic_compare_exchange(
            u32_t,
            None,
            word_pointer,
            one,
            zero,
            zero,
            new_value,
            old_value,
        )?;
        self.b.store(expected_var, result, None, [])?;
        let success = self.b.i_equal(self.u1, None, result, old_value)?;
        self.b
            .branch_conditional(success, merge_block, loop_header, [])?;
        self.begin_label(merge_block)?;
        self.b.ret()?;
        self.b.end_function()?;
        Ok(func)
    }

    fn define_shared_memory_functions(&mut self) -> EmitResult<()> {
        let (array, element, u32_t) = (self.shared_memory_u32_type, self.shared_u32, self.u32[1]);
        if self.info.uses_shared_increment {
            self.increment_cas_shared = self.cas_loop(
                CasOperation::Increment,
                array,
                element,
                u32_t,
                Scope::Workgroup,
            )?;
        }
        if self.info.uses_shared_decrement {
            self.decrement_cas_shared = self.cas_loop(
                CasOperation::Decrement,
                array,
                element,
                u32_t,
                Scope::Workgroup,
            )?;
        }
        Ok(())
    }

    fn cas_function(&mut self, operation: CasOperation, value_type: Word) -> EmitResult<Word> {
        let func_type = self.b.type_function(value_type, [value_type, value_type]);
        let func = self
            .b
            .begin_function(value_type, None, FunctionControl::NONE, func_type)?;
        let op_a = self.b.function_parameter(value_type)?;
        let op_b = self.b.function_parameter(value_type)?;
        self.b.begin_block(None)?;
        let u1 = self.u1;
        let result = match operation {
            CasOperation::Increment => {
                let one = self.const_u32(1);
                let pred = self.b.u_greater_than_equal(u1, None, op_a, op_b)?;
                let incr = self.b.i_add(value_type, None, op_a, one)?;
                self.b
                    .select(value_type, None, pred, self.u32_zero_value, incr)?
            }
            CasOperation::Decrement => {
                let one = self.const_u32(1);
                let lhs = self.b.i_equal(u1, None, op_a, self.u32_zero_value)?;
                let rhs = self.b.u_greater_than(u1, None, op_a, op_b)?;
                let pred = self.b.logical_or(u1, None, lhs, rhs)?;
                let decr = self.b.i_sub(value_type, None, op_a, one)?;
                self.b.select(value_type, None, pred, op_b, decr)?
            }
            CasOperation::FpAdd => self.b.f_add(value_type, None, op_a, op_b)?,
            CasOperation::FpMin => self.glsl_inst(value_type, GlslStd450Op::NMin, &[op_a, op_b])?,
            CasOperation::FpMax => self.glsl_inst(value_type, GlslStd450Op::NMax, &[op_a, op_b])?,
        };
        self.b.ret_value(result)?;
        self.b.end_function()?;
        Ok(func)
    }

    /// A function applying `operation` to a 32-bit word of shared memory or of a storage
    /// buffer with a compare-exchange loop. The word holds a u32, an f32, or an f16x2 or
    /// f32x2 packed as two halves; `value_type` says which.
    fn cas_loop(
        &mut self,
        operation: CasOperation,
        array_pointer: Word,
        element_pointer: Word,
        value_type: Word,
        scope: Scope,
    ) -> EmitResult<Word> {
        let is_shared = scope == Scope::Workgroup;
        let is_struct = !is_shared || self.uses_explicit_workgroup_layout;
        let cas_func = self.cas_function(operation, value_type)?;
        let u32_t = self.u32[1];
        let zero = self.u32_zero_value;
        let scope_id = self.const_u32(scope as u32);
        let loop_header = self.b.id();
        let continue_block = self.b.id();
        let merge_block = self.b.id();
        let func_type = if is_shared {
            self.b.type_function(value_type, [u32_t, value_type])
        } else {
            self.b
                .type_function(value_type, [u32_t, value_type, array_pointer])
        };
        let func = self
            .b
            .begin_function(value_type, None, FunctionControl::NONE, func_type)?;
        let index = self.b.function_parameter(u32_t)?;
        let op_b = self.b.function_parameter(value_type)?;
        let base = if is_shared {
            self.shared_memory_u32
        } else {
            self.b.function_parameter(array_pointer)?
        };
        self.b.begin_block(None)?;
        let expected_var = self.function_local_u32();
        let word_pointer = if is_struct {
            self.b
                .access_chain(element_pointer, None, base, [zero, index])?
        } else {
            self.b.access_chain(element_pointer, None, base, [index])?
        };
        let initial = self.b.load(u32_t, None, word_pointer, None, [])?;
        self.b.store(expected_var, initial, None, [])?;
        self.b.branch(loop_header)?;
        self.begin_label(loop_header)?;
        self.b
            .loop_merge(merge_block, continue_block, LoopControl::NONE, [])?;
        self.b.branch(continue_block)?;
        self.begin_label(continue_block)?;
        let expected = self.b.load(u32_t, None, expected_var, None, [])?;
        let current = self.cas_from_word(value_type, expected)?;
        let new_value = self
            .b
            .function_call(value_type, None, cas_func, [current, op_b])?;
        let new_word = if value_type == self.f32[2] {
            self.glsl_inst(u32_t, GlslStd450Op::PackHalf2x16, &[new_value])?
        } else if value_type == u32_t {
            new_value
        } else {
            self.b.bitcast(u32_t, None, new_value)?
        };
        let result = self.b.atomic_compare_exchange(
            u32_t,
            None,
            word_pointer,
            scope_id,
            zero,
            zero,
            new_word,
            expected,
        )?;
        self.b.store(expected_var, result, None, [])?;
        let success = self.b.i_equal(self.u1, None, result, expected)?;
        self.b
            .branch_conditional(success, merge_block, loop_header, [])?;
        self.begin_label(merge_block)?;
        let old = self.cas_from_word(value_type, result)?;
        self.b.ret_value(old)?;
        self.b.end_function()?;
        Ok(func)
    }

    fn cas_from_word(&mut self, value_type: Word, word: Word) -> EmitResult<Word> {
        if value_type == self.f32[2] {
            return self.glsl_inst(value_type, GlslStd450Op::UnpackHalf2x16, &[word]);
        }
        if value_type == self.u32[1] {
            return Ok(word);
        }
        Ok(self.b.bitcast(value_type, None, word)?)
    }

    fn define_const_buffers(
        &mut self,
        element: CbufElement,
        mut binding: u32,
        ty: Word,
        type_char: char,
        element_size: u32,
    ) {
        let array = self.type_array(ty, 65536 / element_size);
        self.decorate(array, Decoration::ArrayStride, &[element_size]);
        let block = self.b.type_struct([array]);
        self.b.name(
            block,
            format!(
                "{}_cbuf_block_{type_char}{}",
                stage_name(self.stage),
                element_size * 8
            ),
        );
        self.decorate(block, Decoration::Block, &[]);
        self.b.member_name(block, 0, "data");
        self.member_decorate(block, 0, Decoration::Offset, &[0]);
        let block_pointer = self.type_pointer(StorageClass::Uniform, block);
        self.uniform_types[element as usize] = self.type_pointer(StorageClass::Uniform, ty);
        for desc in &self.info.constant_buffer_descriptors {
            let id = self.global_variable(block_pointer, StorageClass::Uniform, None);
            self.decorate(id, Decoration::Binding, &[binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            self.b.name(id, format!("c{}", desc.index));
            for i in 0..desc.count {
                self.cbufs[(desc.index + i) as usize][element as usize] = id;
            }
            self.add_interface(id, StorageClass::Uniform);
            binding += desc.count;
        }
    }

    fn define_constant_buffers(&mut self, binding: &mut u32) {
        let info = self.info;
        let profile = self.profile;
        if info.constant_buffer_descriptors.is_empty() {
            return;
        }
        if !profile.support_descriptor_aliasing {
            self.define_const_buffers(CbufElement::U32x4, *binding, self.u32[4], 'u', 16);
            *binding += info
                .constant_buffer_descriptors
                .iter()
                .map(|d| d.count)
                .sum::<u32>();
            return;
        }
        let mut types = info.used_constant_buffer_types | info.used_indirect_cbuf_types;
        if types.contains(Type::U8) {
            if profile.support_int8 && profile.support_uniform_and_storage_buffer_8bit {
                self.define_const_buffers(CbufElement::U8, *binding, self.u8, 'u', 1);
                self.define_const_buffers(CbufElement::S8, *binding, self.s8, 's', 1);
            } else {
                types |= Type::U32;
            }
        }
        if types.contains(Type::U16) {
            if profile.support_int16 && profile.support_uniform_and_storage_buffer_16bit {
                self.define_const_buffers(CbufElement::U16, *binding, self.u16, 'u', 2);
                self.define_const_buffers(CbufElement::S16, *binding, self.s16, 's', 2);
            } else {
                types |= Type::U32;
            }
        }
        if types.contains(Type::U32) {
            self.define_const_buffers(CbufElement::U32, *binding, self.u32[1], 'u', 4);
        }
        if types.contains(Type::F32) {
            self.define_const_buffers(CbufElement::F32, *binding, self.f32[1], 'f', 4);
        }
        if types.contains(Type::U32X2) {
            self.define_const_buffers(CbufElement::U32x2, *binding, self.u32[2], 'u', 8);
        }
        *binding += info
            .constant_buffer_descriptors
            .iter()
            .map(|d| d.count)
            .sum::<u32>();
    }

    fn define_constant_buffer_indirect_functions(&mut self) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        if !info.uses_cbuf_indirect {
            return Ok(());
        }
        let mut types = info.used_indirect_cbuf_types;
        let aliasing = profile.support_descriptor_aliasing;
        if aliasing && types.contains(Type::U8) {
            if profile.support_int8 && profile.support_uniform_and_storage_buffer_8bit {
                self.load_const_func_u8 = self.cbuf_accessor(self.u8, CbufElement::U8)?;
            } else {
                types |= Type::U32;
            }
        }
        if aliasing && types.contains(Type::U16) {
            if profile.support_int16 && profile.support_uniform_and_storage_buffer_16bit {
                self.load_const_func_u16 = self.cbuf_accessor(self.u16, CbufElement::U16)?;
            } else {
                types |= Type::U32;
            }
        }
        if aliasing && types.contains(Type::F32) {
            self.load_const_func_f32 = self.cbuf_accessor(self.f32[1], CbufElement::F32)?;
        }
        if aliasing && types.contains(Type::U32) {
            self.load_const_func_u32 = self.cbuf_accessor(self.u32[1], CbufElement::U32)?;
        }
        if aliasing && types.contains(Type::U32X2) {
            self.load_const_func_u32x2 = self.cbuf_accessor(self.u32[2], CbufElement::U32x2)?;
        }
        if !aliasing || types.contains(Type::U32X4) {
            self.load_const_func_u32x4 = self.cbuf_accessor(self.u32[4], CbufElement::U32x4)?;
        }
        Ok(())
    }

    /// A function loading `element` at an element offset of a constant buffer picked at run
    /// time, for constant buffer reads with a non-immediate binding.
    fn cbuf_accessor(&mut self, buffer_type: Word, element: CbufElement) -> EmitResult<Word> {
        let u32_t = self.u32[1];
        let func_type = self.b.type_function(buffer_type, [u32_t, u32_t]);
        let func = self
            .b
            .begin_function(buffer_type, None, FunctionControl::NONE, func_type)?;
        let binding = self.b.function_parameter(u32_t)?;
        let offset = self.b.function_parameter(u32_t)?;
        self.b.begin_block(None)?;
        let merge_label = self.b.id();
        let uniform_type = self.uniform_type(element);
        let labels: [Word; ShaderInfo::MAX_INDIRECT_CBUFS] = std::array::from_fn(|_| self.b.id());
        self.b
            .selection_merge(merge_label, SelectionControl::NONE)?;
        self.b.switch(
            binding,
            labels[0],
            labels
                .iter()
                .enumerate()
                .map(|(i, &label)| (Operand::LiteralBit32(i as u32), label)),
        )?;
        for (i, &label) in labels.iter().enumerate() {
            self.begin_label(label)?;
            let cbuf = self.cbuf(i, element);
            let pointer =
                self.b
                    .access_chain(uniform_type, None, cbuf, [self.u32_zero_value, offset])?;
            let result = self.b.load(buffer_type, None, pointer, None, [])?;
            self.b.ret_value(result)?;
        }
        self.begin_label(merge_label)?;
        self.b.unreachable()?;
        self.b.end_function()?;
        Ok(func)
    }

    fn define_ssbos(&mut self, element: StorageElement, mut binding: u32, ty: Word, stride: u32) {
        let array = self.b.type_runtime_array(ty);
        self.decorate(array, Decoration::ArrayStride, &[stride]);
        let block = self.b.type_struct([array]);
        self.decorate(block, Decoration::Block, &[]);
        self.member_decorate(block, 0, Decoration::Offset, &[0]);
        if self.info.requires_coherent_global_memory {
            self.member_decorate(block, 0, Decoration::Coherent, &[]);
        }
        let block_pointer = self.type_pointer(StorageClass::StorageBuffer, block);
        self.storage_types[element as usize] = StorageTypeDefinition {
            array: block_pointer,
            element: self.type_pointer(StorageClass::StorageBuffer, ty),
        };
        let mut index = 0;
        for desc in &self.info.storage_buffers_descriptors {
            let id = self.global_variable(block_pointer, StorageClass::StorageBuffer, None);
            self.decorate(id, Decoration::Binding, &[binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            self.b.name(id, format!("ssbo{index}"));
            self.add_interface(id, StorageClass::StorageBuffer);
            for i in 0..desc.count {
                self.ssbos[(index + i) as usize][element as usize] = id;
            }
            index += desc.count;
            binding += desc.count;
        }
    }

    fn define_storage_buffers(&mut self, binding: &mut u32) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        if info.storage_buffers_descriptors.is_empty() {
            return Ok(());
        }
        let slots: u32 = info
            .storage_buffers_descriptors
            .iter()
            .map(|d| d.count)
            .sum();
        self.ssbos = vec![[0; 9]; slots as usize];
        self.add_extension("SPV_KHR_storage_buffer_storage_class");
        let mut used = if profile.support_descriptor_aliasing {
            info.used_storage_buffer_types
        } else {
            Type::U32
        };
        used |= Type::U32;
        if profile.support_int8 && profile.support_storage_buffer_8bit && used.contains(Type::U8) {
            self.define_ssbos(StorageElement::U8, *binding, self.u8, 1);
            self.define_ssbos(StorageElement::S8, *binding, self.s8, 1);
        }
        if profile.support_int16 && profile.support_storage_buffer_16bit && used.contains(Type::U16)
        {
            self.define_ssbos(StorageElement::U16, *binding, self.u16, 2);
            self.define_ssbos(StorageElement::S16, *binding, self.s16, 2);
        }
        if used.contains(Type::U32) {
            self.define_ssbos(StorageElement::U32, *binding, self.u32[1], 4);
        }
        if used.contains(Type::F32) {
            self.define_ssbos(StorageElement::F32, *binding, self.f32[1], 4);
        }
        if used.contains(Type::U64) {
            self.define_ssbos(StorageElement::U64, *binding, self.u64, 8);
        }
        if used.contains(Type::U32X2) {
            self.define_ssbos(StorageElement::U32x2, *binding, self.u32[2], 8);
        }
        if used.contains(Type::U32X4) {
            self.define_ssbos(StorageElement::U32x4, *binding, self.u32[4], 16);
        }
        *binding += slots;
        if info.uses_global_increment
            || info.uses_global_decrement
            || info.uses_atomic_f32_add
            || info.uses_atomic_f16x2_add
            || info.uses_atomic_f16x2_min
            || info.uses_atomic_f16x2_max
            || info.uses_atomic_f32x2_add
            || info.uses_atomic_f32x2_min
            || info.uses_atomic_f32x2_max
        {
            self.add_capability(Capability::VariablePointersStorageBuffer);
        }
        let StorageTypeDefinition { array, element } = self.storage_type(StorageElement::U32);
        let device = Scope::Device;
        let cas = [
            (
                info.uses_global_increment,
                CasOperation::Increment,
                self.u32[1],
            ),
            (
                info.uses_global_decrement,
                CasOperation::Decrement,
                self.u32[1],
            ),
            (info.uses_atomic_f32_add, CasOperation::FpAdd, self.f32[1]),
            (info.uses_atomic_f16x2_add, CasOperation::FpAdd, self.f16[2]),
            (info.uses_atomic_f16x2_min, CasOperation::FpMin, self.f16[2]),
            (info.uses_atomic_f16x2_max, CasOperation::FpMax, self.f16[2]),
            (info.uses_atomic_f32x2_add, CasOperation::FpAdd, self.f32[2]),
            (info.uses_atomic_f32x2_min, CasOperation::FpMin, self.f32[2]),
            (info.uses_atomic_f32x2_max, CasOperation::FpMax, self.f32[2]),
        ];
        let mut funcs = [0; 9];
        for (func, (used, operation, value_type)) in funcs.iter_mut().zip(cas) {
            if used {
                *func = self.cas_loop(operation, array, element, value_type, device)?;
            }
        }
        [
            self.increment_cas_ssbo,
            self.decrement_cas_ssbo,
            self.f32_add_cas,
            self.f16x2_add_cas,
            self.f16x2_min_cas,
            self.f16x2_max_cas,
            self.f32x2_add_cas,
            self.f32x2_min_cas,
            self.f32x2_max_cas,
        ] = funcs;
        Ok(())
    }

    fn define_texture_buffers(&mut self, binding: &mut u32) -> EmitResult<()> {
        let info = self.info;
        if info.texture_buffer_descriptors.is_empty() {
            return Ok(());
        }
        self.image_buffer_type = self.b.type_image(
            self.f32[1],
            Dim::DimBuffer,
            0,
            0,
            0,
            1,
            SpvFormat::Unknown,
            None,
        );
        let ty = self.type_pointer(StorageClass::UniformConstant, self.image_buffer_type);
        for desc in &info.texture_buffer_descriptors {
            if desc.count != 1 {
                return Err(EmitError::Unsupported("arrays of texture buffers"));
            }
            let id = self.global_variable(ty, StorageClass::UniformConstant, None);
            self.decorate(id, Decoration::Binding, &[*binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            let name = descriptor_name(
                self.stage,
                "texbuf",
                desc.handle.cbuf_index,
                desc.handle.cbuf_offset,
                desc.count,
            );
            self.b.name(id, name);
            self.texture_buffers.push(TextureBufferDefinition {
                id,
                count: desc.count,
            });
            self.add_interface(id, StorageClass::UniformConstant);
            *binding += 1;
        }
        Ok(())
    }

    fn define_image_buffers(&mut self, binding: &mut u32) {
        for desc in &self.info.image_buffer_descriptors {
            let ImageBufferDescriptor {
                format,
                is_integer,
                cbuf_index,
                cbuf_offset,
                count,
                ..
            } = *desc;
            let sampled_type = if is_integer { self.u32[1] } else { self.f32[1] };
            let image_type = self.b.type_image(
                sampled_type,
                Dim::DimBuffer,
                0,
                0,
                0,
                2,
                spv_image_format(format),
                None,
            );
            let pointer_type = self.type_pointer(StorageClass::UniformConstant, image_type);
            let id = self.global_variable(pointer_type, StorageClass::UniformConstant, None);
            self.decorate(id, Decoration::Binding, &[*binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            let name = descriptor_name(self.stage, "imgbuf", cbuf_index, cbuf_offset, count);
            self.b.name(id, name);
            if self.info.requires_coherent_image_memory {
                self.decorate(id, Decoration::Coherent, &[]);
            }
            self.image_buffers.push(ImageBufferDefinition {
                id,
                image_type,
                pointer_type,
                count,
                is_integer,
            });
            self.add_interface(id, StorageClass::UniformConstant);
            *binding += 1;
        }
    }

    fn sampled_image_type(&mut self, desc: &TextureDescriptor) -> EmitResult<Word> {
        let ty = if desc.is_integer {
            self.u32[1]
        } else {
            self.f32[1]
        };
        let depth = u32::from(desc.is_depth);
        let ms = u32::from(desc.is_multisample);
        let (dim, arrayed, ms) = match desc.ty {
            TextureType::Color1D => (Dim::Dim1D, 0, 0),
            TextureType::ColorArray1D => (Dim::Dim1D, 1, 0),
            TextureType::Color2D | TextureType::Color2DRect => (Dim::Dim2D, 0, ms),
            TextureType::ColorArray2D => (Dim::Dim2D, 1, ms),
            TextureType::Color3D => (Dim::Dim3D, 0, 0),
            TextureType::ColorCube => (Dim::DimCube, 0, 0),
            TextureType::ColorArrayCube => (Dim::DimCube, 1, 0),
            TextureType::Buffer => {
                return Err(EmitError::InvalidIr(
                    "texture descriptor of buffer type".into(),
                ));
            }
        };
        Ok(self
            .b
            .type_image(ty, dim, depth, arrayed, ms, 1, SpvFormat::Unknown, None))
    }

    fn storage_image_type(
        &mut self,
        desc: &ImageDescriptor,
        sampled_type: Word,
    ) -> EmitResult<Word> {
        let (dim, arrayed) = match desc.ty {
            TextureType::Color1D => (Dim::Dim1D, 0),
            TextureType::ColorArray1D => (Dim::Dim1D, 1),
            TextureType::Color2D => (Dim::Dim2D, 0),
            TextureType::ColorArray2D => (Dim::Dim2D, 1),
            TextureType::Color3D => (Dim::Dim3D, 0),
            TextureType::Buffer => return Err(EmitError::Unsupported("image buffer descriptors")),
            ty => {
                return Err(EmitError::InvalidIr(format!(
                    "image descriptor of type {ty:?}"
                )));
            }
        };
        Ok(self.b.type_image(
            sampled_type,
            dim,
            0,
            arrayed,
            0,
            2,
            spv_image_format(desc.format),
            None,
        ))
    }

    fn define_textures(&mut self, binding: &mut u32, scaling_index: &mut u32) -> EmitResult<()> {
        let info = self.info;
        for desc in &info.texture_descriptors {
            let image_type = self.sampled_image_type(desc)?;
            let sampled_type = self.b.type_sampled_image(image_type);
            let pointer_type = self.type_pointer(StorageClass::UniformConstant, sampled_type);
            let desc_type = if desc.count > 1 {
                let array = self.type_array(sampled_type, desc.count);
                self.type_pointer(StorageClass::UniformConstant, array)
            } else {
                pointer_type
            };
            let id = self.global_variable(desc_type, StorageClass::UniformConstant, None);
            self.decorate(id, Decoration::Binding, &[*binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            let name = descriptor_name(
                self.stage,
                "tex",
                desc.handle.cbuf_index,
                desc.handle.cbuf_offset,
                desc.count,
            );
            self.b.name(id, name);
            self.textures.push(TextureDefinition {
                id,
                sampled_type,
                pointer_type,
                image_type,
                count: desc.count,
                is_multisample: desc.is_multisample,
                is_integer: desc.is_integer,
            });
            self.add_interface(id, StorageClass::UniformConstant);
            *binding += 1;
            *scaling_index += 1;
        }
        if info.uses_atomic_image_u32 {
            self.image_u32 = self.type_pointer(StorageClass::Image, self.u32[1]);
        }
        Ok(())
    }

    fn define_images(&mut self, binding: &mut u32, scaling_index: &mut u32) -> EmitResult<()> {
        for desc in &self.info.image_descriptors {
            let sampled_type = if desc.is_integer {
                self.u32[1]
            } else {
                self.f32[1]
            };
            let image_type = self.storage_image_type(desc, sampled_type)?;
            let pointer_type = self.type_pointer(StorageClass::UniformConstant, image_type);
            let id = self.global_variable(pointer_type, StorageClass::UniformConstant, None);
            self.decorate(id, Decoration::Binding, &[*binding]);
            self.decorate(id, Decoration::DescriptorSet, &[0]);
            let name = descriptor_name(
                self.stage,
                "img",
                desc.cbuf_index,
                desc.cbuf_offset,
                desc.count,
            );
            self.b.name(id, name);
            if self.info.requires_coherent_image_memory {
                self.decorate(id, Decoration::Coherent, &[]);
            }
            self.images.push(ImageDefinition {
                id,
                image_type,
                pointer_type,
                count: desc.count,
                is_integer: desc.is_integer,
            });
            self.add_interface(id, StorageClass::UniformConstant);
            *binding += 1;
            *scaling_index += 1;
        }
        Ok(())
    }

    fn define_attribute_mem_access(&mut self) -> EmitResult<()> {
        if self.info.loads_indexed_attributes {
            self.indexed_load_func = self.indexed_load_function()?;
        }
        if self.info.stores_indexed_attributes {
            self.indexed_store_func = self.indexed_store_function()?;
        }
        Ok(())
    }

    /// Splits an attribute byte offset into the vec4 index the switch selects on and the
    /// component within it.
    fn attribute_switch_indices(&mut self, offset: Word) -> EmitResult<(Word, Word)> {
        let u32_t = self.u32[1];
        let two = self.const_u32(2);
        let three = self.const_u32(3);
        let base_index = self.b.shift_right_arithmetic(u32_t, None, offset, two)?;
        let masked_index = self.b.bitwise_and(u32_t, None, base_index, three)?;
        let compare_index = self
            .b
            .shift_right_arithmetic(u32_t, None, base_index, two)?;
        Ok((masked_index, compare_index))
    }

    fn indexed_load_function(&mut self) -> EmitResult<Word> {
        let info = self.info;
        let is_array = self.stage == ShaderStage::Geometry;
        let (u32_t, f32_t) = (self.u32[1], self.f32[1]);
        let end_block = self.b.id();
        let default_label = self.b.id();
        let func_type = if is_array {
            self.b.type_function(f32_t, [u32_t, u32_t])
        } else {
            self.b.type_function(f32_t, [u32_t])
        };
        let func = self
            .b
            .begin_function(f32_t, None, FunctionControl::NONE, func_type)?;
        let offset = self.b.function_parameter(u32_t)?;
        let vertex = if is_array {
            Some(self.b.function_parameter(u32_t)?)
        } else {
            None
        };
        self.b.begin_block(None)?;
        let (masked_index, compare_index) = self.attribute_switch_indices(offset)?;
        let mut cases = Vec::new();
        let position = info.loads.any_component(Attribute::POSITION_X);
        if position {
            cases.push((u32::from(Attribute::POSITION_X.0) >> 2, self.b.id()));
        }
        let generic_base = u32::from(Attribute::GENERIC_0_X.0) >> 2;
        for index in 0..Attribute::NUM_GENERICS {
            if info.loads.generic(index) {
                cases.push((generic_base + u32::from(index), self.b.id()));
            }
        }
        self.b.selection_merge(end_block, SelectionControl::NONE)?;
        self.b.switch(
            compare_index,
            default_label,
            cases
                .iter()
                .map(|&(literal, label)| (Operand::LiteralBit32(literal), label)),
        )?;
        self.begin_label(default_label)?;
        self.b.ret_value(self.f32_zero_value)?;
        let mut labels = cases.iter().map(|&(_, label)| label);
        if position {
            self.begin_label(labels.next().unwrap_or_default())?;
            let mut indices = SmallVec::<[Word; 3]>::new();
            indices.extend(vertex);
            if self.need_input_position_indirect {
                indices.push(self.u32_zero_value);
            }
            indices.push(masked_index);
            let pointer =
                self.b
                    .access_chain(self.input_f32, None, self.input_position, indices)?;
            let result = self.b.load(f32_t, None, pointer, None, [])?;
            self.b.ret_value(result)?;
        }
        for index in 0..Attribute::NUM_GENERICS {
            if !info.loads.generic(index) {
                continue;
            }
            self.begin_label(labels.next().unwrap_or_default())?;
            let generic = self.input_generics[index as usize];
            if generic.id == 0 {
                self.b.ret_value(self.f32_zero_value)?;
                continue;
            }
            let mut indices = SmallVec::<[Word; 2]>::new();
            indices.extend(vertex);
            indices.push(masked_index);
            let pointer = self
                .b
                .access_chain(generic.pointer_type, None, generic.id, indices)?;
            let value = self
                .b
                .load(generic.component_type, None, pointer, None, [])?;
            let result = self.input_to_f32(generic.load_op, value)?;
            self.b.ret_value(result)?;
        }
        self.begin_label(end_block)?;
        self.b.unreachable()?;
        self.b.end_function()?;
        Ok(func)
    }

    /// Converts a loaded generic input component to the F32 the IR reads.
    pub fn input_to_f32(&mut self, load_op: InputGenericLoadOp, value: Word) -> EmitResult<Word> {
        let f32_t = self.f32[1];
        Ok(match load_op {
            InputGenericLoadOp::None => value,
            InputGenericLoadOp::Bitcast => self.b.bitcast(f32_t, None, value)?,
            InputGenericLoadOp::SToF => self.b.convert_s_to_f(f32_t, None, value)?,
            InputGenericLoadOp::UToF => self.b.convert_u_to_f(f32_t, None, value)?,
        })
    }

    fn indexed_store_function(&mut self) -> EmitResult<Word> {
        let info = self.info;
        let max_clip = self.profile.max_user_clip_distances;
        let (u32_t, f32_t) = (self.u32[1], self.f32[1]);
        let end_block = self.b.id();
        let default_label = self.b.id();
        let func_type = self.b.type_function(self.void_id, [u32_t, f32_t]);
        let func = self
            .b
            .begin_function(self.void_id, None, FunctionControl::NONE, func_type)?;
        let offset = self.b.function_parameter(u32_t)?;
        let store_value = self.b.function_parameter(f32_t)?;
        self.b.begin_block(None)?;
        let (masked_index, compare_index) = self.attribute_switch_indices(offset)?;
        let mut cases = Vec::new();
        let position = info.stores.any_component(Attribute::POSITION_X);
        if position {
            cases.push((u32::from(Attribute::POSITION_X.0) >> 2, self.b.id()));
        }
        let generic_base = u32::from(Attribute::GENERIC_0_X.0) >> 2;
        for index in 0..Attribute::NUM_GENERICS {
            if info.stores.generic(index) {
                cases.push((generic_base + u32::from(index), self.b.id()));
            }
        }
        let clip = info.stores.clip_distances();
        let clip_base = u32::from(Attribute::CLIP_DISTANCE_0.0) >> 2;
        if clip && max_clip >= 4 {
            cases.push((clip_base, self.b.id()));
        }
        if clip && max_clip >= 8 {
            cases.push((clip_base + 1, self.b.id()));
        }
        self.b.selection_merge(end_block, SelectionControl::NONE)?;
        self.b.switch(
            compare_index,
            default_label,
            cases
                .iter()
                .map(|&(literal, label)| (Operand::LiteralBit32(literal), label)),
        )?;
        self.begin_label(default_label)?;
        self.b.ret()?;
        let mut labels = cases.iter().map(|&(_, label)| label);
        if position {
            self.begin_label(labels.next().unwrap_or_default())?;
            let pointer =
                self.b
                    .access_chain(self.output_f32, None, self.output_position, [masked_index])?;
            self.b.store(pointer, store_value, None, [])?;
            self.b.ret()?;
        }
        for index in 0..Attribute::NUM_GENERICS {
            if !info.stores.generic(index) {
                continue;
            }
            let generic = self.output_generics[index as usize][0];
            if generic.num_components != 4 {
                return Err(EmitError::Unsupported(
                    "indexed stores to transform feedback outputs",
                ));
            }
            self.begin_label(labels.next().unwrap_or_default())?;
            let pointer = self
                .b
                .access_chain(self.output_f32, None, generic.id, [masked_index])?;
            self.b.store(pointer, store_value, None, [])?;
            self.b.ret()?;
        }
        if clip && max_clip >= 4 {
            self.begin_label(labels.next().unwrap_or_default())?;
            let pointer =
                self.b
                    .access_chain(self.output_f32, None, self.clip_distances, [masked_index])?;
            self.b.store(pointer, store_value, None, [])?;
            self.b.ret()?;
        }
        if clip && max_clip >= 8 {
            self.begin_label(labels.next().unwrap_or_default())?;
            let four = self.const_u32(4);
            let fixed_index = self.b.i_add(u32_t, None, masked_index, four)?;
            let pointer =
                self.b
                    .access_chain(self.output_f32, None, self.clip_distances, [fixed_index])?;
            self.b.store(pointer, store_value, None, [])?;
            self.b.ret()?;
        }
        self.begin_label(end_block)?;
        self.b.unreachable()?;
        self.b.end_function()?;
        Ok(func)
    }

    fn define_write_storage_cas_loop_function(&mut self) -> EmitResult<()> {
        let info = self.info;
        let profile = self.profile;
        let direct_int8 = profile.support_int8
            && profile.support_storage_buffer_8bit
            && profile.support_descriptor_aliasing;
        let direct_int16 = profile.support_int16
            && profile.support_storage_buffer_16bit
            && profile.support_descriptor_aliasing;
        if (direct_int8 || !info.uses_int8) && (direct_int16 || !info.uses_int16) {
            return Ok(());
        }
        self.add_capability(Capability::VariablePointersStorageBuffer);
        let u32_t = self.u32[1];
        let pointer_type = self.type_pointer(StorageClass::StorageBuffer, u32_t);
        let func_type = self
            .b
            .type_function(self.void_id, [pointer_type, u32_t, u32_t, u32_t]);
        let func = self
            .b
            .begin_function(self.void_id, None, FunctionControl::NONE, func_type)?;
        let pointer = self.b.function_parameter(pointer_type)?;
        let value = self.b.function_parameter(u32_t)?;
        let bit_offset = self.b.function_parameter(u32_t)?;
        let bit_count = self.b.function_parameter(u32_t)?;
        self.b.begin_block(None)?;
        let expected_var = self.function_local_u32();
        let initial = self.b.load(u32_t, None, pointer, None, [])?;
        self.b.store(expected_var, initial, None, [])?;
        let scope_device = self.const_u32(1);
        let relaxed = self.u32_zero_value;
        let body_label = self.b.id();
        let continue_label = self.b.id();
        let endloop_label = self.b.id();
        let beginloop_label = self.b.id();
        self.b.branch(beginloop_label)?;
        self.begin_label(beginloop_label)?;
        self.b
            .loop_merge(endloop_label, continue_label, LoopControl::NONE, [])?;
        self.b.branch(body_label)?;
        self.begin_label(body_label)?;
        let expected = self.b.load(u32_t, None, expected_var, None, [])?;
        let desired = self
            .b
            .bit_field_insert(u32_t, None, expected, value, bit_offset, bit_count)?;
        let actual = self.b.atomic_compare_exchange(
            u32_t,
            None,
            pointer,
            scope_device,
            relaxed,
            relaxed,
            desired,
            expected,
        )?;
        self.b.store(expected_var, actual, None, [])?;
        let stored = self.b.i_equal(self.u1, None, expected, actual)?;
        self.b
            .branch_conditional(stored, endloop_label, continue_label, [])?;
        self.begin_label(endloop_label)?;
        self.b.ret()?;
        self.begin_label(continue_label)?;
        self.b.branch(beginloop_label)?;
        self.b.end_function()?;
        self.write_storage_cas_loop_func = func;
        Ok(())
    }

    /// Base GPU address of storage buffer descriptor `index`, aligned down to the binding
    /// alignment as the renderer binds it.
    fn storage_base_address(&mut self, index: usize) -> EmitResult<Word> {
        let info = self.info;
        let align = self.profile.min_ssbo_alignment.max(1);
        let ssbo = info.storage_buffers_descriptors[index];
        let u64_t = self.u64;
        let u32_t = self.u32[1];
        let align_mask = self.const_u64(!(align - 1));
        let words = get_cbuf_u32x2_at(
            self,
            Value::U32(ssbo.cbuf_index),
            Value::U32(ssbo.cbuf_offset),
        )?;
        if ssbo.indirect_offset == StorageBufferDescriptor::DIRECT {
            let address = self.b.bitcast(u64_t, None, words)?;
            return Ok(self.b.bitwise_and(u64_t, None, address, align_mask)?);
        }
        let holder = info.storage_buffers_descriptors.iter().position(|d| {
            d.cbuf_index == ssbo.cbuf_index
                && d.cbuf_offset == ssbo.cbuf_offset
                && d.indirect_offset == StorageBufferDescriptor::DIRECT
        });
        let Some(holder) = holder else {
            return Ok(self.const_u64(u64::MAX));
        };
        let holder_low = self.b.composite_extract(u32_t, None, words, [0])?;
        let low_mask = self.const_u32((align - 1) as u32);
        let indirect = self.const_u32(ssbo.indirect_offset);
        let masked = self.b.bitwise_and(u32_t, None, holder_low, low_mask)?;
        let byte = self.b.i_add(u32_t, None, masked, indirect)?;
        let two = self.const_u32(2);
        let one = self.const_u32(1);
        let word_index = self.b.shift_right_logical(u32_t, None, byte, two)?;
        let next_index = self.b.i_add(u32_t, None, word_index, one)?;
        let storage = self.ssbo(holder, StorageElement::U32);
        let element = self.storage_type(StorageElement::U32).element;
        let zero = self.u32_zero_value;
        let low_pointer = self
            .b
            .access_chain(element, None, storage, [zero, word_index])?;
        let low = self.b.load(u32_t, None, low_pointer, None, [])?;
        let high_pointer = self
            .b
            .access_chain(element, None, storage, [zero, next_index])?;
        let high = self.b.load(u32_t, None, high_pointer, None, [])?;
        let pointer_words = self.b.composite_construct(self.u32[2], None, [low, high])?;
        let address = self.b.bitcast(u64_t, None, pointer_words)?;
        Ok(self.b.bitwise_and(u64_t, None, address, align_mask)?)
    }

    /// Opens the routing body of a global memory function: for every storage buffer that may
    /// hold `addr`, a selection whose then-block calls `access` with the element pointer, the
    /// buffer view and the element index. Leaves the block after the last selection open for
    /// the caller's fallback.
    fn route_global(
        &mut self,
        element: StorageElement,
        addr: Word,
        element_pointer: Word,
        shift: u32,
        access: &mut dyn FnMut(&mut Self, Word, Word, Word) -> EmitResult<()>,
    ) -> EmitResult<()> {
        let info = self.info;
        self.b.begin_block(None)?;
        let num_buffers = info.storage_buffers_descriptors.len();
        let (u1, u32_t, u64_t) = (self.u1, self.u32[1], self.u64);
        let mut route = |ctx: &mut Self, index: usize, base: Word, cond: Word| -> EmitResult<()> {
            let then_label = ctx.b.id();
            let else_label = ctx.b.id();
            ctx.b.selection_merge(else_label, SelectionControl::NONE)?;
            ctx.b.branch_conditional(cond, then_label, else_label, [])?;
            ctx.begin_label(then_label)?;
            let ssbo = ctx.ssbo(index, element);
            let distance = ctx.b.i_sub(u64_t, None, addr, base)?;
            let offset = ctx.b.u_convert(u32_t, None, distance)?;
            let shift = ctx.const_u32(shift);
            let ssbo_index = ctx.b.shift_right_logical(u32_t, None, offset, shift)?;
            let zero = ctx.u32_zero_value;
            let pointer = ctx
                .b
                .access_chain(element_pointer, None, ssbo, [zero, ssbo_index])?;
            access(ctx, pointer, ssbo, ssbo_index)?;
            ctx.begin_label(else_label)
        };
        if info.native_global_bounds {
            let mut bases = Vec::with_capacity(num_buffers);
            let mut best_index = self.const_u32(u32::MAX);
            let mut best_distance = self.const_u64(u64::MAX);
            for index in 0..num_buffers {
                let base = self.storage_base_address(index)?;
                bases.push(base);
                let distance = self.b.i_sub(u64_t, None, addr, base)?;
                let above = self.b.u_greater_than_equal(u1, None, addr, base)?;
                let closer = self.b.u_less_than(u1, None, distance, best_distance)?;
                let better = self.b.logical_and(u1, None, above, closer)?;
                best_distance = self
                    .b
                    .select(u64_t, None, better, distance, best_distance)?;
                let index_id = self.const_u32(index as u32);
                best_index = self.b.select(u32_t, None, better, index_id, best_index)?;
            }
            for (index, base) in bases.into_iter().enumerate() {
                let index_id = self.const_u32(index as u32);
                let cond = self.b.i_equal(u1, None, best_index, index_id)?;
                route(self, index, base, cond)?;
            }
            return Ok(());
        }
        for index in 0..num_buffers {
            let ssbo = info.storage_buffers_descriptors[index];
            if !uses_nvn_storage_descriptor(info, &ssbo) {
                continue;
            }
            let base = self.storage_base_address(index)?;
            let size = get_cbuf_u32_at(
                self,
                Value::U32(ssbo.cbuf_index),
                Value::U32(ssbo.cbuf_offset + 8),
            )?;
            let size = self.b.u_convert(u64_t, None, size)?;
            let end = self.b.i_add(u64_t, None, base, size)?;
            let above = self.b.u_greater_than_equal(u1, None, addr, base)?;
            let below = self.b.u_less_than(u1, None, addr, end)?;
            let cond = self.b.logical_and(u1, None, above, below)?;
            route(self, index, base, cond)?;
        }
        Ok(())
    }

    fn define_global_load(
        &mut self,
        element: StorageElement,
        element_pointer: Word,
        ty: Word,
        shift: u32,
    ) -> EmitResult<Word> {
        let u64_t = self.u64;
        let function_type = self.b.type_function(ty, [u64_t]);
        let func = self
            .b
            .begin_function(ty, None, FunctionControl::NONE, function_type)?;
        let addr = self.b.function_parameter(u64_t)?;
        let count = if shift > 2 { 1u32 << (shift - 2) } else { 1 };
        if !self.profile.support_descriptor_aliasing && shift > 2 {
            self.b.begin_block(None)?;
            let mut words = SmallVec::<[Word; 4]>::new();
            for i in 0..count {
                let offset = self.const_u64(u64::from(i) * 4);
                let word_address = self.b.i_add(u64_t, None, addr, offset)?;
                let load = self.load_global_func_u32;
                words.push(
                    self.b
                        .function_call(self.u32[1], None, load, [word_address])?,
                );
            }
            let result = self.b.composite_construct(ty, None, words)?;
            self.b.ret_value(result)?;
            self.b.end_function()?;
            return Ok(func);
        }
        if shift > 2 && self.profile.min_ssbo_alignment < (1u64 << shift) {
            let u32_element = self.storage_type(StorageElement::U32).element;
            self.route_global(
                StorageElement::U32,
                addr,
                u32_element,
                2,
                &mut |ctx, _, ssbo, word_index| {
                    let mut words = SmallVec::<[Word; 4]>::new();
                    let u32_t = ctx.u32[1];
                    let zero = ctx.u32_zero_value;
                    for i in 0..count {
                        let index = if i == 0 {
                            word_index
                        } else {
                            let i = ctx.const_u32(i);
                            ctx.b.i_add(u32_t, None, word_index, i)?
                        };
                        let pointer = ctx.b.access_chain(u32_element, None, ssbo, [zero, index])?;
                        words.push(ctx.b.load(u32_t, None, pointer, None, [])?);
                    }
                    let result = ctx.b.composite_construct(ty, None, words)?;
                    Ok(ctx.b.ret_value(result)?)
                },
            )?;
        } else {
            self.route_global(
                element,
                addr,
                element_pointer,
                shift,
                &mut |ctx, pointer, _, _| {
                    let value = ctx.b.load(ty, None, pointer, None, [])?;
                    Ok(ctx.b.ret_value(value)?)
                },
            )?;
        }
        let null = self.const_null(ty);
        self.b.ret_value(null)?;
        self.b.end_function()?;
        Ok(func)
    }

    fn define_global_write(
        &mut self,
        element: StorageElement,
        element_pointer: Word,
        ty: Word,
        shift: u32,
    ) -> EmitResult<Word> {
        let u64_t = self.u64;
        let function_type = self.b.type_function(self.void_id, [u64_t, ty]);
        let func =
            self.b
                .begin_function(self.void_id, None, FunctionControl::NONE, function_type)?;
        let addr = self.b.function_parameter(u64_t)?;
        let data = self.b.function_parameter(ty)?;
        let count = if shift > 2 { 1u32 << (shift - 2) } else { 1 };
        if !self.profile.support_descriptor_aliasing && shift > 2 {
            self.b.begin_block(None)?;
            for i in 0..count {
                let offset = self.const_u64(u64::from(i) * 4);
                let word_address = self.b.i_add(u64_t, None, addr, offset)?;
                let word = self.b.composite_extract(self.u32[1], None, data, [i])?;
                let write = self.write_global_func_u32;
                self.b
                    .function_call(self.void_id, None, write, [word_address, word])?;
            }
            self.b.ret()?;
            self.b.end_function()?;
            return Ok(func);
        }
        if shift > 2 && self.profile.min_ssbo_alignment < (1u64 << shift) {
            let u32_element = self.storage_type(StorageElement::U32).element;
            self.route_global(
                StorageElement::U32,
                addr,
                u32_element,
                2,
                &mut |ctx, _, ssbo, word_index| {
                    let u32_t = ctx.u32[1];
                    let zero = ctx.u32_zero_value;
                    for i in 0..count {
                        let index = if i == 0 {
                            word_index
                        } else {
                            let i = ctx.const_u32(i);
                            ctx.b.i_add(u32_t, None, word_index, i)?
                        };
                        let pointer = ctx.b.access_chain(u32_element, None, ssbo, [zero, index])?;
                        let word = ctx.b.composite_extract(u32_t, None, data, [i])?;
                        ctx.b.store(pointer, word, None, [])?;
                    }
                    Ok(ctx.b.ret()?)
                },
            )?;
        } else {
            self.route_global(
                element,
                addr,
                element_pointer,
                shift,
                &mut |ctx, pointer, _, _| {
                    ctx.b.store(pointer, data, None, [])?;
                    Ok(ctx.b.ret()?)
                },
            )?;
        }
        self.b.ret()?;
        self.b.end_function()?;
        Ok(func)
    }

    /// The functions that route untracked global loads and stores to the storage buffer
    /// holding the address at run time.
    fn define_global_memory_functions(&mut self) -> EmitResult<()> {
        let info = self.info;
        if !info.uses_global_memory || !self.profile.support_int64 {
            return Ok(());
        }
        let u32_def = self.storage_type(StorageElement::U32);
        let u32x2_def = self.storage_type(StorageElement::U32x2);
        let u32x4_def = self.storage_type(StorageElement::U32x4);
        let (u32_t, u32x2, u32x4) = (self.u32[1], self.u32[2], self.u32[4]);
        self.load_global_func_u32 =
            self.define_global_load(StorageElement::U32, u32_def.element, u32_t, 2)?;
        self.write_global_func_u32 =
            self.define_global_write(StorageElement::U32, u32_def.element, u32_t, 2)?;
        self.load_global_func_u32x2 =
            self.define_global_load(StorageElement::U32x2, u32x2_def.element, u32x2, 3)?;
        self.write_global_func_u32x2 =
            self.define_global_write(StorageElement::U32x2, u32x2_def.element, u32x2, 3)?;
        self.load_global_func_u32x4 =
            self.define_global_load(StorageElement::U32x4, u32x4_def.element, u32x4, 4)?;
        self.write_global_func_u32x4 =
            self.define_global_write(StorageElement::U32x4, u32x4_def.element, u32x4, 4)?;
        self.global_routing = info
            .storage_buffers_descriptors
            .iter()
            .any(|d| info.native_global_bounds || uses_nvn_storage_descriptor(info, d));
        Ok(())
    }

    /// Emits the bodies of the helper functions emission asked for by id while the entry point
    /// was open: the global atomic routing functions.
    pub(crate) fn define_deferred_functions(&mut self) -> EmitResult<()> {
        for kind in GlobalAtomic::ALL {
            let func = self.global_atomic_funcs[kind as usize];
            if func != 0 {
                self.define_global_atomic(kind, func)?;
            }
        }
        if self.atomic_cas_global_func_u32 != 0 {
            self.define_global_compare_exchange(self.atomic_cas_global_func_u32)?;
        }
        Ok(())
    }

    fn define_global_atomic(&mut self, kind: GlobalAtomic, func: Word) -> EmitResult<()> {
        let ty = if kind == GlobalAtomic::AddF32 {
            self.f32[1]
        } else {
            self.u32[1]
        };
        let u64_t = self.u64;
        let function_type = self.b.type_function(ty, [u64_t, ty]);
        self.b
            .begin_function(ty, Some(func), FunctionControl::NONE, function_type)?;
        let addr = self.b.function_parameter(u64_t)?;
        let value = self.b.function_parameter(ty)?;
        let element = self.storage_type(StorageElement::U32).element;
        let scope = self.const_u32(Scope::Device as u32);
        self.route_global(
            StorageElement::U32,
            addr,
            element,
            2,
            &mut |ctx, pointer, ssbo, word| {
                let u32_t = ctx.u32[1];
                let zero = ctx.u32_zero_value;
                let b = &mut ctx.b;
                let result = match kind {
                    GlobalAtomic::IAdd => {
                        b.atomic_i_add(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::SMin => {
                        b.atomic_s_min(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::UMin => {
                        b.atomic_u_min(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::SMax => {
                        b.atomic_s_max(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::UMax => {
                        b.atomic_u_max(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::And => b.atomic_and(u32_t, None, pointer, scope, zero, value)?,
                    GlobalAtomic::Or => b.atomic_or(u32_t, None, pointer, scope, zero, value)?,
                    GlobalAtomic::Xor => b.atomic_xor(u32_t, None, pointer, scope, zero, value)?,
                    GlobalAtomic::Exchange => {
                        b.atomic_exchange(u32_t, None, pointer, scope, zero, value)?
                    }
                    GlobalAtomic::Inc => {
                        let f = ctx.increment_cas_ssbo;
                        ctx.b.function_call(u32_t, None, f, [word, value, ssbo])?
                    }
                    GlobalAtomic::Dec => {
                        let f = ctx.decrement_cas_ssbo;
                        ctx.b.function_call(u32_t, None, f, [word, value, ssbo])?
                    }
                    GlobalAtomic::AddF32 => {
                        let f = ctx.f32_add_cas;
                        ctx.b.function_call(ty, None, f, [word, value, ssbo])?
                    }
                };
                Ok(ctx.b.ret_value(result)?)
            },
        )?;
        let null = self.const_null(ty);
        self.b.ret_value(null)?;
        self.b.end_function()?;
        Ok(())
    }

    fn define_global_compare_exchange(&mut self, func: Word) -> EmitResult<()> {
        let (u32_t, u64_t) = (self.u32[1], self.u64);
        let function_type = self.b.type_function(u32_t, [u64_t, u32_t, u32_t]);
        self.b
            .begin_function(u32_t, Some(func), FunctionControl::NONE, function_type)?;
        let address = self.b.function_parameter(u64_t)?;
        let comparator = self.b.function_parameter(u32_t)?;
        let value = self.b.function_parameter(u32_t)?;
        let element = self.storage_type(StorageElement::U32).element;
        let scope = self.const_u32(Scope::Device as u32);
        self.route_global(
            StorageElement::U32,
            address,
            element,
            2,
            &mut |ctx, pointer, _, _| {
                let zero = ctx.u32_zero_value;
                let result = ctx.b.atomic_compare_exchange(
                    u32_t, None, pointer, scope, zero, zero, value, comparator,
                )?;
                Ok(ctx.b.ret_value(result)?)
            },
        )?;
        self.b.ret_value(comparator)?;
        self.b.end_function()?;
        Ok(())
    }

    fn define_rescaling_input(&mut self) {
        if !self.info.uses_rescaling_uniform {
            return;
        }
        if self.profile.unified_descriptor_binding {
            self.define_rescaling_push_constant();
        } else {
            let pointer = self.type_pointer(StorageClass::UniformConstant, self.f32[4]);
            self.rescaling_uniform_constant =
                self.global_variable(pointer, StorageClass::UniformConstant, None);
            self.decorate(self.rescaling_uniform_constant, Decoration::Location, &[0]);
            self.add_interface(
                self.rescaling_uniform_constant,
                StorageClass::UniformConstant,
            );
        }
    }

    fn define_rescaling_push_constant(&mut self) {
        let u32_t = self.u32[1];
        self.rescaling_textures_type = self.type_array(u32_t, NUM_TEXTURE_SCALING_WORDS);
        self.decorate(self.rescaling_textures_type, Decoration::ArrayStride, &[4]);
        self.rescaling_images_type = self.type_array(u32_t, NUM_IMAGE_SCALING_WORDS);
        self.decorate(self.rescaling_images_type, Decoration::ArrayStride, &[4]);
        let mut members = SmallVec::<[Word; 3]>::new();
        members.push(self.rescaling_textures_type);
        members.push(self.rescaling_images_type);
        self.rescaling_textures_member_index = 0;
        self.rescaling_images_member_index = 1;
        let graphics = self.stage != ShaderStage::Compute;
        if graphics {
            members.push(self.f32[1]);
            self.rescaling_downfactor_member_index = 2;
        }
        let block = self.b.type_struct(members);
        self.decorate(block, Decoration::Block, &[]);
        self.b.name(block, "ResolutionInfo");
        self.member_decorate(
            block,
            0,
            Decoration::Offset,
            &[RESCALING_LAYOUT_TEXTURES_OFFSET],
        );
        self.b.member_name(block, 0, "rescaling_textures");
        self.member_decorate(
            block,
            1,
            Decoration::Offset,
            &[RESCALING_LAYOUT_IMAGES_OFFSET],
        );
        self.b.member_name(block, 1, "rescaling_images");
        if graphics {
            self.member_decorate(
                block,
                2,
                Decoration::Offset,
                &[RESCALING_LAYOUT_DOWN_FACTOR_OFFSET],
            );
            self.b.member_name(block, 2, "down_factor");
        }
        let pointer = self.type_pointer(StorageClass::PushConstant, block);
        self.rescaling_push_constants =
            self.global_variable(pointer, StorageClass::PushConstant, None);
        self.b
            .name(self.rescaling_push_constants, "rescaling_push_constants");
        self.add_interface(self.rescaling_push_constants, StorageClass::PushConstant);
    }

    fn define_render_area(&mut self) {
        if !self.info.uses_render_area || !self.profile.unified_descriptor_binding {
            return;
        }
        let block = self.b.type_struct([self.f32[4]]);
        self.decorate(block, Decoration::Block, &[]);
        self.b.name(block, "RenderAreaInfo");
        self.render_area_member_index = 0;
        self.member_decorate(block, 0, Decoration::Offset, &[RENDER_AREA_LAYOUT_OFFSET]);
        self.b.member_name(block, 0, "render_area");
        let pointer = self.type_pointer(StorageClass::PushConstant, block);
        self.render_area_push_constant =
            self.global_variable(pointer, StorageClass::PushConstant, None);
        self.b
            .name(self.render_area_push_constant, "render_area_push_constants");
        self.add_interface(self.render_area_push_constant, StorageClass::PushConstant);
    }
}
