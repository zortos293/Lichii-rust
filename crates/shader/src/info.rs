//! What a translated program uses, so the renderer can build descriptor set layouts, vertex
//! input state and pipeline state. Filled by the collect-info, storage buffer and texture passes.

use crate::handle::HandleProgram;
use crate::ir::{Attribute, ImageFormat, TextureType, Type};

/// One bit per attribute (indexed by `Attribute.0`).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct VaryingState(pub [u64; 8]);

impl VaryingState {
    pub fn set(&mut self, attribute: Attribute, state: bool) {
        let (word, bit) = (attribute.0 as usize / 64, attribute.0 % 64);
        if word >= self.0.len() {
            return;
        }
        if state {
            self.0[word] |= 1 << bit;
        } else {
            self.0[word] &= !(1 << bit);
        }
    }

    pub fn get(&self, attribute: Attribute) -> bool {
        let (word, bit) = (attribute.0 as usize / 64, attribute.0 % 64);
        self.0.get(word).is_some_and(|w| w & (1 << bit) != 0)
    }

    pub fn any_component(&self, base: Attribute) -> bool {
        (0..4).any(|i| self.get(base.offset(i)))
    }

    pub fn all_components(&self, base: Attribute) -> bool {
        (0..4).all(|i| self.get(base.offset(i)))
    }

    pub fn generic(&self, index: u16) -> bool {
        self.any_component(Attribute::generic(index, 0))
    }

    pub fn generic_component(&self, index: u16, component: u16) -> bool {
        self.get(Attribute::generic(index, component))
    }

    pub fn clip_distances(&self) -> bool {
        self.any_component(Attribute::CLIP_DISTANCE_0)
            || self.any_component(Attribute::CLIP_DISTANCE_0.offset(4))
    }

    pub fn fixed_function_texture(&self) -> bool {
        (0..Attribute::NUM_FIXED_FNC_TEXTURES)
            .any(|i| self.any_component(Attribute::FIXED_FNC_TEXTURE_0_S.offset(i * 4)))
    }

    pub fn legacy(&self) -> bool {
        self.any_component(Attribute::COLOR_FRONT_DIFFUSE_R)
            || self.any_component(Attribute::COLOR_FRONT_SPECULAR_R)
            || self.any_component(Attribute::COLOR_BACK_DIFFUSE_R)
            || self.any_component(Attribute::COLOR_BACK_SPECULAR_R)
            || self.fixed_function_texture()
            || self.get(Attribute::FOG_COORDINATE)
    }
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Interpolation {
    #[default]
    Smooth,
    Flat,
    NoPerspective,
}

/// A guest constant buffer bound as a uniform buffer.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct ConstantBufferDescriptor {
    pub index: u32,
    pub count: u32,
}

/// A guest global memory region bound as a storage buffer. Its GPU address and size are the
/// 64-bit address and 32-bit size at `cbuf_offset` of constant buffer `cbuf_index`.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct StorageBufferDescriptor {
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub count: u32,
    pub is_written: bool,
    /// Offset of the buffer pointer in the memory the constant buffer pointer points to, or
    /// [`StorageBufferDescriptor::DIRECT`] when the constant buffer holds the pointer itself.
    pub indirect_offset: u32,
    /// How far the program's stores reach past the pointer; unbounded when unknown or unwritten.
    pub write_extent: StorageWriteExtent,
}

impl StorageBufferDescriptor {
    /// `indirect_offset` of a pointer read straight from the constant buffer.
    pub const DIRECT: u32 = u32::MAX;
}

/// How far a program's stores and atomics to one storage buffer can reach, in bytes past the
/// pointer it reads from its constant buffer. Every store ends at or below
/// `const_end + sum(local_stride[i] * local_id[i]) + sum(group_stride[i] * group_id[i])` for every
/// invocation. The bound only holds while each value the store offsets split into 16-bit halves
/// stays below 2^16: `guard_*` bounds the largest of them in the same form.
///
/// A Switch 2 native program's storage pointer carries no size, so the renderer binds a large
/// range behind it; this is what it can claim a dispatch wrote instead of all of it.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct StorageWriteExtent {
    /// False: the reach is not known (or the buffer is never written); take the whole binding.
    pub bounded: bool,
    /// Whether the bound needs the guard.
    pub guarded: bool,
    pub const_end: u32,
    pub local_stride: [u32; 3],
    pub group_stride: [u32; 3],
    pub guard_const: u32,
    pub guard_local_stride: [u32; 3],
    pub guard_group_stride: [u32; 3],
}

impl StorageWriteExtent {
    /// Widens this extent to also cover `other`; either one unbounded leaves it unbounded.
    pub fn merge(&mut self, other: &StorageWriteExtent) {
        if !self.bounded || !other.bounded {
            *self = StorageWriteExtent::default();
            return;
        }
        let max3 = |lhs: &mut [u32; 3], rhs: &[u32; 3]| {
            for (l, r) in lhs.iter_mut().zip(rhs) {
                *l = (*l).max(*r);
            }
        };
        self.const_end = self.const_end.max(other.const_end);
        max3(&mut self.local_stride, &other.local_stride);
        max3(&mut self.group_stride, &other.group_stride);
        if !other.guarded {
            return;
        }
        if self.guarded {
            self.guard_const = self.guard_const.max(other.guard_const);
            max3(&mut self.guard_local_stride, &other.guard_local_stride);
            max3(&mut self.guard_group_stride, &other.guard_group_stride);
        } else {
            self.guarded = true;
            self.guard_const = other.guard_const;
            self.guard_local_stride = other.guard_local_stride;
            self.guard_group_stride = other.guard_group_stride;
        }
    }
}

/// Where a texture handle comes from: the word at `cbuf_offset` of constant buffer
/// `cbuf_index`, shifted left by `shift_left`, ORed with an optional secondary word (separate
/// sampler handles).
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    serde::Serialize,
    serde::Deserialize,
)]
pub struct HandleSource {
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub shift_left: u32,
    pub secondary: Option<(u32, u32, u32)>,
}

impl HandleSource {
    /// The raw handle the words hold, read through `read_cbuf(index, offset)`: the primary word
    /// shifted, ORed with the secondary word shifted (with no secondary, the primary word
    /// itself, as the C++ does).
    pub fn handle(&self, mut read_cbuf: impl FnMut(u32, u32) -> u32) -> u32 {
        let lhs = read_cbuf(self.cbuf_index, self.cbuf_offset);
        let (rhs, shift) = match self.secondary {
            Some((index, offset, shift)) => (read_cbuf(index, offset), shift),
            None => (lhs, 0),
        };
        lhs.checked_shl(self.shift_left).unwrap_or(0) | rhs.checked_shl(shift).unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TextureBufferDescriptor {
    pub handle: HandleSource,
    /// Array length; consecutive handles are `1 << size_shift` bytes apart.
    pub count: u32,
    pub size_shift: u32,
    /// Index in [`ShaderInfo::handle_programs`] of the program computing the handle for each
    /// draw (a handle read from a storage buffer); `None` when the handle is the constant buffer
    /// word above.
    pub handle_program: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImageBufferDescriptor {
    pub format: ImageFormat,
    pub is_written: bool,
    pub is_read: bool,
    pub is_integer: bool,
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub count: u32,
    pub size_shift: u32,
    /// Index in [`ShaderInfo::handle_programs`] of the program computing the handle for each
    /// draw (a handle read from a storage buffer); `None` when the handle is the constant buffer
    /// word above.
    pub handle_program: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct TextureDescriptor {
    pub ty: TextureType,
    pub is_depth: bool,
    pub is_multisample: bool,
    pub is_integer: bool,
    pub handle: HandleSource,
    pub count: u32,
    pub size_shift: u32,
    /// The shader indexes the descriptor, so the descriptor limits may shrink `count`; it stays
    /// set when they cut the array to one element.
    pub is_descriptor_array: bool,
    /// The renderer binds no texture: the handle has no value a draw can compute (the shader
    /// changes it in a loop, or reads it per invocation from memory no table describes).
    pub handle_is_pointer_word: bool,
    /// Index in [`ShaderInfo::handle_programs`] of the program computing the handle for each
    /// draw (a handle read from a storage buffer); `None` when the handle is the constant buffer
    /// word above. With a `count` above 1 the program computes the handle of each element
    /// ([`crate::handle::HandleOp::Index`]): a table entry the shader picks per invocation.
    /// No texture header is readable when the shader compiles, so such a descriptor's `ty` is
    /// the type its instructions name (else 2D) and `is_integer` is false; bind the null
    /// texture for a handle whose image differs in either.
    pub handle_program: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ImageDescriptor {
    pub ty: TextureType,
    pub format: ImageFormat,
    pub is_written: bool,
    pub is_read: bool,
    pub is_integer: bool,
    pub cbuf_index: u32,
    pub cbuf_offset: u32,
    pub count: u32,
    pub size_shift: u32,
    /// Stores address X in bytes; the program writes 32-bit texels at `x >> 2`.
    pub is_byte_addressed: bool,
    /// Index in [`ShaderInfo::handle_programs`] of the program computing the handle for each
    /// draw (a handle read from a storage buffer); `None` when the handle is the constant buffer
    /// word above.
    pub handle_program: Option<u32>,
}

/// Task and mesh program layout (Switch 2 NVN task/mesh, see NX2_NOTES row 89).
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MeshInfo {
    /// Task output / mesh input region, in 32-bit words.
    pub payload_words: u32,
    pub max_vertices: u32,
    pub max_primitives: u32,
    /// SET_MESH_SHADER_A output topology (0 points, 1 lines, 4 triangles).
    pub topology: u32,
    pub vertex_lines: Vec<u32>,
    pub primitive_lines: Vec<u32>,
}

impl MeshInfo {
    pub fn primitive_vertices(&self) -> u32 {
        match self.topology {
            0 => 1,
            1 => 2,
            _ => 3,
        }
    }
}

/// Everything the renderer needs to know about a translated program.
///
/// Descriptor order is binding order: uniform buffers, storage buffers, texture buffers, image
/// buffers, textures, images, each in the order of its vector.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ShaderInfo {
    /// The module holds the program's absolute address (Ampere LEPC), so it must not be reused
    /// for the same code at another address; [`crate::ShaderModule::hash`] covers the address.
    pub reads_start_address: bool,
    pub uses_workgroup_id: bool,
    pub uses_local_invocation_id: bool,
    pub uses_invocation_id: bool,
    pub uses_invocation_info: bool,
    pub uses_sample_id: bool,
    pub uses_is_helper_invocation: bool,
    pub uses_subgroup_invocation_id: bool,
    pub uses_subgroup_shuffles: bool,
    pub uses_quad_shuffles: bool,
    pub uses_patches: [bool; 30],

    pub interpolation: [Interpolation; 32],
    pub loads: VaryingState,
    pub stores: VaryingState,
    pub passthrough: VaryingState,
    /// Legacy attributes (colours, fog, fixed function textures) rewritten to generics.
    pub legacy_stores_mapping: Vec<(Attribute, Attribute)>,

    pub loads_indexed_attributes: bool,
    pub stores_frag_color: [bool; 8],
    pub frag_color_components: [u8; 8],
    pub stores_sample_mask: bool,
    pub stores_frag_depth: bool,
    pub stores_tess_level_outer: bool,
    pub stores_tess_level_inner: bool,
    pub stores_indexed_attributes: bool,
    pub stores_global_memory: bool,

    pub uses_local_memory: bool,
    pub uses_fp16: bool,
    pub uses_fp64: bool,
    pub uses_fp16_denorms_flush: bool,
    pub uses_fp16_denorms_preserve: bool,
    pub uses_fp32_denorms_flush: bool,
    pub uses_fp32_denorms_preserve: bool,
    pub uses_int8: bool,
    pub uses_int16: bool,
    pub uses_int64: bool,
    pub uses_image_1d: bool,
    pub uses_sampled_1d: bool,
    pub uses_sparse_residency: bool,
    pub uses_demote_to_helper_invocation: bool,
    pub uses_subgroup_vote: bool,
    pub uses_subgroup_mask: bool,
    pub uses_fswzadd: bool,
    pub uses_derivatives: bool,
    pub uses_typeless_image_reads: bool,
    pub uses_typeless_image_writes: bool,
    pub uses_image_buffers: bool,
    pub uses_shared_increment: bool,
    pub uses_shared_decrement: bool,
    pub uses_global_increment: bool,
    pub uses_global_decrement: bool,
    pub uses_atomic_f32_add: bool,
    pub uses_atomic_f16x2_add: bool,
    pub uses_atomic_f16x2_min: bool,
    pub uses_atomic_f16x2_max: bool,
    pub uses_atomic_f32x2_add: bool,
    pub uses_atomic_f32x2_min: bool,
    pub uses_atomic_f32x2_max: bool,
    pub uses_atomic_s32_min: bool,
    pub uses_atomic_s32_max: bool,
    pub uses_int64_bit_atomics: bool,
    pub uses_global_memory: bool,
    pub uses_atomic_image_u32: bool,
    pub uses_shadow_lod: bool,
    pub uses_rescaling_uniform: bool,
    pub uses_cbuf_indirect: bool,
    pub uses_render_area: bool,
    pub removed_conditional_barrier: bool,
    /// Some global accesses stayed global with several bound candidate buffers; the backend
    /// picks the buffer at run time from the bound pointers (Switch 2 native programs).
    pub native_global_bounds: bool,

    pub used_constant_buffer_types: Type,
    pub used_storage_buffer_types: Type,
    pub used_indirect_cbuf_types: Type,
    /// Bit `i` is set when constant buffer `i` is read.
    pub constant_buffer_mask: u32,
    /// Bytes read from each constant buffer, rounded up.
    pub constant_buffer_used_sizes: [u32; ShaderInfo::MAX_CBUFS],
    /// NVN driver storage buffer slots (constant buffer 0) the program may address.
    pub nvn_buffer_used: u16,
    pub requires_layer_emulation: bool,
    pub emulated_layer: Option<Attribute>,
    pub used_clip_distances: u32,

    pub constant_buffer_descriptors: Vec<ConstantBufferDescriptor>,
    pub storage_buffers_descriptors: Vec<StorageBufferDescriptor>,
    pub texture_buffer_descriptors: Vec<TextureBufferDescriptor>,
    pub image_buffer_descriptors: Vec<ImageBufferDescriptor>,
    pub texture_descriptors: Vec<TextureDescriptor>,
    /// Texture handles the shader reads from storage buffers, computed for each draw; the
    /// descriptors name them by index.
    pub handle_programs: Vec<HandleProgram>,
    pub image_descriptors: Vec<ImageDescriptor>,

    pub mesh: MeshInfo,

    /// Byte offset in constant buffer 0 of the NVN storage buffer descriptors for this stage.
    pub nvn_buffer_base: u32,
    pub requires_coherent_global_memory: bool,
    pub requires_coherent_image_memory: bool,
}

impl ShaderInfo {
    pub const MAX_CBUFS: usize = 18;
    pub const MAX_INDIRECT_CBUFS: usize = 14;
    pub const MAX_SSBOS: usize = 64;
}

/// Pixel format of a texture descriptor (TIC), as far as translation cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TexturePixelFormat {
    A8b8g8r8Unorm,
    A8b8g8r8Snorm,
    A8b8g8r8Sint,
    A8b8g8r8Uint,
    R5g6b5Unorm,
    B5g6r5Unorm,
    A1r5g5b5Unorm,
    A2b10g10r10Unorm,
    A2b10g10r10Uint,
    A2r10g10b10Unorm,
    A1b5g5r5Unorm,
    A5b5g5r1Unorm,
    R8Unorm,
    R8Snorm,
    R8Sint,
    R8Uint,
    R16g16b16a16Float,
    R16g16b16a16Unorm,
    R16g16b16a16Snorm,
    R16g16b16a16Sint,
    R16g16b16a16Uint,
    B10g11r11Float,
    R32g32b32a32Uint,
    Bc1RgbaUnorm,
    Bc2Unorm,
    Bc3Unorm,
    Bc4Unorm,
    Bc4Snorm,
    Bc5Unorm,
    Bc5Snorm,
    Bc7Unorm,
    Bc6hUfloat,
    Bc6hSfloat,
    Astc2d4x4Unorm,
    B8g8r8a8Unorm,
    R32g32b32a32Float,
    R32g32b32a32Sint,
    R32g32Float,
    R32g32Sint,
    R32Float,
    R16Float,
    R16Unorm,
    R16Snorm,
    R16Uint,
    R16Sint,
    R16g16Unorm,
    R16g16Float,
    R16g16Uint,
    R16g16Sint,
    R16g16Snorm,
    R32g32b32Float,
    A8b8g8r8Srgb,
    R8g8Unorm,
    R8g8Snorm,
    R8g8Sint,
    R8g8Uint,
    R32g32Uint,
    R16g16b16x16Float,
    R32Uint,
    R32Sint,
    Astc2d8x8Unorm,
    Astc2d8x5Unorm,
    Astc2d5x4Unorm,
    B8g8r8a8Srgb,
    Bc1RgbaSrgb,
    Bc2Srgb,
    Bc3Srgb,
    Bc7Srgb,
    A4b4g4r4Unorm,
    G4r4Unorm,
    Astc2d4x4Srgb,
    Astc2d8x8Srgb,
    Astc2d8x5Srgb,
    Astc2d5x4Srgb,
    Astc2d5x5Unorm,
    Astc2d5x5Srgb,
    Astc2d10x8Unorm,
    Astc2d10x8Srgb,
    Astc2d6x6Unorm,
    Astc2d6x6Srgb,
    Astc2d10x6Unorm,
    Astc2d10x6Srgb,
    Astc2d10x5Unorm,
    Astc2d10x5Srgb,
    Astc2d10x10Unorm,
    Astc2d10x10Srgb,
    Astc2d12x10Unorm,
    Astc2d12x10Srgb,
    Astc2d12x12Unorm,
    Astc2d12x12Srgb,
    Astc2d8x6Unorm,
    Astc2d8x6Srgb,
    Astc2d6x5Unorm,
    Astc2d6x5Srgb,
    E5b9g9r9Float,
    D32Float,
    D16Unorm,
    X8D24Unorm,
    S8Uint,
    D24UnormS8Uint,
    S8UintD24Unorm,
    D32FloatS8Uint,
    Etc2RgbUnorm,
    Etc2RgbSrgb,
    Etc2RgbaUnorm,
    Etc2RgbaSrgb,
    Etc2RgbPtaUnorm,
    Etc2RgbPtaSrgb,
    EacR11Unorm,
    EacR11Snorm,
    EacR11G11Unorm,
    EacR11G11Snorm,
}

/// Draw parameter a constant buffer word holds when HLE macros replaced it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReplaceConstant {
    BaseInstance,
    BaseVertex,
    DrawId,
}
