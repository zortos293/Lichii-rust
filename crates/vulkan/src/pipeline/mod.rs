//! Graphics and compute pipelines: Maxwell state translated to Vulkan, the fixed-function
//! pipeline key extracted from the 3D registers, and a cache that builds pipelines on worker
//! threads and keeps the driver's compiled pipelines on disk.
//!
//! Shader identities come from the shader translator: a [`ShaderModule`] is one host stage's
//! SPIR-V plus the resources it binds, under a hash that stays the same across runs for the
//! same guest program, so the disk cache can name pipelines by their stages.

pub mod cache;
mod disk;
pub mod environment;
pub mod fixed_state;
pub mod key;
pub mod maxwell_to_vk;
pub mod sampler;
pub mod shader_cache;
mod shader_disk;

use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use ash::vk;
use lichii_shader::{DescriptorKind, ShaderModule, ShaderStage};

pub use cache::{LayoutBinding, Pipeline, PipelineCache, PipelineLayout, PipelineStats};
pub use fixed_state::{FixedPipelineState, FixedStateTracker};
pub use key::{AttachmentState, ComputePipelineKey, GraphicsPipelineKey};

use crate::device::Device;

/// Host stages of a graphics pipeline in pipeline order; [`GraphicsShaders`] and the shader
/// hashes in [`GraphicsPipelineKey`] are indexed the same way.
pub const GRAPHICS_STAGES: [vk::ShaderStageFlags; 5] = [
    vk::ShaderStageFlags::VERTEX,
    vk::ShaderStageFlags::TESSELLATION_CONTROL,
    vk::ShaderStageFlags::TESSELLATION_EVALUATION,
    vk::ShaderStageFlags::GEOMETRY,
    vk::ShaderStageFlags::FRAGMENT,
];

/// The programs of one graphics pipeline, one slot per [`GRAPHICS_STAGES`] entry.
pub type GraphicsShaders = [Option<Arc<ShaderModule>>; 5];

/// The Vulkan stage a translated module runs in; VertexA never reaches the host on its own.
pub fn stage_flags(stage: ShaderStage) -> vk::ShaderStageFlags {
    match stage {
        ShaderStage::VertexA | ShaderStage::VertexB => vk::ShaderStageFlags::VERTEX,
        ShaderStage::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderStage::TessellationEval => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderStage::Geometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderStage::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderStage::Compute => vk::ShaderStageFlags::COMPUTE,
        ShaderStage::Task => vk::ShaderStageFlags::TASK_EXT,
        ShaderStage::Mesh => vk::ShaderStageFlags::MESH_EXT,
    }
}

pub fn descriptor_type(kind: DescriptorKind) -> vk::DescriptorType {
    match kind {
        DescriptorKind::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
        DescriptorKind::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        DescriptorKind::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        DescriptorKind::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        DescriptorKind::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        DescriptorKind::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
    }
}

/// What the device lets the pipeline leave to dynamic state, and the optional fixed-function
/// features a static pipeline may use. Key extraction drops dynamic fields from the key and
/// clears state the device cannot express, so the key holds exactly what pipelines are built
/// with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineFeatures {
    /// Cull mode, front face, topology (within its class), viewport and scissor counts, vertex
    /// strides, depth and stencil test state (VK_EXT_extended_dynamic_state, core in 1.3).
    pub extended_dynamic_state: bool,
    /// Rasterizer discard, depth bias enable and primitive restart (core in 1.3).
    pub extended_dynamic_state2: bool,
    /// VK_EXT_vertex_input_dynamic_state: attributes, bindings and strides.
    pub dynamic_vertex_input: bool,
    /// VK_EXT_extended_dynamic_state3 colour blend enable, equation and write mask.
    pub dynamic_blend: bool,
    pub dynamic_depth_clamp: bool,
    pub dynamic_logic_op_enable: bool,
    pub dynamic_alpha_to_coverage: bool,
    pub dynamic_alpha_to_one: bool,
    pub dynamic_polygon_mode: bool,
    pub dynamic_depth_clip_negative_one_to_one: bool,
    pub dynamic_provoking_vertex: bool,
    pub depth_clamp: bool,
    pub depth_bounds: bool,
    pub logic_op: bool,
    pub alpha_to_one: bool,
    pub fill_mode_non_solid: bool,
    pub dual_src_blend: bool,
    pub depth_clip_control: bool,
    pub provoking_vertex_last: bool,
    pub multi_viewport: bool,
    /// VK_EXT_graphics_pipeline_library with fast linking: while a pipeline builds, draws use
    /// one linked from per-part libraries.
    pub fast_link: bool,
}

impl PipelineFeatures {
    /// Everything the device supports, which the device also enabled at creation.
    pub fn from_device(device: &Device) -> Self {
        let raw = &device.instance.raw;
        // SAFETY: `physical` belongs to this instance.
        let core = unsafe { raw.get_physical_device_features(device.physical) };
        let mut eds3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
        if device.features.extensions.extended_dynamic_state3 {
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut eds3);
            // SAFETY: the device reports VK_EXT_extended_dynamic_state3.
            unsafe { raw.get_physical_device_features2(device.physical, &mut features) };
        }
        let on = |bit: vk::Bool32| bit != vk::FALSE;
        let f = &device.features;
        Self {
            extended_dynamic_state: true,
            extended_dynamic_state2: true,
            dynamic_vertex_input: f.vertex_input_dynamic_state,
            dynamic_blend: on(eds3.extended_dynamic_state3_color_blend_enable)
                && on(eds3.extended_dynamic_state3_color_blend_equation)
                && on(eds3.extended_dynamic_state3_color_write_mask),
            dynamic_depth_clamp: on(eds3.extended_dynamic_state3_depth_clamp_enable),
            dynamic_logic_op_enable: on(eds3.extended_dynamic_state3_logic_op_enable),
            dynamic_alpha_to_coverage: on(eds3.extended_dynamic_state3_alpha_to_coverage_enable),
            dynamic_alpha_to_one: on(eds3.extended_dynamic_state3_alpha_to_one_enable),
            dynamic_polygon_mode: on(eds3.extended_dynamic_state3_polygon_mode),
            dynamic_depth_clip_negative_one_to_one: f.depth_clip_control
                && on(eds3.extended_dynamic_state3_depth_clip_negative_one_to_one),
            dynamic_provoking_vertex: f.extensions.provoking_vertex
                && on(eds3.extended_dynamic_state3_provoking_vertex_mode),
            depth_clamp: on(core.depth_clamp),
            depth_bounds: on(core.depth_bounds),
            logic_op: on(core.logic_op),
            alpha_to_one: on(core.alpha_to_one),
            fill_mode_non_solid: on(core.fill_mode_non_solid),
            dual_src_blend: f.dual_src_blend,
            depth_clip_control: f.depth_clip_control,
            provoking_vertex_last: f.provoking_vertex_last,
            multi_viewport: f.multi_viewport,
            fast_link: f.graphics_pipeline_library && device.properties.fast_pipeline_linking,
        }
    }

    /// The dynamic states pipelines built with these features declare; the rasterizer sets
    /// each of them before drawing. `mesh` pipelines have no vertex input or input assembly
    /// state, so they leave out the dynamic states that belong to those.
    pub fn dynamic_states(&self, mesh: bool) -> Vec<vk::DynamicState> {
        use vk::DynamicState as D;
        let mut states = vec![
            D::LINE_WIDTH,
            D::DEPTH_BIAS,
            D::BLEND_CONSTANTS,
            D::DEPTH_BOUNDS,
            D::STENCIL_COMPARE_MASK,
            D::STENCIL_WRITE_MASK,
            D::STENCIL_REFERENCE,
        ];
        if self.extended_dynamic_state {
            states.extend([
                D::VIEWPORT_WITH_COUNT,
                D::SCISSOR_WITH_COUNT,
                D::CULL_MODE,
                D::FRONT_FACE,
                D::DEPTH_TEST_ENABLE,
                D::DEPTH_WRITE_ENABLE,
                D::DEPTH_COMPARE_OP,
                D::DEPTH_BOUNDS_TEST_ENABLE,
                D::STENCIL_TEST_ENABLE,
                D::STENCIL_OP,
            ]);
            if !mesh {
                states.push(D::PRIMITIVE_TOPOLOGY);
            }
            if !mesh && !self.dynamic_vertex_input {
                states.push(D::VERTEX_INPUT_BINDING_STRIDE);
            }
        } else {
            states.extend([D::VIEWPORT, D::SCISSOR]);
        }
        if self.extended_dynamic_state2 {
            states.extend([D::RASTERIZER_DISCARD_ENABLE, D::DEPTH_BIAS_ENABLE]);
            if !mesh {
                states.push(D::PRIMITIVE_RESTART_ENABLE);
            }
        }
        for (enabled, dynamic) in [
            (
                self.dynamic_vertex_input && !mesh,
                &[D::VERTEX_INPUT_EXT][..],
            ),
            (
                self.dynamic_blend,
                &[
                    D::COLOR_BLEND_ENABLE_EXT,
                    D::COLOR_BLEND_EQUATION_EXT,
                    D::COLOR_WRITE_MASK_EXT,
                ][..],
            ),
            (self.dynamic_depth_clamp, &[D::DEPTH_CLAMP_ENABLE_EXT][..]),
            (self.dynamic_logic_op_enable, &[D::LOGIC_OP_ENABLE_EXT][..]),
            (
                self.dynamic_alpha_to_coverage,
                &[D::ALPHA_TO_COVERAGE_ENABLE_EXT][..],
            ),
            (self.dynamic_alpha_to_one, &[D::ALPHA_TO_ONE_ENABLE_EXT][..]),
            (self.dynamic_polygon_mode, &[D::POLYGON_MODE_EXT][..]),
            (
                self.dynamic_depth_clip_negative_one_to_one,
                &[D::DEPTH_CLIP_NEGATIVE_ONE_TO_ONE_EXT][..],
            ),
            (
                self.dynamic_provoking_vertex,
                &[D::PROVOKING_VERTEX_MODE_EXT][..],
            ),
        ] {
            if enabled {
                states.extend_from_slice(dynamic);
            }
        }
        states
    }
}

/// FxHash over 8-byte words: deterministic across runs (the disk cache checksums with it) and
/// a few cycles per word for the per-draw key lookup.
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher(u64);

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (chunks, rest) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.add(u64::from_le_bytes(*chunk));
        }
        if !rest.is_empty() {
            let mut last = [0; 8];
            last[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(last));
        }
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

pub(crate) fn fx_hash(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(bytes);
    hasher.finish()
}
