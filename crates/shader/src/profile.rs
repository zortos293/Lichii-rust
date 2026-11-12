//! Host capabilities. Translation reads the fields that change the IR (see
//! [`Profile::translation_key`]); SPIR-V emission reads the rest.

/// What the host Vulkan device supports. Fill it once per device.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Profile {
    /// Highest SPIR-V version, as `0x00MMmm00` (Vulkan 1.3 is 0x00010600).
    pub supported_spirv: u32,
    pub unified_descriptor_binding: bool,

    pub support_int8: bool,
    pub support_int16: bool,
    pub support_int64: bool,
    pub support_float16: bool,
    pub support_float64: bool,
    pub support_storage_buffer_8bit: bool,
    pub support_uniform_and_storage_buffer_8bit: bool,
    pub support_storage_buffer_16bit: bool,
    pub support_uniform_and_storage_buffer_16bit: bool,

    pub support_float_controls: bool,
    pub support_separate_denorm_behavior: bool,
    pub support_separate_rounding_mode: bool,
    pub support_fp16_denorm_preserve: bool,
    pub support_fp32_denorm_preserve: bool,
    pub support_fp16_denorm_flush: bool,
    pub support_fp32_denorm_flush: bool,
    pub support_fp16_signed_zero_nan_preserve: bool,
    pub support_fp32_signed_zero_nan_preserve: bool,
    pub support_fp64_signed_zero_nan_preserve: bool,

    pub support_explicit_workgroup_layout: bool,
    pub support_workgroup_layout_8bit_access: bool,
    pub support_workgroup_layout_16bit_access: bool,

    pub support_vote: bool,
    pub support_shuffle_relative: bool,
    pub support_quad_shuffles: bool,
    pub support_shader_quad_control: bool,
    /// Bit per [`crate::ShaderStage`] index where subgroup operations are available.
    pub supported_subgroup_stages: u32,
    pub warp_size_potentially_larger_than_guest: bool,
    pub max_subgroup_size: u32,

    pub support_vertex_instance_id: bool,
    pub support_viewport_index_layer_non_geometry: bool,
    pub support_viewport_mask: bool,
    pub support_multi_viewport: bool,
    pub support_geometry_streams: bool,
    pub support_geometry_shader_passthrough: bool,
    pub support_scaled_attributes: bool,
    pub support_typeless_image_loads: bool,
    pub support_demote_to_helper_invocation: bool,
    pub support_int64_atomics: bool,
    pub support_shared_int64_atomics: bool,
    pub support_derivative_control: bool,
    pub support_native_ndc: bool,
    pub support_mesh_shader: bool,
    pub support_conditional_barrier: bool,
    pub support_snorm_render_buffer: bool,
    pub support_sampled_image_array_nonuniform_indexing: bool,
    pub support_storage_image_array_nonuniform_indexing: bool,
    pub support_uniform_texel_buffer_array_nonuniform_indexing: bool,
    pub support_storage_texel_buffer_array_nonuniform_indexing: bool,

    pub lower_left_origin_mode: bool,
    /// Declare every fragment colour output even when unwritten.
    pub need_declared_frag_colors: bool,
    /// Decorate float operations so the driver cannot apply fast-math rewrites.
    pub need_fastmath_off: bool,
    /// Offset gather coordinates by a subpixel to match Maxwell's 16.8 rounding.
    pub need_gather_subpixel_offset: bool,
    /// Move conditional demotes to the end (drivers whose demote breaks helper invocations).
    pub needs_demote_reorder: bool,
    /// OpFClamp is broken: use FMax and FMin.
    pub has_broken_spirv_clamp: bool,
    /// Position must be wrapped in a block when used as an input.
    pub has_broken_spirv_position_input: bool,
    pub has_broken_unsigned_image_offsets: bool,
    pub has_broken_signed_operations: bool,
    pub has_broken_fp16_float_controls: bool,
    pub has_broken_fp32_denorm_flush: bool,
    pub has_broken_spirv_subgroup_mask_vector_extract_dynamic: bool,
    pub has_broken_spirv_subgroup_shuffle: bool,
    pub has_broken_robust: bool,
    pub ignore_nan_fp_comparisons: bool,

    pub max_compute_shared_memory_size: u32,
    pub min_ssbo_alignment: u64,
    pub max_user_clip_distances: u32,
    pub max_per_stage_storage_buffers: u32,
    pub max_per_stage_sampled_images: u32,
    pub max_per_stage_resources: u32,
}

impl Default for Profile {
    /// A conservative baseline: Vulkan 1.1 core features only.
    fn default() -> Profile {
        Profile {
            supported_spirv: 0x0001_0300,
            unified_descriptor_binding: true,
            support_int8: false,
            support_int16: false,
            support_int64: false,
            support_float16: false,
            support_float64: false,
            support_storage_buffer_8bit: false,
            support_uniform_and_storage_buffer_8bit: false,
            support_storage_buffer_16bit: false,
            support_uniform_and_storage_buffer_16bit: false,
            support_float_controls: false,
            support_separate_denorm_behavior: false,
            support_separate_rounding_mode: false,
            support_fp16_denorm_preserve: false,
            support_fp32_denorm_preserve: false,
            support_fp16_denorm_flush: false,
            support_fp32_denorm_flush: false,
            support_fp16_signed_zero_nan_preserve: false,
            support_fp32_signed_zero_nan_preserve: false,
            support_fp64_signed_zero_nan_preserve: false,
            support_explicit_workgroup_layout: false,
            support_workgroup_layout_8bit_access: false,
            support_workgroup_layout_16bit_access: false,
            support_vote: true,
            support_shuffle_relative: true,
            support_quad_shuffles: true,
            support_shader_quad_control: false,
            supported_subgroup_stages: 0x7f,
            warp_size_potentially_larger_than_guest: false,
            max_subgroup_size: 32,
            support_vertex_instance_id: false,
            support_viewport_index_layer_non_geometry: false,
            support_viewport_mask: false,
            support_multi_viewport: true,
            support_geometry_streams: false,
            support_geometry_shader_passthrough: false,
            support_scaled_attributes: false,
            support_typeless_image_loads: false,
            support_demote_to_helper_invocation: false,
            support_int64_atomics: false,
            support_shared_int64_atomics: false,
            support_derivative_control: false,
            support_native_ndc: false,
            support_mesh_shader: false,
            support_conditional_barrier: false,
            support_snorm_render_buffer: false,
            support_sampled_image_array_nonuniform_indexing: false,
            support_storage_image_array_nonuniform_indexing: false,
            support_uniform_texel_buffer_array_nonuniform_indexing: false,
            support_storage_texel_buffer_array_nonuniform_indexing: false,
            lower_left_origin_mode: false,
            need_declared_frag_colors: false,
            need_fastmath_off: false,
            need_gather_subpixel_offset: false,
            needs_demote_reorder: false,
            has_broken_spirv_clamp: false,
            has_broken_spirv_position_input: false,
            has_broken_unsigned_image_offsets: false,
            has_broken_signed_operations: false,
            has_broken_fp16_float_controls: false,
            has_broken_fp32_denorm_flush: false,
            has_broken_spirv_subgroup_mask_vector_extract_dynamic: false,
            has_broken_spirv_subgroup_shuffle: false,
            has_broken_robust: false,
            ignore_nan_fp_comparisons: false,
            max_compute_shared_memory_size: 32 * 1024,
            min_ssbo_alignment: 256,
            max_user_clip_distances: 8,
            max_per_stage_storage_buffers: 16,
            max_per_stage_sampled_images: 1024,
            max_per_stage_resources: 1024,
        }
    }
}

impl Profile {
    pub fn supports_subgroup_stage(&self, stage: crate::ShaderStage) -> bool {
        self.supported_subgroup_stages & (1 << stage as u32) != 0
    }

    /// Bits of the capabilities that change the translated IR, for shader cache keys.
    pub fn translation_key(&self) -> u32 {
        [
            self.support_float64,
            self.support_float16,
            self.support_int64,
            self.support_conditional_barrier,
            self.needs_demote_reorder,
            self.support_geometry_shader_passthrough,
            self.support_viewport_mask,
            self.support_snorm_render_buffer,
            self.support_mesh_shader,
        ]
        .iter()
        .enumerate()
        .fold(0, |key, (bit, &set)| key | (set as u32) << bit)
    }
}
