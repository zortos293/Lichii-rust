//! The shader translator's view of the device: which SPIR-V features it may emit and which
//! limits bound its descriptor counts.

use ash::vk;
use lichii_shader::Profile;
use lichii_shader::profile::HostTranslateInfo;

use crate::device::DeviceFeatures;

/// What [`shader_profile`] reads from the device besides its features.
pub(crate) struct ShaderQueries<'a> {
    pub api_version: u32,
    pub limits: &'a vk::PhysicalDeviceLimits,
    pub vulkan11: &'a vk::PhysicalDeviceVulkan11Properties<'a>,
    pub vulkan12: &'a vk::PhysicalDeviceVulkan12Properties<'a>,
}

/// The translator profile and host info of a device with `features`, every one of which the
/// device enabled. Constant buffers are word arrays, which needs `uniformBufferStandardLayout`
/// (required at device creation); storage buffers bind at their address rounded down to
/// `minStorageBufferOffsetAlignment`, which the profile passes on so shaders add the rest.
pub(crate) fn shader_profile(
    features: &DeviceFeatures,
    queries: &ShaderQueries<'_>,
) -> (Profile, HostTranslateInfo) {
    let limits = queries.limits;
    let v11 = queries.vulkan11;
    let v12 = queries.vulkan12;
    let on = |b: vk::Bool32| b != vk::FALSE;
    let subgroup_ops = v11.subgroup_supported_operations;
    let subgroup_stages = v11.subgroup_supported_stages;
    let stage_bits = [
        vk::ShaderStageFlags::VERTEX,
        vk::ShaderStageFlags::VERTEX,
        vk::ShaderStageFlags::TESSELLATION_CONTROL,
        vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        vk::ShaderStageFlags::GEOMETRY,
        vk::ShaderStageFlags::FRAGMENT,
        vk::ShaderStageFlags::COMPUTE,
        vk::ShaderStageFlags::TASK_EXT,
        vk::ShaderStageFlags::MESH_EXT,
    ]
    .iter()
    .enumerate()
    .fold(0, |bits, (i, &stage)| {
        bits | u32::from(subgroup_stages.contains(stage)) << i
    });
    let profile = Profile {
        supported_spirv: if queries.api_version >= vk::API_VERSION_1_3 {
            0x0001_0600
        } else {
            0x0001_0500
        },
        unified_descriptor_binding: true,
        support_descriptor_aliasing: true,
        support_int8: features.shader_int8,
        support_int16: features.shader_int16,
        support_int64: features.shader_int64,
        support_float16: features.shader_float16,
        support_float64: features.shader_float64,
        support_storage_buffer_8bit: features.storage_buffer_8bit,
        support_uniform_and_storage_buffer_8bit: features.uniform_and_storage_buffer_8bit,
        support_storage_buffer_16bit: features.storage_buffer_16bit,
        support_uniform_and_storage_buffer_16bit: features.uniform_and_storage_buffer_16bit,
        support_float_controls: true,
        support_separate_denorm_behavior: v12.denorm_behavior_independence
            == vk::ShaderFloatControlsIndependence::ALL,
        support_separate_rounding_mode: v12.rounding_mode_independence
            == vk::ShaderFloatControlsIndependence::ALL,
        support_fp16_denorm_preserve: on(v12.shader_denorm_preserve_float16),
        support_fp32_denorm_preserve: on(v12.shader_denorm_preserve_float32),
        support_fp16_denorm_flush: on(v12.shader_denorm_flush_to_zero_float16),
        support_fp32_denorm_flush: on(v12.shader_denorm_flush_to_zero_float32),
        support_fp16_signed_zero_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float16),
        support_fp32_signed_zero_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float32),
        support_fp64_signed_zero_nan_preserve: on(v12.shader_signed_zero_inf_nan_preserve_float64),
        support_vote: subgroup_ops.contains(vk::SubgroupFeatureFlags::VOTE),
        support_shuffle_relative: subgroup_ops.contains(vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE),
        support_quad_shuffles: subgroup_ops.contains(vk::SubgroupFeatureFlags::QUAD),
        supported_subgroup_stages: stage_bits,
        warp_size_potentially_larger_than_guest: v11.subgroup_size > 32,
        max_subgroup_size: v11.subgroup_size,
        support_viewport_index_layer_non_geometry: features.shader_output_layer
            && features.shader_output_viewport_index,
        support_multi_viewport: features.multi_viewport,
        support_sample_rate_shading: features.sample_rate_shading,
        support_typeless_image_loads: features.storage_image_read_without_format,
        support_demote_to_helper_invocation: features.demote_to_helper_invocation,
        support_derivative_control: true,
        support_mesh_shader: features.mesh_shader,
        support_quad_divergent_implicit_lod: on(v12.quad_divergent_implicit_lod),
        support_sampled_image_array_nonuniform_indexing: features
            .sampled_image_array_non_uniform_indexing,
        support_storage_image_array_nonuniform_indexing: features
            .storage_image_array_non_uniform_indexing,
        support_uniform_texel_buffer_array_nonuniform_indexing: features
            .uniform_texel_buffer_array_non_uniform_indexing,
        support_storage_texel_buffer_array_nonuniform_indexing: features
            .storage_texel_buffer_array_non_uniform_indexing,
        support_native_ndc: features.depth_clip_control,
        max_user_clip_distances: limits.max_clip_distances,
        max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
        min_ssbo_alignment: limits.min_storage_buffer_offset_alignment,
        max_per_stage_storage_buffers: limits.max_per_stage_descriptor_storage_buffers,
        max_per_stage_sampled_images: limits.max_per_stage_descriptor_sampled_images,
        max_per_stage_resources: limits.max_per_stage_resources,
        ..Profile::default()
    };
    let host = HostTranslateInfo {
        min_ssbo_alignment: limits.min_storage_buffer_offset_alignment,
        max_per_stage_descriptor_sampled_images: limits.max_per_stage_descriptor_sampled_images,
        max_per_stage_resources: limits.max_per_stage_resources,
        max_per_stage_descriptor_storage_buffers: limits.max_per_stage_descriptor_storage_buffers,
        max_per_stage_descriptor_samplers: limits.max_per_stage_descriptor_samplers,
        max_descriptor_set_samplers: limits.max_descriptor_set_samplers,
        max_descriptor_set_uniform_buffers: limits.max_descriptor_set_uniform_buffers,
        max_descriptor_set_uniform_buffers_dynamic: limits
            .max_descriptor_set_uniform_buffers_dynamic,
        max_descriptor_set_storage_buffers: limits.max_descriptor_set_storage_buffers,
        max_descriptor_set_storage_buffers_dynamic: limits
            .max_descriptor_set_storage_buffers_dynamic,
        max_descriptor_set_sampled_images: limits.max_descriptor_set_sampled_images,
        max_descriptor_set_storage_images: limits.max_descriptor_set_storage_images,
        max_descriptor_set_input_attachements: limits.max_descriptor_set_input_attachments,
        support_float64: features.shader_float64,
        support_float16: features.shader_float16,
        support_int64: features.shader_int64,
        support_viewport_index_layer: profile.support_viewport_index_layer_non_geometry,
        support_handle_programs: true,
        support_handle_table_arrays: features.sampled_image_array_non_uniform_indexing,
        ..HostTranslateInfo::default()
    };
    (profile, host)
}
