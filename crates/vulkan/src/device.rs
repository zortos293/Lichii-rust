use std::ffi::{CStr, c_char};
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use ash::{ext, khr, vk};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::{AllocationSizes, AllocatorDebugSettings};
use lichii_common::settings::GraphicsSettings;
use parking_lot::Mutex;

use lichii_shader::Profile;
use lichii_shader::profile::HostTranslateInfo;

use crate::error::{Error, Result, VkResultExt};
use crate::instance::Instance;
use crate::profile::{ShaderQueries, shader_profile};

/// Device memory blocks are 64 MiB: with 256 MiB blocks ~38% of device memory sat empty in
/// half-used blocks and evictions never lowered the driver's usage (NX2_NOTES row 114).
const DEVICE_BLOCK_SIZE: u64 = 64 << 20;
const HOST_BLOCK_SIZE: u64 = 32 << 20;

/// Device extensions the renderer uses when present.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceExtensions {
    pub swapchain: bool,
    pub memory_budget: bool,
    pub push_descriptor: bool,
    pub extended_dynamic_state3: bool,
    pub vertex_input_dynamic_state: bool,
    pub custom_border_color: bool,
    pub robustness2: bool,
    pub transform_feedback: bool,
    pub conditional_rendering: bool,
    pub mesh_shader: bool,
    pub pipeline_library: bool,
    pub graphics_pipeline_library: bool,
    pub provoking_vertex: bool,
    pub depth_clip_control: bool,
    pub depth_clip_enable: bool,
    pub index_type_uint8: bool,
    pub line_rasterization: bool,
    pub shader_stencil_export: bool,
}

type ExtensionFlag = fn(&mut DeviceExtensions) -> &mut bool;

const OPTIONAL_EXTENSIONS: &[(&CStr, ExtensionFlag)] = &[
    (ext::memory_budget::NAME, |e| &mut e.memory_budget),
    (khr::push_descriptor::NAME, |e| &mut e.push_descriptor),
    (ext::extended_dynamic_state3::NAME, |e| {
        &mut e.extended_dynamic_state3
    }),
    (ext::vertex_input_dynamic_state::NAME, |e| {
        &mut e.vertex_input_dynamic_state
    }),
    (ext::custom_border_color::NAME, |e| {
        &mut e.custom_border_color
    }),
    (ext::robustness2::NAME, |e| &mut e.robustness2),
    (ext::transform_feedback::NAME, |e| &mut e.transform_feedback),
    (ext::conditional_rendering::NAME, |e| {
        &mut e.conditional_rendering
    }),
    (ext::mesh_shader::NAME, |e| &mut e.mesh_shader),
    (khr::pipeline_library::NAME, |e| &mut e.pipeline_library),
    (ext::graphics_pipeline_library::NAME, |e| {
        &mut e.graphics_pipeline_library
    }),
    (ext::provoking_vertex::NAME, |e| &mut e.provoking_vertex),
    (ext::depth_clip_control::NAME, |e| &mut e.depth_clip_control),
    (ext::depth_clip_enable::NAME, |e| &mut e.depth_clip_enable),
    (ext::index_type_uint8::NAME, |e| &mut e.index_type_uint8),
    (ext::line_rasterization::NAME, |e| &mut e.line_rasterization),
    (ext::shader_stencil_export::NAME, |e| {
        &mut e.shader_stencil_export
    }),
];

/// Optional capabilities, after checking both the extension and its feature bits. Everything
/// here is enabled on the device when supported.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceFeatures {
    pub extensions: DeviceExtensions,
    pub sampler_anisotropy: bool,
    pub texture_compression_bc: bool,
    pub texture_compression_astc_ldr: bool,
    pub texture_compression_etc2: bool,
    pub geometry_shader: bool,
    pub tessellation_shader: bool,
    pub multi_viewport: bool,
    pub depth_clamp: bool,
    pub wide_lines: bool,
    pub logic_op: bool,
    pub dual_src_blend: bool,
    pub sample_rate_shading: bool,
    pub shader_float64: bool,
    pub shader_int64: bool,
    pub shader_int16: bool,
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub storage_image_read_without_format: bool,
    pub storage_image_write_without_format: bool,
    pub host_query_reset: bool,
    pub occlusion_query_precise: bool,
    pub sampler_filter_minmax: bool,
    pub shader_output_layer: bool,
    pub shader_output_viewport_index: bool,
    pub draw_indirect_count: bool,
    pub sampled_image_array_non_uniform_indexing: bool,
    pub storage_image_array_non_uniform_indexing: bool,
    pub uniform_texel_buffer_array_non_uniform_indexing: bool,
    pub storage_texel_buffer_array_non_uniform_indexing: bool,
    pub maintenance4: bool,
    pub null_descriptor: bool,
    /// Sparse binding and 2D sparse residency, on the universal queue.
    pub sparse_residency: bool,
    pub custom_border_color_without_format: bool,
    pub vertex_input_dynamic_state: bool,
    pub provoking_vertex_last: bool,
    pub index_type_uint8: bool,
    pub depth_clip_control: bool,
    pub transform_feedback: bool,
    pub conditional_rendering: bool,
    /// `VK_EXT_mesh_shader` with both task and mesh stages.
    pub mesh_shader: bool,
    /// `VK_EXT_graphics_pipeline_library`: pipelines link from separately built parts.
    pub graphics_pipeline_library: bool,
    pub storage_buffer_8bit: bool,
    pub uniform_and_storage_buffer_8bit: bool,
    pub storage_buffer_16bit: bool,
    pub uniform_and_storage_buffer_16bit: bool,
    pub demote_to_helper_invocation: bool,
}

/// Facts about the physical device the renderer needs at run time.
#[derive(Clone, Debug)]
pub struct DeviceProperties {
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub vendor_id: u32,
    pub driver_id: vk::DriverId,
    pub api_version: u32,
    pub driver_version: u32,
    pub limits: vk::PhysicalDeviceLimits,
    /// Nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// 0 when the queue cannot write timestamps.
    pub timestamp_valid_bits: u32,
    /// The most groups `vkCmdDrawMeshTasksEXT` takes along X, `[without, with]` a task stage
    /// (0 without mesh shaders).
    pub mesh_group_limit: [u32; 2],
    /// Linking pipelines from graphics pipeline libraries is fast enough to do at draw time.
    pub fast_pipeline_linking: bool,
}

/// The logical device, its single universal queue and the memory allocator. Shared between
/// the GPU thread's rasterizer and the presenter through an `Arc`.
pub struct Device {
    pub(crate) instance: Arc<Instance>,
    pub(crate) physical: vk::PhysicalDevice,
    pub(crate) raw: ash::Device,
    pub properties: DeviceProperties,
    pub features: DeviceFeatures,
    pub(crate) queue_family: u32,
    queue: Mutex<vk::Queue>,
    pub(crate) allocator: ManuallyDrop<Mutex<Allocator>>,
    /// Bytes handed out by the allocator; it reports only block capacity cheaply.
    pub(crate) allocated: AtomicU64,
    pub(crate) memory_properties: vk::PhysicalDeviceMemoryProperties,
    /// What the shader translator may emit for this device.
    pub shader_profile: Profile,
    pub shader_host: HostTranslateInfo,
}

struct Candidate {
    physical: vk::PhysicalDevice,
    name: String,
    device_type: vk::PhysicalDeviceType,
    queue_family: u32,
    local_memory: u64,
}

fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    properties
        .device_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn type_rank(device_type: vk::PhysicalDeviceType) -> u32 {
    match device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
        vk::PhysicalDeviceType::CPU => 4,
        _ => 3,
    }
}

fn candidates(instance: &Instance, surface: Option<vk::SurfaceKHR>) -> Result<Vec<Candidate>> {
    let raw = &instance.raw;
    // SAFETY: enumeration and property queries on a valid instance.
    let physicals = unsafe {
        raw.enumerate_physical_devices()
            .call("vkEnumeratePhysicalDevices")?
    };
    let mut out = Vec::new();
    for physical in physicals {
        // SAFETY: `physical` came from this instance.
        let (properties, families, memory) = unsafe {
            (
                raw.get_physical_device_properties(physical),
                raw.get_physical_device_queue_family_properties(physical),
                raw.get_physical_device_memory_properties(physical),
            )
        };
        if properties.api_version < vk::API_VERSION_1_3 {
            continue;
        }
        let wanted = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let queue_family = families.iter().enumerate().position(|(index, family)| {
            family.queue_flags.contains(wanted)
                && surface.is_none_or(|surface| {
                    instance.surface.as_ref().is_some_and(|loader| {
                        // SAFETY: valid physical device, family index and surface.
                        unsafe {
                            loader.get_physical_device_surface_support(
                                physical,
                                index as u32,
                                surface,
                            )
                        }
                        .unwrap_or(false)
                    })
                })
        });
        let Some(queue_family) = queue_family else {
            continue;
        };
        let local_memory = memory.memory_heaps[..memory.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        out.push(Candidate {
            physical,
            name: device_name(&properties),
            device_type: properties.device_type,
            queue_family: queue_family as u32,
            local_memory,
        });
    }
    Ok(out)
}

/// Names of the Vulkan 1.3 devices the renderer can use, for the settings UI.
pub fn available_devices(instance: &Instance) -> Result<Vec<String>> {
    Ok(candidates(instance, None)?
        .into_iter()
        .map(|c| c.name)
        .collect())
}

impl Device {
    /// Picks the device named in `settings.device` (case-insensitive), or else the best one:
    /// discrete before integrated before software, then the most device-local memory. With a
    /// `surface`, only devices that can present to it qualify.
    pub fn new(
        instance: Arc<Instance>,
        settings: &GraphicsSettings,
        surface: Option<vk::SurfaceKHR>,
    ) -> Result<Arc<Self>> {
        let mut candidates = candidates(&instance, surface)?;
        candidates.sort_by_key(|c| (type_rank(c.device_type), std::cmp::Reverse(c.local_memory)));
        let chosen = settings
            .device
            .as_deref()
            .and_then(|wanted| {
                let found = candidates
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(wanted));
                if found.is_none() {
                    tracing::warn!("Vulkan device {wanted:?} not found; using the best available");
                }
                found
            })
            .unwrap_or(0);
        if candidates.is_empty() {
            return Err(Error::NoDevice);
        }
        let candidate = candidates.swap_remove(chosen);
        Self::create(instance, candidate, surface.is_some())
    }

    fn create(
        instance: Arc<Instance>,
        candidate: Candidate,
        want_swapchain: bool,
    ) -> Result<Arc<Self>> {
        let raw_instance = &instance.raw;
        let physical = candidate.physical;
        // SAFETY: `physical` came from this instance.
        let available = unsafe {
            raw_instance
                .enumerate_device_extension_properties(physical)
                .call("vkEnumerateDeviceExtensionProperties")?
        };
        let has = |name: &CStr| {
            available
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut extensions = DeviceExtensions {
            swapchain: want_swapchain && has(khr::swapchain::NAME),
            ..Default::default()
        };
        let mut names: Vec<*const c_char> = Vec::new();
        if extensions.swapchain {
            names.push(khr::swapchain::NAME.as_ptr());
        }
        for (name, flag) in OPTIONAL_EXTENSIONS {
            if has(name) {
                *flag(&mut extensions) = true;
                names.push(name.as_ptr());
            }
        }

        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut eds3 = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT::default();
        let mut vertex_input = vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT::default();
        let mut border = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default();
        let mut robustness2 = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        let mut xfb = vk::PhysicalDeviceTransformFeedbackFeaturesEXT::default();
        let mut conditional = vk::PhysicalDeviceConditionalRenderingFeaturesEXT::default();
        let mut mesh = vk::PhysicalDeviceMeshShaderFeaturesEXT::default();
        let mut library = vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default();
        let mut provoking = vk::PhysicalDeviceProvokingVertexFeaturesEXT::default();
        let mut clip_control = vk::PhysicalDeviceDepthClipControlFeaturesEXT::default();
        let mut clip_enable = vk::PhysicalDeviceDepthClipEnableFeaturesEXT::default();
        let mut uint8 = vk::PhysicalDeviceIndexTypeUint8FeaturesKHR::default();
        let mut lines = vk::PhysicalDeviceLineRasterizationFeaturesKHR::default();

        macro_rules! chain {
            ($features:expr) => {{
                v11.p_next = std::ptr::null_mut();
                v12.p_next = std::ptr::null_mut();
                v13.p_next = std::ptr::null_mut();
                eds3.p_next = std::ptr::null_mut();
                vertex_input.p_next = std::ptr::null_mut();
                border.p_next = std::ptr::null_mut();
                robustness2.p_next = std::ptr::null_mut();
                xfb.p_next = std::ptr::null_mut();
                conditional.p_next = std::ptr::null_mut();
                mesh.p_next = std::ptr::null_mut();
                library.p_next = std::ptr::null_mut();
                provoking.p_next = std::ptr::null_mut();
                clip_control.p_next = std::ptr::null_mut();
                clip_enable.p_next = std::ptr::null_mut();
                uint8.p_next = std::ptr::null_mut();
                lines.p_next = std::ptr::null_mut();
                let mut features = vk::PhysicalDeviceFeatures2::default()
                    .push_next(&mut v11)
                    .push_next(&mut v12)
                    .push_next(&mut v13);
                if extensions.extended_dynamic_state3 {
                    features = features.push_next(&mut eds3);
                }
                if extensions.vertex_input_dynamic_state {
                    features = features.push_next(&mut vertex_input);
                }
                if extensions.custom_border_color {
                    features = features.push_next(&mut border);
                }
                if extensions.robustness2 {
                    features = features.push_next(&mut robustness2);
                }
                if extensions.transform_feedback {
                    features = features.push_next(&mut xfb);
                }
                if extensions.conditional_rendering {
                    features = features.push_next(&mut conditional);
                }
                if extensions.mesh_shader {
                    features = features.push_next(&mut mesh);
                }
                if extensions.graphics_pipeline_library {
                    features = features.push_next(&mut library);
                }
                if extensions.provoking_vertex {
                    features = features.push_next(&mut provoking);
                }
                if extensions.depth_clip_control {
                    features = features.push_next(&mut clip_control);
                }
                if extensions.depth_clip_enable {
                    features = features.push_next(&mut clip_enable);
                }
                if extensions.index_type_uint8 {
                    features = features.push_next(&mut uint8);
                }
                if extensions.line_rasterization {
                    features = features.push_next(&mut lines);
                }
                features.features = $features;
                features
            }};
        }

        let mut queried = chain!(Default::default());
        // SAFETY: every struct in the chain belongs to an extension the device reports.
        unsafe { raw_instance.get_physical_device_features2(physical, &mut queried) };
        let mut core = queried.features;
        if v12.timeline_semaphore == vk::FALSE
            || v12.uniform_buffer_standard_layout == vk::FALSE
            || v13.dynamic_rendering == vk::FALSE
            || v13.synchronization2 == vk::FALSE
        {
            return Err(Error::NoDevice);
        }

        core.robust_buffer_access = vk::FALSE;
        v11.protected_memory = vk::FALSE;
        v13.robust_image_access = vk::FALSE;
        robustness2.robust_buffer_access2 = vk::FALSE;
        robustness2.robust_image_access2 = vk::FALSE;
        mesh.multiview_mesh_shader &= v11.multiview;
        mesh.primitive_fragment_shading_rate_mesh_shader = vk::FALSE;
        mesh.mesh_shader_queries = vk::FALSE;
        let on = |b: vk::Bool32| b != vk::FALSE;
        let features = DeviceFeatures {
            extensions,
            sampler_anisotropy: on(core.sampler_anisotropy),
            texture_compression_bc: on(core.texture_compression_bc),
            texture_compression_astc_ldr: on(core.texture_compression_astc_ldr),
            texture_compression_etc2: on(core.texture_compression_etc2),
            geometry_shader: on(core.geometry_shader),
            tessellation_shader: on(core.tessellation_shader),
            multi_viewport: on(core.multi_viewport),
            depth_clamp: on(core.depth_clamp),
            wide_lines: on(core.wide_lines),
            logic_op: on(core.logic_op),
            dual_src_blend: on(core.dual_src_blend),
            sample_rate_shading: on(core.sample_rate_shading),
            shader_float64: on(core.shader_float64),
            shader_int64: on(core.shader_int64),
            shader_int16: on(core.shader_int16),
            shader_float16: on(v12.shader_float16),
            shader_int8: on(v12.shader_int8),
            storage_image_read_without_format: on(core.shader_storage_image_read_without_format),
            storage_image_write_without_format: on(core.shader_storage_image_write_without_format),
            host_query_reset: on(v12.host_query_reset),
            occlusion_query_precise: on(core.occlusion_query_precise),
            sampler_filter_minmax: on(v12.sampler_filter_minmax),
            shader_output_layer: on(v12.shader_output_layer),
            shader_output_viewport_index: on(v12.shader_output_viewport_index),
            draw_indirect_count: on(v12.draw_indirect_count),
            sampled_image_array_non_uniform_indexing: on(
                v12.shader_sampled_image_array_non_uniform_indexing
            ),
            storage_image_array_non_uniform_indexing: on(
                v12.shader_storage_image_array_non_uniform_indexing
            ),
            uniform_texel_buffer_array_non_uniform_indexing: on(
                v12.shader_uniform_texel_buffer_array_non_uniform_indexing
            ),
            storage_texel_buffer_array_non_uniform_indexing: on(
                v12.shader_storage_texel_buffer_array_non_uniform_indexing
            ),
            maintenance4: on(v13.maintenance4),
            null_descriptor: extensions.robustness2 && on(robustness2.null_descriptor),
            sparse_residency: on(core.sparse_binding)
                && on(core.sparse_residency_image2_d)
                // SAFETY: valid physical device.
                && unsafe {
                    raw_instance.get_physical_device_queue_family_properties(physical)
                }
                .get(candidate.queue_family as usize)
                .is_some_and(|f| f.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING)),
            custom_border_color_without_format: extensions.custom_border_color
                && on(border.custom_border_color_without_format),
            vertex_input_dynamic_state: extensions.vertex_input_dynamic_state
                && on(vertex_input.vertex_input_dynamic_state),
            provoking_vertex_last: extensions.provoking_vertex
                && on(provoking.provoking_vertex_last),
            index_type_uint8: extensions.index_type_uint8 && on(uint8.index_type_uint8),
            depth_clip_control: extensions.depth_clip_control
                && on(clip_control.depth_clip_control),
            transform_feedback: extensions.transform_feedback && on(xfb.transform_feedback),
            conditional_rendering: extensions.conditional_rendering
                && on(conditional.conditional_rendering),
            mesh_shader: extensions.mesh_shader && on(mesh.task_shader) && on(mesh.mesh_shader),
            graphics_pipeline_library: extensions.pipeline_library
                && extensions.graphics_pipeline_library
                && on(library.graphics_pipeline_library),
            storage_buffer_8bit: on(v12.storage_buffer8_bit_access),
            uniform_and_storage_buffer_8bit: on(v12.uniform_and_storage_buffer8_bit_access),
            storage_buffer_16bit: on(v11.storage_buffer16_bit_access),
            uniform_and_storage_buffer_16bit: on(v11.uniform_and_storage_buffer16_bit_access),
            demote_to_helper_invocation: on(v13.shader_demote_to_helper_invocation),
        };

        let priorities = [1.0];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(candidate.queue_family)
            .queue_priorities(&priorities)];
        let mut enabled = chain!(core);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_info)
            .enabled_extension_names(&names)
            .push_next(&mut enabled);
        // SAFETY: the extensions and features requested were all reported by the device.
        let raw = unsafe {
            raw_instance
                .create_device(physical, &info, None)
                .call("vkCreateDevice")?
        };

        let mut v11_properties = vk::PhysicalDeviceVulkan11Properties::default();
        let mut v12_properties = vk::PhysicalDeviceVulkan12Properties::default();
        let mut mesh_properties = vk::PhysicalDeviceMeshShaderPropertiesEXT::default();
        let mut library_properties =
            vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut v11_properties)
            .push_next(&mut v12_properties);
        if features.mesh_shader {
            properties2 = properties2.push_next(&mut mesh_properties);
        }
        if features.graphics_pipeline_library {
            properties2 = properties2.push_next(&mut library_properties);
        }
        // SAFETY: valid physical device and queue family.
        let (queue, families, memory_properties) = unsafe {
            raw_instance.get_physical_device_properties2(physical, &mut properties2);
            (
                raw.get_device_queue(candidate.queue_family, 0),
                raw_instance.get_physical_device_queue_family_properties(physical),
                raw_instance.get_physical_device_memory_properties(physical),
            )
        };
        let base = properties2.properties;
        let (shader_profile, shader_host) = shader_profile(
            &features,
            &ShaderQueries {
                api_version: base.api_version,
                limits: &base.limits,
                vulkan11: &v11_properties,
                vulkan12: &v12_properties,
            },
        );
        let properties = DeviceProperties {
            name: candidate.name,
            device_type: base.device_type,
            vendor_id: base.vendor_id,
            driver_id: v12_properties.driver_id,
            api_version: base.api_version,
            driver_version: base.driver_version,
            limits: base.limits,
            timestamp_period: base.limits.timestamp_period,
            timestamp_valid_bits: families[candidate.queue_family as usize].timestamp_valid_bits,
            mesh_group_limit: [
                mesh_properties.max_mesh_work_group_count[0]
                    .min(mesh_properties.max_mesh_work_group_total_count),
                mesh_properties.max_task_work_group_count[0]
                    .min(mesh_properties.max_task_work_group_total_count),
            ],
            fast_pipeline_linking: library_properties.graphics_pipeline_library_fast_linking
                != vk::FALSE,
        };
        tracing::info!(
            "Vulkan device: {} ({:?}, driver {:?}, API {}.{}.{})",
            properties.name,
            properties.device_type,
            properties.driver_id,
            vk::api_version_major(properties.api_version),
            vk::api_version_minor(properties.api_version),
            vk::api_version_patch(properties.api_version),
        );

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: raw_instance.clone(),
            device: raw.clone(),
            physical_device: physical,
            debug_settings: AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: AllocationSizes::new(DEVICE_BLOCK_SIZE, HOST_BLOCK_SIZE),
        });
        let allocator = match allocator {
            Ok(allocator) => allocator,
            Err(error) => {
                // SAFETY: nothing was created from the device yet.
                unsafe { raw.destroy_device(None) };
                return Err(error.into());
            }
        };

        Ok(Arc::new(Self {
            instance,
            physical,
            raw,
            properties,
            features,
            queue_family: candidate.queue_family,
            queue: Mutex::new(queue),
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
            allocated: AtomicU64::new(0),
            memory_properties,
            shader_profile,
            shader_host,
        }))
    }

    pub fn raw(&self) -> &ash::Device {
        &self.raw
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    /// Submits to the universal queue. The queue lock is held only for the call, so the
    /// rasterizer and the presenter can submit from their own threads.
    pub fn submit(&self, submits: &[vk::SubmitInfo2<'_>], fence: vk::Fence) -> Result<()> {
        let queue = self.queue.lock();
        // SAFETY: the queue is externally synchronised by the lock; the caller keeps every
        // referenced object alive until the submission completes.
        unsafe { self.raw.queue_submit2(*queue, submits, fence) }.call("vkQueueSubmit2")
    }

    /// Binds sparse memory on the universal queue, under the same lock as submissions.
    pub fn bind_sparse(&self, infos: &[vk::BindSparseInfo<'_>]) -> Result<()> {
        let queue = self.queue.lock();
        // SAFETY: the queue is externally synchronised by the lock; the caller keeps the images
        // and memory alive until the binding completes.
        unsafe { self.raw.queue_bind_sparse(*queue, infos, vk::Fence::null()) }
            .call("vkQueueBindSparse")
    }

    /// Block shape of sparse images of `format` with `usage` (single-sampled, 2D, optimal
    /// tiling), `None` when the device cannot make such images sparse.
    pub fn sparse_granularity(
        &self,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
    ) -> Option<vk::Extent3D> {
        if !self.features.sparse_residency {
            return None;
        }
        // SAFETY: valid physical device.
        let properties = unsafe {
            self.instance
                .raw
                .get_physical_device_sparse_image_format_properties(
                    self.physical,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::SampleCountFlags::TYPE_1,
                    usage,
                    vk::ImageTiling::OPTIMAL,
                )
        };
        properties.first().map(|p| p.image_granularity)
    }

    /// Runs `f` with the queue locked (presentation).
    pub(crate) fn with_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        f(*self.queue.lock())
    }

    /// Blocks until the device finished all work.
    pub fn wait_idle(&self) {
        let _queue = self.queue.lock();
        // SAFETY: holding the queue lock keeps other threads from submitting meanwhile.
        if let Err(error) = unsafe { self.raw.device_wait_idle() } {
            tracing::error!("vkDeviceWaitIdle: {error}");
        }
    }

    /// Format support for `format` in texel buffers.
    pub fn buffer_features(&self, format: vk::Format) -> vk::FormatFeatureFlags {
        // SAFETY: valid physical device.
        unsafe {
            self.instance
                .raw
                .get_physical_device_format_properties(self.physical, format)
        }
        .buffer_features
    }

    /// Format support for `format` with optimal tiling.
    pub fn format_features(&self, format: vk::Format) -> vk::FormatFeatureFlags {
        // SAFETY: valid physical device.
        unsafe {
            self.instance
                .raw
                .get_physical_device_format_properties(self.physical, format)
        }
        .optimal_tiling_features
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        // SAFETY: every object created from the device holds an Arc to it and is gone; the
        // allocator frees its blocks before the device is destroyed.
        unsafe {
            let _ = self.raw.device_wait_idle();
            ManuallyDrop::drop(&mut self.allocator);
            self.raw.destroy_device(None);
        }
    }
}
