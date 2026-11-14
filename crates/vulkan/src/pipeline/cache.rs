//! The pipeline cache: graphics and compute pipelines by key, built on worker threads (or on the
//! caller's thread when asynchronous compilation is off) and kept across runs in a
//! `VkPipelineCache` blob on disk together with the keys that produced it.
//!
//! A lookup that hits is one hash of the key's used bytes, a vector index and an atomic load;
//! it allocates nothing and takes no lock. With asynchronous compilation a miss queues the
//! build and returns `None` until it lands, so the caller skips the draw, as Lichii does.

use std::collections::HashMap;
use std::ffi::CStr;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, mpsc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ash::vk;
use lichii_common::perf::PerfStats;
use lichii_common::settings::GraphicsSettings;
use lichii_shader::{DescriptorKind, ShaderModule, ShaderStage};
use parking_lot::{Condvar, Mutex};

use super::disk::{self, Contents, DeviceIdentity};
use super::fixed_state::{
    DEPTH_BOUNDS, DEPTH_TEST, DEPTH_WRITE, EDS2_DEPTH_BIAS, EDS2_PRIMITIVE_RESTART,
    EDS2_RASTERIZER_DISCARD, GROUP_BLEND, GROUP_EDS2, GROUP_STRIDES, GROUP_VERTEX,
    RASTER_ALPHA_TO_COVERAGE, RASTER_ALPHA_TO_ONE, RASTER_DEPTH_CLAMP, RASTER_LOGIC_OP,
    RASTER_NDC_MINUS_ONE_TO_ONE, RASTER_PROVOKING_LAST, STENCIL_TEST, StencilFace,
};
use super::{
    ComputePipelineKey, FxBuildHasher, GraphicsPipelineKey, GraphicsShaders, PipelineFeatures,
    descriptor_type, stage_flags,
};
use crate::device::Device;
use crate::error::{Result, VkResultExt};

const FILE_NAME: &str = "pipelines.bin";
const ENTRY: &CStr = c"main";

/// One binding of a pipeline layout, with the stages of the pipeline that declare it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayoutBinding {
    pub set: u32,
    pub binding: u32,
    pub kind: DescriptorKind,
    pub count: u32,
    pub stages: vk::ShaderStageFlags,
}

/// Descriptor set layouts and the pipeline layout shared by pipelines whose stages bind the
/// same resources.
pub struct PipelineLayout {
    device: Arc<Device>,
    raw: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    bindings: Vec<LayoutBinding>,
    push_constants: (vk::ShaderStageFlags, u32),
}

impl PipelineLayout {
    pub fn raw(&self) -> vk::PipelineLayout {
        self.raw
    }

    /// One layout per set index up to the highest set any stage uses.
    pub fn set_layouts(&self) -> &[vk::DescriptorSetLayout] {
        &self.set_layouts
    }

    /// Every binding, sorted by set and binding.
    pub fn bindings(&self) -> &[LayoutBinding] {
        &self.bindings
    }

    /// The one push constant range: from offset 0, every stage that reads push constants, as
    /// large as the largest block. Pushes must name all of these stages, since a range per stage
    /// at the same offset would overlap.
    pub fn push_constants(&self) -> (vk::ShaderStageFlags, u32) {
        self.push_constants
    }
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        // SAFETY: every pipeline using this layout holds an Arc to it and is gone.
        unsafe {
            self.device.raw.destroy_pipeline_layout(self.raw, None);
            for &set in &self.set_layouts {
                self.device.raw.destroy_descriptor_set_layout(set, None);
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct LayoutDesc {
    bindings: Vec<LayoutBinding>,
    push_constants: (vk::ShaderStageFlags, u32),
}

impl LayoutDesc {
    fn merge<'a>(
        modules: impl IntoIterator<Item = &'a ShaderModule>,
    ) -> std::result::Result<Self, String> {
        let mut bindings: Vec<LayoutBinding> = Vec::new();
        let mut push_constants = (vk::ShaderStageFlags::empty(), 0);
        for module in modules {
            let stage = stage_flags(module.stage);
            for set in &module.resources.sets {
                for b in &set.bindings {
                    match bindings
                        .iter_mut()
                        .find(|l| (l.set, l.binding) == (set.set, b.binding))
                    {
                        Some(existing) if (existing.kind, existing.count) == (b.kind, b.count) => {
                            existing.stages |= stage;
                        }
                        Some(existing) => {
                            return Err(format!(
                                "stages disagree on binding {existing:?} and {b:?} of set {}",
                                set.set
                            ));
                        }
                        None => bindings.push(LayoutBinding {
                            set: set.set,
                            binding: b.binding,
                            kind: b.kind,
                            count: b.count,
                            stages: stage,
                        }),
                    }
                }
            }
            let size = module.resources.push_constant_size;
            if size > 0 {
                push_constants = (push_constants.0 | stage, push_constants.1.max(size));
            }
        }
        bindings.sort_by_key(|b| (b.set, b.binding));
        Ok(Self {
            bindings,
            push_constants,
        })
    }

    fn create(self, device: &Arc<Device>) -> Result<PipelineLayout> {
        let mut layout = PipelineLayout {
            device: device.clone(),
            raw: vk::PipelineLayout::null(),
            set_layouts: Vec::new(),
            bindings: self.bindings,
            push_constants: self.push_constants,
        };
        let sets = layout.bindings.iter().map(|b| b.set + 1).max().unwrap_or(0);
        for set in 0..sets {
            let bindings: Vec<_> = layout
                .bindings
                .iter()
                .filter(|b| b.set == set)
                .map(|b| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(b.binding)
                        .descriptor_type(descriptor_type(b.kind))
                        .descriptor_count(b.count)
                        .stage_flags(b.stages)
                })
                .collect();
            let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            // SAFETY: valid create info; the layout's Drop destroys what was created so far.
            let set_layout = unsafe {
                device
                    .raw
                    .create_descriptor_set_layout(&info, None)
                    .call("vkCreateDescriptorSetLayout")?
            };
            layout.set_layouts.push(set_layout);
        }
        let (stages, size) = layout.push_constants;
        let ranges = [vk::PushConstantRange {
            stage_flags: stages,
            offset: 0,
            size,
        }];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layout.set_layouts)
            .push_constant_ranges(if size > 0 { &ranges } else { &[] });
        // SAFETY: valid create info referencing live set layouts.
        layout.raw = unsafe {
            device
                .raw
                .create_pipeline_layout(&info, None)
                .call("vkCreatePipelineLayout")?
        };
        Ok(layout)
    }
}

/// A built pipeline.
pub struct Pipeline {
    raw: vk::Pipeline,
    layout: Arc<PipelineLayout>,
}

impl Pipeline {
    pub fn raw(&self) -> vk::Pipeline {
        self.raw
    }

    pub fn layout(&self) -> &PipelineLayout {
        &self.layout
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: the cache drops pipelines only after the device went idle.
        unsafe { self.layout.device.raw.destroy_pipeline(self.raw, None) };
    }
}

/// Counts since the cache was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub built: u64,
    pub failed: u64,
    /// Builds the driver served from the `VkPipelineCache` (creation feedback).
    pub cache_hits: u64,
    /// Bytes of driver cache the disk cache seeded the `VkPipelineCache` with.
    pub loaded_bytes: usize,
    /// Graphics pipeline libraries built, one per state subset (fast linking).
    pub libraries: u64,
    /// Pipelines linked from libraries for draws whose pipeline was still building.
    pub fast_linked: u64,
}

/// `None` once set: the build failed and the draw is skipped from then on instead of being
/// rebuilt (and logged) on every draw.
type Slot = OnceLock<Option<Pipeline>>;

struct Shared {
    building: Mutex<u32>,
    idle: Condvar,
    perf: Option<Arc<PerfStats>>,
    shutdown: AtomicBool,
    built: AtomicU64,
    failed: AtomicU64,
    cache_hits: AtomicU64,
    libraries: AtomicU64,
    fast_linked: AtomicU64,
}

impl Shared {
    fn adjust_building(&self, delta: i32) {
        let mut building = self.building.lock();
        *building = building.wrapping_add_signed(delta);
        if let Some(perf) = &self.perf {
            perf.set_shaders_building(*building);
        }
        if *building == 0 {
            self.idle.notify_all();
        }
    }
}

struct Context {
    device: Arc<Device>,
    vk_cache: vk::PipelineCache,
    features: PipelineFeatures,
    shared: Shared,
    /// Held for writing by tests to keep queued builds from starting.
    #[cfg(test)]
    gate: parking_lot::RwLock<()>,
}

const ALL_SUBSETS: vk::GraphicsPipelineLibraryFlagsEXT =
    vk::GraphicsPipelineLibraryFlagsEXT::from_raw(0xf);

/// The state subsets a graphics build creates: all of them for a pipeline, one for a library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Part {
    subsets: vk::GraphicsPipelineLibraryFlagsEXT,
    mesh: bool,
}

impl Part {
    fn pipeline(programs: &GraphicsShaders) -> Self {
        Self {
            subsets: ALL_SUBSETS,
            mesh: programs
                .iter()
                .flatten()
                .any(|p| p.stage == ShaderStage::Mesh),
        }
    }

    fn has(self, subset: vk::GraphicsPipelineLibraryFlagsEXT) -> bool {
        self.subsets.contains(subset)
    }

    fn is_library(self) -> bool {
        self.subsets != ALL_SUBSETS
    }
}

/// A graphics pipeline library: `key` holds only the fields and shaders of `part`, and
/// `layout` is the pipeline layout's address for the shader subsets (0 otherwise).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LibraryKey {
    part: Part,
    layout: usize,
    key: GraphicsPipelineKey,
}

/// Fast linking of one pending graphics pipeline.
enum Linking {
    /// Waiting for the libraries, one per state subset.
    Parts(Vec<Arc<Slot>>, Arc<PipelineLayout>),
    Linked(Pipeline),
    Failed,
}

enum Build {
    Graphics {
        key: Box<GraphicsPipelineKey>,
        programs: GraphicsShaders,
        layout: Arc<PipelineLayout>,
        part: Part,
    },
    Compute {
        key: ComputePipelineKey,
        program: Arc<ShaderModule>,
        layout: Arc<PipelineLayout>,
    },
}

impl Context {
    fn build(&self, build: Build) -> Option<Pipeline> {
        let library = matches!(&build, Build::Graphics { part, .. } if part.is_library());
        let (result, layout) = match build {
            Build::Graphics {
                key,
                programs,
                layout,
                part,
            } => (self.build_graphics(&key, &programs, &layout, part), layout),
            Build::Compute {
                key,
                program,
                layout,
            } => (self.build_compute(&key, &program, &layout), layout),
        };
        match result {
            Ok((raw, _)) if library => {
                self.shared.libraries.fetch_add(1, Ordering::Relaxed);
                Some(Pipeline { raw, layout })
            }
            Ok((raw, cache_hit)) => {
                self.shared.built.fetch_add(1, Ordering::Relaxed);
                if cache_hit {
                    self.shared.cache_hits.fetch_add(1, Ordering::Relaxed);
                }
                Some(Pipeline { raw, layout })
            }
            Err(error) => {
                self.shared.failed.fetch_add(1, Ordering::Relaxed);
                tracing::error!("building a pipeline failed: {error}");
                None
            }
        }
    }

    fn modules<'a>(
        &'a self,
        programs: impl IntoIterator<Item = &'a ShaderModule>,
    ) -> std::result::Result<Modules<'a>, String> {
        let mut modules = Modules {
            device: &self.device,
            stages: Vec::new(),
        };
        for program in programs {
            let info = vk::ShaderModuleCreateInfo::default().code(&program.spirv);
            // SAFETY: the shader translator emits valid SPIR-V; `Modules` destroys the module.
            let module = unsafe { self.device.raw.create_shader_module(&info, None) }
                .call("vkCreateShaderModule")
                .map_err(|e| e.to_string())?;
            modules.stages.push((stage_flags(program.stage), module));
        }
        Ok(modules)
    }

    fn vertex_fetchable(&self, format: vk::Format) -> bool {
        // SAFETY: `physical` belongs to this instance.
        let properties = unsafe {
            self.device
                .instance
                .raw
                .get_physical_device_format_properties(self.device.physical, format)
        };
        properties
            .buffer_features
            .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
    }

    fn build_graphics(
        &self,
        key: &GraphicsPipelineKey,
        programs: &GraphicsShaders,
        layout: &PipelineLayout,
        part: Part,
    ) -> std::result::Result<(vk::Pipeline, bool), String> {
        use vk::GraphicsPipelineLibraryFlagsEXT as S;
        let fixed = &key.fixed;
        let features = &self.features;
        let topology = fixed.vk_topology();
        let mesh = part.mesh;
        let tessellated = !mesh && programs[1].is_some() && programs[2].is_some();
        if part.has(S::PRE_RASTERIZATION_SHADERS) {
            if programs[0].is_none() {
                return Err("no vertex stage".into());
            }
            if !mesh
                && (programs[1].is_some() != programs[2].is_some()
                    || tessellated != (topology == vk::PrimitiveTopology::PATCH_LIST))
            {
                return Err(format!(
                    "tessellation stages do not match topology {topology:?}"
                ));
            }
        }
        let modules = self.modules(programs.iter().flatten().map(|p| &**p))?;
        let stages: Vec<_> = modules
            .stages
            .iter()
            .map(|&(stage, module)| {
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage)
                    .module(module)
                    .name(ENTRY)
            })
            .collect();

        let limits = &self.device.properties.limits;
        let mut attributes = Vec::new();
        let mut bindings = Vec::new();
        if fixed.groups & GROUP_VERTEX != 0 {
            let mut used = 0u32;
            for (location, attribute) in fixed.attributes.iter().enumerate() {
                let format = vk::Format::from_raw(i32::from(attribute.format));
                if format == vk::Format::UNDEFINED
                    || u32::from(attribute.offset) > limits.max_vertex_input_attribute_offset
                {
                    continue;
                }
                if !self.vertex_fetchable(format) {
                    tracing::warn!(
                        "vertex format {format:?} cannot be fetched; attribute {location} is dropped"
                    );
                    continue;
                }
                attributes.push(vk::VertexInputAttributeDescription {
                    location: location as u32,
                    binding: u32::from(attribute.binding),
                    format,
                    offset: u32::from(attribute.offset),
                });
                used |= 1 << attribute.binding;
            }
            for binding in (0..32).filter(|b| used & 1 << b != 0) {
                bindings.push(vk::VertexInputBindingDescription {
                    binding,
                    stride: if fixed.groups & GROUP_STRIDES != 0 {
                        u32::from(fixed.strides[binding as usize])
                    } else {
                        0
                    },
                    input_rate: if fixed.instanced & 1 << binding != 0 {
                        vk::VertexInputRate::INSTANCE
                    } else {
                        vk::VertexInputRate::VERTEX
                    },
                });
            }
        }
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&attributes)
            .vertex_binding_descriptions(&bindings);

        let eds2 = |bit| fixed.groups & GROUP_EDS2 != 0 && fixed.eds2 & bit != 0;
        let raster = |bit| fixed.raster & bit != 0;
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(topology)
            .primitive_restart_enable(eds2(EDS2_PRIMITIVE_RESTART));
        let tessellation = vk::PipelineTessellationStateCreateInfo::default()
            .patch_control_points(u32::from(fixed.patch_control_points.max(1)));
        let viewports = match (features.extended_dynamic_state, features.multi_viewport) {
            (true, _) => 0,
            (false, true) => 16,
            (false, false) => 1,
        };
        let mut clip_control =
            vk::PipelineViewportDepthClipControlCreateInfoEXT::default().negative_one_to_one(true);
        let mut viewport = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(viewports)
            .scissor_count(viewports);
        if raster(RASTER_NDC_MINUS_ONE_TO_ONE) {
            viewport = viewport.push_next(&mut clip_control);
        }
        let mut provoking = vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT::default()
            .provoking_vertex_mode(vk::ProvokingVertexModeEXT::LAST_VERTEX);
        let mut rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(raster(RASTER_DEPTH_CLAMP))
            .rasterizer_discard_enable(eds2(EDS2_RASTERIZER_DISCARD))
            .polygon_mode(vk::PolygonMode::from_raw(i32::from(fixed.polygon_mode)))
            .cull_mode(vk::CullModeFlags::from_raw(u32::from(fixed.cull_mode)))
            .front_face(vk::FrontFace::from_raw(i32::from(fixed.front_face)))
            .depth_bias_enable(eds2(EDS2_DEPTH_BIAS))
            .line_width(1.0);
        if raster(RASTER_PROVOKING_LAST) {
            rasterization = rasterization.push_next(&mut provoking);
        }
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(key.attachments.samples())
            .alpha_to_coverage_enable(raster(RASTER_ALPHA_TO_COVERAGE))
            .alpha_to_one_enable(raster(RASTER_ALPHA_TO_ONE));
        let stencil = |face: StencilFace| vk::StencilOpState {
            fail_op: vk::StencilOp::from_raw(i32::from(face.fail)),
            pass_op: vk::StencilOp::from_raw(i32::from(face.pass)),
            depth_fail_op: vk::StencilOp::from_raw(i32::from(face.depth_fail)),
            compare_op: vk::CompareOp::from_raw(i32::from(face.compare)),
            ..Default::default()
        };
        let depth = |bit| fixed.depth & bit != 0;
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(depth(DEPTH_TEST))
            .depth_write_enable(depth(DEPTH_WRITE))
            .depth_compare_op(vk::CompareOp::from_raw(i32::from(fixed.depth_compare)))
            .depth_bounds_test_enable(depth(DEPTH_BOUNDS))
            .stencil_test_enable(depth(STENCIL_TEST))
            .front(stencil(fixed.stencil_front))
            .back(stencil(fixed.stencil_back))
            .max_depth_bounds(1.0);

        let color_formats: Vec<_> = key.attachments.color_formats().collect();
        let blend_attachments: Vec<_> = color_formats
            .iter()
            .zip(&fixed.blend)
            .map(|(&format, blend)| {
                let mut state = vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                if fixed.groups & GROUP_BLEND == 0 || format == vk::Format::UNDEFINED {
                    return state;
                }
                let blendable = self
                    .device
                    .format_features(format)
                    .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND);
                state = state.color_write_mask(vk::ColorComponentFlags::from_raw(u32::from(
                    blend.write_mask,
                )));
                if blend.enable == 0 || !blendable {
                    return state;
                }
                let factor = |raw: u8| vk::BlendFactor::from_raw(i32::from(raw));
                let op = |raw: u8| vk::BlendOp::from_raw(i32::from(raw));
                state
                    .blend_enable(true)
                    .color_blend_op(op(blend.color_op))
                    .src_color_blend_factor(factor(blend.src_color))
                    .dst_color_blend_factor(factor(blend.dst_color))
                    .alpha_blend_op(op(blend.alpha_op))
                    .src_alpha_blend_factor(factor(blend.src_alpha))
                    .dst_alpha_blend_factor(factor(blend.dst_alpha))
            })
            .collect();
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(raster(RASTER_LOGIC_OP))
            .logic_op(vk::LogicOp::from_raw(i32::from(fixed.logic_op)))
            .attachments(&blend_attachments);
        let dynamic_states = features.dynamic_states(mesh);
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let depth_stencil_format = key.attachments.depth_stencil_format();
        let has_stencil = matches!(
            depth_stencil_format,
            vk::Format::S8_UINT
                | vk::Format::D16_UNORM_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
        );
        let has_depth = depth_stencil_format != vk::Format::UNDEFINED
            && depth_stencil_format != vk::Format::S8_UINT;
        let mut rendering = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(if has_depth {
                depth_stencil_format
            } else {
                vk::Format::UNDEFINED
            })
            .stencil_attachment_format(if has_stencil {
                depth_stencil_format
            } else {
                vk::Format::UNDEFINED
            });
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut stage_feedback = vec![vk::PipelineCreationFeedback::default(); stages.len()];
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback)
            .pipeline_stage_creation_feedbacks(&mut stage_feedback);
        let mut library = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(part.subsets);
        let mut info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .dynamic_state(&dynamic)
            .push_next(&mut feedback_info);
        if part.is_library() {
            info = info
                .flags(vk::PipelineCreateFlags::LIBRARY_KHR)
                .push_next(&mut library);
        }
        if part.has(S::VERTEX_INPUT_INTERFACE) && !mesh {
            info = info
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly);
        }
        if part.has(S::PRE_RASTERIZATION_SHADERS) {
            info = info
                .viewport_state(&viewport)
                .rasterization_state(&rasterization);
            if tessellated {
                info = info.tessellation_state(&tessellation);
            }
        }
        if part.has(S::FRAGMENT_SHADER) {
            info = info.depth_stencil_state(&depth_stencil);
        }
        if part.has(S::PRE_RASTERIZATION_SHADERS) || part.has(S::FRAGMENT_SHADER) {
            info = info.layout(layout.raw);
        }
        if part.has(S::FRAGMENT_OUTPUT_INTERFACE) {
            info = info
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .push_next(&mut rendering);
        }
        // SAFETY: every referenced object is live; the cache is internally synchronised.
        let pipelines = unsafe {
            self.device
                .raw
                .create_graphics_pipelines(self.vk_cache, &[info], None)
        };
        let raw =
            pipelines.map_err(|(_, result)| format!("vkCreateGraphicsPipelines: {result}"))?[0];
        Ok((raw, cache_hit(&feedback)))
    }

    /// Links `libraries` without link-time optimisation.
    fn link(
        &self,
        libraries: &[vk::Pipeline],
        layout: &Arc<PipelineLayout>,
    ) -> std::result::Result<Pipeline, String> {
        let mut libraries = vk::PipelineLibraryCreateInfoKHR::default().libraries(libraries);
        let info = vk::GraphicsPipelineCreateInfo::default()
            .layout(layout.raw)
            .push_next(&mut libraries);
        // SAFETY: the libraries and the layout are live; no pipeline cache is shared.
        let pipelines = unsafe {
            self.device
                .raw
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        let raw = pipelines.map_err(|(_, result)| format!("linking a pipeline: {result}"))?[0];
        Ok(Pipeline {
            raw,
            layout: layout.clone(),
        })
    }

    fn build_compute(
        &self,
        key: &ComputePipelineKey,
        program: &ShaderModule,
        layout: &PipelineLayout,
    ) -> std::result::Result<(vk::Pipeline, bool), String> {
        let limits = &self.device.properties.limits;
        let [x, y, z] = key.workgroup;
        let fits = key
            .workgroup
            .iter()
            .zip(&limits.max_compute_work_group_size)
            .all(|(&size, &max)| (1..=max).contains(&size))
            && u64::from(x) * u64::from(y) * u64::from(z)
                <= u64::from(limits.max_compute_work_group_invocations)
            && key.shared_memory <= limits.max_compute_shared_memory_size;
        if !fits {
            return Err(format!(
                "workgroup {:?} with {} bytes of shared memory exceeds the device's limits",
                key.workgroup, key.shared_memory
            ));
        }
        let modules = self.modules([program])?;
        let (stage, module) = modules.stages[0];
        let mut feedback = vk::PipelineCreationFeedback::default();
        let mut stage_feedback = [vk::PipelineCreationFeedback::default()];
        let mut feedback_info = vk::PipelineCreationFeedbackCreateInfo::default()
            .pipeline_creation_feedback(&mut feedback)
            .pipeline_stage_creation_feedbacks(&mut stage_feedback);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(stage)
                    .module(module)
                    .name(ENTRY),
            )
            .layout(layout.raw)
            .push_next(&mut feedback_info);
        // SAFETY: every referenced object is live; the cache is internally synchronised.
        let pipelines = unsafe {
            self.device
                .raw
                .create_compute_pipelines(self.vk_cache, &[info], None)
        };
        let raw =
            pipelines.map_err(|(_, result)| format!("vkCreateComputePipelines: {result}"))?[0];
        Ok((raw, cache_hit(&feedback)))
    }
}

fn cache_hit(feedback: &vk::PipelineCreationFeedback) -> bool {
    feedback.flags.contains(
        vk::PipelineCreationFeedbackFlags::VALID
            | vk::PipelineCreationFeedbackFlags::APPLICATION_PIPELINE_CACHE_HIT,
    )
}

/// Shader modules of one build, destroyed once the pipeline is created.
struct Modules<'a> {
    device: &'a Device,
    stages: Vec<(vk::ShaderStageFlags, vk::ShaderModule)>,
}

impl Drop for Modules<'_> {
    fn drop(&mut self) {
        for &(_, module) in &self.stages {
            // SAFETY: pipeline creation finished; modules are not needed afterwards.
            unsafe { self.device.raw.destroy_shader_module(module, None) };
        }
    }
}

type Job = Box<dyn FnOnce() + Send>;

struct Workers {
    sender: Option<mpsc::Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    /// `count` threads, or `None` when none could be started.
    fn new(count: usize) -> Option<Self> {
        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));
        let threads: Vec<_> = (0..count)
            .filter_map(|index| {
                let receiver = receiver.clone();
                std::thread::Builder::new()
                    .name(format!("pipeline worker {index}"))
                    .spawn(move || {
                        loop {
                            let job = receiver.lock().recv();
                            match job {
                                Ok(job) => job(),
                                Err(_) => break,
                            }
                        }
                    })
                    .map_err(|error| tracing::warn!("starting a pipeline worker: {error}"))
                    .ok()
            })
            .collect();
        (!threads.is_empty()).then(|| Self {
            sender: Some(sender),
            threads,
        })
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.sender = None;
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// Graphics and compute pipelines of one renderer, owned by its GPU thread.
pub struct PipelineCache {
    ctx: Arc<Context>,
    async_compile: bool,
    worker_count: usize,
    workers: Option<Workers>,
    path: Option<PathBuf>,
    identity: DeviceIdentity,
    loaded_blob_bytes: usize,
    stored: Contents,
    layouts: HashMap<LayoutDesc, Arc<PipelineLayout>>,
    graphics: HashMap<GraphicsPipelineKey, usize, FxBuildHasher>,
    compute: HashMap<ComputePipelineKey, usize, FxBuildHasher>,
    slots: Vec<Arc<Slot>>,
    libraries: HashMap<LibraryKey, Arc<Slot>, FxBuildHasher>,
    /// Graphics slots drawn while building, with fast linking on.
    linking: HashMap<usize, Linking, FxBuildHasher>,
    /// Graphics lookups that found the pipeline still building, with nothing fast-linked yet.
    waits: u64,
    /// Pipeline count and driver compilations at the last [`Self::autosave`] write.
    saved: (usize, u64),
    next_save: Option<Instant>,
    /// Set while an [`Self::autosave`] write runs on a worker.
    saving: Arc<AtomicBool>,
}

/// How often [`PipelineCache::autosave`] writes the disk cache while its pipelines change.
const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(10);

fn identity(device: &Device) -> DeviceIdentity {
    // SAFETY: `physical` belongs to this instance.
    let properties = unsafe {
        device
            .instance
            .raw
            .get_physical_device_properties(device.physical)
    };
    DeviceIdentity {
        vendor_id: properties.vendor_id,
        device_id: properties.device_id,
        driver_version: properties.driver_version,
        pipeline_cache_uuid: properties.pipeline_cache_uuid,
    }
}

/// The disk cache at `path`, empty when it is missing, unreadable or from another device.
fn load(path: &Path, identity: &DeviceIdentity) -> Contents {
    let stored = match std::fs::read(path) {
        Ok(bytes) => disk::decode(&bytes, identity),
        Err(error) if error.kind() == io::ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!("reading {}: {error}", path.display());
            None
        }
    }
    .unwrap_or_default();
    if !stored.graphics.is_empty() || !stored.compute.is_empty() {
        tracing::info!(
            "pipeline cache: {} graphics and {} compute pipelines, {} KiB of driver cache",
            stored.graphics.len(),
            stored.compute.len(),
            stored.blob.len() / 1024
        );
    }
    stored
}

fn create_vk_cache(device: &Device, blob: &[u8]) -> Result<vk::PipelineCache> {
    let info = vk::PipelineCacheCreateInfo::default().initial_data(blob);
    // SAFETY: the blob came from this device and driver (checked by the header); drivers
    // validate it again and start empty when they reject it.
    unsafe {
        device
            .raw
            .create_pipeline_cache(&info, None)
            .call("vkCreatePipelineCache")
    }
}

impl PipelineCache {
    /// Honours `settings.async_shader_compilation` and `settings.disk_shader_cache`; the disk
    /// cache lives in `disk_dir` (per title), and is off without one. `perf` receives the
    /// number of pipelines building.
    pub fn new(
        device: &Arc<Device>,
        settings: &GraphicsSettings,
        disk_dir: Option<&Path>,
        perf: Option<Arc<PerfStats>>,
    ) -> Result<Self> {
        let identity = identity(device);
        let path = disk_dir
            .filter(|_| settings.disk_shader_cache)
            .map(|dir| dir.join(FILE_NAME));
        let stored = path
            .as_deref()
            .map(|path| load(path, &identity))
            .unwrap_or_default();
        let vk_cache = create_vk_cache(device, &stored.blob)?;
        let worker_count = std::thread::available_parallelism()
            .map_or(1, |n| n.get().saturating_sub(1))
            .max(1);
        Ok(Self {
            ctx: Arc::new(Context {
                device: device.clone(),
                vk_cache,
                features: PipelineFeatures::from_device(device),
                shared: Shared {
                    building: Mutex::new(0),
                    idle: Condvar::new(),
                    perf,
                    shutdown: AtomicBool::new(false),
                    built: AtomicU64::new(0),
                    failed: AtomicU64::new(0),
                    cache_hits: AtomicU64::new(0),
                    libraries: AtomicU64::new(0),
                    fast_linked: AtomicU64::new(0),
                },
                #[cfg(test)]
                gate: parking_lot::RwLock::new(()),
            }),
            async_compile: settings.async_shader_compilation,
            worker_count,
            workers: None,
            path,
            identity,
            loaded_blob_bytes: stored.blob.len(),
            stored: Contents {
                blob: Vec::new(),
                ..stored
            },
            layouts: HashMap::new(),
            graphics: HashMap::default(),
            compute: HashMap::default(),
            slots: Vec::new(),
            libraries: HashMap::default(),
            linking: HashMap::default(),
            waits: 0,
            saved: (0, 0),
            next_save: None,
            saving: Arc::new(AtomicBool::new(false)),
        })
    }

    /// The features pipelines are built with; build the [`FixedStateTracker`] from them.
    ///
    /// [`FixedStateTracker`]: super::FixedStateTracker
    pub fn features(&self) -> &PipelineFeatures {
        &self.ctx.features
    }

    /// The cache building with `features` instead of the device's, before any build.
    #[cfg(test)]
    fn with_features(mut self, features: PipelineFeatures) -> Self {
        Arc::get_mut(&mut self.ctx)
            .expect("no build started")
            .features = features;
        self
    }

    pub fn set_async_compilation(&mut self, enabled: bool) {
        self.async_compile = enabled;
    }

    /// The pipeline for `key`, built from `shaders` (whose hashes `key` holds) on a miss.
    /// While it builds asynchronously, one fast-linked from graphics pipeline libraries once
    /// they are built (when the device links fast). `None` while neither is ready or when it
    /// failed to build: skip the draw.
    pub fn graphics(
        &mut self,
        key: &GraphicsPipelineKey,
        shaders: &GraphicsShaders,
    ) -> Option<&Pipeline> {
        let index = match self.graphics.get(key) {
            Some(&index) => index,
            None => self.insert_graphics(key, shaders, true),
        };
        if self.async_compile && self.slots[index].get().is_none() {
            if self.ctx.features.fast_link && self.fast_linked(index, key, shaders).is_some() {
                return match self.linking.get(&index) {
                    Some(Linking::Linked(pipeline)) => Some(pipeline),
                    _ => None,
                };
            }
            self.waits += 1;
            return None;
        }
        self.ready(index)
    }

    /// Graphics lookups since the cache was created that found the pipeline still building.
    pub fn waits(&self) -> u64 {
        self.waits
    }

    fn fast_linked(
        &mut self,
        index: usize,
        key: &GraphicsPipelineKey,
        shaders: &GraphicsShaders,
    ) -> Option<&Pipeline> {
        if !self.linking.contains_key(&index) {
            let layout = self.layout(shaders.iter().flatten().map(|p| &**p))?;
            self.start_linking(index, key, shaders, layout);
        }
        let linking = self.linking.get_mut(&index)?;
        if let Linking::Parts(parts, layout) = linking {
            let mut libraries = [vk::Pipeline::null(); 4];
            for (raw, part) in libraries.iter_mut().zip(parts.iter()) {
                match part.get() {
                    None => return None,
                    Some(None) => {
                        *linking = Linking::Failed;
                        return None;
                    }
                    Some(Some(library)) => *raw = library.raw,
                }
            }
            *linking = match self.ctx.link(&libraries[..parts.len()], layout) {
                Ok(pipeline) => {
                    self.ctx.shared.fast_linked.fetch_add(1, Ordering::Relaxed);
                    Linking::Linked(pipeline)
                }
                Err(error) => {
                    tracing::error!("{error}");
                    Linking::Failed
                }
            };
        }
        match linking {
            Linking::Linked(pipeline) => Some(pipeline),
            _ => None,
        }
    }

    /// Finds or starts the libraries slot `index` links from: the shader subsets on the
    /// workers, the interface subsets (no shaders, cheap) right away.
    fn start_linking(
        &mut self,
        index: usize,
        key: &GraphicsPipelineKey,
        shaders: &GraphicsShaders,
        layout: Arc<PipelineLayout>,
    ) {
        use vk::GraphicsPipelineLibraryFlagsEXT as S;
        let mesh = Part::pipeline(shaders).mesh;
        let subsets = [
            S::VERTEX_INPUT_INTERFACE,
            S::PRE_RASTERIZATION_SHADERS,
            S::FRAGMENT_SHADER,
            S::FRAGMENT_OUTPUT_INTERFACE,
        ];
        let mut parts = Vec::with_capacity(subsets.len());
        for subsets in subsets {
            if mesh && subsets == S::VERTEX_INPUT_INTERFACE {
                continue;
            }
            let part = Part { subsets, mesh };
            let mut programs = GraphicsShaders::default();
            let shaded = match subsets {
                S::PRE_RASTERIZATION_SHADERS => {
                    programs[..4].clone_from_slice(&shaders[..4]);
                    true
                }
                S::FRAGMENT_SHADER => {
                    programs[4] = shaders[4].clone();
                    true
                }
                _ => false,
            };
            let attachments = if subsets == S::FRAGMENT_OUTPUT_INTERFACE {
                key.attachments
            } else {
                bytemuck::Zeroable::zeroed()
            };
            let library = LibraryKey {
                part,
                layout: if shaded {
                    Arc::as_ptr(&layout) as usize
                } else {
                    0
                },
                key: GraphicsPipelineKey::new(&programs, attachments, &key.fixed.subset(subsets)),
            };
            if let Some(slot) = self.libraries.get(&library) {
                parts.push(slot.clone());
                continue;
            }
            let slot = Arc::new(Slot::new());
            self.libraries.insert(library, slot.clone());
            let build = Build::Graphics {
                key: Box::new(library.key),
                programs,
                layout: layout.clone(),
                part,
            };
            if shaded {
                self.start(slot.clone(), build);
            } else {
                let _ = slot.set(self.ctx.build(build));
            }
            parts.push(slot);
        }
        self.linking.insert(index, Linking::Parts(parts, layout));
    }

    /// Fast-linked pipelines whose own pipeline finished building. Drop them once the GPU
    /// finished the work recorded with them.
    pub fn replaced(&mut self) -> impl Iterator<Item = Pipeline> + '_ {
        let slots = &self.slots;
        self.linking
            .extract_if(|&index, _| slots[index].get().is_some())
            .filter_map(|(_, linking)| match linking {
                Linking::Linked(pipeline) => Some(pipeline),
                _ => None,
            })
    }

    /// The compute pipeline for `key`, built from `shader` on a miss; `None` as for
    /// [`Self::graphics`].
    pub fn compute(
        &mut self,
        key: &ComputePipelineKey,
        shader: &Arc<ShaderModule>,
    ) -> Option<&Pipeline> {
        let index = match self.compute.get(key) {
            Some(&index) => index,
            None => self.insert_compute(key, shader),
        };
        self.ready(index)
    }

    fn ready(&self, index: usize) -> Option<&Pipeline> {
        let slot = &self.slots[index];
        match slot.get() {
            Some(pipeline) => pipeline.as_ref(),
            None if self.async_compile => None,
            None => slot.wait().as_ref(),
        }
    }

    /// `link`: also start the libraries fast linking uses, ahead of the pipeline itself.
    fn insert_graphics(
        &mut self,
        key: &GraphicsPipelineKey,
        shaders: &GraphicsShaders,
        link: bool,
    ) -> usize {
        debug_assert_eq!(
            GraphicsPipelineKey::new(shaders, key.attachments, &key.fixed),
            *key,
            "the key names other shaders"
        );
        let index = self.slots.len();
        self.graphics.insert(*key, index);
        let layout = self.layout(shaders.iter().flatten().map(|p| &**p));
        if let Some(layout) = &layout
            && link
            && self.async_compile
            && self.ctx.features.fast_link
        {
            self.start_linking(index, key, shaders, layout.clone());
        }
        let build = layout.map(|layout| Build::Graphics {
            key: Box::new(*key),
            programs: shaders.clone(),
            layout,
            part: Part::pipeline(shaders),
        });
        self.queue(build);
        index
    }

    fn insert_compute(&mut self, key: &ComputePipelineKey, shader: &Arc<ShaderModule>) -> usize {
        debug_assert_eq!(key.shader, shader.hash, "the key names another shader");
        let index = self.slots.len();
        self.compute.insert(*key, index);
        let build = self.layout([&**shader]).map(|layout| Build::Compute {
            key: *key,
            program: shader.clone(),
            layout,
        });
        self.queue(build);
        index
    }

    fn layout<'a>(
        &mut self,
        programs: impl IntoIterator<Item = &'a ShaderModule>,
    ) -> Option<Arc<PipelineLayout>> {
        let desc = LayoutDesc::merge(programs)
            .map_err(|error| tracing::error!("pipeline layout: {error}"))
            .ok()?;
        if let Some(layout) = self.layouts.get(&desc) {
            return Some(layout.clone());
        }
        let layout = Arc::new(
            desc.clone()
                .create(&self.ctx.device)
                .map_err(|error| tracing::error!("pipeline layout: {error}"))
                .ok()?,
        );
        self.layouts.insert(desc, layout.clone());
        Some(layout)
    }

    /// Pushes a slot for `build` (`None`: it already failed) and starts it.
    fn queue(&mut self, build: Option<Build>) {
        let slot = Arc::new(Slot::new());
        self.slots.push(slot.clone());
        let Some(build) = build else {
            self.ctx.shared.failed.fetch_add(1, Ordering::Relaxed);
            let _ = slot.set(None);
            return;
        };
        self.start(slot, build);
    }

    /// Builds into `slot`: on a worker with asynchronous compilation, else right away.
    fn start(&mut self, slot: Arc<Slot>, build: Build) {
        if self.async_compile && self.workers.is_none() {
            self.workers = Workers::new(self.worker_count);
        }
        let sender = self.workers.as_ref().and_then(|w| w.sender.as_ref());
        let (true, Some(sender)) = (self.async_compile, sender) else {
            let _ = slot.set(self.ctx.build(build));
            return;
        };
        self.ctx.shared.adjust_building(1);
        let job_ctx = self.ctx.clone();
        let job: Job = Box::new(move || {
            #[cfg(test)]
            let _gate = job_ctx.gate.read();
            if !job_ctx.shared.shutdown.load(Ordering::Relaxed) {
                let _ = slot.set(job_ctx.build(build));
            }
            job_ctx.shared.adjust_building(-1);
        });
        if let Err(mpsc::SendError(job)) = sender.send(job) {
            job();
        }
    }

    /// Pipelines queued and not finished yet.
    pub fn building(&self) -> u32 {
        *self.ctx.shared.building.lock()
    }

    /// Blocks until every queued build finished.
    pub fn wait_for_builds(&self) {
        let shared = &self.ctx.shared;
        let mut building = shared.building.lock();
        while *building > 0 {
            shared.idle.wait(&mut building);
        }
    }

    pub fn stats(&self) -> PipelineStats {
        let shared = &self.ctx.shared;
        PipelineStats {
            built: shared.built.load(Ordering::Relaxed),
            failed: shared.failed.load(Ordering::Relaxed),
            cache_hits: shared.cache_hits.load(Ordering::Relaxed),
            loaded_bytes: self.loaded_blob_bytes,
            libraries: shared.libraries.load(Ordering::Relaxed),
            fast_linked: shared.fast_linked.load(Ordering::Relaxed),
        }
    }

    /// Opens the disk cache in `dir` (per title) for a cache created without one: merges its
    /// driver cache into the `VkPipelineCache` and keeps its keys for [`Self::prebuild`] and
    /// [`Self::save`]. Honours `settings.disk_shader_cache`.
    pub fn open_disk(&mut self, dir: &Path, settings: &GraphicsSettings) -> Result<()> {
        if !settings.disk_shader_cache {
            return Ok(());
        }
        let path = dir.join(FILE_NAME);
        let stored = load(&path, &self.identity);
        if !stored.blob.is_empty() {
            self.wait_for_builds();
            let source = create_vk_cache(&self.ctx.device, &stored.blob)?;
            // SAFETY: no build runs, so nothing else uses the destination cache; the source
            // cache is ours and destroyed right after the merge.
            let merged = unsafe {
                let raw = &self.ctx.device.raw;
                let merged = raw.merge_pipeline_caches(self.ctx.vk_cache, &[source]);
                raw.destroy_pipeline_cache(source, None);
                merged
            };
            merged.call("vkMergePipelineCaches")?;
        }
        self.loaded_blob_bytes = stored.blob.len();
        self.stored = Contents {
            blob: Vec::new(),
            ..stored
        };
        self.path = Some(path);
        Ok(())
    }

    /// Pipelines the disk cache named at start-up.
    pub fn stored_pipelines(&self) -> usize {
        self.stored.graphics.len() + self.stored.compute.len()
    }

    /// Queues every pipeline the disk cache named whose shaders `resolve` returns, asynchronously
    /// when asynchronous compilation is on; returns how many were queued.
    pub fn prebuild(&mut self, mut resolve: impl FnMut(u64) -> Option<Arc<ShaderModule>>) -> usize {
        let stored = std::mem::take(&mut self.stored);
        let mut queued = 0;
        for key in &stored.graphics {
            if self.graphics.contains_key(key) {
                continue;
            }
            let mut shaders = GraphicsShaders::default();
            let resolved = key.shaders.iter().zip(&mut shaders).all(|(&hash, slot)| {
                *slot = (hash != 0).then(|| resolve(hash)).flatten();
                hash == 0 || slot.is_some()
            });
            if resolved {
                self.insert_graphics(key, &shaders, false);
                queued += 1;
            }
        }
        for key in &stored.compute {
            if self.compute.contains_key(key) {
                continue;
            }
            if let Some(shader) = resolve(key.shader) {
                self.insert_compute(key, &shader);
                queued += 1;
            }
        }
        self.stored = stored;
        queued
    }

    /// Writes the driver cache and the keys of every pipeline that did not fail (this run's
    /// and those the file named before) to the disk cache. Does nothing when it is off.
    pub fn save(&self) -> io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        write(&self.ctx, path, &self.identity, self.keys())
    }

    /// Writes the disk cache on a pipeline worker once [`AUTOSAVE_INTERVAL`] passed since the
    /// last write and pipelines were added or compiled since, so a run that never drops the
    /// cache (a killed process) keeps what it built. Call it once a frame with the time.
    pub fn autosave(&mut self, now: Instant) {
        let Some(path) = &self.path else {
            return;
        };
        let Some(next) = self.next_save else {
            self.next_save = Some(now + AUTOSAVE_INTERVAL);
            return;
        };
        if now < next || self.saving.load(Ordering::Acquire) {
            return;
        }
        let shared = &self.ctx.shared;
        let state = (
            self.graphics.len() + self.compute.len(),
            shared.built.load(Ordering::Relaxed) - shared.cache_hits.load(Ordering::Relaxed),
        );
        let sender = self.workers.as_ref().and_then(|w| w.sender.as_ref());
        let (false, Some(sender)) = (state == self.saved, sender) else {
            return;
        };
        let (ctx, path, identity, keys) =
            (self.ctx.clone(), path.clone(), self.identity, self.keys());
        let saving = self.saving.clone();
        saving.store(true, Ordering::Release);
        let job: Job = Box::new(move || {
            if !ctx.shared.shutdown.load(Ordering::Relaxed)
                && let Err(error) = write(&ctx, &path, &identity, keys)
            {
                tracing::warn!("writing the pipeline cache: {error}");
            }
            saving.store(false, Ordering::Release);
        });
        if sender.send(job).is_err() {
            self.saving.store(false, Ordering::Release);
            return;
        }
        self.saved = state;
        self.next_save = Some(now + AUTOSAVE_INTERVAL);
    }

    /// The pipeline keys [`Self::save`] writes, without the driver cache.
    fn keys(&self) -> Contents {
        let failed = |index: usize| matches!(self.slots[index].get(), Some(None));
        let mut graphics: Vec<_> = self
            .graphics
            .iter()
            .filter(|&(_, &index)| !failed(index))
            .map(|(key, _)| *key)
            .collect();
        graphics.extend(
            self.stored
                .graphics
                .iter()
                .filter(|key| !self.graphics.contains_key(*key)),
        );
        let mut compute: Vec<_> = self
            .compute
            .iter()
            .filter(|&(_, &index)| !failed(index))
            .map(|(key, _)| *key)
            .collect();
        compute.extend(
            self.stored
                .compute
                .iter()
                .filter(|key| !self.compute.contains_key(*key)),
        );
        Contents {
            graphics,
            compute,
            blob: Vec::new(),
        }
    }
}

/// Writes `keys` and the driver cache of `ctx` to `path`.
fn write(ctx: &Context, path: &Path, identity: &DeviceIdentity, keys: Contents) -> io::Result<()> {
    // SAFETY: the cache belongs to this device; drivers synchronise reads internally.
    let blob = unsafe { ctx.device.raw.get_pipeline_cache_data(ctx.vk_cache) }
        .map_err(io::Error::other)?;
    let contents = Contents { blob, ..keys };
    disk::write_atomic(path, &disk::encode(identity, &contents))
}

impl Drop for PipelineCache {
    fn drop(&mut self) {
        self.ctx.shared.shutdown.store(true, Ordering::Relaxed);
        self.workers = None;
        if let Err(error) = self.save() {
            tracing::warn!("writing the pipeline cache: {error}");
        }
        self.ctx.device.wait_idle();
        self.linking.clear();
        self.libraries.clear();
        self.slots.clear();
        self.layouts.clear();
        // SAFETY: no build is running and no pipeline creation uses the cache any more.
        unsafe {
            self.ctx
                .device
                .raw
                .destroy_pipeline_cache(self.ctx.vk_cache, None)
        };
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use gpu_allocator::MemoryLocation;
    use lichii_gpu::engines::maxwell_3d::regs::{BlendFactor, PrimitiveTopology};
    use lichii_gpu::engines::maxwell_3d::{RegisterState, Regs};

    use super::*;
    use crate::commands::{COLOR_RANGE, ImageAccess, transition};
    use crate::memory::{Buffer, Image, ImageDesc};
    use crate::pipeline::{AttachmentState, FixedStateTracker};
    use crate::scheduler::Scheduler;
    use crate::{Instance, SurfaceSupport, shaders};
    use lichii_shader::{
        DescriptorBinding, DescriptorSetLayout, ResourceLayout, ShaderInfo, ShaderStage,
    };

    const VERTEX_HASH: u64 = 0x1111;
    const FRAGMENT_HASH: u64 = 0x2222;
    const COMPUTE_HASH: u64 = 0x3333;
    const TARGET: vk::Format = vk::Format::R8G8B8A8_UNORM;
    const SIZE: u32 = 4;

    fn device() -> Option<Arc<Device>> {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: {error}");
                None
            }
        }
    }

    fn settings(async_compile: bool) -> GraphicsSettings {
        GraphicsSettings {
            async_shader_compilation: async_compile,
            ..Default::default()
        }
    }

    /// A module of the renderer's own SPIR-V declaring `bindings` (binding, kind) in set 0.
    fn module(
        hash: u64,
        stage: ShaderStage,
        spirv: &[u32],
        bindings: &[(u32, DescriptorKind)],
        push_constant_size: u32,
    ) -> ShaderModule {
        let bindings: Vec<_> = bindings
            .iter()
            .map(|&(binding, kind)| DescriptorBinding {
                binding,
                kind,
                count: 1,
                descriptor: 0,
            })
            .collect();
        ShaderModule {
            spirv: spirv.to_vec(),
            hash,
            stage,
            resources: ResourceLayout {
                sets: vec![DescriptorSetLayout { set: 0, bindings }],
                push_constant_size,
            },
            info: ShaderInfo::default(),
        }
    }

    fn present_shaders() -> GraphicsShaders {
        let vertex = module(
            VERTEX_HASH,
            ShaderStage::VertexB,
            shaders::FULLSCREEN_VERT,
            &[],
            48,
        );
        let fragment = module(
            FRAGMENT_HASH,
            ShaderStage::Fragment,
            shaders::PRESENT_COPY_FRAG,
            &[(0, DescriptorKind::CombinedImageSampler)],
            48,
        );
        [
            Some(Arc::new(vertex)),
            None,
            None,
            None,
            Some(Arc::new(fragment)),
        ]
    }

    fn compute_shader() -> Arc<ShaderModule> {
        Arc::new(module(
            COMPUTE_HASH,
            ShaderStage::Compute,
            shaders::BLOCK_LINEAR_UNSWIZZLE_COMP,
            &[
                (0, DescriptorKind::StorageBuffer),
                (1, DescriptorKind::StorageBuffer),
            ],
            32,
        ))
    }

    fn compute_key() -> ComputePipelineKey {
        ComputePipelineKey {
            shader: COMPUTE_HASH,
            workgroup: [32, 8, 1],
            shared_memory: 0,
        }
    }

    fn resolve(hash: u64) -> Option<Arc<ShaderModule>> {
        let [vertex, _, _, _, fragment] = present_shaders();
        match hash {
            VERTEX_HASH => vertex,
            FRAGMENT_HASH => fragment,
            COMPUTE_HASH => Some(compute_shader()),
            _ => None,
        }
    }

    fn key(cache: &PipelineCache, regs: &Regs, shaders: &GraphicsShaders) -> GraphicsPipelineKey {
        let mut state = RegisterState::new();
        let fixed = *FixedStateTracker::new(*cache.features()).refresh(
            regs,
            &mut state.dirty,
            PrimitiveTopology::TRIANGLES,
        );
        let attachments = AttachmentState::new(
            &[TARGET],
            vk::Format::UNDEFINED,
            vk::SampleCountFlags::TYPE_1,
        );
        GraphicsPipelineKey::new(shaders, attachments, &fixed)
    }

    /// Register files that each need their own pipeline on any device: blend factors stay
    /// static without VK_EXT_extended_dynamic_state3, and the logic op is static everywhere.
    fn variants(count: usize) -> Vec<Box<Regs>> {
        (0..count)
            .map(|i| {
                let mut regs = Regs::with_defaults();
                regs.logic_op.op =
                    lichii_gpu::engines::maxwell_3d::regs::LogicOpOp(0x1500 + i as u32);
                regs.blend.enable[0] = 1;
                regs.blend.color_source = if i % 2 == 0 {
                    BlendFactor::ONE_GL
                } else {
                    BlendFactor::SOURCE_ALPHA_GL
                };
                regs
            })
            .collect()
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lichii-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    /// Sets every dynamic state `features` declares, the way the rasterizer will before a draw.
    /// Dynamic binding strides need a vertex buffer bind with strides even for a draw that
    /// fetches no vertices, so `vertices` is bound at binding 0.
    fn set_dynamic_state(
        device: &Device,
        cmd: vk::CommandBuffer,
        features: &PipelineFeatures,
        vertices: vk::Buffer,
    ) {
        let raw = &device.raw;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: SIZE as f32,
            height: SIZE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: SIZE,
                height: SIZE,
            },
        };
        let both = vk::StencilFaceFlags::FRONT_AND_BACK;
        // SAFETY: `cmd` is recording inside a dynamic rendering; every extension function is
        // used only when the device enabled its extension.
        unsafe {
            raw.cmd_set_line_width(cmd, 1.0);
            raw.cmd_set_depth_bias(cmd, 0.0, 0.0, 0.0);
            raw.cmd_set_blend_constants(cmd, &[0.0; 4]);
            raw.cmd_set_depth_bounds(cmd, 0.0, 1.0);
            raw.cmd_set_stencil_compare_mask(cmd, both, 0);
            raw.cmd_set_stencil_write_mask(cmd, both, 0);
            raw.cmd_set_stencil_reference(cmd, both, 0);
            if features.extended_dynamic_state {
                raw.cmd_set_viewport_with_count(cmd, &[viewport]);
                raw.cmd_set_scissor_with_count(cmd, &[scissor]);
                raw.cmd_set_cull_mode(cmd, vk::CullModeFlags::NONE);
                raw.cmd_set_front_face(cmd, vk::FrontFace::COUNTER_CLOCKWISE);
                raw.cmd_set_primitive_topology(cmd, vk::PrimitiveTopology::TRIANGLE_LIST);
                raw.cmd_set_depth_test_enable(cmd, false);
                raw.cmd_set_depth_write_enable(cmd, false);
                raw.cmd_set_depth_compare_op(cmd, vk::CompareOp::ALWAYS);
                raw.cmd_set_depth_bounds_test_enable(cmd, false);
                raw.cmd_set_stencil_test_enable(cmd, false);
                raw.cmd_set_stencil_op(
                    cmd,
                    both,
                    vk::StencilOp::KEEP,
                    vk::StencilOp::KEEP,
                    vk::StencilOp::KEEP,
                    vk::CompareOp::ALWAYS,
                );
            } else {
                raw.cmd_set_viewport(cmd, 0, &[viewport]);
                raw.cmd_set_scissor(cmd, 0, &[scissor]);
                raw.cmd_bind_vertex_buffers(cmd, 0, &[vertices], &[0]);
            }
            if features.extended_dynamic_state2 {
                raw.cmd_set_rasterizer_discard_enable(cmd, false);
                raw.cmd_set_depth_bias_enable(cmd, false);
                raw.cmd_set_primitive_restart_enable(cmd, false);
            }
            if features.extended_dynamic_state && !features.dynamic_vertex_input {
                raw.cmd_bind_vertex_buffers2(cmd, 0, &[vertices], &[0], None, Some(&[0]));
            }
            if features.dynamic_vertex_input {
                ash::ext::vertex_input_dynamic_state::Device::new(&device.instance.raw, raw)
                    .cmd_set_vertex_input(cmd, &[], &[]);
            }
            let eds3 = ash::ext::extended_dynamic_state3::Device::new(&device.instance.raw, raw);
            if features.dynamic_blend {
                eds3.cmd_set_color_blend_enable(cmd, 0, &[vk::FALSE]);
                eds3.cmd_set_color_blend_equation(
                    cmd,
                    0,
                    &[vk::ColorBlendEquationEXT::default()
                        .src_color_blend_factor(vk::BlendFactor::ONE)
                        .src_alpha_blend_factor(vk::BlendFactor::ONE)],
                );
                eds3.cmd_set_color_write_mask(cmd, 0, &[vk::ColorComponentFlags::RGBA]);
            }
            if features.dynamic_depth_clamp {
                eds3.cmd_set_depth_clamp_enable(cmd, false);
            }
            if features.dynamic_logic_op_enable {
                eds3.cmd_set_logic_op_enable(cmd, false);
            }
            if features.dynamic_alpha_to_coverage {
                eds3.cmd_set_alpha_to_coverage_enable(cmd, false);
            }
            if features.dynamic_alpha_to_one {
                eds3.cmd_set_alpha_to_one_enable(cmd, false);
            }
            if features.dynamic_polygon_mode {
                eds3.cmd_set_polygon_mode(cmd, vk::PolygonMode::FILL);
            }
            if features.dynamic_depth_clip_negative_one_to_one {
                eds3.cmd_set_depth_clip_negative_one_to_one(cmd, false);
            }
            if features.dynamic_provoking_vertex {
                eds3.cmd_set_provoking_vertex_mode(cmd, vk::ProvokingVertexModeEXT::FIRST_VERTEX);
            }
        }
    }

    /// Draws `pipeline` (the present copy pass) sampling a 1x1 texture cleared to `color` into
    /// a 4x4 target and returns the target's texels.
    fn draw(
        device: &Arc<Device>,
        features: &PipelineFeatures,
        pipeline: &Pipeline,
        color: [f32; 4],
    ) -> Vec<u8> {
        let raw = &device.raw;
        let mut scheduler = Scheduler::new(device).unwrap();
        let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let texture = Image::new(device, &ImageDesc::d2(TARGET, 1, 1, usage), "texture").unwrap();
        let texture_view = texture
            .view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())
            .unwrap();
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;
        let target =
            Image::new(device, &ImageDesc::d2(TARGET, SIZE, SIZE, usage), "target").unwrap();
        let target_view = target
            .view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())
            .unwrap();
        let readback = Buffer::new(
            device,
            u64::from(SIZE * SIZE * 4),
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "readback",
        )
        .unwrap();
        let vertices = Buffer::new(
            device,
            16,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::GpuOnly,
            "no vertices",
        )
        .unwrap();
        let layout = pipeline.layout();
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&pool_sizes);
        let sampler_info = vk::SamplerCreateInfo::default().max_lod(0.25);
        // SAFETY: valid create infos; everything is destroyed below after the GPU finished.
        let (pool, sampler) = unsafe {
            (
                raw.create_descriptor_pool(&pool_info, None).unwrap(),
                raw.create_sampler(&sampler_info, None).unwrap(),
            )
        };
        let set_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(layout.set_layouts());
        // SAFETY: the pool has room for the one set.
        let set = unsafe { raw.allocate_descriptor_sets(&set_info).unwrap()[0] };
        let image_info = [vk::DescriptorImageInfo {
            sampler,
            image_view: texture_view.raw(),
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);
        // SAFETY: the set and image view are live.
        unsafe { raw.update_descriptor_sets(&[write], &[]) };

        let cmd = scheduler.record().unwrap();
        transition(
            device,
            cmd,
            texture.raw(),
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::TRANSFER_DST,
        );
        // SAFETY: the texture is in TRANSFER_DST_OPTIMAL.
        unsafe {
            raw.cmd_clear_color_image(
                cmd,
                texture.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue { float32: color },
                &[COLOR_RANGE],
            );
        }
        transition(
            device,
            cmd,
            texture.raw(),
            COLOR_RANGE,
            ImageAccess::TRANSFER_DST,
            ImageAccess::SHADER_READ,
        );
        transition(
            device,
            cmd,
            target.raw(),
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::COLOR_ATTACHMENT,
        );
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target_view.raw())
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)];
        let rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: SIZE,
                    height: SIZE,
                },
            })
            .layer_count(1)
            .color_attachments(&attachments);
        let mut params = [0u8; 48];
        for (i, value) in [0.0f32, 0.0, 1.0, 1.0, 1.0, 1.0, SIZE as f32, SIZE as f32]
            .iter()
            .enumerate()
        {
            params[i * 4..i * 4 + 4].copy_from_slice(&value.to_le_bytes());
        }
        let (push_stages, _) = layout.push_constants();
        // SAFETY: `cmd` is recording; the pipeline matches the rendering's formats.
        unsafe {
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.raw());
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout.raw(),
                0,
                &[set],
                &[],
            );
            raw.cmd_push_constants(cmd, layout.raw(), push_stages, 0, &params);
            set_dynamic_state(device, cmd, features, vertices.raw());
            raw.cmd_draw(cmd, 3, 1, 0, 0);
            raw.cmd_end_rendering(cmd);
        }
        transition(
            device,
            cmd,
            target.raw(),
            COLOR_RANGE,
            ImageAccess::COLOR_ATTACHMENT,
            ImageAccess::TRANSFER_SRC,
        );
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: SIZE,
                height: SIZE,
                depth: 1,
            });
        // SAFETY: the target is in TRANSFER_SRC_OPTIMAL and the buffer holds the whole image.
        unsafe {
            raw.cmd_copy_image_to_buffer(
                cmd,
                target.raw(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.raw(),
                &[region],
            );
        }
        scheduler.finish().unwrap();
        // SAFETY: the GPU finished every use of the pool and sampler.
        unsafe {
            raw.destroy_descriptor_pool(pool, None);
            raw.destroy_sampler(sampler, None);
        }
        readback.mapped().unwrap().to_vec()
    }

    #[test]
    fn builds_and_draws_a_pipeline_synchronously() {
        let Some(device) = device() else { return };
        let mut cache = PipelineCache::new(&device, &settings(false), None, None).unwrap();
        let features = *cache.features();
        assert!(features.extended_dynamic_state && features.extended_dynamic_state2);
        let shaders = present_shaders();
        let present = key(&cache, &Regs::with_defaults(), &shaders);
        let raw = {
            let pipeline = cache
                .graphics(&present, &shaders)
                .expect("built on the caller's thread");
            let layout = pipeline.layout();
            assert_eq!(layout.set_layouts().len(), 1);
            assert_eq!(layout.bindings().len(), 1);
            assert_eq!(layout.bindings()[0].stages, vk::ShaderStageFlags::FRAGMENT);
            assert_eq!(
                layout.push_constants(),
                (
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    48
                )
            );
            pipeline.raw()
        };
        assert_eq!(
            cache.graphics(&present, &shaders).map(Pipeline::raw),
            Some(raw)
        );
        assert_eq!(cache.stats().built, 1, "the second lookup hits");

        let pipeline = cache.graphics(&present, &shaders).unwrap();
        let texels = draw(&device, &features, pipeline, [1.0, 0.0, 1.0, 1.0]);
        assert!(
            texels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|t| *t == [255, 0, 255, 255]),
            "{texels:?}"
        );

        let mut static_cache = PipelineCache::new(&device, &settings(false), None, None)
            .unwrap()
            .with_features(PipelineFeatures {
                extended_dynamic_state: true,
                extended_dynamic_state2: true,
                ..PipelineFeatures::default()
            });
        let static_features = *static_cache.features();
        let mut regs = Regs::with_defaults();
        regs.blend.enable[0] = 1;
        regs.blend.color_source = BlendFactor::ZERO_GL;
        regs.blend.color_dest = BlendFactor::ONE_MINUS_SOURCE_COLOR_GL;
        let static_key = key(&static_cache, &regs, &shaders);
        assert_eq!(
            static_key.fixed.len(),
            216,
            "blend and vertex input in the key"
        );
        let pipeline = static_cache
            .graphics(&static_key, &shaders)
            .expect("static pipeline");
        let texels = draw(&device, &static_features, pipeline, [0.0, 1.0, 0.0, 1.0]);
        assert!(
            texels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|t| *t == [0, 0, 0, 255]),
            "blended over black: {texels:?}"
        );

        let mut legacy = PipelineCache::new(&device, &settings(false), None, None)
            .unwrap()
            .with_features(PipelineFeatures::default());
        let legacy_features = *legacy.features();
        let mut regs = Regs::with_defaults();
        regs.vertex_attrib_format[0] = regs.vertex_attrib_format[0]
            .with_bits(6, 1, 0)
            .with_bits(21, 6, 0x1)
            .with_bits(27, 3, 0x7);
        regs.vertex_streams[0].bits.0 = 16;
        regs.depth_test_enable = 1;
        regs.depth_test_func = lichii_gpu::engines::maxwell_3d::regs::ComparisonOp::LESS_GL;
        regs.primitive_restart.enabled = 1;
        let legacy_key = key(&legacy, &regs, &shaders);
        assert_eq!(legacy_key.fixed.len(), 280, "nothing is dynamic");
        assert_eq!(legacy_key.fixed.strides[0], 16);
        let pipeline = legacy
            .graphics(&legacy_key, &shaders)
            .expect("fully static pipeline");
        let texels = draw(&device, &legacy_features, pipeline, [1.0, 0.0, 0.0, 1.0]);
        assert!(
            texels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|t| *t == [255, 0, 0, 255]),
            "{texels:?}"
        );
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn builds_asynchronously_and_reports_progress() {
        let Some(device) = device() else { return };
        let perf = Arc::new(PerfStats::default());
        let cache = PipelineCache::new(&device, &settings(true), None, Some(perf.clone())).unwrap();
        let features = PipelineFeatures {
            fast_link: false,
            ..*cache.features()
        };
        let mut cache = cache.with_features(features);
        let shaders = present_shaders();
        let keys: Vec<_> = variants(4)
            .iter()
            .map(|regs| key(&cache, regs, &shaders))
            .collect();
        {
            let ctx = cache.ctx.clone();
            let _paused = ctx.gate.write();
            for key in &keys {
                assert!(
                    cache.graphics(key, &shaders).is_none(),
                    "not built yet: the draw is skipped"
                );
            }
            assert!(cache.compute(&compute_key(), &compute_shader()).is_none());
            assert_eq!(cache.building(), 5);
            assert_eq!(
                perf.snapshot(Instant::now(), std::time::Duration::from_secs(1))
                    .shaders_building,
                5
            );
            assert!(
                cache.graphics(&keys[0], &shaders).is_none(),
                "a queued key is not queued twice"
            );
            assert_eq!(cache.building(), 5);
            assert_eq!(cache.waits(), 5, "every skipped graphics draw is counted");
        }
        cache.wait_for_builds();
        assert_eq!(cache.building(), 0);
        assert_eq!(
            perf.snapshot(Instant::now(), std::time::Duration::from_secs(1))
                .shaders_building,
            0
        );
        for key in &keys {
            assert!(cache.graphics(key, &shaders).is_some());
        }
        assert!(cache.compute(&compute_key(), &compute_shader()).is_some());
        assert_eq!(cache.waits(), 5, "hits are not waits");
        assert_eq!(cache.stats().built, 5);
        assert_eq!(cache.stats().failed, 0);

        cache.set_async_compilation(false);
        let blocking = key(&cache, &variants(5)[4], &shaders);
        assert!(
            cache.graphics(&blocking, &shaders).is_some(),
            "sync lookups block"
        );
        let features = *cache.features();
        let texels = draw(
            &device,
            &features,
            cache.graphics(&keys[0], &shaders).unwrap(),
            [0.0, 0.0, 1.0, 1.0],
        );
        assert!(
            texels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|t| *t == [0, 0, 255, 255]),
            "{texels:?}"
        );
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn fast_links_while_the_pipeline_builds() {
        let Some(device) = device() else { return };
        let cache = PipelineCache::new(&device, &settings(true), None, None).unwrap();
        if !cache.features().fast_link {
            eprintln!("skipping: the device does not link pipeline libraries fast");
            return;
        }
        let features = PipelineFeatures {
            dynamic_blend: false,
            ..*cache.features()
        };
        let mut cache = cache.with_features(features);
        let shaders = present_shaders();
        let plain = key(&cache, &Regs::with_defaults(), &shaders);
        let mut regs = Regs::with_defaults();
        regs.blend.enable[0] = 1;
        regs.blend.color_source = BlendFactor::ZERO_GL;
        regs.blend.color_dest = BlendFactor::ONE_MINUS_SOURCE_COLOR_GL;
        let blended = key(&cache, &regs, &shaders);
        assert_ne!(plain, blended);
        let ctx = cache.ctx.clone();
        {
            let _paused = ctx.gate.write();
            assert!(
                cache.graphics(&plain, &shaders).is_none(),
                "no shader libraries yet"
            );
        }
        cache.wait_for_builds();
        assert!(cache.graphics(&plain, &shaders).is_some());
        assert_eq!(cache.stats().libraries, 4);

        let fast = {
            let _paused = ctx.gate.write();
            let fast = cache
                .graphics(&blended, &shaders)
                .expect("linked from the shader libraries of the plain pipeline")
                .raw();
            assert_eq!(
                cache.graphics(&blended, &shaders).map(Pipeline::raw),
                Some(fast),
                "linked once"
            );
            let stats = cache.stats();
            assert_eq!((stats.libraries, stats.fast_linked), (5, 1));
            let texels = draw(
                &device,
                &features,
                cache.graphics(&blended, &shaders).unwrap(),
                [0.0, 1.0, 0.0, 1.0],
            );
            assert!(
                texels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|t| *t == [0, 0, 0, 255]),
                "blended over black: {texels:?}"
            );
            assert_eq!(cache.replaced().count(), 0, "still building");
            fast
        };
        cache.wait_for_builds();
        let built = cache.graphics(&blended, &shaders).unwrap().raw();
        assert_ne!(built, fast, "the built pipeline replaces the linked one");
        assert_eq!(cache.replaced().count(), 1);
        assert_eq!(cache.stats().built, 2);
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn failed_builds_are_remembered() {
        let Some(device) = device() else { return };
        let mut cache = PipelineCache::new(&device, &settings(false), None, None).unwrap();
        let shaders = present_shaders();
        let mut state = RegisterState::new();
        let fixed = *FixedStateTracker::new(*cache.features()).refresh(
            &Regs::with_defaults(),
            &mut state.dirty,
            PrimitiveTopology::PATCHES,
        );
        let attachments = AttachmentState::new(
            &[TARGET],
            vk::Format::UNDEFINED,
            vk::SampleCountFlags::TYPE_1,
        );
        let patches = GraphicsPipelineKey::new(&shaders, attachments, &fixed);
        assert!(
            cache.graphics(&patches, &shaders).is_none(),
            "patches need tessellation stages"
        );
        assert!(cache.graphics(&patches, &shaders).is_none());
        let huge = ComputePipelineKey {
            shared_memory: device.properties.limits.max_compute_shared_memory_size + 1,
            ..compute_key()
        };
        assert!(cache.compute(&huge, &compute_shader()).is_none());
        let wide = ComputePipelineKey {
            workgroup: [
                device.properties.limits.max_compute_work_group_invocations + 1,
                1,
                1,
            ],
            ..compute_key()
        };
        assert!(cache.compute(&wide, &compute_shader()).is_none());
        assert_eq!(cache.stats().failed, 3, "each failure is built once");
        assert_eq!(cache.stats().built, 0);
        let mut conflicting = present_shaders();
        let mut fragment = (*conflicting[4].take().unwrap()).clone();
        fragment.hash = 0x4444;
        fragment.resources.sets[0].bindings[0].kind = DescriptorKind::StorageBuffer;
        let mut vertex = (*conflicting[0].take().unwrap()).clone();
        vertex.resources.sets[0].bindings = vec![DescriptorBinding {
            binding: 0,
            kind: DescriptorKind::UniformBuffer,
            count: 1,
            descriptor: 0,
        }];
        let conflicting: GraphicsShaders = [
            Some(Arc::new(vertex)),
            None,
            None,
            None,
            Some(Arc::new(fragment)),
        ];
        let key = key(&cache, &Regs::with_defaults(), &conflicting);
        assert!(
            cache.graphics(&key, &conflicting).is_none(),
            "stages disagree on binding 0"
        );
        assert_eq!(cache.stats().failed, 4);
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn autosave_writes_new_pipelines_on_a_worker_without_dropping_the_cache() {
        let Some(device) = device() else { return };
        let dir = temp_dir("pipeline-autosave");
        let shaders = present_shaders();
        let mut cache = PipelineCache::new(&device, &settings(true), Some(&dir), None).unwrap();
        let keys: Vec<_> = variants(2)
            .iter()
            .map(|regs| key(&cache, regs, &shaders))
            .collect();
        for key in &keys {
            let _ = cache.graphics(key, &shaders);
        }
        cache.wait_for_builds();
        let path = dir.join(FILE_NAME);
        let start = Instant::now();
        cache.autosave(start);
        cache.autosave(start + AUTOSAVE_INTERVAL / 2);
        assert!(
            !path.exists(),
            "nothing is written before the interval passes"
        );
        cache.autosave(start + AUTOSAVE_INTERVAL);
        let deadline = Instant::now() + Duration::from_secs(10);
        while cache.saving.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        let stored = disk::decode(&std::fs::read(&path).unwrap(), &identity(&device))
            .expect("this device's file");
        assert_eq!(stored.graphics.len(), keys.len());
        assert!(keys.iter().all(|key| stored.graphics.contains(key)));
        assert!(stored.blob.len() >= 32, "the driver cache is written too");
        std::fs::remove_file(&path).unwrap();
        cache.autosave(start + AUTOSAVE_INTERVAL * 3);
        assert!(
            !cache.saving.load(Ordering::Acquire) && !path.exists(),
            "an unchanged cache is not written again"
        );
    }

    #[test]
    fn disk_cache_round_trips_and_rejects_other_drivers() {
        let Some(device) = device() else { return };
        let dir = temp_dir("pipeline-cache");
        let shaders = present_shaders();
        let keys = {
            let mut cache =
                PipelineCache::new(&device, &settings(false), Some(&dir), None).unwrap();
            assert_eq!(cache.stored_pipelines(), 0);
            let keys: Vec<_> = variants(3)
                .iter()
                .map(|regs| key(&cache, regs, &shaders))
                .collect();
            for key in &keys {
                assert!(cache.graphics(key, &shaders).is_some());
            }
            assert!(cache.compute(&compute_key(), &compute_shader()).is_some());
            keys
        };
        let path = dir.join(FILE_NAME);
        let file = std::fs::read(&path).unwrap();
        let stored = disk::decode(&file, &identity(&device)).expect("this device's file");
        assert_eq!(stored.graphics.len(), keys.len());
        assert!(keys.iter().all(|key| stored.graphics.contains(key)));
        assert_eq!(stored.compute, [compute_key()]);
        let blob = &stored.blob;
        assert!(
            blob.len() >= 32,
            "the driver's VkPipelineCache data is stored"
        );
        let word = |i: usize| u32::from_le_bytes(blob[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            (word(0), word(1)),
            (32, 1),
            "VkPipelineCacheHeaderVersionOne"
        );
        assert_eq!(blob[16..32], identity(&device).pipeline_cache_uuid);

        let hits = {
            let mut cache = PipelineCache::new(&device, &settings(true), Some(&dir), None).unwrap();
            assert_eq!(
                cache.loaded_blob_bytes,
                blob.len(),
                "the blob seeds the VkPipelineCache"
            );
            assert_eq!(cache.stored_pipelines(), 4);
            assert_eq!(cache.prebuild(resolve), 4, "every stored key resolves");
            assert_eq!(
                cache.prebuild(resolve),
                0,
                "prebuilt keys are not queued again"
            );
            cache.wait_for_builds();
            assert_eq!(
                cache.stats().built,
                4,
                "built before any draw asked for them"
            );
            for key in &keys {
                assert!(cache.graphics(key, &shaders).is_some());
            }
            assert!(cache.compute(&compute_key(), &compute_shader()).is_some());
            assert_eq!(cache.stats().built, 4);
            cache.stats().cache_hits
        };
        eprintln!(
            "second run: {hits} of 4 pipelines reported as driver cache hits ({} byte blob)",
            blob.len()
        );

        let disabled = GraphicsSettings {
            disk_shader_cache: false,
            ..settings(false)
        };
        let off = PipelineCache::new(&device, &disabled, Some(&dir), None).unwrap();
        assert_eq!(
            off.stored_pipelines(),
            0,
            "disk_shader_cache off ignores the file"
        );
        drop(off);

        let mut other_driver = std::fs::read(&path).unwrap();
        other_driver[28] ^= 0xff;
        std::fs::write(&path, &other_driver).unwrap();
        {
            let mut cache =
                PipelineCache::new(&device, &settings(false), Some(&dir), None).unwrap();
            assert_eq!(
                cache.stored_pipelines(),
                0,
                "another driver version discards the file"
            );
            assert_eq!(
                cache.loaded_blob_bytes, 0,
                "and its blob never reaches the driver"
            );
            assert_eq!(cache.prebuild(resolve), 0);
            assert!(cache.graphics(&keys[0], &shaders).is_some());
        }
        let rewritten = std::fs::read(&path).unwrap();
        assert_eq!(
            rewritten[28], file[28],
            "the next save writes the current driver's header"
        );
        std::fs::remove_dir_all(dir).unwrap();
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn per_draw_lookup_cost() {
        let Some(device) = device() else { return };
        let mut cache = PipelineCache::new(&device, &settings(false), None, None).unwrap();
        let shaders = present_shaders();
        let mut state = RegisterState::new();
        FixedStateTracker::register_dirty_flags(&mut state.dirty);
        let mut tracker = FixedStateTracker::new(*cache.features());
        let attachments = AttachmentState::new(
            &[TARGET],
            vk::Format::UNDEFINED,
            vk::SampleCountFlags::TYPE_1,
        );
        let fixed = *tracker.refresh(&state.regs, &mut state.dirty, PrimitiveTopology::TRIANGLES);
        let key = GraphicsPipelineKey::new(&shaders, attachments, &fixed);
        assert!(cache.graphics(&key, &shaders).is_some());
        for _ in 0..1000 {
            cache.graphics(&key, &shaders);
        }
        const DRAWS: u32 = 1_000_000;
        let clean = Instant::now();
        for _ in 0..DRAWS {
            let fixed =
                tracker.refresh(&state.regs, &mut state.dirty, PrimitiveTopology::TRIANGLES);
            let key = GraphicsPipelineKey::new(&shaders, attachments, fixed);
            std::hint::black_box(cache.graphics(&key, &shaders).map(Pipeline::raw));
        }
        let clean = clean.elapsed() / DRAWS;
        let dirty = Instant::now();
        for _ in 0..DRAWS {
            state.dirty.set_all();
            let fixed =
                tracker.refresh(&state.regs, &mut state.dirty, PrimitiveTopology::TRIANGLES);
            let key = GraphicsPipelineKey::new(&shaders, attachments, fixed);
            std::hint::black_box(cache.graphics(&key, &shaders).map(Pipeline::raw));
        }
        let dirty = dirty.elapsed() / DRAWS;
        eprintln!(
            "per draw: {} ns with no state changed, {} ns with every group dirty ({}-byte key)",
            clean.as_nanos(),
            dirty.as_nanos(),
            key.as_bytes().len()
        );
        assert_eq!(cache.stats().built, 1);
    }
}
