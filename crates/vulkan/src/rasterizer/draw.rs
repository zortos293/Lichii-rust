//! Draws and compute dispatches: translated programs from the shader cache, pipelines from the
//! pipeline cache, render targets and textures from the texture cache and buffers from the
//! buffer cache, recorded with dynamic rendering.
//!
//! Every per-draw list lives in [`DrawState`] and is cleared, not reallocated, so a draw whose
//! shaders, pipeline, images and samplers are cached allocates nothing.

use std::collections::HashMap;
use std::sync::Arc;

use ash::vk;
use lichii_common::GpuVAddr;
use lichii_common::settings::{Anisotropy, ResolutionScale};
use lichii_gpu::backend::{
    ChannelId, ComputeContext, ConstBufferBinding, DrawParams, GraphicsContext, IndirectDraw,
};
use lichii_gpu::engines::maxwell_3d::Regs;
use lichii_gpu::engines::maxwell_3d::regs::{
    DepthMode, IndexFormat, PrimitiveTopology, ProvokingVertex, StencilOp, VertexAttributeType,
    WindowOriginMode,
};
use lichii_shader::backend::Bindings;
use lichii_shader::backend::spirv::{
    RENDER_AREA_LAYOUT_OFFSET, RESCALING_LAYOUT_DOWN_FACTOR_OFFSET,
};
use lichii_shader::info::StorageBufferDescriptor;
use lichii_shader::ir::TextureType;
use lichii_shader::runtime::{AttributeType, CompareFunction, InputTopology};
use lichii_shader::{DescriptorKind, RuntimeInfo, ShaderArch, ShaderInfo, ShaderModule};
use lichii_texture::Numeric;
use lichii_texture::block_linear::GobLayout;

use super::VulkanRasterizer;
use crate::buffer_cache::{IndexBinding, assembled_topology, vertex_stream_range};
use crate::commands::{COLOR_RANGE, ImageAccess, transition};
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::memory::{Buffer, Image, ImageDesc, ImageView};
use crate::pipeline::environment::{ProgramEnv, TexturePool};
use crate::pipeline::fixed_state::flags;
use crate::pipeline::maxwell_to_vk as mvk;
use crate::pipeline::sampler::{SamplerCaps, SamplerParams, Tsc};
use crate::pipeline::shader_cache::ShaderCache;
use crate::pipeline::{
    AttachmentState, ComputePipelineKey, FixedStateTracker, FxBuildHasher, GraphicsPipelineKey,
    GraphicsShaders, PipelineCache, PipelineFeatures,
};
use crate::texture_cache::{DEPTH_ATTACHMENT, ImageId, ImageInfo, Transfer, ViewInfo};

const NULL_BUFFER_SIZE: u64 = 64 << 10;

const SHADER_SAMPLED: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::SHADER_SAMPLED_READ,
};

/// Warnings each logged once per renderer.
#[derive(Clone, Copy)]
enum Warning {
    UnsupportedBinding = 1,
    NullTexture = 1 << 1,
    IndirectAssembly = 1 << 2,
    IndirectScattered = 1 << 3,
    NoIndirectCount = 1 << 4,
    RescaledTexture = 1 << 6,
    DrawTextureTarget = 1 << 7,
    DrawTextureSource = 1 << 8,
}

/// A descriptor write of the current draw: `count` infos from `first` of the info list for
/// `kind`.
#[derive(Clone, Copy)]
struct Write {
    binding: u32,
    kind: DescriptorKind,
    first: usize,
    count: u32,
}

/// One bound render target of the current draw.
#[derive(Clone, Copy)]
struct Target {
    id: ImageId,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
}

/// Draw and dispatch state of the renderer, beside the caches it shares with clears and copies.
pub(super) struct DrawState {
    pub(super) pipelines: PipelineCache,
    pub(super) shaders: ShaderCache,
    fixed: FixedStateTracker,
    descriptors: DescriptorAllocator,
    /// By sampler header words and the anisotropy setting they were created with.
    samplers: HashMap<([u32; 8], u8), vk::Sampler, FxBuildHasher>,
    sampler_caps: SamplerCaps,
    anisotropy: Anisotropy,
    eds3: ash::ext::extended_dynamic_state3::Device,
    vertex_input: ash::ext::vertex_input_dynamic_state::Device,
    null_buffer: Buffer,
    null_view: ImageView,
    _null_image: Image,
    arch: ShaderArch,
    channel: Option<ChannelId>,
    warned: u32,
    stages: GraphicsShaders,
    colors: [Option<Target>; 8],
    depth: Option<Target>,
    textures: Vec<ImageId>,
    buffer_infos: Vec<vk::DescriptorBufferInfo>,
    image_infos: Vec<vk::DescriptorImageInfo>,
    writes: Vec<Write>,
    vk_writes: Vec<vk::WriteDescriptorSet<'static>>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_offsets: Vec<u64>,
    vertex_strides: Vec<u64>,
    vertex_bindings: Vec<vk::VertexInputBindingDescription2EXT<'static>>,
    vertex_attributes: Vec<vk::VertexInputAttributeDescription2EXT<'static>>,
    attachments: Vec<vk::RenderingAttachmentInfo<'static>>,
    blend_enables: Vec<vk::Bool32>,
    blend_equations: Vec<vk::ColorBlendEquationEXT>,
    write_masks: Vec<vk::ColorComponentFlags>,
    texture_pipelines: Option<texture::TexturePipelines>,
}

impl DrawState {
    pub(super) fn new(
        device: &Arc<Device>,
        pipelines: PipelineCache,
        gob_layout: GobLayout,
        cmd: vk::CommandBuffer,
    ) -> Result<Self> {
        let null_buffer = Buffer::new(
            device,
            NULL_BUFFER_SIZE,
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "null buffer",
        )?;
        let null_image = Image::new(
            device,
            &ImageDesc::d2(
                vk::Format::R8G8B8A8_UNORM,
                1,
                1,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ),
            "null texture",
        )?;
        let null_view =
            null_image.view(vk::ImageViewType::TYPE_2D, vk::ComponentMapping::default())?;
        transition(
            device,
            cmd,
            null_image.raw(),
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::TRANSFER_DST,
        );
        // SAFETY: `cmd` is recording; the buffer and image are transfer destinations in the
        // layouts given.
        unsafe {
            device
                .raw
                .cmd_fill_buffer(cmd, null_buffer.raw(), 0, NULL_BUFFER_SIZE, 0);
            device.raw.cmd_clear_color_image(
                cmd,
                null_image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue::default(),
                &[COLOR_RANGE],
            );
        }
        transition(
            device,
            cmd,
            null_image.raw(),
            COLOR_RANGE,
            ImageAccess::TRANSFER_DST,
            SHADER_SAMPLED,
        );
        crate::commands::transfer_to_all_barrier(device, cmd);
        let features = *pipelines.features();
        Ok(Self {
            fixed: FixedStateTracker::new(features),
            pipelines,
            shaders: ShaderCache::new(device.shader_profile.clone(), device.shader_host.clone()),
            descriptors: DescriptorAllocator::new(device)?,
            samplers: HashMap::default(),
            sampler_caps: SamplerCaps::from_device(device),
            anisotropy: Anisotropy::Auto,
            eds3: ash::ext::extended_dynamic_state3::Device::new(&device.instance.raw, &device.raw),
            vertex_input: ash::ext::vertex_input_dynamic_state::Device::new(
                &device.instance.raw,
                &device.raw,
            ),
            null_buffer,
            null_view,
            _null_image: null_image,
            arch: if gob_layout == GobLayout::Switch2 {
                ShaderArch::Ampere
            } else {
                ShaderArch::Maxwell
            },
            channel: None,
            warned: 0,
            stages: GraphicsShaders::default(),
            colors: [None; 8],
            depth: None,
            textures: Vec::new(),
            buffer_infos: Vec::new(),
            image_infos: Vec::new(),
            writes: Vec::new(),
            vk_writes: Vec::new(),
            vertex_buffers: Vec::new(),
            vertex_offsets: Vec::new(),
            vertex_strides: Vec::new(),
            vertex_bindings: Vec::new(),
            vertex_attributes: Vec::new(),
            attachments: Vec::new(),
            blend_enables: Vec::new(),
            blend_equations: Vec::new(),
            write_masks: Vec::new(),
            texture_pipelines: None,
        })
    }

    pub(super) fn set_anisotropy(&mut self, anisotropy: Anisotropy) {
        self.anisotropy = anisotropy;
    }

    fn warn_once(&mut self, warning: Warning, message: &str) {
        if self.warned & warning as u32 == 0 {
            self.warned |= warning as u32;
            tracing::warn!("{message}");
        }
    }

    pub(super) fn destroy(&mut self, device: &Device) {
        for (_, sampler) in self.samplers.drain() {
            // SAFETY: the device is idle; no recorded work uses the sampler any more.
            unsafe { device.raw.destroy_sampler(sampler, None) };
        }
    }
}

/// The draw a graphics call issues.
#[derive(Clone, Copy)]
pub(super) enum DrawCall<'a> {
    Direct(&'a DrawParams<'a>),
    Indirect(&'a IndirectDraw),
}

impl DrawCall<'_> {
    fn topology(&self) -> PrimitiveTopology {
        match self {
            Self::Direct(draw) => draw.topology,
            Self::Indirect(draw) => draw.topology,
        }
    }
}

/// Maxwell program slots in host stage order: `(slot, host stage index)`; VertexA (slot 0)
/// joins VertexB.
const GRAPHICS_SLOTS: [(usize, usize); 5] = [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)];

fn attribute_type(regs: &Regs, index: usize) -> AttributeType {
    let attribute = regs.vertex_attrib_format[index];
    if attribute.constant() != 0 {
        return AttributeType::Float;
    }
    match attribute.type_() {
        VertexAttributeType::S_INT => AttributeType::SignedInt,
        VertexAttributeType::U_INT => AttributeType::UnsignedInt,
        VertexAttributeType::S_SCALED => AttributeType::SignedScaled,
        VertexAttributeType::U_SCALED => AttributeType::UnsignedScaled,
        _ => AttributeType::Float,
    }
}

fn compare_function(regs: &Regs) -> CompareFunction {
    match mvk::comparison_op(regs.alpha_test_func).unwrap_or(vk::CompareOp::ALWAYS) {
        vk::CompareOp::NEVER => CompareFunction::Never,
        vk::CompareOp::LESS => CompareFunction::Less,
        vk::CompareOp::EQUAL => CompareFunction::Equal,
        vk::CompareOp::LESS_OR_EQUAL => CompareFunction::LessThanEqual,
        vk::CompareOp::GREATER => CompareFunction::Greater,
        vk::CompareOp::NOT_EQUAL => CompareFunction::NotEqual,
        vk::CompareOp::GREATER_OR_EQUAL => CompareFunction::GreaterThanEqual,
        _ => CompareFunction::Always,
    }
}

fn input_topology(topology: vk::PrimitiveTopology) -> InputTopology {
    use vk::PrimitiveTopology as T;
    match topology {
        T::POINT_LIST => InputTopology::Points,
        T::LINE_LIST | T::LINE_STRIP => InputTopology::Lines,
        T::LINE_LIST_WITH_ADJACENCY | T::LINE_STRIP_WITH_ADJACENCY => InputTopology::LinesAdjacency,
        T::TRIANGLE_LIST_WITH_ADJACENCY | T::TRIANGLE_STRIP_WITH_ADJACENCY => {
            InputTopology::TrianglesAdjacency
        }
        _ => InputTopology::Triangles,
    }
}

/// The pipeline state one graphics stage translates for: everything [`RuntimeInfo`] takes
/// from the draw, so hashing it keys the module and `runtime` rebuilds the info on a miss.
#[derive(Clone, Copy, Hash)]
struct StageState {
    /// Module hash of the previous stage, whose outputs this one reads (0 for vertex).
    previous: u64,
    generic_inputs: [AttributeType; 32],
    convert_depth_mode: bool,
    y_negate: bool,
    input_topology: InputTopology,
    point_size: Option<u32>,
    alpha_test: Option<(CompareFunction, u32)>,
    color_outputs: [AttributeType; 8],
}

impl StageState {
    fn new(
        regs: &Regs,
        slot: usize,
        last_pre_raster: bool,
        previous: Option<&ShaderModule>,
        topology: vk::PrimitiveTopology,
        color_outputs: [AttributeType; 8],
        features: &PipelineFeatures,
    ) -> Self {
        let fragment = slot == 5;
        let mut state = Self {
            previous: previous.map_or(0, |m| m.hash),
            generic_inputs: [AttributeType::Float; 32],
            convert_depth_mode: false,
            y_negate: regs.window_origin.mode() != WindowOriginMode::UPPER_LEFT,
            input_topology: InputTopology::Points,
            point_size: None,
            alpha_test: None,
            color_outputs: [AttributeType::Float; 8],
        };
        if slot == 1 {
            state.generic_inputs = std::array::from_fn(|i| attribute_type(regs, i));
        }
        if slot == 4 {
            state.input_topology = input_topology(topology);
        }
        if last_pre_raster {
            state.convert_depth_mode =
                regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE && !features.depth_clip_control;
            if topology == vk::PrimitiveTopology::POINT_LIST
                && regs.point_size_attribute.enabled() == 0
            {
                state.point_size = Some(regs.point_size.to_bits());
            }
        }
        if fragment {
            if regs.alpha_test_enabled != 0 {
                state.alpha_test = Some((compare_function(regs), regs.alpha_test_ref.to_bits()));
            }
            state.color_outputs = color_outputs;
        }
        state
    }

    fn key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = crate::pipeline::FxHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn runtime(&self, previous: Option<&ShaderInfo>) -> RuntimeInfo {
        let mut info = RuntimeInfo {
            generic_input_types: self.generic_inputs,
            convert_depth_mode: self.convert_depth_mode,
            y_negate: self.y_negate,
            input_topology: self.input_topology,
            fixed_state_point_size: self.point_size.map(f32::from_bits),
            color_output_types: self.color_outputs,
            ..RuntimeInfo::default()
        };
        if let Some((func, reference)) = self.alpha_test {
            info.alpha_test_func = Some(func);
            info.alpha_test_reference = f32::from_bits(reference);
        }
        match previous {
            Some(previous) => {
                info.previous_stage_stores = previous.stores;
                info.previous_stage_legacy_stores_mapping = previous.legacy_stores_mapping.clone();
            }
            None => info.previous_stage_stores = lichii_shader::info::VaryingState([u64::MAX; 8]),
        }
        info
    }
}

/// The guest render area in pixels: the surface clip, or render target 0 without one.
fn guest_render_area(regs: &Regs) -> (f32, f32) {
    let clip = &regs.surface_clip;
    let (width, height) = match (clip.bits.width(), clip.bits1.height()) {
        (0, _) | (_, 0) => (regs.rt[0].width, regs.rt[0].height),
        size => size,
    };
    (width as f32, height as f32)
}

fn render_target_index(regs: &Regs, i: usize) -> usize {
    ((regs.rt_control.0 >> (4 + 3 * i)) & 7) as usize
}

fn view_type(ty: TextureType) -> vk::ImageViewType {
    match ty {
        TextureType::Color1D => vk::ImageViewType::TYPE_1D,
        TextureType::ColorArray1D => vk::ImageViewType::TYPE_1D_ARRAY,
        TextureType::Color2D | TextureType::Color2DRect => vk::ImageViewType::TYPE_2D,
        TextureType::ColorArray2D => vk::ImageViewType::TYPE_2D_ARRAY,
        TextureType::Color3D | TextureType::Buffer => vk::ImageViewType::TYPE_3D,
        TextureType::ColorCube => vk::ImageViewType::CUBE,
        TextureType::ColorArrayCube => vk::ImageViewType::CUBE_ARRAY,
    }
}

fn read_cbuf<T: bytemuck::Pod + Default>(
    gmmu: &lichii_gpu::gmmu::Gmmu,
    cbufs: &[ConstBufferBinding],
    index: u32,
    offset: u32,
) -> T {
    match cbufs.get(index as usize) {
        Some(cbuf) if cbuf.enabled && offset < cbuf.size => gmmu
            .read::<T>(cbuf.address + u64::from(offset))
            .unwrap_or_default(),
        _ => T::default(),
    }
}

/// Where a stage reads its resources from.
struct StageSources<'a> {
    gmmu: &'a lichii_gpu::gmmu::Gmmu,
    cbufs: &'a [ConstBufferBinding],
    pool: TexturePool,
    samplers: GpuVAddr,
    sampler_limit: u32,
}

impl VulkanRasterizer {
    pub(super) fn draw_graphics(
        &mut self,
        ctx: &mut GraphicsContext<'_>,
        call: DrawCall<'_>,
    ) -> Result<()> {
        if self.draw.channel != Some(ctx.channel) {
            self.draw.channel = Some(ctx.channel);
            ctx.dirty.set_all();
        }
        let [(first_method, _, _), ..] = FixedStateTracker::methods();
        if ctx.dirty.flags_for(first_method)[0] != flags::VERTEX_INPUT {
            FixedStateTracker::register_dirty_flags(ctx.dirty);
        }
        let regs = ctx.regs;
        let host_topology = mvk::primitive_topology(assembled_topology(call.topology()))
            .unwrap_or(vk::PrimitiveTopology::TRIANGLE_LIST);
        if !self.graphics_stages(ctx, host_topology) {
            return Ok(());
        }
        if !self.render_targets(ctx)? {
            return Ok(());
        }

        let draw = &mut self.draw;
        let fixed = *draw
            .fixed
            .refresh(regs, ctx.dirty, assembled_topology(call.topology()));
        let colors: [vk::Format; 8] = draw
            .colors
            .map(|t| t.map_or(vk::Format::UNDEFINED, |t| t.format));
        let samples =
            mvk::sample_count(regs.anti_alias_samples_mode).unwrap_or(vk::SampleCountFlags::TYPE_1);
        let attachments = AttachmentState::new(
            &colors,
            draw.depth.map_or(vk::Format::UNDEFINED, |t| t.format),
            samples,
        );
        let key = GraphicsPipelineKey::new(&draw.stages, attachments, &fixed);
        let Some(pipeline) = draw.pipelines.graphics(&key, &draw.stages) else {
            return Ok(());
        };
        let pipeline_raw = pipeline.raw();
        let layout = pipeline.layout();
        let layout_raw = layout.raw();
        let push = layout.push_constants();
        let set_layout = layout.set_layouts().first().copied();
        let features = *draw.pipelines.features();

        let gmmu = ctx.memory;
        let (index, indirect) = loop {
            let merged = self.buffer_cache.stats().merged_buffers;
            self.draw.buffer_infos.clear();
            self.draw.image_infos.clear();
            self.draw.writes.clear();
            self.draw.textures.clear();
            for &(slot, host) in &GRAPHICS_SLOTS {
                let Some(module) = self.draw.stages[host].clone() else {
                    continue;
                };
                let sources = StageSources {
                    gmmu,
                    cbufs: &ctx.const_buffers[slot - 1],
                    pool: TexturePool {
                        address: regs.tex_header.address(),
                        limit: regs.tex_header.limit,
                        via_header_index: regs.sampler_binding
                            == lichii_gpu::engines::maxwell_3d::regs::SamplerBinding::VIA_HEADER_BINDING,
                        switch2: self.draw.arch == ShaderArch::Ampere,
                    },
                    samplers: regs.tex_sampler.address(),
                    sampler_limit: regs.tex_sampler.limit,
                };
                if !self.bind_resources(&module, &sources)? {
                    return Ok(());
                }
            }
            self.vertex_buffers(gmmu, regs)?;
            let index = match call {
                DrawCall::Direct(params) => self.buffer_cache.index_buffer(
                    &mut self.scheduler,
                    &mut self.staging,
                    gmmu,
                    regs,
                    params,
                )?,
                DrawCall::Indirect(indirect) => self.indirect_index_buffer(gmmu, regs, indirect)?,
            };
            let indirect = match call {
                DrawCall::Indirect(draw) => match self.indirect_buffers(gmmu, draw)? {
                    Some(buffers) => Some(buffers),
                    None => return Ok(()),
                },
                DrawCall::Direct(_) => None,
            };
            if self.buffer_cache.stats().merged_buffers == merged {
                break (index, indirect);
            }
        };
        if index.is_some_and(|i: IndexBinding| i.index_count == 0) {
            return Ok(());
        }
        let set = self.descriptor_set(set_layout)?;

        let cmd = self.scheduler.record()?;
        let draw = &mut self.draw;
        for target in draw.colors.iter().flatten() {
            self.texture_cache
                .prepare(cmd, target.id, ImageAccess::COLOR_ATTACHMENT);
        }
        if let Some(depth) = draw.depth {
            self.texture_cache.prepare(cmd, depth.id, DEPTH_ATTACHMENT);
        }
        for &id in &draw.textures {
            self.texture_cache.prepare(cmd, id, SHADER_SAMPLED);
        }
        let extent = self.render_area(regs);
        let scale = self.target_scale();
        let raw = &self.device.raw;
        let draw = &mut self.draw;
        draw.attachments.clear();
        let last = draw
            .colors
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |i| i + 1);
        for target in &draw.colors[..last] {
            draw.attachments.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(target.map_or(vk::ImageView::null(), |t| t.view))
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE),
            );
        }
        let depth_attachment = draw.depth.map(|depth| {
            vk::RenderingAttachmentInfo::default()
                .image_view(depth.view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        let depth_format = draw.depth.map_or(vk::Format::UNDEFINED, |d| d.format);
        let mut rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(&draw.attachments);
        let has_depth = !matches!(depth_format, vk::Format::UNDEFINED | vk::Format::S8_UINT);
        let has_stencil = matches!(
            depth_format,
            vk::Format::S8_UINT
                | vk::Format::D16_UNORM_S8_UINT
                | vk::Format::D24_UNORM_S8_UINT
                | vk::Format::D32_SFLOAT_S8_UINT
        );
        if let Some(depth) = depth_attachment.as_ref().filter(|_| has_depth) {
            rendering = rendering.depth_attachment(depth);
        }
        if let Some(stencil) = depth_attachment.as_ref().filter(|_| has_stencil) {
            rendering = rendering.stencil_attachment(stencil);
        }
        // SAFETY: `cmd` is recording outside any rendering; every view, buffer and descriptor
        // set is live until the scheduler's current tick completes, and the pipeline was built
        // for these attachment formats.
        unsafe {
            raw.cmd_begin_rendering(cmd, &rendering);
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_raw);
            if let Some(set) = set {
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout_raw,
                    0,
                    &[set],
                    &[],
                );
            }
        }
        let stages = self.draw.stages.iter().flatten();
        let rescaling = stages.clone().any(|m| m.info.uses_rescaling_uniform);
        let render_area = stages
            .clone()
            .any(|m| m.info.uses_render_area)
            .then(|| guest_render_area(regs));
        self.push_constants(cmd, layout_raw, push, rescaling, render_area, scale);
        self.set_dynamic_state(cmd, regs, &features, host_topology, extent, scale);
        let raw = &self.device.raw;
        let draw = &self.draw;
        // SAFETY: as above; the pipeline's dynamic state is all set.
        unsafe {
            if let Some(index) = index {
                raw.cmd_bind_index_buffer(cmd, index.buffer, index.offset, index.index_type);
            }
            match (call, indirect, index) {
                (DrawCall::Direct(params), _, Some(index)) => raw.cmd_draw_indexed(
                    cmd,
                    index.index_count,
                    params.instance_count,
                    index.first_index,
                    index.vertex_offset,
                    params.base_instance,
                ),
                (DrawCall::Direct(params), _, None) => raw.cmd_draw(
                    cmd,
                    params.count,
                    params.instance_count,
                    params.first,
                    params.base_instance,
                ),
                (DrawCall::Indirect(params), Some((buffer, count)), index) => {
                    let indexed = index.is_some();
                    match (count, indexed) {
                        (Some(count), true) => raw.cmd_draw_indexed_indirect_count(
                            cmd,
                            buffer.buffer,
                            buffer.offset,
                            count.buffer,
                            count.offset,
                            params.max_draws,
                            params.stride,
                        ),
                        (Some(count), false) => raw.cmd_draw_indirect_count(
                            cmd,
                            buffer.buffer,
                            buffer.offset,
                            count.buffer,
                            count.offset,
                            params.max_draws,
                            params.stride,
                        ),
                        (None, true) => raw.cmd_draw_indexed_indirect(
                            cmd,
                            buffer.buffer,
                            buffer.offset,
                            params.max_draws,
                            params.stride,
                        ),
                        (None, false) => raw.cmd_draw_indirect(
                            cmd,
                            buffer.buffer,
                            buffer.offset,
                            params.max_draws,
                            params.stride,
                        ),
                    }
                }
                (DrawCall::Indirect(_), None, _) => {}
            }
            raw.cmd_end_rendering(cmd);
        }
        let written: [Option<ImageId>; 9] = std::array::from_fn(|i| match i {
            8 => draw.depth.map(|t| t.id),
            i => draw.colors[i].map(|t| t.id),
        });
        for id in written.into_iter().flatten() {
            self.texture_cache.mark_modified(id);
        }
        self.unflushed = true;
        Ok(())
    }

    /// Translates (or finds) the enabled programs into `self.draw.stages`. `false` skips the
    /// draw.
    fn graphics_stages(
        &mut self,
        ctx: &GraphicsContext<'_>,
        topology: vk::PrimitiveTopology,
    ) -> bool {
        let regs = ctx.regs;
        let enabled = |slot: usize| slot == 1 || regs.pipelines[slot].enable() != 0;
        let last_pre_raster = (1..5).rev().find(|&s| enabled(s)).unwrap_or(1);
        let color_outputs: [AttributeType; 8] = std::array::from_fn(|i| {
            let count = (regs.rt_control.count() as usize).min(8);
            if i >= count {
                return AttributeType::Float;
            }
            let rt = &regs.rt[render_target_index(regs, i)];
            match ImageInfo::from_render_target(rt, 0, self.texture_cache.gob_layout())
                .map(|info| info.format.numeric())
            {
                Some(Numeric::Uint) => AttributeType::UnsignedInt,
                Some(Numeric::Sint) => AttributeType::SignedInt,
                _ => AttributeType::Float,
            }
        });
        let features = *self.draw.pipelines.features();
        let arch = self.draw.arch;
        let mut bindings = Bindings::default();
        let mut previous: Option<Arc<ShaderModule>> = None;
        for (slot, host) in GRAPHICS_SLOTS {
            if !enabled(slot) {
                self.draw.stages[host] = None;
                continue;
            }
            let env = ProgramEnv::graphics(ctx, slot, arch);
            let vertex_a = (slot == 1 && regs.pipelines[0].enable() != 0)
                .then(|| ProgramEnv::graphics(ctx, 0, arch));
            let state = StageState::new(
                regs,
                slot,
                slot == last_pre_raster,
                previous.as_deref(),
                topology,
                color_outputs,
                &features,
            );
            let module = self.draw.shaders.module(
                vertex_a.as_ref(),
                &env,
                state.key(),
                || state.runtime(previous.as_deref().map(|m| &m.info)),
                &mut bindings,
            );
            let Some(module) = module else {
                return false;
            };
            if slot != 5 {
                previous = Some(module.clone());
            }
            self.draw.stages[host] = Some(module);
        }
        true
    }

    /// Finds the colour and depth targets in the texture cache. `false` skips the draw.
    fn render_targets(&mut self, ctx: &GraphicsContext<'_>) -> Result<bool> {
        let regs = ctx.regs;
        let msaa = regs.anti_alias_samples_mode.0;
        let gob = self.texture_cache.gob_layout();
        self.draw.colors = [None; 8];
        self.draw.depth = None;
        let count = (regs.rt_control.count() as usize).min(8);
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
        };
        for i in 0..count {
            let rt = &regs.rt[render_target_index(regs, i)];
            let Some(info) = ImageInfo::from_render_target(rt, msaa, gob) else {
                continue;
            };
            let Some(found) = self.texture_cache.render_target(
                &mut tx,
                ctx.memory,
                rt.address(),
                &info,
                false,
            )?
            else {
                continue;
            };
            self.draw.colors[i] = Self::target_view(&mut self.texture_cache, found, &info)?;
        }
        if regs.zeta_enable != 0
            && let Some(info) = ImageInfo::from_zeta(&regs.zeta, &regs.zeta_size, msaa, gob)
            && let Some(found) = self.texture_cache.render_target(
                &mut tx,
                ctx.memory,
                regs.zeta.address(),
                &info,
                false,
            )?
        {
            self.draw.depth = Self::target_view(&mut self.texture_cache, found, &info)?;
        }
        Ok(true)
    }

    fn target_view(
        cache: &mut crate::texture_cache::TextureCache,
        found: crate::texture_cache::Found,
        info: &ImageInfo,
    ) -> Result<Option<Target>> {
        let Some(state) = cache.state(found.id) else {
            return Ok(None);
        };
        let mut view = ViewInfo::whole(info);
        view.base_level += found.level;
        view.levels = 1;
        view.base_layer += found.layer;
        view.layers = 1;
        view.view_type = vk::ImageViewType::TYPE_2D;
        let Some(raw) = cache.view(found.id, &view)? else {
            return Ok(None);
        };
        let extent = vk::Extent2D {
            width: (state.extent.width >> found.level).max(1),
            height: (state.extent.height >> found.level).max(1),
        };
        Ok(Some(Target {
            id: found.id,
            view: raw,
            format: state.format,
            extent,
        }))
    }

    /// The rendered area: the smallest target, or the surface clip without targets.
    fn render_area(&self, regs: &Regs) -> vk::Extent2D {
        let draw = &self.draw;
        let targets = draw.colors.iter().flatten().chain(draw.depth.iter());
        targets
            .map(|t| t.extent)
            .reduce(|a, b| vk::Extent2D {
                width: a.width.min(b.width),
                height: a.height.min(b.height),
            })
            .unwrap_or(vk::Extent2D {
                width: regs.surface_clip.bits.width().max(1),
                height: regs.surface_clip.bits1.height().max(1),
            })
    }

    fn target_scale(&self) -> ResolutionScale {
        let draw = &self.draw;
        draw.colors
            .iter()
            .flatten()
            .chain(draw.depth.iter())
            .find_map(|t| self.texture_cache.state(t.id))
            .map_or(ResolutionScale::X1, |s| s.scale)
    }

    /// Appends `module`'s descriptors to the current draw's writes. `false` skips the draw.
    fn bind_resources(&mut self, module: &ShaderModule, src: &StageSources<'_>) -> Result<bool> {
        let info = &module.info;
        for set in &module.resources.sets {
            for b in &set.bindings {
                let descriptor = b.descriptor as usize;
                let first = match b.kind {
                    DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer => {
                        self.draw.buffer_infos.len()
                    }
                    _ => self.draw.image_infos.len(),
                };
                match b.kind {
                    DescriptorKind::UniformBuffer => {
                        let desc = info.constant_buffer_descriptors[descriptor];
                        for i in 0..b.count {
                            let cbuf = src
                                .cbufs
                                .get((desc.index + i) as usize)
                                .copied()
                                .unwrap_or_default();
                            let binding = if cbuf.enabled {
                                self.buffer_cache.uniform_buffer(
                                    &mut self.scheduler,
                                    &mut self.staging,
                                    src.gmmu,
                                    cbuf.address,
                                    u64::from(cbuf.size),
                                )?
                            } else {
                                None
                            };
                            self.push_buffer(binding);
                        }
                    }
                    DescriptorKind::StorageBuffer => {
                        let desc = info.storage_buffers_descriptors[descriptor];
                        for i in 0..b.count {
                            let (address, size) = storage_pointer(src, &desc, i);
                            let binding = self.buffer_cache.storage_buffer(
                                &mut self.scheduler,
                                &mut self.staging,
                                src.gmmu,
                                address,
                                size,
                                desc.is_written,
                            )?;
                            self.push_buffer(binding);
                        }
                    }
                    DescriptorKind::CombinedImageSampler => {
                        let desc = info.texture_descriptors[descriptor];
                        for i in 0..b.count {
                            let h = desc.handle;
                            let offset = h.cbuf_offset + (i << desc.size_shift);
                            let mut raw =
                                read_cbuf::<u32>(src.gmmu, src.cbufs, h.cbuf_index, offset)
                                    << h.shift_left;
                            if let Some((index, offset, shift)) = h.secondary {
                                raw |=
                                    read_cbuf::<u32>(src.gmmu, src.cbufs, index, offset) << shift;
                            }
                            let texture = if desc.handle_is_pointer_word {
                                None
                            } else {
                                self.texture_view(src, raw, desc.ty)?
                            };
                            if let Some((id, _)) = texture
                                && self
                                    .texture_cache
                                    .state(id)
                                    .is_some_and(|s| s.scale != ResolutionScale::X1)
                            {
                                self.draw.warn_once(
                                    Warning::RescaledTexture,
                                    "a draw samples a rescaled image, which needs per-binding scale bits; skipping it",
                                );
                                return Ok(false);
                            }
                            let view = match texture {
                                Some((id, view)) => {
                                    self.draw.textures.push(id);
                                    view
                                }
                                None if desc.ty == TextureType::Color2D && !desc.is_depth => {
                                    self.draw.null_view.raw()
                                }
                                None => {
                                    self.draw.warn_once(
                                        Warning::NullTexture,
                                        "a draw samples an unbound non-2D texture; skipping it",
                                    );
                                    return Ok(false);
                                }
                            };
                            let (_, sampler_index) = src.pool.split(raw);
                            let sampler = self.sampler(src, sampler_index)?;
                            self.draw.image_infos.push(vk::DescriptorImageInfo {
                                sampler,
                                image_view: view,
                                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            });
                        }
                    }
                    DescriptorKind::UniformTexelBuffer
                    | DescriptorKind::StorageTexelBuffer
                    | DescriptorKind::StorageImage => {
                        self.draw.warn_once(
                            Warning::UnsupportedBinding,
                            "texel buffers and storage images are not bound yet; skipping draws that use them",
                        );
                        return Ok(false);
                    }
                }
                self.draw.writes.push(Write {
                    binding: b.binding,
                    kind: b.kind,
                    first,
                    count: b.count,
                });
            }
        }
        Ok(true)
    }

    fn push_buffer(&mut self, binding: Option<crate::buffer_cache::BufferBinding>) {
        let info = match binding {
            Some(b) => vk::DescriptorBufferInfo {
                buffer: b.buffer,
                offset: b.offset,
                range: b.size,
            },
            None => vk::DescriptorBufferInfo {
                buffer: self.draw.null_buffer.raw(),
                offset: 0,
                range: NULL_BUFFER_SIZE,
            },
        };
        self.draw.buffer_infos.push(info);
    }

    /// The image and view of the texture `raw` selects, read as `ty`; `None` when nothing
    /// backs it.
    fn texture_view(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        ty: TextureType,
    ) -> Result<Option<(ImageId, vk::ImageView)>> {
        let (index, _) = src.pool.split(raw);
        if index > src.pool.limit {
            return Ok(None);
        }
        let header = src.pool.header(src.gmmu, raw);
        if header.address == 0 {
            return Ok(None);
        }
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
        };
        let Some((id, mut view)) = self.texture_cache.texture(&mut tx, src.gmmu, &header)? else {
            return Ok(None);
        };
        view.view_type = view_type(ty);
        view.layers = match view.view_type {
            vk::ImageViewType::TYPE_1D
            | vk::ImageViewType::TYPE_2D
            | vk::ImageViewType::TYPE_3D => 1,
            vk::ImageViewType::CUBE if view.layers >= 6 => 6,
            vk::ImageViewType::CUBE_ARRAY if view.layers >= 6 => view.layers / 6 * 6,
            vk::ImageViewType::CUBE | vk::ImageViewType::CUBE_ARRAY => return Ok(None),
            _ => view.layers,
        };
        Ok(self.texture_cache.view(id, &view)?.map(|raw| (id, raw)))
    }

    /// The host sampler for sampler header `index`, created on first use.
    fn sampler(&mut self, src: &StageSources<'_>, index: u32) -> Result<vk::Sampler> {
        let words = if index <= src.sampler_limit {
            src.gmmu
                .read::<[u32; 8]>(src.samplers + u64::from(index) * 32)
                .unwrap_or_default()
        } else {
            [0; 8]
        };
        let key = (words, self.draw.anisotropy as u8);
        if let Some(&sampler) = self.draw.samplers.get(&key) {
            return Ok(sampler);
        }
        let params = SamplerParams::new(&Tsc(words), self.draw.anisotropy, &self.draw.sampler_caps);
        let sampler = params.create(&self.device)?;
        self.draw.samplers.insert(key, sampler);
        Ok(sampler)
    }

    /// Binds every vertex stream an enabled attribute reads, into the draw's vertex lists.
    fn vertex_buffers(&mut self, gmmu: &lichii_gpu::gmmu::Gmmu, regs: &Regs) -> Result<()> {
        let used = regs
            .vertex_attrib_format
            .iter()
            .filter(|a| a.constant() == 0 && mvk::vertex_format(a.type_(), a.size()).is_some())
            .fold(0u32, |used, a| used | 1 << a.buffer());
        self.draw.vertex_buffers.clear();
        self.draw.vertex_offsets.clear();
        self.draw.vertex_strides.clear();
        let count = 32 - used.leading_zeros() as usize;
        for stream in 0..count {
            let range = (used & 1 << stream != 0)
                .then(|| vertex_stream_range(regs, stream))
                .flatten();
            let binding = match range {
                Some((address, size)) => self.buffer_cache.vertex_buffer(
                    &mut self.scheduler,
                    &mut self.staging,
                    gmmu,
                    address,
                    size,
                )?,
                None => None,
            };
            let (buffer, offset) =
                binding.map_or((self.draw.null_buffer.raw(), 0), |b| (b.buffer, b.offset));
            self.draw.vertex_buffers.push(buffer);
            self.draw.vertex_offsets.push(offset);
            self.draw
                .vertex_strides
                .push(u64::from(regs.vertex_streams[stream].stride()));
        }
        Ok(())
    }

    /// The whole index buffer of an indexed indirect draw, whose counts live in GPU memory.
    fn indirect_index_buffer(
        &mut self,
        gmmu: &lichii_gpu::gmmu::Gmmu,
        regs: &Regs,
        draw: &IndirectDraw,
    ) -> Result<Option<IndexBinding>> {
        if !draw.indexed {
            return Ok(None);
        }
        let width = match regs.index_buffer.format {
            IndexFormat::UNSIGNED_BYTE => 1,
            IndexFormat::UNSIGNED_SHORT => 2,
            _ => 4,
        };
        let buffer = &regs.index_buffer;
        let count = buffer
            .limit_addr()
            .wrapping_add(1)
            .saturating_sub(buffer.start_addr())
            / width;
        let params = DrawParams {
            topology: draw.topology,
            indexed: true,
            first: 0,
            count: count.min(u64::from(u32::MAX)) as u32,
            base_vertex: 0,
            base_instance: 0,
            instance_count: 1,
            inline_indices: None,
        };
        self.buffer_cache
            .index_buffer(&mut self.scheduler, &mut self.staging, gmmu, regs, &params)
    }

    /// The command buffer and count buffer of an indirect draw. `None` skips the draw.
    #[allow(clippy::type_complexity)]
    fn indirect_buffers(
        &mut self,
        gmmu: &lichii_gpu::gmmu::Gmmu,
        draw: &IndirectDraw,
    ) -> Result<
        Option<(
            crate::buffer_cache::BufferBinding,
            Option<crate::buffer_cache::BufferBinding>,
        )>,
    > {
        if assembled_topology(draw.topology) != draw.topology {
            self.draw.warn_once(
                Warning::IndirectAssembly,
                "indirect draws of quads and line loops are not assembled yet; skipping them",
            );
            return Ok(None);
        }
        if draw.count_buffer.is_some() && !self.device.features.draw_indirect_count {
            self.draw.warn_once(
                Warning::NoIndirectCount,
                "the device cannot take indirect draw counts; skipping those draws",
            );
            return Ok(None);
        }
        let command = if draw.indexed { 20 } else { 16 };
        let size = u64::from(draw.stride) * u64::from(draw.max_draws.saturating_sub(1)) + command;
        let mut acquire = |address: GpuVAddr, size: u64| -> Result<Option<_>> {
            if draw.max_draws == 0 || !gmmu.is_continuous(address, size) {
                return Ok(None);
            }
            let Some(daddr) = gmmu.translate(address) else {
                return Ok(None);
            };
            self.buffer_cache
                .acquire(&mut self.scheduler, &mut self.staging, daddr, size)
                .map(Some)
        };
        let Some(commands) = acquire(draw.buffer, size)? else {
            self.draw.warn_once(
                Warning::IndirectScattered,
                "indirect draw commands over scattered memory are not supported; skipping them",
            );
            return Ok(None);
        };
        let count = match draw.count_buffer {
            Some(address) => match acquire(address, 4)? {
                Some(count) => Some(count),
                None => return Ok(None),
            },
            None => None,
        };
        Ok(Some((commands, count)))
    }

    /// Allocates and writes the draw's descriptor set.
    fn descriptor_set(
        &mut self,
        layout: Option<vk::DescriptorSetLayout>,
    ) -> Result<Option<vk::DescriptorSet>> {
        let Some(layout) = layout else {
            return Ok(None);
        };
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, layout)?;
        let draw = &mut self.draw;
        draw.vk_writes.clear();
        for write in &draw.writes {
            let mut vk_write = vk::WriteDescriptorSet {
                dst_set: set,
                dst_binding: write.binding,
                descriptor_count: write.count,
                descriptor_type: crate::pipeline::descriptor_type(write.kind),
                ..Default::default()
            };
            match write.kind {
                DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer => {
                    vk_write.p_buffer_info = draw.buffer_infos[write.first..].as_ptr();
                }
                _ => vk_write.p_image_info = draw.image_infos[write.first..].as_ptr(),
            }
            draw.vk_writes.push(vk_write);
        }
        // SAFETY: every write points into the draw's info lists, which outlive the call and
        // hold `count` infos from `first`; the set is live and unused by recorded work.
        unsafe { self.device.raw.update_descriptor_sets(&draw.vk_writes, &[]) };
        Ok(Some(set))
    }

    /// Pushes the one range every stage of `layout` shares (`push` is its stages and size): the
    /// rescaling words at 0 (no texture or image rescaled) with the down factor of `scale` at 24,
    /// and the render area at `RENDER_AREA_LAYOUT_OFFSET` (32), so a 48-byte range holds both.
    fn push_constants(
        &mut self,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        (stages, size): (vk::ShaderStageFlags, u32),
        rescaling: bool,
        render_area: Option<(f32, f32)>,
        scale: ResolutionScale,
    ) {
        if size == 0 {
            return;
        }
        let mut bytes = [0u8; RENDER_AREA_LAYOUT_OFFSET as usize + 16];
        if rescaling {
            let (num, den) = scale.ratio();
            let down = den as f32 / num as f32;
            let at = RESCALING_LAYOUT_DOWN_FACTOR_OFFSET as usize;
            bytes[at..at + 4].copy_from_slice(&down.to_le_bytes());
        }
        if let Some((width, height)) = render_area {
            let at = RENDER_AREA_LAYOUT_OFFSET as usize;
            bytes[at..at + 4].copy_from_slice(&width.to_le_bytes());
            bytes[at + 4..at + 8].copy_from_slice(&height.to_le_bytes());
        }
        // SAFETY: `stages` and `size` are the layout's one push constant range at offset 0.
        unsafe {
            self.device.raw.cmd_push_constants(
                cmd,
                layout,
                stages,
                0,
                &bytes[..(size as usize).min(bytes.len())],
            );
        }
    }

    /// Sets every dynamic state `features` declares from the registers.
    fn set_dynamic_state(
        &mut self,
        cmd: vk::CommandBuffer,
        regs: &Regs,
        features: &PipelineFeatures,
        topology: vk::PrimitiveTopology,
        extent: vk::Extent2D,
        scale: ResolutionScale,
    ) {
        let raw = &self.device.raw;
        let draw = &mut self.draw;
        let factor = {
            let (num, den) = scale.ratio();
            num as f32 / den as f32
        };
        let src = &regs.viewport_transform[0];
        let mut x = (src.translate_x - src.scale_x) * factor;
        let mut width = src.scale_x * 2.0 * factor;
        if width < 0.0 {
            x += width;
            width = -width;
        }
        let mut y = (src.translate_y - src.scale_y) * factor;
        let mut height = src.scale_y * 2.0 * factor;
        if regs.window_origin.mode() != WindowOriginMode::UPPER_LEFT {
            y += regs.surface_clip.bits1.height() as f32 * factor;
            height = -height;
        }
        let reduce_z = if regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE {
            1.0
        } else {
            0.0
        };
        let viewport = vk::Viewport {
            x,
            y,
            width: if width == 0.0 { 1.0 } else { width },
            height: if height == 0.0 { 1.0 } else { height },
            min_depth: (src.translate_z - src.scale_z * reduce_z).clamp(0.0, 1.0),
            max_depth: (src.translate_z + src.scale_z).clamp(0.0, 1.0),
        };
        let full = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        };
        let scissor_regs = &regs.scissor_test[0];
        let scissor = if scissor_regs.enable != 0 {
            let s = |v: u32| (v as f32 * factor) as u32;
            let (x0, x1) = (s(scissor_regs.min_x()), s(scissor_regs.max_x()));
            let (y0, y1) = (s(scissor_regs.min_y()), s(scissor_regs.max_y()));
            vk::Rect2D {
                offset: vk::Offset2D {
                    x: x0.min(i32::MAX as u32) as i32,
                    y: y0.min(i32::MAX as u32) as i32,
                },
                extent: vk::Extent2D {
                    width: x1.saturating_sub(x0),
                    height: y1.saturating_sub(y0),
                },
            }
        } else {
            full
        };
        let two_sided = regs.stencil_two_side_enable != 0;
        let (back_ref, back_write, back_func) = if two_sided {
            (
                regs.stencil_back_ref,
                regs.stencil_back_mask,
                regs.stencil_back_func_mask,
            )
        } else {
            (
                regs.stencil_front_ref,
                regs.stencil_front_mask,
                regs.stencil_front_func_mask,
            )
        };
        use vk::StencilFaceFlags as F;
        let line_width = if self.device.features.wide_lines {
            regs.line_width_aliased.max(1.0)
        } else {
            1.0
        };
        let color = &regs.blend_color;
        let (min_bound, max_bound) = (
            regs.depth_bounds[0].clamp(0.0, 1.0),
            regs.depth_bounds[1].clamp(0.0, 1.0),
        );
        // SAFETY: `cmd` is recording inside a dynamic rendering with a pipeline that declares
        // these dynamic states; extension commands run only when their feature is on.
        unsafe {
            raw.cmd_set_line_width(cmd, line_width);
            raw.cmd_set_depth_bias(
                cmd,
                regs.depth_bias,
                regs.depth_bias_clamp,
                regs.slope_scale_depth_bias,
            );
            raw.cmd_set_blend_constants(cmd, &[color.r, color.g, color.b, color.a]);
            raw.cmd_set_depth_bounds(cmd, min_bound, max_bound.max(min_bound));
            raw.cmd_set_stencil_compare_mask(cmd, F::FRONT, regs.stencil_front_func_mask);
            raw.cmd_set_stencil_compare_mask(cmd, F::BACK, back_func);
            raw.cmd_set_stencil_write_mask(cmd, F::FRONT, regs.stencil_front_mask);
            raw.cmd_set_stencil_write_mask(cmd, F::BACK, back_write);
            raw.cmd_set_stencil_reference(cmd, F::FRONT, regs.stencil_front_ref);
            raw.cmd_set_stencil_reference(cmd, F::BACK, back_ref);
            if features.extended_dynamic_state {
                raw.cmd_set_viewport_with_count(cmd, &[viewport]);
                raw.cmd_set_scissor_with_count(cmd, &[scissor]);
                let cull = if regs.gl_cull_test_enabled != 0 {
                    mvk::cull_mode(regs.gl_cull_face).unwrap_or(vk::CullModeFlags::BACK)
                } else {
                    vk::CullModeFlags::NONE
                };
                raw.cmd_set_cull_mode(cmd, cull);
                let mut face =
                    mvk::front_face(regs.gl_front_face).unwrap_or(vk::FrontFace::COUNTER_CLOCKWISE);
                if regs.window_origin.flip_y() != 0 {
                    face = if face == vk::FrontFace::CLOCKWISE {
                        vk::FrontFace::COUNTER_CLOCKWISE
                    } else {
                        vk::FrontFace::CLOCKWISE
                    };
                }
                raw.cmd_set_front_face(cmd, face);
                raw.cmd_set_primitive_topology(cmd, topology);
                raw.cmd_set_depth_test_enable(cmd, regs.depth_test_enable != 0);
                raw.cmd_set_depth_write_enable(cmd, regs.depth_write_enabled != 0);
                raw.cmd_set_depth_compare_op(
                    cmd,
                    mvk::comparison_op(regs.depth_test_func).unwrap_or(vk::CompareOp::ALWAYS),
                );
                raw.cmd_set_depth_bounds_test_enable(
                    cmd,
                    features.depth_bounds && regs.depth_bounds_enable != 0,
                );
                raw.cmd_set_stencil_test_enable(cmd, regs.stencil_enable != 0);
                let op = |o: &StencilOp| {
                    let s = |op| mvk::stencil_op(op).unwrap_or(vk::StencilOp::KEEP);
                    (
                        s(o.fail),
                        s(o.zpass),
                        s(o.zfail),
                        mvk::comparison_op(o.func).unwrap_or(vk::CompareOp::ALWAYS),
                    )
                };
                let front = op(&regs.stencil_front_op);
                let back = if two_sided {
                    op(&regs.stencil_back_op)
                } else {
                    front
                };
                raw.cmd_set_stencil_op(cmd, F::FRONT, front.0, front.1, front.2, front.3);
                raw.cmd_set_stencil_op(cmd, F::BACK, back.0, back.1, back.2, back.3);
            } else {
                raw.cmd_set_viewport(cmd, 0, &[viewport]);
                raw.cmd_set_scissor(cmd, 0, &[scissor]);
            }
            if features.extended_dynamic_state2 {
                use vk::PrimitiveTopology as T;
                raw.cmd_set_rasterizer_discard_enable(cmd, regs.rasterize_enable == 0);
                let offset = match topology {
                    T::POINT_LIST => regs.polygon_offset_point_enable,
                    T::LINE_LIST
                    | T::LINE_STRIP
                    | T::LINE_LIST_WITH_ADJACENCY
                    | T::LINE_STRIP_WITH_ADJACENCY => regs.polygon_offset_line_enable,
                    _ => regs.polygon_offset_fill_enable,
                };
                raw.cmd_set_depth_bias_enable(cmd, offset != 0);
                let restartable = matches!(
                    topology,
                    T::LINE_STRIP
                        | T::TRIANGLE_STRIP
                        | T::TRIANGLE_FAN
                        | T::LINE_STRIP_WITH_ADJACENCY
                        | T::TRIANGLE_STRIP_WITH_ADJACENCY
                );
                raw.cmd_set_primitive_restart_enable(
                    cmd,
                    restartable && regs.primitive_restart.enabled != 0,
                );
            }
            if !draw.vertex_buffers.is_empty() {
                if features.extended_dynamic_state && !features.dynamic_vertex_input {
                    raw.cmd_bind_vertex_buffers2(
                        cmd,
                        0,
                        &draw.vertex_buffers,
                        &draw.vertex_offsets,
                        None,
                        Some(&draw.vertex_strides),
                    );
                } else {
                    raw.cmd_bind_vertex_buffers(cmd, 0, &draw.vertex_buffers, &draw.vertex_offsets);
                }
            }
            if features.dynamic_vertex_input {
                draw.vertex_bindings.clear();
                draw.vertex_attributes.clear();
                for (location, a) in regs.vertex_attrib_format.iter().enumerate() {
                    let Some(format) = (a.constant() == 0)
                        .then(|| mvk::vertex_format(a.type_(), a.size()))
                        .flatten()
                    else {
                        continue;
                    };
                    draw.vertex_attributes.push(
                        vk::VertexInputAttributeDescription2EXT::default()
                            .location(location as u32)
                            .binding(a.buffer())
                            .format(format)
                            .offset(a.offset()),
                    );
                }
                for (binding, &stride) in draw.vertex_strides.iter().enumerate() {
                    let instanced = regs.vertex_stream_instances.is_instanced[binding] != 0;
                    draw.vertex_bindings.push(
                        vk::VertexInputBindingDescription2EXT::default()
                            .binding(binding as u32)
                            .stride(stride as u32)
                            .input_rate(if instanced {
                                vk::VertexInputRate::INSTANCE
                            } else {
                                vk::VertexInputRate::VERTEX
                            })
                            .divisor(1),
                    );
                }
                draw.vertex_input.cmd_set_vertex_input(
                    cmd,
                    &draw.vertex_bindings,
                    &draw.vertex_attributes,
                );
            }
            let attachments = draw.attachments.len() as u32;
            if features.dynamic_blend && attachments > 0 {
                draw.blend_enables.clear();
                draw.blend_equations.clear();
                draw.write_masks.clear();
                let common = regs.color_mask_common != 0;
                for index in 0..attachments as usize {
                    let mask = regs.color_mask[if common { 0 } else { index }];
                    draw.write_masks.push(vk::ColorComponentFlags::from_raw(
                        mask.r() | mask.g() << 1 | mask.b() << 2 | mask.a() << 3,
                    ));
                    let b = &regs.blend;
                    let t = &regs.blend_per_target[index];
                    let per_target = regs.blend_per_target_enabled != 0;
                    let (color_op, src_color, dst_color, alpha_op, src_alpha, dst_alpha) =
                        if per_target {
                            (
                                t.color_op,
                                t.color_source,
                                t.color_dest,
                                t.alpha_op,
                                t.alpha_source,
                                t.alpha_dest,
                            )
                        } else {
                            (
                                b.color_op,
                                b.color_source,
                                b.color_dest,
                                b.alpha_op,
                                b.alpha_source,
                                b.alpha_dest,
                            )
                        };
                    let factor = |f| {
                        let host = mvk::blend_factor(f).unwrap_or(vk::BlendFactor::ONE);
                        if features.dual_src_blend {
                            host
                        } else {
                            mvk::single_source(host)
                        }
                    };
                    let op = |o| mvk::blend_equation(o).unwrap_or(vk::BlendOp::ADD);
                    let blendable = draw.colors[index].is_some_and(|t| {
                        self.device
                            .format_features(t.format)
                            .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
                    });
                    draw.blend_enables
                        .push(vk::Bool32::from(regs.blend.enable[index] != 0 && blendable));
                    draw.blend_equations.push(vk::ColorBlendEquationEXT {
                        src_color_blend_factor: factor(src_color),
                        dst_color_blend_factor: factor(dst_color),
                        color_blend_op: op(color_op),
                        src_alpha_blend_factor: factor(src_alpha),
                        dst_alpha_blend_factor: factor(dst_alpha),
                        alpha_blend_op: op(alpha_op),
                    });
                }
                draw.eds3
                    .cmd_set_color_blend_enable(cmd, 0, &draw.blend_enables);
                draw.eds3
                    .cmd_set_color_blend_equation(cmd, 0, &draw.blend_equations);
                draw.eds3
                    .cmd_set_color_write_mask(cmd, 0, &draw.write_masks);
            }
            let alpha = regs.anti_alias_alpha_control;
            if features.dynamic_depth_clamp {
                use lichii_gpu::engines::maxwell_3d::regs::ViewportClipControlGeometryClip as G;
                let clamps = !matches!(
                    regs.viewport_clip_control.geometry_clip(),
                    G::PASSTHROUGH | G::FRUSTUM_XYZ | G::FRUSTUM_Z
                );
                draw.eds3
                    .cmd_set_depth_clamp_enable(cmd, features.depth_clamp && clamps);
            }
            if features.dynamic_logic_op_enable {
                draw.eds3
                    .cmd_set_logic_op_enable(cmd, features.logic_op && regs.logic_op.enable != 0);
            }
            if features.dynamic_alpha_to_coverage {
                draw.eds3
                    .cmd_set_alpha_to_coverage_enable(cmd, alpha.alpha_to_coverage() != 0);
            }
            if features.dynamic_alpha_to_one {
                draw.eds3.cmd_set_alpha_to_one_enable(
                    cmd,
                    features.alpha_to_one && alpha.alpha_to_one() != 0,
                );
            }
            if features.dynamic_polygon_mode {
                let mode = if features.fill_mode_non_solid {
                    mvk::polygon_mode(regs.polygon_mode_front).unwrap_or(vk::PolygonMode::FILL)
                } else {
                    vk::PolygonMode::FILL
                };
                draw.eds3.cmd_set_polygon_mode(cmd, mode);
            }
            if features.dynamic_depth_clip_negative_one_to_one {
                draw.eds3.cmd_set_depth_clip_negative_one_to_one(
                    cmd,
                    regs.depth_mode == DepthMode::MINUS_ONE_TO_ONE,
                );
            }
            if features.dynamic_provoking_vertex {
                draw.eds3.cmd_set_provoking_vertex_mode(
                    cmd,
                    if regs.provoking_vertex == ProvokingVertex::LAST
                        && features.provoking_vertex_last
                    {
                        vk::ProvokingVertexModeEXT::LAST_VERTEX
                    } else {
                        vk::ProvokingVertexModeEXT::FIRST_VERTEX
                    },
                );
            }
        }
    }

    pub(super) fn dispatch(&mut self, ctx: &ComputeContext<'_>) -> Result<()> {
        let env = ProgramEnv::compute(ctx);
        let mut bindings = Bindings::default();
        let Some(module) =
            self.draw
                .shaders
                .module(None, &env, 0, RuntimeInfo::default, &mut bindings)
        else {
            return Ok(());
        };
        let key = ComputePipelineKey {
            shader: module.hash,
            workgroup: ctx.block,
            shared_memory: ctx.shared_memory_size,
        };
        let Some(pipeline) = self.draw.pipelines.compute(&key, &module) else {
            return Ok(());
        };
        let pipeline_raw = pipeline.raw();
        let layout_raw = pipeline.layout().raw();
        let push = pipeline.layout().push_constants();
        let set_layout = pipeline.layout().set_layouts().first().copied();
        let sources = StageSources {
            gmmu: ctx.memory,
            cbufs: ctx.const_buffers,
            pool: *env.pool(),
            samplers: ctx.samplers,
            sampler_limit: ctx.sampler_limit,
        };
        loop {
            let merged = self.buffer_cache.stats().merged_buffers;
            self.draw.buffer_infos.clear();
            self.draw.image_infos.clear();
            self.draw.writes.clear();
            self.draw.textures.clear();
            if !self.bind_resources(&module, &sources)? {
                return Ok(());
            }
            if self.buffer_cache.stats().merged_buffers == merged {
                break;
            }
        }
        let set = self.descriptor_set(set_layout)?;
        let cmd = self.scheduler.record()?;
        for &id in &self.draw.textures {
            self.texture_cache.prepare(cmd, id, SHADER_SAMPLED);
        }
        let raw = &self.device.raw;
        let [x, y, z] = ctx.grid;
        // SAFETY: `cmd` is recording outside any rendering; the set and pipeline are live
        // until the scheduler's current tick completes.
        unsafe {
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_raw);
            if let Some(set) = set {
                raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout_raw,
                    0,
                    &[set],
                    &[],
                );
            }
        }
        self.push_constants(cmd, layout_raw, push, false, None, ResolutionScale::X1);
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let raw = &self.device.raw;
        // SAFETY: as above.
        unsafe {
            raw.cmd_dispatch(cmd, x, y, z);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }
        self.unflushed = true;
        Ok(())
    }
}

/// The GPU address and size of storage buffer `element` of `desc`: 16-byte NVN descriptors in
/// the constant buffer, or (indirect) a pointer table the constant buffer points to.
fn storage_pointer(
    src: &StageSources<'_>,
    desc: &StorageBufferDescriptor,
    element: u32,
) -> (GpuVAddr, u64) {
    let offset = desc.cbuf_offset + element * 16;
    let pointer: u64 = read_cbuf(src.gmmu, src.cbufs, desc.cbuf_index, offset);
    if desc.indirect_offset == StorageBufferDescriptor::DIRECT {
        let size: u32 = read_cbuf(src.gmmu, src.cbufs, desc.cbuf_index, offset + 8);
        return (pointer, u64::from(size));
    }
    let address = src
        .gmmu
        .read::<u64>(pointer + u64::from(desc.indirect_offset))
        .unwrap_or(0);
    (address, 64 << 10)
}

mod texture;

#[cfg(test)]
mod tests;
