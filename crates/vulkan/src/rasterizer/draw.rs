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
use lichii_gpu::engines::maxwell_3d::regs::{MESH_CONTROL, MESH_SHADER_A};
use lichii_shader::backend::Bindings;
use lichii_shader::backend::spirv::{
    NUM_IMAGE_SCALING_WORDS, NUM_TEXTURE_SCALING_WORDS, RENDER_AREA_LAYOUT_OFFSET,
    RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, RESCALING_LAYOUT_IMAGES_OFFSET,
    RESCALING_LAYOUT_TEXTURES_OFFSET,
};
use lichii_shader::info::{HandleSource, ImageDescriptor, StorageBufferDescriptor};
use lichii_shader::ir::{Attribute, ImageFormat, TextureType};
use lichii_shader::runtime::{AttributeType, CompareFunction, InputTopology};
use lichii_shader::{
    DescriptorKind, RuntimeInfo, ShaderArch, ShaderInfo, ShaderModule, ShaderStage,
};
use lichii_texture::block_linear::GobLayout;
use lichii_texture::tic::TextureMemory;
use lichii_texture::{Numeric, PixelFormat};

use super::VulkanRasterizer;
use crate::buffer_cache::{BufferBinding, IndexBinding, assembled_topology, vertex_stream_range};
use crate::commands::ImageAccess;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::memory::{Buffer, BufferView};
use crate::pipeline::environment::{ProgramEnv, TexturePool, texture_type};
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
/// How often a draw or dispatch looks its images up again after a lookup removed one it held,
/// before it gives up: a render target and a texture over the same memory in layouts the
/// texture cache cannot share replace each other on every lookup.
const MAX_RETARGETS: u32 = 2;

const SHADER_SAMPLED: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::SHADER_SAMPLED_READ,
};

const SHADER_STORAGE: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::from_raw(
        vk::AccessFlags2::SHADER_STORAGE_READ.as_raw()
            | vk::AccessFlags2::SHADER_STORAGE_WRITE.as_raw(),
    ),
};

/// Warnings each logged once per renderer.
#[derive(Clone, Copy)]
enum Warning {
    NullTexture = 1 << 1,
    IndirectAssembly = 1 << 2,
    IndirectScattered = 1 << 3,
    NoIndirectCount = 1 << 4,
    DrawTextureTarget = 1 << 7,
    DrawTextureSource = 1 << 8,
    MixedScaleArray = 1 << 9,
    TexelBufferFormat = 1 << 10,
    NullStorageImage = 1 << 11,
    StorageAndSampled = 1 << 12,
    NoMeshShader = 1 << 13,
    IndirectMesh = 1 << 14,
    MeshGroupLimit = 1 << 15,
    ImageChurn = 1 << 16,
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

/// Rescaling words of the push constant block, as Lichii's `RescalingPushConstant`, for
/// texture descriptors (`N` = `NUM_TEXTURE_SCALING_WORDS`) or storage image descriptors
/// (`NUM_IMAGE_SCALING_WORDS`): bit `n` is set when the descriptor with scaling index `n` reads
/// a rescaled image, so the shader scales its texel coordinates and size queries to it.
/// Scaling indices number the descriptors of the draw's stages in stage order, one per
/// descriptor however many elements it has, as `Bindings::texture_scaling_index` and
/// `image_scaling_index` count them when the stages translate.
#[derive(Clone, Copy)]
struct ScalingBits<const N: usize> {
    words: [u32; N],
    stage_base: u32,
}

impl<const N: usize> Default for ScalingBits<N> {
    fn default() -> Self {
        Self {
            words: [0; N],
            stage_base: 0,
        }
    }
}

impl<const N: usize> ScalingBits<N> {
    /// Marks descriptor `descriptor` of the current stage as reading a rescaled image.
    fn set(&mut self, descriptor: u32) {
        let index = self.stage_base + descriptor;
        if let Some(word) = self.words.get_mut((index / 32) as usize) {
            *word |= 1 << (index % 32);
        }
    }

    /// Moves past the current stage's `count` descriptors.
    fn next_stage(&mut self, count: usize) {
        self.stage_base += count as u32;
    }
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
    mesh: ash::ext::mesh_shader::Device,
    null_buffer: Buffer,
    null_images: null::NullImages,
    arch: ShaderArch,
    channel: Option<ChannelId>,
    warned: u32,
    stages: GraphicsShaders,
    colors: [Option<Target>; 8],
    depth: Option<Target>,
    textures: Vec<ImageId>,
    texture_scaling: ScalingBits<{ NUM_TEXTURE_SCALING_WORDS as usize }>,
    /// Storage images of the current draw and whether it writes them.
    storage_images: Vec<(ImageId, bool)>,
    /// The raw handles of the texture array elements bound so far, in element order: an
    /// element whose handle an earlier one has reuses its descriptor.
    element_handles: Vec<u32>,
    /// A storage image was scaled down while binding, so views collected before it are stale.
    rebind: bool,
    image_scaling: ScalingBits<{ NUM_IMAGE_SCALING_WORDS as usize }>,
    texel_views: Vec<vk::BufferView>,
    /// Texel buffer formats by `PixelFormat`: the host format and whether uniform and storage
    /// texel buffers support it, `None` when no host format views it as is.
    texel_formats: Vec<Option<Option<TexelFormat>>>,
    /// Views of the null buffer for unbound texel buffers, by host format.
    null_texel_views: Vec<(vk::Format, BufferView)>,
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
    assembly: Option<assembly::AssemblyPass>,
    grid: Option<dispatch_grid::GridPass>,
    frame_decode: Option<frame_decode::FrameDecodePass>,
    mesh_indirect: Option<mesh_indirect::MeshIndirectPass>,
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
                | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER
                | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::GpuOnly,
            "null buffer",
        )?;
        // SAFETY: `cmd` is recording; the buffer is a transfer destination.
        unsafe {
            device
                .raw
                .cmd_fill_buffer(cmd, null_buffer.raw(), 0, NULL_BUFFER_SIZE, 0);
        }
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
            mesh: ash::ext::mesh_shader::Device::new(&device.instance.raw, &device.raw),
            null_buffer,
            null_images: null::NullImages::default(),
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
            texture_scaling: ScalingBits::default(),
            storage_images: Vec::new(),
            element_handles: Vec::new(),
            rebind: false,
            image_scaling: ScalingBits::default(),
            texel_views: Vec::new(),
            texel_formats: vec![None; PixelFormat::ALL.len()],
            null_texel_views: Vec::new(),
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
            assembly: None,
            grid: None,
            frame_decode: None,
            mesh_indirect: None,
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

/// The GPU buffers an indirect draw reads.
#[derive(Clone, Copy)]
pub(super) struct IndirectBuffers {
    commands: BufferBinding,
    count: Option<BufferBinding>,
    /// The guest index buffer, for an assembled indexed draw.
    indices: Option<assembly::GuestIndices>,
    stride: u32,
    max_draws: u32,
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

/// Whether the draw runs task and mesh programs (SET_MESH_CONTROL, NVN2 only), whose
/// workgroup count is the draw's vertex count.
fn is_mesh_draw(regs: &Regs, arch: ShaderArch) -> bool {
    arch == ShaderArch::Ampere && regs.words()[MESH_CONTROL] & 1 != 0
}

/// The primitives a mesh program outputs (SET_MESH_SHADER_A bits 0..4).
fn mesh_topology(regs: &Regs) -> vk::PrimitiveTopology {
    match regs.words()[MESH_SHADER_A] & 0xf {
        0 => vk::PrimitiveTopology::POINT_LIST,
        1 => vk::PrimitiveTopology::LINE_LIST,
        _ => vk::PrimitiveTopology::TRIANGLE_LIST,
    }
}

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
                info.per_primitive_generics = previous
                    .mesh
                    .primitive_lines
                    .iter()
                    .filter_map(|&line| Attribute(line as u16).generic_index())
                    .fold(0, |mask, generic| mask | 1 << generic);
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
        let mesh = is_mesh_draw(regs, self.draw.arch);
        let host_topology = match mesh {
            true => mesh_topology(regs),
            false => mvk::primitive_topology(assembled_topology(call.topology()))
                .unwrap_or(vk::PrimitiveTopology::TRIANGLE_LIST),
        };
        if !self.graphics_stages(ctx, host_topology) {
            crate::rasterizer::frame_diag::with(|f| f.skips[0] += 1);
            return Ok(());
        }
        if !self.render_targets(ctx)? {
            crate::rasterizer::frame_diag::with(|f| f.skips[1] += 1);
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
            crate::rasterizer::frame_diag::with(|f| f.skips[2] += 1);
            return Ok(());
        };
        let pipeline_raw = pipeline.raw();
        let layout = pipeline.layout();
        let layout_raw = layout.raw();
        let push = layout.push_constants();
        let set_layout = layout.set_layouts().first().copied();
        let features = *draw.pipelines.features();

        let gmmu = ctx.memory;
        let mut removals = self.texture_cache.removals();
        let mut retargets = 0;
        let (index, indirect) = loop {
            let merged = self.buffer_cache.stats().merged_buffers;
            self.draw.rebind = false;
            self.draw.buffer_infos.clear();
            self.draw.image_infos.clear();
            self.draw.writes.clear();
            self.draw.textures.clear();
            self.draw.texture_scaling = ScalingBits::default();
            self.draw.image_scaling = ScalingBits::default();
            self.draw.storage_images.clear();
            self.draw.texel_views.clear();
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
            if mesh {
                self.draw.vertex_buffers.clear();
                self.draw.vertex_offsets.clear();
                self.draw.vertex_strides.clear();
            } else {
                self.vertex_buffers(gmmu, regs)?;
            }
            let index = match call {
                _ if mesh => None,
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
                DrawCall::Indirect(draw) => match self.indirect_buffers(gmmu, regs, draw, mesh)? {
                    Some(buffers) => Some(buffers),
                    None => return Ok(()),
                },
                DrawCall::Direct(_) => None,
            };
            let removed = self.texture_cache.removals() != removals;
            if self.buffer_cache.stats().merged_buffers == merged && !self.draw.rebind && !removed {
                break (index, indirect);
            }
            if removed {
                retargets += 1;
                if retargets > MAX_RETARGETS {
                    self.draw.warn_once(
                        Warning::ImageChurn,
                        "a draw's render targets and textures keep replacing each other in the texture cache; skipping it",
                    );
                    return Ok(());
                }
                if !self.render_targets(ctx)? {
                    return Ok(());
                }
                removals = self.texture_cache.removals();
            }
        };
        if index.is_some_and(|i: IndexBinding| i.index_count == 0) {
            return Ok(());
        }
        let set = self.descriptor_set(set_layout)?;

        let task = self.draw.stages[0]
            .as_ref()
            .is_some_and(|m| m.stage == ShaderStage::Task);
        let mesh_limit = self.device.properties.mesh_group_limit[usize::from(task)];
        let cmd = self.scheduler.record()?;
        let (index, indirect) = match (call, indirect) {
            (DrawCall::Indirect(draw), Some(inputs)) if mesh => {
                let commands = self.rewrite_mesh_indirect(cmd, draw, &inputs, mesh_limit)?;
                (None, Some(commands))
            }
            (DrawCall::Indirect(draw), Some(inputs))
                if assembled_topology(draw.topology) != draw.topology =>
            {
                let (index, commands) = self.assemble_indirect(cmd, regs, draw, &inputs)?;
                (Some(index), Some(commands))
            }
            (_, indirect) => (index, indirect),
        };
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
        for &(id, _) in &draw.storage_images {
            self.texture_cache.prepare(cmd, id, SHADER_STORAGE);
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
        let render_area = self
            .draw
            .stages
            .iter()
            .flatten()
            .any(|m| m.info.uses_render_area)
            .then(|| guest_render_area(regs));
        self.push_constants(cmd, layout_raw, push, render_area, scale);
        self.set_dynamic_state(cmd, regs, &features, host_topology, extent, scale);
        let mesh_groups = match call {
            DrawCall::Direct(params) if mesh => {
                if params.count > mesh_limit {
                    self.draw.warn_once(
                        Warning::MeshGroupLimit,
                        "a task/mesh draw has more groups than the device takes; clamping it",
                    );
                }
                params.count.min(mesh_limit)
            }
            _ => 0,
        };
        let raw = &self.device.raw;
        let draw = &self.draw;
        // SAFETY: as above; the pipeline's dynamic state is all set.
        unsafe {
            if let Some(index) = index {
                raw.cmd_bind_index_buffer(cmd, index.buffer, index.offset, index.index_type);
            }
            match (call, indirect, index) {
                (DrawCall::Direct(_), _, _) if mesh => {
                    draw.mesh.cmd_draw_mesh_tasks(cmd, mesh_groups, 1, 1)
                }
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
                (DrawCall::Indirect(_), Some(buffers), _) if mesh => match buffers.count {
                    Some(count) => draw.mesh.cmd_draw_mesh_tasks_indirect_count(
                        cmd,
                        buffers.commands.buffer,
                        buffers.commands.offset,
                        count.buffer,
                        count.offset,
                        buffers.max_draws,
                        buffers.stride,
                    ),
                    None => draw.mesh.cmd_draw_mesh_tasks_indirect(
                        cmd,
                        buffers.commands.buffer,
                        buffers.commands.offset,
                        buffers.max_draws,
                        buffers.stride,
                    ),
                },
                (DrawCall::Indirect(_), Some(buffers), index) => {
                    let IndirectBuffers {
                        commands,
                        count,
                        stride,
                        max_draws,
                        ..
                    } = buffers;
                    match (count, index.is_some()) {
                        (Some(count), true) => raw.cmd_draw_indexed_indirect_count(
                            cmd,
                            commands.buffer,
                            commands.offset,
                            count.buffer,
                            count.offset,
                            max_draws,
                            stride,
                        ),
                        (Some(count), false) => raw.cmd_draw_indirect_count(
                            cmd,
                            commands.buffer,
                            commands.offset,
                            count.buffer,
                            count.offset,
                            max_draws,
                            stride,
                        ),
                        (None, true) => raw.cmd_draw_indexed_indirect(
                            cmd,
                            commands.buffer,
                            commands.offset,
                            max_draws,
                            stride,
                        ),
                        (None, false) => raw.cmd_draw_indirect(
                            cmd,
                            commands.buffer,
                            commands.offset,
                            max_draws,
                            stride,
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
        for &(id, written) in &self.draw.storage_images {
            if written {
                self.texture_cache.mark_modified(id);
            }
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
        let mesh = is_mesh_draw(regs, self.draw.arch);
        if mesh && !self.device.features.mesh_shader {
            self.draw.warn_once(
                Warning::NoMeshShader,
                "a task/mesh draw on a device without VK_EXT_mesh_shader; skipping it",
            );
            return false;
        }
        let enabled = |slot: usize| match mesh {
            true => {
                slot == 1
                    || (slot == 3 && regs.pipelines[3].enable() != 0)
                    || (slot == 5 && regs.pipelines[5].enable() != 0)
            }
            false => slot == 1 || regs.pipelines[slot].enable() != 0,
        };
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
            let vertex_a = (slot == 1 && !mesh && regs.pipelines[0].enable() != 0)
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
            let info = info.with_layers(1);
            let daddr = ctx.memory.translate(rt.address()).unwrap_or(0);
            crate::rasterizer::frame_diag::with(|f| {
                f.rts
                    .entry((
                        daddr,
                        crate::rasterizer::frame_diag::key(&info, rt.width, rt.height),
                    ))
                    .or_default()[0] += 1
            });
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
            crate::rasterizer::frame_diag::with(|f| f.ids.insert(daddr, found.id));
            self.draw.colors[i] = Self::target_view(&mut self.texture_cache, found, &info)?;
        }
        if regs.zeta_enable != 0
            && let Some(info) = ImageInfo::from_zeta(&regs.zeta, &regs.zeta_size, msaa, gob)
                .map(|info| info.with_layers(1))
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
                    DescriptorKind::UniformTexelBuffer | DescriptorKind::StorageTexelBuffer => {
                        self.draw.texel_views.len()
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
                        let mut array_rescaled = None;
                        let start = self.draw.image_infos.len();
                        self.draw.element_handles.clear();
                        for i in 0..b.count {
                            let raw = match desc.handle_program {
                                Some(program) => self.program_handle(src, info, program, i)?,
                                None => handle_word(src, desc.handle, i << desc.size_shift),
                            };
                            let earlier = self.draw.element_handles.iter().position(|&h| h == raw);
                            self.draw.element_handles.push(raw);
                            if let Some(earlier) = earlier {
                                let info = self.draw.image_infos[start + earlier];
                                self.draw.image_infos.push(info);
                                continue;
                            }
                            let readable = desc.handle_program.is_none_or(|_| {
                                let header = src.pool.header(src.gmmu, raw);
                                texture_type(&header) == desc.ty
                                    && header.format.is_integer() == desc.is_integer
                            });
                            let texture = if desc.handle_is_pointer_word || !readable {
                                None
                            } else {
                                self.texture_view(src, raw, desc.ty)?
                            };
                            let scale = texture
                                .and_then(|(id, _)| self.texture_cache.state(id))
                                .map_or(ResolutionScale::X1, |s| s.scale);
                            let rescaled = scale != ResolutionScale::X1;
                            if texture.is_some()
                                && *array_rescaled.get_or_insert(rescaled) != rescaled
                            {
                                self.draw.warn_once(
                                    Warning::MixedScaleArray,
                                    "a draw samples a texture array mixing rescaled and native images; skipping it",
                                );
                                return Ok(false);
                            }
                            if rescaled {
                                self.draw.texture_scaling.set(b.descriptor);
                            }
                            let (view, layout) = match texture {
                                Some((id, view)) => {
                                    self.draw.textures.push(id);
                                    (view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                                }
                                None => {
                                    let format = match (desc.is_depth, desc.is_integer) {
                                        (true, _) => vk::Format::D16_UNORM,
                                        (false, true) => vk::Format::R8G8B8A8_UINT,
                                        (false, false) => vk::Format::R8G8B8A8_UNORM,
                                    };
                                    let null = match desc.is_multisample {
                                        _ if self.device.features.null_descriptor => {
                                            Some(vk::ImageView::null())
                                        }
                                        true => None,
                                        false => self.null_view(view_type(desc.ty), format)?,
                                    };
                                    let Some(view) = null else {
                                        self.draw.warn_once(
                                            Warning::NullTexture,
                                            "a draw samples an unbound multisampled texture; skipping it",
                                        );
                                        return Ok(false);
                                    };
                                    (view, null::NULL_LAYOUT)
                                }
                            };
                            let (_, sampler_index) = src.pool.split(raw);
                            let sampler = self.sampler(src, sampler_index)?;
                            self.draw.image_infos.push(vk::DescriptorImageInfo {
                                sampler,
                                image_view: view,
                                image_layout: layout,
                            });
                        }
                    }
                    DescriptorKind::UniformTexelBuffer => {
                        let desc = info.texture_buffer_descriptors[descriptor];
                        for i in 0..b.count {
                            let raw = match desc.handle_program {
                                Some(program) => self.program_handle(src, info, program, i)?,
                                None => handle_word(src, desc.handle, i << desc.size_shift),
                            };
                            let view = self.texel_view(src, raw, None, false, false)?;
                            self.push_texel_view(view, vk::Format::R8G8B8A8_UNORM)?;
                        }
                    }
                    DescriptorKind::StorageTexelBuffer => {
                        let desc = info.image_buffer_descriptors[descriptor];
                        let format = storage_format(desc.format);
                        for i in 0..b.count {
                            let raw = match desc.handle_program {
                                Some(program) => self.program_handle(src, info, program, i)?,
                                None => read_cbuf(
                                    src.gmmu,
                                    src.cbufs,
                                    desc.cbuf_index,
                                    desc.cbuf_offset + (i << desc.size_shift),
                                ),
                            };
                            let view = if numeric_mismatch(
                                src,
                                desc.handle_program,
                                desc.format,
                                desc.is_integer,
                                raw,
                            ) {
                                None
                            } else {
                                self.texel_view(src, raw, format, true, desc.is_written)?
                            };
                            let null = match format.and_then(|f| self.texel_format(f)) {
                                Some(texel) => texel.format,
                                None if desc.is_integer || format.is_some() => vk::Format::R32_UINT,
                                None => vk::Format::R32_SFLOAT,
                            };
                            self.push_texel_view(view, null)?;
                        }
                    }
                    DescriptorKind::StorageImage => {
                        let desc = info.image_descriptors[descriptor];
                        let mut array_rescaled = None;
                        for i in 0..b.count {
                            let raw = match desc.handle_program {
                                Some(program) => self.program_handle(src, info, program, i)?,
                                None => read_cbuf(
                                    src.gmmu,
                                    src.cbufs,
                                    desc.cbuf_index,
                                    desc.cbuf_offset + (i << desc.size_shift),
                                ),
                            };
                            let found = if numeric_mismatch(
                                src,
                                desc.handle_program,
                                desc.format,
                                desc.is_integer,
                                raw,
                            ) {
                                None
                            } else {
                                self.storage_view(src, raw, &desc)?
                            };
                            let Some((id, view)) = found else {
                                if !self.device.features.null_descriptor {
                                    self.draw.warn_once(
                                        Warning::NullStorageImage,
                                        "a draw binds an unbound storage image without null descriptors; skipping it",
                                    );
                                    return Ok(false);
                                }
                                self.draw.image_infos.push(vk::DescriptorImageInfo {
                                    sampler: vk::Sampler::null(),
                                    image_view: vk::ImageView::null(),
                                    image_layout: vk::ImageLayout::GENERAL,
                                });
                                continue;
                            };
                            let rescaled = self
                                .texture_cache
                                .state(id)
                                .is_some_and(|s| s.scale != ResolutionScale::X1);
                            if *array_rescaled.get_or_insert(rescaled) != rescaled {
                                self.draw.warn_once(
                                    Warning::MixedScaleArray,
                                    "a draw binds a storage image array mixing rescaled and native images; skipping it",
                                );
                                return Ok(false);
                            }
                            self.draw.storage_images.push((id, desc.is_written));
                            self.draw.image_infos.push(vk::DescriptorImageInfo {
                                sampler: vk::Sampler::null(),
                                image_view: view,
                                image_layout: vk::ImageLayout::GENERAL,
                            });
                        }
                        if array_rescaled == Some(true) {
                            self.draw.image_scaling.set(b.descriptor);
                        }
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
        self.draw
            .texture_scaling
            .next_stage(info.texture_descriptors.len());
        self.draw
            .image_scaling
            .next_stage(info.image_descriptors.len());
        let draw = &mut self.draw;
        if draw
            .storage_images
            .iter()
            .any(|(id, _)| draw.textures.contains(id))
        {
            draw.warn_once(
                Warning::StorageAndSampled,
                "a draw reads an image as a texture and a storage image at once; skipping it",
            );
            return Ok(false);
        }
        Ok(true)
    }

    /// The host format texel buffers view `format` with, looked up once per format.
    fn texel_format(&mut self, format: PixelFormat) -> Option<TexelFormat> {
        if let Some(known) = self.draw.texel_formats[format as usize] {
            return known;
        }
        let host = self.texture_cache.host(format);
        let identity = [
            host.swizzle.r,
            host.swizzle.g,
            host.swizzle.b,
            host.swizzle.a,
        ]
        .iter()
        .all(|&c| c == vk::ComponentSwizzle::IDENTITY);
        let found = (host.conversion == crate::format::Conversion::None && identity).then(|| {
            let features = self.device.buffer_features(host.format);
            TexelFormat {
                format: host.format,
                texel: u64::from(format.bytes_per_block()),
                uniform: features.contains(vk::FormatFeatureFlags::UNIFORM_TEXEL_BUFFER),
                storage: features.contains(vk::FormatFeatureFlags::STORAGE_TEXEL_BUFFER),
            }
        });
        self.draw.texel_formats[format as usize] = Some(found);
        found
    }

    /// The texel buffer view of the buffer texture `raw` selects, read as `explicit` when the
    /// shader names a format and as the header's otherwise (Lichii's texture buffer binding).
    /// `None` when no buffer texture or device memory backs it.
    fn texel_view(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        explicit: Option<PixelFormat>,
        storage: bool,
        written: bool,
    ) -> Result<Option<vk::BufferView>> {
        let (index, _) = src.pool.split(raw);
        if index > src.pool.limit {
            return Ok(None);
        }
        let header = src.pool.header(src.gmmu, raw);
        if header.address == 0 || header.memory != TextureMemory::Buffer {
            return Ok(None);
        }
        let format = explicit.unwrap_or(header.format);
        let Some(texel) = self
            .texel_format(format)
            .filter(|f| if storage { f.storage } else { f.uniform })
        else {
            self.draw.warn_once(
                Warning::TexelBufferFormat,
                "a texel buffer format the device cannot view; binding the null buffer",
            );
            return Ok(None);
        };
        self.buffer_cache.texel_buffer(
            &mut self.scheduler,
            &mut self.staging,
            src.gmmu,
            header.address,
            u64::from(header.width) * texel.texel,
            (texel.format, texel.texel),
            written,
        )
    }

    /// Appends `view`, or a view of the null buffer as `null` without one.
    fn push_texel_view(&mut self, view: Option<vk::BufferView>, null: vk::Format) -> Result<()> {
        let view = match view {
            Some(view) => view,
            None => match self.draw.null_texel_views.iter().find(|(f, _)| *f == null) {
                Some((_, view)) => view.raw(),
                None => {
                    let view = BufferView::new(
                        &self.device,
                        self.draw.null_buffer.raw(),
                        null,
                        0,
                        NULL_BUFFER_SIZE,
                    )?;
                    let raw = view.raw();
                    self.draw.null_texel_views.push((null, view));
                    raw
                }
            },
        };
        self.draw.texel_views.push(view);
        Ok(())
    }

    /// The image and storage view of the storage image `raw` selects for `desc`: in the
    /// shader's format when it names one, the header's otherwise. A rescaled image the shader
    /// writes is scaled down first, as Lichii's `FillImageViews` does for blacklisted views.
    fn storage_view(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        desc: &ImageDescriptor,
    ) -> Result<Option<(ImageId, vk::ImageView)>> {
        let Some((id, mut view)) = self.texture_info(src, raw, desc.ty)? else {
            return Ok(None);
        };
        if desc.is_written
            && self
                .texture_cache
                .state(id)
                .is_some_and(|s| s.scale != ResolutionScale::X1)
        {
            let mut tx = Transfer {
                scheduler: &mut self.scheduler,
                staging: &mut self.staging,
                memory: &self.handles.device,
            };
            self.texture_cache.scale_down(&mut tx, id)?;
            self.draw.rebind = true;
        }
        if let Some(format) = storage_format(desc.format) {
            view.format = format;
        }
        Ok(self
            .texture_cache
            .storage_view(id, &view)?
            .map(|raw| (id, raw)))
    }

    /// A view of `view_type` and `format` reading zeros, for an unbound texture on a device
    /// without null descriptors; `None` when the device cannot sample the format.
    fn null_view(
        &mut self,
        view_type: vk::ImageViewType,
        format: vk::Format,
    ) -> Result<Option<vk::ImageView>> {
        let scheduler = &mut self.scheduler;
        self.draw
            .null_images
            .view(&self.device, || scheduler.record(), view_type, format)
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
        let Some((id, view)) = self.texture_info(src, raw, ty)? else {
            return Ok(None);
        };
        Ok(self
            .texture_cache
            .sampled_view(id, &view)?
            .map(|raw| (id, raw)))
    }

    /// The image the texture header `raw` selects and the view of it `ty` reads; `None` when
    /// nothing backs it or the image's layers cannot form the view.
    fn texture_info(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        ty: TextureType,
    ) -> Result<Option<(ImageId, ViewInfo)>> {
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
        Ok(Some((id, view)))
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
    /// Assembled draws read the guest indices in their assembly pass instead.
    fn indirect_index_buffer(
        &mut self,
        gmmu: &lichii_gpu::gmmu::Gmmu,
        regs: &Regs,
        draw: &IndirectDraw,
    ) -> Result<Option<IndexBinding>> {
        if !draw.indexed || assembled_topology(draw.topology) != draw.topology {
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

    /// The buffers an indirect draw (a task/mesh one with `mesh`) reads. `None` skips the draw.
    fn indirect_buffers(
        &mut self,
        gmmu: &lichii_gpu::gmmu::Gmmu,
        regs: &Regs,
        draw: &IndirectDraw,
        mesh: bool,
    ) -> Result<Option<IndirectBuffers>> {
        let assembled = !mesh && assembled_topology(draw.topology) != draw.topology;
        if !assembled
            && !mesh
            && draw.count_buffer.is_some()
            && !self.device.features.draw_indirect_count
        {
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
        let indices = if assembled && draw.indexed {
            let Some(indices) = self.guest_indices(gmmu, regs)? else {
                return Ok(None);
            };
            Some(indices)
        } else {
            None
        };
        Ok(Some(IndirectBuffers {
            commands,
            count,
            indices,
            stride: draw.stride,
            max_draws: draw.max_draws,
        }))
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
                DescriptorKind::UniformTexelBuffer | DescriptorKind::StorageTexelBuffer => {
                    vk_write.p_texel_buffer_view = draw.texel_views[write.first..].as_ptr();
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
    /// texture rescaling words of the bound textures at 0 (storage images are never bound, so
    /// their words stay 0) with the down factor of `scale`, the targets' scale, at 24,
    /// and the render area at `RENDER_AREA_LAYOUT_OFFSET` (32), so a 48-byte range holds both.
    fn push_constants(
        &mut self,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        (stages, size): (vk::ShaderStageFlags, u32),
        render_area: Option<(f32, f32)>,
        scale: ResolutionScale,
    ) {
        if size == 0 {
            return;
        }
        let mut bytes = [0u8; RENDER_AREA_LAYOUT_OFFSET as usize + 16];
        let words: &[u8] = bytemuck::cast_slice(&self.draw.texture_scaling.words);
        let at = RESCALING_LAYOUT_TEXTURES_OFFSET as usize;
        bytes[at..at + words.len()].copy_from_slice(words);
        let words: &[u8] = bytemuck::cast_slice(&self.draw.image_scaling.words);
        let at = RESCALING_LAYOUT_IMAGES_OFFSET as usize;
        bytes[at..at + words.len()].copy_from_slice(words);
        let (num, den) = scale.ratio();
        let down = den as f32 / num as f32;
        let at = RESCALING_LAYOUT_DOWN_FACTOR_OFFSET as usize;
        bytes[at..at + 4].copy_from_slice(&down.to_le_bytes());
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
        let mesh = is_mesh_draw(regs, self.draw.arch);
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
                if !mesh {
                    raw.cmd_set_primitive_topology(cmd, topology);
                }
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
                if !mesh {
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
            if features.dynamic_vertex_input && !mesh {
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
        let mut removals = self.texture_cache.removals();
        let mut retargets = 0;
        let grid = loop {
            let merged = self.buffer_cache.stats().merged_buffers;
            self.draw.rebind = false;
            self.draw.buffer_infos.clear();
            self.draw.image_infos.clear();
            self.draw.writes.clear();
            self.draw.textures.clear();
            self.draw.texture_scaling = ScalingBits::default();
            self.draw.image_scaling = ScalingBits::default();
            self.draw.storage_images.clear();
            self.draw.texel_views.clear();
            if !self.bind_resources(&module, &sources)? {
                return Ok(());
            }
            let grid = match ctx.indirect {
                Some(grid) => match self.buffer_cache.storage_buffer(
                    &mut self.scheduler,
                    &mut self.staging,
                    ctx.memory,
                    grid.address,
                    grid.size(),
                    false,
                )? {
                    Some(binding) => Some((binding, grid)),
                    None => return Ok(()),
                },
                None => None,
            };
            let removed = self.texture_cache.removals() != removals;
            if self.buffer_cache.stats().merged_buffers == merged && !self.draw.rebind && !removed {
                break grid;
            }
            if removed {
                retargets += 1;
                if retargets > MAX_RETARGETS {
                    self.draw.warn_once(
                        Warning::ImageChurn,
                        "a dispatch's images keep replacing each other in the texture cache; skipping it",
                    );
                    return Ok(());
                }
                removals = self.texture_cache.removals();
            }
        };
        let set = self.descriptor_set(set_layout)?;
        let cmd = self.scheduler.record()?;
        let align = self.buffer_cache.storage_alignment();
        let grid = match grid {
            Some((binding, grid)) => Some(self.unpack_grid(
                cmd,
                binding,
                (grid.address & (align - 1)) as u32,
                grid.layout,
            )?),
            None => None,
        };
        for &id in &self.draw.textures {
            self.texture_cache.prepare(cmd, id, SHADER_SAMPLED);
        }
        for &(id, _) in &self.draw.storage_images {
            self.texture_cache.prepare(cmd, id, SHADER_STORAGE);
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
        self.push_constants(cmd, layout_raw, push, None, ResolutionScale::X1);
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let raw = &self.device.raw;
        // SAFETY: as above.
        unsafe {
            match grid {
                Some(command) => raw.cmd_dispatch_indirect(cmd, command, 0),
                None => raw.cmd_dispatch(cmd, x, y, z),
            }
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }
        for &(id, written) in &self.draw.storage_images {
            if written {
                self.texture_cache.mark_modified(id);
            }
        }
        self.unflushed = true;
        Ok(())
    }
}

/// How texel buffers view a guest format.
#[derive(Clone, Copy)]
struct TexelFormat {
    format: vk::Format,
    texel: u64,
    uniform: bool,
    storage: bool,
}

/// The guest format of a typed storage image or image buffer access; `None` for typeless
/// accesses, which take the header's.
fn storage_format(format: ImageFormat) -> Option<PixelFormat> {
    Some(match format {
        ImageFormat::Typeless => return None,
        ImageFormat::R8Uint => PixelFormat::R8Uint,
        ImageFormat::R8Sint => PixelFormat::R8Sint,
        ImageFormat::R16Uint => PixelFormat::R16Uint,
        ImageFormat::R16Sint => PixelFormat::R16Sint,
        ImageFormat::R32Uint => PixelFormat::R32Uint,
        ImageFormat::R32G32Uint => PixelFormat::R32G32Uint,
        ImageFormat::R32G32B32A32Uint => PixelFormat::R32G32B32A32Uint,
    })
}

/// The handle word `handle` names, `offset` bytes into its array.
fn handle_word(src: &StageSources<'_>, handle: HandleSource, offset: u32) -> u32 {
    let mut raw = read_cbuf::<u32>(
        src.gmmu,
        src.cbufs,
        handle.cbuf_index,
        handle.cbuf_offset + offset,
    ) << handle.shift_left;
    if let Some((index, offset, shift)) = handle.secondary {
        raw |= read_cbuf::<u32>(src.gmmu, src.cbufs, index, offset) << shift;
    }
    raw
}

impl VulkanRasterizer {
    /// The raw handle `info`'s handle program `program` computes for descriptor array
    /// `element`. Its storage reads are relative to the binding address, the buffer's pointer
    /// aligned down to the storage alignment; words past the buffer, and of an unbound one,
    /// read as 0. A word GPU work wrote is downloaded first, so no handle reads stale memory;
    /// the download waits for the GPU, which frees views retired since, so the bind loop reruns.
    fn program_handle(
        &mut self,
        src: &StageSources<'_>,
        info: &ShaderInfo,
        program: u32,
        element: u32,
    ) -> Result<u32> {
        let align = self.buffer_cache.storage_alignment();
        loop {
            let cache = &self.buffer_cache;
            let mut stale = None;
            let read_storage = |slot: u32, offset: u32| {
                let mut slot = slot;
                let Some((desc, element)) = info.storage_buffers_descriptors.iter().find_map(|d| {
                    let found = (slot < d.count).then_some((d, slot));
                    slot = slot.wrapping_sub(d.count);
                    found
                }) else {
                    return 0;
                };
                let (address, size) = storage_pointer(src, desc, element);
                let base = address & !(align - 1);
                let word = base + u64::from(offset);
                if address == 0 || word + 4 > address + size {
                    return 0;
                }
                match src.gmmu.translate(word) {
                    Some(daddr) if cache.is_gpu_written(daddr, 4) => {
                        stale = Some(daddr);
                        0
                    }
                    _ => src.gmmu.read::<u32>(word).unwrap_or(0),
                }
            };
            let raw = info.handle_programs[program as usize].evaluate(
                |index, offset| read_cbuf(src.gmmu, src.cbufs, index, offset),
                read_storage,
                element,
            );
            let Some(daddr) = stale else {
                return Ok(raw);
            };
            self.buffer_cache.flush(&mut self.scheduler, daddr, 4)?;
            self.draw.rebind = true;
        }
    }
}

/// Whether the image `raw` selects has another numeric type than the typeless access a handle
/// program feeds was compiled for: with no header to read, the shader declared `is_integer`.
fn numeric_mismatch(
    src: &StageSources<'_>,
    program: Option<u32>,
    format: ImageFormat,
    is_integer: bool,
    raw: u32,
) -> bool {
    program.is_some()
        && format == ImageFormat::Typeless
        && src.pool.header(src.gmmu, raw).format.is_integer() != is_integer
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

mod assembly;
mod dispatch_grid;
mod frame_decode;
mod host_compute;
mod mesh_indirect;
mod null;
mod texture;

#[cfg(test)]
mod tests;
