//! Draws and compute dispatches: translated programs from the shader cache, pipelines from the
//! pipeline cache, render targets and textures from the texture cache and buffers from the
//! buffer cache, recorded with dynamic rendering.
//!
//! Every per-draw list lives in [`DrawState`] and is cleared, not reallocated, so a draw whose
//! shaders, pipeline, images and samplers are cached allocates nothing.

use std::collections::HashMap;
use std::mem::offset_of;
use std::sync::{Arc, Weak};

use ash::vk;
use lichii_common::GpuVAddr;
use lichii_common::settings::{Anisotropy, ResolutionScale};
use lichii_gpu::backend::{
    ChannelId, ComputeContext, ConstBufferBinding, DrawParams, GraphicsContext, IndirectDraw,
};
use lichii_gpu::engines::maxwell_3d::regs::{
    AttributePointSize, Blend, BlendPerTarget, ComparisonOp, DepthMode, IndexFormat, Pipeline,
    PrimitiveTopology, ProgramRegion, RtLayerControl, SamplerBinding, TexHeader, VertexAttribute,
    VertexAttributeType, WindowOrigin, WindowOriginMode,
};
use lichii_gpu::engines::maxwell_3d::regs::{MESH_CONTROL, MESH_SHADER_A};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs, dirty};
use lichii_shader::backend::Bindings;
use lichii_shader::backend::spirv::{
    NUM_IMAGE_SCALING_WORDS, NUM_TEXTURE_SCALING_WORDS, RENDER_AREA_LAYOUT_OFFSET,
    RESCALING_LAYOUT_DOWN_FACTOR_OFFSET, RESCALING_LAYOUT_IMAGES_OFFSET,
    RESCALING_LAYOUT_TEXTURES_OFFSET,
};
use lichii_shader::info::{
    HandleSource, ImageDescriptor, StorageBufferDescriptor, TextureDescriptor, VaryingState,
};
use lichii_shader::ir::{Attribute, ImageFormat, TextureType};
use lichii_shader::maxwell::header::ProgramHeader;
use lichii_shader::runtime::{AttributeType, CompareFunction, InputTopology};
use lichii_shader::{
    DescriptorKind, RuntimeInfo, ShaderArch, ShaderInfo, ShaderModule, ShaderStage,
};
use lichii_texture::block_linear::GobLayout;
use lichii_texture::guest::{DepthFormat, RenderTargetFormat};
use lichii_texture::tic::TextureMemory;
use lichii_texture::{Numeric, PixelFormat};

use super::Release;
use super::VulkanRasterizer;
use super::samples::SampleCounter;
use crate::buffer_cache::{BufferBinding, IndexBinding, assembled_topology, vertex_stream_range};
use crate::commands::ImageAccess;
use crate::debug::{Program, Programs};
use crate::descriptors::{DescriptorAllocator, DescriptorInfo};
use crate::device::{Device, DeviceFeatures};
use crate::error::Result;
use crate::memory::{Buffer, BufferView};
use crate::pipeline::cache::SetTemplate;
use crate::pipeline::environment::{
    ProgramEnv, TexturePool, graphics_program_header, texture_type,
};
use crate::pipeline::fixed_state::flags;
use crate::pipeline::maxwell_to_vk as mvk;
use crate::pipeline::sampler::{SamplerCaps, SamplerParams, Tsc};
use crate::pipeline::shader_cache::{ModuleKey, ShaderCache};
use crate::pipeline::{
    AttachmentState, ComputePipelineKey, FixedStateTracker, FxBuildHasher, GraphicsPipelineKey,
    GraphicsShaders, PipelineCache, PipelineFeatures,
};
use crate::scheduler::RenderingKey;
use crate::texture_cache::{
    DEPTH_ATTACHMENT, ImageId, ImageInfo, ImageKind, Transfer, Usage, ViewInfo,
};
use compute_batch::ComputeBatch;
use dynamic::{Command, DynamicInputs, DynamicShadow};

const NULL_BUFFER_SIZE: u64 = 64 << 10;
/// The constant buffer slot NVN2 graphics programs read root table 0 from (their slot 24).
const ROOT_TABLE_SLOT: usize = 17;
/// How often a draw or dispatch looks its images up again after a lookup removed one it held,
/// before it gives up: a render target and a texture over the same memory in layouts the
/// texture cache cannot share replace each other on every lookup.
const MAX_RETARGETS: u32 = 2;

const SHADER_SAMPLED: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::SHADER_SAMPLED_READ,
};

/// A colour target the same draw samples, in the one layout both uses allow.
const FEEDBACK_COLOR: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::from_raw(
        vk::AccessFlags2::COLOR_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw(),
    ),
};

/// A depth target the same draw samples, in the one layout both uses allow.
const FEEDBACK_DEPTH: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::GENERAL,
    stage: vk::PipelineStageFlags2::ALL_COMMANDS,
    access: vk::AccessFlags2::from_raw(
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw()
            | vk::AccessFlags2::SHADER_SAMPLED_READ.as_raw(),
    ),
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
    SampleQueries = 1 << 17,
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

/// The state a draw's render targets resolve from that the `RENDER_TARGETS` dirty flag does
/// not cover: the channel whose flags those are, the sample mode, layered rendering, the depth
/// target's layer count, and the GPU mappings and texture cache images the lookups went through.
#[derive(Clone, Copy, PartialEq, Eq)]
struct TargetsKey {
    channel: ChannelId,
    msaa: u32,
    layered: bool,
    zeta_size: u32,
    memory: u64,
    cache: u64,
}

/// `(first word, words)` of each register `graphics_stages` reads, except the render targets
/// (compared as colour outputs). The program registers are compared by value, so a draw after a
/// guest rewrote them unchanged still reuses the last draw's modules.
const STAGE_REGISTERS: [(usize, usize); 18] = {
    const fn field<T>(offset: usize) -> (usize, usize) {
        (offset / 4, size_of::<T>() / 4)
    }
    [
        field::<ProgramRegion>(offset_of!(Regs, program_region)),
        field::<[Pipeline; 6]>(offset_of!(Regs, pipelines)),
        field::<WindowOrigin>(offset_of!(Regs, window_origin)),
        field::<DepthMode>(offset_of!(Regs, depth_mode)),
        field::<AttributePointSize>(offset_of!(Regs, point_size_attribute)),
        field::<f32>(offset_of!(Regs, point_size)),
        field::<u32>(offset_of!(Regs, alpha_test_enabled)),
        field::<f32>(offset_of!(Regs, alpha_test_ref)),
        field::<ComparisonOp>(offset_of!(Regs, alpha_test_func)),
        field::<Blend>(offset_of!(Regs, blend)),
        field::<BlendPerTarget>(offset_of!(Regs, blend_per_target)),
        field::<u32>(offset_of!(Regs, blend_per_target_enabled)),
        field::<[VertexAttribute; 0x20]>(offset_of!(Regs, vertex_attrib_format)),
        field::<TexHeader>(offset_of!(Regs, tex_header)),
        field::<SamplerBinding>(offset_of!(Regs, sampler_binding)),
        field::<u32>(offset_of!(Regs, viewport_scale_offset_enabled)),
        field::<u32>(offset_of!(Regs, bindless_texture_const_buffer_slot)),
        (MESH_CONTROL, 4),
    ]
};

const STAGE_REGISTER_WORDS: usize = {
    let mut words = 0;
    let mut i = 0;
    while i < STAGE_REGISTERS.len() {
        words += STAGE_REGISTERS[i].1;
        i += 1;
    }
    words
};

/// Everything `graphics_stages` finds its modules from: equal inputs find the same modules, as
/// long as each one's variant still answers its translation's queries.
#[derive(Clone, Copy, PartialEq, Eq)]
struct StageInputs {
    channel: ChannelId,
    topology: vk::PrimitiveTopology,
    /// GPU mappings and program bytes the module keys were hashed through.
    memory: u64,
    programs: u64,
    color_outputs: [AttributeType; 8],
    registers: [u32; STAGE_REGISTER_WORDS],
    /// The header of each program slot in use, zero for the others.
    headers: [[u32; 20]; 6],
}

/// The stages of the last draw and what they were found from.
struct StagesMemo {
    inputs: StageInputs,
    keys: [Option<ModuleKey>; 5],
}

/// A texture header a draw sampled, with what its descriptor asked of it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct TextureKey {
    header: [u32; 8],
    ty: TextureType,
    /// The descriptor's handle came from a handle program, so the header's type and numeric
    /// kind were checked against `ty` and `integer`.
    checked: bool,
    integer: bool,
}

/// What a texture header bound: the image, its sampled view and whether it is rescaled.
#[derive(Clone, Copy)]
struct BoundTexture {
    memory: u64,
    id: ImageId,
    view: vk::ImageView,
    rescaled: bool,
}

/// Sampled textures by header while the texture cache's host images and lookups and the address
/// space stay the same.
#[derive(Default)]
struct TextureMemo {
    /// The [`crate::texture_cache::TextureCache::texture_generation`] the entries were found
    /// under.
    cache: u64,
    /// The address space the headers were read from. Holding it keeps its allocation, so no
    /// later address space can take its place.
    gmmu: Weak<lichii_gpu::gmmu::Gmmu>,
    bound: HashMap<TextureKey, BoundTexture, FxBuildHasher>,
}

/// Textures [`TextureMemo`] keeps before it starts over.
const MAX_BOUND_TEXTURES: usize = 4096;

/// The host sampler last found for a sampler index, with the header words and anisotropy
/// setting it was created for.
#[derive(Clone, Copy, Default)]
struct MemoSampler {
    words: [u32; 8],
    anisotropy: u8,
    sampler: vk::Sampler,
}

/// Slots of [`DrawState::sampler_memo`], by sampler index modulo the count: 48 bytes each,
/// 12 KiB in all.
const SAMPLER_MEMO: usize = 256;

/// The bytes [`VulkanRasterizer::push_constants`] fills: the rescaling words and the render
/// area after them.
const PUSH_BYTES: usize = RENDER_AREA_LAYOUT_OFFSET as usize + 16;

/// A draw's push constants: the layout and its one range's stages and size, and the bytes.
#[derive(Clone, Copy, PartialEq, Eq)]
struct PushConstants {
    layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    size: u32,
    bytes: [u8; PUSH_BYTES],
}

/// What the draws of the open rendering bound, so a draw records only the binds that changed.
/// Everything else that binds graphics state (clears, texture draws, presents, dispatches) and
/// a new command buffer records outside a rendering, so a draw that begins one forgets it all.
#[derive(Default)]
struct Bound {
    pipeline: vk::Pipeline,
    push: Option<PushConstants>,
    index: Option<(vk::Buffer, u64, vk::IndexType)>,
}

impl PushConstants {
    fn record(&self, device: &Device, cmd: vk::CommandBuffer) {
        let size = (self.size as usize).min(PUSH_BYTES);
        // SAFETY: `stages` and `size` are the layout's one push constant range at offset 0.
        unsafe {
            device
                .raw
                .cmd_push_constants(cmd, self.layout, self.stages, 0, &self.bytes[..size]);
        }
    }
}

impl Bound {
    fn begin(&mut self, began: bool) {
        if began {
            *self = Self::default();
        }
    }
}

/// Holds `value` in `bound`; `true` when it differs from what `bound` held, so it must be
/// recorded.
fn rebinds<T: Copy + PartialEq>(bound: &mut T, value: T) -> bool {
    std::mem::replace(bound, value) != value
}

/// One bound render target of the current draw.
#[derive(Clone, Copy)]
struct Target {
    id: ImageId,
    view: vk::ImageView,
    format: vk::Format,
    extent: vk::Extent2D,
    layers: u32,
}

/// Draw and dispatch state of the renderer, beside the caches it shares with clears and copies.
pub(super) struct DrawState {
    pub(super) pipelines: PipelineCache,
    pub(super) shaders: ShaderCache,
    fixed: FixedStateTracker,
    descriptors: DescriptorAllocator,
    /// By sampler header words and the anisotropy setting they were created with.
    samplers: HashMap<([u32; 8], u8), vk::Sampler, FxBuildHasher>,
    /// The sampler last found for each sampler index, checked against the header words read
    /// for every lookup, so `samplers` is probed only when they changed.
    sampler_memo: Box<[MemoSampler; SAMPLER_MEMO]>,
    sampler_caps: SamplerCaps,
    anisotropy: Anisotropy,
    eds3: ash::ext::extended_dynamic_state3::Device,
    vertex_input: ash::ext::vertex_input_dynamic_state::Device,
    mesh: ash::ext::mesh_shader::Device,
    depth_bias_control: ash::ext::depth_bias_control::Device,
    push_descriptor: ash::khr::push_descriptor::Device,
    null_buffer: Buffer,
    null_images: null::NullImages,
    arch: ShaderArch,
    channel: Option<ChannelId>,
    warned: u32,
    stages: GraphicsShaders,
    colors: [Option<Target>; 8],
    depth: Option<Target>,
    /// What `colors` and `depth` were resolved from beyond the registers the
    /// `RENDER_TARGETS` dirty flag covers; `None` until the next resolve.
    targets_key: Option<TargetsKey>,
    /// What `stages` were found from; `None` until the next lookup.
    stages_memo: Option<StagesMemo>,
    /// The images the last [`PRESENTED`] presents showed. Only a draw into them alone may be
    /// skipped while its program translates or its pipeline builds: one frame shows it missing.
    /// A draw into any other target waits, so no later pass reads a target a skip left stale.
    presented: [Option<ImageId>; PRESENTED],
    textures: Vec<ImageId>,
    /// What texture headers bound, reused while their images need no refresh.
    bound_textures: TextureMemo,
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
    /// The storage buffers of `buffer_infos` the shader writes.
    written_buffers: Vec<vk::DescriptorBufferInfo>,
    /// The dispatches recorded since the last compute barrier.
    compute_batch: ComputeBatch,
    image_infos: Vec<vk::DescriptorImageInfo>,
    writes: Vec<Write>,
    /// The data of the current draw's [`SetTemplate`].
    descriptor_infos: Vec<DescriptorInfo>,
    vertex_buffers: Vec<vk::Buffer>,
    vertex_offsets: Vec<u64>,
    vertex_strides: Vec<u64>,
    attachments: Vec<vk::RenderingAttachmentInfo<'static>>,
    /// What the open rendering's draws last bound.
    bound: Bound,
    dynamic: DynamicShadow,
    texture_pipelines: Option<texture::TexturePipelines>,
    assembly: Option<assembly::AssemblyPass>,
    grid: Option<dispatch_grid::GridPass>,
    frame_decode: Option<frame_decode::FrameDecodePass>,
    mesh_indirect: Option<mesh_indirect::MeshIndirectPass>,
}

/// Presents whose images count as the presented frame for [`DrawState::presented_only`].
const PRESENTED: usize = 3;

impl DrawState {
    /// Notes that a present showed `image`.
    pub(super) fn presented(&mut self, image: ImageId) {
        if !self.presented.contains(&Some(image)) {
            self.presented.rotate_right(1);
            self.presented[0] = Some(image);
        }
    }

    /// Whether the draw's bound colour targets are all images a recent present showed, so
    /// skipping it only leaves a gap in one shown frame.
    fn presented_only(&self) -> bool {
        let mut colors = self.colors.iter().flatten().peekable();
        colors.peek().is_some() && colors.all(|t| self.presented.contains(&Some(t.id)))
    }

    /// The layers a rendering into the draw's targets spans: the fewest any target has.
    fn layers(&self) -> u32 {
        self.colors
            .iter()
            .flatten()
            .chain(&self.depth)
            .map(|t| t.layers)
            .min()
            .unwrap_or(1)
    }

    /// Whether `id` is one of the draw's colour or depth targets.
    fn is_target(&self, id: ImageId) -> bool {
        self.colors.iter().flatten().any(|t| t.id == id) || self.depth.is_some_and(|d| d.id == id)
    }

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
            sampler_memo: Box::new([MemoSampler::default(); SAMPLER_MEMO]),
            sampler_caps: SamplerCaps::from_device(device),
            anisotropy: Anisotropy::Auto,
            eds3: ash::ext::extended_dynamic_state3::Device::new(&device.instance.raw, &device.raw),
            vertex_input: ash::ext::vertex_input_dynamic_state::Device::new(
                &device.instance.raw,
                &device.raw,
            ),
            mesh: ash::ext::mesh_shader::Device::new(&device.instance.raw, &device.raw),
            depth_bias_control: ash::ext::depth_bias_control::Device::new(
                &device.instance.raw,
                &device.raw,
            ),
            push_descriptor: ash::khr::push_descriptor::Device::new(
                &device.instance.raw,
                &device.raw,
            ),
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
            targets_key: None,
            stages_memo: None,
            depth: None,
            presented: [None; PRESENTED],
            textures: Vec::new(),
            bound_textures: TextureMemo::default(),
            texture_scaling: ScalingBits::default(),
            storage_images: Vec::new(),
            element_handles: Vec::new(),
            rebind: false,
            image_scaling: ScalingBits::default(),
            texel_views: Vec::new(),
            texel_formats: vec![None; PixelFormat::ALL.len()],
            null_texel_views: Vec::new(),
            buffer_infos: Vec::new(),
            written_buffers: Vec::new(),
            compute_batch: ComputeBatch::default(),
            image_infos: Vec::new(),
            writes: Vec::new(),
            descriptor_infos: Vec::new(),
            vertex_buffers: Vec::new(),
            vertex_offsets: Vec::new(),
            vertex_strides: Vec::new(),
            attachments: Vec::new(),
            bound: Bound::default(),
            dynamic: DynamicShadow::default(),
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
        self.sampler_memo.fill(MemoSampler::default());
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

/// The components of each colour output the fragment program writes, as RGBA bits. The guest
/// leaves the others untouched, while Vulkan writes undefined values, so they are masked out.
fn fragment_output_components(stages: &GraphicsShaders) -> [u8; 8] {
    stages[4]
        .as_ref()
        .map_or([0; 8], |fragment| fragment.info.frag_color_components)
}

/// Whether render target 0 blends with the fragment program's second colour output.
fn blends_second_source(regs: &Regs) -> bool {
    if regs.blend.enable[0] == 0 {
        return false;
    }
    let (b, t) = (&regs.blend, &regs.blend_per_target[0]);
    let factors = match regs.blend_per_target_enabled {
        0 => [b.color_source, b.color_dest, b.alpha_source, b.alpha_dest],
        _ => [t.color_source, t.color_dest, t.alpha_source, t.alpha_dest],
    };
    factors.into_iter().any(|factor| {
        use vk::BlendFactor as F;
        matches!(
            mvk::blend_factor(factor),
            Some(F::SRC1_COLOR | F::ONE_MINUS_SRC1_COLOR | F::SRC1_ALPHA | F::ONE_MINUS_SRC1_ALPHA)
        )
    })
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
    /// What `runtime` takes from the previous stage, whose outputs this one reads (0 for
    /// vertex): the module hash would key every pairing with another program apart.
    previous: u64,
    generic_inputs: [AttributeType; 32],
    convert_depth_mode: bool,
    y_negate: bool,
    input_topology: InputTopology,
    point_size: Option<u32>,
    alpha_test: Option<(CompareFunction, u32)>,
    dual_source_blend: bool,
    color_outputs: [AttributeType; 8],
    noperspective_outputs: u32,
    scaled_noperspective_inputs: bool,
    /// The previous stage is a task program, so this mesh program reads its payload.
    task_stage: bool,
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
            previous: previous.map_or(0, |m| previous_outputs(&m.info)),
            generic_inputs: [AttributeType::Float; 32],
            convert_depth_mode: false,
            y_negate: regs.window_origin.mode() != WindowOriginMode::UPPER_LEFT,
            input_topology: InputTopology::Points,
            point_size: None,
            alpha_test: None,
            dual_source_blend: false,
            color_outputs: [AttributeType::Float; 8],
            noperspective_outputs: 0,
            scaled_noperspective_inputs: false,
            task_stage: previous.is_some_and(|m| m.stage == ShaderStage::Task),
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
            state.dual_source_blend = features.dual_src_blend && blends_second_source(regs);
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
            dual_source_blend: self.dual_source_blend,
            color_output_types: self.color_outputs,
            noperspective_outputs: self.noperspective_outputs,
            scaled_noperspective_inputs: self.scaled_noperspective_inputs,
            task_stage: self.task_stage,
            ..RuntimeInfo::default()
        };
        if let Some((func, reference)) = self.alpha_test {
            info.alpha_test_func = Some(func);
            info.alpha_test_reference = f32::from_bits(reference);
        }
        match previous {
            Some(previous) => {
                info.previous_stage_stores = previous_stores(previous);
                info.previous_stage_legacy_stores_mapping = previous.legacy_stores_mapping.clone();
                info.per_primitive_generics = per_primitive_generics(previous);
            }
            None => info.previous_stage_stores = VaryingState([u64::MAX; 8]),
        }
        info
    }
}

/// The hash of what [`StageState::runtime`] takes from the previous stage's `info`.
fn previous_outputs(info: &ShaderInfo) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = crate::pipeline::FxHasher::default();
    (
        previous_stores(info),
        &info.legacy_stores_mapping,
        per_primitive_generics(info),
    )
        .hash(&mut hasher);
    hasher.finish()
}

/// The attributes the next stage can read from the stage `info` describes: those it stores,
/// and those a passthrough geometry stage passes through, which a host running it natively
/// never stores itself.
fn previous_stores(info: &ShaderInfo) -> VaryingState {
    VaryingState(std::array::from_fn(|i| {
        info.stores.0[i] | info.passthrough.0[i]
    }))
}

/// The generic outputs the previous stage writes per primitive, as a mask.
fn per_primitive_generics(info: &ShaderInfo) -> u32 {
    info.mesh
        .primitive_lines
        .iter()
        .filter_map(|&line| Attribute(line as u16).generic_index())
        .fold(0, |mask, generic| mask | 1 << generic)
}

/// The constant, clamp, slope and representation of the draw's depth bias. The constant is
/// half the register in units of the guest depth format's minimum resolvable difference.
///
/// A unorm guest format's units hold whatever host format stands in for it: the float
/// representation turns them into absolute depth, and the force-unorm one keeps them at
/// 2^-n over the whole range. Force-unorm takes n from the host attachment (24 for a float
/// one), so only a guest D16 on a float host would get 2^-24 units; D16 is native on every
/// driver we run. Float guest formats, and devices with neither representation, use the host
/// format's own units.
fn depth_bias(
    regs: &Regs,
    features: &DeviceFeatures,
) -> (f32, f32, f32, vk::DepthBiasRepresentationEXT) {
    use DepthFormat as D;
    use vk::DepthBiasRepresentationEXT as R;
    let units = regs.depth_bias / 2.0;
    let unorm_bits = match DepthFormat::try_from(regs.zeta.format) {
        Ok(D::Z16Unorm) => Some(16),
        Ok(D::Z24UnormS8Uint | D::X8Z24Unorm | D::S8Z24Unorm | D::V8Z24Unorm) => Some(24),
        _ => None,
    };
    let (constant, representation) = match unorm_bits {
        Some(bits) if features.depth_bias_float => (units / (1u32 << bits) as f32, R::FLOAT),
        Some(_) if features.depth_bias_force_unorm => {
            (units, R::LEAST_REPRESENTABLE_VALUE_FORCE_UNORM)
        }
        _ => (units, R::LEAST_REPRESENTABLE_VALUE_FORMAT),
    };
    (
        constant,
        regs.depth_bias_clamp,
        regs.slope_scale_depth_bias,
        representation,
    )
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
    gmmu: &'a Arc<lichii_gpu::gmmu::Gmmu>,
    cbufs: &'a [ConstBufferBinding],
    /// Root table 0, which NVN2 graphics programs read as constant buffer slot 17.
    root_table: Option<&'a [u32; 64]>,
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
        DynamicShadow::register_dirty_flags(ctx.dirty);
        let regs = ctx.regs;
        let mesh = is_mesh_draw(regs, self.draw.arch);
        let host_topology = match mesh {
            true => mesh_topology(regs),
            false => mvk::primitive_topology(assembled_topology(call.topology()))
                .unwrap_or(vk::PrimitiveTopology::TRIANGLE_LIST),
        };
        if !self.render_targets(ctx)? {
            return Ok(());
        }
        let may_skip = self.draw.presented_only();
        if !self.graphics_stages(ctx, host_topology, may_skip) {
            return Ok(());
        }

        let draw = &mut self.draw;
        let mut fixed = *draw
            .fixed
            .refresh(regs, ctx.dirty, assembled_topology(call.topology()));
        let outputs = fragment_output_components(&draw.stages);
        for (blend, components) in fixed.blend.iter_mut().zip(outputs) {
            blend.write_mask &= components;
        }
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
        let pipeline = match may_skip {
            true => draw.pipelines.graphics(&key, &draw.stages),
            false => draw.pipelines.graphics_waiting(&key, &draw.stages),
        };
        let Some(pipeline) = pipeline else {
            return Ok(());
        };
        let pipeline_raw = pipeline.raw();
        let layout = pipeline.layout();
        let layout_raw = layout.raw();
        let push = layout.push_constants();
        let template = layout.set_template();

        let gmmu = ctx.memory;
        let mut removals = self.texture_cache.removals();
        let mut retargets = 0;
        let (index, indirect) = loop {
            let merged = self.buffer_cache.stats().merged_buffers;
            self.draw.rebind = false;
            self.draw.buffer_infos.clear();
            self.draw.written_buffers.clear();
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
                    root_table: Some(&ctx.root_tables[0]),
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
        let set = self.descriptor_set(template)?;

        let task = self.draw.stages[0]
            .as_ref()
            .is_some_and(|m| m.stage == ShaderStage::Task);
        let mesh_limit = self.device.properties.mesh_group_limit[usize::from(task)];
        let (index, indirect) = match (call, indirect) {
            (DrawCall::Indirect(draw), Some(inputs)) if mesh => {
                let cmd = self.scheduler.record()?;
                let commands = self.rewrite_mesh_indirect(cmd, draw, &inputs, mesh_limit)?;
                (None, Some(commands))
            }
            (DrawCall::Indirect(draw), Some(inputs))
                if assembled_topology(draw.topology) != draw.topology =>
            {
                let cmd = self.scheduler.record()?;
                let (index, commands) = self.assemble_indirect(cmd, regs, draw, &inputs)?;
                (Some(index), Some(commands))
            }
            (_, indirect) => (index, indirect),
        };
        for &id in &self.draw.textures {
            if !self.draw.is_target(id) {
                self.texture_cache
                    .prepare(&mut self.scheduler, id, SHADER_SAMPLED)?;
            }
        }
        for &(id, _) in &self.draw.storage_images {
            self.texture_cache
                .prepare(&mut self.scheduler, id, SHADER_STORAGE)?;
        }
        let scale = self.target_scale();
        let query = match regs.zpass_pixel_count_enable {
            0 => None,
            _ => {
                let slot = self
                    .samples
                    .reserve(&self.device, &mut self.scheduler, scale)?;
                match slot {
                    None => self.draw.warn_once(
                        Warning::SampleQueries,
                        "every occlusion query waits for its draw to finish; draws go uncounted",
                    ),
                    Some(slot) if SampleCounter::starts_block(slot) => self.fences.push_back((
                        self.scheduler.tick(),
                        Release::SamplesFold(self.samples.report()),
                    )),
                    Some(_) => {}
                }
                slot
            }
        };
        let extent = self.render_area(regs);
        let key = RenderingKey {
            colors: self
                .draw
                .colors
                .map(|t| t.map_or(vk::ImageView::null(), |t| t.view)),
            depth: self.draw.depth.map_or(vk::ImageView::null(), |d| d.view),
            extent,
        };
        let draw = &self.draw;
        let colors = draw.colors.map(|t| {
            t.map(|t| match draw.textures.contains(&t.id) {
                true => (t.id, FEEDBACK_COLOR),
                false => (t.id, ImageAccess::COLOR_ATTACHMENT),
            })
        });
        let depth = draw.depth.map(|d| match draw.textures.contains(&d.id) {
            true => (d.id, FEEDBACK_DEPTH),
            false => (d.id, DEPTH_ATTACHMENT),
        });
        let continues = self.scheduler.rendering() == Some(key)
            && colors.iter().flatten().chain(&depth).all(|&(id, access)| {
                ![FEEDBACK_COLOR, FEEDBACK_DEPTH].contains(&access)
                    && self.texture_cache.is_in(id, access)
            });
        if !continues {
            for &(id, access) in colors.iter().flatten().chain(&depth) {
                self.texture_cache
                    .prepare(&mut self.scheduler, id, access)?;
            }
        }
        let raw = &self.device.raw;
        let draw = &mut self.draw;
        draw.attachments.clear();
        let last = draw
            .colors
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |i| i + 1);
        for (target, access) in draw.colors[..last].iter().zip(colors) {
            let layout =
                access.map_or(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, |(_, a)| a.layout);
            draw.attachments.push(
                vk::RenderingAttachmentInfo::default()
                    .image_view(target.map_or(vk::ImageView::null(), |t| t.view))
                    .image_layout(layout)
                    .load_op(vk::AttachmentLoadOp::LOAD)
                    .store_op(vk::AttachmentStoreOp::STORE),
            );
        }
        let depth_attachment = draw.depth.zip(depth).map(|(target, (_, access))| {
            vk::RenderingAttachmentInfo::default()
                .image_view(target.view)
                .image_layout(access.layout)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE)
        });
        let depth_format = draw.depth.map_or(vk::Format::UNDEFINED, |d| d.format);
        let mut rendering = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(draw.layers())
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
        let (cmd, began) = self.scheduler.record_rendering(key, &rendering)?;
        self.buffer_cache.commit_writes(self.scheduler.tick());
        self.device.begin_label(
            cmd,
            format_args!(
                "draw {} -> {}x{}",
                Programs(&draw.stages),
                extent.width,
                extent.height
            ),
        );
        // SAFETY: `cmd` is recording inside the rendering of this draw's attachments; every
        // view, buffer and descriptor set is live until the scheduler's current tick completes,
        // and the pipeline was built for these attachment formats.
        unsafe {
            if let Some(slot) = query {
                self.samples.begin(&self.device, cmd, slot);
            }
            draw.dynamic.begin(began, mesh);
            draw.bound.begin(began);
            if rebinds(&mut draw.bound.pipeline, pipeline_raw) {
                raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline_raw);
            }
            match (set, template) {
                (Some(set), _) => raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    layout_raw,
                    0,
                    &[set],
                    &[],
                ),
                (None, Some(template)) => {
                    draw.push_descriptor.cmd_push_descriptor_set_with_template(
                        cmd,
                        template.raw,
                        layout_raw,
                        0,
                        draw.descriptor_infos.as_ptr().cast(),
                    )
                }
                (None, None) => {}
            }
        }
        let render_area = self
            .draw
            .stages
            .iter()
            .flatten()
            .any(|m| m.info.uses_render_area)
            .then(|| guest_render_area(regs));
        if let Some(push) = self.push_constants(layout_raw, push, render_area, scale)
            && rebinds(&mut self.draw.bound.push, Some(push))
        {
            push.record(&self.device, cmd);
        }
        self.set_dynamic_state(cmd, regs, ctx.dirty, host_topology, extent, scale);
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
        let bind_index = index.filter(|i| {
            rebinds(
                &mut self.draw.bound.index,
                Some((i.buffer, i.offset, i.index_type)),
            )
        });
        let raw = &self.device.raw;
        let draw = &self.draw;
        // SAFETY: as above; the pipeline's dynamic state is all set.
        unsafe {
            if let Some(index) = bind_index {
                raw.cmd_bind_index_buffer(cmd, index.buffer, index.offset, index.index_type);
            }
            let conditional = self.conditions.as_ref().and_then(|c| c.begin(cmd));
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
            if let Some(conditions) = conditional {
                conditions.end(cmd);
            }
            if let Some(slot) = query {
                self.samples.end(&self.device, cmd, slot);
            }
        }
        self.device.end_label(cmd);
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
        ctx: &mut GraphicsContext<'_>,
        topology: vk::PrimitiveTopology,
        may_skip: bool,
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
        let color_outputs: [AttributeType; 8] = std::array::from_fn(|i| {
            let count = (regs.rt_control.count() as usize).min(8);
            if i >= count {
                return AttributeType::Float;
            }
            let rt = &regs.rt[render_target_index(regs, i)];
            if rt.address() == 0 || rt.width == 0 || rt.height == 0 {
                return AttributeType::Float;
            }
            match RenderTargetFormat::try_from(rt.format)
                .ok()
                .and_then(PixelFormat::from_render_target)
                .map(PixelFormat::numeric)
            {
                Some(Numeric::Uint) => AttributeType::UnsignedInt,
                Some(Numeric::Sint) => AttributeType::SignedInt,
                _ => AttributeType::Float,
            }
        });
        let arch = self.draw.arch;
        let words = regs.words();
        let mut inputs = StageInputs {
            channel: ctx.channel,
            topology,
            memory: ctx.memory.generation(),
            programs: self.draw.shaders.generation(),
            color_outputs,
            registers: [0; STAGE_REGISTER_WORDS],
            headers: [[0; 20]; 6],
        };
        let mut at = 0;
        for (first, len) in STAGE_REGISTERS {
            inputs.registers[at..at + len].copy_from_slice(&words[first..first + len]);
            at += len;
        }
        for (slot, header) in inputs.headers.iter_mut().enumerate() {
            if slot == 1 || regs.pipelines[slot].enable() != 0 {
                *header = graphics_program_header(ctx, slot, arch).words;
            }
        }
        if self.reuse_stages(ctx, mesh, &inputs) {
            return true;
        }
        self.draw.stages_memo = None;

        let last_pre_raster = (1..5).rev().find(|&s| enabled(s)).unwrap_or(1);
        let features = *self.draw.pipelines.features();
        let passthrough_geometry = enabled(4)
            && ProgramHeader {
                words: inputs.headers[4],
            }
            .geometry_passthrough();
        let noperspective = if mesh || !enabled(5) || passthrough_geometry {
            0
        } else {
            ProgramHeader {
                words: inputs.headers[5],
            }
            .ps()
            .screen_linear_generics()
        };
        let mut bindings = Bindings::default();
        let mut previous: Option<Arc<ShaderModule>> = None;
        let mut keys = [None; 5];
        let mut tracked = true;
        for (slot, host) in GRAPHICS_SLOTS {
            if !enabled(slot) {
                self.draw.stages[host] = None;
                continue;
            }
            let env = ProgramEnv::graphics(ctx, slot, arch);
            let vertex_a = (slot == 1 && !mesh && regs.pipelines[0].enable() != 0)
                .then(|| ProgramEnv::graphics(ctx, 0, arch));
            tracked &= [Some(&env), vertex_a.as_ref()]
                .into_iter()
                .flatten()
                .all(|env| ctx.memory.translate(env.code_start()).is_some());
            let mut state = StageState::new(
                regs,
                slot,
                slot == last_pre_raster,
                previous.as_deref(),
                topology,
                color_outputs,
                &features,
            );
            if slot == last_pre_raster {
                state.noperspective_outputs = noperspective;
            }
            state.scaled_noperspective_inputs = slot == 5 && noperspective != 0;
            let found = self.draw.shaders.module(
                vertex_a.as_ref(),
                &env,
                state.key(),
                || state.runtime(previous.as_deref().map(|m| &m.info)),
                &mut bindings,
                may_skip,
            );
            let Some((key, module)) = found else {
                return false;
            };
            if slot != 5 {
                previous = Some(module.clone());
            }
            keys[host] = Some(key);
            self.draw.stages[host] = Some(module);
        }
        if tracked {
            self.draw.stages_memo = Some(StagesMemo { inputs, keys });
        }
        true
    }

    /// Whether the stages of the last draw are this draw's: its inputs match and every
    /// program's variant still answers its translation's queries.
    fn reuse_stages(
        &mut self,
        ctx: &GraphicsContext<'_>,
        mesh: bool,
        inputs: &StageInputs,
    ) -> bool {
        let draw = &mut self.draw;
        let Some(memo) = draw
            .stages_memo
            .as_ref()
            .filter(|memo| memo.inputs == *inputs)
        else {
            return false;
        };
        let regs = ctx.regs;
        GRAPHICS_SLOTS.into_iter().all(|(slot, host)| {
            let Some(key) = &memo.keys[host] else {
                return true;
            };
            let env = ProgramEnv::graphics(ctx, slot, draw.arch);
            let vertex_a = (slot == 1 && !mesh && regs.pipelines[0].enable() != 0)
                .then(|| ProgramEnv::graphics(ctx, 0, draw.arch));
            let found = draw.shaders.find(key, vertex_a.as_ref(), &env);
            found
                .zip(draw.stages[host].as_ref())
                .is_some_and(|(found, current)| Arc::ptr_eq(&found, current))
        })
    }

    /// Finds the colour and depth targets in the texture cache. `false` skips the draw.
    fn render_targets(&mut self, ctx: &mut GraphicsContext<'_>) -> Result<bool> {
        let regs = ctx.regs;
        let msaa = regs.anti_alias_samples_mode.0;
        let layered = regs.rt_layer.control() == RtLayerControl::GEOMETRY_SHADER_SELECTS_LAYER.0;
        let key = TargetsKey {
            channel: ctx.channel,
            msaa,
            layered,
            zeta_size: regs.zeta_size.bits.0,
            memory: ctx.memory.generation(),
            cache: self.texture_cache.generation(),
        };
        if !ctx.dirty.is_set(dirty::RENDER_TARGETS) && self.draw.targets_key == Some(key) {
            let draw = &self.draw;
            let cache = &mut self.texture_cache;
            let mut targets = draw.colors.iter().flatten().chain(draw.depth.iter());
            if targets.all(|t| cache.keep_target(t.id)) {
                return Ok(true);
            }
        }
        let gob = self.texture_cache.gob_layout();
        self.draw.colors = [None; 8];
        self.draw.depth = None;
        let count = (regs.rt_control.count() as usize).min(8);
        let layers = |info: ImageInfo| if layered { info } else { info.with_layers(1) };
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
            buffers: &mut self.buffer_cache,
        };
        for i in 0..count {
            let rt = &regs.rt[render_target_index(regs, i)];
            let Some(info) = ImageInfo::from_render_target(rt, msaa, gob).map(layers) else {
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
            self.draw.colors[i] =
                Self::target_view(&mut self.texture_cache, found, &info, layered)?;
        }
        if regs.zeta_enable != 0
            && let Some(info) =
                ImageInfo::from_zeta(&regs.zeta, &regs.zeta_size, msaa, gob).map(layers)
            && let Some(found) = self.texture_cache.render_target(
                &mut tx,
                ctx.memory,
                regs.zeta.address(),
                &info,
                false,
            )?
        {
            self.draw.depth = Self::target_view(&mut self.texture_cache, found, &info, layered)?;
        }
        ctx.dirty.clear(dirty::RENDER_TARGETS);
        self.draw.targets_key = Some(TargetsKey {
            cache: self.texture_cache.generation(),
            ..key
        });
        Ok(true)
    }

    fn target_view(
        cache: &mut crate::texture_cache::TextureCache,
        found: crate::texture_cache::Found,
        info: &ImageInfo,
        layered: bool,
    ) -> Result<Option<Target>> {
        let Some(state) = cache.state(found.id) else {
            return Ok(None);
        };
        let mut view = ViewInfo::whole(info);
        view.base_level += found.level;
        view.levels = 1;
        view.base_layer += found.layer;
        view.layers = match state.info.kind {
            ImageKind::D3 if layered => (state.extent.depth >> found.level)
                .saturating_sub(view.base_layer)
                .max(1),
            _ => info.layers,
        };
        view.view_type = match view.layers {
            1 => vk::ImageViewType::TYPE_2D,
            _ => vk::ImageViewType::TYPE_2D_ARRAY,
        };
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
            layers: view.layers,
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
                            let index = (desc.index + i) as usize;
                            let size = info
                                .constant_buffer_used_sizes
                                .get(index)
                                .map_or(cbuf.size, |&used| cbuf.size.min(used));
                            let binding = match src.root_table {
                                _ if cbuf.enabled => self.buffer_cache.uniform_buffer(
                                    &mut self.scheduler,
                                    &mut self.staging,
                                    src.gmmu,
                                    cbuf.address,
                                    u64::from(size),
                                )?,
                                Some(table) if index == ROOT_TABLE_SLOT => {
                                    Some(self.buffer_cache.stream_uniform(
                                        &mut self.scheduler,
                                        bytemuck::bytes_of(table),
                                        info.constant_buffer_used_sizes[index],
                                    )?)
                                }
                                _ => None,
                            };
                            self.push_buffer(binding, false);
                        }
                    }
                    DescriptorKind::StorageBuffer => {
                        let desc = info.storage_buffers_descriptors[descriptor];
                        for i in 0..b.count {
                            let limit = storage_range_limit(self);
                            let (address, size) = storage_pointer(src, &desc, i, limit);
                            let binding = self.buffer_cache.storage_buffer(
                                &mut self.scheduler,
                                &mut self.staging,
                                src.gmmu,
                                address,
                                size,
                                desc.is_written,
                            )?;
                            self.push_buffer(binding, desc.is_written);
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
                            let bound = self.sampled_texture(src, raw, &desc)?;
                            let rescaled = bound.is_some_and(|b| b.rescaled);
                            let texture = bound.map(|b| (b.id, b.view));
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
                                    let feedback = module.stage != ShaderStage::Compute
                                        && self.draw.is_target(id);
                                    match feedback {
                                        true => (view, vk::ImageLayout::GENERAL),
                                        false => (view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
                                    }
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
        let Some((id, mut view)) = self.texture_info(src, raw, desc.ty, Usage::Storage)? else {
            return Ok(None);
        };
        if !desc.is_written && self.texture_cache.unwritten(id) {
            return Ok(None);
        }
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
                buffers: &mut self.buffer_cache,
            };
            self.texture_cache.scale_down(&mut tx, id)?;
            self.draw.rebind = true;
        }
        if let Some(format) = storage_format(desc.format) {
            view.format = format;
        }
        if desc.is_written {
            let mut tx = Transfer {
                scheduler: &mut self.scheduler,
                staging: &mut self.staging,
                memory: &self.handles.device,
                buffers: &mut self.buffer_cache,
            };
            self.texture_cache.bind_written(
                &mut tx,
                id,
                view.base_level,
                view.base_layer..view.base_layer + view.layers,
            )?;
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

    fn push_buffer(&mut self, binding: Option<crate::buffer_cache::BufferBinding>, written: bool) {
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
        if written {
            self.draw.written_buffers.push(info);
        }
        self.draw.buffer_infos.push(info);
    }

    /// The texture handle `raw` binds for `desc`; `None` binds a null texture. A header bound
    /// before is bound again without a lookup while its image needs no refresh.
    fn sampled_texture(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        desc: &TextureDescriptor,
    ) -> Result<Option<BoundTexture>> {
        if desc.handle_is_pointer_word {
            return Ok(None);
        }
        let (index, _) = src.pool.split(raw);
        let key = (index <= src.pool.limit)
            .then(|| {
                src.gmmu
                    .read::<[u32; 8]>(src.pool.address + u64::from(index) * 32)
                    .ok()
            })
            .flatten()
            .map(|header| TextureKey {
                header,
                ty: desc.ty,
                checked: desc.handle_program.is_some(),
                integer: desc.is_integer,
            });
        let memory = src.gmmu.generation();
        let memo = &mut self.draw.bound_textures;
        if memo.cache != self.texture_cache.texture_generation()
            || memo.gmmu.as_ptr() != Arc::as_ptr(src.gmmu)
        {
            memo.cache = self.texture_cache.texture_generation();
            memo.gmmu = Arc::downgrade(src.gmmu);
            memo.bound.clear();
        }
        if let Some(bound) = key.and_then(|key| memo.bound.get(&key).copied())
            && bound.memory == memory
            && self.texture_cache.keep_texture(bound.id)
        {
            return Ok(Some(bound));
        }
        let readable = desc.handle_program.is_none_or(|_| {
            let header = src.pool.header(src.gmmu, raw);
            texture_type(&header) == desc.ty && header.format.is_integer() == desc.is_integer
        });
        if !readable {
            return Ok(None);
        }
        let Some((id, view)) = self.texture_view(src, raw, desc.ty)? else {
            return Ok(None);
        };
        let bound = BoundTexture {
            memory,
            id,
            view,
            rescaled: self
                .texture_cache
                .state(id)
                .is_some_and(|s| s.scale != ResolutionScale::X1),
        };
        let memo = &mut self.draw.bound_textures;
        if memo.cache != self.texture_cache.texture_generation()
            || memo.bound.len() >= MAX_BOUND_TEXTURES
        {
            memo.cache = self.texture_cache.texture_generation();
            memo.bound.clear();
        }
        if let Some(key) = key {
            memo.bound.insert(key, bound);
        }
        Ok(Some(bound))
    }

    /// The image and view of the texture `raw` selects, read as `ty`; `None` when nothing
    /// backs it.
    fn texture_view(
        &mut self,
        src: &StageSources<'_>,
        raw: u32,
        ty: TextureType,
    ) -> Result<Option<(ImageId, vk::ImageView)>> {
        let Some((id, view)) = self.texture_info(src, raw, ty, Usage::Sample)? else {
            return Ok(None);
        };
        if self.texture_cache.unwritten(id) {
            return Ok(None);
        }
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
        usage: Usage,
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
            buffers: &mut self.buffer_cache,
        };
        let Some((id, mut view)) = self
            .texture_cache
            .texture(&mut tx, src.gmmu, &header, usage)?
        else {
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
        let anisotropy = self.draw.anisotropy as u8;
        let slot = &mut self.draw.sampler_memo[index as usize % SAMPLER_MEMO];
        if slot.words == words
            && slot.anisotropy == anisotropy
            && slot.sampler != vk::Sampler::null()
        {
            return Ok(slot.sampler);
        }
        let sampler = match self.draw.samplers.get(&(words, anisotropy)) {
            Some(&sampler) => sampler,
            None => {
                let params =
                    SamplerParams::new(&Tsc(words), self.draw.anisotropy, &self.draw.sampler_caps);
                let sampler = params.create(&self.device)?;
                self.draw.samplers.insert((words, anisotropy), sampler);
                sampler
            }
        };
        self.draw.sampler_memo[index as usize % SAMPLER_MEMO] = MemoSampler {
            words,
            anisotropy,
            sampler,
        };
        Ok(sampler)
    }

    /// Binds every vertex stream an enabled attribute reads, into the draw's vertex lists.
    fn vertex_buffers(&mut self, gmmu: &Arc<lichii_gpu::gmmu::Gmmu>, regs: &Regs) -> Result<()> {
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
                .then(|| vertex_stream_range(regs, stream, self.draw.arch))
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
        gmmu: &Arc<lichii_gpu::gmmu::Gmmu>,
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
        let count = crate::buffer_cache::bound_index_capacity(gmmu, buffer, width);
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
        gmmu: &Arc<lichii_gpu::gmmu::Gmmu>,
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

    /// Gathers the draw's descriptors into `descriptor_infos` in `template`'s binding order and,
    /// when the template writes an allocated set, allocates and writes it. `None`: there is no
    /// set to bind, because the layout has no descriptors or the template pushes them.
    fn descriptor_set(
        &mut self,
        template: Option<SetTemplate>,
    ) -> Result<Option<vk::DescriptorSet>> {
        let Some(template) = template else {
            return Ok(None);
        };
        let draw = &mut self.draw;
        draw.writes
            .sort_unstable_by_key(|write| (write.binding, write.first));
        draw.descriptor_infos.clear();
        let mut last = None;
        for write in &draw.writes {
            let range = write.first..write.first + write.count as usize;
            if last == Some(write.binding) {
                let len = draw.descriptor_infos.len();
                draw.descriptor_infos.truncate(len - range.len());
            }
            last = Some(write.binding);
            let infos = &mut draw.descriptor_infos;
            match write.kind {
                DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer => infos.extend(
                    draw.buffer_infos[range]
                        .iter()
                        .map(|&buffer| DescriptorInfo { buffer }),
                ),
                DescriptorKind::UniformTexelBuffer | DescriptorKind::StorageTexelBuffer => infos
                    .extend(
                        draw.texel_views[range]
                            .iter()
                            .map(|&texel| DescriptorInfo { texel }),
                    ),
                _ => infos.extend(
                    draw.image_infos[range]
                        .iter()
                        .map(|&image| DescriptorInfo { image }),
                ),
            }
        }
        assert_eq!(
            draw.descriptor_infos.len(),
            template.descriptors,
            "the draw's writes cover its layout's set 0"
        );
        let Some(layout) = template.allocate else {
            return Ok(None);
        };
        let set = draw.descriptors.allocate(&mut self.scheduler, layout)?;
        // SAFETY: `descriptor_infos` holds the template's `descriptors` infos in its entries'
        // order and stride: the layout merges the bindings of the draw's modules, and
        // `bind_resources` writes all `count` elements of every one of them or skips the draw.
        // The set is live and unused by recorded work.
        unsafe {
            self.device.raw.update_descriptor_set_with_template(
                set,
                template.raw,
                draw.descriptor_infos.as_ptr().cast(),
            )
        };
        Ok(Some(set))
    }

    /// The push constants of the one range every stage of `layout` shares (`push` is its stages
    /// and size), `None` when it has none: the texture rescaling words of the bound textures at 0
    /// (storage images are never bound, so their words stay 0) with the down factor of `scale`,
    /// the targets' scale, at 24, and the render area at `RENDER_AREA_LAYOUT_OFFSET` (32), so a
    /// 48-byte range holds both.
    fn push_constants(
        &self,
        layout: vk::PipelineLayout,
        (stages, size): (vk::ShaderStageFlags, u32),
        render_area: Option<(f32, f32)>,
        scale: ResolutionScale,
    ) -> Option<PushConstants> {
        if size == 0 {
            return None;
        }
        let mut bytes = [0u8; PUSH_BYTES];
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
        Some(PushConstants {
            layout,
            stages,
            size,
            bytes,
        })
    }

    /// Records the dynamic state the pipeline declares that changed since the open rendering
    /// last set it.
    fn set_dynamic_state(
        &mut self,
        cmd: vk::CommandBuffer,
        regs: &Regs,
        dirty: &mut DirtyState,
        topology: vk::PrimitiveTopology,
        extent: vk::Extent2D,
        scale: ResolutionScale,
    ) {
        let device = &self.device;
        let raw = &device.raw;
        let draw = &mut self.draw;
        let features = *draw.pipelines.features();
        let inputs = DynamicInputs {
            features,
            device: &device.features,
            mesh: is_mesh_draw(regs, draw.arch),
            topology,
            extent,
            scale,
            vertex_buffers: &draw.vertex_buffers,
            vertex_offsets: &draw.vertex_offsets,
            vertex_strides: &draw.vertex_strides,
            colors: draw.colors.map(|t| t.map(|t| t.format)),
            attachments: draw.attachments.len(),
            outputs: fragment_output_components(&draw.stages),
        };
        let blendable = |format| {
            device
                .format_features(format)
                .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND)
        };
        let (eds3, vertex_input, bias_control) =
            (&draw.eds3, &draw.vertex_input, &draw.depth_bias_control);
        let device_features = &device.features;
        let mut record = |command: &Command<'_>| {
            use vk::StencilFaceFlags as F;
            // SAFETY: `cmd` is recording inside a dynamic rendering with a pipeline that
            // declares these dynamic states; extension commands run only when their feature is
            // on.
            unsafe {
                match *command {
                    Command::LineWidth(width) => raw.cmd_set_line_width(cmd, width),
                    Command::DepthBias((constant, clamp, slope, representation)) => {
                        if !device_features.depth_bias_control {
                            return raw.cmd_set_depth_bias(cmd, constant, clamp, slope);
                        }
                        let mut representation = vk::DepthBiasRepresentationInfoEXT::default()
                            .depth_bias_representation(representation)
                            .depth_bias_exact(device_features.depth_bias_exact);
                        let info = vk::DepthBiasInfoEXT::default()
                            .depth_bias_constant_factor(constant)
                            .depth_bias_clamp(clamp)
                            .depth_bias_slope_factor(slope)
                            .push_next(&mut representation);
                        (bias_control.fp().cmd_set_depth_bias2_ext)(cmd, &info)
                    }
                    Command::BlendConstants(constants) => {
                        raw.cmd_set_blend_constants(cmd, &constants)
                    }
                    Command::DepthBounds((min, max)) => raw.cmd_set_depth_bounds(cmd, min, max),
                    Command::StencilCompareMask((front, back)) => {
                        raw.cmd_set_stencil_compare_mask(cmd, F::FRONT, front);
                        raw.cmd_set_stencil_compare_mask(cmd, F::BACK, back);
                    }
                    Command::StencilWriteMask((front, back)) => {
                        raw.cmd_set_stencil_write_mask(cmd, F::FRONT, front);
                        raw.cmd_set_stencil_write_mask(cmd, F::BACK, back);
                    }
                    Command::StencilReference((front, back)) => {
                        raw.cmd_set_stencil_reference(cmd, F::FRONT, front);
                        raw.cmd_set_stencil_reference(cmd, F::BACK, back);
                    }
                    Command::Viewport(viewport) if features.extended_dynamic_state => {
                        raw.cmd_set_viewport_with_count(cmd, &[viewport])
                    }
                    Command::Viewport(viewport) => raw.cmd_set_viewport(cmd, 0, &[viewport]),
                    Command::Scissor(scissor) if features.extended_dynamic_state => {
                        raw.cmd_set_scissor_with_count(cmd, &[scissor])
                    }
                    Command::Scissor(scissor) => raw.cmd_set_scissor(cmd, 0, &[scissor]),
                    Command::CullMode(cull) => raw.cmd_set_cull_mode(cmd, cull),
                    Command::FrontFace(face) => raw.cmd_set_front_face(cmd, face),
                    Command::PrimitiveTopology(topology) => {
                        raw.cmd_set_primitive_topology(cmd, topology)
                    }
                    Command::DepthTestEnable(on) => raw.cmd_set_depth_test_enable(cmd, on),
                    Command::DepthWriteEnable(on) => raw.cmd_set_depth_write_enable(cmd, on),
                    Command::DepthCompareOp(op) => raw.cmd_set_depth_compare_op(cmd, op),
                    Command::DepthBoundsTestEnable(on) => {
                        raw.cmd_set_depth_bounds_test_enable(cmd, on)
                    }
                    Command::StencilTestEnable(on) => raw.cmd_set_stencil_test_enable(cmd, on),
                    Command::StencilOp((front, back)) => {
                        raw.cmd_set_stencil_op(cmd, F::FRONT, front.0, front.1, front.2, front.3);
                        raw.cmd_set_stencil_op(cmd, F::BACK, back.0, back.1, back.2, back.3);
                    }
                    Command::RasterizerDiscardEnable(on) => {
                        raw.cmd_set_rasterizer_discard_enable(cmd, on)
                    }
                    Command::DepthBiasEnable(on) => raw.cmd_set_depth_bias_enable(cmd, on),
                    Command::PrimitiveRestartEnable(on) => {
                        raw.cmd_set_primitive_restart_enable(cmd, on)
                    }
                    Command::VertexBuffers((buffers, offsets, strides)) => {
                        if features.extended_dynamic_state && !features.dynamic_vertex_input {
                            raw.cmd_bind_vertex_buffers2(
                                cmd,
                                0,
                                buffers,
                                offsets,
                                None,
                                Some(strides),
                            );
                        } else {
                            raw.cmd_bind_vertex_buffers(cmd, 0, buffers, offsets);
                        }
                    }
                    Command::VertexInput((bindings, attributes)) => {
                        vertex_input.cmd_set_vertex_input(cmd, bindings, attributes)
                    }
                    Command::ColorBlendEnable(enables) => {
                        eds3.cmd_set_color_blend_enable(cmd, 0, enables)
                    }
                    Command::ColorBlendEquation(equations) => {
                        eds3.cmd_set_color_blend_equation(cmd, 0, equations)
                    }
                    Command::ColorWriteMask(masks) => eds3.cmd_set_color_write_mask(cmd, 0, masks),
                    Command::DepthClampEnable(on) => eds3.cmd_set_depth_clamp_enable(cmd, on),
                    Command::LogicOpEnable(on) => eds3.cmd_set_logic_op_enable(cmd, on),
                    Command::AlphaToCoverageEnable(on) => {
                        eds3.cmd_set_alpha_to_coverage_enable(cmd, on)
                    }
                    Command::AlphaToOneEnable(on) => eds3.cmd_set_alpha_to_one_enable(cmd, on),
                    Command::PolygonMode(mode) => eds3.cmd_set_polygon_mode(cmd, mode),
                    Command::DepthClipNegativeOneToOne(on) => {
                        eds3.cmd_set_depth_clip_negative_one_to_one(cmd, on)
                    }
                    Command::ProvokingVertexMode(mode) => {
                        eds3.cmd_set_provoking_vertex_mode(cmd, mode)
                    }
                }
            }
        };
        draw.dynamic
            .apply(&inputs, regs, dirty, blendable, &mut record);
    }

    pub(super) fn dispatch(&mut self, ctx: &ComputeContext<'_>) -> Result<()> {
        let env = ProgramEnv::compute(ctx);
        let mut bindings = Bindings::default();
        let Some((_, module)) =
            self.draw
                .shaders
                .module(None, &env, 0, RuntimeInfo::default, &mut bindings, true)
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
        let template = pipeline.layout().set_template();
        let sources = StageSources {
            gmmu: ctx.memory,
            cbufs: ctx.const_buffers,
            root_table: None,
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
            self.draw.written_buffers.clear();
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
        let set = self.descriptor_set(template)?;
        let batched = grid.is_none()
            && self.draw.texel_views.is_empty()
            && compute_batch::bounded_by_bindings(&module.info, env.pool().switch2);
        let draw = &self.draw;
        let cmd = if batched
            && self.scheduler.compute_barrier_owed()
            && draw
                .compute_batch
                .admits(&draw.buffer_infos, &draw.written_buffers)
        {
            self.scheduler.record_beside_dispatches()?
        } else {
            self.scheduler.record()?
        };
        self.buffer_cache.commit_writes(self.scheduler.tick());
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
            self.texture_cache
                .prepare(&mut self.scheduler, id, SHADER_SAMPLED)?;
        }
        for &(id, _) in &self.draw.storage_images {
            self.texture_cache
                .prepare(&mut self.scheduler, id, SHADER_STORAGE)?;
        }
        let raw = &self.device.raw;
        let [x, y, z] = ctx.grid;
        // SAFETY: `cmd` is recording outside any rendering; the set and pipeline are live
        // until the scheduler's current tick completes.
        unsafe {
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline_raw);
            match (set, template) {
                (Some(set), _) => raw.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    layout_raw,
                    0,
                    &[set],
                    &[],
                ),
                (None, Some(template)) => self
                    .draw
                    .push_descriptor
                    .cmd_push_descriptor_set_with_template(
                        cmd,
                        template.raw,
                        layout_raw,
                        0,
                        self.draw.descriptor_infos.as_ptr().cast(),
                    ),
                (None, None) => {}
            }
        }
        if let Some(push) = self.push_constants(layout_raw, push, None, ResolutionScale::X1) {
            push.record(&self.device, cmd);
        }
        match grid {
            Some(_) => self
                .device
                .begin_label(cmd, format_args!("dispatch indirect {}", Program(&module))),
            None => self.device.begin_label(
                cmd,
                format_args!("dispatch {x}x{y}x{z} {}", Program(&module)),
            ),
        }
        let raw = &self.device.raw;
        // SAFETY: as above.
        unsafe {
            match grid {
                Some(command) => raw.cmd_dispatch_indirect(cmd, command, 0),
                None => raw.cmd_dispatch(cmd, x, y, z),
            }
            self.device.end_label(cmd);
        }
        if !self.scheduler.compute_barrier_owed() {
            self.draw.compute_batch.clear();
        }
        self.scheduler.owe_compute_barrier();
        if batched {
            let draw = &mut self.draw;
            draw.compute_batch
                .add(&draw.buffer_infos, &draw.written_buffers);
        } else {
            self.scheduler.record()?;
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
        let limit = storage_range_limit(self);
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
                let (address, size) = storage_pointer(src, desc, element, limit);
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
            self.texture_cache.rearm(daddr, daddr + 4);
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
/// the constant buffer, or (indirect) a pointer table the constant buffer points to. No size
/// passes `limit`, the largest range the host binds after alignment.
fn storage_pointer(
    src: &StageSources<'_>,
    desc: &StorageBufferDescriptor,
    element: u32,
    limit: u64,
) -> (GpuVAddr, u64) {
    let offset = desc.cbuf_offset + element * 16;
    let pointer: u64 = read_cbuf(src.gmmu, src.cbufs, desc.cbuf_index, offset);
    if desc.indirect_offset == StorageBufferDescriptor::DIRECT {
        let size: u32 = read_cbuf(src.gmmu, src.cbufs, desc.cbuf_index, offset + 8);
        return (pointer, u64::from(size).min(limit));
    }
    let address = src
        .gmmu
        .read::<u64>(pointer + u64::from(desc.indirect_offset))
        .unwrap_or(0);
    (address, (64 << 10u64).min(limit))
}

/// The largest storage buffer range a binding may ask for: the host's limit, less the bytes
/// [`BufferCache::storage_buffer`](crate::buffer_cache::BufferCache::storage_buffer) adds when it
/// aligns the start down.
fn storage_range_limit(rasterizer: &VulkanRasterizer) -> u64 {
    let range = u64::from(rasterizer.device.properties.limits.max_storage_buffer_range);
    range.saturating_sub(rasterizer.buffer_cache.storage_alignment())
}

mod assembly;
mod compute_batch;
mod dispatch_grid;
mod dynamic;
mod frame_decode;
mod mesh_indirect;
mod null;
mod texture;

#[cfg(test)]
mod tests;
