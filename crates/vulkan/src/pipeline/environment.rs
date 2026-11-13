//! What the shader translator reads from the guest for one program: its code through the
//! GPU address space, its constant buffers, the texture headers its handles select and the few
//! registers translation depends on.
//!
//! The environment records every texture query the translator makes; a cached module stays
//! valid while [`ProgramEnv::replay`] gives the same answers, so changing a texture's type or
//! format under an unchanged program translates it again.

use std::cell::{Cell, RefCell};

use lichii_common::GpuVAddr;
use lichii_gpu::backend::{ComputeContext, ConstBufferBinding, GraphicsContext, QmdVersion};
use lichii_gpu::engines::maxwell_3d::regs::SamplerBinding;
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::info::TexturePixelFormat;
use lichii_shader::ir::TextureType;
use lichii_shader::{ShaderArch, ShaderEnvironment, ShaderStage};
use lichii_texture::tic::{self, TextureDescriptor, TicEntry};
use lichii_texture::{Numeric, PixelFormat};

/// Size of the Maxwell program header in front of a graphics program's first instruction.
pub const HEADER_SIZE: u64 = 0x50;

/// A query the translator made of guest state outside the program bytes, and the answer it
/// got: texture header queries by raw handle, and constant buffer words (bindless handles,
/// indirect branch tables) by binding and offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvRead {
    Type(u32, TextureType),
    PixelFormat(u32, TexturePixelFormat),
    Integer(u32, bool),
    Cbuf { index: u32, offset: u32, value: u32 },
}

/// The texture header pool texture handles index.
#[derive(Clone, Copy, Debug)]
pub struct TexturePool {
    pub address: GpuVAddr,
    pub limit: u32,
    /// Samplers take the header index (`SamplerBinding::VIA_HEADER_BINDING`, linked TSC).
    pub via_header_index: bool,
    /// Switch 2 handles may hold byte offsets into the pool (see `tic::texture_header_index`).
    pub switch2: bool,
}

impl TexturePool {
    /// The header index and sampler index of a raw handle.
    pub fn split(&self, raw: u32) -> (u32, u32) {
        let header = raw & 0xf_ffff;
        let sampler = if self.via_header_index {
            header
        } else {
            raw >> 20
        };
        (
            tic::texture_header_index(header, self.limit, self.switch2),
            sampler,
        )
    }

    /// The decoded header a raw handle selects. Handles past the pool and undecodable headers
    /// read as a plain 2D RGBA8 texture, which is what the null view bound for them samples.
    pub fn header(&self, gmmu: &Gmmu, raw: u32) -> TextureDescriptor {
        let (index, _) = self.split(raw);
        let entry = (index <= self.limit)
            .then(|| {
                gmmu.read::<TicEntry>(self.address + u64::from(index) * 32)
                    .ok()
            })
            .flatten();
        entry.and_then(|e| e.decode().ok()).unwrap_or(NULL_TEXTURE)
    }
}

const NULL_TEXTURE: TextureDescriptor = TextureDescriptor {
    address: 0,
    format: PixelFormat::A8B8G8R8Unorm,
    swizzle: tic::SwizzleSource::IDENTITY,
    texture_type: tic::TextureType::D2,
    memory: tic::TextureMemory::Pitch { pitch: 0 },
    width: 1,
    height: 1,
    depth: 1,
    levels: 1,
    base_layer: 0,
    min_level: 0,
    max_level: 0,
    msaa: tic::MsaaMode::X1,
    srgb: false,
    sparse: false,
    depth_texture: false,
    normalized_coords: true,
};

fn texture_type(desc: &TextureDescriptor) -> TextureType {
    use tic::TextureType as T;
    match desc.texture_type {
        T::D1 => TextureType::Color1D,
        T::D2 | T::D2NoMipmap if desc.normalized_coords => TextureType::Color2D,
        T::D2 | T::D2NoMipmap => TextureType::Color2DRect,
        T::D3 => TextureType::Color3D,
        T::Cube => TextureType::ColorCube,
        T::D1Array => TextureType::ColorArray1D,
        T::D2Array => TextureType::ColorArray2D,
        T::D1Buffer => TextureType::Buffer,
        T::CubeArray => TextureType::ColorArrayCube,
    }
}

/// The translator's view of a pixel format; formats it does not name read as their closest
/// layout.
pub fn shader_pixel_format(format: PixelFormat) -> TexturePixelFormat {
    use PixelFormat as P;
    use TexturePixelFormat as T;
    match format {
        P::A8B8G8R8Unorm => T::A8b8g8r8Unorm,
        P::A8B8G8R8Snorm => T::A8b8g8r8Snorm,
        P::A8B8G8R8Sint => T::A8b8g8r8Sint,
        P::A8B8G8R8Uint => T::A8b8g8r8Uint,
        P::R5G6B5Unorm => T::R5g6b5Unorm,
        P::B5G6R5Unorm => T::B5g6r5Unorm,
        P::A1R5G5B5Unorm => T::A1r5g5b5Unorm,
        P::A2B10G10R10Unorm => T::A2b10g10r10Unorm,
        P::A2B10G10R10Uint => T::A2b10g10r10Uint,
        P::A2R10G10B10Unorm => T::A2r10g10b10Unorm,
        P::A1B5G5R5Unorm => T::A1b5g5r5Unorm,
        P::A5B5G5R1Unorm => T::A5b5g5r1Unorm,
        P::R8Unorm => T::R8Unorm,
        P::R8Snorm => T::R8Snorm,
        P::R8Sint => T::R8Sint,
        P::R8Uint => T::R8Uint,
        P::R16G16B16A16Float => T::R16g16b16a16Float,
        P::R16G16B16A16Unorm => T::R16g16b16a16Unorm,
        P::R16G16B16A16Snorm => T::R16g16b16a16Snorm,
        P::R16G16B16A16Sint => T::R16g16b16a16Sint,
        P::R16G16B16A16Uint => T::R16g16b16a16Uint,
        P::B10G11R11Float => T::B10g11r11Float,
        P::R32G32B32A32Uint => T::R32g32b32a32Uint,
        P::Bc1RgbaUnorm => T::Bc1RgbaUnorm,
        P::Bc2Unorm => T::Bc2Unorm,
        P::Bc3Unorm => T::Bc3Unorm,
        P::Bc4Unorm => T::Bc4Unorm,
        P::Bc4Snorm => T::Bc4Snorm,
        P::Bc5Unorm => T::Bc5Unorm,
        P::Bc5Snorm => T::Bc5Snorm,
        P::Bc7Unorm => T::Bc7Unorm,
        P::Bc6hUfloat => T::Bc6hUfloat,
        P::Bc6hSfloat => T::Bc6hSfloat,
        P::Astc4x4Unorm => T::Astc2d4x4Unorm,
        P::B8G8R8A8Unorm => T::B8g8r8a8Unorm,
        P::R32G32B32A32Float => T::R32g32b32a32Float,
        P::R32G32B32A32Sint => T::R32g32b32a32Sint,
        P::R32G32Float => T::R32g32Float,
        P::R32G32Sint => T::R32g32Sint,
        P::R32Float => T::R32Float,
        P::R16Float => T::R16Float,
        P::R16Unorm => T::R16Unorm,
        P::R16Snorm => T::R16Snorm,
        P::R16Uint => T::R16Uint,
        P::R16Sint => T::R16Sint,
        P::R16G16Unorm => T::R16g16Unorm,
        P::R16G16Float => T::R16g16Float,
        P::R16G16Uint => T::R16g16Uint,
        P::R16G16Sint => T::R16g16Sint,
        P::R16G16Snorm => T::R16g16Snorm,
        P::R32G32B32Float => T::R32g32b32Float,
        P::A8B8G8R8Srgb => T::A8b8g8r8Srgb,
        P::R8G8Unorm => T::R8g8Unorm,
        P::R8G8Snorm => T::R8g8Snorm,
        P::R8G8Sint => T::R8g8Sint,
        P::R8G8Uint => T::R8g8Uint,
        P::R32G32Uint => T::R32g32Uint,
        P::R16G16B16X16Float => T::R16g16b16x16Float,
        P::R32Uint => T::R32Uint,
        P::R32Sint => T::R32Sint,
        P::Astc8x8Unorm => T::Astc2d8x8Unorm,
        P::Astc8x5Unorm => T::Astc2d8x5Unorm,
        P::Astc5x4Unorm => T::Astc2d5x4Unorm,
        P::B8G8R8A8Srgb => T::B8g8r8a8Srgb,
        P::Bc1RgbaSrgb => T::Bc1RgbaSrgb,
        P::Bc2Srgb => T::Bc2Srgb,
        P::Bc3Srgb => T::Bc3Srgb,
        P::Bc7Srgb => T::Bc7Srgb,
        P::A4B4G4R4Unorm => T::A4b4g4r4Unorm,
        P::G4R4Unorm => T::G4r4Unorm,
        P::Astc4x4Srgb => T::Astc2d4x4Srgb,
        P::Astc8x8Srgb => T::Astc2d8x8Srgb,
        P::Astc8x5Srgb => T::Astc2d8x5Srgb,
        P::Astc5x4Srgb => T::Astc2d5x4Srgb,
        P::Astc5x5Unorm => T::Astc2d5x5Unorm,
        P::Astc5x5Srgb => T::Astc2d5x5Srgb,
        P::Astc10x8Unorm => T::Astc2d10x8Unorm,
        P::Astc10x8Srgb => T::Astc2d10x8Srgb,
        P::Astc6x6Unorm => T::Astc2d6x6Unorm,
        P::Astc6x6Srgb => T::Astc2d6x6Srgb,
        P::Astc10x6Unorm => T::Astc2d10x6Unorm,
        P::Astc10x6Srgb => T::Astc2d10x6Srgb,
        P::Astc10x5Unorm => T::Astc2d10x5Unorm,
        P::Astc10x5Srgb => T::Astc2d10x5Srgb,
        P::Astc10x10Unorm => T::Astc2d10x10Unorm,
        P::Astc10x10Srgb => T::Astc2d10x10Srgb,
        P::Astc12x10Unorm => T::Astc2d12x10Unorm,
        P::Astc12x10Srgb => T::Astc2d12x10Srgb,
        P::Astc12x12Unorm => T::Astc2d12x12Unorm,
        P::Astc12x12Srgb => T::Astc2d12x12Srgb,
        P::Astc8x6Unorm => T::Astc2d8x6Unorm,
        P::Astc8x6Srgb => T::Astc2d8x6Srgb,
        P::Astc6x5Unorm => T::Astc2d6x5Unorm,
        P::Astc6x5Srgb => T::Astc2d6x5Srgb,
        P::E5B9G9R9Float => T::E5b9g9r9Float,
        P::D32Float => T::D32Float,
        P::D16Unorm => T::D16Unorm,
        P::X8D24Unorm => T::X8D24Unorm,
        P::S8Uint => T::S8Uint,
        P::D24UnormS8Uint => T::D24UnormS8Uint,
        P::S8UintD24Unorm => T::S8UintD24Unorm,
        P::D32FloatS8Uint => T::D32FloatS8Uint,
        P::Etc2RgbUnorm => T::Etc2RgbUnorm,
        P::Etc2RgbSrgb => T::Etc2RgbSrgb,
        P::Etc2RgbaUnorm => T::Etc2RgbaUnorm,
        P::Etc2RgbaSrgb => T::Etc2RgbaSrgb,
        P::Etc2RgbPtaUnorm => T::Etc2RgbPtaUnorm,
        P::Etc2RgbPtaSrgb => T::Etc2RgbPtaSrgb,
        P::EacR11Unorm => T::EacR11Unorm,
        P::EacR11Snorm => T::EacR11Snorm,
        P::EacR11G11Unorm => T::EacR11G11Unorm,
        P::EacR11G11Snorm => T::EacR11G11Snorm,
    }
}

/// One guest program, as the translator reads it.
pub struct ProgramEnv<'a> {
    arch: ShaderArch,
    stage: ShaderStage,
    start: GpuVAddr,
    gmmu: &'a Gmmu,
    const_buffers: &'a [ConstBufferBinding],
    pool: TexturePool,
    viewport_transform: u32,
    texture_bound: u32,
    local_memory: u32,
    shared_memory: u32,
    workgroup: [u32; 3],
    /// Texture queries of the translation running on this environment.
    reads: RefCell<Vec<EnvRead>>,
    /// Lowest and one past the highest program byte the translator read.
    extent: Cell<(GpuVAddr, GpuVAddr)>,
}

impl<'a> ProgramEnv<'a> {
    /// Maxwell program `slot` (0 VertexA ... 5 Pixel) of the 3D engine state in `ctx`.
    pub fn graphics(ctx: &'a GraphicsContext<'_>, slot: usize, arch: ShaderArch) -> Self {
        let regs = ctx.regs;
        let stage = [
            ShaderStage::VertexA,
            ShaderStage::VertexB,
            ShaderStage::TessellationControl,
            ShaderStage::TessellationEval,
            ShaderStage::Geometry,
            ShaderStage::Fragment,
        ][slot];
        let start =
            regs.program_region.address() + u64::from(regs.pipelines[slot].offset) + HEADER_SIZE;
        let mut env = Self {
            arch,
            stage,
            start,
            gmmu: ctx.memory,
            const_buffers: &ctx.const_buffers[slot.saturating_sub(1)],
            pool: TexturePool {
                address: regs.tex_header.address(),
                limit: regs.tex_header.limit,
                via_header_index: regs.sampler_binding == SamplerBinding::VIA_HEADER_BINDING,
                switch2: arch == ShaderArch::Ampere,
            },
            viewport_transform: regs.viewport_scale_offset_enabled,
            texture_bound: regs.bindless_texture_const_buffer_slot,
            local_memory: 0,
            shared_memory: 0,
            workgroup: [1; 3],
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((start, start)),
        };
        let word = |index: u64| env.word(start - HEADER_SIZE + index * 4);
        let low = word(1) & 0xff_ffff;
        let high = word(2) & 0xff_ffff;
        let crs = word(3) & 0xff_ffff;
        env.local_memory = (low | high << 24).saturating_add(crs);
        env
    }

    /// The program a compute dispatch launches.
    pub fn compute(ctx: &'a ComputeContext<'_>) -> Self {
        let arch = match ctx.qmd_version {
            QmdVersion::V2 => ShaderArch::Maxwell,
            QmdVersion::V3 => ShaderArch::Ampere,
        };
        Self {
            arch,
            stage: ShaderStage::Compute,
            start: ctx.program,
            gmmu: ctx.memory,
            const_buffers: ctx.const_buffers,
            pool: TexturePool {
                address: ctx.texture_headers,
                limit: ctx.texture_header_limit,
                via_header_index: ctx.linked_samplers,
                switch2: arch == ShaderArch::Ampere,
            },
            viewport_transform: 0,
            texture_bound: 0,
            local_memory: ctx.local_memory_size,
            shared_memory: ctx.shared_memory_size,
            workgroup: ctx.block,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((ctx.program, ctx.program)),
        }
    }

    fn word(&self, address: GpuVAddr) -> u32 {
        self.gmmu.read::<u32>(address).unwrap_or(0)
    }

    /// GPU address of the first byte that identifies the program: its header for graphics
    /// stages, its first instruction for compute.
    pub fn code_start(&self) -> GpuVAddr {
        if self.stage == ShaderStage::Compute {
            self.start
        } else {
            self.start - HEADER_SIZE
        }
    }

    /// The range the last translation read, from [`Self::code_start`].
    pub fn code_range(&self) -> (GpuVAddr, u64) {
        let (_, end) = self.extent.get();
        let start = self.code_start();
        (start, end.max(self.start).saturating_sub(start))
    }

    pub fn gmmu(&self) -> &'a Gmmu {
        self.gmmu
    }

    pub fn pool(&self) -> &TexturePool {
        &self.pool
    }

    pub fn const_buffers(&self) -> &[ConstBufferBinding] {
        self.const_buffers
    }

    /// The register state that keys every program of this environment, besides its recorded
    /// reads.
    pub fn state_words(&self) -> [u32; 7] {
        let [x, y, z] = self.workgroup;
        [
            self.viewport_transform,
            self.texture_bound,
            self.shared_memory,
            self.local_memory,
            x,
            y,
            z,
        ]
    }

    /// Takes the queries of the last translation.
    pub fn take_reads(&self) -> Vec<EnvRead> {
        self.reads.take()
    }

    /// Whether every recorded query still gives its recorded answer.
    pub fn replay(&self, reads: &[EnvRead]) -> bool {
        reads.iter().all(|read| match *read {
            EnvRead::Type(handle, ty) => self.query_type(handle) == ty,
            EnvRead::PixelFormat(handle, format) => self.query_format(handle) == format,
            EnvRead::Integer(handle, integer) => self.query_integer(handle) == integer,
            EnvRead::Cbuf {
                index,
                offset,
                value,
            } => self.cbuf_word(index, offset) == value,
        })
    }

    fn query_type(&self, handle: u32) -> TextureType {
        texture_type(&self.pool.header(self.gmmu, handle))
    }

    fn query_format(&self, handle: u32) -> TexturePixelFormat {
        shader_pixel_format(self.pool.header(self.gmmu, handle).format)
    }

    fn query_integer(&self, handle: u32) -> bool {
        matches!(
            self.pool.header(self.gmmu, handle).format.numeric(),
            Numeric::Uint | Numeric::Sint
        )
    }

    fn cbuf_word(&self, index: u32, offset: u32) -> u32 {
        match self.const_buffers.get(index as usize) {
            Some(cbuf) if cbuf.enabled && offset < cbuf.size => {
                self.word(cbuf.address + u64::from(offset))
            }
            _ => 0,
        }
    }

    fn record(&self, read: EnvRead) {
        let mut reads = self.reads.borrow_mut();
        if !reads.contains(&read) {
            reads.push(read);
        }
    }
}

impl ShaderEnvironment for ProgramEnv<'_> {
    fn arch(&self) -> ShaderArch {
        self.arch
    }

    fn stage(&self) -> ShaderStage {
        self.stage
    }

    fn start_address(&self) -> GpuVAddr {
        self.start
    }

    fn read_u64(&self, address: GpuVAddr) -> u64 {
        let (low, high) = self.extent.get();
        self.extent
            .set((low.min(address), high.max(address.saturating_add(8))));
        self.gmmu.read::<u64>(address).unwrap_or(0)
    }

    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        let value = self.cbuf_word(index, offset);
        self.record(EnvRead::Cbuf {
            index,
            offset,
            value,
        });
        value
    }

    fn shared_memory_size(&self) -> u32 {
        self.shared_memory
    }

    fn workgroup_size(&self) -> [u32; 3] {
        self.workgroup
    }

    fn texture_type(&self, raw_handle: u32) -> TextureType {
        let ty = self.query_type(raw_handle);
        self.record(EnvRead::Type(raw_handle, ty));
        ty
    }

    fn texture_pixel_format(&self, raw_handle: u32) -> TexturePixelFormat {
        let format = self.query_format(raw_handle);
        self.record(EnvRead::PixelFormat(raw_handle, format));
        format
    }

    fn is_texture_pixel_format_integer(&self, raw_handle: u32) -> bool {
        let integer = self.query_integer(raw_handle);
        self.record(EnvRead::Integer(raw_handle, integer));
        integer
    }

    fn viewport_transform_state(&self) -> u32 {
        self.viewport_transform
    }

    fn texture_bound_buffer(&self) -> u32 {
        self.texture_bound
    }

    fn local_memory_size(&self) -> u32 {
        self.local_memory
    }

    fn has_hle_macro_state(&self) -> bool {
        false
    }
}
