//! What the shader translator reads from the guest for one program: its code through the
//! GPU address space, its constant buffers, the texture headers its handles select and the few
//! registers translation depends on.
//!
//! The environment records every texture query the translator makes; a cached module stays
//! valid while [`ProgramEnv::replay`] gives the same answers, so changing a texture's type or
//! format under an unchanged program translates it again.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use lichii_common::GpuVAddr;
use lichii_gpu::backend::{ComputeContext, ConstBufferBinding, GraphicsContext, QmdVersion};
use lichii_gpu::engines::maxwell_3d::Regs;
use lichii_gpu::engines::maxwell_3d::regs::{
    MESH_CONTROL, MESH_INIT_SHADER, MESH_SHADER_A, MESH_SHADER_B, SamplerBinding,
};
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::ampere::{GRAPHICS_HEADER_SIZE, graphics_header_address};
use lichii_shader::info::{HandleSource, TexturePixelFormat};
use lichii_shader::ir::TextureType;
use lichii_shader::maxwell::header::ProgramHeader;
use lichii_shader::{MeshState, ShaderArch, ShaderEnvironment, ShaderStage};
use lichii_texture::tic::{self, TextureDescriptor, TicEntry};
use lichii_texture::{Numeric, PixelFormat};

/// Size of the Maxwell program header in front of a graphics program's first instruction.
pub const HEADER_SIZE: u64 = 0x50;

/// Bytes the 3D and compute engines let one constant buffer binding cover.
const MAX_CONST_BUFFER_SIZE: u32 = 0x1_0000;

/// Bytes from a graphics program's header to its first instruction.
fn header_size(arch: ShaderArch) -> u64 {
    match arch {
        ShaderArch::Maxwell => HEADER_SIZE,
        ShaderArch::Ampere => GRAPHICS_HEADER_SIZE,
    }
}

/// GPU address of the header of graphics program `slot`: an offset into the program region
/// on Maxwell, a full address in the pipeline register's reserved words on NVN2.
fn graphics_header(regs: &Regs, slot: usize, arch: ShaderArch) -> GpuVAddr {
    let pipeline = &regs.pipelines[slot];
    match arch {
        ShaderArch::Maxwell => regs.program_region.address() + u64::from(pipeline.offset),
        ShaderArch::Ampere => graphics_header_address(pipeline.reserved[0], pipeline.reserved[1]),
    }
}

/// The header of graphics program `slot`, zero where its memory is unmapped.
pub(crate) fn graphics_program_header(
    ctx: &GraphicsContext<'_>,
    slot: usize,
    arch: ShaderArch,
) -> ProgramHeader {
    ProgramHeader {
        words: ctx
            .memory
            .read::<[u32; 20]>(graphics_header(ctx.regs, slot, arch))
            .unwrap_or_default(),
    }
}

impl EnvRead {
    /// The recorded answer as a number.
    pub fn answer(&self) -> u32 {
        match *self {
            EnvRead::Type(_, ty) | EnvRead::TypeOf(_, ty) => ty as u32,
            EnvRead::PixelFormat(_, format) | EnvRead::PixelFormatOf(_, format) => format as u32,
            EnvRead::Integer(_, integer) | EnvRead::IntegerOf(_, integer) => u32::from(integer),
            EnvRead::Cbuf { value, .. } => value,
            EnvRead::StoragePointer { mapped, .. }
            | EnvRead::IndirectStoragePointer { mapped, .. } => u32::from(mapped),
        }
    }

    /// Whether it asks if a storage pointer is mapped: the questions a failure keeps, since a
    /// pointer that was null when the program failed to translate may route once it is mapped.
    pub fn is_storage_pointer(&self) -> bool {
        matches!(
            self,
            EnvRead::StoragePointer { .. } | EnvRead::IndirectStoragePointer { .. }
        )
    }

    /// Whether both ask the same question, whatever their answers.
    pub fn same_question(&self, other: &EnvRead) -> bool {
        use EnvRead::*;
        match (*self, *other) {
            (Type(a, _), Type(b, _))
            | (PixelFormat(a, _), PixelFormat(b, _))
            | (Integer(a, _), Integer(b, _)) => a == b,
            (TypeOf(a, _), TypeOf(b, _))
            | (PixelFormatOf(a, _), PixelFormatOf(b, _))
            | (IntegerOf(a, _), IntegerOf(b, _)) => a == b,
            (
                Cbuf {
                    index: a,
                    offset: x,
                    ..
                },
                Cbuf {
                    index: b,
                    offset: y,
                    ..
                },
            ) => (a, x) == (b, y),
            (
                StoragePointer {
                    index: a,
                    offset: x,
                    ..
                },
                StoragePointer {
                    index: b,
                    offset: y,
                    ..
                },
            ) => (a, x) == (b, y),
            (
                IndirectStoragePointer {
                    index: a,
                    offset: x,
                    indirect: i,
                    ..
                },
                IndirectStoragePointer {
                    index: b,
                    offset: y,
                    indirect: j,
                    ..
                },
            ) => (a, x, i) == (b, y, j),
            _ => false,
        }
    }
}

/// A query the translator made of guest state outside the program bytes, and the answer it
/// got: texture header queries by the constant buffer words that hold the handle (`*Of`) or by
/// raw handle, constant buffer words read as values (indirect branch tables) by binding and
/// offset, and whether a Switch 2 program's constant buffer pointer (or the pointer table entry
/// it points to) is mapped. A query by words replays against the handle the words hold at the
/// time, so a different handle that selects the same kind of texture still matches. A pointer
/// query replays whether the pointer is mapped, not its value, so a pool that moves keeps its
/// translation while one that was null when translated gets a new one once it is mapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvRead {
    Type(u32, TextureType),
    PixelFormat(u32, TexturePixelFormat),
    Integer(u32, bool),
    TypeOf(HandleSource, TextureType),
    PixelFormatOf(HandleSource, TexturePixelFormat),
    IntegerOf(HandleSource, bool),
    Cbuf {
        index: u32,
        offset: u32,
        value: u32,
    },
    StoragePointer {
        index: u32,
        offset: u32,
        mapped: bool,
    },
    IndirectStoragePointer {
        index: u32,
        offset: u32,
        indirect: u32,
        mapped: bool,
    },
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
        self.entry(gmmu, raw)
            .and_then(|entry| entry.decode().ok())
            .unwrap_or(NULL_TEXTURE)
    }

    /// The entry a raw handle selects, `None` past the pool or where it is unmapped.
    fn entry(&self, gmmu: &Gmmu, raw: u32) -> Option<TicEntry> {
        let (index, _) = self.split(raw);
        if index > self.limit {
            return None;
        }
        gmmu.read::<TicEntry>(self.address + u64::from(index) * 32)
            .ok()
    }

    /// What the shader questions read of [`Self::header`].
    fn answers(&self, gmmu: &Gmmu, raw: u32) -> HeaderAnswers {
        header_answers(&self.header(gmmu, raw))
    }
}

fn header_answers(desc: &TextureDescriptor) -> HeaderAnswers {
    HeaderAnswers {
        ty: texture_type(desc),
        format: shader_pixel_format(desc.format),
        integer: matches!(desc.format.numeric(), Numeric::Uint | Numeric::Sint),
    }
}

/// What the shader questions read of the header `entry` decodes to.
fn entry_answers(entry: TicEntry) -> HeaderAnswers {
    header_answers(&entry.decode().unwrap_or(NULL_TEXTURE))
}

/// Slots of [`HeaderMemo`], a power of two.
const MEMO_SLOTS: usize = 256;

/// Header answers by the 32 bytes they were decoded from, kept across lookups, so a header the
/// guest left as it was answers with a compare instead of a decode. The answers depend on the
/// bytes alone, so a slot holds whichever entry last hashed to it.
struct HeaderMemo {
    slots: Box<[(TicEntry, HeaderAnswers); MEMO_SLOTS]>,
}

impl Default for HeaderMemo {
    fn default() -> Self {
        let empty = TicEntry::default();
        Self {
            slots: Box::new([(empty, entry_answers(empty)); MEMO_SLOTS]),
        }
    }
}

impl HeaderMemo {
    fn answers(&mut self, entry: TicEntry) -> HeaderAnswers {
        let hash = entry.0.iter().fold(0u32, |h, &w| {
            (h.rotate_left(5) ^ w).wrapping_mul(0x9e37_79b9)
        });
        let slot = &mut self.slots[(hash >> (32 - MEMO_SLOTS.trailing_zeros())) as usize];
        if slot.0 != entry {
            *slot = (entry, entry_answers(entry));
        }
        slot.1
    }
}

/// The answers to the texture questions about one handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeaderAnswers {
    ty: TextureType,
    format: TexturePixelFormat,
    integer: bool,
}

/// How a texture question names its texture: by raw handle, or by the constant buffer words
/// that hold the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextureRef {
    Handle(u32),
    Words(HandleSource),
}

/// Textures whose answers [`TextureAnswers`] keeps; questions about others read the guest.
const KNOWN_TEXTURES: usize = 16;

/// What [`ProgramEnv::answer`] found for the textures one environment's questions named, so
/// the type, format and integer questions about one texture read its handle and header once,
/// and the answers of the headers it decoded.
pub struct TextureAnswers {
    known: [(TextureRef, HeaderAnswers); KNOWN_TEXTURES],
    len: usize,
    headers: HeaderMemo,
}

impl Default for TextureAnswers {
    fn default() -> Self {
        let unused = HeaderAnswers {
            ty: TextureType::Color2D,
            format: TexturePixelFormat::A8b8g8r8Unorm,
            integer: false,
        };
        Self {
            known: [(TextureRef::Handle(0), unused); KNOWN_TEXTURES],
            len: 0,
            headers: HeaderMemo::default(),
        }
    }
}

impl TextureAnswers {
    /// Forgets the textures; the header answers stay, as they hold for the same bytes.
    pub fn clear(&mut self) {
        self.len = 0;
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

/// The texture type a shader reads the header `desc` as.
pub fn texture_type(desc: &TextureDescriptor) -> TextureType {
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

/// The task and mesh draw state of program `slot`, when mesh control is on (NVN2 only) and the
/// slot holds a task or mesh program. A mesh program's per-primitive outputs come from the header of the
/// pass-through geometry program bound beside it, whose address the geometry pipeline's
/// reserved words hold.
fn mesh_state(ctx: &GraphicsContext<'_>, slot: usize, arch: ShaderArch) -> Option<MeshState> {
    let regs = ctx.regs;
    let words = regs.words();
    let has_task = regs.pipelines[3].enable() != 0;
    let holds_mesh = slot == 1 || (slot == 3 && has_task);
    if arch != ShaderArch::Ampere || words[MESH_CONTROL] & 1 == 0 || !holds_mesh {
        return None;
    }
    let primitive_header = if (slot == 3 || !has_task) && regs.pipelines[4].enable() != 0 {
        graphics_program_header(ctx, 4, ShaderArch::Ampere)
    } else {
        ProgramHeader::default()
    };
    Some(MeshState {
        shader_a: words[MESH_SHADER_A],
        shader_b: words[MESH_SHADER_B],
        init_shader: words[MESH_INIT_SHADER],
        primitive_header,
    })
}

/// The register state and bindings of a [`ProgramEnv`], owned; [`ProgramEnv::from_snapshot`]
/// translates from it on another thread.
pub struct EnvSnapshot {
    arch: ShaderArch,
    stage: ShaderStage,
    start: GpuVAddr,
    gmmu: Arc<Gmmu>,
    const_buffers: Vec<ConstBufferBinding>,
    /// The bytes of each enabled constant buffer when the snapshot was taken: the guest
    /// rewrites them between draws, before a worker translates.
    const_data: Vec<Box<[u8]>>,
    pool: TexturePool,
    viewport_transform: u32,
    texture_bound: u32,
    local_memory: u32,
    shared_memory: u32,
    workgroup: [u32; 3],
    mesh: Option<MeshState>,
}

/// One guest program, as the translator reads it.
pub struct ProgramEnv<'a> {
    arch: ShaderArch,
    stage: ShaderStage,
    start: GpuVAddr,
    gmmu: &'a Arc<Gmmu>,
    const_buffers: &'a [ConstBufferBinding],
    /// Constant buffer contents captured by [`Self::snapshot`]; empty reads guest memory.
    const_data: &'a [Box<[u8]>],
    pool: TexturePool,
    viewport_transform: u32,
    texture_bound: u32,
    local_memory: u32,
    shared_memory: u32,
    workgroup: [u32; 3],
    mesh: Option<MeshState>,
    /// Texture queries of the translation running on this environment.
    reads: RefCell<Vec<EnvRead>>,
    /// Lowest and one past the highest program byte the translator read.
    extent: Cell<(GpuVAddr, GpuVAddr)>,
    /// Every program word the translator read, kept on a worker's environment, where the guest
    /// may rewrite the program while it translates.
    code_log: Option<RefCell<Vec<(GpuVAddr, u64)>>>,
}

impl<'a> ProgramEnv<'a> {
    /// Maxwell program `slot` (0 VertexA ... 5 Pixel) of the 3D engine state in `ctx`. While
    /// mesh control is on, the VertexB slot holds the task program (the mesh program without
    /// one) and the tessellation slot the mesh program.
    pub fn graphics(ctx: &'a GraphicsContext<'_>, slot: usize, arch: ShaderArch) -> Self {
        let regs = ctx.regs;
        let mesh = mesh_state(ctx, slot, arch);
        let stage = match mesh {
            Some(_) if slot == 1 && regs.pipelines[3].enable() != 0 => ShaderStage::Task,
            Some(_) => ShaderStage::Mesh,
            None => [
                ShaderStage::VertexA,
                ShaderStage::VertexB,
                ShaderStage::TessellationControl,
                ShaderStage::TessellationEval,
                ShaderStage::Geometry,
                ShaderStage::Fragment,
            ][slot],
        };
        let header = graphics_header(regs, slot, arch);
        let start = header + header_size(arch);
        let mut env = Self {
            arch,
            stage,
            start,
            gmmu: ctx.memory,
            const_buffers: &ctx.const_buffers[slot.saturating_sub(1)],
            const_data: &[],
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
            mesh,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((start, start)),
            code_log: None,
        };
        let word = |index: u64| env.word(header + index * 4);
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
            const_data: &[],
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
            mesh: None,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((ctx.program, ctx.program)),
            code_log: None,
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
            self.start - header_size(self.arch)
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

    /// Whether `bytes`, program memory from `start`, still hold every word the translator read
    /// from a snapshot. Always true off a snapshot, where nothing else runs during translation.
    pub fn consumed_unchanged(&self, start: GpuVAddr, bytes: &[u8]) -> bool {
        self.code_log.as_ref().is_none_or(|log| {
            log.borrow().iter().all(|&(address, value)| {
                let offset = address.wrapping_sub(start) as usize;
                bytes
                    .get(offset..offset + 8)
                    .is_some_and(|word| word == value.to_le_bytes())
            })
        })
    }

    /// The channel's address space, to share with another thread.
    pub fn memory(&self) -> &'a Arc<Gmmu> {
        self.gmmu
    }

    /// What this environment reads, owned, so another thread can translate the program.
    pub fn snapshot(&self) -> EnvSnapshot {
        EnvSnapshot {
            arch: self.arch,
            stage: self.stage,
            start: self.start,
            gmmu: self.gmmu.clone(),
            const_buffers: self.const_buffers.to_vec(),
            const_data: self
                .const_buffers
                .iter()
                .map(|cbuf| self.const_bytes(cbuf))
                .collect(),
            pool: self.pool,
            viewport_transform: self.viewport_transform,
            texture_bound: self.texture_bound,
            local_memory: self.local_memory,
            shared_memory: self.shared_memory,
            workgroup: self.workgroup,
            mesh: self.mesh,
        }
    }

    /// The environment `snapshot` was taken of, reading guest memory as it is now.
    pub fn from_snapshot(snapshot: &'a EnvSnapshot) -> Self {
        Self {
            arch: snapshot.arch,
            stage: snapshot.stage,
            start: snapshot.start,
            gmmu: &snapshot.gmmu,
            const_buffers: &snapshot.const_buffers,
            const_data: &snapshot.const_data,
            pool: snapshot.pool,
            viewport_transform: snapshot.viewport_transform,
            texture_bound: snapshot.texture_bound,
            local_memory: snapshot.local_memory,
            shared_memory: snapshot.shared_memory,
            workgroup: snapshot.workgroup,
            mesh: snapshot.mesh,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((snapshot.start, snapshot.start)),
            code_log: Some(RefCell::new(Vec::new())),
        }
    }

    pub fn pool(&self) -> &TexturePool {
        &self.pool
    }

    pub fn const_buffers(&self) -> &[ConstBufferBinding] {
        self.const_buffers
    }

    /// The register state that keys every program of this environment, besides its recorded
    /// reads.
    pub fn state_words(&self) -> ([u32; 7], ShaderStage, Option<MeshState>) {
        let [x, y, z] = self.workgroup;
        let words = [
            self.viewport_transform,
            self.texture_bound,
            self.shared_memory,
            self.local_memory,
            x,
            y,
            z,
        ];
        (words, self.stage, self.mesh)
    }

    /// Takes the queries of the last translation.
    pub fn take_reads(&self) -> Vec<EnvRead> {
        self.reads.take()
    }

    /// The answer `read`'s question gets now, in the form [`EnvRead::answer`] gives the
    /// recorded one. Records nothing. `textures` keeps what earlier questions about the same
    /// texture found; clear it before asking another environment or after guest memory changed.
    pub fn answer(&self, read: &EnvRead, textures: &mut TextureAnswers) -> u32 {
        let texture = match *read {
            EnvRead::Type(handle, _)
            | EnvRead::PixelFormat(handle, _)
            | EnvRead::Integer(handle, _) => TextureRef::Handle(handle),
            EnvRead::TypeOf(source, _)
            | EnvRead::PixelFormatOf(source, _)
            | EnvRead::IntegerOf(source, _) => TextureRef::Words(source),
            EnvRead::Cbuf { index, offset, .. } => return self.cbuf_word(index, offset),
            EnvRead::StoragePointer { index, offset, .. } => {
                return u32::from(self.mapped_pointer(index, offset).is_some());
            }
            EnvRead::IndirectStoragePointer {
                index,
                offset,
                indirect,
                ..
            } => return u32::from(self.mapped_table_entry(index, offset, indirect)),
        };
        let known = textures.known[..textures.len]
            .iter()
            .find(|(t, _)| *t == texture);
        let answers = match known {
            Some(&(_, answers)) => answers,
            None => {
                let handle = match texture {
                    TextureRef::Handle(handle) => handle,
                    TextureRef::Words(source) => self.handle(source),
                };
                let answers = match self.pool.entry(self.gmmu, handle) {
                    Some(entry) => textures.headers.answers(entry),
                    None => header_answers(&NULL_TEXTURE),
                };
                if let Some(slot) = textures.known.get_mut(textures.len) {
                    *slot = (texture, answers);
                    textures.len += 1;
                }
                answers
            }
        };
        match *read {
            EnvRead::Type(..) | EnvRead::TypeOf(..) => answers.ty as u32,
            EnvRead::PixelFormat(..) | EnvRead::PixelFormatOf(..) => answers.format as u32,
            _ => u32::from(answers.integer),
        }
    }

    /// The raw handle the constant buffer words of `source` hold now, without recording them.
    fn handle(&self, source: HandleSource) -> u32 {
        source.handle(|index, offset| self.cbuf_word(index, offset))
    }

    fn query_type(&self, handle: u32) -> TextureType {
        self.pool.answers(self.gmmu, handle).ty
    }

    fn query_format(&self, handle: u32) -> TexturePixelFormat {
        self.pool.answers(self.gmmu, handle).format
    }

    fn query_integer(&self, handle: u32) -> bool {
        self.pool.answers(self.gmmu, handle).integer
    }

    fn cbuf_word(&self, index: u32, offset: u32) -> u32 {
        match self.const_buffers.get(index as usize) {
            Some(cbuf) if cbuf.enabled && offset < cbuf.size => {
                match self.const_data.get(index as usize) {
                    Some(bytes) => {
                        let offset = offset as usize;
                        bytes
                            .get(offset..offset + 4)
                            .map_or(0, |word| u32::from_le_bytes(word.try_into().unwrap()))
                    }
                    None => self.word(cbuf.address + u64::from(offset)),
                }
            }
            _ => 0,
        }
    }

    /// The address the 8-byte pointer at constant buffer `index`, `offset` holds, when it is
    /// mapped.
    fn mapped_pointer(&self, index: u32, offset: u32) -> Option<GpuVAddr> {
        let low = u64::from(self.cbuf_word(index, offset));
        let address = low | u64::from(self.cbuf_word(index, offset + 4)) << 32;
        (address != 0 && self.gmmu.translate(address).is_some()).then_some(address)
    }

    /// Whether the pointer `indirect` bytes past the address the constant buffer pointer at
    /// `index`, `offset` holds is mapped.
    fn mapped_table_entry(&self, index: u32, offset: u32, indirect: u32) -> bool {
        self.mapped_pointer(index, offset)
            .and_then(|table| self.gmmu.read::<u64>(table + u64::from(indirect)).ok())
            .is_some_and(|address| address != 0 && self.gmmu.translate(address).is_some())
    }

    /// The bytes `cbuf_word` can read from `cbuf`: its size and the word that starts at its
    /// last byte.
    fn const_bytes(&self, cbuf: &ConstBufferBinding) -> Box<[u8]> {
        if !cbuf.enabled {
            return Box::default();
        }
        let mut bytes = vec![0; cbuf.size.min(MAX_CONST_BUFFER_SIZE) as usize + 3];
        let _unmapped_bytes_read_zero = self.gmmu.read_block(cbuf.address, &mut bytes);
        bytes.into_boxed_slice()
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
        let value = self.gmmu.read::<u64>(address).unwrap_or(0);
        if let Some(log) = &self.code_log {
            log.borrow_mut().push((address, value));
        }
        value
    }

    fn read_code(&self, address: GpuVAddr, out: &mut [u8]) {
        let (low, high) = self.extent.get();
        let end = address.saturating_add(out.len() as u64);
        self.extent.set((low.min(address), high.max(end)));
        if self.gmmu.read_block(address, out).is_err() {
            for (i, word) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let value = self.gmmu.read::<u64>(address + i as u64 * 8).unwrap_or(0);
                word.copy_from_slice(&value.to_le_bytes());
            }
        }
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

    fn texture_type_of(&self, source: HandleSource) -> TextureType {
        let ty = self.query_type(self.handle(source));
        self.record(EnvRead::TypeOf(source, ty));
        ty
    }

    fn texture_pixel_format_of(&self, source: HandleSource) -> TexturePixelFormat {
        let format = self.query_format(self.handle(source));
        self.record(EnvRead::PixelFormatOf(source, format));
        format
    }

    fn is_texture_pixel_format_integer_of(&self, source: HandleSource) -> bool {
        let integer = self.query_integer(self.handle(source));
        self.record(EnvRead::IntegerOf(source, integer));
        integer
    }

    fn is_texture_multisample_of(&self, _: HandleSource) -> bool {
        false
    }

    fn mesh_state(&self) -> Option<MeshState> {
        self.mesh
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

    fn is_native_program(&self) -> bool {
        self.arch == ShaderArch::Ampere
    }

    fn is_cbuf_storage_buffer_pointer(&self, index: u32, offset: u32) -> bool {
        let mapped = self.mapped_pointer(index, offset).is_some();
        self.record(EnvRead::StoragePointer {
            index,
            offset,
            mapped,
        });
        mapped
    }

    fn is_cbuf_indirect_storage_buffer_pointer(
        &self,
        index: u32,
        offset: u32,
        indirect: u32,
    ) -> bool {
        let mapped = self.mapped_table_entry(index, offset, indirect);
        self.record(EnvRead::IndirectStoragePointer {
            index,
            offset,
            indirect,
            mapped,
        });
        mapped
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use lichii_gpu::gmmu::BigPageSize;
    use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

    use super::*;

    const SIZE: u64 = 2 << 20;
    const DADDR: u64 = 0x1_0000_0000;
    const GPU_VA: GpuVAddr = 0x4_0000_0000;
    const CBUF: GpuVAddr = GPU_VA + 0x1000;

    #[test]
    fn a_snapshot_reads_constant_buffers_as_the_draw_saw_them() {
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, DADDR, SIZE, 0, false).unwrap();
        gmmu.write(CBUF + 8, 0x1234_u32).unwrap();
        let const_buffers = [ConstBufferBinding {
            address: CBUF,
            size: 0x10,
            enabled: true,
        }];
        let draw = ProgramEnv {
            arch: ShaderArch::Maxwell,
            stage: ShaderStage::Fragment,
            start: GPU_VA,
            gmmu: &gmmu,
            const_buffers: &const_buffers,
            const_data: &[],
            pool: TexturePool {
                address: 0,
                limit: 0,
                via_header_index: false,
                switch2: false,
            },
            viewport_transform: 0,
            texture_bound: 0,
            local_memory: 0,
            shared_memory: 0,
            workgroup: [1; 3],
            mesh: None,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((GPU_VA, GPU_VA)),
            code_log: None,
        };
        let snapshot = draw.snapshot();
        gmmu.write(CBUF + 8, 0x5678_u32).unwrap();
        let worker = ProgramEnv::from_snapshot(&snapshot);
        assert_eq!(worker.read_cbuf(0, 8), 0x1234);
        assert_eq!(worker.read_cbuf(0, 0x10), 0, "past the binding");
        assert_eq!(draw.read_cbuf(0, 8), 0x5678);
    }

    fn env<'a>(
        gmmu: &'a Arc<Gmmu>,
        const_buffers: &'a [ConstBufferBinding],
        pool: TexturePool,
    ) -> ProgramEnv<'a> {
        ProgramEnv {
            arch: ShaderArch::Maxwell,
            stage: ShaderStage::Fragment,
            start: GPU_VA,
            gmmu,
            const_buffers,
            const_data: &[],
            pool,
            viewport_transform: 0,
            texture_bound: 0,
            local_memory: 0,
            shared_memory: 0,
            workgroup: [1; 3],
            mesh: None,
            reads: RefCell::new(Vec::new()),
            extent: Cell::new((GPU_VA, GPU_VA)),
            code_log: None,
        }
    }

    /// Packs a header from its format, one component type for all four, header version,
    /// texture type, normalized coordinates, multisample mode and red swizzle source.
    fn tic(
        format: u32,
        component: u32,
        version: u32,
        ty: u32,
        normalized: bool,
        msaa: u32,
        red: u32,
    ) -> TicEntry {
        let mut w = [0; 8];
        w[0] = format | (component * 0o1111) << 7 | (0o5430 | red) << 19;
        w[2] = version << 21;
        w[4] = ty << 23;
        w[5] = u32::from(normalized) << 31;
        w[7] = msaa << 8;
        TicEntry(w)
    }

    #[test]
    fn answers_agree_with_the_decoded_headers() {
        const MAPPED: u32 = 24;
        const LIMIT: u32 = 40;
        const TIC: GpuVAddr = GPU_VA + SIZE - MAPPED as u64 * 32;
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, DADDR, SIZE, 0, false).unwrap();
        let headers = [
            tic(0x08, 2, 3, 1, true, 0, 2),
            tic(0x08, 4, 3, 1, true, 0, 2),
            tic(0x08, 3, 2, 7, false, 0, 2),
            tic(0x08, 2, 3, 1, false, 0, 2),
            tic(0x24, 2, 3, 3, true, 0, 2),
            tic(0x0f, 7, 3, 5, true, 2, 2),
            tic(0x1d, 4, 0, 6, true, 0, 2),
            tic(0x2f, 7, 3, 1, true, 0, 2),
            tic(0x17, 2, 4, 2, true, 0, 2),
            tic(0x08, 6, 1, 4, true, 0, 2),
            tic(0x0c, 3, 3, 8, true, 0, 2),
            tic(0x08, 2, 3, 0, true, 0, 2),
            tic(0x08, 2, 3, 9, true, 0, 2),
            tic(0x08, 2, 5, 1, true, 0, 2),
            tic(0x08, 2, 3, 1, true, 7, 2),
            tic(0x08, 2, 3, 1, true, 0, 1),
            tic(0x43, 2, 3, 1, true, 0, 2),
            tic(0x08, 0, 3, 1, true, 0, 2),
            TicEntry::default(),
        ];
        for i in 0..MAPPED {
            gmmu.write(TIC + u64::from(i) * 32, headers[i as usize % headers.len()])
                .unwrap();
        }
        let handles: Vec<u32> = (0..48)
            .flat_map(|i| [i, i | 5 << 20, i * 32, 0xf_ffff])
            .collect();
        for (i, &handle) in handles.iter().enumerate() {
            gmmu.write(CBUF + i as u64 * 8, handle & 0xf_ffff).unwrap();
            gmmu.write(CBUF + i as u64 * 8 + 4, handle & !0xf_ffff)
                .unwrap();
        }
        let const_buffers = [ConstBufferBinding {
            address: CBUF,
            size: 0x1000,
            enabled: true,
        }];
        let source = |i: usize| HandleSource {
            cbuf_index: 0,
            cbuf_offset: i as u32 * 8,
            shift_left: 0,
            secondary: (i % 2 == 1).then_some((0, i as u32 * 8 + 4, 0)),
        };
        let ty = TextureType::Color2D;
        let format = TexturePixelFormat::A8b8g8r8Unorm;
        let mut textures = TextureAnswers::default();
        for (switch2, via_header_index) in [(false, false), (true, true), (true, false)] {
            let pool = TexturePool {
                address: TIC,
                limit: LIMIT,
                via_header_index,
                switch2,
            };
            let env = env(&gmmu, &const_buffers, pool);
            let expected = |handle: u32| {
                let desc = pool.header(&gmmu, handle);
                [
                    texture_type(&desc) as u32,
                    shader_pixel_format(desc.format) as u32,
                    u32::from(matches!(
                        desc.format.numeric(),
                        Numeric::Uint | Numeric::Sint
                    )),
                ]
            };
            textures.clear();
            let mut seen = HashSet::new();
            for _ in 0..2 {
                for (i, &handle) in handles.iter().enumerate() {
                    let words = if i % 2 == 1 {
                        handle
                    } else {
                        handle & 0xf_ffff
                    };
                    seen.insert(expected(handle));
                    let questions = [
                        (EnvRead::Type(handle, ty), expected(handle)[0]),
                        (EnvRead::PixelFormat(handle, format), expected(handle)[1]),
                        (EnvRead::Integer(handle, false), expected(handle)[2]),
                        (EnvRead::TypeOf(source(i), ty), expected(words)[0]),
                        (
                            EnvRead::PixelFormatOf(source(i), format),
                            expected(words)[1],
                        ),
                        (EnvRead::IntegerOf(source(i), false), expected(words)[2]),
                    ];
                    for (read, answer) in questions {
                        assert_eq!(env.answer(&read, &mut textures), answer, "{read:?}");
                    }
                    assert_eq!(
                        [
                            env.texture_type(handle) as u32,
                            env.texture_pixel_format(handle) as u32,
                            u32::from(env.is_texture_pixel_format_integer(handle)),
                        ],
                        expected(handle)
                    );
                }
            }
            assert!(seen.len() >= 8, "{} kinds of answers", seen.len());
        }
        let distinct = handles.iter().collect::<HashSet<_>>().len();
        assert!(
            distinct > KNOWN_TEXTURES,
            "questions past what the answers keep"
        );

        let pool = TexturePool {
            address: TIC,
            limit: LIMIT,
            via_header_index: false,
            switch2: false,
        };
        let env = env(&gmmu, &const_buffers, pool);
        textures.clear();
        let read = EnvRead::Integer(0, false);
        assert_eq!(env.answer(&read, &mut textures), 0);
        gmmu.write(TIC, headers[1]).unwrap();
        assert_eq!(env.answer(&read, &mut textures), 0, "kept until cleared");
        textures.clear();
        assert_eq!(env.answer(&read, &mut textures), 1);
    }

    /// Over a random stream of header rewrites, answers kept across lookups match a fresh decode
    /// of every header, for more distinct headers than the kept answers have slots.
    #[test]
    fn kept_header_answers_match_a_fresh_decode_over_random_rewrites() {
        const LIMIT: u32 = 63;
        const TIC: GpuVAddr = GPU_VA + 0x1_0000;
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, DADDR, SIZE, 0, false).unwrap();
        let const_buffers = [ConstBufferBinding {
            address: CBUF,
            size: 0x100,
            enabled: true,
        }];
        let pool = TexturePool {
            address: TIC,
            limit: LIMIT,
            via_header_index: false,
            switch2: true,
        };
        let env = env(&gmmu, &const_buffers, pool);
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        fn random_header(next: &mut dyn FnMut() -> u64) -> TicEntry {
            let pick = |n: u64, next: &mut dyn FnMut() -> u64| (next() % n) as u32;
            let mut entry = tic(
                [0x08, 0x24, 0x0f, 0x1d, 0x17, 0x2f, 0x43][pick(7, next) as usize],
                pick(8, next),
                pick(6, next),
                pick(10, next),
                pick(2, next) == 1,
                pick(8, next),
                2,
            );
            entry.0[1] = next() as u32;
            if pick(8, next) == 0 {
                entry.0 = std::array::from_fn(|_| next() as u32);
            }
            entry
        }
        let mut textures = TextureAnswers::default();
        let mut seen = HashSet::new();
        let ty = TextureType::Color2D;
        let format = TexturePixelFormat::A8b8g8r8Unorm;
        for _ in 0..20_000 {
            let index = (next() % u64::from(LIMIT + 1)) as u32;
            let entry = random_header(&mut next);
            seen.insert(entry);
            gmmu.write(TIC + u64::from(index) * 32, entry).unwrap();
            textures.clear();
            for handle in [index, (next() % u64::from(LIMIT + 8)) as u32] {
                gmmu.write(CBUF, handle).unwrap();
                let desc = pool.header(&gmmu, handle);
                let expected = [
                    texture_type(&desc) as u32,
                    shader_pixel_format(desc.format) as u32,
                    u32::from(matches!(
                        desc.format.numeric(),
                        Numeric::Uint | Numeric::Sint
                    )),
                ];
                let source = HandleSource {
                    cbuf_index: 0,
                    cbuf_offset: 0,
                    shift_left: 0,
                    secondary: None,
                };
                textures.clear();
                let asked = [
                    env.answer(&EnvRead::Type(handle, ty), &mut textures),
                    env.answer(&EnvRead::PixelFormat(handle, format), &mut textures),
                    env.answer(&EnvRead::Integer(handle, false), &mut textures),
                ];
                assert_eq!(asked, expected, "handle {handle}");
                textures.clear();
                let asked = [
                    env.answer(&EnvRead::TypeOf(source, ty), &mut textures),
                    env.answer(&EnvRead::PixelFormatOf(source, format), &mut textures),
                    env.answer(&EnvRead::IntegerOf(source, false), &mut textures),
                ];
                assert_eq!(
                    asked, expected,
                    "handle {handle} through the constant buffer"
                );
            }
        }
        assert!(seen.len() > 16 * MEMO_SLOTS, "{} headers", seen.len());
    }

    #[test]
    fn switch2_programs_classify_pointers_by_mapping_and_replay_it() {
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, DADDR, SIZE, 0, false).unwrap();
        let table = GPU_VA + 0x2000;
        let unmapped: GpuVAddr = 0x9_0000_0000;
        gmmu.write(CBUF + 0x10, table).unwrap();
        gmmu.write(CBUF + 0x20, unmapped).unwrap();
        gmmu.write(table + 8, GPU_VA + 0x3000).unwrap();
        gmmu.write(table + 0x10, unmapped).unwrap();
        let const_buffers = [ConstBufferBinding {
            address: CBUF,
            size: 0x100,
            enabled: true,
        }];
        let pool = TexturePool {
            address: 0,
            limit: 0,
            via_header_index: false,
            switch2: true,
        };
        let mut env = env(&gmmu, &const_buffers, pool);
        assert!(!env.is_native_program());
        env.arch = ShaderArch::Ampere;
        assert!(env.is_native_program());

        assert!(env.is_cbuf_storage_buffer_pointer(0, 0x10));
        assert!(!env.is_cbuf_storage_buffer_pointer(0, 0x18), "null");
        assert!(!env.is_cbuf_storage_buffer_pointer(0, 0x20), "unmapped");
        assert!(env.is_cbuf_indirect_storage_buffer_pointer(0, 0x10, 8));
        assert!(!env.is_cbuf_indirect_storage_buffer_pointer(0, 0x10, 0x10));
        assert!(!env.is_cbuf_indirect_storage_buffer_pointer(0, 0x18, 8));
        let reads = env.take_reads();
        assert_eq!(reads.len(), 6, "{reads:?}");
        assert!(
            !reads.iter().any(|r| matches!(r, EnvRead::Cbuf { .. })),
            "pointer values are not variant questions"
        );

        let mut textures = TextureAnswers::default();
        let replay = |env: &ProgramEnv<'_>, textures: &mut TextureAnswers| {
            reads
                .iter()
                .map(|read| env.answer(read, textures))
                .collect::<Vec<_>>()
        };
        let recorded = reads.iter().map(EnvRead::answer).collect::<Vec<_>>();
        assert_eq!(replay(&env, &mut textures), recorded);

        gmmu.write(CBUF + 0x10, GPU_VA + 0x4000).unwrap();
        gmmu.write(GPU_VA + 0x4008, GPU_VA + 0x5000).unwrap();
        assert_eq!(
            replay(&env, &mut textures),
            recorded,
            "a pool that moves keeps its translation"
        );

        gmmu.write(CBUF + 0x18, GPU_VA + 0x6000).unwrap();
        let null_read = EnvRead::StoragePointer {
            index: 0,
            offset: 0x18,
            mapped: false,
        };
        assert!(reads.contains(&null_read));
        assert_eq!(
            env.answer(&null_read, &mut textures),
            1,
            "a pointer that was null at translation asks for a new one once mapped"
        );
    }
}
