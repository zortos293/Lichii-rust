//! What one translation asked its environment, with the answers, and replaying it.

use std::cell::{Cell, RefCell};

use lichii_common::GpuVAddr;
use serde::{Deserialize, Serialize};

use crate::info::{HandleSource, ReplaceConstant, TexturePixelFormat};
use crate::ir::TextureType;
use crate::{MeshState, ShaderArch, ShaderEnvironment, ShaderStage};

/// One question translation asked and the answer it got, other than program memory, which
/// [`EnvRecord`] keeps as one hash. Translation may embed the program's address itself
/// ([`Query::StartAddress`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Query {
    Arch(ShaderArch),
    Stage(ShaderStage),
    StartAlignment(u64),
    Cbuf {
        index: u32,
        offset: u32,
        value: u32,
    },
    SharedMemorySize(u32),
    WorkgroupSize([u32; 3]),
    TextureType {
        handle: u32,
        ty: TextureType,
    },
    PixelFormat {
        handle: u32,
        format: TexturePixelFormat,
    },
    IntegerFormat {
        handle: u32,
        integer: bool,
    },
    Multisample {
        handle: u32,
        multisample: bool,
    },
    TextureTypeOf {
        source: HandleSource,
        ty: TextureType,
    },
    PixelFormatOf {
        source: HandleSource,
        format: TexturePixelFormat,
    },
    IntegerFormatOf {
        source: HandleSource,
        integer: bool,
    },
    MultisampleOf {
        source: HandleSource,
        multisample: bool,
    },
    ViewportTransform(u32),
    TextureBoundBuffer(u32),
    LocalMemorySize(u32),
    HleMacroState(bool),
    StoragePointer {
        index: u32,
        offset: u32,
        answer: bool,
    },
    NativeProgram(bool),
    IndirectStoragePointer {
        index: u32,
        offset: u32,
        indirect: u32,
        answer: bool,
    },
    PassthroughMask([u32; 8]),
    ReplaceConstBuffer {
        bank: u32,
        offset: u32,
        answer: Option<ReplaceConstant>,
    },
    ProprietaryDriver(bool),
    Mesh(Option<MeshState>),
    StartAddress(GpuVAddr),
}

impl Query {
    /// Whether `env` gives the recorded answer to this question.
    fn holds(&self, env: &dyn ShaderEnvironment) -> bool {
        match *self {
            Query::Arch(arch) => env.arch() == arch,
            Query::Stage(stage) => env.stage() == stage,
            Query::StartAlignment(alignment) => env.start_address() % 32 == alignment,
            Query::Cbuf {
                index,
                offset,
                value,
            } => env.read_cbuf(index, offset) == value,
            Query::SharedMemorySize(size) => env.shared_memory_size() == size,
            Query::WorkgroupSize(size) => env.workgroup_size() == size,
            Query::TextureType { handle, ty } => env.texture_type(handle) == ty,
            Query::PixelFormat { handle, format } => env.texture_pixel_format(handle) == format,
            Query::IntegerFormat { handle, integer } => {
                env.is_texture_pixel_format_integer(handle) == integer
            }
            Query::Multisample {
                handle,
                multisample,
            } => env.is_texture_multisample(handle) == multisample,
            Query::TextureTypeOf { source, ty } => env.texture_type_of(source) == ty,
            Query::PixelFormatOf { source, format } => {
                env.texture_pixel_format_of(source) == format
            }
            Query::IntegerFormatOf { source, integer } => {
                env.is_texture_pixel_format_integer_of(source) == integer
            }
            Query::MultisampleOf {
                source,
                multisample,
            } => env.is_texture_multisample_of(source) == multisample,
            Query::ViewportTransform(state) => env.viewport_transform_state() == state,
            Query::TextureBoundBuffer(buffer) => env.texture_bound_buffer() == buffer,
            Query::LocalMemorySize(size) => env.local_memory_size() == size,
            Query::HleMacroState(state) => env.has_hle_macro_state() == state,
            Query::StoragePointer {
                index,
                offset,
                answer,
            } => env.is_cbuf_storage_buffer_pointer(index, offset) == answer,
            Query::NativeProgram(native) => env.is_native_program() == native,
            Query::IndirectStoragePointer {
                index,
                offset,
                indirect,
                answer,
            } => env.is_cbuf_indirect_storage_buffer_pointer(index, offset, indirect) == answer,
            Query::PassthroughMask(mask) => env.gp_passthrough_mask() == mask,
            Query::ReplaceConstBuffer {
                bank,
                offset,
                answer,
            } => env.replace_const_buffer(bank, offset) == answer,
            Query::ProprietaryDriver(proprietary) => env.is_proprietary_driver() == proprietary,
            Query::Mesh(state) => env.mesh_state() == state,
            Query::StartAddress(address) => env.start_address() == address,
        }
    }
}

/// The program bytes translation read and their hash. Addressed relative to the program's
/// first instruction, so the same code elsewhere gives the same record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Code {
    /// The first byte's offset from the first instruction, negative for the header.
    offset: i64,
    /// Up to the end of the last word read; 0 when translation read none.
    len: u64,
    hash: u64,
}

/// Every question one program's translation asked, in order, with its answers. Translation is
/// deterministic, so an environment that gives the same answers in the same order translates
/// to the same module.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct EnvRecord {
    queries: Vec<Query>,
    code: Code,
}

impl EnvRecord {
    /// Whether `env` answers every question as recorded, program memory too unless `code` is
    /// `None`. `side` tells VertexA (0) from the program (1) in `code`.
    pub(super) fn matches(
        &self,
        env: &dyn ShaderEnvironment,
        side: usize,
        code: Option<&mut CodeCheck>,
    ) -> bool {
        let Code { offset, len, hash } = self.code;
        let code_holds = code.is_none_or(|check| {
            len == 0
                || check.hash(
                    env,
                    side,
                    env.start_address().wrapping_add_signed(offset),
                    len,
                ) == hash
        });
        code_holds && self.queries.iter().all(|query| query.holds(env))
    }

    /// The program bytes translation read: the first one's offset from the program's first
    /// instruction (negative for its header) and the length up to the last one read.
    pub(super) fn code_range(&self) -> (i64, u64) {
        (self.code.offset, self.code.len)
    }
}

/// Hashes of live program memory, each range hashed once while one lookup tries several
/// entries. Clear it before a lookup on other environments.
#[derive(Debug, Default)]
pub struct CodeCheck {
    /// By side, address and length.
    hashes: Vec<(usize, GpuVAddr, u64, u64)>,
    bytes: Vec<u8>,
}

impl CodeCheck {
    /// Forgets every hash. Call it before a lookup on other environments or after program
    /// memory may have changed, or a stale hash would match the old bytes.
    pub fn clear(&mut self) {
        self.hashes.clear();
    }

    fn hash(
        &mut self,
        env: &dyn ShaderEnvironment,
        side: usize,
        address: GpuVAddr,
        len: u64,
    ) -> u64 {
        let known = self
            .hashes
            .iter()
            .find(|&&(s, a, l, _)| (s, a, l) == (side, address, len));
        if let Some(&(.., hash)) = known {
            return hash;
        }
        let hash = hash_code(env, address, len, &mut self.bytes);
        self.hashes.push((side, address, len, hash));
        hash
    }
}

/// A stable hash of the `len` bytes of program memory at `address`, one multiply per word.
fn hash_code(env: &dyn ShaderEnvironment, address: GpuVAddr, len: u64, bytes: &mut Vec<u8>) -> u64 {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    bytes.resize(len as usize, 0);
    env.read_code(address, bytes);
    let mut hash = len.wrapping_mul(K);
    for &word in bytes.as_chunks::<8>().0 {
        hash = (hash.rotate_left(5) ^ u64::from_le_bytes(word)).wrapping_mul(K);
    }
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xff51_afd7_ed55_8ccd);
    hash ^ (hash >> 33)
}

/// Forwards every query to `env` and records the questions and answers.
pub(super) struct Logger<'a> {
    env: &'a dyn ShaderEnvironment,
    start: GpuVAddr,
    log: RefCell<Vec<Query>>,
    /// The offsets of the first and last program word read.
    code: Cell<Option<(i64, i64)>>,
}

impl<'a> Logger<'a> {
    pub(super) fn new(env: &'a dyn ShaderEnvironment) -> Logger<'a> {
        Logger {
            env,
            start: env.start_address(),
            log: RefCell::new(Vec::new()),
            code: Cell::new(None),
        }
    }

    /// Pins the record to the program's address, for a module that holds it.
    pub(super) fn pin_start_address(&self) {
        self.log(Query::StartAddress(self.start));
    }

    pub(super) fn into_record(self) -> EnvRecord {
        let code = self.code.get().map_or(Code::default(), |(first, last)| {
            let len = (last - first) as u64 + 8;
            let address = self.start.wrapping_add_signed(first);
            Code {
                offset: first,
                len,
                hash: hash_code(self.env, address, len, &mut Vec::new()),
            }
        });
        EnvRecord {
            queries: self.log.into_inner(),
            code,
        }
    }

    fn log(&self, query: Query) {
        self.log.borrow_mut().push(query);
    }
}

impl ShaderEnvironment for Logger<'_> {
    fn arch(&self) -> ShaderArch {
        let arch = self.env.arch();
        self.log(Query::Arch(arch));
        arch
    }
    fn stage(&self) -> ShaderStage {
        let stage = self.env.stage();
        self.log(Query::Stage(stage));
        stage
    }
    fn start_address(&self) -> GpuVAddr {
        self.log(Query::StartAlignment(self.start % 32));
        self.start
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        let offset = address.wrapping_sub(self.start) as i64;
        let range = self.code.get().map_or((offset, offset), |(first, last)| {
            (first.min(offset), last.max(offset))
        });
        self.code.set(Some(range));
        self.env.read_u64(address)
    }
    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        let value = self.env.read_cbuf(index, offset);
        self.log(Query::Cbuf {
            index,
            offset,
            value,
        });
        value
    }
    fn shared_memory_size(&self) -> u32 {
        let size = self.env.shared_memory_size();
        self.log(Query::SharedMemorySize(size));
        size
    }
    fn workgroup_size(&self) -> [u32; 3] {
        let size = self.env.workgroup_size();
        self.log(Query::WorkgroupSize(size));
        size
    }
    fn texture_type(&self, handle: u32) -> TextureType {
        let ty = self.env.texture_type(handle);
        self.log(Query::TextureType { handle, ty });
        ty
    }
    fn texture_pixel_format(&self, handle: u32) -> TexturePixelFormat {
        let format = self.env.texture_pixel_format(handle);
        self.log(Query::PixelFormat { handle, format });
        format
    }
    fn is_texture_pixel_format_integer(&self, handle: u32) -> bool {
        let integer = self.env.is_texture_pixel_format_integer(handle);
        self.log(Query::IntegerFormat { handle, integer });
        integer
    }
    fn is_texture_multisample(&self, handle: u32) -> bool {
        let multisample = self.env.is_texture_multisample(handle);
        self.log(Query::Multisample {
            handle,
            multisample,
        });
        multisample
    }
    fn texture_type_of(&self, source: HandleSource) -> TextureType {
        let ty = self.env.texture_type_of(source);
        self.log(Query::TextureTypeOf { source, ty });
        ty
    }
    fn texture_pixel_format_of(&self, source: HandleSource) -> TexturePixelFormat {
        let format = self.env.texture_pixel_format_of(source);
        self.log(Query::PixelFormatOf { source, format });
        format
    }
    fn is_texture_pixel_format_integer_of(&self, source: HandleSource) -> bool {
        let integer = self.env.is_texture_pixel_format_integer_of(source);
        self.log(Query::IntegerFormatOf { source, integer });
        integer
    }
    fn is_texture_multisample_of(&self, source: HandleSource) -> bool {
        let multisample = self.env.is_texture_multisample_of(source);
        self.log(Query::MultisampleOf {
            source,
            multisample,
        });
        multisample
    }
    fn viewport_transform_state(&self) -> u32 {
        let state = self.env.viewport_transform_state();
        self.log(Query::ViewportTransform(state));
        state
    }
    fn texture_bound_buffer(&self) -> u32 {
        let buffer = self.env.texture_bound_buffer();
        self.log(Query::TextureBoundBuffer(buffer));
        buffer
    }
    fn local_memory_size(&self) -> u32 {
        let size = self.env.local_memory_size();
        self.log(Query::LocalMemorySize(size));
        size
    }
    fn has_hle_macro_state(&self) -> bool {
        let state = self.env.has_hle_macro_state();
        self.log(Query::HleMacroState(state));
        state
    }
    fn is_cbuf_storage_buffer_pointer(&self, index: u32, offset: u32) -> bool {
        let answer = self.env.is_cbuf_storage_buffer_pointer(index, offset);
        self.log(Query::StoragePointer {
            index,
            offset,
            answer,
        });
        answer
    }
    fn is_native_program(&self) -> bool {
        let native = self.env.is_native_program();
        self.log(Query::NativeProgram(native));
        native
    }
    fn is_cbuf_indirect_storage_buffer_pointer(
        &self,
        index: u32,
        offset: u32,
        indirect: u32,
    ) -> bool {
        let answer = self
            .env
            .is_cbuf_indirect_storage_buffer_pointer(index, offset, indirect);
        self.log(Query::IndirectStoragePointer {
            index,
            offset,
            indirect,
            answer,
        });
        answer
    }
    fn gp_passthrough_mask(&self) -> [u32; 8] {
        let mask = self.env.gp_passthrough_mask();
        self.log(Query::PassthroughMask(mask));
        mask
    }
    fn replace_const_buffer(&self, bank: u32, offset: u32) -> Option<ReplaceConstant> {
        let answer = self.env.replace_const_buffer(bank, offset);
        self.log(Query::ReplaceConstBuffer {
            bank,
            offset,
            answer,
        });
        answer
    }
    fn is_proprietary_driver(&self) -> bool {
        let proprietary = self.env.is_proprietary_driver();
        self.log(Query::ProprietaryDriver(proprietary));
        proprietary
    }
    fn mesh_state(&self) -> Option<MeshState> {
        let state = self.env.mesh_state();
        self.log(Query::Mesh(state));
        state
    }
}
