//! What one translation asked its environment, with the answers, and replaying it.

use std::cell::RefCell;

use lichii_common::GpuVAddr;
use serde::{Deserialize, Serialize};

use crate::info::{ReplaceConstant, TexturePixelFormat};
use crate::ir::TextureType;
use crate::{MeshState, ShaderArch, ShaderEnvironment, ShaderStage};

/// One question translation asked and the answer it got. Program memory is addressed relative
/// to the program's first instruction, so the same code elsewhere gives the same record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
enum Query {
    Arch(ShaderArch),
    Stage(ShaderStage),
    StartAlignment(u64),
    Code {
        offset: i64,
        value: u64,
    },
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
}

impl Query {
    /// Whether `env` gives the recorded answer to this question.
    fn holds(&self, env: &dyn ShaderEnvironment) -> bool {
        match *self {
            Query::Arch(arch) => env.arch() == arch,
            Query::Stage(stage) => env.stage() == stage,
            Query::StartAlignment(alignment) => env.start_address() % 32 == alignment,
            Query::Code { offset, value } => {
                env.read_u64(env.start_address().wrapping_add_signed(offset)) == value
            }
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
        }
    }
}

/// Every question one program's translation asked, in order, with its answers. Translation is
/// deterministic, so an environment that gives the same answers in the same order translates
/// to the same module.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(super) struct EnvRecord(Vec<Query>);

impl EnvRecord {
    /// Whether `env` answers every question as recorded; program memory only with `code`.
    pub(super) fn matches(&self, env: &dyn ShaderEnvironment, code: bool) -> bool {
        self.0
            .iter()
            .filter(|query| code || !matches!(query, Query::Code { .. }))
            .all(|query| query.holds(env))
    }

    /// The program bytes translation read: the first one's offset from the program's first
    /// instruction (negative for its header) and the length up to the last one read.
    pub(super) fn code_range(&self) -> (i64, u64) {
        let offsets = self.0.iter().filter_map(|query| match *query {
            Query::Code { offset, .. } => Some(offset),
            _ => None,
        });
        let (first, last) = offsets.fold((i64::MAX, i64::MIN), |(first, last), offset| {
            (first.min(offset), last.max(offset))
        });
        if first > last {
            return (0, 0);
        }
        (first, (last - first) as u64 + 8)
    }
}

/// Forwards every query to `env` and records the questions and answers.
pub(super) struct Logger<'a> {
    env: &'a dyn ShaderEnvironment,
    start: GpuVAddr,
    log: RefCell<Vec<Query>>,
}

impl<'a> Logger<'a> {
    pub(super) fn new(env: &'a dyn ShaderEnvironment) -> Logger<'a> {
        Logger {
            env,
            start: env.start_address(),
            log: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn into_record(self) -> EnvRecord {
        EnvRecord(self.log.into_inner())
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
        let value = self.env.read_u64(address);
        let offset = address.wrapping_sub(self.start) as i64;
        self.log(Query::Code { offset, value });
        value
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
