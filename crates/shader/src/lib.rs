//! Guest shader translation: Maxwell (SM5x) and Ampere (NVN2) SASS -> IR -> SPIR-V.
//!
//! The renderer supplies a [`ShaderEnvironment`] that reads the program and the GPU state it
//! depends on, and gets back a [`ShaderModule`] with SPIR-V and the resource layout the
//! pipeline needs. The shader crew owns everything here; the Vulkan crew consumes it.

#[cfg(test)]
macro_rules! test_gpu_state {
    () => {
        test_gpu_state!(@shared);
        fn local_memory_size(&self) -> u32 {
            0
        }
    };
    (@shared) => {
        fn texture_type(&self, _: u32) -> $crate::ir::TextureType {
            $crate::ir::TextureType::Color2D
        }
        fn texture_pixel_format(&self, _: u32) -> $crate::info::TexturePixelFormat {
            $crate::info::TexturePixelFormat::A8b8g8r8Unorm
        }
        fn is_texture_pixel_format_integer(&self, _: u32) -> bool {
            false
        }
        fn viewport_transform_state(&self) -> u32 {
            1
        }
        fn texture_bound_buffer(&self) -> u32 {
            2
        }
        fn has_hle_macro_state(&self) -> bool {
            false
        }
    };
}

pub mod ampere;
pub mod backend;
pub mod cache;
pub mod flow;
pub mod handle;
mod hash;
pub mod info;
pub mod ir;
pub mod maxwell;
pub mod module;
pub mod opt;
pub mod profile;
pub mod runtime;
pub mod translate;

pub use info::ShaderInfo;
pub use module::{
    DescriptorBinding, DescriptorKind, DescriptorSetLayout, ResourceLayout, ShaderModule, compile,
    compile_dual_vertex,
};
pub use profile::Profile;
pub use runtime::RuntimeInfo;

use lichii_common::GpuVAddr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShaderStage {
    VertexA,
    VertexB,
    TessellationControl,
    TessellationEval,
    Geometry,
    Fragment,
    Compute,
    Task,
    Mesh,
}

/// Instruction set of the guest program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ShaderArch {
    /// 64-bit instructions with scheduling words (Switch 1).
    Maxwell,
    /// 128-bit instructions (Switch 2, NVN2).
    Ampere,
}

/// What the translator may read from the guest while translating one program.
pub trait ShaderEnvironment {
    fn arch(&self) -> ShaderArch;
    fn stage(&self) -> ShaderStage;
    /// GPU address of the program's first instruction.
    fn start_address(&self) -> GpuVAddr;
    /// Reads one 64-bit word of program memory at `address`.
    fn read_u64(&self, address: GpuVAddr) -> u64;
    /// Reads program memory from `address` into `out`, whose length is a multiple of 8, as
    /// [`Self::read_u64`] would word by word. Override it where a block read is cheaper.
    fn read_code(&self, address: GpuVAddr, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(8).enumerate() {
            let value = self.read_u64(address.wrapping_add(i as u64 * 8));
            word.copy_from_slice(&value.to_le_bytes());
        }
    }
    /// Value of constant buffer `index` at byte `offset`, for bindless handle tracking.
    fn read_cbuf(&self, index: u32, offset: u32) -> u32;
    /// Shared memory size in bytes (compute).
    fn shared_memory_size(&self) -> u32;
    /// Workgroup size (compute).
    fn workgroup_size(&self) -> [u32; 3];
    /// Type of the texture descriptor (TIC) a raw handle selects; texture instructions whose
    /// encoding leaves the type open take it from here.
    fn texture_type(&self, raw_handle: u32) -> ir::TextureType;
    /// Pixel format of the texture descriptor a raw handle selects.
    fn texture_pixel_format(&self, raw_handle: u32) -> info::TexturePixelFormat;
    /// Whether that descriptor's pixel format is an integer format.
    fn is_texture_pixel_format_integer(&self, raw_handle: u32) -> bool;
    /// Whether that descriptor holds a multisampled texture. Size and sample count queries,
    /// which address no sample, take it from here; without it only a per-sample fetch marks a
    /// texture multisampled.
    fn is_texture_multisample(&self, _raw_handle: u32) -> bool {
        false
    }
    /// [`Self::texture_type`] of the handle the constant buffer words of `source` hold.
    /// Translation asks through these `_of` queries, so an environment that records what
    /// translation read overrides them to record `source` and the answer instead of the words:
    /// another handle that selects the same kind of texture then translates the same.
    fn texture_type_of(&self, source: info::HandleSource) -> ir::TextureType {
        self.texture_type(source.handle(|index, offset| self.read_cbuf(index, offset)))
    }
    /// [`Self::texture_pixel_format`] of the handle at `source`.
    fn texture_pixel_format_of(&self, source: info::HandleSource) -> info::TexturePixelFormat {
        self.texture_pixel_format(source.handle(|index, offset| self.read_cbuf(index, offset)))
    }
    /// [`Self::is_texture_pixel_format_integer`] of the handle at `source`.
    fn is_texture_pixel_format_integer_of(&self, source: info::HandleSource) -> bool {
        let handle = source.handle(|index, offset| self.read_cbuf(index, offset));
        self.is_texture_pixel_format_integer(handle)
    }
    /// [`Self::is_texture_multisample`] of the handle at `source`.
    fn is_texture_multisample_of(&self, source: info::HandleSource) -> bool {
        self.is_texture_multisample(source.handle(|index, offset| self.read_cbuf(index, offset)))
    }
    /// The viewport transform enable register (0 when the guest disabled it).
    fn viewport_transform_state(&self) -> u32;
    /// Constant buffer that holds bound texture handles.
    fn texture_bound_buffer(&self) -> u32;
    /// Local memory size in bytes the program header or launch asks for.
    fn local_memory_size(&self) -> u32;
    /// Whether HLE macros replaced some draw parameters, so their constant buffer reads must
    /// become shader inputs.
    fn has_hle_macro_state(&self) -> bool;
    /// Whether constant buffer `index` at `offset` may hold a storage buffer pointer.
    fn is_cbuf_storage_buffer_pointer(&self, _index: u32, _offset: u32) -> bool {
        true
    }
    /// A Switch 2 native program lowered to Maxwell instructions: its storage buffer pointers
    /// are raw 8-byte arguments in constant buffers, without the NVN descriptor layout.
    fn is_native_program(&self) -> bool {
        false
    }
    /// Native programs: whether the 8 bytes at `indirect` past the pointer at constant buffer
    /// `index`, `offset` are a mapped GPU address (a pointer table the program loads its buffer
    /// pointer from).
    fn is_cbuf_indirect_storage_buffer_pointer(
        &self,
        _index: u32,
        _offset: u32,
        _indirect: u32,
    ) -> bool {
        true
    }
    /// Per-pipeline geometry passthrough mask (geometry shaders).
    fn gp_passthrough_mask(&self) -> [u32; 8] {
        [0; 8]
    }
    /// Draw parameter that HLE macros store in constant buffer `bank` at `offset`, if any.
    fn replace_const_buffer(&self, _bank: u32, _offset: u32) -> Option<info::ReplaceConstant> {
        None
    }
    /// Whether the host runs NVIDIA's proprietary driver, for which constant propagation folds
    /// driver constant buffer (bank 1) reads to immediates.
    fn is_proprietary_driver(&self) -> bool {
        false
    }
    /// Task and mesh programs: the mesh state they are drawn with. Without it (the default)
    /// they fail to translate with [`TranslateError::MissingMeshState`].
    fn mesh_state(&self) -> Option<MeshState> {
        None
    }
}

/// The draw state a task or mesh program is translated against.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct MeshState {
    /// SET_MESH_SHADER_A: output topology in bits 3:0 (0 points, 1 lines, 4 triangles), maximum
    /// vertices in 12:4, maximum primitives in 22:13.
    pub shader_a: u32,
    /// SET_MESH_SHADER_B: mesh workgroup threads in bits 25:20.
    pub shader_b: u32,
    /// SET_MESH_INIT_SHADER: task workgroup threads in bits 5:0, local buffer lines in 20:12,
    /// lines passed to the mesh stage in 30:22 (a line is 128 bytes).
    pub init_shader: u32,
    /// Mesh programs: the header of the pass-through geometry program bound with them, whose
    /// output map lists the per-primitive outputs.
    pub primitive_header: maxwell::header::ProgramHeader,
}

#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("unsupported instruction {opcode:#x} at {address:#x}")]
    Unsupported { address: GpuVAddr, opcode: u128 },
    #[error("invalid instruction encoding {raw:#x} at {address:#x}")]
    Invalid { address: GpuVAddr, raw: u128 },
    #[error("indirect branch at {address:#x} has no trackable jump table")]
    UntrackedIndirectBranch { address: GpuVAddr },
    #[error("control flow at {address:#x} cannot be structured")]
    Unstructured { address: GpuVAddr },
    #[error("cannot lower {0}")]
    Lowering(String),
    #[error(transparent)]
    Emit(#[from] backend::spirv::EmitError),
    #[error("translation produced invalid IR: {0}")]
    InvalidIr(String),
    #[error("invalid program header at {address:#x}")]
    InvalidHeader { address: GpuVAddr },
    #[error("no {arch:?} translator for {stage:?} programs yet")]
    UnsupportedProgram {
        arch: ShaderArch,
        stage: ShaderStage,
    },
    #[error("binding {count} storage buffers to a stage that binds {limit}")]
    TooManyStorageBuffers { count: usize, limit: usize },
    #[error(transparent)]
    Texture(#[from] opt::TextureError),
    #[error("{stage:?} program translated without mesh state")]
    MissingMeshState { stage: ShaderStage },
    #[error("unsupported mesh program: {0}")]
    UnsupportedMesh(&'static str),
}

impl TranslateError {
    /// Whether the texture headers translation read may have caused the failure, so the same
    /// program may translate under other headers: only the texture pass reads them to decide.
    pub fn depends_on_textures(&self) -> bool {
        matches!(self, Self::Texture(_))
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    #[test]
    fn only_failures_a_texture_header_may_cause_depend_on_textures() {
        let format = TranslateError::from(opt::TextureError::ByteAddressedFormat {
            format: info::TexturePixelFormat::B10g11r11Float,
            ty: ir::TextureType::Color2D,
            index: 0,
            offset: 0x70,
        });
        assert!(format.depends_on_textures());
        let narrow = TranslateError::from(backend::spirv::EmitError::Unsupported(
            "8 and 16-bit global memory access the storage buffer pass left",
        ));
        assert!(!narrow.depends_on_textures());
        let unsupported = TranslateError::Unsupported {
            address: GpuVAddr::default(),
            opcode: 0,
        };
        assert!(!unsupported.depends_on_textures());
        assert!(!TranslateError::Lowering("64-bit integers".into()).depends_on_textures());
    }
}
