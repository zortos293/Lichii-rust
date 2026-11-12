//! Guest shader translation: Maxwell (SM5x) and Ampere (NVN2) SASS -> IR -> SPIR-V.
//!
//! The renderer supplies a [\`ShaderEnvironment\`] that reads the program and the GPU state it
//! depends on, and gets back a [\`ShaderModule\`] with SPIR-V and the resource layout the
//! pipeline needs. The shader crew owns everything here; the Vulkan crew consumes it.

pub mod ampere;
pub mod flow;
pub mod info;
pub mod ir;
pub mod maxwell;
pub mod opt;
pub mod profile;
pub mod runtime;

pub use info::ShaderInfo;
pub use profile::Profile;
pub use runtime::RuntimeInfo;

use lichii_common::GpuVAddr;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// Reads one 64-bit word of program memory at \`address\`.
    fn read_u64(&self, address: GpuVAddr) -> u64;
    /// Value of constant buffer \`index\` at byte \`offset\`, for bindless handle tracking.
    fn read_cbuf(&self, index: u32, offset: u32) -> u32;
    /// Shared memory size in bytes (compute).
    fn shared_memory_size(&self) -> u32;
    /// Workgroup size (compute).
    fn workgroup_size(&self) -> [u32; 3];
}

/// A translated program.
#[derive(Clone, Debug, Default)]
pub struct ShaderModule {
    pub spirv: Vec<u32>,
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
}
