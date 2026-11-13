//! SPIR-V backend, a port of the C++ `backend/spirv` on rspirv.
//!
//! [`emit_spirv`] builds an [`context::EmitContext`] (types, constants, resources, interface
//! variables and helper functions for the program), walks the structured syntax list emitting
//! every instruction through the per-opcode families in `emit/`, then adds the entry point,
//! execution modes and capabilities and patches the phis.

mod context;
mod driver;
mod emit;
#[cfg(test)]
mod tests;

use rspirv::dr;

use crate::Profile;
use crate::RuntimeInfo;
use crate::ShaderStage;
use crate::backend::Bindings;
use crate::ir::{Attribute, Opcode};
use crate::translate::TranslatedProgram;

/// Words of the texture rescaling bit set in the rescaling push constant block.
pub const NUM_TEXTURE_SCALING_WORDS: u32 = 4;
/// Words of the image rescaling bit set in the rescaling push constant block.
pub const NUM_IMAGE_SCALING_WORDS: u32 = 2;
/// Byte offset of the texture rescaling words in the rescaling push constant block.
pub const RESCALING_LAYOUT_TEXTURES_OFFSET: u32 = 0;
/// Byte offset of the image rescaling words in the rescaling push constant block.
pub const RESCALING_LAYOUT_IMAGES_OFFSET: u32 = 16;
/// Byte offset of the resolution down factor (graphics stages) in the rescaling block.
pub const RESCALING_LAYOUT_DOWN_FACTOR_OFFSET: u32 = 24;
/// Byte offset of the render area vector in the render area push constant block.
pub const RENDER_AREA_LAYOUT_OFFSET: u32 = 0;

/// Why a program could not be emitted.
#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    /// A read of an attribute the backend has no input for, such as a fixed-function varying.
    #[error("reads of attribute {0} are not supported by the SPIR-V backend")]
    UnsupportedAttribute(Attribute),
    /// Mesh, task and tensor instructions, which this backend does not emit.
    #[error("{0:?} is not supported by the SPIR-V backend")]
    UnsupportedOpcode(Opcode),
    #[error("{0:?} programs are not supported by the SPIR-V backend")]
    UnsupportedStage(ShaderStage),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// The IR breaks an invariant emission relies on.
    #[error("invalid IR for SPIR-V emission: {0}")]
    InvalidIr(String),
    #[error("SPIR-V builder: {0}")]
    Builder(#[from] dr::Error),
}

/// Result of an emission step.
pub type EmitResult<T> = Result<T, EmitError>;

/// Emits `program` as a SPIR-V module. `bindings` holds the first free binding of each
/// descriptor kind and is advanced past the descriptors this stage declares.
pub fn emit_spirv(
    profile: &Profile,
    runtime: &RuntimeInfo,
    program: &TranslatedProgram,
    bindings: &mut Bindings,
) -> Result<Vec<u32>, EmitError> {
    driver::emit(profile, runtime, program, bindings)
}
