//! Ampere (SM86, NVN2) frontend: 128-bit instructions with embedded scheduling bits.

mod decode;
mod flow;
mod header;
mod instruction;
mod opcode;
pub(crate) mod translate;

#[cfg(test)]
mod tests;

pub use decode::{Decoded, Form, decode};
pub use flow::AmpereFlow;
pub use header::{
    GRAPHICS_HEADER_SIZE, SPH_SIZE, graphics_header_address, graphics_start_address, read_sph,
};
pub use instruction::{CbufBank, CbufOperand, INSTRUCTION_SIZE, Instruction};
pub use opcode::Opcode;
