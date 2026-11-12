//! Maxwell (SM5x) frontend: 64-bit instructions, one scheduling word per three instructions.

pub mod decode;
pub mod flow;
pub mod header;
pub mod instruction;
pub mod opcode;

pub use decode::decode;
pub use flow::MaxwellFlow;
pub use header::ProgramHeader;
pub use instruction::Instruction;
pub use opcode::Opcode;
