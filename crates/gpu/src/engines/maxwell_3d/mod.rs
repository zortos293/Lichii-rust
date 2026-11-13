//! The 3D engine (Maxwell B197, Ampere C797).

mod const_buffers;
mod draw;
mod engine;
mod queries;
pub mod regs;
pub mod state;

pub use engine::Maxwell3d;
pub use regs::Regs;
pub use state::{
    DirtyState, MACRO_REGISTERS_START, MethodSet, ROOT_TABLE_LOAD, ROOT_TABLE_SELECT,
    RegisterState, dirty,
};
