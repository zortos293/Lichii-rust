//! The 3D engine (Maxwell B197, Ampere C797).

pub mod regs;
pub mod state;

pub use regs::Regs;
pub use state::{
    DirtyState, MACRO_REGISTERS_START, MethodSet, ROOT_TABLE_LOAD, ROOT_TABLE_SELECT,
    RegisterState, dirty,
};
