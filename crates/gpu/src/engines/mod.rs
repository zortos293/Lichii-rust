//! GPU engines bound to channel subchannels.

pub mod maxwell_3d;

/// Word index (method number) of a register field: `reg_index!(Regs, draw.begin)`.
#[macro_export]
macro_rules! reg_index {
    ($regs:ty, $($field:tt)+) => {
        (core::mem::offset_of!($regs, $($field)+) / 4) as u32
    };
}
