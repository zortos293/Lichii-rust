//! The SSA intermediate representation both guest instruction sets translate into.

mod condition;
mod value;

pub use condition::{Condition, FlowTest};
pub use value::{Attribute, Patch, Pred, Reg};
