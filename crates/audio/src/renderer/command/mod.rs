//! Command generation: each frame, the renderer state becomes the DSP's command list.
//!
//! The generator walks the contexts of a [`RendererSystem`](super::RendererSystem) in the
//! reference's order (voices, sub mixes, the final mix, sinks), [`CommandBuffer`] stamps every
//! command with its node id and an estimate from [`ProcessingTimeEstimator`], and the system's
//! voice drop pass disables the lowest priority voices when the estimate passes the render time
//! limit.

mod buffer;
mod estimator;
mod generator;

pub use buffer::CommandBuffer;
pub use estimator::{EstimatorVersion, ProcessingTimeEstimator};
pub use generator::INVALID_NODE_ID;
pub(crate) use generator::{CommandGenerator, Output};
