//! The guest GPU (Maxwell for Switch 1, Ampere/NVN2 for Switch 2): GPFIFO and the DMA pusher,
//! the 3D/compute/DMA/2D/inline engines, Maxwell and Turing+ macros, the GMMU, host1x
//! syncpoints, and the [`backend`] contract the renderer implements.
//!
//! [`gpu::Gpu`] runs it: a front thread parses channels' command streams and runs macros, a back
//! thread executes the engines and drives the renderer.

pub mod activity;
pub mod diag_ring;
mod back;
pub mod backend;
mod bits;
pub mod channel;
pub mod clock;
pub mod device_memory;
mod dma_pusher;
pub mod engines;
pub mod gmmu;
pub mod gpu;
pub mod host;
pub mod host1x;
pub mod lazy;
pub mod macros;
pub mod nx2_memory;
pub mod pushbuffer;
mod ring;
pub mod semaphore_mirror;
mod stream;
mod written_pages;

#[cfg(test)]
mod gpu_tests;

pub use gpu::{Console, Gpu, GpuConfig, Threading};
