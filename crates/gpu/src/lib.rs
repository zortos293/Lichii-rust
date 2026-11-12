//! The guest GPU (Maxwell for Switch 1, Ampere/NVN2 for Switch 2): GPFIFO and the DMA pusher,
//! the 3D/compute/DMA/2D/inline engines, Maxwell and Turing+ macros, the GMMU, host1x
//! syncpoints, and the [`backend`] contract the renderer implements.

pub mod backend;
mod bits;
pub mod device_memory;
pub mod engines;
pub mod gmmu;
pub mod host;
pub mod host1x;
pub mod macros;
pub mod nx2_memory;
pub mod pushbuffer;
