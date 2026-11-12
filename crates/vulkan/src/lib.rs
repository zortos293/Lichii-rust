//! Vulkan renderer: implements `lichii_gpu::backend::Rasterizer` and presents frames.
//!
//! Requires Vulkan 1.3 (timeline semaphores, dynamic rendering, synchronization2).

pub mod commands;
pub mod descriptors;
pub mod device;
mod error;
pub mod format;
pub mod instance;
pub mod memory;
pub mod pipeline;
pub mod present;
pub mod rasterizer;
pub mod scheduler;
pub mod shaders;
pub mod staging;
pub mod swapchain;

pub use device::{Device, DeviceExtensions, DeviceFeatures, DeviceProperties, available_devices};
pub use error::{Error, Result};
pub use instance::{Instance, SurfaceSupport};
