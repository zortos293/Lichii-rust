//! The contract between the GPU and the renderer.
//!
//! The GPU parses command streams, runs macros and executes engine methods; whenever a method
//! needs the host GPU (a draw, a dispatch, a clear, a copy it can accelerate, a query, a fence) it
//! calls the [`Rasterizer`]. The Vulkan crew implements it.
//!
//! Every call comes from the GPU's back thread (see `crate::split`), in guest order. Requests
//! that start elsewhere (CPU writes, flushes, presentation) are forwarded into the same stream, so
//! the renderer never sees two threads.
//!
//! Draw setup stays cheap because nothing is copied: a draw borrows the 3D engine's registers and
//! dirty flags ([`GraphicsContext`]); the renderer reads the state it needs and clears the dirty
//! flags it consumed. Same-value register writes never set dirty flags, so unchanged state costs
//! one flag test per group.

use std::sync::Arc;

use lichii_common::{DAddr, GpuVAddr};
use lichii_texture::PixelFormat;

use crate::device_memory::DeviceMemory;
use crate::engines::maxwell_3d::regs::{MAX_CONST_BUFFERS, MAX_SHADER_STAGE, PrimitiveTopology};
use crate::engines::maxwell_3d::{DirtyState, Regs};
use crate::gmmu::Gmmu;
use crate::host1x::Syncpoints;

/// A GPU channel (one per nvhost-gpu channel or NVN2 queue).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId(pub u32);

/// Shared GPU state the renderer needs outside individual calls.
#[derive(Clone)]
pub struct GpuHandles {
    pub device: Arc<DeviceMemory>,
    pub syncpoints: Arc<Syncpoints>,
}

/// A constant buffer bound to a shader stage slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstBufferBinding {
    pub address: GpuVAddr,
    pub size: u32,
    pub enabled: bool,
}

/// Graphics constant buffers, indexed `[stage][slot]` (stages: vertex, tessellation control,
/// tessellation evaluation, geometry, fragment).
pub type GraphicsConstBuffers = [[ConstBufferBinding; MAX_CONST_BUFFERS]; MAX_SHADER_STAGE];

/// NVN2 root tables (constant banks 24 to 31), loaded through the 3D engine's root-table methods.
pub type RootTables = [[u32; 64]; 8];

/// What the renderer sees of the 3D engine for a draw or a clear.
pub struct GraphicsContext<'a> {
    pub channel: ChannelId,
    pub memory: &'a Gmmu,
    pub regs: &'a Regs,
    /// Flags set by register writes since the renderer last cleared them.
    pub dirty: &'a mut DirtyState,
    pub const_buffers: &'a GraphicsConstBuffers,
    pub root_tables: &'a RootTables,
}

/// Index data a draw carries inline instead of reading from an index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineIndices<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
}

/// A direct draw. Vertex and index buffer bindings are in the registers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawParams<'a> {
    pub topology: PrimitiveTopology,
    pub indexed: bool,
    /// First vertex, or first index when indexed.
    pub first: u32,
    pub count: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
    pub instance_count: u32,
    /// Indices written through the inline-index methods; the draw is indexed with them.
    pub inline_indices: Option<InlineIndices<'a>>,
}

/// A draw whose parameters live in GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndirectDraw {
    pub topology: PrimitiveTopology,
    pub indexed: bool,
    /// First command (VkDrawIndirectCommand or VkDrawIndexedIndirectCommand layout).
    pub buffer: GpuVAddr,
    pub stride: u32,
    pub max_draws: u32,
    /// When set, the draw count is the u32 at this address (clamped to `max_draws`).
    pub count_buffer: Option<GpuVAddr>,
}

/// Which compute launch descriptor layout a dispatch carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QmdVersion {
    /// Maxwell/Kepler compute (B1C0).
    V2,
    /// Ampere native compute (C7C0, NVN2).
    V3,
}

/// A compute dispatch, decoded from its launch descriptor (QMD).
pub struct ComputeContext<'a> {
    pub channel: ChannelId,
    pub memory: &'a Gmmu,
    /// GPU address of the program's first instruction.
    pub program: GpuVAddr,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_size: u32,
    pub const_buffers: &'a [ConstBufferBinding; 8],
    pub texture_headers: GpuVAddr,
    pub texture_header_limit: u32,
    pub samplers: GpuVAddr,
    pub sampler_limit: u32,
    /// Samplers are indexed with the texture header index.
    pub linked_samplers: bool,
    pub qmd_version: QmdVersion,
    /// The raw descriptor, for the shader environment.
    pub qmd: &'a [u32; 64],
}

/// A surface as the copy and blit engines describe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopySurface {
    pub address: GpuVAddr,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layer: u32,
    /// Row pitch in bytes when pitch-linear.
    pub pitch: u32,
    /// `Some`((width, height, depth) log2 block size in GOBs) when block-linear.
    pub block: Option<(u32, u32, u32)>,
    pub bytes_per_pixel: u32,
}

/// A copy engine (DMA) transfer between two surfaces, in pixels of `bytes_per_pixel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageCopy {
    pub src: CopySurface,
    pub dst: CopySurface,
    pub src_origin: (u32, u32),
    pub dst_origin: (u32, u32),
    pub width: u32,
    pub height: u32,
}

/// A 2D engine blit with scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Blit {
    pub src: CopySurface,
    pub dst: CopySurface,
    pub src_format: PixelFormat,
    pub dst_format: PixelFormat,
    /// Source rectangle in source pixels, 32.32 fixed point.
    pub src_x0: i64,
    pub src_y0: i64,
    pub du_dx: i64,
    pub dv_dy: i64,
    /// Destination rectangle in pixels.
    pub dst_x0: i32,
    pub dst_y0: i32,
    pub dst_width: i32,
    pub dst_height: i32,
    pub linear_filter: bool,
}

/// What a query or semaphore release writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryKind {
    /// The given payload.
    Payload,
    /// Samples that passed the depth test since the counter was reset.
    SamplesPassed,
    PrimitivesGenerated,
    TransformFeedbackPrimitivesWritten,
    /// Any other counter, by its report number; renderers may write 0.
    Counter(u32),
}

/// How many bytes a release writes at its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseSize {
    /// The low 32 bits of the value.
    Four,
    /// The 64-bit value.
    Eight,
    /// The 64-bit value, then a 64-bit GPU timestamp.
    Sixteen,
}

/// A query result or semaphore release to write to GPU memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Query {
    pub address: GpuVAddr,
    pub kind: QueryKind,
    /// Value for [`QueryKind::Payload`].
    pub payload: u64,
    pub size: ReleaseSize,
    /// Write only after all earlier host GPU work completed. When false the renderer may write
    /// at once.
    pub after_work: bool,
}

/// An action to run once the host GPU finished all work submitted before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FenceAction {
    /// Write a semaphore value to device memory.
    WriteSemaphore {
        address: DAddr,
        value: u64,
        size: ReleaseSize,
    },
    /// Increment a host1x syncpoint's guest-visible value (its work has finished).
    IncrementSyncpoint(u32),
}

impl FenceAction {
    /// Performs the action. The renderer calls this from whichever thread notices completion.
    pub fn execute(self, handles: &GpuHandles, timestamp: u64) {
        match self {
            Self::WriteSemaphore {
                address,
                value,
                size,
            } => {
                let result = match size {
                    ReleaseSize::Four => handles.device.write(address, value as u32),
                    ReleaseSize::Eight => handles.device.write(address, value),
                    ReleaseSize::Sixteen => handles.device.write(address, [value, timestamp]),
                };
                if let Err(fault) = result {
                    tracing::warn!(?fault, "semaphore release to unmapped device memory");
                }
            }
            Self::IncrementSyncpoint(id) => {
                if handles.syncpoints.guest().increment(id).is_err() {
                    tracing::warn!(id, "fence increments an invalid syncpoint");
                }
            }
        }
    }
}

/// A guest frame ready to present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramebufferConfig {
    /// Device address of the first byte.
    pub address: DAddr,
    pub width: u32,
    pub height: u32,
    /// Row pitch in bytes for pitch-linear surfaces.
    pub stride: u32,
    pub format: PixelFormat,
    /// `Some(log2 block height in GOBs)` for block-linear surfaces.
    pub block_height_log2: Option<u32>,
    pub flip_vertical: bool,
    pub crop: Option<CropRect>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// Implemented by the renderer; called from the GPU's back thread only.
///
/// Methods with a default body are optional: the defaults do the work in software (copies) or
/// nothing (hints).
pub trait Rasterizer: Send {
    /// A channel was created.
    fn bind_channel(&mut self, _channel: ChannelId) {}

    /// A channel was destroyed; drop its caches.
    fn release_channel(&mut self, _channel: ChannelId) {}

    fn draw(&mut self, ctx: &mut GraphicsContext<'_>, draw: &DrawParams<'_>);

    fn draw_indirect(&mut self, ctx: &mut GraphicsContext<'_>, draw: &IndirectDraw);

    /// Clears the render targets `regs.clear_surface` selects, over `layer_count` layers.
    fn clear(&mut self, ctx: &mut GraphicsContext<'_>, layer_count: u32);

    fn dispatch_compute(&mut self, ctx: &ComputeContext<'_>);

    /// Copies `size` bytes between linear buffers on the host GPU. `false` makes the GPU copy
    /// guest memory itself (after a flush of both ranges).
    fn accelerate_buffer_copy(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        _src: GpuVAddr,
        _dst: GpuVAddr,
        _size: u64,
    ) -> bool {
        false
    }

    /// A copy engine transfer between surfaces (tiling or untiling). `false` makes the GPU
    /// do it in software.
    fn accelerate_image_copy(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        _copy: &ImageCopy,
    ) -> bool {
        false
    }

    /// A 2D engine blit. `false` makes the GPU do it in software.
    fn accelerate_blit(&mut self, _channel: ChannelId, _memory: &Gmmu, _blit: &Blit) -> bool {
        false
    }

    /// The GPU wrote `data` to guest memory at `address` (inline-to-memory or constant
    /// buffer updates). The memory already holds the data; drop or patch cached copies.
    fn inline_written(&mut self, _channel: ChannelId, memory: &Gmmu, address: GpuVAddr, size: u64) {
        memory.for_each_range(address, size, |range| {
            if let crate::gmmu::RangeTarget::Device(daddr) = range.target {
                self.invalidate_region(daddr, range.size);
            }
        });
    }

    /// Writes a query result or semaphore release.
    fn query(&mut self, channel: ChannelId, memory: &Gmmu, query: &Query);

    /// Resets a counter [`QueryKind`] reads from.
    fn reset_counter(&mut self, _kind: QueryKind) {}

    /// Runs `action` once the host GPU finished the work submitted so far (or at once when
    /// fences are not tracked).
    fn signal_fence(&mut self, action: FenceAction);

    /// Runs the actions of finished fences; `wait` blocks until all of them finished.
    fn release_fences(&mut self, wait: bool);

    /// The guest CPU wrote `[addr, addr + size)`; drop or re-upload cached copies.
    fn invalidate_region(&mut self, addr: DAddr, size: u64);

    /// Write GPU-modified data in `[addr, addr + size)` back to guest memory.
    fn flush_region(&mut self, addr: DAddr, size: u64);

    /// Device memory `[addr, addr + size)` is no longer mapped in some GPU address space.
    fn unmap_region(&mut self, addr: DAddr, size: u64) {
        self.invalidate_region(addr, size);
    }

    /// Texture and shader caches must not use stale guest data (cache invalidation methods).
    fn invalidate_gpu_caches(&mut self) {}

    /// The guest asked for a fragment or tiled cache barrier.
    fn barrier(&mut self) {}

    /// Submits recorded host GPU work (end of a GPFIFO submission).
    fn flush_commands(&mut self) {}

    /// Block until the host GPU finished all submitted work.
    fn wait_for_idle(&mut self);

    /// Presents a guest frame (scaled, filtered and anti-aliased per settings).
    fn present(&mut self, framebuffer: &FramebufferConfig);
}
