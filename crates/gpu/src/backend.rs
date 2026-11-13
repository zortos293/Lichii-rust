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
use std::sync::atomic::{AtomicU32, Ordering};

use lichii_common::{DAddr, GpuVAddr};
use lichii_memory::DeviceAddressSpace;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;

use crate::clock::GpuTimestamp;
use crate::engines::maxwell_3d::regs::{MAX_CONST_BUFFERS, MAX_SHADER_STAGE, PrimitiveTopology};
use crate::engines::maxwell_3d::{DirtyState, Regs};
use crate::gmmu::Gmmu;
use crate::host::Reduction;
use crate::host1x::Syncpoints;
use crate::semaphore_mirror::{self, FirstPage};
pub use crate::written_pages::GpuWrittenPages;

/// A GPU channel (one per nvhost-gpu channel or NVN2 queue).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId(pub u32);

/// Shared GPU state the renderer needs outside individual calls.
#[derive(Clone)]
pub struct GpuHandles {
    pub device: Arc<DeviceAddressSpace>,
    pub syncpoints: Arc<Syncpoints>,
    /// The clock 16-byte releases and query reports take their timestamp from.
    pub timestamp: Arc<GpuTimestamp>,
    /// Device pages GPU work may have written. The renderer marks a range when it records work
    /// that writes it; the front synchronises before fetching commands from a marked page.
    pub written_pages: Arc<GpuWrittenPages>,
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

/// A DRAW_TEXTURE: one texture of the bound pool drawn over a rectangle of the render targets,
/// outside the pipeline. Destination coordinates are render-target pixels and source coordinates
/// texels of `texture`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrawTexture {
    pub dst_x0: f32,
    pub dst_y0: f32,
    pub dst_x1: f32,
    pub dst_y1: f32,
    pub src_x0: f32,
    pub src_y0: f32,
    pub src_x1: f32,
    pub src_y1: f32,
    /// Texture header index.
    pub texture: u32,
    /// Sampler index.
    pub sampler: u32,
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
    /// Local memory each thread uses, in bytes: the descriptor's positive and call-return-stack
    /// allocations.
    pub local_memory_size: u32,
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
    /// Version 1.x descriptors: GPU work may have written the grid the guest uploaded into the
    /// descriptor after the guest's copy of it was read, so `grid` may be stale. The grid's
    /// current words are at this GPU address, in the descriptor's layout (words 0xc and 0xd: the
    /// width in bits 0..31, then the height and depth as 16-bit halves); dispatch indirectly from
    /// them.
    pub indirect_grid: Option<GpuVAddr>,
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

/// How a 2D blit combines the source with the destination. The software path performs every
/// operation as [`BlitOperation::SrcCopy`], as the reference does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlitOperation {
    SrcCopyAnd,
    RopAnd,
    Blend,
    SrcCopy,
    Rop,
    SrcCopyPremult,
    BlendPremult,
    SrcCopySrgbToLinear,
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
    pub operation: BlitOperation,
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
    /// Combine a 32-bit semaphore with `payload` (a reduction release): the old value is read
    /// when the action runs, so the result includes every earlier release.
    ReduceSemaphore {
        address: DAddr,
        op: Reduction,
        signed: bool,
        payload: u32,
    },
    /// Increment a host1x syncpoint's guest-visible value (its work has finished).
    IncrementSyncpoint(u32),
}

impl FenceAction {
    /// Performs the action. The renderer calls this from whichever thread notices completion;
    /// 16-byte releases take their timestamp from [`GpuHandles::timestamp`]. A semaphore write
    /// resumes the channels waiting for its value ([`Syncpoints::semaphore_waiters`](crate::host1x::Syncpoints::semaphore_waiters)).
    pub fn execute(self, handles: &GpuHandles) {
        match self {
            Self::WriteSemaphore {
                address,
                value,
                size,
            } => {
                let timestamp = match size {
                    ReleaseSize::Sixteen => handles.timestamp.now(),
                    ReleaseSize::Four | ReleaseSize::Eight => 0,
                };
                let words = [
                    value as u32,
                    (value >> 32) as u32,
                    timestamp as u32,
                    (timestamp >> 32) as u32,
                ];
                let words = &words[..match size {
                    ReleaseSize::Four => 1,
                    ReleaseSize::Eight => 2,
                    ReleaseSize::Sixteen => 4,
                }];
                let bytes = 4 * words.len() as u64;
                let mirror = handles.syncpoints.semaphore_mirror();
                let mut page = mirror.first_page(address);
                let mirrored = mirror.mirror_of(address);
                if let Err(fault) = handles
                    .device
                    .write_bytes(address, bytemuck::cast_slice(words))
                {
                    tracing::warn!(?fault, "semaphore release to unmapped device memory");
                    return;
                }
                if let Some(page) = &mut page {
                    page.note_direct(address, bytes);
                }
                drop(page);
                handles
                    .syncpoints
                    .semaphore_waiters()
                    .written(&handles.device, address, bytes);
                if let Some(page) = mirrored {
                    mirror_release(handles, page, address - semaphore_mirror::PAGE, words);
                }
            }
            Self::ReduceSemaphore {
                address,
                op,
                signed,
                payload,
            } => {
                let mirror = handles.syncpoints.semaphore_mirror();
                let mut page = mirror.first_page(address);
                let mirrored = mirror.mirror_of(address);
                let Some(word) = semaphore_word(&handles.device, address) else {
                    tracing::warn!(
                        address,
                        "semaphore reduction on unmapped or unaligned memory"
                    );
                    return;
                };
                let mut old = word.load(Ordering::Acquire);
                while let Err(current) = word.compare_exchange_weak(
                    old,
                    op.apply(old, payload, signed),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    old = current;
                }
                let after = op.apply(old, payload, signed);
                let seen = match &mut page {
                    Some(page) => page.combine(address, old, after, |current| {
                        op.apply(current, payload, signed)
                    }),
                    None => after,
                };
                drop(page);
                let waiters = handles.syncpoints.semaphore_waiters();
                if seen == after {
                    waiters.written(&handles.device, address, 4);
                } else {
                    waiters.met_by(address, seen);
                }
                if let Some(page) = mirrored {
                    mirror_release(handles, page, address - semaphore_mirror::PAGE, &[after]);
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
    /// GOB byte order of a block-linear surface (Switch 2 native surfaces swap bits 5 and 6,
    /// NX2_NOTES row 48).
    pub gob_layout: GobLayout,
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

    /// Records one direct draw with the pipeline, render targets and bindings `ctx.regs`
    /// describes. The renderer reads only the state `ctx.dirty` flags as changed since its last
    /// call and clears the flags it consumed; the GPU sets them again on the next writes. Guest
    /// memory the draw reads (vertex, index, constant and texture data) is read through
    /// `ctx.memory` during the call; nothing borrowed from `ctx` or `draw` outlives it.
    fn draw(&mut self, ctx: &mut GraphicsContext<'_>, draw: &DrawParams<'_>);

    /// Records draws whose commands, and optionally whose count, the GPU reads from guest memory
    /// when it executes them. State handling is the same as for [`Rasterizer::draw`].
    fn draw_indirect(&mut self, ctx: &mut GraphicsContext<'_>, draw: &IndirectDraw);

    /// Draws `draw` into the render targets `ctx.regs` describes, sampling the texture and
    /// sampler pools it binds. State handling is the same as for [`Rasterizer::draw`].
    fn draw_texture(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawTexture) {}

    /// Clears the render targets `regs.clear_surface` selects, over `layer_count` layers.
    fn clear(&mut self, ctx: &mut GraphicsContext<'_>, layer_count: u32);

    /// Records one compute dispatch decoded from a launch descriptor. The context carries the
    /// whole launch, so compute has no dirty tracking; the renderer caches pipelines by
    /// `ctx.program` itself.
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

    /// Fills `size` bytes of a linear buffer at `dst` with the 32-bit `value` repeated (a copy
    /// engine constant fill). `false` makes the GPU write guest memory itself.
    fn accelerate_buffer_clear(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        _dst: GpuVAddr,
        _size: u64,
        _value: u32,
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

    /// Writes a query result or semaphore release. With `after_work` the write may land after
    /// the call returns, but before the next `release_fences(true)` returns.
    fn query(&mut self, channel: ChannelId, memory: &Gmmu, query: &Query);

    /// Resets a counter [`QueryKind`] reads from.
    fn reset_counter(&mut self, _kind: QueryKind) {}

    /// Runs `action` once the host GPU finished the work submitted so far (or at once when
    /// fences are not tracked).
    fn signal_fence(&mut self, action: FenceAction);

    /// Runs the actions of finished fences; `wait` blocks until all of them finished and ran,
    /// and deferred [`Query`] writes landed.
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

    /// The guest asked for a fragment or tiled cache barrier. Treat it as a full pipeline
    /// barrier: later draws and dispatches may read anything earlier work wrote.
    fn barrier(&mut self) {}

    /// Submits recorded host GPU work (end of a GPFIFO submission).
    fn flush_commands(&mut self) {}

    /// Block until the host GPU finished all submitted work.
    fn wait_for_idle(&mut self);

    /// Presents a guest frame (scaled, filtered and anti-aliased per settings).
    fn present(&mut self, framebuffer: &FramebufferConfig);
}

/// A renderer that draws nothing, for headless runs and tests that build a real GPU. Every fence
/// runs at once, and queries write their release as soon as they are made: payloads write the
/// payload and counters write 0.
pub struct NullRasterizer {
    handles: GpuHandles,
}

impl NullRasterizer {
    /// A renderer whose fences and releases act on `handles`.
    pub fn new(handles: GpuHandles) -> Self {
        Self { handles }
    }
}

impl Rasterizer for NullRasterizer {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}

    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}

    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}

    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}

    fn query(&mut self, _channel: ChannelId, memory: &Gmmu, query: &Query) {
        let Some(address) = memory.translate(query.address) else {
            tracing::warn!(address = query.address, "query to unmapped GPU memory");
            return;
        };
        let value = match query.kind {
            QueryKind::Payload => query.payload,
            _ => 0,
        };
        FenceAction::WriteSemaphore {
            address,
            value,
            size: query.size,
        }
        .execute(&self.handles);
    }

    fn signal_fence(&mut self, action: FenceAction) {
        action.execute(&self.handles);
    }

    fn release_fences(&mut self, _wait: bool) {}

    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}

    fn flush_region(&mut self, _addr: DAddr, _size: u64) {}

    fn wait_for_idle(&mut self) {}

    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}

/// The 32-bit semaphore at `address` as an atomic, so reductions running on different threads
/// never lose an update.
/// Writes the `words` of a release that landed in a second page to `target` in the region's
/// first `page`. A word with releases of its own keeps its value in memory, and acquires see the
/// mirrored one beside it.
fn mirror_release(handles: &GpuHandles, mut page: FirstPage<'_>, target: DAddr, words: &[u32]) {
    let mut kept = [None; 4];
    let mut written = 0;
    for (index, &word) in words.iter().enumerate() {
        let address = target + 4 * index as u64;
        if !page.contains(address) {
            break;
        }
        if !page.diverts(address) {
            if let Err(fault) = handles.device.write(address, word) {
                tracing::warn!(
                    ?fault,
                    "mirrored semaphore release to unmapped device memory"
                );
                break;
            }
        } else if let Ok(guest) = handles.device.read::<u32>(address) {
            page.keep(address, word, guest);
            kept[index] = Some(word);
        }
        written += 4;
    }
    drop(page);
    let waiters = handles.syncpoints.semaphore_waiters();
    waiters.written(&handles.device, target, written);
    for (index, value) in kept.into_iter().enumerate() {
        if let Some(value) = value {
            waiters.met_by(target + 4 * index as u64, value);
        }
    }
}

fn semaphore_word(device: &DeviceAddressSpace, address: DAddr) -> Option<&AtomicU32> {
    if !address.is_multiple_of(4) {
        return None;
    }
    let ptr = device.translate(address)?;
    // SAFETY: `translate` points into a mapped page of guest memory that outlives `device`; the
    // aligned word lies inside that page, and guest memory is only accessed as plain bytes or
    // atomics, never through Rust references.
    Some(unsafe { AtomicU32::from_ptr(ptr.cast()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::GuestTime;
    use lichii_memory::{AddressSpace, MemoryPermission, PhysicalMemory};

    struct Boot;

    impl GuestTime for Boot {
        fn global_time_ns(&self) -> u64 {
            1_000_000_000
        }
    }

    fn handles() -> (AddressSpace, GpuHandles) {
        let physical = Arc::new(PhysicalMemory::new(1 << 20).unwrap());
        let space = AddressSpace::new(physical.clone(), 39);
        space.map(0x1000_0000, 0, 0x1000, MemoryPermission::READ_WRITE);
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        device
            .map_process(0x10_0000, &space, 0x1000_0000, 0x1000)
            .unwrap();
        let handles = GpuHandles {
            device,
            syncpoints: Arc::new(Syncpoints::new()),
            timestamp: Arc::new(GpuTimestamp::new(Arc::new(Boot), false)),
            written_pages: Arc::new(GpuWrittenPages::new(36)),
        };
        (space, handles)
    }

    #[test]
    fn sixteen_byte_release_carries_the_gpu_timestamp() {
        let (_space, h) = handles();
        FenceAction::WriteSemaphore {
            address: 0x10_0000,
            value: 7,
            size: ReleaseSize::Sixteen,
        }
        .execute(&h);
        assert_eq!(
            h.device.read::<[u64; 2]>(0x10_0000).unwrap(),
            [7, 614_400_000]
        );
    }

    #[test]
    fn reduction_combines_with_the_value_at_execution() {
        let (_space, h) = handles();
        FenceAction::WriteSemaphore {
            address: 0x10_0000,
            value: 10,
            size: ReleaseSize::Four,
        }
        .execute(&h);
        FenceAction::ReduceSemaphore {
            address: 0x10_0000,
            op: Reduction::Add,
            signed: false,
            payload: 5,
        }
        .execute(&h);
        assert_eq!(h.device.read::<u32>(0x10_0000).unwrap(), 15);
    }

    const FIRST: DAddr = 0x10_0000;

    fn handles_with_region() -> (AddressSpace, GpuHandles) {
        let (space, handles) = handles();
        space.map(0x1000_1000, 0x1000, 0x1000, MemoryPermission::READ_WRITE);
        handles
            .device
            .map_process(FIRST + 0x1000, &space, 0x1000_1000, 0x1000)
            .unwrap();
        handles.syncpoints.semaphore_mirror().add_region(FIRST);
        (space, handles)
    }

    fn release(h: &GpuHandles, address: DAddr, value: u64, size: ReleaseSize) {
        FenceAction::WriteSemaphore {
            address,
            value,
            size,
        }
        .execute(h);
    }

    fn reduce(h: &GpuHandles, address: DAddr, op: Reduction, payload: u32) {
        FenceAction::ReduceSemaphore {
            address,
            op,
            signed: false,
            payload,
        }
        .execute(h);
    }

    fn acquire_sees(h: &GpuHandles, address: DAddr) -> u32 {
        h.syncpoints
            .semaphore_mirror()
            .acquire_view(address, || h.device.read::<u32>(address))
            .unwrap()
    }

    #[test]
    fn wide_mirrors_keep_each_word_with_releases_of_its_own() {
        let (_space, h) = handles_with_region();
        release(&h, FIRST + 4, 9, ReleaseSize::Four);
        release(&h, FIRST + 0x1000, 6 << 32 | 5, ReleaseSize::Eight);
        assert_eq!(h.device.read::<[u32; 2]>(FIRST + 0x1000).unwrap(), [5, 6]);
        assert_eq!(h.device.read::<[u32; 2]>(FIRST).unwrap(), [5, 9]);
        assert_eq!(acquire_sees(&h, FIRST), 5);
        assert_eq!(acquire_sees(&h, FIRST + 4), 6);

        release(&h, FIRST + 0x1010, 7, ReleaseSize::Sixteen);
        assert_eq!(
            h.device.read::<[u64; 2]>(FIRST + 0x10).unwrap(),
            h.device.read::<[u64; 2]>(FIRST + 0x1010).unwrap()
        );
    }

    #[test]
    fn second_page_reductions_reach_the_first_page() {
        let (_space, h) = handles_with_region();
        reduce(&h, FIRST + 0x1010, Reduction::Max, 5);
        assert_eq!(h.device.read::<u32>(FIRST + 0x10).unwrap(), 5);

        release(&h, FIRST + 0x20, 9, ReleaseSize::Four);
        reduce(&h, FIRST + 0x1020, Reduction::Add, 4);
        assert_eq!(h.device.read::<u32>(FIRST + 0x20).unwrap(), 9);
        assert_eq!(acquire_sees(&h, FIRST + 0x20), 4);
    }

    #[test]
    fn acquires_polled_during_reductions_keep_the_mirrored_value() {
        let (_space, h) = handles_with_region();
        release(&h, FIRST, 0, ReleaseSize::Four);
        release(&h, FIRST + 0x1000, 10, ReleaseSize::Four);
        assert_eq!(acquire_sees(&h, FIRST), 10);
        let done = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    acquire_sees(&h, FIRST);
                }
            });
            scope.spawn(|| {
                for _ in 0..2000 {
                    reduce(&h, FIRST, Reduction::Add, 1);
                }
                done.store(true, Ordering::Release);
            });
        });
        assert_eq!(h.device.read::<u32>(FIRST).unwrap(), 2000);
        assert_eq!(acquire_sees(&h, FIRST), 2010);
    }

    #[test]
    fn concurrent_second_page_reductions_mirror_the_final_value() {
        let (_space, h) = handles_with_region();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        reduce(&h, FIRST + 0x1008, Reduction::Add, 1);
                    }
                });
            }
        });
        assert_eq!(h.device.read::<u32>(FIRST + 0x1008).unwrap(), 4000);
        assert_eq!(h.device.read::<u32>(FIRST + 8).unwrap(), 4000);
    }

    #[test]
    fn concurrent_reductions_lose_no_update() {
        let (_space, h) = handles();
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..1000 {
                        FenceAction::ReduceSemaphore {
                            address: 0x10_0004,
                            op: Reduction::Add,
                            signed: false,
                            payload: 1,
                        }
                        .execute(&h);
                    }
                });
            }
        });
        assert_eq!(h.device.read::<u32>(0x10_0004).unwrap(), 4000);
    }

    #[test]
    fn null_rasterizer_writes_releases_and_runs_fences_at_once() {
        let (_space, h) = handles();
        let memory = Gmmu::new(h.device.clone(), crate::gmmu::BigPageSize::Size64K);
        memory
            .map(0x2000_0000, 0x10_0000, 0x1000, 0, false)
            .unwrap();
        let mut null = NullRasterizer::new(h.clone());
        let query = |address, kind, size| Query {
            address,
            kind,
            payload: 0x1234_5678_9abc,
            size,
            after_work: true,
        };
        null.query(
            ChannelId(0),
            &memory,
            &query(0x2000_0000, QueryKind::Payload, ReleaseSize::Eight),
        );
        assert_eq!(h.device.read::<u64>(0x10_0000).unwrap(), 0x1234_5678_9abc);
        null.query(
            ChannelId(0),
            &memory,
            &query(0x2000_0010, QueryKind::SamplesPassed, ReleaseSize::Sixteen),
        );
        assert_eq!(
            h.device.read::<[u64; 2]>(0x10_0010).unwrap(),
            [0, 614_400_000]
        );
        null.query(
            ChannelId(0),
            &memory,
            &query(0x3000_0000, QueryKind::Payload, ReleaseSize::Four),
        );
        null.signal_fence(FenceAction::IncrementSyncpoint(3));
        assert_eq!(h.syncpoints.guest().value(3).unwrap(), 1);
    }
}
