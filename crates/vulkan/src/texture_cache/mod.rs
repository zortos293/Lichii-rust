//! The texture cache: host images for guest images, found by guest address and layout.
//!
//! Every image is backed by device memory ranges (`Segment`s), so the same memory mapped at
//! two GPU addresses or in two channels shares one host image, and CPU writes, flushes and
//! unmaps, which name device addresses, find the images they touch through a page index.
//!
//! Lookups first try a per-address-space table keyed by the GPU address and the guest layout,
//! validated by the GMMU's mapping generation: a hit costs one hash lookup and allocates
//! nothing. A miss walks the mapping, reuses an image of the same layout (or one that holds the
//! requested image as a level/layer run), and otherwise creates a new image. Images that fit as
//! subresources of the new one are copied into it on the GPU. Other overlapping images stay
//! live beside it as overlaps: whichever was modified later owns the shared guest bytes, so
//! write-backs leave those bytes to it, and an image whose shared bytes a later overlap wrote
//! takes them through guest memory only when it is next used without being overwritten whole.
//! A transient target the guest clears before each use never round-trips.
//! Pitch-linear images with the same pitch, such as a render target and a narrower texture over
//! it, stay live side by side as aliases: whichever the GPU wrote last is copied into the other
//! on the GPU when that one is next used, so neither round-trips through guest memory.
//!
//! Images the guest CPU wrote are re-uploaded on their next use; images the GPU wrote are
//! written back to guest memory when a flush asks. Render targets that may scale (Lichii's
//! rules: 2D, one level, taller than 288 rows, or 512 for downscaling) render at the configured
//! resolution scale and are scaled back down on flush.
//!
//! Once per presented frame [`TextureCache::tick_frame`] runs the collector: past thresholds
//! from the driver's memory budget and the VRAM usage setting, it evicts the least recently
//! used images with bounded work per frame. GPU-modified ones are read back without a GPU wait
//! and reach guest memory before they go.

mod blit;
mod clear;
mod convert;
mod guest_sync;
mod info;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_common::settings::{AstcRecompression, ResolutionScale, VramUsage};
use lichii_common::{DAddr, GpuVAddr, PAGE_SIZE};
use lichii_gpu::backend::{FramebufferConfig, GpuWrittenPages};
use lichii_gpu::gmmu::{Gmmu, RangeTarget};
use lichii_gpu::lazy::LazyDeviceRanges;
use lichii_memory::{CachedPages, DeviceAddressSpace};
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::{GobLayout, Tiling, swizzle, unswizzle};
use lichii_texture::tic::{SwizzleSource, TextureDescriptor};

pub use blit::BlitSide;
pub use info::{GuestLayout, ImageInfo, ImageKind, ViewInfo};

use crate::commands::{ImageAccess, transfer_to_all_barrier, transition};
use crate::device::Device;
use crate::error::Result;
use crate::format::{Conversion, HostFormat, host_format};
use crate::memory::{Buffer, Image, ImageDesc, ImageView, MemoryUsage, aspect_of};
use crate::scheduler::Scheduler;
use crate::staging::StagingRing;
use clear::ClearPipelines;

/// Device memory is indexed in pages of this many bits (Lichii's texture cache uses 1 MiB).
const PAGE_BITS: u32 = 20;
/// Images with more texels are created sparse with no memory, as an 8192x8192x68 virtual
/// texture would take ~23 GB (NX2_NOTES row 87): the subresources the GPU writes get memory,
/// and nothing is uploaded or written back.
const MAX_TEXELS: u64 = 1 << 28;
/// Device memory one sparse image may bind by default; writes to subresources past it are
/// dropped.
const SPARSE_IMAGE_CAP: u64 = 2 << 30;
/// The lookup table is emptied when it grows past this many entries.
const MAX_LOOKUPS: usize = 1 << 16;

/// No neighbour in the LRU list.
const NIL: u32 = u32::MAX;
/// Frames an image must go unused before the collector may evict it, while device memory use
/// is under the retention line (images of the last few seconds stay) and past it.
const OLD_AGE: u64 = 150;
const RECENT_AGE: u64 = 10;
/// Images the collector evicts per frame at most, in normal and aggressive mode; it looks at
/// [`VISITS_PER_EVICTION`] times as many at most.
const NORMAL_EVICTIONS: u32 = 20;
const AGGRESSIVE_EVICTIONS: u32 = 40;
const VISITS_PER_EVICTION: u32 = 8;
/// Readback bytes of GPU-modified images the collector starts flushing per frame (always at
/// least one image).
const MAX_FLUSH_BYTES: u64 = 16 << 20;

/// Depth/stencil attachment access for clears through dynamic rendering.
pub(crate) const DEPTH_ATTACHMENT: ImageAccess = ImageAccess {
    layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    stage: vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
    ),
    access: vk::AccessFlags2::from_raw(
        vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ.as_raw()
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE.as_raw(),
    ),
};

/// A cached image. Ids stay unique: a deleted image's id never names a later image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageId {
    index: u32,
    generation: u32,
}

/// A cached image holding the requested guest image, which starts at `level` and `layer` of
/// it (both 0 unless the request was a level/layer run of a larger image).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Found {
    pub id: ImageId,
    pub level: u32,
    pub layer: u32,
}

/// What cache operations record into and read guest memory through.
pub struct Transfer<'a> {
    pub scheduler: &'a mut Scheduler,
    pub staging: &'a mut StagingRing,
    pub memory: &'a DeviceAddressSpace,
}

/// A rectangle in guest pixels of a render target (before resolution scaling).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    /// Everything: clipped to the target by the clear.
    pub const ALL: Self = Self {
        x: 0,
        y: 0,
        width: u32::MAX,
        height: u32::MAX,
    };
}

/// What a clear writes.
#[derive(Clone, Copy)]
pub enum ClearValue {
    /// A colour in the target's numeric type, written to the components in `mask`.
    Color {
        value: vk::ClearColorValue,
        mask: vk::ColorComponentFlags,
    },
    /// Depth and stencil values; `None` keeps that aspect, and only the stencil bits set in
    /// `stencil_mask` are written.
    DepthStencil {
        depth: Option<f32>,
        stencil: Option<u8>,
        stencil_mask: u8,
    },
}

/// A cached image's state as other renderer parts see it.
#[derive(Clone, Copy, Debug)]
pub struct ImageState {
    pub info: ImageInfo,
    pub format: vk::Format,
    /// Host size of level 0, scaled.
    pub extent: vk::Extent3D,
    pub scale: ResolutionScale,
    pub gpu_modified: bool,
    pub cpu_dirty: bool,
}

/// Guest texture data the cache moved since it was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    /// Images uploaded from guest memory, and the staged bytes.
    pub uploads: u64,
    pub upload_bytes: u64,
    /// GPU-modified images written back to guest memory, and their guest bytes.
    pub downloads: u64,
    pub download_bytes: u64,
    /// Overlap syncs done on the GPU instead of through guest memory.
    pub gpu_syncs: u64,
}

/// What the collector did since the cache was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectorStats {
    /// Images evicted, and the device memory they held.
    pub evicted: u64,
    pub evicted_bytes: u64,
    /// Evicted images that were GPU-modified and written back to guest memory first.
    pub flushed: u64,
}

/// Device memory use past which the collector runs (`expected`), turns aggressive
/// (`critical`), and also evicts images used in the last [`OLD_AGE`] frames (`retain`). All
/// follow the driver's budget (NX2_NOTES row 119) less a margin of an eighth of it, at most
/// 1 GiB. Conservative VRAM use also caps the critical line at 5/8 of the budget or 5 GiB,
/// whichever is more, as Lichii does: past what a frame needs, more memory only keeps earlier
/// scenes' textures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Thresholds {
    expected: u64,
    critical: u64,
    retain: u64,
}

impl Thresholds {
    fn new(budget: u64, vram_usage: VramUsage) -> Self {
        let margin = (budget / 8).min(1 << 30);
        let retain = budget - margin;
        let critical = match vram_usage {
            VramUsage::Conservative => retain.min((budget / 8 * 5).max(5 << 30)),
            VramUsage::Aggressive => retain,
        };
        Self {
            expected: critical - margin.min(critical / 2),
            critical,
            retain,
        }
    }
}

/// A copy of an image into a readback buffer at `offset`; `modified` is the image's last GPU
/// write when it was recorded.
#[derive(Clone, Copy)]
struct Download {
    id: ImageId,
    offset: u64,
    modified: u64,
}

/// The collector's readback of GPU-modified images, submitted at `tick` in frame `frame`,
/// holding `bytes` of device memory.
#[derive(Clone, Copy)]
struct PendingEvictions {
    tick: u64,
    frame: u64,
    bytes: u64,
}

/// A run of the image's guest bytes that is contiguous in device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    offset: u64,
    daddr: DAddr,
    size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ViewKey {
    format: i32,
    view_type: i32,
    components: [i32; 4],
    aspect: u32,
    base_level: u32,
    levels: u32,
    base_layer: u32,
    layers: u32,
}

struct CachedImage {
    info: ImageInfo,
    host: HostFormat,
    image: Image,
    scale: ResolutionScale,
    access: ImageAccess,
    segments: Vec<Segment>,
    views: Vec<(ViewKey, ImageView)>,
    /// Guest memory changed since the last upload.
    cpu_dirty: bool,
    /// Write tracking over the segments the last upload read, so a guest CPU write to them
    /// invalidates the image. Empty before the first upload and after an invalidation.
    tracked: Vec<CachedPages>,
    /// The host image holds data guest memory does not.
    gpu_modified: bool,
    /// Order of the last GPU write, for aliasing.
    modified: u64,
    visit: u64,
    /// Frame of the last use, for the collector.
    last_use: u64,
    /// Live images over some of the same texels, kept in step with GPU copies.
    aliases: Vec<Alias>,
    /// Tick of the last GPU work that wrote this image itself rather than copied from an alias.
    written: u64,
    /// Live images over some of the same guest bytes that neither can copy texels from.
    overlaps: Vec<ImageId>,
    /// Tick up to which the image holds what its overlaps wrote: its last GPU write, upload or
    /// whole overwrite. An overlap modified after it makes the shared texels stale.
    current: u64,
    /// Device ranges of removed overlaps modified after this image: write-backs skip them.
    keep_out: Vec<(DAddr, DAddr)>,
}

/// A live image that shares a rectangle of texels with another, at level 0 of layer 0:
/// `at` in the alias, `own` in the image that lists it. `synced` is the alias's `written`
/// tick when the rectangle last took its texels.
#[derive(Clone, Copy, Debug)]
struct Alias {
    id: ImageId,
    at: Texel,
    own: Texel,
    extent: Texel,
    synced: u64,
}

/// A texel position or extent: x and y, or width and height.
type Texel = (u32, u32);

impl CachedImage {
    fn base(&self) -> DAddr {
        self.segments[0].daddr
    }

    /// Tracks the image's pages for guest CPU writes, before the cache relies on their bytes.
    fn track(&mut self, memory: &DeviceAddressSpace) {
        self.tracked = self
            .segments
            .iter()
            .map(|s| memory.cache_pages(s.daddr, s.size))
            .collect();
    }

    /// Whether the guest format can be written back from the host image.
    fn downloadable(&self) -> bool {
        !self.image.is_sparse()
            && !matches!(
                self.host.conversion,
                Conversion::Decode | Conversion::DecodeToBc(_)
            )
    }

    /// Bytes a native-resolution readback of the image takes, and their alignment.
    fn readback_size(&self) -> (u64, u64) {
        let format = self.info.format;
        let align = alignment(&self.host, format);
        let mut total = 0u64;
        for _ in 0..self.info.layers {
            for level in 0..self.image.desc().mip_levels {
                let layout =
                    convert::host_layout(format, &self.host, self.info.level_extent(level));
                total = total.next_multiple_of(align) + layout.total() as u64;
            }
        }
        (total, align)
    }

    fn contiguous(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].size == self.info.size_bytes()
    }

    fn overlaps(&self, daddr: DAddr, size: u64) -> bool {
        self.segments
            .iter()
            .any(|s| s.daddr < daddr + size && daddr < s.daddr + s.size)
    }

    fn host_extent(&self, level: u32) -> vk::Extent3D {
        let e = self.image.desc().extent;
        vk::Extent3D {
            width: (e.width >> level).max(1),
            height: (e.height >> level).max(1),
            depth: (e.depth >> level).max(1),
        }
    }

    fn transition(&mut self, device: &Device, cmd: vk::CommandBuffer, to: ImageAccess) {
        transition(
            device,
            cmd,
            self.image.raw(),
            self.image.full_range(),
            self.access,
            to,
        );
        self.access = to;
    }
}

/// FxHash: lookup keys are small and come from the guest, so a multiply-rotate hash beats
/// SipHash on the hot path and needs no flooding resistance.
#[derive(Default)]
struct FxHasher(u64);

impl FxHasher {
    fn add(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(word));
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type FxBuild = BuildHasherDefault<FxHasher>;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct LookupKey {
    gmmu: usize,
    gpu_va: GpuVAddr,
    info: ImageInfo,
}

#[derive(Clone, Copy)]
struct LookupEntry {
    generation: u64,
    found: Found,
}

struct Slot {
    generation: u32,
    image: Option<CachedImage>,
    /// Neighbours in the LRU list, least recently used first.
    prev: u32,
    next: u32,
}

/// Scratch memory kept between calls, so steady-state uploads and downloads allocate nothing.
#[derive(Default)]
struct Scratch {
    tiled: Vec<u8>,
    linear: Vec<u8>,
    convert: Vec<u8>,
    regions: Vec<vk::BufferImageCopy>,
}

/// Host images for guest images. Owned by the rasterizer on the GPU thread.
pub struct TextureCache {
    device: Arc<Device>,
    /// Device pages GPU work wrote, marked with every segment of an image the GPU modifies.
    written_pages: Arc<GpuWrittenPages>,
    /// Device ranges that get memory as they are written: flushes go through them.
    lazy: Arc<LazyDeviceRanges>,
    gob_layout: GobLayout,
    recompression: AstcRecompression,
    scale: ResolutionScale,
    slots: Vec<Slot>,
    free: Vec<u32>,
    pages: HashMap<u64, Vec<ImageId>>,
    lookup: HashMap<LookupKey, LookupEntry, FxBuild>,
    host_formats: Vec<Option<HostFormat>>,
    tick: u64,
    visit: u64,
    traffic: Traffic,
    /// Images removed so far; see [`TextureCache::removals`].
    removals: u64,
    ids: Vec<ImageId>,
    joins: Vec<(ImageId, u32, u32)>,
    new_aliases: Vec<Alias>,
    alias_scratch: Vec<Alias>,
    new_overlaps: Vec<ImageId>,
    dropped: Vec<ImageId>,
    overlap_scratch: Vec<ImageId>,
    /// A sparse image hit its memory cap (warned once).
    sparse_capped: bool,
    /// Device memory one sparse image may bind.
    sparse_cap: u64,
    keep: Vec<(DAddr, DAddr)>,
    segments: Vec<Segment>,
    scratch: Scratch,
    readback: Option<Buffer>,
    downloads: Vec<Download>,
    clear_pipelines: Option<ClearPipelines>,
    guest_sync: Option<guest_sync::GuestSync>,
    vram_usage: VramUsage,
    /// Frames presented so far.
    frame: u64,
    lru_head: u32,
    lru_tail: u32,
    /// Device memory of evicted images the GPU may still use, by the tick that frees it.
    sentenced: VecDeque<(u64, u64)>,
    sentenced_bytes: u64,
    pending: Option<PendingEvictions>,
    pending_downloads: Vec<Download>,
    eviction_readback: Option<Buffer>,
    stats: CollectorStats,
}

/// A depth or stencil view returns its value in R and leaves G undefined, while guest
/// swizzles of depth formats (S8D24 among them) read the depth from G.
fn sampled_swizzle(
    swizzle: [SwizzleSource; 4],
    aspect: vk::ImageAspectFlags,
) -> [SwizzleSource; 4] {
    if !aspect.intersects(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL) {
        return swizzle;
    }
    swizzle.map(|s| match s {
        SwizzleSource::G => SwizzleSource::R,
        s => s,
    })
}

fn components(swizzle: [SwizzleSource; 4], host: vk::ComponentMapping) -> vk::ComponentMapping {
    use vk::ComponentSwizzle as C;
    let resolve = |c: C, own: C| if c == C::IDENTITY { own } else { c };
    let hosts = [
        resolve(host.r, C::R),
        resolve(host.g, C::G),
        resolve(host.b, C::B),
        resolve(host.a, C::A),
    ];
    let map = |s: SwizzleSource| match s {
        SwizzleSource::Zero => C::ZERO,
        SwizzleSource::R => hosts[0],
        SwizzleSource::G => hosts[1],
        SwizzleSource::B => hosts[2],
        SwizzleSource::A => hosts[3],
        SwizzleSource::OneInt | SwizzleSource::OneFloat => C::ONE,
    };
    vk::ComponentMapping {
        r: map(swizzle[0]),
        g: map(swizzle[1]),
        b: map(swizzle[2]),
        a: map(swizzle[3]),
    }
}

/// The rectangle two single-level 2D block-linear images at the same address share at level 0:
/// the same as [`pitch_overlap`], for images whose level 0 tiles alike (equal GOBs per row,
/// block height and GOB layout) and so differ at most in width and height.
fn block_linear_overlap(
    a: &ImageInfo,
    a_base: DAddr,
    b: &ImageInfo,
    b_base: DAddr,
) -> Option<(Texel, Texel, Texel)> {
    let tiling = |i: &ImageInfo| {
        let single = matches!(i.layout, GuestLayout::BlockLinear { .. })
            && i.format.block_width() == 1
            && i.format.block_height() == 1
            && i.kind == ImageKind::D2
            && i.depth == 1
            && i.levels == 1
            && i.layers == 1
            && i.samples == 1;
        single.then(|| i.surface().level_tiling(0))
    };
    let (ta, tb) = (tiling(a)?, tiling(b)?);
    let shape = |t: &Tiling| {
        (
            t.bytes_per_block,
            t.block_height_log2,
            t.block_depth_log2,
            t.row_gobs,
            t.layout,
        )
    };
    let same_encoding = a.format.is_color() || a.format == b.format;
    if a_base != b_base || shape(&ta) != shape(&tb) || !same_encoding {
        return None;
    }
    Some((
        (0, 0),
        (0, 0),
        (a.width.min(b.width), a.height.min(b.height)),
    ))
}

/// Where two pitch-linear images with the same pitch share texels: the rectangle's texel
/// offset in `a`, in `b`, and its extent, which is empty when they share only padding.
/// `None` when they cannot alias: other layouts, pitches or texel sizes, or rows of the later
/// image that run past the earlier one's pitch.
fn pitch_overlap(
    a: &ImageInfo,
    a_base: DAddr,
    b: &ImageInfo,
    b_base: DAddr,
) -> Option<(Texel, Texel, Texel)> {
    let (GuestLayout::Pitch { pitch }, GuestLayout::Pitch { pitch: other }) = (a.layout, b.layout)
    else {
        return None;
    };
    let bytes = a.format.bytes_per_block();
    let single = |i: &ImageInfo| {
        i.format.block_width() == 1
            && i.format.block_height() == 1
            && i.kind == ImageKind::D2
            && i.levels == 1
            && i.layers == 1
            && i.samples == 1
    };
    if pitch != other
        || pitch == 0
        || bytes != b.format.bytes_per_block()
        || !single(a)
        || !single(b)
    {
        return None;
    }
    let (first, second) = if a_base <= b_base { (a, b) } else { (b, a) };
    let delta = a_base.abs_diff(b_base);
    let column = delta % u64::from(pitch);
    if !column.is_multiple_of(u64::from(bytes)) {
        return None;
    }
    let x = u32::try_from(column / u64::from(bytes)).ok()?;
    let y = u32::try_from(delta / u64::from(pitch)).ok()?;
    if x + second.width > pitch / bytes {
        return None;
    }
    let extent = (
        first.width.saturating_sub(x).min(second.width),
        first.height.saturating_sub(y).min(second.height),
    );
    Some(if a_base <= b_base {
        ((x, y), (0, 0), extent)
    } else {
        ((0, 0), (x, y), extent)
    })
}

/// Reads guest bytes `[offset, offset + out.len())` of an image; unmapped pages read zero, and
/// are only reported outside lazy device ranges, whose pages without memory are zero.
fn read_guest(
    memory: &DeviceAddressSpace,
    lazy: &LazyDeviceRanges,
    segments: &[Segment],
    offset: u64,
    out: &mut [u8],
) {
    out.fill(0);
    let end = offset + out.len() as u64;
    for s in segments {
        let (start, stop) = (s.offset.max(offset), (s.offset + s.size).min(end));
        if start >= stop {
            continue;
        }
        let chunk = &mut out[(start - offset) as usize..(stop - offset) as usize];
        let daddr = s.daddr + (start - s.offset);
        if memory.read_bytes(daddr, chunk).is_ok() {
            continue;
        }
        let mut unmapped = None;
        let mut done = 0;
        while done < chunk.len() {
            let addr = daddr + done as u64;
            let len = ((PAGE_SIZE - addr % PAGE_SIZE) as usize).min(chunk.len() - done);
            let page = &mut chunk[done..done + len];
            if memory.read_bytes(addr, page).is_err() {
                page.fill(0);
                if !lazy.is_lazy(addr, len as u64) {
                    unmapped.get_or_insert(addr);
                }
            }
            done += len;
        }
        if let Some(daddr) = unmapped {
            tracing::warn!(daddr, "texture reads unmapped device memory");
        }
    }
}

/// Writes guest bytes of an image, except those at device addresses in `keep`; bytes over
/// unmapped gaps are dropped, and blocks of lazy device ranges get memory.
fn write_guest(
    lazy: &LazyDeviceRanges,
    segments: &[Segment],
    offset: u64,
    data: &[u8],
    keep: &[(DAddr, DAddr)],
) {
    let end = offset + data.len() as u64;
    let write = |daddr: DAddr, chunk: &[u8]| {
        if let Err(fault) = lazy.write(daddr, chunk) {
            tracing::warn!(?fault, "texture flush to unmapped device memory");
        }
    };
    for s in segments {
        let (start, stop) = (s.offset.max(offset), (s.offset + s.size).min(end));
        if start >= stop {
            continue;
        }
        let chunk = &data[(start - offset) as usize..(stop - offset) as usize];
        let daddr = s.daddr + (start - s.offset);
        let chunk_end = daddr + chunk.len() as u64;
        let mut from = daddr;
        while from < chunk_end {
            let next = keep
                .iter()
                .filter(|&&(k0, k1)| k0 < chunk_end && k1 > from)
                .min_by_key(|k| k.0);
            let to = next.map_or(chunk_end, |k| k.0.max(from));
            if to > from {
                write(from, &chunk[(from - daddr) as usize..(to - daddr) as usize]);
            }
            match next {
                Some(&(_, k1)) => from = k1,
                None => break,
            }
        }
    }
}

fn pages(daddr: DAddr, size: u64) -> std::ops::RangeInclusive<u64> {
    (daddr >> PAGE_BITS)..=((daddr + size.max(1) - 1) >> PAGE_BITS)
}

fn scaled(scale: ResolutionScale, value: u32) -> u32 {
    let (num, den) = scale.ratio();
    (u64::from(value) * u64::from(num) / u64::from(den)) as u32
}

fn aspects_of(format: vk::Format) -> [Option<vk::ImageAspectFlags>; 2] {
    let aspects = aspect_of(format);
    if aspects == vk::ImageAspectFlags::COLOR {
        return [Some(aspects), None];
    }
    [
        aspects
            .contains(vk::ImageAspectFlags::DEPTH)
            .then_some(vk::ImageAspectFlags::DEPTH),
        aspects
            .contains(vk::ImageAspectFlags::STENCIL)
            .then_some(vk::ImageAspectFlags::STENCIL),
    ]
}

impl TextureCache {
    /// `gob_layout` is the tiling block-linear guest images use (`Switch2` for native Switch 2
    /// titles); `recompression` picks the host format of ASTC textures the device cannot
    /// sample.
    pub fn new(
        device: &Arc<Device>,
        written_pages: Arc<GpuWrittenPages>,
        lazy: Arc<LazyDeviceRanges>,
        gob_layout: GobLayout,
        recompression: AstcRecompression,
    ) -> Self {
        Self {
            device: device.clone(),
            written_pages,
            lazy,
            gob_layout,
            recompression,
            scale: ResolutionScale::X1,
            slots: Vec::new(),
            free: Vec::new(),
            pages: HashMap::new(),
            lookup: HashMap::default(),
            host_formats: vec![None; PixelFormat::ALL.len()],
            tick: 0,
            visit: 0,
            traffic: Traffic::default(),
            removals: 0,
            ids: Vec::new(),
            joins: Vec::new(),
            new_aliases: Vec::new(),
            alias_scratch: Vec::new(),
            new_overlaps: Vec::new(),
            dropped: Vec::new(),
            overlap_scratch: Vec::new(),
            sparse_capped: false,
            sparse_cap: SPARSE_IMAGE_CAP,
            keep: Vec::new(),
            segments: Vec::new(),
            scratch: Scratch::default(),
            readback: None,
            downloads: Vec::new(),
            clear_pipelines: None,
            guest_sync: None,
            vram_usage: VramUsage::default(),
            frame: 0,
            lru_head: NIL,
            lru_tail: NIL,
            sentenced: VecDeque::new(),
            sentenced_bytes: 0,
            pending: None,
            pending_downloads: Vec::new(),
            eviction_readback: None,
            stats: CollectorStats::default(),
        }
    }

    pub fn gob_layout(&self) -> GobLayout {
        self.gob_layout
    }

    /// The scale render targets render at from now on. Images keep their scale until they are
    /// next bound as render targets.
    pub fn set_resolution_scale(&mut self, scale: ResolutionScale) {
        self.scale = scale;
    }

    pub fn resolution_scale(&self) -> ResolutionScale {
        self.scale
    }

    /// The host format ASTC textures created from now on decode to.
    pub fn set_astc_recompression(&mut self, recompression: AstcRecompression) {
        self.recompression = recompression;
        self.host_formats.fill(None);
    }

    /// How much of the device budget the collector lets images use.
    pub fn set_vram_usage(&mut self, vram_usage: VramUsage) {
        self.vram_usage = vram_usage;
    }

    pub fn collector_stats(&self) -> CollectorStats {
        self.stats
    }

    pub fn traffic(&self) -> Traffic {
        self.traffic
    }

    /// The host format of a guest format, queried from the device once.
    pub(crate) fn host(&mut self, format: PixelFormat) -> HostFormat {
        let slot = &mut self.host_formats[format as usize];
        *slot.get_or_insert_with(|| host_format(&self.device, format, self.recompression))
    }

    fn get(&self, id: ImageId) -> Option<&CachedImage> {
        let slot = self.slots.get(id.index as usize)?;
        (slot.generation == id.generation)
            .then_some(slot.image.as_ref())
            .flatten()
    }

    fn get_mut(&mut self, id: ImageId) -> Option<&mut CachedImage> {
        let slot = self.slots.get_mut(id.index as usize)?;
        (slot.generation == id.generation)
            .then_some(slot.image.as_mut())
            .flatten()
    }

    fn image(&mut self, id: ImageId) -> &mut CachedImage {
        self.get_mut(id).expect("a live image id")
    }

    /// The state of a cached image, `None` once it was deleted.
    pub fn state(&self, id: ImageId) -> Option<ImageState> {
        let image = self.get(id)?;
        Some(ImageState {
            info: image.info,
            format: image.host.format,
            extent: image.image.desc().extent,
            scale: image.scale,
            gpu_modified: image.gpu_modified,
            cpu_dirty: image.cpu_dirty,
        })
    }

    /// How many images have been removed so far. A lookup removes the overlapping images it
    /// cannot reuse, so ids taken before a lookup that changes this may name removed images.
    pub fn removals(&self) -> u64 {
        self.removals
    }

    /// Number of live images.
    pub fn len(&self) -> usize {
        self.slots.iter().filter(|s| s.image.is_some()).count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The image holding the guest image `info` describes at `gpu_va` in `gmmu`, created and
    /// uploaded when needed, and re-uploaded when the guest CPU wrote it. `None` when the
    /// address is unmapped or the image too large to back.
    pub fn find(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        info: &ImageInfo,
    ) -> Result<Option<Found>> {
        let found = self.lookup_or_create(tx, gmmu, gpu_va, info)?;
        if let Some(found) = found {
            self.refresh(tx, found.id)?;
        }
        Ok(found)
    }

    /// The image and view a texture header samples.
    pub fn texture(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        desc: &TextureDescriptor,
    ) -> Result<Option<(ImageId, ViewInfo)>> {
        let Some(info) = ImageInfo::from_texture(desc, self.gob_layout) else {
            return Ok(None);
        };
        let Some(found) = self.find(tx, gmmu, desc.address, &info)? else {
            return Ok(None);
        };
        let scale = self.image(found.id).scale;
        if scale != ResolutionScale::X1 && scale != self.scale {
            let target = if self.can_rescale(found.id) {
                self.scale
            } else {
                ResolutionScale::X1
            };
            self.rescale(tx, found.id, target, true)?;
        }
        let mut view = ViewInfo::from_texture(desc, &info);
        view.base_level += found.level;
        view.base_layer += found.layer;
        Ok(Some((found.id, view)))
    }

    /// Whether `id` is a sparse image nothing has written yet, which samples as no image.
    pub fn unwritten(&self, id: ImageId) -> bool {
        self.get(id)
            .is_some_and(|image| image.image.is_sparse() && image.image.memory_size() == 0)
    }

    /// The image a render target or depth target writes, at the scale the rules allow.
    /// `discard` skips uploading guest data the caller overwrites entirely (a full clear).
    pub fn render_target(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        info: &ImageInfo,
        discard: bool,
    ) -> Result<Option<Found>> {
        let Some(found) = self.lookup_or_create(tx, gmmu, gpu_va, info)? else {
            return Ok(None);
        };
        let whole = found.level == 0 && found.layer == 0 && {
            let image = self.image(found.id);
            image.info.levels == 1 && image.info.layers == info.layers
        };
        if discard && whole {
            let tick = self.tick;
            let image = self.image(found.id);
            image.cpu_dirty = false;
            image.current = tick;
        }
        self.refresh(tx, found.id)?;
        let target = if self.can_rescale(found.id) {
            self.scale
        } else {
            ResolutionScale::X1
        };
        self.rescale(tx, found.id, target, !(discard && whole))?;
        self.bind_written(
            tx,
            found.id,
            found.level,
            found.layer..found.layer + info.layers,
        )?;
        Ok(Some(found))
    }

    /// A view a shader samples: [`TextureCache::view`] with the guest swizzle a depth or stencil
    /// image needs on the host.
    pub fn sampled_view(&mut self, id: ImageId, view: &ViewInfo) -> Result<Option<vk::ImageView>> {
        let Some(image) = self.get(id) else {
            return Ok(None);
        };
        let view = ViewInfo {
            swizzle: sampled_swizzle(view.swizzle, aspect_of(image.host.format)),
            ..*view
        };
        self.view(id, &view)
    }

    /// A view of a cached image. `None` when the image cannot be read as `view.format`.
    pub fn view(&mut self, id: ImageId, view: &ViewInfo) -> Result<Option<vk::ImageView>> {
        if self.get(id).is_none() || !self.view_compatible(id, view.format) {
            return Ok(None);
        }
        let host = self.host(view.format);
        let image = self.get(id).expect("a live image id");
        let host = if view.format == image.info.format {
            image.host
        } else {
            host
        };
        let mapping = components(view.swizzle, host.swizzle);
        let aspect = match aspect_of(image.host.format) {
            a if a.contains(vk::ImageAspectFlags::DEPTH) => vk::ImageAspectFlags::DEPTH,
            a => a,
        };
        let key = ViewKey {
            format: host.format.as_raw(),
            view_type: view.view_type.as_raw(),
            components: [mapping.r, mapping.g, mapping.b, mapping.a].map(|c| c.as_raw()),
            aspect: aspect.as_raw(),
            base_level: view.base_level,
            levels: view.levels,
            base_layer: view.base_layer,
            layers: view.layers,
        };
        self.view_for(id, key, host.format, view.view_type, mapping)
            .map(Some)
    }

    /// A storage image view of `view`'s first level with the identity swizzle, as storage
    /// descriptors need. sRGB formats are stored through their linear twin, as the guest
    /// stores raw texels. `None` when the image or `view.format` cannot be stored to as is.
    pub fn storage_view(&mut self, id: ImageId, view: &ViewInfo) -> Result<Option<vk::ImageView>> {
        let format = view.format.to_linear();
        let host = self.host(format);
        let image_storage = self.get(id).is_some_and(|i| i.host.storage);
        if !image_storage
            || !host.storage
            || host.conversion != Conversion::None
            || [
                host.swizzle.r,
                host.swizzle.g,
                host.swizzle.b,
                host.swizzle.a,
            ]
            .iter()
            .any(|&c| c != vk::ComponentSwizzle::IDENTITY)
        {
            return Ok(None);
        }
        let view = ViewInfo {
            swizzle: [
                SwizzleSource::R,
                SwizzleSource::G,
                SwizzleSource::B,
                SwizzleSource::A,
            ],
            levels: 1,
            format,
            ..*view
        };
        self.view(id, &view)
    }

    /// Scales `id` back to native resolution, keeping its contents: Lichii does so for images a
    /// shader writes as storage images, which the shader cannot address at a scale.
    pub fn scale_down(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        if self.get(id).is_none() {
            return Ok(());
        }
        self.rescale(tx, id, ResolutionScale::X1, true)
    }

    fn view_for(
        &mut self,
        id: ImageId,
        key: ViewKey,
        format: vk::Format,
        view_type: vk::ImageViewType,
        mapping: vk::ComponentMapping,
    ) -> Result<vk::ImageView> {
        let device = self.device.clone();
        let image = self.image(id);
        if let Some((_, view)) = image.views.iter().find(|(k, _)| *k == key) {
            return Ok(view.raw());
        }
        let view = ImageView::new(
            &device,
            image.image.raw(),
            view_type,
            format,
            mapping,
            vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::from_raw(key.aspect),
                base_mip_level: key.base_level,
                level_count: key.levels,
                base_array_layer: key.base_layer,
                layer_count: key.layers,
            },
            image.image.view_usage(format),
        )?;
        let raw = view.raw();
        image.views.push((key, view));
        Ok(raw)
    }

    /// Records the barrier that makes the image ready for `access` and returns it; `None`, with
    /// a warning, once the image was removed.
    pub fn prepare(
        &mut self,
        cmd: vk::CommandBuffer,
        id: ImageId,
        access: ImageAccess,
    ) -> Option<vk::Image> {
        if self.get(id).is_none() {
            tracing::warn!(
                ?id,
                "preparing a removed texture cache image; skipping its barrier"
            );
            return None;
        }
        Some(self.prepare_live(cmd, id, access))
    }

    fn prepare_live(
        &mut self,
        cmd: vk::CommandBuffer,
        id: ImageId,
        access: ImageAccess,
    ) -> vk::Image {
        let device = self.device.clone();
        self.touch(id);
        let image = self.image(id);
        image.transition(&device, cmd, access);
        image.image.raw()
    }

    /// Marks a cached image as written by the GPU, and its device pages in the written-pages
    /// bitmap. Called when the writing work is recorded.
    pub fn mark_modified(&mut self, id: ImageId) {
        if let Some(image) = self.get(id) {
            for s in &image.segments {
                self.written_pages.mark(s.daddr, s.size);
            }
        }
        self.tick += 1;
        let tick = self.tick;
        let Some(image) = self.get_mut(id) else {
            return;
        };
        image.gpu_modified = true;
        image.cpu_dirty = false;
        image.modified = tick;
        image.written = tick;
        image.current = tick;
        image.keep_out.clear();
    }

    /// The guest CPU wrote `[daddr, daddr + size)`: overlapping images re-upload on next use.
    /// GPU-modified images first write their other texels back, so only the CPU's bytes replace
    /// GPU data.
    pub fn invalidate(&mut self, tx: &mut Transfer<'_>, daddr: DAddr, size: u64) -> Result<()> {
        self.collect_overlaps(&[Segment {
            offset: 0,
            daddr,
            size,
        }]);
        let mut ids = std::mem::take(&mut self.ids);
        let mut result = Ok(());
        for &id in &ids {
            if self.image(id).gpu_modified {
                let written_back =
                    self.download(tx, &[id], Some((daddr, daddr.saturating_add(size))));
                result = result.and(written_back);
            }
            let image = self.image(id);
            image.cpu_dirty = true;
            image.gpu_modified = false;
            image.tracked.clear();
        }
        ids.clear();
        self.ids = ids;
        result
    }

    /// Write tracking saw guest CPU writes in the pages of `[daddr, daddr + size)`: images
    /// that hold only guest data re-upload on next use. GPU-modified images keep their texels,
    /// which a write beside them on a shared page must not replace.
    pub fn cpu_written(&mut self, daddr: DAddr, size: u64) {
        self.collect_overlaps(&[Segment {
            offset: 0,
            daddr,
            size,
        }]);
        let ids = std::mem::take(&mut self.ids);
        for &id in &ids {
            let image = self.image(id);
            if !image.gpu_modified {
                image.cpu_dirty = true;
                image.tracked.clear();
            }
        }
        self.ids = ids;
    }

    /// Writes GPU-modified images overlapping `[daddr, daddr + size)` back to guest memory.
    pub fn flush(&mut self, tx: &mut Transfer<'_>, daddr: DAddr, size: u64) -> Result<()> {
        self.collect_overlaps(&[Segment {
            offset: 0,
            daddr,
            size,
        }]);
        let mut modified = std::mem::take(&mut self.ids);
        modified.retain(|&id| self.get(id).is_some_and(|i| i.gpu_modified));
        let result = self.download(tx, &modified, None);
        modified.clear();
        self.ids = modified;
        result
    }

    /// Device memory `[daddr, daddr + size)` was unmapped: drop the images in it.
    pub fn unmap(&mut self, scheduler: &mut Scheduler, daddr: DAddr, size: u64) {
        self.collect_overlaps(&[Segment {
            offset: 0,
            daddr,
            size,
        }]);
        let ids = std::mem::take(&mut self.ids);
        for &id in &ids {
            self.remove(scheduler, id);
        }
        self.ids = ids;
    }

    /// Texture descriptors may have changed: forget the address lookups (images stay).
    pub fn invalidate_lookups(&mut self) {
        self.lookup.clear();
    }

    /// A GPU-modified image whose first byte is the framebuffer's and that holds at least the
    /// framebuffer's texels in the same layout, brought up to date with its aliases.
    pub fn presentable(
        &mut self,
        scheduler: &mut Scheduler,
        fb: &FramebufferConfig,
    ) -> Result<Option<ImageId>> {
        let info = ImageInfo::from_framebuffer(fb);
        self.collect_overlaps(&[Segment {
            offset: 0,
            daddr: fb.address,
            size: info.size_bytes(),
        }]);
        let layout_matches = |image: &ImageInfo| match (image.layout, info.layout) {
            (
                GuestLayout::BlockLinear {
                    block_height_log2: a,
                    gob: ga,
                    ..
                },
                GuestLayout::BlockLinear {
                    block_height_log2: b,
                    gob: gb,
                    ..
                },
            ) => {
                a == b
                    && ga == gb
                    && image.layout == info.layout
                    && image.surface().level_tiling(0).row_gobs
                        == info.surface().level_tiling(0).row_gobs
            }
            (GuestLayout::Pitch { pitch: a }, GuestLayout::Pitch { pitch: b }) => a == b,
            _ => false,
        };
        let found = self
            .ids
            .iter()
            .copied()
            .filter(|&id| {
                let image = self.get(id).expect("a live image id");
                image.gpu_modified
                    && image.base() == fb.address
                    && image.segments[0].offset == 0
                    && image.info.kind == ImageKind::D2
                    && image.info.format.is_color()
                    && image.info.format.bytes_per_block() == fb.format.bytes_per_block()
                    && image.info.width >= fb.width
                    && image.info.height >= fb.height
                    && layout_matches(&image.info)
                    && !self.stale(id)
            })
            .max_by_key(|&id| self.get(id).map_or(0, |i| i.modified));
        if let Some(id) = found {
            self.take_from_aliases(scheduler, id)?;
        }
        Ok(found)
    }

    /// Clears a render target (level `target.level`, layers or 3D slices `target.layer +
    /// base_layer` on) inside `rect`, writing only what `value` masks in, and marks it
    /// GPU-modified. Whole unmasked clears are transfer clears; partial ones render into the
    /// target, through a masked-clear pipeline when channels or stencil bits are masked.
    pub fn clear(
        &mut self,
        tx: &mut Transfer<'_>,
        target: Found,
        value: ClearValue,
        rect: Rect,
        base_layer: u32,
        layer_count: u32,
    ) -> Result<()> {
        let device = self.device.clone();
        self.touch(target.id);
        let image = self.image(target.id);
        let level = target.level;
        let extent = image.host_extent(level);
        let (num, den) = image.scale.ratio();
        let scale = |v: u32| {
            (u64::from(v) * u64::from(num) / u64::from(den)).min(u64::from(u32::MAX)) as u32
        };
        let x0 = scale(rect.x).min(extent.width);
        let y0 = scale(rect.y).min(extent.height);
        let x1 = scale(rect.x.saturating_add(rect.width)).min(extent.width);
        let y1 = scale(rect.y.saturating_add(rect.height)).min(extent.height);
        let volume = image.info.kind == ImageKind::D3;
        let slices = if volume {
            extent.depth
        } else {
            image.image.desc().array_layers
        };
        let first = (target.layer + base_layer).min(slices);
        let count = layer_count.min(slices - first);
        if x1 <= x0 || y1 <= y0 || count == 0 {
            return Ok(());
        }
        let format = image.host.format;
        let numeric = image.info.format.numeric();
        let attachable = image.host.attachable;
        let image_aspects = aspect_of(format);
        let (aspects, masked) = match value {
            ClearValue::Color { mask, .. } => (
                vk::ImageAspectFlags::COLOR,
                mask != vk::ColorComponentFlags::RGBA,
            ),
            ClearValue::DepthStencil {
                depth,
                stencil,
                stencil_mask,
            } => {
                let mut a = vk::ImageAspectFlags::empty();
                if depth.is_some() {
                    a |= vk::ImageAspectFlags::DEPTH;
                }
                if stencil.is_some() {
                    a |= vk::ImageAspectFlags::STENCIL;
                }
                let a = a & image_aspects;
                (
                    a,
                    a.contains(vk::ImageAspectFlags::STENCIL) && stencil_mask != 0xff,
                )
            }
        };
        if aspects.is_empty() {
            return Ok(());
        }
        let whole = x0 == 0
            && y0 == 0
            && x1 == extent.width
            && y1 == extent.height
            && (!volume || count == slices);
        if !(whole && !masked) && !attachable {
            tracing::warn!(
                format = ?image.info.format,
                "partial clear of an image that cannot be an attachment; dropped"
            );
            return Ok(());
        }
        let cmd = tx.scheduler.record()?;
        let (color, depth_stencil) = match value {
            ClearValue::Color { value, .. } => (value, vk::ClearDepthStencilValue::default()),
            ClearValue::DepthStencil { depth, stencil, .. } => (
                vk::ClearColorValue::default(),
                vk::ClearDepthStencilValue {
                    depth: depth.unwrap_or(0.0),
                    stencil: u32::from(stencil.unwrap_or(0)),
                },
            ),
        };
        let color_target = aspects == vk::ImageAspectFlags::COLOR;
        if whole && !masked {
            let range = vk::ImageSubresourceRange {
                aspect_mask: aspects,
                base_mip_level: level,
                level_count: 1,
                base_array_layer: if volume { 0 } else { first },
                layer_count: if volume { 1 } else { count },
            };
            image.transition(&device, cmd, ImageAccess::TRANSFER_DST);
            // SAFETY: the image is in TRANSFER_DST layout and the range lies inside it.
            unsafe {
                if color_target {
                    device.raw.cmd_clear_color_image(
                        cmd,
                        image.image.raw(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &color,
                        &[range],
                    );
                } else {
                    device.raw.cmd_clear_depth_stencil_image(
                        cmd,
                        image.image.raw(),
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &depth_stencil,
                        &[range],
                    );
                }
            }
            self.mark_modified(target.id);
            return Ok(());
        }
        image.transition(
            &device,
            cmd,
            if color_target {
                ImageAccess::COLOR_ATTACHMENT
            } else {
                DEPTH_ATTACHMENT
            },
        );
        let layout = image.access.layout;
        let pipeline = if masked {
            if self.clear_pipelines.is_none() {
                self.clear_pipelines = Some(ClearPipelines::new(&device)?);
            }
            let pipelines = self.clear_pipelines.as_mut().expect("just created");
            Some(match value {
                ClearValue::Color { mask, .. } => pipelines.color(format, numeric, mask)?,
                ClearValue::DepthStencil {
                    depth,
                    stencil_mask,
                    ..
                } => pipelines.depth_stencil(
                    format,
                    depth.is_some() && image_aspects.contains(vk::ImageAspectFlags::DEPTH),
                    stencil_mask,
                )?,
            })
        } else {
            None
        };
        let words = match value {
            // SAFETY: every field of the union is four 32-bit words.
            ClearValue::Color { value, .. } => unsafe { value.uint32 },
            ClearValue::DepthStencil { depth, .. } => [depth.unwrap_or(0.0).to_bits(), 0, 0, 0],
        };
        let stencil_reference = match value {
            ClearValue::Color { .. } => None,
            ClearValue::DepthStencil { stencil, .. } => stencil,
        };
        let area = vk::Rect2D {
            offset: vk::Offset2D {
                x: x0 as i32,
                y: y0 as i32,
            },
            extent: vk::Extent2D {
                width: x1 - x0,
                height: y1 - y0,
            },
        };
        let render_area = vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.width,
                height: extent.height,
            },
        };
        let clear = [vk::ClearAttachment {
            aspect_mask: aspects,
            color_attachment: 0,
            clear_value: if color_target {
                vk::ClearValue { color }
            } else {
                vk::ClearValue { depth_stencil }
            },
        }];
        let per_pass = if masked || volume { 1 } else { count };
        let view_type = if per_pass == 1 {
            vk::ImageViewType::TYPE_2D
        } else {
            vk::ImageViewType::TYPE_2D_ARRAY
        };
        let mut layer = first;
        while layer < first + count {
            let layers = per_pass.min(first + count - layer);
            let key = ViewKey {
                format: format.as_raw(),
                view_type: view_type.as_raw(),
                components: [0; 4],
                aspect: image_aspects.as_raw(),
                base_level: level,
                levels: 1,
                base_layer: layer,
                layers,
            };
            let view = self.view_for(
                target.id,
                key,
                format,
                view_type,
                vk::ComponentMapping::default(),
            )?;
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(layout)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let colors = [attachment];
            let mut info = vk::RenderingInfo::default()
                .render_area(render_area)
                .layer_count(layers);
            if color_target {
                info = info.color_attachments(&colors);
            } else {
                if image_aspects.contains(vk::ImageAspectFlags::DEPTH) {
                    info = info.depth_attachment(&attachment);
                }
                if image_aspects.contains(vk::ImageAspectFlags::STENCIL) {
                    info = info.stencil_attachment(&attachment);
                }
            }
            // SAFETY: the view is a live single-level attachment view of the image, which is in
            // the matching attachment layout; the area and layers lie inside the render area.
            unsafe { device.raw.cmd_begin_rendering(cmd, &info) };
            match (pipeline, &self.clear_pipelines) {
                (Some(pipeline), Some(pipelines)) => {
                    pipelines.draw(cmd, pipeline, area, words, stencil_reference);
                }
                _ => {
                    let rects = [vk::ClearRect {
                        rect: area,
                        base_array_layer: 0,
                        layer_count: layers,
                    }];
                    // SAFETY: inside the rendering begun above; the rectangle lies in its area.
                    unsafe { device.raw.cmd_clear_attachments(cmd, &clear, &rects) };
                }
            }
            // SAFETY: ends the rendering begun above.
            unsafe { device.raw.cmd_end_rendering(cmd) };
            layer += layers;
        }
        self.mark_modified(target.id);
        Ok(())
    }

    fn view_compatible(&mut self, id: ImageId, format: PixelFormat) -> bool {
        let other = self.host(format);
        let image = self.get(id).expect("a live image id");
        let own = image.info.format;
        if own == format {
            return true;
        }
        image.host.conversion == Conversion::None
            && other.conversion == Conversion::None
            && own.is_color()
            && format.is_color()
            && own.bytes_per_block() == format.bytes_per_block()
            && own.block_width() == format.block_width()
            && own.block_height() == format.block_height()
            && own.compression() == format.compression()
    }

    /// Whether `image` can be copied into a new image of `format` on the GPU.
    fn copy_compatible(image: &CachedImage, host: HostFormat, format: PixelFormat) -> bool {
        let own = image.info.format;
        let aspects_match = own.is_color() == format.is_color()
            && (own.is_color() || image.host.format == host.format);
        image.host.conversion == Conversion::None
            && host.conversion == Conversion::None
            && aspects_match
            && own.bytes_per_block() == format.bytes_per_block()
            && own.block_width() == format.block_width()
            && own.block_height() == format.block_height()
            && own.compression() == format.compression()
    }

    fn can_blit(&self, format: vk::Format) -> bool {
        self.device
            .format_features(format)
            .contains(vk::FormatFeatureFlags::BLIT_SRC | vk::FormatFeatureFlags::BLIT_DST)
    }

    /// Lichii's rules: rescaleable images (2D, one level, tall enough), only downscaleable ones
    /// below native resolution, and only formats the host can blit.
    fn can_rescale(&self, id: ImageId) -> bool {
        let image = self.get(id).expect("a live image id");
        let info = &image.info;
        self.scale != ResolutionScale::X1
            && !image.image.is_sparse()
            && info.rescaleable
            && info.kind == ImageKind::D2
            && info.levels == 1
            && (self.scale > ResolutionScale::X1 || info.downscaleable)
            && self.can_blit(image.host.format)
    }

    /// Whether the device can create `info` as a sparse image (2D, single-sampled).
    fn can_be_sparse(&mut self, info: &ImageInfo) -> bool {
        let host = self.host(info.format);
        let desc = Self::image_desc(info, &host, ResolutionScale::X1);
        desc.image_type == vk::ImageType::TYPE_2D
            && self
                .device
                .sparse_granularity(desc.format, desc.usage)
                .is_some()
    }

    /// Gives memory to the subresources of `level` in `layers` the GPU is about to write when
    /// `id` is sparse, clearing what it binds to the zeros unbound memory reads as. Returns
    /// whether all of them have memory; false when the cap stopped the binding, and writes
    /// to what it left unbound are dropped.
    pub fn bind_written(
        &mut self,
        tx: &mut Transfer<'_>,
        id: ImageId,
        level: u32,
        layers: std::ops::Range<u32>,
    ) -> Result<bool> {
        let device = self.device.clone();
        let cap = self.sparse_cap;
        let Some(image) = self.get_mut(id) else {
            return Ok(true);
        };
        if !image.image.is_sparse() {
            return Ok(true);
        }
        let bound = image.image.bind(tx.scheduler, level, layers, cap)?;
        if !bound.complete && !self.sparse_capped {
            self.sparse_capped = true;
            tracing::warn!(
                cap,
                "a sparse image reached its memory cap; GPU writes past it are dropped"
            );
        }
        if bound.newly.is_empty() {
            return Ok(bound.complete);
        }
        let cmd = tx.scheduler.record()?;
        let image = self.image(id);
        image.transition(&device, cmd, ImageAccess::TRANSFER_DST);
        let aspect = aspect_of(image.host.format);
        let ranges: Vec<_> = bound
            .newly
            .iter()
            .map(|&layer| vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: level,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            })
            .collect();
        let raw = image.image.raw();
        // SAFETY: `cmd` is recording and the image was just made a transfer destination.
        unsafe {
            if aspect.contains(vk::ImageAspectFlags::COLOR) {
                device.raw.cmd_clear_color_image(
                    cmd,
                    raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearColorValue::default(),
                    &ranges,
                );
            } else {
                device.raw.cmd_clear_depth_stencil_image(
                    cmd,
                    raw,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &vk::ClearDepthStencilValue::default(),
                    &ranges,
                );
            }
        }
        Ok(bound.complete)
    }

    fn image_desc(info: &ImageInfo, host: &HostFormat, scale: ResolutionScale) -> ImageDesc {
        let color = info.format.is_color();
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        if host.attachable {
            usage |= if color {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            } else {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            };
        }
        let mut flags = vk::ImageCreateFlags::empty();
        if host.storage {
            usage |= vk::ImageUsageFlags::STORAGE;
            if info.format.is_srgb() {
                flags |= vk::ImageCreateFlags::EXTENDED_USAGE;
            }
        }
        if color && host.conversion == Conversion::None {
            flags |= vk::ImageCreateFlags::MUTABLE_FORMAT;
        }
        let (width, height) = match info.kind {
            ImageKind::D2 => (
                scaled(scale, info.width).max(1),
                scaled(scale, info.height).max(1),
            ),
            ImageKind::D1 => (info.width, 1),
            ImageKind::D3 => (info.width, info.height),
        };
        if info.kind == ImageKind::D3 && host.attachable {
            flags |= vk::ImageCreateFlags::TYPE_2D_ARRAY_COMPATIBLE;
        }
        if info.kind == ImageKind::D2 && info.layers >= 6 && width == height {
            flags |= vk::ImageCreateFlags::CUBE_COMPATIBLE;
        }
        let max_levels = 32 - width.max(height).max(info.depth).leading_zeros();
        ImageDesc {
            image_type: match info.kind {
                ImageKind::D1 => vk::ImageType::TYPE_1D,
                ImageKind::D2 => vk::ImageType::TYPE_2D,
                ImageKind::D3 => vk::ImageType::TYPE_3D,
            },
            format: host.format,
            extent: vk::Extent3D {
                width,
                height,
                depth: info.depth,
            },
            mip_levels: info.levels.min(max_levels),
            array_layers: info.layers,
            samples: vk::SampleCountFlags::TYPE_1,
            usage,
            flags,
        }
    }

    fn lookup_or_create(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        info: &ImageInfo,
    ) -> Result<Option<Found>> {
        let key = LookupKey {
            gmmu: std::ptr::from_ref(gmmu) as usize,
            gpu_va,
            info: *info,
        };
        let generation = gmmu.generation();
        if let Some(entry) = self.lookup.get(&key)
            && entry.generation == generation
            && self.get(entry.found.id).is_some()
        {
            let found = entry.found;
            self.touch(found.id);
            return Ok(Some(found));
        }
        let found = self.find_slow(tx, gmmu, gpu_va, info)?;
        if let Some(found) = found {
            self.touch(found.id);
            if self.lookup.len() >= MAX_LOOKUPS {
                self.lookup.clear();
            }
            self.lookup.insert(key, LookupEntry { generation, found });
        }
        Ok(found)
    }

    fn collect_overlaps(&mut self, ranges: &[Segment]) {
        self.visit += 1;
        let visit = self.visit;
        self.ids.clear();
        for range in ranges {
            for page in pages(range.daddr, range.size) {
                let Some(bucket) = self.pages.get(&page) else {
                    continue;
                };
                for &id in bucket {
                    let slot = &mut self.slots[id.index as usize];
                    let Some(image) = slot.image.as_mut() else {
                        continue;
                    };
                    if image.visit != visit && image.overlaps(range.daddr, range.size) {
                        image.visit = visit;
                        self.ids.push(id);
                    }
                }
            }
        }
        let slots = &self.slots;
        self.ids.sort_by_key(|id| {
            slots[id.index as usize]
                .image
                .as_ref()
                .map_or(0, |i| i.modified)
        });
    }

    fn find_slow(
        &mut self,
        tx: &mut Transfer<'_>,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        info: &ImageInfo,
    ) -> Result<Option<Found>> {
        let size = info.size_bytes();
        self.segments.clear();
        let mut offset = 0;
        let segments = &mut self.segments;
        gmmu.for_each_range(gpu_va, size, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                segments.push(Segment {
                    offset,
                    daddr,
                    size: range.size,
                });
            }
            offset += range.size;
        });
        if self.segments.first().is_none_or(|s| s.offset != 0) {
            tracing::debug!(gpu_va, "image at unmapped GPU memory is not cached");
            return Ok(None);
        }
        if info.texels() > MAX_TEXELS && !self.can_be_sparse(info) {
            tracing::warn!(?info, "image too large to back on the host; not cached");
            return Ok(None);
        }
        let base = self.segments[0].daddr;
        let contiguous = self.segments.len() == 1 && self.segments[0].size == size;
        let ranges = std::mem::take(&mut self.segments);
        self.collect_overlaps(&ranges);
        self.segments = ranges;
        for i in 0..self.ids.len() {
            let id = self.ids[i];
            let image = self.get(id).expect("a live image id");
            let candidate = if image.segments == self.segments && image.info.same_layout(info) {
                Some((0, 0))
            } else if contiguous && image.contiguous() && base >= image.base() {
                image.info.find_subresource(info, base - image.base())
            } else {
                None
            };
            if let Some((level, layer)) = candidate
                && self.view_compatible(id, info.format)
            {
                return Ok(Some(Found { id, level, layer }));
            }
        }
        self.joins.clear();
        self.new_aliases.clear();
        self.new_overlaps.clear();
        self.dropped.clear();
        let host = self.host(info.format);
        let overlaps = std::mem::take(&mut self.ids);
        for &id in &overlaps {
            let image = self.get(id).expect("a live image id");
            if contiguous
                && image.contiguous()
                && image.scale == ResolutionScale::X1
                && Self::copy_compatible(image, host, info.format)
                && let Some((at, own, extent)) =
                    pitch_overlap(&image.info, image.base(), info, base)
                        .or_else(|| block_linear_overlap(&image.info, image.base(), info, base))
            {
                self.new_aliases.push(Alias {
                    id,
                    at,
                    own,
                    extent,
                    synced: 0,
                });
                continue;
            }
            let subresource = (contiguous && image.contiguous() && image.base() >= base)
                .then(|| info.find_subresource(&image.info, image.base() - base))
                .flatten();
            match subresource {
                Some((level, layer))
                    if image.gpu_modified
                        && Self::copy_compatible(image, host, info.format)
                        && !self.stale(id) =>
                {
                    self.joins.push((id, level, layer));
                }
                Some(_) if !image.gpu_modified => self.dropped.push(id),
                _ => self.new_overlaps.push(id),
            }
        }
        self.ids = overlaps;
        let dropped = std::mem::take(&mut self.dropped);
        for &other in &dropped {
            self.remove(tx.scheduler, other);
        }
        self.dropped = dropped;
        let segments = self.segments.clone();
        let id = self.create(info, segments)?;
        let new_overlaps = std::mem::take(&mut self.new_overlaps);
        for &other in &new_overlaps {
            self.image(other).overlaps.push(id);
        }
        self.image(id).overlaps.extend_from_slice(&new_overlaps);
        self.new_overlaps = new_overlaps;
        let new_aliases = std::mem::take(&mut self.new_aliases);
        for alias in new_aliases
            .iter()
            .filter(|a| a.extent.0 > 0 && a.extent.1 > 0)
        {
            self.image(alias.id).aliases.push(Alias {
                id,
                at: alias.own,
                own: alias.at,
                extent: alias.extent,
                synced: 0,
            });
            self.image(id).aliases.push(*alias);
        }
        self.new_aliases = new_aliases;
        if !self.joins.is_empty() {
            self.refresh(tx, id)?;
            self.join(tx, id)?;
        }
        Ok(Some(Found {
            id,
            level: 0,
            layer: 0,
        }))
    }

    /// Copies the GPU-modified images in `self.joins` into `id`, which holds each of them as
    /// a level/layer run, then drops them. Scales follow Lichii: when any of them is scaled and
    /// the new image may scale, all scale up; otherwise all scale down.
    fn join(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        if self.joins.is_empty() {
            return Ok(());
        }
        let joins = std::mem::take(&mut self.joins);
        let any_scaled = joins
            .iter()
            .any(|&(j, ..)| self.get(j).is_some_and(|i| i.scale != ResolutionScale::X1));
        let target = if any_scaled && self.can_rescale(id) {
            self.scale
        } else {
            ResolutionScale::X1
        };
        self.rescale(tx, id, target, true)?;
        let device = self.device.clone();
        for &(join, level, layer) in &joins {
            self.rescale(tx, join, target, true)?;
            let desc = *self.image(join).image.desc();
            let mut resident = true;
            for l in 0..desc.mip_levels {
                resident &=
                    self.bind_written(tx, id, level + l, layer..layer + desc.array_layers)?;
            }
            let cmd = tx.scheduler.record()?;
            let (src_levels, src_layers, src_aspect) = {
                let src = self.image(join);
                src.transition(&device, cmd, ImageAccess::TRANSFER_SRC);
                let desc = src.image.desc();
                (desc.mip_levels, desc.array_layers, aspect_of(desc.format))
            };
            let mut regions = Vec::with_capacity(src_levels as usize);
            for l in 0..src_levels {
                let extent = self.image(join).host_extent(l);
                let layers = |base| vk::ImageSubresourceLayers {
                    aspect_mask: src_aspect,
                    mip_level: 0,
                    base_array_layer: base,
                    layer_count: src_layers,
                };
                regions.push(vk::ImageCopy {
                    src_subresource: vk::ImageSubresourceLayers {
                        mip_level: l,
                        ..layers(0)
                    },
                    dst_subresource: vk::ImageSubresourceLayers {
                        mip_level: level + l,
                        ..layers(layer)
                    },
                    extent,
                    ..Default::default()
                });
            }
            let src = self.image(join).image.raw();
            let dst = self.prepare_live(cmd, id, ImageAccess::TRANSFER_DST);
            // SAFETY: both images are in the layouts named and the regions lie inside them.
            unsafe {
                device.raw.cmd_copy_image(
                    cmd,
                    src,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &regions,
                )
            };
            if resident {
                self.remove(tx.scheduler, join);
            } else {
                self.image(join).overlaps.push(id);
                self.image(id).overlaps.push(join);
            }
            self.mark_modified(id);
        }
        Ok(())
    }

    fn create(&mut self, info: &ImageInfo, segments: Vec<Segment>) -> Result<ImageId> {
        let host = self.host(info.format);
        let desc = Self::image_desc(info, &host, ResolutionScale::X1);
        let image = if info.texels() > MAX_TEXELS {
            Image::new_sparse(&self.device, &desc)?
        } else {
            Image::new(&self.device, &desc, "guest image")?
        };
        let cached = CachedImage {
            info: *info,
            host,
            image,
            scale: ResolutionScale::X1,
            access: ImageAccess::UNDEFINED,
            segments,
            views: Vec::new(),
            cpu_dirty: true,
            tracked: Vec::new(),
            gpu_modified: false,
            modified: 0,
            visit: 0,
            last_use: self.frame,
            aliases: Vec::new(),
            written: 0,
            overlaps: Vec::new(),
            current: 0,
            keep_out: Vec::new(),
        };
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    image: None,
                    prev: NIL,
                    next: NIL,
                });
                self.slots.len() as u32 - 1
            }
        };
        let slot = &mut self.slots[index as usize];
        let id = ImageId {
            index,
            generation: slot.generation,
        };
        for s in &cached.segments {
            for page in pages(s.daddr, s.size) {
                self.pages.entry(page).or_default().push(id);
            }
        }
        slot.image = Some(cached);
        self.lru_push(index);
        Ok(id)
    }

    fn remove(&mut self, scheduler: &mut Scheduler, id: ImageId) {
        if self.get(id).is_some_and(|i| i.gpu_modified) {
            self.hand_off(scheduler, id);
        }
        let Some(slot) = self.slots.get_mut(id.index as usize) else {
            return;
        };
        if slot.generation != id.generation {
            return;
        }
        let Some(image) = slot.image.take() else {
            return;
        };
        slot.generation = slot.generation.wrapping_add(1);
        for alias in &image.aliases {
            if let Some(other) = self.get_mut(alias.id) {
                other.aliases.retain(|a| a.id != id);
            }
        }
        for &other in &image.overlaps {
            let other = self.get_mut(other).expect("overlaps are live");
            other.overlaps.retain(|&o| o != id);
            if other.gpu_modified && other.current < image.modified {
                other
                    .keep_out
                    .extend(image.segments.iter().map(|s| (s.daddr, s.daddr + s.size)));
            }
        }
        self.removals += 1;
        self.free.push(id.index);
        for s in &image.segments {
            for page in pages(s.daddr, s.size) {
                if let Some(bucket) = self.pages.get_mut(&page) {
                    bucket.retain(|&other| other != id);
                    if bucket.is_empty() {
                        self.pages.remove(&page);
                    }
                }
            }
        }
        self.lru_unlink(id.index);
        scheduler.retire((image.views, image.image));
    }

    /// Replaces the host image with one at `scale`; `blit` carries the contents over.
    fn rescale(
        &mut self,
        tx: &mut Transfer<'_>,
        id: ImageId,
        scale: ResolutionScale,
        blit: bool,
    ) -> Result<()> {
        let device = self.device.clone();
        let image = self.image(id);
        if image.scale == scale {
            return Ok(());
        }
        let desc = Self::image_desc(&image.info, &image.host, scale);
        let new = Image::new(&device, &desc, "guest image (scaled)")?;
        let mut access = ImageAccess::UNDEFINED;
        if blit && !image.cpu_dirty {
            let cmd = tx.scheduler.record()?;
            image.transition(&device, cmd, ImageAccess::TRANSFER_SRC);
            transition(
                &device,
                cmd,
                new.raw(),
                new.full_range(),
                ImageAccess::UNDEFINED,
                ImageAccess::TRANSFER_DST,
            );
            blit_image(&device, cmd, &image.image, &new, image.info.format);
            access = ImageAccess::TRANSFER_DST;
        }
        let old = std::mem::replace(&mut image.image, new);
        let views = std::mem::take(&mut image.views);
        image.scale = scale;
        image.access = access;
        tx.scheduler.retire((views, old));
        Ok(())
    }

    /// Uploads guest memory into the image when the guest CPU or a later overlap wrote it since
    /// the last upload, then copies in what its aliases' GPU work wrote since.
    fn refresh(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        if self.image(id).image.is_sparse() {
            let tick = self.tick;
            let image = self.image(id);
            image.cpu_dirty = false;
            image.current = tick;
            return self.take_from_aliases(tx.scheduler, id);
        }
        self.take_from_overlaps(tx, id)?;
        if self.image(id).cpu_dirty {
            self.rescale(tx, id, ResolutionScale::X1, false)?;
            self.upload(tx, id)?;
            let tick = self.tick;
            let image = self.image(id);
            image.cpu_dirty = false;
            image.current = tick;
            image.keep_out.clear();
            for alias in &mut image.aliases {
                alias.synced = 0;
            }
        }
        self.take_from_aliases(tx.scheduler, id)
    }

    /// Readies an upload of what overlaps modified after the image last held their bytes:
    /// writes back the GPU-modified ones, and its own texels around theirs unless they cover
    /// all of it.
    fn take_from_overlaps(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        let mut newer = std::mem::take(&mut self.overlap_scratch);
        newer.clear();
        let image = self.get(id).expect("a live image id");
        let current = image.current;
        newer.extend(image.overlaps.iter().copied().filter(|&o| {
            self.get(o)
                .is_some_and(|o| o.modified > current && !o.image.is_sparse())
        }));
        if newer.is_empty() || self.sync_on_gpu(tx, id, &mut newer)? {
            self.overlap_scratch = newer;
            return Ok(());
        }
        let image = self.get(id).expect("a live image id");
        let own = image.gpu_modified && !self.covered(id, &newer);
        newer.retain(|&o| self.get(o).is_some_and(|o| o.gpu_modified));
        if own {
            newer.push(id);
        }
        let result = self.download(tx, &newer, None);
        self.overlap_scratch = newer;
        result?;
        let image = self.image(id);
        image.gpu_modified = false;
        image.cpu_dirty = true;
        Ok(())
    }

    /// Whether an overlap was modified after `id` last held the shared bytes.
    fn stale(&self, id: ImageId) -> bool {
        let image = self.get(id).expect("a live image id");
        image
            .overlaps
            .iter()
            .any(|&o| self.get(o).expect("overlaps are live").modified > image.current)
    }

    /// Whether the segments of the `newer` images cover every byte of `id`.
    fn covered(&self, id: ImageId, newer: &[ImageId]) -> bool {
        let image = self.get(id).expect("a live image id");
        image.segments.iter().all(|s| {
            let (mut at, end) = (s.daddr, s.daddr + s.size);
            while at < end {
                let reach = newer
                    .iter()
                    .flat_map(|&o| &self.get(o).expect("overlaps are live").segments)
                    .filter(|o| o.daddr <= at && at < o.daddr + o.size)
                    .map(|o| o.daddr + o.size)
                    .max();
                let Some(reach) = reach else {
                    return false;
                };
                at = reach;
            }
            true
        })
    }

    /// Appends the device ranges a write-back of `id` leaves alone: those of overlaps modified
    /// after it last held the shared bytes, live or removed.
    fn keep_newer(&self, id: ImageId, keep: &mut Vec<(DAddr, DAddr)>) {
        let image = self.get(id).expect("a live image id");
        keep.extend_from_slice(&image.keep_out);
        for &other in &image.overlaps {
            let other = self.get(other).expect("overlaps are live");
            if other.modified > image.current && !other.image.is_sparse() {
                keep.extend(other.segments.iter().map(|s| (s.daddr, s.daddr + s.size)));
            }
        }
    }

    /// Copies in, oldest first, the rectangles of GPU-modified aliases that wrote themselves
    /// since this image last took them or wrote itself. Texels an alias only copied in come
    /// straight from the alias that wrote them, which shares them too.
    fn take_from_aliases(&mut self, scheduler: &mut Scheduler, id: ImageId) -> Result<()> {
        let mut newer = std::mem::take(&mut self.alias_scratch);
        newer.clear();
        let image = self.get(id).expect("a live image id");
        let written = image.written;
        newer.extend(image.aliases.iter().copied());
        newer.retain(|a| {
            self.get(a.id)
                .is_some_and(|other| other.gpu_modified && other.written > a.synced.max(written))
        });
        newer.sort_by_key(|a| self.get(a.id).map_or(0, |other| other.written));
        let mut result = Ok(());
        for alias in &newer {
            result = self.copy_texels(scheduler, alias.id, alias.at, id, alias.own, alias.extent);
            if result.is_err() {
                break;
            }
            let other = self.image(alias.id);
            let (tick, modified) = (other.written, other.modified);
            Self::set_synced(other, id, written);
            let image = self.image(id);
            image.modified = image.modified.max(modified);
            image.gpu_modified = true;
            Self::set_synced(image, alias.id, tick);
        }
        self.alias_scratch = newer;
        result
    }

    /// Copies what a GPU-modified image wrote itself into each alias whose rectangle neither
    /// took it yet nor was written since, so its data outlives its write-back or removal without
    /// another guest round trip. Aliases also take its pending write-back when guest memory does
    /// not hold its texels yet.
    fn hand_off(&mut self, scheduler: &mut Scheduler, id: ImageId) {
        let mut older = std::mem::take(&mut self.alias_scratch);
        older.clear();
        let image = self.get(id).expect("a live image id");
        let (written, modified) = (image.written, image.modified);
        let unwritten = image.gpu_modified;
        older.extend(image.aliases.iter().copied());
        older.retain(|a| {
            self.get(a.id).is_some_and(|other| {
                other.written < written
                    && other
                        .aliases
                        .iter()
                        .any(|back| back.id == id && back.synced < written)
            })
        });
        for alias in &older {
            if let Err(error) =
                self.copy_texels(scheduler, id, alias.own, alias.id, alias.at, alias.extent)
            {
                tracing::error!(%error, "copying an image into its alias failed");
                continue;
            }
            let other = self.image(alias.id);
            other.modified = other.modified.max(modified);
            other.gpu_modified |= unwritten;
            Self::set_synced(other, id, written);
        }
        self.alias_scratch = older;
    }

    fn set_synced(image: &mut CachedImage, alias: ImageId, tick: u64) {
        if let Some(link) = image.aliases.iter_mut().find(|a| a.id == alias) {
            link.synced = tick;
        }
    }

    /// Records a copy of the `extent` texels at `from_at` in `from` to `to_at` in `to`, at
    /// level 0 of layer 0.
    fn copy_texels(
        &mut self,
        scheduler: &mut Scheduler,
        from: ImageId,
        from_at: Texel,
        to: ImageId,
        to_at: Texel,
        extent: Texel,
    ) -> Result<()> {
        let device = self.device.clone();
        let cmd = scheduler.record()?;
        let (src, aspect) = {
            let image = self.image(from);
            image.transition(&device, cmd, ImageAccess::TRANSFER_SRC);
            (image.image.raw(), aspect_of(image.image.desc().format))
        };
        let dst = self.prepare_live(cmd, to, ImageAccess::TRANSFER_DST);
        let layers = vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        };
        let offset = |(x, y): Texel| vk::Offset3D {
            x: x as i32,
            y: y as i32,
            z: 0,
        };
        let region = vk::ImageCopy {
            src_subresource: layers,
            src_offset: offset(from_at),
            dst_subresource: layers,
            dst_offset: offset(to_at),
            extent: vk::Extent3D {
                width: extent.0,
                height: extent.1,
                depth: 1,
            },
        };
        // SAFETY: both images are in the layouts named and the region lies inside both.
        unsafe {
            device.raw.cmd_copy_image(
                cmd,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            )
        };
        Ok(())
    }

    fn upload(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        let device = self.device.clone();
        let Self {
            slots,
            scratch,
            traffic,
            lazy,
            ..
        } = self;
        let image = slots[id.index as usize]
            .image
            .as_mut()
            .expect("a live image id");
        image.track(tx.memory);
        let info = image.info;
        let host = image.host;
        let format = info.format;
        let levels = image.image.desc().mip_levels;
        let align = alignment(&host, format);
        let mut total = 0u64;
        for _ in 0..info.layers {
            for level in 0..levels {
                let layout = convert::host_layout(format, &host, info.level_extent(level));
                total = total.next_multiple_of(align) + layout.total() as u64;
                total = total.next_multiple_of(align);
            }
        }
        traffic.uploads += 1;
        traffic.upload_bytes += total;
        let slice = tx.staging.alloc(tx.scheduler, total.max(align), align)?;
        let (buffer, base_offset) = (slice.buffer, slice.offset);
        scratch.regions.clear();
        let mut offset = 0usize;
        let [main_aspect, stencil_aspect] = aspects_of(host.format);
        let direct =
            host.conversion == Conversion::None && convert::depth_planes(host.format).is_none();
        for layer in 0..info.layers {
            for level in 0..levels {
                let extent = info.level_extent(level);
                let layout = convert::host_layout(format, &host, extent);
                offset = offset.next_multiple_of(align as usize);
                let out = &mut slice.data[offset..offset + layout.total()];
                let guest_offset = info.subresource_offset(level, layer);
                let linear_len = convert::guest_level_bytes(format, extent);
                match info.layout {
                    GuestLayout::BlockLinear { .. } => {
                        let tiling = info.surface().level_tiling(level);
                        scratch.tiled.resize(tiling.size_bytes(), 0);
                        read_guest(
                            tx.memory,
                            lazy,
                            &image.segments,
                            guest_offset,
                            &mut scratch.tiled,
                        );
                        if direct {
                            unswizzle(&tiling, &scratch.tiled, out);
                        } else {
                            scratch.linear.resize(linear_len, 0);
                            unswizzle(&tiling, &scratch.tiled, &mut scratch.linear);
                        }
                    }
                    GuestLayout::Pitch { pitch } => {
                        let rows = extent.1.div_ceil(format.block_height()) as usize;
                        let row = linear_len / rows.max(1);
                        scratch.tiled.resize(pitch as usize * rows, 0);
                        read_guest(
                            tx.memory,
                            lazy,
                            &image.segments,
                            guest_offset,
                            &mut scratch.tiled,
                        );
                        scratch.linear.resize(linear_len, 0);
                        for (dst, src) in scratch
                            .linear
                            .chunks_exact_mut(row)
                            .zip(scratch.tiled.chunks(pitch as usize))
                        {
                            dst.copy_from_slice(&src[..row]);
                        }
                        if direct {
                            out.copy_from_slice(&scratch.linear[..out.len()]);
                        }
                    }
                }
                if !direct
                    && let Err(error) = convert::to_host(
                        format,
                        &host,
                        extent,
                        &scratch.linear,
                        out,
                        &mut scratch.convert,
                    )
                {
                    tracing::warn!(%error, "texture upload conversion failed");
                }
                let (w, h, d) = extent;
                let copy = |buffer_offset: usize, aspect_mask| vk::BufferImageCopy {
                    buffer_offset: base_offset + buffer_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask,
                        mip_level: level,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: d,
                    },
                    ..Default::default()
                };
                if let Some(aspect) = main_aspect {
                    scratch.regions.push(copy(offset, aspect));
                }
                if let Some(aspect) = stencil_aspect {
                    scratch.regions.push(copy(offset + layout.main, aspect));
                }
                offset += layout.total();
            }
        }
        let cmd = tx.scheduler.record()?;
        image.access.layout = vk::ImageLayout::UNDEFINED;
        image.transition(&device, cmd, ImageAccess::TRANSFER_DST);
        // SAFETY: the image is in TRANSFER_DST layout and every region reads staged bytes the
        // host layout sized.
        unsafe {
            device.raw.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                image.image.raw(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &scratch.regions,
            )
        };
        Ok(())
    }

    /// Writes GPU-modified images back to guest memory at native resolution, with one GPU wait
    /// for all of them, leaving the device addresses in `keep` untouched.
    fn download(
        &mut self,
        tx: &mut Transfer<'_>,
        ids: &[ImageId],
        keep: Option<(DAddr, DAddr)>,
    ) -> Result<()> {
        for &id in ids {
            self.take_from_aliases(tx.scheduler, id)?;
        }
        let mut downloads = std::mem::take(&mut self.downloads);
        downloads.clear();
        for &id in ids {
            let image = self.image(id);
            if image.downloadable() {
                downloads.push(Download {
                    id,
                    offset: 0,
                    modified: image.modified,
                });
            } else {
                image.gpu_modified = false;
                image.track(tx.memory);
            }
        }
        let mut readback = self.readback.take();
        let result = if downloads.is_empty() {
            Ok(())
        } else {
            self.record_downloads(tx, &mut downloads, &mut readback)
                .and_then(|()| tx.scheduler.finish())
        };
        if result.is_ok()
            && let Some(data) = readback.as_ref().and_then(Buffer::mapped)
        {
            let mut ranges = std::mem::take(&mut self.keep);
            for download in &downloads {
                ranges.clear();
                ranges.extend(keep);
                self.keep_newer(download.id, &mut ranges);
                self.write_back(
                    tx.memory,
                    download.id,
                    &data[download.offset as usize..],
                    &ranges,
                );
                let image = self.image(download.id);
                image.gpu_modified = false;
                let bytes = image.info.size_bytes();
                self.traffic.downloads += 1;
                self.traffic.download_bytes += bytes;
                self.hand_off(tx.scheduler, download.id);
            }
            self.keep = ranges;
        }
        self.readback = readback;
        self.downloads = downloads;
        result
    }

    /// Records copies of the images in `downloads` into `readback` (grown when too small), and
    /// sets each one's offset there. The copies are visible to the host once they complete.
    fn record_downloads(
        &mut self,
        tx: &mut Transfer<'_>,
        downloads: &mut [Download],
        readback: &mut Option<Buffer>,
    ) -> Result<()> {
        let mut total = 0u64;
        for download in downloads.iter_mut() {
            let (size, align) = self
                .get(download.id)
                .expect("a live image id")
                .readback_size();
            download.offset = total.next_multiple_of(align);
            total = download.offset + size;
        }
        let size = total.max(16);
        if readback.as_ref().is_none_or(|b| b.size() < size) {
            let buffer = Buffer::new(
                &self.device,
                size.next_power_of_two(),
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "texture readback",
            )?;
            if let Some(old) = readback.replace(buffer) {
                tx.scheduler.retire(old);
            }
        }
        let buffer = readback.as_ref().expect("just created").raw();
        let cmd = tx.scheduler.record()?;
        for download in downloads.iter() {
            self.record_download(tx, cmd, download.id, buffer, download.offset)?;
        }
        transfer_to_all_barrier(&self.device, cmd);
        Ok(())
    }

    /// Records the copy of one image, at native resolution, into `buffer` from `base` on, laid
    /// out as [`TextureCache::write_back`] reads it.
    fn record_download(
        &mut self,
        tx: &mut Transfer<'_>,
        cmd: vk::CommandBuffer,
        id: ImageId,
        buffer: vk::Buffer,
        base: u64,
    ) -> Result<()> {
        let device = self.device.clone();
        let image = self.image(id);
        let info = image.info;
        let host = image.host;
        let format = info.format;
        let levels = image.image.desc().mip_levels;
        let align = alignment(&host, format);
        let [main_aspect, stencil_aspect] = aspects_of(host.format);
        let mut regions = std::mem::take(&mut self.scratch.regions);
        regions.clear();
        let mut offset = 0usize;
        for layer in 0..info.layers {
            for level in 0..levels {
                let (w, h, d) = info.level_extent(level);
                let layout = convert::host_layout(format, &host, (w, h, d));
                offset = offset.next_multiple_of(align as usize);
                let copy = |buffer_offset: usize, aspect_mask| vk::BufferImageCopy {
                    buffer_offset: base + buffer_offset as u64,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask,
                        mip_level: level,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    image_extent: vk::Extent3D {
                        width: w,
                        height: h,
                        depth: d,
                    },
                    ..Default::default()
                };
                if let Some(aspect) = main_aspect {
                    regions.push(copy(offset, aspect));
                }
                if let Some(aspect) = stencil_aspect {
                    regions.push(copy(offset + layout.main, aspect));
                }
                offset += layout.total();
            }
        }
        let image = self.image(id);
        image.transition(&device, cmd, ImageAccess::TRANSFER_SRC);
        let native = if image.scale == ResolutionScale::X1 {
            None
        } else {
            let desc = Self::image_desc(&info, &host, ResolutionScale::X1);
            let native = Image::new(&device, &desc, "guest image (downscaled)")?;
            transition(
                &device,
                cmd,
                native.raw(),
                native.full_range(),
                ImageAccess::UNDEFINED,
                ImageAccess::TRANSFER_DST,
            );
            blit_image(&device, cmd, &image.image, &native, format);
            transition(
                &device,
                cmd,
                native.raw(),
                native.full_range(),
                ImageAccess::TRANSFER_DST,
                ImageAccess::TRANSFER_SRC,
            );
            Some(native)
        };
        let source = native.as_ref().unwrap_or(&image.image).raw();
        // SAFETY: the source is in TRANSFER_SRC layout and the regions fit the buffer, which
        // record_downloads sized for them.
        unsafe {
            device.raw.cmd_copy_image_to_buffer(
                cmd,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &regions,
            )
        };
        self.scratch.regions = regions;
        if let Some(native) = native {
            tx.scheduler.retire(native);
        }
        Ok(())
    }

    /// Converts the host texels `data` holds for an image (as [`TextureCache::record_download`]
    /// laid them out) and writes them to guest memory.
    fn write_back(
        &mut self,
        memory: &DeviceAddressSpace,
        id: ImageId,
        data: &[u8],
        keep: &[(DAddr, DAddr)],
    ) {
        let Self {
            slots,
            scratch,
            lazy,
            ..
        } = self;
        let image = slots[id.index as usize]
            .image
            .as_mut()
            .expect("a live image id");
        image.track(memory);
        let info = image.info;
        let host = image.host;
        let format = info.format;
        let levels = image.image.desc().mip_levels;
        let align = alignment(&host, format);
        let mut offset = 0usize;
        for layer in 0..info.layers {
            for level in 0..levels {
                let extent = info.level_extent(level);
                let layout = convert::host_layout(format, &host, extent);
                offset = offset.next_multiple_of(align as usize);
                let host_bytes = &data[offset..offset + layout.total()];
                offset += layout.total();
                scratch
                    .linear
                    .resize(convert::guest_level_bytes(format, extent), 0);
                if let Err(error) =
                    convert::to_guest(format, &host, extent, host_bytes, &mut scratch.linear)
                {
                    tracing::warn!(%error, "texture download conversion failed");
                    continue;
                }
                let guest_offset = info.subresource_offset(level, layer);
                match info.layout {
                    GuestLayout::BlockLinear { .. } => {
                        let tiling = info.surface().level_tiling(level);
                        scratch.tiled.resize(tiling.size_bytes(), 0);
                        read_guest(
                            memory,
                            lazy,
                            &image.segments,
                            guest_offset,
                            &mut scratch.tiled,
                        );
                        swizzle(&tiling, &scratch.linear, &mut scratch.tiled);
                        write_guest(lazy, &image.segments, guest_offset, &scratch.tiled, keep);
                    }
                    GuestLayout::Pitch { pitch } => {
                        let rows = extent.1.div_ceil(format.block_height()).max(1) as usize;
                        let row = scratch.linear.len() / rows;
                        for (y, line) in scratch.linear.chunks_exact(row).enumerate() {
                            write_guest(
                                lazy,
                                &image.segments,
                                guest_offset + y as u64 * u64::from(pitch),
                                line,
                                keep,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Runs the collector, once per presented frame. `usage` reads device memory use; it is
    /// called after the scheduler released what the GPU finished, so evicted images are no
    /// longer counted. Past thresholds from the driver's budget and the VRAM usage setting, it
    /// evicts images unused for a while, least recently used first, with bounded work per call;
    /// images used this frame always stay. GPU-modified images are read back without a wait
    /// and written to guest memory on a later call, once the copy finished and only if nothing
    /// used them in between.
    pub fn tick_frame(
        &mut self,
        tx: &mut Transfer<'_>,
        usage: impl FnOnce() -> MemoryUsage,
    ) -> Result<()> {
        let freed = tx.scheduler.collect();
        while let Some(&(tick, bytes)) = self.sentenced.front()
            && tick <= freed
        {
            self.sentenced.pop_front();
            self.sentenced_bytes -= bytes;
        }
        self.finish_evictions(tx);
        let result = self.collect_garbage(tx, usage());
        self.frame += 1;
        result
    }

    /// Evicts the images of the pending readback once the GPU finished it.
    fn finish_evictions(&mut self, tx: &mut Transfer<'_>) {
        let Some(pending) = self.pending else {
            return;
        };
        if !tx.scheduler.is_complete(pending.tick) {
            return;
        }
        self.pending = None;
        let downloads = std::mem::take(&mut self.pending_downloads);
        let readback = self.eviction_readback.take();
        let data = readback
            .as_ref()
            .and_then(Buffer::mapped)
            .expect("readback memory is host visible");
        for download in &downloads {
            let Some(image) = self.get(download.id) else {
                continue;
            };
            if image.last_use > pending.frame || image.modified != download.modified {
                continue;
            }
            if image.gpu_modified {
                let mut ranges = std::mem::take(&mut self.keep);
                ranges.clear();
                self.keep_newer(download.id, &mut ranges);
                self.write_back(
                    tx.memory,
                    download.id,
                    &data[download.offset as usize..],
                    &ranges,
                );
                self.keep = ranges;
                self.stats.flushed += 1;
            }
            self.evict(tx.scheduler, download.id);
        }
        self.eviction_readback = readback;
        self.pending_downloads = downloads;
    }

    fn collect_garbage(&mut self, tx: &mut Transfer<'_>, usage: MemoryUsage) -> Result<()> {
        if usage.budget == 0 {
            return Ok(());
        }
        let can_flush = self.pending.is_none();
        if can_flush {
            self.pending_downloads.clear();
        }
        let pending_bytes = self.pending.map_or(0, |p| p.bytes);
        let mut used = usage
            .used
            .saturating_sub(self.sentenced_bytes + pending_bytes);
        let limits = Thresholds::new(usage.budget, self.vram_usage);
        if used < limits.expected {
            return Ok(());
        }
        let mut aggressive = used >= limits.critical;
        let age = if used >= limits.retain {
            RECENT_AGE
        } else {
            OLD_AGE
        };
        let mut evictions = if aggressive {
            AGGRESSIVE_EVICTIONS
        } else {
            NORMAL_EVICTIONS
        };
        let mut visits = evictions * VISITS_PER_EVICTION;
        let (mut flush_bytes, mut queued_bytes) = (0, 0);
        let mut cursor = self.lru_head;
        while cursor != NIL && evictions > 0 && visits > 0 {
            visits -= 1;
            let slot = &self.slots[cursor as usize];
            let id = ImageId {
                index: cursor,
                generation: slot.generation,
            };
            cursor = slot.next;
            let image = slot.image.as_ref().expect("listed slots hold images");
            if image.last_use + age > self.frame {
                break;
            }
            let bytes = image.image.memory_size();
            if image.gpu_modified && image.downloadable() {
                let (size, _) = image.readback_size();
                if !aggressive
                    || !can_flush
                    || (flush_bytes > 0 && flush_bytes + size > MAX_FLUSH_BYTES)
                {
                    continue;
                }
                flush_bytes += size;
                queued_bytes += bytes;
                self.pending_downloads.push(Download {
                    id,
                    offset: 0,
                    modified: image.modified,
                });
            } else {
                self.evict(tx.scheduler, id);
            }
            used = used.saturating_sub(bytes);
            evictions -= 1;
            if aggressive && used < limits.critical {
                aggressive = false;
                evictions >>= 2;
            }
            if used < limits.expected {
                break;
            }
        }
        if !can_flush || self.pending_downloads.is_empty() {
            return Ok(());
        }
        let mut downloads = std::mem::take(&mut self.pending_downloads);
        let mut readback = self.eviction_readback.take();
        let recorded = self.record_downloads(tx, &mut downloads, &mut readback);
        self.eviction_readback = readback;
        self.pending_downloads = downloads;
        recorded?;
        let tick = tx.scheduler.flush()?;
        self.pending = Some(PendingEvictions {
            tick,
            frame: self.frame,
            bytes: queued_bytes,
        });
        Ok(())
    }

    fn evict(&mut self, scheduler: &mut Scheduler, id: ImageId) {
        let bytes = self.get(id).map_or(0, |i| i.image.memory_size());
        self.remove(scheduler, id);
        self.sentenced.push_back((scheduler.tick(), bytes));
        self.sentenced_bytes += bytes;
        self.stats.evicted += 1;
        self.stats.evicted_bytes += bytes;
    }

    /// Marks an image used this frame, moving it to the recent end of the LRU list.
    fn touch(&mut self, id: ImageId) {
        let frame = self.frame;
        let Some(image) = self.get_mut(id) else {
            return;
        };
        if image.last_use == frame {
            return;
        }
        image.last_use = frame;
        self.lru_unlink(id.index);
        self.lru_push(id.index);
    }

    fn lru_unlink(&mut self, index: u32) {
        let slot = &mut self.slots[index as usize];
        let (prev, next) = (slot.prev, slot.next);
        slot.prev = NIL;
        slot.next = NIL;
        match prev {
            NIL => self.lru_head = next,
            prev => self.slots[prev as usize].next = next,
        }
        match next {
            NIL => self.lru_tail = prev,
            next => self.slots[next as usize].prev = prev,
        }
    }

    fn lru_push(&mut self, index: u32) {
        let tail = self.lru_tail;
        let slot = &mut self.slots[index as usize];
        slot.prev = tail;
        slot.next = NIL;
        match tail {
            NIL => self.lru_head = index,
            tail => self.slots[tail as usize].next = index,
        }
        self.lru_tail = index;
    }
}

/// Buffer offset alignment of a host image's copies: a multiple of its texel block and of 16.
fn alignment(host: &HostFormat, format: PixelFormat) -> u64 {
    let block = match host.conversion {
        Conversion::None if convert::depth_planes(host.format).is_none() => {
            u64::from(format.bytes_per_block())
        }
        _ => 16,
    };
    let mut align = 16u64;
    while !align.is_multiple_of(block) {
        align += 16;
    }
    align
}

/// Blits every level and layer of `src` (TRANSFER_SRC) over `dst` (TRANSFER_DST), linear for
/// filterable colour formats and nearest otherwise.
fn blit_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: &Image,
    dst: &Image,
    format: PixelFormat,
) {
    let filter = if format.is_color()
        && !format.is_integer()
        && device
            .format_features(src.desc().format)
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        vk::Filter::LINEAR
    } else {
        vk::Filter::NEAREST
    };
    let aspect = aspect_of(src.desc().format);
    let corner = |e: vk::Extent3D, level: u32| vk::Offset3D {
        x: (e.width >> level).max(1) as i32,
        y: (e.height >> level).max(1) as i32,
        z: (e.depth >> level).max(1) as i32,
    };
    let levels = src.desc().mip_levels.min(dst.desc().mip_levels);
    let regions: Vec<_> = (0..levels)
        .map(|level| {
            let layers = vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: level,
                base_array_layer: 0,
                layer_count: src.desc().array_layers,
            };
            vk::ImageBlit {
                src_subresource: layers,
                src_offsets: [vk::Offset3D::default(), corner(src.desc().extent, level)],
                dst_subresource: layers,
                dst_offsets: [vk::Offset3D::default(), corner(dst.desc().extent, level)],
            }
        })
        .collect();
    // SAFETY: the images are in the layouts named and the regions cover them.
    unsafe {
        device.raw.cmd_blit_image(
            cmd,
            src.raw(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            dst.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &regions,
            filter,
        )
    };
}
