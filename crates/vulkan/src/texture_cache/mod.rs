//! The texture cache: host images for guest images, found by guest address and layout.
//!
//! Every image is backed by device memory ranges (`Segment`s), so the same memory mapped at
//! two GPU addresses or in two channels shares one host image, and CPU writes, flushes and
//! unmaps, which name device addresses, find the images they touch through a page index.
//!
//! Lookups first try a per-address-space table keyed by the GPU address and the guest layout,
//! validated by the GMMU's mapping generation: a hit costs one hash lookup and allocates
//! nothing. A miss walks the mapping, reuses an image of the same layout (or one that holds the
//! requested image as a level/layer run), and otherwise creates a new image. Overlapping images
//! that a new one cannot take in are flushed to guest memory when the GPU modified them and
//! then dropped; images that fit as subresources of the new one are copied into it on the GPU.
//!
//! Images the guest CPU wrote are re-uploaded on their next use; images the GPU wrote are
//! written back to guest memory when a flush asks. Render targets that may scale (Lichii's
//! rules: 2D, one level, taller than 288 rows, or 512 for downscaling) render at the configured
//! resolution scale and are scaled back down on flush.

mod convert;
mod info;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_common::settings::{AstcRecompression, ResolutionScale};
use lichii_common::{DAddr, GpuVAddr};
use lichii_gpu::backend::FramebufferConfig;
use lichii_gpu::gmmu::{Gmmu, RangeTarget};
use lichii_memory::DeviceAddressSpace;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::{GobLayout, swizzle, unswizzle};
use lichii_texture::tic::{SwizzleSource, TextureDescriptor};

pub use info::{GuestLayout, ImageInfo, ImageKind, ViewInfo};

use crate::commands::{ImageAccess, transfer_to_all_barrier, transition};
use crate::device::Device;
use crate::error::Result;
use crate::format::{Conversion, HostFormat, host_format};
use crate::memory::{Buffer, Image, ImageDesc, ImageView, aspect_of};
use crate::scheduler::Scheduler;
use crate::staging::StagingRing;

/// Device memory is indexed in pages of this many bits (Lichii's texture cache uses 1 MiB).
const PAGE_BITS: u32 = 20;
/// Images with more texels are not created: an 8192x8192x68 virtual texture would take ~23 GB
/// (NX2_NOTES row 87).
const MAX_TEXELS: u64 = 1 << 28;
/// The lookup table is emptied when it grows past this many entries.
const MAX_LOOKUPS: usize = 1 << 16;

/// Depth/stencil attachment access for clears through dynamic rendering.
const DEPTH_ATTACHMENT: ImageAccess = ImageAccess {
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
    Color(vk::ClearColorValue),
    /// Depth and stencil values; `None` keeps that aspect.
    DepthStencil {
        depth: Option<f32>,
        stencil: Option<u8>,
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
    /// The host image holds data guest memory does not.
    gpu_modified: bool,
    /// Order of the last GPU write, for aliasing.
    modified: u64,
    visit: u64,
}

impl CachedImage {
    fn base(&self) -> DAddr {
        self.segments[0].daddr
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
    ids: Vec<ImageId>,
    joins: Vec<(ImageId, u32, u32)>,
    segments: Vec<Segment>,
    scratch: Scratch,
    readback: Option<Buffer>,
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

/// Reads guest bytes `[offset, offset + out.len())` of an image; unmapped gaps read zero.
fn read_guest(memory: &DeviceAddressSpace, segments: &[Segment], offset: u64, out: &mut [u8]) {
    out.fill(0);
    let end = offset + out.len() as u64;
    for s in segments {
        let (start, stop) = (s.offset.max(offset), (s.offset + s.size).min(end));
        if start >= stop {
            continue;
        }
        let chunk = &mut out[(start - offset) as usize..(stop - offset) as usize];
        if let Err(fault) = memory.read_bytes(s.daddr + (start - s.offset), chunk) {
            tracing::warn!(?fault, "texture reads unmapped device memory");
        }
    }
}

/// Writes guest bytes of an image, except those at device addresses in `keep`; bytes over
/// unmapped gaps are dropped.
fn write_guest(
    memory: &DeviceAddressSpace,
    segments: &[Segment],
    offset: u64,
    data: &[u8],
    keep: Option<(DAddr, DAddr)>,
) {
    let end = offset + data.len() as u64;
    let write = |daddr: DAddr, chunk: &[u8]| {
        if let Err(fault) = memory.write_bytes(daddr, chunk) {
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
        match keep {
            Some((k0, k1)) if k0 < chunk_end && k1 > daddr => {
                if k0 > daddr {
                    write(daddr, &chunk[..(k0 - daddr) as usize]);
                }
                if k1 < chunk_end {
                    write(k1, &chunk[(k1 - daddr) as usize..]);
                }
            }
            _ => write(daddr, chunk),
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
        gob_layout: GobLayout,
        recompression: AstcRecompression,
    ) -> Self {
        Self {
            device: device.clone(),
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
            ids: Vec::new(),
            joins: Vec::new(),
            segments: Vec::new(),
            scratch: Scratch::default(),
            readback: None,
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

    /// The host format of a guest format, queried from the device once.
    fn host(&mut self, format: PixelFormat) -> HostFormat {
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
        let mut view = ViewInfo::from_texture(desc, &info);
        view.base_level += found.level;
        view.base_layer += found.layer;
        Ok(Some((found.id, view)))
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
            self.image(found.id).cpu_dirty = false;
        }
        self.refresh(tx, found.id)?;
        let target = if self.can_rescale(found.id) {
            self.scale
        } else {
            ResolutionScale::X1
        };
        self.rescale(tx, found.id, target, !(discard && whole))?;
        Ok(Some(found))
    }

    /// A view of a cached image. `None` when the image cannot be read as `view.format`.
    pub fn view(&mut self, id: ImageId, view: &ViewInfo) -> Result<Option<vk::ImageView>> {
        if !self.view_compatible(id, view.format) {
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
        )?;
        let raw = view.raw();
        image.views.push((key, view));
        Ok(raw)
    }

    /// Records the barrier that makes the image ready for `access` and returns it.
    pub fn prepare(
        &mut self,
        cmd: vk::CommandBuffer,
        id: ImageId,
        access: ImageAccess,
    ) -> vk::Image {
        let device = self.device.clone();
        let image = self.image(id);
        image.transition(&device, cmd, access);
        image.image.raw()
    }

    /// Marks a cached image as written by the GPU.
    pub fn mark_modified(&mut self, id: ImageId) {
        self.tick += 1;
        let tick = self.tick;
        let image = self.image(id);
        image.gpu_modified = true;
        image.cpu_dirty = false;
        image.modified = tick;
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
        let result = ids.iter().try_for_each(|&id| {
            if self.image(id).gpu_modified {
                self.download(tx, id, Some((daddr, daddr.saturating_add(size))))?;
            }
            let image = self.image(id);
            image.cpu_dirty = true;
            image.gpu_modified = false;
            Ok(())
        });
        ids.clear();
        self.ids = ids;
        result
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
        let result = modified
            .iter()
            .try_for_each(|&id| self.download(tx, id, None));
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
    /// framebuffer's texels in the same layout.
    pub fn presentable(&mut self, fb: &FramebufferConfig) -> Option<ImageId> {
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
        self.ids
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
            })
            .max_by_key(|&id| self.get(id).map_or(0, |i| i.modified))
    }

    /// Clears a render target (level `target.level`, layers `target.layer + base_layer` on)
    /// inside `rect`, and marks it GPU-modified.
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
        let layers = image.image.desc().array_layers;
        let first_layer = (target.layer + base_layer).min(layers);
        let layer_count = layer_count.min(layers - first_layer);
        if x1 <= x0 || y1 <= y0 || layer_count == 0 {
            return Ok(());
        }
        let image_aspects = aspect_of(image.host.format);
        let aspects = match value {
            ClearValue::Color(_) => vk::ImageAspectFlags::COLOR,
            ClearValue::DepthStencil { depth, stencil } => {
                let mut a = vk::ImageAspectFlags::empty();
                if depth.is_some() {
                    a |= vk::ImageAspectFlags::DEPTH;
                }
                if stencil.is_some() {
                    a |= vk::ImageAspectFlags::STENCIL;
                }
                a & image_aspects
            }
        };
        if aspects.is_empty() {
            return Ok(());
        }
        let whole = x0 == 0 && y0 == 0 && x1 == extent.width && y1 == extent.height;
        let cmd = tx.scheduler.record()?;
        let range = vk::ImageSubresourceRange {
            aspect_mask: aspects,
            base_mip_level: level,
            level_count: 1,
            base_array_layer: first_layer,
            layer_count,
        };
        let (color, depth_stencil) = match value {
            ClearValue::Color(color) => (color, vk::ClearDepthStencilValue::default()),
            ClearValue::DepthStencil { depth, stencil } => (
                vk::ClearColorValue::default(),
                vk::ClearDepthStencilValue {
                    depth: depth.unwrap_or(0.0),
                    stencil: u32::from(stencil.unwrap_or(0)),
                },
            ),
        };
        if whole {
            image.transition(&device, cmd, ImageAccess::TRANSFER_DST);
            // SAFETY: the image is in TRANSFER_DST layout and the range lies inside it.
            unsafe {
                if aspects == vk::ImageAspectFlags::COLOR {
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
        } else if image.host.attachable && image.info.kind != ImageKind::D3 {
            let format = image.host.format;
            let key = ViewKey {
                format: format.as_raw(),
                view_type: vk::ImageViewType::TYPE_2D_ARRAY.as_raw(),
                components: [0; 4],
                aspect: image_aspects.as_raw(),
                base_level: level,
                levels: 1,
                base_layer: 0,
                layers,
            };
            let view = self.view_for(
                target.id,
                key,
                format,
                vk::ImageViewType::TYPE_2D_ARRAY,
                vk::ComponentMapping::default(),
            )?;
            let image = self.image(target.id);
            let color_target = aspects == vk::ImageAspectFlags::COLOR;
            image.transition(
                &device,
                cmd,
                if color_target {
                    ImageAccess::COLOR_ATTACHMENT
                } else {
                    DEPTH_ATTACHMENT
                },
            );
            let attachment = vk::RenderingAttachmentInfo::default()
                .image_view(view)
                .image_layout(image.access.layout)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let colors = [attachment];
            let render_area = vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            };
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
            let clear = [vk::ClearAttachment {
                aspect_mask: aspects,
                color_attachment: 0,
                clear_value: if color_target {
                    vk::ClearValue { color }
                } else {
                    vk::ClearValue { depth_stencil }
                },
            }];
            let rects = [vk::ClearRect {
                rect: vk::Rect2D {
                    offset: vk::Offset2D {
                        x: x0 as i32,
                        y: y0 as i32,
                    },
                    extent: vk::Extent2D {
                        width: x1 - x0,
                        height: y1 - y0,
                    },
                },
                base_array_layer: first_layer,
                layer_count,
            }];
            // SAFETY: the view is a live attachment view of the image, which is in the matching
            // attachment layout; the rectangle and layers lie inside the render area.
            unsafe {
                device.raw.cmd_begin_rendering(cmd, &info);
                device.raw.cmd_clear_attachments(cmd, &clear, &rects);
                device.raw.cmd_end_rendering(cmd);
            }
        } else {
            tracing::warn!(
                format = ?image.info.format,
                "partial clear of an image that cannot be an attachment; dropped"
            );
            return Ok(());
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
            && info.rescaleable
            && info.kind == ImageKind::D2
            && info.levels == 1
            && (self.scale > ResolutionScale::X1 || info.downscaleable)
            && self.can_blit(image.host.format)
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
        if host.storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let mut flags = vk::ImageCreateFlags::empty();
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
            return Ok(Some(entry.found));
        }
        let found = self.find_slow(tx, gmmu, gpu_va, info)?;
        if let Some(found) = found {
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
        if info.texels() > MAX_TEXELS {
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
        let host = self.host(info.format);
        let overlaps = std::mem::take(&mut self.ids);
        for &id in &overlaps {
            let image = self.get(id).expect("a live image id");
            let subresource = (contiguous && image.contiguous() && image.base() >= base)
                .then(|| info.find_subresource(&image.info, image.base() - base))
                .flatten();
            match subresource {
                Some((level, layer))
                    if image.gpu_modified && Self::copy_compatible(image, host, info.format) =>
                {
                    self.joins.push((id, level, layer));
                }
                _ => {
                    if image.gpu_modified {
                        self.download(tx, id, None)?;
                    }
                    self.remove(tx.scheduler, id);
                }
            }
        }
        self.ids = overlaps;
        let segments = self.segments.clone();
        let id = self.create(info, segments)?;
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
            let dst = self.prepare(cmd, id, ImageAccess::TRANSFER_DST);
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
            self.remove(tx.scheduler, join);
            self.mark_modified(id);
        }
        Ok(())
    }

    fn create(&mut self, info: &ImageInfo, segments: Vec<Segment>) -> Result<ImageId> {
        let host = self.host(info.format);
        let desc = Self::image_desc(info, &host, ResolutionScale::X1);
        let image = Image::new(&self.device, &desc, "guest image")?;
        let cached = CachedImage {
            info: *info,
            host,
            image,
            scale: ResolutionScale::X1,
            access: ImageAccess::UNDEFINED,
            segments,
            views: Vec::new(),
            cpu_dirty: true,
            gpu_modified: false,
            modified: 0,
            visit: 0,
        };
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    image: None,
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
        Ok(id)
    }

    fn remove(&mut self, scheduler: &mut Scheduler, id: ImageId) {
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

    /// Uploads guest memory into the image when the guest CPU wrote it since the last upload.
    fn refresh(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        if !self.image(id).cpu_dirty {
            return Ok(());
        }
        self.rescale(tx, id, ResolutionScale::X1, false)?;
        self.upload(tx, id)?;
        self.image(id).cpu_dirty = false;
        Ok(())
    }

    fn upload(&mut self, tx: &mut Transfer<'_>, id: ImageId) -> Result<()> {
        let device = self.device.clone();
        let Self { slots, scratch, .. } = self;
        let image = slots[id.index as usize]
            .image
            .as_mut()
            .expect("a live image id");
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
                        read_guest(tx.memory, &image.segments, guest_offset, &mut scratch.tiled);
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
                        read_guest(tx.memory, &image.segments, guest_offset, &mut scratch.tiled);
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

    /// Writes a GPU-modified image back to guest memory, at native resolution, leaving the
    /// device addresses in `keep` untouched.
    fn download(
        &mut self,
        tx: &mut Transfer<'_>,
        id: ImageId,
        keep: Option<(DAddr, DAddr)>,
    ) -> Result<()> {
        let device = self.device.clone();
        let image = self.image(id);
        let info = image.info;
        let host = image.host;
        let format = info.format;
        if matches!(
            host.conversion,
            Conversion::Decode | Conversion::DecodeToBc(_)
        ) {
            image.gpu_modified = false;
            return Ok(());
        }
        let levels = image.image.desc().mip_levels;
        let align = alignment(&host, format);
        let mut layouts_total = 0u64;
        for _ in 0..info.layers {
            for level in 0..levels {
                let layout = convert::host_layout(format, &host, info.level_extent(level));
                layouts_total = layouts_total.next_multiple_of(align) + layout.total() as u64;
            }
        }
        let size = layouts_total.next_multiple_of(align).max(align);
        if self.readback.as_ref().is_none_or(|b| b.size() < size) {
            let buffer = Buffer::new(
                &device,
                size.next_power_of_two(),
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "texture readback",
            )?;
            if let Some(old) = self.readback.replace(buffer) {
                tx.scheduler.retire(old);
            }
        }
        let readback = self.readback.as_ref().expect("just created").raw();
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
                    buffer_offset: buffer_offset as u64,
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
        let cmd = tx.scheduler.record()?;
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
        // SAFETY: the source is in TRANSFER_SRC layout and the regions fit the readback buffer.
        unsafe {
            device.raw.cmd_copy_image_to_buffer(
                cmd,
                source,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback,
                &regions,
            )
        };
        self.scratch.regions = regions;
        transfer_to_all_barrier(&device, cmd);
        let tick = tx.scheduler.flush()?;
        tx.scheduler.wait(tick)?;
        if let Some(native) = native {
            tx.scheduler.retire(native);
        }
        let Self {
            slots,
            scratch,
            readback,
            ..
        } = self;
        let image = slots[id.index as usize]
            .image
            .as_mut()
            .expect("a live image id");
        let data = readback
            .as_ref()
            .and_then(Buffer::mapped)
            .expect("readback memory is host visible");
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
                        read_guest(tx.memory, &image.segments, guest_offset, &mut scratch.tiled);
                        swizzle(&tiling, &scratch.linear, &mut scratch.tiled);
                        write_guest(
                            tx.memory,
                            &image.segments,
                            guest_offset,
                            &scratch.tiled,
                            keep,
                        );
                    }
                    GuestLayout::Pitch { pitch } => {
                        let rows = extent.1.div_ceil(format.block_height()).max(1) as usize;
                        let row = scratch.linear.len() / rows;
                        for (y, line) in scratch.linear.chunks_exact(row).enumerate() {
                            write_guest(
                                tx.memory,
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
        image.gpu_modified = false;
        Ok(())
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
