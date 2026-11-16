//! Overlap syncs on the GPU. When an image's bytes are shared with overlaps the GPU modified
//! later and no rectangle copy relates them (another size, format or block layout over the same
//! memory), the image's own texels and then every newer overlap's are swizzled into a device
//! buffer of guest bytes, which is unswizzled back into the image. Nothing waits for the GPU and
//! no texel crosses to the host, where a write-back and re-upload through guest memory would.

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;

use super::{
    CachedImage, GuestLayout, ImageId, ImageInfo, ImageKind, ResolutionScale, Segment,
    TextureCache, Transfer,
};
use crate::commands::ImageAccess;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::format::Conversion;
use crate::host_compute::HostCompute;
use crate::memory::{Buffer, aspect_of};
use crate::shaders;

/// The workgroup size of `guest_bytes.comp`.
const GROUP: [u32; 2] = [32, 8];

/// How a host texel's aspects make up the guest's 32-bit word, as `guest_bytes.comp` packs it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Packing {
    /// The host texel is the guest's bytes.
    Plain = 0,
    /// D24 in the low bits, S8 in the high byte (D24UnormS8Uint).
    DepthLowStencilHigh = 1,
    /// S8 in the low byte, D24 above it (S8UintD24Unorm).
    StencilLowDepthHigh = 2,
    /// D24 in the low bits and a zero high byte, with no stencil plane (X8D24Unorm).
    DepthLowZeroHigh = 3,
}

impl Packing {
    /// Whether the host texel has a stencil plane the guest word holds.
    fn stencil(self) -> bool {
        matches!(self, Self::DepthLowStencilHigh | Self::StencilLowDepthHigh)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    ToGuest = 0,
    FromGuest = 1,
}

/// The push constants of `guest_bytes.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    row_words: u32,
    row_bytes: u32,
    height: u32,
    row_gobs: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    switch2_layout: u32,
    subresource: u32,
    run_first: u32,
    run_count: u32,
    y_base: u32,
    direction: u32,
    packing: u32,
    stencil_pitch: u32,
    depth_float: u32,
    z_base: u32,
}

/// The pass and its buffers, grown to the largest image synced.
pub(super) struct GuestSync {
    compute: HostCompute,
    descriptors: DescriptorAllocator,
    guest: Option<Buffer>,
    linear: Option<Buffer>,
    stencil: Option<Buffer>,
    runs: Option<Buffer>,
    scratch: RunScratch,
}

impl GuestSync {
    fn new(device: &std::sync::Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::GUEST_BYTES_COMP,
                4,
                size_of::<Params>() as u32,
            )?,
            descriptors: DescriptorAllocator::new(device)?,
            guest: None,
            linear: None,
            stencil: None,
            runs: None,
            scratch: RunScratch::default(),
        })
    }
}

/// How `image` packs into guest words on the GPU, or `None` when the pass can't move it: only
/// block-linear 2D images or 3D ones of one layer at native scale whose texels are whole
/// 32-bit words or pack whole into one, laid out as the guest's, D24S8 kept as the host's D24S8
/// or D32S8 and mapped whole, or X8D24 kept as the host's X8D24 or D32. Rows that end inside
/// a word need segments that share no device memory: alias trimming gives whole words to one
/// source, which would drop an earlier alias's bytes in a partly written word.
fn packing(image: &CachedImage) -> Option<Packing> {
    let info = &image.info;
    let bytes = info.format.bytes_per_block();
    let shape = matches!(info.layout, GuestLayout::BlockLinear { .. })
        && image.scale == ResolutionScale::X1
        && image.trim == 0
        && match info.kind {
            ImageKind::D2 => true,
            ImageKind::D3 => info.layers == 1,
            ImageKind::D1 => false,
        }
        && info.samples == 1
        && info.format.block_width() == 1
        && info.format.block_height() == 1
        && (4 % bytes == 0 || bytes.is_multiple_of(4))
        && !(image.self_aliased
            && (0..info.levels)
                .any(|level| !(info.level_extent(level).0 * bytes).is_multiple_of(4)));
    if !shape {
        return None;
    }
    let host = image.host.format;
    match info.format {
        PixelFormat::D24UnormS8Uint if depth_stencil_host(host) => {
            image.contiguous().then_some(Packing::DepthLowStencilHigh)
        }
        PixelFormat::S8UintD24Unorm if depth_stencil_host(host) => {
            image.contiguous().then_some(Packing::StencilLowDepthHigh)
        }
        PixelFormat::X8D24Unorm
            if matches!(
                host,
                vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT
            ) =>
        {
            Some(Packing::DepthLowZeroHigh)
        }
        _ if image.host.conversion == Conversion::None
            && !aspect_of(host).contains(vk::ImageAspectFlags::STENCIL) =>
        {
            Some(Packing::Plain)
        }
        _ => None,
    }
}

/// Host formats a guest D24S8 texel moves through on the GPU: D24S8 itself, or the D32S8 that
/// drivers without D24S8 hold it in, whose float depth `guest_bytes.comp` converts.
fn depth_stencil_host(host: vk::Format) -> bool {
    matches!(
        host,
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

/// Bytes `image` shares with the guest bytes buffer: its offsets `[from, from + size)` are the
/// buffer's bytes from `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    from: u64,
    size: u64,
    at: u64,
}

/// Reused storage for [`shared_runs`] and [`trim_aliases`]: the target's segments by device
/// address, the runs, the guest bytes already claimed, the bytes one run moves, the claims
/// being merged and the trimmed runs, and the runs as `guest_bytes.comp` reads them.
#[derive(Default)]
struct RunScratch {
    targets: Vec<Segment>,
    runs: Vec<Run>,
    claimed: Vec<(u64, u64)>,
    moved: Vec<(u64, u64)>,
    merged: Vec<(u64, u64)>,
    trimmed: Vec<Run>,
    words: Vec<[u32; 4]>,
}

/// Fills `scratch.runs`, by image offset, with where the bytes of `image` land in the
/// guest bytes buffer, which holds the offsets `window` of `target`: one run per pair of
/// their segments sharing device memory. False when two of the target's segments share device
/// memory, which the search can't place.
fn shared_runs(
    image: &CachedImage,
    target: &CachedImage,
    window: (u64, u64),
    scratch: &mut RunScratch,
) -> bool {
    let RunScratch { targets, runs, .. } = scratch;
    targets.clear();
    targets.extend_from_slice(&target.segments);
    targets.sort_unstable_by_key(|t| t.daddr);
    if targets
        .windows(2)
        .any(|t| t[0].daddr + t[0].size > t[1].daddr)
    {
        return false;
    }
    runs.clear();
    for s in &image.segments {
        let end = s.daddr + s.size;
        let first = targets.partition_point(|t| t.daddr + t.size <= s.daddr);
        for t in targets[first..].iter().take_while(|t| t.daddr < end) {
            let (lo, hi) = (s.daddr.max(t.daddr), end.min(t.daddr + t.size));
            let offset = t.offset + (lo - t.daddr);
            let (from, to) = (offset.max(window.0), (offset + (hi - lo)).min(window.1));
            if from < to {
                runs.push(Run {
                    from: s.offset + (lo - s.daddr) + (from - offset),
                    size: to - from,
                    at: from - window.0,
                });
            }
        }
    }
    runs.sort_unstable_by_key(|r| r.from);
    true
}

/// Pushes onto `out`, in order and merged, the image offsets in `[lo, hi)` that
/// `guest_bytes.comp` moves for the first `levels` levels of every layer of `info`: the words
/// that hold texels, never the padding of rows past their last word, of GOBs or of blocks. The
/// shader writes only the texel bytes of a row's last word; its other bytes count as moved so
/// runs stay whole words.
fn moved_bytes(info: &ImageInfo, levels: u32, lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
    let mut push = |a: u64, b: u64| {
        let (a, b) = (a.max(lo), b.min(hi));
        if a >= b {
            return;
        }
        match out.last_mut() {
            Some(last) if last.1 == a => last.1 = b,
            _ => out.push((a, b)),
        }
    };
    for layer in 0..info.layers {
        for level in 0..levels {
            let start = info.subresource_offset(level, layer);
            if start >= hi || start + info.level_size(level) <= lo {
                continue;
            }
            let tiling = info.surface().level_tiling(level);
            let (w, h, d) = info.level_extent(level);
            let row_bytes = (w * tiling.bytes_per_block).next_multiple_of(4);
            let (bh, bd) = (tiling.block_height_log2, tiling.block_depth_log2);
            let column = 1u64 << (9 + bh + bd);
            let block_bytes = u64::from(tiling.row_gobs) * column;
            let (full_cols, cols) = (row_bytes / 64, row_bytes.div_ceil(64));
            let (full_rows, rows) = (h / 8, h.div_ceil(8));
            let blocks = u64::from(rows.div_ceil(1 << bh));
            let slab = blocks * block_bytes;
            let first_slab = (lo.saturating_sub(start) / slab) as u32;
            for z0 in (first_slab << bd..d).step_by(1 << bd) {
                let slab_start = start + u64::from(z0 >> bd) * slab;
                if slab_start >= hi {
                    break;
                }
                let slices = (d - z0).min(1 << bd);
                let dense = full_cols == tiling.row_gobs && slices == 1 << bd;
                for block in lo.saturating_sub(slab_start) / block_bytes..blocks {
                    let row = slab_start + block * block_bytes;
                    if row >= hi {
                        break;
                    }
                    let gob_y = (block as u32) << bh;
                    if dense && gob_y + (1 << bh) <= full_rows {
                        push(row, row + block_bytes);
                        continue;
                    }
                    for c in 0..cols {
                        let base = row + u64::from(c) * column;
                        if base >= hi {
                            break;
                        }
                        for z in 0..slices {
                            for g in gob_y..(gob_y + (1 << bh)).min(rows) {
                                let gob =
                                    base + (((u64::from(z) << bh) + u64::from(g - gob_y)) << 9);
                                if c < full_cols && g < full_rows {
                                    push(gob, gob + 512);
                                    continue;
                                }
                                let mut words = 0u128;
                                for y in g * 8..(g * 8 + 8).min(h) {
                                    for x in (c * 64..(c * 64 + 64).min(row_bytes)).step_by(4) {
                                        let at = start + tiling.offset(x, y, z0 + z) as u64 - gob;
                                        words |= 1 << (at / 4);
                                    }
                                }
                                while words != 0 {
                                    let bit = words.trailing_zeros();
                                    let len = (words >> bit).trailing_ones();
                                    push(gob + u64::from(bit) * 4, gob + u64::from(bit + len) * 4);
                                    words &= !(u128::MAX >> (128 - len) << bit);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Trims `scratch.runs` so every guest byte has one source run: where runs of an image whose
/// segments alias device memory land on the same guest bytes, the run with the highest image
/// offset that moves a byte there keeps it, as when the runs were painted one after another in
/// image order. `moved` pushes, in order, the image offsets in `[lo, hi)` the shader moves, so
/// a run's padding claims nothing. Leaves the runs sorted by image offset.
fn trim_aliases(scratch: &mut RunScratch, mut moved: impl FnMut(u64, u64, &mut Vec<(u64, u64)>)) {
    let RunScratch {
        runs,
        claimed,
        moved: moves,
        merged,
        trimmed,
        ..
    } = scratch;
    claimed.clear();
    trimmed.clear();
    for run in runs.iter().rev() {
        let (start, end) = (run.at, run.at + run.size);
        let first = claimed.partition_point(|c| c.1 <= start);
        let last = claimed.partition_point(|c| c.0 < end);
        let mut at = start;
        for &(lo, hi) in &claimed[first..last] {
            if lo > at {
                trimmed.push(Run {
                    from: run.from + (at - start),
                    size: lo - at,
                    at,
                });
            }
            at = at.max(hi);
        }
        if at < end {
            trimmed.push(Run {
                from: run.from + (at - start),
                size: end - at,
                at,
            });
        }
        moves.clear();
        moved(run.from, run.from + run.size, moves);
        if moves.is_empty() {
            continue;
        }
        merged.clear();
        let mut next = moves
            .iter()
            .map(|&(lo, hi)| (lo - run.from + start, hi - run.from + start))
            .peekable();
        let mut i = 0;
        loop {
            let item = match (claimed.get(i), next.peek()) {
                (Some(&c), Some(&m)) if c.0 <= m.0 => {
                    i += 1;
                    c
                }
                (_, Some(&m)) => {
                    next.next();
                    m
                }
                (Some(&c), None) => {
                    i += 1;
                    c
                }
                (None, None) => break,
            };
            match merged.last_mut() {
                Some(last) if item.0 <= last.1 => last.1 = last.1.max(item.1),
                _ => merged.push(item),
            }
        }
        std::mem::swap(claimed, merged);
    }
    trimmed.sort_unstable_by_key(|r| r.from);
    std::mem::swap(runs, trimmed);
}

/// Fills `scratch.words` with `scratch.runs` as `guest_bytes.comp` reads them, trimmed first
/// when the image's bytes go to the guest, so one pass gives every guest byte one writer.
/// `moved` is as for [`trim_aliases`].
fn run_words(
    scratch: &mut RunScratch,
    direction: Direction,
    moved: impl FnMut(u64, u64, &mut Vec<(u64, u64)>),
) {
    if direction == Direction::ToGuest {
        trim_aliases(scratch, moved);
    }
    scratch.words.clear();
    scratch.words.extend(
        scratch
            .runs
            .iter()
            .map(|r| [r.from as u32, r.size as u32, r.at as u32, 0]),
    );
}

pub(super) fn barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
    // SAFETY: `cmd` is recording outside any rendering.
    unsafe {
        device.raw.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(&[barrier]),
        );
    }
}

/// Grows `slot` to at least `size` bytes, retiring the buffer it replaces.
pub(super) fn ensure(
    device: &std::sync::Arc<Device>,
    tx: &mut Transfer<'_>,
    slot: &mut Option<Buffer>,
    size: u64,
    usage: vk::BufferUsageFlags,
    name: &'static str,
) -> Result<vk::Buffer> {
    if slot.as_ref().is_none_or(|b| b.size() < size) {
        let buffer = Buffer::new(
            device,
            size.next_power_of_two(),
            usage,
            MemoryLocation::GpuOnly,
            name,
        )?;
        if let Some(old) = slot.replace(buffer) {
            tx.scheduler.retire(old);
        }
    }
    Ok(slot.as_ref().expect("just ensured").raw())
}

impl TextureCache {
    /// Brings `id` up to date with `newer`, the overlaps modified after it last held the
    /// shared bytes, on the GPU. What the guest CPU wrote into `id` or an overlap since its
    /// last upload is uploaded first, so the pass moves the CPU's bytes along with the GPU's.
    /// Returns false, having recorded nothing, when the pass can't move one of the images, an
    /// overlap the CPU wrote is stale itself, or removed overlaps left their texels only in
    /// guest memory.
    pub(super) fn sync_on_gpu(
        &mut self,
        tx: &mut Transfer<'_>,
        id: ImageId,
        newer: &mut [ImageId],
    ) -> Result<bool> {
        let image = self.get(id).expect("a live image id");
        let movable = |o: &CachedImage| {
            o.keep_out.is_empty() && o.info.size_bytes() < 1 << 31 && packing(o).is_some()
        };
        if !movable(image)
            || newer.iter().any(|&o| {
                let overlap = self.get(o).expect("overlaps are live");
                !movable(overlap) || overlap.cpu_dirty && self.stale(o)
            })
        {
            return Ok(false);
        }
        if self.guest_sync.is_none() {
            self.guest_sync = Some(GuestSync::new(&self.device)?);
        }
        let Self {
            slots, guest_sync, ..
        } = self;
        let live = |id: ImageId| {
            slots[id.index as usize]
                .image
                .as_ref()
                .expect("a live image id")
        };
        let scratch = &mut guest_sync.as_mut().expect("just created").scratch;
        let image = live(id);
        let whole = (0, image.info.size_bytes());
        if !shared_runs(image, image, whole, scratch) {
            return Ok(false);
        }
        let mut window = whole;
        if packing(image).is_some_and(|p| !p.stencil()) {
            window = (u64::MAX, 0);
            for &o in newer.iter() {
                if !shared_runs(live(o), image, whole, scratch) {
                    return Ok(false);
                }
                for r in &scratch.runs {
                    window = (window.0.min(r.at), window.1.max(r.at + r.size));
                }
            }
        }
        if window.0 >= window.1 {
            return Ok(false);
        }
        for &o in newer.iter() {
            if self.image(o).cpu_dirty {
                self.refresh(tx, o)?;
            }
        }
        if self.image(id).cpu_dirty {
            self.take_cpu_writes(tx, id)?;
        }
        newer.sort_by_key(|&o| self.get(o).map_or(0, |o| o.modified));
        let mut linear = 4;
        let mut stencil = 4;
        for &o in newer.iter().chain(std::iter::once(&id)) {
            let info = &self.get(o).expect("a live image id").info;
            let (w, h, d) = info.level_extent(0);
            let row = (w * info.format.bytes_per_block()).next_multiple_of(4);
            linear = linear.max(u64::from(row) * u64::from(h) * u64::from(d));
            stencil = stencil.max((u64::from(w) * u64::from(h)).next_multiple_of(4));
        }
        let device = self.device.clone();
        let sync = self.guest_sync.as_mut().expect("just created");
        let storage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        ensure(
            &device,
            tx,
            &mut sync.guest,
            (window.1 - window.0).max(4),
            storage,
            "guest bytes",
        )?;
        ensure(
            &device,
            tx,
            &mut sync.linear,
            linear,
            storage,
            "guest sync texels",
        )?;
        ensure(
            &device,
            tx,
            &mut sync.stencil,
            stencil,
            storage,
            "guest sync stencil",
        )?;
        let cmd = tx.scheduler.record()?;
        self.paint(tx, cmd, id, id, window, Direction::ToGuest)?;
        for &o in newer.iter() {
            self.paint(tx, cmd, o, id, window, Direction::ToGuest)?;
        }
        self.paint(tx, cmd, id, id, window, Direction::FromGuest)?;
        let tick = self.tick;
        let image = self.image(id);
        image.current = tick;
        image.gpu_modified = true;
        image.retake_aliases();
        for &o in newer.iter() {
            self.hand_off(tx.scheduler, o);
        }
        self.traffic.gpu_syncs += 1;
        Ok(true)
    }

    /// Records the move of the subresources of `id` that share bytes with the guest bytes
    /// buffer, which holds the offsets `window` of `target`, between the image and the buffer.
    /// Texels outside the buffer keep the image's own values.
    fn paint(
        &mut self,
        tx: &mut Transfer<'_>,
        cmd: vk::CommandBuffer,
        id: ImageId,
        target: ImageId,
        window: (u64, u64),
        direction: Direction,
    ) -> Result<()> {
        let device = self.device.clone();
        let access = match direction {
            Direction::ToGuest => ImageAccess::TRANSFER_SRC,
            Direction::FromGuest => ImageAccess::STORAGE_WRITE,
        };
        self.get_mut(id)
            .expect("a live image id")
            .transition(&device, cmd, access);
        barrier(&device, cmd);
        let layout = access.layout;
        let Self {
            slots, guest_sync, ..
        } = self;
        let image = slots[id.index as usize]
            .image
            .as_ref()
            .expect("a live image id");
        let target = slots[target.index as usize]
            .image
            .as_ref()
            .expect("a live image id");
        let packing = packing(image).expect("checked by sync_on_gpu");
        let info = image.info;
        let raw = image.image.raw();
        let levels = info.levels.min(image.image.desc().mip_levels);
        let host = image.host.format;
        let main_aspect = if packing == Packing::Plain {
            aspect_of(host)
        } else {
            vk::ImageAspectFlags::DEPTH
        };
        let sync = guest_sync.as_mut().expect("created by sync_on_gpu");
        let placed = shared_runs(image, target, window, &mut sync.scratch);
        debug_assert!(placed, "checked by sync_on_gpu");
        let scratch = &mut sync.scratch;
        run_words(scratch, direction, |lo, hi, out| {
            moved_bytes(&info, levels, lo, hi, out)
        });
        let run_bytes: &[u8] = bytemuck::cast_slice(&scratch.words);
        ensure(
            &device,
            tx,
            &mut sync.runs,
            run_bytes.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            "guest sync runs",
        )?;
        let runs_buffer = sync.runs.as_ref().expect("ensured").raw();
        for (at, chunk) in (0u64..).step_by(65536).zip(run_bytes.chunks(65536)) {
            // SAFETY: `cmd` is recording outside any rendering, the buffer holds `run_bytes`, and
            // the barrier above orders this write after the last pass that read the buffer.
            unsafe { device.raw.cmd_update_buffer(cmd, runs_buffer, at, chunk) };
        }
        let guest = sync.guest.as_ref().expect("ensured").raw();
        let linear = sync.linear.as_ref().expect("ensured").raw();
        let stencil = sync.stencil.as_ref().expect("ensured").raw();
        let buffers =
            [guest, linear, stencil, runs_buffer].map(|buffer| vk::DescriptorBufferInfo {
                buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            });
        let all = (
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
        );
        for layer in 0..info.layers {
            for level in 0..levels {
                let start = info.subresource_offset(level, layer);
                let end = start + info.level_size(level);
                let runs = &sync.scratch.runs;
                let first = runs.partition_point(|r| r.from + r.size <= start);
                let count = runs[first..].iter().take_while(|r| r.from < end).count();
                if count == 0 {
                    continue;
                }
                let shared = &runs[first..first + count];
                let tiling = info.surface().level_tiling(level);
                let (w, h, d) = info.level_extent(level);
                let row_bytes = w * tiling.bytes_per_block;
                let row_words = row_bytes.div_ceil(4);
                let block_rows = 8u32 << tiling.block_height_log2;
                let block_bytes = u64::from(tiling.row_gobs)
                    << (9 + tiling.block_height_log2 + tiling.block_depth_log2);
                let lo = shared[0].from.max(start) - start;
                let hi = shared
                    .iter()
                    .map(|r| r.from + r.size)
                    .max()
                    .expect("count > 0")
                    .min(end)
                    - start;
                let (y0, y1, z0, z1, band) = if d == 1 {
                    let y0 = ((lo / block_bytes) as u32 * block_rows).min(h);
                    let y1 = (((hi - 1) / block_bytes + 1) as u32 * block_rows).min(h);
                    let band = (
                        start + u64::from(y0 / block_rows) * block_bytes,
                        (start + u64::from(y1.div_ceil(block_rows)) * block_bytes).min(end),
                    );
                    (y0, y1, 0, 1, band)
                } else {
                    let slices = 1u32 << tiling.block_depth_log2;
                    let slab = u64::from(h.div_ceil(block_rows)) * block_bytes;
                    let z0 = (lo / slab) as u32 * slices;
                    let z1 = (((hi - 1) / slab + 1) as u32 * slices).min(d);
                    let band = (
                        start + u64::from(z0 / slices) * slab,
                        (start + u64::from(z1.div_ceil(slices)) * slab).min(end),
                    );
                    (0, h, z0, z1, band)
                };
                if y0 >= y1 || z0 >= z1 {
                    continue;
                }
                let region = |aspect_mask, row_texels: u32, texel_bytes: u32| vk::BufferImageCopy {
                    buffer_offset: u64::from(z0 * h + y0) * u64::from(row_texels * texel_bytes),
                    buffer_row_length: row_texels,
                    buffer_image_height: h,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask,
                        mip_level: level,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D {
                        x: 0,
                        y: y0 as i32,
                        z: z0 as i32,
                    },
                    image_extent: vk::Extent3D {
                        width: w,
                        height: y1 - y0,
                        depth: z1 - z0,
                    },
                };
                let main = region(
                    main_aspect,
                    row_words * 4 / tiling.bytes_per_block,
                    tiling.bytes_per_block,
                );
                let stencil_region = region(vk::ImageAspectFlags::STENCIL, w, 1);
                let groups = [
                    row_words.div_ceil(GROUP[0]),
                    (y1 - y0).div_ceil(GROUP[1]),
                    z1 - z0,
                ];
                let keep = direction == Direction::ToGuest
                    || !shared
                        .iter()
                        .any(|r| r.from <= band.0 && band.1 <= r.from + r.size);
                // SAFETY: `cmd` is recording outside any rendering; the image is in `layout`,
                // which copies both ways allow, and every buffer holds the subresource's texels.
                unsafe {
                    if keep {
                        device
                            .raw
                            .cmd_copy_image_to_buffer(cmd, raw, layout, linear, &[main]);
                        if packing.stencil() {
                            device.raw.cmd_copy_image_to_buffer(
                                cmd,
                                raw,
                                layout,
                                stencil,
                                &[stencil_region],
                            );
                        }
                    }
                    barrier(&device, cmd);
                    let set = sync
                        .descriptors
                        .allocate(tx.scheduler, sync.compute.set_layout)?;
                    let params = Params {
                        z_base: z0,
                        row_words,
                        row_bytes,
                        height: h,
                        row_gobs: tiling.row_gobs,
                        block_height_log2: tiling.block_height_log2,
                        block_depth_log2: tiling.block_depth_log2,
                        switch2_layout: u32::from(tiling.layout == GobLayout::Switch2),
                        subresource: start as u32,
                        run_first: first as u32,
                        run_count: count as u32,
                        y_base: y0,
                        direction: direction as u32,
                        packing: packing as u32,
                        stencil_pitch: w,
                        depth_float: u32::from(
                            packing != Packing::Plain
                                && matches!(
                                    host,
                                    vk::Format::D32_SFLOAT | vk::Format::D32_SFLOAT_S8_UINT
                                ),
                        ),
                    };
                    sync.compute.dispatch(
                        cmd,
                        set,
                        &buffers,
                        bytemuck::bytes_of(&params),
                        groups,
                        all,
                    );
                    barrier(&device, cmd);
                    if direction == Direction::FromGuest {
                        device
                            .raw
                            .cmd_copy_buffer_to_image(cmd, linear, raw, layout, &[main]);
                        if packing.stencil() {
                            device.raw.cmd_copy_buffer_to_image(
                                cmd,
                                stencil,
                                raw,
                                layout,
                                &[stencil_region],
                            );
                        }
                        barrier(&device, cmd);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GUEST: usize = 16 << 10;

    type Moved = fn(u64, u64, &mut Vec<(u64, u64)>);

    fn source_byte(offset: u64) -> u8 {
        (offset * 7 + offset / 13) as u8
    }

    fn all_moved(lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
        out.push((lo, hi));
    }

    fn paint_in_order(runs: &[Run], moved: Moved) -> Vec<u8> {
        let mut guest = vec![0; GUEST];
        let mut spans = Vec::new();
        for r in runs {
            spans.clear();
            moved(r.from, r.from + r.size, &mut spans);
            for &(lo, hi) in &spans {
                for offset in lo..hi {
                    guest[(r.at + offset - r.from) as usize] = source_byte(offset);
                }
            }
        }
        guest
    }

    fn dispatched(runs: &[Run], moved: Moved) -> Vec<Run> {
        let mut scratch = RunScratch {
            runs: runs.to_vec(),
            ..RunScratch::default()
        };
        run_words(&mut scratch, Direction::ToGuest, moved);
        scratch
            .words
            .iter()
            .map(|w| Run {
                from: w[0].into(),
                size: w[1].into(),
                at: w[2].into(),
            })
            .collect()
    }

    fn assert_one_writer_with_the_serial_result(runs: &[Run], moved: Moved) {
        let serial = paint_in_order(runs, moved);
        let trimmed = dispatched(runs, moved);
        assert!(
            trimmed
                .windows(2)
                .all(|w| w[0].from + w[0].size <= w[1].from),
            "trimmed runs stay sorted and disjoint by image offset: {trimmed:?}"
        );
        let mut writers = vec![0u8; GUEST];
        let mut spans = Vec::new();
        for r in &trimmed {
            spans.clear();
            moved(r.from, r.from + r.size, &mut spans);
            for &(lo, hi) in &spans {
                for offset in lo..hi {
                    writers[(r.at + offset - r.from) as usize] += 1;
                }
            }
        }
        assert!(
            writers.iter().all(|&n| n <= 1),
            "a guest byte has two runs: {trimmed:?}"
        );
        assert_eq!(paint_in_order(&trimmed, moved), serial);
        let reversed: Vec<Run> = trimmed.iter().rev().copied().collect();
        assert_eq!(paint_in_order(&reversed, moved), serial);
    }

    #[test]
    fn aliased_runs_trim_to_one_writer_per_byte_with_the_serial_result() {
        let runs = [
            Run {
                from: 0,
                size: 4096,
                at: 4096,
            },
            Run {
                from: 4096,
                size: 4096,
                at: 4096,
            },
            Run {
                from: 8192,
                size: 4096,
                at: 6144,
            },
            Run {
                from: 12288,
                size: 2048,
                at: 0,
            },
            Run {
                from: 14336,
                size: 2048,
                at: 12288,
            },
        ];
        let reversed: Vec<Run> = runs.iter().rev().copied().collect();
        assert_ne!(
            paint_in_order(&reversed, all_moved),
            paint_in_order(&runs, all_moved),
            "untrimmed, one pass over aliased runs depends on which run writes last"
        );
        assert_one_writer_with_the_serial_result(&runs, all_moved);
    }

    #[test]
    fn padding_of_a_later_alias_leaves_the_bytes_to_the_earlier_run() {
        fn texels(lo: u64, hi: u64, out: &mut Vec<(u64, u64)>) {
            for (a, b) in [(0, 1024), (4096, 4608)] {
                if a.max(lo) < b.min(hi) {
                    out.push((a.max(lo), b.min(hi)));
                }
            }
        }
        let runs = [
            Run {
                from: 0,
                size: 4096,
                at: 0,
            },
            Run {
                from: 4096,
                size: 4096,
                at: 0,
            },
        ];
        let serial = paint_in_order(&runs, texels);
        assert_ne!(
            paint_in_order(&dispatched(&runs, all_moved), texels),
            serial,
            "claiming a later run's padding drops the earlier run's texels"
        );
        assert_one_writer_with_the_serial_result(&runs, texels);
    }

    #[test]
    fn runs_without_aliases_trim_to_themselves() {
        let runs = [
            Run {
                from: 0,
                size: 4096,
                at: 8192,
            },
            Run {
                from: 4096,
                size: 4096,
                at: 0,
            },
            Run {
                from: 8192,
                size: 1024,
                at: 4096,
            },
        ];
        assert_eq!(dispatched(&runs, all_moved), runs);
    }

    fn block_linear_info(
        format: PixelFormat,
        (width, height): (u32, u32),
        levels: u32,
        layers: u32,
        bh: u32,
        gob: GobLayout,
    ) -> ImageInfo {
        let mut info = ImageInfo {
            format,
            kind: ImageKind::D2,
            layout: GuestLayout::BlockLinear {
                block_height_log2: bh,
                block_depth_log2: 0,
                tile_width_spacing: 0,
                gob,
            },
            width,
            height,
            depth: 1,
            levels,
            layers,
            samples: 1,
            layer_stride: 0,
            rescaleable: false,
            downscaleable: false,
        };
        info.layer_stride = info.surface().layer_stride();
        info
    }

    #[test]
    fn moved_bytes_are_the_words_of_texels() {
        let images = [
            block_linear_info(
                PixelFormat::A8B8G8R8Unorm,
                (16, 16),
                1,
                1,
                4,
                GobLayout::Tegra,
            ),
            block_linear_info(
                PixelFormat::A8B8G8R8Unorm,
                (64, 64),
                1,
                1,
                0,
                GobLayout::Tegra,
            ),
            block_linear_info(
                PixelFormat::A8B8G8R8Unorm,
                (12, 10),
                1,
                1,
                1,
                GobLayout::Tegra,
            ),
            block_linear_info(PixelFormat::R8Unorm, (70, 21), 2, 2, 2, GobLayout::Switch2),
            block_linear_info(
                PixelFormat::R16G16B16A16Float,
                (40, 40),
                3,
                2,
                3,
                GobLayout::Tegra,
            ),
            block_linear_info(
                PixelFormat::A8B8G8R8Unorm,
                (128, 136),
                2,
                1,
                4,
                GobLayout::Switch2,
            ),
        ];
        let volume = |format, (width, height, depth), levels, bh, bd, gob| {
            let mut info = block_linear_info(format, (width, height), levels, 1, bh, gob);
            info.kind = ImageKind::D3;
            info.depth = depth;
            info.layout = GuestLayout::BlockLinear {
                block_height_log2: bh,
                block_depth_log2: bd,
                tile_width_spacing: 0,
                gob,
            };
            info.layer_stride = info.surface().layer_stride();
            info
        };
        let volumes = [
            volume(
                PixelFormat::R16G16B16A16Float,
                (56, 32, 40),
                1,
                0,
                4,
                GobLayout::Switch2,
            ),
            volume(PixelFormat::R8Unorm, (57, 20, 9), 2, 1, 2, GobLayout::Tegra),
            volume(
                PixelFormat::A8B8G8R8Unorm,
                (32, 32, 32),
                3,
                0,
                4,
                GobLayout::Tegra,
            ),
        ];
        for info in images.into_iter().chain(volumes) {
            let size = info.size_bytes() as usize;
            let mut moved = vec![false; size];
            for layer in 0..info.layers {
                for level in 0..info.levels {
                    let start = info.subresource_offset(level, layer) as usize;
                    let tiling = info.surface().level_tiling(level);
                    let (w, h, d) = info.level_extent(level);
                    for z in 0..d {
                        for y in 0..h {
                            for x in (0..w * tiling.bytes_per_block).step_by(4) {
                                let at = start + tiling.offset(x, y, z);
                                moved[at..at + 4].fill(true);
                            }
                        }
                    }
                }
            }
            for (lo, hi) in [
                (0, size),
                (100, size / 2 + 6),
                (size / 3, size - 4),
                (513, 515),
            ] {
                let mut expected: Vec<(u64, u64)> = Vec::new();
                for (at, _) in moved.iter().enumerate().take(hi).skip(lo).filter(|m| *m.1) {
                    match expected.last_mut() {
                        Some(last) if last.1 == at as u64 => last.1 += 1,
                        _ => expected.push((at as u64, at as u64 + 1)),
                    }
                }
                let mut out = Vec::new();
                moved_bytes(&info, info.levels, lo as u64, hi as u64, &mut out);
                assert_eq!(out, expected, "{info:?} over {lo}..{hi}");
            }
        }
    }
}
