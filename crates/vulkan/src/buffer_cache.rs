//! Guest buffer memory in host device-local buffers.
//!
//! The cache keys host buffers by device address, so every GPU address space (and the CPU) that
//! reaches the same guest memory shares one host copy. Buffers cover whole 64 KiB caching pages;
//! a range that touches existing buffers' pages gets one grown buffer that takes over their
//! contents with a GPU copy. Guest writes (`BufferCache::invalidate`) mark 4 KiB pages dirty,
//! and acquiring a range uploads only its dirty pages through the `StagingRing`. Ranges the
//! GPU wrote (`BufferCache::mark_gpu_written`) are kept byte-exact: uploads skip them, and
//! `BufferCache::flush` downloads them back to guest memory.
//!
//! Acquiring a clean range is a table lookup and a bitmap test: it records nothing, allocates
//! nothing and takes no lock. Every acquire may record transfer commands, so the draw path binds
//! all its buffers before it begins rendering.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_common::{DAddr, GpuVAddr, PAGE_BITS, PAGE_SIZE};
use lichii_gpu::backend::{DrawParams, GpuWrittenPages, InlineIndices};
use lichii_gpu::engines::maxwell_3d::Regs;
use lichii_gpu::engines::maxwell_3d::regs::{IndexBuffer, IndexFormat, PrimitiveTopology};
use lichii_gpu::gmmu::{Gmmu, GpuRange, RangeTarget};
use lichii_gpu::lazy::LazyDeviceRanges;
use lichii_memory::{CachedPages, DeviceAddressSpace};
use lichii_shader::ShaderArch;

use crate::commands::transfer_to_all_barrier;
use crate::device::Device;
use crate::error::Result;
use crate::memory::{Buffer, BufferView, LiveMemory};
use crate::scheduler::Scheduler;
use crate::staging::{StagingRing, StagingSlice, live_oversized};

const CACHING_PAGE_BITS: u32 = 16;
const CACHING_PAGE_SIZE: u64 = 1 << CACHING_PAGE_BITS;
/// Uniform buffers up to this size are streamed instead of cached (Lichii's skip-cache size).
pub const UNIFORM_STREAM_MAX: u64 = 4 << 10;
const STREAM_RING_SIZE: u64 = 16 << 20;
/// Joins of joins past this score mark a stream buffer, which then grows ahead of its writes.
const STREAM_LEAP_THRESHOLD: u32 = 16;
const STREAM_LEAP_BYTES: u64 = 128 * CACHING_PAGE_SIZE;
/// Scattered reads at least this large bind their first contiguous run instead of a gathered copy.
const MAX_GATHER_SIZE: u64 = 64 << 20;

const CACHED_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDIRECT_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);
const STREAM_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::VERTEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::INDEX_BUFFER.as_raw()
        | vk::BufferUsageFlags::UNIFORM_BUFFER.as_raw()
        | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw()
        | vk::BufferUsageFlags::TRANSFER_SRC.as_raw()
        | vk::BufferUsageFlags::TRANSFER_DST.as_raw(),
);

/// A range of a host buffer for a descriptor or a vertex binding. Valid for the work recorded
/// in the scheduler until its current tick completes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
}

/// How to issue an indexed host draw: bind `buffer` at `offset` as `index_type`, then
/// `vkCmdDrawIndexed(index_count, instances, first_index, vertex_offset, first_instance)`.
/// An `index_count` of 0 means there is nothing to draw (and `buffer` is null).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexBinding {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub index_type: vk::IndexType,
    pub first_index: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
}

impl IndexBinding {
    const EMPTY: Self = Self {
        buffer: vk::Buffer::null(),
        offset: 0,
        index_type: vk::IndexType::UINT32,
        first_index: 0,
        index_count: 0,
        vertex_offset: 0,
    };
}

/// Running totals, for tests and the frame view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferCacheStats {
    /// Guest bytes copied into cached buffers.
    pub uploaded_bytes: u64,
    /// GPU-written bytes copied back to guest memory.
    pub downloaded_bytes: u64,
    /// Of [`BufferCacheStats::downloaded_bytes`], bytes served from copies taken when their
    /// writers were submitted ([`BufferCache::capture`]).
    pub captured_bytes: u64,
    /// GPU-written bytes [`BufferCache::capture`] copied, read back or not.
    pub capture_copy_bytes: u64,
    /// Bytes written to the stream ring: small uniforms, gathered ranges, converted indices.
    pub streamed_bytes: u64,
    pub created_buffers: u64,
    /// Buffers absorbed into a grown one.
    pub merged_buffers: u64,
}

/// The topology the host draws for a guest topology after the index conversion
/// [`BufferCache::index_buffer`] applies: quads and quad strips become triangle lists, line loops
/// line strips. The pipeline must use this topology.
pub fn assembled_topology(topology: PrimitiveTopology) -> PrimitiveTopology {
    match Assembly::of(topology) {
        Assembly::Direct => topology,
        Assembly::Quads | Assembly::QuadStrip => PrimitiveTopology::TRIANGLES,
        Assembly::LineLoop => PrimitiveTopology::LINE_STRIP,
    }
}

/// First register of Switch 2's per-stream byte sizes (`SET_VERTEX_STREAM_SIZE_A/B`, two words
/// per stream: the high byte, then the low word).
const VERTEX_STREAM_SIZE: usize = 0x600 / 4;

/// The GPU range vertex stream `index` reads; `None` when the stream is disabled or empty.
/// Switch 1 bounds a stream by its (inclusive) limit address. Switch 2's 3D class gives its byte
/// size instead and leaves the limit registers at zero.
pub fn vertex_stream_range(regs: &Regs, index: usize, arch: ShaderArch) -> Option<(GpuVAddr, u64)> {
    let stream = &regs.vertex_streams[index];
    let start = stream.address();
    let size = match arch {
        ShaderArch::Maxwell => regs.vertex_stream_limits[index]
            .address()
            .wrapping_add(1)
            .saturating_sub(start),
        ShaderArch::Ampere => {
            let words = regs.words();
            let at = VERTEX_STREAM_SIZE + 2 * index;
            u64::from(words[at] & 0xff) << 32 | u64::from(words[at + 1])
        }
    };
    (stream.enable() != 0 && start != 0 && size != 0).then_some((start, size))
}

struct CachedBuffer {
    buffer: Buffer,
    daddr: DAddr,
    size: u64,
    /// One bit per 4 KiB page whose guest bytes the host copy lacks.
    dirty: Vec<u64>,
    dirty_pages: u64,
    stream_score: u32,
    /// Texel buffer views by offset, range and format, dropped with the buffer.
    views: Vec<((u64, u64, i32), BufferView)>,
    /// Write tracking over each uploaded range (start, end), so a guest CPU write to it marks
    /// the pages dirty.
    tracked: Vec<(DAddr, DAddr, CachedPages)>,
    /// The scheduler tick of the newest recorded work that wrote GPU-written bytes into it, or
    /// that created it from older buffers.
    written_tick: u64,
}

impl CachedBuffer {
    fn end(&self) -> DAddr {
        self.daddr + self.size
    }

    fn contains(&self, daddr: DAddr, size: u64) -> bool {
        daddr >= self.daddr && daddr + size <= self.end()
    }

    fn mark_dirty(&mut self, start: DAddr, end: DAddr) {
        let first = ((start - self.daddr) >> PAGE_BITS) as usize;
        let last = (end - self.daddr).div_ceil(PAGE_SIZE) as usize;
        for page in first..last {
            let (word, bit) = (page / 64, 1 << (page % 64));
            if self.dirty[word] & bit == 0 {
                self.dirty[word] |= bit;
                self.dirty_pages += 1;
            }
        }
    }

    /// Stops tracking the pages of `[start, end)`, which upload again and are tracked anew.
    /// The rest of each range tracked with them keeps a guard of its own, taken before the old
    /// one is dropped, so it never goes untracked.
    fn untrack(&mut self, memory: &DeviceAddressSpace, start: DAddr, end: DAddr) {
        let (start, end) = (start & !(PAGE_SIZE - 1), end.next_multiple_of(PAGE_SIZE));
        let mut i = 0;
        while i < self.tracked.len() {
            let (s, e) = (self.tracked[i].0, self.tracked[i].1);
            if s >= end || e <= start {
                i += 1;
                continue;
            }
            let (_, _, old) = self.tracked.swap_remove(i);
            for (a, b) in [(s, start), (end, e)] {
                if a < b {
                    self.tracked.push((a, b, memory.cache_pages(a, b - a)));
                }
            }
            drop(old);
        }
    }
}

/// Clears the set bits of `[start, end)` and calls `f` with each run of them; returns how many
/// were set.
fn drain_runs(words: &mut [u64], start: usize, end: usize, mut f: impl FnMut(usize, usize)) -> u64 {
    let mut cleared = 0;
    let mut run = None;
    let mut page = start;
    while page < end {
        let (word, bit) = (page / 64, page % 64);
        if run.is_none() && words[word] >> bit == 0 {
            page = (word + 1) * 64;
            continue;
        }
        if words[word] & (1 << bit) != 0 {
            words[word] &= !(1 << bit);
            cleared += 1;
            run.get_or_insert(page);
        } else if let Some(first) = run.take() {
            f(first, page);
        }
        page += 1;
    }
    if let Some(first) = run {
        f(first, end);
    }
    cleared
}

/// Disjoint, non-touching byte ranges.
#[derive(Default)]
struct RangeSet {
    ranges: BTreeMap<u64, u64>,
}

impl RangeSet {
    fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    fn insert(&mut self, mut start: u64, mut end: u64) {
        if let Some((&s, &e)) = self.ranges.range(..=start).next_back()
            && e >= start
        {
            self.ranges.remove(&s);
            start = s;
            end = end.max(e);
        }
        while let Some((&s, &e)) = self.ranges.range(start..=end).next() {
            self.ranges.remove(&s);
            end = end.max(e);
        }
        self.ranges.insert(start, end);
    }

    fn subtract(&mut self, start: u64, end: u64) {
        if let Some((&s, &e)) = self.ranges.range(..start).next_back()
            && e > start
        {
            self.ranges.insert(s, start);
            if e > end {
                self.ranges.insert(end, e);
                return;
            }
        }
        while let Some((&s, &e)) = self.ranges.range(start..end).next() {
            self.ranges.remove(&s);
            if e > end {
                self.ranges.insert(end, e);
                break;
            }
        }
    }

    fn covers(&self, start: u64, end: u64) -> bool {
        self.ranges
            .range(..=start)
            .next_back()
            .is_some_and(|(_, &e)| e >= end)
    }

    /// The smallest range holding `[start, end)` and every range of the set it touches.
    fn area(&self, start: u64, end: u64) -> (u64, u64) {
        let first = self
            .ranges
            .range(..start)
            .next_back()
            .filter(|(_, e)| **e > start)
            .map_or(start, |(&s, _)| s);
        let last = self
            .ranges
            .range(..end)
            .next_back()
            .filter(|(_, e)| **e > start)
            .map_or(end, |(_, &e)| e.max(end));
        (first, last)
    }

    fn intersects(&self, start: u64, end: u64) -> bool {
        self.ranges
            .range(..end)
            .next_back()
            .is_some_and(|(_, &e)| e > start)
    }

    /// Calls `f` with each part of the set inside `[start, end)`, in order.
    fn for_each_in(&self, start: u64, end: u64, mut f: impl FnMut(u64, u64)) {
        let first = self
            .ranges
            .range(..start)
            .next_back()
            .filter(|(_, e)| **e > start)
            .map_or(start, |(&s, _)| s);
        for (&s, &e) in self.ranges.range(first..end) {
            f(s.max(start), e.min(end));
        }
    }
}

/// Size of the ring [`BufferCache::capture`] copies GPU-written ranges into.
const CAPTURE_RING_SIZE: u64 = 8 << 20;
/// The longest GPU-written range a download marks for capture.
const CAPTURE_MAX: u64 = CAPTURE_RING_SIZE / 8;

/// A copy of the GPU-written device bytes `[daddr, end)` at `offset` in the capture ring,
/// taken at the end of the work completing at `tick`.
#[derive(Clone, Copy)]
struct Capture {
    daddr: DAddr,
    end: DAddr,
    offset: u64,
    tick: u64,
}

/// GPU-written ranges that guest CPU reads come back to, copied to host memory at the end of
/// the work that writes them. A read of a captured range waits at most for that work, instead
/// of submitting a copy behind everything queued since and waiting for all of it.
#[derive(Default)]
struct Captures {
    ring: Option<Buffer>,
    head: u64,
    /// Ranges downloads read GPU-written bytes from: their later GPU writes are captured.
    hot: RangeSet,
    /// Hot ranges of writes recorded since the last capture.
    queued: Vec<(DAddr, DAddr)>,
    /// Disjoint copies of current GPU-written bytes: a write recorded over one drops it.
    live: Vec<Capture>,
    /// Parts of live captures left over after a drop.
    split: Vec<Capture>,
}

impl Captures {
    /// GPU work writing `[daddr, end)` is being recorded: copies taken before it are stale.
    fn drop_overlapping(&mut self, daddr: DAddr, end: DAddr) {
        if self.live.is_empty() {
            return;
        }
        let split = &mut self.split;
        self.live.retain(|capture| {
            if capture.end <= daddr || end <= capture.daddr {
                return true;
            }
            if capture.daddr < daddr {
                split.push(Capture {
                    end: daddr,
                    ..*capture
                });
            }
            if end < capture.end {
                split.push(Capture {
                    daddr: end,
                    offset: capture.offset + (end - capture.daddr),
                    ..*capture
                });
            }
            false
        });
        self.live.append(split);
    }

    /// The live capture holding `daddr`.
    fn find(&self, daddr: DAddr) -> Option<&Capture> {
        self.live
            .iter()
            .find(|capture| capture.daddr <= daddr && daddr < capture.end)
    }
}

/// Host-visible memory the GPU reads directly: streamed uniforms, gathered ranges and converted
/// indices. Allocations are reused once the tick that consumed them completed, as in the
/// [`StagingRing`].
struct StreamRing {
    device: Arc<Device>,
    ring: Buffer,
    capacity: u64,
    head: u64,
    /// In-flight regions in ring order: (start, end, tick).
    regions: VecDeque<(u64, u64, u64)>,
    oversized: VecDeque<(u64, Buffer)>,
}

impl StreamRing {
    fn new(device: &Arc<Device>, capacity: u64) -> Result<Self> {
        let ring = Buffer::new(
            device,
            capacity,
            STREAM_USAGE,
            MemoryLocation::CpuToGpu,
            "buffer cache stream",
        )?;
        Ok(Self {
            device: device.clone(),
            ring,
            capacity,
            head: 0,
            regions: VecDeque::new(),
            oversized: VecDeque::new(),
        })
    }

    fn alloc(
        &mut self,
        scheduler: &mut Scheduler,
        size: u64,
        align: u64,
    ) -> Result<StagingSlice<'_>> {
        if size > self.capacity / 4 {
            self.reclaim(scheduler);
            let buffer = Buffer::new(
                &self.device,
                size,
                STREAM_USAGE,
                MemoryLocation::CpuToGpu,
                "buffer cache stream (oversized)",
            )?;
            self.oversized.push_back((scheduler.tick(), buffer));
            let (_, buffer) = self.oversized.back_mut().expect("just pushed");
            return Ok(StagingSlice {
                buffer: buffer.raw(),
                offset: 0,
                data: buffer.mapped_mut().expect("stream memory is host visible"),
            });
        }
        let start = loop {
            self.reclaim(scheduler);
            let aligned = self.head.next_multiple_of(align);
            let start = if aligned + size <= self.capacity {
                aligned
            } else {
                0
            };
            let end = start + size;
            if self.regions.iter().all(|&(s, e, _)| end <= s || start >= e) {
                break start;
            }
            let (_, _, oldest) = *self.regions.front().expect("a full ring has regions");
            scheduler.wait(oldest)?;
        };
        let tick = scheduler.tick();
        let end = start + size;
        match self.regions.back_mut() {
            Some(last) if last.2 == tick && last.1 <= start => last.1 = end,
            _ => self.regions.push_back((start, end, tick)),
        }
        self.head = end;
        let buffer = self.ring.raw();
        let data = &mut self
            .ring
            .mapped_mut()
            .expect("stream memory is host visible")[start as usize..end as usize];
        Ok(StagingSlice {
            buffer,
            offset: start,
            data,
        })
    }

    fn reclaim(&mut self, scheduler: &mut Scheduler) {
        while let Some(&(_, _, tick)) = self.regions.front() {
            if !scheduler.is_complete(tick) {
                break;
            }
            self.regions.pop_front();
        }
        if self.regions.is_empty() {
            self.head = 0;
        }
        while self
            .oversized
            .front()
            .is_some_and(|(tick, _)| scheduler.is_complete(*tick))
        {
            self.oversized.pop_front();
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Assembly {
    Direct,
    Quads,
    QuadStrip,
    LineLoop,
}

impl Assembly {
    fn of(topology: PrimitiveTopology) -> Self {
        match topology {
            PrimitiveTopology::QUADS => Self::Quads,
            PrimitiveTopology::QUAD_STRIP => Self::QuadStrip,
            PrimitiveTopology::LINE_LOOP => Self::LineLoop,
            _ => Self::Direct,
        }
    }

    fn output_count(self, count: u32) -> u32 {
        match self {
            Self::Direct => count,
            Self::Quads => count / 4 * 6,
            Self::QuadStrip if count >= 4 => (count - 2) / 2 * 6,
            Self::QuadStrip => 0,
            Self::LineLoop if count > 0 => count + 1,
            Self::LineLoop => 0,
        }
    }
}

fn index_size(index_type: vk::IndexType) -> usize {
    match index_type {
        vk::IndexType::UINT8_EXT => 1,
        vk::IndexType::UINT16 => 2,
        _ => 4,
    }
}

/// The `(start, end)` runs of guest indices that assemble independently: the whole range, or
/// with `split` the runs between restart indices (`u32::MAX` after [`index_at`]).
fn runs(
    count: usize,
    index: &impl Fn(usize) -> u32,
    split: bool,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut start = 0;
    std::iter::from_fn(move || {
        while start <= count {
            let end = if split {
                (start..count)
                    .find(|&i| index(i) == u32::MAX)
                    .unwrap_or(count)
            } else {
                count
            };
            let run = (start, end);
            start = end + 1;
            if run.1 > run.0 {
                return Some(run);
            }
        }
        None
    })
}

/// Host indices `assembly` makes of `count` guest indices; with `restart`, each run between
/// restart indices assembles on its own and line loops keep a restart index between runs.
fn assembled_count(
    assembly: Assembly,
    count: usize,
    index: impl Fn(usize) -> u32,
    restart: bool,
) -> u32 {
    if assembly == Assembly::Direct {
        return count as u32;
    }
    let (mut total, mut drawn) = (0u32, 0u32);
    for (start, end) in runs(count, &index, restart) {
        let n = assembly.output_count((end - start) as u32);
        if n > 0 {
            total += n;
            drawn += 1;
        }
    }
    if assembly == Assembly::LineLoop {
        total += drawn.saturating_sub(1);
    }
    total
}

/// Writes the host indices for `count` guest indices read through `index` into `out`, as
/// `index_type` (truncating, so a restart index stays the all-ones value of the narrower type).
/// `out` holds [`assembled_count`] indices.
fn assemble(
    assembly: Assembly,
    count: usize,
    index: impl Fn(usize) -> u32,
    restart: bool,
    index_type: vk::IndexType,
    out: &mut [u8],
) {
    let width = index_size(index_type);
    let mut slot = 0;
    let mut put = |value: u32| {
        out[slot * width..(slot + 1) * width].copy_from_slice(&value.to_le_bytes()[..width]);
        slot += 1;
    };
    if assembly == Assembly::Direct {
        for i in 0..count {
            put(index(i));
        }
        return;
    }
    let mut first = true;
    for (start, end) in runs(count, &index, restart) {
        let quads = assembly.output_count((end - start) as u32) as usize / 6;
        match assembly {
            Assembly::LineLoop => {
                if !first {
                    put(u32::MAX);
                }
                for i in start..end {
                    put(index(i));
                }
                put(index(start));
            }
            Assembly::Quads | Assembly::QuadStrip => {
                if quads == 0 {
                    continue;
                }
                let (stride, pattern) = if assembly == Assembly::Quads {
                    (4, [0, 1, 2, 0, 2, 3])
                } else {
                    (2, [0, 3, 1, 0, 2, 3])
                };
                for quad in 0..quads {
                    for offset in pattern {
                        put(index(start + quad * stride + offset));
                    }
                }
            }
            Assembly::Direct => unreachable!("handled above"),
        }
        first = false;
    }
}

fn indices_len(indices: InlineIndices<'_>) -> usize {
    match indices {
        InlineIndices::U8(s) => s.len(),
        InlineIndices::U16(s) => s.len(),
        InlineIndices::U32(s) => s.len(),
    }
}

/// Index `i` of `indices`, the restart value mapped to `u32::MAX` when `restart` is on.
fn index_at(indices: InlineIndices<'_>, restart: bool, i: usize) -> u32 {
    let (value, max) = match indices {
        InlineIndices::U8(s) => (u32::from(s[i]), 0xff),
        InlineIndices::U16(s) => (u32::from(s[i]), 0xffff),
        InlineIndices::U32(s) => (s[i], u32::MAX),
    };
    if restart && value == max {
        u32::MAX
    } else {
        value
    }
}

fn sub_indices(indices: InlineIndices<'_>, first: u32, count: u32) -> InlineIndices<'_> {
    let len = indices_len(indices);
    let start = (first as usize).min(len);
    let end = (first as usize).saturating_add(count as usize).min(len);
    match indices {
        InlineIndices::U8(s) => InlineIndices::U8(&s[start..end]),
        InlineIndices::U16(s) => InlineIndices::U16(&s[start..end]),
        InlineIndices::U32(s) => InlineIndices::U32(&s[start..end]),
    }
}

/// The device address `[gpu_va, +size)` maps to when it is one contiguous device range.
fn contiguous(gmmu: &Gmmu, gpu_va: GpuVAddr, size: u64) -> Option<DAddr> {
    let mut runs = 0;
    let mut daddr = None;
    gmmu.for_each_range(gpu_va, size, |range| {
        runs += 1;
        daddr = match range.target {
            RangeTarget::Device(daddr) => Some(daddr),
            _ => None,
        };
    });
    if runs == 1 { daddr } else { None }
}

/// Copies guest memory; unmapped pages read zero.
fn read_guest(memory: &DeviceAddressSpace, daddr: DAddr, out: &mut [u8]) {
    if memory.read_bytes(daddr, out).is_ok() {
        return;
    }
    let mut done = 0;
    while done < out.len() {
        let addr = daddr + done as u64;
        let len = ((PAGE_SIZE - addr % PAGE_SIZE) as usize).min(out.len() - done);
        let chunk = &mut out[done..done + len];
        if memory.read_bytes(addr, chunk).is_err() {
            chunk.fill(0);
        }
        done += len;
    }
}

/// Orders transfers after every earlier read and write of the buffers they touch.
fn before_transfer(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = [vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE)];
    let dependency = vk::DependencyInfo::default().memory_barriers(&barrier);
    // SAFETY: `cmd` is recording.
    unsafe { device.raw.cmd_pipeline_barrier2(cmd, &dependency) };
}

fn copy_buffer(
    device: &Device,
    cmd: vk::CommandBuffer,
    src: vk::Buffer,
    dst: vk::Buffer,
    regions: &[vk::BufferCopy],
) {
    // SAFETY: `cmd` is recording and the regions lie inside both live buffers.
    unsafe { device.raw.cmd_copy_buffer(cmd, src, dst, regions) };
}

/// A generated quad index buffer: `quads` quads' triangles for vertices counted from 0.
#[derive(Default)]
struct QuadLut {
    buffer: Option<Buffer>,
    quads: u32,
}

/// Guest buffers in host memory. Owned by the GPU thread's rasterizer, which passes its
/// scheduler and staging ring to every call that may record work.
///
/// A binding can name a buffer that a later acquire absorbs into a grown one: the old buffer
/// lives until the GPU passes the current tick, but later uploads and shader writes go to the new
/// one. A draw resolves all its bindings, then resolves them again if
/// [`BufferCacheStats::merged_buffers`] changed meanwhile; the second pass finds every range
/// inside an existing buffer.
pub struct BufferCache {
    device: Arc<Device>,
    memory: Arc<DeviceAddressSpace>,
    /// Device pages GPU work wrote, marked with every range [`BufferCache::mark_gpu_written`]
    /// records.
    written_pages: Arc<GpuWrittenPages>,
    /// Device ranges that get memory as they are written: downloads go through them.
    lazy: Arc<LazyDeviceRanges>,
    limit: DAddr,
    /// Buffer slot + 1 for each 64 KiB caching page, 0 where no buffer covers it.
    page_table: Vec<u32>,
    buffers: Vec<Option<CachedBuffer>>,
    free_slots: Vec<usize>,
    gpu_written: RangeSet,
    /// Whether GPU-written ranges trap guest CPU accesses until they are downloaded
    /// ([`lichii_gpu::gpu::flush_traps_enabled`]).
    flush_traps: bool,
    stream: StreamRing,
    readback: Option<Buffer>,
    captures: Captures,
    luts: [QuadLut; 2],
    widen_u8: bool,
    warned_scattered_write: bool,
    stats: BufferCacheStats,
    uploads: Vec<(DAddr, u64)>,
    copies: Vec<vk::BufferCopy>,
    overlaps: Vec<usize>,
    downloads: Vec<(vk::Buffer, u64, DAddr, u64)>,
    segments: Vec<GpuRange>,
    scratch: Vec<u32>,
    raced: Vec<(DAddr, DAddr)>,
    /// Ranges bound for writing by the draw or dispatch being prepared, stamped with the tick of
    /// the work it is recorded into by [`BufferCache::commit_writes`].
    pending_writes: Vec<(DAddr, DAddr)>,
}

impl BufferCache {
    /// A cache over the device address space `memory`.
    pub fn new(
        device: &Arc<Device>,
        memory: Arc<DeviceAddressSpace>,
        written_pages: Arc<GpuWrittenPages>,
        lazy: Arc<LazyDeviceRanges>,
    ) -> Result<Self> {
        let limit = 1u64 << memory.address_space_bits();
        Ok(Self {
            device: device.clone(),
            memory,
            written_pages,
            lazy,
            limit,
            page_table: vec![0; (limit >> CACHING_PAGE_BITS) as usize],
            buffers: Vec::new(),
            free_slots: Vec::new(),
            gpu_written: RangeSet::default(),
            flush_traps: lichii_gpu::gpu::flush_traps_enabled(),
            stream: StreamRing::new(device, STREAM_RING_SIZE)?,
            readback: None,
            captures: Captures::default(),
            luts: Default::default(),
            widen_u8: !device.features.index_type_uint8,
            warned_scattered_write: false,
            stats: BufferCacheStats::default(),
            uploads: Vec::new(),
            copies: Vec::new(),
            overlaps: Vec::new(),
            downloads: Vec::new(),
            segments: Vec::new(),
            scratch: Vec::new(),
            raced: Vec::new(),
            pending_writes: Vec::new(),
        })
    }

    pub fn stats(&self) -> BufferCacheStats {
        self.stats
    }

    /// Whether GPU work wrote bytes of `[daddr, +size)` that [`BufferCache::flush`] has not
    /// downloaded yet.
    pub fn is_gpu_written(&self, daddr: DAddr, size: u64) -> bool {
        self.gpu_written
            .intersects(daddr, daddr.saturating_add(size))
    }

    /// The oversized stream buffers not freed yet.
    pub fn oversized_stream(&self) -> LiveMemory {
        live_oversized(&self.stream.oversized)
    }

    /// The alignment storage bindings start at: [`BufferCache::storage_buffer`] rounds the
    /// address down to it, so shaders address the binding from that base.
    pub fn storage_alignment(&self) -> u64 {
        self.device
            .properties
            .limits
            .min_storage_buffer_offset_alignment
    }

    fn buffer(&self, id: usize) -> &CachedBuffer {
        self.buffers[id]
            .as_ref()
            .expect("page table names a live buffer")
    }

    fn lookup(page_table: &[u32], daddr: DAddr) -> Option<usize> {
        let entry = *page_table.get((daddr >> CACHING_PAGE_BITS) as usize)?;
        (entry != 0).then(|| entry as usize - 1)
    }

    /// The guest wrote `[daddr, +size)`: its pages upload again before their next use, and its
    /// bytes are no longer GPU-written.
    pub fn invalidate(&mut self, daddr: DAddr, size: u64) {
        let end = daddr.saturating_add(size).min(self.limit);
        if daddr >= end {
            return;
        }
        if self.gpu_written.intersects(daddr, end) {
            self.gpu_written.subtract(daddr, end);
            self.disarm_written_back(daddr, end);
        }
        self.cpu_written(daddr, size);
    }

    /// Write tracking saw guest CPU writes in the pages of `[daddr, +size)`: they upload
    /// again before their next use, except bytes GPU work wrote. Those may lie beside the
    /// CPU's bytes on a page, so they stay until [`BufferCache::flush`] downloads them.
    pub fn cpu_written(&mut self, daddr: DAddr, size: u64) {
        let end = daddr.saturating_add(size).min(self.limit);
        if daddr >= end {
            return;
        }
        let mut page = daddr >> CACHING_PAGE_BITS;
        while page < end.div_ceil(CACHING_PAGE_SIZE) {
            let Some(id) = Self::lookup(&self.page_table, page << CACHING_PAGE_BITS) else {
                page += 1;
                continue;
            };
            let buffer = self.buffers[id].as_mut().expect("live buffer");
            let (start, stop) = (daddr.max(buffer.daddr), end.min(buffer.end()));
            buffer.mark_dirty(start, stop);
            buffer.untrack(&self.memory, start, stop);
            page = buffer.end() >> CACHING_PAGE_BITS;
        }
    }

    /// The GPU wrote `[daddr, +size)` (storage writes, transform feedback, copies): the bytes
    /// stay in the host buffer until [`BufferCache::flush`] downloads them, the pages are marked
    /// in the written-pages bitmap, and guest CPU accesses to them trap until then. Called when
    /// the writing work is recorded, with the tick it completes at ([`Scheduler::tick`]), which
    /// the buffers holding the range note for [`BufferCache::flush`].
    pub fn mark_gpu_written(&mut self, tick: u64, daddr: DAddr, size: u64) {
        if let Some(end) = self.track_gpu_write(daddr, size) {
            self.stamp_writer(tick, daddr, end);
        }
    }

    /// The draw or dispatch whose writable bindings were prepared since the last call is
    /// recorded in the work completing at `tick`: the buffers those bindings write note it for
    /// [`BufferCache::flush`]. Preparing bindings can submit recorded work (sparse binds, waits),
    /// so the tick is only known once the draw or dispatch is recorded.
    pub fn commit_writes(&mut self, tick: u64) {
        let mut pending = std::mem::take(&mut self.pending_writes);
        for &(daddr, end) in &pending {
            self.stamp_writer(tick, daddr, end);
        }
        pending.clear();
        self.pending_writes = pending;
    }

    fn bind_gpu_write(&mut self, daddr: DAddr, size: u64) {
        if let Some(end) = self.track_gpu_write(daddr, size) {
            self.pending_writes.push((daddr, end));
        }
    }

    fn track_gpu_write(&mut self, daddr: DAddr, size: u64) -> Option<DAddr> {
        let end = daddr.saturating_add(size).min(self.limit);
        if daddr >= end {
            return None;
        }
        self.written_pages.mark(daddr, end - daddr);
        self.captures.drop_overlapping(daddr, end);
        if self.flush_traps && !self.gpu_written.covers(daddr, end) {
            self.memory.arm_flush_traps(daddr, end - daddr);
        }
        self.gpu_written.insert(daddr, end);
        Some(end)
    }

    fn stamp_writer(&mut self, tick: u64, daddr: DAddr, end: DAddr) {
        let captures = &mut self.captures;
        captures.drop_overlapping(daddr, end);
        if captures.hot.intersects(daddr, end) {
            let queued = &mut captures.queued;
            captures
                .hot
                .for_each_in(daddr, end, |start, end| queued.push((start, end)));
        }
        let mut addr = daddr;
        while addr < end {
            let Some(id) = Self::lookup(&self.page_table, addr) else {
                addr = (addr + 1).next_multiple_of(CACHING_PAGE_SIZE);
                continue;
            };
            let buffer = self.buffers[id].as_mut().expect("live buffer");
            buffer.written_tick = tick;
            addr = buffer.end();
        }
    }

    /// Readies `[daddr, +size)` for GPU work about to be recorded that writes it: arms its flush
    /// traps, then uploads the pages the guest CPU wrote since write tracking was last drained.
    /// Such a write may have landed after the upload that fed the binding; once uploaded, the
    /// range's download gives it back instead of the bytes it replaced, and writes after the arm
    /// trap. Without flush traps, or over a range already GPU-written (and so armed), it does
    /// nothing.
    pub fn prepare_gpu_write(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        daddr: DAddr,
        size: u64,
    ) -> Result<()> {
        let end = daddr.saturating_add(size).min(self.limit);
        if !self.flush_traps || daddr >= end || self.gpu_written.covers(daddr, end) {
            return Ok(());
        }
        self.memory.arm_flush_traps(daddr, end - daddr);
        if !self.memory.has_cpu_writes() {
            return Ok(());
        }
        let mut raced = std::mem::take(&mut self.raced);
        self.memory
            .for_each_cpu_written_page(daddr, end - daddr, |page| match raced.last_mut() {
                Some((_, stop)) if *stop == page => *stop += PAGE_SIZE,
                _ => raced.push((page, page + PAGE_SIZE)),
            });
        let mut result = Ok(());
        for &(start, stop) in &raced {
            let (start, stop) = (start.max(daddr & !(PAGE_SIZE - 1)), stop.min(self.limit));
            let Some(id) = Self::lookup(&self.page_table, start) else {
                continue;
            };
            let buffer = self.buffers[id].as_mut().expect("live buffer");
            let stop = stop.min(buffer.end());
            buffer.mark_dirty(start, stop);
            buffer.untrack(&self.memory, start, stop);
            result = self.synchronize(scheduler, staging, id, start, stop - start);
            if result.is_err() {
                break;
            }
        }
        raced.clear();
        self.raced = raced;
        result
    }

    /// Ends the flush traps of the pages of `[daddr, end)` no GPU-written byte is left on.
    pub(crate) fn disarm_written_back(&self, daddr: DAddr, end: DAddr) {
        if !self.flush_traps {
            return;
        }
        let mut run: Option<DAddr> = None;
        let mut page = daddr & !(PAGE_SIZE - 1);
        while page < end {
            let next = page + PAGE_SIZE;
            if self.gpu_written.intersects(page, next) {
                if let Some(start) = run.take() {
                    self.memory.disarm_flush_traps(start, page - start);
                }
            } else {
                run.get_or_insert(page);
            }
            page = next;
        }
        if let Some(start) = run {
            self.memory.disarm_flush_traps(start, page - start);
        }
    }

    /// A guest CPU thread waits on a flush trap in `[daddr, +size)`: downloads the whole
    /// GPU-written range around it, so a CPU pass over the rest of the range traps no more.
    /// Returns that range.
    pub fn flush_for_cpu_read(
        &mut self,
        scheduler: &mut Scheduler,
        daddr: DAddr,
        size: u64,
    ) -> Result<(DAddr, DAddr)> {
        let (start, end) = self.gpu_written.area(daddr, daddr.saturating_add(size));
        self.flush(scheduler, start, end - start)?;
        Ok((start, end))
    }

    /// Copies the GPU-written bytes of `[daddr, +size)` back to guest memory, waiting for the GPU.
    pub fn flush(&mut self, scheduler: &mut Scheduler, daddr: DAddr, size: u64) -> Result<()> {
        let end = daddr.saturating_add(size).min(self.limit);
        if !self.gpu_written.intersects(daddr, end) || self.read_captured(scheduler, daddr, end)? {
            return Ok(());
        }
        let Self {
            page_table,
            buffers,
            gpu_written,
            downloads,
            captures,
            ..
        } = self;
        downloads.clear();
        let mut total = 0;
        let mut newest = 0;
        gpu_written.for_each_in(daddr, end, |start, end| {
            let mut addr = start;
            while addr < end {
                let Some(id) = Self::lookup(page_table, addr) else {
                    addr = (addr + 1).next_multiple_of(CACHING_PAGE_SIZE).min(end);
                    continue;
                };
                let buffer = buffers[id].as_ref().expect("live buffer");
                let len = end.min(buffer.end()) - addr;
                downloads.push((buffer.buffer.raw(), addr - buffer.daddr, addr, len));
                newest = newest.max(buffer.written_tick);
                total += len;
                if len <= CAPTURE_MAX {
                    captures.hot.insert(addr, addr + len);
                }
                addr += len;
            }
        });
        if total > self.readback.as_ref().map_or(0, Buffer::size) {
            self.readback = Some(Buffer::new(
                &self.device,
                total.next_power_of_two().max(CACHING_PAGE_SIZE),
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "buffer cache readback",
            )?);
        }
        let readback = self.readback.as_ref().expect("just ensured");
        let record = |cmd| {
            before_transfer(&self.device, cmd);
            let mut offset = 0;
            for &(src, src_offset, _, len) in &self.downloads {
                let region = [vk::BufferCopy {
                    src_offset,
                    dst_offset: offset,
                    size: len,
                }];
                copy_buffer(&self.device, cmd, src, readback.raw(), &region);
                offset += len;
            }
            transfer_to_all_barrier(&self.device, cmd);
        };
        if newest < scheduler.tick() {
            scheduler.run_aside(record)?;
        } else {
            record(scheduler.record()?);
            scheduler.finish()?;
        }
        let data = readback.mapped().expect("readback memory is host visible");
        let mut offset = 0;
        for &(_, _, daddr, len) in &self.downloads {
            let bytes = &data[offset as usize..(offset + len) as usize];
            if let Err(fault) = self.lazy.write(daddr, bytes) {
                tracing::warn!(
                    ?fault,
                    daddr,
                    "GPU-written buffer data over unmapped memory"
                );
            }
            offset += len;
        }
        self.gpu_written.subtract(daddr, end);
        self.disarm_written_back(daddr, end);
        self.stats.downloaded_bytes += total;
        Ok(())
    }

    /// Serves [`BufferCache::flush`] of `[daddr, end)` from captures when they hold every
    /// GPU-written byte there, waiting only for the work they were taken at the end of. `false`
    /// when they do not, leaving everything as it was.
    fn read_captured(
        &mut self,
        scheduler: &mut Scheduler,
        daddr: DAddr,
        end: DAddr,
    ) -> Result<bool> {
        let Self {
            gpu_written,
            captures,
            lazy,
            ..
        } = self;
        let Some(ring) = captures.ring.as_ref().filter(|_| !captures.live.is_empty()) else {
            return Ok(false);
        };
        let mut newest = 0;
        let mut covered = true;
        gpu_written.for_each_in(daddr, end, |start, end| {
            let mut addr = start;
            while covered && addr < end {
                match captures.find(addr) {
                    Some(capture) => {
                        newest = newest.max(capture.tick);
                        addr = capture.end;
                    }
                    None => covered = false,
                }
            }
        });
        if !covered || newest >= scheduler.tick() {
            return Ok(false);
        }
        scheduler.wait(newest)?;
        let data = ring.mapped().expect("capture memory is host visible");
        let mut total = 0;
        gpu_written.for_each_in(daddr, end, |start, end| {
            let mut addr = start;
            while addr < end {
                let capture = captures.find(addr).expect("covered above");
                let stop = end.min(capture.end);
                let from = (capture.offset + (addr - capture.daddr)) as usize;
                if let Err(fault) = lazy.write(addr, &data[from..from + (stop - addr) as usize]) {
                    tracing::warn!(
                        ?fault,
                        daddr = addr,
                        "GPU-written buffer data over unmapped memory"
                    );
                }
                total += stop - addr;
                addr = stop;
            }
        });
        captures.drop_overlapping(daddr, end);
        self.gpu_written.subtract(daddr, end);
        self.disarm_written_back(daddr, end);
        self.stats.downloaded_bytes += total;
        self.stats.captured_bytes += total;
        Ok(true)
    }

    /// Copies the GPU-written bytes of hot ranges that the work being recorded writes into the
    /// capture ring, at the end of that work, so a guest CPU read of them once it completes
    /// needs no GPU work of its own. Call right before the recorded work is submitted.
    pub fn capture(&mut self, scheduler: &mut Scheduler) -> Result<()> {
        if self.captures.queued.is_empty() {
            return Ok(());
        }
        if self.captures.ring.is_none() {
            self.captures.ring = Some(Buffer::new(
                &self.device,
                CAPTURE_RING_SIZE,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "buffer cache capture ring",
            )?);
        }
        let Self {
            device,
            page_table,
            buffers,
            gpu_written,
            captures,
            downloads,
            ..
        } = self;
        let queued = &mut captures.queued;
        queued.sort_unstable_by_key(|&(start, _)| start);
        let mut merged: usize = 0;
        for i in 0..queued.len() {
            let (start, end) = queued[i];
            match merged.checked_sub(1).map(|last| &mut queued[last]) {
                Some((_, last_end)) if *last_end >= start => *last_end = (*last_end).max(end),
                _ => {
                    queued[merged] = (start, end);
                    merged += 1;
                }
            }
        }
        downloads.clear();
        for &(start, end) in &queued[..merged] {
            gpu_written.for_each_in(start, end, |start, end| {
                let mut addr = start;
                while addr < end {
                    let Some(id) = Self::lookup(page_table, addr) else {
                        addr = (addr + 1).next_multiple_of(CACHING_PAGE_SIZE).min(end);
                        continue;
                    };
                    let buffer = buffers[id].as_ref().expect("live buffer");
                    let len = end.min(buffer.end()) - addr;
                    downloads.push((buffer.buffer.raw(), addr - buffer.daddr, addr, len));
                    addr += len;
                }
            });
        }
        queued.clear();
        if downloads.is_empty() {
            return Ok(());
        }
        let ring = captures.ring.as_ref().expect("just ensured").raw();
        let tick = scheduler.tick();
        let cmd = scheduler.record()?;
        before_transfer(device, cmd);
        for &(src, src_offset, daddr, len) in downloads.iter() {
            if len > CAPTURE_RING_SIZE {
                continue;
            }
            if captures.head + len > CAPTURE_RING_SIZE {
                captures.head = 0;
                before_transfer(device, cmd);
            }
            let offset = captures.head;
            captures
                .live
                .retain(|c| c.offset + (c.end - c.daddr) <= offset || offset + len <= c.offset);
            captures.drop_overlapping(daddr, daddr + len);
            let region = [vk::BufferCopy {
                src_offset,
                dst_offset: offset,
                size: len,
            }];
            copy_buffer(device, cmd, src, ring, &region);
            self.stats.capture_copy_bytes += len;
            captures.live.push(Capture {
                daddr,
                end: daddr + len,
                offset,
                tick,
            });
            captures.head = (offset + len).next_multiple_of(16);
        }
        transfer_to_all_barrier(device, cmd);
        Ok(())
    }

    /// The host buffer holding `[daddr, +size)` of device memory, its dirty pages uploaded.
    pub fn acquire(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        daddr: DAddr,
        size: u64,
    ) -> Result<BufferBinding> {
        assert!(
            daddr < self.limit,
            "{daddr:#x} is outside the device address space"
        );
        let size = size.min(self.limit - daddr);
        let id = match Self::lookup(&self.page_table, daddr) {
            Some(id) if self.buffer(id).contains(daddr, size.max(1)) => id,
            _ => self.create(scheduler, daddr, size.max(1))?,
        };
        if self.buffer(id).dirty_pages != 0 {
            self.synchronize(scheduler, staging, id, daddr, size)?;
        }
        let buffer = self.buffer(id);
        Ok(BufferBinding {
            buffer: buffer.buffer.raw(),
            offset: daddr - buffer.daddr,
            size,
        })
    }

    fn synchronize(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        id: usize,
        daddr: DAddr,
        size: u64,
    ) -> Result<()> {
        let Self {
            buffers,
            uploads,
            copies,
            gpu_written,
            ..
        } = self;
        let memory = &self.memory;
        let buffer = buffers[id].as_mut().expect("live buffer");
        let base = buffer.daddr;
        let first = ((daddr - base) >> PAGE_BITS) as usize;
        let last = (daddr + size - base).div_ceil(PAGE_SIZE) as usize;
        uploads.clear();
        let tracked = &mut buffer.tracked;
        buffer.dirty_pages -= drain_runs(&mut buffer.dirty, first, last, |first, last| {
            let start = base + first as u64 * PAGE_SIZE;
            let end = base + last as u64 * PAGE_SIZE;
            tracked.push((start, end, memory.cache_pages(start, end - start)));
            let mut next = start;
            gpu_written.for_each_in(start, end, |written, written_end| {
                if written > next {
                    uploads.push((next, written - next));
                }
                next = written_end;
            });
            if end > next {
                uploads.push((next, end - next));
            }
        });
        if uploads.is_empty() {
            return Ok(());
        }
        let dst = buffer.buffer.raw();
        let total = uploads.iter().map(|&(_, len)| len).sum();
        let slice = staging.alloc(scheduler, total, 16)?;
        copies.clear();
        let mut offset = 0;
        for &(addr, len) in uploads.iter() {
            read_guest(
                &self.memory,
                addr,
                &mut slice.data[offset as usize..(offset + len) as usize],
            );
            copies.push(vk::BufferCopy {
                src_offset: slice.offset + offset,
                dst_offset: addr - base,
                size: len,
            });
            offset += len;
        }
        let src = slice.buffer;
        let cmd = scheduler.record()?;
        before_transfer(&self.device, cmd);
        copy_buffer(&self.device, cmd, src, dst, copies);
        transfer_to_all_barrier(&self.device, cmd);
        self.stats.uploaded_bytes += total;
        Ok(())
    }

    /// Creates the buffer for `[daddr, +size)`, absorbing every buffer on its caching pages.
    fn create(&mut self, scheduler: &mut Scheduler, daddr: DAddr, size: u64) -> Result<usize> {
        let mut begin = daddr & !(CACHING_PAGE_SIZE - 1);
        let mut end = (daddr + size)
            .next_multiple_of(CACHING_PAGE_SIZE)
            .min(self.limit);
        self.overlaps.clear();
        let mut score = 0;
        let mut leap = false;
        let mut page = begin >> CACHING_PAGE_BITS;
        while page < end >> CACHING_PAGE_BITS {
            let entry = self.page_table[page as usize];
            page += 1;
            if entry == 0 || self.overlaps.contains(&(entry as usize - 1)) {
                continue;
            }
            let id = entry as usize - 1;
            self.overlaps.push(id);
            let overlap = self.buffer(id);
            let expands_left = overlap.daddr < begin;
            let expands_right = overlap.end() > end;
            begin = begin.min(overlap.daddr);
            end = end.max(overlap.end());
            score += overlap.stream_score;
            if score > STREAM_LEAP_THRESHOLD && !leap {
                leap = true;
                if expands_right {
                    begin = begin.saturating_sub(STREAM_LEAP_BYTES);
                    page = begin >> CACHING_PAGE_BITS;
                }
                if expands_left {
                    end = (end + STREAM_LEAP_BYTES).min(self.limit);
                }
            }
        }
        let size = end - begin;
        let pages = (size / PAGE_SIZE) as usize;
        let mut dirty = vec![u64::MAX; pages.div_ceil(64)];
        if !pages.is_multiple_of(64) {
            *dirty.last_mut().expect("a buffer has pages") = (1 << (pages % 64)) - 1;
        }
        let buffer = Buffer::new(
            &self.device,
            size,
            CACHED_USAGE,
            MemoryLocation::GpuOnly,
            "guest buffer",
        )?;
        self.device.name(
            buffer.raw(),
            format_args!("guest buffer {begin:#x}-{end:#x}"),
        );
        let mut new = CachedBuffer {
            buffer,
            daddr: begin,
            size,
            dirty,
            dirty_pages: pages as u64,
            stream_score: 0,
            views: Vec::new(),
            tracked: Vec::new(),
            written_tick: scheduler.tick(),
        };
        if !self.overlaps.is_empty() {
            let cmd = scheduler.record()?;
            before_transfer(&self.device, cmd);
            for i in 0..self.overlaps.len() {
                let id = self.overlaps[i];
                let mut old = self.buffers[id].take().expect("live buffer");
                new.tracked.append(&mut old.tracked);
                self.free_slots.push(id);
                let offset = old.daddr - begin;
                let region = [vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: offset,
                    size: old.size,
                }];
                copy_buffer(
                    &self.device,
                    cmd,
                    old.buffer.raw(),
                    new.buffer.raw(),
                    &region,
                );
                let first = (offset / PAGE_SIZE) as usize;
                for page in 0..(old.size / PAGE_SIZE) as usize {
                    if old.dirty[page / 64] & (1 << (page % 64)) == 0 {
                        let page = first + page;
                        new.dirty[page / 64] &= !(1 << (page % 64));
                        new.dirty_pages -= 1;
                    }
                }
                if !leap {
                    new.stream_score += old.stream_score + 1;
                }
                scheduler.retire((old.buffer, old.views));
                self.stats.merged_buffers += 1;
            }
            transfer_to_all_barrier(&self.device, cmd);
        }
        let id = match self.free_slots.pop() {
            Some(id) => id,
            None => {
                self.buffers.push(None);
                self.buffers.len() - 1
            }
        };
        self.buffers[id] = Some(new);
        let first = (begin >> CACHING_PAGE_BITS) as usize;
        let last = (end >> CACHING_PAGE_BITS) as usize;
        self.page_table[first..last].fill(id as u32 + 1);
        self.stats.created_buffers += 1;
        Ok(id)
    }

    /// Whether any device memory behind `[gpu_va, +size)` holds GPU-written bytes.
    fn gpu_written_in(&self, gmmu: &Gmmu, gpu_va: GpuVAddr, size: u64) -> bool {
        if self.gpu_written.is_empty() {
            return false;
        }
        let mut written = false;
        gmmu.for_each_range(gpu_va, size, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                written |= self.gpu_written.intersects(daddr, daddr + range.size);
            }
        });
        written
    }

    /// Copies a GPU range that is not one device range into stream memory: device runs from
    /// their cached buffers, everything else as zero. The GPU writes the zeros, so a gather
    /// costs the host nothing per byte.
    fn gather(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
        align: u64,
    ) -> Result<BufferBinding> {
        self.segments.clear();
        gmmu.for_each_range(gpu_va, size, |range| self.segments.push(range));
        let padded = size.next_multiple_of(4);
        let slice = self.stream.alloc(scheduler, padded, align.max(4))?;
        let (dst, dst_offset) = (slice.buffer, slice.offset);
        let cmd = scheduler.record()?;
        before_transfer(&self.device, cmd);
        let mut holes = false;
        let device = |range: &GpuRange| matches!(range.target, RangeTarget::Device(_));
        for run in self.segments.chunk_by(|a, b| device(a) == device(b)) {
            if device(&run[0]) {
                continue;
            }
            let last = run[run.len() - 1];
            let start = (run[0].gpu_va - gpu_va) & !3;
            let end = (last.gpu_va + last.size - gpu_va).next_multiple_of(4);
            // SAFETY: `cmd` is recording and `[start, end)` lies inside the padded slice, at
            // 4-byte aligned offsets. Runs between holes are whole GMMU pages, so no two widened
            // holes overlap.
            unsafe {
                self.device
                    .raw
                    .cmd_fill_buffer(cmd, dst, dst_offset + start, end - start, 0)
            };
            holes = true;
        }
        if holes {
            before_transfer(&self.device, cmd);
        }
        for i in 0..self.segments.len() {
            let segment = self.segments[i];
            let RangeTarget::Device(daddr) = segment.target else {
                continue;
            };
            let src = self.acquire(scheduler, staging, daddr, segment.size)?;
            let region = [vk::BufferCopy {
                src_offset: src.offset,
                dst_offset: dst_offset + (segment.gpu_va - gpu_va),
                size: src.size,
            }];
            copy_buffer(&self.device, scheduler.record()?, src.buffer, dst, &region);
        }
        transfer_to_all_barrier(&self.device, scheduler.record()?);
        self.stats.streamed_bytes += size;
        Ok(BufferBinding {
            buffer: dst,
            offset: dst_offset,
            size,
        })
    }

    /// `[gpu_va, +size)` for reading: its cached buffer when it is one device range at an
    /// `align`-aligned offset, else a gathered copy.
    fn read_range(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
        align: u64,
    ) -> Result<BufferBinding> {
        if let Some(daddr) = contiguous(gmmu, gpu_va, size) {
            let binding = self.acquire(scheduler, staging, daddr, size)?;
            if binding.offset % align == 0 {
                return Ok(binding);
            }
        }
        self.gather(scheduler, staging, gmmu, gpu_va, size, align)
    }

    /// The first run of `[gpu_va, +size)` when it is device memory, and its device address.
    fn first_run(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
    ) -> Result<Option<(BufferBinding, DAddr)>> {
        let mut first = None;
        gmmu.for_each_range(gpu_va, size, |range| {
            first.get_or_insert(range);
        });
        let Some(GpuRange {
            target: RangeTarget::Device(daddr),
            size,
            ..
        }) = first
        else {
            return Ok(None);
        };
        let binding = self.acquire(scheduler, staging, daddr, size)?;
        Ok(Some((binding, daddr)))
    }

    /// A vertex buffer binding for `[gpu_va, +size)` (see [`vertex_stream_range`]). Ranges over
    /// scattered memory are gathered, except huge ones, which bind only their first contiguous
    /// run as in Lichii.
    pub fn vertex_buffer(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
    ) -> Result<Option<BufferBinding>> {
        if gpu_va == 0 || size == 0 {
            return Ok(None);
        }
        if let Some(daddr) = contiguous(gmmu, gpu_va, size) {
            return self.acquire(scheduler, staging, daddr, size).map(Some);
        }
        if size >= MAX_GATHER_SIZE {
            let run = self.first_run(scheduler, staging, gmmu, gpu_va, size)?;
            return Ok(run.map(|(binding, _)| binding));
        }
        self.gather(scheduler, staging, gmmu, gpu_va, size, 16)
            .map(Some)
    }

    /// A uniform buffer binding for `[gpu_va, +size)`. Buffers up to [`UNIFORM_STREAM_MAX`]
    /// that hold no GPU-written bytes are copied from guest memory into stream memory with no
    /// GPU commands; larger ones bind the cached buffer.
    pub fn uniform_buffer(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
    ) -> Result<Option<BufferBinding>> {
        let limits = &self.device.properties.limits;
        let size = size.min(u64::from(limits.max_uniform_buffer_range));
        let align = limits.min_uniform_buffer_offset_alignment;
        if gpu_va == 0 || size == 0 {
            return Ok(None);
        }
        if size <= UNIFORM_STREAM_MAX && !self.gpu_written_in(gmmu, gpu_va, size) {
            let slice = self.stream.alloc(scheduler, size, align)?;
            let _ = gmmu.read_block(gpu_va, slice.data);
            let binding = BufferBinding {
                buffer: slice.buffer,
                offset: slice.offset,
                size,
            };
            self.stats.streamed_bytes += size;
            return Ok(Some(binding));
        }
        self.read_range(scheduler, staging, gmmu, gpu_va, size, align)
            .map(Some)
    }

    /// A uniform buffer binding of `bytes` followed by zeros up to `size` (at least the bytes,
    /// at most the device's uniform range), copied to the stream ring.
    pub fn stream_uniform(
        &mut self,
        scheduler: &mut Scheduler,
        bytes: &[u8],
        size: u32,
    ) -> Result<BufferBinding> {
        let limits = &self.device.properties.limits;
        let size = u64::from(size.min(limits.max_uniform_buffer_range)).max(bytes.len() as u64);
        let align = limits.min_uniform_buffer_offset_alignment;
        let slice = self.stream.alloc(scheduler, size, align)?;
        let (head, tail) = slice.data.split_at_mut(bytes.len());
        head.copy_from_slice(bytes);
        tail.fill(0);
        self.stats.streamed_bytes += size;
        Ok(BufferBinding {
            buffer: slice.buffer,
            offset: slice.offset,
            size,
        })
    }

    /// A storage buffer binding for `[gpu_va, +size)`, starting at `gpu_va` rounded down to
    /// [`BufferCache::storage_alignment`]. With `written`, the range is marked GPU-written; a
    /// written range over scattered memory binds only its first contiguous run (as Lichii does
    /// without sparse binding).
    pub fn storage_buffer(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
        written: bool,
    ) -> Result<Option<BufferBinding>> {
        if gpu_va == 0 || size == 0 {
            return Ok(None);
        }
        let align = self.storage_alignment();
        let start = gpu_va & !(align - 1);
        let size = size + (gpu_va - start);
        if !written {
            return self
                .read_range(scheduler, staging, gmmu, start, size, align)
                .map(Some);
        }
        if let Some(daddr) = contiguous(gmmu, start, size) {
            let binding = self.acquire(scheduler, staging, daddr, size)?;
            self.prepare_gpu_write(scheduler, staging, daddr, size)?;
            self.bind_gpu_write(daddr, size);
            return Ok(Some(binding));
        }
        if !self.warned_scattered_write {
            self.warned_scattered_write = true;
            tracing::warn!(
                gpu_va,
                size,
                "written storage buffer over scattered memory: binding its first run only"
            );
        }
        let Some((binding, daddr)) = self.first_run(scheduler, staging, gmmu, start, size)? else {
            return Ok(None);
        };
        self.prepare_gpu_write(scheduler, staging, daddr, binding.size)?;
        self.bind_gpu_write(daddr, binding.size);
        Ok(Some(binding))
    }

    /// A texel buffer view of `[gpu_va, +size)` read as `format`, `texel` bytes an element, over
    /// the range's first contiguous run as Lichii binds texture buffers. Views are created on
    /// first use and dropped with their buffer. With `written`, the viewed range is marked
    /// GPU-written. `None` when no device memory backs the range or the run's offset breaks
    /// the device's texel buffer alignment.
    #[allow(clippy::too_many_arguments)]
    pub fn texel_buffer(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        size: u64,
        (format, texel): (vk::Format, u64),
        written: bool,
    ) -> Result<Option<vk::BufferView>> {
        if gpu_va == 0 || size < texel {
            return Ok(None);
        }
        let Some((binding, daddr)) = self.first_run(scheduler, staging, gmmu, gpu_va, size)? else {
            return Ok(None);
        };
        let limits = &self.device.properties.limits;
        let elements = (binding.size / texel).min(u64::from(limits.max_texel_buffer_elements));
        if elements == 0 || binding.offset % limits.min_texel_buffer_offset_alignment != 0 {
            return Ok(None);
        }
        let range = elements * texel;
        if written {
            self.prepare_gpu_write(scheduler, staging, daddr, range)?;
            self.bind_gpu_write(daddr, range);
        }
        let id = Self::lookup(&self.page_table, daddr).expect("an acquired range has a buffer");
        let key = (binding.offset, range, format.as_raw());
        let buffer = self.buffers[id].as_mut().expect("live buffer");
        if let Some((_, view)) = buffer.views.iter().find(|(k, _)| *k == key) {
            return Ok(Some(view.raw()));
        }
        let view = BufferView::new(
            &self.device,
            buffer.buffer.raw(),
            format,
            binding.offset,
            range,
        )?;
        let raw = view.raw();
        buffer.views.push((key, view));
        Ok(Some(raw))
    }

    /// The index buffer for a draw, converted where the host cannot draw the guest's indices:
    /// quads and quad strips become triangle lists (generated indices for non-indexed draws),
    /// line loops line strips with the closing index, and `u8` indices `u16` without
    /// `VK_EXT_index_type_uint8`. `None` when the host draw is not indexed. The pipeline uses
    /// [`assembled_topology`].
    pub fn index_buffer(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        gmmu: &Gmmu,
        regs: &Regs,
        draw: &DrawParams<'_>,
    ) -> Result<Option<IndexBinding>> {
        let assembly = Assembly::of(draw.topology);
        let restart = regs.primitive_restart.enabled != 0;
        if let Some(inline) = draw.inline_indices {
            let indices = sub_indices(inline, draw.first, draw.count);
            return self
                .assemble_indices(scheduler, assembly, indices, restart, draw.base_vertex)
                .map(Some);
        }
        if !draw.indexed {
            return match assembly {
                Assembly::Direct => Ok(None),
                Assembly::LineLoop => self.line_loop(scheduler, draw).map(Some),
                quads => self.quad_lut(scheduler, staging, quads, draw).map(Some),
            };
        }
        let index_buffer = &regs.index_buffer;
        let (width, index_type) = match index_buffer.format {
            IndexFormat::UNSIGNED_BYTE => (1, vk::IndexType::UINT8_EXT),
            IndexFormat::UNSIGNED_SHORT => (2, vk::IndexType::UINT16),
            IndexFormat::UNSIGNED_INT => (4, vk::IndexType::UINT32),
            _ => return Ok(Some(IndexBinding::EMPTY)),
        };
        let start = index_buffer.start_addr();
        let available = index_capacity(index_buffer, width).unwrap_or(u64::from(u32::MAX));
        let first = u64::from(draw.first).min(available);
        let count = u64::from(draw.count).min(available - first) as u32;
        if start == 0 || count == 0 {
            return Ok(Some(IndexBinding::EMPTY));
        }
        if assembly == Assembly::Direct && !(self.widen_u8 && width == 1) {
            let size = (first + u64::from(count)) * width;
            let binding = self.read_range(scheduler, staging, gmmu, start, size, width)?;
            return Ok(Some(IndexBinding {
                buffer: binding.buffer,
                offset: binding.offset,
                index_type,
                first_index: first as u32,
                index_count: count,
                vertex_offset: draw.base_vertex,
            }));
        }
        let gpu_va = start + first * width;
        let bytes = u64::from(count) * width;
        if !self.gpu_written.is_empty() {
            self.segments.clear();
            gmmu.for_each_range(gpu_va, bytes, |range| self.segments.push(range));
            for i in 0..self.segments.len() {
                let segment = self.segments[i];
                if let RangeTarget::Device(daddr) = segment.target {
                    self.flush(scheduler, daddr, segment.size)?;
                }
            }
        }
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(bytes.div_ceil(4) as usize, 0);
        let _ = gmmu.read_block(
            gpu_va,
            &mut bytemuck::cast_slice_mut(&mut scratch)[..bytes as usize],
        );
        let count = count as usize;
        let indices = match width {
            1 => InlineIndices::U8(&bytemuck::cast_slice(&scratch)[..count]),
            2 => InlineIndices::U16(&bytemuck::cast_slice(&scratch)[..count]),
            _ => InlineIndices::U32(&scratch[..count]),
        };
        let binding =
            self.assemble_indices(scheduler, assembly, indices, restart, draw.base_vertex);
        self.scratch = scratch;
        binding.map(Some)
    }

    fn assemble_indices(
        &mut self,
        scheduler: &mut Scheduler,
        assembly: Assembly,
        indices: InlineIndices<'_>,
        restart: bool,
        vertex_offset: i32,
    ) -> Result<IndexBinding> {
        let count = indices_len(indices);
        let index_type = match (assembly, indices) {
            (Assembly::Direct, InlineIndices::U8(_)) if self.widen_u8 => vk::IndexType::UINT16,
            (Assembly::Direct, InlineIndices::U8(_)) => vk::IndexType::UINT8_EXT,
            (Assembly::Direct, InlineIndices::U16(_)) => vk::IndexType::UINT16,
            _ => vk::IndexType::UINT32,
        };
        let index = |i| index_at(indices, restart, i);
        let index_count = assembled_count(assembly, count, index, restart);
        if index_count == 0 {
            return Ok(IndexBinding::EMPTY);
        }
        let size = u64::from(index_count) * index_size(index_type) as u64;
        let slice = self.stream.alloc(scheduler, size, 4)?;
        assemble(assembly, count, index, restart, index_type, slice.data);
        let binding = IndexBinding {
            buffer: slice.buffer,
            offset: slice.offset,
            index_type,
            first_index: 0,
            index_count,
            vertex_offset,
        };
        self.stats.streamed_bytes += size;
        Ok(binding)
    }

    fn line_loop(
        &mut self,
        scheduler: &mut Scheduler,
        draw: &DrawParams<'_>,
    ) -> Result<IndexBinding> {
        let index_count = Assembly::LineLoop.output_count(draw.count);
        if index_count == 0 {
            return Ok(IndexBinding::EMPTY);
        }
        let size = u64::from(index_count) * 4;
        let slice = self.stream.alloc(scheduler, size, 4)?;
        assemble(
            Assembly::LineLoop,
            draw.count as usize,
            |i| i as u32,
            false,
            vk::IndexType::UINT32,
            slice.data,
        );
        let binding = IndexBinding {
            buffer: slice.buffer,
            offset: slice.offset,
            index_type: vk::IndexType::UINT32,
            first_index: 0,
            index_count,
            vertex_offset: draw.first as i32,
        };
        self.stats.streamed_bytes += size;
        Ok(binding)
    }

    /// Generated triangle indices for a non-indexed quad or quad strip draw, grown by half when
    /// a draw needs more so a creeping count does not regenerate them every draw.
    fn quad_lut(
        &mut self,
        scheduler: &mut Scheduler,
        staging: &mut StagingRing,
        assembly: Assembly,
        draw: &DrawParams<'_>,
    ) -> Result<IndexBinding> {
        let index_count = assembly.output_count(draw.count);
        if index_count == 0 {
            return Ok(IndexBinding::EMPTY);
        }
        let lut = &mut self.luts[usize::from(assembly == Assembly::QuadStrip)];
        let quads = index_count / 6;
        if lut.buffer.is_none() || quads > lut.quads {
            let quads = quads.max(lut.quads + lut.quads / 2);
            let size = u64::from(quads) * 6 * 4;
            let buffer = Buffer::new(
                &self.device,
                size,
                vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "quad indices",
            )?;
            let vertices = match assembly {
                Assembly::Quads => quads * 4,
                _ => quads * 2 + 2,
            };
            let slice = staging.alloc(scheduler, size, 16)?;
            assemble(
                assembly,
                vertices as usize,
                |i| i as u32,
                false,
                vk::IndexType::UINT32,
                slice.data,
            );
            let region = [vk::BufferCopy {
                src_offset: slice.offset,
                dst_offset: 0,
                size,
            }];
            let src = slice.buffer;
            let cmd = scheduler.record()?;
            before_transfer(&self.device, cmd);
            copy_buffer(&self.device, cmd, src, buffer.raw(), &region);
            transfer_to_all_barrier(&self.device, cmd);
            if let Some(old) = lut.buffer.replace(buffer) {
                scheduler.retire(old);
            }
            lut.quads = quads;
        }
        Ok(IndexBinding {
            buffer: lut.buffer.as_ref().expect("just ensured").raw(),
            offset: 0,
            index_type: vk::IndexType::UINT32,
            first_index: 0,
            index_count,
            vertex_offset: draw.first as i32,
        })
    }
}

/// Bytes from its start that an index buffer without a programmed limit binds for draws whose
/// counts only the GPU knows (indirect and assembled draws).
const UNSET_LIMIT_SPAN: u64 = 16 << 20;

/// Indices of `width` bytes up to the guest index buffer's inclusive limit. `None` when the
/// limit lies below the start: Switch 2 never programs it, and the reference (which takes the
/// span as a wrapping `u32`) does not clamp those draws.
pub(crate) fn index_capacity(buffer: &IndexBuffer, width: u64) -> Option<u64> {
    let span = buffer.limit_addr().checked_sub(buffer.start_addr())?;
    Some((span + 1) / width)
}

/// [`index_capacity`] for draws whose counts live in GPU memory: without a programmed limit,
/// the indices mapped contiguously from the start, at most [`UNSET_LIMIT_SPAN`] bytes.
pub(crate) fn bound_index_capacity(gmmu: &Gmmu, buffer: &IndexBuffer, width: u64) -> u64 {
    index_capacity(buffer, width).unwrap_or_else(|| {
        let mut mapped = 0;
        let mut contiguous = true;
        gmmu.for_each_range(buffer.start_addr(), UNSET_LIMIT_SPAN, |range| {
            contiguous &= matches!(range.target, RangeTarget::Device(_));
            if contiguous {
                mapped += range.size;
            }
        });
        mapped / width
    })
}

#[cfg(test)]
mod tests {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::time::Instant;

    use lichii_common::settings::GraphicsSettings;
    use lichii_gpu::gmmu::BigPageSize;
    use lichii_memory::{AddressSpace, MemoryPermission, PhysicalMemory};

    use super::*;
    use crate::{Instance, SurfaceSupport};

    thread_local! {
        static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    }

    /// Counts this thread's heap allocations, so a test can prove a path allocates nothing.
    struct CountingAllocator;

    // SAFETY: every call forwards to the system allocator unchanged.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            // SAFETY: the caller's layout is passed through.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: `ptr` came from `alloc` with this layout.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
            // SAFETY: `ptr` came from `alloc` with this layout.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    fn allocations() -> u64 {
        ALLOCATIONS.with(Cell::get)
    }

    const DRAM: u64 = 16 << 20;
    const DADDR: DAddr = 0x1_0000_0000;
    const GPU_VA: GpuVAddr = 0x10_0000;

    struct Fixture {
        scheduler: Scheduler,
        staging: StagingRing,
        cache: BufferCache,
        device: Arc<Device>,
        memory: Arc<DeviceAddressSpace>,
        gmmu: Gmmu,
    }

    fn fixture() -> Option<Fixture> {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        let device = match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
        physical.commit(0, DRAM).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, DRAM);
        Some(Fixture {
            scheduler: Scheduler::new(&device).unwrap(),
            staging: StagingRing::new(&device, 8 << 20).unwrap(),
            cache: BufferCache::new(
                &device,
                memory.clone(),
                Arc::new(GpuWrittenPages::new(36)),
                Arc::new(LazyDeviceRanges::new(memory.clone())),
            )
            .unwrap(),
            gmmu: Gmmu::new(memory.clone(), BigPageSize::Size64K),
            device,
            memory,
        })
    }

    impl Fixture {
        fn acquire(&mut self, daddr: DAddr, size: u64) -> BufferBinding {
            self.cache
                .acquire(&mut self.scheduler, &mut self.staging, daddr, size)
                .unwrap()
        }

        fn read_back(&mut self, buffer: vk::Buffer, offset: u64, size: u64) -> Vec<u8> {
            let readback = Buffer::new(
                &self.device,
                size,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                "test readback",
            )
            .unwrap();
            let cmd = self.scheduler.record().unwrap();
            before_transfer(&self.device, cmd);
            let region = [vk::BufferCopy {
                src_offset: offset,
                dst_offset: 0,
                size,
            }];
            copy_buffer(&self.device, cmd, buffer, readback.raw(), &region);
            transfer_to_all_barrier(&self.device, cmd);
            self.scheduler.finish().unwrap();
            readback.mapped().unwrap().to_vec()
        }

        fn read_binding(&mut self, binding: BufferBinding) -> Vec<u8> {
            self.read_back(binding.buffer, binding.offset, binding.size)
        }

        fn read_indices(&mut self, binding: IndexBinding) -> Vec<u32> {
            let width = index_size(binding.index_type);
            let start = binding.offset + u64::from(binding.first_index) * width as u64;
            let bytes = self.read_back(
                binding.buffer,
                start,
                u64::from(binding.index_count) * width as u64,
            );
            bytes
                .chunks_exact(width)
                .map(|c| c.iter().rev().fold(0, |v, &b| v << 8 | u32::from(b)))
                .collect()
        }

        /// A shader-style GPU write: fills `size` bytes of the binding at `offset` with `value`.
        fn gpu_fill(&mut self, binding: BufferBinding, offset: u64, size: u64, value: u32) {
            let cmd = self.scheduler.record().unwrap();
            before_transfer(&self.device, cmd);
            // SAFETY: `cmd` is recording and the range lies inside the buffer.
            unsafe {
                self.device.raw.cmd_fill_buffer(
                    cmd,
                    binding.buffer,
                    binding.offset + offset,
                    size,
                    value,
                )
            };
            transfer_to_all_barrier(&self.device, cmd);
        }

        /// Submits the recorded work the way the rasterizer does.
        fn submit(&mut self) {
            self.cache.capture(&mut self.scheduler).unwrap();
            self.scheduler.flush().unwrap();
        }

        /// Makes `[daddr, +size)` of `binding` hot: GPU-written, then downloaded.
        fn read_once(&mut self, binding: BufferBinding, daddr: DAddr, size: u64) {
            self.gpu_fill(binding, daddr - DADDR - binding.offset, size, 0x0101_0101);
            self.cache
                .mark_gpu_written(self.scheduler.tick(), daddr, size);
            self.submit();
            self.cache.flush(&mut self.scheduler, daddr, size).unwrap();
            assert_eq!(
                self.cache.stats().captured_bytes,
                0,
                "nothing was captured yet"
            );
        }

        fn guest(&self, daddr: DAddr, size: usize) -> Vec<u8> {
            let mut out = vec![0; size];
            self.memory.read_bytes(daddr, &mut out).unwrap();
            out
        }

        fn index_buffer(&mut self, regs: &Regs, draw: &DrawParams<'_>) -> Option<IndexBinding> {
            self.cache
                .index_buffer(
                    &mut self.scheduler,
                    &mut self.staging,
                    &self.gmmu,
                    regs,
                    draw,
                )
                .unwrap()
        }

        fn assert_no_validation_errors(&self) {
            assert_eq!(
                self.device.instance().validation_errors(),
                0,
                "validation reported errors"
            );
        }
    }

    fn pattern(len: usize, seed: u32) -> Vec<u8> {
        (0..len as u32)
            .map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 24) as u8)
            .collect()
    }

    fn draw(
        topology: PrimitiveTopology,
        indexed: bool,
        first: u32,
        count: u32,
    ) -> DrawParams<'static> {
        DrawParams {
            topology,
            indexed,
            first,
            count,
            base_vertex: 0,
            base_instance: 0,
            instance_count: 1,
            inline_indices: None,
        }
    }

    #[test]
    fn range_set_merges_and_subtracts() {
        let mut set = RangeSet::default();
        set.insert(10, 20);
        set.insert(30, 40);
        set.insert(20, 25);
        assert_eq!(
            set.ranges.iter().collect::<Vec<_>>(),
            [(&10, &25), (&30, &40)]
        );
        set.insert(5, 35);
        assert_eq!(set.ranges.iter().collect::<Vec<_>>(), [(&5, &40)]);
        set.subtract(10, 12);
        set.subtract(38, 50);
        assert_eq!(
            set.ranges.iter().collect::<Vec<_>>(),
            [(&5, &10), (&12, &38)]
        );
        let mut parts = Vec::new();
        set.for_each_in(8, 20, |s, e| parts.push((s, e)));
        assert_eq!(parts, [(8, 10), (12, 20)]);
        assert!(set.intersects(37, 100) && !set.intersects(10, 12) && !set.intersects(38, 60));
    }

    #[test]
    fn switch2_vertex_streams_are_sized_by_their_size_registers() {
        use lichii_gpu::engines::maxwell_3d::regs::VertexStreamBits;
        const ADDRESS: u64 = 0x14_93ef_9000;
        let mut regs = Regs::with_defaults();
        let stream = &mut regs.vertex_streams[1];
        stream.bits = VertexStreamBits(24 | 1 << 12);
        stream.address_high = (ADDRESS >> 32) as u32;
        stream.address_low = ADDRESS as u32;
        regs.words_mut()[VERTEX_STREAM_SIZE + 3] = 96;
        assert_eq!(
            vertex_stream_range(&regs, 1, ShaderArch::Ampere),
            Some((ADDRESS, 96))
        );
        assert_eq!(
            vertex_stream_range(&regs, 1, ShaderArch::Maxwell),
            None,
            "a zero limit is below the stream"
        );

        let limit = ADDRESS + 95;
        regs.vertex_stream_limits[1].address_high = (limit >> 32) as u32;
        regs.vertex_stream_limits[1].address_low = limit as u32;
        regs.words_mut()[VERTEX_STREAM_SIZE + 2] = 0x101;
        assert_eq!(
            vertex_stream_range(&regs, 1, ShaderArch::Maxwell),
            Some((ADDRESS, 96))
        );
        assert_eq!(
            vertex_stream_range(&regs, 1, ShaderArch::Ampere),
            Some((ADDRESS, 1 << 32 | 96)),
            "only the low byte of the high word counts"
        );

        regs.words_mut()[VERTEX_STREAM_SIZE + 2] = 0;
        regs.words_mut()[VERTEX_STREAM_SIZE + 3] = 0;
        assert_eq!(vertex_stream_range(&regs, 1, ShaderArch::Ampere), None);
        regs.words_mut()[VERTEX_STREAM_SIZE + 3] = 96;
        regs.vertex_streams[1].bits = VertexStreamBits(24);
        assert_eq!(vertex_stream_range(&regs, 1, ShaderArch::Ampere), None);
    }

    #[test]
    fn assembles_quads_strips_and_loops() {
        let run = |assembly, indices: &[u16], restart, index_type| {
            let source = InlineIndices::U16(indices);
            let index = |i| index_at(source, restart, i);
            let count = assembled_count(assembly, indices.len(), index, restart) as usize;
            let mut out = vec![0; count * index_size(index_type)];
            assemble(
                assembly,
                indices.len(),
                index,
                restart,
                index_type,
                &mut out,
            );
            out.chunks_exact(index_size(index_type))
                .map(|c| c.iter().rev().fold(0, |v, &b| v << 8 | u32::from(b)))
                .collect::<Vec<u32>>()
        };
        let u32_ = vk::IndexType::UINT32;
        assert_eq!(
            run(Assembly::Quads, &[1, 2, 3, 4, 5, 6, 7, 8, 9], false, u32_),
            [1, 2, 3, 1, 3, 4, 5, 6, 7, 5, 7, 8]
        );
        assert_eq!(
            run(Assembly::QuadStrip, &[0, 1, 2, 3, 4, 5], false, u32_),
            [0, 3, 1, 0, 2, 3, 2, 5, 3, 2, 4, 5]
        );
        assert_eq!(
            run(Assembly::LineLoop, &[4, 5, 6], false, u32_),
            [4, 5, 6, 4]
        );
        assert_eq!(run(Assembly::LineLoop, &[1, 0xffff], true, u32_), [1, 1]);
        assert_eq!(
            run(Assembly::LineLoop, &[1, 2, 3, 0xffff, 4, 5, 6], true, u32_),
            [1, 2, 3, 1, u32::MAX, 4, 5, 6, 4]
        );
        assert_eq!(
            run(Assembly::LineLoop, &[1, 2, 3, 0xffff, 4, 5, 6], false, u32_),
            [1, 2, 3, 0xffff, 4, 5, 6, 1]
        );
        assert_eq!(
            run(
                Assembly::Quads,
                &[0, 1, 2, 3, 0xffff, 9, 4, 5, 6, 7],
                true,
                u32_
            ),
            [0, 1, 2, 0, 2, 3, 9, 4, 5, 9, 5, 6]
        );
        assert_eq!(
            run(
                Assembly::QuadStrip,
                &[0, 1, 2, 3, 0xffff, 4, 5, 6, 7, 0xffff],
                true,
                u32_
            ),
            [0, 3, 1, 0, 2, 3, 4, 7, 5, 4, 6, 7]
        );
        assert_eq!(
            run(Assembly::Direct, &[1, 0xffff], false, u32_),
            [1, 0xffff]
        );
        assert!(run(Assembly::QuadStrip, &[0, 1, 2], false, u32_).is_empty());
    }

    #[test]
    fn uploads_and_reads_back() {
        let Some(mut f) = fixture() else { return };
        let data = pattern(0x5000, 1);
        f.memory.write_bytes(DADDR + 0x1234, &data).unwrap();
        let binding = f.acquire(DADDR + 0x1234, 0x5000);
        assert_eq!((binding.offset, binding.size), (0x1234, 0x5000));
        assert!(f.read_binding(binding) == data, "uploaded bytes differ");
        let stats = f.cache.stats();
        assert_eq!(stats.uploaded_bytes, 6 * PAGE_SIZE, "pages 1 to 6 upload");
        assert_eq!(stats.created_buffers, 1);
        f.assert_no_validation_errors();
    }

    /// Guest CPU writes to uploaded pages come back through write tracking: only the written
    /// pages upload again, the rest stay tracked, and a page uploaded anew is tracked anew.
    #[test]
    fn cpu_writes_to_uploaded_pages_are_tracked() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        space.write_bytes(CPU_VA, &pattern(0x10000, 2)).unwrap();
        let binding = f.acquire(DADDR, 0x10000);
        for (offset, byte) in [(0x5010, 7), (0x9ffc, 8), (0x5010, 9)] {
            space.write_bytes(CPU_VA + offset, &[byte; 8]).unwrap();
            assert!(
                f.memory.has_cpu_writes(),
                "{offset:#x}: the write was tracked"
            );
            let memory = f.memory.clone();
            memory.take_cpu_writes(|daddr, size| f.cache.cpu_written(daddr, size));
            f.acquire(DADDR, 0x10000);
        }
        assert_eq!(
            f.cache.stats().uploaded_bytes,
            0x10000 + 4 * PAGE_SIZE,
            "pages 5, 9, 10 and 5 again"
        );
        let mut guest = vec![0; 0x10000];
        space.read_bytes(CPU_VA, &mut guest).unwrap();
        assert!(
            f.read_binding(binding) == guest,
            "the buffer holds the CPU's bytes"
        );
        f.assert_no_validation_errors();
    }

    /// GPU-written bytes whose writer was submitted download ahead of the work recorded since,
    /// which stays unsubmitted; a writer still being recorded is submitted with the download.
    #[test]
    fn downloads_wait_only_for_submitted_writers() {
        let Some(mut f) = fixture() else { return };
        let binding = f.acquire(DADDR, 0x1000);
        f.gpu_fill(binding, 0, 0x100, 0x1111_1111);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x100);
        f.scheduler.flush().unwrap();
        f.gpu_fill(binding, 0x800, 0x100, 0x2222_2222);
        let tick = f.scheduler.tick();
        f.cache.flush(&mut f.scheduler, DADDR, 0x100).unwrap();
        assert_eq!(
            f.scheduler.tick(),
            tick,
            "the recorded work stays unsubmitted"
        );
        assert_eq!(
            f.guest(DADDR, 0x100),
            0x1111_1111_u32.to_le_bytes().repeat(0x40)
        );
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x800, 0x100);
        f.cache
            .flush(&mut f.scheduler, DADDR + 0x800, 0x100)
            .unwrap();
        assert!(
            f.scheduler.tick() > tick,
            "a recorded writer is submitted with the download"
        );
        assert_eq!(
            f.guest(DADDR + 0x800, 0x100),
            0x2222_2222_u32.to_le_bytes().repeat(0x40)
        );
        f.assert_no_validation_errors();
    }

    /// A writable binding prepared before a submission belongs to the work its draw is recorded
    /// into, not to the work submitted while the bindings were prepared.
    #[test]
    fn writable_bindings_belong_to_the_work_they_are_recorded_into() {
        let Some(mut f) = fixture() else { return };
        f.gmmu.map(GPU_VA, DADDR, PAGE_SIZE, 0, false).unwrap();
        let binding = f
            .cache
            .storage_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA,
                0x100,
                true,
            )
            .unwrap()
            .unwrap();
        f.scheduler.flush().unwrap();
        f.gpu_fill(binding, 0, 0x100, 0x3333_3333);
        let tick = f.scheduler.tick();
        f.cache.commit_writes(tick);
        f.cache.flush(&mut f.scheduler, DADDR, 0x100).unwrap();
        assert!(
            f.scheduler.tick() > tick,
            "the draw's work is submitted with the download"
        );
        assert_eq!(
            f.guest(DADDR, 0x100),
            0x3333_3333_u32.to_le_bytes().repeat(0x40)
        );
        f.assert_no_validation_errors();
    }

    fn words(value: u32, bytes: usize) -> Vec<u8> {
        value.to_le_bytes().repeat(bytes / 4)
    }

    /// A GPU write to a range a download read before is copied out at the end of its work: once
    /// that work is submitted, a download of the range takes the copy, behind no work recorded
    /// since.
    #[test]
    fn submitted_writes_to_read_ranges_download_from_their_capture() {
        let Some(mut f) = fixture() else { return };
        let binding = f.acquire(DADDR, 0x1000);
        f.read_once(binding, DADDR, 0x100);
        f.gpu_fill(binding, 0, 0x100, 0x2222_2222);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x100);
        f.submit();
        f.gpu_fill(binding, 0x800, 0x100, 0x3333_3333);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x800, 0x100);
        let tick = f.scheduler.tick();
        f.cache.flush(&mut f.scheduler, DADDR, 0x100).unwrap();
        assert_eq!(f.cache.stats().captured_bytes, 0x100);
        assert_eq!(
            f.scheduler.tick(),
            tick,
            "the recorded work stays unsubmitted"
        );
        assert_eq!(f.guest(DADDR, 0x100), words(0x2222_2222, 0x100));
        assert!(!f.cache.is_gpu_written(DADDR, 0x100));
        f.assert_no_validation_errors();
    }

    /// A GPU write recorded after a range's capture, not yet submitted, makes the capture stale
    /// where it lands: a download there waits for the write, and the rest still downloads from
    /// the capture.
    #[test]
    fn writes_recorded_after_a_capture_download_their_own_bytes() {
        let Some(mut f) = fixture() else { return };
        let binding = f.acquire(DADDR, 0x1000);
        f.read_once(binding, DADDR, 0x100);
        f.gpu_fill(binding, 0, 0x100, 0x2222_2222);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x100);
        f.submit();
        f.gpu_fill(binding, 0x40, 0x40, 0x4444_4444);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x40, 0x40);
        f.cache.flush(&mut f.scheduler, DADDR + 0x80, 0x80).unwrap();
        assert_eq!(f.cache.stats().captured_bytes, 0x80);
        assert_eq!(f.guest(DADDR + 0x80, 0x80), words(0x2222_2222, 0x80));
        let tick = f.scheduler.tick();
        f.cache.flush(&mut f.scheduler, DADDR, 0x100).unwrap();
        assert_eq!(
            f.cache.stats().captured_bytes,
            0x80,
            "the capture was stale"
        );
        assert!(
            f.scheduler.tick() > tick,
            "the write was submitted with the download"
        );
        let mut expected = words(0x2222_2222, 0x100);
        expected[0x40..0x80].copy_from_slice(&words(0x4444_4444, 0x40));
        assert_eq!(f.guest(DADDR, 0x100), expected);
        f.assert_no_validation_errors();
    }

    /// A writable binding prepared before a submission that captured its range, its draw
    /// recorded after: the capture predates the draw, so a download waits for the draw.
    #[test]
    fn captures_taken_while_a_draw_is_prepared_do_not_hide_it() {
        let Some(mut f) = fixture() else { return };
        f.gmmu.map(GPU_VA, DADDR, PAGE_SIZE, 0, false).unwrap();
        let binding = f.acquire(DADDR, 0x100);
        f.read_once(binding, DADDR, 0x100);
        f.gpu_fill(binding, 0, 0x100, 0x2222_2222);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x100);
        let storage = f
            .cache
            .storage_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA,
                0x100,
                true,
            )
            .unwrap()
            .unwrap();
        f.submit();
        f.gpu_fill(storage, 0, 0x100, 0x5555_5555);
        let tick = f.scheduler.tick();
        f.cache.commit_writes(tick);
        f.cache.flush(&mut f.scheduler, DADDR, 0x100).unwrap();
        assert_eq!(f.cache.stats().captured_bytes, 0);
        assert!(
            f.scheduler.tick() > tick,
            "the draw was submitted with the download"
        );
        assert_eq!(f.guest(DADDR, 0x100), words(0x5555_5555, 0x100));
        f.assert_no_validation_errors();
    }

    /// A guest CPU write beside GPU-written bytes on one page, seen by write tracking for the
    /// whole page, keeps both.
    #[test]
    fn cpu_writes_beside_gpu_writes_keep_both() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        let original = pattern(0x3000, 4);
        space.write_bytes(CPU_VA + 0x2000, &original).unwrap();
        let binding = f.acquire(DADDR + 0x2000, 0x3000);
        f.gpu_fill(binding, 0x40, 0x100, 0xdead_beef);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x2040, 0x100);
        let cpu = [0x5a; 16];
        space.write_bytes(CPU_VA + 0x2400, &cpu).unwrap();
        let memory = f.memory.clone();
        let mut drained = Vec::new();
        memory.take_cpu_writes(|daddr, size| {
            drained.push((daddr, size));
            f.cache.cpu_written(daddr, size);
        });
        assert_eq!(drained, [(DADDR + 0x2000, PAGE_SIZE)]);
        f.acquire(DADDR + 0x2000, 0x3000);
        f.cache.flush(&mut f.scheduler, DADDR, 0x10000).unwrap();
        let mut expected = original;
        expected[0x40..0x140].copy_from_slice(&0xdead_beef_u32.to_le_bytes().repeat(0x40));
        expected[0x400..0x410].copy_from_slice(&cpu);
        assert_eq!(f.guest(DADDR + 0x2000, 0x3000), expected);
        f.assert_no_validation_errors();
    }

    /// Serves flush traps the way the GPU back does, on the test's thread: a guest CPU thread
    /// that meets one sends its range and waits for the reply.
    struct BackThread(parking_lot::Mutex<std::sync::mpsc::Sender<(DAddr, u64, SyncSender<()>)>>);

    type SyncSender<T> = std::sync::mpsc::SyncSender<T>;

    impl lichii_memory::FlushTrapHandler for BackThread {
        fn flush_for_cpu(&self, daddr: DAddr, size: u64) {
            let (done, wait) = std::sync::mpsc::sync_channel(1);
            self.0.lock().send((daddr, size, done)).unwrap();
            let _ = wait.recv();
        }
    }

    /// GPU work writes a buffer, and a guest CPU read of it gets the GPU's bytes: the read traps,
    /// the whole GPU-written range is downloaded once, and the next GPU write arms it again.
    #[test]
    fn cpu_reads_of_gpu_written_buffers_trap_until_downloaded() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        f.cache.flush_traps = true;
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        let (requests, served) = std::sync::mpsc::channel();
        f.memory
            .set_flush_trap_handler(Some(Arc::new(BackThread(parking_lot::Mutex::new(
                requests,
            )))));
        let read = |offset: u64| {
            let (done, value) = std::sync::mpsc::channel();
            let space = space.clone();
            std::thread::spawn(move || done.send(space.read::<u32>(CPU_VA + offset).unwrap()));
            value
        };
        let timeout = std::time::Duration::from_secs(5);
        let binding = f.acquire(DADDR, 0x3000);

        for value in [0xc0ff_ee01, 0xc0ff_ee02] {
            f.gpu_fill(binding, 0, 0x2000, value);
            f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x2000);
            let reader = read(0x1004);
            let (daddr, size, done) = served.recv_timeout(timeout).expect("the read trapped");
            assert_eq!((daddr, size), (DADDR + 0x1000, PAGE_SIZE));
            f.cache
                .flush_for_cpu_read(&mut f.scheduler, daddr, size)
                .unwrap();
            done.send(()).unwrap();
            assert_eq!(reader.recv_timeout(timeout).unwrap(), value);
            assert_eq!(
                read(0x10).recv_timeout(timeout).unwrap(),
                value,
                "the first page was downloaded with the second"
            );
            assert!(served.try_recv().is_err(), "nothing else trapped");
        }
        assert_eq!(
            read(0x2000).recv_timeout(timeout).unwrap(),
            0,
            "never GPU-written"
        );

        f.cache.flush_traps = false;
        f.gpu_fill(binding, 0, 0x2000, 0xc0ff_ee03);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x2000);
        assert_eq!(
            read(0x1004).recv_timeout(timeout).unwrap(),
            0xc0ff_ee02,
            "without flush traps the read sees guest memory"
        );
        assert!(served.try_recv().is_err());
        f.assert_no_validation_errors();
    }

    /// A guest CPU write that lands after a written binding's upload but before GPU work that
    /// writes part of the binding is recorded keeps its bytes, beside and inside the binding,
    /// and the GPU's bytes download around them.
    #[test]
    fn a_cpu_write_before_the_arm_survives_the_download_beside_gpu_writes() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        f.cache.flush_traps = true;
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        let original = pattern(0x2000, 4);
        space.write_bytes(CPU_VA, &original).unwrap();
        let binding = f.acquire(DADDR, 0x1000);
        let cpu = [0x5a; 16];
        space.write_bytes(CPU_VA + 0x800, &cpu).unwrap();
        f.cache
            .prepare_gpu_write(&mut f.scheduler, &mut f.staging, DADDR, 0x1000)
            .unwrap();
        f.gpu_fill(binding, 0, 0x100, 0xdead_beef);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x1000);
        f.cache.flush(&mut f.scheduler, DADDR, 0x1000).unwrap();
        let mut expected = original[..0x1000].to_vec();
        expected[..0x100].copy_from_slice(&0xdead_beef_u32.to_le_bytes().repeat(0x40));
        expected[0x800..0x810].copy_from_slice(&cpu);
        assert_eq!(f.guest(DADDR, 0x1000), expected);
        assert!(
            f.memory.has_cpu_writes(),
            "the write still reaches the next drain"
        );
        f.assert_no_validation_errors();
    }

    /// A download of part of a page's GPU-written bytes leaves the page trapped for the rest.
    #[test]
    fn a_partial_download_keeps_the_trap_of_a_page_still_gpu_written() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        f.cache.flush_traps = true;
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        let binding = f.acquire(DADDR, 0x2000);
        f.gpu_fill(binding, 0, 0x2000, 0x1234_5678);
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x2000);
        f.cache.flush(&mut f.scheduler, DADDR, 0x1800).unwrap();
        let before = f.memory.take_flush_trap_stats();
        assert_eq!((before.arms, before.disarms), (1, 1), "{before:?}");
        f.cache
            .flush(&mut f.scheduler, DADDR + 0x1800, 0x800)
            .unwrap();
        assert_eq!(
            f.memory.take_flush_trap_stats().disarms,
            1,
            "the second page last"
        );
        f.cache.mark_gpu_written(f.scheduler.tick(), DADDR, 0x2000);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x100, 0x100);
        assert_eq!(
            f.memory.take_flush_trap_stats().arms,
            1,
            "a range already GPU-written arms nothing"
        );
        f.cache.flush(&mut f.scheduler, DADDR, 0x2000).unwrap();
        let before = allocations();
        f.memory.arm_flush_traps(DADDR, 0x2000);
        f.memory.disarm_flush_traps(DADDR, 0x2000);
        assert_eq!(
            allocations() - before,
            0,
            "arming and disarming again reuse their scratch"
        );
        f.assert_no_validation_errors();
    }

    /// A page GPU work wrote whole uploads nothing after a CPU write, yet stays tracked: once
    /// flushed, the next CPU write to it reaches the buffer.
    #[test]
    fn fully_gpu_written_pages_stay_tracked() {
        const CPU_VA: u64 = 0x1000_0000;
        let Some(mut f) = fixture() else { return };
        let space = Arc::new(AddressSpace::new(f.memory.physical().clone(), 39));
        space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
        f.memory.attach_process(space.clone());
        let binding = f.acquire(DADDR, PAGE_SIZE);
        f.gpu_fill(binding, 0, PAGE_SIZE, 0x0102_0304);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR, PAGE_SIZE);
        let memory = f.memory.clone();
        for byte in [7, 9] {
            space.write_bytes(CPU_VA + 0x10, &[byte; 8]).unwrap();
            assert!(f.memory.has_cpu_writes(), "write of {byte} tracked");
            memory.take_cpu_writes(|daddr, size| f.cache.cpu_written(daddr, size));
            f.acquire(DADDR, PAGE_SIZE);
            f.cache.flush(&mut f.scheduler, DADDR, PAGE_SIZE).unwrap();
        }
        let mut guest = vec![0; PAGE_SIZE as usize];
        space.read_bytes(CPU_VA, &mut guest).unwrap();
        assert_eq!(&guest[0x10..0x18], &[9; 8]);
        assert!(
            f.read_binding(binding) == guest,
            "the buffer holds the CPU's bytes"
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn reuploads_only_dirty_pages() {
        let Some(mut f) = fixture() else { return };
        f.memory.write_bytes(DADDR, &pattern(0x10000, 2)).unwrap();
        let binding = f.acquire(DADDR, 0x10000);
        assert_eq!(f.cache.stats().uploaded_bytes, 0x10000);
        f.acquire(DADDR, 0x10000);
        assert_eq!(
            f.cache.stats().uploaded_bytes,
            0x10000,
            "a clean range uploads nothing"
        );

        f.memory.write_bytes(DADDR + 0x5010, &[1, 2, 3]).unwrap();
        f.cache.invalidate(DADDR + 0x5010, 3);
        f.memory.write_bytes(DADDR + 0x9ffc, &[9; 8]).unwrap();
        f.cache.invalidate(DADDR + 0x9ffc, 8);
        f.cache.invalidate(DADDR + 0x40_0000, 0x1000);
        f.acquire(DADDR + 0x8000, 0x1000);
        assert_eq!(
            f.cache.stats().uploaded_bytes,
            0x10000,
            "page 8 stayed clean"
        );
        f.acquire(DADDR, 0x10000);
        assert_eq!(
            f.cache.stats().uploaded_bytes,
            0x10000 + 3 * PAGE_SIZE,
            "pages 5, 9 and 10 upload again"
        );
        let expected = f.guest(DADDR, 0x10000);
        assert!(
            f.read_binding(binding) == expected,
            "re-uploaded bytes differ"
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn merges_buffers_and_keeps_their_contents() {
        let Some(mut f) = fixture() else { return };
        f.memory.write_bytes(DADDR, &pattern(0x40000, 3)).unwrap();
        let a = f.acquire(DADDR + 0x1000, 0x1000);
        let b = f.acquire(DADDR + 0x10100, 0x100);
        assert_ne!(
            a.buffer, b.buffer,
            "buffers adjacent at a caching page stay apart"
        );
        assert_eq!(f.acquire(DADDR + 0xff00, 0x100).buffer, a.buffer);
        f.gpu_fill(a, 0, 0x100, 0xaabb_ccdd);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x1000, 0x100);
        f.gpu_fill(b, 0, 0x100, 0x1122_3344);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x10100, 0x100);
        let uploaded = f.cache.stats().uploaded_bytes;

        let spanning = f.acquire(DADDR + 0xf800, 0x1000);
        assert_eq!(spanning.offset, 0xf800);
        let stats = f.cache.stats();
        assert_eq!((stats.created_buffers, stats.merged_buffers), (3, 2));
        assert_eq!(stats.uploaded_bytes, uploaded, "pages 15 and 16 were clean");
        let whole = f.acquire(DADDR, 0x20000);
        assert_eq!(whole.buffer, spanning.buffer);
        assert_eq!(
            f.cache.stats().uploaded_bytes,
            uploaded + 29 * PAGE_SIZE,
            "only the never-uploaded pages upload"
        );
        let mut expected = f.guest(DADDR, 0x20000);
        expected[0x1000..0x1100].copy_from_slice(&0xaabb_ccdd_u32.to_le_bytes().repeat(0x40));
        expected[0x10100..0x10200].copy_from_slice(&0x1122_3344_u32.to_le_bytes().repeat(0x40));
        assert!(
            f.read_binding(whole) == expected,
            "the merged buffer lost GPU-written or uploaded bytes"
        );

        f.cache.flush(&mut f.scheduler, DADDR, 0x20000).unwrap();
        assert_eq!(f.guest(DADDR, 0x20000), expected);
        f.assert_no_validation_errors();
    }

    #[test]
    fn gpu_writes_flush_to_guest_memory() {
        let Some(mut f) = fixture() else { return };
        let original = pattern(0x3000, 4);
        f.memory.write_bytes(DADDR + 0x2000, &original).unwrap();
        let binding = f.acquire(DADDR + 0x2000, 0x3000);
        f.gpu_fill(binding, 0x40, 0x100, 0xdead_beef);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x2040, 0x100);

        let cpu = [0x5a; 16];
        f.memory.write_bytes(DADDR + 0x2400, &cpu).unwrap();
        f.cache.invalidate(DADDR + 0x2400, 16);
        let uploaded = f.cache.stats().uploaded_bytes;
        f.acquire(DADDR + 0x2000, 0x3000);
        assert_eq!(
            f.cache.stats().uploaded_bytes - uploaded,
            PAGE_SIZE - 0x100,
            "the page re-uploads around the GPU-written bytes"
        );
        f.cache.flush(&mut f.scheduler, DADDR, 0x10000).unwrap();
        let mut expected = original.clone();
        expected[0x40..0x140].copy_from_slice(&0xdead_beef_u32.to_le_bytes().repeat(0x40));
        expected[0x400..0x410].copy_from_slice(&cpu);
        assert_eq!(f.guest(DADDR + 0x2000, 0x3000), expected);
        assert_eq!(f.cache.stats().downloaded_bytes, 0x100);
        f.cache.flush(&mut f.scheduler, DADDR, 0x10000).unwrap();
        assert_eq!(
            f.cache.stats().downloaded_bytes,
            0x100,
            "flushed bytes are clean"
        );

        f.gpu_fill(binding, 0x40, 0x100, 0x0102_0304);
        f.cache
            .mark_gpu_written(f.scheduler.tick(), DADDR + 0x2040, 0x100);
        let newer = pattern(0x100, 5);
        f.memory.write_bytes(DADDR + 0x2040, &newer).unwrap();
        f.cache.invalidate(DADDR + 0x2040, 0x100);
        f.cache.flush(&mut f.scheduler, DADDR, 0x10000).unwrap();
        assert_eq!(
            f.guest(DADDR + 0x2040, 0x100),
            newer,
            "a later CPU write wins"
        );
        assert_eq!(f.cache.stats().downloaded_bytes, 0x100);
        f.assert_no_validation_errors();
    }

    #[test]
    fn gathered_holes_read_zero_over_reused_stream_memory() {
        let Some(mut f) = fixture() else { return };
        let mut contents = Vec::new();
        for (page, daddr) in [(0, 0x40000), (1, 0x10000), (2, 0x80000)] {
            f.gmmu
                .map(
                    GPU_VA + page * PAGE_SIZE,
                    DADDR + daddr,
                    PAGE_SIZE,
                    0,
                    false,
                )
                .unwrap();
            let data = pattern(PAGE_SIZE as usize, page as u32 + 20);
            f.memory.write_bytes(DADDR + daddr, &data).unwrap();
            contents.push(data);
        }
        let holed = GPU_VA + 8 * PAGE_SIZE;
        for (page, daddr) in [(0, 0x40000), (2, 0x80000)] {
            f.gmmu
                .map(holed + page * PAGE_SIZE, DADDR + daddr, PAGE_SIZE, 0, false)
                .unwrap();
        }

        let dirty = f
            .cache
            .vertex_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA,
                3 * PAGE_SIZE,
            )
            .unwrap()
            .unwrap();
        assert_eq!(f.read_binding(dirty), contents.concat());

        let gathered = f
            .cache
            .vertex_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                holed + 0x802,
                2 * PAGE_SIZE,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            (gathered.buffer, gathered.offset),
            (dirty.buffer, dirty.offset),
            "the second gather reuses the stream memory the first one filled"
        );
        let mut expected = contents[0][0x802..].to_vec();
        expected.extend_from_slice(&[0; PAGE_SIZE as usize]);
        expected.extend_from_slice(&contents[2][..0x802]);
        assert!(
            f.read_binding(gathered) == expected,
            "the hole reads zero and the unaligned edges keep their device bytes"
        );

        let mixed = GPU_VA + 16 * PAGE_SIZE;
        f.gmmu.map_sparse(mixed, PAGE_SIZE, false).unwrap();
        f.gmmu
            .map(mixed + 2 * PAGE_SIZE, DADDR + 0x80000, PAGE_SIZE, 0, false)
            .unwrap();
        let gathered = f
            .cache
            .vertex_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                mixed + 2,
                2 * PAGE_SIZE,
            )
            .unwrap()
            .unwrap();
        assert_eq!(gathered.offset, dirty.offset);
        let mut expected = vec![0; 2 * PAGE_SIZE as usize - 2];
        expected.extend_from_slice(&contents[2][..2]);
        assert!(
            f.read_binding(gathered) == expected,
            "adjacent sparse and unmapped pages read zero as one hole"
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn binds_scattered_gpu_ranges() {
        let Some(mut f) = fixture() else { return };
        let pages = [(0, 0x40000), (1, 0x10000), (3, 0x80000)];
        let mut contents = Vec::new();
        for (page, daddr) in pages {
            f.gmmu
                .map(
                    GPU_VA + page * PAGE_SIZE,
                    DADDR + daddr,
                    PAGE_SIZE,
                    0,
                    false,
                )
                .unwrap();
            let data = pattern(PAGE_SIZE as usize, page as u32 + 10);
            f.memory.write_bytes(DADDR + daddr, &data).unwrap();
            contents.push(data);
        }
        assert!(!f.gmmu.is_continuous(GPU_VA, 0x2000));

        let vertex = f
            .cache
            .vertex_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA + 0x800,
                0x3000,
            )
            .unwrap()
            .unwrap();
        let mut expected = contents[0][0x800..].to_vec();
        expected.extend_from_slice(&contents[1]);
        expected.extend_from_slice(&[0; PAGE_SIZE as usize]);
        expected.extend_from_slice(&contents[2][..0x800]);
        assert!(f.read_binding(vertex) == expected, "gathered bytes differ");

        let inside = f
            .cache
            .vertex_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA + 0x1010,
                0x100,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            inside.offset, 0x10,
            "one device page binds its cached buffer"
        );
        assert_eq!(f.read_binding(inside), contents[1][0x10..0x110]);

        let uniform = f
            .cache
            .uniform_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA + 0xf00,
                0x200,
            )
            .unwrap()
            .unwrap();
        let mut expected = contents[0][0xf00..].to_vec();
        expected.extend_from_slice(&contents[1][..0x100]);
        assert_eq!(
            f.read_binding(uniform),
            expected,
            "streamed uniform bytes differ"
        );

        let written = f
            .cache
            .storage_buffer(
                &mut f.scheduler,
                &mut f.staging,
                &f.gmmu,
                GPU_VA + 0x800,
                0x1000,
                true,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            (written.offset, written.size),
            (0x800, 0x800),
            "first run only"
        );
        assert!(
            f.cache
                .gpu_written
                .intersects(DADDR + 0x40800, DADDR + 0x41000)
        );
        assert!(
            !f.cache
                .gpu_written
                .intersects(DADDR + 0x10000, DADDR + 0x11000)
        );
        f.assert_no_validation_errors();
    }

    #[test]
    fn converts_index_buffers() {
        let Some(mut f) = fixture() else { return };
        let mut regs = bytemuck::zeroed_box::<Regs>();

        let quads = f
            .index_buffer(&regs, &draw(PrimitiveTopology::QUADS, false, 4, 8))
            .unwrap();
        assert_eq!(
            (quads.index_type, quads.index_count, quads.vertex_offset),
            (vk::IndexType::UINT32, 12, 4)
        );
        assert_eq!(f.read_indices(quads), [0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7]);
        let strip = f
            .index_buffer(&regs, &draw(PrimitiveTopology::QUAD_STRIP, false, 0, 6))
            .unwrap();
        assert_eq!(f.read_indices(strip), [0, 3, 1, 0, 2, 3, 2, 5, 3, 2, 4, 5]);
        let grown = f
            .index_buffer(&regs, &draw(PrimitiveTopology::QUADS, false, 0, 4000))
            .unwrap();
        assert_ne!(grown.buffer, quads.buffer, "the quad indices grew");
        let indices = f.read_indices(grown);
        assert_eq!(
            (indices.len(), &indices[5994..]),
            (6000, &[3996, 3997, 3998, 3996, 3998, 3999][..])
        );
        let again = f
            .index_buffer(&regs, &draw(PrimitiveTopology::QUADS, false, 0, 40))
            .unwrap();
        assert_eq!(
            again.buffer, grown.buffer,
            "smaller draws reuse the quad indices"
        );
        let lines = f
            .index_buffer(&regs, &draw(PrimitiveTopology::LINE_LOOP, false, 2, 3))
            .unwrap();
        assert_eq!(
            (f.read_indices(lines), lines.vertex_offset),
            (vec![0, 1, 2, 0], 2)
        );
        assert!(
            f.index_buffer(&regs, &draw(PrimitiveTopology::TRIANGLES, false, 0, 3))
                .is_none()
        );

        f.gmmu.map(GPU_VA, DADDR, 0x10000, 0, false).unwrap();
        let mut set_indices = |f: &mut Fixture, format, bytes: &[u8]| {
            f.memory.write_bytes(DADDR + 0x3000, bytes).unwrap();
            f.cache.invalidate(DADDR + 0x3000, bytes.len() as u64);
            regs.index_buffer.start_addr_low = (GPU_VA + 0x3000) as u32;
            regs.index_buffer.limit_addr_low = (GPU_VA + 0x3000 + bytes.len() as u64 - 1) as u32;
            regs.index_buffer.format = format;
            regs.clone()
        };
        let u16s = |v: &[u16]| v.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<_>>();

        let r = set_indices(
            &mut f,
            IndexFormat::UNSIGNED_SHORT,
            &u16s(&[10, 11, 12, 13, 20, 21, 22, 23]),
        );
        let mut indexed = draw(PrimitiveTopology::QUADS, true, 0, 8);
        indexed.base_vertex = 5;
        let binding = f.index_buffer(&r, &indexed).unwrap();
        assert_eq!(
            (binding.vertex_offset, binding.index_type),
            (5, vk::IndexType::UINT32)
        );
        assert_eq!(
            f.read_indices(binding),
            [10, 11, 12, 10, 12, 13, 20, 21, 22, 20, 22, 23]
        );

        let triangles = f
            .index_buffer(&r, &draw(PrimitiveTopology::TRIANGLES, true, 2, 3))
            .unwrap();
        assert_eq!(
            (
                triangles.index_type,
                triangles.first_index,
                triangles.offset
            ),
            (vk::IndexType::UINT16, 2, 0x3000),
            "native indices bind the cached buffer"
        );
        assert_eq!(f.read_indices(triangles), [12, 13, 20]);

        let r = set_indices(
            &mut f,
            IndexFormat::UNSIGNED_INT,
            &[7u32, 8, 9].map(u32::to_le_bytes).concat(),
        );
        let lines = f
            .index_buffer(&r, &draw(PrimitiveTopology::LINE_LOOP, true, 0, 3))
            .unwrap();
        assert_eq!(f.read_indices(lines), [7, 8, 9, 7]);

        let native_u8 = !f.cache.widen_u8;
        f.cache.widen_u8 = true;
        let mut r = set_indices(&mut f, IndexFormat::UNSIGNED_BYTE, &[1, 2, 0xff, 3]);
        let widened = f
            .index_buffer(&r, &draw(PrimitiveTopology::TRIANGLES, true, 0, 4))
            .unwrap();
        assert_eq!(widened.index_type, vk::IndexType::UINT16);
        assert_eq!(f.read_indices(widened), [1, 2, 255, 3]);
        r.primitive_restart.enabled = 1;
        let restart = f
            .index_buffer(&r, &draw(PrimitiveTopology::TRIANGLES, true, 0, 4))
            .unwrap();
        assert_eq!(f.read_indices(restart), [1, 2, 0xffff, 3]);
        if native_u8 {
            f.cache.widen_u8 = false;
            let bytes = f
                .index_buffer(&r, &draw(PrimitiveTopology::TRIANGLES, true, 0, 4))
                .unwrap();
            assert_eq!(bytes.index_type, vk::IndexType::UINT8_EXT);
            assert_eq!(f.read_indices(bytes), [1, 2, 0xff, 3]);
        }

        let inline = [3u16, 4, 5, 6];
        let inline_draw = DrawParams {
            inline_indices: Some(InlineIndices::U16(&inline)),
            ..draw(PrimitiveTopology::QUADS, true, 0, 4)
        };
        let binding = f.index_buffer(&r, &inline_draw).unwrap();
        assert_eq!(f.read_indices(binding), [3, 4, 5, 3, 5, 6]);
        f.assert_no_validation_errors();
    }

    #[test]
    fn index_buffers_with_a_limit_below_their_start_draw_every_index() {
        let Some(mut f) = fixture() else { return };
        f.gmmu.map(GPU_VA, DADDR, 0x10000, 0, false).unwrap();
        let indices = [4u16, 5, 6, 7, 8, 9];
        let bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
        f.memory.write_bytes(DADDR + 0x3000, &bytes).unwrap();
        f.cache.invalidate(DADDR + 0x3000, bytes.len() as u64);
        let mut regs = bytemuck::zeroed_box::<Regs>();
        regs.index_buffer.start_addr_low = (GPU_VA + 0x3000) as u32;
        regs.index_buffer.format = IndexFormat::UNSIGNED_SHORT;
        for (limit, case) in [
            (0, "a limit never programmed"),
            (GPU_VA + 0x1000, "a stale limit"),
        ] {
            regs.index_buffer.limit_addr_low = limit as u32;
            let binding = f
                .index_buffer(&regs, &draw(PrimitiveTopology::TRIANGLES, true, 1, 3))
                .unwrap();
            assert_eq!((binding.first_index, binding.index_count), (1, 3), "{case}");
            assert_eq!(f.read_indices(binding), [5, 6, 7], "{case}");
        }
        f.assert_no_validation_errors();
    }

    #[test]
    fn unset_index_limits_bind_the_mapped_run_for_gpu_counted_draws() {
        let Some(f) = fixture() else { return };
        f.gmmu.map(GPU_VA, DADDR, 0x10000, 0, false).unwrap();
        let mut regs = bytemuck::zeroed_box::<Regs>();
        regs.index_buffer.start_addr_low = (GPU_VA + 0x3000) as u32;
        assert_eq!(index_capacity(&regs.index_buffer, 2), None);
        assert_eq!(
            bound_index_capacity(&f.gmmu, &regs.index_buffer, 2),
            (0x10000 - 0x3000) / 2,
            "the mapped run, not u32::MAX indices"
        );
        regs.index_buffer.limit_addr_low = (GPU_VA + 0x3000 + 11) as u32;
        assert_eq!(index_capacity(&regs.index_buffer, 2), Some(6));
        assert_eq!(bound_index_capacity(&f.gmmu, &regs.index_buffer, 2), 6);
    }

    #[test]
    fn clean_acquires_do_not_allocate() {
        let Some(mut f) = fixture() else { return };
        f.gmmu.map(GPU_VA, DADDR, 0x10000, 0, false).unwrap();
        f.acquire(DADDR, 0x10000);
        f.scheduler.finish().unwrap();
        let before = allocations();
        for i in 0..1000 {
            let binding = f.acquire(DADDR + i * 16, 0x1000);
            assert_eq!(binding.offset, i * 16);
            f.cache
                .vertex_buffer(&mut f.scheduler, &mut f.staging, &f.gmmu, GPU_VA, 0x8000)
                .unwrap()
                .unwrap();
        }
        assert_eq!(allocations() - before, 0, "clean acquires allocated");
        f.assert_no_validation_errors();
    }

    #[test]
    fn acquire_throughput() {
        let Some(mut f) = fixture() else { return };
        f.acquire(DADDR, 1 << 20);
        f.scheduler.finish().unwrap();
        let clean = 1_000_000;
        let start = Instant::now();
        for i in 0..clean {
            f.acquire(DADDR + (i % 240) * 0x1000, 0x10000);
        }
        let clean_ns = start.elapsed().as_nanos() as f64 / clean as f64;

        let dirty = 20_000;
        let start = Instant::now();
        for i in 0..dirty {
            let page = DADDR + (i % 256) * PAGE_SIZE;
            f.cache.invalidate(page, 4);
            f.acquire(page & !0xffff, 0x10000);
        }
        f.scheduler.finish().unwrap();
        let dirty_ns = start.elapsed().as_nanos() as f64 / dirty as f64;

        let full = 2_000;
        let start = Instant::now();
        for i in 0..full {
            let base = DADDR + (i % 16) * 0x10000;
            f.cache.invalidate(base, 0x10000);
            f.acquire(base, 0x10000);
        }
        f.scheduler.finish().unwrap();
        let full_s = start.elapsed().as_secs_f64();
        eprintln!(
            "clean acquire: {clean_ns:.1} ns; one dirty page of 64 KiB: {:.2} us; \
             fully dirty 64 KiB: {:.1} us ({:.0} MiB/s)",
            dirty_ns / 1000.0,
            full_s * 1e6 / full as f64,
            (full * 0x10000) as f64 / full_s / f64::from(1 << 20),
        );
        f.assert_no_validation_errors();
    }
}
