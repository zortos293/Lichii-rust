//! `/dev/nvhost-as-gpu`: a GPU address space, its VA allocators and the buffer mappings made in it.
//!
//! The address space is a [`Gmmu`] from [`Gpu::create_address_space`](lichii_gpu::Gpu); on
//! Switch 2 it also receives every nx2:gpu mapping. Small pages come from `[va_range_start,
//! va_range_split)`, big pages from the rest. Switch 2 NvRm passes no split (flags 5, no big page
//! size, `[0x8000000, 0xC000000000)`): small pages then cover its whole range and a small
//! big-page region follows it (NX2_NOTES row 25).

use std::collections::BTreeMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_common::{DAddr, GpuVAddr, PAGE_BITS, PAGE_SIZE, align_up};
use lichii_gpu::gmmu::{BigPageSize, Gmmu, MappedRange};
use parking_lot::Mutex;

use super::ioctl::{Args, Variant, items, read_struct, write_struct};
use super::syncpoints::MAX_SYNCPOINTS;
use super::{Device, NvResult, Nvdrv, file};

const SUPPORTED_BIG_PAGE_SIZES: u32 = 0x30000;
const DEFAULT_BIG_PAGE_SIZE: u32 = 0x20000;
const VA_START_SHIFT: u32 = 10;
const DEFAULT_VA_SPLIT: u64 = 1 << 34;
const DEFAULT_VA_RANGE: u64 = 1 << 37;
/// Big-page room Switch 2 address spaces get past the end NvRm asked for.
const NX2_BIG_PAGE_TAIL: u64 = 4 << 30;

const MAPPING_FIXED: u32 = 1 << 0;
const MAPPING_SPARSE: u32 = 1 << 1;
const MAPPING_REMAP: u32 = 1 << 8;

/// First-fit allocator of pages in `[start, limit)`.
struct VaAllocator {
    start: u64,
    limit: u64,
    free: BTreeMap<u64, u64>,
}

impl VaAllocator {
    fn new(start: u64, limit: u64) -> Self {
        let mut free = BTreeMap::new();
        if limit > start {
            free.insert(start, limit - start);
        }
        Self { start, limit, free }
    }

    fn allocate(&mut self, count: u64) -> Option<u64> {
        if count == 0 {
            return None;
        }
        let (&start, &len) = self.free.iter().find(|&(_, &len)| len >= count)?;
        self.free.remove(&start);
        if len > count {
            self.free.insert(start + count, len - count);
        }
        Some(start)
    }

    /// Takes `[start, start + count)` out of the free ranges, allocated parts included.
    fn allocate_fixed(&mut self, start: u64, count: u64) {
        let end = start.saturating_add(count);
        let overlapping: Vec<(u64, u64)> = self
            .free
            .range(..end)
            .filter(|&(&s, &len)| s + len > start)
            .map(|(&s, &len)| (s, len))
            .collect();
        for (s, len) in overlapping {
            self.free.remove(&s);
            if s < start {
                self.free.insert(s, start - s);
            }
            if s + len > end {
                self.free.insert(end, s + len - end);
            }
        }
    }

    fn free(&mut self, start: u64, count: u64) {
        let start = start.max(self.start);
        let end = start.saturating_add(count).min(self.limit);
        if end <= start {
            return;
        }
        self.allocate_fixed(start, end - start);
        let (mut first, mut last) = (start, end);
        if let Some((&s, &len)) = self.free.range(..start).next_back()
            && s + len == start
        {
            self.free.remove(&s);
            first = s;
        }
        if let Some(&len) = self.free.get(&end) {
            self.free.remove(&end);
            last = end + len;
        }
        self.free.insert(first, last - first);
    }
}

struct Vm {
    big_page_size: u64,
    big_page_bits: u32,
    va_range_end: u64,
    small: VaAllocator,
    big: VaAllocator,
}

impl Vm {
    fn page(&self, big: bool) -> (u64, u32) {
        if big {
            (self.big_page_size, self.big_page_bits)
        } else {
            (PAGE_SIZE, PAGE_BITS)
        }
    }

    fn allocator(&mut self, big: bool) -> &mut VaAllocator {
        if big { &mut self.big } else { &mut self.small }
    }
}

#[derive(Clone, Copy)]
struct Mapping {
    daddr: DAddr,
    size: u64,
    handle: u32,
    fixed: bool,
    big_page: bool,
    sparse_alloc: bool,
}

struct Allocation {
    mappings: Vec<u64>,
    size: u64,
    page_size: u32,
    sparse: bool,
    big_pages: bool,
}

#[derive(Default)]
struct AsState {
    vm: Option<Vm>,
    gmmu: Option<Arc<Gmmu>>,
    mappings: BTreeMap<u64, Mapping>,
    allocations: BTreeMap<u64, Allocation>,
    shim: Option<GpuVAddr>,
}

pub(crate) struct AsGpu {
    nvdrv: Arc<Nvdrv>,
    state: Mutex<AsState>,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AllocAsEx {
    flags: u32,
    as_fd: i32,
    big_page_size: u32,
    reserved: u32,
    va_range_start: u64,
    va_range_end: u64,
    va_range_split: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AllocSpace {
    pages: u32,
    page_size: u32,
    flags: u32,
    _pad: u32,
    offset: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FreeSpace {
    offset: u64,
    pages: u32,
    page_size: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct RemapEntry {
    flags: u16,
    kind: u16,
    handle: u32,
    handle_offset_big_pages: u32,
    as_offset_big_pages: u32,
    big_pages: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MapBufferEx {
    flags: u32,
    kind: u32,
    handle: u32,
    page_size: u32,
    buffer_offset: i64,
    mapping_size: u64,
    offset: i64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct VaRegion {
    offset: u64,
    page_size: u32,
    _pad: u32,
    pages: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GetVaRegions {
    buf_addr: u64,
    buf_size: u32,
    reserved: u32,
    regions: [VaRegion; 2],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SyncpointShimInfo {
    address: u64,
    stride: u32,
    count: u32,
}

impl AsGpu {
    pub fn new(nvdrv: Arc<Nvdrv>) -> Self {
        Self {
            nvdrv,
            state: Mutex::default(),
        }
    }

    pub fn ioctl(&self, files: &[Option<Device>], args: &mut Args) -> Option<NvResult> {
        Some(match (args.variant, args.number()) {
            (Variant::One, (b'A', 0x1)) => args.fixed(|fd: &mut i32| self.bind_channel(files, *fd)),
            (Variant::One, (b'A', 0x2)) => args.fixed(|p| self.alloc_space(p)),
            (Variant::One, (b'A', 0x3)) => args.fixed(|p| self.free_space(p)),
            (Variant::One, (b'A', 0x5)) => {
                args.fixed(|offset: &mut i64| self.unmap_buffer(*offset))
            }
            (Variant::One, (b'A', 0x6)) => args.fixed(|p| self.map_buffer_ex(p)),
            (Variant::One, (b'A', 0x8)) => args.fixed(|p| self.va_regions(p)),
            (Variant::Three, (b'A', 0x8)) => {
                let result = args.fixed(|p| self.va_regions(p));
                let regions: GetVaRegions = read_struct(args.output);
                write_struct(args.inline_output, &regions.regions);
                result
            }
            (Variant::One, (b'A', 0x9)) => args.fixed(|p| self.alloc_as_ex(p)),
            (Variant::One, (b'A', 0xb)) if self.nvdrv.nx2 => args.fixed(|p| self.syncpoint_shim(p)),
            (Variant::One, (b'A', 0x14)) => {
                args.echo();
                self.remap(args.input)
            }
            _ => return None,
        })
    }

    /// Creates the address space (AllocAsEx, also called InitializeEx).
    fn alloc_as_ex(&self, params: &mut AllocAsEx) -> NvResult {
        let mut state = self.state.lock();
        if state.vm.is_some() {
            tracing::error!("nvhost-as-gpu initialized twice");
            return NvResult::InvalidState;
        }
        let mut big_page_size = DEFAULT_BIG_PAGE_SIZE;
        let mut va_range_start = u64::from(DEFAULT_BIG_PAGE_SIZE) << VA_START_SHIFT;
        let mut va_range_split = DEFAULT_VA_SPLIT;
        let mut va_range_end = DEFAULT_VA_RANGE;
        let mut absolute_big_page_limit = false;
        if params.big_page_size != 0 {
            if !params.big_page_size.is_power_of_two()
                || params.big_page_size & SUPPORTED_BIG_PAGE_SIZES == 0
            {
                tracing::error!(size = params.big_page_size, "unsupported big page size");
                return NvResult::BadValue;
            }
            big_page_size = params.big_page_size;
            va_range_start = u64::from(big_page_size) << VA_START_SHIFT;
        }
        if params.va_range_start != 0 {
            va_range_start = params.va_range_start;
            va_range_split = params.va_range_split;
            va_range_end = params.va_range_end;
            if va_range_split == 0 {
                va_range_split = va_range_end;
                va_range_end += NX2_BIG_PAGE_TAIL;
                absolute_big_page_limit = true;
                tracing::warn!(
                    split = format_args!("{va_range_split:#x}"),
                    "no VA split given; small pages up to the end"
                );
            }
        }
        let big_page_bits = big_page_size.trailing_zeros();
        let big_limit = if absolute_big_page_limit {
            va_range_end
        } else {
            va_range_end.saturating_sub(va_range_split)
        };
        let big_page = if big_page_size == 0x10000 {
            BigPageSize::Size64K
        } else {
            BigPageSize::Size128K
        };
        state.vm = Some(Vm {
            big_page_size: u64::from(big_page_size),
            big_page_bits,
            va_range_end,
            small: VaAllocator::new(va_range_start >> PAGE_BITS, va_range_split >> PAGE_BITS),
            big: VaAllocator::new(va_range_split >> big_page_bits, big_limit >> big_page_bits),
        });
        state.gmmu = Some(self.nvdrv.gpu.create_address_space(big_page));
        tracing::info!(
            start = format_args!("{va_range_start:#x}"),
            split = format_args!("{va_range_split:#x}"),
            end = format_args!("{va_range_end:#x}"),
            "GPU address space created"
        );
        NvResult::Success
    }

    fn alloc_space(&self, params: &mut AllocSpace) -> NvResult {
        let mut state = self.state.lock();
        let AsState {
            vm: Some(vm),
            gmmu: Some(gmmu),
            allocations,
            ..
        } = &mut *state
        else {
            return NvResult::BadValue;
        };
        let big = u64::from(params.page_size) == vm.big_page_size;
        if params.page_size as u64 != PAGE_SIZE && !big {
            return NvResult::BadValue;
        }
        let sparse = params.flags & MAPPING_SPARSE != 0;
        if sparse && !big {
            tracing::error!("sparse small-page allocations are not supported");
            return NvResult::NotImplemented;
        }
        let (page_size, bits) = vm.page(big);
        let pages = u64::from(params.pages);
        if params.flags & MAPPING_FIXED != 0 {
            vm.allocator(big)
                .allocate_fixed(params.offset >> bits, pages);
        } else {
            let Some(page) = vm.allocator(big).allocate(pages) else {
                tracing::error!(pages, big, "GPU address space exhausted");
                return NvResult::InsufficientMemory;
            };
            params.offset = page << bits;
        }
        let size = pages * page_size;
        if sparse {
            let replaced = gmmu.map_sparse(params.offset, size, true);
            self.forget(replaced);
        }
        allocations.insert(
            params.offset,
            Allocation {
                mappings: Vec::new(),
                size,
                page_size: params.page_size,
                sparse,
                big_pages: big,
            },
        );
        NvResult::Success
    }

    fn free_space(&self, params: &mut FreeSpace) -> NvResult {
        let mut state = self.state.lock();
        let AsState {
            vm: Some(vm),
            gmmu: Some(gmmu),
            allocations,
            mappings,
            ..
        } = &mut *state
        else {
            return NvResult::BadValue;
        };
        let Some(allocation) = allocations.get(&params.offset) else {
            return NvResult::BadValue;
        };
        if allocation.page_size != params.page_size
            || allocation.size != u64::from(params.pages) * u64::from(params.page_size)
        {
            return NvResult::BadValue;
        }
        let allocation = allocations.remove(&params.offset).expect("looked up above");
        for offset in allocation.mappings {
            if let Some(mapping) = mappings.remove(&offset) {
                self.release_mapping(vm, gmmu, offset, mapping);
            }
        }
        if allocation.sparse {
            self.forget(gmmu.unmap(params.offset, allocation.size));
        }
        let (_, bits) = vm.page(allocation.big_pages);
        vm.allocator(allocation.big_pages)
            .free(params.offset >> bits, allocation.size >> bits);
        NvResult::Success
    }

    /// Frees a mapping's VA (unless it was fixed) and pin, returning a sparse range to sparse.
    fn release_mapping(&self, vm: &mut Vm, gmmu: &Gmmu, offset: u64, mapping: Mapping) {
        let (page_size, bits) = vm.page(mapping.big_page);
        if !mapping.fixed {
            vm.allocator(mapping.big_page)
                .free(offset >> bits, align_up(mapping.size, page_size) >> bits);
        }
        let size = align_up(mapping.size, page_size);
        let removed = if mapping.sparse_alloc {
            gmmu.map_sparse(offset, size, mapping.big_page)
        } else {
            gmmu.unmap(offset, size)
        };
        self.forget(removed);
        self.nvdrv.nvmap.unpin(mapping.handle);
    }

    fn map_buffer_ex(&self, params: &mut MapBufferEx) -> NvResult {
        let mut state = self.state.lock();
        let AsState {
            vm: Some(vm),
            gmmu: Some(gmmu),
            allocations,
            mappings,
            ..
        } = &mut *state
        else {
            return NvResult::BadValue;
        };
        let buffer_offset = params.buffer_offset as u64;
        if params.flags & MAPPING_REMAP != 0 {
            let offset = params.offset as u64;
            let Some(mapping) = mappings.get(&offset) else {
                tracing::warn!(offset, "remap of an unmapped GPU range");
                return NvResult::BadValue;
            };
            if mapping.size < params.mapping_size {
                tracing::warn!(offset, "remap of a partly mapped GPU range");
                return NvResult::BadValue;
            }
            return self.map(
                gmmu,
                offset + buffer_offset,
                mapping.daddr + buffer_offset,
                params.mapping_size,
                params.kind as u8,
                mapping.big_page.then_some(vm.big_page_size),
            );
        }

        let nvmap = &self.nvdrv.nvmap;
        let (Some(orig_size), Some(align)) =
            (nvmap.size(params.handle), nvmap.align(params.handle))
        else {
            return NvResult::BadValue;
        };
        let daddr = match nvmap.pin(params.handle) {
            Ok(daddr) => daddr + buffer_offset,
            Err(result) => return result,
        };
        let size = if params.mapping_size != 0 {
            params.mapping_size
        } else {
            orig_size
        };
        let big_page = align % vm.big_page_size == 0;
        let kind = params.kind as u8;

        if params.flags & MAPPING_FIXED != 0 {
            let offset = params.offset as u64;
            let Some((&base, allocation)) = allocations.range_mut(..=offset).next_back() else {
                nvmap.unpin(params.handle);
                return NvResult::BadValue;
            };
            if offset - base + size > allocation.size {
                tracing::error!(offset, "fixed mapping outside an allocated region");
                nvmap.unpin(params.handle);
                return NvResult::BadValue;
            }
            let use_big = allocation.big_pages && big_page;
            let result = self.map(
                gmmu,
                offset,
                daddr,
                size,
                kind,
                use_big.then_some(vm.big_page_size),
            );
            if result != NvResult::Success {
                nvmap.unpin(params.handle);
                return result;
            }
            allocation.mappings.push(offset);
            let sparse_alloc = allocation.sparse;
            mappings.insert(
                offset,
                Mapping {
                    daddr,
                    size,
                    handle: params.handle,
                    fixed: true,
                    big_page: use_big,
                    sparse_alloc,
                },
            );
            return NvResult::Success;
        }

        let (page_size, bits) = vm.page(big_page);
        let Some(page) = vm
            .allocator(big_page)
            .allocate(align_up(size, page_size) >> bits)
        else {
            tracing::error!(size, "GPU address space exhausted");
            nvmap.unpin(params.handle);
            return NvResult::InsufficientMemory;
        };
        let offset = page << bits;
        let result = self.map(
            gmmu,
            offset,
            daddr,
            size,
            kind,
            big_page.then_some(vm.big_page_size),
        );
        if result != NvResult::Success {
            vm.allocator(big_page)
                .free(page, align_up(size, page_size) >> bits);
            nvmap.unpin(params.handle);
            return result;
        }
        params.offset = offset as i64;
        mappings.insert(
            offset,
            Mapping {
                daddr,
                size,
                handle: params.handle,
                fixed: false,
                big_page,
                sparse_alloc: false,
            },
        );
        NvResult::Success
    }

    fn unmap_buffer(&self, offset: i64) -> NvResult {
        let mut state = self.state.lock();
        let offset = offset as u64;
        let AsState {
            vm: Some(vm),
            gmmu: Some(gmmu),
            mappings,
            ..
        } = &mut *state
        else {
            return NvResult::Success;
        };
        if let Some(mapping) = mappings.remove(&offset) {
            self.release_mapping(vm, gmmu, offset, mapping);
        }
        NvResult::Success
    }

    fn remap(&self, input: &[u8]) -> NvResult {
        let state = self.state.lock();
        let (Some(vm), Some(gmmu)) = (&state.vm, &state.gmmu) else {
            return NvResult::BadValue;
        };
        for entry in items::<RemapEntry>(input) {
            let va = u64::from(entry.as_offset_big_pages) << vm.big_page_bits;
            let size = u64::from(entry.big_pages) << vm.big_page_bits;
            let Some((&base, allocation)) = state.allocations.range(..=va).next_back() else {
                tracing::warn!(va, "remap into an unallocated region");
                return NvResult::BadValue;
            };
            if va - base + size > allocation.size {
                tracing::warn!(va, "remap into an unallocated region");
                return NvResult::BadValue;
            }
            if !allocation.sparse {
                tracing::warn!(va, "remap of a non-sparse allocation");
                return NvResult::BadValue;
            }
            let big = allocation.big_pages;
            if entry.handle == 0 {
                if let Err(error) = gmmu.map_sparse(va, size, big).map(|r| self.forget(Ok(r))) {
                    tracing::warn!(%error, "remap to sparse");
                    return NvResult::BadValue;
                }
                continue;
            }
            let base = match self.nvdrv.nvmap.pin(entry.handle) {
                Ok(daddr) => daddr,
                Err(result) => return result,
            };
            let daddr = base + (u64::from(entry.handle_offset_big_pages) << vm.big_page_bits);
            let big_page = big.then_some(vm.big_page_size);
            let result = self.map(gmmu, va, daddr, size, entry.kind as u8, big_page);
            if result != NvResult::Success {
                return result;
            }
        }
        NvResult::Success
    }

    fn bind_channel(&self, files: &[Option<Device>], fd: i32) -> NvResult {
        let Some(Device::Channel(channel)) = file(files, fd) else {
            tracing::error!(
                fd,
                "binding an address space to a descriptor that is not a channel"
            );
            return NvResult::BadValue;
        };
        match self.state.lock().gmmu.clone() {
            Some(gmmu) => channel.bind_address_space(gmmu),
            None => tracing::warn!(fd, "binding a channel to an uninitialized address space"),
        }
        NvResult::Success
    }

    fn va_regions(&self, params: &mut GetVaRegions) -> NvResult {
        let state = self.state.lock();
        let Some(vm) = &state.vm else {
            return NvResult::BadValue;
        };
        params.buf_size = size_of::<[VaRegion; 2]>() as u32;
        params.regions = [
            VaRegion {
                offset: vm.small.start << PAGE_BITS,
                page_size: PAGE_SIZE as u32,
                _pad: 0,
                pages: vm.small.limit - vm.small.start,
            },
            VaRegion {
                offset: vm.big.start << vm.big_page_bits,
                page_size: vm.big_page_size as u32,
                _pad: 0,
                pages: vm.big.limit - vm.big.start,
            },
        ];
        NvResult::Success
    }

    /// The read-only syncpoint page Switch 2 NVN reads syncpoint values through (NX2_NOTES row
    /// 46), on the last page of the address space.
    fn syncpoint_shim(&self, params: &mut SyncpointShimInfo) -> NvResult {
        let mut state = self.state.lock();
        let AsState {
            vm: Some(vm),
            gmmu: Some(gmmu),
            shim,
            ..
        } = &mut *state
        else {
            return NvResult::NotSupported;
        };
        let address = vm.va_range_end - PAGE_SIZE;
        if shim.is_none() {
            let syncpoints = self.nvdrv.syncpoints.syncpoints().clone();
            if let Err(error) = gmmu.map_syncpoint_shim(address, syncpoints) {
                tracing::error!(%error, "mapping the syncpoint shim");
                return NvResult::InsufficientMemory;
            }
            let bits = vm.big_page_bits;
            vm.big.allocate_fixed(address >> bits, 1);
            *shim = Some(address);
            tracing::info!(
                address = format_args!("{address:#x}"),
                "Switch 2 syncpoint shim"
            );
        }
        params.address = address;
        params.stride = size_of::<u32>() as u32;
        params.count = MAX_SYNCPOINTS;
        NvResult::Success
    }

    /// Maps `[gpu_va, +size)` (rounded up to pages) to `daddr`, with pages of `big_page` when
    /// `gpu_va` is aligned to them.
    fn map(
        &self,
        gmmu: &Gmmu,
        gpu_va: GpuVAddr,
        daddr: DAddr,
        size: u64,
        kind: u8,
        big_page: Option<u64>,
    ) -> NvResult {
        let page = big_page
            .filter(|&big| gpu_va.is_multiple_of(big))
            .unwrap_or(PAGE_SIZE);
        match gmmu.map(gpu_va, daddr, align_up(size, page), kind, page != PAGE_SIZE) {
            Ok(replaced) => {
                self.forget(Ok(replaced));
                NvResult::Success
            }
            Err(error) => {
                tracing::error!(%error, "GPU map");
                NvResult::BadValue
            }
        }
    }

    /// Tells the GPU the device ranges a GMMU change dropped.
    fn forget(&self, ranges: Result<Vec<MappedRange>, lichii_gpu::gmmu::GmmuError>) {
        match ranges {
            Ok(ranges) => {
                for range in ranges {
                    self.nvdrv.gpu.unmap_region(range.daddr, range.size);
                }
            }
            Err(error) => tracing::error!(%error, "GPU unmap"),
        }
    }
}

impl Drop for AsGpu {
    fn drop(&mut self) {
        if let Some(gmmu) = &self.state.get_mut().gmmu {
            self.nvdrv.gpu.destroy_address_space(gmmu);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn va_allocator_reuses_and_coalesces() {
        let mut va = VaAllocator::new(0x10, 0x100);
        let a = va.allocate(0x10).unwrap();
        let b = va.allocate(0x10).unwrap();
        assert_eq!((a, b), (0x10, 0x20));
        va.free(a, 0x10);
        assert_eq!(va.allocate(0x8), Some(0x10));
        va.allocate_fixed(0x40, 0x20);
        assert_eq!(va.allocate(0x18), Some(0x60));
        va.free(0x10, 0x8);
        va.free(0x20, 0x10);
        va.free(0x40, 0x20);
        va.free(0x60, 0x18);
        assert_eq!(va.free.len(), 1);
        assert_eq!(va.allocate(0xf0), Some(0x10));
        assert_eq!(va.allocate(1), None);
    }
}
