//! The Switch 2 GPU memory model behind nx2:gpu (nn::gpu::sdk), as an API for the services.
//!
//! Switch 2 titles never map through nvhost-as-gpu. Memory made GPU accessible (gpu sub 101)
//! is used at GPU VA == CPU VA, NormalMapGpuMemory4K (211) maps memory at a chosen GPU VA, also
//! into sparse regions (214), and every GMMU of the process sees every mapping, including those
//! made before it registered (NX2_NOTES rows 30 and 42). Each memory gets one device range that
//! every GMMU shares, freed for reuse when it is destroyed (rows 42 and 85). Destroying memory
//! unmaps every GPU range that reaches it from every GMMU, because the title reuses it as heap
//! right away (row 68). Sparse regions commit no memory until something is mapped into them
//! (rows 57, 80 and 81).
//!
//! Ranges named by first and last address are inclusive, as the requests carry them.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroU32;
use std::sync::Arc;

use lichii_common::{DAddr, GpuVAddr, PAGE_MASK, VAddr};
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use crate::device_memory::{DeviceMemory, DeviceMemoryError};
use crate::gmmu::{GPU_VA_SIZE, Gmmu, MappedRange, PTE_KIND_INVALID, PTE_KIND_PITCH};

const DEVICE_ALIGN: u64 = 0x2_0000;

/// A GpuMemoryHandle as nn::gpu::sdk stores it (reply word 3 of gpu sub 101).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuMemoryHandle(NonZeroU32);

impl GpuMemoryHandle {
    /// The handle a request names; `None` for 0, which is never a handle.
    pub fn new(raw: u32) -> Option<Self> {
        NonZeroU32::new(raw).map(Self)
    }

    pub fn raw(self) -> u32 {
        self.0.get()
    }
}

/// A GMMU registered with [`Nx2GpuMemory::register_address_space`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressSpaceId(pub u32);

/// A mapping removed or replaced in one address space; the GPU passes it on to the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unmapped {
    pub space: AddressSpaceId,
    pub range: MappedRange,
}

/// Where GPU accessible memory lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuMemoryInfo {
    pub cpu_va: VAddr,
    pub size: u64,
    pub daddr: DAddr,
}

/// Why a request failed; the services turn these into Horizon results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Nx2MemoryError {
    #[error("unknown GPU memory handle {0}")]
    InvalidHandle(u32),
    #[error("range {va:#x}+{size:#x} is empty, unaligned or outside the GPU address space")]
    InvalidRange { va: u64, size: u64 },
    #[error("offset {offset:#x}+{size:#x} is outside GPU memory {handle}")]
    OffsetOutOfRange { handle: u32, offset: u64, size: u64 },
    #[error("device address space exhausted ({size:#x} bytes requested)")]
    OutOfDeviceMemory { size: u64 },
    #[error("no GPU memory handles left")]
    OutOfHandles,
    #[error("GPU range {va:#x}+{size:#x} is already in use")]
    RangeInUse { va: GpuVAddr, size: u64 },
    #[error("no sparse region at {va:#x}+{size:#x}")]
    NoSuchRegion { va: GpuVAddr, size: u64 },
    #[error("GPU range {va:#x}+{size:#x} is not mapped")]
    NotMapped { va: GpuVAddr, size: u64 },
    #[error(transparent)]
    Device(#[from] DeviceMemoryError),
}

/// The Switch 2 GPU memory registry of one process.
pub struct Nx2GpuMemory {
    device: Arc<DeviceMemory>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    last_handle: u32,
    last_space: u32,
    memories: HashMap<u32, GpuMemoryInfo>,
    sparse_regions: BTreeMap<GpuVAddr, u64>,
    ranges: BTreeMap<GpuVAddr, Range>,
    spaces: Vec<(AddressSpaceId, Arc<Gmmu>)>,
}

#[derive(Clone, Copy, Debug)]
struct Range {
    size: u64,
    backing: Backing,
    kind: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backing {
    Sparse,
    Memory { handle: u32, offset: u64 },
}

impl Backing {
    fn advanced(self, by: u64) -> Self {
        match self {
            Self::Sparse => Self::Sparse,
            Self::Memory { handle, offset } => Self::Memory {
                handle,
                offset: offset + by,
            },
        }
    }
}

impl Nx2GpuMemory {
    pub fn new(device: Arc<DeviceMemory>) -> Self {
        Self {
            device,
            state: Mutex::new(State::default()),
        }
    }

    /// Adds a GMMU that every mapping goes into, starting with the ones already made.
    pub fn register_address_space(&self, gmmu: Arc<Gmmu>) -> AddressSpaceId {
        let mut state = self.state.lock();
        state.last_space += 1;
        let id = AddressSpaceId(state.last_space);
        for (&start, range) in &state.ranges {
            apply(&state.memories, &gmmu, start, range);
        }
        state.spaces.push((id, gmmu));
        id
    }

    /// Stops sending mappings to a GMMU; it keeps what it has.
    pub fn unregister_address_space(&self, id: AddressSpaceId) -> Option<Arc<Gmmu>> {
        let mut state = self.state.lock();
        let index = state.spaces.iter().position(|(space, _)| *space == id)?;
        Some(state.spaces.remove(index).1)
    }

    /// CreateGpuAccessibleMemory (gpu sub 101): gives process memory [`cpu_va`, +`size`) a device
    /// range and maps it at GPU VA == CPU VA in every address space. Returns the handle and the
    /// mappings the identity mapping replaced.
    pub fn create_gpu_accessible_memory(
        &self,
        space: &AddressSpace,
        cpu_va: VAddr,
        size: u64,
    ) -> Result<(GpuMemoryHandle, Vec<Unmapped>), Nx2MemoryError> {
        check_range(cpu_va, size)?;
        let mut state = self.state.lock();
        let handle = state
            .last_handle
            .checked_add(1)
            .ok_or(Nx2MemoryError::OutOfHandles)?;
        let daddr = self
            .device
            .allocate(size, DEVICE_ALIGN)
            .ok_or(Nx2MemoryError::OutOfDeviceMemory { size })?;
        if let Err(e) = self.device.map(daddr, space, cpu_va, size) {
            self.device
                .free(daddr, size)
                .expect("the range was just allocated");
            return Err(e.into());
        }
        state.last_handle = handle;
        state.memories.insert(
            handle,
            GpuMemoryInfo {
                cpu_va,
                size,
                daddr,
            },
        );
        state.take(cpu_va, size);
        let replaced = state.insert(
            cpu_va,
            Range {
                size,
                backing: Backing::Memory { handle, offset: 0 },
                kind: PTE_KIND_PITCH,
            },
        );
        Ok((
            GpuMemoryHandle::new(handle).expect("handles start at 1"),
            replaced,
        ))
    }

    /// Destroys GPU memory (gpu sub 102): unmaps every GPU range that reaches it, wherever it is
    /// mapped and in every address space (parts of sparse regions read zero again), then frees
    /// its device range. Returns the mappings removed.
    pub fn destroy(&self, handle: GpuMemoryHandle) -> Result<Vec<Unmapped>, Nx2MemoryError> {
        let raw = handle.raw();
        let mut state = self.state.lock();
        let memory = state
            .memories
            .remove(&raw)
            .ok_or(Nx2MemoryError::InvalidHandle(raw))?;
        let reaching: Vec<(GpuVAddr, u64)> = state
            .ranges
            .iter()
            .filter(|(_, r)| matches!(r.backing, Backing::Memory { handle, .. } if handle == raw))
            .map(|(&start, r)| (start, r.size))
            .collect();
        let mut unmapped = Vec::new();
        for (start, size) in reaching {
            unmapped.extend(state.unmap(start, size));
        }
        self.device.unmap(memory.daddr, memory.size)?;
        self.device.free(memory.daddr, memory.size)?;
        Ok(unmapped)
    }

    /// NormalMapGpuMemory4K (gpu sub 211): maps [`first_va`, `last_va`] to the memory of `handle`
    /// starting `offset` bytes in, in every address space. When `first_va` is the memory's own
    /// address plus `offset` this restores the identity mapping. Returns the mappings replaced.
    pub fn normal_map_4k(
        &self,
        first_va: GpuVAddr,
        last_va: GpuVAddr,
        handle: GpuMemoryHandle,
        offset: u64,
    ) -> Result<Vec<Unmapped>, Nx2MemoryError> {
        let size = inclusive_size(first_va, last_va)?;
        let raw = handle.raw();
        let mut state = self.state.lock();
        let memory = state
            .memories
            .get(&raw)
            .ok_or(Nx2MemoryError::InvalidHandle(raw))?;
        if offset & PAGE_MASK != 0 || offset.checked_add(size).is_none_or(|end| end > memory.size) {
            return Err(Nx2MemoryError::OffsetOutOfRange {
                handle: raw,
                offset,
                size,
            });
        }
        state.take(first_va, size);
        Ok(state.insert(
            first_va,
            Range {
                size,
                backing: Backing::Memory {
                    handle: raw,
                    offset,
                },
                kind: PTE_KIND_PITCH,
            },
        ))
    }

    /// NormalUnmapGpuMemory4K (gpu sub 213): unmaps [`first_va`, `last_va`] from every address
    /// space; parts inside a sparse region become sparse again. Returns the mappings removed.
    pub fn unmap_4k(
        &self,
        first_va: GpuVAddr,
        last_va: GpuVAddr,
    ) -> Result<Vec<Unmapped>, Nx2MemoryError> {
        let size = inclusive_size(first_va, last_va)?;
        Ok(self.state.lock().unmap(first_va, size))
    }

    /// CreateSparseRegion (gpu sub 214): reserves [`va`, +`size`) in every address space. It reads
    /// zero and commits no memory until memory is mapped into it, however large it is.
    pub fn create_sparse_region(
        &self,
        va: GpuVAddr,
        size: u64,
    ) -> Result<Vec<Unmapped>, Nx2MemoryError> {
        check_range(va, size)?;
        let mut state = self.state.lock();
        let end = va + size;
        let in_use = |last: Option<(&GpuVAddr, u64)>| last.is_some_and(|(&s, len)| s + len > va);
        if in_use(
            state
                .sparse_regions
                .range(..end)
                .next_back()
                .map(|(s, &l)| (s, l)),
        ) || in_use(
            state
                .ranges
                .range(..end)
                .next_back()
                .map(|(s, r)| (s, r.size)),
        ) {
            return Err(Nx2MemoryError::RangeInUse { va, size });
        }
        state.sparse_regions.insert(va, size);
        Ok(state.insert(
            va,
            Range {
                size,
                backing: Backing::Sparse,
                kind: PTE_KIND_INVALID,
            },
        ))
    }

    /// DestroySparseRegion (gpu sub 215): unmaps the region made at [`va`, +`size`) and everything
    /// mapped into it from every address space. Returns the mappings removed.
    pub fn destroy_sparse_region(
        &self,
        va: GpuVAddr,
        size: u64,
    ) -> Result<Vec<Unmapped>, Nx2MemoryError> {
        let mut state = self.state.lock();
        if state.sparse_regions.get(&va) != Some(&size) {
            return Err(Nx2MemoryError::NoSuchRegion { va, size });
        }
        state.sparse_regions.remove(&va);
        let mut unmapped = Vec::new();
        for (start, range) in state.take(va, size) {
            unmapped.extend(state.unmap_everywhere(start, range.size));
        }
        Ok(unmapped)
    }

    /// NormalModifyMemoryKind4K (gpu sub 212, behind NvRmGpuMappingModify): sets the PTE kind of
    /// [`first_va`, `last_va`], which must be mapped or sparse throughout, in every address space.
    /// Returns the kind applied (NX2_NOTES row 149).
    pub fn modify_kind(
        &self,
        first_va: GpuVAddr,
        last_va: GpuVAddr,
        kind: u8,
    ) -> Result<u8, Nx2MemoryError> {
        let size = inclusive_size(first_va, last_va)?;
        let end = first_va + size;
        let mut state = self.state.lock();
        let mut at = first_va;
        for (&start, range) in state.ranges.range(..end) {
            if start + range.size <= at {
                continue;
            }
            if start > at {
                break;
            }
            at = start + range.size;
        }
        if at < end {
            return Err(Nx2MemoryError::NotMapped { va: first_va, size });
        }
        state.split_at(first_va);
        state.split_at(end);
        for range in state.ranges.range_mut(first_va..end).map(|(_, r)| r) {
            range.kind = kind;
        }
        for (_, gmmu) in &state.spaces {
            gmmu.set_kind(first_va, size, kind)
                .expect("registry ranges are validated");
        }
        Ok(kind)
    }

    pub fn memory_info(&self, handle: GpuMemoryHandle) -> Option<GpuMemoryInfo> {
        self.state.lock().memories.get(&handle.raw()).copied()
    }
}

impl State {
    fn split_at(&mut self, at: GpuVAddr) {
        let Some((&start, &range)) = self.ranges.range(..at).next_back() else {
            return;
        };
        if start + range.size <= at {
            return;
        }
        let head = at - start;
        self.ranges.insert(
            start,
            Range {
                size: head,
                ..range
            },
        );
        self.ranges.insert(
            at,
            Range {
                size: range.size - head,
                backing: range.backing.advanced(head),
                kind: range.kind,
            },
        );
    }

    /// Removes the recorded ranges inside [`start`, +`size`), splitting ones that cross its ends,
    /// without touching the GMMUs.
    fn take(&mut self, start: GpuVAddr, size: u64) -> Vec<(GpuVAddr, Range)> {
        let end = start + size;
        self.split_at(start);
        self.split_at(end);
        let starts: Vec<GpuVAddr> = self.ranges.range(start..end).map(|(&s, _)| s).collect();
        starts
            .into_iter()
            .map(|s| (s, self.ranges.remove(&s).expect("just listed")))
            .collect()
    }

    fn insert(&mut self, start: GpuVAddr, range: Range) -> Vec<Unmapped> {
        let replaced = self
            .spaces
            .iter()
            .flat_map(|(space, gmmu)| {
                apply(&self.memories, gmmu, start, &range)
                    .into_iter()
                    .map(|range| Unmapped {
                        space: *space,
                        range,
                    })
            })
            .collect();
        self.ranges.insert(start, range);
        replaced
    }

    fn unmap_everywhere(&self, start: GpuVAddr, size: u64) -> Vec<Unmapped> {
        self.spaces
            .iter()
            .flat_map(|(space, gmmu)| {
                gmmu.unmap(start, size)
                    .expect("registry ranges are validated")
                    .into_iter()
                    .map(|range| Unmapped {
                        space: *space,
                        range,
                    })
            })
            .collect()
    }

    /// Unmaps the recorded ranges inside [`start`, +`size`); what lies in a sparse region turns
    /// sparse again.
    fn unmap(&mut self, start: GpuVAddr, size: u64) -> Vec<Unmapped> {
        let mut unmapped = Vec::new();
        for (piece_start, range) in self.take(start, size) {
            let kind = match range.backing {
                Backing::Sparse => range.kind,
                Backing::Memory { .. } => PTE_KIND_INVALID,
            };
            for (va, len, sparse) in self.sparse_pieces(piece_start, range.size) {
                if sparse {
                    unmapped.extend(self.insert(
                        va,
                        Range {
                            size: len,
                            backing: Backing::Sparse,
                            kind,
                        },
                    ));
                } else {
                    unmapped.extend(self.unmap_everywhere(va, len));
                }
            }
        }
        unmapped
    }

    /// [`start`, +`size`) cut where sparse regions begin and end, each piece marked with whether
    /// it lies in one.
    fn sparse_pieces(&self, start: GpuVAddr, size: u64) -> Vec<(GpuVAddr, u64, bool)> {
        let end = start + size;
        let mut pieces = Vec::new();
        let mut at = start;
        for (&region, &len) in self.sparse_regions.range(..end) {
            let region_end = region + len;
            if region_end <= at {
                continue;
            }
            let from = region.max(at);
            if from > at {
                pieces.push((at, from - at, false));
            }
            let to = region_end.min(end);
            pieces.push((from, to - from, true));
            at = to;
        }
        if at < end {
            pieces.push((at, end - at, false));
        }
        pieces
    }
}

fn apply(
    memories: &HashMap<u32, GpuMemoryInfo>,
    gmmu: &Gmmu,
    start: GpuVAddr,
    range: &Range,
) -> Vec<MappedRange> {
    let result = match range.backing {
        Backing::Sparse => gmmu
            .map_sparse(start, range.size, false)
            .and_then(|replaced| {
                if range.kind != PTE_KIND_INVALID {
                    gmmu.set_kind(start, range.size, range.kind)?;
                }
                Ok(replaced)
            }),
        Backing::Memory { handle, offset } => gmmu.map(
            start,
            memories[&handle].daddr + offset,
            range.size,
            range.kind,
            false,
        ),
    };
    result.expect("registry ranges are validated")
}

fn check_range(va: u64, size: u64) -> Result<(), Nx2MemoryError> {
    let fits = va.checked_add(size).is_some_and(|end| end <= GPU_VA_SIZE);
    if (va | size) & PAGE_MASK != 0 || size == 0 || !fits {
        return Err(Nx2MemoryError::InvalidRange { va, size });
    }
    Ok(())
}

fn inclusive_size(first: u64, last: u64) -> Result<u64, Nx2MemoryError> {
    let size = last
        .checked_sub(first)
        .and_then(|span| span.checked_add(1))
        .ok_or(Nx2MemoryError::InvalidRange { va: first, size: 0 })?;
    check_range(first, size)?;
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gmmu::{BigPageSize, RangeTarget};
    use lichii_common::PAGE_SIZE;
    use lichii_memory::{MemoryPermission, PhysicalMemory};

    const CPU_VA: u64 = 0x8000_0000;
    const SPARSE_VA: u64 = 0x81_0000_0000;

    struct Fixture {
        space: AddressSpace,
        device: Arc<DeviceMemory>,
        memory: Nx2GpuMemory,
        gmmus: [Arc<Gmmu>; 2],
    }

    fn fixture() -> Fixture {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let space = AddressSpace::new(physical.clone(), 39);
        space.map(CPU_VA, 0, 0x20_0000, MemoryPermission::READ_WRITE);
        let device = Arc::new(DeviceMemory::new(physical));
        let gmmus = [
            Arc::new(Gmmu::new(device.clone(), BigPageSize::Size64K)),
            Arc::new(Gmmu::new(device.clone(), BigPageSize::Size128K)),
        ];
        Fixture {
            space,
            memory: Nx2GpuMemory::new(device.clone()),
            device,
            gmmus,
        }
    }

    fn target(gmmu: &Gmmu, va: GpuVAddr) -> RangeTarget {
        let mut target = RangeTarget::Unmapped;
        gmmu.for_each_range(va, 1, |r| target = r.target);
        target
    }

    #[test]
    fn identity_maps_into_every_gmmu() {
        let f = fixture();
        let first = f.memory.register_address_space(f.gmmus[0].clone());
        let (handle, replaced) = f
            .memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x10_0000)
            .unwrap();
        assert!(replaced.is_empty());
        let second = f.memory.register_address_space(f.gmmus[1].clone());
        assert_ne!(first, second);
        f.space.write(CPU_VA + 0x1230, 0xdead_beefu32).unwrap();
        for gmmu in &f.gmmus {
            assert_eq!(gmmu.read::<u32>(CPU_VA + 0x1230), Ok(0xdead_beef));
        }
        let info = f.memory.memory_info(handle).unwrap();
        assert_eq!(
            f.gmmus[0].translate(CPU_VA + 0x5000),
            Some(info.daddr + 0x5000)
        );
        assert_eq!(
            f.gmmus[1].translate(CPU_VA + 0x5000),
            Some(info.daddr + 0x5000)
        );
        f.gmmus[1].write(CPU_VA + 0xf_fffc, 7u32).unwrap();
        assert_eq!(f.space.read::<u32>(CPU_VA + 0xf_fffc), Ok(7));
        assert_eq!(f.gmmus[0].translate(CPU_VA + 0x10_0000), None);
    }

    #[test]
    fn maps_into_sparse_regions_and_unmaps_back_to_sparse() {
        let f = fixture();
        for gmmu in &f.gmmus {
            f.memory.register_address_space(gmmu.clone());
        }
        let (handle, _) = f
            .memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x10_0000)
            .unwrap();
        f.memory
            .create_sparse_region(SPARSE_VA, 0x100_0000)
            .unwrap();
        let va = SPARSE_VA + 0x1_0000;
        assert!(
            f.memory
                .normal_map_4k(va, va + 0x2fff, handle, 0x2000)
                .unwrap()
                .is_empty()
        );
        f.space.write(CPU_VA + 0x2004, 0x1234u32).unwrap();
        for gmmu in &f.gmmus {
            assert_eq!(gmmu.read::<u32>(va + 4), Ok(0x1234));
            assert_eq!(target(gmmu, va - PAGE_SIZE), RangeTarget::Sparse);
        }
        let removed = f.memory.unmap_4k(va + 0x1000, va + 0x1fff).unwrap();
        assert_eq!(removed.len(), 2);
        assert!(
            removed
                .iter()
                .all(|u| u.range.gpu_va == va + 0x1000 && u.range.size == 0x1000)
        );
        for gmmu in &f.gmmus {
            assert_eq!(target(gmmu, va + 0x1000), RangeTarget::Sparse);
            assert_eq!(gmmu.read::<u32>(va + 0x1000), Ok(0));
            assert!(matches!(target(gmmu, va + 0x2000), RangeTarget::Device(_)));
        }
        let info = f.memory.memory_info(handle).unwrap();
        f.memory.unmap_4k(CPU_VA, CPU_VA + 0xfff).unwrap();
        for gmmu in &f.gmmus {
            assert_eq!(target(gmmu, CPU_VA), RangeTarget::Unmapped);
            assert_eq!(gmmu.translate(CPU_VA + 0x1000), Some(info.daddr + 0x1000));
        }
    }

    #[test]
    fn destroy_unmaps_every_range_from_every_gmmu() {
        let f = fixture();
        for gmmu in &f.gmmus {
            f.memory.register_address_space(gmmu.clone());
        }
        let (handle, _) = f
            .memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x10_0000)
            .unwrap();
        let info = f.memory.memory_info(handle).unwrap();
        f.memory.create_sparse_region(SPARSE_VA, 0x10_0000).unwrap();
        f.memory
            .normal_map_4k(SPARSE_VA, SPARSE_VA + 0x3fff, handle, 0x8000)
            .unwrap();
        let removed = f.memory.destroy(handle).unwrap();
        let mut expected = Vec::new();
        for space in [AddressSpaceId(1), AddressSpaceId(2)] {
            expected.push(Unmapped {
                space,
                range: MappedRange {
                    gpu_va: CPU_VA,
                    daddr: info.daddr,
                    size: 0x10_0000,
                },
            });
            expected.push(Unmapped {
                space,
                range: MappedRange {
                    gpu_va: SPARSE_VA,
                    daddr: info.daddr + 0x8000,
                    size: 0x4000,
                },
            });
        }
        let key = |u: &Unmapped| (u.space.0, u.range.gpu_va);
        let mut removed = removed;
        removed.sort_by_key(key);
        expected.sort_by_key(key);
        assert_eq!(removed, expected);
        for gmmu in &f.gmmus {
            assert_eq!(target(gmmu, CPU_VA + 0x4000), RangeTarget::Unmapped);
            assert_eq!(target(gmmu, SPARSE_VA), RangeTarget::Sparse);
        }
        assert_eq!(f.device.allocated_bytes(), 0);
        assert!(f.device.translate(info.daddr).is_none());
        assert_eq!(
            f.memory.destroy(handle),
            Err(Nx2MemoryError::InvalidHandle(1))
        );
        let (again, _) = f
            .memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x10_0000)
            .unwrap();
        assert_ne!(again, handle);
        assert_eq!(f.memory.memory_info(again).unwrap().daddr, info.daddr);
    }

    #[test]
    fn huge_sparse_regions_commit_no_memory() {
        let f = fixture();
        for gmmu in &f.gmmus {
            f.memory.register_address_space(gmmu.clone());
        }
        let size = 0x5_0000_0000;
        f.memory.create_sparse_region(SPARSE_VA, size).unwrap();
        f.memory
            .create_sparse_region(SPARSE_VA + size, size)
            .unwrap();
        assert_eq!(f.device.allocated_bytes(), 0);
        for gmmu in &f.gmmus {
            assert!(gmmu.table_bytes() < 16 << 20, "{}", gmmu.table_bytes());
            assert_eq!(gmmu.read::<u64>(SPARSE_VA + 2 * size - 8), Ok(0));
        }
        f.memory.destroy_sparse_region(SPARSE_VA, size).unwrap();
        for gmmu in &f.gmmus {
            assert_eq!(target(gmmu, SPARSE_VA), RangeTarget::Unmapped);
            assert_eq!(target(gmmu, SPARSE_VA + size), RangeTarget::Sparse);
        }
    }

    #[test]
    fn modify_kind_reaches_every_gmmu() {
        let f = fixture();
        f.memory.register_address_space(f.gmmus[0].clone());
        f.memory.create_sparse_region(SPARSE_VA, 0x40_0000).unwrap();
        let first = SPARSE_VA + 0x1_0000;
        assert_eq!(f.memory.modify_kind(first, first + 0xffff, 0xdb), Ok(0xdb));
        f.memory.register_address_space(f.gmmus[1].clone());
        for gmmu in &f.gmmus {
            assert_eq!(gmmu.kind(first), Some(0xdb));
            assert_eq!(gmmu.kind(first + 0x1_0000), Some(PTE_KIND_INVALID));
        }
        assert_eq!(
            f.memory
                .modify_kind(SPARSE_VA + 0x3f_f000, SPARSE_VA + 0x40_0fff, 1),
            Err(Nx2MemoryError::NotMapped {
                va: SPARSE_VA + 0x3f_f000,
                size: 0x2000
            })
        );
    }

    #[test]
    fn rejects_bad_requests() {
        let f = fixture();
        let missing = GpuMemoryHandle::new(9).unwrap();
        assert_eq!(
            f.memory.normal_map_4k(0x1000, 0x1fff, missing, 0),
            Err(Nx2MemoryError::InvalidHandle(9))
        );
        let (handle, _) = f
            .memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x1_0000)
            .unwrap();
        assert_eq!(
            f.memory.normal_map_4k(0x1000, 0x2fff, handle, 0xf000),
            Err(Nx2MemoryError::OffsetOutOfRange {
                handle: handle.raw(),
                offset: 0xf000,
                size: 0x2000
            })
        );
        assert_eq!(
            f.memory.unmap_4k(0x1000, 0x17ff),
            Err(Nx2MemoryError::InvalidRange {
                va: 0x1000,
                size: 0x800
            })
        );
        assert_eq!(
            f.memory.unmap_4k(0x2000, 0x1fff),
            Err(Nx2MemoryError::InvalidRange {
                va: 0x2000,
                size: 0
            })
        );
        f.memory.create_sparse_region(SPARSE_VA, 0x10_0000).unwrap();
        assert_eq!(
            f.memory
                .create_sparse_region(SPARSE_VA + 0xf_0000, 0x10_0000),
            Err(Nx2MemoryError::RangeInUse {
                va: SPARSE_VA + 0xf_0000,
                size: 0x10_0000
            })
        );
        assert_eq!(
            f.memory.create_sparse_region(CPU_VA, 0x1000),
            Err(Nx2MemoryError::RangeInUse {
                va: CPU_VA,
                size: 0x1000
            })
        );
        assert_eq!(
            f.memory.destroy_sparse_region(SPARSE_VA, 0x1000),
            Err(Nx2MemoryError::NoSuchRegion {
                va: SPARSE_VA,
                size: 0x1000
            })
        );
        assert_eq!(
            f.memory
                .create_gpu_accessible_memory(&f.space, CPU_VA + 0x30_0000, 0x1000),
            Err(Nx2MemoryError::Device(DeviceMemoryError::SourceUnmapped {
                va: CPU_VA + 0x30_0000
            }))
        );
        assert_eq!(f.device.allocated_bytes(), 0x1_0000);
    }

    #[test]
    fn unregistered_gmmus_stop_receiving_mappings() {
        let f = fixture();
        let id = f.memory.register_address_space(f.gmmus[0].clone());
        assert!(f.memory.unregister_address_space(id).is_some());
        f.memory
            .create_gpu_accessible_memory(&f.space, CPU_VA, 0x1000)
            .unwrap();
        assert_eq!(f.gmmus[0].translate(CPU_VA), None);
        assert!(f.memory.unregister_address_space(id).is_none());
    }
}
