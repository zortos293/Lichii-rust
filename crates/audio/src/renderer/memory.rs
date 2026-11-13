//! Memory pools: the guest regions a game attaches so the renderer may read and write them, and
//! the address records that resolve a buffer through its pool.

use std::cell::Cell;

use bytemuck::{Pod, Zeroable};
use lichii_common::PAGE_SIZE;

use super::behavior::ErrorInfo;
use super::result::INVALID_ADDRESS_INFO;

/// Which address space a pool lives in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolLocation {
    Cpu = 1,
    Dsp = 2,
}

/// Pool states as the game encodes them.
pub mod pool_state {
    pub const INVALID: u32 = 0;
    pub const ACQUIRED: u32 = 1;
    pub const REQUEST_DETACH: u32 = 2;
    pub const DETACHED: u32 = 3;
    pub const REQUEST_ATTACH: u32 = 4;
    pub const ATTACHED: u32 = 5;
    pub const RELEASED: u32 = 6;
}

/// Outcome of a pool state change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolUpdate {
    Success,
    BadParam,
    MapFailed,
    InUse,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MemoryPoolInParameter {
    pub address: u64,
    pub size: u64,
    pub state: u32,
    pub in_use: u8,
    pub _pad: [u8; 3],
    pub _unk_18: [u8; 8],
}
const _: () = assert!(size_of::<MemoryPoolInParameter>() == 0x20);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct MemoryPoolOutStatus {
    pub state: u32,
    pub _unk_04: [u8; 0xC],
}
const _: () = assert!(size_of::<MemoryPoolOutStatus>() == 0x10);

/// One memory pool. Mapping is the identity: the DSP address equals the guest address.
#[derive(Clone, Debug)]
pub struct MemoryPoolInfo {
    cpu_address: u64,
    dsp_address: u64,
    size: u64,
    location: PoolLocation,
    in_use: Cell<bool>,
}

impl MemoryPoolInfo {
    pub fn new(location: PoolLocation) -> Self {
        Self {
            cpu_address: 0,
            dsp_address: 0,
            size: 0,
            location,
            in_use: Cell::new(false),
        }
    }

    pub fn cpu_address(&self) -> u64 {
        self.cpu_address
    }

    pub fn dsp_address(&self) -> u64 {
        self.dsp_address
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn location(&self) -> PoolLocation {
        self.location
    }

    pub fn contains(&self, address: u64, size: u64) -> bool {
        self.cpu_address <= address
            && address.wrapping_add(size) <= self.cpu_address.wrapping_add(self.size)
    }

    pub fn is_mapped(&self) -> bool {
        self.dsp_address != 0
    }

    /// DSP address of `address`, or 0 when the pool does not hold it or is not mapped.
    pub fn translate(&self, address: u64, size: u64) -> u64 {
        if !self.contains(address, size) || !self.is_mapped() {
            return 0;
        }
        self.dsp_address + (address - self.cpu_address)
    }

    pub fn set_used(&self, used: bool) {
        self.in_use.set(used);
    }

    pub fn is_used(&self) -> bool {
        self.in_use.get()
    }

    fn map(&mut self) {
        self.dsp_address = self.cpu_address;
    }

    fn unmap(&mut self) {
        self.cpu_address = 0;
        self.size = 0;
        self.dsp_address = 0;
    }

    /// Applies a requested attach or detach.
    pub fn update(
        &mut self,
        params: &MemoryPoolInParameter,
        out: &mut MemoryPoolOutStatus,
    ) -> PoolUpdate {
        if params.state != pool_state::REQUEST_ATTACH && params.state != pool_state::REQUEST_DETACH
        {
            return PoolUpdate::Success;
        }
        if params.address == 0
            || params.size == 0
            || !params.address.is_multiple_of(PAGE_SIZE)
            || !params.size.is_multiple_of(PAGE_SIZE)
        {
            return PoolUpdate::BadParam;
        }
        if params.state == pool_state::REQUEST_ATTACH {
            self.cpu_address = params.address;
            self.size = params.size;
            self.map();
            if !self.is_mapped() {
                self.cpu_address = 0;
                self.size = 0;
                return PoolUpdate::MapFailed;
            }
            out.state = pool_state::ATTACHED;
            return PoolUpdate::Success;
        }
        if self.cpu_address != params.address || self.size != params.size {
            return PoolUpdate::BadParam;
        }
        if self.is_used() {
            return PoolUpdate::InUse;
        }
        self.unmap();
        out.state = pool_state::DETACHED;
        PoolUpdate::Success
    }
}

/// Clears the in-use marks before a command list marks the pools it reads.
pub fn clear_use_state(pools: &[MemoryPoolInfo]) {
    for pool in pools {
        pool.set_used(false);
    }
}

/// A guest buffer and the pool (or forced mapping) it resolves through.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AddressInfo {
    cpu_address: u64,
    size: u64,
    pool: Option<u32>,
    dsp_address: u64,
}

impl AddressInfo {
    pub fn setup(&mut self, cpu_address: u64, size: u64) {
        *self = Self {
            cpu_address,
            size,
            pool: None,
            dsp_address: 0,
        };
    }

    pub fn cpu_address(&self) -> u64 {
        self.cpu_address
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn pool(&self) -> Option<u32> {
        self.pool
    }

    pub fn force_mapped_dsp_address(&self) -> u64 {
        self.dsp_address
    }

    fn mapped_pool<'p>(&self, pools: &'p [MemoryPoolInfo]) -> Option<&'p MemoryPoolInfo> {
        self.pool
            .and_then(|index| pools.get(index as usize))
            .filter(|pool| pool.dsp_address() != 0)
    }

    pub fn has_mapped_pool(&self, pools: &[MemoryPoolInfo]) -> bool {
        self.mapped_pool(pools).is_some()
    }

    pub fn is_mapped(&self, pools: &[MemoryPoolInfo]) -> bool {
        self.has_mapped_pool(pools) || self.dsp_address != 0
    }

    /// DSP address of the buffer, marking its pool used when `mark_in_use` is set.
    pub fn reference(&self, pools: &[MemoryPoolInfo], mark_in_use: bool) -> u64 {
        let Some(pool) = self.mapped_pool(pools) else {
            return self.dsp_address;
        };
        if mark_in_use {
            pool.set_used(true);
        }
        pool.translate(self.cpu_address, self.size)
    }
}

/// Resolves guest buffers against the pools of one update.
pub struct PoolMapper<'a> {
    pools: &'a [MemoryPoolInfo],
    force_map: bool,
}

impl<'a> PoolMapper<'a> {
    pub fn new(pools: &'a [MemoryPoolInfo], force_map: bool) -> Self {
        Self { pools, force_map }
    }

    pub fn is_force_map_enabled(&self) -> bool {
        self.force_map
    }

    pub fn find_pool(&self, address: u64, size: u64) -> Option<u32> {
        self.pools
            .iter()
            .position(|pool| pool.contains(address, size))
            .map(|index| index as u32)
    }

    fn fill_dsp_address(&self, info: &mut AddressInfo) -> bool {
        if info.cpu_address == 0 {
            info.pool = None;
            return false;
        }
        if let Some(pool) = self.find_pool(info.cpu_address, info.size) {
            info.pool = Some(pool);
            return true;
        }
        if self.force_map {
            info.dsp_address = info.cpu_address;
        } else {
            info.pool = None;
        }
        false
    }

    /// Points `info` at `address`. Returns whether the buffer is usable: inside a pool, or
    /// forced. A buffer outside every pool also reports `ResultInvalidAddressInfo`.
    pub fn try_attach_buffer(
        &self,
        error: &mut ErrorInfo,
        info: &mut AddressInfo,
        address: u64,
        size: u64,
    ) -> bool {
        info.setup(address, size);
        if !self.fill_dsp_address(info) {
            error.set(INVALID_ADDRESS_INFO, address);
            return self.force_map;
        }
        error.set_success();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn attach(pool: &mut MemoryPoolInfo, address: u64, size: u64) -> PoolUpdate {
        let params = MemoryPoolInParameter {
            address,
            size,
            state: pool_state::REQUEST_ATTACH,
            ..Default::default()
        };
        pool.update(&params, &mut MemoryPoolOutStatus::default())
    }

    #[test]
    fn attach_detach_cycle() {
        let mut pool = MemoryPoolInfo::new(PoolLocation::Dsp);
        assert_eq!(attach(&mut pool, 0x1001, 0x1000), PoolUpdate::BadParam);
        assert_eq!(attach(&mut pool, 0x10000, 0x2000), PoolUpdate::Success);
        assert_eq!(pool.translate(0x10800, 0x100), 0x10800);
        assert_eq!(pool.translate(0x11f00, 0x200), 0);
        pool.set_used(true);
        let detach = MemoryPoolInParameter {
            address: 0x10000,
            size: 0x2000,
            state: pool_state::REQUEST_DETACH,
            ..Default::default()
        };
        let mut out = MemoryPoolOutStatus::default();
        assert_eq!(pool.update(&detach, &mut out), PoolUpdate::InUse);
        pool.set_used(false);
        assert_eq!(pool.update(&detach, &mut out), PoolUpdate::Success);
        assert_eq!(out.state, pool_state::DETACHED);
        assert!(!pool.is_mapped());
    }

    #[test]
    fn buffers_resolve_through_pools_or_force_mapping() {
        let mut pools = vec![MemoryPoolInfo::new(PoolLocation::Dsp); 2];
        attach(&mut pools[1], 0x20000, 0x1000);
        let mut error = ErrorInfo::default();
        let mut info = AddressInfo::default();
        let mapper = PoolMapper::new(&pools, false);
        assert!(mapper.try_attach_buffer(&mut error, &mut info, 0x20100, 0x100));
        assert!(!error.is_error());
        assert_eq!(info.reference(&pools, true), 0x20100);
        assert!(pools[1].is_used());

        assert!(!mapper.try_attach_buffer(&mut error, &mut info, 0x50000, 0x100));
        assert_eq!(error, ErrorInfo::new(INVALID_ADDRESS_INFO, 0x50000));
        assert_eq!(info.reference(&pools, false), 0);

        let forced = PoolMapper::new(&pools, true);
        assert!(forced.try_attach_buffer(&mut error, &mut info, 0x50000, 0x100));
        assert!(error.is_error());
        assert_eq!(info.reference(&pools, false), 0x50000);
    }
}
