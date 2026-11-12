//! Memory states, attributes and the per-process block map behind `QueryMemory`.

use std::collections::BTreeMap;

use lichii_common::{HleResult, VAddr};
use lichii_memory::MemoryPermission;

use crate::result;

/// Horizon's `KMemoryState`: the guest-visible state in bits 0..8, capability flags above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MemoryState(pub u32);

impl MemoryState {
    pub const MASK: u32 = 0xff;
    pub const FLAG_CAN_REPROTECT: u32 = 1 << 8;
    pub const FLAG_CAN_DEBUG: u32 = 1 << 9;
    pub const FLAG_CAN_USE_IPC: u32 = 1 << 10;
    pub const FLAG_CAN_USE_NON_DEVICE_IPC: u32 = 1 << 11;
    pub const FLAG_CAN_USE_NON_SECURE_IPC: u32 = 1 << 12;
    pub const FLAG_MAPPED: u32 = 1 << 13;
    pub const FLAG_CODE: u32 = 1 << 14;
    pub const FLAG_CAN_ALIAS: u32 = 1 << 15;
    pub const FLAG_CAN_CODE_ALIAS: u32 = 1 << 16;
    pub const FLAG_CAN_TRANSFER: u32 = 1 << 17;
    pub const FLAG_CAN_QUERY_PHYSICAL: u32 = 1 << 18;
    pub const FLAG_CAN_DEVICE_MAP: u32 = 1 << 19;
    pub const FLAG_CAN_ALIGNED_DEVICE_MAP: u32 = 1 << 20;
    pub const FLAG_CAN_IPC_USER_BUFFER: u32 = 1 << 21;
    pub const FLAG_REFERENCE_COUNTED: u32 = 1 << 22;
    pub const FLAG_CAN_MAP_PROCESS: u32 = 1 << 23;
    pub const FLAG_CAN_CHANGE_ATTRIBUTE: u32 = 1 << 24;
    pub const FLAG_CAN_CODE_MEMORY: u32 = 1 << 25;
    pub const FLAG_LINEAR_MAPPED: u32 = 1 << 26;
    pub const FLAG_CAN_PERMISSION_LOCK: u32 = 1 << 27;

    pub const FREE: Self = Self(0x0000_0000);
    pub const IO_MEMORY: Self = Self(0x0018_2001);
    pub const STATIC: Self = Self(0x0004_2002);
    pub const CODE: Self = Self(0x04dc_7e03);
    pub const CODE_DATA: Self = Self(0x0ffe_bd04);
    pub const NORMAL: Self = Self(0x077e_bd05);
    pub const SHARED: Self = Self(0x0440_2006);
    pub const ALIAS_CODE: Self = Self(0x04dd_7e08);
    pub const ALIAS_CODE_DATA: Self = Self(0x0fff_bd09);
    pub const IPC: Self = Self(0x045c_3c0a);
    pub const STACK: Self = Self(0x045c_3c0b);
    pub const THREAD_LOCAL: Self = Self(0x0400_000c);
    pub const TRANSFERRED: Self = Self(0x055c_3c0d);
    pub const SHARED_TRANSFERRED: Self = Self(0x045c_380e);
    pub const SHARED_CODE: Self = Self(0x0440_380f);
    pub const INACCESSIBLE: Self = Self(0x0000_0010);
    pub const NON_SECURE_IPC: Self = Self(0x045c_3811);
    pub const NON_DEVICE_IPC: Self = Self(0x044c_2812);
    pub const KERNEL: Self = Self(0x0000_0013);
    pub const GENERATED_CODE: Self = Self(0x0440_2214);
    pub const CODE_OUT: Self = Self(0x0440_2015);
    pub const COVERAGE: Self = Self(0x0000_2016);
    pub const INSECURE: Self = Self(0x055c_3817);

    pub const fn has(self, flag: u32) -> bool {
        self.0 & flag != 0
    }

    /// The value `QueryMemory` reports.
    pub const fn svc_value(self) -> u32 {
        self.0 & Self::MASK
    }
}

/// `MemoryAttribute` bits.
pub mod attribute {
    pub const LOCKED: u32 = 1 << 0;
    pub const IPC_LOCKED: u32 = 1 << 1;
    pub const DEVICE_SHARED: u32 = 1 << 2;
    pub const UNCACHED: u32 = 1 << 3;
    pub const PERMISSION_LOCKED: u32 = 1 << 4;
    /// Switch 2: memory registered with the GPU (svc 0x80).
    pub const GPU_ACCESSIBLE: u32 = 1 << 5;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub size: u64,
    pub state: MemoryState,
    pub perm: MemoryPermission,
    pub attr: u32,
    pub ipc_count: u16,
    pub device_count: u16,
}

impl Block {
    fn same_kind(&self, other: &Block) -> bool {
        self.state == other.state
            && self.perm == other.perm
            && self.attr == other.attr
            && self.ipc_count == other.ipc_count
            && self.device_count == other.device_count
    }
}

/// What `QueryMemory` writes (`MemoryInfo`, 0x28 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct MemoryInfo {
    pub base_address: u64,
    pub size: u64,
    pub state: u32,
    pub attribute: u32,
    pub permission: u32,
    pub ipc_count: u32,
    pub device_count: u32,
    pub padding: u32,
}

/// Expected state of a range: every block must satisfy `(field & mask) == value`.
#[derive(Clone, Copy, Debug)]
pub struct Expect {
    pub state_mask: u32,
    pub state: u32,
    pub perm_mask: u8,
    pub perm: u8,
    pub attr_mask: u32,
    pub attr: u32,
}

impl Expect {
    /// Any block with `flag` in its state, the given permission bits and none of `attr_mask`.
    pub const fn flag(flag: u32, perm_mask: u8, perm: u8, attr_mask: u32) -> Self {
        Self {
            state_mask: flag,
            state: flag,
            perm_mask,
            perm,
            attr_mask,
            attr: 0,
        }
    }

    /// Exactly `state`, any permission, none of `attr_mask`.
    pub const fn state(state: MemoryState, attr_mask: u32) -> Self {
        Self {
            state_mask: u32::MAX,
            state: state.0,
            perm_mask: 0,
            perm: 0,
            attr_mask,
            attr: 0,
        }
    }

    fn matches(&self, block: &Block) -> bool {
        block.state.0 & self.state_mask == self.state
            && block.perm.0 & self.perm_mask == self.perm
            && block.attr & self.attr_mask == self.attr
    }
}

/// Covers the whole address space with blocks; adjacent equal blocks are merged.
pub struct BlockMap {
    end: u64,
    blocks: BTreeMap<VAddr, Block>,
}

impl BlockMap {
    pub fn new(address_space_bits: u32) -> Self {
        let end = 1u64 << address_space_bits;
        let mut blocks = BTreeMap::new();
        blocks.insert(
            0,
            Block {
                size: end,
                state: MemoryState::FREE,
                perm: MemoryPermission::NONE,
                attr: 0,
                ipc_count: 0,
                device_count: 0,
            },
        );
        Self { end, blocks }
    }

    pub fn end(&self) -> u64 {
        self.end
    }

    /// The block containing `addr` and its start.
    pub fn find(&self, addr: VAddr) -> Option<(VAddr, Block)> {
        if addr >= self.end {
            return None;
        }
        self.blocks
            .range(..=addr)
            .next_back()
            .map(|(start, block)| (*start, *block))
    }

    pub fn query(&self, addr: VAddr) -> MemoryInfo {
        match self.find(addr) {
            Some((start, block)) => MemoryInfo {
                base_address: start,
                size: block.size,
                state: block.state.svc_value(),
                attribute: block.attr,
                permission: block.perm.0 as u32,
                ipc_count: block.ipc_count as u32,
                device_count: block.device_count as u32,
                padding: 0,
            },
            None => MemoryInfo {
                base_address: self.end,
                size: 0u64.wrapping_sub(self.end),
                state: MemoryState::INACCESSIBLE.svc_value(),
                attribute: 0,
                permission: 0,
                ipc_count: 0,
                device_count: 0,
                padding: 0,
            },
        }
    }

    /// Blocks overlapping `[addr, addr + size)`, clipped to it.
    pub fn iter_range(&self, addr: VAddr, size: u64) -> impl Iterator<Item = (VAddr, Block)> + '_ {
        let end = addr + size;
        let first = self.find(addr).map_or(addr, |(start, _)| start);
        self.blocks.range(first..end).map(move |(&start, block)| {
            let s = start.max(addr);
            let e = (start + block.size).min(end);
            (
                s,
                Block {
                    size: e - s,
                    ..*block
                },
            )
        })
    }

    /// Checks that every block of the range matches. Returns the first block's
    /// `(state, perm, attr)` when they also all agree.
    pub fn check(
        &self,
        addr: VAddr,
        size: u64,
        expect: Expect,
    ) -> HleResult<(MemoryState, MemoryPermission, u32)> {
        if size == 0 || addr.checked_add(size).is_none_or(|end| end > self.end) {
            return Err(result::INVALID_CURRENT_MEMORY);
        }
        let mut first: Option<Block> = None;
        for (_, block) in self.iter_range(addr, size) {
            if !expect.matches(&block) {
                return Err(result::INVALID_CURRENT_MEMORY);
            }
            match first {
                None => first = Some(block),
                Some(f)
                    if f.state != block.state || f.perm != block.perm || f.attr != block.attr =>
                {
                    return Err(result::INVALID_CURRENT_MEMORY);
                }
                Some(_) => {}
            }
        }
        let f = first.ok_or(result::INVALID_CURRENT_MEMORY)?;
        Ok((f.state, f.perm, f.attr))
    }

    /// Like [`BlockMap::check`] but blocks may differ from each other.
    pub fn check_each(&self, addr: VAddr, size: u64, expect: Expect) -> HleResult<()> {
        if size == 0 || addr.checked_add(size).is_none_or(|end| end > self.end) {
            return Err(result::INVALID_CURRENT_MEMORY);
        }
        if self.iter_range(addr, size).all(|(_, b)| expect.matches(&b)) {
            Ok(())
        } else {
            Err(result::INVALID_CURRENT_MEMORY)
        }
    }

    fn split_at(&mut self, addr: VAddr) {
        if addr >= self.end {
            return;
        }
        let (start, block) = self.find(addr).expect("address inside the map");
        if start == addr {
            return;
        }
        let head = addr - start;
        self.blocks.insert(
            start,
            Block {
                size: head,
                ..block
            },
        );
        self.blocks.insert(
            addr,
            Block {
                size: block.size - head,
                ..block
            },
        );
    }

    fn merge_around(&mut self, addr: VAddr, end: VAddr) {
        let keys: Vec<VAddr> = {
            let from = self
                .blocks
                .range(..addr)
                .next_back()
                .map_or(addr, |(s, _)| *s);
            self.blocks
                .range(from..=end.min(self.end - 1))
                .map(|(s, _)| *s)
                .collect()
        };
        let mut iter = keys.into_iter();
        let Some(mut current) = iter.next() else {
            return;
        };
        for next in iter {
            let a = self.blocks[&current];
            let b = self.blocks[&next];
            if current + a.size == next && a.same_kind(&b) {
                self.blocks.remove(&next);
                self.blocks.insert(
                    current,
                    Block {
                        size: a.size + b.size,
                        ..a
                    },
                );
            } else {
                current = next;
            }
        }
    }

    /// Applies `f` to every block of the range (splitting at its edges), then merges.
    pub fn update(&mut self, addr: VAddr, size: u64, mut f: impl FnMut(&mut Block)) {
        let end = addr + size;
        assert!(end <= self.end, "block update past the address space");
        self.split_at(addr);
        self.split_at(end);
        let keys: Vec<VAddr> = self.blocks.range(addr..end).map(|(s, _)| *s).collect();
        for key in keys {
            f(self.blocks.get_mut(&key).expect("block"));
        }
        self.merge_around(addr, end);
    }

    /// Sets the range to one state/permission/attribute.
    pub fn set(
        &mut self,
        addr: VAddr,
        size: u64,
        state: MemoryState,
        perm: MemoryPermission,
        attr: u32,
    ) {
        self.update(addr, size, |b| {
            b.state = state;
            b.perm = perm;
            b.attr = attr;
        });
    }

    /// Lowest `align`-aligned address in `[start, end)` where `size` free bytes fit with
    /// `guard` free bytes on each side.
    pub fn find_free(
        &self,
        start: VAddr,
        end: VAddr,
        size: u64,
        align: u64,
        guard: u64,
    ) -> Option<VAddr> {
        let needed = size + 2 * guard;
        let first = self.find(start).map_or(start, |(s, _)| s);
        for (&block_start, block) in self.blocks.range(first..end) {
            if block.state != MemoryState::FREE {
                continue;
            }
            let lo = block_start.max(start);
            let hi = (block_start + block.size).min(end);
            let addr = lichii_common::align_up(lo + guard, align);
            if addr >= lo + guard && addr + size + guard <= hi && hi - lo >= needed {
                return Some(addr);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_splits_and_merges() {
        let mut map = BlockMap::new(39);
        map.set(
            0x10000,
            0x4000,
            MemoryState::NORMAL,
            MemoryPermission::READ_WRITE,
            0,
        );
        map.set(
            0x14000,
            0x4000,
            MemoryState::NORMAL,
            MemoryPermission::READ_WRITE,
            0,
        );
        let info = map.query(0x15000);
        assert_eq!((info.base_address, info.size), (0x10000, 0x8000));
        assert_eq!(info.state, 5);
        map.set(
            0x12000,
            0x1000,
            MemoryState::NORMAL,
            MemoryPermission::READ,
            0,
        );
        assert_eq!(map.query(0x12000).size, 0x1000);
        assert_eq!(map.query(0x13000).base_address, 0x13000);
        map.set(
            0x12000,
            0x1000,
            MemoryState::NORMAL,
            MemoryPermission::READ_WRITE,
            0,
        );
        assert_eq!(map.query(0x10000).size, 0x8000);
        assert_eq!(map.query(0).size, 0x10000);
    }

    #[test]
    fn check_and_find_free() {
        let mut map = BlockMap::new(36);
        map.set(
            0x20_0000,
            0x1000,
            MemoryState::CODE,
            MemoryPermission::READ_EXECUTE,
            0,
        );
        assert!(
            map.check(0x20_0000, 0x1000, Expect::state(MemoryState::CODE, 0))
                .is_ok()
        );
        assert_eq!(
            map.check(0x20_0000, 0x2000, Expect::state(MemoryState::CODE, 0)),
            Err(result::INVALID_CURRENT_MEMORY)
        );
        let free = map
            .find_free(0x20_0000, 0x40_0000, 0x1000, 0x1000, 0x1000)
            .unwrap();
        assert_eq!(free, 0x20_2000);
        let past_end = map.query(1 << 36);
        assert_eq!(past_end.state, MemoryState::INACCESSIBLE.svc_value());
    }
}
