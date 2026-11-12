//! Decoded blocks keyed by guest address.

use std::collections::BTreeMap;

use lichii_common::{PAGE_MASK, PAGE_SIZE, VAddr};
use lichii_memory::AddressSpace;

use crate::CpuFault;
use crate::decoder::{Inst, decode};

const FAST_BITS: u32 = 14;
const EMPTY: (u64, u32) = (u64::MAX, u32::MAX);

/// Decoded blocks, found through a direct-mapped table in front of an ordered map. Blocks never
/// cross a page, which bounds the search when a range is invalidated.
pub(super) struct BlockCache {
    blocks: Vec<Box<[Inst]>>,
    free: Vec<u32>,
    by_start: BTreeMap<u64, u32>,
    fast: Box<[(u64, u32)]>,
}

#[inline]
fn fast_slot(pc: u64) -> usize {
    ((pc >> 2) as usize) & ((1 << FAST_BITS) - 1)
}

impl BlockCache {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            free: Vec::new(),
            by_start: BTreeMap::new(),
            fast: vec![EMPTY; 1 << FAST_BITS].into_boxed_slice(),
        }
    }

    /// The block starting at `pc`, decoding it on first use.
    #[inline]
    pub fn get_or_translate(
        &mut self,
        pc: u64,
        memory: &AddressSpace,
    ) -> Result<&[Inst], CpuFault> {
        let slot = fast_slot(pc);
        let (start, index) = self.fast[slot];
        let index = if start == pc && index != EMPTY.1 {
            index
        } else {
            let index = match self.by_start.get(&pc) {
                Some(&index) => index,
                None => self.translate(pc, memory)?,
            };
            self.fast[slot] = (pc, index);
            index
        };
        Ok(&self.blocks[index as usize])
    }

    fn translate(&mut self, pc: u64, memory: &AddressSpace) -> Result<u32, CpuFault> {
        if pc & 3 != 0 {
            return Err(CpuFault::Prefetch { pc });
        }
        let mut insts = Vec::new();
        let mut addr = pc;
        loop {
            let word = memory
                .read::<u32>(addr)
                .map_err(|_| CpuFault::Prefetch { pc })?;
            let inst = decode(word);
            insts.push(inst);
            addr += 4;
            if inst.ends_block() || addr & PAGE_MASK == 0 {
                break;
            }
        }
        let block = insts.into_boxed_slice();
        let index = match self.free.pop() {
            Some(index) => {
                self.blocks[index as usize] = block;
                index
            }
            None => {
                self.blocks.push(block);
                (self.blocks.len() - 1) as u32
            }
        };
        self.by_start.insert(pc, index);
        Ok(index)
    }

    /// Drops every block that overlaps `[addr, addr + size)`.
    pub fn invalidate(&mut self, addr: VAddr, size: u64) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size);
        let first = addr.saturating_sub(PAGE_SIZE - 4);
        let doomed: Vec<(u64, u32)> = self
            .by_start
            .range(first..end)
            .filter(|&(&start, &index)| start + 4 * self.blocks[index as usize].len() as u64 > addr)
            .map(|(&start, &index)| (start, index))
            .collect();
        for (start, index) in doomed {
            self.by_start.remove(&start);
            let slot = fast_slot(start);
            if self.fast[slot].0 == start {
                self.fast[slot] = EMPTY;
            }
            self.blocks[index as usize] = Box::new([]);
            self.free.push(index);
        }
    }

    pub fn clear(&mut self) {
        self.blocks.clear();
        self.free.clear();
        self.by_start.clear();
        self.fast.fill(EMPTY);
    }
}
