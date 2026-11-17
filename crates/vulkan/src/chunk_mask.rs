//! Which 64 KiB chunks of device memory a cache holds objects in, so that a guest write touching
//! none of them skips the cache's lookups after a load or two.

use lichii_common::DAddr;

/// Device bytes one bit covers: `1 << CHUNK_BITS`.
const CHUNK_BITS: u32 = 16;

/// One bit per 64 KiB chunk of device memory, set while an object occupies part of the chunk,
/// in words grown to the highest chunk ever set: 8 bytes per 4 MiB of device memory below it,
/// so 32 KiB at most for a 34-bit device address space and 128 KiB for 36 bits.
#[derive(Default)]
pub(crate) struct ChunkMask {
    words: Vec<u64>,
}

impl ChunkMask {
    /// Marks the chunks of `[daddr, daddr + size)`; an empty range marks the chunk of `daddr`.
    pub fn insert(&mut self, daddr: DAddr, size: u64) {
        let (first, last) = chunks(daddr, size);
        let len = (last / 64 + 1) as usize;
        if self.words.len() < len {
            self.words.resize(len, 0);
        }
        for_each_word(first, last, |word, mask| self.words[word] |= mask);
    }

    /// Clears each marked chunk of `[daddr, daddr + size)` (an empty range: the chunk of
    /// `daddr`) for whose bytes `[lo, hi)` `occupied(lo, hi)` is false.
    pub fn retain(
        &mut self,
        daddr: DAddr,
        size: u64,
        mut occupied: impl FnMut(DAddr, DAddr) -> bool,
    ) {
        let (first, last) = chunks(daddr, size);
        let last = last.min((self.words.len() as u64 * 64).saturating_sub(1));
        for chunk in first..=last {
            let (word, bit) = ((chunk / 64) as usize, 1 << (chunk % 64));
            let Some(bits) = self.words.get_mut(word) else {
                return;
            };
            let lo = chunk << CHUNK_BITS;
            if *bits & bit != 0 && !occupied(lo, lo.saturating_add(1 << CHUNK_BITS)) {
                *bits &= !bit;
            }
        }
    }

    /// False only when no chunk of `[daddr, daddr + size)` is marked; an empty range tests the
    /// chunk of `daddr`.
    #[inline]
    pub fn any(&self, daddr: DAddr, size: u64) -> bool {
        let (first, last) = chunks(daddr, size);
        if first / 64 >= self.words.len() as u64 {
            return false;
        }
        let last = last.min(self.words.len() as u64 * 64 - 1);
        let mut found = false;
        for_each_word(first, last, |word, mask| {
            found |= self.words[word] & mask != 0
        });
        found
    }
}

#[cfg(test)]
impl ChunkMask {
    /// The marked chunks' first bytes, in order.
    pub fn marked(&self) -> std::collections::BTreeSet<DAddr> {
        (0..self.words.len() as u64 * 64)
            .filter(|&c| self.words[(c / 64) as usize] & 1 << (c % 64) != 0)
            .map(|c| c << CHUNK_BITS)
            .collect()
    }
}

/// The first bytes of the chunks of `[daddr, daddr + size.max(1))`, which `insert` marks.
#[cfg(test)]
pub fn chunks_of(daddr: DAddr, size: u64) -> impl Iterator<Item = DAddr> {
    let (first, last) = chunks(daddr, size);
    (first..=last).map(|c| c << CHUNK_BITS)
}

/// The first and last chunk of `[daddr, daddr + size.max(1))`, clamped to the address space.
#[inline]
fn chunks(daddr: DAddr, size: u64) -> (u64, u64) {
    let last = daddr.saturating_add(size.max(1) - 1);
    (daddr >> CHUNK_BITS, last >> CHUNK_BITS)
}

/// Calls `f` with the index of each word holding chunks `first..=last` and the mask of those
/// chunks in it.
#[inline]
fn for_each_word(first: u64, last: u64, mut f: impl FnMut(usize, u64)) {
    let (first_word, last_word) = (first / 64, last / 64);
    for word in first_word..=last_word {
        let lo = if word == first_word { first % 64 } else { 0 };
        let hi = if word == last_word { last % 64 } else { 63 };
        f(word as usize, (u64::MAX << lo) & (u64::MAX >> (63 - hi)));
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// A seeded random stream of object insertions and removals keeps the mask equal to the
    /// chunks the live objects occupy, and `any` answers as a scan over every object does.
    #[test]
    fn mask_answers_as_a_scan_over_every_object() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut mask = ChunkMask::default();
        let mut objects: Vec<(DAddr, u64)> = Vec::new();
        let occupies = |&(daddr, size): &(DAddr, u64), lo: DAddr, hi: DAddr| {
            daddr < hi && daddr.saturating_add(size.max(1)) > lo
        };
        for step in 0..20_000 {
            let span = 1u64 << (20 + next() % 12);
            let daddr = next() % span;
            let size = match next() % 6 {
                0 => 0,
                1 => 1 << CHUNK_BITS,
                2 => next() % (4 << 20),
                _ => next() % 0x2000,
            };
            if next() % 3 == 0 && !objects.is_empty() {
                let (daddr, size) = objects.swap_remove((next() % objects.len() as u64) as usize);
                mask.retain(daddr, size, |lo, hi| {
                    objects.iter().any(|o| occupies(o, lo, hi))
                });
            } else if next() % 2 == 0 {
                objects.push((daddr, size));
                mask.insert(daddr, size);
            }
            let expected: BTreeSet<DAddr> = objects
                .iter()
                .flat_map(|&(daddr, size)| chunks_of(daddr, size))
                .collect();
            assert_eq!(
                mask.any(daddr, size),
                chunks_of(daddr, size).any(|c| expected.contains(&c)),
                "step {step}: [{daddr:#x}, +{size:#x})"
            );
            if step % 512 == 0 {
                assert_eq!(mask.marked(), expected, "step {step}");
            }
        }
        assert!(!mask.any(u64::MAX - 4, 100));
        let mut mask = ChunkMask::default();
        mask.insert(0, 0);
        assert!(mask.any(0, 0) && mask.any(0xffff, 1) && !mask.any(0x1_0000, 0));
    }
}
