//! Device pages GPU work may have written, shared by the renderer, which marks them, and the GPU,
//! which reads them. Switch 2 titles fetch command lists their own GPU work writes each frame:
//! a list on a marked page must be synchronised before it is read.

use std::sync::atomic::{AtomicU64, Ordering};

use lichii_common::DAddr;

const PAGE_BITS: u32 = 12;

/// One bit per 4 KiB page of the device address space, allocated once: 2 MiB for 36 bits,
/// 512 KiB for 34. Marks stay for the whole session; work that rewrites a page may still be
/// queued on the back when the front reads it, so a cleared bit could hide that write.
pub struct GpuWrittenPages {
    words: Box<[AtomicU64]>,
}

impl GpuWrittenPages {
    /// Covers a device address space of `2^address_space_bits` bytes.
    pub fn new(address_space_bits: u32) -> Self {
        let pages = 1u64 << address_space_bits.saturating_sub(PAGE_BITS);
        Self {
            words: (0..pages.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    /// Marks every page `[addr, addr + size)` touches. Lock-free and allocation-free; words
    /// that already hold the bits are only read. Pages past the address space are ignored.
    pub fn mark(&self, addr: DAddr, size: u64) {
        let Some((first, last)) = self.pages(addr, size) else {
            return;
        };
        self.for_each_word(first, last.min(self.page_count() - 1), |word, mask| {
            if word.load(Ordering::Relaxed) & mask != mask {
                word.fetch_or(mask, Ordering::Release);
            }
            false
        });
    }

    /// False only when no page of `[addr, addr + size)` was marked. Ranges reaching past the
    /// address space answer true.
    pub fn may_be_written(&self, addr: DAddr, size: u64) -> bool {
        let Some((first, last)) = self.pages(addr, size) else {
            return size != 0;
        };
        if last >= self.page_count() {
            return true;
        }
        self.for_each_word(first, last, |word, mask| {
            word.load(Ordering::Acquire) & mask != 0
        })
    }

    /// A new emulation session starts: no page is GPU-written.
    pub fn clear(&self) {
        for word in &self.words {
            word.store(0, Ordering::Relaxed);
        }
    }

    fn page_count(&self) -> u64 {
        self.words.len() as u64 * 64
    }

    /// First and last page of a non-empty range that starts inside the address space.
    fn pages(&self, addr: DAddr, size: u64) -> Option<(u64, u64)> {
        let end = addr.checked_add(size.checked_sub(1)?)?;
        let first = addr >> PAGE_BITS;
        (first < self.page_count()).then_some((first, end >> PAGE_BITS))
    }

    /// Calls `f` with each word of pages `first..=last` and the mask of those pages in it,
    /// until it returns true; returns whether it did.
    fn for_each_word(
        &self,
        first: u64,
        last: u64,
        mut f: impl FnMut(&AtomicU64, u64) -> bool,
    ) -> bool {
        let mut page = first;
        while page <= last {
            let bit = page % 64;
            let count = (64 - bit).min(last - page + 1);
            let mask = u64::MAX
                .checked_shl(count as u32)
                .map_or(u64::MAX, |high| !high)
                << bit;
            if f(&self.words[(page / 64) as usize], mask) {
                return true;
            }
            page += count;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 1 << PAGE_BITS;

    #[test]
    fn marks_exactly_the_pages_a_range_touches() {
        let pages = GpuWrittenPages::new(36);
        pages.mark(3 * PAGE + 100, 8);
        assert!(pages.may_be_written(3 * PAGE, 1));
        assert!(!pages.may_be_written(2 * PAGE, PAGE));
        assert!(!pages.may_be_written(4 * PAGE, PAGE));
        pages.mark(63 * PAGE + PAGE - 1, 2);
        assert!(pages.may_be_written(63 * PAGE, 1));
        assert!(pages.may_be_written(64 * PAGE, 1));
        assert!(!pages.may_be_written(65 * PAGE, 1));
        assert!(pages.may_be_written(10 * PAGE, 60 * PAGE));
        assert!(!pages.may_be_written(4 * PAGE, 59 * PAGE));
    }

    #[test]
    fn large_marks_fill_whole_words() {
        let pages = GpuWrittenPages::new(34);
        pages.mark(5 * PAGE, 200 * PAGE);
        assert!(!pages.may_be_written(4 * PAGE, PAGE));
        assert!((5..205).all(|page| pages.may_be_written(page * PAGE, 1)));
        assert!(!pages.may_be_written(205 * PAGE, PAGE));
    }

    #[test]
    fn edges_of_the_address_space_and_empty_ranges() {
        let pages = GpuWrittenPages::new(34);
        let end = 1u64 << 34;
        assert!(!pages.may_be_written(0, 0));
        assert!(pages.may_be_written(end - PAGE, 2 * PAGE));
        assert!(pages.may_be_written(end, 1));
        assert!(pages.may_be_written(u64::MAX, 2));
        pages.mark(end - 1, 16 * PAGE);
        assert!(pages.may_be_written(end - 1, 1));
        pages.mark(end, PAGE);
        pages.mark(0, 0);
        assert!(!pages.may_be_written(0, PAGE));
    }

    #[test]
    fn clear_forgets_every_mark() {
        let pages = GpuWrittenPages::new(34);
        pages.mark(0, 1 << 30);
        pages.clear();
        assert!(!pages.may_be_written(0, 1 << 30));
    }
}
