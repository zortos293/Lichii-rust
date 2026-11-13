//! Switch 2 queue semaphores (NX2_NOTES row 35). NVN2 releases a queue's progress values into
//! the second page of a two-page GPU-accessible memory, the CPU polls them there, and GPU acquires
//! read the same offsets in the first page. Something on the console copies one page into the
//! other; until that is known, every semaphore release that lands in a second page is mirrored
//! into the first.
//!
//! A first-page word can also take releases of its own. Once it has, a mirrored value no longer
//! goes to memory, where it would undo them: it is kept aside, and acquires see it until the
//! word changes again.
//!
//! Regions live in a fixed open-addressed table that releases and acquires probe with atomic
//! loads. Each region's first page has its own lock, which covers both the words kept aside and
//! every GPU write and acquire read of that page, so an acquire never sees a release halfway.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use lichii_common::DAddr;
use parking_lot::{Mutex, MutexGuard};

/// Size of the GPU-accessible memories that hold queue semaphores.
pub const REGION_SIZE: u64 = 0x2000;
pub(crate) const PAGE: u64 = 0x1000;
const WORDS: usize = (PAGE / 4) as usize;
const SLOTS: usize = 1 << 10;
const MAX_PROBE: usize = 16;
const EMPTY: u64 = u64::MAX;
const REMOVED: u64 = u64::MAX - 1;
const CLAIMED: u64 = u64::MAX - 2;

/// The two-page semaphore regions and the mirrored values kept out of memory. Without a region
/// (Switch 1, or before NVN2 makes one), every check is one atomic load.
#[derive(Default)]
pub struct SemaphoreMirror {
    slots: OnceLock<Box<[Slot]>>,
    region_count: AtomicUsize,
}

struct Slot {
    /// Device address of the region's first page, or one of the markers.
    start: AtomicU64,
    words: OnceLock<Box<Mutex<Words>>>,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            start: AtomicU64::new(EMPTY),
            words: OnceLock::new(),
        }
    }
}

struct Words {
    /// First-page words with releases of their own, one bit each.
    direct: [u64; WORDS / 64],
    /// Mirrored values of such words, with what guest memory held when they were set.
    kept: [Option<Kept>; WORDS],
}

impl Default for Words {
    fn default() -> Self {
        Self {
            direct: [0; WORDS / 64],
            kept: [None; WORDS],
        }
    }
}

#[derive(Clone, Copy)]
struct Kept {
    value: u32,
    guest: u32,
}

fn home(start: DAddr) -> usize {
    ((start >> 12).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - SLOTS.trailing_zeros())) as usize
}

impl SemaphoreMirror {
    /// GPU-accessible memory of [`REGION_SIZE`] bytes was created at device address `start`.
    pub fn add_region(&self, start: DAddr) {
        if self.slot(start).is_some() {
            return;
        }
        let slots = self
            .slots
            .get_or_init(|| (0..SLOTS).map(|_| Slot::default()).collect());
        let home = home(start);
        for probe in 0..MAX_PROBE {
            let slot = &slots[(home + probe) % SLOTS];
            let current = slot.start.load(Ordering::Acquire);
            if current != EMPTY && current != REMOVED {
                continue;
            }
            if slot
                .start
                .compare_exchange(current, CLAIMED, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let mut words = slot.words.get_or_init(Box::default).lock();
            words.direct.fill(0);
            words.kept.fill(None);
            slot.start.store(start, Ordering::Release);
            drop(words);
            self.region_count.fetch_add(1, Ordering::AcqRel);
            return;
        }
        tracing::warn!(
            start,
            "no room to mirror another Switch 2 queue semaphore region"
        );
    }

    /// The region's memory was destroyed: nothing is mirrored into it any more.
    pub fn remove_region(&self, start: DAddr) {
        let Some(slot) = self.slot(start) else {
            return;
        };
        let _words = slot.words.get().map(|words| words.lock());
        if slot
            .start
            .compare_exchange(start, REMOVED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.region_count.fetch_sub(1, Ordering::AcqRel);
        }
    }

    fn slot(&self, start: DAddr) -> Option<&Slot> {
        if self.region_count.load(Ordering::Acquire) == 0 {
            return None;
        }
        let slots = self.slots.get()?;
        let home = home(start);
        (0..MAX_PROBE)
            .map(|probe| &slots[(home + probe) % SLOTS])
            .take_while(|slot| slot.start.load(Ordering::Acquire) != EMPTY)
            .find(|slot| slot.start.load(Ordering::Acquire) == start)
    }

    /// The first page holding `address`, locked, when it is a region's.
    pub(crate) fn first_page(&self, address: DAddr) -> Option<FirstPage<'_>> {
        let start = address & !(PAGE - 1);
        let slot = self.slot(start)?;
        let words = slot.words.get()?.lock();
        (slot.start.load(Ordering::Acquire) == start).then_some(FirstPage { start, words })
    }

    /// The first page, locked, that a release landing on `address` is mirrored into, when it
    /// landed in a region's second page. The release's words go to the same offsets there.
    pub(crate) fn mirror_of(&self, address: DAddr) -> Option<FirstPage<'_>> {
        self.first_page(address.checked_sub(PAGE)?)
    }

    /// What a GPU acquire of the 32-bit word at `address` sees: guest memory, as `load` reads
    /// it, unless the word holds a mirrored value kept out of memory.
    pub(crate) fn acquire_view<E>(
        &self,
        address: DAddr,
        load: impl FnOnce() -> Result<u32, E>,
    ) -> Result<u32, E> {
        let mut page = self.first_page(address);
        let guest = load()?;
        Ok(page
            .as_mut()
            .and_then(|page| page.seen(address, guest))
            .unwrap_or(guest))
    }
}

/// A region's first page, locked: GPU writes to it and acquire reads of it go through here.
pub(crate) struct FirstPage<'a> {
    start: DAddr,
    words: MutexGuard<'a, Words>,
}

impl FirstPage<'_> {
    fn index(&self, address: DAddr) -> Option<usize> {
        let offset = address.checked_sub(self.start)?;
        (offset < PAGE).then_some((offset / 4) as usize)
    }

    /// Whether `address` lies in this page.
    pub(crate) fn contains(&self, address: DAddr) -> bool {
        self.index(address).is_some()
    }

    /// A release that is not a mirror's wrote `bytes` bytes at `address`: every word it covers
    /// has releases of its own from now on.
    pub(crate) fn note_direct(&mut self, address: DAddr, bytes: u64) {
        let mut word = address & !3;
        while let Some(index) = self.index(word).filter(|_| word < address + bytes) {
            self.words.direct[index / 64] |= 1 << (index % 64);
            self.words.kept[index] = None;
            word += 4;
        }
    }

    /// Whether a mirrored release into the word at `address` is kept aside instead of written to
    /// memory.
    pub(crate) fn diverts(&self, address: DAddr) -> bool {
        self.index(address)
            .is_some_and(|index| self.words.direct[index / 64] >> (index % 64) & 1 != 0)
    }

    /// Keeps `value` aside for the word at `address`, whose guest memory holds `guest` now.
    pub(crate) fn keep(&mut self, address: DAddr, value: u32, guest: u32) {
        if let Some(index) = self.index(address) {
            self.words.kept[index] = Some(Kept { value, guest });
        }
    }

    /// The mirrored value an acquire of the word at `address` sees while guest memory holds
    /// `guest`, if any. Once the word changed in memory without a release (a CPU write), memory
    /// is newer.
    pub(crate) fn seen(&mut self, address: DAddr, guest: u32) -> Option<u32> {
        let index = self.index(address)?;
        let kept = self.words.kept[index]?;
        if kept.guest != guest {
            self.words.kept[index] = None;
            return None;
        }
        Some(kept.value)
    }

    /// A release that combines with its word (a reduction) took guest memory at `address` from
    /// `before` to `after`. While the word holds a mirrored value, acquires see
    /// `combine(mirrored)` instead: what the word would hold without the mirror keeping its value
    /// aside. Returns what acquires see.
    pub(crate) fn combine(
        &mut self,
        address: DAddr,
        before: u32,
        after: u32,
        combine: impl FnOnce(u32) -> u32,
    ) -> u32 {
        let mirrored = self.seen(address, before);
        self.note_direct(address, 4);
        let Some(mirrored) = mirrored else {
            return after;
        };
        let seen = combine(mirrored);
        self.keep(address, seen, after);
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGION: DAddr = 0x8_0002_7000;
    const OTHER: DAddr = 0x8_0010_0000;

    fn target(mirror: &SemaphoreMirror, address: DAddr) -> Option<DAddr> {
        mirror.mirror_of(address).map(|page| page.start)
    }

    #[test]
    fn second_page_releases_mirror_into_the_first_page() {
        let mirror = SemaphoreMirror::default();
        assert_eq!(target(&mirror, REGION + 0x1010), None);
        mirror.add_region(REGION);
        assert_eq!(target(&mirror, REGION + 0x1000), Some(REGION));
        assert_eq!(target(&mirror, REGION + 0x1ffc), Some(REGION));
        assert_eq!(target(&mirror, REGION), None);
        assert_eq!(target(&mirror, REGION + 0xffc), None);
        assert_eq!(target(&mirror, REGION + 0x2000), None);
        assert_eq!(target(&mirror, OTHER + 0x1010), None);
        assert!(mirror.first_page(REGION + 0xffc).is_some());
        assert!(mirror.first_page(REGION + 0x1000).is_none());
        mirror.add_region(OTHER);
        mirror.remove_region(REGION);
        assert_eq!(target(&mirror, REGION + 0x1010), None);
        assert_eq!(target(&mirror, OTHER + 0x1010), Some(OTHER));
        assert_eq!(target(&mirror, 0x10), None);
    }

    #[test]
    fn mirrored_values_stay_out_of_words_with_direct_releases() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        let mut page = mirror.first_page(REGION).unwrap();
        assert!(!page.diverts(REGION));
        page.note_direct(REGION, 4);
        assert!(page.diverts(REGION));
        assert!(page.diverts(REGION + 2));
        assert!(!page.diverts(REGION + 4));
        page.note_direct(REGION + 0x10, 16);
        assert!((0x10..0x20).step_by(4).all(|o| page.diverts(REGION + o)));
        assert!(!page.diverts(REGION + 0x20));
        page.note_direct(REGION + 0xff8, 16);
        assert!(page.diverts(REGION + 0xffc));
        assert!(!page.diverts(REGION + 0x1000));

        assert_eq!(page.seen(REGION, 7), None);
        page.keep(REGION, 42, 7);
        assert_eq!(page.seen(REGION, 7), Some(42));
        assert_eq!(page.seen(REGION, 8), None);
        assert_eq!(page.seen(REGION, 7), None);

        page.keep(REGION, 43, 7);
        page.note_direct(REGION, 4);
        assert_eq!(page.seen(REGION, 7), None);
    }

    #[test]
    fn combined_releases_combine_with_the_mirrored_value() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        let mut page = mirror.first_page(REGION).unwrap();
        page.note_direct(REGION, 4);
        let max_with = |payload: u32| move |current: u32| current.max(payload);
        page.keep(REGION, 10, 0);
        assert_eq!(page.combine(REGION, 0, 3, max_with(3)), 10);
        assert_eq!(page.seen(REGION, 3), Some(10));
        assert_eq!(page.combine(REGION, 3, 12, max_with(12)), 12);
        assert_eq!(page.seen(REGION, 12), Some(12));
        assert_eq!(page.combine(REGION + 4, 5, 6, max_with(6)), 6);
        assert_eq!(page.seen(REGION + 4, 6), None);
    }

    #[test]
    fn removing_a_region_forgets_its_words() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        let mut page = mirror.first_page(REGION).unwrap();
        page.note_direct(REGION, 4);
        page.keep(REGION, 44, 7);
        drop(page);
        mirror.remove_region(REGION);
        assert!(mirror.first_page(REGION).is_none());
        mirror.add_region(REGION);
        let mut page = mirror.first_page(REGION).unwrap();
        assert!(!page.diverts(REGION));
        assert_eq!(page.seen(REGION, 7), None);
    }

    #[test]
    fn many_regions_come_and_go() {
        let mirror = SemaphoreMirror::default();
        let region = |n: u64| 0x8_0000_0000 + n * REGION_SIZE;
        (0..512).for_each(|n| mirror.add_region(region(n)));
        assert!((0..512).all(|n| target(&mirror, region(n) + 0x1004) == Some(region(n))));
        (0..512)
            .step_by(2)
            .for_each(|n| mirror.remove_region(region(n)));
        (512..768).for_each(|n| mirror.add_region(region(n)));
        for n in 0..768 {
            let expected = (n % 2 == 1 || n >= 512).then_some(region(n));
            assert_eq!(target(&mirror, region(n) + 0x1004), expected, "region {n}");
        }
    }
}
