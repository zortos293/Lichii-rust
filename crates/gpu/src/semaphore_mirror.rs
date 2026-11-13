//! Switch 2 queue semaphores (NX2_NOTES row 35). NVN2 releases a queue's progress values into
//! the second page of a two-page GPU-accessible memory, the CPU polls them there, and GPU acquires
//! read the same offsets in the first page. Something on the console copies one page into the
//! other; until that is known, every semaphore release that lands in a second page is mirrored
//! into the first.
//!
//! A first-page word can also take releases of its own. Once it has, a mirrored value no longer
//! goes to memory, where it would undo them: it is kept aside, and acquires see it until the
//! word changes again.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lichii_common::DAddr;
use parking_lot::{Mutex, RwLock};

/// Size of the GPU-accessible memories that hold queue semaphores.
pub const REGION_SIZE: u64 = 0x2000;
const PAGE: u64 = 0x1000;
/// First-page words with releases of their own the mirror tracks, at most.
const MAX_DIRECT_WORDS: usize = 1024;

/// The two-page semaphore regions and the mirrored values kept out of memory. Without a region
/// (Switch 1, or before NVN2 makes one), every check is one atomic load.
#[derive(Default)]
pub struct SemaphoreMirror {
    /// Device addresses of the regions' first pages.
    regions: RwLock<HashSet<DAddr>>,
    region_count: AtomicUsize,
    shadow: Mutex<Shadow>,
    has_direct: AtomicBool,
    has_words: AtomicBool,
}

#[derive(Default)]
struct Shadow {
    /// First-page words with releases of their own.
    direct: HashSet<DAddr>,
    /// Mirrored values of such words, with what guest memory held when they were set.
    words: HashMap<DAddr, Word>,
}

#[derive(Clone, Copy)]
struct Word {
    value: u32,
    guest: u32,
}

impl SemaphoreMirror {
    /// GPU-accessible memory of [`REGION_SIZE`] bytes was created at device address `start`.
    pub fn add_region(&self, start: DAddr) {
        let mut regions = self.regions.write();
        regions.insert(start);
        self.region_count.store(regions.len(), Ordering::Release);
    }

    /// The region's memory was destroyed: nothing is mirrored into it any more.
    pub fn remove_region(&self, start: DAddr) {
        let mut regions = self.regions.write();
        regions.remove(&start);
        self.region_count.store(regions.len(), Ordering::Release);
        drop(regions);
        let mut shadow = self.shadow.lock();
        let first_page = start..start + PAGE;
        shadow.direct.retain(|word| !first_page.contains(word));
        shadow.words.retain(|word, _| !first_page.contains(word));
    }

    fn is_region(&self, start: DAddr) -> bool {
        self.region_count.load(Ordering::Acquire) != 0 && self.regions.read().contains(&start)
    }

    /// Where a release that landed on `address` is mirrored to, when it landed in a second page.
    pub(crate) fn mirror_of(&self, address: DAddr) -> Option<DAddr> {
        let start = (address & !(PAGE - 1)).checked_sub(PAGE)?;
        self.is_region(start).then_some(address - PAGE)
    }

    fn in_first_page(&self, address: DAddr) -> bool {
        self.is_region(address & !(PAGE - 1))
    }

    /// A release that is not a mirror's landed on `address`.
    pub(crate) fn note_direct(&self, address: DAddr) {
        if !self.in_first_page(address) {
            return;
        }
        let word = address & !3;
        let mut shadow = self.shadow.lock();
        if shadow.direct.len() < MAX_DIRECT_WORDS {
            shadow.direct.insert(word);
            self.has_direct.store(true, Ordering::Release);
        }
        shadow.words.remove(&word);
    }

    /// A release that combines with its word (a reduction or a max release) took guest memory
    /// from `before` to `after`. While the word holds a mirrored value, acquires see
    /// `combine(mirrored)` instead: what the word would hold without the mirror keeping its value
    /// aside. Returns what acquires see.
    pub(crate) fn note_combined(
        &self,
        address: DAddr,
        before: u32,
        after: u32,
        combine: impl FnOnce(u32) -> u32,
    ) -> u32 {
        let mirrored = self.read(address, before);
        self.note_direct(address);
        let Some(mirrored) = mirrored else {
            return after;
        };
        let seen = combine(mirrored);
        self.set(address, seen, after);
        seen
    }

    /// Whether a mirrored release into `address` is kept aside instead of written to memory.
    pub(crate) fn diverts(&self, address: DAddr) -> bool {
        self.has_direct.load(Ordering::Acquire)
            && self.shadow.lock().direct.contains(&(address & !3))
    }

    /// Keeps `value` aside for the word at `address`, whose guest memory holds `guest` now.
    pub(crate) fn set(&self, address: DAddr, value: u32, guest: u32) {
        self.shadow
            .lock()
            .words
            .insert(address & !3, Word { value, guest });
        self.has_words.store(true, Ordering::Release);
    }

    /// The mirrored value an acquire of the word at `address` sees while guest memory holds
    /// `guest`, if any. Once the word changed in memory, memory is newer.
    pub(crate) fn read(&self, address: DAddr, guest: u32) -> Option<u32> {
        if !self.has_words.load(Ordering::Acquire) {
            return None;
        }
        let mut shadow = self.shadow.lock();
        let word = address & !3;
        let kept = *shadow.words.get(&word)?;
        if kept.guest != guest {
            shadow.words.remove(&word);
            return None;
        }
        Some(kept.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGION: DAddr = 0x8_0002_7000;
    const OTHER: DAddr = 0x8_0010_0000;

    #[test]
    fn second_page_releases_mirror_into_the_first_page() {
        let mirror = SemaphoreMirror::default();
        assert_eq!(mirror.mirror_of(REGION + 0x1010), None);
        mirror.add_region(REGION);
        assert_eq!(mirror.mirror_of(REGION + 0x1000), Some(REGION));
        assert_eq!(mirror.mirror_of(REGION + 0x1010), Some(REGION + 0x10));
        assert_eq!(mirror.mirror_of(REGION + 0x1ffc), Some(REGION + 0xffc));
        assert_eq!(mirror.mirror_of(REGION), None);
        assert_eq!(mirror.mirror_of(REGION + 0xffc), None);
        assert_eq!(mirror.mirror_of(REGION + 0x2000), None);
        assert_eq!(mirror.mirror_of(OTHER + 0x1010), None);
        mirror.add_region(OTHER);
        mirror.remove_region(REGION);
        assert_eq!(mirror.mirror_of(REGION + 0x1010), None);
        assert_eq!(mirror.mirror_of(OTHER + 0x1010), Some(OTHER + 0x10));
        assert_eq!(mirror.mirror_of(0x10), None);
    }

    #[test]
    fn mirrored_values_stay_out_of_words_with_direct_releases() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        assert!(!mirror.diverts(REGION));
        mirror.note_direct(REGION + 0x1010);
        assert!(!mirror.diverts(REGION + 0x10));
        mirror.note_direct(REGION);
        assert!(mirror.diverts(REGION));
        assert!(mirror.diverts(REGION + 2));
        assert!(!mirror.diverts(REGION + 4));

        assert_eq!(mirror.read(REGION, 7), None);
        mirror.set(REGION, 42, 7);
        assert_eq!(mirror.read(REGION, 7), Some(42));
        assert_eq!(mirror.read(REGION, 8), None);
        assert_eq!(mirror.read(REGION, 7), None);

        mirror.set(REGION, 43, 7);
        mirror.note_direct(REGION);
        assert_eq!(mirror.read(REGION, 7), None);
    }

    #[test]
    fn combined_releases_combine_with_the_mirrored_value() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        mirror.note_direct(REGION);
        let max_with = |payload: u32| move |current: u32| current.max(payload);
        mirror.set(REGION, 10, 0);
        assert_eq!(mirror.note_combined(REGION, 0, 3, max_with(3)), 10);
        assert_eq!(mirror.read(REGION, 3), Some(10));
        assert_eq!(mirror.note_combined(REGION, 3, 12, max_with(12)), 12);
        assert_eq!(mirror.read(REGION, 12), Some(12));
        assert_eq!(mirror.note_combined(REGION + 4, 5, 6, max_with(6)), 6);
        assert_eq!(mirror.read(REGION + 4, 6), None);
    }

    #[test]
    fn removing_a_region_forgets_its_words() {
        let mirror = SemaphoreMirror::default();
        mirror.add_region(REGION);
        mirror.note_direct(REGION);
        mirror.set(REGION, 44, 7);
        mirror.remove_region(REGION);
        assert!(!mirror.diverts(REGION));
        assert_eq!(mirror.read(REGION, 7), None);
    }
}
