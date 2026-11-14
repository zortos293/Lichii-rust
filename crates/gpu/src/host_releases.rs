//! Host semaphore releases the back has queued but the host GPU has not landed yet. A macro DMA
//! read of their words gets their values in stream order from here instead of waiting for the
//! host GPU to finish the work before them: a release writes a value the stream already holds.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use lichii_common::DAddr;
use parking_lot::Mutex;

const LANDING_SLOTS: usize = 128;
const EMPTY: u64 = u64::MAX;

/// Counts the semaphore writes that land on the device addresses the back watches. Part of
/// [`SemaphoreWaiters`](crate::channel::SemaphoreWaiters), which every
/// [`FenceAction`](crate::backend::FenceAction) semaphore write reaches once it is in memory.
pub(crate) struct ReleaseLandings {
    watched: AtomicUsize,
    slots: [LandingSlot; LANDING_SLOTS],
}

struct LandingSlot {
    daddr: AtomicU64,
    landed: AtomicU64,
}

impl Default for ReleaseLandings {
    fn default() -> Self {
        Self {
            watched: AtomicUsize::new(0),
            slots: std::array::from_fn(|_| LandingSlot {
                daddr: AtomicU64::new(EMPTY),
                landed: AtomicU64::new(0),
            }),
        }
    }
}

impl ReleaseLandings {
    /// `[address, address + size)` was written: counts a landing on every watched address in it.
    #[inline]
    pub(crate) fn written(&self, address: DAddr, size: u64) {
        if self.watched.load(Ordering::Acquire) == 0 {
            return;
        }
        let end = address.saturating_add(size);
        for slot in &self.slots {
            let daddr = slot.daddr.load(Ordering::Acquire);
            if daddr != EMPTY && (address..end).contains(&daddr) {
                slot.landed.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// Back thread only: the slot counting landings on `daddr`, claiming a free one.
    fn watch(&self, daddr: DAddr) -> Option<usize> {
        let mut free = None;
        for (index, slot) in self.slots.iter().enumerate() {
            match slot.daddr.load(Ordering::Relaxed) {
                d if d == daddr => return Some(index),
                EMPTY if free.is_none() => free = Some(index),
                _ => {}
            }
        }
        let index = free?;
        self.slots[index].daddr.store(daddr, Ordering::Release);
        self.watched.fetch_add(1, Ordering::Release);
        Some(index)
    }

    fn unwatch(&self, index: usize) {
        self.slots[index].daddr.store(EMPTY, Ordering::Release);
        self.watched.fetch_sub(1, Ordering::Release);
    }

    fn landed(&self, index: usize) -> u64 {
        self.slots[index].landed.load(Ordering::Acquire)
    }
}

const ENTRIES: usize = 64;

#[derive(Clone, Copy)]
struct Entry {
    daddr: DAddr,
    /// The words the releases queued here write, the last one's over the earlier ones'.
    words: [u32; 4],
    len: usize,
    /// Whether words 2 and 3 are a timestamp, taken when the host GPU gets to the release.
    timestamped: bool,
    slot: usize,
    /// The landing count at which every release queued here is in memory.
    queued: u64,
}

/// The host releases the back queued, by device address.
pub(crate) struct HostReleases {
    entries: [Option<Entry>; ENTRIES],
    /// Landing slots of dropped entries whose releases were still on their way: freed once they
    /// land, so no late landing counts for the next address a slot watches.
    draining: [Option<(usize, u64)>; ENTRIES],
}

impl HostReleases {
    pub(crate) fn new() -> Self {
        Self {
            entries: [None; ENTRIES],
            draining: [None; ENTRIES],
        }
    }

    /// A release of `payload` to `daddr` was queued behind the work recorded so far.
    pub(crate) fn release(
        &mut self,
        landings: &ReleaseLandings,
        daddr: DAddr,
        payload: u64,
        len: usize,
    ) {
        self.reap(landings);
        let end = daddr + 4 * len as u64;
        let index = self
            .entries
            .iter()
            .position(|e| e.is_some_and(|e| e.daddr == daddr));
        let overlapped = self.entries.iter().enumerate().any(|(i, e)| {
            Some(i) != index
                && e.is_some_and(|e| e.daddr < end && daddr < e.daddr + 4 * e.len as u64)
        });
        if !daddr.is_multiple_of(4) || overlapped {
            self.forget(landings, daddr, 4 * len as u64);
            return;
        }
        let value = [payload as u32, (payload >> 32) as u32, 0, 0];
        if let Some(entry) = index.and_then(|i| self.entries[i].as_mut()) {
            if landings.landed(entry.slot) >= entry.queued {
                entry.len = 0;
                entry.timestamped = false;
                entry.queued = landings.landed(entry.slot);
            }
            entry.words[..len].copy_from_slice(&value[..len]);
            entry.len = entry.len.max(len);
            entry.timestamped |= len == 4;
            entry.queued += 1;
            return;
        }
        let free = self.entries.iter().position(Option::is_none).or_else(|| {
            let landed = self
                .entries
                .iter()
                .position(|e| e.is_some_and(|e| landings.landed(e.slot) >= e.queued))?;
            if let Some(e) = self.entries[landed].take() {
                landings.unwatch(e.slot);
            }
            Some(landed)
        });
        let (Some(free), Some(slot)) = (free, landings.watch(daddr)) else {
            return;
        };
        let mut words = [0; 4];
        words[..len].copy_from_slice(&value[..len]);
        self.entries[free] = Some(Entry {
            daddr,
            words,
            len,
            timestamped: len == 4,
            slot,
            queued: landings.landed(slot) + 1,
        });
    }

    /// Queued work writes `[daddr, daddr + size)` with values the back does not track: reads
    /// there synchronize with the host GPU again.
    pub(crate) fn forget(&mut self, landings: &ReleaseLandings, daddr: DAddr, size: u64) {
        let end = daddr.saturating_add(size);
        for index in 0..ENTRIES {
            let Some(e) = self.entries[index] else {
                continue;
            };
            if e.daddr >= end || daddr >= e.daddr + 4 * e.len as u64 {
                continue;
            }
            self.entries[index] = None;
            if landings.landed(e.slot) >= e.queued {
                landings.unwatch(e.slot);
            } else if let Some(free) = self.draining.iter_mut().find(|d| d.is_none()) {
                *free = Some((e.slot, e.queued));
            }
        }
    }

    fn reap(&mut self, landings: &ReleaseLandings) {
        for draining in &mut self.draining {
            if let Some((slot, queued)) = *draining
                && landings.landed(slot) >= queued
            {
                landings.unwatch(slot);
                *draining = None;
            }
        }
    }

    /// Calls `put` with each word of `[daddr, daddr + size)` a release that has not landed yet
    /// writes, and its value; timestamps come from `now`. Returns whether there was one.
    pub(crate) fn pending(
        &self,
        landings: &ReleaseLandings,
        daddr: DAddr,
        size: u64,
        now: impl Fn() -> u64,
        mut put: impl FnMut(DAddr, u32),
    ) -> bool {
        let end = daddr.saturating_add(size);
        let mut any = false;
        for e in self.entries.iter().flatten() {
            if e.daddr >= end
                || daddr >= e.daddr + 4 * e.len as u64
                || landings.landed(e.slot) >= e.queued
            {
                continue;
            }
            any = true;
            let timestamp = if e.timestamped { now() } else { 0 };
            for (i, &word) in e.words[..e.len].iter().enumerate() {
                let at = e.daddr + 4 * i as u64;
                if !(daddr..end).contains(&at) {
                    continue;
                }
                let word = match i {
                    2 if e.timestamped => timestamp as u32,
                    3 if e.timestamped => (timestamp >> 32) as u32,
                    _ => word,
                };
                put(at, word);
            }
        }
        any
    }
}

const OVERLAY_WORDS: usize = 16;

/// The words of a synchronized read that pending host releases write, handed from the back to
/// the front with the read: the front reads guest memory and puts these over it.
#[derive(Default)]
pub(crate) struct ReadOverlay(Mutex<Overlay>);

#[derive(Default)]
struct Overlay {
    words: [(u32, u32); OVERLAY_WORDS],
    len: usize,
}

impl ReadOverlay {
    /// Back: sets the overlay of the read being synchronized; false when it holds too many words.
    pub(crate) fn set(&self, words: impl IntoIterator<Item = (u32, u32)>) -> bool {
        let mut overlay = self.0.lock();
        overlay.len = 0;
        for word in words {
            if overlay.len == OVERLAY_WORDS {
                overlay.len = 0;
                return false;
            }
            let len = overlay.len;
            overlay.words[len] = word;
            overlay.len += 1;
        }
        true
    }

    /// Back: the read being synchronized takes guest memory as it is.
    pub(crate) fn clear(&self) {
        self.0.lock().len = 0;
    }

    /// Front: puts the overlay over `out`, the words just read, and clears it.
    pub(crate) fn apply(&self, out: &mut [u32]) {
        let mut overlay = self.0.lock();
        let len = std::mem::take(&mut overlay.len);
        for &(index, word) in &overlay.words[..len] {
            if let Some(slot) = out.get_mut(index as usize) {
                *slot = word;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: DAddr = 0x1_0000;

    fn pending(
        releases: &HostReleases,
        landings: &ReleaseLandings,
        size: u64,
    ) -> Option<Vec<(DAddr, u32)>> {
        let mut words = Vec::new();
        releases
            .pending(
                landings,
                A,
                size,
                || 0x7_1234_5678,
                |at, w| words.push((at, w)),
            )
            .then_some(words)
    }

    #[test]
    fn a_release_is_pending_until_its_write_lands() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 0, 4);
        assert_eq!(
            pending(&releases, &landings, 24),
            Some(vec![(A, 0), (A + 4, 0), (A + 8, 0x1234_5678), (A + 12, 7)])
        );
        landings.written(A, 16);
        assert_eq!(pending(&releases, &landings, 24), None);
    }

    #[test]
    fn an_earlier_landing_leaves_the_later_release_pending() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 1, 1);
        releases.release(&landings, A, 2, 1);
        landings.written(A, 4);
        assert_eq!(pending(&releases, &landings, 4), Some(vec![(A, 2)]));
        landings.written(A, 4);
        assert_eq!(pending(&releases, &landings, 4), None);
    }

    #[test]
    fn a_narrower_release_keeps_the_pending_words_of_a_wider_one() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 0x1_0000_0001, 2);
        releases.release(&landings, A, 2, 1);
        assert_eq!(
            pending(&releases, &landings, 8),
            Some(vec![(A, 2), (A + 4, 1)])
        );
        landings.written(A, 8);
        landings.written(A, 4);
        landings.written(A, 4);
        releases.release(&landings, A, 3, 1);
        assert_eq!(pending(&releases, &landings, 8), Some(vec![(A, 3)]));
    }

    #[test]
    fn overlapping_and_forgotten_releases_are_not_served() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 1, 4);
        releases.release(&landings, A + 8, 2, 1);
        assert_eq!(pending(&releases, &landings, 16), None);
        releases.release(&landings, A + 0x20, 3, 1);
        releases.forget(&landings, A + 0x20, 8);
        let mut any = false;
        releases.pending(&landings, A + 0x20, 4, || 0, |_, _| any = true);
        assert!(!any);
    }

    #[test]
    fn landing_slots_of_dropped_releases_free_once_they_land() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 1, 1);
        releases.forget(&landings, A, 4);
        assert_eq!(landings.watched.load(Ordering::Relaxed), 1);
        landings.written(A, 4);
        releases.release(&landings, A + 0x40, 1, 1);
        assert_eq!(landings.watched.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_overlay_lands_on_the_words_it_names_once() {
        let overlay = ReadOverlay::default();
        assert!(overlay.set([(0, 5), (3, 7)]));
        let mut out = [1u32; 6];
        overlay.apply(&mut out);
        assert_eq!(out, [5, 1, 1, 7, 1, 1]);
        let mut again = [1u32; 6];
        overlay.apply(&mut again);
        assert_eq!(again, [1; 6]);
    }
}
