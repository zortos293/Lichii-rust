//! Host semaphore writes the back has queued but the host GPU has not landed yet. A macro DMA
//! read of their words gets the released values in stream order from here instead of waiting for
//! the host GPU to finish the work before them: a release writes a value the stream already holds.
//!
//! The writes tracked are the host class's semaphore releases and reductions. Fence actions run
//! in the order they were signalled, so the landings counted on an address retire its writes in
//! queue order.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use lichii_common::DAddr;
use parking_lot::Mutex;

const LANDING_SLOTS: usize = 128;
const EMPTY: u64 = u64::MAX;

/// Counts the semaphore writes that land at the device addresses the back watches. Part of
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
    /// A semaphore write to `address` landed: counts it if the address is watched.
    #[inline]
    pub(crate) fn written(&self, address: DAddr) {
        if self.watched.load(Ordering::Acquire) == 0 {
            return;
        }
        for slot in &self.slots {
            if slot.daddr.load(Ordering::Acquire) == address {
                slot.landed.fetch_add(1, Ordering::Release);
                return;
            }
        }
    }

    /// Back thread only: claims a free slot to count the landings at `daddr`.
    fn watch(&self, daddr: DAddr) -> Option<usize> {
        let index = self
            .slots
            .iter()
            .position(|s| s.daddr.load(Ordering::Relaxed) == EMPTY)?;
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
    slot: usize,
    /// The landing count at which every write queued to `daddr` is in memory.
    queued: u64,
    /// While fewer writes landed, a reduction with a result the back can't know is on its way.
    unknown_until: u64,
    /// The words the releases queued here write, the last one's over the earlier ones'.
    words: [u32; 4],
    len: usize,
    /// Whether words 2 and 3 are a timestamp, taken when the host GPU gets to the release.
    timestamped: bool,
    /// Queue order of the last write, between entries whose words overlap.
    order: u64,
}

impl Entry {
    fn end(&self) -> DAddr {
        self.daddr + 4 * self.len as u64
    }
}

/// What a read finds of the host writes still on their way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pending {
    /// None of its words.
    Nothing,
    /// Releases with the values given.
    Known,
    /// A write with a value the back can't tell, or one it could not track.
    Unknown,
}

/// The host semaphore writes the back queued, by device address.
pub(crate) struct HostReleases {
    entries: [Option<Entry>; ENTRIES],
    order: u64,
    /// A write could not be tracked: no read is served until a tracked write queued after it
    /// lands, which puts the untracked one in memory first. The slot and count of that write.
    blind: Option<Option<(usize, u64)>>,
}

impl HostReleases {
    pub(crate) fn new() -> Self {
        Self {
            entries: [None; ENTRIES],
            order: 0,
            blind: None,
        }
    }

    /// A release of `payload` (`len` words) to `daddr` was queued behind the work recorded so far.
    pub(crate) fn release(
        &mut self,
        landings: &ReleaseLandings,
        daddr: DAddr,
        payload: u64,
        len: usize,
    ) {
        let value = [payload as u32, (payload >> 32) as u32, 0, 0];
        if !daddr.is_multiple_of(4) {
            self.blind = Some(None);
            return;
        }
        let Some(entry) = self.entry(landings, daddr) else {
            return;
        };
        if landings.landed(entry.slot) >= entry.queued {
            entry.len = 0;
            entry.timestamped = false;
        }
        entry.words[..len].copy_from_slice(&value[..len]);
        entry.len = entry.len.max(len);
        entry.timestamped |= len == 4;
        entry.queued += 1;
        let marker = (entry.slot, entry.queued);
        self.queued(marker);
    }

    /// A semaphore reduction at `daddr` was queued: reads of it synchronize until it lands.
    pub(crate) fn reduce(&mut self, landings: &ReleaseLandings, daddr: DAddr) {
        let Some(entry) = self.entry(landings, daddr) else {
            return;
        };
        if landings.landed(entry.slot) >= entry.queued {
            entry.len = 0;
            entry.timestamped = false;
        }
        entry.len = entry.len.max(1);
        entry.queued += 1;
        entry.unknown_until = entry.queued;
        let marker = (entry.slot, entry.queued);
        self.queued(marker);
    }

    fn queued(&mut self, marker: (usize, u64)) {
        self.order += 1;
        if let Some(None) = self.blind {
            self.blind = Some(Some(marker));
        }
    }

    /// The entry of `daddr`, made in place of a landed one when needed; None, and blind, when
    /// there is no room.
    fn entry(&mut self, landings: &ReleaseLandings, daddr: DAddr) -> Option<&mut Entry> {
        let order = self.order + 1;
        let index = match self
            .entries
            .iter()
            .position(|e| e.is_some_and(|e| e.daddr == daddr))
        {
            Some(index) => index,
            None => {
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
                let Some((free, slot)) = free.and_then(|free| Some((free, landings.watch(daddr)?)))
                else {
                    self.blind = Some(None);
                    return None;
                };
                let landed = landings.landed(slot);
                self.entries[free] = Some(Entry {
                    daddr,
                    slot,
                    queued: landed,
                    unknown_until: 0,
                    words: [0; 4],
                    len: 0,
                    timestamped: false,
                    order,
                });
                free
            }
        };
        let entry = self.entries[index].as_mut().expect("just found or made");
        entry.order = order;
        Some(entry)
    }

    /// Calls `put` with each word of `[daddr, daddr + size)` a release still on its way writes,
    /// the last queued one's where they overlap, and its value; timestamps come from `now`.
    pub(crate) fn pending(
        &mut self,
        landings: &ReleaseLandings,
        daddr: DAddr,
        size: u64,
        now: impl Fn() -> u64,
        mut put: impl FnMut(DAddr, u32),
    ) -> Pending {
        if let Some(marker) = self.blind {
            match marker {
                Some((slot, queued)) if landings.landed(slot) >= queued => self.blind = None,
                _ => return Pending::Unknown,
            }
        }
        let end = daddr.saturating_add(size);
        let on_way =
            |e: &&Entry| e.daddr < end && daddr < e.end() && landings.landed(e.slot) < e.queued;
        if self
            .entries
            .iter()
            .flatten()
            .filter(on_way)
            .any(|e| landings.landed(e.slot) < e.unknown_until)
        {
            return Pending::Unknown;
        }
        let mut found = Pending::Nothing;
        let first = daddr.next_multiple_of(4);
        for at in (first..end).step_by(4) {
            let Some(e) = self
                .entries
                .iter()
                .flatten()
                .filter(on_way)
                .filter(|e| (e.daddr..e.end()).contains(&at))
                .max_by_key(|e| e.order)
            else {
                continue;
            };
            found = Pending::Known;
            let i = ((at - e.daddr) / 4) as usize;
            let word = match i {
                2 | 3 if e.timestamped => {
                    let timestamp = now();
                    if i == 2 {
                        timestamp as u32
                    } else {
                        (timestamp >> 32) as u32
                    }
                }
                _ => e.words[i],
            };
            put(at, word);
        }
        found
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

    fn read(
        releases: &mut HostReleases,
        landings: &ReleaseLandings,
        at: DAddr,
        size: u64,
    ) -> (Pending, Vec<(DAddr, u32)>) {
        let mut words = Vec::new();
        let found = releases.pending(
            landings,
            at,
            size,
            || 0x7_1234_5678,
            |at, w| words.push((at, w)),
        );
        (found, words)
    }

    #[test]
    fn a_release_is_on_its_way_until_its_write_lands() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 0, 4);
        assert_eq!(
            read(&mut releases, &landings, A, 24),
            (
                Pending::Known,
                vec![(A, 0), (A + 4, 0), (A + 8, 0x1234_5678), (A + 12, 7)]
            )
        );
        landings.written(A);
        assert_eq!(
            read(&mut releases, &landings, A, 24),
            (Pending::Nothing, vec![])
        );
    }

    #[test]
    fn an_earlier_landing_leaves_the_later_release_on_its_way() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 1, 1);
        releases.release(&landings, A, 2, 1);
        landings.written(A);
        assert_eq!(
            read(&mut releases, &landings, A, 4),
            (Pending::Known, vec![(A, 2)])
        );
        landings.written(A);
        assert_eq!(read(&mut releases, &landings, A, 4).0, Pending::Nothing);
    }

    #[test]
    fn a_narrower_release_keeps_the_pending_words_of_a_wider_one() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 0x1_0000_0001, 2);
        releases.release(&landings, A, 2, 1);
        assert_eq!(
            read(&mut releases, &landings, A, 8),
            (Pending::Known, vec![(A, 2), (A + 4, 1)])
        );
        landings.written(A);
        landings.written(A);
        releases.release(&landings, A, 3, 1);
        assert_eq!(
            read(&mut releases, &landings, A, 8),
            (Pending::Known, vec![(A, 3)])
        );
    }

    #[test]
    fn a_reduction_on_its_way_makes_the_words_unknown_until_it_lands() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A, 1, 1);
        releases.reduce(&landings, A);
        releases.release(&landings, A, 2, 1);
        releases.release(&landings, A + 8, 3, 1);
        assert_eq!(read(&mut releases, &landings, A, 12).0, Pending::Unknown);
        landings.written(A);
        landings.written(A);
        assert_eq!(
            read(&mut releases, &landings, A, 12),
            (Pending::Known, vec![(A, 2), (A + 8, 3)])
        );
    }

    #[test]
    fn overlapping_releases_to_other_addresses_give_the_last_queued_value() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        releases.release(&landings, A + 8, 5, 1);
        releases.release(&landings, A, 1, 4);
        assert_eq!(
            read(&mut releases, &landings, A + 8, 4).1,
            vec![(A + 8, 0x1234_5678)]
        );
        releases.release(&landings, A + 8, 6, 1);
        assert_eq!(read(&mut releases, &landings, A + 8, 4).1, vec![(A + 8, 6)]);
    }

    #[test]
    fn an_untracked_write_blinds_reads_until_a_later_tracked_one_lands() {
        let landings = ReleaseLandings::default();
        let mut releases = HostReleases::new();
        for i in 0..ENTRIES as u64 {
            releases.release(&landings, A + 0x100 * i, 1, 1);
        }
        releases.release(&landings, A + 0x100 * ENTRIES as u64, 1, 1);
        assert_eq!(
            read(&mut releases, &landings, A + 0x10_0000, 4).0,
            Pending::Unknown
        );
        landings.written(A);
        releases.release(&landings, A + 0x20, 2, 1);
        assert_eq!(
            read(&mut releases, &landings, A + 0x10_0000, 4).0,
            Pending::Unknown
        );
        landings.written(A + 0x20);
        assert_eq!(
            read(&mut releases, &landings, A + 0x10_0000, 4).0,
            Pending::Nothing
        );
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
