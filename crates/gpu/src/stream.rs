//! The stream the front sends the back: method writes with the data of multi-word methods
//! copied in, the host actions that must run in order with them, and the CPU-side commands
//! forwarded among them.
//!
//! A record's first slot is its header: tag (bits 0..8), engine (8..16), method (16..32) and
//! argument (32..64). Records with more data carry it in the slots after the header.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;

use lichii_common::GpuVAddr;

use crate::back::{Back, SideCommand};
use crate::backend::{ChannelId, ReleaseSize};
use crate::engines::EngineKind;
use crate::gpu::Inbox;
use crate::host::Reduction;
use crate::ring::{Consumer, Producer};

pub(crate) mod tag {
    pub const METHOD: u8 = 1;
    /// Argument: word count; the words follow, two per slot.
    pub const MULTI: u8 = 2;
    /// Argument: the channel id the following records belong to.
    pub const CHANNEL: u8 = 3;
    /// Argument: the bound class.
    pub const BIND: u8 = 4;
    /// Method: release size code; then the address and the payload.
    pub const RELEASE: u8 = 5;
    /// Method: reduction code, bit 4 signed; argument: payload; then the address.
    pub const REDUCE: u8 = 6;
    /// Argument: the syncpoint id.
    pub const SYNCPOINT_INCREMENT: u8 = 7;
    pub const WAIT_FOR_IDLE: u8 = 8;
    pub const INVALIDATE_CACHES: u8 = 9;
    pub const SET_REFERENCE: u8 = 10;
    pub const RELEASE_FENCES: u8 = 11;
    pub const FLUSH_COMMANDS: u8 = 12;
    /// The next [`SideCommand`](crate::back::SideCommand) in the side queue.
    pub const SIDE: u8 = 13;
    /// Then the GPU address and the size of a range the front reads.
    pub const SYNCHRONIZE_READ: u8 = 14;
    /// Then the GPU address and the size of the GPFIFO entry a channel fetches next.
    pub const SYNCHRONIZE_FETCH: u8 = 15;
}

/// Words of one multi-word record; longer runs are split.
pub(crate) const MULTI_MAX_WORDS: usize = 4096;
/// Smallest ring that holds the longest record.
pub(crate) const MIN_RING_SLOTS: usize = 2 * (1 + MULTI_MAX_WORDS / 2);
/// The front publishes at least this often while it writes.
const PUBLISH_SLOTS: u64 = 512;

pub(crate) const REDUCTIONS: [Reduction; 8] = [
    Reduction::Min,
    Reduction::Max,
    Reduction::Xor,
    Reduction::And,
    Reduction::Or,
    Reduction::Add,
    Reduction::Inc,
    Reduction::Dec,
];
pub(crate) const RELEASE_SIZES: [ReleaseSize; 3] =
    [ReleaseSize::Four, ReleaseSize::Eight, ReleaseSize::Sixteen];

#[inline]
pub(crate) const fn header(tag: u8, engine: u8, method: u32, argument: u32) -> u64 {
    tag as u64 | (engine as u64) << 8 | ((method & 0xffff) as u64) << 16 | (argument as u64) << 32
}

/// How many of the fetches [`Stream::synchronize_fetch`] sent the back has synchronized.
pub(crate) struct FetchSyncs {
    completed: AtomicU64,
    front: Arc<Inbox>,
}

impl FetchSyncs {
    pub(crate) fn new(front: Arc<Inbox>) -> Self {
        Self {
            completed: AtomicU64::new(0),
            front,
        }
    }

    /// The back synchronized the next fetch: counts it and wakes the front.
    pub(crate) fn complete(&self) {
        self.completed.fetch_add(1, Ordering::Release);
        self.front.wake();
    }
}

/// Where the back runs.
pub(crate) enum Link {
    /// On its own thread, reading the ring concurrently.
    Thread,
    /// On the front's thread: it runs whenever the front waits for it.
    Inline(Box<(Back, Consumer)>),
}

/// The front's end of the stream.
pub(crate) struct Stream {
    producer: Producer,
    link: Link,
    side: mpsc::Sender<SideCommand>,
    /// The engine and method of the words in `words`, not yet written.
    pending: Option<(EngineKind, u32)>,
    words: Vec<u32>,
    channel: Option<ChannelId>,
    fetch_syncs: Arc<FetchSyncs>,
    /// Fetches sent to [`FetchSyncs`] so far.
    fetch_syncs_sent: u64,
}

impl Stream {
    pub(crate) fn new(
        producer: Producer,
        link: Link,
        side: mpsc::Sender<SideCommand>,
        fetch_syncs: Arc<FetchSyncs>,
    ) -> Self {
        Self {
            producer,
            link,
            side,
            pending: None,
            words: Vec::with_capacity(MULTI_MAX_WORDS),
            channel: None,
            fetch_syncs,
            fetch_syncs_sent: 0,
        }
    }

    /// An engine method write. Consecutive words to the same method become one multi-word
    /// record.
    #[inline]
    pub(crate) fn engine_word(&mut self, engine: EngineKind, method: u32, argument: u32) {
        if self.pending != Some((engine, method)) || self.words.len() == MULTI_MAX_WORDS {
            self.flush_pending();
            self.pending = Some((engine, method));
        }
        self.words.push(argument);
    }

    /// Writes the pending engine words.
    #[inline]
    pub(crate) fn flush_pending(&mut self) {
        let Some((engine, method)) = self.pending.take() else {
            return;
        };
        let engine = engine.index() as u8;
        if let [argument] = self.words[..] {
            self.reserve(1);
            self.producer
                .push(header(tag::METHOD, engine, method, argument));
        } else {
            let count = self.words.len();
            self.reserve(1 + count.div_ceil(2));
            self.producer
                .push(header(tag::MULTI, engine, method, count as u32));
            for pair in self.words.chunks(2) {
                let high = pair.get(1).copied().unwrap_or(0);
                self.producer
                    .push(u64::from(pair[0]) | u64::from(high) << 32);
            }
        }
        self.words.clear();
        self.maybe_publish();
    }

    /// Makes the following records belong to `channel`.
    pub(crate) fn select(&mut self, channel: ChannelId) {
        if self.channel != Some(channel) {
            self.channel = Some(channel);
            self.record(&[header(tag::CHANNEL, 0, 0, channel.0)]);
        }
    }

    /// Forgets the selected channel (it was destroyed).
    pub(crate) fn deselect(&mut self, channel: ChannelId) {
        if self.channel == Some(channel) {
            self.channel = None;
        }
    }

    pub(crate) fn bind(&mut self, engine: EngineKind, class: u32) {
        self.record(&[header(tag::BIND, engine.index() as u8, 0, class)]);
    }

    pub(crate) fn release(&mut self, address: GpuVAddr, payload: u64, size: ReleaseSize) {
        let code = RELEASE_SIZES.iter().position(|&s| s == size).unwrap_or(0) as u32;
        self.record(&[header(tag::RELEASE, 0, code, 0), address, payload]);
    }

    pub(crate) fn reduce(&mut self, address: GpuVAddr, payload: u32, op: Reduction, signed: bool) {
        let code = REDUCTIONS.iter().position(|&r| r == op).unwrap_or(0) as u32;
        self.record(&[
            header(tag::REDUCE, 0, code | u32::from(signed) << 4, payload),
            address,
        ]);
    }

    pub(crate) fn syncpoint_increment(&mut self, id: u32) {
        self.record(&[header(tag::SYNCPOINT_INCREMENT, 0, 0, id)]);
    }

    pub(crate) fn wait_for_idle(&mut self) {
        self.record(&[header(tag::WAIT_FOR_IDLE, 0, 0, 0)]);
    }

    pub(crate) fn invalidate_caches(&mut self) {
        self.record(&[header(tag::INVALIDATE_CACHES, 0, 0, 0)]);
    }

    pub(crate) fn set_reference(&mut self, value: u32) {
        self.record(&[header(tag::SET_REFERENCE, 0, 0, value)]);
    }

    pub(crate) fn release_fences(&mut self) {
        self.record(&[header(tag::RELEASE_FENCES, 0, 0, 0)]);
    }

    pub(crate) fn flush_commands(&mut self) {
        self.record(&[header(tag::FLUSH_COMMANDS, 0, 0, 0)]);
    }

    /// Forwards a CPU-side command, in order with everything sent before it.
    pub(crate) fn side(&mut self, command: SideCommand) {
        self.flush_pending();
        if self.side.send(command).is_err() {
            tracing::error!("GPU back side queue closed");
            return;
        }
        self.record(&[header(tag::SIDE, 0, 0, 0)]);
    }

    /// Waits until the back executed everything sent so far: a sync point.
    pub(crate) fn drain(&mut self) {
        self.flush_pending();
        self.producer.publish();
        match &mut self.link {
            Link::Thread => self.producer.wait_consumed(self.producer.position()),
            Link::Inline(back) => {
                let (back, consumer) = &mut **back;
                back.run(consumer);
            }
        }
    }

    /// Waits until the back executed everything sent so far, the GPU work before completed and
    /// GPU-written data in `[address, address + size)` is in guest memory: a sync point for a
    /// front read of memory the GPU may write.
    pub(crate) fn synchronize_read(&mut self, address: GpuVAddr, size: u64) {
        self.record(&[header(tag::SYNCHRONIZE_READ, 0, 0, 0), address, size]);
        self.drain();
    }

    /// Has the back synchronize the fetch of `[address, address + size)` as
    /// [`Stream::synchronize_read`] does, without waiting for it: the GPU work before completes,
    /// its fences land and GPU-written data in the range reaches guest memory. Returns the ticket
    /// [`Stream::fetch_synced`] takes.
    pub(crate) fn synchronize_fetch(&mut self, address: GpuVAddr, size: u64) -> u64 {
        self.record(&[header(tag::SYNCHRONIZE_FETCH, 0, 0, 0), address, size]);
        self.producer.publish();
        self.fetch_syncs_sent += 1;
        self.fetch_syncs_sent
    }

    /// Whether the back synchronized the fetch `ticket` names.
    pub(crate) fn fetch_synced(&self, ticket: u64) -> bool {
        self.fetch_syncs.completed.load(Ordering::Acquire) >= ticket
    }

    /// The front has nothing to do: hand the back everything, and run an inline back now.
    pub(crate) fn idle(&mut self) {
        self.flush_pending();
        self.producer.publish();
        if let Link::Inline(back) = &mut self.link {
            let (back, consumer) = &mut **back;
            back.run(consumer);
        }
    }

    /// Hands the back everything and ends the stream.
    pub(crate) fn close(&mut self) {
        self.idle();
        self.producer.close();
    }

    fn record(&mut self, slots: &[u64]) {
        self.flush_pending();
        self.reserve(slots.len());
        for &slot in slots {
            self.producer.push(slot);
        }
        self.maybe_publish();
    }

    #[inline]
    fn maybe_publish(&mut self) {
        if self.producer.unpublished() >= PUBLISH_SLOTS {
            self.producer.publish();
        }
    }

    #[inline]
    fn reserve(&mut self, len: usize) {
        while !self.producer.try_reserve(len) {
            self.producer.publish();
            match &mut self.link {
                Link::Thread => self.producer.wait_for_room(),
                Link::Inline(back) => {
                    let (back, consumer) = &mut **back;
                    back.run(consumer);
                }
            }
        }
    }
}
