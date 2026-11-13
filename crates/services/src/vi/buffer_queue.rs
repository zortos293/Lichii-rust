//! A layer's buffer queue: the producer end the title drives with binder transactions
//! (IGraphicBufferProducer) and the consumer end the compositor takes frames from.
//!
//! Titles preallocate their buffers (nvmap handles) with SetPreallocatedBuffer, then dequeue,
//! render and queue them. A DequeueBuffer that finds no free slot is answered once the
//! compositor releases one, so the title runs at the pace its frames are shown.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use lichii_common::perf::PerfStats;
use lichii_kernel::Event;
use lichii_kernel::ipc::{DeferredRequest, HleRequestContext};
use parking_lot::{Mutex, MutexGuard};

use super::Wakeup;
use super::parcel::{ParcelReader, ParcelWriter};
use crate::nvdrv::{NvFence, Nvdrv};

const SLOTS: usize = 64;
const DEFAULT_MAX_BUFFERS: usize = 2;
pub(super) const REFRESH_HZ: u32 = 60;

const REQUEST_BUFFER: u32 = 1;
const SET_BUFFER_COUNT: u32 = 2;
const DEQUEUE_BUFFER: u32 = 3;
const DETACH_BUFFER: u32 = 4;
const QUEUE_BUFFER: u32 = 7;
const CANCEL_BUFFER: u32 = 8;
const QUERY: u32 = 9;
const CONNECT: u32 = 10;
const DISCONNECT: u32 = 11;
const SET_PREALLOCATED_BUFFER: u32 = 14;
const GET_BUFFER_HISTORY: u32 = 17;

const NO_ERROR: i32 = 0;
const BUFFER_NEEDS_REALLOCATION: i32 = 1;
const NO_INIT: i32 = -19;
const BAD_VALUE: i32 = -22;

/// Up to four syncpoint thresholds.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub(super) struct Fence {
    pub count: u32,
    pub fences: [NvFence; 4],
}

impl Fence {
    pub const NONE: Self = Self {
        count: 0,
        fences: [NvFence { id: -1, value: 0 }; 4],
    };

    pub fn active(&self) -> &[NvFence] {
        &self.fences[..(self.count as usize).min(4)]
    }
}

/// `NvGraphicBuffer`: a buffer the title allocated, by nvmap handle.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub(super) struct GraphicBuffer {
    _magic: u32,
    pub width: i32,
    pub height: i32,
    /// Row length in pixels.
    pub stride: i32,
    pub format: u32,
    _usage: i32,
    _pad0: u32,
    _index: u32,
    _pad1: [u32; 3],
    /// The nvmap handle id of the memory.
    pub buffer_id: u32,
    _pad2: [u32; 6],
    pub external_format: u32,
    _pad3: [u32; 10],
    _handle: u32,
    /// Byte offset of the image in the nvmap buffer.
    pub offset: u32,
    _pad4: [u32; 60],
}

const _: () = assert!(std::mem::size_of::<GraphicBuffer>() == 0x16c);

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct QueueBufferInput {
    _timestamp: [u32; 2],
    _auto_timestamp: i32,
    crop: [i32; 4],
    scaling_mode: i32,
    transform: u32,
    _sticky_transform: u32,
    _async: i32,
    swap_interval: i32,
    fence: Fence,
}

const _: () = assert!(std::mem::size_of::<QueueBufferInput>() == 84);

/// A queued frame.
#[derive(Clone, Copy)]
pub(super) struct Item {
    slot: usize,
    pub buffer: GraphicBuffer,
    /// left, top, right, bottom; empty for the whole buffer.
    pub crop: [i32; 4],
    pub transform: u32,
    /// Vblanks the frame stays on screen: 1 to 4.
    pub swap_interval: u32,
    pub fence: Fence,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotState {
    Free,
    Dequeued,
    Queued,
    Acquired,
}

#[derive(Clone, Copy)]
struct Slot {
    state: SlotState,
    buffer: Option<GraphicBuffer>,
    /// The buffer's nvmap handle is pinned on the queue's behalf.
    pinned: bool,
    fence: Fence,
    frame: u64,
}

impl Slot {
    const EMPTY: Self = Self {
        state: SlotState::Free,
        buffer: None,
        pinned: false,
        fence: Fence::NONE,
        frame: 0,
    };
}

struct State {
    slots: [Slot; SLOTS],
    queue: VecDeque<Item>,
    connected: bool,
    /// Set by SetBufferCount or by the preallocated buffers; 0 for the default.
    max_buffers: usize,
    default_size: (u32, u32),
    default_format: u32,
    frames: u64,
    /// The compositor frame before which no new item is taken: the frame shown last stays for
    /// its swap interval.
    hold_until: u64,
    /// The compositor took a frame since the producer connected.
    shown: bool,
    /// DequeueBuffer requests waiting for a free slot.
    waiting: Vec<DeferredRequest>,
}

impl State {
    fn max_buffers(&self) -> usize {
        match self.max_buffers {
            0 => DEFAULT_MAX_BUFFERS,
            count => count.min(SLOTS),
        }
    }

    fn slot(&mut self, slot: i32) -> Option<&mut Slot> {
        let max = self.max_buffers();
        usize::try_from(slot)
            .ok()
            .filter(|&slot| slot < max)
            .map(|slot| &mut self.slots[slot])
    }

    fn output(&self) -> [u32; 4] {
        let (width, height) = self.default_size;
        [width, height, 0, self.queue.len() as u32]
    }
}

pub(super) struct BufferQueue {
    state: Mutex<State>,
    /// Signalled whenever a slot becomes free (GetNativeHandle).
    released: Arc<Event>,
    nvdrv: Arc<Nvdrv>,
    perf: Arc<PerfStats>,
    queued: Arc<Wakeup>,
}

impl BufferQueue {
    pub fn new(nvdrv: Arc<Nvdrv>, perf: Arc<PerfStats>, queued: Arc<Wakeup>) -> Self {
        Self {
            state: Mutex::new(State {
                slots: [Slot::EMPTY; SLOTS],
                queue: VecDeque::new(),
                connected: false,
                max_buffers: 0,
                default_size: (1, 1),
                default_format: 1,
                frames: 0,
                hold_until: 0,
                shown: false,
                waiting: Vec::new(),
            }),
            released: Arc::new(Event::new()),
            nvdrv,
            perf,
            queued,
        }
    }

    /// The event GetNativeHandle returns: a buffer was released.
    pub fn native_handle(&self) -> &Arc<Event> {
        &self.released
    }

    /// Runs transaction `code` on parcel `input` and returns the reply parcel, or `None` when
    /// the request was deferred until a buffer is free.
    pub fn transact(
        &self,
        code: u32,
        input: &[u8],
        ctx: &mut HleRequestContext,
    ) -> Option<Vec<u8>> {
        let mut parcel = ParcelReader::new(input);
        let mut out = ParcelWriter::default();
        let mut state = self.state.lock();
        let status = match code {
            REQUEST_BUFFER => {
                let buffer = state.slot(parcel.read()).map(|slot| slot.buffer);
                out.write_object(buffer.flatten().as_ref());
                if buffer.is_some() {
                    NO_ERROR
                } else {
                    BAD_VALUE
                }
            }
            SET_BUFFER_COUNT => {
                let count: i32 = parcel.read();
                let dequeued = state.slots.iter().any(|s| s.state == SlotState::Dequeued);
                match usize::try_from(count) {
                    Ok(count) if count <= SLOTS && !dequeued => {
                        state.max_buffers = count;
                        self.wake_producers(state);
                        return Some(reply(out, NO_ERROR));
                    }
                    _ => BAD_VALUE,
                }
            }
            DEQUEUE_BUFFER => {
                let _async = parcel.read_bool();
                let (width, height): (u32, u32) = (parcel.read(), parcel.read());
                if (width == 0) != (height == 0) {
                    BAD_VALUE
                } else if !state.connected {
                    NO_INIT
                } else {
                    let max = state.max_buffers();
                    let free = (0..max)
                        .filter(|&slot| state.slots[slot].state == SlotState::Free)
                        .min_by_key(|&slot| state.slots[slot].frame);
                    let Some(slot) = free else {
                        state.waiting.push(ctx.defer());
                        return None;
                    };
                    let entry = &mut state.slots[slot];
                    entry.state = SlotState::Dequeued;
                    let fence = std::mem::replace(&mut entry.fence, Fence::NONE);
                    out.write(slot as i32);
                    out.write_object(Some(&fence));
                    if entry.buffer.is_some() {
                        NO_ERROR
                    } else {
                        BUFFER_NEEDS_REALLOCATION
                    }
                }
            }
            DETACH_BUFFER => match state.slot(parcel.read()) {
                Some(slot) if slot.state == SlotState::Dequeued => {
                    let freed = std::mem::replace(slot, Slot::EMPTY);
                    self.unpin(&freed);
                    self.wake_producers(state);
                    return Some(reply(out, NO_ERROR));
                }
                _ => BAD_VALUE,
            },
            QUEUE_BUFFER => {
                let slot: i32 = parcel.read();
                let input: QueueBufferInput = parcel.read_flattened();
                let status = self.queue(&mut state, slot, &input);
                out.write(state.output());
                if status == NO_ERROR {
                    drop(state);
                    let interval = swap_interval(input.swap_interval);
                    self.perf
                        .record_guest_frame(Instant::now(), REFRESH_HZ as f32 / interval as f32);
                    self.queued.notify();
                    return Some(reply(out, NO_ERROR));
                }
                status
            }
            CANCEL_BUFFER => {
                let slot: i32 = parcel.read();
                let fence: Fence = parcel.read_flattened();
                if let Some(slot) = state.slot(slot)
                    && slot.state == SlotState::Dequeued
                {
                    slot.state = SlotState::Free;
                    slot.fence = fence;
                    self.wake_producers(state);
                    return Some(reply(out, NO_ERROR));
                }
                NO_ERROR
            }
            QUERY => {
                let value = match parcel.read::<i32>() {
                    0 => Some(state.default_size.0),
                    1 => Some(state.default_size.1),
                    2 => Some(state.default_format),
                    3 | 10 | 11 => Some(0),
                    9 => Some(u32::from(state.queue.len() > 1)),
                    12 => Some(state.max_buffers() as u32),
                    what => {
                        tracing::warn!("vi: binder query {what} is not implemented");
                        None
                    }
                };
                out.write(value.unwrap_or(0));
                if value.is_some() { NO_ERROR } else { BAD_VALUE }
            }
            CONNECT => {
                let _listener = parcel.read_bool();
                let api: i32 = parcel.read();
                let status = if state.connected || !(1..=4).contains(&api) {
                    BAD_VALUE
                } else {
                    state.connected = true;
                    NO_ERROR
                };
                out.write(state.output());
                status
            }
            DISCONNECT => {
                if !state.connected {
                    NO_INIT
                } else {
                    state.connected = false;
                    state.shown = false;
                    state.queue.clear();
                    let freed = std::mem::replace(&mut state.slots, [Slot::EMPTY; SLOTS]);
                    freed.iter().for_each(|slot| self.unpin(slot));
                    self.wake_producers(state);
                    return Some(reply(out, NO_ERROR));
                }
            }
            SET_PREALLOCATED_BUFFER => {
                let slot: i32 = parcel.read();
                let buffer: Option<GraphicBuffer> = parcel.read_object();
                let Some(index) = usize::try_from(slot).ok().filter(|&slot| slot < SLOTS) else {
                    return Some(reply(out, BAD_VALUE));
                };
                let pinned = buffer.is_some_and(|buffer| self.pin(buffer.buffer_id));
                let old = std::mem::replace(
                    &mut state.slots[index],
                    Slot {
                        buffer,
                        pinned,
                        ..Slot::EMPTY
                    },
                );
                self.unpin(&old);
                if let Some(buffer) = buffer {
                    state.max_buffers = state.slots.iter().filter(|s| s.buffer.is_some()).count();
                    state.default_size = (buffer.width as u32, buffer.height as u32);
                    state.default_format = buffer.format;
                }
                self.wake_producers(state);
                return Some(reply(out, NO_ERROR));
            }
            GET_BUFFER_HISTORY => NO_ERROR,
            code => {
                tracing::warn!("vi: binder transaction {code} is not implemented");
                BAD_VALUE
            }
        };
        Some(reply(out, status))
    }

    fn queue(&self, state: &mut State, slot: i32, input: &QueueBufferInput) -> i32 {
        if !(0..=4).contains(&input.scaling_mode) {
            return BAD_VALUE;
        }
        let Some(index) = state.slot(slot).is_some().then_some(slot as usize) else {
            return BAD_VALUE;
        };
        let entry = &mut state.slots[index];
        let Some(buffer) = entry.buffer.filter(|_| entry.state == SlotState::Dequeued) else {
            return BAD_VALUE;
        };
        state.frames += 1;
        entry.state = SlotState::Queued;
        entry.frame = state.frames;
        state.queue.push_back(Item {
            slot: index,
            buffer,
            crop: input.crop,
            transform: input.transform,
            swap_interval: swap_interval(input.swap_interval),
            fence: input.fence,
        });
        NO_ERROR
    }

    /// Takes the frame to show at compositor frame `frame`: the oldest queued one, or with
    /// `latest` the newest, dropping the others. `None` while the frame taken last is held for
    /// its swap interval, or when nothing is queued.
    pub fn acquire(&self, frame: u64, latest: bool) -> Option<Item> {
        let mut state = self.state.lock();
        if !latest && frame < state.hold_until {
            return None;
        }
        let mut item = state.queue.pop_front()?;
        while latest && let Some(next) = state.queue.pop_front() {
            state.slots[item.slot].state = SlotState::Free;
            item = next;
        }
        state.slots[item.slot].state = SlotState::Acquired;
        state.hold_until = frame + u64::from(item.swap_interval);
        state.shown = true;
        Some(item)
    }

    /// The layer has a frame on screen or queued: it covers the layers below it.
    pub fn has_frames(&self) -> bool {
        let state = self.state.lock();
        state.shown || !state.queue.is_empty()
    }

    /// Gives a shown frame's buffer back to the producer.
    pub fn release(&self, item: &Item) {
        let mut state = self.state.lock();
        let slot = &mut state.slots[item.slot];
        if slot.state == SlotState::Acquired {
            slot.state = SlotState::Free;
            slot.fence = Fence::NONE;
        }
        self.wake_producers(state);
    }

    /// A slot became free: signal the native handle and run the DequeueBuffer requests waiting
    /// for one again.
    fn wake_producers(&self, mut state: MutexGuard<'_, State>) {
        let waiting = std::mem::take(&mut state.waiting);
        drop(state);
        self.released.signal();
        for request in waiting {
            request.retry();
        }
    }

    fn pin(&self, handle: u32) -> bool {
        handle != 0 && self.nvdrv.pin_buffer(handle)
    }

    fn unpin(&self, slot: &Slot) {
        if let Some(buffer) = slot.buffer.filter(|_| slot.pinned) {
            self.nvdrv.unpin_buffer(buffer.buffer_id);
        }
    }
}

impl Drop for BufferQueue {
    fn drop(&mut self) {
        let slots = std::mem::replace(&mut self.state.get_mut().slots, [Slot::EMPTY; SLOTS]);
        for slot in &slots {
            self.unpin(slot);
        }
    }
}

fn reply(mut out: ParcelWriter, status: i32) -> Vec<u8> {
    out.write(status);
    out.serialize()
}

/// A queued swap interval as vblanks: 0, negative and above 4 are speed hints the compositor
/// does not take, shown as 1.
fn swap_interval(raw: i32) -> u32 {
    if (1..=4).contains(&raw) {
        raw as u32
    } else {
        1
    }
}
