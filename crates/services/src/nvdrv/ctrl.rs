//! `/dev/nvhost-ctrl`: syncpoint reads, increments and waits, and the events NVN waits on for
//! syncpoint thresholds.
//!
//! An event wait that is not reached at once arms the event: a host1x action signals it when the
//! guest value gets there, and the guest waits on the event's readable end.

use std::sync::Arc;
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use lichii_gpu::host1x::{ActionHandle, SyncpointWaitError};
use lichii_kernel::Event;
use parking_lot::Mutex;

use super::ioctl::Args;
use super::syncpoints::MAX_SYNCPOINTS;
use super::{NvFence, NvResult, Nvdrv};

const MAX_EVENTS: u32 = 64;

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Status {
    #[default]
    Available,
    Waiting,
    Signalled,
    Cancelled,
}

/// Event waits that failed this often in a row wait on the host instead.
const MAX_FAILS: u32 = 2;

/// The part of an event a host1x action touches.
#[derive(Default)]
struct EventSlot {
    state: Mutex<SlotState>,
    event: Arc<Event>,
}

/// Every arm, clear and rearm bumps `generation`; an action armed under an older generation, even
/// one host1x is already running, changes nothing.
#[derive(Default)]
struct SlotState {
    status: Status,
    generation: u64,
}

impl EventSlot {
    fn in_use(&self) -> bool {
        self.state.lock().status == Status::Waiting
    }

    /// Starts a wait and returns the generation its action must match.
    fn arm(&self) -> u64 {
        let mut state = self.state.lock();
        state.generation += 1;
        state.status = Status::Waiting;
        state.generation
    }

    fn signal(&self, generation: u64) {
        let mut state = self.state.lock();
        if state.generation != generation {
            return;
        }
        if state.status == Status::Waiting {
            self.event.signal();
        }
        state.status = Status::Signalled;
    }
}

#[derive(Default)]
struct NvEvent {
    /// `Some` while registered.
    slot: Option<Arc<EventSlot>>,
    fails: u32,
    syncpoint: u32,
    value: u32,
    wait: Option<ActionHandle>,
}

impl NvEvent {
    fn in_use(&self) -> bool {
        self.slot.as_ref().is_some_and(|slot| slot.in_use())
    }
}

pub(crate) struct Ctrl {
    nvdrv: Arc<Nvdrv>,
    events: Mutex<[NvEvent; MAX_EVENTS as usize]>,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SyncptValue {
    id: u32,
    value: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SyncptWait {
    id: u32,
    threshold: u32,
    timeout: i32,
    value: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct EventWait {
    fence: NvFence,
    timeout: u32,
    value: u32,
}

/// Set in the event ids allocating waits return (`SyncpointEventValue::event_allocated`).
const ALLOCATED_BIT: u32 = 1 << 28;

/// The event id an event wait returns and `QueryEvent` takes (`SyncpointEventValue`). An
/// allocating wait puts the slot in bits 0..16 and the syncpoint in bits 16..28 and sets bit 28.
/// A plain wait ORs the slot over the syncpoint shifted to bit 4; only its low four bits (the
/// partial slot) have bits of their own.
fn encode_event_id(slot: u32, syncpoint: u32, allocated: bool) -> u32 {
    if allocated {
        (syncpoint & 0xfff) << 16 | ALLOCATED_BIT | slot
    } else {
        syncpoint << 4 | slot
    }
}

impl Ctrl {
    pub fn new(nvdrv: Arc<Nvdrv>) -> Self {
        Self {
            nvdrv,
            events: Mutex::new(std::array::from_fn(|_| NvEvent::default())),
        }
    }

    pub fn ioctl(&self, args: &mut Args) -> Option<NvResult> {
        Some(match args.number() {
            (0, 0x14) => args.fixed(|p| self.syncpt_read(p)),
            (0, 0x15) => args.fixed(|p| self.syncpt_incr(p)),
            (0, 0x16) => args.fixed(|p| self.syncpt_wait(p)),
            (0, 0x19) => args.fixed(|p| self.syncpt_wait(p)),
            (0, 0x1a) => args.fixed(|p| self.syncpt_read_max(p)),
            (0, 0x1b) => NvResult::ConfigVarNotFound,
            (0, 0x1c) => args.fixed(|p: &mut u32| self.clear_event_wait(*p)),
            (0, 0x1d) => args.fixed(|p| self.event_wait(p, true)),
            (0, 0x1e) => args.fixed(|p| self.event_wait(p, false)),
            (0, 0x1f) => args.fixed(|p: &mut u32| self.register_event(*p)),
            (0, 0x20) => args.fixed(|p: &mut u32| self.unregister_event(*p & 0xff)),
            (0, 0x21) => args.fixed(|p: &mut u64| self.unregister_events(*p)),
            _ => return None,
        })
    }

    fn syncpoints(&self) -> &lichii_gpu::host1x::Syncpoints {
        self.nvdrv.syncpoints.syncpoints()
    }

    fn syncpt_read(&self, params: &mut SyncptValue) -> NvResult {
        match self.syncpoints().guest().value(params.id) {
            Ok(value) => {
                params.value = value;
                NvResult::Success
            }
            Err(_) => NvResult::BadParameter,
        }
    }

    fn syncpt_read_max(&self, params: &mut SyncptValue) -> NvResult {
        match self.syncpoints().max(params.id) {
            Ok(value) => {
                params.value = value;
                NvResult::Success
            }
            Err(_) => NvResult::BadParameter,
        }
    }

    /// A CPU increment: reserved in the maximum, then counted on both banks at once.
    fn syncpt_incr(&self, id: &mut u32) -> NvResult {
        let syncpoints = self.syncpoints();
        if syncpoints.increase_max(*id, 1).is_err() {
            return NvResult::BadParameter;
        }
        let _ = syncpoints.host().increment(*id);
        let _ = syncpoints.guest().increment(*id);
        NvResult::Success
    }

    /// Blocks until the guest value reaches the threshold; a negative timeout (milliseconds)
    /// waits for good.
    fn syncpt_wait(&self, params: &mut SyncptWait) -> NvResult {
        let timeout = u64::try_from(params.timeout)
            .ok()
            .map(Duration::from_millis);
        match self
            .syncpoints()
            .guest()
            .wait(params.id, params.threshold, timeout)
        {
            Ok(value) => {
                params.value = value;
                NvResult::Success
            }
            Err(SyncpointWaitError::TimedOut { value }) => {
                params.value = value;
                NvResult::Timeout
            }
            Err(SyncpointWaitError::InvalidId(_)) => NvResult::BadParameter,
        }
    }

    fn event_wait(&self, params: &mut EventWait, allocation: bool) -> NvResult {
        let mut events = self.events.lock();
        let event_id = params.value;
        let unmark_fail = |events: &mut [NvEvent], result| {
            if !allocation && let Some(event) = events.get_mut(event_id as usize) {
                event.fails = 0;
            }
            result
        };
        let fence_id = params.fence.id as u32;
        if fence_id >= MAX_SYNCPOINTS {
            return unmark_fail(&mut *events, NvResult::BadParameter);
        }
        let table = &self.nvdrv.syncpoints;
        if params.fence.value == 0 {
            if table.is_allocated(fence_id) {
                params.value = table.min(fence_id);
            } else {
                tracing::warn!(
                    syncpoint = fence_id,
                    "event wait on an unallocated syncpoint"
                );
            }
            return unmark_fail(&mut *events, NvResult::Success);
        }
        if table.is_signalled(params.fence) {
            params.value = table.min(fence_id);
            return unmark_fail(&mut *events, NvResult::Success);
        }

        let target = params.fence.value;
        let slot = if allocation {
            params.value = 0;
            Self::find_free_event(&mut *events, fence_id)
        } else {
            params.value
        };
        if slot >= MAX_EVENTS {
            return NvResult::BadParameter;
        }
        let event = &mut events[slot as usize];
        let failing = event.fails > MAX_FAILS;
        if params.timeout == 0 && !failing {
            return NvResult::Timeout;
        }
        if params.timeout != 0 && (event.slot.is_none() || event.in_use()) {
            return NvResult::BadParameter;
        }
        if failing {
            if let Err(error) = self.syncpoints().guest().wait(fence_id, target, None) {
                tracing::error!(%error, "event wait fallback");
            }
            params.value = target;
            event.fails = 0;
            return NvResult::Success;
        }

        let shared = event.slot.clone().expect("registered events have a slot");
        let generation = shared.arm();
        event.syncpoint = fence_id;
        event.value = target;
        params.value = encode_event_id(slot, fence_id, allocation);
        let action = self
            .syncpoints()
            .guest()
            .register_action(fence_id, target, move || shared.signal(generation));
        event.wait = action.ok().flatten();
        NvResult::Timeout
    }

    fn find_free_event(events: &mut [NvEvent], syncpoint: u32) -> u32 {
        let mut reusable = None;
        let mut free = None;
        for (index, event) in events.iter().enumerate() {
            if event.slot.is_some() {
                if !event.in_use() {
                    if event.syncpoint == syncpoint {
                        return index as u32;
                    }
                    reusable = Some(index);
                }
            } else if free.is_none() {
                free = Some(index);
            }
        }
        if let Some(index) = free {
            Self::create_event(&mut events[index]);
            return index as u32;
        }
        reusable.map_or_else(
            || {
                tracing::error!("no free nvhost-ctrl event");
                0
            },
            |index| index as u32,
        )
    }

    fn create_event(event: &mut NvEvent) {
        *event = NvEvent {
            slot: Some(Arc::default()),
            ..NvEvent::default()
        };
    }

    fn free_event(events: &mut [NvEvent], slot: u32) -> NvResult {
        let Some(event) = events.get_mut(slot as usize) else {
            return NvResult::BadParameter;
        };
        if event.slot.is_none() {
            return NvResult::Success;
        }
        if event.in_use() {
            return NvResult::Busy;
        }
        *event = NvEvent::default();
        NvResult::Success
    }

    fn register_event(&self, id: u32) -> NvResult {
        if id >= MAX_EVENTS {
            return NvResult::BadParameter;
        }
        let mut events = self.events.lock();
        if events[id as usize].slot.is_some() {
            let result = Self::free_event(&mut *events, id);
            if result != NvResult::Success {
                return result;
            }
        }
        Self::create_event(&mut events[id as usize]);
        NvResult::Success
    }

    fn unregister_event(&self, id: u32) -> NvResult {
        Self::free_event(&mut *self.events.lock(), id)
    }

    fn unregister_events(&self, mut mask: u64) -> NvResult {
        let mut events = self.events.lock();
        while mask != 0 {
            let id = mask.trailing_zeros();
            mask &= !(1 << id);
            let result = Self::free_event(&mut *events, id);
            if result != NvResult::Success {
                return result;
            }
        }
        NvResult::Success
    }

    fn clear_event_wait(&self, event_id: u32) -> NvResult {
        let slot = event_id & 0xffff;
        if slot >= MAX_EVENTS {
            return NvResult::BadParameter;
        }
        let mut events = self.events.lock();
        let event = &mut events[slot as usize];
        let Some(shared) = event.slot.clone() else {
            return NvResult::BadParameter;
        };
        let mut state = shared.state.lock();
        state.generation += 1;
        if let Some(wait) = event.wait.take() {
            self.syncpoints().guest().cancel(wait);
        }
        event.fails += 1;
        state.status = Status::Cancelled;
        shared.event.clear();
        NvResult::Success
    }

    /// The event behind an id an event wait returned. A plain wait's id names every registered
    /// slot with its partial slot whose id it re-encodes to; the lowest wins when slots 16 apart
    /// on syncpoints whose low bits collide encode alike.
    pub fn query_event(&self, id: u32) -> Option<Arc<Event>> {
        let allocated = id & ALLOCATED_BIT != 0;
        let events = self.events.lock();
        let matches = |slot: u32| {
            let event = &events[slot as usize];
            event.slot.is_some() && encode_event_id(slot, event.syncpoint, allocated) == id
        };
        let slot = if allocated {
            Some(id & 0xffff).filter(|&slot| slot < MAX_EVENTS && matches(slot))
        } else {
            (id & 0xf..MAX_EVENTS)
                .step_by(16)
                .find(|&slot| matches(slot))
        };
        let Some(slot) = slot else {
            tracing::error!(
                id = format_args!("{id:#x}"),
                "query of an event that is not armed"
            );
            return None;
        };
        events[slot as usize]
            .slot
            .as_ref()
            .map(|shared| shared.event.clone())
    }
}

impl Drop for Ctrl {
    fn drop(&mut self) {
        let guest = self.nvdrv.syncpoints.syncpoints().guest();
        for event in self.events.get_mut().iter_mut() {
            if let Some(wait) = event.wait.take() {
                guest.cancel(wait);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{SWITCH1_TITLE, context};

    fn ctrl() -> (Arc<Nvdrv>, Ctrl) {
        let (context, _client) = context(SWITCH1_TITLE, Settings::default());
        let nvdrv = super::super::register(&context).unwrap();
        (nvdrv.clone(), Ctrl::new(nvdrv))
    }

    fn wait(ctrl: &Ctrl, syncpoint: u32, value: u32, allocation: bool) -> u32 {
        let mut params = EventWait {
            fence: NvFence {
                id: syncpoint as i32,
                value: 1,
            },
            timeout: 100,
            value,
        };
        assert_eq!(ctrl.event_wait(&mut params, allocation), NvResult::Timeout);
        params.value
    }

    fn slot_event(ctrl: &Ctrl, slot: u32) -> Arc<Event> {
        ctrl.events.lock()[slot as usize]
            .slot
            .as_ref()
            .unwrap()
            .event
            .clone()
    }

    #[test]
    fn allocating_wait_ids_round_trip_for_every_slot() {
        let (_nvdrv, ctrl) = ctrl();
        for slot in 0..MAX_EVENTS {
            let syncpoint = 0x40 + slot % 3;
            let id = wait(&ctrl, syncpoint, 0, true);
            assert_eq!(id, syncpoint << 16 | ALLOCATED_BIT | slot);
            let event = ctrl.query_event(id).expect("armed");
            assert!(Arc::ptr_eq(&event, &slot_event(&ctrl, slot)));
        }
        assert!(ctrl.query_event(0x41 << 16 | ALLOCATED_BIT).is_none());
    }

    #[test]
    fn plain_wait_ids_round_trip_for_every_slot() {
        let (_nvdrv, ctrl) = ctrl();
        for slot in 0..MAX_EVENTS {
            assert_eq!(ctrl.register_event(slot), NvResult::Success);
            let syncpoint = 0x20 + 4 * (slot % 3);
            let id = wait(&ctrl, syncpoint, slot, false);
            assert_eq!(id, syncpoint << 4 | slot);
            let event = ctrl.query_event(id).expect("armed");
            assert!(Arc::ptr_eq(&event, &slot_event(&ctrl, slot)));
        }
        assert!(ctrl.query_event(0x30 << 4).is_none());
    }

    #[test]
    fn a_cleared_wait_never_signals_the_rearmed_slot() {
        let (nvdrv, ctrl) = ctrl();
        let guest = nvdrv.syncpoints.syncpoints().guest();
        assert_eq!(ctrl.register_event(3), NvResult::Success);
        wait(&ctrl, 0x21, 3, false);
        let shared = ctrl.events.lock()[3].slot.clone().unwrap();
        let stale = shared.state.lock().generation;
        assert_eq!(ctrl.clear_event_wait(3), NvResult::Success);
        wait(&ctrl, 0x22, 3, false);

        shared.signal(stale);
        assert!(!shared.event.is_signaled(), "the cancelled action is stale");
        assert!(shared.in_use());

        guest.increment(0x21).unwrap();
        assert!(!shared.event.is_signaled());
        guest.increment(0x22).unwrap();
        assert!(shared.event.is_signaled());
        assert!(!shared.in_use());
    }
}
