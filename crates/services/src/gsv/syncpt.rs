//! `nn::syncpt`: the slot session (gsv 202) and the shared syncpoint page (gsv 204).
//!
//! A slot request (202's command 201) signals the slot's event once a syncpoint reaches a value;
//! the action runs on whichever thread makes the increment. The page is how the client waits
//! without a request per wait: it reads syncpoint values from the read-only half, and to wait it
//! flips its armed bit in the writable half and signals the request event. The server answers by
//! making the read-only acknowledge bit equal again and signalling the notify event; the client
//! then rereads the value and rearms when its threshold isn't reached yet.
//!
//! Slot and page waits release when the command processor reaches the increment (the host bank),
//! before the host GPU finished the work in front of it: nx2_gsv waits on the reference's guest
//! syncpoint, which SignalSyncPoint increments as the method executes. Switch 1's nvhost-ctrl
//! keeps waiting for host completion.
//!
//! Every increment is mirrored into the page as it is reached, before the waits it satisfies are
//! answered, as the reference does: the client's TryWaitNativeSync reads the page without a
//! request, and Mario Kart World aborts (2162-0001) when vdec's Stop moves a syncpoint the page
//! still shows at its old value. A request runs the server at once (the request event is
//! observed), an armed syncpoint whose value changed since its last answer is answered right away,
//! and the rest are answered by a syncpoint action on their next increment.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use lichii_common::{ResultCode, VAddr};
use lichii_gpu::host1x::{ActionHandle, MirrorHandle, Syncpoints};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use super::{handshake, push_readable};
use crate::{Service, ServiceContext};

/// Read-only half: header, u64 value per syncpoint, summary word, acknowledge bits.
const VALUES: u64 = 0x40;
const RO_SUMMARY: u64 = 0x1040;
const RO_BITS: u64 = 0x1100;
/// Writable half: summary word and armed bits.
const RW_SUMMARY: u64 = 0;
const RW_BITS: u64 = 0x100;
/// Syncpoints whose values fit before the read-only summary word.
const MAX_SYNCPOINTS: u32 = ((RO_SUMMARY - VALUES) / 8) as u32;
const WORDS: usize = MAX_SYNCPOINTS.div_ceil(64) as usize;
const RO_SIZE: u64 = 0x2000;

/// The page gsv 204 attached, if any.
#[derive(Default)]
pub(super) struct PageSlot(Mutex<Option<Arc<Page>>>);

/// gsv 204(version u32, max syncpoints u32; copy handles: writable then read-only transfer
/// memory). Lays out the read-only header and returns the request event (writable end, the client
/// signals it) and the notify event.
pub(super) fn attach(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    slot: &PageSlot,
) -> ResultCode {
    let version: u32 = ctx.pop();
    let count = ctx.pop::<u32>().min(MAX_SYNCPOINTS);
    let memory = (
        ctx.copy_handle::<TransferMemory>(0),
        ctx.copy_handle::<TransferMemory>(1),
    );
    let (Ok(rw), Ok(ro)) = memory else {
        return lichii_kernel::result::INVALID_HANDLE;
    };
    if ro.size() < RO_SIZE {
        tracing::error!(
            "nx2:gsv 204: read-only syncpt memory is {:#x} bytes",
            ro.size()
        );
        return lichii_kernel::result::INVALID_SIZE;
    }
    tracing::info!("nx2:gsv 204: syncpt page version {version}, {count} syncpoints");
    let page = Page::new(rw, ro, count, context.gpu.host1x().syncpoints.clone());
    ctx.push_copy_object(KObject::WritableEvent(page.request.clone()));
    ctx.push_copy_object(KObject::ReadableEvent(page.notify.clone()));
    *slot.0.lock() = Some(page);
    ResultCode::SUCCESS
}

struct Page {
    memory: Arc<AddressSpace>,
    rw: VAddr,
    ro: VAddr,
    /// Held so the client cannot unlock, shrink or remap the page while it is attached.
    _transfer: [Arc<TransferMemory>; 2],
    count: u32,
    syncpoints: Arc<Syncpoints>,
    request: Arc<Event>,
    notify: Arc<Event>,
    /// Syncpoints whose page value changed since a wait on them was last answered. A wait on a
    /// fresh id is answered even when the page value looks current: the client may have read it
    /// before the change.
    fresh: Arc<[AtomicU64; WORDS]>,
    mirrored: MirrorHandle,
    watches: Mutex<Watches>,
}

/// Syncpoints with an action registered for their next increment.
#[derive(Default)]
struct Watches {
    watching: [u64; WORDS],
    handles: HashMap<u32, ActionHandle>,
}

/// Writes `value` into the page's slot for `id` and marks the id fresh when the slot held an
/// older value. Runs on every mirrored increment, so it allocates nothing and takes no lock.
fn write_value(memory: &AddressSpace, ro: VAddr, fresh: &[AtomicU64; WORDS], id: u32, value: u32) {
    let address = ro + VALUES + u64::from(id) * 8;
    if memory.read::<u64>(address).ok() == Some(u64::from(value)) {
        return;
    }
    if memory.write(address, u64::from(value)).is_err() {
        tracing::warn!("nx2:syncpt page write at {address:#x} faulted");
        return;
    }
    fresh[id as usize / 64].fetch_or(1 << (id % 64), Ordering::SeqCst);
}

impl Page {
    fn new(
        rw: Arc<TransferMemory>,
        ro: Arc<TransferMemory>,
        count: u32,
        syncpoints: Arc<Syncpoints>,
    ) -> Arc<Self> {
        let memory = ro.owner().memory().clone();
        let fresh: Arc<[AtomicU64; WORDS]> = Arc::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let mirrored = syncpoints.host().set_mirror(count, {
            let (memory, ro, fresh) = (memory.clone(), ro.address(), fresh.clone());
            move |id, value| write_value(&memory, ro, &fresh, id, value)
        });
        let page = Arc::new_cyclic(|this: &Weak<Page>| {
            let this = this.clone();
            Self {
                memory,
                rw: rw.address(),
                ro: ro.address(),
                _transfer: [rw, ro],
                count,
                syncpoints,
                request: Arc::new(Event::observed(move || {
                    if let Some(page) = this.upgrade() {
                        page.serve();
                    }
                })),
                notify: Arc::new(Event::new()),
                fresh,
                mirrored,
                watches: Mutex::default(),
            }
        });
        let header = [
            1,
            0,
            count,
            VALUES as u32,
            RW_SUMMARY as u32,
            RO_SUMMARY as u32,
            RW_BITS as u32,
            RO_BITS as u32,
        ];
        for (index, word) in header.into_iter().enumerate() {
            if page.memory.write(page.ro + index as u64 * 4, word).is_err() {
                tracing::warn!("nx2:syncpt page header write faulted");
            }
        }
        let bank = page.syncpoints.host();
        for id in 0..count {
            let _ = bank.publish(id);
        }
        page
    }

    fn read(&self, address: VAddr) -> u64 {
        self.memory.read(address).unwrap_or(0)
    }

    fn write(&self, address: VAddr, value: u64) {
        if self.memory.write(address, value).is_err() {
            tracing::warn!("nx2:syncpt page write at {address:#x} faulted");
        }
    }

    /// Answers every armed wait whose page value changed since its last answer, and watches the
    /// rest. Values reach the page only through the bank's mirror, which orders them.
    fn serve(self: &Arc<Self>) {
        self.request.clear();
        let bank = self.syncpoints.host();
        let mut watch = Vec::new();
        let mut answered = false;
        {
            let mut watches = self.watches.lock();
            let words = self.count.div_ceil(64) as usize;
            for (word, (watching, fresh)) in watches
                .watching
                .iter_mut()
                .zip(self.fresh.iter())
                .enumerate()
                .take(words)
            {
                let armed = self.read(self.rw + RW_BITS + word as u64 * 8);
                let acknowledged_at = self.ro + RO_BITS + word as u64 * 8;
                let acknowledged = self.read(acknowledged_at);
                let mut answer = 0;
                let mut pending = armed ^ acknowledged;
                while pending != 0 {
                    let bit = 1u64 << pending.trailing_zeros();
                    pending &= !bit;
                    let id = word as u32 * 64 + bit.trailing_zeros();
                    let mirrored = id < self.count && bank.publish(id).is_ok();
                    let value = bank.value(id).unwrap_or(0);
                    if !mirrored || fresh.load(Ordering::SeqCst) & bit != 0 {
                        answer |= bit;
                    } else if *watching & bit == 0 {
                        *watching |= bit;
                        watch.push((id, value));
                    }
                }
                fresh.fetch_and(!answer, Ordering::SeqCst);
                if answer != 0 {
                    self.write(acknowledged_at, acknowledged ^ answer);
                    answered = true;
                }
            }
            self.write(self.ro + RO_SUMMARY, self.read(self.rw + RW_SUMMARY));
        }
        if answered {
            self.notify.signal();
        }
        for (id, value) in watch {
            self.watch(id, value);
        }
    }

    /// Watches for the increment after `value`, the value the wait was checked against, so an
    /// increment since then runs the action at once.
    fn watch(self: &Arc<Self>, id: u32, value: u32) {
        let bank = self.syncpoints.host();
        let this = Arc::downgrade(self);
        let registered = bank.register_action(id, value.wrapping_add(1), move || {
            if let Some(page) = this.upgrade() {
                page.incremented(id);
            }
        });
        if let Ok(Some(handle)) = registered {
            self.watches.lock().handles.insert(id, handle);
        }
    }

    fn incremented(self: &Arc<Self>, id: u32) {
        {
            let mut watches = self.watches.lock();
            let (word, bit) = (id as usize / 64, 1u64 << (id % 64));
            watches.watching[word] &= !bit;
            watches.handles.remove(&id);
        }
        self.serve();
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        let bank = self.syncpoints.host();
        bank.clear_mirror(self.mirrored);
        for handle in self.watches.get_mut().handles.values() {
            bank.cancel(*handle);
        }
    }
}

/// gsv 202. Slots are event handles NVN's `NvOsEventPool` multi-waits on; a request signals its
/// slot when a syncpoint reaches a value.
pub(super) struct SyncptSession {
    syncpoints: Arc<Syncpoints>,
    slots: Arc<Mutex<HashMap<u32, Slot>>>,
}

#[derive(Default)]
struct Slot {
    event: Arc<Event>,
    generation: u64,
    request: Option<(u32, ActionHandle)>,
}

impl SyncptSession {
    pub fn new(context: &Arc<ServiceContext>) -> Self {
        Self {
            syncpoints: context.gpu.host1x().syncpoints.clone(),
            slots: Arc::default(),
        }
    }

    fn slot_event(&self, slot: u32) -> Arc<Event> {
        self.slots.lock().entry(slot).or_default().event.clone()
    }

    /// Drops the slot's request and returns its event. The generation bump makes an action that
    /// is already running signal nothing.
    fn cancel(&self, slots: &mut HashMap<u32, Slot>, slot: u32) -> (Arc<Event>, u64) {
        let slot = slots.entry(slot).or_default();
        if let Some((_, handle)) = slot.request.take() {
            self.syncpoints.host().cancel(handle);
        }
        slot.generation += 1;
        (slot.event.clone(), slot.generation)
    }

    /// 201(slot, counter, u64 value, mode): signals the slot once syncpoint `!counter` reaches
    /// `value`. A new request starts unsignalled: slots are reused, and a slot an earlier fence
    /// wait left signalled read as a finished movie frame (NX2_NOTES row 147).
    fn request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let slot: u32 = ctx.pop();
        let id = !ctx.pop::<u32>();
        let threshold = ctx.pop::<u64>() as u32;
        let (event, generation) = self.cancel(&mut self.slots.lock(), slot);
        event.clear();
        let bank = self.syncpoints.host();
        if bank.value(id).is_err() {
            tracing::warn!("nx2:syncpt 201: syncpoint {id:#x} out of range, signalling");
            event.signal();
            return ResultCode::SUCCESS;
        }
        let slots = Arc::downgrade(&self.slots);
        let signal = event.clone();
        let registered = bank.register_action(id, threshold, move || {
            let Some(slots) = slots.upgrade() else {
                return;
            };
            let mut slots = slots.lock();
            if let Some(slot) = slots.get_mut(&slot).filter(|s| s.generation == generation) {
                slot.request = None;
                signal.signal();
            }
        });
        if let Ok(Some(handle)) = registered {
            let mut slots = self.slots.lock();
            match slots.get_mut(&slot).filter(|s| s.generation == generation) {
                Some(slot) => slot.request = Some((id, handle)),
                None => {
                    bank.cancel(handle);
                }
            }
        }
        ResultCode::SUCCESS
    }
}

impl Drop for SyncptSession {
    fn drop(&mut self) {
        let bank = self.syncpoints.host();
        for slot in self.slots.lock().values_mut() {
            if let Some((_, handle)) = slot.request.take() {
                bank.cancel(handle);
            }
            slot.generation += 1;
        }
    }
}

impl Service for SyncptSession {
    fn name(&self) -> &str {
        "nx2:syncpt"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => handshake(ctx),
            1 => {
                ctx.push(32u32);
                ResultCode::SUCCESS
            }
            22 => {
                let slot: u32 = ctx.pop();
                let event = self.slot_event(slot);
                push_readable(ctx, &event)
            }
            103 => {
                let id = !ctx.pop::<u32>();
                let value = self.syncpoints.host().value(id).unwrap_or(0);
                ctx.push(u64::from(value));
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            201 => self.request(ctx),
            202 => {
                let slot: u32 = ctx.pop();
                let (event, _) = self.cancel(&mut self.slots.lock(), slot);
                event.signal();
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
