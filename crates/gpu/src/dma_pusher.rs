//! The front half of a channel (the DMA pusher and the puller): fetches GPFIFO entries from GPU
//! memory, parses them, executes host class methods, binds subchannels, keeps the register
//! mirrors macros read, runs macros, and sends every engine method to the back.
//!
//! The front waits for the back only where it reads what the back writes: an unmet semaphore
//! acquire, a syncpoint wait, wait-for-idle and MME DMA reads (NX2_GPU_THREAD_SPLIT.md). An acquire
//! or syncpoint wait that still fails after that suspends the channel; the front runs other
//! channels and resumes it at the next word once the condition holds. A syncpoint wait holds the
//! channel only until the back has executed the increments it waits for, never until the host GPU
//! finished their work, and only for increments a submission reserved: the reference GPU does not
//! wait on syncpoints at all. A channel suspended for [`LONG_WAIT`] is reported once.
//!
//! A GPFIFO entry with its SYNC bit is fetched once the back executed the work before it, as the
//! host waits for the methods before such an entry to be processed: the channel suspends until
//! the back reached it, without the front waiting. It does not wait for that work to finish on
//! the host GPU; the reference ignores the bit on Switch 1 and only orders it behind the back on
//! Switch 2, and a wait for the host GPU there kept every frame's commands behind the previous
//! frame's rendering. On Switch 2 an entry on a page GPU work may have written is fetched after
//! that work completed and its words reached guest memory: titles fetch command lists their own
//! compute writes each frame. A Switch 2 SYNC entry is checked for such pages again once the back
//! executed the work before it, which may be what marked them.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use lichii_common::GpuVAddr;

use crate::back::{BackChannelSetup, SideCommand};
use crate::backend::{ChannelId, GpuWrittenPages, ReleaseSize};
use crate::channel::{ChannelConfig, SemaphoreWaiters, SyncpointFence, SyncpointIncrement};
use crate::engines::EngineKind;
use crate::engines::kepler_compute::Regs as ComputeRegs;
use crate::engines::maxwell_3d::{MACRO_REGISTERS_START, RegisterState, Regs};
use crate::gmmu::{Gmmu, GpuFault, RangeTarget};
use crate::gpu::{Inbox, LONG_WAIT};
use crate::host::{AcquireOp, FIRST_ENGINE_METHOD, HostAction, HostClass, HostState, ReleaseKind};
use crate::host1x::{Syncpoints, is_reached};
use crate::macros::{
    LOAD_MME_INSTRUCTION_RAM_POINTER, LOAD_MME_START_ADDRESS_RAM,
    LOAD_MME_START_ADDRESS_RAM_POINTER, MACRO_METHOD_BASE, MacroDataRam, MacroEngine, MacroHost,
    MacroIsa,
};
use crate::pushbuffer::{CommandParser, Flow, GpEntry, MethodCall, MethodHandler};
use crate::reg_index;
use crate::semaphore_mirror::SemaphoreMirror;
use crate::stream::{FetchSync, Stream};

/// Engines every channel has bound before any SET_OBJECT: NVN uses subchannel 2 as
/// inline-to-memory without binding it (NX2_NOTES row 49).
const DEFAULT_SUBCHANNELS: [(usize, EngineKind, u32); 1] =
    [(2, EngineKind::InlineToMemory, 0xa140)];

const CONST_BUFFER_DATA: u32 = reg_index!(Regs, const_buffer.buffer);
const CONST_BUFFER_OFFSET: usize = reg_index!(Regs, const_buffer.offset) as usize;
const FALCON_4: u32 = reg_index!(Regs, falcon) + 4;
const COMPUTE_DATA_UPLOAD: u32 = reg_index!(ComputeRegs, data_upload);
const SHADOW_SCRATCH_0: usize = reg_index!(Regs, shadow_scratch) as usize;

/// What the front's channels share.
pub(crate) struct FrontShared {
    pub stream: Stream,
    /// Switch 2: the channels of a process share one graphics context, and with it the macro
    /// code of their 3D engines (NX2_NOTES row 53). Each keeps its own data RAM.
    pub shared_macros: Option<MacroEngine>,
    /// Switch 2: the compute engines' own macro code, used by the channels whose compute class
    /// set its instruction or start-address pointer (`ChannelState::compute_macros`). The
    /// other channels' compute uploads and calls use the 3D macros, as in the reference, and no
    /// compute upload overwrites the 3D macros at the same code addresses.
    pub compute_macros: Option<MacroEngine>,
    /// The words of the entry being parsed.
    pub words: Vec<u32>,
    pub waiters: Arc<SemaphoreWaiters>,
    pub inbox: Arc<Inbox>,
    pub syncpoints: Arc<Syncpoints>,
    /// Switch 2: the device pages GPU work may have written. An entry on one is fetched after the
    /// back synchronised its words.
    pub written_pages: Option<Arc<GpuWrittenPages>>,
}

/// A channel's queued work.
pub(crate) enum Work {
    Entry(GpEntry),
    SyncpointWait(SyncpointFence),
    Increment(SyncpointIncrement),
    /// The renderer submits what it recorded.
    EndOfSubmission,
    /// The channel's later work uses this GPU address space.
    BindAddressSpace(Arc<Gmmu>),
}

/// Why a channel is suspended.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Wait {
    Semaphore {
        /// The host method that acquired.
        method: u32,
        address: GpuVAddr,
        payload: u64,
        op: AcquireOp,
        wide: bool,
    },
    Syncpoint(SyncpointFence),
    /// The fetch of the entry at the queue's front, which has its SYNC bit or lies on GPU-written
    /// pages, until
    /// [`Stream::fetch_synced`] holds for this ticket.
    Fetch(u64),
}

/// One channel on the front.
pub(crate) struct FrontChannel {
    id: ChannelId,
    memory: Arc<Gmmu>,
    parser: CommandParser,
    state: ChannelState,
    queue: VecDeque<Work>,
    /// Words of the entry at the queue's front already processed.
    offset: u32,
    wait: Option<Wait>,
    /// The entry at the queue's front waited for its fetch synchronisation.
    fetch_synced: bool,
    /// When the front first saw the current wait, and whether it reported it.
    waiting_since: Option<Instant>,
    wait_reported: bool,
}

/// What the method handler of a channel changes.
struct ChannelState {
    host: HostState,
    subchannels: [Option<EngineKind>; 8],
    /// The 3D registers as the back will have them after the methods sent so far.
    graphics: RegisterState,
    /// The compute registers, likewise.
    compute: Box<[u32]>,
    data_ram: MacroDataRam,
    /// Switch 1: the channel's own 3D macros.
    macros: Option<MacroEngine>,
    /// Parameters of the macro call being received, for the 3D and the compute engine.
    calls: [MacroCall; 2],
    /// Switch 1: GPU address the back takes the next compute DATA_UPLOAD word to come from.
    upload_next: Option<GpuVAddr>,
    /// Switch 2: the compute class set a macro instruction or start-address pointer, so its
    /// uploads and calls use [`FrontShared::compute_macros`] from then on.
    compute_macros: bool,
}

/// Parameters one macro call can have: the words of one command (a 13-bit count).
const MAX_MACRO_PARAMETERS: usize = 0x1fff;

struct MacroCall {
    method: u32,
    /// Holds [`MAX_MACRO_PARAMETERS`] from the start, so collecting never allocates.
    parameters: Vec<u32>,
}

impl MacroCall {
    fn new() -> Self {
        Self {
            method: 0,
            parameters: Vec::with_capacity(MAX_MACRO_PARAMETERS),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MacroTarget {
    Graphics = 0,
    Compute = 1,
}

impl FrontChannel {
    /// Creates the channel on the front and the back.
    pub(crate) fn open(
        id: ChannelId,
        config: &ChannelConfig,
        reference: Arc<std::sync::atomic::AtomicU32>,
        shared: &mut FrontShared,
    ) -> Self {
        shared
            .stream
            .side(SideCommand::CreateChannel(BackChannelSetup {
                id,
                memory: config.memory.clone(),
                gob_layout: config.gob_layout,
                reference,
            }));
        shared.stream.select(id);
        let mut subchannels = [None; 8];
        for (subchannel, engine, class) in DEFAULT_SUBCHANNELS {
            subchannels[subchannel] = Some(engine);
            shared.stream.bind(engine, class);
        }
        Self {
            id,
            memory: config.memory.clone(),
            parser: CommandParser::new(),
            state: ChannelState {
                host: HostState::new(config.host_class),
                subchannels,
                graphics: RegisterState::new(),
                compute: vec![0; MACRO_METHOD_BASE as usize].into_boxed_slice(),
                data_ram: MacroDataRam::new(),
                macros: None,
                calls: [MacroCall::new(), MacroCall::new()],
                upload_next: None,
                compute_macros: false,
            },
            queue: VecDeque::new(),
            offset: 0,
            wait: None,
            fetch_synced: false,
            waiting_since: None,
            wait_reported: false,
        }
    }

    /// Removes the channel from the back.
    pub(crate) fn close(self, shared: &mut FrontShared) {
        shared.waiters.remove(self.id);
        shared.stream.side(SideCommand::DestroyChannel(self.id));
        shared.stream.deselect(self.id);
    }

    pub(crate) fn push(&mut self, work: Work) {
        self.queue.push_back(work);
    }

    pub(crate) fn is_waiting(&self) -> bool {
        self.wait.is_some()
    }

    /// Whether the channel is parked until the back synchronised an entry's fetch.
    pub(crate) fn waits_for_fetch(&self) -> bool {
        matches!(self.wait, Some(Wait::Fetch(_)))
    }

    /// Whether the channel has work it can run now.
    pub(crate) fn is_runnable(&self, shared: &FrontShared) -> bool {
        match self.wait {
            None => !self.queue.is_empty(),
            Some(wait) => self.wait_met(wait, shared),
        }
    }

    fn wait_met(&self, wait: Wait, shared: &FrontShared) -> bool {
        match wait {
            Wait::Semaphore {
                address,
                payload,
                op,
                wide,
                ..
            } => {
                shared.waiters.latched(self.id)
                    || semaphore_met(
                        &self.memory,
                        shared.syncpoints.semaphore_mirror(),
                        address,
                        payload,
                        op,
                        wide,
                    )
            }
            Wait::Syncpoint(fence) => shared
                .syncpoints
                .host()
                .is_reached(fence.id, fence.value)
                .unwrap_or(true),
            Wait::Fetch(ticket) => shared.stream.fetch_synced(ticket),
        }
    }

    /// Reports a channel suspended for [`LONG_WAIT`], once per wait, naming what it waits on, as
    /// the reference does for acquires: a title that stops presenting usually waits on one.
    pub(crate) fn report_long_wait(&mut self, shared: &FrontShared) {
        let Some(wait) = self.wait else {
            return;
        };
        let now = Instant::now();
        let since = *self.waiting_since.get_or_insert(now);
        if self.wait_reported || now.duration_since(since) < LONG_WAIT {
            return;
        }
        self.wait_reported = true;
        let channel = self.id.0;
        match wait {
            Wait::Semaphore {
                method,
                address,
                payload,
                op,
                wide,
            } => {
                let hex =
                    |value: Option<u64>| value.map_or("unmapped".into(), |v| format!("{v:#x}"));
                let current = semaphore_value(
                    &self.memory,
                    shared.syncpoints.semaphore_mirror(),
                    address,
                    wide,
                );
                tracing::warn!(
                    channel,
                    method = format_args!("{method:#x}"),
                    address = format_args!("{address:#x}"),
                    device = %hex(self.memory.translate(address)),
                    ?op,
                    payload = format_args!("{payload:#x}"),
                    wide,
                    current = %hex(current.ok()),
                    latched = shared.waiters.latched(self.id),
                    "GPU channel has waited 2 s on a semaphore acquire"
                );
            }
            Wait::Syncpoint(fence) => {
                let syncpoints = &shared.syncpoints;
                tracing::warn!(
                    channel,
                    id = fence.id,
                    value = fence.value,
                    host = ?syncpoints.host().value(fence.id).ok(),
                    guest = ?syncpoints.guest().value(fence.id).ok(),
                    max = ?syncpoints.max(fence.id).ok(),
                    "GPU channel has waited 2 s on a syncpoint"
                );
            }
            Wait::Fetch(ticket) => tracing::warn!(
                channel,
                ticket,
                "GPU channel has waited 2 s for the back to synchronise a GPFIFO entry's fetch"
            ),
        }
    }

    /// Runs the channel until its queue is empty or it waits again.
    pub(crate) fn run(&mut self, shared: &mut FrontShared) {
        if let Some(wait) = self.wait {
            if !self.wait_met(wait, shared) {
                return;
            }
            self.wait = None;
            self.waiting_since = None;
            self.wait_reported = false;
            shared.waiters.remove(self.id);
        }
        if self.queue.is_empty() {
            return;
        }
        shared.stream.select(self.id);
        while let Some(work) = self.queue.front() {
            match *work {
                Work::Entry(entry) => self.run_entry(entry, shared),
                Work::SyncpointWait(fence) => {
                    self.queue.pop_front();
                    self.dispatch(shared).syncpoint_wait(fence);
                }
                Work::BindAddressSpace(_) => {
                    let Some(Work::BindAddressSpace(memory)) = self.queue.pop_front() else {
                        unreachable!("the queue's front was just matched");
                    };
                    shared
                        .stream
                        .side(SideCommand::BindAddressSpace(self.id, memory.clone()));
                    self.memory = memory;
                }
                Work::Increment(increment) => {
                    self.queue.pop_front();
                    let mut dispatch = self.dispatch(shared);
                    if increment.wait_for_idle {
                        dispatch.wait_for_idle();
                    }
                    for _ in 0..increment.count {
                        dispatch.shared.stream.syncpoint_increment(increment.id);
                    }
                }
                Work::EndOfSubmission => {
                    self.queue.pop_front();
                    shared.stream.flush_commands();
                }
            }
            if self.wait.is_some() {
                break;
            }
        }
        shared.stream.flush_pending();
    }

    fn run_entry(&mut self, entry: GpEntry, shared: &mut FrontShared) {
        let length = entry.words();
        let start = self.offset;
        if start >= length {
            self.queue.pop_front();
            self.offset = 0;
            return;
        }
        if start == 0 && !self.fetch_synced {
            let sync = if self.on_written_pages(entry, shared) {
                Some(FetchSync::Host)
            } else if entry.sync() {
                Some(if shared.written_pages.is_some() {
                    FetchSync::HostIfWritten
                } else {
                    FetchSync::Order
                })
            } else {
                None
            };
            if let Some(sync) = sync {
                let ticket =
                    shared
                        .stream
                        .synchronize_fetch(entry.address(), u64::from(length) * 4, sync);
                self.fetch_synced = true;
                self.wait = Some(Wait::Fetch(ticket));
                return;
            }
        }
        self.fetch_synced = false;
        let mut words = std::mem::take(&mut shared.words);
        words.clear();
        words.resize((length - start) as usize, 0);
        let address = entry.address() + u64::from(start) * 4;
        let consumed = match self
            .memory
            .read_block(address, bytemuck::cast_slice_mut(&mut words))
        {
            Ok(()) => self.dispatch_words(&words, address, shared),
            Err(fault) => {
                tracing::warn!(?fault, ?entry, "GPFIFO entry in unmapped GPU memory");
                words.len()
            }
        };
        shared.words = words;
        let done = start + consumed as u32;
        if done >= length {
            self.queue.pop_front();
            self.offset = 0;
        } else {
            self.offset = done;
        }
    }

    /// Switch 2: whether GPU work may have written the entry's words.
    fn on_written_pages(&self, entry: GpEntry, shared: &FrontShared) -> bool {
        let Some(pages) = &shared.written_pages else {
            return false;
        };
        let mut written = false;
        self.memory
            .for_each_range(entry.address(), u64::from(entry.words()) * 4, |range| {
                if let RangeTarget::Device(device) = range.target {
                    written |= pages.may_be_written(device, range.size);
                }
            });
        written
    }

    /// Parses `words`, read from GPU address `address`.
    fn dispatch_words(
        &mut self,
        words: &[u32],
        address: GpuVAddr,
        shared: &mut FrontShared,
    ) -> usize {
        let mut dispatch = Dispatch {
            id: self.id,
            memory: &self.memory,
            state: &mut self.state,
            shared,
            wait: &mut self.wait,
            words_address: address,
        };
        self.parser.process(words, &mut dispatch)
    }

    fn dispatch<'a>(&'a mut self, shared: &'a mut FrontShared) -> Dispatch<'a> {
        Dispatch {
            id: self.id,
            memory: &self.memory,
            state: &mut self.state,
            shared,
            wait: &mut self.wait,
            words_address: 0,
        }
    }
}

/// The semaphore at `address` as GPU acquires see it: a 32-bit word may hold a mirrored Switch 2
/// queue semaphore value kept out of memory.
fn semaphore_value(
    memory: &Gmmu,
    mirror: &SemaphoreMirror,
    address: GpuVAddr,
    wide: bool,
) -> Result<u64, GpuFault> {
    if wide {
        return memory.read::<u64>(address);
    }
    let load = || memory.read::<u32>(address);
    match memory.translate(address) {
        Some(device) => mirror.acquire_view(device, load),
        None => load(),
    }
    .map(u64::from)
}

/// Whether the semaphore at `address` meets the acquire.
fn semaphore_met(
    memory: &Gmmu,
    mirror: &SemaphoreMirror,
    address: GpuVAddr,
    payload: u64,
    op: AcquireOp,
    wide: bool,
) -> bool {
    match semaphore_value(memory, mirror, address, wide) {
        Ok(current) => op.is_met(current, payload, wide),
        Err(fault) => {
            tracing::warn!(?fault, "semaphore acquire on unmapped GPU memory");
            true
        }
    }
}

/// The method handler of one channel while it parses.
struct Dispatch<'a> {
    id: ChannelId,
    memory: &'a Gmmu,
    state: &'a mut ChannelState,
    shared: &'a mut FrontShared,
    wait: &'a mut Option<Wait>,
    /// GPU address of the first word being parsed.
    words_address: GpuVAddr,
}

impl MethodHandler for Dispatch<'_> {
    #[inline]
    fn call(&mut self, call: MethodCall) -> Flow {
        if call.method < FIRST_ENGINE_METHOD {
            return self.host_method(call);
        }
        match self.state.subchannels[call.subchannel as usize] {
            None => {}
            Some(EngineKind::Maxwell3d) => self.graphics_method(call),
            Some(EngineKind::Compute) => self.compute_method(call),
            Some(engine) => self
                .shared
                .stream
                .engine_word(engine, call.method, call.argument),
        }
        Flow::Continue
    }
}

impl Dispatch<'_> {
    fn host_method(&mut self, call: MethodCall) -> Flow {
        let stream = &mut self.shared.stream;
        match self
            .state
            .host
            .call(call.subchannel, call.method, call.argument)
        {
            HostAction::None => {}
            HostAction::Bind { class: 0, .. } => {}
            HostAction::Bind { subchannel, class } => {
                let engine = EngineKind::from_class(class);
                self.state.subchannels[subchannel as usize] = engine;
                match engine {
                    Some(engine) => stream.bind(engine, class),
                    None => {
                        tracing::debug!(subchannel, class, "unknown class unbinds a subchannel")
                    }
                }
            }
            HostAction::Acquire {
                address,
                payload,
                op,
                wide,
            } => return self.acquire(call.method, address, payload, op, wide),
            HostAction::Release {
                address,
                payload,
                kind,
            } => stream.release(
                address,
                payload,
                match kind {
                    ReleaseKind::Four => ReleaseSize::Four,
                    ReleaseKind::Eight => ReleaseSize::Eight,
                    ReleaseKind::WithTimestamp => ReleaseSize::Sixteen,
                },
            ),
            HostAction::Reduce {
                address,
                payload,
                op,
                signed,
            } => stream.reduce(address, payload, op, signed),
            HostAction::SyncpointWait { id, value } => {
                return self.syncpoint_wait(SyncpointFence { id, value });
            }
            HostAction::SyncpointIncrement { id } => stream.syncpoint_increment(id),
            HostAction::SetReference(value) => stream.set_reference(value),
            HostAction::WaitForIdle => self.wait_for_idle(),
            HostAction::InvalidateCaches => stream.invalidate_caches(),
        }
        Flow::Continue
    }

    /// An acquire the word in memory does not meet yet may be met by a release still queued for
    /// the back: drain it and look again, then suspend the channel (NX2_NOTES row 50). The
    /// channel waits from before its first look, so a release that lands and is overwritten at
    /// any point before the second look still latches it.
    fn acquire(
        &mut self,
        method: u32,
        address: GpuVAddr,
        payload: u64,
        op: AcquireOp,
        wide: bool,
    ) -> Flow {
        if let Some(device) = self.memory.translate(address) {
            self.shared.waiters.add(self.id, device, payload, op, wide);
        }
        if semaphore_met(
            self.memory,
            self.shared.syncpoints.semaphore_mirror(),
            address,
            payload,
            op,
            wide,
        ) {
            self.shared.waiters.remove(self.id);
            return Flow::Continue;
        }
        self.shared.stream.drain();
        if semaphore_met(
            self.memory,
            self.shared.syncpoints.semaphore_mirror(),
            address,
            payload,
            op,
            wide,
        ) {
            self.shared.waiters.remove(self.id);
            return Flow::Continue;
        }
        self.shared.stream.release_fences();
        *self.wait = Some(Wait::Semaphore {
            method,
            address,
            payload,
            op,
            wide,
        });
        Flow::Suspend
    }

    /// A syncpoint wait holds the channel until the back has executed the increments it waits
    /// for (the host bank), so its later work follows the work before them in the stream. A value
    /// past every increment submissions reserved counts as reached: nothing queued reaches it.
    fn syncpoint_wait(&mut self, fence: SyncpointFence) -> Flow {
        let syncpoints = self.shared.syncpoints.clone();
        let bank = syncpoints.host();
        let max = match (
            bank.is_reached(fence.id, fence.value),
            syncpoints.max(fence.id),
        ) {
            (Ok(true), _) => return Flow::Continue,
            (Ok(false), Ok(max)) => max,
            (Err(error), _) | (_, Err(error)) => {
                tracing::warn!(%error, "GPU waits on an invalid syncpoint");
                return Flow::Continue;
            }
        };
        if !is_reached(max, fence.value) {
            tracing::debug!(
                id = fence.id,
                value = fence.value,
                max,
                "GPU waits for a syncpoint value no submission reserved"
            );
            return Flow::Continue;
        }
        self.shared.stream.drain();
        if bank.is_reached(fence.id, fence.value).unwrap_or(true) {
            return Flow::Continue;
        }
        let inbox = self.shared.inbox.clone();
        let _ = bank.register_action(fence.id, fence.value, move || inbox.wake());
        *self.wait = Some(Wait::Syncpoint(fence));
        Flow::Suspend
    }

    fn wait_for_idle(&mut self) {
        self.shared.stream.wait_for_idle();
        self.shared.stream.drain();
    }

    #[inline]
    fn graphics_method(&mut self, call: MethodCall) {
        let method = call.method;
        if method >= MACRO_REGISTERS_START {
            self.macro_word(MacroTarget::Graphics, call);
            return;
        }
        let state = &mut *self.state;
        if is_load_mme(method) {
            state.graphics.write(method, call.argument);
            let isa = match state.host.class() {
                HostClass::Maxwell => MacroIsa::Fermi,
                HostClass::Ampere => MacroIsa::Mme64,
            };
            let engine = match isa {
                MacroIsa::Fermi => state.macros.get_or_insert_with(|| MacroEngine::new(isa)),
                MacroIsa::Mme64 => self
                    .shared
                    .shared_macros
                    .get_or_insert_with(|| MacroEngine::new(isa)),
            };
            engine.write_load_mme(method, call.argument);
            return;
        }
        send_graphics(
            &mut state.graphics,
            &mut self.shared.stream,
            method,
            call.argument,
        );
    }

    #[inline]
    fn compute_method(&mut self, call: MethodCall) {
        let method = call.method;
        let native = self.state.host.class() == HostClass::Ampere;
        if method >= MACRO_METHOD_BASE {
            if native {
                self.macro_word(MacroTarget::Compute, call);
            }
            return;
        }
        if native && is_load_mme(method) {
            self.state.compute[method as usize] = call.argument;
            let shared = &mut *self.shared;
            self.state.compute_macros |= matches!(
                method,
                LOAD_MME_INSTRUCTION_RAM_POINTER | LOAD_MME_START_ADDRESS_RAM_POINTER
            );
            let engine = if self.state.compute_macros {
                &mut shared.compute_macros
            } else {
                &mut shared.shared_macros
            };
            engine
                .get_or_insert_with(|| MacroEngine::new(MacroIsa::Mme64))
                .write_load_mme(method, call.argument);
            return;
        }
        if !native && method == COMPUTE_DATA_UPLOAD {
            let address = self.words_address + u64::from(call.position) * 4;
            if self.state.upload_next != Some(address) {
                self.shared.stream.upload_source(address);
            }
            self.state.upload_next = Some(address + 4);
        }
        send_compute(
            &mut self.state.compute,
            &mut self.shared.stream,
            method,
            call.argument,
        );
    }

    /// One parameter word of a macro call; the macro runs once its command's last word arrived,
    /// in whichever GPFIFO entry that is (NX2_NOTES row 73).
    fn macro_word(&mut self, target: MacroTarget, call: MethodCall) {
        let ChannelState {
            host,
            graphics,
            compute,
            data_ram,
            macros,
            calls,
            compute_macros,
            ..
        } = &mut *self.state;
        let pending = &mut calls[target as usize];
        if pending.parameters.is_empty() {
            pending.method = call.method;
        }
        pending.parameters.push(call.argument);
        if call.remaining != 0 {
            return;
        }
        let shared = &mut *self.shared;
        let engine = match (host.class(), target) {
            (HostClass::Maxwell, _) => macros.as_mut(),
            (HostClass::Ampere, MacroTarget::Compute) if *compute_macros => {
                shared.compute_macros.as_mut()
            }
            (HostClass::Ampere, _) => shared.shared_macros.as_mut(),
        };
        if let Some(engine) = engine {
            A0_MACRO.with(|m| m.set(pending.method));
            A0_PARAMS.with(|p| { let mut p = p.borrow_mut(); p.clear(); p.extend(pending.parameters.iter().take(24)); });
            let mut host = FrontMacroHost {
                target,
                graphics,
                compute,
                stream: &mut shared.stream,
                memory: self.memory,
            };
            if let Err(error) =
                engine.execute(pending.method, &pending.parameters, data_ram, &mut host)
            {
                tracing::warn!(%error, method = pending.method, "macro call failed");
            }
            A0_MACRO.with(|m| m.set(0));
        }
        pending.parameters.clear();
    }
}

thread_local! {
    static A0_MACRO: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static A0_PARAMS: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn a0_check(engine: &str, regs: &[u32], method: u32) {
    if method != 0x6c3 {
        return;
    }
    {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        static C: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
        let zero = regs[0x6c0] == 0 && regs[0x6c1] == 0;
        let from_macro = A0_MACRO.with(|m| m.get()) != 0;
        let compute = engine == "compute";
        let total = C[0].fetch_add(1, Relaxed) + 1;
        if zero {
            C[1 + usize::from(compute) * 2 + usize::from(from_macro)].fetch_add(1, Relaxed);
        }
        if compute {
            C[5].fetch_add(1, Relaxed);
        }
        if total % 5000 == 0 {
            tracing::warn!(
                "DIAG a0 census triggers={total} compute_triggers={} zero_3d_pushbuffer={} zero_3d_macro={} zero_compute_pushbuffer={} zero_compute_macro={}",
                C[5].load(Relaxed), C[1].load(Relaxed), C[2].load(Relaxed), C[3].load(Relaxed), C[4].load(Relaxed)
            );
        }
        if !zero {
            return;
        }
    }
    static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if !crate::diag_ring::first(&N, 16) {
        return;
    }
    let m = A0_MACRO.with(|m| m.get());
    let params = A0_PARAMS.with(|p| format!("{:x?}", p.borrow()));
    tracing::warn!("DIAG a0 front {engine} report to 0 from {} d={:#x} params={}", if m == 0 { "pushbuffer".to_string() } else { format!("macro {:#x}", m) }, regs[0x6c3], if m == 0 { String::new() } else { params });
}

#[inline]
fn is_load_mme(method: u32) -> bool {
    (LOAD_MME_INSTRUCTION_RAM_POINTER..=LOAD_MME_START_ADDRESS_RAM).contains(&method)
}

/// Writes a 3D register in the mirror and sends it, unless it changed nothing: a state register
/// written with the value it holds, through shadow RAM, is dropped here (NX2_NOTES rows 135, 144).
/// The mirror also does what the back's engine does to registers on its own.
#[inline]
fn send_graphics(state: &mut RegisterState, stream: &mut Stream, method: u32, argument: u32) {
    let index = method as usize;
    let before = (state.regs.words()[index], state.shadow(method));
    match state.write(method, argument) {
        None if (state.regs.words()[index], state.shadow(method)) == before => return,
        None => {}
        Some(_) => {
            if method.wrapping_sub(CONST_BUFFER_DATA) < 16 {
                let offset = &mut state.regs.words_mut()[CONST_BUFFER_OFFSET];
                *offset = offset.wrapping_add(4);
            } else if method == FALCON_4 {
                state.regs.words_mut()[SHADOW_SCRATCH_0] = 1;
            }
        }
    }
    a0_check("3d", state.regs.words(), method);
    stream.engine_word(EngineKind::Maxwell3d, method, argument);
}

#[inline]
fn send_compute(registers: &mut [u32], stream: &mut Stream, method: u32, argument: u32) {
    if let Some(register) = registers.get_mut(method as usize) {
        *register = argument;
    }
    if registers.len() > 0x6c3 {
        a0_check("compute", registers, method);
    }
    stream.engine_word(EngineKind::Compute, method, argument);
}

/// What a macro on the front runs against: the calling engine's mirror, and the stream its
/// writes go to.
struct FrontMacroHost<'a> {
    target: MacroTarget,
    graphics: &'a mut RegisterState,
    compute: &'a mut [u32],
    stream: &'a mut Stream,
    memory: &'a Gmmu,
}

impl MacroHost for FrontMacroHost<'_> {
    #[inline]
    fn read_register(&mut self, method: u32) -> u32 {
        let registers: &[u32] = match self.target {
            MacroTarget::Graphics => self.graphics.regs.words(),
            MacroTarget::Compute => self.compute,
        };
        registers.get(method as usize).copied().unwrap_or(0)
    }

    #[inline]
    fn write_method(&mut self, method: u32, argument: u32) {
        if is_load_mme(method) {
            tracing::debug!(method, "macro writes LOAD_MME; ignored");
            return;
        }
        match self.target {
            MacroTarget::Graphics if method < MACRO_REGISTERS_START => {
                send_graphics(self.graphics, self.stream, method, argument);
            }
            MacroTarget::Graphics => {}
            MacroTarget::Compute => send_compute(self.compute, self.stream, method, argument),
        }
    }

    /// Reads memory as of the macro's position in the stream: the work sent before it completes
    /// and what the GPU wrote to the words reaches guest memory first.
    fn dma_read(&mut self, gpu_va: u64, out: &mut [u32]) {
        self.stream
            .synchronize_read(gpu_va, size_of_val(out) as u64);
        if let Err(fault) = self
            .memory
            .read_block(gpu_va, bytemuck::cast_slice_mut(out))
        {
            tracing::warn!(?fault, "MME DMA read from unmapped GPU memory");
            out.fill(0);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

    use super::*;
    use crate::back::Back;
    use crate::backend::{GpuHandles, GpuWrittenPages};
    use crate::clock::{GpuTimestamp, GuestTime};
    use crate::engines::create_engine;
    use crate::engines::testing::Recorder;
    use crate::gmmu::BigPageSize;
    use crate::pushbuffer::{CommandHeader, SubmissionMode};
    use crate::ring;
    use crate::stream::{FetchSyncs, Link, MIN_RING_SLOTS};

    #[test]
    fn macro_parameters_are_collected_without_allocating() {
        struct Boot;
        impl GuestTime for Boot {
            fn global_time_ns(&self) -> u64 {
                0
            }
        }
        let syncpoints = Arc::new(Syncpoints::new());
        let physical = Arc::new(PhysicalMemory::new(1 << 16).unwrap());
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        let handles = GpuHandles {
            lazy: Arc::new(crate::lazy::LazyDeviceRanges::new(device.clone())),
            device: device.clone(),
            syncpoints: syncpoints.clone(),
            timestamp: Arc::new(GpuTimestamp::new(Arc::new(Boot), false)),
            written_pages: Arc::new(GpuWrittenPages::new(36)),
            activity: Arc::new(crate::activity::Activity::new()),
        };
        let (producer, consumer) = ring::ring(MIN_RING_SLOTS, MIN_RING_SLOTS);
        let (side, side_receiver) = mpsc::channel();
        let inbox = Arc::new(Inbox::new());
        let fetch_syncs = Arc::new(FetchSyncs::new(inbox.clone()));
        let back = Back::new(
            Box::new(Recorder::default()),
            &handles,
            Arc::new(create_engine),
            side_receiver,
            fetch_syncs.clone(),
            Arc::new(crate::activity::Activity::new()),
            Arc::default(),
        );
        let mut shared = FrontShared {
            stream: Stream::new(
                producer,
                Link::Inline(Box::new((back, consumer))),
                side,
                fetch_syncs,
                Arc::new(crate::activity::Activity::new()),
            ),
            shared_macros: None,
            compute_macros: None,
            words: Vec::new(),
            waiters: Arc::default(),
            inbox,
            syncpoints,
            written_pages: None,
        };
        let config = ChannelConfig::new(
            Arc::new(Gmmu::new(device, BigPageSize::Size64K)),
            HostClass::Maxwell,
        );
        let mut channel = FrontChannel::open(ChannelId(0), &config, Arc::default(), &mut shared);
        let count = MAX_MACRO_PARAMETERS as u32;
        let mut words = vec![
            CommandHeader::new(SubmissionMode::Increasing, 0, 0, 1).0,
            0xb197,
            CommandHeader::new(SubmissionMode::IncreaseOnce, 0, MACRO_METHOD_BASE, count).0,
        ];
        words.extend(0..count);
        let storage = channel.state.calls[0].parameters.as_ptr();
        assert_eq!(channel.dispatch_words(&words, 0, &mut shared), words.len());
        let parameters = &channel.state.calls[0].parameters;
        assert!(parameters.is_empty());
        assert_eq!(parameters.as_ptr(), storage);
    }
}
