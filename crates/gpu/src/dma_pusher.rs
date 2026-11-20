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
use crate::channel::{
    Acquire, ChannelConfig, SemaphoreAddress, SemaphoreStatus, SemaphoreWaiters, SyncpointFence,
    SyncpointIncrement,
};
use crate::engines::EngineKind;
use crate::engines::kepler_compute::Regs as ComputeRegs;
use crate::engines::maxwell_3d::regs::{NUM_REGS, ShadowRamControl};
use crate::engines::maxwell_3d::{MACRO_REGISTERS_START, MethodSet, RegisterState, Regs};
use crate::gmmu::{Gmmu, GpuFault, RangeTarget};
use crate::gpu::{Inbox, LONG_WAIT};
use crate::host::{AcquireOp, FIRST_ENGINE_METHOD, HostAction, HostClass, HostState, ReleaseKind};
use crate::host1x::{Syncpoints, is_reached};
use crate::macros::{
    LOAD_MME_INSTRUCTION_RAM_POINTER, LOAD_MME_START_ADDRESS_RAM_POINTER, MACRO_METHOD_BASE,
    MacroDataRam, MacroEngine, MacroErrorKind, MacroHost, MacroIsa, ReadTaint, is_load_mme,
};
use crate::pushbuffer::{CommandParser, Flow, GpEntry, MethodCall, MethodHandler};
use crate::queue_trace::{Event, QueueTrace};
use crate::reg_index;
use crate::semaphore_mirror::SemaphoreMirror;
use crate::stream::{FetchSync, MULTI_MAX_WORDS, Stream};

/// Engines every channel has bound before any SET_OBJECT: NVN uses subchannel 2 as
/// inline-to-memory without binding it (NX2_NOTES row 49).
const DEFAULT_SUBCHANNELS: [(usize, EngineKind, u32); 1] =
    [(2, EngineKind::InlineToMemory, 0xa140)];

const CONST_BUFFER_DATA: u32 = reg_index!(Regs, const_buffer.buffer);
const CONST_BUFFER_OFFSET: usize = reg_index!(Regs, const_buffer.offset) as usize;
const FALCON_4: u32 = reg_index!(Regs, falcon) + 4;
const COMPUTE_DATA_UPLOAD: u32 = reg_index!(ComputeRegs, data_upload);
const SHADOW_SCRATCH_0: usize = reg_index!(Regs, shadow_scratch) as usize;
const SHADOW_RAM_CONTROL: u32 = reg_index!(Regs, shadow_ram_control);

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
    pub ordered_semaphore_acquires: bool,
    pub queue_trace: Option<Arc<QueueTrace>>,
    pub inbox: Arc<Inbox>,
    pub syncpoints: Arc<Syncpoints>,
    /// Switch 2: the device pages GPU work may have written. An entry on one is fetched after the
    /// back synchronised its words.
    pub written_pages: Option<Arc<GpuWrittenPages>>,
}

/// A channel's queued work.
pub(crate) enum Work {
    Entry(GpEntry),
    #[cfg(test)]
    Nvn2(Box<crate::back::Nvn2Work>),
    SyncpointWait(SyncpointFence),
    /// Increments, and the fence they reach.
    Increment(SyncpointIncrement, Option<SyncpointFence>, u64),
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
    trace_unready: u64,
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
    back_macros: BackMacros,
}

/// Switch 2: 3D macro calls that read guest memory run on the back, at their point of the stream,
/// so the front never waits for the work before them. The front runs each such call first
/// without the words it reads ([`MacroEngine::shadow_execute`]): when no value from them steers
/// the call and it never uses the data RAM, the mirror takes the writes whose values it knows,
/// and the registers written with values from the reads become the back's.
struct BackMacros {
    /// 3D registers a macro on the back wrote with values only the back knows: the mirror's
    /// copies are stale until the front writes them or fetches them from the back.
    owned: MethodSet,
    /// The code generation of the 3D macros the back holds for this channel.
    code: Option<u64>,
    /// Start addresses whose calls run on the front, and those that ran on the back, for code
    /// generation `generation`.
    refused: Vec<u32>,
    sent: Vec<u32>,
    generation: u64,
    /// What the shadow run being made changed, to undo it.
    undo: Vec<Undo>,
}

#[derive(Clone, Copy)]
enum Undo {
    Word(u32, (u32, u32)),
    Owned(u32),
    Disowned(u32),
}

impl BackMacros {
    fn new() -> Self {
        Self {
            owned: MethodSet::empty(),
            code: None,
            refused: Vec::new(),
            sent: Vec::new(),
            generation: 0,
            undo: Vec::new(),
        }
    }

    /// Sends the call to the back if it can run there; false leaves it to the front.
    fn try_send(
        &mut self,
        engine: &mut MacroEngine,
        call: &MacroCall,
        graphics: &mut RegisterState,
        data_ram: &mut MacroDataRam,
        stream: &mut Stream,
    ) -> bool {
        if engine.isa() != MacroIsa::Mme64
            || call.parameters.len() > MULTI_MAX_WORDS
            || !engine.reads_memory(call.method)
        {
            return false;
        }
        let generation = engine.code_generation();
        if self.generation != generation {
            self.refused.clear();
            self.sent.clear();
            self.generation = generation;
        }
        let start = engine.start_of(call.method);
        if self.refused.contains(&start) {
            return false;
        }
        self.undo.clear();
        let mut taint = ReadTaint::without_data_ram();
        let mut paths = 0;
        let mut too_many = false;
        let outcome = loop {
            paths += 1;
            let mut host = ShadowHost {
                graphics: &mut *graphics,
                owned: &mut self.owned,
                undo: &mut self.undo,
                replay: paths > 1,
                blocked: false,
            };
            let result = engine.shadow_execute(
                call.method,
                &call.parameters,
                data_ram,
                &mut host,
                &mut taint,
            );
            match result {
                Err(error) => break Err(Some(error)),
                Ok(()) if host.blocked => break Err(None),
                Ok(()) => {}
            }
            match taint.next_path() {
                None => break Ok(()),
                Some(_) if paths == MAX_SHADOW_PATHS => {
                    too_many = true;
                    break Err(None);
                }
                Some(next) => taint = next,
            }
        };
        if let Err(error) = outcome {
            if too_many {
                tracing::info!(
                    start,
                    "3D macro reads memory and has too many ways to run on the back"
                );
                self.refused.push(start);
            } else if let Some(error) = error
                && matches!(
                    error.kind,
                    MacroErrorKind::DependsOnRead | MacroErrorKind::UsesDataRam
                )
            {
                tracing::info!(%error, "3D macro reads memory and runs on the front");
                self.refused.push(start);
            }
            self.undo(graphics);
            return false;
        }
        let code = match self.code {
            Some(sent) if sent == generation => None,
            _ => engine.code_copy(),
        };
        if let Some(code) = code {
            stream.side(SideCommand::MacroCode(Box::new(code)));
            self.code = Some(generation);
        }
        if !self.sent.contains(&start) {
            self.sent.push(start);
            tracing::info!(start, paths, "3D macro reads memory and runs on the back");
        }
        stream.macro_call(EngineKind::Maxwell3d, start, &call.parameters);
        true
    }

    fn undo(&mut self, graphics: &mut RegisterState) {
        for undo in self.undo.drain(..).rev() {
            match undo {
                Undo::Word(method, word) => graphics.restore(method, word),
                Undo::Owned(method) => self.owned.remove(method),
                Undo::Disowned(method) => self.owned.insert(method),
            }
        }
    }
}

/// The ways through branches on read values a shadow run follows before a call stays on the front.
const MAX_SHADOW_PATHS: usize = 16;

/// Whether the back holds register `method` (see [`BackMacros::owned`]).
#[inline]
fn back_owned(owned: &MethodSet, method: u32) -> bool {
    (method as usize) < NUM_REGS && owned.contains(method)
}

/// Brings the mirror's copies of the registers the back holds up to date, waiting for the back
/// to reach this point of the stream. Rare: macros seldom read what a macro on the back wrote.
#[cold]
fn fetch_back_owned(graphics: &mut RegisterState, owned: &mut MethodSet, stream: &mut Stream) {
    static FETCHES: crate::throttle::Throttle = crate::throttle::Throttle::new();
    let fetched = stream.fetch_registers(|words| {
        owned.drain(|method| words.push((method, graphics.word(method))));
        if let Some(dropped) = FETCHES.allow() {
            let methods: Vec<u32> = words.iter().map(|&(method, _)| method).collect();
            tracing::info!(
                ?methods,
                dropped,
                "front fetches registers a macro on the back wrote"
            );
        }
    });
    for &(method, word) in fetched.iter() {
        graphics.restore(method, word);
    }
}

/// Whether a write takes register `method` back from the back ([`BackMacros::owned`]): only
/// tracking shadow RAM modes write both the register and its shadow RAM copy.
#[inline]
fn write_owns(state: &RegisterState) -> bool {
    matches!(
        state.shadow_ram_control(),
        ShadowRamControl::TRACK | ShadowRamControl::TRACK_WITH_FILTER
    )
}

/// A shadow run on the front (see [`BackMacros`]): writes go to the mirror with an undo record, and
/// none reaches the stream.
struct ShadowHost<'a> {
    graphics: &'a mut RegisterState,
    owned: &'a mut MethodSet,
    undo: &'a mut Vec<Undo>,
    /// A later way through the call (see [`ReadTaint::next_path`]): its writes before its first
    /// branch on a read are the ones the first way already applied.
    replay: bool,
    /// The call read a register the back holds or never receives (LOAD_MME), so its run here may
    /// differ from the back's.
    blocked: bool,
}

impl ShadowHost<'_> {
    fn save(&mut self, method: u32) {
        self.undo
            .push(Undo::Word(method, self.graphics.word(method)));
    }
}

impl MacroHost for ShadowHost<'_> {
    fn read_register(&mut self, method: u32) -> u32 {
        self.blocked |= is_load_mme(method) || back_owned(self.owned, method);
        self.graphics
            .regs
            .words()
            .get(method as usize)
            .copied()
            .unwrap_or(0)
    }

    fn write_method(&mut self, method: u32, argument: u32) {
        if self.replay || is_load_mme(method) || method >= MACRO_REGISTERS_START {
            return;
        }
        self.save(method);
        if method.wrapping_sub(CONST_BUFFER_DATA) < 16 {
            self.save(CONST_BUFFER_OFFSET as u32);
        } else if method == FALCON_4 {
            self.save(SHADOW_SCRATCH_0 as u32);
        }
        if self.owned.contains(method) && write_owns(self.graphics) {
            self.owned.remove(method);
            self.undo.push(Undo::Disowned(method));
        }
        mirror_graphics(self.graphics, method, argument);
    }

    fn dma_read(&mut self, _gpu_va: u64, _out: &mut [u32]) {}

    /// The value only the back will know: the register becomes the back's, unless the front
    /// itself needs the value (a shadow RAM mode, a constant buffer position or a scratch flag
    /// the mirror keeps).
    fn write_unknown(&mut self, method: u32) -> bool {
        if is_load_mme(method) || method >= MACRO_REGISTERS_START {
            return true;
        }
        if method == SHADOW_RAM_CONTROL
            || method == FALCON_4
            || method == CONST_BUFFER_OFFSET as u32
            || method.wrapping_sub(CONST_BUFFER_DATA) < 16
        {
            return false;
        }
        if !self.owned.contains(method) {
            self.owned.insert(method);
            self.undo.push(Undo::Owned(method));
        }
        true
    }
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
                back_macros: BackMacros::new(),
            },
            queue: VecDeque::new(),
            offset: 0,
            wait: None,
            fetch_synced: false,
            waiting_since: None,
            wait_reported: false,
            trace_unready: 0,
        }
    }

    /// Removes the channel from the back.
    pub(crate) fn close(self, shared: &mut FrontShared) {
        if let Some(trace) = &shared.queue_trace {
            trace.record(Event::Close(self.id.0));
        }
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

    /// Whether the channel waits on a semaphore acquire.
    pub(crate) fn waits_on_semaphore(&self) -> bool {
        matches!(self.wait, Some(Wait::Semaphore { .. }))
    }

    /// Whether the channel is parked until the back synchronised an entry's fetch.
    pub(crate) fn waits_for_fetch(&self) -> bool {
        matches!(self.wait, Some(Wait::Fetch(_)))
    }

    /// Whether the channel has work it can run now.
    pub(crate) fn is_runnable(&self, shared: &FrontShared) -> bool {
        match self.wait {
            None => !self.queue.is_empty(),
            Some(wait) => self.wait_dependency(wait, shared).is_some(),
        }
    }

    pub(crate) fn probe_runnable(&mut self, shared: &FrontShared) -> bool {
        let probe = self
            .wait
            .and(shared.queue_trace.as_deref())
            .map_or(0, QueueTrace::now);
        let ready = self.is_runnable(shared);
        if !ready && probe != 0 {
            self.trace_unready = probe;
        }
        ready
    }

    fn wait_dependency(&self, wait: Wait, shared: &FrontShared) -> Option<u64> {
        match wait {
            Wait::Semaphore {
                address,
                payload,
                op,
                wide,
                ..
            } => match shared.waiters.status(self.id) {
                SemaphoreStatus::Met { dependency } => Some(dependency),
                SemaphoreStatus::Waiting => None,
                SemaphoreStatus::Memory => semaphore_met(
                    &self.memory,
                    shared.syncpoints.semaphore_mirror(),
                    address,
                    payload,
                    op,
                    wide,
                )
                .then_some(0),
            },
            Wait::Syncpoint(fence) => shared
                .syncpoints
                .host()
                .is_reached(fence.id, fence.value)
                .unwrap_or(true)
                .then_some(0),
            Wait::Fetch(ticket) => shared.stream.fetch_synced(ticket).then_some(0),
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
                    latched = matches!(shared.waiters.status(self.id), SemaphoreStatus::Met { .. }),
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
            let probe = shared.queue_trace.as_deref().map_or(0, QueueTrace::now);
            let Some(dependency) = self.wait_dependency(wait, shared) else {
                self.trace_unready = probe;
                return;
            };
            if let Some(trace) = &shared.queue_trace {
                trace.record(Event::Resume(
                    self.id.0,
                    std::mem::take(&mut self.trace_unready),
                ));
            }
            self.wait = None;
            self.waiting_since = None;
            self.wait_reported = false;
            shared.waiters.remove(self.id);
            if dependency != 0 {
                shared.stream.select(self.id);
                shared.stream.semaphore_dependency(dependency);
            }
        }
        if self.queue.is_empty() {
            return;
        }
        if let Some(trace) = &shared.queue_trace {
            trace.record(Event::Run(self.id.0));
        }
        shared.stream.select(self.id);
        while let Some(work) = self.queue.front() {
            match *work {
                Work::Entry(entry) => self.run_entry(entry, shared),
                #[cfg(test)]
                Work::Nvn2(_) => {
                    let Some(Work::Nvn2(work)) = self.queue.pop_front() else {
                        unreachable!("the queue's front was just matched");
                    };
                    shared.stream.side(SideCommand::Nvn2(self.id, work));
                }
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
                Work::Increment(increment, fence, sample) => {
                    self.queue.pop_front();
                    if sample != 0
                        && let Some(trace) = &shared.queue_trace
                    {
                        trace.record(Event::Front(sample));
                    }
                    if let Some(fence) = fence
                        && let Some(latency) = shared.syncpoints.fence_latency()
                    {
                        latency.front(fence);
                    }
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
        if let Some(trace) = &shared.queue_trace {
            trace.record(Event::Stop(self.id.0));
        }
    }

    fn run_entry(&mut self, entry: GpEntry, shared: &mut FrontShared) {
        let length = entry.words();
        let start = self.offset;
        if start >= length {
            self.queue.pop_front();
            self.offset = 0;
            return;
        }
        if !self.fetch_synced {
            let sync = if self.on_written_pages(entry, shared) {
                Some(FetchSync::Host)
            } else if start == 0 && entry.sync() {
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
                if let Some(trace) = &shared.queue_trace {
                    trace.record(Event::Wait(self.id.0, 'f', ticket, ticket));
                }
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

    /// An acquire the word in memory does not meet yet suspends the channel (NX2_NOTES row 50)
    /// without waiting for the back: a release still queued for it latches the channel when it
    /// lands, and the front parses the other channels meanwhile. The channel waits from before
    /// its look, so a release that lands and is overwritten at any point after it still latches
    /// it.
    fn acquire(
        &mut self,
        method: u32,
        address: GpuVAddr,
        payload: u64,
        op: AcquireOp,
        wide: bool,
    ) -> Flow {
        let ordered =
            self.shared.ordered_semaphore_acquires && self.state.host.class() == HostClass::Ampere;
        let device = SemaphoreAddress {
            low: self.memory.translate(address),
            high: wide.then(|| self.memory.translate(address + 4)).flatten(),
        };
        let trace = self.shared.queue_trace.as_deref();
        let trace_id = trace.map_or(0, QueueTrace::id);
        if ordered || device.low.is_some() {
            self.shared.waiters.add(
                self.id,
                Acquire {
                    address: device,
                    payload,
                    op,
                    wide,
                },
                ordered,
                trace_id,
            );
        }
        if ordered {
            self.shared.stream.acquire(address, wide);
        } else if semaphore_met(
            self.memory,
            self.shared.syncpoints.semaphore_mirror(),
            address,
            payload,
            op,
            wide,
        ) {
            self.shared.waiters.remove(self.id);
            return Flow::Continue;
        } else {
            self.shared.stream.release_fences();
        }
        if let Some(trace) = trace {
            trace.record(Event::Acquire(
                trace_id,
                device.contiguous(wide),
                payload,
                op as u8,
                wide,
            ));
            trace.record(Event::Wait(self.id.0, 's', trace_id, trace_id));
        }
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
        if let Some(trace) = &self.shared.queue_trace {
            let key = u64::from(fence.id) << 32 | u64::from(fence.value);
            trace.record(Event::Wait(self.id.0, 'y', trace.id(), key));
        }
        let _ = bank.register_action(fence.id, fence.value, move || inbox.wake());
        *self.wait = Some(Wait::Syncpoint(fence));
        Flow::Suspend
    }

    fn wait_for_idle(&mut self) {
        self.shared.stream.wait_for_idle();
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
            &mut state.back_macros.owned,
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
            back_macros,
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
            if target == MacroTarget::Graphics
                && host.class() == HostClass::Ampere
                && back_macros.try_send(engine, pending, graphics, data_ram, &mut shared.stream)
            {
                pending.parameters.clear();
                return;
            }
            let mut host = FrontMacroHost {
                target,
                graphics,
                owned: &mut back_macros.owned,
                compute,
                stream: &mut shared.stream,
                memory: self.memory,
            };
            if let Err(error) =
                engine.execute(pending.method, &pending.parameters, data_ram, &mut host)
            {
                tracing::warn!(%error, method = pending.method, "macro call failed");
            }
        }
        pending.parameters.clear();
    }
}

/// Writes a 3D register in the mirror and sends it, unless it changed nothing: a state register
/// written with the value it holds, through shadow RAM, is dropped here (NX2_NOTES rows 135, 144).
/// The mirror also does what the back's engine does to registers on its own.
/// A register the back holds ([`BackMacros::owned`]) is always sent.
#[inline]
fn send_graphics(
    state: &mut RegisterState,
    owned: &mut MethodSet,
    stream: &mut Stream,
    method: u32,
    argument: u32,
) {
    let changed = mirror_graphics(state, method, argument);
    if owned.contains(method) {
        if write_owns(state) {
            owned.remove(method);
        }
    } else if !changed {
        return;
    }
    stream.engine_word(EngineKind::Maxwell3d, method, argument);
}

/// Writes a 3D register in the mirror as [`send_graphics`] does; returns whether the write
/// changed anything or runs an action.
#[inline]
fn mirror_graphics(state: &mut RegisterState, method: u32, argument: u32) -> bool {
    let index = method as usize;
    let before = (state.regs.words()[index], state.shadow(method));
    match state.write(method, argument) {
        None => (state.regs.words()[index], state.shadow(method)) != before,
        Some(_) => {
            if method.wrapping_sub(CONST_BUFFER_DATA) < 16 {
                let offset = &mut state.regs.words_mut()[CONST_BUFFER_OFFSET];
                *offset = offset.wrapping_add(4);
            } else if method == FALCON_4 {
                state.regs.words_mut()[SHADOW_SCRATCH_0] = 1;
            }
            true
        }
    }
}

#[inline]
fn send_compute(registers: &mut [u32], stream: &mut Stream, method: u32, argument: u32) {
    if let Some(register) = registers.get_mut(method as usize) {
        *register = argument;
    }
    stream.engine_word(EngineKind::Compute, method, argument);
}

/// What a macro on the front runs against: the calling engine's mirror, and the stream its
/// writes go to.
struct FrontMacroHost<'a> {
    target: MacroTarget,
    graphics: &'a mut RegisterState,
    owned: &'a mut MethodSet,
    compute: &'a mut [u32],
    stream: &'a mut Stream,
    memory: &'a Gmmu,
}

impl MacroHost for FrontMacroHost<'_> {
    #[inline]
    fn read_register(&mut self, method: u32) -> u32 {
        if self.target == MacroTarget::Graphics && back_owned(self.owned, method) {
            fetch_back_owned(self.graphics, self.owned, self.stream);
        }
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
                send_graphics(self.graphics, self.owned, self.stream, method, argument);
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
        self.stream.overlay_read(out);
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
        let fetch_syncs = Arc::new(FetchSyncs::new(inbox.clone(), None));
        let back = Back::new(
            Box::new(Recorder::default()),
            &handles,
            Arc::new(create_engine),
            side_receiver,
            fetch_syncs.clone(),
            Arc::new(crate::activity::Activity::new()),
            Arc::default(),
            Arc::new(crate::cpu_reads::CpuReads::new(Box::new(|| {}))),
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
            ordered_semaphore_acquires: false,
            queue_trace: None,
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
