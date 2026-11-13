//! The GPU: channels, the command queue from the CPU side, and the front/back thread split
//! (NX2_GPU_THREAD_SPLIT.md).
//!
//! The front thread takes submissions and other CPU-side commands in order, parses command
//! streams, executes host methods and runs macros. Everything that touches engines or the
//! renderer goes to the back through one ring, in order. [`Threading::Single`] runs the same
//! front and back on one thread: the back then runs whenever the front waits for it.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread::{JoinHandle, ThreadId};
use std::time::Duration;

use lichii_common::DAddr;
use parking_lot::{Condvar, Mutex};

use crate::back::{Back, SideCommand};
use crate::backend::{ChannelId, FramebufferConfig, GpuHandles, Rasterizer};
use crate::channel::{ChannelConfig, GpEntry, Submission, SyncpointFence, SyncpointIncrement};
use crate::clock::{GpuTimestamp, GuestTime};
use crate::dma_pusher::{FrontChannel, FrontShared, Work};
use crate::engines::{EngineFactory, create_engine};
use crate::gmmu::{BigPageSize, Gmmu};
use crate::host1x::Host1x;
use crate::nx2_memory::{AddressSpaceId, Nx2GpuMemory};
use crate::ring;
use crate::stream::{FetchSyncs, Link, MIN_RING_SLOTS, Stream};

/// Default ring size: 8 MiB of slots.
pub const DEFAULT_RING_SLOTS: usize = 1 << 20;
/// How long the front first sleeps while every channel with work waits, doubling up to
/// [`MAX_POLL`]. A semaphore write that meets a waiter, a syncpoint increment or [`Gpu::wake`]
/// ends the sleep at once; the poll only catches memory the guest CPU wrote. An idle back polls
/// the renderer's pending fences at the same pace.
pub(crate) const MIN_POLL: Duration = Duration::from_micros(50);
pub(crate) const MAX_POLL: Duration = Duration::from_millis(1);
/// How long a channel stays suspended, or the front waits for the back, before it is reported.
pub(crate) const LONG_WAIT: Duration = Duration::from_secs(2);

/// Where the back half runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Threading {
    /// Front and back on their own threads.
    #[default]
    Split,
    /// Both on one GPU thread, through the same ring.
    Single,
}

/// The console whose GPU runs: Switch 1 (Maxwell) or Switch 2 (Ampere/NVN2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Console {
    Switch1,
    Switch2,
}

/// How the GPU is built.
#[derive(Clone)]
pub struct GpuConfig {
    /// Picks the timestamp clock, the Switch 2 memory registry and the channel defaults.
    pub console: Console,
    /// Split (the default) or one GPU thread.
    pub threading: Threading,
    /// Slots of the front-to-back ring (rounded up to a power of two).
    pub ring_slots: usize,
    /// Creates the engines channels bind.
    pub engines: EngineFactory,
}

impl GpuConfig {
    /// The defaults for `console`: split threads, [`DEFAULT_RING_SLOTS`], [`create_engine`].
    pub fn new(console: Console) -> Self {
        Self {
            console,
            threading: Threading::default(),
            ring_slots: DEFAULT_RING_SLOTS,
            engines: Arc::new(create_engine),
        }
    }
}

enum CpuCommand {
    CreateChannel(ChannelId, ChannelConfig, Arc<AtomicU32>),
    DestroyChannel(ChannelId),
    BindAddressSpace(ChannelId, Arc<Gmmu>),
    Submit(ChannelId, QueuedSubmission),
    Side(SideCommand),
    Synchronize(mpsc::SyncSender<()>),
}

/// A [`Submission`] on its way to the front, its entries copied into a buffer of the inbox's
/// pool.
struct QueuedSubmission {
    entries: Vec<GpEntry>,
    wait: Option<SyncpointFence>,
    increment: Option<SyncpointIncrement>,
}

/// Entry buffers the inbox keeps for later submissions; more are dropped once the front is done
/// with them.
const POOLED_ENTRY_BUFFERS: usize = 16;

/// The CPU side's queue to the front, and the front's wake-up.
pub(crate) struct Inbox {
    state: Mutex<InboxState>,
    cond: Condvar,
}

struct InboxState {
    commands: VecDeque<CpuCommand>,
    woken: bool,
    paused: bool,
    closed: bool,
    entry_buffers: Vec<Vec<GpEntry>>,
}

impl Inbox {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(InboxState {
                commands: VecDeque::new(),
                woken: false,
                paused: false,
                closed: false,
                entry_buffers: Vec::with_capacity(POOLED_ENTRY_BUFFERS),
            }),
            cond: Condvar::new(),
        }
    }

    fn push(&self, command: CpuCommand) {
        self.push_with(|_| Some(command));
    }

    /// Queues the command `make` returns, if any. `make` runs under the inbox lock, so what it
    /// does is ordered like the queue, and may take a buffer from the entry pool; it does not run
    /// once the inbox is closed.
    fn push_with(&self, make: impl FnOnce(&mut Vec<Vec<GpEntry>>) -> Option<CpuCommand>) {
        let mut state = self.state.lock();
        if state.closed {
            return;
        }
        if let Some(command) = make(&mut state.entry_buffers) {
            state.commands.push_back(command);
            self.cond.notify_one();
        }
    }

    /// Returns a submission's entry buffer to the pool.
    fn recycle(&self, mut entries: Vec<GpEntry>) {
        let mut state = self.state.lock();
        if state.entry_buffers.len() < POOLED_ENTRY_BUFFERS {
            entries.clear();
            state.entry_buffers.push(entries);
        }
    }

    /// Has the front look at its suspended channels again.
    pub(crate) fn wake(&self) {
        self.state.lock().woken = true;
        self.cond.notify_one();
    }

    /// Ends the front; commands still queued are dropped.
    fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        state.commands.clear();
        self.cond.notify_one();
    }

    /// Holds the commands, and the front, while `paused`.
    fn pause(&self, paused: bool) {
        self.state.lock().paused = paused;
        self.cond.notify_one();
    }

    /// Moves the queued commands into `into` (empty), first sleeping until there are some, a
    /// wake, or `timeout`, and for as long as the inbox is paused. Returns whether it is closed.
    fn take(&self, into: &mut VecDeque<CpuCommand>, timeout: Option<Duration>) -> bool {
        let mut state = self.state.lock();
        loop {
            if state.closed || (!state.paused && (!state.commands.is_empty() || state.woken)) {
                break;
            }
            match timeout {
                Some(timeout) if !state.paused => {
                    if self.cond.wait_for(&mut state, timeout).timed_out() && !state.paused {
                        break;
                    }
                }
                _ => self.cond.wait(&mut state),
            }
        }
        state.woken = false;
        std::mem::swap(&mut state.commands, into);
        state.closed
    }
}

/// The front thread's state.
struct Front {
    shared: FrontShared,
    channels: Vec<Option<FrontChannel>>,
    /// Commands not handled yet. A side command waits here while a channel is parked on a
    /// fetch synchronisation, so it cannot overtake that channel's earlier entries, and every
    /// command after it waits too.
    held: VecDeque<CpuCommand>,
}

impl Front {
    fn run(mut self, inbox: Arc<Inbox>) {
        struct Closed<'a>(&'a Inbox);
        impl Drop for Closed<'_> {
            fn drop(&mut self) {
                self.0.close();
            }
        }
        let _closed = Closed(&inbox);
        let mut batch = VecDeque::new();
        let mut poll = MIN_POLL;
        let mut fences = false;
        loop {
            let waiting = self.channels.iter().flatten().any(FrontChannel::is_waiting);
            let closed = inbox.take(&mut batch, (waiting || fences).then_some(poll));
            let busy = !batch.is_empty();
            self.held.append(&mut batch);
            self.handle_held();
            let resumed = self.resume();
            if resumed {
                self.handle_held();
            }
            fences = self.shared.stream.idle();
            if closed {
                break;
            }
            if busy || resumed {
                poll = MIN_POLL;
            } else if waiting || fences {
                if waiting && poll == MAX_POLL {
                    self.shared.stream.release_fences();
                }
                poll = (poll * 2).min(MAX_POLL);
            }
        }
        self.shared.stream.close();
    }

    fn handle(&mut self, command: CpuCommand) {
        match command {
            CpuCommand::CreateChannel(id, config, reference) => {
                let index = id.0 as usize;
                if self.channels.len() <= index {
                    self.channels.resize_with(index + 1, || None);
                }
                self.channels[index] =
                    Some(FrontChannel::open(id, &config, reference, &mut self.shared));
            }
            CpuCommand::DestroyChannel(id) => {
                if let Some(channel) = self.channels.get_mut(id.0 as usize).and_then(Option::take) {
                    channel.close(&mut self.shared);
                }
            }
            CpuCommand::BindAddressSpace(id, memory) => {
                let Some(channel) = self
                    .channels
                    .get_mut(id.0 as usize)
                    .and_then(Option::as_mut)
                else {
                    tracing::warn!(?id, "address space bound to a channel that does not exist");
                    return;
                };
                channel.push(Work::BindAddressSpace(memory));
                channel.run(&mut self.shared);
            }
            CpuCommand::Submit(id, submission) => {
                let QueuedSubmission {
                    entries,
                    wait,
                    increment,
                } = submission;
                let Some(channel) = self
                    .channels
                    .get_mut(id.0 as usize)
                    .and_then(Option::as_mut)
                else {
                    tracing::warn!(?id, "submission to a channel that does not exist");
                    self.shared.inbox.recycle(entries);
                    return;
                };
                if let Some(fence) = wait {
                    channel.push(Work::SyncpointWait(fence));
                }
                for &entry in &entries {
                    channel.push(Work::Entry(entry));
                }
                self.shared.inbox.recycle(entries);
                if let Some(increment) = increment {
                    channel.push(Work::Increment(increment));
                }
                channel.push(Work::EndOfSubmission);
                channel.run(&mut self.shared);
            }
            CpuCommand::Side(command) => self.shared.stream.side(command),
            CpuCommand::Synchronize(done) => {
                loop {
                    self.shared.stream.drain();
                    if !self.resume() {
                        break;
                    }
                }
                let _ = done.send(());
            }
        }
    }

    /// Handles the held commands in queue order. A side command stops the queue while a channel
    /// is parked on a fetch synchronisation, because it must follow that channel's earlier
    /// entries; the commands after it wait too, so that a later submission cannot land before
    /// it. The wait ends once the back reaches the synchronisation.
    fn handle_held(&mut self) {
        while let Some(command) = self.held.pop_front() {
            if matches!(command, CpuCommand::Side(_))
                && self
                    .channels
                    .iter()
                    .flatten()
                    .any(FrontChannel::waits_for_fetch)
            {
                self.held.push_front(command);
                return;
            }
            self.handle(command);
        }
    }

    /// Runs the channels whose wait is over. Returns whether any ran.
    fn resume(&mut self) -> bool {
        let mut resumed = false;
        for channel in self.channels.iter_mut().flatten() {
            if channel.is_runnable(&self.shared) {
                channel.run(&mut self.shared);
                resumed = true;
            } else {
                channel.report_long_wait(&self.shared);
            }
        }
        resumed
    }
}

/// The emulated GPU. Every method may be called from any thread and none waits for GPU work
/// unless it says so; work is executed in the order it was queued.
pub struct Gpu {
    host1x: Host1x,
    handles: GpuHandles,
    nx2_memory: Option<Nx2GpuMemory>,
    /// Switch 2 address spaces registered with `nx2_memory`.
    address_spaces: Mutex<Vec<(Arc<Gmmu>, AddressSpaceId)>>,
    inbox: Arc<Inbox>,
    next_channel: AtomicU32,
    references: Mutex<HashMap<ChannelId, Arc<AtomicU32>>>,
    /// The front thread, then the back thread when it has one; emptied by `shutdown`.
    threads: Mutex<Vec<JoinHandle<()>>>,
    /// Ids of those threads, so `shutdown` recognises a GPU thread without taking `threads`.
    thread_ids: Vec<ThreadId>,
}

const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Gpu>();
};

impl Gpu {
    /// Starts the GPU threads for `config.console`. `renderer` is called once, on this thread,
    /// with the handles the GPU shares with it; the rasterizer it returns moves to the back
    /// thread, the only one that calls it.
    pub fn new(
        host1x: Host1x,
        time: Arc<dyn GuestTime>,
        config: GpuConfig,
        renderer: impl FnOnce(GpuHandles) -> Box<dyn Rasterizer>,
    ) -> Self {
        let handles = GpuHandles {
            device: host1x.memory.clone(),
            syncpoints: host1x.syncpoints.clone(),
            timestamp: Arc::new(GpuTimestamp::new(time, config.console == Console::Switch2)),
            written_pages: host1x.written_pages.clone(),
            lazy: host1x.lazy.clone(),
        };
        let nx2_memory = (config.console == Console::Switch2).then(|| {
            Nx2GpuMemory::new(
                host1x.memory.clone(),
                host1x.allocator.clone(),
                host1x.syncpoints.semaphore_mirror().clone(),
                host1x.lazy.clone(),
            )
        });
        let rasterizer = renderer(handles.clone());
        let inbox = Arc::new(Inbox::new());
        handles.syncpoints.semaphore_waiters().attach(&inbox);
        let (producer, consumer) = ring::ring(config.ring_slots, MIN_RING_SLOTS);
        let (side_sender, side_receiver) = mpsc::channel();
        let fetch_syncs = Arc::new(FetchSyncs::new(inbox.clone()));
        let back = Back::new(
            rasterizer,
            &handles,
            config.engines,
            side_receiver,
            fetch_syncs.clone(),
        );
        let (link, back) = match config.threading {
            Threading::Split => {
                let thread = std::thread::Builder::new()
                    .name("gpu-back".into())
                    .spawn(move || back.run_thread(consumer))
                    .expect("failed to start the GPU back thread");
                (Link::Thread, Some(thread))
            }
            Threading::Single => (Link::Inline(Box::new((back, consumer))), None),
        };
        let front = Front {
            shared: FrontShared {
                stream: Stream::new(producer, link, side_sender, fetch_syncs),
                shared_macros: None,
                compute_macros: None,
                words: Vec::new(),
                waiters: handles.syncpoints.semaphore_waiters().clone(),
                inbox: inbox.clone(),
                syncpoints: handles.syncpoints.clone(),
                written_pages: (config.console == Console::Switch2)
                    .then(|| handles.written_pages.clone()),
            },
            channels: Vec::new(),
            held: VecDeque::new(),
        };
        let front_inbox = inbox.clone();
        let front = std::thread::Builder::new()
            .name("gpu-front".into())
            .spawn(move || front.run(front_inbox))
            .expect("failed to start the GPU front thread");
        Self {
            host1x,
            handles,
            nx2_memory,
            address_spaces: Mutex::new(Vec::new()),
            inbox,
            next_channel: AtomicU32::new(0),
            references: Mutex::new(HashMap::new()),
            thread_ids: [Some(&front), back.as_ref()]
                .into_iter()
                .flatten()
                .map(|thread| thread.thread().id())
                .collect(),
            threads: Mutex::new([Some(front), back].into_iter().flatten().collect()),
        }
    }

    /// The host1x the GPU was built on.
    pub fn host1x(&self) -> &Host1x {
        &self.host1x
    }

    /// What the GPU shares with the renderer: host1x's device memory and syncpoints, and the
    /// console's timestamp clock.
    pub fn handles(&self) -> &GpuHandles {
        &self.handles
    }

    /// The Switch 2 GPU memory registry (gsv gpu sessions); `None` on Switch 1.
    pub fn nx2_memory(&self) -> Option<&Nx2GpuMemory> {
        self.nx2_memory.as_ref()
    }

    /// A new GPU address space on host1x's device memory. On Switch 2 it receives the mappings
    /// of [`Gpu::nx2_memory`], the ones made before included, until it is destroyed.
    pub fn create_address_space(&self, big_page: BigPageSize) -> Arc<Gmmu> {
        let gmmu = Arc::new(Gmmu::new(self.host1x.memory.clone(), big_page));
        if let Some(nx2) = &self.nx2_memory {
            let id = nx2.register_address_space(gmmu.clone());
            self.address_spaces.lock().push((gmmu.clone(), id));
        }
        gmmu
    }

    /// Stops sending [`Gpu::nx2_memory`] mappings to `gmmu`. Channels still bound to it keep
    /// using it.
    pub fn destroy_address_space(&self, gmmu: &Arc<Gmmu>) {
        let Some(nx2) = &self.nx2_memory else {
            return;
        };
        let mut spaces = self.address_spaces.lock();
        if let Some(index) = spaces
            .iter()
            .position(|(space, _)| Arc::ptr_eq(space, gmmu))
        {
            nx2.unregister_address_space(spaces.swap_remove(index).1);
        }
    }

    /// Creates a channel; usable at once.
    pub fn create_channel(&self, config: ChannelConfig) -> ChannelId {
        let id = ChannelId(self.next_channel.fetch_add(1, Ordering::Relaxed));
        let reference = Arc::new(AtomicU32::new(0));
        self.references.lock().insert(id, reference.clone());
        self.inbox
            .push(CpuCommand::CreateChannel(id, config, reference));
        id
    }

    /// Makes the channel's work queued after this call use `gmmu`; work queued before keeps the
    /// address space it was queued with, also on the back.
    pub fn bind_address_space(&self, channel: ChannelId, gmmu: Arc<Gmmu>) {
        self.inbox.push(CpuCommand::BindAddressSpace(channel, gmmu));
    }

    /// Destroys a channel once the work queued before has run (a suspended channel's remaining
    /// work is dropped).
    pub fn destroy_channel(&self, channel: ChannelId) {
        self.inbox.push_with(|_| {
            self.references.lock().remove(&channel);
            Some(CpuCommand::DestroyChannel(channel))
        });
    }

    /// Queues a GPFIFO submission without waiting for it. Reserves the submission's syncpoint
    /// increments in [`Syncpoints::increase_max`](crate::host1x::Syncpoints::increase_max), in
    /// the order submissions are queued, and returns the fence they reach; `None` without
    /// increments or for an invalid syncpoint. A submission to a destroyed channel is dropped
    /// and reserves nothing. The entries are copied into a buffer the GPU reuses, so a submit
    /// allocates nothing once the pool is warm.
    pub fn submit(&self, channel: ChannelId, submission: Submission<'_>) -> Option<SyncpointFence> {
        let mut fence = None;
        self.inbox.push_with(|pool| {
            if !self.references.lock().contains_key(&channel) {
                tracing::warn!(?channel, "submission to a channel that does not exist");
                return None;
            }
            fence = submission.increment.and_then(|increment| {
                match self
                    .host1x
                    .syncpoints
                    .increase_max(increment.id, increment.count)
                {
                    Ok(value) => Some(SyncpointFence {
                        id: increment.id,
                        value,
                    }),
                    Err(error) => {
                        tracing::warn!(%error, "submission increments an invalid syncpoint");
                        None
                    }
                }
            });
            let mut entries = pool.pop().unwrap_or_default();
            entries.extend_from_slice(submission.entries);
            Some(CpuCommand::Submit(
                channel,
                QueuedSubmission {
                    entries,
                    wait: submission.wait,
                    increment: submission.increment,
                },
            ))
        });
        fence
    }

    /// The channel's reference count (SET_REFERENCE), as of the work the back executed.
    pub fn reference_count(&self, channel: ChannelId) -> Option<u32> {
        self.references
            .lock()
            .get(&channel)
            .map(|reference| reference.load(Ordering::Acquire))
    }

    /// Writes data the GPU modified in `[addr, addr + size)` back to guest memory; returns once
    /// it is there.
    pub fn flush_region(&self, addr: DAddr, size: u64) {
        self.side(SideCommand::FlushRegion(addr, size));
        let (done, wait) = mpsc::sync_channel(1);
        self.side(SideCommand::Signal(done));
        if wait.recv().is_err() {
            tracing::error!("GPU stopped before a flush");
        }
    }

    /// Device memory was unmapped from a GPU address space.
    pub fn unmap_region(&self, addr: DAddr, size: u64) {
        self.side(SideCommand::UnmapRegion(addr, size));
    }

    /// Presents a frame after the work queued before it, then restarts the timestamp clock's
    /// frame ([`GpuTimestamp::frame_presented`]).
    pub fn present(&self, framebuffer: FramebufferConfig) {
        self.side(SideCommand::Present(framebuffer));
    }

    /// Runs `action` on the back thread after the work queued before it.
    pub fn run_in_order(&self, action: impl FnOnce(&mut dyn Rasterizer) + Send + 'static) {
        self.side(SideCommand::Run(Box::new(action)));
    }

    /// Returns once everything queued so far ran, except the work of channels still waiting on a
    /// semaphore or syncpoint nothing queued releases.
    pub fn synchronize(&self) {
        let (done, wait) = mpsc::sync_channel(1);
        self.inbox.push(CpuCommand::Synchronize(done));
        if wait.recv().is_err() {
            tracing::error!("GPU stopped before it synchronized");
        }
    }

    /// The guest CPU may have released a semaphore a channel waits on: check now instead of at the
    /// next poll.
    pub fn wake(&self) {
        self.inbox.wake();
    }

    /// While `paused`, the front takes no submissions or other commands and runs no channel;
    /// they queue until it is resumed, and calls that wait for the GPU wait that long. Work the
    /// back already has still executes.
    pub fn pause(&self, paused: bool) {
        self.inbox.pause(paused);
    }

    /// Stops the GPU: commands still queued are dropped, and the threads finish what they hold
    /// and are joined before this returns. Calling it again, or from several threads, is safe;
    /// commands sent afterwards are dropped, and calls that wait for the GPU return at once.
    /// From a GPU thread (a [`Gpu::run_in_order`] action) it closes the GPU without joining.
    pub fn shutdown(&self) {
        self.inbox.close();
        if self.thread_ids.contains(&std::thread::current().id()) {
            tracing::error!("GPU shutdown from a GPU thread; not joining it");
            return;
        }
        for thread in self.threads.lock().drain(..) {
            if thread.join().is_err() {
                tracing::error!("a GPU thread panicked");
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn pooled_entry_buffers(&self) -> Vec<usize> {
        let state = self.inbox.state.lock();
        state.entry_buffers.iter().map(Vec::capacity).collect()
    }

    fn side(&self, command: SideCommand) {
        self.inbox.push(CpuCommand::Side(command));
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.shutdown();
    }
}
