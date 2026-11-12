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
use std::thread::JoinHandle;
use std::time::Duration;

use lichii_common::DAddr;
use parking_lot::{Condvar, Mutex};

use crate::back::{Back, SideCommand};
use crate::backend::{ChannelId, FramebufferConfig, GpuHandles, Rasterizer};
use crate::channel::{ChannelConfig, Submission};
use crate::dma_pusher::{FrontChannel, FrontShared, Work};
use crate::engines::{EngineFactory, create_engine};
use crate::ring;
use crate::stream::{Link, MIN_RING_SLOTS, Stream};

/// Default ring size: 8 MiB of slots.
pub const DEFAULT_RING_SLOTS: usize = 1 << 20;
/// How long the front first sleeps while every channel with work waits, doubling up to
/// [`MAX_POLL`]. A semaphore write that meets a waiter, a syncpoint increment or [`Gpu::wake`]
/// ends the sleep at once; the poll only catches memory the guest CPU wrote.
const MIN_POLL: Duration = Duration::from_micros(50);
const MAX_POLL: Duration = Duration::from_millis(1);

/// Where the back half runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Threading {
    /// Front and back on their own threads.
    #[default]
    Split,
    /// Both on one GPU thread, through the same ring.
    Single,
}

/// How the GPU is built.
#[derive(Clone)]
pub struct GpuOptions {
    /// Split (the default) or one GPU thread.
    pub threading: Threading,
    /// Slots of the front-to-back ring (rounded up to a power of two).
    pub ring_slots: usize,
    /// Creates the engines channels bind.
    pub engines: EngineFactory,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            threading: Threading::default(),
            ring_slots: DEFAULT_RING_SLOTS,
            engines: Arc::new(create_engine),
        }
    }
}

enum CpuCommand {
    CreateChannel(ChannelId, ChannelConfig, Arc<AtomicU32>),
    DestroyChannel(ChannelId),
    Submit(ChannelId, Submission),
    Side(SideCommand),
    Synchronize(mpsc::SyncSender<()>),
}

/// The CPU side's queue to the front, and the front's wake-up.
pub(crate) struct Inbox {
    state: Mutex<InboxState>,
    cond: Condvar,
}

struct InboxState {
    commands: VecDeque<CpuCommand>,
    woken: bool,
    closed: bool,
}

impl Inbox {
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(InboxState {
                commands: VecDeque::new(),
                woken: false,
                closed: false,
            }),
            cond: Condvar::new(),
        }
    }

    fn push(&self, command: CpuCommand) {
        let mut state = self.state.lock();
        if !state.closed {
            state.commands.push_back(command);
            self.cond.notify_one();
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

    /// Moves the queued commands into `into` (empty), first sleeping until there are some, a
    /// wake, or `timeout`. Returns whether the inbox is closed.
    fn take(&self, into: &mut VecDeque<CpuCommand>, timeout: Option<Duration>) -> bool {
        let mut state = self.state.lock();
        if state.commands.is_empty() && !state.woken && !state.closed {
            match timeout {
                Some(timeout) => {
                    self.cond.wait_for(&mut state, timeout);
                }
                None => self.cond.wait(&mut state),
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
        loop {
            let waiting = self.channels.iter().flatten().any(FrontChannel::is_waiting);
            let closed = inbox.take(&mut batch, waiting.then_some(poll));
            let busy = !batch.is_empty();
            for command in batch.drain(..) {
                self.handle(command);
            }
            let resumed = self.resume();
            self.shared.stream.idle();
            if closed {
                break;
            }
            if busy || resumed {
                poll = MIN_POLL;
            } else if waiting {
                if poll == MAX_POLL {
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
            CpuCommand::Submit(id, submission) => {
                let Some(channel) = self
                    .channels
                    .get_mut(id.0 as usize)
                    .and_then(Option::as_mut)
                else {
                    tracing::warn!(?id, "submission to a channel that does not exist");
                    return;
                };
                if let Some(fence) = submission.wait {
                    channel.push(Work::SyncpointWait(fence));
                }
                for entry in submission.entries {
                    channel.push(Work::Entry(entry));
                }
                if let Some(increment) = submission.increment {
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

    /// Runs the channels whose wait is over. Returns whether any ran.
    fn resume(&mut self) -> bool {
        let mut resumed = false;
        for channel in self.channels.iter_mut().flatten() {
            if channel.is_runnable(&self.shared) {
                channel.run(&mut self.shared);
                resumed = true;
            }
        }
        resumed
    }
}

/// The emulated GPU. Every method may be called from any thread; work is executed in the order
/// it was queued.
pub struct Gpu {
    inbox: Arc<Inbox>,
    next_channel: AtomicU32,
    references: Mutex<HashMap<ChannelId, Arc<AtomicU32>>>,
    front: Option<JoinHandle<()>>,
    back: Option<JoinHandle<()>>,
}

impl Gpu {
    /// Starts the GPU threads. `rasterizer` moves to the back thread, the only one that calls it;
    /// `handles` are the ones it was created with.
    pub fn new(rasterizer: Box<dyn Rasterizer>, handles: &GpuHandles, options: GpuOptions) -> Self {
        let inbox = Arc::new(Inbox::new());
        handles.waiters.attach(&inbox);
        let (producer, consumer) = ring::ring(options.ring_slots, MIN_RING_SLOTS);
        let (side_sender, side_receiver) = mpsc::channel();
        let back = Back::new(
            rasterizer,
            handles.syncpoints.clone(),
            options.engines,
            side_receiver,
        );
        let (link, back) = match options.threading {
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
                stream: Stream::new(producer, link, side_sender),
                shared_macros: None,
                words: Vec::new(),
                waiters: handles.waiters.clone(),
                inbox: inbox.clone(),
                syncpoints: handles.syncpoints.clone(),
            },
            channels: Vec::new(),
        };
        let front_inbox = inbox.clone();
        let front = std::thread::Builder::new()
            .name("gpu-front".into())
            .spawn(move || front.run(front_inbox))
            .expect("failed to start the GPU front thread");
        Self {
            inbox,
            next_channel: AtomicU32::new(0),
            references: Mutex::new(HashMap::new()),
            front: Some(front),
            back,
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

    /// Destroys a channel once the work queued before has run (a suspended channel's remaining
    /// work is dropped).
    pub fn destroy_channel(&self, channel: ChannelId) {
        self.references.lock().remove(&channel);
        self.inbox.push(CpuCommand::DestroyChannel(channel));
    }

    /// Queues a GPFIFO submission.
    pub fn submit(&self, channel: ChannelId, submission: Submission) {
        self.inbox.push(CpuCommand::Submit(channel, submission));
    }

    /// The channel's reference count (SET_REFERENCE), as of the work the back executed.
    pub fn reference_count(&self, channel: ChannelId) -> Option<u32> {
        self.references
            .lock()
            .get(&channel)
            .map(|reference| reference.load(Ordering::Acquire))
    }

    /// The guest CPU wrote device memory: cached copies are stale.
    pub fn invalidate_region(&self, addr: DAddr, size: u64) {
        self.side(SideCommand::InvalidateRegion(addr, size));
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

    /// Presents a frame after the work queued before it.
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

    fn side(&self, command: SideCommand) {
        self.inbox.push(CpuCommand::Side(command));
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.inbox.close();
        for thread in [self.front.take(), self.back.take()].into_iter().flatten() {
            if thread.join().is_err() {
                tracing::error!("a GPU thread panicked");
            }
        }
    }
}
