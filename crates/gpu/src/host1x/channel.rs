//! The host1x channels of the video engines (NVDEC, VIC, NVJPG). Each open channel runs the command
//! buffers submitted to it in order on a thread of its own, as the channel DMA does: a syncpoint
//! increment lands when the stream reaches it, and a host wait holds only its own channel.
//!
//! An NVDEC job decodes its frame and a VIC job composes its output surface for the channel.
//! Every NVDEC job takes at least [`DECODE_TIME`], decoded or not. On a channel opened without
//! decoding, or without FFmpeg, it only takes that and a VIC job writes nothing, so a title's
//! video waits still complete and it moves on past them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use lichii_memory::AddressSpace;

use super::nvdec::Nvdec;
use super::vic::Vic;
use super::{Host1x, SyncpointWaitError, Syncpoints};

/// A video engine behind a host1x channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoEngine {
    NvDec,
    Vic,
    NvJpg,
}

impl VideoEngine {
    fn class(self) -> u32 {
        match self {
            Self::NvDec => 0xf0,
            Self::Vic => 0x5d,
            Self::NvJpg => 0xc0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::NvDec => "nvdec",
            Self::Vic => "vic",
            Self::NvJpg => "nvjpg",
        }
    }
}

/// The least time an NVDEC job takes, and all it takes without a decoder: half a 60 FPS frame, as
/// the reference sleeps with decoding off, because titles stall when a decode completes in well
/// under a millisecond.
pub const DECODE_TIME: Duration = Duration::from_millis(8);
/// How often a host wait checks whether its channel is closing.
const WAIT_SLICE: Duration = Duration::from_millis(50);
/// A host wait blocked this long is logged, so a hang names the syncpoint it stopped on.
const STALL_WARNING: Duration = Duration::from_secs(2);

/// The host1x class whose methods wait on syncpoints.
pub(super) const CLASS_CONTROL: u32 = 0x1;
const CONTROL_WAIT_SYNCPT: u32 = 0x8;
const CONTROL_LOAD_SYNCPT_PAYLOAD_32: u32 = 0x4e;
const CONTROL_WAIT_SYNCPT_32: u32 = 0x50;
/// Methods every engine class takes through its THI block.
pub(super) const THI_INCR_SYNCPT: u32 = 0x0;
pub(super) const THI_METHOD_0: u32 = 0x10;
pub(super) const THI_METHOD_1: u32 = 0x11;

/// The engine behind a channel's THI block, which takes the engine method writes.
enum Engine {
    NvDec(Box<Nvdec>),
    Vic(Box<Vic>),
    NvJpg,
}

impl Engine {
    fn new(
        engine: VideoEngine,
        host1x: &Host1x,
        process: Option<Arc<AddressSpace>>,
        decode: bool,
    ) -> Self {
        let memory = host1x.memory.clone();
        let frames = host1x.decoded_frames.clone();
        let decode = decode && lichii_video::available();
        match engine {
            VideoEngine::NvDec => Self::NvDec(Box::new(Nvdec::new(memory, frames, decode))),
            VideoEngine::Vic => Self::Vic(Box::new(Vic::new(memory, process, frames, decode))),
            VideoEngine::NvJpg => Self::NvJpg,
        }
    }

    fn write(&mut self, method: u32, argument: u32) {
        match self {
            Self::NvDec(nvdec) => {
                let started = Instant::now();
                if nvdec.write(method, argument)
                    && let Some(rest) = DECODE_TIME.checked_sub(started.elapsed())
                {
                    std::thread::sleep(rest);
                }
            }
            Self::Vic(vic) => vic.write(method, argument),
            Self::NvJpg => {}
        }
    }
}

/// An open video engine channel. Closing or dropping it stops the channel once its current command
/// returns.
pub struct EngineChannel {
    submits: Option<mpsc::Sender<Vec<u32>>>,
    closing: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl EngineChannel {
    /// Opens a channel of `engine` over `host1x`'s device memory. VIC output is reported to the
    /// write tracking of `process`, the process attached to the device address space, so GPU
    /// caches holding an output surface reload it. With `decode` false, NVDEC jobs only take
    /// [`DECODE_TIME`] and VIC jobs write nothing, as without FFmpeg.
    pub fn open(
        engine: VideoEngine,
        host1x: &Host1x,
        process: Option<Arc<AddressSpace>>,
        decode: bool,
    ) -> Self {
        let (submits, received) = mpsc::channel::<Vec<u32>>();
        let closing = Arc::new(AtomicBool::new(false));
        let mut stream = CommandStream {
            kind: engine,
            engine: Engine::new(engine, host1x, process, decode),
            syncpoints: host1x.syncpoints.clone(),
            closing: closing.clone(),
            class: engine.class(),
            method: 0,
            count: 0,
            mask: 0,
            incrementing: false,
            payload: 0,
            engine_method: 0,
        };
        let thread = std::thread::Builder::new()
            .name(format!("host1x-{}", engine.name()))
            .spawn(move || {
                for words in received {
                    stream.run(&words);
                }
            })
            .expect("failed to start a host1x channel thread");
        Self {
            submits: Some(submits),
            closing,
            thread: Some(thread),
        }
    }

    /// Queues a command buffer; it runs after every buffer submitted before it.
    pub fn submit(&self, words: Vec<u32>) {
        if let Some(submits) = &self.submits {
            let _ = submits.send(words);
        }
    }

    /// Stops the channel and returns once its thread has; queued buffers that had not started
    /// are dropped. Submissions after this are ignored.
    pub fn close(&mut self) {
        self.closing.store(true, Ordering::Relaxed);
        self.submits = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for EngineChannel {
    fn drop(&mut self) {
        self.close();
    }
}

/// The channel's command processor. Its state carries over from one command buffer to the next.
struct CommandStream {
    kind: VideoEngine,
    engine: Engine,
    syncpoints: Arc<Syncpoints>,
    closing: Arc<AtomicBool>,
    class: u32,
    method: u32,
    count: u32,
    mask: u32,
    incrementing: bool,
    payload: u32,
    engine_method: u32,
}

impl CommandStream {
    fn run(&mut self, words: &[u32]) {
        for &word in words {
            if self.closing.load(Ordering::Relaxed) {
                return;
            }
            if self.mask != 0 {
                let bit = self.mask.trailing_zeros();
                self.mask &= !(1 << bit);
                self.execute(self.method + bit, word);
                continue;
            }
            if self.count != 0 {
                self.count -= 1;
                self.execute(self.method, word);
                if self.incrementing {
                    self.method += 1;
                }
                continue;
            }
            let value = word & 0xffff;
            let method = (word >> 16) & 0xfff;
            match word >> 28 {
                0 => {
                    self.mask = value & 0x3f;
                    self.method = method;
                    self.class = (value >> 6) & 0x3ff;
                }
                mode @ (1 | 2) => {
                    self.count = value;
                    self.method = method;
                    self.incrementing = mode == 1;
                }
                3 => {
                    self.mask = value;
                    self.method = method;
                }
                4 => self.execute(method, value & 0xfff),
                mode => tracing::warn!(
                    engine = self.kind.name(),
                    mode,
                    "unsupported host1x command mode"
                ),
            }
        }
    }

    fn execute(&mut self, method: u32, argument: u32) {
        if self.class == CLASS_CONTROL {
            match method {
                CONTROL_LOAD_SYNCPT_PAYLOAD_32 => self.payload = argument,
                CONTROL_WAIT_SYNCPT | CONTROL_WAIT_SYNCPT_32 => self.wait(argument, self.payload),
                _ => tracing::debug!(method, argument, "unhandled host1x control method"),
            }
            return;
        }
        match method {
            THI_INCR_SYNCPT => {
                let id = argument & 0xff;
                let _ = self.syncpoints.guest().increment(id);
                let _ = self.syncpoints.host().increment(id);
            }
            THI_METHOD_0 => self.engine_method = argument,
            THI_METHOD_1 => self.engine.write(self.engine_method, argument),
            _ => {}
        }
    }

    /// Holds the channel until syncpoint `id` reaches `threshold` or the channel closes.
    fn wait(&self, id: u32, threshold: u32) {
        let started = Instant::now();
        let mut stalled = false;
        loop {
            match self.syncpoints.host().wait(id, threshold, Some(WAIT_SLICE)) {
                Err(SyncpointWaitError::TimedOut { value })
                    if !self.closing.load(Ordering::Relaxed) =>
                {
                    if !stalled && started.elapsed() >= STALL_WARNING {
                        stalled = true;
                        tracing::warn!(
                            engine = self.kind.name(),
                            syncpoint = id,
                            threshold,
                            value,
                            "host1x channel wait blocked for over 2 s"
                        );
                    }
                }
                _ => break,
            }
        }
        if stalled {
            tracing::warn!(
                engine = self.kind.name(),
                syncpoint = id,
                threshold,
                waited_ms = started.elapsed().as_millis() as u64,
                "host1x channel wait resumed"
            );
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    use std::time::Instant;

    use lichii_memory::PhysicalMemory;

    use super::*;

    /// The engine method that runs a job: NVDEC's registers are 8 bytes wide and VIC's 4, so both
    /// sit at index 0xc0.
    pub(in crate::host1x) const ENGINE_EXECUTE: u32 = 0xc0;

    pub(in crate::host1x) fn host1x() -> Host1x {
        Host1x::new(Arc::new(PhysicalMemory::new(1 << 26).unwrap()), 34)
    }

    pub(in crate::host1x) fn set_class(class: u32, method: u32, mask: u32) -> u32 {
        (method << 16) | (class << 6) | mask
    }

    pub(in crate::host1x) fn incrementing(method: u32, count: u32) -> u32 {
        (1 << 28) | (method << 16) | count
    }

    pub(in crate::host1x) fn immediate(method: u32, value: u32) -> u32 {
        (4 << 28) | (method << 16) | value
    }

    pub(in crate::host1x) fn wait_for(syncpoints: &Syncpoints, id: u32, value: u32) {
        syncpoints
            .host()
            .wait(id, value, Some(Duration::from_secs(5)))
            .expect("the channel reached the syncpoint");
    }

    #[test]
    fn nvdec_jobs_increment_their_syncpoint_after_the_decode_time() {
        let host1x = host1x();
        let syncpoints = host1x.syncpoints.clone();
        let channel = EngineChannel::open(VideoEngine::NvDec, &host1x, None, true);
        let started = Instant::now();
        channel.submit(vec![
            incrementing(THI_METHOD_0, 2),
            ENGINE_EXECUTE,
            0,
            set_class(VideoEngine::NvDec.class(), THI_INCR_SYNCPT, 1),
            7,
        ]);
        wait_for(&syncpoints, 7, 1);
        assert!(started.elapsed() >= DECODE_TIME);
        assert_eq!(syncpoints.guest().value(7), Ok(1));
        channel.submit(vec![immediate(THI_INCR_SYNCPT, 7)]);
        wait_for(&syncpoints, 7, 2);
    }

    #[test]
    fn control_waits_hold_the_channel_until_the_syncpoint() {
        let host1x = host1x();
        let syncpoints = host1x.syncpoints.clone();
        let channel = EngineChannel::open(VideoEngine::Vic, &host1x, None, true);
        channel.submit(vec![
            set_class(CLASS_CONTROL, 0, 0),
            immediate(CONTROL_LOAD_SYNCPT_PAYLOAD_32, 2),
            immediate(CONTROL_WAIT_SYNCPT_32, 3),
            set_class(VideoEngine::Vic.class(), 0, 0),
            incrementing(THI_METHOD_0, 2),
            ENGINE_EXECUTE,
            0,
            immediate(THI_INCR_SYNCPT, 4),
        ]);
        syncpoints.host().increment(3).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(syncpoints.host().value(4), Ok(0), "still waiting on 3");
        syncpoints.host().increment(3).unwrap();
        wait_for(&syncpoints, 4, 1);
    }

    #[test]
    fn closing_a_channel_abandons_its_wait() {
        let host1x = host1x();
        let syncpoints = host1x.syncpoints.clone();
        let channel = EngineChannel::open(VideoEngine::NvJpg, &host1x, None, true);
        channel.submit(vec![
            set_class(CLASS_CONTROL, 0, 0),
            immediate(CONTROL_LOAD_SYNCPT_PAYLOAD_32, 1),
            immediate(CONTROL_WAIT_SYNCPT_32, 5),
        ]);
        std::thread::sleep(Duration::from_millis(10));
        drop(channel);
        assert_eq!(syncpoints.host().value(5), Ok(0));
    }
}
