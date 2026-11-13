//! The host1x channels of the video engines (NVDEC, VIC, NVJPG). Each open channel runs the command
//! buffers submitted to it in order on a thread of its own, as the channel DMA does: a syncpoint
//! increment lands when the stream reaches it, and a host wait holds only its own channel.
//!
//! Nothing is decoded yet. An NVDEC job takes a frame's worth of time and leaves its output
//! surface as it is, and a VIC job composes nothing, so a title's video waits complete and it moves
//! on past them.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::Duration;

use super::{SyncpointWaitError, Syncpoints};

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

/// How long an NVDEC job takes without a decoder: half a 60 FPS frame, as the reference sleeps with
/// decoding off, because titles stall when a decode completes in well under a millisecond.
pub const DECODE_TIME: Duration = Duration::from_millis(8);
/// How long closing a channel waits for its thread before saying so.
const CLOSE_WARNING: Duration = Duration::from_secs(2);
/// How often a host wait checks whether its channel is closing.
const WAIT_SLICE: Duration = Duration::from_millis(50);

/// The host1x class whose methods wait on syncpoints.
const CLASS_CONTROL: u32 = 0x1;
const CONTROL_WAIT_SYNCPT: u32 = 0x8;
const CONTROL_LOAD_SYNCPT_PAYLOAD_32: u32 = 0x4e;
const CONTROL_WAIT_SYNCPT_32: u32 = 0x50;
/// Methods every engine class takes through its THI block.
const THI_INCR_SYNCPT: u32 = 0x0;
const THI_METHOD_0: u32 = 0x10;
const THI_METHOD_1: u32 = 0x11;
/// The engine method that runs a job: NVDEC's registers are 8 bytes wide and VIC's 4, so both sit
/// at index 0xc0.
const ENGINE_EXECUTE: u32 = 0xc0;

/// An open video engine channel. Closing or dropping it stops the channel once its current command
/// returns.
pub struct EngineChannel {
    engine: VideoEngine,
    submits: Option<mpsc::Sender<Vec<u32>>>,
    closing: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl EngineChannel {
    pub fn open(engine: VideoEngine, syncpoints: Arc<Syncpoints>) -> Self {
        let (submits, received) = mpsc::channel::<Vec<u32>>();
        let closing = Arc::new(AtomicBool::new(false));
        let mut stream = CommandStream::new(engine, syncpoints, closing.clone());
        let thread = std::thread::Builder::new()
            .name(format!("host1x-{}", engine.name()))
            .spawn(move || {
                for words in received {
                    stream.run(&words);
                }
            })
            .expect("failed to start a host1x channel thread");
        Self {
            engine,
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
            let started = std::time::Instant::now();
            let mut warned = false;
            while !thread.is_finished() {
                if !warned && started.elapsed() >= CLOSE_WARNING {
                    warned = true;
                    tracing::warn!(
                        engine = self.engine.name(),
                        "host1x channel still running after close"
                    );
                }
                std::thread::sleep(Duration::from_millis(1));
            }
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
    engine: VideoEngine,
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
    fn new(engine: VideoEngine, syncpoints: Arc<Syncpoints>, closing: Arc<AtomicBool>) -> Self {
        Self {
            engine,
            syncpoints,
            closing,
            class: engine.class(),
            method: 0,
            count: 0,
            mask: 0,
            incrementing: false,
            payload: 0,
            engine_method: 0,
        }
    }

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
                    engine = self.engine.name(),
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
            THI_METHOD_1 if self.engine_method == ENGINE_EXECUTE => self.run_job(),
            _ => {}
        }
    }

    fn run_job(&self) {
        match self.engine {
            VideoEngine::NvDec => std::thread::sleep(DECODE_TIME),
            VideoEngine::Vic | VideoEngine::NvJpg => {}
        }
    }

    /// Holds the channel until syncpoint `id` reaches `threshold` or the channel closes.
    fn wait(&self, id: u32, threshold: u32) {
        loop {
            match self.syncpoints.host().wait(id, threshold, Some(WAIT_SLICE)) {
                Err(SyncpointWaitError::TimedOut { .. })
                    if !self.closing.load(Ordering::Relaxed) => {}
                _ => return,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    fn set_class(class: u32, method: u32, mask: u32) -> u32 {
        (method << 16) | (class << 6) | mask
    }

    fn incrementing(method: u32, count: u32) -> u32 {
        (1 << 28) | (method << 16) | count
    }

    fn immediate(method: u32, value: u32) -> u32 {
        (4 << 28) | (method << 16) | value
    }

    fn wait_for(syncpoints: &Syncpoints, id: u32, value: u32) {
        syncpoints
            .host()
            .wait(id, value, Some(Duration::from_secs(5)))
            .expect("the channel reached the syncpoint");
    }

    #[test]
    fn nvdec_jobs_increment_their_syncpoint_after_the_decode_time() {
        let syncpoints = Arc::new(Syncpoints::new());
        let channel = EngineChannel::open(VideoEngine::NvDec, syncpoints.clone());
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
        let syncpoints = Arc::new(Syncpoints::new());
        let channel = EngineChannel::open(VideoEngine::Vic, syncpoints.clone());
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
        let syncpoints = Arc::new(Syncpoints::new());
        let channel = EngineChannel::open(VideoEngine::NvJpg, syncpoints.clone());
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
