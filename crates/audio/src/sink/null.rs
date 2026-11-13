//! The null sink's clock: consumes streams in real time without a device.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use super::stream::{StreamState, StreamType};
use super::{MAX_CHANNELS, SAMPLE_RATE};

const TICK: Duration = Duration::from_millis(5);
/// Frames handed to one stream callback at a time.
const CHUNK_FRAMES: usize = 1024;
/// The most a late tick catches up, so a host stall does not drain every queue at once.
const MAX_FRAMES_PER_TICK: u64 = SAMPLE_RATE as u64 / 10;

#[derive(Default)]
struct PacerShared {
    streams: Mutex<Vec<Weak<StreamState>>>,
    stop: AtomicBool,
}

/// A host thread that plays every stream without a device every 5 ms, as a device callback
/// would.
pub(super) struct Pacer {
    shared: Arc<PacerShared>,
    thread: Option<JoinHandle<()>>,
}

impl Pacer {
    pub(super) fn start() -> Self {
        let shared = Arc::new(PacerShared::default());
        let thread = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("audio-null".into())
                .spawn(move || run(&shared))
                .map_err(|err| tracing::error!("starting the null audio clock: {err}"))
                .ok()
        };
        Self { shared, thread }
    }

    pub(super) fn add(&self, stream: &Arc<StreamState>) {
        self.handle().add(stream);
    }

    pub(super) fn handle(&self) -> PacerHandle {
        PacerHandle(self.shared.clone())
    }
}

/// Adds streams to a running [`Pacer`] later, as a host stream that fails to start does.
#[derive(Clone)]
pub(crate) struct PacerHandle(Arc<PacerShared>);

impl PacerHandle {
    pub(crate) fn add(&self, stream: &Arc<StreamState>) {
        self.0.streams.lock().push(Arc::downgrade(stream));
    }
}

impl Drop for Pacer {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn run(shared: &PacerShared) {
    let mut scratch = vec![0i16; CHUNK_FRAMES * MAX_CHANNELS];
    let start = Instant::now();
    let mut done = 0u64;
    while !shared.stop.load(Ordering::Acquire) {
        std::thread::sleep(TICK);
        let due = (start.elapsed().as_nanos() * u128::from(SAMPLE_RATE) / 1_000_000_000) as u64;
        let frames = (due - done).min(MAX_FRAMES_PER_TICK) as usize;
        done = due;
        let mut streams = shared.streams.lock();
        streams.retain(|stream| stream.strong_count() > 0);
        for stream in streams.iter().filter_map(Weak::upgrade) {
            if stream.is_paused() {
                continue;
            }
            if stream.ty() == StreamType::In {
                stream.process_input(frames);
                continue;
            }
            let channels = stream.device_channels();
            let mut left = frames;
            while left > 0 {
                let chunk = left.min(CHUNK_FRAMES);
                stream.process_output(&mut scratch[..chunk * channels]);
                left -= chunk;
            }
        }
    }
}
