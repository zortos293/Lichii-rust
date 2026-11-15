//! A thread that wakes the GPU's back once the host GPU work a fence waits for has finished, so
//! an idle back runs the fence's actions at once instead of at its next poll.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use ash::vk;
use parking_lot::{Condvar, Mutex};

use crate::device::Device;
use crate::scheduler::Timeline;

/// How long one wait on the host GPU lasts before the thread looks for a stop.
const WAIT_SLICE: Duration = Duration::from_millis(100);

#[derive(Default)]
struct Target {
    tick: u64,
    stop: bool,
}

pub(super) struct FenceWaiter {
    shared: Arc<(Mutex<Target>, Condvar)>,
    /// The last tick handed to the thread, so a fence already waited for takes no lock.
    posted: u64,
    thread: Option<JoinHandle<()>>,
}

impl FenceWaiter {
    pub(super) fn spawn(
        device: Arc<Device>,
        timeline: Arc<Timeline>,
        wake: Box<dyn Fn() + Send + Sync>,
    ) -> std::io::Result<Self> {
        let shared = Arc::new((Mutex::new(Target::default()), Condvar::new()));
        let target = shared.clone();
        let thread = std::thread::Builder::new()
            .name("gpu-fence-wake".into())
            .spawn(move || {
                let (lock, cond) = &*target;
                let mut reached = 0;
                loop {
                    let tick = {
                        let mut target = lock.lock();
                        while !target.stop && target.tick <= reached {
                            cond.wait(&mut target);
                        }
                        if target.stop {
                            return;
                        }
                        target.tick
                    };
                    let semaphores = [timeline.raw()];
                    let values = [tick];
                    let info = vk::SemaphoreWaitInfo::default()
                        .semaphores(&semaphores)
                        .values(&values);
                    // SAFETY: this thread holds the timeline, and `tick` was submitted.
                    match unsafe {
                        device
                            .raw
                            .wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64)
                    } {
                        Ok(()) => {
                            reached = tick;
                            wake();
                        }
                        Err(vk::Result::TIMEOUT) => {}
                        Err(error) => {
                            tracing::error!(%error, "waiting for a fence's host GPU work failed");
                            reached = tick;
                        }
                    }
                }
            })?;
        Ok(Self {
            shared,
            posted: 0,
            thread: Some(thread),
        })
    }

    /// Wakes the back once the host GPU reaches `tick`, which must be submitted.
    pub(super) fn wait_for(&mut self, tick: u64) {
        if tick <= self.posted {
            return;
        }
        self.posted = tick;
        let (lock, cond) = &*self.shared;
        lock.lock().tick = tick;
        cond.notify_one();
    }
}

impl Drop for FenceWaiter {
    fn drop(&mut self) {
        let (lock, cond) = &*self.shared;
        lock.lock().stop = true;
        cond.notify_one();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
