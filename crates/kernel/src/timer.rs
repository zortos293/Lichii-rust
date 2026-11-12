//! Host timer thread: sleeps, wait timeouts and the preemption tick.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::result;
use crate::scheduler::Scheduler;
use crate::thread::{Thread, WakeResult};

const PREEMPTION_PERIOD: Duration = Duration::from_millis(10);

struct Entry {
    deadline: Instant,
    seq: u64,
    thread: Weak<Thread>,
    token: u64,
    wake: WakeResult,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        (self.deadline, self.seq) == (other.deadline, other.seq)
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.deadline, self.seq).cmp(&(other.deadline, other.seq))
    }
}

#[derive(Default)]
struct State {
    heap: BinaryHeap<Reverse<Entry>>,
    seq: u64,
    shutdown: bool,
}

#[derive(Default)]
pub(crate) struct Timer {
    state: Mutex<State>,
    cv: Condvar,
}

impl Timer {
    /// Wakes `thread` with `TimedOut` at `deadline` if `token` is still its wait then.
    pub fn schedule_timeout(&self, deadline: Instant, thread: &Arc<Thread>, token: u64) {
        self.schedule(
            deadline,
            thread,
            token,
            WakeResult::error(result::TIMED_OUT),
        );
    }

    /// Wakes `thread` with success at `deadline` (sleeps).
    pub fn schedule_wakeup(&self, deadline: Instant, thread: &Arc<Thread>, token: u64) {
        self.schedule(deadline, thread, token, WakeResult::SUCCESS);
    }

    fn schedule(&self, deadline: Instant, thread: &Arc<Thread>, token: u64, wake: WakeResult) {
        let mut state = self.state.lock();
        state.seq += 1;
        let seq = state.seq;
        let earliest = state
            .heap
            .peek()
            .is_none_or(|Reverse(e)| deadline < e.deadline);
        state.heap.push(Reverse(Entry {
            deadline,
            seq,
            thread: Arc::downgrade(thread),
            token,
            wake,
        }));
        if earliest {
            self.cv.notify_one();
        }
    }

    pub fn run(&self, scheduler: &Scheduler) {
        let mut next_tick = Instant::now() + PREEMPTION_PERIOD;
        let mut due = Vec::new();
        loop {
            {
                let mut state = self.state.lock();
                loop {
                    if state.shutdown {
                        return;
                    }
                    let now = Instant::now();
                    while state
                        .heap
                        .peek()
                        .is_some_and(|Reverse(e)| e.deadline <= now)
                    {
                        due.push(state.heap.pop().expect("peeked").0);
                    }
                    if !due.is_empty() || now >= next_tick {
                        break;
                    }
                    let wake_at = state
                        .heap
                        .peek()
                        .map_or(next_tick, |Reverse(e)| e.deadline.min(next_tick));
                    self.cv.wait_until(&mut state, wake_at);
                }
            }
            for entry in due.drain(..) {
                if let Some(thread) = entry.thread.upgrade() {
                    thread.wake(entry.token, entry.wake);
                }
            }
            if Instant::now() >= next_tick {
                scheduler.preemption_tick();
                next_tick += PREEMPTION_PERIOD;
                let now = Instant::now();
                if next_tick < now {
                    next_tick = now + PREEMPTION_PERIOD;
                }
            }
        }
    }

    pub fn shutdown(&self) {
        let mut state = self.state.lock();
        state.shutdown = true;
        state.heap.clear();
        self.cv.notify_all();
    }
}
