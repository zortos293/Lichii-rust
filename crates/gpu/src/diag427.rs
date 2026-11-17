//! LIRU-6 diag (ZOR-427, never merged): per syncpoint, the time from a guest submission to the
//! back reaching its increment, from there to the guest-visible increment, and how long guest
//! waits registered on it took. Logged every 5 s.

use std::collections::{HashMap, VecDeque};
use std::fmt::Write;
use std::sync::LazyLock;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

#[derive(Default)]
struct Point {
    submits: VecDeque<(u32, Instant, Option<Instant>, Option<Instant>)>,
    stage: [Duration; 3],
    stage_max: [Duration; 3],
    staged: u64,
    reached: VecDeque<(u32, Instant)>,
    waits: Vec<(u32, Instant, bool)>,
    channels: u64,
    n_submit: u64,
    n_reach: u64,
    n_release: u64,
    queue: Duration,
    queue_max: Duration,
    gpu: Duration,
    gpu_max: Duration,
    n_wait: u64,
    wait: Duration,
    wait_max: Duration,
    wait_unreached: u64,
    host: u32,
}

struct State {
    since: Instant,
    points: HashMap<u32, Point>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        since: Instant::now(),
        points: HashMap::new(),
    })
});

fn reached(value: u32, threshold: u32) -> bool {
    value.wrapping_sub(threshold) as i32 >= 0
}

pub fn submitted(id: u32, value: u32, channel: u32) {
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    p.n_submit += 1;
    p.channels |= 1 << (channel % 64);
    p.submits.push_back((value, Instant::now(), None, None));
    if p.submits.len() > 4096 {
        p.submits.pop_front();
    }
}

pub fn host_reached(id: u32, value: u32) {
    let now = Instant::now();
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    p.n_reach += 1;
    p.host = value;
    while let Some(&(v, t, t1, t2)) = p.submits.front() {
        if !reached(value, v) {
            break;
        }
        p.submits.pop_front();
        if v == value {
            let d = now - t;
            p.queue += d;
            p.queue_max = p.queue_max.max(d);
            if let (Some(t1), Some(t2)) = (t1, t2) {
                for (i, d) in [t1 - t, t2 - t1, now - t2].into_iter().enumerate() {
                    p.stage[i] += d;
                    p.stage_max[i] = p.stage_max[i].max(d);
                }
                p.staged += 1;
            }
        }
    }
    p.reached.push_back((value, now));
    if p.reached.len() > 4096 {
        p.reached.pop_front();
    }
}

pub fn guest_released(id: u32, value: u32) {
    let now = Instant::now();
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    p.n_release += 1;
    while let Some(&(v, t)) = p.reached.front() {
        if !reached(value, v) {
            break;
        }
        p.reached.pop_front();
        if v == value {
            let d = now - t;
            p.gpu += d;
            p.gpu_max = p.gpu_max.max(d);
        }
    }
    let mut i = 0;
    while i < p.waits.len() {
        let (threshold, t, unreached) = p.waits[i];
        if reached(value, threshold) {
            let d = now - t;
            p.n_wait += 1;
            p.wait += d;
            p.wait_max = p.wait_max.max(d);
            p.wait_unreached += unreached as u64;
            p.waits.swap_remove(i);
        } else {
            i += 1;
        }
    }
    maybe_log(&mut s, now);
}

pub fn front_received(id: u32) {
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    if let Some(e) = p.submits.iter_mut().find(|e| e.2.is_none()) {
        e.2 = Some(Instant::now());
    }
}

pub fn front_pushed(id: u32) {
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    if let Some(e) = p
        .submits
        .iter_mut()
        .find(|e| e.2.is_some() && e.3.is_none())
    {
        e.3 = Some(Instant::now());
    }
}

pub fn guest_wait(id: u32, threshold: u32) {
    let mut s = STATE.lock();
    let p = s.points.entry(id).or_default();
    let unreached = !reached(p.host, threshold);
    p.waits.push((threshold, Instant::now(), unreached));
    if p.waits.len() > 256 {
        p.waits.remove(0);
    }
}

static CB_UPLOADS: AtomicU64 = AtomicU64::new(0);
static CB_WORDS: AtomicU64 = AtomicU64::new(0);
static CB_CONTIG: AtomicU64 = AtomicU64::new(0);
static CB_BREAKS: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static CB_END: Cell<u64> = const { Cell::new(u64::MAX) };
}

pub fn cbuf_upload(address: u64, bytes: u64) {
    CB_UPLOADS.fetch_add(1, Ordering::Relaxed);
    CB_WORDS.fetch_add(bytes / 4, Ordering::Relaxed);
    CB_END.with(|end| {
        if end.get() == address {
            CB_CONTIG.fetch_add(1, Ordering::Relaxed);
        }
        end.set(address + bytes);
    });
}

pub fn cbuf_break() {
    CB_END.with(|end| {
        if end.get() != u64::MAX {
            CB_BREAKS.fetch_add(1, Ordering::Relaxed);
            end.set(u64::MAX);
        }
    });
}

fn maybe_log(s: &mut State, now: Instant) {
    let period = now - s.since;
    if period < Duration::from_secs(5) {
        return;
    }
    s.since = now;
    let ms = |d: Duration| d.as_secs_f64() * 1000.0;
    let mut out = format!(
        "LIRU6SYNC period_ms={} cbuf uploads={} words={} contiguous={} runs_broken={}",
        period.as_millis(),
        CB_UPLOADS.swap(0, Ordering::Relaxed),
        CB_WORDS.swap(0, Ordering::Relaxed),
        CB_CONTIG.swap(0, Ordering::Relaxed),
        CB_BREAKS.swap(0, Ordering::Relaxed)
    );
    let mut ids: Vec<_> = s.points.keys().copied().collect();
    ids.sort();
    for id in ids {
        let p = s.points.get_mut(&id).expect("listed");
        if p.n_submit + p.n_reach + p.n_release + p.n_wait == 0 {
            continue;
        }
        let _ = write!(
            out,
            "\n  sp{id} ch={:#x} submits={} reached={} released={} submit_to_back_ms avg={:.2} max={:.1} back_to_guest_ms avg={:.2} max={:.1} waits={} wait_ms avg={:.2} max={:.1} waits_back_not_there={} staged={} inbox_ms avg={:.2} max={:.1} front_ms avg={:.2} max={:.1} ring_back_ms avg={:.2} max={:.1}",
            p.channels,
            p.n_submit,
            p.n_reach,
            p.n_release,
            ms(p.queue) / p.n_submit.max(1) as f64,
            ms(p.queue_max),
            ms(p.gpu) / p.n_release.max(1) as f64,
            ms(p.gpu_max),
            p.n_wait,
            ms(p.wait) / p.n_wait.max(1) as f64,
            ms(p.wait_max),
            p.wait_unreached,
            p.staged,
            ms(p.stage[0]) / p.staged.max(1) as f64,
            ms(p.stage_max[0]),
            ms(p.stage[1]) / p.staged.max(1) as f64,
            ms(p.stage_max[1]),
            ms(p.stage[2]) / p.staged.max(1) as f64,
            ms(p.stage_max[2])
        );
        p.stage = [Duration::ZERO; 3];
        p.stage_max = [Duration::ZERO; 3];
        p.staged = 0;
        p.n_submit = 0;
        p.n_reach = 0;
        p.n_release = 0;
        p.queue = Duration::ZERO;
        p.queue_max = Duration::ZERO;
        p.gpu = Duration::ZERO;
        p.gpu_max = Duration::ZERO;
        p.n_wait = 0;
        p.wait = Duration::ZERO;
        p.wait_max = Duration::ZERO;
        p.wait_unreached = 0;
        p.channels = 0;
    }
    tracing::info!("{out}");
}
