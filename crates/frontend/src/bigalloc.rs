//! LIRU-5 diag (ZOR-432), not for main: records live heap allocations of 8 MiB or more with
//! their backtraces and writes them at LICHII_BIGALLOC_AT seconds into LICHII_BIGALLOC_DIR.

use std::alloc::{GlobalAlloc, Layout, System};
use std::backtrace::Backtrace;
use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MIN: usize = 8 << 20;
static ON: AtomicBool = AtomicBool::new(false);
static CAPTURES: AtomicU64 = AtomicU64::new(0);
static CAPTURE_NS: AtomicU64 = AtomicU64::new(0);
type Live = HashMap<usize, (usize, Arc<Backtrace>)>;
static LIVE: Mutex<Option<Live>> = Mutex::new(None);

thread_local! {
    static INSIDE: Cell<bool> = const { Cell::new(false) };
}

fn guarded(f: impl FnOnce()) {
    if !INSIDE
        .try_with(|inside| !inside.replace(true))
        .unwrap_or(false)
    {
        return;
    }
    f();
    let _ = INSIDE.try_with(|inside| inside.set(false));
}

fn record(ptr: *mut u8, size: usize) {
    if size < MIN || ptr.is_null() || !ON.load(Ordering::Relaxed) {
        return;
    }
    guarded(|| {
        let start = Instant::now();
        let trace = Arc::new(Backtrace::force_capture());
        CAPTURE_NS.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CAPTURES.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut live) = LIVE.lock() {
            live.get_or_insert_with(HashMap::new)
                .insert(ptr as usize, (size, trace));
        }
    });
}

fn forget(ptr: *mut u8, size: usize) {
    if size < MIN || !ON.load(Ordering::Relaxed) {
        return;
    }
    guarded(|| {
        let removed = LIVE
            .lock()
            .ok()
            .and_then(|mut live| live.as_mut().and_then(|live| live.remove(&(ptr as usize))));
        drop(removed);
    });
}

pub struct Tracking;

// SAFETY: every call forwards to System unchanged; the bookkeeping never touches the memory.
unsafe impl GlobalAlloc for Tracking {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded from the caller.
        let ptr = unsafe { System.alloc(layout) };
        record(ptr, layout.size());
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded from the caller.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        record(ptr, layout.size());
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        forget(ptr, layout.size());
        // SAFETY: forwarded from the caller.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: forwarded from the caller.
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            forget(ptr, layout.size());
            record(new, new_size);
        }
        new
    }
}

fn frames(trace: &Backtrace, out: &mut String) {
    let text = format!("{trace}");
    let mut kept = 0;
    let mut keep_at = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("at ") {
            if keep_at {
                let _ = writeln!(out, "\t\t{line}");
            }
            continue;
        }
        let name = line.split_once(": ").map_or(line, |(_, name)| name);
        keep_at = !(name.starts_with("std::")
            || name.starts_with("core::")
            || name.starts_with("alloc::")
            || name.starts_with("hashbrown::")
            || name.contains("bigalloc")
            || name.contains("__rust")
            || name.starts_with("BaseThreadInitThunk")
            || name.starts_with("RtlUserThreadStart"));
        if keep_at && kept < 16 {
            kept += 1;
            let _ = writeln!(out, "\t{name}");
        } else {
            keep_at = false;
        }
    }
}

pub fn start() {
    let (Ok(dir), Ok(at)) = (
        std::env::var("LICHII_BIGALLOC_DIR"),
        std::env::var("LICHII_BIGALLOC_AT"),
    ) else {
        return;
    };
    let times: Vec<u64> = at
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    ON.store(true, Ordering::Relaxed);
    let started = Instant::now();
    let _ = std::thread::Builder::new()
        .name("bigalloc".into())
        .spawn(move || {
            let mut snaps = Vec::new();
            for &t in &times {
                if let Some(wait) = Duration::from_secs(t).checked_sub(started.elapsed()) {
                    std::thread::sleep(wait);
                }
                let mut taken = Vec::new();
                guarded(|| {
                    if let Ok(live) = LIVE.lock()
                        && let Some(live) = live.as_ref()
                    {
                        taken = live
                            .iter()
                            .map(|(&ptr, (size, trace))| (ptr, *size, trace.clone()))
                            .collect();
                    }
                });
                snaps.push((started.elapsed().as_secs(), taken));
            }
            std::thread::sleep(Duration::from_secs(2));
            let mut out = String::new();
            let mut ids: HashMap<*const Backtrace, usize> = HashMap::new();
            let mut traces = Vec::new();
            for (t, taken) in &snaps {
                let total: usize = taken.iter().map(|(_, size, _)| size).sum();
                let _ = writeln!(out, "snap\t{t}\t{}\t{}", taken.len(), total >> 20);
                for (ptr, size, trace) in taken {
                    let next = ids.len();
                    let id = *ids.entry(Arc::as_ptr(trace)).or_insert_with(|| {
                        traces.push(trace.clone());
                        next
                    });
                    let _ = writeln!(out, "alloc\t{t}\t{ptr:#x}\t{}\t{id}", size >> 20);
                }
            }
            let _ = writeln!(
                out,
                "captures\t{}\tcapture_ms\t{}",
                CAPTURES.load(Ordering::Relaxed),
                CAPTURE_NS.load(Ordering::Relaxed) / 1_000_000
            );
            for (id, trace) in traces.iter().enumerate() {
                let _ = writeln!(out, "trace\t{id}");
                frames(trace, &mut out);
            }
            let _ = std::fs::write(std::path::Path::new(&dir).join("bigalloc.txt"), out);
        });
}
