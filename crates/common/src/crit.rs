//! DIAG ONLY (never merge): a critical-path event trace. With LICHII_CRIT_TRACE=<file>, events
//! between LICHII_CRIT_FROM and LICHII_CRIT_TO seconds after the first event go to the file as
//! records of six little-endian u64: kind, host thread index, nanoseconds, a, b, c.

use std::cell::Cell;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

pub const NAME: u64 = 0;
pub const GSUB: u64 = 1;
pub const GINC: u64 = 2;
pub const BINC: u64 = 3;
pub const SWAIT: u64 = 4;
pub const EVREG: u64 = 5;
pub const EVSIG: u64 = 6;
pub const KBLOCK: u64 = 7;
pub const KRUN: u64 = 8;
pub const KWAKE: u64 = 9;
pub const KSIG: u64 = 10;
pub const KWOBJ: u64 = 11;
pub const PARK: u64 = 12;
pub const CORE: u64 = 13;
pub const GWAIT: u64 = 14;
pub const RCALL: u64 = 15;
pub const GPUCB: u64 = 16;
pub const VBLANK: u64 = 17;
pub const SVC: u64 = 18;
pub const FSYNC: u64 = 19;

struct Trace {
    base: Instant,
    from: u64,
    to: u64,
    records: Mutex<Vec<[u64; 6]>>,
}

static TRACE: OnceLock<Option<Trace>> = OnceLock::new();
static THREADS: AtomicU64 = AtomicU64::new(1);

thread_local! {
    static THREAD: Cell<u64> = const { Cell::new(0) };
}

fn trace() -> Option<&'static Trace> {
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os("LICHII_CRIT_TRACE")?;
            let seconds = |name: &str, default: u64| {
                std::env::var(name)
                    .ok()
                    .and_then(|v| v.parse::<u64>().ok())
                    .unwrap_or(default)
                    * 1_000_000_000
            };
            let mut file = std::io::BufWriter::new(std::fs::File::create(path).ok()?);
            std::thread::Builder::new()
                .name("crit-trace".into())
                .spawn(move || {
                    loop {
                        std::thread::sleep(Duration::from_millis(250));
                        let Some(trace) = TRACE.get().and_then(Option::as_ref) else {
                            return;
                        };
                        let records = std::mem::take(&mut *trace.records.lock().unwrap());
                        for record in &records {
                            for word in record {
                                let _ = file.write_all(&word.to_le_bytes());
                            }
                        }
                        let _ = file.flush();
                    }
                })
                .ok()?;
            Some(Trace {
                base: Instant::now(),
                from: seconds("LICHII_CRIT_FROM", 0),
                to: seconds("LICHII_CRIT_TO", u64::MAX / 1_000_000_000),
                records: Mutex::new(Vec::with_capacity(1 << 16)),
            })
        })
        .as_ref()
}

/// Nanoseconds since the trace began; 0 without a trace.
#[inline]
pub fn now() -> u64 {
    trace().map_or(0, |t| t.base.elapsed().as_nanos() as u64)
}

/// Whether events are recorded now.
#[inline]
pub fn on() -> bool {
    trace().is_some_and(|t| {
        let now = t.base.elapsed().as_nanos() as u64;
        now >= t.from && now < t.to
    })
}

fn thread_index(trace: &Trace) -> u64 {
    THREAD.with(|cell| {
        if cell.get() != 0 {
            return cell.get();
        }
        let index = THREADS.fetch_add(1, Ordering::Relaxed);
        cell.set(index);
        let name = std::thread::current()
            .name()
            .unwrap_or("unnamed")
            .to_owned();
        let mut bytes = [0u8; 24];
        let len = name.len().min(24);
        bytes[..len].copy_from_slice(&name.as_bytes()[..len]);
        let word = |i: usize| u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
        trace
            .records
            .lock()
            .unwrap()
            .push([NAME, index, 0, word(0), word(1), word(2)]);
        index
    })
}

/// Records event `kind` now.
#[inline]
pub fn ev(kind: u64, a: u64, b: u64, c: u64) {
    let Some(t) = trace() else { return };
    let now = t.base.elapsed().as_nanos() as u64;
    if now < t.from || now >= t.to {
        return;
    }
    let thread = thread_index(t);
    t.records.lock().unwrap().push([kind, thread, now, a, b, c]);
}
