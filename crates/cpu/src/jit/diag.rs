//! Diag-only hooks for the DKB tonemap NaN hunt (scratch branch `diag/liru3-nan`, never merged).
//!
//! LICHII_DIAG_SIMD_INTERP: "all", or comma-separated mask:value hex pairs; SIMD/FP words that
//! match run on the interpreter through CallSimd. LICHII_DIAG_SIMD_KEEP takes the same pairs and
//! keeps matching words native, for bisecting.
//! LICHII_DIAG_NAN_STORES=1: every Store/StoreVec takes the slow path, and stores holding an f32
//! NaN lane are logged once per (pc, 1 MiB physical region).
//! LICHII_DIAG_NAN_MAKERS=1: CallSimd logs, once per pc and kind, words whose destination gets a
//! NaN or infinity lane that none of its sources held.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use lichii_memory::AddressSpace;

use crate::ThreadContext;

const MAX_LINES: u64 = 40_000;

fn flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

fn pairs(name: &str) -> Option<Vec<(u32, u32)>> {
    let v = std::env::var(name).ok()?;
    if v.trim() == "all" {
        return Some(Vec::new());
    }
    let hex = |s: &str| u32::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok();
    Some(
        v.split(',')
            .filter_map(|p| {
                let (m, k) = p.split_once(':')?;
                Some((hex(m)?, hex(k)?))
            })
            .collect(),
    )
}

fn start() -> Instant {
    static S: OnceLock<Instant> = OnceLock::new();
    *S.get_or_init(Instant::now)
}

fn t() -> f64 {
    start().elapsed().as_secs_f64()
}

/// Whether `word` must run on the interpreter.
pub(super) fn simd_interp(word: u32) -> bool {
    type Lists = (Option<Vec<(u32, u32)>>, Vec<(u32, u32)>);
    static V: OnceLock<Lists> = OnceLock::new();
    let (force, keep) = V.get_or_init(|| {
        let force = pairs("LICHII_DIAG_SIMD_INTERP");
        let keep = pairs("LICHII_DIAG_SIMD_KEEP").unwrap_or_default();
        tracing::warn!(?force, ?keep, "LIRU3 diag simd_interp");
        (force, keep)
    });
    let Some(force) = force else {
        return false;
    };
    let hit = |l: &[(u32, u32)]| l.iter().any(|&(m, k)| word & m == k);
    (force.is_empty() || hit(force)) && !hit(keep)
}

pub(super) fn nan_stores() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        let on = flag("LICHII_DIAG_NAN_STORES");
        tracing::warn!(on, "LIRU3 diag nan_stores");
        start();
        on
    })
}

fn makers() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| {
        let on = flag("LICHII_DIAG_NAN_MAKERS");
        tracing::warn!(on, "LIRU3 diag nan_makers");
        start();
        on
    })
}

fn nan32(w: u32) -> bool {
    w & 0x7f80_0000 == 0x7f80_0000 && w & 0x007f_ffff != 0 && w != u32::MAX
}

fn inf32(w: u32) -> bool {
    w & 0x7fff_ffff == 0x7f80_0000
}

fn nan64(w: u64) -> bool {
    w & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
        && w & 0x000f_ffff_ffff_ffff != 0
        && w != u64::MAX
}

fn inf64(w: u64) -> bool {
    w & 0x7fff_ffff_ffff_ffff == 0x7ff0_0000_0000_0000
}

/// Lane masks: f32 NaN, f32 inf, f64 NaN, f64 inf.
fn classify(v: u128) -> [u8; 4] {
    let mut m = [0u8; 4];
    for i in 0..4 {
        let w = (v >> (32 * i)) as u32;
        m[0] |= u8::from(nan32(w)) << i;
        m[1] |= u8::from(inf32(w)) << i;
    }
    for i in 0..2 {
        let w = (v >> (64 * i)) as u64;
        m[2] |= u8::from(nan64(w)) << i;
        m[3] |= u8::from(inf64(w)) << i;
    }
    m
}

fn code(memory: &AddressSpace, pc: u64, before: u64) -> String {
    (0..=before)
        .rev()
        .map(|k| {
            memory
                .read::<u32>(pc.wrapping_sub(4 * k))
                .map_or_else(|_| "????????".into(), |w| format!("{w:08x}"))
        })
        .collect::<Vec<_>>()
        .join(" ")
}

struct Log {
    seen: HashMap<(u64, u64), u64>,
    printed: u64,
    last: Instant,
}

impl Log {
    fn new() -> Mutex<Self> {
        Mutex::new(Self {
            seen: HashMap::new(),
            printed: 0,
            last: Instant::now(),
        })
    }

    /// Counts `key`; prints `line()` the first time. Every 10 s prints the 40 busiest keys.
    fn hit(&mut self, what: &str, key: (u64, u64), line: impl FnOnce() -> String) {
        let count = self.seen.entry(key).or_insert(0);
        *count += 1;
        if *count == 1 && self.printed < MAX_LINES {
            self.printed += 1;
            tracing::warn!("LIRU3 {what} t={:.3} {}", t(), line());
        }
        if self.last.elapsed() >= Duration::from_secs(10) {
            self.last = Instant::now();
            let mut top: Vec<_> = self.seen.iter().map(|(k, n)| (*n, *k)).collect();
            top.sort_unstable_by(|a, b| b.cmp(a));
            let list: Vec<String> = top
                .iter()
                .take(40)
                .map(|(n, (pc, r))| format!("{pc:#x}/{r:#x}={n}"))
                .collect();
            tracing::warn!(
                "LIRU3 {what}-top t={:.3} keys={} {}",
                t(),
                self.seen.len(),
                list.join(" ")
            );
        }
    }
}

/// Logs a guest store of `1 << size` bytes of `value` at `addr` when a 32-bit lane is a NaN.
pub(super) fn store(memory: &AddressSpace, pc: u64, addr: u64, size: u32, value: u128) {
    let lanes = match size {
        2 => 1,
        3 => 2,
        4 => 4,
        _ => return,
    };
    let mask = (0..lanes).fold(0u8, |m, i| {
        m | u8::from(nan32((value >> (32 * i)) as u32)) << i
    });
    if mask == 0 {
        return;
    }
    static LOG: OnceLock<Mutex<Log>> = OnceLock::new();
    let pa = memory.physical_address(addr).unwrap_or(u64::MAX);
    let Ok(mut log) = LOG.get_or_init(Log::new).lock() else {
        return;
    };
    log.hit("NANSTORE", (pc, pa >> 20), || {
        let bytes = 1usize << size;
        let v = value & (u128::MAX >> (128 - 8 * bytes));
        let width = 2 * bytes;
        format!(
            "pc={pc:#x} va={addr:#x} pa={pa:#x} size={bytes} lanes={mask:#x} value={v:0width$x} code=[{}]",
            code(memory, pc, 12)
        )
    });
}

fn sources(word: u32) -> ([u8; 3], usize) {
    let f = |low: u32| ((word >> low) & 31) as u8;
    let (rd, rn, rm, ra) = (f(0), f(5), f(16), f(10));
    if word & 0xff00_0000 == 0x1f00_0000 {
        return ([rn, rm, ra], 3);
    }
    let accumulate = matches!(word & 0xbfa0_fc00, 0x0e20_cc00 | 0x0ea0_cc00)
        || matches!(word & 0xdf80_b400, 0x0f80_1000 | 0x5f80_1000);
    if accumulate {
        ([rn, rm, rd], 3)
    } else {
        ([rn, rm, 0], 2)
    }
}

/// The sources of `word` before CallSimd runs it, when makers are logged.
pub(super) fn before(ctx: &ThreadContext, word: u32) -> Option<[u128; 3]> {
    if !makers() {
        return None;
    }
    let (regs, n) = sources(word);
    let mut v = [0u128; 3];
    for (slot, r) in v.iter_mut().zip(regs).take(n) {
        *slot = ctx.v[r as usize];
    }
    Some(v)
}

/// Logs `word` at `pc` when its destination now holds a NaN or infinity no source held.
pub(super) fn after(
    memory: &AddressSpace,
    ctx: &ThreadContext,
    word: u32,
    pc: u64,
    src: [u128; 3],
) {
    let out = ctx.v[(word & 31) as usize];
    let o = classify(out);
    if o == [0; 4] {
        return;
    }
    let n = sources(word).1;
    let s = src[..n].iter().fold([0u8; 4], |a, &v| {
        let c = classify(v);
        [a[0] | c[0], a[1] | c[1], a[2] | c[2], a[3] | c[3]]
    });
    let nan_in = s[0] | s[2] != 0;
    let inf_in = s[1] | s[3] != 0;
    let (kind, tag) = if o[0] | o[2] != 0 && !nan_in {
        if inf_in {
            ("nan-from-inf", 2)
        } else {
            ("nan", 1)
        }
    } else if o[1] | o[3] != 0 && !nan_in && !inf_in {
        ("inf", 3)
    } else {
        return;
    };
    static LOG: OnceLock<Mutex<Log>> = OnceLock::new();
    let Ok(mut log) = LOG.get_or_init(Log::new).lock() else {
        return;
    };
    log.hit("NANMAKER", (pc, tag), || {
        let srcs: Vec<String> = src[..n].iter().map(|v| format!("{v:032x}")).collect();
        format!(
            "kind={kind} pc={pc:#x} word={word:08x} out={out:032x} src=[{}] fpcr={:#x} code=[{}]",
            srcs.join(" "),
            ctx.fpcr,
            code(memory, pc, 6)
        )
    });
}
