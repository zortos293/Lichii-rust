//! Opt-in JIT counters for ranking JIT work on real titles. `LICHII_JIT_PROFILE=1` turns them on
//! for every core built afterwards. Unset, no counter exists and translated code is unchanged.
//!
//! Each core counts into its own [`Profile`] without locks. When `run` returns, at most every
//! [`FLUSH_INTERVAL`], it sends the counts down a channel, which never blocks, and drains the
//! channel into the process-wide total if that is not locked. The frame view's [`summary`] and
//! dropping cores lock the total and drain it too. The last profiled core to drop logs the total
//! as a table.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt::Write;
use std::mem::{Discriminant, discriminant};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::decoder::Inst;

pub(super) const FLUSH_INTERVAL: Duration = Duration::from_millis(250);
const TABLE_ROWS: usize = 40;

static ENABLED: LazyLock<bool> = LazyLock::new(|| {
    std::env::var_os("LICHII_JIT_PROFILE").is_some_and(|v| !v.is_empty() && v != "0")
});

static SINK: LazyLock<Sink> = LazyLock::new(|| {
    let (tx, rx) = mpsc::channel();
    Sink {
        tx,
        global: Mutex::new(Global {
            rx,
            total: Profile::default(),
            live: 0,
            cores: 0,
            started: None,
            last: None,
            line: String::new(),
        }),
    }
});

/// Whether `LICHII_JIT_PROFILE` asks for the profile.
pub(super) fn enabled() -> bool {
    *ENABLED
}

/// Why translated code returned to the dispatcher.
#[derive(Clone, Copy)]
pub(super) enum Exit {
    /// A direct jump not linked yet, or whose target never links (an interpreted instruction).
    Unlinked,
    IndirectMiss,
    Interrupt,
    Fault,
}

const EXITS: [&str; 4] = [
    "unlinked jump",
    "indirect-branch miss",
    "interrupt",
    "fault",
];

#[derive(Clone, Copy)]
pub(super) enum Flush {
    /// The code region filled up during a translation.
    Full,
    Clear,
    Invalidate,
}

const FLUSHES: [&str; 3] = ["region full", "clear", "invalidate range"];

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Class {
    Inst(Discriminant<Inst>, u8),
    Simd(u32),
}

#[derive(Default)]
struct Counts(HashMap<Class, (String, u64)>);

impl Counts {
    fn add(&mut self, class: Class, name: impl FnOnce() -> String) {
        self.0.entry(class).or_insert_with(|| (name(), 0)).1 += 1;
    }

    fn merge(&mut self, other: Counts) {
        for (class, (name, n)) in other.0 {
            match self.0.entry(class) {
                Entry::Occupied(mut e) => e.get_mut().1 += n,
                Entry::Vacant(e) => {
                    e.insert((name, n));
                }
            }
        }
    }

    fn total(&self) -> u64 {
        self.0.values().map(|&(_, n)| n).sum()
    }

    fn section(&self, out: &mut String, title: &str) {
        if self.0.is_empty() {
            return;
        }
        let total = self.total();
        let mut rows: Vec<(&str, u64)> = self.0.values().map(|(s, n)| (s.as_str(), *n)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        let pct = |n: u64| n as f64 * 100.0 / total as f64;
        let _ = writeln!(out, "  {title}: {total}");
        for &(name, n) in rows.iter().take(TABLE_ROWS) {
            let _ = writeln!(out, "    {n:>12} {:>5.1}%  {name}", pct(n));
        }
        if rows.len() > TABLE_ROWS {
            let rest: u64 = rows[TABLE_ROWS..].iter().map(|&(_, n)| n).sum();
            let more = rows.len() - TABLE_ROWS;
            let _ = writeln!(
                out,
                "    {rest:>12} {:>5.1}%  {more} more classes",
                pct(rest)
            );
        }
    }
}

/// One core's counts since its last merge into the process-wide total.
#[derive(Default)]
pub(super) struct Profile {
    fallbacks: Counts,
    simd_calls: Counts,
    slow_fpcr: Counts,
    slow_inputs: Counts,
    exits: [u64; 4],
    /// Lazy NZCV written back by the native stub and by the dispatcher.
    materialized: [u64; 2],
    flushes: [u64; 3],
    blocks: u64,
    instructions: u64,
    code_bytes: u64,
    translate: Duration,
}

impl Profile {
    /// An instruction the translator rejected, run on the interpreter.
    pub(super) fn fallback(&mut self, inst: &Inst) {
        let class = match *inst {
            Inst::SimdFp { word } => Class::Simd(simd_key(word)),
            Inst::VecMulti { load, selem, .. } => {
                Class::Inst(discriminant(inst), selem << 1 | u8::from(load))
            }
            _ => Class::Inst(discriminant(inst), 0),
        };
        self.fallbacks.add(class, || match *inst {
            Inst::SimdFp { word } => simd_name(word),
            Inst::VecMulti { load, selem, .. } => {
                format!(
                    "{}{selem} (multiple structures)",
                    if load { "LD" } else { "ST" }
                )
            }
            _ => {
                let debug = format!("{inst:?}");
                let end = debug
                    .find(|c: char| !c.is_alphanumeric())
                    .unwrap_or(debug.len());
                debug[..end].to_owned()
            }
        });
    }

    /// An `Op::CallSimd`: a SIMD/FP instruction the JIT does not lower natively.
    pub(super) fn simd_call(&mut self, word: u32) {
        self.simd_calls
            .add(Class::Simd(simd_key(word)), || simd_name(word));
    }

    /// A natively lowered SIMD/FP op that took its slow path, because FPCR selects a mode the
    /// host does not provide (`fpcr`) or because of its inputs or a clear FPSR.IXC.
    pub(super) fn simd_slow(&mut self, word: u32, fpcr: bool) {
        let counts = if fpcr {
            &mut self.slow_fpcr
        } else {
            &mut self.slow_inputs
        };
        counts.add(Class::Simd(simd_key(word)), || simd_name(word));
    }

    pub(super) fn exit(&mut self, exit: Exit) {
        self.exits[exit as usize] += 1;
    }

    pub(super) fn materialized(&mut self, native: u64, dispatcher: u64) {
        self.materialized[0] += native;
        self.materialized[1] += dispatcher;
    }

    pub(super) fn flush(&mut self, flush: Flush) {
        self.flushes[flush as usize] += 1;
    }

    /// One translation: `code_bytes` is 0 when the block's first instruction is interpreted.
    pub(super) fn translated(&mut self, time: Duration, instructions: u64, code_bytes: usize) {
        self.translate += time;
        if code_bytes > 0 {
            self.blocks += 1;
            self.instructions += instructions;
            self.code_bytes += code_bytes as u64;
        }
    }

    fn merge(&mut self, other: Profile) {
        self.fallbacks.merge(other.fallbacks);
        self.simd_calls.merge(other.simd_calls);
        self.slow_fpcr.merge(other.slow_fpcr);
        self.slow_inputs.merge(other.slow_inputs);
        let add = |a: &mut [u64], b: &[u64]| a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        add(&mut self.exits, &other.exits);
        add(&mut self.materialized, &other.materialized);
        add(&mut self.flushes, &other.flushes);
        self.blocks += other.blocks;
        self.instructions += other.instructions;
        self.code_bytes += other.code_bytes;
        self.translate += other.translate;
    }

    /// Translation microseconds, interpreter fallbacks, and SIMD helper calls and slow paths,
    /// for rates.
    fn rates(&self) -> [u64; 3] {
        [
            self.translate.as_micros() as u64,
            self.fallbacks.total(),
            self.simd_calls.total() + self.slow_fpcr.total() + self.slow_inputs.total(),
        ]
    }

    pub(super) fn table(&self, cores: usize, wall: Duration) -> String {
        let mut out = String::new();
        let ms = self.translate.as_secs_f64() * 1e3;
        let per_block = ms * 1e3 / self.blocks.max(1) as f64;
        let named = |names: &[&str], counts: &[u64]| {
            names
                .iter()
                .zip(counts)
                .map(|(name, n)| format!("{name} {n}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let _ = writeln!(
            out,
            "JIT profile: {cores} cores over {:.1} s",
            wall.as_secs_f64()
        );
        let _ = writeln!(
            out,
            "  translation: {} blocks, {} instructions, {:.1} KiB of code, {ms:.1} ms ({per_block:.1} us per block)",
            self.blocks,
            self.instructions,
            self.code_bytes as f64 / 1024.0,
        );
        let _ = writeln!(out, "  dispatcher exits: {}", named(&EXITS, &self.exits));
        let _ = writeln!(
            out,
            "  lazy NZCV write-backs: {}",
            named(&["native", "dispatcher"], &self.materialized)
        );
        let _ = writeln!(
            out,
            "  code cache flushes: {}",
            named(&FLUSHES, &self.flushes)
        );
        self.fallbacks.section(&mut out, "interpreter fallbacks");
        self.simd_calls.section(&mut out, "CallSimd calls");
        self.slow_fpcr
            .section(&mut out, "native FP slow paths, FPCR mode");
        self.slow_inputs
            .section(&mut out, "native FP slow paths, inputs or FPSR.IXC");
        out
    }
}

/// Cores send their counts through `tx`; only the frame view and dropping cores lock `global`.
struct Sink {
    tx: Sender<Profile>,
    global: Mutex<Global>,
}

struct Global {
    rx: Receiver<Profile>,
    total: Profile,
    live: usize,
    cores: usize,
    started: Option<Instant>,
    last: Option<(Instant, [u64; 3])>,
    line: String,
}

impl Global {
    fn drain(&mut self) {
        while let Ok(profile) = self.rx.try_recv() {
            self.total.merge(profile);
        }
    }
}

fn global() -> MutexGuard<'static, Global> {
    SINK.global.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts a new profiled core.
pub(super) fn register() {
    let mut g = global();
    g.live += 1;
    g.cores += 1;
    g.started.get_or_insert_with(Instant::now);
}

/// Hands a running core's counts to the total without waiting for a lock: when the total is free,
/// this core drains the channel into it, so the channel stays short with the frame view off.
pub(super) fn merge(profile: Profile) {
    let _ = SINK.tx.send(profile);
    if let Ok(mut g) = SINK.global.try_lock() {
        g.drain();
    }
}

/// Merges a dropped core's last counts; the last live core logs the table and resets the total.
pub(super) fn retire(profile: Profile) {
    let mut g = global();
    g.total.merge(profile);
    g.live -= 1;
    if g.live > 0 {
        return;
    }
    g.drain();
    let total = std::mem::take(&mut g.total);
    let cores = std::mem::take(&mut g.cores);
    let wall = g.started.take().map_or(Duration::ZERO, |t| t.elapsed());
    g.last = None;
    drop(g);
    tracing::info!("{}", total.table(cores, wall));
}

/// Translation time, interpreter fallbacks and SIMD helper calls per second, over the last
/// second; `None` unless `LICHII_JIT_PROFILE` is set.
pub fn summary() -> Option<String> {
    if !enabled() {
        return None;
    }
    let mut g = global();
    g.drain();
    let now = Instant::now();
    let rates = g.total.rates();
    match g.last {
        Some((at, last)) if now - at >= Duration::from_secs(1) => {
            let secs = (now - at).as_secs_f64();
            let rate = |i: usize| rates[i].saturating_sub(last[i]) as f64 / secs;
            g.line = format!(
                "JIT/s tr {}ms int {} simd {}",
                compact(rate(0) / 1e3),
                compact(rate(1)),
                compact(rate(2)),
            );
            g.last = Some((now, rates));
        }
        Some(_) => {}
        None => {
            g.line = "JIT profile on".to_owned();
            g.last = Some((now, rates));
        }
    }
    Some(g.line.clone())
}

/// At most four characters: `7`, `950`, `1.2k`, `85k`, `3.1M`.
fn compact(n: f64) -> String {
    let short = |v: f64, unit: &str| match v {
        v if v < 9.95 => format!("{v:.1}{unit}"),
        v => format!("{v:.0}{unit}"),
    };
    match n {
        n if n >= 999.5e3 => short(n / 1e6, "M"),
        n if n >= 999.5 => short(n / 1e3, "k"),
        n => format!("{n:.0}"),
    }
}

/// The encoding group of a SIMD/FP data-processing instruction or an `MSR`/`MRS`, after the
/// Arm ARM's top-level decode, and the mask of its register and immediate fields.
fn simd_group(word: u32) -> (&'static str, u32) {
    const RD_RN: u32 = 0x3ff;
    const RD_RN_RM: u32 = 0x001f_03ff;
    if word & 0xffd0_0000 == 0xd510_0000 {
        return ("MSR/MRS", 0x1f);
    }
    if word & 0xff3e_0c00 == 0x4e28_0800 {
        return ("AES", RD_RN);
    }
    if word & 0xff20_8c00 == 0x5e00_0000 {
        return ("SHA 3-reg", RD_RN_RM);
    }
    if word & 0xff3e_0c00 == 0x5e28_0800 {
        return ("SHA 2-reg", RD_RN);
    }
    if word & 0x5f00_0000 == 0x1f00_0000 {
        return ("FP 3-source", 0x001f_7fff);
    }
    if word & 0x5f20_0000 == 0x1e00_0000 {
        return ("FP<->fixed", 0xffff);
    }
    if word & 0x5f20_0000 == 0x1e20_0000 {
        return match word >> 10 & 3 {
            1 => ("FP cond compare", 0x001f_f3ef),
            2 => ("FP 2-source", RD_RN_RM),
            3 => ("FP cond select", 0x001f_f3ff),
            _ if word & 0xfc00 == 0 => ("FP<->int", RD_RN),
            _ if word & 0x7c00 == 0x4000 => ("FP 1-source", RD_RN),
            _ if word & 0x3c00 == 0x2000 => ("FP compare", 0x001f_03e0),
            _ if word & 0x1c00 == 0x1000 => ("FP immediate", 0x001f_e01f),
            _ => ("FP other", RD_RN_RM),
        };
    }
    let scalar = word & 0xdf00_0000 == 0x5e00_0000 || word & 0xdf00_0000 == 0x5f00_0000;
    let vector = word & 0x9f00_0000 == 0x0e00_0000 || word & 0x9f00_0000 == 0x0f00_0000;
    if !scalar && !vector {
        return ("SIMD/FP other", RD_RN_RM);
    }
    let name = |s: &'static str, v: &'static str| if scalar { s } else { v };
    if word & 1 << 24 != 0 {
        return match word >> 10 & 1 {
            0 => (
                name("SIMD scalar by element", "SIMD by element"),
                0x003f_0bff,
            ),
            _ if word & 0x0078_0000 == 0 && vector => ("SIMD modified immediate", 0x0007_03ff),
            _ => (name("SIMD scalar shift", "SIMD shift"), 0x0007_03ff),
        };
    }
    if word & 1 << 21 != 0 {
        return match word >> 10 & 3 {
            1 | 3 => (name("SIMD scalar 3-same", "SIMD 3-same"), RD_RN_RM),
            0 => (
                name("SIMD scalar 3-different", "SIMD 3-different"),
                RD_RN_RM,
            ),
            _ if word & 0x001e_0000 == 0 => {
                (name("SIMD scalar 2-reg misc", "SIMD 2-reg misc"), RD_RN)
            }
            _ if word & 0x001e_0000 == 0x0010_0000 => {
                (name("SIMD scalar pairwise", "SIMD across lanes"), RD_RN)
            }
            _ => (name("SIMD scalar other", "SIMD other"), RD_RN_RM),
        };
    }
    match word {
        w if w & 0x01e0_8400 == 0x0000_0400 => (name("SIMD scalar copy", "SIMD copy"), RD_RN),
        w if vector && w & 0xbf20_8c00 == 0x0e00_0800 => ("SIMD permute", RD_RN_RM),
        w if vector && w & 0xbf20_8c00 == 0x0e00_0000 => ("SIMD table lookup", RD_RN_RM),
        w if vector && w & 0xbf20_8400 == 0x2e00_0000 => ("SIMD extract", 0x001f_7bff),
        _ => (name("SIMD scalar other", "SIMD other"), RD_RN_RM),
    }
}

fn simd_key(word: u32) -> u32 {
    word & !simd_group(word).1
}

fn simd_name(word: u32) -> String {
    let (group, mask) = simd_group(word);
    format!("{group} {:#010x}", word & !mask)
}

#[cfg(test)]
mod tests {
    use super::{compact, simd_name};

    #[test]
    fn compact_numbers_fit_four_characters() {
        for (n, s) in [
            (7.0, "7"),
            (950.0, "950"),
            (999.6, "1.0k"),
            (1234.0, "1.2k"),
            (85_000.0, "85k"),
            (999_700.0, "1.0M"),
            (3.14e6, "3.1M"),
            (42e6, "42M"),
        ] {
            assert_eq!(compact(n), s);
        }
    }

    #[test]
    fn simd_groups_follow_the_top_level_decode() {
        for (word, name) in [
            (0x1e62_2820, "FP 2-source 0x1e602800"),
            (0x1f42_0c20, "FP 3-source 0x1f400000"),
            (0x1e61_2000, "FP compare 0x1e602000"),
            (0x1e6e_1000, "FP immediate 0x1e601000"),
            (0x9e78_0000, "FP<->int 0x9e780000"),
            (0x1e62_0c20, "FP cond select 0x1e600c00"),
            (0x1e61_0400, "FP cond compare 0x1e600400"),
            (0x9e42_f420, "FP<->fixed 0x9e420000"),
            (0x1e61_c020, "FP 1-source 0x1e61c000"),
            (0x4ea2_8420, "SIMD 3-same 0x4ea08400"),
            (0x0e62_0020, "SIMD 3-different 0x0e600000"),
            (0x6e02_1820, "SIMD extract 0x6e000000"),
            (0x4e82_3820, "SIMD permute 0x4e803800"),
            (0x4e02_0020, "SIMD table lookup 0x4e000000"),
            (0x4e04_0c20, "SIMD copy 0x4e040c00"),
            (0x4f00_0420, "SIMD modified immediate 0x4f000400"),
            (0x4f23_5420, "SIMD shift 0x4f205400"),
            (0x4fa2_9020, "SIMD by element 0x4f809000"),
            (0x4eb1_b820, "SIMD across lanes 0x4eb1b800"),
            (0x4ea0_b820, "SIMD 2-reg misc 0x4ea0b800"),
            (0x4ea1_d820, "SIMD 2-reg misc 0x4ea1d800"),
            (0x4e28_4820, "AES 0x4e284800"),
            (0x5e28_0820, "SHA 2-reg 0x5e280800"),
            (0x5e02_0020, "SHA 3-reg 0x5e000000"),
            (0xd53b_4400, "MSR/MRS 0xd53b4400"),
            (0x7e30_d820, "SIMD scalar pairwise 0x7e30d800"),
            (0x5ee2_8420, "SIMD scalar 3-same 0x5ee08400"),
        ] {
            assert_eq!(simd_name(word), name, "{word:#010x}");
        }
    }
}
