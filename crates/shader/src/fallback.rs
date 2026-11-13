//! Census of the places where translation approximates or stubs guest behaviour instead of
//! failing: an ignored mode, a value that reads as zero, a texture variant mapped to another.
//! Off by default; when [`set_enabled`] turns it on, each fallback site the translator reaches
//! counts once per site and once per program, and the first sighting of each
//! (opcode, variant) pair logs one line. Nothing runs per draw: the counters move only while a
//! program translates.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

static ENABLED: AtomicBool = AtomicBool::new(false);
static REVISION: AtomicU64 = AtomicU64::new(0);
static CENSUS: Mutex<Vec<FallbackCount>> = Mutex::new(Vec::new());

thread_local! {
    /// The fallbacks the program translating on this thread already hit.
    static PROGRAM: RefCell<Vec<Fallback>> = const { RefCell::new(Vec::new()) };
}

/// One way translation approximates the guest: the guest opcode (or pass), the mode it
/// approximates, and the encoding field that selects it when the mode has many values (a
/// special register number, a constant buffer bank).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fallback {
    pub opcode: &'static str,
    pub variant: &'static str,
    pub value: Option<u32>,
}

/// How often a [`Fallback`] was hit since the census was turned on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FallbackCount {
    pub fallback: Fallback,
    /// Programs that hit it at least once.
    pub programs: u64,
    /// Instruction sites (or pass decisions) that hit it.
    pub sites: u64,
}

/// Turns the census on or off for every later translation.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Changes whenever a count changes, so a caller can log the census only when it moved.
pub fn revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}

/// Every fallback hit so far, the most widespread first.
pub fn census() -> Vec<FallbackCount> {
    let mut census = CENSUS
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    census.sort_by(|a, b| {
        (b.programs, b.sites)
            .cmp(&(a.programs, a.sites))
            .then_with(|| {
                let key =
                    |c: &FallbackCount| (c.fallback.opcode, c.fallback.variant, c.fallback.value);
                key(a).cmp(&key(b))
            })
    });
    census
}

/// A new program starts translating on this thread.
pub(crate) fn begin_program() {
    if ENABLED.load(Ordering::Relaxed) {
        PROGRAM.with(|hit| hit.borrow_mut().clear());
    }
}

/// The program `env` describes finished translating on this thread: logs the fallbacks it hit,
/// so a census can be tied to the programs of one draw.
pub(crate) fn end_program(env: &dyn crate::ShaderEnvironment) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let hit = PROGRAM.with(|hit| hit.borrow().clone());
    if hit.is_empty() {
        return;
    }
    let fallbacks: Vec<String> = hit
        .iter()
        .map(|f| match f.value {
            Some(value) => format!("{}/{}={value}", f.opcode, f.variant),
            None => format!("{}/{}", f.opcode, f.variant),
        })
        .collect();
    tracing::info!(
        target: "shader_fallback",
        stage = ?env.stage(),
        address = format_args!("{:#x}", env.start_address()),
        "program fallbacks {fallbacks:?}"
    );
}

/// Translation of the current program approximates `opcode` in mode `variant`.
pub(crate) fn note(opcode: &'static str, variant: &'static str) {
    record(opcode, variant, None);
}

/// Like [`note`] for a mode an encoding field selects, counted per field value.
pub(crate) fn note_value(opcode: &'static str, variant: &'static str, value: u32) {
    record(opcode, variant, Some(value));
}

fn record(opcode: &'static str, variant: &'static str, value: Option<u32>) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let fallback = Fallback {
        opcode,
        variant,
        value,
    };
    let new_in_program = PROGRAM.with(|hit| {
        let mut hit = hit.borrow_mut();
        let new = !hit.contains(&fallback);
        if new {
            hit.push(fallback);
        }
        new
    });
    let mut census = CENSUS.lock().unwrap_or_else(PoisonError::into_inner);
    match census.iter_mut().find(|count| count.fallback == fallback) {
        Some(count) => {
            count.sites += 1;
            count.programs += u64::from(new_in_program);
        }
        None => {
            tracing::warn!(
                target: "shader_fallback",
                opcode,
                variant,
                value,
                "translation approximates a guest mode"
            );
            census.push(FallbackCount {
                fallback,
                programs: 1,
                sites: 1,
            });
        }
    }
    REVISION.fetch_add(1, Ordering::Relaxed);
}

/// Serializes the tests that switch the census on or off, which is process-global, and
/// switches it on.
#[cfg(test)]
pub(crate) fn test_census() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    set_enabled(true);
    guard
}

/// How many sites hit `opcode` in mode `variant` (for `value`) since the census was turned on.
#[cfg(test)]
pub(crate) fn sites(opcode: &str, variant: &str, value: Option<u32>) -> u64 {
    census()
        .into_iter()
        .find(|c| {
            c.fallback.opcode == opcode
                && c.fallback.variant == variant
                && c.fallback.value == value
        })
        .map_or(0, |c| c.sites)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_sites_and_programs_once_enabled() {
        let _guard = test_census();
        set_enabled(false);
        let count = |opcode| {
            census()
                .into_iter()
                .find(|c| c.fallback.opcode == opcode)
                .map(|c| (c.programs, c.sites))
        };
        note("TEST_OFF", "disabled");
        assert_eq!(count("TEST_OFF"), None);
        set_enabled(true);
        let before = revision();
        begin_program();
        note("TEST_ON", "mode");
        note("TEST_ON", "mode");
        begin_program();
        note("TEST_ON", "mode");
        note_value("TEST_ON", "mode", 7);
        assert_eq!(count("TEST_ON"), Some((2, 3)));
        let valued = census()
            .into_iter()
            .find(|c| c.fallback.value == Some(7))
            .map(|c| (c.fallback.opcode, c.programs, c.sites));
        assert_eq!(valued, Some(("TEST_ON", 1, 1)));
        assert!(revision() > before);
    }
}
