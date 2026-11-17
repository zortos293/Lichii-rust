//! Diagnostic block trace (ZOR-421): with `LICHII_JIT_TRACE=1` the JIT dispatcher records every
//! block entry it runs on this host thread, and block linking and the indirect-branch table are
//! off so every block passes through the dispatcher.

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::ThreadContext;

const LEN: usize = 4096;

/// `[pc, x0, x19, x20, x30, tpidrro_el0]` at a block entry.
pub type Entry = [u64; 6];

thread_local! {
    static RING: RefCell<(usize, Vec<Entry>)> = const { RefCell::new((0, Vec::new())) };
}

/// Whether `LICHII_JIT_TRACE` asks for the trace.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LICHII_JIT_TRACE").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

pub(crate) fn record(ctx: &ThreadContext) {
    RING.with_borrow_mut(|(next, ring)| {
        if ring.is_empty() {
            ring.resize(LEN, [0; 6]);
        }
        ring[*next % LEN] = [
            ctx.pc,
            ctx.x[0],
            ctx.x[19],
            ctx.x[20],
            ctx.x[30],
            ctx.tpidrro_el0,
        ];
        *next += 1;
    });
}

/// The entries recorded on this host thread, oldest first.
pub fn recent() -> Vec<Entry> {
    RING.with_borrow(|(next, ring)| {
        let n = (*next).min(ring.len());
        (0..n).map(|i| ring[(*next - n + i) % LEN]).collect()
    })
}
