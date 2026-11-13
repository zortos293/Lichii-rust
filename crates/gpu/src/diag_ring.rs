use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

const LEN: usize = 96;

thread_local! {
    static RING: RefCell<(usize, [(u8, u32, u32); LEN])> = const { RefCell::new((0, [(0, 0, 0); LEN])) };
}

pub fn note(engine: u8, method: u32, argument: u32) {
    RING.with(|r| {
        let mut r = r.borrow_mut();
        let i = r.0;
        r.1[i % LEN] = (engine, method, argument);
        r.0 = i + 1;
    });
}

pub fn dump() -> String {
    RING.with(|r| {
        let r = r.borrow();
        let n = r.0.min(LEN);
        let mut out = String::new();
        for k in (r.0 - n)..r.0 {
            let (e, m, a) = r.1[k % LEN];
            out.push_str(&format!("{}:{:#x}={:#x} ", e as char, m * 4, a));
        }
        out
    })
}

pub fn first(counter: &AtomicU32, limit: u32) -> bool {
    counter.fetch_add(1, Ordering::Relaxed) < limit
}
