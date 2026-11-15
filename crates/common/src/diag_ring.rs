use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

pub struct Entry {
    pub name: [u8; 28],
    pub command: u32,
    pub result: u32,
    pub micros: u64,
}

static RING: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());

fn epoch() -> &'static Instant {
    static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    E.get_or_init(Instant::now)
}

pub fn push(name: &str, command: u32, result: u32) {
    let mut entry = Entry {
        name: [0; 28],
        command,
        result,
        micros: epoch().elapsed().as_micros() as u64,
    };
    let len = name.len().min(28);
    entry.name[..len].copy_from_slice(&name.as_bytes()[..len]);
    let mut ring = RING.lock().unwrap();
    if ring.len() >= 400 {
        ring.pop_front();
    }
    ring.push_back(entry);
}

pub fn dump() {
    let ring = RING.lock().unwrap();
    let now = epoch().elapsed().as_micros() as u64;
    for e in ring.iter() {
        let end = e.name.iter().position(|&b| b == 0).unwrap_or(28);
        tracing::error!(
            "diag ipc -{}us {} cmd={:#x} result={:#x}",
            now.saturating_sub(e.micros),
            String::from_utf8_lossy(&e.name[..end]),
            e.command,
            e.result
        );
    }
}
