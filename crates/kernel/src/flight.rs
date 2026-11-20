//! The flight recorder: each guest thread's latest supervisor calls and service requests, kept
//! in a small ring that costs no allocation, and written to the log when its process crashes so
//! a crash report shows what led to it in any title, without tracing everything.

use std::fmt::Write;

/// Events each thread keeps.
pub(crate) const ENTRIES: usize = 48;

/// Bytes of a service name kept.
const NAME: usize = 24;

#[derive(Clone, Copy, Default)]
pub(crate) enum Event {
    #[default]
    None,
    /// A supervisor call: its number, x0..x3 on entry, x0..x1 on return, and the caller's pc.
    Svc {
        imm: u32,
        args: [u64; 4],
        out: [u64; 2],
        pc: u64,
    },
    /// A request a service of the emulator answered.
    Ipc {
        name: [u8; NAME],
        len: u8,
        command: u32,
        result: u32,
    },
}

impl Event {
    pub(crate) fn ipc(service: &str, command: u32, result: u32) -> Self {
        let mut name = [0; NAME];
        let len = service.len().min(NAME);
        name[..len].copy_from_slice(&service.as_bytes()[..len]);
        Self::Ipc {
            name,
            len: len as u8,
            command,
            result,
        }
    }
}

/// One thread's ring of events.
pub(crate) struct Recorder {
    events: [Event; ENTRIES],
    next: usize,
}

impl Default for Recorder {
    fn default() -> Self {
        Self {
            events: [Event::None; ENTRIES],
            next: 0,
        }
    }
}

impl Recorder {
    pub(crate) fn push(&mut self, event: Event) {
        self.events[self.next] = event;
        self.next = (self.next + 1) % ENTRIES;
    }

    /// The events from the oldest to the latest, one line each.
    pub(crate) fn lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for i in 0..ENTRIES {
            let mut line = String::new();
            match self.events[(self.next + i) % ENTRIES] {
                Event::None => continue,
                Event::Svc { imm, args, out, pc } => {
                    let _ = write!(
                        line,
                        "svc {imm:#04x} at {pc:#x} args [{:x} {:x} {:x} {:x}] -> [{:x} {:x}]",
                        args[0], args[1], args[2], args[3], out[0], out[1]
                    );
                }
                Event::Ipc {
                    name,
                    len,
                    command,
                    result,
                } => {
                    let name = String::from_utf8_lossy(&name[..usize::from(len)]);
                    let _ = write!(line, "ipc {name} command {command} -> {result:#x}");
                }
            }
            lines.push(line);
        }
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ring_keeps_the_latest_events_in_order() {
        let mut recorder = Recorder::default();
        for i in 0..(ENTRIES as u32 + 3) {
            recorder.push(Event::ipc("fsp-srv", i, 0));
        }
        let lines = recorder.lines();
        assert_eq!(lines.len(), ENTRIES);
        assert_eq!(lines[0], "ipc fsp-srv command 3 -> 0x0");
        assert_eq!(
            lines.last().unwrap(),
            &format!("ipc fsp-srv command {} -> 0x0", ENTRIES + 2)
        );
    }

    #[test]
    fn long_service_names_are_cut() {
        let mut recorder = Recorder::default();
        recorder.push(Event::ipc("a-very-long-service-name-indeed", 1, 2));
        assert_eq!(
            recorder.lines(),
            ["ipc a-very-long-service-name command 1 -> 0x2"]
        );
    }
}
