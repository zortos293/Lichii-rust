//! Pushbuffer parsing: GPFIFO entries, command headers and the method stream they encode.
//!
//! The parser keeps the method state of the hardware's PBDMA (current method, subchannel, words
//! left, increment mode) across GPFIFO entries, wherever the next entry is: NVN ends entries
//! inside a method's data and continues it in the next one, a subroutine entry elsewhere in
//! memory included (NX2_NOTES rows 51, 73).

use lichii_common::GpuVAddr;

use crate::bits::bitfield;

/// One GPFIFO entry: a pushbuffer segment in GPU memory.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct GpEntry(pub u64);

impl GpEntry {
    pub const fn new(address: GpuVAddr, words: u32) -> Self {
        Self((address & ((1 << 40) - 4)) | (((words as u64) & 0x1f_ffff) << 42))
    }

    /// GET\[39:2\]; bits 0 and 1 are the FETCH flags, not address bits.
    #[inline]
    pub const fn address(self) -> GpuVAddr {
        self.0 & ((1 << 40) - 4)
    }

    #[inline]
    pub const fn words(self) -> u32 {
        ((self.0 >> 42) & 0x1f_ffff) as u32
    }

    /// The SUBROUTINE fetch level bit.
    #[inline]
    pub const fn is_subroutine(self) -> bool {
        self.0 & (1 << 41) != 0
    }

    #[inline]
    pub const fn sync(self) -> bool {
        self.0 & (1 << 63) != 0
    }
}

impl core::fmt::Debug for GpEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "GpEntry({:#x}, {} words)", self.address(), self.words())
    }
}

/// How a command header's data words map to methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmissionMode {
    /// Each word goes to the next method.
    Increasing,
    /// Every word goes to the same method.
    NonIncreasing,
    /// The header carries one argument in its count field.
    Inline,
    /// The first word goes to the method, the rest to the method after it.
    IncreaseOnce,
}

bitfield! {
    /// The first word of a pushbuffer command.
    pub struct CommandHeader {
        method: u32 @ 0, 13;
        subchannel: u32 @ 13, 3;
        count: u32 @ 16, 13;
        mode: u32 @ 29, 3;
    }
}

impl CommandHeader {
    pub const fn new(mode: SubmissionMode, subchannel: u32, method: u32, count: u32) -> Self {
        let mode = match mode {
            SubmissionMode::Increasing => 1,
            SubmissionMode::NonIncreasing => 3,
            SubmissionMode::Inline => 4,
            SubmissionMode::IncreaseOnce => 5,
        };
        Self((mode << 29) | ((count & 0x1fff) << 16) | ((subchannel & 7) << 13) | (method & 0x1fff))
    }

    /// The submission mode; `None` for the legacy and reserved encodings, which are skipped.
    #[inline]
    pub fn submission_mode(self) -> Option<SubmissionMode> {
        match self.mode() {
            1 => Some(SubmissionMode::Increasing),
            3 => Some(SubmissionMode::NonIncreasing),
            4 => Some(SubmissionMode::Inline),
            5 => Some(SubmissionMode::IncreaseOnce),
            _ => None,
        }
    }
}

/// One method write the parser produced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MethodCall {
    pub subchannel: u32,
    pub method: u32,
    pub argument: u32,
    /// Data words of the same command after this one (0 on the last).
    pub remaining: u32,
}

/// Whether the parser continues after a method.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Continue,
    /// Stop after this word (an unmet semaphore acquire); the next `process` call resumes at
    /// the following word.
    Suspend,
}

/// Receives the method stream.
pub trait MethodHandler {
    fn call(&mut self, call: MethodCall) -> Flow;
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct MethodState {
    method: u32,
    subchannel: u32,
    count: u32,
    non_incrementing: bool,
    increment_once: bool,
}

/// Where a partly processed entry stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resume {
    pub entry: GpEntry,
    /// Words of the entry already processed.
    pub offset: u32,
}

/// The PBDMA's method state for one channel.
#[derive(Default)]
pub struct CommandParser {
    state: MethodState,
}

impl CommandParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds words of the current entry. Returns how many were consumed: all of them, or fewer
    /// when the handler suspended.
    pub fn process(&mut self, words: &[u32], handler: &mut impl MethodHandler) -> usize {
        let mut index = 0;
        while index < words.len() {
            let word = words[index];
            index += 1;
            let state = &mut self.state;
            if state.count != 0 {
                state.count -= 1;
                let call = MethodCall {
                    subchannel: state.subchannel,
                    method: state.method,
                    argument: word,
                    remaining: state.count,
                };
                if !state.non_incrementing {
                    state.method += 1;
                }
                if state.increment_once {
                    state.non_incrementing = true;
                }
                if handler.call(call) == Flow::Suspend {
                    return index;
                }
                continue;
            }
            let header = CommandHeader(word);
            let Some(mode) = header.submission_mode() else {
                continue;
            };
            if mode == SubmissionMode::Inline {
                let call = MethodCall {
                    subchannel: header.subchannel(),
                    method: header.method(),
                    argument: header.count(),
                    remaining: 0,
                };
                if handler.call(call) == Flow::Suspend {
                    return index;
                }
                continue;
            }
            *state = MethodState {
                method: header.method(),
                subchannel: header.subchannel(),
                count: header.count(),
                non_incrementing: mode == SubmissionMode::NonIncreasing,
                increment_once: mode == SubmissionMode::IncreaseOnce,
            };
        }
        index
    }

    /// Whether a command's data words are still expected.
    pub fn in_method(&self) -> bool {
        self.state.count != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Recorder {
        calls: Vec<(u32, u32, u32)>,
        suspend_on: Option<u32>,
    }

    impl MethodHandler for Recorder {
        fn call(&mut self, call: MethodCall) -> Flow {
            self.calls
                .push((call.subchannel, call.method, call.argument));
            if self.suspend_on == Some(call.method) {
                self.suspend_on = None;
                Flow::Suspend
            } else {
                Flow::Continue
            }
        }
    }

    fn h(mode: SubmissionMode, sub: u32, method: u32, count: u32) -> u32 {
        CommandHeader::new(mode, sub, method, count).0
    }

    #[test]
    fn decodes_every_submission_mode() {
        let words = [
            h(SubmissionMode::Increasing, 1, 0x100, 2),
            10,
            11,
            h(SubmissionMode::NonIncreasing, 2, 0x200, 2),
            20,
            21,
            h(SubmissionMode::Inline, 3, 0x300, 0x1234),
            h(SubmissionMode::IncreaseOnce, 0, 0x400, 3),
            40,
            41,
            42,
        ];
        let mut parser = CommandParser::new();
        let mut rec = Recorder::default();
        assert_eq!(parser.process(&words, &mut rec), words.len());
        assert_eq!(
            rec.calls,
            vec![
                (1, 0x100, 10),
                (1, 0x101, 11),
                (2, 0x200, 20),
                (2, 0x200, 21),
                (3, 0x300, 0x1234),
                (0, 0x400, 40),
                (0, 0x401, 41),
                (0, 0x401, 42),
            ]
        );
    }

    #[test]
    fn suspends_and_resumes_at_the_next_word() {
        let words = [h(SubmissionMode::Increasing, 0, 0x10, 3), 1, 2, 3];
        let mut parser = CommandParser::new();
        let mut rec = Recorder {
            suspend_on: Some(0x11),
            ..Recorder::default()
        };
        assert_eq!(parser.process(&words, &mut rec), 3);
        assert_eq!(parser.process(&words[3..], &mut rec), 1);
        assert_eq!(rec.calls, vec![(0, 0x10, 1), (0, 0x11, 2), (0, 0x12, 3)]);
    }

    #[test]
    fn method_data_continues_into_the_next_entry() {
        let mut parser = CommandParser::new();
        let mut rec = Recorder::default();
        parser.process(&[h(SubmissionMode::Increasing, 0, 0x10, 3), 1], &mut rec);
        parser.process(&[2, 3], &mut rec);
        assert_eq!(rec.calls, vec![(0, 0x10, 1), (0, 0x11, 2), (0, 0x12, 3)]);
    }

    #[test]
    fn method_data_continues_into_an_entry_elsewhere() {
        let mut parser = CommandParser::new();
        let mut rec = Recorder::default();
        parser.process(&[h(SubmissionMode::NonIncreasing, 0, 0x10, 3), 1], &mut rec);
        parser.process(&[2, 3], &mut rec);
        parser.process(&[h(SubmissionMode::Increasing, 1, 0x20, 1), 7], &mut rec);
        assert_eq!(
            rec.calls,
            vec![(0, 0x10, 1), (0, 0x10, 2), (0, 0x10, 3), (1, 0x20, 7)]
        );
        assert!(!parser.in_method());
    }

    #[test]
    fn gp_entry_fields() {
        let e = GpEntry::new(0x12_3456_7890, 0x400);
        assert_eq!(e.address(), 0x12_3456_7890);
        assert_eq!(e.words(), 0x400);
        assert!(!e.sync());
    }

    #[test]
    fn gp_entry_fetch_flags_are_not_address_bits() {
        let conditional_fetch = GpEntry(0x1000 | 1 | (8 << 42));
        assert_eq!(conditional_fetch.address(), 0x1000);
        assert_eq!(conditional_fetch.words(), 8);
    }
}
