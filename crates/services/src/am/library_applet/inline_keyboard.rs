//! The inline software keyboard's state machine. The title drives it with requests in the
//! interactive in-data; am keeps the keyboard's state, turns each request into replies and
//! frontend events, and encodes what the user typed as the replies the title pops.

use super::{KEYBOARD_STRING_SIZE, text_lengths, u32_at, u64_at, utf16_at, write_text};
use crate::am::applets::{InlineKeyboardAppear, InlineKeyboardEvent, InlineKeyboardReply};

/// `SwkbdState`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum State {
    #[default]
    NotInitialized = 0,
    Hidden = 1,
    Appearing = 2,
    Shown = 3,
    Disappearing = 4,
}

/// `SwkbdRequestCommand`.
const FINALIZE: u32 = 0x4;
const SET_USER_WORD_INFO: u32 = 0x6;
const SET_CUSTOMIZE_DIC: u32 = 0x7;
const CALC: u32 = 0xa;
const SET_CUSTOMIZED_DICTIONARIES: u32 = 0xb;
const UNSET_CUSTOMIZED_DICTIONARIES: u32 = 0xc;
const SET_CHANGED_STRING_V2_FLAG: u32 = 0xd;
const SET_MOVED_CURSOR_V2_FLAG: u32 = 0xe;

/// `SwkbdReplyType`.
const FINISHED_INITIALIZE: u32 = 0x0;
const DEFAULT: u32 = 0x1;
const CHANGED_STRING: u32 = 0x2;
const MOVED_CURSOR: u32 = 0x3;
const DECIDED_ENTER: u32 = 0x5;
const DECIDED_CANCEL: u32 = 0x6;
const CHANGED_STRING_UTF8: u32 = 0x7;
const MOVED_CURSOR_UTF8: u32 = 0x8;
const DECIDED_ENTER_UTF8: u32 = 0x9;
const UNSET_CUSTOMIZE_DIC_REPLY: u32 = 0xa;
const RELEASED_USER_WORD_INFO: u32 = 0xb;
const UNSET_CUSTOMIZED_DICTIONARIES_REPLY: u32 = 0xc;
const CHANGED_STRING_V2: u32 = 0xd;
const MOVED_CURSOR_V2: u32 = 0xe;
const CHANGED_STRING_UTF8_V2: u32 = 0xf;
const MOVED_CURSOR_UTF8_V2: u32 = 0x10;

/// `SwkbdCalcArgFlags`.
const SET_INITIALIZE_ARG: u64 = 1 << 0;
const APPEAR: u64 = 1 << 2;
const SET_INPUT_TEXT: u64 = 1 << 3;
const SET_CURSOR_POSITION: u64 = 1 << 4;
const SET_UTF8_MODE: u64 = 1 << 5;
const UNSET_CUSTOMIZE_DIC: u64 = 1 << 6;
const DISAPPEAR: u64 = 1 << 7;
const UNSET_USER_WORD_INFO: u64 = 1 << 10;

/// A calc request is the command, `SwkbdCalcArgCommon`, then the old or new calc argument.
const CALC_COMMON: usize = 4;
const CALC_ARG: usize = CALC_COMMON + 0x18;
const INPUT_TEXT_UNITS: usize = 0x1fa;
const UTF16_TEXT_SIZE: usize = 0x3ec;

/// Offsets in `SwkbdCalcArgOld` and `SwkbdCalcArgNew`; `size` is the `calc_arg_size` that names
/// the layout, the common part included.
struct CalcLayout {
    size: usize,
    appear: usize,
    cursor: usize,
    input_text: usize,
    utf8: usize,
    backspace: usize,
}

const CALC_OLD: CalcLayout = CalcLayout {
    size: 0x4a0,
    appear: 0x8,
    cursor: 0x4,
    input_text: 0x50,
    utf8: 0x444,
    backspace: 0x446,
};

const CALC_NEW: CalcLayout = CalcLayout {
    size: 0x4e8,
    appear: 0x0,
    cursor: 0x74,
    input_text: 0x78,
    utf8: 0x46c,
    backspace: 0x46e,
};

/// What a request did: the replies to push, the events for the frontend, and whether the title
/// finalized the keyboard.
#[derive(Default)]
pub(super) struct Outcome {
    pub replies: Vec<Vec<u8>>,
    pub events: Vec<InlineKeyboardEvent>,
    pub finalized: bool,
}

#[derive(Default)]
pub(super) struct InlineKeyboard {
    state: State,
    utf8: bool,
    changed_string_v2: bool,
    moved_cursor_v2: bool,
    text: String,
    cursor: i32,
}

impl InlineKeyboard {
    pub fn request(&mut self, data: &[u8]) -> Outcome {
        let mut outcome = Outcome::default();
        let flag = || data.get(4).is_some_and(|&byte| byte != 0);
        match u32_at(data, 0) {
            FINALIZE => {
                self.change_state(State::NotInitialized, &mut outcome.replies);
                outcome.events.push(InlineKeyboardEvent::Finalize);
                outcome.finalized = true;
            }
            SET_USER_WORD_INFO => outcome
                .replies
                .push(self.reply_base(RELEASED_USER_WORD_INFO)),
            UNSET_CUSTOMIZED_DICTIONARIES => outcome
                .replies
                .push(self.reply_base(UNSET_CUSTOMIZED_DICTIONARIES_REPLY)),
            SET_CUSTOMIZE_DIC | SET_CUSTOMIZED_DICTIONARIES => {
                tracing::debug!("inline software keyboard ignores custom dictionaries");
            }
            CALC => self.calc(data, &mut outcome),
            SET_CHANGED_STRING_V2_FLAG => self.changed_string_v2 = flag(),
            SET_MOVED_CURSOR_V2_FLAG => self.moved_cursor_v2 = flag(),
            command => tracing::warn!(command, "unknown inline software keyboard request"),
        }
        outcome
    }

    fn calc(&mut self, data: &[u8], outcome: &mut Outcome) {
        let size = (u32_at(data, CALC_COMMON + 4) & 0xffff) as usize;
        let flags = u64_at(data, CALC_COMMON + 8);
        let layout = if size == CALC_OLD.size {
            &CALC_OLD
        } else {
            &CALC_NEW
        };
        if size != layout.size {
            tracing::warn!(
                size,
                "unknown inline keyboard calc size, reading the new layout"
            );
        }
        if data.len() < CALC_COMMON + layout.size {
            tracing::warn!(
                len = data.len(),
                "inline keyboard calc request is too short"
            );
            return;
        }
        let arg = &data[CALC_ARG..];
        let flag = |bit: u64| flags & bit != 0;
        if flag(SET_INPUT_TEXT) {
            self.text = utf16_at(arg, layout.input_text, INPUT_TEXT_UNITS);
        }
        if flag(SET_CURSOR_POSITION) {
            self.cursor = u32_at(arg, layout.cursor) as i32;
        }
        if flag(SET_UTF8_MODE) {
            self.utf8 = arg[layout.utf8] != 0;
        }
        if self.state <= State::Hidden && flag(UNSET_CUSTOMIZE_DIC) {
            outcome
                .replies
                .push(self.reply_base(UNSET_CUSTOMIZE_DIC_REPLY));
        }
        if self.state <= State::Hidden && flag(UNSET_USER_WORD_INFO) {
            outcome
                .replies
                .push(self.reply_base(RELEASED_USER_WORD_INFO));
        }
        if self.state == State::NotInitialized && flag(SET_INITIALIZE_ARG) {
            self.change_state(State::Hidden, &mut outcome.replies);
            let mut reply = self.reply_base(FINISHED_INITIALIZE);
            reply.push(0);
            outcome.replies.push(reply);
        }
        if !flag(SET_INITIALIZE_ARG) && (flag(SET_INPUT_TEXT) || flag(SET_CURSOR_POSITION)) {
            outcome.events.push(InlineKeyboardEvent::TextChanged {
                text: self.text.clone(),
                cursor: self.cursor,
            });
        }
        if self.state == State::Hidden && flag(APPEAR) {
            self.change_state(State::Appearing, &mut outcome.replies);
            self.change_state(State::Shown, &mut outcome.replies);
            let appear = &arg[layout.appear..];
            let (max_length, min_length) = text_lengths(u32_at(appear, 0x20), u32_at(appear, 0x24));
            outcome
                .events
                .push(InlineKeyboardEvent::Appear(InlineKeyboardAppear {
                    kind: u32_at(appear, 0),
                    ok_text: utf16_at(appear, 0x4, 9),
                    text: self.text.clone(),
                    cursor: self.cursor.max(0),
                    max_length,
                    min_length,
                    cancel_disabled: appear[0x1b] != 0,
                    return_enabled: appear[0x28] != 0,
                    backspace_enabled: arg[layout.backspace] != 0,
                }));
        } else if self.state == State::Shown && flag(DISAPPEAR) {
            self.hide(&mut outcome.replies);
            outcome.events.push(InlineKeyboardEvent::Disappear);
        }
    }

    /// Encodes what the user did in the reply type the title asked for (UTF-8, V2), followed by
    /// the state changes of a keyboard the decision hides.
    pub fn reply(&mut self, reply: InlineKeyboardReply) -> Vec<Vec<u8>> {
        let (reply, hides) = match reply {
            InlineKeyboardReply::ChangedString { text, cursor } => {
                self.text = text;
                self.cursor = cursor;
                let kind = match (self.utf8, self.changed_string_v2) {
                    (false, false) => CHANGED_STRING,
                    (true, false) => CHANGED_STRING_UTF8,
                    (false, true) => CHANGED_STRING_V2,
                    (true, true) => CHANGED_STRING_UTF8_V2,
                };
                let mut out = self.reply_with_text(kind);
                out.extend(self.text_length().to_le_bytes());
                out.extend((-1i32).to_le_bytes());
                out.extend((-1i32).to_le_bytes());
                out.extend(cursor.to_le_bytes());
                if self.changed_string_v2 {
                    out.push(0);
                }
                (out, false)
            }
            InlineKeyboardReply::MovedCursor { text, cursor } => {
                self.text = text;
                self.cursor = cursor;
                let kind = match (self.utf8, self.moved_cursor_v2) {
                    (false, false) => MOVED_CURSOR,
                    (true, false) => MOVED_CURSOR_UTF8,
                    (false, true) => MOVED_CURSOR_V2,
                    (true, true) => MOVED_CURSOR_UTF8_V2,
                };
                let mut out = self.reply_with_text(kind);
                out.extend(self.text_length().to_le_bytes());
                out.extend(cursor.to_le_bytes());
                if self.moved_cursor_v2 {
                    out.push(0);
                }
                (out, false)
            }
            InlineKeyboardReply::DecidedEnter { text } => {
                self.text = text;
                let kind = if self.utf8 {
                    DECIDED_ENTER_UTF8
                } else {
                    DECIDED_ENTER
                };
                let mut out = self.reply_with_text(kind);
                out.extend(self.text_length().to_le_bytes());
                (out, true)
            }
            InlineKeyboardReply::DecidedCancel => (self.reply_base(DECIDED_CANCEL), true),
        };
        let mut replies = vec![reply];
        if hides {
            self.hide(&mut replies);
        }
        replies
    }

    fn hide(&mut self, replies: &mut Vec<Vec<u8>>) {
        if self.state == State::Shown {
            self.change_state(State::Disappearing, replies);
            self.change_state(State::Hidden, replies);
        }
    }

    /// Moves to `state` and tells the title with a `Default` reply, which is how the guest
    /// learns the keyboard's state.
    fn change_state(&mut self, state: State, replies: &mut Vec<Vec<u8>>) {
        self.state = state;
        replies.push(self.reply_base(DEFAULT));
    }

    /// `SwkbdState` and `SwkbdReplyType`.
    fn reply_base(&self, kind: u32) -> Vec<u8> {
        [(self.state as u32).to_le_bytes(), kind.to_le_bytes()].concat()
    }

    fn reply_with_text(&self, kind: u32) -> Vec<u8> {
        let mut out = self.reply_base(kind);
        let start = out.len();
        let size = if self.utf8 {
            KEYBOARD_STRING_SIZE
        } else {
            UTF16_TEXT_SIZE
        };
        out.resize(start + size, 0);
        write_text(&mut out[start..], &self.text, self.utf8);
        out
    }

    /// The text's length in UTF-16 units, whatever encoding the reply carries it in.
    fn text_length(&self) -> u32 {
        self.text.encode_utf16().count() as u32
    }
}
