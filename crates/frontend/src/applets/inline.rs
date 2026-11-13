//! The inline software keyboard, after Lichii's inline QtSoftwareKeyboardDialog: a text line
//! over the bottom of the game, which keeps running. What the user types reaches the title as
//! they type it, and Enter or OK submits it.

use egui::text::{CCursor, CCursorRange};
use egui::{Align2, Event, Key, RichText};
use lichii_services::applets::{InlineKeyboardAppear, InlineKeyboardEvent, InlineKeyboardReply};

/// Sends one reply to the title.
pub(super) type Reply = Box<dyn Fn(InlineKeyboardReply) + Send>;

/// `SwkbdType` 1: digits only.
const NUMBER_PAD: u32 = 1;

/// What the UI thread is told about the inline keyboard.
pub(super) enum InlineMsg {
    Appear(InlineKeyboardAppear, Reply),
    Text { text: String, cursor: i32 },
    Hide,
}

impl InlineMsg {
    pub fn new(event: InlineKeyboardEvent, reply: Reply) -> Self {
        match event {
            InlineKeyboardEvent::Appear(appear) => Self::Appear(appear, reply),
            InlineKeyboardEvent::TextChanged { text, cursor } => Self::Text { text, cursor },
            InlineKeyboardEvent::Disappear | InlineKeyboardEvent::Finalize => Self::Hide,
        }
    }
}

fn utf16_len(text: &str) -> usize {
    text.encode_utf16().count()
}

/// The UTF-16 offset of char index `chars` in `text`.
fn utf16_at(text: &str, chars: usize) -> i32 {
    text.chars().take(chars).map(char::len_utf16).sum::<usize>() as i32
}

/// The char index of UTF-16 offset `units` in `text`, clamped to the text.
fn char_at(text: &str, units: i32) -> usize {
    let mut seen = 0;
    text.chars()
        .take_while(|c| {
            seen += c.len_utf16();
            seen as i32 <= units
        })
        .count()
}

pub(super) struct InlineKeyboard {
    appear: InlineKeyboardAppear,
    text: String,
    /// In UTF-16 units, as the title counts them.
    cursor: i32,
    reply: Reply,
    /// The title placed the cursor; the text field takes it on the next frame.
    place_cursor: bool,
}

impl InlineKeyboard {
    pub fn new(appear: InlineKeyboardAppear, reply: Reply) -> Self {
        Self {
            text: appear.text.clone(),
            cursor: appear.cursor,
            appear,
            reply,
            place_cursor: true,
        }
    }

    /// The title set the text or the cursor itself.
    pub fn set_text(&mut self, text: String, cursor: i32) {
        self.text = text;
        self.cursor = cursor;
        self.place_cursor = true;
    }

    fn long_enough(&self) -> bool {
        utf16_len(&self.text) >= self.appear.min_length as usize
    }

    /// Keeps what the title accepts from `raw`: digits on the number pad, newlines only with
    /// return, and at most `max_length` units. The char cursor `at` moves with the kept text.
    fn accept(&self, raw: &str, at: usize) -> (String, usize) {
        let max = self.appear.max_length.max(1) as usize;
        let (mut text, mut cursor, mut units) = (String::new(), 0, 0);
        for (index, c) in raw.chars().enumerate() {
            let allowed = (self.appear.kind != NUMBER_PAD || c.is_ascii_digit())
                && (self.appear.return_enabled || c != '\n');
            if !allowed || units + c.len_utf16() > max {
                continue;
            }
            text.push(c);
            units += c.len_utf16();
            if index < at {
                cursor += 1;
            }
        }
        (text, cursor)
    }

    /// Draws the keyboard. Returns whether the user submitted or cancelled, which hides it.
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        let id = egui::Id::unique("inline keyboard text");
        let set_cursor = |ctx: &egui::Context, at: usize| {
            let mut state = egui::text_edit::TextEditState::load(ctx, id).unwrap_or_default();
            state
                .cursor
                .set_char_range(Some(CCursorRange::one(CCursor::new(at))));
            state.store(ctx, id);
        };
        if std::mem::take(&mut self.place_cursor) {
            set_cursor(ctx, char_at(&self.text, self.cursor));
        }
        if !self.appear.backspace_enabled {
            ctx.input_mut(|i| {
                i.events.retain(|e| {
                    !matches!(
                        e,
                        Event::Key {
                            key: Key::Backspace,
                            ..
                        }
                    )
                });
            });
        }
        let (enter, escape) =
            ctx.input(|i| (i.key_pressed(Key::Enter), i.key_pressed(Key::Escape)));
        let mut decided = None;
        egui::Area::new(egui::Id::unique("inline keyboard"))
            .anchor(Align2::CENTER_BOTTOM, egui::vec2(0.0, -24.0))
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    ui.set_width(560.0);
                    let mut text = self.text.clone();
                    let edit = if self.appear.return_enabled {
                        egui::TextEdit::multiline(&mut text).desired_rows(2)
                    } else {
                        egui::TextEdit::singleline(&mut text)
                    };
                    let output = edit.id(id).desired_width(f32::INFINITY).show(ui);
                    let at = output
                        .cursor_range
                        .map_or(char_at(&self.text, self.cursor), |range| {
                            range.primary.index.0
                        });
                    if output.response.changed() {
                        let (accepted, at) = self.accept(&text, at);
                        set_cursor(ui.ctx(), at);
                        let cursor = utf16_at(&accepted, at);
                        if accepted != self.text || cursor != self.cursor {
                            self.text = accepted;
                            self.cursor = cursor;
                            (self.reply)(InlineKeyboardReply::ChangedString {
                                text: self.text.clone(),
                                cursor,
                            });
                        }
                    } else if utf16_at(&text, at) != self.cursor {
                        self.cursor = utf16_at(&text, at);
                        (self.reply)(InlineKeyboardReply::MovedCursor {
                            text: self.text.clone(),
                            cursor: self.cursor,
                        });
                    }
                    ui.horizontal(|ui| {
                        let ok = if self.appear.ok_text.is_empty() {
                            "OK"
                        } else {
                            self.appear.ok_text.as_str()
                        };
                        let submit = ui
                            .add_enabled(self.long_enough(), egui::Button::new(ok))
                            .clicked()
                            || (enter && !self.appear.return_enabled && self.long_enough());
                        if submit {
                            decided = Some(InlineKeyboardReply::DecidedEnter {
                                text: self.text.clone(),
                            });
                        }
                        if !self.appear.cancel_disabled && (ui.button("Cancel").clicked() || escape)
                        {
                            decided = Some(InlineKeyboardReply::DecidedCancel);
                        }
                        let max = self.appear.max_length.max(1);
                        ui.label(RichText::new(format!("{}/{max}", utf16_len(&self.text))).weak());
                    });
                    if decided.is_none() && !output.response.has_focus() {
                        output.response.request_focus();
                    }
                });
            });
        let Some(reply) = decided else {
            return false;
        };
        (self.reply)(reply);
        true
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::Receiver;

    use super::super::testing::Harness;
    use super::*;

    fn appear(text: &str) -> InlineKeyboardAppear {
        InlineKeyboardAppear {
            text: text.into(),
            cursor: utf16_len(text) as i32,
            max_length: 20,
            ..InlineKeyboardAppear::default()
        }
    }

    fn open(ui: &mut Harness, appear: InlineKeyboardAppear) -> Receiver<InlineKeyboardReply> {
        let (tx, rx) = crossbeam_channel::unbounded();
        ui.dialogs.inline(
            InlineKeyboardEvent::Appear(appear),
            Box::new(move |reply| {
                let _ = tx.send(reply);
            }),
        );
        ui.frame(Vec::new());
        rx
    }

    fn changed(text: &str, cursor: i32) -> InlineKeyboardReply {
        InlineKeyboardReply::ChangedString {
            text: text.into(),
            cursor,
        }
    }

    #[test]
    fn typing_reaches_the_title_and_enter_submits() {
        let mut ui = Harness::new();
        let replies = open(&mut ui, appear("ab"));
        ui.typed("c");
        assert_eq!(replies.try_recv(), Ok(changed("abc", 3)));
        ui.key(Key::Enter);
        assert_eq!(
            replies.try_recv(),
            Ok(InlineKeyboardReply::DecidedEnter { text: "abc".into() })
        );
        ui.typed("d");
        assert!(replies.try_recv().is_err(), "the keyboard hid after Enter");
    }

    #[test]
    fn escape_cancels_unless_the_title_disabled_it() {
        let mut ui = Harness::new();
        let replies = open(
            &mut ui,
            InlineKeyboardAppear {
                cancel_disabled: true,
                ..appear("")
            },
        );
        ui.key(Key::Escape);
        assert!(replies.try_recv().is_err());
        ui.typed("x");
        assert_eq!(
            replies.try_recv(),
            Ok(changed("x", 1)),
            "the text keeps focus"
        );

        let mut ui = Harness::new();
        let replies = open(&mut ui, appear("hi"));
        ui.key(Key::Escape);
        assert_eq!(replies.try_recv(), Ok(InlineKeyboardReply::DecidedCancel));
    }

    #[test]
    fn the_number_pad_takes_digits_up_to_the_maximum() {
        let mut ui = Harness::new();
        let replies = open(
            &mut ui,
            InlineKeyboardAppear {
                kind: NUMBER_PAD,
                max_length: 3,
                ..appear("")
            },
        );
        ui.typed("1a2");
        assert_eq!(replies.try_recv(), Ok(changed("12", 2)));
        ui.typed("345");
        assert_eq!(replies.try_recv(), Ok(changed("123", 3)));
    }

    #[test]
    fn enter_waits_for_the_minimum_length() {
        let mut ui = Harness::new();
        let replies = open(
            &mut ui,
            InlineKeyboardAppear {
                min_length: 3,
                ..appear("ab")
            },
        );
        ui.key(Key::Enter);
        assert!(replies.try_recv().is_err());
    }

    #[test]
    fn the_title_moves_the_text_and_hides_the_keyboard() {
        let mut ui = Harness::new();
        let replies = open(&mut ui, appear("ab"));
        ui.dialogs.inline(
            InlineKeyboardEvent::TextChanged {
                text: "xyz".into(),
                cursor: 1,
            },
            Box::new(|_| {}),
        );
        ui.frame(Vec::new());
        ui.typed("Q");
        assert_eq!(replies.try_recv(), Ok(changed("xQyz", 2)));
        ui.dialogs
            .inline(InlineKeyboardEvent::Disappear, Box::new(|_| {}));
        ui.typed("z");
        assert!(
            replies.try_recv().is_err(),
            "a hidden keyboard sends nothing"
        );
    }

    #[test]
    fn the_keyboard_takes_focus_back_after_a_dialog() {
        let mut ui = Harness::new();
        let replies = open(&mut ui, appear("ab"));
        let answer = ui.ask(|frontend| {
            frontend.select_profile(&lichii_services::applets::ProfileSelectRequest::default())
        });
        for _ in 0..3 {
            ui.frame(Vec::new());
        }
        ui.key(Key::Escape);
        assert_eq!(
            answer.join().unwrap(),
            lichii_services::applets::ProfileSelectAnswer::Cancelled
        );
        assert!(replies.try_recv().is_err(), "Escape went to the dialog");
        ui.frame(Vec::new());
        ui.typed("c");
        assert_eq!(replies.try_recv(), Ok(changed("abc", 3)));
    }

    #[test]
    fn backspace_works_only_when_the_title_allows_it() {
        let mut ui = Harness::new();
        let replies = open(&mut ui, appear("ab"));
        ui.key(Key::Backspace);
        assert!(replies.try_recv().is_err());
        let mut ui = Harness::new();
        let replies = open(
            &mut ui,
            InlineKeyboardAppear {
                backspace_enabled: true,
                ..appear("ab")
            },
        );
        ui.key(Key::Backspace);
        assert_eq!(replies.try_recv(), Ok(changed("a", 1)));
    }

    #[test]
    fn a_rejected_character_leaves_the_cursor_in_place() {
        let mut ui = Harness::new();
        let replies = open(
            &mut ui,
            InlineKeyboardAppear {
                kind: NUMBER_PAD,
                cursor: 1,
                ..appear("12")
            },
        );
        ui.typed("a");
        assert!(
            replies.try_recv().is_err(),
            "nothing the title accepts changed"
        );
        ui.typed("3");
        assert_eq!(replies.try_recv(), Ok(changed("132", 2)));
    }

    #[test]
    fn cursors_count_utf16_units() {
        let text = "a\u{e9}\u{1f600}b";
        assert_eq!(utf16_at(text, 3), 4);
        assert_eq!(char_at(text, 4), 3);
        assert_eq!(char_at(text, 3), 2, "inside the surrogate pair");
        assert_eq!(char_at(text, 99), 4);
    }
}
