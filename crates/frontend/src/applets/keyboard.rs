//! The full-screen software keyboard.

use crossbeam_channel::Sender;
use egui::{Color32, Key, RichText};
use lichii_services::applets::{SoftwareKeyboardAnswer, SoftwareKeyboardRequest, TextCheckPrompt};

/// `SwkbdType::NumberPad`: only digits can be typed.
const NUMBER_PAD: u32 = 1;

/// The full-screen software keyboard, after Lichii's QtSoftwareKeyboardDialog: the header, sub
/// and guide texts, the length limits, number pad and password modes, and the text check's
/// message before the text can be edited again.
pub(super) struct KeyboardDialog {
    request: SoftwareKeyboardRequest,
    pub reply: Sender<SoftwareKeyboardAnswer>,
    text: String,
    /// The text check's message while it is shown, before editing.
    prompt: Option<TextCheckPrompt>,
    focus: bool,
}

impl KeyboardDialog {
    pub fn new(request: SoftwareKeyboardRequest, reply: Sender<SoftwareKeyboardAnswer>) -> Self {
        let mut text = request.initial_text.clone();
        limit(&mut text, &request);
        Self {
            prompt: request.text_check.clone(),
            request,
            reply,
            text,
            focus: true,
        }
    }

    fn valid(&self) -> bool {
        utf16_len(&self.text) >= self.request.min_length
    }

    pub fn show(&mut self, ctx: &egui::Context) -> Option<SoftwareKeyboardAnswer> {
        let (enter, escape) =
            ctx.input(|i| (i.key_pressed(Key::Enter), i.key_pressed(Key::Escape)));
        match self.prompt.clone() {
            Some(prompt) => self.show_prompt(ctx, &prompt, enter, escape),
            None => self.show_editor(ctx, enter, escape),
        }
    }

    fn show_prompt(
        &mut self,
        ctx: &egui::Context,
        prompt: &TextCheckPrompt,
        enter: bool,
        escape: bool,
    ) -> Option<SoftwareKeyboardAnswer> {
        let (message, confirm) = match prompt {
            TextCheckPrompt::Rejected(message) => (message, false),
            TextCheckPrompt::Confirm(message) => (message, true),
        };
        let mut clicked = None;
        egui::Modal::new(egui::Id::unique("software keyboard check")).show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.label(message.as_str());
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if confirm && ui.button("Cancel").clicked() {
                    clicked = Some(false);
                }
                if ui.button("OK").clicked() {
                    clicked = Some(true);
                }
            });
        });
        let ok = clicked
            .or(enter.then_some(true))
            .or((escape && confirm).then_some(false))?;
        if ok && confirm {
            return Some(SoftwareKeyboardAnswer::Submitted(self.text.clone()));
        }
        self.prompt = None;
        self.focus = true;
        None
    }

    fn show_editor(
        &mut self,
        ctx: &egui::Context,
        enter: bool,
        escape: bool,
    ) -> Option<SoftwareKeyboardAnswer> {
        let request = &self.request;
        let mut answer = None;
        egui::Modal::new(egui::Id::unique("software keyboard")).show(ctx, |ui| {
            ui.set_width(420.0);
            if !request.header_text.is_empty() {
                ui.heading(request.header_text.as_str());
            }
            if !request.sub_text.is_empty() {
                ui.label(request.sub_text.as_str());
            }
            ui.add_space(4.0);
            let edit = ui.add(
                egui::TextEdit::singleline(&mut self.text)
                    .hint_text(request.guide_text.as_str())
                    .password(request.password)
                    .desired_width(f32::INFINITY),
            );
            if std::mem::take(&mut self.focus) {
                edit.request_focus();
            }
            if edit.changed() {
                limit(&mut self.text, request);
            }
            let valid = utf16_len(&self.text) >= request.min_length;
            ui.horizontal(|ui| {
                let count = format!("{}/{}", utf16_len(&self.text), request.max_length);
                ui.label(RichText::new(count).weak());
                if request.min_length > 0 && !valid {
                    ui.label(
                        RichText::new(format!("at least {}", request.min_length))
                            .color(Color32::from_rgb(220, 120, 80)),
                    );
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if !request.cancel_disabled && ui.button("Cancel").clicked() {
                    answer = Some(SoftwareKeyboardAnswer::Cancelled);
                }
                let ok = if request.ok_text.is_empty() {
                    "OK"
                } else {
                    request.ok_text.as_str()
                };
                if ui.add_enabled(valid, egui::Button::new(ok)).clicked() {
                    answer = Some(SoftwareKeyboardAnswer::Submitted(self.text.clone()));
                }
            });
        });
        if answer.is_none() && enter && self.valid() {
            answer = Some(SoftwareKeyboardAnswer::Submitted(self.text.clone()));
        }
        if answer.is_none() && escape && !self.request.cancel_disabled {
            answer = Some(SoftwareKeyboardAnswer::Cancelled);
        }
        if answer.is_none() && (enter || escape) {
            self.focus = true;
        }
        answer
    }
}

fn utf16_len(text: &str) -> u32 {
    text.encode_utf16().count() as u32
}

/// Keeps `text` to what the keyboard accepts: digits only on the number pad, and at most
/// `max_length` UTF-16 units.
fn limit(text: &mut String, request: &SoftwareKeyboardRequest) {
    if request.kind == NUMBER_PAD {
        text.retain(|c| c.is_ascii_digit());
    }
    let mut units = 0;
    let cut = text.char_indices().find_map(|(at, c)| {
        units += c.len_utf16() as u32;
        (units > request.max_length).then_some(at)
    });
    if let Some(at) = cut {
        text.truncate(at);
    }
}

#[cfg(test)]
mod tests {
    use std::thread::JoinHandle;

    use super::super::testing::Harness;
    use super::*;

    fn keyboard(
        ui: &Harness,
        request: SoftwareKeyboardRequest,
    ) -> JoinHandle<SoftwareKeyboardAnswer> {
        ui.ask(move |frontend| frontend.software_keyboard(&request))
    }

    fn request(initial: &str, min: u32, max: u32) -> SoftwareKeyboardRequest {
        SoftwareKeyboardRequest {
            header_text: "Name".into(),
            guide_text: "Enter a name".into(),
            initial_text: initial.into(),
            min_length: min,
            max_length: max,
            ..SoftwareKeyboardRequest::default()
        }
    }

    #[test]
    fn typing_is_cut_at_the_maximum_and_enter_submits() {
        let mut ui = Harness::new();
        let answer = keyboard(&ui, request("ab", 0, 5));
        assert!(ui.dialogs.is_open());
        ui.frame(Vec::new());
        ui.typed("cdefg");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("abcde".into())
        );
        assert!(!ui.dialogs.is_open());
    }

    #[test]
    fn enter_waits_for_the_minimum_length() {
        let mut ui = Harness::new();
        let answer = keyboard(&ui, request("", 3, 10));
        ui.frame(Vec::new());
        ui.typed("ab");
        ui.key(Key::Enter);
        assert!(ui.dialogs.is_open(), "two characters are too few");
        ui.frame(Vec::new());
        ui.typed("c");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("abc".into())
        );
    }

    #[test]
    fn the_number_pad_takes_digits_only() {
        let mut ui = Harness::new();
        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                kind: NUMBER_PAD,
                ..request("4x", 0, 8)
            },
        );
        ui.frame(Vec::new());
        ui.typed("1a2");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("412".into())
        );
    }

    #[test]
    fn escape_cancels_unless_the_title_disabled_cancel() {
        let mut ui = Harness::new();
        let answer = keyboard(&ui, request("ab", 0, 5));
        ui.frame(Vec::new());
        ui.key(Key::Escape);
        assert_eq!(answer.join().unwrap(), SoftwareKeyboardAnswer::Cancelled);

        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                cancel_disabled: true,
                ..request("ab", 0, 5)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Escape);
        assert!(ui.dialogs.is_open(), "cancel is disabled");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("ab".into())
        );
    }

    #[test]
    fn a_rejected_text_shows_the_message_then_edits_it_again() {
        let mut ui = Harness::new();
        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                text_check: Some(TextCheckPrompt::Rejected("That name is taken.".into())),
                ..request("bob", 1, 10)
            },
        );
        ui.frame(Vec::new());
        ui.typed("x");
        ui.key(Key::Enter);
        assert!(
            ui.dialogs.is_open(),
            "OK on the message returns to the editor"
        );
        ui.frame(Vec::new());
        ui.typed("by");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("bobby".into())
        );
    }

    #[test]
    fn a_confirm_prompt_submits_the_text_unchanged() {
        let mut ui = Harness::new();
        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                text_check: Some(TextCheckPrompt::Confirm("Use this name?".into())),
                ..request("bob", 1, 10)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("bob".into())
        );

        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                text_check: Some(TextCheckPrompt::Confirm("Use this name?".into())),
                ..request("bob", 1, 10)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Escape);
        ui.frame(Vec::new());
        ui.typed("o");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("bobo".into())
        );
    }

    #[test]
    fn enter_on_the_focused_cancel_button_returns_to_the_editor() {
        let mut ui = Harness::new();
        let answer = keyboard(
            &ui,
            SoftwareKeyboardRequest {
                text_check: Some(TextCheckPrompt::Confirm("Use this name?".into())),
                ..request("bob", 1, 10)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Tab);
        ui.key(Key::Enter);
        assert!(ui.dialogs.is_open(), "Cancel wins over the Enter shortcut");
        ui.frame(Vec::new());
        ui.typed("o");
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            SoftwareKeyboardAnswer::Submitted("bobo".into())
        );
    }

    #[test]
    fn requests_are_cancelled_once_the_dialogs_are_gone() {
        let ui = Harness::new();
        let answer = keyboard(&ui, request("ab", 0, 5));
        let frontend = ui.dialogs.frontend();
        drop(ui);
        assert_eq!(answer.join().unwrap(), SoftwareKeyboardAnswer::Cancelled);
        assert_eq!(
            frontend.software_keyboard(&request("", 0, 5)),
            SoftwareKeyboardAnswer::Cancelled
        );
    }

    #[test]
    fn the_maximum_counts_utf16_units() {
        let mut text = "a😀b".to_owned();
        limit(&mut text, &request("", 0, 3));
        assert_eq!(text, "a😀");
        let mut text = "a😀".to_owned();
        limit(&mut text, &request("", 0, 2));
        assert_eq!(text, "a");
    }
}
