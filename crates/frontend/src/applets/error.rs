//! The error applet, after Lichii's QtErrorDisplay: the error code and the title's message in
//! a dialog the user closes with OK. The title waits until then.

use crossbeam_channel::Sender;
use egui::Key;
use lichii_services::applets::{ErrorCode, ErrorRequest};

/// `ErrorAppletMode` values that carry the title's own text.
const SYSTEM_ERROR: u8 = 1;
const APPLICATION_ERROR: u8 = 2;

/// The heading Lichii shows: `Error Code: 2XXX-YYYY (0xRAW)`, with the result value when the
/// code is one.
fn heading(code: ErrorCode) -> String {
    let module = code.category.wrapping_sub(2000);
    if module < 0x200 && code.number < 0x2000 {
        let raw = module | (code.number << 9);
        format!("Error Code: {code} (0x{raw:08x})")
    } else {
        format!("Error Code: {code}")
    }
}

/// The message under the heading, worded like Lichii's.
fn message(request: &ErrorRequest) -> String {
    match request.mode {
        SYSTEM_ERROR | APPLICATION_ERROR => format!(
            "An error has occurred.\n\n{}\n\n{}",
            request.main_text, request.detail_text
        ),
        _ => "An error has occurred.\nPlease try again or contact the developer of the software."
            .into(),
    }
}

pub(super) struct ErrorDialog {
    heading: String,
    message: String,
    pub reply: Sender<()>,
}

impl ErrorDialog {
    pub fn new(request: &ErrorRequest, reply: Sender<()>) -> Self {
        Self {
            heading: heading(request.code),
            message: message(request),
            reply,
        }
    }

    /// Whether the user closed the dialog.
    pub fn show(&mut self, ctx: &egui::Context) -> bool {
        let keys = ctx.input(|i| i.key_pressed(Key::Enter) || i.key_pressed(Key::Escape));
        let mut ok = false;
        egui::Modal::new(egui::Id::unique("error applet")).show(ctx, |ui| {
            ui.set_max_width(480.0);
            ui.heading(&self.heading);
            ui.add_space(6.0);
            ui.label(&self.message);
            ui.add_space(8.0);
            ok = ui.button("OK").clicked();
        });
        ok || keys
    }
}

#[cfg(test)]
mod tests {
    use lichii_services::applets::AppletFrontend;

    use super::super::testing::Harness;
    use super::*;

    fn show(ui: &Harness, request: ErrorRequest) -> std::thread::JoinHandle<()> {
        ui.ask(move |frontend: &dyn AppletFrontend| frontend.show_error(&request))
    }

    #[test]
    fn the_title_waits_until_ok() {
        let mut ui = Harness::new();
        let shown = show(
            &ui,
            ErrorRequest {
                code: ErrorCode::from_result(0x2a8),
                ..ErrorRequest::default()
            },
        );
        ui.frame(Vec::new());
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!shown.is_finished(), "the dialog is still open");
        assert!(ui.dialogs.is_open());
        ui.key(Key::Enter);
        shown.join().unwrap();
        assert!(!ui.dialogs.is_open());
    }

    #[test]
    fn the_heading_shows_the_code_and_its_result() {
        assert_eq!(
            heading(ErrorCode::from_result(0x2a8)),
            "Error Code: 2168-0001 (0x000002a8)"
        );
        let custom = ErrorCode {
            category: 2618,
            number: 6,
        };
        assert_eq!(heading(custom), "Error Code: 2618-0006");
    }

    #[test]
    fn system_and_application_errors_show_the_titles_text() {
        let request = ErrorRequest {
            mode: APPLICATION_ERROR,
            main_text: "Could not connect.".into(),
            detail_text: "Check your network.".into(),
            ..ErrorRequest::default()
        };
        assert_eq!(
            message(&request),
            "An error has occurred.\n\nCould not connect.\n\nCheck your network."
        );
        let plain = ErrorRequest::default();
        assert!(message(&plain).ends_with("contact the developer of the software."));
    }
}
