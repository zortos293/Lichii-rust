//! What the user answers in a library applet: the error dialog, the profile selector, the
//! controller applet and the software keyboard.
//!
//! am parses each applet's arguments into a request, asks the [`AppletFrontend`] installed with
//! [`crate::ServiceContext::set_applet_frontend`] and writes the answer back in the applet's
//! layout. [`HeadlessApplets`] answers without UI and is installed by default.

use std::sync::Weak;

use super::library_applet::LibraryApplet;

/// Answers library applets on behalf of the user. am calls it on its service thread, inside the
/// guest's `Start` (or `PushInteractiveInData`), and the guest waits until it returns.
pub trait AppletFrontend: Send + Sync {
    /// Shows an error the title reported. The applet exits when this returns.
    fn show_error(&self, request: &ErrorRequest);

    /// Lets the user pick a profile.
    fn select_profile(&self, request: &ProfileSelectRequest) -> ProfileSelectAnswer;

    /// Lets the user set up controllers.
    fn configure_controllers(&self, request: &ControllerRequest) -> ControllerAnswer;

    /// Lets the user type text in the full-screen software keyboard. When the title checks the
    /// text and sends it back, this is called again with [`SoftwareKeyboardRequest::text_check`]
    /// set and the text as [`SoftwareKeyboardRequest::initial_text`].
    fn software_keyboard(&self, request: &SoftwareKeyboardRequest) -> SoftwareKeyboardAnswer;

    /// Drives the inline software keyboard, which the title draws over its own frame while it
    /// keeps running. am calls this for each change the title asks for and expects it to return
    /// at once; the frontend sends what the user types later, from any thread, through
    /// `replies` (clone it to keep it). The default cancels the keyboard as soon as it appears.
    fn inline_keyboard(&self, event: InlineKeyboardEvent, replies: &InlineKeyboardReplies) {
        if let InlineKeyboardEvent::Appear(_) = event {
            replies.send(InlineKeyboardReply::DecidedCancel);
        }
    }
}

/// `nn::err::ErrorCode` the error applet shows, as `2XXX-YYYY`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ErrorCode {
    /// `2000 + module`.
    pub category: u32,
    pub number: u32,
}

impl ErrorCode {
    /// The error code of a result value.
    pub fn from_result(raw: u32) -> Self {
        Self {
            category: 2000 + (raw & 0x1ff),
            number: (raw >> 9) & 0x1fff,
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}-{:04}", self.category, self.number)
    }
}

/// An error applet launch.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ErrorRequest {
    /// `ErrorAppletMode`: 0 error, 1 system error, 2 application error, 4 parental control
    /// error, 5 error record.
    pub mode: u8,
    pub code: ErrorCode,
    /// The title's own text (system and application errors only).
    pub main_text: String,
    pub detail_text: String,
}

/// A profile selector launch (`nn::account::UiSettings`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ProfileSelectRequest {
    /// `nn::account::UiMode`; 0 selects a user.
    pub mode: u32,
    /// `nn::account::UserSelectionPurpose`.
    pub purpose: u32,
    /// Users the title does not allow; zero entries are unused.
    pub invalid_users: [[u8; 16]; 8],
    /// The user acc opened at boot, if any.
    pub current_user: Option<[u8; 16]>,
}

/// What the user did in the profile selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileSelectAnswer {
    Selected([u8; 16]),
    Cancelled,
}

/// The controller setup a title sees: how many npads are connected and which comes first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControllerSetup {
    pub player_count: i8,
    /// `NpadIdType`: 0-7 for players 1-8, 0x20 for handheld.
    pub npad_id: u32,
}

/// A controller applet launch (`ControllerSupportArgPrivate` and the argument header).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControllerRequest {
    /// `ControllerSupportMode`: 0 controller support, 1 strap guide, 2 firmware update,
    /// 3 key remapping.
    pub mode: u8,
    pub min_players: i8,
    pub max_players: i8,
    pub single_mode: bool,
    /// `NpadStyleSet` the title accepts.
    pub style_set: u32,
    /// Controllers already connected stay connected.
    pub keep_connected: bool,
    /// The npads hid shows the title as connected now.
    pub current: ControllerSetup,
}

/// What the user did in the controller applet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerAnswer {
    Confirmed(ControllerSetup),
    Cancelled,
}

/// A full-screen software keyboard launch (`SwkbdConfigCommon`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SoftwareKeyboardRequest {
    /// `SwkbdType`: 0 normal, 1 number pad, 2 QWERTY, ...
    pub kind: u32,
    pub ok_text: String,
    pub header_text: String,
    pub sub_text: String,
    pub guide_text: String,
    pub initial_text: String,
    /// In UTF-16 units, 1 to 500.
    pub max_length: u32,
    pub min_length: u32,
    pub password: bool,
    pub cancel_disabled: bool,
    /// What the title said about the previous text, when this is a retry.
    pub text_check: Option<TextCheckPrompt>,
}

/// What a title's text check said about the submitted text, with the title's message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TextCheckPrompt {
    /// The text was rejected; whatever the user submits next is checked again.
    Rejected(String),
    /// The user must confirm the text: submitting it unchanged confirms it, and edited text is
    /// checked again.
    Confirm(String),
}

/// What the user did in the software keyboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SoftwareKeyboardAnswer {
    Submitted(String),
    Cancelled,
}

/// What the title asked of the inline software keyboard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlineKeyboardEvent {
    /// Show the keyboard. It stays up until the user decides or the title makes it disappear.
    Appear(InlineKeyboardAppear),
    /// The title set the text or the cursor itself.
    TextChanged { text: String, cursor: i32 },
    /// Hide the keyboard; the title can make it appear again.
    Disappear,
    /// The title closed the keyboard; replies sent from now on are dropped.
    Finalize,
}

/// How the inline keyboard appears (`SwkbdAppearArg` and the calc argument around it).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InlineKeyboardAppear {
    /// `SwkbdType`: 0 normal, 1 number pad, 2 QWERTY, ...
    pub kind: u32,
    pub ok_text: String,
    /// The text the title set, which the user edits.
    pub text: String,
    /// In UTF-16 units.
    pub cursor: i32,
    /// In UTF-16 units, 1 to 500.
    pub max_length: u32,
    pub min_length: u32,
    pub cancel_disabled: bool,
    pub return_enabled: bool,
    pub backspace_enabled: bool,
}

/// What the user did on the inline keyboard. Texts are the whole current text and cursors
/// count UTF-16 units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InlineKeyboardReply {
    ChangedString {
        text: String,
        cursor: i32,
    },
    MovedCursor {
        text: String,
        cursor: i32,
    },
    /// The user submitted the text; the keyboard hides.
    DecidedEnter {
        text: String,
    },
    /// The user cancelled; the keyboard hides.
    DecidedCancel,
}

/// Sends the user's input on one inline keyboard back to the title.
#[derive(Clone)]
pub struct InlineKeyboardReplies(pub(super) Weak<LibraryApplet>);

impl InlineKeyboardReplies {
    /// Queues `reply` for the title, unless the title already closed the keyboard.
    pub fn send(&self, reply: InlineKeyboardReply) {
        if let Some(applet) = self.0.upgrade() {
            applet.inline_reply(reply);
        }
    }
}

/// Answers every applet without UI: errors are acknowledged, the profile selector picks the
/// current user, the controller applet keeps the current setup and the keyboard is cancelled.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeadlessApplets;

impl AppletFrontend for HeadlessApplets {
    fn show_error(&self, request: &ErrorRequest) {
        tracing::warn!(
            code = %request.code,
            mode = request.mode,
            main = request.main_text,
            detail = request.detail_text,
            "the title showed an error"
        );
    }

    fn select_profile(&self, request: &ProfileSelectRequest) -> ProfileSelectAnswer {
        request.current_user.map_or(
            ProfileSelectAnswer::Cancelled,
            ProfileSelectAnswer::Selected,
        )
    }

    fn configure_controllers(&self, request: &ControllerRequest) -> ControllerAnswer {
        ControllerAnswer::Confirmed(request.current)
    }

    fn software_keyboard(&self, _request: &SoftwareKeyboardRequest) -> SoftwareKeyboardAnswer {
        SoftwareKeyboardAnswer::Cancelled
    }
}
