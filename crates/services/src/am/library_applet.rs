//! `ILibraryAppletAccessor`: a library applet the application launched, run by am itself.
//!
//! `Start` runs the applet at once: am parses the `CommonArguments` and the applet's own
//! arguments from the pushed in-data, asks the [`AppletFrontend`], pushes the answer and exits
//! the applet. The software keyboard stays running while the title checks the submitted text,
//! and the inline keyboard runs its own state machine until the title finalizes it.

use std::collections::VecDeque;
use std::sync::Arc;

use inline_keyboard::InlineKeyboard;

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::applets::{
    AppletFrontend, ControllerAnswer, ControllerRequest, ControllerSetup, ErrorCode, ErrorRequest,
    InlineKeyboardEvent, InlineKeyboardReplies, InlineKeyboardReply, ProfileSelectAnswer,
    ProfileSelectRequest, SoftwareKeyboardAnswer, SoftwareKeyboardRequest, TextCheckPrompt,
};
use super::functions::push_storage;
use super::storage::{Storage, StorageData};
use super::{NO_DATA_IN_CHANNEL, UNKNOWN, push_bool, push_event};
use crate::{Service, ServiceContext};

mod inline_keyboard;

/// `AppletId`s am answers through the frontend.
pub(super) const CONTROLLER: u32 = 0x0c;
pub(super) const ERROR: u32 = 0x0e;
pub(super) const PROFILE_SELECT: u32 = 0x10;
pub(super) const SOFTWARE_KEYBOARD: u32 = 0x11;

/// `LibraryAppletMode`s of the inline software keyboard.
const PARTIAL_FOREGROUND: u32 = 1;
const PARTIAL_FOREGROUND_INDIRECT_DISPLAY: u32 = 3;

const COMMON_ARGUMENTS_SIZE: usize = 0x20;

/// The zeroed out-data the error applet pushes when it closes.
const ERROR_OUT_DATA_SIZE: usize = 0x1000;

/// `nn::account::ResultCancelledByUser`, as the profile selector's `UiReturnArg` carries it.
const ACCOUNT_CANCELLED_BY_USER: ResultCode = ResultCode::new(124, 1);

/// `ControllerSupportResult`.
const CONTROLLER_SUCCESS: u32 = 0;
const CONTROLLER_CANCEL: u32 = 2;

/// `SwkbdResult`.
const KEYBOARD_OK: u32 = 0;
const KEYBOARD_CANCEL: u32 = 1;
/// `SwkbdTextCheckResult`.
const TEXT_CHECK_SUCCESS: u32 = 0;
const TEXT_CHECK_FAILURE: u32 = 1;
const TEXT_CHECK_CONFIRM: u32 = 2;
const KEYBOARD_STRING_SIZE: usize = 0x7d4;
const KEYBOARD_MAX_LENGTH: u32 = 500;
/// `SwkbdAppletVersion`s whose config layout differs from `SwkbdConfigCommon`.
const KEYBOARD_DISABLE_CANCEL_VERSION: u32 = 0x6000b;
const KEYBOARD_UTF8_VERSION: u32 = 0x14_0010;
const KEYBOARD_NATIVE_VERSION: u32 = 0x16_0012;
const KEYBOARD_NATIVE_CONFIG_SIZE: usize = 0x530;
const KEYBOARD_CONFIG_SIZE: usize = 0x3d4;

/// Maps an `AppletId` with unknown upper bits to its low byte when only that names an applet,
/// as titles that pass garbage in the upper bits expect.
pub(super) fn normalize_applet_id(id: u32) -> u32 {
    let known = |id| matches!(id, 0x02..=0x04 | 0x0a..=0x1a);
    if !known(id) && known(id & 0xff) {
        tracing::warn!(
            id = format_args!("{id:#x}"),
            "normalizing the applet id to its low byte"
        );
        id & 0xff
    } else {
        id
    }
}

/// A queue of storages between the application and the applet, with the event that says it is
/// not empty.
struct Channel {
    data: VecDeque<Arc<StorageData>>,
    event: Arc<Event>,
}

impl Default for Channel {
    fn default() -> Self {
        Self {
            data: VecDeque::new(),
            event: Arc::new(Event::new()),
        }
    }
}

impl Channel {
    fn push(&mut self, data: Arc<StorageData>) {
        self.data.push_back(data);
        self.event.signal();
    }

    fn pop(&mut self) -> Option<Arc<StorageData>> {
        let data = self.data.pop_front();
        if self.data.is_empty() {
            self.event.clear();
        }
        data
    }
}

#[derive(Default)]
struct RunState {
    in_data: Channel,
    interactive_in: Channel,
    out_data: Channel,
    interactive_out: Channel,
    completed: bool,
    text_check: Option<TextCheck>,
    inline: Option<InlineKeyboard>,
}

/// A library applet am runs for the application.
pub(super) struct LibraryApplet {
    id: u32,
    mode: u32,
    state: Mutex<RunState>,
    state_changed_event: Arc<Event>,
}

impl LibraryApplet {
    pub fn new(id: u32, mode: u32) -> Self {
        Self {
            id,
            mode,
            state: Mutex::default(),
            state_changed_event: Arc::new(Event::new()),
        }
    }

    fn start(self: &Arc<Self>, context: &ServiceContext) {
        let inputs: Vec<Vec<u8>> = {
            let mut state = self.state.lock();
            std::iter::from_fn(|| state.in_data.pop())
                .map(|data| data.bytes().clone())
                .collect()
        };
        let Some(common) = inputs
            .first()
            .filter(|common| common.len() >= COMMON_ARGUMENTS_SIZE)
        else {
            tracing::warn!(
                applet = self.id,
                "library applet started without CommonArguments"
            );
            return self.apply(Step::default());
        };
        let version = u32_at(common, 8);
        let argument = |index: usize| inputs.get(index).map_or(&[][..], Vec::as_slice);
        let frontend = context.applet_frontend();
        let step = match self.id {
            ERROR => {
                frontend.show_error(&error_request(argument(1)));
                Step::exit(vec![0; ERROR_OUT_DATA_SIZE])
            }
            PROFILE_SELECT => {
                let request = profile_request(version, argument(1), context.user.get().copied());
                Step::exit(profile_answer(frontend.select_profile(&request)))
            }
            CONTROLLER => {
                let current = context
                    .hid
                    .get()
                    .map_or_else(ControllerSetup::default, |hid| hid.controller_setup());
                let request = controller_request(argument(1), argument(2), current);
                Step::exit(controller_answer(
                    frontend.configure_controllers(&request),
                    request.current,
                ))
            }
            SOFTWARE_KEYBOARD
                if matches!(
                    self.mode,
                    PARTIAL_FOREGROUND | PARTIAL_FOREGROUND_INDIRECT_DISPLAY
                ) =>
            {
                self.state.lock().inline = Some(InlineKeyboard::default());
                return;
            }
            SOFTWARE_KEYBOARD => match keyboard_config(version, argument(1), argument(2)) {
                Some(config) => keyboard_step(frontend.as_ref(), config),
                None => {
                    tracing::warn!(version, "unknown software keyboard config");
                    Step::default()
                }
            },
            id => {
                if context.unknown.first("LibraryApplet", id) {
                    tracing::warn!(
                        applet = format_args!("{id:#x}"),
                        mode = self.mode,
                        "library applet without a frontend exits with no output"
                    );
                }
                Step::default()
            }
        };
        self.apply(step);
    }

    fn push_interactive(self: &Arc<Self>, context: &ServiceContext, data: Arc<StorageData>) {
        let mut state = self.state.lock();
        if let Some(keyboard) = state.inline.as_mut() {
            let outcome = keyboard.request(&data.bytes());
            for reply in outcome.replies {
                state
                    .interactive_out
                    .push(Arc::new(StorageData::new(reply, None)));
            }
            if outcome.finalized {
                self.exit(&mut state);
            }
            drop(state);
            return self.notify_inline(context, outcome.events);
        }
        let Some(check) = state.text_check.take() else {
            return state.interactive_in.push(data);
        };
        drop(state);
        let data = data.bytes().clone();
        let step = text_checked(context.applet_frontend().as_ref(), check, &data);
        self.apply(step);
    }

    fn apply(&self, step: Step) {
        let mut state = self.state.lock();
        if let Some(data) = step.out {
            state.out_data.push(Arc::new(StorageData::new(data, None)));
        }
        if let Some(data) = step.interactive_out {
            state
                .interactive_out
                .push(Arc::new(StorageData::new(data, None)));
        }
        state.text_check = step.text_check;
        if state.text_check.is_none() {
            self.exit(&mut state);
        }
    }

    /// Exits on the title's request, closing the inline keyboard if it runs.
    fn terminate(self: &Arc<Self>, context: &ServiceContext) {
        let inline = {
            let mut state = self.state.lock();
            let inline = state.inline.is_some();
            self.exit(&mut state);
            inline
        };
        if inline {
            self.notify_inline(context, [InlineKeyboardEvent::Finalize]);
        }
    }

    fn exit(&self, state: &mut RunState) {
        state.completed = true;
        state.text_check = None;
        state.inline = None;
        self.state_changed_event.signal();
    }

    fn notify_inline(
        self: &Arc<Self>,
        context: &ServiceContext,
        events: impl IntoIterator<Item = InlineKeyboardEvent>,
    ) {
        let frontend = context.applet_frontend();
        let replies = InlineKeyboardReplies(Arc::downgrade(self));
        for event in events {
            frontend.inline_keyboard(event, &replies);
        }
    }

    /// Queues the frontend's inline keyboard reply, unless the keyboard is closed.
    pub(super) fn inline_reply(&self, reply: InlineKeyboardReply) {
        let mut state = self.state.lock();
        let Some(keyboard) = state.inline.as_mut() else {
            return;
        };
        let data = keyboard.reply(reply);
        state
            .interactive_out
            .push(Arc::new(StorageData::new(data, None)));
    }
}

/// What an applet does after the frontend answered: the storages it pushes, and whether it waits
/// for the title's text check instead of exiting.
#[derive(Default)]
struct Step {
    out: Option<Vec<u8>>,
    interactive_out: Option<Vec<u8>>,
    text_check: Option<TextCheck>,
}

impl Step {
    fn exit(out: Vec<u8>) -> Self {
        Self {
            out: Some(out),
            ..Self::default()
        }
    }
}

pub(super) struct LibraryAppletAccessor {
    pub context: Arc<ServiceContext>,
    pub applet: Arc<LibraryApplet>,
}

impl Service for LibraryAppletAccessor {
    fn name(&self) -> &str {
        "ILibraryAppletAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let applet = &self.applet;
        Some(match ctx.command() {
            0 => push_event(ctx, &applet.state_changed_event),
            1 => push_bool(ctx, applet.state.lock().completed),
            10 => {
                applet.start(&self.context);
                ResultCode::SUCCESS
            }
            20 | 25 => {
                applet.terminate(&self.context);
                ResultCode::SUCCESS
            }
            30 | 50 | 60 => ResultCode::SUCCESS,
            100 => match input_storage(ctx) {
                Some(data) => {
                    applet.state.lock().in_data.push(data);
                    ResultCode::SUCCESS
                }
                None => UNKNOWN,
            },
            101 => match applet.state.lock().out_data.pop() {
                Some(data) => push_storage(&self.context, ctx, data),
                None => NO_DATA_IN_CHANNEL,
            },
            103 => match input_storage(ctx) {
                Some(data) => {
                    applet.push_interactive(&self.context, data);
                    ResultCode::SUCCESS
                }
                None => UNKNOWN,
            },
            104 => match applet.state.lock().interactive_out.pop() {
                Some(data) => push_storage(&self.context, ctx, data),
                None => NO_DATA_IN_CHANNEL,
            },
            105 => push_event(ctx, &applet.state.lock().out_data.event),
            106 => push_event(ctx, &applet.state.lock().interactive_out.event),
            120 => {
                ctx.push([applet.id, applet.mode]);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// The data of input object 0 when it is one of am's `IStorage`s.
fn input_storage(ctx: &HleRequestContext) -> Option<Arc<StorageData>> {
    let object = ctx.input_object(0)?;
    let storage = object.as_any()?.downcast_ref::<Storage>()?;
    Some(storage.data.clone())
}

fn error_request(data: &[u8]) -> ErrorRequest {
    let mode = data.first().copied().unwrap_or_default();
    let code_64 = || {
        let code = u64_at(data, 8);
        ErrorCode {
            category: (code >> 32) as u32,
            number: code as u32,
        }
    };
    let (code, main_text, detail_text) = match mode {
        0 if data.get(6).is_some_and(|&wide| wide != 0) => (code_64(), "", ""),
        0 => (ErrorCode::from_result(u32_at(data, 0x10)), "", ""),
        1 => (
            code_64(),
            utf8_at(data, 0x18, 0x800),
            utf8_at(data, 0x818, 0x800),
        ),
        2 => (
            ErrorCode::from_result(u32_at(data, 8)),
            utf8_at(data, 0x14, 0x800),
            utf8_at(data, 0x814, 0x800),
        ),
        4 | 5 => (code_64(), "", ""),
        _ => {
            tracing::warn!(mode, "unknown error applet mode");
            (ErrorCode::default(), "", "")
        }
    };
    ErrorRequest {
        mode,
        code,
        main_text: main_text.into(),
        detail_text: detail_text.into(),
    }
}

/// `UiSettings` (`UiSettingsV1` for library version 1, which has no purpose).
fn profile_request(version: u32, data: &[u8], user: Option<[u8; 16]>) -> ProfileSelectRequest {
    let mut invalid_users = [[0; 16]; 8];
    for (index, uid) in invalid_users.iter_mut().enumerate() {
        if let Some(bytes) = data.get(8 + index * 16..8 + (index + 1) * 16) {
            uid.copy_from_slice(bytes);
        }
    }
    ProfileSelectRequest {
        mode: u32_at(data, 0),
        purpose: if version == 1 { 0 } else { u32_at(data, 0x98) },
        invalid_users,
        current_user: user,
    }
}

/// `UiReturnArg`: the result and the selected user.
fn profile_answer(answer: ProfileSelectAnswer) -> Vec<u8> {
    let (result, uid) = match answer {
        ProfileSelectAnswer::Selected(uid) if uid != [0; 16] => (0, uid),
        _ => (u64::from(ACCOUNT_CANCELLED_BY_USER.0), [0; 16]),
    };
    let mut out = vec![0; 0x18];
    out[..8].copy_from_slice(&result.to_le_bytes());
    out[8..].copy_from_slice(&uid);
    out
}

/// `ControllerSupportArgPrivate` and the header of the mode's argument.
fn controller_request(
    private: &[u8],
    argument: &[u8],
    mut current: ControllerSetup,
) -> ControllerRequest {
    let mode = match private.get(0xa).copied().unwrap_or_default() {
        mode @ 0..=3 => mode,
        _ => match u32_at(private, 4) {
            4 => 2,
            0x10 => 3,
            _ => 0,
        },
    };
    let header = |offset: usize| match mode {
        0 | 1 => argument.get(offset).copied().unwrap_or_default(),
        _ => 0,
    };
    let single_mode = header(5) != 0;
    if single_mode {
        current.player_count = 1;
    }
    ControllerRequest {
        mode,
        min_players: header(0) as i8,
        max_players: header(1) as i8,
        single_mode,
        style_set: u32_at(private, 0xc),
        keep_connected: header(2) != 0,
        current,
    }
}

/// `ControllerSupportResultInfo`.
fn controller_answer(answer: ControllerAnswer, current: ControllerSetup) -> Vec<u8> {
    let (setup, result) = match answer {
        ControllerAnswer::Confirmed(setup) => (setup, CONTROLLER_SUCCESS),
        ControllerAnswer::Cancelled => (current, CONTROLLER_CANCEL),
    };
    let mut out = vec![0; 0xc];
    out[0] = setup.player_count as u8;
    out[4..8].copy_from_slice(&setup.npad_id.to_le_bytes());
    out[8..12].copy_from_slice(&result.to_le_bytes());
    out
}

/// A parsed full-screen keyboard config and how the title wants the text back.
struct KeyboardConfig {
    request: SoftwareKeyboardRequest,
    utf8: bool,
    text_check: bool,
}

/// Text a title is checking, and the keyboard to reopen when it rejects it.
struct TextCheck {
    config: KeyboardConfig,
    text: String,
}

/// `SwkbdConfigCommon` plus the version's tail, or the Switch 2 native config; the initial text
/// is UTF-16 in the work buffer.
fn keyboard_config(version: u32, config: &[u8], work: &[u8]) -> Option<KeyboardConfig> {
    struct Layout {
        utf8: usize,
        initial_offset: usize,
        text_check: usize,
        cancel_disabled: Option<usize>,
    }
    let native = version == KEYBOARD_NATIVE_VERSION && config.len() == KEYBOARD_NATIVE_CONFIG_SIZE;
    if !native && config.len() < KEYBOARD_CONFIG_SIZE {
        return None;
    }
    let layout = if native {
        Layout {
            utf8: 0x3aa,
            initial_offset: 0x3bc,
            text_check: 0x48d,
            cancel_disabled: Some(0x48e),
        }
    } else {
        Layout {
            utf8: 0x3bd,
            initial_offset: 0x3c0,
            text_check: 0x3d0,
            cancel_disabled: (version >= KEYBOARD_DISABLE_CANCEL_VERSION).then_some(0x4b5),
        }
    };
    let flag = |offset: usize| config.get(offset).is_some_and(|&byte| byte != 0);
    let (max_length, min_length) = text_lengths(u32_at(config, 0x3ac), u32_at(config, 0x3b0));
    let initial_offset = u32_at(config, layout.initial_offset) as usize;
    let initial_length = u32_at(config, layout.initial_offset + 4) as usize;
    Some(KeyboardConfig {
        request: SoftwareKeyboardRequest {
            kind: u32_at(config, 0),
            ok_text: utf16_at(config, 0x4, 9),
            header_text: utf16_at(config, 0x24, 65),
            sub_text: utf16_at(config, 0xa6, 129),
            guide_text: utf16_at(config, 0x1a8, 257),
            initial_text: utf16_at(work, initial_offset, initial_length),
            max_length,
            min_length,
            password: u32_at(config, 0x3b4) != 0,
            cancel_disabled: layout.cancel_disabled.is_some_and(flag),
            text_check: None,
        },
        utf8: version == KEYBOARD_UTF8_VERSION || flag(layout.utf8),
        text_check: flag(layout.text_check),
    })
}

/// A keyboard's maximum and minimum text length: the maximum is 1 to 500, else 500, and a
/// minimum above it is 0.
fn text_lengths(max: u32, min: u32) -> (u32, u32) {
    let max = match max {
        length @ 1..=KEYBOARD_MAX_LENGTH => length,
        _ => KEYBOARD_MAX_LENGTH,
    };
    (max, if min <= max { min } else { 0 })
}

fn keyboard_step(frontend: &dyn AppletFrontend, config: KeyboardConfig) -> Step {
    let answer = frontend.software_keyboard(&config.request);
    keyboard_answer(config, answer)
}

fn keyboard_answer(config: KeyboardConfig, answer: SoftwareKeyboardAnswer) -> Step {
    match answer {
        SoftwareKeyboardAnswer::Cancelled => {
            Step::exit(keyboard_output(KEYBOARD_CANCEL, "", config.utf8))
        }
        SoftwareKeyboardAnswer::Submitted(text) if config.text_check => Step {
            interactive_out: Some(text_check_request(&text, config.utf8)),
            text_check: Some(TextCheck { config, text }),
            ..Step::default()
        },
        SoftwareKeyboardAnswer::Submitted(text) => {
            Step::exit(keyboard_output(KEYBOARD_OK, &text, config.utf8))
        }
    }
}

/// Handles the title's `SwkbdTextCheck`: success exits with the text. Otherwise the keyboard
/// reopens on the text with the title's message; text confirmed unchanged exits, and anything
/// else submitted goes back to the title's check.
fn text_checked(frontend: &dyn AppletFrontend, check: TextCheck, data: &[u8]) -> Step {
    let TextCheck { mut config, text } = check;
    let result = u32_at(data, 0);
    if result == TEXT_CHECK_SUCCESS {
        return Step::exit(keyboard_output(KEYBOARD_OK, &text, config.utf8));
    }
    let message = || {
        if config.utf8 {
            utf8_at(data, 4, KEYBOARD_STRING_SIZE).into()
        } else {
            utf16_at(data, 4, KEYBOARD_STRING_SIZE / 2)
        }
    };
    let prompt = match result {
        TEXT_CHECK_FAILURE => Some(TextCheckPrompt::Rejected(message())),
        TEXT_CHECK_CONFIRM => Some(TextCheckPrompt::Confirm(message())),
        _ => None,
    };
    config.request.text_check = prompt;
    config.request.initial_text.clone_from(&text);
    match frontend.software_keyboard(&config.request) {
        SoftwareKeyboardAnswer::Submitted(submitted)
            if result == TEXT_CHECK_CONFIRM && submitted == text =>
        {
            Step::exit(keyboard_output(KEYBOARD_OK, &text, config.utf8))
        }
        answer => keyboard_answer(config, answer),
    }
}

/// The full-screen keyboard's out-data: `SwkbdResult` and the text.
fn keyboard_output(result: u32, text: &str, utf8: bool) -> Vec<u8> {
    let mut out = vec![0; 4 + KEYBOARD_STRING_SIZE];
    out[..4].copy_from_slice(&result.to_le_bytes());
    write_text(&mut out[4..], text, utf8);
    out
}

/// The interactive out-data that asks the title to check `text`: its size with the terminator,
/// then the text.
fn text_check_request(text: &str, utf8: bool) -> Vec<u8> {
    let mut out = vec![0; 8 + KEYBOARD_STRING_SIZE];
    let written = write_text(&mut out[8..], text, utf8);
    let terminator = if utf8 { 1 } else { 2 };
    out[..8].copy_from_slice(&((written + terminator) as u64).to_le_bytes());
    out
}

/// Writes `text` NUL-terminated into `buffer`, truncated to fit; returns the bytes written.
fn write_text(buffer: &mut [u8], text: &str, utf8: bool) -> usize {
    if utf8 {
        let len = text.len().min(buffer.len() - 1);
        buffer[..len].copy_from_slice(&text.as_bytes()[..len]);
        return len;
    }
    let units = text.encode_utf16().take(buffer.len() / 2 - 1);
    let mut written = 0;
    for (chunk, unit) in buffer.as_chunks_mut::<2>().0.iter_mut().zip(units) {
        *chunk = unit.to_le_bytes();
        written += 2;
    }
    written
}

fn u32_at(data: &[u8], offset: usize) -> u32 {
    data.get(offset..offset + 4)
        .map_or(0, |bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
}

fn u64_at(data: &[u8], offset: usize) -> u64 {
    data.get(offset..offset + 8)
        .map_or(0, |bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
}

/// A NUL-terminated UTF-16 string of at most `units` units.
fn utf16_at(data: &[u8], offset: usize, units: usize) -> String {
    let bytes = data.get(offset..).unwrap_or_default();
    let units: Vec<u16> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take(units)
        .map(|&unit| u16::from_le_bytes(unit))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16_lossy(&units)
}

/// A NUL-terminated UTF-8 string of at most `len` bytes; invalid UTF-8 reads as empty.
fn utf8_at(data: &[u8], offset: usize, len: usize) -> &str {
    let bytes = data.get(offset..).unwrap_or_default();
    let bytes = &bytes[..bytes.len().min(len)];
    let end = bytes
        .iter()
        .position(|&byte| byte == 0)
        .unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).unwrap_or_default()
}
