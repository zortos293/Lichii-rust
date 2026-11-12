//! Applet manager: `appletOE`, the service an application uses to talk to the system (messages,
//! focus, operation mode, launch parameters, display layers, exit).
//!
//! There is one application and no home menu, so the application is always in the foreground and
//! in focus. Its first message is `FocusStateChanged`.

mod controllers;
mod functions;
mod proxy;
mod storage;

use std::collections::VecDeque;
use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use crate::ServiceContext;

const fn am(description: u32) -> ResultCode {
    ResultCode::new(128, description)
}

pub(crate) const NO_DATA_IN_CHANNEL: ResultCode = am(2);
pub(crate) const NO_MESSAGES: ResultCode = am(3);
pub(crate) const INVALID_OFFSET: ResultCode = am(503);
pub(crate) const FATAL_SECTION_COUNT_IMBALANCE: ResultCode = am(512);

/// `nn::am::AppletMessage` values the application can receive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum AppletMessage {
    FocusStateChanged = 15,
    DetectShortPressingHomeButton = 20,
    DetectLongPressingHomeButton = 21,
    StartupLogoDisappeared = 95,
}

/// `nn::am::service::FocusState`: the application is always in focus.
const IN_FOCUS: u8 = 1;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let applet = Arc::new(Applet::new());
    context.register("appletOE", "am", move |context| {
        proxy::ApplicationProxyService {
            context: context.clone(),
            applet: applet.clone(),
        }
    })
}

/// Messages waiting for the application, and the event that says some are.
#[derive(Default)]
struct Lifecycle {
    focus_notification: bool,
    focus_changed: bool,
    unordered: VecDeque<AppletMessage>,
    signaled: bool,
}

impl Lifecycle {
    fn pending(&self) -> bool {
        (self.focus_notification && self.focus_changed) || !self.unordered.is_empty()
    }

    fn pop(&mut self) -> Option<AppletMessage> {
        if self.focus_notification && std::mem::take(&mut self.focus_changed) {
            return Some(AppletMessage::FocusStateChanged);
        }
        self.unordered.pop_front()
    }

    fn sync(&mut self, event: &Event) {
        let pending = self.pending();
        if pending != self.signaled {
            self.signaled = pending;
            if pending {
                event.signal();
            } else {
                event.clear();
            }
        }
    }
}

/// State of the application applet, shared by every `appletOE` session and its interfaces.
pub(crate) struct Applet {
    lifecycle: Mutex<Lifecycle>,
    state: Mutex<AppletState>,
    message_event: Arc<Event>,
    operation_mode_event: Arc<Event>,
    hdcp_event: Arc<Event>,
    sleep_lock_event: Arc<Event>,
    library_applet_launchable_event: Arc<Event>,
    suspended_tick_event: Arc<Event>,
    gpu_error_event: Arc<Event>,
    friend_invitation_event: Arc<Event>,
    notification_event: Arc<Event>,
    health_warning_event: Arc<Event>,
    stub_event: Arc<Event>,
}

/// Settings the application changes and reads back.
#[derive(Default)]
struct AppletState {
    fatal_sections: u32,
    vr_mode: bool,
    auto_sleep_disabled: bool,
    idle_time_detection_extension: u32,
    home_button_short_blocked: bool,
    home_button_long_blocked: bool,
    main_volume: f32,
    library_volume: f32,
    user_channel: Vec<Vec<u8>>,
    friend_invitations: Vec<Vec<u8>>,
}

impl Applet {
    fn new() -> Self {
        let applet = Self {
            lifecycle: Mutex::new(Lifecycle {
                focus_notification: true,
                focus_changed: true,
                ..Lifecycle::default()
            }),
            state: Mutex::new(AppletState {
                main_volume: 0.25,
                library_volume: 1.0,
                ..AppletState::default()
            }),
            message_event: Arc::new(Event::new()),
            operation_mode_event: Arc::new(Event::new()),
            hdcp_event: Arc::new(Event::new()),
            sleep_lock_event: Arc::new(Event::new()),
            library_applet_launchable_event: Arc::new(Event::new()),
            suspended_tick_event: Arc::new(Event::new()),
            gpu_error_event: Arc::new(Event::new()),
            friend_invitation_event: Arc::new(Event::new()),
            notification_event: Arc::new(Event::new()),
            health_warning_event: Arc::new(Event::new()),
            stub_event: Arc::new(Event::new()),
        };
        applet.lifecycle.lock().sync(&applet.message_event);
        applet
    }

    fn push_message(&self, message: AppletMessage) {
        let mut lifecycle = self.lifecycle.lock();
        lifecycle.unordered.push_back(message);
        lifecycle.sync(&self.message_event);
    }

    fn pop_message(&self) -> Option<AppletMessage> {
        let mut lifecycle = self.lifecycle.lock();
        let message = lifecycle.pop();
        lifecycle.sync(&self.message_event);
        message
    }

    fn set_focus_notification(&self, enabled: bool) {
        let mut lifecycle = self.lifecycle.lock();
        lifecycle.focus_notification = enabled;
        lifecycle.sync(&self.message_event);
    }
}

fn push_event(ctx: &mut HleRequestContext, event: &Arc<Event>) -> ResultCode {
    ctx.push_copy_object(KObject::ReadableEvent(event.clone()));
    ResultCode::SUCCESS
}

fn push_bool(ctx: &mut HleRequestContext, value: bool) -> ResultCode {
    ctx.push(u8::from(value));
    ResultCode::SUCCESS
}

#[cfg(test)]
mod tests;
