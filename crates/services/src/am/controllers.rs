//! `ICommonStateGetter`, `ISelfController`, `IWindowController`, `IAudioController`,
//! `IDisplayController` and `ILockAccessor`.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;

use super::{
    Applet, AppletMessage, FATAL_SECTION_COUNT_IMBALANCE, IN_FOCUS, NO_MESSAGES, push_bool,
    push_event,
};
use crate::dispatch::{push_nx2_stub, push_stub_zeroes};
use crate::{Service, ServiceContext};

/// An interface without commands of its own; everything it gets is an unknown command.
pub(super) struct Interface(pub &'static str);

impl Service for Interface {
    fn name(&self) -> &str {
        self.0
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

fn push_stub_event(ctx: &mut HleRequestContext, event: &Arc<Event>) -> ResultCode {
    push_stub_zeroes(ctx);
    push_event(ctx, event)
}

pub(super) struct CommonStateGetter {
    pub context: Arc<ServiceContext>,
    pub applet: Arc<Applet>,
}

impl Service for CommonStateGetter {
    fn name(&self) -> &str {
        "ICommonStateGetter"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let applet = &self.applet;
        let docked = self.context.settings.system.docked;
        Some(match ctx.command() {
            0 => push_event(ctx, &applet.message_event),
            1 => match applet.pop_message() {
                Some(message) => {
                    ctx.push(message as u32);
                    ResultCode::SUCCESS
                }
                None => NO_MESSAGES,
            },
            5 => push_bool(ctx, docked),
            6 => {
                ctx.push(u32::from(docked));
                ResultCode::SUCCESS
            }
            8 => push_bool(ctx, false),
            9 => {
                ctx.push(IN_FOCUS);
                ResultCode::SUCCESS
            }
            10 => {
                applet.sleep_lock_event.signal();
                ResultCode::SUCCESS
            }
            11 | 12 => {
                applet.sleep_lock_event.clear();
                ResultCode::SUCCESS
            }
            13 => push_event(ctx, &applet.sleep_lock_event),
            15 => push_stub_event(ctx, &applet.stub_event),
            30..=32 => {
                ctx.push_interface(self.context.handler(LockAccessor::new()));
                ResultCode::SUCCESS
            }
            50 => push_bool(ctx, applet.state.lock().vr_mode),
            51 => {
                applet.state.lock().vr_mode = ctx.pop::<u8>() != 0;
                ResultCode::SUCCESS
            }
            53 | 54 => {
                applet.state.lock().vr_mode = ctx.command() == 53;
                ResultCode::SUCCESS
            }
            52 | 66 | 900 => ResultCode::SUCCESS,
            55 => push_bool(ctx, false),
            60 => {
                let resolution: [i32; 2] = if docked { [1920, 1080] } else { [1280, 720] };
                ctx.push(resolution);
                ResultCode::SUCCESS
            }
            61 => push_event(ctx, &applet.operation_mode_event),
            62 => {
                ctx.push(1i32);
                ResultCode::SUCCESS
            }
            63 => push_event(ctx, &applet.hdcp_event),
            68 | 200 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            80 => {
                let state = applet.state.lock();
                let message = match ctx.pop::<u32>() {
                    1 if !state.home_button_short_blocked => {
                        Some(AppletMessage::DetectShortPressingHomeButton)
                    }
                    2 if !state.home_button_long_blocked => {
                        Some(AppletMessage::DetectLongPressingHomeButton)
                    }
                    _ => None,
                };
                drop(state);
                if let Some(message) = message {
                    applet.push_message(message);
                }
                ResultCode::SUCCESS
            }
            100 => {
                applet.state.lock().home_button_short_blocked = ctx.pop::<u8>() == 0;
                ResultCode::SUCCESS
            }
            110 => push_nx2_stub(&self.context, ctx, "ICommonStateGetter"),
            120 => {
                const APPLICATION: u32 = 1;
                let written = ctx.write_buffer(0, &APPLICATION.to_le_bytes()).unwrap_or(0);
                ctx.push(i32::from(written >= 4));
                ResultCode::SUCCESS
            }
            130 => {
                applet.push_message(AppletMessage::StartupLogoDisappeared);
                ResultCode::SUCCESS
            }
            300 => {
                ctx.push(1i32);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(super) struct SelfController {
    pub context: Arc<ServiceContext>,
    pub applet: Arc<Applet>,
}

impl Service for SelfController {
    fn name(&self) -> &str {
        "ISelfController"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let applet = &self.applet;
        Some(match ctx.command() {
            0 => {
                tracing::info!("application requested exit");
                self.context.kernel.terminate_process(&self.context.process);
                ResultCode::SUCCESS
            }
            3 => {
                applet.state.lock().fatal_sections += 1;
                ResultCode::SUCCESS
            }
            4 => {
                let mut state = applet.state.lock();
                if state.fatal_sections == 0 {
                    return Some(FATAL_SECTION_COUNT_IMBALANCE);
                }
                state.fatal_sections -= 1;
                ResultCode::SUCCESS
            }
            9 => {
                applet.library_applet_launchable_event.signal();
                push_event(ctx, &applet.library_applet_launchable_event)
            }
            13 => {
                let notify = ctx.pop::<u8>() != 0;
                applet.set_focus_notification(notify);
                ResultCode::SUCCESS
            }
            1 | 2 | 10..=12 | 14..=16 | 19 | 50 | 51 | 60 | 61 | 65 | 72 | 100 | 120 | 130 => {
                ResultCode::SUCCESS
            }
            62 => {
                applet.state.lock().idle_time_detection_extension = ctx.pop();
                ResultCode::SUCCESS
            }
            63 => {
                ctx.push(applet.state.lock().idle_time_detection_extension);
                ResultCode::SUCCESS
            }
            67 => push_bool(ctx, false),
            68 => {
                applet.state.lock().auto_sleep_disabled = ctx.pop::<u8>() != 0;
                ResultCode::SUCCESS
            }
            69 => push_bool(ctx, applet.state.lock().auto_sleep_disabled),
            90 => {
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            91 => push_event(ctx, &applet.suspended_tick_event),
            1000 => push_nx2_stub(&self.context, ctx, "ISelfController"),
            _ => return None,
        })
    }
}

pub(super) struct WindowController {
    pub context: Arc<ServiceContext>,
}

impl Service for WindowController {
    fn name(&self) -> &str {
        "IWindowController"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => push_nx2_stub(&self.context, ctx, "IWindowController"),
            1 => {
                ctx.push(self.context.process.id());
                ResultCode::SUCCESS
            }
            2 => {
                ctx.push(0u64);
                ResultCode::SUCCESS
            }
            10..=12 | 20 | 21 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}

pub(super) struct AudioController {
    pub applet: Arc<Applet>,
}

impl Service for AudioController {
    fn name(&self) -> &str {
        "IAudioController"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let mut state = self.applet.state.lock();
        match ctx.command() {
            0 => {
                let (main, library): (f32, f32) = (ctx.pop(), ctx.pop());
                state.main_volume = main.clamp(0.0, 1.0);
                state.library_volume = library.clamp(0.0, 1.0);
            }
            1 => ctx.push(state.main_volume),
            2 => ctx.push(state.library_volume),
            3 => state.main_volume = ctx.pop::<f32>().clamp(0.0, 1.0),
            4 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

pub(super) struct DisplayController {
    pub applet: Arc<Applet>,
}

impl Service for DisplayController {
    fn name(&self) -> &str {
        "IDisplayController"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            10 | 12 | 14 | 16..=18 => push_stub_event(ctx, &self.applet.stub_event),
            _ => return None,
        })
    }
}

/// `ILockAccessor`: a lock nothing else contends for, with its event, which starts signalled
/// because the lock starts free (libnx waits on it before `TryLock`).
struct LockAccessor {
    locked: AtomicBool,
    event: Arc<Event>,
}

impl LockAccessor {
    fn new() -> Self {
        let event = Arc::new(Event::new());
        event.signal();
        Self {
            locked: AtomicBool::new(false),
            event,
        }
    }
}

impl Service for LockAccessor {
    fn name(&self) -> &str {
        "ILockAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => {
                let return_handle = ctx.pop::<u8>() != 0;
                let acquired = !self.locked.swap(true, Ordering::AcqRel);
                ctx.push(u8::from(acquired));
                if return_handle {
                    push_event(ctx, &self.event);
                }
                ResultCode::SUCCESS
            }
            2 => {
                self.locked.store(false, Ordering::Release);
                self.event.signal();
                ResultCode::SUCCESS
            }
            3 => push_event(ctx, &self.event),
            4 => push_bool(ctx, self.locked.load(Ordering::Acquire)),
            _ => return None,
        })
    }
}
