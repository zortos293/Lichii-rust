//! `audctl`: the system volume and output settings applets and some games use.
//!
//! The target volume of the active output target (the speaker) drives the sink's device
//! volume, the console's master volume. The host volume and mute from the settings apply on top
//! of it, inside the sink, so the reported target volumes start at the console default (15)
//! whatever the host volume is.

use std::sync::Arc;

use lichii_audio::sink::result;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use super::Audio;
use crate::{Service, ServiceContext};

/// `nn::settings::system::AudioOutputModeTarget` values (`None`, `Hdmi`, `Speaker`, `Headphone`,
/// `Type3`, `Type4`).
const TARGETS: usize = 6;
const TARGET_SPEAKER: u32 = 2;
const VOLUME_MAX: i32 = 15;
/// `nn::settings::system::AudioOutputMode::ch_2`.
const OUTPUT_MODE_STEREO: u32 = 1;
/// `nn::settings::system::AudioOutputMode::ch_7_1`, what `GetOutputModeSetting` reports.
const OUTPUT_MODE_7_1: u32 = 3;

struct State {
    volumes: [i32; TARGETS],
    muted: [bool; TARGETS],
    output_modes: [u32; TARGETS],
    active: u32,
    speaker_auto_mute: bool,
}

/// What every `audctl` session shares.
struct Controller {
    audio: Arc<Audio>,
    notification: Arc<Event>,
    state: Mutex<State>,
}

impl Controller {
    /// Applies the active target's volume (0 when muted) to the sink.
    fn apply(&self, state: &State) {
        let target = state.active as usize;
        let volume = if state.muted[target] {
            0.0
        } else {
            state.volumes[target] as f32 / VOLUME_MAX as f32
        };
        self.audio.sink.set_device_volume(volume);
    }
}

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let controller = Arc::new(Controller {
        audio: audio.clone(),
        notification: Arc::new(Event::new()),
        state: Mutex::new(State {
            volumes: [VOLUME_MAX; TARGETS],
            muted: [false; TARGETS],
            output_modes: [OUTPUT_MODE_STEREO; TARGETS],
            active: TARGET_SPEAKER,
            speaker_auto_mute: false,
        }),
    });
    context.register("audctl", "audio", move |context| AudCtl {
        context: context.clone(),
        controller: controller.clone(),
    })
}

struct AudCtl {
    context: Arc<ServiceContext>,
    controller: Arc<Controller>,
}

fn target_index(target: u32) -> HleResult<usize> {
    let index = target as usize;
    if index >= TARGETS {
        tracing::error!(target, "invalid audio output target");
        return Err(result::INVALID_ARGUMENT);
    }
    Ok(index)
}

impl Service for AudCtl {
    fn name(&self) -> &str {
        "audctl"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let controller = &self.controller;
        Some(match ctx.command() {
            0 => {
                let target: u32 = ctx.pop();
                target_index(target)
                    .map(|index| ctx.push(controller.state.lock().volumes[index]))
                    .into()
            }
            1 => {
                let (target, volume): (u32, i32) = (ctx.pop(), ctx.pop());
                target_index(target)
                    .map(|index| {
                        let mut state = controller.state.lock();
                        state.volumes[index] = volume.clamp(0, VOLUME_MAX);
                        if state.active == target {
                            controller.apply(&state);
                        }
                    })
                    .into()
            }
            2 => {
                ctx.push(0i32);
                ResultCode::SUCCESS
            }
            3 => {
                ctx.push(VOLUME_MAX);
                ResultCode::SUCCESS
            }
            4 => {
                let target: u32 = ctx.pop();
                target_index(target)
                    .map(|index| ctx.push(u8::from(controller.state.lock().muted[index])))
                    .into()
            }
            5 => {
                let (muted, target): (u8, u32) = (ctx.pop(), ctx.pop());
                target_index(target)
                    .map(|index| {
                        let mut state = controller.state.lock();
                        state.muted[index] = muted != 0;
                        if state.active == target {
                            controller.apply(&state);
                        }
                    })
                    .into()
            }
            9 => {
                let target: u32 = ctx.pop();
                target_index(target)
                    .map(|index| ctx.push(controller.state.lock().output_modes[index]))
                    .into()
            }
            10 => {
                let (target, mode): (u32, u32) = (ctx.pop(), ctx.pop());
                target_index(target)
                    .map(|index| controller.state.lock().output_modes[index] = mode)
                    .into()
            }
            12 | 18 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            13 => {
                ctx.push(OUTPUT_MODE_7_1);
                ResultCode::SUCCESS
            }
            14 | 17 | 22 => ResultCode::SUCCESS,
            30 => {
                let enabled: u8 = ctx.pop();
                controller.state.lock().speaker_auto_mute = enabled != 0;
                ResultCode::SUCCESS
            }
            31 => {
                ctx.push(u8::from(controller.state.lock().speaker_auto_mute));
                ResultCode::SUCCESS
            }
            32 => {
                ctx.push(controller.state.lock().active);
                ResultCode::SUCCESS
            }
            34 => {
                ctx.push_copy_object(KObject::ReadableEvent(controller.notification.clone()));
                ResultCode::SUCCESS
            }
            5000 => {
                ctx.push_interface(self.context.handler(AudCtl {
                    context: self.context.clone(),
                    controller: controller.clone(),
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
