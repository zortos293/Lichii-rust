//! `audin:u` and `IAudioIn`. There is no host capture: every session records silence.

use std::sync::Arc;

use lichii_audio::r#in::{AudioIn, AudioInParameter, is_uac};
use lichii_audio::sink::result;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;

use super::audout::{read_buffer_struct, released_buffers};
use super::{
    Audio, buffer_event, device_name, has_device_name_output, read_device_name, write_device_names,
};
use crate::{Service, ServiceContext};

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let audio = audio.clone();
    context.register("audin:u", "audio", move |context| manager(context, &audio))
}

/// `IAudioInManager`, also reached through `aud:u` command 0.
pub(super) fn manager(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> impl Service + use<> {
    AudInU {
        context: context.clone(),
        audio: audio.clone(),
    }
}

struct AudInU {
    context: Arc<ServiceContext>,
    audio: Arc<Audio>,
}

impl Service for AudInU {
    fn name(&self) -> &str {
        "audin:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 2 | 4 => {
                let count = write_device_names(ctx, self.audio.input.device_names());
                ctx.push(count);
                ResultCode::SUCCESS
            }
            1 | 3 => self.open(ctx, [0; 2]).into(),
            5 => {
                let protocol: [u32; 2] = ctx.pop();
                self.open(ctx, protocol).into()
            }
            _ => return None,
        })
    }
}

impl AudInU {
    /// `OpenAudioIn`, `OpenAudioInAuto` and `OpenAudioInProtocolSpecified`.
    fn open(&self, ctx: &mut HleRequestContext, protocol: [u32; 2]) -> HleResult<()> {
        let parameter: AudioInParameter = ctx.pop();
        let applet_resource_user_id: u64 = ctx.pop();
        let process = ctx.copy_handle::<Process>(0).map_err(|_| {
            tracing::error!("OpenAudioIn without a process handle");
            result::UNKNOWN
        })?;
        let name = read_device_name(ctx)
            .filter(|_| has_device_name_output(ctx))
            .ok_or(result::UNKNOWN)?;
        let (event, signal) = buffer_event();
        let session = self.audio.input.open(
            &name,
            parameter,
            process.memory().clone(),
            applet_resource_user_id,
            signal,
        )?;
        let out_name = match (protocol, is_uac(session.name())) {
            ([0, 0], true) => "UacIn",
            ([0, 0], false) => "DeviceIn",
            _ => session.name(),
        };
        ctx.write_buffer(0, &device_name(out_name))?;
        ctx.push(session.parameter_internal());
        ctx.push_interface(self.context.handler(IAudioIn { session, event }));
        Ok(())
    }
}

struct IAudioIn {
    session: Arc<AudioIn>,
    event: Arc<Event>,
}

impl Service for IAudioIn {
    fn name(&self) -> &str {
        "IAudioIn"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let session = &self.session;
        Some(match ctx.command() {
            0 => {
                ctx.push(session.state() as u32);
                ResultCode::SUCCESS
            }
            1 => session.start().into(),
            2 => session.stop().into(),
            3 | 7 | 8 | 10 => {
                let tag: u64 = ctx.pop();
                read_buffer_struct(ctx)
                    .and_then(|buffer| session.append_buffer(buffer.samples, buffer.size, tag))
                    .into()
            }
            4 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            5 | 9 => {
                let count = released_buffers(ctx, |tags| session.released_buffers(tags));
                ctx.push(count);
                ResultCode::SUCCESS
            }
            6 => {
                let tag: u64 = ctx.pop();
                ctx.push(u8::from(session.contains_buffer(tag)));
                ResultCode::SUCCESS
            }
            11 => {
                ctx.push(session.buffer_count());
                ResultCode::SUCCESS
            }
            12 => {
                let gain: f32 = ctx.pop();
                session.set_volume(gain);
                ResultCode::SUCCESS
            }
            13 => {
                ctx.push(session.volume());
                ResultCode::SUCCESS
            }
            14 => {
                ctx.push(u8::from(session.flush()));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
