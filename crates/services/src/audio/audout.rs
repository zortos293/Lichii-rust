//! `audout:u` and `IAudioOut`.

use std::sync::Arc;

use lichii_audio::out::{AudioOut, AudioOutBuffer, AudioOutParameter};
use lichii_audio::sink::result;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;

use super::{
    Audio, buffer_event, device_name, has_device_name_output, read_device_name, write_device_names,
};
use crate::{Service, ServiceContext};

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let audio = audio.clone();
    context.register("audout:u", "audio", move |context| AudOutU {
        context: context.clone(),
        audio: audio.clone(),
    })
}

struct AudOutU {
    context: Arc<ServiceContext>,
    audio: Arc<Audio>,
}

impl Service for AudOutU {
    fn name(&self) -> &str {
        "audout:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 2 => {
                let count = write_device_names(ctx, self.audio.out.device_names());
                ctx.push(count);
                ResultCode::SUCCESS
            }
            1 | 3 => self.open(ctx).into(),
            _ => return None,
        })
    }
}

impl AudOutU {
    /// `OpenAudioOut` and `OpenAudioOutAuto`.
    fn open(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let parameter: AudioOutParameter = ctx.pop();
        let applet_resource_user_id: u64 = ctx.pop();
        let process = ctx.copy_handle::<Process>(0).map_err(|_| {
            tracing::error!("OpenAudioOut without a process handle");
            result::UNKNOWN
        })?;
        let name = read_device_name(ctx)
            .filter(|_| has_device_name_output(ctx))
            .ok_or(result::UNKNOWN)?;
        let (event, signal) = buffer_event();
        let session = self.audio.out.open(
            &name,
            parameter,
            process.memory().clone(),
            applet_resource_user_id,
            signal,
        )?;
        tracing::debug!(
            session = session.id(),
            free = self.audio.out.free_sessions(),
            "audio out opened"
        );
        ctx.write_buffer(0, &device_name(session.name()))?;
        ctx.push(session.parameter_internal());
        ctx.push_interface(self.context.handler(IAudioOut { session, event }));
        Ok(())
    }
}

struct IAudioOut {
    session: Arc<AudioOut>,
    event: Arc<Event>,
}

impl Service for IAudioOut {
    fn name(&self) -> &str {
        "IAudioOut"
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
            3 | 7 => self.append(ctx).into(),
            4 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            5 | 8 => {
                let count = released_buffers(ctx, |tags| session.released_buffers(tags));
                ctx.push(count);
                ResultCode::SUCCESS
            }
            6 => {
                let tag: u64 = ctx.pop();
                ctx.push(u8::from(session.contains_buffer(tag)));
                ResultCode::SUCCESS
            }
            9 => {
                ctx.push(session.buffer_count());
                ResultCode::SUCCESS
            }
            10 => {
                ctx.push(session.played_sample_count());
                ResultCode::SUCCESS
            }
            11 => {
                ctx.push(u8::from(session.flush()));
                ResultCode::SUCCESS
            }
            12 => {
                let volume: f32 = ctx.pop();
                session.set_volume(volume);
                ResultCode::SUCCESS
            }
            13 => {
                ctx.push(session.volume());
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

impl IAudioOut {
    /// `AppendAudioOutBuffer` and `AppendAudioOutBufferAuto`.
    fn append(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let tag: u64 = ctx.pop();
        let buffer = read_buffer_struct(ctx)?;
        self.session.append_buffer(buffer.samples, buffer.size, tag)
    }
}

/// The `AudioOutBuffer` (or `AudioInBuffer`, the same layout) in input buffer 0.
pub(super) fn read_buffer_struct(ctx: &HleRequestContext) -> HleResult<AudioOutBuffer> {
    let descriptor = ctx
        .readable_buffer(0)
        .filter(|buffer| buffer.size as usize >= size_of::<AudioOutBuffer>())
        .ok_or_else(|| {
            tracing::error!("input buffer is too small for an AudioOutBuffer");
            result::INSUFFICIENT_BUFFER
        })?;
    ctx.memory()
        .read(descriptor.address)
        .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)
}

/// `GetReleasedAudioOutBuffers` (and the audio in variant): writes the released tags to output
/// buffer 0, zero-filled past the last one, and returns how many were released.
pub(super) fn released_buffers(
    ctx: &HleRequestContext,
    take: impl FnOnce(&mut [u64]) -> u32,
) -> u32 {
    let mut tags = [0u64; lichii_audio::sink::device::BUFFER_COUNT];
    let capacity = ctx
        .writable_buffer(0)
        .map_or(0, |buffer| buffer.size as usize / size_of::<u64>())
        .min(tags.len());
    let count = take(&mut tags[..capacity]);
    if capacity > 0 {
        let _ = ctx.write_buffer(0, bytemuck::cast_slice(&tags[..capacity]));
    }
    count
}
