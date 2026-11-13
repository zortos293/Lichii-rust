//! `audren:u` (`IAudioRendererManager`), `IAudioRenderer` and `IAudioDevice`, on the render loop
//! in [`lichii_audio::renderer::render`]. `audren:a` and `audren:d` are registered without
//! commands, as the reference has them.

use std::sync::Arc;

use lichii_audio::renderer::device::{AUDIO_OUTPUT_DEVICE_NAMES, audio_device_names};
use lichii_audio::renderer::make_magic;
use lichii_audio::renderer::render::{self, RendererSession};
use lichii_audio::renderer::result::{INSUFFICIENT_BUFFER, INVALID_HANDLE, NOT_SUPPORTED};
use lichii_audio::renderer::{AudioRendererParameterInternal, ExecutionMode};
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;

use super::{Audio, device_name, has_device_name_output, read_device_name, write_device_names};
use crate::dispatch::Nx2Stub;
use crate::{Service, ServiceContext};

/// The device `IAudioDevice` reports as active, and the one whose volume it controls.
const ACTIVE_DEVICE: &str = "AudioTvOutput";

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let audio = audio.clone();
    context.register("audren:u", "audio", move |context| {
        AudRenU::new(context, &audio)
    })?;
    for name in ["audren:a", "audren:d"] {
        context.register(name, "audio", move |_| Nx2Stub(name.into()))?;
    }
    Ok(())
}

/// `IAudioRendererManager`, also reached through `aud:u` command 2.
pub(super) struct AudRenU {
    context: Arc<ServiceContext>,
    audio: Arc<Audio>,
}

impl AudRenU {
    pub(super) fn new(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> Self {
        Self {
            context: context.clone(),
            audio: audio.clone(),
        }
    }
}

impl Service for AudRenU {
    fn name(&self) -> &str {
        "audren:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => self.open(ctx).into(),
            1 => {
                let params: AudioRendererParameterInternal = ctx.pop();
                match render::work_buffer_size_for(&params) {
                    Ok(size) => {
                        ctx.push(size);
                        ResultCode::SUCCESS
                    }
                    Err(result) => result,
                }
            }
            2 => {
                let _applet_resource_user_id: u64 = ctx.pop();
                self.device(ctx, make_magic(b'R', b'E', b'V', b'1'))
            }
            4 => {
                let revision: u32 = ctx.pop();
                let _applet_resource_user_id: u64 = ctx.pop();
                self.device(ctx, revision)
            }
            _ => return None,
        })
    }
}

impl AudRenU {
    /// `OpenAudioRenderer`: the parameters, the work buffer's transfer memory (copy handle 0) and
    /// size, and the client process (copy handle 1).
    fn open(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let params: AudioRendererParameterInternal = ctx.pop();
        let transfer_memory_size: u64 = ctx.pop();
        let _applet_resource_user_id: u64 = ctx.pop();
        let renderer = &self.audio.renderer;
        if renderer.session_count() >= render::MAX_RENDERER_SESSIONS {
            tracing::error!("both audio renderer sessions are in use");
            return Err(lichii_audio::renderer::result::OUT_OF_SESSIONS);
        }
        let process = ctx.copy_handle::<Process>(1).map_err(|_| {
            tracing::error!("OpenAudioRenderer without a process handle");
            INVALID_HANDLE
        })?;
        let transfer = ctx.copy_handle::<TransferMemory>(0).ok();
        let event = Arc::new(Event::new());
        let signal = event.clone();
        let session = renderer.open(
            &params,
            transfer_memory_size,
            process.memory().clone(),
            Box::new(move || signal.signal()),
        )?;
        tracing::debug!(
            session = session.id(),
            revision = params.revision,
            sample_rate = params.sample_rate,
            voices = params.voices,
            "audio renderer opened"
        );
        ctx.push_interface(self.context.handler(IAudioRenderer {
            session,
            event,
            _transfer: transfer,
        }));
        Ok(())
    }

    fn device(&self, ctx: &mut HleRequestContext, revision: u32) -> ResultCode {
        let event = Arc::new(Event::new());
        event.signal();
        ctx.push_interface(self.context.handler(IAudioDevice {
            audio: self.audio.clone(),
            revision,
            event,
            notification: Arc::new(Event::new()),
        }));
        ResultCode::SUCCESS
    }
}

struct IAudioRenderer {
    session: Arc<RendererSession>,
    event: Arc<Event>,
    _transfer: Option<Arc<TransferMemory>>,
}

impl Service for IAudioRenderer {
    fn name(&self) -> &str {
        "IAudioRenderer"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let rate = self.session.system().sample_rate;
                ctx.push(rate);
                ResultCode::SUCCESS
            }
            1 => {
                let count = self.session.system().sample_count;
                ctx.push(count);
                ResultCode::SUCCESS
            }
            2 => {
                let count = self.session.system().mix_buffer_count as i32 as u32;
                ctx.push(count);
                ResultCode::SUCCESS
            }
            3 => {
                let stopped = !self.session.system().active;
                ctx.push(u32::from(stopped));
                ResultCode::SUCCESS
            }
            4 | 10 => self.request_update(ctx).into(),
            5 => {
                self.session.system().start();
                ResultCode::SUCCESS
            }
            6 => {
                self.session.system().stop();
                ResultCode::SUCCESS
            }
            7 => {
                if self.session.execution_mode() == ExecutionMode::Manual {
                    return Some(NOT_SUPPORTED);
                }
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            8 => {
                let limit: u32 = ctx.pop();
                self.session.system().render_time_limit_percent = limit;
                ResultCode::SUCCESS
            }
            9 => {
                let limit = self.session.system().render_time_limit_percent;
                ctx.push(limit);
                ResultCode::SUCCESS
            }
            12 => {
                let parameter: f32 = ctx.pop();
                self.session.system().voice_drop_parameter = parameter;
                ResultCode::SUCCESS
            }
            13 => {
                let parameter = self.session.system().voice_drop_parameter;
                ctx.push(parameter);
                ResultCode::SUCCESS
            }
            14 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}

impl IAudioRenderer {
    /// `RequestUpdate` and `RequestUpdateAuto`: the update in input buffer 0, the output in output
    /// buffer 0 and the performance frames in output buffer 1. The output is written back even
    /// when the update fails, as the reference writes it in place.
    fn request_update(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let input = ctx.read_buffer(0)?;
        let size = |index| {
            ctx.writable_buffer(index)
                .map_or(0, |buffer| buffer.size as usize)
        };
        let mut output = vec![0; size(0)];
        let mut performance = vec![0; size(1)];
        let result = self
            .session
            .system()
            .update(&input, &mut performance, &mut output);
        if !output.is_empty() {
            ctx.write_buffer(0, &output)?;
        }
        if !performance.is_empty() {
            ctx.write_buffer(1, &performance)?;
        }
        if let Err(err) = result {
            tracing::warn!("RequestUpdate failed: {:#x}", err.0);
        }
        result
    }
}

struct IAudioDevice {
    audio: Arc<Audio>,
    revision: u32,
    event: Arc<Event>,
    notification: Arc<Event>,
}

impl Service for IAudioDevice {
    fn name(&self) -> &str {
        "IAudioDevice"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let sink = &self.audio.sink;
        Some(match ctx.command() {
            0 | 6 => {
                let count = write_device_names(ctx, audio_device_names(self.revision));
                ctx.push(count);
                ResultCode::SUCCESS
            }
            1 | 7 => {
                let volume: f32 = ctx.pop();
                let Some(name) = read_device_name(ctx) else {
                    return Some(INSUFFICIENT_BUFFER);
                };
                if name == ACTIVE_DEVICE {
                    sink.set_device_volume(volume);
                }
                ResultCode::SUCCESS
            }
            2 | 8 => {
                let Some(name) = read_device_name(ctx) else {
                    return Some(INSUFFICIENT_BUFFER);
                };
                let volume = if name == ACTIVE_DEVICE {
                    sink.device_volume()
                } else {
                    1.0
                };
                ctx.push(volume);
                ResultCode::SUCCESS
            }
            3 | 10 | 13 => {
                if !has_device_name_output(ctx) {
                    return Some(INSUFFICIENT_BUFFER);
                }
                match ctx.write_buffer(0, &device_name(ACTIVE_DEVICE)) {
                    Ok(_) => ResultCode::SUCCESS,
                    Err(result) => result,
                }
            }
            4 => {
                self.event.signal();
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            11 | 12 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            5 => {
                ctx.push(sink.device_channels() as u32);
                ResultCode::SUCCESS
            }
            9 | 21 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            14 => {
                let count = write_device_names(ctx, &AUDIO_OUTPUT_DEVICE_NAMES);
                ctx.push(count);
                ResultCode::SUCCESS
            }
            15 | 17 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.notification.clone()));
                ResultCode::SUCCESS
            }
            16 | 18 | 19 => ResultCode::SUCCESS,
            20 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
