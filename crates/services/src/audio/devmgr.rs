//! Switch 2 `auddmg:u` (the virtual audio device manager) and `audsmx:u` (the sound mixer user
//! session and its streams), on the [`SysmixStreams`] they share.
//!
//! The host plays one stereo 48 kHz output, so there is one virtual output device and one virtual
//! microphone. Unknown commands keep the `nx2:` auto-stub, as the reference's do.

use std::sync::Arc;

use lichii_audio::renderer::result::{INVALID_ARGUMENT, INVALID_HANDLE, NOT_SUPPORTED};
use lichii_audio::sink::SAMPLE_RATE;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;

use super::sysmix::{
    FUNCTIONALITY_INPUT, FUNCTIONALITY_OUTPUT, SysmixStream, SysmixStreams, is_format_supported,
    is_input_format_supported, validate_region,
};
use super::{Audio, DEVICE_NAME_SIZE, device_name};
use crate::{Service, ServiceContext};

const OUTPUT_DEVICE: &str = "AudioVirtualOutput0";
const PHYSICAL_OUTPUT_DEVICE: &str = "AudioTvOutput";
const INPUT_DEVICE: &str = "AudioVirtualInput0";
const PHYSICAL_INPUT_DEVICE: &str = "AudioBuiltInMicrophone";
/// Size of the `GetVirtualOutputDeviceProperties` block.
const PROPERTIES_SIZE: usize = 0x228;
/// Output latency the device properties report.
const LATENCY_NS: u64 = 10_000_000;

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let streams = audio.sysmix.clone();
    context.register("auddmg:u", "audio", move |context| DeviceManagerService {
        context: context.clone(),
        streams: streams.clone(),
    })?;
    let streams = audio.sysmix.clone();
    context.register("audsmx:u", "audio", move |context| SoundMixerService {
        context: context.clone(),
        streams: streams.clone(),
    })
}

fn device_for(functionality: u32) -> &'static str {
    if functionality == FUNCTIONALITY_INPUT {
        INPUT_DEVICE
    } else {
        OUTPUT_DEVICE
    }
}

/// `auddmg:u`: command 0 (16 bytes and the pid) opens the virtual device manager.
struct DeviceManagerService {
    context: Arc<ServiceContext>,
    streams: Arc<SysmixStreams>,
}

impl Service for DeviceManagerService {
    fn name(&self) -> &str {
        "nx2:auddmg:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push_interface(self.context.handler(VirtualDeviceManager {
                    streams: self.streams.clone(),
                    state_event: Arc::new(Event::new()),
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// The virtual device manager (`nn::audio::ListVirtualOutputDevices` and friends).
struct VirtualDeviceManager {
    streams: Arc<SysmixStreams>,
    state_event: Arc<Event>,
}

impl Service for VirtualDeviceManager {
    fn name(&self) -> &str {
        "nx2:auddmg:u:manager"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let functionality: u32 = ctx.pop();
                let fits = ctx
                    .writable_buffer(0)
                    .is_some_and(|buffer| buffer.size as usize >= DEVICE_NAME_SIZE);
                let count = u32::from(fits);
                if fits
                    && let Err(result) =
                        ctx.write_buffer(0, &device_name(device_for(functionality)))
                {
                    return Some(result);
                }
                ctx.push(count);
                ResultCode::SUCCESS
            }
            100 => {
                let functionality: u32 = ctx.pop();
                match ctx.write_buffer(0, &device_name(device_for(functionality))) {
                    Ok(_) => ResultCode::SUCCESS,
                    Err(result) => result,
                }
            }
            300 => {
                let [sample_type, channels, rate, functionality]: [u32; 4] = ctx.pop();
                let supported = match functionality {
                    FUNCTIONALITY_INPUT => is_input_format_supported(sample_type, channels, rate),
                    FUNCTIONALITY_OUTPUT => is_format_supported(sample_type, channels, rate),
                    _ => false,
                };
                if supported {
                    ResultCode::SUCCESS
                } else {
                    NOT_SUPPORTED
                }
            }
            401 => self.properties(ctx).into(),
            500 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.state_event.clone()));
                ResultCode::SUCCESS
            }
            700 => {
                let stream: u32 = ctx.pop();
                let functionality: u32 = ctx.pop();
                self.streams
                    .attach(stream, functionality, ctx.process())
                    .into()
            }
            800 => {
                let stream: u32 = ctx.pop();
                self.streams
                    .set_engaged(stream, false, ctx.process())
                    .into()
            }
            _ => return None,
        })
    }
}

impl VirtualDeviceManager {
    /// `GetVirtualOutputDeviceProperties`: the name (0x000), type 3 (0x100), 2 (0x104), the
    /// channels (0x108), the rate (0x10c), the physical device's name (0x110), the latency in ns
    /// (0x218) and a nonzero device id (0x220), which `CreateStream` requires.
    fn properties(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let functionality: u32 = ctx.pop();
        let input = functionality == FUNCTIONALITY_INPUT;
        let mut block = [0u8; PROPERTIES_SIZE];
        let name = ctx.read_buffer(0).unwrap_or_default();
        let len = name.len().min(DEVICE_NAME_SIZE - 1);
        block[..len].copy_from_slice(&name[..len]);
        if block[0] == 0 {
            block[..DEVICE_NAME_SIZE].copy_from_slice(&device_name(device_for(functionality)));
        }
        let mut put = |offset: usize, bytes: &[u8]| {
            block[offset..offset + bytes.len()].copy_from_slice(bytes);
        };
        put(0x100, &3u32.to_le_bytes());
        put(0x104, &2u32.to_le_bytes());
        put(0x108, &2u32.to_le_bytes());
        put(0x10c, &SAMPLE_RATE.to_le_bytes());
        let physical = if input {
            PHYSICAL_INPUT_DEVICE
        } else {
            PHYSICAL_OUTPUT_DEVICE
        };
        put(0x110, &device_name(physical));
        put(0x218, &LATENCY_NS.to_le_bytes());
        put(0x220, &functionality.clamp(1, 2).to_le_bytes());
        ctx.write_buffer(0, &block)?;
        Ok(())
    }
}

/// `audsmx:u`: command 0 opens a user session for the process in copy handle 0.
struct SoundMixerService {
    context: Arc<ServiceContext>,
    streams: Arc<SysmixStreams>,
}

impl Service for SoundMixerService {
    fn name(&self) -> &str {
        "nx2:audsmx:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                let process = match ctx.copy_handle::<Process>(0) {
                    Ok(process) if Arc::ptr_eq(&process, ctx.process()) => process,
                    _ => return Some(INVALID_HANDLE),
                };
                ctx.push_interface(self.context.handler(UserSession {
                    context: self.context.clone(),
                    streams: self.streams.clone(),
                    process,
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// A process's sound mixer session: it creates streams and engages them in batches.
struct UserSession {
    context: Arc<ServiceContext>,
    streams: Arc<SysmixStreams>,
    process: Arc<Process>,
}

impl Service for UserSession {
    fn name(&self) -> &str {
        "nx2:audsmx:u:session"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => self.create_stream(ctx).into(),
            2 => self.set_engaged(ctx, true).into(),
            3 => self.set_engaged(ctx, false).into(),
            _ => return None,
        })
    }
}

impl UserSession {
    /// `CreateStream`: device (1 output, 2 input), creation flags, the applet resource user id
    /// and the guest ring; replies with the stream id and its session.
    fn create_stream(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let device: u32 = ctx.pop();
        let flags: u32 = ctx.pop();
        let _applet_resource_user_id: u64 = ctx.pop();
        let buffer: u64 = ctx.pop();
        let size: u64 = ctx.pop();
        let known = device == FUNCTIONALITY_OUTPUT || device == FUNCTIONALITY_INPUT;
        if !known || !Arc::ptr_eq(&self.process, ctx.process()) {
            return Err(INVALID_ARGUMENT);
        }
        let input = device == FUNCTIONALITY_INPUT;
        validate_region(self.process.memory(), buffer, size, flags, input)?;
        let stream = self
            .streams
            .create(&self.process, buffer, size, flags, input);
        tracing::info!(
            stream = stream.id(),
            input,
            buffer = format_args!("{buffer:#x}"),
            size = format_args!("{size:#x}"),
            "sysmix stream created"
        );
        ctx.push(stream.id());
        ctx.push_interface(self.context.handler(StreamSession {
            streams: self.streams.clone(),
            stream,
        }));
        Ok(())
    }

    /// `EngageStreams` and `DisengageStreams`: the stream ids in input buffer 0, stopping at the
    /// first failure.
    fn set_engaged(&self, ctx: &HleRequestContext, engaged: bool) -> HleResult<()> {
        let ids = ctx.read_buffer(0).unwrap_or_default();
        if !ids.len().is_multiple_of(4) {
            return Err(INVALID_ARGUMENT);
        }
        for id in ids.as_chunks::<4>().0 {
            self.streams
                .set_engaged(u32::from_le_bytes(*id), engaged, ctx.process())?;
        }
        Ok(())
    }
}

/// One stream's session. Closing it closes the stream.
struct StreamSession {
    streams: Arc<SysmixStreams>,
    stream: Arc<SysmixStream>,
}

impl Drop for StreamSession {
    fn drop(&mut self) {
        self.streams.remove(self.stream.id());
    }
}

impl Service for StreamSession {
    fn name(&self) -> &str {
        "nx2:audsmx:u:stream"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let stream = &self.stream;
        Some(match ctx.command() {
            0 => {
                let [sample_type, channels, rate, frames, prefill, flags]: [u32; 6] = ctx.pop();
                stream
                    .open(sample_type, channels, rate, frames, prefill, flags)
                    .into()
            }
            1 => {
                stream.close();
                ResultCode::SUCCESS
            }
            2 => {
                ctx.push(stream.state());
                ResultCode::SUCCESS
            }
            3 => {
                ctx.push(stream.gain());
                ResultCode::SUCCESS
            }
            4 => {
                let gain: f32 = ctx.pop();
                let _: u32 = ctx.pop();
                let ramp: i64 = ctx.pop();
                stream.set_gain(gain, ramp).into()
            }
            5 => {
                ctx.push_copy_object(KObject::ReadableEvent(stream.state_event().clone()));
                ResultCode::SUCCESS
            }
            6 | 12 => ResultCode::SUCCESS,
            7 => {
                let mode: u32 = ctx.pop();
                let threshold: i32 = ctx.pop();
                if let Err(result) = stream.set_data_notification(mode, threshold) {
                    return Some(result);
                }
                ctx.push_copy_object(KObject::ReadableEvent(stream.data_event().clone()));
                ResultCode::SUCCESS
            }
            8 => stream.set_data_notification(0, 0).into(),
            9 | 10 => NOT_SUPPORTED,
            11 => {
                let frames: u32 = ctx.pop();
                let (result, purged) = match stream.purge(frames) {
                    Ok(purged) => (ResultCode::SUCCESS, purged),
                    Err(result) => (result, 0),
                };
                ctx.push(purged);
                result
            }
            _ => return None,
        })
    }
}
