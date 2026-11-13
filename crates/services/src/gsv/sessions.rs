//! The small sessions gsv hands out: GPU time, camera, chat and the video decoder.

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::{handshake, push_readable, push_session};
use crate::{Service, ServiceContext};

/// HIPC `ResultNotSupported`.
const NOT_SUPPORTED: ResultCode = ResultCode::new(11, 1);

/// gsv 1121. Titles open it during graphics setup; GPU time spans are not measured.
pub(super) struct GpuTime;

impl Service for GpuTime {
    fn name(&self) -> &str {
        "nx2:gpu_time"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => handshake(ctx),
            11 => NOT_SUPPORTED,
            _ => return None,
        })
    }
}

/// gsv 602, the `nn::camera` context GameChat builds its camera video storage from and polls
/// every frame. 3 and 21 take the camera name, 29 the camera image width; titles run on an
/// empty success, logged once per command.
#[derive(Default)]
pub(super) struct Camera {
    logged: Mutex<BTreeSet<u32>>,
}

impl Service for Camera {
    fn name(&self) -> &str {
        "nx2:camera"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        if !matches!(command, 3 | 21 | 29) {
            return None;
        }
        if self.logged.lock().insert(command) {
            tracing::warn!("nx2:camera {command}: answering an empty success");
        }
        Some(ResultCode::SUCCESS)
    }
}

/// gsv 1202 (`nn::chat::gsv`). 1 names an event id, 11 returns that event; nothing signals it.
#[derive(Default)]
pub(super) struct Chat {
    events: Mutex<HashMap<u32, Arc<Event>>>,
}

impl Service for Chat {
    fn name(&self) -> &str {
        "nx2:chat:gsv"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            11 => {
                let id: u32 = ctx.pop();
                let event = self.events.lock().entry(id).or_default().clone();
                push_readable(ctx, &event)
            }
            _ => return None,
        })
    }
}

/// gsv 902, 905, 906 and 1002 (`nn::codec::video::detail::ipc`): TIPC 1
/// (`CreateVideoDecoderService`) hands out a [`VideoDecoderService`]. Without that session
/// `InitializeVp9Decoder` fails with InvalidHandle and the movie player aborts.
pub(super) struct VideoDecoder {
    pub context: Arc<ServiceContext>,
    pub name: &'static str,
}

impl Service for VideoDecoder {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => push_session(&self.context, ctx, VideoDecoderService::default()),
            _ => return None,
        })
    }
}

/// A video decoder. Frames are not decoded: 3 returns the submission-ready event, signalled so
/// submissions never wait, and the rest (initialize, configure, buffers, input, output info) fall
/// to the auto-stub, so movies play without a picture.
pub(super) struct VideoDecoderService {
    ready: Arc<Event>,
}

impl Default for VideoDecoderService {
    fn default() -> Self {
        let ready = Arc::new(Event::default());
        ready.signal();
        Self { ready }
    }
}

impl Service for VideoDecoderService {
    fn name(&self) -> &str {
        "nx2:vdecsvc"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            3 => push_readable(ctx, &self.ready),
            _ => return None,
        })
    }
}
