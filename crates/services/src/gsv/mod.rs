//! `nx2:gsv`, the Switch 2 graphics system. `ISelfController` 2000 hands the application a session
//! to it, and every Switch 2 graphics SDK boots through it: `nn::syncpt` (202, 204), `nn::gpu`
//! (497), `nn::vi2` (305), the codec, camera and chat SDKs (901-1002, 601/602, 1201/1202) and
//! the GPU time session (1121). All of these speak TIPC: data follows a one-word result, and a
//! returned session or event is a copy handle.
//!
//! Commands nobody has decoded fall to the `nx2:` auto-stub (success and zeroed data), which is
//! what the console's server answers for the feature queries titles make at boot.

mod gpu;
mod native;
mod reserve;
mod sessions;
mod syncpt;
#[cfg(test)]
mod tests;
mod vdec;
mod vi2;

use std::collections::BTreeSet;
use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

pub(crate) use reserve::Reserve;

/// The service thread every gsv session runs on, except vi2.
const THREAD: &str = "gsv";
/// vi2's thread: presents wait for the GPU and pace the title there.
const VI2_THREAD: &str = "nx2:vi2";

/// Eden's `ResultUnknown`.
const UNKNOWN: ResultCode = ResultCode(u32::MAX);

/// `ISelfController` 2000: a new graphics system session, as a copy handle.
pub(crate) fn open(context: &Arc<ServiceContext>, ctx: &mut HleRequestContext) -> ResultCode {
    push_session(
        context,
        ctx,
        GraphicsSystem {
            context: context.clone(),
            syncpt: Arc::default(),
            objects: Arc::default(),
            state: Mutex::default(),
        },
    )
}

/// Replies with a copy handle to a new session served by `service` on the gsv thread.
fn push_session<S: Service>(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    service: S,
) -> ResultCode {
    push_session_on(context, ctx, THREAD, service)
}

/// Like [`push_session`], on service thread `thread`.
fn push_session_on<S: Service>(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    thread: &str,
    service: S,
) -> ResultCode {
    match context.service_thread(thread) {
        Ok(thread) => {
            let session = thread.session(context.handler(service));
            ctx.push_copy_object(KObject::ClientSession(session));
            ResultCode::SUCCESS
        }
        Err(code) => code,
    }
}

/// 1402: a session to `nx2:nvsrv`, which speaks the nvdrv protocol, initialized for the title's
/// process. It runs on the nvdrv thread with the other nvdrv sessions.
fn open_nvsrv(context: &Arc<ServiceContext>, ctx: &mut HleRequestContext) -> ResultCode {
    let Some(nvdrv) = context.nvdrv.get() else {
        tracing::error!("nx2:gsv 1402: nvdrv is not registered");
        return UNKNOWN;
    };
    match context.service_thread("nvdrv") {
        Ok(thread) => {
            let handler = nvdrv.nvsrv_session(context, context.process.clone());
            ctx.push_copy_object(KObject::ClientSession(thread.session(handler)));
            ResultCode::SUCCESS
        }
        Err(code) => code,
    }
}

/// TIPC 0(u64 token) on the Switch 2 sessions: newer SDKs send it first and abort unless the
/// reply echoes the token.
fn handshake(ctx: &mut HleRequestContext) -> ResultCode {
    let token: u64 = ctx.pop();
    ctx.push(token);
    ResultCode::SUCCESS
}

fn push_readable(ctx: &mut HleRequestContext, event: &Arc<Event>) -> ResultCode {
    ctx.push_copy_object(KObject::ReadableEvent(event.clone()));
    ResultCode::SUCCESS
}

struct GraphicsSystem {
    context: Arc<ServiceContext>,
    syncpt: Arc<syncpt::PageSlot>,
    objects: Arc<Mutex<native::NativeObjects>>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    client_handle_registered: bool,
    address_spaces: BTreeSet<u32>,
}

impl Service for GraphicsSystem {
    fn name(&self) -> &str {
        "nx2:gsv"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let context = &self.context;
        Some(match ctx.command() {
            4 | 5 | 101 | 601 | 901 | 1001 | 1201 => ResultCode::SUCCESS,
            11 => {
                let mut state = self.state.lock();
                if std::mem::replace(&mut state.client_handle_registered, true) {
                    tracing::warn!("nx2:gsv 11: client handle registered twice");
                    UNKNOWN
                } else {
                    ResultCode::SUCCESS
                }
            }
            115 => {
                let requested: u64 = ctx.pop();
                tracing::info!("nx2:gsv 115: granting features {requested:#x}");
                ctx.push(requested);
                ResultCode::SUCCESS
            }
            126 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            201 | 301 | 401..=403 | 494 | 801 | 1101 | 1102 | 1501..=1503 | 1601 => {
                ResultCode::SUCCESS
            }
            202 => push_session(
                context,
                ctx,
                syncpt::SyncptSession::new(context, self.syncpt.clone()),
            ),
            204 => syncpt::attach(context, ctx, &self.syncpt),
            305 => push_session_on(
                context,
                ctx,
                VI2_THREAD,
                vi2::Vi2::new(context, self.objects.clone()),
            ),
            497 => push_session(context, ctx, gpu::GpuSession::new(context)),
            1402 => open_nvsrv(context, ctx),
            111 | 112 | 114 | 423 | 426 | 428 | 429 | 430 | 435 | 911 | 912 | 961 | 991 | 1211
            | 1291 | 1510 => native::handle(&self.objects, ctx),
            434 => {
                let id: u32 = ctx.pop();
                self.state.lock().address_spaces.insert(id);
                ResultCode::SUCCESS
            }
            602 => push_session(context, ctx, sessions::Camera::default()),
            902 | 905 | 906 | 1002 => {
                let name = if ctx.command() == 1002 {
                    "nx2:vdec2"
                } else {
                    "nx2:vdec"
                };
                let decoder = vdec::VideoDecoder {
                    context: context.clone(),
                    objects: self.objects.clone(),
                    name,
                };
                push_session(context, ctx, decoder)
            }
            1111 => {
                let version: u32 = ctx.pop();
                tracing::info!("nx2:gsv 1111: feature version {version:#x}");
                ResultCode::SUCCESS
            }
            1121 => push_session(context, ctx, sessions::GpuTime),
            1202 => push_session(context, ctx, sessions::Chat::default()),
            _ => return None,
        })
    }
}

/// Little-endian words, indexed from `first`. Missing words read as zero.
pub(super) struct Words<'a> {
    data: &'a [u8],
    first: usize,
}

impl<'a> Words<'a> {
    /// A TIPC request's data, indexed as the reference's command buffer is: from word 2.
    pub(super) fn command(ctx: &'a HleRequestContext) -> Self {
        Self {
            data: ctx.input(),
            first: 2,
        }
    }

    pub(super) fn of(data: &'a [u8]) -> Self {
        Self { data, first: 0 }
    }

    pub(super) fn u32(&self, index: usize) -> u32 {
        let at = (index - self.first) * 4;
        self.data
            .get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    pub(super) fn u64(&self, index: usize) -> u64 {
        u64::from(self.u32(index)) | (u64::from(self.u32(index + 1)) << 32)
    }
}
