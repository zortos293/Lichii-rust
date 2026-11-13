//! NVIDIA driver services: `nvdrv`, `nvdrv:a`, `nvdrv:s`, `nvdrv:t` and the Switch 2 `nx2:nvsrv`
//! sessions gsv hands out (TIPC 1402), which speak the same INvDrvServices protocol and need no
//! `Initialize` (NX2_NOTES row 23).
//!
//! A session opens device nodes as file descriptors and drives them with ioctls. The devices sit on
//! the GPU crate: `/dev/nvmap` pins guest memory into host1x's device space,
//! `/dev/nvhost-as-gpu` maps it into a [`Gmmu`](lichii_gpu::gmmu::Gmmu), `/dev/nvhost-gpu` submits
//! GPFIFO entries through [`Gpu::submit`] and `/dev/nvhost-ctrl` waits on host1x syncpoints. The
//! maximum each syncpoint reaches is host1x's alone; nvdrv only records which ids it handed out.
//!
//! Every session has its own descriptor table. nvmap handles, the syncpoint reservations and the
//! display are shared by every session ([`Nvdrv`]).

mod as_gpu;
mod ctrl;
mod ctrl_gpu;
mod disp;
mod gpu_channel;
mod host1x_client;
mod ioctl;
mod nvmap;
mod stub;
mod syncpoints;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytemuck::{Pod, Zeroable};
use lichii_common::{ConsoleGeneration, HleResult, ResultCode};
use lichii_gpu::{Console, Gpu};
use lichii_kernel::Event;
use lichii_kernel::ipc::{HleRequestContext, SessionHandler};
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};
use ioctl::{Args, Ioctl, Variant};
use nvmap::NvMap;
use syncpoints::SyncpointTable;

/// `NvResult`: the status word every INvDrvServices reply carries next to a successful IPC result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum NvResult {
    Success = 0x0,
    NotImplemented = 0x1,
    NotSupported = 0x2,
    NotInitialized = 0x3,
    BadParameter = 0x4,
    Timeout = 0x5,
    InsufficientMemory = 0x6,
    InvalidState = 0x8,
    InvalidAddress = 0x9,
    InvalidSize = 0xA,
    BadValue = 0xB,
    AlreadyAllocated = 0xD,
    Busy = 0xE,
    FileOperationFailed = 0x3_0003,
    ConfigVarNotFound = 0x3_0006,
    AccessDenied = 0x3_0010,
}

/// A syncpoint threshold as ioctls carry it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub(crate) struct NvFence {
    pub id: i32,
    pub value: u32,
}

/// What every nvdrv session shares: the GPU, nvmap handles and syncpoint reservations.
pub(crate) struct Nvdrv {
    gpu: Arc<Gpu>,
    /// The title is a Switch 2 title: its NvRm gets the Ampere device and Switch 2 behaviour.
    nx2: bool,
    nvmap: NvMap,
    syncpoints: SyncpointTable,
    next_session: AtomicU64,
    /// `debug.ipc_trace`: log every ioctl with its device and result.
    ipc_trace: bool,
}

impl Nvdrv {
    fn new(context: &ServiceContext) -> Self {
        let gpu = context.gpu.clone();
        Self {
            nvmap: NvMap::new(gpu.clone()),
            syncpoints: SyncpointTable::new(gpu.host1x().syncpoints.clone()),
            nx2: context.program.generation == ConsoleGeneration::Switch2,
            next_session: AtomicU64::new(1),
            ipc_trace: context.settings.debug.ipc_trace,
            gpu,
        }
    }

    fn console(&self) -> Console {
        if self.nx2 {
            Console::Switch2
        } else {
            Console::Switch1
        }
    }
}

/// Registers the nvdrv services. Returns the shared state: vi presents through it and gsv opens
/// `nx2:nvsrv` sessions with it.
pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<Arc<Nvdrv>> {
    let nvdrv = Arc::new(Nvdrv::new(context));
    for name in ["nvdrv", "nvdrv:a", "nvdrv:s", "nvdrv:t"] {
        let shared = nvdrv.clone();
        context.register(name, "nvdrv", move |_| {
            NvdrvService::new(shared.clone(), name, None)
        })?;
    }
    let _ = context.nvdrv.set(nvdrv.clone());
    Ok(nvdrv)
}

impl Nvdrv {
    /// A new `nx2:nvsrv` session for `process`, already initialized, for gsv to return.
    pub(crate) fn nvsrv_session(
        self: &Arc<Self>,
        context: &Arc<ServiceContext>,
        process: Arc<Process>,
    ) -> Arc<dyn SessionHandler> {
        context.handler(NvdrvService::new(self.clone(), "nx2:nvsrv", Some(process)))
    }
}

/// The process a session works for, set by `Initialize`.
struct Client {
    process: Arc<Process>,
    session: u64,
}

impl Client {
    fn memory(&self) -> &Arc<AddressSpace> {
        self.process.memory()
    }
}

/// Reused per session, so an ioctl allocates nothing once the buffers reached their size.
#[derive(Default)]
struct Buffers {
    input: Vec<u8>,
    inline_input: Vec<u8>,
    output: Vec<u8>,
    inline_output: Vec<u8>,
}

impl Buffers {
    fn load(&mut self, ctx: &HleRequestContext, variant: Variant) -> Result<(), ResultCode> {
        read_into(ctx, 0, &mut self.input)?;
        if variant == Variant::Two {
            read_into(ctx, 1, &mut self.inline_input)?;
        } else {
            self.inline_input.clear();
        }
        resize(
            &mut self.output,
            ctx.writable_buffer(0).map_or(0, |b| b.size),
        );
        let inline_output = match variant {
            Variant::Three => ctx.writable_buffer(1).map_or(0, |b| b.size),
            _ => 0,
        };
        resize(&mut self.inline_output, inline_output);
        Ok(())
    }

    fn store(&self, ctx: &HleRequestContext, variant: Variant) -> Result<(), ResultCode> {
        write_from(ctx, 0, &self.output)?;
        if variant == Variant::Three {
            write_from(ctx, 1, &self.inline_output)?;
        }
        Ok(())
    }

    fn args(&mut self, variant: Variant, ioctl: Ioctl) -> Args<'_> {
        Args {
            variant,
            ioctl,
            input: &self.input,
            inline_input: &self.inline_input,
            output: &mut self.output,
            inline_output: &mut self.inline_output,
        }
    }
}

fn resize(buffer: &mut Vec<u8>, size: u64) {
    buffer.clear();
    buffer.resize(size as usize, 0);
}

fn read_into(ctx: &HleRequestContext, index: usize, into: &mut Vec<u8>) -> Result<(), ResultCode> {
    let Some(buffer) = ctx.readable_buffer(index) else {
        into.clear();
        return Ok(());
    };
    resize(into, buffer.size);
    ctx.memory()
        .read_bytes(buffer.address, into)
        .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)
}

fn write_from(ctx: &HleRequestContext, index: usize, data: &[u8]) -> Result<(), ResultCode> {
    let Some(buffer) = ctx.writable_buffer(index) else {
        return Ok(());
    };
    let len = data.len().min(buffer.size as usize);
    ctx.memory()
        .write_bytes(buffer.address, &data[..len])
        .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)
}

/// An open device node.
enum Device {
    NvMap(nvmap::NvMapDevice),
    Ctrl(Box<ctrl::Ctrl>),
    CtrlGpu(ctrl_gpu::CtrlGpu),
    AsGpu(as_gpu::AsGpu),
    Channel(gpu_channel::GpuChannel),
    Host1x(host1x_client::Host1xClient),
    Disp,
    Stub(stub::Stub),
}

impl Device {
    /// Opens the node at `path`; `None` for nodes that do not exist.
    fn open(nvdrv: &Arc<Nvdrv>, client: &Client, path: &str) -> Option<Self> {
        Some(match path {
            "/dev/nvmap" => Self::NvMap(nvmap::NvMapDevice::new(nvdrv.clone(), client)),
            "/dev/nvhost-ctrl" => Self::Ctrl(Box::new(ctrl::Ctrl::new(nvdrv.clone()))),
            "/dev/nvhost-ctrl-gpu" => Self::CtrlGpu(ctrl_gpu::CtrlGpu::new(nvdrv.clone())),
            "/dev/nvhost-as-gpu" => Self::AsGpu(as_gpu::AsGpu::new(nvdrv.clone())),
            "/dev/nvhost-gpu" => Self::Channel(gpu_channel::GpuChannel::new(nvdrv.clone(), client)),
            "/dev/nvhost-nvdec" | "/dev/nvhost-vic" | "/dev/nvhost-nvjpg" => {
                Self::Host1x(host1x_client::Host1xClient::new(nvdrv.clone(), path))
            }
            "/dev/nvdisp_disp0" => Self::Disp,
            _ if path.starts_with("/dev/nvhost-") => Self::Stub(stub::Stub::new(path)),
            _ => return None,
        })
    }

    fn name(&self) -> &str {
        match self {
            Self::NvMap(_) => "/dev/nvmap",
            Self::Ctrl(_) => "/dev/nvhost-ctrl",
            Self::CtrlGpu(_) => "/dev/nvhost-ctrl-gpu",
            Self::AsGpu(_) => "/dev/nvhost-as-gpu",
            Self::Channel(_) => "/dev/nvhost-gpu",
            Self::Host1x(device) => device.name(),
            Self::Disp => "/dev/nvdisp_disp0",
            Self::Stub(device) => device.name(),
        }
    }

    fn ioctl(&self, files: &[Option<Device>], args: &mut Args) -> NvResult {
        let result = match self {
            Self::NvMap(device) => device.ioctl(args),
            Self::Ctrl(device) => device.ioctl(args),
            Self::CtrlGpu(device) => device.ioctl(args),
            Self::AsGpu(device) => device.ioctl(files, args),
            Self::Channel(device) => device.ioctl(args),
            Self::Host1x(device) => device.ioctl(args),
            Self::Disp => None,
            Self::Stub(device) => Some(device.ioctl(args)),
        };
        result.unwrap_or_else(|| {
            tracing::warn!(
                device = self.name(),
                ioctl = format_args!("{:#010x}", args.ioctl.0),
                variant = ?args.variant,
                "unimplemented ioctl"
            );
            NvResult::NotImplemented
        })
    }

    fn query_event(&self, id: u32) -> Option<Arc<Event>> {
        match self {
            Self::Ctrl(device) => device.query_event(id),
            Self::CtrlGpu(device) => device.query_event(id),
            Self::Channel(device) => device.query_event(id),
            _ => None,
        }
    }
}

/// The device open as `fd` in `files`.
fn file(files: &[Option<Device>], fd: i32) -> Option<&Device> {
    usize::try_from(fd)
        .ok()
        .and_then(|fd| files.get(fd)?.as_ref())
}

struct SessionState {
    client: Option<Client>,
    /// Open devices by descriptor; descriptor 0 is never used.
    files: Vec<Option<Device>>,
    buffers: Buffers,
}

/// One INvDrvServices session.
struct NvdrvService {
    nvdrv: Arc<Nvdrv>,
    name: &'static str,
    state: Mutex<SessionState>,
}

impl NvdrvService {
    /// A session; `process` initializes it at once (`nx2:nvsrv`).
    fn new(nvdrv: Arc<Nvdrv>, name: &'static str, process: Option<Arc<Process>>) -> Self {
        let client = process.map(|process| Client {
            process,
            session: nvdrv.next_session.fetch_add(1, Ordering::Relaxed),
        });
        Self {
            nvdrv,
            name,
            state: Mutex::new(SessionState {
                client,
                files: vec![None],
                buffers: Buffers::default(),
            }),
        }
    }

    fn open(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let mut state = self.state.lock();
        let Some(client) = &state.client else {
            tracing::error!(service = self.name, "Open before Initialize");
            ctx.push(0i32);
            return reply(ctx, NvResult::NotInitialized);
        };
        let mut raw = [0u8; 0x80];
        if let Some(buffer) = ctx.readable_buffer(0) {
            let len = (buffer.size as usize).min(raw.len());
            if ctx
                .memory()
                .read_bytes(buffer.address, &mut raw[..len])
                .is_err()
            {
                return lichii_kernel::result::INVALID_CURRENT_MEMORY;
            }
        }
        let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
        let path = std::str::from_utf8(&raw[..end]).unwrap_or_default();
        if path == "/dev/nvhost-prof-gpu" && !self.nvdrv.nx2 {
            tracing::warn!("/dev/nvhost-prof-gpu cannot be opened in production");
            ctx.push(0i32);
            return reply(ctx, NvResult::NotSupported);
        }
        let Some(device) = Device::open(&self.nvdrv, client, path) else {
            tracing::error!(path, "opening an unknown device");
            ctx.push(-1i32);
            return reply(ctx, NvResult::FileOperationFailed);
        };
        if let Device::Stub(_) = device {
            tracing::warn!(path, "opening an unknown nvhost device as a logged stub");
        }
        let files = &mut state.files;
        let fd = match files.iter().skip(1).position(Option::is_none) {
            Some(free) => free + 1,
            None => {
                files.push(None);
                files.len() - 1
            }
        };
        files[fd] = Some(device);
        tracing::debug!(path, fd, "opened");
        ctx.push(fd as i32);
        reply(ctx, NvResult::Success)
    }

    fn ioctl(&self, ctx: &mut HleRequestContext, variant: Variant) -> ResultCode {
        let fd: i32 = ctx.pop();
        let ioctl = Ioctl(ctx.pop());
        let mut state = self.state.lock();
        let SessionState {
            client,
            files,
            buffers,
        } = &mut *state;
        if client.is_none() {
            return reply(ctx, NvResult::NotInitialized);
        }
        if fd < 0 {
            return reply(ctx, NvResult::InvalidState);
        }
        let Some(device) = file(files, fd) else {
            tracing::error!(fd, "ioctl on a descriptor that is not open");
            return reply(ctx, NvResult::NotImplemented);
        };
        if let Err(result) = buffers.load(ctx, variant) {
            return result;
        }
        let result = device.ioctl(files, &mut buffers.args(variant, ioctl));
        if self.nvdrv.ipc_trace {
            tracing::info!(
                target: "ipc",
                device = device.name(),
                fd,
                ioctl = format_args!("{:#010x}", ioctl.0),
                result = ?result,
            );
        }
        if ioctl.is_out()
            && let Err(error) = buffers.store(ctx, variant)
        {
            return error;
        }
        reply(ctx, result)
    }

    fn close(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let mut state = self.state.lock();
        if state.client.is_none() {
            return reply(ctx, NvResult::NotInitialized);
        }
        if fd < 0 {
            return reply(ctx, NvResult::InvalidState);
        }
        let Some(slot) = state
            .files
            .get_mut(fd as usize)
            .filter(|slot| slot.is_some())
        else {
            return reply(ctx, NvResult::NotImplemented);
        };
        let device = slot.take();
        drop(state);
        drop(device);
        reply(ctx, NvResult::Success)
    }

    fn initialize(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let mut state = self.state.lock();
        if state.client.is_none() {
            let process = ctx
                .copy_handle::<Process>(0)
                .unwrap_or_else(|_| ctx.process().clone());
            state.client = Some(Client {
                process,
                session: self.nvdrv.next_session.fetch_add(1, Ordering::Relaxed),
            });
        }
        reply(ctx, NvResult::Success)
    }

    fn query_event(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let id: u32 = ctx.pop();
        let state = self.state.lock();
        if state.client.is_none() {
            return reply(ctx, NvResult::NotInitialized);
        }
        if fd < 0 {
            return reply(ctx, NvResult::InvalidState);
        }
        let Some(device) = file(&state.files, fd) else {
            return reply(ctx, NvResult::NotImplemented);
        };
        let Some(event) = device.query_event(id) else {
            tracing::error!(device = device.name(), id, "unknown event");
            return reply(ctx, NvResult::BadParameter);
        };
        ctx.push_copy_object(KObject::ReadableEvent(event));
        reply(ctx, NvResult::Success)
    }
}

fn reply(ctx: &mut HleRequestContext, result: NvResult) -> ResultCode {
    ctx.push(result as u32);
    ResultCode::SUCCESS
}

impl Service for NvdrvService {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => self.open(ctx),
            1 => self.ioctl(ctx, Variant::One),
            2 => self.close(ctx),
            3 => self.initialize(ctx),
            4 => self.query_event(ctx),
            5 => {
                tracing::warn!("(stubbed) MapSharedMem");
                reply(ctx, NvResult::Success)
            }
            6 => reply(ctx, NvResult::Success),
            7 | 8 => {
                let aruid: u64 = ctx.pop();
                tracing::debug!(aruid, "SetAruid");
                reply(ctx, NvResult::Success)
            }
            9 | 13 => ResultCode::SUCCESS,
            11 => self.ioctl(ctx, Variant::Two),
            12 => self.ioctl(ctx, Variant::Three),
            _ => return None,
        })
    }
}

impl Drop for NvdrvService {
    fn drop(&mut self) {
        let state = self.state.get_mut();
        state.files.clear();
        if let Some(client) = &state.client {
            self.nvdrv.nvmap.release_session(client.session);
        }
    }
}
