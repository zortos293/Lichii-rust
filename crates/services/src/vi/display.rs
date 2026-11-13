//! The vi root services, IApplicationDisplayService with its system and manager extensions, and
//! IHOSBinderDriver.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use super::parcel::ParcelWriter;
use super::{DEFAULT_DISPLAY, DISPLAYS, NOT_FOUND, NOT_SUPPORTED, OPERATION_FAILED, Vi};
use super::{PERMISSION_DENIED, REFRESH_HZ};
use crate::{Service, ServiceContext};

type DisplayName = [u8; 0x40];

/// `vi:u`, `vi:s` and `vi:m`: GetDisplayService is command 0, 1 or 2 by permission.
pub(super) struct Root {
    pub context: Arc<ServiceContext>,
    pub vi: Arc<Vi>,
    pub name: &'static str,
    pub command: u32,
}

impl Service for Root {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != self.command {
            return None;
        }
        let _policy: u32 = ctx.pop();
        push_service(
            &self.context,
            ctx,
            ApplicationDisplay {
                context: self.context.clone(),
                vi: self.vi.clone(),
                session: Mutex::default(),
            },
        )
    }
}

fn push_service<S: Service>(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    service: S,
) -> Option<ResultCode> {
    ctx.push_interface(context.handler(service));
    Some(ResultCode::SUCCESS)
}

#[derive(Default)]
struct Session {
    vsync: BTreeMap<u64, Arc<Event>>,
    open: BTreeSet<u64>,
    stray: BTreeSet<u64>,
}

/// `DisplayInfo`.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct DisplayInfo {
    name: DisplayName,
    has_limited_layers: u8,
    _pad: [u8; 7],
    max_layers: u64,
    width: u64,
    height: u64,
}

/// `DisplayMode`.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct DisplayMode {
    width: u32,
    height: u32,
    refresh_rate: f32,
    _unknown: u32,
}

/// The native window parcel OpenLayer and CreateStrayLayer return: the producer's binder id
/// on `dispdrv`.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct NativeWindow {
    magic: u32,
    process_id: u32,
    id: u64,
    _pad0: [u32; 2],
    dispdrv: [u8; 8],
    _pad1: [u32; 2],
}

fn native_window(binder: i32) -> Vec<u8> {
    let mut parcel = ParcelWriter::default();
    parcel.write_interface(NativeWindow {
        magic: 2,
        process_id: 1,
        id: binder as u64,
        _pad0: [0; 2],
        dispdrv: *b"dispdrv\0",
        _pad1: [0; 2],
    });
    parcel.serialize()
}

pub(super) struct ApplicationDisplay {
    context: Arc<ServiceContext>,
    vi: Arc<Vi>,
    session: Mutex<Session>,
}

impl Drop for ApplicationDisplay {
    fn drop(&mut self) {
        let session = self.session.get_mut();
        for &layer in &session.open {
            self.vi.close_layer(layer);
        }
        for &layer in &session.stray {
            self.vi.destroy_layer(layer);
        }
    }
}

impl ApplicationDisplay {
    fn open_display(ctx: &mut HleRequestContext) -> ResultCode {
        let name: DisplayName = ctx.pop();
        match Vi::display_id(&name) {
            Some(id) => {
                ctx.push(id);
                ResultCode::SUCCESS
            }
            None => OPERATION_FAILED,
        }
    }

    /// 2020(DisplayName, u64 layer, u64 aruid, pid) -> u64 size, B native window.
    fn open_layer(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let name: DisplayName = ctx.pop();
        let (layer, aruid): (u64, u64) = (ctx.pop(), ctx.pop());
        if Vi::display_id(&name).is_none() {
            return NOT_FOUND;
        }
        let binder = match self.vi.open_layer(layer, aruid) {
            Ok(binder) => binder,
            Err(result) => return result,
        };
        self.session.lock().open.insert(layer);
        let parcel = native_window(binder);
        if let Err(result) = ctx.write_buffer(0, &parcel) {
            return result;
        }
        ctx.push(parcel.len() as u64);
        ResultCode::SUCCESS
    }

    /// 2030(u32 flags, u64 display) -> u64 layer, u64 size, B native window.
    fn create_stray_layer(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let _flags: u32 = ctx.pop();
        let display: u64 = ctx.pop();
        if display as usize >= DISPLAYS.len() {
            return NOT_FOUND;
        }
        let layer = self.vi.create_layer(display, 0);
        let binder = match self.vi.open_layer(layer, 0) {
            Ok(binder) => binder,
            Err(result) => return result,
        };
        self.session.lock().stray.insert(layer);
        let parcel = native_window(binder);
        if let Err(result) = ctx.write_buffer(0, &parcel) {
            return result;
        }
        ctx.push(layer);
        ctx.push(parcel.len() as u64);
        ResultCode::SUCCESS
    }

    fn vsync_event(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let display: u64 = ctx.pop();
        let mut session = self.session.lock();
        if session.vsync.contains_key(&display) {
            return PERMISSION_DENIED;
        }
        let event = Arc::new(Event::new());
        self.vi.link_vsync(display, &event);
        session.vsync.insert(display, event.clone());
        ctx.push_copy_object(KObject::ReadableEvent(event));
        ResultCode::SUCCESS
    }
}

impl Service for ApplicationDisplay {
    fn name(&self) -> &str {
        "IApplicationDisplayService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            100 | 103 => {
                let binder = Binder {
                    vi: self.vi.clone(),
                };
                return push_service(&self.context, ctx, binder);
            }
            101 => {
                let system = SystemDisplay {
                    vi: self.vi.clone(),
                };
                return push_service(&self.context, ctx, system);
            }
            102 => {
                let manager = ManagerDisplay {
                    vi: self.vi.clone(),
                };
                return push_service(&self.context, ctx, manager);
            }
            1000 => {
                let mut name = [0u8; 0x40];
                name[..7].copy_from_slice(b"Default");
                let info = DisplayInfo {
                    name,
                    has_limited_layers: 1,
                    _pad: [0; 7],
                    max_layers: 1,
                    width: 1920,
                    height: 1080,
                };
                let written = ctx.write_buffer(0, bytemuck::bytes_of(&info)).unwrap_or(0);
                ctx.push(u64::from(written >= std::mem::size_of::<DisplayInfo>()));
                ResultCode::SUCCESS
            }
            1010 => Self::open_display(ctx),
            1011 => {
                ctx.push(DEFAULT_DISPLAY);
                ResultCode::SUCCESS
            }
            1020 | 1101 => ResultCode::SUCCESS,
            1102 => {
                ctx.push([1280i64, 720]);
                ResultCode::SUCCESS
            }
            2020 => self.open_layer(ctx),
            2021 => {
                let layer: u64 = ctx.pop();
                if !self.session.lock().open.remove(&layer) {
                    return Some(NOT_FOUND);
                }
                self.vi.close_layer(layer)
            }
            2030 => self.create_stray_layer(ctx),
            2031 => {
                let layer: u64 = ctx.pop();
                if !self.session.lock().stray.remove(&layer) {
                    return Some(NOT_FOUND);
                }
                self.vi.close_layer(layer);
                self.vi.destroy_layer(layer)
            }
            2101 => match ctx.pop::<u32>() {
                2 | 4 => ResultCode::SUCCESS,
                0..=3 => NOT_SUPPORTED,
                _ => OPERATION_FAILED,
            },
            2102 => {
                let converted: u64 = match ctx.pop::<u32>() {
                    0 => 3,
                    1 => 0,
                    2 => 1,
                    3 => 2,
                    4 => 4,
                    _ => return Some(OPERATION_FAILED),
                };
                ctx.push(converted);
                ResultCode::SUCCESS
            }
            2103 => {
                ctx.push([0u64; 3]);
                ResultCode::SUCCESS
            }
            2450 => {
                ctx.push([0u64; 2]);
                ResultCode::SUCCESS
            }
            2460 => {
                let (width, height): (i64, i64) = (ctx.pop(), ctx.pop());
                const BLOCK: i64 = 0x20000;
                let size = (width * height * 4 + BLOCK - 1) / BLOCK * BLOCK;
                ctx.push([size, 0x1000]);
                ResultCode::SUCCESS
            }
            5202 => self.vsync_event(ctx),
            _ => return None,
        })
    }
}

pub(super) struct SystemDisplay {
    vi: Arc<Vi>,
}

impl SystemDisplay {
    fn mode(&self) -> DisplayMode {
        let (width, height) = self.vi.resolution();
        DisplayMode {
            width,
            height,
            refresh_rate: REFRESH_HZ as f32,
            _unknown: 0,
        }
    }
}

impl Service for SystemDisplay {
    fn name(&self) -> &str {
        "ISystemDisplayService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            2204 => {
                let layer: u64 = ctx.pop();
                match self.vi.with_layer(layer, |layer| layer.z) {
                    Ok(z) => {
                        ctx.push(z);
                        ResultCode::SUCCESS
                    }
                    Err(result) => result,
                }
            }
            2205 => {
                let (layer, z): (u64, i64) = (ctx.pop(), ctx.pop());
                result(self.vi.with_layer(layer, |layer| layer.z = z))
            }
            2207 => set_visibility(&self.vi, ctx),
            3000 => {
                let mode = self.mode();
                let written = ctx.write_buffer(0, bytemuck::bytes_of(&mode)).unwrap_or(0);
                ctx.push(u64::from(written >= std::mem::size_of::<DisplayMode>()));
                ResultCode::SUCCESS
            }
            3200 => {
                ctx.push(self.mode());
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

pub(super) struct ManagerDisplay {
    vi: Arc<Vi>,
}

impl Service for ManagerDisplay {
    fn name(&self) -> &str {
        "IManagerDisplayService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            2010 => {
                let _flags: u32 = ctx.pop();
                let (display, aruid): (u64, u64) = (ctx.pop(), ctx.pop());
                if display as usize >= DISPLAYS.len() {
                    return Some(NOT_FOUND);
                }
                ctx.push(self.vi.create_layer(display, aruid));
                ResultCode::SUCCESS
            }
            2011 => {
                let layer: u64 = ctx.pop();
                self.vi.close_layer(layer);
                self.vi.destroy_layer(layer)
            }
            6000 => ResultCode::SUCCESS,
            6002 => set_visibility(&self.vi, ctx),
            _ => return None,
        })
    }
}

/// (bool visible, u64 layer).
fn set_visibility(vi: &Vi, ctx: &mut HleRequestContext) -> ResultCode {
    let visible = ctx.pop::<u8>() != 0;
    let layer: u64 = ctx.pop();
    result(vi.with_layer(layer, |layer| layer.visible = visible))
}

fn result(outcome: Result<(), ResultCode>) -> ResultCode {
    outcome.err().unwrap_or(ResultCode::SUCCESS)
}

/// IHOSBinderDriver (`dispdrv`, and vi's relay service).
pub(super) struct Binder {
    pub vi: Arc<Vi>,
}

impl Binder {
    /// 0/3(i32 binder, u32 code, A/X parcel, B/C reply, u32 flags).
    fn transact(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let (binder, code): (i32, u32) = (ctx.pop(), ctx.pop());
        let Some(queue) = self.vi.binder(binder) else {
            tracing::warn!("vi: transaction {code} on unknown binder {binder}");
            return ResultCode::SUCCESS;
        };
        let input = match ctx.read_buffer(0) {
            Ok(input) => input,
            Err(result) => return result,
        };
        let Some(reply) = queue.transact(code, &input, ctx) else {
            return ResultCode::SUCCESS;
        };
        match ctx.write_buffer(0, &reply) {
            Ok(_) => ResultCode::SUCCESS,
            Err(result) => result,
        }
    }
}

impl Service for Binder {
    fn name(&self) -> &str {
        "IHOSBinderDriver"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 3 => self.transact(ctx),
            1 => {
                let (_binder, _add, kind): (i32, i32, i32) = (ctx.pop(), ctx.pop(), ctx.pop());
                if kind == 0 || kind == 1 {
                    ResultCode::SUCCESS
                } else {
                    OPERATION_FAILED
                }
            }
            2 => {
                let binder: i32 = ctx.pop();
                let Some(queue) = self.vi.binder(binder) else {
                    return Some(OPERATION_FAILED);
                };
                ctx.push_copy_object(KObject::ReadableEvent(queue.native_handle().clone()));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
