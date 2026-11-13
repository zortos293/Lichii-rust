//! Switch 1 display services: `vi:u`, `vi:s` and `vi:m` (displays and layers), `dispdrv` (the
//! HOS binder that carries each layer's IGraphicBufferProducer) and the compositor that shows
//! the frames titles queue.
//!
//! am creates the application's layer ([`create_managed_layer`]), the title opens it through
//! vi and gets the binder id of its buffer queue in a native window parcel. A host thread ticks
//! at the display's vblank: it signals the vsync events, takes the next frame of the top-most
//! visible layer and flips it through `/dev/nvdisp_disp0`. `PerfStats::record_guest_frame` is
//! called once per QueueBuffer, at the rate the swap interval asks for.

mod buffer_queue;
mod display;
mod parcel;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use lichii_common::perf::PerfStats;
use lichii_common::{HleResult, ResultCode};
use lichii_gpu::backend::{CropRect, FramebufferConfig};
use lichii_kernel::Event;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;
use parking_lot::{Condvar, Mutex};

use crate::ServiceContext;
use crate::nvdrv::Nvdrv;
use buffer_queue::{BufferQueue, Item, REFRESH_HZ};

const MODULE: u32 = 114;
const OPERATION_FAILED: ResultCode = ResultCode::new(MODULE, 1);
pub(crate) const PERMISSION_DENIED: ResultCode = ResultCode::new(MODULE, 5);
const NOT_SUPPORTED: ResultCode = ResultCode::new(MODULE, 6);
const NOT_FOUND: ResultCode = ResultCode::new(MODULE, 7);

/// The displays by id. Only "Default" is shown.
const DISPLAYS: [&str; 5] = ["Default", "External", "Edid", "Internal", "Null"];
const DEFAULT_DISPLAY: u64 = 0;
/// Android's `NATIVE_WINDOW_TRANSFORM_FLIP_V`.
const TRANSFORM_FLIP_V: u32 = 0x2;
/// Switch 1 scanout buffers are block linear with 16-GOB blocks.
const BLOCK_HEIGHT_LOG2: u32 = 4;

/// Registers the vi services and the binder. Needs nvdrv, which the compositor flips through.
pub(crate) fn register(context: &Arc<ServiceContext>, nvdrv: Arc<Nvdrv>) -> HleResult<()> {
    let vi = Vi::new(context, nvdrv);
    let _ = context.vi.set(vi.clone());
    for (name, command) in [("vi:u", 0), ("vi:s", 1), ("vi:m", 2)] {
        let vi = vi.clone();
        context.register(name, "vi", move |context| display::Root {
            context: context.clone(),
            vi: vi.clone(),
            name,
            command,
        })?;
    }
    context.register("dispdrv", "vi", move |_| display::Binder { vi: vi.clone() })
}

/// Creates the application's layer on the Default display, owned by applet resource user
/// `aruid` (am's CreateManagedDisplayLayer).
pub(crate) fn create_managed_layer(context: &ServiceContext, aruid: u64) -> HleResult<u64> {
    let vi = context.vi.get().ok_or(OPERATION_FAILED)?;
    Ok(vi.create_layer(DEFAULT_DISPLAY, aruid))
}

/// Wakes the compositor when a frame is queued, so an unlimited speed shows it at once.
#[derive(Default)]
pub(super) struct Wakeup {
    queued: Mutex<bool>,
    condvar: Condvar,
}

impl Wakeup {
    fn notify(&self) {
        *self.queued.lock() = true;
        self.condvar.notify_one();
    }

    /// Waits until `deadline`, or until a frame is queued when `early` is set. Returns whether a
    /// frame was queued.
    fn wait(&self, deadline: Instant, early: bool) -> bool {
        let mut queued = self.queued.lock();
        while !(early && *queued) {
            if self.condvar.wait_until(&mut queued, deadline).timed_out() {
                break;
            }
        }
        std::mem::take(&mut *queued)
    }
}

struct Layer {
    display: u64,
    owner: u64,
    binder: i32,
    open: bool,
    visible: bool,
    z: i64,
}

#[derive(Default)]
struct State {
    layers: BTreeMap<u64, Layer>,
    binders: BTreeMap<i32, Arc<BufferQueue>>,
    next_layer: u64,
    next_binder: i32,
    vsync: Vec<(u64, Weak<Event>)>,
}

/// What every vi session shares: the layers, their buffer queues and the vsync events.
pub(crate) struct Vi {
    nvdrv: Arc<Nvdrv>,
    perf: Arc<PerfStats>,
    docked: bool,
    /// The time between vblanks; `None` at an unlimited speed, where frames show as they come.
    vblank: Option<Duration>,
    state: Mutex<State>,
    queued: Arc<Wakeup>,
}

impl Vi {
    fn new(context: &ServiceContext, nvdrv: Arc<Nvdrv>) -> Arc<Self> {
        let speed = context
            .settings
            .graphics
            .speed_limit
            .filter(|&speed| speed != 0)
            .map(u64::from);
        let vi = Arc::new(Self {
            nvdrv,
            perf: context.perf.clone(),
            docked: context.settings.system.docked,
            vblank: speed
                .map(|speed| Duration::from_nanos(100_000_000_000 / u64::from(REFRESH_HZ) / speed)),
            state: Mutex::new(State {
                next_layer: 1,
                next_binder: 1,
                ..State::default()
            }),
            queued: Arc::default(),
        });
        spawn_compositor(&vi);
        vi
    }

    fn display_id(name: &[u8]) -> Option<u64> {
        let end = name.iter().position(|&b| b == 0).unwrap_or(name.len());
        DISPLAYS
            .iter()
            .position(|display| display.as_bytes() == &name[..end])
            .map(|id| id as u64)
    }

    fn create_layer(&self, display: u64, owner: u64) -> u64 {
        let queue = Arc::new(BufferQueue::new(
            self.nvdrv.clone(),
            self.perf.clone(),
            self.queued.clone(),
        ));
        let mut state = self.state.lock();
        let (id, binder) = (state.next_layer, state.next_binder);
        state.next_layer += 1;
        state.next_binder += 1;
        state.binders.insert(binder, queue);
        state.layers.insert(
            id,
            Layer {
                display,
                owner,
                binder,
                open: false,
                visible: true,
                z: 0,
            },
        );
        id
    }

    fn destroy_layer(&self, id: u64) -> ResultCode {
        let mut state = self.state.lock();
        let Some(layer) = state.layers.remove(&id) else {
            return NOT_FOUND;
        };
        state.binders.remove(&layer.binder);
        ResultCode::SUCCESS
    }

    /// Opens layer `id` for `aruid` and returns its producer's binder id.
    fn open_layer(&self, id: u64, aruid: u64) -> Result<i32, ResultCode> {
        let mut state = self.state.lock();
        let layer = state.layers.get_mut(&id).ok_or(NOT_FOUND)?;
        if layer.open {
            return Err(OPERATION_FAILED);
        }
        if layer.owner != aruid {
            return Err(PERMISSION_DENIED);
        }
        layer.open = true;
        Ok(layer.binder)
    }

    fn close_layer(&self, id: u64) -> ResultCode {
        match self.state.lock().layers.get_mut(&id) {
            Some(layer) if layer.open => {
                layer.open = false;
                ResultCode::SUCCESS
            }
            Some(_) => OPERATION_FAILED,
            None => NOT_FOUND,
        }
    }

    fn with_layer<T>(&self, id: u64, f: impl FnOnce(&mut Layer) -> T) -> Result<T, ResultCode> {
        self.state
            .lock()
            .layers
            .get_mut(&id)
            .map(f)
            .ok_or(NOT_FOUND)
    }

    fn binder(&self, id: i32) -> Option<Arc<BufferQueue>> {
        self.state.lock().binders.get(&id).cloned()
    }

    fn link_vsync(&self, display: u64, event: &Arc<Event>) {
        let mut state = self.state.lock();
        state.vsync.retain(|(_, event)| event.strong_count() > 0);
        state.vsync.push((display, Arc::downgrade(event)));
    }

    /// The display's size: the TV's when docked, else the panel's.
    fn resolution(&self) -> (u32, u32) {
        if self.docked {
            (1920, 1080)
        } else {
            (1280, 720)
        }
    }

    fn vblank_tick(&self) {
        let state = self.state.lock();
        for (_, event) in &state.vsync {
            if let Some(event) = event.upgrade() {
                event.signal();
            }
        }
    }

    /// Shows the next frame of the top-most visible layer of the Default display that has a
    /// frame. While that layer holds its frame (swap interval) or has none queued, the screen
    /// keeps it; the layers below are drained without being shown. With `latest`, layers skip to
    /// their newest frame and ignore swap intervals.
    fn compose(&self, frame: u64, latest: bool) {
        let mut layers: Vec<(i64, Arc<BufferQueue>)> = {
            let state = self.state.lock();
            state
                .layers
                .values()
                .filter(|layer| layer.open && layer.visible && layer.display == DEFAULT_DISPLAY)
                .filter_map(|layer| Some((layer.z, state.binders.get(&layer.binder)?.clone())))
                .collect()
        };
        layers.sort_by_key(|(z, _)| std::cmp::Reverse(*z));
        let top = layers.iter().position(|(_, queue)| queue.has_frames());
        for (index, (_, queue)) in layers.iter().enumerate() {
            let Some(item) = queue.acquire(frame, latest) else {
                continue;
            };
            if Some(index) == top {
                let queue = queue.clone();
                self.flip(&item, move || queue.release(&item));
            } else {
                queue.release(&item);
            }
        }
    }

    /// Shows `item` once its fence is reached; `presented` gives its buffer back after that.
    fn flip(&self, item: &Item, presented: impl FnOnce() + Send + 'static) {
        let buffer = &item.buffer;
        let format = match buffer.external_format {
            4 => PixelFormat::R5G6B5Unorm,
            5 => PixelFormat::B8G8R8A8Unorm,
            22 => PixelFormat::R16G16B16A16Float,
            43 => PixelFormat::A2B10G10R10Unorm,
            _ => PixelFormat::A8B8G8R8Unorm,
        };
        let (width, height) = (buffer.width.max(0) as u32, buffer.height.max(0) as u32);
        let stride = (buffer.stride.max(0) as u32).max(width);
        let [left, top, right, bottom] = item.crop.map(|v| v.max(0) as u32);
        let crop = if right > left && bottom > top {
            CropRect {
                left,
                top,
                right: right.min(width),
                bottom: bottom.min(height),
            }
        } else {
            CropRect {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            }
        };
        let framebuffer = FramebufferConfig {
            address: 0,
            width: stride,
            height,
            stride: stride * format.bytes_per_block(),
            format,
            block_height_log2: Some(BLOCK_HEIGHT_LOG2),
            gob_layout: GobLayout::Tegra,
            flip_vertical: item.transform & TRANSFORM_FLIP_V != 0,
            crop: Some(crop),
        };
        self.nvdrv.flip(
            buffer.buffer_id,
            u64::from(buffer.offset),
            framebuffer,
            item.fence.active(),
            presented,
        );
    }
}

/// Starts the host thread that ticks at the vblank rate while `vi` lives.
fn spawn_compositor(vi: &Arc<Vi>) {
    let weak = Arc::downgrade(vi);
    let queued = vi.queued.clone();
    let vblank = vi.vblank;
    let spawned = std::thread::Builder::new()
        .name("vi compositor".into())
        .spawn(move || {
            let period =
                vblank.unwrap_or(Duration::from_nanos(1_000_000_000 / u64::from(REFRESH_HZ)));
            let mut next = Instant::now() + period;
            let mut frame = 0u64;
            loop {
                let woken = queued.wait(next, vblank.is_none());
                let Some(vi) = weak.upgrade() else { return };
                let now = Instant::now();
                if now >= next {
                    next += period;
                    if next < now {
                        next = now + period;
                    }
                    frame += 1;
                    vi.vblank_tick();
                    vi.compose(frame, vblank.is_none());
                } else if woken {
                    vi.compose(frame, true);
                }
            }
        });
    if let Err(err) = spawned {
        tracing::error!("spawning the vi compositor: {err}");
    }
}
