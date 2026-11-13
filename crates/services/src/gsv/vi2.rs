//! `nn::vi2` (gsv 305): the Switch 2 display session. Titles present their frames with 229, which
//! shows a native image, waits for the GPU and paces the title to its swap interval, and start
//! frames on native events (31/32) that fire with the 60 Hz vblank.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_gpu::Gpu;
use lichii_gpu::backend::{CropRect, FramebufferConfig};
use lichii_gpu::gmmu::{BigPageSize, Gmmu};
use lichii_gpu::host1x::SyncpointWaitError;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;
use parking_lot::Mutex;

use super::native::{FORMAT_RGB10A2, FORMAT_RGBA16_FLOAT, NativeObjects, bytes_per_pixel};
use super::{UNKNOWN, Words, handshake, push_readable};
use crate::{Service, ServiceContext};

const NATIVE_EVENTS: usize = 8;
const VBLANK: Duration = Duration::from_nanos(1_000_000_000 / 60);
/// Buffer slots 228/229/212 report the client may use: all 15.
const AVAILABLE_SLOTS: u64 = 0x7fff;
/// "default", the display every query names.
const DEFAULT_DISPLAY: u64 = u64::from_le_bytes(*b"default\0");
/// How long a present waits for the previous frame before going on.
const FENCE_TIMEOUT: Duration = Duration::from_secs(4);
/// The handheld panel: handheld titles draw into its top-left corner of their docked images.
const HANDHELD: (u32, u32) = (1920, 1080);

pub(super) struct Vi2 {
    context: Arc<ServiceContext>,
    objects: Arc<Mutex<NativeObjects>>,
    native: Arc<NativeEvents>,
    display: Option<DisplaySpace>,
    presents: Mutex<Presents>,
}

/// The session's native event slots. A host thread fires the ones bound to vsync every vblank
/// for as long as the session lives.
#[derive(Default)]
struct NativeEvents {
    slots: Mutex<[Option<Arc<Event>>; NATIVE_EVENTS]>,
    vsync: AtomicU8,
}

impl NativeEvents {
    fn vblank(&self) {
        let vsync = self.vsync.load(Ordering::Acquire);
        let slots = self.slots.lock();
        for (index, event) in slots.iter().enumerate() {
            if vsync & (1 << index) != 0
                && let Some(event) = event
            {
                event.signal();
            }
        }
    }

    fn set_vsync(&self, index: usize, on: bool) {
        if on {
            self.vsync.fetch_or(1 << index, Ordering::AcqRel);
        } else {
            self.vsync.fetch_and(!(1 << index), Ordering::AcqRel);
        }
    }
}

fn spawn_vsync(native: &Arc<NativeEvents>) {
    let weak: Weak<NativeEvents> = Arc::downgrade(native);
    let spawned = std::thread::Builder::new()
        .name("nx2:vi2 vsync".into())
        .spawn(move || {
            let mut next = Instant::now();
            loop {
                next += VBLANK;
                match next.checked_duration_since(Instant::now()) {
                    Some(wait) => std::thread::sleep(wait),
                    None => next = Instant::now(),
                }
                let Some(native) = weak.upgrade() else { return };
                native.vblank();
            }
        });
    if let Err(err) = spawned {
        tracing::error!("spawning the nx2:vi2 vsync thread: {err}");
    }
}

/// An address space that receives the Switch 2 GPU memory mappings, through which the display
/// finds the device memory of a presented image.
struct DisplaySpace {
    gpu: Weak<Gpu>,
    gmmu: Arc<Gmmu>,
}

impl Drop for DisplaySpace {
    fn drop(&mut self) {
        if let Some(gpu) = self.gpu.upgrade() {
            gpu.destroy_address_space(&self.gmmu);
        }
    }
}

#[derive(Default)]
struct Presents {
    count: u64,
    /// The fence of the frame presented last: the next present waits for it.
    previous_fence: Option<(u32, u32)>,
    /// When the next frame is due, while the title is paced.
    deadline: Option<Instant>,
    timeouts: u32,
}

/// Present word 12: the swap interval in its high half, the refresh rate in its low half.
struct Timing {
    swap_interval: u32,
    refresh_hz: u32,
}

impl Timing {
    fn of(word: u32) -> Self {
        let refresh = word & 0xffff;
        Self {
            swap_interval: (word >> 16).clamp(1, 4),
            refresh_hz: if (30..=240).contains(&refresh) {
                refresh
            } else {
                60
            },
        }
    }

    fn target_hz(&self) -> f32 {
        self.refresh_hz as f32 / self.swap_interval as f32
    }

    /// The time between frames at `speed` percent, or none when unlimited.
    fn interval(&self, speed: Option<u16>) -> Option<Duration> {
        let speed = u64::from(speed.filter(|&speed| speed != 0)?);
        Some(Duration::from_nanos(
            100_000_000_000 * u64::from(self.swap_interval) / u64::from(self.refresh_hz) / speed,
        ))
    }
}

impl Vi2 {
    pub fn new(context: &Arc<ServiceContext>, objects: Arc<Mutex<NativeObjects>>) -> Self {
        let display = context.gpu.nx2_memory().is_some().then(|| DisplaySpace {
            gpu: Arc::downgrade(&context.gpu),
            gmmu: context.gpu.create_address_space(BigPageSize::Size64K),
        });
        Self {
            context: context.clone(),
            objects,
            native: Arc::default(),
            display,
            presents: Mutex::default(),
        }
    }

    fn native_event(&self, ctx: &mut HleRequestContext, index: u32) -> ResultCode {
        let index = index as usize;
        if index >= NATIVE_EVENTS {
            tracing::error!("nx2:vi2: native event {index} out of range");
            return UNKNOWN;
        }
        let mut slots = self.native.slots.lock();
        let event = match &slots[index] {
            Some(event) => event.clone(),
            None => {
                let event = Arc::new(Event::new());
                slots[index] = Some(event.clone());
                self.native.set_vsync(index, true);
                if slots.iter().flatten().count() == 1 {
                    spawn_vsync(&self.native);
                }
                event
            }
        };
        drop(slots);
        push_readable(ctx, &event)
    }

    /// 261-272(u64 display name, u32 slot): bind a slot to vsync (261), to a change of the
    /// display's state (odd: fires once so the title reads the current state), or unbind it.
    fn bind(&self, ctx: &mut HleRequestContext, command: u32) {
        let _display: u64 = ctx.pop();
        let index = ctx.pop::<u32>() as usize;
        let slots = self.native.slots.lock();
        let Some(Some(event)) = slots.get(index) else {
            return;
        };
        self.native.set_vsync(index, command == 261);
        if command != 261 && command % 2 == 1 {
            event.signal();
        }
    }

    /// 229(u64 client token, u32 slot, u32 image, ..., u32 drawn size at word 7, u32 timing at
    /// word 12, ~fence syncpoint at word 22, fence value at word 24): shows the image once its
    /// fence is reached, waits for the previous frame's fence, so the title runs at
    /// most a frame ahead of the GPU, and paces the title to its swap interval. Waiting for the
    /// previous frame rather than this one cannot deadlock a title whose GPU work waits on a
    /// semaphore its CPU releases only after the present returns.
    fn present(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let words = Words::command(ctx);
        let (image, drawn, timing) = (words.u32(5), words.u32(7), Timing::of(words.u32(12)));
        let fence = (!words.u32(22), words.u32(24));
        let mut presents = self.presents.lock();
        presents.count += 1;
        if presents.count <= 4 || presents.count.is_multiple_of(600) {
            tracing::info!(
                "nx2:vi2 229: present #{} of image {image}, {} Hz / {}",
                presents.count,
                timing.refresh_hz,
                timing.swap_interval
            );
        }
        if let Some((id, value)) = presents.previous_fence.replace(fence) {
            let guest = self.context.gpu.host1x().syncpoints.guest();
            if let Err(err @ SyncpointWaitError::TimedOut { .. }) =
                crate::context::wait_guest_syncpoint(
                    &self.context.kernel,
                    guest,
                    id,
                    value,
                    Some(FENCE_TIMEOUT),
                )
                && presents.timeouts < 16
            {
                presents.timeouts += 1;
                tracing::warn!("nx2:vi2 229: previous frame's fence ({id}, {value:#x}): {err}");
            }
        }
        self.composite(image, drawn, fence);
        self.context
            .perf
            .record_guest_frame(Instant::now(), timing.target_hz());
        let interval = timing.interval(self.context.settings.graphics.speed_limit);
        pace(&mut presents, interval);
        ctx.push([AVAILABLE_SLOTS, 0, 0]);
        ResultCode::SUCCESS
    }

    /// Presents native image `handle` once `fence`, this frame's, is reached: the frame's GPU
    /// work may still wait on a semaphore the title releases after the present returns.
    fn composite(&self, handle: u32, drawn: u32, (id, value): (u32, u32)) {
        let Some(framebuffer) = self.framebuffer(handle, drawn) else {
            return;
        };
        let guest = self.context.gpu.host1x().syncpoints.guest();
        if guest.value(id).is_err() {
            self.context.gpu.present(framebuffer);
            return;
        }
        let gpu: Weak<Gpu> = Arc::downgrade(&self.context.gpu);
        let _ = guest.register_action(id, value, move || {
            if let Some(gpu) = gpu.upgrade() {
                gpu.present(framebuffer);
            }
        });
    }

    /// How native image `handle` is shown. `drawn` is ((height - 1) << 16) | (width - 1) of the
    /// area the title drew into from the top-left corner, or 0.
    pub(super) fn framebuffer(&self, handle: u32, drawn: u32) -> Option<FramebufferConfig> {
        let Some(image) = self.objects.lock().presentable(handle) else {
            tracing::warn!("nx2:vi2 229: native image {handle} has no bound memory");
            return None;
        };
        let address = image
            .parent
            .and_then(|(parent, offset)| {
                let memory = self.context.gpu.nx2_memory()?.memory_info(parent)?;
                Some(memory.daddr + offset)
            })
            .or_else(|| {
                let display = self.display.as_ref()?;
                display.gmmu.translate(image.address)
            });
        let Some(address) = address else {
            tracing::warn!(
                "nx2:vi2 229: native image {handle} at {:#x} is not GPU memory",
                image.address
            );
            return None;
        };
        let layout = image.layout;
        let (drawn_width, drawn_height) = ((drawn & 0xffff) + 1, (drawn >> 16) + 1);
        let (mut width, mut height) =
            if drawn != 0 && drawn_width <= layout.width && drawn_height <= layout.height {
                (drawn_width, drawn_height)
            } else {
                (layout.width, layout.height)
            };
        if !self.context.settings.system.docked {
            width = width.min(HANDHELD.0);
            height = height.min(HANDHELD.1);
        }
        let format = match image.format {
            FORMAT_RGBA16_FLOAT => PixelFormat::R16G16B16A16Float,
            FORMAT_RGB10A2 => PixelFormat::A2B10G10R10Unorm,
            _ => PixelFormat::A8B8G8R8Unorm,
        };
        Some(FramebufferConfig {
            address,
            width: (layout.pitch / bytes_per_pixel(image.format)) as u32,
            height: layout.height,
            stride: layout.pitch as u32,
            format,
            block_height_log2: Some(layout.block_rows_log2.saturating_sub(3)),
            gob_layout: GobLayout::Switch2,
            flip_vertical: false,
            crop: Some(CropRect {
                left: 0,
                top: 0,
                right: width,
                bottom: height,
            }),
        })
    }
}

/// Sleeps until the next frame is due. A title that fell more than a frame behind starts over
/// from now instead of rushing to catch up.
fn pace(presents: &mut Presents, interval: Option<Duration>) {
    let Some(interval) = interval else {
        presents.deadline = None;
        return;
    };
    let now = Instant::now();
    let deadline = presents
        .deadline
        .map(|last| last + interval)
        .filter(|&next| next + interval >= now)
        .unwrap_or(now);
    presents.deadline = Some(deadline);
    if let Some(wait) = deadline.checked_duration_since(now) {
        std::thread::sleep(wait);
    }
}

impl Service for Vi2 {
    fn name(&self) -> &str {
        "nx2:vi2"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        Some(match command {
            0 => handshake(ctx),
            1 | 3 => {
                ctx.push(1u32);
                ResultCode::SUCCESS
            }
            5 => {
                ctx.push(7u32);
                ResultCode::SUCCESS
            }
            11 => push_readable(ctx, &Arc::new(Event::new())),
            31 | 32 => {
                let index: u32 = ctx.pop();
                self.native_event(ctx, index)
            }
            103 => {
                ctx.push([60u32, 0]);
                ResultCode::SUCCESS
            }
            104 => {
                ctx.push([1u32, 0, 0]);
                ResultCode::SUCCESS
            }
            105 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            106 => {
                let display: u64 = ctx.pop();
                if display != DEFAULT_DISPLAY {
                    ctx.push(0u64);
                    return Some(ResultCode::new(114, 7));
                }
                ctx.push(if self.context.settings.system.docked {
                    4u64
                } else {
                    2
                });
                ResultCode::SUCCESS
            }
            212 => {
                ctx.push(AVAILABLE_SLOTS);
                ResultCode::SUCCESS
            }
            228 => {
                ctx.push([AVAILABLE_SLOTS, 0, 0]);
                ResultCode::SUCCESS
            }
            229 => self.present(ctx),
            261..=272 => {
                self.bind(ctx, command);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}
