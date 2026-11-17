//! Bounds on the host-visible memory one rasterizer holds above the fixed sizes of its rings
//! (ZOR-432): buffers of their own for ring requests too large to share a ring ([`Oversized`]),
//! and readback buffers grown for large downloads ([`Readback`]).

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::device::Device;
use crate::error::Result;
use crate::memory::{Buffer, LiveMemory};
use crate::scheduler::Scheduler;

/// Host-visible bytes one rasterizer holds at most above its rings' sizes and its readbacks'
/// base sizes: its stream and staging rings' [`Oversized`] buffers plus its readbacks' growth.
pub const RASTERIZER_HOST_CAP: u64 = 512 << 20;

/// Readbacks one rasterizer keeps: the buffer cache's, and the texture cache's flush and
/// eviction readbacks.
const RASTERIZER_READBACKS: u64 = 3;

/// The size a [`Readback`] shrinks back to once downloads fit in it again.
pub const READBACK_BASE: u64 = 8 << 20;

/// The largest [`Readback`] buffer kept between downloads (a power of two): a larger download
/// gets a buffer of its own, freed once it was read.
pub const READBACK_LIMIT: u64 = 32 << 20;

/// Downloads in a row that fit in [`READBACK_BASE`] after which a larger [`Readback`] shrinks.
pub const READBACK_DECAY_DOWNLOADS: u32 = 8;

/// In-flight [`Oversized`] bytes each of the rasterizer's stream and staging rings may hold:
/// half of what [`RASTERIZER_HOST_CAP`] leaves after its readbacks' growth.
pub const RING_OVERSIZED_CAP: u64 =
    (RASTERIZER_HOST_CAP - RASTERIZER_READBACKS * (READBACK_LIMIT - READBACK_BASE)) / 2;

/// In-flight [`Oversized`] bytes the presenter's staging ring may hold.
pub const PRESENT_OVERSIZED_CAP: u64 = 64 << 20;

/// Host-visible buffers of their own for ring requests too large to share the ring, each freed
/// once the tick that uses it completed.
///
/// An allocation that would take the in-flight total past `cap` first waits for submitted
/// buffers, oldest first, until it fits or none is left, so one larger than `cap` waits until
/// nothing submitted is in flight. It never waits for the work being recorded: the operation
/// being set up may hold earlier buffers of it that it has not recorded its use of yet, and a
/// wait would submit and free them. Its owner submits that work between operations instead,
/// once [`Oversized::should_submit`]. With operations starting below half the cap, the in-flight
/// total stays within `cap` while one operation allocates at most half of it.
pub(crate) struct Oversized {
    cap: u64,
    bytes: u64,
    buffers: VecDeque<(u64, Buffer)>,
}

impl Oversized {
    pub(crate) fn new(cap: u64) -> Self {
        Self {
            cap,
            bytes: 0,
            buffers: VecDeque::new(),
        }
    }

    /// A `CpuToGpu` buffer of `size` bytes, valid until the work recorded in `scheduler` right
    /// now completes.
    pub(crate) fn alloc(
        &mut self,
        device: &Arc<Device>,
        scheduler: &mut Scheduler,
        size: u64,
        usage: vk::BufferUsageFlags,
        name: &str,
    ) -> Result<&mut Buffer> {
        self.reclaim(scheduler);
        while self.bytes + size > self.cap
            && let Some(&(oldest, _)) = self.buffers.front()
            && oldest < scheduler.tick()
        {
            scheduler.wait(oldest)?;
            self.reclaim(scheduler);
        }
        let buffer = Buffer::new(device, size, usage, MemoryLocation::CpuToGpu, name)?;
        self.bytes += buffer.size();
        self.buffers.push_back((scheduler.tick(), buffer));
        Ok(&mut self.buffers.back_mut().expect("just pushed").1)
    }

    /// Frees the buffers whose work completed.
    pub(crate) fn reclaim(&mut self, scheduler: &mut Scheduler) {
        while let Some((tick, buffer)) = self.buffers.front()
            && scheduler.is_complete(*tick)
        {
            self.bytes -= buffer.size();
            self.buffers.pop_front();
        }
    }

    /// Whether the work being recorded at `tick` holds half the cap or more, so it should be
    /// submitted before the next operation for later allocations to be able to wait for it.
    pub(crate) fn should_submit(&self, tick: u64) -> bool {
        self.bytes >= self.cap / 2
            && self
                .buffers
                .iter()
                .rev()
                .take_while(|(t, _)| *t == tick)
                .map(|(_, buffer)| buffer.size())
                .sum::<u64>()
                >= self.cap / 2
    }

    /// Bytes and count of the buffers not freed yet.
    pub(crate) fn live(&self) -> LiveMemory {
        LiveMemory {
            bytes: self.bytes,
            count: self.buffers.len() as u64,
        }
    }
}

/// A `GpuToCpu` buffer downloads reuse. It grows to the next power of two of the largest
/// download up to [`READBACK_LIMIT`]; a larger download gets a buffer of its exact size, freed
/// by [`Readback::release`]. After [`READBACK_DECAY_DOWNLOADS`] downloads in a row that fit in
/// [`READBACK_BASE`], a larger buffer is freed and the next one sized for the download.
#[derive(Default)]
pub(crate) struct Readback {
    buffer: Option<Buffer>,
    fitting: u32,
    /// The GPU finished every download into the buffer ([`Readback::release`]), so it can be
    /// freed at once instead of retired against the tick being recorded.
    idle: bool,
}

impl Readback {
    /// A buffer of at least `size` bytes for the next download, replacing one that is too
    /// small, above the limit or decayed.
    pub(crate) fn ensure(
        &mut self,
        device: &Arc<Device>,
        scheduler: &mut Scheduler,
        size: u64,
        name: &str,
    ) -> Result<&Buffer> {
        self.fitting = if size <= READBACK_BASE {
            self.fitting.saturating_add(1)
        } else {
            0
        };
        let decayed = self.fitting >= READBACK_DECAY_DOWNLOADS;
        if let Some(old) = self.buffer.take_if(|b| {
            b.size() < size || b.size() > READBACK_LIMIT || decayed && b.size() > READBACK_BASE
        }) && !self.idle
        {
            scheduler.retire(old);
        }
        self.idle = false;
        if self.buffer.is_none() {
            let capacity = if size > READBACK_LIMIT {
                size
            } else {
                size.next_power_of_two()
            };
            self.buffer = Some(Buffer::new(
                device,
                capacity,
                vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuToCpu,
                name,
            )?);
        }
        Ok(self.buffer.as_ref().expect("just ensured"))
    }

    /// Marks the buffer idle once the GPU finished its download, freeing it when it is above
    /// [`READBACK_LIMIT`].
    pub(crate) fn release(&mut self) {
        self.idle = true;
        if self
            .buffer
            .as_ref()
            .is_some_and(|b| b.size() > READBACK_LIMIT)
        {
            self.buffer = None;
        }
    }

    pub(crate) fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Bytes the kept buffer holds.
    #[cfg(test)]
    pub(crate) fn size(&self) -> u64 {
        self.buffer.as_ref().map_or(0, Buffer::size)
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread::JoinHandle;
    use std::time::Duration;

    use lichii_common::settings::GraphicsSettings;

    use super::*;
    use crate::scheduler::SemaphoreWait;
    use crate::{Instance, SurfaceSupport};

    pub(crate) fn device() -> Option<Arc<Device>> {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: {error}");
                None
            }
        }
    }

    /// A timeline semaphore submissions wait on. It holds them all back until started, then
    /// another thread raises it by one every 2 ms, so they complete slowly and in order, as
    /// when the GPU lags behind.
    pub(crate) struct Gate {
        device: Arc<Device>,
        semaphore: vk::Semaphore,
        done: Arc<AtomicBool>,
        thread: Option<JoinHandle<()>>,
        next: u64,
    }

    impl Gate {
        pub(crate) fn new(device: &Arc<Device>) -> Self {
            let mut kind = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
            // SAFETY: valid create info on a live device.
            let semaphore = unsafe { device.raw.create_semaphore(&info, None) }.unwrap();
            Self {
                device: device.clone(),
                semaphore,
                done: Arc::new(AtomicBool::new(false)),
                thread: None,
                next: 0,
            }
        }

        /// Starts letting one submission through every 2 ms; until then the gate holds every
        /// submission back.
        pub(crate) fn start(&mut self) {
            let (device, done, semaphore) =
                (self.device.clone(), self.done.clone(), self.semaphore);
            self.thread = Some(std::thread::spawn(move || {
                let mut value = 0;
                while !done.load(Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(2));
                    value += 1;
                    signal(&device, semaphore, value);
                }
            }));
        }

        /// Submits the work recorded in `scheduler`, held back until the gate's next value.
        pub(crate) fn submit(&mut self, scheduler: &mut Scheduler) -> u64 {
            self.next += 1;
            let wait = SemaphoreWait {
                semaphore: self.semaphore,
                value: self.next,
                stage: vk::PipelineStageFlags2::ALL_COMMANDS,
            };
            scheduler.flush_with(&[wait], &[]).unwrap()
        }

        /// Lets every submission through.
        pub(crate) fn open(&mut self) {
            if self.done.swap(true, Ordering::Relaxed) {
                return;
            }
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
            }
            signal(&self.device, self.semaphore, u64::MAX >> 1);
        }
    }

    fn signal(device: &Device, semaphore: vk::Semaphore, value: u64) {
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(semaphore)
            .value(value);
        // SAFETY: the semaphore lives until the gate is dropped, after this thread joined.
        unsafe { device.raw.signal_semaphore(&info) }.unwrap();
    }

    impl Drop for Gate {
        fn drop(&mut self) {
            self.open();
            self.device.wait_idle();
            // SAFETY: the device is idle, so no submission waits on the semaphore.
            unsafe { self.device.raw.destroy_semaphore(self.semaphore, None) };
        }
    }

    /// The largest in-flight oversized total a flood saw, and the allocations made while
    /// earlier submissions were still held back.
    pub(crate) struct Flood {
        pub(crate) peak: u64,
        pub(crate) lagged: u32,
    }

    /// Makes `ops` operations of one oversized allocation each through `alloc` on a lagging
    /// GPU, submitting between operations whenever `should_submit` as the rasterizer does, then
    /// waits for everything.
    pub(crate) fn flood<R>(
        device: &Arc<Device>,
        ring: &mut R,
        ops: u32,
        mut alloc: impl FnMut(&mut R, &mut Scheduler),
        bytes: impl Fn(&R) -> u64,
        should_submit: impl Fn(&R, u64) -> bool,
    ) -> Flood {
        let mut scheduler = Scheduler::new(device).unwrap();
        let mut gate = Gate::new(device);
        gate.start();
        let mut flood = Flood { peak: 0, lagged: 0 };
        for _ in 0..ops {
            if should_submit(ring, scheduler.tick()) {
                gate.submit(&mut scheduler);
            }
            if scheduler.completed() + 1 < scheduler.tick() {
                flood.lagged += 1;
            }
            alloc(ring, &mut scheduler);
            flood.peak = flood.peak.max(bytes(ring));
        }
        gate.submit(&mut scheduler);
        gate.open();
        scheduler.finish().unwrap();
        flood
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{Gate, device};
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn an_allocation_larger_than_the_cap_waits_until_nothing_submitted_is_in_flight() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        let mut gate = Gate::new(&device);
        let mut oversized = Oversized::new(4 * MIB);
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let mut last = 0;
        for _ in 0..3 {
            oversized
                .alloc(&device, &mut scheduler, MIB, usage, "test")
                .unwrap();
            last = gate.submit(&mut scheduler);
        }
        assert!(!scheduler.is_complete(last), "the gate holds the GPU back");
        assert_eq!(oversized.live().bytes, 3 * MIB);
        gate.start();
        oversized
            .alloc(&device, &mut scheduler, 6 * MIB, usage, "test")
            .unwrap();
        assert!(
            scheduler.is_complete(last),
            "it waited for every submission"
        );
        assert_eq!(oversized.live().bytes, 6 * MIB, "it alone is in flight");
        gate.submit(&mut scheduler);
        gate.open();
        scheduler.finish().unwrap();
        oversized.reclaim(&mut scheduler);
        assert_eq!(oversized.live().bytes, 0);
    }

    #[test]
    fn allocations_never_wait_for_the_work_being_recorded() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        let mut oversized = Oversized::new(4 * MIB);
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let tick = scheduler.tick();
        for _ in 0..3 {
            oversized
                .alloc(&device, &mut scheduler, 2 * MIB, usage, "test")
                .unwrap();
        }
        assert_eq!(
            scheduler.tick(),
            tick,
            "nothing was submitted under the operation"
        );
        assert_eq!(oversized.live().bytes, 6 * MIB);
        assert!(oversized.should_submit(tick));
        scheduler.finish().unwrap();
        oversized.reclaim(&mut scheduler);
        assert!(!oversized.should_submit(scheduler.tick()));
        assert_eq!(oversized.live().bytes, 0);
    }

    #[test]
    fn readbacks_grow_for_large_downloads_and_shrink_once_downloads_fit_again() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        let mut readback = Readback::default();
        let mut download = |readback: &mut Readback, size| {
            readback
                .ensure(&device, &mut scheduler, size, "test readback")
                .unwrap();
            readback.release();
            readback.size()
        };
        assert_eq!(download(&mut readback, MIB), MIB);
        assert_eq!(download(&mut readback, 20 * MIB), READBACK_LIMIT);
        for _ in 1..READBACK_DECAY_DOWNLOADS {
            assert_eq!(
                download(&mut readback, MIB),
                READBACK_LIMIT,
                "kept while recent"
            );
        }
        assert_eq!(
            download(&mut readback, MIB),
            MIB,
            "decayed after a run of small ones"
        );
        assert_eq!(download(&mut readback, 2 * MIB), 2 * MIB);
        assert_eq!(
            download(&mut readback, READBACK_LIMIT + MIB),
            0,
            "a download above the limit frees its buffer once read"
        );
        assert_eq!(download(&mut readback, MIB), MIB);
    }

    #[test]
    fn the_rasterizer_caps_sum_to_its_host_cap() {
        assert_eq!(
            2 * RING_OVERSIZED_CAP + RASTERIZER_READBACKS * (READBACK_LIMIT - READBACK_BASE),
            RASTERIZER_HOST_CAP
        );
        assert!(READBACK_LIMIT.is_power_of_two() && READBACK_BASE <= READBACK_LIMIT);
    }
}
