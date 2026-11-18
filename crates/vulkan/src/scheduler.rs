use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use lichii_gpu::activity::{Activity, Wait};

use crate::device::Device;
use crate::error::{Error, Result, VkResultExt};

/// A semaphore a submission waits on: binary when `value` is 0, timeline otherwise.
#[derive(Clone, Copy, Debug)]
pub struct SemaphoreWait {
    pub semaphore: vk::Semaphore,
    pub value: u64,
    pub stage: vk::PipelineStageFlags2,
}

/// Records command buffers and submits them in order, each submission signalling the next
/// value ("tick") of one timeline semaphore. Resources used by recorded work are retired
/// against the current tick and dropped once the GPU passes it.
///
/// One scheduler belongs to one thread: the GPU thread's rasterizer and the presenter each own
/// one, sharing only the device's queue lock, so presentation cannot race the rasterizer's
/// command recording (NX2_NOTES row 118).
///
/// Once the device is lost it records, submits and waits for nothing: those calls return
/// [`Error::DeviceLost`].
pub struct Scheduler {
    device: Arc<Device>,
    timeline: Arc<Timeline>,
    tick: u64,
    completed: u64,
    pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    in_flight: VecDeque<(u64, vk::CommandBuffer)>,
    current: Option<vk::CommandBuffer>,
    garbage: VecDeque<(u64, Box<dyn Send>)>,
    /// Where blocking waits are timed as [`Wait::HostGpu`]: the GPU thread's scheduler only.
    activity: Option<Arc<Activity>>,
    /// Tick of a sparse binding the next submission must wait for.
    bound: Option<u64>,
    /// The dynamic rendering open in `current`, which later draws with the same attachments
    /// continue instead of beginning their own.
    rendering: Option<RenderingKey>,
    /// Dynamic renderings begun so far.
    renderings: u64,
    /// The command buffer and fence of [`Scheduler::run_aside`], made on first use.
    aside: Option<(vk::CommandBuffer, vk::Fence)>,
    /// A timeline semaphore each [`Scheduler::submit_part`] signals the next value of.
    parts: vk::Semaphore,
    part: u64,
    /// Whether a command buffer of the current tick was begun since the last flush, so the
    /// tick's submission carries recorded work.
    recorded: bool,
    /// A guest barrier asked for while nothing was being recorded, recorded at the start of the
    /// next command buffer instead.
    barrier: bool,
    /// Queue submissions and sparse bindings made so far.
    submits: u64,
    /// Reused submit infos, so a submission allocates nothing.
    wait_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    signal_infos: Vec<vk::SemaphoreSubmitInfo<'static>>,
    /// The dispatches recorded since the last compute barrier still owe one, which orders their
    /// shader accesses before later commands; [`Scheduler::record`] records it.
    compute_barrier_owed: bool,
    /// Compute barriers recorded so far.
    compute_barriers: u64,
}

/// The attachments and area of a dynamic rendering. Draws whose key matches the open
/// rendering record into it; anything else ends it first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderingKey {
    pub colors: [vk::ImageView; 8],
    pub depth: vk::ImageView,
    pub extent: vk::Extent2D,
}

impl Scheduler {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let semaphore_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(device.queue_family)
            .flags(
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                    | vk::CommandPoolCreateFlags::TRANSIENT,
            );
        // SAFETY: valid create info on a live device.
        let raw = unsafe {
            device
                .raw
                .create_semaphore(&semaphore_info, None)
                .call("vkCreateSemaphore")?
        };
        let timeline = Arc::new(Timeline {
            device: device.clone(),
            raw,
        });
        // SAFETY: valid create info on a live device.
        let parts = unsafe {
            device
                .raw
                .create_semaphore(&semaphore_info, None)
                .call("vkCreateSemaphore")?
        };
        // SAFETY: valid create info on a live device.
        let pool = unsafe {
            device
                .raw
                .create_command_pool(&pool_info, None)
                .call("vkCreateCommandPool")?
        };
        Ok(Self {
            device: device.clone(),
            timeline,
            tick: 1,
            completed: 0,
            pool,
            free: Vec::new(),
            in_flight: VecDeque::new(),
            current: None,
            garbage: VecDeque::new(),
            activity: None,
            bound: None,
            rendering: None,
            renderings: 0,
            aside: None,
            parts,
            part: 0,
            recorded: false,
            barrier: false,
            submits: 0,
            wait_infos: Vec::new(),
            signal_infos: Vec::new(),
            compute_barrier_owed: false,
            compute_barriers: 0,
        })
    }

    /// Times every blocking [`Scheduler::wait`] from now on in `activity`.
    pub fn time_waits(&mut self, activity: Arc<Activity>) {
        self.activity = Some(activity);
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// The timeline semaphore; other queues' submissions wait on it to consume this
    /// scheduler's output.
    pub fn timeline(&self) -> vk::Semaphore {
        self.timeline.raw
    }

    /// The timeline semaphore as a shared handle, for consumers that wait on it after this
    /// scheduler may be gone (published guest frames).
    pub fn shared_timeline(&self) -> Arc<Timeline> {
        self.timeline.clone()
    }

    /// The value the next [`Scheduler::flush`] signals; work recorded now completes at it.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// The tick everything recorded so far completes at, when it was all submitted already:
    /// nothing was recorded since the last flush. A fence signalled now can wait for it instead
    /// of for a submission of its own.
    pub fn submitted_tick(&self) -> Option<u64> {
        (!self.recorded).then(|| self.tick - 1)
    }

    /// Queue submissions and sparse bindings made so far, [`Scheduler::run_aside`]'s included.
    pub fn submits(&self) -> u64 {
        self.submits
    }

    /// A full memory barrier: later commands may read anything earlier ones wrote. With no
    /// command buffer being recorded it is recorded at the start of the next one, so a barrier
    /// alone gives a submission nothing to carry. It orders everything an owed compute barrier
    /// would, which it pays. `true` when it was recorded now.
    pub fn barrier(&mut self) -> Result<bool> {
        let Some(cmd) = self.current else {
            self.barrier = true;
            return Ok(false);
        };
        self.end_rendering(cmd);
        self.record_barrier(cmd);
        Ok(true)
    }

    fn record_barrier(&mut self, cmd: vk::CommandBuffer) {
        self.compute_barrier_owed = false;
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        // SAFETY: `cmd` is recording outside any rendering.
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }
    }

    /// The command buffer being recorded, begun on first use after a flush, outside any
    /// rendering: an open one ends here.
    pub fn record(&mut self) -> Result<vk::CommandBuffer> {
        let cmd = self.command_buffer()?;
        self.end_rendering(cmd);
        self.record_compute_barrier(cmd);
        Ok(cmd)
    }

    /// The command buffer being recorded, outside any rendering, for a dispatch that touches
    /// nothing the dispatches since the last compute barrier wrote, and writes nothing they
    /// accessed: the owed barrier stays owed.
    pub fn record_beside_dispatches(&mut self) -> Result<vk::CommandBuffer> {
        let cmd = self.command_buffer()?;
        self.end_rendering(cmd);
        Ok(cmd)
    }

    /// Owes the barrier that orders the dispatches recorded so far before later commands.
    pub fn owe_compute_barrier(&mut self) {
        self.compute_barrier_owed = true;
    }

    /// Whether the dispatches recorded since the last compute barrier still owe one.
    pub fn compute_barrier_owed(&self) -> bool {
        self.compute_barrier_owed
    }

    /// Compute barriers recorded so far.
    pub fn compute_barriers(&self) -> u64 {
        self.compute_barriers
    }

    fn record_compute_barrier(&mut self, cmd: vk::CommandBuffer) {
        if !std::mem::take(&mut self.compute_barrier_owed) {
            return;
        }
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        // SAFETY: `cmd` is recording outside any rendering.
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            )
        };
        self.compute_barriers += 1;
    }

    /// The command buffer being recorded, inside a dynamic rendering of `key`: the open one when
    /// its key matches, else a new one begun with `info`. `true` when it began one.
    pub fn record_rendering(
        &mut self,
        key: RenderingKey,
        info: &vk::RenderingInfo<'_>,
    ) -> Result<(vk::CommandBuffer, bool)> {
        let cmd = self.command_buffer()?;
        if self.rendering == Some(key) {
            return Ok((cmd, false));
        }
        self.end_rendering(cmd);
        self.record_compute_barrier(cmd);
        // SAFETY: `cmd` is recording outside any rendering, and the caller built `info` from
        // live attachment views in their attachment layouts.
        unsafe { self.device.raw.cmd_begin_rendering(cmd, info) };
        self.rendering = Some(key);
        self.renderings += 1;
        Ok((cmd, true))
    }

    /// The key of the open dynamic rendering.
    pub fn rendering(&self) -> Option<RenderingKey> {
        self.rendering
    }

    /// Dynamic renderings begun so far.
    pub fn renderings(&self) -> u64 {
        self.renderings
    }

    fn end_rendering(&mut self, cmd: vk::CommandBuffer) {
        if self.rendering.take().is_some() {
            // SAFETY: `cmd` is recording inside the rendering `record_rendering` began.
            unsafe { self.device.raw.cmd_end_rendering(cmd) };
        }
    }

    fn command_buffer(&mut self) -> Result<vk::CommandBuffer> {
        if crate::debug::device_is_lost() {
            return Err(Error::DeviceLost);
        }
        if let Some(cmd) = self.current {
            return Ok(cmd);
        }
        self.collect();
        let cmd = match self.free.pop() {
            Some(cmd) => cmd,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                // SAFETY: the pool belongs to this scheduler and this thread.
                unsafe {
                    self.device
                        .raw
                        .allocate_command_buffers(&info)
                        .call("vkAllocateCommandBuffers")?[0]
                }
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` is not pending: it is new or its tick completed.
        unsafe {
            self.device
                .raw
                .begin_command_buffer(cmd, &begin)
                .call("vkBeginCommandBuffer")?
        };
        self.device
            .checkpoint(cmd, format_args!("commands of tick {}", self.tick));
        if std::mem::take(&mut self.barrier) {
            self.record_barrier(cmd);
        }
        self.current = Some(cmd);
        self.recorded = true;
        Ok(cmd)
    }

    /// Drops `object` once the GPU finished the work recorded so far.
    pub fn retire<T: Send + 'static>(&mut self, object: T) {
        self.garbage.push_back((self.tick, Box::new(object)));
    }

    /// Submits the recorded work; returns the tick it signals.
    pub fn flush(&mut self) -> Result<u64> {
        self.flush_with(&[], &[])
    }

    /// Submits the recorded work (possibly none) after `waits`, signalling the timeline and
    /// `signals` (binary semaphores) when done.
    pub fn flush_with(
        &mut self,
        waits: &[SemaphoreWait],
        signals: &[vk::Semaphore],
    ) -> Result<u64> {
        let tick = self.tick;
        let mut signal_infos = std::mem::take(&mut self.signal_infos);
        signal_infos.clear();
        signal_infos.push(
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.raw)
                .value(tick)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        );
        signal_infos.extend(signals.iter().map(|&s| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(s)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        }));
        let bound = self.bound.take();
        let result = self.submit(waits, bound, &signal_infos);
        self.signal_infos = signal_infos;
        result?;
        self.tick += 1;
        self.recorded = false;
        Ok(tick)
    }

    /// Submits the work recorded so far without ending the tick, and returns the part
    /// [`Scheduler::wait_part`] waits for. Everything tagged with or retired against the tick
    /// stays alive until the tick's own submission completes, so an operation still being set up
    /// can submit in the middle: allocations it made earlier and has not recorded its use of yet
    /// stay valid. Commands recorded later go to a new command buffer of the same tick.
    pub fn submit_part(&mut self) -> Result<u64> {
        let part = self.part + 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.parts)
            .value(part)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        self.submit(&[], self.bound, &signal)?;
        self.part = part;
        Ok(part)
    }

    /// Blocks until the work [`Scheduler::submit_part`] submitted as `part` finished.
    pub fn wait_part(&mut self, part: u64) -> Result<()> {
        // SAFETY: the semaphore belongs to this scheduler.
        if unsafe { self.device.raw.get_semaphore_counter_value(self.parts) }
            .is_ok_and(|done| done >= part)
        {
            return Ok(());
        }
        if crate::debug::device_is_lost() {
            return Err(Error::DeviceLost);
        }
        let semaphores = [self.parts];
        let values = [part];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        let _waiting = self.activity.as_deref().map(|a| a.waiting(Wait::HostGpu));
        // SAFETY: the semaphore belongs to this scheduler and `part` was submitted.
        unsafe {
            self.device
                .raw
                .wait_semaphores(&info, u64::MAX)
                .call("vkWaitSemaphores")
        }
    }

    /// Ends and submits the command buffer being recorded (possibly none) after `waits` and
    /// the sparse binding `bound`, signalling `signals`.
    fn submit(
        &mut self,
        waits: &[SemaphoreWait],
        bound: Option<u64>,
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<()> {
        if crate::debug::device_is_lost() {
            return Err(Error::DeviceLost);
        }
        if let Some(cmd) = self.current {
            self.end_rendering(cmd);
            self.record_compute_barrier(cmd);
        }
        let cmd = self.current.take();
        if let Some(cmd) = cmd {
            // SAFETY: `cmd` is in the recording state.
            unsafe {
                self.device
                    .raw
                    .end_command_buffer(cmd)
                    .call("vkEndCommandBuffer")?
            };
        }
        let bound = bound.map(|value| SemaphoreWait {
            semaphore: self.timeline.raw,
            value,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        });
        self.wait_infos.clear();
        self.wait_infos
            .extend(waits.iter().chain(bound.iter()).map(|w| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(w.semaphore)
                    .value(w.value)
                    .stage_mask(w.stage)
            }));
        let cmd_info = cmd.map(|c| vk::CommandBufferSubmitInfo::default().command_buffer(c));
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&self.wait_infos)
            .command_buffer_infos(cmd_info.as_slice())
            .signal_semaphore_infos(signals);
        self.submits += 1;
        self.device.submit(&[submit], vk::Fence::null())?;
        if let Some(cmd) = cmd {
            self.in_flight.push_back((self.tick, cmd));
        }
        Ok(())
    }

    /// Binds sparse memory to `image` after the work recorded so far and before any recorded
    /// later. The binding takes a tick of its own, which the next submission waits for; it
    /// submits the recorded work first only when there is some.
    pub fn bind_sparse(
        &mut self,
        image: vk::Image,
        binds: &[vk::SparseImageMemoryBind],
        opaque: &[vk::SparseMemoryBind],
    ) -> Result<()> {
        let after = match self.submitted_tick() {
            Some(tick) => tick,
            None => self.flush()?,
        };
        let tick = self.tick;
        let image_info = [vk::SparseImageMemoryBindInfo::default()
            .image(image)
            .binds(binds)];
        let opaque_info = [vk::SparseImageOpaqueMemoryBindInfo::default()
            .image(image)
            .binds(opaque)];
        let semaphores = [self.timeline.raw];
        let (waits, signals) = ([after], [tick]);
        let mut values = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&waits)
            .signal_semaphore_values(&signals);
        let mut info = vk::BindSparseInfo::default()
            .wait_semaphores(&semaphores)
            .signal_semaphores(&semaphores)
            .push_next(&mut values);
        if !binds.is_empty() {
            info = info.image_binds(&image_info);
        }
        if !opaque.is_empty() {
            info = info.image_opaque_binds(&opaque_info);
        }
        self.submits += 1;
        self.device.bind_sparse(&[info])?;
        self.bound = Some(tick);
        self.tick += 1;
        Ok(())
    }

    /// The last tick the GPU finished, refreshed from the semaphore.
    pub fn completed(&mut self) -> u64 {
        // SAFETY: the timeline semaphore belongs to this scheduler.
        if let Ok(value) = unsafe {
            self.device
                .raw
                .get_semaphore_counter_value(self.timeline.raw)
        } {
            self.completed = self.completed.max(value);
        }
        self.completed
    }

    /// Whether `tick` finished, without querying the GPU when the cached value suffices.
    pub fn is_complete(&mut self, tick: u64) -> bool {
        tick <= self.completed || tick <= self.completed()
    }

    /// Blocks until `tick` finished, flushing first when it is still being recorded.
    pub fn wait(&mut self, tick: u64) -> Result<()> {
        if self.is_complete(tick) {
            return Ok(());
        }
        if crate::debug::device_is_lost() {
            return Err(Error::DeviceLost);
        }
        if tick >= self.tick {
            self.flush()?;
        }
        let semaphores = [self.timeline.raw];
        let values = [tick];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        {
            let _waiting = self.activity.as_deref().map(|a| a.waiting(Wait::HostGpu));
            // SAFETY: the semaphore belongs to this scheduler and `tick` was submitted.
            unsafe {
                self.device
                    .raw
                    .wait_semaphores(&info, u64::MAX)
                    .call("vkWaitSemaphores")?
            };
        }
        self.completed = self.completed.max(tick);
        self.collect();
        Ok(())
    }

    /// Flushes and waits for everything recorded so far.
    pub fn finish(&mut self) -> Result<()> {
        let tick = self.flush()?;
        self.wait(tick)
    }

    /// Runs what `record` records in a command buffer of its own, submitted ahead of the work
    /// being recorded, and waits for it alone. For a read whose writers were all submitted
    /// already: the wait covers neither the recorded work nor its submission. The recorded
    /// commands must not depend on work recorded since the last flush.
    pub fn run_aside(&mut self, record: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        if crate::debug::device_is_lost() {
            return Err(Error::DeviceLost);
        }
        let (cmd, fence) = match self.aside {
            Some(aside) => aside,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                // SAFETY: the pool belongs to this scheduler and this thread.
                let cmd = unsafe {
                    self.device
                        .raw
                        .allocate_command_buffers(&info)
                        .call("vkAllocateCommandBuffers")?[0]
                };
                // SAFETY: valid create info on a live device.
                let fence = unsafe {
                    self.device
                        .raw
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .call("vkCreateFence")?
                };
                self.aside = Some((cmd, fence));
                (cmd, fence)
            }
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: `cmd` is not pending: every earlier use was waited for below.
        unsafe {
            self.device
                .raw
                .begin_command_buffer(cmd, &begin)
                .call("vkBeginCommandBuffer")?
        };
        record(cmd);
        // SAFETY: `cmd` is in the recording state.
        unsafe {
            self.device
                .raw
                .end_command_buffer(cmd)
                .call("vkEndCommandBuffer")?
        };
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
        let submit = vk::SubmitInfo2::default().command_buffer_infos(&cmd_infos);
        self.submits += 1;
        self.device.submit(&[submit], fence)?;
        let _waiting = self.activity.as_deref().map(|a| a.waiting(Wait::HostGpu));
        // SAFETY: the fence belongs to this scheduler and was just submitted.
        unsafe {
            self.device
                .raw
                .wait_for_fences(&[fence], true, u64::MAX)
                .call("vkWaitForFences")?;
            self.device
                .raw
                .reset_fences(&[fence])
                .call("vkResetFences")?;
        }
        Ok(())
    }

    /// Recycles the command buffers and drops the retired objects of the work the GPU
    /// finished; returns the last finished tick.
    pub fn collect(&mut self) -> u64 {
        let completed = self.completed();
        while let Some(&(tick, cmd)) = self.in_flight.front() {
            if tick > completed {
                break;
            }
            self.in_flight.pop_front();
            // SAFETY: the command buffer's submission completed.
            if unsafe {
                self.device
                    .raw
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
            }
            .is_ok()
            {
                self.free.push(cmd);
            }
        }
        while self
            .garbage
            .front()
            .is_some_and(|(tick, _)| *tick <= completed)
        {
            self.garbage.pop_front();
        }
        completed
    }
}

impl Drop for Scheduler {
    fn drop(&mut self) {
        if self.current.is_some() {
            let _ = self.flush();
        }
        let last = self.tick - 1;
        if let Err(error) = self.wait(last).and_then(|()| self.wait_part(self.part)) {
            tracing::error!("waiting for the GPU on shutdown: {error}");
            self.device.wait_idle();
        }
        self.garbage.clear();
        // SAFETY: all submissions finished; the pool frees its command buffers.
        unsafe {
            if let Some((_, fence)) = self.aside {
                self.device.raw.destroy_fence(fence, None);
            }
            self.device.raw.destroy_semaphore(self.parts, None);
            self.device.raw.destroy_command_pool(self.pool, None);
        }
    }
}

/// A timeline semaphore, destroyed when its last holder drops it.
pub struct Timeline {
    device: Arc<Device>,
    raw: vk::Semaphore,
}

impl Timeline {
    pub fn raw(&self) -> vk::Semaphore {
        self.raw
    }
}

impl Drop for Timeline {
    fn drop(&mut self) {
        // SAFETY: no holder remains, so no pending submission waits on or signals it.
        unsafe { self.device.raw.destroy_semaphore(self.raw, None) };
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::GraphicsSettings;

    use super::*;
    use crate::{Instance, SurfaceSupport};

    fn device() -> Option<Arc<Device>> {
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

    #[test]
    fn draws_with_one_key_share_a_rendering_until_other_work_records() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        let key = |size| RenderingKey {
            colors: [vk::ImageView::null(); 8],
            depth: vk::ImageView::null(),
            extent: vk::Extent2D {
                width: size,
                height: size,
            },
        };
        let info = |size| {
            vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: size,
                        height: size,
                    },
                })
                .layer_count(1)
        };

        let (cmd, began) = scheduler.record_rendering(key(16), &info(16)).unwrap();
        assert!(began);
        assert_eq!(
            scheduler.record_rendering(key(16), &info(16)).unwrap(),
            (cmd, false)
        );
        assert!(scheduler.record_rendering(key(8), &info(8)).unwrap().1);
        assert_eq!(scheduler.rendering(), Some(key(8)));

        assert_eq!(scheduler.record().unwrap(), cmd);
        assert_eq!(scheduler.rendering(), None);
        assert!(scheduler.record_rendering(key(8), &info(8)).unwrap().1);

        scheduler.finish().unwrap();
        assert_eq!(scheduler.rendering(), None);
        assert_eq!(scheduler.renderings(), 3);
        assert!(scheduler.record_rendering(key(8), &info(8)).unwrap().1);
        scheduler.finish().unwrap();
    }

    #[test]
    fn barriers_with_nothing_recorded_open_the_next_command_buffer() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        assert!(!scheduler.barrier().unwrap(), "nothing is being recorded");
        assert!(scheduler.barrier);
        assert_eq!(scheduler.submitted_tick(), Some(0));
        let cmd = scheduler.record().unwrap();
        assert!(!scheduler.barrier, "the new command buffer starts with it");
        assert!(
            scheduler.barrier().unwrap(),
            "recorded at once while recording"
        );
        assert!(!scheduler.barrier);
        assert_eq!(scheduler.record().unwrap(), cmd);
        scheduler.finish().unwrap();
        assert_eq!(device.instance().validation_errors(), 0);
    }

    #[test]
    fn held_guest_barriers_and_owed_compute_barriers_are_both_paid() {
        let Some(device) = device() else { return };
        let mut scheduler = Scheduler::new(&device).unwrap();
        let barriers = scheduler.compute_barriers();
        scheduler.record().unwrap();
        scheduler.owe_compute_barrier();
        scheduler.flush().unwrap();
        assert!(!scheduler.compute_barrier_owed());
        assert_eq!(
            scheduler.compute_barriers(),
            barriers + 1,
            "a submission ends with the owed compute barrier"
        );

        let submits = scheduler.submits();
        assert!(!scheduler.barrier().unwrap());
        scheduler.owe_compute_barrier();
        assert!(
            scheduler.submitted_tick().is_some(),
            "neither barrier opens a command buffer"
        );
        assert_eq!(scheduler.submits(), submits);
        scheduler.record().unwrap();
        assert!(
            !scheduler.barrier,
            "the guest barrier opens the next command buffer"
        );
        assert!(
            !scheduler.compute_barrier_owed(),
            "the guest barrier orders the dispatches too"
        );
        assert_eq!(scheduler.compute_barriers(), barriers + 1);

        scheduler.owe_compute_barrier();
        assert!(scheduler.barrier().unwrap());
        assert!(!scheduler.compute_barrier_owed());
        scheduler.owe_compute_barrier();
        scheduler.submit_part().unwrap();
        assert!(!scheduler.compute_barrier_owed());
        assert_eq!(
            scheduler.compute_barriers(),
            barriers + 2,
            "a part submission ends with the owed compute barrier"
        );
        scheduler.finish().unwrap();
        assert_eq!(device.instance().validation_errors(), 0);
    }
}
