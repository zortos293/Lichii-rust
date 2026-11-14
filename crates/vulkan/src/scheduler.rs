use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

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
    /// Tick of a sparse binding the next submission must wait for.
    bound: Option<u64>,
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
            bound: None,
        })
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

    /// The command buffer being recorded, begun on first use after a flush.
    pub fn record(&mut self) -> Result<vk::CommandBuffer> {
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
        self.current = Some(cmd);
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
        let bound = self.bound.take().map(|value| SemaphoreWait {
            semaphore: self.timeline.raw,
            value,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        });
        let wait_infos: Vec<_> = waits
            .iter()
            .chain(bound.iter())
            .map(|w| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(w.semaphore)
                    .value(w.value)
                    .stage_mask(w.stage)
            })
            .collect();
        let mut signal_infos = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.timeline.raw)
                .value(tick)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        signal_infos.extend(signals.iter().map(|&s| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(s)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        }));
        let cmd_infos: Vec<_> = cmd
            .iter()
            .map(|&c| vk::CommandBufferSubmitInfo::default().command_buffer(c))
            .collect();
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal_infos);
        self.device.submit(&[submit], vk::Fence::null())?;
        if let Some(cmd) = cmd {
            self.in_flight.push_back((tick, cmd));
        }
        self.tick += 1;
        Ok(tick)
    }

    /// Binds sparse memory to `image` after the work recorded so far and before any recorded
    /// later. The binding takes a tick of its own, which the next submission waits for.
    pub fn bind_sparse(
        &mut self,
        image: vk::Image,
        binds: &[vk::SparseImageMemoryBind],
        opaque: &[vk::SparseMemoryBind],
    ) -> Result<()> {
        let after = self.flush()?;
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
        if tick >= self.tick {
            self.flush()?;
        }
        let semaphores = [self.timeline.raw];
        let values = [tick];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore belongs to this scheduler and `tick` was submitted.
        unsafe {
            self.device
                .raw
                .wait_semaphores(&info, u64::MAX)
                .call("vkWaitSemaphores")?
        };
        self.completed = self.completed.max(tick);
        self.collect();
        Ok(())
    }

    /// Flushes and waits for everything recorded so far.
    pub fn finish(&mut self) -> Result<()> {
        let tick = self.flush()?;
        self.wait(tick)
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
        if let Err(error) = self.wait(last) {
            tracing::error!("waiting for the GPU on shutdown: {error}");
            self.device.wait_idle();
        }
        self.garbage.clear();
        // SAFETY: all submissions finished; the pool frees its command buffers.
        unsafe {
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
