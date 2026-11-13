//! The compute engine (Kepler B1C0, Ampere C7C0).
//!
//! B1C0: LAUNCH reads the version 1.x launch descriptor (QMD) at `launch_desc_loc` and hands
//! the dispatch to the renderer.
//!
//! C7C0 (NVN2 native compute): version 3.0 QMDs run through PCAS scheduling ([`pcas`]).
//! SEND_SIGNALING_PCAS(2)_B act on the QMD whose address SEND_PCAS_A holds; the last
//! LOAD_INLINE_QMD_DATA word writes the inline QMD to its address and schedules it unless it waits
//! for a scheduling PCAS. A QMD that runs dispatches, performs its release semaphores, then acts
//! on its dependent QMD. SET_REPORT_SEMAPHORE_D releases a semaphore after the work before it.
//!
//! Inline uploads share the Kepler memory engine's registers on both classes. On B1C0, an upload
//! that writes a descriptor's grid from command words GPU work may have written makes its
//! dispatch indirect ([`ComputeContext::indirect`]). On C7C0 the front already synchronises
//! before fetching such words, so a QMD's grid is stale only when GPU work wrote the QMD itself:
//! a QMD whose grid words may be GPU-written dispatches indirectly from them.

pub mod pcas;
pub mod qmd;
pub mod regs;

use std::time::Instant;

use lichii_common::GpuVAddr;

use super::EngineContext;
use super::upload::Upload;
use crate::activity::RendererCall;
use crate::backend::{
    ComputeContext, FenceAction, IndirectGrid, QmdVersion, Query, QueryKind, ReleaseSize,
};
use crate::gmmu::{Gmmu, GpuFault, RangeTarget};
use crate::host::Reduction;
use crate::reg_index;
use pcas::{CHAIN_LIMIT, PcasAction, PcasError};
use qmd::{QMD_WORDS, QmdV2, QmdV3, Release, ReleaseKind};
pub use regs::Regs;

const EXEC_UPLOAD: u32 = reg_index!(Regs, exec_upload);
const DATA_UPLOAD: u32 = reg_index!(Regs, data_upload);
const LAUNCH: u32 = reg_index!(Regs, launch);
const SEND_SIGNALING_PCAS2_B: u32 = reg_index!(Regs, send_signaling_pcas2_b);
const LOAD_INLINE_QMD_LAST: u32 = reg_index!(Regs, inline_qmd) + QMD_WORDS as u32 - 1;
const SET_REPORT_SEMAPHORE_D: u32 = reg_index!(Regs, report_semaphore.control);

/// Uploads a LAUNCH looks back at, at most.
const MAX_UPLOADS: usize = 8;
/// Byte offset of a version 1.x descriptor's grid.
const QMD_V2_GRID: u64 = core::mem::offset_of!(QmdV2, grid_x) as u64;
/// Byte offset of a version 3.0 descriptor's grid.
const QMD_V3_GRID: u64 = core::mem::offset_of!(QmdV3, grid_width) as u64;
/// Words of a version 1.x grid.
const GRID_WORDS: u32 = 2;

/// An inline upload since the last LAUNCH: where it wrote, how many bytes it wrote contiguously
/// from there (its first line, when linear), and where its first
/// [`GRID_WORDS`] data words came from when they were contiguous in the command stream.
#[derive(Clone, Copy)]
struct UploadSource {
    target: GpuVAddr,
    contiguous: u64,
    source: Option<GpuVAddr>,
}

/// The compute engine bound to a subchannel.
pub struct KeplerCompute {
    pub regs: Box<Regs>,
    version: QmdVersion,
    upload: Upload,
    /// Where the next DATA_UPLOAD word comes from, when the front said.
    upload_next: Option<GpuVAddr>,
    uploads: Vec<UploadSource>,
    /// Data words the last upload received, up to [`GRID_WORDS`].
    upload_words: u32,
}

impl KeplerCompute {
    /// An engine for `class`: Ampere compute classes (C6C0, C7C0) launch version 3.0
    /// descriptors, every older class version 1.x.
    pub fn new(class: u32) -> Self {
        Self {
            regs: bytemuck::zeroed_box(),
            version: match class {
                0xc6c0 | 0xc7c0 => QmdVersion::V3,
                _ => QmdVersion::V2,
            },
            upload: Upload::default(),
            upload_next: None,
            uploads: Vec::with_capacity(MAX_UPLOADS),
            upload_words: GRID_WORDS,
        }
    }

    /// See [`Engine::upload_source`](super::Engine::upload_source).
    pub fn upload_source(&mut self, address: GpuVAddr) {
        if self.upload_words != 0
            && self.upload_words < GRID_WORDS
            && let Some(last) = self.uploads.last_mut()
        {
            last.source = None;
        }
        self.upload_next = Some(address);
    }

    /// The descriptor layout LAUNCH decodes.
    pub fn qmd_version(&self) -> QmdVersion {
        self.version
    }

    pub fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        let Some(word) =
            bytemuck::cast_slice_mut::<Regs, u32>(std::slice::from_mut(&mut *self.regs))
                .get_mut(method as usize)
        else {
            return;
        };
        *word = argument;
        match (self.version, method) {
            (_, EXEC_UPLOAD) => {
                if self.version == QmdVersion::V2 {
                    if self.uploads.len() == MAX_UPLOADS {
                        self.uploads.remove(0);
                    }
                    let upload = &self.regs.upload;
                    let linear = self.regs.exec_upload.linear() != 0;
                    self.uploads.push(UploadSource {
                        target: upload.dest.address(),
                        contiguous: if linear && upload.line_count != 0 {
                            u64::from(upload.line_length_in)
                        } else {
                            0
                        },
                        source: None,
                    });
                    self.upload_words = 0;
                }
                self.upload
                    .exec(&self.regs.upload, self.regs.exec_upload.linear() != 0)
            }
            (_, DATA_UPLOAD) => {
                if self.upload_words == 0
                    && let Some(last) = self.uploads.last_mut()
                {
                    last.source = self.upload_next;
                }
                self.upload_words = (self.upload_words + 1).min(GRID_WORDS);
                self.upload_next = self.upload_next.map(|address| address + 4);
                self.upload.data(ctx, argument)
            }
            (QmdVersion::V2, LAUNCH) => {
                self.launch(ctx);
                self.uploads.clear();
            }
            (QmdVersion::V2, _) => {}
            (QmdVersion::V3, LAUNCH) => {
                if let Some(action) = PcasAction::signaling(argument) {
                    self.schedule(ctx, self.regs.launch_desc_loc.address(), action);
                }
            }
            (QmdVersion::V3, SEND_SIGNALING_PCAS2_B) => match PcasAction::signaling2(argument) {
                Ok(Some(action)) => self.schedule(ctx, self.regs.launch_desc_loc.address(), action),
                Ok(None) => {}
                Err(error) => tracing::warn!(%error, argument, "native compute PCAS skipped"),
            },
            (QmdVersion::V3, LOAD_INLINE_QMD_LAST) => self.load_inline_qmd(ctx),
            (QmdVersion::V3, SET_REPORT_SEMAPHORE_D) => self.report_semaphore(ctx),
            (QmdVersion::V3, _) => {}
        }
    }

    fn launch(&self, ctx: &mut EngineContext<'_>) {
        let address = self.regs.launch_desc_loc.address();
        let words = match read_qmd(ctx.memory, address) {
            Ok(words) => words,
            Err(fault) => {
                tracing::warn!(
                    ?fault,
                    address,
                    "compute launch descriptor in unmapped memory"
                );
                return;
            }
        };
        let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV2>(&words);
        let grid = address + QMD_V2_GRID;
        let indirect = self
            .uploads
            .iter()
            .rev()
            .find(|upload| upload.target == grid)
            .filter(|upload| upload.contiguous >= u64::from(GRID_WORDS) * 4)
            .and_then(|upload| upload.source)
            .filter(|&source| gpu_written(ctx, source, u64::from(GRID_WORDS) * 4))
            .map(|address| IndirectGrid {
                address,
                layout: QmdVersion::V2,
            });
        self.dispatch(
            ctx,
            qmd.launch(self.regs.code_loc.address()),
            &words,
            indirect,
        );
    }

    /// Applies a PCAS to the QMD at `address`, then runs every QMD it makes due, following
    /// their dependent links.
    fn schedule(&self, ctx: &mut EngineContext<'_>, address: GpuVAddr, action: PcasAction) {
        let (mut address, mut action) = (address, action);
        let result = (|| {
            for _ in 0..CHAIN_LIMIT {
                let key = ctx
                    .memory
                    .translate(address)
                    .ok_or(PcasError::Unmapped(address))?;
                let memory = ctx.memory;
                let due = ctx.qmd_scheduler.signal(key, action, || {
                    let words =
                        read_qmd(memory, address).map_err(|_| PcasError::Unmapped(address))?;
                    let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV3>(&words);
                    qmd.check()?;
                    Ok(qmd.control.dependence_counter())
                })?;
                if !due {
                    return Ok(());
                }
                match self.run(ctx, address)? {
                    Some(dependent) => (address, action) = dependent,
                    None => return Ok(()),
                }
            }
            Err(PcasError::ChainLimit)
        })();
        if let Err(error) = result {
            tracing::warn!(%error, address, "native compute QMD stopped");
        }
    }

    /// Runs a due QMD: reads it again (NVN2 patches release payloads while a QMD waits),
    /// dispatches it, performs its releases and returns what it does to its dependent QMD. A
    /// dispatch the renderer cannot take still releases: other queues and the CPU wait on them
    /// (NX2_NOTES row 61).
    fn run(
        &self,
        ctx: &mut EngineContext<'_>,
        address: GpuVAddr,
    ) -> Result<Option<(GpuVAddr, PcasAction)>, PcasError> {
        let words = read_qmd(ctx.memory, address).map_err(|_| PcasError::Unmapped(address))?;
        let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV3>(&words);
        qmd.check()?;
        let grid = IndirectGrid {
            address: address + QMD_V3_GRID,
            layout: QmdVersion::V3,
        };
        let indirect = gpu_written(ctx, grid.address, grid.size()).then_some(grid);
        match qmd.launch() {
            Ok(launch) => self.dispatch(ctx, launch, &words, indirect),
            Err(error) => tracing::warn!(%error, address, "compute dispatch skipped"),
        }
        for (slot, release) in qmd.release_slots().into_iter().enumerate() {
            if !release.enabled() {
                continue;
            }
            match release.decode() {
                Some(release) => perform_release(ctx, &release),
                None => tracing::warn!(
                    slot,
                    flags = release.flags.0,
                    "compute release in an unsupported format skipped"
                ),
            }
        }
        let dependent = qmd.dependent;
        if dependent.second_enable() {
            return Err(PcasError::SecondDependent);
        }
        if !dependent.enable() {
            return Ok(None);
        }
        Ok(Some((
            u64::from(qmd.dependent_qmd_pointer) << 8,
            PcasAction::dependent(dependent.action())?,
        )))
    }

    /// The last LOAD_INLINE_QMD_DATA word: the QMD goes to memory at the inline QMD address and
    /// is scheduled there unless it waits for a scheduling PCAS. SEND_PCAS_A keeps its value.
    fn load_inline_qmd(&self, ctx: &mut EngineContext<'_>) {
        let address = self.regs.inline_qmd_address.address();
        let bytes = bytemuck::bytes_of(&self.regs.inline_qmd);
        if let Err(fault) = ctx.memory.write_block(address, bytes) {
            tracing::warn!(?fault, address, "inline QMD to unmapped GPU memory");
            return;
        }
        ctx.rasterizer
            .inline_written(ctx.channel, ctx.memory, address, bytes.len() as u64);
        let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV3>(&self.regs.inline_qmd);
        if self.regs.qmd_virtualization_control.send_pcas()
            || !qmd.control.require_scheduling_pcas()
        {
            self.schedule(ctx, address, PcasAction::Schedule { invalidate: true });
        }
    }

    /// SET_REPORT_SEMAPHORE_D: a release (operation 0, or 2 as the reference also takes) of the
    /// payload, as one word or with a timestamp, or a reduction into one word. Traps are not
    /// emulated.
    fn report_semaphore(&self, ctx: &mut EngineContext<'_>) {
        let semaphore = &self.regs.report_semaphore;
        let control = semaphore.control;
        if !matches!(control.operation(), 0 | 2) {
            tracing::warn!(
                operation = control.operation(),
                "compute report semaphore operation not supported"
            );
            return;
        }
        let kind = if control.reduction_enable() {
            ReleaseKind::Reduction {
                op: qmd::semaphore_reduction(control.reduction_op()),
                signed: control.reduction_signed(),
            }
        } else if control.one_word() {
            ReleaseKind::Word
        } else {
            ReleaseKind::WordWithTimestamp
        };
        perform_release(
            ctx,
            &Release {
                address: semaphore.address(),
                payload: semaphore.payload,
                kind,
            },
        );
    }

    fn dispatch(
        &self,
        ctx: &mut EngineContext<'_>,
        launch: qmd::Launch,
        words: &[u32; QMD_WORDS],
        indirect: Option<IndirectGrid>,
    ) {
        if launch.block.contains(&0) || (indirect.is_none() && launch.grid.contains(&0)) {
            return;
        }
        let tic = &self.regs.tic;
        let tsc = &self.regs.tsc;
        let start = Instant::now();
        ctx.rasterizer.dispatch_compute(&ComputeContext {
            channel: ctx.channel,
            memory: ctx.memory,
            program: launch.program,
            grid: launch.grid,
            block: launch.block,
            shared_memory_size: launch.shared_memory_size,
            local_memory_size: launch.local_memory_size,
            const_buffers: &launch.const_buffers,
            texture_headers: tic.address(),
            texture_header_limit: tic.limit,
            samplers: tsc.address(),
            sampler_limit: tsc.limit,
            linked_samplers: launch.linked_samplers,
            qmd_version: self.version,
            qmd: words,
            indirect_grid: indirect
                .filter(|grid| grid.layout == QmdVersion::V2)
                .map(|grid| grid.address),
            indirect,
        });
        ctx.activity.renderer_call(RendererCall::Dispatch, start);
    }
}

/// Whether GPU work may have written any of [`address`, +`size`).
fn gpu_written(ctx: &EngineContext<'_>, address: GpuVAddr, size: u64) -> bool {
    let mut written = false;
    ctx.memory.for_each_range(address, size, |range| {
        if let RangeTarget::Device(device) = range.target {
            written |= ctx.written_pages.may_be_written(device, range.size);
        }
    });
    written
}

fn read_qmd(memory: &Gmmu, address: GpuVAddr) -> Result<[u32; QMD_WORDS], GpuFault> {
    let mut words = [0u32; QMD_WORDS];
    memory.read_block(address, bytemuck::bytes_of_mut(&mut words))?;
    Ok(words)
}

/// Performs one release after the dispatch. Plain releases are queries; reductions and max
/// releases are fence actions, so they combine with the word in memory only once every earlier
/// release and the dispatch's GPU work finished.
fn perform_release(ctx: &mut EngineContext<'_>, release: &Release) {
    let Release {
        address,
        payload,
        kind,
    } = *release;
    let (op, signed) = match kind {
        ReleaseKind::Reduction { op, signed } => (op, signed),
        ReleaseKind::Max => (Reduction::Max, false),
        ReleaseKind::Word | ReleaseKind::WordWithTimestamp => {
            let timestamped = kind == ReleaseKind::WordWithTimestamp;
            let mut probe = [0u8; 16];
            if let Err(fault) = ctx
                .memory
                .read_block(address, &mut probe[..if timestamped { 16 } else { 4 }])
            {
                tracing::warn!(?fault, address, "compute release to unmapped memory");
                return;
            }
            ctx.rasterizer.query(
                ctx.channel,
                ctx.memory,
                &Query {
                    address,
                    kind: QueryKind::Payload,
                    payload: u64::from(payload),
                    size: if timestamped {
                        ReleaseSize::Sixteen
                    } else {
                        ReleaseSize::Four
                    },
                    after_work: true,
                },
            );
            return;
        }
    };
    let Some(address) = ctx.memory.translate(address) else {
        tracing::warn!(address, "compute reduction on unmapped memory");
        return;
    };
    ctx.rasterizer.signal_fence(FenceAction::ReduceSemaphore {
        address,
        op,
        signed,
        payload,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{
        ChannelId, ConstBufferBinding, DrawParams, FenceAction, FramebufferConfig, GraphicsContext,
        IndirectDraw, Rasterizer,
    };
    use crate::engines::testing::Harness;
    use crate::gmmu::Gmmu;
    use lichii_common::{DAddr, GpuVAddr};

    #[derive(Debug, PartialEq)]
    struct Dispatch {
        program: GpuVAddr,
        grid: [u32; 3],
        block: [u32; 3],
        shared_memory_size: u32,
        const_buffers: [ConstBufferBinding; 8],
        texture_headers: GpuVAddr,
        texture_header_limit: u32,
        samplers: GpuVAddr,
        sampler_limit: u32,
        linked_samplers: bool,
        qmd_version: QmdVersion,
        qmd: [u32; 64],
        indirect_grid: Option<GpuVAddr>,
        indirect: Option<IndirectGrid>,
    }

    #[derive(Default)]
    struct Spy {
        dispatches: Vec<Dispatch>,
        queries: Vec<Query>,
        fences: Vec<FenceAction>,
        inline_writes: Vec<(GpuVAddr, u64)>,
        order: Vec<&'static str>,
    }

    impl Rasterizer for Spy {
        fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
        fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
        fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
        fn dispatch_compute(&mut self, ctx: &ComputeContext<'_>) {
            self.order.push("dispatch");
            self.dispatches.push(Dispatch {
                program: ctx.program,
                grid: ctx.grid,
                block: ctx.block,
                shared_memory_size: ctx.shared_memory_size,
                const_buffers: *ctx.const_buffers,
                texture_headers: ctx.texture_headers,
                texture_header_limit: ctx.texture_header_limit,
                samplers: ctx.samplers,
                sampler_limit: ctx.sampler_limit,
                linked_samplers: ctx.linked_samplers,
                qmd_version: ctx.qmd_version,
                qmd: *ctx.qmd,
                indirect_grid: ctx.indirect_grid,
                indirect: ctx.indirect,
            });
        }
        fn inline_written(&mut self, _c: ChannelId, _m: &Gmmu, address: GpuVAddr, size: u64) {
            self.inline_writes.push((address, size));
        }
        fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, query: &Query) {
            self.order.push("query");
            self.queries.push(*query);
        }
        fn signal_fence(&mut self, action: FenceAction) {
            self.order.push("fence");
            self.fences.push(action);
        }
        fn release_fences(&mut self, _wait: bool) {}
        fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
        fn flush_region(&mut self, _addr: DAddr, _size: u64) {}
        fn wait_for_idle(&mut self) {}
        fn present(&mut self, _framebuffer: &FramebufferConfig) {}
    }

    struct Rig {
        harness: Harness,
        spy: Spy,
        engine: KeplerCompute,
    }

    impl Rig {
        fn new(class: u32) -> Self {
            Self {
                harness: Harness::new(),
                spy: Spy::default(),
                engine: KeplerCompute::new(class),
            }
        }

        fn call(&mut self, method: u32, argument: u32) {
            let mut ctx = EngineContext {
                channel: ChannelId(0),
                memory: &self.harness.memory,
                rasterizer: &mut self.spy,
                syncpoints: &self.harness.syncpoints,
                gob_layout: self.harness.gob_layout,
                qmd_scheduler: &mut self.harness.qmd_scheduler,
                written_pages: &self.harness.written_pages,
                activity: &crate::engines::testing::ACTIVITY,
            };
            self.engine.call_method(&mut ctx, method, argument);
        }

        fn put_qmd(&self, address: u64, words: &[u32; 64]) {
            self.harness
                .memory
                .write_block(address, bytemuck::bytes_of(words))
                .unwrap();
        }

        /// B1C0: LAUNCH. C7C0: SEND_SIGNALING_PCAS_B with INVALIDATE and SCHEDULE.
        fn launch(&mut self, qmd_address: u64) {
            self.call(reg_index!(Regs, launch_desc_loc), (qmd_address >> 8) as u32);
            let signal = match self.engine.qmd_version() {
                QmdVersion::V2 => 0,
                QmdVersion::V3 => 3,
            };
            self.call(reg_index!(Regs, launch), signal);
        }

        fn pcas2(&mut self, qmd_address: u64, argument: u32) {
            self.call(reg_index!(Regs, launch_desc_loc), (qmd_address >> 8) as u32);
            self.call(reg_index!(Regs, send_signaling_pcas2_b), argument);
        }

        fn load_inline(&mut self, qmd_address: u64, words: &[u32; 64]) {
            let address = reg_index!(Regs, inline_qmd_address);
            self.call(address, (qmd_address >> 40) as u32);
            self.call(address + 1, (qmd_address >> 8) as u32);
            for (index, &word) in words.iter().enumerate() {
                self.call(reg_index!(Regs, inline_qmd) + index as u32, word);
            }
        }

        fn report(&mut self, address: u64, payload: u32, control: u32) {
            let semaphore = reg_index!(Regs, report_semaphore);
            self.call(semaphore, (address >> 32) as u32);
            self.call(semaphore + 1, address as u32);
            self.call(semaphore + 2, payload);
            self.call(semaphore + 3, control);
        }

        fn word(&self, address: u64) -> u32 {
            self.harness.memory.read(address).unwrap()
        }
    }

    const QMD: u64 = 0x1000;

    fn v2_words() -> [u32; 64] {
        let mut w = [0u32; 64];
        w[8] = 0x120;
        w[11] = 1 << 30;
        w[12] = 7;
        w[13] = 3 | (5 << 16);
        w[17] = 0x1800;
        w[18] = 64 << 16;
        w[19] = 4 | (2 << 16);
        w[20] = 0b101;
        w[29] = 0x0010_0000;
        w[30] = 0x2 | (0x100 << 15);
        w[31] = 0x0020_0100;
        w[32] = 0x3 | (0x40 << 15);
        w[33] = 0x0030_0200;
        w[34] = 0x4 | (0x80 << 15);
        w
    }

    fn v3_words() -> [u32; 64] {
        let mut w = [0u32; 64];
        w[12] = 9;
        w[13] = 4;
        w[14] = 2;
        w[17] = 0x2000;
        w[18] = 0x30 | (128 << 16);
        w[19] = 2 | (1 << 16);
        w[20] = 0b10;
        w[34] = 0x0030_0000;
        w[35] = 0x1 | (0x20 << 19);
        w[48] = 0x0040_0000;
        w[49] = 0x1;
        w
    }

    /// Uploads a version 1.x grid of 7x3x5 to `target`, its data words from `source` when given.
    fn upload_grid(rig: &mut Rig, target: u64, source: Option<u64>) {
        let upload = reg_index!(Regs, upload);
        rig.call(upload, 8);
        rig.call(upload + 1, 1);
        rig.call(upload + 2, (target >> 32) as u32);
        rig.call(upload + 3, target as u32);
        rig.call(reg_index!(Regs, exec_upload), 1);
        if let Some(source) = source {
            rig.engine.upload_source(source);
        }
        rig.call(reg_index!(Regs, data_upload), 7);
        rig.call(reg_index!(Regs, data_upload), 3 | 5 << 16);
    }

    #[test]
    fn a_grid_uploaded_from_gpu_written_words_dispatches_indirectly() {
        const SOURCE: u64 = 0x5000;
        let grid = QMD + 0x30;
        let mut rig = Rig::new(0xb1c0);
        rig.put_qmd(QMD, &v2_words());
        upload_grid(&mut rig, grid, Some(SOURCE));
        rig.launch(QMD);
        let device = rig.harness.memory.translate(SOURCE).unwrap();
        rig.harness.written_pages.mark(device, 8);
        upload_grid(&mut rig, grid, Some(SOURCE + 0x10));
        rig.launch(QMD);
        upload_grid(&mut rig, QMD + 0x40, Some(SOURCE + 0x20));
        rig.launch(QMD);
        let dispatches: Vec<_> = rig
            .spy
            .dispatches
            .iter()
            .map(|dispatch| (dispatch.grid, dispatch.indirect_grid))
            .collect();
        assert_eq!(
            dispatches,
            [
                ([7, 3, 5], None),
                ([7, 3, 5], Some(SOURCE + 0x10)),
                ([7, 3, 5], None),
            ]
        );
    }

    #[test]
    fn a_grid_from_split_words_or_a_short_upload_stays_direct() {
        const SOURCE: u64 = 0x5000;
        let grid = QMD + 0x30;
        let mut rig = Rig::new(0xb1c0);
        rig.put_qmd(QMD, &v2_words());
        let device = rig.harness.memory.translate(SOURCE).unwrap();
        rig.harness.written_pages.mark(device, 0x100);
        let upload = reg_index!(Regs, upload);
        rig.call(upload, 8);
        rig.call(upload + 1, 1);
        rig.call(upload + 2, 0);
        rig.call(upload + 3, grid as u32);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.engine.upload_source(SOURCE);
        rig.call(reg_index!(Regs, data_upload), 7);
        rig.engine.upload_source(SOURCE + 8);
        rig.call(reg_index!(Regs, data_upload), 3 | 5 << 16);
        rig.launch(QMD);
        rig.call(upload, 4);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.engine.upload_source(SOURCE + 0x10);
        rig.call(reg_index!(Regs, data_upload), 7);
        rig.launch(QMD);
        rig.call(upload, 4);
        rig.call(upload + 1, 2);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.engine.upload_source(SOURCE + 0x20);
        rig.call(reg_index!(Regs, data_upload), 7);
        rig.call(reg_index!(Regs, data_upload), 3 | 5 << 16);
        rig.launch(QMD);
        let indirect: Vec<_> = rig.spy.dispatches.iter().map(|d| d.indirect_grid).collect();
        assert_eq!(indirect, [None, None, None]);
    }

    #[test]
    fn an_indirect_grid_dispatches_even_when_the_guest_copy_is_zero() {
        const SOURCE: u64 = 0x5000;
        let mut rig = Rig::new(0xb1c0);
        rig.put_qmd(QMD, &v2_words());
        let device = rig.harness.memory.translate(SOURCE).unwrap();
        rig.harness.written_pages.mark(device, 8);
        let upload = reg_index!(Regs, upload);
        rig.call(upload, 8);
        rig.call(upload + 1, 1);
        rig.call(upload + 2, 0);
        rig.call(upload + 3, (QMD + 0x30) as u32);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.engine.upload_source(SOURCE);
        rig.call(reg_index!(Regs, data_upload), 0);
        rig.call(reg_index!(Regs, data_upload), 0);
        rig.launch(QMD);
        rig.call(upload + 3, (QMD + 0x40) as u32);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.call(reg_index!(Regs, data_upload), 0);
        rig.call(reg_index!(Regs, data_upload), 0);
        rig.put_qmd(QMD, &{
            let mut words = v2_words();
            words[12] = 0;
            words
        });
        rig.launch(QMD);
        let dispatches: Vec<_> = rig
            .spy
            .dispatches
            .iter()
            .map(|d| (d.grid, d.indirect_grid))
            .collect();
        assert_eq!(dispatches, [([0, 0, 0], Some(SOURCE))]);
    }

    #[test]
    fn upload_sources_follow_the_data_words() {
        const SOURCE: u64 = 0x5000;
        let grid = QMD + 0x30;
        let mut rig = Rig::new(0xb1c0);
        rig.put_qmd(QMD, &v2_words());
        let device = rig.harness.memory.translate(SOURCE + 0x1000).unwrap();
        rig.harness.written_pages.mark(device, 4);
        upload_grid(&mut rig, QMD + 0x80, Some(SOURCE + 0x1000 - 8));
        upload_grid(&mut rig, grid, None);
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches[0].indirect_grid, Some(SOURCE + 0x1000));
    }

    /// The grid the renderer dispatches: the indirect words as GPU memory holds them now, or the
    /// descriptor's grid.
    fn resolved_grid(rig: &Rig, dispatch: &Dispatch) -> [u32; 3] {
        match dispatch.indirect {
            Some(grid) => grid.unpack(rig.harness.memory.read(grid.address).unwrap()),
            None => dispatch.grid,
        }
    }

    #[test]
    fn v2_and_v3_indirect_grids_resolve_in_their_own_layout() {
        const SOURCE: u64 = 0x5000;
        let packed = [1 << 31 | 11, 0xabcd << 16 | 6, 0x1234 << 16 | 2];

        let mut v2 = Rig::new(0xb1c0);
        v2.put_qmd(QMD, &v2_words());
        let device = v2.harness.memory.translate(SOURCE).unwrap();
        v2.harness.written_pages.mark(device, 8);
        upload_grid(&mut v2, QMD + 0x30, Some(SOURCE));
        v2.launch(QMD);
        v2.harness
            .memory
            .write_block(SOURCE, bytemuck::bytes_of(&packed))
            .unwrap();
        let dispatch = &v2.spy.dispatches[0];
        let grid = IndirectGrid {
            address: SOURCE,
            layout: QmdVersion::V2,
        };
        assert_eq!(
            (dispatch.indirect, dispatch.indirect_grid),
            (Some(grid), Some(SOURCE))
        );
        assert_eq!(resolved_grid(&v2, dispatch), [11, 6, 0xabcd]);

        let mut v3 = Rig::new(0xc7c0);
        v3.put_qmd(QMD, &v3_words());
        let device = v3.harness.memory.translate(QMD + QMD_V3_GRID).unwrap();
        v3.harness.written_pages.mark(device, 12);
        v3.launch(QMD);
        v3.harness
            .memory
            .write_block(QMD + QMD_V3_GRID, bytemuck::bytes_of(&packed))
            .unwrap();
        let dispatch = &v3.spy.dispatches[0];
        let grid = IndirectGrid {
            address: QMD + 0x30,
            layout: QmdVersion::V3,
        };
        assert_eq!(dispatch.grid, [9, 4, 2]);
        assert_eq!(
            (dispatch.indirect, dispatch.indirect_grid),
            (Some(grid), None)
        );
        assert_eq!(resolved_grid(&v3, dispatch), [1 << 31 | 11, 6, 2]);
    }

    #[test]
    fn a_v3_qmd_dispatches_indirectly_only_while_its_grid_may_be_gpu_written() {
        const OTHER_PAGE: u64 = QMD + 0x1000;
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &v3_words());
        rig.put_qmd(OTHER_PAGE, &v3_words());
        rig.launch(QMD);
        let device = rig.harness.memory.translate(QMD + QMD_V3_GRID).unwrap();
        rig.harness.written_pages.mark(device, 4);
        let mut zero_grid = v3_words();
        zero_grid[12] = 0;
        rig.load_inline(QMD, &zero_grid);
        rig.launch(OTHER_PAGE);
        let indirect = Some(IndirectGrid {
            address: QMD + QMD_V3_GRID,
            layout: QmdVersion::V3,
        });
        let dispatches: Vec<_> = rig
            .spy
            .dispatches
            .iter()
            .map(|d| (d.grid, d.indirect))
            .collect();
        assert_eq!(
            dispatches,
            [([9, 4, 2], None), ([0, 4, 2], indirect), ([9, 4, 2], None)]
        );
    }

    #[test]
    fn v2_launch_decodes_end_to_end() {
        let mut rig = Rig::new(0xb1c0);
        assert_eq!(rig.engine.qmd_version(), QmdVersion::V2);
        rig.call(reg_index!(Regs, code_loc), 0x1);
        rig.call(reg_index!(Regs, code_loc) + 1, 0x2000_0000);
        rig.call(reg_index!(Regs, tic), 0);
        rig.call(reg_index!(Regs, tic) + 1, 0x8000);
        rig.call(reg_index!(Regs, tic) + 2, 31);
        rig.call(reg_index!(Regs, tsc) + 1, 0x9000);
        rig.call(reg_index!(Regs, tsc) + 2, 15);
        let words = v2_words();
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        let mut cbufs = [ConstBufferBinding::default(); 8];
        cbufs[0] = ConstBufferBinding {
            address: 0x2_0010_0000,
            size: 0x100,
            enabled: true,
        };
        cbufs[1] = ConstBufferBinding {
            address: 0x3_0020_0100,
            size: 0x40,
            enabled: false,
        };
        cbufs[2] = ConstBufferBinding {
            address: 0x4_0030_0200,
            size: 0x80,
            enabled: true,
        };
        assert_eq!(
            rig.spy.dispatches,
            vec![Dispatch {
                program: 0x1_2000_0120,
                grid: [7, 3, 5],
                block: [64, 4, 2],
                shared_memory_size: 0x1800,
                const_buffers: cbufs,
                texture_headers: 0x8000,
                texture_header_limit: 31,
                samplers: 0x9000,
                sampler_limit: 15,
                linked_samplers: true,
                qmd_version: QmdVersion::V2,
                qmd: words,
                indirect_grid: None,
                indirect: None,
            }]
        );
        assert!(rig.spy.queries.is_empty());
    }

    #[test]
    fn v3_launch_decodes_end_to_end() {
        let mut rig = Rig::new(0xc7c0);
        assert_eq!(rig.engine.qmd_version(), QmdVersion::V3);
        let words = v3_words();
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        let mut cbufs = [ConstBufferBinding::default(); 8];
        cbufs[1] = ConstBufferBinding {
            address: 0x1_0030_0000,
            size: 0x200,
            enabled: true,
        };
        assert_eq!(
            rig.spy.dispatches,
            vec![Dispatch {
                program: 0x1_0040_0000,
                grid: [9, 4, 2],
                block: [128, 2, 1],
                shared_memory_size: 0x2000,
                const_buffers: cbufs,
                texture_headers: 0,
                texture_header_limit: 0,
                samplers: 0,
                sampler_limit: 0,
                linked_samplers: false,
                qmd_version: QmdVersion::V3,
                qmd: words,
                indirect_grid: None,
                indirect: None,
            }]
        );
    }

    #[test]
    fn v3_rejects_wrong_version_queue_and_wide_cbuf_addresses() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[18] = 0x20 | (128 << 16);
        words[26] = 7;
        words[25] = 1 << 23 | 1 << 30;
        words[24] = 0x2000;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert!(rig.spy.dispatches.is_empty());
        assert!(rig.spy.queries.is_empty());

        words = v3_words();
        words[4] = 1 << 8;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert!(rig.spy.dispatches.is_empty());

        words = v3_words();
        words[35] |= 1 << 8;
        words[25] = 1 << 23 | 1 << 30;
        words[24] = 0x2000;
        words[26] = 5;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert!(rig.spy.dispatches.is_empty());
        assert_eq!(rig.spy.queries.len(), 1);
    }

    #[test]
    fn empty_grid_skips_dispatch_but_still_releases() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[13] = 0;
        words[24] = 0x2000;
        words[25] = (1 << 23) | (1 << 30);
        words[26] = 0xfeed;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert!(rig.spy.dispatches.is_empty());
        assert_eq!(rig.spy.queries.len(), 1);
    }

    #[test]
    fn release_semaphore_follows_the_dispatch() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[24] = 0x2000;
        words[25] = (1 << 23) | (1 << 30);
        words[26] = 0xcafe;
        words[28] = 0x2010;
        words[29] = (1 << 23) | (1 << 30);
        words[30] = 0x1234;
        words[52] = 0x2040;
        words[53] = 1 << 23;
        words[54] = 0x77;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert_eq!(rig.spy.order, ["dispatch", "query", "query", "query"]);
        assert_eq!(
            rig.spy.queries,
            vec![
                Query {
                    address: 0x2000,
                    kind: QueryKind::Payload,
                    payload: 0xcafe,
                    size: ReleaseSize::Four,
                    after_work: true,
                },
                Query {
                    address: 0x2010,
                    kind: QueryKind::Payload,
                    payload: 0x1234,
                    size: ReleaseSize::Four,
                    after_work: true,
                },
                Query {
                    address: 0x2040,
                    kind: QueryKind::Payload,
                    payload: 0x77,
                    size: ReleaseSize::Sixteen,
                    after_work: true,
                },
            ]
        );
    }

    #[test]
    fn reduction_and_max_releases_are_fences_ordered_after_the_dispatch() {
        let mut rig = Rig::new(0xc7c0);
        rig.harness.memory.write(0x2000u64, 10u32).unwrap();
        let mut words = v3_words();
        words[24] = 0x2000;
        words[25] = (1 << 23) | (1 << 26) | (1 << 30);
        words[26] = 5;
        words[28] = 0x2100;
        words[29] = (1 << 23) | (1 << 19) | (1 << 30);
        words[30] = 20;
        words[52] = 0x2200;
        words[53] = (1 << 23) | (1 << 26) | (1 << 30) | (1 << 24) | (2 << 20);
        words[54] = 3;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        let device = |address| rig.harness.memory.translate(address).unwrap();
        assert_eq!(rig.spy.order, ["dispatch", "fence", "fence", "fence"]);
        assert_eq!(
            rig.spy.fences,
            vec![
                FenceAction::ReduceSemaphore {
                    address: device(0x2000),
                    op: Reduction::Add,
                    signed: false,
                    payload: 5,
                },
                FenceAction::ReduceSemaphore {
                    address: device(0x2100),
                    op: Reduction::Max,
                    signed: false,
                    payload: 20,
                },
                FenceAction::ReduceSemaphore {
                    address: device(0x2200),
                    op: Reduction::Max,
                    signed: true,
                    payload: 3,
                },
            ]
        );
        assert_eq!(rig.word(0x2000), 10);
        assert!(rig.spy.queries.is_empty());
    }

    #[test]
    fn reduction_on_unmapped_memory_is_skipped() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[24] = 0;
        words[25] = (1 << 23) | (1 << 26) | (1 << 30) | 0x7f;
        words[26] = 1;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches.len(), 1);
        assert!(rig.spy.fences.is_empty());
    }

    #[test]
    fn unsupported_release_formats_are_skipped() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[24] = 0x2000;
        words[25] = (1 << 23) | (1 << 29) | (1 << 30);
        words[26] = 1;
        words[28] = 0x2010;
        words[29] = (1 << 23) | (2 << 30);
        words[30] = 2;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches.len(), 1);
        assert!(rig.spy.queries.is_empty());
    }

    #[test]
    fn v2_descriptors_carry_no_releases() {
        let mut rig = Rig::new(0xb1c0);
        let mut words = v2_words();
        words[24] = 0x2000;
        words[25] = 1 << 23;
        rig.put_qmd(QMD, &words);
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches.len(), 1);
        assert!(rig.spy.queries.is_empty());
    }

    #[test]
    fn inline_upload_writes_through_the_compute_engine() {
        let mut rig = Rig::new(0xb1c0);
        let upload = reg_index!(Regs, upload);
        rig.call(upload, 8);
        rig.call(upload + 1, 1);
        rig.call(upload + 2, 0);
        rig.call(upload + 3, 0x3000);
        rig.call(reg_index!(Regs, exec_upload), 1);
        rig.call(reg_index!(Regs, data_upload), 0x0403_0201);
        rig.call(reg_index!(Regs, data_upload), 0x0807_0605);
        let mut out = [0u8; 8];
        rig.harness.memory.read_block(0x3000, &mut out).unwrap();
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(rig.spy.inline_writes, vec![(0x3000, 8)]);
        assert!(rig.spy.dispatches.is_empty());
    }

    #[test]
    fn a_descriptor_uploaded_inline_launches() {
        let mut rig = Rig::new(0xb1c0);
        let words = v2_words();
        let upload = reg_index!(Regs, upload);
        rig.call(upload, (QMD_WORDS * 4) as u32);
        rig.call(upload + 1, 1);
        rig.call(upload + 3, QMD as u32);
        rig.call(reg_index!(Regs, exec_upload), 1);
        for word in words {
            rig.call(reg_index!(Regs, data_upload), word);
        }
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches.len(), 1);
        assert_eq!(rig.spy.dispatches[0].grid, [7, 3, 5]);
    }

    const INLINE: u64 = 0x8000;
    const PCAS_SCHEDULE: u32 = 2;
    const PCAS_DECREMENT: u32 = 7;
    const PCAS_INCREMENT: u32 = 12;

    fn with_dependencies(mut words: [u32; 64], count: u32) -> [u32; 64] {
        words[4] |= count << 14;
        words
    }

    #[test]
    fn a_pcas_scheduled_qmd_dispatches_its_decoded_launch() {
        let mut rig = Rig::new(0xc7c0);
        let words = v3_words();
        rig.put_qmd(QMD, &words);
        rig.call(reg_index!(Regs, launch_desc_loc), (QMD >> 8) as u32);
        rig.call(reg_index!(Regs, launch), 0);
        rig.call(reg_index!(Regs, launch), 1);
        assert!(rig.spy.dispatches.is_empty());
        rig.pcas2(QMD, PCAS_SCHEDULE);
        let [dispatch] = &rig.spy.dispatches[..] else {
            panic!("one dispatch expected");
        };
        assert_eq!(dispatch.program, 0x1_0040_0000);
        assert_eq!(dispatch.grid, [9, 4, 2]);
        assert_eq!(dispatch.block, [128, 2, 1]);
        assert_eq!(dispatch.shared_memory_size, 0x2000);
        assert_eq!(dispatch.qmd_version, QmdVersion::V3);
        assert_eq!(dispatch.qmd, words);
    }

    #[test]
    fn a_qmd_waits_for_its_dependence_counter() {
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &with_dependencies(v3_words(), 3));
        rig.pcas2(QMD, PCAS_SCHEDULE);
        rig.pcas2(QMD, PCAS_DECREMENT | (1 << 14));
        rig.pcas2(QMD, PCAS_INCREMENT);
        rig.pcas2(QMD, PCAS_DECREMENT);
        assert!(rig.spy.dispatches.is_empty());
        rig.pcas2(QMD, PCAS_DECREMENT);
        assert_eq!(rig.spy.dispatches.len(), 1);
        rig.pcas2(QMD, PCAS_SCHEDULE);
        assert_eq!(rig.spy.dispatches.len(), 1);
    }

    #[test]
    fn a_decrement_before_the_schedule_is_kept() {
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &with_dependencies(v3_words(), 1));
        rig.pcas2(QMD, PCAS_DECREMENT);
        assert!(rig.spy.dispatches.is_empty());
        rig.pcas2(QMD, PCAS_SCHEDULE);
        assert_eq!(rig.spy.dispatches.len(), 1);
    }

    #[test]
    fn channels_share_one_qmd_table() {
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &with_dependencies(v3_words(), 1));
        rig.pcas2(QMD, PCAS_SCHEDULE);
        rig.engine = KeplerCompute::new(0xc7c0);
        rig.pcas2(QMD, PCAS_DECREMENT);
        assert_eq!(rig.spy.dispatches.len(), 1);
    }

    #[test]
    fn a_finished_qmd_acts_on_its_dependent() {
        const NEXT: u64 = 0x1200;
        let mut rig = Rig::new(0xc7c0);
        let mut first = v3_words();
        first[15] = (NEXT >> 8) as u32;
        first[16] = 1 | (4 << 1);
        first[24] = 0x2000;
        first[25] = (1 << 23) | (1 << 30);
        first[26] = 1;
        let mut second = with_dependencies(v3_words(), 1);
        second[12] = 5;
        rig.put_qmd(QMD, &first);
        rig.put_qmd(NEXT, &second);
        rig.pcas2(NEXT, PCAS_SCHEDULE);
        assert!(rig.spy.dispatches.is_empty());
        rig.pcas2(QMD, PCAS_SCHEDULE);
        assert_eq!(rig.spy.order, ["dispatch", "query", "dispatch"]);
        let grids: Vec<_> = rig.spy.dispatches.iter().map(|d| d.grid[0]).collect();
        assert_eq!(grids, [9, 5]);
    }

    #[test]
    fn a_second_dependent_stops_after_the_releases() {
        const NEXT: u64 = 0x1200;
        let mut rig = Rig::new(0xc7c0);
        let mut first = v3_words();
        first[15] = (NEXT >> 8) as u32;
        first[16] = 1 | (1 << 1) | (1 << 5);
        first[24] = 0x2000;
        first[25] = (1 << 23) | (1 << 30);
        rig.put_qmd(QMD, &first);
        rig.put_qmd(NEXT, &v3_words());
        rig.pcas2(QMD, PCAS_SCHEDULE);
        assert_eq!(rig.spy.order, ["dispatch", "query"]);
    }

    #[test]
    fn unsupported_pcas_actions_do_nothing() {
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &v3_words());
        for action in [0, 1, 6, 8, 13] {
            rig.pcas2(QMD, action);
        }
        assert!(rig.spy.dispatches.is_empty());
    }

    #[test]
    fn an_inline_qmd_is_written_and_dispatched() {
        let mut rig = Rig::new(0xc7c0);
        rig.call(reg_index!(Regs, launch_desc_loc), 0x77);
        let words = v3_words();
        rig.load_inline(INLINE, &words);
        let mut stored = [0u32; 64];
        rig.harness
            .memory
            .read_block(INLINE, bytemuck::bytes_of_mut(&mut stored))
            .unwrap();
        assert_eq!(stored, words);
        assert_eq!(rig.spy.inline_writes, vec![(INLINE, 256)]);
        let [dispatch] = &rig.spy.dispatches[..] else {
            panic!("one dispatch expected");
        };
        assert_eq!(dispatch.grid, [9, 4, 2]);
        assert_eq!(dispatch.qmd, words);
        assert_eq!(rig.engine.regs.launch_desc_loc.address, 0x77);
    }

    #[test]
    fn an_inline_qmd_can_wait_for_a_scheduling_pcas() {
        let mut rig = Rig::new(0xc7c0);
        let mut words = v3_words();
        words[4] |= 1 << 12;
        rig.load_inline(INLINE, &words);
        assert_eq!(rig.spy.inline_writes.len(), 1);
        assert!(rig.spy.dispatches.is_empty());
        rig.pcas2(INLINE, PCAS_SCHEDULE);
        assert_eq!(rig.spy.dispatches.len(), 1);
        rig.call(reg_index!(Regs, qmd_virtualization_control), 1 << 24);
        rig.load_inline(INLINE, &words);
        assert_eq!(rig.spy.dispatches.len(), 2);
    }

    #[test]
    fn the_report_semaphore_releases_after_the_dispatch() {
        let mut rig = Rig::new(0xc7c0);
        rig.put_qmd(QMD, &v3_words());
        rig.launch(QMD);
        rig.report(0x2000, 0xabcd, 1 << 28);
        rig.report(0x2010, 0x1234, 0);
        rig.report(0x2100, 3, (1 << 3) | (2 << 9) | (1 << 17));
        assert_eq!(rig.spy.order, ["dispatch", "query", "query", "fence"]);
        assert_eq!(
            rig.spy.queries,
            vec![
                Query {
                    address: 0x2000,
                    kind: QueryKind::Payload,
                    payload: 0xabcd,
                    size: ReleaseSize::Four,
                    after_work: true,
                },
                Query {
                    address: 0x2010,
                    kind: QueryKind::Payload,
                    payload: 0x1234,
                    size: ReleaseSize::Sixteen,
                    after_work: true,
                },
            ]
        );
        assert_eq!(
            rig.spy.fences,
            vec![FenceAction::ReduceSemaphore {
                address: rig.harness.memory.translate(0x2100).unwrap(),
                op: Reduction::Max,
                signed: true,
                payload: 3,
            }]
        );
    }

    #[test]
    fn report_semaphore_traps_are_skipped() {
        let mut rig = Rig::new(0xc7c0);
        rig.report(0x2000, 1, 3 | (1 << 28));
        assert!(rig.spy.queries.is_empty());
        assert!(rig.spy.fences.is_empty());
    }

    #[test]
    fn b1c0_ignores_the_native_compute_methods() {
        let mut rig = Rig::new(0xb1c0);
        rig.put_qmd(QMD, &v2_words());
        rig.pcas2(QMD, PCAS_SCHEDULE);
        rig.load_inline(INLINE, &v2_words());
        rig.report(0x2000, 1, 1 << 28);
        assert!(rig.spy.dispatches.is_empty());
        assert!(rig.spy.queries.is_empty());
        assert!(rig.spy.inline_writes.is_empty());
        assert_eq!(rig.word(INLINE), 0);
        rig.launch(QMD);
        assert_eq!(rig.spy.dispatches.len(), 1);
    }
}
