//! The compute engine (Kepler B1C0, Ampere C7C0).
//!
//! LAUNCH reads the launch descriptor (QMD) from the address `launch_desc_loc` holds, decodes it
//! as version 1.x (Maxwell) or 3.0 (Ampere) by the class the engine was created for, hands the
//! dispatch to the renderer, then performs the descriptor's release semaphores. Inline uploads
//! share the Kepler memory engine's registers. The native QMD scheduler (PCAS dependency counting,
//! inline QMD loads) is not part of this engine.

pub mod qmd;
pub mod regs;

use super::EngineContext;
use super::upload::Upload;
use crate::backend::{ComputeContext, FenceAction, QmdVersion, Query, QueryKind, ReleaseSize};
use crate::host::Reduction;
use crate::reg_index;
use qmd::{QMD_WORDS, QmdV2, QmdV3, Release, ReleaseKind};
pub use regs::Regs;

/// The compute engine bound to a subchannel.
pub struct KeplerCompute {
    pub regs: Box<Regs>,
    version: QmdVersion,
    upload: Upload,
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
        }
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
        if method == reg_index!(Regs, exec_upload) {
            self.upload
                .exec(&self.regs.upload, self.regs.exec_upload.linear() != 0);
        } else if method == reg_index!(Regs, data_upload) {
            self.upload.data(ctx, argument);
        } else if method == reg_index!(Regs, launch) {
            self.launch(ctx);
        }
    }

    fn launch(&self, ctx: &mut EngineContext<'_>) {
        let address = self.regs.launch_desc_loc.address();
        let mut words = [0u32; QMD_WORDS];
        if let Err(fault) = ctx
            .memory
            .read_block(address, bytemuck::bytes_of_mut(&mut words))
        {
            tracing::warn!(
                ?fault,
                address,
                "compute launch descriptor in unmapped memory"
            );
            return;
        }
        match self.version {
            QmdVersion::V2 => {
                let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV2>(&words);
                self.dispatch(ctx, qmd.launch(self.regs.code_loc.address()), &words);
            }
            QmdVersion::V3 => {
                let qmd = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV3>(&words);
                if let Err(error) = qmd.check() {
                    tracing::warn!(%error, address, "compute launch descriptor skipped");
                    return;
                }
                match qmd.launch() {
                    Ok(launch) => self.dispatch(ctx, launch, &words),
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
            }
        }
    }

    fn dispatch(&self, ctx: &mut EngineContext<'_>, launch: qmd::Launch, words: &[u32; QMD_WORDS]) {
        if launch.grid.contains(&0) || launch.block.contains(&0) {
            return;
        }
        let tic = &self.regs.tic;
        let tsc = &self.regs.tsc;
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
        });
    }
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
            };
            self.engine.call_method(&mut ctx, method, argument);
        }

        fn put_qmd(&self, address: u64, words: &[u32; 64]) {
            self.harness
                .memory
                .write_block(address, bytemuck::bytes_of(words))
                .unwrap();
        }

        fn launch(&mut self, qmd_address: u64) {
            self.call(reg_index!(Regs, launch_desc_loc), (qmd_address >> 8) as u32);
            self.call(reg_index!(Regs, launch), 0);
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
}
