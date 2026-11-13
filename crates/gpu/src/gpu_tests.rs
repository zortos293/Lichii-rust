//! End-to-end tests: synthetic pushbuffers in GPU memory, submitted through [`Gpu`] and run on
//! recording engines and a recording renderer, in both threading modes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use lichii_memory::{AddressSpace, MemoryPermission, PhysicalMemory};
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;
use parking_lot::Mutex;

use crate::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GraphicsContext, IndirectDraw, Query, QueryKind, Rasterizer, ReleaseSize,
};
use crate::channel::{ChannelConfig, GpEntry, Submission, SyncpointFence, SyncpointIncrement};
use crate::clock::{GpuClockSpeed, GuestTime};
use crate::engines::maxwell_3d::Regs;
use crate::engines::maxwell_3d::regs::PrimitiveTopology;
use crate::engines::{Engine, EngineContext, EngineKind, create_engine};
use crate::gmmu::{BigPageSize, Gmmu};
use crate::gpu::{Console, Gpu, GpuConfig, Threading};
use crate::host::HostClass;
use crate::host1x::Host1x;
use crate::host1x::Syncpoints;
use crate::pushbuffer::{CommandHeader, SubmissionMode};
use crate::reg_index;
use crate::stream::MIN_RING_SLOTS;

const CPU_VA: u64 = 0x1000_0000;
const MAPPED: u64 = 0x40_0000;
/// Device address of GPU VA 0.
const DEVICE_BASE: u64 = 0x4_0000_0000;

mod host {
    pub const SET_OBJECT: u32 = 0x00;
    pub const SEMAPHORE_A: u32 = 0x04;
    pub const SEMAPHORE_B: u32 = 0x05;
    pub const SEMAPHORE_C: u32 = 0x06;
    pub const SEMAPHORE_D: u32 = 0x07;
    pub const SET_REFERENCE: u32 = 0x14;
    pub const SEM_ADDR_LO: u32 = 0x17;
    pub const SYNCPOINT_A: u32 = 0x1c;
    pub const SYNCPOINT_B: u32 = 0x1d;
    pub const WFI: u32 = 0x1e;
}

const STATE_REGISTER: u32 = reg_index!(Regs, viewport_transform);

#[derive(Clone, Debug, PartialEq)]
enum Event {
    Method {
        channel: u32,
        engine: EngineKind,
        method: u32,
        argument: u32,
    },
    Multi {
        engine: EngineKind,
        method: u32,
        count: usize,
    },
    Release {
        address: u64,
        payload: u64,
    },
    Fence(FenceAction),
    WaitForIdle,
    InlineWritten(u64, u64),
    Invalidate(u64, u64),
    Ran,
    /// An engine finished the work it held back.
    Flush(EngineKind),
    FlushRegion(u64, u64),
    Draw {
        topology: PrimitiveTopology,
        count: u32,
    },
    Dispatch {
        grid: [u32; 3],
        indirect_grid: Option<u64>,
    },
}

/// Engines a rig binds: recording ones that log every method, or the real ones.
#[derive(Clone, Copy)]
enum Engines {
    Recording,
    Real,
}

type Log = Arc<Mutex<Vec<Event>>>;

struct RecordingEngine {
    engine: EngineKind,
    log: Log,
}

impl Engine for RecordingEngine {
    fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        self.log.lock().push(Event::Method {
            channel: ctx.channel.0,
            engine: self.engine,
            method,
            argument,
        });
    }

    fn call_multi(&mut self, ctx: &mut EngineContext<'_>, method: u32, arguments: &[u32]) {
        self.log.lock().push(Event::Multi {
            engine: self.engine,
            method,
            count: arguments.len(),
        });
        for &argument in arguments {
            self.call_method(ctx, method, argument);
        }
    }

    fn flush(&mut self, _ctx: &mut EngineContext<'_>) {
        self.log.lock().push(Event::Flush(self.engine));
    }
}

/// When the recording renderer's host GPU completes work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Completion {
    /// At once: fences and queries land when they are issued.
    Immediate,
    /// Only when the GPU waits for it (`release_fences(true)`, or a wait for idle and then
    /// `release_fences`) or the test completes it ([`Rig::complete`]), as on a real host GPU.
    Deferred,
}

#[derive(Clone, Copy)]
enum Pending {
    Fence(FenceAction),
    Query(Query),
}

/// The recording renderer's host GPU work not completed yet.
struct HostWork {
    completion: Completion,
    handles: GpuHandles,
    memory: Arc<Gmmu>,
    /// The pending work, and how much of it from the front finished.
    pending: Mutex<(Vec<Pending>, usize)>,
    /// How often the GPU asked whether fences are pending.
    fence_polls: AtomicUsize,
}

impl HostWork {
    fn submit(&self, work: Pending) {
        match self.completion {
            Completion::Immediate => self.run(work),
            Completion::Deferred => self.pending.lock().0.push(work),
        }
    }

    fn finish_all(&self) {
        let mut pending = self.pending.lock();
        pending.1 = pending.0.len();
    }

    /// Runs the finished work, or all of it.
    fn run_finished(&self, all: bool) {
        let finished: Vec<Pending> = {
            let mut pending = self.pending.lock();
            let count = if all { pending.0.len() } else { pending.1 };
            pending.1 = 0;
            pending.0.drain(..count).collect()
        };
        for work in finished {
            self.run(work);
        }
    }

    fn run(&self, work: Pending) {
        match work {
            Pending::Fence(action) => action.execute(&self.handles),
            Pending::Query(query) => {
                let memory = &self.memory;
                let result = match query.size {
                    ReleaseSize::Four => memory.write(query.address, query.payload as u32),
                    ReleaseSize::Eight => memory.write(query.address, query.payload),
                    ReleaseSize::Sixteen => memory.write(query.address, [query.payload, 0]),
                };
                result.unwrap();
            }
        }
    }
}

struct Recorder {
    log: Log,
    work: Arc<HostWork>,
}

impl Rasterizer for Recorder {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, draw: &DrawParams<'_>) {
        self.log.lock().push(Event::Draw {
            topology: draw.topology,
            count: draw.count,
        });
    }
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
    fn dispatch_compute(&mut self, ctx: &ComputeContext<'_>) {
        self.log.lock().push(Event::Dispatch {
            grid: ctx.grid,
            indirect_grid: ctx.indirect_grid,
        });
    }
    fn inline_written(&mut self, _channel: ChannelId, _memory: &Gmmu, address: u64, size: u64) {
        self.log.lock().push(Event::InlineWritten(address, size));
    }
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, query: &Query) {
        assert_eq!(query.kind, QueryKind::Payload);
        self.log.lock().push(Event::Release {
            address: query.address,
            payload: query.payload,
        });
        self.work.submit(Pending::Query(*query));
    }
    fn signal_fence(&mut self, action: FenceAction) {
        self.log.lock().push(Event::Fence(action));
        self.work.submit(Pending::Fence(action));
    }
    fn release_fences(&mut self, wait: bool) {
        self.work.run_finished(wait);
    }
    fn has_pending_fences(&self) -> bool {
        self.work.fence_polls.fetch_add(1, Ordering::Relaxed);
        !self.work.pending.lock().0.is_empty()
    }
    fn invalidate_region(&mut self, addr: u64, size: u64) {
        self.log.lock().push(Event::Invalidate(addr, size));
    }
    fn flush_region(&mut self, addr: u64, size: u64) {
        self.log.lock().push(Event::FlushRegion(addr, size));
    }
    fn wait_for_idle(&mut self) {
        self.log.lock().push(Event::WaitForIdle);
        self.work.finish_all();
    }
    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}

struct Rig {
    gpu: Gpu,
    memory: Arc<Gmmu>,
    syncpoints: Arc<Syncpoints>,
    log: Log,
    work: Arc<HostWork>,
    next_pushbuffer: u64,
    space: Arc<AddressSpace>,
    time: Arc<TestTime>,
}

impl Rig {
    fn new(threading: Threading) -> Self {
        Self::build(
            threading,
            crate::gpu::DEFAULT_RING_SLOTS,
            Completion::Immediate,
        )
    }

    fn deferred(threading: Threading) -> Self {
        Self::build(
            threading,
            crate::gpu::DEFAULT_RING_SLOTS,
            Completion::Deferred,
        )
    }

    fn with_ring(threading: Threading, ring_slots: usize) -> Self {
        Self::build(threading, ring_slots, Completion::Immediate)
    }

    fn build(threading: Threading, ring_slots: usize, completion: Completion) -> Self {
        Self::build_for(
            Console::Switch1,
            threading,
            ring_slots,
            completion,
            Engines::Recording,
        )
    }

    fn build_for(
        console: Console,
        threading: Threading,
        ring_slots: usize,
        completion: Completion,
        engines: Engines,
    ) -> Self {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let space = Arc::new(AddressSpace::new(physical.clone(), 39));
        space.map(CPU_VA, 0x10_0000, MAPPED, MemoryPermission::READ_WRITE);
        let host1x = Host1x::new(physical, 36);
        host1x.memory.attach_process(space.clone());
        host1x
            .memory
            .map_process(DEVICE_BASE, &space, CPU_VA, MAPPED)
            .unwrap();
        let memory = Arc::new(Gmmu::new(host1x.memory.clone(), BigPageSize::Size64K));
        memory.map(0, DEVICE_BASE, MAPPED, 0, false).unwrap();
        let syncpoints = host1x.syncpoints.clone();
        let log = Log::default();
        let engines_log = log.clone();
        let mut work = None;
        let time = Arc::new(TestTime::default());
        let gpu = Gpu::new(
            host1x,
            time.clone(),
            GpuConfig {
                threading,
                ring_slots,
                engines: Arc::new(move |engine, class| match (engines, engine) {
                    (Engines::Real, _) | (_, EngineKind::InlineToMemory) => {
                        create_engine(engine, class)
                    }
                    _ => Some(Box::new(RecordingEngine {
                        engine,
                        log: engines_log.clone(),
                    })),
                }),
                ..GpuConfig::new(console)
            },
            |handles| {
                let host = Arc::new(HostWork {
                    completion,
                    handles,
                    memory: memory.clone(),
                    pending: Mutex::default(),
                    fence_polls: AtomicUsize::new(0),
                });
                work = Some(host.clone());
                Box::new(Recorder {
                    log: log.clone(),
                    work: host,
                })
            },
        );
        Self {
            gpu,
            memory,
            syncpoints,
            log,
            work: work.expect("the renderer was created"),
            next_pushbuffer: 0x1_0000,
            space,
            time,
        }
    }

    /// The host GPU finishes the work it holds and runs its fences, on the test's thread.
    fn complete(&self) {
        self.work.run_finished(true);
    }

    fn channel(&self, host_class: HostClass) -> ChannelId {
        self.gpu
            .create_channel(ChannelConfig::new(self.memory.clone(), host_class))
    }

    /// Writes `words` to fresh GPU memory and returns their GPFIFO entry.
    fn entry(&mut self, words: &[u32]) -> GpEntry {
        let address = self.next_pushbuffer;
        self.next_pushbuffer += (words.len() as u64 * 4).next_multiple_of(0x100) + 0x100;
        self.memory
            .write_block(address, bytemuck::cast_slice(words))
            .unwrap();
        GpEntry::new(address, words.len() as u32)
    }

    fn submit(&mut self, channel: ChannelId, words: &[u32]) {
        let entry = self.entry(words);
        self.gpu.submit(channel, Submission::new(&[entry]));
    }

    fn events(&self) -> Vec<Event> {
        self.gpu.synchronize();
        self.log.lock().clone()
    }

    /// The engine methods executed, one per word.
    fn methods(&self) -> Vec<(u32, EngineKind, u32, u32)> {
        self.events()
            .into_iter()
            .filter_map(|event| match event {
                Event::Method {
                    channel,
                    engine,
                    method,
                    argument,
                } => Some((channel, engine, method, argument)),
                _ => None,
            })
            .collect()
    }

    fn read(&self, address: u64) -> u32 {
        self.memory.read::<u32>(address).unwrap()
    }
}

#[derive(Default)]
struct Pushbuffer(Vec<u32>);

impl Pushbuffer {
    fn command(mut self, mode: SubmissionMode, subchannel: u32, method: u32, data: &[u32]) -> Self {
        self.0
            .push(CommandHeader::new(mode, subchannel, method, data.len() as u32).0);
        self.0.extend_from_slice(data);
        self
    }

    fn inc(self, subchannel: u32, method: u32, data: &[u32]) -> Self {
        self.command(SubmissionMode::Increasing, subchannel, method, data)
    }

    fn non_inc(self, subchannel: u32, method: u32, data: &[u32]) -> Self {
        self.command(SubmissionMode::NonIncreasing, subchannel, method, data)
    }

    fn once(self, subchannel: u32, method: u32, data: &[u32]) -> Self {
        self.command(SubmissionMode::IncreaseOnce, subchannel, method, data)
    }

    fn host(self, method: u32, argument: u32) -> Self {
        self.inc(0, method, &[argument])
    }

    fn bind(self, subchannel: u32, class: u32) -> Self {
        self.inc(subchannel, host::SET_OBJECT, &[class])
    }

    /// Ampere SEM_ADDR/SEM_PAYLOAD/SEM_EXECUTE.
    fn semaphore(self, address: u64, payload: u32, execute: u32) -> Self {
        self.inc(
            0,
            host::SEM_ADDR_LO,
            &[address as u32, (address >> 32) as u32, payload, 0, execute],
        )
    }

    /// Uploads macro code through LOAD_MME and binds macro 0 to its start.
    fn load_macro(self, subchannel: u32, code: &[u32]) -> Self {
        self.inc(
            subchannel,
            crate::macros::LOAD_MME_INSTRUCTION_RAM_POINTER,
            &[0],
        )
        .non_inc(subchannel, crate::macros::LOAD_MME_INSTRUCTION_RAM, code)
        .inc(
            subchannel,
            crate::macros::LOAD_MME_START_ADDRESS_RAM_POINTER,
            &[0, 0],
        )
    }

    /// An inline-to-memory upload of `data` to `address` on subchannel 2.
    fn upload(self, address: u64, data: &[u32]) -> Self {
        self.inc(
            2,
            0x60,
            &[
                data.len() as u32 * 4,
                1,
                (address >> 32) as u32,
                address as u32,
            ],
        )
        .inc(2, 0x6c, &[1])
        .non_inc(2, 0x6d, data)
    }
}

fn each_mode(test: impl Fn(Threading)) {
    for threading in [Threading::Split, Threading::Single] {
        test(threading);
    }
}

const MAXWELL_3D: u32 = 0xb197;
const AMPERE_3D: u32 = 0xc797;

#[test]
fn methods_reach_their_subchannels_in_order() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .bind(1, 0xb1c0)
            .bind(3, 0x902d)
            .bind(4, 0xb0b5)
            .inc(0, STATE_REGISTER, &[1, 2])
            .inc(1, 0x200, &[3])
            .command(SubmissionMode::Inline, 4, 0x300, &[])
            .inc(3, 0x400, &[5])
            .inc(5, 0x500, &[6])
            .non_inc(0, 0x700, &[7, 8, 9])
            .inc(0, STATE_REGISTER, &[1])
            .inc(0, 0x701, &[10]);
        rig.submit(channel, &words.0);
        let c = channel.0;
        assert_eq!(
            rig.methods(),
            [
                (c, EngineKind::Maxwell3d, STATE_REGISTER, 1),
                (c, EngineKind::Maxwell3d, STATE_REGISTER + 1, 2),
                (c, EngineKind::Compute, 0x200, 3),
                (c, EngineKind::Dma, 0x300, 0),
                (c, EngineKind::Fermi2d, 0x400, 5),
                (c, EngineKind::Maxwell3d, 0x700, 7),
                (c, EngineKind::Maxwell3d, 0x700, 8),
                (c, EngineKind::Maxwell3d, 0x700, 9),
                (c, EngineKind::Maxwell3d, 0x701, 10),
            ],
            "{threading:?}"
        );
        assert!(rig.events().contains(&Event::Multi {
            engine: EngineKind::Maxwell3d,
            method: 0x700,
            count: 3
        }));
    });
}

#[test]
fn a_bound_3d_class_draws_through_the_real_engine() {
    each_mode(|threading| {
        let mut rig = Rig::build_for(
            Console::Switch1,
            threading,
            crate::gpu::DEFAULT_RING_SLOTS,
            Completion::Immediate,
            Engines::Real,
        );
        let channel = rig.channel(HostClass::Maxwell);
        let pushbuffer = Pushbuffer::default()
            .bind(0, 0xb197)
            .once(0, reg_index!(Regs, vertex_buffer.count), &[3])
            .once(
                0,
                reg_index!(Regs, draw.begin),
                &[PrimitiveTopology::TRIANGLES.0],
            )
            .once(0, reg_index!(Regs, draw.end), &[0]);
        rig.submit(channel, &pushbuffer.0);
        rig.gpu.synchronize();
        let draws: Vec<_> = rig
            .events()
            .into_iter()
            .filter(|event| matches!(event, Event::Draw { .. }))
            .collect();
        assert_eq!(
            draws,
            [Event::Draw {
                topology: PrimitiveTopology::TRIANGLES,
                count: 3,
            }],
            "{threading:?}"
        );
    });
}

#[test]
fn subchannel_2_is_inline_to_memory_without_a_bind() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default().upload(0x20_0000, &[0x1122_3344, 0x5566_7788]);
        rig.submit(channel, &words.0);
        assert!(rig.events().contains(&Event::InlineWritten(0x20_0000, 8)));
        assert_eq!(rig.read(0x20_0000), 0x1122_3344);
        assert_eq!(rig.read(0x20_0004), 0x5566_7788);
    });
}

/// Fermi macro assembly (the subset these tests use).
mod fasm {
    pub const MOVE_SET_METHOD: u32 = 2;
    pub const FETCH_SEND: u32 = 3;
    pub const MOVE_SEND: u32 = 4;

    pub fn addi(result: u32, dst: u32, a: u32, imm: i32) -> u32 {
        1 | (result << 4) | (dst << 8) | (a << 11) | ((imm as u32 & 0x3_ffff) << 14)
    }

    pub fn exit(word: u32) -> u32 {
        word | 0x80
    }

    /// Sends parameter 0 and 1 to methods 0x100.., with 5 and parameter 1 + 1 between.
    pub fn program() -> [u32; 6] {
        [
            addi(MOVE_SET_METHOD, 0, 0, 0x100 | 1 << 12),
            addi(MOVE_SEND, 0, 1, 0),
            addi(FETCH_SEND, 2, 0, 5),
            exit(addi(MOVE_SEND, 0, 2, 0)),
            addi(MOVE_SEND, 0, 2, 1),
            addi(MOVE_SEND, 0, 0, 0x777),
        ]
    }
}

#[test]
fn macro_upload_and_call_emit_methods() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .load_macro(0, &fasm::program())
            .once(0, crate::macros::MACRO_METHOD_BASE, &[0xaa, 0xbb]);
        rig.submit(channel, &words.0);
        let c = channel.0;
        assert_eq!(
            rig.methods(),
            [
                (c, EngineKind::Maxwell3d, 0x100, 0xaa),
                (c, EngineKind::Maxwell3d, 0x101, 5),
                (c, EngineKind::Maxwell3d, 0x102, 0xbb),
                (c, EngineKind::Maxwell3d, 0x103, 0xbc),
            ],
            "{threading:?}"
        );
    });
}

#[test]
fn methods_and_macro_calls_continue_across_entries() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let setup = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .load_macro(0, &fasm::program());
        let entries = vec![
            rig.entry(&setup.0),
            rig.entry(&[
                CommandHeader::new(SubmissionMode::NonIncreasing, 0, 0x700, 3).0,
                1,
            ]),
            rig.entry(&[
                2,
                3,
                CommandHeader::new(
                    SubmissionMode::IncreaseOnce,
                    0,
                    crate::macros::MACRO_METHOD_BASE,
                    2,
                )
                .0,
                0x11,
            ]),
            rig.entry(&[0x22]),
        ];
        rig.gpu.submit(channel, Submission::new(&entries));
        let c = channel.0;
        let d3 = EngineKind::Maxwell3d;
        assert_eq!(
            rig.methods(),
            [
                (c, d3, 0x700, 1),
                (c, d3, 0x700, 2),
                (c, d3, 0x700, 3),
                (c, d3, 0x100, 0x11),
                (c, d3, 0x101, 5),
                (c, d3, 0x102, 0x22),
                (c, d3, 0x103, 0x23),
            ],
            "{threading:?}"
        );
    });
}

const SEMAPHORE: u64 = 0x30_0000;

#[test]
fn cross_channel_acquire_suspends_until_the_release() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let releaser = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .inc(0, 0x700, &[1])
            .semaphore(SEMAPHORE, 1, 2)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        let w = waiter.0;
        assert_eq!(
            rig.methods(),
            [(w, EngineKind::Maxwell3d, 0x700, 1)],
            "{threading:?}"
        );
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .inc(0, 0x702, &[3])
            .semaphore(SEMAPHORE, 1, 1);
        rig.submit(releaser, &words.0);
        let methods = rig.methods();
        assert_eq!(
            methods[1..],
            [
                (releaser.0, EngineKind::Maxwell3d, 0x702, 3),
                (w, EngineKind::Maxwell3d, 0x701, 2),
            ],
            "{threading:?}"
        );
        assert_eq!(rig.read(SEMAPHORE), 1);
    });
}

#[test]
fn acquire_latches_a_release_overwritten_before_the_front_looks() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let releaser = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(SEMAPHORE, 1, 0)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        assert!(rig.methods().is_empty());
        let words = Pushbuffer::default()
            .semaphore(SEMAPHORE, 1, 1)
            .semaphore(SEMAPHORE, 2, 1);
        rig.submit(releaser, &words.0);
        assert_eq!(rig.methods(), [(waiter.0, EngineKind::Maxwell3d, 0x701, 2)]);
        assert_eq!(rig.read(SEMAPHORE), 2);
    });
}

#[test]
fn acquire_released_by_the_cpu_resumes() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .host(host::SEMAPHORE_A, (SEMAPHORE >> 32) as u32)
            .host(host::SEMAPHORE_B, SEMAPHORE as u32)
            .host(host::SEMAPHORE_C, 5)
            .host(host::SEMAPHORE_D, 4)
            .inc(0, 0x701, &[2])
            .host(host::SEMAPHORE_C, 9)
            .host(host::SEMAPHORE_D, 2 | 1 << 24);
        rig.submit(channel, &words.0);
        assert!(rig.methods().is_empty());
        rig.memory.write(SEMAPHORE, 7u32).unwrap();
        rig.gpu.wake();
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Maxwell3d, 0x701, 2)]
        );
        assert_eq!(rig.read(SEMAPHORE), 9);
        assert_eq!(rig.read(SEMAPHORE + 4), 0);
    });
}

fn increments(events: &[Event]) -> Vec<Event> {
    events
        .iter()
        .filter(|e| {
            matches!(
                e,
                Event::Fence(FenceAction::IncrementSyncpoint(_)) | Event::WaitForIdle
            )
        })
        .cloned()
        .collect()
}

#[test]
fn switch_1_fence_increments_after_a_wait_for_idle() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 8;
        let words = Pushbuffer::default()
            .host(host::SYNCPOINT_A, 0)
            .host(host::SYNCPOINT_B, id << 8 | 1);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(
            channel,
            Submission {
                entries: &[entry],
                wait: None,
                increment: Some(SyncpointIncrement::fence_increment(id, false)),
            },
        );
        let fence = Event::Fence(FenceAction::IncrementSyncpoint(id));
        assert_eq!(
            increments(&rig.events()),
            [fence.clone(), Event::WaitForIdle, fence.clone(), fence],
            "{threading:?}"
        );
        assert_eq!(rig.syncpoints.guest().value(id), Ok(3));
        assert_eq!(rig.syncpoints.host().value(id), Ok(3));
    });
}

#[test]
fn switch_2_increment_value_counts_the_fence_value() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Ampere);
        let id = 0x31;
        let words = Pushbuffer::default().semaphore(SEMAPHORE, 1, 1);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(
            channel,
            Submission {
                entries: &[entry],
                wait: None,
                increment: Some(SyncpointIncrement::increment_value(id, 4)),
            },
        );
        let events = rig.events();
        let fence = Event::Fence(FenceAction::IncrementSyncpoint(id));
        assert_eq!(
            increments(&events),
            [
                Event::WaitForIdle,
                fence.clone(),
                fence.clone(),
                fence.clone(),
                fence
            ],
            "{threading:?}"
        );
        assert_eq!(
            events[0],
            Event::Fence(FenceAction::WriteSemaphore {
                address: DEVICE_BASE + SEMAPHORE,
                value: 1,
                size: ReleaseSize::Four,
            })
        );
        assert_eq!(rig.syncpoints.guest().value(id), Ok(4));
    });
}

#[test]
fn syncpoint_wait_holds_the_submission_until_reached() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 5;
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .inc(0, 0x700, &[1])
            .host(host::SYNCPOINT_A, 2)
            .host(host::SYNCPOINT_B, id << 8)
            .inc(0, 0x701, &[2]);
        let entry = rig.entry(&words.0);
        rig.syncpoints.increase_max(id, 2).unwrap();
        rig.gpu.submit(
            channel,
            Submission {
                entries: &[entry],
                wait: Some(SyncpointFence { id, value: 1 }),
                increment: None,
            },
        );
        assert!(rig.methods().is_empty());
        rig.syncpoints.host().increment(id).unwrap();
        let c = channel.0;
        assert_eq!(rig.methods(), [(c, EngineKind::Maxwell3d, 0x700, 1)]);
        rig.syncpoints.host().increment(id).unwrap();
        assert_eq!(
            rig.methods(),
            [
                (c, EngineKind::Maxwell3d, 0x700, 1),
                (c, EngineKind::Maxwell3d, 0x701, 2)
            ]
        );
    });
}

#[test]
fn a_syncpoint_wait_past_every_reserved_increment_does_not_hold_the_channel() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 5;
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .inc(0, 0x700, &[1])
            .host(host::SYNCPOINT_A, 2)
            .host(host::SYNCPOINT_B, id << 8)
            .inc(0, 0x701, &[2]);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(
            channel,
            Submission {
                entries: &[entry],
                wait: Some(SyncpointFence { id: 0x40, value: 1 }),
                increment: None,
            },
        );
        let c = channel.0;
        assert_eq!(
            rig.methods(),
            [
                (c, EngineKind::Maxwell3d, 0x700, 1),
                (c, EngineKind::Maxwell3d, 0x701, 2)
            ],
            "{threading:?}"
        );
    });
}

#[test]
fn a_syncpoint_wait_follows_a_suspended_channels_increments_without_their_host_gpu_work() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let producer = rig.channel(HostClass::Maxwell);
        let consumer = rig.channel(HostClass::Maxwell);
        let id = 9;
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .host(host::SEMAPHORE_A, (SEMAPHORE >> 32) as u32)
            .host(host::SEMAPHORE_B, SEMAPHORE as u32)
            .host(host::SEMAPHORE_C, 1)
            .host(host::SEMAPHORE_D, 1)
            .inc(0, 0x700, &[1]);
        let entry = rig.entry(&words.0);
        let fence = rig
            .gpu
            .submit(
                producer,
                Submission {
                    entries: &[entry],
                    wait: None,
                    increment: Some(SyncpointIncrement::fence_increment(id, false)),
                },
            )
            .unwrap();
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .inc(0, 0x701, &[2]);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(
            consumer,
            Submission {
                entries: &[entry],
                wait: Some(fence),
                increment: None,
            },
        );
        assert!(rig.methods().is_empty(), "{threading:?}");
        rig.memory.write(SEMAPHORE, 1u32).unwrap();
        rig.gpu.wake();
        assert_eq!(
            rig.methods(),
            [
                (producer.0, EngineKind::Maxwell3d, 0x700, 1),
                (consumer.0, EngineKind::Maxwell3d, 0x701, 2)
            ],
            "{threading:?}"
        );
        assert_eq!(rig.syncpoints.host().value(id), Ok(fence.value));
        assert_eq!(rig.syncpoints.guest().value(id), Ok(0));
    });
}

#[test]
fn wait_for_idle_and_reference_count_run_in_order() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let words = Pushbuffer::default()
            .bind(0, MAXWELL_3D)
            .inc(0, 0x700, &[1])
            .host(host::WFI, 0)
            .inc(0, 0x701, &[2])
            .host(host::SET_REFERENCE, 7);
        rig.submit(channel, &words.0);
        let log = rig.log.clone();
        rig.gpu.run_in_order(move |_| log.lock().push(Event::Ran));
        let c = channel.0;
        let events: Vec<Event> = rig
            .events()
            .into_iter()
            .filter(|e| !matches!(e, Event::Multi { .. } | Event::Flush(_)))
            .collect();
        assert_eq!(
            events,
            [
                Event::Method {
                    channel: c,
                    engine: EngineKind::Maxwell3d,
                    method: 0x700,
                    argument: 1
                },
                Event::WaitForIdle,
                Event::Method {
                    channel: c,
                    engine: EngineKind::Maxwell3d,
                    method: 0x701,
                    argument: 2
                },
                Event::Ran,
            ],
            "{threading:?}"
        );
        assert_eq!(rig.gpu.reference_count(channel), Some(7));
    });
}

/// MME64 assembly (the subset these tests use).
mod masm {
    pub const ADD: u8 = 0;
    pub const EXTENDED: u8 = 7;
    pub const ZERO: u8 = 24;
    pub const IMM: u8 = 25;
    pub const LOAD0: u8 = 28;
    pub const LOAD1: u8 = 29;
    pub const NONE: u8 = 0;
    pub const ALU0: u8 = 1;
    pub const ALU1: u8 = 2;
    pub const OUT_LOAD0: u8 = 3;
    pub const OUT_LOAD1: u8 = 4;
    pub const IMM0: u8 = 5;
    pub const IMM1: u8 = 6;

    #[derive(Clone, Copy)]
    pub struct Alu(u8, u8, u8, u8, u16);

    pub fn alu(op: u8, dst: u8, a: u8, b: u8, imm: u16) -> Alu {
        Alu(op, dst, a, b, imm)
    }

    pub fn imm(imm: u16) -> Alu {
        alu(ADD, ZERO, ZERO, ZERO, imm)
    }

    pub fn nop() -> Alu {
        imm(0)
    }

    #[derive(Clone, Copy)]
    pub struct Inst {
        end: bool,
        alu: [Alu; 2],
        out: [(u8, u8); 2],
    }

    pub fn inst(alu0: Alu, alu1: Alu) -> Inst {
        Inst {
            end: false,
            alu: [alu0, alu1],
            out: [(NONE, NONE); 2],
        }
    }

    impl Inst {
        pub fn out0(mut self, method: u8, emit: u8) -> Self {
            self.out[0] = (method, emit);
            self
        }

        pub fn out1(mut self, method: u8, emit: u8) -> Self {
            self.out[1] = (method, emit);
            self
        }

        pub fn end(mut self) -> Self {
            self.end = true;
            self
        }

        fn encode(&self) -> [u32; 3] {
            let mut bits = u128::from(self.end);
            for (lane, &Alu(op, dst, a, b, imm)) in self.alu.iter().enumerate() {
                let base = 10 + 36 * lane as u32;
                bits |= u128::from(op) << base;
                bits |= u128::from(dst) << (base + 5);
                bits |= u128::from(a) << (base + 10);
                bits |= u128::from(b) << (base + 15);
                bits |= u128::from(imm) << (base + 20);
            }
            for (lane, &(method, emit)) in self.out.iter().enumerate() {
                let base = 82 + 7 * lane as u32;
                bits |= u128::from(method) << base;
                bits |= u128::from(emit) << (base + 3);
            }
            [(bits >> 64) as u32, (bits >> 32) as u32, bits as u32]
        }
    }

    /// Writes `value` to `method`, then ends.
    pub fn emit_program(method: u16, value: u16) -> Vec<u32> {
        [
            inst(imm(method), imm(value)).out0(IMM0, IMM1).end(),
            inst(nop(), nop()),
        ]
        .iter()
        .flat_map(Inst::encode)
        .collect()
    }

    /// Reads `count` words at the address in parameters 0 (high) and 1 (low) through
    /// MME_DMA_READ_FIFOED and sends them to methods 0xc00...
    pub fn dma_read_program(count: u16) -> Vec<u32> {
        [
            inst(imm(0x1154), nop())
                .out0(IMM0, OUT_LOAD0)
                .out1(NONE, OUT_LOAD1),
            inst(imm(0x0158), imm(count)).out0(IMM0, IMM1),
            inst(alu(EXTENDED, ZERO, IMM, IMM, 0x1000), nop()),
            inst(
                alu(ADD, 0, LOAD0, ZERO, 0x1c00),
                alu(ADD, 1, LOAD1, ZERO, 0),
            )
            .out0(IMM0, ALU0)
            .out1(NONE, ALU1)
            .end(),
            inst(nop(), nop()),
        ]
        .iter()
        .flat_map(Inst::encode)
        .collect()
    }
}

#[test]
fn mme_dma_read_sees_a_preceding_back_write() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Ampere);
        let data = 0x24_0000;
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .load_macro(0, &masm::dma_read_program(2))
            .upload(data, &[0x11, 0x22])
            .once(0, crate::macros::MACRO_METHOD_BASE, &[0, data as u32]);
        rig.submit(channel, &words.0);
        let read: Vec<_> = rig
            .methods()
            .into_iter()
            .filter(|&(_, _, method, _)| method >= 0xc00)
            .collect();
        assert_eq!(
            read,
            [
                (channel.0, EngineKind::Maxwell3d, 0xc00, 0x11),
                (channel.0, EngineKind::Maxwell3d, 0xc01, 0x22),
            ],
            "{threading:?}"
        );
    });
}

#[test]
fn a_small_ring_wraps_and_keeps_order() {
    each_mode(|threading| {
        let mut rig = Rig::with_ring(threading, MIN_RING_SLOTS);
        let channel = rig.channel(HostClass::Maxwell);
        let data: Vec<u32> = (1..=10_000).collect();
        let mut words = Pushbuffer::default().bind(0, MAXWELL_3D);
        for chunk in data.chunks(5000) {
            words = words.non_inc(0, 0x700, chunk);
            for &value in chunk.iter().take(3000) {
                words = words.inc(0, 0x701, &[value]);
            }
        }
        rig.submit(channel, &words.0);
        let mut expected = Vec::new();
        for chunk in data.chunks(5000) {
            expected.extend(chunk.iter().map(|&v| (0x700, v)));
            expected.extend(chunk.iter().take(3000).map(|&v| (0x701, v)));
        }
        let methods: Vec<_> = rig
            .methods()
            .into_iter()
            .map(|(_, _, method, argument)| (method, argument))
            .collect();
        assert_eq!(methods, expected, "{threading:?}");
    });
}

/// SEM_EXECUTE: the acquire or release compares or writes 64 bits.
const WIDE: u32 = 1 << 24;

#[test]
fn a_waiter_resumes_only_once_its_deferred_release_lands() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let releaser = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(SEMAPHORE, 1, 0)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        rig.submit(
            releaser,
            &Pushbuffer::default().semaphore(SEMAPHORE, 1, 1).0,
        );
        assert!(rig.methods().is_empty(), "{threading:?}");
        assert_eq!(rig.read(SEMAPHORE), 0);
        rig.complete();
        assert_eq!(
            rig.methods(),
            [(waiter.0, EngineKind::Maxwell3d, 0x701, 2)],
            "{threading:?}"
        );
        assert_eq!(rig.read(SEMAPHORE), 1);
    });
}

#[test]
fn deferred_releases_latch_a_value_overwritten_before_the_front_looks() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let releaser = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(SEMAPHORE, 1, 0)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        let words = Pushbuffer::default()
            .semaphore(SEMAPHORE, 1, 1)
            .semaphore(SEMAPHORE, 2, 1);
        rig.submit(releaser, &words.0);
        assert!(rig.methods().is_empty(), "{threading:?}");
        rig.complete();
        assert_eq!(
            rig.methods(),
            [(waiter.0, EngineKind::Maxwell3d, 0x701, 2)],
            "{threading:?}"
        );
        assert_eq!(rig.read(SEMAPHORE), 2);
    });
}

#[test]
fn a_narrow_release_does_not_meet_a_wide_acquire() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let releaser = rig.channel(HostClass::Ampere);
        rig.memory.write(SEMAPHORE + 4, 5u32).unwrap();
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(SEMAPHORE, 1, WIDE)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        rig.submit(
            releaser,
            &Pushbuffer::default().semaphore(SEMAPHORE, 1, 1).0,
        );
        assert!(rig.methods().is_empty(), "{threading:?}");
        assert_eq!(rig.memory.read::<u64>(SEMAPHORE).unwrap(), 0x5_0000_0001);
        let words = Pushbuffer::default().semaphore(SEMAPHORE, 1, 1 | WIDE);
        rig.submit(releaser, &words.0);
        assert_eq!(
            rig.methods(),
            [(waiter.0, EngineKind::Maxwell3d, 0x701, 2)],
            "{threading:?}"
        );
    });
}

#[test]
fn mme_dma_read_waits_for_deferred_work_and_flushes_what_it_reads() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let channel = rig.channel(HostClass::Ampere);
        let data = 0x24_0000;
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .load_macro(0, &masm::dma_read_program(2))
            .semaphore(data, 0x55, 1)
            .once(0, crate::macros::MACRO_METHOD_BASE, &[0, data as u32]);
        rig.submit(channel, &words.0);
        let events = rig.events();
        let read: Vec<_> = rig
            .methods()
            .into_iter()
            .filter(|&(_, _, method, _)| method >= 0xc00)
            .collect();
        assert_eq!(
            read,
            [(channel.0, EngineKind::Maxwell3d, 0xc00, 0x55)],
            "{threading:?}"
        );
        let flush = events
            .iter()
            .position(|e| *e == Event::FlushRegion(DEVICE_BASE + data, 8))
            .expect("the read range is flushed");
        assert_eq!(events[flush - 1], Event::Flush(EngineKind::Maxwell3d));
        let first_read = events
            .iter()
            .position(|e| matches!(e, Event::Method { method: 0xc00, .. }))
            .unwrap();
        assert!(flush < first_read, "{threading:?}");
    });
}

#[test]
fn set_object_of_an_unknown_class_unbinds_and_class_0_keeps_the_binding() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let words = Pushbuffer::default()
            .bind(1, 0xb1c0)
            .inc(1, 0x200, &[1])
            .bind(1, 0)
            .inc(1, 0x200, &[2])
            .bind(0, MAXWELL_3D)
            .bind(0, 0x1234)
            .inc(0, 0x700, &[3])
            .bind(2, 0)
            .upload(0x20_0000, &[0x1122_3344]);
        rig.submit(channel, &words.0);
        assert_eq!(
            rig.methods(),
            [
                (channel.0, EngineKind::Compute, 0x200, 1),
                (channel.0, EngineKind::Compute, 0x200, 2)
            ],
            "{threading:?}"
        );
        assert_eq!(rig.read(0x20_0000), 0x1122_3344);
    });
}

#[derive(Default)]
struct TestTime(AtomicU64);

impl GuestTime for TestTime {
    fn global_time_ns(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

#[test]
fn switch_2_address_spaces_get_the_identity_mappings_until_destroyed() {
    let rig = Rig::build_for(
        Console::Switch2,
        Threading::Split,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Immediate,
        Engines::Recording,
    );
    let nx2 = rig.gpu.nx2_memory().expect("Switch 2 has the registry");
    let early = rig.gpu.create_address_space(BigPageSize::Size64K);
    let (first, _) = nx2
        .create_gpu_accessible_memory(&rig.space, CPU_VA, 0x1_0000)
        .unwrap();
    let late = rig.gpu.create_address_space(BigPageSize::Size128K);
    rig.space.write(CPU_VA + 0x120, 0x5eed_u32).unwrap();
    let daddr = nx2.memory_info(first).unwrap().daddr;
    for gmmu in [&early, &late] {
        assert_eq!(gmmu.translate(CPU_VA + 0x120), Some(daddr + 0x120));
        assert_eq!(gmmu.read::<u32>(CPU_VA + 0x120), Ok(0x5eed));
    }
    rig.gpu.destroy_address_space(&early);
    nx2.create_gpu_accessible_memory(&rig.space, CPU_VA + 0x10_0000, 0x1_0000)
        .unwrap();
    assert_eq!(early.translate(CPU_VA + 0x10_0000), None);
    assert!(late.translate(CPU_VA + 0x10_0000).is_some());
    assert_eq!(early.translate(CPU_VA + 0x120), Some(daddr + 0x120));
}

/// A Switch 2 rig with a two-page queue semaphore memory mapped at [`REGION_VA`, +0x2000).
const REGION_VA: u64 = 0x100_0000;

fn rig_with_semaphore_region(threading: Threading) -> Rig {
    let rig = Rig::build_for(
        Console::Switch2,
        threading,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Immediate,
        Engines::Recording,
    );
    let nx2 = rig.gpu.nx2_memory().expect("Switch 2 has the registry");
    let (handle, _) = nx2
        .create_gpu_accessible_memory(&rig.space, CPU_VA + 0x20_0000, 0x2000)
        .unwrap();
    let daddr = nx2.memory_info(handle).unwrap().daddr;
    rig.memory.map(REGION_VA, daddr, 0x2000, 0, false).unwrap();
    rig
}

#[test]
fn switch_2_queue_semaphore_releases_reach_the_first_page() {
    each_mode(|threading| {
        let mut rig = rig_with_semaphore_region(threading);
        let channel = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default().semaphore(REGION_VA + 0x1010, 5, 1);
        rig.submit(channel, &words.0);
        rig.gpu.synchronize();
        assert_eq!(rig.read(REGION_VA + 0x1010), 5, "{threading:?}");
        assert_eq!(rig.read(REGION_VA + 0x10), 5, "{threading:?}");
        assert_eq!(rig.read(0x1010), 0, "{threading:?}");
    });
}

#[test]
fn a_mirrored_value_stays_beside_a_word_with_releases_of_its_own() {
    each_mode(|threading| {
        let mut rig = rig_with_semaphore_region(threading);
        let releaser = rig.channel(HostClass::Ampere);
        let waiter = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .semaphore(REGION_VA + 0x20, 9, 1)
            .semaphore(REGION_VA + 0x1020, 6, 1);
        rig.submit(releaser, &words.0);
        rig.gpu.synchronize();
        assert_eq!(rig.read(REGION_VA + 0x20), 9, "{threading:?}");
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(REGION_VA + 0x20, 6, 0)
            .inc(0, 0x701, &[2]);
        rig.submit(waiter, &words.0);
        assert_eq!(
            rig.methods(),
            [(waiter.0, EngineKind::Maxwell3d, 0x701, 2)],
            "{threading:?}"
        );
    });
}

#[test]
fn switch_1_has_no_nx2_registry_and_plain_address_spaces() {
    let rig = Rig::new(Threading::Split);
    assert!(rig.gpu.nx2_memory().is_none());
    let gmmu = rig.gpu.create_address_space(BigPageSize::Size64K);
    assert_eq!(gmmu.translate(0), None);
    rig.gpu.destroy_address_space(&gmmu);
}

#[test]
fn channel_defaults_follow_the_console() {
    let rig = Rig::new(Threading::Split);
    let switch1 = ChannelConfig::for_console(rig.memory.clone(), Console::Switch1);
    assert_eq!(switch1.host_class, HostClass::Maxwell);
    assert_eq!(switch1.gob_layout, GobLayout::Tegra);
    let switch2 = ChannelConfig::for_console(rig.memory.clone(), Console::Switch2);
    assert_eq!(switch2.host_class, HostClass::Ampere);
    assert_eq!(switch2.gob_layout, GobLayout::Switch2);
}

#[test]
fn bind_address_space_switches_memory_for_later_work_only() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let other = rig.gpu.create_address_space(BigPageSize::Size64K);
        other.map(0, DEVICE_BASE, 0x20_0000, 0, false).unwrap();
        other
            .map(0x20_0000, DEVICE_BASE + 0x30_0000, 0x10_0000, 0, false)
            .unwrap();
        let channel = rig.channel(HostClass::Maxwell);
        let id = 9;
        let first = rig.entry(&Pushbuffer::default().upload(0x20_0000, &[1]).0);
        rig.syncpoints.increase_max(id, 1).unwrap();
        rig.gpu.submit(
            channel,
            Submission {
                entries: &[first],
                wait: Some(SyncpointFence { id, value: 1 }),
                increment: None,
            },
        );
        rig.gpu.bind_address_space(channel, other.clone());
        let second = Pushbuffer::default().upload(0x20_0000, &[2]).0;
        other
            .write_block(0x28_0000, bytemuck::cast_slice(&second))
            .unwrap();
        rig.gpu.submit(
            channel,
            Submission::new(&[GpEntry::new(0x28_0000, second.len() as u32)]),
        );
        rig.gpu.synchronize();
        assert_eq!(rig.read(0x20_0000), 0, "{threading:?}");
        rig.syncpoints.host().increment(id).unwrap();
        rig.gpu.synchronize();
        assert_eq!(rig.read(0x20_0000), 1, "{threading:?}");
        assert_eq!(rig.read(0x30_0000), 2, "{threading:?}");
    });
}

#[test]
fn submit_reserves_increasing_fences_that_execution_reaches() {
    each_mode(|threading| {
        let rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 12;
        let increment = |count| Submission {
            entries: &[],
            wait: None,
            increment: Some(SyncpointIncrement::increment_value(id, count)),
        };
        assert_eq!(
            rig.gpu.submit(channel, increment(2)),
            Some(SyncpointFence { id, value: 2 })
        );
        assert_eq!(rig.gpu.submit(channel, Submission::new(&[])), None);
        assert_eq!(
            rig.gpu.submit(channel, increment(3)),
            Some(SyncpointFence { id, value: 5 })
        );
        assert_eq!(rig.syncpoints.max(id), Ok(5));
        rig.gpu.synchronize();
        assert_eq!(rig.syncpoints.guest().value(id), Ok(5), "{threading:?}");
        assert_eq!(
            rig.gpu.submit(
                channel,
                Submission {
                    entries: &[],
                    wait: None,
                    increment: Some(SyncpointIncrement::increment_value(0x400, 1)),
                },
            ),
            None
        );
    });
}

#[test]
fn submit_to_a_destroyed_channel_reserves_nothing() {
    each_mode(|threading| {
        let rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 12;
        let increment = || Submission {
            entries: &[],
            wait: None,
            increment: Some(SyncpointIncrement::increment_value(id, 2)),
        };
        assert_eq!(
            rig.gpu.submit(channel, increment()),
            Some(SyncpointFence { id, value: 2 })
        );
        rig.gpu.destroy_channel(channel);
        assert_eq!(rig.gpu.submit(channel, increment()), None);
        assert_eq!(rig.syncpoints.max(id), Ok(2));
        rig.gpu.synchronize();
        assert_eq!(rig.syncpoints.guest().value(id), Ok(2), "{threading:?}");
    });
}

#[test]
fn submissions_reuse_the_pooled_entry_buffer() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let entry = rig.entry(&Pushbuffer::default().upload(0x20_0000, &[7]).0);
        let entries = [entry; 8];
        rig.gpu.submit(channel, Submission::new(&entries));
        rig.gpu.synchronize();
        let pooled = rig.gpu.pooled_entry_buffers();
        assert!(
            matches!(pooled[..], [capacity] if capacity >= 8),
            "{pooled:?}"
        );
        rig.gpu.submit(channel, Submission::new(&entries[..4]));
        rig.gpu.synchronize();
        assert_eq!(rig.gpu.pooled_entry_buffers(), pooled, "{threading:?}");
        assert_eq!(rig.read(0x20_0000), 7);
    });
}

#[test]
fn shutdown_joins_the_gpu_once_and_drops_later_commands() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let entry = rig.entry(&Pushbuffer::default().upload(0x20_0000, &[7]).0);
        rig.gpu.submit(channel, Submission::new(&[entry]));
        rig.gpu.synchronize();
        rig.gpu.shutdown();
        rig.gpu.shutdown();
        let id = 12;
        let late = Submission {
            entries: &[],
            wait: None,
            increment: Some(SyncpointIncrement::increment_value(id, 1)),
        };
        assert_eq!(rig.gpu.submit(channel, late), None);
        assert_eq!(rig.syncpoints.max(id), Ok(0));
        rig.gpu.synchronize();
        assert_eq!(rig.read(0x20_0000), 7, "{threading:?}");
    });
}

#[test]
fn shutdown_from_a_gpu_callback_during_an_outside_shutdown_finishes() {
    each_mode(|threading| {
        let gpu = Arc::new(Rig::new(threading).gpu);
        let (started, wait_started) = std::sync::mpsc::channel();
        let (go, wait_go) = std::sync::mpsc::channel::<()>();
        let inner = gpu.clone();
        gpu.run_in_order(move |_| {
            started.send(()).unwrap();
            let _ = wait_go.recv();
            inner.shutdown();
        });
        wait_started.recv().unwrap();
        let outer = gpu.clone();
        let outside = std::thread::spawn(move || outer.shutdown());
        std::thread::sleep(std::time::Duration::from_millis(50));
        go.send(()).unwrap();
        outside.join().unwrap();
    });
}

#[test]
fn pause_holds_submissions_until_resumed() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        rig.gpu.synchronize();
        rig.gpu.pause(true);
        let entry = rig.entry(&Pushbuffer::default().upload(0x20_0000, &[7]).0);
        let fence = rig.gpu.submit(
            channel,
            Submission {
                entries: &[entry],
                wait: None,
                increment: Some(SyncpointIncrement::increment_value(3, 1)),
            },
        );
        rig.gpu.wake();
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(rig.read(0x20_0000), 0, "{threading:?}");
        assert_eq!(rig.syncpoints.guest().value(3), Ok(0), "{threading:?}");
        assert_eq!(fence, Some(SyncpointFence { id: 3, value: 1 }));
        rig.gpu.pause(false);
        rig.gpu.synchronize();
        assert_eq!(rig.read(0x20_0000), 7, "{threading:?}");
        assert_eq!(rig.syncpoints.guest().value(3), Ok(1), "{threading:?}");
    });
}

#[test]
fn present_restarts_the_switch_2_timestamp_frame_in_order() {
    each_mode(|threading| {
        let rig = Rig::build_for(
            Console::Switch2,
            threading,
            crate::gpu::DEFAULT_RING_SLOTS,
            Completion::Immediate,
            Engines::Recording,
        );
        let timestamp = rig.gpu.handles().timestamp.clone();
        timestamp.set_speed(GpuClockSpeed::Boost);
        rig.time.0.store(1_000, Ordering::Relaxed);
        rig.gpu.present(FramebufferConfig {
            address: DEVICE_BASE,
            width: 16,
            height: 16,
            stride: 64,
            format: PixelFormat::ALL[0],
            block_height_log2: None,
            gob_layout: GobLayout::Switch2,
            flip_vertical: false,
            crop: None,
        });
        rig.gpu.synchronize();
        rig.time.0.store(1_400, Ordering::Relaxed);
        assert_eq!(timestamp.now(), 1_100, "{threading:?}");
    });
}

/// A Switch 1 compute dispatch whose 7x3x5 grid an inline upload in the same entry writes into
/// its descriptor; the upload's data words are marked GPU-written when `written`. Returns the
/// dispatch events and the GPU address of the data words.
fn dispatch_with_uploaded_grid(threading: Threading, written: bool) -> (Vec<Event>, u64) {
    use crate::engines::kepler_compute::Regs as ComputeRegs;
    const QMD: u64 = 0x30_0000;
    let mut rig = Rig::build_for(
        Console::Switch1,
        threading,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Immediate,
        Engines::Real,
    );
    let channel = rig.channel(HostClass::Maxwell);
    let mut qmd = [0u32; 64];
    qmd[18] = 64 << 16;
    qmd[19] = 4 | 2 << 16;
    rig.memory
        .write_block(QMD, bytemuck::bytes_of(&qmd))
        .unwrap();
    let grid = QMD + 0x30;
    let head = Pushbuffer::default()
        .bind(1, 0xb1c0)
        .inc(
            1,
            reg_index!(ComputeRegs, launch_desc_loc),
            &[(QMD >> 8) as u32],
        )
        .inc(
            1,
            reg_index!(ComputeRegs, upload),
            &[8, 1, (grid >> 32) as u32, grid as u32],
        )
        .inc(1, reg_index!(ComputeRegs, exec_upload), &[1]);
    let data = (head.0.len() as u64 + 1) * 4;
    let words = head
        .non_inc(1, reg_index!(ComputeRegs, data_upload), &[7, 3 | 5 << 16])
        .inc(1, reg_index!(ComputeRegs, launch), &[0])
        .0;
    let entry = rig.entry(&words);
    if written {
        rig.gpu
            .handles()
            .written_pages
            .mark(DEVICE_BASE + entry.address() + data, 8);
    }
    rig.gpu.submit(channel, Submission::new(&[entry]));
    rig.gpu.synchronize();
    let dispatches = rig
        .events()
        .into_iter()
        .filter(|e| matches!(e, Event::Dispatch { .. }))
        .collect();
    (dispatches, entry.address() + data)
}

#[test]
fn switch_1_dispatches_a_grid_uploaded_from_gpu_written_commands_indirectly() {
    each_mode(|threading| {
        let (events, data) = dispatch_with_uploaded_grid(threading, true);
        assert_eq!(
            events,
            [Event::Dispatch {
                grid: [7, 3, 5],
                indirect_grid: Some(data),
            }],
            "{threading:?}"
        );
        let (events, _) = dispatch_with_uploaded_grid(threading, false);
        assert_eq!(
            events,
            [Event::Dispatch {
                grid: [7, 3, 5],
                indirect_grid: None,
            }],
            "{threading:?}"
        );
    });
}

/// Sets the entry's SYNC bit.
fn sync(entry: GpEntry) -> GpEntry {
    GpEntry(entry.0 | 1 << 63)
}

/// A command list and the entry before it, whose release overwrites the list's argument word
/// 0xdead with 0x1234, as compute or copy work writes a command list: (writer, list).
fn list_written_by_a_release(rig: &mut Rig) -> (GpEntry, GpEntry) {
    let list = rig.entry(&Pushbuffer::default().inc(0, 0x701, &[0xdead]).0);
    let writer = rig.entry(
        &Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .semaphore(list.address() + 4, 0x1234, 1)
            .0,
    );
    (writer, list)
}

/// A rig whose channels fetch the entries of [`list_written_by_a_release`] after the work before
/// them completed on the host GPU: Switch 2, with the list's words on a page GPU work wrote.
fn rig_with_written_list(threading: Threading) -> (Rig, GpEntry, GpEntry) {
    let mut rig = Rig::build_for(
        Console::Switch2,
        threading,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Deferred,
        Engines::Recording,
    );
    let (writer, list) = list_written_by_a_release(&mut rig);
    rig.gpu
        .handles()
        .written_pages
        .mark(DEVICE_BASE + list.address() + 4, 4);
    (rig, writer, list)
}

#[test]
fn a_sync_entry_is_fetched_after_the_back_ran_the_work_before_it_but_not_the_host_gpu() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let list = rig.entry(&Pushbuffer::default().inc(0, 0x701, &[0xdead]).0);
        let writer = rig.entry(
            &Pushbuffer::default()
                .bind(0, MAXWELL_3D)
                .host(host::SEMAPHORE_A, (SEMAPHORE >> 32) as u32)
                .host(host::SEMAPHORE_B, SEMAPHORE as u32)
                .host(host::SEMAPHORE_C, 7)
                .host(host::SEMAPHORE_D, 2 | 1 << 24)
                .upload(list.address() + 4, &[0x1234])
                .0,
        );
        rig.gpu
            .submit(channel, Submission::new(&[writer, sync(list)]));
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Maxwell3d, 0x701, 0x1234)],
            "{threading:?}: the list is read after the back wrote it"
        );
        assert!(
            !rig.events()
                .iter()
                .any(|e| matches!(e, Event::FlushRegion(..))),
            "{threading:?}"
        );
        assert_eq!(
            rig.read(SEMAPHORE),
            0,
            "{threading:?}: the fetch did not wait for the host GPU"
        );
        rig.complete();
        assert_eq!(rig.read(SEMAPHORE), 7);
    });
}

#[test]
fn a_switch_2_sync_entry_on_a_page_the_work_before_it_marks_waits_for_the_host_gpu() {
    each_mode(|threading| {
        let mut rig = Rig::build_for(
            Console::Switch2,
            threading,
            crate::gpu::DEFAULT_RING_SLOTS,
            Completion::Deferred,
            Engines::Recording,
        );
        let channel = rig.channel(HostClass::Ampere);
        let (writer, list) = list_written_by_a_release(&mut rig);
        rig.gpu.submit(channel, Submission::new(&[writer]));
        let pages = rig.gpu.handles().written_pages.clone();
        let page = DEVICE_BASE + list.address() + 4;
        rig.gpu.run_in_order(move |_| pages.mark(page, 4));
        rig.gpu.submit(channel, Submission::new(&[sync(list)]));
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Maxwell3d, 0x701, 0x1234)],
            "{threading:?}: the list is read with the released word"
        );
        assert!(
            rig.events()
                .contains(&Event::FlushRegion(DEVICE_BASE + list.address(), 8)),
            "{threading:?}"
        );
    });
}

#[test]
fn entries_without_sync_are_fetched_without_waiting() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let channel = rig.channel(HostClass::Ampere);
        let (writer, list) = list_written_by_a_release(&mut rig);
        rig.gpu.submit(channel, Submission::new(&[writer, list]));
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Maxwell3d, 0x701, 0xdead)],
            "{threading:?}"
        );
        assert!(
            !rig.events()
                .iter()
                .any(|e| matches!(e, Event::FlushRegion(..))),
            "{threading:?}"
        );
        assert_eq!(rig.read(list.address() + 4), 0xdead);
        rig.complete();
        assert_eq!(rig.read(list.address() + 4), 0x1234);
    });
}

fn gpu_written_list(console: Console, threading: Threading) -> (Rig, ChannelId, GpEntry) {
    let mut rig = Rig::build_for(
        console,
        threading,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Deferred,
        Engines::Recording,
    );
    let channel = rig.channel(HostClass::Ampere);
    let (writer, list) = list_written_by_a_release(&mut rig);
    rig.gpu
        .handles()
        .written_pages
        .mark(DEVICE_BASE + list.address() + 4, 4);
    rig.gpu.submit(channel, Submission::new(&[writer, list]));
    (rig, channel, list)
}

#[test]
fn switch_2_fetches_an_entry_on_gpu_written_pages_after_the_work_before_it() {
    each_mode(|threading| {
        let (rig, channel, list) = gpu_written_list(Console::Switch2, threading);
        let events = rig.events();
        let read = events
            .iter()
            .position(|e| {
                *e == Event::Method {
                    channel: channel.0,
                    engine: EngineKind::Maxwell3d,
                    method: 0x701,
                    argument: 0x1234,
                }
            })
            .unwrap_or_else(|| panic!("{threading:?}: the list is read with the released word"));
        let flush = events
            .iter()
            .position(|e| *e == Event::FlushRegion(DEVICE_BASE + list.address(), 8))
            .unwrap_or_else(|| panic!("{threading:?}: the list's words are flushed"));
        assert!(flush < read, "{threading:?}");
    });
}

#[test]
fn switch_1_ignores_gpu_written_pages() {
    each_mode(|threading| {
        let (rig, channel, _) = gpu_written_list(Console::Switch1, threading);
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Maxwell3d, 0x701, 0xdead)],
            "{threading:?}"
        );
    });
}

#[test]
fn side_commands_wait_behind_a_sync_entry_queued_before_them() {
    each_mode(|threading| {
        let (rig, writer, list) = rig_with_written_list(threading);
        let waiter = rig.channel(HostClass::Ampere);
        rig.gpu.pause(true);
        rig.gpu
            .submit(waiter, Submission::new(&[writer, sync(list)]));
        let log = rig.log.clone();
        rig.gpu.run_in_order(move |_| log.lock().push(Event::Ran));
        rig.gpu.pause(false);
        rig.gpu.synchronize();
        let order: Vec<_> = rig
            .events()
            .into_iter()
            .filter(|event| matches!(event, Event::Ran | Event::Method { method: 0x701, .. }))
            .collect();
        assert!(
            matches!(order[..], [Event::Method { .. }, Event::Ran]),
            "{threading:?}: {order:?}"
        );
    });
}

#[test]
fn submissions_after_a_held_side_command_keep_their_place() {
    each_mode(|threading| {
        let (mut rig, writer, list) = rig_with_written_list(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let other = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default().bind(0, AMPERE_3D).inc(0, 0x702, &[2]);
        rig.gpu.pause(true);
        rig.gpu
            .submit(waiter, Submission::new(&[writer, sync(list)]));
        let log = rig.log.clone();
        rig.gpu.run_in_order(move |_| log.lock().push(Event::Ran));
        rig.submit(other, &words.0);
        rig.gpu.pause(false);
        rig.gpu.synchronize();
        let order: Vec<_> = rig
            .events()
            .into_iter()
            .filter_map(|event| match event {
                Event::Ran => Some(0),
                Event::Method { method, .. } if method == 0x701 || method == 0x702 => Some(method),
                _ => None,
            })
            .collect();
        assert_eq!(order, [0x701, 0, 0x702], "{threading:?}");
    });
}

#[test]
fn a_sync_entry_suspends_only_its_channel() {
    each_mode(|threading| {
        let (mut rig, writer, list) = rig_with_written_list(threading);
        let waiter = rig.channel(HostClass::Ampere);
        let other = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default().bind(0, AMPERE_3D).inc(0, 0x702, &[2]);
        rig.gpu.pause(true);
        rig.gpu
            .submit(waiter, Submission::new(&[writer, sync(list)]));
        rig.submit(other, &words.0);
        rig.gpu.pause(false);
        assert_eq!(
            rig.methods(),
            [
                (other.0, EngineKind::Maxwell3d, 0x702, 2),
                (waiter.0, EngineKind::Maxwell3d, 0x701, 0x1234),
            ],
            "{threading:?}"
        );
    });
}

#[test]
fn a_syncpoint_increment_lands_once_the_host_gpu_finishes_without_more_work() {
    each_mode(|threading| {
        let mut rig = Rig::deferred(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 9;
        let words = Pushbuffer::default()
            .host(host::SYNCPOINT_A, 0)
            .host(host::SYNCPOINT_B, id << 8 | 1);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(channel, Submission::new(&[entry]));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let landed = loop {
            rig.work.finish_all();
            let poll = Some(std::time::Duration::from_millis(1));
            match rig.syncpoints.guest().wait(id, 1, poll) {
                Ok(value) => break Ok(value),
                Err(error) if std::time::Instant::now() >= deadline => break Err(error),
                Err(_) => {}
            }
        };
        assert_eq!(landed, Ok(1), "{threading:?}");
    });
}

#[test]
fn an_idle_gpu_with_no_pending_fences_sleeps() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Maxwell);
        let id = 10;
        let words = Pushbuffer::default()
            .host(host::SYNCPOINT_A, 0)
            .host(host::SYNCPOINT_B, id << 8 | 1);
        let entry = rig.entry(&words.0);
        rig.gpu.submit(channel, Submission::new(&[entry]));
        rig.gpu.synchronize();
        assert_eq!(rig.syncpoints.guest().value(id), Ok(1), "{threading:?}");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let before = rig.work.fence_polls.load(Ordering::Relaxed);
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert_eq!(
            rig.work.fence_polls.load(Ordering::Relaxed),
            before,
            "{threading:?}: the GPU polls fences while none are pending"
        );
    });
}

#[test]
fn switch_2_compute_macros_keep_their_own_code_beside_the_3d_macros() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Ampere);
        let call = |words: Pushbuffer, subchannel| {
            words.once(subchannel, crate::macros::MACRO_METHOD_BASE, &[0])
        };
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .bind(1, 0xc7c0)
            .load_macro(0, &masm::emit_program(0x701, 0x3d))
            .load_macro(1, &masm::emit_program(0x200, 0xc0));
        let words = call(call(words, 0), 1);
        rig.submit(channel, &words.0);
        assert_eq!(
            rig.methods(),
            [
                (channel.0, EngineKind::Maxwell3d, 0x701, 0x3d),
                (channel.0, EngineKind::Compute, 0x200, 0xc0),
            ],
            "{threading:?}"
        );
    });
}

#[test]
fn switch_2_compute_calls_run_the_3d_macros_until_compute_uploads_its_own() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let channel = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .bind(1, 0xc7c0)
            .load_macro(0, &masm::emit_program(0x200, 0x3d))
            .once(1, crate::macros::MACRO_METHOD_BASE, &[0]);
        rig.submit(channel, &words.0);
        assert_eq!(
            rig.methods(),
            [(channel.0, EngineKind::Compute, 0x200, 0x3d)],
            "{threading:?}"
        );
    });
}

#[test]
fn one_channels_compute_upload_leaves_other_channels_on_the_3d_macros() {
    each_mode(|threading| {
        let mut rig = Rig::new(threading);
        let uploader = rig.channel(HostClass::Ampere);
        let caller = rig.channel(HostClass::Ampere);
        let words = Pushbuffer::default()
            .bind(1, 0xc7c0)
            .load_macro(1, &masm::emit_program(0x200, 0xc0));
        rig.submit(uploader, &words.0);
        let words = Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .bind(1, 0xc7c0)
            .load_macro(0, &masm::emit_program(0x200, 0x3d))
            .once(1, crate::macros::MACRO_METHOD_BASE, &[0]);
        rig.submit(caller, &words.0);
        assert_eq!(
            rig.methods(),
            [(caller.0, EngineKind::Compute, 0x200, 0x3d)],
            "{threading:?}"
        );
    });
}

/// A CPU write to a page a GPU cache holds reaches the renderer's `invalidate_region` ahead of
/// the work submitted after it, again once the cache takes the page anew, and not at all once the
/// cache let it go.
#[test]
fn cpu_writes_to_cached_pages_invalidate_before_later_work() {
    each_mode(|threading| {
        let rig = Rig::new(threading);
        let device = rig.gpu.handles().device.clone();
        let ran = |rig: &Rig| {
            let log = rig.log.clone();
            rig.gpu.run_in_order(move |_| log.lock().push(Event::Ran));
            rig.gpu.synchronize();
        };
        for round in 0..2u32 {
            let cached = device.cache_pages(DEVICE_BASE + 0x8000, 0x1000);
            rig.space.write(CPU_VA + 0x8010, round).unwrap();
            ran(&rig);
            drop(cached);
        }
        rig.space.write(CPU_VA + 0x8010, 2u32).unwrap();
        ran(&rig);
        let events: Vec<Event> = rig
            .events()
            .into_iter()
            .filter(|event| matches!(event, Event::Invalidate(..) | Event::Ran))
            .collect();
        let invalidate = Event::Invalidate(DEVICE_BASE + 0x8000, 0x1000);
        assert_eq!(
            events,
            [
                invalidate.clone(),
                Event::Ran,
                invalidate,
                Event::Ran,
                Event::Ran
            ],
            "{threading:?}"
        );
    });
}
