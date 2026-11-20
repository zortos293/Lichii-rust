use std::sync::mpsc;
use std::time::Duration;

use bytemuck::Zeroable;

use super::*;
use crate::back::Nvn2Work;

const TARGET: u64 = 0x20_0000;
const COLOR: [f32; 4] = [0., 1., 0., 1.];
const PIXEL: u32 = u32::from_le_bytes([0, 255, 0, 255]);
const DONE: u64 = SEMAPHORE + 0x1000;
const FENCE: u32 = 17;
const TIMEOUT: Duration = Duration::from_secs(5);

impl Recorder {
    pub(super) fn try_nvn2(
        &mut self,
        packet: &Nvn2Submission,
    ) -> Result<ReplayAdmission, ReplayError> {
        let Some(&mode) = self.work.nvn2.get() else {
            return Ok(ReplayAdmission::UseStandard);
        };
        assert!(Arc::ptr_eq(&packet.memory, &self.work.memory));
        assert_eq!(packet.mapping_generation, self.work.memory.generation());
        assert_eq!(packet.target, TARGET);
        assert_eq!(packet.color, COLOR);
        assert_eq!(packet.color_mask, 0xf);
        assert_eq!(packet.fence, release());
        self.log.lock().push(Event::Nvn2Attempt);
        if mode != Ok(ReplayAdmission::UseStandard) {
            self.record_nvn2(packet.color, mode.is_ok());
            if mode.is_ok() {
                self.signal_fence(packet.fence);
            }
        }
        mode
    }

    pub(super) fn standard_nvn2(&mut self, ctx: &GraphicsContext<'_>) {
        if self.work.nvn2.get().is_none() {
            return;
        }
        assert!(Arc::ptr_eq(ctx.memory, &self.work.memory));
        assert_eq!(ctx.regs.rt[0].address(), TARGET);
        self.record_nvn2(ctx.regs.clear_color, true);
    }

    fn record_nvn2(&mut self, color: [f32; 4], can_complete: bool) {
        assert_eq!(color, COLOR);
        self.log.lock().push(Event::Nvn2Clear);
        self.nvn2_unflushed = true;
        if can_complete {
            self.work
                .submit(Pending::Fence(FenceAction::WriteSemaphore {
                    address: DEVICE_BASE + TARGET,
                    value: u64::from(PIXEL),
                    size: ReleaseSize::Four,
                }));
        }
    }
}

fn release() -> FenceAction {
    FenceAction::WriteSemaphore {
        address: DEVICE_BASE + DONE,
        value: 1,
        size: ReleaseSize::Four,
    }
}

fn check(threading: Threading, mode: Result<ReplayAdmission, ReplayError>) {
    let mut rig = Rig::build_for(
        Console::Switch2,
        threading,
        crate::gpu::DEFAULT_RING_SLOTS,
        Completion::Deferred,
        Engines::Recording,
    );
    rig.work.nvn2.set(mode).unwrap();
    let channel = rig.channel(HostClass::Ampere);
    let other = rig.channel(HostClass::Ampere);
    rig.submit(
        channel,
        &Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .inc(0, 0x700, &[1])
            .semaphore(SEMAPHORE, 1, 2)
            .inc(0, 0x701, &[2])
            .0,
    );
    let packet = Nvn2Submission {
        memory: rig.memory.clone(),
        target: TARGET,
        mapping_generation: rig.memory.generation(),
        color: COLOR,
        color_mask: 0xf,
        fence: release(),
    };
    let memory = packet.memory.clone();
    let standard_calls = Arc::new(AtomicUsize::new(0));
    let calls = standard_calls.clone();
    let (done, outcome) = mpsc::channel();
    rig.gpu.submit_nvn2(
        channel,
        Nvn2Work {
            packet,
            standard: Box::new(move |r| {
                calls.fetch_add(1, Ordering::Relaxed);
                let mut regs = Box::new(Regs::zeroed());
                regs.rt[0].address_low = TARGET as u32;
                regs.clear_color = COLOR;
                r.clear(
                    &mut GraphicsContext {
                        channel,
                        memory: &memory,
                        regs: &regs,
                        dirty: &mut crate::engines::maxwell_3d::DirtyState::new(),
                        const_buffers: &Default::default(),
                        root_tables: &[[0; 64]; 8],
                    },
                    1,
                );
            }),
            done,
        },
    );
    let fence = rig
        .gpu
        .submit(
            channel,
            Submission {
                entries: &[],
                wait: None,
                increment: Some(SyncpointIncrement::increment_value(FENCE, 1)),
            },
        )
        .unwrap();
    let (published, publication) = mpsc::channel();
    let backing = rig.memory.clone();
    let waiter = rig
        .syncpoints
        .guest()
        .register_action(fence.id, fence.value, move || {
            let bytes = [
                backing.read::<u32>(TARGET).unwrap(),
                backing.read(DONE).unwrap(),
            ];
            published.send(bytes).unwrap();
        })
        .unwrap()
        .unwrap();
    rig.submit(
        other,
        &Pushbuffer::default()
            .bind(0, AMPERE_3D)
            .inc(0, 0x702, &[3])
            .0,
    );

    assert_eq!(
        rig.methods(),
        [
            (channel.0, EngineKind::Maxwell3d, 0x700, 1),
            (other.0, EngineKind::Maxwell3d, 0x702, 3),
        ]
    );
    assert_eq!(outcome.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(publication.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(rig.syncpoints.host().value(fence.id), Ok(0));
    assert_eq!(rig.syncpoints.guest().value(fence.id), Ok(0));
    assert_eq!([rig.read(TARGET), rig.read(DONE)], [0, 0]);
    let before_resume = {
        let log = rig.log.lock();
        assert!(!log.contains(&Event::Nvn2Attempt));
        log.len()
    };

    rig.memory.write(SEMAPHORE, 1_u32).unwrap();
    rig.gpu.wake();
    assert_eq!(outcome.recv_timeout(TIMEOUT).unwrap(), mode);
    let mut expected = vec![
        Event::Method {
            channel: channel.0,
            engine: EngineKind::Maxwell3d,
            method: 0x701,
            argument: 2,
        },
        Event::Flush(EngineKind::Maxwell3d),
        Event::Nvn2Attempt,
        Event::Nvn2Clear,
    ];
    if mode.is_ok() {
        expected.extend([
            Event::Fence(release()),
            Event::Nvn2Submit,
            Event::Barrier,
            Event::Fence(FenceAction::IncrementSyncpoint(fence.id)),
        ]);
    }
    rig.gpu.synchronize();
    if mode.is_ok() {
        assert_eq!(rig.syncpoints.host().value(fence.id), Ok(fence.value));
    } else {
        assert_eq!(rig.syncpoints.host().value(fence.id), Ok(0));
        rig.gpu.shutdown();
        assert_eq!(
            Arc::strong_count(&rig.work),
            1,
            "the back dropped the renderer"
        );
        assert_eq!(rig.syncpoints.host().value(fence.id), Ok(0));
        assert!(rig.work.pending.lock().pending.is_empty());
    }
    let log = rig.log.lock();
    assert_eq!(log[before_resume..], expected);
    drop(log);
    assert_eq!(
        standard_calls.load(Ordering::Relaxed),
        usize::from(mode == Ok(ReplayAdmission::UseStandard))
    );
    assert_eq!(rig.syncpoints.guest().value(fence.id), Ok(0));
    assert_eq!(publication.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!([rig.read(TARGET), rig.read(DONE)], [0, 0]);
    if mode.is_ok() {
        rig.complete();
        assert_eq!(publication.recv_timeout(TIMEOUT).unwrap(), [PIXEL, 1]);
        assert_eq!([rig.read(TARGET), rig.read(DONE)], [PIXEL, 1]);
        assert_eq!(rig.syncpoints.guest().value(fence.id), Ok(fence.value));
    } else {
        assert!(rig.syncpoints.guest().cancel(waiter));
        assert_eq!(
            publication.recv_timeout(TIMEOUT),
            Err(mpsc::RecvTimeoutError::Disconnected)
        );
    }
}

#[test]
fn nvn2_packets_wait_on_their_channel_and_fall_back_in_place() {
    each_mode(|threading| {
        check(threading, Ok(ReplayAdmission::Recorded));
        check(threading, Ok(ReplayAdmission::UseStandard));
    });
}

#[test]
fn nvn2_terminal_failure_stops_before_the_dependent_fence() {
    each_mode(|threading| check(threading, Err(ReplayError)));
}
