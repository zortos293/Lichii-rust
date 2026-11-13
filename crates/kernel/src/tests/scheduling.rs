use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lichii_common::ConsoleGeneration;
use lichii_common::settings::{CpuAccuracy, CpuBackend};
use lichii_cpu::CpuConfig;
use parking_lot::Mutex;

use super::{Script, TestKernel, word};
use crate::process::{Process, ProcessParams};
use crate::util::Rng;
use crate::{Kernel, KernelConfig};

pub(super) fn heap_base(test: &TestKernel) -> u64 {
    let params = ProcessParams::from_image(&test.image(), false).unwrap();
    let layout = Process::layout(
        &params,
        crate::memory::Region::new(super::CODE_BASE, super::CODE_SIZE),
    );
    crate::memory::Regions::compute(layout, &mut Rng::new(0))
        .unwrap()
        .heap
        .start
}

/// Main thread script: creates `count` threads running `entry` with arg = index, priority and core
/// from `spec`, starts them and waits for each to exit. The caller appends the rest (and the
/// exit).
pub(super) fn spawn_and_join(
    heap: u64,
    count: u64,
    entry: u64,
    spec: impl Fn(u64) -> (u32, u32) + Send + Sync + 'static,
) -> Script {
    let spec = Arc::new(spec);
    let handles = heap + 0x1000;
    Script::new()
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x08, move |ctx, _| {
            let i = ctx.x[19];
            let (priority, core) = spec(i);
            ctx.x[1] = entry;
            ctx.x[2] = i;
            ctx.x[3] = heap + 0x10_0000 + 0x1_0000 * i;
            ctx.x[4] = priority as u64;
            ctx.x[5] = core as u64;
        })
        .then(move |ctx, env| {
            assert_eq!(ctx.x[0], 0, "CreateThread failed");
            env.memory
                .write(handles + 4 * ctx.x[19], ctx.x[1] as u32)
                .unwrap();
        })
        .svc(0x09, move |ctx, env| {
            ctx.x[0] = env.memory.read::<u32>(handles + 4 * ctx.x[19]).unwrap() as u64;
        })
        .then(|ctx, _| {
            assert_eq!(ctx.x[0], 0, "StartThread failed");
            ctx.x[19] += 1;
        })
        .jump_if(1, move |ctx, _| ctx.x[19] < count)
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x18, move |ctx, _| {
            ctx.x[1] = handles + 4 * ctx.x[19];
            ctx.x[2] = 1;
            ctx.x[3] = u64::MAX;
        })
        .then(|ctx, _| {
            assert_eq!(ctx.x[0], 0, "WaitSynchronization on a thread failed");
            ctx.x[19] += 1;
        })
        .jump_if(7, move |ctx, _| ctx.x[19] < count)
}

#[test]
fn switch2_runs_six_cores_natively() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = seen.clone();
    let worker = test.program(
        Script::new()
            .then(|ctx, _| ctx.x[20] = ctx.x[0])
            .svc(0x10, |_, _| {})
            .then(move |ctx, env| record.lock().push((ctx.x[20], ctx.x[0], env.core as u64)))
            .exit(),
    );
    let main = test.program(spawn_and_join(heap, 6, worker, |i| (40, i as u32)).exit());
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    let mut seen = seen.lock().clone();
    seen.sort();
    assert_eq!(seen, (0..6).map(|i| (i, i, i)).collect::<Vec<_>>());
}

#[test]
fn switch1_rejects_cores_four_and_five() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let result = Arc::new(AtomicU64::new(0));
    let r = result.clone();
    let main = test.program(
        Script::new()
            .svc(0x08, |ctx, _| {
                ctx.x[1] = super::CODE_BASE;
                ctx.x[4] = 44;
                ctx.x[5] = 4;
            })
            .then(move |ctx, _| r.store(ctx.x[0], Ordering::SeqCst))
            .exit(),
    );
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    assert_eq!(
        result.load(Ordering::SeqCst),
        crate::result::INVALID_CORE_ID.0 as u64
    );
}

#[test]
fn higher_priority_thread_preempts_a_spinner() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let flag = heap + 0x800;
    let spinner = test.program(
        Script::new()
            .spin_until(move |_, env| word(env, flag).load(Ordering::SeqCst) == 1)
            .exit(),
    );
    let setter = test.program(
        Script::new()
            .then(move |_, env| word(env, flag).store(1, Ordering::SeqCst))
            .exit(),
    );
    let main = test.program(spawn_and_join_entries(
        heap,
        vec![spinner, setter],
        vec![(50, 1), (30, 1)],
    ));
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
}

/// Like [`spawn_and_join`] with a different entry per thread; waits a little between starts so
/// the first thread is running when the second becomes ready.
pub(super) fn spawn_and_join_entries(
    heap: u64,
    entries: Vec<u64>,
    spec: Vec<(u32, u32)>,
) -> Script {
    let count = entries.len() as u64;
    let handles = heap + 0x1000;
    let entries = Arc::new(entries);
    let spec = Arc::new(spec);
    Script::new()
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x08, move |ctx, _| {
            let i = ctx.x[19] as usize;
            ctx.x[1] = entries[i];
            ctx.x[2] = i as u64;
            ctx.x[3] = heap + 0x10_0000 + 0x1_0000 * i as u64;
            ctx.x[4] = spec[i].0 as u64;
            ctx.x[5] = spec[i].1 as u64;
        })
        .then(move |ctx, env| {
            assert_eq!(ctx.x[0], 0);
            env.memory
                .write(handles + 4 * ctx.x[19], ctx.x[1] as u32)
                .unwrap();
        })
        .svc(0x09, move |ctx, env| {
            ctx.x[0] = env.memory.read::<u32>(handles + 4 * ctx.x[19]).unwrap() as u64;
        })
        .svc(0x0b, |ctx, _| ctx.x[0] = 2_000_000)
        .then(|ctx, _| ctx.x[19] += 1)
        .jump_if(1, move |ctx, _| ctx.x[19] < count)
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x18, move |ctx, _| {
            ctx.x[1] = handles + 4 * ctx.x[19];
            ctx.x[2] = 1;
            ctx.x[3] = u64::MAX;
        })
        .then(|ctx, _| ctx.x[19] += 1)
        .jump_if(8, move |ctx, _| ctx.x[19] < count)
        .exit()
}

#[test]
fn preemption_priority_threads_rotate() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let counters = heap + 0x900;
    let worker = test.program(
        Script::new()
            .then(|ctx, _| ctx.x[20] = ctx.x[0])
            .spin_until(move |ctx, env| {
                let me = ctx.x[20];
                word(env, counters + 4 * me).fetch_add(1, Ordering::SeqCst);
                word(env, counters + 4 * (1 - me)).load(Ordering::SeqCst) > 0
            })
            .exit(),
    );
    let main = test.program(spawn_and_join(heap, 2, worker, |_| (59, 1)).exit());
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
}

#[test]
fn sleep_and_yield_return() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let elapsed = Arc::new(AtomicU64::new(0));
    let e = elapsed.clone();
    let start = Arc::new(Mutex::new(None));
    let s1 = start.clone();
    let main = test.program(
        Script::new()
            .svc(0x0b, |ctx, _| ctx.x[0] = 0)
            .svc(0x0b, |ctx, _| ctx.x[0] = u64::MAX)
            .svc(0x0b, move |ctx, _| {
                *s1.lock() = Some(std::time::Instant::now());
                ctx.x[0] = 5_000_000;
            })
            .then(move |_, _| {
                let t = start.lock().expect("start time");
                e.store(t.elapsed().as_micros() as u64, Ordering::SeqCst);
            })
            .exit(),
    );
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    assert!(elapsed.load(Ordering::SeqCst) >= 5_000);
}

#[test]
fn shutdown_stops_a_spinning_core() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let main = test.program(Script::new().step(|_, _| super::Action::Stay));
    test.start(test.image(), main);
    test.kernel.run().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let (tx, rx) = crossbeam_channel::bounded(1);
    let kernel = test.kernel.clone();
    std::thread::spawn(move || {
        kernel.shutdown();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(5))
        .expect("shutdown hung on a spinning core");
}

#[test]
fn pause_stops_guest_code_until_resume() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let (counter, done) = (heap + 0x800, heap + 0x804);
    let main = test.program(
        Script::new()
            .step(move |_, env| {
                word(env, counter).fetch_add(1, Ordering::SeqCst);
                if word(env, done).load(Ordering::SeqCst) == 1 {
                    super::Action::Next
                } else {
                    super::Action::Stay
                }
            })
            .exit(),
    );
    let process = test.start(test.image(), main);
    let read = |addr| process.memory().read::<u32>(addr).unwrap();
    test.kernel.run().unwrap();
    std::thread::sleep(Duration::from_millis(30));
    assert!(read(counter) > 0, "guest never ran");
    test.kernel.pause();
    assert!(test.kernel.is_paused());
    std::thread::sleep(Duration::from_millis(30));
    let paused_at = read(counter);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(read(counter), paused_at, "guest ran while paused");
    test.kernel.resume();
    std::thread::sleep(Duration::from_millis(30));
    assert!(read(counter) > paused_at, "guest did not resume");
    process.memory().write(done, 1u32).unwrap();
    test.wait_for_exit(Duration::from_secs(10));
}

/// A guest counter read the way translated code reads it: host time plus an offset the clock
/// moves only at pause and resume, which a core still in guest code would not see.
struct InlineCounter {
    epoch: std::time::Instant,
    offset_ns: std::sync::atomic::AtomicI64,
    paused_at: std::sync::atomic::AtomicI64,
}

impl InlineCounter {
    fn read(&self) -> i64 {
        self.offset_ns.load(Ordering::SeqCst) + self.epoch.elapsed().as_nanos() as i64
    }

    fn pause(&self) {
        self.paused_at.store(self.read(), Ordering::SeqCst);
    }

    fn resume(&self) {
        let elapsed = self.epoch.elapsed().as_nanos() as i64;
        self.offset_ns.store(
            self.paused_at.load(Ordering::SeqCst) - elapsed,
            Ordering::SeqCst,
        );
    }
}

#[test]
fn guest_time_never_steps_back_across_pause() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let done = heap + 0x800;
    let counter = Arc::new(InlineCounter {
        epoch: std::time::Instant::now(),
        offset_ns: Default::default(),
        paused_at: Default::default(),
    });
    let last = Arc::new(std::sync::atomic::AtomicI64::new(i64::MIN));
    let stepped_back = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let main = {
        let (counter, last, stepped_back) = (counter.clone(), last.clone(), stepped_back.clone());
        test.program(
            Script::new()
                .step(move |_, env| {
                    let observe = |now: i64| {
                        if now < last.swap(now, Ordering::SeqCst) {
                            stepped_back.store(true, Ordering::SeqCst);
                        }
                    };
                    observe(counter.read());
                    let busy = std::time::Instant::now();
                    while busy.elapsed() < Duration::from_millis(1) {
                        std::hint::spin_loop();
                    }
                    observe(counter.read());
                    if word(env, done).load(Ordering::SeqCst) == 1 {
                        super::Action::Next
                    } else {
                        super::Action::Stay
                    }
                })
                .exit(),
        )
    };
    let process = test.start(test.image(), main);
    test.kernel.run().unwrap();
    std::thread::sleep(Duration::from_millis(10));
    for _ in 0..20 {
        test.kernel.pause();
        counter.pause();
        std::thread::sleep(Duration::from_millis(2));
        counter.resume();
        test.kernel.resume();
        std::thread::sleep(Duration::from_millis(2));
    }
    process.memory().write(done, 1u32).unwrap();
    test.wait_for_exit(Duration::from_secs(10));
    assert!(last.load(Ordering::SeqCst) > i64::MIN, "guest never ran");
    assert!(
        !stepped_back.load(Ordering::SeqCst),
        "a core read the counter after the pause returned"
    );
}

#[test]
fn cores_are_created_with_the_configured_cpu_accuracy() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let recorded = seen.clone();
    let kernel = Kernel::new(KernelConfig {
        console: ConsoleGeneration::Switch1,
        dram_size: 256 << 20,
        pool_layout: None,
        fastmem: false,
        cpu_accuracy: CpuAccuracy::Paranoid,
        aslr: false,
        rng_seed: 1,
        enforce_service_access: true,
        svc_trace: false,
        clock: Arc::new(super::HostClock(std::time::Instant::now())),
        cpu: Arc::new(move |config: CpuConfig| {
            recorded.lock().push(config.accuracy);
            lichii_cpu::new_core(CpuBackend::Interpreter, config)
        }),
        events: None,
    })
    .unwrap();
    let image = crate::testing::program_image(0x0100_0000_0000_1000);
    let process = kernel.create_process(&image).unwrap();
    let _cpu = kernel.0.new_cpu(&process, 0);
    assert_eq!(*seen.lock(), [CpuAccuracy::Paranoid]);
}

#[test]
fn cores_drop_the_cpu_state_of_an_exited_process() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let main = test.program(Script::new().exit());
    let first = test.start(test.image(), main);
    let memory = Arc::downgrade(first.memory());
    drop(first);
    test.kernel.run().unwrap();
    test.exits
        .recv_timeout(Duration::from_secs(10))
        .expect("first process exits");
    test.start(test.image(), main);
    test.exits
        .recv_timeout(Duration::from_secs(10))
        .expect("second process exits");
    let retired = memory.upgrade().is_none();
    test.kernel.shutdown();
    assert!(
        retired,
        "a core still holds the exited process's address space"
    );
}

#[test]
fn exited_processes_give_back_their_fastmem_arenas() {
    let probe = Arc::new(lichii_memory::PhysicalMemory::new(1 << 20).unwrap());
    match lichii_memory::AddressSpace::with_fastmem(probe, 39) {
        Ok(_) => {}
        Err(lichii_memory::FastmemError::NoFaultHandler | lichii_memory::FastmemError::NoViews) => {
            return;
        }
        Err(err) => panic!("{err}"),
    }
    let test = TestKernel::with_fastmem(ConsoleGeneration::Switch1, true);
    let main = test.program(Script::new().exit());
    test.kernel.run().unwrap();
    for round in 0..300 {
        let process = test.start(test.image(), main);
        assert!(
            process.memory().fastmem().is_some(),
            "process {round} got no fastmem arena"
        );
        drop(process);
        test.exits
            .recv_timeout(Duration::from_secs(10))
            .expect("process exits");
    }
    test.kernel.shutdown();
}
