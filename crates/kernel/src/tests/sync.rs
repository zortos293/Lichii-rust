use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use lichii_common::ConsoleGeneration;
use lichii_cpu::ThreadContext;
use parking_lot::Mutex;

use super::scheduling::{heap_base, spawn_and_join};
use super::{Script, TestKernel, word};
use crate::arbiter::HANDLE_WAIT_MASK;
use crate::event::Event;
use crate::object::KObject;
use crate::result;

#[test]
fn event_signal_wakes_a_waiting_thread() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let event = Arc::new(Event::new());
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = results.clone();
    let handle = Arc::new(AtomicU64::new(0));
    let h = handle.clone();
    let list = heap + 0x2000;
    let waiter = test.program(
        Script::new()
            .then(move |_, env| {
                env.memory
                    .write(list, h.load(Ordering::SeqCst) as u32)
                    .unwrap()
            })
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = list;
                ctx.x[2] = 1;
                ctx.x[3] = u64::MAX;
            })
            .then(move |ctx, _| r.lock().push((ctx.x[0], ctx.x[1] as u32)))
            .exit(),
    );
    let main = test.program(spawn_and_join(heap, 1, waiter, |_| (40, 1)).exit());
    let process = test.start(test.image(), main);
    handle.store(
        process
            .handles()
            .write()
            .add(KObject::ReadableEvent(event.clone()))
            .unwrap() as u64,
        Ordering::SeqCst,
    );
    let signaller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(20));
        event.signal();
    });
    test.run_until_exit(Duration::from_secs(10));
    signaller.join().unwrap();
    assert_eq!(*results.lock(), vec![(0, 0)]);
}

#[test]
fn wait_synchronization_times_out_and_polls() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let event = Arc::new(Event::new());
    let results = Arc::new(Mutex::new(Vec::new()));
    let (r1, r2) = (results.clone(), results.clone());
    let handle = Arc::new(AtomicU64::new(0));
    let h = handle.clone();
    let list = heap + 0x2000;
    let main = test.program(
        Script::new()
            .then(move |_, env| {
                env.memory
                    .write(list, h.load(Ordering::SeqCst) as u32)
                    .unwrap()
            })
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = list;
                ctx.x[2] = 1;
                ctx.x[3] = 2_000_000;
            })
            .then(move |ctx, _| r1.lock().push(ctx.x[0]))
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = list;
                ctx.x[2] = 1;
                ctx.x[3] = 0;
            })
            .then(move |ctx, _| r2.lock().push(ctx.x[0]))
            .exit(),
    );
    let process = test.start(test.image(), main);
    handle.store(
        process
            .handles()
            .write()
            .add(KObject::ReadableEvent(event))
            .unwrap() as u64,
        Ordering::SeqCst,
    );
    test.run_until_exit(Duration::from_secs(10));
    let timed_out = result::TIMED_OUT.0 as u64;
    assert_eq!(*results.lock(), vec![timed_out, timed_out]);
}

#[test]
fn cancel_synchronization_interrupts_a_wait() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let results = Arc::new(Mutex::new(Vec::new()));
    let r = results.clone();
    let handles = heap + 0x1000;
    let list = heap + 0x2000;
    let event = Arc::new(Event::new());
    let handle = Arc::new(AtomicU64::new(0));
    let h = handle.clone();
    let waiter = test.program(
        Script::new()
            .then(move |_, env| {
                env.memory
                    .write(list, h.load(Ordering::SeqCst) as u32)
                    .unwrap()
            })
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = list;
                ctx.x[2] = 1;
                ctx.x[3] = u64::MAX;
            })
            .then(move |ctx, _| r.lock().push(ctx.x[0]))
            .exit(),
    );
    let main = test.program(
        Script::new()
            .svc(0x08, move |ctx, _| {
                ctx.x[1] = waiter;
                ctx.x[3] = heap + 0x10_0000;
                ctx.x[4] = 40;
                ctx.x[5] = 1;
            })
            .then(move |ctx, env| env.memory.write(handles, ctx.x[1] as u32).unwrap())
            .svc(0x09, move |ctx, env| {
                ctx.x[0] = env.memory.read::<u32>(handles).unwrap() as u64
            })
            .svc(0x0b, |ctx, _| ctx.x[0] = 10_000_000)
            .svc(0x19, move |ctx, env| {
                ctx.x[0] = env.memory.read::<u32>(handles).unwrap() as u64
            })
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = handles;
                ctx.x[2] = 1;
                ctx.x[3] = u64::MAX;
            })
            .exit(),
    );
    let process = test.start(test.image(), main);
    handle.store(
        process
            .handles()
            .write()
            .add(KObject::ReadableEvent(event))
            .unwrap() as u64,
        Ordering::SeqCst,
    );
    test.run_until_exit(Duration::from_secs(10));
    assert_eq!(*results.lock(), vec![result::CANCELLED.0 as u64]);
}

/// Two threads on two cores hand a token back and forth through `WaitForAddress` /
/// `SignalToAddress`, the job-system pattern of NX2_NOTES row 109.
#[test]
fn address_arbiter_ping_pong_across_cores() {
    const ROUNDS: u32 = 5000;
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let turn = heap + 0x3000;
    let done = Arc::new(AtomicU64::new(0));
    let d = done.clone();
    let player = test.program(
        Script::new()
            .then(|ctx, _| {
                ctx.x[20] = ctx.x[0];
                ctx.x[21] = 0;
            })
            .jump_if(4, move |ctx, env| {
                ctx.x[24] = word(env, turn).load(Ordering::SeqCst) as u64;
                ctx.x[24] % 2 == ctx.x[20]
            })
            .svc(0x34, move |ctx, _| {
                ctx.x[0] = turn;
                ctx.x[1] = 2;
                ctx.x[2] = ctx.x[24];
                ctx.x[3] = u64::MAX;
            })
            .jump_if(1, |_, _| true)
            .then(move |_, env| {
                word(env, turn).fetch_add(1, Ordering::SeqCst);
            })
            .svc(0x35, move |ctx, _| {
                ctx.x[0] = turn;
                ctx.x[1] = 0;
                ctx.x[2] = 0;
                ctx.x[3] = u32::MAX as u64;
            })
            .then(|ctx, _| ctx.x[21] += 1)
            .jump_if(1, move |ctx, _| ctx.x[21] < ROUNDS as u64)
            .then(move |_, _| {
                d.fetch_add(1, Ordering::SeqCst);
            })
            .exit(),
    );
    let last = Arc::new(AtomicU64::new(0));
    let l = last.clone();
    let main = test.program(
        spawn_and_join(heap, 2, player, |i| (40, 2 + i as u32))
            .then(move |_, env| {
                l.store(
                    word(env, turn).load(Ordering::SeqCst) as u64,
                    Ordering::SeqCst,
                )
            })
            .exit(),
    );
    test.start(test.image(), main);
    let started = std::time::Instant::now();
    test.run_until_exit(Duration::from_secs(30));
    eprintln!("{} arbiter round trips in {:?}", ROUNDS, started.elapsed());
    assert_eq!(done.load(Ordering::SeqCst), 2);
    assert_eq!(last.load(Ordering::SeqCst), 2 * ROUNDS as u64);
}

/// Userland mutex (the libnx/SDK protocol) over `ArbitrateLock`/`ArbitrateUnlock`, contended by
/// four threads on four cores with a non-atomic increment inside the critical section.
#[test]
fn arbitrated_mutex_serialises_four_cores() {
    const ITERATIONS: u64 = 300;
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let mutex = heap + 0x4000;
    let counter = heap + 0x4004;
    let handles = heap + 0x1000;
    let worker = test.program(
        Script::new()
            .then(|ctx, _| {
                ctx.x[20] = ctx.x[0];
                ctx.x[21] = 0;
            })
            .svc(0x25, |ctx, _| {
                ctx.x[1] = crate::handle_table::CURRENT_THREAD as u64
            })
            .then(move |ctx, env| {
                ctx.x[22] = env.memory.read::<u32>(handles + 4 * ctx.x[20]).unwrap() as u64;
            })
            .jump_if(7, move |ctx, env| {
                let tag = ctx.x[22] as u32;
                let w = word(env, mutex);
                loop {
                    let cur = w.load(Ordering::SeqCst);
                    if cur == 0 {
                        if w.compare_exchange(0, tag, Ordering::SeqCst, Ordering::SeqCst)
                            .is_ok()
                        {
                            return true;
                        }
                        continue;
                    }
                    if cur & HANDLE_WAIT_MASK != 0
                        || w.compare_exchange(
                            cur,
                            cur | HANDLE_WAIT_MASK,
                            Ordering::SeqCst,
                            Ordering::SeqCst,
                        )
                        .is_ok()
                    {
                        ctx.x[23] = (cur | HANDLE_WAIT_MASK) as u64;
                        return false;
                    }
                }
            })
            .svc(0x1a, move |ctx, _| {
                ctx.x[0] = ctx.x[23] & !(HANDLE_WAIT_MASK as u64);
                ctx.x[1] = mutex;
                ctx.x[2] = ctx.x[22];
            })
            .jump_if(3, move |ctx, env| {
                word(env, mutex).load(Ordering::SeqCst) as u64 & !(HANDLE_WAIT_MASK as u64)
                    != ctx.x[22]
            })
            .then(|_, _| {})
            .then(move |_, env| {
                let v = env.memory.read::<u32>(counter).unwrap();
                std::thread::yield_now();
                env.memory.write(counter, v + 1).unwrap();
            })
            .jump_if(11, move |ctx, env| {
                word(env, mutex)
                    .compare_exchange(ctx.x[22] as u32, 0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            })
            .svc(0x1b, move |ctx, _| ctx.x[0] = mutex)
            .then(|_, _| {})
            .then(|ctx, _| ctx.x[21] += 1)
            .jump_if(3, move |ctx, _| ctx.x[21] < ITERATIONS)
            .exit(),
    );
    let last = Arc::new(Mutex::new((0, 0)));
    let l = last.clone();
    let main = test.program(
        spawn_and_join(heap, 4, worker, |i| (40 + i as u32, i as u32))
            .then(move |_, env| {
                *l.lock() = (
                    env.memory.read::<u32>(counter).unwrap(),
                    env.memory.read::<u32>(mutex).unwrap(),
                )
            })
            .exit(),
    );
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(30));
    assert_eq!(*last.lock(), (4 * ITERATIONS as u32, 0));
}

#[test]
fn abandoned_wait_does_not_replay_its_result() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let event = Arc::new(Event::new());
    event.signal();
    let results = Arc::new(Mutex::new(Vec::new()));
    let (r1, r2) = (results.clone(), results.clone());
    let handle = Arc::new(AtomicU64::new(0));
    let h = handle.clone();
    let list = heap + 0x2000;
    let other = test.program(Script::new().exit());
    let main = test.program(
        Script::new()
            .then(move |_, env| {
                env.memory
                    .write(list, h.load(Ordering::SeqCst) as u32)
                    .unwrap()
            })
            .svc(0x08, move |ctx, _| {
                ctx.x[1] = other;
                ctx.x[2] = 0;
                ctx.x[3] = heap + 0x10_0000;
                ctx.x[4] = 44;
                ctx.x[5] = 0;
            })
            .svc(0x09, |ctx, _| ctx.x[0] = ctx.x[1])
            .svc(0x18, move |ctx, _| {
                ctx.x[1] = list;
                ctx.x[2] = 1;
                ctx.x[3] = 1_000_000_000;
            })
            .then(move |ctx, _| {
                r1.lock().push((ctx.x[0], ctx.x[1]));
                ctx.x[1] = 0x1234;
            })
            .svc(0x0b, |ctx, _| ctx.x[0] = 0)
            .then(move |ctx, _| r2.lock().push((ctx.x[0], ctx.x[1])))
            .exit(),
    );
    let process = test.start(test.image(), main);
    handle.store(
        process
            .handles()
            .write()
            .add(KObject::ReadableEvent(event))
            .unwrap() as u64,
        Ordering::SeqCst,
    );
    test.run_until_exit(Duration::from_secs(10));
    assert_eq!(*results.lock(), vec![(0, 0), (0, 0x1234)]);
}

#[test]
fn a_mutex_owner_runs_at_the_priority_of_its_best_waiter() {
    const LOW: u32 = 50;
    const HIGH: u32 = 20;
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let handles = heap + 0x1000;
    let mutex = heap + 0x4000;
    let locked = heap + 0x4004;
    let release = heap + 0x4008;
    let seen = Arc::new(AtomicU32::new(0));
    let after = Arc::new(AtomicU32::new(0));
    let (seen_by_owner, after_by_owner) = (seen.clone(), after.clone());
    let own_handle = move |ctx: &mut ThreadContext, env: &super::Env| {
        let slot = handles + 4 * ctx.x[20];
        ctx.x[22] = word(env, slot).load(Ordering::SeqCst) as u64;
        ctx.x[22] == 0
    };
    let worker = test.program(
        Script::new()
            .then(|ctx, _| ctx.x[20] = ctx.x[0])
            .jump_if(1, own_handle)
            .jump_if(13, |ctx, _| ctx.x[20] == 1)
            .then(move |ctx, env| {
                let tag = ctx.x[22] as u32;
                while word(env, mutex)
                    .compare_exchange(0, tag, Ordering::SeqCst, Ordering::SeqCst)
                    .is_err()
                {}
                word(env, locked).store(1, Ordering::SeqCst);
            })
            .svc(0x0c, |ctx, _| {
                ctx.x[1] = crate::handle_table::CURRENT_THREAD as u64
            })
            .then(move |ctx, _| seen_by_owner.store(ctx.x[1] as u32, Ordering::SeqCst))
            .jump_if(4, move |_, env| {
                word(env, release).load(Ordering::SeqCst) == 0
            })
            .jump_if(9, move |ctx, env| {
                word(env, mutex)
                    .compare_exchange(ctx.x[22] as u32, 0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            })
            .svc(0x1b, move |ctx, _| ctx.x[0] = mutex)
            .svc(0x0c, |ctx, _| {
                ctx.x[1] = crate::handle_table::CURRENT_THREAD as u64
            })
            .then(move |ctx, _| after_by_owner.store(ctx.x[1] as u32, Ordering::SeqCst))
            .exit()
            .then(|_, _| {})
            .jump_if(13, move |_, env| {
                word(env, locked).load(Ordering::SeqCst) == 0
            })
            .then(move |ctx, env| {
                let w = word(env, mutex);
                let current = w.fetch_or(HANDLE_WAIT_MASK, Ordering::SeqCst);
                ctx.x[23] = u64::from(current & !HANDLE_WAIT_MASK);
            })
            .svc(0x1a, move |ctx, _| {
                ctx.x[0] = ctx.x[23];
                ctx.x[1] = mutex;
                ctx.x[2] = ctx.x[22];
            })
            .jump_if(18, move |ctx, env| {
                word(env, mutex)
                    .compare_exchange(ctx.x[22] as u32, 0, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            })
            .svc(0x1b, move |ctx, _| ctx.x[0] = mutex)
            .exit(),
    );
    let main = test.program(
        spawn_and_join(heap, 2, worker, |i| {
            (if i == 0 { LOW } else { HIGH }, i as u32)
        })
        .exit(),
    );
    let process = test.start(test.image(), main);
    test.kernel.run().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while seen.load(Ordering::SeqCst) != HIGH {
        assert!(
            std::time::Instant::now() < deadline,
            "the owner never inherited its waiter's priority (seen {})",
            seen.load(Ordering::SeqCst)
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    process.memory().write(release, 1u32).unwrap();
    test.wait_for_exit(Duration::from_secs(10));
    assert_eq!(
        after.load(Ordering::SeqCst),
        LOW,
        "releasing the mutex gives the priority back"
    );
}
