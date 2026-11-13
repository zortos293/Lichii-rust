use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use lichii_common::{ConsoleGeneration, ResultCode};
use lichii_cpu::ThreadContext;
use parking_lot::Mutex;

use super::scheduling::heap_base;
use super::{Script, TestKernel};
use crate::event::Event;
use crate::ipc::{HleRequestContext, SessionHandler, UNKNOWN_COMMAND_ID};
use crate::object::KObject;
use crate::result;

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const SFCO: u32 = u32::from_le_bytes(*b"SFCO");

/// A request as a guest would encode it.
#[derive(Clone, Default)]
struct Msg {
    ty: u16,
    pid: bool,
    copy: Vec<u32>,
    moves: Vec<u32>,
    a: Vec<(u64, u64)>,
    b: Vec<(u64, u64)>,
    domain: Option<(u32, u32)>,
    command: u32,
    data: Vec<u32>,
}

impl Msg {
    fn cmif(command: u32, data: &[u32]) -> Self {
        Self {
            ty: 4,
            command,
            data: data.to_vec(),
            ..Self::default()
        }
    }

    fn tipc(command: u32, data: &[u32]) -> Self {
        Self {
            ty: 16 + command as u16,
            data: data.to_vec(),
            ..Self::default()
        }
    }

    fn encode(&self) -> Vec<u32> {
        let mut w = vec![
            u32::from(self.ty) | (self.a.len() as u32) << 20 | (self.b.len() as u32) << 24,
            0,
        ];
        let has_handles = self.pid || !self.copy.is_empty() || !self.moves.is_empty();
        if has_handles {
            w.push(
                u32::from(self.pid)
                    | (self.copy.len() as u32) << 1
                    | (self.moves.len() as u32) << 5,
            );
            if self.pid {
                w.extend([0, 0]);
            }
            w.extend(&self.copy);
            w.extend(&self.moves);
        }
        for &(addr, size) in self.a.iter().chain(&self.b) {
            w.push(size as u32);
            w.push(addr as u32);
            w.push(
                ((addr >> 32) as u32 & 0xf) << 28
                    | ((addr >> 36) as u32 & 0x7) << 2
                    | ((size >> 32) as u32 & 0xf) << 24,
            );
        }
        let raw_start = w.len();
        if self.ty < 16 {
            while w.len() % 4 != 0 {
                w.push(0);
            }
            if let Some((command, object)) = self.domain {
                w.extend([
                    command | (16 + 4 * self.data.len() as u32) << 16,
                    object,
                    0,
                    0,
                ]);
            }
            w.extend([SFCI, 0, self.command, 0]);
        }
        w.extend(&self.data);
        w[1] = (w.len() - raw_start) as u32 | u32::from(has_handles) << 31;
        w
    }
}

type Replies = Arc<Mutex<Vec<(u64, Vec<u32>)>>>;

/// Appends: write the message to TLS, `SendSyncRequest` on the handle in register `handle`,
/// record `(result, reply words)`, and keep reply word 3 (the first handle) in `stash`.
fn call(
    script: Script,
    replies: &Replies,
    handle: usize,
    msg: Msg,
    stash: Option<usize>,
) -> Script {
    call_with(script, replies, handle, move |_| msg.clone(), stash)
}

/// Like [`call`], with a message built from the registers at run time.
fn call_with(
    script: Script,
    replies: &Replies,
    handle: usize,
    msg: impl Fn(&ThreadContext) -> Msg + Send + Sync + 'static,
    stash: Option<usize>,
) -> Script {
    let replies = replies.clone();
    script
        .then(move |ctx, env| {
            env.memory
                .write_bytes(ctx.tpidrro_el0, bytemuck::cast_slice(&msg(ctx).encode()))
                .unwrap()
        })
        .svc(0x21, move |ctx, _| ctx.x[0] = ctx.x[handle])
        .then(move |ctx, env| {
            let mut words = vec![0u32; 64];
            env.memory
                .read_bytes(ctx.tpidrro_el0, bytemuck::cast_slice_mut(&mut words))
                .unwrap();
            if let Some(register) = stash {
                ctx.x[register] = u64::from(words[3]);
            }
            replies.lock().push((ctx.x[0], words));
        })
}

fn connect_sm(script: Script, name_at: u64) -> Script {
    script
        .then(move |_, env| env.memory.write_bytes(name_at, b"sm:\0").unwrap())
        .svc(0x1f, move |ctx, _| ctx.x[1] = name_at)
        .then(|ctx: &mut ThreadContext, _| {
            assert_eq!(ctx.x[0], 0, "ConnectToNamedPort(sm:) failed");
            ctx.x[19] = ctx.x[1];
        })
}

fn name(name: &str) -> [u32; 2] {
    let mut raw = [0u8; 8];
    raw[..name.len()].copy_from_slice(name.as_bytes());
    let value = u64::from_le_bytes(raw);
    [value as u32, (value >> 32) as u32]
}

#[derive(Default)]
struct Counter(AtomicU32);

impl SessionHandler for Counter {
    fn name(&self) -> &str {
        "ICounter"
    }

    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        match ctx.command() {
            0 => {
                ctx.push(self.0.fetch_add(1, Ordering::Relaxed) + 1);
                ResultCode::SUCCESS
            }
            _ => UNKNOWN_COMMAND_ID,
        }
    }
}

struct TestService {
    event: Arc<Event>,
    deferred_once: AtomicBool,
}

impl SessionHandler for TestService {
    fn name(&self) -> &str {
        "test:u"
    }

    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        match ctx.command() {
            1 => {
                let a: u32 = ctx.pop();
                let b: u64 = ctx.pop();
                ctx.push(u64::from(a) + b);
                ResultCode::SUCCESS
            }
            2 => {
                ctx.push_interface(Arc::new(Counter::default()));
                ResultCode::SUCCESS
            }
            3 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            4 => {
                let mut data = match ctx.read_buffer(0) {
                    Ok(data) => data,
                    Err(code) => return code,
                };
                data.reverse();
                let written = ctx.write_buffer(0, &data).unwrap();
                let pid = ctx.pid().unwrap_or(0);
                ctx.push(written as u32);
                ctx.push(pid);
                ResultCode::SUCCESS
            }
            5 => {
                ctx.push(1u32);
                ResultCode(0x1234)
            }
            6 => {
                if !self.deferred_once.swap(true, Ordering::SeqCst) {
                    let request = ctx.defer();
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(20));
                        request.retry();
                    });
                    return ResultCode::SUCCESS;
                }
                let moved = ctx.take_move_object(0).map(|object| object.type_name());
                ctx.push(u32::from(moved == Some("KEvent")));
                ResultCode::SUCCESS
            }
            _ => UNKNOWN_COMMAND_ID,
        }
    }
}

fn register_test_service(test: &TestKernel, event: Arc<Event>) {
    let thread = test.kernel.create_service_thread("test").unwrap();
    test.kernel
        .service_manager()
        .register(
            "test:u",
            &thread,
            Arc::new(move || {
                Arc::new(TestService {
                    event: event.clone(),
                    deferred_once: AtomicBool::new(false),
                }) as Arc<dyn SessionHandler>
            }),
        )
        .unwrap();
}

#[test]
fn cmif_round_trips_through_sm_sessions_and_domains() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let event = Arc::new(Event::new());
    register_test_service(&test, event);
    let replies = Replies::default();
    let (input, output) = (heap + 0x4000, heap + 0x5000);
    let mut s = connect_sm(Script::new(), heap + 0x3000);
    s = call(
        s,
        &replies,
        19,
        Msg {
            pid: true,
            ..Msg::cmif(0, &[0, 0])
        },
        None,
    );
    s = call(s, &replies, 19, Msg::cmif(1, &name("test:u")), Some(20));
    s = call(s, &replies, 20, Msg::cmif(1, &[5, 0, 7, 1]), None);
    s = call(s, &replies, 20, Msg::cmif(2, &[]), Some(21));
    s = call(s, &replies, 21, Msg::cmif(0, &[]), None);
    s = call(s, &replies, 21, Msg::cmif(0, &[]), None);
    s = call(s, &replies, 20, Msg::cmif(3, &[]), Some(22));
    let checks = Arc::new(Mutex::new(Vec::new()));
    let c = checks.clone();
    let list = heap + 0x6000;
    s = s
        .then(move |ctx, env| env.memory.write(list, ctx.x[22] as u32).unwrap())
        .svc(0x18, move |ctx, _| {
            ctx.x[1] = list;
            ctx.x[2] = 1;
            ctx.x[3] = 0;
        })
        .then(move |ctx, _| c.lock().push(ctx.x[0]));
    s = s.then(move |_, env| env.memory.write_bytes(input, &[1, 2, 3, 4]).unwrap());
    s = call(
        s,
        &replies,
        20,
        Msg {
            pid: true,
            a: vec![(input, 4)],
            b: vec![(output, 8)],
            ..Msg::cmif(4, &[])
        },
        None,
    );
    let c = checks.clone();
    s = s.then(move |_, env| {
        let mut bytes = [0u8; 4];
        env.memory.read_bytes(output, &mut bytes).unwrap();
        c.lock().push(u64::from(u32::from_le_bytes(bytes)));
    });
    s = call(s, &replies, 20, Msg::cmif(5, &[]), None);
    s = call(s, &replies, 21, Msg::cmif(9, &[]), None);
    s = call(
        s,
        &replies,
        20,
        Msg {
            ty: 5,
            ..Msg::cmif(3, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        20,
        Msg {
            ty: 5,
            ..Msg::cmif(0, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        20,
        Msg {
            domain: Some((1, 1)),
            ..Msg::cmif(2, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        20,
        Msg {
            domain: Some((1, 2)),
            ..Msg::cmif(0, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        20,
        Msg {
            domain: Some((2, 2)),
            ..Msg::cmif(0, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        20,
        Msg {
            domain: Some((1, 2)),
            ..Msg::cmif(0, &[])
        },
        None,
    );
    s = call(
        s,
        &replies,
        21,
        Msg {
            ty: 2,
            ..Msg::default()
        },
        None,
    );
    s = call(s, &replies, 21, Msg::cmif(0, &[]), None);
    let main = test.program(s.exit());
    let process = test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    let r = replies.lock().clone();
    let ok = |i: usize| {
        assert_eq!(r[i].0, 0, "svc result of request {i}");
        &r[i].1
    };
    // Initialize, then GetService returns one move handle.
    assert_eq!(ok(0)[..8], [0, 8, 0, 0, SFCO, 0, 0, 0]);
    assert_eq!(
        ok(1)[..8],
        [0, 8 | 1 << 31, 1 << 5, ok(1)[3], SFCO, 0, 0, 0]
    );
    assert_ne!(ok(1)[3], 0);
    // u32 + u64 (naturally aligned), reply data after the 4-word CMIF header.
    assert_eq!(ok(2)[6..10], [0, 0, 12, 1]);
    // A sub-interface is a new session; its counter is per object.
    assert_eq!(ok(3)[2], 1 << 5);
    assert_eq!((ok(4)[8], ok(5)[8]), (1, 2));
    // A copy handle to the (waitable, unsignalled) event; A and B buffers, and the sent pid.
    assert_eq!(ok(6)[2], 1 << 1);
    assert_eq!(
        *checks.lock(),
        [
            u64::from(result::TIMED_OUT.0),
            u64::from(u32::from_le_bytes([4, 3, 2, 1]))
        ]
    );
    assert_eq!(ok(7)[6..11], [0, 0, 4, 0, process.id() as u32]);
    // An error result drops the reply data; an unknown command.
    assert_eq!(ok(8)[..8], [0, 8, 0, 0, SFCO, 0, 0x1234, 0]);
    assert_eq!(ok(9)[6], UNKNOWN_COMMAND_ID.0);
    // QueryPointerBufferSize, then ConvertCurrentObjectToDomain returns object 1.
    assert_eq!(ok(10)[6..9], [0, 0, 0x8000]);
    assert_eq!(ok(11)[6..9], [0, 0, 1]);
    // In the domain the sub-interface becomes object 2 (domain header, then the id after the
    // data), its counter starts again, and closing the object makes it unknown.
    assert_eq!(ok(12)[..13], [0, 13, 0, 0, 1, 0, 0, 0, SFCO, 0, 0, 0, 2]);
    assert_eq!(ok(13)[4..13], [0, 0, 0, 0, SFCO, 0, 0, 0, 1]);
    assert_eq!(ok(14)[10], 0);
    assert_eq!(ok(15)[10], result::NOT_FOUND.0);
    // Close, then the session is gone.
    assert_eq!(r[16].0, 0);
    assert_eq!(r[17].0, u64::from(result::SESSION_CLOSED.0));
}

#[test]
fn tipc_replies_have_a_one_word_result() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    register_test_service(&test, Arc::new(Event::new()));
    let replies = Replies::default();
    let mut s = connect_sm(Script::new(), heap + 0x3000);
    s = call(s, &replies, 19, Msg::tipc(1, &name("test:u")), None);
    s = call(
        s,
        &replies,
        19,
        Msg {
            pid: true,
            ..Msg::tipc(0, &[])
        },
        None,
    );
    s = call(s, &replies, 19, Msg::tipc(1, &name("test:u")), Some(20));
    s = call(s, &replies, 20, Msg::tipc(1, &[5, 0, 7, 0]), None);
    s = call(s, &replies, 20, Msg::tipc(2, &[]), Some(21));
    s = call(s, &replies, 21, Msg::tipc(0, &[]), None);
    s = call(s, &replies, 19, Msg::tipc(1, &[0x41, 0x41]), None);
    s = call(
        s,
        &replies,
        21,
        Msg {
            ty: 15,
            ..Msg::default()
        },
        None,
    );
    s = call(s, &replies, 21, Msg::tipc(0, &[]), None);
    let main = test.program(s.exit());
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    let r = replies.lock().clone();
    assert!(r[..8].iter().all(|(svc, _)| *svc == 0));
    assert_eq!(r[0].1[..3], [17, 1, crate::sm::INVALID_CLIENT.0]);
    assert_eq!(r[1].1[..3], [16, 1, 0]);
    assert_eq!(r[2].1[..5], [17, 1 | 1 << 31, 1 << 5, r[2].1[3], 0]);
    assert_eq!(r[3].1[..5], [17, 3, 0, 12, 0]);
    assert_eq!(r[5].1[..4], [16, 2, 0, 1]);
    assert_eq!(r[6].1[..3], [17, 1, crate::sm::INVALID_SERVICE_NAME.0]);
    assert_eq!(r[8].0, u64::from(result::SESSION_CLOSED.0));
}

#[test]
fn get_service_waits_for_a_late_registration() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let replies = Replies::default();
    let mut s = connect_sm(Script::new(), heap + 0x3000);
    s = call(
        s,
        &replies,
        19,
        Msg {
            pid: true,
            ..Msg::tipc(0, &[])
        },
        None,
    );
    s = call(s, &replies, 19, Msg::tipc(1, &name("test:u")), Some(20));
    s = call(s, &replies, 20, Msg::tipc(1, &[1, 0, 2, 0]), None);
    let main = test.program(s.exit());
    test.start(test.image(), main);
    test.kernel.run().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        replies.lock().len(),
        1,
        "GetService answered before registration"
    );
    assert_eq!(test.kernel.service_manager().waiting_for(), ["test:u"]);
    register_test_service(&test, Arc::new(Event::new()));
    test.wait_for_exit(Duration::from_secs(10));
    let r = replies.lock().clone();
    assert_eq!(r[1].0, 0);
    assert_ne!(r[1].1[3], 0);
    assert_eq!(r[2].1[..4], [17, 3, 0, 3]);
}

#[test]
fn a_client_stalled_on_an_unregistered_service_is_reported() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let replies = Replies::default();
    let mut s = connect_sm(Script::new(), heap + 0x3000);
    s = call(
        s,
        &replies,
        19,
        Msg {
            pid: true,
            ..Msg::tipc(0, &[])
        },
        None,
    );
    s = call(s, &replies, 19, Msg::tipc(1, &name("test:u")), Some(20));
    let main = test.program(s.exit());
    test.start(test.image(), main);
    test.kernel.run().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while test.kernel.service_manager().stalled().is_empty() {
        assert!(std::time::Instant::now() < deadline, "no stall reported");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(test.kernel.service_manager().stalled(), ["test:u"]);
    assert_eq!(test.kernel.service_manager().waiting_for(), ["test:u"]);
    register_test_service(&test, Arc::new(Event::new()));
    test.wait_for_exit(Duration::from_secs(10));
    assert_eq!(replies.lock()[1].0, 0, "answered once registered");
}

#[test]
fn user_buffers_clones_and_deferred_move_handles() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    register_test_service(&test, Arc::new(Event::new()));
    let replies = Replies::default();
    let buffer = heap + 0x8000;
    let mut s = connect_sm(Script::new(), heap + 0x3000);
    s = call(
        s,
        &replies,
        19,
        Msg {
            pid: true,
            ..Msg::cmif(0, &[0, 0])
        },
        None,
    );
    s = call(s, &replies, 19, Msg::cmif(1, &name("test:u")), Some(20));
    let user = replies.clone();
    s = s
        .then(move |_, env| {
            let words = Msg::tipc(1, &[2, 0, 3, 0]).encode();
            env.memory
                .write_bytes(buffer, bytemuck::cast_slice(&words))
                .unwrap()
        })
        .svc(0x22, move |ctx, _| {
            ctx.x[0] = buffer;
            ctx.x[1] = 0x1000;
            ctx.x[2] = ctx.x[20];
        })
        .then(move |ctx, env| {
            let mut words = vec![0u32; 8];
            env.memory
                .read_bytes(buffer, bytemuck::cast_slice_mut(&mut words))
                .unwrap();
            user.lock().push((ctx.x[0], words));
        });
    s = call(
        s,
        &replies,
        20,
        Msg {
            ty: 5,
            ..Msg::cmif(2, &[])
        },
        Some(23),
    );
    s = call(
        s,
        &replies,
        23,
        Msg {
            ty: 2,
            ..Msg::default()
        },
        None,
    );
    s = call(s, &replies, 20, Msg::cmif(1, &[1, 0, 1, 0]), None);
    s = call(s, &replies, 23, Msg::cmif(1, &[1, 0, 1, 0]), None);
    s = s.svc(0x45, |_, _| {}).then(|ctx, _| {
        assert_eq!(ctx.x[0], 0, "CreateEvent failed");
        ctx.x[24] = ctx.x[1];
    });
    s = call_with(
        s,
        &replies,
        20,
        |ctx| Msg {
            moves: vec![ctx.x[24] as u32],
            ..Msg::cmif(6, &[])
        },
        None,
    );
    let main = test.program(s.exit());
    test.start(test.image(), main);
    test.run_until_exit(Duration::from_secs(10));
    let r = replies.lock().clone();
    // SendSyncRequestWithUserBuffer: the reply is in the buffer.
    assert_eq!(r[2].0, 0);
    assert_eq!(r[2].1[..5], [17, 3, 0, 5, 0]);
    // Closing a clone leaves the original open.
    assert_eq!(r[3].0, 0);
    assert_eq!(r[3].1[2], 1 << 5);
    assert_eq!(r[4].0, 0);
    assert_eq!((r[5].0, r[5].1[8]), (0, 2));
    assert_eq!(r[6].0, u64::from(result::SESSION_CLOSED.0));
    // A deferred request keeps the object of its move handle for the retry.
    assert_eq!((r[7].0, r[7].1[6], r[7].1[8]), (0, 0, 1));
}
