//! bsd:s, bsd:u and bsd:a with no network. Sockets cannot be created, so every socket call fails
//! with EBADF. Eventfds work: a blocking read or write, or a poll or select without a timeout,
//! waits until another call changes a counter or closes a descriptor.

use std::sync::{Arc, Mutex};

use lichii_common::ResultCode;
use lichii_kernel::Event;
use lichii_kernel::ipc::{DeferredRequest, HleRequestContext};
use lichii_kernel::object::KObject;
use lichii_kernel::thread::ThreadId;

use crate::Service;

pub(super) const SUCCESS: u32 = 0;
const BADF: u32 = 9;
const AGAIN: u32 = 11;
const INVAL: u32 = 22;
const MFILE: u32 = 24;
const AFNOSUPPORT: u32 = 97;
/// What a socket gets with no network, as the C++ answers in airplane mode.
const NOTCONN: u32 = 107;

const MAX_FD: usize = 128;
const STUB_ZEROED_WORDS: usize = 32;
const DOMAIN_UNSPECIFIED: u32 = 0;
const DOMAIN_INET: u32 = 2;
const TYPE_RAW: u32 = 3;
const TYPE_SEQPACKET: u32 = 5;
const PROTOCOL_ICMP: u32 = 1;
const EVENTFD_SEMAPHORE: u32 = 1;
const EVENTFD_NONBLOCK: u32 = 4;
const O_NONBLOCK: i32 = 0x800;
const MAX_COUNTER: u64 = u64::MAX - 1;
const FCNTL_GETFL: i32 = 3;
const FCNTL_SETFL: i32 = 4;
const POLL_IN: u16 = 1;
const POLL_OUT: u16 = 4;
const POLL_NVAL: u16 = 0x20;
const POLL_RDNORM: u16 = 0x40;

struct EventFd {
    counter: u64,
    semaphore: bool,
    status_flags: i32,
}

impl EventFd {
    fn readable(&self) -> bool {
        self.counter != 0
    }

    fn writable(&self) -> bool {
        self.counter < MAX_COUNTER
    }

    fn blocking(&self) -> bool {
        self.status_flags & O_NONBLOCK == 0
    }
}

/// One service's descriptor table; duplicates share an eventfd.
pub(super) struct Descriptors {
    slots: Vec<Option<Arc<Mutex<EventFd>>>>,
    /// Reads and writes that blocked and have not finished, whether waiting or queued to retry.
    /// A guest thread has one request in flight, so its thread names the request. Closing the
    /// fd marks its entries closed: their retry fails with EBADF, and the slot is not reused
    /// until it has, so a retry never lands on a newer eventfd with that number.
    blocked: Vec<Blocked>,
    /// Requests waiting for an eventfd to change; each runs again on every change.
    waiting: Vec<DeferredRequest>,
}

struct Blocked {
    fd: usize,
    thread: ThreadId,
    closed: bool,
}

impl Descriptors {
    pub(super) fn new() -> Self {
        Self {
            slots: vec![None; MAX_FD],
            blocked: Vec::new(),
            waiting: Vec::new(),
        }
    }

    fn get(&self, fd: i32) -> Option<&Arc<Mutex<EventFd>>> {
        let fd = usize::try_from(fd).ok()?;
        self.slots.get(fd)?.as_ref()
    }

    /// The eventfd a read or write on `fd` from `thread` uses. The retry of a request that
    /// blocked on an fd closed since finds none.
    fn take_for_io(&mut self, fd: i32, thread: ThreadId) -> Option<Arc<Mutex<EventFd>>> {
        let index = usize::try_from(fd).ok().filter(|&i| i < MAX_FD)?;
        if let Some(position) = self
            .blocked
            .iter()
            .position(|blocked| blocked.fd == index && blocked.thread == thread)
            && self.blocked.swap_remove(position).closed
        {
            return None;
        }
        self.slots[index].clone()
    }

    fn block(&mut self, fd: usize, thread: ThreadId, request: DeferredRequest) {
        self.blocked.push(Blocked {
            fd,
            thread,
            closed: false,
        });
        self.waiting.push(request);
    }

    fn insert(&mut self, event_fd: Arc<Mutex<EventFd>>) -> Option<i32> {
        let fd = (0..MAX_FD).find(|&fd| {
            self.slots[fd].is_none() && !self.blocked.iter().any(|blocked| blocked.fd == fd)
        })?;
        self.slots[fd] = Some(event_fd);
        Some(fd as i32)
    }

    fn changed(&mut self) {
        for request in self.waiting.drain(..) {
            request.retry();
        }
    }
}

pub(super) struct Bsd {
    pub(super) name: &'static str,
    /// bsd:u and bsd:a, which refuse raw and seqpacket sockets and duplicates.
    pub(super) user: bool,
    pub(super) descriptors: Arc<Mutex<Descriptors>>,
    pub(super) stub_event: Arc<Event>,
}

fn reply(ctx: &mut HleRequestContext, ret: i32, errno: u32) -> ResultCode {
    ctx.push(ret);
    ctx.push(errno);
    ResultCode::SUCCESS
}

fn errno_reply(ctx: &mut HleRequestContext, errno: u32) -> ResultCode {
    reply(ctx, if errno == SUCCESS { 0 } else { -1 }, errno)
}

/// A socket call that takes an fd and fills output buffer `index`, which reports its length.
fn bad_socket_with_length(ctx: &mut HleRequestContext, index: usize) -> ResultCode {
    let len = ctx.writable_buffer(index).map_or(0, |b| b.size as u32);
    reply(ctx, -1, BADF);
    ctx.push(len);
    ResultCode::SUCCESS
}

fn fd_is_set(set: &[u8], fd: usize) -> bool {
    set.get(fd / 8)
        .is_some_and(|byte| byte & (1 << (fd % 8)) != 0)
}

impl Service for Bsd {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 | 36 => {
                ctx.push(0i32);
                ResultCode::SUCCESS
            }
            1 => ResultCode::SUCCESS,
            2 | 3 => {
                let (domain, ty, protocol) = (ctx.pop(), ctx.pop(), ctx.pop());
                let errno = self.socket_errno(domain, ty, protocol);
                reply(ctx, -1, errno)
            }
            4 | 7 | 19 | 23 | 28 | 32..=35 | 37..=40 | 42..=44 | 200 | 201 => {
                ctx.push([0u32; STUB_ZEROED_WORDS]);
                ResultCode::SUCCESS
            }
            41 => {
                ctx.push([0u32; STUB_ZEROED_WORDS]);
                ctx.push_copy_object(KObject::ReadableEvent(self.stub_event.clone()));
                ResultCode::SUCCESS
            }
            5 => self.select(ctx),
            6 => self.poll(ctx),
            8 | 10 | 11 | 29 | 30 => reply(ctx, -1, BADF),
            9 => bad_socket_with_length(ctx, 1),
            12 | 15..=17 => bad_socket_with_length(ctx, 0),
            13 | 14 | 18 | 21 | 22 => errno_reply(ctx, BADF),
            20 => self.fcntl(ctx),
            24 => self.write(ctx),
            25 => self.read(ctx),
            26 => self.close(ctx),
            27 => self.duplicate(ctx),
            31 => self.event_fd(ctx),
            _ => return None,
        })
    }
}

impl Bsd {
    fn socket_errno(&self, domain: u32, ty: u32, protocol: u32) -> u32 {
        if domain != DOMAIN_UNSPECIFIED && domain != DOMAIN_INET {
            return AFNOSUPPORT;
        }
        let icmp = ty == TYPE_RAW && domain == DOMAIN_INET && protocol == PROTOCOL_ICMP;
        if self.user && (ty == TYPE_SEQPACKET || ty == TYPE_RAW) && !icmp {
            return INVAL;
        }
        NOTCONN
    }

    fn event_fd(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let flags: u32 = ctx.pop();
        let initval: u64 = ctx.pop();
        if initval > MAX_COUNTER {
            return reply(ctx, -1, INVAL);
        }
        let status_flags = if flags & EVENTFD_NONBLOCK != 0 {
            O_NONBLOCK
        } else {
            0
        };
        let event_fd = EventFd {
            counter: initval,
            semaphore: flags & EVENTFD_SEMAPHORE != 0,
            status_flags,
        };
        let fd = self
            .descriptors
            .lock()
            .unwrap()
            .insert(Arc::new(Mutex::new(event_fd)));
        match fd {
            Some(fd) => reply(ctx, fd, SUCCESS),
            None => reply(ctx, -1, MFILE),
        }
    }

    fn read(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let mut descriptors = self.descriptors.lock().unwrap();
        let Some(event_fd) = descriptors.take_for_io(fd, ctx.thread().id()) else {
            return reply(ctx, -1, BADF);
        };
        if ctx.writable_buffer(0).map_or(0, |b| b.size) < 8 {
            return reply(ctx, -1, INVAL);
        }
        let mut state = event_fd.lock().unwrap();
        if state.counter == 0 {
            if !state.blocking() {
                return reply(ctx, -1, AGAIN);
            }
            drop(state);
            let thread = ctx.thread().id();
            descriptors.block(fd as usize, thread, ctx.defer());
            return ResultCode::SUCCESS;
        }
        let value = if state.semaphore { 1 } else { state.counter };
        if let Err(code) = ctx.write_buffer(0, &value.to_le_bytes()) {
            return code;
        }
        state.counter -= value;
        drop(state);
        descriptors.changed();
        reply(ctx, 8, SUCCESS)
    }

    fn write(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let mut descriptors = self.descriptors.lock().unwrap();
        let Some(event_fd) = descriptors.take_for_io(fd, ctx.thread().id()) else {
            return reply(ctx, -1, BADF);
        };
        let data = ctx.read_buffer(0).unwrap_or_default();
        let Some(&value) = data.as_chunks::<8>().0.first() else {
            return reply(ctx, -1, INVAL);
        };
        let value = u64::from_le_bytes(value);
        if value == u64::MAX {
            return reply(ctx, -1, INVAL);
        }
        let mut state = event_fd.lock().unwrap();
        if value > MAX_COUNTER - state.counter {
            if !state.blocking() {
                return reply(ctx, -1, AGAIN);
            }
            drop(state);
            let thread = ctx.thread().id();
            descriptors.block(fd as usize, thread, ctx.defer());
            return ResultCode::SUCCESS;
        }
        state.counter += value;
        drop(state);
        descriptors.changed();
        reply(ctx, 8, SUCCESS)
    }

    fn close(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let mut descriptors = self.descriptors.lock().unwrap();
        let slot = usize::try_from(fd)
            .ok()
            .and_then(|fd| descriptors.slots.get_mut(fd));
        if slot.and_then(Option::take).is_none() {
            return errno_reply(ctx, BADF);
        }
        for blocked in &mut descriptors.blocked {
            if blocked.fd == fd as usize {
                blocked.closed = true;
            }
        }
        descriptors.changed();
        errno_reply(ctx, SUCCESS)
    }

    fn duplicate(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        if self.user {
            return reply(ctx, 0, INVAL);
        }
        let mut descriptors = self.descriptors.lock().unwrap();
        let Some(event_fd) = descriptors.get(fd).cloned() else {
            return reply(ctx, 0, BADF);
        };
        match descriptors.insert(event_fd) {
            Some(new_fd) => reply(ctx, new_fd, SUCCESS),
            None => reply(ctx, 0, MFILE),
        }
    }

    fn fcntl(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let fd: i32 = ctx.pop();
        let cmd: i32 = ctx.pop();
        let arg: i32 = ctx.pop();
        let descriptors = self.descriptors.lock().unwrap();
        let Some(event_fd) = descriptors.get(fd) else {
            return reply(ctx, -1, BADF);
        };
        let mut state = event_fd.lock().unwrap();
        match cmd {
            FCNTL_GETFL => reply(ctx, state.status_flags, SUCCESS),
            FCNTL_SETFL => {
                state.status_flags = arg;
                reply(ctx, 0, SUCCESS)
            }
            _ => reply(ctx, -1, SUCCESS),
        }
    }

    fn poll(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let nfds: i32 = ctx.pop();
        let timeout: i32 = ctx.pop();
        if nfds <= 0 {
            return reply(ctx, -1, SUCCESS);
        }
        let size = nfds as usize * 8;
        let mut fds = ctx.read_buffer(0).unwrap_or_default();
        let out_len = ctx.writable_buffer(0).map_or(0, |b| b.size as usize);
        if fds.len() < size || out_len < size || timeout < -1 {
            return reply(ctx, -1, INVAL);
        }
        fds.truncate(size);
        let mut descriptors = self.descriptors.lock().unwrap();
        let mut count = 0;
        for pollfd in fds.as_chunks_mut::<8>().0 {
            let fd = i32::from_le_bytes([pollfd[0], pollfd[1], pollfd[2], pollfd[3]]);
            let events = u16::from_le_bytes([pollfd[4], pollfd[5]]);
            let revents = if fd < 0 {
                0
            } else if let Some(event_fd) = descriptors.get(fd) {
                let state = event_fd.lock().unwrap();
                let mut revents = 0;
                if state.readable() {
                    revents |= events & (POLL_IN | POLL_RDNORM);
                }
                if state.writable() {
                    revents |= events & POLL_OUT;
                }
                revents
            } else {
                POLL_NVAL
            };
            pollfd[6..].copy_from_slice(&revents.to_le_bytes());
            count += i32::from(revents != 0);
        }
        if count == 0 && timeout == -1 {
            descriptors.waiting.push(ctx.defer());
            return ResultCode::SUCCESS;
        }
        drop(descriptors);
        if let Err(code) = ctx.write_buffer(0, &fds) {
            return code;
        }
        reply(ctx, count, SUCCESS)
    }

    fn select(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let nfds: i32 = ctx.pop();
        let _reserved: u32 = ctx.pop();
        let seconds: i64 = ctx.pop();
        let microseconds: i64 = ctx.pop();
        let timeout_is_null = ctx.pop::<u8>() != 0;
        if nfds < 0 {
            return reply(ctx, -1, INVAL);
        }
        let nfds = nfds as usize;
        let examined = nfds.min(MAX_FD);
        let set_size = examined.div_ceil(64) * 8;
        let mut sets: [Option<Vec<u8>>; 3] = Default::default();
        for (index, set) in sets.iter_mut().enumerate() {
            if ctx.readable_buffer(index).is_none_or(|b| b.size == 0) {
                continue;
            }
            let mut buffer = ctx.read_buffer(index).unwrap_or_default();
            let end = nfds.min(buffer.len() * 8);
            if (MAX_FD..end).any(|fd| fd_is_set(&buffer, fd)) {
                return reply(ctx, -1, BADF);
            }
            buffer.truncate(set_size);
            *set = Some(buffer);
        }
        if !timeout_is_null && (seconds < 0 || !(0..1_000_000).contains(&microseconds)) {
            return reply(ctx, -1, INVAL);
        }
        let forever = timeout_is_null || seconds > i64::from(i32::MAX);

        let mut descriptors = self.descriptors.lock().unwrap();
        let mut ready = 0;
        for (index, set) in sets.iter_mut().enumerate() {
            let Some(set) = set else { continue };
            for fd in 0..examined.min(set.len() * 8) {
                if !fd_is_set(set, fd) {
                    continue;
                }
                let Some(event_fd) = descriptors.get(fd as i32) else {
                    return reply(ctx, -1, BADF);
                };
                let state = event_fd.lock().unwrap();
                let is_ready = match index {
                    0 => state.readable(),
                    1 => state.writable(),
                    _ => false,
                };
                if is_ready {
                    ready += 1;
                } else {
                    set[fd / 8] &= !(1 << (fd % 8));
                }
            }
        }
        if ready == 0 && forever {
            descriptors.waiting.push(ctx.defer());
            return ResultCode::SUCCESS;
        }
        drop(descriptors);
        for (index, set) in sets.iter().enumerate() {
            if let Some(set) = set
                && ctx.writable_buffer(index).is_some()
                && let Err(code) = ctx.write_buffer(index, set)
            {
                return code;
            }
        }
        reply(ctx, ready, SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, Reply, ReplyExt, Request};

    fn guest() -> Guest {
        Guest::new(testing::image(), Settings::default(), |context| {
            super::super::register(context).unwrap()
        })
    }

    fn result(reply: &Reply) -> (i32, u32) {
        (reply.at::<i32>(0), reply.at::<u32>(4))
    }

    fn call(guest: &Guest, handle: u32, request: Request) -> (i32, u32) {
        result(&guest.call(handle, request).ok())
    }

    fn event_fd(guest: &Guest, bsd: u32, flags: u32, initval: u64) -> i32 {
        let (fd, errno) = call(guest, bsd, Request::new(31).arg(flags).arg(initval));
        assert_eq!(errno, SUCCESS);
        fd
    }

    #[test]
    fn sockets_cannot_be_created() {
        let guest = guest();
        let user = guest.service("bsd:u");
        let system = guest.service("bsd:s");
        guest.service("bsd:a");
        assert_eq!(guest.call(user, Request::new(0)).ok().at::<i32>(0), 0);
        assert_eq!(guest.call(user, Request::new(36)).ok().at::<i32>(0), 0);
        guest.call(user, Request::new(1)).ok();
        let socket = |bsd, command, domain: u32, ty: u32, protocol: u32| {
            let request = Request::new(command).arg(domain).arg(ty).arg(protocol);
            call(&guest, bsd, request)
        };
        for command in [2, 3] {
            assert_eq!(socket(user, command, 2, 1, 6), (-1, NOTCONN));
            assert_eq!(socket(user, command, 2, 2, 17), (-1, NOTCONN));
            assert_eq!(socket(user, command, 28, 1, 0), (-1, AFNOSUPPORT));
            assert_eq!(socket(user, command, 2, TYPE_RAW, 0), (-1, INVAL));
            assert_eq!(socket(user, command, 2, TYPE_SEQPACKET, 0), (-1, INVAL));
            assert_eq!(
                socket(user, command, 2, TYPE_RAW, PROTOCOL_ICMP),
                (-1, NOTCONN)
            );
            assert_eq!(socket(system, command, 2, TYPE_SEQPACKET, 0), (-1, NOTCONN));
        }

        let fd = event_fd(&guest, user, 0, 0);
        let buffer = guest.heap;
        for command in [13, 14, 18, 21, 22] {
            let request = Request::new(command)
                .arg(fd)
                .arg(0u32)
                .buffer_a(buffer, 0x10);
            assert_eq!(call(&guest, user, request), (-1, BADF), "command {command}");
        }
        for command in [8, 10, 11, 29, 30] {
            let request = Request::new(command).arg(fd).buffer_a(buffer, 0x10);
            assert_eq!(call(&guest, user, request), (-1, BADF), "command {command}");
        }
        for command in [12, 15, 16, 17] {
            let reply = guest
                .call(user, Request::new(command).arg(fd).buffer_b(buffer, 0x10))
                .ok();
            assert_eq!(result(&reply), (-1, BADF));
            assert_eq!(reply.at::<u32>(8), 0x10);
        }
        let recv_from = Request::new(9)
            .arg(fd)
            .buffer_b(buffer, 0x40)
            .buffer_b(buffer + 0x40, 0x10);
        let reply = guest.call(user, recv_from).ok();
        assert_eq!((result(&reply), reply.at::<u32>(8)), ((-1, BADF), 0x10));
    }

    #[test]
    fn stubs_answer_zeroed() {
        let guest = guest();
        let bsd = guest.service("bsd:s");
        for command in [4, 7, 19, 23, 28, 32, 35, 37, 44, 200, 201] {
            let reply = guest.call(bsd, Request::new(command)).ok();
            assert_eq!(reply.at::<[u32; 32]>(0), [0; 32], "command {command}");
        }
        let reply = guest.call(bsd, Request::new(41)).ok();
        assert_eq!(reply.copy_handles.len(), 1);
    }

    #[test]
    fn eventfds_count() {
        let guest = guest();
        let bsd = guest.service("bsd:s");
        let out = guest.heap;
        let input = guest.heap + 0x100;
        let read = |fd: i32, len| {
            let reply = guest
                .call(bsd, Request::new(25).arg(fd).buffer_b(out, len))
                .ok();
            (
                result(&reply),
                u64::from_le_bytes(guest.read(out, 8).try_into().unwrap()),
            )
        };
        let write = |fd: i32, value: u64| {
            guest.write(input, &value.to_le_bytes());
            call(&guest, bsd, Request::new(24).arg(fd).buffer_a(input, 8))
        };

        let counter = event_fd(&guest, bsd, EVENTFD_NONBLOCK, 2);
        assert_eq!(counter, 0);
        assert_eq!(read(counter, 8), ((8, SUCCESS), 2));
        assert_eq!(read(counter, 8).0, (-1, AGAIN));
        assert_eq!(read(counter, 4).0, (-1, INVAL));
        assert_eq!(write(counter, 3), (8, SUCCESS));
        assert_eq!(write(counter, u64::MAX), (-1, INVAL));
        assert_eq!(write(counter, MAX_COUNTER), (-1, AGAIN));
        assert_eq!(read(counter, 8), ((8, SUCCESS), 3));

        let semaphore = event_fd(&guest, bsd, EVENTFD_SEMAPHORE | EVENTFD_NONBLOCK, 2);
        assert_eq!(semaphore, 1);
        assert_eq!(read(semaphore, 8), ((8, SUCCESS), 1));
        assert_eq!(read(semaphore, 8), ((8, SUCCESS), 1));
        assert_eq!(read(semaphore, 8).0, (-1, AGAIN));
        let too_big = Request::new(31).arg(0u32).arg(u64::MAX);
        assert_eq!(call(&guest, bsd, too_big), (-1, INVAL));

        let fcntl = |fd: i32, cmd: i32, arg: i32| {
            call(&guest, bsd, Request::new(20).arg(fd).arg(cmd).arg(arg))
        };
        assert_eq!(fcntl(semaphore, FCNTL_GETFL, 0), (O_NONBLOCK, SUCCESS));
        assert_eq!(fcntl(semaphore, FCNTL_SETFL, 0), (0, SUCCESS));
        assert_eq!(fcntl(semaphore, FCNTL_GETFL, 0), (0, SUCCESS));
        assert_eq!(fcntl(semaphore, 7, 0), (-1, SUCCESS));
        assert_eq!(fcntl(9, FCNTL_GETFL, 0), (-1, BADF));

        let close = |fd: i32| call(&guest, bsd, Request::new(26).arg(fd));
        assert_eq!(close(semaphore), (0, SUCCESS));
        assert_eq!(close(semaphore), (-1, BADF));
        assert_eq!(read(semaphore, 8).0, (-1, BADF));
        assert_eq!(write(-1, 1), (-1, BADF));

        let duplicate = |bsd, fd: i32| call(&guest, bsd, Request::new(27).arg(fd).arg(0u64));
        let copy = duplicate(bsd, counter);
        assert_eq!(copy, (1, SUCCESS));
        assert_eq!(write(counter, 5), (8, SUCCESS));
        assert_eq!(close(counter), (0, SUCCESS));
        assert_eq!(read(copy.0, 8), ((8, SUCCESS), 5));
        assert_eq!(duplicate(bsd, counter), (0, BADF));
        let user = guest.service("bsd:u");
        assert_eq!(duplicate(user, 0), (0, INVAL));
        assert_eq!(event_fd(&guest, user, 0, 0), 0);

        for fd in std::iter::once(0).chain(2..MAX_FD as i32) {
            assert_eq!(event_fd(&guest, bsd, 0, 0), fd);
        }
        assert_eq!(
            call(&guest, bsd, Request::new(31).arg(0u32).arg(0u64)),
            (-1, MFILE)
        );
        assert_eq!(duplicate(bsd, 1), (0, MFILE));
    }

    #[test]
    fn poll_and_select_report_eventfd_readiness() {
        let guest = guest();
        let bsd = guest.service("bsd:s");
        let fd = event_fd(&guest, bsd, EVENTFD_NONBLOCK, 0);
        let fds = guest.heap;
        let out = guest.heap + 0x100;
        let mut pollfds = Vec::new();
        for (fd, events) in [(fd, POLL_IN | POLL_OUT), (-1, POLL_IN), (50, POLL_IN)] {
            pollfds.extend_from_slice(&fd.to_le_bytes());
            pollfds.extend_from_slice(&events.to_le_bytes());
            pollfds.extend_from_slice(&0xffffu16.to_le_bytes());
        }
        guest.write(fds, &pollfds);
        let poll = |nfds: i32, timeout: i32| {
            let request = Request::new(6)
                .arg(nfds)
                .arg(timeout)
                .buffer_a(fds, 24)
                .buffer_b(out, 24);
            call(&guest, bsd, request)
        };
        let revents = |index: usize| {
            let bytes = guest.read(out + index as u64 * 8 + 6, 2);
            u16::from_le_bytes([bytes[0], bytes[1]])
        };
        assert_eq!(poll(3, 0), (2, SUCCESS));
        assert_eq!(
            [revents(0), revents(1), revents(2)],
            [POLL_OUT, 0, POLL_NVAL]
        );
        guest.write(guest.heap + 0x200, &1u64.to_le_bytes());
        let write = Request::new(24).arg(fd).buffer_a(guest.heap + 0x200, 8);
        assert_eq!(call(&guest, bsd, write), (8, SUCCESS));
        assert_eq!(poll(1, 10), (1, SUCCESS));
        assert_eq!(revents(0), POLL_IN | POLL_OUT);
        assert_eq!(poll(0, 0), (-1, SUCCESS));
        assert_eq!(poll(1, -2), (-1, INVAL));
        assert_eq!(poll(4, 0), (-1, INVAL));

        let sets = guest.heap + 0x400;
        let outs = guest.heap + 0x500;
        let select = |nfds: i32, bits: [u64; 3], timeout: (i64, i64, u8)| {
            for (index, bits) in bits.iter().enumerate() {
                guest.write(sets + index as u64 * 0x20, &bits.to_le_bytes());
                guest.write(outs + index as u64 * 0x20, &[0xaa; 8]);
            }
            let mut request = Request::new(5)
                .arg(nfds)
                .arg(0u32)
                .arg(timeout.0)
                .arg(timeout.1)
                .arg(timeout.2);
            for index in 0..3 {
                request = request.buffer_a(sets + index * 0x20, 8);
            }
            for index in 0..3 {
                request = request.buffer_b(outs + index * 0x20, 8);
            }
            call(&guest, bsd, request)
        };
        let out_set =
            |index: u64| u64::from_le_bytes(guest.read(outs + index * 0x20, 8).try_into().unwrap());
        assert_eq!(select(64, [1, 1, 1], (0, 0, 0)), (2, SUCCESS));
        assert_eq!([out_set(0), out_set(1), out_set(2)], [1, 1, 0]);
        assert_eq!(select(64, [0, 1 << 5, 0], (0, 0, 0)), (-1, BADF));
        assert_eq!(select(64, [0, 0, 0], (0, 0, 0)), (0, SUCCESS));
        assert_eq!(out_set(0), 0);
        assert_eq!(select(1, [1, 0, 0], (0, 1_000_000, 0)), (-1, INVAL));
        assert_eq!(select(-1, [0, 0, 0], (0, 0, 1)), (-1, INVAL));
    }
    #[test]
    fn blocking_reads_and_polls_wait_for_a_write() {
        let guest = guest();
        let bsd = guest.service("bsd:s");
        let fd = event_fd(&guest, bsd, EVENTFD_SEMAPHORE, 0);
        let (out, fds, polled, input) = (
            guest.heap,
            guest.heap + 0x100,
            guest.heap + 0x200,
            guest.heap + 0x300,
        );
        let mut pollfd = fd.to_le_bytes().to_vec();
        pollfd.extend_from_slice(&POLL_IN.to_le_bytes());
        pollfd.extend_from_slice(&0u16.to_le_bytes());
        guest.write(fds, &pollfd);
        guest.write(input, &2u64.to_le_bytes());
        let (poller, writer) = (guest.other_client(), guest.other_client());

        std::thread::scope(|scope| {
            let read = scope.spawn(|| {
                let request = Request::new(25).arg(fd).buffer_b(out, 8);
                result(&guest.call(bsd, request).ok())
            });
            let poll = scope.spawn(|| {
                let request = Request::new(6)
                    .arg(1i32)
                    .arg(-1i32)
                    .buffer_a(fds, 8)
                    .buffer_b(polled, 8);
                result(&poller.send(bsd, &request).unwrap().ok())
            });
            std::thread::sleep(std::time::Duration::from_millis(200));
            assert!(
                !read.is_finished() && !poll.is_finished(),
                "nothing to read yet"
            );

            let write = Request::new(24).arg(fd).buffer_a(input, 8);
            let written = result(&writer.send(bsd, &write).unwrap().ok());
            assert_eq!(written, (8, SUCCESS));
            assert_eq!(read.join().unwrap(), (8, SUCCESS));
            assert_eq!(poll.join().unwrap(), (1, SUCCESS));
        });
        assert_eq!(guest.read(out, 8), 1u64.to_le_bytes());
        assert_eq!(guest.read(polled + 6, 2), POLL_IN.to_le_bytes());
    }

    #[test]
    fn closed_fds_with_blocked_io_are_not_reused_until_it_retries() {
        let event_fd = || {
            Arc::new(Mutex::new(EventFd {
                counter: 0,
                semaphore: false,
                status_flags: 0,
            }))
        };
        let mut descriptors = Descriptors::new();
        assert_eq!(descriptors.insert(event_fd()), Some(0));
        descriptors.blocked.push(Blocked {
            fd: 0,
            thread: 7,
            closed: false,
        });
        descriptors.slots[0] = None;
        descriptors.blocked[0].closed = true;
        assert_eq!(descriptors.insert(event_fd()), Some(1));
        assert!(descriptors.take_for_io(0, 8).is_none());
        assert_eq!(descriptors.insert(event_fd()), Some(2));
        assert!(descriptors.take_for_io(0, 7).is_none());
        assert_eq!(descriptors.insert(event_fd()), Some(0));
        assert!(descriptors.take_for_io(0, 7).is_some());
        assert!(descriptors.take_for_io(-1, 7).is_none());
        assert!(descriptors.take_for_io(MAX_FD as i32, 7).is_none());
    }
}
