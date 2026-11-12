//! Service threads, HLE sessions and the request loop.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use lichii_common::{ResultCode, VAddr};
use parking_lot::{Condvar, Mutex};

use super::message::{DomainCommand, HANDLES_OFFSET, ReplyKind, encode_reply};
use super::{CommandType, HleRequestContext, SessionHandler, UNKNOWN_COMMAND_ID};
use crate::handle_table::{CURRENT_PROCESS, CURRENT_THREAD};
use crate::object::KObject;
use crate::result;
use crate::thread::{Thread, WakeResult};

/// Largest message read from a user buffer (`SendSyncRequestWithUserBuffer`).
const MAX_MESSAGE: usize = 0x2000;
/// Pointer buffer size reported by `QueryPointerBufferSize`.
const POINTER_BUFFER_SIZE: u16 = 0x8000;

/// A request waiting for its service thread.
pub(crate) struct Pending {
    session: Arc<HleSession>,
    thread: Arc<Thread>,
    token: u64,
    message: VAddr,
    size: usize,
    /// Copy and move objects already taken from the client, kept by a deferred request so a
    /// retry does not translate the handles again.
    inputs: Option<(InputObjects, InputObjects)>,
}

/// Objects behind a request's copy or move handles, in order (`None` for handle 0).
type InputObjects = Vec<Option<KObject>>;

impl Pending {
    pub(crate) fn keep_inputs(&mut self, copies: InputObjects, moves: InputObjects) {
        self.inputs = Some((copies, moves));
    }
}

struct Queue {
    requests: VecDeque<Pending>,
    stopped: bool,
}

struct Shared {
    name: String,
    queue: Mutex<Queue>,
    ready: Condvar,
}

/// A host thread that serves HLE sessions one request at a time. Every session belongs to one
/// service thread; a service that may block (file I/O) gets its own so others keep answering.
#[derive(Clone)]
pub struct ServiceThread(Arc<Shared>);

impl ServiceThread {
    pub(crate) fn new(name: &str) -> Self {
        Self(Arc::new(Shared {
            name: name.into(),
            queue: Mutex::new(Queue {
                requests: VecDeque::new(),
                stopped: false,
            }),
            ready: Condvar::new(),
        }))
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// A new session served by this thread. Hand it to the guest with
    /// [`HleRequestContext::push_move_object`] or put it in a handle table.
    pub fn session(&self, handler: Arc<dyn SessionHandler>) -> Arc<ClientSession> {
        let server = Arc::new(Server {
            thread: self.clone(),
            target: Mutex::new(Target::Single(handler)),
        });
        Arc::new(ClientSession {
            session: HleSession::new(server),
        })
    }

    pub(crate) fn spawn(&self) -> std::io::Result<JoinHandle<()>> {
        let this = self.clone();
        std::thread::Builder::new()
            .name(format!("service {}", self.0.name))
            .spawn(move || this.run())
    }

    /// Stops the thread after the request in progress; queued requests are dropped.
    pub(crate) fn stop(&self) {
        let dropped = {
            let mut queue = self.0.queue.lock();
            queue.stopped = true;
            std::mem::take(&mut queue.requests)
        };
        self.0.ready.notify_all();
        drop(dropped);
    }

    fn submit(&self, pending: Pending) {
        let mut queue = self.0.queue.lock();
        if queue.stopped {
            return;
        }
        queue.requests.push_back(pending);
        drop(queue);
        self.0.ready.notify_one();
    }

    fn run(&self) {
        let mut ctx = HleRequestContext::new();
        loop {
            let pending = {
                let mut queue = self.0.queue.lock();
                loop {
                    if queue.stopped {
                        return;
                    }
                    if let Some(pending) = queue.requests.pop_front() {
                        break pending;
                    }
                    self.0.ready.wait(&mut queue);
                }
            };
            self.serve(&mut ctx, pending);
        }
    }

    fn serve(&self, ctx: &mut HleRequestContext, pending: Pending) {
        if !pending.thread.is_waiting_on(pending.token) {
            return;
        }
        let (thread, token) = (pending.thread.clone(), pending.token);
        let outcome = self.dispatch(ctx, pending);
        ctx.pending = None;
        ctx.thread = None;
        ctx.server = None;
        ctx.copy_objects.clear();
        ctx.move_objects.clear();
        clear_outputs(ctx);
        if let Err(code) = outcome {
            thread.wake(token, WakeResult::error(code));
        }
    }

    /// Runs one request. On success the client was woken with the reply, or the handler
    /// deferred the request.
    fn dispatch(
        &self,
        ctx: &mut HleRequestContext,
        mut pending: Pending,
    ) -> Result<(), ResultCode> {
        let session = pending.session.clone();
        let server = session.server.clone();
        let thread = pending.thread.clone();
        let process = thread.process().clone();
        let words = pending.size.min(MAX_MESSAGE) / 4;
        ctx.message.clear();
        ctx.message.resize(words, 0);
        process
            .memory()
            .read_bytes(pending.message, bytemuck::cast_slice_mut(&mut ctx.message))
            .map_err(|_| result::INVALID_CURRENT_MEMORY)?;
        let domain = session.is_domain().ok_or(result::SESSION_CLOSED)?;
        ctx.request.parse(&ctx.message, domain)?;
        if matches!(ctx.request.ty, CommandType::Close | CommandType::TipcClose) {
            session.close();
            thread.wake(pending.token, WakeResult::SUCCESS);
            return Ok(());
        }
        ctx.domain = domain;
        ctx.cursor = 0;
        ctx.thread = Some(thread.clone());
        ctx.server = Some(self.clone());
        if let Some((copies, moves)) = pending.inputs.take() {
            ctx.copy_objects = copies;
            ctx.move_objects = moves;
        } else {
            let mut table = process.handles().write();
            for &handle in &ctx.request.copy_handles {
                let object = match handle {
                    0 => None,
                    CURRENT_THREAD => Some(KObject::Thread(thread.clone())),
                    CURRENT_PROCESS => Some(KObject::Process(process.clone())),
                    _ => Some(table.get(handle).ok_or(result::INVALID_HANDLE)?),
                };
                ctx.copy_objects.push(object);
            }
            for &handle in &ctx.request.move_handles {
                let object = match handle {
                    0 => None,
                    _ => Some(table.remove(handle).ok_or(result::INVALID_HANDLE)?),
                };
                ctx.move_objects.push(object);
            }
        }
        ctx.pending = Some(pending);
        let code = match ctx.request.ty {
            CommandType::Control | CommandType::ControlWithContext => control(ctx, &session),
            _ => match ctx.request.domain {
                Some((DomainCommand::CloseVirtualHandle, id)) => {
                    drop(server.remove_object(id));
                    ResultCode::SUCCESS
                }
                Some((DomainCommand::SendMessage, id)) => match server.object(id) {
                    Some(handler) => handler.handle_request(ctx),
                    None => result::NOT_FOUND,
                },
                None => match server.handler() {
                    Some(handler) => handler.handle_request(ctx),
                    None => result::SESSION_CLOSED,
                },
            },
        };
        let Some(pending) = ctx.pending.take() else {
            return Ok(());
        };
        reply(ctx, &server, &pending, code)
    }
}

fn clear_outputs(ctx: &mut HleRequestContext) {
    ctx.out_data.clear();
    ctx.out_copy.clear();
    ctx.out_move.clear();
    ctx.out_objects.clear();
}

/// Session control requests (CMIF type 5): domains, cloning and the pointer buffer size.
fn control(ctx: &mut HleRequestContext, session: &Arc<HleSession>) -> ResultCode {
    match ctx.request.command {
        0 => match session.server.convert_to_domain() {
            Some(id) => {
                ctx.push(id);
                ResultCode::SUCCESS
            }
            None => result::INVALID_STATE,
        },
        1 => {
            let id: u32 = ctx.pop();
            let Some(handler) = session.server.object(id) else {
                return result::NOT_FOUND;
            };
            let copy = session.server.thread.session(handler);
            ctx.push_move_object(KObject::ClientSession(copy));
            ResultCode::SUCCESS
        }
        2 | 4 => {
            let clone = Arc::new(ClientSession {
                session: HleSession::new(session.server.clone()),
            });
            ctx.push_move_object(KObject::ClientSession(clone));
            ResultCode::SUCCESS
        }
        3 => {
            ctx.push(POINTER_BUFFER_SIZE);
            ResultCode::SUCCESS
        }
        command => {
            tracing::warn!(command, "unknown IPC control command");
            UNKNOWN_COMMAND_ID
        }
    }
}

/// Writes the reply into the client's message buffer, installs the returned handles in its table
/// and wakes it, all under the client thread's lock: a client that was cancelled or terminated
/// meanwhile gets nothing, and a reply that cannot be written is rolled back.
fn reply(
    ctx: &mut HleRequestContext,
    server: &Server,
    pending: &Pending,
    code: ResultCode,
) -> Result<(), ResultCode> {
    if !code.is_success() {
        clear_outputs(ctx);
    }
    let kind = match ctx.request.ty {
        CommandType::Tipc(command) => ReplyKind::Tipc(command as u16 + 16),
        _ => ReplyKind::Cmif {
            domain: ctx.request.domain.map(|_| ctx.out_objects.len() as u32),
        },
    };
    let (copies, moves, objects) = (
        ctx.out_copy.len(),
        ctx.out_move.len(),
        ctx.out_objects.len(),
    );
    ctx.handles.clear();
    ctx.handles.resize(copies + moves + objects, 0);
    let (handles, ids) = ctx.handles.split_at(copies + moves);
    let (copy_slots, move_slots) = handles.split_at(copies);
    encode_reply(
        &mut ctx.reply,
        kind,
        code,
        &ctx.out_data,
        copy_slots,
        move_slots,
        ids,
    );
    if ctx.reply.len() * 4 > pending.size {
        return Err(result::MESSAGE_TOO_LARGE);
    }
    let process = pending.thread.process();
    let installed = HANDLES_OFFSET..HANDLES_OFFSET + copies + moves;
    let mut undone = Vec::new();
    let mut s = pending.thread.sched.lock();
    if s.wait != pending.token {
        return Ok(());
    }
    {
        let mut table = process.handles().write();
        let returned = ctx.out_copy.drain(..).chain(ctx.out_move.drain(..));
        for (i, object) in returned.enumerate() {
            match table.add(object) {
                Ok(handle) => ctx.reply[HANDLES_OFFSET + i] = handle,
                Err(code) => {
                    for &handle in &ctx.reply[HANDLES_OFFSET..HANDLES_OFFSET + i] {
                        undone.extend(table.remove(handle));
                    }
                    drop(table);
                    drop(s);
                    drop(undone);
                    return Err(code);
                }
            }
        }
    }
    let object_ids = ctx.reply.len() - objects..ctx.reply.len();
    if objects != 0 {
        server.add_objects(
            ctx.out_objects.drain(..),
            &mut ctx.reply[object_ids.clone()],
        );
    }
    let written = process
        .memory()
        .write_bytes(pending.message, bytemuck::cast_slice(&ctx.reply));
    if written.is_err() {
        let mut table = process.handles().write();
        for &handle in &ctx.reply[installed] {
            undone.extend(table.remove(handle));
        }
        drop(table);
        let removed: Vec<_> = ctx.reply[object_ids]
            .iter()
            .filter_map(|&id| server.remove_object(id))
            .collect();
        drop(s);
        drop((undone, removed));
        return Err(result::INVALID_CURRENT_MEMORY);
    }
    pending.thread.wake_locked(&mut s, WakeResult::SUCCESS);
    Ok(())
}

enum Target {
    Single(Arc<dyn SessionHandler>),
    /// Domain objects; object id `n` is slot `n - 1`.
    Domain(Vec<Option<Arc<dyn SessionHandler>>>),
}

/// What the client ends of one session share: its service thread and handler (or domain).
struct Server {
    thread: ServiceThread,
    target: Mutex<Target>,
}

impl Server {
    fn is_domain(&self) -> bool {
        matches!(*self.target.lock(), Target::Domain(_))
    }

    fn handler(&self) -> Option<Arc<dyn SessionHandler>> {
        match &*self.target.lock() {
            Target::Single(handler) => Some(handler.clone()),
            Target::Domain(_) => None,
        }
    }

    fn object(&self, id: u32) -> Option<Arc<dyn SessionHandler>> {
        match &*self.target.lock() {
            Target::Domain(objects) => objects
                .get((id as usize).checked_sub(1)?)?
                .as_ref()
                .cloned(),
            Target::Single(_) => None,
        }
    }

    /// Removes a domain object; the caller drops it outside the lock.
    fn remove_object(&self, id: u32) -> Option<Arc<dyn SessionHandler>> {
        match &mut *self.target.lock() {
            Target::Domain(objects) => objects.get_mut((id as usize).checked_sub(1)?)?.take(),
            Target::Single(_) => None,
        }
    }

    fn add_objects(
        &self,
        handlers: impl Iterator<Item = Arc<dyn SessionHandler>>,
        ids: &mut [u32],
    ) {
        let mut target = self.target.lock();
        let Target::Domain(objects) = &mut *target else {
            return;
        };
        for (handler, id) in handlers.zip(ids) {
            let index = match objects.iter().position(Option::is_none) {
                Some(index) => {
                    objects[index] = Some(handler);
                    index
                }
                None => {
                    objects.push(Some(handler));
                    objects.len() - 1
                }
            };
            *id = index as u32 + 1;
        }
    }

    fn convert_to_domain(&self) -> Option<u32> {
        let mut target = self.target.lock();
        let Target::Single(handler) = &*target else {
            return None;
        };
        *target = Target::Domain(vec![Some(handler.clone())]);
        Some(1)
    }
}

/// One client end of a session. Clones share the [`Server`] but close independently.
struct HleSession {
    server: Arc<Server>,
    closed: AtomicBool,
}

impl HleSession {
    fn new(server: Arc<Server>) -> Arc<Self> {
        Arc::new(Self {
            server,
            closed: AtomicBool::new(false),
        })
    }

    /// Whether the session is a domain; `None` once this end was closed.
    fn is_domain(&self) -> Option<bool> {
        (!self.closed.load(Ordering::Acquire)).then(|| self.server.is_domain())
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
}

/// The client end of an HLE session, as the guest holds it in its handle table. Clones made by
/// `CloneCurrentObject` share the server end.
pub struct ClientSession {
    session: Arc<HleSession>,
}

impl ClientSession {
    /// Name of the handler behind the session, for logs.
    pub fn name(&self) -> String {
        match &*self.session.server.target.lock() {
            Target::Single(handler) => handler.name().into(),
            Target::Domain(_) => "domain".into(),
        }
    }

    /// Queues a request whose message is at `message` (`size` bytes) in `thread`'s process.
    /// The thread must be in the wait `token`; the service thread wakes it with the result.
    pub(crate) fn send(&self, thread: Arc<Thread>, token: u64, message: VAddr, size: usize) {
        self.session.server.thread.submit(Pending {
            session: self.session.clone(),
            thread,
            token,
            message,
            size,
            inputs: None,
        });
    }
}

/// A request a handler answers later (see [`HleRequestContext::defer`]).
pub struct DeferredRequest {
    pending: Option<Pending>,
}

impl DeferredRequest {
    pub(crate) fn new(pending: Pending) -> Self {
        Self {
            pending: Some(pending),
        }
    }

    /// Runs the request through its session's handler again, on the session's service thread.
    pub fn retry(mut self) {
        if let Some(pending) = self.pending.take() {
            pending.session.server.thread.clone().submit(pending);
        }
    }
}

impl Drop for DeferredRequest {
    fn drop(&mut self) {
        if let Some(pending) = self.pending.take() {
            pending
                .thread
                .wake(pending.token, WakeResult::error(result::SESSION_CLOSED));
        }
    }
}
