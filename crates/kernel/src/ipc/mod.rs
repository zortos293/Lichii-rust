//! IPC between guest code and HLE services.
//!
//! A guest thread that sends a request on a client session blocks; the request goes to the
//! [`ServiceThread`] that serves the session, which parses the HIPC message (CMIF, CMIF domains
//! or TIPC) from the thread's TLS or user buffer into a [`HleRequestContext`], runs the
//! [`SessionHandler`], writes the reply into the same buffer and wakes the thread. The kernel
//! crew owns the context's layout; services only use its methods. See `crates/kernel/IPC.md`.

mod message;
mod server;

use std::sync::Arc;

use bytemuck::Pod;
use lichii_common::{HleResult, ResultCode};
use lichii_memory::AddressSpace;

pub use message::BufferDescriptor;
pub use server::{ClientSession, DeferredRequest, ServiceThread};

use crate::object::{FromObject, KObject};
use crate::process::Process;
use crate::result;
use crate::thread::Thread;

/// A kernel handle in a process's handle table.
pub type Handle = u32;

/// Makes the handler of each new session to a service or named port.
pub type SessionFactory = Arc<dyn Fn() -> Arc<dyn SessionHandler> + Send + Sync>;

/// `sf::cmif::ResultUnknownCommandId` (2010-0221): what a handler returns for a command id it
/// does not know.
pub const UNKNOWN_COMMAND_ID: ResultCode = ResultCode::new(10, 221);

/// HIPC message type (header word 0, bits 0..16). Values 16 and up are TIPC command ids + 16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandType {
    Invalid,
    LegacyRequest,
    Close,
    LegacyControl,
    Request,
    Control,
    RequestWithContext,
    ControlWithContext,
    TipcClose,
    Tipc(u32),
}

impl CommandType {
    pub fn from_raw(raw: u16) -> Self {
        match raw {
            0 => Self::Invalid,
            1 => Self::LegacyRequest,
            2 => Self::Close,
            3 => Self::LegacyControl,
            4 => Self::Request,
            5 => Self::Control,
            6 => Self::RequestWithContext,
            7 => Self::ControlWithContext,
            15 => Self::TipcClose,
            16.. => Self::Tipc(raw as u32 - 16),
            _ => Self::Invalid,
        }
    }
}

/// The server side of an HLE session: one service or sub-interface object.
pub trait SessionHandler: Send + Sync {
    /// Service or interface name for logs (`fsp-srv`, `IFileSystem`).
    fn name(&self) -> &str;
    /// Handles one request. Returning an error code makes the kernel write it as the reply result
    /// and drop everything pushed into the reply.
    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode;
}

/// A parsed request and the reply being built. One context per service thread is reused for
/// every request, so pops, pushes and buffer lookups allocate nothing.
pub struct HleRequestContext {
    pending: Option<server::Pending>,
    thread: Option<Arc<Thread>>,
    server: Option<ServiceThread>,
    domain: bool,
    message: Vec<u32>,
    request: message::Request,
    copy_objects: Vec<Option<KObject>>,
    move_objects: Vec<Option<KObject>>,
    cursor: usize,
    out_data: Vec<u8>,
    out_copy: Vec<KObject>,
    out_move: Vec<KObject>,
    out_objects: Vec<Arc<dyn SessionHandler>>,
    reply: Vec<u32>,
    handles: Vec<u32>,
}

impl HleRequestContext {
    pub(crate) fn new() -> Self {
        Self {
            pending: None,
            thread: None,
            server: None,
            domain: false,
            message: Vec::new(),
            request: message::Request::default(),
            copy_objects: Vec::new(),
            move_objects: Vec::new(),
            cursor: 0,
            out_data: Vec::new(),
            out_copy: Vec::new(),
            out_move: Vec::new(),
            out_objects: Vec::new(),
            reply: Vec::new(),
            handles: Vec::new(),
        }
    }

    /// The command id (CMIF) or TIPC command number.
    pub fn command(&self) -> u32 {
        self.request.command
    }

    pub fn command_type(&self) -> CommandType {
        self.request.ty
    }

    pub fn is_tipc(&self) -> bool {
        matches!(self.request.ty, CommandType::Tipc(_))
    }

    /// Whether the session is a domain (interfaces pushed become domain objects).
    pub fn is_domain(&self) -> bool {
        self.domain
    }

    /// The client's process id, when the request asked the kernel to send it.
    pub fn pid(&self) -> Option<u64> {
        self.request.send_pid.then(|| self.process().id())
    }

    /// The requesting guest thread.
    pub fn thread(&self) -> &Arc<Thread> {
        self.thread.as_ref().expect("request in progress")
    }

    pub fn process(&self) -> &Arc<Process> {
        self.thread().process()
    }

    /// The client's address space, where every buffer lives.
    pub fn memory(&self) -> &Arc<AddressSpace> {
        self.process().memory()
    }

    /// The raw input data after the CMIF or TIPC header.
    pub fn input(&self) -> &[u8] {
        &bytemuck::cast_slice::<u32, u8>(&self.message)[self.request.payload.clone()]
    }

    /// Reads the next input value, naturally aligned from the start of the input data (the
    /// layout of a CMIF/TIPC parameter struct). Missing bytes read as zero.
    pub fn pop<T: Pod>(&mut self) -> T {
        let size = std::mem::size_of::<T>();
        let start = self
            .cursor
            .next_multiple_of(std::mem::align_of::<T>().min(8));
        self.cursor = start + size;
        self.input()
            .get(start..start + size)
            .map_or_else(T::zeroed, bytemuck::pod_read_unaligned)
    }

    /// The object behind copy handle `index` (pseudo-handles resolve to the client's process
    /// and thread; handle 0 is `None`).
    pub fn copy_object(&self, index: usize) -> Option<&KObject> {
        self.copy_objects.get(index)?.as_ref()
    }

    /// Typed access to copy handle `index`.
    pub fn copy_handle<T: FromObject>(&self, index: usize) -> HleResult<Arc<T>> {
        self.copy_object(index)
            .and_then(T::from_object)
            .ok_or(result::INVALID_HANDLE)
    }

    /// Takes the object behind move handle `index`; the client no longer has the handle.
    pub fn take_move_object(&mut self, index: usize) -> Option<KObject> {
        self.move_objects.get_mut(index)?.take()
    }

    pub fn buffers_a(&self) -> &[BufferDescriptor] {
        &self.request.a
    }

    pub fn buffers_b(&self) -> &[BufferDescriptor] {
        &self.request.b
    }

    pub fn buffers_w(&self) -> &[BufferDescriptor] {
        &self.request.w
    }

    pub fn buffers_x(&self) -> &[BufferDescriptor] {
        &self.request.x
    }

    pub fn buffers_c(&self) -> &[BufferDescriptor] {
        &self.request.c
    }

    /// Input buffer `index`: the A (map) buffer when it is non-empty, else the X (pointer)
    /// buffer, as HipcAutoSelect sends them.
    pub fn readable_buffer(&self, index: usize) -> Option<BufferDescriptor> {
        let a = self.request.a.get(index).filter(|d| d.size != 0);
        a.or_else(|| self.request.x.get(index)).copied()
    }

    /// Output buffer `index`: the B (map) buffer when it is non-empty, else the C (receive)
    /// buffer.
    pub fn writable_buffer(&self, index: usize) -> Option<BufferDescriptor> {
        let b = self.request.b.get(index).filter(|d| d.size != 0);
        b.or_else(|| self.request.c.get(index)).copied()
    }

    /// Reads input buffer `index` (see [`HleRequestContext::readable_buffer`]).
    pub fn read_buffer(&self, index: usize) -> HleResult<Vec<u8>> {
        let buffer = self.readable_buffer(index).ok_or(result::INVALID_POINTER)?;
        let mut data = vec![0; buffer.size as usize];
        self.memory()
            .read_bytes(buffer.address, &mut data)
            .map_err(|_| result::INVALID_CURRENT_MEMORY)?;
        Ok(data)
    }

    /// Writes `data` into output buffer `index`, truncated to its size. Returns the bytes written.
    pub fn write_buffer(&self, index: usize, data: &[u8]) -> HleResult<usize> {
        let buffer = self.writable_buffer(index).ok_or(result::INVALID_POINTER)?;
        let len = data.len().min(buffer.size as usize);
        self.memory()
            .write_bytes(buffer.address, &data[..len])
            .map_err(|_| result::INVALID_CURRENT_MEMORY)?;
        Ok(len)
    }

    /// Appends a value to the reply data, naturally aligned from its start.
    pub fn push<T: Pod>(&mut self, value: T) {
        let start = self
            .out_data
            .len()
            .next_multiple_of(std::mem::align_of::<T>().min(8));
        self.out_data.resize(start, 0);
        self.out_data.extend_from_slice(bytemuck::bytes_of(&value));
    }

    pub fn push_bytes(&mut self, data: &[u8]) {
        self.out_data.extend_from_slice(data);
    }

    /// Returns an object as a copy handle: the client gets a handle and the service keeps its
    /// reference.
    pub fn push_copy_object(&mut self, object: KObject) {
        self.out_copy.push(object);
    }

    /// Returns an object as a move handle (also in domains).
    pub fn push_move_object(&mut self, object: KObject) {
        self.out_move.push(object);
    }

    /// Returns a sub-interface: a domain object id when the session is a domain, otherwise a
    /// move handle to a new session served by the same service thread.
    pub fn push_interface(&mut self, handler: Arc<dyn SessionHandler>) {
        if self.domain {
            self.out_objects.push(handler);
        } else {
            let session = self
                .server
                .as_ref()
                .expect("request in progress")
                .session(handler);
            self.out_move.push(KObject::ClientSession(session));
        }
    }

    /// Answers later: the client stays blocked until the returned request is retried (it runs
    /// through the session's handler again) or dropped (the client gets `SessionClosed`). The
    /// handler's return value and anything pushed are ignored.
    pub fn defer(&mut self) -> DeferredRequest {
        DeferredRequest::new(self.pending.take().expect("request already deferred"))
    }
}
