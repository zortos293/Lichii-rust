# Kernel IPC for HLE services

This is the API the services crate uses to serve guest IPC. Everything lives in
`lichii_kernel::ipc` and `lichii_kernel::sm`.

## How a request flows

1. The guest calls `svcSendSyncRequest` (0x21) or `svcSendSyncRequestWithUserBuffer` (0x22) on
   a client session handle. Its thread blocks. No core waits for the service.
2. The request is queued on the [`ServiceThread`] that serves the session. The service thread
   reads the message from the client's TLS (0x100 bytes) or user buffer and parses it: CMIF,
   CMIF domains or TIPC.
3. The service thread translates the copy and move handles, then calls
   `SessionHandler::handle_request` with a [`HleRequestContext`].
4. The kernel encodes the reply in the same buffer, installs the returned handles in the client's
   handle table and wakes the client thread with `x0 = 0`. The IPC result code is in the reply.
   This happens under the client thread's lock: a client that was cancelled or terminated while
   the handler ran gets nothing, and a reply that cannot be written is rolled back.

Each service thread handles one request at a time and reuses one context, so pops, pushes and
descriptor lookups allocate nothing. Handlers on different service threads run in parallel with
each other and with the guest cores.

## Registering a service

```rust
use std::sync::Arc;
use lichii_kernel::ipc::{HleRequestContext, SessionHandler, UNKNOWN_COMMAND_ID};

let thread = kernel.create_service_thread("fs")?; // one host thread; may be shared by services
kernel.service_manager().register(
    "fsp-srv",
    &thread,
    Arc::new(|| Arc::new(FspSrv::default()) as Arc<dyn SessionHandler>),
)?;
```

- `register(name, thread, factory)`: `factory` makes the handler of each new session, so every
  `GetServiceHandle` gets its own handler state. Names are 1 to 8 bytes. Registering a name twice
  fails with `sm::ALREADY_REGISTERED`.
- Register before or after `Kernel::run`. Service threads start with `run` (or at once, if the
  kernel already runs). A guest that asks for a name before it is registered stays blocked in
  `GetServiceHandle` until it is registered; `ServiceManager::waiting_for()` lists those names.
- `unregister(name)` and `is_registered(name)` exist for completeness.
- A service whose commands may block (file I/O, waiting for the GPU) should get its own service
  thread, so other services keep answering.
- `Kernel::register_named_port(name, thread, factory)` adds an `svcConnectToNamedPort` port.
  The kernel registers `sm:` itself.

## Writing a handler

```rust
impl SessionHandler for FspSrv {
    fn name(&self) -> &str { "fsp-srv" }

    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        match ctx.command() {
            1 => {                                   // SetCurrentProcess(u64 pid, PID)
                let _reserved: u64 = ctx.pop();
                self.set_pid(ctx.pid());
                ResultCode::SUCCESS
            }
            18 => {                                  // OpenSdCardFileSystem -> IFileSystem
                ctx.push_interface(Arc::new(FileSystem::sd_card()));
                ResultCode::SUCCESS
            }
            _ => UNKNOWN_COMMAND_ID,
        }
    }
}
```

The return value is the reply result. When it is an error, the kernel writes only the result and
drops everything the handler pushed (data, handles, interfaces).

### Request

| Method | What it returns |
| --- | --- |
| `command()` | The CMIF command id, or the TIPC command number. |
| `command_type()`, `is_tipc()`, `is_domain()` | The message flavour, and whether the session is a domain. |
| `pop::<T: Pod>()` | The next input value, naturally aligned from the start of the input data (the layout of the parameter struct). Bytes past the end read as zero. |
| `input()` | The raw input bytes after the CMIF or TIPC header. |
| `pid()` | The client's process id when the request sent its PID, else `None`. |
| `thread()`, `process()`, `memory()` | The client thread, its process and its address space. |
| `copy_object(i)`, `copy_handle::<T>(i)` | The object behind copy handle `i`. Pseudo-handles resolve to the client's process or thread; handle 0 is `None`. |
| `take_move_object(i)` | The object behind move handle `i`. The client has already lost that handle. |

### Buffers

All buffers are addresses in the client's address space. Services read and write them directly;
no pointer-buffer copy is involved.

| Method | What it does |
| --- | --- |
| `buffers_a()`, `buffers_b()`, `buffers_w()`, `buffers_x()`, `buffers_c()` | Raw descriptors (`address`, `size`). |
| `readable_buffer(i)` / `read_buffer(i)` | Input buffer `i`: the A buffer if it is non-empty, else X (HipcAutoSelect). |
| `writable_buffer(i)` / `write_buffer(i, data)` | Output buffer `i`: the B buffer if it is non-empty, else C. Writes are truncated to the buffer size; returns the bytes written. |

### Reply

| Method | What it does |
| --- | --- |
| `push(value)`, `push_bytes(data)` | Appends reply data. `push` aligns naturally from the start of the data. |
| `push_copy_object(KObject)` | Returns a copy handle. The service keeps its own reference, for example `KObject::ReadableEvent(event)`. |
| `push_move_object(KObject)` | Returns a move handle. In a domain it stays a handle. |
| `push_interface(Arc<dyn SessionHandler>)` | Returns a sub-interface: a domain object id when the session is a domain, else a move handle to a new session on the same service thread. |
| `defer()` | Answers later; see below. |

Handles come out in the order they were pushed, copies before moves, as HIPC lays them out.

### Deferred replies

`let request = ctx.defer();` keeps the client blocked. Later, from any thread, call
`request.retry()`: the original message runs through the same session's handler again, on its
service thread, and that run replies. The retry sees the same copy and move objects (move handles
are taken from the client only once). Dropping the `DeferredRequest` instead fails the client's
`SendSyncRequest` with `SessionClosed`. `sm:` uses this to defer `GetServiceHandle`.

### Sessions you create yourself

`ServiceThread::session(handler)` returns an `Arc<ClientSession>`. Push it with
`push_move_object(KObject::ClientSession(session))` when a command must return an existing or
specially built session; `push_interface` is the usual way.

## What the kernel handles itself

- **CMIF control (type 5/7):** `ConvertCurrentObjectToDomain` (returns object id 1),
  `CopyFromCurrentDomain`, `CloneCurrentObject`/`CloneCurrentObjectEx` (the clone shares the
  handler and domain but closes on its own) and `QueryPointerBufferSize` (0x8000).
- **Domains:** `SendMessage` routes to the object id in the domain header;
  `CloseVirtualHandle` drops the object. Object ids start at 1 and freed ids are reused. Domain
  replies carry the domain header and list the new object ids after the data.
- **Close (CMIF type 2, TIPC type 15):** closes the session. Later requests on it fail with
  `SessionClosed`.
- **TIPC replies:** the header type repeats the request's, then one result word, then the data.
  This avoids Eden's two-word CMIF result in TIPC replies (NX2_NOTES row 22).
- **`sm:` (CMIF and TIPC, same ids):** 0 `RegisterClient`, 1 `GetServiceHandle` (fails with
  `sm::INVALID_CLIENT` before `RegisterClient`, and `sm::INVALID_SERVICE_NAME` for malformed
  names). 2/3 (guest `RegisterService`/`UnregisterService`) return `sm::NOT_ALLOWED`, because
  only HLE services register.
  `GetServiceHandle` for a name the client's NPDM service list does not match (`*` ends a
  prefix) is logged, and fails with `sm::NOT_ALLOWED` when
  `Settings::system.enforce_service_access` is on (off by default until NPDM parsing has been
  checked against real titles). Homebrew without an NPDM gets the loader's `*` list.

## Not supported yet

- Guest servers: `CreateSession`, `AcceptSession`, `ReplyAndReceive`, `CreatePort`,
  `ManageNamedPort`, `ConnectToPort` and guest `RegisterService`. Titles do not use them;
  sysmodules would.
- `SendAsyncRequestWithUserBuffer` (0x23), domain input objects, and inline C buffers.
