//! IPC supervisor calls: named ports and synchronous requests to HLE sessions.

use lichii_common::{PAGE_SIZE, VAddr};

use super::{Flow, SvcContext};
use crate::ipc::ClientSession;
use crate::object::KObject;
use crate::result;
use crate::thread::{Resume, TLS_SIZE};

/// Size of the IPC message at the start of a thread's TLS.
const TLS_MESSAGE_SIZE: usize = 0x100;
const _: () = assert!(TLS_MESSAGE_SIZE as u64 <= TLS_SIZE);

/// `svcConnectToNamedPort(name x1)` -> `w1` session handle.
pub(super) fn connect_to_named_port(svc: &mut SvcContext) -> Flow {
    let result = read_port_name(svc, svc.arg(1)).and_then(|name| {
        let session = svc
            .kernel
            .connect_named_port(&name)
            .ok_or(result::NOT_FOUND)?;
        svc.process()
            .handles()
            .write()
            .add(KObject::ClientSession(session))
    });
    match result {
        Ok(handle) => {
            svc.set(1, handle as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// Port names are NUL terminated and at most 11 characters.
fn read_port_name(svc: &SvcContext, addr: VAddr) -> Result<String, lichii_common::ResultCode> {
    let mut name = Vec::with_capacity(12);
    for i in 0..12 {
        let byte: u8 = svc
            .process()
            .memory()
            .read(addr + i)
            .map_err(|_| result::INVALID_POINTER)?;
        if byte == 0 {
            return String::from_utf8(name).map_err(|_| result::NOT_FOUND);
        }
        name.push(byte);
    }
    Err(result::OUT_OF_RANGE)
}

/// `svcSendSyncRequest(handle w0)`: the message is in the thread's TLS.
pub(super) fn send_sync_request(svc: &mut SvcContext) -> Flow {
    let message = svc.thread.tls_address();
    send(svc, svc.arg32(0), message, TLS_MESSAGE_SIZE)
}

/// `svcSendSyncRequestWithUserBuffer(buffer x1, size x2, handle w3)`.
pub(super) fn send_sync_request_with_user_buffer(svc: &mut SvcContext) -> Flow {
    let (buffer, size) = (svc.arg(1), svc.arg(2));
    if !buffer.is_multiple_of(PAGE_SIZE) {
        return svc.finish(Err(result::INVALID_ADDRESS));
    }
    if size == 0 || !size.is_multiple_of(PAGE_SIZE) || buffer.checked_add(size).is_none() {
        return svc.finish(Err(result::INVALID_SIZE));
    }
    send(svc, svc.arg32(3), buffer, size as usize)
}

fn send(svc: &mut SvcContext, handle: u32, message: VAddr, size: usize) -> Flow {
    let session = match svc.typed::<ClientSession>(handle) {
        Ok(session) => session,
        Err(code) => return svc.finish(Err(code)),
    };
    let token = match svc.thread.begin_wait(Resume::Result, false) {
        Ok(token) => token,
        Err(code) => return svc.finish(Err(code)),
    };
    session.send(svc.thread.clone(), token, message, size);
    svc.block(token, None, false)
}
