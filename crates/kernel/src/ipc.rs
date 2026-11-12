//! IPC between guest code and HLE services.
//!
//! The kernel parses HIPC messages (CMIF and TIPC) from a thread's TLS or user buffer into a
//! [\`HleRequestContext\`], hands it to the [\`SessionHandler\`] behind the session, and writes the
//! reply back. The kernel crew owns the context's layout; services only use its methods.

use lichii_common::ResultCode;

/// A kernel handle in a process's handle table.
pub type Handle = u32;

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
            16.. => Self::Tipc(raw as u32 - 16),
            _ => Self::Invalid,
        }
    }
}

/// A parsed request and the reply being built. Defined by the kernel crew.
pub struct HleRequestContext {
    _private: (),
}

/// The server side of an HLE session: one service or sub-interface object.
pub trait SessionHandler: Send + Sync {
    /// Service or interface name for logs (\`fsp-srv\`, \`IFileSystem\`).
    fn name(&self) -> &str;
    /// Handles one request. Returning an error code makes the kernel write it as the reply result.
    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode;
}
