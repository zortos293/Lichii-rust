//! The layer between the kernel and each service: unknown commands and IPC tracing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_kernel::ipc::{HleRequestContext, SessionHandler, UNKNOWN_COMMAND_ID};
use parking_lot::Mutex;

use crate::ServiceContext;

/// A service or sub-interface.
pub trait Service: Send + Sync + 'static {
    /// The name in logs: the sm name for services (`fsp-srv`), the interface for the rest
    /// (`IFileSystem`).
    fn name(&self) -> &str;

    /// Handles one command, or returns `None` when the service does not implement it.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode>;
}

pub(crate) struct Dispatch<S> {
    pub service: S,
    pub context: Arc<ServiceContext>,
}

impl<S: Service> SessionHandler for Dispatch<S> {
    fn name(&self) -> &str {
        self.service.name()
    }

    fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let flags = &self.context.settings.debug;
        let result = self.service.handle(ctx).unwrap_or_else(|| {
            let (name, command) = (self.service.name(), ctx.command());
            if self.context.unknown.first(name, command) {
                tracing::warn!(
                    service = name,
                    command,
                    tipc = ctx.is_tipc(),
                    stubbed = flags.auto_stub,
                    "unimplemented service command"
                );
            }
            stub_result(flags.auto_stub)
        });
        if flags.ipc_trace {
            tracing::info!(
                target: "ipc",
                service = self.service.name(),
                command = ctx.command(),
                tipc = ctx.is_tipc(),
                result = format_args!("{:#x}", result.0),
            );
        }
        result
    }
}

/// What an unimplemented command answers: an empty success with `debug.auto_stub` (like
/// `EDEN_NX2_AUTO_STUB`), otherwise `UnknownCommandId`.
fn stub_result(auto_stub: bool) -> ResultCode {
    if auto_stub {
        ResultCode::SUCCESS
    } else {
        UNKNOWN_COMMAND_ID
    }
}

/// The (service, command) pairs already reported, so each is logged once.
#[derive(Default)]
pub(crate) struct UnknownCommands {
    seen: Mutex<HashMap<String, HashSet<u32>>>,
}

impl UnknownCommands {
    /// Whether this is the first time `service` got `command`.
    pub fn first(&self, service: &str, command: u32) -> bool {
        let mut seen = self.seen.lock();
        match seen.get_mut(service) {
            Some(commands) => commands.insert(command),
            None => {
                seen.insert(service.into(), HashSet::from([command]));
                true
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_commands_are_reported_once_per_service_and_command() {
        let unknown = UnknownCommands::default();
        assert!(unknown.first("appletOE", 1));
        assert!(!unknown.first("appletOE", 1));
        assert!(unknown.first("appletOE", 2));
        assert!(unknown.first("hid", 1));
        assert!(!unknown.first("hid", 1));
    }

    #[test]
    fn auto_stub_answers_success() {
        assert_eq!(stub_result(true), ResultCode::SUCCESS);
        assert_eq!(stub_result(false), UNKNOWN_COMMAND_ID);
    }
}
