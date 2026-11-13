//! The layer between the kernel and each service: unknown commands and IPC tracing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use lichii_common::{ConsoleGeneration, ResultCode};
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
            let stubbed = auto_stub(flags.auto_stub, name, self.context.program.generation);
            if self.context.unknown.first(name, command) {
                tracing::warn!(
                    service = name,
                    command,
                    tipc = ctx.is_tipc(),
                    stubbed,
                    "unimplemented service command"
                );
            }
            if stubbed {
                push_stub_zeroes(ctx);
                ResultCode::SUCCESS
            } else {
                UNKNOWN_COMMAND_ID
            }
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

/// Prefix of Switch 2 interfaces whose commands are not known yet; their unknown commands are
/// always auto-stubbed.
pub(crate) const NX2_PREFIX: &str = "nx2:";

/// Words of zeroes an auto-stubbed reply carries, so a caller that expects output reads zeroes
/// rather than stale message words.
const STUB_WORDS: usize = 32;

/// Whether an unimplemented command answers success with zeroes instead of `UnknownCommandId`:
/// with `debug.auto_stub` (like `EDEN_NX2_AUTO_STUB`), on `nx2:` interfaces, and for Switch 2
/// titles, which still call many commands nobody has implemented.
fn auto_stub(setting: bool, service: &str, title: ConsoleGeneration) -> bool {
    setting || service.starts_with(NX2_PREFIX) || title == ConsoleGeneration::Switch2
}

/// Pushes the zeroes of an auto-stubbed reply.
pub(crate) fn push_stub_zeroes(ctx: &mut HleRequestContext) {
    ctx.push_bytes(&[0; STUB_WORDS * 4]);
}

/// A Switch 2 interface nobody has implemented: every command gets the auto-stub reply.
pub(crate) struct Nx2Stub(pub String);

impl Service for Nx2Stub {
    fn name(&self) -> &str {
        &self.0
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

/// Answers with a new [`Nx2Stub`] named after `owner` and the command, for Switch 2 getters whose
/// interface is unknown.
pub(crate) fn push_nx2_stub(
    context: &Arc<ServiceContext>,
    ctx: &mut HleRequestContext,
    owner: &str,
) -> ResultCode {
    let name = format!("{NX2_PREFIX}{owner}::{}", ctx.command());
    tracing::warn!(interface = %name, "returning a Switch 2 stub interface");
    ctx.push_interface(context.handler(Nx2Stub(name)));
    ResultCode::SUCCESS
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
    fn auto_stub_covers_the_setting_nx2_interfaces_and_switch2_titles() {
        use ConsoleGeneration::{Switch1, Switch2};
        assert!(auto_stub(true, "hid", Switch1));
        assert!(!auto_stub(false, "hid", Switch1));
        assert!(auto_stub(false, "nx2:IGraphicsScaleController", Switch1));
        assert!(auto_stub(false, "hid", Switch2));
    }
}
