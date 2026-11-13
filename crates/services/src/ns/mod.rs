//! ns: application management for the running title (control data, languages, records,
//! documents) behind the ns:am2/ec/rid/rt/web/ro service getters, plus ns:su, ns:vm, ns:dev and
//! pdm:ntfy. pl:u and pl:s live with the shared font elsewhere.

mod application_manager;
mod control;
mod interfaces;
mod language;

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;

use self::application_manager::ApplicationManager;
use self::control::ReadOnlyApplicationControlData;
use self::interfaces::{
    ContentManagement, Document, DownloadTask, DynamicRights, ReadOnlyApplicationRecord,
    SystemUpdate, VulnerabilityManager,
};
use crate::{Service, ServiceContext};

const THREAD: &str = "ns";

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for name in ["ns:am2", "ns:ec", "ns:rid", "ns:rt", "ns:web", "ns:ro"] {
        context.register(name, THREAD, move |context| ServiceGetter {
            name,
            context: context.clone(),
        })?;
    }
    context.register("ns:dev", THREAD, |_| Placeholder("ns:dev"))?;
    let notification = Arc::new(Event::new());
    context.register("ns:su", THREAD, move |context| SystemUpdate {
        context: context.clone(),
        notification: notification.clone(),
    })?;
    context.register("ns:vm", THREAD, |_| VulnerabilityManager)?;
    context.register("pdm:ntfy", THREAD, |_| Placeholder("pdm:ntfy"))
}

/// A service or interface whose commands are all unimplemented, so titles that only open it
/// get a session and the rest goes through the unknown-command path.
pub(crate) struct Placeholder(pub &'static str);

impl Service for Placeholder {
    fn name(&self) -> &str {
        self.0
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

/// IServiceGetterInterface: hands out the ns interfaces.
struct ServiceGetter {
    name: &'static str,
    context: Arc<ServiceContext>,
}

impl Service for ServiceGetter {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let context = self.context.clone();
        let handler = match ctx.command() {
            7988 => self.context.handler(DynamicRights),
            7989 => self
                .context
                .handler(ReadOnlyApplicationControlData { context }),
            7991 => self.context.handler(ReadOnlyApplicationRecord { context }),
            7992 => self.context.handler(Placeholder("IECommerceInterface")),
            7993 => self
                .context
                .handler(Placeholder("IApplicationVersionInterface")),
            7994 => self.context.handler(Placeholder("IFactoryResetInterface")),
            7995 => self.context.handler(Placeholder("IAccountProxyInterface")),
            7996 => self.context.handler(ApplicationManager::new(context)),
            7997 => self.context.handler(DownloadTask),
            7998 => self.context.handler(ContentManagement),
            7999 => self.context.handler(Document { context }),
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests;
