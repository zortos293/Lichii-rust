//! HLE Horizon system services.
//!
//! Every service is a [`Service`] in its own module (`src/am`, `src/hid`, ...) with a
//! `register(&Arc<ServiceContext>)` function that [`register_all`] calls, so crews add services
//! without touching each other's files. [`ServiceContext::register`] wraps a service for the
//! kernel: commands a service does not implement go through one logged path that can answer an
//! empty success (`debug.auto_stub`), and `debug.ipc_trace` logs every command. The kernel side
//! of IPC is described in `crates/kernel/IPC.md`.
//!
//! A service module looks like this, and adds one `pl_u::register(context)?;` line to
//! [`register_all`]:
//!
//! ```ignore
//! pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
//!     context.register("pl:u", "services", |context| PlU { context: context.clone() })
//! }
//!
//! struct PlU {
//!     context: Arc<ServiceContext>,
//! }
//!
//! impl Service for PlU {
//!     fn name(&self) -> &str {
//!         "pl:u"
//!     }
//!
//!     fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
//!         Some(match ctx.command() {
//!             1 => { /* GetLoadState */ ctx.push(1u32); ResultCode::SUCCESS }
//!             5 => { ctx.push_interface(self.context.handler(SharedFont::new())); ResultCode::SUCCESS }
//!             _ => return None,
//!         })
//!     }
//! }
//! ```

mod am;
mod chat;
mod context;
mod dispatch;
mod fs;
mod hid;
mod pl_u;
#[cfg(test)]
mod testing;

use std::sync::Arc;

use lichii_common::HleResult;

pub use context::{ServiceContext, ServiceParts};
pub use dispatch::Service;

/// Registers every HLE service with `sm:`. Call it after the application process exists and
/// before `Kernel::run`.
pub fn register_all(context: &Arc<ServiceContext>) -> HleResult<()> {
    fs::register(context)?;
    am::register(context)?;
    chat::register(context)?;
    hid::register(context)?;
    pl_u::register(context)?;
    Ok(())
}
