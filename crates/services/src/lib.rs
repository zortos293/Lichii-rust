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

mod acc;
mod am;
mod aoc;
mod apm;
mod arp;
mod audio;
mod bcat;
mod caps;
mod chat;
mod context;
mod dispatch;
mod erpt;
mod eupld;
mod fatal;
mod friend;
mod fs;
mod gsv;
mod hid;
mod ldn;
mod lm;
mod mm;
mod ncm;
mod ngc;
mod nifm;
mod nim;
mod notif;
mod npns;
mod ns;
mod nvdrv;
mod pctl;
mod pl_u;
mod prepo;
mod ptm;
mod ro;
mod set;
mod sockets;
mod spl;
mod ssl;
#[cfg(test)]
mod testing;
mod time;
mod vi;

use std::sync::Arc;

use lichii_common::HleResult;

pub use am::applets;
pub use context::{ServiceContext, ServiceParts};
pub use dispatch::Service;

/// Registers every HLE service with `sm:`. Call it after the application process exists and
/// before `Kernel::run`.
pub fn register_all(context: &Arc<ServiceContext>) -> HleResult<()> {
    fs::register(context)?;
    am::register(context)?;
    audio::register(context)?;
    apm::register(context)?;
    pctl::register(context)?;
    prepo::register(context)?;
    lm::register(context)?;
    erpt::register(context)?;
    eupld::register(context)?;
    caps::register(context)?;
    set::register(context)?;
    acc::register(context)?;
    time::register(context)?;
    chat::register(context)?;
    hid::register(context)?;
    pl_u::register(context)?;
    aoc::register(context)?;
    arp::register(context)?;
    bcat::register(context)?;
    friend::register(context)?;
    ncm::register(context)?;
    nifm::register(context)?;
    nim::register(context)?;
    ns::register(context)?;
    sockets::register(context)?;
    ssl::register(context)?;
    spl::register(context)?;
    ptm::register(context)?;
    fatal::register(context)?;
    mm::register(context)?;
    ngc::register(context)?;
    notif::register(context)?;
    npns::register(context)?;
    ldn::register(context)?;
    ro::register(context)?;
    let nvdrv = nvdrv::register(context)?;
    vi::register(context, nvdrv)?;
    Ok(())
}
