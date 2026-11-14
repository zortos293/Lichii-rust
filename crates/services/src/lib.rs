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
mod ectx;
mod erpt;
mod eupld;
mod fatal;
mod friend;
mod fs;
mod gsv;
mod hid;
mod hidbus;
mod irs;
mod jit;
mod ldn;
mod lm;
mod mii;
mod mm;
mod ncm;
mod nfc;
mod nfp;
mod ngc;
mod nifm;
mod nim;
mod notif;
mod npns;
mod ns;
mod nvdrv;
mod olsc;
mod pctl;
mod pl_u;
mod prepo;
mod ptm;
mod reference_services;
mod ro;
mod set;
mod sockets;
mod spl;
mod ssl;
mod stpl;
#[cfg(test)]
mod testing;
mod time;
mod vi;

use std::sync::Arc;

use lichii_common::HleResult;

pub use am::applets;
pub use context::{Handover, ServiceContext, ServiceParts, SessionEvents};
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
    irs::register(context)?;
    hidbus::register(context)?;
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
    nfc::register(context)?;
    nfp::register(context)?;
    mii::register(context)?;
    ectx::register(context)?;
    olsc::register(context)?;
    notif::register(context)?;
    npns::register(context)?;
    ldn::register(context)?;
    ro::register(context)?;
    jit::register(context)?;
    let nvdrv = nvdrv::register(context)?;
    vi::register(context, nvdrv)?;
    stpl::register(context)?;
    stub_missing_switch2_services(context)?;
    let missing = reference_services::missing(context);
    if !missing.is_empty() {
        tracing::warn!(
            count = missing.len(),
            services = ?missing,
            "the title may open services the reference registers and Lichii does not"
        );
    }
    Ok(())
}

/// Switch 2 titles also open services nothing here provides (`mnpp:app`, `htc:tenv`), and
/// waiting in `sm:` would hang them for good: they get an [`dispatch::Nx2Stub`] session whose
/// commands are all auto-stubbed, as the reference does.
fn stub_missing_switch2_services(context: &Arc<ServiceContext>) -> HleResult<()> {
    let thread = context.service_thread("services")?;
    let stubs = context.clone();
    context.kernel.service_manager().set_switch2_stub(
        &thread,
        Arc::new(move |name| {
            stubs.handler(dispatch::Nx2Stub(format!("{}{name}", dispatch::NX2_PREFIX)))
        }),
    );
    Ok(())
}
