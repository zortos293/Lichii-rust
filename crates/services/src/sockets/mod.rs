//! bsd, sfdnsres and nsd with no host network.

use std::sync::{Arc, Mutex};

use lichii_common::HleResult;
use lichii_kernel::Event;

use crate::ServiceContext;

mod bsd;
mod nsd;
mod sfdnsres;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for (name, user) in [("bsd:s", false), ("bsd:u", true), ("bsd:a", true)] {
        let descriptors = Arc::new(Mutex::new(bsd::Descriptors::new()));
        let stub_event = Arc::new(Event::new());
        context.register(name, "bsdsocket", move |_| bsd::Bsd {
            name,
            user,
            descriptors: descriptors.clone(),
            stub_event: stub_event.clone(),
        })?;
    }
    for name in ["nsd:u", "nsd:a"] {
        context.register(name, "bsdsocket", move |_| nsd::Nsd { name })?;
    }
    context.register("sfdnsres", "bsdsocket", |_| sfdnsres::Sfdnsres)
}
