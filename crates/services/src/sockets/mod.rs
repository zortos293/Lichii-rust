//! bsd, sfdnsres and nsd with no host network.

use std::sync::{Arc, Mutex};

use lichii_common::HleResult;
use lichii_kernel::Event;

use crate::ServiceContext;
use crate::dispatch::Commandless;

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
    context.register("bsd:nu", "bsdsocket", |_| Commandless("bsd:nu"))?;
    context.register("sfdnsres", "bsdsocket", |_| sfdnsres::Sfdnsres)
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;

    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    #[test]
    fn bsd_nu_is_stubbed_only_for_switch2_titles() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let client = booted(title, Settings::default(), super::register);
            let bsd = client.connect("bsd:nu").unwrap();
            assert_eq!(client.send(bsd, &Request::new(0)).unwrap().result, expected);
        }
    }
}
