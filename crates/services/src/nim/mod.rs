//! nim, nim:eca/ecas/shp and ntc. Shop requests answer at once with an empty JSON object and the
//! network clock task finishes without a connection, as the C++ stubs do.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::ns::Placeholder;
use crate::{Service, ServiceContext};

const SHOP_RESPONSE: &[u8] = b"{}";

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for name in ["nim", "nim:ecas", "nim:shp"] {
        context.register(name, "nim", move |_| Placeholder(name))?;
    }
    context.register("nim:eca", "nim", |context| Eca {
        context: context.clone(),
    })?;
    context.register("ntc", "nim", |context| Ntc {
        context: context.clone(),
    })?;
    Ok(())
}

struct Eca {
    context: Arc<ServiceContext>,
}

impl Service for Eca {
    fn name(&self) -> &str {
        "nim:eca"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 5 => {
                let server = ShopServiceAccessServer {
                    context: self.context.clone(),
                };
                ctx.push_interface(self.context.handler(server));
            }
            4 => ctx.push(0u8),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// IShopServiceAccessServer.
struct ShopServiceAccessServer {
    context: Arc<ServiceContext>,
}

impl Service for ShopServiceAccessServer {
    fn name(&self) -> &str {
        "IShopServiceAccessServer"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let accessor = ShopServiceAccessor {
            context: self.context.clone(),
        };
        ctx.push_interface(self.context.handler(accessor));
        Some(ResultCode::SUCCESS)
    }
}

/// IShopServiceAccessor.
struct ShopServiceAccessor {
    context: Arc<ServiceContext>,
}

impl Service for ShopServiceAccessor {
    fn name(&self) -> &str {
        "IShopServiceAccessor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let completion = Arc::new(Event::new());
        ctx.push_copy_object(KObject::ReadableEvent(completion.clone()));
        let async_interface = ShopServiceAsync {
            completion,
            data: Mutex::new(Vec::new()),
        };
        ctx.push_interface(self.context.handler(async_interface));
        Some(ResultCode::SUCCESS)
    }
}

/// IShopServiceAsync: a request completes as soon as it is made, with `{}` as the body.
struct ShopServiceAsync {
    completion: Arc<Event>,
    data: Mutex<Vec<u8>>,
}

impl Service for ShopServiceAsync {
    fn name(&self) -> &str {
        "IShopServiceAsync"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 5 => {}
            1 => {
                let size = self.data.lock().unwrap().len() as u64;
                ctx.push(size);
            }
            2 => {
                let offset: u64 = ctx.pop();
                let data = self.data.lock().unwrap();
                let rest = data.get(offset as usize..).unwrap_or_default();
                let read = if rest.is_empty() {
                    0
                } else {
                    match ctx.write_buffer(0, rest) {
                        Ok(read) => read,
                        Err(code) => return Some(code),
                    }
                };
                ctx.push(read as u64);
            }
            3 => ctx.push(0u32),
            4 => {
                *self.data.lock().unwrap() = SHOP_RESPONSE.to_vec();
                self.completion.signal();
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

struct Ntc {
    context: Arc<ServiceContext>,
}

impl Service for Ntc {
    fn name(&self) -> &str {
        "ntc"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {
                let service = EnsureNetworkClockAvailability {
                    finished: Arc::new(Event::new()),
                };
                ctx.push_interface(self.context.handler(service));
            }
            100 | 101 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// IEnsureNetworkClockAvailabilityService: the task finishes as soon as it starts.
struct EnsureNetworkClockAvailability {
    finished: Arc<Event>,
}

impl Service for EnsureNetworkClockAvailability {
    fn name(&self) -> &str {
        "IEnsureNetworkClockAvailabilityService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => self.finished.signal(),
            1 => ctx.push_copy_object(KObject::ReadableEvent(self.finished.clone())),
            2 | 3 => {}
            4 => ctx.push(0u32),
            5 => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map_or(0, |time| time.as_secs() as i64);
                ctx.push(now);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, ReplyExt, Request};

    fn guest() -> Guest {
        Guest::new(testing::image(), Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    #[test]
    fn shop_requests_return_an_empty_object() {
        let guest = guest();
        for name in ["nim", "nim:ecas", "nim:shp"] {
            guest.service(name);
        }
        let eca = guest.service("nim:eca");
        assert_eq!(guest.call(eca, Request::new(4)).ok().at::<u8>(0), 0);
        guest.call(eca, Request::new(5)).ok().interface();
        let server = guest.call(eca, Request::new(0)).ok().interface();
        let accessor = guest.call(server, Request::new(0)).ok().interface();
        let created = guest.call(accessor, Request::new(0)).ok();
        let (completion, shop) = (created.copy_handles[0], created.interface());
        assert!(!guest.is_signaled(completion));

        guest.call(shop, Request::new(5)).ok();
        assert_eq!(guest.call(shop, Request::new(1)).ok().at::<u64>(0), 0);
        guest.call(shop, Request::new(4)).ok();
        assert!(guest.is_signaled(completion));
        assert_eq!(guest.call(shop, Request::new(3)).ok().at::<u32>(0), 0);
        assert_eq!(guest.call(shop, Request::new(1)).ok().at::<u64>(0), 2);

        let out = guest.heap;
        guest.write(out, &[0xaa; 8]);
        let read = guest
            .call(shop, Request::new(2).arg(0u64).buffer_b(out, 8))
            .ok();
        assert_eq!(read.at::<u64>(0), 2);
        assert_eq!(guest.read(out, 3), b"{}\xaa");
        let past = guest
            .call(shop, Request::new(2).arg(5u64).buffer_b(out, 8))
            .ok();
        assert_eq!(past.at::<u64>(0), 0);
        guest.call(shop, Request::new(0)).ok();
    }

    #[test]
    fn the_network_clock_task_finishes_at_once() {
        let guest = guest();
        let ntc = guest.service("ntc");
        guest.call(ntc, Request::new(100)).ok();
        guest.call(ntc, Request::new(101)).ok();
        let service = guest.call(ntc, Request::new(0)).ok().interface();
        let finished = guest.call(service, Request::new(1)).ok().handle();
        assert!(!guest.is_signaled(finished));
        guest.call(service, Request::new(0)).ok();
        assert!(guest.is_signaled(finished));
        guest.call(service, Request::new(2)).ok();
        assert_eq!(guest.call(service, Request::new(4)).ok().at::<u32>(0), 0);
        let time = guest.call(service, Request::new(5)).ok().at::<i64>(0);
        assert!(time > 1_700_000_000);
    }
}
