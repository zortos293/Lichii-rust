//! `mm:u`: multimedia clock requests. Titles ask for a clock range per module (NVDEC, NVJPG, ...)
//! before decoding. The host clocks itself, so the service only remembers each request's minimum
//! and answers it back, as the reference does.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let requests = Arc::new(Mutex::new(Requests {
        list: Vec::new(),
        next_id: 1,
    }));
    context.register("mm:u", "services", move |_| Mm {
        requests: requests.clone(),
    })
}

struct Request {
    module: u32,
    id: u32,
    min: u32,
}

struct Requests {
    list: Vec<Request>,
    next_id: u32,
}

struct Mm {
    /// Shared by every session, like the reference's single service object.
    requests: Arc<Mutex<Requests>>,
}

impl Service for Mm {
    fn name(&self) -> &str {
        "mm:u"
    }

    /// Commands 0-3 (Initialize, Finalize, SetAndWait, Get) find a request by module; 4-7 are the
    /// same commands keyed by the id Initialize (4) returns.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        if command > 7 {
            return None;
        }
        let key: u32 = ctx.pop();
        let by_id = command >= 4;
        let mut requests = self.requests.lock();
        let found = requests
            .list
            .iter()
            .position(|request| if by_id { request.id } else { request.module } == key);
        match command % 4 {
            0 => {
                let id = requests.next_id;
                requests.next_id += 1;
                requests.list.push(Request {
                    module: key,
                    id,
                    min: 0,
                });
                if by_id {
                    ctx.push(id);
                }
            }
            1 => {
                if let Some(index) = found {
                    requests.list.remove(index);
                }
            }
            2 => {
                let min: u32 = ctx.pop();
                if let Some(index) = found {
                    requests.list[index].min = min;
                }
            }
            _ => ctx.push(found.map_or(0, |index| requests.list[index].min)),
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;

    use crate::testing::{Request, SWITCH1_TITLE, booted};

    const NVDEC: u32 = 5;

    #[test]
    fn clock_requests_remember_their_minimum() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let mm = client.connect("mm:u").unwrap();
        let call = |request: Request| {
            let reply = client.send(mm, &request).unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            reply
        };
        let id = call(Request::new(4).arg(NVDEC).arg(0u32).arg(1u32)).pop::<u32>();
        assert_eq!(id, 1);
        call(Request::new(6).arg(id).arg(600_000_000u32).arg(-1i32));
        assert_eq!(call(Request::new(7).arg(id)).pop::<u32>(), 600_000_000);
        call(Request::new(5).arg(id));
        assert_eq!(
            call(Request::new(7).arg(id)).pop::<u32>(),
            0,
            "a finalized request is gone"
        );

        call(Request::new(0).arg(NVDEC).arg(0u32).arg(1u32));
        call(Request::new(2).arg(NVDEC).arg(384_000_000u32).arg(-1i32));
        assert_eq!(call(Request::new(3).arg(NVDEC)).pop::<u32>(), 384_000_000);
        call(Request::new(1).arg(NVDEC));
        assert_eq!(call(Request::new(3).arg(NVDEC)).pop::<u32>(), 0);
    }
}
