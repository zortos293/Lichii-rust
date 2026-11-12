//! ssl and ssl:s with no TLS backend: contexts are created, connections fail with the C++
//! `ResultInternalError`, and the certificate store is empty because there is no system
//! certificate archive.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

/// `ResultInternalError` (SSLSrv 999): the C++ none backend's answer to every connection.
const INTERNAL_ERROR: ResultCode = ResultCode::new(123, 999);
const RESULT_UNKNOWN: ResultCode = ResultCode(u32::MAX);
/// `BuiltInCertificateInfo` that ends the list: id `All`, status `Invalid`, no data.
const CERTIFICATE_TERMINATOR: [u8; 0x18] = {
    let mut info = [0u8; 0x18];
    let mut i = 0;
    while i < 8 {
        info[i] = 0xff;
        i += 1;
    }
    info
};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for (name, system) in [("ssl", false), ("ssl:s", true)] {
        context.register(name, "ssl", move |context| SslService {
            name,
            system,
            context: context.clone(),
        })?;
    }
    Ok(())
}

/// ISslService (`ssl`) and ISslServiceForSystem (`ssl:s`).
struct SslService {
    name: &'static str,
    system: bool,
    context: Arc<ServiceContext>,
}

impl Service for SslService {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match (ctx.command(), self.system) {
            (0, _) | (100, true) => {
                let context = SslContext {
                    system: ctx.command() == 100,
                };
                ctx.push_interface(self.context.handler(context));
            }
            (2, _) => {
                if ctx.writable_buffer(0).map_or(0, |b| b.size)
                    < CERTIFICATE_TERMINATOR.len() as u64
                {
                    return Some(RESULT_UNKNOWN);
                }
                if let Err(code) = ctx.write_buffer(0, &CERTIFICATE_TERMINATOR) {
                    return Some(code);
                }
                ctx.push(0u32);
            }
            (3, _) => ctx.push(CERTIFICATE_TERMINATOR.len() as u32),
            (5, _) | (4 | 7 | 8 | 9 | 101 | 103, true) => {}
            (1 | 6, true) => ctx.push(0u32),
            (102, true) => ctx.push(0u64),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// ISslContext and ISslContextForSystem.
struct SslContext {
    system: bool,
}

impl Service for SslContext {
    fn name(&self) -> &str {
        if self.system {
            "ISslContextForSystem"
        } else {
            "ISslContext"
        }
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match (ctx.command(), self.system) {
            (0 | 1, _) => {}
            (2, _) | (100, true) => return Some(INTERNAL_ERROR),
            (3, _) => ctx.push(0u32),
            (4 | 5 | 8, _) => ctx.push(0u64),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, Request};

    fn guest() -> Guest {
        Guest::new(testing::image(), Settings::default(), |context| {
            register(context).unwrap()
        })
    }

    fn create_context(guest: &Guest, service: u32, command: u32) -> u32 {
        let request = Request::new(command).arg(1u32).arg(0u32).arg(0u64);
        guest.call(service, request).ok().interface()
    }

    #[test]
    fn contexts_exist_but_connections_fail() {
        let guest = guest();
        let ssl = guest.service("ssl");
        guest.call(ssl, Request::new(5).arg(3u32)).ok();
        let context = create_context(&guest, ssl, 0);
        guest
            .call(context, Request::new(0).arg(1u32).arg(1i32))
            .ok();
        guest.call(context, Request::new(1).arg(1u32)).ok();
        assert_eq!(guest.call(context, Request::new(2)).result, INTERNAL_ERROR);
        assert_eq!(guest.call(context, Request::new(3)).ok().at::<u32>(0), 0);
        let pki = guest.heap;
        guest.write(pki, &[1; 0x10]);
        for command in [4, 5] {
            let request = Request::new(command).arg(0u32).input(pki, 0x10);
            assert_eq!(guest.call(context, request).ok().at::<u64>(0), 0);
        }
        assert_eq!(guest.call(context, Request::new(8)).ok().at::<u64>(0), 0);
        let unknown = guest.call(context, Request::new(100)).result;
        assert_ne!(unknown, INTERNAL_ERROR);
        assert_eq!(guest.call(ssl, Request::new(100)).result, unknown);

        let ssl_s = guest.service("ssl:s");
        let system = create_context(&guest, ssl_s, 100);
        assert_eq!(guest.call(system, Request::new(100)).result, INTERNAL_ERROR);
        create_context(&guest, ssl_s, 0);
        assert_eq!(guest.call(ssl_s, Request::new(1)).ok().at::<u32>(0), 0);
        assert_eq!(guest.call(ssl_s, Request::new(102)).ok().at::<u64>(0), 0);
    }

    #[test]
    fn the_certificate_store_is_empty() {
        let guest = guest();
        for name in ["ssl", "ssl:s"] {
            let service = guest.service(name);
            let ids = guest.heap;
            guest.write(ids, &(-1i32).to_le_bytes());
            let size = guest.call(service, Request::new(3).input(ids, 4)).ok();
            assert_eq!(size.at::<u32>(0), 0x18);

            let out = guest.heap + 0x100;
            guest.write(out, &[0xaa; 0x20]);
            let small = Request::new(2).input(ids, 4).output(out, 0x10);
            assert_eq!(guest.call(service, small).result, RESULT_UNKNOWN);
            let certificates = Request::new(2).input(ids, 4).output(out, 0x20);
            assert_eq!(guest.call(service, certificates).ok().at::<u32>(0), 0);
            let info = guest.read(out, 0x20);
            assert_eq!(info[..0x18], CERTIFICATE_TERMINATOR);
            assert_eq!(info[0x18..], [0xaa; 8]);
        }
    }
}
