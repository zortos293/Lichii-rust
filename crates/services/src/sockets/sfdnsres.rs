//! sfdnsres with no network: every lookup fails with `EAI_AGAIN`, as the C++ answers for a
//! blocked Nintendo host.

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;

use super::bsd::SUCCESS;
use crate::Service;

const GAI_AGAIN: i32 = 2;
const NETDB_TRY_AGAIN: i32 = 2;

/// The bionic `gai_strerror` text the C++ returns.
fn gai_string(error: i32) -> &'static str {
    match error {
        0 => "Success",
        1 => "Address family for hostname not supported",
        2 => "Temporary failure in name resolution",
        3 => "Invalid value for ai_flags",
        4 => "Non-recoverable failure in name resolution",
        5 => "ai_family not supported",
        6 => "Memory allocation failure",
        7 => "No address associated with hostname",
        8 => "hostname nor servname provided, or not known",
        9 => "servname not supported for ai_socktype",
        10 => "ai_socktype not supported",
        11 => "System error returned in errno",
        12 => "Invalid value for hints",
        13 => "Resolved protocol is unknown",
        14 => "Argument buffer overflow",
        _ => "Unknown error",
    }
}

pub(super) struct Sfdnsres;

impl Service for Sfdnsres {
    fn name(&self) -> &str {
        "sfdnsres"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let errno = SUCCESS as i32;
        match ctx.command() {
            2 => ctx.push([NETDB_TRY_AGAIN, errno, 0]),
            10 => ctx.push([0, NETDB_TRY_AGAIN, errno]),
            6 => ctx.push([errno, GAI_AGAIN, 0]),
            12 => ctx.push([0, GAI_AGAIN, NETDB_TRY_AGAIN, errno]),
            5 => {
                let error: i32 = ctx.pop();
                let mut text = gai_string(error).as_bytes().to_vec();
                text.push(0);
                if let Err(code) = ctx.write_buffer(0, &text) {
                    return Some(code);
                }
            }
            14 => ctx.push(errno),
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

    #[test]
    fn lookups_fail_and_errors_have_text() {
        let guest = Guest::new(testing::image(), Settings::default(), |context| {
            super::super::register(context).unwrap()
        });
        let dns = guest.service("sfdnsres");
        let host = guest.heap;
        guest.write(host, b"example.com\0");
        let out = guest.heap + 0x100;
        let lookup = |command| {
            let request = Request::new(command)
                .arg(0u8)
                .arg(0u32)
                .arg(0u64)
                .buffer_a(host, 12)
                .buffer_b(out, 0x400);
            guest.call(dns, request).ok()
        };
        assert_eq!(lookup(2).at::<[i32; 3]>(0), [NETDB_TRY_AGAIN, 0, 0]);
        assert_eq!(lookup(10).at::<[i32; 3]>(0), [0, NETDB_TRY_AGAIN, 0]);
        assert_eq!(lookup(6).at::<[i32; 3]>(0), [0, GAI_AGAIN, 0]);
        assert_eq!(
            lookup(12).at::<[i32; 4]>(0),
            [0, GAI_AGAIN, NETDB_TRY_AGAIN, 0]
        );
        assert_eq!(guest.call(dns, Request::new(14)).ok().at::<i32>(0), 0);

        for (error, text) in [
            (GAI_AGAIN, &b"Temporary failure in name resolution\0"[..]),
            (99, b"Unknown error\0"),
        ] {
            guest.write(out, &[0xaa; 0x40]);
            let request = Request::new(5).arg(error).buffer_b(out, 0x40);
            guest.call(dns, request).ok();
            assert_eq!(guest.read(out, text.len()), text);
        }
    }
}
