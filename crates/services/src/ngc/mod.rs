//! `ngct:u` and `ngc:u`: the profanity filter behind software keyboards and user-made names.
//! There is no word list, so no text matches and masking returns the text unchanged, as the
//! reference answers.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

const CONTENT_VERSION: u32 = 1;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("ngct:u", "services", |_| Ngct)?;
    context.register("ngc:u", "services", |_| Ngc)
}

/// Copies input buffer 0 to output buffer 0.
fn echo(ctx: &HleRequestContext) -> Result<(), ResultCode> {
    ctx.write_buffer(0, &ctx.read_buffer(0)?)?;
    Ok(())
}

struct Ngct;

impl Service for Ngct {
    fn name(&self) -> &str {
        "ngct:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(0u8),
            1 => {
                if let Err(code) = echo(ctx) {
                    return Some(code);
                }
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

struct Ngc;

impl Service for Ngc {
    fn name(&self) -> &str {
        "ngc:u"
    }

    /// Check (1, 4) and Mask (2, 5) take flags and a 0x20-byte option they need not read.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(CONTENT_VERSION),
            1 | 4 => ctx.push(0u32),
            2 | 5 => {
                if let Err(code) = echo(ctx) {
                    return Some(code);
                }
                ctx.push(0u32);
            }
            3 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;

    use crate::testing::{Request, SWITCH1_TITLE, booted};

    #[test]
    fn nothing_matches_and_masking_keeps_the_text() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let text = client.alloc(b"hello\0");
        let out = client.alloc(&[0xff; 6]);

        let ngct = client.connect("ngct:u").unwrap();
        let mut reply = client
            .send(ngct, &Request::new(0).buffer_a(text, 6))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u8>(), 0, "no match");
        let reply = client
            .send(ngct, &Request::new(1).buffer_a(text, 6).buffer_b(out, 6))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(client.read(out, 6), b"hello\0");

        let ngc = client.connect("ngc:u").unwrap();
        assert_eq!(client.send(ngc, &Request::new(0)).unwrap().pop::<u32>(), 1);
        let options = [0u32; 9];
        for command in [1, 4] {
            let mut reply = client
                .send(ngc, &Request::new(command).arg(options).buffer_a(text, 6))
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            assert_eq!(reply.pop::<u32>(), 0, "check {command}");
        }
        for command in [2, 5] {
            let out = client.alloc(&[0xff; 6]);
            let mut reply = client
                .send(
                    ngc,
                    &Request::new(command)
                        .arg(options)
                        .buffer_a(text, 6)
                        .buffer_b(out, 6),
                )
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            assert_eq!(reply.pop::<u32>(), 0);
            assert_eq!(client.read(out, 6), b"hello\0", "mask {command}");
        }
        assert_eq!(
            client.send(ngc, &Request::new(3)).unwrap().result,
            ResultCode::SUCCESS
        );
    }
}
