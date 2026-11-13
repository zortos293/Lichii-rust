//! `nsd:u` and `nsd:a`: Nintendo server names. With no network every name resolves to itself, and
//! the environment is `lp1`, the retail one, as the reference answers.

use lichii_common::ResultCode;
use lichii_kernel::ipc::HleRequestContext;

use crate::Service;

/// `nn::nsd::ResultOverflow`: the resolved name does not fit the 0x100-byte output.
const OVERFLOW: ResultCode = ResultCode::new(141, 6);
const FQDN_SIZE: usize = 0x100;
/// `ServerEnvironmentType::Lp`.
const LP: u32 = 1;

pub(super) struct Nsd {
    pub(super) name: &'static str,
}

impl Nsd {
    /// The name in input buffer 0, resolved to itself, as a NUL-terminated 0x100-byte field.
    fn resolve(ctx: &HleRequestContext) -> Result<[u8; FQDN_SIZE], ResultCode> {
        let input = ctx.read_buffer(0)?;
        let name = &input[..input.iter().position(|&b| b == 0).unwrap_or(input.len())];
        tracing::debug!("nsd resolves {} to itself", String::from_utf8_lossy(name));
        if name.len() >= FQDN_SIZE {
            return Err(OVERFLOW);
        }
        let mut fqdn = [0; FQDN_SIZE];
        fqdn[..name.len()].copy_from_slice(name);
        Ok(fqdn)
    }
}

impl Service for Nsd {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            11 => {
                if let Err(code) = ctx.write_buffer(0, b"lp1\0\0\0\0\0") {
                    return Some(code);
                }
            }
            15 => {}
            20 => {
                let resolved = Self::resolve(ctx);
                let fqdn = resolved.unwrap_or([0; FQDN_SIZE]);
                if let Err(code) = ctx.write_buffer(0, &fqdn) {
                    return Some(code);
                }
                return Some(resolved.err().unwrap_or(ResultCode::SUCCESS));
            }
            21 => {
                let fqdn = match Self::resolve(ctx) {
                    Ok(fqdn) => fqdn,
                    Err(code) => return Some(code),
                };
                if let Err(code) = ctx.write_buffer(0, &fqdn) {
                    return Some(code);
                }
                ctx.push(ResultCode::SUCCESS.0);
            }
            100 => ctx.push(LP),
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
    fn names_resolve_to_themselves_in_the_retail_environment() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::super::register);
        for name in ["nsd:u", "nsd:a"] {
            let nsd = client.connect(name).unwrap();
            let out = client.alloc(&[0xff; 0x100]);
            let reply = client
                .send(nsd, &Request::new(11).buffer_b(out, 8))
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            assert_eq!(client.read(out, 8), b"lp1\0\0\0\0\0");

            let host = client.alloc(b"api.example.nintendo.net\0junk");
            for command in [20, 21] {
                let reply = client
                    .send(
                        nsd,
                        &Request::new(command)
                            .buffer_a(host, 29)
                            .buffer_b(out, 0x100),
                    )
                    .unwrap();
                assert_eq!(reply.result, ResultCode::SUCCESS, "{name} {command}");
                let fqdn = client.read(out, 0x100);
                assert_eq!(&fqdn[..25], b"api.example.nintendo.net\0");
                assert!(fqdn[25..].iter().all(|&b| b == 0));
            }

            let long = client.alloc(&[b'a'; 0x100]);
            let resolve_ex = Request::new(21).buffer_a(long, 0x100).buffer_b(out, 0x100);
            assert_eq!(
                client.send(nsd, &resolve_ex).unwrap().result,
                super::OVERFLOW
            );
            let mut reply = client.send(nsd, &Request::new(100)).unwrap();
            assert_eq!(reply.pop::<u32>(), 1, "Lp");
        }
    }
}
