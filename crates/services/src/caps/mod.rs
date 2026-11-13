//! `caps:a`, `caps:c`, `caps:u`, `caps:ss`, `caps:sc` and `caps:su`: the album. It is mounted
//! and empty; screenshots are accepted and not stored.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

const MODULE: u32 = 206;
const INVALID_STORAGE: ResultCode = ResultCode::new(MODULE, 13);
const FILE_NOT_FOUND: ResultCode = ResultCode::new(MODULE, 23);

const SD: u8 = 1;
const STUB_ZEROES: [u8; 128] = [0; 128];

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for album in [
        Album::Accessor,
        Album::Control,
        Album::Application,
        Album::ScreenShot,
        Album::ScreenShotControl,
        Album::ScreenShotApplication,
    ] {
        context.register(album.service_name(), "caps", move |_| album)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Album {
    Accessor,
    Control,
    Application,
    ScreenShot,
    ScreenShotControl,
    ScreenShotApplication,
}

impl Album {
    fn service_name(self) -> &'static str {
        match self {
            Self::Accessor => "caps:a",
            Self::Control => "caps:c",
            Self::Application => "caps:u",
            Self::ScreenShot => "caps:ss",
            Self::ScreenShotControl => "caps:sc",
            Self::ScreenShotApplication => "caps:su",
        }
    }
}

impl Service for Album {
    fn name(&self) -> &str {
        self.service_name()
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        use Album::*;
        match (self, ctx.command()) {
            (Accessor, 1 | 101) | (Application, 102 | 142) => ctx.push(0u64),
            (Accessor, 3 | 1002 | 1003) => return Some(FILE_NOT_FOUND),
            (Accessor, 5) => {
                let storage: u8 = ctx.pop();
                let mounted = storage <= SD;
                ctx.push(u8::from(mounted));
                if !mounted {
                    return Some(INVALID_STORAGE);
                }
            }
            (Accessor, 18 | 50011) => ctx.push(0u32),
            (Accessor, 401) => ctx.push(0u8),
            (Control, 33) | (Application | ScreenShotApplication, 32) => {}
            (Application, 103..=141 | 143..=148) | (ScreenShotApplication, 201 | 210) => {
                ctx.push_bytes(&STUB_ZEROES)
            }
            (ScreenShot, 203 | 206) | (ScreenShotApplication, 203 | 205) => {
                tracing::debug!(service = self.service_name(), "caps: screenshot not stored");
                ctx.push([0u64; 4]);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {

    use super::{FILE_NOT_FOUND, INVALID_STORAGE};

    use lichii_common::settings::Settings;
    use lichii_kernel::testing::Request;

    use crate::testing::{ReplyExt, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    #[test]
    fn the_album_is_mounted_and_empty() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let a = client.connect("caps:a").unwrap();
        let list = client.alloc(&vec![0; 0x100]);
        let count = |request: Request| {
            client
                .send(a, &request.buffer_b(list, 0x100))
                .unwrap()
                .ok()
                .get::<u64>(0)
        };
        assert_eq!(count(Request::new(1).arg(1u8)), 0);
        assert_eq!(count(Request::new(101).arg(1u8).arg(0u8)), 0);
        assert_eq!(
            client
                .send(a, &Request::new(5).arg(1u8))
                .unwrap()
                .ok()
                .get::<u8>(0),
            1
        );
        let nand_and_sd = client.send(a, &Request::new(5).arg(0u8)).unwrap().ok();
        assert_eq!(nand_and_sd.get::<u8>(0), 1);
        assert_eq!(
            client.send(a, &Request::new(5).arg(2u8)).unwrap().result,
            INVALID_STORAGE
        );
        assert_eq!(
            client
                .send(a, &Request::new(18))
                .unwrap()
                .ok()
                .get::<u32>(0),
            0
        );
        assert_eq!(
            client
                .send(a, &Request::new(401))
                .unwrap()
                .ok()
                .get::<u8>(0),
            0
        );
        assert_eq!(
            client
                .send(a, &Request::new(50011))
                .unwrap()
                .ok()
                .get::<u32>(0),
            0
        );
        assert_eq!(
            client.send(a, &Request::new(3)).unwrap().result,
            FILE_NOT_FOUND
        );
        let u = client.connect("caps:u").unwrap();
        client
            .send(u, &Request::new(32).arg(1u64).send_pid())
            .unwrap()
            .ok();
        assert_eq!(
            client
                .send(u, &Request::new(142).buffer_b(list, 0x100))
                .unwrap()
                .ok()
                .get::<u64>(0),
            0
        );
        assert_eq!(
            client.send(u, &Request::new(104)).unwrap().ok().data(),
            [0; 128]
        );
        client
            .send(client.connect("caps:c").unwrap(), &Request::new(33))
            .unwrap()
            .ok();
        client.connect("caps:sc").unwrap();
    }

    #[test]
    fn screenshots_are_accepted() {
        let client = booted(SWITCH2_TITLE, Settings::default(), super::register);
        let image = client.alloc(&vec![0; 0x1000]);
        for (service, command) in [
            ("caps:su", 203),
            ("caps:su", 205),
            ("caps:ss", 203),
            ("caps:ss", 206),
        ] {
            let handle = client.connect(service).unwrap();
            let entry = client
                .send(handle, &Request::new(command).buffer_a(image, 0x1000))
                .unwrap()
                .ok();
            assert_eq!(entry.data(), [0; 0x20]);
        }
        let su = client.connect("caps:su").unwrap();
        assert_eq!(
            client
                .send(su, &Request::new(201))
                .unwrap()
                .ok()
                .data()
                .len(),
            128
        );
    }
}
