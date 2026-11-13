//! `set`, `set:sys`, `set:cal` and `set:fd`: language, region and system settings. Values live in
//! memory for the session of the emulator and start from [`lichii_common::settings::Settings`].

mod key_code_map;
mod system;

use std::sync::Arc;

use lichii_common::settings::Region;
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const MODULE: u32 = 105;
const INVALID_LANGUAGE: ResultCode = ResultCode::new(MODULE, 625);
const NULL_POINTER: ResultCode = ResultCode::new(MODULE, 1261);

const KEY_CODE_MAP_SIZE: usize = 0x1000;
const PRE_4_0_0_MAX_LANGUAGES: usize = 0xF;
const POST_4_0_0_MAX_LANGUAGES: usize = 0x40;
const STUB_ZEROES: [u8; 128] = [0; 128];

const fn language_code(code: &[u8]) -> u64 {
    let mut value = 0;
    let mut i = code.len();
    while i > 0 {
        i -= 1;
        value = value << 8 | code[i] as u64;
    }
    value
}

/// `nn::settings::LanguageCode` of each `nn::settings::Language`, with the keyboard layout the
/// settings sysmodule picks for it.
const LANGUAGES: [(u64, KeyboardLayout); 20] = [
    (language_code(b"ja"), KeyboardLayout::Japanese),
    (language_code(b"en-US"), KeyboardLayout::EnglishUs),
    (language_code(b"fr"), KeyboardLayout::French),
    (language_code(b"de"), KeyboardLayout::German),
    (language_code(b"it"), KeyboardLayout::Italian),
    (language_code(b"es"), KeyboardLayout::Spanish),
    (language_code(b"zh-CN"), KeyboardLayout::ChineseSimplified),
    (language_code(b"ko"), KeyboardLayout::Korean),
    (language_code(b"nl"), KeyboardLayout::EnglishUsInternational),
    (language_code(b"pt"), KeyboardLayout::Portuguese),
    (language_code(b"ru"), KeyboardLayout::Russian),
    (language_code(b"zh-TW"), KeyboardLayout::ChineseTraditional),
    (language_code(b"en-GB"), KeyboardLayout::EnglishUk),
    (language_code(b"fr-CA"), KeyboardLayout::FrenchCa),
    (language_code(b"es-419"), KeyboardLayout::SpanishLatin),
    (language_code(b"zh-Hans"), KeyboardLayout::ChineseSimplified),
    (
        language_code(b"zh-Hant"),
        KeyboardLayout::ChineseTraditional,
    ),
    (language_code(b"pt-BR"), KeyboardLayout::Portuguese),
    (language_code(b"pl"), KeyboardLayout::EnglishUsInternational),
    (language_code(b"th"), KeyboardLayout::EnglishUsInternational),
];

const LANGUAGE_CODES: [u8; 20 * 8] = {
    let mut bytes = [0; 20 * 8];
    let mut i = 0;
    while i < bytes.len() {
        bytes[i] = (LANGUAGES[i / 8].0 >> (i % 8 * 8)) as u8;
        i += 1;
    }
    bytes
};

/// `nn::settings::KeyboardLayout`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
enum KeyboardLayout {
    Japanese = 0,
    EnglishUs = 1,
    EnglishUsInternational = 2,
    EnglishUk = 3,
    French = 4,
    FrenchCa = 5,
    Spanish = 6,
    SpanishLatin = 7,
    German = 8,
    Italian = 9,
    Portuguese = 10,
    Russian = 11,
    Korean = 12,
    ChineseSimplified = 13,
    ChineseTraditional = 14,
}

/// The `nn::kpr::KeyCodeMap` for the language's layout, as `GetKeyCodeMap` returns it.
fn key_code_map(language_code: u64) -> &'static [u8] {
    let Some(&(_, layout)) = LANGUAGES.iter().find(|(code, _)| *code == language_code) else {
        return key_code_map::ENGLISH_US_INTERNATIONAL;
    };
    match layout {
        KeyboardLayout::Japanese => key_code_map::JAPANESE,
        KeyboardLayout::EnglishUs | KeyboardLayout::EnglishUsInternational => {
            key_code_map::ENGLISH_US_INTERNATIONAL
        }
        KeyboardLayout::EnglishUk => key_code_map::ENGLISH_UK,
        KeyboardLayout::French => key_code_map::FRENCH,
        KeyboardLayout::FrenchCa => key_code_map::FRENCH_CA,
        KeyboardLayout::Spanish => key_code_map::SPANISH,
        KeyboardLayout::SpanishLatin => key_code_map::SPANISH_LATIN,
        KeyboardLayout::German => key_code_map::GERMAN,
        KeyboardLayout::Italian => key_code_map::ITALIAN,
        KeyboardLayout::Portuguese => key_code_map::PORTUGUESE,
        KeyboardLayout::Russian => key_code_map::RUSSIAN,
        KeyboardLayout::Korean => key_code_map::KOREAN,
        KeyboardLayout::ChineseSimplified => key_code_map::CHINESE_SIMPLIFIED,
        KeyboardLayout::ChineseTraditional => key_code_map::CHINESE_TRADITIONAL,
    }
}

/// `nn::settings::system::RegionCode`.
fn region_code(region: Region) -> u32 {
    match region {
        Region::Japan => 0,
        Region::Usa => 1,
        Region::Europe => 2,
        Region::Australia => 3,
        Region::Korea | Region::Taiwan => 4,
        Region::China => 5,
    }
}

/// Settings `set` and `set:sys` share: `set:sys` writes what `set` reads.
struct Shared {
    language_code: u64,
    region_code: u32,
    quest_flag: u8,
    device_nickname: [u8; 0x80],
}

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let system = &context.settings.system;
    let language = system.language as usize;
    let mut device_nickname = [0; 0x80];
    device_nickname[..6].copy_from_slice(b"Lichii");
    let shared = Arc::new(Mutex::new(Shared {
        language_code: LANGUAGES[language].0,
        region_code: region_code(system.region),
        quest_flag: 0,
        device_nickname,
    }));
    let keyboard_layout = LANGUAGES[language].1;
    let state = Arc::new(Mutex::new(system::State::new(keyboard_layout)));

    let set = shared.clone();
    context.register("set", "set", move |_| Settings {
        shared: set.clone(),
    })?;
    context.register("set:sys", "set", move |context| system::SystemSettings {
        region: context.settings.system.region,
        shared: shared.clone(),
        state: state.clone(),
    })?;
    for name in ["set:cal", "set:fd"] {
        context.register(name, "set", move |_| Unimplemented { name })?;
    }
    Ok(())
}

/// `ISettingsServer`.
struct Settings {
    shared: Arc<Mutex<Shared>>,
}

impl Settings {
    fn available_language_codes(ctx: &mut HleRequestContext, max: usize) -> ResultCode {
        let capacity = ctx
            .writable_buffer(0)
            .map_or(0, |buffer| buffer.size as usize / 8);
        let count = LANGUAGES.len().min(max).min(capacity);
        if let Err(result) = ctx.write_buffer(0, &LANGUAGE_CODES[..count * 8]) {
            return result;
        }
        ctx.push(count as i32);
        ResultCode::SUCCESS
    }

    fn key_code_map(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let Some(buffer) = ctx.writable_buffer(0).filter(|buffer| buffer.size != 0) else {
            return NULL_POINTER;
        };
        let mut map = [0; KEY_CODE_MAP_SIZE];
        let source = key_code_map(self.shared.lock().language_code);
        map[..source.len()].copy_from_slice(source);
        let len = map.len().min(buffer.size as usize);
        match ctx.write_buffer(0, &map[..len]) {
            Ok(_) => ResultCode::SUCCESS,
            Err(result) => result,
        }
    }
}

impl Service for Settings {
    fn name(&self) -> &str {
        "set"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(self.shared.lock().language_code),
            1 => return Some(Self::available_language_codes(ctx, PRE_4_0_0_MAX_LANGUAGES)),
            2 => {
                let language = ctx.pop::<u32>() as usize;
                let Some(&(code, _)) = LANGUAGES.get(language) else {
                    return Some(INVALID_LANGUAGE);
                };
                ctx.push(code);
            }
            3 => ctx.push(PRE_4_0_0_MAX_LANGUAGES as i32),
            4 => ctx.push(self.shared.lock().region_code),
            5 => {
                return Some(Self::available_language_codes(
                    ctx,
                    POST_4_0_0_MAX_LANGUAGES,
                ));
            }
            6 => ctx.push(POST_4_0_0_MAX_LANGUAGES as i32),
            7 | 9 | 12 => return Some(self.key_code_map(ctx)),
            8 => ctx.push(self.shared.lock().quest_flag),
            10 | 13..=17 => ctx.push_bytes(&STUB_ZEROES),
            11 => {
                let nickname = self.shared.lock().device_nickname;
                if let Err(result) = ctx.write_buffer(0, &nickname) {
                    return Some(result);
                }
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `set:cal` and `set:fd`: system-only, and no title calls them.
struct Unimplemented {
    name: &'static str,
}

impl Service for Unimplemented {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, _: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

#[cfg(test)]
mod tests;
