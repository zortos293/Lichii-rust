//! Finding the shared fonts on the host and laying them out the way `pl:u` serves them. Fonts
//! come from the user's NAND system archives (with their keys) or from font files in
//! `<data>/fonts`, where `<data>` is the parent of the NAND directory. When neither has any,
//! `pl:u` serves the open-source stand-ins the reference synthesizes (`fallback/README.md`).

use std::path::Path;

use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;
use lichii_crypto::keys::KeySet;
use lichii_fs::host::HostFile;
use lichii_fs::romfs::open_romfs;
use lichii_fs::{DirectoryExt, FileExt, Nca, NcaContentType};

/// The shared font types (`SharedFontType`), in order: the system archive holding each font and
/// its file in the archive's RomFS. The seventh is the second extension font.
pub(super) const FONTS: [(u64, &str); FONT_COUNT] = [
    (0x0100_0000_0000_0811, "nintendo_udsg-r_std_003.bfttf"),
    (0x0100_0000_0000_0814, "nintendo_udsg-r_org_zh-cn_003.bfttf"),
    (0x0100_0000_0000_0814, "nintendo_udsg-r_ext_zh-cn_003.bfttf"),
    (0x0100_0000_0000_0813, "nintendo_udjxh-db_zh-tw_003.bfttf"),
    (0x0100_0000_0000_0812, "nintendo_udsg-r_ko_003.bfttf"),
    (0x0100_0000_0000_0810, "nintendo_ext_003.bfttf"),
    (0x0100_0000_0000_0810, "nintendo_ext2_003.bfttf"),
];
pub(super) const FONT_COUNT: usize = 7;

/// First word of a BFTTF file (big endian) and of the decrypted header in shared memory.
const BFTTF_MAGIC: u32 = 0x36f8_1a1e;
const DECRYPTED_MAGIC: u32 = 0x7f9a_0218;
/// The key BFTTF files are XORed with: `BFTTF_MAGIC ^ DECRYPTED_MAGIC`.
pub(super) const FONT_KEY: u32 = BFTTF_MAGIC ^ DECRYPTED_MAGIC;

/// Each font type's stand-in: Open Sans, Source Sans Pro and yuzu's button icon font. They draw
/// Latin text and the button icons, so titles that wait for their fonts keep going.
pub(super) const STAND_INS: [&[u8]; FONT_COUNT] = [
    include_bytes!("fallback/standard.ttf"),
    include_bytes!("fallback/chinese_simplified_korean.ttf"),
    include_bytes!("fallback/extended_chinese_simplified.ttf"),
    include_bytes!("fallback/chinese_traditional.ttf"),
    include_bytes!("fallback/chinese_simplified_korean.ttf"),
    include_bytes!("fallback/nintendo_extended.ttf"),
    include_bytes!("fallback/nintendo_extended.ttf"),
];

/// Plain TrueType/OpenType data of each font type, `None` when the font was not found.
pub(super) type FontData = [Option<Vec<u8>>; FONT_COUNT];

/// Looks for every font: first in the NAND system archives, then in the fonts directory. Without
/// any font in either, every type gets its stand-in.
pub(super) fn find(settings: &Settings) -> FontData {
    let nand = settings
        .paths
        .nand_dir
        .clone()
        .unwrap_or_else(|| UserPaths::discover().nand_dir());
    let keys = settings
        .paths
        .keys_dir
        .clone()
        .unwrap_or_else(|| UserPaths::discover().keys_dir());
    let registered = nand.join("system/Contents/registered");
    let fonts_dir = nand.parent().unwrap_or(&nand).join("fonts");
    let mut fonts: FontData = Default::default();
    from_system_archives(&registered, &keys, &mut fonts);
    from_font_files(&fonts_dir, &mut fonts);
    if fonts.iter().all(Option::is_none) {
        tracing::warn!(
            searched = ?[registered, fonts_dir],
            "no shared fonts found: pl:u serves open-source stand-ins"
        );
        return STAND_INS.map(|font| Some(font.to_vec()));
    }
    fonts
}

fn from_system_archives(registered: &Path, keys_dir: &Path, fonts: &mut FontData) {
    let Ok(entries) = std::fs::read_dir(registered) else {
        return;
    };
    let keys = match KeySet::load_dir(keys_dir) {
        Ok((keys, _)) if keys.header_key().is_ok() => keys,
        _ => {
            tracing::warn!(dir = %keys_dir.display(), "no header key: skipping the NAND font archives");
            return;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let path = if path.is_dir() { path.join("00") } else { path };
        let Some(nca) = HostFile::open(&path)
            .ok()
            .and_then(|file| Nca::open(file, &keys).ok())
        else {
            continue;
        };
        let header = nca.header();
        if header.content_type != NcaContentType::Data
            || !FONTS.iter().any(|&(id, _)| id == header.program_id)
        {
            continue;
        }
        let Some(romfs) = nca
            .romfs()
            .ok()
            .flatten()
            .and_then(|file| open_romfs(&file).ok())
        else {
            continue;
        };
        for (font, &(id, name)) in fonts.iter_mut().zip(&FONTS) {
            if font.is_some() || id != header.program_id {
                continue;
            }
            let data = romfs
                .open_file(name)
                .ok()
                .flatten()
                .and_then(|f| f.read_all().ok());
            *font = data.and_then(|data| decrypt_bfttf(&data));
        }
    }
}

/// Fonts the system archives did not provide, from `<name>.bfttf` (as stored in the archive) or
/// a plain `<stem>.ttf` in `dir`.
fn from_font_files(dir: &Path, fonts: &mut FontData) {
    for (font, (_, name)) in fonts.iter_mut().zip(FONTS) {
        if font.is_some() {
            continue;
        }
        if let Ok(data) = std::fs::read(dir.join(name)) {
            *font = decrypt_bfttf(&data);
        }
        if font.is_none() {
            let ttf = Path::new(name).with_extension("ttf");
            *font = std::fs::read(dir.join(ttf))
                .ok()
                .filter(|data| !data.is_empty());
        }
    }
}

/// The plain font inside a BFTTF file: big-endian words XORed with a key derived from the magic.
pub(super) fn decrypt_bfttf(data: &[u8]) -> Option<Vec<u8>> {
    let word = |i: usize| u32::from_be_bytes(data[4 * i..4 * i + 4].try_into().unwrap());
    if data.len() < 8 || word(0) != BFTTF_MAGIC {
        tracing::warn!("not a BFTTF font file");
        return None;
    }
    let size = ((word(1) ^ FONT_KEY) as usize).min(data.len() - 8);
    let mut plain: Vec<u8> = (2..data.len() / 4)
        .flat_map(|i| (word(i) ^ FONT_KEY).to_be_bytes())
        .collect();
    plain.truncate(size);
    Some(plain)
}

/// A font as `pl:u` shared memory holds it: the decrypted magic, the size XORed with the key
/// (both little endian), then the plain font.
pub(super) fn shared_memory_header(size: u32) -> [u8; 8] {
    let mut header = [0u8; 8];
    header[..4].copy_from_slice(&DECRYPTED_MAGIC.swap_bytes().to_le_bytes());
    header[4..].copy_from_slice(&(size ^ FONT_KEY).to_le_bytes());
    header
}
