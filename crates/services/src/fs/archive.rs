//! System archives synthesised when the NAND has no system data, as the C++ does: only the
//! archives it builds itself, from synthetic content.
//!
//! Gaps: the shared fonts (0x810..0x814) are open-source fonts the C++ embeds; they are not
//! synthesised here, so those ids fail until real system data or host font files are wired
//! in. The time zone binary holds every zone of the tz database, which the C++ embeds too.

use std::collections::BTreeMap;

use lichii_fs::romfs::build_romfs;
use lichii_fs::{FileRef, MemDir};

/// Firmware version the HLE services report (the C++ `HLE::ApiVersion`).
const HOS_VERSION: [u8; 3] = [23, 0, 0];
const SDK_REVISION: [u8; 2] = [1, 0];
const PLATFORM_STRING: &str = "NX";
const VERSION_HASH: &str = "c2663accb7490a6ccfe9bc4dfd120ca215490525";
const DISPLAY_VERSION: &str = "23.0.0";
const DISPLAY_TITLE: &str = "NintendoSDK Firmware for NX 23.0.0-4.0";
/// The longest location name a `nn::time::LocationName` holds with its terminator.
const LOCATION_NAME_MAX: usize = 0x23;

/// The system data archive `title_id` as a RomFS image, when it can be synthesised.
pub(super) fn synthesize(title_id: u64) -> Option<FileRef> {
    let (name, root) = match title_id {
        0x0100_0000_0000_0802 => ("MiiModel", Some(mii_model())),
        0x0100_0000_0000_0806 => ("NgWord", Some(ng_word1())),
        0x0100_0000_0000_0809 => ("SystemVersion", Some(system_version())),
        0x0100_0000_0000_080E => ("TimeZoneBinary", Some(time_zone_binary())),
        0x0100_0000_0000_0823 => ("NgWord2", Some(ng_word2())),
        0x0100_0000_0000_0800..=0x0100_0000_0000_0835
        | 0x0100_0000_0000_0859
        | 0x0100_0000_0000_085C => ("known system archive", None),
        _ if title_id >> 56 == 0x04 => {
            tracing::warn!("empty archive for Switch 2 system data {title_id:#018x}");
            ("Switch 2 system data", Some(MemDir::new()))
        }
        _ => {
            tracing::warn!("no system archive {title_id:#018x} to synthesize");
            return None;
        }
    };
    let Some(root) = root else {
        tracing::warn!("{name} {title_id:#018x} cannot be synthesized");
        return None;
    };
    tracing::info!("synthesizing system archive {name} ({title_id:#018x})");
    build_romfs(&root)
        .inspect_err(|e| tracing::error!("building {name}: {e}"))
        .ok()
}

fn mii_model() -> MemDir {
    let header = |magic: &[u8; 4]| {
        let mut data = vec![0u8; 0x10];
        data[..4].copy_from_slice(magic);
        data[4] = 1;
        data
    };
    let mut dir = MemDir::new();
    for name in [
        "NXTextureLowLinear.dat",
        "NXTextureLowSRGB.dat",
        "NXTextureMidLinear.dat",
        "NXTextureMidSRGB.dat",
    ] {
        dir = dir.with_bytes(name, header(b"NFTR"));
    }
    dir.with_bytes("ShapeHigh.dat", header(b"NFSR"))
        .with_bytes("ShapeMid.dat", header(b"NFSR"))
}

/// One word list, `^verybadword$` in UTF-16BE with a byte order mark. The C++ meant to write
/// 16 numbered lists too but its loop never runs, so only `common.txt` exists.
fn ng_word1() -> MemDir {
    let mut word = vec![0xfe, 0xff];
    word.extend("^verybadword$\n".encode_utf16().flat_map(u16::to_be_bytes));
    MemDir::new()
        .with_bytes("common.txt", word)
        .with_bytes("version.dat", vec![0, 0, 0, 0x20])
}

/// A gzip stream that deserialises to no bad words.
const AC_NX_DATA: [u8; 0x2c] = [
    0x1f, 0x8b, 0x08, 0x08, 0xd5, 0x2c, 0x09, 0x5c, 0x04, 0x00, 0x61, 0x63, 0x72, 0x61, 0x77, 0x00,
    0xed, 0xc1, 0x01, 0x0d, 0x00, 0x00, 0x00, 0xc2, 0x20, 0xfb, 0xa7, 0xb6, 0xc7, 0x07, 0x0c, 0x00,
    0x00, 0x00, 0xc8, 0x3b, 0x11, 0x00, 0x1c, 0xc7, 0x00, 0x10, 0x00, 0x00,
];

fn ng_word2() -> MemDir {
    let mut dir = MemDir::new();
    let lists = (0..0x10)
        .map(|i| i.to_string())
        .chain(std::iter::once("common".to_owned()));
    for list in lists {
        for kind in ["b1", "b2", "not_b"] {
            dir = dir.with_bytes(format!("ac_{list}_{kind}_nx"), AC_NX_DATA.to_vec());
        }
    }
    dir.with_bytes("version.dat", vec![0, 0, 0, 0x1a])
}

/// The 0x100-byte version `file` `set:sys` reads.
fn system_version() -> MemDir {
    let mut file = vec![0u8; 0x100];
    file[..3].copy_from_slice(&HOS_VERSION);
    file[4..6].copy_from_slice(&SDK_REVISION);
    for (offset, text) in [
        (0x08, PLATFORM_STRING),
        (0x28, VERSION_HASH),
        (0x68, DISPLAY_VERSION),
        (0x80, DISPLAY_TITLE),
    ] {
        file[offset..offset + text.len()].copy_from_slice(text.as_bytes());
    }
    MemDir::new().with_bytes("file", file)
}

/// The tz database's zones as `zoneinfo/<name>` TZif files, every name in `binaryList.txt`
/// and the release in `version.txt`.
fn time_zone_binary() -> MemDir {
    enum Node {
        File(&'static [u8]),
        Dir(BTreeMap<&'static str, Node>),
    }
    fn into_dir(entries: BTreeMap<&'static str, Node>) -> MemDir {
        entries
            .into_iter()
            .fold(MemDir::new(), |dir, (name, node)| match node {
                Node::File(data) => dir.with_bytes(name, data),
                Node::Dir(entries) => dir.with_dir(name, into_dir(entries)),
            })
    }

    let names: Vec<&'static str> = tzdb_data::TZ_NAMES
        .iter()
        .copied()
        .filter(|name| name.len() <= LOCATION_NAME_MAX)
        .collect();
    let mut zoneinfo = BTreeMap::new();
    for &name in &names {
        let Some(data) = tzdb_data::find_raw(name.as_bytes()) else {
            continue;
        };
        let mut parts = name.split('/').peekable();
        let mut dir = &mut zoneinfo;
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                dir.insert(part, Node::File(data));
                break;
            }
            let Node::Dir(entries) = dir
                .entry(part)
                .or_insert_with(|| Node::Dir(BTreeMap::new()))
            else {
                unreachable!("tz database zone names never nest under another zone");
            };
            dir = entries;
        }
    }
    let list: String = names.iter().map(|name| format!("{name}\n")).collect();
    MemDir::new()
        .with_bytes("binaryList.txt", list.into_bytes())
        .with_bytes(
            "version.txt",
            format!("{}\n", tzdb_data::VERSION).into_bytes(),
        )
        .with_dir("zoneinfo", into_dir(zoneinfo))
}

#[cfg(test)]
mod tests {
    use lichii_fs::romfs::open_romfs;
    use lichii_fs::{DirectoryExt, FileExt};

    use super::*;

    fn read(title_id: u64, path: &str) -> Vec<u8> {
        let root = open_romfs(&synthesize(title_id).unwrap()).unwrap();
        root.open_file_path(path)
            .unwrap()
            .unwrap()
            .read_all()
            .unwrap()
    }

    #[test]
    fn synthesizes_the_archives_the_cpp_builds() {
        let version = read(0x0100_0000_0000_0809, "file");
        assert_eq!(version.len(), 0x100);
        assert_eq!(version[..3], [23, 0, 0]);
        assert_eq!(&version[8..10], b"NX");
        assert_eq!(&version[0x68..0x6e], b"23.0.0");

        let utc = read(0x0100_0000_0000_080E, "zoneinfo/Etc/UTC");
        assert_eq!(&utc[..4], b"TZif");
        assert!(utc.ends_with(b"\nUTC0\n"));
        let amsterdam = read(0x0100_0000_0000_080E, "zoneinfo/Europe/Amsterdam");
        assert!(amsterdam.ends_with(b"\nCET-1CEST,M3.5.0,M10.5.0/3\n"));
        let list = String::from_utf8(read(0x0100_0000_0000_080E, "binaryList.txt")).unwrap();
        let names: Vec<&str> = list.lines().collect();
        assert_eq!(names.len(), tzdb_data::TZ_NAMES.len());
        assert!(names.contains(&"Europe/Amsterdam") && names.contains(&"UTC"));
        assert_eq!(
            read(0x0100_0000_0000_080E, "version.txt"),
            format!("{}\n", tzdb_data::VERSION).into_bytes()
        );

        assert_eq!(
            read(0x0100_0000_0000_0802, "ShapeMid.dat")[..5],
            *b"NFSR\x01"
        );
        let word = read(0x0100_0000_0000_0806, "common.txt");
        assert_eq!(word[..4], [0xfe, 0xff, 0x00, b'^']);
        assert_eq!(read(0x0100_0000_0000_0806, "version.dat"), [0, 0, 0, 0x20]);
        assert_eq!(read(0x0100_0000_0000_0823, "ac_15_not_b_nx"), AC_NX_DATA);
        assert_eq!(read(0x0100_0000_0000_0823, "ac_common_b1_nx"), AC_NX_DATA);

        let empty = open_romfs(&synthesize(0x0400_0000_0000_0849).unwrap()).unwrap();
        assert!(empty.entries().unwrap().is_empty());
        assert!(synthesize(0x0100_0000_0000_0811).is_none());
        assert!(synthesize(0x0100_0000_0000_0800).is_none());
        assert!(synthesize(0x0100_0000_0000_9999).is_none());
    }
}
