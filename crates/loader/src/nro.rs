//! NRO homebrew executables, with their optional asset section (icon, NACP, RomFS).

use lichii_fs::{FileExt, FileRef, slice};

use crate::error::{LoaderError, LoaderResult};
use crate::program::Module;

const HEADER_SIZE: usize = 0x80;
const MAX_NRO_SIZE: u32 = 1 << 30;

/// The asset section appended after an NRO's image.
#[derive(Clone, Default)]
pub struct NroAssets {
    pub icon: Option<FileRef>,
    pub nacp: Option<FileRef>,
    pub romfs: Option<FileRef>,
}

/// Reads `file` as an NRO: its single module and its assets.
pub fn load_nro(file: &FileRef, name: &str) -> LoaderResult<(Module, NroAssets)> {
    let header: [u8; HEADER_SIZE] = file
        .read_array(0)
        .map_err(|_| LoaderError::Corrupt("NRO", "shorter than its header"))?;
    if &header[0x10..0x14] != b"NRO0" {
        return Err(LoaderError::BadMagic("NRO"));
    }
    let word = |offset: usize| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let size = word(0x18);
    if size > MAX_NRO_SIZE || u64::from(size) > file.size() {
        return Err(LoaderError::Corrupt(
            "NRO",
            "image size past the end of the file",
        ));
    }
    let layout: [(u64, u64); 3] =
        std::array::from_fn(|i| (u64::from(word(0x20 + i * 8)), u64::from(word(0x24 + i * 8))));
    let mut end = 0;
    for &(offset, len) in &layout {
        if offset % 0x1000 != 0 || offset < end || offset + len > u64::from(size) {
            return Err(LoaderError::Corrupt(
                "NRO",
                "segments out of order or past the image",
            ));
        }
        end = offset + len;
    }
    let image = file.read_vec(0, size as usize)?;
    let build_id = header[0x40..0x60].try_into().unwrap();
    let module = Module::from_layout(
        name.to_owned(),
        image,
        layout,
        u64::from(word(0x38)),
        build_id,
    );
    Ok((module, read_assets(file, u64::from(size))?))
}

fn read_assets(file: &FileRef, base: u64) -> LoaderResult<NroAssets> {
    let Ok(header) = file.read_array::<0x38>(base) else {
        return Ok(NroAssets::default());
    };
    if &header[..4] != b"ASET" {
        return Ok(NroAssets::default());
    }
    let part = |index: usize| -> LoaderResult<Option<FileRef>> {
        let at = 8 + index * 16;
        let offset = u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
        let len = u64::from_le_bytes(header[at + 8..at + 16].try_into().unwrap());
        if len == 0 {
            return Ok(None);
        }
        let start = base
            .checked_add(offset)
            .filter(|s| s.checked_add(len).is_some_and(|e| e <= file.size()));
        let start = start.ok_or(LoaderError::Corrupt(
            "NRO",
            "asset past the end of the file",
        ))?;
        Ok(Some(slice(file, start, len)))
    };
    Ok(NroAssets {
        icon: part(0)?,
        nacp: part(1)?,
        romfs: part(2)?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use lichii_fs::MemFile;

    pub fn build_nro(assets: &[&[u8]; 3]) -> Vec<u8> {
        let mut image = vec![0u8; 0x3000];
        image[0x10..0x14].copy_from_slice(b"NRO0");
        image[0x18..0x1c].copy_from_slice(&0x3000u32.to_le_bytes());
        for (i, (offset, size)) in [(0u32, 0x1000u32), (0x1000, 0x1000), (0x2000, 0x800)]
            .iter()
            .enumerate()
        {
            image[0x20 + i * 8..0x24 + i * 8].copy_from_slice(&offset.to_le_bytes());
            image[0x24 + i * 8..0x28 + i * 8].copy_from_slice(&size.to_le_bytes());
        }
        image[0x38..0x3c].copy_from_slice(&0x1800u32.to_le_bytes());
        image[0x1000] = 0xaa;
        image[0x2000] = 0xbb;
        let mut asset_header = b"ASET".to_vec();
        asset_header.extend([0; 4]);
        let mut payload = Vec::new();
        for asset in assets {
            asset_header.extend((0x38 + payload.len() as u64).to_le_bytes());
            asset_header.extend((asset.len() as u64).to_le_bytes());
            payload.extend_from_slice(asset);
        }
        image.extend(asset_header);
        image.extend(payload);
        image
    }

    #[test]
    fn loads_module_and_assets() {
        let file = MemFile::from_bytes(build_nro(&[b"icon", b"", b"romfs!"]));
        let (module, assets) = load_nro(&file, "hello.nro").unwrap();
        assert_eq!(module.image.len(), 0x3000);
        assert_eq!(module.size, 0x4000);
        assert_eq!((module.image[0x1000], module.image[0x2000]), (0xaa, 0xbb));
        assert_eq!(module.segments[2].size, 0x2000);
        assert_eq!(assets.icon.unwrap().read_all().unwrap(), b"icon");
        assert!(assets.nacp.is_none());
        assert_eq!(assets.romfs.unwrap().read_all().unwrap(), b"romfs!");

        let mut bad = build_nro(&[b"", b"", b""]);
        bad[0x2c..0x30].copy_from_slice(&0x9000u32.to_le_bytes());
        assert!(load_nro(&MemFile::from_bytes(bad), "x").is_err());
        assert!(matches!(
            load_nro(&MemFile::from_bytes(vec![0; 0x80]), "x"),
            Err(LoaderError::BadMagic("NRO"))
        ));
    }
}
